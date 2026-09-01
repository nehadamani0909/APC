"""Run the real Gate-1 grid: LLMLingua-2 compressor, local target model.

This is the run the project was built for and had never executed. The pilot
corpus in ``data/corpus/pilot`` used ``truncate_tail`` as the backend and a
0.5B target that solved only 26% of its prompts uncompressed, which made its
Gate 1 verdict uninterpretable by the gate's own validity checks.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import zip_longest
from pathlib import Path

from frontier.compress.base import CachedCompressor, TargetTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.compress.llmlingua2 import LLMLingua2Compressor
from frontier.corpus.build_grid import GridCell, GridRunner
from frontier.harness.ledger import Ledger
from frontier.harness.metrics import verify_sandbox
from frontier.harness.models import APIBackend
from frontier.harness.ollama_client import OllamaClient
from frontier.harness.outputs import OutputStore
from frontier.harness.prices import ModelPrice, register_price
from frontier.harness.tasks import (
    GSM8KTask,
    Instance,
    JsonlTask,
    LongBenchTask,
)

RATES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)


class HFTokenizer:
    """Target-model tokenizer, so realised rate is measured in the target's
    units rather than the compressor's (APC-04: the two disagree)."""

    def __init__(self, name: str = "meta-llama/Llama-3.2-3B-Instruct") -> None:
        from transformers import AutoTokenizer

        try:
            self._tok = AutoTokenizer.from_pretrained(name)
        except Exception:
            # Gated or unavailable: fall back to a public BPE with similar
            # granularity. Recorded in the run manifest so it is not silent.
            self._tok = AutoTokenizer.from_pretrained("gpt2")
            self.fallback = True
        else:
            self.fallback = False

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))


def load_instances(
    path: Path, task_name: str, family: str, *, min_tok: int, max_tok: int,
    n: int, tokenizer: TargetTokenizer, seed: int = 0,
) -> list[Instance]:
    """Load instances whose context fits the run's length window.

    Compression is only meaningful on contexts long enough to have redundancy,
    and the target's context window bounds the other end. Instances outside
    the window are dropped here rather than failing mid-grid.
    """
    rows: list[Instance] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            context = record.get("context", "")
            length = len(tokenizer.encode(context))
            if min_tok <= length <= max_tok:
                rows.append(
                    Instance(
                        id=str(record["id"]),
                        context=context,
                        query=str(record.get("query", "")),
                        gold=record.get("gold", ""),
                        meta={**record.get("meta", {}), "context_tokens": length},
                    )
                )
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/real_v1"))
    parser.add_argument("--n-per-family", type=int, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-tok", type=int, default=500)
    parser.add_argument("--max-tok", type=int, default=4500)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--compressor", default="llmlingua2_base")
    parser.add_argument("--with-truncate-control", action="store_true")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    verify_sandbox()  # never let a broken sandbox silently score code as 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    register_price(args.model, ModelPrice(0.0, 0.0), overwrite=True)

    tokenizer = HFTokenizer()
    # Only families whose *natural* context length fits the target's window.
    # hotpotqa (median 14k tok) and qmsum (13.5k) are excluded deliberately:
    # capping them to fit would cut the supporting document, forcing
    # rho(x, 1) = 0 and reproducing exactly the degenerate-label failure that
    # made the original pilot's Gate 1 verdict uninterpretable.
    # gsm8k is short but is LLMLingua's own flagship setting -- compressing
    # few-shot exemplars -- and contributes binary EM quality against the
    # graded F1 of the QA families.
    families = [
        ("gsm8k", "reason"),
        ("multifieldqa_en", "qa"),
        ("2wikimqa", "multidoc_qa"),
    ]
    pairs: list[tuple[JsonlTask, Instance]] = []
    for name, family in families:
        path = Path("data/raw") / f"{name}.jsonl"
        if not path.exists():
            print(f"SKIP {name}: not fetched")
            continue
        task = (
            GSM8KTask(source=path)
            if name == "gsm8k"
            else LongBenchTask(name, source=path, family=family)
        )
        rows = load_instances(
            path, name, family, min_tok=args.min_tok, max_tok=args.max_tok,
            n=args.n_per_family, tokenizer=tokenizer,
        )
        print(f"{name}: {len(rows)} instances in [{args.min_tok},{args.max_tok}] tok")
        pairs.extend((task, row) for row in rows)

    if not pairs:
        raise SystemExit("no instances loaded; fetch data first")

    model_map = {
        "llmlingua2_base":
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        "llmlingua2_large":
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    }
    backends = [
        CachedCompressor(
            LLMLingua2Compressor(
                tokenizer, model_name=model_map[args.compressor], device_map=args.device
            ),
            cache_dir="data/cache",
        )
    ]
    if args.with_truncate_control:
        backends.append(TruncateTailCompressor(tokenizer))

    # Output budget per family. A single global num_predict either truncates
    # GSM8K's chain of thought (destroying the answer) or lets the QA families
    # ramble for hundreds of tokens they never need, which is most of the
    # grid's wall-clock.
    predict_by_family = {"reason": 256, "qa": 64, "multidoc_qa": 64, "summ": 160}

    def target_for(family: str) -> APIBackend:
        return APIBackend(
            client=OllamaClient(
                model=args.model, num_ctx=args.num_ctx,
                num_predict=predict_by_family.get(family, 128),
            ),
            model=args.model,
            model_revision=args.model,
        )

    targets = {family: target_for(family) for _, family in families}

    # Round-robin the families. GridRunner.run_parallel splits the pending
    # list into contiguous chunks, so a family-sequential order would mean an
    # interrupted run had covered some families and not others. Interleaving
    # makes any prefix of the grid a balanced sample. Each instance's seven
    # budgets stay adjacent because Gate 1 needs the whole curve or none of it.
    by_family: dict[str, list] = {}
    for task, instance in pairs:
        by_family.setdefault(task.family, []).append((task, instance))
    interleaved: list = []
    for row in zip_longest(*by_family.values()):
        interleaved.extend(item for item in row if item is not None)

    cells = [
        GridCell(
            instance=instance, task=task, backend=backend, requested_b=rate,
            target=targets[task.family], sample_idx=k, temperature=0.7, seed=k,
        )
        for task, instance in interleaved
        for backend in backends
        for rate in RATES
        for k in range(args.samples)
    ]
    print(f"grid: {len(cells)} cells "
          f"({len(pairs)} instances x {len(RATES)} rates x "
          f"{len(backends)} backends x {args.samples} samples)")

    runner = GridRunner(
        ledger=Ledger(args.out / "ledger.jsonl"),
        completion_index=args.out / "completed.jsonl",
        failure_index=args.out / "failures.jsonl",
        output_store=OutputStore(args.out / "outputs"),
    )
    (args.out / "manifest.json").write_text(json.dumps({
        "model": args.model, "num_ctx": args.num_ctx,
        "compressor": args.compressor, "device": args.device, "rates": RATES,
        "samples": args.samples, "n_per_family": args.n_per_family,
        "token_window": [args.min_tok, args.max_tok],
        "tokenizer_fallback": getattr(tokenizer, "fallback", None),
        "cells": len(cells), "instances": len(pairs),
    }, indent=2), encoding="utf-8")
    executed = runner.run_parallel(cells, workers=args.workers)
    print(f"executed {executed} cells -> {args.out/'ledger.jsonl'}")


if __name__ == "__main__":
    main()
