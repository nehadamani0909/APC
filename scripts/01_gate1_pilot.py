"""Run the Gate 1 pilot on real instances, compressors, and target models.

APC-06 §E1: N prompts stratified over task families x 7 budgets x k samples.
The gate asks whether within-family Var[b*] exceeds the k-sample noise floor,
so ``--samples`` is the one dimension that should not be trimmed -- the noise
floor is estimated by splitting those samples in half.

Everything is resumable: re-running the same command skips completed cells.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from frontier.compress.base import Compressor, WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.compress.llmlingua2 import LLMLingua2Compressor
from frontier.corpus.build_grid import GridCell, GridRunner
from frontier.eval.gate1 import write_gate1_report
from frontier.harness.ledger import Ledger
from frontier.harness.models import APIBackend, TargetLLM
from frontier.harness.outputs import OutputStore
from frontier.harness.tasks import Instance, Task, default_tasks

RATES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)


def _task_by_name(name: str, instances_dir: Path) -> Task:
    for task in default_tasks(instances_dir):
        if task.name == name:
            return task
    raise ValueError(f"no task adapter registered for {name!r}")


def load_pilot_instances(
    families: Sequence[str],
    *,
    instances_dir: Path,
    per_family: int,
    split: str,
    max_context_words: int | None,
) -> list[tuple[Task, Instance]]:
    """Stratified sample: the first ``per_family`` usable instances each."""

    pairs: list[tuple[Task, Instance]] = []
    for family in families:
        task = _task_by_name(family, instances_dir)
        if split == "all":
            # Draw across all four splits in the 60/10/10/20 proportions, so
            # the resulting corpus can train, calibrate, AND report. Drawing
            # only from D_train yields a corpus whose D_test is empty, and the
            # evaluation then has nothing to report on.
            instances = []
            for name, share in (
                ("D_train", 0.6),
                ("D_cal_a", 0.1),
                ("D_cal_b", 0.1),
                ("D_test", 0.2),
            ):
                take = max(1, round(per_family * share))
                instances.extend(task.load(name)[:take])
        else:
            instances = task.load(split)
        if not instances:
            raise ValueError(
                f"no instances for {family!r} split {split!r} under {instances_dir}. "
                f"Run scripts/03_fetch_data.py {family} first."
            )
        kept = 0
        for instance in instances:
            if (
                max_context_words is not None
                and len(instance.context.split()) > max_context_words
            ):
                continue
            pairs.append((task, instance))
            kept += 1
            if kept >= per_family:
                break
        if kept < per_family:
            print(
                f"warning: {family} yielded {kept}/{per_family} instances "
                f"within --max-context-words",
                file=sys.stderr,
            )
    return pairs


def build_target(
    args: argparse.Namespace, task_for_stop: Task
) -> tuple[TargetLLM, Any]:
    """Return the target adapter and a tokenizer for realised-rate measurement."""

    if args.provider == "hf":
        from frontier.harness.providers import HFClient

        local_client = HFClient(
            model=args.model,
            revision=args.model_revision,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            # Task-specific: a few-shot model must stop before inventing the
            # next question. Kept on the client so the frozen Task protocol
            # is untouched.
            stop=getattr(task_for_stop, "STOP_SEQUENCES", ()),
        )
        backend = APIBackend(
            local_client,
            model=args.price_row,
            model_revision=f"{args.model}@{args.model_revision}",
        )
        return backend, local_client.load_tokenizer()
    if args.provider == "openai":
        from frontier.harness.providers import (
            CachedProvider,
            OpenAIClient,
            RetryingClient,
            SpendCap,
        )

        cap = SpendCap(args.spend_cap_usd)
        api_client = OpenAIClient(
            model=args.model,
            spend_cap=cap,
            estimated_cost_usd=args.estimated_cost_per_call_usd,
        )
        wrapped = CachedProvider(
            RetryingClient(api_client, concurrency=args.workers),
            model=args.model,
            revision=args.model_revision,
            cache_dir=args.cache_dir,
        )
        backend = APIBackend(
            wrapped, model=args.model, model_revision=args.model_revision
        )
        return backend, WhitespaceTokenizer()
    raise ValueError(f"unknown provider: {args.provider}")


def build_compressor(args: argparse.Namespace, tokenizer: Any) -> Compressor:
    if args.compressor == "llmlingua2":
        return LLMLingua2Compressor(
            tokenizer, cache_dir=args.cache_dir, device_map=args.device
        )
    if args.compressor == "truncate_tail":
        return TruncateTailCompressor(tokenizer, cache_dir=args.cache_dir)
    raise ValueError(f"unknown compressor: {args.compressor}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="gsm8k", help="comma-separated")
    parser.add_argument("--instances-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--split",
        default="all",
        help='"all" draws across the four splits in 60/10/10/20 proportions '
        "so the corpus can train, calibrate, and report",
    )
    parser.add_argument("--prompts-per-family", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5, help="k in APC-04 §3.1.1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-context-words", type=int)
    parser.add_argument("--provider", choices=("hf", "openai"), default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--price-row", default="local-default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--spend-cap-usd", type=float, default=0.0)
    parser.add_argument("--estimated-cost-per-call-usd", type=float, default=0.001)
    parser.add_argument(
        "--compressor", choices=("llmlingua2", "truncate_tail"), default="llmlingua2"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ledger", type=Path, default=Path("data/pilot/ledger.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/gate1.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    families = [name.strip() for name in args.families.split(",") if name.strip()]
    pairs = load_pilot_instances(
        families,
        instances_dir=args.instances_dir,
        per_family=args.prompts_per_family,
        split=args.split,
        max_context_words=args.max_context_words,
    )
    total = len(pairs) * len(RATES) * args.samples
    words = [len(instance.context.split()) for _, instance in pairs]
    print(
        f"families={families} prompts={len(pairs)} budgets={len(RATES)} "
        f"k={args.samples} -> {total} generations\n"
        f"context words: min={min(words)} median={sorted(words)[len(words) // 2]} "
        f"max={max(words)}"
    )
    if args.dry_run:
        print("dry run: no model loaded, nothing executed")
        return

    target, tokenizer = build_target(args, pairs[0][0])
    compressor = build_compressor(args, tokenizer)
    ledger = Ledger(args.ledger, phase="gate1-pilot")
    runner = GridRunner(
        ledger,
        args.ledger.with_suffix(".completed.jsonl"),
        args.ledger.with_suffix(".failed.jsonl"),
        # Keeps the raw generations so answer preservation (APC-04 §3.1.1)
        # can be computed without re-running the grid.
        OutputStore(args.ledger.with_suffix(".outputs.jsonl")),
    )

    cells = [
        GridCell(
            instance=instance,
            task=task,
            backend=compressor,
            requested_b=rate,
            target=target,
            sample_idx=sample,
            temperature=args.temperature,
            # A distinct seed per sample: k>1 at T>0 needs genuine variation,
            # and the ledger records which seed produced which sample.
            seed=sample,
        )
        for task, instance in pairs
        for rate in RATES
        for sample in range(args.samples)
    ]
    remaining = [cell for cell in cells if cell.key not in runner.completed]
    print(f"{len(cells) - len(remaining)} already complete; {len(remaining)} to run")

    started = time.perf_counter()
    done = 0
    for index in range(0, len(remaining), 25):
        chunk = remaining[index : index + 25]
        done += runner.run_parallel(chunk, workers=args.workers)
        elapsed = time.perf_counter() - started
        rate_per_s = done / elapsed if elapsed > 0 else 0.0
        eta = (len(remaining) - done) / rate_per_s if rate_per_s > 0 else float("nan")
        print(
            f"  {done}/{len(remaining)} cells | {elapsed / 60:.1f} min elapsed | "
            f"{rate_per_s * 60:.1f} cells/min | ETA {eta / 60:.1f} min",
            flush=True,
        )

    print(f"executed {done} cells; ledger at {args.ledger}")
    if args.ledger.exists():
        write_gate1_report(args.ledger, args.report)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
