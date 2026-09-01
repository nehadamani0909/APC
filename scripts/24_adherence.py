"""Measure requested-vs-realised compression rate at scale (contribution C4).

No target-LLM call is involved, so this runs over far more instances than the
quality grid can afford, and it can share a machine with a running grid: the
compressor is on CPU here precisely so it does not contend for the GPU.

Realised rate is measured with the *target's* tokenizer, not the compressor's.
Those two disagree, and it is the target's count that determines what the user
is billed -- so a rate reported in the compressor's own units is not the rate
the user pays for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from frontier.compress.llmlingua2 import LLMLingua2Compressor

RATES = (0.8, 0.65, 0.5, 0.4, 0.3, 0.2)
MODELS = {
    "llmlingua2_base":
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "llmlingua2_large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
}


class HFTokenizer:
    def __init__(self, name: str = "gpt2") -> None:
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name)
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/adherence_v1"))
    ap.add_argument("--families", nargs="+",
                    default=["multifieldqa_en", "2wikimqa", "gsm8k"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-tok", type=int, default=500)
    ap.add_argument("--max-tok", type=int, default=6000)
    ap.add_argument("--compressor", default="llmlingua2_base")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = HFTokenizer()
    backend = LLMLingua2Compressor(
        tok, model_name=MODELS[args.compressor], device_map=args.device
    )
    rows: list[dict] = []
    for family in args.families:
        path = Path("data/raw") / f"{family}.jsonl"
        if not path.exists():
            print(f"SKIP {family}: not fetched")
            continue
        records = []
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            n = len(tok.encode(rec.get("context", "")))
            if args.min_tok <= n <= args.max_tok:
                records.append((rec, n))
            if len(records) >= args.n:
                break
        print(f"{family}: {len(records)} instances", flush=True)
        for rec, n_ctx in records:
            for rate in RATES:
                try:
                    res = backend.compress(rec["context"], rec.get("query", ""), rate)
                except Exception as exc:  # one bad instance must not end the sweep
                    rows.append({
                        "family": family, "prompt_id": rec["id"], "requested_b": rate,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                rows.append({
                    "family": family, "prompt_id": rec["id"],
                    "requested_b": rate, "realised_r": res.realised_rate,
                    "signed_error": res.realised_rate - rate,
                    "context_tokens": n_ctx, "compress_ms": res.wall_ms,
                    "backend": backend.name,
                    "backend_version": backend.backend_version,
                    "model_version": backend.model_version,
                    "tokenizer": tok.name,
                })
        pd.DataFrame(rows).to_csv(args.out / "adherence.csv", index=False)

    frame = pd.DataFrame(rows)
    ok = frame[frame.get("realised_r").notna()] if "realised_r" in frame else frame
    frame.to_csv(args.out / "adherence.csv", index=False)
    print(f"\n{len(ok)} measurements, {len(frame) - len(ok)} errors")
    if len(ok):
        summary = ok.groupby(["backend", "requested_b"]).agg(
            realised_mean=("realised_r", "mean"),
            realised_sd=("realised_r", "std"),
            signed_error=("signed_error", "mean"),
            abs_error=("signed_error", lambda s: s.abs().mean()),
            n=("realised_r", "size"),
        ).round(4)
        summary.to_csv(args.out / "adherence_summary.csv")
        print(summary.to_string())


if __name__ == "__main__":
    main()
