"""Requested-versus-realised rate over real prompts (E1c, contribution C4).

``r != b`` in general (audit D4): the compressor's rate is expressed in its
own tokenizer, while the quantity that drives cost is the target model's
token count.  This measures the gap that the H2 adherence head models and
inverts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from frontier.compress.base import Compressor, WhitespaceTokenizer
from frontier.compress.baselines import (
    RandomDropCompressor,
    TruncateHeadCompressor,
    TruncateTailCompressor,
)

RATES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)


def build_backend(
    name: str, tokenizer: Any, cache_dir: Path, device: str
) -> Compressor:
    if name == "llmlingua2":
        from frontier.compress.llmlingua2 import LLMLingua2Compressor

        return LLMLingua2Compressor(tokenizer, cache_dir=cache_dir, device_map=device)
    if name == "longllmlingua":
        from frontier.compress.longllmlingua import LongLLMLinguaCompressor

        return LongLLMLinguaCompressor(
            tokenizer, cache_dir=cache_dir, device_map=device
        )
    if name == "truncate_tail":
        return TruncateTailCompressor(tokenizer, cache_dir=cache_dir)
    if name == "truncate_head":
        return TruncateHeadCompressor(tokenizer, cache_dir=cache_dir)
    if name == "random_drop":
        return RandomDropCompressor(tokenizer, cache_dir=cache_dir)
    raise ValueError(f"unknown backend: {name}")


def load_tokenizer(model: str | None) -> Any:
    """The TARGET model's tokenizer, or a whitespace stand-in offline."""

    if model is None:
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="normalised instance JSONL")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--backend",
        default="llmlingua2",
        choices=(
            "llmlingua2",
            "longllmlingua",
            "truncate_tail",
            "truncate_head",
            "random_drop",
        ),
    )
    parser.add_argument(
        "--target-model",
        help="tokenizer that realised rate is measured on; omit for whitespace",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, help="write a markdown table here")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.n]
    if not rows:
        raise SystemExit(f"no instances in {args.input}")

    tokenizer = load_tokenizer(args.target_model)
    backend = build_backend(args.backend, tokenizer, args.cache_dir, args.device)

    by_rate: dict[float, list[float]] = {rate: [] for rate in RATES}
    print(f"backend={backend.name} version={backend.backend_version}")
    print(f"{'prompt':>16} {'requested':>10} {'realised':>9} {'error':>8}")
    for row in rows:
        for rate in RATES:
            result = backend.compress(str(row["context"]), str(row["query"]), rate)
            by_rate[rate].append(result.realised_rate)
            print(
                f"{str(row['id'])[-16:]:>16} {rate:10.2f} "
                f"{result.realised_rate:9.4f} {result.realised_rate - rate:+8.4f}"
            )

    lines = [
        "# E1c - rate adherence",
        "",
        f"Backend `{backend.name}` ({backend.backend_version}) over "
        f"{len(rows)} prompts; realised rate measured on "
        f"`{args.target_model or 'whitespace'}`.",
        "",
        "| Requested b | Mean realised r | s.d. | Mean r - b |",
        "|---:|---:|---:|---:|",
    ]
    print("\nsummary:")
    for rate in RATES:
        values = by_rate[rate]
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {rate:.2f} | {mean:.4f} | {sd:.4f} | {mean - rate:+.4f} |")
        print(
            f"  b={rate:.2f}  mean r={mean:.4f}  sd={sd:.4f}  "
            f"bias={mean - rate:+.4f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
