"""Run the 200-instance offline Gate 1 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.corpus.build_grid import GridCell, GridRunner, cell_count, dry_run
from frontier.harness.ledger import Ledger
from frontier.harness.models import GenerationResult
from frontier.harness.tasks import Family, Instance, JsonlTask

RATES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)


class PilotTarget:
    model = "local-default"
    model_revision = "offline-pilot-v1"

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
        words = len(context.split())
        fragile = "fragile" in context
        text = "wrong answer" if fragile and words < 12 else "fixture answer"
        output_tokens = 2 if text == "fixture answer" else 2 + max(0, 18 - words)
        return GenerationResult(
            text=text,
            T_in=len(prompt.split()),
            T_out=output_tokens,
            latency_s=0.001,
            usd=0.0,
            model=self.model,
            model_revision=self.model_revision,
        )


def _instances() -> list[tuple[JsonlTask, Instance]]:
    definitions: tuple[tuple[str, Family], ...] = (
        ("pilot_qa", "qa"),
        ("pilot_multidoc", "multidoc_qa"),
        ("pilot_summ", "summ"),
        ("pilot_conv", "conv"),
    )
    result: list[tuple[JsonlTask, Instance]] = []
    for family_index, (name, family) in enumerate(definitions):
        task = JsonlTask(name, family, metric_fn=lambda pred, gold: float(pred == gold))
        for index in range(50):
            marker = " fragile" if index % 2 == 0 else " stable"
            context = " ".join(
                [name, marker.strip()] + [f"token{j}" for j in range(20)]
            )
            result.append(
                (
                    task,
                    Instance(
                        id=f"pilot-{family_index}-{index}",
                        context=context,
                        query="What is the answer?",
                        gold="fixture answer",
                        meta={"source": "offline-pilot", "family": family},
                    ),
                )
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", type=Path, default=Path("data/pilot/ledger.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/gate1.md"))
    args = parser.parse_args()
    pairs = _instances()
    target = PilotTarget()
    backend = TruncateTailCompressor(WhitespaceTokenizer(), cache_dir="data/cache")
    instances = [instance for _, instance in pairs]
    if args.dry_run:
        print(dry_run(instances, RATES, [backend], [target], 5))
        return
    ledger = Ledger(args.ledger, phase="p3-pilot")
    runner = GridRunner(ledger, args.ledger.with_suffix(".completed.jsonl"))
    cells = (
        GridCell(
            task=task,
            instance=instance,
            backend=backend,
            requested_b=rate,
            target=target,
            sample_idx=sample,
            temperature=0.7,
            seed=sample,
        )
        for task, instance in pairs
        for rate in RATES
        for sample in range(5)
    )
    executed = runner.run(cells)
    total = cell_count(instances, RATES, [backend], [target], 5)
    print(f"executed cells: {executed}; total cells: {total}")
    from frontier.eval.gate1 import write_gate1_report

    write_gate1_report(args.ledger, args.report)


if __name__ == "__main__":
    main()
