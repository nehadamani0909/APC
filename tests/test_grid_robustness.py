"""Grid-runner failure isolation and concurrency safety.

The pre-existing resilience test only raised from ``generate``, which was
inside the guarded block. Cost reconciliation, metric evaluation, and
``GridRow`` validation all sat outside it, so a single bad cell aborted the
whole run.
"""

import json
from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.corpus.build_grid import GridCell, GridRunner
from frontier.harness.ledger import Ledger, read_ledger, validate_ledger
from frontier.harness.models import GenerationResult, TargetLLM
from frontier.harness.tasks import GSM8KTask, Instance, JsonlTask, Task


class WorkingTarget:
    model = "local-default"
    model_revision = "test"

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        return GenerationResult(
            "fixture answer", len(prompt.split()), 2, 0.001, 0.0,
            self.model, self.model_revision,
        )


class MisreportingTarget:
    """Claims a USD figure that its own token usage does not support."""

    model = "gpt-4o-mini"
    model_revision = "test"

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        return GenerationResult(
            "answer", 1_000, 100, 0.001, 999.0, self.model, self.model_revision
        )


class UnpricedTarget(WorkingTarget):
    model = "model-not-in-price-table"


class ExplodingMetricTask(JsonlTask):
    def metric(self, pred: str, gold: str) -> float:
        raise RuntimeError("metric blew up")


def _cell(target: TargetLLM, task: Task | None = None) -> GridCell:
    return GridCell(
        Instance("p", "one two three four", "q", "a"),
        task or GSM8KTask(),
        TruncateTailCompressor(WhitespaceTokenizer()),
        0.5,
        target,
        0,
        0.0,
        0,
    )


def _runner(tmp_path: Path) -> GridRunner:
    return GridRunner(
        Ledger(tmp_path / "ledger.jsonl"),
        tmp_path / "completed.jsonl",
        tmp_path / "failed.jsonl",
    )


def test_cost_reconciliation_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.run([_cell(MisreportingTarget())]) == 0
    recorded = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert "does not reconcile" in recorded["error"]
    assert not (tmp_path / "ledger.jsonl").exists()


def test_unknown_price_row_is_recorded_not_raised(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.run([_cell(UnpricedTarget())]) == 0
    recorded = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert recorded["error_type"] == "ValueError"


def test_metric_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    task = ExplodingMetricTask("boom", "code")
    assert runner.run([_cell(WorkingTarget(), task)]) == 0
    recorded = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert "metric blew up" in recorded["error"]


def test_one_bad_cell_does_not_stop_the_rest(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    cells = [
        GridCell(
            Instance(f"p{index}", "one two three four", "q", "fixture answer"),
            GSM8KTask(),
            TruncateTailCompressor(WhitespaceTokenizer()),
            0.5,
            MisreportingTarget() if index == 2 else WorkingTarget(),
            0,
            0.0,
            0,
        )
        for index in range(6)
    ]
    assert runner.run(cells) == 5
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 5


def test_parallel_run_writes_a_parseable_ledger(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    cells = [
        GridCell(
            Instance(f"p{index}", "one two three four five", "q", "fixture answer"),
            GSM8KTask(),
            TruncateTailCompressor(WhitespaceTokenizer()),
            0.5,
            WorkingTarget(),
            0,
            0.0,
            0,
        )
        for index in range(40)
    ]
    assert runner.run_parallel(cells, workers=4) == 40
    frame = read_ledger(tmp_path / "ledger.jsonl")
    assert len(frame) == 40
    assert frame["prompt_id"].nunique() == 40
    assert validate_ledger(frame)


def test_completed_cells_are_skipped_on_resume(tmp_path: Path) -> None:
    cells = [_cell(WorkingTarget())]
    assert _runner(tmp_path).run(cells) == 1
    # A fresh runner rebuilds its completion set from the ledger on disk.
    assert _runner(tmp_path).run(cells) == 0
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 1
