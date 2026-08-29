from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.corpus.build_grid import GridCell, GridRunner
from frontier.harness.ledger import Ledger
from frontier.harness.models import GenerationResult
from frontier.harness.tasks import GSM8KTask, Instance


class FailingTarget:
    model = "local-default"
    model_revision = "test"

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        raise RuntimeError("synthetic failure")


def test_grid_records_failure_and_continues(tmp_path: Path) -> None:
    runner = GridRunner(
        Ledger(tmp_path / "ledger.jsonl"),
        tmp_path / "completed.jsonl",
        tmp_path / "failed.jsonl",
    )
    cell = GridCell(
        Instance("p", "one two", "q", "a"),
        GSM8KTask(),
        TruncateTailCompressor(WhitespaceTokenizer()),
        0.5,
        FailingTarget(),
        0,
        0.0,
        0,
    )
    assert runner.run([cell]) == 0
    assert "synthetic failure" in (tmp_path / "failed.jsonl").read_text()
