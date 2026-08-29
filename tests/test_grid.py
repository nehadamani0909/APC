from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor
from frontier.corpus.build_grid import GridCell, GridRunner
from frontier.harness.ledger import Ledger, read_ledger, validate_ledger
from frontier.harness.models import GenerationResult
from frontier.harness.tasks import Instance, JsonlTask


class FakeTarget:
    model = "local-default"
    model_revision = "test"

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        return GenerationResult(
            "fixture answer", 10, 2, 0.001, 0.0, self.model, self.model_revision
        )


def test_grid_is_idempotent_and_resumable(tmp_path: Path) -> None:
    task = JsonlTask("test", "qa")
    instance = Instance("p", "one two three four", "q", "fixture answer")
    backend = TruncateTailCompressor(WhitespaceTokenizer())
    target = FakeTarget()
    ledger = Ledger(tmp_path / "ledger.jsonl", phase="test")
    runner = GridRunner(ledger, tmp_path / "completed.jsonl")
    cell = GridCell(instance, task, backend, 0.5, target, 0, 0.0, 0)
    assert runner.run([cell]) == 1
    assert runner.run([cell]) == 0
    assert len(read_ledger(ledger.path)) == 1
    assert validate_ledger(ledger.path)
