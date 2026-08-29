from pathlib import Path

from frontier.harness.ledger import Ledger, read_ledger, validate_ledger
from frontier.harness.models import (
    APIBackend,
    ProviderResponse,
    run_task_once,
)
from frontier.harness.tasks import default_tasks


class FakeProvider:
    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        return ProviderResponse("fixture answer", len(prompt.split()), 2)


def test_one_generation_per_registered_task_reconciles(tmp_path: Path) -> None:
    target = APIBackend(
        FakeProvider(), model="gpt-4o-mini", model_revision="test-revision"
    )
    ledger = Ledger(tmp_path / "p1.jsonl", phase="p1")
    for task in default_tasks():
        run_task_once(task, task.load("test", n=1)[0], target, ledger)
    frame = read_ledger(ledger.path)
    assert len(frame) == 11
    assert validate_ledger(frame)
