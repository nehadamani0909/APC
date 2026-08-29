from pathlib import Path

from frontier.harness.tasks import (
    LONGBENCH_SUBSETS,
    GSM8KTask,
    Instance,
    default_tasks,
)


def test_registry_and_separate_context_query() -> None:
    tasks = default_tasks()
    assert len(tasks) == 11
    assert {task.name for task in tasks[1:7]} == set(LONGBENCH_SUBSETS)
    for task in tasks:
        instances = task.load("test", n=5)
        assert len(instances) == 5
        assert all(isinstance(instance, Instance) for instance in instances)
        assert all(instance.context and instance.query for instance in instances)


def test_jsonl_records_are_loaded(tmp_path: Path) -> None:
    source = tmp_path / "task.jsonl"
    source.write_text(
        '{"id":"x","context":"ctx","query":"q","gold":"a","split":"test"}\n',
        encoding="utf-8",
    )
    task = GSM8KTask(source)
    assert task.load("test", n=5)[0].id == "x"
