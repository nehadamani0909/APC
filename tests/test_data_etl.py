"""ETL normalisation against the real field shapes of each benchmark.

These use recorded record shapes rather than the network, so they stay
offline while still failing if a normaliser assumes fields the dataset does
not have -- which is exactly what the previous shared alias table did.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from frontier.data.etl import (
    DEFAULT_SPECS,
    LONGBENCH_GROUPS,
    assign_splits,
    normalize_gsm8k,
    normalize_humaneval,
    normalize_mbpp,
    normalize_meetingbank,
    normalize_sharegpt,
    write_jsonl,
)
from frontier.harness.metrics import pass_at_1
from frontier.harness.tasks import JsonlTask

GSM8K_ROWS: list[dict[str, Any]] = [
    {
        "question": f"Question number {index}?",
        "answer": f"Some reasoning for {index}.\n#### {index}",
    }
    for index in range(12)
]

HUMANEVAL_ROW: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": "def add_one(n):\n    '''Return n plus one.'''\n",
    "canonical_solution": "    return n + 1\n",
    "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    "entry_point": "add_one",
}

MBPP_ROW: dict[str, Any] = {
    "task_id": 601,
    "text": "Write a function to add one to a number.",
    "code": "def add_one(n):\n    return n + 1",
    "test_list": ["assert add_one(1) == 2"],
    "test_setup_code": "",
}

LONGBENCH_ROW: dict[str, Any] = {
    "_id": "abc123",
    "input": "Who wrote the report?",
    "context": "A long multi-document context passage.",
    "answers": ["Alice"],
    "length": 42,
}


def test_gsm8k_uses_few_shot_context_and_the_final_numeric_answer() -> None:
    instances = normalize_gsm8k(GSM8K_ROWS)
    # The eight pooled exemplars are held out of the emitted instances.
    assert len(instances) == len(GSM8K_ROWS) - 8
    first = instances[0]
    assert first.query == "Question number 8?"
    assert first.gold == "8"
    assert "Question number 0?" in first.context
    # The evaluated question must never appear inside its own context.
    assert first.query not in first.context
    assert first.meta["family"] == "reason"


def test_humaneval_gold_is_an_executable_harness() -> None:
    instance = normalize_humaneval([HUMANEVAL_ROW])[0]
    assert instance.id == "HumanEval-0"
    assert instance.context.startswith("def add_one")
    assert "add_one" in instance.query
    # gold must run: pass_at_1 executes candidate + gold together.
    assert pass_at_1("def add_one(n):\n    return n + 1\n", instance.gold) == 1.0
    assert pass_at_1("def add_one(n):\n    return n + 99\n", instance.gold) == 0.0


def test_mbpp_gold_is_the_assert_list() -> None:
    instance = normalize_mbpp([MBPP_ROW])[0]
    assert instance.id == "mbpp-601"
    assert instance.context == MBPP_ROW["text"]
    assert pass_at_1("def add_one(n):\n    return n + 1\n", instance.gold) == 1.0


def test_longbench_keeps_context_and_input_separate() -> None:
    subset = LONGBENCH_GROUPS["multidoc_qa"][1][0]
    spec = DEFAULT_SPECS[subset]
    instance = spec.normalize([LONGBENCH_ROW])[0]
    assert instance.context == LONGBENCH_ROW["context"]
    assert instance.query == LONGBENCH_ROW["input"]
    assert instance.gold == "Alice"
    assert instance.meta["subfamily"] == subset


def test_longbench_configs_are_real_names_and_families_are_distinct() -> None:
    assert DEFAULT_SPECS["hotpotqa"].config == "hotpotqa"
    assert DEFAULT_SPECS["gov_report"].family == "summ"
    assert DEFAULT_SPECS["lcc"].family == "code"
    # The six sub-families must not all collapse onto one family label.
    families = {family for family, _ in LONGBENCH_GROUPS.values()}
    assert len(families) > 1


def test_meetingbank_and_sharegpt_normalise() -> None:
    meeting = normalize_meetingbank(
        [{"uid": "m1", "transcript": "a long transcript", "summary": "short"}]
    )[0]
    assert meeting.gold == "short"
    assert meeting.meta["family"] == "summ"

    conversation = normalize_sharegpt(
        [
            {
                "id": "c1",
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hi there"},
                    {"from": "human", "value": "what next?"},
                    {"from": "gpt", "value": "this next"},
                ],
            }
        ]
    )[0]
    assert conversation.query == "what next?"
    assert conversation.gold == "this next"
    assert "hello" in conversation.context
    assert "this next" not in conversation.context


def test_empty_fields_are_rejected_rather_than_silently_normalised() -> None:
    with pytest.raises(ValueError, match="empty gold"):
        normalize_meetingbank([{"uid": "x", "transcript": "text", "summary": ""}])


def test_splits_are_disjoint_by_source_document() -> None:
    instances = normalize_gsm8k(GSM8K_ROWS) + normalize_mbpp(
        [dict(MBPP_ROW, task_id=index, text=f"Task {index}") for index in range(20)]
    )
    splits = assign_splits(instances, seed=2)
    documents = [
        {str(row.meta["source_document_id"]) for row in rows}
        for rows in splits.values()
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(documents)
        for right in documents[index + 1 :]
    )


def test_written_split_is_readable_by_the_task_loader(tmp_path: Path) -> None:
    instances = normalize_gsm8k(GSM8K_ROWS)
    path = tmp_path / "gsm8k.jsonl"
    write_jsonl(instances, path, split="D_test")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    # Top-level, because JsonlTask.load filters on the top-level key.
    assert record["split"] == "D_test"
    assert record["meta"]["split"] == "D_test"

    task = JsonlTask("gsm8k", "reason", source=path)
    assert len(task.load("D_test")) == len(instances)
    # The filter must actually filter; before this fix every split matched.
    assert task.load("D_train") == []


@pytest.mark.integration
@pytest.mark.parametrize("family", ["gsm8k", "hotpotqa"])
def test_real_download_normalises_and_splits_cleanly(
    family: str, tmp_path: Path
) -> None:
    """Opt-in: downloads the real dataset and checks the split invariants."""

    pytest.importorskip("datasets")
    from frontier.data.etl import fetch_family

    paths = fetch_family(
        family, output_dir=tmp_path / "raw", split_dir=tmp_path / "splits", n=64
    )
    seen: dict[str, str] = {}
    for partition in ("D_train", "D_cal_a", "D_cal_b", "D_test"):
        rows = [
            json.loads(line)
            for line in paths[partition].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            assert row["context"] and row["query"] and row["gold"]
            assert row["split"] == partition
            document = str(row["meta"]["source_document_id"])
            # No prompt id and no source document may cross a split boundary.
            assert seen.setdefault(row["id"], partition) == partition
            assert seen.setdefault(f"doc:{document}", partition) == partition
    assert any(paths[name].stat().st_size > 0 for name in ("D_train", "D_test"))
