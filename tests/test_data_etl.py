from pathlib import Path

import pytest

from frontier.data.etl import DatasetSpec, assign_splits, normalize_record, write_jsonl


def test_normalization_keeps_context_and_query_separate() -> None:
    instance = normalize_record(
        {"id": "1", "document": "ctx", "question": "q", "answer": "a"},
        spec=DatasetSpec("qa", "fixture/source"),
        index=0,
        split="train",
    )
    assert instance.context == "ctx"
    assert instance.query == "q"
    assert instance.meta["source_document_id"] == "1"


def test_splits_are_disjoint_by_source_document(tmp_path: Path) -> None:
    spec = DatasetSpec("qa", "fixture/source")
    instances = [
        normalize_record(
            {
                "id": str(index),
                "document_id": f"doc-{index // 2}",
                "context": "c",
                "query": "q",
                "gold": "a",
            },
            spec=spec,
            index=index,
            split="train",
        )
        for index in range(20)
    ]
    splits = assign_splits(instances, seed=2)
    ids = [
        {str(row.meta["source_document_id"]) for row in rows}
        for rows in splits.values()
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    )
    write_jsonl(splits["D_train"], tmp_path / "train.jsonl")
    assert (tmp_path / "train.jsonl").is_file()


@pytest.mark.integration
def test_real_huggingface_integration_is_opt_in() -> None:
    pytest.skip("network and benchmark credentials are intentionally opt-in")
