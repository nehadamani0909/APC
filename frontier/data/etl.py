"""Optional Hugging Face ETL for the benchmark families in APC-05."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from frontier.harness.tasks import Family, Instance
from frontier.predict.train import split_prompt_ids


@dataclass(frozen=True)
class DatasetSpec:
    family: Family
    path: str
    config: str | None = None
    source_document_field: str = "id"


DEFAULT_SPECS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec("reason", "openai/gsm8k", "main"),
    "meetingbank": DatasetSpec("summ", "huuuyeah/meetingbank"),
    "humaneval": DatasetSpec("code", "openai_humaneval"),
    "mbpp": DatasetSpec("code", "google-research-datasets/mbpp"),
    "sharegpt": DatasetSpec("conv", "anon8231489123/ShareGPT_Vicuna_unfiltered"),
}
for _subset in (
    "multidoc_qa",
    "single_doc_qa",
    "summarisation",
    "few_shot",
    "code",
    "synthetic",
):
    DEFAULT_SPECS[_subset] = DatasetSpec(
        "multidoc_qa", "THUDM/LongBench", _subset
    )

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "context": ("context", "document", "prompt", "text", "code", "completion"),
    "query": ("query", "question", "instruction", "problem", "input"),
    "gold": ("gold", "answer", "answers", "target", "response", "canonical_solution"),
    "id": ("id", "idx", "question_id", "task_id"),
}


def _first(record: dict[str, Any], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            if isinstance(value, list):
                return str(value[0]) if value else default
            return str(value)
    return default


def normalize_record(
    record: dict[str, Any], *, spec: DatasetSpec, index: int, split: str
) -> Instance:
    prompt_id = _first(record, FIELD_ALIASES["id"], f"{spec.path}-{index}")
    context = _first(record, FIELD_ALIASES["context"])
    query = _first(record, FIELD_ALIASES["query"])
    gold = _first(record, FIELD_ALIASES["gold"])
    if not context or not query or not gold:
        raise ValueError(f"record {prompt_id} lacks context, query, or gold")
    source_document_id = _first(
        record,
        (spec.source_document_field, "source_document_id", "document_id"),
        prompt_id,
    )
    meta: dict[str, object] = {
        "source": spec.path,
        "source_document_id": source_document_id,
        "split": split,
        "family": spec.family,
    }
    return Instance(prompt_id, context, query, gold, meta)


def load_huggingface(
    spec: DatasetSpec,
    *,
    split: str = "train",
    n: int | None = None,
    cache_dir: Path | None = None,
) -> list[Instance]:
    """Load and normalize one HF dataset; import is optional until this is called."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'real' extra to use Hugging Face ETL") from exc
    dataset = load_dataset(
        spec.path,
        name=spec.config,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    records: list[Instance] = []
    for index, row in enumerate(dataset):
        records.append(
            normalize_record(
                cast(dict[str, Any], row), spec=spec, index=index, split=split
            )
        )
        if n is not None and len(records) >= n:
            break
    return records


def assign_splits(
    instances: list[Instance], *, seed: int = 0
) -> dict[str, list[Instance]]:
    """Assign deterministic 60/10/10/20 splits without source-document leakage."""
    groups: dict[str, list[Instance]] = {}
    for instance in instances:
        document_id = str(instance.meta.get("source_document_id", instance.id))
        groups.setdefault(document_id, []).append(instance)
    split_ids = split_prompt_ids(list(groups), seed=seed)
    result: dict[str, list[Instance]] = {
        name: [] for name in ("D_train", "D_cal_a", "D_cal_b", "D_test")
    }
    for name, ids in split_ids.items():
        for document_id in ids:
            result[name].extend(groups[document_id])
    return result


def write_jsonl(instances: list[Instance], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for instance in instances:
            record = {
                "id": instance.id,
                "context": instance.context,
                "query": instance.query,
                "gold": instance.gold,
                "meta": instance.meta,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def fetch_family(
    name: str,
    *,
    output_dir: Path,
    split_dir: Path = Path("data/splits"),
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
) -> dict[str, Path]:
    if name not in DEFAULT_SPECS:
        raise ValueError(f"unknown dataset family: {name}")
    instances = load_huggingface(DEFAULT_SPECS[name], split=split, n=n)
    partitions = assign_splits(instances, seed=seed)
    write_jsonl(instances, output_dir / f"{name}.jsonl")
    paths: dict[str, Path] = {}
    for partition, rows in partitions.items():
        destination = split_dir / f"{name}_{partition}.jsonl"
        write_jsonl(rows, destination)
        paths[partition] = destination
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(DEFAULT_SPECS))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for partition, path in fetch_family(
        args.family,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        split=args.split,
        n=args.n,
        seed=args.seed,
    ).items():
        print(f"{partition}: {path}")


if __name__ == "__main__":
    main()
