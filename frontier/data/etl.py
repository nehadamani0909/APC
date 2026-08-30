"""Benchmark ETL: real dataset schemas normalised to the Instance contract.

Each dataset gets its own normaliser.  A shared field-alias table cannot work
here: GSM8K exposes ``question``/``answer``, HumanEval ``prompt``/``test``/
``entry_point``, MBPP ``text``/``test_list``, LongBench ``context``/``input``/
``answers``, and for the code families the gold value has to be the *test
harness* rather than a reference solution, because ``pass_at_1`` executes it.

Context and query are kept strictly separate throughout: ``x`` is compressed,
``q`` never is (APC-04 §4.3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from frontier.harness.tasks import Family, Instance
from frontier.predict.train import split_prompt_ids

Record = Mapping[str, Any]
Normalizer = Callable[[Sequence[Record]], list[Instance]]

# DECISION: GSM8K has no long context of its own, so the compressible text is
# the few-shot chain-of-thought block, matching how LLMLingua evaluates GSM8K.
# The exemplar pool is taken from the head of the split and excluded from the
# emitted instances, so no evaluated question appears in its own context.
GSM8K_SHOTS = 8


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: Family
    path: str
    normalize: Normalizer
    config: str | None = None
    split: str = "test"


def _document_id(text: str) -> str:
    """Stable id for datasets with no document key of their own.

    Several benchmarks reuse one document across many instances; hashing the
    context detects that reuse so the split assignment keeps it on one side
    (APC-04 §5.4).
    """

    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def _instance(
    *,
    prompt_id: str,
    context: str,
    query: str,
    gold: str,
    family: Family,
    source: str,
    source_document_id: str,
    extra: dict[str, object] | None = None,
) -> Instance:
    if not context.strip():
        raise ValueError(f"record {prompt_id} has an empty context")
    if not query.strip():
        raise ValueError(f"record {prompt_id} has an empty query")
    if not gold.strip():
        raise ValueError(f"record {prompt_id} has an empty gold value")
    meta: dict[str, object] = {
        "source": source,
        "source_document_id": source_document_id,
        "family": family,
    }
    meta.update(extra or {})
    return Instance(prompt_id, context, query, gold, meta)


def _text(record: Record, key: str, default: str = "") -> str:
    value = record.get(key, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def normalize_gsm8k(records: Sequence[Record]) -> list[Instance]:
    """GSM8K: few-shot CoT block as context, the question as query."""

    if len(records) <= GSM8K_SHOTS:
        raise ValueError("GSM8K needs more records than the few-shot pool size")
    pool = records[:GSM8K_SHOTS]
    exemplars = "\n\n".join(
        f"Question: {_text(row, 'question')}\nAnswer: {_text(row, 'answer')}"
        for row in pool
    )
    context_id = _document_id(exemplars)
    instances: list[Instance] = []
    for index, record in enumerate(records[GSM8K_SHOTS:]):
        answer = _text(record, "answer")
        instances.append(
            _instance(
                prompt_id=f"gsm8k-{index}",
                context=exemplars,
                query=_text(record, "question"),
                # The graded answer is the final numeric value after '####'.
                gold=answer.split("####")[-1].strip(),
                family="reason",
                source="openai/gsm8k",
                source_document_id=context_id,
                extra={"full_solution": answer},
            )
        )
    return instances


def normalize_humaneval(records: Sequence[Record]) -> list[Instance]:
    """HumanEval: signature+docstring compressed; gold is the test harness."""

    instances: list[Instance] = []
    for record in records:
        task_id = _text(record, "task_id")
        entry_point = _text(record, "entry_point")
        prompt = _text(record, "prompt")
        # pass_at_1 executes gold as the oracle, so it has to be runnable
        # tests plus the call that invokes them.
        harness = f"{_text(record, 'test')}\n\ncheck({entry_point})\n"
        instances.append(
            _instance(
                prompt_id=task_id.replace("/", "-") or f"humaneval-{len(instances)}",
                context=prompt,
                query=f"Complete the Python function `{entry_point}`.",
                gold=harness,
                family="code",
                source="openai_humaneval",
                source_document_id=task_id or _document_id(prompt),
                extra={"entry_point": entry_point},
            )
        )
    return instances


def normalize_mbpp(records: Sequence[Record]) -> list[Instance]:
    """MBPP: the natural-language spec is compressed; gold is the assert list."""

    instances: list[Instance] = []
    for record in records:
        task_id = _text(record, "task_id")
        tests = record.get("test_list") or []
        setup = _text(record, "test_setup_code")
        harness = "\n".join([setup, *(str(item) for item in tests)]).strip()
        description = _text(record, "text")
        instances.append(
            _instance(
                prompt_id=f"mbpp-{task_id}" if task_id else f"mbpp-{len(instances)}",
                context=description,
                query="Write a Python function that satisfies the description.",
                gold=harness,
                family="code",
                source="google-research-datasets/mbpp",
                source_document_id=task_id or _document_id(description),
            )
        )
    return instances


def normalize_meetingbank(records: Sequence[Record]) -> list[Instance]:
    """MeetingBank: transcript compressed, summary as gold."""

    instances: list[Instance] = []
    for index, record in enumerate(records):
        transcript = _text(record, "transcript") or _text(record, "source")
        summary = _text(record, "summary") or _text(record, "target")
        identifier = _text(record, "uid") or _text(record, "id") or str(index)
        instances.append(
            _instance(
                prompt_id=f"meetingbank-{identifier}",
                context=transcript,
                query="Summarise the meeting transcript.",
                gold=summary,
                family="summ",
                source="huuuyeah/meetingbank",
                source_document_id=_document_id(transcript),
            )
        )
    return instances


def normalize_sharegpt(records: Sequence[Record]) -> list[Instance]:
    """ShareGPT: prior turns compressed, final human turn is the query."""

    instances: list[Instance] = []
    for index, record in enumerate(records):
        turns = list(record.get("conversations") or [])
        if len(turns) < 3:
            continue
        history, last_query, last_reply = turns[:-2], turns[-2], turns[-1]
        if not history:
            continue
        context = "\n".join(
            f"{str(turn.get('from', '?')).upper()}: {turn.get('value', '')}"
            for turn in history
        )
        instances.append(
            _instance(
                prompt_id=f"sharegpt-{_text(record, 'id') or index}",
                context=context,
                query=str(last_query.get("value", "")),
                gold=str(last_reply.get("value", "")),
                family="conv",
                source="anon8231489123/ShareGPT_Vicuna_unfiltered",
                source_document_id=_document_id(context),
            )
        )
    return instances


def _longbench_normalizer(subset: str, family: Family) -> Normalizer:
    def normalize(records: Sequence[Record]) -> list[Instance]:
        instances: list[Instance] = []
        for index, record in enumerate(records):
            context = _text(record, "context")
            answers = record.get("answers") or []
            instances.append(
                _instance(
                    prompt_id=f"{subset}-{_text(record, '_id') or index}",
                    context=context,
                    query=_text(record, "input"),
                    gold=str(answers[0]) if answers else "",
                    family=family,
                    source=f"THUDM/LongBench:{subset}",
                    # LongBench exposes no document key and reuses passages
                    # across instances, so the context hash is the doc id.
                    source_document_id=_document_id(context),
                    # DECISION: the six LongBench sub-families do not all map
                    # onto the frozen Family literal, so the sub-family is
                    # recorded in meta for the E6b leave-one-family-out split.
                    extra={"subfamily": subset},
                )
            )
        return instances

    return normalize


# Real LongBench config names, grouped by the APC-05 sub-family they serve.
LONGBENCH_GROUPS: dict[str, tuple[Family, tuple[str, ...]]] = {
    "multidoc_qa": ("multidoc_qa", ("hotpotqa", "2wikimqa", "musique")),
    "single_doc_qa": ("qa", ("narrativeqa", "qasper", "multifieldqa_en")),
    "summarisation": ("summ", ("gov_report", "qmsum", "multi_news")),
    "few_shot": ("qa", ("trec", "triviaqa", "samsum")),
    "code": ("code", ("lcc", "repobench-p")),
    "synthetic": ("qa", ("passage_count", "passage_retrieval_en")),
}


def _build_specs() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {
        "gsm8k": DatasetSpec(
            "gsm8k", "reason", "openai/gsm8k", normalize_gsm8k, "main", "test"
        ),
        "meetingbank": DatasetSpec(
            "meetingbank",
            "summ",
            "huuuyeah/meetingbank",
            normalize_meetingbank,
            None,
            "test",
        ),
        "humaneval": DatasetSpec(
            "humaneval", "code", "openai_humaneval", normalize_humaneval, None, "test"
        ),
        "mbpp": DatasetSpec(
            "mbpp",
            "code",
            "google-research-datasets/mbpp",
            normalize_mbpp,
            "full",
            "test",
        ),
        "sharegpt": DatasetSpec(
            "sharegpt",
            "conv",
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            normalize_sharegpt,
            None,
            "train",
        ),
    }
    for group, (family, configs) in LONGBENCH_GROUPS.items():
        for config in configs:
            specs[config] = DatasetSpec(
                config,
                family,
                "THUDM/LongBench",
                _longbench_normalizer(config, family),
                config,
                "test",
            )
        # The sub-family name is an alias for its first config, so the
        # APC-05 family names stay usable on the command line.
        specs[group] = specs[configs[0]]
    return specs


DEFAULT_SPECS: dict[str, DatasetSpec] = _build_specs()


def load_huggingface(
    spec: DatasetSpec,
    *,
    split: str | None = None,
    n: int | None = None,
    cache_dir: Path | None = None,
) -> list[Instance]:
    """Load and normalise one dataset; the import stays optional until called."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'real' extra to use Hugging Face ETL") from exc
    dataset = load_dataset(
        spec.path,
        name=spec.config,
        split=split or spec.split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    records = [cast(Record, row) for row in dataset]
    instances = spec.normalize(records)
    return instances if n is None else instances[:n]


def assign_splits(
    instances: list[Instance], *, seed: int = 0
) -> dict[str, list[Instance]]:
    """Deterministic 60/10/10/20 partition without source-document leakage."""

    groups: dict[str, list[Instance]] = {}
    for instance in instances:
        document_id = str(instance.meta.get("source_document_id", instance.id))
        groups.setdefault(document_id, []).append(instance)
    split_ids = split_prompt_ids(sorted(groups), seed=seed)
    result: dict[str, list[Instance]] = {
        name: [] for name in ("D_train", "D_cal_a", "D_cal_b", "D_test")
    }
    for name, ids in split_ids.items():
        for document_id in sorted(ids):
            result[name].extend(groups[document_id])
    return result


def write_jsonl(
    instances: list[Instance], destination: Path, *, split: str | None = None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for instance in instances:
            meta = dict(instance.meta)
            resolved = split if split is not None else str(meta.get("split", ""))
            meta["split"] = resolved
            record = {
                "id": instance.id,
                "context": instance.context,
                "query": instance.query,
                "gold": instance.gold,
                # Top level as well as in meta: JsonlTask.load filters on the
                # top-level key, so writing it only into meta silently
                # returned every split for any requested split.
                "split": resolved,
                "meta": meta,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def fetch_family(
    name: str,
    *,
    output_dir: Path,
    split_dir: Path = Path("data/splits"),
    split: str | None = None,
    n: int | None = None,
    seed: int = 0,
) -> dict[str, Path]:
    if name not in DEFAULT_SPECS:
        raise ValueError(f"unknown dataset family: {name}")
    spec = DEFAULT_SPECS[name]
    instances = load_huggingface(spec, split=split, n=n)
    partitions = assign_splits(instances, seed=seed)
    combined: list[Instance] = []
    paths: dict[str, Path] = {}
    for partition, rows in partitions.items():
        destination = split_dir / f"{name}_{partition}.jsonl"
        write_jsonl(rows, destination, split=partition)
        paths[partition] = destination
        combined.extend(
            Instance(
                row.id,
                row.context,
                row.query,
                row.gold,
                {**row.meta, "split": partition},
            )
            for row in rows
        )
    write_jsonl(combined, output_dir / f"{name}.jsonl")
    paths["raw"] = output_dir / f"{name}.jsonl"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(DEFAULT_SPECS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--split")
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
