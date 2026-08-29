"""Task and instance contracts for the P1 harness.

The adapters accept benchmark records in JSONL form.  When no source is
configured, they provide small deterministic fixtures so harness tests remain
offline and reproducible; no benchmark data is silently downloaded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from frontier.harness.metrics import (
    exact_match,
    pass_at_1,
    rouge_l,
    token_f1,
)

Family = Literal["qa", "multidoc_qa", "summ", "reason", "code", "conv"]


@dataclass(frozen=True)
class Instance:
    """One task example, keeping compressible context separate from query."""

    id: str
    context: str
    query: str
    gold: str
    meta: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    name: str
    family: Family

    def load(self, split: str, n: int | None = None) -> list[Instance]: ...

    def build_prompt(self, ctx: str, query: str) -> str: ...

    def metric(self, pred: str, gold: str) -> float: ...

    def parse(self, raw: str) -> str: ...


def _fixture(name: str, family: Family, split: str) -> list[Instance]:
    return [
        Instance(
            id=f"{name}-{split}-{index}",
            context=f"Fixture context {index} for {name}.",
            query="What is the answer?",
            gold="fixture answer",
            meta={"source": "offline-fixture", "split": split},
        )
        for index in range(5)
    ]


def _read_jsonl(path: Path, name: str, split: str) -> list[Instance]:
    instances: list[Instance] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            record_split = str(record.get("split", split))
            if record_split != split:
                continue
            try:
                instance = Instance(
                    id=str(record["id"]),
                    context=str(record["context"]),
                    query=str(record["query"]),
                    gold=str(record["gold"]),
                    meta=dict(record.get("meta", {})),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid instance at {path}:{line_number}") from exc
            instances.append(instance)
    return instances


class JsonlTask:
    """Common adapter for normalized benchmark JSONL records."""

    def __init__(
        self,
        name: str,
        family: Family,
        *,
        source: str | Path | None = None,
        metric_fn: Callable[[str, str], float] = exact_match,
    ) -> None:
        self.name = name
        self.family = family
        self.source = Path(source) if source is not None else None
        self._metric_fn = metric_fn

    def load(self, split: str, n: int | None = None) -> list[Instance]:
        records = (
            _read_jsonl(self.source, self.name, split)
            if self.source is not None
            else _fixture(self.name, self.family, split)
        )
        return records if n is None else records[:n]

    def build_prompt(self, ctx: str, query: str) -> str:
        return f"Context:\n{ctx}\n\nQuestion:\n{query}\n\nAnswer:"

    def metric(self, pred: str, gold: str) -> float:
        return float(self._metric_fn(pred, gold))

    def parse(self, raw: str) -> str:
        return raw.strip()


class GSM8KTask(JsonlTask):
    def __init__(self, source: str | Path | None = None) -> None:
        super().__init__("gsm8k", "reason", source=source, metric_fn=exact_match)


class LongBenchTask(JsonlTask):
    def __init__(self, subset: str, source: str | Path | None = None) -> None:
        super().__init__(subset, "multidoc_qa", source=source, metric_fn=token_f1)


class MeetingBankTask(JsonlTask):
    def __init__(self, source: str | Path | None = None) -> None:
        super().__init__("meetingbank", "summ", source=source, metric_fn=rouge_l)


class HumanEvalTask(JsonlTask):
    def __init__(self, source: str | Path | None = None) -> None:
        super().__init__("humaneval", "code", source=source, metric_fn=pass_at_1)


class MBPPTask(JsonlTask):
    def __init__(self, source: str | Path | None = None) -> None:
        super().__init__("mbpp", "code", source=source, metric_fn=pass_at_1)


class ShareGPTTask(JsonlTask):
    def __init__(self, source: str | Path | None = None) -> None:
        super().__init__("sharegpt", "conv", source=source, metric_fn=token_f1)


LONGBENCH_SUBSETS = (
    "multidoc_qa",
    "single_doc_qa",
    "summarisation",
    "few_shot",
    "code",
    "synthetic",
)


def default_tasks(source_dir: str | Path | None = None) -> list[Task]:
    """Return tasks, using normalized JSONL sources when a directory is set."""
    root = Path(source_dir) if source_dir is not None else None

    def source(name: str) -> Path | None:
        candidate = root / f"{name}.jsonl" if root is not None else None
        return candidate if candidate is not None and candidate.exists() else None

    return [
        GSM8KTask(source("gsm8k")),
        *(LongBenchTask(subset, source=source(subset)) for subset in LONGBENCH_SUBSETS),
        MeetingBankTask(source("meetingbank")),
        HumanEvalTask(source("humaneval")),
        MBPPTask(source("mbpp")),
        ShareGPTTask(source("sharegpt")),
    ]
