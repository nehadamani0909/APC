"""Append-only JSONL cost ledger and its validation utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import pandas as pd
import structlog

from frontier.harness.prices import PRICE_TABLE_VERSION, cost_from_tokens

LOGGER = structlog.get_logger(__name__)
DEFAULT_LEDGER_PATH = Path("data") / "ledger.jsonl"
LEDGER_FIELDS = (
    "run_id",
    "ts",
    "phase",
    "prompt_id",
    "task",
    "family",
    "backend",
    "requested_b",
    "realised_r",
    "target_model",
    "sample_idx",
    "temperature",
    "seed",
    "quality",
    "T_in",
    "T_out",
    "latency_ms",
    "compress_ms",
    "usd_in",
    "usd_out",
    "usd_total",
    "gpu_seconds",
    "raw_output_hash",
    "code_version",
    "price_table_version",
)
KEY_FIELDS = ("prompt_id", "backend", "requested_b", "target_model", "sample_idx")


@dataclass(frozen=True)
class GridRow:
    """Atomic corpus/ledger row defined by APC-04 §8."""

    prompt_id: str
    task: str
    family: str
    backend: str
    requested_b: float
    realised_r: float
    target_model: str
    sample_idx: int
    temperature: float
    seed: int
    quality: float
    T_in: int
    T_out: int
    latency_ms: float
    compress_ms: float
    usd_in: float
    usd_out: float
    usd_total: float
    gpu_seconds: float
    raw_output_hash: str
    code_version: str
    price_table_version: str

    def __post_init__(self) -> None:
        if not self.price_table_version:
            raise ValueError("price_table_version is required")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if self.T_in < 0 or self.T_out < 0 or self.sample_idx < 0:
            raise ValueError("token counts and sample_idx cannot be negative")


class Ledger:
    """Writer for an append-only JSONL ledger."""

    def __init__(
        self, path: str | Path = DEFAULT_LEDGER_PATH, *, phase: str = "unknown"
    ) -> None:
        self.path = Path(path)
        self.run_id = str(uuid4())
        self.phase = phase
        # POSIX makes short O_APPEND writes effectively atomic; Windows does
        # not, so parallel grid workers need an explicit lock or the ledger
        # interleaves and becomes unparseable.
        self._lock = Lock()

    def append(self, row: GridRow) -> None:
        """Append exactly one JSON object, creating parent directories if needed."""

        record = asdict(row)
        # ``ts`` is mandatory ledger metadata in APC-04 §4.4, but is not part
        # of the frozen GridRow data contract in §8.
        record["run_id"] = self.run_id
        record["phase"] = self.phase
        record["ts"] = datetime.now(UTC).isoformat()
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def append_many(self, rows: Iterable[GridRow]) -> None:
        for row in rows:
            self.append(row)


def read_ledger(path: str | Path = DEFAULT_LEDGER_PATH) -> pd.DataFrame:
    """Read a JSONL ledger into a DataFrame with the frozen schema."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return pd.DataFrame(columns=LEDGER_FIELDS)
    records: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Ledger line {line_number} is not an object")
            records.append(record)
    return pd.DataFrame.from_records(records, columns=LEDGER_FIELDS)


def validate_ledger(
    ledger: str | Path | pd.DataFrame = DEFAULT_LEDGER_PATH,
    *,
    cost_tolerance: float = 0.01,
) -> bool:
    """Validate schema, completeness, uniqueness, and token-price consistency.

    Raises ``ValueError`` with all discovered problems; returns ``True`` when
    validation succeeds so callers can use it directly in checks and scripts.
    """

    frame = ledger if isinstance(ledger, pd.DataFrame) else read_ledger(ledger)
    problems: list[str] = []
    missing = [field for field in LEDGER_FIELDS if field not in frame.columns]
    if missing:
        problems.append(f"missing fields: {missing}")
        raise ValueError("Ledger validation failed: " + "; ".join(problems))
    frame = frame.loc[:, LEDGER_FIELDS]
    if frame.empty:
        problems.append("ledger is empty")
    if frame.isna().any().any():
        problems.append("NaN values present")
    if frame.duplicated(subset=list(KEY_FIELDS)).any():
        problems.append("duplicate ledger keys present")

    for index, row in frame.iterrows():
        try:
            expected_in, expected_out, expected_total = cost_from_tokens(
                str(row["target_model"]),
                int(row["T_in"]),
                int(row["T_out"]),
                str(row["price_table_version"]),
            )
            values = {
                "usd_in": (float(row["usd_in"]), expected_in),
                "usd_out": (float(row["usd_out"]), expected_out),
                "usd_total": (float(row["usd_total"]), expected_total),
            }
            for field, (actual, expected) in values.items():
                scale = max(1e-9, abs(expected))
                if not math.isclose(actual, expected, abs_tol=cost_tolerance * scale):
                    problems.append(
                        f"row {index}: {field}={actual} inconsistent with {expected}"
                    )
        except (TypeError, ValueError, OverflowError) as exc:
            problems.append(f"row {index}: invalid cost fields: {exc}")

    if problems:
        raise ValueError("Ledger validation failed: " + "; ".join(problems))
    return True


def _synthetic_rows(count: int = 100) -> list[GridRow]:
    rows: list[GridRow] = []
    for index in range(count):
        model = "local-default"
        tokens_in = 100 + index
        tokens_out = 20 + index % 7
        raw = f"synthetic output {index}"
        rows.append(
            GridRow(
                prompt_id=f"selftest-{index}",
                task="synthetic",
                family="qa",
                backend="selftest",
                requested_b=1.0,
                realised_r=1.0,
                target_model=model,
                sample_idx=0,
                temperature=0.0,
                seed=index,
                quality=1.0,
                T_in=tokens_in,
                T_out=tokens_out,
                latency_ms=1.0,
                compress_ms=0.0,
                usd_in=0.0,
                usd_out=0.0,
                usd_total=0.0,
                gpu_seconds=0.0,
                raw_output_hash=hashlib.sha256(raw.encode()).hexdigest(),
                code_version="selftest",
                price_table_version=PRICE_TABLE_VERSION,
            )
        )
    return rows


def _selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="frontier-ledger-") as directory:
        path = Path(directory) / "selftest.jsonl"
        ledger = Ledger(path)
        ledger.append_many(_synthetic_rows())
        frame = read_ledger(path)
        if len(frame) != 100 or not validate_ledger(frame):
            raise RuntimeError("ledger selftest failed")
        LOGGER.info("ledger_selftest_passed", rows=len(frame), path=str(path))
        print(f"selftest passed: wrote and validated {len(frame)} rows at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.error("pass --selftest")


if __name__ == "__main__":
    main()
