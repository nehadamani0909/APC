from pathlib import Path

import pytest

from frontier.harness.ledger import GridRow, Ledger, read_ledger, validate_ledger


def row(prompt_id: str = "p1", **overrides: object) -> GridRow:
    values: dict[str, object] = {
        "prompt_id": prompt_id,
        "task": "task",
        "family": "qa",
        "backend": "test",
        "requested_b": 0.8,
        "realised_r": 0.75,
        "target_model": "gpt-4o-mini",
        "sample_idx": 0,
        "temperature": 0.7,
        "seed": 1,
        "quality": 0.9,
        "T_in": 1_000,
        "T_out": 200,
        "latency_ms": 10.0,
        "compress_ms": 2.0,
        "usd_in": 0.00015,
        "usd_out": 0.00012,
        "usd_total": 0.00027,
        "gpu_seconds": 0.1,
        "raw_output_hash": "abc",
        "code_version": "test",
        "price_table_version": "v1",
    }
    values.update(overrides)
    return GridRow(**values)  # type: ignore[arg-type]


def test_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append(row())
    assert len(read_ledger(path)) == 1
    assert validate_ledger(path)


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append_many([row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(path)


def test_cost_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append(row(usd_total=0.5))
    with pytest.raises(ValueError, match="usd_total"):
        validate_ledger(path)
