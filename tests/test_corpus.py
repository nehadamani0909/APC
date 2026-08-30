from pathlib import Path

import pandas as pd
import pytest

from frontier.corpus.build_full import (
    TIER_SPECS,
    attach_metadata,
    planned_generations,
)
from frontier.corpus.validate import (
    export_parquet,
    read_corpus,
    validate_corpus,
    write_stats_report,
)
from frontier.harness.ledger import GridRow


def _frame() -> pd.DataFrame:
    row = GridRow(
        prompt_id="p1", task="task", family="qa", backend="test",
        requested_b=1.0, realised_r=1.0, target_model="local-default",
        sample_idx=0, temperature=0.0, seed=0, quality=1.0, T_in=10, T_out=2,
        latency_ms=1.0, compress_ms=0.0, usd_in=0.0, usd_out=0.0,
        usd_total=0.0, gpu_seconds=0.1, raw_output_hash="hash",
        code_version="test", price_table_version="v1",
    )
    record = row.__dict__ | {"run_id": "run", "ts": "now", "phase": "test"}
    return attach_metadata(pd.DataFrame([record]))


def test_tier_plan_and_parquet_round_trip(tmp_path: Path) -> None:
    assert planned_generations() == 254800
    assert [spec.name for spec in TIER_SPECS] == ["A", "B", "C", "D"]
    frame = _frame()
    path = export_parquet(frame, tmp_path / "corpus.parquet")
    loaded = read_corpus(path)
    assert validate_corpus(loaded)
    report = write_stats_report(loaded, tmp_path / "corpus_stats.md")
    assert "T2" in report.read_text(encoding="utf-8")


def test_split_leakage_is_rejected() -> None:
    frame = _frame()
    second = frame.copy()
    second["split"] = "test"
    with pytest.raises(ValueError, match="multiple splits"):
        validate_corpus(pd.concat([frame, second], ignore_index=True))


def test_api_spend_cap_is_enforced() -> None:
    with pytest.raises(ValueError, match="spend cap exceeded"):
        validate_corpus(_frame(), api_spend_cap_usd=-0.01)


def test_stratified_subset_is_family_balanced_and_paired() -> None:
    from frontier.corpus.build_full import stratified_subset

    prompt_ids = [f"p{i}" for i in range(40)]
    families = ["qa" if i % 2 else "code" for i in range(40)]
    chosen = stratified_subset(prompt_ids, families, 10, seed=0)
    assert len(chosen) == 10
    # Tiers B/C must sample the SAME prompts as tier A, or cross-model
    # comparison stops being paired.
    assert set(chosen).issubset(set(prompt_ids))
    picked = {pid: fam for pid, fam in zip(prompt_ids, families, strict=True)}
    counts = {"qa": 0, "code": 0}
    for pid in chosen:
        counts[picked[pid]] += 1
    assert counts["qa"] == counts["code"] == 5
    # Deterministic for a fixed seed.
    assert stratified_subset(prompt_ids, families, 10, seed=0) == chosen


def test_attach_metadata_accepts_real_lookups() -> None:
    frame = _frame()
    attached = attach_metadata(
        frame,
        source_document={"p1": "doc-7"},
        split={"p1": "D_test"},
        tier="B",
    )
    assert attached["source_document_id"].tolist() == ["doc-7"]
    assert attached["split"].tolist() == ["D_test"]
    assert attached["tier"].tolist() == ["B"]
