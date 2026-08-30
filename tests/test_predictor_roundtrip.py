"""Predictor artifact round-trip and Gate 1 verdict handling.

Both were at or near zero coverage: ``from_artifact`` is how OURS becomes
the method rather than an untrained head, and the Gate 1 verdict is the
project's go/no-go.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from frontier.eval.gate1 import _noise_floor, write_gate1_report
from frontier.predict.frontier import FEATURES, FrontierPredictor
from frontier.predict.heads import BUDGETS

Array = np.ndarray[Any, np.dtype[np.float64]]


def _train_and_save(tmp_path: Path) -> Path:
    from frontier.predict.heads import (
        CumulativeLogitHead,
        OutputLengthHead,
        RateAdherenceHead,
    )
    from frontier.predict.scaling import FeatureScaler

    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, len(FEATURES)))
    labels = np.tile(np.linspace(0.0, 1.0, len(BUDGETS)), (120, 1))
    labels[:, 0] = (features[:, 0] > 0).astype(float)

    scaler = FeatureScaler.fit(features)
    scaled = scaler.transform(features)
    quality = CumulativeLogitHead(len(FEATURES), seed=0)
    quality.fit(scaled, labels)
    adherence = RateAdherenceHead(len(FEATURES))
    adherence.fit(scaled, np.tile(BUDGETS * 0.9, (120, 1)))
    output = OutputLengthHead(len(FEATURES))
    output.fit(scaled, np.tile(np.linspace(400.0, 100.0, len(BUDGETS)), (120, 1)))

    path = tmp_path / "predictor.json"
    path.write_text(
        json.dumps(
            {
                "artifact": "frontier-predictor",
                "version": "2",
                "provenance": {"price_table_version": "v1"},
                "features": list(FEATURES),
                "budgets": [float(b) for b in BUDGETS],
                "scaler": scaler.to_dict(),
                "heads": {
                    "quality": {
                        "base_weights": quality.base_weights.tolist(),
                        "base_bias": quality.base_bias,
                        "increment_weights": quality.increment_weights.tolist(),
                        "increment_bias": quality.increment_bias.tolist(),
                    },
                    "adherence": {
                        "weights": adherence.weights.tolist(),
                        "bias": adherence.bias,
                    },
                    "output_length": {
                        "weights": output.weights.tolist(),
                        "bias": output.bias.tolist(),
                    },
                },
                "calibration": {"method": "temperature", "temperature": 2.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_from_artifact_restores_the_trained_weights(tmp_path: Path) -> None:
    path = _train_and_save(tmp_path)
    loaded = FrontierPredictor.from_artifact(path, model="gpt-4o-mini")
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert np.allclose(
        loaded.quality.increment_weights,
        np.asarray(saved["heads"]["quality"]["increment_weights"]),
    )
    # A round trip that silently dropped the increment weights would restore
    # an instance-independent curve, which is the C3 failure mode.
    assert np.abs(loaded.quality.increment_weights).max() > 1e-3
    assert loaded.output_length.weights.shape == (len(BUDGETS), len(FEATURES))
    assert loaded.temperature == 2.0
    assert loaded.scaler is not None


def test_loaded_predictor_differs_from_an_untrained_one(tmp_path: Path) -> None:
    path = _train_and_save(tmp_path)
    trained = FrontierPredictor.from_artifact(path, model="gpt-4o-mini")
    untrained = FrontierPredictor(model="gpt-4o-mini")

    context = "A fairly long context with several sentences. " * 6
    a = trained.predict(context, "a query", "qa")
    b = untrained.predict(context, "a query", "qa")
    assert not np.allclose(a.quality, b.quality)

    # Contract: one value per budget, monotone quality, positive cost.
    for prediction in (a, b):
        assert prediction.quality.shape == (len(BUDGETS),)
        assert np.all(prediction.quality[:-1] <= prediction.quality[1:] + 1e-12)
        assert np.all(prediction.cost > 0.0)


def test_output_length_head_makes_cost_budget_dependent(tmp_path: Path) -> None:
    path = _train_and_save(tmp_path)
    predictor = FrontierPredictor.from_artifact(path, model="gpt-4o-mini")
    prediction = predictor.predict("some context " * 40, "q", "qa")
    # C2 needs cost to vary with the budget beyond the input-token term.
    assert float(prediction.cost.std()) > 0.0


def test_from_artifact_rejects_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"artifact": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a frontier predictor"):
        FrontierPredictor.from_artifact(path)


def _ledger_frame(samples: int) -> pd.DataFrame:
    rows = []
    for prompt in range(6):
        for budget in BUDGETS:
            for sample in range(samples):
                rows.append(
                    {
                        "prompt_id": f"p{prompt}",
                        "family": "qa",
                        "backend": "truncate_tail",
                        "requested_b": float(budget),
                        "realised_r": float(budget),
                        "quality": 1.0 if budget >= 0.5 else float(prompt % 2),
                        "T_out": 100.0,
                        "sample_idx": sample,
                    }
                )
    return pd.DataFrame(rows)


def test_noise_floor_is_none_when_it_cannot_be_estimated() -> None:
    # k=1 gives no split-half estimate. Returning 0.0 would make the gate
    # compare against a zero threshold and look decisive.
    assert _noise_floor(_ledger_frame(samples=1), 0.05) is None
    assert _noise_floor(_ledger_frame(samples=4), 0.05) is not None


def test_gate1_reports_inconclusive_rather_than_fail_at_k1(tmp_path: Path) -> None:
    ledger = tmp_path / "k1.jsonl"
    _ledger_frame(samples=1).to_json(ledger, orient="records", lines=True)
    report = tmp_path / "gate1.md"
    write_gate1_report(ledger, report)
    text = report.read_text(encoding="utf-8")
    # A FAIL asserts heterogeneity is within noise; that was never measured.
    assert "INCONCLUSIVE" in text
    assert "not estimable" in text
    assert "Overall Gate 1 result: **FAIL**" not in text


def test_gate1_reaches_a_verdict_when_the_floor_is_estimable(tmp_path: Path) -> None:
    ledger = tmp_path / "k4.jsonl"
    _ledger_frame(samples=4).to_json(ledger, orient="records", lines=True)
    report = tmp_path / "gate1.md"
    write_gate1_report(ledger, report)
    text = report.read_text(encoding="utf-8")
    assert "INCONCLUSIVE" not in text
    assert "σ²_noise=" in text
    assert ("**PASS**" in text) or ("**FAIL**" in text)
