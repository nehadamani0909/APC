"""Train the predictor heads on corpus-derived curve labels.

Fits on ``D_train`` only, calibrates H1 on ``D_cal_a`` only, and never touches
``D_cal_b`` (reserved for the CRC threshold) or ``D_test``.  Calibrating and
selecting the CRC threshold on the same split would silently void the C1
guarantee (APC-04 §5.3), so the splits are kept apart here by construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from frontier.corpus.labels import (
    CurveLabels,
    build_curves,
    monotone_safe_budget,
    naive_safe_budget,
    safety_indicator,
)
from frontier.corpus.text import PromptText, load_prompt_text
from frontier.corpus.validate import read_corpus, validate_corpus
from frontier.features.surface import SurfaceExtractor
from frontier.harness.prices import PRICE_TABLE_VERSION
from frontier.predict.calibrate import (
    IsotonicModel,
    expected_calibration_error,
    temperature_scale,
)
from frontier.predict.frontier import FEATURES
from frontier.predict.heads import (
    BUDGETS,
    BudgetClassifier,
    CumulativeLogitHead,
    FreeFormHead,
    OutputLengthHead,
    RateAdherenceHead,
)
from frontier.predict.scaling import FeatureScaler
from frontier.predict.train import assert_disjoint, resolve_splits

Array = np.ndarray[Any, np.dtype[np.float64]]
DEFAULT_EPSILON = 0.05


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _features_for(
    prompt_ids: tuple[str, ...],
    families: tuple[str, ...],
    texts: dict[str, PromptText],
) -> tuple[Array, bool]:
    """Extract L0 features, reporting whether real prompt text was available."""

    extractor = SurfaceExtractor()
    rows: list[list[float]] = []
    real = bool(texts)
    for prompt_id, family in zip(prompt_ids, families, strict=True):
        text = texts.get(prompt_id)
        if text is None:
            real = False
            # Placeholder so the pipeline still runs on a corpus with no
            # accompanying instance text; the artifact records that the
            # features are not derived from real prompts.
            text = PromptText(f"context for {prompt_id} family {family}", str(family))
        result = extractor.extract(prompt_id, text.context, text.query)
        rows.append([result.features[name] for name in FEATURES])
    return cast(Array, np.asarray(rows, dtype=float)), real


def _subset(curves: CurveLabels, keep: set[str]) -> tuple[CurveLabels, Array]:
    mask = np.asarray([pid in keep for pid in curves.prompt_ids], dtype=bool)
    indices = np.flatnonzero(mask)
    subset = CurveLabels(
        tuple(curves.prompt_ids[i] for i in indices),
        tuple(curves.families[i] for i in indices),
        curves.quality[indices],
        curves.realised_rate[indices],
        curves.output_tokens[indices],
        curves.input_tokens[indices],
    )
    return subset, cast(Array, indices)


def train_from_corpus(
    corpus_path: Path,
    artifact_path: Path,
    *,
    instances_dir: Path | None = None,
    epsilon: float = DEFAULT_EPSILON,
    seed: int = 0,
    report_path: Path | None = None,
) -> dict[str, Any]:
    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    curves = build_curves(corpus)
    splits = resolve_splits(corpus, seed=seed)
    assert_disjoint(splits)

    texts = load_prompt_text(instances_dir)
    features, real_text = _features_for(curves.prompt_ids, curves.families, texts)

    train, train_index = _subset(curves, splits["D_train"])
    cal_a, cal_a_index = _subset(curves, splits["D_cal_a"])
    if len(train) == 0:
        raise ValueError("D_train is empty; cannot train")

    scaler = FeatureScaler.fit(features[train_index])
    scaled_train = scaler.transform(features[train_index])

    # H1's target is the binary safety indicator, per APC-04 §5.3.
    train_labels = safety_indicator(train.quality, epsilon)
    quality = CumulativeLogitHead(len(FEATURES), seed=seed)
    quality.fit(scaled_train, train_labels)

    free_form = FreeFormHead(len(FEATURES), seed=seed)
    free_form.fit(scaled_train, train_labels)
    classifier = BudgetClassifier(len(FEATURES), seed=seed)
    classifier.fit(scaled_train, train_labels)

    adherence = RateAdherenceHead(len(FEATURES))
    adherence.fit(scaled_train, train.realised_rate)
    output_length = OutputLengthHead(len(FEATURES))
    output_length.fit(scaled_train, train.output_tokens)

    # Calibration uses D_cal_a and nothing else.
    calibration: dict[str, Any] = {"method": "none", "temperature": 1.0}
    isotonic: IsotonicModel | None = None
    if len(cal_a) > 1:
        scaled_cal_a = scaler.transform(features[cal_a_index])
        cal_a_labels = safety_indicator(cal_a.quality, epsilon)
        raw = quality.predict(scaled_cal_a)
        scaled = temperature_scale(raw, cal_a_labels)
        isotonic = IsotonicModel.fit(raw, cal_a_labels)
        calibration = {
            "method": "temperature",
            "temperature": scaled.temperature,
            "split": "D_cal_a",
            "ece_per_budget_raw": [
                float(value) for value in expected_calibration_error(raw, cal_a_labels)
            ],
            "ece_per_budget_calibrated": [
                float(value)
                for value in expected_calibration_error(
                    scaled.probabilities, cal_a_labels
                )
            ],
            "isotonic": isotonic.to_dict(),
        }

    artifact: dict[str, Any] = {
        "artifact": "frontier-predictor",
        "version": "2",
        "provenance": {
            "created_utc": datetime.now(UTC).isoformat(),
            "code_version": _git_revision(),
            "corpus_path": str(corpus_path),
            "corpus_rows": int(len(corpus)),
            "price_table_version": PRICE_TABLE_VERSION,
            "seed": seed,
            "epsilon": epsilon,
            "feature_tier": "L0",
            "features_from_real_prompt_text": real_text,
            "backends": sorted(corpus["backend"].astype(str).unique()),
            "target_models": sorted(corpus["target_model"].astype(str).unique()),
            "split_sizes": {name: len(ids) for name, ids in splits.items()},
        },
        "features": list(FEATURES),
        "budgets": [float(budget) for budget in BUDGETS],
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
                "weights": output_length.weights.tolist(),
                "bias": output_length.bias.tolist(),
            },
        },
        "calibration": calibration,
    }
    if not real_text:
        artifact["status"] = (
            "SCAFFOLD: features were derived from prompt metadata, not real "
            "prompt text. Pass --instances-dir with normalised JSONL to train "
            "a usable predictor."
        )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    if report_path is not None:
        _write_e2_e4_report(
            report_path,
            scaler=scaler,
            features=features,
            curves=curves,
            splits=splits,
            epsilon=epsilon,
            quality=quality,
            free_form=free_form,
            classifier=classifier,
            real_text=real_text,
        )
    return artifact


def _write_e2_e4_report(
    path: Path,
    *,
    scaler: FeatureScaler,
    features: Array,
    curves: CurveLabels,
    splits: dict[str, set[str]],
    epsilon: float,
    quality: CumulativeLogitHead,
    free_form: FreeFormHead,
    classifier: BudgetClassifier,
    real_text: bool,
) -> None:
    """E2 curve prediction and E4 monotone-vs-free-form-vs-classifier."""

    test, index = _subset(curves, splits["D_test"])
    lines = ["# E2 / E4 predictor report", ""]
    if not real_text:
        lines += [
            "> **SCAFFOLD OUTPUT - NOT A RESULT.** Features were derived from",
            "> prompt metadata rather than real prompt text.",
            "",
        ]
    if len(test) == 0:
        lines.append("D_test is empty; no held-out metrics available.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    scaled = scaler.transform(features[index])
    labels = safety_indicator(test.quality, epsilon)
    truth = monotone_safe_budget(test.quality, epsilon)
    naive = naive_safe_budget(test.quality, epsilon)

    def budget_from_curve(predictions: Array) -> Array:
        safe = (predictions >= 0.5).astype(float)
        suffix = np.minimum.accumulate(safe[:, ::-1], axis=1)[:, ::-1]
        chosen = np.full(len(predictions), float(BUDGETS[-1]))
        for position in range(len(BUDGETS)):
            unset = chosen == float(BUDGETS[-1])
            chosen = np.where(
                (suffix[:, position] > 0) & unset, float(BUDGETS[position]), chosen
            )
        return cast(Array, chosen)

    rows: list[tuple[str, Array]] = [
        ("monotone (H1)", quality.predict(scaled)),
        ("free-form (E4 ablation)", free_form.predict(scaled)),
    ]
    lines += [
        f"Held-out prompts: {len(test)} | epsilon = {epsilon}",
        "",
        "## E2 curve prediction",
        "",
        "| Head | Curve MAE | b* MAE | b* exact match | mean ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, predictions in rows:
        curve_mae = float(np.abs(predictions - labels).mean())
        predicted_budget = budget_from_curve(predictions)
        lines.append(
            f"| {name} | {curve_mae:.4f} | "
            f"{float(np.abs(predicted_budget - truth).mean()):.4f} | "
            f"{float((predicted_budget == truth).mean()):.4f} | "
            f"{float(expected_calibration_error(predictions, labels).mean()):.4f} |"
        )

    predicted_class = BUDGETS[np.argmax(classifier.predict(scaled), axis=1)]
    curve_budget = budget_from_curve(rows[0][1])
    curve_mae = float(np.abs(curve_budget - truth).mean())
    curve_exact = float((curve_budget == truth).mean())
    class_mae = float(np.abs(predicted_class - truth).mean())
    class_exact = float((predicted_class == truth).mean())
    lines += [
        "",
        "## E4 curve versus direct classification",
        "",
        "| Approach | b* MAE | b* exact match |",
        "|---|---:|---:|",
        f"| curve head | {curve_mae:.4f} | {curve_exact:.4f} |",
        f"| K-way classifier | {class_mae:.4f} | {class_exact:.4f} |",
        "",
        "## E1b label non-monotonicity",
        "",
        f"- P(b*_naive != b*) = {float((naive != truth).mean()):.4f}",
        f"- mean |b*_naive - b*| = {float(np.abs(naive - truth).mean()):.4f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/predictor.json")
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("data/raw"),
        help="normalised instance JSONL, for real L0 features",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("reports/e2_e4.md"))
    args = parser.parse_args()
    artifact = train_from_corpus(
        args.corpus,
        args.artifact,
        instances_dir=args.instances_dir,
        epsilon=args.epsilon,
        seed=args.seed,
        report_path=args.report,
    )
    print(f"wrote {args.artifact}")
    if "status" in artifact:
        print(artifact["status"])


if __name__ == "__main__":
    main()
