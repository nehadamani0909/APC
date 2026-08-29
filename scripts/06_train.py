"""Train the deterministic NumPy predictor on corpus-derived curve labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontier.corpus.validate import read_corpus, validate_corpus
from frontier.features.surface import SurfaceExtractor
from frontier.predict.frontier import FEATURES
from frontier.predict.heads import BUDGETS, CumulativeLogitHead, OutputLengthHead


def train_from_corpus(corpus_path: Path, artifact_path: Path) -> None:
    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    extractor = SurfaceExtractor()
    prompts = corpus[["prompt_id", "family"]].drop_duplicates()
    features = []
    labels = []
    outputs = []
    for prompt_id, family in prompts.itertuples(index=False, name=None):
        context = f"context for {prompt_id} family {family}"
        result = extractor.extract(str(prompt_id), context, str(family))
        features.append([result.features[name] for name in FEATURES])
        rows = corpus[corpus["prompt_id"] == prompt_id]
        grouped = rows.groupby("requested_b")
        quality_by_budget = {
            round(float(rate), 8): float(group["quality"].mean())
            for rate, group in grouped
        }
        output_by_budget = {
            round(float(rate), 8): float(group["T_out"].mean())
            for rate, group in grouped
        }
        labels.append(
            [quality_by_budget.get(round(float(rate), 8), 0.0) for rate in BUDGETS]
        )
        outputs.append(
            [output_by_budget.get(round(float(rate), 8), 1.0) for rate in BUDGETS]
        )
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    output_array = np.asarray(outputs, dtype=float)
    quality = CumulativeLogitHead(len(FEATURES), seed=0)
    output = OutputLengthHead(len(FEATURES))
    quality.fit(feature_array, label_array)
    output.fit(feature_array, output_array)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "artifact": "frontier-predictor",
                "version": "corpus-fit-v1",
                "feature_tier": "L0",
                "features": list(FEATURES),
                "budgets": [0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0],
                "training_split": "D_train",
                "prompt_count": len(prompts),
                "quality_base_weights": quality.base_weights.tolist(),
                "quality_base_bias": quality.base_bias,
                "quality_increment_bias": quality.increment_bias.tolist(),
                "output_weights": output.weights.tolist(),
                "output_bias": output.bias,
                "status": "trained on committed corpus artifact",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/predictor.json")
    )
    args = parser.parse_args()
    train_from_corpus(args.corpus, args.artifact)
    print(f"wrote trained predictor to {args.artifact}")


if __name__ == "__main__":
    main()
