"""Feature standardisation shared by every predictor head.

L0 features span several orders of magnitude (``context_token_count`` is a
raw count; the ratios live in ``[0, 1]``).  Fitting on the raw values makes
the optimiser's effective learning rate wildly feature-dependent and drives
the fitted weights toward zero, so every head is fitted on standardised
features and the constants are persisted alongside the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

Array = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class FeatureScaler:
    """Zero-mean, unit-variance scaling with persisted constants."""

    mean: Array
    scale: Array

    @classmethod
    def fit(cls, features: Array) -> FeatureScaler:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("features must be a non-empty 2-D array")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        # A constant column carries no signal; leaving its scale at 1.0 maps
        # it to a column of zeros rather than dividing by ~0.
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(cast(Array, mean), cast(Array, scale))

    def transform(self, features: Array) -> Array:
        values = np.asarray(features, dtype=float)
        if values.shape[-1] != len(self.mean):
            raise ValueError("feature count does not match the fitted scaler")
        return cast(Array, (values - self.mean) / self.scale)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> FeatureScaler:
        return cls(
            cast(Array, np.asarray(payload["mean"], dtype=float)),
            cast(Array, np.asarray(payload["scale"], dtype=float)),
        )
