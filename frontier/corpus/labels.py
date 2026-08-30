"""Curve labels derived from the corpus (APC-04 §3.1-3.2).

The corpus stores one row per generation sample.  Everything the predictor
trains against is an aggregate over those rows: the graded quality curve
``rho(x, b)``, the monotone-safe budget ``b*(x)``, the realised-rate curve,
and the output-length curve.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from frontier.predict.heads import BUDGETS

Array = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class CurveLabels:
    """Per-prompt curves, aligned row-wise with :attr:`prompt_ids`."""

    prompt_ids: tuple[str, ...]
    families: tuple[str, ...]
    quality: Array
    realised_rate: Array
    output_tokens: Array
    input_tokens: Array

    def __len__(self) -> int:
        return len(self.prompt_ids)


def _pivot(frame: pd.DataFrame, column: str, fill: float) -> Array:
    table = frame.pivot_table(
        index="prompt_id", columns="requested_b", values=column, aggfunc="mean"
    )
    # Reindex onto the canonical budget grid so a missing cell becomes an
    # explicit fill rather than silently shifting the column order.
    table = table.reindex(columns=[float(budget) for budget in BUDGETS])
    return cast(Array, table.to_numpy(dtype=float, na_value=fill))


def build_curves(frame: pd.DataFrame) -> CurveLabels:
    """Aggregate generation rows into per-prompt curves.

    Graded quality is the mean over the ``k`` sampled generations at each
    budget, which is exactly ``rho(x, q, b)`` in APC-04 §3.1.1.
    """

    if frame.empty:
        raise ValueError("cannot build curves from an empty corpus")
    ordered = frame.sort_values("prompt_id")
    quality = _pivot(ordered, "quality", 0.0)
    realised = _pivot(ordered, "realised_r", 1.0)
    outputs = _pivot(ordered, "T_out", 1.0)
    inputs = _pivot(ordered, "T_in", 1.0)
    prompt_ids = tuple(
        str(value)
        for value in ordered.pivot_table(
            index="prompt_id", columns="requested_b", values="quality", aggfunc="mean"
        ).index
    )
    family_by_prompt = (
        ordered.groupby("prompt_id")["family"].first().reindex(list(prompt_ids))
    )
    families = tuple(str(value) for value in family_by_prompt)
    return CurveLabels(prompt_ids, families, quality, realised, outputs, inputs)


def safety_indicator(quality: Array, epsilon: float) -> Array:
    """``1[rho(x, b) >= rho(x, 1) - epsilon]`` — the H1 training target.

    APC-04 §5.3 states H1's loss as binary cross-entropy against this
    indicator, so it is what the head is fitted on; the graded curve is kept
    separately for the E2 curve-MAE diagnostic.
    """

    baseline = quality[:, -1][:, None]
    return cast(Array, (quality >= baseline - epsilon).astype(float))


def monotone_safe_budget(quality: Array, epsilon: float) -> Array:
    """``b*(x) = min{b : rho(b') >= rho(1) - epsilon for all b' >= b}``.

    The infimum of the maximal safe *suffix* (APC-04 §3.2).  Robust to a
    single lucky low budget, which the naive ``min{b : safe}`` label is not.
    """

    safe = safety_indicator(quality, epsilon)
    # A budget qualifies only if it and every larger budget are safe.
    suffix_safe = np.minimum.accumulate(safe[:, ::-1], axis=1)[:, ::-1]
    result = np.full(len(quality), float(BUDGETS[-1]))
    for index in range(len(BUDGETS)):
        qualifies = suffix_safe[:, index] > 0.0
        unset = result == float(BUDGETS[-1])
        result = np.where(qualifies & unset, float(BUDGETS[index]), result)
    return cast(Array, result)


def naive_safe_budget(quality: Array, epsilon: float) -> Array:
    """``min{b : rho(b) >= rho(1) - epsilon}`` — reported only for the E1b gap."""

    safe = safety_indicator(quality, epsilon)
    indices = np.argmax(safe > 0.0, axis=1)
    has_safe = safe.max(axis=1) > 0.0
    return cast(
        Array, np.where(has_safe, BUDGETS[indices], float(BUDGETS[-1]))
    )


def answer_preservation(
    frame: pd.DataFrame,
    outputs: Mapping[str, str],
    parse: Callable[[str], str],
    prompt_ids: Sequence[str] | None = None,
) -> Array:
    """``rho_ap(x,b) = agreement(y(x,b), y(x,1))`` (APC-04 §3.1.1).

    Fidelity to the *uncompressed system's* behaviour rather than to a gold
    label.  This is the robustness definition, and it is the one that still
    carries signal when the target model fails a prompt at every budget:
    graded accuracy makes those prompts degenerate (every budget is
    trivially "safe", so ``b*`` collapses), whereas answer preservation still
    measures whether compression changed the answer.

    The uncompressed answer is taken as the modal parsed output at ``b=1.0``,
    and ``rho_ap`` is the fraction of samples at each budget reproducing it.
    """

    ordered = frame.assign(b=frame["requested_b"].round(4))
    ordered = ordered.assign(
        parsed=[
            parse(outputs.get(str(digest), "")) for digest in ordered["raw_output_hash"]
        ]
    )
    ids = (
        list(prompt_ids)
        if prompt_ids is not None
        else sorted(ordered["prompt_id"].astype(str).unique())
    )
    result = np.zeros((len(ids), len(BUDGETS)))
    for row, prompt_id in enumerate(ids):
        prompt = ordered[ordered["prompt_id"].astype(str) == prompt_id]
        uncompressed = prompt[prompt["b"] == 1.0]["parsed"].tolist()
        if not uncompressed:
            continue
        reference = Counter(uncompressed).most_common(1)[0][0]
        for column, budget in enumerate(BUDGETS):
            answers = prompt[prompt["b"] == round(float(budget), 4)]["parsed"].tolist()
            if answers:
                result[row, column] = sum(
                    answer == reference for answer in answers
                ) / len(answers)
    return cast(Array, result)
