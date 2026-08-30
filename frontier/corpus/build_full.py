"""Tiered corpus plan and metadata attachment (APC-05 §4.1).

A naive full factorial is 630,000 generations and infeasible.  Tiering is
what makes the project affordable: the cheap local model carries the full
N x K x k grid, and the expensive API models run only on a family-stratified
subset of the same prompts, so cross-model transfer stays comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from frontier.corpus.validate import CORPUS_FIELDS, export_parquet


@dataclass(frozen=True)
class TierSpec:
    name: str
    prompts: int
    budgets: int
    models: int
    samples: int
    purpose: str = ""

    @property
    def generations(self) -> int:
        return self.prompts * self.budgets * self.models * self.samples


TIER_SPECS = (
    TierSpec("A", 6_000, 7, 1, 5, "main corpus: training and instance analysis"),
    TierSpec("B", 1_200, 7, 1, 1, "cross-model transfer, API model 1"),
    TierSpec("C", 1_200, 7, 1, 1, "cross-model transfer, API model 2"),
    TierSpec("D", 800, 7, 1, 5, "cross-compressor transfer (E7)"),
)
PILOT = TierSpec("Pilot", 200, 7, 1, 5, "Gate 1")


def planned_generations() -> int:
    return sum(spec.generations for spec in TIER_SPECS)


def stratified_subset(
    prompt_ids: Sequence[str],
    families: Sequence[str],
    size: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Family-stratified prompt sample for the API tiers.

    Tiers B and C must run on a *subset of the same prompts* as tier A, or
    cross-model comparison stops being paired and E6a measures two different
    prompt distributions rather than two models.
    """

    if size >= len(prompt_ids):
        return list(prompt_ids)
    frame = pd.DataFrame({"prompt_id": list(prompt_ids), "family": list(families)})
    per_family = max(1, size // max(1, frame["family"].nunique()))
    chosen: list[str] = []
    for _, group in frame.groupby("family", sort=True):
        take = min(per_family, len(group))
        chosen.extend(
            group.sample(n=take, random_state=seed)["prompt_id"].astype(str).tolist()
        )
    if len(chosen) < size:
        remaining = frame[~frame["prompt_id"].isin(chosen)]
        extra = min(size - len(chosen), len(remaining))
        if extra:
            chosen.extend(
                remaining.sample(n=extra, random_state=seed)["prompt_id"]
                .astype(str)
                .tolist()
            )
    return sorted(chosen[:size])


def attach_metadata(
    frame: pd.DataFrame,
    *,
    source_document: Callable[[str], str] | Mapping[str, str] | None = None,
    split: Callable[[str], str] | Mapping[str, str] | None = None,
    tier: str = "A",
) -> pd.DataFrame:
    """Add leakage-control and tier metadata without altering measurements."""

    def _resolve(
        lookup: Callable[[str], str] | Mapping[str, str] | None, default: str
    ) -> Callable[[str], str]:
        if lookup is None:
            return lambda prompt_id: (
                prompt_id if default == "__id__" else default
            )
        if isinstance(lookup, Mapping):
            mapping = dict(lookup)
            return lambda prompt_id: mapping.get(
                prompt_id, prompt_id if default == "__id__" else default
            )
        return lookup

    result = frame.copy()
    ids = result["prompt_id"].astype(str)
    result["source_document_id"] = ids.map(_resolve(source_document, "__id__"))
    result["split"] = ids.map(_resolve(split, "unspecified"))
    result["tier"] = tier
    return result.loc[:, CORPUS_FIELDS]


def export_from_ledger(
    ledger_frame: pd.DataFrame,
    output: str | Path,
    *,
    tier: str = "A",
    source_document: Mapping[str, str] | None = None,
    split: Mapping[str, str] | None = None,
) -> Path:
    return export_parquet(
        attach_metadata(
            ledger_frame,
            source_document=source_document,
            split=split,
            tier=tier,
        ),
        output,
    )
