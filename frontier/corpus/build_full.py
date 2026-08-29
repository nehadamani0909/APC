"""Tiered P4 corpus plan and metadata attachment utilities."""

from __future__ import annotations

from collections.abc import Callable
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

    @property
    def generations(self) -> int:
        return self.prompts * self.budgets * self.models * self.samples


TIER_SPECS = (
    TierSpec("A", 6_000, 7, 1, 5),
    TierSpec("B", 1_200, 7, 1, 1),
    TierSpec("C", 1_200, 7, 1, 1),
    TierSpec("D", 800, 7, 1, 5),
)


def planned_generations() -> int:
    return sum(spec.generations for spec in TIER_SPECS)


def attach_metadata(
    frame: pd.DataFrame,
    *,
    source_document: Callable[[str], str] | None = None,
    split: Callable[[str], str] | None = None,
    tier: str = "A",
) -> pd.DataFrame:
    """Add required corpus metadata without changing ledger measurements."""

    result = frame.copy()
    result["source_document_id"] = result["prompt_id"].astype(str).map(
        source_document or (lambda prompt_id: prompt_id)
    )
    result["split"] = result["prompt_id"].astype(str).map(
        split or (lambda prompt_id: "unspecified")
    )
    result["tier"] = tier
    return result.loc[:, CORPUS_FIELDS]


def export_from_ledger(
    ledger_frame: pd.DataFrame,
    output: str | Path,
    *,
    tier: str = "A",
) -> Path:
    return export_parquet(attach_metadata(ledger_frame, tier=tier), output)
