"""Corpus record schema layered on the frozen ledger row."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from frontier.harness.ledger import GridRow

CORPUS_METADATA_FIELDS = ("source_document_id", "split", "tier")


@dataclass(frozen=True)
class CorpusRow:
    """A grid row plus leakage-control and tier metadata."""

    grid: GridRow
    source_document_id: str
    split: str
    tier: str

    def to_record(self) -> dict[str, object]:
        record = asdict(self.grid)
        record.update(
            {
                "source_document_id": self.source_document_id,
                "split": self.split,
                "tier": self.tier,
            }
        )
        return record
