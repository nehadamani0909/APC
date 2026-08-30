"""Join corpus rows back to the prompt text they were generated from.

The corpus stores measurements, not prompts (APC-04 §8 keeps raw text out of
the row and records only ``raw_output_hash``).  Feature extraction and any
policy that reads ``x`` or ``q`` therefore has to reload the normalised
instance JSONL and join on ``prompt_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptText:
    context: str
    query: str


def load_prompt_text(instances_dir: Path | None) -> dict[str, PromptText]:
    """Map prompt id to context and query from normalised JSONL."""

    texts: dict[str, PromptText] = {}
    if instances_dir is None or not Path(instances_dir).exists():
        return texts
    for path in sorted(Path(instances_dir).rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            texts[str(record["id"])] = PromptText(
                str(record["context"]), str(record["query"])
            )
    return texts
