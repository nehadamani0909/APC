"""Append-only store for raw generations, keyed by ``raw_output_hash``.

APC-04 §8 keeps the generated text out of the ledger row and records only its
hash, with the text "stored separately".  This is that store.

Without it the secondary quality definition cannot be computed at all.
APC-04 §3.1.1 defines answer preservation as
``rho_ap(x,q,b) = agreement(y(x,b), y(x,1))``, whose whole point is that it
"sidesteps cases where the model was wrong anyway" -- exactly the situation
where graded accuracy degenerates, because a prompt the model always fails
has no compression tolerance to measure.  Recovering it after the fact means
re-running the grid, so the text has to be kept at generation time.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class OutputStore:
    """JSONL store mapping output hash to the text that produced it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Same reason as the ledger: concurrent appends are not atomic on
        # Windows, and the grid runs parallel workers.
        self._lock = Lock()
        self._seen: set[str] = set()
        if self.path.exists():
            self._seen = set(self.load().keys())

    def put(self, digest: str, text: str) -> None:
        """Record one generation. Repeated hashes are written once."""

        with self._lock:
            if digest in self._seen:
                return
            self._seen.add(digest)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"hash": digest, "text": text}, ensure_ascii=False)
                    + "\n"
                )

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        outputs: dict[str, str] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                outputs[str(record["hash"])] = str(record["text"])
        return outputs

    def __len__(self) -> int:
        return len(self._seen)
