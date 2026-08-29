"""CPC adapter stub.

No usable released CPC implementation is installed in this environment, so
P2 records the backend without pretending that it is runnable.
"""

from __future__ import annotations

from frontier.compress.base import TextCompressor


class CPCCompressor(TextCompressor):
    name = "cpc"
    backend_version = "unavailable"
    model_version = "unavailable"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        raise RuntimeError("CPC has no usable released implementation installed")
