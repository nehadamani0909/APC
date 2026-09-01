"""Ollama provider client for zero-cost local target-model runs.

The accounting contract is the same as every other provider: token counts
come from the server's own response (``prompt_eval_count`` / ``eval_count``),
never from local re-tokenisation, so the ledger's usage column means the same
thing across local and API backends.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from frontier.harness.models import ProviderResponse


class ContextOverflow(RuntimeError):
    """The server truncated the prompt to fit its context window.

    This is fatal for compression research specifically.  Ollama silently
    drops the overflow rather than erroring, so an over-long prompt is
    quietly *truncated* -- which is itself a compression operation, and the
    crudest one available.  At ``b = 1.0`` that would mean the "uncompressed"
    reference quality rho(x, 1) was measured on a truncated prompt, so every
    per-instance frontier in the corpus would be computed against a
    contaminated baseline, and nothing downstream could detect it.

    Failing loudly here is the only way the b = 1.0 cell can be trusted.
    """


@dataclass
class OllamaClient:
    """Minimal Ollama client with an enforced context window."""

    model: str = "llama3.2:latest"
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    num_predict: int = 512
    timeout_s: float = 600.0

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            payload,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.load(response)

        in_tokens = int(body.get("prompt_eval_count", 0))
        out_tokens = int(body.get("eval_count", 0))

        # Ollama reports how much of the prompt it actually read. If that
        # lands at the ceiling, the rest was discarded -- see ContextOverflow.
        if in_tokens >= self.num_ctx - self.num_predict:
            raise ContextOverflow(
                f"prompt_eval_count={in_tokens} reached the usable context "
                f"(num_ctx={self.num_ctx}, num_predict={self.num_predict}): "
                "the prompt was silently truncated by the server. Raise "
                "num_ctx or shorten the instance; do not record this row."
            )
        return ProviderResponse(
            text=str(body.get("response", "")),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
