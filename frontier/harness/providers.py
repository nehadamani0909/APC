"""Credential-gated provider clients with usage-based cost caps.

SDK imports are lazy so offline tests never require credentials or network.
The installed SDK signatures were checked for OpenAI ``Completions.create``
and Anthropic ``Messages.create`` before these adapters were written.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

from frontier.harness.models import ProviderResponse


class SpendCap:
    def __init__(self, cap_usd: float) -> None:
        if cap_usd < 0.0:
            raise ValueError("spend cap must be non-negative")
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._lock = Lock()

    def reserve(self, amount_usd: float) -> None:
        with self._lock:
            if self.spent_usd + amount_usd > self.cap_usd:
                raise RuntimeError("provider spend cap exceeded")
            self.spent_usd += amount_usd


class RetryingClient:
    """Wrap a provider request with bounded retries and concurrency control."""

    def __init__(
        self,
        request: Callable[[], ProviderResponse],
        *,
        spend_cap: SpendCap,
        estimated_cost_usd: float = 0.0,
        max_retries: int = 3,
        concurrency: int = 1,
    ) -> None:
        self.request = request
        self.spend_cap = spend_cap
        self.estimated_cost_usd = estimated_cost_usd
        self.max_retries = max_retries
        self.semaphore = Semaphore(concurrency)

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        del prompt, temperature, seed
        with self.semaphore:
            self.spend_cap.reserve(self.estimated_cost_usd)
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.request()
                    return response
                except RuntimeError:
                    raise
                except Exception as exc:  # provider-specific transient errors vary
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(0.25 * (2**attempt))
            raise RuntimeError("provider request failed after retries") from last_error


class CachedProvider:
    """Disk cache keyed by the full generation identity."""

    def __init__(self, client: Any, cache_dir: str | Path = "data/cache") -> None:
        self.client = client
        self.cache_dir = Path(cache_dir)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int,
        model: str,
        revision: str,
    ) -> ProviderResponse:
        material = json.dumps(
            [prompt, model, revision, temperature, seed], separators=(",", ":")
        )
        digest = hashlib.sha256(material.encode()).hexdigest()
        path = self.cache_dir / f"provider-{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return ProviderResponse(
                str(cached["text"]),
                int(cached["input_tokens"]),
                int(cached["output_tokens"]),
            )
        response = self.client.generate(prompt, temperature=temperature, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ProviderResponse(
            str(response.text), int(response.input_tokens), int(response.output_tokens)
        )


class OpenAIClient:
    """OpenAI chat-completions adapter using provider-reported token usage."""

    def __init__(
        self,
        *,
        model: str,
        spend_cap: SpendCap,
        client: Any = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client = client
        self.model = model
        self.spend_cap = spend_cap
        self.estimated_cost_usd = estimated_cost_usd

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        self.spend_cap.reserve(self.estimated_cost_usd)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
        )
        usage = response.usage
        return ProviderResponse(
            response.choices[0].message.content or "",
            int(usage.prompt_tokens),
            int(usage.completion_tokens),
        )


class AnthropicClient:
    """Anthropic messages adapter using provider-reported token usage."""

    def __init__(
        self,
        *,
        model: str,
        spend_cap: SpendCap,
        client: Any = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.client = client
        self.model = model
        self.spend_cap = spend_cap
        self.estimated_cost_usd = estimated_cost_usd

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        del seed
        self.spend_cap.reserve(self.estimated_cost_usd)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return ProviderResponse(
            str(response.content[0].text),
            int(response.usage.input_tokens),
            int(response.usage.output_tokens),
        )
