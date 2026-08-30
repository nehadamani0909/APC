"""Credential-gated provider clients with usage-based cost caps.

SDK imports are lazy so offline tests never require credentials or network.
API keys are read from the environment only and are never accepted as
arguments, written to disk, or logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

from frontier.harness.models import ProviderClient, ProviderResponse
from frontier.harness.prices import PRICE_TABLE_VERSION, cost_from_tokens


class SpendCapExceeded(RuntimeError):
    """Raised when a call would take the run past its hard USD cap."""


class SpendCap:
    """Hard USD ceiling, reserved before a call and settled against usage.

    Reserving an estimate first means a run cannot blow through the cap
    between issuing a call and learning what it cost; settling afterwards
    against provider-reported usage keeps the running total honest rather
    than drifting with the estimate.
    """

    def __init__(self, cap_usd: float) -> None:
        if cap_usd < 0.0:
            raise ValueError("spend cap must be non-negative")
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._lock = Lock()

    def reserve(self, amount_usd: float) -> None:
        with self._lock:
            if self.spent_usd + amount_usd > self.cap_usd:
                raise SpendCapExceeded(
                    f"spend cap exceeded: {self.spent_usd:.6f} + {amount_usd:.6f} "
                    f"> {self.cap_usd:.6f}"
                )
            self.spent_usd += amount_usd

    def settle(self, reserved_usd: float, actual_usd: float) -> None:
        """Replace a reservation with the amount actually billed."""
        with self._lock:
            self.spent_usd += actual_usd - reserved_usd

    def release(self, reserved_usd: float) -> None:
        with self._lock:
            self.spent_usd -= reserved_usd

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return self.cap_usd - self.spent_usd


class RetryingClient:
    """Bounded retries with exponential backoff and a concurrency limit."""

    def __init__(
        self,
        client: ProviderClient,
        *,
        max_retries: int = 3,
        concurrency: int = 1,
        backoff_s: float = 0.25,
        sleep: Any = time.sleep,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.client = client
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.semaphore = Semaphore(concurrency)
        self._sleep = sleep

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        with self.semaphore:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    return self.client.generate(
                        prompt, temperature=temperature, seed=seed
                    )
                except SpendCapExceeded:
                    # A budget stop is terminal: retrying cannot help and
                    # would only burn the remaining allowance.
                    raise
                except Exception as exc:  # provider transient errors vary
                    last_error = exc
                    if attempt < self.max_retries:
                        self._sleep(self.backoff_s * (2**attempt))
            raise RuntimeError("provider request failed after retries") from last_error


class CachedProvider:
    """Disk cache keyed by the full generation identity.

    The key covers model and revision as well as the prompt and sampling
    parameters, so a pinned model change invalidates the cache rather than
    silently serving another model's output (APC-04 §7).
    """

    def __init__(
        self,
        client: ProviderClient,
        *,
        model: str,
        revision: str,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.client = client
        self.model = model
        self.revision = revision
        self.cache_dir = Path(cache_dir)

    def _path(self, prompt: str, temperature: float, seed: int) -> Path:
        material = json.dumps(
            [prompt, self.model, self.revision, temperature, seed],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return self.cache_dir / f"provider-{digest}.json"

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        path = self._path(prompt, temperature, seed)
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
                    "model": self.model,
                    "revision": self.revision,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return response


class _CappedClient:
    """Shared spend-cap bookkeeping for the API clients."""

    def __init__(
        self,
        *,
        model: str,
        spend_cap: SpendCap,
        price_table_version: str,
        estimated_cost_usd: float,
    ) -> None:
        self.model = model
        self.spend_cap = spend_cap
        self.price_table_version = price_table_version
        self.estimated_cost_usd = estimated_cost_usd

    def _settle(self, response: ProviderResponse) -> ProviderResponse:
        actual = cost_from_tokens(
            self.model,
            response.input_tokens,
            response.output_tokens,
            self.price_table_version,
        )[2]
        self.spend_cap.settle(self.estimated_cost_usd, actual)
        return response


class OpenAIClient(_CappedClient):
    """OpenAI chat-completions adapter using provider-reported token usage."""

    def __init__(
        self,
        *,
        model: str,
        spend_cap: SpendCap,
        client: Any = None,
        estimated_cost_usd: float = 0.0,
        price_table_version: str = PRICE_TABLE_VERSION,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            spend_cap=spend_cap,
            price_table_version=price_table_version,
            estimated_cost_usd=estimated_cost_usd,
        )
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client = client
        self.max_tokens = max_tokens

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        self.spend_cap.reserve(self.estimated_cost_usd)
        try:
            extra: dict[str, Any] = {}
            if self.max_tokens is not None:
                extra["max_tokens"] = self.max_tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                seed=seed,
                **extra,
            )
            usage = response.usage
            return self._settle(
                ProviderResponse(
                    response.choices[0].message.content or "",
                    int(usage.prompt_tokens),
                    int(usage.completion_tokens),
                )
            )
        except SpendCapExceeded:
            raise
        except Exception:
            self.spend_cap.release(self.estimated_cost_usd)
            raise


class AnthropicClient(_CappedClient):
    """Anthropic messages adapter using provider-reported token usage."""

    def __init__(
        self,
        *,
        model: str,
        spend_cap: SpendCap,
        client: Any = None,
        estimated_cost_usd: float = 0.0,
        price_table_version: str = PRICE_TABLE_VERSION,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            model=model,
            spend_cap=spend_cap,
            price_table_version=price_table_version,
            estimated_cost_usd=estimated_cost_usd,
        )
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.client = client
        self.max_tokens = max_tokens

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        # The Messages API has no seed parameter; k>1 sampling varies through
        # temperature alone, which is recorded in the ledger with the seed so
        # the distinction stays visible.
        del seed
        self.spend_cap.reserve(self.estimated_cost_usd)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return self._settle(
                ProviderResponse(
                    str(response.content[0].text),
                    int(response.usage.input_tokens),
                    int(response.usage.output_tokens),
                )
            )
        except SpendCapExceeded:
            raise
        except Exception:
            self.spend_cap.release(self.estimated_cost_usd)
            raise
