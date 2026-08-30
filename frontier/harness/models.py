"""Provider-agnostic target-LLM adapters with usage-based accounting."""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from frontier.harness.ledger import GridRow, Ledger
from frontier.harness.prices import PRICE_TABLE_VERSION, cost_from_tokens
from frontier.harness.tasks import Instance, Task


@dataclass(frozen=True)
class GenerationResult:
    text: str
    T_in: int
    T_out: int
    latency_s: float
    usd: float
    model: str
    model_revision: str


@runtime_checkable
class TargetLLM(Protocol):
    model: str
    model_revision: str

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderClient(Protocol):
    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse: ...


class APIBackend:
    """Adapter around an injected provider client.

    The client owns the provider-specific SDK call.  This prevents local
    re-tokenisation from contaminating usage and makes the adapter testable
    without credentials or network access.
    """

    def __init__(
        self,
        client: ProviderClient,
        *,
        model: str,
        model_revision: str,
        price_table_version: str = PRICE_TABLE_VERSION,
    ) -> None:
        self.client = client
        self.model = model
        self.model_revision = model_revision
        self.price_table_version = price_table_version

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        started = time.perf_counter()
        response = self.client.generate(
            prompt, temperature=temperature, seed=seed
        )
        latency_s = time.perf_counter() - started
        usd = cost_from_tokens(
            self.model,
            response.input_tokens,
            response.output_tokens,
            self.price_table_version,
        )[2]
        return GenerationResult(
            response.text,
            response.input_tokens,
            response.output_tokens,
            latency_s,
            usd,
            self.model,
            self.model_revision,
        )


def run_task_once(
    task: Task,
    instance: Instance,
    target: TargetLLM,
    ledger: Ledger,
    *,
    sample_idx: int = 0,
    temperature: float = 0.0,
    seed: int = 0,
    code_version: str = "p1",
) -> GridRow:
    """Generate once and append a complete uncompressed P1 ledger row."""

    result = target.generate(
        task.build_prompt(instance.context, instance.query),
        temperature=temperature,
        seed=seed,
    )
    price_table_version = getattr(target, "price_table_version", PRICE_TABLE_VERSION)
    usd_in, usd_out, usd_total = cost_from_tokens(
        target.model, result.T_in, result.T_out, price_table_version
    )
    if abs(usd_total - result.usd) > max(1e-9, usd_total) * 0.01:
        raise ValueError("target adapter USD does not reconcile with provider usage")
    row = GridRow(
        prompt_id=instance.id,
        task=task.name,
        family=task.family,
        backend="none",
        requested_b=1.0,
        realised_r=1.0,
        target_model=target.model,
        sample_idx=sample_idx,
        temperature=temperature,
        seed=seed,
        quality=task.metric(task.parse(result.text), instance.gold),
        T_in=result.T_in,
        T_out=result.T_out,
        latency_ms=result.latency_s * 1000.0,
        compress_ms=0.0,
        usd_in=usd_in,
        usd_out=usd_out,
        usd_total=usd_total,
        gpu_seconds=0.0,
        raw_output_hash=hashlib.sha256(result.text.encode()).hexdigest(),
        code_version=f"{code_version};model_revision={target.model_revision}",
        price_table_version=price_table_version,
    )
    ledger.append(row)
    return row


class VLLMBackend:
    """Adapter for an injected vLLM engine using its public ``generate`` call.

    A real engine and sampling-parameter factory are supplied by deployment
    code because vLLM is optional.  Constructor signatures are checked at
    runtime before invocation, avoiding an unverified package API at import.
    """

    def __init__(
        self,
        engine: Any,
        sampling_params_factory: Any,
        *,
        model: str,
        model_revision: str,
        price_table_version: str = PRICE_TABLE_VERSION,
    ) -> None:
        generate = getattr(engine, "generate", None)
        if not callable(generate):
            raise TypeError("vLLM engine must expose callable generate")
        if not callable(sampling_params_factory):
            raise TypeError("sampling_params_factory must be callable")
        self.engine = engine
        self.sampling_params_factory = sampling_params_factory
        self.model = model
        self.model_revision = model_revision
        self.price_table_version = price_table_version

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int = 0
    ) -> GenerationResult:
        params = self.sampling_params_factory(temperature=temperature, seed=seed)
        signature = inspect.signature(self.engine.generate)
        if len(signature.parameters) < 2:
            raise TypeError(
                "vLLM engine.generate signature must accept prompts and params"
            )
        started = time.perf_counter()
        outputs = self.engine.generate([prompt], params)
        latency_s = time.perf_counter() - started
        try:
            request_output = outputs[0]
            completion = request_output.outputs[0]
            text = str(completion.text)
            input_tokens = len(request_output.prompt_token_ids)
            output_tokens = len(completion.token_ids)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise TypeError(
                "Unexpected vLLM output; provider usage could not be read"
            ) from exc
        usd = cost_from_tokens(
            self.model, input_tokens, output_tokens, self.price_table_version
        )[2]
        return GenerationResult(
            text,
            input_tokens,
            output_tokens,
            latency_s,
            usd,
            self.model,
            self.model_revision,
        )
