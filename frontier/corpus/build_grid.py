"""Resumable, idempotent pilot grid runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from frontier.compress.base import Compressor
from frontier.harness.ledger import GridRow, Ledger, read_ledger
from frontier.harness.models import TargetLLM
from frontier.harness.outputs import OutputStore
from frontier.harness.prices import PRICE_TABLE_VERSION, cost_from_tokens
from frontier.harness.tasks import Instance, Task


def _canonical_rate(rate: float) -> str:
    return f"{float(rate):.8f}"


@dataclass(frozen=True)
class GridCell:
    instance: Instance
    task: Task
    backend: Compressor
    requested_b: float
    target: TargetLLM
    sample_idx: int
    temperature: float
    seed: int

    @property
    def key(self) -> str:
        material = "|".join(
            (
                self.instance.id,
                self.backend.name,
                _canonical_rate(self.requested_b),
                self.target.model,
                str(self.sample_idx),
            )
        )
        return hashlib.sha256(material.encode()).hexdigest()


class GridRunner:
    """Execute cells once, recording a completion index after each row."""

    def __init__(
        self,
        ledger: Ledger,
        completion_index: str | Path,
        failure_index: str | Path | None = None,
        output_store: OutputStore | None = None,
    ) -> None:
        self.ledger = ledger
        # The generated text is not in the ledger row (APC-04 §8 stores only
        # its hash), so answer preservation -- the secondary quality
        # definition -- is uncomputable unless it is kept here.
        self.output_store = output_store
        self.completion_index = Path(completion_index)
        self.failure_index = Path(failure_index) if failure_index is not None else None
        self._write_lock = Lock()
        self.completed = self._read_completed()

    def _read_completed(self) -> set[str]:
        keys: set[str] = set()
        if self.completion_index.exists():
            with self.completion_index.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        keys.add(str(json.loads(line)["key"]))
        if self.ledger.path.exists():
            frame = read_ledger(self.ledger.path)
            for _, row in frame.iterrows():
                material = "|".join(
                    (
                        str(row["prompt_id"]),
                        str(row["backend"]),
                        _canonical_rate(row["requested_b"]),
                        str(row["target_model"]),
                        str(row["sample_idx"]),
                    )
                )
                keys.add(hashlib.sha256(material.encode()).hexdigest())
        return keys

    def _mark_completed(self, cell: GridCell) -> None:
        self.completion_index.parent.mkdir(parents=True, exist_ok=True)
        with self.completion_index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": cell.key}) + "\n")
        self.completed.add(cell.key)

    def _mark_failed(self, cell: GridCell, error: Exception) -> None:
        if self.failure_index is None:
            return
        with self._write_lock:
            self.failure_index.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_index.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "key": cell.key,
                            "prompt_id": cell.instance.id,
                            "backend": cell.backend.name,
                            "requested_b": cell.requested_b,
                            "target_model": cell.target.model,
                            "sample_idx": cell.sample_idx,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    + "\n"
                )

    def _build_row(self, cell: GridCell) -> GridRow:
        """Execute one cell. Every failure mode raises rather than aborting."""

        compressed = cell.backend.compress(
            cell.instance.context, cell.instance.query, cell.requested_b
        )
        prompt = cell.task.build_prompt(
            compressed.compressed_text, cell.instance.query
        )
        generation = cell.target.generate(
            prompt, temperature=cell.temperature, seed=cell.seed
        )
        # Reconcile against the price table the target actually billed with,
        # not the module default: an adapter constructed with a pinned older
        # version would otherwise be compared against the wrong prices.
        price_table_version = getattr(
            cell.target, "price_table_version", PRICE_TABLE_VERSION
        )
        usd_in, usd_out, usd_total = cost_from_tokens(
            cell.target.model,
            generation.T_in,
            generation.T_out,
            price_table_version,
        )
        if abs(usd_total - generation.usd) > max(1e-9, usd_total) * 0.01:
            raise ValueError(
                f"target usage cost does not reconcile: ledger={usd_total} "
                f"provider={generation.usd}"
            )
        digest = hashlib.sha256(generation.text.encode()).hexdigest()
        if self.output_store is not None:
            self.output_store.put(digest, generation.text)
        return GridRow(
            prompt_id=cell.instance.id,
            task=cell.task.name,
            family=cell.task.family,
            backend=cell.backend.name,
            requested_b=cell.requested_b,
            realised_r=compressed.realised_rate,
            target_model=cell.target.model,
            sample_idx=cell.sample_idx,
            temperature=cell.temperature,
            seed=cell.seed,
            quality=cell.task.metric(
                cell.task.parse(generation.text), cell.instance.gold
            ),
            T_in=generation.T_in,
            T_out=generation.T_out,
            latency_ms=generation.latency_s * 1000.0,
            compress_ms=compressed.wall_ms,
            usd_in=usd_in,
            usd_out=usd_out,
            usd_total=usd_total,
            gpu_seconds=compressed.gpu_ms / 1000.0,
            raw_output_hash=digest,
            code_version=f"p3;model_revision={cell.target.model_revision}",
            price_table_version=price_table_version,
        )

    def run(self, cells: Iterable[GridCell]) -> int:
        executed = 0
        for cell in cells:
            if cell.key in self.completed:
                continue
            # The guarded region deliberately covers cost reconciliation,
            # metric evaluation, and GridRow validation as well as the model
            # calls. Each of those can raise on a single bad cell, and a grid
            # of this size must record the failure and keep going.
            try:
                row = self._build_row(cell)
            except Exception as error:
                self._mark_failed(cell, error)
                continue
            # Appending the row and marking the cell complete must be atomic
            # with respect to other workers: on Windows, concurrent appends
            # to the same file are not guaranteed not to interleave.
            with self._write_lock:
                self.ledger.append(row)
                self._mark_completed(cell)
            executed += 1
        return executed

    def run_parallel(self, cells: Iterable[GridCell], workers: int = 1) -> int:
        """Run independent cells concurrently while preserving idempotence."""
        if workers <= 1:
            return self.run(cells)
        pending = [cell for cell in cells if cell.key not in self.completed]
        if not pending:
            return 0
        # One future per chunk rather than per cell; a full grid is hundreds
        # of thousands of cells and one future each is pure overhead.
        chunk_size = max(1, (len(pending) + workers - 1) // workers)
        chunks = [
            pending[start : start + chunk_size]
            for start in range(0, len(pending), chunk_size)
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.run, chunk) for chunk in chunks]
            return sum(future.result() for future in as_completed(futures))


def cell_count(
    instances: Sequence[Instance],
    rates: Sequence[float],
    backends: Sequence[Compressor],
    targets: Sequence[TargetLLM],
    samples: int,
) -> int:
    return len(instances) * len(rates) * len(backends) * len(targets) * samples


def dry_run(
    instances: Sequence[Instance],
    rates: Sequence[float],
    backends: Sequence[Compressor],
    targets: Sequence[TargetLLM],
    samples: int,
    *,
    estimated_input_tokens: int = 1_000,
    estimated_output_tokens: int = 100,
    estimated_gpu_seconds_per_cell: float = 0.5,
) -> str:
    count = cell_count(instances, rates, backends, targets, samples)
    cost = sum(
        cost_from_tokens(target.model, estimated_input_tokens, estimated_output_tokens)[
            2
        ]
        for target in targets
    )
    cost *= len(instances) * len(rates) * len(backends) * samples
    gpu_hours = count * estimated_gpu_seconds_per_cell / 3600.0
    return (
        f"cells: {count}\nestimated_usd: {cost:.6f}\n"
        f"estimated_gpu_hours: {gpu_hours:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cells", type=int, default=200 * 7 * 5)
    args = parser.parse_args()
    if args.dry_run:
        print(
            f"cells: {args.cells}\nestimated_usd: unavailable\n"
            "estimated_gpu_hours: unavailable"
        )
    else:
        parser.error("use scripts/00_pilot.py for the configured pilot")


if __name__ == "__main__":
    main()
