# Cost ledger

Every LLM and compressor call appends one row (APC-04 §4.4). This is not
bookkeeping: it is the evidence for C2 and the net-efficiency analysis, and
it cannot be reconstructed afterwards.

## What every row records

`run_id`, `ts`, `phase`, `prompt_id`, `task`, `family`, `backend`,
`requested_b`, `realised_r`, `target_model`, `sample_idx`, `temperature`,
`seed`, `quality`, `T_in`, `T_out`, `latency_ms`, `compress_ms`, `usd_in`,
`usd_out`, `usd_total`, `gpu_seconds`, `raw_output_hash`, `code_version`
(including the model revision), `price_table_version`.

## Invariants enforced in code

- **USD reconciles to provider-reported usage within 1%**, checked per row by
  `validate_ledger` and again at grid time. Token counts always come from the
  provider's response, never from local re-tokenisation.
- **The price table is versioned**, and each row records the version it was
  billed under. Rows are added only from a provider's published rates — a
  guessed price would corrupt every USD figure, and the 1% check would not
  catch it, because it verifies the ledger against the table, not the table
  against the invoice.
- **Writes are lock-guarded.** On Windows concurrent appends are not atomic,
  so parallel grid workers would otherwise interleave.
- **A local model is zero-priced.** Its compute is recorded as wall time in
  `latency_ms` (generation) and `compress_ms` (compression); `gpu_seconds` is
  GPU time specifically, and is legitimately 0 for a CPU run. Analyses cost
  such a corpus at an explicit price row, which is stated in the report.

## Spend control

`SpendCap` reserves an estimate before each API call and settles against
actual usage afterwards. Exceeding it raises `SpendCapExceeded`, which is
deliberately not retried — retrying a budget stop only burns the remaining
allowance.

## Totals

Populated from a real run. The committed fixture corpus records 0 USD and
0 GPU-hours and is not a measurement.
