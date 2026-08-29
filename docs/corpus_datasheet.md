# Compression Frontier Corpus datasheet

## Motivation

To be completed before releasing corpus v1; the corpus measures per-instance
prompt-compression frontiers, realised rates, quality, output expansion, and
cost.

## Composition

- Source datasets and task families: pending real-data collection.
- Prompt/document counts, models, backends, budgets, and generations: generated
  in reproducible table T2.

## Collection process

Record model IDs/revisions, compressor versions, target tokenizer versions,
temperatures, seeds, price-table version, and API spend cap.

## Preprocessing

Keep context and query separate. The query is never compressed. Record source
document IDs so split leakage can be validated.

## Uses

Measurement and training of risk-controlled instance-adaptive compression
policies after Gate 1.

## Limitations

The current repository contains an offline fixture pilot, not a released
scientific corpus. Distribution shift, benchmark licensing, model revisions,
and sampling noise must be documented for corpus v1.
