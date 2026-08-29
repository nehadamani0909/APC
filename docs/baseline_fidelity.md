# Baseline fidelity notes

P8 exposes every baseline through the frozen `Policy.select(x, q, task_family)`
contract and the evaluator calls that method identically for every row. The
current repository is an offline, deterministic scaffold; no baseline is
silently presented as a paper reproduction.

| ID | Implementation and explicit adaptation |
|---|---|
| B0 | Fixed `b=1.0`, the uncompressed reference. |
| B1 | One policy per budget in the frozen budget grid. |
| B2a/B2b/B2c | Fixed-budget shells. They currently use the fixture default `b=0.5`; validation tuning by global, family, or family×model is deferred until corpus labels exist. |
| B3 | Context-length heuristic mapped to the nearest grid rate. |
| B3b | Gzip compressed-byte ratio as the documented free redundancy statistic. |
| B4 | Seeded random budget; rate-matched analysis belongs in the evaluator. |
| B5a | AdaComp-style point estimate adapted from document/top-k granularity to token-rate grid selection; the current estimator is transparent and label-free. |
| B5b | Adaptive QuerySelect-style query-aware variable rate, adapted to lexical query/context relevance because no retriever or target model is available in the offline scaffold. |
| B5c | AttnComp-style threshold policy; attention mass is replaced by the documented lexical relevance statistic until an attention provider is injected. |
| B6 | Seeded random operating point representing random-token-drop rate; actual token deletion remains the compressor backend's responsibility. |
| B7 | Model-routing decision represented as `b=1.0`; the frozen Selection schema has no model identifier, so model assignment must be attached by the deployment evaluator. |
| B8 | Fixed operating point used for cache-hit sensitivity sweeps; cache accounting is evaluator-side. |
| ORACLE | Injected true held-out `b*`, always labelled `(upper bound)`. |
| ORACLE-noisy | Injected noisy `b*` estimate, always labelled `(upper bound)`. |
| OURS | Uses the P7 CRC policy when an instance predictor is supplied. |

AdaComp's original document-level knob and Adaptive QuerySelect's original
query-aware variable-rate idea cannot be reproduced byte-for-byte without
their training data and model artifacts; the token-grid and lexical-statistic
adaptations above are explicit and are not claimed as faithful neural replicas.
