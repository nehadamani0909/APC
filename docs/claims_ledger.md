# Resolved claims ledger

This release resolves claim status conservatively. A claim is not marked
supported when the available evidence is only deterministic offline fixture
data.

| # | Claim | Evidence | Release status |
|---:|---|---|---|
| 1 | Per-instance frontiers are heterogeneous beyond label noise | E1 | deferred: real corpus required |
| 2 | Curves are frequently non-monotone | E1b | deferred: real compressor/model required |
| 3 | Cheap features predict frontier position | E2 | deferred: real training required |
| 4 | Curve prediction beats classification | E4 | deferred: real training required |
| 5 | CRC attains target in-distribution risk | E5 | pipeline verified; scientific result deferred |
| 6 | OURS beats per-family fixed budget | E3 | deferred: real held-out evaluation required |
| 7 | OURS beats AdaComp-style point prediction | E3 | deferred: faithful trained baseline required |
| 8 | OURS beats threshold adaptation | E3 | deferred: real attention baseline required |
| 9 | Output expansion changes optimal budget | E9 | pipeline verified; scientific result deferred |
| 10 | Net cost and latency savings are positive | E10 | explicitly mixed; no blanket claim |
| 11 | Adherence correction improves control | E2c | deferred: real compressor measurements required |
| 12 | Policy transfers across target LLMs | E6a | deferred: API model runs required |
| 13 | Policy transfers across task families | E6b | deferred: held-out benchmark families required |
| 14 | Policy transfers across compression backends | E7 | deferred: real backend runs required |
| 15 | Shift behaviour is measured honestly | E6c | supported as a measurement protocol; guarantee under shift not claimed |
