# Reproduce

This release is reproducible with Python 3.11+ and uv:

```bash
uv sync
uv run python scripts/reproduce.py
```

The command validates the released parquet corpus, regenerates T3 from that
corpus, regenerates P9 figures and power output, and regenerates P10 tables and
failure analysis. Reports are intentionally ignored by Git and are rebuilt in
the working tree.

The checked-in corpus is the deterministic offline fixture pilot. Its reports
are software-pipeline smoke results; the model card and claims ledger state
which scientific claims remain deferred.
