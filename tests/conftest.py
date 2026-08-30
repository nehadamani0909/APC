"""Shared pytest configuration.

Integration tests reach the network, a provider API, or model weights, so
they are skipped unless explicitly enabled.  This keeps the default suite
offline and deterministic (ground rule 3) while letting the same tests run
for real on a machine that has credentials and data.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest

ENABLE_ENV_VAR = "FRONTIER_RUN_INTEGRATION"


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    if os.environ.get(ENABLE_ENV_VAR) == "1":
        return
    skip_integration = pytest.mark.skip(
        reason=f"integration test; set {ENABLE_ENV_VAR}=1 to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
