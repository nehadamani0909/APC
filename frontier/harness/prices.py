"""Versioned model pricing used by the cost ledger."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million input and output tokens."""

    input_usd_per_million: float
    output_usd_per_million: float


PRICE_TABLE_VERSION = "v1"

# Prices are explicit data, rather than hidden constants in ledger code.  The
# local entry is zero-priced because local inference has no provider token
# bill; its compute cost is accounted separately as ``gpu_seconds``.
#
# Rows are only ever added from a provider's published price list, with the
# table version bumped when they change.  A guessed price would silently
# corrupt every USD figure in the paper, and the ledger's 1% reconciliation
# would not catch it -- it checks that the ledger agrees with THIS table, not
# that this table agrees with the invoice.  Register additional models
# (Anthropic, other OpenAI rows) with ``register_price`` at configuration
# time; see RUNBOOK.md.
PRICE_TABLES: dict[str, dict[str, ModelPrice]] = {
    "v1": {
        "local-default": ModelPrice(0.0, 0.0),
        "gpt-4o-mini": ModelPrice(0.15, 0.60),
        "gpt-4o": ModelPrice(2.50, 10.00),
    }
}


def register_price(
    model: str,
    price: ModelPrice,
    version: str = PRICE_TABLE_VERSION,
    *,
    overwrite: bool = False,
) -> None:
    """Add a model's published prices to a price table.

    Raises rather than silently replacing an existing row: a run whose costs
    changed underneath it is not reproducible, and the version should be
    bumped instead.
    """

    table = PRICE_TABLES.setdefault(version, {})
    if model in table and not overwrite and table[model] != price:
        raise ValueError(
            f"{model!r} already priced in table {version}; bump the table "
            "version rather than redefining a row mid-project"
        )
    table[model] = price


def get_price(model: str, version: str = PRICE_TABLE_VERSION) -> ModelPrice:
    """Return the immutable price entry for *model* and *version*."""

    try:
        table = PRICE_TABLES[version]
    except KeyError as exc:
        raise ValueError(f"Unknown price table version: {version}") from exc
    try:
        return table[model]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model!r} in price table {version}. Add it with "
            "register_price() using the provider's published rates; prices "
            "are never inferred."
        ) from exc


def cost_from_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int,
    version: str = PRICE_TABLE_VERSION,
) -> tuple[float, float, float]:
    """Calculate input, output, and total USD cost from provider token counts."""

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    price = get_price(model, version)
    usd_in = input_tokens * price.input_usd_per_million / 1_000_000
    usd_out = output_tokens * price.output_usd_per_million / 1_000_000
    return usd_in, usd_out, usd_in + usd_out
