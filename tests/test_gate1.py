import pandas as pd

from frontier.eval.gate1 import BUDGETS, _budget_label, _noise_floor


def test_monotone_safe_label_rejects_an_unsafe_middle_budget() -> None:
    values = {rate: 1.0 for rate in BUDGETS}
    values[0.2] = 1.0
    values[0.3] = 0.0
    assert _budget_label(values, 0.05, naive=True) == 0.2
    assert _budget_label(values, 0.05, naive=False) == 0.4


def test_noise_floor_uses_split_halves() -> None:
    records = []
    for sample_idx in range(5):
        for rate in BUDGETS:
            records.append(
                {
                    "prompt_id": "p",
                    "family": "qa",
                    "requested_b": rate,
                    "sample_idx": sample_idx,
                    "quality": 1.0,
                }
            )
    frame = pd.DataFrame(records)
    assert _noise_floor(frame, 0.05) == 0.0
