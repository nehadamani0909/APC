from pathlib import Path


def test_release_artifacts_exist() -> None:
    assert Path("LICENSE").is_file()
    assert Path("REPRODUCE.md").is_file()
    assert Path("artifacts/predictor.json").is_file()
    assert Path("artifacts/calibration.json").is_file()
    assert Path("docs/model_card.md").is_file()
    assert Path("docs/claims_ledger.md").is_file()
    assert Path("docs/cost_ledger_summary.md").is_file()
