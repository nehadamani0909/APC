from pathlib import Path

from frontier.eval.p10 import generate_reports


def test_p10_generates_required_tables_and_failure_examples(tmp_path: Path) -> None:
    generate_reports(tmp_path)
    for table in ("t4.md", "t5.md", "t6.md", "t7.md", "t8.md", "t9.md"):
        assert (tmp_path / table).exists()
        assert "95% BCa CI" in (tmp_path / table).read_text(encoding="utf-8")
    failures = (tmp_path / "failures.md").read_text(encoding="utf-8")
    assert failures.count("### Worked example") == 18
    assert failures.count("Original context:") == 18
    assert failures.count("Compressed context:") == 18
    assert failures.count("Model output:") == 18
