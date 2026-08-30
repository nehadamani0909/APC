"""Guard: no scaffold placeholder may reach ``paper/``.

The report generators emit hardcoded placeholder numbers wrapped in genuine
BCa intervals, which read exactly like measurements.  ``paper/`` is where a
draft reaches for its tables and figures, so anything carrying the scaffold
banner appearing there is a fabrication hazard, not a style issue.
"""

from pathlib import Path

from frontier.eval.p10 import SCAFFOLD_BANNER, generate_reports

MARKER = "SCAFFOLD"
PAPER = Path(__file__).resolve().parents[1] / "paper"


def test_scaffold_banner_is_actually_emitted(tmp_path: Path) -> None:
    generate_reports(tmp_path)
    for name in ("t4.md", "t5.md", "t6.md", "t7.md", "t8.md", "t9.md",
                 "failures.md"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert SCAFFOLD_BANNER.splitlines()[0] in text, name


def test_paper_directory_contains_no_scaffold_output() -> None:
    offenders = [
        path
        for path in PAPER.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".svg", ".tex"}
        and MARKER in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not offenders, (
        "placeholder output found under paper/: "
        f"{[str(path) for path in offenders]}. These are not measurements; "
        "regenerate from a real corpus or delete them."
    )


def test_generators_do_not_default_to_the_paper_directory() -> None:
    import inspect

    signature = inspect.signature(generate_reports)
    default = signature.parameters["output_dir"].default
    assert "paper" not in str(default)
