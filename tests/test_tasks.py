from pathlib import Path

from frontier.harness.tasks import (
    LONGBENCH_SUBSETS,
    GSM8KTask,
    Instance,
    default_tasks,
)


def test_registry_and_separate_context_query() -> None:
    tasks = default_tasks()
    assert len(tasks) == 11
    assert {task.name for task in tasks[1:7]} == set(LONGBENCH_SUBSETS)
    for task in tasks:
        instances = task.load("test", n=5)
        assert len(instances) == 5
        assert all(isinstance(instance, Instance) for instance in instances)
        assert all(instance.context and instance.query for instance in instances)


def test_jsonl_records_are_loaded(tmp_path: Path) -> None:
    source = tmp_path / "task.jsonl"
    source.write_text(
        '{"id":"x","context":"ctx","query":"q","gold":"a","split":"test"}\n',
        encoding="utf-8",
    )
    task = GSM8KTask(source)
    assert task.load("test", n=5)[0].id == "x"


def test_gsm8k_parses_the_final_numeric_answer() -> None:
    task = GSM8KTask()
    # A chain-of-thought answer, which is what the model actually emits.
    raw = "Janet sells 16 - 3 - 4 = 9 eggs.\nShe makes 9 * $2 = $18 per day.\n#### 18"
    assert task.parse(raw) == "18"
    assert task.metric(task.parse(raw), "18") == 1.0
    # Without '####', fall back to the last number in the text.
    assert task.parse("So the total is 1,234 dollars.") == "1234"
    assert task.parse("The answer is -5.") == "-5"
    assert task.parse("no digits here") == "no digits here"


def test_gsm8k_prompt_continues_the_few_shot_pattern() -> None:
    prompt = GSM8KTask().build_prompt("Question: A?\nAnswer: 1", "B?")
    assert prompt.endswith("Question: B?\nAnswer:")


def test_gsm8k_parse_survives_a_preamble_before_the_first_blank_line() -> None:
    """Regression: an instruction-tuned model opens with a preamble sentence.

    Cutting the response at the first blank line discarded the derivation and
    left a digit-free fragment, so ``parse`` returned the raw text and every
    compressed budget scored 0.0 -- indistinguishable from a compression
    cliff, and wrong.
    """

    from frontier.harness.tasks import GSM8KTask

    raw = (
        "To find the total cost, we need to first calculate the discount.\n\n"
        "Discount = $20 x 0.3 = $6\n"
        "Discounted price = $20 - $6 = $14\n\n"
        "Total Cost = $14 x 4\n= $56"
    )
    assert GSM8KTask().parse(raw) == "56"


def test_gsm8k_parse_still_ignores_a_hallucinated_next_question() -> None:
    from frontier.harness.tasks import GSM8KTask

    raw = "The answer is 18.\n\nQuestion: unrelated\nAnswer: 99"
    assert GSM8KTask().parse(raw) == "18"
