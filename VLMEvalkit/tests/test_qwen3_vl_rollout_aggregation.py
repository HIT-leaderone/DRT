from vlmeval.vlm.qwen3_vl.aggregation import (
    extract_answer,
    normalize_answer,
    parse_selector_index,
    select_self_consistent,
)


def test_extract_answer_prefers_answer_tag():
    text = "<think>work</think><answer>F. friend</answer>"
    assert extract_answer(text) == "F. friend"
    assert normalize_answer(extract_answer(text)) == "F"


def test_self_consistency_plurality_tie_breaks_by_first_occurrence():
    candidates = [
        "<answer>B</answer>",
        "<answer>A</answer>",
        "<answer>A. option text</answer>",
        "<answer>B</answer>",
    ]
    selection = select_self_consistent(candidates)
    assert selection["selected_index"] == 0
    assert selection["vote_counts"] == {"B": 2, "A": 2}


def test_self_consistency_selects_plurality():
    candidates = [
        "<answer>18</answer>",
        "Final answer: 19",
        "\\boxed{18}",
    ]
    selection = select_self_consistent(candidates)
    assert selection["selected_index"] == 0
    assert selection["normalized_answer"] == "18"


def test_parse_selector_index():
    assert parse_selector_index("<answer>2</answer>", 4) == 1
    assert parse_selector_index("I choose Candidate 3.", 4) == 2
    assert parse_selector_index("<answer>D</answer>", 4) == 3
    assert parse_selector_index("<answer>9</answer>", 4) is None
