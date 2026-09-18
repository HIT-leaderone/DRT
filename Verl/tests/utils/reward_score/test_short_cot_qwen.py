import asyncio
import json

import pytest

from verl.utils.reward_score import short_cot_qwen


def test_count_reasoning_steps_filters_short_segments():
    think = "a -> b -> observe circle relation clearly -> use midpoint property carefully -> derive exterior-angle equation cleanly"
    assert short_cot_qwen.count_reasoning_steps(think, min_tokens_per_step=4) == 3


def test_get_deep_exploration_difficulty_schedule_by_source():
    mapping = {
        "geo3k": {"threshold": 7, "cap": 11},
        "dolci_math": {"threshold": 9, "cap": 16},
    }

    assert short_cot_qwen.get_deep_exploration_difficulty_schedule(
        data_source="geo3k",
        extra_info={"dataset": "geo3k"},
        difficulty_by_source=mapping,
        default_threshold=9,
        default_cap=16,
    ) == (7, 11, "geo3k")

    assert short_cot_qwen.get_deep_exploration_difficulty_schedule(
        data_source="unknown_source",
        extra_info={"dataset": "dolci_math"},
        difficulty_by_source=mapping,
        default_threshold=9,
        default_cap=16,
    ) == (9, 16, "dolci_math")

    assert short_cot_qwen.get_deep_exploration_difficulty_schedule(
        data_source="custom",
        extra_info={},
        difficulty_by_source=mapping,
        default_threshold=9,
        default_cap=16,
    ) == (9, 16, "custom")


def test_evaluate_step_bonus_with_gpt_parses_new_fields(monkeypatch):
    async def fake_generate_chat_aiohttp(*args, **kwargs):
        return json.dumps(
            {
                "matched_step_count": 5,
                "total_reference_steps": 7,
                "effective_reasoning_step_count": 6,
                "deep_exploration_step_count": 2,
                "hallucinated_matched_step_count": 2,
                "has_hallucination_in_matched_steps": True,
            }
        )

    monkeypatch.setattr(short_cot_qwen, "generate_chat_aiohttp", fake_generate_chat_aiohttp)

    result = asyncio.run(
        short_cot_qwen.evaluate_step_bonus_with_gpt(
            pred_think="step one -> step two",
            pred_final="18",
            gt_steps=[f"step {i}" for i in range(7)],
            router_address="127.0.0.1:8000",
            problem_text="dummy",
        )
    )

    assert result["matched_step_count"] == 5
    assert result["total_reference_steps"] == 7
    assert result["effective_reasoning_step_count"] == 6
    assert result["deep_exploration_step_count"] == 2
    assert result["hallucinated_matched_step_count"] == 2
    assert result["has_hallucination_in_matched_steps"] is True


def test_evaluate_step_bonus_with_gpt_guided_json_passes_schema(monkeypatch):
    captured = {}

    async def fake_generate_chat_aiohttp(*args, **kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "matched_step_count": 5,
                "total_reference_steps": 7,
                "effective_reasoning_step_count": 6,
                "deep_exploration_step_count": 2,
                "hallucinated_matched_step_count": 2,
                "has_hallucination_in_matched_steps": True,
                "reason": "ok",
            }
        )

    monkeypatch.setattr(short_cot_qwen, "generate_chat_aiohttp", fake_generate_chat_aiohttp)

    result = asyncio.run(
        short_cot_qwen.evaluate_step_bonus_with_gpt_guided_json(
            pred_think="step one -> step two",
            pred_final="18",
            gt_steps=[f"step {i}" for i in range(7)],
            router_address="127.0.0.1:8000",
            reward_model_name="/tmp/custom-reward-model",
            problem_text="dummy",
        )
    )

    assert result["matched_step_count"] == 5
    assert captured["guided_json"] == short_cot_qwen.STEP_BONUS_GRADER_JSON_SCHEMA
    assert captured["model"] == "/tmp/custom-reward-model"


def test_gpt_step_bonus_reward_uses_hallucinated_matched_step_ratio(monkeypatch):
    async def no_hallucination(*args, **kwargs):
        return {
            "matched_step_count": 4,
            "total_reference_steps": 5,
            "effective_reasoning_step_count": 5,
            "deep_exploration_step_count": 0,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    async def with_hallucination(*args, **kwargs):
        return {
            "matched_step_count": 4,
            "total_reference_steps": 5,
            "effective_reasoning_step_count": 5,
            "deep_exploration_step_count": 0,
            "hallucinated_matched_step_count": 1,
            "has_hallucination_in_matched_steps": True,
        }

    solution = "<think>observe carefully -> derive relation -> simplify equation -> finish answer</think><answer>17</answer>"
    ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(5)]})
    completion_ratio = 4 / 5
    expected_main_reward = (completion_ratio ** 1.5) * 0.8
    expected_hallucination_ratio = 1 / 4

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", no_hallucination)
    score_without_hallucination = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            score=0.9,
            step_bonus_lambda=0.2,
            hallucination_penalty_ratio=0.5,
        )
    )

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", with_hallucination)
    score_with_hallucination = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            score=0.9,
            step_bonus_lambda=0.2,
            hallucination_penalty_ratio=0.5,
        )
    )

    assert score_without_hallucination - score_with_hallucination == pytest.approx(
        expected_main_reward * 0.5 * expected_hallucination_ratio
    )


def test_gpt_step_bonus_reward_adds_deep_exploration_bonus_for_hard_cases(monkeypatch):
    async def easy_case(*args, **kwargs):
        return {
            "matched_step_count": 10,
            "total_reference_steps": 10,
            "effective_reasoning_step_count": 10,
            "deep_exploration_step_count": 2,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    async def hard_case(*args, **kwargs):
        return {
            "matched_step_count": 18,
            "total_reference_steps": 18,
            "effective_reasoning_step_count": 18,
            "deep_exploration_step_count": 3,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    solution = "<think>reason carefully -> verify a branch -> compare alternatives -> finish</think><answer>18</answer>"
    easy_ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(10)]})
    hard_ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(18)]})

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", easy_case)
    easy_score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=easy_ground_truth,
            step_bonus_lambda=0.2,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            deep_exploration_difficulty_threshold=9,
            deep_exploration_difficulty_cap=18,
        )
    )

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", hard_case)
    hard_score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=hard_ground_truth,
            step_bonus_lambda=0.2,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            deep_exploration_difficulty_threshold=9,
            deep_exploration_difficulty_cap=18,
        )
    )

    assert hard_score > easy_score


def test_gpt_step_bonus_reward_is_not_gated_by_difficulty_for_effective_reasoning(monkeypatch):
    async def same_judge(*args, **kwargs):
        return {
            "matched_step_count": 6,
            "total_reference_steps": 6,
            "effective_reasoning_step_count": 6,
            "deep_exploration_step_count": 0,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", same_judge)

    solution = "<think>reason carefully -> verify relation -> simplify -> finish</think><answer>18</answer>"
    ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(6)]})

    score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            step_bonus_lambda=0.3,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            deep_exploration_difficulty_threshold=9,
            deep_exploration_difficulty_cap=18,
        )
    )

    assert score == pytest.approx(0.9 + 0.3 + 0.1)


def test_gpt_step_bonus_reward_treats_full_completion_as_correct(monkeypatch):
    async def same_judge(*args, **kwargs):
        return {
            "matched_step_count": 5,
            "total_reference_steps": 5,
            "effective_reasoning_step_count": 5,
            "deep_exploration_step_count": 0,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", same_judge)

    solution = "<think>reason carefully -> verify relation -> simplify -> finish</think><answer>17</answer>"
    ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(5)]})

    score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            step_bonus_lambda=0.3,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
        )
    )

    assert score == pytest.approx(0.9 + 0.3 + 0.1)


def test_gpt_step_bonus_reward_uses_source_specific_difficulty_schedule(monkeypatch):
    async def same_judge(*args, **kwargs):
        return {
            "matched_step_count": 10,
            "total_reference_steps": 10,
            "effective_reasoning_step_count": 10,
            "deep_exploration_step_count": 2,
            "hallucinated_matched_step_count": 0,
            "has_hallucination_in_matched_steps": False,
        }

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt", same_judge)

    solution = "<think>reason carefully -> verify a branch -> compare alternatives -> finish</think><answer>18</answer>"
    ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(10)]})
    difficulty_by_source = {
        "geo3k": {"threshold": 7, "cap": 11},
        "dolci_math": {"threshold": 9, "cap": 16},
    }

    geo_score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            step_bonus_lambda=0.2,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            deep_exploration_difficulty_threshold=9,
            deep_exploration_difficulty_cap=16,
            deep_exploration_difficulty_by_source=difficulty_by_source,
        )
    )

    dolci_score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus(
            data_source="dolci_math",
            solution_str=solution,
            ground_truth=ground_truth,
            step_bonus_lambda=0.2,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            deep_exploration_difficulty_threshold=9,
            deep_exploration_difficulty_cap=16,
            deep_exploration_difficulty_by_source=difficulty_by_source,
        )
    )

    assert geo_score > dolci_score


def test_gpt_step_bonus_guided_json_reward_matches_base_logic(monkeypatch):
    async def same_judge(*args, **kwargs):
        return {
            "matched_step_count": 4,
            "total_reference_steps": 5,
            "effective_reasoning_step_count": 5,
            "deep_exploration_step_count": 1,
            "hallucinated_matched_step_count": 1,
            "has_hallucination_in_matched_steps": True,
        }

    monkeypatch.setattr(short_cot_qwen, "evaluate_step_bonus_with_gpt_guided_json", same_judge)

    solution = "<think>observe carefully -> derive relation -> simplify equation -> finish answer</think><answer>17</answer>"
    ground_truth = json.dumps({"answer": "18", "steps": [f"step {i}" for i in range(5)]})

    score = asyncio.run(
        short_cot_qwen.compute_score_qwen_gpt_step_bonus_guided_json(
            data_source="geo3k",
            solution_str=solution,
            ground_truth=ground_truth,
            score=0.9,
            step_bonus_lambda=0.2,
            deep_exploration_bonus_lambda=0.12,
            deep_exploration_step_cap=3,
            hallucination_penalty_ratio=0.5,
        )
    )

    completion_ratio = 4 / 5
    main_reward = (completion_ratio ** 1.5) * 0.8
    hallucination_ratio = 1 / 4
    expected = (
        0.1
        + main_reward * (1 - 0.5 * hallucination_ratio)
        + 0.2 * completion_ratio * 1.0
        + 0.12 * completion_ratio * (0.0) * (1 / 3)
    )
    assert score == pytest.approx(expected)
