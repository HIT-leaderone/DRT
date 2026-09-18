from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
OPTION_RE = re.compile(r"^\s*(?:option\s*)?[\(\[]?([A-Z])[\)\].:：]?(?:\s|$)", re.IGNORECASE)
INTEGER_RE = re.compile(r"-?\d+")


def env_flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def extract_answer(text: Any) -> str:
    """Extract a compact final answer from a DRT/CoT response.

    The function is intentionally dataset-agnostic: it prefers explicit
    ``<answer>`` tags, then ``\\boxed{}``, then common final-answer markers,
    and finally falls back to the full text.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    matches = ANSWER_TAG_RE.findall(raw)
    if matches:
        return matches[-1].strip()

    boxed = BOXED_RE.findall(raw)
    if boxed:
        return boxed[-1].strip()

    without_think = raw.split("</think>")[-1].strip() if "</think>" in raw else raw
    marker_patterns = [
        r"(?:final\s+answer|answer|therefore|so)\s*(?:is|:|=)\s*([^\n]+)",
        r"答案\s*(?:是|为|:|：)\s*([^\n]+)",
    ]
    for pattern in marker_patterns:
        marker_matches = re.findall(pattern, without_think, flags=re.IGNORECASE)
        if marker_matches:
            return marker_matches[-1].strip()

    lines = [line.strip() for line in without_think.splitlines() if line.strip()]
    return lines[-1] if lines else without_think


def normalize_answer(answer: Any) -> str:
    """Normalize answers for self-consistency voting."""
    text = "" if answer is None else str(answer)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip()
    text = text.strip("`*_ \t\r\n")
    text = text.strip("。.!！?？,，;；:")

    option = OPTION_RE.match(text)
    if option:
        return option.group(1).upper()

    lower = re.sub(r"\s+", " ", text.lower()).strip()
    if lower in {"yes", "y", "true"}:
        return "yes"
    if lower in {"no", "n", "false"}:
        return "no"

    numeric = lower.replace(",", "")
    numeric = numeric.strip("$% ")
    if re.fullmatch(r"-?\d+(?:\.0+)?", numeric):
        return str(int(float(numeric)))
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)", numeric):
        return numeric.rstrip("0").rstrip(".")

    # Keep simple algebraic/text answers comparable while avoiding aggressive
    # transformations that can merge genuinely different free-form answers.
    return lower.strip("。.!！?？,，;；:")


def select_self_consistent(candidates: list[str]) -> dict[str, Any]:
    answers = [extract_answer(candidate) for candidate in candidates]
    keys = [normalize_answer(answer) for answer in answers]
    counts = Counter(keys)

    first_index: dict[str, int] = {}
    for idx, key in enumerate(keys):
        first_index.setdefault(key, idx)

    if counts:
        winner_key = max(counts, key=lambda key: (counts[key], -first_index[key]))
        selected_index = first_index[winner_key]
    else:
        winner_key = ""
        selected_index = 0

    return {
        "selected_index": selected_index,
        "selected_answer": answers[selected_index] if answers else "",
        "normalized_answer": winner_key,
        "candidate_answers": answers,
        "normalized_answers": keys,
        "vote_counts": dict(counts),
    }


def build_bon_selector_text(
    original_text: str,
    candidates: list[str],
    *,
    candidate_max_chars: int = 4096,
) -> str:
    candidate_blocks = []
    for idx, candidate in enumerate(candidates, start=1):
        answer = extract_answer(candidate)
        candidate_blocks.append(
            f"Candidate {idx}:\n"
            f"Extracted final answer: {answer}\n"
            f"Full response:\n{truncate_middle(str(candidate), candidate_max_chars)}"
        )

    return (
        "You are selecting the best answer among multiple independent DRT rollouts.\n"
        "Use the original question, visual evidence, and candidate reasoning. "
        "Choose the candidate most likely to be correct.\n\n"
        f"Original text prompt:\n{truncate_middle(original_text, 4096)}\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nReturn only the selected candidate number in this exact format:\n"
        "<answer>1</answer>"
    )


def parse_selector_index(selector_output: Any, n_candidates: int) -> int | None:
    if n_candidates <= 0:
        return None
    answer = extract_answer(selector_output)
    haystack = f"{answer}\n{selector_output}"

    for value in INTEGER_RE.findall(haystack):
        idx = int(value) - 1
        if 0 <= idx < n_candidates:
            return idx

    option = OPTION_RE.match(answer)
    if option:
        idx = ord(option.group(1).upper()) - ord("A")
        if 0 <= idx < n_candidates:
            return idx
    return None


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
