import argparse
import ast
import asyncio
import importlib.util
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_OUTPUT_DIR = Path("./fig")
DEFAULT_ROUTER_JUDGE_MODEL = "./Qwen3-235B-A22B-Instruct-2507"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.1-2025-11-13"
ANALYSIS_VERSION = "visual_hallucination_v1"

MODEL_PLOT_ORDER = [
    "Qwen-Standard",
    "DRT-SFT-800it",
    "w/o process reward",
    "w/o step bouns",
    "DRT-RL",
]


VISUAL_HALLUCINATION_PROMPT = """
You are a strict hallucination judge for a visual math benchmark.

Task:
Evaluate the student's whole reasoning trace and separately identify:
1. VISUAL hallucinations.
2. REASONING hallucinations.

Use the Original Problem and Reference GT Steps as the supported fact set. If a visual fact is not explicit in the
problem text or not supported by the reference steps, treat it as unsupported. Do not reward a statement merely
because it sounds plausible from a drawing.

VISUAL hallucination definition:
- A claim about the image, diagram, chart, graph, table, object layout, labels, colors, counts, measurements, or
  geometric properties that is not supported by the Original Problem or Reference GT Steps.
- This includes unverified quantities inferred from the image and then used as implicit premises.
- Examples: inventing or reading off a length/angle/count/value not given; assuming a rough drawing is to scale;
  assuming parallel, perpendicular, equal, collinear, tangent, cyclic, midpoint, symmetry, orientation, or spatial
  relations from appearance only; misreading labels; fabricating objects, regions, axes, chart values, or table cells.

REASONING hallucination definition:
- Unsupported conditions, relations, or intermediate values introduced to bridge gaps in deduction.
- These are not primarily claims about what is visually shown. They are deductive assumptions or invented math facts
  used inside the reasoning chain.
- Examples: introducing an equation, variable value, congruence, similarity, formula condition, case condition,
  divisibility fact, algebraic relation, or intermediate result that is neither given nor validly derived.

Strict distinction:
- If the unsupported premise is presented as being observed/read/inferred from the image, count it as VISUAL.
- If the unsupported premise is introduced as a deductive bridge without relying on image observation, count it as
  REASONING.
- If the same unsupported statement is both visual and deductive, classify it by its source. A claimed image readout
  such as "from the diagram, x = 6" is VISUAL, even if later algebra uses it.

Do NOT count:
- A wrong final answer by itself.
- Restating explicit givens from the problem.
- A valid fact derived from explicit givens/reference facts.
- Harmless uncertainty or a rejected hypothesis that is not used as a premise.
- Verbose filler that introduces no unsupported premise.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Visual/Reasoning Trace:
{pred_reasoning}

Student Final Answer:
{pred_final}

Output strict JSON only:
- Return only one JSON object. Do not use markdown fences.
- Each `quote` must be a single-line substring with internal newlines collapsed to spaces.
- Keep each `quote` under 240 characters.
- Include at most 5 representative span objects per hallucination category. The count fields should still count all
  hallucinations you identify, even if the span list is capped.
{{
  "visual_hallucination_present": false,
  "visual_hallucination_count": 0,
  "visual_hallucination_spans": [
    {{
      "quote": "verbatim unsupported visual claim",
      "type": "unverified_quantity|implicit_visual_premise|fabricated_visual_object|unsupported_spatial_relation|unsupported_visual_attribute|misread_label|other",
      "explanation": "why this is a visual hallucination"
    }}
  ],
  "reasoning_hallucination_present": false,
  "reasoning_hallucination_count": 0,
  "reasoning_hallucination_spans": [
    {{
      "quote": "verbatim unsupported reasoning claim",
      "type": "unsupported_condition|unsupported_relation|unsupported_intermediate_value|deductive_gap_bridge|other",
      "explanation": "why this is a reasoning hallucination rather than a visual hallucination"
    }}
  ],
  "reason": "brief explanation of the classification boundary for this sample"
}}
"""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    prediction_path: Path
    extract_pkl_path: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    gt_parquet: Path
    model_specs: List[ModelSpec]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_verl_root() -> Path:
    candidates = []
    for env_name in ("VERL_ROOT", "VERL_QWEN3VL_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))

    candidates.extend(
        [
            Path("."),
            Path("."),
        ]
    )

    for candidate in candidates:
        short_cot_path = candidate / "verl/utils/reward_score/short_cot_qwen.py"
        if short_cot_path.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find VERL_Qwen3VL root. Set VERL_ROOT or VERL_QWEN3VL_ROOT to a tree containing "
        "verl/utils/reward_score/short_cot_qwen.py."
    )


VERL_ROOT = resolve_verl_root()
short_cot_qwen = _load_module_from_file(
    "short_cot_qwen_local",
    VERL_ROOT / "verl/utils/reward_score/short_cot_qwen.py",
)


def load_gpt4o_class():
    gpt_model_path = VERL_ROOT / "gpt_model.py"
    if not gpt_model_path.exists():
        raise FileNotFoundError(f"gpt_model.py not found under VERL root: {gpt_model_path}")
    gpt_model_module = _load_module_from_file("verl_root_gpt_model_local", gpt_model_path)
    return gpt_model_module.GPT4o


def default_mathvista_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            label="Qwen-Standard",
            prediction_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVista_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
            ),
        ),
        ModelSpec(
            label="w/o process reward",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVista_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
            ),
        ),
        ModelSpec(
            label="w/o step bouns",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVista_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-RL",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVista_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-SFT-800it",
            prediction_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVista_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVista_MINI_gpt-4.1-2025-04-14.pkl"
            ),
        ),
    ]


def default_mathverse_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            label="Qwen-Standard",
            prediction_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVerse_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_Geaaeb9c9/Qwen3-VL-8B-Instruct_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
            ),
        ),
        ModelSpec(
            label="w/o process reward",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVerse_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
            ),
        ),
        ModelSpec(
            label="w/o step bouns",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVerse_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-RL",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVerse_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-SFT-800it",
            prediction_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVerse_MINI.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_MathVerse_MINI_gpt-4.1-2025-04-14_extract.pkl"
            ),
        ),
    ]


def default_logicvista_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            label="Qwen-Standard",
            prediction_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_G624a0651/Qwen3-VL-8B-Instruct_LogicVista.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Qwen3-VL/Qwen3-VL-8B-Instruct/"
                "T20260112_G624a0651/Qwen3-VL-8B-Instruct_LogicVista_gpt4.1.pkl"
            ),
        ),
        ModelSpec(
            label="w/o process reward",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_LogicVista.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only/"
                "T20260406_G6d220c41/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only_LogicVista_gpt4.1.pkl"
            ),
        ),
        ModelSpec(
            label="w/o step bouns",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_LogicVista.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP_LogicVista_gpt4.1.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-RL",
            prediction_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_LogicVista.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/170it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL/"
                "T20260411_Gf7429a73/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL_LogicVista_gpt4.1.pkl"
            ),
        ),
        ModelSpec(
            label="DRT-SFT-800it",
            prediction_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_LogicVista.xlsx"
            ),
            extract_pkl_path=Path(
                "./results/Short-COT-Image/800it/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it/"
                "T20260409_Gc7c180bc/"
                "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it_LogicVista_gpt4.1.pkl"
            ),
        ),
    ]


def benchmark_configs() -> Dict[str, BenchmarkConfig]:
    return {
        "mathvista": BenchmarkConfig(
            name="mathvista",
            gt_parquet=Path("./data/mathvista/testmini.parquet"),
            model_specs=default_mathvista_specs(),
        ),
        "mathverse": BenchmarkConfig(
            name="mathverse",
            gt_parquet=Path("./data/mathverse/testmini.parquet"),
            model_specs=default_mathverse_specs(),
        ),
        "logicvista": BenchmarkConfig(
            name="logicvista",
            gt_parquet=Path("./data/logicvista/test.parquet"),
            model_specs=default_logicvista_specs(),
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate visual hallucination in the whole reasoning traces of the benchmark models. "
            "Visual hallucinations are logged separately from reasoning hallucinations."
        )
    )
    parser.add_argument("--benchmark", choices=["mathvista", "mathverse", "logicvista", "all"], default="mathvista")
    parser.add_argument(
        "--gt_parquet",
        type=Path,
        default=None,
        help="Override GT parquet for a single selected benchmark. Not allowed with --benchmark all.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model_labels",
        type=str,
        default=None,
        help="Comma-separated subset of model labels, e.g. 'Qwen-Standard,DRT-RL'.",
    )
    parser.add_argument(
        "--model_order",
        type=str,
        default=None,
        help=(
            "Comma-separated model labels to run first, in this order. "
            "Models not listed here are appended in the benchmark default order."
        ),
    )
    parser.add_argument(
        "--model_concurrency_overrides",
        type=str,
        default=None,
        help="Comma-separated overrides like 'Qwen-Standard=8,DRT-SFT-800it=32'.",
    )
    parser.add_argument(
        "--judge_backend",
        choices=["openai", "router"],
        default="openai",
        help="Use OPENAI_API_* env vars or a local reward router endpoint.",
    )
    parser.add_argument(
        "--reward_router_address",
        type=str,
        default=None,
        help="OpenAI-compatible router address like 127.0.0.1:8000 for --judge_backend router.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="auto",
        help="Judge model name. 'auto' selects a backend-appropriate default.",
    )
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--judge_max_tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per model.")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--max_problem_chars", type=int, default=5000)
    parser.add_argument("--max_gt_steps_chars", type=int, default=6000)
    parser.add_argument("--max_reason_chars", type=int, default=10000)
    return parser.parse_args()


def _clean_prediction_text(text):
    text = str(text or "").strip()
    text = text.replace("_x000D_", "\n")
    return text.strip()


def _extract_final_answer_heuristic(text):
    text = _clean_prediction_text(text)
    if not text:
        return ""

    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return short_cot_qwen.clean_final_answer(answer_match[-1])

    cue_match = re.findall(
        r"(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if cue_match:
        return short_cot_qwen.clean_final_answer(cue_match[-1])

    tail = text.strip().splitlines()[-1]
    return short_cot_qwen.clean_final_answer(tail)


def _extract_visual_text(raw_prediction: str) -> str:
    matches = re.findall(r"<visual>\s*(.*?)\s*</visual>", raw_prediction, flags=re.IGNORECASE | re.DOTALL)
    return "\n".join(match.strip() for match in matches if str(match).strip())


def build_reasoning_view(raw_prediction, extracted_answer):
    raw_prediction = _clean_prediction_text(raw_prediction)
    extracted_answer = short_cot_qwen.clean_final_answer("" if extracted_answer is None else str(extracted_answer))

    pred_visual = _extract_visual_text(raw_prediction)
    pred_think, pred_final = short_cot_qwen.extract_parts(raw_prediction)
    final_answer = short_cot_qwen.clean_final_answer(extracted_answer or pred_final or _extract_final_answer_heuristic(raw_prediction))

    if pred_think is None:
        think_text = raw_prediction
        think_text = re.sub(r"<answer>.*?</answer>", " ", think_text, flags=re.IGNORECASE | re.DOTALL)
        think_text = re.sub(r"<visual>.*?</visual>", " ", think_text, flags=re.IGNORECASE | re.DOTALL)
        if final_answer:
            think_text = re.sub(
                r"(?:^|\n)\s*(?:final\s*answer|answer)\s*[:：]\s*.*$",
                " ",
                think_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        think_text = think_text.strip()
    else:
        think_text = pred_think.strip()

    reasoning_blocks = []
    if pred_visual:
        reasoning_blocks.append(f"[visual]\n{pred_visual}")
    if think_text:
        reasoning_blocks.append(f"[think]\n{think_text}")
    pred_reasoning = "\n\n".join(reasoning_blocks).strip()

    return {
        "pred_visual": pred_visual,
        "pred_think": think_text,
        "pred_reasoning": pred_reasoning,
        "pred_final": final_answer,
        "normalized_solution": f"<visual>{pred_visual}</visual><think>{think_text}</think><answer>{final_answer}</answer>",
    }


def load_gt_records(gt_parquet: Path):
    df = pd.read_parquet(gt_parquet)
    records = {}
    for row in df.to_dict("records"):
        extra_info = row.get("extra_info", {}) or {}
        reward_model = row.get("reward_model", {}) or {}
        gt_steps, gt_answer = short_cot_qwen.parse_ground_truth(reward_model.get("ground_truth", ""))
        original_index = extra_info.get("original_index")
        if original_index is None:
            continue
        eval_index = int(original_index) + 1
        records[eval_index] = {
            "gt_answer": gt_answer,
            "gt_steps": gt_steps,
            "gt_step_count": len(gt_steps),
            "problem": extra_info.get("problem", ""),
            "extra_info": extra_info,
        }
    return records


def load_extract_results(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        return {int(k): v for k, v in obj.items()}

    raise ValueError(f"Unsupported extract result format in {path}: {type(obj).__name__}")


def load_model_samples(model_spec: ModelSpec, gt_records: dict, benchmark: str, limit=None):
    df = pd.read_excel(model_spec.prediction_path)
    extract_results = load_extract_results(model_spec.extract_pkl_path)

    samples = []
    for row in df.to_dict("records"):
        index_value = row.get("index")
        if pd.isna(index_value):
            continue

        eval_index = int(index_value)
        gt_item = gt_records.get(eval_index)
        if gt_item is None:
            continue

        extract_item = extract_results.get(eval_index, {})
        extracted_answer = ""
        if isinstance(extract_item, dict):
            extracted_answer = extract_item.get("res", "")
        elif extract_item is not None:
            extracted_answer = str(extract_item)

        reasoning_view = build_reasoning_view(row.get("prediction", ""), extracted_answer)
        samples.append(
            {
                "benchmark": benchmark,
                "model_label": model_spec.label,
                "eval_index": eval_index,
                "question": row.get("question", ""),
                "raw_prediction": row.get("prediction", ""),
                "pred_visual": reasoning_view["pred_visual"],
                "pred_think": reasoning_view["pred_think"],
                "pred_reasoning": reasoning_view["pred_reasoning"],
                "pred_final": reasoning_view["pred_final"],
                "normalized_solution": reasoning_view["normalized_solution"],
                "judge_extract_answer": extracted_answer,
                "gt_answer": gt_item["gt_answer"],
                "gt_steps": gt_item["gt_steps"],
                "gt_step_count": gt_item["gt_step_count"],
                "problem": gt_item["problem"] or row.get("question", ""),
                "extra_info": gt_item["extra_info"],
            }
        )

        if limit is not None and len(samples) >= limit:
            break

    return samples


def format_gt_steps(gt_steps):
    return "".join(f"Step {idx + 1}: {step}\n" for idx, step in enumerate(gt_steps))


def clip_text(text: Any, max_chars: int) -> str:
    text = str(text or "")
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


def load_cache(cache_path: Path, ignore_cache=False):
    if ignore_cache or not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, cache_data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_int(value, default=0):
    try:
        return max(0, int(value))
    except Exception:
        return default


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def _normalize_spans(value):
    if not isinstance(value, list):
        return []
    spans = []
    for item in value:
        if isinstance(item, dict):
            quote = str(item.get("quote", "") or "").strip()
            span_type = str(item.get("type", "other") or "other").strip()
            explanation = str(item.get("explanation", "") or "").strip()
        else:
            quote = str(item or "").strip()
            span_type = "other"
            explanation = ""
        if quote or explanation:
            spans.append(
                {
                    "quote": quote[:1000],
                    "type": span_type[:120],
                    "explanation": explanation[:1200],
                }
            )
    return spans


def _span_types(spans):
    return ",".join(sorted({span.get("type", "other") for span in spans if span.get("type")}))


class VisualHallucinationAnalyzer:
    def __init__(
        self,
        judge_backend,
        reward_router_address=None,
        judge_model="auto",
        max_concurrency=4,
        judge_max_tokens=2048,
        max_problem_chars=5000,
        max_gt_steps_chars=6000,
        max_reason_chars=10000,
    ):
        self.judge_backend = judge_backend
        self.reward_router_address = reward_router_address
        self.max_concurrency = max_concurrency
        self.judge_max_tokens = judge_max_tokens
        self.max_problem_chars = max_problem_chars
        self.max_gt_steps_chars = max_gt_steps_chars
        self.max_reason_chars = max_reason_chars
        self.semaphore = None
        self.openai_client = None

        if judge_backend == "openai":
            resolved_judge_model = judge_model if judge_model != "auto" else DEFAULT_OPENAI_JUDGE_MODEL
            api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_base or not api_key:
                raise ValueError("OPENAI_API_BASE/OPENAI_BASE_URL and OPENAI_API_KEY are required for --judge_backend openai")
            GPT4o = load_gpt4o_class()
            self.openai_client = GPT4o(
                deployment_name=resolved_judge_model,
                api_base=api_base,
                api_key=api_key,
            )
        elif judge_backend == "router":
            resolved_judge_model = judge_model if judge_model != "auto" else DEFAULT_ROUTER_JUDGE_MODEL
            if not reward_router_address:
                raise ValueError("--reward_router_address is required for --judge_backend router")
            if resolved_judge_model != DEFAULT_ROUTER_JUDGE_MODEL:
                print(
                    f"Warning: router backend ignores custom judge model '{resolved_judge_model}' "
                    f"and uses the server-side configured model."
                )
        else:
            raise ValueError(f"Unsupported judge backend: {judge_backend}")

    async def _send_messages(self, messages, max_tokens=2048):
        if self.judge_backend == "openai":
            def _send_openai_direct():
                last_error = None
                for retry_idx in range(3):
                    try:
                        result = self.openai_client.client.chat.completions.create(
                            model=self.openai_client.deployment_name,
                            messages=messages,
                            temperature=min(retry_idx * 0.1, 1.0),
                        )
                        content = result.choices[0].message.content
                        if content:
                            return content
                    except Exception as exc:
                        last_error = exc
                        time.sleep(1)
                if last_error is not None:
                    raise last_error
                raise RuntimeError("OpenAI judge returned empty content after retries")

            return await asyncio.to_thread(_send_openai_direct)
        return await short_cot_qwen.generate_chat_aiohttp(
            router_address=self.reward_router_address,
            messages=messages,
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
        )

    @staticmethod
    def _extract_json(text):
        text = str(text or "")
        candidates = []

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.append(fenced.group(1))

        start = text.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1])
                        break

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        def _escape_control_chars_in_strings(candidate):
            output = []
            in_string = False
            escape = False
            for char in candidate:
                if in_string:
                    if escape:
                        output.append(char)
                        escape = False
                    elif char == "\\":
                        output.append(char)
                        escape = True
                    elif char == '"':
                        output.append(char)
                        in_string = False
                    elif char == "\n":
                        output.append("\\n")
                    elif char == "\r":
                        output.append("\\r")
                    elif char == "\t":
                        output.append("\\t")
                    else:
                        output.append(char)
                    continue

                output.append(char)
                if char == '"':
                    in_string = True
            return "".join(output)

        for candidate in candidates:
            candidate = candidate.strip()
            for repaired in (candidate, _escape_control_chars_in_strings(candidate)):
                try:
                    return json.loads(repaired)
                except Exception:
                    pass
                try:
                    parsed = ast.literal_eval(repaired)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return None

    @staticmethod
    def _default_result(raw_response="", parse_ok=False, error=""):
        return {
            "visual_hallucination_present": False,
            "visual_hallucination_count": 0,
            "visual_hallucination_spans": [],
            "reasoning_hallucination_present": False,
            "reasoning_hallucination_count": 0,
            "reasoning_hallucination_spans": [],
            "judge_reason": "",
            "judge_raw_response": raw_response,
            "judge_parse_ok": parse_ok,
            "judge_error": error,
        }

    def _normalize_result(self, result, raw_response):
        if not isinstance(result, dict):
            return self._default_result(raw_response=raw_response, parse_ok=False, error="judge response is not JSON")

        visual_spans = _normalize_spans(result.get("visual_hallucination_spans", []))
        reasoning_spans = _normalize_spans(result.get("reasoning_hallucination_spans", []))

        visual_count = _coerce_int(result.get("visual_hallucination_count", len(visual_spans)), default=len(visual_spans))
        reasoning_count = _coerce_int(
            result.get("reasoning_hallucination_count", len(reasoning_spans)),
            default=len(reasoning_spans),
        )
        if visual_count == 0 and visual_spans:
            visual_count = len(visual_spans)
        if reasoning_count == 0 and reasoning_spans:
            reasoning_count = len(reasoning_spans)

        visual_present = _coerce_bool(
            result.get("visual_hallucination_present", visual_count > 0),
            default=visual_count > 0,
        )
        reasoning_present = _coerce_bool(
            result.get("reasoning_hallucination_present", reasoning_count > 0),
            default=reasoning_count > 0,
        )
        if visual_count > 0:
            visual_present = True
        if reasoning_count > 0:
            reasoning_present = True
        if visual_present and visual_count == 0:
            visual_count = max(1, len(visual_spans))
        if reasoning_present and reasoning_count == 0:
            reasoning_count = max(1, len(reasoning_spans))

        return {
            "visual_hallucination_present": visual_present,
            "visual_hallucination_count": visual_count,
            "visual_hallucination_spans": visual_spans,
            "reasoning_hallucination_present": reasoning_present,
            "reasoning_hallucination_count": reasoning_count,
            "reasoning_hallucination_spans": reasoning_spans,
            "judge_reason": str(result.get("reason", "") or "").strip()[:2000],
            "judge_raw_response": raw_response,
            "judge_parse_ok": True,
            "judge_error": "",
        }

    async def judge_sample(self, sample):
        async with self.semaphore:
            try:
                gt_steps_str = clip_text(format_gt_steps(sample["gt_steps"]), self.max_gt_steps_chars)
                problem_text = clip_text(sample.get("problem", "") or "N/A", self.max_problem_chars)
                pred_reasoning = clip_text(sample.get("pred_reasoning", "") or "", self.max_reason_chars)
                pred_final = clip_text(sample.get("pred_final", "") or "", 1000)

                prompt = VISUAL_HALLUCINATION_PROMPT.format(
                    problem_text=problem_text,
                    gt_steps_str=gt_steps_str,
                    pred_reasoning=pred_reasoning,
                    pred_final=pred_final,
                )
                last_response_text = ""
                for attempt_idx in range(2):
                    attempt_prompt = prompt
                    if attempt_idx > 0:
                        attempt_prompt = (
                            prompt
                            + "\n\nYour previous response was not valid JSON. Retry with one strict JSON object only. "
                            + "Keep quote fields single-line and short."
                        )
                    response_text = await self._send_messages(
                        [{"role": "user", "content": attempt_prompt}],
                        max_tokens=self.judge_max_tokens,
                    )
                    last_response_text = response_text
                    result = self._extract_json(response_text)
                    normalized = self._normalize_result(result, response_text)
                    if normalized["judge_parse_ok"]:
                        return normalized
                return self._default_result(
                    raw_response=last_response_text,
                    parse_ok=False,
                    error="judge response is not JSON",
                )
            except Exception as exc:
                print(f"[Judge Error] benchmark={sample.get('benchmark')} model={sample.get('model_label')} "
                      f"index={sample.get('eval_index')} error={exc}")
                return self._default_result(parse_ok=False, error=str(exc))


def enrich_metrics(sample, judge_result):
    visual_spans = judge_result.get("visual_hallucination_spans", []) or []
    reasoning_spans = judge_result.get("reasoning_hallucination_spans", []) or []
    pred_reasoning = sample.get("pred_reasoning", "") or ""

    return {
        "analysis_version": ANALYSIS_VERSION,
        "benchmark": sample["benchmark"],
        "model_label": sample["model_label"],
        "eval_index": sample["eval_index"],
        "gt_step_count": int(sample["gt_step_count"]),
        "visual_hallucination_present": bool(judge_result.get("visual_hallucination_present", False)),
        "visual_hallucination_count": int(judge_result.get("visual_hallucination_count", 0)),
        "visual_hallucination_types": _span_types(visual_spans),
        "visual_hallucination_spans": visual_spans,
        "reasoning_hallucination_present": bool(judge_result.get("reasoning_hallucination_present", False)),
        "reasoning_hallucination_count": int(judge_result.get("reasoning_hallucination_count", 0)),
        "reasoning_hallucination_types": _span_types(reasoning_spans),
        "reasoning_hallucination_spans": reasoning_spans,
        "judge_reason": judge_result.get("judge_reason", ""),
        "judge_parse_ok": bool(judge_result.get("judge_parse_ok", False)),
        "judge_error": judge_result.get("judge_error", ""),
        "judge_raw_response": judge_result.get("judge_raw_response", ""),
        "judge_extract_answer": sample["judge_extract_answer"],
        "gt_answer": sample["gt_answer"],
        "pred_final": sample["pred_final"],
        "prediction_has_visual_tag": "<visual>" in str(sample["raw_prediction"]).lower(),
        "prediction_has_think_tag": "<think>" in str(sample["raw_prediction"]).lower(),
        "prediction_length_chars": len(str(sample["raw_prediction"] or "")),
        "pred_visual_token_count": short_cot_qwen.estimate_tokens(sample.get("pred_visual", "")),
        "pred_think_token_count": short_cot_qwen.estimate_tokens(sample.get("pred_think", "")),
        "pred_reasoning_token_count": short_cot_qwen.estimate_tokens(pred_reasoning),
    }


async def run_analysis(
    samples,
    analyzer: Optional[VisualHallucinationAnalyzer],
    cache_path: Path,
    ignore_cache=False,
    desc="Judging visual hallucination",
):
    cache_data = load_cache(cache_path, ignore_cache=ignore_cache)
    results = []
    pending = []

    for sample in samples:
        cache_key = f"{ANALYSIS_VERSION}::{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
        cached = cache_data.get(cache_key)
        if cached is not None:
            results.append(cached)
            continue
        pending.append((cache_key, sample))

    if pending and analyzer is None:
        raise ValueError("Pending samples require a configured analyzer, but analyzer is None.")

    if pending:
        analyzer.semaphore = asyncio.Semaphore(analyzer.max_concurrency)
        progress = tqdm(total=len(pending), desc=desc)
        save_interval = 20
        completed_since_save = 0

        async def _run_single(cache_key, sample):
            judge_result = await analyzer.judge_sample(sample)
            metrics = enrich_metrics(sample, judge_result)
            return cache_key, metrics

        tasks = [asyncio.create_task(_run_single(cache_key, sample)) for cache_key, sample in pending]
        for task in asyncio.as_completed(tasks):
            cache_key, metrics = await task
            cache_data[cache_key] = metrics
            results.append(metrics)
            progress.update(1)
            completed_since_save += 1
            if completed_since_save >= save_interval:
                save_cache(cache_path, cache_data)
                completed_since_save = 0

        progress.close()
        save_cache(cache_path, cache_data)

    return pd.DataFrame(results)


def filter_model_specs(model_specs: List[ModelSpec], model_labels_arg: Optional[str]) -> List[ModelSpec]:
    if not model_labels_arg:
        return model_specs
    wanted = {item.strip() for item in model_labels_arg.split(",") if item.strip()}
    filtered = [spec for spec in model_specs if spec.label in wanted]
    missing = sorted(wanted - {spec.label for spec in filtered})
    if missing:
        raise ValueError(f"Unknown model labels for selected benchmark: {missing}")
    return filtered


def apply_model_order(model_specs: List[ModelSpec], model_order_arg: Optional[str]) -> List[ModelSpec]:
    if not model_order_arg:
        return model_specs

    order_labels = [item.strip() for item in model_order_arg.split(",") if item.strip()]
    specs_by_label = {spec.label: spec for spec in model_specs}
    missing = [label for label in order_labels if label not in specs_by_label]
    if missing:
        raise ValueError(f"Unknown labels in --model_order for selected benchmark: {missing}")

    ordered = []
    seen = set()
    for label in order_labels:
        if label in seen:
            continue
        ordered.append(specs_by_label[label])
        seen.add(label)
    ordered.extend(spec for spec in model_specs if spec.label not in seen)
    return ordered


def parse_model_concurrency_overrides(overrides_arg: Optional[str]) -> Dict[str, int]:
    if not overrides_arg:
        return {}

    overrides = {}
    for item in overrides_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --model_concurrency_overrides item: {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        try:
            concurrency = int(value.strip())
        except Exception as exc:
            raise ValueError(f"Invalid concurrency override value in item: {item!r}") from exc
        if concurrency <= 0:
            raise ValueError(f"Concurrency override must be positive in item: {item!r}")
        overrides[label] = concurrency
    return overrides


def model_concurrency(model_label: str, args, overrides: Dict[str, int]) -> int:
    return int(overrides.get(model_label, args.max_concurrency))


def ordered_model_labels(values):
    present = list(dict.fromkeys(values))
    ordered = [label for label in MODEL_PLOT_ORDER if label in present]
    ordered.extend(label for label in present if label not in ordered)
    return ordered


def build_grouped_summary(per_sample_df: pd.DataFrame, group_cols: List[str]):
    grouped = (
        per_sample_df.groupby(group_cols, as_index=False)
        .agg(
            sample_count=("eval_index", "size"),
            visual_hallucination_rate=("visual_hallucination_present", "mean"),
            mean_visual_hallucination_count=("visual_hallucination_count", "mean"),
            reasoning_hallucination_rate=("reasoning_hallucination_present", "mean"),
            mean_reasoning_hallucination_count=("reasoning_hallucination_count", "mean"),
            judge_parse_ok_rate=("judge_parse_ok", "mean"),
            mean_pred_reasoning_tokens=("pred_reasoning_token_count", "mean"),
            mean_pred_think_tokens=("pred_think_token_count", "mean"),
        )
        .sort_values(group_cols)
    )
    return grouped


def build_gt_step_summary(per_sample_df: pd.DataFrame):
    grouped = build_grouped_summary(per_sample_df, ["benchmark", "model_label", "gt_step_count"])
    return grouped.sort_values(["benchmark", "model_label", "gt_step_count"])


def plot_model_rates(summary_df: pd.DataFrame, output_path: Path, title: str):
    if plt is None or summary_df.empty:
        return

    model_labels = ordered_model_labels(summary_df["model_label"].tolist())
    plot_df = summary_df.set_index("model_label").loc[model_labels].reset_index()

    x_positions = list(range(len(plot_df)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(
        [x - width / 2 for x in x_positions],
        plot_df["visual_hallucination_rate"],
        width=width,
        label="Visual hallucination",
        color="#d62728",
    )
    axis.bar(
        [x + width / 2 for x in x_positions],
        plot_df["reasoning_hallucination_rate"],
        width=width,
        label="Reasoning hallucination",
        color="#4c78a8",
    )
    axis.set_title(title)
    axis.set_ylabel("Sample rate")
    axis.set_ylim(0, 1)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_df["model_label"], rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def csv_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    for column in ("visual_hallucination_spans", "reasoning_hallucination_spans"):
        if column in output_df.columns:
            output_df[column] = output_df[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    if "judge_raw_response" in output_df.columns:
        output_df = output_df.drop(columns=["judge_raw_response"])
    return output_df


def validate_input_files(model_specs: List[ModelSpec], gt_parquet: Path):
    if not gt_parquet.exists():
        raise FileNotFoundError(f"GT parquet not found: {gt_parquet}")
    for model_spec in model_specs:
        if not model_spec.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {model_spec.prediction_path}")
        if not model_spec.extract_pkl_path.exists():
            raise FileNotFoundError(f"Extract PKL not found: {model_spec.extract_pkl_path}")


def selected_benchmarks(args, configs: Dict[str, BenchmarkConfig]) -> List[BenchmarkConfig]:
    if args.benchmark == "all":
        if args.gt_parquet is not None:
            raise ValueError("--gt_parquet override is only allowed when selecting a single benchmark.")
        return [configs["mathvista"], configs["mathverse"], configs["logicvista"]]

    config = configs[args.benchmark]
    if args.gt_parquet is None:
        return [config]
    return [
        BenchmarkConfig(
            name=config.name,
            gt_parquet=args.gt_parquet,
            model_specs=config.model_specs,
        )
    ]


def run_benchmark(config: BenchmarkConfig, args):
    model_specs = filter_model_specs(config.model_specs, args.model_labels)
    model_specs = apply_model_order(model_specs, args.model_order)
    concurrency_overrides = parse_model_concurrency_overrides(args.model_concurrency_overrides)
    validate_input_files(model_specs, config.gt_parquet)

    gt_records = load_gt_records(config.gt_parquet)
    cache_path = args.output_dir / f"{ANALYSIS_VERSION}_{config.name}_cache.json"
    model_frames = []

    for model_spec in model_specs:
        samples = load_model_samples(model_spec, gt_records, benchmark=config.name, limit=args.limit)
        if not samples:
            print(f"[{config.name}] model={model_spec.label} has no overlapping samples; skipping.")
            continue

        cache_data = load_cache(cache_path, ignore_cache=args.ignore_cache)
        pending_count = 0
        for sample in samples:
            cache_key = f"{ANALYSIS_VERSION}::{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
            if cache_key not in cache_data:
                pending_count += 1

        current_concurrency = model_concurrency(model_spec.label, args, concurrency_overrides)
        print(
            f"[{config.name}] model={model_spec.label} samples={len(samples)} "
            f"pending={pending_count} max_concurrency={current_concurrency}"
        )

        analyzer = None
        if pending_count > 0:
            analyzer = VisualHallucinationAnalyzer(
                judge_backend=args.judge_backend,
                reward_router_address=args.reward_router_address,
                judge_model=args.judge_model,
                max_concurrency=current_concurrency,
                judge_max_tokens=args.judge_max_tokens,
                max_problem_chars=args.max_problem_chars,
                max_gt_steps_chars=args.max_gt_steps_chars,
                max_reason_chars=args.max_reason_chars,
            )

        model_df = asyncio.run(
            run_analysis(
                samples,
                analyzer,
                cache_path=cache_path,
                ignore_cache=args.ignore_cache,
                desc=f"{config.name}:{model_spec.label}",
            )
        )
        model_frames.append(model_df)

    if not model_frames:
        raise ValueError(f"No overlapping samples found for benchmark={config.name}.")

    per_sample_df = pd.concat(model_frames, ignore_index=True).sort_values(["benchmark", "model_label", "eval_index"])

    grouped_by_model_df = build_grouped_summary(per_sample_df, ["benchmark", "model_label"])
    grouped_by_gt_step_df = build_gt_step_summary(per_sample_df)

    prefix = f"{ANALYSIS_VERSION}_{config.name}"
    per_sample_path = args.output_dir / f"{prefix}_per_sample.csv"
    grouped_by_model_path = args.output_dir / f"{prefix}_grouped_by_model.csv"
    grouped_by_gt_step_path = args.output_dir / f"{prefix}_grouped_by_gt_steps.csv"
    result_log_path = args.output_dir / f"{prefix}_result_log.jsonl"
    figure_path = args.output_dir / f"{prefix}_model_rates.png"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_ready_df(per_sample_df).to_csv(per_sample_path, index=False)
    grouped_by_model_df.to_csv(grouped_by_model_path, index=False)
    grouped_by_gt_step_df.to_csv(grouped_by_gt_step_path, index=False)
    write_jsonl(result_log_path, per_sample_df.to_dict("records"))
    plot_model_rates(grouped_by_model_df, figure_path, f"{config.name}: visual vs reasoning hallucination rate")

    print(f"[{config.name}] cache: {cache_path}")
    print(f"[{config.name}] per-sample CSV: {per_sample_path}")
    print(f"[{config.name}] grouped CSV: {grouped_by_model_path}")
    print(f"[{config.name}] GT-step grouped CSV: {grouped_by_gt_step_path}")
    print(f"[{config.name}] result log: {result_log_path}")
    if plt is not None:
        print(f"[{config.name}] plot: {figure_path}")

    return per_sample_df


def write_combined_outputs(per_sample_frames: List[pd.DataFrame], output_dir: Path):
    if len(per_sample_frames) <= 1:
        return

    combined_df = pd.concat(per_sample_frames, ignore_index=True).sort_values(["benchmark", "model_label", "eval_index"])
    grouped_by_benchmark_model_df = build_grouped_summary(combined_df, ["benchmark", "model_label"])
    grouped_by_model_df = build_grouped_summary(combined_df, ["model_label"])

    prefix = f"{ANALYSIS_VERSION}_all"
    per_sample_path = output_dir / f"{prefix}_per_sample.csv"
    grouped_by_benchmark_model_path = output_dir / f"{prefix}_grouped_by_benchmark_model.csv"
    grouped_by_model_path = output_dir / f"{prefix}_grouped_by_model.csv"
    result_log_path = output_dir / f"{prefix}_result_log.jsonl"
    figure_path = output_dir / f"{prefix}_model_rates.png"

    csv_ready_df(combined_df).to_csv(per_sample_path, index=False)
    grouped_by_benchmark_model_df.to_csv(grouped_by_benchmark_model_path, index=False)
    grouped_by_model_df.to_csv(grouped_by_model_path, index=False)
    write_jsonl(result_log_path, combined_df.to_dict("records"))
    plot_model_rates(grouped_by_model_df, figure_path, "All benchmarks: visual vs reasoning hallucination rate")

    print(f"[all] per-sample CSV: {per_sample_path}")
    print(f"[all] grouped by benchmark/model CSV: {grouped_by_benchmark_model_path}")
    print(f"[all] grouped by model CSV: {grouped_by_model_path}")
    print(f"[all] result log: {result_log_path}")
    if plt is not None:
        print(f"[all] plot: {figure_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = benchmark_configs()
    frames = []
    for config in selected_benchmarks(args, configs):
        frames.append(run_benchmark(config, args))
    write_combined_outputs(frames, args.output_dir)


if __name__ == "__main__":
    main()
