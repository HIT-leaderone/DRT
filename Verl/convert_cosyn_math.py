import argparse
import base64
import contextlib
import io
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd
import openai
from datasets import get_dataset_split_names, load_dataset
from PIL import Image

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency is expected with pandas parquet support
    pa = None  # type: ignore
    pq = None  # type: ignore
from tqdm import tqdm

try:
    from gpt_model import GPT4o
except ImportError:
    print("Warning: gpt_model.py not found. Please ensure it is in the Python path.")

    class GPT4o:  # type: ignore
        def __init__(self, **kwargs):
            pass

        def send_stable_request(self, *args, **kwargs):
            return "{}"


DEFAULT_DATASET_NAME = "allenai/CoSyn-400K"
DEFAULT_DATASET_CONFIG = "math"
DEFAULT_SPLITS = ["train", "validation"]
DEFAULT_OUTPUT_DIR = "data/cosyn_math_gpt51_filtered"
DEFAULT_MODEL_NAME = "gpt-5.1-2025-11-13"
DEFAULT_TARGET_BENCHES = ["MathVista", "MathVerse", "LogicVista", "GSM8K", "VideoHolmes"]
MAX_RETRIES = 4
DEFAULT_API_MAX_WORKERS = 4
DEFAULT_SKIP_GEN_MAX_WORKERS = 16


class AzureChatClientNoTemperature:
    """Azure Chat Completions adapter for endpoints that reject temperature.

    Some GPT-5.x modelhub deployments accept multimodal Chat Completions
    payloads but reject explicit `temperature` values; the legacy GPT4o wrapper
    always sends temperature, so this client intentionally omits it.
    """

    def __init__(self, deployment_name: str, api_base: Optional[str], api_key: Optional[str]):
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or getattr(GPT4o, "api_key", None)
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for GPT-5.x direct client")
        resolved_api_base = (
            api_base
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.deployment_name = deployment_name
        self.client = openai.AzureOpenAI(
            azure_endpoint=resolved_api_base,
            api_version="2024-03-01-preview",
            api_key=resolved_api_key,
        )

    def send_stable_request(self, messages, retry: int = 3, temperature: float = 0.0):
        last_exc = None
        for attempt in range(retry):
            try:
                result = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    timeout=120,
                )
                content = result.choices[0].message.content
                if content:
                    return content
                last_exc = RuntimeError(f"empty response on attempt {attempt + 1}")
            except Exception as exc:
                last_exc = exc
                print(f"[GPT-5.x API Error] attempt={attempt + 1}: {exc}")
                time.sleep(1)
        raise RuntimeError(f"Retry time limit exceeded: {last_exc}")

DENSE_PROMPT_TEMPLATE_VL = (
    "{Question}\n\n"
    "Analyze this visual math question to provide a **Dense Cognitive Trace**.\n"
    "--- Requirements ---\n"
    "1. **Format**: Telegraphic style (concise phrases, arrows '->', symbols). No filler.\n"
    "2. **Structure**: Strict XML: "
    "<visual>...visible problem text, diagram/table/chart evidence...</visual>"
    "<think>...[Priors] -> compressed mathematical reasoning...</think>"
    "<answer>...final answer...</answer>\n"
    "3. Ground all visual claims in the image/transcribed problem."
)


def repair_json_content(text: str) -> str:
    return text.replace("\\", "\\\\")


def extract_json(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "")
    json_str = ""
    try:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                json_str = match.group(0)
        if not json_str:
            return None
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json_content(json_str))
        except Exception:
            return None
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_answer_text(answer_text: Any) -> str:
    text = normalize_text(answer_text)
    text = re.sub(r"^<answer>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</answer>$", "", text, flags=re.IGNORECASE)
    return text.strip()


def safe_json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def first_list_value(value: Any) -> str:
    if isinstance(value, list):
        return normalize_text(value[0]) if value else ""
    return normalize_text(value)


def extract_source_qa(item: Dict[str, Any]) -> Dict[str, Any]:
    qa_pairs = item.get("qa_pairs") if isinstance(item.get("qa_pairs"), dict) else {}
    data = safe_json_loads(item.get("data"))
    question = first_list_value(qa_pairs.get("question")) or normalize_text(data.get("question"))
    answer = first_list_value(qa_pairs.get("answer")) or clean_answer_text(data.get("answer"))
    explanation = first_list_value(qa_pairs.get("explanation")) or normalize_text(data.get("explanation"))
    return {"question": question, "answer": answer, "explanation": explanation, "raw_qa_pairs": qa_pairs, "raw_data": data}


def pil_image_to_png_bytes(image: Any) -> Optional[bytes]:
    if image is None or not isinstance(image, Image.Image):
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:
        print(f"[Image Encode Error] {exc}")
        return None


def build_image_records(images: Iterable[Any]) -> List[Dict[str, bytes]]:
    records = []
    for image in images or []:
        image_bytes = pil_image_to_png_bytes(image)
        if image_bytes:
            records.append({"bytes": image_bytes})
    return records


def build_image_url_content(images: Iterable[Any]) -> List[Dict[str, Any]]:
    content = []
    for image in images or []:
        image_bytes = pil_image_to_png_bytes(image)
        if not image_bytes:
            continue
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return content


def init_llm_client(model_name: str):
    api_base = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        if str(model_name).lower().startswith("gpt-5."):
            return AzureChatClientNoTemperature(deployment_name=model_name, api_base=api_base, api_key=api_key)
        # Some local GPT wrappers print masked credentials during init; suppress
        # stdout here so this converter never logs secrets or secret fragments.
        with contextlib.redirect_stdout(io.StringIO()):
            return GPT4o(deployment_name=model_name, api_base=api_base, api_key=api_key)
    except Exception as exc:
        print(f"Warning: Failed to initialize LLM client for model {model_name}: {exc}")
        return None


def resolve_max_workers(max_workers: Optional[int], skip_generation: bool) -> int:
    if max_workers is not None:
        return max_workers
    return DEFAULT_SKIP_GEN_MAX_WORKERS if skip_generation else DEFAULT_API_MAX_WORKERS


def as_list_string(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(x).lower() for x in value if normalize_text(x)]
    if not value:
        return []
    return [normalize_text(value).lower()]


def canonical_visual_dependency(value: Any, fallback_needs_visual: bool = True) -> str:
    dep = normalize_text(value).lower()
    if dep in {"none", "weak", "medium", "strong"}:
        return dep
    return "medium" if fallback_needs_visual else "none"


def deterministic_relabel(item: Dict[str, Any], source_qa: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    figure_type = normalize_text(metadata.get("figure_type") or metadata.get("type") or "rendered_text_problem")
    topic = normalize_text(metadata.get("topic"))
    question = source_qa.get("question") or topic or "Solve the math problem shown in the image."
    answer = clean_answer_text(source_qa.get("answer"))
    explanation = normalize_text(source_qa.get("explanation"))
    tags = infer_capability_tags(" ".join([question, topic, figure_type, explanation]))
    domain = infer_math_domain(" ".join([question, topic, figure_type, explanation]))
    return {
        "clean_question": question,
        "transcribed_problem": question,
        "final_answer": answer,
        "short_solution": explanation[:2000] if explanation else "",
        "image_type": figure_type or "rendered_text_problem",
        "needs_visual": True,
        "visual_dependency": "medium",
        "capability_tags": tags,
        "math_domain": domain,
        "difficulty": "unknown",
        "confidence": 0.55 if answer else 0.25,
    }


def infer_capability_tags(text: str) -> List[str]:
    t = text.lower()
    tags = []
    checks = [
        ("geometry", ["triangle", "circle", "angle", "perimeter", "area", "radius", "diameter", "polygon"]),
        ("chart_table", ["chart", "graph", "plot", "table", "bar", "axis"]),
        ("counting", ["ways", "arrange", "permutation", "combination", "count", "probability"]),
        ("algebra", ["equation", "solve", "variable", "linear", "quadratic", "function"]),
        ("arithmetic", ["sum", "difference", "product", "ratio", "percent", "fraction"]),
        ("logic", ["logic", "puzzle", "deduce", "truth", "valid", "if and only if"]),
        ("ocr", ["latex", "text", "problem", "question"]),
    ]
    for tag, keywords in checks:
        if any(k in t for k in keywords):
            tags.append(tag)
    return tags or ["visual_math", "ocr"]


def infer_math_domain(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["triangle", "circle", "angle", "area", "perimeter", "geometry"]):
        return "geometry"
    if any(k in t for k in ["probability", "permutation", "combination", "arrange", "counting"]):
        return "counting_probability"
    if any(k in t for k in ["equation", "variable", "linear", "quadratic", "function"]):
        return "algebra"
    if any(k in t for k in ["chart", "graph", "table", "plot", "statistics"]):
        return "data_analysis"
    if any(k in t for k in ["logic", "puzzle", "deduce"]):
        return "logic"
    return "general_math"


def build_relabel_messages(item: Dict[str, Any], source_qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    source_payload = {
        "id": item.get("id"),
        "source_question": source_qa.get("question"),
        "source_answer_weak_reference": source_qa.get("answer"),
        "source_explanation_weak_reference": source_qa.get("explanation"),
        "metadata": metadata,
    }
    text = (
        "You are relabeling a CoSyn visual math sample for high-quality multimodal RL data. "
        "The source QA may be noisy; use it only as weak reference, and verify against the image.\n\n"
        f"Source payload:\n{json.dumps(source_payload, ensure_ascii=False)}\n\n"
        "Return strict JSON only with these fields:\n"
        "{\n"
        '  "clean_question": "self-contained question for the student",\n'
        '  "transcribed_problem": "verbatim or faithful transcription of the problem visible in the image",\n'
        '  "final_answer": "answer that solves the image problem",\n'
        '  "short_solution": "brief derivation",\n'
        '  "image_type": "diagram|chart_table|rendered_text_problem|geometry_figure|other",\n'
        '  "needs_visual": true,\n'
        '  "visual_dependency": "none|weak|medium|strong",\n'
        '  "capability_tags": ["ocr", "geometry", "counting", "chart_table", "logic", "algebra", ...],\n'
        '  "math_domain": "geometry|algebra|counting_probability|data_analysis|logic|general_math",\n'
        '  "difficulty": "easy|medium|hard|unknown",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "If the image and source disagree, prefer the image. If unreadable or ambiguous, lower confidence and explain in short_solution."
    )
    content = build_image_url_content([item.get("image")])
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": "You are a meticulous visual math data relabeler. Output strict JSON only."},
        {"role": "user", "content": content},
    ]


def normalize_relabel(parsed: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(fallback)
    for key in [
        "clean_question",
        "transcribed_problem",
        "final_answer",
        "short_solution",
        "image_type",
        "visual_dependency",
        "math_domain",
        "difficulty",
    ]:
        if normalize_text(parsed.get(key)):
            result[key] = clean_answer_text(parsed.get(key)) if key == "final_answer" else normalize_text(parsed.get(key))
    if "needs_visual" in parsed:
        result["needs_visual"] = bool(parsed.get("needs_visual"))
    if parsed.get("capability_tags"):
        result["capability_tags"] = as_list_string(parsed.get("capability_tags"))
    try:
        result["confidence"] = float(parsed.get("confidence", result.get("confidence", 0.0)))
    except Exception:
        result["confidence"] = float(result.get("confidence", 0.0) or 0.0)
    result["visual_dependency"] = canonical_visual_dependency(result.get("visual_dependency"), result.get("needs_visual", True))
    if not result.get("capability_tags"):
        result["capability_tags"] = infer_capability_tags(" ".join([result.get("clean_question", ""), result.get("image_type", "")]))
    if not result.get("math_domain"):
        result["math_domain"] = infer_math_domain(result.get("clean_question", ""))
    return result


def relabel_item(item: Dict[str, Any], source_qa: Dict[str, Any], llm_client: Any, skip_generation: bool) -> Tuple[Optional[Dict[str, Any]], str]:
    fallback = deterministic_relabel(item, source_qa)
    if skip_generation:
        return fallback, "skip_deterministic_relabel"
    if llm_client is None:
        return None, "missing_llm_client"
    messages = build_relabel_messages(item, source_qa)
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.send_stable_request(messages, temperature=0.0)
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                relabeled = normalize_relabel(parsed, fallback)
                if relabeled.get("clean_question") and relabeled.get("final_answer"):
                    return relabeled, "gpt_relabel"
        except Exception as exc:
            print(f"[Relabel Error] id={item.get('id')} attempt={attempt + 1}: {exc}")
            time.sleep(1)
    return None, "relabel_failed"


def build_answer_verifier_messages(item: Dict[str, Any], relabeled: Dict[str, Any], source_qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = {
        "id": item.get("id"),
        "relabeled": relabeled,
        "source_question_weak_reference": source_qa.get("question"),
        "source_answer_weak_reference": source_qa.get("answer"),
        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
    }
    text = (
        "Audit this relabeled CoSyn visual math sample against the image. Do not trust the original answer as ground truth.\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "This is a DATA/LABEL quality audit, not a trajectory audit. Check the relabeled question, "
        "transcription, and final answer against explicit visible image/text evidence.\n\n"
        "Audit dimensions:\n"
        "1. `visual_grounding`: Every quantity, label, relation, diagram property, count, measurement, "
        "or condition used by the relabeled question/answer must be directly visible in the image/text, "
        "or be a standard mathematical consequence of visible information. Mark false if the relabel "
        "invents values, assumes missing visual facts, or silently changes the problem.\n"
        "2. `answer_correct`: The final_answer must solve the visible/transcribed problem. Mark false for "
        "wrong arithmetic, wrong modeling, unit mistakes, or answers that only solve an added/changed problem.\n"
        "3. `image_question_consistent`: The clean_question and transcribed_problem must faithfully match "
        "the image. Mark false if the clean question adds constraints, independence assumptions, log bases, "
        "payoff rules, coordinate conventions, or interpretation choices that are not stated or forced.\n"
        "4. `not_ambiguous_or_broken`: Mark false if the image/problem is unreadable, internally inconsistent, "
        "under-specified, ambiguous between materially different answers, unrelated to the relabel, or has "
        "placeholder/irrelevant visual content that is needed for the task. If the visible problem explicitly "
        "asks to detect infeasibility/inconsistency, it may pass only when that conclusion follows directly "
        "from the image and the question remains well-posed.\n"
        "5. `no_unstated_assumptions`: Mark false for ad hoc assumptions, speculative shortcuts, or defaults "
        "that materially affect the answer unless the assumption is explicitly stated in the visible problem.\n\n"
        "Decision rule: `verdict` must be `PASS` only when all five audit dimensions are true; otherwise FAIL.\n"
        "Output strict JSON only:\n"
        "{\n"
        '  "visual_grounding": true,\n'
        '  "answer_correct": true,\n'
        '  "image_question_consistent": true,\n'
        '  "not_ambiguous_or_broken": true,\n'
        '  "no_unstated_assumptions": true,\n'
        '  "verdict": "PASS",\n'
        '  "reason": "brief reason"\n'
        "}"
    )
    content = build_image_url_content([item.get("image")])
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": "You are a strict verifier for visual math dataset labels. Output JSON only."},
        {"role": "user", "content": content},
    ]


def verifier_audit_dict(parsed: Optional[Dict[str, Any]], fields: List[str], default_reason: str = "") -> Dict[str, Any]:
    parsed = parsed if isinstance(parsed, dict) else {}
    audit: Dict[str, Any] = {
        "verdict": normalize_text(parsed.get("verdict")),
        "reason": normalize_text(parsed.get("reason")) or default_reason,
    }
    for field in fields:
        audit[field] = parsed.get(field) if isinstance(parsed.get(field), bool) else None
    return audit


def verify_answer_sample(item: Dict[str, Any], relabeled: Dict[str, Any], source_qa: Dict[str, Any], llm_client: Any, skip_generation: bool) -> Tuple[bool, str, Dict[str, Any]]:
    audit_fields = [
        "visual_grounding",
        "answer_correct",
        "image_question_consistent",
        "not_ambiguous_or_broken",
        "no_unstated_assumptions",
    ]
    if skip_generation:
        if not relabeled.get("clean_question") or not relabeled.get("final_answer"):
            return False, "missing_question_or_answer", verifier_audit_dict(None, audit_fields, "missing_question_or_answer")
        return True, "skip_deterministic_answer_check", {
            "verdict": "SKIP",
            "reason": "skip_deterministic_answer_check",
            **{field: None for field in audit_fields},
        }
    messages = build_answer_verifier_messages(item, relabeled, source_qa)
    last_audit = verifier_audit_dict(None, audit_fields, "answer_verifier_failed")
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.send_stable_request(messages, temperature=0.0)
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                audit = verifier_audit_dict(parsed, audit_fields)
                last_audit = audit
                ok = (
                    str(parsed.get("verdict", "")).upper() == "PASS"
                    and parsed.get("visual_grounding") is True
                    and parsed.get("answer_correct") is True
                    and parsed.get("image_question_consistent") is True
                    and parsed.get("not_ambiguous_or_broken") is True
                    and parsed.get("no_unstated_assumptions") is True
                )
                return ok, normalize_text(parsed.get("reason")) or ("answer_verifier_pass" if ok else "answer_verifier_fail"), audit
        except Exception as exc:
            print(f"[Answer Verifier Error] id={item.get('id')} attempt={attempt + 1}: {exc}")
            time.sleep(1)
    return False, "answer_verifier_failed", last_audit


def compute_bench_relevance(relabeled: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, int]:
    tags = set(as_list_string(relabeled.get("capability_tags")))
    domain = normalize_text(relabeled.get("math_domain")).lower()
    image_type = normalize_text(relabeled.get("image_type")).lower()
    dep = canonical_visual_dependency(relabeled.get("visual_dependency"), bool(relabeled.get("needs_visual", True)))
    text = " ".join([domain, image_type, " ".join(tags), normalize_text(metadata.get("figure_type")).lower(), relabeled.get("clean_question", "").lower()])
    dep_bonus = {"none": 0, "weak": 1, "medium": 2, "strong": 3}.get(dep, 1)

    mathvista = 0
    if dep in {"medium", "strong"} and any(k in text for k in ["geometry", "chart", "table", "graph", "diagram", "data", "visual", "rendered"]):
        mathvista = min(4, 2 + dep_bonus)
    elif dep in {"medium", "strong"}:
        mathvista = 3
    elif "math" in text:
        mathvista = 1

    mathverse = 0
    if dep == "strong":
        mathverse = 4
    elif dep == "medium":
        mathverse = 3
    elif dep == "weak":
        mathverse = 1

    logicvista = 0
    if any(k in text for k in ["logic", "puzzle", "deduce", "spatial", "arrange", "counting", "truth"]):
        logicvista = 4 if dep in {"medium", "strong"} else 2
    elif any(k in text for k in ["geometry", "diagram"]):
        logicvista = 2 if dep in {"medium", "strong"} else 1

    gsm8k = 0
    if dep in {"none", "weak"} and any(k in text for k in ["arithmetic", "word", "counting", "algebra", "general_math"]):
        gsm8k = 3 if dep == "none" else 2
    elif any(k in text for k in ["arithmetic", "counting", "algebra", "general_math"]):
        gsm8k = 1

    videoholmes = 0
    if any(k in text for k in ["video", "temporal", "sequence", "motion"]):
        videoholmes = 3 if dep in {"medium", "strong"} else 1

    return {
        "MathVista": int(mathvista),
        "MathVerse": int(mathverse),
        "LogicVista": int(logicvista),
        "GSM8K": int(gsm8k),
        "VideoHolmes": int(videoholmes),
    }


def passes_bench_filter(relabeled: Dict[str, Any], bench_relevance: Dict[str, int], target_benches: List[str], min_bench_score: int, allow_text_only: bool) -> Tuple[bool, str]:
    dep = canonical_visual_dependency(relabeled.get("visual_dependency"), bool(relabeled.get("needs_visual", True)))
    max_target = max([bench_relevance.get(bench, 0) for bench in target_benches] or [0])
    if max_target < min_bench_score:
        return False, f"bench_score_below_{min_bench_score}"
    if not allow_text_only and dep not in {"medium", "strong"}:
        return False, f"visual_dependency_{dep}_not_allowed"
    return True, "bench_filter_pass"


def build_generation_messages(question: str, answer: str, relabeled: Dict[str, Any], raw_images: List[Any]) -> List[Dict[str, Any]]:
    user_text = (
        "Generate a concise dense reasoning trajectory for this visual math problem.\n\n"
        f"Question: {question}\n"
        f"Transcribed problem: {relabeled.get('transcribed_problem', '')}\n"
        f"Correct Answer (hidden validation only): {answer}\n\n"
        "Rules:\n"
        "1. Use only visible image evidence, the transcribed problem, and valid math.\n"
        "2. Do not cite the hidden answer as a premise.\n"
        "3. Prefer grounded concise steps; no unsupported visual guesses.\n"
        "4. Output strict JSON only: {\"steps\": [\"...\"], \"final_answer\": \"...\"}."
    )
    content = build_image_url_content(raw_images)
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": "You are an expert visual mathematics reasoner producing reliable dense traces."},
        {"role": "user", "content": content},
    ]


def build_trajectory_verifier_messages(question: str, answer: str, pred_answer: str, steps: List[str], relabeled: Dict[str, Any], raw_images: List[Any]) -> List[Dict[str, Any]]:
    text = (
        f"Question: {question}\n"
        f"Transcribed problem: {relabeled.get('transcribed_problem', '')}\n"
        f"Correct Answer: {answer}\n"
        f"Generated Answer: {pred_answer}\n"
        f"Candidate Steps: {json.dumps(steps, ensure_ascii=False)}\n\n"
        "You are auditing a candidate stepwise reasoning trajectory conditioned on the question "
        "and the ground-truth answer. A correct final answer alone is not sufficient. Accept the "
        "trajectory only if it is reliable across perception, intermediate reasoning, and final answer.\n\n"
        "Audit dimensions:\n"
        "1. `visual_fidelity`: For image-based problems, every perceptual claim must be directly supported "
        "by explicit visual/textual evidence, such as visible labels, symbols, quantities, spatial relations, "
        "or clearly specified attributes. Mark false if the trajectory invents or assumes visual values, "
        "diagram properties, counts, relations, or measurements that are not observable. For text-only "
        "problems, mark true unless the trajectory makes unsupported visual/perceptual claims.\n"
        "2. `reasoning_faithfulness`: Each deductive step must follow from verified observations, the problem "
        "statement, explicitly introduced priors, or established theorems. Mark false for logical "
        "inconsistencies, circular reasoning, speculative shortcuts, ad hoc assumptions, unsupported "
        "intermediate values, use of the ground-truth answer as a premise, or traces that merely restate "
        "the answer without a non-trivial derivation.\n"
        "3. `semantic_answer_correctness`: Mark true if the Generated Answer is mathematically or semantically "
        "equivalent to the Correct Answer, allowing equivalent expressions, units, or formatting.\n\n"
        "Decision rule: `verdict` must be `PASS` only when all three audit dimensions are true; otherwise "
        "it must be `FAIL`.\n"
        "Output strict JSON only:\n"
        "{\n"
        '  "visual_fidelity": true,\n'
        '  "reasoning_faithfulness": true,\n'
        '  "semantic_answer_correctness": true,\n'
        '  "verdict": "PASS",\n'
        '  "reason": "brief reason"\n'
        "}"
    )
    content = build_image_url_content(raw_images)
    content.append({"type": "text", "text": text})
    return [
        {
            "role": "system",
            "content": (
                "You are a strict verifier for process-level multimodal reasoning data. "
                "Audit visual fidelity, reasoning faithfulness, and semantic answer correctness."
            ),
        },
        {"role": "user", "content": content},
    ]


def generate_verified_steps(question: str, answer: str, relabeled: Dict[str, Any], raw_images: List[Any], llm_client: Any, skip_generation: bool) -> Tuple[Optional[List[str]], str, Dict[str, Any]]:
    audit_fields = ["visual_fidelity", "reasoning_faithfulness", "semantic_answer_correctness"]
    if skip_generation:
        solution = normalize_text(relabeled.get("short_solution"))
        return [solution or "(Placeholder dense trace due to skip_generation.)"], "skip_placeholder", {
            "verdict": "SKIP",
            "reason": "skip_generation",
            **{field: None for field in audit_fields},
        }
    if llm_client is None:
        return None, "missing_llm_client", verifier_audit_dict(None, audit_fields, "missing_llm_client")
    last_audit = verifier_audit_dict(None, audit_fields, "trajectory_verifier_failed")
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.send_stable_request(build_generation_messages(question, answer, relabeled, raw_images), temperature=0.0)
            parsed = extract_json(response)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("steps"), list):
                continue
            steps = [normalize_text(step) for step in parsed.get("steps", []) if normalize_text(step)]
            pred_answer = clean_answer_text(parsed.get("final_answer"))
            if not steps or not pred_answer:
                continue
            review = llm_client.send_stable_request(
                build_trajectory_verifier_messages(question, answer, pred_answer, steps, relabeled, raw_images),
                temperature=0.0,
            )
            review_parsed = extract_json(review)
            if isinstance(review_parsed, dict):
                last_audit = verifier_audit_dict(review_parsed, audit_fields)
            if (
                isinstance(review_parsed, dict)
                and str(review_parsed.get("verdict", "")).upper() == "PASS"
                and review_parsed.get("visual_fidelity") is True
                and review_parsed.get("reasoning_faithfulness") is True
                and review_parsed.get("semantic_answer_correctness") is True
            ):
                return steps, "gpt_verified", last_audit
        except Exception as exc:
            print(f"[Generation Error] attempt={attempt + 1}: {exc}")
            time.sleep(1)
    return None, "trajectory_verifier_failed", last_audit


def build_prompt(question: str) -> List[Dict[str, str]]:
    prompt_content = DENSE_PROMPT_TEMPLATE_VL.replace("{Question}", question)
    if "<image>" not in prompt_content:
        prompt_content = "<image>\n" + prompt_content
    return [{"role": "user", "content": prompt_content}]


def build_reject(index: int, split: str, item: Dict[str, Any], reason: str, relabeled: Optional[Dict[str, Any]] = None, bench_relevance: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "id": item.get("id"),
        "split": split,
        "original_index": index,
        "reject_reason": reason,
        "figure_type": metadata.get("figure_type"),
        "bench_relevance": bench_relevance or {},
        "image_type": (relabeled or {}).get("image_type"),
        "visual_dependency": (relabeled or {}).get("visual_dependency"),
    }


def process_single_item(
    item: Dict[str, Any],
    index: int,
    split: str,
    dataset_name: str,
    dataset_config: str,
    llm_client: Any,
    skip_generation: bool,
    min_bench_score: int,
    target_benches: List[str],
    allow_text_only: bool,
    keep_source_code: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    source_qa = extract_source_qa(item)
    raw_images = [item.get("image")] if item.get("image") is not None else []
    images_field = build_image_records(raw_images)
    if not images_field:
        return None, build_reject(index, split, item, "missing_or_unencodable_image")

    relabeled, relabel_mode = relabel_item(item, source_qa, llm_client, skip_generation)
    if not relabeled:
        return None, build_reject(index, split, item, relabel_mode)

    answer = clean_answer_text(relabeled.get("final_answer"))
    question = normalize_text(relabeled.get("clean_question"))
    if not question or not answer:
        return None, build_reject(index, split, item, "empty_relabel_question_or_answer", relabeled)

    answer_ok, answer_reason, answer_audit = verify_answer_sample(item, relabeled, source_qa, llm_client, skip_generation)
    if not answer_ok:
        return None, build_reject(index, split, item, answer_reason, relabeled)

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    bench_relevance = compute_bench_relevance(relabeled, metadata)
    bench_ok, bench_reason = passes_bench_filter(relabeled, bench_relevance, target_benches, min_bench_score, allow_text_only)
    if not bench_ok:
        return None, build_reject(index, split, item, bench_reason, relabeled, bench_relevance)

    steps, step_mode, trajectory_audit = generate_verified_steps(question, answer, relabeled, raw_images, llm_client, skip_generation)
    if not steps:
        return None, build_reject(index, split, item, step_mode, relabeled, bench_relevance)

    extra_info: Dict[str, Any] = {
        "answer": answer,
        "problem": question,
        "transcribed_problem": relabeled.get("transcribed_problem", ""),
        "short_solution": relabeled.get("short_solution", ""),
        "index": index,
        "original_index": index,
        "id": item.get("id"),
        "dataset": "cosyn_math",
        "hf_dataset": dataset_name,
        "hf_config": dataset_config,
        "split": split,
        "data_type": "vision",
        "num_images": len(images_field),
        "source_question_weak_reference": source_qa.get("question"),
        "source_answer_weak_reference": source_qa.get("answer"),
        "source_explanation_weak_reference": source_qa.get("explanation"),
        "metadata": metadata,
        "image_type": relabeled.get("image_type"),
        "needs_visual": relabeled.get("needs_visual"),
        "visual_dependency": relabeled.get("visual_dependency"),
        "capability_tags": relabeled.get("capability_tags"),
        "math_domain": relabeled.get("math_domain"),
        "difficulty": relabeled.get("difficulty"),
        "confidence": relabeled.get("confidence"),
        "bench_relevance": bench_relevance,
        "target_benches": target_benches,
        "relabel_mode": relabel_mode,
        "answer_verifier_reason": answer_reason,
        "answer_verifier_visual_grounding": answer_audit.get("visual_grounding"),
        "answer_verifier_answer_correct": answer_audit.get("answer_correct"),
        "answer_verifier_image_question_consistent": answer_audit.get("image_question_consistent"),
        "answer_verifier_not_ambiguous_or_broken": answer_audit.get("not_ambiguous_or_broken"),
        "answer_verifier_no_unstated_assumptions": answer_audit.get("no_unstated_assumptions"),
        "answer_verifier_verdict": answer_audit.get("verdict"),
        "step_generation_mode": step_mode,
        "trajectory_verifier_reason": trajectory_audit.get("reason"),
        "trajectory_verifier_visual_fidelity": trajectory_audit.get("visual_fidelity"),
        "trajectory_verifier_reasoning_faithfulness": trajectory_audit.get("reasoning_faithfulness"),
        "trajectory_verifier_semantic_answer_correctness": trajectory_audit.get("semantic_answer_correctness"),
        "trajectory_verifier_verdict": trajectory_audit.get("verdict"),
    }
    if keep_source_code:
        extra_info["source_code"] = item.get("code")
        extra_info["source_data"] = item.get("data")

    result = {
        "data_source": "cosyn_math",
        "prompt": build_prompt(question),
        "ability": "visual_math",
        "reward_model": {"ground_truth": json.dumps({"answer": answer, "steps": steps}, ensure_ascii=False), "style": "rule"},
        "extra_info": extra_info,
        "images": images_field,
    }
    return result, None


def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda record: record.get("extra_info", {}).get("original_index", -1))


def iter_dataset_rows(dataset: Any, limit: Optional[int], processed_indices: Set[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield rows lazily so full splits do not materialize row objects."""
    max_rows = len(dataset) if limit is None else min(limit, len(dataset))
    for idx in range(max_rows):
        if idx in processed_indices:
            continue
        yield idx, dataset[idx]


def iter_dataset_batches(
    dataset: Any,
    limit: Optional[int],
    processed_indices: Set[int],
    submit_batch_size: int,
) -> Iterator[List[Tuple[int, Dict[str, Any]]]]:
    batch: List[Tuple[int, Dict[str, Any]]] = []
    for row in iter_dataset_rows(dataset, limit, processed_indices):
        batch.append(row)
        if len(batch) >= submit_batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_hf_split(dataset_name: str, dataset_config: str, split: str, limit: Optional[int], use_streaming: bool = False):
    # Prefer HF split slicing over streaming for small experiments. The CoSyn image
    # streaming iterator can abort at interpreter finalization in some PIL/pyarrow
    # environments, while split slicing keeps smoke tests deterministic.
    split_expr = f"{split}[:{limit}]" if limit is not None else split
    return load_dataset(dataset_name, dataset_config, split=split_expr, streaming=use_streaming)


def write_rejects(reject_file: str, rejects: List[Dict[str, Any]], append: bool) -> None:
    if not rejects:
        return
    os.makedirs(os.path.dirname(reject_file) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(reject_file, mode, encoding="utf-8") as f:
        for record in rejects:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def accepted_parts_dir(output_file: str) -> str:
    return f"{output_file}.parts"


def iter_part_files(output_file: str) -> List[str]:
    parts_dir = accepted_parts_dir(output_file)
    if not os.path.isdir(parts_dir):
        return []
    return [os.path.join(parts_dir, name) for name in sorted(os.listdir(parts_dir)) if name.endswith(".parquet")]


def extract_original_index_from_record(record: Dict[str, Any]) -> Optional[int]:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        idx = extra_info.get("original_index")
        try:
            return int(idx) if idx is not None else None
        except Exception:
            return None
    return None


def read_accepted_indices_from_parquet(path: str) -> Set[int]:
    indices: Set[int] = set()
    if not os.path.exists(path):
        return indices
    try:
        # Read only extra_info where possible; this avoids loading image payloads.
        df = pd.read_parquet(path, columns=["extra_info"])
        for record in df.to_dict("records"):
            idx = extract_original_index_from_record(record)
            if idx is not None:
                indices.add(idx)
    except Exception as exc:
        print(f"Warning: failed to read accepted indices from {path}: {exc}")
    return indices


def read_reject_indices_and_reasons(reject_file: str) -> Tuple[Set[int], Counter]:
    indices: Set[int] = set()
    reasons: Counter = Counter()
    if not os.path.exists(reject_file):
        return indices, reasons
    try:
        with open(reject_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                idx = record.get("original_index")
                try:
                    if idx is not None:
                        indices.add(int(idx))
                except Exception:
                    pass
                reasons[normalize_text(record.get("reject_reason")) or "unknown"] += 1
    except Exception as exc:
        print(f"Warning: failed to read rejected cache {reject_file}: {exc}")
    return indices, reasons


def next_part_path(output_file: str) -> str:
    parts_dir = accepted_parts_dir(output_file)
    os.makedirs(parts_dir, exist_ok=True)
    existing = iter_part_files(output_file)
    return os.path.join(parts_dir, f"part-{len(existing) + 1:06d}.parquet")


def write_accepted_part(output_file: str, accepted: List[Dict[str, Any]]) -> Optional[str]:
    if not accepted:
        return None
    part_path = next_part_path(output_file)
    pd.DataFrame(sort_results(accepted)).to_parquet(part_path, index=False)
    return part_path


def parquet_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        if pq is not None:
            return int(pq.ParquetFile(path).metadata.num_rows)
        return int(len(pd.read_parquet(path, columns=["extra_info"])))
    except Exception:
        return 0


def consolidate_accepted_parquets(output_file: str, include_existing_output: bool) -> int:
    sources: List[str] = []
    if include_existing_output and os.path.exists(output_file):
        sources.append(output_file)
    sources.extend(iter_part_files(output_file))
    sources = [path for path in sources if os.path.exists(path)]
    if not sources:
        return 0

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if len(sources) == 1 and sources[0] == output_file:
        return parquet_row_count(output_file)

    tmp_output = f"{output_file}.tmp"
    if pq is None:
        frames = [pd.read_parquet(path) for path in sources]
        df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df_out.empty:
            df_out = pd.DataFrame(sort_results(df_out.to_dict("records")))
        df_out.to_parquet(tmp_output, index=False)
    else:
        writer = None
        try:
            for path in sources:
                parquet_file = pq.ParquetFile(path)
                for batch in parquet_file.iter_batches(batch_size=64):
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_output, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    os.replace(tmp_output, output_file)
    return parquet_row_count(output_file)


def convert_split(
    dataset_name: str,
    dataset_config: str,
    split: str,
    output_file: str,
    reject_file: str,
    limit: Optional[int],
    skip_generation: bool,
    max_workers: int,
    ignore_cache: bool,
    min_bench_score: int,
    target_benches: List[str],
    allow_text_only: bool,
    model_name: str,
    save_rejects: bool,
    keep_source_code: bool,
    save_interval: int,
    submit_batch_size: int,
    resume_rejects: bool,
) -> Tuple[int, int]:
    print(f"Loading {dataset_name}/{dataset_config} [{split}]...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(reject_file) or ".", exist_ok=True)

    if ignore_cache:
        if os.path.exists(output_file):
            tmp_output = output_file + ".old"
            try:
                os.replace(output_file, tmp_output)
                print(f"Moved old accepted parquet to {tmp_output} for safe rebuild.")
            except Exception as exc:
                print(f"Warning: could not move old accepted parquet {output_file}: {exc}")
        parts_dir = accepted_parts_dir(output_file)
        if os.path.isdir(parts_dir):
            tmp_parts = parts_dir + ".old"
            try:
                if os.path.exists(tmp_parts):
                    shutil.rmtree(tmp_parts)
                os.replace(parts_dir, tmp_parts)
                print(f"Moved old accepted parquet parts to {tmp_parts} for safe rebuild.")
            except Exception as exc:
                print(f"Warning: could not move old accepted parts {parts_dir}: {exc}")
        if save_rejects and os.path.exists(reject_file):
            tmp_reject = reject_file + ".old"
            try:
                os.replace(reject_file, tmp_reject)
                print(f"Moved old reject file to {tmp_reject} for safe rebuild.")
            except Exception as exc:
                print(f"Warning: could not move old reject file {reject_file}: {exc}")

    processed_indices: Set[int] = set()
    accepted_cache_indices: Set[int] = set()
    if not ignore_cache:
        accepted_cache_indices.update(read_accepted_indices_from_parquet(output_file))
        for part_file in iter_part_files(output_file):
            accepted_cache_indices.update(read_accepted_indices_from_parquet(part_file))
        processed_indices.update(accepted_cache_indices)
        if accepted_cache_indices:
            print(f"Loaded {len(accepted_cache_indices)} accepted original_index values from {output_file} and parts cache.")

    cached_reject_reasons: Counter = Counter()
    if save_rejects and resume_rejects and not ignore_cache:
        reject_indices, cached_reject_reasons = read_reject_indices_and_reasons(reject_file)
        processed_indices.update(reject_indices)
        if reject_indices:
            print(f"Loaded {len(reject_indices)} rejected original_index values from cache {reject_file}.")
    elif save_rejects and (ignore_cache or not os.path.exists(reject_file)):
        open(reject_file, "w", encoding="utf-8").close()

    dataset = load_hf_split(dataset_name, dataset_config, split, limit=limit, use_streaming=False)
    max_scan_rows = len(dataset) if limit is None else min(limit, len(dataset))
    cached_in_scan = sum(1 for idx in processed_indices if 0 <= idx < max_scan_rows)
    rows_to_process_count = max_scan_rows - cached_in_scan
    print(
        f"Rows to process in {split}: {rows_to_process_count} / {max_scan_rows} "
        f"(workers={max_workers}, skip_generation={skip_generation}, submit_batch_size={submit_batch_size}, "
        f"save_interval={save_interval})"
    )
    print(f"Output accepted parquet: {output_file}")
    print(f"Output rejected jsonl: {reject_file if save_rejects else 'disabled'}")
    if rows_to_process_count <= 0:
        total_accepted = consolidate_accepted_parquets(output_file, include_existing_output=(not ignore_cache))
        total_rejected = len(processed_indices - accepted_cache_indices) if save_rejects else 0
        print(f"No uncached rows for {split}; accepted={total_accepted}, rejected_cache_or_run={total_rejected}")
        return total_accepted, total_rejected

    llm_client = None if skip_generation else init_llm_client(model_name)
    accepted_buffer: List[Dict[str, Any]] = []
    rejects_buffer: List[Dict[str, Any]] = []
    reason_counter: Counter = Counter(cached_reject_reasons)
    accepted_this_run = 0
    rejected_this_run = 0
    completed_this_run = 0
    completed_since_save = 0
    saved_part_count = 0

    def flush_checkpoint(force: bool = False) -> None:
        nonlocal accepted_buffer, rejects_buffer, completed_since_save, saved_part_count
        if not force and completed_since_save < save_interval and len(accepted_buffer) < save_interval and len(rejects_buffer) < save_interval:
            return
        wrote_any = False
        part_path = write_accepted_part(output_file, accepted_buffer)
        if part_path:
            saved_part_count += 1
            wrote_any = True
            print(f"Checkpoint: saved {len(accepted_buffer)} accepted rows to {part_path}")
            accepted_buffer = []
        if save_rejects and rejects_buffer:
            write_rejects(reject_file, rejects_buffer, append=True)
            wrote_any = True
            print(f"Checkpoint: appended {len(rejects_buffer)} rejected rows to {reject_file}")
            rejects_buffer = []
        if wrote_any or force:
            print(
                f"Progress [{split}]: completed_this_run={completed_this_run}, accepted_this_run={accepted_this_run}, "
                f"rejected_this_run={rejected_this_run}, cached_processed={cached_in_scan}, "
                f"top_reject_reasons={reason_counter.most_common(8)}"
            )
        completed_since_save = 0

    batch_iter = iter_dataset_batches(dataset, limit, processed_indices, max(1, submit_batch_size))
    for batch in batch_iter:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_item,
                    item,
                    idx,
                    split,
                    dataset_name,
                    dataset_config,
                    llm_client,
                    skip_generation,
                    min_bench_score,
                    target_benches,
                    allow_text_only,
                    keep_source_code,
                ): idx
                for idx, item in batch
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    result, reject = future.result()
                    if result:
                        accepted_buffer.append(result)
                        accepted_this_run += 1
                    if reject:
                        rejects_buffer.append(reject)
                        rejected_this_run += 1
                        reason_counter[normalize_text(reject.get("reject_reason")) or "unknown"] += 1
                except Exception as exc:
                    reject = {"id": None, "split": split, "original_index": idx, "reject_reason": f"exception:{exc}", "figure_type": None, "bench_relevance": {}}
                    rejects_buffer.append(reject)
                    rejected_this_run += 1
                    reason_counter[reject["reject_reason"]] += 1
                finally:
                    processed_indices.add(idx)
                    completed_this_run += 1
                    completed_since_save += 1
                    flush_checkpoint(force=False)
        flush_checkpoint(force=True)

    flush_checkpoint(force=True)
    total_accepted = consolidate_accepted_parquets(output_file, include_existing_output=(not ignore_cache))
    total_rejected = 0
    if save_rejects and os.path.exists(reject_file):
        reject_indices, reason_counter = read_reject_indices_and_reasons(reject_file)
        total_rejected = len(reject_indices)
    print(
        f"Finished {split}: accepted_total={total_accepted}, rejected_total={total_rejected}, "
        f"completed_this_run={completed_this_run}, accepted_this_run={accepted_this_run}, "
        f"rejected_this_run={rejected_this_run}, saved_parts={saved_part_count}"
    )
    print(f"Reject reason counter [{split}]: {reason_counter.most_common()}")
    print(f"Saved accepted parquet to {output_file}")
    if save_rejects:
        print(f"Saved rejected jsonl to {reject_file}")
    return total_accepted, total_rejected

def convert_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    splits: Optional[List[str]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    skip_generation: bool = False,
    max_workers: Optional[int] = None,
    ignore_cache: bool = False,
    min_bench_score: int = 3,
    target_benches: Optional[List[str]] = None,
    allow_text_only: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    save_rejects: bool = True,
    keep_source_code: bool = False,
    save_interval: int = 20,
    submit_batch_size: int = 256,
    resume_rejects: bool = True,
) -> Dict[str, Dict[str, int]]:
    requested_splits = splits or DEFAULT_SPLITS
    target_benches = target_benches or DEFAULT_TARGET_BENCHES
    max_workers_resolved = resolve_max_workers(max_workers, skip_generation)
    available_splits = get_dataset_split_names(dataset_name, config_name=dataset_config)
    valid_splits = [split for split in requested_splits if split in available_splits]
    for split in requested_splits:
        if split not in available_splits:
            print(f"Warning: split '{split}' not found in {dataset_name}/{dataset_config}; skipping.")
    if not valid_splits:
        raise ValueError(f"No valid splits requested. Available: {available_splits}")
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    for split in valid_splits:
        output_file = os.path.join(output_dir, f"{split}.parquet")
        reject_file = os.path.join(output_dir, f"{split}.rejected.jsonl")
        accepted, rejected = convert_split(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            output_file=output_file,
            reject_file=reject_file,
            limit=limit,
            skip_generation=skip_generation,
            max_workers=max_workers_resolved,
            ignore_cache=ignore_cache,
            min_bench_score=min_bench_score,
            target_benches=target_benches,
            allow_text_only=allow_text_only,
            model_name=model_name,
            save_rejects=save_rejects,
            keep_source_code=keep_source_code,
            save_interval=save_interval,
            submit_batch_size=submit_batch_size,
            resume_rejects=resume_rejects,
        )
        summary[split] = {"accepted": accepted, "rejected": rejected}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert allenai/CoSyn-400K math to VERL visual math parquet.")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows scanned per split")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true", help="Skip GPT relabel/verification/trace calls and use deterministic placeholders")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--min_bench_score", type=int, default=3)
    parser.add_argument("--target_benches", nargs="+", default=DEFAULT_TARGET_BENCHES)
    parser.add_argument("--allow_text_only", action="store_true")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--save_rejects", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_rejects", action=argparse.BooleanOptionalAction, default=True, help="Skip original_index values already present in rejected jsonl when resuming")
    parser.add_argument("--save_interval", type=int, default=20, help="Checkpoint after this many completed futures or accepted/rejected buffered rows")
    parser.add_argument("--submit_batch_size", type=int, default=256, help="Number of dataset rows to materialize and submit to the executor at a time")
    parser.add_argument("--keep_source_code", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = convert_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        splits=args.splits,
        output_dir=args.output_dir,
        limit=args.limit,
        skip_generation=args.skip_generation,
        max_workers=args.max_workers,
        ignore_cache=args.ignore_cache,
        min_bench_score=args.min_bench_score,
        target_benches=args.target_benches,
        allow_text_only=args.allow_text_only,
        model_name=args.model_name,
        save_rejects=args.save_rejects,
        keep_source_code=args.keep_source_code,
        save_interval=args.save_interval,
        submit_batch_size=args.submit_batch_size,
        resume_rejects=args.resume_rejects,
    )
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))
