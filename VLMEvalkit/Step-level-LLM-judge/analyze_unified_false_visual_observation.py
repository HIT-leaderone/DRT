import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

import analyze_visual_hallucination as base


ANALYSIS_VERSION = "unified_false_visual_observation_v1"
DEFAULT_OUTPUT_DIR = Path("./fig")
QWEN_REASONING_MODELS = {"Qwen-Standard"}
TRANSIENT_JUDGE_ERROR_MARKERS = (
    "429",
    "qpm limit",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "timeout",
    "connection error",
)


UNIFIED_FALSE_VISUAL_OBSERVATION_PROMPT = """
You are a strict false-visual-observation judge for a visual reasoning benchmark.

Scope:
- Evaluate ONLY the Student Text To Evaluate below.
- The text may be a full reasoning trace or a separate visual block. Apply exactly the same criteria in both cases.
- Count unique visual observations in the text, then count how many of those unique observations are clearly false.
- A visual observation is a claim that reads or describes information from the image, diagram, chart, table, or visual options.

Count as visual observations:
- Objects, labels, options, chart/table values, axes, colors, counts, positions, geometric marks, and spatial relations.
- Visual quantities such as angles, lengths, counts, read-off values, coordinates, table cells, or bar heights.
- Visual relations such as parallel, perpendicular, equal, tangent, cyclic, collinear, midpoint, order, containment, or relative position.

Count a visual observation as hallucinated ONLY when it is clearly false relative to the Original Problem or Reference GT Steps:
- Misread label/value/quantity/choice/table cell/axis/object/color/geometry mark.
- Fabricated visual object, diagram structure, chart/table entry, or option content.
- Wrong spatial/geometric relation contradicted by the problem or GT steps.
- A visual quantity/value that conflicts with a supported value.

Deduplication:
- Count the same visual observation only once, even if it appears multiple times or is paraphrased.
- If the same unique visual observation appears multiple times and any occurrence is clearly false, count that unique observation as hallucinated once.
- Do not count multiple spans for the same underlying observation as multiple hallucinated observations.

Do NOT count:
- Perception omissions or incomplete visual coverage.
- A visual detail that is merely unverified by GT but not contradicted.
- Pure algebra, pure deductive gaps, formula mistakes, or unsupported non-visual reasoning.
- A wrong final answer by itself.
- Restating explicit givens from the problem, unless the restatement changes the visual fact.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Text Source:
{text_source}

Student Text To Evaluate:
{text_to_evaluate}

Output strict JSON only:
- Return only one JSON object. Do not use markdown fences.
- Keep quote fields single-line and under 240 characters.
- Include at most 8 representative hallucinated observation objects. Counts should still count all unique observations.
{{
  "visual_observation_count": 0,
  "hallucinated_visual_observation_count": 0,
  "visual_hallucination_present": false,
  "hallucinated_visual_observations": [
    {{
      "quote": "verbatim false visual observation from Student Text To Evaluate",
      "normalized_observation": "canonical short form of the unique visual observation",
      "type": "misread_label|wrong_quantity|fabricated_visual_object|wrong_spatial_relation|wrong_visual_attribute|other",
      "location": "student_text",
      "explanation": "why this unique visual observation is clearly false rather than merely omitted or unverified"
    }}
  ],
  "reason": "brief explanation of the counting and deduplication"
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified false-visual-observation judge. Uses one prompt for Qwen and DRT: "
            "Qwen defaults to full reasoning text, DRT defaults to the visual block."
        )
    )
    parser.add_argument("--benchmark", choices=["mathvista", "mathverse", "logicvista", "all"], default="all")
    parser.add_argument("--model_label", required=True)
    parser.add_argument(
        "--text_scope",
        choices=["auto", "reasoning", "visual"],
        default="auto",
        help="auto uses reasoning for Qwen-Standard and visual for DRT/ablation models.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Output/cache namespace. Defaults to '<model_label>_<resolved_text_scope>'.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge_backend", choices=["openai", "router"], default="openai")
    parser.add_argument("--reward_router_address", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="auto")
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--judge_max_tokens", type=int, default=1536)
    parser.add_argument("--judge_request_retries", type=int, default=6)
    parser.add_argument("--retry_base_sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--max_problem_chars", type=int, default=5000)
    parser.add_argument("--max_gt_steps_chars", type=int, default=6000)
    parser.add_argument("--max_reason_chars", type=int, default=12000)
    parser.add_argument("--max_visual_chars", type=int, default=6000)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_").lower()


def resolve_text_scope(model_label: str, requested_scope: str) -> str:
    if requested_scope != "auto":
        return requested_scope
    if model_label in QWEN_REASONING_MODELS:
        return "reasoning"
    return "visual"


def default_run_name(model_label: str, text_scope: str) -> str:
    return sanitize_name(f"{model_label}_{text_scope}_unified_false_visual")


def load_cache(cache_path: Path, ignore_cache=False):
    if ignore_cache or not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, cache_data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def is_retryable_judge_error(error: str) -> bool:
    error_text = str(error or "").lower()
    return any(marker in error_text for marker in TRANSIENT_JUDGE_ERROR_MARKERS)


def is_usable_cached_result(result) -> bool:
    if result is None:
        return False
    if not isinstance(result, dict):
        return True
    return not is_retryable_judge_error(result.get("judge_error", ""))


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_observation_spans(value):
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            quote = str(item.get("quote", "") or "").strip()
            observation = str(item.get("normalized_observation", "") or "").strip()
            span_type = str(item.get("type", "other") or "other").strip()
            location = str(item.get("location", "") or "").strip()
            explanation = str(item.get("explanation", "") or "").strip()
        else:
            quote = str(item or "").strip()
            observation = ""
            span_type = "other"
            location = ""
            explanation = ""
        if quote or observation or explanation:
            normalized.append(
                {
                    "quote": quote[:1000],
                    "normalized_observation": observation[:500],
                    "type": span_type[:120],
                    "location": location[:120],
                    "explanation": explanation[:1200],
                }
            )
    return normalized


class UnifiedFalseVisualObservationAnalyzer(base.VisualHallucinationAnalyzer):
    def __init__(self, text_scope: str, judge_request_retries: int = 6, retry_base_sleep: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.text_scope = text_scope
        self.judge_request_retries = max(0, judge_request_retries)
        self.retry_base_sleep = max(0.0, retry_base_sleep)

    @staticmethod
    def _default_result(raw_response="", parse_ok=False, error=""):
        return {
            "visual_observation_count": 0,
            "hallucinated_visual_observation_count": 0,
            "visual_hallucination_present": False,
            "hallucinated_visual_observations": [],
            "judge_reason": "",
            "judge_raw_response": raw_response,
            "judge_parse_ok": parse_ok,
            "judge_error": error,
        }

    def _normalize_result(self, result, raw_response):
        if not isinstance(result, dict):
            return self._default_result(raw_response=raw_response, parse_ok=False, error="judge response is not JSON")

        spans = normalize_observation_spans(result.get("hallucinated_visual_observations", []))
        hallucinated_count = base._coerce_int(
            result.get("hallucinated_visual_observation_count", len(spans)),
            default=len(spans),
        )
        if hallucinated_count == 0 and spans:
            hallucinated_count = len(spans)

        observation_count = base._coerce_int(result.get("visual_observation_count", 0), default=0)
        observation_count = max(observation_count, hallucinated_count)

        present = base._coerce_bool(
            result.get("visual_hallucination_present", hallucinated_count > 0),
            default=hallucinated_count > 0,
        )
        if hallucinated_count > 0:
            present = True
        if present and hallucinated_count == 0:
            hallucinated_count = max(1, len(spans))
            observation_count = max(observation_count, hallucinated_count)

        return {
            "visual_observation_count": observation_count,
            "hallucinated_visual_observation_count": hallucinated_count,
            "visual_hallucination_present": present,
            "hallucinated_visual_observations": spans,
            "judge_reason": str(result.get("reason", "") or "").strip()[:2000],
            "judge_raw_response": raw_response,
            "judge_parse_ok": True,
            "judge_error": "",
        }

    def _sample_text(self, sample):
        if self.text_scope == "reasoning":
            return "reasoning_trace", base.clip_text(sample.get("pred_reasoning", "") or "", self.max_reason_chars)
        if self.text_scope == "visual":
            return "visual_block", base.clip_text(sample.get("pred_visual", "") or "", self.max_visual_chars)
        raise ValueError(f"Unsupported text scope: {self.text_scope}")

    def _build_prompt(self, sample):
        gt_steps_str = base.clip_text(base.format_gt_steps(sample["gt_steps"]), self.max_gt_steps_chars)
        problem_text = base.clip_text(sample.get("problem", "") or "N/A", self.max_problem_chars)
        text_source, text_to_evaluate = self._sample_text(sample)
        return UNIFIED_FALSE_VISUAL_OBSERVATION_PROMPT.format(
            problem_text=problem_text,
            gt_steps_str=gt_steps_str,
            text_source=text_source,
            text_to_evaluate=text_to_evaluate,
        )

    async def judge_sample(self, sample):
        async with self.semaphore:
            prompt = self._build_prompt(sample)
            last_response_text = ""
            last_error = ""
            for request_attempt_idx in range(self.judge_request_retries + 1):
                try:
                    if request_attempt_idx > 0:
                        sleep_seconds = min(30.0, self.retry_base_sleep * (2 ** (request_attempt_idx - 1)))
                        await asyncio.sleep(sleep_seconds)
                    last_response_text = ""
                    last_error = ""
                    if request_attempt_idx > 0:
                        print(
                            f"[Judge Retry] scope={self.text_scope} benchmark={sample.get('benchmark')} "
                            f"model={sample.get('model_label')} index={sample.get('eval_index')} "
                            f"attempt={request_attempt_idx + 1}/{self.judge_request_retries + 1}"
                        )

                    prompt_for_request = prompt
                    for attempt_idx in range(2):
                        attempt_prompt = prompt_for_request
                        if attempt_idx > 0:
                            attempt_prompt = (
                                prompt_for_request
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
                    last_error = str(exc)
                    if is_retryable_judge_error(last_error) and request_attempt_idx < self.judge_request_retries:
                        continue
                    print(
                        f"[Judge Error] scope={self.text_scope} benchmark={sample.get('benchmark')} "
                        f"model={sample.get('model_label')} index={sample.get('eval_index')} error={exc}"
                    )
                    return self._default_result(parse_ok=False, error=last_error)

            return self._default_result(parse_ok=False, error=last_error)


def span_types(spans):
    return ",".join(sorted({span.get("type", "other") for span in spans if span.get("type")}))


def enrich_metrics(sample: Dict[str, Any], judge_result: Dict[str, Any], text_scope: str, run_name: str):
    spans = judge_result.get("hallucinated_visual_observations", []) or []
    observation_count = int(judge_result.get("visual_observation_count", 0))
    hallucinated_count = int(judge_result.get("hallucinated_visual_observation_count", 0))
    hallucination_rate = hallucinated_count / observation_count if observation_count > 0 else 0.0
    text_source = "reasoning_trace" if text_scope == "reasoning" else "visual_block"
    return {
        "analysis_version": ANALYSIS_VERSION,
        "run_name": run_name,
        "text_scope": text_scope,
        "text_source": text_source,
        "benchmark": sample["benchmark"],
        "model_label": sample["model_label"],
        "eval_index": sample["eval_index"],
        "gt_step_count": int(sample["gt_step_count"]),
        "visual_hallucination_present": bool(judge_result.get("visual_hallucination_present", False)),
        "visual_observation_count": observation_count,
        "hallucinated_visual_observation_count": hallucinated_count,
        "observation_level_visual_hallucination_rate": hallucination_rate,
        "hallucinated_visual_observation_types": span_types(spans),
        "hallucinated_visual_observations": spans,
        # Compatibility aliases for existing analysis helpers.
        "perception_issue_present": bool(judge_result.get("visual_hallucination_present", False)),
        "perception_issue_count": hallucinated_count,
        "perception_issue_types": span_types(spans),
        "perception_issue_spans": spans,
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
        "pred_visual_token_count": base.short_cot_qwen.estimate_tokens(sample.get("pred_visual", "")),
        "pred_think_token_count": base.short_cot_qwen.estimate_tokens(sample.get("pred_think", "")),
        "pred_reasoning_token_count": base.short_cot_qwen.estimate_tokens(sample.get("pred_reasoning", "")),
    }


async def run_analysis(samples, analyzer, cache_path: Path, text_scope: str, run_name: str, ignore_cache=False):
    cache_data = load_cache(cache_path, ignore_cache=ignore_cache)
    results = []
    pending = []

    for sample in samples:
        cache_key = (
            f"{ANALYSIS_VERSION}::{run_name}::{text_scope}::"
            f"{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
        )
        cached = cache_data.get(cache_key)
        if is_usable_cached_result(cached):
            results.append(cached)
            continue
        pending.append((cache_key, sample))

    if pending:
        analyzer.semaphore = asyncio.Semaphore(analyzer.max_concurrency)
        progress = tqdm(total=len(pending), desc=sample_desc(samples))
        save_interval = 20
        completed_since_save = 0

        async def _run_single(cache_key, sample):
            judge_result = await analyzer.judge_sample(sample)
            metrics = enrich_metrics(sample, judge_result, text_scope=text_scope, run_name=run_name)
            return cache_key, metrics

        tasks = [asyncio.create_task(_run_single(cache_key, sample)) for cache_key, sample in pending]
        for task in asyncio.as_completed(tasks):
            cache_key, metrics = await task
            if not is_retryable_judge_error(metrics.get("judge_error", "")):
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


def sample_desc(samples):
    if not samples:
        return "unified false visual observation"
    first = samples[0]
    return f"{first['benchmark']}:{first['model_label']}"


def selected_benchmark_configs(args):
    configs = base.benchmark_configs()
    if args.benchmark == "all":
        return [configs["mathvista"], configs["mathverse"], configs["logicvista"]]
    return [configs[args.benchmark]]


def select_model_spec(config, model_label: str):
    for spec in config.model_specs:
        if spec.label == model_label:
            return spec
    raise ValueError(f"Model label {model_label!r} not found for benchmark {config.name}.")


def build_grouped_summary(df: pd.DataFrame, group_cols: List[str]):
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            sample_count=("eval_index", "size"),
            sample_visual_hallucination_rate=("visual_hallucination_present", "mean"),
            total_visual_observation_count=("visual_observation_count", "sum"),
            total_hallucinated_visual_observation_count=("hallucinated_visual_observation_count", "sum"),
            mean_visual_observation_count=("visual_observation_count", "mean"),
            mean_hallucinated_visual_observation_count=("hallucinated_visual_observation_count", "mean"),
            macro_observation_level_visual_hallucination_rate=(
                "observation_level_visual_hallucination_rate",
                "mean",
            ),
            judge_parse_ok_rate=("judge_parse_ok", "mean"),
            mean_pred_visual_tokens=("pred_visual_token_count", "mean"),
            mean_pred_think_tokens=("pred_think_token_count", "mean"),
        )
        .sort_values(group_cols)
    )
    grouped["micro_observation_level_visual_hallucination_rate"] = (
        grouped["total_hallucinated_visual_observation_count"]
        / grouped["total_visual_observation_count"].replace(0, pd.NA)
    )
    return grouped


def csv_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    for column in ("hallucinated_visual_observations", "perception_issue_spans"):
        if column in output_df.columns:
            output_df[column] = output_df[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    if "judge_raw_response" in output_df.columns:
        output_df = output_df.drop(columns=["judge_raw_response"])
    return output_df


def run_benchmark(config, args, text_scope: str, run_name: str):
    model_spec = select_model_spec(config, args.model_label)
    base.validate_input_files([model_spec], config.gt_parquet)
    gt_records = base.load_gt_records(config.gt_parquet)
    samples = base.load_model_samples(model_spec, gt_records, benchmark=config.name, limit=args.limit)
    if not samples:
        raise ValueError(f"No overlapping samples found for benchmark={config.name}, model={args.model_label}.")

    cache_path = args.output_dir / f"{ANALYSIS_VERSION}_{run_name}_{config.name}_cache.json"
    cache_data = load_cache(cache_path, ignore_cache=args.ignore_cache)
    pending_count = 0
    for sample in samples:
        cache_key = (
            f"{ANALYSIS_VERSION}::{run_name}::{text_scope}::"
            f"{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
        )
        if not is_usable_cached_result(cache_data.get(cache_key)):
            pending_count += 1

    print(
        f"[{config.name}] run={run_name} text_scope={text_scope} model={args.model_label} "
        f"samples={len(samples)} pending={pending_count} max_concurrency={args.max_concurrency}"
    )

    analyzer = UnifiedFalseVisualObservationAnalyzer(
        text_scope=text_scope,
        judge_backend=args.judge_backend,
        reward_router_address=args.reward_router_address,
        judge_model=args.judge_model,
        max_concurrency=args.max_concurrency,
        judge_max_tokens=args.judge_max_tokens,
        judge_request_retries=args.judge_request_retries,
        retry_base_sleep=args.retry_base_sleep,
        max_problem_chars=args.max_problem_chars,
        max_gt_steps_chars=args.max_gt_steps_chars,
        max_reason_chars=args.max_reason_chars,
    )
    analyzer.max_visual_chars = args.max_visual_chars

    df = asyncio.run(
        run_analysis(
            samples,
            analyzer,
            cache_path=cache_path,
            text_scope=text_scope,
            run_name=run_name,
            ignore_cache=args.ignore_cache,
        )
    ).sort_values(["benchmark", "model_label", "eval_index"])

    prefix = f"{ANALYSIS_VERSION}_{run_name}_{config.name}"
    per_sample_path = args.output_dir / f"{prefix}_per_sample.csv"
    grouped_path = args.output_dir / f"{prefix}_grouped_by_model.csv"
    grouped_by_gt_path = args.output_dir / f"{prefix}_grouped_by_gt_step.csv"
    log_path = args.output_dir / f"{prefix}_result_log.jsonl"

    csv_ready_df(df).to_csv(per_sample_path, index=False)
    build_grouped_summary(df, ["run_name", "text_scope", "benchmark", "model_label"]).to_csv(
        grouped_path,
        index=False,
    )
    build_grouped_summary(df, ["run_name", "text_scope", "benchmark", "model_label", "gt_step_count"]).to_csv(
        grouped_by_gt_path,
        index=False,
    )
    write_jsonl(log_path, df.to_dict("records"))

    print(f"[{config.name}] cache: {cache_path}")
    print(f"[{config.name}] per-sample CSV: {per_sample_path}")
    print(f"[{config.name}] grouped CSV: {grouped_path}")
    print(f"[{config.name}] grouped-by-GT-step CSV: {grouped_by_gt_path}")
    print(f"[{config.name}] result log: {log_path}")
    return df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_scope = resolve_text_scope(args.model_label, args.text_scope)
    run_name = sanitize_name(args.run_name) if args.run_name else default_run_name(args.model_label, text_scope)

    frames = []
    for config in selected_benchmark_configs(args):
        frames.append(run_benchmark(config, args, text_scope=text_scope, run_name=run_name))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_prefix = f"{ANALYSIS_VERSION}_{run_name}_all"
    combined_per_sample = args.output_dir / f"{combined_prefix}_per_sample.csv"
    combined_grouped = args.output_dir / f"{combined_prefix}_grouped_by_model.csv"
    combined_grouped_gt = args.output_dir / f"{combined_prefix}_grouped_by_gt_step.csv"

    csv_ready_df(combined).to_csv(combined_per_sample, index=False)
    build_grouped_summary(combined, ["run_name", "text_scope", "model_label"]).to_csv(
        combined_grouped,
        index=False,
    )
    build_grouped_summary(combined, ["run_name", "text_scope", "model_label", "gt_step_count"]).to_csv(
        combined_grouped_gt,
        index=False,
    )

    print(f"[all] combined per-sample CSV: {combined_per_sample}")
    print(f"[all] combined grouped CSV: {combined_grouped}")
    print(f"[all] combined grouped-by-GT-step CSV: {combined_grouped_gt}")


if __name__ == "__main__":
    main()
