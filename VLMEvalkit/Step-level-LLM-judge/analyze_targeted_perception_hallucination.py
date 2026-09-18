import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

import analyze_visual_hallucination as base


ANALYSIS_VERSION = "targeted_perception_v1"
DEFAULT_OUTPUT_DIR = Path("./fig")


DRT_VISUAL_ERRORS_PROMPT = """
You are a strict visual-extraction error judge for a visual reasoning benchmark.

Scope:
- Evaluate ONLY the Student Visual Block.
- Ignore the Student Reasoning and Student Final Answer except as context for identifying the task.
- Count only visual claims in the visual block that are clearly WRONG relative to the Original Problem or Reference GT Steps.
- Do NOT count a visual detail merely because it is extra or not mentioned in the GT steps.
- Do NOT count an unverified but plausible visual detail unless it contradicts the problem/GT or changes a supported visual fact.

Count as visual extraction errors:
- Misread labels, choices, chart/table values, axes, colors, objects, counts, positions, or geometric marks.
- Fabricated visual objects or structure contradicted by the problem/GT.
- Wrong spatial/geometric relations contradicted by the problem/GT.
- A visual quantity/value that conflicts with a supported value.

Do NOT count:
- Pure math/reasoning mistakes outside the visual block.
- Extra visual descriptions that may be true but are simply not used by the reference solution.
- A wrong final answer by itself.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Visual Block:
{pred_visual}

Output strict JSON only:
- Return only one JSON object. Do not use markdown fences.
- Keep each quote single-line and under 240 characters.
- Include at most 5 representative span objects. The count should still count all issues you identify.
{{
  "perception_issue_present": false,
  "perception_issue_count": 0,
  "perception_issue_spans": [
    {{
      "quote": "verbatim wrong visual claim from Student Visual Block",
      "type": "misread_label|wrong_quantity|fabricated_visual_object|wrong_spatial_relation|wrong_visual_attribute|other",
      "location": "visual_block",
      "explanation": "why this visual claim is clearly wrong rather than merely extra/unverified"
    }}
  ],
  "reason": "brief explanation"
}}
"""


QWEN_PERCEPTION_STEPS_PROMPT = """
You are a strict perception-premise judge for a visual reasoning benchmark.

Scope:
- Evaluate the Student Reasoning Trace step by step.
- Count only claims that involve perception or reading information from the image/diagram/chart/table.
- Count perception-related claims that are WRONG or UNVERIFIED.
- Include unverified quantities inferred from the image when they are stated or used as implicit premises.
- Ignore pure algebra, pure deductive gaps, formula mistakes, or unsupported non-visual reasoning unless the claim is framed as being read from the visual input.

Count as perception issues:
- A value/count/label/angle/length/readout inferred from the image without support.
- A visual relation such as parallel, perpendicular, equal, tangent, cyclic, collinear, midpoint, order, position, or chart ordering that is not supported.
- Misread labels, options, table cells, axes, object identities, colors, or geometry marks.
- Fabricated visual objects, diagram structure, chart/table entries, or visual facts.

Do NOT count:
- Pure reasoning hallucinations that do not claim to read or use a visual fact.
- A wrong final answer by itself.
- Restating explicit givens from the problem.
- A valid derived mathematical fact that does not introduce a new visual premise.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Reasoning Trace:
{pred_reasoning}

Student Final Answer:
{pred_final}

Output strict JSON only:
- Return only one JSON object. Do not use markdown fences.
- Keep each quote single-line and under 240 characters.
- Include at most 5 representative span objects. The count should still count all issues you identify.
{{
  "perception_issue_present": false,
  "perception_issue_count": 0,
  "perception_issue_spans": [
    {{
      "quote": "verbatim perceptual claim from Student Reasoning Trace",
      "type": "unverified_quantity|implicit_visual_premise|misread_label|fabricated_visual_object|unsupported_spatial_relation|unsupported_visual_attribute|other",
      "location": "reasoning_trace",
      "explanation": "why this is a wrong or unverified perceptual premise"
    }}
  ],
  "reason": "brief explanation"
}}
"""


DRT_VISUAL_ERRORS_AFFECT_REASONING_PROMPT = """
You are a strict causal visual-error judge for a visual reasoning benchmark.

Scope:
- The model is a DRT-SFT style model with a separate Student Visual Block and Student Reasoning Trace.
- Count ONLY errors that originate as observations in the Student Visual Block.
- Count an observation error ONLY IF it is clearly wrong AND is used by the Student Reasoning Trace as a premise,
  intermediate fact, option elimination basis, calculation input, or final-answer support.
- This is an answer-impact / reasoning-impact metric, not a visual-description quality metric.

Count an issue only when ALL are true:
1. The Student Visual Block contains a visual observation that is clearly wrong relative to the Original Problem or
   Reference GT Steps.
2. The same wrong observation, or a clear paraphrase of it, is used in the Student Reasoning Trace.
3. The use is material to the reasoning path or final answer. If the wrong observation were corrected or removed,
   the reasoning would likely change, become unsupported, or pick a different option/value.

Do NOT count:
- Extra visual details that are not used later.
- Visual observations that are merely unverified by GT but not contradicted.
- Wrong visual observations that are mentioned only in the visual block and ignored by the reasoning.
- Pure reasoning mistakes that do not come from a visual-block observation.
- A wrong final answer by itself.

Original Problem:
{problem_text}

Reference GT Steps:
{gt_steps_str}

Student Visual Block:
{pred_visual}

Student Reasoning Trace:
{pred_reasoning}

Student Final Answer:
{pred_final}

Output strict JSON only:
- Return only one JSON object. Do not use markdown fences.
- Keep each quote single-line and under 240 characters.
- Include at most 5 representative span objects. The count should still count all causal visual-observation errors.
{{
  "perception_issue_present": false,
  "perception_issue_count": 0,
  "perception_issue_spans": [
    {{
      "quote": "verbatim wrong visual observation from Student Visual Block",
      "type": "misread_label|wrong_quantity|fabricated_visual_object|wrong_spatial_relation|wrong_visual_attribute|other",
      "location": "visual_block_used_by_reasoning",
      "explanation": "why this visual observation is clearly wrong and how the reasoning uses it"
    }}
  ],
  "reason": "brief explanation of why the counted issues do or do not affect the reasoning/final answer"
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Targeted perception hallucination judge with separate Qwen and DRT-SFT scopes."
    )
    parser.add_argument("--benchmark", choices=["mathvista", "mathverse", "logicvista", "all"], default="all")
    parser.add_argument("--model_label", required=True)
    parser.add_argument(
        "--judge_scope",
        choices=["drt_visual_errors", "drt_visual_errors_affect_reasoning", "qwen_perception_steps"],
        required=True,
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Output/cache namespace. Defaults to '<model_label>_<judge_scope>' sanitized.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge_backend", choices=["openai", "router"], default="openai")
    parser.add_argument("--reward_router_address", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="auto")
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--judge_max_tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--max_problem_chars", type=int, default=5000)
    parser.add_argument("--max_gt_steps_chars", type=int, default=6000)
    parser.add_argument("--max_reason_chars", type=int, default=10000)
    parser.add_argument("--max_visual_chars", type=int, default=6000)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_").lower()


def default_run_name(model_label: str, judge_scope: str) -> str:
    return sanitize_name(f"{model_label}_{judge_scope}")


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


def normalize_spans(value):
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, dict):
            quote = str(item.get("quote", "") or "").strip()
            span_type = str(item.get("type", "other") or "other").strip()
            location = str(item.get("location", "") or "").strip()
            explanation = str(item.get("explanation", "") or "").strip()
        else:
            quote = str(item or "").strip()
            span_type = "other"
            location = ""
            explanation = ""
        if quote or explanation:
            normalized.append(
                {
                    "quote": quote[:1000],
                    "type": span_type[:120],
                    "location": location[:120],
                    "explanation": explanation[:1200],
                }
            )
    return normalized


class TargetedPerceptionAnalyzer(base.VisualHallucinationAnalyzer):
    def __init__(self, judge_scope: str, **kwargs):
        super().__init__(**kwargs)
        self.judge_scope = judge_scope

    @staticmethod
    def _default_result(raw_response="", parse_ok=False, error=""):
        return {
            "perception_issue_present": False,
            "perception_issue_count": 0,
            "perception_issue_spans": [],
            "judge_reason": "",
            "judge_raw_response": raw_response,
            "judge_parse_ok": parse_ok,
            "judge_error": error,
        }

    def _normalize_target_result(self, result, raw_response):
        if not isinstance(result, dict):
            return self._default_result(raw_response=raw_response, parse_ok=False, error="judge response is not JSON")

        spans = normalize_spans(result.get("perception_issue_spans", []))
        count = base._coerce_int(result.get("perception_issue_count", len(spans)), default=len(spans))
        if count == 0 and spans:
            count = len(spans)
        present = base._coerce_bool(result.get("perception_issue_present", count > 0), default=count > 0)
        if count > 0:
            present = True
        if present and count == 0:
            count = max(1, len(spans))

        return {
            "perception_issue_present": present,
            "perception_issue_count": count,
            "perception_issue_spans": spans,
            "judge_reason": str(result.get("reason", "") or "").strip()[:2000],
            "judge_raw_response": raw_response,
            "judge_parse_ok": True,
            "judge_error": "",
        }

    def _build_prompt(self, sample):
        gt_steps_str = base.clip_text(base.format_gt_steps(sample["gt_steps"]), self.max_gt_steps_chars)
        problem_text = base.clip_text(sample.get("problem", "") or "N/A", self.max_problem_chars)

        if self.judge_scope == "drt_visual_errors":
            return DRT_VISUAL_ERRORS_PROMPT.format(
                problem_text=problem_text,
                gt_steps_str=gt_steps_str,
                pred_visual=base.clip_text(sample.get("pred_visual", "") or "", self.max_visual_chars),
            )

        if self.judge_scope == "drt_visual_errors_affect_reasoning":
            return DRT_VISUAL_ERRORS_AFFECT_REASONING_PROMPT.format(
                problem_text=problem_text,
                gt_steps_str=gt_steps_str,
                pred_visual=base.clip_text(sample.get("pred_visual", "") or "", self.max_visual_chars),
                pred_reasoning=base.clip_text(sample.get("pred_reasoning", "") or "", self.max_reason_chars),
                pred_final=base.clip_text(sample.get("pred_final", "") or "", 1000),
            )

        return QWEN_PERCEPTION_STEPS_PROMPT.format(
            problem_text=problem_text,
            gt_steps_str=gt_steps_str,
            pred_reasoning=base.clip_text(sample.get("pred_reasoning", "") or "", self.max_reason_chars),
            pred_final=base.clip_text(sample.get("pred_final", "") or "", 1000),
        )

    async def judge_sample(self, sample):
        async with self.semaphore:
            try:
                prompt = self._build_prompt(sample)
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
                    normalized = self._normalize_target_result(result, response_text)
                    if normalized["judge_parse_ok"]:
                        return normalized
                return self._default_result(
                    raw_response=last_response_text,
                    parse_ok=False,
                    error="judge response is not JSON",
                )
            except Exception as exc:
                print(
                    f"[Judge Error] scope={self.judge_scope} benchmark={sample.get('benchmark')} "
                    f"model={sample.get('model_label')} index={sample.get('eval_index')} error={exc}"
                )
                return self._default_result(parse_ok=False, error=str(exc))


def span_types(spans):
    return ",".join(sorted({span.get("type", "other") for span in spans if span.get("type")}))


def enrich_metrics(sample: Dict[str, Any], judge_result: Dict[str, Any], judge_scope: str, run_name: str):
    spans = judge_result.get("perception_issue_spans", []) or []
    return {
        "analysis_version": ANALYSIS_VERSION,
        "run_name": run_name,
        "judge_scope": judge_scope,
        "benchmark": sample["benchmark"],
        "model_label": sample["model_label"],
        "eval_index": sample["eval_index"],
        "gt_step_count": int(sample["gt_step_count"]),
        "perception_issue_present": bool(judge_result.get("perception_issue_present", False)),
        "perception_issue_count": int(judge_result.get("perception_issue_count", 0)),
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


async def run_analysis(samples, analyzer, cache_path: Path, judge_scope: str, run_name: str, ignore_cache=False):
    cache_data = load_cache(cache_path, ignore_cache=ignore_cache)
    results = []
    pending = []

    for sample in samples:
        cache_key = (
            f"{ANALYSIS_VERSION}::{run_name}::{judge_scope}::"
            f"{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
        )
        cached = cache_data.get(cache_key)
        if cached is not None:
            results.append(cached)
            continue
        pending.append((cache_key, sample))

    if pending:
        analyzer.semaphore = asyncio.Semaphore(analyzer.max_concurrency)
        progress = tqdm(total=len(pending), desc=f"{sample_desc(samples)}")
        save_interval = 20
        completed_since_save = 0

        async def _run_single(cache_key, sample):
            judge_result = await analyzer.judge_sample(sample)
            metrics = enrich_metrics(sample, judge_result, judge_scope=judge_scope, run_name=run_name)
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


def sample_desc(samples):
    if not samples:
        return "targeted perception"
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
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            sample_count=("eval_index", "size"),
            perception_issue_rate=("perception_issue_present", "mean"),
            mean_perception_issue_count=("perception_issue_count", "mean"),
            judge_parse_ok_rate=("judge_parse_ok", "mean"),
            mean_pred_visual_tokens=("pred_visual_token_count", "mean"),
            mean_pred_think_tokens=("pred_think_token_count", "mean"),
        )
        .sort_values(group_cols)
    )


def csv_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    if "perception_issue_spans" in output_df.columns:
        output_df["perception_issue_spans"] = output_df["perception_issue_spans"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    if "judge_raw_response" in output_df.columns:
        output_df = output_df.drop(columns=["judge_raw_response"])
    return output_df


def run_benchmark(config, args, run_name: str):
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
            f"{ANALYSIS_VERSION}::{run_name}::{args.judge_scope}::"
            f"{sample['benchmark']}::{sample['model_label']}::{sample['eval_index']}"
        )
        if cache_key not in cache_data:
            pending_count += 1

    print(
        f"[{config.name}] run={run_name} scope={args.judge_scope} model={args.model_label} "
        f"samples={len(samples)} pending={pending_count} max_concurrency={args.max_concurrency}"
    )

    analyzer = TargetedPerceptionAnalyzer(
        judge_scope=args.judge_scope,
        judge_backend=args.judge_backend,
        reward_router_address=args.reward_router_address,
        judge_model=args.judge_model,
        max_concurrency=args.max_concurrency,
        judge_max_tokens=args.judge_max_tokens,
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
            judge_scope=args.judge_scope,
            run_name=run_name,
            ignore_cache=args.ignore_cache,
        )
    ).sort_values(["benchmark", "model_label", "eval_index"])

    prefix = f"{ANALYSIS_VERSION}_{run_name}_{config.name}"
    per_sample_path = args.output_dir / f"{prefix}_per_sample.csv"
    grouped_path = args.output_dir / f"{prefix}_grouped_by_model.csv"
    log_path = args.output_dir / f"{prefix}_result_log.jsonl"

    csv_ready_df(df).to_csv(per_sample_path, index=False)
    build_grouped_summary(df, ["run_name", "judge_scope", "benchmark", "model_label"]).to_csv(grouped_path, index=False)
    write_jsonl(log_path, df.to_dict("records"))

    print(f"[{config.name}] cache: {cache_path}")
    print(f"[{config.name}] per-sample CSV: {per_sample_path}")
    print(f"[{config.name}] grouped CSV: {grouped_path}")
    print(f"[{config.name}] result log: {log_path}")
    return df


def write_all_outputs(frames: List[pd.DataFrame], output_dir: Path, run_name: str):
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).sort_values(["benchmark", "model_label", "eval_index"])
    prefix = f"{ANALYSIS_VERSION}_{run_name}_all"
    per_sample_path = output_dir / f"{prefix}_per_sample.csv"
    grouped_benchmark_path = output_dir / f"{prefix}_grouped_by_benchmark_model.csv"
    grouped_model_path = output_dir / f"{prefix}_grouped_by_model.csv"
    log_path = output_dir / f"{prefix}_result_log.jsonl"
    csv_ready_df(df).to_csv(per_sample_path, index=False)
    build_grouped_summary(df, ["run_name", "judge_scope", "benchmark", "model_label"]).to_csv(
        grouped_benchmark_path,
        index=False,
    )
    build_grouped_summary(df, ["run_name", "judge_scope", "model_label"]).to_csv(grouped_model_path, index=False)
    write_jsonl(log_path, df.to_dict("records"))
    print(f"[all] per-sample CSV: {per_sample_path}")
    print(f"[all] grouped benchmark/model CSV: {grouped_benchmark_path}")
    print(f"[all] grouped model CSV: {grouped_model_path}")
    print(f"[all] result log: {log_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = sanitize_name(args.run_name or default_run_name(args.model_label, args.judge_scope))
    frames = []
    for config in selected_benchmark_configs(args):
        frames.append(run_benchmark(config, args, run_name=run_name))
    write_all_outputs(frames, args.output_dir, run_name=run_name)


if __name__ == "__main__":
    main()
