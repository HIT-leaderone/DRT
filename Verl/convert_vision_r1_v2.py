import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import get_dataset_split_names, load_dataset
from PIL import Image
from tqdm import tqdm

try:
    from gpt_model import GPT4o
except ImportError:
    print("Warning: gpt_model.py not found. Please ensure it is in the Python path.")

    class GPT4o:
        def __init__(self, **kwargs):
            pass

        def send_stable_request(self, **kwargs):
            return "{}"


MODEL_NAME = "gpt-5.1-2025-11-13"
MAX_RETRIES = 5
# For GPT generation + verification under QPM=300 / TPM=120k, 4 workers is a safe default.
# Pure format conversion (`--skip_generation`) does not hit the API and can use higher parallelism.
DEFAULT_API_MAX_WORKERS = 4
DEFAULT_SKIP_GEN_MAX_WORKERS = 16

DEFAULT_DATASET_NAMES = {
    "vision_r1_rl": "Osilly/Vision-R1-rl",
    "dapo": "allenai/Dolci-Instruct-RL",
}

DEFAULT_SPLITS = {
    "vision_r1_rl": ["train", "test"],
    "dapo": ["train"],
}

DEFAULT_OUTPUT_DIRS = {
    "vision_r1_rl": "data/vision_r1_rl",
    "dapo": "data/dapo_math_17k",
}

DEFAULT_SOURCE_FILTERS = {
    "dapo": "hamishivi/DAPO-Math-17k-Processed_filtered",
}

DEFAULT_MAX_SAMPLES = {
    "dapo": 7000,
}

DENSE_PROMPT_TEMPLATE_VL = (
    "{Question}\n\n"
    "Analyze this question to provide a **Dense Cognitive Trace**.\n"
    "--- Requirements ---\n"
    "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols). NO conversational fillers.\n"
    "2. **Structure**: Your response MUST strictly follow this XML structure:\n"
    "   <visual>...concise visual evidence...</visual><think>...[Priors] -> ...compressed reasoning...</think><answer>...final answer...</answer>\n"
    "3. **Content**: Deconstruct into visual observations, logical reasoning, and the final answer."
)

DENSE_PROMPT_TEMPLATE_LLM = (
    "{Question}\n\n"
    "Analyze the given text input to provide a **Dense Cognitive Trace**.\n"
    "--- Requirements ---\n"
    "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols). NO conversational fillers.\n"
    "2. **Structure**: Your response MUST strictly follow this XML structure:\n"
    "   <evidence>...key textual evidence, constraints, or cues...</evidence>"
    "<think>...[Priors] -> ...compressed reasoning...</think>"
    "<answer>...final answer...</answer>\n"
    "3. **Content**: Extract the relevant textual evidence, infer the necessary logic, and provide the final answer."
)


def repair_json_content(text):
    return text.replace("\\", "\\\\")


def extract_json(text):
    json_str = ""
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
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


def normalize_question(problem_text, remove_image_token=False):
    text = str(problem_text or "")
    if remove_image_token:
        text = re.sub(r"\s*<image>\s*", " ", text)
    text = re.sub(r"^user:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_answer_text(answer_text):
    text = str(answer_text or "").strip()
    text = re.sub(r"^<answer>\s*", "", text)
    text = re.sub(r"\s*</answer>$", "", text)
    return text.strip()


def extract_text_from_prompt(prompt_value):
    if isinstance(prompt_value, str):
        return prompt_value

    if isinstance(prompt_value, list):
        text_parts = []
        for message in prompt_value:
            if not isinstance(message, dict):
                continue

            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        text_parts.append(chunk.get("text", ""))

        return "\n\n".join(part for part in text_parts if part)

    return str(prompt_value or "")


def extract_dapo_question(item):
    if isinstance(item.get("prompt"), str):
        return normalize_question(item.get("prompt", ""))

    source_prompt_text = extract_text_from_prompt(item.get("source_prompt"))
    if not source_prompt_text:
        source_prompt_text = extract_text_from_prompt(item.get("prompt"))

    source_prompt_text = normalize_question(source_prompt_text)
    source_prompt_text = re.sub(r"\n\nRemember, your answer.*$", "", source_prompt_text, flags=re.DOTALL)

    match = re.search(
        r"Solve the following math problem step by step\..*?\n\n(.*)$",
        source_prompt_text,
        flags=re.DOTALL,
    )
    if match:
        return normalize_question(match.group(1))

    return source_prompt_text


def pil_image_to_png_bytes(image):
    if image is None or not isinstance(image, Image.Image):
        return None
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:
        print(f"[Image Encode Error] {exc}")
        return None


def build_image_records(images):
    image_records = []
    for image in images or []:
        image_bytes = pil_image_to_png_bytes(image)
        if image_bytes:
            image_records.append({"bytes": image_bytes})
    return image_records


def build_image_url_content(images):
    image_content = []
    for image in images or []:
        image_bytes = pil_image_to_png_bytes(image)
        if not image_bytes:
            continue
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return image_content


def init_llm_client():
    api_base = os.environ.get("OPENAI_BASE_URL", None)
    api_key = os.environ.get("OPENAI_API_KEY", None)
    try:
        return GPT4o(deployment_name=MODEL_NAME, api_base=api_base, api_key=api_key)
    except Exception as exc:
        print(f"Warning: Failed to initialize LLM client: {exc}")
        return None


def resolve_dataset_name(dataset_type, dataset_name):
    return dataset_name or DEFAULT_DATASET_NAMES[dataset_type]


def resolve_output_dir(dataset_type, output_dir):
    return output_dir or DEFAULT_OUTPUT_DIRS[dataset_type]


def get_available_splits(dataset_name, dataset_config=None):
    if dataset_config:
        return get_dataset_split_names(dataset_name, config_name=dataset_config)
    return get_dataset_split_names(dataset_name)


def resolve_source_filter(dataset_type, source_filter):
    return source_filter if source_filter is not None else DEFAULT_SOURCE_FILTERS.get(dataset_type)


def resolve_max_samples(dataset_type, max_samples):
    return max_samples if max_samples is not None else DEFAULT_MAX_SAMPLES.get(dataset_type)


def resolve_max_workers(max_workers, skip_generation):
    if max_workers is not None:
        return max_workers
    if skip_generation:
        return DEFAULT_SKIP_GEN_MAX_WORKERS
    return DEFAULT_API_MAX_WORKERS


def load_hf_split(dataset_name, split, dataset_config=None, source_filter=None, max_samples=None):
    if source_filter:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)

        filtered_rows = []
        for example in dataset:
            if example.get("dataset_source") != source_filter:
                continue
            filtered_rows.append(example)
            if max_samples is not None and len(filtered_rows) >= max_samples:
                break

        print(
            f"Filtered {len(filtered_rows)} samples from {dataset_name}[{split}] "
            f"with dataset_source={source_filter}"
        )
        return filtered_rows

    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def canonicalize_item(item, index, split, dataset_type, dataset_name):
    raw_extra_info = item.get("extra_info", {}) if isinstance(item.get("extra_info"), dict) else {}

    if dataset_type == "vision_r1_rl":
        raw_images = item.get("images", []) or []
        question = normalize_question(item.get("problem", ""), remove_image_token=True)
        answer = clean_answer_text(item.get("answer", ""))
        images_field = build_image_records(raw_images)

        return {
            "question": question,
            "answer": answer,
            "raw_images": raw_images,
            "images_field": images_field,
            "data_source": "vision_r1_rl",
            "ability": "visual_reasoning",
            "prompt_template": DENSE_PROMPT_TEMPLATE_VL,
            "extra_info": {
                "answer": answer,
                "problem": question,
                "index": index,
                "original_index": index,
                "dataset": "vision_r1_rl",
                "hf_dataset": dataset_name,
                "split": split,
                "data_type": "vision",
                "num_images": len(images_field),
            },
        }

    if dataset_type == "dapo":
        question = extract_dapo_question(item)
        answer = clean_answer_text(item.get("solution", item.get("reward_model", {}).get("ground_truth", "")))
        source_index = raw_extra_info.get("index")
        data_source = str(item.get("data_source") or "math_dapo")
        ability = str(item.get("ability") or "math").lower()

        return {
            "question": question,
            "answer": answer,
            "raw_images": [],
            "images_field": [],
            "data_source": data_source,
            "ability": ability,
            "prompt_template": DENSE_PROMPT_TEMPLATE_LLM,
            "extra_info": {
                "answer": answer,
                "problem": question,
                "index": index,
                "original_index": index,
                "dataset": "dapo_math_17k",
                "hf_dataset": dataset_name,
                "split": split,
                "data_type": "llm",
                "source_index": source_index,
            },
        }

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def build_generation_messages(question, answer, raw_images):
    has_images = len(raw_images) > 0
    if has_images:
        user_content = build_image_url_content(raw_images)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Solve the following visual math problem step-by-step.\n\n"
                    f"Problem: {question}\n"
                    f"Correct Answer (for hidden validation only): {answer}\n\n"
                    "Requirements:\n"
                    "1. Use only evidence visible in the provided image(s) and text.\n"
                    "2. The Correct Answer is metadata for validation only. Do NOT mention it, quote it, "
                    "or use phrases like 'given answer' inside the steps.\n"
                    "3. Generate a concise but logically complete reasoning trace.\n"
                    "4. If a detail is not clearly supported by the image/text, omit it instead of guessing.\n"
                    "5. Prefer fewer grounded steps over speculative steps.\n"
                    "6. Output strictly in JSON format:\n"
                    "{\n"
                    '  "steps": ["Step 1...", "Step 2..."],\n'
                    '  "final_answer": "The final answer"\n'
                    "}\n"
                ),
            }
        )
        return [
            {"role": "system", "content": "You are an expert in visual geometry and mathematical reasoning."},
            {"role": "user", "content": user_content},
        ]

    user_text = (
        "Solve the following math problem step-by-step.\n\n"
        f"Problem: {question}\n"
        f"Correct Answer (for hidden validation only): {answer}\n\n"
        "Requirements:\n"
        "1. Use only evidence from the problem statement and valid math reasoning.\n"
        "2. The Correct Answer is metadata for validation only. Do NOT mention it, quote it, "
        "or use phrases like 'given answer' inside the steps.\n"
        "3. Generate a concise but logically complete reasoning trace.\n"
        "4. Output strictly in JSON format:\n"
        "{\n"
        '  "steps": ["Step 1...", "Step 2..."],\n'
        '  "final_answer": "The final answer"\n'
        "}\n"
    )
    return [
        {"role": "system", "content": "You are a mathematics expert."},
        {"role": "user", "content": user_text},
    ]


def build_verifier_messages(question, answer, pred_answer, steps, raw_images):
    has_images = len(raw_images) > 0
    verifier_text = (
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Generated Answer: {pred_answer}\n"
        f"Candidate Trajectory: {json.dumps(steps, ensure_ascii=False)}\n\n"
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
        '  "reason": "brief explanation"\n'
        "}\n"
    )

    if has_images:
        user_content = build_image_url_content(raw_images)
        user_content.append({"type": "text", "text": verifier_text})
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict verifier for process-level multimodal reasoning data. "
                    "Audit visual fidelity, reasoning faithfulness, and semantic answer correctness."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are a strict verifier for process-level reasoning data. "
                "Audit reasoning faithfulness and semantic answer correctness."
            ),
        },
        {"role": "user", "content": verifier_text},
    ]


def process_single_item(item, index, split, dataset_type, dataset_name, llm_client, skip_generation=False):
    normalized = canonicalize_item(item, index, split, dataset_type, dataset_name)
    question = normalized["question"]
    answer = normalized["answer"]
    raw_images = normalized["raw_images"]
    images_field = normalized["images_field"]
    prompt_template = normalized["prompt_template"]
    data_source = normalized["data_source"]
    ability = normalized["ability"]
    extra_info = normalized["extra_info"]

    if not question or not answer:
        print(f"Row {split}/{index}: Empty question or answer, skipping sample.")
        return None

    result = {
        "index": index,
        "data_source": data_source,
        "prompt": [],
        "ability": ability,
        "reward_model": {},
        "extra_info": {},
        "images": images_field,
        "success": False,
    }

    if skip_generation:
        steps = ["(Placeholder step due to skip_generation)"]
        step_generation_mode = "skip_placeholder"
    else:
        if llm_client is None:
            print(f"Row {split}/{index}: No LLM client, skipping sample.")
            return None

        generation_messages = build_generation_messages(question, answer, raw_images)
        step_generation_mode = "gpt_verified"
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                response_text = llm_client.send_stable_request(generation_messages, temperature=0.0)
                parsed = extract_json(response_text)
                if not parsed or "steps" not in parsed:
                    attempt += 1
                    continue

                steps = parsed["steps"]
                pred_answer = clean_answer_text(parsed.get("final_answer", ""))
                if not isinstance(steps, list) or len(steps) == 0 or not pred_answer:
                    attempt += 1
                    continue

                verifier_messages = build_verifier_messages(question, answer, pred_answer, steps, raw_images)
                review_response = llm_client.send_stable_request(verifier_messages, temperature=0.0)
                review_parsed = extract_json(review_response)

                if (
                    isinstance(review_parsed, dict)
                    and str(review_parsed.get("verdict", "")).upper() == "PASS"
                    and review_parsed.get("visual_fidelity") is True
                    and review_parsed.get("reasoning_faithfulness") is True
                    and review_parsed.get("semantic_answer_correctness") is True
                ):
                    break

                attempt += 1
            except Exception as exc:
                print(f"Row {split}/{index} Error: {exc}")
                time.sleep(1)
                attempt += 1
        else:
            print(f"Row {split}/{index}: Failed GPT verification after retries, skipping sample.")
            return None

    prompt_content = prompt_template.replace("{Question}", question)
    if len(images_field) > 0 and "<image>" not in prompt_content:
        prompt_content = "<image>\n" + prompt_content
    result["prompt"] = [{"role": "user", "content": prompt_content}]
    result["reward_model"] = {
        "ground_truth": json.dumps({"answer": answer, "steps": steps}, ensure_ascii=False),
        "style": "rule",
    }
    extra_info["step_generation_mode"] = step_generation_mode
    result["extra_info"] = extra_info
    result["success"] = True
    return result


def sort_results(results):
    return sorted(results, key=lambda record: record.get("extra_info", {}).get("original_index", -1))


def convert_split(
    dataset_type,
    dataset_name,
    split,
    output_file,
    dataset_config=None,
    source_filter=None,
    max_samples=None,
    limit=None,
    skip_generation=False,
    max_workers=None,
    ignore_cache=False,
):
    max_workers = resolve_max_workers(max_workers=max_workers, skip_generation=skip_generation)
    print(f"Loading {dataset_name} [{split}]...")
    dataset = load_hf_split(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        source_filter=source_filter,
        max_samples=max_samples,
    )

    if limit is not None:
        limit = min(limit, len(dataset))
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(limit))
        else:
            dataset = dataset[:limit]

    print(f"Total samples to process in {split}: {len(dataset)}")

    processed_indices = set()
    results = []
    if os.path.exists(output_file) and not ignore_cache:
        try:
            print(f"Found existing output file {output_file}, loading cache...")
            existing_df = pd.read_parquet(output_file)
            results = existing_df.to_dict("records")
            for existing_item in results:
                original_index = existing_item.get("extra_info", {}).get("original_index")
                if original_index is not None:
                    processed_indices.add(original_index)
            print(f"Loaded {len(results)} processed samples from cache.")
        except Exception as exc:
            print(f"Error loading cache from {output_file}: {exc}")

    rows_to_process = [(idx, dataset[idx]) for idx in range(len(dataset)) if idx not in processed_indices]
    print(f"Remaining samples to process in {split}: {len(rows_to_process)}")

    if not rows_to_process:
        print(f"All samples in {split} have been processed.")
        return

    llm_client = None if skip_generation else init_llm_client()
    save_interval = 20
    processed_in_this_run = 0

    print(f"Starting processing for {split} with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_item,
                item,
                idx,
                split,
                dataset_type,
                dataset_name,
                llm_client,
                skip_generation,
            ): idx
            for idx, item in rows_to_process
        }

        for future in tqdm(as_completed(future_to_idx), total=len(rows_to_process)):
            idx = future_to_idx[future]
            try:
                res = future.result()
                if res and res["success"]:
                    del res["success"]
                    del res["index"]
                    results.append(res)
                    processed_in_this_run += 1

                    if processed_in_this_run % save_interval == 0:
                        print(f"\n--- Periodic Save [{split}] at {len(results)} samples ---")
                        df_temp = pd.DataFrame(sort_results(results))
                        df_temp.to_parquet(output_file, index=False)
                        sample = res
                        print(">> Showcase <<")
                        print(f"Prompt (excerpt): {str(sample['prompt'])[:180]}...")
                        print(f"Images: {len(sample['images'])}")
                        print(f"Reward Model: {sample['reward_model']}")
                        print("----------------------------------------------\n")
            except Exception as exc:
                print(f"Row {split}/{idx} exception: {exc}")

    if results:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df_out = pd.DataFrame(sort_results(results))
        print(f"Saving final {len(df_out)} processed samples for {split} to {output_file}...")
        df_out.to_parquet(output_file, index=False)
    else:
        print(f"No results to save for split {split}.")


def convert_dataset(
    dataset_type,
    dataset_name=None,
    dataset_config=None,
    source_filter=None,
    max_samples=None,
    output_dir=None,
    splits=None,
    limit=None,
    skip_generation=False,
    max_workers=None,
    ignore_cache=False,
):
    dataset_name = resolve_dataset_name(dataset_type, dataset_name)
    source_filter = resolve_source_filter(dataset_type, source_filter)
    max_samples = resolve_max_samples(dataset_type, max_samples)
    output_dir = resolve_output_dir(dataset_type, output_dir)
    max_workers = resolve_max_workers(max_workers=max_workers, skip_generation=skip_generation)
    available_splits = get_available_splits(dataset_name, dataset_config=dataset_config)

    if splits is None:
        requested_splits = DEFAULT_SPLITS[dataset_type]
    else:
        requested_splits = splits

    valid_splits = [split for split in requested_splits if split in available_splits]
    missing_splits = [split for split in requested_splits if split not in available_splits]
    for split in missing_splits:
        print(f"Warning: split '{split}' not found in {dataset_name}, skipping it.")

    if not valid_splits:
        raise ValueError(f"No valid splits to process. Available splits for {dataset_name}: {available_splits}")

    os.makedirs(output_dir, exist_ok=True)
    for split in valid_splits:
        output_file = os.path.join(output_dir, f"{split}.parquet")
        convert_split(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            source_filter=source_filter,
            max_samples=max_samples,
            split=split,
            output_file=output_file,
            limit=limit,
            skip_generation=skip_generation,
            max_workers=max_workers,
            ignore_cache=ignore_cache,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["vision_r1_rl", "dapo"], default="vision_r1_rl")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--source_filter", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split for testing")
    parser.add_argument("--skip_generation", action="store_true", help="Skip LLM generation and use placeholders")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--ignore_cache", action="store_true", help="Ignore existing parquet cache and rebuild")

    args = parser.parse_args()

    convert_dataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        source_filter=args.source_filter,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        splits=args.splits,
        limit=args.limit,
        skip_generation=args.skip_generation,
        max_workers=args.max_workers,
        ignore_cache=args.ignore_cache,
    )
