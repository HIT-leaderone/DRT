import argparse
import csv
import fnmatch
import json
import math
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import pandas as pd
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
valid_models = [
    # "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    # "VisionThink-Efficient",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it",
    # # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Geo-merged-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-Ray-Training-RL-autoeval",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Ray-Training-RL-autoeval",
    # # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Geo-merged-RL-Length-Bonus",
    # # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Geo-merged-RL-Length-Bonus-Uniform_Prompt",
    # # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Geo-merged-RL-Length-Bonus-Multi-Modal",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-Geo-merged-RL-GPTStepBonus",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-RL-NOSTEP",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VisionR1-DAPO-RL-Adaptive",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VisionR1-DAPO-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VisionR1-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VisionR1-RL-NOSTEP",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VisionR1-DAPO-RL-NOSTEP",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-Qwen3-235B-VL-VisionR1-DAPO-RL",
    # "Qwen3-VL-8B-Instruct-VisionR1-SFT-500it-Qwen3-235B-VL-VisionR1-DAPO-RL",
    # "Qwen3-VL-8B-Instruct-Raw-VisionR1-SFT-1600it-Qwen3-235B-VisionR1-DAPO-RL-Answer-Only",
    "Qwen3-VL-8B-Instruct-Raw-VisionR1-SFT-1600it-Qwen3-235B-VisionR1-DAPO-RL-Process-Reward",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-RM-Qwen3-8B",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-RM-Qwen3-14B",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-RM-Qwen3-32B",
    "Qwen3-VL-8B-Instruct-VisionR1-SFT-800it-RM-Qwen3-30B-A3B",
    "Qwen3-VL-8B-Instruct-Raw-VisionR1-SFT-1600it-Qwen3-235B-VisionR1-DAPO-RL-Process-Reward-Retry"
]
VALID_MODEL_SET = set(valid_models)

RAW_FILE_EXTENSIONS = ['tsv', 'xlsx', 'json']
TOKEN_SOURCE_COLUMNS = ['prediction', 'response']
XLSX_CELL_CHAR_LIMIT = 32767
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "benchmark_summary.csv")
ITEM_CACHE_FILE = os.path.join(SCRIPT_DIR, "benchmark_summary_item_cache.json")
EXTERNAL_RESULTS_FILE = os.path.join(SCRIPT_DIR, "benchmark_summary_external_results.csv")
ITEM_CACHE_VERSION = 2
FORCE_REFRESH = os.getenv("SHOW_BENCH_FORCE_REFRESH", "0") == "1"
PLOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_summary_plots")
SCORE_AXIS_LIMITS = (55.0, 70.0)
BASELINE_MODEL_NAME = "Qwen3-VL-8B-Instruct"
BASELINE_SCORE_OVERRIDES = {
    "MathVista": 77.2,
    "MathVerse": 62.1,
    "LogicVista": 55.3,
}

PROMPT_DISPLAY_NAMES = {
    "Qwen3-VL": "Qwen-Standard",
    "Qwen3-VL-Thinking-vLLM0191-TF4571": "Qwen-Thinking",
    "Directly-Answer": "Qwen-DA",
    "Short-COT-Image": "Qwen-DRT",
    "CoD": "CoD",
    "vlmeval_qwen3_vl_8b_cod": "CoD",
    "thinkless_formal_full": "Thinkless",
    "visionthink": "VisionThink",
}

RESULT_ROOT = os.getenv("SHOW_BENCH_RESULT_ROOT", os.path.join(SCRIPT_DIR, "results"))
QWEN3_VL_ROOT = os.getenv("SHOW_BENCH_QWEN3_VL_ROOT", os.path.join(RESULT_ROOT, "Qwen3-VL"))
SHORT_COT_IMAGE_ROOT = os.getenv("SHOW_BENCH_SHORT_COT_IMAGE_ROOT", os.path.join(RESULT_ROOT, "Short-COT-Image"))
DIRECTLY_ANSWER_ROOT = os.getenv("SHOW_BENCH_DIRECTLY_ANSWER_ROOT", os.path.join(RESULT_ROOT, "Directly-Answer"))
COD_ROOT = os.getenv("SHOW_BENCH_COD_ROOT", os.path.join(RESULT_ROOT, "CoD"))
THINKLESS_ROOT = os.getenv("SHOW_BENCH_THINKLESS_ROOT", os.path.join(RESULT_ROOT, "thinkless_formal_full"))
VISIONTHINK_ROOT = os.getenv("SHOW_BENCH_VISIONTHINK_ROOT", os.path.join(RESULT_ROOT, "visionthink"))
QWEN3_VL_THINKING_VLLM0191_ROOT = os.getenv(
    "SHOW_BENCH_QWEN3_VL_THINKING_ROOT",
    os.path.join(RESULT_ROOT, "Qwen3-VL-Thinking-vLLM0191-TF4571"),
)
TARGET_DIRS = [
    QWEN3_VL_ROOT,
    SHORT_COT_IMAGE_ROOT,
    DIRECTLY_ANSWER_ROOT,
    COD_ROOT,
    THINKLESS_ROOT,
    VISIONTHINK_ROOT,
    QWEN3_VL_THINKING_VLLM0191_ROOT,
]

MODEL_PATH = os.getenv("MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MAX_WORKERS = max(1, int(os.getenv("SHOW_BENCH_MAX_WORKERS", min(8, os.cpu_count() or 1))))


def load_tokenizer(model_path):
    print(f"Loading tokenizer from: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def _score_to_display(value):
    value = float(value)
    if -1.0 <= value <= 1.0:
        value *= 100.0
    return str(round(value, 2))


def _find_first_match(folder, filenames, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    def _iter_task_dirs():
        try:
            task_dirs = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.startswith('T2026') and os.path.isdir(os.path.join(folder, name))
            ]
        except Exception:
            return []
        return sorted(task_dirs)

    for pattern in patterns:
        matches = sorted(
            f for f in filenames
            if not f.startswith('~$') and fnmatch.fnmatch(f, pattern)
        )
        for match in matches:
            candidate = os.path.join(folder, match)
            if os.path.exists(candidate):
                return candidate

        for task_dir in _iter_task_dirs():
            try:
                task_files = sorted(
                    f for f in os.listdir(task_dir)
                    if not f.startswith('~$') and fnmatch.fnmatch(f, pattern)
                )
            except Exception:
                continue
            for match in task_files:
                candidate = os.path.join(task_dir, match)
                if os.path.exists(candidate):
                    return candidate
    return None


def _load_table(filepath):
    suffix = os.path.splitext(filepath)[1].lower()
    if suffix == '.xlsx':
        return pd.read_excel(filepath)
    if suffix == '.tsv':
        return pd.read_csv(filepath, sep='\t')
    if suffix == '.csv':
        return pd.read_csv(filepath)
    if suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame([data])
        return pd.DataFrame()
    raise ValueError(f'Unsupported file type: {filepath}')


def _normalize_text_series(series):
    return series.apply(lambda x: '' if pd.isna(x) else str(x))


def _reconstruct_raw_outputs(df):
    base_col = next((col for col in TOKEN_SOURCE_COLUMNS if col in df.columns), None)
    if base_col is None:
        return None, []

    outputs = _normalize_text_series(df[base_col])
    source_columns = [base_col]

    if 'thinking' in df.columns:
        thinking = _normalize_text_series(df['thinking'])
        source_columns.append('thinking')
        merged = []
        for think_text, pred_text in zip(thinking, outputs):
            if think_text.strip():
                if '<think>' in think_text or '</think>' in think_text:
                    merged.append(f'{think_text}\n\n{pred_text}'.strip())
                else:
                    merged.append(f'<think>\n{think_text}\n</think>\n\n{pred_text}'.strip())
            else:
                merged.append(pred_text)
        outputs = pd.Series(merged)

    fail_prefixes = (
        'Failed to obtain answer',
        'Failed to obtain answer via API.',
    )
    outputs = outputs.apply(
        lambda x: '' if any(str(x).startswith(prefix) for prefix in fail_prefixes) else str(x)
    )
    return outputs.tolist(), source_columns


def _count_xlsx_truncation_rows(filepath, df, source_columns):
    if not filepath.lower().endswith('.xlsx') or not source_columns:
        return 0

    row_mask = pd.Series(False, index=df.index)
    for col in source_columns:
        if col in df.columns:
            lengths = _normalize_text_series(df[col]).str.len()
            row_mask |= (lengths == XLSX_CELL_CHAR_LIMIT)
    return int(row_mask.sum())


def _find_raw_prediction_file(folder, filenames, model_name, dataset_names):
    file_set = set(filenames)
    for dataset_name in dataset_names:
        for ext in RAW_FILE_EXTENSIONS:
            candidate = f'{model_name}_{dataset_name}.{ext}'
            if candidate in file_set:
                candidate_path = os.path.join(folder, candidate)
                if os.path.exists(candidate_path):
                    return candidate_path
    for dataset_name in dataset_names:
        for ext in RAW_FILE_EXTENSIONS:
            matched = _find_first_match(folder, filenames, f'{model_name}_{dataset_name}.{ext}')
            if matched is not None:
                return matched
    return None


def _find_score_file(folder, filenames, model_name, score_templates, dataset_names):
    patterns = []
    for dataset_name in dataset_names:
        for template in score_templates:
            patterns.append(template.format(model=model_name, dataset=dataset_name))
    return _find_first_match(folder, filenames, patterns)


def _load_item_cache(cache_file):
    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Failed to load item cache from {cache_file}: {e}")
        return {}


def _save_item_cache(cache_file, item_cache):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(item_cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def _get_file_signature(filepath):
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _get_cached_token_stats(filepath, item_cache):
    entry = item_cache.get(filepath)
    if entry is None:
        return None

    if entry.get('cache_version') != ITEM_CACHE_VERSION:
        return None

    if entry.get('model_path') != MODEL_PATH:
        return None

    try:
        signature = _get_file_signature(filepath)
    except OSError:
        return None

    if entry.get('size') != signature['size'] or entry.get('mtime_ns') != signature['mtime_ns']:
        return None

    return entry


def _load_prev_predictions(filepath):
    import pickle

    prev_path = os.path.splitext(filepath)[0] + '_PREV.pkl'
    if not os.path.exists(prev_path):
        return None

    try:
        with open(prev_path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    return [str(v) for _, v in sorted(data.items())]


def _is_overwritten_text_math_file(df, benchmark_name):
    if benchmark_name != 'MATH500':
        return False
    return (
        'prediction' in df.columns and
        'hit' in df.columns and
        'solution' in df.columns and
        'thinking' not in df.columns
    )


def _print_summary(df_res, warnings_data, show_warning_log=False):
    print("\n" + "=" * 100)
    print("Benchmark Summary (Acc/Token)")
    print("=" * 100)
    print(df_res.to_string(index=False, col_space=10, justify='center'))
    print("\n" + "=" * 100)

    if not show_warning_log:
        return

    truncation_warnings = [w for w in warnings_data if w.get('type') == 'xlsx_truncation']
    overwrite_warnings = [w for w in warnings_data if w.get('type') == 'overwritten_prediction']
    fallback_warnings = [w for w in warnings_data if w.get('type') == 'used_prev_pkl']

    if truncation_warnings:
        print("Potential XLSX truncation detected in raw prediction files:")
        for warning in truncation_warnings:
            print(
                f"- {warning['benchmark']}: {warning['truncated_rows']} rows hit "
                f"{XLSX_CELL_CHAR_LIMIT} chars in {warning['file']}"
            )
        print("=" * 100)

    if overwrite_warnings:
        print("Raw output length unavailable for files overwritten by evaluation:")
        for warning in overwrite_warnings:
            print(f"- {warning['benchmark']}: {warning['file']}")
        print("=" * 100)

    if fallback_warnings:
        print("Raw output length recovered from *_PREV.pkl fallback:")
        for warning in fallback_warnings:
            print(f"- {warning['benchmark']}: {warning['file']}")
        print("=" * 100)


def _extract_score_from_cell(cell):
    text = str(cell).strip()
    if not text:
        return float('-inf')

    score_part = text.split('/', 1)[0].strip()
    if score_part in {'', '-'}:
        return float('-inf')

    try:
        return float(score_part)
    except Exception:
        return float('-inf')


def _extract_optional_score_from_cell(cell):
    text = str(cell).strip()
    if not text:
        return None

    score_part = text.split('/', 1)[0].strip()
    if score_part in {'', '-'}:
        return None

    try:
        return float(score_part)
    except Exception:
        return None


def _format_avg_score(value):
    return f"{value:.2f}".rstrip('0').rstrip('.')


def _extract_length_from_cell(cell):
    text = str(cell).strip()
    if not text or text.lower() == 'nan':
        return "-"
    if '/' not in text:
        return "-"

    length_part = text.split('/', 1)[1].strip()
    return length_part or "-"


def _extract_optional_length_from_cell(cell):
    length_part = _extract_length_from_cell(cell)
    if length_part in {'', '-'}:
        return None

    try:
        return float(length_part)
    except Exception:
        return None


def _override_score_in_cell(cell, score):
    return format_cell(_format_avg_score(score), _extract_length_from_cell(cell))


def _map_dataframe(df, func):
    if hasattr(df, 'map'):
        return df.map(func)
    return df.applymap(func)


def _build_avg_cells(display_df, benchmark_names):
    score_matrix = _map_dataframe(display_df[benchmark_names], _extract_optional_score_from_cell)
    length_matrix = _map_dataframe(display_df[benchmark_names], _extract_optional_length_from_cell)

    score_complete_rows = score_matrix.notna().all(axis=1)
    length_complete_rows = length_matrix.notna().all(axis=1)
    avg_scores = score_matrix.mean(axis=1)
    avg_lengths = length_matrix.mean(axis=1)

    avg_cells = [
        format_cell(
            _format_avg_score(avg_score) if score_complete else '-',
            f'{avg_length:.1f}' if length_complete else '-',
        )
        for avg_score, avg_length, score_complete, length_complete in zip(
            avg_scores, avg_lengths, score_complete_rows, length_complete_rows
        )
    ]

    return avg_cells, avg_scores, score_complete_rows


def _load_external_benchmark_cell_overrides(filepath=EXTERNAL_RESULTS_FILE):
    path = filepath
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        return {}

    overrides = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = str(row.get('model', '')).strip()
            prompt_name = str(row.get('prompt', '')).strip()
            benchmark_name = str(row.get('benchmark', '')).strip()
            score = str(row.get('score', '')).strip()
            tokens = str(row.get('tokens', '')).strip()
            if not (model_name and prompt_name and benchmark_name and score and tokens):
                continue
            overrides.setdefault((model_name, prompt_name), {})[benchmark_name] = {
                'score': score,
                'tokens': tokens,
            }
    return overrides


EXTERNAL_BENCHMARK_CELL_OVERRIDES = None


def _get_benchmark_cell_override(model_name, prompt_name, benchmark_name):
    global EXTERNAL_BENCHMARK_CELL_OVERRIDES
    if EXTERNAL_BENCHMARK_CELL_OVERRIDES is None:
        EXTERNAL_BENCHMARK_CELL_OVERRIDES = _load_external_benchmark_cell_overrides()

    override = EXTERNAL_BENCHMARK_CELL_OVERRIDES.get((str(model_name), str(prompt_name)))
    if not override:
        return None

    item = override.get(benchmark_name)
    if item is None:
        return None

    return (
        _format_avg_score(float(item["score"])),
        f'{float(item["tokens"]):.1f}',
    )


def _refresh_avg_column(display_df):
    if display_df.empty or 'AVG' not in display_df.columns:
        return display_df

    benchmark_names = [benchmark['name'] for benchmark in BENCHMARKS if benchmark['name'] in display_df.columns]
    if not benchmark_names:
        return display_df

    display_df = display_df.copy()
    avg_cells, _, _ = _build_avg_cells(display_df, benchmark_names)
    display_df['AVG'] = avg_cells
    return display_df


def _apply_baseline_score_overrides(df_res):
    if df_res.empty:
        return df_res

    benchmark_names = [name for name in BASELINE_SCORE_OVERRIDES if name in df_res.columns]
    if not benchmark_names:
        return df_res

    if 'Model' in df_res.columns:
        if 'Prompt' in df_res.columns:
            normalized_prompt = df_res['Prompt'].apply(lambda x: _normalize_prompt_label(x) or str(x))
            baseline_mask = (
                (df_res['Model'].astype(str) == BASELINE_MODEL_NAME) &
                (normalized_prompt == 'Qwen-Standard')
            )
        else:
            baseline_mask = df_res['Model'].astype(str) == BASELINE_MODEL_NAME
    elif 'SFT Iteration' in df_res.columns:
        prompt_col = df_res['Prompt'].astype(str) if 'Prompt' in df_res.columns else None
        if prompt_col is not None:
            baseline_mask = (
                (df_res['SFT Iteration'].astype(str) == 'base') &
                (prompt_col == 'Qwen-Standard')
            )
        else:
            baseline_mask = df_res['SFT Iteration'].astype(str) == 'base'
    else:
        return df_res

    if not baseline_mask.any():
        return df_res

    df_res = df_res.copy()
    for benchmark_name in benchmark_names:
        score = BASELINE_SCORE_OVERRIDES[benchmark_name]
        df_res.loc[baseline_mask, benchmark_name] = df_res.loc[baseline_mask, benchmark_name].apply(
            lambda cell: _override_score_in_cell(cell, score)
        )

    if 'AVG' in df_res.columns:
        df_res = _refresh_avg_column(df_res)

    return df_res


def _normalize_prompt_label(prompt):
    text = str(prompt).strip()
    if not text or text.lower() == 'nan':
        return None
    return PROMPT_DISPLAY_NAMES.get(text, text)


def _extract_sft_iteration(model_name):
    if model_name == "Qwen3-VL-8B-Instruct":
        return "base"
    if model_name == "Qwen3-VL-8B-Thinking":
        return "base"

    match = re.search(r'(?:Raw-)?VisionR1-SFT-(\d+)it', str(model_name))
    if match:
        return f"{match.group(1)}it"

    return "others"


def _sft_iteration_sort_key(value):
    text = str(value)
    if text == "base":
        return (0, 0)

    match = re.match(r'^(\d+)it$', text)
    if match:
        return (1, int(match.group(1)))

    return (2, text)


def _split_model_metadata(model_name):
    model_name = str(model_name)

    if model_name == "Qwen3-VL-8B-Instruct":
        return {
            'SFT Iteration': 'base',
            'Dataset': 'Base',
            'Step': '-',
            'Others': 'Base',
        }
    if model_name == "Qwen3-VL-8B-Thinking":
        return {
            'SFT Iteration': 'base',
            'Dataset': 'Base',
            'Step': '-',
            'Others': 'Thinking',
        }
    if model_name == "VisionThink-Efficient":
        return {
            'SFT Iteration': 'base',
            'Dataset': 'Base',
            'Step': '-',
            'Others': 'Base',
        }

    pure_sft_match = re.fullmatch(
        r'Qwen3-VL-8B-Instruct-(?:Raw-)?VisionR1-SFT-(\d+)it',
        model_name,
    )
    if pure_sft_match:
        return {
            'SFT Iteration': f"{pure_sft_match.group(1)}it",
            'Dataset': 'Base',
            'Step': '-',
            'Others': 'SFT',
        }

    sft_iteration = _extract_sft_iteration(model_name)

    match = re.search(r'(?:Raw-)?VisionR1-SFT-\d+it-Qwen3-235B-(.+)$', model_name)
    remainder = match.group(1) if match else model_name

    has_nostep = 'NOSTEP' in remainder
    clean_remainder = remainder.replace('-NOSTEP', '').strip('-')

    # Dataset 解析规则：
    # 1) Geo-merged-*   -> Dataset = Geo-merged
    # 2) VisionR1-DAPO-* -> Dataset = VisionR1-DAPO
    # 3) VisionR1-*     -> Dataset = VisionR1
    # 4) 其他全部默认   -> Dataset = Geo3k
    if clean_remainder.startswith('Geo-merged-'):
        dataset = 'Geo-merged'
        others = clean_remainder[len('Geo-merged-'):].strip('-') or '-'
    elif clean_remainder.startswith('VL-VisionR1-DAPO-RL'):
        dataset = 'VisionR1-DAPO'
        others = 'VL judge'
    elif clean_remainder.startswith('VisionR1-DAPO-RL-'):
        dataset = 'VisionR1-DAPO'
        others = clean_remainder[len('VisionR1-DAPO-RL-'):].strip('-') or '-'
    elif clean_remainder == 'VisionR1-DAPO-RL':
        dataset = 'VisionR1-DAPO'
        others = 'RL'
    elif clean_remainder.startswith('VisionR1-DAPO-'):
        dataset = 'VisionR1-DAPO'
        others = clean_remainder[len('VisionR1-DAPO-'):].strip('-') or '-'
    elif clean_remainder.startswith('VisionR1-'):
        dataset = 'VisionR1'
        others = clean_remainder[len('VisionR1-'):].strip('-') or '-'
    else:
        dataset = 'Geo3k'
        others = clean_remainder or '-'

    # Step 规则：
    # 只有带 VisionR1 的才区分 step；其他一律 '-'
    if dataset in {'VisionR1', 'VisionR1-DAPO'}:
        step = 'Wo Step' if has_nostep else 'With Step'
    else:
        step = '-'

    return {
        'SFT Iteration': sft_iteration,
        'Dataset': dataset,
        'Step': step,
        'Others': others,
    }



def _prepare_display_table(df_res):
    if df_res.empty:
        return df_res

    benchmark_names = [benchmark['name'] for benchmark in BENCHMARKS]

    metadata_df = pd.DataFrame([_split_model_metadata(model) for model in df_res['Model']])

    display_df = pd.concat(
        [
            df_res[['Prompt']].reset_index(drop=True),
            metadata_df,
            df_res.drop(columns=['Prompt', 'Model']).reset_index(drop=True),
        ],
        axis=1
    )
    display_df['Prompt'] = display_df['Prompt'].apply(lambda x: _normalize_prompt_label(x) or x)

    avg_cells, avg_scores, complete_rows = _build_avg_cells(display_df, benchmark_names)
    display_df['AVG'] = avg_cells

    display_df['_prompt_sort'] = display_df['Prompt'].apply(_prompt_sort_key)
    display_df['_sft_sort'] = display_df['SFT Iteration'].apply(_sft_iteration_sort_key)

    display_df['_avg_sort'] = [
        avg if is_complete and pd.notna(avg) else float('-inf')
        for avg, is_complete in zip(avg_scores, complete_rows)
    ]

    display_df = display_df.sort_values(
        by=[
            '_prompt_sort',
            '_avg_sort',
            '_sft_sort',
            'Dataset',
            'Step',
            'Others',
        ],
        ascending=[True, False, True, True, True, True],
    ).drop(columns=['_prompt_sort', '_sft_sort', '_avg_sort'])

    ordered_cols = [
        'Prompt', 'SFT Iteration', 'Dataset', 'Step', 'Others', 'AVG'
    ] + benchmark_names

    return display_df[ordered_cols]

def _insert_prompt_separators(df_res):
    if df_res.empty:
        return df_res

    rows = []
    prev_prompt = None
    columns = list(df_res.columns)

    for _, row in df_res.iterrows():
        prompt = row['Prompt']
        if prev_prompt is not None and prompt != prev_prompt:
            rows.append({col: '' for col in columns})
        rows.append(row.to_dict())
        prev_prompt = prompt

    return pd.DataFrame(rows, columns=columns)


def _plot_prompt_sort_key(prompt):
    text = _normalize_prompt_label(prompt)
    if text is None:
        return (4, '')
    if text == 'Qwen-DA':
        return (2, 0)
    if text == 'Qwen-DRT':
        return (2, 1)
    if text == 'Qwen-Standard':
        return (2, 2)
    if text == 'Qwen-Thinking':
        return (2, 3)
    if text == 'CoD':
        return (2, 4)
    if text == 'Thinkless':
        return (2, 5)
    if text == 'VisionThink':
        return (2, 6)

    match = re.match(r'^(\d+)it$', text)
    if match:
        return (1, int(match.group(1)))

    return (4, text)


def _build_plot_series_label(row):
    if str(row['SFT Iteration']) == 'base':
        prompt = str(row['Prompt'])
        if prompt not in {'', 'nan'}:
            return prompt
        return 'base'

    parts = [str(row['SFT Iteration']), str(row['Dataset'])]
    step = str(row['Step'])
    others = str(row['Others'])

    if step not in {'', '-', 'nan'}:
        parts.append(step)
    if others not in {'', '-', 'Base', 'nan'}:
        parts.append(others)

    return ' | '.join(parts)


def _extract_plot_metric_value(metric_name, cell):
    if metric_name == 'AVG':
        try:
            return float(cell)
        except Exception:
            return None
    return _extract_optional_score_from_cell(cell)


def _get_score_axis_limits(_values):
    return SCORE_AXIS_LIMITS


def generate_benchmark_line_plots(df_res, output_dir=PLOT_OUTPUT_DIR):
    if df_res.empty:
        return []

    plot_df = df_res.copy()
    if 'SFT Iteration' not in plot_df.columns and {'Model', 'Prompt'}.issubset(plot_df.columns):
        plot_df = _prepare_display_table(_apply_baseline_score_overrides(plot_df))
    plot_df = _apply_baseline_score_overrides(plot_df)
    plot_df = plot_df.dropna(subset=['Prompt'])
    plot_df['PromptLabel'] = plot_df['Prompt'].apply(_normalize_prompt_label)
    plot_df = plot_df.dropna(subset=['PromptLabel'])
    plot_df['SeriesLabel'] = plot_df.apply(_build_plot_series_label, axis=1)

    prompt_order = [
        prompt for prompt in sorted(plot_df['PromptLabel'].unique(), key=_plot_prompt_sort_key)
        if prompt != 'base'
    ]
    if not prompt_order:
        prompt_order = sorted(plot_df['PromptLabel'].unique(), key=_plot_prompt_sort_key)
    metrics = ['AVG'] + [benchmark['name'] for benchmark in BENCHMARKS]
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for metric_name in metrics:
        metric_df = plot_df[['PromptLabel', 'SeriesLabel', metric_name]].copy()
        metric_df['Score'] = metric_df[metric_name].apply(
            lambda value: _extract_plot_metric_value(metric_name, value)
        )
        metric_df = metric_df.dropna(subset=['Score'])
        baseline_scores = metric_df.loc[metric_df['SeriesLabel'] == 'base', 'Score'].tolist()
        non_baseline_df = metric_df[metric_df['SeriesLabel'] != 'base']

        if non_baseline_df.empty:
            pivot_df = pd.DataFrame(index=prompt_order)
        else:
            pivot_df = non_baseline_df.pivot_table(
                index='PromptLabel',
                columns='SeriesLabel',
                values='Score',
                aggfunc='mean',
            ).reindex(prompt_order)

        fig, ax = plt.subplots(figsize=(14, 8))
        x_positions = list(range(len(prompt_order)))
        plotted_labels = []
        plotted_scores = []

        for series_label in pivot_df.columns:
            series = pivot_df[series_label]
            if not series.notna().any():
                continue
            ax.plot(
                x_positions,
                series.to_numpy(),
                marker='o',
                linewidth=2,
                markersize=5,
                label=series_label,
            )
            plotted_labels.append(series_label)
            plotted_scores.extend(series.dropna().tolist())

        if baseline_scores:
            baseline_score = baseline_scores[0]
            ax.axhline(
                baseline_score,
                linestyle='--',
                linewidth=2,
                color='tab:green',
                alpha=0.9,
                label='base',
            )
            plotted_labels.append('base')
            plotted_scores.append(baseline_score)

        if not plotted_labels:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        else:
            y_limits = _get_score_axis_limits(plotted_scores)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

        ax.set_title(f'{metric_name} Line Chart')
        ax.set_xlabel('Prompt')
        ax.set_ylabel('Score')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(prompt_order, rotation=45, ha='right')
        ax.margins(x=0.02)
        ax.grid(True, linestyle='--', alpha=0.3)

        if plotted_labels:
            legend_cols = min(4, max(1, math.ceil(len(plotted_labels) / 2)))
            legend_rows = math.ceil(len(plotted_labels) / legend_cols)
            legend_bottom = min(0.34, 0.14 + 0.05 * legend_rows)
            fig.tight_layout(rect=(0, legend_bottom, 1, 1))
            fig.legend(
                loc='lower center',
                bbox_to_anchor=(0.5, 0.015),
                ncol=legend_cols,
                frameon=False,
                fontsize=9,
                columnspacing=1.4,
                handlelength=2.2,
            )
        else:
            fig.tight_layout()
        safe_name = re.sub(r'[^A-Za-z0-9]+', '_', metric_name).strip('_').lower()
        output_path = os.path.join(output_dir, f'{safe_name}_line_chart.png')
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def calculate_avg_tokens(filepath, tokenizer, item_cache, benchmark_name):
    if not FORCE_REFRESH:
        cached_entry = _get_cached_token_stats(filepath, item_cache)
        if cached_entry is not None:
            return (
                cached_entry['avg_len'],
                cached_entry.get('truncated_rows', 0),
                None,
                cached_entry.get('warning'),
            )

    try:
        df = _load_table(filepath)
        warning = None

        if _is_overwritten_text_math_file(df, benchmark_name):
            prev_predictions = _load_prev_predictions(filepath)
            current_predictions, _ = _reconstruct_raw_outputs(df)
            current_avg_char = (
                sum(len(x) for x in current_predictions) / len(current_predictions)
                if current_predictions else 0.0
            )
            if prev_predictions:
                prev_avg_char = sum(len(x) for x in prev_predictions) / len(prev_predictions)
                if prev_avg_char > max(current_avg_char * 2.0, current_avg_char + 32.0):
                    predictions = prev_predictions
                    source_columns = ['prediction']
                    warning = {
                        'benchmark': benchmark_name,
                        'file': filepath,
                        'type': 'used_prev_pkl',
                    }
                else:
                    signature = _get_file_signature(filepath)
                    cache_entry = {
                        'avg_len': '-',
                        'truncated_rows': 0,
                        'warning': {
                            'benchmark': benchmark_name,
                            'file': filepath,
                            'type': 'overwritten_prediction',
                        },
                        'model_path': MODEL_PATH,
                        'cache_version': ITEM_CACHE_VERSION,
                        **signature,
                    }
                    return '-', 0, cache_entry, cache_entry['warning']
            else:
                signature = _get_file_signature(filepath)
                cache_entry = {
                    'avg_len': '-',
                    'truncated_rows': 0,
                    'warning': {
                        'benchmark': benchmark_name,
                        'file': filepath,
                        'type': 'overwritten_prediction',
                    },
                    'model_path': MODEL_PATH,
                    'cache_version': ITEM_CACHE_VERSION,
                    **signature,
                }
                return '-', 0, cache_entry, cache_entry['warning']
        else:
            predictions, source_columns = _reconstruct_raw_outputs(df)

        if predictions is None:
            avg_len = "-"
            truncated_rows = 0
            signature = _get_file_signature(filepath)
            cache_entry = {
                'avg_len': avg_len,
                'truncated_rows': truncated_rows,
                'warning': warning,
                'model_path': MODEL_PATH,
                'cache_version': ITEM_CACHE_VERSION,
                **signature,
            }
            return avg_len, truncated_rows, cache_entry, warning
        if not predictions:
            avg_len = "0.0"
            truncated_rows = 0
            signature = _get_file_signature(filepath)
            cache_entry = {
                'avg_len': avg_len,
                'truncated_rows': truncated_rows,
                'warning': warning,
                'model_path': MODEL_PATH,
                'cache_version': ITEM_CACHE_VERSION,
                **signature,
            }
            return avg_len, truncated_rows, cache_entry, warning

        if tokenizer:
            try:
                encoded = tokenizer(
                    predictions,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )
                token_counts = [len(ids) for ids in encoded['input_ids']]
            except Exception:
                token_counts = [len(tokenizer.encode(p, add_special_tokens=False)) for p in predictions]
            avg_len = sum(token_counts) / len(token_counts)
        else:
            char_counts = [len(p) for p in predictions]
            avg_len = (sum(char_counts) / len(char_counts)) / 4.0

        truncated_rows = _count_xlsx_truncation_rows(filepath, df, source_columns)
        avg_len = str(round(avg_len, 1))
        signature = _get_file_signature(filepath)
        cache_entry = {
            'avg_len': avg_len,
            'truncated_rows': truncated_rows,
            'warning': warning,
            'model_path': MODEL_PATH,
            'cache_version': ITEM_CACHE_VERSION,
            **signature,
        }
        return avg_len, truncated_rows, cache_entry, warning
    except Exception as e:
        print(f"Error reading token stats from {filepath}: {e}")
        return "Err", 0, None, None


def parse_mathverse(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        if 'Overall' not in df.columns:
            return "-"

        target_splits = ["Text Dominant", "Vision Intensive", "Vision Only", "Text Lite", "Vision Dominant"]
        first_col = df.columns[0]
        filtered = df[df[first_col].isin(target_splits)]
        score = filtered['Overall'].mean() if not filtered.empty else df['Overall'].mean()
        return _score_to_display(score)
    except Exception:
        return "-"


def parse_common_benchmark(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]

        acc_col = next((c for c in df.columns if c.lower() == 'acc'), None)
        if acc_col:
            first_col = df.columns[0]
            row = df[df[first_col].astype(str).str.contains('Overall', case=False, na=False)]
            if not row.empty:
                return _score_to_display(row.iloc[0][acc_col])

        overall_col = next((c for c in df.columns if c.lower() == 'overall'), None)
        if overall_col and not df.empty:
            return _score_to_display(df.iloc[0][overall_col])

        return "-"
    except Exception:
        return "-"


def parse_overall_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'overall' not in data:
            return "-"
        return _score_to_display(data['overall'])
    except Exception:
        return "-"


def parse_video_holmes_score(filepath):
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return _score_to_display(data['total']['acc'])

        df = pd.read_excel(filepath)
        if 'score' not in df.columns or df.empty:
            return "-"
        return _score_to_display(df['score'].mean())
    except Exception:
        return "-"


BENCHMARKS = [
    {
        'name': 'MathVista',
        'dataset_names': ['MathVista_MINI', 'MathVista'],
        'score_templates': [
            '{model}_{dataset}_score.csv',
            '{model}_{dataset}_*_score.csv',
        ],
        'score_parser': parse_common_benchmark
    },
    {
        'name': 'MathVerse',
        'dataset_names': ['MathVerse_MINI'],
        'score_templates': [
            '{model}_{dataset}_score.csv',
            '{model}_{dataset}_*_score.csv',
        ],
        'score_parser': parse_mathverse
    },
    # {
    #     'name': 'MathVision',
    #     'dataset_names': ['MathVision', 'MathVision_MINI'],
    #     'score_templates': [
    #         '{model}_{dataset}_score.csv',
    #         '{model}_{dataset}_*_score.csv',
    #     ],
    #     'score_parser': parse_common_benchmark
    # },
    {
        'name': 'LogicVista',
        'dataset_names': ['LogicVista'],
        'score_templates': [
            '{model}_{dataset}_score.csv',
            '{model}_{dataset}_*_score.csv',
        ],
        'score_parser': parse_common_benchmark
    },
    {
        'name': 'GSM8K',
        'dataset_names': ['GSM8K_0shot', 'GSM8K', 'GSM8K_4shot'],
        'score_templates': [
            '{model}_{dataset}_score.json',
            '{model}_{dataset}_*_score.json',
        ],
        'score_parser': parse_overall_json
    },
    # {
    #     'name': 'MATH500',
    #     'dataset_names': ['MATH-500', 'MATH-500_0shot', 'MATH-500_4shot'],
    #     'score_templates': [
    #         '{model}_{dataset}_score.json',
    #         '{model}_{dataset}_*_score.json',
    #     ],
    #     'score_parser': parse_overall_json
    # },
    {
        'name': 'VideoHolmes',
        'dataset_names': ['Video_Holmes', 'Video_Holmes_32frame', 'Video_Holmes_64frame'],
        'score_templates': [
            '{model}_{dataset}_rating.json',
            '{model}_{dataset}_*_rating.json',
            '{model}_{dataset}_score.xlsx',
            '{model}_{dataset}_*_score.xlsx',
        ],
        'score_parser': parse_video_holmes_score
    }
]

def format_cell(score, length):
    """
    格式化单元格，确保对齐。
    格式：[分数(右对齐)] / [长度(左对齐)]
    """
    s_str = str(score) if score != "-" else "-"
    l_str = str(length) if length != "-" else "-"
    return f"{s_str:>6}/{l_str:<6}"


def _is_valid_folder(folder_path, files):
    base_name = os.path.basename(folder_path)
    if base_name.startswith('T2026') or base_name.startswith('bak'):
        return False

    if base_name not in VALID_MODEL_SET:
        return False

    prompt_name = os.path.basename(os.path.dirname(folder_path))
    if not _is_supported_prompt_folder(prompt_name):
        return False

    folder_real = os.path.realpath(folder_path)
    qwen3_root_real = os.path.realpath(QWEN3_VL_ROOT)
    short_cot_root_real = os.path.realpath(SHORT_COT_IMAGE_ROOT)
    directly_answer_root_real = os.path.realpath(DIRECTLY_ANSWER_ROOT)
    cod_root_real = os.path.realpath(COD_ROOT)
    thinkless_root_real = os.path.realpath(THINKLESS_ROOT)
    visionthink_root_real = os.path.realpath(VISIONTHINK_ROOT)
    qwen3_vl_thinking_vllm0191_root_real = os.path.realpath(QWEN3_VL_THINKING_VLLM0191_ROOT)

    if 'Raw-VisionR1-SFT' in base_name:
        if os.path.commonpath([folder_real, qwen3_root_real]) != qwen3_root_real:
            return False
    elif 'VisionR1-SFT' in base_name:
        if os.path.commonpath([folder_real, short_cot_root_real]) != short_cot_root_real:
            return False
    elif base_name == BASELINE_MODEL_NAME:
        allowed_roots = [qwen3_root_real, short_cot_root_real, directly_answer_root_real, cod_root_real, thinkless_root_real]
        if not any(os.path.commonpath([folder_real, root]) == root for root in allowed_roots):
            return False
    elif base_name == "Qwen3-VL-8B-Thinking":
        if prompt_name != 'Qwen3-VL-Thinking-vLLM0191-TF4571':
            return False
        if os.path.commonpath([folder_real, qwen3_vl_thinking_vllm0191_root_real]) != qwen3_vl_thinking_vllm0191_root_real:
            return False
    elif base_name == "VisionThink-Efficient":
        if os.path.commonpath([folder_real, visionthink_root_real]) != visionthink_root_real:
            return False

    return any(f.endswith(('.csv', '.xlsx', '.json', '.tsv')) for f in files)


def _prompt_sort_key(prompt):
    text = str(prompt)
    normalized = _normalize_prompt_label(text) or text
    if normalized == 'Qwen-DA':
        return (2, 0)
    if normalized == 'Qwen-DRT':
        return (2, 1)
    if normalized == 'Qwen-Standard':
        return (2, 2)
    if normalized == 'Qwen-Thinking':
        return (2, 3)
    if normalized == 'CoD':
        return (2, 4)
    if normalized == 'Thinkless':
        return (2, 5)
    if normalized == 'VisionThink':
        return (2, 6)
    match = re.match(r'^(\d+)it$', text)
    if match:
        return (1, int(match.group(1)))
    return (4, normalized)


def _is_supported_prompt_folder(prompt):
    text = str(prompt)
    if text == 'Qwen3-VL':
        return True
    if text == 'Qwen3-VL-Thinking-vLLM0191-TF4571':
        return True
    if text == 'Directly-Answer':
        return True
    if text == 'Short-COT-Image':
        return True
    if text == 'CoD':
        return True
    if text == 'vlmeval_qwen3_vl_8b_cod':
        return True
    if text == 'thinkless_formal_full':
        return True
    if text == 'visionthink':
        return True

    match = re.match(r'^(\d+)it$', text)
    if not match:
        return False

    return int(match.group(1)) % 10 == 0


def _folder_sort_key(item):
    folder_path, _ = item
    parts = folder_path.rstrip('/').split('/')
    model = parts[-1]
    prompt = parts[-2] if len(parts) >= 2 else ''
    return (_prompt_sort_key(prompt), model, folder_path)


def collect_candidate_folders():
    candidates = {}
    for root_dir in TARGET_DIRS:
        if not os.path.exists(root_dir):
            continue

        for folder_path, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if not (d.startswith('T2026') or d.startswith('bak'))]
            if _is_valid_folder(folder_path, files):
                candidates[folder_path] = tuple(files)

    return sorted(candidates.items(), key=_folder_sort_key)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize MathBench-style benchmark results and optional output lengths.'
    )
    parser.add_argument(
        '--skip-output-tokens',
        action='store_true',
        help='Skip output token length calculation and leave Len as "-".',
    )
    parser.add_argument(
        '--show-warning-log',
        action='store_true',
        help='Print warning logs after the benchmark summary table.',
    )
    return parser.parse_args()


def process_folder(args):
    folder_path, files, tokenizer, item_cache, skip_output_tokens = args
    model_name = folder_path.split("/")[-1]
    prompt_name = folder_path.split("/")[-2]
    folder_scores = {
        'Model': model_name,
        'Prompt': prompt_name
    }
    has_target = False
    folder_warnings = []
    folder_cache_updates = {}

    for benchmark in BENCHMARKS:
        score_path = _find_score_file(
            folder_path, files, model_name, benchmark['score_templates'], benchmark['dataset_names'])
        score_val = benchmark['score_parser'](score_path) if score_path else "-"

        raw_path = _find_raw_prediction_file(
            folder_path, files, model_name, benchmark['dataset_names'])
        len_val = "-"
        truncated_rows = 0
        cell_override = _get_benchmark_cell_override(model_name, prompt_name, benchmark['name'])
        if cell_override is not None:
            score_val, len_val = cell_override
        elif raw_path and not skip_output_tokens:
            len_val, truncated_rows, cache_entry, warning = calculate_avg_tokens(
                raw_path, tokenizer, item_cache, benchmark['name'])
            if cache_entry is not None:
                folder_cache_updates[raw_path] = cache_entry
            if warning is not None:
                folder_warnings.append(warning)

        if score_val != "-" or len_val != "-":
            has_target = True
        folder_scores[benchmark['name']] = format_cell(score_val, len_val)

        if raw_path and truncated_rows > 0:
            folder_warnings.append(
                {
                    'type': 'xlsx_truncation',
                    'benchmark': benchmark['name'],
                    'file': raw_path,
                    'truncated_rows': truncated_rows,
                }
            )

    if not has_target:
        return None
    return dict(scores=folder_scores, warnings=folder_warnings, cache_updates=folder_cache_updates)


def main():
    args = parse_args()
    tokenizer = None if args.skip_output_tokens else load_tokenizer(MODEL_PATH)
    item_cache = {} if args.skip_output_tokens else _load_item_cache(ITEM_CACHE_FILE)
    candidates = collect_candidate_folders()
    results = []
    warnings_data = []
    cache_updated = False

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        task_iter = (
            (folder_path, files, tokenizer, item_cache, args.skip_output_tokens)
            for folder_path, files in candidates
        )
        for item in executor.map(process_folder, task_iter):
            if item is not None:
                results.append(item['scores'])
                warnings_data.extend(item['warnings'])
                if not args.skip_output_tokens:
                    for path, cache_entry in item['cache_updates'].items():
                        item_cache[path] = cache_entry
                        cache_updated = True

    if results:
        df_res = pd.DataFrame(results)
        cols = ['Model', 'Prompt'] + [benchmark['name'] for benchmark in BENCHMARKS]

        for c in cols:
            if c not in df_res.columns:
                df_res[c] = format_cell("-", "-")

        df_res = _apply_baseline_score_overrides(df_res[cols])
        df_res = _prepare_display_table(df_res)
        df_res = _apply_baseline_score_overrides(df_res)
        plot_paths = generate_benchmark_line_plots(df_res)
        df_res = _insert_prompt_separators(df_res)

        _print_summary(df_res, warnings_data, show_warning_log=args.show_warning_log)

        output_file = os.path.join(SCRIPT_DIR, "benchmark_summary_Qwen3.csv")
        df_res.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        if plot_paths:
            print("Line charts saved to:")
            for path in plot_paths:
                print(f"- {path}")
    else:
        print("No benchmark result files found.")

    if cache_updated:
        _save_item_cache(ITEM_CACHE_FILE, item_cache)


if __name__ == "__main__":
    main()
