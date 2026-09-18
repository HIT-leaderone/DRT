from ...smp import *
from .multiple_choice import extract_answer_from_item
from ...utils import can_infer
import numpy as np
import ast
import re

FAIL_MSG = 'Failed to obtain answer via API.'

TASK_CATEGORIES = [
    'SR','IMC','TCI','TA','MHR','PAR','CTI',
]

OPTION_LETTERS = "ABCDEF"

def extract_option_gpt(model, input_item, dataset_name):
    options = ast.literal_eval(input_item['candidates'])
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']

def get_dimension_rating(data_path, score_col='score', type_col='question_type'):
    data = load(data_path)
    acc_by_type = {}
    for qtype, group in data.groupby(type_col):
        correct = (group[score_col] == 1).sum()
        total = len(group)
        acc = correct / total if total > 0 else 0
        acc_by_type[qtype] = {
            'correct': int(correct),
            'total': int(total),
            'acc': acc
        }

    total_correct = (data[score_col] == 1).sum()
    total_count = len(data)
    total_acc = total_correct / total_count if total_count > 0 else 0

    result = {
        'acc_by_type': acc_by_type,
        'total': {
            'correct': int(total_correct),
            'total': int(total_count),
            'acc': total_acc
        }
    }

    return result


def _extract_answer_span(pred):
    pred = re.sub(r'.*?</think>', '', str(pred), flags=re.DOTALL).strip()
    matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', pred, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else pred.strip()


def _normalize_choice_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' .,:;!?-')
    return text


def _build_choice_dict(input_item):
    if input_item is None or 'candidates' not in input_item:
        return {}

    try:
        options = ast.literal_eval(input_item['candidates'])
    except Exception:
        return {}

    choices = {}
    for option in options:
        match = re.match(r'^\s*([A-F])\.\s*(.*)$', str(option).strip(), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        choices[match.group(1).upper()] = match.group(2).strip()
    return choices


def _extract_explicit_option(answer_span):
    span = str(answer_span).strip()
    if not span:
        return None

    patterns = [
        r'^\s*(?:final\s+answer|correct\s+answer|answer|option)\s*(?:is)?\s*[:：\-]?\s*[\(\[]?\s*([A-F])\s*[\)\]]?(?:[\s\.:：\-]|$)',
        r'^\s*[\(\[]?\s*([A-F])\s*[\)\]]?(?:[\s\.:：\-]|$)',
    ]
    for pattern in patterns:
        match = re.match(pattern, span, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def _extract_option_by_exact_choice_text(answer_span, input_item):
    options = _build_choice_dict(input_item)
    if not options:
        return None

    answer_norm = _normalize_choice_text(answer_span)
    if not answer_norm:
        return None

    matched = set()
    for label, option_text_raw in options.items():
        option_text = _normalize_choice_text(option_text_raw)
        if answer_norm == option_text:
            matched.add(label)

    if len(matched) == 1:
        return next(iter(matched))
    return None


def _extract_option_by_local_inference(answer_span, input_item):
    choices = _build_choice_dict(input_item)
    if not choices:
        return None
    inferred = can_infer(answer_span, choices)
    return inferred if inferred else None


def extract_option(pred, input_item=None):
    answer_span = _extract_answer_span(pred)

    explicit = _extract_explicit_option(answer_span)
    if explicit is not None and explicit in OPTION_LETTERS:
        return explicit

    from_choice_text = _extract_option_by_exact_choice_text(answer_span, input_item)
    if from_choice_text is not None and from_choice_text in OPTION_LETTERS:
        return from_choice_text

    inferred = _extract_option_by_local_inference(answer_span, input_item)
    if inferred is not None and inferred in OPTION_LETTERS:
        return inferred

    return 'WRONG'
