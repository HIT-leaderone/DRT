import os
import re

import numpy as np
import pandas as pd
from sympy import N, sympify

from .mmmath import AutoScoringJudge
from .text_base import TextBaseDataset
from ..smp import LMUDataRoot, dump, get_intermediate_file_path, load, osp
from ..utils import track_progress_rich


def _ensure_hf_cache():
    root = osp.join(LMUDataRoot(), 'huggingface')
    hub = osp.join(root, 'hub')
    datasets_cache = osp.join(root, 'datasets')
    os.makedirs(hub, exist_ok=True)
    os.makedirs(datasets_cache, exist_ok=True)

    current_home = os.environ.get('HF_HOME')
    if current_home is None or not osp.exists(current_home) or not os.access(current_home, os.W_OK):
        os.environ['HF_HOME'] = root

    current_hub = os.environ.get('HUGGINGFACE_HUB_CACHE')
    if current_hub is None or not osp.exists(current_hub) or not os.access(current_hub, os.W_OK):
        os.environ['HUGGINGFACE_HUB_CACHE'] = hub

    try:
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import file_download as hf_file_download

        hf_constants.HF_HOME = os.environ['HF_HOME']
        hf_constants.HUGGINGFACE_HUB_CACHE = os.environ['HUGGINGFACE_HUB_CACHE']
        if hasattr(hf_constants, 'default_cache_path'):
            hf_constants.default_cache_path = os.environ['HUGGINGFACE_HUB_CACHE']
        hf_file_download.HUGGINGFACE_HUB_CACHE = os.environ['HUGGINGFACE_HUB_CACHE']
    except Exception:
        pass

    try:
        from datasets import config as datasets_config

        datasets_config.HF_DATASETS_CACHE = datasets_cache
    except Exception:
        pass

    return root


def _extract_last_boxed_content(text):
    text = str(text)
    marker = r'\boxed{'
    matches = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        pos = idx + len(marker)
        depth = 1
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        if depth == 0:
            matches.append(text[idx + len(marker):pos - 1].strip())
        start = idx + len(marker)
    return matches[-1] if matches else None


def _extract_xml_answer(text):
    text = str(text)
    matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.S | re.I)
    return matches[-1].strip() if matches else None


def _extract_math_answer(text):
    text = str(text).strip()
    candidate = _extract_xml_answer(text)
    if candidate is None:
        candidate = _extract_last_boxed_content(text)
    if candidate is None:
        answer_matches = re.findall(
            r'(?is)(?:final answer|the answer is|answer is)\s*[:：]?\s*(.+)',
            text,
        )
        if answer_matches:
            candidate = answer_matches[-1].strip()
    if candidate is None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and len(lines[-1]) <= 128:
            candidate = lines[-1]
        else:
            candidate = text
    candidate = re.sub(
        r'(?is)^(?:the\s+)?final\s+answer(?:\s+is)?\s*[:：]?\s*',
        '',
        candidate,
    )
    return candidate.strip()


def _replace_latex_frac(text):
    pattern = re.compile(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
    while pattern.search(text):
        text = pattern.sub(r'\1/\2', text)
    return text


def _normalize_numeric_expression(text):
    text = str(text).strip()
    text = text.replace('$', '').replace(',', '').replace('−', '-').replace('–', '-')
    text = text.replace('π', r'\pi')
    text = text.replace(' ', '').replace('\n', '')
    text = text.rstrip('.。')
    text = _replace_latex_frac(text)
    text = re.sub(r'\\text\{[^{}]*\}', '', text)
    text = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\left|\\right', '', text)
    text = text.strip()
    if text.startswith('='):
        text = text[1:].strip()
    return text


def _extract_gsm8k_answer(text):
    text = str(text).strip()
    if '####' in text:
        candidate = text.split('####')[-1]
    else:
        xml_answer = _extract_xml_answer(text)
        if xml_answer is not None:
            candidate = xml_answer
        else:
            boxed = _extract_last_boxed_content(text)
            if boxed is not None:
                candidate = boxed
            else:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                candidate = lines[-1] if lines else text
                answer_matches = re.findall(
                    r'(?is)(?:final answer|the answer is|answer is|so the answer is)\s*[:：]?\s*(.+)',
                    candidate)
                if answer_matches:
                    candidate = answer_matches[-1]
    candidate = _normalize_numeric_expression(candidate)
    number_matches = re.findall(r'-?(?:\d+\.\d+|\d+/\d+|\d+)', candidate)
    if number_matches:
        return number_matches[-1]
    return candidate


def _gsm8k_is_correct(gold, pred):
    gold = _extract_gsm8k_answer(gold)
    pred = _extract_gsm8k_answer(pred)
    if gold == '' or pred == '':
        return False
    if gold == pred:
        return True
    try:
        gold_expr = sympify(gold)
        pred_expr = sympify(pred)
        return bool(abs(N(gold_expr - pred_expr)) < 1e-8)
    except Exception:
        return False


class MathReasoningDataset(TextBaseDataset):
    TYPE = 'TEXT'
    COD_PROMPT_TEMPLATE = (
        "{Question}\n\n"
        "Think step by step, but keep only a minimum draft for each step with at most 5 words.\n"
        "Respond in this format exactly:\n"
        "<think>minimal draft steps</think>\n"
        "<answer>{AnswerHint}</answer>"
    )
    DENSE_PROMPT_TEMPLATE = (
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
    DIRECT_ANSWER_TEMPLATE = (
        "{Question}\n\n"
        "Provide only the final answer.\n"
        "Do not include any reasoning, explanation, XML tags, or extra text."
    )
    GSM8K_ZERO_SHOT_COT_PROMPT = (
        "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    MATH_ZERO_SHOT_COT_PROMPT = (
        "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )

    GSM8K_HF_REPO = 'openai/gsm8k'
    GSM8K_HF_CONFIG = 'main'
    MATH_HF_REPO = 'EleutherAI/hendrycks_math'
    MATH500_HF_REPO = 'HuggingFaceH4/MATH-500'
    GSM8K_4SHOT_COT_PROMPT = """Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Let's think step by step
Answer:
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step
Answer:
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
Let's think step by step
Answer:
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
Let's think step by step
Answer:
For the first three baskets, the number of apples and oranges in one basket is 9+15=24
In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.
Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.
The number of apples in the fourth basket is 9-2=7
There are also 15-2=13 oranges in the fourth basket
The combined number of oranges and apples in the fourth basket is 13+7=20
The fourth basket also contains 14-2=12 bananas.
In total, the fourth basket has 20+12=32 fruits.
The four baskets together have 32+114=146 fruits.
The answer is 146"""
    MATH_4SHOT_COT_PROMPT = """Problem:
Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations
\\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\\end{align*}
has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."""
    MATH_SUBSETS = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus',
    ]

    DATASET_SPECS = {
        'GSM8K': dict(task='gsm8k', cache_file='GSM8K.tsv', shots='0shot'),
        'GSM8K_0shot': dict(task='gsm8k', cache_file='GSM8K.tsv', shots='0shot'),
        'GSM8K_4shot': dict(task='gsm8k', cache_file='GSM8K.tsv', shots='4shot'),
        'MATH': dict(task='math', cache_file='MATH.tsv', shots='4shot'),
        'MATH_0shot': dict(task='math', cache_file='MATH.tsv', shots='0shot'),
        'MATH_4shot': dict(task='math', cache_file='MATH.tsv', shots='4shot'),
        'MATH-500': dict(task='math500', cache_file='MATH-500.tsv', shots='0shot'),
        'MATH-500_0shot': dict(task='math500', cache_file='MATH-500.tsv', shots='0shot'),
        'MATH-500_4shot': dict(task='math500', cache_file='MATH-500.tsv', shots='4shot'),
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SPECS)

    def post_build(self, dataset):
        self.spec = self.DATASET_SPECS[dataset]
        self.task = self.spec['task']
        self.shots = self.spec['shots']
        self.prompt_prefix = (
            self.GSM8K_4SHOT_COT_PROMPT if self.task == 'gsm8k' else self.MATH_4SHOT_COT_PROMPT
        )

    def load_data(self, dataset):
        spec = self.DATASET_SPECS[dataset]
        data_file = osp.join(LMUDataRoot(), spec['cache_file'])
        if osp.exists(data_file):
            return load(data_file)

        if spec['task'] == 'gsm8k':
            data = self._build_gsm8k_dataframe()
        elif spec['task'] == 'math500':
            data = self._build_math500_dataframe()
        else:
            data = self._build_math_dataframe()
        dump(data, data_file)
        return data

    def _load_hf_dataset(self, repo, config=None, split='test'):
        _ensure_hf_cache()
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError('Loading GSM8K/MATH requires the `datasets` package.') from err
        cache_dir = osp.join(LMUDataRoot(), 'huggingface', 'datasets')
        return load_dataset(repo, config, split=split, cache_dir=cache_dir)

    def _build_gsm8k_dataframe(self):
        dataset = self._load_hf_dataset(self.GSM8K_HF_REPO, self.GSM8K_HF_CONFIG, split='test')
        records = []
        for idx, item in enumerate(dataset):
            records.append(
                dict(
                    index=idx,
                    question=item['question'].strip(),
                    answer=_extract_gsm8k_answer(item['answer']),
                    solution=item['answer'].strip(),
                ))
        return pd.DataFrame(records)

    def _build_math_dataframe(self):
        records = []
        idx = 0
        for subset in self.MATH_SUBSETS:
            dataset = self._load_hf_dataset(self.MATH_HF_REPO, subset, split='test')
            for item in dataset:
                solution = item['solution'].strip()
                answer = _extract_last_boxed_content(solution)
                if answer is None:
                    answer = solution
                records.append(
                    dict(
                        index=idx,
                        question=item['problem'].strip(),
                        answer=answer.strip(),
                        solution=solution,
                        subject=item.get('type', subset),
                        level=item.get('level', ''),
                        source_subset=subset,
                    ))
                idx += 1
        return pd.DataFrame(records)

    def _build_math500_dataframe(self):
        dataset = self._load_hf_dataset(self.MATH500_HF_REPO, split='test')
        records = []
        for idx, item in enumerate(dataset):
            answer = str(item.get('answer', '')).strip()
            solution = str(item.get('solution', '')).strip()
            if answer == '':
                answer = _extract_last_boxed_content(solution) or solution
            records.append(
                dict(
                    index=idx,
                    question=str(item['problem']).strip(),
                    answer=answer,
                    solution=solution,
                    subject=str(item.get('subject', '')),
                    level=item.get('level', ''),
                    unique_id=str(item.get('unique_id', '')),
                    source_subset='math500',
                ))
        return pd.DataFrame(records)

    def _get_prompt_type(self):
        return os.environ.get('PROMPT_TYPE', '')

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        prompt_type = self._get_prompt_type()
        use_cod = prompt_type.startswith('CoD')
        use_short_cot = prompt_type.startswith('Short-COT')
        use_direct_answer = prompt_type == 'Directly-Answer'
        if self.task == 'gsm8k':
            question = f'Question: {line["question"]}'
            if use_cod:
                query_prompt = self.COD_PROMPT_TEMPLATE.format(
                    Question=question,
                    AnswerHint='single number or simplified fraction',
                )
            elif use_short_cot:
                query_prompt = self.DENSE_PROMPT_TEMPLATE.format(Question=question)
            elif use_direct_answer:
                query_prompt = (
                    self.DIRECT_ANSWER_TEMPLATE.format(Question=question) +
                    '\nAnswer with a single number or a simplified fraction.'
                )
            elif self.shots == '0shot':
                query_prompt = self.GSM8K_ZERO_SHOT_COT_PROMPT.format(question=line["question"])
            else:
                query_prompt = f"{question}\nLet's think step by step\nAnswer:"
        else:
            question = f'Problem:\n{line["question"]}'
            if use_cod:
                query_prompt = self.COD_PROMPT_TEMPLATE.format(
                    Question=question,
                    AnswerHint='final mathematical answer',
                )
            elif use_short_cot:
                query_prompt = self.DENSE_PROMPT_TEMPLATE.format(Question=question)
            elif use_direct_answer:
                query_prompt = (
                    self.DIRECT_ANSWER_TEMPLATE.format(Question=question) +
                    '\nOutput only the final mathematical answer.'
                )
            elif self.shots == '0shot':
                query_prompt = self.MATH_ZERO_SHOT_COT_PROMPT.format(problem=line["question"])
            else:
                query_prompt = f'{question}\n\nSolution:'
        prompt = query_prompt if self.shots == '0shot' else f'{self.prompt_prefix}\n\n{query_prompt}'
        return [dict(type='text', value=prompt)]

    def evaluate(self, eval_file, **judge_kwargs):
        if self.task == 'gsm8k':
            return self._evaluate_gsm8k(eval_file)
        return self._evaluate_math(eval_file, nproc=judge_kwargs.get('nproc', 4))

    def _evaluate_gsm8k(self, eval_file):
        data = load(eval_file)
        data['answer'] = [str(x) for x in data['answer']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['extracted_prediction'] = [_extract_gsm8k_answer(x) for x in data['prediction']]
        data['hit'] = [_gsm8k_is_correct(g, p) for g, p in zip(data['answer'], data['prediction'])]
        dump(data, eval_file)

        score = {
            'overall': float(np.mean(data['hit'])),
            'num_examples': int(len(data)),
            'num_correct': int(np.sum(data['hit'])),
        }
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(score, score_file)
        return score

    @staticmethod
    def _strip_inline_latex(text):
        """Strip inline math delimiters \\(...\\) and normalize \\text{}, \\dfrac."""
        text = str(text).strip()
        text = text.replace('π', r'\pi').replace('−', '-').replace('–', '-')
        # Remove \(...\) wrapping
        m = re.match(r'^\\\((.+)\\\)$', text, flags=re.S)
        if m:
            text = m.group(1).strip()
        # \text{X} -> X
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
        # \dfrac -> \frac
        text = text.replace('\\dfrac', '\\frac')
        return text.strip().rstrip('.。')

    @staticmethod
    def _normalize_plain_text_answer(text):
        text = str(text).strip().rstrip('.。')
        # Lowercase word answers like "East" vs "east" without touching symbol-like answers such as "AB".
        if re.fullmatch(r'[A-Za-z ]+', text) and (re.search(r'[a-z]', text) or ' ' in text):
            text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def _evaluate_math(self, eval_file, nproc=4):
        data = load(eval_file)
        data['answer'] = [str(x) for x in data['answer']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['prediction'] = [
            _extract_math_answer(x)
            for x in data['prediction']
        ]
        # Normalize both answer and prediction to handle format variants
        data['answer'] = [self._strip_inline_latex(x) for x in data['answer']]
        data['prediction'] = [self._strip_inline_latex(x) for x in data['prediction']]
        data['answer'] = [self._normalize_plain_text_answer(x) for x in data['answer']]
        data['prediction'] = [self._normalize_plain_text_answer(x) for x in data['prediction']]

        try:
            judger = AutoScoringJudge()
        except ImportError as err:
            raise ImportError(
                'MATH evaluation requires `antlr4-python3-runtime==4.11.*` so SymPy can parse LaTeX.'
            ) from err

        tups = [dict(expression1=x, expression2=y) for x, y in zip(data['answer'], data['prediction'])]
        data['hit'] = track_progress_rich(judger.judge, tups, nproc=max(1, min(int(nproc), 16)))
        dump(data, eval_file)

        score = {'overall': float(np.mean(data['hit']))}
        if 'subject' in data:
            for subject in sorted(set(data['subject'])):
                score[f'Subject-{subject}'] = float(np.mean(data[data['subject'] == subject]['hit']))
        if 'level' in data:
            for level in sorted(set(data['level'])):
                score[f'Level-{level}'] = float(np.mean(data[data['level'] == level]['hit']))

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(score, score_file)
        return score
