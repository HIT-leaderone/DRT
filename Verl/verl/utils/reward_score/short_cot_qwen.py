import re
import json
import ast
import os
from mathruler.grader import extract_boxed_content, grade_answer

import aiohttp
import logging

logger = logging.getLogger(__name__)

DEFAULT_ROUTER_MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
XML_TAG_PATTERN = re.compile(r"</?[^>]+>")

async def generate_chat_aiohttp(router_address: str, messages: list[dict], sampling_params: dict, **request_kwargs):
    model_name = (
        request_kwargs.pop("model", None)
        or request_kwargs.pop("reward_model_name", None)
        or os.environ.get("REWARD_OPENAI_MODEL_NAME")
        or os.environ.get("GRM_PATH")
        or DEFAULT_ROUTER_MODEL_NAME
    )
    # 1. 构造 OpenAI 兼容的 payload
    # 注意：OpenAI API 将 temperature 等参数直接放在顶层，而不是 sampling_params 字典里
    payload = {
        "model": model_name,  # vLLM 的 OpenAI 接口需要该字段，优先与实际 served model 对齐
        "messages": messages,
        # 将 sampling_params 展开合并到 payload 中
        **sampling_params
    }
    for key, value in dict(request_kwargs).items():
        if value is not None:
            payload[key] = value

    # 2. 指向 OpenAI 兼容的 endpoint
    url = f"http://{router_address}/v1/chat/completions"

    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output_json = json.loads(output)
                if output_json.get("choices") and output_json["choices"][0]["message"].get("content"): # 加强健壮性
                    return output_json["choices"][0]["message"]["content"] # 这里原代码有一处逻辑没返回，顺便修了
                else:
                    logger.error(f"API Error: {output_json}")
                    return "{}" # 返回空JSON字符串防止后续解析挂掉
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {output} | Error: {e}")
                return "{}"
    finally:
        await session.close()

def extract_parts(text):
    """
    分离 <think> 和 最终答案
    """
    if not text:
        return None, None

    think_match = THINK_TAG_PATTERN.search(text)
    if not think_match:
        return None, None

    pred_think = think_match.group(1).strip()

    answer_match = ANSWER_TAG_PATTERN.search(text)
    if answer_match:
        pred_final = answer_match.group(1).strip()
    else:
        tail_text = text[think_match.end() :].strip()
        pred_final = clean_final_answer(tail_text)

    pred_final = clean_final_answer(pred_final)
    return pred_think, pred_final


def clean_final_answer(text):
    """清洗最终答案，兼容 <answer> 标签、boxed 答案和纯文本答案。"""
    if text is None:
        return None

    text = text.strip()
    if not text:
        return ""

    answer_match = ANSWER_TAG_PATTERN.search(text)
    if answer_match:
        text = answer_match.group(1).strip()

    boxed_content = extract_boxed_content(text)
    if boxed_content != "None":
        return boxed_content.strip()

    return XML_TAG_PATTERN.sub(" ", text).strip()

def count_words(text):
    """
    使用正则统计有效词数，忽略标点符号。
    例如: "Key visual: red/blue." -> ['Key', 'visual', 'red', 'blue'] -> 4 words
    """
    # \w+ 匹配所有字母、数字和下划线
    # 如果涉及中文，\w 也能匹配汉字
    words = re.findall(r'\w+', text)
    return len(words)

def check_density(think_content, max_len=10):
    """
    密度检查：
    1. 按 -> 或 + 切分逻辑块
    2. 对每个块进行正则分词统计
    """
    # 1. 结构切分：按 -> 或 + 将思考过程切分为多个 Segment
    segments = re.split(r'->|\+|;|\.', think_content)

    violations = 0

    for seg in segments:
        # 2. 词数统计：使用正则提取单词
        w_count = count_words(seg)

        # 3. 阈值判断
        if w_count > max_len:
            violations += 1
            # (可选) 打印日志用于调试
            # print(f"[Violation] Len: {w_count} | Content: {seg.strip()[:30]}...")

    return violations

# --- 定义按步骤阅卷的 Prompt ---
STEP_GRADER_PROMPT = """
You are a strict Math Teacher grading a student's reasoning for a {problem_type}.
**Original Problem:**
{problem_text}
**Reference Solution (Standard Steps):**
{gt_steps_str}
**Student's Thinking Process:**
{pred_think}
**Student's Final Answer:**
{pred_final}
**Grading Tasks:**
1. **Calculate `matched_step_count`**:
   - Iterate through the Reference Steps.
   - A step counts as "matched" **IF AND ONLY IF**:
     a) The student derives the correct logic or intermediate result corresponding to that reference step.
     b) The student's reasoning for this step is **free of hallucinations** (no made-up numbers or unproven assumptions).
   - If a step is skipped or derived using hallucinated values, do NOT count it.
2. **Determine `has_severe_hallucination`**:
   - Check the entire reasoning chain leading to the Final Answer.
   - Set to **True** if the student invents numbers, conditions, visual observations, or mathematical properties not provided in the problem or not validly derived.
   - Set to **False** if the reasoning is grounded in the provided information, even if there are calculation errors.
**Output (JSON):**
{{
    "matched_step_count": (integer), // Count of steps that are BOTH logically correct AND hallucination-free.
    "total_reference_steps": (integer), // Total number of steps in the Reference.
    "has_severe_hallucination": (boolean), // True if the path to the final answer contains ANY hallucination.
    "reason": "Brief explanation of which steps matched and where hallucination occurred (if any)."
}}
"""

STEP_BONUS_GRADER_PROMPT = """
You are a strict Math Teacher grading a student's reasoning for a {problem_type}.
**Original Problem:**
{problem_text}
**Reference Solution (Standard Steps):**
{gt_steps_str}
**Student's Thinking Process:**
{pred_think}
**Student's Final Answer:**
{pred_final}
**Grading Tasks:**
1. **Calculate `matched_step_count`**:
   - Iterate through the Reference Steps.
   - A step counts as "matched" if the student reaches the corresponding correct logic or intermediate result.
   - A matched step may still be counted as matched even if the derivation contains hallucination. Hallucination should be tracked separately.
2. **Calculate `effective_reasoning_step_count`**:
   - Count the student's distinct, non-trivial, solution-advancing reasoning steps.
   - This includes the main solution path and any genuinely useful extra verification or exploration steps.
   - Do NOT count repetition, filler, paraphrases of the same idea, or artificial over-splitting of one idea into many tiny fragments.
3. **Calculate `deep_exploration_step_count`**:
   - Count only the subset of steps that go BEYOND the minimal necessary solution path and provide genuinely useful extra exploration.
   - Examples that MAY count: checking a non-obvious branch, verifying a fragile visual assumption, comparing two plausible methods before choosing one, or adding a meaningful consistency check.
   - Examples that must NOT count: simply splitting one necessary step into many smaller pieces, repeating the same idea, stylistic verbosity, or irrelevant side branches.
   - Every deep exploration step MUST also be included in `effective_reasoning_step_count`.
   - Therefore `deep_exploration_step_count <= effective_reasoning_step_count`.
4. **Do NOT estimate any target counts**:
   - The reward code will use the number of reference steps as the target for effective reasoning steps.
   - The reward code will use a fixed external cap for deep exploration steps.
   - Your job is only to count the student's actual effective reasoning steps and actual deep exploration steps as defined above.
5. **Determine hallucination inside matched steps only**:
   - Consider ONLY the subset of student reasoning that you already counted toward `matched_step_count`.
   - `hallucinated_matched_step_count` means: among the steps already counted in `matched_step_count`, how many matched steps reach the correct intermediate result but rely on invented numbers, invented visual observations, invalid assumptions, or unjustified properties.
   - Ignore unmatched, skipped, incorrect, or extra reasoning when deciding this hallucination field.
6. **Output JSON only**:
{{
    "matched_step_count": (integer),
    "total_reference_steps": (integer),
    "effective_reasoning_step_count": (integer),
    "deep_exploration_step_count": (integer), // integer in [0, effective_reasoning_step_count]
    "hallucinated_matched_step_count": (integer), // integer in [0, matched_step_count]
    "has_hallucination_in_matched_steps": (boolean),
    "reason": "Brief explanation of matched steps, effective reasoning steps, deep exploration steps, and whether matched steps contain hallucinated derivations."
}}
"""

STEP_BONUS_GRADER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_step_count": {"type": "integer"},
        "total_reference_steps": {"type": "integer"},
        "effective_reasoning_step_count": {"type": "integer"},
        "deep_exploration_step_count": {"type": "integer"},
        "hallucinated_matched_step_count": {"type": "integer"},
        "has_hallucination_in_matched_steps": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "matched_step_count",
        "total_reference_steps",
        "effective_reasoning_step_count",
        "deep_exploration_step_count",
        "hallucinated_matched_step_count",
        "has_hallucination_in_matched_steps",
        "reason",
    ],
    "additionalProperties": False,
}

def parse_ground_truth(gt_str):
    """解析 GT，提取 steps 列表"""
    try:
        gt_data = ast.literal_eval(gt_str)
        if isinstance(gt_data, dict):
            return gt_data.get('steps', []), gt_data.get('answer', '')
    except:
        pass
    try:
        gt_data = json.loads(gt_str)
        if isinstance(gt_data, dict):
            return gt_data.get('steps', []), gt_data.get('answer', '')
    except:
        pass
    return [], gt_str

def infer_problem_type(data_source, extra_info=None):
    extra_info = extra_info or {}
    dataset = str(extra_info.get("dataset", "") or data_source or "").lower()
    data_type = str(extra_info.get("data_type", "") or "").lower()

    if dataset == "geo3k" or data_type == "geometry":
        return "geometry problem with visual context"
    if dataset == "dolci_math" or data_type in {"math", "text_only_math"}:
        return "text-only math problem"
    if "geo" in dataset or "geometry" in dataset or data_type == "vision":
        return "geometry or visual math problem"
    return "math problem"


async def evaluate_steps_with_gpt(
    pred_think,
    pred_final,
    gt_steps,
    router_address,
    reward_model_name=None,
    problem_text="",
    problem_type="math problem",
    max_tokens=512,
):
    """
    调用 GPT 计算匹配的步骤数量
    """
    try:
        if not router_address:
            logger.warning("reward_router_address is empty, skip GPT step grading.")
            return {"matched_step_count": 0, "total_reference_steps": len(gt_steps), "has_severe_hallucination": False}

        # 格式化步骤列表，方便 GPT 阅读
        gt_steps_formatted = ""
        for idx, step in enumerate(gt_steps):
            gt_steps_formatted += f"Step {idx+1}: {step}\n"

        prompt = STEP_GRADER_PROMPT.format(
            problem_type=problem_type,
            problem_text=(problem_text or "N/A")[:4000],
            gt_steps_str=gt_steps_formatted,
            pred_think=pred_think[:6000],
            pred_final=pred_final
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = await generate_chat_aiohttp(
            router_address=router_address,
            messages=messages,
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
            model=reward_model_name,
        )  # 温度设为0，追求稳定

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            # 兜底：假设没过
            return {"matched_step_count": 0, "total_reference_steps": len(gt_steps), "has_severe_hallucination": False}

    except Exception as e:
        print(f"[GPT Grader Error]: {e}")
        return {"matched_step_count": 0, "total_reference_steps": len(gt_steps), "has_severe_hallucination": False}


async def evaluate_step_bonus_with_gpt(
    pred_think,
    pred_final,
    gt_steps,
    router_address,
    reward_model_name=None,
    problem_text="",
    problem_type="math problem",
    max_tokens=768,
):
    """
    调用 GPT 同时评估：
    1. 参考步骤匹配率
    2. 学生推理中的有效 step 数
    3. 仅在“已匹配成功的步骤”里是否存在带幻觉的推导
    """
    default_result = {
        "matched_step_count": 0,
        "total_reference_steps": len(gt_steps),
        "effective_reasoning_step_count": 0,
        "deep_exploration_step_count": 0,
        "hallucinated_matched_step_count": 0,
        "has_hallucination_in_matched_steps": False,
    }

    try:
        if not router_address:
            logger.warning("reward_router_address is empty, skip GPT step-bonus grading.")
            return default_result

        gt_steps_formatted = ""
        for idx, step in enumerate(gt_steps):
            gt_steps_formatted += f"Step {idx+1}: {step}\n"

        prompt = STEP_BONUS_GRADER_PROMPT.format(
            problem_type=problem_type,
            problem_text=(problem_text or "N/A")[:4000],
            gt_steps_str=gt_steps_formatted,
            pred_think=pred_think[:6000],
            pred_final=pred_final,
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = await generate_chat_aiohttp(
            router_address=router_address,
            messages=messages,
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
            model=reward_model_name,
        )

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return default_result

        result = json.loads(match.group(0))
        matched_step_count = max(0, min(int(result.get("matched_step_count", 0)), len(gt_steps)))
        effective_reasoning_step_count = max(0, int(result.get("effective_reasoning_step_count", 0)))
        deep_exploration_step_count = max(0, int(result.get("deep_exploration_step_count", 0)))
        deep_exploration_step_count = min(deep_exploration_step_count, effective_reasoning_step_count)
        hallucinated_matched_step_count = max(
            0,
            min(int(result.get("hallucinated_matched_step_count", 0)), matched_step_count),
        )
        has_hallucination_in_matched_steps = bool(
            result.get("has_hallucination_in_matched_steps", hallucinated_matched_step_count > 0)
        )

        return {
            "matched_step_count": matched_step_count,
            "total_reference_steps": len(gt_steps),
            "effective_reasoning_step_count": effective_reasoning_step_count,
            "deep_exploration_step_count": deep_exploration_step_count,
            "hallucinated_matched_step_count": hallucinated_matched_step_count,
            "has_hallucination_in_matched_steps": (
                has_hallucination_in_matched_steps or hallucinated_matched_step_count > 0
            ),
        }
    except Exception as e:
        print(f"[GPT Step Bonus Grader Error]: {e}")
        return default_result


async def evaluate_step_bonus_with_gpt_guided_json(
    pred_think,
    pred_final,
    gt_steps,
    router_address,
    reward_model_name=None,
    problem_text="",
    problem_type="math problem",
    max_tokens=768,
):
    """
    与 evaluate_step_bonus_with_gpt 逻辑一致，但额外通过 vLLM OpenAI-compatible
    guided_json 约束输出必须满足 JSON schema。
    """
    default_result = {
        "matched_step_count": 0,
        "total_reference_steps": len(gt_steps),
        "effective_reasoning_step_count": 0,
        "deep_exploration_step_count": 0,
        "hallucinated_matched_step_count": 0,
        "has_hallucination_in_matched_steps": False,
    }

    try:
        if not router_address:
            logger.warning("reward_router_address is empty, skip GPT step-bonus grading.")
            return default_result

        gt_steps_formatted = ""
        for idx, step in enumerate(gt_steps):
            gt_steps_formatted += f"Step {idx+1}: {step}\n"

        prompt = STEP_BONUS_GRADER_PROMPT.format(
            problem_type=problem_type,
            problem_text=(problem_text or "N/A")[:4000],
            gt_steps_str=gt_steps_formatted,
            pred_think=pred_think[:6000],
            pred_final=pred_final,
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = await generate_chat_aiohttp(
            router_address=router_address,
            messages=messages,
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens},
            model=reward_model_name,
            guided_json=STEP_BONUS_GRADER_JSON_SCHEMA,
        )

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return default_result

        result = json.loads(match.group(0))
        matched_step_count = max(0, min(int(result.get("matched_step_count", 0)), len(gt_steps)))
        effective_reasoning_step_count = max(0, int(result.get("effective_reasoning_step_count", 0)))
        deep_exploration_step_count = max(0, int(result.get("deep_exploration_step_count", 0)))
        deep_exploration_step_count = min(deep_exploration_step_count, effective_reasoning_step_count)
        hallucinated_matched_step_count = max(
            0,
            min(int(result.get("hallucinated_matched_step_count", 0)), matched_step_count),
        )
        has_hallucination_in_matched_steps = bool(
            result.get("has_hallucination_in_matched_steps", hallucinated_matched_step_count > 0)
        )

        return {
            "matched_step_count": matched_step_count,
            "total_reference_steps": len(gt_steps),
            "effective_reasoning_step_count": effective_reasoning_step_count,
            "deep_exploration_step_count": deep_exploration_step_count,
            "hallucinated_matched_step_count": hallucinated_matched_step_count,
            "has_hallucination_in_matched_steps": (
                has_hallucination_in_matched_steps or hallucinated_matched_step_count > 0
            ),
        }
    except Exception as e:
        print(f"[GPT Step Bonus Guided JSON Grader Error]: {e}")
        return default_result

async def compute_score_qwen(data_source, solution_str, ground_truth,
                  extra_info=None,
                  reward_router_address=None,
                  reward_model_name=None,
                  format_score=1.0,
                  style_score=0.1,
                  density_penalty=0.05,
                  max_segment_len=15,
                  score=0.9,           # 答案正确的基础分
                  process_weight=0.8,  # 过程分的满分权重 (如果答案错，最高能拿多少分)
                  hallucination_penalty=0.4, # 幻觉扣分
                  **kwargs):

    total_score = 0.0

    # 1. 解析数据
    gt_steps, gt_final = parse_ground_truth(ground_truth)
    pred_think, pred_final = extract_parts(solution_str)
    problem_text = (extra_info or {}).get("problem", "")
    problem_type = infer_problem_type(data_source, extra_info)

    # [Fail] 格式错误
    if pred_think is None or pred_final is None:
        return -1.0 * format_score

    if "->" in pred_think:
        total_score += style_score
    else:
        total_score -= style_score

    # B. 密度惩罚 (使用新的分词逻辑)
    violation_count = check_density(pred_think, max_len=max_segment_len)

    if violation_count > 0:
        # 扣分：违规次数 * 单次惩罚
        total_score -= (min(5, violation_count) * density_penalty)

    # 2. 答案正确性检查 (MathRuler)
    is_answer_correct = grade_answer(pred_final, gt_final)

    # 3. GPT 过程阅卷
    # 只有当有参考步骤时才进行详细阅卷
    if gt_steps and len(gt_steps) > 0:

        gpt_result = await evaluate_steps_with_gpt(
            pred_think,
            pred_final,
            gt_steps,
            reward_router_address,
            reward_model_name=reward_model_name,
            problem_text=problem_text,
            problem_type=problem_type,
        )
        # print("ground_truth:", ground_truth, "\nsolution_str:", solution_str, "\ngpt result:", gpt_result)

        matched_count = gpt_result.get("matched_step_count", 0)
        total_steps = len(gt_steps) # 使用 Python 统计的长度，比 GPT 返回的更准
        has_hallucination = gpt_result.get("has_severe_hallucination", False)

        # 计算完成率 (0.0 - 1.0)
        completion_ratio = 0.0
        if total_steps > 0:
            completion_ratio = min(matched_count / total_steps, 1.0)

        # --- 评分策略分支 ---

        if is_answer_correct:
            # Case A: 答案正确
            # 基础分拿满
            total_score += score

            # 检查幻觉：如果答案对了但 GPT 说有严重幻觉，扣分
            if has_hallucination:
                # print(f"Correct Answer but Hallucinated! Penalty applied.")
                total_score -= hallucination_penalty

            # (可选) 奖励过程：如果用户希望答案对+过程对有额外奖励，可以在这里加
            # 但通常答案对就是满分，除非有幻觉

        else:
            # Case B: 答案错误 -> 给过程分 (Partial Credit)
            # 分数 = 完成比例 * 过程权重
            # 例如：5步里做对了3步，权重是1.0，则得 0.6 分
            step_score = completion_ratio * process_weight
            total_score += step_score

            # 如果有幻觉，在过程分基础上再扣一点 (可选，防止负分)
            # if has_hallucination:
            #     total_score -= step_score / 2

        print(f"Total: {total_score:.2f} | Ans: {is_answer_correct} | Steps: {matched_count}/{total_steps} | Hallucination: {has_hallucination}")

    else:
        # 如果没有 GT Steps 数据，回退到只看答案
        if is_answer_correct:
            total_score += score
        print(f"Total: {total_score:.2f} | Ans: {is_answer_correct} | No GT Steps")

    return total_score


import math


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def resolve_source_key(data_source, extra_info=None) -> str:
    extra_info = extra_info or {}
    for candidate in (extra_info.get("dataset"), data_source):
        if candidate:
            return str(candidate).strip().lower()
    return ""


def get_deep_exploration_difficulty_schedule(
    data_source,
    extra_info=None,
    difficulty_by_source=None,
    default_threshold: int = 9,
    default_cap: int = 16,
) -> tuple[int, int, str]:
    source_key = resolve_source_key(data_source, extra_info)
    normalized_mapping = {}
    if difficulty_by_source:
        for key, value in dict(difficulty_by_source).items():
            normalized_mapping[str(key).strip().lower()] = dict(value)

    source_cfg = normalized_mapping.get(source_key, {})
    threshold = int(source_cfg.get("threshold", default_threshold))
    cap = int(source_cfg.get("cap", default_cap))
    return threshold, cap, source_key


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def estimate_tokens(text: str) -> int:
    """
    轻量估算 token 数（避免依赖 tokenizer）。
    - 对英文/数字：按“词”计
    - 对中文：按“汉字”粗略计
    - 混合文本：两者相加
    你如果有真实 tokenizer（如 tiktoken / sentencepiece），建议替换这里。
    """
    if not text:
        return 0
    # 英文/数字“词”
    en_words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    en_count = len(en_words)
    # 中文汉字数
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_count = len(zh_chars)
    # 标点/符号给少量权重（避免极端符号刷长度）
    punct = re.findall(r"[^\w\s\u4e00-\u9fff]", text)
    punct_count = len(punct)
    # 一个很粗的折中：中文 1 字≈1 token，英文 1 词≈1 token，标点按 1/3 token
    return int(zh_count + en_count + punct_count / 3)


def length_bonus(L: int,
                 Lmin: int,
                 Lmax: int,
                 beta: float = 1.0,
                 alpha: float = 0.005,
                 eta: float = 0.01) -> float:
    """
    目标区间长度奖励（软约束）：
    - L < Lmin：线性惩罚（鼓励“不要太短”）
    - Lmin <= L <= Lmax：给常数正奖励 beta（鼓励“到达一个足够深的推理区间”）
    - L > Lmax：线性惩罚（防止无限膨胀）
    """
    if L < Lmin:
        return -alpha * (Lmin - L)
    elif L <= Lmax:
        return +beta
    else:
        return -eta * (L - Lmax)


def split_reasoning_steps(think_content: str) -> list[str]:
    if not think_content:
        return []
    parts = re.split(r"(?:->|=>|\n+|;|；)", think_content)
    return [part.strip() for part in parts if part and part.strip()]


def count_reasoning_steps(think_content: str, min_tokens_per_step: int = 4) -> int:
    """
    统计“有内容”的推理 step 数，避免模型通过空箭头或极短碎片刷 step reward。
    """
    segments = split_reasoning_steps(think_content)
    return sum(1 for segment in segments if estimate_tokens(segment) >= min_tokens_per_step)


def build_difficulty_profile(
    total_steps: int,
    Lmin: int,
    Lmax: int,
    gate_tau: float,
    len_lambda: float,
    process_weight: float,
    hard_step_threshold: int = 9,
    hard_step_cap: int = 18,
    hard_Lmin_per_step: int = 12,
    hard_Lmax_per_step: int = 28,
    hard_gate_tau_relax: float = 0.08,
    hard_len_lambda_scale: float = 1.0,
    hard_process_weight_scale: float = 0.4,
    max_process_weight: float = 1.2,
) -> dict:
    capped_total_steps = min(total_steps, hard_step_cap)
    scaled_extra_steps = max(capped_total_steps - hard_step_threshold, 0)

    if hard_step_cap <= hard_step_threshold:
        hard_ratio = 1.0 if total_steps > hard_step_threshold else 0.0
    else:
        hard_ratio = clamp(
            (capped_total_steps - hard_step_threshold) / (hard_step_cap - hard_step_threshold),
            0.0,
            1.0,
        )

    effective_Lmin = Lmin + scaled_extra_steps * hard_Lmin_per_step
    effective_Lmax = max(effective_Lmin + 64, Lmax + scaled_extra_steps * hard_Lmax_per_step)
    effective_gate_tau = clamp(gate_tau - hard_ratio * hard_gate_tau_relax, 0.05, 0.95)
    effective_len_lambda = len_lambda * (1.0 + hard_ratio * hard_len_lambda_scale)
    effective_process_weight = min(
        max_process_weight,
        process_weight * (1.0 + hard_ratio * hard_process_weight_scale),
    )

    return {
        "hard_ratio": hard_ratio,
        "capped_total_steps": capped_total_steps,
        "effective_Lmin": effective_Lmin,
        "effective_Lmax": effective_Lmax,
        "effective_gate_tau": effective_gate_tau,
        "effective_len_lambda": effective_len_lambda,
        "effective_process_weight": effective_process_weight,
    }

async def compute_score_qwen_length_bonus(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    reward_model_name=None,
    format_score=1.0,
    style_score=0.1,
    density_penalty=0.05,
    max_segment_len=15,
    score=0.9,                 # 答案正确的基础分
    process_weight=0.8,        # 过程分的满分权重 (如果答案错，最高能拿多少分)
    hallucination_penalty=0.4, # 幻觉扣分
    # -------- 新增：PGLB（Progress-Gated Length Bonus）相关参数 --------
    enable_length_bonus=True,
    # 目标推理长度区间（按 pred_think 的 token 估算）
    Lmin=200,
    Lmax=600,
    # 门控：只有完成率足够高才开始给长度奖励
    gate_tau=0.5,   # completion_ratio 阈值
    gate_k=10.0,    # sigmoid 陡峭度（越大越接近硬阈值）
    # 长度奖励强度
    len_lambda=0.08,  # 长度项总体权重（建议 0.03~0.10 起步）
    len_beta=1.0,     # 区间内常数奖励
    len_alpha=0.005,  # 低于 Lmin 的惩罚斜率
    len_eta=0.01,     # 高于 Lmax 的惩罚斜率（建议大于 alpha 防膨胀）
    # 过程分形状（让“高质量进展”更值钱，降低刷分空间）
    process_gamma=1.5,
    # 是否在“无 GT steps”时也给长度奖励（一般不建议）
    length_bonus_without_gt_steps=False,
    **kwargs
):
    total_score = 0.0
    # 1. 解析数据
    gt_steps, gt_final = parse_ground_truth(ground_truth)
    pred_think, pred_final = extract_parts(solution_str)
    problem_text = (extra_info or {}).get("problem", "")
    problem_type = infer_problem_type(data_source, extra_info)
    # [Fail] 格式错误
    if pred_think is None or pred_final is None:
        return -1.0 * format_score
    # A. 轻微风格奖励：是否出现 "->"
    if "->" in pred_think:
        total_score += style_score
    else:
        total_score -= style_score
    # B. 密度惩罚
    violation_count = check_density(pred_think, max_len=max_segment_len)
    if violation_count > 0:
        total_score -= (min(5, violation_count) * density_penalty)
    # 2. 答案正确性检查
    is_answer_correct = grade_answer(pred_final, gt_final)
    # 3. GPT 过程阅卷（只有当有参考步骤时才进行）
    if gt_steps and len(gt_steps) > 0:
        gpt_result = await evaluate_steps_with_gpt(
            pred_think,
            pred_final,
            gt_steps,
            reward_router_address,
            reward_model_name=reward_model_name,
            problem_text=problem_text,
            problem_type=problem_type,
        )
        matched_count = gpt_result.get("matched_step_count", 0)
        total_steps = len(gt_steps)  # 用 Python 长度更准
        has_hallucination = gpt_result.get("has_severe_hallucination", False)
        completion_ratio = 0.0
        if total_steps > 0:
            completion_ratio = min(matched_count / total_steps, 1.0)
        # --- 评分策略分支 ---
        if is_answer_correct:
            # Case A: 答案正确 -> 基础分拿满
            total_score += score
        else:
            # Case B: 答案错误 -> 给过程分（更偏向高完成率）
            # step_score = completion_ratio^gamma * process_weight
            step_score = (completion_ratio ** process_gamma) * process_weight
            total_score += step_score
        # 幻觉惩罚：覆盖“答案对/错”两种情况（更防刷）
        if has_hallucination:
            total_score -= hallucination_penalty
        # --- PGLB：门控长度奖励（鼓励“有进展的更深推理”，而不是纯刷长度） ---
        if enable_length_bonus:
            L = estimate_tokens(pred_think)
            # gate: 只有无严重幻觉 + completion_ratio 达标才给长度奖励
            # g ∈ (0,1)，completion_ratio < tau 时接近 0
            g = 0.0
            if not has_hallucination:
                g = sigmoid(gate_k * (completion_ratio - gate_tau))
            # 区间奖励：到 [Lmin, Lmax] 得正奖励，过短/过长都会扣
            b = length_bonus(
                L, Lmin, Lmax,
                beta=len_beta,
                alpha=len_alpha,
                eta=len_eta
            )
            total_score += (len_lambda * g * b)
        print(
            f"Total: {total_score:.2f} | Ans: {is_answer_correct} | "
            f"Steps: {matched_count}/{total_steps} | "
            f"Completion: {completion_ratio:.2f} | "
            f"Hallucination: {has_hallucination}"
        )
    else:
        # 没有 GT Steps：回退到只看答案（默认不做长度奖励，避免无监督刷长度）
        if is_answer_correct:
            total_score += score
        if enable_length_bonus and length_bonus_without_gt_steps:
            # 注意：没有 steps 的门控信号非常弱，容易刷长度，不建议默认开启
            L = estimate_tokens(pred_think)
            b = length_bonus(
                L, Lmin, Lmax,
                beta=len_beta,
                alpha=len_alpha,
                eta=len_eta
            )
            # 这里 gate 只能用 “无幻觉” 之类的信号（但你此分支没有 GPT judge）
            # 所以仅作可选示例：用一个很小的权重
            total_score += 0.1 * len_lambda * b
        print(f"Total: {total_score:.2f} | Ans: {is_answer_correct} | No GT Steps")
    return total_score

async def compute_score_qwen_gpt_step_bonus(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    reward_model_name=None,
    format_score=1.0,
    style_score=0.1,
    density_penalty=0.05,
    max_segment_len=15,
    score=0.9,
    process_weight=0.8,
    process_gamma=1.5,
    step_bonus_lambda=0.3,
    deep_exploration_bonus_lambda=0.12,
    deep_exploration_step_cap=5,
    deep_exploration_difficulty_threshold=9,
    deep_exploration_difficulty_cap=16,
    deep_exploration_difficulty_by_source=None,
    hallucination_penalty_ratio=0.5,
    **kwargs
):
    """
    简化版本：
    - 不使用长度 bonus
    - 只使用 GPT judge 返回的 matched_step_count / effective_reasoning_step_count 做 step bonus
    - 有效思考 step bonus 不做难度门控，避免简单题完全拿不到过程鼓励
    - 对较难样本额外奖励 deep_exploration_step_count
    - 幻觉惩罚仅针对“已经匹配成功的步骤”里存在带幻觉推导的情况
    - 幻觉惩罚按 hallucinated_matched_step_count / matched_step_count 的比例衰减主分项（答案分或过程分），不衰减 step bonus
    """
    total_score = 0.0

    gt_steps, gt_final = parse_ground_truth(ground_truth)
    pred_think, pred_final = extract_parts(solution_str)
    problem_text = (extra_info or {}).get("problem", "")
    problem_type = infer_problem_type(data_source, extra_info)

    if pred_think is None or pred_final is None:
        return -1.0 * format_score

    if "->" in pred_think:
        total_score += style_score
    else:
        total_score -= style_score

    violation_count = check_density(pred_think, max_len=max_segment_len)
    if violation_count > 0:
        total_score -= (min(5, violation_count) * density_penalty)

    is_answer_correct = grade_answer(pred_final, gt_final)

    if gt_steps and len(gt_steps) > 0:
        gpt_result = await evaluate_step_bonus_with_gpt(
            pred_think,
            pred_final,
            gt_steps,
            reward_router_address,
            reward_model_name=reward_model_name,
            problem_text=problem_text,
            problem_type=problem_type,
        )

        matched_count = gpt_result.get("matched_step_count", 0)
        total_steps = len(gt_steps)
        effective_reasoning_step_count = gpt_result.get("effective_reasoning_step_count", 0)
        deep_exploration_step_count = gpt_result.get("deep_exploration_step_count", 0)
        hallucinated_matched_step_count = gpt_result.get("hallucinated_matched_step_count", 0)
        has_hallucination_in_matched_steps = gpt_result.get("has_hallucination_in_matched_steps", False)

        completion_ratio = min(matched_count / total_steps, 1.0) if total_steps > 0 else 0.0
        reasoning_step_target = max(total_steps, 1)
        reasoning_step_ratio = min(effective_reasoning_step_count / reasoning_step_target, 1.0)
        if deep_exploration_step_cap > 0:
            deep_exploration_ratio = min(
                deep_exploration_step_count / deep_exploration_step_cap,
                1.0,
            )
        else:
            deep_exploration_ratio = 0.0

        difficulty_threshold, difficulty_cap, source_key = get_deep_exploration_difficulty_schedule(
            data_source=data_source,
            extra_info=extra_info,
            difficulty_by_source=deep_exploration_difficulty_by_source,
            default_threshold=deep_exploration_difficulty_threshold,
            default_cap=deep_exploration_difficulty_cap,
        )

        if difficulty_cap <= difficulty_threshold:
            difficulty_ratio = 1.0 if total_steps > difficulty_threshold else 0.0
        else:
            difficulty_ratio = clamp(
                (total_steps - difficulty_threshold)
                / (difficulty_cap - difficulty_threshold),
                0.0,
                1.0,
            )

        is_effectively_correct = is_answer_correct or matched_count == total_steps

        if is_effectively_correct:
            main_reward = score
        else:
            main_reward = (completion_ratio ** process_gamma) * process_weight

        hallucination_ratio = 0.0
        if matched_count > 0:
            hallucination_ratio = min(hallucinated_matched_step_count / matched_count, 1.0)

        if has_hallucination_in_matched_steps and hallucination_ratio > 0.0:
            main_reward *= (1.0 - hallucination_penalty_ratio * hallucination_ratio)

        total_score += main_reward

        total_score += step_bonus_lambda * completion_ratio * reasoning_step_ratio
        total_score += deep_exploration_bonus_lambda * completion_ratio * difficulty_ratio * deep_exploration_ratio

        print(
            f"Total: {total_score:.2f} | Ans: {is_effectively_correct} | "
            f"MatchedSteps: {matched_count}/{total_steps} | "
            f"ReasoningSteps: {effective_reasoning_step_count}/{reasoning_step_target} | "
            f"DeepExplore: {deep_exploration_step_count}/{deep_exploration_step_cap} "
            f"(difficulty={difficulty_ratio:.2f}, source={source_key}, "
            f"threshold={difficulty_threshold}, cap={difficulty_cap}) | "
            f"MatchedHallucination: {has_hallucination_in_matched_steps} "
            f"({hallucinated_matched_step_count}, ratio={hallucination_ratio:.2f})"
        )
    else:
        if is_answer_correct:
            total_score += score
        print(f"Total: {total_score:.2f} | Ans: {is_answer_correct} | No GT Steps")

    return total_score


async def compute_score_qwen_gpt_step_bonus_guided_json(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    reward_model_name=None,
    format_score=1.0,
    style_score=0.1,
    density_penalty=0.05,
    max_segment_len=15,
    score=0.9,
    process_weight=0.8,
    process_gamma=1.5,
    step_bonus_lambda=0.3,
    deep_exploration_bonus_lambda=0.12,
    deep_exploration_step_cap=5,
    deep_exploration_difficulty_threshold=9,
    deep_exploration_difficulty_cap=16,
    deep_exploration_difficulty_by_source=None,
    hallucination_penalty_ratio=0.5,
    **kwargs,
):
    """
    与 compute_score_qwen_gpt_step_bonus 内容一致，
    仅 LLM judge 阶段改为 guided_json 约束输出格式。
    适用于 vLLM OpenAI-compatible router 支持 guided_json 的场景。
    """
    total_score = 0.0

    gt_steps, gt_final = parse_ground_truth(ground_truth)
    pred_think, pred_final = extract_parts(solution_str)
    problem_text = (extra_info or {}).get("problem", "")
    problem_type = infer_problem_type(data_source, extra_info)

    if pred_think is None or pred_final is None:
        return -1.0 * format_score

    if "->" in pred_think:
        total_score += style_score
    else:
        total_score -= style_score

    violation_count = check_density(pred_think, max_len=max_segment_len)
    if violation_count > 0:
        total_score -= (min(5, violation_count) * density_penalty)

    is_answer_correct = grade_answer(pred_final, gt_final)

    if gt_steps and len(gt_steps) > 0:
        gpt_result = await evaluate_step_bonus_with_gpt_guided_json(
            pred_think,
            pred_final,
            gt_steps,
            reward_router_address,
            reward_model_name=reward_model_name,
            problem_text=problem_text,
            problem_type=problem_type,
        )

        matched_count = gpt_result.get("matched_step_count", 0)
        total_steps = len(gt_steps)
        effective_reasoning_step_count = gpt_result.get("effective_reasoning_step_count", 0)
        deep_exploration_step_count = gpt_result.get("deep_exploration_step_count", 0)
        hallucinated_matched_step_count = gpt_result.get("hallucinated_matched_step_count", 0)
        has_hallucination_in_matched_steps = gpt_result.get("has_hallucination_in_matched_steps", False)

        completion_ratio = min(matched_count / total_steps, 1.0) if total_steps > 0 else 0.0
        reasoning_step_target = max(total_steps, 1)
        reasoning_step_ratio = min(effective_reasoning_step_count / reasoning_step_target, 1.0)
        if deep_exploration_step_cap > 0:
            deep_exploration_ratio = min(
                deep_exploration_step_count / deep_exploration_step_cap,
                1.0,
            )
        else:
            deep_exploration_ratio = 0.0

        difficulty_threshold, difficulty_cap, source_key = get_deep_exploration_difficulty_schedule(
            data_source=data_source,
            extra_info=extra_info,
            difficulty_by_source=deep_exploration_difficulty_by_source,
            default_threshold=deep_exploration_difficulty_threshold,
            default_cap=deep_exploration_difficulty_cap,
        )

        if difficulty_cap <= difficulty_threshold:
            difficulty_ratio = 1.0 if total_steps > difficulty_threshold else 0.0
        else:
            difficulty_ratio = clamp(
                (total_steps - difficulty_threshold)
                / (difficulty_cap - difficulty_threshold),
                0.0,
                1.0,
            )

        is_effectively_correct = is_answer_correct or matched_count == total_steps

        if is_effectively_correct:
            main_reward = score
        else:
            main_reward = (completion_ratio ** process_gamma) * process_weight

        hallucination_ratio = 0.0
        if matched_count > 0:
            hallucination_ratio = min(hallucinated_matched_step_count / matched_count, 1.0)

        if has_hallucination_in_matched_steps and hallucination_ratio > 0.0:
            main_reward *= (1.0 - hallucination_penalty_ratio * hallucination_ratio)

        total_score += main_reward

        total_score += step_bonus_lambda * completion_ratio * reasoning_step_ratio
        total_score += deep_exploration_bonus_lambda * completion_ratio * difficulty_ratio * deep_exploration_ratio

        print(
            f"Total: {total_score:.2f} | Ans: {is_effectively_correct} | "
            f"MatchedSteps: {matched_count}/{total_steps} | "
            f"ReasoningSteps: {effective_reasoning_step_count}/{reasoning_step_target} | "
            f"DeepExplore: {deep_exploration_step_count}/{deep_exploration_step_cap} "
            f"(difficulty={difficulty_ratio:.2f}, source={source_key}, "
            f"threshold={difficulty_threshold}, cap={difficulty_cap}) | "
            f"MatchedHallucination: {has_hallucination_in_matched_steps} "
            f"({hallucinated_matched_step_count}, ratio={hallucination_ratio:.2f})"
        )
    else:
        if is_answer_correct:
            total_score += score
        print(f"Total: {total_score:.2f} | Ans: {is_answer_correct} | No GT Steps")

    return total_score
