from __future__ import annotations
import os


class Qwen3VLPromptMixin:
    """
    Mixin class for Qwen3VLChat to build prompts consistent with Qwen3-VL README.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        # Follow Qwen3-VL convention: apply concise, task-specified prompts for MCQ/YN/VQA
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N':
            return True
        if dataset_type == 'VQA':
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        dataset_type = DATASET_TYPE(dataset, default=None)
        prompt_type = os.environ.get("PROMPT_TYPE", "")
        if prompt_type.startswith("CoD"):
            return self._build_cod_prompt(line, dataset, dataset_type)
        if prompt_type == "Short-COT-Image":
            return self._build_dense_cot_image_prompt(line, dataset, dataset_type)
        if prompt_type == "Short-COT-Video":
            return self._build_dense_cot_video_prompt(line, dataset, dataset_type)
        
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _collect_options(self, line) -> str:
        import ast
        import string
        import pandas as pd

        options = []
        if 'options' in line and line['options'] not in [None, '']:
            raw = line['options']
            if isinstance(raw, dict):
                options.extend([f'{k}. {v}' for k, v in raw.items()])
        if 'choices' in line and line['choices'] not in [None, '']:
            raw = line['choices']
            if isinstance(raw, str):
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    raw = raw
            if isinstance(raw, dict):
                options.extend([f'{k}. {v}' for k, v in raw.items()])
            elif isinstance(raw, (list, tuple)):
                options.extend([f'{chr(65 + i)}. {v}' for i, v in enumerate(raw)])
        for cand in string.ascii_uppercase:
            if cand in line and not pd.isna(line[cand]):
                options.append(f'{cand}. {line[cand]}')
        # Preserve order while removing duplicates.
        seen = set()
        deduped = []
        for opt in options:
            if opt not in seen:
                deduped.append(opt)
                seen.add(opt)
        return '\n'.join(deduped)

    def _build_cod_prompt(self, line, dataset: str, dataset_type: str | None) -> list[dict[str, str]]:
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'} or dataset_type == 'MCQ':
            return self._build_cod_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_cod_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_cod_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported CoD dataset: {dataset}')

    def _build_cod_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        question = line['question']
        options_prompt = self._collect_options(line)
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if options_prompt:
            prompt += f'Options:\n{options_prompt}\n'
        prompt += (
            'Think step by step, but keep only a minimum draft for each step with at most 5 words.\n'
            'Respond in this format exactly:\n'
            '<think>minimal draft steps</think>\n'
            '<answer>single option letter</answer>'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt.rstrip()))
        return msgs

    def _build_cod_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        prompt = (
            f'Question: {question}\n'
            'Think step by step, but keep only a minimum draft for each step with at most 5 words.\n'
            'Respond in this format exactly:\n'
            '<think>minimal draft steps</think>\n'
            '<answer>yes or no</answer>'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_cod_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        question = line['question']
        options_prompt = self._collect_options(line)
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if options_prompt:
            prompt += f'Options:\n{options_prompt}\n'
        prompt += (
            'Think step by step, but keep only a minimum draft for each step with at most 5 words.\n'
            'Respond in this format exactly:\n'
            '<think>minimal draft steps</think>\n'
            '<answer>final concise answer</answer>'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt.rstrip()))
        return msgs

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Keep all images at the beginning; then a single user text message.
        Matches Qwen3-VL multi-image style shown in README examples.
        """
        import string
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}

        options_prompt = ''
        if len(options):
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.'
        prompt = prompt.rstrip()

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Multi-choice prompt: include options and require a single option letter.
        Keep images before the text per Qwen3-VL convention.
        """
        import string
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}

        options_prompt = ''
        if len(options):
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.'
        prompt = prompt.rstrip()

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Yes/No prompt: require explicit yes or no answer only.
        Keep images before the text.
        """
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        prompt = f'{question} Please answer yes or no.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """VQA prompt: concise question with preference for short answers.
        Keep images before the text.
        """
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if os.environ.get("PROMPT_TYPE", "") == "Directly-Answer":
            question = question + '\nPlease answer concisely with short words or phrases when possible.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    def _build_dense_cot_video_prompt(self, line, dataset: str, dataset_type) -> list[dict[str, str]]:
        """VQA prompt: concise question with preference for short answers.
        Keep images before the text.
        """
        tgt_path = self.dump_image(line, dataset)
        question = line['question']

        QUESTION_TMPL = (
            "{}\n\n"
            "Analyze this question with **maximum information density** and strict logical precision.\n"
            "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols).\n"
            "2. **Constraint**: STRICTLY FORBIDDEN to use conversational fillers (e.g., 'Let me think', 'Hmm', 'I see', 'Wait').\n"
            "3. **Content**: Focus only on: [Key Visual Evidence] -> [Logical Inference] -> [Conclusion].\n"
            "Provide your dense cognitive trace between the <think> </think> tags, and then give your final answer."
        )
        prompt = QUESTION_TMPL.format(question)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_dense_cot_image_prompt(self, line, dataset: str, dataset_type) -> list[dict[str, str]]:
        """VQA prompt: concise question with preference for short answers.
        Keep images before the text.
        """
        tgt_path = self.dump_image(line, dataset)
        question = line['question']

        QUESTION_TMPL = (
            "{}\n"
            "{}\n"
            "Analyze this question to provide a **Dense Cognitive Trace**.\n"
            "--- Requirements ---\n"
            "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols). NO conversational fillers.\n"
            "2. **Structure**: Your response MUST strictly follow this XML structure:\n"
            "   <visual>...concise visual evidence...</visual><think>...[Priors] -> ...compressed reasoning...</think><answer>...final answer...</answer>\n"
            "3. **Content**: Deconstruct into visual observations, logical reasoning, and the final answer."
        )
        if 'options' in line:
            options = line['options']
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
        else:
            options_prompt = ''
        prompt = QUESTION_TMPL.format(question, options_prompt)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs
        
