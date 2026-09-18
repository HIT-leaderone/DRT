from __future__ import annotations

import os

import torch

from .base import BaseModel


class Qwen3TextChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    allowed_types = ['text']

    def __init__(
        self,
        model_path: str = 'Qwen/Qwen3-8B',
        max_new_tokens: int = 16384,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.use_vllm = kwargs.get('use_vllm', False)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.use_vllm:
            from vllm import LLM

            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get('gpu_utils', 0.9),
                trust_remote_code=True,
                seed=0,
            )
        else:
            model_kwargs = dict(
                pretrained_model_name_or_path=model_path,
                torch_dtype='auto',
                device_map='auto',
                trust_remote_code=True,
            )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    attn_implementation='flash_attention_2',
                    **model_kwargs,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.model.eval()

        torch.cuda.empty_cache()

    def _prompt_type(self) -> str:
        return os.environ.get('PROMPT_TYPE', '')

    def _enable_thinking(self) -> bool:
        prompt_type = self._prompt_type()
        return prompt_type in {'Thinking', 'Qwen3-Thinking'}

    def _sampling_config(self) -> dict:
        if self._enable_thinking():
            return dict(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                presence_penalty=0.0,
                max_new_tokens=max(self.max_new_tokens, 32768),
            )
        return dict(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            presence_penalty=self.presence_penalty,
            max_new_tokens=self.max_new_tokens,
        )

    def _build_messages(self, message: list[dict[str, str]]) -> list[dict[str, str]]:
        text = '\n'.join([item['value'] for item in message if item['type'] == 'text']).strip()
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': text})
        return messages

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self._enable_thinking(),
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)

    def generate_inner_transformers(self, message, dataset=None):
        messages = self._build_messages(message)
        prompt = self._apply_chat_template(messages)
        if self.verbose:
            print(f'\033[31m{prompt}\033[0m')

        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        sampling = self._sampling_config()

        generate_kwargs = dict(
            max_new_tokens=sampling['max_new_tokens'],
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if sampling['temperature'] and sampling['temperature'] > 0:
            generate_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=sampling['temperature'],
                    top_p=sampling['top_p'],
                    top_k=sampling['top_k'],
                ))
        else:
            generate_kwargs['do_sample'] = False

        outputs = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        messages = self._build_messages(message)
        prompt = self._apply_chat_template(messages)
        if self.verbose:
            print(f'\033[31m{prompt}\033[0m')
        sampling = self._sampling_config()

        sampling_params = SamplingParams(
            temperature=sampling['temperature'],
            max_tokens=sampling['max_new_tokens'],
            top_p=sampling['top_p'],
            top_k=sampling['top_k'],
            repetition_penalty=self.repetition_penalty,
            presence_penalty=sampling['presence_penalty'],
        )
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text.strip()
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
