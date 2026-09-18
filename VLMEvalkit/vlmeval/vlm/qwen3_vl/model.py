from __future__ import annotations

import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from .aggregation import (
    build_bon_selector_text,
    env_flag,
    extract_answer,
    json_dumps,
    parse_selector_index,
    select_self_consistent,
)
from ...smp import get_gpu_memory, listinstr


VLLM_MAX_IMAGE_INPUT_NUM = 128


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    path_parts = model_path.split('/')
    non_moe_patterns = ['2B','4B','8B','32B']
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def sanitize_colon_env_var(name: str) -> None:
    raw = os.environ.get(name, '')
    if not raw:
        return

    cleaned = []
    for part in raw.split(':'):
        if not part:
            continue
        try:
            if os.path.isdir(part) and os.access(part, os.X_OK):
                cleaned.append(part)
        except OSError:
            continue
    os.environ[name] = ':'.join(cleaned)


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.drt_rollout_n = max(1, int(os.environ.get("DRT_ROLLOUT_N", "1")))
        self.drt_aggregation = os.environ.get("DRT_AGGREGATION", "none").strip().lower()
        if self.drt_aggregation in {"self_consistency", "self-consistency", "majority", "vote"}:
            self.drt_aggregation = "sc"
        if self.drt_aggregation in {"best-of-n", "best_of_n", "bon", "rerank"}:
            self.drt_aggregation = "bon"
        if self.drt_rollout_n == 1:
            self.drt_aggregation = "none"
        if self.drt_aggregation not in {"none", "sc", "bon"}:
            raise ValueError(
                "DRT_AGGREGATION must be one of none, sc, or bon; "
                f"got {self.drt_aggregation!r}"
            )
        self.drt_rollout_temperature = float(
            os.environ.get("DRT_ROLLOUT_TEMPERATURE", str(self.temperature))
        )
        self.drt_bon_selector = os.environ.get("DRT_BON_SELECTOR", "self").strip().lower()
        if self.drt_bon_selector not in {"self"}:
            raise ValueError(
                "Only DRT_BON_SELECTOR=self is currently supported for in-model BoN reranking."
            )
        self.drt_bon_max_tokens = int(os.environ.get("DRT_BON_MAX_TOKENS", "128"))
        self.drt_bon_candidate_max_chars = int(os.environ.get("DRT_BON_CANDIDATE_MAX_CHARS", "4096"))
        self.drt_store_rollouts = env_flag(os.environ.get("DRT_STORE_ROLLOUTS"), default=False)
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', None)
        self.max_frames = kwargs.pop('max_frames', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            # Some environments inject inaccessible directories such as
            # /var/lib/fastrak/lib64 into LD_LIBRARY_PATH, which breaks
            # flashinfer/tvm_ffi during vLLM startup.
            sanitize_colon_env_var('PATH')
            sanitize_colon_env_var('LD_LIBRARY_PATH')
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=8,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

        torch.cuda.empty_cache()

    def _is_thinking_model(self) -> bool:
        return 'thinking' in self.model_path.lower()

    def _effective_presence_penalty(self, dataset: str | None = None) -> float:
        if not self._is_thinking_model() or dataset is None:
            return self.presence_penalty
        from ...dataset import DATASET_MODALITY
        if DATASET_MODALITY(dataset, default='IMAGE') == 'TEXT':
            # Follow the official Qwen3-VL model card:
            # thinking models use presence_penalty=1.5 for text tasks
            # and presence_penalty=0.0 for VL tasks.
            return 1.5
        return self.presence_penalty

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        all_image_indices = [i for i, x in enumerate(inputs) if x['type'] == 'image']
        total_images = len(all_image_indices)
        kept_image_indices = set(all_image_indices)
        if total_images > VLLM_MAX_IMAGE_INPUT_NUM:
            sampled_positions = [int(i * (total_images - 1) / (VLLM_MAX_IMAGE_INPUT_NUM - 1)) for i in range(VLLM_MAX_IMAGE_INPUT_NUM)]
            kept_image_indices = set([all_image_indices[p] for p in sampled_positions])
            print(f"Downsampling images from {total_images} to {VLLM_MAX_IMAGE_INPUT_NUM}")

        for i, s in enumerate(inputs): # 注意这里加了 enumerate 以获取索引 i
            if s['type'] == 'image':
                if i not in kept_image_indices: continue
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    total_frames = len(value)
                    if total_frames > VLLM_MAX_IMAGE_INPUT_NUM:
                        indices = [int(i * (total_frames - 1) / (VLLM_MAX_IMAGE_INPUT_NUM - 1)) for i in range(VLLM_MAX_IMAGE_INPUT_NUM)]
                        value = [value[i] for i in indices]
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                        item['max_frames'] = self.max_frames
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        # print("content:", content)
        return content

    def _post_process_response(self, response: str) -> str:
        if not self.post_process:
            return response
        resp = response.split('\\boxed{')[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == '{':
                counter += 1
            elif resp[i] == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        return resp[:end] if end is not None else response

    @staticmethod
    def _message_text(message: list[dict[str, str]]) -> str:
        return "\n".join(str(item["value"]) for item in message if item.get("type") == "text")

    def _build_bon_selector_message(self, message: list[dict[str, str]], candidates: list[str]) -> list[dict[str, str]]:
        selector_text = build_bon_selector_text(
            self._message_text(message),
            candidates,
            candidate_max_chars=self.drt_bon_candidate_max_chars,
        )
        media = [dict(item) for item in message if item.get("type") != "text"]
        media.append({"type": "text", "value": selector_text})
        return media

    def _aggregate_drt_rollouts(
        self,
        message: list[dict[str, str]],
        dataset: str | None,
        candidates: list[str],
        output_lengths: list[int],
        selector_generate,
    ):
        if self.drt_aggregation == "none":
            return candidates[0]

        sc = select_self_consistent(candidates)
        selected_index = sc["selected_index"]
        selector_output = ""
        selector_output_tokens = 0
        fallback_to_sc = False

        if self.drt_aggregation == "bon":
            selector_message = self._build_bon_selector_message(message, candidates)
            selector_outputs, selector_lengths = selector_generate(selector_message, dataset)
            selector_output = selector_outputs[0] if selector_outputs else ""
            selector_output_tokens = selector_lengths[0] if selector_lengths else 0
            parsed_index = parse_selector_index(selector_output, len(candidates))
            if parsed_index is None:
                fallback_to_sc = True
            else:
                selected_index = parsed_index

        selected_text = candidates[selected_index]
        candidate_answers = [extract_answer(candidate) for candidate in candidates]
        total_output_tokens = int(sum(output_lengths) + selector_output_tokens)
        record = {
            "prediction": selected_text,
            "drt_aggregation": self.drt_aggregation,
            "drt_rollout_n": len(candidates),
            "drt_selected_index": selected_index,
            "drt_selected_rank": selected_index + 1,
            "drt_selected_answer": extract_answer(selected_text),
            "drt_candidate_answers": json_dumps(candidate_answers),
            "drt_normalized_answers": json_dumps(sc["normalized_answers"]),
            "drt_vote_counts": json_dumps(sc["vote_counts"]),
            "drt_candidate_output_tokens": json_dumps(output_lengths),
            "drt_selector": self.drt_bon_selector if self.drt_aggregation == "bon" else "",
            "drt_selector_prediction": selector_output,
            "drt_selector_output_tokens": selector_output_tokens,
            "drt_selector_fallback_to_sc": fallback_to_sc,
            "output_length_tokens": total_output_tokens,
        }
        if self.drt_store_rollouts:
            record["drt_rollouts"] = json_dumps(candidates)
        return record

    def _generate_inner_transformers_outputs(
        self,
        message,
        dataset=None,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[list[str], list[int]]:
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            if n != 1:
                raise ValueError('DRT_ROLLOUT_N > 1 is not supported for Qwen3-Omni transformers path.')
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            completion_ids = text_ids.sequences[:, inputs["input_ids"].shape[1]:]
            outputs = self.processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            generate_kwargs = dict(self.generate_kwargs)
            if temperature is not None:
                generate_kwargs['temperature'] = temperature
            if max_tokens is not None:
                generate_kwargs['max_new_tokens'] = max_tokens
            if n > 1:
                generate_kwargs['num_return_sequences'] = n
                generate_kwargs['do_sample'] = True

            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            input_len = inputs.input_ids.shape[1]
            completion_ids = generated_ids[:, input_len:]
            outputs = self.processor.tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        outputs = [self._post_process_response(output) for output in outputs]
        lengths = [len(ids) for ids in completion_ids]
        if self.verbose:
            for output in outputs:
                print(f'\033[32m{output}\033[0m')
        return outputs, lengths

    def generate_inner_transformers(self, message, dataset=None):
        candidates, output_lengths = self._generate_inner_transformers_outputs(
            message,
            dataset=dataset,
            n=self.drt_rollout_n,
            temperature=self.drt_rollout_temperature,
        )
        return self._aggregate_drt_rollouts(
            message,
            dataset,
            candidates,
            output_lengths,
            selector_generate=lambda selector_message, selector_dataset: self._generate_inner_transformers_outputs(
                selector_message,
                dataset=selector_dataset,
                n=1,
                temperature=0.0,
                max_tokens=self.drt_bon_max_tokens,
            ),
        )

    def _generate_inner_vllm_outputs(
        self,
        message,
        dataset=None,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[list[str], list[int]]:
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_new_tokens if max_tokens is None else max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self._effective_presence_penalty(dataset),
            stop_token_ids=None,
            n=n,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)
        if not outputs:
            return [""], [0]

        generated_outputs = outputs[0].outputs
        generated_texts = [self._post_process_response(output.text) for output in generated_outputs]
        output_lengths = [len(output.token_ids) for output in generated_outputs]

        if self.verbose:
            for generated_text in generated_texts:
                print(f'\033[32m{generated_text}\033[0m')
        return generated_texts, output_lengths

    def generate_inner_vllm(self, message, dataset=None):
        candidates, output_lengths = self._generate_inner_vllm_outputs(
            message,
            dataset=dataset,
            n=self.drt_rollout_n,
            temperature=self.drt_rollout_temperature,
        )
        return self._aggregate_drt_rollouts(
            message,
            dataset,
            candidates,
            output_lengths,
            selector_generate=lambda selector_message, selector_dataset: self._generate_inner_vllm_outputs(
                selector_message,
                dataset=selector_dataset,
                n=1,
                temperature=0.0,
                max_tokens=self.drt_bon_max_tokens,
            ),
        )

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
