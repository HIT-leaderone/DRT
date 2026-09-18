from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.vlm.qwen3_vl.prompt import Qwen3VLPromptMixin


PROMPT_TYPE_ENV_MAP = {
    "custom-prompt": "Custom-Prompt",
    "directly-answer": "Directly-Answer",
    "short-cot-image": "Short-COT-Image",
    "short-cot-video": "Short-COT-Video",
    "cod": "CoD",
}


class OfficialQwen3VLPromptBuilder(Qwen3VLPromptMixin):
    def __init__(self) -> None:
        super().__init__(use_custom_prompt=True)


OFFICIAL_PROMPT_BUILDER = OfficialQwen3VLPromptBuilder()


def set_official_prompt_env(prompt_type: str | None) -> None:
    if prompt_type is None:
        os.environ.pop("PROMPT_TYPE", None)
        return

    mapped = PROMPT_TYPE_ENV_MAP.get(str(prompt_type).strip().lower())
    if mapped is None:
        os.environ.pop("PROMPT_TYPE", None)
    else:
        os.environ["PROMPT_TYPE"] = mapped


def build_official_eval_message(dataset, dataset_name: str, row, video_llm: bool = False):
    if getattr(dataset, "MODALITY", None) == "VIDEO":
        return dataset.build_prompt(row, video_llm=video_llm)

    if OFFICIAL_PROMPT_BUILDER.use_custom_prompt(dataset_name):
        OFFICIAL_PROMPT_BUILDER.set_dump_image(dataset.dump_image)
        return OFFICIAL_PROMPT_BUILDER.build_prompt(row, dataset=dataset_name)

    return dataset.build_prompt(row)


def ensure_file_url(path: str) -> str:
    prefixes = ("http://", "https://", "file://", "data:image", "data:video")
    if path.startswith(prefixes):
        return path
    return "file://" + path


def to_processor_conversation(message, system_prompt: str | None = None):
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    content = []
    for item in message:
        item_type = item["type"]
        value = item["value"]

        if item_type == "text":
            content.append({"type": "text", "text": value})
            continue

        if item_type == "image":
            values = value if isinstance(value, list) else [value]
            for image_path in values:
                content.append({"type": "image", "image": ensure_file_url(image_path)})
            continue

        if item_type == "video":
            if isinstance(value, list):
                for frame_path in value:
                    content.append({"type": "image", "image": ensure_file_url(frame_path)})
            else:
                content.append({"type": "video", "video": ensure_file_url(value)})
            continue

        if item_type == "audio":
            content.append({"type": "audio", "audio": value})
            continue

        raise ValueError(f"Unsupported message type: {item_type}")

    conversation.append({"role": "user", "content": content})
    return conversation
