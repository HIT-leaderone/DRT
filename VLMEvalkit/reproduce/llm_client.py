import os
from typing import Any

import requests
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.openai_client = None
        if not self._is_direct_chat_endpoint(base_url):
            self.openai_client = OpenAI(base_url=base_url, api_key=self.api_key)
        self.anthropic_client = Anthropic() if Anthropic else None

    @staticmethod
    def _is_direct_chat_endpoint(base_url: str | None) -> bool:
        if not base_url:
            return False
        normalized = base_url.rstrip("/")
        return normalized.endswith("/chat/completions") or normalized.endswith("/completions")

    @staticmethod
    def _flatten_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(item.text or "")
            return "".join(parts)
        return str(content)

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        if model.startswith("claude"):
            if self.anthropic_client is None:
                raise ImportError("anthropic is required to query Claude models")
            message = self.anthropic_client.messages.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = message.content[0].text
            token_count = message.usage.output_tokens
        elif self._is_direct_chat_endpoint(self.base_url):
            completion = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "messages": [{"role": "user", "content": payload}],
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=600,
            )
            completion.raise_for_status()
            completion = completion.json()
            response = self._flatten_content(completion["choices"][0]["message"]["content"])
            token_count = completion.get("usage", {}).get("completion_tokens", 0)
        else:
            completion = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = self._flatten_content(completion.choices[0].message.content)
            token_count = getattr(completion.usage, "completion_tokens", 0)
        return response, token_count


if __name__ == "__main__":
    llm = LLMClient()
    response, count = llm.request("hello", "claude-3-7-sonnet-latest")
    print(response, count)
