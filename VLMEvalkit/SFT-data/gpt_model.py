import openai
from openai import OpenAI
import os
import time

class GPT4o:
    completion_tokens = 0
    prompt_tokens = 0
    def __init__(self, deployment_name, api_base=None, api_key=None, use_azure=None):
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        use_azure = use_azure if use_azure is not None else os.environ.get("OPENAI_API_TYPE") == "azure"
        self.client = OpenAI(
            api_key=self.api_key,  
            base_url=self.api_base
        )
        self.deployment_name = deployment_name
        if use_azure:
            self.client = openai.AzureOpenAI(
                azure_endpoint=self.api_base,
                api_version=os.environ.get("OPENAI_API_VERSION", "2024-03-01-preview"),
                api_key=self.api_key,
            )
    def send_single_request(self, messages, temperature=0.0):
        result = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
        )
        print(f"token=({result.usage.completion_tokens}, {result.usage.prompt_tokens})")
        self.completion_tokens += result.usage.completion_tokens
        self.prompt_tokens += result.usage.prompt_tokens
        return result

    def send_stable_request(self, messages, retry=3, temperature=0.0):
        print("GPT API CALL")
        for i in range(retry):
            try:
                result = self.send_single_request(messages, temperature = min(temperature + i * 0.1, 1))
                print(f"try {i}: total_token=({self.completion_tokens}, {self.prompt_tokens})")
                if not result.choices[0].message.content:
                    print(f"[Error] empty response: {result}")
                    continue
                return result.choices[0].message.content
            except Exception as e:
                print(f"[Error] API Return Error: {e}")
                time.sleep(1)
                continue
        raise(Exception("Retry time limit exceeded"))
