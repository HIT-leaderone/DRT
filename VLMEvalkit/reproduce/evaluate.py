import argparse
import csv
import os

from llm_client import LLMClient
from tasks.coin_flip import CoinFlip
from tasks.date import DateUnderstanding
from tasks.gsm8k import GSM8K
from tasks.sports import SportsUnderstanding
from utils import average, nth_percentile

MODEL_MAPPING = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "sonnet": "claude-3-5-sonnet-20240620",
    "haiku": "claude-3-5-haiku-20241022",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "date", "sports", "coin_flip"])
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument(
        "--prompt",
        choices=["baseline", "cod", "cot"],
        default="cod",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Number of fewshot to be included, by default, include all fewshot examples",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base url for llm model endpoint",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for model access, will use api keys in environment variables for openai and claude models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples. Useful for smoke tests before full reproduction.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the model backend.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens per example.",
    )

    args = parser.parse_args()
    llm_client = LLMClient(args.url, args.api_key)
    match args.task:
        case "gsm8k":
            task = GSM8K(llm_client)
        case "date":
            task = DateUnderstanding(llm_client)
        case "sports":
            task = SportsUnderstanding(llm_client)
        case "coin_flip":
            task = CoinFlip(llm_client)
        case _:
            raise ValueError("Invalid task")

    model = MODEL_MAPPING.get(args.model, args.model)
    accuracy = task.evaluate(
        model,
        args.prompt,
        args.shot,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    results = [
        [
            "Accuracy",
            "Avg Token #",
            "Average Latency (s)",
            "P90 Latency (s)",
            "P95 Latency (s)",
            "P99 Latency (s)",
        ],
        [
            accuracy,
            average(task.token_count_tracker),
            average(task.latency_tracker),
            nth_percentile(task.latency_tracker, 0.9),
            nth_percentile(task.latency_tracker, 0.95),
            nth_percentile(task.latency_tracker, 0.99),
        ],
    ]
    for i in range(len(results[0])):
        print(f"{results[0][i]}: {results[1][i]}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    model_name = args.model.split(":")[1] if ":" in args.model else args.model
    model_name = model_name.replace("/", "_")
    fname = (
        f"{args.task}-{model_name}-{args.prompt}-{args.shot}"
        if args.shot
        else f"{args.task}-{model_name}-{args.prompt}"
    )
    if args.limit is not None:
        fname += f"-limit{args.limit}"
    with open(f"./results/{fname}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
