# Environment Requirements

Use separate environments for downloads, GPU evaluation and RL training. The GPU frameworks have coupled CUDA, PyTorch and extension requirements; installing all latest packages into one environment is not a reproducible setup.

## Downloads

Python 3.11 and `huggingface_hub` are sufficient to download artifacts. Use `pyarrow` to inspect parquet metadata and `datasets` to load dataset configurations.

```bash
uv venv --python 3.11 .venv-download
source .venv-download/bin/activate
uv pip install huggingface_hub pyarrow datasets
```

Plan for approximately 30.4 GiB of SFT data, 1.6 GiB of RL data, plus model weights, benchmark assets and package caches. These sizes exclude any extra copies created by dataset processing.

## Evaluation

- Linux, NVIDIA GPU(s), and a compatible NVIDIA driver/CUDA runtime.
- Python 3.11; CUDA-enabled PyTorch, torchvision and vLLM compatible with Qwen3-VL.
- The bundled `VLMEvalkit/requirements.txt` pins `transformers==4.57.1`.
- A compatible FlashAttention build when required by the selected inference backend; video decoding dependencies from VLMEvalKit.
- Network access or pre-downloaded model and benchmark files.
- An answer-judging API credential supplied through the environment. The reproduction command requests `gpt-4o-2024-05-13`; using a different judge may change scores.

In an environment where the GPU stack is already installed:

```bash
uv pip install -e ./VLMEvalkit
```

[`configs/eval.json`](configs/eval.json) uses the same decoding settings as the bundled model registrations. It does not introduce an exact version lock for the entire GPU stack: the original private image is not a public reproducible environment. Record your resolved packages with `uv pip freeze` alongside each result.

## RL Training

- The bundled verl package, Ray, Hydra/OmegaConf and training dependencies.
- Megatron-Core, TransformerEngine, a Qwen3-VL-capable mbridge installation, and vLLM with mutually compatible CUDA/PyTorch builds.
- **Two GPU pools with 8 GPUs each** for the provided recipe: one for the actor/reference/rollout and one for the Qwen3-235B reward model. GPU memory, host RAM for offload and checkpoint disk capacity must be sized for those models.
- Connectivity between Ray nodes and shared visibility of the checkout, data and checkpoint paths.

Install the bundled package into an existing compatible training environment:

```bash
uv pip install -e ./Verl
```

See the [bundled verl documentation](../Verl/docs/start/install.rst) for backend installation. The bundled Dockerfiles describe upstream build options, not a guarantee of an identical paper environment. The optimizer CPU-offload options and model bridge must be verified against the exact installed versions before a long run.

## Public Environment Variables

| Variable | Purpose |
| --- | --- |
| `DRT_REPO_ROOT` | Absolute path to this checkout |
| `DRT_DATA_DIR` | Directory containing original RL `train.parquet` and `test.parquet` |
| `DRT_SFT_MODEL` | Public SFT model ID or a local downloaded directory |
| `DRT_REWARD_MODEL` | Public reward model ID or a local downloaded directory |
| `DRT_OUTPUT_DIR` | Writable checkpoint directory visible to the training workers |
| `PROMPT_TYPE` | Set to `Short-COT-Image` for the evaluation recipe |
| `LMUData` | Writable benchmark-data cache |
| `OPENAI_API_BASE` / `OPENAI_API_KEY` | Complete chat-completions endpoint URL and credential for the bundled evaluation adapter |

Set credentials only in the runtime environment. Public configs contain no account IDs, private image registries, job identifiers, queue names or original storage paths.
