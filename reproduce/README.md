# Reproducing DRT

This directory provides public, platform-independent settings for the released models. Begin with the [environment requirements](environment.md) and [data downloads](../training_data/README.md).

## Evaluation

The bundled evaluation config uses public model IDs and explicitly selects `Qwen3VLChat`. It does not depend on the model registry's historical local paths.

In an evaluation environment with the bundled VLMEvalKit and compatible vLLM installed, run from the repository root:

```bash
export DRT_REPO_ROOT="$PWD"
export PROMPT_TYPE=Short-COT-Image
export LMUData="$DRT_REPO_ROOT/benchmark_data"
export DRT_ROLLOUT_N=1
export DRT_AGGREGATION=none

# Set OPENAI_API_KEY in your environment. OPENAI_API_BASE, when overridden,
# must be the complete chat-completions URL, not just a /v1 SDK base URL.
# Never put credentials in the configuration file.
cd "$DRT_REPO_ROOT/VLMEvalkit"
python run.py \
  --config "$DRT_REPO_ROOT/reproduce/configs/eval.json" \
  --work-dir "$DRT_REPO_ROOT/outputs/eval" \
  --judge gpt-4o-2024-05-13 \
  --api-nproc 4 \
  --use-vllm
```

The config evaluates both checkpoints on `MathVista_MINI`, `MathVerse_MINI`, `LogicVista`, `GSM8K`, and `Video_Holmes`. Video-Holmes uses the adapter's 32-frame setting. To evaluate one model, remove the other entry from the JSON's `model` mapping. To smoke-test the pipeline, use `--limit 2` and a separate output directory; a limited run is not a benchmark result.

| Setting | Value |
| --- | --- |
| Prompt selector | `Short-COT-Image` |
| Temperature | 0.7 |
| Top-p / top-k | 0.8 / 20 |
| Maximum generated tokens | 16,384 |
| Repetition / presence penalty | 1.0 / 1.5 |
| Rollouts / aggregation | 1 / none |
| Answer judge | `gpt-4o-2024-05-13` |

The evaluation answer judge is separate from the Qwen reward model used during RL training. Sampling, engine versions and judge responses can cause variation between reruns. Benchmark assets may be downloaded on first use; use their original access and licensing procedures.

## RL Training

[`configs/drt_rl.yaml`](configs/drt_rl.yaml) expresses the VisionR1 + DAPO recipe using the **bundled verl schema**, including `reward.reward_model` and `reward.custom_reward_function`. It is a portable configuration, not an exported scheduler job.

Download `train.parquet` and `test.parquet` using the data guide. From the checkout root:

```bash
export DRT_REPO_ROOT="$PWD"
export DRT_DATA_DIR="$DRT_REPO_ROOT/training_data/VisionR1_DAPO"
export DRT_OUTPUT_DIR="$DRT_REPO_ROOT/checkpoints/drt-rl"
export DRT_SFT_MODEL=leaderonehit/DRT-SFT-8B
export DRT_REWARD_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

cd "$DRT_REPO_ROOT/Verl"
# Resolve the config before allocating a training run.
python -m verl.trainer.main_ppo \
  --config-path "$DRT_REPO_ROOT/reproduce/configs" \
  --config-name drt_rl --cfg job --resolve

# Run after your Ray cluster exposes both GPU resource pools.
python -m verl.trainer.main_ppo \
  --config-path "$DRT_REPO_ROOT/reproduce/configs" \
  --config-name drt_rl
```

Paths can point to pre-downloaded local model directories. The recipe requests **8 GPUs for the policy/rollout pool plus 8 GPUs for a separate reward-model pool**. One policy node is not the total hardware requirement. All Ray workers must see the code, model/data paths and required environment variables. Checkpoint storage is local/shared filesystem by default; automatic scheduler submission is disabled.

Key settings: GRPO, prompt batch 512, 16 responses per prompt, PPO mini-batch 128, micro-batch 1 per GPU, actor learning rate `1e-6`, prompt/response limits 2,048 each, KL loss coefficient 0.01, actor TP/PP/CP 2/2/1, rollout TP 4, reward-model TP 8. Per-source process-reward settings are inherited from the [VisionR1 + DAPO config](../Verl/verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo.yaml).

The public YAML retains the legacy recipe's optimizer-offload and transformer override settings. Their support depends on the installed Megatron, TransformerEngine and model bridge versions. Config resolution alone does not validate those GPU kernels.

## SFT

[`configs/drt_sft.yaml`](configs/drt_sft.yaml) records the checkpoint's non-private SFT hyperparameters. The released model selects iteration **800** from a run configured for **2,000** iterations. The original ms-swift/Megatron SFT entrypoint is not included; the YAML is a parameter record, not a directly runnable verl config. The SFT weights and all 20 data shards are publicly downloadable.

## Source Versions

The vendored training snapshot originated from `official-latest-qwen3vl-port` at `600716c1c7f88fce9097ff2f78f5e2835d20e690`; the evaluation snapshot originated from `dev_xw` at `1d19f737c28e4ec0d3fc4f81cebc2ad476757595`. The recorded training anchors were `official-latest-qwen3vl-port:436584c88b95` and `dev_qwen3vl:9930c77819f0`.

These are provenance identifiers from the source repositories, **not branches or checkouts available in this snapshot repository**. Public-facing sanitation and configuration changes are recorded in this repository's own commits.

## Validation Scope

The publication update checks configuration composition, local references, syntax and release metadata. It does not claim that a clean-room SFT/RL training run or a fresh full benchmark evaluation has been completed on the portable recipe.

Run the CPU-only config/documentation checks with `python -m unittest discover -s reproduce -p 'test_*.py'` in an environment containing `hydra-core` and `omegaconf`.
