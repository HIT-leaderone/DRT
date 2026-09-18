# Practical Verl Extensions: Multimodal RL, LMM Reward Judge, and Auto-Eval

**Scope:** reusable engineering experience from our Verl-based RL work.  
**Focus:** what we validated, where the code lives, what interfaces matter, and how others can reuse or extend it.

Repository: <https://github.com/HIT-leaderone/DRT>

---

## 0. Executive Summary

We validated three reusable capabilities on top of Verl:

1. **Multimodal RL Training**  
   Run VLM RL with image/video inputs, Megatron actor training, vLLM rollout, and GRPO updates.

2. **Reward Loop with LMM Judge**  
   Call an external LMM/reward model during reward computation, so reward can go beyond simple rule-based final-answer matching.

3. **Auto-save + Auto-eval**  
   Save checkpoints, upload them to remote storage, and automatically launch evaluation jobs.

The overall experiment loop is:

```text
multimodal rollout -> flexible LMM-based reward -> checkpoint -> auto evaluation
```

---

## 1. Multimodal RL Training

### What we validated

We validated the end-to-end VLM RL path in Verl:

```text
multimodal parquet
  -> dataset / processor
  -> rollout engine generates responses
  -> reward function scores responses
  -> GRPO/PPO advantage computation
  -> actor update
  -> checkpoint save
```

Key validated pieces:

- Reading multimodal samples with text + image/video fields.
- Running rollout with vLLM.
- Training the actor with Megatron.
- Passing sample metadata into reward computation.
- Keeping dynamic batching / sequence balancing compatible with multimodal fields.

### Key code locations

| Purpose | Location |
| --- | --- |
| Training entrypoint | `verl/trainer/main_ppo.py` |
| PPO/GRPO trainer loop | `verl/trainer/ppo/ray_trainer.py` |
| vLLM rollout | `verl/workers/rollout/vllm_rollout/` |
| SGLang rollout | `verl/workers/rollout/sglang_rollout/` |
| Multimodal data config | `verl/trainer/config/data/Qwen3VL.yaml` |
| Example launch scripts | `launch.sh`, `run.sh`, `examples/grpo_trainer/*.sh` |

### Training interface

Typical command shape:

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=<your_config>.yaml \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  actor_rollout_ref.model.path=/path/to/vlm \
  actor_rollout_ref.rollout.name=vllm
```

Important config fields:

```yaml
data:
  train_files: /path/to/train.parquet
  val_files: /path/to/test.parquet
  prompt_key: prompt
  image_key: images
  video_key: videos
  return_multi_modal_inputs: True
  return_raw_chat: True

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  model:
    path: /path/to/vlm
  rollout:
    name: vllm
    n: 8
```

### Data interface

To reuse this path with a new dataset, convert the dataset into Verl parquet with a stable schema:

```python
{
    "data_source": "your_dataset_name",
    "prompt": [
        {"role": "user", "content": "question text"}
    ],
    "ability": "math_or_vqa_or_reasoning",
    "reward_model": {
        "ground_truth": "reference answer or serialized metadata",
        "style": "rule"
    },
    "extra_info": {
        "problem": "raw problem text",
        # other metadata if needed
    },
    "images": [...],
    "videos": [...],
}
```

Notes:

- Keep `prompt` in chat format when possible.
- Put reward-side metadata in `reward_model.ground_truth` and `extra_info`.
- Align multimodal field names with `image_key` and `video_key` in the data config.
- For images, the most stable parquet format is `{"bytes": image_bytes}`.
- For videos, each item should be a dict with a `video` key, following Qwen-VL conventions, e.g. `{"video": "file:///path/to/video.mp4", "fps": 2, "max_frames": 32}`.
- `data_source` and `extra_info` are passed to the reward function.
- Top-level `ability` is not passed as a separate reward-function argument by default. If reward logic needs it, put it inside `extra_info`.

### Note on `reward_model.style`

For our LMM-based reward loop, keep:

```json
"reward_model": {
  "ground_truth": "reference answer or serialized metadata",
  "style": "rule"
}
```

This is because the LMM judge is called inside a custom reward function. From Verl's data schema perspective, this still follows the rule/custom-reward path.

Use `"style": "model"` only when a sample should be scored directly by Verl's built-in reward-model forward path.

### Example training data

```text
data/VisionR1_DAPO
```

---

## 2. Reward Loop with LMM Judge

### What we validated

We validated that Verl reward computation can call an external judge service:

```text
rollout response
  -> decode response
  -> custom reward function
  -> call rule judge / LMM judge / reward model
  -> return score + optional metrics
  -> reward manager writes reward
```

This enables more flexible reward definitions:

- Final-answer equivalence judge.
- Process-level judge.
- Reasoning quality judge.
- Hallucination judge.
- Multimodal judge using images.
- Structured-output judge with guided JSON.
- Reward model ablations across different judge models.

### How to enable it: from config to custom reward

A clear way to understand the reward loop is:

```text
YAML / CLI config
  -> launch judge model service
  -> pass reward_router_address to custom reward function
  -> custom reward function calls judge endpoint
  -> custom reward function returns score
```

Some options can be written in the selected YAML config, or passed as CLI overrides. In our experiments, several reward-loop and VL-judge options were already included in the config file, so they did not need to be repeated in the launch script.

### Step 1: Enable the reward model service

Example CLI overrides:

```bash
reward_model.enable=True \
reward_model.enable_resource_pool=True \
reward_model.n_gpus_per_node=8 \
reward_model.nnodes=1 \
reward_model.model.path=/path/to/judge_model \
reward_model.rollout.name=vllm \
reward_model.rollout.tensor_model_parallel_size=8 \
reward_model.rollout.max_model_len=12800 \
reward_model.rollout.prompt_length=12000 \
reward_model.rollout.response_length=512
```

`reward_model.use_reward_loop=True` may already come from the reward-loop base config, for example a config that inherits `megatron_reward_loop.yaml`.

### Step 2: Point Verl to the custom reward function

Example:

```bash
custom_reward_function.path=/abs/path/to/verl/utils/reward_score/my_reward.py \
custom_reward_function.name=compute_score_my_reward
```

Optional reward parameters can be passed through `reward_kwargs`:

```bash
+custom_reward_function.reward_kwargs.my_weight=0.5 \
+custom_reward_function.reward_kwargs.judge_max_tokens=512
```

### Step 3: Preserve multimodal context for VL judge if needed

For text-only judge, this is not needed.

For VL judge, make sure raw prompt / multimodal data are preserved. This can be done in YAML:

```yaml
reward_model:
  reward_kwargs:
    preserve_non_tensor_keys:
      - raw_prompt
      - multi_modal_data
    forward_non_tensor_keys_to_extra_info:
      - raw_prompt
      - multi_modal_data
```

or as CLI overrides:

```bash
+reward_model.reward_kwargs.preserve_non_tensor_keys='[raw_prompt,multi_modal_data]' \
+reward_model.reward_kwargs.forward_non_tensor_keys_to_extra_info='[raw_prompt,multi_modal_data]'
```

In our VL-judge experiment, these fields were already included in the selected config, so they were not repeated in the launch script.

### What Verl passes into the custom reward function

The reward manager decodes each rollout response and calls the custom reward function roughly like this:

```python
result = compute_score_my_reward(
    data_source=data_source,
    solution_str=response_str,
    ground_truth=ground_truth,
    extra_info=extra_info,
    reward_router_address=reward_router_address,
    **reward_kwargs,
)
```

Important fields:

- `data_source`: from the parquet top-level `data_source` field.
- `solution_str`: decoded model rollout response.
- `ground_truth`: from `reward_model.ground_truth`.
- `extra_info`: from the parquet `extra_info` field.
- `reward_router_address`: address of the launched judge model service.
- `reward_kwargs`: parameters from `custom_reward_function.reward_kwargs`.

### How the custom reward function calls the judge model

Inside the reward function, call the judge service through the OpenAI-compatible endpoint:

```text
POST http://{reward_router_address}/v1/chat/completions
```

Minimal example:

```python
async def compute_score_my_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    judge_max_tokens=512,
    **kwargs,
):
    prompt = build_judge_prompt(
        question=(extra_info or {}).get("problem", ""),
        prediction=solution_str,
        reference=ground_truth,
    )

    response = await generate_chat_aiohttp(
        router_address=reward_router_address,
        messages=[{"role": "user", "content": prompt}],
        sampling_params={"temperature": 0.0, "max_tokens": judge_max_tokens},
    )

    judge_result = parse_judge_json(response)
    score = float(judge_result.get("score", 0.0))

    return {
        "score": score,  # required if returning a dict
        # optional logging/debug metrics:
        # "acc": float(judge_result.get("correct", False)),
        # "judge_reason": judge_result.get("reason", ""),
    }
```

Return interface:

```python
return score
```

or:

```python
return {"score": score}
```

If returning a dict, `score` is required. Other keys are optional logging/debug metrics.

### Multimodal judge call

If the reward judge needs images, read them from `extra_info["multi_modal_data"]`, convert them to data URLs, and send a multimodal chat request:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,...",
                "detail": "auto",
            },
        },
    ],
}]
```

Reference implementation:

```text
verl/utils/reward_score/short_cot_qwen_gpt_step_bonus_multimodal.py
```

Practical recommendations:

- Return extra fields only for debugging/metrics.
- Add fallback logic if the judge output is not valid JSON.
- For multimodal judge, return a metric such as `judge_used_images` to confirm images were actually used.

### Minimal mental model

```text
config enables reward model service
  -> Verl launches judge model and provides reward_router_address
  -> rollout response is decoded
  -> compute_score_xxx(...) is called
  -> custom reward function calls http://reward_router_address/v1/chat/completions
  -> custom reward function returns score or {"score": ...}
  -> reward manager writes reward
```

### Key code locations

| Purpose | Location |
| --- | --- |
| Custom reward loading | `verl/trainer/ppo/reward.py` |
| Reward manager registry | `verl/workers/reward_manager/registry.py` |
| Naive reward manager | `verl/workers/reward_manager/naive.py` |
| DAPO reward manager | `verl/workers/reward_manager/dapo.py` |
| Batch reward manager | `verl/workers/reward_manager/batch.py` |
| Reward functions | `verl/utils/reward_score/` |
| LMM judge example | `verl/utils/reward_score/short_cot_qwen.py` |
| Multimodal judge example | `verl/utils/reward_score/short_cot_qwen_gpt_step_bonus_multimodal.py` |
| Reward tests | `tests/utils/reward_score/test_short_cot_qwen.py` |

---

## 3. Auto-save + Auto-eval

### What we validated

We validated automatic evaluation after checkpoint save:

```text
save checkpoint
  -> local checkpoint ready
  -> upload/copy to remote storage
  -> wait until remote files are ready
  -> submit evaluation job
  -> benchmark results generated
```

This reduces manual experiment overhead:

- No need to manually locate checkpoints.
- No need to manually launch every benchmark.
- Checkpoint step and eval result are naturally aligned.
- Failed or missing evals can be re-submitted with a standalone script.

### Key code locations

| Purpose | Location |
| --- | --- |
| Checkpoint save and auto-eval trigger | `verl/trainer/ppo/ray_trainer.py` |
| Eval job launcher | `verl/trainer/ppo/launch_eval_task.py` |
| Optional eval command hook | `AUTO_EVAL_LAUNCH_COMMAND` |
| Megatron checkpoint upload | `verl/utils/checkpoint/megatron_checkpoint_manager.py` |
| FSDP checkpoint upload | `verl/utils/checkpoint/fsdp_checkpoint_manager.py` |
| Manual eval submission | `submit_auto_eval.py` |

### Config interface

Auto-eval is disabled by default. Enable it only after setting `AUTO_EVAL_LAUNCH_COMMAND` for your evaluation environment:

```bash
+trainer.auto_eval.enabled=True \
+trainer.auto_eval.prompt_type=$PROMPT_TYPE \
+trainer.auto_eval.model_type=$MODEL_TYPE \
+trainer.auto_eval.tasks=$TASKS \
+trainer.auto_eval.pool=$POOL
```

Set checkpoint locations:

```bash
trainer.default_local_dir=/path/to/local_ckpt \
trainer.default_hdfs_dir=hdfs://path/to/remote_ckpt
```

Example task list:

```bash
TASKS="[MathVista_MINI,MathVerse_MINI,LogicVista,GSM8K,MATH-500]"
```

### Control flow

In `ray_trainer.py`, after checkpoint save:

```python
auto_eval_config = self.config.trainer.get("auto_eval", {})
```

If enabled:

```python
auto_eval_config.get("enabled", False)
```

the trainer submits:

```python
submit_auto_eval_task(
    local_path=actor_local_path_hf,
    hdfs_path=actor_remote_path_hf,
    auto_eval_config=auto_eval_config,
    global_steps=str(self.global_steps),
)
```

`submit_auto_eval_task` then:

1. Traverses local checkpoint files.
2. Maps them to remote paths.
3. Waits until remote files exist.
4. Calls `launch_eval_tasks(...)` for each benchmark task.

---

## 4. Short Talk Version

The reusable outcome is not a specific experiment setup. It is a practical Verl workflow with three pieces:

1. **Multimodal RL training**: VLM policies can run through dataset → rollout → reward → GRPO/PPO update → checkpoint.
2. **LMM-based reward loop**: reward functions can call an external LMM/reward model for flexible judging, including multimodal judging.
3. **Auto-eval loop**: saved checkpoints can automatically trigger benchmark evaluation.

For reuse, the key interfaces are:

- **Data**: convert to Verl parquet.
- **Reward**: implement `compute_score_xxx(...)` and configure it in YAML.
- **Eval**: add tasks to `trainer.auto_eval.tasks` or customize `launch_eval_task.py`.
