# Practical Verl Extensions: Multimodal RL, LMM Reward Judge, and Auto-Eval

> Scope: reusable engineering experience from our Verl-based RL work.  
> Focus: what we validated, where the code lives, what interfaces matter, and how others can reuse or extend it.

## 0. Executive Summary

We validated three reusable capabilities on top of Verl:

1. **Multimodal RL Training**  
   Running VLM RL with image/video inputs, Megatron actor training, vLLM rollout, and GRPO/PPO-style updates.

2. **Reward Loop with LMM Judge**  
   Calling an external LMM/reward model during reward computation, so reward can go beyond simple rule-based final-answer matching.

3. **Auto-save + Auto-eval**  
   Saving checkpoints, uploading them to remote storage, and automatically launching benchmark evaluation jobs.

Together, these form a practical experiment loop:

```text
train multimodal policy -> compute flexible rewards -> auto-evaluate checkpoints
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

The key validated pieces are:

- Reading multimodal samples: text + image/video fields.
- Running rollout with vLLM.
- Training the actor with Megatron.
- Passing sample metadata into reward computation.
- Keeping dynamic batching / sequence balancing compatible with multimodal fields.

### Key code locations

| Purpose | Location |
| --- | --- |
| Training entrypoint | `verl/trainer/main_ppo.py` |
| PPO/GRPO trainer loop | `verl/trainer/ppo/ray_trainer.py` |
| Megatron workers | `verl/workers/megatron_workers.py` |
| FSDP workers | `verl/workers/fsdp_workers.py` |
| vLLM rollout | `verl/workers/rollout/vllm_rollout/` |
| SGLang rollout | `verl/workers/rollout/sglang_rollout/` |
| Multimodal data config | `verl/trainer/config/data/Qwen3VL.yaml` |
| Multimodal sequence balancing test | `tests/utils/test_seqlen_balancing.py` |
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
        "style": "rule"  # keep "rule" when using a custom reward function, even if it calls an LMM judge
    },
    "extra_info": {
        "problem": "raw problem text",
        "original_index": 0,
    },
    "images": [...],
    "videos": [...],
}
```

Practical notes:

- Keep `prompt` in chat format when possible.
- Put reward-side metadata in `reward_model.ground_truth` and `extra_info`.
- Align multimodal field names with `image_key` and `video_key` in the config.
- Start with a small parquet and a small batch smoke test before scaling.

Reference conversion scripts:

- `examples/grpo_trainer/convert_dense_cot_with_process.py`
- `convert_vision_r1.py`
- `convert_vision_r1_v2.py`
- `prepare_multimodal_data.py`
- `convert_mathvista.py`
- `convert_mathverse.py`

---

## 2. Reward Loop with LMM Judge

### What we validated

We validated that Verl reward computation can call an external judge service:

```text
rollout response
  -> decode response
  -> custom reward function
  -> call rule judge / LMM judge / reward model
  -> return score + metrics
  -> reward manager writes token-level reward
```

This enables more flexible reward definitions:

- Final-answer equivalence judge.
- Process-level judge.
- Reasoning quality judge.
- Hallucination judge.
- Multimodal judge using images.
- Structured-output judge with guided JSON.
- Reward model ablations across different judge models.

### How to use the reward loop: from entry args to custom reward

A clearer way to understand the reward loop is to start from the training entry command, then follow how the custom reward function calls the judge model.

#### Step 1: Add reward-loop related arguments at training entry

At launch time, we still use the normal PPO/GRPO entrypoint:

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=<your_config>.yaml \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  actor_rollout_ref.model.path=/path/to/policy_model \
  actor_rollout_ref.rollout.name=vllm
```

To use an LMM judge during reward computation, add three groups of configs.

**1. Enable the reward model service:**

```bash
reward_model.enable=True \
reward_model.use_reward_loop=True \
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

**2. Point Verl to the custom reward function:**

```bash
custom_reward_function.path=/abs/path/to/verl/utils/reward_score/my_reward.py \
custom_reward_function.name=compute_score_my_reward
```

Optional reward parameters can be passed through `reward_kwargs`:

```bash
+custom_reward_function.reward_kwargs.my_weight=0.5 \
+custom_reward_function.reward_kwargs.judge_max_tokens=512
```

**3. Preserve multimodal context if the judge needs images/videos:**

```bash
+reward_model.reward_kwargs.preserve_non_tensor_keys='[raw_prompt,multi_modal_data]' \
+reward_model.reward_kwargs.forward_non_tensor_keys_to_extra_info='[raw_prompt,multi_modal_data]'
```

If the judge is text-only, this part is not needed.

#### Step 2: Keep training data on the custom/rule reward path

Even if the reward is computed by an LMM, the parquet data should still use:

```json
"reward_model": {
  "ground_truth": "reference answer or serialized metadata",
  "style": "rule"
}
```

Reason: the LMM judge is called inside the custom reward function. From Verl's data schema perspective, this is still the rule/custom-reward path.

Use `"style": "model"` only when a sample should be scored directly by Verl's built-in reward-model forward path.

#### Step 3: What Verl passes into the custom reward function

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

Note: top-level `ability` is not passed as a separate argument by default. If reward logic needs it, put it inside `extra_info`.

#### Step 4: How the custom reward function calls the judge model

Inside the reward function, call the judge service through the OpenAI-compatible endpoint:

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
        # optional metrics for logging/debugging:
        # "acc": float(judge_result.get("correct", False)),
        # "judge_reason": judge_result.get("reason", ""),
    }
```

The endpoint called by `generate_chat_aiohttp` is:

```text
POST http://{reward_router_address}/v1/chat/completions
```

#### Step 5: Multimodal judge call

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

- Return only `score` if no extra logging is needed.
- Return extra fields only for debugging/metrics.
- Add fallback logic if the judge output is not valid JSON.
- For multimodal judge, return a metric such as `judge_used_images` to confirm images were actually used.

Minimal mental model:

```text
entry args enable reward model service
  -> Verl launches judge model and provides reward_router_address
  -> rollout response is decoded
  -> compute_score_xxx(...) is called
  -> custom reward function calls http://reward_router_address/v1/chat/completions
  -> custom reward function returns {"score": ...}
  -> reward manager writes token-level reward
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

### Important note: `style: rule` vs LMM judge

For our LMM-based reward loop, the training data still uses:

```json
"reward_model": {
  "style": "rule",
  "ground_truth": "..."
}
```

This is because the LMM judge is called inside a custom reward function. From Verl's data schema perspective, it is still a rule/custom reward path, not the built-in model-RM scoring path. Use `style: model` only when the sample should be scored directly by a reward model forward pass.

### Custom reward interface

Config interface:

```yaml
custom_reward_function:
  path: /abs/path/to/reward_file.py
  name: compute_score_xxx
  reward_kwargs:
    key1: value1
    key2: value2
```

Recommended Python interface:

```python
async def compute_score_xxx(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    **kwargs,
):
    return {
        "score": score,
        "acc": acc,
        "reward_type": "xxx",
        # extra metrics
    }
```

Field meanings:

- `data_source`: dataset identifier; useful for dataset-specific scoring.
- `solution_str`: decoded model response.
- `ground_truth`: from `reward_model.ground_truth` in parquet.
- `extra_info`: sample metadata.
- `reward_router_address`: external judge service address.
- `kwargs`: values passed from `reward_kwargs` in YAML.

Return value:

```python
{
    "score": 0.8,        # required
    "acc": 1.0,          # recommended
    "reward_type": "...",
    "metric_a": ...,
    "metric_b": ...,
}
```

`score` is used as the training reward. Other fields are collected as reward extra info for logging/debugging.

### LMM judge service interface

The judge service uses an OpenAI-compatible chat completions API:

```text
POST http://{router_address}/v1/chat/completions
```

Text judge payload:

```python
payload = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.0,
    "max_tokens": 512,
}
```

Multimodal judge payload:

```python
payload = {
    "model": model_name,
    "messages": [
        {
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
        }
    ],
    "temperature": 0.0,
    "max_tokens": 768,
}
```

Reference implementations:

- `generate_chat_aiohttp(...)`
- `generate_multimodal_chat_aiohttp(...)`

### Multimodal judge requirements

For a reward judge to see images, the reward path must preserve non-tensor fields:

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

Practical recommendations:

- Extract images from `extra_info` inside the reward function.
- Convert images to base64 data URLs before calling the judge.
- Fallback to text-only judge if multimodal judge output is invalid.
- Return a metric such as `judge_used_images` to verify that image input was actually used.

### How to add a new reward

1. Add a reward file, for example:

```text
verl/utils/reward_score/my_reward.py
```

2. Implement:

```python
async def compute_score_my_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    **kwargs,
):
    # parse response
    # call local rule / remote judge / multimodal judge
    # aggregate score
    return {
        "score": score,
        "acc": acc,
        "reward_type": "my_reward",
    }
```

3. Wire it in YAML:

```yaml
custom_reward_function:
  path: /abs/path/to/verl/utils/reward_score/my_reward.py
  name: compute_score_my_reward
  reward_kwargs:
    my_weight: 0.5
```

4. If images are needed, preserve multimodal non-tensor fields.

5. Add tests for parsing, judge output handling, score aggregation, and fallback behavior.

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

### Eval launcher interface

Core function:

```python
def launch_eval_tasks(
    prompt_type,
    model_type,
    task_type,
    hdfs_path=None,
    pool="public",
    iter=0,
    retry=None,
):
    ...
```

Arguments:

- `prompt_type`: evaluation prompt type.
- `model_type`: model name registered in the eval system.
- `task_type`: benchmark name.
- `hdfs_path`: checkpoint path.
- `pool`: resource pool / queue.
- `iter`: checkpoint step.
- `retry`: retry count.

### How to add a new eval

- If the benchmark already exists in the eval platform, add it to:

```bash
+trainer.auto_eval.tasks="[TaskA,TaskB,NewTask]"
```

- To change resources, image, environment variables, or entrypoint, set:

```text
AUTO_EVAL_LAUNCH_COMMAND
```

- To switch evaluation platforms, update the command hook or replace the implementation in:

```text
verl/trainer/ppo/launch_eval_task.py
```

while keeping the same high-level `launch_eval_tasks(...)` interface.

- To manually re-submit evaluation:

```bash
python3 submit_auto_eval.py ...
```

---

## 4. Reusable Takeaways

### Adding a new dataset

Minimal steps:

1. Convert the dataset to Verl parquet.
2. Ensure fields such as `prompt`, `reward_model`, `extra_info`, `images`, and `videos` are present as needed.
3. Replace `data.train_files` and `data.val_files` in YAML or CLI overrides.
4. Reuse an existing reward if possible; otherwise add a custom reward function.
5. Start with a small smoke test before launching large training.

### Adding a new reward

Minimal steps:

1. Implement `compute_score_xxx(...)`.
2. Return a dict with at least `score`.
3. Configure `custom_reward_function.path/name/reward_kwargs`.
4. If using an LMM judge, expose an OpenAI-compatible endpoint.
5. If using images, preserve non-tensor multimodal fields.
6. Add unit tests for parsing and fallback behavior.

### Adding a new evaluation task

Minimal steps:

1. Make sure the eval platform supports the benchmark.
2. Add the task name to `trainer.auto_eval.tasks`.
3. Adjust the job template if resources or environment differ.
4. Use `submit_auto_eval.py` for re-submission or debugging.

---

## 5. Short Talk Version

The main reusable outcome is not a specific experiment setup. It is a practical Verl workflow with three pieces:

1. **Multimodal RL training**: VLM policies can run through dataset → rollout → reward → GRPO/PPO update → checkpoint.
2. **LMM-based reward loop**: reward functions can call an external LMM/reward model for flexible judging, including multimodal judging.
3. **Auto-eval loop**: saved checkpoints can automatically trigger benchmark evaluation.

For reuse, the key interfaces are:

- **Data**: convert to Verl parquet.
- **Reward**: implement `compute_score_xxx(...)` and configure it in YAML.
- **Eval**: add tasks to `trainer.auto_eval.tasks` or customize `launch_eval_task.py`.
