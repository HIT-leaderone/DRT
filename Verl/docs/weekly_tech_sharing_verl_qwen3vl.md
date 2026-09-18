# Verl 新功能实践总结：Qwen3-VL / Vision-R1 多模态 RL

> Weekly technique sharing 草稿  
> 主题：基于开源 Verl 跑通 Qwen3-VL 多模态 GRPO，并扩展过程奖励、远程 Reward Model、自动评测与 ablation 实验能力。

## 1. 一句话总结

这次工作基于开源 Verl，已经把 **Qwen3-VL-8B 的多模态 GRPO 训练链路**跑通，并在原始 RL 框架上补齐了面向 Vision-R1 / Geo3K / DAPO-Math 任务的几类能力：

- Qwen3-VL / Qwen3-VL-MoE 的 Megatron + vLLM 训练与 rollout 适配；
- dense CoT 数据格式与 process supervision 数据转换；
- 基于 Qwen3-235B 的远程 reward judge / process reward；
- step bonus、deep exploration bonus、hallucination penalty 等更细粒度奖励设计；
- 多模态 judge：reward model 在判分时可以看原图；
- checkpoint 保存后自动拉起 VLMEvalKit 评测；
- result-only、process-partial、no-step-bonus、guided-json、multi-modal judge 等 ablation 配置。

## 2. 现在已经跑通的 Verl 部分

### 2.1 多模态 GRPO 主训练链路

当前主链路是：

```text
Parquet 多模态数据
  -> Qwen3-VL dataset / processor
  -> vLLM rollout 采样 n 条 responses
  -> Qwen3-235B reward judge / rule reward 计算 token-level score
  -> GRPO advantage
  -> Megatron actor update
  -> checkpoint 保存到 local + HDFS
  -> 可选 auto-eval
```

已经具备的训练形态：

- **算法**：GRPO，`algorithm.adv_estimator=grpo`；
- **Actor**：Qwen3-VL-8B SFT checkpoint；
- **Rollout engine**：vLLM；
- **训练后端**：Megatron，支持 TP / PP / CP；
- **Reward**：自定义 process reward + 可选 Qwen3-235B 远程 judge；
- **数据**：Geo3K dense-CoT、VisionR1_DAPO、Dapo Math 等混合/扩展数据；
- **评测**：保存 checkpoint 后自动提交 VLMEvalKit 任务。

关键入口：

- `launch.sh`：当前更完整的主训练入口，带参数化路径、reward model、auto-eval；
- `run.sh`：早期 Geo3K dense-CoT with process reward 训练入口；
- `examples/grpo_trainer/run_qwen3_vl-8b-megatron_dense_cot_with_process.sh`：Qwen3-VL + Geo3K process reward 示例；
- `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo.yaml`：当前主 reward 方案配置。

### 2.2 Qwen3-VL / Qwen3-VL-MoE 支持

仓库里补齐了 Qwen3-VL 在 Verl 训练中的模型适配：

- `verl/models/transformers/qwen3_vl.py`
  - Qwen3-VL position ids / RoPE index 处理；
  - image/video token embedding scatter；
  - PPO 训练需要的 forward output 适配；
  - Qwen3-VL-MoE sparse MoE block 的 transformers 版本 bug patch。
- `verl/models/transformers/monkey_patch.py`
  - 将 `qwen3_vl` / `qwen3_vl_moe` 接入 Verl monkey patch 路径；
  - 支持 Ulysses input slicing。
- `verl/models/mcore/registry.py`
  - 注册 `Qwen3VLForConditionalGeneration` 与 `Qwen3VLMoeForConditionalGeneration`。

这部分的结果是：Qwen3-VL 不只是能推理，而是可以进入 Verl 的 actor / rollout / logprob / ref policy 训练闭环。

### 2.3 Megatron + vLLM 大模型训练配置

训练脚本中已经使用了比较完整的工程配置：

- TP / PP / CP 参数化：`GEN_TP`, `TP`, `PP`, `CP`；
- vLLM rollout tensor parallel；
- actor / ref 的 dynamic batch size；
- Megatron `mbridge`；
- 参数、梯度、优化器 offload；
- recompute、gradient accumulation fusion、MoE router dtype 等配置；
- rollout 多采样：主配置里 `rollout.n=16`。

示例配置位置：

- `launch.sh`
- `run.sh`
- `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen.yaml`

## 3. 新增 / 强化的 Verl 功能

### 3.1 Dense Cognitive Trace 数据格式

为了让 VLM 学习“短而密”的推理，数据转换脚本统一构造了 dense CoT prompt：

```text
<visual>...concise visual evidence...</visual>
<think>...[Priors] -> ...compressed reasoning...</think>
<answer>...final answer...</answer>
```

目标不是鼓励长 CoT，而是鼓励：

- 视觉证据显式化；
- 推理链压缩、结构化；
- 最终答案可抽取、可规则判分；
- process reward 可以对齐 reference steps。

相关脚本：

- `examples/grpo_trainer/convert_dense_cot_with_process.py`：Geo3K 转 Verl parquet，并把 `processed_steps` 写入 `reward_model.ground_truth`；
- `convert_vision_r1_v2.py`：Vision-R1 / DAPO 数据转换与可选 GPT 生成 reference steps；
- `merge_geo_datasets.py`, `merge_and_check_geo_datasets.py`：数据合并与检查。

### 3.2 自定义 reward function 支持 async / kwargs

原始 Verl 的 reward function 更多是同步 rule reward。这里做了两点增强：

1. **reward function 可以是 async**：便于调用 OpenAI-compatible vLLM server / 远程 judge；
2. **reward kwargs 可以从 yaml 注入**：不同实验只改 config，不改代码。

关键实现：

- `verl/trainer/ppo/reward.py`
  - `_bind_reward_kwargs`：把 config 中的 `reward_kwargs` 绑定到 reward function；
  - `_ensure_sync_reward_fn`：让 async reward function 可以被同步 reward manager 调用；
  - `_split_reward_manager_kwargs`：区分 reward manager 初始化参数与 reward function 参数。

这让我们可以在 yaml 中快速切换：

- process reward 参数；
- step bonus 权重；
- deep exploration bonus；
- hallucination penalty；
- judge model / router address；
- 是否使用 guided JSON；
- 是否开启多模态 judge。

### 3.3 Qwen3-235B 远程 reward judge

主 reward 不只依赖 rule-based answer match，而是可以通过 OpenAI-compatible endpoint 调用 Qwen3-235B 作为 grader：

- `verl/utils/reward_score/short_cot_qwen.py`
  - final answer 抽取；
  - `<think>` / `<answer>` 格式检查；
  - GPT/Qwen judge 调用；
  - step-level matched count；
  - hallucination 检测；
  - deep exploration 计数。

主配置：

- `custom_reward_function.name=compute_score_qwen_gpt_step_bonus`
- `reward_model.model.path=Qwen/Qwen3-235B-A22B-Instruct-2507`
- `reward_model.rollout.name=vllm`

### 3.4 Process reward：部分正确也能给 credit

当前 reward 不是简单 0/1，而是拆成多个组件：

| 组件 | 作用 |
| --- | --- |
| format/style reward | 鼓励固定 XML 结构、箭头式 dense reasoning |
| answer reward | final answer rule judge 或 GPT judge 正确时给分 |
| process partial credit | 按 matched reference steps 比例给部分分 |
| step bonus | 推理步骤覆盖越充分，额外奖励越高 |
| deep exploration bonus | 难题中有有效额外验证/分支探索时给奖励 |
| hallucination penalty | 匹配步骤里出现编造视觉/数学依据时扣分 |
| density penalty | 防止伪 dense、长句灌水 |

相关文件：

- `verl/utils/reward_score/short_cot_qwen.py`
- `verl/utils/reward_score/short_cot_qwen_unit_rewards.py`
- `verl/utils/reward_score/short_cot_qwen_adaptive_depth.py`

### 3.5 多模态 reward judge：判分时看图

新增了多模态 judge 版本：

- `verl/utils/reward_score/short_cot_qwen_gpt_step_bonus_multimodal.py`

核心能力：

- 从 `extra_info.multi_modal_data.image` 中取回原图；
- 转成 base64 data URL；
- 把题面、reference steps、学生推理和图像一起发给 judge；
- 如果多模态返回不可解析，自动 fallback 到 text-only judge；
- 返回 `judge_used_images`，方便训练日志里确认是否真的用了图像。

对应配置：

- `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_mmjudge_visionr1_dapo.yaml`

为了让 reward 侧拿到原始图像，还在 config 中保留了 non-tensor 字段：

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

### 3.6 Reward ablation 配置

为了验证奖励设计是否真的有效，已经补了几套 ablation：

| 配置 | 目的 |
| --- | --- |
| `...result_only_visionr1_dapo.yaml` | 只看最终答案 |
| `...process_partial_1p0_visionr1_dapo.yaml` | process partial credit，不加额外 step bonus |
| `...format_answer_only_visionr1_dapo.yaml` | 保留格式 + 答案，移除过程奖励 |
| `...gpt_step_bonus_visionr1_dapo_no_step_bonus.yaml` | 主 reward 但关闭 step/deep bonus |
| `...gpt_step_bonus_guided_json_visionr1_dapo.yaml` | judge 使用 guided JSON schema，提升解析稳定性 |
| `...gpt_step_bonus_mmjudge_visionr1_dapo.yaml` | 多模态 judge 看图判分 |
| `...adaptive_depth_visionr1_dapo.yaml` | 按数据源/难度自适应 depth 奖励 |

这部分对 weekly sharing 比较重要：不是只“加了一个 reward”，而是把 reward 方案拆成了可复现实验矩阵。

### 3.7 DAPO reward manager / overlong penalty

训练中使用 `reward_model.reward_manager=dapo`，并开启 overlong buffer：

- `verl/workers/reward_manager/dapo.py`
- `+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True`
- `+reward_model.reward_kwargs.overlong_buffer_cfg.len=512`
- `+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0`

作用：

- reward 落在 response 的最后一个有效 token；
- 保留 reward extra info，便于日志分析；
- 对超长 response 给连续惩罚，避免模型通过变长/灌水拿奖励。

### 3.8 Reward function 异步 Ray 计算

训练 loop 中支持 reward function 异步提交：

- `verl/trainer/ppo/ray_trainer.py`
- `reward_model.launch_reward_fn_async`
- `compute_reward_async.remote(...)`

这样 reward judge 较慢时，可以把 reward computation 与后续部分训练流程解耦，降低主 loop 阻塞风险。

### 3.9 Checkpoint 后自动评测

新增 auto-eval 逻辑：checkpoint 保存后，等待 HDFS 文件齐全，然后自动提交 VLMEvalKit 任务。

相关文件：

- `verl/trainer/ppo/ray_trainer.py`
  - `submit_auto_eval_task(...)`
  - checkpoint 保存后读取 `trainer.auto_eval` 并异步提交；
- `verl/trainer/ppo/launch_eval_task.py`
  - 构造 VLMEvalKit job；
  - 支持按 task 拆分提交；
- `submit_auto_eval.py`
  - 独立补交/调试 auto-eval 任务。

当前入口脚本中默认评测任务包括：

```text
MathVista_MINI, MathVerse_MINI, LogicVista, Video_Holmes, GSM8K, MATH-500
```

### 3.10 评测与数据前处理辅助工具

为了支撑多 benchmark 评测和数据清洗，近期还加了多类脚本：

- `convert_mathvista.py`
- `convert_mathbench_vqa.py`
- `convert_mathverse.py`
- `convert_logicvista.py`
- `mathverse_policy_prefilter_experiment.py`
- `prefilter_mathverse_gpt4o.py`
- `submit_reward_model_size_jobs.py`

其中 `mathverse_policy_prefilter_experiment.py` / `prefilter_mathverse_gpt4o.py` 是为了解决评测时部分 prompt 可能被外部模型安全策略拦截的问题：先做 prefilter / probe，减少无效评测失败。

## 4. 当前主实验配置

主训练入口可以概括为：

```bash
bash launch.sh vllm
```

核心默认设置：

```text
Actor model: Qwen/Qwen3-VL-8B-Instruct
Reward model: Qwen/Qwen3-235B-A22B-Instruct-2507
Train data:  data/VisionR1_DAPO/train.parquet
Test data:   data/VisionR1_DAPO/test.parquet
Config:      ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo.yaml
Algorithm:   GRPO
Rollout n:   16
Batch size:  512
Max prompt:  2048
Max response:2048
```

并行配置：

```text
GEN_TP=4
TP=2
PP=2
CP=1
RM_NODES=1
```

日志与产物：

```text
Local ckpt: checkpoints/<project>/<exp>
HDFS ckpt:  optional hdfs://.../<project>/<exp>
Logger:     console by default
Auto eval:  disabled unless AUTO_EVAL_ENABLED=True and AUTO_EVAL_LAUNCH_COMMAND is set
```

## 5. 可以在分享里重点讲的技术点

### 5.1 为什么要 dense CoT？

普通长 CoT 在 VLM RL 中容易带来两个问题：

1. 训练 token 成本高；
2. 模型会学会“说很多”，但不一定更准确。

Dense Cognitive Trace 的目标是让模型输出：

- 短；
- 信息密；
- 可解析；
- 可判分；
- 对视觉证据和推理链都有约束。

### 5.2 为什么 reward 要拆成多个组件？

只看最终答案会导致 credit assignment 太粗。当前拆法能区分：

- 答案对，但过程乱；
- 答案错，但关键中间步骤对；
- 过程覆盖了 reference steps，但有 hallucination；
- 简单题过度推理 vs 难题必要探索。

这使得 RL 信号更稳定，也更容易做 ablation。

### 5.3 为什么要 Qwen3-235B 做 judge？

多模态数学/几何题中，规则判分只能覆盖 final answer。过程是否合理、是否编造视觉条件，很难只用 regex 判断。远程大模型 judge 负责：

- reference step matching；
- hallucination 检查；
- deep exploration 计数；
- answer 等价判断 fallback。

### 5.4 为什么要多模态 judge？

如果 judge 只看文字，它无法判断模型是否“编造了图中不存在的视觉证据”。多模态 judge 让 reward model 也看到原图，能更准确惩罚视觉 hallucination。

### 5.5 为什么 auto-eval 很重要？

RL 实验往往 checkpoint 很多，手动评测成本高且容易漏。自动评测带来的收益：

- 每个 checkpoint 产出后自动进入 benchmark；
- 训练曲线与 eval 曲线更容易对齐；
- 支持多任务横向观察：MathVista / MathVerse / LogicVista / Video / Text Math。

## 6. 当前风险和后续 TODO

### 风险

- Reward judge 调用成本较高，吞吐容易成为瓶颈；
- 多模态 judge 依赖 non-tensor 字段完整透传，数据格式不一致时容易 fallback 到 text-only；
- GPT/Qwen judge 输出 JSON 仍可能解析失败，因此 guided JSON 和 fallback 很重要；
- auto-eval 依赖外部队列、HDFS、VLMEvalKit 环境，失败时需要单独补交；
- reward 组件较多，必须依赖 ablation 避免“奖励堆叠但不知道哪个有效”。

### 后续可以继续做

- 把 reward judge 做 batch 化 / server-side batching；
- 增加 reward cache，避免同一 response 反复 judge；
- 整理统一实验表：result-only / process / step-bonus / mmjudge / adaptive-depth；
- 将 auto-eval 结果回写到统一 dashboard；
- 对 hallucination penalty 做更细粒度分桶：visual hallucination vs math hallucination；
- 将数据转换脚本规范化成一个 CLI，减少散落脚本。

## 7. 代码索引

| 模块 | 文件 |
| --- | --- |
| 主训练入口 | `launch.sh`, `run.sh` |
| Qwen3-VL 示例 | `examples/grpo_trainer/run_qwen3_vl-8b-megatron_dense_cot_with_process.sh` |
| 主训练配置 | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo.yaml` |
| Qwen3-VL 数据配置 | `verl/trainer/config/data/Qwen3VL.yaml` |
| Qwen3-VL 模型 patch | `verl/models/transformers/qwen3_vl.py`, `verl/models/transformers/monkey_patch.py` |
| Reward 加载/async 适配 | `verl/trainer/ppo/reward.py` |
| DAPO reward manager | `verl/workers/reward_manager/dapo.py` |
| 主 process reward | `verl/utils/reward_score/short_cot_qwen.py` |
| 多模态 judge reward | `verl/utils/reward_score/short_cot_qwen_gpt_step_bonus_multimodal.py` |
| ablation reward | `verl/utils/reward_score/short_cot_qwen_unit_rewards.py` |
| adaptive depth reward | `verl/utils/reward_score/short_cot_qwen_adaptive_depth.py` |
| Geo3K 转换 | `examples/grpo_trainer/convert_dense_cot_with_process.py` |
| Vision-R1 / DAPO 转换 | `convert_vision_r1_v2.py` |
| auto-eval | `verl/trainer/ppo/launch_eval_task.py`, `submit_auto_eval.py` |
| reward 单测 | `tests/utils/reward_score/test_short_cot_qwen.py` |

## 8. 分享时可用的结尾

这次工作不是简单地“把 Verl 跑起来”，而是把开源 Verl 扩展成了一个可支撑多模态数学 RL 实验的平台：

1. **训练链路跑通**：Qwen3-VL + Megatron + vLLM + GRPO；
2. **奖励可实验**：answer / process / step bonus / hallucination / mmjudge 都能配置化切换；
3. **数据可扩展**：Geo3K、Vision-R1、DAPO-Math 都能转成统一 Verl parquet；
4. **评测可自动化**：checkpoint -> HDFS -> VLMEvalKit 自动评测；
5. **后续可复现 ablation**：多个 yaml 已经对应好不同实验假设。
