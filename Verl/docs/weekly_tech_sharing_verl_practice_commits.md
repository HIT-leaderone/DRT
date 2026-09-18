# Verl 实践复盘：从 Geo3K 跑通到 Vision-R1-DAPO 多模态 RL

> 范围：`1fb43c4ad9acd17d1e585a68a103b7b815d502dc..HEAD` 之后的所有提交。  
> 目标：比“功能介绍”更偏实践，按提交顺序说明我们每一步解决了什么问题、加了什么能力、最终跑通了哪条链路。

## 0. 最终跑通的主线

我们这轮改动不是一次性大重构，而是沿着一条很实践的路线迭代出来的：

```text
Geo3K parquet / Qwen3-VL baseline
  -> dense CoT prompt + dense reward
  -> process step reward + Qwen/Qwen3-235B judge
  -> 多节点 Megatron + vLLM GRPO
  -> checkpoint auto-save + HDFS + VLMEvalKit auto-eval
  -> merge Vision-R1 + DAPO 数据
  -> step-bonus / length-bonus / adaptive-depth / answer-only / raw-ablation
  -> 多模态 reward judge 看图判分
  -> MathVista/MathVerse/LogicVista 等评测数据转换与 reward model ablation
```

最终我们跑通/补齐的 Verl 能力：

- Qwen3-VL / Qwen2.5-VL 的 GRPO 训练配置；
- Geo3K、Video-R1、Vision-R1、DAPO-Math 等数据转换；
- dense-CoT 和 process-supervision 格式；
- Qwen3-235B 远程 reward judge；
- step-level process reward、length bonus、deep exploration bonus、hallucination penalty；
- Vision-R1-DAPO 主实验配置和多组 reward ablation；
- 多模态 reward judge：judge 能拿到原图；
- checkpoint 自动上传 HDFS，并自动提交 VLMEvalKit 评测；
- 为评测与 reward model 对比补充了 MathVista / MathVerse / LogicVista / MathBench-VQA 转换脚本。

## 1. Commit-by-commit 实践时间线

| Commit | 提交信息 | 主要改动 | 实践意义 |
| --- | --- | --- | --- |
| `6313f6f0` | fix parquet bug | 修 `examples/grpo_trainer/convert.py` 和单样本保存工具 `save_oneitem.py` 的 parquet 处理问题。 | 先把数据格式打通，避免训练入口读 parquet 直接失败。 |
| `eccf1bee` | add geo3k config | 在 reward registry 里接入 Geo3K 数据源。 | 让 Geo3K 可以走 Verl 的 `data_source -> reward_fn` 路由。 |
| `322a7aa5` | add RL data generator | 加 `generate_cot_data.py`、`gpt_model.py`、`read_parquet.py` 和 `main_CurriculumPPO.py`。 | 开始搭建 RL 数据生成/读取和 curriculum PPO 的实验入口。 |
| `a966989d` | add grpo config | 更新 Geo3K convert，并加 Qwen3-VL 8B Megatron baseline GRPO 脚本。 | 第一版 Qwen3-VL + GRPO 启动脚本成型。 |
| `bc35c9a3` | fix bug | 修 baseline run 脚本和 `use_gpu.py`。 | 处理环境/资源脚本问题，让 baseline 能实际启动。 |
| `aa2df8fb` | update transformers | 更新 run 脚本中的 transformers 依赖处理。 | 对齐 Qwen3-VL 所需 transformers 版本。 |
| `468b3a50` | add config | 继续补 Qwen3-VL baseline 脚本配置。 | 收敛训练命令中的路径、并行与依赖参数。 |
| `7d66e643` | geo3k ok | 调整 baseline 脚本直到 Geo3K 路径跑通。 | 标志 Geo3K baseline 训练链路基本可跑。 |
| `bf184b99` | add dense reward | 加 dense reward 配置 `ppo_megatron_trainer_dense_cot.yaml`，扩展 `reward.py` 和 `short_cot.py`。 | 从普通 answer reward 进入 dense-CoT 奖励实验。 |
| `65649070` | update dense cot prompt | 新增 `convert_dense_cot.py` 和 dense-CoT run 脚本，调整 prompt/reward。 | 统一 dense cognitive trace 格式，开始约束 `<think>/<answer>` 输出。 |
| `8e67fa93` | auto download model | run 脚本自动拉模型，并继续调 dense reward。 | 减少手工准备模型步骤，提高任务可复现性。 |
| `85b5d79e` | run video-r1 | 加 Video-R1 数据转换/解压/运行脚本，更新 Qwen3VL 数据配置。 | 把链路从静态图像 Geo3K 扩到 Video-R1 多模态数据。 |
| `07fc99cc` | add test config | 加 Video-R1 test 脚本与 `generate_videor1_test.py`。 | 给 Video-R1 增加小规模测试入口。 |
| `3566bbd4` | 128 max frames | 调整 Video-R1 转换和测试脚本，限制最大帧数。 | 控制视频输入长度，避免 rollout/显存爆炸。 |
| `510ce9ef` | video-r1 test | 完善 Video-R1 test、`.gitignore`、GPU 工具。 | 让 Video-R1 测试更稳定可跑。 |
| `fb65b706` | update qwen2.5vl | 加 Qwen2.5-VL 7B Megatron 脚本，并同步 dense-CoT 脚本。 | 增加 Qwen2.5-VL 对照模型。 |
| `c5340a09` | update qwen2.5vl grpo config | 新增 Qwen2.5-VL / Qwen3-VL dense-CoT Megatron 配置。 | 将模型差异沉到 yaml，训练入口更标准化。 |
| `e25c30c5` | upate qwen2.5vl | 更新 Qwen2.5-VL run 与 sequence balancing 脚本。 | 尝试解决 VL batch 中长度不均衡的问题。 |
| `16f48816` | modify score | 修改 `short_cot.py` 的评分逻辑。 | 调整 dense-CoT 初版 reward shape。 |
| `96516098` | add convert geo3k process data | 加 `convert_geo3k.py`，更新 baseline 和 reward。 | 准备带 reference steps 的 Geo3K process reward 数据。 |
| `8f5a3c37` | update gpt reward | 加 `convert_dense_cot_with_process.py`、process reward 配置、`gpt_model` 和 GPT reward 逻辑。 | 关键节点：从 final-answer reward 升级到 GPT/Qwen step-level process reward。 |
| `783c1f54` | update multi-node train | 加 short-cot 多节点脚本、Qwen reward 配置和 `short_cot_qwen.py`。 | 开始把 reward judge 切到 Qwen 风格，并支持多节点训练。 |
| `5e13d2e0` | fi qwen-reward bug | 加 API 检查、Ray trainer 修复、Qwen reward bugfix。 | 修远程 reward judge 调用/训练集成中的实际问题。 |
| `3b4e2d5a` | 2 node | 更新环境脚本、convert 脚本和 Qwen reward。 | 继续收敛 2 节点 Megatron/Ray 训练配置。 |
| `978c1e26` | update 235B model | 更新 `run.sh` 和 reward 默认模型到 235B judge。 | 将 reward judge 升级为 Qwen3-235B。 |
| `00a9fea9` | update 2 node | 更新 `run.sh`。 | 调整 2 节点启动参数。 |
| `eb714f9b` | Update run.sh | 更新 `run.sh`。 | 继续修训练入口路径/参数。 |
| `ca147ea2` | Update run.sh | 更新 `run.sh`。 | 继续收敛训练启动脚本。 |
| `d516c0a8` | Update prepare_env.sh | 更新环境准备脚本。 | 固化依赖安装/环境准备步骤。 |
| `20931b0b` | Update run.sh | 更新 `run.sh`。 | 训练脚本参数继续迭代。 |
| `10f9b92e` | add qwen32b config | 加 VA 环境下 Qwen3-VL short-CoT 脚本。 | 准备更大/不同模型配置的训练实验。 |
| `955887cb` | update gpt | 更新 `gpt_model.py`、`test_api.py`、`vllm.sh`。 | 完善 GPT/Qwen OpenAI-compatible 调用和 vLLM 服务测试。 |
| `2aee1e6f` | update qwen online infer | 加 `online_qwen.sh`，更新在线推理和 reward。 | 给 reward judge / online infer 提供独立调试入口。 |
| `b201acf5` | update config | 更新 online qwen 配置。 | 调整在线推理参数。 |
| `2939c6e0` | fix os bug | 修 online 脚本和 reward 中 OS/路径问题。 | 提升跨环境可运行性。 |
| `7d0312b8` | preprare_env.sh | 加 short_cot 环境准备脚本。 | 将实验环境安装步骤脚本化。 |
| `92b5e612` | fix 2node run | 更新环境准备脚本、run 脚本和 reward。 | 修复 2 节点训练实际启动问题。 |
| `1126e18c` | update model bug | 修 `short_cot_qwen.py` 模型相关 bug。 | 解决 reward judge 模型名/调用细节问题。 |
| `25305e37` | update pre_process | 更新预处理脚本。 | 调整数据/环境预处理。 |
| `26770372` | update preprocess | 加/改环境预处理。 | 支持多套运行环境。 |
| `809772ef` | add auto-save & auto-eval | 加远程 checkpoint IO、`launch_eval_task.py`、Ray trainer auto-save/auto-eval。 | 关键节点：checkpoint 保存后可选提交评测任务。 |
| `1aee7768` | fix config bug | 修 `run.sh` 和 Ray trainer 配置读取。 | 修 auto-save/auto-eval 配置接入问题。 |
| `5e602be3` | Update environment setup | 更新环境脚本。 | 修环境依赖/路径。 |
| `23a7e6c0` | Update short_cot_qwen.py | 更新 Qwen reward。 | 调整 step reward 细节。 |
| `5a0d8b26` | Update environment setup | 更新环境脚本。 | 环境继续固化。 |
| `74020641` | Update ray_trainer.py | 更新 Ray trainer。 | 修训练 loop / auto-eval 逻辑。 |
| `a1d07d3b` | Update H100 config | 更新 `use_gpu.py`。 | 适配 H100 资源申请/使用。 |
| `19ba647f` | Update ray_trainer.py | 更新 Ray trainer。 | 继续修 auto-eval/checkpoint 逻辑。 |
| `b2a84f79` | Update ray_trainer.py | 更新 Ray trainer。 | 继续修训练 loop 中的保存/评测细节。 |
| `18584b0a` | add auto eval | 改 Ray trainer 和 FSDP checkpoint manager。 | 将 auto-eval 更完整地接入 checkpoint 保存流程。 |
| `1d8a1bd7` | fix loader bug | 修 Ray trainer loader。 | 修 checkpoint / model loading 相关 bug。 |
| `dc2920bf` | fix library functions import bug | 修 Ray trainer import。 | 解决运行时函数导入失败。 |
| `1c3b4caa` | add auto-launch log | 给 auto-eval/auto-launch 增加日志。 | 方便定位评测任务有没有被提交。 |
| `58326961` | no detailed log | 调整 reward 日志。 | 减少 reward judge 过多详细日志干扰。 |
| `89698fe4` | fix hdfs_io bug | 修 Ray trainer 和 Megatron checkpoint manager 的 HDFS 逻辑。 | 修 checkpoint 上传/路径问题。 |
| `85ffc1f3` | update auto-eval local path | 修 auto-eval 本地路径。 | 让评测任务能找到本地 HF checkpoint。 |
| `f487ff3a` | fix extra args.json bug | auto-eval HDFS 检查跳过 `args.json`。 | 避免等待非必要文件导致评测卡住。 |
| `843695b8` | fix auto-eval bug | 修 Ray trainer auto-eval bug。 | 继续提高自动评测成功率。 |
| `ccece1a9` | fix auto_eval_config get bug | 修 auto_eval_config 获取。 | 避免 Hydra/OmegaConf 配置读取错误。 |
| `6f2d2a4c` | fix async upload | 修 Megatron checkpoint manager 异步上传。 | 解决异步 checkpoint 上传与 auto-eval 之间的竞态。 |
| `34164a70` | more auto-eval logs | 增加 auto-eval 日志。 | 方便线上排查 checkpoint/HDFS/评测提交状态。 |
| `7d68100e` | Update ray_trainer.py | 更新 Ray trainer。 | 继续修 auto-eval 或保存流程细节。 |
| `4f5fcb72` | more logs | 增加 Ray trainer 日志。 | 继续提高线上可观测性。 |
| `bb313a13` | add auto-eval job | 加 `submit_auto_eval.py`。 | 支持脱离训练主流程，手动补交/重交评测任务。 |
| `76a0008a` | support merge_geo dataset | 更新 Geo3K process convert 和环境脚本。 | 支持合并版 Geo 数据集。 |
| `dfc74a28` | add special token | 加 GeoQA/Geo3K merge 检查脚本和 special token 处理。 | 改善几何数据合并后的 prompt/token 格式。 |
| `395d153e` | update length reward | 加 merged run 脚本、length-bonus 配置，更新 Qwen reward。 | 开始尝试 length-aware reward，控制短 CoT/长 CoT tradeoff。 |
| `415369e2` | same prompt with eval | 更新 dense process convert。 | 让训练 prompt 与评测 prompt 更一致，减少 train/eval mismatch。 |
| `d044786f` | merge_multi_moddal data | 加 Dolci 转换、math solution 提取、多模态 merge/showcase。 | 开始把 text math 和 VL 数据合并成多模态 RL 数据。 |
| `510e0086` | update multi-modal reward | 更新 Qwen reward。 | 让 reward 逻辑适配混合/多模态数据字段。 |
| `c3cb1d2d` | fix max_model_len bug | 修 reward 和 vLLM/SGLang async server max_model_len。 | 解决 reward/rollout server 长上下文参数不一致问题。 |
| `65ee7442` | update use_gpu.py | 更新 GPU 工具。 | 调整资源使用脚本。 |
| `3dcecd07` | update use_gpu.py | 更新 GPU 工具。 | 继续修资源脚本。 |
| `94e24335` | maintain.sh | 加 `run_maintain.sh`。 | 给长任务/环境维护提供脚本。 |
| `da2b0ee7` | update len bouns | 更新 step-bonus 配置和 reward。 | 将 length bonus 与 step bonus 融合到 reward 方案。 |
| `caae0987` | fix bug | 修 Qwen reward。 | 修 length/step reward 实际 bug。 |
| `e424576c` | update vision-r1 convert & step-reward func | 加 `convert_vision_r1.py`，更新 step reward config/函数。 | 关键节点：Vision-R1 数据进入当前 RL/reward pipeline。 |
| `0ee83853` | update reward | 更新 `verl/trainer/ppo/reward.py`。 | 增强 custom reward function 加载/参数绑定能力。 |
| `96aa380f` | merge Vision-R1-DAPO | 更新 Vision-R1 转换并加 `prepare_multimodal_data.py`。 | 把 Vision-R1 和 DAPO 数据合并为主训练数据方向。 |
| `3fe21ad7` | fix reward bug & add VisionR1-DAPO config | 加主配置 `...gpt_step_bonus_visionr1_dapo.yaml` 和 reward 单测。 | 主实验配置成型：VisionR1-DAPO + Qwen step bonus reward。 |
| `bd9f47c` | add prepare config | 更新 VA 环境准备。 | 固化 VisionR1-DAPO 训练环境。 |
| `eec1123b` | step bouns ablation | 加 no-step-bonus ablation yaml。 | 为验证 step bonus 是否有效提供对照组。 |
| `5e9223ef` | update correct judge | 更新 Qwen reward 的 answer correctness judge。 | 修 final answer 判对逻辑，降低误判。 |
| `d935e620` | update dapo reward manager | 更新 DAPO reward manager 与相关配置。 | 将 overlong buffer / reward extra info 更好接入 DAPO manager。 |
| `cbd322b4` | update adaptive_depth | 加 adaptive-depth reward 配置与实现。 | 按数据源/难度自适应鼓励合理推理深度。 |
| `e747710b` | fix reward bug | 修 adaptive-depth reward。 | 修新 reward 分支实际问题。 |
| `4dadb655` | update resume.sh | 更新 resume 和多个 trainer/checkpoint manager。 | 改善断点恢复，对长时间 RL 训练很关键。 |
| `285ea024` | add syd hdfs download | 更新 VA 环境脚本，支持 SYD HDFS 下载。 | 支持跨集群/跨 HDFS 拉取模型或数据。 |
| `63ddcfd1` | update answer-only reward | 加 answer-only reward 配置与实现。 | 新增只看答案的 ablation，对比 process reward 增益。 |
| `7b418db3` | fix yaml bug | 修 answer-only yaml。 | 修 ablation 配置可运行性。 |
| `513aefa3` | add raw-visionr1 sft RL ablation | 加 result-only / process-partial 配置和 unit reward 文件。 | 补齐 raw VisionR1 SFT RL 的 reward ablation 矩阵。 |
| `a83d8a72` | update VL judge | 加多模态 reward judge 文件，并修改 trainer/manager 透传图像字段。 | 关键节点：reward judge 可以看原图，判断视觉 hallucination。 |
| `4b5f7e75` | update VL judge | 加 mmjudge config。 | 将多模态 judge 配置化，方便直接跑实验。 |
| `88f98994` | fix raw visionr1 process reward bug | 修 unit reward。 | 修 raw VisionR1 process reward 分支。 |
| `7c075cf8` | fix key error bug | 修多模态 reward judge key error。 | 让 judge 在缺字段/字段名不一致时更稳。 |
| `95a121c8` | update mathvista convert | 加 MathVista 转换、监控脚本，更新 GPT client。 | 补齐 MathVista 评测/数据转换链路。 |
| `950db8c8` | update reward_model ablation | 加 LogicVista/MathBench/MathVerse 转换、MathVerse prefilter、guided JSON reward config 和测试。 | reward model ablation 与多 benchmark 评测准备基本成型。 |
| `e7588bc8` | fix model_ablation bug | 修 Qwen reward 中 model ablation bug。 | 修 ablation 时 reward model 名称/参数传递问题。 |
| `a3537e69` | update reward model | 加 MathVerse prefilter、reward model size jobs，更新 reward manager 和测试。 | 支持不同 reward model size/版本的对比实验。 |
| `4ca79a02` | bug fix | 加 multimodal sequence balancing 测试，修 Megatron actor/reward model。 | 修多模态 dynamic batching/sequence balancing 中的实际训练 bug。 |

## 2. 按阶段看我们做了什么

### 阶段一：先把 Geo3K + Qwen3-VL GRPO 跑起来

对应提交：`6313f6f0` 到 `7d66e643`。

主要解决：

- parquet 数据读写；
- Geo3K reward 路由；
- Qwen3-VL 8B Megatron baseline 脚本；
- transformers / mbridge / GPU 环境问题；
- Geo3K baseline 初步跑通。

这一步的核心价值：**先证明 Verl 可以带 Qwen3-VL 跑多模态 GRPO**。

### 阶段二：dense CoT 与 process reward

对应提交：`bf184b99` 到 `8f5a3c37`。

主要解决：

- dense-CoT prompt：`<visual>/<think>/<answer>`；
- Geo3K dense-CoT 数据转换；
- process steps 写进 `reward_model.ground_truth`；
- GPT/Qwen judge 作为 process reward；
- 训练脚本从普通 reward 切到 dense/process reward。

这一步的核心价值：**reward 不再只是答案对错，而能给中间过程 credit**。

### 阶段三：多节点训练与 Qwen3-235B reward judge

对应提交：`783c1f54` 到 `92b5e612`。

主要解决：

- 多节点训练脚本；
- Qwen-style reward function；
- 235B reward model 路径；
- online infer / vLLM 服务调试；
- VA/AU 环境准备。

这一步的核心价值：**把 reward judge 和训练扩展到真实大模型/多节点环境**。

### 阶段四：auto-save + auto-eval 工程化

对应提交：`809772ef` 到 `bb313a13`。

主要解决：

- checkpoint 保存后上传 HDFS；
- 等待 HDFS 文件齐全；
- 自动提交 VLMEvalKit job；
- 修异步上传、路径、`args.json`、配置读取等实际问题；
- 加日志和独立 `submit_auto_eval.py`。

这一步的核心价值：**训练不再只是产 checkpoint，而是 checkpoint 产出后自动进入 benchmark 评测**。

### 阶段五：混合数据、length/step reward 与 Vision-R1-DAPO 主线

对应提交：`76a0008a` 到 `e747710b`。

主要解决：

- merge Geo 数据；
- special token / train-eval prompt 对齐；
- Dolci / Vision-R1 / DAPO 数据合并；
- length bonus、step bonus；
- VisionR1-DAPO 主配置；
- no-step-bonus、adaptive-depth 等 ablation。

这一步的核心价值：**主实验从 Geo3K demo 变成 Vision-R1-DAPO 混合数据上的系统实验**。

### 阶段六：resume、answer-only/raw ablation、多模态 judge、评测转换

对应提交：`4dadb655` 到 `4ca79a02`。

主要解决：

- 长任务 resume；
- answer-only / result-only / process-partial ablation；
- 多模态 reward judge 看图；
- MathVista / MathVerse / LogicVista / MathBench 转换；
- guided JSON reward；
- reward model size ablation；
- 多模态 sequence balancing bugfix。

这一步的核心价值：**从“能跑”变成“能系统做 ablation 和评测”**。

## 3. 实践视角下的几个重点经验

### 3.1 真正耗时的是工程闭环，不是 reward 公式

提交历史里 auto-eval 相关 bugfix 非常多：路径、HDFS、异步上传、配置读取、日志、`args.json` 等。这说明 RL 实验平台的关键不只是算法，而是：

- checkpoint 是否稳定保存；
- 保存后评测是否自动触发；
- 失败时是否有日志；
- 能不能补交评测。

### 3.2 Reward 需要 ablation，而不是一次性相信复杂设计

我们不是只加了一个复杂 reward，而是逐步补了：

- answer-only；
- result-only；
- process-partial；
- no-step-bonus；
- step-bonus；
- length-bonus；
- adaptive-depth；
- multimodal judge；
- guided JSON judge。

这样后面可以回答：到底是 final answer、process credit、step bonus、还是看图 judge 在起作用。

### 3.3 多模态 reward 最大的问题是字段透传

多模态 judge 要看图，不只是写 judge prompt，还要保证：

- dataset 保留 `multi_modal_data`；
- reward manager 不把 non-tensor key pop 掉；
- `extra_info` 能拿到原图；
- 缺字段时能 fallback text-only；
- 训练日志里能看到 `judge_used_images`。

这就是 `a83d8a72`、`4b5f7e75`、`7c075cf8` 这几个提交的实践意义。

### 3.4 Train/eval prompt mismatch 会影响结论

`415369e2 same prompt with eval` 专门修了训练 prompt 与评测 prompt 对齐。这个看起来小，但对 RL 很关键：训练奖励鼓励的输出格式必须和评测时模型收到的 prompt 一致，否则可能 reward 学到了但 benchmark 不涨。

## 4. 分享时可以直接讲的版本

如果 weekly 只有 10 分钟，可以按下面讲：

1. **第一步：跑通 baseline**  
   从 Geo3K parquet bug、Geo3K reward config、Qwen3-VL GRPO 脚本开始，把 Qwen3-VL + Verl + Megatron/vLLM 跑起来。

2. **第二步：把 reward 做细**  
   加 dense-CoT prompt、process steps、Qwen3-235B judge，让 reward 能看中间步骤，不只是 final answer。

3. **第三步：把训练工程化**  
   支持多节点、HDFS checkpoint、auto-eval，解决一堆真实训练中的路径、异步上传、配置和日志问题。

4. **第四步：扩数据和主实验**  
   从 Geo3K 扩到 Vision-R1 + DAPO，做 mixed multimodal data，并形成主配置。

5. **第五步：做可解释 ablation**  
   answer-only、result-only、process-partial、no-step-bonus、adaptive-depth、mmjudge、guided JSON、reward model size 都有对应配置。

6. **第六步：补多模态 judge 和评测**  
   reward judge 能看图，评测侧补 MathVista / MathVerse / LogicVista / MathBench 转换，最终可以系统比较。

## 5. 代码入口索引

| 目的 | 文件 |
| --- | --- |
| 主训练入口 | `launch.sh`, `run.sh`, `entry.sh` |
| Qwen3-VL Geo3K GRPO | `examples/grpo_trainer/run_qwen3_vl-8b-megatron_dense_cot_with_process.sh` |
| VisionR1-DAPO 主配置 | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo.yaml` |
| no-step-bonus ablation | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_visionr1_dapo_no_step_bonus.yaml` |
| answer-only ablation | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_format_answer_only_visionr1_dapo.yaml` |
| result/process ablation | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_result_only_visionr1_dapo.yaml`, `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_partial_1p0_visionr1_dapo.yaml` |
| adaptive-depth reward | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_adaptive_depth_visionr1_dapo.yaml`, `verl/utils/reward_score/short_cot_qwen_adaptive_depth.py` |
| 多模态 judge | `verl/trainer/config/ppo_megatron_trainer_dense_cot_qwen3vl_process_reward_qwen_gpt_step_bonus_mmjudge_visionr1_dapo.yaml`, `verl/utils/reward_score/short_cot_qwen_gpt_step_bonus_multimodal.py` |
| 主 reward | `verl/utils/reward_score/short_cot_qwen.py` |
| reward loader/async | `verl/trainer/ppo/reward.py` |
| DAPO reward manager | `verl/workers/reward_manager/dapo.py` |
| Geo3K dense/process 转换 | `examples/grpo_trainer/convert_dense_cot_with_process.py` |
| Vision-R1/DAPO 转换 | `convert_vision_r1.py`, `convert_vision_r1_v2.py`, `prepare_multimodal_data.py` |
| auto-eval | `verl/trainer/ppo/launch_eval_task.py`, `submit_auto_eval.py` |
| 评测转换 | `convert_mathvista.py`, `convert_mathverse.py`, `convert_logicvista.py`, `convert_mathbench_vqa.py` |
| 单测 | `tests/utils/reward_score/test_short_cot_qwen.py`, `tests/utils/test_seqlen_balancing.py` |
