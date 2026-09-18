# Re-eval results vs paper original

- Current suffix: `-re-eval-gpt4o0513`
- Current scores are VLMEvalKit score artifacts from the paper reproduction run. MathVerse uses mean of the five `Overall` split rows. GSM8K/Video-Holmes JSON ratios are converted to percent.
- Diffs are `current - paper original` using rounded Acc values extracted from `Efficient_Thinking_NIPS2026 (2).pdf`; expect ±0.05 rounding noise.

- Completed score cells: `80/80`
- Missing: none

| Paper table | Group | Label | MathVista cur(Δ) | MathVerse cur(Δ) | LogicVista cur(Δ) | GSM8K cur(Δ) | Video-Holmes cur(Δ) | AVG cur(Δ) | max|Δ| |
|---|---|---|---|---|---|---|---|---|---|
| Table 1 main | `qwen_da` | Qwen-DA | 66.6 (+0.1) | 43.2 (+0.7) | 38.9 (-2.7) | 25.6 (+0.0) | 43.2 (+0.4) | 43.5 (-0.3) | 2.7 |
| Table 1 main | `qwen_drt` | Qwen-DRT | 73.3 (-1.6) | 60.5 (+0.0) | 52.3 (-0.1) | 75.2 (+0.0) | 43.0 (-0.4) | 60.9 (-0.4) | 1.6 |
| Table 1 main | `qwen_standard` | Qwen-Standard | 76.2 (-1.0) | 54.8 (-7.3) | 57.5 (+2.2) | 95.2 (+0.0) | 39.9 (+0.3) | 64.7 (-1.2) | 7.3 |
| Table 1 main | `qwen_thinking_vllm0191` | Qwen-Thinking | 79.7 (-0.2) | 74.2 (+0.0) | 64.2 (-4.0) | 95.5 (-0.6) | 44.3 (+1.8) | 71.6 (-0.6) | 4.0 |
| Table 1 main | `cod` | CoD | 73.2 (+1.2) | 57.8 (+0.2) | 52.3 (+2.0) | 80.9 (+0.2) | 43.0 (+0.8) | 61.4 (+0.9) | 2.0 |
| Table 1 main | `drt_sft_800` | DRT-SFT | 71.5 (+0.0) | 56.2 (-0.0) | 46.8 (-0.4) | 90.6 (-0.4) | 42.8 (-0.0) | 61.6 (-0.2) | 0.4 |
| Table 1 main | `drt_rl_170` | DRT-RL | 76.7 (-0.1) | 64.3 (+0.0) | 55.9 (-0.9) | 94.3 (+0.0) | 43.7 (-0.0) | 67.0 (-0.2) | 0.9 |
| Table 2 ablation | `paper_t2_raw_answer_only_170` | VisionR1-cold \| Image+Text \| Rans | 72.1 (+0.1) | 65.1 (+0.7) | 57.0 (-0.3) | 95.1 (-0.0) | 40.0 (+1.4) | 65.9 (+0.4) | 1.4 |
| Table 2 ablation | `paper_t2_raw_process_reward_retry_80` | VisionR1-cold \| Image+Text \| Rformat+Rmain+Rbonus | 72.3 (+0.0) | 64.2 (+0.8) | 57.9 (+0.2) | 95.5 (-0.0) | 41.4 (-0.0) | 66.3 (+0.2) | 0.8 |
| Table 2 ablation | `paper_t2_vl_judge_170` | DRT-SFT \| Image-only \| Rformat+Rmain+Rbonus | 75.3 (-0.2) | 65.0 (+0.8) | 55.3 (+1.6) | 94.2 (+0.6) | 44.1 (+1.0) | 66.8 (+0.8) | 1.6 |
| Table 2 ablation | `paper_t2_answer_only_100` | DRT-SFT \| Image+Text \| Rformat+Rans | 74.3 (-0.1) | 62.1 (+0.1) | 50.3 (-0.5) | 92.5 (-0.2) | 42.5 (-0.0) | 64.3 (-0.1) | 0.5 |
| Table 2 ablation | `paper_t2_no_step_170` | DRT-SFT \| Image+Text \| Rformat+Rmain | 74.3 (-0.2) | 63.3 (+0.2) | 53.9 (-0.2) | 94.1 (-0.0) | 44.7 (-0.1) | 66.1 (-0.0) | 0.2 |
| Table 4 reward model size | `paper_t4_rm_qwen3_8b_190` | Qwen3-8B | 72.0 (+0.1) | 59.1 (+0.5) | 54.6 (+1.4) | 92.5 (+0.4) | 41.6 (+0.0) | 64.0 (+0.5) | 1.4 |
| Table 4 reward model size | `paper_t4_rm_qwen3_14b_170` | Qwen3-14B | 74.6 (+0.0) | 63.7 (+0.7) | 56.8 (+0.0) | 92.3 (+0.0) | 44.2 (+0.0) | 66.3 (+0.1) | 0.7 |
| Table 4 reward model size | `paper_t4_rm_qwen3_32b_170` | Qwen3-32B | 74.8 (-0.1) | 64.2 (+0.1) | 55.0 (-0.7) | 94.2 (+0.0) | 43.9 (+0.6) | 66.4 (+0.0) | 0.7 |
| Table 4 reward model size | `paper_t4_rm_qwen3_30b_a3b_170` | Qwen3-30B-A3B | 75.9 (+0.0) | 64.6 (+0.5) | 55.0 (+0.2) | 93.9 (+0.0) | 45.0 (+0.0) | 66.9 (+0.2) | 0.5 |

## Notable diffs (|Δ| >= 1.0)
| Label | Bench | Current | Paper | Δ |
|---|---|---:|---:|---:|
| Qwen-Standard | MathVerse | 54.8 | 62.1 | -7.3 |
| Qwen-Thinking | LogicVista | 64.2 | 68.2 | -4.0 |
| Qwen-DA | LogicVista | 38.9 | 41.6 | -2.7 |
| Qwen-Standard | LogicVista | 57.5 | 55.3 | +2.2 |
| CoD | LogicVista | 52.3 | 50.3 | +2.0 |
| Qwen-Thinking | Video-Holmes | 44.3 | 42.5 | +1.8 |
| Qwen-DRT | MathVista | 73.3 | 74.9 | -1.6 |
| DRT-SFT \| Image-only \| Rformat+Rmain+Rbonus | LogicVista | 55.3 | 53.7 | +1.6 |
| VisionR1-cold \| Image+Text \| Rans | Video-Holmes | 40.0 | 38.6 | +1.4 |
| Qwen3-8B | LogicVista | 54.6 | 53.2 | +1.4 |
| CoD | MathVista | 73.2 | 72.0 | +1.2 |
| DRT-SFT \| Image-only \| Rformat+Rmain+Rbonus | Video-Holmes | 44.1 | 43.1 | +1.0 |
| Qwen-Standard | MathVista | 76.2 | 77.2 | -1.0 |