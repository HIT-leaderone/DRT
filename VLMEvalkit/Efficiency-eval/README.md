# Efficiency Eval

This folder contains the QPS / latency evaluation workflow for VLMEvalKit.

## Active entry points

- `serve_vllm_qwen3vl.sh`: start a local vLLM server for Qwen3-VL checkpoints.
- `eval_vllm_qwen3vl_time.py`: run vLLM TimeEval for standard prompt settings.
- `eval_vllm_qwen3vl_thinkless.py`: run the two-stage Thinkless vLLM TimeEval.
- `eval_transformers_qwen3vl_thinkless.py`: run local Transformers-based Thinkless TimeEval.
- `eval_visionthink_time.py`: run VisionThink TimeEval while reusing `reproduce/visionthink`.
- `summarize_time_eval_results.py`: collect TimeEval xlsx summaries into CSV and LaTeX outputs.
- `run_local_time_eval_grid.sh`: local helper for running prompt/setting grids against an existing vLLM server.

## Supporting files

- `time_eval_prompt_utils.py`: shared prompt/message conversion helpers.
- `results/`: generated summary tables and local run outputs.
- `docs/qps_latency_original_notes.txt`: original request/command notes, updated to point at the reorganized paths.
- `legacy/`: older vLLM evaluation prototypes kept for reference.

The public release keeps only local, portable entry points. Private cluster
submission templates and generated job logs are intentionally omitted.
