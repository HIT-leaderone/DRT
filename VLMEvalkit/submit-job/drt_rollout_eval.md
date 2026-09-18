# DRT@k rollout eval

This repo supports two paper-facing DRT@k settings through environment
variables consumed by `Qwen3VLChat`:

- `DRT_AGGREGATION=sc`: DRT-SC@k. Generate `DRT_ROLLOUT_N=k`
  independent DRT responses, extract final answers, and choose the plurality
  answer. Ties are resolved by earliest rollout. No extra selector tokens are
  used.
- `DRT_AGGREGATION=bon`: DRT-BoN@k. Generate `k` responses, then ask the same
  DRT model to select the best candidate. Selector tokens are counted in
  `output_length_tokens`.

Compact metadata is written as extra result columns, including
`drt_candidate_answers`, `drt_vote_counts`, `drt_candidate_output_tokens`,
`drt_selected_index`, and `output_length_tokens`. Full rollout text storage is
off by default; enable with `DRT_STORE_ROLLOUTS=1` or `--drt-store-rollouts` if
you need audit traces.

## Paper settings

Public releases should run these settings through the portable VLMEvalKit
entry points, not private cluster submit templates. The paper settings expand
to model suffixes:

- `-drt-sc2`, `-drt-sc4`, `-drt-sc8`
- `-drt-bon2`, `-drt-bon4`, `-drt-bon8`
