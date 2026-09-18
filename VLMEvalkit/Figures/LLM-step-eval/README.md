# LLM Step Evaluation Figure

Main output:

- `assets/combined_step_bonus_grouped_by_metric_with_side_panels_gt3to10.pdf`
- `assets/combined_step_bonus_grouped_by_metric_with_side_panels_gt3to10.png`
- `assets/combined_step_bonus_grouped_by_metric_with_side_panels_gt3to10.svg`

Primary script:

- `analyze_combined_step_bonus.py`

Related variant:

- `analyze_combined_step_bonus_with_visual_hallucination.py`
- `assets/combined_step_bonus_with_visual_hallucination_grouped_by_metric_with_side_panels_gt3to10.*`

Rebuild from this folder:

```bash
cd Figures/LLM-step-eval
VERL_QWEN3VL_ROOT=/path/to/VERL_Qwen3VL \
python3 analyze_combined_step_bonus.py --output_dir assets --backup_dir assets
```

Notes:

- The local `assets/` directory contains the generated plots and step-level CSV inputs.
- The copied `assets/fonts/` directory is used by the plotting code.
