# Figure Packages

This directory groups figure-generation code with the assets needed to
rebuild or inspect each figure family. Root-level plotting scripts and
duplicated figure assets were cleaned after these self-contained packages were
created.

## Packages

- `LLM-step-eval/`
  - Main figure: `assets/combined_step_bonus_grouped_by_metric_with_side_panels_gt3to10.pdf`
  - Main script: `analyze_combined_step_bonus.py`
  - Also includes the visual-hallucination variant and step-eval caches/CSVs used by the figure.

- `teaser/`
  - Main figure: `assets/showcase_tradeoff_combined_v3_integrated.pdf`
  - Main script: `scripts/plot_combined_teaser_v3.py`
  - Contains only the current integrated teaser pipeline and the assets needed to rebuild it.

- `main-showcase.zip`
  - Final packaged code/assets for the main-result showcase figure.

- `method-diagrams/`
  - Main figures:
    - `assets/drt_method_diagram.pdf`
    - `assets/drt_reward_design_diagram.pdf`
  - Main scripts:
    - `scripts/build_drt_method_diagram.py`
    - `scripts/build_drt_reward_design_diagram.py`
