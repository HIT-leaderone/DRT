# Teaser Figure Package

Main output:

- `assets/showcase_tradeoff_combined_v3_integrated.pdf`
- `assets/showcase_tradeoff_combined_v3_integrated.png`
- `assets/showcase_tradeoff_combined_v3_integrated.svg`

Primary script:

- `scripts/plot_combined_teaser_v3.py`

Rebuild from anywhere:

```bash
python3 Figures/teaser/scripts/plot_combined_teaser_v3.py
```

The current integrated teaser is self-contained in this folder. Its right
panel is generated in memory by `scripts/plot_drt_token_efficiency_v3.py`
from the local `main_result.tex`; it does not read root-level `assets/`,
`example/`, or `scripts/` files.

Important supporting files:

- `scripts/main_result_parser.py`
- `scripts/plot_mathverse_showcase_v3.py`
- `scripts/plot_drt_token_efficiency_v3.py`
- `main_result.tex`
- `example/mathverse_3527_surface_area_inferred_x_image.png`
- `mathverse_asset_pack/assets/semantic_blocks/drt_{observe,think,answer}_box.png`
- `assets/fonts/`
