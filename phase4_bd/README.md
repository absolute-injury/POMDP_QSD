# Phase IV-B / IV-D — Discretization Robustness + Theorem-Verifying Synthesis

## Goal

Implement the narrowed Phase IV scope:

- **IV-B**: robustness under belief/action discretization refinement
- **IV-D**: theorem-verifying numerical synthesis

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib pillow imageio imageio-ffmpeg
```

## Main Runner

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase4_bd/code/scripts/run_phase4_bd.py --outdir phase4_bd/results_full
```

Default matrix and policy:

- Baseline: `N=80, M_alpha=240`
- Belief sweep: `40, 80, 120`
- Action sweep: `120, 240, 360`
- Shared cost: `c_meas=0.02`
- Axis A: excluded (already completed)
- Axis C: excluded unless instability is detected

Useful options:

- `--c-meas <float>`: lock shared measurement cost
- `--belief-resolutions <csv>`: override B-1 sweep
- `--action-resolutions <csv>`: override B-2 sweep
- `--skip-figures`: compute diagnostics without figure generation

## Main Outputs

`phase4_bd/results_full/` contains:

- `data/phase4B_values_[config].npz`
- `diagnostics/phase4B_diag_[config].json`
- `data/phase4B_compare_summary.csv`
- `data/phase4_rep_point_audit.csv`
- `diagnostics/phase4D_summary.md`
- `figures/phase4B_fig_compare_[group].png/.pdf`

## Optional B-4 Focused Rerun

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase4_bd/code/scripts/run_phase4_focused_rerun.py
```

Focused outputs are stored under `results_full/focused/` by default.

Focused outputs:

- `focused/data/phase4B_focused_points.csv`
- `focused/diagnostics/phase4B_focused_summary.md`
- `focused/figures/phase4B_focused_map_D0_error.png`
- `focused/figures/phase4B_focused_sorted_error.png`

## Plot Interpretation Cheatsheet

- `phase4B_fig_compare_values`: Comparison of absolute deviations in V1/V0/D1/D0 relative to the baseline. Minimal deviations without configuration-specific spikes indicate high structural stability.
- `phase4B_fig_compare_policy`: Comparison of decision/alpha disagreement ratios. Low decision disagreement is interpreted as to be stable within the policy regime.
- `phase4B_fig_compare_region`: Comparison of `D1 > 0`, `D0 > 0`, and continuation fractions. The absence of significant fluctuations suggests the preservation of continuation topology.
- `phase4B_focused_map_D0_error`: Map the distribution of top-k suspicious boundary points on the simplex and identify regions with high baseline D0 exact errors.
- `phase4B_focused_sorted_error`: Comparative analysis of local exact error curves across baseline, coarse, and refined grids. Consistently lower errors in the refined grid suggest that prior deviations were likely due to the coarse-grid effect.
