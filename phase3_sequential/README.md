# Phase III — Two-Step Sequential Bellman Solver

## Goal

Solve horizon-2 sequential control (`H=2`) on the Trine belief grid.

- Stage 2 terminal value: `V2(b)=max(b)`
- Stage 1 decision: stop vs measure -> `V1`, `D1=V1-S`
- Stage 0 decision: stop vs measure -> `V0`, `D0=V0-V1`

This phase will also quantify when a second measurement is worth its cost.

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[media]"
```

Tested environment: Python `3.12.7` on macOS; in headless setups, use `MPLCONFIGDIR=/tmp/mpl`.

Phase I artifact is required:

- `phase1_one_step/results/data/one_step_maps.npz`

## Main Runner

Recommended run (writes directly to `phase3_sequential/results/`):

```bash
MPLCONFIGDIR=/tmp/mpl python phase3_sequential/code/scripts/run_phase3_sequential.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential \
  --tag results
```

Default policy:

- Baseline run: `c0=0.0`
- Sensitivity run: `c_eps=0.02`
- `run_eps` is enabled unless `--skip-eps`

Useful options:

- `--c0 <float>` / `--c-eps <float>`
- `--skip-eps`
- `--decision-tol <float>`
- `--plot-clip-tol <float>`

## Main Outputs

`phase3_sequential/results/` contains:

- `phase3_values_run0.npz`
- `phase3_diag_run0.json`
- `phase3_values_run_eps.npz` (if not skipped)
- `phase3_diag_run_eps.json` (if not skipped)
- `phase3_summary.md`
- `phase3_fig_[V1|V0|D1|D0]_[run0|run_eps].png/.pdf`
- `phase3_fig_[action_V1|action_V0|delta_alpha_idx]_[run0|run_eps].png/.pdf`

## Optional Cost-Sweep Animations

```bash
MPLCONFIGDIR=/tmp/mpl python phase3_sequential/code/scripts/make_phase3_cost_gifs.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential/results/figures/animations
```

Animation outputs include:

- `phase3_cost_sweep_panel_2x2.mp4`
- `phase3_cost_sweep_[V0|V1|D0|D1].mp4`
- `phase3_cost_sweep_action_[V0|V1].mp4`
- `phase3_cost_sweep_delta_alpha_idx.mp4`

## Plot Interpretation Cheatsheet

- `phase3_fig_V1_*`: Value structure at Step 1, incorporating measurement decisions.
- `phase3_fig_V0_*`: Total expected value of prior measurements at Step 0.
- `phase3_fig_D1_*`, `phase3_fig_D0_*`: The `>0` region indicates the continuation region.
- `phase3_fig_action_*`: Optimal alpha patterns at points where measurement is selected.
- `phase3_fig_delta_alpha_idx_*`: Analyzing adaptive intensity via the difference in alpha indices between Step 0 and Step 1.

## Notes

This phase is the computational baseline that Phase IV-B/D stress-tests under discretization refinements.
