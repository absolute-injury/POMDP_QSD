# Phase III — Two-Step Sequential Bellman Solver

## Goal

Solve horizon-2 sequential control (`H=2`) on the Trine belief grid.

- Stage 2 terminal value: `V2(b)=max(b)`
- Stage 1 decision: stop vs measure -> `V1`, `D1=V1-S`
- Stage 0 decision: stop vs measure -> `V0`, `D0=V0-V1`

This phase quantifies when a second measurement is worth its cost.

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib pillow imageio imageio-ffmpeg
```

Phase I artifact is required:

- `phase1_one_step/results/data/one_step_maps.npz`

## Main Runner

Recommended run (writes directly to `phase3_sequential/results/`):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase3_sequential/code/scripts/run_phase3_sequential.py \
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
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase3_sequential/code/scripts/make_phase3_cost_gifs.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential/results/figures/animations
```

Animation outputs include:

- `phase3_cost_sweep_panel_2x2.mp4`
- `phase3_cost_sweep_[V0|V1|D0|D1].mp4`
- `phase3_cost_sweep_action_[V0|V1].mp4`
- `phase3_cost_sweep_delta_alpha_idx.mp4`

## Plot Interpretation Cheatsheet

- `phase3_fig_V1_*`: 1단계에서 측정 여부를 포함한 가치 구조.
- `phase3_fig_V0_*`: 0단계에서 선행 측정의 총 기대가치.
- `phase3_fig_D1_*`, `phase3_fig_D0_*`: `>0` 영역이 continuation 영역.
- `phase3_fig_action_*`: 측정을 선택한 지점에서 최적 alpha 패턴.
- `phase3_fig_delta_alpha_idx_*`: 0단계와 1단계 alpha 차이로 적응성 강도를 본다.

## Notes

This phase is the computational baseline that Phase IV-B/D stress-tests under discretization refinements.
