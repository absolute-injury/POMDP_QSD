# Phase IV-B / IV-D — Discretization Robustness + Theorem-Facing Synthesis

## Goal

Implement the narrowed Phase IV scope:

- **IV-B**: robustness under belief/action discretization refinement
- **IV-D**: theorem-facing numerical synthesis (proof tone explicitly avoided)

Axis A (cost sensitivity) is treated as complete, and Axis C (projection-rule comparison) is excluded by default.

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

- `phase4B_fig_compare_values`: baseline 대비 `V1/V0/D1/D0` 절대 차이 크기 비교. 값이 작고 특정 config만 튀지 않으면 구조 안정성이 높다.
- `phase4B_fig_compare_policy`: decision disagreement / alpha disagreement 비율 비교. decision disagreement가 낮으면 정책 영역은 안정적이라고 해석한다.
- `phase4B_fig_compare_region`: `D1>0`, `D0>0`, continuation fraction 비교. 큰 폭 변화가 없으면 continuation topology 유지로 본다.
- `phase4B_focused_map_D0_error`: 의심 경계점(top-k)이 simplex 어디에 모이는지, 그리고 baseline D0 exact 오차가 어디서 큰지 확인.
- `phase4B_focused_sorted_error`: baseline/coarse/refined의 local exact 오차 곡선 비교. refined가 일관되게 낮으면 coarse artifact 가능성이 높다.
