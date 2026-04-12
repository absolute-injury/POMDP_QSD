# Phase I — One-Step Optimal Angle Map

## Goal

Solve the single-step Trine QSD problem on a belief simplex grid:

- input belief: `b=(b1,b2,b3)`
- action: choose one measurement angle `alpha in [0, 2pi/3)`
- objective: maximize one-step success probability `J1*(b)`

This phase also computes `G(b)=J1*(b)-S(b)` with `S(b)=max(b)` to quantify the benefit of measuring vs stopping.

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
```

## Main Runner

Recommended run (writes directly to `phase1_one_step/results/`):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase1_one_step/code/scripts/run_one_step.py \
  --N 40 \
  --M-alpha 120 \
  --outdir phase1_one_step \
  --tag results
```

## Default Configuration

- Belief resolution: `N=40`
- Angle library size: `M_alpha=120`
- Batch size: `512`
- Tie tolerance: `1e-10`

Useful options:

- `--N <int>`: belief lattice resolution
- `--M-alpha <int>`: number of alpha samples
- `--batch-size <int>`: solver batch size
- `--tie-tol <float>`: near-tie threshold

## Main Outputs

`phase1_one_step/results/` contains:

- `data/one_step_maps.npz`
- `data/one_step_maps.csv`
- `figures/figure_A_j1_star.png`
- `figures/figure_B_gain.png`
- `figures/figure_C_alpha_star.png`
- `logs/sanity_checks.json`

## Plot Interpretation Cheatsheet

- `figure_A_j1_star`: belief별 한 번 측정 최적값 지도. 중앙부가 높으면 불확실 구간에서 측정 이득이 큼.
- `figure_B_gain`: `J1*-S` 지도로, 측정할 가치가 있는 영역을 바로 보여줌.
- `figure_C_alpha_star`: 최적 alpha 선택 패턴. 부드러운 영역 + near-tie 경계의 공존 여부를 본다.

## Downstream Note

Phase II/III/IV scripts can reuse:

- `phase1_one_step/results/data/one_step_maps.npz`
