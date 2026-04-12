# Phase II — Posterior Routing Analysis

## Goal

Trace posterior branches after one measurement and quantify how routing differs by representative priors.

- start from Phase I optimal policy (`alpha*`)
- apply Bayes update for each outcome branch
- compare posterior landing regions, branch probabilities, and route asymmetry

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib
```

Phase I artifact is required:

- `phase1_one_step/results/data/one_step_maps.npz`

## Main Runner

Recommended run (writes directly to `phase2_posterior_routing/results/`):

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase2_posterior_routing/code/scripts/run_phase2_posterior_routing.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase2_posterior_routing \
  --tag results
```

Default policy for representative beliefs:

- Cases: `A, B, C, D, E`
- Snap metric: `linf`
- Near-tie gap threshold: `1e-6`

Useful options:

- `--snap-metric {linf,l1,l2}`
- `--near-tie-gap <float>`
- `--prob-tol <float>`
- `--posterior-tol <float>`

## Main Outputs

`phase2_posterior_routing/results/` contains:

- `data/phase2_routing_raw.json`
- `data/phase2_summary.csv`
- `data/phase2_branch_routes.csv`
- `figures/figure_D_phase2_posterior_routing.png/.pdf`
- `figures/figure_D2_phase2_diagnostics.png/.pdf`
- `figures/case_details/figure_D_case_[A-E]_routing_detail.png/.pdf`
- `logs/phase2_checks.json`
- `logs/phase2_interpretation_notes.txt`

## Plot Interpretation Cheatsheet

- `figure_D_phase2_posterior_routing`: Observe the movement of outcome branches on the simplex for each case.
- `figure_D2_phase2_diagnostics`: Simultaneously examine normalization/consistency checks and whether it lies in near-tie sensitive regions.
- `case_details/*`: Individually magnify A–E to examine branch probability imbalance and the resulting endpoint structures.

## Notes

This phase is descriptive/diagnostic. It motivates the adaptive sequential policy solved in Phase III.
