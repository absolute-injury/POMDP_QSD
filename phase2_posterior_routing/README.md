# Phase II — Posterior Routing Analysis

## Goal

Trace posterior branches after one measurement and quantify how routing differs by representative priors.

- start from Phase I optimal policy (`alpha*`)
- apply Bayes update for each outcome branch
- compare posterior landing regions, branch probabilities, and route asymmetry

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Tested environment: Python `3.12.7` on macOS; in headless setups, use `MPLCONFIGDIR=/tmp/mpl`.

Phase I artifact is required:

- `phase1_one_step/results/data/one_step_maps.npz`

## Main Runner

Recommended run (writes directly to `phase2_posterior_routing/results/`):

```bash
MPLCONFIGDIR=/tmp/mpl python phase2_posterior_routing/code/scripts/run_phase2_posterior_routing.py \
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
- `--debug-figure-metadata` (optional internal overlays; off by default)

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

## How to Read the Routing Figures

`figure_D_phase2_posterior_routing` presents five representative priors (Cases A-E) on the simplex and shows posterior routing under the Phase I decision rule `alpha*(b)`. Broadly speaking, the figure is intended to summarize routing geometry and branch balance across qualitatively different prior regions.

The representative cases are defined from fixed target priors (center/symmetry, edge-adjacent quasi-binary, near-certainty, generic interior, and off-center interior) and then snapped to the nearest valid grid point under the selected metric ($L_\infty$ by default). For Point E, near-switching validity is checked through the argmax-gap criterion; if the target does not satisfy the threshold, the smallest-gap backup candidate is selected.

Within each panel, the star marks the starting belief, arrows indicate outcome-conditioned Bayesian updates, and terminal markers show posterior beliefs. Labels `p(o1)`, `p(o2)`, and `p(o3)` report branch probabilities. Arrow direction and endpoint placement should be read as routing structure, while exact quantitative detail should be taken from the accompanying tables and CSV outputs.

`figure_D2_phase2_diagnostics` reports normalization and consistency checks, including near-tie diagnostics for Point E. `figures/case_details/*` provides enlarged per-case views that make branch geometry and posterior coordinates easier to be inspected.

The routing visualization is illustrative and diagnostic; it should not be over-interpreted as a standalone statistical claim.

## Notes

This phase is descriptive/diagnostic. It motivates the adaptive sequential policy solved in Phase III.
