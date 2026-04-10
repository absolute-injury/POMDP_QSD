# Phase II — Posterior Routing Analysis

## Goal

After one Trine measurement, the observer's belief is updated via Bayes' rule to a **posterior** that depends on the prior, the chosen angle α\*, and the outcome received. Phase II traces these posterior trajectories — asking *where* in the belief simplex each branch lands, and whether the structure of these routes has practical implications.

## Method

Starting from five representative prior beliefs (labelled A–E, spanning the simplex), Phase II:

1. Loads the Phase I solution (`one_step_maps.npz`) to obtain α\* at each prior.
2. Computes the three posterior beliefs (one per outcome) using Bayes' rule.
3. Records which region of the simplex each posterior falls in, whether it is "near-certain", and how the routes differ across cases.

The five representative cases cover the main regimes of interest:

| Case | Role |
|------|------|
| A | Centre of simplex — maximum symmetry |
| B | Edge-adjacent quasi-binary prior |
| C | Near-certainty, vertex-adjacent |
| D | Generic asymmetric interior point |
| E | Off-centre interior |

## Code

| File | Role |
|------|------|
| `code/src/phase2.py` | Posterior computation, route classification, summary statistics |
| `code/src/phase2_plotting.py` | Routing diagrams, Point E-focused diagnostics, per-case detail plots |
| `code/scripts/run_phase2_posterior_routing.py` | End-to-end driver: load Phase I → route → save → plot |

## Results

```
results/
├── data/
│   ├── phase2_branch_routes.csv   # per-branch route table (prior → posterior)
│   ├── phase2_routing_raw.json    # full routing data including diagnostics
│   └── phase2_summary.csv        # per-case summary statistics
├── figures/
│   ├── figure_D_phase2_posterior_routing.png/.pdf   # main routing overview
│   ├── figure_D2_phase2_diagnostics.png/.pdf        # consistency + Point E diagnostics
│   └── case_details/
│       ├── figure_D_case_A_routing_detail.png/.pdf
│       ├── figure_D_case_B_routing_detail.png/.pdf
│       ├── figure_D_case_C_routing_detail.png/.pdf
│       ├── figure_D_case_D_routing_detail.png/.pdf
│       └── figure_D_case_E_routing_detail.png/.pdf
└── logs/
    ├── phase2_checks.json               # automated validation checks
    └── phase2_interpretation_notes.txt  # annotated observations
```

## Key findings

- Posterior branches from near-certain priors (Case C) collapse tightly toward the nearest vertex, confirming rapid certainty after one measurement.
- Symmetric priors (Case A) produce three equiprobable branches related by cyclic permutation, consistent with the Trine symmetry.
- Asymmetric priors (Cases D, E) produce notably unequal branch weights, motivating the adaptive two-step strategy explored in Phase III.
