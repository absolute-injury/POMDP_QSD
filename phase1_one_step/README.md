# Phase I — One-Step Optimal Angle Map

## Goal

Given a prior belief vector **b = (b₀, b₁, b₂)** over the three Trine states, find the single measurement rotation angle **α ∈ [0, 2π/3)** that maximises the expected probability of correct identification in one shot.

## Method

The belief simplex is discretised into a triangular lattice (`BeliefGrid`). For each grid point the one-step value function

```
J₁(b, α) = Σ_k  max_j  P(outcome k | state j, α) · bⱼ
```

is evaluated over a fine grid of angles. The maximising angle **α\*** and the resulting value **J₁\*(b)** are stored for every belief point.

The **gain** surface `gain = J₁* − max(b)` quantifies the benefit of measuring versus stopping immediately (guessing the most probable state).

## Code

| File | Role |
|------|------|
| `code/src/core.py` | Belief grid, likelihood table, one-step value function |
| `code/src/solver.py` | Batched optimisation over α; sanity checks (symmetry, vertex gain) |
| `code/src/plotting.py` | Heatmap/contour utilities for the simplex |
| `code/scripts/run_one_step.py` | End-to-end driver: solve → save → plot → log |

## Results

```
results/
├── data/
│   ├── one_step_maps.npz   # full arrays: beliefs, j1_star, gain, best_alpha, …
│   └── one_step_maps.csv   # flat table for quick inspection
├── figures/
│   ├── figure_A_j1_star.png    # optimal value surface over the simplex
│   ├── figure_B_gain.png       # gain surface (benefit of measuring)
│   └── figure_C_alpha_star.png # optimal angle map
└── logs/
    └── sanity_checks.json  # symmetry and vertex-gain validation
```

## Key findings

- The gain is zero at the three vertices (certain beliefs) and maximised near the centre of the simplex, reflecting that measurement is most useful when uncertainty is highest.
- The value map respects the 3-fold cyclic symmetry of the Trine geometry, confirmed numerically by `sanity_checks.json`.
- The optimal angle α\* varies smoothly across most of the interior but shows degeneracy regions where multiple angles achieve the same value.
