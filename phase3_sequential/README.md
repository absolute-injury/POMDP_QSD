# Phase III — Two-Step Sequential Bellman Solver

## Goal

Extend the one-step analysis to a **horizon H = 2** sequential decision problem. At each stage the observer may either **stop and guess** or **measure** (paying a cost c_meas). The optimal policy is found by backward induction (Bellman recursion), yielding value functions and decision maps for both stages.

## Method

### Backward induction

**Stage 2 (terminal):** `V2(b) = max(b)` — the observer stops and picks the most probable state.

**Stage 1:** For each belief **b** and each candidate angle **α**, compute the expected value of reaching Stage 2 across all three outcomes, then subtract the measurement cost:

```
Q1(b, α) = c_meas · J1(b, α)  +  (1 − c_meas) · Σ_k P(k|b,α) · V2(posterior(b,α,k))
```

`V1(b) = max(stop, max_α Q1(b, α))` and the decision map `D1(b)` flags whether measuring is optimal at Stage 1.

**Stage 0:** Analogously, `V0` and `D0` are computed by rolling one step further back, with `Q0` referencing `V1`.

Two runs are produced:
- **run0** — baseline cost `c_meas = 0` (measurement is free)
- **run_eps** — small positive cost `c_meas = ε` (slight penalty for measuring)

### Cost sweep animations

The scripts additionally sweep `c_meas` from 0 to its maximum useful value, animating how the value surfaces and decision boundaries evolve.

## Code

| File | Role |
|------|------|
| `code/src/phase3.py` | Transition cache, Bellman solver, `Phase3Run` dataclass |
| `code/src/phase3_plotting.py` | Static simplex figures for V0, V1, D0, D1, Δα |
| `code/scripts/run_phase3_sequential.py` | Driver: build cache → solve both runs → save → plot |
| `code/scripts/make_phase3_cost_gifs.py` | Cost-sweep driver: animate value/decision maps over c_meas |

## Results

```
results/
├── data/
│   ├── phase3_values_run0.npz      # V0, V1, D0, D1, alpha maps for run0
│   └── phase3_values_run_eps.npz   # same for run_eps
├── diagnostics/
│   ├── phase3_diag_run0.json       # solver diagnostics for run0
│   ├── phase3_diag_run_eps.json    # solver diagnostics for run_eps
│   └── phase3_summary.md           # human-readable summary and rerun guide
└── figures/
    ├── static/                     # 14 PNG figures (V0, V1, D0, D1, action, Δα × 2 runs)
    └── animations/                 # 8 MP4 animations (cost sweep for each quantity)
        ├── phase3_cost_sweep_panel_2x2.mp4   # main 2×2 interpretive panel
        ├── phase3_cost_sweep_V0.mp4
        ├── phase3_cost_sweep_V1.mp4
        ├── phase3_cost_sweep_D0.mp4
        ├── phase3_cost_sweep_D1.mp4
        ├── phase3_cost_sweep_action_V0.mp4
        ├── phase3_cost_sweep_action_V1.mp4
        └── phase3_cost_sweep_delta_alpha_idx.mp4
```

## Key findings

- When measurement is free (`c_meas = 0`), a second measurement always weakly improves performance; the gain is largest in the interior of the simplex.
- As `c_meas` grows, the decision boundary contracts inward — near-certain beliefs become unprofitable to measure first.
- The `Δα` map (change in optimal angle between Stage 0 and Stage 1) reveals that the sequential policy is **adaptive**: the first-step angle is adjusted anticipating the second-step correction.
- The 2×2 panel animation (`phase3_cost_sweep_panel_2x2.mp4`) provides the clearest visual summary of how cost reshapes the entire policy.
