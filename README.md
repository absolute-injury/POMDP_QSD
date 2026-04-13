# POMDP + QSD Simulation — Trine Geometry

## Goal

This repository provides numerical experiments and visualization code for trine-state quantum state discrimination (QSD) viewed through a POMDP-style decision framework. The repository is intended to support the paper's problem modeling, conceptual interpretation, and provide reproducibility within the current numerical simulations.

- Phase I: one-step optimal measurement angle map
- Phase II: posterior branch routing diagnostics
- Phase III: two-step sequential Bellman policy/value maps
- Phase IV-B/D: discretization-robustness experiments and numerical consistency checks related to selected theoretical claims in the paper

## Prerequisites

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib pillow imageio imageio-ffmpeg
```

`MPLCONFIGDIR=/tmp/mpl` is recommended in headless environments.

## Repository Layout

- `phase1_one_step/` - Phase I package and outputs
- `phase2_posterior_routing/` - Phase II package and outputs
- `phase3_sequential/` - Phase III package and outputs
- `phase4_bd/` - Phase IV-B/D package and outputs
- `src/trine_one_step/` - shared core modules used by all phase scripts

## Recommended Execution Order

### Phase I

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase1_one_step/code/scripts/run_one_step.py \
  --N 40 \
  --M-alpha 120 \
  --outdir phase1_one_step \
  --tag results
```

See: `phase1_one_step/README.md`

### Phase II

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase2_posterior_routing/code/scripts/run_phase2_posterior_routing.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase2_posterior_routing \
  --tag results
```

See: `phase2_posterior_routing/README.md`

### Phase III

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase3_sequential/code/scripts/run_phase3_sequential.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential \
  --tag results
```

Optional cost-sweep animations:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase3_sequential/code/scripts/make_phase3_cost_gifs.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential/results/figures/animations
```

See: `phase3_sequential/README.md`

### Phase IV-B/D

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase4_bd/code/scripts/run_phase4_bd.py --outdir phase4_bd/results_full
MPLCONFIGDIR=/tmp/mpl .venv/bin/python phase4_bd/code/scripts/run_phase4_focused_rerun.py --outdir phase4_bd/results_full/focused
```

See: `phase4_bd/README.md`

## Core Artifacts To Keep

If you are preserving only paper-core results, keep at least:

- Phase I: `phase1_one_step/results/data/one_step_maps.npz`
- Phase II: `phase2_posterior_routing/results/data/phase2_routing_raw.json`
- Phase III: value/diagnostic NPZ+JSON and key static figures for `V0/V1/D0/D1`
- Phase IV: `phase4_bd/results_full/data/phase4B_compare_summary.csv`, `phase4_bd/results_full/diagnostics/phase4D_summary.md`, focused summary files

## Notes

- Phase IV-B/D is intended as numerical evidence and consistency checking for selected theoretical claims, not as a formal proof or a standalone convergence argument.
- For detailed option descriptions and interpretation notes, use each phase README.
