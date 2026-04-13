# POMDP + QSD Simulation — Trine Geometry

## Goal of this repository and What each phase does

This repository contains numerical experiments, visualizations, and diagnostic scripts for trine-state quantum state discrimination (QSD) viewed through a POMDP-style decision framework. This repository is provided to ensure reproducibility and facilitate the computational interpretation of our findings.

- **Phase I** computes one-step optimal measurement-angle maps.
- **Phase II** analyzes posterior branch routing and representative belief updates.
- **Phase III** generates two-step sequential Bellman value/policy maps and related diagnostics.
- **Phase IV-B/D** runs discretization-robustness experiments and numerical consistency checks, related to selected theoretical claims in the paper.

## Installation
Create a virtual environment, install the required packages, and run the four phase scripts from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For animation/media outputs (Phase III cost-sweep MP4/GIF), install optional extras:

```bash
python -m pip install -e ".[media]"
```

## Tested Environment

- Python: `3.12.7` (also targets `>=3.10` per `pyproject.toml`)
- OS: macOS (Apple Silicon)
- Headless rendering tip: set `MPLCONFIGDIR=/tmp/mpl`. In such headless environments, setting `MPLCONFIGDIR=/tmp/mpl` helps prevent Matplotlib cache permission issues.

## Repository Structure

- `phase1_one_step/` - Phase I package and outputs
- `phase2_posterior_routing/` - Phase II package and outputs
- `phase3_sequential/` - Phase III package and outputs
- `phase4_bd/` - Phase IV-B/D package and outputs
- `src/trine_one_step/` - shared core modules used by all phase scripts

Canonical source code resides in `src/trine_one_step/`, while phase-specific folders contain the execution scripts and generated outputs.

## Recommended Run Order

### Phase I

```bash
MPLCONFIGDIR=/tmp/mpl python phase1_one_step/code/scripts/run_one_step.py \
  --N 40 \
  --M-alpha 120 \
  --outdir phase1_one_step \
  --tag results
```

See: `phase1_one_step/README.md`

### Phase II

```bash
MPLCONFIGDIR=/tmp/mpl python phase2_posterior_routing/code/scripts/run_phase2_posterior_routing.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase2_posterior_routing \
  --tag results
```

See: `phase2_posterior_routing/README.md`

### Phase III

```bash
MPLCONFIGDIR=/tmp/mpl python phase3_sequential/code/scripts/run_phase3_sequential.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential \
  --tag results
```

Optional cost-sweep animations:

```bash
MPLCONFIGDIR=/tmp/mpl python phase3_sequential/code/scripts/make_phase3_cost_gifs.py \
  --phase1-npz phase1_one_step/results/data/one_step_maps.npz \
  --outdir phase3_sequential/results/figures/animations
```

See: `phase3_sequential/README.md`

### Phase IV-B/D

```bash
MPLCONFIGDIR=/tmp/mpl python phase4_bd/code/scripts/run_phase4_bd.py --outdir phase4_bd/results_full
MPLCONFIGDIR=/tmp/mpl python phase4_bd/code/scripts/run_phase4_focused_rerun.py --outdir phase4_bd/results_full/focused
```

See: `phase4_bd/README.md`

## Minimal Artifacts Supporting the Main Results

If you are preserving only paper-core results, keep at least:

- Phase I: `phase1_one_step/results/data/one_step_maps.npz`
- Phase II: `phase2_posterior_routing/results/data/phase2_routing_raw.json`
- Phase III: value/diagnostic NPZ+JSON and key static figures for `V0/V1/D0/D1`
- Phase IV: `phase4_bd/results_full/data/phase4B_compare_summary.csv`, `phase4_bd/results_full/diagnostics/phase4D_summary.md`, focused summary files

## Notes

- Phase IV-B/D is intended as numerical evidence and consistency checking for selected theoretical claims, not as a formal proof or a standalone convergence argument.
- For detailed option descriptions and interpretation notes, use each phase README.
