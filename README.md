# POMDP + QSD Simulation — Trine Geometry

Numerical simulation study of **Quantum State Discrimination (QSD)** under the symmetric three-state (Trine) geometry. The project investigates the structure of optimal measurement strategies across three experimental phases, progressing from a single-step setting to a two-step sequential decision problem.

---

## Background

The **Trine states** are three pure qubit states separated by 120° on the equatorial plane of the Bloch sphere. Discriminating among them is a canonical QSD problem. A key tunable parameter is the rotation angle **α** of the measurement basis.

This project asks: *given a prior belief over which Trine state was sent, what measurement angle(s) maximize the probability of correct identification — and does adding a second measurement step actually help?*

---

## Repository Structure

```
2026QSD_organized/
├── phase1_one_step/              # Phase I  — single-step optimal angle map
├── phase2_posterior_routing/     # Phase II — posterior belief routing analysis
├── phase3_sequential/            # Phase III — two-step sequential Bellman solver
├── pyproject.toml                # Package configuration (trine-one-step v0.1.0)
└── __init__.py
```

Each phase folder contains a `code/` subtree (source + scripts) and a `results/` subtree (data, figures, logs/diagnostics).

---

## Experimental Phases at a Glance

| Phase | Question | Key output |
|-------|----------|------------|
| [Phase I](phase1_one_step/) | What is the best single measurement angle at every prior belief? | `j1_star` value map, gain surface, optimal α map |
| [Phase II](phase2_posterior_routing/) | After one measurement, where does the posterior land — and does routing matter? | Branch route table, posterior routing figures, 5 case studies |
| [Phase III](phase3_sequential/) | Does a second measurement add value, and by how much? | Bellman value maps V0/V1, decision maps D0/D1, cost-sweep animations |

---

## Setup

Python 3.10+ is required. Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Dependencies: `numpy >= 1.24, < 2.0` and `matplotlib >= 3.7`.

For Phase III animations, `imageio-ffmpeg` is also required:

```bash
pip install imageio-ffmpeg
```

---
