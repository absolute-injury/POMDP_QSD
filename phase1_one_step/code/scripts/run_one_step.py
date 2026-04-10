#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trine_one_step.plotting import create_standard_figures
from trine_one_step.solver import run_sanity_checks, solve_one_step_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve trine one-step geometry maps.")
    parser.add_argument("--N", type=int, default=40, help="Belief grid resolution")
    parser.add_argument("--M-alpha", dest="M_alpha", type=int, default=120, help="Alpha samples")
    parser.add_argument("--batch-size", type=int, default=512, help="Belief batch size")
    parser.add_argument("--tie-tol", type=float, default=1e-10, help="Tie tolerance")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs", help="Output directory")
    parser.add_argument("--tag", type=str, default="", help="Optional run folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.tag or f"N{args.N}_M{args.M_alpha}"
    run_dir = args.outdir / run_name
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    log_dir = run_dir / "logs"
    for path in (data_dir, fig_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    result = solve_one_step_maps(
        resolution=args.N,
        alpha_samples=args.M_alpha,
        batch_size=args.batch_size,
        tie_tol=args.tie_tol,
    )
    sanity = run_sanity_checks(result)
    figure_paths = create_standard_figures(result, fig_dir)

    _save_npz(data_dir / "one_step_maps.npz", result, sanity)
    _save_point_table_csv(data_dir / "one_step_maps.csv", result)
    _save_json(log_dir / "sanity_checks.json", sanity)

    print(f"[run] completed {run_name}")
    print(f"[data] {(data_dir / 'one_step_maps.npz').resolve()}")
    print(f"[data] {(data_dir / 'one_step_maps.csv').resolve()}")
    print(f"[log ] {(log_dir / 'sanity_checks.json').resolve()}")
    for key, path in figure_paths.items():
        print(f"[fig ] {key}: {path.resolve()}")
    print(f"[summary] points={result.belief_grid.beliefs.shape[0]}, alpha={result.alpha_grid.size}")
    print(f"[summary] gain_min={sanity['gain_min']:.3e}, vertex_gain_max_abs={sanity['vertex_gain_max_abs']:.3e}")


def _save_npz(path: Path, result, sanity: dict) -> None:
    np.savez_compressed(
        path,
        beliefs=result.belief_grid.beliefs,
        lattice=result.belief_grid.lattice,
        xy=result.belief_grid.xy,
        alpha_grid=result.alpha_grid,
        stopping_value=result.stopping_value,
        j1_star=result.j1_star,
        gain=result.gain,
        best_alpha=result.best_alpha,
        best_alpha_idx=result.best_alpha_idx,
        second_best=result.second_best,
        is_degenerate=result.is_degenerate.astype(np.int8),
        tie_tol=np.array(result.tie_tol),
        sanity_json=np.array(json.dumps(sanity)),
    )


def _save_point_table_csv(path: Path, result) -> None:
    rows = np.column_stack(
        (
            result.belief_grid.beliefs,
            result.belief_grid.xy,
            result.stopping_value,
            result.j1_star,
            result.gain,
            result.best_alpha,
            result.best_alpha_idx,
            result.is_degenerate.astype(int),
        )
    )
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header="b1,b2,b3,x,y,S,J1_star,G,alpha_star,alpha_idx,is_degenerate",
        comments="",
        fmt=[
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%.10f",
            "%d",
            "%d",
        ],
    )


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
