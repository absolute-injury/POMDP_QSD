#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trine_one_step.phase2 import load_phase1_npz
from trine_one_step.phase3 import build_transition_cache, solve_phase3_h2
from trine_one_step.phase3_plotting import create_phase3_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase III H=2 sequential Bellman maps for trine QSD.")
    parser.add_argument(
        "--phase1-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs/paper_final/data/one_step_maps.npz",
        help="Phase I artifact used to reuse belief grid and alpha-grid.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="phase3_sequential",
        help="Output run folder name.",
    )
    parser.add_argument("--c0", type=float, default=0.0, help="Measurement cost for baseline run.")
    parser.add_argument(
        "--c-eps",
        dest="c_eps",
        type=float,
        default=0.02,
        help="Small positive measurement cost for sensitivity run.",
    )
    parser.add_argument(
        "--skip-eps",
        action="store_true",
        help="Skip small-positive-cost run.",
    )
    parser.add_argument("--decision-tol", type=float, default=1e-12, help="Stop-vs-measure tie tolerance.")
    parser.add_argument("--prob-tol", type=float, default=1e-12, help="Probability normalization tolerance.")
    parser.add_argument("--posterior-tol", type=float, default=1e-12, help="Posterior normalization tolerance.")
    parser.add_argument("--nonneg-tol", type=float, default=1e-12, help="Nonnegativity tolerance.")
    parser.add_argument(
        "--tiny-negative-tol",
        type=float,
        default=1e-10,
        help="Threshold for classifying tiny negative D1/D0 values.",
    )
    parser.add_argument(
        "--plot-clip-tol",
        type=float,
        default=1e-10,
        help="Tiny negative clipping threshold for D1/D0 plotting only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.outdir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    phase1 = load_phase1_npz(str(args.phase1_npz))
    beliefs = np.asarray(phase1["beliefs"], dtype=float)
    lattice = np.asarray(phase1["lattice"], dtype=int)
    xy = np.asarray(phase1["xy"], dtype=float)
    alpha_grid = np.asarray(phase1["alpha_grid"], dtype=float)
    resolution = int(np.sum(lattice[0]))

    cache = build_transition_cache(
        beliefs=beliefs,
        lattice=lattice,
        alpha_grid=alpha_grid,
        probability_tol=args.prob_tol,
        posterior_tol=args.posterior_tol,
        nonneg_tol=args.nonneg_tol,
    )

    run0 = solve_phase3_h2(
        beliefs=beliefs,
        alpha_grid=alpha_grid,
        cache=cache,
        c_meas=args.c0,
        decision_tol=args.decision_tol,
        tiny_negative_tol=args.tiny_negative_tol,
    )
    fig_paths = create_phase3_figures(
        run=run0,
        xy=xy,
        out_dir=run_dir,
        suffix="run0",
        tiny_negative_clip_tol=args.plot_clip_tol,
    )

    values_run0 = run_dir / "phase3_values_run0.npz"
    diag_run0 = run_dir / "phase3_diag_run0.json"
    _save_phase3_npz(values_run0, beliefs, lattice, xy, alpha_grid, run0)
    _save_json(diag_run0, run0.diagnostics)

    run_eps = None
    values_run_eps = None
    fig_paths_eps = None
    if not args.skip_eps:
        run_eps = solve_phase3_h2(
            beliefs=beliefs,
            alpha_grid=alpha_grid,
            cache=cache,
            c_meas=args.c_eps,
            decision_tol=args.decision_tol,
            tiny_negative_tol=args.tiny_negative_tol,
        )
        fig_paths_eps = create_phase3_figures(
            run=run_eps,
            xy=xy,
            out_dir=run_dir,
            suffix="run_eps",
            tiny_negative_clip_tol=args.plot_clip_tol,
        )
        values_run_eps = run_dir / "phase3_values_run_eps.npz"
        _save_phase3_npz(values_run_eps, beliefs, lattice, xy, alpha_grid, run_eps)
        _save_json(run_dir / "phase3_diag_run_eps.json", run_eps.diagnostics)

    summary_path = run_dir / "phase3_summary.md"
    summary_text = _build_summary(
        phase1_path=args.phase1_npz,
        beliefs=beliefs,
        resolution=resolution,
        n_beliefs=beliefs.shape[0],
        n_alpha=alpha_grid.size,
        c0=args.c0,
        c_eps=None if args.skip_eps else args.c_eps,
        run0=run0,
        run_eps=run_eps,
        diag_run0=diag_run0,
        values_run0=values_run0,
        values_run_eps=values_run_eps,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"[run] completed {args.tag}")
    print(f"[phase1] {args.phase1_npz.resolve()}")
    print(f"[summary] N={resolution}, points={beliefs.shape[0]}, M_alpha={alpha_grid.size}")
    print(f"[data] {values_run0.resolve()}")
    print(f"[log ] {diag_run0.resolve()}")
    for key in ("V1", "V0", "D1", "D0"):
        print(f"[fig ] {key} png: {fig_paths[key]['png'].resolve()}")
        print(f"[fig ] {key} pdf: {fig_paths[key]['pdf'].resolve()}")
    optional_keys = [key for key in fig_paths.keys() if key not in {"V1", "V0", "D1", "D0"}]
    for key in optional_keys:
        print(f"[fig ] {key} png: {fig_paths[key]['png'].resolve()}")
        print(f"[fig ] {key} pdf: {fig_paths[key]['pdf'].resolve()}")
    if values_run_eps is not None:
        print(f"[data] {values_run_eps.resolve()}")
    if fig_paths_eps is not None:
        for key in ("V1", "V0", "D1", "D0"):
            print(f"[fig ] {key} run_eps png: {fig_paths_eps[key]['png'].resolve()}")
            print(f"[fig ] {key} run_eps pdf: {fig_paths_eps[key]['pdf'].resolve()}")
        optional_eps_keys = [key for key in fig_paths_eps.keys() if key not in {"V1", "V0", "D1", "D0"}]
        for key in optional_eps_keys:
            print(f"[fig ] {key} run_eps png: {fig_paths_eps[key]['png'].resolve()}")
            print(f"[fig ] {key} run_eps pdf: {fig_paths_eps[key]['pdf'].resolve()}")
    print(f"[note] {summary_path.resolve()}")


def _save_phase3_npz(
    path: Path,
    beliefs: np.ndarray,
    lattice: np.ndarray,
    xy: np.ndarray,
    alpha_grid: np.ndarray,
    run,
) -> None:
    np.savez_compressed(
        path,
        beliefs=beliefs,
        lattice=lattice,
        xy=xy,
        alpha_grid=alpha_grid,
        c_meas=np.array(run.c_meas),
        V2=run.V2,
        V1=run.V1,
        V0=run.V0,
        D1=run.D1,
        D0=run.D0,
        stage1_best_alpha_idx=run.stage1_best_alpha_idx,
        stage1_best_alpha=run.stage1_best_alpha,
        stage1_measure_mask=run.stage1_measure_mask.astype(np.int8),
        stage1_stop_mask=(~run.stage1_measure_mask).astype(np.int8),
        stage0_best_alpha_idx=run.stage0_best_alpha_idx,
        stage0_best_alpha=run.stage0_best_alpha,
        stage0_measure_mask=run.stage0_measure_mask.astype(np.int8),
        stage0_stop_mask=(~run.stage0_measure_mask).astype(np.int8),
        delta_alpha_idx=run.delta_alpha_idx,
        diagnostics_json=np.array(json.dumps(run.diagnostics)),
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _build_summary(
    phase1_path: Path,
    beliefs: np.ndarray,
    resolution: int,
    n_beliefs: int,
    n_alpha: int,
    c0: float,
    c_eps: float | None,
    run0,
    run_eps,
    diag_run0: Path,
    values_run0: Path,
    values_run_eps: Path | None,
) -> str:
    center_target = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    center_idx = int(np.argmin(np.sum((beliefs - center_target[None, :]) ** 2, axis=1)))
    run0_branch = run0.diagnostics["branch_statistics"]
    d1_q = np.quantile(run0.D1, [0.5, 0.9, 0.99])
    d0_q = np.quantile(run0.D0, [0.5, 0.9, 0.99])
    valid_delta = run0.delta_alpha_idx >= 0
    action_diff_fraction = float(np.mean((run0.delta_alpha_idx > 0) & valid_delta)) if np.any(valid_delta) else 0.0

    lines = [
        "# Phase III Summary",
        "",
        "## Configuration",
        f"- Phase I source: `{phase1_path.resolve()}`",
        f"- Belief resolution N: `{resolution}`",
        f"- Number of grid points: `{n_beliefs}`",
        f"- Alpha samples M_alpha: `{n_alpha}`",
        f"- Baseline cost c_meas (run0): `{c0}`",
        f"- Epsilon cost c_meas (run_eps): `{c_eps if c_eps is not None else 'skipped'}`",
        "",
        "## Baseline (run0) quick reads",
        f"- Stage-1 continue fraction: `{run0_branch['stage1_measure_fraction']:.4f}`",
        f"- Stage-0 continue fraction: `{run0_branch['stage0_measure_fraction']:.4f}`",
        f"- D1 quantiles (50/90/99%): `{d1_q[0]:.6f}, {d1_q[1]:.6f}, {d1_q[2]:.6f}`",
        f"- D0 quantiles (50/90/99%): `{d0_q[0]:.6f}, {d0_q[1]:.6f}, {d0_q[2]:.6f}`",
        f"- Center D1 / D0: `{run0.D1[center_idx]:.6f}` / `{run0.D0[center_idx]:.6f}`",
        f"- Optional action-difference fraction (delta alpha idx > 0): `{action_diff_fraction:.4f}`",
        "",
        "## Diagnostics",
        f"- Probability normalization pass: `{run0.diagnostics['transition_checks']['probability_normalization']['pass']}`",
        f"- Posterior normalization pass: `{run0.diagnostics['transition_checks']['posterior_normalization']['pass']}`",
        f"- Nonnegativity pass (prob/post): "
        f"`{run0.diagnostics['transition_checks']['nonnegativity']['pass_probability']}` / "
        f"`{run0.diagnostics['transition_checks']['nonnegativity']['pass_posterior']}`",
        "",
        "## Outputs",
        f"- `phase3_values_run0.npz`: `{values_run0.resolve()}`",
        f"- `phase3_diag_run0.json`: `{diag_run0.resolve()}`",
    ]

    if values_run_eps is not None:
        lines.append(f"- `phase3_values_run_eps.npz`: `{values_run_eps.resolve()}`")
    lines.extend(
        [
            "",
            "## Notes",
            "- D1 and D0 are kept raw in arrays. Tiny negative numerical residues are clipped only in plotting.",
            "- Stage-1 and Stage-0 best-action maps are stored as alpha indices and alpha values.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
