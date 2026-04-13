#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parents[2]

from trine_one_step.phase2 import (
    build_interpretation_note,
    load_phase1_npz,
    make_branch_rows,
    make_summary_rows,
    run_phase2_posterior_routing,
)
from trine_one_step.phase2_plotting import create_phase2_routing_figure
from trine_one_step.phase2_plotting import create_phase2_diagnostics_figure
from trine_one_step.phase2_plotting import create_phase2_case_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase II posterior routing visualization based on Phase I artifacts."
    )
    parser.add_argument(
        "--phase1-npz",
        type=Path,
        default=REPO_ROOT / "phase1_one_step/results/data/one_step_maps.npz",
        help="Path to Phase I one_step_maps.npz",
    )
    parser.add_argument(
        "--snap-metric",
        type=str,
        default="linf",
        choices=["linf", "l1", "l2"],
        help="Grid snapping metric for representative beliefs",
    )
    parser.add_argument(
        "--near-tie-gap",
        type=float,
        default=1e-6,
        help="Threshold on argmax gap used for Point E near-tie validation",
    )
    parser.add_argument(
        "--prob-tol",
        type=float,
        default=1e-12,
        help="Probability normalization tolerance",
    )
    parser.add_argument(
        "--posterior-tol",
        type=float,
        default=1e-12,
        help="Posterior normalization tolerance",
    )
    parser.add_argument(
        "--debug-figure-metadata",
        action="store_true",
        help="Render internal metadata overlays in Phase II figures",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PHASE_ROOT / "results",
        help="Output root directory",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="phase2_posterior_routing",
        help="Output run folder name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.outdir / args.tag
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    log_dir = run_dir / "logs"
    for path in (data_dir, fig_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    phase1 = load_phase1_npz(str(args.phase1_npz))
    result = run_phase2_posterior_routing(
        phase1=phase1,
        snap_metric=args.snap_metric,
        near_tie_gap_threshold=args.near_tie_gap,
        prob_tol=args.prob_tol,
        posterior_tol=args.posterior_tol,
    )

    raw_json_path = data_dir / "phase2_routing_raw.json"
    checks_json_path = log_dir / "phase2_checks.json"
    summary_csv_path = data_dir / "phase2_summary.csv"
    branch_csv_path = data_dir / "phase2_branch_routes.csv"
    notes_path = log_dir / "phase2_interpretation_notes.txt"
    figure_png = fig_dir / "figure_D_phase2_posterior_routing.png"
    figure_pdf = fig_dir / "figure_D_phase2_posterior_routing.pdf"
    diag_png = fig_dir / "figure_D2_phase2_diagnostics.png"
    diag_pdf = fig_dir / "figure_D2_phase2_diagnostics.pdf"
    case_dir = fig_dir / "case_details"

    _save_json(raw_json_path, result)
    _save_json(checks_json_path, result["global_checks"])
    _save_csv(summary_csv_path, make_summary_rows(result))
    _save_csv(branch_csv_path, make_branch_rows(result))
    notes_path.write_text(build_interpretation_note(result), encoding="utf-8")
    create_phase2_routing_figure(
        result=result,
        out_png=figure_png,
        out_pdf=figure_pdf,
        show_internal_metadata=args.debug_figure_metadata,
    )
    create_phase2_diagnostics_figure(result=result, out_png=diag_png, out_pdf=diag_pdf)
    case_paths = create_phase2_case_figures(
        result=result,
        out_dir=case_dir,
        show_internal_metadata=args.debug_figure_metadata,
    )

    print(f"[run] completed {args.tag}")
    print(f"[phase1] {args.phase1_npz.resolve()}")
    print(f"[fig ] {figure_png.resolve()}")
    print(f"[fig ] {figure_pdf.resolve()}")
    print(f"[fig ] {diag_png.resolve()}")
    print(f"[fig ] {diag_pdf.resolve()}")
    for label in ["A", "B", "C", "D", "E"]:
        info = case_paths[label]
        print(f"[fig ] case {label}: {info['png'].resolve()}")
        print(f"[fig ] case {label}: {info['pdf'].resolve()}")
    print(f"[data] {raw_json_path.resolve()}")
    print(f"[data] {summary_csv_path.resolve()}")
    print(f"[data] {branch_csv_path.resolve()}")
    print(f"[log ] {checks_json_path.resolve()}")
    print(f"[log ] {notes_path.resolve()}")
    print(
        f"[check] pass_prob={result['global_checks']['pass_all_probability_norm']}, "
        f"pass_post={result['global_checks']['pass_all_posterior_norm']}, "
        f"pass_j1={result['global_checks']['pass_all_j1_consistency']}"
    )
    print(
        f"[check] max_prob_res={result['global_checks']['max_abs_probability_residual']:.3e}, "
        f"max_post_res={result['global_checks']['max_abs_posterior_residual']:.3e}, "
        f"max_j1_res={result['global_checks']['max_abs_j1_residual']:.3e}"
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
