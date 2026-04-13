#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parents[2]

from trine_one_step.phase4 import (
    build_comparison_rows,
    build_phase4_config_matrix,
    build_phase4_diag_payload,
    build_phase4d_summary_markdown,
    build_representative_audit_rows,
    make_config_name,
    representative_rows_to_markdown,
    run_phase4_single_config,
    save_phase4_values_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase IV-B/IV-D package: discretization robustness and theorem-facing synthesis."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PHASE_ROOT / "results",
        help="Output root directory.",
    )
    parser.add_argument(
        "--c-meas",
        type=float,
        default=0.02,
        help="Shared measurement cost (Axis A already completed; no cost sweep here).",
    )
    parser.add_argument("--baseline-N", type=int, default=80, help="Baseline belief resolution.")
    parser.add_argument("--baseline-M-alpha", type=int, default=240, help="Baseline action resolution.")
    parser.add_argument(
        "--belief-resolutions",
        type=str,
        default="40,80,120",
        help="Comma-separated belief resolutions for B-1 sweep.",
    )
    parser.add_argument(
        "--action-resolutions",
        type=str,
        default="120,240,360",
        help="Comma-separated action resolutions for B-2 sweep.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Phase I batch size.")
    parser.add_argument("--tie-tol", type=float, default=1e-10, help="Phase I tie tolerance.")
    parser.add_argument("--decision-tol", type=float, default=1e-12, help="Bellman stop-vs-measure tolerance.")
    parser.add_argument("--prob-tol", type=float, default=1e-12, help="Probability normalization tolerance.")
    parser.add_argument("--posterior-tol", type=float, default=1e-12, help="Posterior normalization tolerance.")
    parser.add_argument("--nonneg-tol", type=float, default=1e-12, help="Nonnegativity tolerance.")
    parser.add_argument(
        "--tiny-negative-tol",
        type=float,
        default=1e-10,
        help="Threshold for classifying tiny negative numerical residues.",
    )
    parser.add_argument(
        "--rep-snap-metric",
        choices=["linf", "l1", "l2"],
        default="l2",
        help="Metric used to snap representative target beliefs onto each grid.",
    )
    parser.add_argument("--skip-figures", action="store_true", help="Skip comparison figure generation.")
    parser.add_argument("--skip-rep-md", action="store_true", help="Skip markdown version of rep-point audit table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    belief_resolutions = _parse_int_csv(args.belief_resolutions)
    action_resolutions = _parse_int_csv(args.action_resolutions)

    configs = build_phase4_config_matrix(
        belief_resolutions=belief_resolutions,
        action_resolutions=action_resolutions,
        baseline_N=args.baseline_N,
        baseline_M_alpha=args.baseline_M_alpha,
    )
    baseline_name = make_config_name(args.baseline_N, args.baseline_M_alpha)

    outdir = args.outdir
    data_dir = outdir / "data"
    diag_dir = outdir / "diagnostics"
    fig_dir = outdir / "figures"
    for path in (data_dir, diag_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)

    bundles = []
    for config in configs:
        bundle = run_phase4_single_config(
            config=config,
            c_meas=args.c_meas,
            batch_size=args.batch_size,
            tie_tol=args.tie_tol,
            decision_tol=args.decision_tol,
            prob_tol=args.prob_tol,
            posterior_tol=args.posterior_tol,
            nonneg_tol=args.nonneg_tol,
            tiny_negative_tol=args.tiny_negative_tol,
        )
        bundles.append(bundle)

        values_path = data_dir / f"phase4B_values_{config.name}.npz"
        save_phase4_values_npz(values_path, bundle)

    comparison_rows = build_comparison_rows(bundles=bundles, baseline_config_name=baseline_name)
    rep_rows = build_representative_audit_rows(
        bundles=bundles,
        snap_metric=args.rep_snap_metric,
    )

    comparison_by_config = {row["config"]: row for row in comparison_rows}

    plotting_seconds_total = 0.0
    fig_paths: dict[str, dict[str, Path]] = {}
    if not args.skip_figures:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from trine_one_step.phase4_plotting import create_phase4_compare_figures

        plot_start = perf_counter()
        fig_paths = create_phase4_compare_figures(comparison_rows=comparison_rows, out_dir=fig_dir)
        plotting_seconds_total = perf_counter() - plot_start

    plotting_seconds_per_run = plotting_seconds_total / max(1, len(bundles))
    for bundle in bundles:
        diag = build_phase4_diag_payload(
            bundle=bundle,
            comparison_row=comparison_by_config.get(bundle.config.name),
            plotting_seconds=plotting_seconds_per_run,
        )
        diag_path = diag_dir / f"phase4B_diag_{bundle.config.name}.json"
        _save_json(diag_path, diag)

    comparison_csv_path = data_dir / "phase4B_compare_summary.csv"
    comparison_json_path = diag_dir / "phase4B_compare_summary.json"
    rep_csv_path = data_dir / "phase4_rep_point_audit.csv"
    rep_md_path = data_dir / "phase4_rep_point_audit.md"
    summary_path = diag_dir / "phase4D_summary.md"
    manifest_path = diag_dir / "phase4B_manifest.json"

    _save_csv(comparison_csv_path, comparison_rows)
    _save_json(comparison_json_path, {"rows": comparison_rows})
    _save_csv(rep_csv_path, rep_rows)
    if not args.skip_rep_md:
        rep_md_path.write_text(representative_rows_to_markdown(rep_rows), encoding="utf-8")

    summary = build_phase4d_summary_markdown(
        comparison_rows=comparison_rows,
        representative_rows=rep_rows,
        baseline_config_name=baseline_name,
        c_meas=args.c_meas,
    )
    summary_path.write_text(summary, encoding="utf-8")

    _save_json(
        manifest_path,
        {
            "scope": {
                "phase": "IV-B/IV-D",
                "axis_A": "completed",
                "axis_C": "excluded_by_default",
            },
            "baseline": baseline_name,
            "c_meas": args.c_meas,
            "configs": [config.name for config in configs],
            "comparison_csv": _display_path(comparison_csv_path),
            "rep_audit_csv": _display_path(rep_csv_path),
            "phase4d_summary": _display_path(summary_path),
        },
    )

    print("[run] completed phase4_bd")
    print(f"[cfg] baseline={baseline_name}, c_meas={args.c_meas:.6f}")
    for config in configs:
        print(f"[data] {_display_path(data_dir / f'phase4B_values_{config.name}.npz')}")
        print(f"[diag] {_display_path(diag_dir / f'phase4B_diag_{config.name}.json')}")
    print(f"[data] {_display_path(comparison_csv_path)}")
    print(f"[diag] {_display_path(comparison_json_path)}")
    print(f"[data] {_display_path(rep_csv_path)}")
    if not args.skip_rep_md:
        print(f"[data] {_display_path(rep_md_path)}")
    for key, paths in fig_paths.items():
        print(f"[fig ] {key}: {_display_path(paths['png'])}")
        print(f"[fig ] {key}: {_display_path(paths['pdf'])}")
    print(f"[note] {_display_path(summary_path)}")
    print(f"[note] {_display_path(manifest_path)}")


def _parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("resolution list must not be empty")
    return out


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


def _display_path(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


if __name__ == "__main__":
    main()
