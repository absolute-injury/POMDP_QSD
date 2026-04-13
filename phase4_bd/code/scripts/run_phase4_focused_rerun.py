#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from trine_one_step.core import SQRT3_OVER_2, make_alpha_grid, likelihood_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase IV-B optional focused rerun: local boundary reevaluation without full reruns."
    )
    parser.add_argument(
        "--baseline-npz",
        type=Path,
        default=REPO_ROOT / "phase4_bd/results_full/data/phase4B_values_N80_M240.npz",
        help="Baseline Phase4 values npz.",
    )
    parser.add_argument(
        "--coarse-npz",
        type=Path,
        default=REPO_ROOT / "phase4_bd/results_full/data/phase4B_values_N40_M240.npz",
        help="Coarse comparison npz.",
    )
    parser.add_argument(
        "--refined-npz",
        type=Path,
        default=REPO_ROOT / "phase4_bd/results_full/data/phase4B_values_N120_M240.npz",
        help="Refined comparison npz.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "phase4_bd/results_full/focused",
        help="Focused-rerun output directory.",
    )
    parser.add_argument("--top-k", type=int, default=40, help="Number of suspicious baseline points to reevaluate.")
    parser.add_argument(
        "--alpha-dense",
        type=int,
        default=360,
        help="Dense alpha resolution used for local exact reevaluation.",
    )
    parser.add_argument(
        "--decision-tol",
        type=float,
        default=1e-12,
        help="Decision tolerance for stop-vs-measure in exact reevaluation.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-15,
        help="Safety epsilon for probability divisions.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip focused-rerun figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    data_dir = outdir / "data"
    diag_dir = outdir / "diagnostics"
    fig_dir = outdir / "figures"
    for path in (data_dir, diag_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)

    baseline = _load_npz(args.baseline_npz)
    coarse = _load_npz(args.coarse_npz)
    refined = _load_npz(args.refined_npz)

    c_meas = float(baseline["c_meas"])

    b = baseline["beliefs"]
    b_xy = baseline["xy"]
    b_D0 = baseline["D0"]
    b_D1 = baseline["D1"]
    b_m0 = baseline["stage0_measure_mask"].astype(bool)
    b_m1 = baseline["stage1_measure_mask"].astype(bool)

    coarse_idx, coarse_map_dist = _map_to_grid(b, coarse["resolution"])
    refined_idx, refined_map_dist = _map_to_grid(b, refined["resolution"])

    c_D0 = coarse["D0"][coarse_idx]
    c_D1 = coarse["D1"][coarse_idx]
    c_m0 = coarse["stage0_measure_mask"][coarse_idx].astype(bool)
    c_m1 = coarse["stage1_measure_mask"][coarse_idx].astype(bool)

    r_D0 = refined["D0"][refined_idx]
    r_D1 = refined["D1"][refined_idx]
    r_m0 = refined["stage0_measure_mask"][refined_idx].astype(bool)
    r_m1 = refined["stage1_measure_mask"][refined_idx].astype(bool)

    diff_c = np.abs(c_D0 - b_D0)
    diff_r = np.abs(r_D0 - b_D0)

    decision_disagree = (c_m0 != b_m0) | (r_m0 != b_m0)
    sign_flip = (_signed_nonzero(c_D0) != _signed_nonzero(b_D0)) | (_signed_nonzero(r_D0) != _signed_nonzero(b_D0))

    score = (
        _robust_norm(diff_c)
        + _robust_norm(diff_r)
        + 1.5 * decision_disagree.astype(float)
        + 1.0 * sign_flip.astype(float)
    )

    selected_idx = _select_indices(score=score, decision_disagree=decision_disagree, sign_flip=sign_flip, top_k=args.top_k)
    selected_idx.sort()

    alpha_dense = make_alpha_grid(args.alpha_dense)
    like_dense = likelihood_table(alpha_dense)

    selected_beliefs = b[selected_idx]
    exact = _evaluate_exact_h2_points(
        beliefs=selected_beliefs,
        like_table_dense=like_dense,
        c_meas=c_meas,
        decision_tol=args.decision_tol,
        eps=args.eps,
    )

    rows = []
    for local_i, idx in enumerate(selected_idx):
        row = {
            "baseline_index": int(idx),
            "b1": float(b[idx, 0]),
            "b2": float(b[idx, 1]),
            "b3": float(b[idx, 2]),
            "x": float(b_xy[idx, 0]),
            "y": float(b_xy[idx, 1]),
            "suspicion_score": float(score[idx]),
            "baseline_D1": float(b_D1[idx]),
            "baseline_D0": float(b_D0[idx]),
            "coarse_D1": float(c_D1[idx]),
            "coarse_D0": float(c_D0[idx]),
            "refined_D1": float(r_D1[idx]),
            "refined_D0": float(r_D0[idx]),
            "exact_D1_dense": float(exact["D1"][local_i]),
            "exact_D0_dense": float(exact["D0"][local_i]),
            "abs_err_baseline_D1_vs_exact": float(abs(b_D1[idx] - exact["D1"][local_i])),
            "abs_err_baseline_D0_vs_exact": float(abs(b_D0[idx] - exact["D0"][local_i])),
            "abs_err_coarse_D0_vs_exact": float(abs(c_D0[idx] - exact["D0"][local_i])),
            "abs_err_refined_D0_vs_exact": float(abs(r_D0[idx] - exact["D0"][local_i])),
            "baseline_stage1_measure": int(b_m1[idx]),
            "baseline_stage0_measure": int(b_m0[idx]),
            "coarse_stage1_measure": int(c_m1[idx]),
            "coarse_stage0_measure": int(c_m0[idx]),
            "refined_stage1_measure": int(r_m1[idx]),
            "refined_stage0_measure": int(r_m0[idx]),
            "exact_stage1_measure": int(exact["stage1_measure"][local_i]),
            "exact_stage0_measure": int(exact["stage0_measure"][local_i]),
            "baseline_vs_exact_D0_sign_flip": int(_signed_nonzero(np.array([b_D0[idx]]))[0] != _signed_nonzero(np.array([exact["D0"][local_i]]))[0]),
            "coarse_vs_exact_D0_sign_flip": int(_signed_nonzero(np.array([c_D0[idx]]))[0] != _signed_nonzero(np.array([exact["D0"][local_i]]))[0]),
            "refined_vs_exact_D0_sign_flip": int(_signed_nonzero(np.array([r_D0[idx]]))[0] != _signed_nonzero(np.array([exact["D0"][local_i]]))[0]),
            "map_dist_to_coarse": float(coarse_map_dist[idx]),
            "map_dist_to_refined": float(refined_map_dist[idx]),
        }
        rows.append(row)

    csv_path = data_dir / "phase4B_focused_points.csv"
    _save_csv(csv_path, rows)

    summary = _build_summary(
        rows=rows,
        top_k=args.top_k,
        alpha_dense=args.alpha_dense,
        c_meas=c_meas,
        coarse_name=coarse["config_name"],
        refined_name=refined["config_name"],
        baseline_name=baseline["config_name"],
    )
    summary_md_path = diag_dir / "phase4B_focused_summary.md"
    summary_json_path = diag_dir / "phase4B_focused_summary.json"
    summary_md_path.write_text(summary["markdown"], encoding="utf-8")
    _save_json(summary_json_path, summary["json"])

    if not args.skip_figures:
        os.environ.setdefault("MPLBACKEND", "Agg")
        _make_figures(
            baseline_xy=b_xy,
            baseline_D0=b_D0,
            rows=rows,
            outdir=fig_dir,
        )

    print("[run] completed phase4 focused rerun")
    print(f"[cfg] baseline={baseline['config_name']}, c_meas={c_meas:.6f}, top_k={args.top_k}, alpha_dense={args.alpha_dense}")
    print(f"[data] {csv_path.resolve()}")
    print(f"[note] {summary_md_path.resolve()}")
    print(f"[note] {summary_json_path.resolve()}")
    if not args.skip_figures:
        print(f"[fig ] {(fig_dir / 'phase4B_focused_map_D0_error.png').resolve()}")
        print(f"[fig ] {(fig_dir / 'phase4B_focused_sorted_error.png').resolve()}")


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    payload["beliefs"] = np.asarray(payload["beliefs"], dtype=float)
    payload["xy"] = np.asarray(payload["xy"], dtype=float)
    payload["D0"] = np.asarray(payload["D0"], dtype=float)
    payload["D1"] = np.asarray(payload["D1"], dtype=float)
    payload["stage0_measure_mask"] = np.asarray(payload["stage0_measure_mask"], dtype=bool)
    payload["stage1_measure_mask"] = np.asarray(payload["stage1_measure_mask"], dtype=bool)
    payload["lattice"] = np.asarray(payload["lattice"], dtype=int)
    payload["resolution"] = int(np.sum(payload["lattice"][0]))
    payload["c_meas"] = float(np.asarray(payload["c_meas"]).item())
    raw_name = payload.get("config_name", np.array("unknown"))
    if isinstance(raw_name, np.ndarray):
        payload["config_name"] = str(raw_name.item())
    else:
        payload["config_name"] = str(raw_name)
    return payload


def _select_indices(
    score: np.ndarray,
    decision_disagree: np.ndarray,
    sign_flip: np.ndarray,
    top_k: int,
) -> np.ndarray:
    n = score.size
    order = np.argsort(-score)

    critical = np.where(decision_disagree | sign_flip)[0]
    critical_order = critical[np.argsort(-score[critical])] if critical.size > 0 else np.array([], dtype=int)

    selected = []
    seen = set()

    for idx in critical_order:
        i = int(idx)
        if i in seen:
            continue
        selected.append(i)
        seen.add(i)
        if len(selected) >= top_k:
            return np.asarray(selected, dtype=int)

    for idx in order:
        i = int(idx)
        if i in seen:
            continue
        selected.append(i)
        seen.add(i)
        if len(selected) >= min(top_k, n):
            break

    return np.asarray(selected, dtype=int)


def _evaluate_exact_h2_points(
    beliefs: np.ndarray,
    like_table_dense: np.ndarray,
    c_meas: float,
    decision_tol: float,
    eps: float,
) -> dict[str, np.ndarray]:
    b = np.asarray(beliefs, dtype=float)
    stop = np.max(b, axis=1)

    V1, s1_idx, s1_measure = _stage1_value_batch(
        beliefs=b,
        like_table_dense=like_table_dense,
        c_meas=c_meas,
        decision_tol=decision_tol,
        eps=eps,
    )

    n_points = b.shape[0]
    n_alpha = like_table_dense.shape[0]
    best_q0 = np.full(n_points, -np.inf, dtype=float)
    best_idx0 = np.zeros(n_points, dtype=int)

    for a in range(n_alpha):
        like_a = like_table_dense[a]  # (outcome, state)
        unnorm = b[:, None, :] * like_a[None, :, :]
        obs = np.sum(unnorm, axis=2)
        safe = np.where(obs > eps, obs, 1.0)
        posts = unnorm / safe[:, :, None]
        zero_mask = obs <= eps
        if np.any(zero_mask):
            posts = np.where(zero_mask[:, :, None], b[:, None, :], posts)

        v1_posts, _, _ = _stage1_value_batch(
            beliefs=posts.reshape(-1, 3),
            like_table_dense=like_table_dense,
            c_meas=c_meas,
            decision_tol=decision_tol,
            eps=eps,
        )
        v1_posts = v1_posts.reshape(n_points, 3)
        q0 = -c_meas + np.sum(obs * v1_posts, axis=1)

        better = q0 > best_q0
        best_q0[better] = q0[better]
        best_idx0[better] = a

    s0_measure = best_q0 > (stop + decision_tol)
    V0 = np.where(s0_measure, best_q0, stop)

    return {
        "V1": V1,
        "V0": V0,
        "D1": V1 - stop,
        "D0": V0 - V1,
        "stage1_best_alpha_idx": s1_idx,
        "stage1_measure": s1_measure,
        "stage0_best_alpha_idx": np.where(s0_measure, best_idx0, -1),
        "stage0_measure": s0_measure,
    }


def _stage1_value_batch(
    beliefs: np.ndarray,
    like_table_dense: np.ndarray,
    c_meas: float,
    decision_tol: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(beliefs, dtype=float)
    stop = np.max(b, axis=1)

    unnorm = b[:, None, None, :] * like_table_dense[None, :, :, :]
    obs = np.sum(unnorm, axis=3)

    safe = np.where(obs > eps, obs, 1.0)
    posts = unnorm / safe[..., None]
    zero_mask = obs <= eps
    if np.any(zero_mask):
        posts = np.where(zero_mask[..., None], b[:, None, None, :], posts)

    v2_post = np.max(posts, axis=3)
    q1 = -c_meas + np.sum(obs * v2_post, axis=2)

    best_idx = np.argmax(q1, axis=1)
    best_q = q1[np.arange(q1.shape[0]), best_idx]
    measure = best_q > (stop + decision_tol)
    V1 = np.where(measure, best_q, stop)

    best_idx_out = np.where(measure, best_idx, -1).astype(int)
    return V1, best_idx_out, measure


def _map_to_grid(beliefs: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    projected = _nearest_lattice_coordinates(np.asarray(beliefs, dtype=float), resolution)
    idx = _lattice_index(projected[:, 0], projected[:, 1], resolution).astype(np.int64)
    projected_beliefs = projected.astype(float) / float(resolution)
    dist = np.linalg.norm(np.asarray(beliefs, dtype=float) - projected_beliefs, axis=1)
    return idx, dist


def _nearest_lattice_coordinates(points: np.ndarray, resolution: int) -> np.ndarray:
    scaled = points * float(resolution)
    floors = np.floor(scaled).astype(np.int64)
    frac = scaled - floors
    remainder = resolution - np.sum(floors, axis=1)

    out = floors.copy()
    order_desc = np.argsort(-frac, axis=1)

    rows_1 = np.nonzero(remainder >= 1)[0]
    if rows_1.size > 0:
        out[rows_1, order_desc[rows_1, 0]] += 1

    rows_2 = np.nonzero(remainder >= 2)[0]
    if rows_2.size > 0:
        out[rows_2, order_desc[rows_2, 1]] += 1

    invalid = (
        (np.sum(out, axis=1) != resolution)
        | np.any(out < 0, axis=1)
        | np.any(out > resolution, axis=1)
    )
    if np.any(invalid):
        bad_rows = np.nonzero(invalid)[0]
        for row in bad_rows:
            out[row] = _nearest_lattice_coordinates_row(points[row], resolution)

    return out.astype(np.int32)


def _nearest_lattice_coordinates_row(point: np.ndarray, resolution: int) -> np.ndarray:
    best = None
    best_dist = float("inf")
    for i in range(resolution + 1):
        for j in range(resolution - i + 1):
            k = resolution - i - j
            cand = np.array([i, j, k], dtype=float) / float(resolution)
            dist = float(np.sum((cand - point) ** 2))
            if dist < best_dist:
                best = np.array([i, j, k], dtype=np.int64)
                best_dist = dist
    if best is None:
        raise RuntimeError("failed to project point to lattice")
    return best


def _lattice_index(i: np.ndarray, j: np.ndarray, resolution: int) -> np.ndarray:
    i64 = np.asarray(i, dtype=np.int64)
    j64 = np.asarray(j, dtype=np.int64)
    return i64 * (resolution + 1) - (i64 * (i64 - 1)) // 2 + j64


def _robust_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    q = float(np.quantile(arr, 0.95))
    scale = max(q, 1e-12)
    return arr / scale


def _signed_nonzero(values: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros(arr.shape, dtype=int)
    out[arr > tol] = 1
    out[arr < -tol] = -1
    return out


def _build_summary(
    rows: list[dict[str, Any]],
    top_k: int,
    alpha_dense: int,
    c_meas: float,
    coarse_name: str,
    refined_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    if not rows:
        md = "# Phase IV-B Focused Rerun Summary\n\nNo rows were selected.\n"
        return {"markdown": md, "json": {"n_points": 0}}

    def arr(key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows], dtype=float)

    err_b = arr("abs_err_baseline_D0_vs_exact")
    err_c = arr("abs_err_coarse_D0_vs_exact")
    err_r = arr("abs_err_refined_D0_vs_exact")

    sign_b = int(np.sum([int(row["baseline_vs_exact_D0_sign_flip"]) for row in rows]))
    sign_c = int(np.sum([int(row["coarse_vs_exact_D0_sign_flip"]) for row in rows]))
    sign_r = int(np.sum([int(row["refined_vs_exact_D0_sign_flip"]) for row in rows]))

    md_lines = [
        "# Phase IV-B Focused Rerun Summary",
        "",
        "## Configuration",
        f"- Baseline: `{baseline_name}`",
        f"- Coarse comparator: `{coarse_name}`",
        f"- Refined comparator: `{refined_name}`",
        f"- Shared cost: `c_meas={c_meas:.6f}`",
        f"- Focused points: `{len(rows)}` (requested top-k=`{top_k}`)",
        f"- Dense local alpha resolution: `M_alpha={alpha_dense}`",
        "",
        "## Local Reevaluation (D0 vs dense exact)",
        f"- Baseline abs error mean/max: `{np.mean(err_b):.6e}` / `{np.max(err_b):.6e}`",
        f"- Coarse abs error mean/max: `{np.mean(err_c):.6e}` / `{np.max(err_c):.6e}`",
        f"- Refined abs error mean/max: `{np.mean(err_r):.6e}` / `{np.max(err_r):.6e}`",
        "",
        "## Sign Stability Against Dense Exact D0",
        f"- Baseline sign flips: `{sign_b}/{len(rows)}`",
        f"- Coarse sign flips: `{sign_c}/{len(rows)}`",
        f"- Refined sign flips: `{sign_r}/{len(rows)}`",
        "",
        "Interpretation note: this is a focused numerical reevaluation around suspicious boundary points, not a proof.",
        "",
    ]

    payload = {
        "n_points": int(len(rows)),
        "baseline": baseline_name,
        "coarse": coarse_name,
        "refined": refined_name,
        "c_meas": float(c_meas),
        "alpha_dense": int(alpha_dense),
        "D0_abs_error": {
            "baseline": {
                "mean": float(np.mean(err_b)),
                "median": float(np.median(err_b)),
                "max": float(np.max(err_b)),
            },
            "coarse": {
                "mean": float(np.mean(err_c)),
                "median": float(np.median(err_c)),
                "max": float(np.max(err_c)),
            },
            "refined": {
                "mean": float(np.mean(err_r)),
                "median": float(np.median(err_r)),
                "max": float(np.max(err_r)),
            },
        },
        "D0_sign_flips_vs_exact": {
            "baseline": int(sign_b),
            "coarse": int(sign_c),
            "refined": int(sign_r),
        },
    }

    return {"markdown": "\n".join(md_lines), "json": payload}


def _make_figures(
    baseline_xy: np.ndarray,
    baseline_D0: np.ndarray,
    rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(baseline_xy[:, 0], dtype=float)
    y = np.asarray(baseline_xy[:, 1], dtype=float)

    sel_x = np.array([float(row["x"]) for row in rows], dtype=float)
    sel_y = np.array([float(row["y"]) for row in rows], dtype=float)
    sel_err = np.array([float(row["abs_err_baseline_D0_vs_exact"]) for row in rows], dtype=float)

    # Figure 1: selected points over baseline D0 landscape.
    fig, ax = plt.subplots(figsize=(11.6, 9.0))
    bg = ax.scatter(x, y, c=baseline_D0, cmap="RdYlBu_r", s=12, alpha=0.55, linewidths=0)
    sc = ax.scatter(sel_x, sel_y, c=sel_err, cmap="magma", s=50, edgecolors="black", linewidths=0.45, zorder=5)
    _decorate_simplex(ax)
    ax.set_title("Focused Points on Baseline D0 Map (color=|D0_baseline - D0_exact|)")
    cbar1 = fig.colorbar(bg, ax=ax, location="left", fraction=0.05, pad=0.03)
    cbar1.set_label("baseline D0")
    cbar1.ax.yaxis.set_label_position("left")
    cbar1.ax.yaxis.tick_left()
    cbar2 = fig.colorbar(sc, ax=ax, location="right", fraction=0.05, pad=0.03)
    cbar2.set_label("|baseline D0 - exact D0|")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.05)
    fig.savefig(outdir / "phase4B_focused_map_D0_error.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: sorted error profile baseline/coarse/refined vs exact.
    err_b = np.array([float(row["abs_err_baseline_D0_vs_exact"]) for row in rows], dtype=float)
    err_c = np.array([float(row["abs_err_coarse_D0_vs_exact"]) for row in rows], dtype=float)
    err_r = np.array([float(row["abs_err_refined_D0_vs_exact"]) for row in rows], dtype=float)

    order = np.argsort(-err_b)
    xx = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.plot(xx, err_b[order], marker="o", linewidth=1.6, markersize=4.0, label="baseline vs exact")
    ax.plot(xx, err_c[order], marker="s", linewidth=1.4, markersize=3.7, label="coarse vs exact")
    ax.plot(xx, err_r[order], marker="^", linewidth=1.4, markersize=3.7, label="refined vs exact")
    ax.set_yscale("log")
    ax.set_xlabel("focused points (sorted by baseline error)")
    ax.set_ylabel("absolute D0 error")
    ax.set_title("Focused Rerun Error Profile (D0)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "phase4B_focused_sorted_error.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _decorate_simplex(ax) -> None:
    tri = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, SQRT3_OVER_2],
            [0.0, 0.0],
        ]
    )
    ax.plot(tri[:, 0], tri[:, 1], color="black", linewidth=1.8, zorder=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3_OVER_2 + 0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
