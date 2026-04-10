#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trine_one_step.core import SQRT3_OVER_2, TWO_PI_OVER_3
from trine_one_step.phase2 import load_phase1_npz
from trine_one_step.phase3 import build_transition_cache, solve_phase3_h2


MAP_LABELS = {
    "V1": r"$V_1(b)$",
    "V0": r"$V_0(b)$",
    "D1": r"$D_1(b)=V_1(b)-S(b)$",
    "D0": r"$D_0(b)=V_0(b)-V_1(b)$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Phase III cost-sweep animations with fixed and auto color scales."
    )
    parser.add_argument(
        "--phase1-npz",
        type=Path,
        default=PROJECT_ROOT / "outputs/paper_final/data/one_step_maps.npz",
        help="Phase I artifact used to reuse belief/alpha grids.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "outputs/phase3_sequential/gif",
        help="Output directory for GIF/MP4 files.",
    )
    parser.add_argument("--c-start", type=float, default=0.0, help="Start of cost sweep.")
    parser.add_argument("--c-end", type=float, default=0.02, help="End of cost sweep.")
    parser.add_argument("--frames", type=int, default=31, help="Number of frames.")
    parser.add_argument("--fps", type=float, default=8.0, help="Frames per second.")
    parser.add_argument("--decision-tol", type=float, default=1e-12, help="Stop-vs-measure tolerance.")
    parser.add_argument("--prob-tol", type=float, default=1e-12, help="Probability normalization tolerance.")
    parser.add_argument("--posterior-tol", type=float, default=1e-12, help="Posterior normalization tolerance.")
    parser.add_argument("--nonneg-tol", type=float, default=1e-12, help="Nonnegativity tolerance.")
    parser.add_argument(
        "--tiny-negative-tol",
        type=float,
        default=1e-10,
        help="Tiny-negative threshold for D1/D0 sanity classification.",
    )
    parser.add_argument(
        "--plot-clip-tol",
        type=float,
        default=1e-10,
        help="Plot-only clipping threshold for tiny negative D1/D0 values.",
    )
    parser.add_argument(
        "--skip-auto",
        action="store_true",
        help="Skip supplementary auto-scaled outputs.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate PNG frame folders.",
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Disable MP4 output (GIF only).",
    )
    parser.add_argument(
        "--mp4-quality",
        type=int,
        default=7,
        help="imageio ffmpeg quality (0-10, higher is better).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("frames must be >= 2")
    if args.fps <= 0.0:
        raise ValueError("fps must be positive")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    phase1 = load_phase1_npz(str(args.phase1_npz))
    beliefs = np.asarray(phase1["beliefs"], dtype=float)
    lattice = np.asarray(phase1["lattice"], dtype=int)
    xy = np.asarray(phase1["xy"], dtype=float)
    alpha_grid = np.asarray(phase1["alpha_grid"], dtype=float)

    cache = build_transition_cache(
        beliefs=beliefs,
        lattice=lattice,
        alpha_grid=alpha_grid,
        probability_tol=args.prob_tol,
        posterior_tol=args.posterior_tol,
        nonneg_tol=args.nonneg_tol,
    )

    costs = np.linspace(args.c_start, args.c_end, args.frames)
    duration_ms = int(round(1000.0 / args.fps))
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    _set_publication_style()

    runs: list[dict[str, np.ndarray | float]] = []
    for c_meas in costs:
        run = solve_phase3_h2(
            beliefs=beliefs,
            alpha_grid=alpha_grid,
            cache=cache,
            c_meas=float(c_meas),
            decision_tol=args.decision_tol,
            tiny_negative_tol=args.tiny_negative_tol,
        )
        runs.append(
            {
                "c_meas": float(c_meas),
                "V1": run.V1,
                "V0": run.V0,
                "D1": run.D1,
                "D0": run.D0,
                "action_V1": run.stage1_best_alpha,
                "action_V0": run.stage0_best_alpha,
                "measure_V1": run.stage1_measure_mask,
                "measure_V0": run.stage0_measure_mask,
                "delta_alpha_idx": run.delta_alpha_idx,
            }
        )

    fixed_norms = _compute_fixed_norms(runs=runs, tiny_negative_clip_tol=args.plot_clip_tol)
    fixed_delta_vmax = _compute_delta_vmax(runs)

    frames_root = outdir / "_frames"
    groups = {
        "fixed": {
            "V1": frames_root / "fixed" / "V1",
            "V0": frames_root / "fixed" / "V0",
            "D1": frames_root / "fixed" / "D1",
            "D0": frames_root / "fixed" / "D0",
            "panel": frames_root / "fixed" / "panel",
            "action_V1": frames_root / "fixed" / "action_V1",
            "action_V0": frames_root / "fixed" / "action_V0",
            "delta_alpha_idx": frames_root / "fixed" / "delta_alpha_idx",
        }
    }
    if not args.skip_auto:
        groups["auto"] = {
            "V1": frames_root / "auto" / "V1",
            "V0": frames_root / "auto" / "V0",
            "D1": frames_root / "auto" / "D1",
            "D0": frames_root / "auto" / "D0",
            "panel": frames_root / "auto" / "panel",
        }
    for group in groups.values():
        for path in group.values():
            path.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in enumerate(runs):
        c_meas = float(frame["c_meas"])
        maps = {
            "V1": np.asarray(frame["V1"]),
            "V0": np.asarray(frame["V0"]),
            "D1": np.asarray(frame["D1"]),
            "D0": np.asarray(frame["D0"]),
        }

        for key, values in maps.items():
            _render_map_frame(
                tri=tri,
                values=values,
                key=key,
                c_meas=c_meas,
                out_path=groups["fixed"][key] / f"{frame_idx:04d}.png",
                tiny_negative_clip_tol=args.plot_clip_tol,
                mode="fixed",
                fixed_norm=fixed_norms[key],
            )
        _render_panel_frame(
            tri=tri,
            values=maps,
            c_meas=c_meas,
            out_path=groups["fixed"]["panel"] / f"{frame_idx:04d}.png",
            tiny_negative_clip_tol=args.plot_clip_tol,
            mode="fixed",
            fixed_norms=fixed_norms,
        )
        _render_action_frame(
            tri=tri,
            best_alpha=np.asarray(frame["action_V1"]),
            measure_mask=np.asarray(frame["measure_V1"], dtype=bool),
            c_meas=c_meas,
            out_path=groups["fixed"]["action_V1"] / f"{frame_idx:04d}.png",
            title=r"Best Action (Stage 1, $V_1$)",
        )
        _render_action_frame(
            tri=tri,
            best_alpha=np.asarray(frame["action_V0"]),
            measure_mask=np.asarray(frame["measure_V0"], dtype=bool),
            c_meas=c_meas,
            out_path=groups["fixed"]["action_V0"] / f"{frame_idx:04d}.png",
            title=r"Best Action (Stage 0, $V_0$)",
        )
        _render_delta_alpha_frame(
            tri=tri,
            delta_alpha_idx=np.asarray(frame["delta_alpha_idx"]),
            c_meas=c_meas,
            out_path=groups["fixed"]["delta_alpha_idx"] / f"{frame_idx:04d}.png",
            vmax=fixed_delta_vmax,
        )

        if not args.skip_auto:
            for key, values in maps.items():
                _render_map_frame(
                    tri=tri,
                    values=values,
                    key=key,
                    c_meas=c_meas,
                    out_path=groups["auto"][key] / f"{frame_idx:04d}.png",
                    tiny_negative_clip_tol=args.plot_clip_tol,
                    mode="auto",
                    fixed_norm=None,
                )
            _render_panel_frame(
                tri=tri,
                values=maps,
                c_meas=c_meas,
                out_path=groups["auto"]["panel"] / f"{frame_idx:04d}.png",
                tiny_negative_clip_tol=args.plot_clip_tol,
                mode="auto",
                fixed_norms=None,
            )

    output_specs = [
        ("V1", groups["fixed"]["V1"], "phase3_cost_sweep_V1"),
        ("V0", groups["fixed"]["V0"], "phase3_cost_sweep_V0"),
        ("D1", groups["fixed"]["D1"], "phase3_cost_sweep_D1"),
        ("D0", groups["fixed"]["D0"], "phase3_cost_sweep_D0"),
        ("panel", groups["fixed"]["panel"], "phase3_cost_sweep_panel_2x2"),
        ("action_V1", groups["fixed"]["action_V1"], "phase3_cost_sweep_action_V1"),
        ("action_V0", groups["fixed"]["action_V0"], "phase3_cost_sweep_action_V0"),
        ("delta_alpha_idx", groups["fixed"]["delta_alpha_idx"], "phase3_cost_sweep_delta_alpha_idx"),
    ]
    if not args.skip_auto:
        output_specs.extend(
            [
                ("V1_auto", groups["auto"]["V1"], "phase3_cost_sweep_V1_auto"),
                ("V0_auto", groups["auto"]["V0"], "phase3_cost_sweep_V0_auto"),
                ("D1_auto", groups["auto"]["D1"], "phase3_cost_sweep_D1_auto"),
                ("D0_auto", groups["auto"]["D0"], "phase3_cost_sweep_D0_auto"),
                ("panel_auto", groups["auto"]["panel"], "phase3_cost_sweep_panel_2x2_auto"),
            ]
        )

    gif_paths: dict[str, Path] = {}
    for label, frame_dir, stem in output_specs:
        path = outdir / f"{stem}.gif"
        _build_gif_from_frames(frame_dir=frame_dir, out_path=path, duration_ms=duration_ms)
        gif_paths[label] = path

    mp4_paths: dict[str, Path] = {}
    if not args.no_mp4:
        for label, frame_dir, stem in output_specs:
            path = outdir / f"{stem}.mp4"
            _build_mp4_from_frames(
                frame_dir=frame_dir,
                out_path=path,
                fps=args.fps,
                quality=args.mp4_quality,
            )
            mp4_paths[label] = path

    if not args.keep_frames:
        shutil.rmtree(frames_root)

    print("[anim] completed phase3 cost sweep animation")
    print(
        f"[cfg] c_start={args.c_start:.6f}, c_end={args.c_end:.6f}, "
        f"frames={args.frames}, fps={args.fps:.2f}, auto={'off' if args.skip_auto else 'on'}"
    )
    for key, path in gif_paths.items():
        print(f"[gif] {key}: {path.resolve()}")
    for key, path in mp4_paths.items():
        print(f"[mp4] {key}: {path.resolve()}")


def _render_map_frame(
    tri: mtri.Triangulation,
    values: np.ndarray,
    key: str,
    c_meas: float,
    out_path: Path,
    tiny_negative_clip_tol: float,
    mode: str,
    fixed_norm: Normalize | TwoSlopeNorm | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 8.4))

    if key in {"D1", "D0"}:
        plotted = _clip_tiny_negative(values, tiny_negative_clip_tol)
        norm = fixed_norm if mode == "fixed" else _auto_norm_for_values(plotted, anchor_zero=True)
        mesh = ax.tripcolor(tri, plotted, shading="gouraud", cmap="RdYlBu_r", norm=norm)
        contour = ax.tricontour(tri, plotted, levels=14, colors="black", linewidths=0.33, alpha=0.28)
    else:
        norm = fixed_norm if mode == "fixed" else None
        cmap = "cividis" if key == "V1" else "viridis"
        mesh = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, norm=norm)
        contour = ax.tricontour(tri, values, levels=14, colors="white", linewidths=0.45, alpha=0.35)
    contour.set_zorder(3)

    _decorate_simplex(ax)
    ax.set_title(f"{MAP_LABELS[key]}\n$c_{{meas}}={c_meas:.4f}$ ({mode})")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(MAP_LABELS[key])
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.92)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _render_panel_frame(
    tri: mtri.Triangulation,
    values: dict[str, np.ndarray],
    c_meas: float,
    out_path: Path,
    tiny_negative_clip_tol: float,
    mode: str,
    fixed_norms: dict[str, Normalize | TwoSlopeNorm] | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8))
    order = ["V1", "V0", "D1", "D0"]

    for ax, key in zip(axes.ravel(), order):
        arr = values[key]
        if key in {"D1", "D0"}:
            plotted = _clip_tiny_negative(arr, tiny_negative_clip_tol)
            norm = fixed_norms[key] if (mode == "fixed" and fixed_norms is not None) else _auto_norm_for_values(
                plotted, anchor_zero=True
            )
            mesh = ax.tripcolor(tri, plotted, shading="gouraud", cmap="RdYlBu_r", norm=norm)
            contour = ax.tricontour(tri, plotted, levels=12, colors="black", linewidths=0.28, alpha=0.26)
        else:
            norm = fixed_norms[key] if (mode == "fixed" and fixed_norms is not None) else None
            cmap = "cividis" if key == "V1" else "viridis"
            mesh = ax.tripcolor(tri, arr, shading="gouraud", cmap=cmap, norm=norm)
            contour = ax.tricontour(tri, arr, levels=12, colors="white", linewidths=0.38, alpha=0.32)
        contour.set_zorder(3)
        _decorate_simplex(ax)
        ax.set_title(key)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.048, pad=0.02, shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"Phase III Cost Sweep   c_meas={c_meas:.4f}   ({mode})", fontsize=18, y=0.99, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.93, wspace=0.15, hspace=0.22)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _render_action_frame(
    tri: mtri.Triangulation,
    best_alpha: np.ndarray,
    measure_mask: np.ndarray,
    c_meas: float,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 8.4))
    masked_alpha = np.ma.masked_where(~measure_mask, np.asarray(best_alpha, dtype=float))

    mesh = ax.tripcolor(
        tri,
        masked_alpha,
        shading="gouraud",
        cmap="twilight_shifted",
        vmin=0.0,
        vmax=TWO_PI_OVER_3,
    )
    if np.any(~measure_mask):
        ax.scatter(
            tri.x[~measure_mask],
            tri.y[~measure_mask],
            s=8,
            c="#1f1f1f",
            alpha=0.75,
            label="stop",
            zorder=5,
        )
        ax.legend(loc="upper right", frameon=True, fontsize=11)

    _decorate_simplex(ax)
    ax.set_title(f"{title}\n$c_{{meas}}={c_meas:.4f}$")
    ax.text(
        0.015,
        0.02,
        f"measure fraction: {float(np.mean(measure_mask)):.4f}",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
        color="#2b2b2b",
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(r"$\alpha^*(b)$ (radian)")
    cbar.set_ticks([0.0, np.pi / 3.0, TWO_PI_OVER_3])
    cbar.set_ticklabels(["0", r"$\pi/3$", r"$2\pi/3$"])
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.92)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _render_delta_alpha_frame(
    tri: mtri.Triangulation,
    delta_alpha_idx: np.ndarray,
    c_meas: float,
    out_path: Path,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 8.4))
    delta = np.asarray(delta_alpha_idx, dtype=float)
    valid = delta >= 0.0
    masked = np.ma.masked_where(~valid, delta)

    mesh = ax.tripcolor(
        tri,
        masked,
        shading="gouraud",
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax, 1.0),
    )
    if np.any(~valid):
        ax.scatter(
            tri.x[~valid],
            tri.y[~valid],
            s=8,
            c="#d3d3d3",
            alpha=0.8,
            label="not measured at both stages",
            zorder=5,
        )
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    _decorate_simplex(ax)
    ax.set_title(rf"$\Delta \alpha$ Index Map" + f"\n$c_{{meas}}={c_meas:.4f}$")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(r"$\min(|i_0-i_1|, M_\alpha-|i_0-i_1|)$")
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.92)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _compute_fixed_norms(
    runs: list[dict[str, np.ndarray | float]],
    tiny_negative_clip_tol: float,
) -> dict[str, Normalize | TwoSlopeNorm]:
    norms: dict[str, Normalize | TwoSlopeNorm] = {}

    for key in ("V1", "V0"):
        values = np.concatenate([np.asarray(row[key], dtype=float).ravel() for row in runs])
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1e-12
        norms[key] = Normalize(vmin=vmin, vmax=vmax)

    for key in ("D1", "D0"):
        values = np.concatenate(
            [_clip_tiny_negative(np.asarray(row[key], dtype=float), tiny_negative_clip_tol).ravel() for row in runs]
        )
        norms[key] = _auto_norm_for_values(values, anchor_zero=True)

    return norms


def _compute_delta_vmax(runs: list[dict[str, np.ndarray | float]]) -> float:
    vals = np.concatenate([np.asarray(row["delta_alpha_idx"], dtype=float).ravel() for row in runs])
    valid = vals >= 0.0
    if not np.any(valid):
        return 1.0
    return float(np.max(vals[valid]))


def _clip_tiny_negative(values: np.ndarray, tiny_negative_clip_tol: float) -> np.ndarray:
    clipped = np.asarray(values, dtype=float).copy()
    tiny_negative = (clipped < 0.0) & (clipped >= -tiny_negative_clip_tol)
    clipped[tiny_negative] = 0.0
    return clipped


def _auto_norm_for_values(values: np.ndarray, anchor_zero: bool) -> Normalize | TwoSlopeNorm:
    arr = np.asarray(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if anchor_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-12
    if anchor_zero and vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def _build_gif_from_frames(frame_dir: Path, out_path: Path, duration_ms: int) -> None:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"no PNG frames found in {frame_dir}")

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    head, tail = images[0], images[1:]
    head.save(
        out_path,
        save_all=True,
        append_images=tail,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for img in images:
        img.close()


def _build_mp4_from_frames(frame_dir: Path, out_path: Path, fps: float, quality: int) -> None:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"no PNG frames found in {frame_dir}")

    first = _ensure_even_frame(imageio.imread(frame_paths[0]))
    target_h, target_w = first.shape[0], first.shape[1]

    writer = imageio.get_writer(
        out_path,
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        quality=int(np.clip(quality, 0, 10)),
        macro_block_size=1,
    )
    try:
        writer.append_data(first)
        for path in frame_paths[1:]:
            frame = _ensure_even_frame(imageio.imread(path))
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = _resize_frame(frame, target_h=target_h, target_w=target_w)
            writer.append_data(frame)
    finally:
        writer.close()


def _ensure_even_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    if h % 2 == 1:
        arr = arr[:-1, :, :]
    if w % 2 == 1:
        arr = arr[:, :-1, :]
    return arr


def _resize_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    img = Image.fromarray(frame)
    img = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(img)


def _decorate_simplex(ax: plt.Axes) -> None:
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, SQRT3_OVER_2],
            [0.0, 0.0],
        ]
    )
    _draw_simplex_guidelines(ax, step=0.1)
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=1.8, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.06, SQRT3_OVER_2 + 0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#fbfbfb")


def _draw_simplex_guidelines(ax: plt.Axes, step: float) -> None:
    if step <= 0.0 or step >= 1.0:
        return
    t_values = np.arange(step, 1.0, step)
    line_kwargs = {"color": "#7f8c8d", "linewidth": 0.55, "alpha": 0.3, "zorder": 2}

    for t in t_values:
        p1 = _bary_to_xy(np.array([t, 1.0 - t, 0.0]))
        p2 = _bary_to_xy(np.array([t, 0.0, 1.0 - t]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **line_kwargs)

        p3 = _bary_to_xy(np.array([1.0 - t, t, 0.0]))
        p4 = _bary_to_xy(np.array([0.0, t, 1.0 - t]))
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], **line_kwargs)

        p5 = _bary_to_xy(np.array([1.0 - t, 0.0, t]))
        p6 = _bary_to_xy(np.array([0.0, 1.0 - t, t]))
        ax.plot([p5[0], p6[0]], [p5[1], p6[1]], **line_kwargs)


def _bary_to_xy(b: np.ndarray) -> np.ndarray:
    x = b[1] + 0.5 * b[2]
    y = SQRT3_OVER_2 * b[2]
    return np.array([x, y], dtype=float)


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "image.interpolation": "none",
        }
    )


if __name__ == "__main__":
    main()
