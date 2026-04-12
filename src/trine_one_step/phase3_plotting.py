from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from .core import SQRT3_OVER_2, TWO_PI_OVER_3
from .phase3 import Phase3Run


def create_phase3_figures(
    run: Phase3Run,
    xy: np.ndarray,
    out_dir: Path,
    suffix: str,
    tiny_negative_clip_tol: float = 1e-10,
    include_optional: bool = True,
) -> dict[str, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_publication_style()

    paths: dict[str, dict[str, Path]] = {}
    scalar_specs = [
        ("V1", run.V1, r"Phase III-A: $V_1(b)$", r"$V_1(b)$", "cividis"),
        ("V0", run.V0, r"Phase III-B: $V_0(b)$", r"$V_0(b)$", "viridis"),
    ]
    for key, values, title, cbar_label, cmap in scalar_specs:
        png_path = out_dir / f"phase3_fig_{key}_{suffix}.png"
        pdf_path = out_dir / f"phase3_fig_{key}_{suffix}.pdf"
        _plot_scalar_map(
            xy=xy,
            values=values,
            title=title,
            colorbar_label=cbar_label,
            cmap=cmap,
            png_path=png_path,
            pdf_path=pdf_path,
        )
        paths[key] = {"png": png_path, "pdf": pdf_path}

    diff_specs = [
        ("D1", run.D1, r"Phase III-C: $D_1(b)=V_1(b)-S(b)$", r"$D_1(b)$"),
        ("D0", run.D0, r"Phase III-D: $D_0(b)=V_0(b)-V_1(b)$", r"$D_0(b)$"),
    ]
    for key, values, title, cbar_label in diff_specs:
        png_path = out_dir / f"phase3_fig_{key}_{suffix}.png"
        pdf_path = out_dir / f"phase3_fig_{key}_{suffix}.pdf"
        _plot_difference_map(
            xy=xy,
            values=values,
            title=title,
            colorbar_label=cbar_label,
            png_path=png_path,
            pdf_path=pdf_path,
            tiny_negative_clip_tol=tiny_negative_clip_tol,
        )
        paths[key] = {"png": png_path, "pdf": pdf_path}

    if include_optional:
        optional_specs = [
            (
                "action_V1",
                run.stage1_best_alpha,
                run.stage1_measure_mask,
                r"Phase III-E (Optional): Best Action Map for $V_1$",
                r"$\alpha^*_{V_1}(b)$ (radian)",
            ),
            (
                "action_V0",
                run.stage0_best_alpha,
                run.stage0_measure_mask,
                r"Phase III-F (Optional): Best Action Map for $V_0$",
                r"$\alpha^*_{V_0}(b)$ (radian)",
            ),
        ]
        for key, alpha_values, measure_mask, title, cbar_label in optional_specs:
            png_path = out_dir / f"phase3_fig_{key}_{suffix}.png"
            pdf_path = out_dir / f"phase3_fig_{key}_{suffix}.pdf"
            _plot_best_action_map(
                xy=xy,
                best_alpha=alpha_values,
                measure_mask=measure_mask,
                title=title,
                colorbar_label=cbar_label,
                png_path=png_path,
                pdf_path=pdf_path,
            )
            paths[key] = {"png": png_path, "pdf": pdf_path}

        png_path = out_dir / f"phase3_fig_delta_alpha_idx_{suffix}.png"
        pdf_path = out_dir / f"phase3_fig_delta_alpha_idx_{suffix}.pdf"
        _plot_delta_alpha_idx_map(
            xy=xy,
            delta_alpha_idx=run.delta_alpha_idx,
            title=r"Phase III (Optional): $\Delta \alpha$ Index Between $V_0$ and $V_1$",
            colorbar_label=r"$\min(|i_0-i_1|, M_\alpha-|i_0-i_1|)$",
            png_path=png_path,
            pdf_path=pdf_path,
        )
        paths["delta_alpha_idx"] = {"png": png_path, "pdf": pdf_path}
    return paths


def _plot_scalar_map(
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    mesh = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap)
    contour = ax.tricontour(tri, values, levels=14, colors="white", linewidths=0.45, alpha=0.35)
    contour.set_zorder(3)
    _decorate_simplex(ax)
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_difference_map(
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    png_path: Path,
    pdf_path: Path,
    tiny_negative_clip_tol: float,
) -> None:
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    tri = mtri.Triangulation(x, y)

    clipped = np.asarray(values, dtype=float).copy()
    tiny_negative = (clipped < 0.0) & (clipped >= -tiny_negative_clip_tol)
    clipped[tiny_negative] = 0.0

    min_val = float(np.min(clipped))
    max_val = float(np.max(clipped))
    if max_val <= 0.0:
        max_val = 1e-12
    vmin = min(0.0, min_val)
    vmax = max(0.0, max_val)
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-12
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    mesh = ax.tripcolor(tri, clipped, shading="gouraud", cmap="RdYlBu_r", norm=norm)
    contour = ax.tricontour(tri, clipped, levels=14, colors="black", linewidths=0.33, alpha=0.28)
    contour.set_zorder(3)
    _decorate_simplex(ax)
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"tiny negatives clipped: {int(np.count_nonzero(tiny_negative))}",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        color="#2b2b2b",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#b0b0b0",
            "linewidth": 0.6,
            "alpha": 0.82,
        },
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_action_map(
    xy: np.ndarray,
    best_alpha: np.ndarray,
    measure_mask: np.ndarray,
    title: str,
    colorbar_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    tri = mtri.Triangulation(x, y)
    alpha_values = np.asarray(best_alpha, dtype=float)
    measured = np.asarray(measure_mask, dtype=bool)
    masked_alpha = np.ma.masked_where(~measured, alpha_values)

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    mesh = ax.tripcolor(
        tri,
        masked_alpha,
        shading="gouraud",
        cmap="twilight_shifted",
        vmin=0.0,
        vmax=TWO_PI_OVER_3,
    )
    if np.any(~measured):
        ax.scatter(
            x[~measured],
            y[~measured],
            s=10,
            c="#202020",
            alpha=0.85,
            label="stop",
            zorder=5,
        )
        ax.legend(loc="upper right", frameon=True, fontsize=12)

    _decorate_simplex(ax)
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"measure fraction: {float(np.mean(measured)):.4f}",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        color="#2b2b2b",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#b0b0b0",
            "linewidth": 0.6,
            "alpha": 0.82,
        },
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(colorbar_label)
    cbar.set_ticks([0.0, np.pi / 3.0, TWO_PI_OVER_3])
    cbar.set_ticklabels(["0", r"$\pi/3$", r"$2\pi/3$"])
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_alpha_idx_map(
    xy: np.ndarray,
    delta_alpha_idx: np.ndarray,
    title: str,
    colorbar_label: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    tri = mtri.Triangulation(x, y)
    delta = np.asarray(delta_alpha_idx, dtype=float)
    valid = delta >= 0.0
    masked = np.ma.masked_where(~valid, delta)

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    mesh = ax.tripcolor(
        tri,
        masked,
        shading="gouraud",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.max(delta[valid])) if np.any(valid) else 1.0,
    )
    if np.any(~valid):
        ax.scatter(
            x[~valid],
            y[~valid],
            s=10,
            c="#d9d9d9",
            alpha=0.85,
            label="not measured at both stages",
            zorder=5,
        )
        ax.legend(loc="upper right", frameon=True, fontsize=11)

    _decorate_simplex(ax)
    ax.set_title(title)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


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
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=2.0, zorder=4)

    ax.text(-0.055, -0.04, r"$b_1=1$", fontsize=18, fontweight="semibold")
    ax.text(1.01, -0.04, r"$b_2=1$", ha="left", fontsize=18, fontweight="semibold")
    ax.text(0.5, SQRT3_OVER_2 + 0.05, r"$b_3=1$", ha="center", fontsize=18, fontweight="semibold")

    ax.text(0.5, -0.082, r"$b_3=0$", ha="center", fontsize=14, alpha=0.9)
    ax.text(0.03, 0.40, r"$b_2=0$", rotation=60, fontsize=14, alpha=0.9)
    ax.text(0.92, 0.40, r"$b_1=0$", rotation=-60, fontsize=14, alpha=0.9)

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
    line_kwargs = {"color": "#7f8c8d", "linewidth": 0.7, "alpha": 0.35, "zorder": 2}

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
            "axes.titlesize": 25,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "font.size": 16,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "image.interpolation": "none",
        }
    )
