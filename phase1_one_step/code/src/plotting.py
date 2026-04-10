from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .core import SQRT3_OVER_2, TWO_PI_OVER_3
from .solver import OneStepMaps


def create_standard_figures(result: OneStepMaps, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_publication_style()
    paths = {
        "j1_star": out_dir / "figure_A_j1_star.png",
        "gain": out_dir / "figure_B_gain.png",
        "alpha_star": out_dir / "figure_C_alpha_star.png",
    }

    _plot_scalar_map(
        result,
        values=result.j1_star,
        title=r"Figure A: Optimal One-Step Value $J_1^*(b)$",
        colorbar_label=r"$J_1^*(b)$",
        cmap="cividis",
        out_path=paths["j1_star"],
        contour_levels=14,
        mark_center=True,
    )
    _plot_scalar_map(
        result,
        values=result.gain,
        title=r"Figure B: One-Step Gain $G(b)=J_1^*(b)-S(b)$",
        colorbar_label=r"$G(b)$",
        cmap="inferno",
        out_path=paths["gain"],
        contour_levels=16,
        mark_center=True,
    )
    _plot_scalar_map(
        result,
        values=result.best_alpha,
        title=r"Figure C: Best Orientation $\alpha^*(b)$",
        colorbar_label=r"$\alpha^*(b)$ (radian)",
        cmap="twilight_shifted",
        out_path=paths["alpha_star"],
        vmin=0.0,
        vmax=TWO_PI_OVER_3,
        alpha_ticks=[0.0, np.pi / 3.0, TWO_PI_OVER_3],
        alpha_ticklabels=["0", r"$\pi/3$", r"$2\pi/3$"],
        show_degenerate=True,
        contour_levels=12,
        mark_center=True,
    )
    return paths


def _plot_scalar_map(
    result: OneStepMaps,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha_ticks: list[float] | None = None,
    alpha_ticklabels: list[str] | None = None,
    show_degenerate: bool = False,
    contour_levels: int = 0,
    mark_center: bool = False,
) -> None:
    x = result.belief_grid.xy[:, 0]
    y = result.belief_grid.xy[:, 1]
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(12.0, 10.2))
    mesh = ax.tripcolor(
        tri,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=False,
    )

    if contour_levels > 0:
        contour = ax.tricontour(
            tri,
            values,
            levels=contour_levels,
            colors="white",
            linewidths=0.5,
            alpha=0.35,
        )
        contour.set_zorder(3)

    _decorate_simplex(ax)

    if show_degenerate:
        mask = result.is_degenerate
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                s=14,
                facecolors="none",
                edgecolors="black",
                alpha=0.8,
                linewidths=0.55,
                label="near-tie region",
                zorder=5,
            )
            ax.legend(loc="upper right", frameon=True, fontsize=13)

    if mark_center:
        center_xy = np.array([0.5, SQRT3_OVER_2 / 3.0])
        ax.scatter(
            center_xy[0],
            center_xy[1],
            s=54,
            marker="*",
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, shrink=0.95)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=14)
    if alpha_ticks is not None:
        cbar.set_ticks(alpha_ticks)
    if alpha_ticklabels is not None:
        cbar.set_ticklabels(alpha_ticklabels)

    ax.set_title(title)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
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
        # Constant b1=t
        p1 = _bary_to_xy(np.array([t, 1.0 - t, 0.0]))
        p2 = _bary_to_xy(np.array([t, 0.0, 1.0 - t]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **line_kwargs)

        # Constant b2=t
        p3 = _bary_to_xy(np.array([1.0 - t, t, 0.0]))
        p4 = _bary_to_xy(np.array([0.0, t, 1.0 - t]))
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], **line_kwargs)

        # Constant b3=t
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
