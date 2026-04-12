from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def create_phase4_compare_figures(
    comparison_rows: list[dict],
    out_dir: Path,
) -> dict[str, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_publication_style()

    rows = sorted(comparison_rows, key=lambda row: (int(row["N"]), int(row["M_alpha"]), str(row["config"])))
    labels = [str(row["config"]) for row in rows]
    x = np.arange(len(rows), dtype=float)

    paths: dict[str, dict[str, Path]] = {}

    value_png = out_dir / "phase4B_fig_compare_values.png"
    value_pdf = out_dir / "phase4B_fig_compare_values.pdf"
    _plot_values(rows=rows, x=x, labels=labels, out_png=value_png, out_pdf=value_pdf)
    paths["values"] = {"png": value_png, "pdf": value_pdf}

    policy_png = out_dir / "phase4B_fig_compare_policy.png"
    policy_pdf = out_dir / "phase4B_fig_compare_policy.pdf"
    _plot_policy(rows=rows, x=x, labels=labels, out_png=policy_png, out_pdf=policy_pdf)
    paths["policy"] = {"png": policy_png, "pdf": policy_pdf}

    region_png = out_dir / "phase4B_fig_compare_region.png"
    region_pdf = out_dir / "phase4B_fig_compare_region.pdf"
    _plot_region(rows=rows, x=x, labels=labels, out_png=region_png, out_pdf=region_pdf)
    paths["region"] = {"png": region_png, "pdf": region_pdf}

    return paths


def _plot_values(rows: list[dict], x: np.ndarray, labels: list[str], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))
    specs = [
        ("V1_max_abs_diff", "Max |ΔV1|", "#33658A"),
        ("V0_max_abs_diff", "Max |ΔV0|", "#2F4858"),
        ("D1_max_abs_diff", "Max |ΔD1|", "#BC4749"),
        ("D0_max_abs_diff", "Max |ΔD0|", "#A44A3F"),
    ]

    for ax, (key, title, color) in zip(axes.ravel(), specs):
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        ax.bar(x, y, color=color, alpha=0.9)
        _highlight_baseline(ax, rows, x)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("difference")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Phase IV-B Comparison: Value Differences vs Baseline", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_policy(rows: list[dict], x: np.ndarray, labels: list[str], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))
    specs = [
        ("stage1_decision_disagreement_rate", "Stage 1 decision disagreement", "#3A5A40"),
        ("stage0_decision_disagreement_rate", "Stage 0 decision disagreement", "#588157"),
        ("stage1_alpha_disagreement_rate_measure_pairs", "Stage 1 alpha disagreement (measure pairs)", "#7F5539"),
        ("stage0_alpha_disagreement_rate_measure_pairs", "Stage 0 alpha disagreement (measure pairs)", "#9C6644"),
    ]

    for ax, (key, title, color) in zip(axes.ravel(), specs):
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        y = np.where(np.isnan(y), 0.0, y)
        ax.bar(x, y, color=color, alpha=0.9)
        _highlight_baseline(ax, rows, x)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("rate")
        ax.set_ylim(0.0, max(1.0e-12, float(np.max(y)) * 1.15))
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Phase IV-B Comparison: Policy Stability vs Baseline", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_region(rows: list[dict], x: np.ndarray, labels: list[str], out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))
    specs = [
        ("D1_positive_fraction", "D1 > 0 fraction", "#1D3557"),
        ("D0_positive_fraction", "D0 > 0 fraction", "#457B9D"),
        ("stage1_continuation_fraction", "Stage 1 continuation fraction", "#6D597A"),
        ("stage0_continuation_fraction", "Stage 0 continuation fraction", "#B56576"),
    ]

    for ax, (key, title, color) in zip(axes.ravel(), specs):
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        ax.bar(x, y, color=color, alpha=0.9)
        _highlight_baseline(ax, rows, x)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Phase IV-B Comparison: Region Stability", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _highlight_baseline(ax: plt.Axes, rows: list[dict], x: np.ndarray) -> None:
    for idx, row in enumerate(rows):
        if int(row["is_baseline"]) == 1:
            ax.axvline(x[idx], color="#111111", linestyle="--", linewidth=1.0, alpha=0.6)
            break


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
