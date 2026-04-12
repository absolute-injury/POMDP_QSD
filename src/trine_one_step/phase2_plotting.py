from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

from .core import SQRT3_OVER_2


def create_phase2_routing_figure(
    result: dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    probability_label_min: float = 0.10,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    _set_paper_style()

    points = {point["label"]: point for point in result["points"]}
    ordered = [points[label] for label in ["A", "B", "C", "D", "E"]]
    outcome_colors = {1: "#2C5B87", 2: "#9A3E38", 3: "#346A4D"}

    fig = plt.figure(figsize=(24.0, 17.0))
    # Keep simplex panels larger by shrinking the non-simplex legend column.
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.18, 0.78], wspace=0.16, hspace=0.24)
    panel_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, point in zip(panel_axes, ordered):
        _draw_routing_panel(
            ax=ax,
            point=point,
            outcome_colors=outcome_colors,
            probability_label_min=probability_label_min,
        )

    _draw_legend_panel(ax=fig.add_subplot(gs[1, 2]), result=result, outcome_colors=outcome_colors)
    fig.suptitle(r"Posterior Routing Under $\alpha^*(b)$", fontsize=30, y=0.981, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.992, bottom=0.04, top=0.90, wspace=0.16, hspace=0.24)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def create_phase2_diagnostics_figure(
    result: dict[str, Any],
    out_png: Path,
    out_pdf: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    _set_paper_style()

    checks = result["global_checks"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.8, 5.8),
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]},
    )
    _draw_residual_subplot(axes[0], checks)
    _draw_switch_subplot(axes[1], checks["switching_point_E"])
    _draw_summary_subplot(axes[2], checks)
    fig.suptitle(
        "Phase II Consistency and Point E (Off-Center Interior) Diagnostics",
        fontsize=21,
        y=0.995,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.14, top=0.82, wspace=0.32)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def create_phase2_case_figures(
    result: dict[str, Any],
    out_dir: Path,
) -> dict[str, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_paper_style()

    points = {point["label"]: point for point in result["points"]}
    outcome_colors = {1: "#2C5B87", 2: "#9A3E38", 3: "#346A4D"}
    paths: dict[str, dict[str, Path]] = {}

    for label in ["A", "B", "C", "D", "E"]:
        point = points[label]
        png = out_dir / f"figure_D_case_{label}_routing_detail.png"
        pdf = out_dir / f"figure_D_case_{label}_routing_detail.pdf"
        _draw_single_case_figure(point=point, outcome_colors=outcome_colors, out_png=png, out_pdf=pdf)
        paths[label] = {"png": png, "pdf": pdf}
    return paths


def _draw_routing_panel(
    ax: plt.Axes,
    point: dict[str, Any],
    outcome_colors: dict[int, str],
    probability_label_min: float,
) -> None:
    _decorate_simplex(ax)

    start_belief = np.asarray(point["snapped_belief"], dtype=float)
    start_xy = np.asarray(point["start_xy"], dtype=float)
    ax.scatter(
        start_xy[0],
        start_xy[1],
        s=170,
        marker="*",
        c="#E7C96E",
        edgecolors="#222222",
        linewidths=0.7,
        zorder=7,
    )

    # Draw longer branches first to keep short branch labels visible near the start.
    outcomes = sorted(point["outcomes"], key=lambda row: row["branch_length"], reverse=True)
    for outcome in outcomes:
        outcome_id = int(outcome["outcome"])
        color = outcome_colors[outcome_id]
        posterior_belief = np.asarray(outcome["posterior_belief"], dtype=float)
        display_belief = _stretched_belief(start_belief, posterior_belief, stretch=1.45)
        posterior_xy = _belief_to_xy(display_belief)

        direction = posterior_xy - start_xy
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12:
            unit = np.array([1.0, 0.0], dtype=float)
        else:
            unit = direction / dist
        perp = np.array([-unit[1], unit[0]], dtype=float)
        side = {1: 1.0, 2: -1.0, 3: 0.0}[outcome_id]

        ax.annotate(
            "",
            xy=posterior_xy,
            xytext=start_xy,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.55,
                "alpha": 0.9,
            },
            zorder=4,
        )
        ax.scatter(
            posterior_xy[0],
            posterior_xy[1],
            s=43,
            c=color,
            edgecolors="#202020",
            linewidths=0.45,
            zorder=6,
        )

        o_xy = posterior_xy + (0.016 * unit) + (0.010 * side * perp)
        o_label = ax.text(
            o_xy[0],
            o_xy[1],
            f"o{outcome_id}",
            fontsize=18.8,
            color=color,
            fontweight="semibold",
            zorder=8,
        )
        o_label.set_path_effects([pe.withStroke(linewidth=2.1, foreground="white", alpha=0.95)])

        probability = float(outcome["probability"])
        force_show_probability = point.get("label") == "C"
        if force_show_probability or probability >= probability_label_min or dist < 0.12:
            p_xy = start_xy + 0.58 * direction
            dx, dy = _prob_label_offset(outcome_id=outcome_id, branch_length=dist, detailed=False)
            if point.get("label") == "D":
                if outcome_id == 1:
                    dx -= 30.0
                    dy += 20.0
                elif outcome_id == 3:
                    dx += 35.0
                    dy -= 20.0
            p_label = ax.annotate(
                f"p(o{outcome_id})={probability:.2f}",
                xy=(p_xy[0], p_xy[1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=21.5,
                color=color,
                ha="center",
                va="center",
                zorder=9,
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.75,
                    "color": color,
                    "alpha": 0.7,
                },
            )
            p_label.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white", alpha=0.97)])

    snapped = np.asarray(point["snapped_belief"], dtype=float)
    role = _short_role(point["role"])
    ax.set_title(
        f"{point['label']}  {role}\n"
        f"b={_format_belief(snapped)}, alpha*={point['alpha_star']:.3f}, gap={point['argmax_gap']:.2e}",
        fontsize=18.0,
        pad=6.0,
        fontweight="bold",
    )

    if point["source"] != "target":
        ax.text(
            0.98,
            0.99,
            f"source: {point['source']}",
            transform=ax.transAxes,
            fontsize=12.2,
            ha="right",
            va="top",
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.17",
                "facecolor": "#F3F0E2",
                "edgecolor": "#B0A472",
                "linewidth": 0.7,
            },
        )


def _draw_legend_panel(
    ax: plt.Axes,
    result: dict[str, Any],
    outcome_colors: dict[int, str],
) -> None:
    ax.axis("off")
    ax.set_title("Interpretation Guide", fontsize=17.0, pad=7.0, fontweight="bold")

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            markersize=14.8,
            markerfacecolor="#E7C96E",
            markeredgecolor="#222222",
            linestyle="None",
            label="start belief",
        )
    ]
    for outcome, color in outcome_colors.items():
        handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color=color,
                lw=1.8,
                marker="o",
                markersize=8.3,
                markerfacecolor=color,
                markeredgecolor="#202020",
                label=f"outcome {outcome}",
            )
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        fontsize=14.0,
        title="Symbols",
        title_fontsize=14.4,
    )

    checks = result["global_checks"]
    switch = checks["switching_point_E"]
    lines = [
        "Reading notes",
        f"- A spread: {checks['center_point_A_symmetry']['probability_spread']:.3f}",
        f"- C gain percentile: {checks['near_certainty_point_C']['gain_percentile']:.3f}",
        f"- E gap: {switch['argmax_gap']:.2e}",
        f"- E near-tie threshold: {switch['near_tie_threshold']:.2e}",
        f"- E near-tie satisfied: {switch['near_tie_satisfied']}",
        "",
        "See diagnostics figure for",
        "normalization and residual details.",
    ]
    ax.text(
        0.04,
        0.59,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=13.5,
        ha="left",
        va="top",
        color="#2B2F35",
    )


def _draw_residual_subplot(ax: plt.Axes, checks: dict[str, Any]) -> None:
    labels = ["probability", "posterior", "J1"]
    values = np.array(
        [
            checks["max_abs_probability_residual"],
            checks["max_abs_posterior_residual"],
            checks["max_abs_j1_residual"],
        ],
        dtype=float,
    )
    scores = -np.log10(np.maximum(values, 1e-20))
    y = np.arange(len(labels), dtype=float)

    ax.barh(y, scores, color=["#5E7FA8", "#B27D3C", "#5A8E6C"], alpha=0.95)
    ax.set_yticks(y, labels)
    ax.axvline(12.0, linestyle="--", color="#444444", linewidth=0.95, label="tol=1e-12")
    ax.set_xlim(0.0, 20.0)
    ax.set_xlabel(r"$-\log_{10}(\mathrm{max\ residual})$", fontsize=14.2)
    ax.set_title("Consistency Residuals", fontsize=16.6, pad=6.0, fontweight="bold")
    ax.tick_params(labelsize=13.2)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", fontsize=12.4, frameon=False)
    _style_box(ax)


def _draw_switch_subplot(ax: plt.Axes, switch: dict[str, Any]) -> None:
    candidate_gaps = switch.get("candidate_gaps") or {}
    if not candidate_gaps:
        ax.text(0.5, 0.5, "No Point E candidates recorded", ha="center", va="center")
        _style_box(ax)
        return

    labels = list(candidate_gaps.keys())
    values = np.array([candidate_gaps[label] for label in labels], dtype=float)
    x = np.arange(len(labels), dtype=float)
    tick_map = {"target": "T", "backup_1": "B1", "backup_2": "B2"}
    tick_labels = [tick_map.get(label, label) for label in labels]
    selected = switch.get("source")
    colors = ["#9A3E38" if label == selected else "#A0A0A0" for label in labels]

    ax.bar(x, values, color=colors, width=0.72)
    ax.axhline(
        float(switch["near_tie_threshold"]),
        linestyle="--",
        color="#2C5B87",
        linewidth=1.0,
        label="near-tie threshold",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, tick_labels)
    ax.set_ylabel("argmax gap", fontsize=14.2)
    ax.set_title("Point E Candidate Gaps", fontsize=16.6, pad=6.0, fontweight="bold")
    tie_text = "YES" if bool(switch["near_tie_satisfied"]) else "NO"
    ax.text(
        0.98,
        0.95,
        f"near-tie: {tie_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.2,
        bbox={
            "boxstyle": "round,pad=0.17",
            "facecolor": "#F5F5F5",
            "edgecolor": "#B4B4B4",
            "linewidth": 0.7,
        },
    )
    ax.legend(loc="upper left", fontsize=12.2, frameon=False)
    ax.tick_params(labelsize=13.0)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    _style_box(ax)


def _draw_summary_subplot(ax: plt.Axes, checks: dict[str, Any]) -> None:
    ax.axis("off")
    status_rows = [
        ("P(sum_o)", checks["pass_all_probability_norm"]),
        ("P(post norm)", checks["pass_all_posterior_norm"]),
        ("J1 consistency", checks["pass_all_j1_consistency"]),
    ]
    ax.set_title("Checklist", fontsize=16.8, pad=6.0, fontweight="bold")

    y0 = 0.82
    for idx, (label, passed) in enumerate(status_rows):
        face = "#EAF5ED" if passed else "#FBEDEC"
        edge = "#5C8B67" if passed else "#AF6360"
        ax.text(
            0.05,
            y0 - idx * 0.20,
            f"{label}: {'PASS' if passed else 'FAIL'}",
            fontsize=14.3,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.21",
                "facecolor": face,
                "edgecolor": edge,
                "linewidth": 0.8,
            },
        )

    center = checks["center_point_A_symmetry"]
    near_cert = checks["near_certainty_point_C"]
    info = [
        f"A probability spread: {center['probability_spread']:.3f}",
        f"A branch-length spread: {center['branch_length_spread']:.3e}",
        f"C gain value: {near_cert['gain_value']:.4f}",
        f"C gain percentile: {near_cert['gain_percentile']:.3f}",
    ]
    ax.text(0.05, 0.24, "\n".join(info), fontsize=13.2, ha="left", va="top", color="#2E3238")


def _draw_single_case_figure(
    point: dict[str, Any],
    outcome_colors: dict[int, str],
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig = plt.figure(figsize=(21.8, 12.6))
    # Prioritize simplex panel size in per-case detail figures.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.40, 0.96], wspace=0.05)

    ax_route = fig.add_subplot(gs[0, 0])
    _decorate_simplex(ax_route)

    start_belief = np.asarray(point["snapped_belief"], dtype=float)
    start_xy = np.asarray(point["start_xy"], dtype=float)
    ax_route.scatter(
        start_xy[0],
        start_xy[1],
        s=220,
        marker="*",
        c="#E7C96E",
        edgecolors="#222222",
        linewidths=0.8,
        zorder=7,
    )

    outcomes = sorted(point["outcomes"], key=lambda row: row["branch_length"], reverse=True)
    for outcome in outcomes:
        outcome_id = int(outcome["outcome"])
        color = outcome_colors[outcome_id]
        posterior_belief = np.asarray(outcome["posterior_belief"], dtype=float)
        display_belief = _stretched_belief(start_belief, posterior_belief, stretch=1.62)
        posterior_xy = _belief_to_xy(display_belief)
        direction = posterior_xy - start_xy
        dist = float(np.linalg.norm(direction))

        ax_route.annotate(
            "",
            xy=posterior_xy,
            xytext=start_xy,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.1, "alpha": 0.92},
            zorder=4,
        )
        ax_route.scatter(
            posterior_xy[0],
            posterior_xy[1],
            s=72,
            c=color,
            edgecolors="#1f1f1f",
            linewidths=0.5,
            zorder=6,
        )

        # Outcome label near the landing point.
        if dist <= 1e-12:
            unit = np.array([1.0, 0.0], dtype=float)
        else:
            unit = direction / dist
        side = {1: 1.0, 2: -1.0, 3: 0.0}[outcome_id]
        perp = np.array([-unit[1], unit[0]], dtype=float)
        out_xy = posterior_xy + 0.021 * unit + 0.014 * side * perp
        out_text = ax_route.text(
            out_xy[0],
            out_xy[1],
            f"o{outcome_id}",
            fontsize=21.0,
            color=color,
            fontweight="semibold",
            zorder=8,
        )
        out_text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white", alpha=0.95)])

        # Always show branch probability, including short branches (e.g., Case C).
        p_xy = start_xy + 0.58 * direction
        dx, dy = _prob_label_offset(outcome_id=outcome_id, branch_length=dist, detailed=True)
        p_text = ax_route.annotate(
            f"p(o{outcome_id})={float(outcome['probability']):.3f}",
            xy=(p_xy[0], p_xy[1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=24.0,
            color=color,
            ha="center",
            va="center",
            zorder=9,
            arrowprops={"arrowstyle": "-", "lw": 0.95, "color": color, "alpha": 0.72},
        )
        p_text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white", alpha=0.97)])

    snapped = np.asarray(point["snapped_belief"], dtype=float)
    ax_route.set_title(
        f"Case {point['label']}  {_short_role(point['role'])}\n"
        f"b={_format_belief(snapped)}, alpha*={point['alpha_star']:.4f}, "
        f"gap={point['argmax_gap']:.2e}, G={point['gain']:.4f}",
        fontsize=20.0,
        pad=8.0,
        fontweight="bold",
    )

    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.set_title("Branch Details", fontsize=20.0, pad=8.0, fontweight="bold")

    table_data = []
    for outcome in sorted(point["outcomes"], key=lambda row: int(row["outcome"])):
        post = np.asarray(outcome["posterior_belief"], dtype=float)
        table_data.append(
            [
                f"o{int(outcome['outcome'])}",
                f"{float(outcome['probability']):.3f}",
                f"{post[0]:.3f}",
                f"{post[1]:.3f}",
                f"{post[2]:.3f}",
                f"H{int(outcome['dominant_hypothesis'])}",
                f"{float(outcome['branch_length']):.3f}",
            ]
        )

    table = ax_info.table(
        cellText=table_data,
        colLabels=["outcome", "p(o)", "b1'", "b2'", "b3'", "argmax", "|Δ|"],
        loc="upper center",
        colLoc="center",
        cellLoc="center",
        bbox=[0.01, 0.42, 0.98, 0.52],
        colWidths=[0.13, 0.11, 0.17, 0.17, 0.17, 0.14, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13.8)
    table.scale(1.0, 1.36)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.58)
        if row_idx == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")

    probs = [float(outcome["probability"]) for outcome in point["outcomes"]]
    bar_ax = ax_info.inset_axes([0.08, 0.14, 0.86, 0.18])
    x = np.arange(3, dtype=float)
    bar_ax.bar(
        x,
        probs,
        color=[outcome_colors[1], outcome_colors[2], outcome_colors[3]],
        alpha=0.9,
    )
    bar_ax.set_xticks(x, ["o1", "o2", "o3"])
    bar_ax.set_ylim(0.0, max(0.55, max(probs) * 1.15))
    bar_ax.set_title("Outcome probabilities", fontsize=16.6, pad=3.4)
    bar_ax.tick_params(labelsize=14.0)
    bar_ax.grid(axis="y", alpha=0.25, linestyle=":")
    _style_box(bar_ax)

    note_lines = [
        f"source: {point['source']}",
        f"tie flag: {point['tie_flag']}",
        f"J1*: {point['j1_star']:.6f}",
        f"J1 residual: {point['checks']['j1_residual']:.2e}",
    ]
    note_lines.append("routing arrows are visually stretched for readability")
    ax_info.text(0.04, 0.07, "\n".join(note_lines), fontsize=14.8, ha="left", va="top")

    fig.subplots_adjust(left=0.042, right=0.995, bottom=0.062, top=0.91, wspace=0.06)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
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
    ax.plot(triangle[:, 0], triangle[:, 1], color="#1A1A1A", linewidth=1.35, zorder=3)
    ax.text(-0.105, -0.115, r"$b_1=1$", fontsize=22.0, fontweight="semibold", clip_on=False)
    ax.text(1.025, -0.115, r"$b_2=1$", fontsize=22.0, fontweight="semibold", clip_on=False)
    ax.text(0.5, SQRT3_OVER_2 + 0.022, r"$b_3=1$", ha="center", fontsize=22.0, fontweight="semibold", clip_on=False)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.06, SQRT3_OVER_2 + 0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _draw_simplex_guidelines(ax: plt.Axes, step: float) -> None:
    if step <= 0.0 or step >= 1.0:
        return
    style = {"color": "#AEB4BC", "linewidth": 0.43, "alpha": 0.18, "zorder": 1}
    for t in np.arange(step, 1.0, step):
        p1 = _belief_to_xy(np.array([t, 1.0 - t, 0.0], dtype=float))
        p2 = _belief_to_xy(np.array([t, 0.0, 1.0 - t], dtype=float))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)

        p3 = _belief_to_xy(np.array([1.0 - t, t, 0.0], dtype=float))
        p4 = _belief_to_xy(np.array([0.0, t, 1.0 - t], dtype=float))
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], **style)

        p5 = _belief_to_xy(np.array([1.0 - t, 0.0, t], dtype=float))
        p6 = _belief_to_xy(np.array([0.0, 1.0 - t, t], dtype=float))
        ax.plot([p5[0], p6[0]], [p5[1], p6[1]], **style)


def _belief_to_xy(belief: np.ndarray) -> np.ndarray:
    return np.array([belief[1] + 0.5 * belief[2], SQRT3_OVER_2 * belief[2]], dtype=float)


def _format_belief(values: np.ndarray) -> str:
    return f"({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})"


def _short_role(role: str) -> str:
    mapping = {
        "center/symmetry": "Center/Symmetry",
        "edge-adjacent quasi-binary": "Edge Quasi-Binary",
        "near-certainty vertex-adjacent": "Near-Certainty",
        "generic interior asymmetric": "Interior Asymmetric",
        "near switching/near-tie": "Off-Center Interior",
        "off-center interior": "Off-Center Interior",
    }
    return mapping.get(role, role.title())


def _style_box(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")


def _prob_label_offset(outcome_id: int, branch_length: float, detailed: bool) -> tuple[float, float]:
    if detailed:
        if branch_length < 0.11:
            mapping = {1: (-100.0, 66.0), 2: (100.0, 66.0), 3: (20.0, -120.0)}
        else:
            mapping = {1: (-58.0, 38.0), 2: (58.0, 38.0), 3: (0.0, -48.0)}
        return mapping[outcome_id]

    if branch_length < 0.11:
        mapping = {1: (-90.0, 60.0), 2: (90.0, 60.0), 3: (26.0, -124.0)}
    else:
        mapping = {1: (-52.0, 33.0), 2: (52.0, 33.0), 3: (0.0, -42.0)}
    return mapping[outcome_id]


def _stretched_belief(start: np.ndarray, target: np.ndarray, stretch: float) -> np.ndarray:
    s = np.asarray(start, dtype=float)
    t = np.asarray(target, dtype=float)
    delta = t - s

    if stretch <= 1.0:
        return t

    max_scale = np.inf
    eps = 1e-12
    for i in range(3):
        if delta[i] > eps:
            max_scale = min(max_scale, (1.0 - s[i]) / delta[i])
        elif delta[i] < -eps:
            max_scale = min(max_scale, -s[i] / delta[i])

    if not np.isfinite(max_scale):
        max_scale = 1.0

    scale = max(1.0, min(stretch, 0.96 * max_scale))
    stretched = s + scale * delta
    stretched = np.clip(stretched, 0.0, 1.0)
    stretched = stretched / np.sum(stretched)
    return stretched


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "font.size": 15,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
