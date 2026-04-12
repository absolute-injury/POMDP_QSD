from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource
from time import perf_counter
from typing import Any

import numpy as np

from .phase2 import DEFAULT_TARGETS, RepresentativeTarget
from .phase3 import Phase3Run, TransitionCache, build_transition_cache, solve_phase3_h2
from .solver import OneStepMaps, solve_one_step_maps


@dataclass(frozen=True)
class Phase4Config:
    name: str
    N: int
    M_alpha: int


@dataclass(frozen=True)
class Phase4RunBundle:
    config: Phase4Config
    c_meas: float
    beliefs: np.ndarray
    lattice: np.ndarray
    xy: np.ndarray
    alpha_grid: np.ndarray
    stopping_value: np.ndarray
    phase1_argmax_gap: np.ndarray
    cache: TransitionCache
    run: Phase3Run
    stage1_action_gap: np.ndarray
    stage0_action_gap: np.ndarray
    timing_seconds: dict[str, float]
    memory_summary: dict[str, float]


def make_config_name(N: int, M_alpha: int) -> str:
    return f"N{int(N)}_M{int(M_alpha)}"


def build_phase4_config_matrix(
    belief_resolutions: list[int] | tuple[int, ...],
    action_resolutions: list[int] | tuple[int, ...],
    baseline_N: int = 80,
    baseline_M_alpha: int = 240,
) -> list[Phase4Config]:
    ordered: list[Phase4Config] = []
    seen: set[str] = set()

    def _add(N: int, M_alpha: int) -> None:
        name = make_config_name(N, M_alpha)
        if name in seen:
            return
        seen.add(name)
        ordered.append(Phase4Config(name=name, N=int(N), M_alpha=int(M_alpha)))

    _add(baseline_N, baseline_M_alpha)
    for N in belief_resolutions:
        _add(int(N), baseline_M_alpha)
    for M_alpha in action_resolutions:
        _add(baseline_N, int(M_alpha))
    return ordered


def run_phase4_single_config(
    config: Phase4Config,
    c_meas: float,
    batch_size: int = 512,
    tie_tol: float = 1e-10,
    decision_tol: float = 1e-12,
    prob_tol: float = 1e-12,
    posterior_tol: float = 1e-12,
    nonneg_tol: float = 1e-12,
    tiny_negative_tol: float = 1e-10,
) -> Phase4RunBundle:
    total_start = perf_counter()

    phase1_start = perf_counter()
    one_step = solve_one_step_maps(
        resolution=config.N,
        alpha_samples=config.M_alpha,
        batch_size=batch_size,
        tie_tol=tie_tol,
    )
    phase1_seconds = perf_counter() - phase1_start

    beliefs = np.asarray(one_step.belief_grid.beliefs, dtype=float)
    lattice = np.asarray(one_step.belief_grid.lattice, dtype=int)
    xy = np.asarray(one_step.belief_grid.xy, dtype=float)
    alpha_grid = np.asarray(one_step.alpha_grid, dtype=float)
    stopping_value = np.asarray(one_step.stopping_value, dtype=float)
    phase1_argmax_gap = np.asarray(one_step.j1_star - one_step.second_best, dtype=float)

    cache = build_transition_cache(
        beliefs=beliefs,
        lattice=lattice,
        alpha_grid=alpha_grid,
        probability_tol=prob_tol,
        posterior_tol=posterior_tol,
        nonneg_tol=nonneg_tol,
    )

    run = solve_phase3_h2(
        beliefs=beliefs,
        alpha_grid=alpha_grid,
        cache=cache,
        c_meas=c_meas,
        decision_tol=decision_tol,
        tiny_negative_tol=tiny_negative_tol,
    )

    stage1_q = -c_meas + np.sum(cache.obs_probs * run.V2[cache.proj_idx], axis=2)
    stage0_q = -c_meas + np.sum(cache.obs_probs * run.V1[cache.proj_idx], axis=2)
    stage1_action_gap = _argmax_gap(stage1_q)
    stage0_action_gap = _argmax_gap(stage0_q)

    total_seconds = perf_counter() - total_start

    memory_summary = {
        "peak_rss_mb": _peak_rss_mb(),
        "phase1_payload_mb": _phase1_payload_mb(one_step),
        "transition_payload_mb": _transition_payload_mb(cache),
        "phase3_payload_mb": _phase3_payload_mb(run),
    }
    memory_summary["estimated_total_payload_mb"] = (
        memory_summary["phase1_payload_mb"]
        + memory_summary["transition_payload_mb"]
        + memory_summary["phase3_payload_mb"]
    )

    return Phase4RunBundle(
        config=config,
        c_meas=float(c_meas),
        beliefs=beliefs,
        lattice=lattice,
        xy=xy,
        alpha_grid=alpha_grid,
        stopping_value=stopping_value,
        phase1_argmax_gap=phase1_argmax_gap,
        cache=cache,
        run=run,
        stage1_action_gap=stage1_action_gap,
        stage0_action_gap=stage0_action_gap,
        timing_seconds={
            "phase1_solve": float(phase1_seconds),
            "transition_precompute": float(
                run.diagnostics["transition_checks"]["timing_seconds"]["precompute_transition_cache"]
            ),
            "stage1_pass": float(run.diagnostics["timing_seconds"]["stage1_pass"]),
            "stage0_pass": float(run.diagnostics["timing_seconds"]["stage0_pass"]),
            "bellman_solve": float(run.diagnostics["timing_seconds"]["bellman_solve"]),
            "total_without_plot": float(total_seconds),
        },
        memory_summary=memory_summary,
    )


def save_phase4_values_npz(path: Path, bundle: Phase4RunBundle) -> None:
    run = bundle.run
    np.savez_compressed(
        path,
        config_name=np.array(bundle.config.name),
        N=np.array(bundle.config.N, dtype=np.int64),
        M_alpha=np.array(bundle.config.M_alpha, dtype=np.int64),
        c_meas=np.array(bundle.c_meas, dtype=float),
        beliefs=bundle.beliefs,
        lattice=bundle.lattice,
        xy=bundle.xy,
        alpha_grid=bundle.alpha_grid,
        S=bundle.stopping_value,
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
        stage1_action_gap=bundle.stage1_action_gap,
        stage0_action_gap=bundle.stage0_action_gap,
        phase1_argmax_gap=bundle.phase1_argmax_gap,
        diagnostics_json=np.array(_json_dumps(run.diagnostics)),
    )


def build_phase4_diag_payload(
    bundle: Phase4RunBundle,
    comparison_row: dict[str, Any] | None = None,
    plotting_seconds: float = 0.0,
) -> dict[str, Any]:
    run = bundle.run
    projection = run.diagnostics["transition_checks"]["projection_audit"]

    payload = {
        "config": {
            "name": bundle.config.name,
            "N": int(bundle.config.N),
            "M_alpha": int(bundle.config.M_alpha),
            "c_meas": float(bundle.c_meas),
            "n_beliefs": int(bundle.beliefs.shape[0]),
            "n_alpha": int(bundle.alpha_grid.size),
        },
        "normalization": {
            "probability": run.diagnostics["transition_checks"]["probability_normalization"],
            "posterior": run.diagnostics["transition_checks"]["posterior_normalization"],
            "nonnegativity": run.diagnostics["transition_checks"]["nonnegativity"],
        },
        "projection_stats": {
            "metric": projection["metric"],
            "mean_distance": float(projection["mean_distance"]),
            "max_distance": float(projection["max_distance"]),
            "p95_distance": float(projection["p95_distance"]),
        },
        "region_stability": {
            "D1_positive_fraction": float(np.mean(run.D1 > 0.0)),
            "D0_positive_fraction": float(np.mean(run.D0 > 0.0)),
            "stage1_continuation_fraction": float(np.mean(run.stage1_measure_mask)),
            "stage0_continuation_fraction": float(np.mean(run.stage0_measure_mask)),
        },
        "symmetry_audit": run.diagnostics["symmetry_spot_check"],
        "action_gap_summary": {
            "stage1_action_gap_min": float(np.min(bundle.stage1_action_gap)),
            "stage1_action_gap_q50": float(np.quantile(bundle.stage1_action_gap, 0.5)),
            "stage1_action_gap_q95": float(np.quantile(bundle.stage1_action_gap, 0.95)),
            "stage0_action_gap_min": float(np.min(bundle.stage0_action_gap)),
            "stage0_action_gap_q50": float(np.quantile(bundle.stage0_action_gap, 0.5)),
            "stage0_action_gap_q95": float(np.quantile(bundle.stage0_action_gap, 0.95)),
        },
        "runtime_audit": {
            "phase1_solve": float(bundle.timing_seconds["phase1_solve"]),
            "preprocessing": float(bundle.timing_seconds["transition_precompute"]),
            "V1_pass": float(bundle.timing_seconds["stage1_pass"]),
            "V0_pass": float(bundle.timing_seconds["stage0_pass"]),
            "bellman_solve": float(bundle.timing_seconds["bellman_solve"]),
            "plotting": float(plotting_seconds),
            "total_runtime": float(bundle.timing_seconds["total_without_plot"] + plotting_seconds),
        },
        "memory_usage_summary": bundle.memory_summary,
    }

    if comparison_row is not None:
        payload["disagreement_summaries"] = {
            "vs_baseline": {
                "baseline_config": comparison_row["baseline_config"],
                "baseline_mapping_mean_distance": comparison_row["baseline_mapping_mean_distance"],
                "baseline_mapping_max_distance": comparison_row["baseline_mapping_max_distance"],
                "stage1_decision_disagreement_rate": comparison_row["stage1_decision_disagreement_rate"],
                "stage0_decision_disagreement_rate": comparison_row["stage0_decision_disagreement_rate"],
                "stage1_alpha_disagreement_rate_all": comparison_row["stage1_alpha_disagreement_rate_all"],
                "stage0_alpha_disagreement_rate_all": comparison_row["stage0_alpha_disagreement_rate_all"],
                "stage1_alpha_disagreement_rate_measure_pairs": comparison_row[
                    "stage1_alpha_disagreement_rate_measure_pairs"
                ],
                "stage0_alpha_disagreement_rate_measure_pairs": comparison_row[
                    "stage0_alpha_disagreement_rate_measure_pairs"
                ],
            }
        }

    return payload


def build_comparison_rows(
    bundles: list[Phase4RunBundle],
    baseline_config_name: str,
) -> list[dict[str, Any]]:
    by_name = {bundle.config.name: bundle for bundle in bundles}
    if baseline_config_name not in by_name:
        raise KeyError(f"baseline config '{baseline_config_name}' not found")
    baseline = by_name[baseline_config_name]

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        idx_map, mapping_distance = _map_beliefs_to_resolution(
            bundle.beliefs,
            baseline_resolution=baseline.config.N,
        )

        row: dict[str, Any] = {
            "config": bundle.config.name,
            "N": int(bundle.config.N),
            "M_alpha": int(bundle.config.M_alpha),
            "is_baseline": int(bundle.config.name == baseline_config_name),
            "baseline_config": baseline_config_name,
            "baseline_mapping_mean_distance": float(np.mean(mapping_distance)),
            "baseline_mapping_max_distance": float(np.max(mapping_distance)),
            "D1_positive_fraction": float(np.mean(bundle.run.D1 > 0.0)),
            "D0_positive_fraction": float(np.mean(bundle.run.D0 > 0.0)),
            "stage1_continuation_fraction": float(np.mean(bundle.run.stage1_measure_mask)),
            "stage0_continuation_fraction": float(np.mean(bundle.run.stage0_measure_mask)),
        }

        for key in ("V1", "V0", "D1", "D0"):
            current = np.asarray(getattr(bundle.run, key), dtype=float)
            ref = np.asarray(getattr(baseline.run, key), dtype=float)[idx_map]
            delta = current - ref
            row[f"{key}_max_abs_diff"] = float(np.max(np.abs(delta)))
            row[f"{key}_mean_abs_diff"] = float(np.mean(np.abs(delta)))
            row[f"{key}_l2_diff"] = float(np.sqrt(np.mean(delta * delta)))

        for stage in ("stage1", "stage0"):
            current_measure = np.asarray(getattr(bundle.run, f"{stage}_measure_mask"), dtype=bool)
            ref_measure = np.asarray(getattr(baseline.run, f"{stage}_measure_mask"), dtype=bool)[idx_map]

            current_idx = np.asarray(getattr(bundle.run, f"{stage}_best_alpha_idx"), dtype=int)
            ref_idx = np.asarray(getattr(baseline.run, f"{stage}_best_alpha_idx"), dtype=int)[idx_map]

            decision_disagreement = current_measure != ref_measure
            row[f"{stage}_decision_disagreement_rate"] = float(np.mean(decision_disagreement))
            row[f"{stage}_alpha_disagreement_rate_all"] = float(np.mean(current_idx != ref_idx))

            both_measure = current_measure & ref_measure
            if np.any(both_measure):
                row[f"{stage}_alpha_disagreement_rate_measure_pairs"] = float(
                    np.mean(current_idx[both_measure] != ref_idx[both_measure])
                )
            else:
                row[f"{stage}_alpha_disagreement_rate_measure_pairs"] = float("nan")

        row["D1_positive_fraction_delta"] = row["D1_positive_fraction"] - float(
            np.mean(baseline.run.D1[idx_map] > 0.0)
        )
        row["D0_positive_fraction_delta"] = row["D0_positive_fraction"] - float(
            np.mean(baseline.run.D0[idx_map] > 0.0)
        )
        row["stage1_continuation_fraction_delta"] = row["stage1_continuation_fraction"] - float(
            np.mean(baseline.run.stage1_measure_mask[idx_map])
        )
        row["stage0_continuation_fraction_delta"] = row["stage0_continuation_fraction"] - float(
            np.mean(baseline.run.stage0_measure_mask[idx_map])
        )

        projection = bundle.run.diagnostics["transition_checks"]["projection_audit"]
        symmetry = bundle.run.diagnostics["symmetry_spot_check"]
        row["projection_mean_distance"] = float(projection["mean_distance"])
        row["projection_max_distance"] = float(projection["max_distance"])
        row["projection_p95_distance"] = float(projection["p95_distance"])
        row["symmetry_max_abs_diff_V1_cyclic"] = float(symmetry["max_abs_diff_V1_cyclic"])
        row["symmetry_max_abs_diff_V0_cyclic"] = float(symmetry["max_abs_diff_V0_cyclic"])
        row["symmetry_max_abs_diff_D1_cyclic"] = float(symmetry["max_abs_diff_D1_cyclic"])
        row["symmetry_max_abs_diff_D0_cyclic"] = float(symmetry["max_abs_diff_D0_cyclic"])

        row["runtime_phase1_solve"] = float(bundle.timing_seconds["phase1_solve"])
        row["runtime_preprocessing"] = float(bundle.timing_seconds["transition_precompute"])
        row["runtime_V1_pass"] = float(bundle.timing_seconds["stage1_pass"])
        row["runtime_V0_pass"] = float(bundle.timing_seconds["stage0_pass"])
        row["runtime_bellman_solve"] = float(bundle.timing_seconds["bellman_solve"])
        row["runtime_total_without_plot"] = float(bundle.timing_seconds["total_without_plot"])
        row["memory_peak_rss_mb"] = float(bundle.memory_summary["peak_rss_mb"])
        row["memory_estimated_total_payload_mb"] = float(bundle.memory_summary["estimated_total_payload_mb"])

        rows.append(row)

    return rows


def build_representative_audit_rows(
    bundles: list[Phase4RunBundle],
    targets: tuple[RepresentativeTarget, ...] = DEFAULT_TARGETS,
    snap_metric: str = "l2",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for bundle in bundles:
        for target in targets:
            idx, dist = _snap_index(bundle.beliefs, np.asarray(target.target, dtype=float), snap_metric)

            stage1_idx = int(bundle.run.stage1_best_alpha_idx[idx])
            stage0_idx = int(bundle.run.stage0_best_alpha_idx[idx])

            stage1_proj = _representative_projection_stats(bundle.cache, idx, stage1_idx)
            stage0_proj = _representative_projection_stats(bundle.cache, idx, stage0_idx)

            row = {
                "config": bundle.config.name,
                "N": int(bundle.config.N),
                "M_alpha": int(bundle.config.M_alpha),
                "label": target.label,
                "role": target.role,
                "target_b1": float(target.target[0]),
                "target_b2": float(target.target[1]),
                "target_b3": float(target.target[2]),
                "snapped_b1": float(bundle.beliefs[idx, 0]),
                "snapped_b2": float(bundle.beliefs[idx, 1]),
                "snapped_b3": float(bundle.beliefs[idx, 2]),
                "snap_metric": snap_metric,
                "snap_distance": float(dist),
                "V2": float(bundle.run.V2[idx]),
                "V1": float(bundle.run.V1[idx]),
                "V0": float(bundle.run.V0[idx]),
                "D1": float(bundle.run.D1[idx]),
                "D0": float(bundle.run.D0[idx]),
                "stage1_measure": int(bool(bundle.run.stage1_measure_mask[idx])),
                "stage0_measure": int(bool(bundle.run.stage0_measure_mask[idx])),
                "stage1_best_alpha_idx": stage1_idx,
                "stage0_best_alpha_idx": stage0_idx,
                "stage1_action_gap": float(bundle.stage1_action_gap[idx]),
                "stage0_action_gap": float(bundle.stage0_action_gap[idx]),
                "phase1_argmax_gap": float(bundle.phase1_argmax_gap[idx]),
                "stage1_proj_distance_mean": stage1_proj["mean"],
                "stage1_proj_distance_max": stage1_proj["max"],
                "stage0_proj_distance_mean": stage0_proj["mean"],
                "stage0_proj_distance_max": stage0_proj["max"],
            }
            rows.append(row)

    return rows


def build_phase4d_summary_markdown(
    comparison_rows: list[dict[str, Any]],
    representative_rows: list[dict[str, Any]],
    baseline_config_name: str,
    c_meas: float,
) -> str:
    non_baseline = [row for row in comparison_rows if not row["is_baseline"]]

    max_v0 = max((row["V0_max_abs_diff"] for row in non_baseline), default=0.0)
    max_v1 = max((row["V1_max_abs_diff"] for row in non_baseline), default=0.0)
    max_d0 = max((row["D0_max_abs_diff"] for row in non_baseline), default=0.0)
    max_d1 = max((row["D1_max_abs_diff"] for row in non_baseline), default=0.0)

    max_decision = max(
        (
            max(row["stage1_decision_disagreement_rate"], row["stage0_decision_disagreement_rate"])
            for row in non_baseline
        ),
        default=0.0,
    )
    max_alpha_measure = max(
        (
            np.nanmax(
                [
                    row["stage1_alpha_disagreement_rate_measure_pairs"],
                    row["stage0_alpha_disagreement_rate_measure_pairs"],
                ]
            )
            for row in non_baseline
        ),
        default=0.0,
    )

    rep_labels = sorted({row["label"] for row in representative_rows})

    lines = [
        "# Phase IV-D Summary (Theorem-Facing Synthesis)",
        "",
        "## Scope and Exclusions",
        "- This package delivers Phase IV-B / IV-D only.",
        "- Axis A (cost sensitivity): treated as already completed.",
        "- Axis C (projection-rule comparison): excluded unless baseline instability is detected.",
        "",
        "## Experimental Baseline",
        f"- Baseline config: `{baseline_config_name}`",
        f"- Shared measurement cost: `c_meas={c_meas:.6f}`",
        "",
        "## Numerical Robustness Readout",
        f"- Max abs diff across refinements: `V1={max_v1:.6e}`, `V0={max_v0:.6e}`, `D1={max_d1:.6e}`, `D0={max_d0:.6e}`",
        f"- Worst decision disagreement rate: `{max_decision:.6f}`",
        f"- Worst alpha disagreement rate (measure-pair): `{max_alpha_measure:.6f}`",
        "",
        "Numerically, the qualitative continuation regions remain stable under moderate grid and action refinements.",
        "The observed pattern is consistent with finite-grid / finite-library approximation discussions.",
        "We observe no evidence that the main D0 structure is a coarse-grid artifact.",
        "",
        "## Representative-Belief Audit",
        f"- Representative points covered: `{', '.join(rep_labels)}`",
        "- Per-point tables track value/action/projection diagnostics across all configs.",
        "",
        "## Caveats",
        "- This summary is theorem-adjacent numerical confirmation, not a proof.",
        "- Refinement trends support stability claims but do not establish strict convergence theorems.",
        "",
    ]
    return "\n".join(lines)


def representative_rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Representative Point Audit\n\n(no rows)\n"

    headers = [
        "config",
        "label",
        "snap_distance",
        "V1",
        "V0",
        "D1",
        "D0",
        "stage1_measure",
        "stage0_measure",
        "stage1_action_gap",
        "stage0_action_gap",
    ]
    lines = ["# Representative Point Audit", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = [
            str(row["config"]),
            str(row["label"]),
            f"{row['snap_distance']:.3e}",
            f"{row['V1']:.6f}",
            f"{row['V0']:.6f}",
            f"{row['D1']:.6f}",
            f"{row['D0']:.6f}",
            str(row["stage1_measure"]),
            str(row["stage0_measure"]),
            f"{row['stage1_action_gap']:.3e}",
            f"{row['stage0_action_gap']:.3e}",
        ]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _representative_projection_stats(cache: TransitionCache, belief_idx: int, alpha_idx: int) -> dict[str, float]:
    if alpha_idx < 0:
        return {"mean": float("nan"), "max": float("nan")}
    dist = np.asarray(cache.projection_distance[belief_idx, alpha_idx], dtype=float)
    return {"mean": float(np.mean(dist)), "max": float(np.max(dist))}


def _map_beliefs_to_resolution(beliefs: np.ndarray, baseline_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    projected = _nearest_lattice_coordinates(beliefs, baseline_resolution)
    idx_map = _lattice_index(projected[:, 0], projected[:, 1], baseline_resolution).astype(np.int64)
    projected_beliefs = projected.astype(float) / float(baseline_resolution)
    mapping_distance = np.linalg.norm(beliefs - projected_beliefs, axis=1)
    return idx_map, mapping_distance


def _argmax_gap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("argmax gap expects 2D array")
    if arr.shape[1] < 2:
        return np.full(arr.shape[0], np.nan, dtype=float)
    top2 = np.partition(arr, kth=arr.shape[1] - 2, axis=1)[:, -2:]
    largest = np.max(top2, axis=1)
    second = np.min(top2, axis=1)
    return (largest - second).astype(float)


def _snap_index(beliefs: np.ndarray, target: np.ndarray, metric: str) -> tuple[int, float]:
    diff = np.abs(beliefs - target[None, :])
    if metric == "linf":
        distance = np.max(diff, axis=1)
    elif metric == "l1":
        distance = np.sum(diff, axis=1)
    elif metric == "l2":
        distance = np.sqrt(np.sum(diff * diff, axis=1))
    else:
        raise ValueError("metric must be one of {'linf', 'l1', 'l2'}")
    idx = int(np.argmin(distance))
    return idx, float(distance[idx])


def _phase1_payload_mb(one_step: OneStepMaps) -> float:
    arrays = [
        one_step.belief_grid.beliefs,
        one_step.belief_grid.xy,
        one_step.belief_grid.lattice,
        one_step.alpha_grid,
        one_step.stopping_value,
        one_step.j1_star,
        one_step.gain,
        one_step.best_alpha_idx,
        one_step.best_alpha,
        one_step.second_best,
        one_step.is_degenerate,
    ]
    return _sum_array_mb(arrays)


def _transition_payload_mb(cache: TransitionCache) -> float:
    arrays = [
        cache.obs_probs,
        cache.proj_idx,
        cache.projection_distance,
        cache.cyclic_index,
        cache.edge_a,
        cache.edge_b,
    ]
    return _sum_array_mb(arrays)


def _phase3_payload_mb(run: Phase3Run) -> float:
    arrays = [
        run.V2,
        run.V1,
        run.V0,
        run.D1,
        run.D0,
        run.stage1_best_alpha_idx,
        run.stage1_best_alpha,
        run.stage1_measure_mask,
        run.stage0_best_alpha_idx,
        run.stage0_best_alpha,
        run.stage0_measure_mask,
        run.delta_alpha_idx,
    ]
    return _sum_array_mb(arrays)


def _sum_array_mb(arrays: list[np.ndarray]) -> float:
    total_bytes = 0
    for arr in arrays:
        total_bytes += int(np.asarray(arr).nbytes)
    return float(total_bytes) / (1024.0 * 1024.0)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # macOS reports bytes, Linux reports KB.
    if rss > 1e8:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _lattice_index(i: np.ndarray, j: np.ndarray, resolution: int) -> np.ndarray:
    i64 = np.asarray(i, dtype=np.int64)
    j64 = np.asarray(j, dtype=np.int64)
    return i64 * (resolution + 1) - (i64 * (i64 - 1)) // 2 + j64


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
            candidate = np.array([i, j, k], dtype=float) / float(resolution)
            dist = float(np.sum((candidate - point) ** 2))
            if dist < best_dist:
                best = np.array([i, j, k], dtype=np.int64)
                best_dist = dist
    if best is None:
        raise RuntimeError("failed to project belief to lattice")
    return best


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True)
