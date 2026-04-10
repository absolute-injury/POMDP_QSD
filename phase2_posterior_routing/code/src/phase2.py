from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .core import SQRT3_OVER_2


@dataclass(frozen=True)
class RepresentativeTarget:
    label: str
    role: str
    target: tuple[float, float, float]
    backups: tuple[tuple[float, float, float], ...] = ()


DEFAULT_TARGETS: tuple[RepresentativeTarget, ...] = (
    RepresentativeTarget(
        label="A",
        role="center/symmetry",
        target=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ),
    RepresentativeTarget(
        label="B",
        role="edge-adjacent quasi-binary",
        target=(0.48, 0.48, 0.04),
    ),
    RepresentativeTarget(
        label="C",
        role="near-certainty vertex-adjacent",
        target=(0.90, 0.05, 0.05),
    ),
    RepresentativeTarget(
        label="D",
        role="generic interior asymmetric",
        target=(0.55, 0.30, 0.15),
    ),
    RepresentativeTarget(
        label="E",
        role="off-center interior",
        target=(0.44, 0.28, 0.28),
        backups=((0.46, 0.27, 0.27), (0.42, 0.29, 0.29)),
    ),
)


def load_phase1_npz(path: str) -> dict[str, Any]:
    required = {
        "beliefs",
        "xy",
        "alpha_grid",
        "best_alpha",
        "best_alpha_idx",
        "j1_star",
        "second_best",
        "gain",
        "stopping_value",
        "is_degenerate",
        "tie_tol",
        "lattice",
    }

    with np.load(path, allow_pickle=True) as npz:
        payload = {key: npz[key] for key in npz.files}

    missing = required.difference(payload)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"phase1 npz is missing required fields: {missing_text}")

    if "sanity_json" in payload:
        sanity_json = str(payload["sanity_json"].item())
        try:
            payload["sanity"] = json.loads(sanity_json)
        except json.JSONDecodeError:
            payload["sanity"] = {"raw": sanity_json}

    payload["is_degenerate"] = np.asarray(payload["is_degenerate"]).astype(bool)
    payload["tie_tol"] = float(np.asarray(payload["tie_tol"]).item())
    payload["best_alpha_idx"] = np.asarray(payload["best_alpha_idx"]).astype(int)
    return payload


def run_phase2_posterior_routing(
    phase1: dict[str, Any],
    snap_metric: str = "linf",
    near_tie_gap_threshold: float = 1e-6,
    prob_tol: float = 1e-12,
    posterior_tol: float = 1e-12,
    targets: tuple[RepresentativeTarget, ...] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    if snap_metric not in {"linf", "l1", "l2"}:
        raise ValueError("snap_metric must be one of {'linf', 'l1', 'l2'}")
    if near_tie_gap_threshold <= 0.0:
        raise ValueError("near_tie_gap_threshold must be positive")
    if prob_tol <= 0.0 or posterior_tol <= 0.0:
        raise ValueError("prob_tol and posterior_tol must be positive")

    beliefs = np.asarray(phase1["beliefs"], dtype=float)
    xy = np.asarray(phase1["xy"], dtype=float)
    alpha_grid = np.asarray(phase1["alpha_grid"], dtype=float)
    best_alpha = np.asarray(phase1["best_alpha"], dtype=float)
    best_alpha_idx = np.asarray(phase1["best_alpha_idx"], dtype=int)
    j1_star = np.asarray(phase1["j1_star"], dtype=float)
    second_best = np.asarray(phase1["second_best"], dtype=float)
    gain = np.asarray(phase1["gain"], dtype=float)
    stopping_value = np.asarray(phase1["stopping_value"], dtype=float)
    is_degenerate = np.asarray(phase1["is_degenerate"], dtype=bool)

    argmax_gap = j1_star - second_best
    likelihood_table = _likelihood_table(alpha_grid)
    used_indices: set[int] = set()
    point_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for target_spec in targets:
        if target_spec.label != "E":
            selected = _snap_single_target(
                beliefs=beliefs,
                target=np.asarray(target_spec.target, dtype=float),
                metric=snap_metric,
                used_indices=used_indices,
                source="target",
            )
        else:
            selected = _select_switching_point(
                beliefs=beliefs,
                metric=snap_metric,
                used_indices=used_indices,
                target_spec=target_spec,
                argmax_gap=argmax_gap,
                is_degenerate=is_degenerate,
                near_tie_gap_threshold=near_tie_gap_threshold,
            )
            if not selected["near_tie_satisfied"]:
                warnings.append(
                    "Point E did not satisfy the near-tie threshold with target/backups; "
                    "selected the smallest-gap candidate and flagged this in metadata."
                )
            if selected["source"] != "target":
                warnings.append(
                    f"Point E target replaced by {selected['source']} due to switching/tie validation."
                )

        point_record = _compute_point_record(
            target_spec=target_spec,
            selected=selected,
            beliefs=beliefs,
            xy=xy,
            best_alpha=best_alpha,
            best_alpha_idx=best_alpha_idx,
            j1_star=j1_star,
            gain=gain,
            stopping_value=stopping_value,
            argmax_gap=argmax_gap,
            is_degenerate=is_degenerate,
            likelihood_table=likelihood_table,
            snap_metric=snap_metric,
            prob_tol=prob_tol,
            posterior_tol=posterior_tol,
        )
        point_records.append(point_record)
        used_indices.add(point_record["grid_index"])

    global_checks = _build_global_checks(
        point_records=point_records,
        all_gain=gain,
        near_tie_gap_threshold=near_tie_gap_threshold,
    )
    warnings.extend(global_checks["warnings"])

    return {
        "config": {
            "snap_metric": snap_metric,
            "near_tie_gap_threshold": near_tie_gap_threshold,
            "probability_tol": prob_tol,
            "posterior_tol": posterior_tol,
            "phase1_tie_tol": float(phase1["tie_tol"]),
            "n_beliefs": int(beliefs.shape[0]),
            "n_alpha": int(alpha_grid.shape[0]),
        },
        "points": point_records,
        "point_order": [point["label"] for point in point_records],
        "global_checks": global_checks,
        "warnings": warnings,
    }


def make_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in result["points"]:
        probs = [outcome["probability"] for outcome in point["outcomes"]]
        posteriors = [outcome["posterior_belief"] for outcome in point["outcomes"]]

        row = {
            "point": point["label"],
            "role": point["role"],
            "target_belief": _format_triplet(point["target_belief"]),
            "snapped_belief": _format_triplet(point["snapped_belief"]),
            "alpha_star": point["alpha_star"],
            "alpha_idx": point["alpha_idx"],
            "argmax_gap": point["argmax_gap"],
            "tie_flag": int(point["tie_flag"]),
            "p_1": probs[0],
            "p_2": probs[1],
            "p_3": probs[2],
            "posterior_o1": _format_triplet(posteriors[0]),
            "posterior_o2": _format_triplet(posteriors[1]),
            "posterior_o3": _format_triplet(posteriors[2]),
            "J1_star": point["j1_star"],
            "G": point["gain"],
            "J1_recomputed": point["checks"]["j1_recomputed"],
            "J1_residual": point["checks"]["j1_residual"],
            "interpretation_tag": point["interpretation_tag"],
        }
        rows.append(row)
    return rows


def make_branch_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in result["points"]:
        for outcome in point["outcomes"]:
            bridge = outcome["phase3_bridge"]
            row = {
                "point": point["label"],
                "outcome": outcome["outcome"],
                "probability": outcome["probability"],
                "dominant_hypothesis": outcome["dominant_hypothesis"],
                "start_belief": _format_triplet(point["snapped_belief"]),
                "posterior_belief": _format_triplet(outcome["posterior_belief"]),
                "start_xy": _format_pair(point["start_xy"]),
                "posterior_xy": _format_pair(outcome["posterior_xy"]),
                "displacement_xy": _format_pair(outcome["displacement_xy"]),
                "branch_length": outcome["branch_length"],
                "argmax_gap": point["argmax_gap"],
                "projection_metric": bridge["projection_metric"],
                "projection_index": bridge["nearest_grid_index"],
                "projection_distance": bridge["projection_distance"],
                "projection_belief": _format_triplet(bridge["projected_belief"]),
                "projection_xy": _format_pair(bridge["projected_xy"]),
                "S_posterior": bridge["stopping_value"],
            }
            rows.append(row)
    return rows


def build_interpretation_note(result: dict[str, Any]) -> str:
    point_lookup = {point["label"]: point for point in result["points"]}
    lines = [
        "Phase II posterior routing interpretation notes",
        "",
    ]

    a = point_lookup["A"]
    a_probs = np.array([outcome["probability"] for outcome in a["outcomes"]], dtype=float)
    lines.append(
        "A (center/symmetry): "
        f"probability spread={float(np.max(a_probs) - np.min(a_probs)):.3f}; "
        "branches remain balanced and support a symmetry-consistent reading."
    )

    b = point_lookup["B"]
    b_dom = [outcome["dominant_hypothesis"] for outcome in b["outcomes"]]
    lines.append(
        "B (quasi-binary edge): "
        f"dominant posteriors={b_dom}; routing emphasizes a two-state contest with one low-mass tail."
    )

    c = point_lookup["C"]
    gain_percentile = result["global_checks"]["near_certainty_point_C"]["gain_percentile"]
    lines.append(
        "C (near-certainty): "
        f"gain={c['gain']:.4f} (percentile={gain_percentile:.3f}); panel is consistent with low incremental gain."
    )

    d = point_lookup["D"]
    d_dom = [outcome["dominant_hypothesis"] for outcome in d["outcomes"]]
    lines.append(
        "D (generic interior): "
        f"dominant posteriors split across hypotheses {d_dom}, indicating routing structure beyond confirmation."
    )

    e = point_lookup["E"]
    lines.append(
        "E (off-center interior): "
        f"argmax gap={e['argmax_gap']:.3e}; interpret region boundaries cautiously and use recorded diagnostics."
    )

    if result["warnings"]:
        lines.append("")
        lines.append("Warnings")
        for warning in result["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def _compute_point_record(
    target_spec: RepresentativeTarget,
    selected: dict[str, Any],
    beliefs: np.ndarray,
    xy: np.ndarray,
    best_alpha: np.ndarray,
    best_alpha_idx: np.ndarray,
    j1_star: np.ndarray,
    gain: np.ndarray,
    stopping_value: np.ndarray,
    argmax_gap: np.ndarray,
    is_degenerate: np.ndarray,
    likelihood_table: np.ndarray,
    snap_metric: str,
    prob_tol: float,
    posterior_tol: float,
) -> dict[str, Any]:
    idx = int(selected["index"])
    belief = beliefs[idx]
    start_xy = xy[idx]
    alpha_idx = int(best_alpha_idx[idx])
    alpha_star = float(best_alpha[idx])

    likelihood = likelihood_table[alpha_idx]
    probabilities = likelihood @ belief
    outcomes: list[dict[str, Any]] = []

    for outcome_idx in range(3):
        unnormalized = belief * likelihood[outcome_idx]
        p_o = float(np.sum(unnormalized))
        if p_o <= 0.0:
            posterior = belief.copy()
        else:
            posterior = unnormalized / p_o

        posterior_xy = _belief_to_xy(posterior)
        displacement = posterior_xy - start_xy
        projection_idx, projection_dist = _nearest_index(
            beliefs=beliefs,
            target=posterior,
            metric=snap_metric,
            used_indices=None,
        )

        outcomes.append(
            {
                "outcome": outcome_idx + 1,
                "probability": float(probabilities[outcome_idx]),
                "posterior_belief": posterior.tolist(),
                "posterior_xy": posterior_xy.tolist(),
                "dominant_hypothesis": int(np.argmax(posterior) + 1),
                "displacement_xy": displacement.tolist(),
                "branch_length": float(np.linalg.norm(displacement)),
                "entropy_posterior": _entropy(posterior),
                "max_coordinate_posterior": float(np.max(posterior)),
                "phase3_bridge": {
                    "stopping_value": float(np.max(posterior)),
                    "nearest_grid_index": int(projection_idx),
                    "projected_belief": beliefs[projection_idx].tolist(),
                    "projected_xy": xy[projection_idx].tolist(),
                    "projection_distance": float(projection_dist),
                    "projection_metric": snap_metric,
                },
            }
        )

    posterior_matrix = np.array([row["posterior_belief"] for row in outcomes], dtype=float)
    posterior_sums = np.sum(posterior_matrix, axis=1)
    posterior_sum_residuals = posterior_sums - 1.0
    probability_sum = float(np.sum(probabilities))
    probability_sum_residual = probability_sum - 1.0
    j1_recomputed = float(np.sum(probabilities * np.max(posterior_matrix, axis=1)))
    j1_residual = j1_recomputed - float(j1_star[idx])
    gain_recomputed = j1_recomputed - float(stopping_value[idx])
    gain_residual = gain_recomputed - float(gain[idx])
    j1_tol = max(1e-10, 10.0 * prob_tol)

    interpretation_tag = _interpretation_tag(
        label=target_spec.label,
        outcomes=outcomes,
        argmax_gap=float(argmax_gap[idx]),
        near_tie_threshold=selected.get("near_tie_threshold"),
        near_tie_satisfied=selected.get("near_tie_satisfied"),
    )

    return {
        "label": target_spec.label,
        "role": target_spec.role,
        "source": selected["source"],
        "target_belief": list(target_spec.target),
        "snapped_belief": belief.tolist(),
        "grid_index": idx,
        "snap_metric": snap_metric,
        "snap_distance": float(selected["distance"]),
        "start_xy": start_xy.tolist(),
        "alpha_star": alpha_star,
        "alpha_idx": alpha_idx,
        "argmax_gap": float(argmax_gap[idx]),
        "tie_flag": bool(is_degenerate[idx]),
        "stopping_value": float(stopping_value[idx]),
        "j1_star": float(j1_star[idx]),
        "gain": float(gain[idx]),
        "entropy_prior": _entropy(belief),
        "max_coordinate_prior": float(np.max(belief)),
        "outcomes": outcomes,
        "interpretation_tag": interpretation_tag,
        "switching_validation": {
            "near_tie_threshold": selected.get("near_tie_threshold"),
            "near_tie_satisfied": selected.get("near_tie_satisfied"),
            "candidate_gaps": selected.get("candidate_gaps"),
        },
        "checks": {
            "probability_sum": probability_sum,
            "probability_sum_residual": probability_sum_residual,
            "posterior_sums": posterior_sums.tolist(),
            "posterior_sum_residuals": posterior_sum_residuals.tolist(),
            "j1_recomputed": j1_recomputed,
            "j1_residual": j1_residual,
            "gain_recomputed": gain_recomputed,
            "gain_residual": gain_residual,
            "pass_probability_norm": bool(abs(probability_sum_residual) <= prob_tol),
            "pass_posterior_norm": bool(np.all(np.abs(posterior_sum_residuals) <= posterior_tol)),
            "pass_j1_consistency": bool(abs(j1_residual) <= j1_tol),
        },
    }


def _select_switching_point(
    beliefs: np.ndarray,
    metric: str,
    used_indices: set[int],
    target_spec: RepresentativeTarget,
    argmax_gap: np.ndarray,
    is_degenerate: np.ndarray,
    near_tie_gap_threshold: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    candidate_specs = [("target", target_spec.target)]
    for backup_idx, backup in enumerate(target_spec.backups, start=1):
        candidate_specs.append((f"backup_{backup_idx}", backup))

    for source, coordinate in candidate_specs:
        snapped = _snap_single_target(
            beliefs=beliefs,
            target=np.asarray(coordinate, dtype=float),
            metric=metric,
            used_indices=used_indices,
            source=source,
        )
        idx = int(snapped["index"])
        gap = float(argmax_gap[idx])
        near_tie = bool(is_degenerate[idx] or (gap <= near_tie_gap_threshold))
        snapped["gap"] = gap
        snapped["near_tie"] = near_tie
        candidates.append(snapped)

    for candidate in candidates:
        if candidate["near_tie"]:
            candidate["near_tie_threshold"] = near_tie_gap_threshold
            candidate["near_tie_satisfied"] = True
            candidate["candidate_gaps"] = {
                entry["source"]: entry["gap"] for entry in candidates
            }
            return candidate

    fallback = min(candidates, key=lambda entry: (entry["gap"], entry["distance"], entry["index"]))
    fallback["near_tie_threshold"] = near_tie_gap_threshold
    fallback["near_tie_satisfied"] = False
    fallback["candidate_gaps"] = {entry["source"]: entry["gap"] for entry in candidates}
    return fallback


def _snap_single_target(
    beliefs: np.ndarray,
    target: np.ndarray,
    metric: str,
    used_indices: set[int],
    source: str,
) -> dict[str, Any]:
    index, distance = _nearest_index(
        beliefs=beliefs,
        target=target,
        metric=metric,
        used_indices=used_indices,
    )
    return {
        "index": int(index),
        "distance": float(distance),
        "source": source,
    }


def _nearest_index(
    beliefs: np.ndarray,
    target: np.ndarray,
    metric: str,
    used_indices: set[int] | None,
) -> tuple[int, float]:
    diff = np.abs(beliefs - target[None, :])
    metric_distance = _distance(diff, metric)
    l1_distance = np.sum(diff, axis=1)
    l2_distance = np.sqrt(np.sum(diff * diff, axis=1))
    index_key = np.arange(beliefs.shape[0], dtype=int)

    ranking = np.lexsort((index_key, l2_distance, l1_distance, metric_distance))

    if used_indices is None:
        best = int(ranking[0])
        return best, float(metric_distance[best])

    for candidate in ranking:
        idx = int(candidate)
        if idx not in used_indices:
            return idx, float(metric_distance[idx])

    best = int(ranking[0])
    return best, float(metric_distance[best])


def _distance(diff: np.ndarray, metric: str) -> np.ndarray:
    if metric == "linf":
        return np.max(diff, axis=1)
    if metric == "l1":
        return np.sum(diff, axis=1)
    if metric == "l2":
        return np.sqrt(np.sum(diff * diff, axis=1))
    raise ValueError(f"unsupported metric: {metric}")


def _belief_to_xy(belief: np.ndarray) -> np.ndarray:
    x = belief[1] + 0.5 * belief[2]
    y = SQRT3_OVER_2 * belief[2]
    return np.array([x, y], dtype=float)


def _likelihood_table(alpha_grid: np.ndarray) -> np.ndarray:
    phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=float)
    phase = phases[None, None, :] - alpha_grid[:, None, None] - phases[None, :, None]
    table = (1.0 / 3.0) * (1.0 + np.cos(phase))
    return np.clip(table, 0.0, 1.0)


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    finite = np.isfinite(p) & (p > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(-np.sum(p[finite] * np.log(p[finite])))


def _interpretation_tag(
    label: str,
    outcomes: list[dict[str, Any]],
    argmax_gap: float,
    near_tie_threshold: float | None,
    near_tie_satisfied: bool | None,
) -> str:
    dominant = {row["dominant_hypothesis"] for row in outcomes}
    if label == "A":
        return "center-balanced-branching"
    if label == "B":
        return "edge-quasi-binary-routing"
    if label == "C":
        return "near-certainty-confirmatory"
    if label == "D":
        if len(dominant) >= 2:
            return "interior-routing-multi-destination"
        return "interior-mostly-confirmatory"
    if label == "E":
        if near_tie_satisfied and near_tie_threshold is not None:
            return "off-center-interior-near-tie"
        return f"off-center-interior-caution-gap-{argmax_gap:.2e}"
    return "routing"


def _build_global_checks(
    point_records: list[dict[str, Any]],
    all_gain: np.ndarray,
    near_tie_gap_threshold: float,
) -> dict[str, Any]:
    checks = [point["checks"] for point in point_records]
    warnings: list[str] = []

    probability_residuals = np.array(
        [abs(check["probability_sum_residual"]) for check in checks], dtype=float
    )
    posterior_residuals = np.array(
        [
            np.max(np.abs(check["posterior_sum_residuals"]))
            for check in checks
        ],
        dtype=float,
    )
    j1_residuals = np.array([abs(check["j1_residual"]) for check in checks], dtype=float)

    point_by_label = {point["label"]: point for point in point_records}

    center = point_by_label["A"]
    center_probs = np.array([outcome["probability"] for outcome in center["outcomes"]], dtype=float)
    center_lengths = np.array(
        [outcome["branch_length"] for outcome in center["outcomes"]], dtype=float
    )
    center_symmetry = {
        "probability_spread": float(np.max(center_probs) - np.min(center_probs)),
        "branch_length_spread": float(np.max(center_lengths) - np.min(center_lengths)),
        "dominant_hypotheses": [
            int(outcome["dominant_hypothesis"]) for outcome in center["outcomes"]
        ],
    }

    near_certainty = point_by_label["C"]
    gain_c = float(near_certainty["gain"])
    gain_percentile = float(np.mean(all_gain <= gain_c))
    gain_q25 = float(np.quantile(all_gain, 0.25))
    near_certainty_check = {
        "gain_value": gain_c,
        "gain_percentile": gain_percentile,
        "gain_q25_reference": gain_q25,
        "is_low_gain_region": bool(gain_c <= gain_q25),
    }

    switching = point_by_label["E"]
    switching_validation = switching["switching_validation"]
    near_tie_satisfied = bool(switching_validation.get("near_tie_satisfied", False))
    if not near_tie_satisfied:
        warnings.append(
            f"Point E gap ({switching['argmax_gap']:.3e}) is above near-tie "
            f"threshold ({near_tie_gap_threshold:.3e})."
        )

    return {
        "pass_all_probability_norm": bool(
            np.all([check["pass_probability_norm"] for check in checks])
        ),
        "pass_all_posterior_norm": bool(
            np.all([check["pass_posterior_norm"] for check in checks])
        ),
        "pass_all_j1_consistency": bool(
            np.all([check["pass_j1_consistency"] for check in checks])
        ),
        "max_abs_probability_residual": float(np.max(probability_residuals)),
        "max_abs_posterior_residual": float(np.max(posterior_residuals)),
        "max_abs_j1_residual": float(np.max(j1_residuals)),
        "center_point_A_symmetry": center_symmetry,
        "near_certainty_point_C": near_certainty_check,
        "switching_point_E": {
            "argmax_gap": float(switching["argmax_gap"]),
            "near_tie_threshold": switching_validation.get("near_tie_threshold"),
            "near_tie_satisfied": near_tie_satisfied,
            "candidate_gaps": switching_validation.get("candidate_gaps"),
            "source": switching["source"],
        },
        "warnings": warnings,
    }


def _format_triplet(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"[{arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f}]"


def _format_pair(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"[{arr[0]:.6f}, {arr[1]:.6f}]"
