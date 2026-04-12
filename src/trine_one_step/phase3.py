from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .core import likelihood_table


@dataclass(frozen=True)
class TransitionCache:
    obs_probs: np.ndarray
    proj_idx: np.ndarray
    projection_distance: np.ndarray
    cyclic_index: np.ndarray
    edge_a: np.ndarray
    edge_b: np.ndarray
    transition_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase3Run:
    c_meas: float
    V2: np.ndarray
    V1: np.ndarray
    V0: np.ndarray
    D1: np.ndarray
    D0: np.ndarray
    stage1_best_alpha_idx: np.ndarray
    stage1_best_alpha: np.ndarray
    stage1_measure_mask: np.ndarray
    stage0_best_alpha_idx: np.ndarray
    stage0_best_alpha: np.ndarray
    stage0_measure_mask: np.ndarray
    delta_alpha_idx: np.ndarray
    diagnostics: dict[str, Any]


def build_transition_cache(
    beliefs: np.ndarray,
    lattice: np.ndarray,
    alpha_grid: np.ndarray,
    probability_tol: float = 1e-12,
    posterior_tol: float = 1e-12,
    nonneg_tol: float = 1e-12,
    zero_prob_floor: float = 1e-15,
) -> TransitionCache:
    beliefs_arr = np.asarray(beliefs, dtype=float)
    lattice_arr = np.asarray(lattice, dtype=int)
    alpha_arr = np.asarray(alpha_grid, dtype=float)
    resolution = int(np.sum(lattice_arr[0]))

    start = perf_counter()
    like = likelihood_table(alpha_arr)

    unnormalized = beliefs_arr[:, None, None, :] * like[None, :, :, :]
    obs_probs = np.sum(unnormalized, axis=3)

    prob_sum_residual = np.sum(obs_probs, axis=2) - 1.0
    min_prob = float(np.min(obs_probs))

    safe_obs = np.where(obs_probs > zero_prob_floor, obs_probs, 1.0)
    posteriors = unnormalized / safe_obs[..., None]
    zero_mask = obs_probs <= zero_prob_floor
    if np.any(zero_mask):
        posteriors = np.where(zero_mask[..., None], beliefs_arr[:, None, None, :], posteriors)

    posterior_sum_residual = np.sum(posteriors, axis=3) - 1.0
    min_posterior = float(np.min(posteriors))

    proj_idx, projection_distance = _project_posteriors_to_grid(posteriors, resolution)
    cyclic_index = _cyclic_permutation_index(lattice_arr, resolution)
    edge_a, edge_b = _neighbor_edges(lattice_arr, resolution)

    elapsed = perf_counter() - start
    transition_diag = {
        "n_beliefs": int(beliefs_arr.shape[0]),
        "n_alpha": int(alpha_arr.size),
        "probability_normalization": {
            "max_abs_residual": float(np.max(np.abs(prob_sum_residual))),
            "pass": bool(np.max(np.abs(prob_sum_residual)) <= probability_tol),
            "tolerance": float(probability_tol),
        },
        "posterior_normalization": {
            "max_abs_residual": float(np.max(np.abs(posterior_sum_residual))),
            "pass": bool(np.max(np.abs(posterior_sum_residual)) <= posterior_tol),
            "tolerance": float(posterior_tol),
        },
        "nonnegativity": {
            "min_probability": min_prob,
            "min_posterior_coordinate": min_posterior,
            "pass_probability": bool(min_prob >= -nonneg_tol),
            "pass_posterior": bool(min_posterior >= -nonneg_tol),
            "tolerance": float(nonneg_tol),
        },
        "zero_probability_events": int(np.count_nonzero(zero_mask)),
        "projection_audit": {
            "metric": "l2_belief_space",
            "mean_distance": float(np.mean(projection_distance)),
            "max_distance": float(np.max(projection_distance)),
            "p95_distance": float(np.quantile(projection_distance, 0.95)),
        },
        "timing_seconds": {
            "precompute_transition_cache": float(elapsed),
        },
    }

    return TransitionCache(
        obs_probs=obs_probs,
        proj_idx=proj_idx,
        projection_distance=projection_distance,
        cyclic_index=cyclic_index,
        edge_a=edge_a,
        edge_b=edge_b,
        transition_diagnostics=transition_diag,
    )


def solve_phase3_h2(
    beliefs: np.ndarray,
    alpha_grid: np.ndarray,
    cache: TransitionCache,
    c_meas: float,
    decision_tol: float = 1e-12,
    tiny_negative_tol: float = 1e-10,
) -> Phase3Run:
    beliefs_arr = np.asarray(beliefs, dtype=float)
    alpha_arr = np.asarray(alpha_grid, dtype=float)

    start = perf_counter()
    stopping_value = np.max(beliefs_arr, axis=1)
    V2 = stopping_value.copy()

    stage1_start = perf_counter()
    continuation_q1 = -c_meas + np.sum(cache.obs_probs * V2[cache.proj_idx], axis=2)
    q1_best_idx = np.argmax(continuation_q1, axis=1)
    q1_best = continuation_q1[np.arange(continuation_q1.shape[0]), q1_best_idx]
    measure_stage1 = q1_best > (stopping_value + decision_tol)

    V1 = np.where(measure_stage1, q1_best, stopping_value)
    stage1_idx = np.where(measure_stage1, q1_best_idx, -1).astype(int)
    stage1_alpha = np.full(stage1_idx.shape, np.nan, dtype=float)
    stage1_alpha[measure_stage1] = alpha_arr[stage1_idx[measure_stage1]]
    stage1_seconds = perf_counter() - stage1_start

    stage0_start = perf_counter()
    continuation_q0 = -c_meas + np.sum(cache.obs_probs * V1[cache.proj_idx], axis=2)
    q0_best_idx = np.argmax(continuation_q0, axis=1)
    q0_best = continuation_q0[np.arange(continuation_q0.shape[0]), q0_best_idx]
    measure_stage0 = q0_best > (stopping_value + decision_tol)

    V0 = np.where(measure_stage0, q0_best, stopping_value)
    stage0_idx = np.where(measure_stage0, q0_best_idx, -1).astype(int)
    stage0_alpha = np.full(stage0_idx.shape, np.nan, dtype=float)
    stage0_alpha[measure_stage0] = alpha_arr[stage0_idx[measure_stage0]]
    stage0_seconds = perf_counter() - stage0_start

    D1 = V1 - stopping_value
    D0 = V0 - V1
    delta_alpha_idx = _circular_alpha_delta(stage0_idx, stage1_idx, alpha_arr.size)

    elapsed = perf_counter() - start
    diagnostics = _build_run_diagnostics(
        V2=V2,
        V1=V1,
        V0=V0,
        D1=D1,
        D0=D0,
        stage1_idx=stage1_idx,
        stage0_idx=stage0_idx,
        stage1_measure=measure_stage1,
        stage0_measure=measure_stage0,
        cache=cache,
        tiny_negative_tol=tiny_negative_tol,
        decision_tol=decision_tol,
        c_meas=c_meas,
        bellman_seconds=elapsed,
        stage1_seconds=stage1_seconds,
        stage0_seconds=stage0_seconds,
    )

    return Phase3Run(
        c_meas=float(c_meas),
        V2=V2,
        V1=V1,
        V0=V0,
        D1=D1,
        D0=D0,
        stage1_best_alpha_idx=stage1_idx,
        stage1_best_alpha=stage1_alpha,
        stage1_measure_mask=measure_stage1,
        stage0_best_alpha_idx=stage0_idx,
        stage0_best_alpha=stage0_alpha,
        stage0_measure_mask=measure_stage0,
        delta_alpha_idx=delta_alpha_idx,
        diagnostics=diagnostics,
    )


def _build_run_diagnostics(
    V2: np.ndarray,
    V1: np.ndarray,
    V0: np.ndarray,
    D1: np.ndarray,
    D0: np.ndarray,
    stage1_idx: np.ndarray,
    stage0_idx: np.ndarray,
    stage1_measure: np.ndarray,
    stage0_measure: np.ndarray,
    cache: TransitionCache,
    tiny_negative_tol: float,
    decision_tol: float,
    c_meas: float,
    bellman_seconds: float,
    stage1_seconds: float,
    stage0_seconds: float,
) -> dict[str, Any]:
    cycle = cache.cyclic_index
    edge_a = cache.edge_a
    edge_b = cache.edge_b

    symmetry = {
        "max_abs_diff_V2_cyclic": float(np.max(np.abs(V2 - V2[cycle]))),
        "max_abs_diff_V1_cyclic": float(np.max(np.abs(V1 - V1[cycle]))),
        "max_abs_diff_V0_cyclic": float(np.max(np.abs(V0 - V0[cycle]))),
        "max_abs_diff_D1_cyclic": float(np.max(np.abs(D1 - D1[cycle]))),
        "max_abs_diff_D0_cyclic": float(np.max(np.abs(D0 - D0[cycle]))),
    }

    d1_tiny = (D1 < 0.0) & (D1 >= -tiny_negative_tol)
    d0_tiny = (D0 < 0.0) & (D0 >= -tiny_negative_tol)
    d1_large = D1 < -tiny_negative_tol
    d0_large = D0 < -tiny_negative_tol

    stage1_stability = _action_stability(stage1_idx, stage1_measure, edge_a, edge_b)
    stage0_stability = _action_stability(stage0_idx, stage0_measure, edge_a, edge_b)

    branch = {
        "stage1_measure_fraction": float(np.mean(stage1_measure)),
        "stage1_stop_fraction": float(1.0 - np.mean(stage1_measure)),
        "stage0_measure_fraction": float(np.mean(stage0_measure)),
        "stage0_stop_fraction": float(1.0 - np.mean(stage0_measure)),
    }

    return {
        "c_meas": float(c_meas),
        "decision_tol": float(decision_tol),
        "tiny_negative_tol": float(tiny_negative_tol),
        "transition_checks": cache.transition_diagnostics,
        "map_consistency": {
            "min_D1_raw": float(np.min(D1)),
            "min_D0_raw": float(np.min(D0)),
            "tiny_negative_count_D1": int(np.count_nonzero(d1_tiny)),
            "tiny_negative_count_D0": int(np.count_nonzero(d0_tiny)),
            "large_negative_count_D1": int(np.count_nonzero(d1_large)),
            "large_negative_count_D0": int(np.count_nonzero(d0_large)),
        },
        "symmetry_spot_check": symmetry,
        "branch_statistics": branch,
        "action_stability": {
            "stage1": stage1_stability,
            "stage0": stage0_stability,
        },
        "timing_seconds": {
            "stage1_pass": float(stage1_seconds),
            "stage0_pass": float(stage0_seconds),
            "bellman_solve": float(bellman_seconds),
            "total_with_precompute": float(
                bellman_seconds + cache.transition_diagnostics["timing_seconds"]["precompute_transition_cache"]
            ),
        },
    }


def _action_stability(
    best_idx: np.ndarray,
    measure_mask: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
) -> dict[str, Any]:
    if edge_a.size == 0:
        return {
            "neighbor_edges": 0,
            "change_fraction_all_edges": float("nan"),
            "measure_pair_edges": 0,
            "change_fraction_measure_pairs": float("nan"),
        }

    idx_a = best_idx[edge_a]
    idx_b = best_idx[edge_b]
    changed = idx_a != idx_b
    measure_pairs = measure_mask[edge_a] & measure_mask[edge_b]

    if np.any(measure_pairs):
        measure_change = float(np.mean(changed[measure_pairs]))
        measure_count = int(np.count_nonzero(measure_pairs))
    else:
        measure_change = float("nan")
        measure_count = 0

    return {
        "neighbor_edges": int(edge_a.size),
        "change_fraction_all_edges": float(np.mean(changed)),
        "measure_pair_edges": measure_count,
        "change_fraction_measure_pairs": measure_change,
    }


def _project_posteriors_to_grid(
    posteriors: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat = posteriors.reshape(-1, 3)
    projected_lattice = _nearest_lattice_coordinates(flat, resolution)
    proj_idx = _lattice_index(projected_lattice[:, 0], projected_lattice[:, 1], resolution)
    projected_beliefs = projected_lattice.astype(float) / float(resolution)
    distance = np.linalg.norm(flat - projected_beliefs, axis=1)

    return (
        proj_idx.reshape(posteriors.shape[:3]).astype(np.int32),
        distance.reshape(posteriors.shape[:3]),
    )


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
        raise RuntimeError("failed to project posterior to lattice")
    return best


def _lattice_index(i: np.ndarray, j: np.ndarray, resolution: int) -> np.ndarray:
    i64 = np.asarray(i, dtype=np.int64)
    j64 = np.asarray(j, dtype=np.int64)
    return i64 * (resolution + 1) - (i64 * (i64 - 1)) // 2 + j64


def _cyclic_permutation_index(lattice: np.ndarray, resolution: int) -> np.ndarray:
    i = lattice[:, 1]
    j = lattice[:, 2]
    return _lattice_index(i, j, resolution).astype(np.int32)


def _neighbor_edges(lattice: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    edge_a: list[int] = []
    edge_b: list[int] = []
    idx_of = lambda i, j: i * (resolution + 1) - (i * (i - 1)) // 2 + j

    for coord in lattice:
        i = int(coord[0])
        j = int(coord[1])
        k = int(coord[2])
        idx = int(idx_of(i, j))

        if j > 0:
            edge_a.append(idx)
            edge_b.append(int(idx_of(i + 1, j - 1)))
        if k > 0:
            edge_a.append(idx)
            edge_b.append(int(idx_of(i + 1, j)))
            edge_a.append(idx)
            edge_b.append(int(idx_of(i, j + 1)))

    return np.asarray(edge_a, dtype=np.int32), np.asarray(edge_b, dtype=np.int32)


def _circular_alpha_delta(
    stage0_idx: np.ndarray,
    stage1_idx: np.ndarray,
    n_alpha: int,
) -> np.ndarray:
    out = np.full(stage0_idx.shape, -1, dtype=np.int32)
    both_measure = (stage0_idx >= 0) & (stage1_idx >= 0)
    if not np.any(both_measure):
        return out

    a = stage0_idx[both_measure]
    b = stage1_idx[both_measure]
    raw = np.abs(a - b)
    out[both_measure] = np.minimum(raw, n_alpha - raw).astype(np.int32)
    return out
