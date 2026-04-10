from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import BeliefGrid, likelihood_table, make_alpha_grid, make_belief_grid, one_step_curve


@dataclass(frozen=True)
class OneStepMaps:
    belief_grid: BeliefGrid
    alpha_grid: np.ndarray
    likelihood_table: np.ndarray
    stopping_value: np.ndarray
    j1_star: np.ndarray
    gain: np.ndarray
    best_alpha_idx: np.ndarray
    best_alpha: np.ndarray
    second_best: np.ndarray
    is_degenerate: np.ndarray
    tie_tol: float


def solve_one_step_maps(
    resolution: int,
    alpha_samples: int,
    batch_size: int = 512,
    tie_tol: float = 1e-10,
) -> OneStepMaps:
    belief_grid = make_belief_grid(resolution)
    alpha_grid = make_alpha_grid(alpha_samples)
    like_table = likelihood_table(alpha_grid)
    beliefs = belief_grid.beliefs

    n_points = beliefs.shape[0]
    j1_star = np.empty(n_points, dtype=float)
    best_alpha_idx = np.empty(n_points, dtype=int)
    second_best = np.empty(n_points, dtype=float)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch = beliefs[start:end]

        weighted = like_table[None, :, :, :] * batch[:, None, None, :]
        j1_curve = np.max(weighted, axis=3).sum(axis=2)

        idx = np.argmax(j1_curve, axis=1)
        best = np.take_along_axis(j1_curve, idx[:, None], axis=1).ravel()

        if alpha_samples > 1:
            top2 = np.partition(j1_curve, kth=-2, axis=1)[:, -2:]
            second = np.min(top2, axis=1)
        else:
            second = np.full(end - start, -np.inf, dtype=float)

        j1_star[start:end] = best
        best_alpha_idx[start:end] = idx
        second_best[start:end] = second

    stopping_value = np.max(beliefs, axis=1)
    gain = j1_star - stopping_value
    best_alpha = alpha_grid[best_alpha_idx]
    is_degenerate = (j1_star - second_best) < tie_tol

    return OneStepMaps(
        belief_grid=belief_grid,
        alpha_grid=alpha_grid,
        likelihood_table=like_table,
        stopping_value=stopping_value,
        j1_star=j1_star,
        gain=gain,
        best_alpha_idx=best_alpha_idx,
        best_alpha=best_alpha,
        second_best=second_best,
        is_degenerate=is_degenerate,
        tie_tol=tie_tol,
    )


def run_sanity_checks(result: OneStepMaps) -> dict:
    beliefs = result.belief_grid.beliefs
    gains = result.gain
    j1_star = result.j1_star
    lattice = result.belief_grid.lattice

    vertex_indices = []
    for vertex in np.eye(3):
        matches = np.where(np.all(np.isclose(beliefs, vertex[None, :], atol=1e-12), axis=1))[0]
        if matches.size == 0:
            raise RuntimeError("vertex belief is missing from the grid")
        vertex_indices.append(int(matches[0]))
    vertex_gain_max_abs = float(np.max(np.abs(gains[vertex_indices])))

    center_target = np.array([1.0 / 3.0] * 3)
    center_idx = _nearest_belief_index(beliefs, center_target)
    center_curve = one_step_curve(beliefs[center_idx], result.likelihood_table)
    center_range = float(np.max(center_curve) - np.min(center_curve))
    center_best_count = int(np.sum(np.abs(center_curve - np.max(center_curve)) < result.tie_tol))

    permutation_max_diff = _cyclic_symmetry_max_diff(lattice, j1_star)
    gain_min = float(np.min(gains))

    return {
        "vertex_gain_max_abs": vertex_gain_max_abs,
        "center_index": center_idx,
        "center_j1_range_over_alpha": center_range,
        "center_tie_count": center_best_count,
        "cyclic_symmetry_max_diff": permutation_max_diff,
        "gain_min": gain_min,
        "checks": {
            "vertex_gain_near_zero": vertex_gain_max_abs <= 1e-9,
            "center_profile_computed": bool(np.isfinite(center_range)),
            "cyclic_permutation_symmetry": permutation_max_diff <= 1e-10,
            "gain_nonnegative_with_tolerance": gain_min >= -1e-10,
        },
    }


def _nearest_belief_index(beliefs: np.ndarray, target: np.ndarray) -> int:
    dist2 = np.sum((beliefs - target[None, :]) ** 2, axis=1)
    return int(np.argmin(dist2))


def _cyclic_symmetry_max_diff(lattice: np.ndarray, values: np.ndarray) -> float:
    index_of = {tuple(coord.tolist()): i for i, coord in enumerate(lattice)}
    max_diff = 0.0

    for idx, coord in enumerate(lattice):
        cyc = (int(coord[1]), int(coord[2]), int(coord[0]))
        cyc_idx = index_of[cyc]
        diff = abs(float(values[idx] - values[cyc_idx]))
        if diff > max_diff:
            max_diff = diff
    return max_diff
