from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=float)
SQRT3_OVER_2 = np.sqrt(3.0) / 2.0
TWO_PI_OVER_3 = 2.0 * np.pi / 3.0


@dataclass(frozen=True)
class BeliefGrid:
    resolution: int
    beliefs: np.ndarray
    xy: np.ndarray
    lattice: np.ndarray


def make_belief_grid(resolution: int) -> BeliefGrid:
    if resolution <= 0:
        raise ValueError("resolution must be a positive integer")

    lattice = []
    beliefs = []
    for i in range(resolution + 1):
        for j in range(resolution - i + 1):
            k = resolution - i - j
            lattice.append((i, j, k))
            beliefs.append((i / resolution, j / resolution, k / resolution))

    lattice_arr = np.asarray(lattice, dtype=int)
    belief_arr = np.asarray(beliefs, dtype=float)

    b3 = belief_arr[:, 2]
    x = belief_arr[:, 1] + 0.5 * b3
    y = SQRT3_OVER_2 * b3
    xy = np.column_stack((x, y))

    return BeliefGrid(
        resolution=resolution,
        beliefs=belief_arr,
        xy=xy,
        lattice=lattice_arr,
    )


def make_alpha_grid(alpha_samples: int) -> np.ndarray:
    if alpha_samples <= 0:
        raise ValueError("alpha_samples must be a positive integer")
    return TWO_PI_OVER_3 * np.arange(alpha_samples, dtype=float) / alpha_samples


def likelihood(alpha: float, outcome: int, state: int) -> float:
    if not (0 <= outcome <= 2 and 0 <= state <= 2):
        raise ValueError("outcome and state must be in {0, 1, 2}")
    phase = PHASES[state] - alpha - PHASES[outcome]
    value = (1.0 / 3.0) * (1.0 + np.cos(phase))
    return float(np.clip(value, 0.0, 1.0))


def likelihood_table(alphas: np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alphas, dtype=float)
    phase = PHASES[None, None, :] - alpha_arr[:, None, None] - PHASES[None, :, None]
    table = (1.0 / 3.0) * (1.0 + np.cos(phase))
    return np.clip(table, 0.0, 1.0)


def posterior(belief: np.ndarray, alpha: float, outcome: int) -> np.ndarray:
    b = np.asarray(belief, dtype=float)
    like = np.array([likelihood(alpha, outcome, i) for i in range(3)], dtype=float)
    unnorm = b * like
    normalizer = float(np.sum(unnorm))
    if normalizer <= 0.0:
        raise ValueError("observation probability is non-positive")
    return unnorm / normalizer


def one_step_value(belief: np.ndarray, alpha: float) -> float:
    b = np.asarray(belief, dtype=float)
    phase = PHASES[None, :] - alpha - PHASES[:, None]
    like = np.clip((1.0 / 3.0) * (1.0 + np.cos(phase)), 0.0, 1.0)
    weighted = like * b[None, :]
    return float(np.max(weighted, axis=1).sum())


def one_step_curve(belief: np.ndarray, like_table: np.ndarray) -> np.ndarray:
    b = np.asarray(belief, dtype=float)
    weighted = like_table * b[None, None, :]
    return np.max(weighted, axis=2).sum(axis=1)
