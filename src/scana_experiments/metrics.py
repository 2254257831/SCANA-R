from __future__ import annotations

import numpy as np


def flatten_chunks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1)


def sample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def estimate_rbf_gamma(reference: np.ndarray, max_rows: int = 1024, seed: int = 2026) -> float:
    """Estimate one RBF gamma from a training-only reference distribution."""

    reference = sample_rows(flatten_chunks(reference), max_rows, seed)
    if len(reference) < 2:
        return 1.0
    distances = _squared_distances(reference, reference)
    upper = distances[np.triu_indices(len(reference), k=1)]
    valid = upper[upper > 0]
    median = float(np.median(valid)) if len(valid) else 1.0
    return 1.0 / max(median, 1.0e-6)


def rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int = 1024,
    seed: int = 2026,
    gamma: float | None = None,
) -> float:
    """Return the biased empirical RBF MMD-squared estimator."""

    x = sample_rows(flatten_chunks(x), max_rows, seed)
    y = sample_rows(flatten_chunks(y), max_rows, seed + 1)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if gamma is None:
        gamma = estimate_rbf_gamma(y, max_rows=max_rows, seed=seed + 11)
    kxx = np.exp(-gamma * _squared_distances(x, x)).mean()
    kyy = np.exp(-gamma * _squared_distances(y, y)).mean()
    kxy = np.exp(-gamma * _squared_distances(x, y)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def diagonal_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_chunks(x)
    y = flatten_chunks(y)
    mean_sq = np.square(x.mean(0) - y.mean(0)).sum()
    scale_sq = np.square(x.std(0) - y.std(0)).sum()
    return float(np.sqrt(mean_sq + scale_sq))


def moment_error(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_chunks(x)
    y = flatten_chunks(y)
    mean_err = np.mean(np.abs(x.mean(0) - y.mean(0)))
    std_err = np.mean(np.abs(x.std(0) - y.std(0)))
    return float(mean_err + std_err)


def second_difference_roughness(chunks: np.ndarray) -> float:
    """Mean norm of the discrete second action difference (not physical jerk)."""

    if chunks.shape[1] < 3:
        return float("nan")
    second = chunks[:, 2:] - 2.0 * chunks[:, 1:-1] + chunks[:, :-2]
    return float(np.mean(np.linalg.norm(second, axis=-1)))


def jerk(chunks: np.ndarray) -> float:
    """Backward-compatible alias; values are discrete second-difference roughness."""

    return second_difference_roughness(chunks)


def step_variation(chunks: np.ndarray) -> float:
    if chunks.shape[1] < 2:
        return float("nan")
    return float(np.mean(np.linalg.norm(np.diff(chunks, axis=1), axis=-1)))


def range_violation_rate(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    outside = np.logical_or(action < low, action > high)
    return float(np.mean(outside))


def direction_consistency(original: np.ndarray, perturbed: np.ndarray) -> float:
    if original.shape[1] < 2 or perturbed.shape[1] < 2:
        return float("nan")
    x = flatten_chunks(np.diff(original, axis=1))
    y = flatten_chunks(np.diff(perturbed, axis=1))
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1.0e-6
    return float(np.mean(np.sum(x * y, axis=1) / denom))


def distribution_row(
    method: str,
    generated_delta: np.ndarray,
    real_delta: np.ndarray,
    clean_delta: np.ndarray | None = None,
    gamma: float | None = None,
) -> dict[str, float | str]:
    mmd = rbf_mmd(generated_delta, real_delta, gamma=gamma)
    base = rbf_mmd(clean_delta, real_delta, gamma=gamma) if clean_delta is not None else np.nan
    reduction = 0.0 if not np.isfinite(base) or base <= 1.0e-9 else (base - mmd) / base * 100.0
    return {
        "method": method,
        "mmd_to_real_delta": mmd,
        "diag_wasserstein": diagonal_wasserstein(generated_delta, real_delta),
        "moment_error": moment_error(generated_delta, real_delta),
        "mmd_reduction_pct_vs_clean": float(reduction),
        "mmd_estimator": "biased_mmd_squared",
        "rbf_gamma": float(gamma) if gamma is not None else np.nan,
        "delta_second_difference_roughness": second_difference_roughness(generated_delta),
        "delta_step_variation": step_variation(generated_delta),
    }
