from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunks import ActionChunk
from .config import ScanaConfig


@dataclass
class ActionConstraints:
    action_low: np.ndarray
    action_high: np.ndarray
    max_abs_delta: np.ndarray
    smooth_window: int = 3


def estimate_constraints(
    chunks: list[ActionChunk],
    cfg: ScanaConfig,
    residual_samples: np.ndarray,
) -> ActionConstraints:
    """Estimate action quantiles and the paired-action-difference envelope.

    ``residual_samples`` must use the same residual target that trains the
    generator.  This prevents the manuscript's matched action difference from
    being silently replaced by the unrelated action-state tracking difference.
    """

    if residual_samples.ndim != 3 or residual_samples.shape[-1] != cfg.action_dim:
        raise ValueError(
            "residual_samples must have shape [N, H, action_dim]; "
            f"received {residual_samples.shape}."
        )
    actions = np.concatenate([c.action for c in chunks], axis=0)
    action_low = np.quantile(actions, cfg.action_bound_low_quantile, axis=0).astype(np.float32)
    action_high = np.quantile(actions, cfg.action_bound_high_quantile, axis=0).astype(np.float32)
    residual_flat = residual_samples.reshape(-1, cfg.action_dim)
    max_abs_delta = np.quantile(
        np.abs(residual_flat), cfg.max_delta_quantile, axis=0
    ).astype(np.float32)
    return ActionConstraints(
        action_low=action_low,
        action_high=action_high,
        max_abs_delta=max_abs_delta,
        smooth_window=cfg.smooth_window,
    )


def smooth_delta(delta: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or delta.shape[0] < 3:
        return delta
    radius = max(1, window // 2)
    padded = np.pad(delta, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(delta)
    for i in range(delta.shape[0]):
        out[i] = padded[i : i + 2 * radius + 1].mean(axis=0)
    return out


def map_delta(
    action: np.ndarray,
    delta: np.ndarray,
    constraints: ActionConstraints,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the fixed ordered statistical-bound map to one residual chunk."""

    raw = alpha * delta
    smoothed = smooth_delta(raw, constraints.smooth_window)
    bounded_delta = np.clip(smoothed, -constraints.max_abs_delta, constraints.max_abs_delta)
    perturbed = np.clip(action + bounded_delta, constraints.action_low, constraints.action_high)
    final_delta = perturbed - action
    raw_outside = np.logical_or(action + raw < constraints.action_low, action + raw > constraints.action_high)
    mapped_fraction = float(np.mean(np.abs(final_delta - raw) > 1e-6))
    info = {
        "raw_range_violation_rate": float(np.mean(raw_outside)),
        "mapped_fraction": mapped_fraction,
        # Backward-compatible result key for previously generated artifacts.
        "projected_fraction": mapped_fraction,
        "mean_abs_delta": float(np.mean(np.abs(final_delta))),
        "max_abs_delta": float(np.max(np.abs(final_delta))),
    }
    return perturbed.astype(np.float32), final_delta.astype(np.float32), info


def batch_map(
    actions: np.ndarray,
    deltas: np.ndarray,
    constraints: ActionConstraints,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    perturbed, final_delta, infos = [], [], []
    for action, delta in zip(actions, deltas):
        a_tilde, d_tilde, info = map_delta(action, delta, constraints, alpha=alpha)
        perturbed.append(a_tilde)
        final_delta.append(d_tilde)
        infos.append(info)
    return np.stack(perturbed), np.stack(final_delta), infos


def project_delta(
    action: np.ndarray,
    delta: np.ndarray,
    constraints: ActionConstraints,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compatibility wrapper for artifacts that used the former function name."""

    return map_delta(action, delta, constraints, alpha=alpha)


def batch_project(
    actions: np.ndarray,
    deltas: np.ndarray,
    constraints: ActionConstraints,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Compatibility wrapper for artifacts that used the former function name."""

    return batch_map(actions, deltas, constraints, alpha=alpha)
