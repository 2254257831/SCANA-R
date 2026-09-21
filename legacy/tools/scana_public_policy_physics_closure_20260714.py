from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for extra in [ROOT / "tools" / "_public_runtime", ROOT / "code"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
cpu_fallback = ROOT / "tools" / "_translation_runtime"
if str(cpu_fallback) not in sys.path:
    sys.path.append(str(cpu_fallback))

import numpy as np
import pandas as pd
import torch
from torch import nn

from scana_experiments.config import ScanaConfig
from scana_experiments.constraints import ActionConstraints, batch_map
from scana_experiments.metrics import direction_consistency, jerk, range_violation_rate, rbf_mmd
from scana_experiments.models import build_calibrated_pairs, load_checkpoint, sample_generator


OUT = ROOT / "results" / "public_lerobot_aloha_scana_20260714" / "policy_physics_closure"
CACHE = OUT / "cache"
SEEDS = [7, 11, 23]
TASKS = ["Transfer Cube", "Insertion"]
TRAIN_BUDGET = 4096
EVAL_LIMIT = 500
POLICY_STEPS = 300
BATCH = 128


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Standardizer:
    def __init__(self, x: np.ndarray):
        self.mean = x.mean(0).astype(np.float32)
        self.std = np.maximum(x.std(0).astype(np.float32), 1.0e-5)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class ChunkPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Causal state-only input; actions, future states and conditions are excluded.

    Historical outputs before 2026-09-18 used noncausal reconstruction inputs.
    The audited replacement experiment is run by full_repair_v35/run_public.py.
    """
    return np.asarray(arrays["state"][:, 0, :], dtype=np.float32).copy()


def exact_budget(x: np.ndarray, y: np.ndarray, seed: int, n: int = TRAIN_BUDGET) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=len(x) < n)
    return x[idx], y[idx]


def train_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    seed: int,
    steps: int = POLICY_STEPS,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChunkPolicy(train_x.shape[1], train_y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    x = torch.tensor(x_scaler.transform(train_x), dtype=torch.float32, device=device)
    y = torch.tensor(y_scaler.transform(train_y), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(steps):
        idx = rng.integers(0, len(x), size=min(BATCH, len(x)))
        pred = model(x[idx])
        loss = (pred - y[idx]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z = torch.tensor(x_scaler.transform(eval_x), dtype=torch.float32, device=device)
        pred = model(z).cpu().numpy()
    return y_scaler.inverse(pred).astype(np.float32)


def train_residual_regressor(action: np.ndarray, cond: np.ndarray, delta: np.ndarray, seed: int):
    x = np.concatenate([action.reshape(len(action), -1), cond], axis=1).astype(np.float32)
    y = delta.reshape(len(delta), -1).astype(np.float32)
    xs, ys = Standardizer(x), Standardizer(y)
    pred = train_predictor(x, y, x, xs, ys, seed + 900, steps=220)
    return xs, ys, x, pred


def apply_residual_regressor(
    train_action: np.ndarray,
    train_cond: np.ndarray,
    train_delta: np.ndarray,
    query_action: np.ndarray,
    query_cond: np.ndarray,
    seed: int,
) -> np.ndarray:
    x = np.concatenate([train_action.reshape(len(train_action), -1), train_cond], axis=1).astype(np.float32)
    y = train_delta.reshape(len(train_delta), -1).astype(np.float32)
    q = np.concatenate([query_action.reshape(len(query_action), -1), query_cond], axis=1).astype(np.float32)
    xs, ys = Standardizer(x), Standardizer(y)
    pred = train_predictor(x, y, q, xs, ys, seed + 901, steps=260)
    return pred.reshape(query_action.shape).astype(np.float32)


def knn_residual(
    train_action: np.ndarray,
    train_cond: np.ndarray,
    train_delta: np.ndarray,
    query_action: np.ndarray,
    query_cond: np.ndarray,
    seed: int,
    k: int = 8,
) -> np.ndarray:
    def desc(a: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.concatenate([a.mean(1), a.std(1), c], axis=1).astype(np.float32)
    ref, query = desc(train_action, train_cond), desc(query_action, query_cond)
    scale = np.maximum(ref.std(0), 1.0e-5)
    ref, query = (ref - ref.mean(0)) / scale, (query - ref.mean(0)) / scale
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, len(query), 128):
        dist = ((query[start:start + 128, None] - ref[None]) ** 2).sum(2)
        nearest = np.argpartition(dist, kth=min(k, len(ref)) - 1, axis=1)[:, :k]
        pick = nearest[np.arange(len(nearest)), rng.integers(0, nearest.shape[1], size=len(nearest))]
        out.append(train_delta[pick])
    return np.concatenate(out, axis=0).astype(np.float32)


def fit_transition_model(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    x = np.concatenate([state[:, :-1], action[:, :-1], np.ones((*state[:, :-1].shape[:2], 1))], axis=2)
    x = x.reshape(-1, x.shape[-1])
    y = state[:, 1:].reshape(-1, state.shape[-1])
    reg = 1.0e-3 * np.eye(x.shape[1], dtype=np.float32)
    return np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)


def physics_metrics(
    method: str,
    base: np.ndarray,
    candidate: np.ndarray,
    state: np.ndarray,
    transition_w: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    vel_bound: np.ndarray,
    acc_bound: np.ndarray,
) -> dict[str, object]:
    velocity = np.diff(candidate, axis=1)
    acceleration = np.diff(candidate, n=2, axis=1)
    x = np.concatenate([state[:, :-1], candidate[:, :-1], np.ones((*state[:, :-1].shape[:2], 1))], axis=2)
    next_state = np.einsum("ntd,dk->ntk", x, transition_w)
    target_next = state[:, 1:]
    return {
        "method": method,
        "range_violation_rate": range_violation_rate(candidate, action_low, action_high),
        "velocity_envelope_violation_rate": float(np.mean(np.abs(velocity) > vel_bound)),
        "acceleration_envelope_violation_rate": float(np.mean(np.abs(acceleration) > acc_bound)),
        "one_step_state_rmse_proxy": float(np.sqrt(np.mean((next_state - target_next) ** 2))),
        "direction_consistency": direction_consistency(base, candidate),
        "candidate_jerk": jerk(candidate),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        idx = np.flatnonzero(x == value)
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def run_task(task: str, seed: int):
    key = task.lower().replace(" ", "_")
    data = np.load(CACHE / f"{key}.npz", allow_pickle=False)
    dt = {name: data[f"dt_{name}"] for name in ["action", "state", "condition"]}
    real = {name: data[f"real_{name}"] for name in ["action", "state", "condition"]}
    ev = {name: data[f"eval_{name}"] for name in ["action", "state", "condition"]}
    pair_action = data["pair_action"]
    pair_condition = data["pair_condition"]
    pair_delta = data["pair_target_delta"]
    cfg = ScanaConfig(
        output_dir=str(OUT / "tmp" / key / f"seed_{seed}"), action_dim=14, chunk_size=16,
        stride=8, max_train_chunks=1600, max_eval_chunks=500, random_seed=seed,
        hidden_dim=128, train_epochs=80, batch_size=128, learning_rate=2.0e-3,
        weight_decay=1.0e-4, smooth_window=int(data["smooth_window"][0]),
    )
    checkpoint = ROOT / "results" / "public_lerobot_aloha_scana_20260714" / "runs" / task.lower().replace(" ", "_") / f"seed_{seed}" / "noise_generator.pt"
    generator, scalers, _ = load_checkpoint(checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    generated = sample_generator(generator, scalers, dt["action"], dt["condition"], cfg, copies=1)
    constraints = ActionConstraints(
        action_low=data["action_low"], action_high=data["action_high"],
        max_abs_delta=data["max_abs_delta"], smooth_window=int(data["smooth_window"][0]),
    )
    rng = np.random.default_rng(seed + 1000)
    gaussian = rng.normal(0.0, pair_delta.std(0) + 1.0e-6, size=dt["action"].shape).astype(np.float32)
    real_stat = pair_delta[rng.integers(0, len(pair_delta), size=len(dt["action"]))].copy()
    knn = knn_residual(pair_action, pair_condition, pair_delta, dt["action"], dt["condition"], seed)
    residual_reg = apply_residual_regressor(
        pair_action, pair_condition, pair_delta, dt["action"], dt["condition"], seed
    )

    common_x = np.concatenate([features(dt), features(real)], axis=0)
    common_y = np.concatenate([dt["action"], real["action"]], axis=0).reshape(len(common_x), -1)
    x_scaler, y_scaler = Standardizer(common_x), Standardizer(common_y)
    clean_x, clean_y = exact_budget(features(dt), dt["action"].reshape(len(dt["action"]), -1), seed + 41)
    clean_on_real = train_predictor(clean_x, clean_y, features(real), x_scaler, y_scaler, seed + 42)
    dart_errors = real["action"] - clean_on_real.reshape(real["action"].shape)
    dart_mean, dart_std = dart_errors.mean((0, 1)), dart_errors.std((0, 1)) + 1.0e-6
    dart = rng.normal(dart_mean, dart_std, size=dt["action"].shape).astype(np.float32)
    lagged = np.concatenate([dt["action"][:, :1], dt["action"][:, :-1]], axis=1)
    gain = rng.uniform(0.97, 1.03, size=(len(dt["action"]), 1, dt["action"].shape[2])).astype(np.float32)
    actuator_lag = (0.78 * dt["action"] + 0.22 * lagged) * gain - dt["action"]

    residuals = {
        "Gaussian": gaussian,
        "Real-stat resampling": real_stat,
        "DART-style error covariance (offline)": dart,
        "Actuator-lag randomization": actuator_lag,
        "Conditional kNN residual": knn,
        "Deterministic residual regression": residual_reg,
        "SCANA": generated,
    }
    labels = {"Clean DT": dt["action"]}
    for name, delta in residuals.items():
        labels[name] = batch_map(dt["action"], delta, constraints)[0]
    labels["SCANA without mapping"] = dt["action"] + generated

    transitions = fit_transition_model(
        np.concatenate([dt["state"], real["state"]], axis=0),
        np.concatenate([dt["action"], real["action"]], axis=0),
    )
    successful = np.concatenate([dt["action"], real["action"]], axis=0)
    vel_bound = np.quantile(np.abs(np.diff(successful, axis=1)), 0.995, axis=(0, 1)) + 1.0e-6
    acc_bound = np.quantile(np.abs(np.diff(successful, n=2, axis=1)), 0.995, axis=(0, 1)) + 1.0e-6
    policy_rows, physics_rows = [], []
    eval_x, eval_y = features(ev), ev["action"].reshape(len(ev["action"]), -1)
    train_sets = {"Real-only": (features(real), real["action"])}
    train_sets.update({name: (features(dt), value) for name, value in labels.items()})
    for idx, (name, (x0, y0)) in enumerate(train_sets.items()):
        x, y = exact_budget(x0, y0.reshape(len(y0), -1), seed + 100 + idx)
        pred = train_predictor(x, y, eval_x, x_scaler, y_scaler, seed + 200 + idx)
        pred_chunk = pred.reshape(ev["action"].shape)
        policy_rows.append({
            "task": task, "seed": seed, "method": name, "train_chunks": len(x),
            "optimization_steps": POLICY_STEPS, "eval_chunks": len(ev["action"]),
            "action_rmse": float(np.sqrt(np.mean((pred - eval_y) ** 2))),
            "action_mae": float(np.mean(np.abs(pred - eval_y))),
            "predicted_action_mmd": rbf_mmd(pred_chunk, ev["action"], max_rows=256),
            "prediction_jerk": jerk(pred_chunk),
            "target_jerk": jerk(ev["action"]),
            "range_violation_rate": range_violation_rate(pred_chunk, constraints.action_low, constraints.action_high),
            "direction_consistency_to_target": direction_consistency(ev["action"], pred_chunk),
        })
    for name, value in labels.items():
        row = physics_metrics(
            name, dt["action"], value, dt["state"], transitions,
            constraints.action_low, constraints.action_high, vel_bound, acc_bound,
        )
        row.update({"task": task, "seed": seed, "chunks": len(value)})
        physics_rows.append(row)
    return policy_rows, physics_rows


def summarize(rows: list[dict], groups: list[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [c for c in df.columns if c not in groups and c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby(groups, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    OUT.mkdir(parents=True, exist_ok=True)
    policy_rows, physics_rows = [], []
    if not (CACHE / "cache_manifest.json").exists():
        raise FileNotFoundError("Run scana_public_policy_cache_20260714.py before this script.")
    for task in TASKS:
        for seed in SEEDS:
            p, q = run_task(task, seed)
            policy_rows.extend(p)
            physics_rows.extend(q)
    write_rows(OUT / "public_equal4096_policy_per_seed.csv", policy_rows)
    write_rows(OUT / "public_physics_proxy_per_seed.csv", physics_rows)
    policy_summary = summarize(policy_rows, ["task", "method"], OUT / "public_equal4096_policy_summary.csv")
    physics_summary = summarize(physics_rows, ["task", "method"], OUT / "public_physics_proxy_summary.csv")
    merged = pd.DataFrame(policy_rows).merge(pd.DataFrame(physics_rows), on=["task", "seed", "method"], how="inner")
    correlations = []
    for task, frame in merged.groupby("task"):
        for metric in ["range_violation_rate_y", "velocity_envelope_violation_rate", "acceleration_envelope_violation_rate", "one_step_state_rmse_proxy", "candidate_jerk"]:
            correlations.append({"task": task, "mechanism_metric": metric, "outcome": "action_rmse", "spearman_rho": spearman(frame[metric].to_numpy(), frame["action_rmse"].to_numpy()), "points": len(frame)})
    write_rows(OUT / "public_mechanism_correlations.csv", correlations)
    best = policy_summary.sort_values(["task", "action_rmse_mean"]).groupby("task").head(3)
    summary = {
        "status": "completed public equal-budget offline policy and physics-proxy evaluation",
        "train_budget_per_method": TRAIN_BUDGET,
        "policy_optimization_steps": POLICY_STEPS,
        "seeds": SEEDS,
        "tasks": TASKS,
        "policy_model": "compact state-and-trajectory-descriptor chunk MLP; not a VLA and not closed loop",
        "strong_executable_baselines": ["Conditional kNN residual", "Deterministic residual regression"],
        "bounded_analogues": {
            "DART-style error covariance (offline)": "Fits perturbation statistics to errors of a clean behavioral clone on held-out human-simulation training chunks; no displaced-state expert correction.",
            "Actuator-lag randomization": "Applies gain and temporal-lag perturbations to action labels; no simulator parameter randomization or trajectory re-execution.",
        },
        "not_executed_as_canonical_methods": ["online DART", "environment or dynamics randomization", "online expert correction", "public ALOHA closed-loop deployment"],
        "physics_boundary": "Velocity, acceleration, action-range, and learned one-step transition checks are offline consistency proxies, not proof of physical success.",
        "top_three_by_task": best[["task", "method", "action_rmse_mean", "action_rmse_std"]].to_dict("records"),
        "files": {
            "policy_per_seed": "public_equal4096_policy_per_seed.csv",
            "policy_summary": "public_equal4096_policy_summary.csv",
            "physics_per_seed": "public_physics_proxy_per_seed.csv",
            "physics_summary": "public_physics_proxy_summary.csv",
            "mechanism_correlations": "public_mechanism_correlations.csv",
        },
    }
    (OUT / "public_policy_physics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit("Historical runner retired: use revision_work/full_repair_v35_20260918/run_public.py; archived results are noncausal diagnostics.")
