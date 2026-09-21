from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for extra in [ROOT / "tools" / "_public_runtime", ROOT / "tools" / "_translation_runtime"]:
    if str(extra) not in sys.path:
        sys.path.append(str(extra))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from scana_experiments.chunks import cap_chunks, chunks_to_arrays, make_chunks, split_chunks_by_episode
from scana_experiments.config import ScanaConfig
from scana_experiments.constraints import ActionConstraints, batch_map, estimate_constraints
from scana_experiments.io_lerobot import TrajectoryEpisode, episode_content_id
from scana_experiments.metrics import (
    diagonal_wasserstein,
    direction_consistency,
    jerk,
    moment_error,
    range_violation_rate,
    rbf_mmd,
    estimate_rbf_gamma,
    step_variation,
)
from scana_experiments.models import (
    build_calibrated_pairs,
    sample_generator,
    train_noise_generator,
)


DATA_ROOT = ROOT / "Datasets" / "public_lerobot_aloha_scana_20260714"
OUT = ROOT / "results" / "public_lerobot_aloha_scana_20260714"
SEEDS = [7, 11, 23]
TASKS = {
    "Transfer Cube": ("transfer_scripted", "transfer_human"),
    "Insertion": ("insertion_scripted", "insertion_human"),
}
ACTION_DIM = 14
CHUNK = 16
STRIDE = 8
TRAIN_LIMIT = 1600
EVAL_LIMIT = 500
EPOCHS = 80


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_repository(key: str, group: str, task: str) -> list[TrajectoryEpisode]:
    folder = DATA_ROOT / key
    files = sorted((folder / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing public parquet data under {folder}")
    table = pq.read_table(files, columns=["action", "observation.state", "episode_index", "frame_index", "timestamp"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
    frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32).reshape(-1)
    episodes: list[TrajectoryEpisode] = []
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        idx = idx[np.argsort(frame[idx])]
        if len(idx) < CHUNK:
            continue
        episodes.append(
            TrajectoryEpisode(
                path=files[0],
                dataset=key,
                group=group,
                episode_index=int(ep),
                action=action[idx, :ACTION_DIM],
                state=state[idx, :ACTION_DIM],
                timestamp=timestamp[idx],
                frame_index=frame[idx],
                content_id=episode_content_id(action[idx, :ACTION_DIM], state[idx, :ACTION_DIM]),
                task=task,
            )
        )
    return episodes


def split_episodes(episodes: list[TrajectoryEpisode], seed: int = 2026) -> tuple[list[TrajectoryEpisode], list[TrajectoryEpisode]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(episodes))
    rng.shuffle(order)
    cut = max(1, int(round(0.8 * len(order))))
    train_ids = set(int(i) for i in order[:cut])
    return ([e for i, e in enumerate(episodes) if i in train_ids], [e for i, e in enumerate(episodes) if i not in train_ids])


def fit_action_normalizer(episodes: list[TrajectoryEpisode]) -> tuple[np.ndarray, np.ndarray]:
    action = np.concatenate([e.action for e in episodes], axis=0)
    low = np.quantile(action, 0.005, axis=0)
    high = np.quantile(action, 0.995, axis=0)
    center = (low + high) / 2.0
    scale = np.maximum((high - low) / 2.0, 1.0e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize_episodes(episodes: list[TrajectoryEpisode], center: np.ndarray, scale: np.ndarray) -> list[TrajectoryEpisode]:
    rows = []
    for e in episodes:
        rows.append(
            TrajectoryEpisode(
                path=e.path,
                dataset=e.dataset,
                group=e.group,
                episode_index=e.episode_index,
                action=(50.0 * (e.action - center) / scale).astype(np.float32),
                state=(50.0 * (e.state - center) / scale).astype(np.float32),
                timestamp=e.timestamp,
                frame_index=e.frame_index,
                content_id=e.content_id,
                task=e.task,
            )
        )
    return rows


def cfg_for(task_out: Path, seed: int) -> ScanaConfig:
    return ScanaConfig(
        output_dir=str(task_out),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        stride=STRIDE,
        max_train_chunks=TRAIN_LIMIT,
        max_eval_chunks=EVAL_LIMIT,
        random_seed=seed,
        hidden_dim=128,
        train_epochs=EPOCHS,
        batch_size=128,
        learning_rate=2.0e-3,
        weight_decay=1.0e-4,
        smooth_window=3,
    )


def match_rms(delta: np.ndarray, reference: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))
    ref = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return delta if rms < 1.0e-12 else (delta * (ref / rms)).astype(np.float32)


def metrics(
    method: str,
    delta: np.ndarray,
    target: np.ndarray,
    clean: np.ndarray,
    gamma: float,
) -> dict[str, object]:
    return {
        "method": method,
        "paired_residual_mmd": rbf_mmd(delta, target, max_rows=256, gamma=gamma),
        "diag_wasserstein": diagonal_wasserstein(delta, target),
        "moment_error": moment_error(delta, target),
        "mmd_reduction_pct_vs_clean": 100.0
        * (rbf_mmd(clean, target, max_rows=256, gamma=gamma) - rbf_mmd(delta, target, max_rows=256, gamma=gamma))
        / max(rbf_mmd(clean, target, max_rows=256, gamma=gamma), 1.0e-12),
        "mmd_estimator": "biased_mmd_squared",
        "rbf_gamma": gamma,
        "delta_jerk": jerk(delta),
        "delta_step_variation": step_variation(delta),
        "delta_rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
    }


def mapped_metrics(
    method: str,
    action: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    constraints: ActionConstraints,
    gamma: float,
) -> dict[str, object]:
    augmented, mapped, infos = batch_map(action, delta, constraints, alpha=1.0)
    return {
        "method": method,
        "mapped_label_mmd": rbf_mmd(mapped, target, max_rows=256, gamma=gamma),
        "pre_mapping_violation_rate": float(np.mean([x["raw_range_violation_rate"] for x in infos])),
        "post_mapping_violation_rate": range_violation_rate(
            augmented, constraints.action_low, constraints.action_high
        ),
        "mapped_fraction": float(np.mean([x["mapped_fraction"] for x in infos])),
        "direction_consistency": direction_consistency(action, augmented),
        "augmented_action_jerk": jerk(augmented),
        "augmented_step_variation": step_variation(augmented),
        "mean_abs_mapped_delta": float(np.mean(np.abs(mapped))),
    }


def mapping_ablation(
    task: str,
    seed: int,
    action: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    full: ActionConstraints,
    gamma: float,
) -> list[dict[str, object]]:
    variants = {
        "Full mapping": full,
        "No smoothing": ActionConstraints(full.action_low, full.action_high, full.max_abs_delta, 1),
        "No residual clipping": ActionConstraints(
            full.action_low, full.action_high, np.full_like(full.max_abs_delta, 1.0e9), full.smooth_window
        ),
        "No action-range clipping": ActionConstraints(
            np.full_like(full.action_low, -1.0e9), np.full_like(full.action_high, 1.0e9), full.max_abs_delta, full.smooth_window
        ),
    }
    rows = []
    for name, constraints in variants.items():
        augmented, mapped, _ = batch_map(action, delta, constraints)
        rows.append(
            {
                "task": task,
                "seed": seed,
                "variant": name,
                "mapped_label_mmd": rbf_mmd(mapped, target, max_rows=256, gamma=gamma),
                "violation_rate_against_full_bounds": range_violation_rate(
                    augmented, full.action_low, full.action_high
                ),
                "direction_consistency": direction_consistency(action, augmented),
                "action_jerk": jerk(augmented),
                "mean_abs_delta": float(np.mean(np.abs(mapped))),
            }
        )
    augmented = action + delta
    rows.append(
        {
            "task": task,
            "seed": seed,
            "variant": "No mapping",
            "mapped_label_mmd": rbf_mmd(delta, target, max_rows=256, gamma=gamma),
            "violation_rate_against_full_bounds": range_violation_rate(
                augmented, full.action_low, full.action_high
            ),
            "direction_consistency": direction_consistency(action, augmented),
            "action_jerk": jerk(augmented),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
        }
    )
    return rows


def prepare_task(task: str, scripted_key: str, human_key: str) -> tuple[dict[str, list], dict[str, object]]:
    scripted = load_repository(scripted_key, "dt_success", task)
    human = load_repository(human_key, "real_success", task)
    dt_train, dt_eval = split_episodes(scripted, 2026)
    real_train, real_eval = split_episodes(human, 2027)
    center, scale = fit_action_normalizer(dt_train + real_train)
    groups = {
        "dt_train": normalize_episodes(dt_train, center, scale),
        "dt_eval": normalize_episodes(dt_eval, center, scale),
        "real_train": normalize_episodes(real_train, center, scale),
        "real_eval": normalize_episodes(real_eval, center, scale),
    }
    split = {
        "task": task,
        "scripted_repository": scripted_key,
        "human_repository": human_key,
        "scripted_episodes": len(scripted),
        "human_episodes": len(human),
        "scripted_train_episodes": [e.episode_index for e in dt_train],
        "scripted_eval_episodes": [e.episode_index for e in dt_eval],
        "human_train_episodes": [e.episode_index for e in real_train],
        "human_eval_episodes": [e.episode_index for e in real_eval],
        "normalization_center": center.tolist(),
        "normalization_half_range": scale.tolist(),
        "normalized_scale": 50.0,
    }
    return groups, split


def run_task(task: str, groups: dict[str, list]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    pre_rows: list[dict] = []
    post_rows: list[dict] = []
    ablation_rows: list[dict] = []
    train_rows: list[dict] = []
    for seed in SEEDS:
        run_out = OUT / "runs" / task.lower().replace(" ", "_") / f"seed_{seed}"
        cfg = cfg_for(run_out, seed)
        dt_train = cap_chunks(make_chunks(groups["dt_train"], cfg), TRAIN_LIMIT, 2026)
        dt_eval = cap_chunks(make_chunks(groups["dt_eval"], cfg), EVAL_LIMIT, 2028)
        real_train = cap_chunks(make_chunks(groups["real_train"], cfg), TRAIN_LIMIT, 2027)
        real_eval = cap_chunks(make_chunks(groups["real_eval"], cfg), EVAL_LIMIT, 2029)
        train_pairs = build_calibrated_pairs(dt_train, real_train, cfg, max_pairs=TRAIN_LIMIT)
        val_pairs = build_calibrated_pairs(dt_eval, real_eval, cfg, max_pairs=EVAL_LIMIT)
        model, scalers, report = train_noise_generator(train_pairs, val_pairs, cfg, run_out)
        action = val_pairs.action
        target = val_pairs.target_delta
        clean = np.zeros_like(target)
        generated = sample_generator(
            model,
            scalers,
            action,
            val_pairs.condition,
            cfg,
            copies=1,
            seed=seed + 422,
        )
        rng = np.random.default_rng(seed + 100)
        gaussian = rng.normal(0.0, train_pairs.target_delta.std(axis=0) + 1.0e-6, size=target.shape).astype(np.float32)
        sampled = train_pairs.target_delta[rng.integers(0, len(train_pairs.target_delta), size=len(target))].copy()
        natural = {"Clean scripted": clean, "Gaussian": gaussian, "Real-stat resampling": sampled, "SCANA": generated}
        equal_rms = {
            "Gaussian": match_rms(gaussian, generated),
            "Real-stat resampling": match_rms(sampled, generated),
            "SCANA": generated,
        }
        mmd_gamma = estimate_rbf_gamma(train_pairs.target_delta, max_rows=256, seed=seed + 90)
        constraints = estimate_constraints(dt_train + real_train, cfg, train_pairs.target_delta)
        for control, methods in [("natural amplitude", natural), ("SCANA-RMS matched", equal_rms)]:
            for name, delta in methods.items():
                pre = metrics(name, delta, target, clean, mmd_gamma)
                pre.update({"task": task, "seed": seed, "amplitude_control": control, "eval_chunks": len(target)})
                pre_rows.append(pre)
                post = mapped_metrics(name, action, delta, target, constraints, mmd_gamma)
                post.update({"task": task, "seed": seed, "amplitude_control": control, "eval_chunks": len(target)})
                post_rows.append(post)
        ablation_rows.extend(
            mapping_ablation(task, seed, action, generated, target, constraints, mmd_gamma)
        )
        train_rows.append(
            {
                "task": task,
                "seed": seed,
                "train_pairs": report.train_pairs,
                "validation_pairs": report.val_pairs,
                "epochs": report.epochs,
                "device": report.device,
                "final_train_loss": report.final_train_loss,
                "final_validation_loss": report.final_val_loss,
            }
        )
    return pre_rows, post_rows, ablation_rows, train_rows


def aggregate(path: Path, rows: list[dict], groups: list[str]) -> None:
    df = pd.DataFrame(rows)
    numeric = [c for c in df.columns if c not in groups and pd.api.types.is_numeric_dtype(df[c]) and c != "seed"]
    summary = df.groupby(groups, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
    summary.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    OUT.mkdir(parents=True, exist_ok=True)
    if not (DATA_ROOT / "download_manifest.json").exists():
        raise FileNotFoundError("Run scana_public_lerobot_download_20260714.py first.")
    all_pre: list[dict] = []
    all_post: list[dict] = []
    all_ablation: list[dict] = []
    all_train: list[dict] = []
    splits: list[dict] = []
    for task, (scripted_key, human_key) in TASKS.items():
        groups, split = prepare_task(task, scripted_key, human_key)
        splits.append(split)
        pre, post, ablation, train = run_task(task, groups)
        all_pre.extend(pre)
        all_post.extend(post)
        all_ablation.extend(ablation)
        all_train.extend(train)
    write_rows(OUT / "public_pre_mapping_per_seed.csv", all_pre)
    write_rows(OUT / "public_post_mapping_per_seed.csv", all_post)
    write_rows(OUT / "public_mapping_ablation_per_seed.csv", all_ablation)
    write_rows(OUT / "public_generator_training_runs.csv", all_train)
    (OUT / "public_episode_split_manifest.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate(OUT / "public_pre_mapping_summary.csv", all_pre, ["task", "amplitude_control", "method"])
    aggregate(OUT / "public_post_mapping_summary.csv", all_post, ["task", "amplitude_control", "method"])
    aggregate(OUT / "public_mapping_ablation_summary.csv", all_ablation, ["task", "variant"])
    summary = {
        "status": "completed public offline reproduction",
        "dataset": "official LeRobot ALOHA simulated demonstration datasets",
        "tasks": list(TASKS),
        "sources": "scripted demonstrations as idealized source; human demonstrations as variability source",
        "seeds": SEEDS,
        "action_dimension": ACTION_DIM,
        "chunk_horizon": CHUNK,
        "stride": STRIDE,
        "training_limit_per_source": TRAIN_LIMIT,
        "evaluation_limit_per_source": EVAL_LIMIT,
        "epochs": EPOCHS,
        "boundary": "Offline cross-source action-label reproduction only; no public closed-loop or real-robot success claim.",
        "files": {
            "pre_mapping": "public_pre_mapping_per_seed.csv",
            "post_mapping": "public_post_mapping_per_seed.csv",
            "mapping_ablation": "public_mapping_ablation_per_seed.csv",
            "training": "public_generator_training_runs.csv",
        },
    }
    (OUT / "public_reproduction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
