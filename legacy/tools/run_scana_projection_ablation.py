from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from scana_experiments.chunks import chunks_to_arrays, load_chunks_npz
from scana_experiments.config import load_config
from scana_experiments.constraints import ActionConstraints, batch_project, estimate_constraints
from scana_experiments.metrics import (
    direction_consistency,
    distribution_row,
    estimate_rbf_gamma,
    range_violation_rate,
    second_difference_roughness,
    step_variation,
)
from scana_experiments.models import build_calibrated_pairs, load_checkpoint, sample_generator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "scana_single_arm.json"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_variant(base: ActionConstraints, name: str) -> ActionConstraints:
    if name == "full_projection":
        return base
    if name == "no_amplitude_clip":
        return ActionConstraints(
            action_low=base.action_low,
            action_high=base.action_high,
            max_abs_delta=np.full_like(base.max_abs_delta, 1.0e9),
            smooth_window=base.smooth_window,
        )
    if name == "no_action_range_projection":
        return ActionConstraints(
            action_low=np.full_like(base.action_low, -1.0e9),
            action_high=np.full_like(base.action_high, 1.0e9),
            max_abs_delta=base.max_abs_delta,
            smooth_window=base.smooth_window,
        )
    if name == "no_temporal_smoothing":
        return ActionConstraints(
            action_low=base.action_low,
            action_high=base.action_high,
            max_abs_delta=base.max_abs_delta,
            smooth_window=1,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCANA statistical-bound mapping ablations.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    out_dir = Path(cfg.output_dir)
    dt_train = load_chunks_npz(out_dir / "dt_success_chunks_train.npz")
    real_train = load_chunks_npz(out_dir / "real_success_chunks_train.npz")
    dt_eval = load_chunks_npz(out_dir / "dt_success_chunks_eval.npz")
    real_eval = load_chunks_npz(out_dir / "real_success_chunks_eval.npz")
    train_pairs = build_calibrated_pairs(dt_train, real_train, cfg, max_pairs=cfg.max_train_chunks)
    constraints = estimate_constraints(dt_train + real_train, cfg, train_pairs.target_delta)

    model, scalers, _ = load_checkpoint(out_dir / "noise_generator.pt")
    model.eval()
    import torch

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    eval_arrays = chunks_to_arrays(dt_eval)
    action = eval_arrays["action"]
    condition = eval_arrays["condition"]
    val_pairs = build_calibrated_pairs(dt_eval, real_eval, cfg, max_pairs=cfg.max_eval_chunks)
    target_delta = val_pairs.target_delta
    clean_delta = np.zeros_like(target_delta)
    mmd_gamma = estimate_rbf_gamma(train_pairs.target_delta, seed=cfg.random_seed + 90)
    raw_delta = sample_generator(
        model, scalers, action, condition, cfg, copies=1,
        seed=cfg.random_seed + 421,
    )

    rows: list[dict[str, object]] = []
    variants = [
        "full_projection",
        "no_amplitude_clip",
        "no_action_range_projection",
        "no_temporal_smoothing",
    ]
    for variant in variants:
        c = make_variant(constraints, variant)
        perturbed, projected_delta, infos = batch_project(action, raw_delta, c, alpha=1.0)
        row = distribution_row(
            variant,
            projected_delta[: len(target_delta)],
            target_delta,
            clean_delta,
            gamma=mmd_gamma,
        )
        row.update(
            {
                "raw_range_violation_rate": float(np.mean([x["raw_range_violation_rate"] for x in infos])),
                "projected_fraction": float(np.mean([x["projected_fraction"] for x in infos])),
                "range_violation_against_full_constraints": range_violation_rate(
                    perturbed, constraints.action_low, constraints.action_high
                ),
                "direction_consistency": direction_consistency(action, perturbed),
                "perturbed_action_second_difference_roughness": second_difference_roughness(perturbed),
                "perturbed_action_step_variation": step_variation(perturbed),
                "mean_abs_delta": float(np.mean(np.abs(projected_delta))),
                "max_abs_delta": float(np.max(np.abs(projected_delta))),
            }
        )
        rows.append(row)

    no_projection_action = action + raw_delta
    no_projection_delta = raw_delta
    row = distribution_row(
        "no_projection",
        no_projection_delta[: len(target_delta)],
        target_delta,
        clean_delta,
        gamma=mmd_gamma,
    )
    row.update(
        {
            "raw_range_violation_rate": range_violation_rate(no_projection_action, constraints.action_low, constraints.action_high),
            "projected_fraction": 0.0,
            "range_violation_against_full_constraints": range_violation_rate(
                no_projection_action, constraints.action_low, constraints.action_high
            ),
            "direction_consistency": direction_consistency(action, no_projection_action),
            "perturbed_action_second_difference_roughness": second_difference_roughness(no_projection_action),
            "perturbed_action_step_variation": step_variation(no_projection_action),
            "mean_abs_delta": float(np.mean(np.abs(no_projection_delta))),
            "max_abs_delta": float(np.max(np.abs(no_projection_delta))),
        }
    )
    rows.append(row)

    csv_path = out_dir / "exp421_projection_ablation.csv"
    json_path = out_dir / "exp421_projection_ablation.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
