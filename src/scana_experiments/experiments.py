from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .chunks import (
    ActionChunk,
    assert_no_content_overlap,
    cap_chunks,
    chunks_to_arrays,
    load_chunks_npz,
    make_chunks,
    split_chunks_by_episode,
)
from .config import ScanaConfig, config_to_jsonable, save_config
from .constraints import ActionConstraints, batch_map, estimate_constraints
from .io_lerobot import deduplicate_episodes_by_content, load_episodes, summarize_episodes
from .metrics import (
    direction_consistency,
    distribution_row,
    estimate_rbf_gamma,
    range_violation_rate,
    rbf_mmd,
    second_difference_roughness,
    step_variation,
)
from .models import (
    NoiseTrainingData,
    build_calibrated_pairs,
    sample_generator,
    train_noise_generator,
)


class ExperimentRunner:
    def __init__(self, cfg: ScanaConfig):
        self.cfg = cfg
        self.paths = cfg.paths()
        self.out_dir = self.paths["output_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out_dir / "scana_config.json")

        self.episodes = []
        self.episode_deduplication_audit: list[dict[str, object]] = []
        self.dt_train: list[ActionChunk] = []
        self.dt_eval: list[ActionChunk] = []
        self.real_train: list[ActionChunk] = []
        self.real_eval: list[ActionChunk] = []
        self.constraints: ActionConstraints | None = None
        self.model = None
        self.scalers = None
        self.train_pairs: NoiseTrainingData | None = None
        self.val_pairs: NoiseTrainingData | None = None
        self.generator_report = None

    def prepare_data(self) -> dict[str, Any]:
        try:
            self.episodes = load_episodes(self.cfg, groups=["dt_success", "real_success"])
        except Exception as exc:
            cached = self._load_cached_chunks()
            if cached is not None:
                return cached
            raise RuntimeError(
                "Could not load parquet trajectories and no prepared chunk cache exists. "
                "Run stage=prepare once with an environment that has pyarrow or fastparquet."
            ) from exc
        raw_episode_count = len(self.episodes)
        self.episodes, self.episode_deduplication_audit = deduplicate_episodes_by_content(self.episodes)
        write_csv(self.out_dir / "episode_deduplication_audit.csv", self.episode_deduplication_audit)
        chunks = make_chunks(self.episodes, self.cfg)
        dt_chunks = [c for c in chunks if c.group == "dt_success"]
        real_chunks = [c for c in chunks if c.group == "real_success"]
        if not dt_chunks:
            raise RuntimeError("No digital-twin success chunks were found.")
        if not real_chunks:
            raise RuntimeError("No real successful chunks were found.")

        self.dt_train, self.dt_eval = split_chunks_by_episode(
            dt_chunks, self.cfg.train_split, self.cfg.random_seed
        )
        self.real_train, self.real_eval = split_chunks_by_episode(
            real_chunks, self.cfg.train_split, self.cfg.random_seed + 1
        )
        if not self.dt_eval or not self.real_eval:
            raise RuntimeError("Episode-level splitting produced an empty eval set; refusing to reuse train chunks as eval.")

        self.dt_train = cap_chunks(self.dt_train, self.cfg.max_train_chunks, self.cfg.random_seed)
        self.real_train = cap_chunks(self.real_train, self.cfg.max_train_chunks, self.cfg.random_seed + 1)
        self.dt_eval = cap_chunks(self.dt_eval, self.cfg.max_eval_chunks, self.cfg.random_seed + 2)
        self.real_eval = cap_chunks(self.real_eval, self.cfg.max_eval_chunks, self.cfg.random_seed + 3)
        assert_no_content_overlap(self.dt_train, self.dt_eval)
        assert_no_content_overlap(self.real_train, self.real_eval)
        self.train_pairs = build_calibrated_pairs(
            self.dt_train, self.real_train, self.cfg, max_pairs=self.cfg.max_train_chunks
        )
        self.val_pairs = build_calibrated_pairs(
            self.dt_eval, self.real_eval, self.cfg, max_pairs=self.cfg.max_eval_chunks
        )
        self.constraints = estimate_constraints(
            self.dt_train + self.real_train,
            self.cfg,
            self.train_pairs.target_delta,
        )

        rows = summarize_episodes(self.episodes)
        rows.extend(
            [
                {"group": "dt_train_chunks", "datasets": "", "episodes": "", "frames": len(self.dt_train), "mean_frames": ""},
                {"group": "dt_eval_chunks", "datasets": "", "episodes": "", "frames": len(self.dt_eval), "mean_frames": ""},
                {"group": "real_train_chunks", "datasets": "", "episodes": "", "frames": len(self.real_train), "mean_frames": ""},
                {"group": "real_eval_chunks", "datasets": "", "episodes": "", "frames": len(self.real_eval), "mean_frames": ""},
            ]
        )
        write_csv(self.out_dir / "dataset_split_summary.csv", rows)
        save_npz(self.out_dir / "dt_success_chunks_train.npz", chunks_to_arrays(self.dt_train))
        save_npz(self.out_dir / "dt_success_chunks_eval.npz", chunks_to_arrays(self.dt_eval))
        save_npz(self.out_dir / "real_success_chunks_train.npz", chunks_to_arrays(self.real_train))
        save_npz(self.out_dir / "real_success_chunks_eval.npz", chunks_to_arrays(self.real_eval))
        return {
            "episodes_before_content_deduplication": raw_episode_count,
            "episodes": len(self.episodes),
            "duplicate_episodes_removed": len(self.episode_deduplication_audit),
            "dt_train_chunks": len(self.dt_train),
            "dt_eval_chunks": len(self.dt_eval),
            "real_train_chunks": len(self.real_train),
            "real_eval_chunks": len(self.real_eval),
        }

    def _load_cached_chunks(self) -> dict[str, Any] | None:
        paths = {
            "dt_train": self.out_dir / "dt_success_chunks_train.npz",
            "dt_eval": self.out_dir / "dt_success_chunks_eval.npz",
            "real_train": self.out_dir / "real_success_chunks_train.npz",
            "real_eval": self.out_dir / "real_success_chunks_eval.npz",
        }
        if not all(p.exists() for p in paths.values()):
            return None
        self.dt_train = load_chunks_npz(paths["dt_train"])
        self.dt_eval = load_chunks_npz(paths["dt_eval"])
        self.real_train = load_chunks_npz(paths["real_train"])
        self.real_eval = load_chunks_npz(paths["real_eval"])
        all_cached = self.dt_train + self.dt_eval + self.real_train + self.real_eval
        if any(not chunk.content_id for chunk in all_cached):
            return None
        assert_no_content_overlap(self.dt_train, self.dt_eval)
        assert_no_content_overlap(self.real_train, self.real_eval)
        self.train_pairs = build_calibrated_pairs(
            self.dt_train, self.real_train, self.cfg, max_pairs=self.cfg.max_train_chunks
        )
        self.val_pairs = build_calibrated_pairs(
            self.dt_eval, self.real_eval, self.cfg, max_pairs=self.cfg.max_eval_chunks
        )
        self.constraints = estimate_constraints(
            self.dt_train + self.real_train,
            self.cfg,
            self.train_pairs.target_delta,
        )
        rows = [
            {"group": "dt_train_chunks_cached", "datasets": "", "episodes": "", "frames": len(self.dt_train), "mean_frames": ""},
            {"group": "dt_eval_chunks_cached", "datasets": "", "episodes": "", "frames": len(self.dt_eval), "mean_frames": ""},
            {"group": "real_train_chunks_cached", "datasets": "", "episodes": "", "frames": len(self.real_train), "mean_frames": ""},
            {"group": "real_eval_chunks_cached", "datasets": "", "episodes": "", "frames": len(self.real_eval), "mean_frames": ""},
        ]
        write_csv(self.out_dir / "dataset_split_summary_cached.csv", rows)
        return {
            "episodes": "loaded_from_cache",
            "dt_train_chunks": len(self.dt_train),
            "dt_eval_chunks": len(self.dt_eval),
            "real_train_chunks": len(self.real_train),
            "real_eval_chunks": len(self.real_eval),
        }

    def run_422_generator_validation(self) -> list[dict[str, Any]]:
        self._require_data()
        if self.train_pairs is None or self.val_pairs is None:
            raise RuntimeError("Calibrated train/eval pairs were not prepared.")
        self.model, self.scalers, self.generator_report = train_noise_generator(
            self.train_pairs,
            self.val_pairs,
            self.cfg,
            self.out_dir,
        )

        generated = sample_generator(
            self.model,
            self.scalers,
            self.val_pairs.action,
            self.val_pairs.condition,
            self.cfg,
            copies=1,
            seed=self.cfg.random_seed + 422,
        )
        target = self.val_pairs.target_delta
        train_target = self.train_pairs.target_delta
        mmd_gamma = estimate_rbf_gamma(train_target, seed=self.cfg.random_seed + 90)
        rng = np.random.default_rng(self.cfg.random_seed)
        clean = np.zeros_like(target)
        gaussian = rng.normal(0.0, train_target.std(axis=0) + 1.0e-6, size=target.shape).astype(np.float32)
        real_stat = rng.normal(train_target.mean(axis=0), train_target.std(axis=0) + 1.0e-6, size=target.shape).astype(np.float32)

        rows = [
            distribution_row("Clean-DT-zero-delta", clean, target, clean, gamma=mmd_gamma),
            distribution_row("Manual-Gaussian", gaussian, target, clean, gamma=mmd_gamma),
            distribution_row("Real-statistic", real_stat, target, clean, gamma=mmd_gamma),
            distribution_row("SCANA-Generator", generated, target, clean, gamma=mmd_gamma),
        ]
        diversity_copies = max(2, self.cfg.augmentation_copies)
        repeated = sample_generator(
            self.model,
            self.scalers,
            self.val_pairs.action,
            self.val_pairs.condition,
            self.cfg,
            copies=diversity_copies,
            seed=self.cfg.random_seed + 4_220,
        ).reshape(diversity_copies, len(self.val_pairs.action), self.cfg.chunk_size, self.cfg.action_dim)
        within_condition_std = float(np.mean(np.std(repeated, axis=0)))
        global_generated_std = float(np.std(repeated))
        diversity_ratio = within_condition_std / max(global_generated_std, 1.0e-12)
        for row in rows:
            row["experiment"] = "4.2.2_generator_validation"
            if row["method"] == "SCANA-Generator":
                row["within_condition_std"] = within_condition_std
                row["within_to_global_std_ratio"] = diversity_ratio
        write_csv(self.out_dir / "exp422_generator_validation.csv", rows)
        save_npz(
            self.out_dir / "exp422_generator_samples.npz",
            {
                "generated_delta": generated,
                "target_delta": target,
                "clean_delta": clean,
                "gaussian_delta": gaussian,
                "real_stat_delta": real_stat,
                "pair_distance": self.val_pairs.pair_distance,
                "matched_dt_index": self.val_pairs.matched_dt_index,
                "real_index": self.val_pairs.real_index,
            },
        )
        return rows

    def run_421_label_noise_validation(self) -> list[dict[str, Any]]:
        self._require_generator()
        assert self.constraints is not None
        assert self.train_pairs is not None
        assert self.val_pairs is not None

        eval_arrays = chunks_to_arrays(self.dt_eval)
        action = eval_arrays["action"]
        cond = eval_arrays["condition"]
        target = self.val_pairs.target_delta
        clean = np.zeros((len(action), self.cfg.chunk_size, self.cfg.action_dim), dtype=np.float32)
        train_target = self.train_pairs.target_delta
        mmd_gamma = estimate_rbf_gamma(train_target, seed=self.cfg.random_seed + 90)
        rng = np.random.default_rng(self.cfg.random_seed + 10)

        gaussian = rng.normal(0.0, train_target.std(axis=0) + 1.0e-6, size=action.shape).astype(np.float32)
        real_stat = rng.normal(train_target.mean(axis=0), train_target.std(axis=0) + 1.0e-6, size=action.shape).astype(np.float32)
        generated = sample_generator(
            self.model, self.scalers, action, cond, self.cfg, copies=1,
            seed=self.cfg.random_seed + 421,
        )

        methods = {
            "Clean-DT": clean,
            "Manual-Gaussian": gaussian,
            "Real-statistic": real_stat,
            "SCANA-Generator": generated,
        }
        rows = []
        for name, delta in methods.items():
            perturbed, mapped_delta, infos = batch_map(action, delta, self.constraints, alpha=1.0)
            row = distribution_row(name, mapped_delta, target, clean, gamma=mmd_gamma)
            row.update(
                {
                    "experiment": "4.2.1_success_conditioned_label_noise",
                    "projected_fraction": float(np.mean([x["projected_fraction"] for x in infos])),
                    "raw_range_violation_rate": float(np.mean([x["raw_range_violation_rate"] for x in infos])),
                    "perturbed_action_second_difference_roughness": second_difference_roughness(perturbed),
                    "perturbed_action_step_variation": step_variation(perturbed),
                    "direction_consistency": direction_consistency(action, perturbed),
                    "range_violation_rate_after_projection": range_violation_rate(
                        perturbed, self.constraints.action_low, self.constraints.action_high
                    ),
                }
            )
            rows.append(row)
        write_csv(self.out_dir / "exp421_label_noise_validation.csv", rows)
        return rows

    def run_constraint_curve(self) -> list[dict[str, Any]]:
        self._require_generator()
        assert self.constraints is not None
        assert self.val_pairs is not None
        eval_arrays = chunks_to_arrays(self.dt_eval)
        action = eval_arrays["action"]
        cond = eval_arrays["condition"]
        target = self.val_pairs.target_delta
        assert self.train_pairs is not None
        mmd_gamma = estimate_rbf_gamma(self.train_pairs.target_delta, seed=self.cfg.random_seed + 90)
        base_delta = sample_generator(
            self.model, self.scalers, action, cond, self.cfg, copies=1,
            seed=self.cfg.random_seed + 421,
        )
        clean = np.zeros_like(base_delta)
        rows = []
        for alpha in self.cfg.projection_alpha_values:
            perturbed, mapped_delta, infos = batch_map(action, base_delta, self.constraints, alpha=alpha)
            rows.append(
                {
                    "alpha": alpha,
                    "mmd_to_real_delta": rbf_mmd(mapped_delta, target, gamma=mmd_gamma),
                    "mmd_clean_reference": rbf_mmd(clean[: len(target)], target, gamma=mmd_gamma),
                    "mmd_estimator": "biased_mmd_squared",
                    "rbf_gamma": mmd_gamma,
                    "projected_fraction": float(np.mean([x["projected_fraction"] for x in infos])),
                    "raw_range_violation_rate": float(np.mean([x["raw_range_violation_rate"] for x in infos])),
                    "mean_abs_delta": float(np.mean([x["mean_abs_delta"] for x in infos])),
                    "direction_consistency": direction_consistency(action, perturbed),
                    "action_second_difference_roughness": second_difference_roughness(perturbed),
                    "action_step_variation": step_variation(perturbed),
                }
            )
        write_csv(self.out_dir / "exp421_constraint_curve.csv", rows)
        return rows

    def run_423_vla_finetune_assets(self) -> dict[str, Any]:
        self._require_generator()
        assert self.constraints is not None
        arrays = chunks_to_arrays(self.dt_train)
        action = arrays["action"]
        condition = arrays["condition"]
        generated = sample_generator(
            self.model,
            self.scalers,
            action,
            condition,
            self.cfg,
            copies=self.cfg.augmentation_copies,
            seed=self.cfg.random_seed + 4_230,
        )
        tiled_action = np.tile(action, (self.cfg.augmentation_copies, 1, 1))
        tiled_condition = np.tile(condition, (self.cfg.augmentation_copies, 1))
        perturbed, mapped_delta, infos = batch_map(tiled_action, generated, self.constraints, alpha=1.0)
        save_npz(
            self.out_dir / "exp423_augmented_action_labels_scana.npz",
            {
                "base_action": tiled_action,
                "augmented_action": perturbed,
                "delta": mapped_delta,
                "condition": tiled_condition,
                "source_path": np.tile(arrays["source_path"], self.cfg.augmentation_copies),
                "content_id": np.tile(arrays["content_id"], self.cfg.augmentation_copies),
                "episode_index": np.tile(arrays["episode_index"], self.cfg.augmentation_copies),
                "start": np.tile(arrays["start"], self.cfg.augmentation_copies),
                "task": np.tile(arrays["task"], self.cfg.augmentation_copies),
                "copy_id": np.repeat(np.arange(self.cfg.augmentation_copies), len(action)),
                "sampling_seed": np.full(len(perturbed), self.cfg.random_seed + 4_230, dtype=np.int64),
                "action_low": self.constraints.action_low,
                "action_high": self.constraints.action_high,
                "max_abs_delta": self.constraints.max_abs_delta,
            },
        )

        protocol_rows = []
        for method in ["Clean-DT", "Manual-Gaussian", "Real-statistic", "SCANA-Generator"]:
            for seed in self.cfg.vla_seeds:
                protocol_rows.append(
                    {
                        "experiment": "4.2.3_vla_finetune_validation",
                        "method": method,
                        "seed": seed,
                        "base_model": "GR00T/OpenVLA-compatible VLA checkpoint",
                        "train_data": "digital-twin success chunks plus method-specific action labels",
                        "image_policy": "observation-preserved; no image perturbation",
                        "chunk_size": self.cfg.chunk_size,
                        "batch_size": self.cfg.vla_batch_size,
                        "learning_rate": self.cfg.vla_learning_rate,
                        "train_steps": self.cfg.vla_steps,
                        "weight_decay": self.cfg.vla_weight_decay,
                        "warmup_ratio": self.cfg.vla_warmup_ratio,
                        "real_robot_trials": "reported_in_author_supplied_structured_trial_table",
                        "success_rate": "computed_outside_this_offline_pipeline",
                        "mean_completion_time_s": "computed_outside_this_offline_pipeline",
                        "failure_modes": "computed_outside_this_offline_pipeline",
                    }
                )
        write_csv(self.out_dir / "exp423_vla_finetune_protocol.csv", protocol_rows)
        failure_rows = make_failure_case_template()
        write_csv(self.out_dir / "exp43_failure_case_template.csv", failure_rows)
        return {
            "augmented_npz": str(self.out_dir / "exp423_augmented_action_labels_scana.npz"),
            "augmented_chunks": int(len(perturbed)),
            "mean_projected_fraction": float(np.mean([x["projected_fraction"] for x in infos])),
            "vla_protocol_csv": str(self.out_dir / "exp423_vla_finetune_protocol.csv"),
            "failure_template_csv": str(self.out_dir / "exp43_failure_case_template.csv"),
        }

    def run_all(self) -> dict[str, Any]:
        data_summary = self.prepare_data()
        exp422 = self.run_422_generator_validation()
        exp421 = self.run_421_label_noise_validation()
        curve = self.run_constraint_curve()
        exp423 = self.run_423_vla_finetune_assets()
        summary = {
            "config": config_to_jsonable(self.cfg),
            "data_summary": data_summary,
            "generator_report": asdict(self.generator_report) if self.generator_report is not None else None,
            "exp421_rows": exp421,
            "exp421_constraint_curve_rows": curve,
            "exp422_rows": exp422,
            "exp423": exp423,
            "interpretation_boundary": (
                "This pipeline produces offline distribution-alignment and label-constraint results. "
                "Closed-loop VLA statistics are computed separately from the author-supplied "
                "structured trial table and are not generated by this offline pipeline."
            ),
        }
        (self.out_dir / "scana_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _require_data(self) -> None:
        if not self.dt_train or not self.real_train:
            self.prepare_data()

    def _require_generator(self) -> None:
        self._require_data()
        if self.model is None or self.scalers is None:
            self.run_422_generator_validation()


def make_failure_case_template() -> list[dict[str, Any]]:
    modes = [
        "visual_mislocalization",
        "pre_grasp_pose_offset",
        "gripper_timing_error",
        "contact_instability",
        "trajectory_overshoot",
        "recovery_failure",
    ]
    rows = []
    for mode in modes:
        rows.append(
            {
                "failure_mode": mode,
                "Clean-DT": "not_generated_by_offline_pipeline",
                "Manual-Gaussian": "not_generated_by_offline_pipeline",
                "Real-statistic": "not_generated_by_offline_pipeline",
                "SCANA-Generator": "not_generated_by_offline_pipeline",
                "diagnostic_note": "use_author_supplied_structured_failure_annotations",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow(row)


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
