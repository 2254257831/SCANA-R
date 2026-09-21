from __future__ import annotations

import argparse
import json
from pathlib import Path

from scana_experiments.config import ScanaConfig, load_config, save_config
from scana_experiments.experiments import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SCANA single-arm action-label noise experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config path. If omitted, built-in defaults are used.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["write-config", "prepare", "422", "421", "423", "full"],
        help="Experiment stage to run.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override dataset directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override noise-generator epochs.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override action chunk length.")
    parser.add_argument("--stride", type=int, default=None, help="Override action chunk stride.")
    parser.add_argument("--max-train-chunks", type=int, default=None, help="Limit train chunks for quick tests.")
    parser.add_argument("--max-eval-chunks", type=int, default=None, help="Limit eval chunks for quick tests.")
    return parser


def apply_overrides(cfg: ScanaConfig, args: argparse.Namespace) -> ScanaConfig:
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.epochs is not None:
        cfg.train_epochs = args.epochs
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.stride is not None:
        cfg.stride = args.stride
    if args.max_train_chunks is not None:
        cfg.max_train_chunks = args.max_train_chunks
    if args.max_eval_chunks is not None:
        cfg.max_eval_chunks = args.max_eval_chunks
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    runner = ExperimentRunner(cfg)

    if args.stage == "write-config":
        path = Path(cfg.output_dir) / "scana_config.json"
        save_config(cfg, path)
        result = {"config": str(path)}
    elif args.stage == "prepare":
        result = runner.prepare_data()
    elif args.stage == "422":
        runner.prepare_data()
        result = {"exp422": runner.run_422_generator_validation()}
    elif args.stage == "421":
        runner.prepare_data()
        runner.run_422_generator_validation()
        result = {
            "exp421": runner.run_421_label_noise_validation(),
            "constraint_curve": runner.run_constraint_curve(),
        }
    elif args.stage == "423":
        runner.prepare_data()
        runner.run_422_generator_validation()
        result = runner.run_423_vla_finetune_assets()
    else:
        result = runner.run_all()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
