from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_DIR.parent
ROOT = CODE_DIR.parent


@dataclass
class ScanaConfig:
    """Configuration shared by all SCANA experiments."""

    root_dir: str = str(ROOT)
    dataset_dir: str = str(ROOT / "Datasets")
    output_dir: str = str(CODE_DIR / "outputs" / "scana_experiments")

    action_key: str = "action"
    state_key: str = "observation.state"
    timestamp_key: str = "timestamp"
    episode_key: str = "episode_index"
    frame_key: str = "frame_index"

    action_dim: int = 6
    chunk_size: int = 16
    stride: int = 8
    max_train_chunks: int = 4096
    max_eval_chunks: int = 2048
    random_seed: int = 2026

    digital_twin_success_keywords: list[str] = field(
        default_factory=lambda: ["evernorif", "clean"]
    )
    real_success_keywords: list[str] = field(
        default_factory=lambda: ["record-test11", "record-test12"]
    )

    target_mode: str = "paired_action_delta"
    condition_dim: int = 8
    latent_dim: int = 16
    hidden_dim: int = 256
    train_epochs: int = 180
    batch_size: int = 128
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4

    gaussian_scale: float = 1.0
    augmentation_copies: int = 4
    projection_alpha_values: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    )
    max_delta_quantile: float = 0.95
    action_bound_low_quantile: float = 0.005
    action_bound_high_quantile: float = 0.995
    smooth_window: int = 3

    train_split: float = 0.8
    vla_batch_size: int = 32
    vla_learning_rate: float = 1.0e-4
    vla_weight_decay: float = 1.0e-5
    vla_warmup_ratio: float = 0.05
    vla_steps: int = 20_000
    vla_epochs: int = 10
    vla_seeds: list[int] = field(default_factory=lambda: [7, 11, 23])

    def paths(self) -> dict[str, Path]:
        return {
            "root_dir": Path(self.root_dir),
            "dataset_dir": Path(self.dataset_dir),
            "output_dir": Path(self.output_dir),
        }


def load_config(path: str | Path | None = None) -> ScanaConfig:
    if path is None:
        return ScanaConfig()
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return ScanaConfig(**data)


def save_config(config: ScanaConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def config_to_jsonable(config: ScanaConfig) -> dict[str, Any]:
    return asdict(config)
