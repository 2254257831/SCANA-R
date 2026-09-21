from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunks import ActionChunk
from .config import ScanaConfig


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class NoiseTrainingData:
    action: np.ndarray
    condition: np.ndarray
    target_delta: np.ndarray
    pair_distance: np.ndarray
    matched_dt_index: np.ndarray
    real_index: np.ndarray


@dataclass
class TrainReport:
    device: str
    epochs: int
    train_pairs: int
    val_pairs: int
    final_train_loss: float
    final_val_loss: float
    best_epoch: int
    best_val_loss: float
    validation_objective: str
    loss_trace: list[dict[str, float]]


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1.0e-6)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, data: dict[str, list[float]]) -> "Standardizer":
        return cls(np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32))


if nn is None:

    class ConditionalActionNoiseGenerator:  # pragma: no cover
        """Placeholder so parquet-only environments can run data preparation."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to construct ConditionalActionNoiseGenerator.")

else:

    class ConditionalActionNoiseGenerator(nn.Module):
        """G_phi(A_dt, c, z) -> Delta A with the same chunk shape as A_dt."""

        def __init__(self, action_dim: int, chunk_size: int, condition_dim: int, latent_dim: int, hidden_dim: int):
            super().__init__()
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.condition_dim = condition_dim
            self.latent_dim = latent_dim
            flat_action = action_dim * chunk_size
            in_dim = flat_action + condition_dim + latent_dim
            out_dim = flat_action
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, action_flat: "torch.Tensor", condition: "torch.Tensor", z: "torch.Tensor") -> "torch.Tensor":
            x = torch.cat([action_flat, condition, z], dim=-1)
            return self.net(x)


def _chunk_descriptor(chunks: list[ActionChunk]) -> np.ndarray:
    action_mean = np.stack([c.action.mean(axis=0) for c in chunks])
    action_std = np.stack([c.action.std(axis=0) for c in chunks])
    cond = np.stack([c.condition for c in chunks])
    desc = np.concatenate([action_mean / 180.0, action_std / 90.0, cond], axis=1)
    return desc.astype(np.float32)


def build_calibrated_pairs(
    dt_chunks: list[ActionChunk],
    real_chunks: list[ActionChunk],
    cfg: ScanaConfig,
    max_pairs: int | None = None,
) -> NoiseTrainingData:
    if not dt_chunks or not real_chunks:
        raise ValueError("Need both digital-twin and real successful chunks.")

    rng = np.random.default_rng(cfg.random_seed)
    dt_desc = _chunk_descriptor(dt_chunks)
    real_desc = _chunk_descriptor(real_chunks)
    dt_action = np.stack([c.action for c in dt_chunks]).astype(np.float32)
    real_action = np.stack([c.action for c in real_chunks]).astype(np.float32)
    real_state = np.stack([c.state for c in real_chunks]).astype(np.float32)

    indices = np.arange(len(real_chunks))
    if max_pairs is not None and len(indices) > max_pairs:
        indices = rng.choice(indices, size=max_pairs, replace=False)

    matched_actions = []
    matched_conditions = []
    target_delta = []
    pair_distance = []
    matched_dt_index = []
    real_index = []
    for i in indices:
        dist = ((dt_desc - real_desc[i]) ** 2).sum(axis=1)
        j = int(np.argmin(dist))
        if cfg.target_mode == "state_residual":
            delta = real_action[i] - real_state[i]
        elif cfg.target_mode == "paired_action_delta":
            delta = real_action[i] - dt_action[j]
        else:
            raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")
        matched_actions.append(dt_action[j])
        # The real descriptor is used only to retrieve a successful DT anchor.
        # Generator training and inference must both condition on the DT domain.
        matched_conditions.append(dt_chunks[j].condition)
        target_delta.append(delta)
        pair_distance.append(float(dist[j]))
        matched_dt_index.append(j)
        real_index.append(int(i))

    return NoiseTrainingData(
        action=np.stack(matched_actions).astype(np.float32),
        condition=np.stack(matched_conditions).astype(np.float32),
        target_delta=np.stack(target_delta).astype(np.float32),
        pair_distance=np.asarray(pair_distance, dtype=np.float32),
        matched_dt_index=np.asarray(matched_dt_index, dtype=np.int64),
        real_index=np.asarray(real_index, dtype=np.int64),
    )


def _mmd_torch(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    with torch.no_grad():
        z = torch.cat([x.detach(), y.detach()], dim=0)
        d = torch.cdist(z, z).pow(2)
        med = torch.median(d[d > 0]) if torch.any(d > 0) else torch.tensor(1.0, device=x.device)
        gamma = 1.0 / torch.clamp(med, min=1.0e-6)
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return kxx + kyy - 2.0 * kxy


def train_noise_generator(
    train_data: NoiseTrainingData,
    val_data: NoiseTrainingData,
    cfg: ScanaConfig,
    out_dir: Path,
    *,
    mmd_weight: float = 0.10,
) -> tuple[ConditionalActionNoiseGenerator, dict[str, Standardizer], TrainReport]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for train_noise_generator.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n, t, d = train_data.action.shape
    action_scaler = Standardizer.fit(train_data.action.reshape(n, -1))
    cond_scaler = Standardizer.fit(train_data.condition)
    delta_scaler = Standardizer.fit(train_data.target_delta.reshape(n, -1))

    def make_tensors(data: NoiseTrainingData):
        n0 = data.action.shape[0]
        action = torch.tensor(
            action_scaler.transform(data.action.reshape(n0, -1)),
            dtype=torch.float32,
            device=device,
        )
        cond = torch.tensor(cond_scaler.transform(data.condition), dtype=torch.float32, device=device)
        target = torch.tensor(
            delta_scaler.transform(data.target_delta.reshape(n0, -1)),
            dtype=torch.float32,
            device=device,
        )
        return action, cond, target

    train_action, train_cond, train_target = make_tensors(train_data)
    val_action, val_cond, val_target = make_tensors(val_data)
    model = ConditionalActionNoiseGenerator(
        action_dim=d,
        chunk_size=t,
        condition_dim=cfg.condition_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    trace: list[dict[str, float]] = []
    batch_size = min(cfg.batch_size, len(train_action))
    validation_rng = torch.Generator(device=device)
    validation_rng.manual_seed(cfg.random_seed + 100_003)
    validation_z = torch.randn(
        (len(val_action), cfg.latent_dim), device=device, generator=validation_rng
    )
    best_epoch = 0
    best_val_loss = float("inf")
    best_state: dict[str, "torch.Tensor"] | None = None

    def objective(pred: "torch.Tensor", target: "torch.Tensor") -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        mse = ((pred - target) ** 2).mean()
        mmd = _mmd_torch(pred, target) if mmd_weight else pred.new_tensor(0.0)
        pred_chunk = pred.view(len(pred), t, d)
        smooth = (pred_chunk[:, 1:] - pred_chunk[:, :-1]).pow(2).mean() if t > 1 else pred.new_tensor(0.0)
        amp = pred.pow(2).mean()
        total = mse + mmd_weight * mmd + 0.01 * smooth + 0.001 * amp
        return total, {"mse": mse, "mmd": mmd, "smooth": smooth, "amp": amp}

    for epoch in range(max(1, cfg.train_epochs)):
        perm = rng.permutation(len(train_action))
        model.train()
        losses = []
        for start in range(0, len(perm), batch_size):
            idx_np = perm[start : start + batch_size]
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            a = train_action[idx]
            c = train_cond[idx]
            y = train_target[idx]
            z = torch.randn((len(idx), cfg.latent_dim), device=device)
            pred = model(a, c, z)
            loss, _ = objective(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 0 or (epoch + 1) % max(1, cfg.train_epochs // 10) == 0 or epoch + 1 == cfg.train_epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_action, val_cond, validation_z)
                val_loss, val_parts = objective(val_pred, val_target)
            val_loss_value = float(val_loss.detach().cpu())
            if val_loss_value < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = val_loss_value
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            trace.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "val_loss": val_loss_value,
                    "val_mse": float(val_parts["mse"].detach().cpu()),
                    "val_mmd": float(val_parts["mmd"].detach().cpu()),
                    "val_smooth": float(val_parts["smooth"].detach().cpu()),
                    "val_amp": float(val_parts["amp"].detach().cpu()),
                }
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was recorded.")
    model.load_state_dict(best_state)

    report = TrainReport(
        device=str(device),
        epochs=max(1, cfg.train_epochs),
        train_pairs=len(train_action),
        val_pairs=len(val_action),
        final_train_loss=trace[-1]["train_loss"],
        final_val_loss=trace[-1]["val_loss"],
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        validation_objective=f"mse + {mmd_weight:.2f}*mmd + 0.01*smooth + 0.001*amp with fixed validation latent draws",
        loss_trace=trace,
    )
    scalers = {"action": action_scaler, "condition": cond_scaler, "delta": delta_scaler}
    save_checkpoint(model, scalers, cfg, report, out_dir / "noise_generator.pt")
    (out_dir / "noise_generator_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, scalers, report


def save_checkpoint(
    model: ConditionalActionNoiseGenerator,
    scalers: dict[str, Standardizer],
    cfg: ScanaConfig,
    report: TrainReport,
    path: Path,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to save checkpoints.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "scalers": {k: v.state_dict() for k, v in scalers.items()},
            "cfg": {
                "action_dim": cfg.action_dim,
                "chunk_size": cfg.chunk_size,
                "condition_dim": cfg.condition_dim,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
            },
            "report": asdict(report),
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | None = None):
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required to load checkpoints.")
    ckpt = torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = ckpt["cfg"]
    model = ConditionalActionNoiseGenerator(**model_cfg)
    model.load_state_dict(ckpt["model"])
    scalers = {k: Standardizer.from_state_dict(v) for k, v in ckpt["scalers"].items()}
    return model, scalers, ckpt


def sample_generator(
    model: ConditionalActionNoiseGenerator,
    scalers: dict[str, Standardizer],
    actions: np.ndarray,
    conditions: np.ndarray,
    cfg: ScanaConfig,
    copies: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for sample_generator.")
    device = next(model.parameters()).device
    n, t, d = actions.shape
    action_flat = scalers["action"].transform(actions.reshape(n, -1))
    cond = scalers["condition"].transform(conditions)
    action_t = torch.tensor(action_flat, dtype=torch.float32, device=device)
    cond_t = torch.tensor(cond, dtype=torch.float32, device=device)
    model.eval()
    samples = []
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    with torch.no_grad():
        for _ in range(copies):
            z = torch.randn((n, cfg.latent_dim), device=device, generator=generator)
            pred = model(action_t, cond_t, z).detach().cpu().numpy()
            delta = scalers["delta"].inverse(pred).reshape(n, t, d)
            samples.append(delta.astype(np.float32))
    return np.concatenate(samples, axis=0)
