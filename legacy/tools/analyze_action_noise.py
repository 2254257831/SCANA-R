import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"
OUT = ROOT / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class EpisodeSample:
    source: str
    group: str
    episode_index: int
    length: int
    action_mean: np.ndarray
    action_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    state_min: np.ndarray
    state_max: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        return self.action_mean - self.state_mean

    @property
    def std_gap(self) -> np.ndarray:
        return self.action_std - self.state_std

    @property
    def feature(self) -> np.ndarray:
        action_norm = np.linalg.norm(self.action_mean) / 180.0
        std_norm = np.linalg.norm(self.action_std) / 180.0
        gripper = self.action_mean[-1] / 100.0
        length = self.length / 600.0
        return np.array([length, action_norm, std_norm, gripper], dtype=np.float32)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_meta_dirs():
    for record in sorted(DATASETS.iterdir()):
        if not record.is_dir():
            continue
        direct = record / "meta"
        if direct.exists():
            yield record.name, direct
        ever = record / "EverNorif"
        if ever.exists():
            for sub in sorted(ever.iterdir()):
                meta = sub / "meta"
                if meta.exists():
                    yield sub.name, meta


def infer_group(source: str) -> str:
    low = source.lower()
    if "record-test11" in low or "record-test12" in low:
        return "Real successful execution"
    if "clean" in low:
        return "Digital-twin success chunk"
    if "record-test08" in low or "record-test09" in low or "record-test10" in low:
        return "Digital-twin aggregate"
    return "Other"


def as_arr(stats, key):
    return np.asarray(stats[key], dtype=np.float32)


def load_lengths(meta: Path):
    lengths = {}
    p = meta / "episodes.jsonl"
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                lengths[int(row.get("episode_index", len(lengths)))] = int(row.get("length", 0))
    return lengths


def load_episode_samples():
    samples = []
    for source, meta in iter_meta_dirs():
        eps = meta / "episodes_stats.jsonl"
        if not eps.exists():
            continue
        lengths = load_lengths(meta)
        group = infer_group(source)
        with eps.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats = row["stats"]
                if "action" not in stats or "observation.state" not in stats:
                    continue
                a = stats["action"]
                s = stats["observation.state"]
                idx = int(row.get("episode_index", len(samples)))
                length = lengths.get(idx, int(a.get("count", [0])[0]))
                samples.append(
                    EpisodeSample(
                        source=source,
                        group=group,
                        episode_index=idx,
                        length=length,
                        action_mean=as_arr(a, "mean"),
                        action_std=as_arr(a, "std"),
                        action_min=as_arr(a, "min"),
                        action_max=as_arr(a, "max"),
                        state_mean=as_arr(s, "mean"),
                        state_std=as_arr(s, "std"),
                        state_min=as_arr(s, "min"),
                        state_max=as_arr(s, "max"),
                    )
                )
    return samples


def load_dataset_summary():
    rows = []
    for record in sorted(DATASETS.iterdir()):
        info_path = record / "meta" / "info.json"
        stats_path = record / "meta" / "stats.json"
        if not info_path.exists():
            continue
        info = load_json(info_path)
        stats = load_json(stats_path) if stats_path.exists() else {}
        row = {
            "dataset": record.name,
            "group": infer_group(record.name),
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "videos": info.get("total_videos", None),
            "fps": info.get("fps", 30),
            "robot": info.get("robot_type", ""),
            "task": "",
        }
        task_path = record / "meta" / "tasks.jsonl"
        if task_path.exists():
            with task_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
                if first:
                    row["task"] = json.loads(first).get("task", "")
        if "action" in stats and "observation.state" in stats:
            row["action_mean"] = stats["action"].get("mean", [])
            row["state_mean"] = stats["observation.state"].get("mean", [])
            row["action_std"] = stats["action"].get("std", [])
            row["state_std"] = stats["observation.state"].get("std", [])
            row["residual_l2"] = float(
                np.linalg.norm(np.asarray(row["action_mean"]) - np.asarray(row["state_mean"]))
            )
        rows.append(row)
    return rows


def rbf_mmd(x, y, gamma=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if gamma is None:
        z = np.concatenate([x, y], axis=0)
        d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        gamma = 1.0 / max(med, 1e-6)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)).mean()
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean()
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def wasserstein_diag(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.linalg.norm(x.mean(0) - y.mean(0)) + np.linalg.norm(x.std(0) - y.std(0)))


class CondNoiseGenerator(nn.Module):
    def __init__(self, cond_dim=4, z_dim=8, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + z_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, cond, z):
        return self.net(torch.cat([cond, z], dim=-1))


def train_generator(real_samples, dt_samples, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        real = np.stack([s.residual for s in real_samples])
        mu, sd = real.mean(0), real.std(0) + 1e-6
        cond = np.stack([s.feature for s in dt_samples])
        return np.random.normal(mu, sd, size=(len(dt_samples), real.shape[1])), {"available": False}

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 8
    model = CondNoiseGenerator(z_dim=z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    real_cond = torch.tensor(np.stack([s.feature for s in real_samples]), dtype=torch.float32, device=device)
    dt_cond_all = torch.tensor(np.stack([s.feature for s in dt_samples]), dtype=torch.float32, device=device)
    target = torch.tensor(np.stack([s.residual for s in real_samples]), dtype=torch.float32, device=device)

    def torch_mmd(x, y):
        z = torch.cat([x.detach(), y.detach()], dim=0)
        d = torch.cdist(z, z).pow(2)
        med = torch.median(d[d > 0]) if torch.any(d > 0) else torch.tensor(1.0, device=device)
        gamma = 1.0 / torch.clamp(med, min=1e-6)
        kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
        kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
        kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
        return kxx + kyy - 2.0 * kxy

    losses = []
    for epoch in range(900):
        # Distribution-level calibration: the generator is optimized to match
        # the real successful residual distribution, not a one-to-one residual.
        idx = torch.randint(0, len(dt_cond_all), (len(real_samples),), device=device)
        cond = dt_cond_all[idx]
        z = torch.randn((len(real_samples), z_dim), device=device)
        pred = model(cond, z)
        mmd_loss = torch_mmd(pred, target)
        mean_loss = (pred.mean(0) - target.mean(0)).pow(2).mean()
        std_loss = (pred.std(0) - target.std(0)).pow(2).mean()
        # A light anchor prevents the generator from matching moments while
        # producing unrealistically large labels outside the local action band.
        anchor_z = torch.zeros((len(real_samples), z_dim), device=device)
        anchor = model(real_cond, anchor_z)
        anchor_loss = ((anchor - target) ** 2).mean()
        loss = mmd_loss + 0.2 * mean_loss + 0.2 * std_loss + 0.02 * anchor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 75 == 0 or epoch == 899:
            losses.append(float(loss.detach().cpu()))

    dt_cond = torch.tensor(np.stack([s.feature for s in dt_samples]), dtype=torch.float32, device=device)
    with torch.no_grad():
        gen = []
        for _ in range(4):
            z = torch.randn((len(dt_samples), z_dim), device=device)
            gen.append(model(dt_cond, z).detach().cpu().numpy())
        gen = np.concatenate(gen, axis=0)
    meta = {
        "available": True,
        "device": str(device),
        "epochs": 900,
        "objective": "MMD distribution loss + moment matching + light real-sample anchor",
        "loss_trace": losses,
        "train_samples": len(real_samples),
        "condition_dim": 4,
        "latent_dim": z_dim,
    }
    return gen, meta


def simulate_methods(dt_samples, real_samples):
    real = np.stack([s.residual for s in real_samples])
    dt = np.stack([s.residual for s in dt_samples])
    real_mu = real.mean(0)
    real_cov = np.cov(real.T) + np.eye(real.shape[1]) * 1e-4
    real_sd = real.std(0) + 1e-6
    rng = np.random.default_rng(11)
    gaussian = rng.normal(0.0, real_sd, size=(len(dt) * 4, real.shape[1]))
    stats_cal = rng.multivariate_normal(real_mu, real_cov, size=len(dt) * 4)
    generator, gen_meta = train_generator(real_samples, dt_samples)
    methods = {
        "Clean-DT": dt,
        "Gaussian": gaussian,
        "Real-statistic": stats_cal,
        "SCANA-Generator": generator,
    }
    metrics = []
    base_mmd = rbf_mmd(dt, real)
    for name, arr in methods.items():
        mmd = rbf_mmd(arr, real)
        wd = wasserstein_diag(arr, real)
        metrics.append(
            {
                "method": name,
                "mmd": mmd,
                "diag_wasserstein": wd,
                "mmd_reduction_pct": 0.0 if base_mmd == 0 else (base_mmd - mmd) / base_mmd * 100.0,
            }
        )
    return methods, metrics, gen_meta


def constraint_curve(dt_samples, real_samples, generator_samples):
    dt_action = np.stack([s.action_mean for s in dt_samples])
    real_all_min = np.min(np.stack([s.action_min for s in real_samples]), axis=0)
    real_all_max = np.max(np.stack([s.action_max for s in real_samples]), axis=0)
    real = np.stack([s.residual for s in real_samples])
    gen = generator_samples[: len(dt_samples)]
    real_mmd_baseline = rbf_mmd(np.zeros_like(real[: len(dt_samples)]), real)
    rows = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        perturbed = dt_action + alpha * gen
        low = (perturbed < real_all_min).mean()
        high = (perturbed > real_all_max).mean()
        violation = float(low + high)
        noise = alpha * gen
        mmd = rbf_mmd(noise, real)
        align = []
        for a, p in zip(dt_action, perturbed):
            denom = np.linalg.norm(a) * np.linalg.norm(p) + 1e-6
            align.append(float(np.dot(a, p) / denom))
        direction_consistency = float(np.mean(align))
        proxy = 100.0 * (1.0 - min(1.0, mmd / max(real_mmd_baseline, 1e-6)))
        proxy -= 45.0 * violation
        proxy += 5.0 * direction_consistency
        rows.append(
            {
                "alpha": alpha,
                "mmd_to_real_noise": mmd,
                "joint_range_violation_pct": violation * 100.0,
                "direction_consistency": direction_consistency,
                "robust_label_proxy": min(100.0, max(0.0, proxy)),
            }
        )
    return rows


def grouped_statistics(samples):
    out = []
    for group in sorted(set(s.group for s in samples)):
        ss = [s for s in samples if s.group == group]
        residual = np.stack([s.residual for s in ss])
        std_gap = np.stack([s.std_gap for s in ss])
        out.append(
            {
                "group": group,
                "episodes": len(ss),
                "mean_length": float(np.mean([s.length for s in ss])),
                "residual_mean": residual.mean(0).tolist(),
                "residual_abs_mean": np.abs(residual).mean(0).tolist(),
                "residual_l2_mean": float(np.linalg.norm(residual, axis=1).mean()),
                "std_gap_mean": std_gap.mean(0).tolist(),
                "std_gap_l2_mean": float(np.linalg.norm(std_gap, axis=1).mean()),
            }
        )
    return out


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")


def main():
    samples = load_episode_samples()
    dt_samples = [s for s in samples if s.group == "Digital-twin success chunk"]
    real_samples = [s for s in samples if s.group == "Real successful execution"]
    if not dt_samples or not real_samples:
        raise RuntimeError("Need both digital-twin and real successful episode statistics.")

    methods, method_metrics, gen_meta = simulate_methods(dt_samples, real_samples)
    curve = constraint_curve(dt_samples, real_samples, methods["SCANA-Generator"])
    grouped = grouped_statistics(samples)
    dataset_summary = load_dataset_summary()

    real_res = np.stack([s.residual for s in real_samples])
    dt_res = np.stack([s.residual for s in dt_samples])
    joint_rows = []
    for i, name in enumerate(JOINT_NAMES):
        joint_rows.append(
            {
                "joint": name,
                "dt_abs_residual": float(np.abs(dt_res[:, i]).mean()),
                "real_abs_residual": float(np.abs(real_res[:, i]).mean()),
                "ratio_real_to_dt": float((np.abs(real_res[:, i]).mean() + 1e-6) / (np.abs(dt_res[:, i]).mean() + 1e-6)),
            }
        )

    result = {
        "dataset_summary": dataset_summary,
        "grouped_statistics": grouped,
        "joint_residual_table": joint_rows,
        "method_metrics": method_metrics,
        "constraint_curve": curve,
        "generator": gen_meta,
        "notes": {
            "data_used": "metadata info.json, stats.json, episodes.jsonl, episodes_stats.jsonl under Datasets",
            "parquet_status": "Parquet files were not decoded because pyarrow/fastparquet are unavailable in the provided environments.",
            "vla_checkpoint_status": "No VLA checkpoint or training logs were found in Datasets; VLA deployment claims should be treated as protocol-ready, not completed closed-loop results.",
        },
    }
    with (OUT / "analysis_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_csv(OUT / "method_metrics.csv", method_metrics, ["method", "mmd", "diag_wasserstein", "mmd_reduction_pct"])
    write_csv(OUT / "constraint_curve.csv", curve, ["alpha", "mmd_to_real_noise", "joint_range_violation_pct", "direction_consistency", "robust_label_proxy"])
    write_csv(OUT / "joint_residual_table.csv", joint_rows, ["joint", "dt_abs_residual", "real_abs_residual", "ratio_real_to_dt"])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
