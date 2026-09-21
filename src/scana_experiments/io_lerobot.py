from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ScanaConfig


@dataclass
class TrajectoryEpisode:
    path: Path
    dataset: str
    group: str
    episode_index: int
    action: np.ndarray
    state: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    content_id: str
    task: str = ""

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


def infer_group(path: Path, cfg: ScanaConfig) -> str:
    text = str(path).lower()
    if any(k.lower() in text for k in cfg.real_success_keywords):
        return "real_success"
    if all(k.lower() in text for k in cfg.digital_twin_success_keywords):
        return "dt_success"
    if "record-test08" in text or "record-test09" in text or "record-test10" in text:
        return "dt_aggregate"
    return "other"


def infer_dataset(path: Path, dataset_dir: Path) -> str:
    try:
        rel = path.relative_to(dataset_dir)
        return rel.parts[0]
    except ValueError:
        return path.parent.name


def _read_parquet_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns)
        return {c: np.asarray(table[c].to_pylist()) for c in columns if c in table.column_names}
    except Exception:
        try:
            import pandas as pd

            df = pd.read_parquet(path, columns=columns)
            return {c: np.asarray(df[c].tolist()) for c in columns if c in df.columns}
        except Exception as exc:
            raise RuntimeError(f"Could not read parquet file: {path}") from exc


def episode_content_id(action: np.ndarray, state: np.ndarray) -> str:
    """Return a path-independent identity for one physical action/state episode."""

    digest = hashlib.sha256()
    for array in (action, state):
        canonical = np.ascontiguousarray(array, dtype="<f4")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def deduplicate_episodes_by_content(
    episodes: list[TrajectoryEpisode],
) -> tuple[list[TrajectoryEpisode], list[dict[str, object]]]:
    """Remove dual-format copies before any episode split.

    Content identities are compared within the same semantic group.  The first
    path in the deterministic discovery order is retained and every discarded
    path is recorded for audit.
    """

    retained: list[TrajectoryEpisode] = []
    first_by_key: dict[tuple[str, str], TrajectoryEpisode] = {}
    audit: list[dict[str, object]] = []
    for episode in episodes:
        key = (episode.group, episode.content_id)
        first = first_by_key.get(key)
        if first is None:
            first_by_key[key] = episode
            retained.append(episode)
            continue
        audit.append(
            {
                "group": episode.group,
                "content_id": episode.content_id,
                "kept_path": str(first.path),
                "kept_episode_index": first.episode_index,
                "discarded_path": str(episode.path),
                "discarded_episode_index": episode.episode_index,
                "frames": episode.length,
            }
        )
    return retained, audit


def discover_data_parquets(dataset_dir: Path) -> list[Path]:
    files = []
    for p in dataset_dir.rglob("*.parquet"):
        parts = [x.lower() for x in p.parts]
        if "data" in parts and "chunk-000" in parts:
            files.append(p)
    return sorted(files)


def read_task_text(path: Path, dataset_dir: Path) -> str:
    dataset = infer_dataset(path, dataset_dir)
    candidates = [
        dataset_dir / dataset / "meta" / "tasks.jsonl",
        dataset_dir / dataset / "meta" / "tasks.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            first = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if first:
                try:
                    return str(json.loads(first[0]).get("task", ""))
                except Exception:
                    return ""
        if p.suffix == ".parquet":
            try:
                data = _read_parquet_columns(p, ["task"])
                task = data.get("task")
                if task is not None and len(task):
                    return str(task[0])
            except Exception:
                return ""
    return ""


def load_episodes(cfg: ScanaConfig, groups: Iterable[str] | None = None) -> list[TrajectoryEpisode]:
    paths = cfg.paths()
    dataset_dir = paths["dataset_dir"]
    selected = set(groups) if groups is not None else None
    episodes: list[TrajectoryEpisode] = []
    columns = [cfg.action_key, cfg.state_key, cfg.timestamp_key, cfg.episode_key, cfg.frame_key]

    for parquet_path in discover_data_parquets(dataset_dir):
        group = infer_group(parquet_path, cfg)
        if selected is not None and group not in selected:
            continue
        data = _read_parquet_columns(parquet_path, columns)
        required = [cfg.action_key, cfg.state_key, cfg.timestamp_key, cfg.episode_key, cfg.frame_key]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {parquet_path}")
        action = np.asarray(data[cfg.action_key], dtype=np.float32)
        state = np.asarray(data[cfg.state_key], dtype=np.float32)
        if action.ndim != 2 or state.ndim != 2:
            raise ValueError(f"Action/state arrays must be two-dimensional in {parquet_path}")
        n = min(len(action), len(state))
        if action.shape[1] != cfg.action_dim or state.shape[1] != cfg.action_dim:
            raise ValueError(
                f"Expected exactly {cfg.action_dim} ordered action/state channels in {parquet_path}; "
                f"received action={action.shape[1]}, state={state.shape[1]}."
            )
        action = action[:n]
        state = state[:n]
        episode_ids = np.asarray(data[cfg.episode_key], dtype=np.int64).reshape(-1)[:n]
        timestamp = np.asarray(data[cfg.timestamp_key], dtype=np.float32).reshape(-1)[:n]
        frame_index = np.asarray(data[cfg.frame_key], dtype=np.int64).reshape(-1)[:n]
        task = read_task_text(parquet_path, dataset_dir)
        dataset = infer_dataset(parquet_path, dataset_dir)

        for ep in np.unique(episode_ids):
            mask = episode_ids == ep
            if int(mask.sum()) < cfg.chunk_size:
                continue
            positions = np.flatnonzero(mask)
            order = np.lexsort((timestamp[positions], frame_index[positions]))
            positions = positions[order]
            episode_frames = frame_index[positions]
            episode_times = timestamp[positions]
            if len(np.unique(episode_frames)) != len(episode_frames):
                raise ValueError(f"Duplicate frame_index values in episode {ep} of {parquet_path}")
            if np.any(np.diff(episode_times) < 0):
                raise ValueError(f"Non-monotonic timestamps in episode {ep} of {parquet_path}")
            episode_action = action[positions]
            episode_state = state[positions]
            episodes.append(
                TrajectoryEpisode(
                    path=parquet_path,
                    dataset=dataset,
                    group=group,
                    episode_index=int(ep),
                    action=episode_action,
                    state=episode_state,
                    timestamp=episode_times,
                    frame_index=episode_frames,
                    content_id=episode_content_id(episode_action, episode_state),
                    task=task,
                )
            )
    return episodes


def summarize_episodes(episodes: list[TrajectoryEpisode]) -> list[dict[str, object]]:
    rows = []
    for group in sorted({e.group for e in episodes}):
        group_eps = [e for e in episodes if e.group == group]
        if not group_eps:
            continue
        frames = int(sum(e.length for e in group_eps))
        datasets = sorted({e.dataset for e in group_eps})
        rows.append(
            {
                "group": group,
                "datasets": ";".join(datasets),
                "episodes": len(group_eps),
                "frames": frames,
                "mean_frames": float(np.mean([e.length for e in group_eps])),
            }
        )
    return rows

