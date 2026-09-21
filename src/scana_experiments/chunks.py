from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ScanaConfig
from .io_lerobot import TrajectoryEpisode


@dataclass
class ActionChunk:
    group: str
    dataset: str
    source_path: str
    episode_index: int
    start: int
    action: np.ndarray
    state: np.ndarray
    condition: np.ndarray
    content_id: str = ""
    task: str = ""

    @property
    def length(self) -> int:
        return int(self.action.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])


def _safe_norm(x: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(x) / max(scale, 1e-6))


def chunk_condition(action: np.ndarray, state: np.ndarray, start: int, episode_len: int) -> np.ndarray:
    """Condition c_t used by the noise generator.

    Layout:
    0 phase center in [0, 1]
    1 mean gripper command / 100
    2 gripper open-close change / 100
    3 mean action norm / 180
    4 mean first-order speed norm / 180
    5 state-action residual norm / 180
    6 keyframe score from gripper and speed changes
    7 chunk length normalized by episode length
    """

    t = action.shape[0]
    center = start + (t - 1) * 0.5
    phase = center / max(episode_len - 1, 1)
    gripper = action[:, -1]
    diffs = np.diff(action, axis=0) if t > 1 else np.zeros((1, action.shape[1]), dtype=action.dtype)
    speed = np.linalg.norm(diffs, axis=1)
    gripper_delta = float((gripper[-1] - gripper[0]) / 100.0)
    keyframe_score = float(min(1.0, abs(gripper_delta) * 2.0 + (speed.max() / 60.0 if len(speed) else 0.0)))
    residual = action - state
    return np.asarray(
        [
            phase,
            float(np.mean(gripper) / 100.0),
            gripper_delta,
            _safe_norm(np.mean(action, axis=0), 180.0),
            float(np.mean(speed) / 180.0) if len(speed) else 0.0,
            _safe_norm(np.mean(residual, axis=0), 180.0),
            keyframe_score,
            float(t / max(episode_len, 1)),
        ],
        dtype=np.float32,
    )


def make_chunks(episodes: Iterable[TrajectoryEpisode], cfg: ScanaConfig) -> list[ActionChunk]:
    chunks: list[ActionChunk] = []
    for ep in episodes:
        if ep.length < cfg.chunk_size:
            continue
        last = ep.length - cfg.chunk_size
        for start in range(0, last + 1, cfg.stride):
            end = start + cfg.chunk_size
            action = ep.action[start:end].astype(np.float32, copy=True)
            state = ep.state[start:end].astype(np.float32, copy=True)
            chunks.append(
                ActionChunk(
                    group=ep.group,
                    dataset=ep.dataset,
                    source_path=str(ep.path),
                    episode_index=ep.episode_index,
                    start=start,
                    action=action,
                    state=state,
                    condition=chunk_condition(action, state, start, ep.length),
                    content_id=ep.content_id,
                    task=ep.task,
                )
            )
    return chunks


def split_chunks_by_episode(
    chunks: list[ActionChunk], train_split: float, seed: int
) -> tuple[list[ActionChunk], list[ActionChunk]]:
    rng = np.random.default_rng(seed)
    def split_key(chunk: ActionChunk) -> tuple[str, str]:
        if not chunk.content_id:
            raise ValueError("Every chunk must carry a path-independent episode content_id before splitting.")
        return chunk.group, chunk.content_id

    keys = sorted({split_key(c) for c in chunks})
    rng.shuffle(keys)
    n_train = max(1, int(round(len(keys) * train_split))) if keys else 0
    train_keys = set(keys[:n_train])
    train, test = [], []
    for c in chunks:
        key = split_key(c)
        (train if key in train_keys else test).append(c)
    return train, test


def chunk_content_hash(chunk: ActionChunk) -> str:
    digest = hashlib.sha256()
    for array in (chunk.action, chunk.state):
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def assert_no_content_overlap(train: list[ActionChunk], eval_: list[ActionChunk]) -> None:
    """Fail fast when identical action-state content appears in both splits."""

    train_hashes = {chunk_content_hash(chunk) for chunk in train}
    overlap_rows = sum(chunk_content_hash(chunk) in train_hashes for chunk in eval_)
    if overlap_rows:
        raise RuntimeError(
            f"Detected {overlap_rows} evaluation chunks whose action+state content also "
            "appears in training. Regenerate caches after trajectory-content deduplication."
        )


def chunks_to_arrays(chunks: list[ActionChunk]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks to convert.")
    return {
        "action": np.stack([c.action for c in chunks]).astype(np.float32),
        "state": np.stack([c.state for c in chunks]).astype(np.float32),
        "condition": np.stack([c.condition for c in chunks]).astype(np.float32),
        "group": np.asarray([c.group for c in chunks]),
        "dataset": np.asarray([c.dataset for c in chunks]),
        "episode_index": np.asarray([c.episode_index for c in chunks], dtype=np.int64),
        "start": np.asarray([c.start for c in chunks], dtype=np.int64),
        "source_path": np.asarray([c.source_path for c in chunks]),
        "content_id": np.asarray([c.content_id for c in chunks]),
        "task": np.asarray([c.task for c in chunks]),
    }


def arrays_to_chunks(arrays: dict[str, np.ndarray]) -> list[ActionChunk]:
    action = arrays["action"]
    state = arrays["state"]
    condition = arrays["condition"]
    group = arrays["group"]
    dataset = arrays["dataset"]
    episode_index = arrays["episode_index"]
    start = arrays["start"]
    source_path = arrays["source_path"]
    content_id = arrays.get("content_id", np.asarray([""] * len(action)))
    task = arrays.get("task", np.asarray([""] * len(action)))
    chunks: list[ActionChunk] = []
    for i in range(len(action)):
        chunks.append(
            ActionChunk(
                group=str(group[i]),
                dataset=str(dataset[i]),
                source_path=str(source_path[i]),
                episode_index=int(episode_index[i]),
                start=int(start[i]),
                action=action[i].astype(np.float32),
                state=state[i].astype(np.float32),
                condition=condition[i].astype(np.float32),
                content_id=str(content_id[i]),
                task=str(task[i]),
            )
        )
    return chunks


def load_chunks_npz(path: str | Path) -> list[ActionChunk]:
    data = np.load(path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}
    return arrays_to_chunks(arrays)


def cap_chunks(chunks: list[ActionChunk], limit: int, seed: int) -> list[ActionChunk]:
    if limit <= 0 or len(chunks) <= limit:
        return chunks
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chunks), size=limit, replace=False)
    return [chunks[int(i)] for i in sorted(idx)]
