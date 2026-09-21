from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scana_experiments.chunks import ActionChunk, split_chunks_by_episode
from scana_experiments.config import ScanaConfig
from scana_experiments.constraints import estimate_constraints, map_delta
from scana_experiments.io_lerobot import (
    TrajectoryEpisode,
    deduplicate_episodes_by_content,
    episode_content_id,
)
from scana_experiments.metrics import (
    diagonal_wasserstein,
    estimate_rbf_gamma,
    rbf_mmd,
    second_difference_roughness,
)
from scana_experiments.models import (
    ConditionalActionNoiseGenerator,
    Standardizer,
    build_calibrated_pairs,
    sample_generator,
    torch,
)


def make_episode(path: str, content_offset: float = 0.0) -> TrajectoryEpisode:
    action = np.arange(96, dtype=np.float32).reshape(16, 6) + content_offset
    state = action - 0.5
    return TrajectoryEpisode(
        path=Path(path),
        dataset="synthetic",
        group="dt_success",
        episode_index=0,
        action=action,
        state=state,
        timestamp=np.arange(16, dtype=np.float32) / 30.0,
        frame_index=np.arange(16, dtype=np.int64),
        content_id=episode_content_id(action, state),
        task="pick",
    )


class RepairTests(unittest.TestCase):
    def test_content_deduplication_is_path_independent(self) -> None:
        first = make_episode("clean/episode_000000.parquet")
        duplicate = make_episode("clean_v3.0/file-000.parquet")
        unique, audit = deduplicate_episodes_by_content([first, duplicate])
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["kept_path"], str(first.path))

    def test_split_uses_content_identity(self) -> None:
        chunks = []
        for i in range(4):
            action = np.full((16, 6), i, dtype=np.float32)
            chunks.append(
                ActionChunk(
                    group="dt_success",
                    dataset="synthetic",
                    source_path=f"episode_{i}.parquet",
                    episode_index=i,
                    start=0,
                    action=action,
                    state=action,
                    condition=np.zeros(8, dtype=np.float32),
                    content_id=f"content-{i}",
                )
            )
        train, test = split_chunks_by_episode(chunks, 0.5, 2026)
        self.assertTrue({c.content_id for c in train}.isdisjoint({c.content_id for c in test}))

    def test_pair_condition_comes_from_matched_dt(self) -> None:
        cfg = ScanaConfig(max_train_chunks=0)
        dt = ActionChunk(
            group="dt_success", dataset="dt", source_path="dt", episode_index=0, start=0,
            action=np.zeros((16, 6), dtype=np.float32), state=np.zeros((16, 6), dtype=np.float32),
            condition=np.arange(8, dtype=np.float32), content_id="dt-0",
        )
        real = ActionChunk(
            group="real_success", dataset="real", source_path="real", episode_index=0, start=0,
            action=np.ones((16, 6), dtype=np.float32), state=np.zeros((16, 6), dtype=np.float32),
            condition=np.arange(8, dtype=np.float32) + 100.0, content_id="real-0",
        )
        pairs = build_calibrated_pairs([dt], [real], cfg)
        np.testing.assert_array_equal(pairs.condition[0], dt.condition)

    def test_constraint_envelope_uses_supplied_matched_delta(self) -> None:
        cfg = ScanaConfig(max_delta_quantile=0.95)
        chunk = ActionChunk(
            group="dt_success", dataset="dt", source_path="dt", episode_index=0, start=0,
            action=np.zeros((16, 6), dtype=np.float32), state=np.full((16, 6), -100.0, dtype=np.float32),
            condition=np.zeros(8, dtype=np.float32), content_id="dt-0",
        )
        paired_delta = np.full((2, 16, 6), 3.0, dtype=np.float32)
        constraints = estimate_constraints([chunk], cfg, paired_delta)
        np.testing.assert_allclose(constraints.max_abs_delta, 3.0)

    def test_metric_names_match_their_mathematics(self) -> None:
        x = np.zeros((2, 3, 1), dtype=np.float32)
        y = np.ones((2, 3, 1), dtype=np.float32)
        self.assertAlmostEqual(diagonal_wasserstein(x, y), np.sqrt(3.0))
        self.assertEqual(second_difference_roughness(x), 0.0)

    def test_fixed_mmd_bandwidth_is_reused(self) -> None:
        reference = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        gamma = estimate_rbf_gamma(reference)
        near = reference + 0.1
        far = reference + 10.0
        self.assertLess(rbf_mmd(near, reference, gamma=gamma), rbf_mmd(far, reference, gamma=gamma))

    def test_mapping_clips_to_declared_action_quantiles(self) -> None:
        cfg = ScanaConfig(action_bound_low_quantile=0.0, action_bound_high_quantile=1.0)
        base = ActionChunk(
            group="dt_success", dataset="dt", source_path="dt", episode_index=0, start=0,
            action=np.vstack([np.zeros((8, 6)), np.ones((8, 6))]).astype(np.float32),
            state=np.zeros((16, 6), dtype=np.float32), condition=np.zeros(8, dtype=np.float32),
            content_id="dt-clip",
        )
        constraints = estimate_constraints(
            [base], cfg, np.full((2, 16, 6), 100.0, dtype=np.float32)
        )
        action = np.full((16, 6), 2.0, dtype=np.float32)
        mapped, _, _ = map_delta(action, np.zeros_like(action), constraints)
        self.assertTrue(np.all(mapped <= constraints.action_high))
        self.assertTrue(np.all(mapped >= constraints.action_low))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_sampling_seed_is_reproducible(self) -> None:
        cfg = ScanaConfig(action_dim=2, chunk_size=3, condition_dim=1, latent_dim=2, hidden_dim=4)
        torch.manual_seed(1)
        model = ConditionalActionNoiseGenerator(2, 3, 1, 2, 4)
        scalers = {
            "action": Standardizer(np.zeros(6), np.ones(6)),
            "condition": Standardizer(np.zeros(1), np.ones(1)),
            "delta": Standardizer(np.zeros(6), np.ones(6)),
        }
        actions = np.zeros((2, 3, 2), dtype=np.float32)
        conditions = np.zeros((2, 1), dtype=np.float32)
        first = sample_generator(model, scalers, actions, conditions, cfg, copies=2, seed=77)
        second = sample_generator(model, scalers, actions, conditions, cfg, copies=2, seed=77)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
