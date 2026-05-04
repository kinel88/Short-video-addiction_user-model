import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_vectors import EmbeddingConfig, build_embeddings, save_embedding_bundle
from embedding_experiments import evaluate_bundle


class BuildVectorsFullDatasetTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.output_dir = self.root / "vector_outputs"
        self._write_fixture_files()

    def _write_fixture_files(self):
        interactions = pd.DataFrame(
            [
                {"user_id": 1, "video_id": 101, "play_duration": 5000, "video_duration": 10000, "time": "2020-01-01 00:00:01", "date": 20200101, "timestamp": 1.0, "watch_ratio": 0.50},
                {"user_id": 1, "video_id": 101, "play_duration": 6000, "video_duration": 10000, "time": "2020-01-01 00:00:02", "date": 20200101, "timestamp": 2.0, "watch_ratio": 0.60},
                {"user_id": 2, "video_id": 102, "play_duration": 2000, "video_duration": 10000, "time": "2020-01-01 00:00:03", "date": 20200101, "timestamp": 3.0, "watch_ratio": 0.20},
                {"user_id": 1, "video_id": 102, "play_duration": 7000, "video_duration": 10000, "time": "2020-01-01 00:00:04", "date": 20200101, "timestamp": 4.0, "watch_ratio": 0.70},
                {"user_id": 2, "video_id": 103, "play_duration": 3000, "video_duration": 10000, "time": "2020-01-01 00:00:05", "date": 20200101, "timestamp": 5.0, "watch_ratio": 0.30},
                {"user_id": 1, "video_id": 101, "play_duration": 8000, "video_duration": 10000, "time": "2020-01-02 00:00:01", "date": 20200102, "timestamp": 6.0, "watch_ratio": 0.80},
                {"user_id": 2, "video_id": 102, "play_duration": 4000, "video_duration": 10000, "time": "2020-01-02 00:00:02", "date": 20200102, "timestamp": 7.0, "watch_ratio": 0.40},
                {"user_id": 2, "video_id": 103, "play_duration": 5000, "video_duration": 10000, "time": "2020-01-02 00:00:03", "date": 20200102, "timestamp": 8.0, "watch_ratio": 0.50},
                {"user_id": 1, "video_id": 103, "play_duration": 9000, "video_duration": 10000, "time": "2020-01-03 00:00:01", "date": 20200103, "timestamp": 9.0, "watch_ratio": 0.90},
                {"user_id": 2, "video_id": 101, "play_duration": 10000, "video_duration": 10000, "time": "2020-01-03 00:00:02", "date": 20200103, "timestamp": 10.0, "watch_ratio": 1.00},
            ]
        )
        interactions.to_csv(self.root / "big_matrix.csv", index=False)

        user_features = pd.DataFrame(
            [
                {"user_id": 1, "age_feature": 20},
                {"user_id": 2, "age_feature": 35},
                {"user_id": 999, "age_feature": 99},
            ]
        )
        user_features.to_csv(self.root / "user_features.csv", index=False)

        item_categories = pd.DataFrame(
            [
                {"video_id": 101, "feat": "[1]"},
                {"video_id": 102, "feat": "[2]"},
                {"video_id": 103, "feat": "[1, 2]"},
                {"video_id": 999, "feat": "[9]"},
            ]
        )
        item_categories.to_csv(self.root / "item_categories.csv", index=False)

        item_daily_rows = []
        for date in [20200101, 20200102, 20200103]:
            for video_id, play_progress in [(101, 0.4), (102, 0.6), (103, 0.8), (999, 0.2)]:
                item_daily_rows.append(
                    {
                        "video_id": video_id,
                        "date": date,
                        "show_cnt": 10 + video_id % 3,
                        "play_cnt": 6 + video_id % 2,
                        "valid_play_cnt": 5,
                        "complete_play_cnt": 3,
                        "play_progress": play_progress,
                        "like_cnt": 1,
                        "comment_cnt": 1,
                        "share_cnt": 0,
                        "follow_cnt": 0,
                        "collect_cnt": 0,
                        "report_cnt": 0,
                        "reduce_similar_cnt": 0,
                        "cancel_follow_cnt": 0,
                    }
                )
        pd.DataFrame(item_daily_rows).to_csv(self.root / "item_daily_features.csv", index=False)

    def build_config(self):
        return EmbeddingConfig(
            data_dir=str(self.root),
            output_dir=str(self.output_dir),
        )

    def test_build_embeddings_uses_full_big_matrix_without_split(self):
        bundle = build_embeddings(self.build_config())

        self.assertEqual(bundle["inter_file"], "big_matrix.csv")
        self.assertIsNone(bundle["split"])
        self.assertEqual(bundle["interaction_usage_meta"]["mode"], "full_dataset")
        self.assertEqual(bundle["interaction_usage_meta"]["rows_used_for_embeddings"], 10)
        self.assertEqual(bundle["interaction_usage_meta"]["reference_end_date"], 20200103)
        self.assertEqual(bundle["context_meta"]["reference_end_date"], 20200103)
        self.assertSetEqual(set(bundle["all_user_ids"]), {1, 2})
        self.assertSetEqual(set(bundle["item_ids"]), {101, 102, 103})
        self.assertEqual(bundle["user_latent"].shape[1], 1)
        self.assertEqual(bundle["item_latent"].shape[1], 1)
        self.assertEqual(bundle["final_user_emb"].shape[1], bundle["final_item_emb"].shape[1])

    def test_history_block_uses_repeated_big_matrix_interactions_with_watch_centered_weights(self):
        bundle = build_embeddings(self.build_config())

        item_index = {int(video_id): idx for idx, video_id in enumerate(bundle["item_ids"])}
        user_index = {int(user_id): idx for idx, user_id in enumerate(bundle["all_user_ids"])}

        expected = (
            0.4 * bundle["content_item_side"][item_index[101]]
            + 0.5 * bundle["content_item_side"][item_index[101]]
            + 0.6 * bundle["content_item_side"][item_index[102]]
            + 0.7 * bundle["content_item_side"][item_index[101]]
            + 0.8 * bundle["content_item_side"][item_index[103]]
        ) / 3.0
        expected_norm = np.linalg.norm(expected)
        if expected_norm > 0:
            expected = expected / expected_norm

        np.testing.assert_allclose(
            bundle["hist_user_content_pref"][user_index[1]],
            expected.astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

        self.assertEqual(bundle["history_block_meta"]["interaction_mode"], "full_dataset")
        self.assertEqual(bundle["history_block_meta"]["weight_formula"], "clip(watch_ratio, 0.0, 5.0) - 0.1")
        self.assertEqual(bundle["history_block_meta"]["duplicate_interactions"], "counted repeatedly; no per-item deduplication before averaging")

    def test_save_embedding_bundle_records_history_only_metadata_and_removes_cold_outputs(self):
        config = self.build_config()
        bundle = build_embeddings(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in [
            "user_static_side.npy",
            "predicted_user_latent.npy",
            "predicted_user_content_pref.npy",
            "predicted_user_behavior_pref.npy",
        ]:
            (self.output_dir / filename).write_bytes(b"legacy")

        save_embedding_bundle(bundle)

        meta = json.loads((self.output_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["interaction_file"], "big_matrix.csv")
        self.assertEqual(meta["interaction_usage"]["mode"], "full_dataset")
        self.assertEqual(meta["interaction_usage"]["reference_end_date"], 20200103)
        self.assertEqual(meta["final_user_strategy"]["mode"], "history_only")
        self.assertNotIn("projection", meta)
        self.assertNotIn("time_split", meta)
        self.assertEqual(meta["user_history_content_block"]["weight_formula"], "clip(watch_ratio, 0.0, 5.0) - 0.1")

        user_id_map = pd.read_csv(self.output_dir / "user_id_map.csv")
        self.assertListEqual(list(user_id_map.columns), ["row_index", "user_id"])
        for filename in [
            "user_static_side.npy",
            "predicted_user_latent.npy",
            "predicted_user_content_pref.npy",
            "predicted_user_behavior_pref.npy",
        ]:
            self.assertFalse((self.output_dir / filename).exists())

    def test_embedding_experiments_builds_its_own_eval_split(self):
        config = self.build_config()
        bundle = build_embeddings(config)

        report = evaluate_bundle(bundle, config)

        self.assertEqual(report["embedding_interaction_usage"]["mode"], "full_dataset")
        self.assertEqual(report["evaluation_time_split"]["train_rows"], 8)
        self.assertEqual(report["evaluation_time_split"]["val_rows"], 1)
        self.assertEqual(report["evaluation_time_split"]["test_rows"], 1)
        self.assertEqual(report["splits"]["train"]["rows"], 8)
        self.assertEqual(report["splits"]["val"]["rows"], 1)
        self.assertEqual(report["splits"]["test"]["rows"], 1)
        self.assertNotIn("warm_user_mse", report["splits"]["val"])
        self.assertNotIn("cold_user_mse", report["splits"]["val"])


if __name__ == "__main__":
    unittest.main()
