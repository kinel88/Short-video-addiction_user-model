import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import run_content_only_fatigue_90k as content_only_fatigue_runner_module
import run_default_dopamine_dual_mode_90k as default_dopamine_dual_mode_module
import run_dopamine_variant_grid as dopamine_variant_grid_module
import generate_final_watch_ratio_reports as final_reports_module
import generate_rollout_watch_ratio_simple_report as rollout_simple_module
import generate_rollout_dopamine_relaxation_experiment as relaxation_experiment_module
import generate_rollout_watch_ratio_simple_report_post_history_dopamine as post_history_rollout_simple_module
import main as main_module
import numpy as np
import pandas as pd
import torch

from agent_module import LinearAgent
from generate_final_watch_ratio_reports import (
    evaluate_shared_loss_weight_candidates,
    write_final_watch_ratio_report,
)
from env_module import (
    ContentOnlyFatigueUserModel,
    CorpusDocumentSampler,
    NoDopamineUserModel,
    SimpleResponse,
    SimpleUserModel,
    make_content_only_fatigue_env,
    make_env,
    make_no_dopamine_env,
)
from experimental_dopamine_models import (
    IntegratedSignalDopamineBehaviorFitModel,
    PostHistoryPeakRewardDopamineBehaviorFitModel,
    RelaxToHigherBaselineDopamineBehaviorFitModel,
)
from train_module import (
    _apply_scalar_dopamine_update_scaffold,
    _compute_scalar_dopamine_habit_progress,
    _compute_scalar_time_weighted_average_reward,
    _compute_torch_time_weighted_average_reward,
    bucket_index_to_label,
    collect_behavior_predictions,
    collect_dopamine_state_traces,
    _run_behavior_model_pass,
    _prepare_behavior_fit_train_val_split_from_dataframe,
    BehaviorFitModel,
    compare_content_only_fatigue_model_on_kuairec,
    compute_threshold_banded_watch_ratio_components,
    ContentOnlyFatigueBehaviorFitModel,
    DOPAMINE_PARAM_KEYS,
    NoDopamineBehaviorFitModel,
    NpyFeatureProvider,
    compare_behavior_models_on_kuairec,
    compute_novelty_norm,
    evaluate_dopamine_ablation_on_kuairec,
    fit_content_only_fatigue_user_model_from_kuairec,
    load_kuairec_small_matrix,
    prepare_behavior_fit_explicit_train_val_split_from_dataframes,
    prepare_behavior_fit_train_val_split,
    REPEAT_WATCH_PASS_CAP,
    watch_ratio_to_bucket_index,
)


class UserModelRefactorTests(unittest.TestCase):
    @staticmethod
    def scalar(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    @staticmethod
    def score_engagement(score, watch_ratio):
        return float(score) * max(float(watch_ratio), 0.0)

    @staticmethod
    def habit_coef(growth_rate, max_gain, swipe_count):
        return 1.0 + float(max_gain) * _compute_scalar_dopamine_habit_progress(
            swipe_count=swipe_count,
            habit_growth_rate=growth_rate,
        )

    def expected_reward_from_history(self, params, recent_score_history):
        peak_decay = self.scalar(params["peak_decay"])
        average_decay = self.scalar(params["average_reward_decay"])
        work = [float(v) for v in recent_score_history[-10:]]
        if not work:
            session_peak = 0.0
        else:
            session_peak = max(
                value * (peak_decay ** (len(work) - 1 - idx))
                for idx, value in enumerate(work)
            )
        time_weighted_average = _compute_scalar_time_weighted_average_reward(
            work,
            average_decay,
        )
        expected_reward = (
            self.scalar(params["expected_reward_peak_coef"]) * session_peak
            + self.scalar(params["expected_reward_average_coef"]) * time_weighted_average
        )
        return expected_reward, session_peak, time_weighted_average

    @staticmethod
    def kuairec_timestamp(ts: pd.Timestamp):
        return ts.tz_localize("Asia/Shanghai").timestamp()

    def build_provider(self, root: Path):
        user_embeddings = np.zeros((2, 32), dtype=np.float32)
        user_embeddings[0, 0] = 1.0
        user_embeddings[1, 1] = 1.0

        item_embeddings = np.zeros((3, 32), dtype=np.float32)
        item_embeddings[0, :31] = 0.0
        item_embeddings[1, :31] = 1.0
        item_embeddings[2, :31] = 2.0
        item_embeddings[0, 31] = 0.1
        item_embeddings[1, 31] = 0.2
        item_embeddings[2, 31] = 0.3

        np.save(root / "user_embeddings.npy", user_embeddings)
        np.save(root / "item_embeddings.npy", item_embeddings)

        pd.DataFrame(
            {"row_index": [0, 1], "user_id": [10, 11], "is_warm_user": [1, 1]}
        ).to_csv(root / "user_id_map.csv", index=False)
        pd.DataFrame(
            {
                "row_index": [0, 1, 2],
                "video_id": [100, 101, 102],
                "quality_index": [90.0, 30.0, 60.0],
                "quality_confidence": [0.8, 0.2, 0.5],
            }
        ).to_csv(root / "item_id_map.csv", index=False)
        pd.DataFrame(
            {
                "video_id": [100, 101, 102],
                "author_id": [500, 501, 500],
                "video_duration": [5000.0, 8000.0, 6000.0],
            }
        ).to_csv(root / "item_daily_features.csv", index=False)
        pd.DataFrame(
            {
                "user_id": [10, 10, 11, 11],
                "author_id": [500, 501, 500, 501],
                "creator_preference_score": [10.0, 1.0, 0.0, 3.0],
            }
        ).to_csv(root / "user_author_preference_dense.csv", index=False)

        return NpyFeatureProvider(
            user_embedding_path=root / "user_embeddings.npy",
            item_embedding_path=root / "item_embeddings.npy",
            user_id_map_path=root / "user_id_map.csv",
            item_id_map_path=root / "item_id_map.csv",
            item_daily_path=root / "item_daily_features.csv",
            creator_preference_path=root / "user_author_preference_dense.csv",
        )

    def write_small_matrix_csv(self, root: Path):
        day1 = pd.Timestamp("2020-01-01 00:00:00")
        day2 = pd.Timestamp("2020-01-02 00:00:00")
        df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 100,
                    "play_duration": 5000.0,
                    "video_duration": 5000.0,
                    "time": (day1 + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=1)),
                    "watch_ratio": 1.0,
                },
                {
                    "user_id": 10,
                    "video_id": 101,
                    "play_duration": 1000.0,
                    "video_duration": 8000.0,
                    "time": (day1 + pd.Timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=2)),
                    "watch_ratio": 0.125,
                },
                {
                    "user_id": 10,
                    "video_id": 102,
                    "play_duration": 4800.0,
                    "video_duration": 6000.0,
                    "time": (day1 + pd.Timedelta(seconds=3)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=3)),
                    "watch_ratio": 0.8,
                },
                {
                    "user_id": 11,
                    "video_id": 102,
                    "play_duration": 6000.0,
                    "video_duration": 6000.0,
                    "time": (day2 + pd.Timedelta(seconds=4)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200102,
                    "timestamp": self.kuairec_timestamp(day2 + pd.Timedelta(seconds=4)),
                    "watch_ratio": 1.0,
                },
                {
                    "user_id": 11,
                    "video_id": 100,
                    "play_duration": 1500.0,
                    "video_duration": 5000.0,
                    "time": (day2 + pd.Timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200102,
                    "timestamp": self.kuairec_timestamp(day2 + pd.Timedelta(seconds=5)),
                    "watch_ratio": 0.3,
                },
            ]
        )
        csv_path = root / "small_matrix.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def write_session_split_small_matrix_csv(self, root: Path):
        day1 = pd.Timestamp("2020-01-01 00:00:00")
        day2 = pd.Timestamp("2020-01-02 00:00:00")
        df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 100,
                    "play_duration": 5000.0,
                    "video_duration": 5000.0,
                    "time": day1.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": np.nan,
                    "watch_ratio": 1.0,
                },
                {
                    "user_id": 10,
                    "video_id": 101,
                    "play_duration": 1000.0,
                    "video_duration": 8000.0,
                    "time": (day1 + pd.Timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=60)),
                    "watch_ratio": 0.125,
                },
                {
                    "user_id": 10,
                    "video_id": 102,
                    "play_duration": 4800.0,
                    "video_duration": 6000.0,
                    "time": (day1 + pd.Timedelta(seconds=1861)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=1861)),
                    "watch_ratio": 0.8,
                },
                {
                    "user_id": 10,
                    "video_id": 100,
                    "play_duration": 2000.0,
                    "video_duration": 5000.0,
                    "time": np.nan,
                    "date": np.nan,
                    "timestamp": np.nan,
                    "watch_ratio": 0.4,
                },
                {
                    "user_id": 10,
                    "video_id": 101,
                    "play_duration": 3000.0,
                    "video_duration": 8000.0,
                    "time": np.nan,
                    "date": np.nan,
                    "timestamp": np.nan,
                    "watch_ratio": 0.5,
                },
                {
                    "user_id": 10,
                    "video_id": 102,
                    "play_duration": 3500.0,
                    "video_duration": 6000.0,
                    "time": (day1 + pd.Timedelta(seconds=1920)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200101,
                    "timestamp": self.kuairec_timestamp(day1 + pd.Timedelta(seconds=1920)),
                    "watch_ratio": 0.6,
                },
                {
                    "user_id": 11,
                    "video_id": 100,
                    "play_duration": 2200.0,
                    "video_duration": 5000.0,
                    "time": np.nan,
                    "date": np.nan,
                    "timestamp": np.nan,
                    "watch_ratio": 0.44,
                },
                {
                    "user_id": 11,
                    "video_id": 101,
                    "play_duration": 3300.0,
                    "video_duration": 8000.0,
                    "time": (day2 + pd.Timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200102,
                    "timestamp": self.kuairec_timestamp(day2 + pd.Timedelta(seconds=10)),
                    "watch_ratio": 0.41,
                },
                {
                    "user_id": 11,
                    "video_id": 102,
                    "play_duration": 4200.0,
                    "video_duration": 6000.0,
                    "time": (day2 + pd.Timedelta(seconds=1815)).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": 20200102,
                    "timestamp": self.kuairec_timestamp(day2 + pd.Timedelta(seconds=1815)),
                    "watch_ratio": 0.7,
                },
            ]
        )
        csv_path = root / "small_matrix_session_split.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def write_long_session_matrix_csv(
        self,
        root: Path,
        session_lengths,
        filename: str = "small_matrix_long_sessions.csv",
    ):
        base_time = pd.Timestamp("2020-01-01 00:00:00")
        session_user_ids = [10, 10, 11, 11, 10, 11]
        video_ids = [100, 101, 102]
        rows = []

        for session_idx, session_length in enumerate(session_lengths):
            user_id = session_user_ids[session_idx % len(session_user_ids)]
            session_start = base_time + pd.Timedelta(hours=2 * session_idx)
            for row_idx in range(int(session_length)):
                video_id = video_ids[row_idx % len(video_ids)]
                video_duration = {100: 5000.0, 101: 8000.0, 102: 6000.0}[video_id]
                play_duration = min(
                    video_duration,
                    1000.0 + 150.0 * ((row_idx % 5) + 1),
                )
                timestamp = self.kuairec_timestamp(
                    session_start + pd.Timedelta(seconds=row_idx)
                )
                rows.append(
                    {
                        "user_id": user_id,
                        "video_id": video_id,
                        "play_duration": play_duration,
                        "video_duration": video_duration,
                        "time": (session_start + pd.Timedelta(seconds=row_idx)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "date": int((session_start + pd.Timedelta(seconds=row_idx)).strftime("%Y%m%d")),
                        "timestamp": timestamp,
                        "watch_ratio": float(play_duration / max(video_duration, 1.0)),
                    }
                )

        csv_path = root / filename
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_watch_ratio_bucket_index(self):
        self.assertEqual(watch_ratio_to_bucket_index(0.0), 0)
        self.assertEqual(watch_ratio_to_bucket_index(0.34), 0)
        self.assertEqual(watch_ratio_to_bucket_index(0.35), 1)
        self.assertEqual(watch_ratio_to_bucket_index(0.55), 1)
        self.assertEqual(watch_ratio_to_bucket_index(0.6), 2)
        self.assertEqual(watch_ratio_to_bucket_index(0.75), 2)
        self.assertEqual(watch_ratio_to_bucket_index(0.8), 3)
        self.assertEqual(watch_ratio_to_bucket_index(1.0), 4)
        self.assertEqual(watch_ratio_to_bucket_index(1.3), 4)
        self.assertEqual(watch_ratio_to_bucket_index(1.31), 5)

    def test_threshold_banded_watch_ratio_components_follow_four_regimes(self):
        low_components = compute_threshold_banded_watch_ratio_components(
            video_score=0.70,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )
        first_bucket_components = compute_threshold_banded_watch_ratio_components(
            video_score=0.80,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )
        second_bucket_components = compute_threshold_banded_watch_ratio_components(
            video_score=0.95,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )
        high_components = compute_threshold_banded_watch_ratio_components(
            video_score=1.40,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )
        mid_first_bucket_components = compute_threshold_banded_watch_ratio_components(
            video_score=0.7875,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )

        self.assertGreaterEqual(low_components["pred_watch_ratio"], 0.0)
        self.assertLess(low_components["pred_watch_ratio"], 0.1)
        self.assertGreaterEqual(first_bucket_components["pred_watch_ratio"], 0.1)
        self.assertLess(first_bucket_components["pred_watch_ratio"], 0.35)
        self.assertGreaterEqual(second_bucket_components["pred_watch_ratio"], 0.35)
        self.assertLess(second_bucket_components["pred_watch_ratio"], 0.6)
        self.assertGreaterEqual(high_components["pred_watch_ratio"], 0.6)
        self.assertGreater(high_components["pred_watch_ratio"], second_bucket_components["pred_watch_ratio"])
        self.assertNotAlmostEqual(mid_first_bucket_components["pred_watch_ratio"], 0.1625, places=3)

    def test_repeat_watch_ratio_components_accumulate_repeated_views(self):
        repeated_components = compute_threshold_banded_watch_ratio_components(
            video_score=5.0,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.8,
        )
        lower_decay_components = compute_threshold_banded_watch_ratio_components(
            video_score=5.0,
            effective_threshold=1.0,
            watch_duration_scale=1.0,
            watch_gain_base=1.3,
            repeat_watch_decay=0.6,
        )

        self.assertGreater(repeated_components["base_watch_ratio"], 1.0)
        self.assertGreater(repeated_components["pred_watch_ratio"], 1.0)
        self.assertAlmostEqual(repeated_components["pass_contributions"][0], 1.0, places=6)
        self.assertGreater(repeated_components["pass_contributions"][1], 0.0)
        self.assertTrue(
            all(contribution <= 1.0 for contribution in repeated_components["pass_contributions"])
        )
        self.assertLessEqual(repeated_components["pred_watch_ratio"], float(REPEAT_WATCH_PASS_CAP))
        self.assertGreater(
            repeated_components["pred_watch_ratio"],
            lower_decay_components["pred_watch_ratio"],
        )

    def test_time_weighted_average_reward_weights_recent_history_more(self):
        self.assertAlmostEqual(
            _compute_scalar_time_weighted_average_reward([], 0.85),
            0.0,
            places=6,
        )

        old_high = _compute_scalar_time_weighted_average_reward([10.0, 0.0], 0.5)
        recent_high = _compute_scalar_time_weighted_average_reward([0.0, 10.0], 0.5)
        self.assertGreater(recent_high, old_high)

        values = [1.0, 3.0, 9.0]
        scalar_average = _compute_scalar_time_weighted_average_reward(values, 0.8)
        torch_average = _compute_torch_time_weighted_average_reward(
            recent_value_history=[
                torch.tensor(value, dtype=torch.float32) for value in values
            ],
            average_reward_decay=torch.tensor(0.8, dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertAlmostEqual(
            self.scalar(torch_average),
            scalar_average,
            places=6,
        )

    def test_load_kuairec_small_matrix_caps_real_watch_ratio_at_10(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "small_matrix.csv"
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "video_id": 100,
                        "play_duration": 100000.0,
                        "video_duration": 1000.0,
                        "time": "2020-01-01 00:00:00",
                        "date": 20200101,
                        "timestamp": self.kuairec_timestamp(pd.Timestamp("2020-01-01 00:00:00")),
                        "watch_ratio": 25.0,
                    },
                    {
                        "user_id": 10,
                        "video_id": 101,
                        "play_duration": 500.0,
                        "video_duration": 1000.0,
                        "time": "2020-01-01 00:01:00",
                        "date": 20200101,
                        "timestamp": self.kuairec_timestamp(pd.Timestamp("2020-01-01 00:01:00")),
                        "watch_ratio": 0.5,
                    },
                ]
            ).to_csv(csv_path, index=False)

            loaded = load_kuairec_small_matrix(csv_path)

            self.assertAlmostEqual(float(loaded["watch_ratio"].iloc[0]), 10.0, places=6)
            self.assertAlmostEqual(float(loaded["watch_ratio"].iloc[1]), 0.5, places=6)

    def test_novelty_norm(self):
        current = np.zeros(31, dtype=np.float32)
        self.assertAlmostEqual(compute_novelty_norm(current, []), 0.5)

        single = np.ones(31, dtype=np.float32)
        expected = float(np.clip(np.linalg.norm(single - current) / np.sqrt(31.0), 0.0, 1.0))
        self.assertAlmostEqual(compute_novelty_norm(current, [single]), expected)

        history = [np.ones(31, dtype=np.float32) * 100.0] + [np.zeros(31, dtype=np.float32) for _ in range(10)]
        self.assertAlmostEqual(compute_novelty_norm(current, history), 0.0)

    def test_creator_preference_norm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))
            strong = provider.compute_creator_preference_norm(10, 500)
            weak = provider.compute_creator_preference_norm(10, 501)

            self.assertGreater(strong, weak)
            self.assertEqual(provider.compute_creator_preference_norm(10, 9999), 0.0)

    def test_load_kuairec_small_matrix_parses_time_and_marks_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = self.write_session_split_small_matrix_csv(root)

            df = load_kuairec_small_matrix(csv_path)
            expected_fallback_ts = (
                pd.Timestamp("2020-01-01 00:00:00")
                .tz_localize("Asia/Shanghai")
                .timestamp()
            )

            self.assertIn("play_duration", df.columns)
            self.assertIn("event_timestamp", df.columns)
            self.assertIn("session_id", df.columns)
            self.assertIn("session_row_index", df.columns)
            self.assertAlmostEqual(
                float(df.loc[0, "event_timestamp"]),
                expected_fallback_ts,
                places=3,
            )
            self.assertTrue(np.isnan(float(df.loc[3, "event_timestamp"])))
            self.assertEqual(df["session_id"].tolist(), [0, 0, 1, 1, 1, 1, 0, 0, 1])
            self.assertEqual(df["session_row_index"].tolist(), [0, 1, 0, 1, 2, 3, 0, 1, 0])

    def test_load_kuairec_small_matrix_prefers_timestamp_over_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "small_matrix_timestamp_priority.csv"
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "video_id": 100,
                        "play_duration": 5000.0,
                        "video_duration": 5000.0,
                        "time": "2020-01-01 00:00:00",
                        "date": 20200101,
                        "timestamp": 1577836800.123,
                        "watch_ratio": 1.0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            df = load_kuairec_small_matrix(csv_path)
            self.assertAlmostEqual(float(df.loc[0, "event_timestamp"]), 1577836800.123, places=3)

    def test_training_state_updates(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        params = model._params()
        dopamine_expected_reward_coef = self.scalar(params["dopamine_expected_reward_coef"])
        dopamine_peak_coef = self.scalar(params["dopamine_peak_coef"])
        dopamine_base = self.scalar(params["dopamine_base"])
        baseline_return_strength = self.scalar(params["dopamine_baseline_return_strength"])
        habit_growth_rate = self.scalar(params["dopamine_habit_growth_rate"])
        habit_max_gain = self.scalar(params["dopamine_habit_max_gain"])
        fatigue_watch_ratio_coef = self.scalar(params["fatigue_watch_ratio_coef"])
        fatigue_high_engaged_bonus = self.scalar(params["fatigue_high_engaged_bonus"])

        base_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        base_state["fatigue"] = torch.tensor(0.4, dtype=torch.float32)
        base_state["dopamine"] = torch.tensor(1.2, dtype=torch.float32)
        base_state["recent_score_history"] = []

        aux = {
            "video_score": torch.tensor(2.0, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }

        no_watch = model.update_state_with_real_feedback(
            state=base_state,
            y_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            y_bucket=torch.tensor(0, dtype=torch.long),
            aux=aux,
        )
        self.assertAlmostEqual(self.scalar(no_watch["fatigue"]), 0.4, places=6)
        self.assertAlmostEqual(self.scalar(no_watch["session_peak"]), 0.0)
        self.assertAlmostEqual(self.scalar(no_watch["expected_reward"]), 0.0)
        self.assertAlmostEqual(self.scalar(no_watch["swipe_count"]), 1.0, places=6)
        self.assertAlmostEqual(
            self.scalar(no_watch["dopamine"]),
            self.scalar(base_state["dopamine"])
            + baseline_return_strength * (dopamine_base - self.scalar(base_state["dopamine"])),
            places=6,
        )
        self.assertEqual(len(no_watch["recent_category_history"]), 1)

        light_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        high_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)

        light = model.update_state_with_real_feedback(
            state=light_state,
            y_watch_ratio=torch.tensor(0.8, dtype=torch.float32),
            y_bucket=torch.tensor(4, dtype=torch.long),
            aux=aux,
        )
        high = model.update_state_with_real_feedback(
            state=high_state,
            y_watch_ratio=torch.tensor(1.2, dtype=torch.float32),
            y_bucket=torch.tensor(5, dtype=torch.long),
            aux=aux,
        )

        self.assertGreater(self.scalar(light["fatigue"]), self.scalar(light_state["fatigue"]))
        self.assertGreater(self.scalar(high["fatigue"]), self.scalar(light["fatigue"]))
        self.assertAlmostEqual(
            self.scalar(light["fatigue"]),
            fatigue_watch_ratio_coef * 0.8,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(high["fatigue"]),
            min(1.0, fatigue_watch_ratio_coef * 1.2 + fatigue_high_engaged_bonus),
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(light["expected_reward"]),
            self.expected_reward_from_history(
                params,
                [self.score_engagement(2.0, 0.8)],
            )[0],
            places=6,
        )
        light_reward_drive = (
            dopamine_expected_reward_coef * self.scalar(light["expected_reward"])
            + dopamine_peak_coef * self.scalar(light["session_peak"])
        )
        light_expected_dopamine = _apply_scalar_dopamine_update_scaffold(
            prev_dopamine=self.scalar(light_state["dopamine"]),
            session_baseline=dopamine_base,
            baseline_return_strength=baseline_return_strength,
            swipe_count=1.0,
            habit_growth_rate=habit_growth_rate,
            habit_max_gain=habit_max_gain,
            reward_drive=light_reward_drive,
        )[0]
        self.assertAlmostEqual(self.scalar(light["dopamine"]), light_expected_dopamine, places=6)
        self.assertGreater(self.scalar(high["expected_reward"]), self.scalar(light["expected_reward"]))
        self.assertGreater(self.scalar(high["dopamine"]), self.scalar(light["dopamine"]))
        self.assertAlmostEqual(self.scalar(high["swipe_count"]), 1.0, places=6)

        score_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        score_state["recent_score_history"] = [torch.tensor(2.0, dtype=torch.float32)]
        score_state["fatigue"] = torch.tensor(0.1, dtype=torch.float32)
        high_fatigue_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        high_fatigue_state["recent_score_history"] = [torch.tensor(2.0, dtype=torch.float32)]
        high_fatigue_state["fatigue"] = torch.tensor(0.9, dtype=torch.float32)

        score_aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "quality_norm": torch.tensor(0.2, dtype=torch.float32),
            "category_vec": torch.zeros(31, dtype=torch.float32),
        }
        low_fatigue_next = model.update_state_with_real_feedback(
            state=score_state,
            y_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            y_bucket=torch.tensor(0, dtype=torch.long),
            aux=score_aux,
        )
        high_fatigue_next = model.update_state_with_real_feedback(
            state=high_fatigue_state,
            y_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            y_bucket=torch.tensor(0, dtype=torch.long),
            aux=score_aux,
        )

        self.assertGreater(self.scalar(low_fatigue_next["session_peak"]), 1.5)
        self.assertAlmostEqual(
            self.scalar(low_fatigue_next["dopamine"]),
            self.scalar(high_fatigue_next["dopamine"]),
            places=5,
        )

        low_expected_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        high_expected_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        low_expected_aux = {
            "video_score": torch.tensor(0.2, dtype=torch.float32),
            "quality_norm": torch.tensor(0.2, dtype=torch.float32),
            "category_vec": torch.zeros(31, dtype=torch.float32),
        }
        high_expected_aux = {
            "video_score": torch.tensor(0.9, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.zeros(31, dtype=torch.float32),
        }
        low_expected_next = model.update_state_with_real_feedback(
            state=low_expected_state,
            y_watch_ratio=torch.tensor(1.0, dtype=torch.float32),
            y_bucket=torch.tensor(4, dtype=torch.long),
            aux=low_expected_aux,
        )
        high_expected_next = model.update_state_with_real_feedback(
            state=high_expected_state,
            y_watch_ratio=torch.tensor(1.0, dtype=torch.float32),
            y_bucket=torch.tensor(4, dtype=torch.long),
            aux=high_expected_aux,
        )
        self.assertGreater(self.scalar(high_expected_next["expected_reward"]), self.scalar(low_expected_next["expected_reward"]))
        self.assertGreater(self.scalar(high_expected_next["dopamine"]), self.scalar(low_expected_next["dopamine"]))

        low_peak_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        high_peak_state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        low_peak_state["recent_score_history"] = [torch.tensor(0.2, dtype=torch.float32)]
        high_peak_state["recent_score_history"] = [torch.tensor(2.0, dtype=torch.float32)]
        same_aux = {
            "video_score": torch.tensor(0.0, dtype=torch.float32),
            "quality_norm": torch.tensor(0.0, dtype=torch.float32),
            "category_vec": torch.zeros(31, dtype=torch.float32),
        }
        low_peak_next = model.update_state_with_real_feedback(
            state=low_peak_state,
            y_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            y_bucket=torch.tensor(0, dtype=torch.long),
            aux=same_aux,
        )
        high_peak_next = model.update_state_with_real_feedback(
            state=high_peak_state,
            y_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            y_bucket=torch.tensor(0, dtype=torch.long),
            aux=same_aux,
        )
        self.assertAlmostEqual(
            self.scalar(low_peak_next["dopamine"]),
            _apply_scalar_dopamine_update_scaffold(
                prev_dopamine=self.scalar(low_peak_state["dopamine"]),
                session_baseline=dopamine_base,
                baseline_return_strength=baseline_return_strength,
                swipe_count=1.0,
                habit_growth_rate=habit_growth_rate,
                habit_max_gain=habit_max_gain,
                reward_drive=(
                    dopamine_expected_reward_coef * self.scalar(low_peak_next["expected_reward"])
                    + dopamine_peak_coef * self.scalar(low_peak_next["session_peak"])
                ),
            )[0],
            places=5,
        )
        self.assertGreater(self.scalar(high_peak_next["session_peak"]), self.scalar(low_peak_next["session_peak"]))
        self.assertGreater(self.scalar(high_peak_next["dopamine"]), self.scalar(low_peak_next["dopamine"]))

    def test_prediction_state_updates(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        params = model._params()
        dopamine_expected_reward_coef = self.scalar(params["dopamine_expected_reward_coef"])
        dopamine_peak_coef = self.scalar(params["dopamine_peak_coef"])
        dopamine_base = self.scalar(params["dopamine_base"])
        baseline_return_strength = self.scalar(params["dopamine_baseline_return_strength"])
        habit_growth_rate = self.scalar(params["dopamine_habit_growth_rate"])
        habit_max_gain = self.scalar(params["dopamine_habit_max_gain"])

        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        state["fatigue"] = torch.tensor(0.4, dtype=torch.float32)
        state["dopamine"] = torch.tensor(1.2, dtype=torch.float32)

        aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(1.2, dtype=torch.float32),
            aux=aux,
        )

        self.assertGreater(self.scalar(next_state["fatigue"]), 0.4)
        self.assertAlmostEqual(
            self.scalar(next_state["expected_reward"]),
            self.expected_reward_from_history(
                params,
                [self.score_engagement(1.5, 1.2)],
            )[0],
            places=6,
        )
        self.assertGreater(self.scalar(next_state["session_peak"]), 1.0)
        self.assertAlmostEqual(
            self.scalar(next_state["dopamine"]),
            _apply_scalar_dopamine_update_scaffold(
                prev_dopamine=self.scalar(state["dopamine"]),
                session_baseline=dopamine_base,
                baseline_return_strength=baseline_return_strength,
                swipe_count=1.0,
                habit_growth_rate=habit_growth_rate,
                habit_max_gain=habit_max_gain,
                reward_drive=(
                    dopamine_expected_reward_coef * self.scalar(next_state["expected_reward"])
                    + dopamine_peak_coef * self.scalar(next_state["session_peak"])
                ),
            )[0],
            places=5,
        )
        self.assertAlmostEqual(self.scalar(next_state["swipe_count"]), 1.0, places=6)

    def test_behavior_fit_model_uses_baseline_return_when_reward_drive_is_zero(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        with torch.no_grad():
            model.dopamine_base_raw.fill_(4.0)
            model.dopamine_mix_alpha_raw.fill_(2.0)
            model.dopamine_score_engagement_coef_raw.fill_(0.0)
            model.dopamine_expected_reward_coef_raw.fill_(0.0)
            model.dopamine_peak_coef_raw.fill_(0.0)

        params = model._params()
        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        state["dopamine"] = torch.tensor(1.5, dtype=torch.float32)

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            aux={
                "video_score": torch.tensor(0.0, dtype=torch.float32),
                "quality_norm": torch.tensor(0.0, dtype=torch.float32),
                "category_vec": torch.zeros(31, dtype=torch.float32),
            },
        )

        expected_dopamine = self.scalar(state["dopamine"]) + self.scalar(
            params["dopamine_baseline_return_strength"]
        ) * (self.scalar(params["dopamine_base"]) - self.scalar(state["dopamine"]))
        self.assertAlmostEqual(
            self.scalar(next_state["dopamine"]),
            expected_dopamine,
            places=6,
        )

    def test_behavior_fit_model_farther_from_baseline_returns_faster(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        with torch.no_grad():
            model.dopamine_expected_reward_coef_raw.fill_(0.0)
            model.dopamine_peak_coef_raw.fill_(0.0)

        low_state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high_state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        low_state["dopamine"] = torch.tensor(0.6, dtype=torch.float32)
        high_state["dopamine"] = torch.tensor(2.6, dtype=torch.float32)

        aux = {
            "video_score": torch.tensor(0.0, dtype=torch.float32),
            "quality_norm": torch.tensor(0.0, dtype=torch.float32),
            "category_vec": torch.zeros(31, dtype=torch.float32),
        }

        low_next = model.update_state_with_prediction(
            state=low_state,
            pred_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            aux=aux,
        )
        high_next = model.update_state_with_prediction(
            state=high_state,
            pred_watch_ratio=torch.tensor(0.0, dtype=torch.float32),
            aux=aux,
        )

        low_step = abs(self.scalar(low_next["dopamine"]) - self.scalar(low_state["dopamine"]))
        high_step = abs(self.scalar(high_next["dopamine"]) - self.scalar(high_state["dopamine"]))
        self.assertGreater(high_step, low_step)

    def test_behavior_fit_habit_progress_is_monotone_and_slow(self):
        model = BehaviorFitModel(embedding_dim=32)
        params = model._params()
        growth = self.scalar(params["dopamine_habit_growth_rate"])
        max_gain = self.scalar(params["dopamine_habit_max_gain"])
        steps = (0, 1, 5, 10, 25, 50)

        progress_values = [
            _compute_scalar_dopamine_habit_progress(step, growth)
            for step in steps
        ]
        coef_values = [self.habit_coef(growth, max_gain, step) for step in steps]

        self.assertTrue(all(left <= right for left, right in zip(progress_values, progress_values[1:])))
        self.assertTrue(all(left <= right for left, right in zip(coef_values, coef_values[1:])))
        early_one_step = _compute_scalar_dopamine_habit_progress(1, growth) - _compute_scalar_dopamine_habit_progress(0, growth)
        late_one_step = _compute_scalar_dopamine_habit_progress(50, growth) - _compute_scalar_dopamine_habit_progress(49, growth)
        self.assertGreater(early_one_step, late_one_step)
        self.assertLess(progress_values[steps.index(10)], 0.15)
        self.assertLess(progress_values[-1], 0.7)
        self.assertLessEqual(coef_values[-1], 1.0 + max_gain + 1e-6)

    def test_post_history_peak_reward_dopamine_state_updates(self):
        model = PostHistoryPeakRewardDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        with torch.no_grad():
            model.dopamine_base_raw.fill_(0.9)

        params = model._params()
        reward_coef = self.scalar(params["dopamine_expected_reward_coef"])
        peak_coef = self.scalar(params["dopamine_peak_coef"])
        baseline_return_strength = self.scalar(params["dopamine_baseline_return_strength"])
        habit_growth_rate = self.scalar(params["dopamine_habit_growth_rate"])
        habit_max_gain = self.scalar(params["dopamine_habit_max_gain"])
        fatigue_watch_ratio_coef = self.scalar(params["fatigue_watch_ratio_coef"])

        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        state["dopamine"] = torch.tensor(1.3, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.25, dtype=torch.float32)

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(0.75, dtype=torch.float32),
            aux={
                "video_score": torch.tensor(2.0, dtype=torch.float32),
                "quality_norm": torch.tensor(0.4, dtype=torch.float32),
                "category_vec": torch.ones(31, dtype=torch.float32),
            },
        )

        expected_score_engagement = self.score_engagement(2.0, 0.75)
        expected_reward, expected_peak, _ = self.expected_reward_from_history(
            params,
            [expected_score_engagement],
        )

        self.assertEqual(len(next_state["recent_score_history"]), 1)
        self.assertAlmostEqual(
            self.scalar(next_state["recent_score_history"][0]),
            expected_score_engagement,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["expected_reward"]),
            expected_reward,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["session_peak"]),
            expected_peak,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["dopamine"]),
            _apply_scalar_dopamine_update_scaffold(
                prev_dopamine=self.scalar(state["dopamine"]),
                session_baseline=self.scalar(params["dopamine_base"]),
                baseline_return_strength=baseline_return_strength,
                swipe_count=1.0,
                habit_growth_rate=habit_growth_rate,
                habit_max_gain=habit_max_gain,
                reward_drive=reward_coef * expected_reward + peak_coef * expected_peak,
            )[0],
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["fatigue"]),
            0.25 + fatigue_watch_ratio_coef * 0.75,
            places=6,
        )

    def test_integrated_signal_dopamine_state_updates(self):
        model = IntegratedSignalDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        with torch.no_grad():
            model.dopamine_base_raw.fill_(1.2)
            model.dopamine_score_engagement_coef_raw.fill_(
                torch.log(torch.expm1(torch.tensor(0.6)))
            )
            model.dopamine_expected_reward_coef_raw.fill_(
                torch.log(torch.expm1(torch.tensor(0.4)))
            )
            model.dopamine_peak_coef_raw.fill_(
                torch.log(torch.expm1(torch.tensor(0.3)))
            )
            model.dopamine_novelty_coef_raw.fill_(
                torch.log(torch.expm1(torch.tensor(0.2)))
            )

        params = model._params()
        fatigue_watch_ratio_coef = self.scalar(params["fatigue_watch_ratio_coef"])
        score_engagement_coef = self.scalar(params["dopamine_score_engagement_coef"])
        novelty_coef = self.scalar(params["dopamine_novelty_coef"])
        reward_coef = self.scalar(params["dopamine_expected_reward_coef"])
        peak_coef = self.scalar(params["dopamine_peak_coef"])
        dopamine_base = self.scalar(params["dopamine_base"])
        baseline_return_strength = self.scalar(params["dopamine_baseline_return_strength"])
        habit_growth_rate = self.scalar(params["dopamine_habit_growth_rate"])
        habit_max_gain = self.scalar(params["dopamine_habit_max_gain"])

        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        state["dopamine"] = torch.tensor(4.5, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.25, dtype=torch.float32)

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(0.75, dtype=torch.float32),
            aux={
                "video_score": torch.tensor(2.0, dtype=torch.float32),
                "novelty_norm": torch.tensor(0.6, dtype=torch.float32),
                "quality_norm": torch.tensor(0.4, dtype=torch.float32),
                "category_vec": torch.ones(31, dtype=torch.float32),
            },
        )

        expected_score_engagement = self.score_engagement(2.0, 0.75)
        expected_reward, expected_peak, _ = self.expected_reward_from_history(
            params,
            [expected_score_engagement],
        )
        expected_dopamine = _apply_scalar_dopamine_update_scaffold(
            prev_dopamine=self.scalar(state["dopamine"]),
            session_baseline=dopamine_base,
            baseline_return_strength=baseline_return_strength,
            swipe_count=1.0,
            habit_growth_rate=habit_growth_rate,
            habit_max_gain=habit_max_gain,
            reward_drive=(
                score_engagement_coef * expected_score_engagement
                + novelty_coef * 0.6
                + reward_coef * expected_reward
                + peak_coef * expected_peak
            ),
        )[0]

        self.assertEqual(len(next_state["recent_score_history"]), 1)
        self.assertAlmostEqual(
            self.scalar(next_state["recent_score_history"][0]),
            expected_score_engagement,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["expected_reward"]),
            expected_reward,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["session_peak"]),
            expected_peak,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["dopamine"]),
            expected_dopamine,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["fatigue"]),
            0.25 + fatigue_watch_ratio_coef * 0.75,
            places=6,
        )

    def test_integrated_signal_dopamine_depends_on_previous_dopamine_via_baseline_return(self):
        model = IntegratedSignalDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        base_state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        base_state["fatigue"] = torch.tensor(0.1, dtype=torch.float32)

        aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "novelty_norm": torch.tensor(0.4, dtype=torch.float32),
            "quality_norm": torch.tensor(0.2, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }

        low_dopamine_state = dict(base_state)
        low_dopamine_state["dopamine"] = torch.tensor(0.5, dtype=torch.float32)
        high_dopamine_state = dict(base_state)
        high_dopamine_state["dopamine"] = torch.tensor(6.0, dtype=torch.float32)

        low_next = model.update_state_with_prediction(
            state=low_dopamine_state,
            pred_watch_ratio=torch.tensor(0.8, dtype=torch.float32),
            aux=aux,
        )
        high_next = model.update_state_with_prediction(
            state=high_dopamine_state,
            pred_watch_ratio=torch.tensor(0.8, dtype=torch.float32),
            aux=aux,
        )

        self.assertAlmostEqual(
            self.scalar(low_next["expected_reward"]),
            self.scalar(high_next["expected_reward"]),
            places=6,
        )
        self.assertNotAlmostEqual(
            self.scalar(low_next["dopamine"]),
            self.scalar(high_next["dopamine"]),
            places=6,
        )

    def test_relax_to_higher_baseline_dopamine_state_updates(self):
        model = RelaxToHigherBaselineDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")

        with torch.no_grad():
            model.dopamine_base_raw.fill_(1.0)
            model.dopamine_normal_level_gap_raw.fill_(torch.log(torch.expm1(torch.tensor(0.5))))

        params = model._params()
        fatigue_watch_ratio_coef = self.scalar(params["fatigue_watch_ratio_coef"])
        dopamine_base = self.scalar(params["dopamine_base"])
        dopamine_true_baseline = self.scalar(params["dopamine_true_baseline"])
        baseline_return_strength = self.scalar(params["dopamine_baseline_return_strength"])
        habit_growth_rate = self.scalar(params["dopamine_habit_growth_rate"])
        habit_max_gain = self.scalar(params["dopamine_habit_max_gain"])
        reward_coef = self.scalar(params["dopamine_expected_reward_coef"])
        peak_coef = self.scalar(params["dopamine_peak_coef"])

        self.assertGreater(dopamine_true_baseline, dopamine_base)

        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        self.assertAlmostEqual(self.scalar(state["dopamine"]), dopamine_base, places=6)

        state["dopamine"] = torch.tensor(1.25, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.25, dtype=torch.float32)

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(0.75, dtype=torch.float32),
            aux={
                "video_score": torch.tensor(2.0, dtype=torch.float32),
                "quality_norm": torch.tensor(0.4, dtype=torch.float32),
                "category_vec": torch.ones(31, dtype=torch.float32),
            },
        )

        expected_score_engagement = self.score_engagement(2.0, 0.75)
        expected_reward, expected_peak, _ = self.expected_reward_from_history(
            params,
            [expected_score_engagement],
        )
        expected_dopamine = _apply_scalar_dopamine_update_scaffold(
            prev_dopamine=self.scalar(state["dopamine"]),
            session_baseline=dopamine_base,
            baseline_return_strength=baseline_return_strength,
            swipe_count=1.0,
            habit_growth_rate=habit_growth_rate,
            habit_max_gain=habit_max_gain,
            reward_drive=(
                reward_coef * expected_reward
                + peak_coef * expected_peak
                + expected_score_engagement
            ),
        )[0]

        self.assertEqual(len(next_state["recent_score_history"]), 1)
        self.assertAlmostEqual(
            self.scalar(next_state["recent_score_history"][0]),
            expected_score_engagement,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["expected_reward"]),
            expected_reward,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["session_peak"]),
            expected_peak,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["dopamine"]),
            expected_dopamine,
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(next_state["fatigue"]),
            0.25 + fatigue_watch_ratio_coef * 0.75,
            places=6,
        )

    def test_strict_zero_dopamine_eval_copy_zeroes_dopamine_path(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        ablated = model.make_strict_zero_dopamine_eval_copy()

        params = ablated._params()
        for key in DOPAMINE_PARAM_KEYS:
            self.assertAlmostEqual(self.scalar(params[key]), 0.0, places=7)

        state = ablated.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        self.assertAlmostEqual(self.scalar(state["dopamine"]), 0.0, places=7)

        state["fatigue"] = torch.tensor(0.2, dtype=torch.float32)
        state["dopamine"] = torch.tensor(2.5, dtype=torch.float32)
        state["session_peak"] = torch.tensor(1.4, dtype=torch.float32)

        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 1.0,
            "creator_pref_norm": 0.2,
        }

        with torch.no_grad():
            ablated.bucket_w.zero_()
            ablated.bucket_w[0, 4] = 1.0

        bucket_logits, _, aux = ablated.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        self.assertAlmostEqual(self.scalar(aux["current_dopamine"]), 0.0, places=7)
        self.assertAlmostEqual(self.scalar(bucket_logits[0]), 0.0, places=7)

        next_state = ablated.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(1.3, dtype=torch.float32),
            aux=aux,
        )
        self.assertAlmostEqual(self.scalar(next_state["dopamine"]), 0.0, places=7)

    def test_strict_zero_dopamine_eval_copy_preserves_non_dopamine_weights(self):
        model = BehaviorFitModel(embedding_dim=32)

        with torch.no_grad():
            model.bucket_w.copy_(torch.arange(42, dtype=torch.float32).reshape(6, 7))
            model.score_quality_raw.copy_(torch.tensor(1.2345, dtype=torch.float32))

        ablated = model.make_strict_zero_dopamine_eval_copy()

        self.assertTrue(torch.allclose(model.bucket_w, ablated.bucket_w))
        self.assertAlmostEqual(
            self.scalar(model.score_quality_raw),
            self.scalar(ablated.score_quality_raw),
            places=7,
        )

    def test_no_dopamine_prediction_state_updates(self):
        model = NoDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        self.assertNotIn("expected_reward", state)
        self.assertNotIn("session_peak", state)
        self.assertNotIn("recent_score_history", state)
        state["fatigue"] = torch.tensor(0.4, dtype=torch.float32)

        aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }

        next_state = model.update_state_with_prediction(
            state=state,
            pred_watch_ratio=torch.tensor(1.2, dtype=torch.float32),
            aux=aux,
        )

        self.assertNotIn("dopamine", next_state)
        self.assertNotIn("expected_reward", next_state)
        self.assertNotIn("session_peak", next_state)
        self.assertNotIn("recent_score_history", next_state)
        self.assertGreater(self.scalar(next_state["fatigue"]), 0.4)
        self.assertEqual(len(next_state["recent_category_history"]), 1)

    def test_forward_one_current_dopamine_is_not_clipped(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        state["dopamine"] = torch.tensor(3.5, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.0, dtype=torch.float32)

        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 1.0,
            "creator_pref_norm": 0.0,
        }

        _, _, aux = model.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        expected_current_dopamine = self.scalar(state["dopamine"])
        self.assertGreater(expected_current_dopamine, 3.2)
        self.assertAlmostEqual(
            self.scalar(aux["current_dopamine"]),
            expected_current_dopamine,
            places=6,
        )

    def test_forward_one_watch_ratio_threshold_ignores_current_dopamine(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 1.0,
            "creator_pref_norm": 0.0,
        }
        low_dopamine = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high_dopamine = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        low_dopamine["dopamine"] = torch.tensor(0.0, dtype=torch.float32)
        high_dopamine["dopamine"] = torch.tensor(5.0, dtype=torch.float32)

        _, low_pred, low_aux = model.forward_one(
            state=low_dopamine,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, high_pred, high_aux = model.forward_one(
            state=high_dopamine,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(low_aux["effective_threshold"]),
            self.scalar(high_aux["effective_threshold"]),
            places=6,
        )
        self.assertAlmostEqual(self.scalar(low_pred), self.scalar(high_pred), places=6)

    def test_forward_one_expected_reward_and_automaticity_raise_threshold(self):
        model = BehaviorFitModel(embedding_dim=32, user_ids=[10, 11])
        device = torch.device("cpu")
        low = model.init_state(
            user_feat={"user_id": 10, "user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high = model.init_state(
            user_feat={"user_id": 11, "user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        with torch.no_grad():
            model.raw_user_automaticity[0].fill_(-4.0)
            model.raw_user_automaticity[1].fill_(4.0)

        low["user_automaticity"] = model._resolve_user_automaticity(
            {"user_id": 10},
            device=device,
        )
        high["user_automaticity"] = model._resolve_user_automaticity(
            {"user_id": 11},
            device=device,
        )
        low["expected_reward"] = torch.tensor(0.0, dtype=torch.float32)
        high["expected_reward"] = torch.tensor(2.0, dtype=torch.float32)

        user_feat = {"user_id": 10, "user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.8,
            "creator_pref_norm": 0.1,
        }
        _, _, low_aux = model.forward_one(
            state=low,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, _, high_aux = model.forward_one(
            state=high,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        self.assertGreater(
            self.scalar(high_aux["effective_threshold"]),
            self.scalar(low_aux["effective_threshold"]),
        )

    def test_user_automaticity_is_user_fixed_with_default_for_unknown_users(self):
        model = BehaviorFitModel(embedding_dim=32, user_ids=[10])
        device = torch.device("cpu")
        with torch.no_grad():
            model.raw_user_automaticity[0].fill_(2.0)
            model.user_automaticity_default_raw.fill_(-2.0)

        first_session = model.init_state(
            user_feat={"user_id": 10, "user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        second_session = model.init_state(
            user_feat={"user_id": 10, "user_vec": np.ones(32, dtype=np.float32)},
            device=device,
        )
        unknown_user = model.init_state(
            user_feat={"user_id": 99, "user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(first_session["user_automaticity"]),
            self.scalar(second_session["user_automaticity"]),
            places=6,
        )
        self.assertGreater(
            self.scalar(first_session["user_automaticity"]),
            self.scalar(unknown_user["user_automaticity"]),
        )

    def test_behavior_fit_forward_ignores_legacy_fatigue_threshold_and_watch_gain_dopamine(self):
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.8,
            "creator_pref_norm": 0.3,
        }

        baseline = BehaviorFitModel(embedding_dim=32)
        legacy_shifted = BehaviorFitModel(embedding_dim=32)
        legacy_shifted.load_state_dict(baseline.state_dict())

        with torch.no_grad():
            legacy_shifted.effective_threshold_fatigue_raw.fill_(12.0)
            legacy_shifted.watch_gain_dopamine_raw.fill_(12.0)

        base_state = baseline.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        shifted_state = legacy_shifted.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        for state in (base_state, shifted_state):
            state["dopamine"] = torch.tensor(1.5, dtype=torch.float32)
            state["fatigue"] = torch.tensor(0.6, dtype=torch.float32)

        _, baseline_pred, baseline_aux = baseline.forward_one(
            state=base_state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, shifted_pred, shifted_aux = legacy_shifted.forward_one(
            state=shifted_state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(baseline_aux["effective_threshold"]),
            self.scalar(shifted_aux["effective_threshold"]),
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(baseline_pred),
            self.scalar(shifted_pred),
            places=6,
        )

    def test_behavior_fit_forward_fatigue_only_compresses_duration(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.8,
            "creator_pref_norm": 0.1,
        }

        low_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        low_fatigue["dopamine"] = torch.tensor(1.2, dtype=torch.float32)
        high_fatigue["dopamine"] = torch.tensor(1.2, dtype=torch.float32)
        low_fatigue["fatigue"] = torch.tensor(0.0, dtype=torch.float32)
        high_fatigue["fatigue"] = torch.tensor(0.7, dtype=torch.float32)

        _, low_pred, low_aux = model.forward_one(
            state=low_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, high_pred, high_aux = model.forward_one(
            state=high_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(low_aux["effective_threshold"]),
            self.scalar(high_aux["effective_threshold"]),
            places=6,
        )
        self.assertLess(self.scalar(high_pred), self.scalar(low_pred))

    def test_behavior_fit_forward_repeat_watch_accumulates_above_one(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 1.0,
            "creator_pref_norm": 1.0,
        }

        with torch.no_grad():
            model.effective_threshold_base_raw.fill_(-10.0)
            model.effective_threshold_dopamine_raw.fill_(-10.0)
            model.effective_threshold_expected_reward_raw.fill_(-10.0)
            model.effective_threshold_automaticity_raw.fill_(-10.0)
            model.watch_gain_base_raw.fill_(3.0)
            model.repeat_watch_decay_raw.fill_(float(np.log(0.8 / 0.2)))

        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        state["dopamine"] = torch.tensor(2.0, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.0, dtype=torch.float32)

        _, pred_watch_ratio, aux = model.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        params = model._params()
        watch_duration_scale = torch.exp(
            -params["fatigue_duration_penalty"] * state["fatigue"]
        )
        expected_components = compute_threshold_banded_watch_ratio_components(
            video_score=self.scalar(aux["video_score"]),
            effective_threshold=self.scalar(aux["effective_threshold"]),
            watch_duration_scale=self.scalar(watch_duration_scale),
            watch_gain_base=self.scalar(params["watch_gain_base"]),
            repeat_watch_decay=self.scalar(params["repeat_watch_decay"]),
        )

        self.assertGreater(expected_components["base_watch_ratio"], 1.0)
        self.assertGreater(self.scalar(pred_watch_ratio), expected_components["base_watch_ratio"])
        self.assertAlmostEqual(
            self.scalar(pred_watch_ratio),
            expected_components["pred_watch_ratio"],
            places=6,
        )
        self.assertGreater(self.scalar(pred_watch_ratio), 1.0)

    def test_no_dopamine_forward_fatigue_only_compresses_duration(self):
        model = NoDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.8,
            "creator_pref_norm": 0.1,
        }

        low_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        low_fatigue["fatigue"] = torch.tensor(0.0, dtype=torch.float32)
        high_fatigue["fatigue"] = torch.tensor(0.7, dtype=torch.float32)

        _, low_pred, low_aux = model.forward_one(
            state=low_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, high_pred, high_aux = model.forward_one(
            state=high_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(low_aux["effective_threshold"]),
            self.scalar(high_aux["effective_threshold"]),
            places=6,
        )
        self.assertLess(self.scalar(high_pred), self.scalar(low_pred))

    def test_behavior_fit_export_env_params_removes_deleted_forward_fields_and_adds_repeat_decay(self):
        params = BehaviorFitModel(embedding_dim=32).export_env_params()
        self.assertIn("effective_threshold_fatigue", params)
        self.assertIn("watch_gain_dopamine", params)
        self.assertIn("effective_threshold_expected_reward", params)
        self.assertIn("effective_threshold_automaticity", params)
        self.assertIn("expected_reward_peak_coef", params)
        self.assertIn("expected_reward_average_coef", params)
        self.assertIn("average_reward_decay", params)
        self.assertIn("user_automaticity_default", params)
        self.assertIn("user_automaticity_by_id", params)
        self.assertIn("repeat_watch_decay", params)
        self.assertIn("repeat_pass_cap", params)
        self.assertIn("dopamine_normal_level", params)
        self.assertIn("dopamine_baseline_return_strength", params)
        self.assertIn("dopamine_habit_growth_rate", params)
        self.assertIn("dopamine_habit_max_gain", params)
        self.assertNotIn("effective_sharpness", params)
        self.assertNotIn("dopamine_duration_penalty", params)
        self.assertNotIn("dopamine_high_watch_penalty", params)

    def test_bucket_head_uses_ordered_raw_features(self):
        model = BehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        state["dopamine"] = torch.tensor(1.7, dtype=torch.float32)
        state["fatigue"] = torch.tensor(0.25, dtype=torch.float32)
        state["expected_reward"] = torch.tensor(0.8, dtype=torch.float32)
        state["session_peak"] = torch.tensor(0.8, dtype=torch.float32)
        state["recent_category_history"] = [torch.ones(31, dtype=torch.float32)]

        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.4,
            "creator_pref_norm": 0.2,
        }

        with torch.no_grad():
            model.bucket_w.zero_()
            model.bucket_w[0, 0] = 1.0
            model.bucket_w[1, 1] = 1.0
            model.bucket_w[2, 2] = 1.0
            model.bucket_w[3, 3] = 1.0
            model.bucket_w[4, 4] = 1.0
            model.bucket_w[5, 5] = 1.0
            model.bucket_w[5, 6] = 1.0

        bucket_logits, _, aux = model.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        expected_pref_match = 1.0 / (1.0 + np.exp(-(32.0 / np.sqrt(32.0))))
        expected_novelty = np.sqrt(31.0) / np.sqrt(31.0)
        expected_automaticity = self.scalar(state["user_automaticity"])

        self.assertEqual(tuple(bucket_logits.shape), (6,))
        self.assertAlmostEqual(self.scalar(bucket_logits[0]), expected_pref_match, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[1]), 0.4, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[2]), expected_novelty, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[3]), 0.2, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[4]), self.scalar(state["expected_reward"]), places=6)
        self.assertAlmostEqual(
            self.scalar(bucket_logits[5]),
            expected_automaticity + self.scalar(state["fatigue"]),
            places=6,
        )

        baseline_bucket_logits = bucket_logits.detach().clone()
        state["session_peak"] = torch.tensor(9.5, dtype=torch.float32)

        shifted_bucket_logits, _, _ = model.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertTrue(torch.allclose(baseline_bucket_logits, shifted_bucket_logits))

    def test_no_dopamine_model_state_updates(self):
        model = NoDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        self.assertNotIn("dopamine", state)
        self.assertNotIn("expected_reward", state)
        self.assertNotIn("session_peak", state)
        self.assertNotIn("recent_score_history", state)

        state["fatigue"] = torch.tensor(0.4, dtype=torch.float32)
        aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }
        next_state = model.update_state_with_real_feedback(
            state=state,
            y_watch_ratio=torch.tensor(1.2, dtype=torch.float32),
            y_bucket=torch.tensor(5, dtype=torch.long),
            aux=aux,
        )

        self.assertNotIn("dopamine", next_state)
        self.assertNotIn("expected_reward", next_state)
        self.assertNotIn("session_peak", next_state)
        self.assertNotIn("recent_score_history", next_state)
        self.assertGreater(self.scalar(next_state["fatigue"]), 0.4)
        self.assertEqual(len(next_state["recent_category_history"]), 1)

    def test_no_dopamine_bucket_head_uses_non_dopamine_features(self):
        model = NoDopamineBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(user_feat={"user_vec": np.zeros(32, dtype=np.float32)}, device=device)
        state["fatigue"] = torch.tensor(0.25, dtype=torch.float32)
        state["recent_category_history"] = [torch.ones(31, dtype=torch.float32)]

        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.4,
            "creator_pref_norm": 0.2,
        }

        with torch.no_grad():
            model.bucket_w.zero_()
            model.bucket_w[0, 0] = 1.0
            model.bucket_w[1, 1] = 1.0
            model.bucket_w[2, 2] = 1.0
            model.bucket_w[3, 3] = 1.0
            model.bucket_w[4, 4] = 1.0

        bucket_logits, _, _ = model.forward_one(
            state=state,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        expected_pref_match = 1.0 / (1.0 + np.exp(-(32.0 / np.sqrt(32.0))))
        expected_novelty = np.sqrt(31.0) / np.sqrt(31.0)

        self.assertEqual(tuple(bucket_logits.shape), (6,))
        self.assertEqual(tuple(model.bucket_w.shape), (6, 5))
        self.assertAlmostEqual(self.scalar(bucket_logits[0]), expected_pref_match, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[1]), 0.4, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[2]), expected_novelty, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[3]), 0.2, places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[4]), self.scalar(state["fatigue"]), places=6)
        self.assertAlmostEqual(self.scalar(bucket_logits[5]), 0.0, places=6)

    def test_no_dopamine_export_env_params_removes_peak_reward_chain(self):
        model = NoDopamineBehaviorFitModel(embedding_dim=32)
        params = model.export_env_params()

        self.assertIn("fatigue_watch_ratio_coef", params)
        self.assertIn("repeat_watch_decay", params)
        self.assertIn("repeat_pass_cap", params)
        self.assertNotIn("expected_alpha", params)
        self.assertNotIn("peak_decay", params)
        self.assertNotIn("dopamine_base", params)
        self.assertNotIn("dopamine_normal_level", params)
        self.assertNotIn("dopamine_baseline_return_strength", params)
        self.assertNotIn("dopamine_habit_growth_rate", params)
        self.assertNotIn("dopamine_habit_max_gain", params)
        self.assertNotIn("dopamine_expected_reward_coef", params)
        self.assertNotIn("dopamine_peak_coef", params)
        self.assertNotIn("effective_sharpness", params)
        self.assertNotIn("dopamine_duration_penalty", params)
        self.assertNotIn("dopamine_high_watch_penalty", params)

    def test_content_only_fatigue_model_keeps_only_fatigue_state(self):
        model = ContentOnlyFatigueBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )

        self.assertIn("fatigue", state)
        self.assertIn("recent_category_history", state)
        self.assertNotIn("dopamine", state)
        self.assertNotIn("expected_reward", state)
        self.assertNotIn("session_peak", state)
        self.assertNotIn("swipe_count", state)

    def test_content_only_fatigue_forward_uses_fixed_learnable_threshold(self):
        model = ContentOnlyFatigueBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        user_feat = {"user_vec": np.ones(32, dtype=np.float32)}
        video_feat = {
            "item_vec": np.ones(32, dtype=np.float32),
            "category_vec_31": np.zeros(31, dtype=np.float32),
            "quality_norm": 0.8,
            "creator_pref_norm": 0.1,
        }

        low_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        high_fatigue = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        low_fatigue["fatigue"] = torch.tensor(0.0, dtype=torch.float32)
        high_fatigue["fatigue"] = torch.tensor(0.7, dtype=torch.float32)

        _, low_pred, low_aux = model.forward_one(
            state=low_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )
        _, high_pred, high_aux = model.forward_one(
            state=high_fatigue,
            user_feat=user_feat,
            video_feat=video_feat,
            device=device,
        )

        self.assertAlmostEqual(
            self.scalar(low_aux["effective_threshold"]),
            self.scalar(high_aux["effective_threshold"]),
            places=6,
        )
        self.assertAlmostEqual(
            self.scalar(low_aux["effective_threshold"]),
            self.scalar(model._params()["effective_threshold_base"]),
            places=6,
        )
        self.assertLess(self.scalar(high_pred), self.scalar(low_pred))

    def test_content_only_fatigue_model_state_updates(self):
        model = ContentOnlyFatigueBehaviorFitModel(embedding_dim=32)
        device = torch.device("cpu")
        state = model.init_state(
            user_feat={"user_vec": np.zeros(32, dtype=np.float32)},
            device=device,
        )
        state["fatigue"] = torch.tensor(0.4, dtype=torch.float32)
        aux = {
            "video_score": torch.tensor(1.5, dtype=torch.float32),
            "quality_norm": torch.tensor(0.9, dtype=torch.float32),
            "category_vec": torch.ones(31, dtype=torch.float32),
        }

        next_state = model.update_state_with_real_feedback(
            state=state,
            y_watch_ratio=torch.tensor(1.2, dtype=torch.float32),
            y_bucket=torch.tensor(5, dtype=torch.long),
            aux=aux,
        )

        self.assertIn("fatigue", next_state)
        self.assertIn("recent_category_history", next_state)
        self.assertNotIn("dopamine", next_state)
        self.assertNotIn("expected_reward", next_state)
        self.assertNotIn("session_peak", next_state)
        self.assertNotIn("swipe_count", next_state)
        self.assertGreater(self.scalar(next_state["fatigue"]), 0.4)
        self.assertEqual(len(next_state["recent_category_history"]), 1)

    def test_content_only_fatigue_export_env_params_omits_dopamine_and_threshold_fatigue(self):
        params = ContentOnlyFatigueBehaviorFitModel(embedding_dim=32).export_env_params()

        self.assertIn("effective_threshold_base", params)
        self.assertIn("watch_gain_base", params)
        self.assertIn("fatigue_duration_penalty", params)
        self.assertIn("repeat_watch_decay", params)
        self.assertIn("repeat_pass_cap", params)
        self.assertNotIn("effective_threshold_fatigue", params)
        self.assertNotIn("dopamine_base", params)
        self.assertNotIn("dopamine_expected_reward_coef", params)
        self.assertNotIn("dopamine_peak_coef", params)

    def test_content_only_fatigue_compare_helper_trains_content_only_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            csv_path = self.write_small_matrix_csv(root)

            result = compare_content_only_fatigue_model_on_kuairec(
                csv_path=csv_path,
                feature_provider=provider,
                train_start_index=0,
                train_row_count=4,
                val_row_count=1,
                min_session_length=20,
                num_epochs=1,
                lr=1e-3,
            )

            self.assertIn("dopamine", result)
            self.assertIn("content_only_fatigue", result)
            self.assertIn("learned_params", result["content_only_fatigue"])

    def test_fit_content_only_fatigue_user_model_returns_exportable_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            csv_path = self.write_long_session_matrix_csv(root, [20, 20])

            model, learned_params, train_df = fit_content_only_fatigue_user_model_from_kuairec(
                csv_path=csv_path,
                feature_provider=provider,
                train_start_index=0,
                train_row_count=25,
                val_row_count=5,
                min_session_length=20,
                num_epochs=1,
                lr=1e-3,
            )

            self.assertIsInstance(model, ContentOnlyFatigueBehaviorFitModel)
            self.assertGreater(len(train_df), 0)
            self.assertIn("effective_threshold_base", learned_params)
            self.assertNotIn("effective_threshold_fatigue", learned_params)

    def test_env_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))

            env = make_env(
                feature_provider=provider,
                num_candidates=3,
                slate_size=1,
                seed=0,
            )
            obs = env.reset()
            self.assertIn("user", obs)
            self.assertIn("doc", obs)

            user_model = SimpleUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()
            resp = user_model.simulate_response([doc])[0].create_observation()

            self.assertIn("watch_ratio", resp)
            self.assertIn("watch_ratio_bucket", resp)
            self.assertIn("high_engaged", resp)
            self.assertIn("watch_time", resp)
            self.assertNotIn("clicked", resp)

            user_model = SimpleUserModel(
                slate_size=2,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            docs = [doc_sampler.sample_document(), doc_sampler.sample_document()]
            responses = [
                SimpleResponse(watch_ratio=1.0, watch_ratio_bucket=4, high_engaged=False, watch_time=1.0),
                SimpleResponse(watch_ratio=0.5, watch_ratio_bucket=1, high_engaged=False, watch_time=1.0),
            ]
            signals = [user_model._compute_doc_signals(doc) for doc in docs]
            expected_signal = np.mean([
                responses[0].watch_ratio * signals[0]["video_score"],
                responses[1].watch_ratio * signals[1]["video_score"],
            ])
            expected_peak = max(
                responses[0].watch_ratio * signals[0]["video_score"] *
                (user_model.behavior_params["peak_decay"] ** 1),
                responses[1].watch_ratio * signals[1]["video_score"],
            )
            expected_time_weighted_average = _compute_scalar_time_weighted_average_reward(
                [
                    responses[0].watch_ratio * signals[0]["video_score"],
                    responses[1].watch_ratio * signals[1]["video_score"],
                ],
                user_model.behavior_params["average_reward_decay"],
            )
            expected_reward = (
                user_model.behavior_params["expected_reward_peak_coef"] * expected_peak +
                user_model.behavior_params["expected_reward_average_coef"] * expected_time_weighted_average
            )
            expected_fatigue = (
                user_model.behavior_params["fatigue_watch_ratio_coef"] *
                np.mean([responses[0].watch_ratio, responses[1].watch_ratio])
            )
            user_model.update_state(docs, responses)
            self.assertAlmostEqual(user_model._user_state.expected_reward, expected_reward, places=6)
            self.assertAlmostEqual(
                user_model._user_state.time_weighted_average_reward,
                expected_time_weighted_average,
                places=6,
            )
            self.assertAlmostEqual(user_model._user_state.session_peak, expected_peak, places=6)
            self.assertAlmostEqual(user_model._user_state.fatigue, expected_fatigue, places=6)
            expected_dopamine = _apply_scalar_dopamine_update_scaffold(
                prev_dopamine=user_model.behavior_params["dopamine_base"],
                session_baseline=user_model.behavior_params["dopamine_base"],
                baseline_return_strength=user_model.behavior_params["dopamine_baseline_return_strength"],
                swipe_count=len(responses),
                habit_growth_rate=user_model.behavior_params["dopamine_habit_growth_rate"],
                habit_max_gain=user_model.behavior_params["dopamine_habit_max_gain"],
                reward_drive=(
                    user_model.behavior_params["dopamine_expected_reward_coef"] * user_model._user_state.expected_reward
                    + user_model.behavior_params["dopamine_peak_coef"] * user_model._user_state.session_peak
                ),
            )[0]
            self.assertAlmostEqual(user_model._user_state.dopamine, expected_dopamine, places=6)
            self.assertEqual(user_model._user_state.step_count, len(responses))

    def test_simple_user_model_simulate_response_matches_new_forward_formula(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))
            user_model = SimpleUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            user_model._user_state.dopamine = 1.8
            user_model._user_state.expected_reward = 0.7
            user_model._user_state.user_automaticity = 0.3
            user_model._user_state.fatigue = 0.4
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()

            response = user_model.simulate_response([doc])[0]
            signals = user_model._compute_doc_signals(doc)
            p = user_model.behavior_params
            effective_threshold = (
                p["effective_threshold_base"] +
                p["effective_threshold_expected_reward"] * user_model._user_state.expected_reward +
                p["effective_threshold_automaticity"] * user_model._user_state.user_automaticity
            )
            watch_duration_scale = np.exp(
                -p["fatigue_duration_penalty"] * user_model._user_state.fatigue
            )
            expected_watch_ratio = compute_threshold_banded_watch_ratio_components(
                video_score=signals["video_score"],
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=p["watch_gain_base"],
                repeat_watch_decay=p["repeat_watch_decay"],
                repeat_pass_cap=p["repeat_pass_cap"],
            )["pred_watch_ratio"]

            self.assertAlmostEqual(response.watch_ratio, expected_watch_ratio, places=6)

    def test_no_dopamine_user_model_simulate_response_uses_fatigue_only_in_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))
            user_model = NoDopamineUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            user_model._user_state.fatigue = 0.5
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()

            response = user_model.simulate_response([doc])[0]
            signals = user_model._compute_doc_signals(doc)
            p = user_model.behavior_params
            effective_threshold = p["effective_threshold_base"]
            expected_watch_ratio = compute_threshold_banded_watch_ratio_components(
                video_score=signals["video_score"],
                effective_threshold=effective_threshold,
                watch_duration_scale=np.exp(
                    -p["fatigue_duration_penalty"] * user_model._user_state.fatigue
                ),
                watch_gain_base=p["watch_gain_base"],
                repeat_watch_decay=p["repeat_watch_decay"],
                repeat_pass_cap=p["repeat_pass_cap"],
            )["pred_watch_ratio"]

            self.assertAlmostEqual(response.watch_ratio, expected_watch_ratio, places=6)

    def test_no_dopamine_env_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))

            env = make_no_dopamine_env(
                feature_provider=provider,
                num_candidates=3,
                slate_size=1,
                seed=0,
            )
            obs = env.reset()
            self.assertIn("user", obs)
            self.assertIn("doc", obs)
            self.assertEqual(len(np.array(obs["user"], dtype=np.float32)), 1 + provider.embedding_dim)

            user_model = NoDopamineUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            self.assertFalse(hasattr(user_model._user_state, "dopamine"))
            self.assertFalse(hasattr(user_model._user_state, "expected_reward"))
            self.assertFalse(hasattr(user_model._user_state, "session_peak"))
            self.assertFalse(hasattr(user_model._user_state, "recent_score_history"))

    def test_content_only_fatigue_user_model_simulate_response_uses_fatigue_only_in_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))
            user_model = ContentOnlyFatigueUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            user_model._user_state.fatigue = 0.5
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()

            response = user_model.simulate_response([doc])[0]
            signals = user_model._compute_doc_signals(doc)
            p = user_model.behavior_params
            expected_watch_ratio = compute_threshold_banded_watch_ratio_components(
                video_score=signals["video_score"],
                effective_threshold=p["effective_threshold_base"],
                watch_duration_scale=np.exp(
                    -p["fatigue_duration_penalty"] * user_model._user_state.fatigue
                ),
                watch_gain_base=p["watch_gain_base"],
                repeat_watch_decay=p["repeat_watch_decay"],
                repeat_pass_cap=p["repeat_pass_cap"],
            )["pred_watch_ratio"]

            self.assertAlmostEqual(response.watch_ratio, expected_watch_ratio, places=6)
            self.assertFalse(hasattr(user_model._user_state, "dopamine"))
            self.assertFalse(hasattr(user_model._user_state, "expected_reward"))
            self.assertFalse(hasattr(user_model._user_state, "session_peak"))

    def test_content_only_fatigue_env_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = self.build_provider(Path(tmpdir))

            env = make_content_only_fatigue_env(
                feature_provider=provider,
                num_candidates=3,
                slate_size=1,
                seed=0,
            )
            obs = env.reset()
            self.assertIn("user", obs)
            self.assertIn("doc", obs)
            self.assertEqual(
                len(np.array(obs["user"], dtype=np.float32)),
                1 + provider.embedding_dim,
            )

            user_model = ContentOnlyFatigueUserModel(
                slate_size=1,
                feature_provider=provider,
                seed=0,
            )
            user_model._user_state = user_model._simple_user_sampler.sample_user()
            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()
            response = user_model.simulate_response([doc])[0]
            self.assertIn("watch_ratio_bucket", response.create_observation())

            user_model.update_state([doc], [response])
            self.assertFalse(hasattr(user_model._user_state, "dopamine"))
            self.assertFalse(hasattr(user_model._user_state, "expected_reward"))
            self.assertFalse(hasattr(user_model._user_state, "session_peak"))
            self.assertFalse(hasattr(user_model._user_state, "recent_score_history"))

            doc_sampler = CorpusDocumentSampler(feature_provider=provider, seed=0)
            doc = doc_sampler.sample_document()
            response = user_model.simulate_response([doc])[0]
            self.assertIn("watch_ratio_bucket", response.create_observation())

            user_model.update_state([doc], [response])
            self.assertFalse(hasattr(user_model._user_state, "dopamine"))
            self.assertFalse(hasattr(user_model._user_state, "expected_reward"))
            self.assertFalse(hasattr(user_model._user_state, "session_peak"))
            self.assertFalse(hasattr(user_model._user_state, "recent_score_history"))

    def test_prepare_behavior_fit_train_val_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = self.write_long_session_matrix_csv(root, [20, 21, 22, 23])

            split = prepare_behavior_fit_train_val_split(
                csv_path=csv_path,
                train_start_index=1,
                train_row_count=2,
                val_row_count=None,
                min_session_length=20,
            )

            self.assertEqual(len(split["train"]), 21)
            self.assertEqual(len(split["val"]), 45)
            self.assertEqual(split["meta"]["requested_train_start_index"], 1)
            self.assertEqual(split["meta"]["train_start_index"], 20)
            self.assertEqual(split["meta"]["train_end_index_exclusive"], 41)
            self.assertEqual(split["meta"]["val_start_index"], 41)
            self.assertEqual(split["meta"]["val_end_index_exclusive"], 86)
            self.assertEqual(split["meta"]["actual_train_session_count"], 1)
            self.assertEqual(split["meta"]["actual_val_session_count"], 2)
            self.assertEqual(split["meta"]["session_gap_minutes"], 30)
            self.assertEqual(
                split["train"][["user_id", "session_id"]].drop_duplicates().values.tolist(),
                [[10, 1]],
            )
            self.assertEqual(
                split["val"][["user_id", "session_id"]].drop_duplicates().values.tolist(),
                [[11, 0], [11, 1]],
            )

    def test_prepare_behavior_fit_train_val_split_filters_short_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = self.write_long_session_matrix_csv(root, [19, 20, 21])

            split = prepare_behavior_fit_train_val_split(
                csv_path=csv_path,
                train_start_index=0,
                train_row_count=100,
                val_row_count=None,
                min_session_length=20,
            )

            self.assertEqual(split["meta"]["min_session_length"], 20)
            self.assertEqual(split["meta"]["pre_filter_train_row_count"], 60)
            self.assertEqual(split["meta"]["pre_filter_val_row_count"], 0)
            self.assertEqual(split["meta"]["pre_filter_train_session_count"], 3)
            self.assertEqual(split["meta"]["pre_filter_val_session_count"], 0)
            self.assertEqual(split["meta"]["post_filter_train_row_count"], 41)
            self.assertEqual(split["meta"]["post_filter_val_row_count"], 0)
            self.assertEqual(split["meta"]["post_filter_train_session_count"], 2)
            self.assertEqual(split["meta"]["post_filter_val_session_count"], 0)
            self.assertEqual(split["meta"]["actual_train_row_count"], 41)
            self.assertEqual(split["meta"]["actual_val_row_count"], 0)
            self.assertEqual(split["meta"]["actual_train_session_count"], 2)
            self.assertEqual(split["meta"]["actual_val_session_count"], 0)
            self.assertEqual(len(split["train"]), 41)
            self.assertEqual(len(split["val"]), 0)

    def test_prepare_behavior_fit_train_val_split_rejects_min_session_length_below_20(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = self.write_long_session_matrix_csv(root, [20, 20])

            with self.assertRaisesRegex(ValueError, "min_session_length must be at least 20"):
                prepare_behavior_fit_train_val_split(
                    csv_path=csv_path,
                    train_start_index=0,
                    train_row_count=20,
                    val_row_count=20,
                    min_session_length=19,
                )

    def test_prepare_behavior_fit_train_val_split_uses_mode_specific_session_filtering(self):
        rows = []
        start_time = pd.Timestamp("2020-01-01 00:00:00")
        for row_idx in range(21):
            timestamp = (
                4000.0 if row_idx == 20 else float(row_idx)
            )
            event_time = start_time + pd.Timedelta(seconds=timestamp)
            rows.append(
                {
                    "user_id": 10,
                    "video_id": 100 + (row_idx % 3),
                    "play_duration": 3000.0,
                    "video_duration": 5000.0,
                    "time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": int(event_time.strftime("%Y%m%d")),
                    "timestamp": timestamp,
                    "watch_ratio": 0.6,
                    "session_id": 7,
                    "session_row_index": row_idx,
                }
            )
        df = pd.DataFrame(rows)

        rollout_split = _prepare_behavior_fit_train_val_split_from_dataframe(
            df=df,
            train_start_index=0,
            train_row_count=100,
            val_row_count=0,
            min_session_length=20,
            state_update_mode="rollout",
        )
        teacher_split = _prepare_behavior_fit_train_val_split_from_dataframe(
            df=df,
            train_start_index=0,
            train_row_count=100,
            val_row_count=0,
            min_session_length=20,
            state_update_mode="teacher_forcing",
        )

        self.assertEqual(rollout_split["meta"]["post_filter_train_row_count"], 21)
        self.assertEqual(rollout_split["meta"]["post_filter_train_session_count"], 1)
        self.assertEqual(teacher_split["meta"]["post_filter_train_row_count"], 20)
        self.assertEqual(teacher_split["meta"]["post_filter_train_session_count"], 1)

    def test_prepare_behavior_fit_explicit_train_val_split_from_dataframes_tracks_dual_sources(self):
        train_rows = []
        for row_idx in range(21):
            train_rows.append(
                {
                    "user_id": 10,
                    "video_id": 100 + (row_idx % 3),
                    "play_duration": 3000.0,
                    "video_duration": 5000.0,
                    "time": "2020-01-01 00:00:00",
                    "date": 20200101,
                    "timestamp": float(row_idx),
                    "watch_ratio": 0.6,
                    "session_id": 0,
                    "session_row_index": row_idx,
                }
            )
        val_rows = []
        for row_idx in range(19):
            val_rows.append(
                {
                    "user_id": 11,
                    "video_id": 200 + (row_idx % 3),
                    "play_duration": 3000.0,
                    "video_duration": 5000.0,
                    "time": "2020-01-02 00:00:00",
                    "date": 20200102,
                    "timestamp": float(row_idx),
                    "watch_ratio": 0.5,
                    "session_id": 0,
                    "session_row_index": row_idx,
                }
            )
        for row_idx in range(20):
            val_rows.append(
                {
                    "user_id": 12,
                    "video_id": 300 + (row_idx % 3),
                    "play_duration": 3000.0,
                    "video_duration": 5000.0,
                    "time": "2020-01-03 00:00:00",
                    "date": 20200103,
                    "timestamp": float(row_idx),
                    "watch_ratio": 0.7,
                    "session_id": 1,
                    "session_row_index": row_idx,
                }
            )

        split = prepare_behavior_fit_explicit_train_val_split_from_dataframes(
            train_df=pd.DataFrame(train_rows),
            val_df=pd.DataFrame(val_rows),
            min_session_length=20,
            state_update_mode="rollout",
            train_source_name="big_matrix.csv",
            val_source_name="small_matrix.csv",
        )

        self.assertEqual(split["meta"]["split_mode"], "explicit_train_val_sources")
        self.assertEqual(split["meta"]["train_source_name"], "big_matrix.csv")
        self.assertEqual(split["meta"]["val_source_name"], "small_matrix.csv")
        self.assertEqual(split["meta"]["pre_filter_train_row_count"], 21)
        self.assertEqual(split["meta"]["pre_filter_val_row_count"], 39)
        self.assertEqual(split["meta"]["post_filter_train_row_count"], 21)
        self.assertEqual(split["meta"]["post_filter_val_row_count"], 20)
        self.assertEqual(split["meta"]["post_filter_train_session_count"], 1)
        self.assertEqual(split["meta"]["post_filter_val_session_count"], 1)
        self.assertEqual(len(split["train"]), 21)
        self.assertEqual(len(split["val"]), 20)

    def test_collect_behavior_predictions_exports_rowwise_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            df = pd.DataFrame(
                [
                    {
                        "user_id": 11,
                        "video_id": 102,
                        "watch_ratio": 1.0,
                        "timestamp": self.kuairec_timestamp(pd.Timestamp("2020-01-02 00:00:04")),
                        "date": 20200102,
                        "play_duration": 6000.0,
                        "video_duration": 6000.0,
                        "session_id": 0,
                        "session_row_index": 0,
                    },
                    {
                        "user_id": 11,
                        "video_id": 100,
                        "watch_ratio": 0.3,
                        "timestamp": self.kuairec_timestamp(pd.Timestamp("2020-01-02 00:00:05")),
                        "date": 20200102,
                        "play_duration": 1500.0,
                        "video_duration": 5000.0,
                        "session_id": 0,
                        "session_row_index": 1,
                    },
                ]
            )

            model = BehaviorFitModel(embedding_dim=provider.embedding_dim)
            predictions = collect_behavior_predictions(
                model=model,
                df=df,
                feature_provider=provider,
                state_update_mode="rollout",
            )
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions["eval_row_index"].tolist(), [0, 1])
            self.assertEqual(
                predictions["actual_bucket_label"].tolist(),
                [bucket_index_to_label(v) for v in predictions["actual_bucket"].tolist()],
            )
            self.assertIn("session_id", predictions.columns)
            self.assertIn("session_row_index", predictions.columns)
            self.assertIn("pred_watch_ratio", predictions.columns)
            self.assertIn("current_dopamine", predictions.columns)
            self.assertEqual(predictions["session_id"].tolist(), [0, 0])
            self.assertEqual(predictions["session_row_index"].tolist(), [0, 1])

            ablated = model.make_strict_zero_dopamine_eval_copy()
            ablated_predictions = collect_behavior_predictions(
                model=ablated,
                df=df,
                feature_provider=provider,
                state_update_mode="rollout",
            )
            self.assertTrue(np.allclose(ablated_predictions["dopamine_before"].fillna(0.0), 0.0))
            self.assertTrue(np.allclose(ablated_predictions["current_dopamine"].fillna(0.0), 0.0))

    def test_collect_dopamine_state_traces_records_rollout_and_teacher_forcing_updates(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                del user_id
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                del video_id, user_id
                return {
                    "item_vec": np.zeros(1, dtype=np.float32),
                    "category_vec_31": np.zeros(31, dtype=np.float32),
                }

        class DummyDopamineModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            def init_state(self, user_feat, device):
                del user_feat
                return {
                    "dopamine": torch.tensor(1.0, dtype=torch.float32, device=device),
                    "expected_reward": torch.tensor(0.2, dtype=torch.float32, device=device),
                    "session_peak": torch.tensor(0.3, dtype=torch.float32, device=device),
                }

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.6, dtype=torch.float32, device=device),
                    {
                        "video_score": torch.tensor(2.0, dtype=torch.float32, device=device),
                        "novelty_norm": torch.tensor(0.7, dtype=torch.float32, device=device),
                    },
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del aux
                return {
                    "dopamine": state["dopamine"] + pred_watch_ratio,
                    "expected_reward": state["expected_reward"] + pred_watch_ratio,
                    "session_peak": state["session_peak"] + 0.5,
                }

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_bucket, aux
                return {
                    "dopamine": state["dopamine"] + y_watch_ratio,
                    "expected_reward": state["expected_reward"] + y_watch_ratio,
                    "session_peak": state["session_peak"] + 1.0,
                }

        df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 100,
                    "watch_ratio": 0.25,
                    "time": "2020-01-01 00:00:00",
                }
            ]
        )
        model = DummyDopamineModel()

        rollout_traces = collect_dopamine_state_traces(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            state_update_mode="rollout",
            variant_name="dummy_rollout",
        )
        teacher_traces = collect_dopamine_state_traces(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            state_update_mode="teacher_forcing",
            variant_name="dummy_teacher",
        )

        self.assertEqual(rollout_traces["variant_name"].tolist(), ["dummy_rollout"])
        self.assertEqual(
            rollout_traces["state_update_mode"].tolist(),
            ["rollout"],
        )
        self.assertAlmostEqual(
            float(rollout_traces["state_update_watch_ratio"].iloc[0]),
            0.6,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["novelty_norm"].iloc[0]),
            0.7,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["score_engagement"].iloc[0]),
            1.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["dopamine_before"].iloc[0]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["dopamine_after"].iloc[0]),
            1.6,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["expected_reward_after"].iloc[0]),
            0.8,
            places=6,
        )
        self.assertAlmostEqual(
            float(rollout_traces["session_peak_after"].iloc[0]),
            0.8,
            places=6,
        )

        self.assertEqual(teacher_traces["variant_name"].tolist(), ["dummy_teacher"])
        self.assertEqual(
            teacher_traces["state_update_mode"].tolist(),
            ["teacher_forcing"],
        )
        self.assertAlmostEqual(
            float(teacher_traces["state_update_watch_ratio"].iloc[0]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(teacher_traces["score_engagement"].iloc[0]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(teacher_traces["dopamine_after"].iloc[0]),
            1.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(teacher_traces["expected_reward_after"].iloc[0]),
            0.45,
            places=6,
        )
        self.assertAlmostEqual(
            float(teacher_traces["session_peak_after"].iloc[0]),
            1.3,
            places=6,
        )

    def test_rollout_mode_uses_prediction_state_updates(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class RolloutOnlyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rollout_calls = 0

            def init_state(self, user_feat, device):
                del user_feat
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del aux
                self.rollout_calls += 1
                return {"step": state["step"] + 1, "value": pred_watch_ratio}

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del state, y_watch_ratio, y_bucket, aux
                raise AssertionError("rollout mode should not use real labels to update state")

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.1},
                {"user_id": 10, "video_id": 101, "watch_ratio": 0.9},
            ]
        )
        model = RolloutOnlyModel()
        metrics = _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=None,
            state_update_mode="rollout",
        )

        self.assertEqual(model.rollout_calls, 2)
        self.assertEqual(metrics["steps"], 2)

    def test_run_behavior_model_pass_restarts_state_per_session(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class SessionResetModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.init_calls = 0
                self.forward_steps = []

            def init_state(self, user_feat, device):
                del user_feat
                self.init_calls += 1
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del user_feat, video_feat
                self.forward_steps.append(int(state["step"]))
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del aux
                return {"step": state["step"] + 1, "value": pred_watch_ratio}

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_bucket, aux
                return {"step": state["step"] + 1, "value": y_watch_ratio}

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.1, "session_id": 0, "session_row_index": 0},
                {"user_id": 10, "video_id": 101, "watch_ratio": 0.9, "session_id": 0, "session_row_index": 1},
                {"user_id": 10, "video_id": 102, "watch_ratio": 0.2, "session_id": 1, "session_row_index": 0},
            ]
        )
        model = SessionResetModel()
        metrics = _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=None,
            state_update_mode="rollout",
        )

        self.assertEqual(model.init_calls, 2)
        self.assertEqual(model.forward_steps, [0, 1, 0])
        self.assertEqual(metrics["sessions"], 2)
        self.assertEqual(metrics["steps"], 3)

    def test_teacher_forcing_mode_uses_real_feedback_updates(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class TeacherForcingOnlyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.teacher_calls = 0

            def init_state(self, user_feat, device):
                del user_feat
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del state, pred_watch_ratio, aux
                raise AssertionError("teacher_forcing mode should not use prediction updates")

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_bucket, aux
                self.teacher_calls += 1
                return {"step": state["step"] + 1, "value": y_watch_ratio}

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.1, "timestamp": 0.0},
                {"user_id": 10, "video_id": 101, "watch_ratio": 0.9, "timestamp": 60.0},
            ]
        )
        model = TeacherForcingOnlyModel()
        metrics = _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=None,
            state_update_mode="teacher_forcing",
        )

        self.assertEqual(model.teacher_calls, 2)
        self.assertEqual(metrics["steps"], 2)

    def test_teacher_forcing_rebuilds_sessions_from_time_gap(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class TeacherForcingSessionResetModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.init_calls = 0
                self.forward_steps = []

            def init_state(self, user_feat, device):
                del user_feat
                self.init_calls += 1
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del user_feat, video_feat
                self.forward_steps.append(int(state["step"]))
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del state, pred_watch_ratio, aux
                raise AssertionError("teacher_forcing mode should not use prediction updates")

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_bucket, aux
                return {"step": state["step"] + 1, "value": y_watch_ratio}

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.1, "timestamp": 0.0, "session_id": 7, "session_row_index": 0},
                {"user_id": 10, "video_id": 101, "watch_ratio": 0.9, "timestamp": 60.0, "session_id": 7, "session_row_index": 1},
                {"user_id": 10, "video_id": 102, "watch_ratio": 0.2, "timestamp": 1920.0, "session_id": 7, "session_row_index": 2},
            ]
        )
        model = TeacherForcingSessionResetModel()
        metrics = _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=None,
            state_update_mode="teacher_forcing",
        )

        self.assertEqual(model.init_calls, 2)
        self.assertEqual(model.forward_steps, [0, 1, 0])
        self.assertEqual(metrics["sessions"], 2)
        self.assertEqual(metrics["steps"], 3)

    def test_teacher_forcing_preserves_filtered_session_counts(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                del user_id
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                del video_id, user_id
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class TeacherForcingSessionCountModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.init_calls = 0

            def init_state(self, user_feat, device):
                del user_feat
                self.init_calls += 1
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del state, pred_watch_ratio, aux
                raise AssertionError("teacher_forcing mode should not use prediction updates")

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_bucket, aux
                return {"step": state["step"] + 1, "value": y_watch_ratio}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = self.write_long_session_matrix_csv(root, [20, 21])
            split = prepare_behavior_fit_train_val_split(
                csv_path=csv_path,
                train_start_index=2,
                train_row_count=4,
                val_row_count=None,
                min_session_length=20,
                state_update_mode="teacher_forcing",
            )

            model = TeacherForcingSessionCountModel()
            metrics = _run_behavior_model_pass(
                model=model,
                df=split["train"],
                feature_provider=DummyFeatureProvider(),
                device=torch.device("cpu"),
                optimizer=None,
                state_update_mode="teacher_forcing",
            )

            self.assertEqual(model.init_calls, 1)
            self.assertEqual(metrics["sessions"], split["meta"]["post_filter_train_session_count"])
            self.assertEqual(metrics["steps"], split["meta"]["post_filter_train_row_count"])

    def test_teacher_forcing_requires_time_signal_to_rebuild_sessions(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class TeacherForcingOnlyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            def init_state(self, user_feat, device):
                del user_feat
                return {"step": 0, "value": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    torch.tensor(0.5, dtype=torch.float32, device=device),
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del state, pred_watch_ratio, aux
                raise AssertionError("teacher_forcing mode should not use prediction updates")

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del state, y_watch_ratio, y_bucket, aux
                raise AssertionError("teacher_forcing mode should fail before state updates without time fields")

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.1},
                {"user_id": 10, "video_id": 101, "watch_ratio": 0.9},
            ]
        )

        with self.assertRaisesRegex(ValueError, "teacher_forcing requires timestamp, time, or event_timestamp"):
            _run_behavior_model_pass(
                model=TeacherForcingOnlyModel(),
                df=df,
                feature_provider=DummyFeatureProvider(),
                device=torch.device("cpu"),
                optimizer=None,
                state_update_mode="teacher_forcing",
            )

    def test_rollout_mode_propagates_later_loss_to_earlier_prediction(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                return {"user_vec": np.zeros(1, dtype=np.float32)}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                return {"item_vec": np.zeros(1, dtype=np.float32), "category_vec_31": np.zeros(31, dtype=np.float32)}

        class ClosedLoopGradientModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first_step_value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            def init_state(self, user_feat, device):
                del user_feat
                return {"step": 0, "carry": torch.tensor(0.0, dtype=torch.float32, device=device)}

            def forward_one(self, state, user_feat, video_feat, device):
                del user_feat, video_feat
                if state["step"] == 0:
                    pred_watch_ratio = self.first_step_value
                else:
                    pred_watch_ratio = state["carry"]
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    pred_watch_ratio,
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del aux
                return {"step": state["step"] + 1, "carry": pred_watch_ratio}

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del state, y_watch_ratio, y_bucket, aux
                raise AssertionError("this test must stay in rollout mode")

        df = pd.DataFrame(
            [
                {"user_id": 10, "video_id": 100, "watch_ratio": 0.0},
                {"user_id": 10, "video_id": 101, "watch_ratio": 1.0},
            ]
        )
        model = ClosedLoopGradientModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=optimizer,
            state_update_mode="rollout",
        )

        self.assertIsNotNone(model.first_step_value.grad)
        self.assertGreater(abs(self.scalar(model.first_step_value.grad)), 0.0)

    def test_run_behavior_model_pass_uses_session_length_compensation(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                del user_id
                return {}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                del video_id, user_id
                return {}

        class SessionLossReferenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pred_watch_ratio = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            def init_state(self, user_feat, device):
                del user_feat, device
                return {}

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    self.pred_watch_ratio,
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del pred_watch_ratio, aux
                return state

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_watch_ratio, y_bucket, aux
                return state

        long_df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 100 + idx,
                    "watch_ratio": 0.0,
                    "session_id": 0,
                    "session_row_index": idx,
                }
                for idx in range(25)
            ]
        )
        long_model = SessionLossReferenceModel()
        long_optimizer = torch.optim.SGD(long_model.parameters(), lr=0.0)

        _run_behavior_model_pass(
            model=long_model,
            df=long_df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=long_optimizer,
            state_update_mode="rollout",
        )

        self.assertAlmostEqual(self.scalar(long_model.pred_watch_ratio.grad), 25.0 / 20.0, places=6)

        short_df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 200 + idx,
                    "watch_ratio": 0.0,
                    "session_id": 0,
                    "session_row_index": idx,
                }
                for idx in range(5)
            ]
        )
        short_model = SessionLossReferenceModel()
        short_optimizer = torch.optim.SGD(short_model.parameters(), lr=0.0)

        _run_behavior_model_pass(
            model=short_model,
            df=short_df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=short_optimizer,
            state_update_mode="rollout",
        )

        self.assertAlmostEqual(self.scalar(short_model.pred_watch_ratio.grad), 1.0, places=6)

    def test_run_behavior_model_pass_steps_optimizer_once_per_session_even_if_chunk_is_set(self):
        class DummyFeatureProvider:
            embedding_dim = 1

            @staticmethod
            def get_user_features(user_id):
                del user_id
                return {}

            @staticmethod
            def get_video_features(video_id, user_id=None):
                del video_id, user_id
                return {}

        class ChunkedOptimizerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pred_watch_ratio = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            def init_state(self, user_feat, device):
                del user_feat, device
                return {}

            def forward_one(self, state, user_feat, video_feat, device):
                del state, user_feat, video_feat
                return (
                    torch.zeros(6, dtype=torch.float32, device=device),
                    self.pred_watch_ratio,
                    {},
                )

            def update_state_with_prediction(self, state, pred_watch_ratio, aux):
                del pred_watch_ratio, aux
                return state

            def update_state_with_real_feedback(self, state, y_watch_ratio, y_bucket, aux):
                del y_watch_ratio, y_bucket, aux
                return state

        df = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "video_id": 100 + idx,
                    "watch_ratio": 0.0,
                    "session_id": 0 if idx < 12 else 1,
                    "session_row_index": idx if idx < 12 else idx - 12,
                }
                for idx in range(25)
            ]
        )
        model = ChunkedOptimizerModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        optimizer.step = mock.MagicMock(wraps=optimizer.step)
        optimizer.zero_grad = mock.MagicMock(wraps=optimizer.zero_grad)

        _run_behavior_model_pass(
            model=model,
            df=df,
            feature_provider=DummyFeatureProvider(),
            device=torch.device("cpu"),
            optimizer=optimizer,
            state_update_mode="rollout",
            optimizer_chunk_rows=10,
        )

        self.assertEqual(optimizer.step.call_count, 2)
        self.assertEqual(optimizer.zero_grad.call_count, 2)

    def test_compare_behavior_models_runs_on_same_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            csv_path = self.write_long_session_matrix_csv(root, [20, 20])

            result = compare_behavior_models_on_kuairec(
                csv_path=csv_path,
                feature_provider=provider,
                train_start_index=0,
                train_row_count=3,
                val_row_count=2,
                min_session_length=20,
                num_epochs=1,
                lr=1e-3,
            )

            self.assertEqual(result["state_update_mode"], "rollout")
            self.assertEqual(result["split"]["train_row_count"], 20)
            self.assertEqual(result["split"]["val_row_count"], 20)
            self.assertEqual(len(result["dopamine"]["history"]), 1)
            self.assertEqual(len(result["no_dopamine"]["history"]), 1)
            self.assertIn("summary", result)
            self.assertTrue(np.isfinite(result["summary"]["dopamine_val_loss"]))
            self.assertTrue(np.isfinite(result["summary"]["no_dopamine_val_loss"]))

    def test_compare_behavior_models_teacher_forcing_mode_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            csv_path = self.write_long_session_matrix_csv(root, [20, 20])

            result = compare_behavior_models_on_kuairec(
                csv_path=csv_path,
                feature_provider=provider,
                train_start_index=0,
                train_row_count=3,
                val_row_count=2,
                min_session_length=20,
                num_epochs=1,
                lr=1e-3,
                state_update_mode="teacher_forcing",
            )

            self.assertEqual(result["state_update_mode"], "teacher_forcing")
            self.assertTrue(np.isfinite(result["summary"]["dopamine_val_loss"]))
            self.assertTrue(np.isfinite(result["summary"]["no_dopamine_val_loss"]))

    def test_evaluate_dopamine_ablation_on_kuairec_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provider = self.build_provider(root)
            csv_path = self.write_long_session_matrix_csv(root, [20, 20])

            result = evaluate_dopamine_ablation_on_kuairec(
                csv_path=csv_path,
                feature_provider=provider,
                train_start_index=0,
                train_row_count=3,
                val_row_count=2,
                min_session_length=20,
                num_epochs=1,
                lr=1e-3,
            )

            self.assertEqual(result["state_update_mode"], "rollout")
            self.assertEqual(result["split"]["train_row_count"], 20)
            self.assertEqual(result["split"]["val_row_count"], 20)
            self.assertEqual(result["ablated_param_keys"], sorted(DOPAMINE_PARAM_KEYS))
            self.assertIn("dopamine_base", result["baseline_learned_params"])
            self.assertIn("baseline_val", result)
            self.assertIn("ablated_val", result)
            self.assertEqual(result["baseline_val"]["steps"], result["ablated_val"]["steps"])
            self.assertEqual(result["baseline_val"]["rows"], result["ablated_val"]["rows"])
            self.assertEqual(result["baseline_val"]["users"], result["ablated_val"]["users"])
            self.assertAlmostEqual(
                result["delta"]["avg_loss"],
                result["ablated_val"]["avg_loss"] - result["baseline_val"]["avg_loss"],
                places=7,
            )
            self.assertTrue(np.isfinite(result["baseline_val"]["avg_loss"]))
            self.assertTrue(np.isfinite(result["ablated_val"]["avg_loss"]))

    def test_evaluate_shared_loss_weight_candidates_uses_rollout_mean_abs_error(self):
        split = {
            "train": pd.DataFrame({"user_id": [1], "video_id": [10], "watch_ratio": [0.4]}),
            "val": pd.DataFrame({"user_id": [1], "video_id": [10], "watch_ratio": [0.4]}),
            "meta": {},
        }
        candidates = [
            {"value": 1.0, "bucket_ce": 1.0, "bucket_dist": 1.0},
            {"value": 2.0, "bucket_ce": 1.0, "bucket_dist": 1.0},
            {"value": 4.0, "bucket_ce": 1.0, "bucket_dist": 1.0},
        ]
        mean_abs_error_map = {
            ((1.0, 1.0, 1.0), "rollout"): 0.30,
            ((2.0, 1.0, 1.0), "rollout"): 0.20,
            ((4.0, 1.0, 1.0), "rollout"): 0.15,
        }

        class DummyModel:
            def __init__(self, weights, mode):
                self.weights = weights
                self.mode = mode

        def fake_fit_model_fn(**kwargs):
            weights = (
                float(kwargs["watch_ratio_value_loss_weight"]),
                float(kwargs["watch_ratio_bucket_ce_loss_weight"]),
                float(kwargs["watch_ratio_bucket_distance_loss_weight"]),
            )
            model = DummyModel(weights=weights, mode=kwargs["state_update_mode"])
            history = [{"val": {"avg_loss": mean_abs_error_map[(weights, "rollout")]}}]
            return model, {}, history

        def fake_prediction_fn(model, df, feature_provider, state_update_mode):
            del df, feature_provider
            mae = mean_abs_error_map[(model.weights, "rollout")]
            return pd.DataFrame(
                {
                    "abs_error": [mae, mae],
                    "user_id": [1, 2],
                    "session_id": [0, 0],
                    "actual_bucket": [1, 1],
                    "pred_bucket": [1, 1],
                }
            )

        result = evaluate_shared_loss_weight_candidates(
            feature_provider=object(),
            split=split,
            loss_weight_candidates=candidates,
            num_epochs=1,
            lr=1e-3,
            fit_model_fn=fake_fit_model_fn,
            prediction_fn=fake_prediction_fn,
        )

        self.assertEqual(len(result["candidate_results"]), 3)
        self.assertEqual(result["best"]["loss_weights"]["value"], 4.0)
        self.assertAlmostEqual(result["candidate_results"][0]["rollout_mean_abs_error"], 0.30)
        self.assertAlmostEqual(result["candidate_results"][1]["rollout_mean_abs_error"], 0.20)
        self.assertAlmostEqual(result["candidate_results"][2]["rollout_mean_abs_error"], 0.15)

    def test_write_final_watch_ratio_report_writes_csv_and_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions = pd.DataFrame(
                [
                    {
                        "eval_row_index": 0,
                        "user_id": 10,
                        "session_id": 2,
                        "video_id": 100,
                        "timestamp": 123.0,
                        "actual_watch_ratio": 0.4,
                        "pred_watch_ratio": 0.5,
                        "abs_error": 0.1,
                        "actual_bucket": 2,
                        "pred_bucket": 2,
                    }
                ]
            )
            split = {
                "meta": {
                    "train_source_name": "big_matrix.csv",
                    "val_source_name": "small_matrix.csv",
                    "min_session_length": 20,
                    "pre_filter_train_row_count": 12,
                    "pre_filter_val_row_count": 6,
                    "pre_filter_train_session_count": 2,
                    "pre_filter_val_session_count": 2,
                    "post_filter_train_row_count": 10,
                    "post_filter_val_row_count": 4,
                    "post_filter_train_session_count": 1,
                    "post_filter_val_session_count": 1,
                }
            }
            loss_weights = {"value": 2.0, "bucket_ce": 1.0, "bucket_dist": 1.0}
            tuning_result = {
                "best": {"candidate_index": 1},
                "candidate_results": [
                    {
                        "candidate_index": 0,
                        "loss_weights": {"value": 1.0, "bucket_ce": 1.0, "bucket_dist": 1.0},
                        "shared_mean_abs_error": 0.35,
                        "mode_results": {
                            "rollout": {"summary": {"mean_abs_error": 0.30}},
                            "teacher_forcing": {"summary": {"mean_abs_error": 0.40}},
                        },
                    },
                    {
                        "candidate_index": 1,
                        "loss_weights": loss_weights,
                        "shared_mean_abs_error": 0.275,
                        "mode_results": {
                            "rollout": {"summary": {"mean_abs_error": 0.20}},
                            "teacher_forcing": {"summary": {"mean_abs_error": 0.35}},
                        },
                    },
                ],
            }
            precomputed_result = {
                "predictions": predictions,
                "summary": {
                    "users": 1,
                    "mean_abs_error": 0.1,
                    "median_abs_error": 0.1,
                    "p90_abs_error": 0.1,
                    "bucket_match_rate": 1.0,
                },
                "val_metrics": {"avg_loss": 0.2},
            }

            result = write_final_watch_ratio_report(
                feature_provider=None,
                split=split,
                output_dir=output_dir,
                state_update_mode="rollout",
                output_basename="roll-out-final",
                loss_weights=loss_weights,
                tuning_result=tuning_result,
                precomputed_result=precomputed_result,
                num_epochs=1,
                lr=1e-3,
            )

            self.assertTrue(result["csv_path"].exists())
            self.assertTrue(result["html_path"].exists())
            html = result["html_path"].read_text(encoding="utf-8")
            self.assertIn("Rollout Final Watch Ratio Report", html)
            self.assertIn("state_update_mode: rollout", html)
            self.assertIn("train source: big_matrix.csv", html)
            self.assertIn("validation source: small_matrix.csv", html)
            self.assertIn("selected weights: value=2.000, bucket_ce=1.000, bucket_dist=1.000", html)

    def _fake_report_split(self):
        return {
            "train": pd.DataFrame({"user_id": [1], "video_id": [10], "watch_ratio": [0.4]}),
            "val": pd.DataFrame({"user_id": [2], "video_id": [20], "watch_ratio": [0.5]}),
            "meta": {
                "train_source_name": "big_matrix.csv",
                "val_source_name": "small_matrix.csv",
                "min_session_length": 20,
                "train_row_limit": final_reports_module.TRAIN_ROW_LIMIT,
                "val_row_limit": final_reports_module.VAL_ROW_LIMIT,
                "post_filter_train_row_count": 1,
                "post_filter_val_row_count": 1,
                "post_filter_train_session_count": 1,
                "post_filter_val_session_count": 1,
            },
        }

    def _fake_report_weights(self):
        return {"value": 4.0, "bucket_ce": 1.0, "bucket_dist": 1.0}

    def _fake_tuning_result(self, weights=None, mae=0.12):
        weights = dict(weights or self._fake_report_weights())
        candidate = {
            "candidate_index": 0,
            "loss_weights": weights,
            "rollout_mean_abs_error": float(mae),
            "shared_mean_abs_error": float(mae),
            "mode_results": {
                "rollout": {
                    "summary": {
                        "mean_abs_error": float(mae),
                    }
                }
            },
        }
        return {
            "best": dict(candidate),
            "candidate_results": [dict(candidate)],
        }

    def _fake_rollout_result(self, summary=None, val_metrics=None, learned_params=None):
        fake_model = mock.Mock()
        fake_model.state_dict.return_value = {"weight": torch.tensor([1.0], dtype=torch.float32)}
        return {
            "model": fake_model,
            "learned_params": learned_params or {"dopamine_base": 1.0},
            "summary": summary
            or {"mean_abs_error": 0.12, "p90_abs_error": 0.2, "bucket_match_rate": 0.3},
            "val_metrics": val_metrics or {"avg_loss": 0.4},
            "predictions": pd.DataFrame(),
        }

    def _write_fake_rollout_report(self, output_dir: Path, summary, val_metrics):
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{final_reports_module.FINAL_REPORT_BASENAME}.csv"
        html_path = output_dir / f"{final_reports_module.FINAL_REPORT_BASENAME}.html"
        csv_path.write_text("eval_row_index,user_id\n0,1\n", encoding="utf-8")
        html_path.write_text("<html><body>cached-report</body></html>", encoding="utf-8")
        return {
            "csv_path": csv_path,
            "html_path": html_path,
            "summary": summary,
            "val_metrics": val_metrics,
        }

    def test_run_final_watch_ratio_reports_uses_big_train_small_val_and_rollout_only(self):
        fake_provider = object()
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        fake_output = {
            "csv_path": Path("rollout.csv"),
            "html_path": Path("rollout.html"),
            "summary": fake_rollout_result["summary"],
            "val_metrics": fake_rollout_result["val_metrics"],
        }

        with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, Path("data"))):
            with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split) as split_mock:
                with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result) as tuning_mock:
                    with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result) as fit_mock:
                        with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result) as fixed_tuning_mock:
                            with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", return_value=fake_output) as write_mock:
                                with mock.patch.object(
                                    final_reports_module,
                                    "_save_rollout_training_cache",
                                    return_value={"artifact_paths": {"cache_meta_path": "prediction_reports/rollout_training_cache_meta.json"}},
                                ) as save_cache_mock:
                                    result = final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

        split_mock.assert_called_once()
        self.assertEqual(split_mock.call_args.kwargs["train_csv_path"].name, "big_matrix.csv")
        self.assertEqual(split_mock.call_args.kwargs["val_csv_path"].name, "small_matrix.csv")
        self.assertEqual(split_mock.call_args.kwargs["state_update_mode"], "rollout")
        self.assertEqual(
            split_mock.call_args.kwargs["train_row_limit"],
            final_reports_module.TRAIN_ROW_LIMIT,
        )
        self.assertEqual(
            split_mock.call_args.kwargs["val_row_limit"],
            final_reports_module.VAL_ROW_LIMIT,
        )
        tuning_mock.assert_not_called()
        fit_mock.assert_called_once()
        fixed_tuning_mock.assert_called_once_with(
            final_reports_module.FIXED_LOSS_WEIGHTS,
            fake_rollout_result["summary"],
        )
        self.assertEqual(fit_mock.call_args.kwargs["state_update_mode"], "rollout")
        self.assertEqual(
            fit_mock.call_args.kwargs["loss_weights"],
            final_reports_module.FIXED_LOSS_WEIGHTS,
        )
        write_mock.assert_called_once()
        save_cache_mock.assert_called_once()
        self.assertEqual(write_mock.call_args.kwargs["state_update_mode"], "rollout")
        self.assertEqual(result["split"], fake_split)
        self.assertEqual(result["rollout_result"], fake_rollout_result)
        self.assertEqual(result["output"], fake_output)
        self.assertEqual(result["cache_status"], "miss")
        self.assertNotIn("mode_results", result)
        self.assertNotIn("splits_by_mode", result)

    def test_run_final_watch_ratio_reports_writes_training_cache_artifacts(self):
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        fake_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 1, "mtime_ns": 11}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "abc123"}
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / final_reports_module.OUTPUT_DIRNAME
            fake_provider = object()

            def fake_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    fake_rollout_result["summary"],
                    fake_rollout_result["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split):
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result):
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result):
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report):
                                            result = final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

            cache_paths = final_reports_module._cache_artifact_paths(output_dir)
            self.assertTrue(cache_paths["cache_meta_path"].exists())
            self.assertTrue(cache_paths["learned_params_path"].exists())
            self.assertTrue(cache_paths["checkpoint_path"].exists())
            cache_meta = json.loads(cache_paths["cache_meta_path"].read_text(encoding="utf-8"))
            learned_params = json.loads(cache_paths["learned_params_path"].read_text(encoding="utf-8"))
            checkpoint = torch.load(cache_paths["checkpoint_path"], map_location="cpu")

            self.assertEqual(cache_meta["selected_loss_weights"], self._fake_report_weights())
            self.assertEqual(cache_meta["summary"], fake_rollout_result["summary"])
            self.assertEqual(cache_meta["val_metrics"], fake_rollout_result["val_metrics"])
            self.assertEqual(cache_meta["fingerprint"], fake_fingerprint)
            self.assertEqual(
                cache_meta["training_config"]["min_session_length"],
                final_reports_module.MIN_SESSION_LENGTH,
            )
            self.assertEqual(
                cache_meta["training_config"]["train_row_limit"],
                final_reports_module.TRAIN_ROW_LIMIT,
            )
            self.assertEqual(
                cache_meta["training_config"]["val_row_limit"],
                final_reports_module.VAL_ROW_LIMIT,
            )
            self.assertEqual(
                cache_meta["training_config"]["loss_weight_strategy"],
                "fixed",
            )
            self.assertEqual(learned_params, fake_rollout_result["learned_params"])
            self.assertIn("weight", checkpoint)
            self.assertEqual(result["cache_status"], "miss")
            self.assertEqual(result["cache_meta_path"], cache_paths["cache_meta_path"])

    def test_run_final_watch_ratio_reports_reuses_matching_cache_without_retraining(self):
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        fake_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 1, "mtime_ns": 11}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "abc123"}
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_provider = object()

            def fake_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    fake_rollout_result["summary"],
                    fake_rollout_result["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split):
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result):
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result):
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report):
                                            final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split") as split_mock:
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates") as tuning_mock:
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result") as fit_mock:
                                    with mock.patch.object(final_reports_module, "write_final_watch_ratio_report") as write_mock:
                                        result = final_reports_module.run_final_watch_ratio_reports()

            split_mock.assert_not_called()
            tuning_mock.assert_not_called()
            fit_mock.assert_not_called()
            write_mock.assert_not_called()
            self.assertEqual(result["cache_status"], "hit")
            self.assertEqual(result["rollout_result"]["learned_params"], fake_rollout_result["learned_params"])
            self.assertEqual(result["selected_loss_weights"], self._fake_report_weights())
            self.assertEqual(result["split"]["meta"]["train_source_name"], "big_matrix.csv")

    def test_run_final_watch_ratio_reports_invalidates_cache_on_fingerprint_change(self):
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        cache_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 1, "mtime_ns": 11}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "abc123"}
            },
        }
        changed_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 2, "mtime_ns": 22}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "changed"}
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_provider = object()

            def fake_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    fake_rollout_result["summary"],
                    fake_rollout_result["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=cache_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split):
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result):
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result):
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report):
                                            final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=changed_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split) as split_mock:
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result) as tuning_mock:
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result) as fit_mock:
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report) as write_mock:
                                            result = final_reports_module.run_final_watch_ratio_reports()

            split_mock.assert_called_once()
            tuning_mock.assert_not_called()
            fit_mock.assert_called_once()
            write_mock.assert_called_once()
            self.assertEqual(result["cache_status"], "miss")

    def test_run_final_watch_ratio_reports_rebuilds_reports_from_checkpoint_on_cache_hit(self):
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        fake_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 1, "mtime_ns": 11}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "abc123"}
            },
        }
        predictions = pd.DataFrame(
            [
                {
                    "user_id": 2,
                    "session_id": 1,
                    "actual_bucket": 1,
                    "pred_bucket": 1,
                    "abs_error": 0.05,
                },
                {
                    "user_id": 2,
                    "session_id": 1,
                    "actual_bucket": 2,
                    "pred_bucket": 1,
                    "abs_error": 0.15,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / final_reports_module.OUTPUT_DIRNAME
            fake_provider = object()

            def fake_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    kwargs["precomputed_result"]["summary"],
                    kwargs["precomputed_result"]["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split):
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result):
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result):
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report):
                                            final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

            cache_paths = final_reports_module._cache_artifact_paths(output_dir)
            cache_paths["report_csv_path"].unlink()
            cache_paths["report_html_path"].unlink()

            rebuilt_summary = final_reports_module._summarize_predictions(predictions)
            rebuild_output = {
                "csv_path": cache_paths["report_csv_path"],
                "html_path": cache_paths["report_html_path"],
                "summary": rebuilt_summary,
                "val_metrics": fake_rollout_result["val_metrics"],
            }

            def rebuild_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    rebuild_output["summary"],
                    rebuild_output["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split) as split_mock:
                            with mock.patch.object(final_reports_module, "load_behavior_model_checkpoint", return_value=mock.Mock()) as load_model_mock:
                                with mock.patch.object(final_reports_module, "collect_behavior_predictions", return_value=predictions) as prediction_mock:
                                    with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates") as tuning_mock:
                                        with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result") as fit_mock:
                                            with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=rebuild_write_report) as write_mock:
                                                result = final_reports_module.run_final_watch_ratio_reports()

            split_mock.assert_called_once()
            load_model_mock.assert_called_once()
            prediction_mock.assert_called_once()
            tuning_mock.assert_not_called()
            fit_mock.assert_not_called()
            write_mock.assert_called_once()
            self.assertEqual(result["cache_status"], "rebuild_reports_only")
            self.assertTrue(cache_paths["report_csv_path"].exists())
            self.assertTrue(cache_paths["report_html_path"].exists())
            self.assertEqual(result["rollout_result"]["summary"], rebuilt_summary)

    def test_run_final_watch_ratio_reports_force_retrain_ignores_matching_cache(self):
        fake_split = self._fake_report_split()
        fake_tuning_result = self._fake_tuning_result()
        fake_rollout_result = self._fake_rollout_result()
        fake_fingerprint = {
            "policy": "strict_fingerprint",
            "data_and_features": {
                "train_data": {"path": "big_matrix.csv", "exists": True, "size_bytes": 1, "mtime_ns": 11}
            },
            "code": {
                "train_module": {"path": "train_module.py", "exists": True, "sha256": "abc123"}
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_provider = object()

            def fake_write_report(**kwargs):
                return self._write_fake_rollout_report(
                    kwargs["output_dir"],
                    fake_rollout_result["summary"],
                    fake_rollout_result["val_metrics"],
                )

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split):
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result):
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result):
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report):
                                            final_reports_module.run_final_watch_ratio_reports(reuse_saved_results=False)

            with mock.patch.object(final_reports_module, "_resolve_project_root", return_value=root):
                with mock.patch.object(final_reports_module, "build_feature_provider", return_value=(fake_provider, root / "data")):
                    with mock.patch.object(final_reports_module, "_build_runtime_fingerprint", return_value=fake_fingerprint):
                        with mock.patch.object(final_reports_module, "prepare_behavior_fit_explicit_train_val_split", return_value=fake_split) as split_mock:
                            with mock.patch.object(final_reports_module, "evaluate_shared_loss_weight_candidates", return_value=fake_tuning_result) as tuning_mock:
                                with mock.patch.object(final_reports_module, "_fit_and_collect_mode_result", return_value=fake_rollout_result) as fit_mock:
                                    with mock.patch.object(final_reports_module, "_build_fixed_tuning_result", return_value=fake_tuning_result):
                                        with mock.patch.object(final_reports_module, "write_final_watch_ratio_report", side_effect=fake_write_report) as write_mock:
                                            result = final_reports_module.run_final_watch_ratio_reports(force_retrain=True)

            split_mock.assert_called_once()
            tuning_mock.assert_not_called()
            fit_mock.assert_called_once()
            write_mock.assert_called_once()
            self.assertEqual(result["cache_status"], "force_retrain")

    def test_run_rollout_watch_ratio_simple_report_writes_simple_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"
            split = {
                "train": pd.DataFrame([{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]),
                "val": pd.DataFrame([{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]),
                "meta": {
                    "requested_train_row_count": 100000,
                    "requested_val_row_count": 20000,
                    "actual_train_row_count": 24,
                    "actual_val_row_count": 21,
                },
            }
            predictions = pd.DataFrame(
                [
                    {
                        "eval_row_index": 0,
                        "user_id": 10,
                        "video_id": 101,
                        "timestamp": 123.0,
                        "actual_watch_ratio": 0.4,
                        "actual_bucket": 1,
                        "actual_bucket_label": "[0.35, 0.6)",
                        "pred_watch_ratio": 0.5,
                        "pred_bucket": 1,
                        "pred_bucket_label": "[0.35, 0.6)",
                        "abs_error": 0.1,
                    }
                ]
            )
            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            with mock.patch.object(
                rollout_simple_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                rollout_simple_module,
                "prepare_behavior_fit_train_val_split",
                return_value=split,
            ) as prepare_split, mock.patch.object(
                rollout_simple_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"dopamine_base": 1.0}, history),
            ) as fit_model, mock.patch.object(
                rollout_simple_module,
                "collect_behavior_predictions",
                return_value=predictions,
            ) as collect_predictions:
                result = rollout_simple_module.run_rollout_watch_ratio_simple_report(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            prepare_kwargs = prepare_split.call_args.kwargs
            self.assertEqual(prepare_kwargs["train_start_index"], 0)
            self.assertEqual(prepare_kwargs["train_row_count"], 100000)
            self.assertEqual(prepare_kwargs["val_row_count"], 20000)
            self.assertEqual(prepare_kwargs["min_session_length"], 20)
            self.assertEqual(prepare_kwargs["state_update_mode"], "rollout")

            fit_kwargs = fit_model.call_args.kwargs
            self.assertIs(fit_kwargs["model_cls"], BehaviorFitModel)
            self.assertEqual(fit_kwargs["watch_ratio_value_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["watch_ratio_bucket_ce_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["watch_ratio_bucket_distance_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["state_update_mode"], "rollout")
            self.assertEqual(fit_kwargs["model_name"], "dopamine")

            collect_kwargs = collect_predictions.call_args.kwargs
            self.assertEqual(collect_kwargs["state_update_mode"], "rollout")

            self.assertTrue(result["csv_path"].exists())
            self.assertTrue(result["html_path"].exists())
            self.assertIn("split", result)
            self.assertIn("learned_params", result)
            self.assertIn("history", result)
            self.assertIn("predictions", result)
            self.assertIn("summary", result)

            html = result["html_path"].read_text(encoding="utf-8")
            self.assertIn("requested train rows: 0-99999", html)
            self.assertIn("requested validation rows: 100000-119999", html)
            self.assertIn("state_update_mode: rollout", html)
            self.assertIn("loss weights: value=1.0, bucket_ce=1.0, bucket_dist=1.0", html)
            self.assertIn("Actual Bucket", html)
            self.assertIn("Pred Bucket", html)
            self.assertIn("Bucket Match", html)
            self.assertNotIn("selected weights", html)
            self.assertNotIn("Shared Loss-Weight Tuning", html)

    def test_run_post_history_rollout_report_writes_variant_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"
            split = {
                "train": pd.DataFrame([{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]),
                "val": pd.DataFrame([{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]),
                "meta": {
                    "requested_train_row_count": 90000,
                    "requested_val_row_count": 10000,
                    "actual_train_row_count": 24,
                    "actual_val_row_count": 21,
                },
            }
            predictions = pd.DataFrame(
                [
                    {
                        "eval_row_index": 0,
                        "user_id": 10,
                        "video_id": 101,
                        "timestamp": 123.0,
                        "actual_watch_ratio": 0.4,
                        "actual_bucket": 1,
                        "actual_bucket_label": "[0.35, 0.6)",
                        "pred_watch_ratio": 0.5,
                        "pred_bucket": 1,
                        "pred_bucket_label": "[0.35, 0.6)",
                        "abs_error": 0.1,
                    }
                ]
            )
            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            with mock.patch.object(
                post_history_rollout_simple_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                post_history_rollout_simple_module,
                "prepare_behavior_fit_train_val_split",
                return_value=split,
            ) as prepare_split, mock.patch.object(
                post_history_rollout_simple_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"dopamine_peak_coef": 1.0}, history),
            ) as fit_model, mock.patch.object(
                post_history_rollout_simple_module,
                "collect_behavior_predictions",
                return_value=predictions,
            ) as collect_predictions:
                result = post_history_rollout_simple_module.run_rollout_watch_ratio_simple_report_post_history_dopamine(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            prepare_kwargs = prepare_split.call_args.kwargs
            self.assertEqual(prepare_kwargs["train_start_index"], 0)
            self.assertEqual(prepare_kwargs["train_row_count"], 90000)
            self.assertEqual(prepare_kwargs["val_row_count"], 10000)
            self.assertEqual(prepare_kwargs["min_session_length"], 20)
            self.assertEqual(prepare_kwargs["state_update_mode"], "rollout")

            fit_kwargs = fit_model.call_args.kwargs
            self.assertIs(
                fit_kwargs["model_cls"],
                PostHistoryPeakRewardDopamineBehaviorFitModel,
            )
            self.assertEqual(fit_kwargs["watch_ratio_value_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["watch_ratio_bucket_ce_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["watch_ratio_bucket_distance_loss_weight"], 1.0)
            self.assertEqual(fit_kwargs["state_update_mode"], "rollout")
            self.assertEqual(fit_kwargs["optimizer_chunk_rows"], 5)

            collect_kwargs = collect_predictions.call_args.kwargs
            self.assertEqual(collect_kwargs["state_update_mode"], "rollout")

            self.assertTrue(result["csv_path"].exists())
            self.assertTrue(result["html_path"].exists())
            self.assertIn("post_history_peak_reward_dopamine", result["html_path"].name)
            self.assertIn("chunk5", result["html_path"].name)
            self.assertIn("summary", result)

            html = result["html_path"].read_text(encoding="utf-8")
            self.assertIn("requested train rows: 0-89999", html)
            self.assertIn("requested validation rows: 90000-99999", html)
            self.assertIn("model variant: post-history peak/reward dopamine", html)
            self.assertIn("optimizer update: 5 interactions per Adam step", html)
            self.assertIn("score_engagement is appended into history first", html)
            self.assertNotIn("selected weights", html)
            self.assertNotIn("Shared Loss-Weight Tuning", html)

    def test_select_longest_sessions_for_relaxation_plot(self):
        trace_df = pd.DataFrame(
            [
                {"user_id": 1, "session_id": 10, "session_row_index": i}
                for i in range(3)
            ]
            + [
                {"user_id": 2, "session_id": 20, "session_row_index": i}
                for i in range(8)
            ]
            + [
                {"user_id": 3, "session_id": 30, "session_row_index": i}
                for i in range(5)
            ]
            + [
                {"user_id": 4, "session_id": 40, "session_row_index": i}
                for i in range(7)
            ]
            + [
                {"user_id": 5, "session_id": 50, "session_row_index": i}
                for i in range(2)
            ]
            + [
                {"user_id": 6, "session_id": 60, "session_row_index": i}
                for i in range(6)
            ]
        )

        selected = relaxation_experiment_module.select_longest_sessions_for_plot(
            trace_df,
            top_k=5,
        )

        self.assertEqual(
            selected,
            [(2, 20), (4, 40), (6, 60), (3, 30), (1, 10)],
        )

    def test_run_relaxation_experiment_writes_csv_and_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"
            split = {
                "train": pd.DataFrame([{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]),
                "val": pd.DataFrame([{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]),
                "meta": {
                    "requested_train_row_count": 90000,
                    "requested_val_row_count": 10000,
                    "actual_train_row_count": 24,
                    "actual_val_row_count": 21,
                },
            }
            predictions = pd.DataFrame(
                [
                    {
                        "eval_row_index": 0,
                        "user_id": 10,
                        "session_id": 3,
                        "session_row_index": 0,
                        "video_id": 101,
                        "timestamp": 123.0,
                        "actual_watch_ratio": 0.4,
                        "actual_bucket": 1,
                        "actual_bucket_label": "[0.35, 0.6)",
                        "pred_watch_ratio": 0.5,
                        "pred_bucket": 1,
                        "pred_bucket_label": "[0.35, 0.6)",
                        "abs_error": 0.1,
                        "dopamine_before": 1.2,
                        "current_dopamine": 1.2,
                    }
                ]
            )
            traces = pd.DataFrame(
                [
                    {
                        "eval_row_index": 0,
                        "user_id": 10,
                        "session_id": 3,
                        "session_row_index": 0,
                        "video_id": 101,
                        "score_engagement": 0.9,
                        "dopamine_before": 1.2,
                        "dopamine_after": 1.8,
                        "dopamine_true_baseline": 1.5,
                        "state_update_watch_ratio": 0.5,
                    }
                ]
            )
            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            with mock.patch.object(
                relaxation_experiment_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                relaxation_experiment_module,
                "prepare_behavior_fit_train_val_split",
                return_value=split,
            ) as prepare_split, mock.patch.object(
                relaxation_experiment_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"dopamine_true_baseline": 1.5}, history),
            ) as fit_model, mock.patch.object(
                relaxation_experiment_module,
                "collect_behavior_predictions",
                return_value=predictions,
            ) as collect_predictions, mock.patch.object(
                relaxation_experiment_module,
                "collect_session_dopamine_traces",
                return_value=traces,
            ) as collect_traces:
                result = relaxation_experiment_module.run_rollout_dopamine_relaxation_experiment(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            prepare_kwargs = prepare_split.call_args.kwargs
            self.assertEqual(prepare_kwargs["train_start_index"], 0)
            self.assertEqual(prepare_kwargs["train_row_count"], 90000)
            self.assertEqual(prepare_kwargs["val_row_count"], 10000)
            self.assertEqual(prepare_kwargs["min_session_length"], 20)
            self.assertEqual(prepare_kwargs["state_update_mode"], "rollout")

            fit_kwargs = fit_model.call_args.kwargs
            self.assertIs(
                fit_kwargs["model_cls"],
                RelaxToHigherBaselineDopamineBehaviorFitModel,
            )
            self.assertEqual(fit_kwargs["optimizer_chunk_rows"], 5)
            self.assertEqual(fit_kwargs["state_update_mode"], "rollout")

            collect_predictions.assert_called_once()
            collect_traces.assert_called_once()

            self.assertTrue(result["csv_path"].exists())
            self.assertTrue(result["png_path"].exists())
            self.assertIn("relax_to_higher_baseline_dopamine", result["csv_path"].name)
            self.assertIn("chunk5", result["png_path"].name)
            self.assertNotIn("html_path", result)
            self.assertIn("dopamine_after", result["predictions"].columns)
            self.assertIn("dopamine_true_baseline", result["predictions"].columns)

    def test_run_dopamine_variant_grid_executes_six_runs_and_writes_combined_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"

            def build_split_for_mode(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                return {
                    "train": pd.DataFrame(
                        [{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]
                    ),
                    "val": pd.DataFrame(
                        [{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]
                    ),
                    "meta": {
                        "state_update_mode": mode,
                        "post_filter_train_row_count": 24,
                        "post_filter_val_row_count": 21,
                        "post_filter_train_session_count": 3,
                        "post_filter_val_session_count": 2,
                    },
                }

            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            def prediction_side_effect(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                base_error = 0.1 if mode == "rollout" else 0.2
                return pd.DataFrame(
                    [
                        {
                            "eval_row_index": 0,
                            "user_id": 10,
                            "session_id": 5,
                            "video_id": 101,
                            "actual_bucket": 2,
                            "pred_bucket": 2,
                            "abs_error": base_error,
                        },
                        {
                            "eval_row_index": 1,
                            "user_id": 10,
                            "session_id": 7,
                            "video_id": 102,
                            "actual_bucket": 1,
                            "pred_bucket": 0,
                            "abs_error": base_error + 0.1,
                        },
                        {
                            "eval_row_index": 2,
                            "user_id": 10,
                            "session_id": 7,
                            "video_id": 103,
                            "actual_bucket": 3,
                            "pred_bucket": 3,
                            "abs_error": base_error + 0.2,
                        },
                    ]
                )

            def trace_side_effect(*args, **kwargs):
                variant_name = kwargs["variant_name"]
                mode = kwargs["state_update_mode"]
                mode_shift = 0.0 if mode == "rollout" else 0.1
                return pd.DataFrame(
                    [
                        {
                            "variant_name": variant_name,
                            "state_update_mode": mode,
                            "eval_row_index": 0,
                            "user_id": 10,
                            "session_id": 5,
                            "session_row_index": 0,
                            "video_id": 101,
                            "timestamp": 100.0,
                            "event_timestamp": 100.0,
                            "actual_watch_ratio": 0.5,
                            "actual_bucket": 2,
                            "state_update_watch_ratio": 0.5,
                            "novelty_norm": 0.2,
                            "score_engagement": 0.4,
                            "dopamine_before": 1.0 + mode_shift,
                            "dopamine_after": 1.1 + mode_shift,
                            "expected_reward_before": 0.1,
                            "expected_reward_after": 0.2,
                            "session_peak_before": 0.1,
                            "session_peak_after": 0.3,
                        },
                        {
                            "variant_name": variant_name,
                            "state_update_mode": mode,
                            "eval_row_index": 1,
                            "user_id": 10,
                            "session_id": 7,
                            "session_row_index": 0,
                            "video_id": 102,
                            "timestamp": 101.0,
                            "event_timestamp": 101.0,
                            "actual_watch_ratio": 0.4,
                            "actual_bucket": 1,
                            "state_update_watch_ratio": 0.4,
                            "novelty_norm": 0.3,
                            "score_engagement": 0.5,
                            "dopamine_before": 1.2 + mode_shift,
                            "dopamine_after": 1.3 + mode_shift,
                            "expected_reward_before": 0.2,
                            "expected_reward_after": 0.4,
                            "session_peak_before": 0.2,
                            "session_peak_after": 0.5,
                        },
                        {
                            "variant_name": variant_name,
                            "state_update_mode": mode,
                            "eval_row_index": 2,
                            "user_id": 10,
                            "session_id": 7,
                            "session_row_index": 1,
                            "video_id": 103,
                            "timestamp": 102.0,
                            "event_timestamp": 102.0,
                            "actual_watch_ratio": 0.8,
                            "actual_bucket": 3,
                            "state_update_watch_ratio": 0.8,
                            "novelty_norm": 0.5,
                            "score_engagement": 0.8,
                            "dopamine_before": 1.3 + mode_shift,
                            "dopamine_after": 1.6 + mode_shift,
                            "expected_reward_before": 0.4,
                            "expected_reward_after": 0.7,
                            "session_peak_before": 0.5,
                            "session_peak_after": 0.9,
                        },
                    ]
                )

            with mock.patch.object(
                dopamine_variant_grid_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                dopamine_variant_grid_module,
                "prepare_behavior_fit_train_val_split",
                side_effect=build_split_for_mode,
            ) as prepare_split, mock.patch.object(
                dopamine_variant_grid_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"dopamine_base": 1.0}, history),
            ) as fit_model, mock.patch.object(
                dopamine_variant_grid_module,
                "collect_behavior_predictions",
                side_effect=prediction_side_effect,
            ) as collect_predictions, mock.patch.object(
                dopamine_variant_grid_module,
                "collect_dopamine_state_traces",
                side_effect=trace_side_effect,
            ) as collect_traces:
                result = dopamine_variant_grid_module.run_dopamine_variant_grid_experiment(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            self.assertEqual(prepare_split.call_count, 2)
            prepare_modes = {
                call.kwargs["state_update_mode"] for call in prepare_split.call_args_list
            }
            self.assertEqual(prepare_modes, {"rollout", "teacher_forcing"})
            for call in prepare_split.call_args_list:
                self.assertEqual(call.kwargs["train_start_index"], 0)
                self.assertEqual(call.kwargs["train_row_count"], 90000)
                self.assertEqual(call.kwargs["val_row_count"], 10000)
                self.assertEqual(call.kwargs["min_session_length"], 20)

            self.assertEqual(fit_model.call_count, 6)
            expected_pairs = {
                ("full_chain_dopamine", "rollout", BehaviorFitModel),
                ("full_chain_dopamine", "teacher_forcing", BehaviorFitModel),
                (
                    "integrated_signal_dopamine",
                    "rollout",
                    IntegratedSignalDopamineBehaviorFitModel,
                ),
                (
                    "integrated_signal_dopamine",
                    "teacher_forcing",
                    IntegratedSignalDopamineBehaviorFitModel,
                ),
                (
                    "relax_to_higher_baseline_dopamine",
                    "rollout",
                    RelaxToHigherBaselineDopamineBehaviorFitModel,
                ),
                (
                    "relax_to_higher_baseline_dopamine",
                    "teacher_forcing",
                    RelaxToHigherBaselineDopamineBehaviorFitModel,
                ),
            }
            actual_pairs = {
                (
                    call.kwargs["model_name"],
                    call.kwargs["state_update_mode"],
                    call.kwargs["model_cls"],
                )
                for call in fit_model.call_args_list
            }
            self.assertEqual(actual_pairs, expected_pairs)
            for call in fit_model.call_args_list:
                self.assertEqual(call.kwargs["optimizer_chunk_rows"], 5)
                self.assertEqual(call.kwargs["watch_ratio_value_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_ce_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_distance_loss_weight"], 1.0)

            self.assertEqual(collect_predictions.call_count, 6)
            self.assertEqual(collect_traces.call_count, 6)

            self.assertTrue(result["summary_path"].exists())
            self.assertTrue(result["trace_path"].exists())
            self.assertIn("summary_train0_89999_val90000_99999_chunk5", result["summary_path"].name)
            self.assertIn("session_traces_train0_89999_val90000_99999_chunk5", result["trace_path"].name)

            summary_df = pd.read_csv(result["summary_path"])
            trace_df = pd.read_csv(result["trace_path"])
            self.assertEqual(len(summary_df), 6)
            self.assertEqual(set(summary_df["variant_name"]), {
                "full_chain_dopamine",
                "integrated_signal_dopamine",
                "relax_to_higher_baseline_dopamine",
            })
            self.assertEqual(set(summary_df["state_update_mode"]), {"rollout", "teacher_forcing"})
            self.assertTrue((summary_df["representative_session_id"] == 7).all())
            self.assertTrue((summary_df["representative_session_length"] == 2).all())
            self.assertTrue((summary_df["trend_label"] == "up").all())
            self.assertEqual(len(trace_df), 12)
            self.assertEqual(set(trace_df["session_id"]), {7})

    def test_select_shared_representative_sessions_uses_source_row_fingerprint(self):
        rollout_trace = pd.DataFrame(
            [
                {"user_id": 10, "session_id": 1, "source_row_index": 100},
                {"user_id": 10, "session_id": 1, "source_row_index": 101},
                {"user_id": 10, "session_id": 2, "source_row_index": 200},
                {"user_id": 10, "session_id": 2, "source_row_index": 201},
                {"user_id": 10, "session_id": 2, "source_row_index": 202},
                {"user_id": 10, "session_id": 3, "source_row_index": 300},
                {"user_id": 10, "session_id": 3, "source_row_index": 301},
                {"user_id": 10, "session_id": 3, "source_row_index": 302},
                {"user_id": 10, "session_id": 3, "source_row_index": 303},
            ]
        )
        teacher_trace = pd.DataFrame(
            [
                {"user_id": 10, "session_id": 101, "source_row_index": 100},
                {"user_id": 10, "session_id": 101, "source_row_index": 101},
                {"user_id": 10, "session_id": 202, "source_row_index": 200},
                {"user_id": 10, "session_id": 202, "source_row_index": 201},
                {"user_id": 10, "session_id": 202, "source_row_index": 202},
                {"user_id": 10, "session_id": 303, "source_row_index": 300},
                {"user_id": 10, "session_id": 303, "source_row_index": 301},
                {"user_id": 10, "session_id": 303, "source_row_index": 302},
                {"user_id": 10, "session_id": 303, "source_row_index": 303},
                {"user_id": 10, "session_id": 999, "source_row_index": 400},
                {"user_id": 10, "session_id": 999, "source_row_index": 401},
            ]
        )

        selected = default_dopamine_dual_mode_module.select_shared_representative_sessions(
            rollout_trace,
            teacher_trace,
            top_k=3,
        )

        self.assertEqual(list(selected["shared_session_rank"]), [1, 2, 3])
        self.assertEqual(list(selected["shared_session_length"]), [4, 3, 2])
        self.assertEqual(list(selected["rollout_session_id"]), [3, 2, 1])
        self.assertEqual(list(selected["teacher_forcing_session_id"]), [303, 202, 101])
        self.assertEqual(
            list(selected["session_fingerprint"]),
            [
                "10|300|303|4",
                "10|200|202|3",
                "10|100|101|2",
            ],
        )

    def test_render_default_dopamine_dual_mode_jpg_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            jpg_path = root / "dual_mode.jpg"
            summary_df = pd.DataFrame(
                [
                    {
                        "state_update_mode": "rollout",
                        "mean_abs_error": 0.11,
                        "median_abs_error": 0.10,
                    },
                    {
                        "state_update_mode": "teacher_forcing",
                        "mean_abs_error": 0.21,
                        "median_abs_error": 0.20,
                    },
                ]
            )
            trace_df = pd.DataFrame(
                [
                    {
                        "shared_session_rank": 1,
                        "state_update_mode": "rollout",
                        "user_id": 10,
                        "source_row_index_min": 100,
                        "source_row_index_max": 102,
                        "shared_session_length": 3,
                        "session_row_index": 0,
                        "dopamine_after": 1.0,
                    },
                    {
                        "shared_session_rank": 1,
                        "state_update_mode": "rollout",
                        "user_id": 10,
                        "source_row_index_min": 100,
                        "source_row_index_max": 102,
                        "shared_session_length": 3,
                        "session_row_index": 1,
                        "dopamine_after": 1.2,
                    },
                    {
                        "shared_session_rank": 1,
                        "state_update_mode": "teacher_forcing",
                        "user_id": 10,
                        "source_row_index_min": 100,
                        "source_row_index_max": 102,
                        "shared_session_length": 3,
                        "session_row_index": 0,
                        "dopamine_after": 0.9,
                    },
                    {
                        "shared_session_rank": 1,
                        "state_update_mode": "teacher_forcing",
                        "user_id": 10,
                        "source_row_index_min": 100,
                        "source_row_index_max": 102,
                        "shared_session_length": 3,
                        "session_row_index": 1,
                        "dopamine_after": 1.1,
                    },
                ]
            )

            default_dopamine_dual_mode_module.render_default_dopamine_dual_mode_jpg(
                summary_df=summary_df,
                trace_df=trace_df,
                output_path=jpg_path,
            )

            self.assertTrue(jpg_path.exists())
            self.assertGreater(jpg_path.stat().st_size, 0)

    def test_run_default_dopamine_dual_mode_executes_two_runs_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"

            def build_split_for_mode(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                return {
                    "train": pd.DataFrame(
                        [{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]
                    ),
                    "val": pd.DataFrame(
                        [{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]
                    ),
                    "meta": {
                        "state_update_mode": mode,
                        "post_filter_train_row_count": 24,
                        "post_filter_val_row_count": 21,
                        "post_filter_train_session_count": 3,
                        "post_filter_val_session_count": 2,
                    },
                }

            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            def prediction_side_effect(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                base_error = 0.1 if mode == "rollout" else 0.2
                return pd.DataFrame(
                    [
                        {
                            "eval_row_index": 0,
                            "user_id": 10,
                            "session_id": 501 if mode == "rollout" else 9001,
                            "video_id": 101,
                            "actual_bucket": 2,
                            "pred_bucket": 2,
                            "abs_error": base_error,
                        },
                        {
                            "eval_row_index": 1,
                            "user_id": 10,
                            "session_id": 502 if mode == "rollout" else 9002,
                            "video_id": 102,
                            "actual_bucket": 1,
                            "pred_bucket": 0,
                            "abs_error": base_error + 0.1,
                        },
                        {
                            "eval_row_index": 2,
                            "user_id": 10,
                            "session_id": 502 if mode == "rollout" else 9002,
                            "video_id": 103,
                            "actual_bucket": 3,
                            "pred_bucket": 3,
                            "abs_error": base_error + 0.2,
                        },
                    ]
                )

            def trace_side_effect(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                mode_shift = 0.0 if mode == "rollout" else 0.2
                session_id_map = {
                    "rollout": [501, 502, 503],
                    "teacher_forcing": [9001, 9002, 9003],
                }
                source_blocks = [
                    [100, 101],
                    [200, 201, 202],
                    [300, 301, 302, 303],
                ]
                rows = []
                eval_row_index = 0
                for session_idx, source_rows in enumerate(source_blocks):
                    session_id = session_id_map[mode][session_idx]
                    for step_idx, source_row_index in enumerate(source_rows):
                        rows.append(
                            {
                                "variant_name": kwargs["variant_name"],
                                "state_update_mode": mode,
                                "eval_row_index": eval_row_index,
                                "user_id": 10,
                                "session_id": session_id,
                                "session_row_index": step_idx,
                                "source_row_index": source_row_index,
                                "video_id": 101 + step_idx,
                                "timestamp": 100.0 + eval_row_index,
                                "event_timestamp": 100.0 + eval_row_index,
                                "actual_watch_ratio": 0.5 + 0.1 * step_idx,
                                "actual_bucket": 2,
                                "state_update_watch_ratio": 0.5 + 0.1 * step_idx,
                                "novelty_norm": 0.2,
                                "score_engagement": 0.4 + 0.1 * step_idx,
                                "dopamine_before": 1.0 + mode_shift + 0.1 * step_idx,
                                "dopamine_after": 1.1 + mode_shift + 0.2 * step_idx,
                                "expected_reward_before": 0.1,
                                "expected_reward_after": 0.2,
                                "session_peak_before": 0.1,
                                "session_peak_after": 0.3,
                            }
                        )
                        eval_row_index += 1
                return pd.DataFrame(rows)

            with mock.patch.object(
                default_dopamine_dual_mode_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                default_dopamine_dual_mode_module,
                "prepare_behavior_fit_train_val_split",
                side_effect=build_split_for_mode,
            ) as prepare_split, mock.patch.object(
                default_dopamine_dual_mode_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"dopamine_base": 1.0}, history),
            ) as fit_model, mock.patch.object(
                default_dopamine_dual_mode_module,
                "collect_behavior_predictions",
                side_effect=prediction_side_effect,
            ) as collect_predictions, mock.patch.object(
                default_dopamine_dual_mode_module,
                "collect_dopamine_state_traces",
                side_effect=trace_side_effect,
            ) as collect_traces:
                result = default_dopamine_dual_mode_module.run_default_dopamine_dual_mode_90k_experiment(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            self.assertEqual(prepare_split.call_count, 2)
            self.assertEqual(
                {call.kwargs["state_update_mode"] for call in prepare_split.call_args_list},
                {"rollout", "teacher_forcing"},
            )
            for call in prepare_split.call_args_list:
                self.assertEqual(call.kwargs["train_start_index"], 0)
                self.assertEqual(call.kwargs["train_row_count"], 90000)
                self.assertEqual(call.kwargs["val_row_count"], 10000)
                self.assertEqual(call.kwargs["min_session_length"], 20)

            self.assertEqual(fit_model.call_count, 2)
            for call in fit_model.call_args_list:
                self.assertEqual(call.kwargs["model_cls"], BehaviorFitModel)
                self.assertEqual(call.kwargs["optimizer_chunk_rows"], 5)
                self.assertEqual(call.kwargs["watch_ratio_value_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_ce_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_distance_loss_weight"], 1.0)

            self.assertEqual(collect_predictions.call_count, 2)
            self.assertEqual(collect_traces.call_count, 2)

            self.assertTrue(result["summary_path"].exists())
            self.assertTrue(result["trace_path"].exists())
            self.assertTrue(result["jpg_path"].exists())
            self.assertIn("default_dopamine_dual_mode_summary_train0_89999_val90000_99999", result["summary_path"].name)
            self.assertIn("default_dopamine_dual_mode_representative_traces_train0_89999_val90000_99999", result["trace_path"].name)

            summary_df = pd.read_csv(result["summary_path"])
            trace_df = pd.read_csv(result["trace_path"])
            self.assertEqual(len(summary_df), 2)
            self.assertEqual(set(summary_df["state_update_mode"]), {"rollout", "teacher_forcing"})
            self.assertIn("mean_abs_error", summary_df.columns)
            self.assertIn("median_abs_error", summary_df.columns)
            self.assertTrue((summary_df["shared_representative_session_count"] == 3).all())
            self.assertEqual(len(trace_df), 18)
            self.assertEqual(set(trace_df["shared_session_rank"]), {1, 2, 3})
            self.assertEqual(set(trace_df["state_update_mode"]), {"rollout", "teacher_forcing"})

    def test_run_content_only_fatigue_90k_executes_two_modes_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "reports"

            def build_split_for_mode(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                return {
                    "train": pd.DataFrame(
                        [{"user_id": 10, "video_id": 100, "watch_ratio": 0.4}]
                    ),
                    "val": pd.DataFrame(
                        [{"user_id": 10, "video_id": 101, "watch_ratio": 0.5}]
                    ),
                    "meta": {
                        "state_update_mode": mode,
                        "post_filter_train_row_count": 24,
                        "post_filter_val_row_count": 21,
                        "post_filter_train_session_count": 3,
                        "post_filter_val_session_count": 2,
                    },
                }

            history = [
                {
                    "epoch": 1,
                    "val": {
                        "avg_loss": 0.2,
                        "avg_bucket_ce": 0.3,
                        "avg_bucket_distance": 0.4,
                        "avg_value_l1": 0.5,
                    },
                }
            ]

            def prediction_side_effect(*args, **kwargs):
                mode = kwargs["state_update_mode"]
                base_error = 0.1 if mode == "rollout" else 0.2
                return pd.DataFrame(
                    [
                        {
                            "eval_row_index": 0,
                            "user_id": 10,
                            "session_id": 5,
                            "video_id": 101,
                            "actual_bucket": 2,
                            "pred_bucket": 2,
                            "abs_error": base_error,
                        },
                        {
                            "eval_row_index": 1,
                            "user_id": 10,
                            "session_id": 7,
                            "video_id": 102,
                            "actual_bucket": 1,
                            "pred_bucket": 0,
                            "abs_error": base_error + 0.1,
                        },
                        {
                            "eval_row_index": 2,
                            "user_id": 10,
                            "session_id": 7,
                            "video_id": 103,
                            "actual_bucket": 3,
                            "pred_bucket": 3,
                            "abs_error": base_error + 0.2,
                        },
                    ]
                )

            with mock.patch.object(
                content_only_fatigue_runner_module,
                "build_feature_provider",
                return_value=(object(), root / "data"),
            ) as build_provider, mock.patch.object(
                content_only_fatigue_runner_module,
                "prepare_behavior_fit_train_val_split",
                side_effect=build_split_for_mode,
            ) as prepare_split, mock.patch.object(
                content_only_fatigue_runner_module,
                "fit_behavior_model_on_split",
                return_value=(object(), {"effective_threshold_base": 1.0}, history),
            ) as fit_model, mock.patch.object(
                content_only_fatigue_runner_module,
                "collect_behavior_predictions",
                side_effect=prediction_side_effect,
            ) as collect_predictions:
                result = content_only_fatigue_runner_module.run_content_only_fatigue_90k_experiment(
                    project_root=root,
                    output_dir=output_dir,
                )

            build_provider.assert_called_once_with(root)
            self.assertEqual(prepare_split.call_count, 2)
            self.assertEqual(
                {call.kwargs["state_update_mode"] for call in prepare_split.call_args_list},
                {"rollout", "teacher_forcing"},
            )
            for call in prepare_split.call_args_list:
                self.assertEqual(call.kwargs["train_start_index"], 0)
                self.assertEqual(call.kwargs["train_row_count"], 90000)
                self.assertEqual(call.kwargs["val_row_count"], 10000)
                self.assertEqual(call.kwargs["min_session_length"], 20)

            self.assertEqual(fit_model.call_count, 2)
            for call in fit_model.call_args_list:
                self.assertEqual(
                    call.kwargs["model_cls"],
                    ContentOnlyFatigueBehaviorFitModel,
                )
                self.assertEqual(call.kwargs["optimizer_chunk_rows"], 5)
                self.assertEqual(call.kwargs["watch_ratio_value_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_ce_loss_weight"], 1.0)
                self.assertEqual(call.kwargs["watch_ratio_bucket_distance_loss_weight"], 1.0)

            self.assertEqual(collect_predictions.call_count, 2)
            self.assertTrue(result["summary_path"].exists())
            self.assertIn(
                "content_only_fatigue_summary_train0_89999_val90000_99999",
                result["summary_path"].name,
            )

            summary_df = pd.read_csv(result["summary_path"])
            self.assertEqual(len(summary_df), 2)
            self.assertEqual(
                set(summary_df["state_update_mode"]),
                {"rollout", "teacher_forcing"},
            )
            self.assertIn("average_abs_error", summary_df.columns)
            self.assertIn("median_abs_error", summary_df.columns)
            self.assertTrue((summary_df["model_name"] == "content_only_fatigue").all())

    def test_main_uses_rollout_results_for_env(self):
        feature_provider = object()
        rollout_params = {"dopamine_base": 1.0}
        experiment = {
            "feature_provider": feature_provider,
            "reuse_saved_results": True,
            "cache_status": "hit",
            "selected_loss_weights": {"value": 4.0, "bucket_ce": 1.0, "bucket_dist": 1.0},
            "rollout_result": {
                "learned_params": rollout_params,
                "summary": {
                    "mean_abs_error": 0.1,
                    "p90_abs_error": 0.2,
                    "bucket_match_rate": 0.3,
                },
            },
            "split": {
                "meta": {
                    "train_source_name": "big_matrix.csv",
                    "val_source_name": "small_matrix.csv",
                    "post_filter_train_row_count": 10,
                    "post_filter_val_row_count": 4,
                }
            },
            "output": {"html_path": Path("rollout.html")},
        }

        class DummyEnv:
            @staticmethod
            def reset():
                return {"user": np.zeros(1, dtype=np.float32)}

        with mock.patch.object(main_module, "run_final_watch_ratio_reports", return_value=experiment):
            with mock.patch.object(main_module, "make_env", return_value=DummyEnv()) as make_env_mock:
                with mock.patch("builtins.print") as print_mock:
                    main_module.main()

        make_env_mock.assert_called_once_with(
            feature_provider=feature_provider,
            num_candidates=5,
            slate_size=1,
            seed=0,
            behavior_params=rollout_params,
        )
        joined_output = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("cache_status=hit", joined_output)
        self.assertIn("selected_loss_weights=value=4.000, bucket_ce=1.000, bucket_dist=1.000", joined_output)

    def test_linear_agent_dopamine_layout_features(self):
        agent = LinearAgent(embedding_dim=32, observation_layout="dopamine")
        obs = {
            "user": np.concatenate(
                [
                    np.array([0.2, 1.1, 0.3, 0.4], dtype=np.float32),
                    np.ones(32, dtype=np.float32),
                ],
                axis=0,
            ),
            "doc": {
                "0": np.ones(32, dtype=np.float32),
                "1": np.zeros(32, dtype=np.float32),
            },
        }

        features = agent.build_features(obs)
        self.assertEqual(features.shape, (2, 6))
        self.assertAlmostEqual(float(features[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(features[0, 1]), 1.1, places=6)

    def test_linear_agent_no_dopamine_layout_features(self):
        agent = LinearAgent(embedding_dim=32, observation_layout="no_dopamine")
        obs = {
            "user": np.concatenate(
                [
                    np.array([0.25], dtype=np.float32),
                    np.ones(32, dtype=np.float32),
                ],
                axis=0,
            ),
            "doc": {
                "0": np.ones(32, dtype=np.float32),
                "1": np.zeros(32, dtype=np.float32),
            },
        }

        features = agent.build_features(obs)
        self.assertEqual(features.shape, (2, 3))
        self.assertAlmostEqual(float(features[0, 0]), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
