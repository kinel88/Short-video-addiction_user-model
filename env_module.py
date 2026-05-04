import math
from typing import List

import numpy as np
from gym import spaces

from recsim import document
from recsim import user
from recsim.simulator import environment
from recsim.simulator import recsim_gym

from train_module import (
    CATEGORY_BLOCK_DIM,
    DOPAMINE_PARAM_KEYS,
    NOVELTY_HISTORY_WINDOW,
    REPEAT_WATCH_PASS_CAP,
    _apply_scalar_dopamine_update_scaffold,
    _compute_scalar_time_weighted_average_reward,
    compute_threshold_banded_watch_ratio_components,
    compute_novelty_norm,
    watch_ratio_to_bucket_index,
)


# =========================================================
# 0.1 Default behavior parameters
# =========================================================
DEFAULT_BEHAVIOR_PARAMS = {
    "score_preference_match": 1.40,
    "score_quality": 0.40,
    "score_novelty": 0.30,
    "score_creator_preference": 0.50,
    "score_preference_match_by_id": {},
    "score_quality_by_id": {},
    "score_novelty_by_id": {},
    "score_creator_preference_by_id": {},
    "effective_threshold_base": 0.85,
    "effective_threshold_dopamine": 0.25,
    "effective_threshold_expected_reward": 0.25,
    "effective_threshold_automaticity": 0.25,
    "effective_threshold_fatigue": 0.35,
    "watch_gain_base": 1.30,
    "watch_gain_dopamine": 0.55,
    "fatigue_duration_penalty": 0.25,
    "repeat_watch_decay": 0.80,
    "repeat_pass_cap": REPEAT_WATCH_PASS_CAP,
    "fatigue_watch_ratio_coef": 0.06,
    "fatigue_high_engaged_bonus": 0.20,
    "dopamine_base": 1.00,
    "dopamine_normal_level": 1.35,
    "dopamine_baseline_return_strength": 0.20,
    "dopamine_habit_growth_rate": 0.01,
    "dopamine_habit_max_gain": 0.35,
    "dopamine_score_engagement_coef": 0.55,
    "dopamine_expected_reward_coef": 0.40,
    "dopamine_peak_coef": 0.45,
    "dopamine_mix_alpha": 0.70,
    "expected_alpha": 0.35,
    "expected_reward_peak_coef": 0.45,
    "expected_reward_average_coef": 0.40,
    "average_reward_decay": 0.85,
    "peak_decay": 0.85,
    "user_automaticity_default": 0.50,
    "user_automaticity_by_id": {},
}

DEFAULT_NO_DOPAMINE_BEHAVIOR_PARAMS = dict(DEFAULT_BEHAVIOR_PARAMS)
for key in list(DOPAMINE_PARAM_KEYS) + ["expected_alpha", "peak_decay"]:
    DEFAULT_NO_DOPAMINE_BEHAVIOR_PARAMS.pop(key, None)

NO_DOPAMINE_REMOVED_PARAM_KEYS = set(DOPAMINE_PARAM_KEYS) | {"expected_alpha", "peak_decay"}

DEFAULT_CONTENT_ONLY_FATIGUE_BEHAVIOR_PARAMS = dict(DEFAULT_NO_DOPAMINE_BEHAVIOR_PARAMS)
DEFAULT_CONTENT_ONLY_FATIGUE_BEHAVIOR_PARAMS.pop("effective_threshold_fatigue", None)

CONTENT_ONLY_FATIGUE_REMOVED_PARAM_KEYS = (
    set(DOPAMINE_PARAM_KEYS) |
    {"expected_alpha", "peak_decay", "effective_threshold_fatigue"}
)


# =========================================================
# 1. Global embedding dim config
# =========================================================
EMBEDDING_DIM = 32


def set_embedding_dim(dim):
    global EMBEDDING_DIM
    EMBEDDING_DIM = int(dim)


# =========================================================
# 2. Document
# =========================================================
class CorpusDocument(document.AbstractDocument):
    def __init__(
        self,
        doc_id,
        video_id,
        item_vec,
        category_vec,
        quality_norm,
        author_id,
        video_duration_ms,
    ):
        self.video_id = int(video_id)
        self.embedding = np.asarray(item_vec, dtype=np.float32)
        self.category_vec = np.asarray(category_vec, dtype=np.float32)
        self.quality_norm = float(quality_norm)
        self.author_id = int(author_id)
        self.video_duration_ms = float(video_duration_ms)
        super(CorpusDocument, self).__init__(doc_id)

    def create_observation(self):
        return self.embedding.astype(np.float32)

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(EMBEDDING_DIM,),
            dtype=np.float32,
        )


class CorpusDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, feature_provider, doc_ctor=CorpusDocument, seed=0):
        super(CorpusDocumentSampler, self).__init__(doc_ctor, seed)
        self.feature_provider = feature_provider
        self.video_ids = self.feature_provider.get_all_video_ids()
        if len(self.video_ids) == 0:
            raise ValueError("feature_provider must expose at least one video")
        self._shuffled_video_ids: List[int] = []
        self._cursor = 0

    def _reshuffle(self):
        order = self._rng.permutation(len(self.video_ids))
        self._shuffled_video_ids = [int(self.video_ids[idx]) for idx in order.tolist()]
        self._cursor = 0

    def sample_document(self):
        if self._cursor >= len(self._shuffled_video_ids):
            self._reshuffle()

        video_id = int(self._shuffled_video_ids[self._cursor])
        self._cursor += 1
        feat = self.feature_provider.get_video_features(video_id)

        return self._doc_ctor(
            doc_id=video_id,
            video_id=video_id,
            item_vec=feat["item_vec"],
            category_vec=feat["category_vec_31"],
            quality_norm=feat["quality_norm"],
            author_id=feat["author_id"],
            video_duration_ms=feat["video_duration_ms"],
        )


# =========================================================
# 3. User state
# =========================================================
class SimpleUserState(user.AbstractUserState):
    def __init__(
        self,
        user_id,
        preference_vector,
        fatigue,
        dopamine,
        expected_reward,
        session_peak,
        time_weighted_average_reward,
        user_automaticity,
        time_budget,
        recent_category_history=None,
        recent_score_history=None,
        step_count=0,
    ):
        self.user_id = int(user_id)
        self.preference_vector = np.asarray(preference_vector, dtype=np.float32)
        self.fatigue = float(fatigue)
        self.dopamine = float(dopamine)
        self.expected_reward = float(expected_reward)
        self.session_peak = float(session_peak)
        self.time_weighted_average_reward = float(time_weighted_average_reward)
        self.user_automaticity = float(user_automaticity)
        self.time_budget = int(time_budget)
        self.recent_category_history = [
            np.asarray(vec, dtype=np.float32) for vec in (recent_category_history or [])
        ]
        self.recent_score_history = [float(v) for v in (recent_score_history or [])]
        self.step_count = int(step_count)

    def create_observation(self):
        scalar_part = np.array(
            [
                self.fatigue,
                self.dopamine,
                self.expected_reward,
                self.session_peak,
            ],
            dtype=np.float32,
        )
        return np.concatenate([scalar_part, self.preference_vector], axis=0)

    @staticmethod
    def observation_space():
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + EMBEDDING_DIM,),
            dtype=np.float32,
        )


class NoDopamineUserState(user.AbstractUserState):
    def __init__(
        self,
        user_id,
        preference_vector,
        fatigue,
        time_budget,
        recent_category_history=None,
        step_count=0,
    ):
        self.user_id = int(user_id)
        self.preference_vector = np.asarray(preference_vector, dtype=np.float32)
        self.fatigue = float(fatigue)
        self.time_budget = int(time_budget)
        self.recent_category_history = [
            np.asarray(vec, dtype=np.float32) for vec in (recent_category_history or [])
        ]
        self.step_count = int(step_count)

    def create_observation(self):
        scalar_part = np.array([self.fatigue], dtype=np.float32)
        return np.concatenate([scalar_part, self.preference_vector], axis=0)

    @staticmethod
    def observation_space():
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + EMBEDDING_DIM,),
            dtype=np.float32,
        )


class ContentOnlyFatigueUserState(NoDopamineUserState):
    """Content-only state with fatigue and novelty history only."""


class CorpusUserSampler(user.AbstractUserSampler):
    def __init__(
        self,
        feature_provider,
        initial_dopamine,
        user_automaticity_default=0.5,
        user_automaticity_by_id=None,
        time_budget=10,
        user_ctor=SimpleUserState,
        seed=0,
    ):
        super(CorpusUserSampler, self).__init__(user_ctor=user_ctor, seed=seed)
        self.feature_provider = feature_provider
        self.initial_dopamine = float(initial_dopamine)
        self.user_automaticity_default = float(user_automaticity_default)
        self.user_automaticity_by_id = {
            int(user_id): float(value)
            for user_id, value in dict(user_automaticity_by_id or {}).items()
        }
        self.time_budget = int(time_budget)
        self.user_ids = self.feature_provider.get_all_user_ids()
        if len(self.user_ids) == 0:
            raise ValueError("feature_provider must expose at least one user")
        self._shuffled_user_ids: List[int] = []
        self._cursor = 0

    def _reshuffle(self):
        order = self._rng.permutation(len(self.user_ids))
        self._shuffled_user_ids = [int(self.user_ids[idx]) for idx in order.tolist()]
        self._cursor = 0

    def sample_user(self):
        if self._cursor >= len(self._shuffled_user_ids):
            self._reshuffle()

        user_id = int(self._shuffled_user_ids[self._cursor])
        self._cursor += 1
        user_feat = self.feature_provider.get_user_features(user_id)
        user_automaticity = self.user_automaticity_by_id.get(
            user_id,
            self.user_automaticity_default,
        )

        return self._user_ctor(
            user_id=user_id,
            preference_vector=user_feat["user_vec"],
            fatigue=0.0,
            dopamine=self.initial_dopamine,
            expected_reward=0.0,
            session_peak=0.0,
            time_weighted_average_reward=0.0,
            user_automaticity=user_automaticity,
            time_budget=self.time_budget,
            recent_category_history=[],
            recent_score_history=[],
            step_count=0,
        )


class NoDopamineUserSampler(user.AbstractUserSampler):
    def __init__(
        self,
        feature_provider,
        time_budget=10,
        user_ctor=NoDopamineUserState,
        seed=0,
    ):
        super(NoDopamineUserSampler, self).__init__(user_ctor=user_ctor, seed=seed)
        self.feature_provider = feature_provider
        self.time_budget = int(time_budget)
        self.user_ids = self.feature_provider.get_all_user_ids()
        if len(self.user_ids) == 0:
            raise ValueError("feature_provider must expose at least one user")
        self._shuffled_user_ids: List[int] = []
        self._cursor = 0

    def _reshuffle(self):
        order = self._rng.permutation(len(self.user_ids))
        self._shuffled_user_ids = [int(self.user_ids[idx]) for idx in order.tolist()]
        self._cursor = 0

    def sample_user(self):
        if self._cursor >= len(self._shuffled_user_ids):
            self._reshuffle()

        user_id = int(self._shuffled_user_ids[self._cursor])
        self._cursor += 1
        user_feat = self.feature_provider.get_user_features(user_id)

        return self._user_ctor(
            user_id=user_id,
            preference_vector=user_feat["user_vec"],
            fatigue=0.0,
            time_budget=self.time_budget,
            recent_category_history=[],
            step_count=0,
        )


# =========================================================
# 4. Response
# =========================================================
class SimpleResponse(user.AbstractResponse):
    def __init__(
        self,
        watch_ratio=0.0,
        watch_ratio_bucket=0,
        high_engaged=False,
        watch_time=0.0,
    ):
        self.watch_ratio = float(watch_ratio)
        self.watch_ratio_bucket = int(watch_ratio_bucket)
        self.high_engaged = int(high_engaged)
        self.watch_time = float(watch_time)

    def create_observation(self):
        return {
            "watch_ratio": np.array([self.watch_ratio], dtype=np.float32),
            "watch_ratio_bucket": self.watch_ratio_bucket,
            "high_engaged": self.high_engaged,
            "watch_time": np.array([self.watch_time], dtype=np.float32),
        }

    @staticmethod
    def response_space():
        return spaces.Dict({
            "watch_ratio": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1e6], dtype=np.float32),
                dtype=np.float32,
            ),
            "watch_ratio_bucket": spaces.Discrete(6),
            "high_engaged": spaces.Discrete(2),
            "watch_time": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1e6], dtype=np.float32),
                dtype=np.float32,
            ),
        })


# =========================================================
# 5. User model
# =========================================================
class SimpleUserModel(user.AbstractUserModel):
    def __init__(
        self,
        slate_size,
        feature_provider,
        user_state_ctor=SimpleUserState,
        response_model_ctor=SimpleResponse,
        seed=0,
        behavior_params=None,
        time_budget=10,
    ):
        self.feature_provider = feature_provider
        self.embedding_dim = int(feature_provider.embedding_dim)
        self.dot_scale = math.sqrt(float(self.embedding_dim))

        self.behavior_params = dict(DEFAULT_BEHAVIOR_PARAMS)
        if behavior_params is not None:
            self.behavior_params.update(behavior_params)

        sampler = CorpusUserSampler(
            feature_provider=feature_provider,
            initial_dopamine=self._resolve_session_dopamine_baseline(self.behavior_params),
            user_automaticity_default=self.behavior_params.get(
                "user_automaticity_default",
                0.5,
            ),
            user_automaticity_by_id=self.behavior_params.get(
                "user_automaticity_by_id",
                {},
            ),
            time_budget=time_budget,
            user_ctor=user_state_ctor,
            seed=seed,
        )
        self._simple_user_sampler = sampler
        super(SimpleUserModel, self).__init__(
            response_model_ctor=response_model_ctor,
            user_sampler=sampler,
            slate_size=slate_size,
        )

    @staticmethod
    def _resolve_session_dopamine_baseline(behavior_params):
        return float(behavior_params.get("dopamine_base", 0.0))

    def get_behavior_params(self):
        return dict(self.behavior_params)

    def set_behavior_params(self, new_params):
        self.behavior_params.update(new_params)
        if any(
            key in new_params
            for key in ("dopamine_base", "dopamine_normal_level")
        ):
            self._simple_user_sampler.initial_dopamine = self._resolve_session_dopamine_baseline(
                self.behavior_params
            )
        if any(
            key in new_params
            for key in ("user_automaticity_default", "user_automaticity_by_id")
        ):
            self._simple_user_sampler.user_automaticity_default = float(
                self.behavior_params.get("user_automaticity_default", 0.5)
            )
            self._simple_user_sampler.user_automaticity_by_id = {
                int(user_id): float(value)
                for user_id, value in dict(
                    self.behavior_params.get("user_automaticity_by_id", {})
                ).items()
            }

    def _resolve_user_behavior_weight(self, scalar_key, per_user_key):
        user_id = int(self._user_state.user_id)
        per_user = dict(self.behavior_params.get(per_user_key, {}))
        if user_id in per_user:
            return float(per_user[user_id])
        user_id_str = str(user_id)
        if user_id_str in per_user:
            return float(per_user[user_id_str])
        return float(self.behavior_params[scalar_key])

    def _compute_doc_signals(self, doc):
        item_vec = np.asarray(doc.create_observation(), dtype=np.float32)
        user_vec = self._user_state.preference_vector

        pref_dot = float(np.dot(user_vec, item_vec) / self.dot_scale)
        pref_match = 1.0 / (1.0 + np.exp(-pref_dot))

        creator_pref_norm = self.feature_provider.compute_creator_preference_norm(
            user_id=self._user_state.user_id,
            author_id=doc.author_id,
        )
        novelty_norm = compute_novelty_norm(
            category_vec=doc.category_vec,
            history_category_vectors=self._user_state.recent_category_history,
        )
        score_preference_match = self._resolve_user_behavior_weight(
            "score_preference_match",
            "score_preference_match_by_id",
        )
        score_quality = self._resolve_user_behavior_weight(
            "score_quality",
            "score_quality_by_id",
        )
        score_novelty = self._resolve_user_behavior_weight(
            "score_novelty",
            "score_novelty_by_id",
        )
        score_creator_preference = self._resolve_user_behavior_weight(
            "score_creator_preference",
            "score_creator_preference_by_id",
        )
        video_score = (
            score_preference_match * pref_match +
            score_quality * doc.quality_norm +
            score_novelty * novelty_norm +
            score_creator_preference * creator_pref_norm
        )

        return {
            "item_vec": item_vec,
            "pref_dot": pref_dot,
            "pref_match": pref_match,
            "creator_pref_norm": float(creator_pref_norm),
            "novelty_norm": float(novelty_norm),
            "video_score": float(video_score),
            "quality_norm": float(doc.quality_norm),
            "category_vec": np.asarray(doc.category_vec, dtype=np.float32),
        }

    def simulate_response(self, slate_documents):
        responses = []
        p = self.behavior_params

        for doc in slate_documents:
            signals = self._compute_doc_signals(doc)

            current_dopamine = float(self._user_state.dopamine)
            effective_threshold = (
                p["effective_threshold_base"] +
                p["effective_threshold_expected_reward"] * self._user_state.expected_reward +
                p["effective_threshold_automaticity"] * self._user_state.user_automaticity
            )
            effective_signal = signals["video_score"] - effective_threshold
            watch_duration_scale = math.exp(
                -p["fatigue_duration_penalty"] * self._user_state.fatigue
            )
            raw_watch_ratio = compute_threshold_banded_watch_ratio_components(
                video_score=signals["video_score"],
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=p["watch_gain_base"],
                repeat_watch_decay=p["repeat_watch_decay"],
                repeat_pass_cap=p.get("repeat_pass_cap", REPEAT_WATCH_PASS_CAP),
            )["pred_watch_ratio"]
            watch_ratio = 0.0 if raw_watch_ratio < 1e-4 else float(raw_watch_ratio)
            high_engaged = watch_ratio > 1.0
            watch_time = watch_ratio * max(doc.video_duration_ms, 0.0) / 1000.0
            watch_ratio_bucket = watch_ratio_to_bucket_index(watch_ratio)

            responses.append(
                self._response_model_ctor(
                    watch_ratio=watch_ratio,
                    watch_ratio_bucket=watch_ratio_bucket,
                    high_engaged=high_engaged,
                    watch_time=watch_time,
                )
            )

        return responses

    def update_state(self, slate_documents, responses):
        p = self.behavior_params

        current_score_engagements = []
        current_categories = []
        engagements = []
        high_engaged_any = False

        for doc, resp in zip(slate_documents, responses):
            signals = self._compute_doc_signals(doc)
            engagement = max(float(resp.watch_ratio), 0.0)
            current_score_engagements.append(float(signals["video_score"]) * engagement)
            current_categories.append(np.asarray(signals["category_vec"], dtype=np.float32))
            engagements.append(engagement)
            if resp.high_engaged:
                high_engaged_any = True

        self._user_state.recent_category_history.extend(current_categories)
        self._user_state.recent_category_history = self._user_state.recent_category_history[-NOVELTY_HISTORY_WINDOW:]

        self._user_state.recent_score_history.extend(current_score_engagements)
        self._user_state.recent_score_history = self._user_state.recent_score_history[-NOVELTY_HISTORY_WINDOW:]

        decayed_scores = []
        score_history = self._user_state.recent_score_history[-NOVELTY_HISTORY_WINDOW:]
        for idx, score in enumerate(score_history):
            age = len(score_history) - 1 - idx
            decayed_scores.append(float(score) * (float(p["peak_decay"]) ** age))
        self._user_state.session_peak = max(decayed_scores) if len(decayed_scores) > 0 else 0.0
        self._user_state.time_weighted_average_reward = (
            _compute_scalar_time_weighted_average_reward(
                recent_value_history=score_history,
                average_reward_decay=p["average_reward_decay"],
            )
        )
        self._user_state.expected_reward = (
            p["expected_reward_peak_coef"] * self._user_state.session_peak +
            p["expected_reward_average_coef"] * self._user_state.time_weighted_average_reward
        )

        fatigue_gain = 0.0
        if len(engagements) > 0:
            fatigue_gain += p["fatigue_watch_ratio_coef"] * float(np.mean(engagements))
        if high_engaged_any:
            fatigue_gain += p["fatigue_high_engaged_bonus"]
        self._user_state.fatigue = min(1.0, self._user_state.fatigue + fatigue_gain)

        self._user_state.dopamine = float(
            _apply_scalar_dopamine_update_scaffold(
                prev_dopamine=self._user_state.dopamine,
                session_baseline=self._resolve_session_dopamine_baseline(p),
                baseline_return_strength=p["dopamine_baseline_return_strength"],
                swipe_count=self._user_state.step_count + len(responses),
                habit_growth_rate=p["dopamine_habit_growth_rate"],
                habit_max_gain=p["dopamine_habit_max_gain"],
                reward_drive=(
                    p["dopamine_expected_reward_coef"] * self._user_state.expected_reward
                    + p["dopamine_peak_coef"] * self._user_state.session_peak
                ),
            )[0]
        )

        self._user_state.time_budget -= 1
        self._user_state.step_count += len(responses)

    def is_terminal(self):
        return self._user_state.time_budget <= 0


class NoDopamineUserModel(SimpleUserModel):
    """
    Dopamine-free user simulator.

    It removes dopamine and the residual reward/peak-memory state, exposing
    only fatigue plus the user preference vector in the user observation.
    """

    def __init__(
        self,
        slate_size,
        feature_provider,
        user_state_ctor=NoDopamineUserState,
        response_model_ctor=SimpleResponse,
        seed=0,
        behavior_params=None,
        time_budget=10,
    ):
        self.feature_provider = feature_provider
        self.embedding_dim = int(feature_provider.embedding_dim)
        self.dot_scale = math.sqrt(float(self.embedding_dim))

        self.behavior_params = dict(DEFAULT_NO_DOPAMINE_BEHAVIOR_PARAMS)
        if behavior_params is not None:
            filtered_params = {
                key: value
                for key, value in dict(behavior_params).items()
                if key not in NO_DOPAMINE_REMOVED_PARAM_KEYS
            }
            self.behavior_params.update(filtered_params)

        sampler = NoDopamineUserSampler(
            feature_provider=feature_provider,
            time_budget=time_budget,
            user_ctor=user_state_ctor,
            seed=seed,
        )
        self._simple_user_sampler = sampler
        user.AbstractUserModel.__init__(
            self,
            response_model_ctor=response_model_ctor,
            user_sampler=sampler,
            slate_size=slate_size,
        )

    def set_behavior_params(self, new_params):
        filtered_params = {
            key: value
            for key, value in dict(new_params).items()
            if key not in NO_DOPAMINE_REMOVED_PARAM_KEYS
        }
        self.behavior_params.update(filtered_params)

    def simulate_response(self, slate_documents):
        responses = []
        p = self.behavior_params

        for doc in slate_documents:
            signals = self._compute_doc_signals(doc)

            effective_threshold = (
                p["effective_threshold_base"]
            )
            effective_signal = signals["video_score"] - effective_threshold
            watch_duration_scale = math.exp(
                -p["fatigue_duration_penalty"] * self._user_state.fatigue
            )
            raw_watch_ratio = compute_threshold_banded_watch_ratio_components(
                video_score=signals["video_score"],
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=p["watch_gain_base"],
                repeat_watch_decay=p["repeat_watch_decay"],
                repeat_pass_cap=p.get("repeat_pass_cap", REPEAT_WATCH_PASS_CAP),
            )["pred_watch_ratio"]
            watch_ratio = 0.0 if raw_watch_ratio < 1e-4 else float(raw_watch_ratio)
            high_engaged = watch_ratio > 1.0
            watch_time = watch_ratio * max(doc.video_duration_ms, 0.0) / 1000.0
            watch_ratio_bucket = watch_ratio_to_bucket_index(watch_ratio)

            responses.append(
                self._response_model_ctor(
                    watch_ratio=watch_ratio,
                    watch_ratio_bucket=watch_ratio_bucket,
                    high_engaged=high_engaged,
                    watch_time=watch_time,
                )
            )

        return responses

    def update_state(self, slate_documents, responses):
        p = self.behavior_params

        current_categories = []
        engagements = []
        high_engaged_any = False

        for doc, resp in zip(slate_documents, responses):
            signals = self._compute_doc_signals(doc)
            current_categories.append(np.asarray(signals["category_vec"], dtype=np.float32))
            engagements.append(max(float(resp.watch_ratio), 0.0))
            if resp.high_engaged:
                high_engaged_any = True

        self._user_state.recent_category_history.extend(current_categories)
        self._user_state.recent_category_history = self._user_state.recent_category_history[-NOVELTY_HISTORY_WINDOW:]

        fatigue_gain = 0.0
        if len(engagements) > 0:
            fatigue_gain += p["fatigue_watch_ratio_coef"] * float(np.mean(engagements))
        if high_engaged_any:
            fatigue_gain += p["fatigue_high_engaged_bonus"]
        self._user_state.fatigue = min(1.0, self._user_state.fatigue + fatigue_gain)

        self._user_state.time_budget -= 1
        self._user_state.step_count += len(responses)


class ContentOnlyFatigueUserModel(NoDopamineUserModel):
    """
    Content-only user simulator.

    It keeps the current content score and repeat-watch mapping, while
    removing every dopamine-related parameter and state. Fatigue only
    compresses overall watch duration.
    """

    def __init__(
        self,
        slate_size,
        feature_provider,
        user_state_ctor=ContentOnlyFatigueUserState,
        response_model_ctor=SimpleResponse,
        seed=0,
        behavior_params=None,
        time_budget=10,
    ):
        self.feature_provider = feature_provider
        self.embedding_dim = int(feature_provider.embedding_dim)
        self.dot_scale = math.sqrt(float(self.embedding_dim))

        self.behavior_params = dict(DEFAULT_CONTENT_ONLY_FATIGUE_BEHAVIOR_PARAMS)
        if behavior_params is not None:
            filtered_params = {
                key: value
                for key, value in dict(behavior_params).items()
                if key not in CONTENT_ONLY_FATIGUE_REMOVED_PARAM_KEYS
            }
            self.behavior_params.update(filtered_params)

        sampler = NoDopamineUserSampler(
            feature_provider=feature_provider,
            time_budget=time_budget,
            user_ctor=user_state_ctor,
            seed=seed,
        )
        self._simple_user_sampler = sampler
        user.AbstractUserModel.__init__(
            self,
            response_model_ctor=response_model_ctor,
            user_sampler=sampler,
            slate_size=slate_size,
        )

    def set_behavior_params(self, new_params):
        filtered_params = {
            key: value
            for key, value in dict(new_params).items()
            if key not in CONTENT_ONLY_FATIGUE_REMOVED_PARAM_KEYS
        }
        self.behavior_params.update(filtered_params)


# =========================================================
# 6. Reward
# =========================================================
def total_watch_time_reward(responses):
    return sum(r.watch_time for r in responses)


# =========================================================
# 7. Build environment
# =========================================================
def make_env(
    feature_provider,
    num_candidates=5,
    slate_size=1,
    seed=0,
    behavior_params=None,
    time_budget=10,
):
    set_embedding_dim(feature_provider.embedding_dim)

    user_model = SimpleUserModel(
        slate_size=slate_size,
        feature_provider=feature_provider,
        user_state_ctor=SimpleUserState,
        response_model_ctor=SimpleResponse,
        seed=seed,
        behavior_params=behavior_params,
        time_budget=time_budget,
    )

    doc_sampler = CorpusDocumentSampler(
        feature_provider=feature_provider,
        doc_ctor=CorpusDocument,
        seed=seed,
    )

    raw_env = environment.SingleUserEnvironment(
        user_model=user_model,
        document_sampler=doc_sampler,
        num_candidates=num_candidates,
        slate_size=slate_size,
        resample_documents=True,
    )

    env = recsim_gym.RecSimGymEnv(
        raw_environment=raw_env,
        reward_aggregator=total_watch_time_reward,
    )
    return env


def make_no_dopamine_env(
    feature_provider,
    num_candidates=5,
    slate_size=1,
    seed=0,
    behavior_params=None,
    time_budget=10,
):
    set_embedding_dim(feature_provider.embedding_dim)

    user_model = NoDopamineUserModel(
        slate_size=slate_size,
        feature_provider=feature_provider,
        user_state_ctor=NoDopamineUserState,
        response_model_ctor=SimpleResponse,
        seed=seed,
        behavior_params=behavior_params,
        time_budget=time_budget,
    )

    doc_sampler = CorpusDocumentSampler(
        feature_provider=feature_provider,
        doc_ctor=CorpusDocument,
        seed=seed,
    )

    raw_env = environment.SingleUserEnvironment(
        user_model=user_model,
        document_sampler=doc_sampler,
        num_candidates=num_candidates,
        slate_size=slate_size,
        resample_documents=True,
    )

    env = recsim_gym.RecSimGymEnv(
        raw_environment=raw_env,
        reward_aggregator=total_watch_time_reward,
    )
    return env


def make_content_only_fatigue_env(
    feature_provider,
    num_candidates=5,
    slate_size=1,
    seed=0,
    behavior_params=None,
    time_budget=10,
):
    set_embedding_dim(feature_provider.embedding_dim)

    user_model = ContentOnlyFatigueUserModel(
        slate_size=slate_size,
        feature_provider=feature_provider,
        user_state_ctor=ContentOnlyFatigueUserState,
        response_model_ctor=SimpleResponse,
        seed=seed,
        behavior_params=behavior_params,
        time_budget=time_budget,
    )

    doc_sampler = CorpusDocumentSampler(
        feature_provider=feature_provider,
        doc_ctor=CorpusDocument,
        seed=seed,
    )

    raw_env = environment.SingleUserEnvironment(
        user_model=user_model,
        document_sampler=doc_sampler,
        num_candidates=num_candidates,
        slate_size=slate_size,
        resample_documents=True,
    )

    env = recsim_gym.RecSimGymEnv(
        raw_environment=raw_env,
        reward_aggregator=total_watch_time_reward,
    )
    return env


# =========================================================
# 8. Observation helper
# =========================================================
def extract_user_and_docs_from_obs(obs):
    user_obs = np.array(obs["user"], dtype=np.float32)

    doc_dict = obs["doc"]
    sorted_doc_keys = sorted(doc_dict.keys())

    candidate_docs = []
    for k in sorted_doc_keys:
        candidate_docs.append(np.array(doc_dict[k], dtype=np.float32))

    candidate_docs = np.stack(candidate_docs, axis=0)
    return user_obs, candidate_docs, sorted_doc_keys
