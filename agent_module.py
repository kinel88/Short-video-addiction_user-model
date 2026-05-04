# agent_module.py

import math
import numpy as np
from env_module import extract_user_and_docs_from_obs


class LinearAgent:
    """
    推荐 agent。
    现在的候选打分核心特征改为：
    - preference dot product
    - preference match (sigmoid(dot))
    - dopamine layout: fatigue / dopamine / expected_reward / session_peak
    - no_dopamine layout: fatigue
    """

    def __init__(self, slate_size=1, lr=0.05, embedding_dim=32, observation_layout="dopamine"):
        self.slate_size = slate_size
        self.lr = lr
        self.embedding_dim = int(embedding_dim)
        self.dot_scale = math.sqrt(float(self.embedding_dim))
        self.observation_layout = str(observation_layout).strip().lower()
        if self.observation_layout not in {"dopamine", "no_dopamine"}:
            raise ValueError(
                "observation_layout must be either 'dopamine' or 'no_dopamine'"
            )

        if self.observation_layout == "dopamine":
            # [fatigue, dopamine, expected_reward, session_peak, pref_dot, pref_match]
            self.feature_dim = 6
        else:
            # [fatigue, pref_dot, pref_match]
            self.feature_dim = 3
        self.w = np.zeros(self.feature_dim, dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def build_features(self, obs):
        user_obs, candidate_docs, _ = extract_user_and_docs_from_obs(obs)

        if self.observation_layout == "dopamine":
            expected_user_dim = 4 + self.embedding_dim
            if len(user_obs) != expected_user_dim:
                raise ValueError(
                    f"dopamine observation must have length {expected_user_dim}, got {len(user_obs)}"
                )

            fatigue = float(user_obs[0])
            dopamine = float(user_obs[1])
            expected_reward = float(user_obs[2])
            session_peak = float(user_obs[3])
            user_vec = np.asarray(user_obs[4:], dtype=np.float32)
        else:
            expected_user_dim = 1 + self.embedding_dim
            if len(user_obs) != expected_user_dim:
                raise ValueError(
                    f"no_dopamine observation must have length {expected_user_dim}, got {len(user_obs)}"
                )

            fatigue = float(user_obs[0])
            user_vec = np.asarray(user_obs[1:], dtype=np.float32)

        feature_list = []
        for doc_vec in candidate_docs:
            doc_vec = np.asarray(doc_vec, dtype=np.float32)

            pref_dot = float(np.dot(user_vec, doc_vec) / self.dot_scale)
            pref_match = float(self._sigmoid(pref_dot))

            if self.observation_layout == "dopamine":
                x = np.array([
                    fatigue,
                    dopamine,
                    expected_reward,
                    session_peak,
                    pref_dot,
                    pref_match,
                ], dtype=np.float32)
            else:
                x = np.array([
                    fatigue,
                    pref_dot,
                    pref_match,
                ], dtype=np.float32)

            feature_list.append(x)

        return np.stack(feature_list, axis=0)

    def score_candidates(self, obs):
        features = self.build_features(obs)
        scores = features @ self.w
        return scores, features

    def act(self, obs, epsilon=0.1):
        scores, _ = self.score_candidates(obs)
        num_candidates = len(scores)

        if np.random.rand() < epsilon:
            chosen = np.random.choice(num_candidates, size=self.slate_size, replace=False)
        else:
            chosen = np.argsort(scores)[-self.slate_size:][::-1]

        return chosen.tolist()

    def update(self, obs, action, reward):
        features = self.build_features(obs)
        selected_features = features[action]
        mean_feature = selected_features.mean(axis=0)
        self.w += self.lr * reward * mean_feature

    def __repr__(self):
        return f"LinearAgent(w={self.w})"
