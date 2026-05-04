from __future__ import annotations

# Offline embedding-quality evaluation script.
# This script is not the main vector-generation entrypoint.

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from build_vectors import (
    EmbeddingConfig,
    build_embeddings,
    resolve_runtime_path,
    save_embedding_bundle,
    split_interactions_by_time,
)


def score_interactions(df: pd.DataFrame, user_ids, item_ids, user_emb, item_emb):
    user_index = {int(uid): i for i, uid in enumerate(user_ids)}
    item_index = {int(iid): i for i, iid in enumerate(item_ids)}

    work = df[["user_id", "video_id", "watch_ratio"]].copy()
    work["user_row"] = work["user_id"].map(user_index)
    work["item_row"] = work["video_id"].map(item_index)
    work = work.dropna(subset=["user_row", "item_row"]).copy()
    work["user_row"] = work["user_row"].astype(int)
    work["item_row"] = work["item_row"].astype(int)

    user_rows = work["user_row"].values
    item_rows = work["item_row"].values
    scores = np.empty(len(work), dtype=np.float32)
    batch_size = 10000
    for start in range(0, len(work), batch_size):
        end = min(start + batch_size, len(work))
        u = user_emb[user_rows[start:end]]
        v = item_emb[item_rows[start:end]]
        scores[start:end] = np.sum(u * v, axis=1)
    work["score"] = scores
    return work


def fit_linear_calibration(scores: np.ndarray, targets: np.ndarray):
    x = np.stack([scores, np.ones_like(scores)], axis=1)
    coef, *_ = np.linalg.lstsq(x, targets, rcond=None)
    return coef.astype(np.float32)


def apply_linear_calibration(scores: np.ndarray, coef: np.ndarray):
    return coef[0] * scores + coef[1]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def ndcg_at_k_observed(scored_df: pd.DataFrame, k: int = 10, positive_threshold: float = 1.0):
    ndcgs = []
    for _, user_df in scored_df.groupby("user_id", sort=False):
        if len(user_df) < 2:
            continue
        gains = (user_df["watch_ratio"].values >= positive_threshold).astype(np.float32)
        if gains.sum() == 0:
            continue

        order = np.argsort(-user_df["score"].values)
        ranked = gains[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum(ranked * discounts))

        ideal = np.sort(gains)[::-1][:k]
        idcg = float(np.sum(ideal * discounts[: len(ideal)]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)

    if not ndcgs:
        return float("nan")
    return float(np.mean(ndcgs))


def evaluate_bundle(bundle, config: EmbeddingConfig):
    user_ids = bundle["all_user_ids"]
    item_ids = bundle["item_ids"]
    user_emb = bundle["final_user_emb"]
    item_emb = bundle["final_item_emb"]
    eval_split = split_interactions_by_time(
        bundle["interactions"],
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    scored_train = score_interactions(eval_split["train"], user_ids, item_ids, user_emb, item_emb)
    if len(scored_train) > 500000:
        scored_train = scored_train.sample(n=500000, random_state=42)
    coef = fit_linear_calibration(scored_train["score"].values, scored_train["watch_ratio"].values)

    report = {
        "embedding_interaction_usage": bundle["interaction_usage_meta"],
        "evaluation_time_split": eval_split["meta"],
        "calibration": {"a": float(coef[0]), "b": float(coef[1])},
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        scored = score_interactions(eval_split[split_name], user_ids, item_ids, user_emb, item_emb)
        pred = apply_linear_calibration(scored["score"].values, coef)
        scored["pred_watch_ratio"] = pred

        report["splits"][split_name] = {
            "rows": int(len(scored)),
            "mse": mse(scored["watch_ratio"].values, pred),
            "observed_ndcg@10": (
                float("nan") if split_name == "train" else ndcg_at_k_observed(scored, k=10, positive_threshold=1.0)
            ),
        }

    return report


def run_experiment(name: str, config: EmbeddingConfig):
    bundle = build_embeddings(config)

    exp_output_dir = os.path.join(resolve_runtime_path(config.output_dir), "experiments", name)
    os.makedirs(exp_output_dir, exist_ok=True)
    bundle["config"] = replace(config, output_dir=exp_output_dir)
    save_embedding_bundle(bundle)

    report = evaluate_bundle(bundle, config)
    report["experiment"] = name
    report["config"] = config.__dict__

    report_path = os.path.join(exp_output_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path, report


def main():
    base = EmbeddingConfig()
    experiments = {
        "improved_default": base,
        "with_behavior_context": replace(base, include_behavior_context_in_main_item_embedding=True),
        "signed_feedback": replace(base, interaction_weight_mode="signed-feedback"),
    }

    reports = {}
    for name, cfg in experiments.items():
        report_path, report = run_experiment(name, cfg)
        reports[name] = {
            "report_path": report_path,
            "val_mse": report["splits"]["val"]["mse"],
            "test_mse": report["splits"]["test"]["mse"],
            "val_ndcg@10": report["splits"]["val"]["observed_ndcg@10"],
            "test_ndcg@10": report["splits"]["test"]["observed_ndcg@10"],
        }
        print(f"{name}: {json.dumps(reports[name], ensure_ascii=False)}")

    summary_path = os.path.join(resolve_runtime_path(base.output_dir), "experiments", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
