from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INTERACTION_FILE = "small_matrix.csv"
DEFAULT_ITEM_DAILY_FILE = "item_daily_features.csv"
DEFAULT_CHUNK_SIZE = 500_000

INTERACTION_USECOLS = [
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "watch_ratio",
]

INTERACTION_DTYPES = {
    "user_id": "Int64",
    "video_id": "Int64",
    "play_duration": "float64",
    "video_duration": "float64",
    "watch_ratio": "float64",
}


def resolve_default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "KuaiRec" / "KuaiRec 2.0" / "data"


def resolve_default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "vector_outputs"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_video_author_map(item_daily_path: Path) -> pd.DataFrame:
    item_daily = pd.read_csv(item_daily_path, usecols=["video_id", "author_id"])
    item_daily = clean_columns(item_daily)
    item_daily["video_id"] = pd.to_numeric(item_daily["video_id"], errors="coerce")
    item_daily["author_id"] = pd.to_numeric(item_daily["author_id"], errors="coerce")
    item_daily = item_daily.dropna(subset=["video_id", "author_id"]).copy()
    item_daily["video_id"] = item_daily["video_id"].astype(np.int64)
    item_daily["author_id"] = item_daily["author_id"].astype(np.int64)

    author_counts = item_daily.groupby("video_id")["author_id"].nunique()
    conflicting = author_counts[author_counts > 1]
    if not conflicting.empty:
        raise ValueError(
            f"Found {len(conflicting)} video_id values with multiple author_id mappings."
        )

    return item_daily[["video_id", "author_id"]].drop_duplicates("video_id").reset_index(drop=True)


def prepare_interaction_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = clean_columns(chunk)

    for col in INTERACTION_USECOLS:
        if col not in chunk.columns:
            raise ValueError(f"Missing required interaction column: {col}")
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk = chunk.dropna(subset=["user_id", "video_id", "play_duration", "video_duration"]).copy()
    chunk["user_id"] = chunk["user_id"].astype(np.int64)
    chunk["video_id"] = chunk["video_id"].astype(np.int64)
    chunk["play_duration"] = chunk["play_duration"].fillna(0.0).clip(lower=0.0)
    chunk["video_duration"] = chunk["video_duration"].fillna(0.0).clip(lower=0.0)

    duration_floor = chunk["video_duration"].clip(lower=1.0)
    watch_ratio = chunk["watch_ratio"]
    missing_watch = watch_ratio.isna()
    watch_ratio = watch_ratio.where(~missing_watch, chunk["play_duration"] / duration_floor)
    chunk["watch_ratio"] = watch_ratio.fillna(0.0).clip(lower=0.0, upper=2.0)

    chunk["is_effective_watch"] = (chunk["watch_ratio"] >= 0.2).astype(np.float32)
    chunk["is_complete_watch"] = (
        (chunk["video_duration"] > 0.0) & (chunk["play_duration"] >= chunk["video_duration"])
    ).astype(np.float32)
    chunk["is_valid_watch"] = (
        (
            (chunk["video_duration"] > 0.0)
            & (chunk["video_duration"] <= 7000.0)
            & (chunk["play_duration"] >= chunk["video_duration"])
        )
        | ((chunk["video_duration"] > 7000.0) & (chunk["play_duration"] > 7000.0))
    ).astype(np.float32)
    chunk["is_short_watch"] = (
        chunk["play_duration"] < np.minimum(3000.0, duration_floor)
    ).astype(np.float32)
    chunk["is_negative_watch"] = (chunk["watch_ratio"] < 0.2).astype(np.float32)

    chunk["interaction_score"] = (
        chunk["watch_ratio"]
        + 0.4 * chunk["is_effective_watch"]
        + 0.8 * chunk["is_valid_watch"]
        + 0.8 * chunk["is_complete_watch"]
        - 0.8 * chunk["is_short_watch"]
        - 0.6 * chunk["is_negative_watch"]
    ).astype(np.float32)

    chunk["positive_score"] = chunk["interaction_score"].clip(lower=0.0)
    chunk["negative_score_abs"] = (-chunk["interaction_score"].clip(upper=0.0)).astype(np.float32)
    chunk["positive_interaction_count"] = (chunk["interaction_score"] > 0.0).astype(np.int64)
    chunk["negative_interaction_count"] = (chunk["interaction_score"] < 0.0).astype(np.int64)

    return chunk


def aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        chunk.groupby(["user_id", "author_id"], as_index=False)
        .agg(
            creator_preference_score=("interaction_score", "sum"),
            interaction_count=("video_id", "size"),
            positive_interaction_count=("positive_interaction_count", "sum"),
            negative_interaction_count=("negative_interaction_count", "sum"),
            positive_score_sum=("positive_score", "sum"),
            negative_score_abs_sum=("negative_score_abs", "sum"),
            watch_ratio_sum=("watch_ratio", "sum"),
            effective_watch_count=("is_effective_watch", "sum"),
            valid_watch_count=("is_valid_watch", "sum"),
            complete_watch_count=("is_complete_watch", "sum"),
            short_watch_count=("is_short_watch", "sum"),
            negative_watch_count=("is_negative_watch", "sum"),
        )
    )

    int_cols = [
        "interaction_count",
        "positive_interaction_count",
        "negative_interaction_count",
        "effective_watch_count",
        "valid_watch_count",
        "complete_watch_count",
        "short_watch_count",
        "negative_watch_count",
    ]
    for col in int_cols:
        grouped[col] = grouped[col].astype(np.int64)

    return grouped


def combine_partial_aggregates(partials: list[pd.DataFrame]) -> pd.DataFrame:
    observed = pd.concat(partials, ignore_index=True)
    observed = (
        observed.groupby(["user_id", "author_id"], as_index=False)
        .agg(
            creator_preference_score=("creator_preference_score", "sum"),
            interaction_count=("interaction_count", "sum"),
            positive_interaction_count=("positive_interaction_count", "sum"),
            negative_interaction_count=("negative_interaction_count", "sum"),
            positive_score_sum=("positive_score_sum", "sum"),
            negative_score_abs_sum=("negative_score_abs_sum", "sum"),
            watch_ratio_sum=("watch_ratio_sum", "sum"),
            effective_watch_count=("effective_watch_count", "sum"),
            valid_watch_count=("valid_watch_count", "sum"),
            complete_watch_count=("complete_watch_count", "sum"),
            short_watch_count=("short_watch_count", "sum"),
            negative_watch_count=("negative_watch_count", "sum"),
        )
    )

    observed["avg_watch_ratio"] = (
        observed["watch_ratio_sum"] / observed["interaction_count"].clip(lower=1)
    ).astype(np.float32)
    observed["avg_preference_per_interaction"] = (
        observed["creator_preference_score"] / observed["interaction_count"].clip(lower=1)
    ).astype(np.float32)
    observed["creator_preference_score"] = observed["creator_preference_score"].astype(np.float32)
    observed["positive_score_sum"] = observed["positive_score_sum"].astype(np.float32)
    observed["negative_score_abs_sum"] = observed["negative_score_abs_sum"].astype(np.float32)
    observed["watch_ratio_sum"] = observed["watch_ratio_sum"].astype(np.float32)

    observed = observed.sort_values(
        ["user_id", "creator_preference_score", "author_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return observed


def build_dense_table(
    observed: pd.DataFrame,
    user_ids: list[int],
    author_ids: list[int],
) -> pd.DataFrame:
    dense_index = pd.MultiIndex.from_product(
        [user_ids, author_ids],
        names=["user_id", "author_id"],
    ).to_frame(index=False)
    dense = dense_index.merge(
        observed[["user_id", "author_id", "creator_preference_score"]],
        on=["user_id", "author_id"],
        how="left",
    )
    dense["creator_preference_score"] = dense["creator_preference_score"].fillna(0.0).astype(np.float32)
    return dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a user-author preference table from KuaiRec small_matrix interactions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_default_data_dir(),
        help="Directory that contains small_matrix.csv and item_daily_features.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_default_output_dir(),
        help="Directory where output tables will be written.",
    )
    parser.add_argument(
        "--interaction-file",
        default=DEFAULT_INTERACTION_FILE,
        help="Interaction CSV filename inside --data-dir.",
    )
    parser.add_argument(
        "--item-daily-file",
        default=DEFAULT_ITEM_DAILY_FILE,
        help="Item daily CSV filename inside --data-dir.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for reading small_matrix.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    interaction_path = data_dir / args.interaction_file
    item_daily_path = data_dir / args.item_daily_file

    if not interaction_path.exists():
        raise FileNotFoundError(f"Missing interaction file: {interaction_path}")
    if not item_daily_path.exists():
        raise FileNotFoundError(f"Missing item daily file: {item_daily_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    observed_output_path = output_dir / "user_author_preference_observed.csv"
    dense_output_path = output_dir / "user_author_preference_dense.csv"
    meta_output_path = output_dir / "user_author_preference_meta.json"

    video_author_map = load_video_author_map(item_daily_path)
    partials: list[pd.DataFrame] = []
    all_user_ids: set[int] = set()
    all_author_ids: set[int] = set()
    raw_rows = 0
    mapped_rows = 0

    chunk_iter = pd.read_csv(
        interaction_path,
        usecols=INTERACTION_USECOLS,
        dtype=INTERACTION_DTYPES,
        chunksize=args.chunksize,
    )

    for chunk_idx, raw_chunk in enumerate(chunk_iter, start=1):
        raw_rows += len(raw_chunk)
        chunk = prepare_interaction_chunk(raw_chunk)
        chunk = chunk.merge(video_author_map, on="video_id", how="left", validate="many_to_one")
        chunk = chunk.dropna(subset=["author_id"]).copy()
        chunk["author_id"] = chunk["author_id"].astype(np.int64)
        mapped_rows += len(chunk)

        if chunk.empty:
            print(f"chunk {chunk_idx}: no rows after author mapping")
            continue

        partials.append(aggregate_chunk(chunk))
        all_user_ids.update(map(int, chunk["user_id"].unique()))
        all_author_ids.update(map(int, chunk["author_id"].unique()))
        print(
            f"chunk {chunk_idx}: raw_rows={len(raw_chunk)} mapped_rows={len(chunk)} "
            f"observed_pairs={partials[-1].shape[0]}"
        )

    if not partials:
        raise ValueError("No valid interaction rows were available for aggregation.")

    user_ids = sorted(all_user_ids)
    author_ids = sorted(all_author_ids)
    observed = combine_partial_aggregates(partials)
    dense = build_dense_table(observed=observed, user_ids=user_ids, author_ids=author_ids)

    observed.to_csv(observed_output_path, index=False)
    dense.to_csv(dense_output_path, index=False)

    meta = {
        "data_sources": {
            "interaction_path": str(interaction_path),
            "item_daily_path": str(item_daily_path),
        },
        "outputs": {
            "observed_table": str(observed_output_path),
            "dense_table": str(dense_output_path),
        },
        "counts": {
            "raw_interaction_rows": int(raw_rows),
            "mapped_interaction_rows": int(mapped_rows),
            "users": int(len(user_ids)),
            "authors_in_small_matrix": int(len(author_ids)),
            "observed_user_author_pairs": int(len(observed)),
            "dense_user_author_pairs": int(len(dense)),
        },
        "scoring_rule": {
            "baseline_for_unseen_author": 0.0,
            "formula": (
                "interaction_score = clip(watch_ratio, 0, 2)"
                " + 0.4 * is_effective_watch"
                " + 0.8 * is_valid_watch"
                " + 0.8 * is_complete_watch"
                " - 0.8 * is_short_watch"
                " - 0.6 * is_negative_watch"
            ),
            "is_effective_watch": "watch_ratio >= 0.2",
            "is_valid_watch": (
                "(video_duration <= 7000 and play_duration >= video_duration)"
                " or (video_duration > 7000 and play_duration > 7000)"
            ),
            "is_complete_watch": "play_duration >= video_duration and video_duration > 0",
            "is_short_watch": "play_duration < min(3000, max(video_duration, 1))",
            "is_negative_watch": "watch_ratio < 0.2",
            "note": (
                "small_matrix.csv does not contain per-user dislike/report signals, "
                "so negative preference here is inferred from weak watch behavior only."
            ),
        },
    }

    with meta_output_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"observed table written to: {observed_output_path}")
    print(f"dense table written to: {dense_output_path}")
    print(f"meta written to: {meta_output_path}")
    print(
        "summary:",
        json.dumps(meta["counts"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
