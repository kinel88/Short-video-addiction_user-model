from __future__ import annotations

import ast
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from caption_text_pipeline import build_caption_text_feature_bundle


# =========================================================
# 0) CONFIG
# =========================================================
DATA_DIR = r"C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec\KuaiRec 2.0\data"
OUTPUT_DIR = r"C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec"

ITEM_DAILY_FILE = "item_daily_features.csv"
ITEM_CATEGORY_FILE = "item_categories.csv"
USER_FEATURE_FILE = "user_features.csv"
USER_FEATURE_RAW_FILE = "user_features_raw.csv"
CAPTION_CATEGORY_FILE = "kuairec_caption_category.csv"

DEFAULT_INTERACTION_FILE = "big_matrix.csv"
USE_FULL_INTERACTIONS_FOR_EMBEDDINGS = True
TEXT_DIM = 64
LATENT_DIM = 64
WATCH_RATIO_CLIP = 5.0
ITEM_CONTEXT_WINDOW_DAYS = 30
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# =========================================================
# 1) DTYPE / PARSING RULES
# =========================================================
SMALL_MATRIX_DTYPES = {
    "user_id": "Int64",
    "video_id": "Int64",
    "play_duration": "float64",
    "video_duration": "float64",
    "time": "string",
    "date": "string",
    "timestamp": "float64",
    "watch_ratio": "float64",
}

ITEM_CATEGORIES_DTYPES = {
    "video_id": "Int64",
    "feat": "string",
}

USER_FEATURE_STRING_COLS = [
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
]

USER_ONEHOT_COLS = [f"onehot_feat{i}" for i in range(18)]

USER_RAW_FORCE_STRING_COLS = [
    "user_active_degree", "gender", "age_range",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
    "phone_brand", "phone_model",
    "fre_country", "fre_country_region", "fre_province", "fre_city",
    "fre_city_level", "fre_community_type",
    "register_days_range",
    "platform", "os_version", "app_download_channel",
    "app_version", "app_major_version", "app_minor_version", "isp",
]

ITEM_DAILY_STRING_COLS = [
    "video_type", "upload_dt", "upload_type", "visible_status", "video_tag_name",
]

ITEM_DAILY_FLOAT_COLS = [
    "video_duration",
    "play_progress",
]

CAPTION_STRING_COLS = [
    "manual_cover_text", "caption", "topic_tag",
    "first_level_category_name", "second_level_category_name", "third_level_category_name",
]


@dataclass
class EmbeddingConfig:
    data_dir: str = DATA_DIR
    output_dir: str = OUTPUT_DIR
    interaction_file: str = DEFAULT_INTERACTION_FILE
    use_full_interactions_for_embeddings: bool = USE_FULL_INTERACTIONS_FOR_EMBEDDINGS
    text_dim: int = TEXT_DIM
    latent_dim: int = LATENT_DIM
    watch_ratio_clip: float = WATCH_RATIO_CLIP
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    item_context_window_days: int = ITEM_CONTEXT_WINDOW_DAYS
    interaction_weight_mode: str = "watch-centered"
    include_behavior_context_in_main_item_embedding: bool = False


# =========================================================
# 2) BASIC UTILS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(resolve_runtime_path(path), exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_divide(a, b):
    return np.where(b == 0, 0, a / b)


def safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(np.float64)
    return pd.Series(np.full(len(df), default, dtype=np.float64), index=df.index)


def standardize_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
    return (arr - mean) / std


def percentile_rank_array(values) -> np.ndarray:
    arr = pd.Series(np.asarray(values, dtype=np.float64))
    if arr.empty:
        return np.zeros(0, dtype=np.float64)
    return arr.rank(method="average", pct=True).fillna(0.0).values.astype(np.float64)


def parse_feat_list(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    return []


def to_string_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("UNK")
    return df


def to_float_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def zscore_block(df: pd.DataFrame):
    if df.shape[1] == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    x = df.fillna(0.0).astype(np.float32).values
    scaler = StandardScaler()
    return scaler.fit_transform(x).astype(np.float32)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (x / norms).astype(np.float32)


def select_existing_columns(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def resolve_runtime_path(path: str) -> str:
    if os.name == "posix" and re.match(r"^[A-Za-z]:\\", path):
        drive = path[0].lower()
        tail = path[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{tail}"
    return path


# =========================================================
# 3) LOAD / SPLIT INTERACTIONS
# =========================================================
def load_interactions(config: EmbeddingConfig):
    path = os.path.join(resolve_runtime_path(config.data_dir), config.interaction_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing interaction file: {path}")

    df = pd.read_csv(path, dtype=SMALL_MATRIX_DTYPES)
    df = clean_columns(df)

    required = [
        "user_id", "video_id", "play_duration", "video_duration",
        "time", "date", "timestamp", "watch_ratio",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in interactions: {missing}")

    for c in ["user_id", "video_id", "play_duration", "video_duration", "timestamp", "watch_ratio", "date"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["user_id", "video_id", "play_duration", "video_duration", "timestamp"]).copy()

    df["user_id"] = df["user_id"].astype(np.int64)
    df["video_id"] = df["video_id"].astype(np.int64)
    df["play_duration"] = df["play_duration"].fillna(0.0).astype(np.float64)
    df["video_duration"] = df["video_duration"].fillna(0.0).astype(np.float64)
    df["watch_ratio"] = df["watch_ratio"].fillna(0.0).clip(0.0, config.watch_ratio_clip).astype(np.float64)
    df["date_int"] = df["date"].round().astype("Int64")
    df["interaction_time"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    df["is_valid_watch"] = (
        ((df["video_duration"] <= 7000) & (df["play_duration"] >= df["video_duration"])) |
        ((df["video_duration"] > 7000) & (df["play_duration"] > 7000))
    ).astype(np.float32)
    df["is_complete_watch"] = (df["play_duration"] >= df["video_duration"]).astype(np.float32)
    df["is_short_watch"] = (df["play_duration"] < np.minimum(3000, df["video_duration"])).astype(np.float32)

    return df, os.path.basename(path)


def split_interactions_by_time(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(df)
    if n < 3:
        raise ValueError("Not enough interactions to create time split.")

    df = df.sort_values(["timestamp", "user_id", "video_id"]).reset_index(drop=True)
    train_end_idx = max(int(n * train_ratio) - 1, 0)
    val_end_idx = max(int(n * (train_ratio + val_ratio)) - 1, train_end_idx)

    train_end_ts = float(df.iloc[train_end_idx]["timestamp"])
    val_end_ts = float(df.iloc[val_end_idx]["timestamp"])

    train_df = df.loc[df["timestamp"] <= train_end_ts].copy()
    val_df = df.loc[(df["timestamp"] > train_end_ts) & (df["timestamp"] <= val_end_ts)].copy()
    test_df = df.loc[df["timestamp"] > val_end_ts].copy()

    train_end_date = int(train_df["date_int"].dropna().max())
    val_end_date = int(val_df["date_int"].dropna().max()) if not val_df.empty else train_end_date

    split_meta = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_end_ts": train_end_ts,
        "val_end_ts": val_end_ts,
        "train_end_date": train_end_date,
        "val_end_date": val_end_date,
    }
    return {"train": train_df, "val": val_df, "test": test_df, "meta": split_meta}


def get_latest_interaction_date(df: pd.DataFrame) -> int:
    if "date_int" not in df.columns:
        raise ValueError("Interactions must contain date_int before computing the reference end date.")
    valid_dates = df["date_int"].dropna()
    if valid_dates.empty:
        raise ValueError("No valid interaction dates are available to define the reference end date.")
    return int(valid_dates.max())


def compute_interaction_weights(df: pd.DataFrame, mode: str = "watch-centered") -> np.ndarray:
    watch = df["watch_ratio"].values.astype(np.float32)
    valid = df["is_valid_watch"].values.astype(np.float32)
    complete = df["is_complete_watch"].values.astype(np.float32)
    short = df["is_short_watch"].values.astype(np.float32)

    if mode == "watch-centered":
        # Treat the first 0.1 watch ratio as neutral exposure rather than positive preference.
        return (np.clip(watch, 0.0, 5.0) - 0.1).astype(np.float32)

    if mode == "positive-strength":
        weights = 0.2 + 1.0 * np.clip(watch, 0.0, 2.0) + 0.8 * valid + 0.8 * complete - 0.3 * short
        return np.clip(weights.astype(np.float32), 0.01, None)

    if mode == "signed-feedback":
        clipped_watch = np.clip(watch, 0.0, 2.0)
        positive = 0.2 + 1.0 * clipped_watch + 0.8 * valid + 0.8 * complete
        negative = 0.8 * short + 0.6 * ((clipped_watch < 0.1).astype(np.float32))
        weights = positive - negative
        tiny = np.abs(weights) < 1e-3
        weights[tiny] = 1e-3
        return weights.astype(np.float32)

    raise ValueError(f"Unsupported interaction weight mode: {mode}")


def describe_interaction_weight_formula(mode: str) -> str:
    formulas = {
        "watch-centered": "clip(watch_ratio, 0.0, 5.0) - 0.1",
        "positive-strength": "clip(0.2 + clip(watch_ratio, 0.0, 2.0) + 0.8*is_valid_watch + 0.8*is_complete_watch - 0.3*is_short_watch, 0.01, +inf)",
        "signed-feedback": "(0.2 + clip(watch_ratio, 0.0, 2.0) + 0.8*is_valid_watch + 0.8*is_complete_watch) - (0.8*is_short_watch + 0.6*1[clip(watch_ratio, 0.0, 2.0) < 0.1])",
    }
    if mode not in formulas:
        raise ValueError(f"Unsupported interaction weight mode: {mode}")
    return formulas[mode]


# =========================================================
# 4) ITEM FEATURE BUILDERS
# =========================================================
def load_item_categories(data_dir: str):
    path = os.path.join(resolve_runtime_path(data_dir), ITEM_CATEGORY_FILE)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, dtype=ITEM_CATEGORIES_DTYPES)
    df = clean_columns(df)
    if "video_id" not in df.columns or "feat" not in df.columns:
        raise ValueError("item_categories.csv must contain video_id and feat")

    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce")
    df = df.dropna(subset=["video_id"]).copy()
    df["video_id"] = df["video_id"].astype(np.int64)
    df["feat_list"] = df["feat"].apply(parse_feat_list)
    return df


def build_item_content_features(data_dir: str, text_dim: int):
    item_daily_path = os.path.join(resolve_runtime_path(data_dir), ITEM_DAILY_FILE)
    if not os.path.exists(item_daily_path):
        raise FileNotFoundError(f"Missing {ITEM_DAILY_FILE}")

    item_daily = pd.read_csv(item_daily_path)
    item_daily = clean_columns(item_daily)
    item_daily["video_id"] = pd.to_numeric(item_daily["video_id"], errors="coerce")
    item_daily = item_daily.dropna(subset=["video_id"]).copy()
    item_daily["video_id"] = item_daily["video_id"].astype(np.int64)
    # Main content embeddings intentionally exclude all item_daily static metadata.
    # We only keep video ids here as the merge anchor for category/text features.
    item_base = item_daily[["video_id"]].drop_duplicates().copy()

    item_cat_feature = None
    item_cat_df = load_item_categories(data_dir)
    if item_cat_df is not None:
        mlb = MultiLabelBinarizer(sparse_output=False)
        cat_mat = mlb.fit_transform(item_cat_df["feat_list"])
        cat_cols = [f"cat_{c}" for c in mlb.classes_]
        item_cat_feature = pd.concat(
            [
                item_cat_df[["video_id"]].reset_index(drop=True),
                pd.DataFrame(cat_mat, columns=cat_cols),
            ],
            axis=1,
        )

    text_feature = None
    segmented_caption_df = pd.DataFrame()
    caption_text_meta = {
        "caption_csv": os.path.join(resolve_runtime_path(data_dir), CAPTION_CATEGORY_FILE),
        "valid_video_rows": 0,
        "text_feature_dim": 0,
        "available": False,
    }
    caption_path = os.path.join(resolve_runtime_path(data_dir), CAPTION_CATEGORY_FILE)
    if os.path.exists(caption_path):
        try:
            text_feature, segmented_caption_df, caption_text_meta = (
                build_caption_text_feature_bundle(
                    caption_path=caption_path,
                    text_dim=text_dim,
                )
            )
        except Exception as e:
            warnings.warn(f"Skipping caption/category file due to parse error: {e}")
            text_feature = None
            segmented_caption_df = pd.DataFrame()
            caption_text_meta = {
                "caption_csv": caption_path,
                "valid_video_rows": 0,
                "text_feature_dim": 0,
                "available": False,
                "error": str(e),
            }

    item_df = item_base.copy()
    if item_cat_feature is not None:
        item_df = item_df.merge(item_cat_feature, on="video_id", how="outer")
    if text_feature is not None:
        item_df = item_df.merge(text_feature, on="video_id", how="outer")
    return item_df, segmented_caption_df, caption_text_meta


def build_item_behavior_context(data_dir: str, reference_end_date: int, window_days: int):
    item_daily_path = os.path.join(resolve_runtime_path(data_dir), ITEM_DAILY_FILE)
    if not os.path.exists(item_daily_path):
        raise FileNotFoundError(f"Missing {ITEM_DAILY_FILE}")

    item_daily = pd.read_csv(item_daily_path)
    item_daily = clean_columns(item_daily)
    item_daily["video_id"] = pd.to_numeric(item_daily["video_id"], errors="coerce")
    item_daily["date"] = pd.to_numeric(item_daily["date"], errors="coerce")
    item_daily = item_daily.dropna(subset=["video_id", "date"]).copy()
    item_daily["video_id"] = item_daily["video_id"].astype(np.int64)
    item_daily["date_int"] = item_daily["date"].round().astype(np.int64)
    item_daily["date_dt"] = pd.to_datetime(item_daily["date_int"].astype(str), format="%Y%m%d", errors="coerce")

    reference_end_dt = pd.to_datetime(str(reference_end_date), format="%Y%m%d")
    min_dt = reference_end_dt - pd.Timedelta(days=max(window_days - 1, 0))
    item_daily = item_daily[(item_daily["date_dt"] <= reference_end_dt) & (item_daily["date_dt"] >= min_dt)].copy()
    if item_daily.empty:
        raise ValueError("No item_daily_features rows remain after applying the time window.")

    behavior_candidates = [
        "show_cnt", "show_user_num",
        "play_cnt", "play_user_num", "play_duration",
        "complete_play_cnt", "complete_play_user_num",
        "valid_play_cnt", "valid_play_user_num",
        "long_time_play_cnt", "long_time_play_user_num",
        "short_time_play_cnt", "short_time_play_user_num",
        "play_progress", "comment_stay_duration",
        "like_cnt", "like_user_num", "click_like_cnt", "double_click_cnt",
        "cancel_like_cnt", "cancel_like_user_num",
        "comment_cnt", "comment_user_num",
        "direct_comment_cnt", "reply_comment_cnt",
        "delete_comment_cnt", "delete_comment_user_num",
        "comment_like_cnt", "comment_like_user_num",
        "follow_cnt", "follow_user_num",
        "cancel_follow_cnt", "cancel_follow_user_num",
        "share_cnt", "share_user_num",
        "download_cnt", "download_user_num",
        "report_cnt", "report_user_num",
        "reduce_similar_cnt", "reduce_similar_user_num",
        "collect_cnt", "collect_user_num",
        "cancel_collect_cnt", "cancel_collect_user_num",
    ]
    behavior_cols = select_existing_columns(item_daily, behavior_candidates)
    for c in behavior_cols:
        item_daily[c] = pd.to_numeric(item_daily[c], errors="coerce")

    grouped = item_daily.groupby("video_id")[behavior_cols]
    item_behavior_mean = grouped.mean().add_suffix("_mean")
    item_behavior_std = grouped.std().fillna(0).add_suffix("_std")
    item_behavior_max = grouped.max().add_suffix("_max")
    item_behavior = pd.concat([item_behavior_mean, item_behavior_std, item_behavior_max], axis=1).reset_index()

    ratio_specs = [
        ("play_cnt_mean", "show_cnt_mean", "play_per_show"),
        ("valid_play_cnt_mean", "show_cnt_mean", "valid_play_per_show"),
        ("complete_play_cnt_mean", "show_cnt_mean", "complete_play_per_show"),
        ("valid_play_cnt_mean", "play_cnt_mean", "valid_play_per_play"),
        ("complete_play_cnt_mean", "play_cnt_mean", "complete_play_per_play"),
        ("like_cnt_mean", "show_cnt_mean", "like_per_show"),
        ("comment_cnt_mean", "show_cnt_mean", "comment_per_show"),
        ("share_cnt_mean", "show_cnt_mean", "share_per_show"),
        ("follow_cnt_mean", "show_cnt_mean", "follow_per_show"),
        ("collect_cnt_mean", "show_cnt_mean", "collect_per_show"),
        ("report_cnt_mean", "show_cnt_mean", "report_per_show"),
        ("reduce_similar_cnt_mean", "show_cnt_mean", "reduce_similar_per_show"),
    ]
    for num_col, den_col, new_col in ratio_specs:
        if num_col in item_behavior.columns and den_col in item_behavior.columns:
            den = item_behavior[den_col].replace(0, np.nan).values
            item_behavior[new_col] = np.nan_to_num(
            item_behavior[num_col].values / den,
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    show_cnt_mean = safe_numeric_series(item_behavior, "show_cnt_mean")
    play_per_show = safe_numeric_series(item_behavior, "play_per_show")
    play_progress_mean = safe_numeric_series(item_behavior, "play_progress_mean").clip(0.0, 1.0)
    complete_play_per_play = safe_numeric_series(item_behavior, "complete_play_per_play").clip(0.0, 1.0)
    valid_play_per_play = safe_numeric_series(item_behavior, "valid_play_per_play").clip(0.0, 1.0)

    like_per_show = safe_numeric_series(item_behavior, "like_per_show")
    comment_per_show = safe_numeric_series(item_behavior, "comment_per_show")
    share_per_show = safe_numeric_series(item_behavior, "share_per_show")
    follow_per_show = safe_numeric_series(item_behavior, "follow_per_show")
    collect_per_show = safe_numeric_series(item_behavior, "collect_per_show")

    report_per_show = safe_numeric_series(item_behavior, "report_per_show")
    reduce_similar_per_show = safe_numeric_series(item_behavior, "reduce_similar_per_show")
    cancel_follow_per_show = pd.Series(
        safe_divide(
            safe_numeric_series(item_behavior, "cancel_follow_cnt_mean").values,
            show_cnt_mean.replace(0.0, np.nan).values,
        ),
        index=item_behavior.index,
    ).fillna(0.0)

    completion_signal = (
        0.50 * complete_play_per_play.values +
        0.25 * valid_play_per_play.values +
        0.15 * play_progress_mean.values +
        0.10 * play_per_show.values
    )
    positive_interaction_signal = (
        0.35 * follow_per_show.values +
        0.30 * collect_per_show.values +
        0.25 * share_per_show.values +
        0.10 * comment_per_show.values
    )
    like_signal = like_per_show.values
    negative_signal = (
        0.60 * report_per_show.values +
        0.20 * reduce_similar_per_show.values +
        0.20 * cancel_follow_per_show.values
    )

    quality_confidence = percentile_rank_array(np.log1p(np.clip(show_cnt_mean.values, a_min=0.0, a_max=None)))
    quality_index_raw = (
        0.50 * standardize_array(completion_signal) +
        2.00 * standardize_array(positive_interaction_signal) +
        1.50 * standardize_array(like_signal) -
        5.00 * standardize_array(negative_signal)
    )
    quality_index_raw = quality_index_raw * (0.35 + 0.65 * quality_confidence)
    quality_index = percentile_rank_array(quality_index_raw) * 100.0

    item_behavior["quality_completion_signal"] = completion_signal.astype(np.float32)
    item_behavior["quality_positive_interaction_signal"] = positive_interaction_signal.astype(np.float32)
    item_behavior["quality_like_signal"] = like_signal.astype(np.float32)
    item_behavior["quality_negative_signal"] = negative_signal.astype(np.float32)
    item_behavior["quality_confidence"] = quality_confidence.astype(np.float32)
    item_behavior["quality_index_raw"] = quality_index_raw.astype(np.float32)
    item_behavior["quality_index"] = quality_index.astype(np.float32)

    context_meta = {
        "reference_end_date": int(reference_end_date),
        "window_days": int(window_days),
        "rows_used": int(len(item_daily)),
        "distinct_items": int(item_behavior["video_id"].nunique()),
        "quality_index_column": "quality_index",
        "quality_index_range": [0.0, 100.0],
        "quality_index_definition": {
            "completion_signal": {
                "complete_play_per_play": 0.50,
                "valid_play_per_play": 0.25,
                "play_progress_mean": 0.15,
                "play_per_show": 0.10,
            },
            "positive_interaction_signal": {
                "follow_per_show": 0.35,
                "collect_per_show": 0.30,
                "share_per_show": 0.25,
                "comment_per_show": 0.10,
            },
            "quality_raw_weights": {
                "completion_signal": 0.50,
                "positive_interaction_signal": 2.0,
                "like_per_show": 1.5,
                "negative_signal": -5.0,
            },
            "negative_signal": {
                "report_per_show": 0.60,
                "reduce_similar_per_show": 0.20,
                "cancel_follow_per_show": 0.20,
            },
            "note": "Exposure is only used in quality_confidence, not as a direct quality signal.",
            "confidence": "percentile_rank(log1p(show_cnt_mean))",
            "final_index": "percentile_rank(quality_index_raw) * 100",
        },
    }
    return item_behavior, context_meta


# =========================================================
# 5) USER FEATURE BUILDERS
# =========================================================
def build_user_static_features(data_dir: str):
    runtime_data_dir = resolve_runtime_path(data_dir)
    raw_path = os.path.join(runtime_data_dir, USER_FEATURE_RAW_FILE)
    base_path = os.path.join(runtime_data_dir, USER_FEATURE_FILE)

    if os.path.exists(raw_path):
        user_df = pd.read_csv(raw_path)
        user_df = clean_columns(user_df)
        user_df["user_id"] = pd.to_numeric(user_df["user_id"], errors="coerce")
        user_df = user_df.dropna(subset=["user_id"]).copy()
        user_df["user_id"] = user_df["user_id"].astype(np.int64)
        user_df = to_string_cols(user_df, USER_RAW_FORCE_STRING_COLS)
        return user_df

    if os.path.exists(base_path):
        user_df = pd.read_csv(base_path)
        user_df = clean_columns(user_df)
        user_df["user_id"] = pd.to_numeric(user_df["user_id"], errors="coerce")
        user_df = user_df.dropna(subset=["user_id"]).copy()
        user_df["user_id"] = user_df["user_id"].astype(np.int64)
        user_df = to_string_cols(user_df, USER_FEATURE_STRING_COLS)
        for c in USER_ONEHOT_COLS:
            if c in user_df.columns:
                user_df[c] = user_df[c].astype("string").fillna("UNK")
        return user_df

    raise FileNotFoundError("Missing user_features.csv and user_features_raw.csv")


def build_user_behavior_summary(interactions_train: pd.DataFrame, item_category_df=None):
    user_hist = interactions_train.groupby("user_id").agg(
        total_interactions=("video_id", "count"),
        unique_items=("video_id", "nunique"),
        avg_play_duration=("play_duration", "mean"),
        avg_video_duration=("video_duration", "mean"),
        avg_watch_ratio=("watch_ratio", "mean"),
        std_watch_ratio=("watch_ratio", "std"),
        valid_watch_rate=("is_valid_watch", "mean"),
        complete_watch_rate=("is_complete_watch", "mean"),
        short_watch_rate=("is_short_watch", "mean"),
    ).reset_index()
    user_hist["std_watch_ratio"] = user_hist["std_watch_ratio"].fillna(0.0)

    user_cat_pref = None
    if item_category_df is not None and "feat_list" in item_category_df.columns:
        mlb = MultiLabelBinarizer(sparse_output=False)
        item_cat_mat = mlb.fit_transform(item_category_df["feat_list"])
        item_cat_cols = [f"ucat_{c}" for c in mlb.classes_]
        item_cat_binary = pd.concat(
            [
                item_category_df[["video_id"]].reset_index(drop=True),
                pd.DataFrame(item_cat_mat, columns=item_cat_cols),
            ],
            axis=1,
        )

        inter_cat = interactions_train[["user_id", "video_id", "watch_ratio"]].merge(
            item_cat_binary, on="video_id", how="left"
        )
        for c in item_cat_cols:
            inter_cat[c] = inter_cat[c].fillna(0.0) * inter_cat["watch_ratio"].fillna(0.0)
        user_cat_pref = inter_cat.groupby("user_id")[item_cat_cols].mean().reset_index()

    out = user_hist.copy()
    if user_cat_pref is not None:
        out = out.merge(user_cat_pref, on="user_id", how="left")
    return out


# =========================================================
# 6) MATRIX ENCODERS
# =========================================================
def dataframe_to_numeric_matrix(df: pd.DataFrame, id_col: str):
    df = df.copy()
    ids = df[id_col].values
    df = df.drop(columns=[id_col])

    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(np.int8)

    obj_cols = []
    num_cols = []
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            obj_cols.append(c)
            df[c] = df[c].astype("string").fillna("UNK")
        else:
            num_cols.append(c)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_block = zscore_block(df[num_cols]) if num_cols else np.zeros((len(df), 0), dtype=np.float32)
    if obj_cols:
        obj_block = pd.get_dummies(df[obj_cols], dummy_na=False).astype(np.float32).values
    else:
        obj_block = np.zeros((len(df), 0), dtype=np.float32)

    x = np.concatenate([num_block, obj_block], axis=1).astype(np.float32)
    return ids, x


# =========================================================
# 7) SHARED LATENT SPACE
# =========================================================
def build_shared_latent_embeddings(
    interactions_train: pd.DataFrame,
    all_user_ids,
    all_item_ids,
    latent_dim: int,
    weight_mode: str,
):
    user_index = {uid: i for i, uid in enumerate(all_user_ids)}
    item_index = {iid: i for i, iid in enumerate(all_item_ids)}

    weights = compute_interaction_weights(interactions_train, mode=weight_mode)
    rows = interactions_train["user_id"].map(user_index).values
    cols = interactions_train["video_id"].map(item_index).values

    r = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(len(all_user_ids), len(all_item_ids)),
        dtype=np.float32,
    ).tocsr()

    min_dim = min(r.shape)
    if min_dim <= 1:
        raise ValueError("Interaction matrix too small to build latent embeddings.")

    k = min(latent_dim, min_dim - 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    user_proj = svd.fit_transform(r).astype(np.float32)
    singular_values = np.clip(svd.singular_values_.astype(np.float32), 1e-8, None)
    sqrt_s = np.sqrt(singular_values)

    user_latent = user_proj / sqrt_s
    item_latent = svd.components_.T * sqrt_s
    latent_meta = {
        "method": "shared_truncated_svd",
        "latent_dim": int(k),
        "weight_mode": weight_mode,
        "matrix_shape": [int(r.shape[0]), int(r.shape[1])],
        "nnz": int(r.nnz),
    }
    return user_latent.astype(np.float32), item_latent.astype(np.float32), latent_meta


# =========================================================
# 8) USER HISTORY AGGREGATION
# =========================================================
def build_user_history_embedding_from_item_block(
    interactions: pd.DataFrame,
    user_ids,
    item_ids,
    item_block: np.ndarray,
    weight_mode: str,
):
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {vid: i for i, vid in enumerate(item_ids)}
    weights = compute_interaction_weights(interactions, mode=weight_mode)
    rows = interactions["user_id"].map(user_index).values
    cols = interactions["video_id"].map(item_index).values
    valid_mask = ~pd.isna(rows) & ~pd.isna(cols)
    rows = rows[valid_mask].astype(np.int64, copy=False)
    cols = cols[valid_mask].astype(np.int64, copy=False)
    weights = weights[valid_mask]

    history_matrix = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    ).tocsr()

    weighted_sum = history_matrix @ item_block
    weight_sum = np.asarray(history_matrix.sum(axis=1), dtype=np.float32).reshape(-1)
    zero_mask = np.isclose(weight_sum, 0.0)
    safe_den = weight_sum.copy()
    safe_den[zero_mask] = 1.0
    user_emb = weighted_sum / safe_den[:, None]
    user_emb[zero_mask] = 0.0
    return user_emb.astype(np.float32)


# =========================================================
# 9) FINAL ASSEMBLY
# =========================================================
def assemble_history_only_user_embeddings(
    hist_user_content_pref: np.ndarray,
    hist_user_behavior_pref: np.ndarray | None,
    user_latent: np.ndarray,
):
    blocks = [hist_user_content_pref]
    if hist_user_behavior_pref is not None:
        blocks.append(hist_user_behavior_pref)
    blocks.append(user_latent)
    return np.concatenate(blocks, axis=1).astype(np.float32)


# =========================================================
# 10) PIPELINE
# =========================================================
def build_embeddings(config: EmbeddingConfig):
    ensure_dir(config.output_dir)

    interactions, inter_file = load_interactions(config)
    split = None
    if config.use_full_interactions_for_embeddings:
        embedding_interactions = interactions
        reference_end_date = get_latest_interaction_date(embedding_interactions)
        interaction_usage_meta = {
            "mode": "full_dataset",
            "source_rows_total": int(len(interactions)),
            "rows_used_for_embeddings": int(len(embedding_interactions)),
            "reference_end_date": int(reference_end_date),
            "reference_end_ts": float(embedding_interactions["timestamp"].max()),
        }
    else:
        split = split_interactions_by_time(interactions, config.train_ratio, config.val_ratio)
        embedding_interactions = split["train"]
        reference_end_date = int(split["meta"]["train_end_date"])
        interaction_usage_meta = {
            "mode": "time_split_training_subset",
            "source_rows_total": int(len(interactions)),
            "rows_used_for_embeddings": int(len(embedding_interactions)),
            **split["meta"],
        }
    item_cat_raw = load_item_categories(config.data_dir)

    item_content_df, segmented_caption_df, caption_text_meta = build_item_content_features(
        config.data_dir,
        text_dim=config.text_dim,
    )
    item_behavior_context_df, context_meta = build_item_behavior_context(
        config.data_dir,
        reference_end_date=reference_end_date,
        window_days=config.item_context_window_days,
    )

    user_behavior_summary_df = build_user_behavior_summary(embedding_interactions, item_category_df=item_cat_raw)

    all_user_ids = pd.Index(sorted(interactions["user_id"].unique()))
    all_item_ids = pd.Index(sorted(interactions["video_id"].unique()))

    user_behavior_summary_df = pd.DataFrame({"user_id": all_user_ids}).merge(user_behavior_summary_df, on="user_id", how="left")
    item_content_df = pd.DataFrame({"video_id": all_item_ids}).merge(item_content_df, on="video_id", how="left")
    item_behavior_context_df = pd.DataFrame({"video_id": all_item_ids}).merge(item_behavior_context_df, on="video_id", how="left")

    behavior_user_ids, user_behavior_summary = dataframe_to_numeric_matrix(user_behavior_summary_df, "user_id")
    item_ids, content_item_side = dataframe_to_numeric_matrix(item_content_df, "video_id")
    behavior_item_ids, behavior_item_context = dataframe_to_numeric_matrix(item_behavior_context_df, "video_id")

    if not np.array_equal(item_ids, behavior_item_ids):
        raise ValueError("Item matrix alignment mismatch between content and behavior context.")
    if not np.array_equal(np.asarray(all_user_ids), behavior_user_ids):
        raise ValueError("User matrix alignment mismatch between interaction ids and behavior summaries.")

    content_item_side = l2_normalize_rows(content_item_side)
    behavior_item_context = l2_normalize_rows(behavior_item_context)

    user_latent, item_latent, latent_meta = build_shared_latent_embeddings(
        embedding_interactions,
        all_user_ids=all_user_ids,
        all_item_ids=item_ids,
        latent_dim=config.latent_dim,
        weight_mode=config.interaction_weight_mode,
    )
    user_latent = l2_normalize_rows(user_latent)
    item_latent = l2_normalize_rows(item_latent)

    hist_user_content_pref = build_user_history_embedding_from_item_block(
        embedding_interactions,
        user_ids=all_user_ids,
        item_ids=item_ids,
        item_block=content_item_side,
        weight_mode=config.interaction_weight_mode,
    )
    hist_user_content_pref = l2_normalize_rows(hist_user_content_pref)
    hist_user_behavior_pref = build_user_history_embedding_from_item_block(
        embedding_interactions,
        user_ids=all_user_ids,
        item_ids=item_ids,
        item_block=behavior_item_context,
        weight_mode=config.interaction_weight_mode,
    )
    hist_user_behavior_pref = l2_normalize_rows(hist_user_behavior_pref)
    history_block_meta = {
        "interaction_file": inter_file,
        "interaction_mode": interaction_usage_meta["mode"],
        "item_source_block": "content_item_side",
        "item_block_dim": int(content_item_side.shape[1]),
        "weight_mode": config.interaction_weight_mode,
        "weight_formula": describe_interaction_weight_formula(config.interaction_weight_mode),
        "aggregation": "sum_t(w_t * content_item_side(item_t)) / sum_t(w_t)",
        "zero_weight_fallback": "zero vector when sum_t(w_t) == 0",
        "postprocess": "row-wise L2 normalize",
        "duplicate_interactions": "counted repeatedly; no per-item deduplication before averaging",
    }

    item_blocks = [content_item_side]
    if config.include_behavior_context_in_main_item_embedding:
        item_blocks.append(behavior_item_context)
    item_blocks.append(item_latent)
    final_item_emb = np.concatenate(item_blocks, axis=1).astype(np.float32)

    final_user_emb = assemble_history_only_user_embeddings(
        hist_user_content_pref=hist_user_content_pref,
        hist_user_behavior_pref=hist_user_behavior_pref if config.include_behavior_context_in_main_item_embedding else None,
        user_latent=user_latent,
    )

    if final_user_emb.shape[1] != final_item_emb.shape[1]:
        raise ValueError(
            f"user/item embedding dim mismatch: {final_user_emb.shape[1]} vs {final_item_emb.shape[1]}"
        )

    return {
        "config": config,
        "interactions": interactions,
        "split": split,
        "interaction_usage_meta": interaction_usage_meta,
        "item_ids": item_ids,
        "all_user_ids": np.array(all_user_ids),
        "content_item_side": content_item_side,
        "behavior_item_context": behavior_item_context,
        "item_latent": item_latent,
        "user_behavior_summary": user_behavior_summary,
        "user_latent": user_latent,
        "hist_user_content_pref": hist_user_content_pref,
        "hist_user_behavior_pref": hist_user_behavior_pref,
        "final_item_emb": final_item_emb,
        "final_user_emb": final_user_emb,
        "inter_file": inter_file,
        "item_behavior_context_df": item_behavior_context_df,
        "caption_text_artifact_df": segmented_caption_df,
        "caption_text_meta": caption_text_meta,
        "context_meta": context_meta,
        "latent_meta": latent_meta,
        "history_block_meta": history_block_meta,
    }


def save_embedding_bundle(bundle):
    config: EmbeddingConfig = bundle["config"]
    output_dir = resolve_runtime_path(config.output_dir)
    ensure_dir(output_dir)

    obsolete_files = [
        "user_static_side.npy",
        "predicted_user_latent.npy",
        "predicted_user_content_pref.npy",
        "predicted_user_behavior_pref.npy",
    ]
    for filename in obsolete_files:
        obsolete_path = os.path.join(output_dir, filename)
        if os.path.exists(obsolete_path):
            os.remove(obsolete_path)

    np.save(os.path.join(output_dir, "user_embeddings.npy"), bundle["final_user_emb"])
    np.save(os.path.join(output_dir, "item_embeddings.npy"), bundle["final_item_emb"])
    np.save(os.path.join(output_dir, "content_item_side.npy"), bundle["content_item_side"])
    np.save(os.path.join(output_dir, "behavior_item_context.npy"), bundle["behavior_item_context"])
    np.save(os.path.join(output_dir, "user_behavior_summary.npy"), bundle["user_behavior_summary"])
    np.save(os.path.join(output_dir, "user_latent.npy"), bundle["user_latent"])
    np.save(os.path.join(output_dir, "item_latent.npy"), bundle["item_latent"])
    caption_text_artifact_file = "segmented_caption_text.csv"
    if bundle.get("caption_text_artifact_df") is not None and not bundle["caption_text_artifact_df"].empty:
        bundle["caption_text_artifact_df"].to_csv(
            os.path.join(output_dir, caption_text_artifact_file),
            index=False,
            encoding="utf-8-sig",
        )

    user_id_map = pd.DataFrame({
        "row_index": np.arange(len(bundle["all_user_ids"])),
        "user_id": bundle["all_user_ids"],
    })
    item_id_map = pd.DataFrame({
        "row_index": np.arange(len(bundle["item_ids"])),
        "video_id": bundle["item_ids"],
    })

    quality_cols = [
        "video_id",
        "quality_index",
        "quality_index_raw",
        "quality_confidence",
        "quality_completion_signal",
        "quality_positive_interaction_signal",
        "quality_like_signal",
        "quality_negative_signal",
    ]
    available_quality_cols = [
        c for c in quality_cols
        if c in bundle["item_behavior_context_df"].columns
    ]
    if available_quality_cols:
        item_quality_df = bundle["item_behavior_context_df"][available_quality_cols].copy()
        item_quality_df.insert(0, "row_index", np.arange(len(item_quality_df)))
        for c in item_quality_df.columns:
            if c not in {"row_index", "video_id"}:
                item_quality_df[c] = pd.to_numeric(item_quality_df[c], errors="coerce").fillna(0.0)
        item_quality_df.to_csv(os.path.join(output_dir, "item_quality_index.csv"), index=False)
        item_id_map = item_id_map.merge(
            item_quality_df.drop(columns=["row_index"]),
            on="video_id",
            how="left",
        )
        for c in item_id_map.columns:
            if c not in {"row_index", "video_id"}:
                item_id_map[c] = pd.to_numeric(item_id_map[c], errors="coerce").fillna(0.0)

    user_id_map.to_csv(os.path.join(output_dir, "user_id_map.csv"), index=False)
    item_id_map.to_csv(os.path.join(output_dir, "item_id_map.csv"), index=False)

    meta = {
        "interaction_file": bundle["inter_file"],
        "interaction_usage": bundle["interaction_usage_meta"],
        "n_users": int(bundle["final_user_emb"].shape[0]),
        "n_items": int(bundle["final_item_emb"].shape[0]),
        "user_dim": int(bundle["final_user_emb"].shape[1]),
        "item_dim": int(bundle["final_item_emb"].shape[1]),
        "user_item_same_dim": bool(bundle["final_user_emb"].shape[1] == bundle["final_item_emb"].shape[1]),
        "final_item_blocks": (
            ["content_item_side", "behavior_item_context", "item_latent"]
            if config.include_behavior_context_in_main_item_embedding
            else ["content_item_side", "item_latent"]
        ),
        "final_user_strategy": {
            "mode": "history_only",
            "user_embedding": (
                "concat(hist_user_content_pref, hist_user_behavior_pref, user_latent)"
                if config.include_behavior_context_in_main_item_embedding
                else "concat(hist_user_content_pref, user_latent)"
            ),
        },
        "user_history_content_block": bundle["history_block_meta"],
        "item_behavior_context": bundle["context_meta"],
        "item_quality_index_file": "item_quality_index.csv",
        "caption_text_processing": {
            **bundle.get("caption_text_meta", {}),
            "segmented_output_file": (
                caption_text_artifact_file
                if bundle.get("caption_text_artifact_df") is not None and not bundle["caption_text_artifact_df"].empty
                else None
            ),
        },
        "latent_space": bundle["latent_meta"],
        "config": asdict(config),
        "design_principle": (
            "Main item embeddings combine content-only side features with shared collaborative latent factors. "
            "Behavioral daily aggregates are exported separately as time-aware context. "
            "All exported users and items come directly from full big_matrix interactions, "
            "and each user embedding is built from full-history item-content preference plus shared collaborative latent factors."
        ),
    }
    if bundle.get("split") is not None:
        meta["time_split"] = bundle["split"]["meta"]
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# =========================================================
# 11) OPTIONAL ANALYSIS HELPERS
# =========================================================
def top_k_closest_items_for_user(
    user_row_idx: int,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_id_map: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    u = user_embeddings[user_row_idx:user_row_idx + 1]
    sims = cosine_similarity(u, item_embeddings).reshape(-1)
    dists = 1.0 - sims
    out = item_id_map.copy()
    out["similarity"] = sims
    out["distance"] = dists
    return out.sort_values("distance", ascending=True).head(k).reset_index(drop=True)


def top_k_farthest_items_for_user(
    user_row_idx: int,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_id_map: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    u = user_embeddings[user_row_idx:user_row_idx + 1]
    sims = cosine_similarity(u, item_embeddings).reshape(-1)
    dists = 1.0 - sims
    out = item_id_map.copy()
    out["similarity"] = sims
    out["distance"] = dists
    return out.sort_values("distance", ascending=False).head(k).reset_index(drop=True)


# =========================================================
# 12) MAIN
# =========================================================
def main():
    config = EmbeddingConfig()
    bundle = build_embeddings(config)
    save_embedding_bundle(bundle)

    print("Done.")
    print(json.dumps({
        "interaction_file": bundle["inter_file"],
        "interaction_usage": bundle["interaction_usage_meta"],
        "n_users": int(bundle["final_user_emb"].shape[0]),
        "n_items": int(bundle["final_item_emb"].shape[0]),
        "user_dim": int(bundle["final_user_emb"].shape[1]),
        "item_dim": int(bundle["final_item_emb"].shape[1]),
        "final_item_blocks": (
            ["content_item_side", "behavior_item_context", "item_latent"]
            if config.include_behavior_context_in_main_item_embedding
            else ["content_item_side", "item_latent"]
        ),
    }, indent=2, ensure_ascii=False))

    item_id_map = pd.read_csv(os.path.join(resolve_runtime_path(config.output_dir), "item_id_map.csv"))
    if len(bundle["final_user_emb"]) > 0 and len(bundle["final_item_emb"]) > 0:
        print("\nPreview closest items for first user row:")
        print(top_k_closest_items_for_user(
            user_row_idx=0,
            user_embeddings=bundle["final_user_emb"],
            item_embeddings=bundle["final_item_emb"],
            item_id_map=item_id_map,
            k=5,
        ))


if __name__ == "__main__":
    main()
