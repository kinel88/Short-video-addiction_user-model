import os
import ast
import json
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD

from caption_text_pipeline import build_caption_text_feature_bundle


# =========================================================
# 0) CONFIG
# =========================================================
DATA_DIR = r"C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec\KuaiRec 2.0\data"
OUTPUT_DIR = r"C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec"

INTERACTION_CANDIDATES = ["big_matrix.csv", "small_matrix.csv"]
ITEM_DAILY_FILE = "item_daily_features.csv"
ITEM_CATEGORY_FILE = "item_categories.csv"
USER_FEATURE_FILE = "user_features.csv"
USER_FEATURE_RAW_FILE = "user_features_raw.csv"          # optional
CAPTION_CATEGORY_FILE = "__DISABLE__"   # optional

CF_DIM = 64
TEXT_DIM = 64
WATCH_RATIO_CLIP = 5.0


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
    "video_id": "int64",
    "feat": "string",   # parse later
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
    "app_version", "app_major_version", "app_minor_version", "isp"
]

ITEM_DAILY_STRING_COLS = [
    "video_type", "upload_dt", "upload_type", "visible_status", "video_tag_name"
]

# 工程上按 float 处理
ITEM_DAILY_FLOAT_COLS = [
    "video_duration", "play_progress"
]

CAPTION_STRING_COLS = [
    "manual_cover_text", "caption", "topic_tag",
    "first_level_category_name", "second_level_category_name", "third_level_category_name"
]

CAPTION_INT_COLS = [
    "video_id",
    "first_level_category_id",
    "second_level_category_id",
    "third_level_category_id"
]


# =========================================================
# 2) BASIC UTILS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_existing_file(data_dir: str, candidates):
    for f in candidates:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            return p
    return None


def safe_divide(a, b):
    return np.where(b == 0, 0, a / b)


def select_existing_columns(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def parse_feat_list(x):
    """
    item_categories.csv 的 feat 例子通常像:
    "[27, 9]" 或 "[27,9]"
    """
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


# =========================================================
# 3) LOAD INTERACTIONS
# =========================================================
def load_interactions(data_dir: str):
    path = find_existing_file(data_dir, INTERACTION_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Cannot find big_matrix.csv or small_matrix.csv")

    df = pd.read_csv(path, dtype=SMALL_MATRIX_DTYPES)
    df = clean_columns(df)

    required = [
        "user_id", "video_id", "play_duration", "video_duration",
        "time", "date", "timestamp", "watch_ratio"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in interactions: {missing}")

    # 防御性处理
    # numeric cleaning
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce")
    df["play_duration"] = pd.to_numeric(df["play_duration"], errors="coerce")
    df["video_duration"] = pd.to_numeric(df["video_duration"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce")

# date 列可能有 20200101.0 / 缺失 / 混合格式，先尽量转数字
    df["date"] = pd.to_numeric(df["date"], errors="coerce")

# drop impossible essential rows
    df = df.dropna(subset=["user_id", "video_id", "play_duration", "video_duration"])

# cast after cleaning
    df["user_id"] = df["user_id"].astype(np.int64)
    df["video_id"] = df["video_id"].astype(np.int64)
    df["play_duration"] = df["play_duration"].fillna(0).astype(np.int64)
    df["video_duration"] = df["video_duration"].fillna(0).astype(np.int64)

# date 保留成 pandas 可空整数更安全
    df["date"] = df["date"].astype("Int64")

    df["timestamp"] = df["timestamp"].fillna(0.0).astype(np.float64)
    df["watch_ratio"] = df["watch_ratio"].fillna(0.0).clip(0.0, WATCH_RATIO_CLIP).astype(np.float64)
    df["time"] = df["time"].astype("string").fillna("")

    # Derived behavioral signals
    df["is_valid_watch"] = (
        ((df["video_duration"] <= 7000) & (df["play_duration"] >= df["video_duration"])) |
        ((df["video_duration"] > 7000) & (df["play_duration"] > 7000))
    ).astype(np.float32)

    df["is_complete_watch"] = (df["play_duration"] >= df["video_duration"]).astype(np.float32)
    df["is_short_watch"] = (df["play_duration"] < np.minimum(3000, df["video_duration"])).astype(np.float32)

    return df, os.path.basename(path)


# =========================================================
# 4) LOAD ITEM CATEGORIES
# =========================================================
def load_item_categories(data_dir: str):
    path = os.path.join(data_dir, ITEM_CATEGORY_FILE)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, dtype=ITEM_CATEGORIES_DTYPES)
    df = clean_columns(df)

    required = ["video_id", "feat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in item_categories.csv: {missing}")

    df["feat_list"] = df["feat"].apply(parse_feat_list)
    return df


# =========================================================
# 5) BUILD ITEM FEATURES
# =========================================================
def build_item_features(data_dir: str):
    item_daily_path = os.path.join(data_dir, ITEM_DAILY_FILE)
    if not os.path.exists(item_daily_path):
        raise FileNotFoundError(f"Missing {ITEM_DAILY_FILE}")

    item_daily = pd.read_csv(item_daily_path)
    item_daily = clean_columns(item_daily)

    if "video_id" not in item_daily.columns:
        raise ValueError("item_daily_features.csv must contain video_id")

    item_daily = to_string_cols(item_daily, ITEM_DAILY_STRING_COLS)
    item_daily = to_float_cols(item_daily, ITEM_DAILY_FLOAT_COLS)

    # static item metadata
    static_candidates = [
        "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
        "video_duration", "video_width", "video_height", "music_id",
        "video_tag_id", "video_tag_name"
    ]
    static_cols = select_existing_columns(item_daily, static_candidates)
    item_static = item_daily.groupby("video_id", as_index=False)[static_cols].first()

    # daily behavior columns
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

    item_behavior = pd.concat(
        [item_behavior_mean, item_behavior_std, item_behavior_max],
        axis=1
    ).reset_index()

    # Derived ratio features
    ratio_specs = [
        ("play_cnt_mean", "show_cnt_mean", "play_per_show"),
        ("valid_play_cnt_mean", "show_cnt_mean", "valid_play_per_show"),
        ("complete_play_cnt_mean", "show_cnt_mean", "complete_play_per_show"),
        ("like_cnt_mean", "show_cnt_mean", "like_per_show"),
        ("comment_cnt_mean", "show_cnt_mean", "comment_per_show"),
        ("share_cnt_mean", "show_cnt_mean", "share_per_show"),
        ("follow_cnt_mean", "show_cnt_mean", "follow_per_show"),
        ("collect_cnt_mean", "show_cnt_mean", "collect_per_show"),
    ]
    for num_col, den_col, new_col in ratio_specs:
        if num_col in item_behavior.columns and den_col in item_behavior.columns:
            den = item_behavior[den_col].replace(0, np.nan).values
            item_behavior[new_col] = np.nan_to_num(item_behavior[num_col].values / den, nan=0.0, posinf=0.0, neginf=0.0)

    # item_categories multi-hot
    item_cat_df = load_item_categories(data_dir)
    item_cat_feature = None
    if item_cat_df is not None:
        mlb = MultiLabelBinarizer(sparse_output=False)
        cat_mat = mlb.fit_transform(item_cat_df["feat_list"])
        cat_cols = [f"cat_{c}" for c in mlb.classes_]
        item_cat_feature = pd.concat(
            [
                item_cat_df[["video_id"]].reset_index(drop=True),
                pd.DataFrame(cat_mat, columns=cat_cols)
            ],
            axis=1
        )

    # optional text/category file
    text_feature = None
    segmented_caption_df = pd.DataFrame()
    caption_text_meta = {
        "caption_csv": os.path.join(data_dir, CAPTION_CATEGORY_FILE),
        "valid_video_rows": 0,
        "text_feature_dim": 0,
        "available": False,
    }
    caption_path = os.path.join(data_dir, CAPTION_CATEGORY_FILE)
    if CAPTION_CATEGORY_FILE != "__DISABLE__" and os.path.exists(caption_path):
        text_feature, segmented_caption_df, caption_text_meta = build_caption_text_feature_bundle(
            caption_path=caption_path,
            text_dim=TEXT_DIM,
        )

    # merge all
    item_df = item_static.merge(item_behavior, on="video_id", how="outer")

    if item_cat_feature is not None:
        item_df = item_df.merge(item_cat_feature, on="video_id", how="left")

    if text_feature is not None:
        item_df = item_df.merge(text_feature, on="video_id", how="left")

    return item_df, segmented_caption_df, caption_text_meta


# =========================================================
# 6) BUILD USER FEATURES
# =========================================================
def build_user_features(data_dir: str, interactions: pd.DataFrame, item_category_df=None):
    raw_path = os.path.join(data_dir, USER_FEATURE_RAW_FILE)
    base_path = os.path.join(data_dir, USER_FEATURE_FILE)

    if os.path.exists(raw_path):
        user_df = pd.read_csv(raw_path)
        user_df = clean_columns(user_df)
        if "user_id" not in user_df.columns:
            raise ValueError("user_features_raw.csv must contain user_id")
        user_df = to_string_cols(user_df, USER_RAW_FORCE_STRING_COLS)

    elif os.path.exists(base_path):
        user_df = pd.read_csv(base_path)
        user_df = clean_columns(user_df)
        if "user_id" not in user_df.columns:
            raise ValueError("user_features.csv must contain user_id")
        user_df = to_string_cols(user_df, USER_FEATURE_STRING_COLS)

        # 关键修正：onehot_feat0~17 当离散类别处理，不当连续值
        for c in USER_ONEHOT_COLS:
            if c in user_df.columns:
                user_df[c] = user_df[c].astype("string").fillna("UNK")
    else:
        raise FileNotFoundError("Missing user_features.csv and user_features_raw.csv")

    # interaction-side user history
    user_hist = interactions.groupby("user_id").agg(
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

    # category preference profile from watched items
    user_cat_pref = None
    if item_category_df is not None and "video_id" in item_category_df.columns and "feat_list" in item_category_df.columns:
        mlb = MultiLabelBinarizer(sparse_output=False)
        item_cat_mat = mlb.fit_transform(item_category_df["feat_list"])
        item_cat_cols = [f"ucat_{c}" for c in mlb.classes_]

        item_cat_binary = pd.concat(
            [
                item_category_df[["video_id"]].reset_index(drop=True),
                pd.DataFrame(item_cat_mat, columns=item_cat_cols)
            ],
            axis=1
        )

        inter_cat = interactions[["user_id", "video_id", "watch_ratio"]].merge(
            item_cat_binary,
            on="video_id",
            how="left"
        )

        for c in item_cat_cols:
            inter_cat[c] = inter_cat[c].fillna(0.0) * inter_cat["watch_ratio"].fillna(0.0)

        user_cat_pref = inter_cat.groupby("user_id")[item_cat_cols].mean().reset_index()

    out = user_df.merge(user_hist, on="user_id", how="left")
    if user_cat_pref is not None:
        out = out.merge(user_cat_pref, on="user_id", how="left")

    return out


# =========================================================
# 7) COLLABORATIVE FILTERING EMBEDDINGS
# =========================================================
def build_cf_embeddings(interactions, all_user_ids, all_item_ids, cf_dim=64):
    user_index = {uid: i for i, uid in enumerate(all_user_ids)}
    item_index = {iid: i for i, iid in enumerate(all_item_ids)}

    weights = (
        0.2
        + 1.0 * interactions["watch_ratio"].clip(0, 2).values
        + 0.8 * interactions["is_valid_watch"].values
        + 0.8 * interactions["is_complete_watch"].values
        - 0.3 * interactions["is_short_watch"].values
    ).astype(np.float32)
    weights = np.clip(weights, 0.01, None)

    rows = interactions["user_id"].map(user_index).values
    cols = interactions["video_id"].map(item_index).values

    R = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(len(all_user_ids), len(all_item_ids)),
        dtype=np.float32
    ).tocsr()

    min_dim = min(R.shape)
    if min_dim <= 1:
        raise ValueError("Interaction matrix too small to run SVD.")

    k = min(cf_dim, min_dim - 1)

    svd_user = TruncatedSVD(n_components=k, random_state=42)
    user_latent = svd_user.fit_transform(R).astype(np.float32)

    svd_item = TruncatedSVD(n_components=k, random_state=42)
    item_latent = svd_item.fit_transform(R.T).astype(np.float32)

    return user_latent, item_latent


# =========================================================
# 8) DATAFRAME -> NUMERIC MATRIX
# =========================================================
def dataframe_to_numeric_matrix(df: pd.DataFrame, id_col: str):
    df = df.copy()
    ids = df[id_col].values
    df = df.drop(columns=[id_col])

    # bool -> int
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(np.int8)

    # object/string columns -> categorical one-hot
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
        obj_block = pd.get_dummies(df[obj_cols], dummy_na=False)
        obj_block = obj_block.astype(np.float32).values
    else:
        obj_block = np.zeros((len(df), 0), dtype=np.float32)

    X = np.concatenate([num_block, obj_block], axis=1).astype(np.float32)
    return ids, X


# =========================================================
# 9) MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    # interactions
    interactions, inter_file = load_interactions(DATA_DIR)
    print(f"Loaded interaction file: {inter_file}, shape={interactions.shape}")

    # item_categories raw
    item_cat_raw = load_item_categories(DATA_DIR)

    # side features
    item_df, segmented_caption_df, caption_text_meta = build_item_features(DATA_DIR)
    user_df = build_user_features(DATA_DIR, interactions, item_category_df=item_cat_raw)

    # align IDs
    all_user_ids = pd.Index(sorted(set(interactions["user_id"].unique()) | set(user_df["user_id"].unique())))
    all_item_ids = pd.Index(sorted(set(interactions["video_id"].unique()) | set(item_df["video_id"].unique())))

    # CF latent factors
    user_latent, item_latent = build_cf_embeddings(
        interactions=interactions,
        all_user_ids=all_user_ids,
        all_item_ids=all_item_ids,
        cf_dim=CF_DIM
    )

    # align side tables
    user_df = pd.DataFrame({"user_id": all_user_ids}).merge(user_df, on="user_id", how="left")
    item_df = pd.DataFrame({"video_id": all_item_ids}).merge(item_df, on="video_id", how="left")

    # side matrices
    user_ids, user_side = dataframe_to_numeric_matrix(user_df, "user_id")
    item_ids, item_side = dataframe_to_numeric_matrix(item_df, "video_id")

    # final embeddings
    final_user_emb = np.concatenate([user_side, user_latent], axis=1).astype(np.float32)
    final_item_emb = np.concatenate([item_side, item_latent], axis=1).astype(np.float32)

    # save
    np.save(os.path.join(OUTPUT_DIR, "user_embeddings.npy"), final_user_emb)
    np.save(os.path.join(OUTPUT_DIR, "item_embeddings.npy"), final_item_emb)

    pd.DataFrame({
        "row_index": np.arange(len(user_ids)),
        "user_id": user_ids
    }).to_csv(os.path.join(OUTPUT_DIR, "user_id_map.csv"), index=False)

    pd.DataFrame({
        "row_index": np.arange(len(item_ids)),
        "video_id": item_ids
    }).to_csv(os.path.join(OUTPUT_DIR, "item_id_map.csv"), index=False)
    if segmented_caption_df is not None and not segmented_caption_df.empty:
        segmented_caption_df.to_csv(
            os.path.join(OUTPUT_DIR, "segmented_caption_text.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    meta = {
        "interaction_file": inter_file,
        "n_users": int(final_user_emb.shape[0]),
        "n_items": int(final_item_emb.shape[0]),
        "user_dim": int(final_user_emb.shape[1]),
        "item_dim": int(final_item_emb.shape[1]),
        "cf_dim": CF_DIM,
        "text_dim": TEXT_DIM,
        "used_user_features_raw": os.path.exists(os.path.join(DATA_DIR, USER_FEATURE_RAW_FILE)),
        "used_caption_category": os.path.exists(os.path.join(DATA_DIR, CAPTION_CATEGORY_FILE)),
        "caption_text_processing": {
            **caption_text_meta,
            "segmented_output_file": (
                "segmented_caption_text.csv"
                if segmented_caption_df is not None and not segmented_caption_df.empty
                else None
            ),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
