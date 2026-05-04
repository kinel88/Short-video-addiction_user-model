import math
import inspect
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


WATCH_RATIO_BUCKET_EDGES = (0.35, 0.6, 0.8, 1.0, 1.3)
NUM_WATCH_RATIO_BUCKETS = 6
WATCH_RATIO_BUCKET_LABELS = (
    "[0.0, 0.35)",
    "[0.35, 0.6)",
    "[0.6, 0.8)",
    "[0.8, 1.0)",
    "[1.0, 1.3]",
    "> 1.3",
)
CATEGORY_BLOCK_DIM = 31
NOVELTY_HISTORY_WINDOW = 10
SESSION_GAP_SECONDS = 30 * 60
KUAIREC_LOCAL_TIMEZONE = "Asia/Shanghai"
DEFAULT_TRAIN_START_INDEX = 0
DEFAULT_TRAIN_ROW_COUNT = 90000
DEFAULT_VAL_ROW_COUNT = 10000
DEFAULT_MIN_SESSION_LENGTH = 20
SESSION_LOSS_REFERENCE_ROWS = DEFAULT_MIN_SESSION_LENGTH
MAX_REAL_WATCH_RATIO = 10.0
THRESHOLD_RATIO_NEAR_ZERO_CUTOFF = 0.75
THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF = 0.90
THRESHOLD_RATIO_HIGH_REGIME_START = 1.10
NEAR_ZERO_WATCH_RATIO_MAX = 0.10
_SOFTPLUS_ZERO = math.log(2.0)
REPEAT_WATCH_PASS_CAP = 10


def _softplus_inverse(value: float) -> float:
    value = float(max(value, 1e-6))
    return math.log(math.expm1(value))


def _logit(prob: float) -> float:
    prob = float(min(max(prob, 1e-6), 1.0 - 1e-6))
    return math.log(prob / (1.0 - prob))


def softplus(x: float) -> float:
    x = float(x)
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def watch_ratio_to_bucket_index(watch_ratio):
    """Map watch_ratio into 6 ordinal buckets using the current asymmetric watch-ratio split."""
    ratio = float(max(watch_ratio, 0.0))

    for bucket_idx, upper in enumerate(WATCH_RATIO_BUCKET_EDGES[:-1]):
        if ratio < upper:
            return bucket_idx

    if ratio <= WATCH_RATIO_BUCKET_EDGES[-1]:
        return len(WATCH_RATIO_BUCKET_EDGES) - 1

    return len(WATCH_RATIO_BUCKET_EDGES)


def bucket_index_to_label(bucket_index: int) -> str:
    bucket_index = int(bucket_index)
    if bucket_index < 0 or bucket_index >= len(WATCH_RATIO_BUCKET_LABELS):
        raise ValueError(f"bucket_index out of range: {bucket_index}")
    return WATCH_RATIO_BUCKET_LABELS[bucket_index]


def _torch_smoothstep(z):
    return z * z * (3.0 - 2.0 * z)


def _scalar_smoothstep(z):
    z = float(z)
    return z * z * (3.0 - 2.0 * z)


def _compute_single_pass_watch_mapping_torch(video_score, effective_threshold, watch_gain_base):
    threshold_safe = torch.clamp(effective_threshold, min=1e-6)
    threshold_ratio = video_score / threshold_safe

    near_zero_progress = torch.clamp(
        threshold_ratio / THRESHOLD_RATIO_NEAR_ZERO_CUTOFF,
        0.0,
        1.0,
    )
    near_zero_watch_ratio = NEAR_ZERO_WATCH_RATIO_MAX * _torch_smoothstep(
        near_zero_progress
    )

    first_progress = torch.clamp(
        (
            threshold_ratio - THRESHOLD_RATIO_NEAR_ZERO_CUTOFF
        ) / (
            THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF - THRESHOLD_RATIO_NEAR_ZERO_CUTOFF
        ),
        0.0,
        1.0,
    )
    first_bucket_watch_ratio = (
        NEAR_ZERO_WATCH_RATIO_MAX +
        (WATCH_RATIO_BUCKET_EDGES[0] - NEAR_ZERO_WATCH_RATIO_MAX) *
        _torch_smoothstep(first_progress)
    )

    second_progress = torch.clamp(
        (
            threshold_ratio - THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF
        ) / (
            THRESHOLD_RATIO_HIGH_REGIME_START - THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF
        ),
        0.0,
        1.0,
    )
    second_bucket_watch_ratio = (
        WATCH_RATIO_BUCKET_EDGES[0] +
        (WATCH_RATIO_BUCKET_EDGES[1] - WATCH_RATIO_BUCKET_EDGES[0]) *
        _torch_smoothstep(second_progress)
    )

    high_signal = video_score - THRESHOLD_RATIO_HIGH_REGIME_START * effective_threshold
    high_watch_ratio = WATCH_RATIO_BUCKET_EDGES[1] + torch.relu(
        F.softplus(watch_gain_base * high_signal) - _SOFTPLUS_ZERO
    )

    threshold_mapped_watch_ratio = torch.where(
        threshold_ratio < THRESHOLD_RATIO_NEAR_ZERO_CUTOFF,
        near_zero_watch_ratio,
        torch.where(
            threshold_ratio < THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF,
            first_bucket_watch_ratio,
            torch.where(
                threshold_ratio < THRESHOLD_RATIO_HIGH_REGIME_START,
                second_bucket_watch_ratio,
                high_watch_ratio,
            ),
        ),
    )
    return threshold_ratio, threshold_mapped_watch_ratio


def _compute_single_pass_watch_mapping_scalar(video_score, effective_threshold, watch_gain_base):
    video_score = float(video_score)
    effective_threshold = max(float(effective_threshold), 1e-6)
    watch_gain_base = float(watch_gain_base)
    threshold_ratio = video_score / effective_threshold

    near_zero_progress = min(
        max(threshold_ratio / THRESHOLD_RATIO_NEAR_ZERO_CUTOFF, 0.0),
        1.0,
    )
    near_zero_watch_ratio = NEAR_ZERO_WATCH_RATIO_MAX * _scalar_smoothstep(
        near_zero_progress
    )

    first_progress = min(
        max(
            (
                threshold_ratio - THRESHOLD_RATIO_NEAR_ZERO_CUTOFF
            ) / (
                THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF - THRESHOLD_RATIO_NEAR_ZERO_CUTOFF
            ),
            0.0,
        ),
        1.0,
    )
    first_bucket_watch_ratio = (
        NEAR_ZERO_WATCH_RATIO_MAX +
        (WATCH_RATIO_BUCKET_EDGES[0] - NEAR_ZERO_WATCH_RATIO_MAX) *
        _scalar_smoothstep(first_progress)
    )

    second_progress = min(
        max(
            (
                threshold_ratio - THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF
            ) / (
                THRESHOLD_RATIO_HIGH_REGIME_START - THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF
            ),
            0.0,
        ),
        1.0,
    )
    second_bucket_watch_ratio = (
        WATCH_RATIO_BUCKET_EDGES[0] +
        (WATCH_RATIO_BUCKET_EDGES[1] - WATCH_RATIO_BUCKET_EDGES[0]) *
        _scalar_smoothstep(second_progress)
    )

    high_signal = video_score - THRESHOLD_RATIO_HIGH_REGIME_START * effective_threshold
    high_watch_ratio = WATCH_RATIO_BUCKET_EDGES[1] + max(
        softplus(watch_gain_base * high_signal) - _SOFTPLUS_ZERO,
        0.0,
    )

    if threshold_ratio < THRESHOLD_RATIO_NEAR_ZERO_CUTOFF:
        threshold_mapped_watch_ratio = near_zero_watch_ratio
    elif threshold_ratio < THRESHOLD_RATIO_FIRST_BUCKET_CUTOFF:
        threshold_mapped_watch_ratio = first_bucket_watch_ratio
    elif threshold_ratio < THRESHOLD_RATIO_HIGH_REGIME_START:
        threshold_mapped_watch_ratio = second_bucket_watch_ratio
    else:
        threshold_mapped_watch_ratio = high_watch_ratio

    return float(threshold_ratio), float(threshold_mapped_watch_ratio)


def _compute_repeat_watch_ratio_torch(
    video_score,
    effective_threshold,
    watch_duration_scale,
    watch_gain_base,
    repeat_watch_decay,
    repeat_pass_cap=REPEAT_WATCH_PASS_CAP,
):
    total_watch_ratio = torch.zeros_like(video_score)
    active_mask = torch.ones_like(video_score)
    first_threshold_ratio = None
    first_threshold_mapped_watch_ratio = None
    first_base_watch_ratio = None

    for pass_idx in range(int(repeat_pass_cap)):
        decay_multiplier = torch.pow(repeat_watch_decay, pass_idx)
        pass_video_score = video_score * decay_multiplier
        threshold_ratio, threshold_mapped_watch_ratio = (
            _compute_single_pass_watch_mapping_torch(
                video_score=pass_video_score,
                effective_threshold=effective_threshold,
                watch_gain_base=watch_gain_base,
            )
        )
        pass_watch_ratio = threshold_mapped_watch_ratio * watch_duration_scale
        total_watch_ratio = total_watch_ratio + active_mask * torch.clamp(
            pass_watch_ratio,
            max=1.0,
        )

        if pass_idx == 0:
            first_threshold_ratio = threshold_ratio
            first_threshold_mapped_watch_ratio = threshold_mapped_watch_ratio
            first_base_watch_ratio = pass_watch_ratio

        active_mask = active_mask * (pass_watch_ratio > 1.0).to(pass_watch_ratio.dtype)
        if not bool(torch.any(active_mask > 0.0).detach().cpu().item()):
            break

    return (
        total_watch_ratio,
        first_threshold_ratio,
        first_threshold_mapped_watch_ratio,
        first_base_watch_ratio,
    )


def compute_threshold_banded_watch_ratio_components(
    video_score,
    effective_threshold,
    watch_duration_scale,
    watch_gain_base,
    repeat_watch_decay,
    repeat_pass_cap=REPEAT_WATCH_PASS_CAP,
):
    video_score = float(video_score)
    effective_threshold = max(float(effective_threshold), 1e-6)
    watch_duration_scale = float(watch_duration_scale)
    watch_gain_base = float(watch_gain_base)
    repeat_watch_decay = float(min(max(repeat_watch_decay, 0.0), 1.0))
    repeat_pass_cap = int(repeat_pass_cap)

    total_watch_ratio = 0.0
    active = True
    first_threshold_ratio = None
    first_threshold_mapped_watch_ratio = None
    first_base_watch_ratio = None
    pass_watch_ratios = []
    pass_contributions = []

    for pass_idx in range(repeat_pass_cap):
        pass_video_score = video_score * (repeat_watch_decay ** pass_idx)
        threshold_ratio, threshold_mapped_watch_ratio = (
            _compute_single_pass_watch_mapping_scalar(
                video_score=pass_video_score,
                effective_threshold=effective_threshold,
                watch_gain_base=watch_gain_base,
            )
        )
        pass_watch_ratio = threshold_mapped_watch_ratio * watch_duration_scale
        pass_watch_ratios.append(float(pass_watch_ratio))

        if pass_idx == 0:
            first_threshold_ratio = threshold_ratio
            first_threshold_mapped_watch_ratio = threshold_mapped_watch_ratio
            first_base_watch_ratio = pass_watch_ratio

        if active:
            contribution = min(pass_watch_ratio, 1.0)
            total_watch_ratio += contribution
            active = pass_watch_ratio > 1.0
        else:
            contribution = 0.0
        pass_contributions.append(float(contribution))

    return {
        "threshold_ratio": float(first_threshold_ratio),
        "threshold_mapped_watch_ratio": float(first_threshold_mapped_watch_ratio),
        "base_watch_ratio": float(first_base_watch_ratio),
        "pred_watch_ratio": float(total_watch_ratio),
        "repeat_pass_cap": repeat_pass_cap,
        "pass_watch_ratios": pass_watch_ratios,
        "pass_contributions": pass_contributions,
    }


def compute_novelty_norm(
    category_vec: Sequence[float],
    history_category_vectors: Optional[Sequence[Sequence[float]]],
    max_history: int = NOVELTY_HISTORY_WINDOW,
) -> float:
    current = np.asarray(category_vec, dtype=np.float32)
    if current.size == 0:
        return 0.5

    if not history_category_vectors:
        return 0.5

    history = [
        np.asarray(vec, dtype=np.float32)
        for vec in history_category_vectors[-max_history:]
    ]
    if len(history) == 0:
        return 0.5

    stacked = np.stack(history, axis=0)
    denom = math.sqrt(float(current.shape[0]))
    distances = np.linalg.norm(stacked - current.reshape(1, -1), axis=1)
    novelty = np.mean(distances) / max(denom, 1e-8)
    return float(np.clip(novelty, 0.0, 1.0))


def _compute_torch_novelty_norm(category_vec, history_category_vectors):
    if len(history_category_vectors) == 0:
        return torch.tensor(0.5, dtype=category_vec.dtype, device=category_vec.device)

    stacked = torch.stack(history_category_vectors[-NOVELTY_HISTORY_WINDOW:], dim=0)
    denom = math.sqrt(float(category_vec.shape[0]))
    distances = torch.linalg.norm(stacked - category_vec.unsqueeze(0), dim=1)
    novelty = torch.mean(distances) / max(denom, 1e-8)
    return torch.clamp(novelty, 0.0, 1.0)


def _compute_torch_decayed_peak(recent_value_history, peak_decay, device, dtype):
    if len(recent_value_history) == 0:
        return torch.tensor(0.0, dtype=dtype, device=device)

    work = recent_value_history[-NOVELTY_HISTORY_WINDOW:]
    weighted = []
    count = len(work)
    for idx, value in enumerate(work):
        age = count - 1 - idx
        decay_power = torch.tensor(float(age), dtype=dtype, device=device)
        weighted.append(value * torch.pow(peak_decay, decay_power))
    return torch.max(torch.stack(weighted))


def _compute_torch_time_weighted_average_reward(
    recent_value_history,
    average_reward_decay,
    device,
    dtype,
):
    if len(recent_value_history) == 0:
        return torch.tensor(0.0, dtype=dtype, device=device)

    work = recent_value_history[-NOVELTY_HISTORY_WINDOW:]
    weighted = []
    weights = []
    count = len(work)
    for idx, value in enumerate(work):
        age = count - 1 - idx
        decay_power = torch.tensor(float(age), dtype=dtype, device=device)
        weight = torch.pow(average_reward_decay, decay_power)
        weighted.append(value * weight)
        weights.append(weight)

    numerator = torch.sum(torch.stack(weighted))
    denominator = torch.clamp(torch.sum(torch.stack(weights)), min=1e-8)
    return numerator / denominator


def _compute_scalar_time_weighted_average_reward(
    recent_value_history,
    average_reward_decay,
):
    if len(recent_value_history) == 0:
        return 0.0

    decay = float(min(max(average_reward_decay, 0.0), 1.0))
    work = recent_value_history[-NOVELTY_HISTORY_WINDOW:]
    count = len(work)
    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(work):
        age = count - 1 - idx
        weight = decay ** age
        numerator += float(value) * weight
        denominator += weight

    return numerator / max(denominator, 1e-8)


def _compute_torch_dopamine_habit_progress(swipe_count, habit_growth_rate):
    swipe_count = torch.clamp(swipe_count, min=0.0)
    return 1.0 - torch.exp(-habit_growth_rate * swipe_count)


def _compute_scalar_dopamine_habit_progress(swipe_count: float, habit_growth_rate: float) -> float:
    swipe_count = max(float(swipe_count), 0.0)
    habit_growth_rate = max(float(habit_growth_rate), 0.0)
    return 1.0 - math.exp(-habit_growth_rate * swipe_count)


def _apply_torch_dopamine_update_scaffold(
    prev_dopamine,
    session_baseline,
    baseline_return_strength,
    swipe_count,
    habit_growth_rate,
    habit_max_gain,
    reward_drive,
):
    habit_progress = _compute_torch_dopamine_habit_progress(
        swipe_count=swipe_count,
        habit_growth_rate=habit_growth_rate,
    )
    habit_coef = torch.ones_like(prev_dopamine) + habit_max_gain * habit_progress
    baseline_return = baseline_return_strength * (session_baseline - prev_dopamine)
    dopamine_after = torch.clamp(
        prev_dopamine + baseline_return + habit_coef * reward_drive,
        min=0.0,
    )
    return dopamine_after, habit_progress, habit_coef, baseline_return


def _apply_scalar_dopamine_update_scaffold(
    prev_dopamine: float,
    session_baseline: float,
    baseline_return_strength: float,
    swipe_count: float,
    habit_growth_rate: float,
    habit_max_gain: float,
    reward_drive: float,
):
    habit_progress = _compute_scalar_dopamine_habit_progress(
        swipe_count=swipe_count,
        habit_growth_rate=habit_growth_rate,
    )
    habit_coef = 1.0 + max(float(habit_max_gain), 0.0) * habit_progress
    baseline_return = float(baseline_return_strength) * (
        float(session_baseline) - float(prev_dopamine)
    )
    dopamine_after = max(
        0.0,
        float(prev_dopamine) + baseline_return + habit_coef * float(reward_drive),
    )
    return dopamine_after, habit_progress, habit_coef, baseline_return


def _detach_state_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, list):
        return [_detach_state_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_state_tree(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _detach_state_tree(v) for k, v in obj.items()}
    return obj


def _validate_state_update_mode(state_update_mode: str) -> str:
    mode = str(state_update_mode).strip().lower()
    if mode not in {"rollout", "teacher_forcing"}:
        raise ValueError(
            "state_update_mode must be either 'rollout' or 'teacher_forcing'"
        )
    return mode


def _datetime_series_to_unix_seconds(
    series: pd.Series,
    default_timezone: str = KUAIREC_LOCAL_TIMEZONE,
) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=np.float64)

    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(
            default_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        parsed = parsed.dt.tz_convert(default_timezone)

    values_ns = parsed.astype("int64", copy=False).to_numpy(dtype=np.int64, copy=False)
    valid_mask = parsed.notna().to_numpy(dtype=bool, copy=False)
    values_sec = np.where(valid_mask, values_ns / 1e9, np.nan)
    return pd.Series(values_sec, index=series.index, dtype=np.float64)


def annotate_sessions(df: pd.DataFrame, session_gap_seconds: float = SESSION_GAP_SECONDS) -> pd.DataFrame:
    work = df.copy()
    if len(work) == 0:
        work["session_id"] = pd.Series(dtype=np.int64)
        work["session_row_index"] = pd.Series(dtype=np.int64)
        return work

    user_ids = work["user_id"].to_numpy(dtype=np.int64, copy=False)
    event_timestamps = work["event_timestamp"].to_numpy(dtype=np.float64, copy=False)

    session_ids = np.zeros(len(work), dtype=np.int64)
    session_row_indices = np.zeros(len(work), dtype=np.int64)

    current_user_id = None
    current_session_id = 0
    current_session_row_index = 0
    last_valid_timestamp = np.nan

    for idx, (user_id, event_timestamp) in enumerate(zip(user_ids, event_timestamps)):
        if current_user_id is None or user_id != current_user_id:
            current_user_id = int(user_id)
            current_session_id = 0
            current_session_row_index = 0
            last_valid_timestamp = np.nan
        else:
            gap_starts_new_session = (
                not np.isnan(event_timestamp)
                and not np.isnan(last_valid_timestamp)
                and float(event_timestamp - last_valid_timestamp) > float(session_gap_seconds)
            )
            if gap_starts_new_session:
                current_session_id += 1
                current_session_row_index = 0
            else:
                current_session_row_index += 1

        session_ids[idx] = current_session_id
        session_row_indices[idx] = current_session_row_index

        if not np.isnan(event_timestamp):
            last_valid_timestamp = float(event_timestamp)

    work["session_id"] = session_ids
    work["session_row_index"] = session_row_indices
    return work


def _build_event_timestamp_series(df: pd.DataFrame) -> pd.Series:
    event_timestamp = pd.Series(np.nan, index=df.index, dtype=np.float64)

    if "event_timestamp" in df.columns:
        event_timestamp = pd.to_numeric(df["event_timestamp"], errors="coerce").astype(np.float64)

    if "timestamp" in df.columns:
        timestamp_values = pd.to_numeric(df["timestamp"], errors="coerce").astype(np.float64)
        event_timestamp = event_timestamp.where(event_timestamp.notna(), timestamp_values)

    if "time" in df.columns:
        time_values = pd.to_datetime(df["time"], errors="coerce")
        time_fallback = _datetime_series_to_unix_seconds(time_values)
        event_timestamp = event_timestamp.where(event_timestamp.notna(), time_fallback)

    return event_timestamp.astype(np.float64)


def _rebuild_sessions_from_time_gap(
    df: pd.DataFrame,
    session_gap_seconds: float = SESSION_GAP_SECONDS,
) -> pd.DataFrame:
    if len(df) == 0:
        return annotate_sessions(df.copy(), session_gap_seconds=session_gap_seconds)

    work = df.copy()
    if "source_row_index" not in work.columns:
        work["source_row_index"] = np.arange(len(work), dtype=np.int64)

    work = work.sort_values(["user_id", "source_row_index"], kind="mergesort").reset_index(drop=True)
    work["event_timestamp"] = _build_event_timestamp_series(work)

    if not work["event_timestamp"].notna().any():
        raise ValueError(
            "teacher_forcing requires timestamp, time, or event_timestamp so sessions can be rebuilt "
            f"with the configured {int(session_gap_seconds // 60)}-minute gap"
        )

    return annotate_sessions(work, session_gap_seconds=session_gap_seconds)


def _ensure_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df
    if "session_id" not in work.columns:
        work = work.copy()
        work["session_id"] = np.zeros(len(work), dtype=np.int64)

    if "session_row_index" not in work.columns:
        if work is df:
            work = work.copy()
        work["session_row_index"] = work.groupby(["user_id", "session_id"], sort=False).cumcount()

    return work


def _prepare_sequence_dataframe(df: pd.DataFrame, state_update_mode: str) -> pd.DataFrame:
    mode = _validate_state_update_mode(state_update_mode)
    if mode == "teacher_forcing":
        return _rebuild_sessions_from_time_gap(df, session_gap_seconds=SESSION_GAP_SECONDS)
    return _ensure_session_columns(df)


def _session_boundaries(df: pd.DataFrame):
    if len(df) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    work = _ensure_session_columns(df)
    user_ids = work["user_id"].to_numpy(dtype=np.int64, copy=False)
    session_ids = work["session_id"].to_numpy(dtype=np.int64, copy=False)

    change_mask = (user_ids[1:] != user_ids[:-1]) | (session_ids[1:] != session_ids[:-1])
    session_starts = np.concatenate([
        np.array([0], dtype=np.int64),
        np.flatnonzero(change_mask).astype(np.int64) + 1,
    ])
    session_ends = np.concatenate([
        session_starts[1:],
        np.array([len(work)], dtype=np.int64),
    ])
    session_lengths = session_ends - session_starts
    return session_starts, session_ends, session_lengths


def _accumulate_complete_sessions(session_lengths, start_session_idx: int, requested_rows: Optional[int]):
    start_session_idx = int(start_session_idx)
    if start_session_idx >= len(session_lengths):
        return start_session_idx, 0

    if requested_rows is None:
        end_session_idx = len(session_lengths)
        return end_session_idx, int(np.sum(session_lengths[start_session_idx:end_session_idx]))

    requested_rows = int(max(requested_rows, 0))
    if requested_rows == 0:
        return start_session_idx, 0

    end_session_idx = start_session_idx
    accumulated_rows = 0
    while end_session_idx < len(session_lengths) and accumulated_rows < requested_rows:
        accumulated_rows += int(session_lengths[end_session_idx])
        end_session_idx += 1

    return end_session_idx, accumulated_rows


def _count_sessions(df: pd.DataFrame) -> int:
    session_starts, _, _ = _session_boundaries(df)
    return int(len(session_starts))


def _normalize_public_min_session_length(
    min_session_length: Optional[int],
) -> int:
    if min_session_length is None:
        return int(DEFAULT_MIN_SESSION_LENGTH)

    min_session_length = int(min_session_length)
    if min_session_length < DEFAULT_MIN_SESSION_LENGTH:
        raise ValueError(
            f"min_session_length must be at least {DEFAULT_MIN_SESSION_LENGTH}"
        )
    return min_session_length


def _filter_sessions_by_min_length(
    df: pd.DataFrame,
    min_session_length: Optional[int],
) -> pd.DataFrame:
    work = _ensure_session_columns(df)
    if len(work) == 0 or min_session_length is None:
        return work.reset_index(drop=True)

    min_session_length = int(min_session_length)
    if min_session_length <= 1:
        return work.reset_index(drop=True)

    session_starts, session_ends, session_lengths = _session_boundaries(work)
    if len(session_starts) == 0:
        return work.iloc[0:0].copy().reset_index(drop=True)

    keep_mask = np.zeros(len(work), dtype=bool)
    for start, end, session_length in zip(session_starts, session_ends, session_lengths):
        if int(session_length) >= min_session_length:
            keep_mask[start:end] = True

    return work.loc[keep_mask].reset_index(drop=True)


# =========================================================
# 1. KuaiRec small_matrix loader
# =========================================================
def load_kuairec_small_matrix(csv_path, row_limit: Optional[int] = None):
    print("reading csv...")
    read_kwargs = {}
    if row_limit is not None:
        row_limit = int(row_limit)
        if row_limit <= 0:
            raise ValueError("row_limit must be positive when provided")
        read_kwargs["nrows"] = row_limit
    df = pd.read_csv(csv_path, **read_kwargs)
    print("csv loaded, shape=", df.shape)
    df["source_row_index"] = np.arange(len(df), dtype=np.int64)

    numeric_cols = [
        "user_id", "video_id", "play_duration", "video_duration",
        "date", "timestamp", "watch_ratio"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["user_id", "video_id"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time"] = pd.NaT

    df["event_timestamp"] = df["timestamp"].astype(np.float64)
    time_fallback = _datetime_series_to_unix_seconds(df["time"])
    df["event_timestamp"] = df["event_timestamp"].where(df["event_timestamp"].notna(), time_fallback)

    if "watch_ratio" not in df.columns or df["watch_ratio"].isna().all():
        df["watch_ratio"] = df["play_duration"] / np.maximum(df["video_duration"], 1)

    df["watch_ratio"] = (
        df["watch_ratio"]
        .astype(np.float32)
        .clip(lower=0.0, upper=MAX_REAL_WATCH_RATIO)
    )

    print("sorting...")
    # Preserve per-user source order so missing-timestamp blocks stay attached to
    # their neighboring interactions when we mark session boundaries.
    df = df.sort_values(["user_id", "source_row_index"], kind="mergesort").reset_index(drop=True)
    df = annotate_sessions(df, session_gap_seconds=SESSION_GAP_SECONDS)
    print("sorting done")

    return df


# =========================================================
# 2. Feature provider
# =========================================================
class FeatureProvider:
    def get_user_features(self, user_id):
        raise NotImplementedError

    def get_video_features(self, video_id, user_id=None, history_category_vectors=None):
        raise NotImplementedError


class NpyFeatureProvider(FeatureProvider):
    """
    Read user/item vectors plus aligned item metadata and creator preference scores.
    """

    def __init__(
        self,
        user_embedding_path,
        item_embedding_path,
        user_id_map_path,
        item_id_map_path,
        item_daily_path,
        creator_preference_path,
    ):
        self.user_embeddings = np.load(user_embedding_path).astype(np.float32)
        self.item_embeddings = np.load(item_embedding_path).astype(np.float32)

        self.user_id_map = pd.read_csv(user_id_map_path)
        self.item_id_map = pd.read_csv(item_id_map_path)

        self.user_lookup = dict(
            zip(self.user_id_map["user_id"], self.user_id_map["row_index"])
        )
        self.item_lookup = dict(
            zip(self.item_id_map["video_id"], self.item_id_map["row_index"])
        )

        self.user_ids = [int(v) for v in self.user_id_map["user_id"].tolist()]
        self.video_ids = [int(v) for v in self.item_id_map["video_id"].tolist()]

        self.item_quality_lookup = {}
        self.item_quality_confidence_lookup = {}
        if "quality_index" in self.item_id_map.columns:
            self.item_quality_lookup = {
                int(video_id): float(quality) if pd.notna(quality) else 0.0
                for video_id, quality in zip(
                    self.item_id_map["video_id"],
                    self.item_id_map["quality_index"],
                )
            }
        if "quality_confidence" in self.item_id_map.columns:
            self.item_quality_confidence_lookup = {
                int(video_id): float(confidence) if pd.notna(confidence) else 0.0
                for video_id, confidence in zip(
                    self.item_id_map["video_id"],
                    self.item_id_map["quality_confidence"],
                )
            }

        item_daily = pd.read_csv(
            item_daily_path,
            usecols=["video_id", "author_id", "video_duration"],
        )
        for col in ["video_id", "author_id", "video_duration"]:
            item_daily[col] = pd.to_numeric(item_daily[col], errors="coerce")
        item_daily = item_daily.dropna(subset=["video_id"]).copy()
        item_daily["video_id"] = item_daily["video_id"].astype(np.int64)
        item_daily["author_id"] = item_daily["author_id"].fillna(0).astype(np.int64)
        item_daily["video_duration"] = item_daily["video_duration"].fillna(0.0).astype(np.float32)
        item_daily = (
            item_daily.groupby("video_id", as_index=False)
            .agg({"author_id": "first", "video_duration": "first"})
        )
        self.video_author_lookup = {
            int(video_id): int(author_id)
            for video_id, author_id in zip(item_daily["video_id"], item_daily["author_id"])
        }
        self.video_duration_lookup = {
            int(video_id): float(duration)
            for video_id, duration in zip(item_daily["video_id"], item_daily["video_duration"])
        }

        creator_df = pd.read_csv(
            creator_preference_path,
            usecols=["user_id", "author_id", "creator_preference_score"],
        )
        creator_df["user_id"] = pd.to_numeric(creator_df["user_id"], errors="coerce")
        creator_df["author_id"] = pd.to_numeric(creator_df["author_id"], errors="coerce")
        creator_df["creator_preference_score"] = pd.to_numeric(
            creator_df["creator_preference_score"], errors="coerce"
        )
        creator_df = creator_df.dropna(subset=["user_id", "author_id", "creator_preference_score"]).copy()
        creator_df["user_id"] = creator_df["user_id"].astype(np.int64)
        creator_df["author_id"] = creator_df["author_id"].astype(np.int64)
        creator_df["creator_preference_score"] = creator_df["creator_preference_score"].astype(np.float32)

        creator_user_ids = np.sort(creator_df["user_id"].unique())
        creator_author_ids = np.sort(creator_df["author_id"].unique())
        self.creator_user_index = {
            int(user_id): idx for idx, user_id in enumerate(creator_user_ids.tolist())
        }
        self.creator_author_index = {
            int(author_id): idx for idx, author_id in enumerate(creator_author_ids.tolist())
        }

        self.creator_pref_matrix = np.zeros(
            (len(creator_user_ids), len(creator_author_ids)),
            dtype=np.float32,
        )

        user_positions = creator_df["user_id"].map(self.creator_user_index).to_numpy(dtype=np.int64)
        author_positions = creator_df["author_id"].map(self.creator_author_index).to_numpy(dtype=np.int64)
        self.creator_pref_matrix[user_positions, author_positions] = (
            creator_df["creator_preference_score"].to_numpy(dtype=np.float32)
        )

        creator_pref_scale = np.percentile(
            np.abs(self.creator_pref_matrix),
            90,
            axis=1,
        ).astype(np.float32)
        self.creator_pref_scale_by_user = {
            int(user_id): float(max(scale, 1.0))
            for user_id, scale in zip(creator_user_ids.tolist(), creator_pref_scale.tolist())
        }

        self.user_dim = int(self.user_embeddings.shape[1])
        self.item_dim = int(self.item_embeddings.shape[1])

        if self.user_dim != self.item_dim:
            raise ValueError(
                f"user embedding dim ({self.user_dim}) != item embedding dim ({self.item_dim}). "
                "user/item vectors must have the same dimension for dot-product matching."
            )

        self.embedding_dim = self.user_dim

    def get_all_user_ids(self):
        return list(self.user_ids)

    def get_all_video_ids(self):
        return list(self.video_ids)

    def compute_creator_preference_norm(self, user_id, author_id):
        user_id = int(user_id)
        author_id = int(author_id)
        user_row = self.creator_user_index.get(user_id)
        author_col = self.creator_author_index.get(author_id)

        if user_row is None or author_col is None:
            return 0.0

        raw_score = float(self.creator_pref_matrix[user_row, author_col])
        scale = float(self.creator_pref_scale_by_user.get(user_id, 1.0))
        return float(np.tanh(raw_score / max(scale, 1.0)))

    def get_user_features(self, user_id):
        user_id = int(user_id)
        if user_id not in self.user_lookup:
            raise KeyError(f"user_id {user_id} not found in user_id_map.csv")

        row_index = int(self.user_lookup[user_id])
        vec = self.user_embeddings[row_index].astype(np.float32)

        return {
            "user_id": user_id,
            "user_vec": vec,
        }

    def get_video_features(self, video_id, user_id=None, history_category_vectors=None):
        video_id = int(video_id)
        if video_id not in self.item_lookup:
            raise KeyError(f"video_id {video_id} not found in item_id_map.csv")

        row_index = int(self.item_lookup[video_id])
        vec = self.item_embeddings[row_index].astype(np.float32)
        category_vec = vec[:CATEGORY_BLOCK_DIM].astype(np.float32)
        author_id = int(self.video_author_lookup.get(video_id, 0))

        feat = {
            "video_id": video_id,
            "item_vec": vec,
            "category_vec_31": category_vec,
            "quality_index": float(self.item_quality_lookup.get(video_id, 0.0)),
            "quality_norm": float(np.clip(self.item_quality_lookup.get(video_id, 0.0) / 100.0, 0.0, 1.0)),
            "quality_confidence": float(self.item_quality_confidence_lookup.get(video_id, 0.0)),
            "author_id": author_id,
            "video_duration_ms": float(max(self.video_duration_lookup.get(video_id, 0.0), 0.0)),
        }

        if user_id is not None:
            feat["creator_pref_norm"] = self.compute_creator_preference_norm(
                user_id=user_id,
                author_id=author_id,
            )

        if history_category_vectors is not None:
            feat["novelty_norm"] = compute_novelty_norm(
                category_vec=category_vec,
                history_category_vectors=history_category_vectors,
            )

        return feat


# =========================================================
# 3. Trainable behavior fit model
# =========================================================
class BehaviorFitModel(nn.Module):
    def __init__(self, embedding_dim, user_ids=None):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dot_scale = math.sqrt(float(embedding_dim))
        self.user_ids = sorted({int(user_id) for user_id in (user_ids or [])})
        self.user_automaticity_index = {
            int(user_id): idx for idx, user_id in enumerate(self.user_ids)
        }

        self.score_preference_match_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.40), dtype=torch.float32)
        )
        self.score_quality_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.40), dtype=torch.float32)
        )
        self.score_novelty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.30), dtype=torch.float32)
        )
        self.score_creator_preference_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.50), dtype=torch.float32)
        )

        self.effective_threshold_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.85), dtype=torch.float32)
        )
        self.effective_threshold_dopamine_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.effective_threshold_expected_reward_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.effective_threshold_automaticity_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.effective_threshold_fatigue_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.35), dtype=torch.float32)
        )
        self.watch_gain_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.30), dtype=torch.float32)
        )
        self.watch_gain_dopamine_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.55), dtype=torch.float32)
        )
        self.fatigue_duration_penalty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.repeat_watch_decay_raw = nn.Parameter(
            torch.tensor(_logit(0.80), dtype=torch.float32)
        )

        self.bucket_w = nn.Parameter(
            torch.zeros((NUM_WATCH_RATIO_BUCKETS, 7), dtype=torch.float32)
        )

        self.fatigue_watch_ratio_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.06), dtype=torch.float32)
        )
        self.fatigue_high_engaged_bonus_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.20), dtype=torch.float32)
        )

        self.dopamine_base_raw = nn.Parameter(
            torch.tensor(1.00, dtype=torch.float32)
        )
        self.dopamine_normal_level_gap_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.35), dtype=torch.float32)
        )
        self.dopamine_baseline_return_strength_raw = nn.Parameter(
            torch.tensor(_logit(0.20), dtype=torch.float32)
        )
        self.dopamine_habit_growth_rate_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.01), dtype=torch.float32)
        )
        self.dopamine_habit_max_gain_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.35), dtype=torch.float32)
        )
        self.dopamine_score_engagement_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.55), dtype=torch.float32)
        )
        self.dopamine_expected_reward_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.40), dtype=torch.float32)
        )
        self.dopamine_peak_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.45), dtype=torch.float32)
        )
        self.dopamine_mix_alpha_raw = nn.Parameter(
            torch.tensor(_logit(0.70), dtype=torch.float32)
        )

        self.expected_alpha_raw = nn.Parameter(
            torch.tensor(_logit(0.35), dtype=torch.float32)
        )
        self.expected_reward_peak_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.45), dtype=torch.float32)
        )
        self.expected_reward_average_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.40), dtype=torch.float32)
        )
        self.average_reward_decay_raw = nn.Parameter(
            torch.tensor(_logit(0.85), dtype=torch.float32)
        )
        self.peak_decay_raw = nn.Parameter(
            torch.tensor(_logit(0.85), dtype=torch.float32)
        )
        self.user_automaticity_default_raw = nn.Parameter(
            torch.tensor(_logit(0.50), dtype=torch.float32)
        )
        self.raw_user_automaticity = nn.Parameter(
            torch.zeros(len(self.user_ids), dtype=torch.float32)
        )
        self.raw_user_score_weight_delta = nn.Parameter(
            torch.zeros((len(self.user_ids), 4), dtype=torch.float32)
        )
        self.register_buffer(
            "user_score_weight_reg_lambda",
            torch.zeros(len(self.user_ids), dtype=torch.float32),
        )
        self.register_buffer(
            "user_score_weight_reg_session_count",
            torch.ones(len(self.user_ids), dtype=torch.float32),
        )

    def _strict_zero_dopamine_enabled(self):
        return bool(getattr(self, "_strict_zero_dopamine_eval", False))

    def _apply_eval_param_overrides(self, params):
        if not self._strict_zero_dopamine_enabled():
            return params

        overridden = dict(params)
        for key in DOPAMINE_PARAM_KEYS:
            overridden[key] = torch.zeros_like(overridden[key])
        return overridden

    def _params(self):
        params = {
            "score_preference_match": F.softplus(self.score_preference_match_raw),
            "score_quality": F.softplus(self.score_quality_raw),
            "score_novelty": F.softplus(self.score_novelty_raw),
            "score_creator_preference": F.softplus(self.score_creator_preference_raw),
            "effective_threshold_base": F.softplus(self.effective_threshold_base_raw),
            "effective_threshold_dopamine": F.softplus(self.effective_threshold_dopamine_raw),
            "effective_threshold_expected_reward": F.softplus(
                self.effective_threshold_expected_reward_raw
            ),
            "effective_threshold_automaticity": F.softplus(
                self.effective_threshold_automaticity_raw
            ),
            "effective_threshold_fatigue": F.softplus(self.effective_threshold_fatigue_raw),
            "watch_gain_base": F.softplus(self.watch_gain_base_raw),
            "watch_gain_dopamine": F.softplus(self.watch_gain_dopamine_raw),
            "fatigue_duration_penalty": F.softplus(self.fatigue_duration_penalty_raw),
            "repeat_watch_decay": torch.sigmoid(self.repeat_watch_decay_raw),
            "fatigue_watch_ratio_coef": F.softplus(self.fatigue_watch_ratio_coef_raw),
            "fatigue_high_engaged_bonus": F.softplus(self.fatigue_high_engaged_bonus_raw),
            "dopamine_base": self.dopamine_base_raw,
            "dopamine_normal_level": self.dopamine_base_raw + F.softplus(
                self.dopamine_normal_level_gap_raw
            ),
            "dopamine_baseline_return_strength": torch.clamp(
                torch.sigmoid(self.dopamine_baseline_return_strength_raw),
                0.01,
                0.99,
            ),
            "dopamine_habit_growth_rate": F.softplus(self.dopamine_habit_growth_rate_raw),
            "dopamine_habit_max_gain": F.softplus(self.dopamine_habit_max_gain_raw),
            "dopamine_score_engagement_coef": F.softplus(self.dopamine_score_engagement_coef_raw),
            "dopamine_expected_reward_coef": F.softplus(self.dopamine_expected_reward_coef_raw),
            "dopamine_peak_coef": F.softplus(self.dopamine_peak_coef_raw),
            "dopamine_mix_alpha": torch.sigmoid(self.dopamine_mix_alpha_raw),
            "expected_alpha": torch.clamp(torch.sigmoid(self.expected_alpha_raw), 0.01, 0.99),
            "expected_reward_peak_coef": F.softplus(self.expected_reward_peak_coef_raw),
            "expected_reward_average_coef": F.softplus(self.expected_reward_average_coef_raw),
            "average_reward_decay": torch.clamp(
                torch.sigmoid(self.average_reward_decay_raw),
                0.05,
                0.99,
            ),
            "peak_decay": torch.clamp(torch.sigmoid(self.peak_decay_raw), 0.05, 0.99),
        }
        return self._apply_eval_param_overrides(params)

    def _resolve_user_index(self, user_feat=None, user_id=None):
        resolved_user_id = user_id
        if resolved_user_id is None and isinstance(user_feat, dict):
            resolved_user_id = user_feat.get("user_id")

        if resolved_user_id is None:
            return None

        return self.user_automaticity_index.get(int(resolved_user_id))

    def _resolve_user_automaticity(self, user_feat, device):
        user_idx = self._resolve_user_index(user_feat=user_feat)

        if user_idx is None or self.raw_user_automaticity.numel() == 0:
            raw_value = self.user_automaticity_default_raw
        else:
            raw_value = self.raw_user_automaticity[int(user_idx)]
        return torch.sigmoid(raw_value).to(device=device)

    def _global_score_weight_raw_vector(self, device):
        return torch.stack(
            [
                self.score_preference_match_raw,
                self.score_quality_raw,
                self.score_novelty_raw,
                self.score_creator_preference_raw,
            ]
        ).to(device=device)

    def _resolve_user_score_weight_vector(self, user_feat, device):
        global_raw = self._global_score_weight_raw_vector(device=device)
        user_idx = self._resolve_user_index(user_feat=user_feat)
        if user_idx is None or self.raw_user_score_weight_delta.numel() == 0:
            return F.softplus(global_raw)

        delta_raw = self.raw_user_score_weight_delta[int(user_idx)].to(device=device)
        return F.softplus(global_raw + delta_raw)

    def set_user_score_weight_regularization_stats(self, train_df, state_update_mode):
        lambda_values = torch.zeros_like(self.user_score_weight_reg_lambda)
        session_values = torch.ones_like(self.user_score_weight_reg_session_count)

        if len(self.user_ids) == 0 or len(train_df) == 0:
            self.user_score_weight_reg_lambda.copy_(
                lambda_values.to(device=self.user_score_weight_reg_lambda.device)
            )
            self.user_score_weight_reg_session_count.copy_(
                session_values.to(device=self.user_score_weight_reg_session_count.device)
            )
            return

        prepared_train_df = _prepare_sequence_dataframe(
            train_df,
            state_update_mode=state_update_mode,
        )
        row_counts = prepared_train_df.groupby("user_id", sort=False).size().to_dict()
        session_counts = (
            prepared_train_df[["user_id", "session_id"]]
            .drop_duplicates()
            .groupby("user_id", sort=False)
            .size()
            .to_dict()
        )

        for user_id, user_idx in self.user_automaticity_index.items():
            train_rows = int(row_counts.get(user_id, 0))
            if train_rows > 0:
                lambda_value = 1e-3 * max(
                    math.sqrt(100.0 / float(max(train_rows, 1))) - 1.0,
                    0.0,
                )
            else:
                lambda_value = 0.0
            lambda_values[int(user_idx)] = float(lambda_value)
            session_values[int(user_idx)] = float(max(int(session_counts.get(user_id, 1)), 1))

        self.user_score_weight_reg_lambda.copy_(
            lambda_values.to(device=self.user_score_weight_reg_lambda.device)
        )
        self.user_score_weight_reg_session_count.copy_(
            session_values.to(device=self.user_score_weight_reg_session_count.device)
        )

    def compute_user_score_weight_regularization_loss(self, user_id):
        user_idx = self._resolve_user_index(user_id=user_id)
        if user_idx is None or self.raw_user_score_weight_delta.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=self.score_preference_match_raw.device)

        user_idx = int(user_idx)
        lambda_value = self.user_score_weight_reg_lambda[user_idx]
        session_count = torch.clamp(self.user_score_weight_reg_session_count[user_idx], min=1.0)
        delta_raw = self.raw_user_score_weight_delta[user_idx]
        return (lambda_value / session_count) * torch.sum(delta_raw * delta_raw)

    def make_strict_zero_dopamine_eval_copy(self):
        device = next(self.parameters()).device
        clone = BehaviorFitModel(
            embedding_dim=self.embedding_dim,
            user_ids=self.user_ids,
        ).to(device=device)
        clone.load_state_dict(self.state_dict())
        clone._strict_zero_dopamine_eval = True
        clone.eval()
        return clone

    def init_state(self, user_feat, device):
        params = self._params()
        dopamine_base = params["dopamine_base"].to(device=device)
        dopamine_state = dopamine_base.clone()
        if self._strict_zero_dopamine_enabled():
            dopamine_state = torch.zeros_like(dopamine_base)
        user_automaticity = self._resolve_user_automaticity(
            user_feat=user_feat,
            device=device,
        )
        return {
            "fatigue": torch.tensor(0.0, dtype=torch.float32, device=device),
            "dopamine": dopamine_state,
            "expected_reward": torch.tensor(0.0, dtype=torch.float32, device=device),
            "session_peak": torch.tensor(0.0, dtype=torch.float32, device=device),
            "time_weighted_average_reward": torch.tensor(0.0, dtype=torch.float32, device=device),
            "user_automaticity": user_automaticity,
            "swipe_count": torch.tensor(0.0, dtype=torch.float32, device=device),
            "recent_category_history": [],
            "recent_score_history": [],
        }

    def _compute_dopamine_reward_drive(
        self,
        params,
        state,
        new_state,
        aux,
        score_engagement,
    ):
        del state, aux, score_engagement
        return (
            params["dopamine_expected_reward_coef"] * new_state["expected_reward"]
            + params["dopamine_peak_coef"] * new_state["session_peak"]
        )

    def _scaled_dot(self, user_vec, item_vec):
        return torch.sum(user_vec * item_vec) / self.dot_scale

    def forward_one(self, state, user_feat, video_feat, device):
        params = self._params()

        user_vec = torch.tensor(user_feat["user_vec"], dtype=torch.float32, device=device)
        item_vec = torch.tensor(video_feat["item_vec"], dtype=torch.float32, device=device)
        category_vec = torch.tensor(video_feat["category_vec_31"], dtype=torch.float32, device=device)
        quality_norm = torch.tensor(float(video_feat["quality_norm"]), dtype=torch.float32, device=device)
        creator_pref_norm = torch.tensor(
            float(video_feat.get("creator_pref_norm", 0.0)),
            dtype=torch.float32,
            device=device,
        )

        pref_dot = self._scaled_dot(user_vec, item_vec)
        pref_match = torch.sigmoid(pref_dot)
        novelty_norm = _compute_torch_novelty_norm(category_vec, state["recent_category_history"])
        score_weight_vector = self._resolve_user_score_weight_vector(
            user_feat=user_feat,
            device=device,
        )

        video_score = (
            score_weight_vector[0] * pref_match +
            score_weight_vector[1] * quality_norm +
            score_weight_vector[2] * novelty_norm +
            score_weight_vector[3] * creator_pref_norm
        )

        if self._strict_zero_dopamine_enabled():
            current_dopamine = torch.zeros_like(state["dopamine"])
        else:
            current_dopamine = state["dopamine"]
        current_expected_reward = state.get(
            "expected_reward",
            torch.tensor(0.0, dtype=torch.float32, device=device),
        )
        user_automaticity = state.get(
            "user_automaticity",
            self._resolve_user_automaticity(user_feat=user_feat, device=device),
        )
        effective_threshold = (
            params["effective_threshold_base"] +
            params["effective_threshold_expected_reward"] * current_expected_reward +
            params["effective_threshold_automaticity"] * user_automaticity
        )
        effective_signal = video_score - effective_threshold
        watch_duration_scale = torch.exp(
            -params["fatigue_duration_penalty"] * state["fatigue"]
        )
        pred_watch_ratio, threshold_ratio, threshold_mapped_watch_ratio, base_watch_ratio = (
            _compute_repeat_watch_ratio_torch(
                video_score=video_score,
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=params["watch_gain_base"],
                repeat_watch_decay=params["repeat_watch_decay"],
            )
        )
        bucket_x = torch.stack([
            pref_match,
            quality_norm,
            novelty_norm,
            creator_pref_norm,
            current_expected_reward,
            user_automaticity,
            state["fatigue"],
        ])
        bucket_logits = torch.matmul(self.bucket_w, bucket_x)

        aux = {
            "pref_dot": pref_dot,
            "pref_match": pref_match,
            "novelty_norm": novelty_norm,
            "video_score": video_score,
            "quality_norm": quality_norm,
            "score_preference_match": score_weight_vector[0],
            "score_quality": score_weight_vector[1],
            "score_novelty": score_weight_vector[2],
            "score_creator_preference": score_weight_vector[3],
            "current_dopamine": current_dopamine,
            "expected_reward": current_expected_reward,
            "user_automaticity": user_automaticity,
            "effective_threshold": effective_threshold,
            "effective_signal": effective_signal,
            "threshold_ratio": threshold_ratio,
            "threshold_mapped_watch_ratio": threshold_mapped_watch_ratio,
            "base_watch_ratio": base_watch_ratio,
            "category_vec": category_vec,
        }
        return bucket_logits, pred_watch_ratio, aux

    def update_state_with_real_feedback(
        self,
        state,
        y_watch_ratio,
        y_bucket,
        aux,
    ):
        del y_bucket
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=y_watch_ratio,
            aux=aux,
        )

    def update_state_with_prediction(
        self,
        state,
        pred_watch_ratio,
        aux,
    ):
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=pred_watch_ratio,
            aux=aux,
        )

    def _update_state_from_watch_ratio(
        self,
        state,
        watch_ratio,
        aux,
    ):
        params = self._params()

        new_state = {}
        device = state["dopamine"].device
        dtype = state["dopamine"].dtype

        positive_watch_ratio = torch.clamp(watch_ratio, min=0.0)
        high_engaged = (positive_watch_ratio > 1.0).float()
        score_engagement = aux["video_score"] * positive_watch_ratio

        recent_category_history = list(state["recent_category_history"])
        recent_score_history = list(state["recent_score_history"])
        recent_category_history.append(aux["category_vec"])
        recent_category_history = recent_category_history[-NOVELTY_HISTORY_WINDOW:]
        recent_score_history.append(score_engagement)
        recent_score_history = recent_score_history[-NOVELTY_HISTORY_WINDOW:]

        new_state["recent_category_history"] = recent_category_history
        new_state["recent_score_history"] = recent_score_history
        new_state["session_peak"] = _compute_torch_decayed_peak(
            recent_value_history=recent_score_history,
            peak_decay=params["peak_decay"],
            device=device,
            dtype=dtype,
        )
        new_state["time_weighted_average_reward"] = (
            _compute_torch_time_weighted_average_reward(
                recent_value_history=recent_score_history,
                average_reward_decay=params["average_reward_decay"],
                device=device,
                dtype=dtype,
            )
        )
        new_state["expected_reward"] = (
            params["expected_reward_peak_coef"] * new_state["session_peak"] +
            params["expected_reward_average_coef"] * new_state["time_weighted_average_reward"]
        )
        new_state["user_automaticity"] = state.get(
            "user_automaticity",
            torch.sigmoid(self.user_automaticity_default_raw).to(device=device),
        )

        new_state["fatigue"] = torch.clamp(
            state["fatigue"] +
            params["fatigue_watch_ratio_coef"] * positive_watch_ratio +
            high_engaged * params["fatigue_high_engaged_bonus"],
            0.0,
            1.0,
        )
        new_state["swipe_count"] = state.get(
            "swipe_count",
            torch.tensor(0.0, dtype=dtype, device=device),
        ) + torch.tensor(1.0, dtype=dtype, device=device)

        if self._strict_zero_dopamine_enabled():
            new_state["dopamine"] = torch.zeros_like(state["dopamine"])
        else:
            reward_drive = self._compute_dopamine_reward_drive(
                params=params,
                state=state,
                new_state=new_state,
                aux=aux,
                score_engagement=score_engagement,
            )
            dopamine_after, _, _, _ = _apply_torch_dopamine_update_scaffold(
                prev_dopamine=state["dopamine"],
                session_baseline=params["dopamine_base"],
                baseline_return_strength=params["dopamine_baseline_return_strength"],
                swipe_count=new_state["swipe_count"],
                habit_growth_rate=params["dopamine_habit_growth_rate"],
                habit_max_gain=params["dopamine_habit_max_gain"],
                reward_drive=reward_drive,
            )
            new_state["dopamine"] = dopamine_after

        return new_state

    def export_env_params(self):
        params = self._params()
        global_score_weight_raw = self._global_score_weight_raw_vector(device=torch.device("cpu"))
        score_weight_by_id = {
            "score_preference_match_by_id": {},
            "score_quality_by_id": {},
            "score_novelty_by_id": {},
            "score_creator_preference_by_id": {},
        }
        if self.raw_user_score_weight_delta.numel() > 0:
            user_delta_cpu = self.raw_user_score_weight_delta.detach().cpu()
            for user_id, idx in self.user_automaticity_index.items():
                resolved = F.softplus(global_score_weight_raw + user_delta_cpu[int(idx)])
                score_weight_by_id["score_preference_match_by_id"][int(user_id)] = float(
                    resolved[0].item()
                )
                score_weight_by_id["score_quality_by_id"][int(user_id)] = float(
                    resolved[1].item()
                )
                score_weight_by_id["score_novelty_by_id"][int(user_id)] = float(
                    resolved[2].item()
                )
                score_weight_by_id["score_creator_preference_by_id"][int(user_id)] = float(
                    resolved[3].item()
                )
        return {
            "score_preference_match": float(params["score_preference_match"].detach().cpu().item()),
            "score_quality": float(params["score_quality"].detach().cpu().item()),
            "score_novelty": float(params["score_novelty"].detach().cpu().item()),
            "score_creator_preference": float(params["score_creator_preference"].detach().cpu().item()),
            **score_weight_by_id,
            "effective_threshold_base": float(params["effective_threshold_base"].detach().cpu().item()),
            "effective_threshold_dopamine": float(
                params["effective_threshold_dopamine"].detach().cpu().item()
            ),
            "effective_threshold_expected_reward": float(
                params["effective_threshold_expected_reward"].detach().cpu().item()
            ),
            "effective_threshold_automaticity": float(
                params["effective_threshold_automaticity"].detach().cpu().item()
            ),
            "effective_threshold_fatigue": float(
                params["effective_threshold_fatigue"].detach().cpu().item()
            ),
            "watch_gain_base": float(params["watch_gain_base"].detach().cpu().item()),
            "watch_gain_dopamine": float(params["watch_gain_dopamine"].detach().cpu().item()),
            "fatigue_duration_penalty": float(
                params["fatigue_duration_penalty"].detach().cpu().item()
            ),
            "repeat_watch_decay": float(params["repeat_watch_decay"].detach().cpu().item()),
            "repeat_pass_cap": REPEAT_WATCH_PASS_CAP,
            "fatigue_watch_ratio_coef": float(
                params["fatigue_watch_ratio_coef"].detach().cpu().item()
            ),
            "fatigue_high_engaged_bonus": float(params["fatigue_high_engaged_bonus"].detach().cpu().item()),
            "dopamine_base": float(params["dopamine_base"].detach().cpu().item()),
            "dopamine_normal_level": float(
                params["dopamine_normal_level"].detach().cpu().item()
            ),
            "dopamine_baseline_return_strength": float(
                params["dopamine_baseline_return_strength"].detach().cpu().item()
            ),
            "dopamine_habit_growth_rate": float(
                params["dopamine_habit_growth_rate"].detach().cpu().item()
            ),
            "dopamine_habit_max_gain": float(
                params["dopamine_habit_max_gain"].detach().cpu().item()
            ),
            "dopamine_score_engagement_coef": float(
                params["dopamine_score_engagement_coef"].detach().cpu().item()
            ),
            "dopamine_expected_reward_coef": float(
                params["dopamine_expected_reward_coef"].detach().cpu().item()
            ),
            "dopamine_peak_coef": float(params["dopamine_peak_coef"].detach().cpu().item()),
            "dopamine_mix_alpha": float(params["dopamine_mix_alpha"].detach().cpu().item()),
            "expected_alpha": float(params["expected_alpha"].detach().cpu().item()),
            "expected_reward_peak_coef": float(
                params["expected_reward_peak_coef"].detach().cpu().item()
            ),
            "expected_reward_average_coef": float(
                params["expected_reward_average_coef"].detach().cpu().item()
            ),
            "average_reward_decay": float(
                params["average_reward_decay"].detach().cpu().item()
            ),
            "peak_decay": float(params["peak_decay"].detach().cpu().item()),
            "user_automaticity_default": float(
                torch.sigmoid(self.user_automaticity_default_raw).detach().cpu().item()
            ),
            "user_automaticity_by_id": {
                int(user_id): float(
                    torch.sigmoid(self.raw_user_automaticity[idx]).detach().cpu().item()
                )
                for user_id, idx in self.user_automaticity_index.items()
            },
        }


DOPAMINE_PARAM_KEYS = {
    "effective_threshold_dopamine",
    "watch_gain_dopamine",
    "dopamine_base",
    "dopamine_normal_level",
    "dopamine_baseline_return_strength",
    "dopamine_habit_growth_rate",
    "dopamine_habit_max_gain",
    "dopamine_score_engagement_coef",
    "dopamine_expected_reward_coef",
    "dopamine_peak_coef",
    "dopamine_mix_alpha",
}


class NoDopamineBehaviorFitModel(nn.Module):
    """
    Dopamine-free ablation baseline.

    It keeps the current score / novelty / fatigue structure,
    but removes dopamine and the residual reward/peak-memory chain from
    both the forward path and the state dynamics.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dot_scale = math.sqrt(float(embedding_dim))

        self.score_preference_match_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.40), dtype=torch.float32)
        )
        self.score_quality_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.40), dtype=torch.float32)
        )
        self.score_novelty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.30), dtype=torch.float32)
        )
        self.score_creator_preference_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.50), dtype=torch.float32)
        )

        self.effective_threshold_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.85), dtype=torch.float32)
        )
        self.effective_threshold_fatigue_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.35), dtype=torch.float32)
        )
        self.watch_gain_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.30), dtype=torch.float32)
        )
        self.fatigue_duration_penalty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.repeat_watch_decay_raw = nn.Parameter(
            torch.tensor(_logit(0.80), dtype=torch.float32)
        )

        self.bucket_w = nn.Parameter(
            torch.zeros((NUM_WATCH_RATIO_BUCKETS, 5), dtype=torch.float32)
        )

        self.fatigue_watch_ratio_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.06), dtype=torch.float32)
        )
        self.fatigue_high_engaged_bonus_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.20), dtype=torch.float32)
        )

    def _params(self):
        return {
            "score_preference_match": F.softplus(self.score_preference_match_raw),
            "score_quality": F.softplus(self.score_quality_raw),
            "score_novelty": F.softplus(self.score_novelty_raw),
            "score_creator_preference": F.softplus(self.score_creator_preference_raw),
            "effective_threshold_base": F.softplus(self.effective_threshold_base_raw),
            "effective_threshold_fatigue": F.softplus(self.effective_threshold_fatigue_raw),
            "watch_gain_base": F.softplus(self.watch_gain_base_raw),
            "fatigue_duration_penalty": F.softplus(self.fatigue_duration_penalty_raw),
            "repeat_watch_decay": torch.sigmoid(self.repeat_watch_decay_raw),
            "fatigue_watch_ratio_coef": F.softplus(self.fatigue_watch_ratio_coef_raw),
            "fatigue_high_engaged_bonus": F.softplus(self.fatigue_high_engaged_bonus_raw),
        }

    def init_state(self, user_feat, device):
        del user_feat
        return {
            "fatigue": torch.tensor(0.0, dtype=torch.float32, device=device),
            "recent_category_history": [],
        }

    def _scaled_dot(self, user_vec, item_vec):
        return torch.sum(user_vec * item_vec) / self.dot_scale

    def forward_one(self, state, user_feat, video_feat, device):
        params = self._params()

        user_vec = torch.tensor(user_feat["user_vec"], dtype=torch.float32, device=device)
        item_vec = torch.tensor(video_feat["item_vec"], dtype=torch.float32, device=device)
        category_vec = torch.tensor(video_feat["category_vec_31"], dtype=torch.float32, device=device)
        quality_norm = torch.tensor(float(video_feat["quality_norm"]), dtype=torch.float32, device=device)
        creator_pref_norm = torch.tensor(
            float(video_feat.get("creator_pref_norm", 0.0)),
            dtype=torch.float32,
            device=device,
        )

        pref_dot = self._scaled_dot(user_vec, item_vec)
        pref_match = torch.sigmoid(pref_dot)
        novelty_norm = _compute_torch_novelty_norm(category_vec, state["recent_category_history"])

        video_score = (
            params["score_preference_match"] * pref_match +
            params["score_quality"] * quality_norm +
            params["score_novelty"] * novelty_norm +
            params["score_creator_preference"] * creator_pref_norm
        )

        effective_threshold = (
            params["effective_threshold_base"]
        )
        effective_signal = video_score - effective_threshold
        watch_duration_scale = torch.exp(
            -params["fatigue_duration_penalty"] * state["fatigue"]
        )
        pred_watch_ratio, threshold_ratio, threshold_mapped_watch_ratio, base_watch_ratio = (
            _compute_repeat_watch_ratio_torch(
                video_score=video_score,
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=params["watch_gain_base"],
                repeat_watch_decay=params["repeat_watch_decay"],
            )
        )
        bucket_x = torch.stack([
            pref_match,
            quality_norm,
            novelty_norm,
            creator_pref_norm,
            state["fatigue"],
        ])
        bucket_logits = torch.matmul(self.bucket_w, bucket_x)

        aux = {
            "pref_dot": pref_dot,
            "pref_match": pref_match,
            "novelty_norm": novelty_norm,
            "video_score": video_score,
            "quality_norm": quality_norm,
            "effective_threshold": effective_threshold,
            "effective_signal": effective_signal,
            "threshold_ratio": threshold_ratio,
            "threshold_mapped_watch_ratio": threshold_mapped_watch_ratio,
            "base_watch_ratio": base_watch_ratio,
            "category_vec": category_vec,
        }
        return bucket_logits, pred_watch_ratio, aux

    def update_state_with_real_feedback(
        self,
        state,
        y_watch_ratio,
        y_bucket,
        aux,
    ):
        del y_bucket
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=y_watch_ratio,
            aux=aux,
        )

    def update_state_with_prediction(
        self,
        state,
        pred_watch_ratio,
        aux,
    ):
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=pred_watch_ratio,
            aux=aux,
        )

    def _update_state_from_watch_ratio(
        self,
        state,
        watch_ratio,
        aux,
    ):
        params = self._params()

        new_state = {}

        positive_watch_ratio = torch.clamp(watch_ratio, min=0.0)
        high_engaged = (positive_watch_ratio > 1.0).float()

        recent_category_history = list(state["recent_category_history"])
        recent_category_history.append(aux["category_vec"])
        recent_category_history = recent_category_history[-NOVELTY_HISTORY_WINDOW:]

        new_state["recent_category_history"] = recent_category_history
        new_state["fatigue"] = torch.clamp(
            state["fatigue"] +
            params["fatigue_watch_ratio_coef"] * positive_watch_ratio +
            high_engaged * params["fatigue_high_engaged_bonus"],
            0.0,
            1.0,
        )

        return new_state

    def export_env_params(self):
        params = self._params()
        exported = {
            "score_preference_match": float(params["score_preference_match"].detach().cpu().item()),
            "score_quality": float(params["score_quality"].detach().cpu().item()),
            "score_novelty": float(params["score_novelty"].detach().cpu().item()),
            "score_creator_preference": float(params["score_creator_preference"].detach().cpu().item()),
            "effective_threshold_base": float(params["effective_threshold_base"].detach().cpu().item()),
            "effective_threshold_fatigue": float(
                params["effective_threshold_fatigue"].detach().cpu().item()
            ),
            "watch_gain_base": float(params["watch_gain_base"].detach().cpu().item()),
            "fatigue_duration_penalty": float(
                params["fatigue_duration_penalty"].detach().cpu().item()
            ),
            "repeat_watch_decay": float(params["repeat_watch_decay"].detach().cpu().item()),
            "repeat_pass_cap": REPEAT_WATCH_PASS_CAP,
            "fatigue_watch_ratio_coef": float(
                params["fatigue_watch_ratio_coef"].detach().cpu().item()
            ),
            "fatigue_high_engaged_bonus": float(
                params["fatigue_high_engaged_bonus"].detach().cpu().item()
            ),
        }
        return exported


class ContentOnlyFatigueBehaviorFitModel(nn.Module):
    """
    Content-only behavior baseline with fatigue-only duration compression.

    This path keeps the current content scoring and nonlinear repeat-watch
    mapping, but removes every dopamine-related state and parameter. The
    watch threshold is still learned, but it is a fixed session-invariant
    base threshold.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dot_scale = math.sqrt(float(embedding_dim))

        self.score_preference_match_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.40), dtype=torch.float32)
        )
        self.score_quality_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.40), dtype=torch.float32)
        )
        self.score_novelty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.30), dtype=torch.float32)
        )
        self.score_creator_preference_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.50), dtype=torch.float32)
        )

        self.effective_threshold_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.85), dtype=torch.float32)
        )
        self.watch_gain_base_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(1.30), dtype=torch.float32)
        )
        self.fatigue_duration_penalty_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.25), dtype=torch.float32)
        )
        self.repeat_watch_decay_raw = nn.Parameter(
            torch.tensor(_logit(0.80), dtype=torch.float32)
        )

        self.bucket_w = nn.Parameter(
            torch.zeros((NUM_WATCH_RATIO_BUCKETS, 5), dtype=torch.float32)
        )

        self.fatigue_watch_ratio_coef_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.06), dtype=torch.float32)
        )
        self.fatigue_high_engaged_bonus_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(0.20), dtype=torch.float32)
        )

    def _params(self):
        return {
            "score_preference_match": F.softplus(self.score_preference_match_raw),
            "score_quality": F.softplus(self.score_quality_raw),
            "score_novelty": F.softplus(self.score_novelty_raw),
            "score_creator_preference": F.softplus(self.score_creator_preference_raw),
            "effective_threshold_base": F.softplus(self.effective_threshold_base_raw),
            "watch_gain_base": F.softplus(self.watch_gain_base_raw),
            "fatigue_duration_penalty": F.softplus(self.fatigue_duration_penalty_raw),
            "repeat_watch_decay": torch.sigmoid(self.repeat_watch_decay_raw),
            "fatigue_watch_ratio_coef": F.softplus(self.fatigue_watch_ratio_coef_raw),
            "fatigue_high_engaged_bonus": F.softplus(self.fatigue_high_engaged_bonus_raw),
        }

    def init_state(self, user_feat, device):
        del user_feat
        return {
            "fatigue": torch.tensor(0.0, dtype=torch.float32, device=device),
            "recent_category_history": [],
        }

    def _scaled_dot(self, user_vec, item_vec):
        return torch.sum(user_vec * item_vec) / self.dot_scale

    def forward_one(self, state, user_feat, video_feat, device):
        params = self._params()

        user_vec = torch.tensor(user_feat["user_vec"], dtype=torch.float32, device=device)
        item_vec = torch.tensor(video_feat["item_vec"], dtype=torch.float32, device=device)
        category_vec = torch.tensor(video_feat["category_vec_31"], dtype=torch.float32, device=device)
        quality_norm = torch.tensor(float(video_feat["quality_norm"]), dtype=torch.float32, device=device)
        creator_pref_norm = torch.tensor(
            float(video_feat.get("creator_pref_norm", 0.0)),
            dtype=torch.float32,
            device=device,
        )

        pref_dot = self._scaled_dot(user_vec, item_vec)
        pref_match = torch.sigmoid(pref_dot)
        novelty_norm = _compute_torch_novelty_norm(category_vec, state["recent_category_history"])

        video_score = (
            params["score_preference_match"] * pref_match +
            params["score_quality"] * quality_norm +
            params["score_novelty"] * novelty_norm +
            params["score_creator_preference"] * creator_pref_norm
        )

        effective_threshold = params["effective_threshold_base"]
        effective_signal = video_score - effective_threshold
        watch_duration_scale = torch.exp(
            -params["fatigue_duration_penalty"] * state["fatigue"]
        )
        pred_watch_ratio, threshold_ratio, threshold_mapped_watch_ratio, base_watch_ratio = (
            _compute_repeat_watch_ratio_torch(
                video_score=video_score,
                effective_threshold=effective_threshold,
                watch_duration_scale=watch_duration_scale,
                watch_gain_base=params["watch_gain_base"],
                repeat_watch_decay=params["repeat_watch_decay"],
            )
        )
        bucket_x = torch.stack([
            pref_match,
            quality_norm,
            novelty_norm,
            creator_pref_norm,
            state["fatigue"],
        ])
        bucket_logits = torch.matmul(self.bucket_w, bucket_x)

        aux = {
            "pref_dot": pref_dot,
            "pref_match": pref_match,
            "novelty_norm": novelty_norm,
            "video_score": video_score,
            "quality_norm": quality_norm,
            "effective_threshold": effective_threshold,
            "effective_signal": effective_signal,
            "threshold_ratio": threshold_ratio,
            "threshold_mapped_watch_ratio": threshold_mapped_watch_ratio,
            "base_watch_ratio": base_watch_ratio,
            "category_vec": category_vec,
        }
        return bucket_logits, pred_watch_ratio, aux

    def update_state_with_real_feedback(
        self,
        state,
        y_watch_ratio,
        y_bucket,
        aux,
    ):
        del y_bucket
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=y_watch_ratio,
            aux=aux,
        )

    def update_state_with_prediction(
        self,
        state,
        pred_watch_ratio,
        aux,
    ):
        return self._update_state_from_watch_ratio(
            state=state,
            watch_ratio=pred_watch_ratio,
            aux=aux,
        )

    def _update_state_from_watch_ratio(
        self,
        state,
        watch_ratio,
        aux,
    ):
        params = self._params()

        positive_watch_ratio = torch.clamp(watch_ratio, min=0.0)
        high_engaged = (positive_watch_ratio > 1.0).float()

        recent_category_history = list(state["recent_category_history"])
        recent_category_history.append(aux["category_vec"])
        recent_category_history = recent_category_history[-NOVELTY_HISTORY_WINDOW:]

        return {
            "recent_category_history": recent_category_history,
            "fatigue": torch.clamp(
                state["fatigue"] +
                params["fatigue_watch_ratio_coef"] * positive_watch_ratio +
                high_engaged * params["fatigue_high_engaged_bonus"],
                0.0,
                1.0,
            ),
        }

    def export_env_params(self):
        params = self._params()
        return {
            "score_preference_match": float(params["score_preference_match"].detach().cpu().item()),
            "score_quality": float(params["score_quality"].detach().cpu().item()),
            "score_novelty": float(params["score_novelty"].detach().cpu().item()),
            "score_creator_preference": float(params["score_creator_preference"].detach().cpu().item()),
            "effective_threshold_base": float(params["effective_threshold_base"].detach().cpu().item()),
            "watch_gain_base": float(params["watch_gain_base"].detach().cpu().item()),
            "fatigue_duration_penalty": float(
                params["fatigue_duration_penalty"].detach().cpu().item()
            ),
            "repeat_watch_decay": float(params["repeat_watch_decay"].detach().cpu().item()),
            "repeat_pass_cap": REPEAT_WATCH_PASS_CAP,
            "fatigue_watch_ratio_coef": float(
                params["fatigue_watch_ratio_coef"].detach().cpu().item()
            ),
            "fatigue_high_engaged_bonus": float(
                params["fatigue_high_engaged_bonus"].detach().cpu().item()
            ),
        }


def _prepare_behavior_fit_train_val_split_from_dataframe(
    df: pd.DataFrame,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    state_update_mode: str = "rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    base_df = _ensure_session_columns(df)

    requested_train_start_index = int(max(train_start_index, 0))
    requested_train_row_count = int(train_row_count)
    requested_val_row_count = None if val_row_count is None else int(max(val_row_count, 0))
    min_session_length = _normalize_public_min_session_length(min_session_length)

    if requested_train_row_count <= 0:
        raise ValueError("train_row_count must be positive")

    prepared_full_df = _prepare_sequence_dataframe(base_df, state_update_mode=state_update_mode)
    if requested_train_start_index >= len(prepared_full_df):
        raise ValueError("train_start_index must be smaller than the number of available rows")

    session_starts, _, session_lengths = _session_boundaries(prepared_full_df)
    if len(session_starts) == 0:
        raise ValueError("No complete sessions available after sequence preparation")

    start_session_idx = int(
        np.searchsorted(session_starts, requested_train_start_index, side="left")
    )
    if start_session_idx >= len(session_starts):
        raise ValueError("No complete session starts at or after train_start_index")

    actual_train_start_index = int(session_starts[start_session_idx])
    train_end_session_idx, _ = _accumulate_complete_sessions(
        session_lengths=session_lengths,
        start_session_idx=start_session_idx,
        requested_rows=requested_train_row_count,
    )
    actual_train_end_index = (
        int(session_starts[train_end_session_idx])
        if train_end_session_idx < len(session_starts)
        else int(len(prepared_full_df))
    )
    prepared_train_df = prepared_full_df.iloc[
        actual_train_start_index:actual_train_end_index
    ].reset_index(drop=True)

    val_start_session_idx = train_end_session_idx
    actual_val_start_index = actual_train_end_index
    val_end_session_idx, _ = _accumulate_complete_sessions(
        session_lengths=session_lengths,
        start_session_idx=val_start_session_idx,
        requested_rows=requested_val_row_count,
    )
    actual_val_end_index = (
        int(session_starts[val_end_session_idx])
        if val_end_session_idx < len(session_starts)
        else int(len(prepared_full_df))
    )
    prepared_val_df = prepared_full_df.iloc[
        actual_val_start_index:actual_val_end_index
    ].reset_index(drop=True)

    pre_filter_train_session_count = _count_sessions(prepared_train_df)
    pre_filter_val_session_count = _count_sessions(prepared_val_df)
    pre_filter_train_row_count = int(len(prepared_train_df))
    pre_filter_val_row_count = int(len(prepared_val_df))

    train_df = _filter_sessions_by_min_length(
        prepared_train_df,
        min_session_length=min_session_length,
    )
    val_df = _filter_sessions_by_min_length(
        prepared_val_df,
        min_session_length=min_session_length,
    )

    post_filter_train_session_count = _count_sessions(train_df)
    post_filter_val_session_count = _count_sessions(val_df)
    post_filter_train_row_count = int(len(train_df))
    post_filter_val_row_count = int(len(val_df))
    selected_subset_row_count = int(pre_filter_train_row_count + pre_filter_val_row_count)
    total_session_count = int(len(session_starts))

    return {
        "full": prepared_full_df,
        "raw_train": prepared_train_df,
        "raw_val": prepared_val_df,
        "train": train_df,
        "val": val_df,
        "meta": {
            "state_update_mode": state_update_mode,
            "requested_train_start_index": requested_train_start_index,
            "requested_train_row_count": requested_train_row_count,
            "requested_val_row_count": requested_val_row_count,
            "session_gap_minutes": int(SESSION_GAP_SECONDS // 60),
            "min_session_length": min_session_length,
            "train_start_index": actual_train_start_index,
            "train_row_count": int(len(train_df)),
            "val_row_count": int(len(val_df)),
            "subset_row_count": selected_subset_row_count,
            "train_end_index_exclusive": actual_train_end_index,
            "val_start_index": actual_val_start_index,
            "val_end_index_exclusive": actual_val_end_index,
            "actual_train_start_index": actual_train_start_index,
            "actual_train_row_count": int(len(train_df)),
            "actual_val_start_index": actual_val_start_index,
            "actual_val_row_count": int(len(val_df)),
            "actual_train_session_count": post_filter_train_session_count,
            "actual_val_session_count": post_filter_val_session_count,
            "pre_filter_train_row_count": pre_filter_train_row_count,
            "pre_filter_val_row_count": pre_filter_val_row_count,
            "pre_filter_train_session_count": pre_filter_train_session_count,
            "pre_filter_val_session_count": pre_filter_val_session_count,
            "post_filter_train_row_count": post_filter_train_row_count,
            "post_filter_val_row_count": post_filter_val_row_count,
            "post_filter_train_session_count": post_filter_train_session_count,
            "post_filter_val_session_count": post_filter_val_session_count,
            "total_row_count": int(len(base_df)),
            "total_session_count": total_session_count,
        },
    }


def prepare_behavior_fit_explicit_train_val_split_from_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    state_update_mode: str = "rollout",
    train_source_name: str = "train_df",
    val_source_name: str = "val_df",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    min_session_length = _normalize_public_min_session_length(min_session_length)

    train_base_df = _ensure_session_columns(train_df)
    val_base_df = _ensure_session_columns(val_df)

    prepared_train_df = _prepare_sequence_dataframe(
        train_base_df,
        state_update_mode=state_update_mode,
    )
    prepared_val_df = _prepare_sequence_dataframe(
        val_base_df,
        state_update_mode=state_update_mode,
    )

    pre_filter_train_session_count = _count_sessions(prepared_train_df)
    pre_filter_val_session_count = _count_sessions(prepared_val_df)
    pre_filter_train_row_count = int(len(prepared_train_df))
    pre_filter_val_row_count = int(len(prepared_val_df))

    filtered_train_df = _filter_sessions_by_min_length(
        prepared_train_df,
        min_session_length=min_session_length,
    )
    filtered_val_df = _filter_sessions_by_min_length(
        prepared_val_df,
        min_session_length=min_session_length,
    )

    post_filter_train_session_count = _count_sessions(filtered_train_df)
    post_filter_val_session_count = _count_sessions(filtered_val_df)
    post_filter_train_row_count = int(len(filtered_train_df))
    post_filter_val_row_count = int(len(filtered_val_df))

    return {
        "raw_train": prepared_train_df,
        "raw_val": prepared_val_df,
        "train": filtered_train_df,
        "val": filtered_val_df,
        "meta": {
            "split_mode": "explicit_train_val_sources",
            "state_update_mode": state_update_mode,
            "min_session_length": min_session_length,
            "train_source_name": str(train_source_name),
            "val_source_name": str(val_source_name),
            "pre_filter_train_row_count": pre_filter_train_row_count,
            "pre_filter_val_row_count": pre_filter_val_row_count,
            "pre_filter_train_session_count": pre_filter_train_session_count,
            "pre_filter_val_session_count": pre_filter_val_session_count,
            "post_filter_train_row_count": post_filter_train_row_count,
            "post_filter_val_row_count": post_filter_val_row_count,
            "post_filter_train_session_count": post_filter_train_session_count,
            "post_filter_val_session_count": post_filter_val_session_count,
            "total_train_row_count": int(len(train_base_df)),
            "total_val_row_count": int(len(val_base_df)),
        },
    }


def prepare_behavior_fit_explicit_train_val_split(
    train_csv_path,
    val_csv_path,
    *,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    state_update_mode: str = "rollout",
    train_row_limit: Optional[int] = None,
    val_row_limit: Optional[int] = None,
):
    train_df = load_kuairec_small_matrix(train_csv_path, row_limit=train_row_limit)
    val_df = load_kuairec_small_matrix(val_csv_path, row_limit=val_row_limit)
    split = prepare_behavior_fit_explicit_train_val_split_from_dataframes(
        train_df=train_df,
        val_df=val_df,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
        train_source_name=Path(train_csv_path).name,
        val_source_name=Path(val_csv_path).name,
    )
    split["meta"]["train_row_limit"] = (
        None if train_row_limit is None else int(train_row_limit)
    )
    split["meta"]["val_row_limit"] = (
        None if val_row_limit is None else int(val_row_limit)
    )
    return split


def prepare_behavior_fit_train_val_split(
    csv_path,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    state_update_mode: str = "rollout",
):
    df = load_kuairec_small_matrix(csv_path)
    return _prepare_behavior_fit_train_val_split_from_dataframe(
        df=df,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )


def prepare_behavior_fit_train_val_splits_by_mode(
    csv_path,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    state_update_modes=("rollout", "teacher_forcing"),
):
    df = load_kuairec_small_matrix(csv_path)
    splits = {}
    for state_update_mode in state_update_modes:
        mode = _validate_state_update_mode(state_update_mode)
        splits[mode] = _prepare_behavior_fit_train_val_split_from_dataframe(
            df=df,
            train_start_index=train_start_index,
            train_row_count=train_row_count,
            val_row_count=val_row_count,
            min_session_length=min_session_length,
            state_update_mode=mode,
        )
    return splits


def _compute_watch_ratio_loss_terms(
    bucket_logits,
    pred_watch_ratio,
    y_watch_ratio,
    y_bucket,
    device,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
):
    bucket_ce_loss = F.cross_entropy(
        bucket_logits.unsqueeze(0),
        y_bucket.unsqueeze(0),
    )
    bucket_probs = torch.softmax(bucket_logits, dim=0)
    bucket_positions = torch.arange(
        NUM_WATCH_RATIO_BUCKETS,
        dtype=torch.float32,
        device=device,
    )
    bucket_distance_loss = torch.sum(
        bucket_probs * torch.square(bucket_positions - y_bucket.float())
    )
    value_l1_loss = torch.abs(pred_watch_ratio - y_watch_ratio)

    step_loss = (
        watch_ratio_bucket_ce_loss_weight * bucket_ce_loss +
        watch_ratio_bucket_distance_loss_weight * bucket_distance_loss +
        watch_ratio_value_loss_weight * value_l1_loss
    )

    return {
        "step_loss": step_loss,
        "bucket_ce_loss": bucket_ce_loss,
        "bucket_distance_loss": bucket_distance_loss,
        "value_l1_loss": value_l1_loss,
    }


def _session_length_compensation_weight(
    session_length: int,
    reference_rows: int = SESSION_LOSS_REFERENCE_ROWS,
) -> float:
    session_length = max(int(session_length), 1)
    reference_rows = max(int(reference_rows), 1)
    if session_length <= reference_rows:
        return 1.0
    return float(session_length) / float(reference_rows)


def _run_behavior_model_pass(
    model,
    df,
    feature_provider,
    device,
    optimizer=None,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
    optimizer_chunk_rows: Optional[int] = None,
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    # Training now always performs one optimizer step per session.
    # optimizer_chunk_rows is kept only for call-site compatibility.
    df = _prepare_sequence_dataframe(df, state_update_mode=state_update_mode)
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_bucket_ce_loss = 0.0
    total_bucket_distance_loss = 0.0
    total_value_l1_loss = 0.0
    total_steps = 0

    with torch.set_grad_enabled(optimizer is not None):
        for (user_id, session_id), session_df in df.groupby(["user_id", "session_id"], sort=False):
            del session_id
            user_feat = feature_provider.get_user_features(user_id)
            state = model.init_state(user_feat=user_feat, device=device)

            rows = list(session_df.itertuples(index=False))
            session_loss = None
            if optimizer is not None:
                optimizer.zero_grad()
                session_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            for row in rows:
                video_feat = feature_provider.get_video_features(
                    row.video_id,
                    user_id=user_id,
                )

                y_watch_ratio = torch.tensor(
                    float(row.watch_ratio),
                    dtype=torch.float32,
                    device=device,
                )
                y_bucket = torch.tensor(
                    watch_ratio_to_bucket_index(row.watch_ratio),
                    dtype=torch.long,
                    device=device,
                )

                bucket_logits, pred_watch_ratio, aux = model.forward_one(
                    state=state,
                    user_feat=user_feat,
                    video_feat=video_feat,
                    device=device,
                )

                loss_terms = _compute_watch_ratio_loss_terms(
                    bucket_logits=bucket_logits,
                    pred_watch_ratio=pred_watch_ratio,
                    y_watch_ratio=y_watch_ratio,
                    y_bucket=y_bucket,
                    device=device,
                    watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
                    watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
                    watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
                )

                if optimizer is not None:
                    session_loss = session_loss + loss_terms["step_loss"]

                total_loss += float(loss_terms["step_loss"].detach().cpu().item())
                total_bucket_ce_loss += float(loss_terms["bucket_ce_loss"].detach().cpu().item())
                total_bucket_distance_loss += float(
                    loss_terms["bucket_distance_loss"].detach().cpu().item()
                )
                total_value_l1_loss += float(loss_terms["value_l1_loss"].detach().cpu().item())
                total_steps += 1

                if state_update_mode == "rollout":
                    state = model.update_state_with_prediction(
                        state=state,
                        pred_watch_ratio=pred_watch_ratio,
                        aux=aux,
                    )
                else:
                    state = model.update_state_with_real_feedback(
                        state=state,
                        y_watch_ratio=y_watch_ratio,
                        y_bucket=y_bucket,
                        aux=aux,
                    )

            if optimizer is not None and hasattr(model, "compute_user_score_weight_regularization_loss"):
                user_reg_loss = model.compute_user_score_weight_regularization_loss(user_id=user_id)
                session_loss = session_loss + user_reg_loss
                total_loss += float(user_reg_loss.detach().cpu().item())

            if optimizer is not None and session_loss is not None:
                session_length = max(len(rows), 1)
                session_mean_loss = session_loss / float(session_length)
                session_weight = _session_length_compensation_weight(
                    session_length=session_length,
                    reference_rows=SESSION_LOSS_REFERENCE_ROWS,
                )
                compensated_session_loss = session_mean_loss * float(session_weight)
                compensated_session_loss.backward()
                optimizer.step()

            state = _detach_state_tree(state)

    return {
        "avg_loss": total_loss / max(total_steps, 1),
        "avg_bucket_ce": total_bucket_ce_loss / max(total_steps, 1),
        "avg_bucket_distance": total_bucket_distance_loss / max(total_steps, 1),
        "avg_value_l1": total_value_l1_loss / max(total_steps, 1),
        "steps": int(total_steps),
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()) if len(df) > 0 else 0,
        "sessions": int(df[["user_id", "session_id"]].drop_duplicates().shape[0]) if len(df) > 0 else 0,
    }


def _state_scalar_or_none(state, key):
    if key not in state:
        return None

    value = state[key]
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())

    return float(value)


def collect_dopamine_state_traces(
    model,
    df,
    feature_provider,
    state_update_mode="rollout",
    variant_name: Optional[str] = None,
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    df = _prepare_sequence_dataframe(df, state_update_mode=state_update_mode)
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    was_training = model.training
    model.eval()

    records = []
    eval_row_index = 0
    resolved_variant_name = str(variant_name or model.__class__.__name__)

    with torch.no_grad():
        for (user_id, session_id), session_df in df.groupby(["user_id", "session_id"], sort=False):
            user_feat = feature_provider.get_user_features(user_id)
            state = model.init_state(user_feat=user_feat, device=device)
            rows = list(session_df.itertuples(index=False))

            for row in rows:
                video_feat = feature_provider.get_video_features(
                    row.video_id,
                    user_id=user_id,
                )
                actual_watch_ratio = float(row.watch_ratio)
                actual_bucket = watch_ratio_to_bucket_index(actual_watch_ratio)

                bucket_logits, pred_watch_ratio, aux = model.forward_one(
                    state=state,
                    user_feat=user_feat,
                    video_feat=video_feat,
                    device=device,
                )
                del bucket_logits

                if state_update_mode == "rollout":
                    state_update_watch_ratio = pred_watch_ratio
                    next_state = model.update_state_with_prediction(
                        state=state,
                        pred_watch_ratio=pred_watch_ratio,
                        aux=aux,
                    )
                else:
                    y_watch_ratio = torch.tensor(
                        actual_watch_ratio,
                        dtype=torch.float32,
                        device=device,
                    )
                    y_bucket = torch.tensor(
                        actual_bucket,
                        dtype=torch.long,
                        device=device,
                    )
                    state_update_watch_ratio = y_watch_ratio
                    next_state = model.update_state_with_real_feedback(
                        state=state,
                        y_watch_ratio=y_watch_ratio,
                        y_bucket=y_bucket,
                        aux=aux,
                    )

                score_engagement = float(
                    (aux["video_score"] * torch.clamp(state_update_watch_ratio, min=0.0))
                    .detach()
                    .cpu()
                    .item()
                )

                records.append(
                    {
                        "variant_name": resolved_variant_name,
                        "state_update_mode": state_update_mode,
                        "eval_row_index": int(eval_row_index),
                        "user_id": int(row.user_id),
                        "session_id": int(getattr(row, "session_id", session_id)),
                        "session_row_index": int(getattr(row, "session_row_index", 0)),
                        "source_row_index": int(getattr(row, "source_row_index", -1)),
                        "video_id": int(row.video_id),
                        "timestamp": float(getattr(row, "timestamp", np.nan)),
                        "event_timestamp": float(getattr(row, "event_timestamp", np.nan)),
                        "actual_watch_ratio": actual_watch_ratio,
                        "actual_bucket": int(actual_bucket),
                        "state_update_watch_ratio": float(
                            state_update_watch_ratio.detach().cpu().item()
                        ),
                        "novelty_norm": float(aux["novelty_norm"].detach().cpu().item()),
                        "score_engagement": score_engagement,
                        "dopamine_before": _state_scalar_or_none(state, "dopamine"),
                        "dopamine_after": _state_scalar_or_none(next_state, "dopamine"),
                        "expected_reward_before": _state_scalar_or_none(
                            state, "expected_reward"
                        ),
                        "expected_reward_after": _state_scalar_or_none(
                            next_state, "expected_reward"
                        ),
                        "time_weighted_average_reward_before": _state_scalar_or_none(
                            state, "time_weighted_average_reward"
                        ),
                        "time_weighted_average_reward_after": _state_scalar_or_none(
                            next_state, "time_weighted_average_reward"
                        ),
                        "user_automaticity": _state_scalar_or_none(
                            state, "user_automaticity"
                        ),
                        "session_peak_before": _state_scalar_or_none(
                            state, "session_peak"
                        ),
                        "session_peak_after": _state_scalar_or_none(
                            next_state, "session_peak"
                        ),
                    }
                )
                eval_row_index += 1
                state = next_state

            state = _detach_state_tree(state)

    if was_training:
        model.train()

    return pd.DataFrame.from_records(records)


def collect_behavior_predictions(
    model,
    df,
    feature_provider,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    df = _prepare_sequence_dataframe(df, state_update_mode=state_update_mode)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    records = []
    eval_row_index = 0

    with torch.no_grad():
        for (user_id, session_id), session_df in df.groupby(["user_id", "session_id"], sort=False):
            user_feat = feature_provider.get_user_features(user_id)
            state = model.init_state(user_feat=user_feat, device=device)
            rows = list(session_df.itertuples(index=False))

            for row in rows:
                video_feat = feature_provider.get_video_features(
                    row.video_id,
                    user_id=user_id,
                )

                actual_watch_ratio = float(row.watch_ratio)
                actual_bucket = watch_ratio_to_bucket_index(actual_watch_ratio)
                fatigue_before = float(state["fatigue"].detach().cpu().item())
                dopamine_before = (
                    float(state["dopamine"].detach().cpu().item())
                    if "dopamine" in state
                    else None
                )
                session_peak_before = (
                    float(state["session_peak"].detach().cpu().item())
                    if "session_peak" in state
                    else None
                )
                expected_reward_before = (
                    float(state["expected_reward"].detach().cpu().item())
                    if "expected_reward" in state
                    else None
                )
                time_weighted_average_reward_before = (
                    float(state["time_weighted_average_reward"].detach().cpu().item())
                    if "time_weighted_average_reward" in state
                    else None
                )
                user_automaticity = (
                    float(state["user_automaticity"].detach().cpu().item())
                    if "user_automaticity" in state
                    else None
                )

                bucket_logits, pred_watch_ratio, aux = model.forward_one(
                    state=state,
                    user_feat=user_feat,
                    video_feat=video_feat,
                    device=device,
                )

                bucket_probs = torch.softmax(bucket_logits, dim=0)
                pred_bucket = int(torch.argmax(bucket_probs).detach().cpu().item())
                pred_bucket_prob = float(bucket_probs[pred_bucket].detach().cpu().item())
                pred_watch_ratio_value = float(pred_watch_ratio.detach().cpu().item())
                signed_error = pred_watch_ratio_value - actual_watch_ratio
                abs_error = abs(signed_error)

                record = {
                    "eval_row_index": int(eval_row_index),
                    "user_id": int(row.user_id),
                    "session_id": int(getattr(row, "session_id", session_id)),
                    "session_row_index": int(getattr(row, "session_row_index", 0)),
                    "video_id": int(row.video_id),
                    "timestamp": float(getattr(row, "timestamp", np.nan)),
                    "event_timestamp": float(getattr(row, "event_timestamp", np.nan)),
                    "date": float(getattr(row, "date", np.nan)),
                    "play_duration": float(getattr(row, "play_duration", np.nan)),
                    "video_duration": float(getattr(row, "video_duration", np.nan)),
                    "actual_watch_ratio": actual_watch_ratio,
                    "actual_bucket": int(actual_bucket),
                    "actual_bucket_label": bucket_index_to_label(actual_bucket),
                    "pred_watch_ratio": pred_watch_ratio_value,
                    "pred_bucket": int(pred_bucket),
                    "pred_bucket_label": bucket_index_to_label(pred_bucket),
                    "pred_bucket_prob": pred_bucket_prob,
                    "signed_error": float(signed_error),
                    "abs_error": float(abs_error),
                    "fatigue_before": fatigue_before,
                    "dopamine_before": dopamine_before,
                    "expected_reward_before": expected_reward_before,
                    "time_weighted_average_reward_before": time_weighted_average_reward_before,
                    "user_automaticity": user_automaticity,
                    "session_peak_before": session_peak_before,
                    "pref_match": float(aux["pref_match"].detach().cpu().item()),
                    "novelty_norm": float(aux["novelty_norm"].detach().cpu().item()),
                    "quality_norm": float(aux["quality_norm"].detach().cpu().item()),
                    "current_dopamine": (
                        float(aux["current_dopamine"].detach().cpu().item())
                        if "current_dopamine" in aux
                        else None
                    ),
                }
                if "creator_pref_norm" in video_feat:
                    record["creator_pref_norm"] = float(video_feat["creator_pref_norm"])

                records.append(record)
                eval_row_index += 1

                y_watch_ratio = torch.tensor(
                    actual_watch_ratio,
                    dtype=torch.float32,
                    device=device,
                )
                y_bucket = torch.tensor(
                    actual_bucket,
                    dtype=torch.long,
                    device=device,
                )
                if state_update_mode == "rollout":
                    state = model.update_state_with_prediction(
                        state=state,
                        pred_watch_ratio=pred_watch_ratio,
                        aux=aux,
                    )
                else:
                    state = model.update_state_with_real_feedback(
                        state=state,
                        y_watch_ratio=y_watch_ratio,
                        y_bucket=y_bucket,
                        aux=aux,
                    )

            state = _detach_state_tree(state)

    if was_training:
        model.train()

    return pd.DataFrame.from_records(records)


def _build_behavior_fit_model(
    *,
    model_cls,
    feature_provider,
    train_df,
    device,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    user_ids = []
    if "user_id" in train_df.columns:
        user_ids = sorted(
            int(user_id)
            for user_id in pd.Series(train_df["user_id"]).dropna().unique().tolist()
        )

    init_signature = inspect.signature(model_cls.__init__)
    if "user_ids" in init_signature.parameters:
        model = model_cls(
            embedding_dim=feature_provider.embedding_dim,
            user_ids=user_ids,
        ).to(device)
    else:
        model = model_cls(embedding_dim=feature_provider.embedding_dim).to(device)

    if hasattr(model, "set_user_score_weight_regularization_stats"):
        model.set_user_score_weight_regularization_stats(
            train_df=train_df,
            state_update_mode=state_update_mode,
        )
    return model


def load_behavior_model_checkpoint(
    *,
    model_cls,
    feature_provider,
    train_df,
    checkpoint_path,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_behavior_fit_model(
        model_cls=model_cls,
        feature_provider=feature_provider,
        train_df=train_df,
        device=device,
        state_update_mode=state_update_mode,
    )
    state_dict = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def fit_behavior_model_on_split(
    model_cls,
    feature_provider,
    train_df,
    val_df=None,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    model_name=None,
    state_update_mode="rollout",
    optimizer_chunk_rows: Optional[int] = None,
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_behavior_fit_model(
        model_cls=model_cls,
        feature_provider=feature_provider,
        train_df=train_df,
        device=device,
        state_update_mode=state_update_mode,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    label = model_name or model_cls.__name__

    for epoch in range(num_epochs):
        train_metrics = _run_behavior_model_pass(
            model=model,
            df=train_df,
            feature_provider=feature_provider,
            device=device,
            optimizer=optimizer,
            watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
            watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
            watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
            state_update_mode=state_update_mode,
            optimizer_chunk_rows=optimizer_chunk_rows,
        )

        epoch_record = {
            "epoch": int(epoch + 1),
            "train": train_metrics,
        }

        msg = (
            f"[{label}|{state_update_mode}] Epoch {epoch+1:03d} | "
            f"train_loss={train_metrics['avg_loss']:.6f} | "
            f"train_bucket_ce={train_metrics['avg_bucket_ce']:.6f} | "
            f"train_bucket_dist={train_metrics['avg_bucket_distance']:.6f} | "
            f"train_value_l1={train_metrics['avg_value_l1']:.6f}"
        )

        if val_df is not None and len(val_df) > 0:
            val_metrics = _run_behavior_model_pass(
                model=model,
                df=val_df,
                feature_provider=feature_provider,
                device=device,
                optimizer=None,
                watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
                watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
                watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
                state_update_mode=state_update_mode,
                optimizer_chunk_rows=optimizer_chunk_rows,
            )
            epoch_record["val"] = val_metrics
            msg += (
                f" | val_loss={val_metrics['avg_loss']:.6f}"
                f" | val_bucket_ce={val_metrics['avg_bucket_ce']:.6f}"
                f" | val_bucket_dist={val_metrics['avg_bucket_distance']:.6f}"
                f" | val_value_l1={val_metrics['avg_value_l1']:.6f}"
            )

        history.append(epoch_record)
        print(msg)

    return model, model.export_env_params(), history


def compare_behavior_models_on_kuairec(
    csv_path,
    feature_provider,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    split = prepare_behavior_fit_train_val_split(
        csv_path=csv_path,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )

    dopamine_model, dopamine_params, dopamine_history = fit_behavior_model_on_split(
        model_cls=BehaviorFitModel,
        feature_provider=feature_provider,
        train_df=split["train"],
        val_df=split["val"],
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="dopamine",
        state_update_mode=state_update_mode,
    )
    no_dopamine_model, no_dopamine_params, no_dopamine_history = fit_behavior_model_on_split(
        model_cls=NoDopamineBehaviorFitModel,
        feature_provider=feature_provider,
        train_df=split["train"],
        val_df=split["val"],
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="no_dopamine",
        state_update_mode=state_update_mode,
    )

    result = {
        "split": split["meta"],
        "state_update_mode": state_update_mode,
        "dopamine": {
            "model": dopamine_model,
            "learned_params": dopamine_params,
            "history": dopamine_history,
        },
        "no_dopamine": {
            "model": no_dopamine_model,
            "learned_params": no_dopamine_params,
            "history": no_dopamine_history,
        },
    }

    if len(split["val"]) > 0:
        dopamine_val = dopamine_history[-1]["val"]
        no_dopamine_val = no_dopamine_history[-1]["val"]
        result["summary"] = {
            "dopamine_val_loss": float(dopamine_val["avg_loss"]),
            "no_dopamine_val_loss": float(no_dopamine_val["avg_loss"]),
            "val_loss_gap_no_dopamine_minus_dopamine": float(
                no_dopamine_val["avg_loss"] - dopamine_val["avg_loss"]
            ),
        }

    return result


def compare_content_only_fatigue_model_on_kuairec(
    csv_path,
    feature_provider,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    split = prepare_behavior_fit_train_val_split(
        csv_path=csv_path,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )

    dopamine_model, dopamine_params, dopamine_history = fit_behavior_model_on_split(
        model_cls=BehaviorFitModel,
        feature_provider=feature_provider,
        train_df=split["train"],
        val_df=split["val"],
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="dopamine",
        state_update_mode=state_update_mode,
    )
    content_only_model, content_only_params, content_only_history = fit_behavior_model_on_split(
        model_cls=ContentOnlyFatigueBehaviorFitModel,
        feature_provider=feature_provider,
        train_df=split["train"],
        val_df=split["val"],
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="content_only_fatigue",
        state_update_mode=state_update_mode,
    )

    result = {
        "split": split["meta"],
        "state_update_mode": state_update_mode,
        "dopamine": {
            "model": dopamine_model,
            "learned_params": dopamine_params,
            "history": dopamine_history,
        },
        "content_only_fatigue": {
            "model": content_only_model,
            "learned_params": content_only_params,
            "history": content_only_history,
        },
    }

    if len(split["val"]) > 0:
        dopamine_val = dopamine_history[-1]["val"]
        content_only_val = content_only_history[-1]["val"]
        result["summary"] = {
            "dopamine_val_loss": float(dopamine_val["avg_loss"]),
            "content_only_fatigue_val_loss": float(content_only_val["avg_loss"]),
            "val_loss_gap_content_only_fatigue_minus_dopamine": float(
                content_only_val["avg_loss"] - dopamine_val["avg_loss"]
            ),
        }

    return result


def evaluate_dopamine_ablation_on_kuairec(
    csv_path,
    feature_provider,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    split = prepare_behavior_fit_train_val_split(
        csv_path=csv_path,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )

    model, baseline_learned_params, history = fit_behavior_model_on_split(
        model_cls=BehaviorFitModel,
        feature_provider=feature_provider,
        train_df=split["train"],
        val_df=split["val"],
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="dopamine",
        state_update_mode=state_update_mode,
    )

    device = next(model.parameters()).device
    baseline_val = _run_behavior_model_pass(
        model=model,
        df=split["val"],
        feature_provider=feature_provider,
        device=device,
        optimizer=None,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        state_update_mode=state_update_mode,
    )

    ablated_model = model.make_strict_zero_dopamine_eval_copy()
    ablated_val = _run_behavior_model_pass(
        model=ablated_model,
        df=split["val"],
        feature_provider=feature_provider,
        device=device,
        optimizer=None,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        state_update_mode=state_update_mode,
    )

    return {
        "split": split["meta"],
        "state_update_mode": state_update_mode,
        "baseline_learned_params": baseline_learned_params,
        "ablated_param_keys": sorted(DOPAMINE_PARAM_KEYS),
        "baseline_val": baseline_val,
        "ablated_val": ablated_val,
        "delta": {
            "avg_loss": float(ablated_val["avg_loss"] - baseline_val["avg_loss"]),
            "avg_bucket_ce": float(
                ablated_val["avg_bucket_ce"] - baseline_val["avg_bucket_ce"]
            ),
            "avg_bucket_distance": float(
                ablated_val["avg_bucket_distance"] - baseline_val["avg_bucket_distance"]
            ),
            "avg_value_l1": float(
                ablated_val["avg_value_l1"] - baseline_val["avg_value_l1"]
            ),
        },
        "history": history,
    }


# =========================================================
# 4. Fit user model from KuaiRec
# =========================================================
def fit_user_model_from_kuairec(
    csv_path,
    feature_provider,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    split = prepare_behavior_fit_train_val_split(
        csv_path=csv_path,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )
    df = split["train"]
    print(
        "training subset loaded:",
        len(df),
        "rows across",
        split["meta"]["actual_train_session_count"],
        "sessions",
    )
    model, learned_params, _ = fit_behavior_model_on_split(
        model_cls=BehaviorFitModel,
        feature_provider=feature_provider,
        train_df=df,
        val_df=None,
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="dopamine",
        state_update_mode=state_update_mode,
    )
    return model, learned_params, df


def fit_content_only_fatigue_user_model_from_kuairec(
    csv_path,
    feature_provider,
    train_start_index=DEFAULT_TRAIN_START_INDEX,
    train_row_count=DEFAULT_TRAIN_ROW_COUNT,
    val_row_count=DEFAULT_VAL_ROW_COUNT,
    min_session_length: Optional[int] = DEFAULT_MIN_SESSION_LENGTH,
    num_epochs=5,
    lr=1e-3,
    watch_ratio_value_loss_weight=1.0,
    watch_ratio_bucket_ce_loss_weight=1.0,
    watch_ratio_bucket_distance_loss_weight=1.0,
    state_update_mode="rollout",
):
    state_update_mode = _validate_state_update_mode(state_update_mode)
    split = prepare_behavior_fit_train_val_split(
        csv_path=csv_path,
        train_start_index=train_start_index,
        train_row_count=train_row_count,
        val_row_count=val_row_count,
        min_session_length=min_session_length,
        state_update_mode=state_update_mode,
    )
    df = split["train"]
    print(
        "training subset loaded:",
        len(df),
        "rows across",
        split["meta"]["actual_train_session_count"],
        "sessions",
    )
    model, learned_params, _ = fit_behavior_model_on_split(
        model_cls=ContentOnlyFatigueBehaviorFitModel,
        feature_provider=feature_provider,
        train_df=df,
        val_df=None,
        num_epochs=num_epochs,
        lr=lr,
        watch_ratio_value_loss_weight=watch_ratio_value_loss_weight,
        watch_ratio_bucket_ce_loss_weight=watch_ratio_bucket_ce_loss_weight,
        watch_ratio_bucket_distance_loss_weight=watch_ratio_bucket_distance_loss_weight,
        model_name="content_only_fatigue",
        state_update_mode=state_update_mode,
    )
    return model, learned_params, df
