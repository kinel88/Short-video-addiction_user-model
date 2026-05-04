# AGENTS

## WSL `recsim` Environment

This project's `main.py` should be run from WSL, not from the native Windows Python environment.

Validated environment:

- WSL is available
- Conda is installed at `/home/drsqy/miniconda3`
- The working Conda environment is `recsim_fix`
- `recsim` can be imported successfully inside `recsim_fix`

## Manual Run Steps

In PowerShell:

```powershell
wsl
```

Then inside WSL:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recsim_fix
cd "/mnt/c/Users/drsqy/Desktop/伯克利/媒体算法设计/论文作业/工程实现文件-recsim"
python main.py
```

## One-Line Run From PowerShell

If you want to run it directly from PowerShell without entering an interactive WSL shell first:

```powershell
wsl -e bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate recsim_fix && cd '/mnt/c/Users/drsqy/Desktop/伯克利/媒体算法设计/论文作业/工程实现文件-recsim' && python main.py"
```

## Smoke Test

To verify only that the `recsim` package is reachable in WSL:

```powershell
wsl -e bash /mnt/c/Users/drsqy/Desktop/伯克利/媒体算法设计/论文作业/工程实现文件-recsim/run_wsl_recsim_import_check.sh
```

Expected output includes:

```text
RECSIM_IMPORT_OK
```

## Notes

- `main.py` is the single default experiment entrypoint.
- Running `main.py` now:
  - trains only the `rollout` chain
  - uses full `big_matrix.csv` as training data and full `small_matrix.csv` as validation data
  - tunes the final loss weights by validation `mean_abs_error` on `small_matrix.csv`
  - automatically reuses the saved `rollout` cache when the strict fingerprint over code, data, and vector dependencies still matches
  - writes the final rollout validation report under `prediction_reports/`
  - persists cache artifacts under `prediction_reports/`:
    - `rollout_training_cache_meta.json`
    - `rollout_learned_params.json`
    - `rollout_model_state.pt`
  - builds the online environment with the learned `rollout` parameters and performs an `env.reset()` smoke test
- both train and validation paths apply `min_session_length = 20`
- the final reported watch-ratio validation metric is rollout `mean_abs_error`
- `main.py` may run for a while because it performs training before building the environment.
- A Gym deprecation warning may appear during import; it is currently a warning, not the blocking issue.

## Current Behavior Learning Objective

The current user-behavior fitting objective in `train_module.py` is no longer the old binary-interaction / capped-watch setup.

Current training target now learns two watch-ratio outputs per interaction:

- a non-negative continuous prediction: `pred_watch_ratio`
- a 6-way interval probability prediction over:
  - `[0.0, 0.35)`
  - `[0.35, 0.6)`
  - `[0.6, 0.8)`
  - `[0.8, 1.0)`
  - `[1.0, 1.3]`
  - `> 1.3`

Current loss terms:

- `bucket_ce_loss = cross_entropy(bucket_logits, true_bucket)`
- `bucket_distance_loss = sum_j p_j * (j - true_bucket)^2`
- `value_l1_loss = abs(pred_watch_ratio - true_watch_ratio)`
- current bucket head is bias-free and uses `[pref_match, quality_norm, novelty_norm, creator_preference_norm, expected_reward, user_automaticity, fatigue]` as its input features

Current total objective:

- `total_loss = w_bucket_ce * bucket_ce_loss + w_bucket_dist * bucket_distance_loss + w_value * value_l1_loss`

Current default weight-selection flow from the unified `main.py` experiment:

- rollout-only loss-weight tuning selects the final weights by validation `mean_abs_error`
- the current candidate grid is:
  - `value=2.0, bucket_ce=1.0, bucket_dist=1.0`
  - `value=4.0, bucket_ce=1.0, bucket_dist=1.0`
  - `value=6.0, bucket_ce=1.0, bucket_dist=1.0`
  - `value=4.0, bucket_ce=0.5, bucket_dist=1.0`
  - `value=4.0, bucket_ce=1.0, bucket_dist=0.5`
  - `value=4.0, bucket_ce=0.5, bucket_dist=0.5`
- tuning runs only `rollout` during candidate selection, then trains final `rollout` and `teacher_forcing` models with the selected weights

Current watch-ratio handling rules:

- do not reintroduce the old `<= 1.5` training cap
- continuous `watch_ratio` supervision is trained on the non-negative real value after capping any `watch_ratio > 10.0` down to `10.0`
- the bucket target and continuous target are both derived from the same real `watch_ratio`

Current interpretation / implementation notes:

- teacher-forced hidden-state updates in `train_module.py` no longer depend on any legacy binary-interaction proxy
- session-start `dopamine` is now a single shared learned/exported constant `dopamine_base`; it is no longer derived per-user from the embedding
- each known training user has a learned/exported `user_automaticity` scalar; unknown users fall back to `user_automaticity_default`
- each known training user also has learned/exported personalized `video_score` weights for:
  - `preference_match`
  - `quality`
  - `novelty`
  - `creator_preference`
- those four score weights are trained as `shared global baseline + per-user offset`; unknown users fall back to the shared global baseline
- current online / teacher-forced behavior logic is:
  - current video score:
    - `video_score = w_pref * preference_match + w_quality * quality_norm + w_novelty * novelty_norm + w_creator * creator_preference_norm`
  - current-video dopamine effect:
    - current interaction uses `current_dopamine = state_dopamine` without an extra pre-watch quality boost
    - `current_dopamine` is retained for state tracing / compatibility, but it no longer enters the watch-ratio threshold
    - the current video's direct dopamine contribution is applied only after the watch finishes, via `score_engagement = video_score * max(watch_ratio, 0.0)`, and then affects the next interaction through `expected_reward` and `session_peak`
    - the model explicitly tracks two dopamine reference levels:
      - `dopamine_normal_level`: higher daily / non-scrolling reference level
      - `dopamine_base`: lower fixed session baseline used during the scrolling session itself
    - after each watch, dopamine is updated by:
      - a reward drive from `expected_reward` and `decayed_score_peak`
      - multiplied by a slow in-session habit coefficient driven only by swipe count
      - plus a return-to-session-baseline term that pulls dopamine back toward `dopamine_base`
    - the habit coefficient is session-local, monotone increasing, saturating, and intentionally slow so it does not approach its ceiling after only a handful of swipes
  - expected reward:
    - history stores prior `score_engagement = video_score * max(watch_ratio, 0.0)`
    - `time_weighted_average_reward = sum_i score_engagement_i * average_reward_decay^age_i / sum_i average_reward_decay^age_i`
    - `expected_reward = expected_reward_peak_coef * session_peak + expected_reward_average_coef * time_weighted_average_reward`
  - decayed score peak:
    - history stores prior `score_engagement`
    - `decayed_score_peak = max(score_engagement_t * peak_decay^age_t)`
    - current `session_peak` enters `expected_reward`, and then still enters the dopamine reward drive through that expected-reward path and the explicit peak coefficient
  - effective-watch threshold:
    - `effective_threshold = effective_threshold_base + effective_threshold_expected_reward * expected_reward + effective_threshold_automaticity * user_automaticity`
    - `expected_reward` and `user_automaticity` raise the effective-watch threshold through non-negative learned weights
    - `dopamine` no longer raises the effective-watch threshold
    - `fatigue` no longer enters the threshold term
  - watch-duration mechanism:
    - current watch-ratio generation uses a nonlinear threshold-ratio mapping plus repeated-view accumulation
    - let `threshold_ratio = video_score / effective_threshold`
    - if `threshold_ratio < 0.75`, the first-pass mapped watch rises inside `0.0-0.1` using a `smoothstep` curve
    - if `0.75 <= threshold_ratio < 0.90`, the first-pass mapped watch rises inside the first bucket `[0.1, 0.35)` using a `smoothstep` curve
    - if `0.90 <= threshold_ratio < 1.10`, the first-pass mapped watch rises inside the second bucket `[0.35, 0.6)` using a `smoothstep` curve
    - if `threshold_ratio >= 1.10`, the first-pass mapped watch starts from `0.6` and then grows according to `relu(softplus(watch_gain_base * (video_score - 1.10 * effective_threshold)) - log(2))`
    - `fatigue` compresses watch duration for all watches
    - `dopamine` no longer directly compresses watch duration
    - if one pass yields `watch_ratio > 1.0`, the user is interpreted as rewatching
    - each repeated pass adds at most `1.0` watch ratio, decays the video score by `repeat_watch_decay`, and the total repeated-view accumulation is capped at `10` passes
  - fatigue update:
    - `fatigue` grows with `fatigue_watch_ratio_coef * max(watch_ratio, 0.0)`
    - `high_engaged = 1[watch_ratio > 1.0]` adds an extra fatigue jump
  - swipe / habit update:
    - every interaction increments a session-local swipe counter by `1`, regardless of `watch_ratio`
    - that counter affects only the dopamine reward coefficient; it does not modify the session baseline
  - novelty history:
    - novelty compares the current video's first `31` category dims against the recent time-window videos, not only previously consumed videos
- `env_module.py` now uses the same new-path behavior logic:
  - response fields: `watch_ratio`, `watch_ratio_bucket`, `high_engaged`, `watch_time`
  - candidate documents are sampled from the real video pool instead of random Gaussian embeddings
- the repo also exposes a separate parallel content-only path:
  - training model: `ContentOnlyFatigueBehaviorFitModel`
  - environment builder: `make_content_only_fatigue_env(...)`
  - semantics:
    - `video_score` uses the same `preference_match + quality + novelty + creator_preference` composition
    - `watch_ratio` keeps the same nonlinear threshold-ratio mapping plus repeated-view accumulation
    - `effective_threshold` is a learned fixed `effective_threshold_base` only
    - `fatigue` only compresses watch duration
    - this path does not track `dopamine`, `expected_reward`, `session_peak`, or habit / swipe state
  - unlike the older `NoDopamine` baseline, this path intentionally exports no `effective_threshold_fatigue`
- if the watch-ratio bucket scheme, distance penalty, or loss weights change, update both `train_module.py` and this file together

## Current Embedding Encoding

The current exported embeddings live under:

- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec`

Current exported dimensions:

- `item_dim = 159`
- `user_dim = 159`

### Item Embeddings

Current item vectors follow:

- `final_item_embedding = [content_item_side ; item_latent]`

Current block sizes:

- `content_item_side = 95`
- `item_latent = 64`

Current `content_item_side` keeps only:

- `31` dims from `item_categories.csv -> feat -> multi-hot`
- `64` dims from text fields in `kuairec_caption_category.csv` after `TF-IDF + SVD`

Text fields used:

- `manual_cover_text`
- `caption`
- `topic_tag`
- `first_level_category_name`
- `second_level_category_name`
- `third_level_category_name`

`item_latent` is learned from the interaction matrix built from:

- default interaction source: full `big_matrix.csv`
- the main embedding build no longer applies a train/val/test split to this interaction source
- default weight: `clip(watch_ratio, 0.0, 5.0) - 0.1`
- the `-0.1` term treats the first bit of exposure as neutral rather than positive preference
- interactions with `watch_ratio < 0.1` therefore contribute near-zero or slightly negative feedback by default
- legacy non-default weighting modes may still use `play_duration` / `video_duration` derived watch-quality flags

using shared `TruncatedSVD`.

### Removed Item Content Features

The following fields are intentionally excluded from the current `content_item_side`:

- z-score style numeric/id fields removed:
  - `author_id`
  - `video_duration`
  - `video_width`
  - `video_height`
  - `music_id`
  - `video_tag_id`
- one-hot style categorical fields removed:
  - `video_type`
  - `upload_dt`
  - `upload_type`
  - `visible_status`
  - `video_tag_name`

### User Embeddings

Current users follow:

- `final_user_embedding = [hist_user_content_pref ; user_latent]`

Current block sizes:

- content-aligned block = `95`
- latent block = `64`

For all exported users:

- exported users are restricted to `user_id` values that actually appear in `big_matrix.csv`
- `hist_user_content_pref` is a weighted average of historical interacted item `content_item_side`
- historical interactions come from the full `big_matrix.csv` dataset, not a train/val/test subset
- per-interaction weight is `clip(watch_ratio, 0.0, 5.0) - 0.1`
- aggregation is `sum_t(w_t * content_item_side(item_t)) / sum_t(w_t)`
- if `sum_t(w_t) == 0`, the history block falls back to a zero vector
- repeated watches are counted repeatedly; there is no per-item deduplication before averaging
- the aggregated history block is row-wise L2 normalized before concatenation
- the `95` content-aligned dims therefore mean:
  - `31` dims of historical category preference
  - `64` dims of historical text-semantic preference
- `user_latent` is the `64`-dim shared collaborative latent from the interaction matrix

### Behavior Context

`behavior_item_context.npy` is still exported separately, but it is not part of the default main item vector.

- current dimension: `145`
- source: recent 30-day `item_daily_features.csv` behavior aggregates
- structure:
  - `45 x 3 = 135` dims from `mean / std / max`
  - `10` dims from `per_show` ratios

### Practical Interpretation

The current main-path embedding design can be summarized as:

- item = `category multi-hot + text semantics + collaborative latent`
- user = `historical content preference + collaborative latent`

## Quality Index Encoding / Reading Rules

The project now exports a video quality index derived from daily item behavior statistics in `item_daily_features.csv`.

Current output files:

- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec\item_quality_index.csv`
- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\KuaiRec\item_id_map.csv`

Quality-related columns currently exported:

- `quality_index`
- `quality_index_raw`
- `quality_confidence`
- `quality_completion_signal`
- `quality_positive_interaction_signal`
- `quality_like_signal`
- `quality_negative_signal`

Current reading rules:

- If you need a standalone per-video quality table, read `item_quality_index.csv`.
- If you need quality values aligned with `item_embeddings.npy`, read `item_id_map.csv` and use its `row_index <-> video_id` mapping.
- `train_module.NpyFeatureProvider.get_video_features(...)` already reads `quality_index` and `quality_confidence` from `item_id_map.csv` when those columns exist.
- Do not assume `quality_index` is an absolute physical score; it is a relative percentile-style ranking on `[0, 100]`.
- Exposure is not used as a direct positive quality term; it is only used to compute `quality_confidence`.

Current quality definition:

- `completion_signal = 0.50 * complete_play_per_play + 0.25 * valid_play_per_play + 0.15 * play_progress_mean + 0.10 * play_per_show`
- `positive_interaction_signal = 0.35 * follow_per_show + 0.30 * collect_per_show + 0.25 * share_per_show + 0.10 * comment_per_show`
- `negative_signal = 0.60 * report_per_show + 0.20 * reduce_similar_per_show + 0.20 * cancel_follow_per_show`
- `like_signal = like_per_show`
- `quality_index_raw = 0.50 * z(completion_signal) + 2.00 * z(positive_interaction_signal) + 1.50 * z(like_signal) - 5.00 * z(negative_signal)`
- `quality_confidence = percentile_rank(log1p(show_cnt_mean))`
- confidence-adjusted raw score: `quality_index_raw = quality_index_raw * (0.35 + 0.65 * quality_confidence)`
- final score: `quality_index = percentile_rank(quality_index_raw) * 100`

Implementation/source-of-truth notes:

- The source-of-truth implementation lives in `build_vectors.py`.
- When quality weights or formulas are changed, regenerate outputs by rerunning `python build_vectors.py` inside the WSL `recsim_fix` environment.
- If there is ever a mismatch between this document and code/output files, trust the latest `build_vectors.py` implementation and then update this file.

## Creator Preference Table Encoding / Reading Rules

The project now exports a user-to-creator preference table derived from `small_matrix.csv` user-video interactions plus `author_id` from `item_daily_features.csv`.

Current output files:

- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\工程实现文件-recsim\vector_outputs\user_author_preference_dense.csv`
- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\工程实现文件-recsim\vector_outputs\user_author_preference_observed.csv`
- `C:\Users\drsqy\Desktop\伯克利\媒体算法设计\论文作业\工程实现文件-recsim\vector_outputs\user_author_preference_meta.json`

Current coverage:

- users covered: `1411`
- creators covered: `2031`
- source interaction rows: `4676570`
- source creator universe is restricted to creators whose videos appear in `small_matrix.csv`

Current output meanings:

- `user_author_preference_dense.csv`
  - full `user_id x author_id` table over creators that appear in `small_matrix.csv`
  - columns:
    - `user_id`
    - `author_id`
    - `creator_preference_score`
  - if one `(user_id, author_id)` pair never appears in history, then `creator_preference_score = 0.0`
  - this `0.0` is the score baseline for an unseen creator relation; it does **not** mean `author_id = 0`

- `user_author_preference_observed.csv`
  - only keeps `(user_id, author_id)` pairs that were actually observed through watched videos
  - columns:
    - `user_id`
    - `author_id`
    - `creator_preference_score`
    - `interaction_count`
    - `positive_interaction_count`
    - `negative_interaction_count`
    - `positive_score_sum`
    - `negative_score_abs_sum`
    - `watch_ratio_sum`
    - `effective_watch_count`
    - `valid_watch_count`
    - `complete_watch_count`
    - `short_watch_count`
    - `negative_watch_count`
    - `avg_watch_ratio`
    - `avg_preference_per_interaction`

- `user_author_preference_meta.json`
  - stores data source paths, output paths, current counts, and the exact scoring rule used for export

Current reading rules:

- If you need a full table where unseen creators default to `0.0`, read `user_author_preference_dense.csv`.
- If you need only observed creator relations plus diagnostics explaining why a score is high or low, read `user_author_preference_observed.csv`.
- If you need the formula, counts, or provenance for the exported table, read `user_author_preference_meta.json`.
- If you need a matrix view for a model, pivot `user_author_preference_dense.csv` by `user_id` and `author_id`.
- If there is ever a mismatch between this document and the exported files, trust the latest `build_creator_preference.py` implementation and then update this file.

Current creator preference definition:

- video-to-creator mapping:
  - each interaction first maps `video_id -> author_id` using `item_daily_features.csv`
  - current data check shows each `video_id` maps to exactly one `author_id`

- per-interaction score:
  - `interaction_score = clip(watch_ratio, 0, 2) + 0.4 * is_effective_watch + 0.8 * is_valid_watch + 0.8 * is_complete_watch - 0.8 * is_short_watch - 0.6 * is_negative_watch`

- indicator definitions:
  - `is_effective_watch = 1[watch_ratio >= 0.2]`
  - `is_valid_watch = 1[(video_duration <= 7000 and play_duration >= video_duration) or (video_duration > 7000 and play_duration > 7000)]`
  - `is_complete_watch = 1[play_duration >= video_duration and video_duration > 0]`
  - `is_short_watch = 1[play_duration < min(3000, max(video_duration, 1))]`
  - `is_negative_watch = 1[watch_ratio < 0.2]`

- user-creator score:
  - `creator_preference_score = sum(interaction_score over all watched videos of that creator by that user)`
  - more strong positive interactions push the score upward
  - weak watches / fast skips push the score downward
  - unseen creator relations stay at `0.0`

Current limitation / interpretation note:

- `small_matrix.csv` does **not** contain per-user explicit `like`, `dislike`, `report`, or `reduce_similar` events.
- Therefore the current exported negative preference is inferred only from weak watch behavior, not from explicit dislike/report actions.
- Do not interpret a negative `creator_preference_score` as proof that the user explicitly reported or disliked that creator.

Implementation/source-of-truth notes:

- The source-of-truth implementation lives in `build_creator_preference.py`.
- To regenerate the creator preference tables, run `python build_creator_preference.py` inside the WSL `recsim_fix` environment from the project root.
- If the scoring rule changes, regenerate all three creator preference outputs and then update this file.
