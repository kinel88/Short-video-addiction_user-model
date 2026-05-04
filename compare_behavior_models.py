from pathlib import Path

from env_module import make_env, make_no_dopamine_env
from train_module import (
    DEFAULT_MIN_SESSION_LENGTH,
    DEFAULT_TRAIN_ROW_COUNT,
    DEFAULT_TRAIN_START_INDEX,
    DEFAULT_VAL_ROW_COUNT,
    NpyFeatureProvider,
    compare_behavior_models_on_kuairec,
)


TRAIN_START_INDEX = DEFAULT_TRAIN_START_INDEX
TRAIN_ROW_COUNT = DEFAULT_TRAIN_ROW_COUNT
VAL_ROW_COUNT = DEFAULT_VAL_ROW_COUNT
MIN_SESSION_LENGTH = DEFAULT_MIN_SESSION_LENGTH
STATE_UPDATE_MODE = "rollout"


def main():
    project_root = Path(__file__).resolve().parent
    assignment_root = project_root.parent
    kuairec_root = assignment_root / "KuaiRec"
    kuairec_data_root = kuairec_root / "KuaiRec 2.0" / "data"

    feature_provider = NpyFeatureProvider(
        user_embedding_path=kuairec_root / "user_embeddings.npy",
        item_embedding_path=kuairec_root / "item_embeddings.npy",
        user_id_map_path=kuairec_root / "user_id_map.csv",
        item_id_map_path=kuairec_root / "item_id_map.csv",
        item_daily_path=kuairec_data_root / "item_daily_features.csv",
        creator_preference_path=project_root / "vector_outputs" / "user_author_preference_dense.csv",
    )

    result = compare_behavior_models_on_kuairec(
        csv_path=kuairec_data_root / "small_matrix.csv",
        feature_provider=feature_provider,
        train_start_index=TRAIN_START_INDEX,
        train_row_count=TRAIN_ROW_COUNT,
        val_row_count=VAL_ROW_COUNT,
        min_session_length=MIN_SESSION_LENGTH,
        num_epochs=5,
        lr=1e-3,
        watch_ratio_value_loss_weight=1.0,
        watch_ratio_bucket_ce_loss_weight=1.0,
        watch_ratio_bucket_distance_loss_weight=1.0,
        state_update_mode=STATE_UPDATE_MODE,
    )

    print("\n===== Comparison Split =====")
    for key, value in result["split"].items():
        print(f"{key}: {value}")
    print(f"state_update_mode: {result['state_update_mode']}")

    if "summary" in result:
        print("\n===== Validation Loss Comparison =====")
        print(f"dopamine_val_loss: {result['summary']['dopamine_val_loss']:.6f}")
        print(f"no_dopamine_val_loss: {result['summary']['no_dopamine_val_loss']:.6f}")
        print(
            "val_loss_gap_no_dopamine_minus_dopamine: "
            f"{result['summary']['val_loss_gap_no_dopamine_minus_dopamine']:.6f}"
        )

    _ = make_env(
        feature_provider=feature_provider,
        num_candidates=5,
        slate_size=1,
        seed=0,
        behavior_params=result["dopamine"]["learned_params"],
    )
    _ = make_no_dopamine_env(
        feature_provider=feature_provider,
        num_candidates=5,
        slate_size=1,
        seed=0,
        behavior_params=result["no_dopamine"]["learned_params"],
    )
    print("\nBoth dopamine and no-dopamine environments were built successfully.")
