from pathlib import Path

from train_module import (
    DEFAULT_MIN_SESSION_LENGTH,
    DEFAULT_TRAIN_ROW_COUNT,
    DEFAULT_TRAIN_START_INDEX,
    DEFAULT_VAL_ROW_COUNT,
    NpyFeatureProvider,
    evaluate_dopamine_ablation_on_kuairec,
)


TRAIN_START_INDEX = DEFAULT_TRAIN_START_INDEX
TRAIN_ROW_COUNT = DEFAULT_TRAIN_ROW_COUNT
VAL_ROW_COUNT = DEFAULT_VAL_ROW_COUNT
MIN_SESSION_LENGTH = DEFAULT_MIN_SESSION_LENGTH
STATE_UPDATE_MODE = "rollout"


def _format_param(key, value):
    if isinstance(value, dict):
        values = [float(v) for v in value.values()]
        if len(values) == 0:
            return f"{key}: count=0"
        return (
            f"{key}: count={len(values)} "
            f"min={min(values):.6f} "
            f"max={max(values):.6f} "
            f"mean={sum(values) / len(values):.6f}"
        )
    return f"{key}: {float(value):.6f}"


def _print_metric_block(title, metrics):
    print(f"\n===== {title} =====")
    ordered_keys = [
        "avg_loss",
        "avg_bucket_ce",
        "avg_bucket_distance",
        "avg_value_l1",
        "steps",
        "rows",
        "users",
    ]
    for key in ordered_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


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

    result = evaluate_dopamine_ablation_on_kuairec(
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

    print("\n===== Baseline Learned Params =====")
    for key in sorted(result["baseline_learned_params"].keys()):
        print(_format_param(key, result["baseline_learned_params"][key]))

    print("\n===== Ablated Param Keys =====")
    for key in result["ablated_param_keys"]:
        print(key)

    _print_metric_block("Baseline Validation", result["baseline_val"])
    _print_metric_block("Strict Zero-Dopamine Validation", result["ablated_val"])
    _print_metric_block("Loss Delta (Ablated - Baseline)", result["delta"])


if __name__ == "__main__":
    main()
