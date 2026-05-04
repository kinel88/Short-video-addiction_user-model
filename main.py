from env_module import make_env
from generate_final_watch_ratio_reports import run_final_watch_ratio_reports


def _format_behavior_param(key, value):
    if isinstance(value, dict):
        numeric_values = [float(v) for v in value.values()]
        if len(numeric_values) == 0:
            return f"{key}: count=0"
        return (
            f"{key}: count={len(numeric_values)} "
            f"min={min(numeric_values):.6f} "
            f"max={max(numeric_values):.6f} "
            f"mean={sum(numeric_values) / len(numeric_values):.6f}"
        )
    return f"{key}: {float(value):.6f}"


def main():
    experiment = run_final_watch_ratio_reports()
    rollout_result = experiment["rollout_result"]
    rollout_split = experiment["split"]
    feature_provider = experiment["feature_provider"]
    selected_loss_weights = experiment.get("selected_loss_weights", {})
    cache_status = experiment.get("cache_status", "unknown")
    reuse_saved_results = experiment.get("reuse_saved_results", True)

    print("\n===== Cache Summary =====")
    print(
        f"reuse_saved_results={str(bool(reuse_saved_results)).lower()} "
        f"cache_status={cache_status}"
    )
    if selected_loss_weights:
        print(
            "selected_loss_weights="
            f"value={selected_loss_weights['value']:.3f}, "
            f"bucket_ce={selected_loss_weights['bucket_ce']:.3f}, "
            f"bucket_dist={selected_loss_weights['bucket_dist']:.3f}"
        )

    print("\n===== Train / Validation Summary =====")
    meta = rollout_split["meta"]
    print(
        f"train_source={meta['train_source_name']} "
        f"val_source={meta['val_source_name']} "
        f"post_filter_train_rows={meta['post_filter_train_row_count']} "
        f"post_filter_val_rows={meta['post_filter_val_row_count']}"
    )

    print("\n===== Rollout Learned behavior params =====")
    for k, v in rollout_result["learned_params"].items():
        print(_format_behavior_param(k, v))

    env = make_env(
        feature_provider=feature_provider,
        num_candidates=5,
        slate_size=1,
        seed=0,
        behavior_params=rollout_result["learned_params"],
    )

    obs = env.reset()
    print("\n===== Validation Summary =====")
    summary = rollout_result["summary"]
    print(
        f"[rollout] mean_abs_error={summary['mean_abs_error']:.6f} "
        f"p90_abs_error={summary['p90_abs_error']:.6f} "
        f"bucket_match_rate={summary['bucket_match_rate']:.6f}"
    )

    print("\nEnvironment built successfully with rollout behavior parameters.")
    print("Observation keys:", obs.keys())
    print(
        "Report outputs:",
        experiment["output"]["html_path"],
    )


if __name__ == "__main__":
    print("MAIN STARTED")
    main()
