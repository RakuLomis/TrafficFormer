import argparse

from comparison_pipeline.common import (
    add_common_args,
    checkpoint_path,
    estimate_transformer_flops,
    initialize_model,
    merge_cli_config,
    profile_path,
    progress_bar,
    read_json,
    repo_path,
    write_json,
)


def profile_dataset(config, dataset):
    checkpoint = checkpoint_path(config, dataset)
    args, model, _ = initialize_model(config, dataset, checkpoint=checkpoint if checkpoint.exists() else None, cpu_only=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = read_json(repo_path(config["config_path"]))
    dense_flops = estimate_transformer_flops(cfg, args.labels_num, config.get("seq_length", 320), total_params)
    is_moe = bool(config.get("is_moe", False))
    if is_moe and config.get("moebert_expert_num"):
        expert_num = int(config["moebert_expert_num"])
        active_params = int(total_params / max(expert_num, 1))
        active_ratio = active_params / total_params if total_params else 0.0
        effective_flops = int(dense_flops * active_ratio)
        method = "static_transformer_formula_with_moe_ratio_estimate"
    else:
        active_params = int(total_params)
        active_ratio = 1.0
        effective_flops = dense_flops
        method = "static_transformer_formula_dense_equivalent"
    profile = {
        "dataset": dataset,
        "model": config.get("model_name", "TrafficFormer"),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "active_params_estimated": int(active_params),
        "active_param_ratio": float(active_ratio),
        "dense_equivalent_flops": int(dense_flops),
        "effective_flops": int(effective_flops),
        "flops_method": method,
        "seq_length": int(config.get("seq_length", 320)),
        "labels_num": int(args.labels_num),
        "is_moe": is_moe,
        "checkpoint_path": str(checkpoint),
    }
    write_json(profile_path(config, dataset), profile)
    return profile


def main():
    parser = argparse.ArgumentParser(description="Profile TrafficFormer parameter counts and approximate FLOPs.")
    add_common_args(parser)
    args = parser.parse_args()
    config = merge_cli_config(args)
    for dataset in progress_bar(config["datasets"], desc="profile datasets", unit="dataset"):
        print(profile_dataset(config, dataset))


if __name__ == "__main__":
    main()
