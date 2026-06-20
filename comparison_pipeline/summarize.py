import argparse
from pathlib import Path

from comparison_pipeline.common import (
    add_common_args,
    latency_path,
    merge_cli_config,
    metrics_path,
    profile_path,
    progress_bar,
    read_json,
    write_csv,
    write_json,
)


SUMMARY_FIELDS = [
    "dataset",
    "model",
    "accuracy",
    "macro_f1",
    "loss",
    "total_params",
    "trainable_params",
    "active_params_estimated",
    "active_param_ratio",
    "dense_equivalent_flops",
    "effective_flops",
    "cpu_latency_p50_ms",
    "cpu_latency_p90_ms",
    "cpu_latency_p95_ms",
    "cpu_latency_p99_ms",
    "cpu_latency_mean_ms",
    "cpu_throughput_samples_per_s",
    "preprocess_policy",
    "flow_leakage_count",
]


def summarize_dataset(config, dataset):
    metrics = read_json(metrics_path(config, dataset))
    profile = read_json(profile_path(config, dataset))
    latency = read_json(latency_path(config, dataset))
    return {
        "dataset": dataset,
        "model": config.get("model_name", "TrafficFormer"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "loss": metrics.get("loss"),
        "total_params": profile.get("total_params"),
        "trainable_params": profile.get("trainable_params"),
        "active_params_estimated": profile.get("active_params_estimated"),
        "active_param_ratio": profile.get("active_param_ratio"),
        "dense_equivalent_flops": profile.get("dense_equivalent_flops"),
        "effective_flops": profile.get("effective_flops"),
        "cpu_latency_p50_ms": latency.get("latency_p50_ms"),
        "cpu_latency_p90_ms": latency.get("latency_p90_ms"),
        "cpu_latency_p95_ms": latency.get("latency_p95_ms"),
        "cpu_latency_p99_ms": latency.get("latency_p99_ms"),
        "cpu_latency_mean_ms": latency.get("latency_mean_ms"),
        "cpu_throughput_samples_per_s": latency.get("throughput_samples_per_s"),
        "preprocess_policy": metrics.get("preprocess_policy"),
        "flow_leakage_count": metrics.get("flow_leakage_count"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize comparison metrics, profiles, and CPU benchmark outputs.")
    add_common_args(parser)
    args = parser.parse_args()
    config = merge_cli_config(args)
    rows = [
        summarize_dataset(config, dataset)
        for dataset in progress_bar(config["datasets"], desc="summarize datasets", unit="dataset")
    ]
    root = Path(config.get("results_root", "results"))
    write_csv(root / "comparison_summary.csv", rows, SUMMARY_FIELDS)
    write_json(root / "comparison_summary.json", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
