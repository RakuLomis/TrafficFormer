import argparse
import os
import subprocess
import sys
from pathlib import Path

from comparison_pipeline.common import REPO_ROOT, add_common_args, merge_cli_config, progress_bar


STAGES = [
    "prepare_data",
    "train",
    "evaluate",
    "profile_model",
    "benchmark_cpu",
    "summarize",
]


def build_base_args(config):
    args = [
        "--raw_root",
        str(config["raw_root"]),
        "--generated_root",
        str(config["generated_root"]),
        "--results_root",
        str(config.get("results_root", "results")),
        "--checkpoints_root",
        str(config.get("checkpoints_root", "checkpoints")),
        "--model_name",
        str(config.get("model_name", "TrafficFormer")),
        "--vocab_path",
        str(config["vocab_path"]),
        "--config_path",
        str(config["config_path"]),
        "--seq_length",
        str(config.get("seq_length", 320)),
        "--seed",
        str(config.get("seed", 42)),
        "--datasets",
    ]
    args.extend(config["datasets"])
    return args


def build_stage_args(config, stage):
    args = []
    if stage == "prepare_data":
        for key in ["payload_length", "payload_packets", "train_ratio", "val_ratio", "test_ratio"]:
            if key in config and config[key] is not None:
                args.extend([f"--{key}", str(config[key])])
    elif stage == "train":
        for key in ["batch_size", "epochs_num", "earlystop", "learning_rate", "embedding", "encoder", "mask"]:
            if key in config and config[key] is not None:
                args.extend([f"--{key}", str(config[key])])
    elif stage == "evaluate":
        if "batch_size" in config and config["batch_size"] is not None:
            args.extend(["--batch_size", str(config["batch_size"])])
    elif stage == "benchmark_cpu":
        for key in ["cpu_threads", "latency_batch_size", "throughput_batch_size", "warmup_samples", "measured_samples"]:
            if key in config and config[key] is not None:
                args.extend([f"--{key}", str(config[key])])
    return args


def run_stage(config, stage, use_configured_python, config_path=None):
    print(f"\n=== Running stage: {stage} ===", flush=True)
    if use_configured_python:
        python_cmd = [str(x) for x in config.get("python_cmd", [sys.executable])]
    else:
        python_cmd = [sys.executable]
    command = python_cmd + ["-u", "-m", f"comparison_pipeline.{stage}"]
    if config_path:
        command.extend(["--config", str(config_path)])
    command.extend(build_base_args(config))
    command.extend(build_stage_args(config, stage))
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full TrafficFormer fair comparison pipeline.")
    add_common_args(parser)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="Pipeline stages to run.")
    parser.add_argument("--use_configured_python", action="store_true", help="Use python_cmd from config for child stages.")
    parser.add_argument("--payload_length", type=int, default=None)
    parser.add_argument("--payload_packets", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs_num", type=int, default=None)
    parser.add_argument("--earlystop", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--latency_batch_size", type=int, default=None)
    parser.add_argument("--throughput_batch_size", type=int, default=None)
    parser.add_argument("--warmup_samples", type=int, default=None)
    parser.add_argument("--measured_samples", type=int, default=None)
    args = parser.parse_args()
    config = merge_cli_config(args)
    for stage in progress_bar(args.stages, desc="pipeline stages", unit="stage"):
        run_stage(config, stage, args.use_configured_python, args.config)
    if "summarize" in args.stages:
        summary = Path(config.get("results_root", "results")) / "comparison_summary.csv"
        print({"summary": str(summary)})
    else:
        print({"completed_stages": args.stages})


if __name__ == "__main__":
    main()
