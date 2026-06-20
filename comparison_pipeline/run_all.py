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
