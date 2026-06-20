import argparse
import os
import subprocess
import sys

from comparison_pipeline.common import (
    REPO_ROOT,
    add_common_args,
    checkpoint_path,
    ensure_dir,
    merge_cli_config,
    progress_bar,
    test_tsv,
    train_tsv,
    valid_tsv,
)


def train_dataset(config, dataset):
    checkpoint = checkpoint_path(config, dataset)
    ensure_dir(checkpoint.parent)
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "fine-tuning" / "run_classifier.py"),
        "--vocab_path",
        str(REPO_ROOT / config["vocab_path"]),
        "--train_path",
        str(train_tsv(config, dataset)),
        "--dev_path",
        str(valid_tsv(config, dataset)),
        "--test_path",
        str(test_tsv(config, dataset)),
        "--output_model_path",
        str(checkpoint),
        "--epochs_num",
        str(config.get("epochs_num", 4)),
        "--earlystop",
        str(config.get("earlystop", 4)),
        "--batch_size",
        str(config.get("batch_size", 128)),
        "--embedding",
        str(config.get("embedding", "word_pos_seg")),
        "--encoder",
        str(config.get("encoder", "transformer")),
        "--mask",
        str(config.get("mask", "fully_visible")),
        "--seq_length",
        str(config.get("seq_length", 320)),
        "--learning_rate",
        str(config.get("learning_rate", 6e-5)),
        "--config_path",
        str(REPO_ROOT / config["config_path"]),
        "--seed",
        str(config.get("seed", 42)),
    ]
    pretrained = config.get("pretrained_model_path")
    if pretrained:
        command.extend(["--pretrained_model_path", str(pretrained)])
    if config.get("is_moe"):
        command.append("--is_moe")
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train TrafficFormer on prepared comparison datasets.")
    add_common_args(parser)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs_num", type=int, default=None)
    parser.add_argument("--earlystop", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()
    config = merge_cli_config(args)
    for dataset in progress_bar(config["datasets"], desc="train datasets", unit="dataset"):
        checkpoint = train_dataset(config, dataset)
        print({"dataset": dataset, "checkpoint": str(checkpoint)})


if __name__ == "__main__":
    main()
