import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["cstnet_tls_1.3", "CipherSpectrum"]
DEFAULT_PREPROCESS_POLICY = (
    "flow_safe_split; strip_eth; mask_ipv4_ipv6_addresses_to_zero; "
    "mask_tcp_udp_ports_to_zero; zero_ip_tcp_udp_checksums; mask_tls_sni_bytes_to_zero_when_parsable"
)


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_config(path=None):
    default_path = REPO_ROOT / "comparison_pipeline" / "configs" / "default_comparison.json"
    config = read_json(default_path)
    if path:
        config.update(read_json(path))
    return config


def add_common_args(parser):
    parser.add_argument("--config", default=None, help="Optional JSON config overriding default_comparison.json.")
    parser.add_argument("--raw_root", default=None, help="Raw PCAP root directory.")
    parser.add_argument("--generated_root", default=None, help="Generated dataset root directory.")
    parser.add_argument("--results_root", default=None, help="Results root directory.")
    parser.add_argument("--checkpoints_root", default=None, help="Checkpoint root directory.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to process.")
    parser.add_argument("--model_name", default=None, help="Model name used in summaries.")
    parser.add_argument("--vocab_path", default=None, help="Vocabulary path.")
    parser.add_argument("--config_path", default=None, help="TrafficFormer model config path.")
    parser.add_argument("--pretrained_model_path", default=None, help="Optional pretrained checkpoint.")
    parser.add_argument("--seq_length", type=int, default=None, help="Input sequence length.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")


def merge_cli_config(args):
    config = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    if not config.get("datasets"):
        config["datasets"] = DEFAULT_DATASETS
    return config


def dataset_dir(config, dataset):
    return Path(config["generated_root"]) / dataset


def checkpoint_path(config, dataset):
    return Path(config.get("checkpoints_root", "checkpoints")) / config.get("model_name", "TrafficFormer") / dataset / "finetuned_model.bin"


def metrics_path(config, dataset):
    return Path(config.get("results_root", "results")) / "metrics" / f"{dataset}_metrics.json"


def confusion_path(config, dataset):
    return Path(config.get("results_root", "results")) / "metrics" / f"{dataset}_confusion_matrix.csv"


def profile_path(config, dataset):
    return Path(config.get("results_root", "results")) / "profiles" / f"{dataset}_profile.json"


def latency_path(config, dataset):
    return Path(config.get("results_root", "results")) / "latency" / f"{dataset}_cpu_latency.json"


def train_tsv(config, dataset):
    return dataset_dir(config, dataset) / "train_dataset.tsv"


def valid_tsv(config, dataset):
    return dataset_dir(config, dataset) / "valid_dataset.tsv"


def test_tsv(config, dataset):
    return dataset_dir(config, dataset) / "test_dataset.tsv"


def label_map_path(config, dataset):
    return dataset_dir(config, dataset) / "label_map.json"


def manifest_path(config, dataset):
    return dataset_dir(config, dataset) / "split_manifest.tsv"


def policy_path(config, dataset):
    return dataset_dir(config, dataset) / "preprocess_policy.json"


def skipped_path(config, dataset):
    return dataset_dir(config, dataset) / "skipped_files.log"


def set_cpu_only(cpu_threads=None):
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if cpu_threads:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        torch.set_num_threads(cpu_threads)
    torch.set_grad_enabled(False)


def import_run_classifier():
    module_path = REPO_ROOT / "fine-tuning" / "run_classifier.py"
    spec = importlib.util.spec_from_file_location("trafficformer_run_classifier", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def build_classifier_args(config, dataset, batch_size=None, train_path=None, dev_path=None, test_path=None, checkpoint=None):
    args = SimpleNamespace()
    args.pretrained_model_path = checkpoint
    args.output_model_path = str(checkpoint_path(config, dataset))
    args.vocab_path = str(repo_path(config["vocab_path"]))
    args.spm_model_path = None
    args.train_path = str(train_path) if train_path is not None else None
    args.dev_path = str(dev_path) if dev_path is not None else None
    args.test_path = str(test_path) if test_path is not None else None
    args.config_path = str(repo_path(config["config_path"]))
    args.embedding = config.get("embedding", "word_pos_seg")
    args.max_seq_length = 512
    args.relative_position_embedding = False
    args.relative_attention_buckets_num = 32
    args.remove_embedding_layernorm = False
    args.remove_attention_scale = False
    args.encoder = config.get("encoder", "transformer")
    args.mask = config.get("mask", "fully_visible")
    args.layernorm_positioning = "post"
    args.feed_forward = "dense"
    args.remove_transformer_bias = False
    args.layernorm = "normal"
    args.bidirectional = False
    args.factorized_embedding_parameterization = False
    args.parameter_sharing = False
    args.learning_rate = float(config.get("learning_rate", 6e-5))
    args.warmup = 0.1
    args.fp16 = False
    args.fp16_opt_level = "O1"
    args.optimizer = "adamw"
    args.scheduler = "linear"
    args.batch_size = int(batch_size or config.get("batch_size", 128))
    args.seq_length = int(config.get("seq_length", 320))
    args.dropout = 0.1
    args.epochs_num = int(config.get("epochs_num", 4))
    args.report_steps = 100
    args.seed = int(config.get("seed", 42))
    args.pooling = "first"
    args.earlystop = int(config.get("earlystop", 4))
    args.tokenizer = "bert"
    args.soft_targets = False
    args.soft_alpha = 0.5
    args.is_moe = bool(config.get("is_moe", False))
    args.vocab_size = None
    args.moebert_expert_dim = int(config.get("moebert_expert_dim", 3072))
    args.moebert_expert_num = config.get("moebert_expert_num")
    args.moebert_route_method = config.get("moebert_route_method", "hash-random")
    args.moebert_route_hash_list = config.get("moebert_route_hash_list")
    args.moebert_load_balance = float(config.get("moebert_load_balance", 0.0))
    return args


def initialize_model(config, dataset, checkpoint=None, cpu_only=False, batch_size=None):
    import torch

    if cpu_only:
        set_cpu_only(config.get("cpu_threads"))
    run_classifier = import_run_classifier()
    from uer.utils.config import load_hyperparam
    from uer.utils.constants import str2tokenizer

    args = build_classifier_args(config, dataset, batch_size=batch_size, checkpoint=None)
    args = load_hyperparam(args)
    labels = read_json(label_map_path(config, dataset))
    args.labels_num = len(labels)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = run_classifier.Classifier(args)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    args.device = torch.device("cpu" if cpu_only else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = model.to(args.device)
    model.eval()
    args.model = model
    return args, model, run_classifier


def read_tsv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def count_tsv_labels(path):
    labels = set()
    for row in read_tsv_rows(path):
        labels.add(int(row["label"]))
    return len(labels)


def percentile(values, pct):
    import numpy as np

    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def mean(values):
    import numpy as np

    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_subprocess(command, cwd=REPO_ROOT):
    print(" ".join(str(x) for x in command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def progress_bar(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def estimate_transformer_flops(config_data, labels_num, seq_length, total_params):
    layers = int(config_data.get("layers_num", 12))
    hidden = int(config_data.get("hidden_size", 768))
    ffn = int(config_data.get("feedforward_size", hidden * 4))
    emb = int(config_data.get("emb_size", hidden))
    seq = int(seq_length)
    # Approximate multiply-adds for one forward pass, batch size 1.
    per_layer = 0
    per_layer += 4 * seq * hidden * hidden
    per_layer += 2 * seq * seq * hidden
    per_layer += 2 * seq * hidden * ffn
    embedding_proj = seq * emb * hidden if emb != hidden else 0
    classifier = hidden * hidden + hidden * labels_num
    dense_equivalent = 2 * (layers * per_layer + embedding_proj + classifier)
    if dense_equivalent <= 0:
        dense_equivalent = 2 * total_params
    return int(dense_equivalent)
