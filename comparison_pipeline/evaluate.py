import argparse
import csv

from comparison_pipeline.common import (
    add_common_args,
    checkpoint_path,
    confusion_path,
    initialize_model,
    merge_cli_config,
    metrics_path,
    policy_path,
    progress_bar,
    read_json,
    test_tsv,
    write_json,
)


def evaluate_dataset(config, dataset):
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    checkpoint = checkpoint_path(config, dataset)
    args, _, run_classifier = initialize_model(config, dataset, checkpoint=checkpoint, cpu_only=bool(config.get("force_cpu_eval", False)))
    data = run_classifier.read_dataset(args, str(test_tsv(config, dataset)))
    src = torch.LongTensor([sample[0] for sample in data])
    tgt = torch.LongTensor([sample[1] for sample in data])
    seg = torch.LongTensor([sample[2] for sample in data])
    total_loss = 0.0
    total_count = 0
    y_true = []
    y_pred = []
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    args.model.eval()
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size
    batches = run_classifier.batch_loader(args.batch_size, src, tgt, seg)
    for src_batch, tgt_batch, seg_batch, soft_tgt_batch in progress_bar(
        batches, total=total_batches, desc=f"evaluate {dataset}", unit="batch"
    ):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        preds = torch.argmax(logits, dim=1)
        batch_size = int(tgt_batch.size(0))
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        for pred, gold in zip(preds.cpu().tolist(), tgt_batch.cpu().tolist()):
            confusion[pred, gold] += 1
            y_pred.append(pred)
            y_true.append(gold)
    policy = read_json(policy_path(config, dataset))
    metrics = {
        "dataset": dataset,
        "model": config.get("model_name", "TrafficFormer"),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "loss": float(total_loss / total_count) if total_count else 0.0,
        "samples": total_count,
        "labels_num": int(args.labels_num),
        "checkpoint_path": str(checkpoint),
        "preprocess_policy": policy.get("preprocess_policy"),
        "flow_leakage_count": int(policy.get("flow_leakage_count", -1)),
    }
    write_json(metrics_path(config, dataset), metrics)
    with open(confusion_path(config, dataset), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_label"] + [f"gold_{i}" for i in range(args.labels_num)])
        for i, row in enumerate(confusion.tolist()):
            writer.writerow([i] + row)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TrafficFormer checkpoint.")
    add_common_args(parser)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--force_cpu_eval", action="store_true")
    args = parser.parse_args()
    config = merge_cli_config(args)
    for dataset in progress_bar(config["datasets"], desc="evaluate datasets", unit="dataset"):
        print(evaluate_dataset(config, dataset))


if __name__ == "__main__":
    main()
