import argparse
import itertools
import time

from comparison_pipeline.common import (
    add_common_args,
    checkpoint_path,
    initialize_model,
    latency_path,
    mean,
    merge_cli_config,
    percentile,
    progress_bar,
    set_cpu_only,
    test_tsv,
    write_json,
)


def make_batches(run_classifier, args, dataset, batch_size):
    import torch

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    return list(run_classifier.batch_loader(batch_size, src, tgt, seg))


def run_forward(args, batch):
    import torch

    src_batch, tgt_batch, seg_batch, soft_tgt_batch = batch
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    with torch.no_grad():
        args.model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)


def benchmark_latency(config, dataset):
    import torch

    cpu_threads = config.get("cpu_threads")
    set_cpu_only(cpu_threads)
    checkpoint = checkpoint_path(config, dataset)
    args, _, run_classifier = initialize_model(config, dataset, checkpoint=checkpoint, cpu_only=True, batch_size=1)
    data = run_classifier.read_dataset(args, str(test_tsv(config, dataset)))
    warmup = int(config.get("warmup_samples", 100))
    measured = int(config.get("measured_samples", 1000))
    latency_batches = make_batches(run_classifier, args, data, int(config.get("latency_batch_size", 1)))
    if not latency_batches:
        raise RuntimeError(f"No benchmark samples found for {dataset}.")
    iterator = itertools.cycle(latency_batches)
    for _ in progress_bar(range(warmup), desc=f"warmup {dataset}", unit="sample"):
        run_forward(args, next(iterator))
    times_ms = []
    for _ in progress_bar(range(measured), desc=f"latency {dataset}", unit="sample"):
        batch = next(iterator)
        start = time.perf_counter()
        run_forward(args, batch)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    throughput_batch_size = int(config.get("throughput_batch_size", 32))
    throughput_batches = make_batches(run_classifier, args, data, throughput_batch_size)
    iterator = itertools.cycle(throughput_batches)
    throughput_target = max(measured, throughput_batch_size)
    processed = 0
    start = time.perf_counter()
    throughput_steps = (throughput_target + throughput_batch_size - 1) // throughput_batch_size
    for _ in progress_bar(range(throughput_steps), desc=f"throughput {dataset}", unit="batch"):
        batch = next(iterator)
        run_forward(args, batch)
        processed += int(batch[0].size(0))
        if processed >= throughput_target:
            break
    elapsed = time.perf_counter() - start
    result = {
        "dataset": dataset,
        "model": config.get("model_name", "TrafficFormer"),
        "device": "cpu",
        "cpu_threads": int(torch.get_num_threads()),
        "batch_size": int(config.get("latency_batch_size", 1)),
        "throughput_batch_size": throughput_batch_size,
        "seq_length": int(config.get("seq_length", 320)),
        "input_length": int(config.get("seq_length", 320)),
        "warmup_samples": warmup,
        "measured_samples": measured,
        "latency_p50_ms": percentile(times_ms, 50),
        "latency_p90_ms": percentile(times_ms, 90),
        "latency_p95_ms": percentile(times_ms, 95),
        "latency_p99_ms": percentile(times_ms, 99),
        "latency_mean_ms": mean(times_ms),
        "throughput_samples_per_s": float(processed / elapsed) if elapsed > 0 else 0.0,
        "checkpoint_path": str(checkpoint),
    }
    write_json(latency_path(config, dataset), result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark TrafficFormer CPU latency and throughput.")
    add_common_args(parser)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--latency_batch_size", type=int, default=None)
    parser.add_argument("--throughput_batch_size", type=int, default=None)
    parser.add_argument("--warmup_samples", type=int, default=None)
    parser.add_argument("--measured_samples", type=int, default=None)
    args = parser.parse_args()
    config = merge_cli_config(args)
    for dataset in progress_bar(config["datasets"], desc="benchmark datasets", unit="dataset"):
        print(benchmark_latency(config, dataset))


if __name__ == "__main__":
    main()
