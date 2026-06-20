import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except Exception as exc:  # pragma: no cover - reported at runtime.
    scapy = None
    IP = TCP = UDP = IPv6 = None
    SCAPY_IMPORT_ERROR = exc
else:
    SCAPY_IMPORT_ERROR = None

from comparison_pipeline.common import (
    DEFAULT_PREPROCESS_POLICY,
    add_common_args,
    dataset_dir,
    ensure_dir,
    label_map_path,
    manifest_path,
    merge_cli_config,
    policy_path,
    progress_bar,
    skipped_path,
    test_tsv,
    train_tsv,
    valid_tsv,
    write_json,
)


PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}
TLS_SNI_TYPE = b"\x00\x00"


def bigram_generation(hex_string, token_len):
    chars = list(hex_string)
    out = []
    for i in range(max(0, min(len(chars) - 1, token_len))):
        out.append(chars[i] + chars[i + 1])
    return " ".join(out)


def canonical_endpoint(ip, port):
    return f"{ip}:{port}"


def canonical_flow_key(packet):
    proto = None
    sport = 0
    dport = 0
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = int(packet[IP].proto)
    elif IPv6 in packet:
        src_ip = packet[IPv6].src
        dst_ip = packet[IPv6].dst
        proto = int(packet[IPv6].nh)
    else:
        return None
    if TCP in packet:
        sport = int(packet[TCP].sport)
        dport = int(packet[TCP].dport)
        proto = 6
    elif UDP in packet:
        sport = int(packet[UDP].sport)
        dport = int(packet[UDP].dport)
        proto = 17
    endpoints = sorted([canonical_endpoint(src_ip, sport), canonical_endpoint(dst_ip, dport)])
    return f"{endpoints[0]}<->{endpoints[1]}|proto={proto}"


def packet_direction(packet, first_src):
    if IP in packet:
        return 1 if packet[IP].src == first_src else -1
    if IPv6 in packet:
        return 1 if packet[IPv6].src == first_src else -1
    return 1


def find_tls_sni_spans(payload):
    spans = []
    cursor = 0
    while True:
        pos = payload.find(TLS_SNI_TYPE, cursor)
        if pos < 0:
            break
        if pos + 9 > len(payload):
            break
        ext_len = int.from_bytes(payload[pos + 2 : pos + 4], "big")
        ext_end = pos + 4 + ext_len
        if ext_end > len(payload) or ext_len < 5:
            cursor = pos + 2
            continue
        list_len = int.from_bytes(payload[pos + 4 : pos + 6], "big")
        name_pos = pos + 6
        list_end = min(pos + 6 + list_len, ext_end)
        while name_pos + 3 <= list_end:
            name_len = int.from_bytes(payload[name_pos + 1 : name_pos + 3], "big")
            value_start = name_pos + 3
            value_end = value_start + name_len
            if value_end <= list_end:
                spans.append((value_start, value_end))
            name_pos = value_end
        cursor = ext_end
    return spans


def mask_sni_in_transport(packet):
    if TCP not in packet and UDP not in packet:
        return 0
    layer = packet[TCP] if TCP in packet else packet[UDP]
    payload = bytes(layer.payload)
    if not payload:
        return 0
    spans = find_tls_sni_spans(payload)
    if not spans:
        return 0
    masked = bytearray(payload)
    for start, end in spans:
        masked[start:end] = b"\x00" * (end - start)
    layer.remove_payload()
    layer.add_payload(bytes(masked))
    return len(spans)


def strip_eth_and_mask(packet):
    pkt = packet.copy()
    if IP in pkt:
        pkt = pkt[IP].copy()
        pkt.src = "0.0.0.0"
        pkt.dst = "0.0.0.0"
        pkt.chksum = 0
    elif IPv6 in pkt:
        pkt = pkt[IPv6].copy()
        pkt.src = "::"
        pkt.dst = "::"
    else:
        return None, 0
    sni_count = mask_sni_in_transport(pkt)
    if TCP in pkt:
        pkt[TCP].sport = 0
        pkt[TCP].dport = 0
        pkt[TCP].chksum = 0
    if UDP in pkt:
        pkt[UDP].sport = 0
        pkt[UDP].dport = 0
        pkt[UDP].chksum = 0
    return pkt, sni_count


def infer_label(dataset_root, file_path):
    rel = file_path.relative_to(dataset_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return file_path.stem


def discover_pcaps(dataset_root):
    files = []
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES:
            files.append(path)
    return sorted(files)


def make_flow_id(flow_key):
    return hashlib.sha1(flow_key.encode("utf-8")).hexdigest()


def extract_samples(file_path, label_name, label_id, payload_length, payload_packets, skipped):
    try:
        packets = scapy.rdpcap(str(file_path))
    except Exception as exc:
        skipped.append((file_path, f"read_failed:{exc}"))
        return []
    flows = defaultdict(list)
    for packet in packets:
        key = canonical_flow_key(packet)
        if key:
            flows[key].append(packet)
    samples = []
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) < 3:
            skipped.append((file_path, f"flow_too_short:{flow_key}"))
            continue
        first = flow_packets[0]
        first_src = first[IP].src if IP in first else first[IPv6].src
        parts = []
        sni_masked = 0
        used_packets = 0
        for packet in flow_packets[:payload_packets]:
            masked, sni_count = strip_eth_and_mask(packet)
            if masked is None:
                continue
            sni_masked += sni_count
            raw_hex = bytes(masked).hex()[: payload_length * 2]
            if not raw_hex:
                continue
            parts.append("[SEP] " + bigram_generation(raw_hex, token_len=len(raw_hex) - 1))
            used_packets += 1
        if used_packets == 0:
            skipped.append((file_path, f"no_usable_packets:{flow_key}"))
            continue
        samples.append(
            {
                "label": label_id[label_name],
                "label_name": label_name,
                "source_file": str(file_path),
                "flow_key": flow_key,
                "flow_id": make_flow_id(flow_key),
                "text_a": " ".join(parts),
                "packets_used": used_packets,
                "sni_masked_count": sni_masked,
            }
        )
    if not flows:
        skipped.append((file_path, "no_tcp_udp_ipv4_ipv6_flows"))
    return samples


def split_flows(samples, seed, train_ratio, val_ratio):
    by_label = defaultdict(dict)
    for sample in samples:
        by_label[sample["label"]].setdefault(sample["flow_id"], []).append(sample)
    split_by_flow = {}
    rng = random.Random(seed)
    for label, flows in by_label.items():
        flow_ids = sorted(flows)
        rng.shuffle(flow_ids)
        n = len(flow_ids)
        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))
        if n >= 3:
            train_end = min(max(train_end, 1), n - 2)
            val_end = min(max(val_end, train_end + 1), n - 1)
        elif n == 2:
            train_end = 1
            val_end = 1
        for index, flow_id in enumerate(flow_ids):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            split_by_flow[flow_id] = split
    split_samples = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_samples[split_by_flow[sample["flow_id"]]].append(sample)
    return split_samples


def leakage_count_from_splits(split_samples):
    seen = {}
    leaks = 0
    for split, rows in split_samples.items():
        for row in rows:
            previous = seen.setdefault(row["flow_id"], split)
            if previous != split:
                leaks += 1
    return leaks


def write_tsv(path, samples):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", "text_a"])
        for sample in samples:
            writer.writerow([sample["label"], sample["text_a"]])


def process_dataset(config, dataset, dry_run=False):
    raw_root = Path(config["raw_root"]) / dataset
    out_dir = dataset_dir(config, dataset)
    pcaps = discover_pcaps(raw_root)
    labels = sorted({infer_label(raw_root, path) for path in pcaps})
    label_id = {label: index for index, label in enumerate(labels)}
    if dry_run:
        return {"dataset": dataset, "pcap_files": len(pcaps), "labels": len(labels), "output_dir": str(out_dir)}
    if scapy is None:
        raise RuntimeError(f"scapy import failed: {SCAPY_IMPORT_ERROR}")
    ensure_dir(out_dir)
    print(
        {
            "dataset": dataset,
            "pcap_files": len(pcaps),
            "labels": len(labels),
            "output_dir": str(out_dir),
        },
        flush=True,
    )
    skipped = []
    samples = []
    for file_path in progress_bar(pcaps, desc=f"prepare {dataset}", unit="pcap"):
        label_name = infer_label(raw_root, file_path)
        samples.extend(
            extract_samples(
                file_path,
                label_name,
                label_id,
                int(config.get("payload_length", 64)),
                int(config.get("payload_packets", 5)),
                skipped,
            )
        )
    split_samples = split_flows(
        samples,
        int(config.get("seed", 42)),
        float(config.get("train_ratio", 0.8)),
        float(config.get("val_ratio", 0.1)),
    )
    write_tsv(train_tsv(config, dataset), split_samples["train"])
    write_tsv(valid_tsv(config, dataset), split_samples["val"])
    write_tsv(test_tsv(config, dataset), split_samples["test"])
    write_json(label_map_path(config, dataset), label_id)
    manifest_rows = []
    for split, rows in split_samples.items():
        for row in rows:
            manifest_rows.append(
                {
                    "split": split,
                    "label": row["label_name"],
                    "source_file": row["source_file"],
                    "flow_key": row["flow_key"],
                    "flow_id": row["flow_id"],
                }
            )
    with open(manifest_path(config, dataset), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "source_file", "flow_key", "flow_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with open(skipped_path(config, dataset), "w", encoding="utf-8") as f:
        for path, reason in skipped:
            f.write(f"{path}\t{reason}\n")
    policy = {
        "dataset": dataset,
        "preprocess_policy": DEFAULT_PREPROCESS_POLICY,
        "seed": int(config.get("seed", 42)),
        "split_ratio": {
            "train": float(config.get("train_ratio", 0.8)),
            "val": float(config.get("val_ratio", 0.1)),
            "test": float(config.get("test_ratio", 0.1)),
        },
        "payload_length": int(config.get("payload_length", 64)),
        "payload_packets": int(config.get("payload_packets", 5)),
        "flow_leakage_count": leakage_count_from_splits(split_samples),
        "labels": len(label_id),
        "samples": {key: len(value) for key, value in split_samples.items()},
        "pcap_files": len(pcaps),
        "skipped_records": len(skipped),
    }
    write_json(policy_path(config, dataset), policy)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Prepare flow-safe masked TSV data from raw PCAP files.")
    add_common_args(parser)
    parser.add_argument("--payload_length", type=int, default=None)
    parser.add_argument("--payload_packets", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Only report discovered files and labels.")
    args = parser.parse_args()
    config = merge_cli_config(args)
    for dataset in config["datasets"]:
        result = process_dataset(config, dataset, dry_run=args.dry_run)
        print(result)


if __name__ == "__main__":
    main()
