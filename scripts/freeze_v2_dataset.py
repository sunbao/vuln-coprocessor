#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a non-overlapping V2 validation set and a train candidate set from exported JSONL files."
    )
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="v2-frozen")
    parser.add_argument("--validation_target", type=int, default=500)
    parser.add_argument("--validation_splits", default="validation,test")
    parser.add_argument("--train_splits", default="train")
    return parser.parse_args()


def parse_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def load_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def sample_id_of(record: Dict[str, object]) -> str:
    return str(record.get("sample_id") or "").strip()


def split_of(record: Dict[str, object]) -> str:
    return str((record.get("metadata") or {}).get("split") or "").strip()


def stable_rank(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()


def write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    validation_splits = set(parse_csv(args.validation_splits))
    train_splits = set(parse_csv(args.train_splits))
    if not validation_splits:
        raise SystemExit("validation_splits cannot be empty")
    if not train_splits:
        raise SystemExit("train_splits cannot be empty")
    if args.validation_target <= 0:
        raise SystemExit("validation_target must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)

    input_counter: Counter[str] = Counter()
    candidate_counter: Counter[str] = Counter()
    duplicate_counter: Counter[str] = Counter()

    by_sample_id: Dict[str, Dict[str, object]] = {}
    source_file_by_sample_id: Dict[str, str] = {}

    for input_file in args.input_files:
        for record in load_jsonl(input_file):
            sample_id = sample_id_of(record)
            if not sample_id:
                continue
            split = split_of(record)
            input_counter[split] += 1
            if sample_id in by_sample_id:
                duplicate_counter[split] += 1
                continue
            by_sample_id[sample_id] = record
            source_file_by_sample_id[sample_id] = input_file
            candidate_counter[split] += 1

    validation_pool: List[Tuple[str, Dict[str, object]]] = []
    train_pool: List[Tuple[str, Dict[str, object]]] = []
    for sample_id, record in by_sample_id.items():
        split = split_of(record)
        if split in validation_splits:
            validation_pool.append((sample_id, record))
        if split in train_splits:
            train_pool.append((sample_id, record))

    validation_pool.sort(key=lambda item: stable_rank(item[0]))
    frozen_validation = validation_pool[: args.validation_target]
    frozen_validation_ids = {sample_id for sample_id, _ in frozen_validation}

    train_records = [
        record
        for sample_id, record in train_pool
        if sample_id not in frozen_validation_ids
    ]
    train_records.sort(key=lambda record: sample_id_of(record))

    validation_records = [record for _, record in frozen_validation]
    validation_records.sort(key=lambda record: sample_id_of(record))

    validation_path = os.path.join(args.output_dir, f"{args.prefix}-validation.jsonl")
    train_path = os.path.join(args.output_dir, f"{args.prefix}-train.jsonl")
    summary_path = os.path.join(args.output_dir, f"{args.prefix}-summary.json")
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}-validation-manifest.json")

    validation_count = write_jsonl(validation_path, validation_records)
    train_count = write_jsonl(train_path, train_records)

    validation_ecosystems = Counter()
    validation_risks = Counter()
    validation_sources = Counter()
    for record in validation_records:
        sample_input = record.get("input") or {}
        metadata = record.get("metadata") or {}
        validation_ecosystems[str(sample_input.get("ecosystem") or "")] += 1
        validation_risks[str(sample_input.get("risk_level") or "")] += 1
        validation_sources[str(metadata.get("source") or "")] += 1

    manifest = {
        sample_id: source_file_by_sample_id[sample_id]
        for sample_id in sorted(frozen_validation_ids)
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    summary = {
        "prefix": args.prefix,
        "input_files": args.input_files,
        "validation_target": args.validation_target,
        "validation_splits": sorted(validation_splits),
        "train_splits": sorted(train_splits),
        "input_counts_by_split": dict(input_counter),
        "unique_counts_by_split": dict(candidate_counter),
        "duplicate_counts_by_split": dict(duplicate_counter),
        "validation_pool_size": len(validation_pool),
        "frozen_validation_count": validation_count,
        "train_count": train_count,
        "counts_by_validation_ecosystem": dict(validation_ecosystems),
        "counts_by_validation_risk_level": dict(validation_risks),
        "counts_by_validation_source": dict(validation_sources),
        "validation_file": validation_path,
        "train_file": train_path,
        "validation_manifest_file": manifest_path,
        "validation_sample_id_preview": sorted(frozen_validation_ids)[:20],
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
