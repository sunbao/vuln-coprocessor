#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export train/validation/test JSONL files from metadata.split."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="v1-official")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    buckets: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    split_counter: Counter[str] = Counter()

    with open(args.input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            split = (obj.get("metadata") or {}).get("split")
            if split not in buckets:
                raise ValueError(f"Unexpected split value: {split!r}")
            buckets[split].append(raw)
            split_counter[split] += 1

    for split_name, lines in buckets.items():
        output_path = os.path.join(args.output_dir, f"{args.prefix}-{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    summary = {
        "input_file": args.input_file,
        "output_dir": args.output_dir,
        "prefix": args.prefix,
        "counts": dict(split_counter),
    }
    summary_path = os.path.join(args.output_dir, f"{args.prefix}-split-summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
