#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repository-friendly validation report from training artifacts."
    )
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_markdown", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--dataset_summary_file")
    parser.add_argument("--validation_artifacts_dir")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_sample_ids(path: str) -> List[str]:
    sample_ids: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id", "")).strip()
            if sample_id:
                sample_ids.append(sample_id)
    return sample_ids


def compute_overlap(train_file: str, eval_file: str) -> Dict[str, Any]:
    train_ids = set(load_jsonl_sample_ids(train_file))
    eval_ids = load_jsonl_sample_ids(eval_file)
    overlap_count = sum(1 for sample_id in eval_ids if sample_id in train_ids)
    return {
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "overlap_count": overlap_count,
        "overlap_rate": round(overlap_count / len(eval_ids), 4) if eval_ids else 0.0,
    }


def validation_artifact_status(path: str | None) -> Dict[str, Any]:
    if not path:
        return {"available": False, "files": []}
    artifact_dir = Path(path)
    files = []
    for name in ("predictions.jsonl", "review_sheet.csv", "summary.json", "SUMMARY.md"):
        file_path = artifact_dir / name
        if file_path.exists():
            files.append(name)
    return {"available": bool(files), "files": files}


def latest_checkpoint_name(adapter_dir: str) -> str | None:
    root = Path(adapter_dir)
    candidates = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, child.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# {summary['run_label']} Validation Report",
        "",
        "## Status",
        "",
        f"- training_complete: `{summary['training_complete']}`",
        f"- promotion_ready: `{summary['promotion_ready']}`",
        f"- latest_checkpoint: `{summary['latest_checkpoint']}`",
        f"- validation_artifacts_available: `{summary['validation_artifacts']['available']}`",
        "",
        "## Training Metrics",
        "",
        f"- epoch: `{summary['train_metrics'].get('epoch')}`",
        f"- global_step: `{summary['global_step']}`",
        f"- train_loss: `{summary['train_metrics'].get('train_loss')}`",
        f"- train_runtime_seconds: `{summary['train_metrics'].get('train_runtime')}`",
        f"- train_samples: `{summary['train_metrics'].get('train_samples')}`",
        f"- train_steps_per_second: `{summary['train_metrics'].get('train_steps_per_second')}`",
        "",
        "## Eval Metrics",
        "",
        f"- eval_loss: `{summary['eval_metrics'].get('eval_loss')}`",
        f"- eval_samples: `{summary['eval_metrics'].get('eval_samples')}`",
        f"- eval_runtime_seconds: `{summary['eval_metrics'].get('eval_runtime')}`",
        f"- eval_samples_per_second: `{summary['eval_metrics'].get('eval_samples_per_second')}`",
        "",
        "## Dataset Integrity",
        "",
        f"- train_count: `{summary['dataset_integrity'].get('train_count')}`",
        f"- eval_count: `{summary['dataset_integrity'].get('eval_count')}`",
        f"- overlap_count: `{summary['dataset_integrity'].get('overlap_count')}`",
        f"- overlap_rate: `{summary['dataset_integrity'].get('overlap_rate')}`",
    ]

    dataset_summary = summary.get("dataset_summary")
    if dataset_summary:
        lines.extend(
            [
                f"- validation_pool_size: `{dataset_summary.get('validation_pool_size')}`",
                f"- frozen_validation_count: `{dataset_summary.get('frozen_validation_count')}`",
                f"- validation_ecosystem_breakdown: `{dataset_summary.get('counts_by_validation_ecosystem')}`",
                f"- validation_risk_breakdown: `{dataset_summary.get('counts_by_validation_risk_level')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Validation Evidence",
            "",
        ]
    )

    if summary["validation_artifacts"]["available"]:
        lines.append(
            f"- generated_artifacts: `{', '.join(summary['validation_artifacts']['files'])}`"
        )
    else:
        lines.extend(
            [
                "- generated prediction-level validation artifacts are not present in the repository",
                "- training-side eval metrics exist, but prediction-level review is still required before promotion",
            ]
        )

    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
        ]
    )

    if summary["promotion_ready"]:
        lines.append("- decision: `ready_for_next_iteration`")
    else:
        lines.extend(
            [
                "- decision: `hold_for_validation`",
                "- reason: frozen-set loss exists, but prediction-level validation review is incomplete or missing",
            ]
        )

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- commit this report and any validation artifacts to GitHub before starting the next training version",
            "- run prediction-level validation on a CUDA-capable host or another practical inference environment",
            "- only start the next iteration after the validation gate is explicitly accepted",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    train_metrics = load_json(os.path.join(args.adapter_dir, "train_results.json"))
    eval_metrics = load_json(os.path.join(args.adapter_dir, "eval_results.json"))
    trainer_state = load_json(os.path.join(args.adapter_dir, "trainer_state.json"))
    dataset_integrity = compute_overlap(args.train_file, args.eval_file)
    dataset_summary = load_json(args.dataset_summary_file) if args.dataset_summary_file else None
    validation_artifacts = validation_artifact_status(args.validation_artifacts_dir)
    latest_checkpoint = latest_checkpoint_name(args.adapter_dir)
    training_complete = trainer_state.get("global_step") == trainer_state.get("max_steps")
    promotion_ready = (
        training_complete
        and dataset_integrity["overlap_count"] == 0
        and validation_artifacts["available"]
    )

    summary = {
        "run_label": args.run_label,
        "adapter_dir": args.adapter_dir,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "training_complete": training_complete,
        "promotion_ready": promotion_ready,
        "latest_checkpoint": latest_checkpoint,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "dataset_integrity": dataset_integrity,
        "dataset_summary": dataset_summary,
        "validation_artifacts": validation_artifacts,
    }

    with open(args.output_markdown, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
