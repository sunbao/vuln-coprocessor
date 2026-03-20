#!/usr/bin/env python3

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from train_lora import (
    DEFAULT_SYSTEM_PROMPT,
    SECTION_LABELS,
    apply_chat_template,
    render_assistant_response,
    render_user_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation inference for a LoRA adapter and build review artifacts."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--train_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--response_format", choices=["sections", "json"], default="sections")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--bnb_4bit_quant_type", default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Choose only one quantization mode: --load_in_4bit or --load_in_8bit.")
    return args


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def sample_ids(samples: Iterable[Dict[str, Any]]) -> List[str]:
    return [str(sample.get("sample_id", "")).strip() for sample in samples if sample.get("sample_id")]


def build_quantization_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit and not args.load_in_8bit:
        return None

    if not torch.cuda.is_available():
        raise ValueError("4-bit or 8-bit inference requires CUDA in this script.")

    return BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)

    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    if quantization_config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model, tokenizer


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def expected_section_headers(sample: Dict[str, Any], response_format: str) -> List[str]:
    if response_format != "sections":
        return []
    output = sample.get("output")
    if not isinstance(output, dict):
        return []
    headers: List[str] = []
    for key, title in SECTION_LABELS.items():
        if output.get(key):
            headers.append(f"{title}:")
    for key in output:
        if key not in SECTION_LABELS and output.get(key):
            headers.append(f"{key.replace('_', ' ').title()}:")
    return headers


def compute_auto_checks(sample: Dict[str, Any], prediction: str, response_format: str) -> Dict[str, Any]:
    sample_input = sample.get("input") or {}
    headers = expected_section_headers(sample, response_format)
    prediction_normalized = normalize_text(prediction)

    checks = {
        "non_empty": bool(prediction_normalized),
        "contains_vulnerability_id": str(sample_input.get("vulnerability_id", "")).strip() in prediction,
        "contains_component_name": str(sample_input.get("component_name", "")).strip() in prediction,
        "contains_component_version": str(sample_input.get("component_version", "")).strip() in prediction,
        "contains_recommended_version": str(sample_input.get("recommended_version", "")).strip() in prediction,
        "header_count_expected": len(headers),
        "header_count_hit": sum(1 for header in headers if header in prediction),
    }
    checks["format_pass"] = checks["header_count_hit"] == checks["header_count_expected"]

    fact_checks = [
        checks["contains_vulnerability_id"],
        checks["contains_component_name"],
        checks["contains_component_version"],
        checks["contains_recommended_version"],
    ]
    checks["key_fact_hit_rate"] = round(sum(1 for hit in fact_checks if hit) / len(fact_checks), 4)
    return checks


def iter_review_rows(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for record in records:
        sample = record["sample"]
        sample_input = sample.get("input") or {}
        yield {
            "sample_id": sample.get("sample_id", ""),
            "vulnerability_id": sample_input.get("vulnerability_id", ""),
            "component_name": sample_input.get("component_name", ""),
            "component_version": sample_input.get("component_version", ""),
            "risk_level": sample_input.get("risk_level", ""),
            "recommended_version": sample_input.get("recommended_version", ""),
            "format_pass": record["auto_checks"]["format_pass"],
            "key_fact_hit_rate": record["auto_checks"]["key_fact_hit_rate"],
            "prediction": record["prediction"],
            "reference": record["reference"],
            "format_score": "",
            "grounding_score": "",
            "completeness_score": "",
            "remediation_score": "",
            "hallucination_flag": "",
            "reviewer_notes": "",
        }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# V1 Validation Summary",
        "",
        f"- samples: `{summary['sample_count']}`",
        f"- format_pass_rate: `{summary['format_pass_rate']}`",
        f"- non_empty_rate: `{summary['non_empty_rate']}`",
        f"- contains_vulnerability_id_rate: `{summary['contains_vulnerability_id_rate']}`",
        f"- contains_component_name_rate: `{summary['contains_component_name_rate']}`",
        f"- contains_component_version_rate: `{summary['contains_component_version_rate']}`",
        f"- contains_recommended_version_rate: `{summary['contains_recommended_version_rate']}`",
        f"- average_key_fact_hit_rate: `{summary['average_key_fact_hit_rate']}`",
    ]

    if "train_overlap_count" in summary:
        lines.extend(
            [
                f"- train_overlap_count: `{summary['train_overlap_count']}`",
                f"- train_overlap_rate: `{summary['train_overlap_rate']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `predictions.jsonl`: generated answers with automatic checks",
            "- `review_sheet.csv`: manual review sheet for scoring",
            "- `summary.json`: aggregated automatic checks",
            "",
            "## Notes",
            "",
            "- These are rule-based validation helpers, not a replacement for manual review.",
            "- The current validation set is still very small, so conclusions should be treated as directional only.",
        ]
    )

    if summary.get("train_overlap_count", 0) > 0:
        lines.extend(
            [
                "- The current eval file overlaps with the training set, so this run does not measure generalization.",
                "- Build a frozen non-overlapping validation set before accepting model quality conclusions.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    samples = load_jsonl(args.eval_file, max_samples=args.max_samples)
    train_overlap_count = None
    train_overlap_rate = None
    if args.train_file:
        train_samples = load_jsonl(args.train_file)
        train_id_set = set(sample_ids(train_samples))
        eval_ids = sample_ids(samples)
        overlap_count = sum(1 for sample_id in eval_ids if sample_id in train_id_set)
        train_overlap_count = overlap_count
        train_overlap_rate = round(overlap_count / len(eval_ids), 4) if eval_ids else 0.0

    model, tokenizer = build_model_and_tokenizer(args)

    records: List[Dict[str, Any]] = []
    for sample in samples:
        user_text = render_user_prompt(sample)
        reference = render_assistant_response(sample, args.response_format)
        prompt_text = apply_chat_template(
            tokenizer=tokenizer,
            user_text=user_text,
            assistant_text=None,
            system_prompt=args.system_prompt,
        )
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_prompt_length,
        )

        device = getattr(model, "device", None)
        if device is None or str(device) == "cpu":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        auto_checks = compute_auto_checks(sample, prediction, args.response_format)

        records.append(
            {
                "sample": sample,
                "prompt": prompt_text,
                "reference": reference,
                "prediction": prediction,
                "auto_checks": auto_checks,
            }
        )

    sample_count = len(records)
    summary = {
        "sample_count": sample_count,
        "format_pass_rate": round(sum(r["auto_checks"]["format_pass"] for r in records) / sample_count, 4),
        "non_empty_rate": round(sum(r["auto_checks"]["non_empty"] for r in records) / sample_count, 4),
        "contains_vulnerability_id_rate": round(
            sum(r["auto_checks"]["contains_vulnerability_id"] for r in records) / sample_count,
            4,
        ),
        "contains_component_name_rate": round(
            sum(r["auto_checks"]["contains_component_name"] for r in records) / sample_count,
            4,
        ),
        "contains_component_version_rate": round(
            sum(r["auto_checks"]["contains_component_version"] for r in records) / sample_count,
            4,
        ),
        "contains_recommended_version_rate": round(
            sum(r["auto_checks"]["contains_recommended_version"] for r in records) / sample_count,
            4,
        ),
        "average_key_fact_hit_rate": round(
            sum(r["auto_checks"]["key_fact_hit_rate"] for r in records) / sample_count,
            4,
        ),
    }
    if train_overlap_count is not None and train_overlap_rate is not None:
        summary["train_overlap_count"] = train_overlap_count
        summary["train_overlap_rate"] = train_overlap_rate

    serializable_records = [
        {
            "sample_id": record["sample"].get("sample_id"),
            "prediction": record["prediction"],
            "reference": record["reference"],
            "auto_checks": record["auto_checks"],
        }
        for record in records
    ]

    write_jsonl(os.path.join(args.output_dir, "predictions.jsonl"), serializable_records)
    write_review_csv(os.path.join(args.output_dir, "review_sheet.csv"), iter_review_rows(records))
    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    with open(os.path.join(args.output_dir, "SUMMARY.md"), "w", encoding="utf-8") as handle:
        handle.write(render_summary_markdown(summary))


if __name__ == "__main__":
    main()
