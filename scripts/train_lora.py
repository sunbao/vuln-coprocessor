#!/usr/bin/env python3

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a software supply chain security analyst. "
    "Answer only with facts grounded in the provided input."
)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SECTION_LABELS = {
    "summary": "Summary",
    "why_affected": "Why affected",
    "risk_assessment": "Risk assessment",
    "remediation": "Remediation",
    "upgrade_recommendation": "Upgrade recommendation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a chat model with LoRA on the repository JSONL data."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_split_ratio", type=float, default=0.0)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--response_format", choices=["sections", "json"], default="sections")
    parser.add_argument("--run_name", default="qwen25-7b-explain-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--bnb_4bit_quant_type", default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Choose only one quantization mode: --load_in_4bit or --load_in_8bit.")
    if args.fp16 and args.bf16:
        parser.error("Choose only one mixed precision mode: --fp16 or --bf16.")
    if args.validation_split_ratio < 0 or args.validation_split_ratio >= 1:
        parser.error("--validation_split_ratio must be in the range [0, 1).")
    return args


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return stable_json(value)
    return str(value).strip()


def render_user_prompt(sample: Dict[str, Any]) -> str:
    parts: List[str] = []
    instruction = value_to_text(sample.get("instruction"))
    if instruction:
        parts.append("Task:\n" + instruction)

    metadata: Dict[str, Any] = {}
    for key in ("sample_id", "task", "language"):
        if sample.get(key) is not None:
            metadata[key] = sample[key]
    if metadata:
        parts.append("Metadata:\n" + stable_json(metadata))

    sample_input = sample.get("input")
    if sample_input is not None:
        parts.append("Facts:\n" + stable_json(sample_input))

    return "\n\n".join(parts).strip()


def render_assistant_response(sample: Dict[str, Any], response_format: str) -> str:
    output = sample.get("output")
    if response_format == "json":
        return value_to_text(output)

    if isinstance(output, dict):
        sections: List[str] = []
        used_keys = set()
        for key, title in SECTION_LABELS.items():
            if key in output and value_to_text(output[key]):
                sections.append(f"{title}:\n{value_to_text(output[key])}")
                used_keys.add(key)

        for key in output:
            if key not in used_keys and value_to_text(output[key]):
                pretty_key = key.replace("_", " ").title()
                sections.append(f"{pretty_key}:\n{value_to_text(output[key])}")

        return "\n\n".join(sections).strip()

    return value_to_text(output)


def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_text: str,
    assistant_text: Optional[str],
    system_prompt: str,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if assistant_text is None:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        system_block = f"System:\n{system_prompt}\n\n" if system_prompt else ""
        return f"{system_block}User:\n{user_text}\n\nAssistant:\n"

    messages.append({"role": "assistant", "content": assistant_text})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    system_block = f"System:\n{system_prompt}\n\n" if system_prompt else ""
    return f"{system_block}User:\n{user_text}\n\nAssistant:\n{assistant_text}"


def preprocess_sample(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    system_prompt: str,
    response_format: str,
    max_seq_length: int,
) -> Dict[str, Any]:
    user_text = render_user_prompt(sample)
    assistant_text = render_assistant_response(sample, response_format)

    prompt_text = apply_chat_template(
        tokenizer=tokenizer,
        user_text=user_text,
        assistant_text=None,
        system_prompt=system_prompt,
    )
    full_text = apply_chat_template(
        tokenizer=tokenizer,
        user_text=user_text,
        assistant_text=assistant_text,
        system_prompt=system_prompt,
    )

    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text = full_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    prompt_length = min(len(prompt_ids), len(input_ids))

    labels = list(input_ids)
    for index in range(prompt_length):
        labels[index] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "has_supervised_tokens": any(label != -100 for label in labels),
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            batch_labels.append(feature["labels"] + [self.label_pad_token_id] * pad_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def load_raw_datasets(args: argparse.Namespace) -> DatasetDict:
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")

    if args.max_train_samples is not None:
        limit = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(limit))

    if args.eval_file:
        eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    elif args.validation_split_ratio > 0:
        split = train_dataset.train_test_split(
            test_size=args.validation_split_ratio,
            seed=args.seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        eval_dataset = None

    if eval_dataset is not None and args.max_eval_samples is not None:
        limit = min(args.max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(limit))

    data = {"train": train_dataset}
    if eval_dataset is not None:
        data["eval"] = eval_dataset
    return DatasetDict(data)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    response_format: str,
    max_seq_length: int,
) -> Dataset:
    processed = dataset.map(
        lambda sample: preprocess_sample(
            sample=sample,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            response_format=response_format,
            max_seq_length=max_seq_length,
        ),
        remove_columns=dataset.column_names,
        desc="Tokenizing samples",
    )
    processed = processed.filter(
        lambda sample: sample["has_supervised_tokens"],
        desc="Dropping fully truncated samples",
    )
    processed = processed.remove_columns(["has_supervised_tokens"])
    return processed


def build_quantization_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit and not args.load_in_8bit:
        return None

    if not torch.cuda.is_available():
        raise ValueError("4-bit or 8-bit LoRA training requires CUDA in this script.")

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad_token or eos_token; set one before training.")

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)

    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        prepare_signature = inspect.signature(prepare_model_for_kbit_training).parameters
        prepare_kwargs: Dict[str, Any] = {}
        if "use_gradient_checkpointing" in prepare_signature:
            prepare_kwargs["use_gradient_checkpointing"] = args.gradient_checkpointing
        if "gradient_checkpointing_kwargs" in prepare_signature and args.gradient_checkpointing:
            prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    elif args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    report_to: List[str]
    if args.report_to.lower() == "none":
        report_to = []
    else:
        report_to = [args.report_to]

    training_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "do_train": True,
        "do_eval": has_eval,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": report_to,
        "remove_unused_columns": False,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "optim": "paged_adamw_8bit" if (args.load_in_4bit or args.load_in_8bit) else "adamw_torch",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    init_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "save_safetensors" in init_signature:
        training_kwargs["save_safetensors"] = True
    if "gradient_checkpointing_kwargs" in init_signature and args.gradient_checkpointing:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if has_eval:
        if "eval_strategy" in init_signature:
            training_kwargs["eval_strategy"] = "steps"
        else:
            training_kwargs["evaluation_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
    else:
        if "eval_strategy" in init_signature:
            training_kwargs["eval_strategy"] = "no"
        else:
            training_kwargs["evaluation_strategy"] = "no"

    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(args)
    raw_datasets = load_raw_datasets(args)

    train_dataset = preprocess_dataset(
        dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        response_format=args.response_format,
        max_seq_length=args.max_seq_length,
    )
    eval_dataset = None
    if "eval" in raw_datasets:
        eval_dataset = preprocess_dataset(
            dataset=raw_datasets["eval"],
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            response_format=args.response_format,
            max_seq_length=args.max_seq_length,
        )
        if len(eval_dataset) == 0:
            eval_dataset = None

    if len(train_dataset) == 0:
        raise ValueError("No trainable samples left after preprocessing.")

    training_args = build_training_arguments(args, has_eval=eval_dataset is not None)
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
