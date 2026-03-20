#!/usr/bin/env python3

import argparse
import json
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_lora import DEFAULT_SYSTEM_PROMPT, apply_chat_template, render_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-sample inference with a LoRA adapter."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=384)
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


def load_sample(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def main() -> None:
    args = parse_args()
    sample = load_sample(args.input_file)
    model, tokenizer = build_model_and_tokenizer(args)

    user_text = render_user_prompt(sample)
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

    print(json.dumps({"prompt": prompt_text, "prediction": prediction}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
