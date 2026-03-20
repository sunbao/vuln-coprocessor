# vuln-coprocessor

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Repository](https://img.shields.io/badge/GitHub-sunbao%2Fvuln--coprocessor-181717?logo=github)](https://github.com/sunbao/vuln-coprocessor.git)
[![Model Family](https://img.shields.io/badge/Base%20Model-Qwen2.5--7B--Instruct-0F766E)](#release-snapshot)
[![Training](https://img.shields.io/badge/Training-LoRA-C2410C)](#train-a-lora-adapter)

`vuln-coprocessor` is a narrow-domain vulnerability explanation model project.
It turns structured vulnerability facts into grounded, operator-facing outputs such as risk interpretation, remediation guidance, and upgrade recommendations.

This repository is intentionally packaged like a small model release rather than a generic experiment folder.
It includes public dataset snapshots, LoRA training code, evaluation scripts, TensorBoard monitoring, and release-oriented documentation.

## At A Glance

| Item | Current State |
| --- | --- |
| Model name | `vuln-coprocessor` |
| Base model family | `Qwen2.5-7B-Instruct` |
| Fine-tuning method | `LoRA` |
| Python runtime | `3.11` |
| Public data snapshots | `data/processed/` |
| V1 adapter | trained locally, not published in this repo |
| V2 adapter | training and validation in progress |
| Public adapter release | not finalized yet |

Quick links:

- repository: `https://github.com/sunbao/vuln-coprocessor.git`
- model card: [`MODEL_CARD.md`](MODEL_CARD.md)
- capability boundary: [`MODEL_CAPABILITY_SPEC.md`](specs/MODEL_CAPABILITY_SPEC.md)
- training rules: [`TRAINING_SPEC.md`](specs/TRAINING_SPEC.md)
- TensorBoard guide: [`TENSORBOARD_OBSERVATION_GUIDE.md`](specs/TENSORBOARD_OBSERVATION_GUIDE.md)
- release gate: [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)

## What This Project Does

The training target is post-scan vulnerability explanation, not raw vulnerability discovery.

Given structured facts such as:

- component name
- component version
- vulnerability ID
- severity or risk level
- recommended version
- matching evidence

the model is expected to produce:

- a short explanation of why the component is affected
- a grounded view of current risk
- remediation guidance
- an explicit upgrade recommendation

In practical terms, this project is an explanation layer that sits after matching and inventory logic.
It is designed to make vulnerability records more understandable and more actionable for engineers and operators.

## Release Snapshot

This repository currently represents a public-ready code and data release, with model weights still gated by validation.

- code and documentation: publishable now
- processed training data snapshots: publishable now
- local checkpoints and experiment logs: intentionally excluded from Git
- validated adapter weights: planned for a later release once V2 passes evaluation

Recommended public release shape:

- code in GitHub
- approved dataset snapshots in GitHub
- validated LoRA adapter weights in GitHub Releases or a model registry

Project tracking:

- repository publishing checklist: [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)
- project change history: [`CHANGELOG.md`](CHANGELOG.md)

## Capability Boundary

The target capability set is:

- explain why a component is affected
- summarize current risk in grounded language
- suggest remediation actions
- state whether upgrade is recommended

Stable capability boundary:

- [`MODEL_CAPABILITY_SPEC.md`](specs/MODEL_CAPABILITY_SPEC.md)

## Non-Goals

This project should not be treated as:

- a vulnerability discovery engine
- a source-of-truth matching engine
- a replacement for upstream vulnerability databases
- a general-purpose security assistant

## Repository Layout

- `README.md`: release-style project overview
- `MODEL_CARD.md`: model card and release metadata
- `LICENSE`, `NOTICE`, `LICENSE-DATA`, `LICENSE-MODEL`: release license files
- `scripts/`: training, validation, export, and utility scripts
- `examples/`: minimal example inputs
- `.github/`: issue templates, PR template, and basic CI smoke workflow
- `specs/`: stable rules, capability boundary, monitoring guide, and publishing guide
- `data/processed/`: approved public dataset snapshots
- `artifacts/`: internal run notes and local experiment records

## Quick Start

Create a Python environment and install training dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
```

If you train with CUDA 4-bit or 8-bit LoRA, ensure the environment also includes `bitsandbytes`.

## Public Data Snapshots

Currently available local snapshots include:

- `data/processed/v1-official-train.jsonl`
- `data/processed/v1-official-validation.jsonl`
- `data/processed/v1-official-test.jsonl`
- `data/processed/v2-frozen-2026-03-19-train.jsonl`
- `data/processed/v2-frozen-2026-03-19-validation.jsonl`

The JSONL training contract is:

- `instruction`
- `input`
- `output`
- optional metadata such as `sample_id`, `task`, and `language`

Stable training and evaluation rules:

- [`TRAINING_SPEC.md`](specs/TRAINING_SPEC.md)

## Train A LoRA Adapter

Example training command:

```bash
python3 scripts/train_lora.py \
  --model_name_or_path /path/to/base-model \
  --train_file data/processed/v2-frozen-2026-03-19-train.jsonl \
  --eval_file data/processed/v2-frozen-2026-03-19-validation.jsonl \
  --output_dir checkpoints/qwen25-7b-explain-v2 \
  --load_in_4bit \
  --fp16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 100 \
  --report_to tensorboard
```

Useful options:

- `--validation_split_ratio 0.02`
- `--max_train_samples 32 --max_eval_samples 8 --max_steps 5`
- `--response_format json`
- `--target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `--report_to tensorboard`

## Run Inference

When an adapter is available, run a single-sample inference like this:

```bash
python3 scripts/infer_lora.py \
  --base_model_path /path/to/base-model \
  --adapter_path /path/to/adapter \
  --input_file examples/inference_input.json \
  --load_in_4bit
```

## Monitor Training

Start TensorBoard from the training output directory root:

```bash
scripts/run_tensorboard.sh checkpoints/qwen25-7b-explain-v2
```

Then open:

```text
http://<server-ip>:6006
```

If you are viewing from another machine on the LAN, use the training server IP instead of `127.0.0.1`.

TensorBoard operator guide:

- [`TENSORBOARD_OBSERVATION_GUIDE.md`](specs/TENSORBOARD_OBSERVATION_GUIDE.md)

## Evaluate And Validate

Validation should be done on a frozen non-overlapping validation set.

This repository includes:

- validation inference tooling in `scripts/validate_lora.py`
- frozen V2 validation data in `data/processed/v2-frozen-2026-03-19-validation.jsonl`

Do not accept a release based on training loss alone.

## Optional Data Regeneration

If you want to regenerate training data from a live source system, use:

- [`export_live_dataset_154.py`](scripts/export_live_dataset_154.py)

Local secrets should remain outside Git and be loaded from `.env`:

```bash
set -a
source .env
set +a
```

Example staged export:

```bash
python3 scripts/export_live_dataset_154.py \
  --output_dir data/processed \
  --prefix v2-live-online-pilot \
  --page_size 100 \
  --batch_size 100 \
  --max_rows 1000 \
  --sleep_seconds 1.0
```

## Publishing To GitHub

This repository is intended to be publishable as an open model project.

Public:

- `scripts/`
- `specs/`
- `README.md`
- `MODEL_CARD.md`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `RELEASE_CHECKLIST.md`
- `LICENSE`
- `NOTICE`
- `LICENSE-DATA`
- `LICENSE-MODEL`
- `requirements-train.txt`
- `examples/`
- approved data under `data/processed/`

Private-local only:

- `.env`
- `.venv/`
- `.venv311/`
- `cache/`
- `checkpoints/`
- `artifacts/`
- local logs

Publishing rules:

- [`GITHUB_PUBLISHING_GUIDE.md`](specs/GITHUB_PUBLISHING_GUIDE.md)

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md): model card and release metadata
- [`CITATION.cff`](CITATION.cff): citation metadata for GitHub and downstream users
- [`CONTRIBUTING.md`](CONTRIBUTING.md): contribution rules for public collaboration
- [`CHANGELOG.md`](CHANGELOG.md): repository change history
- [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md): release gate before publishing code or model artifacts
- [`LICENSE`](LICENSE): repository license
- [`NOTICE`](NOTICE): attribution and release notice
- [`LICENSE-DATA`](LICENSE-DATA): public data license notice
- [`LICENSE-MODEL`](LICENSE-MODEL): model artifact license notice
- [`TRAINING_SPEC.md`](specs/TRAINING_SPEC.md): training and evaluation rules
- [`MODEL_CAPABILITY_SPEC.md`](specs/MODEL_CAPABILITY_SPEC.md): capability boundary
- [`TENSORBOARD_OBSERVATION_GUIDE.md`](specs/TENSORBOARD_OBSERVATION_GUIDE.md): live monitoring guide
- [`GITHUB_PUBLISHING_GUIDE.md`](specs/GITHUB_PUBLISHING_GUIDE.md): public release guide

## Notes

- The training stack uses `transformers` + `peft` directly.
- Labels are masked on prompt tokens, so only the assistant response contributes to the loss.
- Adapter checkpoints are intentionally ignored by Git and should be released separately when validated.
