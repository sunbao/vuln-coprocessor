# Model Card: vuln-coprocessor

Repository:

- `https://github.com/sunbao/vuln-coprocessor.git`

## Summary

`vuln-coprocessor` is a small domain-focused model project for vulnerability explanation and remediation guidance generation.

It is trained to transform structured vulnerability facts into grounded Chinese analysis.

License:

- repository code: Apache-2.0
- public data snapshots: Apache-2.0
- intended adapter release: Apache-2.0 unless a specific release says otherwise

## Intended Use

Intended uses:

- vulnerability detail explanation
- project vulnerability triage support
- remediation suggestion drafting
- report generation assistance

Not intended for:

- vulnerability discovery
- source-of-truth matching
- arbitrary open-domain security advice

## Base Model

- family: `Qwen2.5-7B-Instruct`
- adaptation method: `LoRA`
- upstream base model license should be reviewed alongside this repository license

## Input Format

Each sample is built from structured fields such as:

- `component_name`
- `component_version`
- `vulnerability_id`
- `risk_level`
- `recommended_version`
- `latest_version`
- evidence and candidate match metadata

## Output Format

The target response is grounded analysis that covers:

- summary
- why affected
- risk assessment
- remediation
- upgrade recommendation

## Data

Public repository dataset snapshots currently include:

- `v1-official-*`
- `v2-frozen-2026-03-19-*`

The current V2 frozen dataset shape is:

- train: `4590`
- validation: `500`
- train/validation overlap: `0`

## Training

Training is performed with:

- `transformers`
- `peft`
- optional 4-bit quantized LoRA
- TensorBoard monitoring

Stable training rules:

- [`TRAINING_SPEC.md`](specs/TRAINING_SPEC.md)

## Evaluation

The project requires evaluation on a frozen non-overlapping validation set.

Success is not defined by loss alone.
The model must also:

- preserve output structure
- stay grounded in input facts
- avoid fabricated relations
- produce usable remediation guidance

## Current Release State

Current state:

- code: public-ready
- public datasets: available in repo
- V1 local adapter: trained locally
- V2 training: complete
- V2 validation report: [`V2_VALIDATION_2026-03-20.md`](V2_VALIDATION_2026-03-20.md)
- public adapter artifact: still pending prediction-level validation

## Quick Inference

Once an adapter release is available, a minimal inference path is:

```bash
python3 scripts/infer_lora.py \
  --base_model_path /path/to/base-model \
  --adapter_path /path/to/adapter \
  --input_file examples/inference_input.json \
  --load_in_4bit
```

## Release Artifacts

Recommended release artifacts:

1. Source repository
2. Public dataset snapshot
3. LoRA adapter weights
4. Validation summary
5. Model card

Recommended artifact hosting:

- GitHub repository for code and docs
- GitHub Releases or a model registry for adapter weights

## Risks And Limitations

Known limitations:

- output quality depends heavily on upstream data quality
- this is not a detector model
- weak validation design can create false confidence
- the model may still produce unsupported claims if poorly constrained

## Governance

Capability boundary:

- [`MODEL_CAPABILITY_SPEC.md`](specs/MODEL_CAPABILITY_SPEC.md)

Publishing rules:

- [`GITHUB_PUBLISHING_GUIDE.md`](specs/GITHUB_PUBLISHING_GUIDE.md)
- [`LICENSE-MODEL`](LICENSE-MODEL)
