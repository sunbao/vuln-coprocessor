# V2 Validation 2026-03-20

## Status

This report records the first repository-tracked validation pass for `qwen25-7b-explain-v2-py311-tmux`.

Current conclusion:

- V2 training is complete
- frozen validation loss is available
- train and validation overlap is `0`
- prediction-level validation artifacts are not yet committed
- V2 is not yet accepted as a promotion-ready release

## Training Completion Evidence

Adapter directory:

- [`checkpoints/qwen25-7b-explain-v2-py311-tmux`](/data-2/vuln-coprocessor/checkpoints/qwen25-7b-explain-v2-py311-tmux)

Completion evidence:

- final global step: `574`
- max steps: `574`
- final epoch: `2.0`
- final checkpoint: [`checkpoint-574`](/data-2/vuln-coprocessor/checkpoints/qwen25-7b-explain-v2-py311-tmux/checkpoint-574)
- final adapter: [`adapter_model.safetensors`](/data-2/vuln-coprocessor/checkpoints/qwen25-7b-explain-v2-py311-tmux/adapter_model.safetensors)

## Training Metrics

From [`train_results.json`](/data-2/vuln-coprocessor/checkpoints/qwen25-7b-explain-v2-py311-tmux/train_results.json):

- train_loss: `0.03734066051521683`
- train_runtime_seconds: `25200.0448`
- train_samples: `4590`
- train_steps_per_second: `0.023`

## Eval Metrics

From [`eval_results.json`](/data-2/vuln-coprocessor/checkpoints/qwen25-7b-explain-v2-py311-tmux/eval_results.json):

- eval_loss: `8.458115189569071e-06`
- eval_runtime_seconds: `404.9958`
- eval_samples: `500`
- eval_samples_per_second: `1.235`

## Dataset Integrity

Frozen dataset snapshot:

- [`data/processed/v2-frozen-2026-03-19-train.jsonl`](/data-2/vuln-coprocessor/data/processed/v2-frozen-2026-03-19-train.jsonl)
- [`data/processed/v2-frozen-2026-03-19-validation.jsonl`](/data-2/vuln-coprocessor/data/processed/v2-frozen-2026-03-19-validation.jsonl)
- [`data/processed/v2-frozen-2026-03-19-summary.json`](/data-2/vuln-coprocessor/data/processed/v2-frozen-2026-03-19-summary.json)

Integrity result:

- train_count: `4590`
- eval_count: `500`
- overlap_count: `0`
- overlap_rate: `0.0`

Validation set composition:

- ecosystem: `{'maven': 365, 'npm': 117, 'yarn': 18}`
- risk: `{'高危': 219, '低危': 22, '中危': 184, '超危': 75}`

## What This Validation Already Supports

- V2 finished training successfully
- the frozen validation set is non-overlapping
- the repository now has a usable V2 adapter baseline
- V2 can be compared against later versions under the same validation split

## What This Validation Does Not Yet Support

- promotion of V2 as a released model artifact
- claims about prediction quality based only on loss
- claims about grounding quality without reviewing generated outputs
- starting V3 as if V2 had already passed the validation gate

## Operational Limitation

`scripts/validate_lora.py` requires practical inference capacity for prediction-level validation.

On the current host at the time of this report:

- `torch.cuda.is_available()` was `False`
- GPU-backed validation inference was therefore unavailable
- CPU loading of the 7B base model was technically possible but operationally too slow for the full frozen validation set

Because of that, the repository currently lacks committed V2 artifacts such as:

- `predictions.jsonl`
- `review_sheet.csv`
- `summary.json`
- `SUMMARY.md`

## Gate Decision

- decision: `hold_for_validation`
- reason: V2 has complete training metrics and clean dataset integrity, but prediction-level validation review is not yet committed

## Required Next Step

Before starting the next training version:

- run `scripts/validate_lora.py` on a CUDA-capable host or other practical inference environment
- keep the validation split fixed
- commit the generated validation summary into the repository
- push the validation result to GitHub

Only after that should the next training round be treated as accepted process, not just another run.
