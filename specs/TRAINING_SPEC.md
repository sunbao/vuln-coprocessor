# Training Spec

## Scope

This file defines the stable training rules for this repository.
It is not an experiment log.

Use this file for:

- training inputs and outputs
- data and evaluation contracts
- version control rules for model iterations

For the stable product goal, capability boundary, and non-goals of the trained model, see:

- [`MODEL_CAPABILITY_SPEC.md`](MODEL_CAPABILITY_SPEC.md)
- [`TENSORBOARD_OBSERVATION_GUIDE.md`](TENSORBOARD_OBSERVATION_GUIDE.md)

Use `artifacts/` for run-specific notes and observations.

## Objective

Train a domain model that produces grounded vulnerability analysis from structured input facts.

The model must:

- follow the expected response format
- stay grounded in the provided facts
- avoid inventing unsupported relations between vulnerabilities and components
- provide actionable remediation guidance when the input supports it

## Data Contract

Training data is provided as local JSONL files.
Each line must be a JSON object.

Required logical fields:

- `instruction`
- `input`
- `output`

Optional metadata:

- `sample_id`
- `task`
- `language`

Current repository-local dataset paths:

- `data/processed/v1-pilot.jsonl`
- `data/processed/v1-real-sample.jsonl`

Training reads local JSONL snapshots only.
If source data originates from an external system, the export and transformation must complete before training starts.

## Data Quality Rules

- Training and validation data must be separated.
- Validation samples must not be copied from training data.
- Output format must be consistent within a dataset version.
- Duplicate or near-duplicate samples should be reduced before training.
- Labeling rules must be stable within a dataset version.

## Evaluation Contract

Every version must be evaluated on a frozen validation set.

Minimum evaluation dimensions:

- format correctness
- factual grounding
- completeness
- remediation quality
- hallucination rate

Minimum review outputs:

- aggregate metrics
- manual error samples
- top failure categories

## Iteration Rules

Training must proceed through controlled versions:

- `v1`: prove the training path and establish a baseline
- `v2`: improve data quality and scale using the same evaluation contract
- `v3`: tune training strategy after data stabilizes
- `v4`: improve robustness and production readiness

Do not change all major variables in one version.
Prefer changing one major axis at a time:

- data
- evaluation
- prompt or output format
- training hyperparameters

## Required Version Record

Every accepted training run must record:

- dataset version
- dataset size
- validation set version
- base model path
- training command
- Python version
- package versions
- checkpoint path
- log path
- final metrics

## Monitoring

Default graphical monitoring should use TensorBoard.

Training runs that need graphical tracking should use:

- `--report_to tensorboard`

TensorBoard event files may be written under the training output directory root, commonly in `runs/`.
Point TensorBoard to the training `output_dir` root unless a run is known to use a different event path.

## Repository Responsibilities

- `README.md` explains how to run training and monitoring
- `specs/` holds stable operating rules
- `artifacts/` holds run records and experiment evidence

Only move content from `artifacts/` into `specs/` after it has been validated and accepted as a stable rule.
