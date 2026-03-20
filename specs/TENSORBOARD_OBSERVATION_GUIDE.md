# TensorBoard Observation Guide

## Scope

This guide explains how to observe training from TensorBoard during live runs.

It focuses on:

- how to open the page
- which charts matter
- how to judge whether training is normal
- when to escalate

## Access Rule

If TensorBoard runs on the training server, use the server LAN IP instead of `127.0.0.1`.

Example:

- correct from another machine: `http://192.168.1.228:6006/`
- wrong from another machine: `http://127.0.0.1:6006/`

TensorBoard must bind to:

- `0.0.0.0:6006`

## Correct Log Directory

For this repository, TensorBoard should point to the training output directory root.

Example:

```bash
scripts/run_tensorboard.sh checkpoints/qwen25-7b-explain-v2-py311-rerun
```

Do not point TensorBoard to a non-existent or unused `logs/` subdirectory when the actual event files are written under:

- `output_dir/runs/...`

## What To Open First

Open:

- `Scalars`

Usually ignore at first:

- `Graphs`
- `Text`
- `Images`

## Primary Metrics

Only focus on these first:

- `train/loss`
- `eval/loss`
- `train/grad_norm`
- `train/learning_rate`

## How To Read The Charts

### `train/loss`

Meaning:

- how much training error remains on the current optimization path

Normal pattern:

- noisy but overall downward trend

Warning signs:

- long flatline with no updates
- obvious upward blow-up
- `nan`

### `eval/loss`

Meaning:

- how well the model performs on the frozen validation set

Importance:

- more important than `train/loss` for judging whether the model is improving in a useful way

Normal pattern:

- appears only at the configured eval interval
- stable or gradually improving is acceptable

Warning signs:

- keeps getting worse while `train/loss` keeps improving
- never appears after the expected eval step

### `train/grad_norm`

Meaning:

- the size of the training update signal

Normal pattern:

- fluctuates

Warning signs:

- sustained extreme spikes
- instability together with broken loss curves

### `train/learning_rate`

Meaning:

- confirms the scheduler is behaving as configured

Usage:

- operational confirmation only
- not the main quality metric

## Expected Update Rhythm

Interpret TensorBoard using the training configuration.

Example for the current V2 run:

- `logging_steps=10`
- `eval_steps=100`
- `save_steps=100`

That means:

- training curves should update about every `10` steps
- eval curves should appear about every `100` steps
- checkpoint files should appear about every `100` steps

Do not panic if `eval/loss` is missing before step `100`.

## What Counts As Normal

Treat the run as normal when:

- TensorBoard page opens
- `train/loss` keeps getting new points
- the training process still occupies GPU
- `eval/loss` appears around the configured eval step
- checkpoints appear around the configured save step

## What Counts As A Problem

Escalate when any of these happen:

- TensorBoard page opens but curves stay empty
- no new points appear for a long time
- `train/loss` becomes `nan`
- `eval/loss` degrades sharply and keeps degrading
- configured eval step has passed but no eval metric appears
- configured save step has passed but no checkpoint appears

## Immediate Check Sequence

If the page looks wrong, check in this order:

1. Is the training process still running
2. Is TensorBoard still running
3. Is TensorBoard bound to `0.0.0.0:6006`
4. Is the browser opening `server-ip:6006` instead of `127.0.0.1:6006`
5. Is TensorBoard pointed to the real event-file directory root
6. Does the event file already contain scalar tags

## Minimal Operator Checklist

During a live run, the operator only needs to watch three things:

- whether `train/loss` keeps updating
- whether `eval/loss` appears at the expected step
- whether checkpoints appear at the expected step

That is enough for first-pass observation.
