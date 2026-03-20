# tmux Training Runbook

## Why This Exists

This repository should not start long-running training with a short-lived shell session.

In this environment, commands started with temporary execution sessions may terminate when the parent session ends.
That behavior can leave partial TensorBoard events without a completed checkpoint.

Use `tmux` for V2 and later long-running training.

## Canonical Command

Start V2 training and TensorBoard:

```bash
scripts/run_v2_tmux.sh start
```

## Session Names

- training session: `vuln-v2-train`
- TensorBoard session: `vuln-v2-tensorboard`

## Daily Operator Commands

Check status:

```bash
scripts/run_v2_tmux.sh status
```

Attach to the training console:

```bash
scripts/run_v2_tmux.sh attach
```

Attach to the TensorBoard console:

```bash
scripts/run_v2_tmux.sh attach-tb
```

Stop both sessions:

```bash
scripts/run_v2_tmux.sh stop
```

Detach from a tmux session with:

```text
Ctrl+b then d
```

## Output Paths

- training output dir: `checkpoints/qwen25-7b-explain-v2-py311-tmux`
- training log: `artifacts/train_qwen25_v2_py311_tmux.log`
- TensorBoard log: `artifacts/tensorboard_v2_tmux.log`

## Observation Rules

- Open TensorBoard from another machine with `http://<server-ip>:6006`
- Do not use `127.0.0.1` from another machine
- Expect `train/loss` every `10` steps
- Expect `eval/loss` and checkpoint output every `100` steps

## Failure Interpretation

If a run stops and there is no OOM record, no Python traceback, and only partial event files exist, suspect the launch method before suspecting the model or data.

For this repository, `tmux` is the required default for long-running local training.
