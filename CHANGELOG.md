# Changelog

All notable changes to this project should be documented in this file.

The format is inspired by Keep a Changelog.

## [Unreleased]

### Added

- Release-style `README.md`
- `MODEL_CARD.md`
- `LICENSE`, `NOTICE`, `LICENSE-DATA`, `LICENSE-MODEL`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `examples/inference_input.json`
- `scripts/infer_lora.py`
- publishing, capability, training, and TensorBoard observation specs

### Changed

- Repository documentation now follows a small-model release style instead of an internal-only training note style
- Secret handling now uses `.env` and `.env.example`
- Export examples were updated to remove hard-coded credentials

## [0.1.0] - 2026-03-20

### Added

- Initial open-source repository structure for `vuln-coprocessor`
- Public training and validation dataset snapshots
- LoRA training entrypoint
- Validation tooling
- TensorBoard monitoring support

### Notes

- V2 training is in progress
- Public adapter release is not finalized yet
