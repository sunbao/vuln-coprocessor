# Release Checklist

Use this checklist before publishing a repository update or a model release artifact.

## Repository Release

1. Confirm `README.md` reflects the current project state.
2. Confirm `MODEL_CARD.md` reflects the current model and release state.
3. Confirm `CHANGELOG.md` has an entry for the release.
4. Confirm `LICENSE`, `NOTICE`, `LICENSE-DATA`, and `LICENSE-MODEL` are present and accurate.
5. Confirm `.env` is not tracked and `.env.example` contains placeholders only.
6. Confirm no tracked file contains real passwords, tokens, or internal-only credentials.
7. Confirm ignored runtime directories remain untracked:
   - `.venv/`
   - `.venv311/`
   - `cache/`
   - `checkpoints/`
   - `artifacts/`
   - local logs
8. Confirm `data/processed/` only contains approved public data.
9. Confirm the GitHub repository URL is correct in:
   - `README.md`
   - `MODEL_CARD.md`
   - `CITATION.cff`

## Model Release

1. Confirm the adapter or model artifact is actually complete and loadable.
2. Confirm the base model name and license are stated in `MODEL_CARD.md`.
3. Confirm the release artifact license matches the intended release policy.
4. Confirm a minimal inference example works with the intended artifact.
5. Confirm evaluation was run on a frozen non-overlapping validation set.
6. Confirm validation summary and known limitations are documented.
7. Confirm release notes explain:
   - what changed
   - what data snapshot was used
   - what validation set was used
   - what remains risky or incomplete

## Training Release Gate

Do not present a model as release-ready unless all of the following are true:

1. Training completed successfully.
2. At least one evaluation pass completed successfully.
3. Checkpoints or final adapter files were saved correctly.
4. Validation does not rely on overlapping train samples.
5. The repository documents do not overstate model capability.
