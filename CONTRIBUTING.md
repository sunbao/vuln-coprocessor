# Contributing

## Scope

This repository accepts contributions that improve:

- training code
- evaluation tooling
- documentation
- public dataset quality
- release usability

## Contribution Rules

Before opening a change:

1. Keep the model capability boundary intact.
2. Do not commit local secrets, caches, checkpoints, or private artifacts.
3. Do not weaken the train/validation separation rules.
4. Prefer small, reviewable changes over mixed refactors.

## Pull Request Guidance

A good contribution should state:

- what changed
- why it changed
- how it was validated

If the change affects model behavior, include:

- dataset impact
- evaluation impact
- any known tradeoffs

## Data Contributions

If you contribute public data:

- confirm the data is approved for public release
- avoid private business identifiers
- keep labeling rules consistent with the existing contract

## Training And Evaluation Changes

If you modify training or evaluation behavior:

- update `README.md` when usage changes
- update `specs/` only when the rule is meant to become stable
- keep experiment-specific notes in local or release-specific records

## Licensing

By contributing to this repository, you agree that your contribution may be
distributed under the Apache License, Version 2.0.
