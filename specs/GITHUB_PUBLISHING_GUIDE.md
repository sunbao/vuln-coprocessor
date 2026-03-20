# GitHub Publishing Guide

## Scope

This guide defines how to prepare this repository for GitHub publication.

It separates:

- content that can be public
- content that must remain private
- how to handle secrets

## Core Rule

This repository must not publish real credentials or private infrastructure details in tracked files.

All real secrets must stay in local environment files such as:

- `.env`

Tracked repository files may include:

- `.env.example`

They must contain placeholders only.

## Public Content

These categories are suitable for GitHub publication after routine review:

- `scripts/`
- `specs/`
- `README.md`
- `requirements-train.txt`
- approved training datasets under `data/`

This project has explicitly approved public training data, so `data/processed/` may be published if no private business identifiers remain.

## Private Content

These categories should not be pushed to GitHub:

- `.env`
- local virtual environments
- model cache
- local checkpoints
- temporary exports
- local logs
- private experiment notes containing internal infrastructure details

In this repository that means at minimum:

- `.venv/`
- `.venv311/`
- `cache/`
- `checkpoints/`
- `exports/`
- `artifacts/`
- `*.log`

## Secret Handling

Do not commit:

- real passwords
- access tokens
- production host credentials

Use:

- `.env` for local real values
- `.env.example` for placeholders

Current required secret names:

- `LASUN_SSH_PASSWORD`
- `LASUN_DB_PASSWORD`

## Infrastructure Details

Internal hosts and topology should be reviewed before publication.

If a document exists only for internal operations, keep it outside the public repository or keep it ignored by Git.

## Publication Checklist

Before publishing, verify all of the following:

1. No tracked file contains real passwords or tokens.
2. `.env` is ignored and only `.env.example` is tracked.
3. `README.md` uses environment variables instead of hard-coded secrets.
4. Large local runtime directories are ignored.
5. Public training data has been reviewed for business-sensitive identifiers.
6. Internal-only run records remain outside the public commit set.

## Current Recommended GitHub Shape

The recommended public repository shape is:

- code
- specs
- sanitized README
- requirements file
- approved public datasets

The recommended private-local shape is:

- caches
- environments
- checkpoints
- local logs
- internal artifacts
