# GitHub Workflows

This folder contains the automated test workflows for `planktonclass`.

## Overview

There are three workflow layers:

- `tests.yml`
  Fast smoke tests on Python `3.10`, `3.11`, and `3.12`.
- `integration.yml`
  Real package workflow checks on Python `3.10`, `3.11`, and `3.12`.
- `gpu-integration.yml`
  GPU workflow checks on a self-hosted Linux GPU runner.

## What Runs When

### `tests.yml`

Runs on:

- push to `main` or `master`
- pull requests
- manual dispatch

Purpose:

- catch fast regressions in CLI behavior and lightweight package logic
- verify the smoke test suite on all supported Python versions

Python versions:

- `3.10`
- `3.11`
- `3.12`

### `integration.yml`

Runs:

- quick integration on every push and pull request
- full integration by manual dispatch

Quick integration covers:

- `planktonclass init`
- `planktonclass validate-config`
- `planktonclass notebooks`
- `planktonclass train --quick`
- model artifact checks

Full integration adds:

- report generation
- in-process prediction
- API startup check
- Docker image build
- Dockerized prediction

Python versions:

- `3.10`
- `3.11`
- `3.12`

Recommendation:

- rely on quick integration for every code change
- use full integration before release work or after larger refactors

### `gpu-integration.yml`

Runs:

- manual dispatch only

Purpose:

- verify the GPU install path with `pip install -e ".[gpu]"`
- confirm TensorFlow GPU visibility
- run quick training and prediction on a real GPU machine

Python versions:

- `3.10`
- `3.11`
- `3.12`

Default checks:

- install package with GPU extra
- run `planktonclass doctor`
- initialize demo project
- validate config
- copy notebooks
- run quick training
- verify artifacts
- run in-process prediction smoke test

Optional checks:

- GPU Docker image build
- Dockerized API startup
- Dockerized prediction

To include Docker checks, run the workflow with:

- `run_docker = true`

## GPU Runner Requirements

The GPU workflow expects a GitHub Actions self-hosted runner with these labels:

- `self-hosted`
- `linux`
- `gpu`

It also assumes:

- NVIDIA drivers are installed and working
- TensorFlow can see the GPU
- Python `3.10`, `3.11`, and `3.12` are available through `actions/setup-python`

If your runner uses different labels, update the `runs-on` field in `gpu-integration.yml`.

## Suggested Usage

Recommended day-to-day setup:

- every push / PR:
  - `tests.yml`
  - quick part of `integration.yml`
- manual when needed:
  - full `integration.yml`
  - `gpu-integration.yml`

This keeps regular CI fast enough for development while still giving full package coverage and a real GPU validation path.
