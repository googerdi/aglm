# AG Language Model

A modular language model built from scratch with Python and PyTorch.

## Goals

- Learn language-model architecture and training.
- Implement a small decoder-only Transformer.
- Keep components independently testable.
- Make model size configurable.
- Support reproducible experiments.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

## Getting Started

Clone the repository:

    git clone https://github.com/googerdi/aglm.git
    cd aglm

Install the project and its development dependencies:

    uv sync

Run the test suite:

    uv run pytest

Check code quality:

    uv run ruff check .
    uv run ruff format --check .

## Status

Early development. The initial version focuses on project
structure, configuration validation, and automated testing.