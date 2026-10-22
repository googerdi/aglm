# AGENTS.md

Instructions for working in the AGLM repository, a modular, decoder-only
language model built with Python and PyTorch. Keep the implementation small,
understandable, independently testable, and suitable for reproducible
experiments.

## Project structure

- `src/aglm/config.py` and `configs/`: configuration loading, validation, and
  model profiles.
- `src/aglm/registry.py` and `src/aglm/tokenizers/`: tokenizer contracts,
  implementations, and registration.
- `src/aglm/model/`: model components and decoder architecture.
- `src/aglm/data/`, `src/aglm/train/`, `src/aglm/inference/`, and
  `src/aglm/eval/`: data, training, generation, and evaluation.
- `tests/unit/`: component-level tests.
- `docs/decisions/`: accepted architecture decisions (ADRs).
- `docs/ROADMAP.md`: implementation status and stage exit criteria.

## Architecture and project decisions

- Read relevant ADRs and the roadmap before changing architecture or declaring
  work complete. ADRs record intended decisions; the roadmap records what is
  implemented. Do not treat a planned or accepted design as implemented without
  code and evidence.
- Keep model components replaceable behind small interfaces and select
  implementations through configuration or registries rather than spreading
  algorithm-specific branches through the pipeline.
- The initial model uses byte-level BPE, tied token embeddings, RoPE, pre-norm
  RMSNorm/SwiGLU blocks, GQA with causal full attention, AdamW, and WSD. Preserve
  configured alternatives and defaults as defined in
  `docs/decisions/0002-model-architecture-and-training.md`.
- Keep tokenizer vocabulary size and model dimensions consistent with the
  effective vocabulary in the tokenizer artifact. Treat normalization or
  pre-tokenization changes as tokenizer-version changes; do not change Persian
  normalization behavior until its policy is documented.
- Preserve reproducibility and checkpoint compatibility. Checkpoints must
  retain the metadata and state needed to identify tokenizer/config versions
  and resume training, as specified in the ADRs.
- Update `docs/ROADMAP.md` when implementation status or stage evidence changes.
  Link evidence such as code, tests, ADRs, or evaluation results; file presence
  alone is not completion.

## Coding rules

- Inspect the project and all callers of touched interfaces before editing.
- Prefer existing project patterns and the smallest complete change; avoid
  unrequested abstractions, options, and dependencies.
- Validate configuration and untrusted inputs at their boundaries. Preserve
  error handling that prevents silent corruption or data loss.
- Add focused unit tests for behavior changes, especially shape/config
  compatibility, causal masking, tokenizer determinism and round-trips,
  persistence, and checkpoint resume where applicable.
- Do not make performance claims without benchmarks on the target hardware.
- Never delete user data or run destructive commands without approval.
- Never read or expose `.env` files, credentials, SSH keys, or tokens. Do not
  use `sudo`.

## Commands

- Install dependencies: `uv sync`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Check formatting: `uv run ruff format --check .`

Run the relevant tests and quality checks after code changes. For documentation-
only changes, inspect the resulting diff instead of running the full test suite.

## Completion

Summarize modified files and checks performed, including any checks not run.
