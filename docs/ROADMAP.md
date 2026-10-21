# AGLM Roadmap

This document tracks implementation progress. Architectural choices belong in
`docs/decisions/`; this roadmap records what is planned, what is implemented,
and the evidence needed to finish each stage.

## Status key

- **In progress** — some required work exists, but the exit criteria are not met.
- **Not started** — no working implementation exists yet.
- **Done** — exit criteria are met and covered by tests or recorded evaluation.

Do not mark a stage done because its files or directories exist. Update the
status and evidence when implementation work is completed.

## Stages

| # | Stage | Status | Exit criteria |
|---|---|---|---|
| 1 | Foundation | In progress | Config and registry contracts, CI, independently runnable tests, and a small CPU/CUDA/MPS device-selection layer. |
| 2 | Tokenizer | Not started | Persian normalization policy documented; byte-level BPE implementation and registry loading work; compare against `tokenizers`; verify round-trip, determinism, persistence, and append-only vocabulary behavior. |
| 3 | Data | Not started | Licensed sources and provenance recorded; cleaning and deduplication implemented; train/validation split protected from leakage; packed token IDs can be read from sharded `.bin` files. |
| 4 | Model | Not started | RMSNorm, RoPE, GQA, SwiGLU, pre-norm block, and decoder LM implemented with independent tests; causal mask behavior verified; model can overfit a tiny fixed batch as a sanity check. |
| 5 | Training | Not started | AdamW and WSD training loop support gradient accumulation, logging, checkpoint save/resume, and deterministic small-run validation. |
| 6 | Inference | Not started | Greedy generation works first; sampling and KV-cache are added and tested against the uncached path. |
| 7 | Evaluation | Not started | Perplexity and held-out Persian evaluation are reproducible; tokenizer comparisons include a byte-normalized metric; basic benchmark results are recorded. |
| 8 | Model growth | Not started | Vocabulary, context, and depth upgrades have separate migration procedures and tests with clearly stated invariants. |
| 9 | Productization | Not started | SFT and preference-tuning experiments, quantization, export, API, and legal review are evaluated against defined product requirements. |

## Current position

Stage 1 is in progress. The repository has structured YAML profiles, config
loading and validation, an initial tokenizer protocol, a tokenizer registry
mechanism, and tests for config and registry behavior. The model, tokenizer BPE,
data, training, inference, and evaluation implementation files are currently
placeholders. CI and device selection are not implemented.

## Open decisions

- Record the attention visibility pattern explicitly: causal full attention is
  the proposed first baseline; sliding-window attention remains a later option.
- Decide and document the exact Persian normalization policy before training
  the tokenizer, since changing it changes tokenizer behavior and artifacts.

## Updating progress

When a stage changes, update its status and add links to the relevant code,
tests, ADRs, or evaluation results. Keep decisions and implementation status
separate: an accepted ADR does not mean its design has been implemented.
