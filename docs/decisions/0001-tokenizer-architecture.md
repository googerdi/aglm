# ADR-0001: Tokenizer Architecture

## Status

Accepted — Initial design

## Context

AGLM requires a tokenizer architecture that supports multiple implementations without coupling the dataset, model, or training pipeline to a specific tokenization algorithm.

## Decision

- Define a public `Tokenizer` interface using `typing.Protocol`.
- Implement BPE as the first tokenizer.
- Keep BPE vocabulary learning separate from the generic tokenizer contract.
- Use byte-level BPE as the intended target implementation, developed incrementally.
- Persist tokenizer vocabulary, merge rules, and relevant configuration.
- Keep token IDs and vocabulary size consistent between the tokenizer and language model.

## Testing

Each tokenizer implementation must support independent unit tests for encoding, decoding, vocabulary validity, deterministic behavior, and save/load compatibility.

## Consequences

The architecture allows new tokenizer implementations to be added with minimal changes to the dataset and model. The interface must remain small and should evolve only when new requirements justify a change.