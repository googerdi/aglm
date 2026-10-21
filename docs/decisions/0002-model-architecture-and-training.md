# ADR-0002: Model Architecture and Training Defaults

## Status

Accepted — Initial implementation defaults

## Context

AGLM is a decoder-only language model intended to be understandable in a small
configuration, runnable on modest hardware, and extensible as compute and data
grow. Components should be replaceable behind small interfaces. A default is a
starting point for implementation and measurement, not a claim that it is best
for every scale or hardware target.

## Decision

### Tokenizer and vocabulary

- Start with byte-level BPE and a versioned regex pre-tokenization pipeline.
- Select implementations through a stable registry identifier such as
  `byte_level_bpe`; the identifier maps to a module and a registered class.
- Keep tokenizer training and serialization behind the tokenizer interface so
  alternatives such as Unigram can be added without coupling model or data code
  to BPE internals.
- Define the requested vocabulary size once, in the tokenizer configuration.
  The model's embedding and output dimensions use the tokenizer artifact's
  effective vocabulary size. Persist the tokenizer version or fingerprint with
  every model checkpoint.
- Test round-trip behavior and deterministic output for Persian, mixed-script
  text, Unicode, and arbitrary byte-level inputs. Any normalization or
  pre-tokenization change is a tokenizer-version change.

### Decoder architecture

- Use token embeddings with tied input and output weights by default. Keep
  untied and factorized embeddings as experimental alternatives.
- Use RoPE for positions. Treat YaRN or NTK-style context extension as an
  explicit, evaluated adaptation; changing the configured context alone does
  not guarantee good long-context behavior.
- Use pre-norm residual blocks with RMSNorm and SwiGLU feed-forward layers.
  QK-Norm is optional and disabled in the first baseline. LayerNorm and GeGLU
  remain replaceable alternatives.
- Use GQA with a configurable `num_kv_heads` as the initial attention
  implementation. The first model uses **causal full attention**: each token
  can attend to all earlier tokens in the configured context.
- Keep attention implementation, visibility pattern, and execution backend as
  separate settings. The model config selects the implementation and pattern;
  a runtime setting controls SDPA backend preference. The initial runtime
  backend is `auto`, allowing PyTorch to choose a compatible implementation.
  Do not assume that `scaled_dot_product_attention` always selects
  FlashAttention; backend availability depends on device, dtype, and shape.
- Use one default attention pattern for all layers and optional per-layer
  pattern overrides for a future hybrid model. Sliding-window attention
  requires a positive window size and an implementation that applies the
  corresponding mask or sparse attention path; it is not enabled by changing
  configuration alone. Full attention has quadratic compute in sequence
  length even when a fused backend avoids materializing the full attention
  matrix.
- Keep the Transformer block dependent on a small sequence-mixer interface,
  rather than on GQA internals. This provides a boundary for later attention
  variants or sequence mixers such as SSMs without requiring those variants in
  the first implementation. Cache state and mask behavior must remain explicit
  in that interface.
- Keep a simple attention implementation available for learning and
  correctness tests.
- Use a documented, depth-aware residual initialization compatible with the
  chosen pre-norm architecture. Keep initialization in a dedicated module.
  μP is a separate research path, not a promised migration: its
  parameterization and width-transfer rules must be chosen before running
  width-scaling experiments.
- Use next-token cross-entropy as the training objective. Keep z-loss optional
  and disabled by default until experiments show a stability benefit.

### Optimization and execution

- Use AdamW as the default optimizer. Evaluate Muon separately; if adopted,
  apply it only to eligible matrix parameters and retain AdamW for embeddings,
  normalization parameters, biases, and other excluded parameters.
- Use WSD (warmup, stable, decay) as the default schedule because the stable
  phase supports continued training from intermediate checkpoints. Keep cosine
  available as a comparison baseline; record schedule state in checkpoints.
- Use bf16 where supported, fp32 for CPU and unsupported hardware, and consider
  fp8 only as a later hardware-specific experiment. Mixed precision and
  `torch.compile` are independent settings. Establish correct eager execution
  before enabling compilation, and measure compiled runs against it.

### Data, checkpointing, and inference

- Read training data as a stream where practical, tokenize and pack examples
  into fixed-length blocks, and support sharded binary token-ID files suitable
  for memory mapping. Store dtype, vocabulary/tokenizer fingerprint, split,
  and shard-boundary metadata; prevent train/validation leakage.
- Use safetensors for model-weight artifacts and distribution. A resumable
  training checkpoint must also preserve optimizer and schedule state, step,
  random-number-generator state, and data/sampler position. Store non-tensor
  metadata in a versioned configuration or state file alongside the tensor
  files.
- Start inference with KV-cache and configurable sampling: temperature,
  top-p, and min-p. Add speculative decoding and paged KV only after profiling
  demonstrates a need.
- Keep model components independently configurable and testable. New
  implementations should be selected through configuration or registries,
  rather than by adding algorithm-specific branches throughout the pipeline.

## Consequences

- The first end-to-end model stays small enough to study while using components
  that have clear upgrade paths.
- Hardware-dependent optimizations remain optional, so CPU and laptop use is
  still possible.
- Vocabulary changes affect model embedding/output dimensions. Extending a
  vocabulary therefore requires an explicit weight-growth and checkpoint
  migration procedure; it is not a tokenizer-only change.
- Architecture alternatives must be compared with reproducible configurations,
  tests, and measurements before replacing a default.

## Validation

Implementation work should add independent tests for component behavior,
shape/config compatibility, deterministic save/load, causal masking, tokenizer
round-trips, checkpoint resume, and eager-versus-optimized output parity where
applicable. Performance claims require benchmarks on the target hardware.
