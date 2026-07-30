# Adam + ASEntmax pointer-next baseline

This experiment trains the same clean pointer-next task as
`experiments/hard_attention_eggroll`, but uses ordinary Adam and the sparse
attention proposed by Vasylenko et al., *Long-Context Generalization with
Sparse Attention* (ICLR 2026).

## Attention

- Exact alpha-entmax with `alpha=1.5`.
- Adaptive-Scalable Entmax (ASEntmax): the NoPE heads receive learned
  per-query `beta` and `gamma` scales of the form
  `1 + softplus(beta) * log(position)^gamma`.
- NAPE: half of the heads use linear ALiBi slopes and half use NoPE.
- The matched modular random-offset input positions are retained so that the
  attention optimizer is compared on the same pointer-next representation.

The task model remains a two-layer, four-head, `d_model=128` SwiGLU
Transformer. Training samples list lengths uniformly from 2 through 20.
Evaluation reports lengths 2, 5, 10, 20, 40, 100, 400, 1,000, and 2,000
throughout training, then length 5,000 at the end. The recurring 1,000- and
2,000-token evaluations use 64 fixed examples instead of 512 because this
attention implementation materializes its sparse score matrix before entmax.

## Run

GPU jobs must use the launcher:

```bash
GPU_POOL=2 experiments/sparse_attention_adam/run_pointer_next.sh
```

The default is 20,000 Adam steps with the paper's Sort learning rate `4e-4`,
betas `(0.9, 0.99)`, zero weight decay, 1,000 warmup steps, and cosine decay.
Checkpoints are written every 1,000 steps.

Useful W&B metrics:

- `eval/length_N/accuracy`: accuracy at each list length.
- `attention/support_size`: mean number of nonzero keys per query.
- `attention/support_fraction`: nonzero fraction among causally valid edges.
- `attention/alibi_support_size` and `attention/nope_support_size`: support
  sizes for the two NAPE head groups.
- `attention/beta_mean`, `attention/gamma_mean`, and `attention/scale_mean`:
diagnostics for adaptive length scaling.

The local entmax implementation uses the exact analytic Jacobian in backward.
Differentiating through the closed-form support threshold directly can produce
NaN gradients when quantized bf16 scores land exactly on a support boundary.
Training also fails immediately on any future non-finite gradient rather than
writing corrupted checkpoints.

Paper: <https://arxiv.org/abs/2506.16640>

Reference implementation: <https://github.com/deep-spin/asentmax>
