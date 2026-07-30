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
- The original matched baseline retains modular random-offset input positions
  so that the attention optimizer is compared on the same pointer-next
  representation.

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

## Paper-matched variant

`run_paper_matched_pointer_next.sh` keeps the pointer-next task and its
length-2-to-20 training range, but matches the paper's Sort experiment more
closely:

- two Gemma2-style decoder blocks with `d_model=256`, eight heads, and a
  1,024-dimensional gated GELU feed-forward layer;
- four linear-ALiBi heads and four NoPE heads, without separate modular input
  positions or scalar number features;
- full bf16 parameters and activations;
- AdamW with batch size 128, learning rate `4e-4`, betas `(0.9, 0.99)`, zero
  weight decay, 20,000 warmup steps, and cosine decay over 312,500 updates;
- seed 4.

Run it with:

```bash
GPU_POOL=0 experiments/sparse_attention_adam/run_paper_matched_pointer_next.sh
```

This is not an exact reproduction of the paper's Sort result. It predicts the
value after a pointer from lists of length 2 through 20, rather than
autoregressively sorting sequences of length 32 through 64 with a 32-symbol
vocabulary. It also uses the local closed-form entmax implementation rather
than the reference AdaSplash kernel.

The initial paper-matched run reached 100% accuracy at every recurring
evaluation length from 2 through 2,000 by step 5,000:
<https://wandb.ai/wobrob101/list-sorting-sparse-attention-adam/runs/gtjlqz5k>.
The run was stopped after confirming that this remained stable through later
checkpoints.

## Key-difference ablation

`run_key_difference_ablations.sh` evaluates a cumulative sequence of changes
at step 5,000 with seed 4:

1. previous model and training recipe;
2. paper training recipe only;
3. NAPE-only token inputs;
4. paper model width and head count;
5. Gemma2-style blocks at the original capacity;
6. width 256 with four heads;
7. width 128 with eight heads;
8. dense softmax in place of entmax for the successful eight-head model.

Results:

| Variant | L40 | L100 | L400 | L1000 | L2000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old recipe, width 128, 4 heads, modular input | 93.8% | 50.4% | 15.8% | 9.4% | 15.6% |
| Paper recipe only | 15.0% | 9.8% | 11.9% | 6.2% | 9.4% |
| NAPE input, width 128, 4 standard heads | 55.9% | 27.5% | 13.1% | 9.4% | 7.8% |
| NAPE input, width 128, 4 Gemma2 heads | 97.7% | 52.9% | 19.9% | 12.5% | 10.9% |
| NAPE input, width 256, 4 standard heads | 100% | 97.7% | 78.9% | 56.2% | 50.0% |
| NAPE input, width 256, 8 standard heads | 100% | 100% | 100% | 100% | 98.4% |
| NAPE input, width 128, 8 standard heads | 100% | 100% | 100% | 100% | 100% |
| NAPE input, width 128, 8 standard heads, softmax | 100% | 100% | 100% | 100% | 100% |
| NAPE input, width 256, 8 Gemma2 heads | 100% | 100% | 100% | 100% | 100% |

The decisive tested factor is eight attention heads, not increased model width
or the Gemma2 block. With the standard small `d_model=128` Transformer,
changing four 32-dimensional heads to eight 16-dimensional heads is sufficient
for perfect accuracy through length 2,000. This ablation does not yet separate
the benefit of more heads from the benefit of a smaller per-head dimension.
Replacing entmax with dense softmax leaves accuracy at 100% throughout, so
sparse normalization provides no demonstrated accuracy or length-generalization
benefit on this task. It may still have computational benefits with the paper's
sparse kernel, which this local closed-form implementation does not use.

Controls and their individual W&B runs are in:
<https://wandb.ai/wobrob101/list-sorting-sparse-attention-ablation>.

Paper: <https://arxiv.org/abs/2506.16640>

Reference implementation: <https://github.com/deep-spin/asentmax>
