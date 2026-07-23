# List Sorting with a Transformer

![Architecture of the list-sorting Transformer](assets/architecture.svg)

A small, standalone benchmark for training a decoder-only Transformer to sort
comma-separated symbols. The default experiment trains on list lengths 2-20
and evaluates every length through 40, making the failure or success of length
extrapolation explicit.

## Task

An example is represented as one causal token sequence:

```text
<bos>8,2,5,2=2,2,5,8<eos>
```

Only tokens after `=` contribute to cross-entropy. Inputs are generated online,
duplicates are allowed, and no finite training set is reused.

Two representation settings use the same token IDs and architecture:

- `numbers`: symbols `0` through `9` receive a learned token embedding plus a
  normalized scalar value feature in `[-1, 1]`.
- `alphabet`: symbols `a` through `j` receive only learned token embeddings.
  Their ordering must therefore be inferred from sorting supervision.

This isolates the effect of exposing ordinal structure while keeping the
sequence task unchanged.

## Model

The default model is a 4-layer, 128-dimensional causal Transformer with SwiGLU
feed-forward blocks. It has no learned absolute position table. Attention
layers alternate between:

1. rotary position embeddings (RoPE),
2. no explicit positional encoding (NoPE),
3. RoPE,
4. NoPE.

The implementation can also run all-RoPE or all-NoPE ablations through
`--position-pattern`.

Generation is unconstrained and greedy. Evaluation does not assume valid model
output: it separately measures exact match, comma syntax, output length,
monotonic order, multiset preservation, and target-token accuracy.

## Install

```bash
python -m pip install -e '.[dev]'
pytest
```

## Train

```bash
sort-transformer-train \
  --representation numbers \
  --position-pattern alternating \
  --output-directory artifacts/numbers_alternating_seed7

sort-transformer-train \
  --representation alphabet \
  --position-pattern alternating \
  --output-directory artifacts/alphabet_alternating_seed7
```

Each run writes:

- `checkpoint.pt`: model and optimizer state, ignored by Git;
- `metrics.json`: configuration, training trace, and per-length metrics;
- `training.png`: loss and teacher-forced token accuracy;
- `length_generalization.png`: strict generative performance by length.

Evaluate a checkpoint on another length range with:

```bash
sort-transformer-eval \
  artifacts/numbers_alternating_seed7/checkpoint.pt \
  --lengths 2-60 \
  --output artifacts/numbers_alternating_seed7/eval_2_60.json \
  --plot artifacts/numbers_alternating_seed7/eval_2_60.png
```

## Literature Context

Sorting is simple algorithmically but remains a demanding length-generalization
test for learned sequence models.

- [Thinking Like Transformers](https://proceedings.mlr.press/v139/weiss21a.html)
  expresses sorting in RASP and shows why bounded-vocabulary sorting can be
  implemented through token counting. Its compiled model is intentionally
  structured, unlike the learned baseline here.
- [Improving Length-Generalization in Transformers via Task
  Hinting](https://arxiv.org/abs/2310.00726) reports that ordinary Transformers
  trained on sorting sequences of length at most 20 have near-zero exact
  accuracy at length 100. Auxiliary successor/counting objectives substantially
  improve that extrapolation, showing that the training signal matters.
- [The Impact of Positional Encoding on Length Generalization in
  Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html)
  finds that positional-encoding choice can dominate extrapolation behavior,
  with NoPE outperforming several explicit schemes on its reasoning tasks. The
  alternating RoPE/NoPE stack here is a direct, testable hybrid rather than an
  assumption that either choice is universally best.
- [Neural Execution Engines](https://proceedings.neurips.cc/paper/2020/hash/c8b9abffb45bf79a630fb613dcd23449-Abstract.html)
  finds that vanilla Transformers can fit sorting examples yet fail on longer
  lists, and explores execution-oriented masking as a remedy.
- [Exploring Length Generalization in Large Language
  Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fb7451e43f9c1c35b774bcfad7a5714b-Abstract-Conference.html)
  similarly shows that standard sequence models often need intermediate
  computation structure to extrapolate beyond training lengths.

The first goal of this repository is a clean baseline, not a claimed solution.
The number/alphabet comparison and exact per-length curves establish whether
the model learns sorting, merely exploits exposed numeric order, or discovers a
rule that continues beyond its training horizon.
