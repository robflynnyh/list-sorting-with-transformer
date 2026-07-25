# Compiled Middle-Layer Transfer to Language Modelling

## Executive summary

This experiment asks whether the exact two-layer routing circuit from the
compiled pointer comparator is a better initialization than random weights for
language modelling.

We trained a six-layer, 1.63M-parameter causal Transformer as a byte-level
language model on text read directly from The Pile at
`/store/store4/data/thepile/00.jsonl`. The transfer model places the two
compiled blocks in layers 3-4:

```text
Random:          byte input -> R -> R -> R  -> R  -> R -> R -> byte logits
Compiled middle: byte input -> R -> R -> C1 -> C2 -> R -> R -> byte logits
```

For a given seed, every parameter outside layers 3-4 starts identically in the
two conditions. Both models see identical training batches and use the same
optimizer. All learned parameters, including the compiled blocks, are
fine-tuned.

The compiled initialization did **not** improve language modelling:

| Initialization | Validation BPC | Byte perplexity | Next-byte accuracy |
| --- | ---: | ---: | ---: |
| Random | **3.026 +/- 0.127** | **8.169 +/- 0.734** | **41.28% +/- 2.37** |
| Compiled middle | 3.095 +/- 0.125 | 8.563 +/- 0.726 | 40.33% +/- 2.32 |

Values are means and sample standard deviations over seeds 7, 17, and 29 at
5,000 optimizer updates. Lower bits per byte (BPC) and perplexity are better;
higher accuracy is better. Random initialization finishes better for every
paired seed. The mean paired difference is `compiled - random = +0.068 +/-
0.081 BPC`.

This is a useful boundary on the positive transfer observed in the
[puzzle-task benchmark](rasp_transfer_report.md): a routing circuit helps tasks
that reuse locate-and-retrieve operations, but it is not a generally superior
language-model initialization in this setup.

![Validation learning curves](assets/language_model_transfer_learning_curves.png)

## Experimental question

The previous transfer benchmark put the two useful compiled blocks at the
start of a four-layer model. The strongest transfer occurred when the
downstream task reused their routing behavior.

Here the two blocks are moved into the middle of a deeper network. Random
layers before them can, in principle, transform byte representations into the
coordinate system expected by the circuit, while random layers after them can
convert its output into language-model features. This tests a stronger claim:

> Can an exact algorithmic routing module serve as a useful internal prior for
> ordinary next-token prediction?

The matched random control tests the same architecture and optimization budget
without that prior.

## Data

The benchmark reads the existing Pile JSONL file:

```text
/store/store4/data/thepile/00.jsonl
```

The source is opened read-only. No file under `/store/store4` is created,
rewritten, or deleted.

Documents are split deterministically:

- document indices divisible by 20 are assigned to validation;
- all other document indices are assigned to training;
- streams are truncated to 32,000,000 training bytes and 4,000,000 validation
  bytes;
- each document is UTF-8 encoded and separated by two newline bytes.

The resulting immutable inputs are:

| Split | Documents used | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Train | 4,954 | 32,000,000 | `b51c1b120938a38f6e323dc63a7a87c51d2d9ed88e1b517844ecde97d308f264` |
| Validation | 815 | 4,000,000 | `35a765a9bbdb53c142bcdb0dc5a8bb1e71ae64cff3a0d495df274d7df154f199` |

Raw UTF-8 bytes are the tokens, giving a fixed vocabulary of 256 and avoiding
tokenizer training, external downloads, or vocabulary mismatches between
runs. Validation contains complete documents disjoint from training.

## Model and initialization

Both conditions use:

| Setting | Value |
| --- | ---: |
| Transformer layers | 6 |
| Model width | 128 |
| Attention heads | 4 |
| SwiGLU multiplier | 4 |
| Total parameters | 1,625,960 |
| Trainable parameters | 1,625,344 |
| Context length | 256 bytes |
| Dropout | 0 |

The input residual stream is split into 64 byte-content dimensions and 64
position dimensions. Both conditions receive the same fixed modular Fourier
position representation using moduli `(2, 3, 5, 7, 11, 13, 17, 19)`. Its
period is 9,699,690 positions, so addresses do not repeat within the 256-byte
context.

### Random

All learned parameters use the project's normal Gaussian initialization.

### Compiled middle

The target model is first created with exactly the same seeded random
initialization as the control. Target layers 3-4 are then replaced:

- target layer 3 receives source compiled block 1, which locates the pointer
  and copies its modular address;
- target layer 4 receives source compiled block 2, which shifts that address
  and retrieves values.

The source uses the gentler transfer configuration from the earlier benchmark:
pointer-selection logit 20, address-score scale 5, and pointer scratch scale 1.

The source token embedding is not copied because the source vocabulary contains
pointer-puzzle symbols while this model consumes arbitrary bytes. The
position-coordinate system is shared. This isolates transfer of the two
compiled Transformer blocks.

## Training

Each condition uses the same settings:

| Setting | Value |
| --- | ---: |
| Seeds | 7, 17, 29 |
| Optimizer updates | 5,000 |
| Batch size | 64 |
| Bytes per sequence | 256 |
| Training bytes processed per run | 81,920,000 |
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Warmup | 100 updates |
| Final learning rate | `3e-5` |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | BF16 |

The training stream is sampled with replacement. Evaluation uses 20 fixed
batches per seed, or 327,680 validation target bytes at every checkpoint. The
same seed produces the same training and evaluation windows in both
initialization conditions.

## Results

### Final paired results

| Seed | Random BPC | Compiled-middle BPC | Compiled - random | Random accuracy | Compiled accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | **3.172** | 3.196 | +0.024 | **38.57%** | 38.43% |
| 17 | **2.937** | 2.955 | +0.018 | **42.93%** | 42.91% |
| 29 | **2.971** | 3.133 | +0.162 | **42.33%** | 39.66% |
| Mean | **3.026** | 3.095 | +0.068 | **41.28%** | 40.33% |

All three BPC differences favor random initialization. The effect is small for
seeds 7 and 17 and much larger for seed 29, so the mean should not be read as a
precise population estimate from only three seeds. The paired BPC differences
have sample standard deviation 0.081.

### Learning curve

| Update | Random BPC | Compiled-middle BPC | Compiled - random |
| ---: | ---: | ---: | ---: |
| 0 | **8.049** | 8.054 | +0.005 |
| 10 | **8.028** | 8.034 | +0.006 |
| 50 | **6.445** | 6.780 | +0.335 |
| 100 | 5.054 | **5.043** | -0.011 |
| 250 | 4.977 | **4.976** | -0.002 |
| 500 | **4.172** | 4.247 | +0.075 |
| 1,000 | **3.894** | 3.942 | +0.049 |
| 2,000 | **3.398** | 3.488 | +0.090 |
| 5,000 | **3.026** | 3.095 | +0.068 |

There is a narrow interval around updates 100-250 where the compiled mean is
ahead by at most 0.011 BPC. It does not persist. The control is better from
update 500 onward.

![BPC advantage over training](assets/language_model_transfer_advantage.png)

Throughput is effectively matched: random averages 0.801M training bytes/s and
compiled-middle averages 0.789M bytes/s at the endpoint, a 1.5% difference.
The result is therefore not explained by one condition receiving materially
more optimizer updates or data.

## Why transfer did not help

The evidence supports two related explanations, but does not distinguish their
relative importance.

### 1. The source algorithm is poorly aligned with byte prediction

The compiled blocks implement a very specific operation: identify a designated
marker, copy its modular address, shift that address by fixed offsets, and
retrieve values. Raw language modelling has no designated pointer marker and
requires a broad mixture of local statistics, induction, syntax, and semantic
prediction.

Random outer layers could theoretically translate language features into the
compiled coordinate system, but doing so may be harder than learning ordinary
attention features directly. This agrees with the earlier benchmark: compiled
transfer was strong for retrieval-aligned tasks and absent for stack-like
Dyck-2.

### 2. Exact compilation creates an optimization handicap

The compiler zeros every unused path. At initialization:

| Middle-layer component | Random nonzero weights | Compiled nonzero weights |
| --- | ---: | ---: |
| Attention | 131,072 / 131,072 | 336 / 131,072 |
| SwiGLU | 393,216 / 393,216 | 0 / 393,216 |

Both matrices in each compiled SwiGLU path are zero. Under backpropagation,
this is a dead initialization: the output matrix sees zero hidden activations,
and the input matrix sees zero signal through the output matrix. AdamW weight
decay also leaves an exact zero at zero. Consequently, 393,216 parameters, or
24.2% of the nominal model, cannot be recruited during this run.

The compiled middle blocks drift only `0.137 +/- 0.013` relative to their
initial norm, compared with `0.499 +/- 0.052` for random middle blocks. Some of
that difference is expected from the compiled blocks' larger initial norm, but
it is also consistent with the sparse circuit being difficult to repurpose.

The negative result therefore applies to transplanting the **exact compiled
blocks**, not to every possible algorithm-informed initialization.

## Conclusions

1. Placing the exact pointer-routing circuit in the middle of a six-layer
   language model does not improve validation loss, accuracy, or learning speed
   over matched random initialization.
2. Random initialization finishes better in all three paired seeds, with a
   mean advantage of 0.068 BPC and 0.94 percentage points of next-byte
   accuracy.
3. Positive transfer from the earlier benchmark does not extend automatically
   to unrelated language modelling.
4. Exact sparse compilation is a poor fine-tuning substrate when it zeroes
   both sides of otherwise useful residual branches.

## Limitations and next experiment

- Only three seeds, one 1.63M-parameter model, one byte-level corpus slice, and
  5,000 updates were tested.
- The model uses fixed modular positions to preserve the compiler's coordinate
  system; this is not a standard language-model positional setup.
- The byte-level task tests generic sequence modelling but not modern
  subword-token language modelling.
- The experiment deliberately copies whole exact blocks, so task mismatch and
  dead capacity are confounded.

The cleanest follow-up is a three-way ablation:

1. fully random;
2. exact compiled middle blocks, as tested here;
3. compiled middle attention with normally initialized SwiGLUs.

The third condition would preserve the routing prior while restoring the
393,216 dead feed-forward parameters. A further option is to blend compiled
weights with a small random residual rather than inserting exact zeros.

## Reproduction

Run the six experiment cells through the cooperative GPU scheduler:

```bash
bash experiments/language_model_transfer/run_matrix.sh
```

Regenerate the aggregate JSON and plots:

```bash
PYTHONPATH=src python -m list_sorting_transformer.language_model_transfer \
  summarize \
  --input-root artifacts/language_model_transfer \
  --output-directory experiments/language_model_transfer/results
```

The committed aggregate is
[`experiments/language_model_transfer/results/summary.json`](../experiments/language_model_transfer/results/summary.json).
Raw per-run logs and metrics remain uncommitted under the local `artifacts/`
directory.
