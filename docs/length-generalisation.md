# Length-Generalisation Track

## Question

Models train on short online-generated sequences and are evaluated at positions
or lengths not encountered during training. The aim is not merely to fit a
larger test set. It is to identify representations, routing mechanisms, or
optimization methods that lead to a rule whose computation remains valid as
the sequence grows.

The main benchmark family uses list lengths 2-20 for training. Evaluation
lengths vary by experiment, from 40 in early sorting baselines to 5,000 in the
hard-attention checkpoint sweep.

## Task Ladder

The project decomposes sorting into increasingly local questions:

1. **Direct sorting:** emit the ordered list.
2. **Algorithm traces:** emit quicksort or adjacent-operation traces.
3. **Pointer retrieval:** find a marked item or its successor.
4. **Position routing:** recover a pointer address, shift it, and retrieve by
   the shifted address.
5. **Comparison:** retrieve adjacent values and emit `KEEP` or `SWAP`.
6. **Compiled routing:** hard-code the position algorithm, then test transfer.

The detailed formats and old training commands are preserved in the
[core sequence-task notebook](core-sequence-tasks.md). The
[experiment index](experiment-index.md#length-generalisation) gives the current
status of each family.

## Findings

### Local retrieval can extrapolate

The modular pointer pipeline predicts an absolute position as residues modulo
several coprime bases. Random absolute offsets prevent memorizing a small table
of positions. In one seed, the pointer-value stage reached 98.9% exact match at
length 1,000 after training at lengths 2-20. Predicting two adjacent values was
perfect through length 200 and 99.95% at length 400, but fell to 33.0% at
length 1,000. See the
[committed endpoint evaluations](experiment-index.md#modular-pointer-pipeline).

This supports a limited conclusion: local learned routing can extrapolate much
farther than the training range, but composing more autoregressive routing
steps introduces new failure modes.

### Compiled routing transfers when operations align

The compiled RASP-style prefix improves three retrieval-aligned tasks across
three seeds, including associative recall without a pointer token. It does not
improve Dyck-2, and placing the exact blocks in a byte language model is worse
than matched random initialization. The result is operation-specific transfer,
not a generally better initialization. See the
[RASP report](rasp_transfer_report.md) and
[language-model report](language_model_transfer_report.md).

### Minimal hard routes can generalise

Forward-only EGGROLL produced a two-layer model with one active top-1 head in
each layer. After recovering from head pruning, the generation-10,000
checkpoint reached 99.6% at length 400 and 95.3% on a 64-example length-5,000
evaluation. This is the strongest extreme-length result in the repository, but
it is one seed and the longest evaluations are small. See the
[checkpoint sweep](../experiments/hard_attention_eggroll/README.md#checkpoint-length-sweep).

### Sparsity is not sufficient or necessary on pointer-next

The Adam ablations show that dense softmax can match ASEntmax on the simple
pointer-next task. Eight heads matter more than width or sparse normalization
in the tested architecture. Mixed ALiBi/NoPE heads work while all-ALiBi and
all-NoPE controls fail OOD. Standard softmax scaling is sufficient.

These are architecture ablations from one seed. The result does not establish
that mixed heads are necessary for sorting generally, and the proposed
division of positional and content roles remains an interpretation.

### Optimizing a longer-length objective is not yet enough

One-step MAML, QKV-only MAML, persistent routers, horizon-24 MAML, and evolved
backward rules all failed to produce a stable length-400 improvement.
Length-50 performance sometimes improved while length-400 performance remained
poor or collapsed. The evidence supports an optimization failure, but does not
cleanly separate optimization from fitness-objective alignment.

## Evaluation Rules

- Report every evaluated length, not only the best OOD checkpoint.
- State evaluation sample counts; the longest hard-attention cells use 64
  examples and are visibly noisy.
- Keep endpoint results separate from best-checkpoint results.
- Report exact accuracy alongside teacher-forced metrics.
- Include prediction diversity for learned-credit experiments.
- Treat longer lengths used by fitness or meta-learning as training signal,
  not untouched OOD evaluation.
- Keep length 400 reporting-only in the current learned-backward controller.

## Bounded Smoke

```bash
scripts/smoke_length.sh
```

This runs two updates of the sparse-attention pointer-next trainer on one GPU
through `with-gpu`. It checks the model, optimizer, evaluation, and artifact
path. It does not reproduce a scientific result.

## Open Questions

1. Does the eight-head mixed ALiBi/NoPE result replicate across seeds?
2. Which routes are selected by the one-head EGGROLL model at lengths far
   beyond training, and do they match the compiled algorithm?
3. Can a differentiable sparse router learn the same minimal circuit without
   EGGROLL's population cost?
4. Can the modular pipeline compose more operations without accumulating
   autoregressive routing errors?
5. What fitness distribution predicts length-400 behavior better than
   length-50 accuracy?
