# Token-gradient selector experiments

## Goal

Learn a training-only policy that selects shortcut-source tokens whose
attention credit should be reversed. The forward Transformer and its inference
behavior remain unchanged. This is a separate follow-up to
[`../learned_backward_shortcuts/PROGRESS.md`](../learned_backward_shortcuts/PROGRESS.md);
that implementation, its checkpoints, and its results are retained.

## Confirmed design

- A two-layer bidirectional Transformer sees only the input prompt.
- It emits one binary selection probability per source token.
- One sampled mask is shared across all forward-model layers and heads.
- For every selected source, attention backward uses multiplier `-alpha`
  instead of `+1` for both value and attention-score credit.
- The output projection's parameter credit is decomposed by attention source
  and receives the same multiplier.
- The ordinary forward pass is bitwise unchanged.
- Start with `alpha=1`.
- Train the selector with grouped policy optimization.
- Reset the forward model for each policy group and share initialization and
  shortcut batches across members.
- Use the same small, fixed clean fitness set in every generation. It is not
  regenerated, and evaluation-only examples never rank policies.
- Begin with a ten-step forward-training horizon and increase it after fitness
  saturates.
- Do not initially penalize the number of selected tokens.

## Gate before policy learning

First test an oracle mask that selects only the answer token immediately after
`<LEAK>`. This checks whether gradient reversal itself can make ordinary Adam
learn the real pointer task while training only on correct-shortcut examples.
The learned selector is justified only if this oracle control beats ordinary
backpropagation on fixed masked and incorrect-shortcut fitness examples.

### Complete-attention oracle result

The first matched two-seed run used `alpha=1` and reversed score, value, and
output-projection parameter credit from the shortcut answer source in every
layer. Ordinary backpropagation reached 100% correct-shortcut accuracy by 80
steps and remained at 8--10% clean accuracy. Complete reversal prevented that
shortcut solution but did not learn the clean task: after 3,000 updates, clean
accuracy was 9.96% for seed 7 and 12.89% for seed 13, close to the 10% chance
level.

This fails the oracle gate for the complete mechanism. Before rejecting token
selection itself, the next diagnostic separates three scopes:

- `attention_scores`: reverse only score/QK credit for edges reading the
  selected source, leaving value and output-projection learning ordinary;
- `qkv`: additionally reverse value credit from the selected source;
- `complete_attention`: additionally reverse that source's contribution to
  the output-projection parameter gradient.

The split checks whether the complete rule poisoned globally shared value
processing rather than simply teaching attention not to read the shortcut
position.

### Score-only oracle passes

The scope diagnostic isolates the useful mechanism. For suffix shortcuts,
score-only reversal reached 100% masked, incorrect-hint, and correct-hint
accuracy by step 320 for both seeds 7 and 13, then remained exactly perfect
through step 3,000. Reversing Q/K/V credit did not work: seed 7 peaked near
22% clean accuracy before collapsing back to 10.35% at step 3,000.

Randomly placing `<LEAK> answer` inside the list is harder but confirms that
the result is not specific to the final source position:

| Condition | Seed | Step 320 clean | Best clean | Step 3,000 clean |
| --- | ---: | ---: | ---: | ---: |
| Ordinary backprop | 7 | 13.87% | 20.12% | 12.50% |
| Score-only oracle | 7 | 95.70% | 95.70% | 93.55% |
| Score-only oracle | 13 | 93.75% | 95.31% | 95.31% |

For random placement, correct-shortcut accuracy is 100% while masked and
incorrect accuracy remain high, so the model learned the real pointer task
rather than merely losing the ability to fit the training examples.

The accepted primitive is therefore narrower than the initial proposal:
reverse selected-source credit only through attention scores
(`QK^T -> softmax`). Value vectors and the output projection must receive
ordinary gradients. The learned selector will use this score-only mechanism.

Tracked output:
[`results/oracle_scope_summary.json`](results/oracle_scope_summary.json).

### Random-placement oracle optimization

The first random-placement runs used forward learning rate `3e-4`. They showed
that score reversal worked but did not establish a perfect oracle: `alpha=4`
reached 99.41% on the fixed set and about 99.1% on fresh 20,000-example sets.
Direct attention inspection showed why the leak was not fully removed.
Although final-query attention to the hint was only 0.0017--1.46% across
layers, intermediate queries still assigned average maximum attention of
8.3--12.4% to it.

Two follow-ups were tested:

- An explicit penalty on selected-source attention mass failed. Weak penalties
  still learned the shortcut; strong penalties damaged shared Q/K learning and
  left the real task near chance.
- Lowering the forward learning rate stabilized score reversal. With
  `alpha=4` and learning rate `1e-4`, both seeds 7 and 13 reached 100% masked,
  incorrect-hint, and correct-hint accuracy by step 320 and stayed perfect
  through at least step 3,000.

The tuned checkpoints were then evaluated on fresh clean examples:

| Seed | Fresh set A (20,000) | Fresh set B (20,000) |
| --- | ---: | ---: |
| 7 | 100.000% | 99.985% |
| 13 | 100.000% | 100.000% |

Ordinary backpropagation's earlier 12.5% aggregate clean score obscured its
failure: it had 25% masked accuracy but 0% incorrect-hint accuracy, so its
worst-mode clean accuracy was 0%.

The oracle gate is accepted. Learned-selector experiments use random leak
placement, score-only reversal, `alpha=4`, and forward learning rate `1e-4`
by default.

Tracked output:
[`results/random_oracle_optimization_summary.json`](results/random_oracle_optimization_summary.json).

### First learned-selector launch: tied-group correction

The first GRPO launch used group size 16 and horizon 10. Across its first four
generations, candidate reward standard deviation was only
`4.1e-5`--`5.7e-5`. Standardizing this numerically tiny spread produced
full-scale advantages despite no meaningful candidate separation. The entropy
bonus then moved both oracle-position and other-position selection
probabilities upward. The run was stopped and rejected after generation 3.

The training loop now treats groups with reward standard deviation below
`1e-4` as ties:

- policy advantages are zero;
- the entropy bonus is not applied;
- selector parameters are not updated;
- three consecutive tied groups promote the forward-training horizon.

This preserves the horizon-10 starting condition without training the policy
on noise. A focused multi-GPU smoke confirms that tied groups leave policy
gradient norm at zero and promote the horizon at the configured patience.

Rejected W&B run:
[`r9o0ejgo`](https://wandb.ai/wobrob101/list-sorting-token-gradient-selector/runs/r9o0ejgo).

### Sparse selector initialization and runtime correction

The corrected tied-group run established that reward separation becomes
measurable around horizon 80, but its initial selector reversed 38--45% of all
tokens. Two consecutive horizon-80 groups showed that selecting either the
oracle leak token or ordinary tokens reduced reward. With no entropy bonus, the
policy therefore learned to turn reversal off globally rather than identify
the leak.

The selector now starts near a configurable 5% reversal probability. This is
an exploration prior, not a sparsity reward: the objective contains no penalty
for selecting many tokens, and the learned policy remains free to become
dense. Fixed-policy correlation probes gave:

| Horizon | Group | Reward std | Reward vs oracle selection | Reward vs other selection |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 64 | 0.000713 | -0.042 | -0.098 |
| 160 | 64 | 0.002756 | +0.035 | -0.103 |

At horizon 160, reversing ordinary tokens is measurably harmful while leak
selection is approximately neutral at the initial 5% rate. This supports a
possible two-phase trajectory in which the selector first suppresses reversal
on ordinary tokens, leaving the useful leak decision relatively enriched.

The original implementation also repeated the same selector Transformer
forward pass once per group member when sampling actions and recomputing policy
log-probabilities. Grouped sampling and policy scoring now share one selector
forward per training batch while retaining independent actions for every group
member. On a matched group-16, horizon-40 benchmark, wall time fell from 49.9
seconds to 20.6 seconds with identical trajectories and rewards. Running
multiple candidates concurrently on each GPU was tested and rejected because
contention made the benchmark slower.

Tracked output:
[`results/sparse_selector_diagnostics.json`](results/sparse_selector_diagnostics.json).

### Vectorized population implementation

The remaining serial candidate loop was replaced with a functional population
path. Parameters and Adam moments carry a leading population axis, and
`torch.vmap` computes independent score-reversal gradients for each member.
The group is sharded over three GPUs by default so GPU 3 remains available to
other users.

The score-only reversal custom backward was first expressed as an equivalent
gradient proxy over live attention weights. A failed prototype accidentally
formed those weights from detached Q/K tensors and produced tied candidates;
that result was rejected. A regression test now requires score reversal to
change Q/K gradients while preserving ordinary value gradients.

The corrected vectorized and serial implementations were compared on the same
generation-7 checkpoint, batches, masks, and random seeds:

| Metric | Serial | Vectorized |
| --- | ---: | ---: |
| Reward std | 0.00177364 | 0.00177359 |
| Best fixed clean accuracy | 23.4375% | 23.4375% |
| Oracle reverse probability | 0.05425819 | 0.05425823 |
| Other reverse probability | 0.05716732 | 0.05716735 |
| Population seconds | 47.32 | 16.67 |
| Wall seconds | 71.06 | 40.48 |

Moving the shared selector sampling and policy update from CPU to GPU 0 then
reduced total wall time to 23.73 seconds for group size 32 and horizon 160 on
three GPUs. This is a 3.0x end-to-end speedup over the matched serial run and
does not occupy the spare fourth GPU.

The trainer now supports exact checkpoint resume, including the fixed fitness
and held-out sets, selector Adam state, horizon state, and existing metrics
history.

### Unattended continuation

The role-level interpretation of the learned policy remains unresolved:
reversing non-leak tokens may be useful if it moves attention toward the
genuine source token, so leak-versus-other selection probability is not by
itself a valid stopping criterion. The decisive evidence remains fixed and
held-out clean performance.

An unattended continuation was launched at `2026-07-27T10:34:06Z` from dense
selector generation 34 with:

- three GPUs (`cuda:0,cuda:1,cuda:2`), leaving GPU 3 available;
- group size 128 in vectorized chunks of 12;
- horizon 160, score-only reversal scale 4, and forward learning rate `1e-4`;
- the original fixed fitness and held-out sets restored from the checkpoint;
- no entropy bonus and reward standard-deviation cutoff `1e-3`;
- a 30-minute external wall-clock limit, with a checkpoint after every
  completed generation.

The run completed 31 generations, from generation 35 through 65. Its outputs
are under
`/exp/exp4/acp21rjf/.scratch/list-sorting-token-reversal/selector_background_30m_h160_g128`.

The continuation did not produce a meaningful reward increase:

| Metric | First 5 generations | Last 5 generations |
| --- | ---: | ---: |
| Mean reward | 0.748195 | 0.751224 |
| Maximum candidate reward | 0.755475 | 0.756892 |
| Best fixed clean loss | 2.297194 | 2.288780 |
| Best fixed clean accuracy | 22.54% | 22.50% |
| Best held-out clean loss | 2.367152 | 2.352241 |
| Best held-out clean accuracy | 19.84% | 19.61% |

Mean reward started at 0.751661 and ended at 0.751948. Although the last-five
average was 0.003029 higher than the first-five average, the fitted trend over
all generations was slightly negative (`-0.000562` per generation) amid large
generation-to-generation variation (`0.601366` to `0.868562`). Fixed and
held-out loss decreased slightly, but neither clean-accuracy series improved.
The longer unattended continuation therefore provides no evidence that this
selector configuration is steadily learning a better credit-assignment rule.
