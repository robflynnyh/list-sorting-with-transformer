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
