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
