# Shortcut-Learning Track

## Question

The shortcut task gives a pointer model normal list context plus an explicit
hint containing the answer. Ordinary backpropagation rapidly learns the hint
and performs near chance when the hint is removed or made incorrect. The
research question is whether backward credit can be changed so the model still
learns the pointer rule from the same shortcut-containing forward data.

The intervention affects training only. The deployed forward model has the
ordinary Transformer architecture.

## Data Protocol

Four evaluation conditions distinguish real task learning from shortcut use:

| Condition | Meaning |
| --- | --- |
| Correct hint | The shortcut contains the target; high accuracy alone proves nothing. |
| Masked hint | The shortcut is unavailable. |
| Incorrect hint | The shortcut points to a wrong answer. |
| Clean | The normal pointer task without an added shortcut. |

The primary robust score is the minimum of masked-hint and incorrect-hint
accuracy. Correct-hint accuracy and cross-entropy are reported separately.
Prediction mode fraction and distinct prediction count detect constant-output
collapse.

### Fixed-fitness invariant

Forward-training examples containing shortcuts are resampled and effectively
unlimited. The clean fitness dataset used to rank backward-rule perturbations
is deliberately small and **fixed across all evolutionary generations**.

All candidates in a generation see the same fixed clean fitness examples.
Fresh clean examples are reporting-only and must not influence ranking, centre
updates, proposal acceptance, horizon promotion, or sigma adaptation. This
scarcity is part of the method: the learned credit rule must extract a reusable
training bias from limited clean evidence.

## Methods

### Evolved backward rule

The forward model trains with Adam on shortcut data. A separate attention
router changes gradients on causal attention-score edges while leaving the
forward logits unchanged. Antithetic low-rank perturbations of the router are
ranked after complete forward-model training trajectories. Accepted router
updates must improve on independently seeded trajectories.

The strongest replicated checkpoint reaches 93.8% mean worst-mode accuracy at
horizon 320 across 20 fresh forward-model replications, compared with 4.5% for
ordinary backpropagation. See the
[replication evidence](experiment-index.md#evolved-shortcut-credit).

This is a confirmed within-task result. The rule has not been shown to transfer
to another shortcut type or architecture.

### Oracle gradient reversal

When the leak location is supplied as oracle information, reversing only
attention-score credit is sufficient. At the tuned scale, two random-position
seeds reached essentially perfect fresh masked and incorrect-hint accuracy.
Reversing all QKV or complete-attention gradients did not work.

This establishes a mechanism target for learned selectors. It does not make
the oracle a usable shortcut-learning algorithm.

### Learned token selector

A binary policy was trained to select source tokens whose attention-score
credit should be reversed. The vectorized implementation substantially reduced
runtime, but the learned selector did not improve held-out clean accuracy.
The oracle mechanism is therefore viable while policy discovery remains a
negative result.

### MAML router

Persistent router MAML differentiates through hypothetical shortcut-training
updates and then performs one real task-model update. Longer lookahead produced
a strong single-seed suppression-only result, but signed routing was worse and
the approach was less robust than evolutionary search. It remains preliminary.

### Collapse-window evolution

Later experiments targeted trajectory windows where an initially robust rule
collapsed. Candidates could improve selected windows, but the improvements did
not survive the strict on-policy full-trajectory acceptance test. These studies
are retained as inconclusive negative evidence.

## What Counts as Success

A shortcut method must satisfy all of the following:

1. masked-hint accuracy is materially above the 10% digit baseline;
2. incorrect-hint accuracy is materially above the 11.1% exclusion shortcut;
3. prediction diversity rules out constant-output collapse;
4. the unperturbed accepted rule, not only a sampled candidate, passes;
5. the result transfers to fresh forward-model seeds and fresh reporting data;
6. correct-hint accuracy remains compatible with learning the task.

## Bounded Smoke

```bash
scripts/smoke_shortcut.sh
```

This runs two tiny evolutionary generations on one GPU through `with-gpu`. It
checks fixed-fitness construction, antithetic candidate execution, forward
training, evaluation, and checkpoint writing. It is not expected to learn a
shortcut-resistant rule in two generations.

## Open Questions

1. Does the evolved rule transfer to a different shortcut format?
2. Can the rule be compressed into an interpretable fixed operation?
3. Can a learned selector approach the oracle without a population of complete
   training trajectories?
4. Which fixed clean examples are essential, and when does the rule overfit
   that finite fitness set?
5. Can collapse-window objectives predict complete on-policy improvements?
