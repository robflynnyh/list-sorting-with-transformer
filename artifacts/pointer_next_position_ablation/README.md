# Pointer-Next Position Ablation

Task: output the value immediately after `<PTR>`. Training lengths were 2-20; evaluation used 2048 examples per length.

| Length | Alternating | Alternating + value RoPE | All-RoPE | All-NoPE |
|---:|---:|---:|---:|---:|
| 20 | 100.0% | 100.0% | 100.0% | 100.0% |
| 40 | 100.0% | 100.0% | 100.0% | 100.0% |
| 60 | 100.0% | 0.0% | 100.0% | 98.3% |
| 80 | 100.0% | 0.0% | 99.9% | 86.6% |
| 100 | 100.0% | 0.0% | 96.0% | 72.9% |
| 110 | 99.9% | 0.0% | 88.1% | 67.1% |
| 120 | 99.7% | 0.0% | 78.5% | 60.8% |
| 130 | 98.9% | 0.0% | 70.9% | 56.9% |
| 140 | 98.6% | 0.0% | 62.4% | 54.4% |
| 150 | 96.6% | 0.0% | 50.0% | 52.0% |
| 160 | 95.4% | 0.0% | 42.3% | 48.5% |
| 180 | 90.8% | 0.0% | 34.6% | 44.2% |
| 200 | 83.3% | 0.0% | 31.6% | 41.7% |
| 300 | 49.1% | 0.0% | 14.5% | 30.3% |
| 400 | 34.8% | 0.0% | 12.2% | 24.6% |

W&B runs:

- Alternating RoPE/NoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/u8vvr26w
- Alternating + value RoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/f1b7hker
- All-RoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/tftyz8tw
- All-NoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/grm6f3ui

Interpretation: rotating values in the alternating RoPE layers hurts the far extrapolation sweep. It still solves lengths 2-40 by the final checkpoint, but drops earlier than the original alternating setup. The original query/key-only alternating RoPE/NoPE remains strongest overall in this seed.

For the value-RoPE run, a finer boundary sweep gave 97.9% at length 45, 27.0%
at length 50, 2.5% at length 55, and 0.0% at length 60. The observed failure
mode at length 60 was immediate `<eos>` generation instead of a value token.
