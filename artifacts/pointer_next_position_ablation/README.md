# Pointer-Next Position Ablation

Task: output the value immediately after `<PTR>`. Training lengths were 2-20; evaluation used 2048 examples per length.

| Length | Alternating exact | All-RoPE exact | All-NoPE exact |
|---:|---:|---:|---:|
| 20 | 100.0% | 100.0% | 100.0% |
| 40 | 100.0% | 100.0% | 100.0% |
| 60 | 100.0% | 100.0% | 98.3% |
| 80 | 100.0% | 99.9% | 86.6% |
| 100 | 100.0% | 96.0% | 72.9% |
| 110 | 99.9% | 88.1% | 67.1% |
| 120 | 99.7% | 78.5% | 60.8% |
| 130 | 98.9% | 70.9% | 56.9% |
| 140 | 98.6% | 62.4% | 54.4% |
| 150 | 96.6% | 50.0% | 52.0% |
| 160 | 95.4% | 42.3% | 48.5% |
| 180 | 90.8% | 34.6% | 44.2% |
| 200 | 83.3% | 31.6% | 41.7% |
| 300 | 49.1% | 14.5% | 30.3% |
| 400 | 34.8% | 12.2% | 24.6% |

W&B runs:

- Alternating RoPE/NoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/u8vvr26w
- All-RoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/tftyz8tw
- All-NoPE: https://wandb.ai/wobrob101/list-sorting-with-transformer/runs/grm6f3ui

Interpretation: all three solve the training range and length 40 by the final
checkpoint. On the longer sweep, alternating RoPE/NoPE is strongest: it stays
near-perfect through length 140 and remains above 80% at length 200. All-RoPE
starts degrading after length 80-100 and is weakest at very long lengths in
this seed. All-NoPE learns more slowly early in training, but its final
long-length degradation is less severe than all-RoPE beyond length 150.
