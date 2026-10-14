# stage5_A (isaac engine)


## A_warehouse  (T = 548 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| POSE (all sectors) | 0.43 ± 0.01 | 520 ± 20 | 205 ± 14 | 12.7 ± 2.3 | 0.78 ± 0.02 | 0.83 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| POSE + geometric prune | 0.42 ± 0.07 | 223 ± 54 | 116 ± 27 | 6.0 ± 2.6 | 0.52 ± 0.11 | 0.56 ± 0.12 | 0.97 ± 0.02 | 6 (0) |
| VLM v4 (model green) | 0.45 ± 0.03 | 286 ± 32 | 151 ± 22 | 4.2 ± 1.2 | 0.58 ± 0.04 | 0.63 ± 0.05 | 0.97 ± 0.03 | 6 (0) |
| VLM v5 (measured green) | 0.45 ± 0.01 | 498 ± 29 | 221 ± 19 | 10.8 ± 2.9 | 0.75 ± 0.03 | 0.80 ± 0.03 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | forklift_0 | forklift_1 | forklift_2 | forklift_3 | forklift_4 | forklift_5 | forklift_6 |
|---|---|---|---|---|---|---|---|
| POSE (all sectors) | 0.74 ± 0.07 | 0.81 ± 0.06 | 0.79 ± 0.05 | 0.62 ± 0.02 | 0.85 ± 0.05 | 0.79 ± 0.10 | 0.83 ± 0.02 |
| POSE + geometric prune | 0.50 ± 0.39 | 0.45 ± 0.17 | 0.41 ± 0.32 | 0.58 ± 0.17 | 0.66 ± 0.22 | 0.40 ± 0.39 | 0.62 ± 0.11 |
| VLM v4 (model green) | 0.72 ± 0.08 | 0.62 ± 0.07 | 0.52 ± 0.17 | 0.53 ± 0.03 | 0.55 ± 0.09 | 0.60 ± 0.16 | 0.55 ± 0.13 |
| VLM v5 (measured green) | 0.74 ± 0.05 | 0.70 ± 0.09 | 0.76 ± 0.07 | 0.71 ± 0.08 | 0.76 ± 0.09 | 0.80 ± 0.09 | 0.78 ± 0.09 |

| method | module | load | quality |
|---|---|---|---|
| VLM v4 (model green) | VLM | 66 calls (0 errors), latency 1.9 s mean / 2.8 s max, tokens 10.0 k per run | accepted verdicts 66, planning ticks on a verdict 94% |
| VLM v5 (measured green) | VLM | 135 calls (0 errors), latency 2.0 s mean / 2.7 s max, tokens 22.3 k per run | accepted verdicts 135, planning ticks on a verdict 93% |
