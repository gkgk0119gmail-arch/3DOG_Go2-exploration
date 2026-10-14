# stage5_B (isaac engine)


## B_construction  (T = 323 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| POSE (all sectors) | 0.42 ± 0.09 | 300 ± 19 | 112 ± 14 | 6.7 ± 2.9 | 0.63 ± 0.02 | 0.78 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| POSE + geometric prune | 0.35 ± 0.07 | 134 ± 21 | 65 ± 7 | 4.5 ± 3.8 | 0.44 ± 0.07 | 0.54 ± 0.09 | 1.00 ± 0.00 | 6 (0) |
| VLM v4 (model green) | 0.35 ± 0.06 | 135 ± 14 | 74 ± 10 | 1.5 ± 1.5 | 0.42 ± 0.06 | 0.53 ± 0.08 | 1.00 ± 0.00 | 6 (0) |
| VLM v5 (measured green) | 0.41 ± 0.04 | 215 ± 32 | 96 ± 20 | 3.5 ± 1.0 | 0.56 ± 0.06 | 0.69 ± 0.07 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | excavator_a | excavator_b | dump_truck |
|---|---|---|---|
| POSE (all sectors) | 0.85 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.03 |
| POSE + geometric prune | 0.43 ± 0.34 | 0.52 ± 0.29 | 0.42 ± 0.15 |
| VLM v4 (model green) | 0.60 ± 0.09 | 0.70 ± 0.07 | 0.33 ± 0.09 |
| VLM v5 (measured green) | 0.81 ± 0.04 | 0.81 ± 0.04 | 0.46 ± 0.08 |

| method | module | load | quality |
|---|---|---|---|
| VLM v4 (model green) | VLM | 19 calls (0 errors), latency 2.3 s mean / 2.9 s max, tokens 3.0 k per run | accepted verdicts 19, planning ticks on a verdict 83% |
| VLM v5 (measured green) | VLM | 41 calls (0 errors), latency 1.9 s mean / 2.5 s max, tokens 6.7 k per run | accepted verdicts 41, planning ticks on a verdict 89% |
