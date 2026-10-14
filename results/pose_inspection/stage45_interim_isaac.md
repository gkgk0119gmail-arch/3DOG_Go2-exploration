# stage45_interim (isaac engine)


## A_warehouse  (T = 1200 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| GT labels (stage 3) | 0.62 ± 0.02 | 520 ± 20 | 205 ± 14 | 12.7 ± 2.3 | 0.78 ± 0.02 | 0.83 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| Grounded-SAM 2 v0 | 0.44 ± 0.04 | 1032 ± 100 | 325 ± 36 | 23.5 ± 6.9 | 0.72 ± 0.04 | 0.77 ± 0.04 | 1.00 ± 0.00 | 6 (1) |
| VLM v2 (A) | 0.61 ± 0.02 | 655 ± 45 | 287 ± 21 | 12.5 ± 1.4 | 0.79 ± 0.02 | 0.85 ± 0.02 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | forklift_0 | forklift_1 | forklift_2 | forklift_3 | forklift_4 | forklift_5 | forklift_6 |
|---|---|---|---|---|---|---|---|
| GT labels (stage 3) | 0.74 ± 0.07 | 0.81 ± 0.06 | 0.79 ± 0.05 | 0.62 ± 0.02 | 0.85 ± 0.05 | 0.79 ± 0.10 | 0.83 ± 0.02 |
| Grounded-SAM 2 v0 | 0.71 ± 0.05 | 0.70 ± 0.06 | 0.72 ± 0.06 | 0.67 ± 0.10 | 0.70 ± 0.12 | 0.80 ± 0.04 | 0.77 ± 0.04 |
| VLM v2 (A) | 0.75 ± 0.07 | 0.82 ± 0.04 | 0.76 ± 0.07 | 0.74 ± 0.07 | 0.85 ± 0.06 | 0.79 ± 0.10 | 0.84 ± 0.02 |

| method | module | load | quality |
|---|---|---|---|
| Grounded-SAM 2 v0 | perception | frames 6188, 0.45 s/frame | forklift: P 0.60 R 0.77 |
| VLM v2 (A) | VLM | 198 calls (0 errors), latency 2.1 s mean / 2.9 s max, tokens 33.7 k per run | accepted verdicts 198, planning ticks on a verdict 71% |

## B_construction  (T = 463 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| GT labels (stage 3) | 0.48 ± 0.07 | 300 ± 19 | 112 ± 14 | 6.7 ± 2.9 | 0.63 ± 0.02 | 0.78 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| Grounded-SAM 2 v0 | 0.27 ± 0.09 | 378 ± 71 | 141 ± 27 | 8.2 ± 2.1 | 0.34 ± 0.12 | 0.42 ± 0.15 | 1.00 ± 0.00 | 6 (0) |
| VLM v1 (B) | 0.48 ± 0.02 | 361 ± 46 | 147 ± 24 | 6.7 ± 2.3 | 0.63 ± 0.02 | 0.78 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| VLM v0 (B) | 0.46 ± 0.08 | 387 ± 23 | 155 ± 7 | 6.5 ± 2.1 | 0.64 ± 0.02 | 0.79 ± 0.02 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | excavator_a | excavator_b | dump_truck |
|---|---|---|---|
| GT labels (stage 3) | 0.85 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.03 |
| Grounded-SAM 2 v0 | 0.01 ± 0.02 | 0.05 ± 0.04 | 0.46 ± 0.17 |
| VLM v1 (B) | 0.81 ± 0.14 | 0.87 ± 0.01 | 0.54 ± 0.01 |
| VLM v0 (B) | 0.87 ± 0.01 | 0.87 ± 0.01 | 0.55 ± 0.02 |

| method | module | load | quality |
|---|---|---|---|
| Grounded-SAM 2 v0 | perception | frames 2293, 0.52 s/frame | dump_truck: P 0.42 R 0.92; excavator: P 0.81 R 0.11 |
| VLM v1 (B) | VLM | 75 calls (0 errors), latency 2.3 s mean / 4.0 s max, tokens 14.3 k per run | accepted verdicts 75, planning ticks on a verdict 91% |
| VLM v0 (B) | VLM | 170 calls (0 errors), latency 2.1 s mean / 3.2 s max, tokens 33.5 k per run | accepted verdicts 168, planning ticks on a verdict 86% |
