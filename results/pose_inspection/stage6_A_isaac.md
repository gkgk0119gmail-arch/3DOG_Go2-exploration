# stage6_A (isaac engine)


## A_warehouse  (T = 571 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| box (POSE) | 0.45 ± 0.01 | 520 ± 20 | 205 ± 14 | 12.7 ± 2.3 | 0.78 ± 0.02 | 0.83 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| class template | 0.43 ± 0.02 | 501 ± 19 | 200 ± 10 | 10.3 ± 3.4 | 0.75 ± 0.02 | 0.80 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| template + VLM boom heading | 0.44 ± 0.01 | 519 ± 36 | 213 ± 18 | 9.0 ± 1.4 | 0.74 ± 0.01 | 0.79 ± 0.01 | 1.00 ± 0.00 | 6 (0) |
| box + prune G*<5 | 0.48 ± 0.07 | 326 ± 53 | 150 ± 27 | 12.3 ± 3.8 | 0.65 ± 0.12 | 0.70 ± 0.13 | 0.99 ± 0.01 | 6 (0) |

per-object Co (mean ± std):

| method | forklift_0 | forklift_1 | forklift_2 | forklift_3 | forklift_4 | forklift_5 | forklift_6 |
|---|---|---|---|---|---|---|---|
| box (POSE) | 0.74 ± 0.07 | 0.81 ± 0.06 | 0.79 ± 0.05 | 0.62 ± 0.02 | 0.85 ± 0.05 | 0.79 ± 0.10 | 0.83 ± 0.02 |
| class template | 0.68 ± 0.06 | 0.75 ± 0.05 | 0.74 ± 0.05 | 0.68 ± 0.09 | 0.81 ± 0.07 | 0.81 ± 0.03 | 0.78 ± 0.04 |
| template + VLM boom heading | 0.67 ± 0.05 | 0.76 ± 0.05 | 0.73 ± 0.06 | 0.72 ± 0.08 | 0.75 ± 0.06 | 0.76 ± 0.05 | 0.81 ± 0.01 |
| box + prune G*<5 | 0.57 ± 0.29 | 0.67 ± 0.26 | 0.47 ± 0.38 | 0.59 ± 0.13 | 0.79 ± 0.12 | 0.70 ± 0.18 | 0.76 ± 0.04 |

| method | module | load | quality |
|---|---|---|---|
| template + VLM boom heading | VLM | 42 calls (0 errors), latency 0.7 s mean / 1.0 s max, tokens 2.0 k per run | accepted verdicts 0, planning ticks on a verdict 0% |
