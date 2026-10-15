# stage7_e2e_A (isaac engine)


## A_warehouse  (T = 1200 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| POSE (GT labels) | 0.62 ± 0.02 | 465 ± 12 | 196 ± 12 | 12.3 ± 3.2 | 0.76 ± 0.02 | 0.81 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| RL final (GT labels) | 0.64 ± 0.03 | 445 ± 36 | 259 ± 25 | 6.7 ± 1.5 | 0.74 ± 0.03 | 0.79 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| POSE (Grounding DINO + SAM 2) | 0.29 ± 0.05 | 1200 ± 0 | 299 ± 28 | 17.8 ± 4.8 | 0.42 ± 0.07 | 0.45 ± 0.08 | 0.99 ± 0.01 | 6 (6) |
| RL final (Grounding DINO + SAM 2) | 0.53 ± 0.04 | 1110 ± 220 | 599 ± 108 | 10.7 ± 2.5 | 0.67 ± 0.05 | 0.71 ± 0.05 | 1.00 ± 0.00 | 6 (5) |

per-object Co (mean ± std):

| method | forklift_0 | forklift_1 | forklift_2 | forklift_3 | forklift_4 | forklift_5 | forklift_6 |
|---|---|---|---|---|---|---|---|
| POSE (GT labels) | 0.77 ± 0.05 | 0.80 ± 0.06 | 0.73 ± 0.03 | 0.59 ± 0.04 | 0.82 ± 0.05 | 0.77 ± 0.12 | 0.84 ± 0.02 |
| RL final (GT labels) | 0.76 ± 0.06 | 0.71 ± 0.11 | 0.72 ± 0.06 | 0.68 ± 0.11 | 0.73 ± 0.11 | 0.79 ± 0.08 | 0.77 ± 0.04 |
| POSE (Grounding DINO + SAM 2) | 0.42 ± 0.37 | 0.46 ± 0.37 | 0.56 ± 0.37 | 0.34 ± 0.18 | 0.26 ± 0.40 | 0.31 ± 0.39 | 0.63 ± 0.31 |
| RL final (Grounding DINO + SAM 2) | 0.62 ± 0.15 | 0.66 ± 0.17 | 0.70 ± 0.13 | 0.67 ± 0.08 | 0.71 ± 0.06 | 0.59 ± 0.31 | 0.72 ± 0.01 |

| method | module | load | quality |
|---|---|---|---|
| POSE (Grounding DINO + SAM 2) | perception | frames 7200, 0.48 s/frame | forklift: P 0.59 R 0.75 |
| RL final (Grounding DINO + SAM 2) | perception | frames 6660, 0.44 s/frame | forklift: P 0.51 R 0.76 |
