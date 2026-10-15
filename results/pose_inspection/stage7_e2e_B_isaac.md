# stage7_e2e_B (isaac engine)


## B_construction  (T = 607 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| POSE (GT labels) | 0.55 ± 0.01 | 285 ± 15 | 101 ± 11 | 9.5 ± 1.4 | 0.64 ± 0.01 | 0.79 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| RL final (GT labels) | 0.60 ± 0.01 | 305 ± 30 | 155 ± 20 | 4.0 ± 0.6 | 0.67 ± 0.01 | 0.83 ± 0.01 | 0.99 ± 0.00 | 6 (0) |
| POSE (Grounding DINO + SAM 2) | 0.31 ± 0.10 | 469 ± 111 | 152 ± 39 | 8.3 ± 2.6 | 0.42 ± 0.03 | 0.52 ± 0.04 | 1.00 ± 0.00 | 6 (0) |
| RL final (Grounding DINO + SAM 2) | 0.38 ± 0.03 | 251 ± 34 | 124 ± 18 | 5.8 ± 1.2 | 0.41 ± 0.04 | 0.51 ± 0.05 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | excavator_a | excavator_b | dump_truck |
|---|---|---|---|
| POSE (GT labels) | 0.85 ± 0.00 | 0.86 ± 0.00 | 0.55 ± 0.02 |
| RL final (GT labels) | 0.86 ± 0.02 | 0.87 ± 0.01 | 0.60 ± 0.01 |
| POSE (Grounding DINO + SAM 2) | 0.00 ± 0.00 | 0.58 ± 0.25 | 0.47 ± 0.02 |
| RL final (Grounding DINO + SAM 2) | 0.00 ± 0.00 | 0.39 ± 0.24 | 0.50 ± 0.01 |

| method | module | load | quality |
|---|---|---|---|
| POSE (Grounding DINO + SAM 2) | perception | frames 2854, 0.48 s/frame | dump_truck: P 0.53 R 0.89; excavator: P 0.86 R 0.12 |
| RL final (Grounding DINO + SAM 2) | perception | frames 1497, 0.43 s/frame | dump_truck: P 0.68 R 0.91; excavator: P 0.83 R 0.09 |
