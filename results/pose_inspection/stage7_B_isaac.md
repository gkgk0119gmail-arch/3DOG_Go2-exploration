# stage7_B (isaac engine)


## B_construction  (T = 342 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| POSE (rule-based) | 0.43 ± 0.08 | 300 ± 19 | 112 ± 14 | 6.7 ± 2.9 | 0.63 ± 0.02 | 0.78 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| POSE, known-free viewpoints | 0.49 ± 0.02 | 285 ± 15 | 101 ± 11 | 9.5 ± 1.4 | 0.64 ± 0.01 | 0.79 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| POSE + prune G*<5 | 0.44 ± 0.03 | 178 ± 16 | 83 ± 8 | 5.8 ± 2.9 | 0.55 ± 0.03 | 0.69 ± 0.04 | 1.00 ± 0.00 | 6 (0) |
| POSE + VLM v5 | 0.42 ± 0.04 | 215 ± 32 | 96 ± 20 | 3.5 ± 1.0 | 0.56 ± 0.06 | 0.69 ± 0.07 | 1.00 ± 0.00 | 6 (0) |
| RL run1 it260 (fast) | 0.55 ± 0.01 | 211 ± 16 | 111 ± 12 | 3.7 ± 2.0 | 0.65 ± 0.01 | 0.81 ± 0.02 | 0.99 ± 0.00 | 6 (0) |
| RL run2 it300 (thorough) | 0.56 ± 0.02 | 250 ± 11 | 125 ± 8 | 2.7 ± 1.0 | 0.66 ± 0.02 | 0.82 ± 0.03 | 0.99 ± 0.00 | 6 (0) |
| RL run3 it250 (+ large sites, known-free; final) | 0.54 ± 0.02 | 305 ± 30 | 155 ± 20 | 4.0 ± 0.6 | 0.67 ± 0.01 | 0.83 ± 0.01 | 0.99 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | excavator_a | excavator_b | dump_truck |
|---|---|---|---|
| POSE (rule-based) | 0.85 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.03 |
| POSE, known-free viewpoints | 0.85 ± 0.00 | 0.86 ± 0.00 | 0.55 ± 0.02 |
| POSE + prune G*<5 | 0.66 ± 0.12 | 0.65 ± 0.15 | 0.51 ± 0.03 |
| POSE + VLM v5 | 0.81 ± 0.04 | 0.81 ± 0.04 | 0.46 ± 0.08 |
| RL run1 it260 (fast) | 0.77 ± 0.03 | 0.82 ± 0.01 | 0.60 ± 0.01 |
| RL run2 it300 (thorough) | 0.84 ± 0.01 | 0.85 ± 0.01 | 0.59 ± 0.03 |
| RL run3 it250 (+ large sites, known-free; final) | 0.86 ± 0.02 | 0.87 ± 0.01 | 0.60 ± 0.01 |

| method | module | load | quality |
|---|---|---|---|
| POSE + VLM v5 | VLM | 41 calls (0 errors), latency 1.9 s mean / 2.5 s max, tokens 6.7 k per run | accepted verdicts 41, planning ticks on a verdict 89% |
