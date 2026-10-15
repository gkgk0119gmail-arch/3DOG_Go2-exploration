# stage6_B (isaac engine)


## B_construction  (T = 335 s)

| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |
|---|---|---|---|---|---|---|---|---|
| box (POSE) | 0.43 ± 0.09 | 300 ± 19 | 112 ± 14 | 6.7 ± 2.9 | 0.63 ± 0.02 | 0.78 ± 0.03 | 1.00 ± 0.00 | 6 (0) |
| class template | 0.44 ± 0.07 | 317 ± 9 | 114 ± 7 | 11.7 ± 1.2 | 0.63 ± 0.02 | 0.78 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| template + VLM boom heading | 0.44 ± 0.06 | 314 ± 15 | 112 ± 7 | 10.7 ± 0.8 | 0.63 ± 0.01 | 0.78 ± 0.02 | 1.00 ± 0.00 | 6 (0) |
| box + prune G*<5 | 0.44 ± 0.03 | 178 ± 16 | 83 ± 8 | 5.8 ± 2.9 | 0.55 ± 0.03 | 0.69 ± 0.04 | 1.00 ± 0.00 | 6 (0) |

per-object Co (mean ± std):

| method | excavator_a | excavator_b | dump_truck |
|---|---|---|---|
| box (POSE) | 0.85 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.03 |
| class template | 0.85 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.02 |
| template + VLM boom heading | 0.86 ± 0.00 | 0.86 ± 0.01 | 0.54 ± 0.02 |
| box + prune G*<5 | 0.66 ± 0.12 | 0.65 ± 0.15 | 0.51 ± 0.03 |

| method | module | load | quality |
|---|---|---|---|
| template + VLM boom heading | VLM | 18 calls (0 errors), latency 0.6 s mean / 0.7 s max, tokens 0.8 k per run | accepted verdicts 0, planning ticks on a verdict 0% |
