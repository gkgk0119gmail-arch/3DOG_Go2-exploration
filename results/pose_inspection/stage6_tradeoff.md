# Stage 6: pruning trade-off per unseen-part model


## A_warehouse (T = 582 s)

| method | prune threshold | t_eff [s] | Co | AUCo | N_post |
|---|---|---|---|---|---|
| box + unknown voxels (POSE) | 40 | 216 ± 51 | 0.51 ± 0.13 | 0.41 ± 0.08 | 6.2 |
| box + unknown voxels (POSE) | 20 | 258 ± 18 | 0.57 ± 0.03 | 0.45 ± 0.02 | 8.2 |
| box + unknown voxels (POSE) | 10 | 323 ± 61 | 0.66 ± 0.11 | 0.48 ± 0.06 | 11.3 |
| box + unknown voxels (POSE) | 5 | 338 ± 32 | 0.71 ± 0.05 | 0.51 ± 0.02 | 12.8 |
| box + unknown voxels (POSE) | all | 523 ± 33 | 0.78 ± 0.02 | 0.46 ± 0.02 | 11.7 |
| class template + missing surface (ours) | 800 | 148 ± 27 | 0.36 ± 0.05 | 0.31 ± 0.04 | 0.5 |
| class template + missing surface (ours) | 400 | 155 ± 16 | 0.37 ± 0.04 | 0.32 ± 0.03 | 1.2 |
| class template + missing surface (ours) | 200 | 248 ± 12 | 0.54 ± 0.04 | 0.42 ± 0.02 | 4.8 |
| class template + missing surface (ours) | 100 | 409 ± 23 | 0.71 ± 0.04 | 0.46 ± 0.02 | 9.7 |
| class template + missing surface (ours) | all | 524 ± 25 | 0.74 ± 0.02 | 0.43 ± 0.02 | 9.7 |
| exact CAD + missing surface (upper bound) | 800 | 155 ± 19 | 0.36 ± 0.04 | 0.31 ± 0.03 | 0.5 |
| exact CAD + missing surface (upper bound) | 400 | 192 ± 40 | 0.47 ± 0.08 | 0.39 ± 0.05 | 2.5 |
| exact CAD + missing surface (upper bound) | 200 | 394 ± 16 | 0.76 ± 0.01 | 0.50 ± 0.01 | 8.5 |
| exact CAD + missing surface (upper bound) | 100 | 474 ± 11 | 0.80 ± 0.01 | 0.49 ± 0.00 | 13.5 |
| exact CAD + missing surface (upper bound) | all | 507 ± 16 | 0.80 ± 0.00 | 0.46 ± 0.01 | 13.0 |

## B_construction (T = 330 s)

| method | prune threshold | t_eff [s] | Co | AUCo | N_post |
|---|---|---|---|---|---|
| box + unknown voxels (POSE) | 40 | 140 ± 20 | 0.47 ± 0.06 | 0.39 ± 0.05 | 4.8 |
| box + unknown voxels (POSE) | 20 | 153 ± 17 | 0.48 ± 0.06 | 0.39 ± 0.06 | 5.5 |
| box + unknown voxels (POSE) | 10 | 174 ± 16 | 0.53 ± 0.02 | 0.42 ± 0.03 | 6.5 |
| box + unknown voxels (POSE) | 5 | 181 ± 23 | 0.55 ± 0.04 | 0.42 ± 0.05 | 6.5 |
| box + unknown voxels (POSE) | all | 303 ± 10 | 0.62 ± 0.01 | 0.42 ± 0.08 | 6.8 |
| class template + missing surface (ours) | 800 | 105 ± 17 | 0.39 ± 0.07 | 0.33 ± 0.07 | 2.0 |
| class template + missing surface (ours) | 400 | 139 ± 12 | 0.49 ± 0.06 | 0.40 ± 0.06 | 3.8 |
| class template + missing surface (ours) | 200 | 256 ± 29 | 0.59 ± 0.06 | 0.44 ± 0.07 | 8.5 |
| class template + missing surface (ours) | 100 | 305 ± 18 | 0.62 ± 0.01 | 0.41 ± 0.08 | 10.8 |
| class template + missing surface (ours) | all | 318 ± 14 | 0.63 ± 0.01 | 0.44 ± 0.05 | 11.2 |
| exact CAD + missing surface (upper bound) | 800 | 105 ± 16 | 0.41 ± 0.06 | 0.35 ± 0.06 | 1.0 |
| exact CAD + missing surface (upper bound) | 400 | 165 ± 23 | 0.57 ± 0.02 | 0.43 ± 0.05 | 5.0 |
| exact CAD + missing surface (upper bound) | 200 | 246 ± 4 | 0.61 ± 0.02 | 0.42 ± 0.08 | 8.7 |
| exact CAD + missing surface (upper bound) | 100 | 297 ± 16 | 0.62 ± 0.02 | 0.40 ± 0.09 | 8.7 |
| exact CAD + missing surface (upper bound) | all | 309 ± 6 | 0.63 ± 0.01 | 0.39 ± 0.09 | 8.8 |
