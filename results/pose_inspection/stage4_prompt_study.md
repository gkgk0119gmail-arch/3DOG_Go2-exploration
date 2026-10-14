# Stage 4 prompt study (Grounding DINO base, SAM 2.1 large)

## 1. Single phrases: recall of GT instances by camera distance, false positives per image

| scene | class | phrase | recall 3 m | 6 m | 10 m | 15 m | all | FP / image |
|---|---|---|---|---|---|---|---|---|
| A_warehouse | forklift | forklift | 0.65 | 0.47 | 0.44 | 0.33 | 0.46 | 0.08 |
| A_warehouse | forklift | reach truck | 0.65 | 0.44 | 0.49 | 0.45 | 0.49 | 0.46 |
| A_warehouse | forklift | pallet stacker | 0.65 | 0.46 | 0.44 | 0.34 | 0.46 | 0.74 |
| A_warehouse | forklift | lift truck | 0.65 | 0.44 | 0.49 | 0.45 | 0.49 | 0.35 |
| A_warehouse | forklift | industrial vehicle | 0.68 | 0.47 | 0.51 | 0.47 | 0.52 | 0.69 |
| B_construction | dump_truck | dump truck | 0.83 | 0.92 | 0.90 | 0.50 | 0.81 | 0.85 |
| B_construction | dump_truck | crawler carrier | 0.83 | 0.92 | 0.90 | 0.38 | 0.79 | 0.93 |
| B_construction | dump_truck | tracked dumper | 0.83 | 0.85 | 0.90 | 0.25 | 0.74 | 0.80 |
| B_construction | dump_truck | crawler dump truck | 0.83 | 0.92 | 0.90 | 0.50 | 0.81 | 0.77 |
| B_construction | dump_truck | truck | 0.83 | 0.92 | 0.90 | 0.50 | 0.81 | 0.88 |
| B_construction | excavator | excavator | 0.76 | 0.81 | 0.94 | 0.73 | 0.81 | 0.53 |
| B_construction | excavator | mini excavator | 0.81 | 0.81 | 0.89 | 0.73 | 0.81 | 0.55 |
| B_construction | excavator | digger | 0.81 | 0.81 | 0.94 | 0.73 | 0.83 | 0.55 |
| B_construction | excavator | backhoe | 0.81 | 0.81 | 0.83 | 0.67 | 0.79 | 0.56 |
| B_construction | excavator | construction machine | 0.81 | 0.81 | 0.94 | 0.73 | 0.83 | 0.56 |

## 2. Configured prompt set + SAM 2: pixel metrics of the class masks

prompts: `{"forklift": ["forklift"], "excavator": ["excavator", "digger"], "dump_truck": ["dump truck"]}`

| scene | class | IoU | precision | recall | images with the class |
|---|---|---|---|---|---|
| A_warehouse | forklift | 0.64 | 0.70 | 0.88 | 125 |
| B_construction | dump_truck | 0.39 | 0.61 | 0.53 | 43 |
| B_construction | excavator | 0.45 | 0.55 | 0.71 | 59 |
