# ariadne3d — ARiADNE fine-tuned for Go2 turning cost + 3D surface coverage

Fork of the RA-L 2024 ARiADNE training code (`../third_party/large-scale-DRL-exploration`, unmodified copy).
Upstream files are copied as-is; everything new is in its own module.

| file | what |
|---|---|
| `lidar3d.py` | 2.5D world from a 2D training map (obstacle heights + ceiling), tilted VLP-16 ray-march, 3D belief, 3D utility (annulus convolutions) |
| `env3d.py` | env: heading, Go2 turn-then-go time, 3D sweeps while turning / walking, reward, termination |
| `agent3d.py` | node features + `[heading cost, 3D utility]` for the policy (belief) and the critic (ground truth) |
| `worker3d.py` | one episode (same 27-slot buffer as the original) |
| `nets.py` | warm start from the original checkpoint, new input weights = 0 |
| `driver3d.py` | SAC learner + worker processes (no Ray) |
| `eval3d.py` | greedy evaluation on held-out maps (episodes ≥ 5600) |

## Reward (per graph step)

```
r = frontier term (original)  -  t / 16  +  W3D · (new wall + ceiling area) / A_REF  (+20 when 2D is complete)
t = |Δψ| / 1.5 rad/s  +  0.5 s · [|Δψ| > 0.44 rad]  +  distance / 1.0 m/s
```
`-t/16` equals the original `-distance/16` when the robot does not turn. Episode ends when 2D is complete and the
3D utility is gone (or 4 steps in a row add < 2 m²), or after 200 steps. Parameters: `parameter.py` (bottom).

## Run

```bash
python driver3d.py --episodes 4000 --workers 20        # ~4.5 h here; logs train/go2_turn_3d/, ckpt model/go2_turn_3d/
tensorboard --logdir train
python eval3d.py pretrained/ariadne_ral2024.pth model/go2_turn_3d/checkpoint.pth --n 60
```

## Baseline (pretrained, greedy, 15° mount, 40 held-out maps)

2D done 92 % at 346 s (3D then 93.9 %) · end 557 s, 3D 96.3 % (ceiling 92.8 %) · turning 4,717° = 79 s · 478 m.

## Limits of the 2.5D model

Obstacles are solid columns: rack interiors (the 0.5–4 m gaps in the warehouse) and tops above the sensor are
not modelled. Utility ignores line of sight. Sensing is at 1 m / 45° spacing, not 10 Hz.

## Run 1 result (go2_turn_3d, 4000 episodes, ~6.2 h; eval on 60 held-out maps, greedy, 15° mount)

| | 2D not finished | paired on 49 maps both finish: total time | turning | distance | 3D end |
|---|---|---|---|---|---|
| pretrained | 10 / 60 | 485 s | 2,808° | 434 m | 96.1 % |
| ep4000 (`model/go2_turn_3d/checkpoint.pth`) | 1 / 60 | 440 s (−7.4 %, 95 % CI −13…−1) | 2,473° (−7.5 %, −15…+0.4) | 393 m (−7.5 %) | 95.5 % (n.s.) |

Main gain: the greedy pretrained policy gets stuck oscillating on 10 / 60 maps, the fine-tuned one on 1.
Turning / time drop modestly. 3D does not rise: with a 15° mount these 2.5D maps are already at ~96 %
(the real warehouse gaps — rack interiors — are not modelled). ep3500 was worse than ep2000 and ep4000:
pick checkpoints by `eval3d.py`, not by the training curve.
