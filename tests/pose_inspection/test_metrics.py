"""Sanity checks for the POSE-inspection metrics (python -m pytest tests/pose_inspection -q)."""

import os
import sys

import numpy as np
import pytest
import torch

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.config import Cfg  # noqa: E402
from go2_lab.pose_inspection.metrics import SurfaceCoverage, Termination, auc  # noqa: E402

SCENE = os.path.join(SIM, "assets", "pose_inspection", "B_construction")
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "surface.npz")), reason="scene not built")


def test_auc_hold_and_normalisation():
    curve = [(0.0, 0.0), (10.0, 0.5), (20.0, 1.0)]
    # 0 on [0,10), 0.5 on [10,20), 1.0 on [20,40) -> (0 + 5 + 20) / 40
    assert auc(curve, 40.0) == pytest.approx(25.0 / 40.0)
    # values after t_end hold the value at t_end
    assert auc(curve, 40.0, t_end=15.0) == pytest.approx(0.5 * 30.0 / 40.0)


def test_termination_window():
    cfg = Cfg(cs_threshold=0.9, window_s=30.0, min_gain=0.001, time_limit_s=1200.0)
    term = Termination(cfg, 2)
    for t in range(0, 200):
        cs = np.array([min(0.95, 0.01 * t), 0.5])
        term.update(float(t), cs)
    # env 0 reaches 0.95 at t = 95 and is flat afterwards -> done when the 30 s window is flat, t_eff = window start
    assert term.done[0] and term.t_eff[0] == pytest.approx(95.0, abs=1.0)
    assert not term.done[1]


@needs_scene
def test_surface_coverage_full_and_targets():
    s = np.load(os.path.join(SCENE, "surface.npz"))
    import json

    meta = json.load(open(os.path.join(SCENE, "scene.json")))
    targets = [o["target"] for o in meta["objects"]]
    sc = SurfaceCoverage(os.path.join(SCENE, "surface.npz"), targets, 2, "cuda:0" if torch.cuda.is_available() else "cpu")
    p = torch.as_tensor(s["points"], device=sc.device)
    o = torch.as_tensor(s["obj"].astype(np.int64), device=sc.device)
    sc.mark(torch.zeros(len(p), dtype=torch.long, device=sc.device), p, o)
    assert sc.co()[0].item() == pytest.approx(1.0)
    assert sc.co()[1].item() == 0.0
    # a return labelled with the wrong object does not count
    sc.reset()
    sc.mark(torch.zeros(len(p), dtype=torch.long, device=sc.device), p, (o + 1) % sc.n_obj)
    assert sc.co()[0].item() < 0.05
