"""VLM pruning control flow with a fake client (no model): verdicts with keep / done / low confidence / errors,
expiry and closing, end to end in the kinematic engine (python -m pytest tests/pose_inspection -q)."""

import os
import sys

import numpy as np
import pytest

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "scripts", "pose_inspection"))

SCENE = os.path.join(SIM, "assets", "pose_inspection", "B_construction")


class FakeClient:
    """Answers immediately; cycles through reply kinds so every branch of VlmPruner runs."""

    def __init__(self):
        from go2_lab.pose_inspection.vlm import Request  # noqa: PLC0415

        self.Request, self.n, self.log = Request, 0, []

    def submit(self, images, text, t_sim, meta=None):
        r = self.Request(images=images, text=text, t_sim=t_sim, max_new_tokens=0, meta=meta or {})
        k = self.n % 4
        self.n += 1
        ids = list(r.meta.get("id2sector", {}))
        if k == 3:
            r.error = "fake failure"
        else:
            r.parsed = {"keep": [int(i) for i in ids[:2]], "confidence": 0.2 if k == 2 else 0.9,
                        "reason": "test", "done": k == 1}
        r.done, r.latency_s = True, 0.5
        return r


@pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "gt.npz")), reason="scene not built")
def test_vlm_pruner_runs_to_the_end(tmp_path):
    from go2_lab.pose_inspection.config import load_config  # noqa: PLC0415
    from go2_lab.pose_inspection.kinematic import KinematicEngine  # noqa: PLC0415
    from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: PLC0415
    from go2_lab.pose_inspection.session import Session  # noqa: PLC0415
    from go2_lab.pose_inspection.vlm_pruner import VlmPruner  # noqa: PLC0415
    from run_kinematic import starts_for  # noqa: PLC0415

    cfg = load_config("exp/pose_vlm.yaml", {"termination.time_limit_s": 400.0})
    meta = load_scene("assets/pose_inspection/B_construction")
    client = FakeClient()
    objs = meta["objects"]
    pruners = {e: VlmPruner(cfg, client, e, lambda o: objs[o]["class"]) for e in range(2)}
    s = Session(cfg, meta, [0, 1], str(tmp_path), pruner_factory=pruners.__getitem__)
    eng = KinematicEngine(s, starts_for(cfg, meta, [0, 1]))
    eng.run(log=lambda *a: None)
    st = [p.summary() for p in pruners.values()]
    assert client.n > 0 and sum(x["consults"] for x in st) == client.n
    assert sum(x.get("failures", 0) for x in st) > 0 and sum(x.get("accepted", 0) for x in st) > 0
    assert all(np.isfinite(s.summary(e)["Co"]) for e in range(2))
