"""Stage 6: where are the not-yet-seen parts of an object? Region functions for the posture gain (and the centre the
viewpoints are sampled around), pluggable as ``InspectionAgent.region_fn``.

* ``box``      POSE: the mapped voxels' box inflated by delta_b
* ``template`` ours: a coarse class shape (union of oriented boxes in the object frame, YAML) fitted to the mapped
               voxels -- lower-body heading from a PCA of the low voxels, upper-body heading (excavator boom) from the
               high voxels (or both directions when they do not tell), the template slid away from the observed side
               when the mapped extent is shorter than the template. Region = template (dilated) U inflated box
* ``template_vlm`` the template with the boom heading / height read by the VLM from the first photo (stage 5 client)
* ``cad``      upper bound: the exact surface of the object (GT triangles), dilated by one voxel
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def pca_heading(xy: np.ndarray) -> float:
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(c.T @ c / max(1, len(c)))
    a = v[:, int(np.argmax(w))]
    return math.atan2(a[1], a[0])


class TemplatePrior:
    def __init__(self, cfg, class_fn, scorer, vlm_hint_fn=None):
        self.cfg = cfg
        self.p = cfg.priors
        self.class_fn = class_fn          # object id -> class name
        self.sc = scorer
        self.vlm_hint_fn = vlm_hint_fn    # (agent, obj, model) -> {"upper_yaw": rad, "height": m} or None
        self.fits: dict[tuple[int, int], dict] = {}

    # ---------------------------------------------------------------------------------------------------------------
    def fit(self, agent, o: int, m: dict) -> dict | None:
        cls = self.class_fn(o)
        tpl = self.p.templates.get(cls)
        vox = m["voxels"].cpu().numpy()
        if tpl is None or len(vox) < self.p.min_fit_voxels:
            return None
        low = vox[vox[:, 2] < tpl.low_z]
        base_xy = low[:, :2] if len(low) >= self.p.min_fit_voxels else vox[:, :2]
        yaw = pca_heading(base_xy)
        # extents in the base frame; slide the template away from the observer where the mapped extent is short
        R = _rot2(-yaw)
        uv = base_xy @ R.T
        lo, hi = uv.min(0), uv.max(0)
        half = np.array(tpl.footprint) / 2
        robot = np.asarray(agent.nav.xy_of(*agent.nav.start)) if agent.nav is not None and agent.nav.start else vox[0, :2]
        ru = R @ robot
        center = (lo + hi) / 2
        for k in range(2):
            if hi[k] - lo[k] < 2 * half[k]:
                center[k] = lo[k] + half[k] if ru[k] < lo[k] else (hi[k] - half[k] if ru[k] > hi[k] else center[k])
        c_xy = _rot2(yaw) @ center
        # upper-body heading: direction from the centre towards the high voxels' far end (excavator boom)
        upper = []
        hint = self.vlm_hint_fn(agent, o, m) if self.vlm_hint_fn is not None else None
        height = tpl.height
        if hint is not None and hint.get("upper_yaw") is not None:
            upper = [hint["upper_yaw"]]
            if hint.get("height"):
                height = float(np.clip(hint["height"], 0.5 * tpl.height, 1.5 * tpl.height))
        elif tpl.get("upper_boxes"):
            high = vox[vox[:, 2] > tpl.high_z]
            if len(high) >= self.p.min_fit_voxels:
                d = high[:, :2] - c_xy
                far = d[np.argsort(-np.linalg.norm(d, axis=1))[: max(3, len(d) // 10)]].mean(0)
                if np.linalg.norm(far) > tpl.boom_min_offset:
                    upper = [math.atan2(far[1], far[0])]
            if not upper:  # undetermined: every multiple of 90 deg from the base heading
                upper = [yaw + k * math.pi / 2 for k in range(4)]
        fit = {"cls": cls, "yaw": yaw, "center": c_xy, "upper": upper, "height_scale": height / tpl.height}
        self.fits[(agent.e, o)] = fit
        return fit

    def boxes_world(self, fit: dict):
        """Oriented boxes [(center_xy, yaw, half_xy, z0, z1)] of the fitted template."""
        tpl = self.p.templates[fit["cls"]]
        s = fit["height_scale"]
        out = []
        for b in tpl.boxes:  # [cx, cy, z0, sx, sy, sz] in the base frame
            cxy = fit["center"] + _rot2(fit["yaw"]) @ np.array(b[:2])
            out.append((cxy, fit["yaw"], np.array(b[3:5]) / 2, b[2] * s, (b[2] + b[5]) * s))
        for a in fit["upper"]:
            for b in tpl.get("upper_boxes") or []:
                cxy = fit["center"] + _rot2(a) @ np.array(b[:2])
                out.append((cxy, a, np.array(b[3:5]) / 2, b[2] * s, (b[2] + b[5]) * s))
        return out

    # ---------------------------------------------------------------------------------------------------------------
    def region(self, agent, o, m):
        """Predicted surface of the fitted template on the fine grid (missing-surface gain); box when no fit."""
        fit = self.fit(agent, o, m)
        if fit is None:
            return self.sc.region_box(m["lo"], m["hi"], self.cfg.planner.posture.delta_b)
        boxes = self.boxes_world(fit)
        w, sh = self.sc.w, self.p.shell
        reach = max(float(np.abs(h).max()) * 1.5 for _, _, h, _, _ in boxes) + sh
        lo = np.min([np.r_[c - reach, z0 - sh] for c, _, _, z0, _ in boxes], 0)
        hi = np.max([np.r_[c + reach, z1 + sh] for c, _, _, _, z1 in boxes], 0)
        f0, fs, ctr = fine_box(w, lo, hi)
        inside_any = np.zeros(len(ctr), bool)
        near_any = np.zeros(len(ctr), bool)
        for c, a, h, z0, z1 in boxes:
            uv = (ctr[:, :2] - c) @ _rot2(-a).T
            near_any |= (np.abs(uv) <= h + sh).all(1) & (ctr[:, 2] >= z0 - sh) & (ctr[:, 2] <= z1 + sh)
            inside_any |= (np.abs(uv) < h - sh).all(1) & (ctr[:, 2] > z0 + sh) & (ctr[:, 2] < z1 - sh)
        mask = near_any & ~inside_any & (ctr[:, 2] >= self.p.min_surface_z)
        return {"f0": f0, "fs": fs, "mask": torch.as_tensor(mask.astype(np.uint8), device=self.sc.dev)}

    def center(self, agent, o, m):
        fit = self.fits.get((agent.e, o)) or self.fit(agent, o, m)
        return None if fit is None else fit["center"]


def fine_box(world, lo, hi):
    """Fine-grid index box covering [lo, hi] (clipped) -> (f0, size, voxel centres [N, 3])."""
    fv = world.fine_voxel
    shape = np.array(world.fine_shape)
    f0 = np.clip(np.floor((np.asarray(lo) - world.vg.lo) / fv).astype(int), 0, shape - 1)
    f1 = np.clip(np.ceil((np.asarray(hi) - world.vg.lo) / fv).astype(int), 1, shape)
    fs = np.maximum(f1 - f0, 1)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in fs], indexing="ij"), -1).reshape(-1, 3)
    return f0, fs, world.vg.lo + (f0 + idx + 0.5) * fv


class CadPrior:
    """Exact object surface (GT) as the region -- the upper bound of shape knowledge."""

    def __init__(self, scene_dir: str, scorer, n_obj: int):
        import os

        s = np.load(os.path.join(scene_dir, "surface.npz"))
        self.sc = scorer
        self.pts = {o: s["points"][s["obj"] == o].astype(np.float32) for o in range(n_obj)}

    def region(self, agent, o, m):
        """The object's exact surface on the fine grid (dilated by one fine voxel)."""
        p = self.pts.get(o)
        if p is None or len(p) == 0:
            return self.sc.region_box(m["lo"], m["hi"], agent.p.posture.delta_b)
        w = self.sc.w
        f0, fs, _ = fine_box(w, p.min(0) - w.fine_voxel, p.max(0) + w.fine_voxel)
        ijk = np.floor((p - w.vg.lo) / w.fine_voxel).astype(int) - f0
        mask = np.zeros(tuple(fs), bool)
        for d in np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]):
            q = ijk + d
            ok = ((q >= 0) & (q < fs)).all(1)
            mask[q[ok, 0], q[ok, 1], q[ok, 2]] = True
        return {"f0": f0, "fs": fs, "mask": torch.as_tensor(mask.reshape(-1).astype(np.uint8), device=self.sc.dev)}

    def center(self, agent, o, m):
        p = self.pts.get(o)
        return None if p is None or len(p) == 0 else (p.min(0)[:2] + p.max(0)[:2]) / 2
