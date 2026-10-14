"""Stage 5: object-centric VLM-assisted viewpoint pruning (POSE Sec. IV-C, Alg. 1), with a local VLM.

Per robot and object a session runs OBSERVING -> CONSULTING (async) -> MONITORING:
  * consult inputs (POSE Fig. 4): a top-down map crop (unknown grey / free white / occupied black, the object's mapped
    voxels green, numbered candidates, the robot), up to three first-person sector photos with the mapped object
    voxels re-projected as a green tint, and a text prompt with the sensor geometry, object heights, sector states
    (0 unseen / 1 seen / 255 unreachable), previous verdicts and the candidate table (sector, standoff, height the
    level and the tilted band reach, visibility score)
  * reply: JSON {"keep": [...], "confidence": c, "reason": "...", "done": bool}; accepted when parsable and
    confidence >= min_confidence; it expires after ``ttl_s`` or when the robot completes a visit of the object (then
    the session re-consults); while a consult is pending the previous verdict stays in force (asynchronous)
  * fallback (no verdict yet, VLM failure, low confidence, expired): POSE's geometric default -- every sector's best
    candidate
  * the VLM only selects candidates; positions, headings and postures always come from the geometry (stage 3)
POSE's lightweight heartbeat model is not used: verdicts are refreshed on visits and on expiry instead.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .mapping import FREE2D, OCC2D

PROMPT = """Select inspection viewpoints for one object so a ground robot's range sensor can 3D-scan all of it, \
especially the top. Sensor: 360 deg LiDAR, vertical FOV +/-{vfov:.0f} deg, mounted {mount:.0f} deg nose-up; the body \
can tilt up to {tilt:.0f} deg. Object: {cls}, mapped height {h_map:.1f} m, footprint {fx:.1f} x {fy:.1f} m. \
Sectors around the object (0..{ns_1}): {sectors} (0 unseen, 1 seen, 255 unreachable). \
Previous verdicts: {prev}. Consult #{n}.
IMG 1: top-down map crop around the object (green = mapped object voxels, numbered circles = candidates, \
blue = robot). {photo_note}
Candidates:
{table}
Task: keep <= {max_keep} candidates (at most one per sector) that would scan surfaces not yet green; prune only \
sectors whose side of the object is fully green; if more than half of the object is green, keep at most one \
candidate and set done to true. Reply with JSON only: {{"keep": [candidate numbers], "confidence": 0-1, \
"reason": "one sentence", "done": true/false}}"""


class ObjectSession:
    def __init__(self):
        self.state = "observing"
        self.request = None
        self.verdict = None          # {"keep_sectors": set, "done": bool, "confidence": float, "t": float}
        self.n_consults = 0
        self.t_last = -1e9
        self.history: list[str] = []
        self.visits_at_verdict = 0


class VlmPruner:
    """pruner hook of one robot: ``__call__(agent, obj, model, targets) -> targets``."""

    def __init__(self, cfg, client, env_id: int, class_fn, log_images: int = 0, image_dir: str | None = None):
        self.cfg, self.v = cfg, cfg.planner.vlm
        self.client, self.e = client, env_id
        self.class_fn = class_fn             # object / hypothesis id -> class name
        self.sessions: dict[int, ObjectSession] = {}
        self.frames: dict[tuple[int, int], dict] = {}   # (obj, sector) -> latest camera frame looking at it
        self.t = 0.0
        self.stats = {"consults": 0, "accepted": 0, "fallback_ticks": 0, "verdict_ticks": 0, "failures": 0}
        self.log_images, self.image_dir = log_images, image_dir

    # ---------------------------------------------------------------------------------------------------------------
    def store_frame(self, obj: int, sector: int, frame: dict):
        self.frames[(obj, sector)] = frame

    def _visits(self, agent, o):
        return len(agent.retired.get(o, set()))

    def __call__(self, agent, o: int, m: dict, targets: list):
        t = agent.t_plan if agent.t_plan > -1e8 else 0.0
        s = self.sessions.setdefault(o, ObjectSession())
        # receive a pending reply once the simulated time has caught up with its latency
        if s.state == "consulting" and s.request is not None and s.request.ready(t):
            r = s.request
            ok = r.error is None and isinstance(r.parsed, dict)
            if ok:
                keep_ids = r.parsed.get("keep", [])
                conf = float(r.parsed.get("confidence", 0.0) or 0.0)
                id2sec = r.meta.get("id2sector", {})
                keep = {id2sec[str(k)] for k in keep_ids if str(k) in id2sec} if isinstance(keep_ids, list) else set()
                ok = conf >= self.v.min_confidence
                s.history.append(f"#{s.n_consults}: keep {sorted(keep)} done {bool(r.parsed.get('done'))}")
                if ok:
                    s.verdict = {"keep_sectors": keep, "done": bool(r.parsed.get("done", False)), "confidence": conf,
                                 "t": t}
                    s.visits_at_verdict = self._visits(agent, o)
                    self.stats["accepted"] += 1
            if not ok:
                self.stats["failures"] += 1
            s.state, s.request = "monitoring", None
        if s.state == "closed":
            return []
        # a verdict holds until every kept sector is visited (then the object is closed, POSE "IDLE") or it expires
        if s.verdict is not None:
            left = s.verdict["keep_sectors"] - agent.retired.get(o, set())
            if not left and s.verdict["done"]:
                s.state = "closed"
                self.stats["closed"] = self.stats.get("closed", 0) + 1
                return []
            if not left:  # kept sectors visited but not done: ask again with the new observations
                s.verdict, s.state, s.t_last = None, "observing", -1e9
            elif t - s.verdict["t"] > self.v.ttl_s:
                s.verdict = None
                s.state = "observing"
        if s.state == "monitoring" and s.verdict is None:
            s.state = "observing"
        if s.state == "observing" and targets and t - s.t_last >= self.v.consult_period_s:
            self._consult(agent, o, m, targets, s, t)
        # apply the verdict, or the geometric default
        if s.verdict is None:
            self.stats["fallback_ticks"] += 1
            return targets
        self.stats["verdict_ticks"] += 1
        kept = [tg for tg in targets if tg.sector in s.verdict["keep_sectors"]]
        return kept[: self.v.max_keep]

    # ---------------------------------------------------------------------------------------------------------------
    def _consult(self, agent, o, m, targets, s: ObjectSession, t: float):
        images, id2sector, table = self._inputs(agent, o, m, targets)
        p = self.cfg.planner
        li = self.cfg.lidar
        ns = p.semantic.n_sectors
        sectors = []
        for k in range(ns):
            if k in agent.retired.get(o, set()) or (o, k) in self.frames:
                sectors.append(1)
            elif any(tg.sector == k for tg in targets):
                sectors.append(0)
            else:
                sectors.append(255)
        ext = m["hi"] - m["lo"]
        photos = len(images) - 1
        text = PROMPT.format(
            vfov=li.vfov_deg[1], mount=li.mount_tilt_deg, tilt=self.cfg.robot.pitch_limit_deg,
            cls=str(self.class_fn(o)).replace("_", " "),
            h_map=float(m["hi"][2]), fx=float(ext[0]), fy=float(ext[1]), ns_1=ns - 1, sectors=sectors,
            prev="; ".join(s.history[-3:]) or "none", n=s.n_consults + 1,
            photo_note=(f"IMG 2..{photos + 1}: robot camera photos of the object from sectors "
                        f"{[f[0] for f in self._photo_meta]}; green tint = already scanned." if photos else
                        "No photos yet."), table="\n".join(table), max_keep=self.v.max_keep)
        s.request = self.client.submit(images, text, t, meta={"env": self.e, "obj": int(o), "id2sector": id2sector,
                                                              "consult": s.n_consults + 1})
        s.state, s.t_last = "consulting", t
        s.n_consults += 1
        self.stats["consults"] += 1
        if self.log_images > 0 and self.image_dir:
            import os

            os.makedirs(self.image_dir, exist_ok=True)
            for k, im in enumerate(images):
                im.save(os.path.join(self.image_dir, f"env{self.e}_obj{o}_c{s.n_consults}_img{k}.png"))
            with open(os.path.join(self.image_dir, f"env{self.e}_obj{o}_c{s.n_consults}_prompt.txt"), "w") as f:
                f.write(text)
            self.log_images -= 1

    def _inputs(self, agent, o, m, targets):
        p = self.cfg.planner
        li = self.cfg.lidar
        # candidate table (one row per sector-best candidate)
        table, id2sector = [], {}
        h0 = self.cfg.robot.base_height + li.mount_pos[2]
        up = math.radians(li.vfov_deg[1] + li.mount_tilt_deg)
        tilt = math.radians(self.cfg.robot.pitch_limit_deg)
        for k, tg in enumerate(targets, 1):
            d = tg.standoff
            table.append(f"#{k} sec {tg.sector}, {d:.0f} m, top@level {h0 + d * math.tan(up):.1f} m, "
                         f"top@{-self.cfg.robot.pitch_limit_deg:.0f}deg {h0 + d * math.tan(min(up + tilt, 1.45)):.1f} m, "
                         f"visibility {tg.score:.0f}, posture gain {tg.gain:.0f} vs level {tg.gain0:.0f}")
            id2sector[str(k)] = tg.sector
        images = [self._bev(agent, o, m, targets)]
        self._photo_meta = []
        cands = sorted(((sec, f) for (ob, sec), f in self.frames.items() if ob == o), key=lambda x: -x[1]["t"])
        for sec, f in cands[: self.v.max_photos]:
            images.append(self._tinted(f, m))
            self._photo_meta.append((sec, f["t"]))
        return images, id2sector, table

    def _bev(self, agent, o, m, targets) -> Image.Image:
        nav = agent.nav.nav
        origin, cell = agent.w.nav_origin()
        c = m["centroid"]
        R = max(self.cfg.planner.semantic.standoffs) + 2.0
        x0, y0 = c[0] - R, c[1] - R
        n = int(2 * R / cell)
        ix0, iy0 = int((x0 - origin[0]) / cell), int((y0 - origin[1]) / cell)
        img = np.full((n, n, 3), 128, np.uint8)
        for r in range(n):
            iy = iy0 + r
            if not (0 <= iy < nav.shape[0]):
                continue
            lo, hi = max(0, ix0), min(nav.shape[1], ix0 + n)
            if hi <= lo:
                continue
            row = nav[iy, lo:hi]
            seg = img[r, lo - ix0:hi - ix0]
            seg[row == FREE2D] = 255
            seg[row == OCC2D] = 0
        vox = m["voxels"].cpu().numpy()
        px = ((vox[:, 0] - x0) / cell).astype(int)
        py = ((vox[:, 1] - y0) / cell).astype(int)
        ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
        img[py[ok], px[ok]] = (40, 200, 60)
        im = Image.fromarray(img[::-1]).resize((self.v.bev_px, self.v.bev_px), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        s = self.v.bev_px / n

        def to_px(xy):
            return ((xy[0] - x0) / cell * s, (n - (xy[1] - y0) / cell) * s)

        for k, tg in enumerate(targets, 1):
            u, v = to_px(tg.xy)
            dr.ellipse([u - 9, v - 9, u + 9, v + 9], outline=(220, 30, 30), width=2, fill=(255, 255, 255))
            dr.text((u - 4, v - 6), str(k), fill=(200, 0, 0))
        if agent.nav is not None:
            rx, ry = agent.nav.xy_of(*agent.nav.start) if agent.nav.start is not None else (None, None)
            if rx is not None:
                u, v = to_px((rx, ry))
                dr.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(30, 90, 230))
        return im

    def _tinted(self, f: dict, m: dict) -> Image.Image:
        """Camera photo with the object's mapped voxels re-projected as a solid green tint (each voxel painted at
        its projected size, hidden ones skipped by the depth image)."""
        rgb = f["rgb"].astype(np.float32).copy()
        H, W = rgb.shape[:2]
        vox = m["voxels"].to(f["R_wc"].device)
        pc = vox @ f["R_wc"].T + f["t_wc"]
        z = pc[:, 2]
        ok = z > 0.2
        K = f["K"]
        uv = pc[ok, :2] / z[ok, None] @ K[:2, :2].T + K[:2, 2]
        u, v = uv[:, 0].cpu().numpy(), uv[:, 1].cpu().numpy()
        zz = z[ok].cpu().numpy()
        half = np.maximum(1, 0.5 * float(K[0, 0]) * self.cfg.metrics.scene_voxel / zz).astype(int)
        mask = np.zeros((H, W), bool)
        dep = f.get("depth")
        for ui, vi, zi, h in zip(u.astype(int), v.astype(int), zz, half):
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            if dep is not None and zi > dep[vi, ui] + 0.5:
                continue
            mask[max(0, vi - h):vi + h + 1, max(0, ui - h):ui + h + 1] = True
        rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array([40, 220, 60])
        return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))

    def summary(self) -> dict:
        return dict(self.stats)


def sector_of(obj_centroid, robot_xy, n_sectors: int) -> int:
    a = math.atan2(robot_xy[1] - obj_centroid[1], robot_xy[0] - obj_centroid[0]) % (2 * math.pi)
    return int(a // (2 * math.pi / n_sectors))


def best_camera(cams_yaw_world: list[float], bearing: float) -> int:
    d = [abs((y - bearing + math.pi) % (2 * math.pi) - math.pi) for y in cams_yaw_world]
    return int(np.argmin(d))

