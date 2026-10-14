"""Stage 4: open-vocabulary perception replacing the ground-truth labels -- Grounded-SAM 2 on a ring of pinhole
cameras, LiDAR returns projected into the masks.

* ``GroundedSam2``: Grounding DINO (text prompts per class) -> boxes -> SAM 2 masks (transformers implementations)
* ``project_labels``: VLP-16 returns of the same instant projected into every camera of the robot; a return inside a
  mask of class c gets class c (nearest-depth wins when masks overlap; the depth test against the return range
  rejects returns behind the masked object)
* ``ClassToInstance``: class-labelled returns are associated to object hypotheses of the robot's own map (nearest
  model of the same class within ``assoc_radius`` of its box, else a new hypothesis) -- no ground-truth ids
* label persistence: between camera frames, returns falling in voxels already labelled with a hypothesis keep it

Co with perception counts a GT surface voxel of object o only for returns labelled with o's class (wrong-class or
background labels do not count).
"""

from __future__ import annotations

import math

import numpy as np
import torch


class GroundedSam2:
    def __init__(self, cfg, device: str = "cuda:0"):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,  # noqa: PLC0415
                                  Sam2Processor)

        self.cfg, self.dev = cfg, device
        self.det_proc = AutoProcessor.from_pretrained(cfg.detector)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.detector).to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(cfg.segmenter)
        self.sam = Sam2Model.from_pretrained(cfg.segmenter).to(device).eval()

    @torch.inference_mode()
    def detect(self, images: list[np.ndarray], classes: list[str] | None = None):
        """images: list of HxWx3 uint8 -> per image list of (class, score, box xyxy, phrase).

        ``query: joint`` (default): the scene's classes in one prompt, so each box goes to the class phrase it
        matches best (Grounding DINO's own competition; with two short phrases the scores are not diluted much).
        ``query: per_class``: one prompt per class. Overlapping boxes of different classes then keep the higher
        score (IoU > nms_iou)."""
        from PIL import Image  # noqa: PLC0415

        pil = [Image.fromarray(np.ascontiguousarray(im[..., :3])) for im in images]
        dets = [[] for _ in images]
        groups = {c: p for c, p in self.cfg.prompts.items() if classes is None or c in classes}
        queries = [groups] if self.cfg.get("query", "joint") == "joint" else [{c: p} for c, p in groups.items()]
        for q in queries:
            text = " . ".join(ph for phs in q.values() for ph in phs) + " ."
            inp = self.det_proc(images=pil, text=[text] * len(pil), return_tensors="pt").to(self.dev)
            out = self.det(**inp)
            res = self.det_proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=self.cfg.box_threshold, text_threshold=self.cfg.text_threshold,
                target_sizes=[im.shape[:2] for im in images])
            for k, r in enumerate(res):
                labels = r.get("text_labels", r.get("labels"))
                for score, box, lab in zip(r["scores"].tolist(), r["boxes"].tolist(), labels):
                    cls = self._class_of(str(lab), q)
                    if cls is not None:
                        dets[k].append((cls, float(score), box, str(lab)))
        return [self._nms(d) for d in dets]

    @staticmethod
    def _class_of(label: str, group: dict):
        """Class whose phrase words best cover the returned phrase (None when ambiguous / empty)."""
        words = set(label.lower().split())
        if not words:
            return None
        score = {c: max(len(words & set(ph.lower().split())) / len(ph.split()) for ph in phs) for c, phs in group.items()}
        best = sorted(score.items(), key=lambda x: -x[1])
        if best[0][1] == 0 or (len(best) > 1 and best[1][1] == best[0][1]):
            return None
        return best[0][0]

    def _nms(self, d):
        d = sorted(d, key=lambda x: -x[1])
        keep = []
        for x in d:
            if all(_box_iou(x[2], y[2]) <= self.cfg.nms_iou for y in keep):
                keep.append(x)
        return keep

    @torch.inference_mode()
    def segment(self, image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        """[K, H, W] bool masks for the boxes."""
        if not boxes:
            return np.zeros((0, *image.shape[:2]), bool)
        from PIL import Image  # noqa: PLC0415

        inp = self.sam_proc(images=Image.fromarray(image[..., :3]), input_boxes=[boxes], return_tensors="pt").to(self.dev)
        out = self.sam(**inp, multimask_output=False)
        masks = self.sam_proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
        return masks[:, 0].numpy().astype(bool)

    def __call__(self, images: list[np.ndarray], classes: list[str] | None = None):
        """-> per image: list of (class, score, mask)."""
        dets = self.detect(images, classes)
        out = []
        for im, d in zip(images, dets):
            masks = self.segment(im, [x[2] for x in d])
            out.append([(c, s, m) for (c, s, _, _), m in zip(d, masks)])
        return out


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def project_labels(points_w: torch.Tensor, cams: list[dict], seg: list[list], class_ids: dict, H: int, W: int,
                   depth_tol: float, erode_px: int = 0) -> torch.Tensor:
    """Class id per point (-1 none). points_w [N, 3]; cams: dicts with 'R_wc' (3x3 world->cam, OpenCV: z forward,
    x right, y down), 't_wc' [3], 'K' [3, 3], 'depth' [H, W] (distance to image plane, for occlusion)."""
    dev = points_w.device
    lab = torch.full((len(points_w),), -1, dtype=torch.long, device=dev)
    best_z = torch.full((len(points_w),), float("inf"), device=dev)
    for cam, s in zip(cams, seg):
        if not s:
            continue
        pc = points_w @ cam["R_wc"].T + cam["t_wc"]
        z = pc[:, 2]
        ok = z > 0.1
        uv = pc[:, :2] / z.clamp(min=1e-6)[:, None] @ cam["K"][:2, :2].T + cam["K"][:2, 2]
        u, v = uv[:, 0].long(), uv[:, 1].long()
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = torch.nonzero(ok).flatten()
        if len(idx) == 0:
            continue
        uu, vv, zz = u[idx], v[idx], z[idx]
        if cam.get("depth") is not None:
            vis = zz <= cam["depth"][vv, uu] + depth_tol
            idx, uu, vv, zz = idx[vis], uu[vis], vv[vis], zz[vis]
        for c, _, m in s:
            mt = torch.as_tensor(m, device=dev)
            if erode_px > 0:  # drop the mask rim (SAM masks bleed onto the ground / background)
                k = 2 * erode_px + 1
                mt = -torch.nn.functional.max_pool2d(-mt[None, None].float(), k, 1, erode_px)[0, 0] > 0.5
            inm = mt[vv, uu]
            take = inm & (zz < best_z[idx])
            sel = idx[take]
            lab[sel] = class_ids[c]
            best_z[sel] = zz[take]
    return lab


class ClassToInstance:
    """Object hypotheses of one robot's map (no GT ids): class-labelled returns join the nearest hypothesis within
    ``assoc_radius`` of its box whatever their class; a hypothesis' class is the majority of the labels it received
    (an excavator prompt firing on the dump truck is outvoted by the dump-truck labels of the same machine)."""

    def __init__(self, assoc_radius: float, max_objects: int, n_classes: int):
        self.r = assoc_radius
        self.max = max_objects
        self.nc = n_classes
        self.boxes: list[dict] = []    # {"cls", "lo", "hi", "votes"}

    def assign(self, pts: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """pts [N, 3], cls [N] (class ids >= 0) -> hypothesis ids [N]."""
        inst = np.full(len(pts), -1, int)
        todo = np.ones(len(pts), bool)
        for k, b in enumerate(self.boxes):
            d = np.maximum(0, np.maximum(b["lo"][:2] - pts[:, :2], pts[:, :2] - b["hi"][:2])).max(1)
            near = todo & (d <= self.r)
            if near.any():
                inst[near] = k
                todo &= ~near
        while todo.any() and len(self.boxes) < self.max:
            seed = pts[np.flatnonzero(todo)[0]]
            grp = todo & (np.linalg.norm(pts[:, :2] - seed[:2], axis=1) <= 2 * self.r)
            self.boxes.append({"cls": -1, "lo": pts[grp].min(0), "hi": pts[grp].max(0),
                               "votes": np.zeros(self.nc)})
            inst[grp] = len(self.boxes) - 1
            todo &= ~grp
        for k in np.unique(inst[inst >= 0]):
            sel = inst == k
            b = self.boxes[k]
            b["lo"] = np.minimum(b["lo"], pts[sel].min(0))
            b["hi"] = np.maximum(b["hi"], pts[sel].max(0))
            b["votes"] += np.bincount(cls[sel], minlength=self.nc)
            b["cls"] = int(np.argmax(b["votes"]))
        return inst


def ring_camera_yaws(n: int) -> list[float]:
    return [2 * math.pi * k / n for k in range(n)]


class Gsam2Perception:
    """Session perception hook (``Session.perception``): labels VLP-16 returns from Grounded-SAM 2 camera masks.

    The engine calls ``set_frames(env_id, cams)`` whenever a new camera frame is available (cams: list of dicts with
    'rgb' [H, W, 3] uint8, 'depth' [H, W] torch, 'K', 'R_wc', 't_wc' torch, optional 'gt' semantic class map); the
    next ``label`` call of that robot projects its returns into those frames. Returns of later scans inherit the
    hypothesis stored in the map voxel they hit (label persistence).
    """

    def __init__(self, cfg, scene_meta: dict, n_envs: int, device: str = "cuda:0"):
        g = cfg.perception.gsam2
        self.g, self.dev = g, device
        self.classes = sorted({o["class"] for o in scene_meta["objects"] if o["target"]})
        self.class_ids = {c: i for i, c in enumerate(self.classes)}
        self.gt_class = torch.as_tensor([self.class_ids.get(o["class"], -1) if o["target"] else -1
                                         for o in scene_meta["objects"]] + [-1], device=device)
        self.model = GroundedSam2(g, device)
        self.assoc = [ClassToInstance(g.assoc_radius, g.max_objects, len(self.classes)) for _ in range(n_envs)]
        self.frames: dict[int, list[dict]] = {}
        # LiDAR-return label quality at camera frames: tp / fp / fn per class (vs the GT class of the hit triangle)
        self.stats = {c: [0, 0, 0] for c in self.classes}
        # where false-positive labels land: background, non-target clutter, a target of another class
        self.fp_on = {c: [0, 0, 0] for c in self.classes}
        self.is_target = torch.as_tensor([bool(o["target"]) for o in scene_meta["objects"]] + [False], device=device)
        self.n_frames = 0
        self.wall_s = 0.0

    def set_frames(self, env_id: int, cams: list[dict]):
        self.frames[env_id] = cams

    def inst_class(self, e: int) -> torch.Tensor:
        cls = [b["cls"] for b in self.assoc[e].boxes]
        out = torch.full((self.g.max_objects + 1,), -1, dtype=torch.long, device=self.dev)
        if cls:
            out[: len(cls)] = torch.as_tensor(cls, device=self.dev)
        return out

    def label(self, session, t, env, origins, rots, th, obj):
        import time

        P, D = th.shape
        dev = self.dev
        labels = torch.full((P, D), -1, dtype=torch.long, device=dev)
        pts = origins[:, None, :] + (rots @ session.dirs.T).transpose(1, 2) * th[..., None]
        # returns that may carry an object label: within the discovery range and above the floor
        hit = (th < session.cfg.perception.discover_range) & (pts[..., 2] > self.g.min_label_height)
        w = session.world
        todo = []
        for i, e in enumerate(env.tolist()):
            # persistence: returns in voxels that already carry a hypothesis
            ijk, ok = w.voxel_index(pts[i])
            ok &= hit[i]
            labels[i, ok] = w.labels[e, ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].long()
            cams = self.frames.pop(e, None)
            if cams is not None:
                todo.append((i, e, cams))
        if todo:
            t0 = time.time()
            imgs = [c["rgb"] for _, _, cams in todo for c in cams]
            seg_all = self.model(imgs, self.classes)
            H, W = imgs[0].shape[:2]
            k = 0
            for i, e, cams in todo:
                seg = seg_all[k:k + len(cams)]
                k += len(cams)
                sel = torch.nonzero(hit[i]).flatten()
                cls = project_labels(pts[i, sel], cams, seg, self.class_ids, H, W, self.g.depth_tol, self.g.erode_px)
                got = cls >= 0
                if got.any():
                    inst = self.assoc[e].assign(pts[i, sel[got]].cpu().numpy(), cls[got].cpu().numpy())
                    labels[i, sel[got]] = torch.as_tensor(inst, device=dev)
                # quality of the per-return class labels at this frame
                gto = obj[i, sel]
                gtc = self.gt_class[gto]
                for c, ci in self.class_ids.items():
                    p, q = cls == ci, gtc == ci
                    self.stats[c][0] += int((p & q).sum())
                    self.stats[c][1] += int((p & ~q).sum())
                    self.stats[c][2] += int((~p & q).sum())
                    fp = p & ~q
                    self.fp_on[c][0] += int((fp & (gto < 0)).sum())
                    self.fp_on[c][1] += int((fp & (gto >= 0) & ~self.is_target[gto]).sum())
                    self.fp_on[c][2] += int((fp & self.is_target[gto]).sum())
                self.n_frames += 1
            self.wall_s += time.time() - t0
        # Co: a return counts for the GT object it hit only if its hypothesis has that object's class
        cov = torch.full_like(labels, -1)
        for i, e in enumerate(env.tolist()):
            ic = self.inst_class(e)
            lc = torch.where(labels[i] >= 0, ic[labels[i].clamp(min=0)], torch.full_like(labels[i], -1))
            good = (lc >= 0) & (lc == self.gt_class[obj[i]])
            cov[i] = torch.where(good, obj[i], torch.full_like(obj[i], -1))
        return labels, cov

    def summary(self) -> dict:
        out = {"frames": self.n_frames, "wall_s": round(self.wall_s, 1)}
        for c, (tp, fp, fn) in self.stats.items():
            b, cl, oc = self.fp_on[c]
            out[c] = {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
                      "fp_on": {"background": b, "clutter": cl, "other_target": oc}}
        return out
