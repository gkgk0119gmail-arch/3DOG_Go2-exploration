"""Render robot-height camera views of every target with GT semantic segmentation (stage 4 prompt study data).

    python scripts/pose_inspection/render_views.py A_warehouse [--dists 3 6 10 15] [--n_az 8]

For each target, cameras at ``--cam_height`` look horizontally (``--pitch_up`` deg up) at the object from n_az
azimuths x dists (positions inside obstacles are skipped). Saves logs/pose_inspection/views/<scene>.npz with rgb,
semantic class map (by label), the target / distance / azimuth of each view.
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--dists", nargs="+", type=float, default=[3.0, 6.0, 10.0, 15.0])
ap.add_argument("--n_az", type=int, default=8)
ap.add_argument("--cam_height", type=float, default=0.42)
ap.add_argument("--pitch_up", type=float, default=10.0)
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--img_height", type=int, default=384)
ap.add_argument("--hfov", type=float, default=90.0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--warmup", type=int, default=12)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402
from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: E402


def decode(seg_raw, info):
    info = info[0] if isinstance(info, list) else info
    raw = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    lab = {k: (v.get("class", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw.items()}
    if raw and str(next(iter(raw))).startswith("("):
        seg = seg_raw[..., 0].astype(np.int64) & 0xFFFFFFFF
        id2 = {}
        for k, v in lab.items():
            r, g, b, a = (int(x) for x in str(k).strip("()").split(","))
            id2[r | (g << 8) | (b << 16) | (a << 24)] = v
    else:
        seg, id2 = seg_raw[..., 0], {int(k): v for k, v in lab.items()}
    return seg, id2


def main():
    meta = load_scene(f"assets/pose_inspection/{args.scene}")
    nav = NavGrid(os.path.join(meta["dir"], "grid.npz"), 0.3)
    views = []
    for ob in meta["objects"]:
        if not ob["target"]:
            continue
        c = np.array(ob["center"])
        for d in args.dists:
            for k in range(args.n_az):
                a = 2 * math.pi * k / args.n_az
                xy = c[:2] + d * np.array([math.cos(a), math.sin(a)])
                if nav.clearance(xy)[0] < 0.3:
                    continue
                views.append((ob["name"], ob["class"], d, k, xy, c))
    print(f"{len(views)} views")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    scfg = sim_utils.UsdFileCfg(usd_path=os.path.join(meta["dir"], meta["files"]["scene"]))
    scfg.func("/World/Scene", scfg)
    for k in range(args.batch):
        sim_utils.create_prim(f"/World/Cam_{k}", "Xform")
    f = 20.955 / 2 / math.tan(math.radians(args.hfov) / 2)
    cam = Camera(CameraCfg(prim_path="/World/Cam_[^/]+/Sensor", update_period=0, height=args.img_height,
                           width=args.width, data_types=["rgb", "semantic_segmentation"],
                           renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False),
                           spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=20.955,
                                                            clipping_range=(0.1, 200.0))))
    sim.reset()
    rgbs, segs, names = [], [], []
    for s in range(0, len(views), args.batch):
        chunk = views[s:s + args.batch]
        eyes, looks = [], []
        for v in chunk:
            xy, c = v[4], v[5]
            dist = np.linalg.norm(c[:2] - xy)
            eyes.append([xy[0], xy[1], args.cam_height])
            looks.append([c[0], c[1], args.cam_height + dist * math.tan(math.radians(args.pitch_up))])
        while len(eyes) < args.batch:  # pad
            eyes.append(eyes[-1])
            looks.append(looks[-1])
        cam.set_world_poses_from_view(torch.tensor(eyes, dtype=torch.float32, device=sim.device),
                                      torch.tensor(looks, dtype=torch.float32, device=sim.device))
        for _ in range(args.warmup):
            sim.step()
            cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"].cpu().numpy()[..., :3]
        seg, id2 = decode(cam.data.output["semantic_segmentation"].cpu().numpy(), cam.data.info["semantic_segmentation"])
        for j in range(len(chunk)):
            rgbs.append(rgb[j])
            # class map: index into a fixed class list (0 = other)
            segs.append(seg[j])
            names.append(id2)
        print(f"  {min(s + args.batch, len(views))}/{len(views)}", flush=True)
    classes = sorted({o["class"] for o in meta["objects"] if o["target"]})
    cls_maps = []
    for sg, id2 in zip(segs, names):
        m = np.zeros(sg.shape, np.uint8)
        for i, lab in id2.items():
            if lab in classes:
                m[sg == i] = classes.index(lab) + 1
        cls_maps.append(m)
    out = os.path.join(SIM, "logs", "pose_inspection", "views")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{args.scene}.npz"), rgb=np.stack(rgbs), cls=np.stack(cls_maps),
                        classes=np.array(classes), target=np.array([v[0] for v in views]),
                        target_class=np.array([v[1] for v in views]), dist=np.array([v[2] for v in views]),
                        az=np.array([v[3] for v in views]))
    print("wrote", os.path.join(out, f"{args.scene}.npz"))


if __name__ == "__main__":
    main()
    app.close()
