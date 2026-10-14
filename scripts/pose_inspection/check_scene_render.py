"""Render each built scene in Isaac Sim and check the semantic labels (stage-1 sanity check).

    python scripts/pose_inspection/check_scene_render.py A_warehouse [--per_scene 2]   (one scene per run)

For a few target objects per scene a pinhole camera is placed ``--standoff`` m from the object at ``--height`` m and
looks at its centre; RGB + semantic segmentation are rendered. Writes results/pose_inspection/stage1_render_<scene>.png
and .json (labels found, pixel fraction of each label in each view).
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--per_scene", type=int, default=2)
ap.add_argument("--standoff", type=float, default=5.0)
ap.add_argument("--cam_height", type=float, default=1.6)
ap.add_argument("--warmup", type=int, default=30)
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
from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: E402


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots, report = [], {}
    for name in [args.scene]:
        meta = load_scene(f"assets/pose_inspection/{name}")
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
        scene_cfg = sim_utils.UsdFileCfg(usd_path=os.path.join(meta["dir"], meta["files"]["scene"]))
        scene_cfg.func("/World/Scene", scene_cfg)
        targets = [o for o in meta["objects"] if o["target"]][: args.per_scene]
        for k, ob in enumerate(targets):
            sim_utils.create_prim(f"/World/Cam_{k}", "Xform")
        cfg = CameraCfg(prim_path="/World/Cam_[^/]+/Sensor", update_period=0, height=480, width=640,
                        data_types=["rgb", "semantic_segmentation"],
                        renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False),
                        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955,
                                                         clipping_range=(0.1, 200.0)))
        cam = Camera(cfg)
        sim.reset()
        eyes, looks = [], []
        for ob in targets:
            c = np.array(ob["center"])
            # look from the scene centre side (warehouse: along the aisle)
            b = meta["bounds"]
            mid = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            d = mid - c[:2]
            if name.startswith("A_"):
                d = np.array([0.0, -1.0 if c[1] > 9 else 1.0])  # along the aisle
            d = d / (np.linalg.norm(d) + 1e-9)
            eyes.append([c[0] + d[0] * args.standoff, c[1] + d[1] * args.standoff, args.cam_height])
            looks.append([c[0], c[1], ob["height_m"] / 2])
        cam.set_world_poses_from_view(torch.tensor(eyes, dtype=torch.float32, device=sim.device),
                                      torch.tensor(looks, dtype=torch.float32, device=sim.device))
        for _ in range(args.warmup):
            sim.step()
            cam.update(dt=sim.get_physics_dt())
        rgb = cam.data.output["rgb"].cpu().numpy()
        seg_raw = cam.data.output["semantic_segmentation"].cpu().numpy()
        info = cam.data.info["semantic_segmentation"]
        info = info[0] if isinstance(info, list) else info
        raw = info.get("idToLabels", {}) if isinstance(info, dict) else {}
        # keys are ids, or "(r, g, b, a)" colours when the renderer colourises the segmentation
        keys = list(raw)
        colour_keys = bool(keys) and str(keys[0]).startswith("(")
        lab = {k: (v.get("class", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw.items()}
        if colour_keys:
            # single channel holding the RGBA colour packed little-endian into a uint32
            seg = seg_raw[..., 0].astype(np.int64) & 0xFFFFFFFF
            id2lab = {}
            for k, v in lab.items():
                r, g, b_, a = (int(x) for x in str(k).strip("()").split(","))
                id2lab[r | (g << 8) | (b_ << 16) | (a << 24)] = v
        else:
            seg = seg_raw[..., 0]
            id2lab = {int(k): v for k, v in lab.items()}
        rep = []
        for k, ob in enumerate(targets):
            ids, cnt = np.unique(seg[k], return_counts=True)
            frac = {id2lab.get(int(i), str(int(i))): round(float(c) / seg[k].size, 3) for i, c in zip(ids, cnt)}
            rep.append({"object": ob["name"], "class": ob["class"], "pixel_fraction": frac})
            shots.append((f"{name} / {ob['name']}", rgb[k], seg[k], id2lab))
        report[name] = {"labels": sorted(set(id2lab.values())), "views": rep}
        print(json.dumps(report[name], indent=1))
    out = os.path.join(SIM, "results", "pose_inspection")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"stage1_render_{args.scene}.json"), "w") as f:
        json.dump(report, f, indent=1)
    fig, axes = plt.subplots(len(shots), 2, figsize=(12, 4.6 * len(shots)))
    axes = np.atleast_2d(axes)
    palette = plt.get_cmap("tab10")
    for r, (title, im, sg, id2lab) in enumerate(shots):
        axes[r, 0].imshow(im[..., :3])
        axes[r, 0].set_title(title)
        col = np.zeros((*sg.shape, 3))
        for j, i in enumerate(np.unique(sg)):
            col[sg == i] = palette(j % 10)[:3]
            ys, xs = np.nonzero(sg == i)
            axes[r, 1].text(xs.mean(), ys.mean(), id2lab.get(int(i), str(int(i))), color="w", fontsize=9,
                            ha="center", bbox={"fc": "k", "alpha": 0.5, "pad": 1})
        axes[r, 1].imshow(col)
        axes[r, 1].set_title("semantic segmentation (class labels)")
        for a in axes[r]:
            a.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"stage1_render_{args.scene}.png"), dpi=70)
    print("wrote", os.path.join(out, f"stage1_render_{args.scene}.png"))


if __name__ == "__main__":
    main()
    app.close()
