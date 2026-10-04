"""Validate the VLP-16 RTX profile: one static sensor in the warehouse vs. ideal ray casting.

Usage (rendering required for RTX sensors):
    python tools/test_vlp16_rtx.py [--pos -5 0 0.434] [--frames 60]

Prints channel elevations, points per scan, intensity range, and range error against a warp ray cast of the baked
warehouse mesh along the same beam directions; saves the scan to logs/rtx_vlp16_scan.npz.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pos", type=float, nargs=3, default=(-5.0, 0.0, 0.434))
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--profile", type=str, default=None, help="OmniLidar USD (default: assets/vlp16/rtx/VLP16_rtx.usda)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # RTX sensors need the rendering pipeline (headless or not)
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402

carb.settings.get_settings().set("/renderer/raytracingMotion/enabled", True)  # motion effects for RTX lidar
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM)
from go2_lab.factory import WAREHOUSE_RAYCAST_USD, WAREHOUSE_USD  # noqa: E402

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.DefinePrim("/World", "Xform")
stage.DefinePrim("/World/Warehouse", "Xform").GetReferences().AddReference(WAREHOUSE_USD)
lidar_prim = stage.DefinePrim("/World/VLP16")  # typeless: inherits OmniLidar from the reference
lidar_prim.GetReferences().AddReference(args.profile or os.path.join(SIM, "assets", "vlp16", "rtx", "VLP16_rtx.usda"))
UsdGeom.XformCommonAPI(lidar_prim).SetTranslate(Gf.Vec3d(*args.pos))
for _ in range(10):
    app.update()

import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True)
from isaacsim.sensors.experimental.rtx import LidarSensor, parse_generic_model_output_data  # noqa: E402

sensor = LidarSensor("/World/VLP16", annotators=["generic-model-output"])
omni.timeline.get_timeline_interface().play()
scans = []
for i in range(args.frames):
    app.update()
    data, _ = sensor.get_data("generic-model-output")
    if data is None:
        if i % 10 == 0:
            print(f"[rtx]   frame {i}: no data", flush=True)
        continue
    gmo = parse_generic_model_output_data(data)
    if i % 10 == 0:
        print(f"[rtx]   frame {i}: bytes {getattr(data, 'shape', None)} elements {gmo.numElements} "
              f"scanComplete {gmo.scanComplete}", flush=True)
    if gmo.numElements > 0:
        scans.append(gmo)
print(f"[rtx] frames {args.frames}, frames with points {len(scans)}")
gmo = max(scans, key=lambda g: g.numElements)
x, y, z = (np.asarray(gmo.x, np.float64), np.asarray(gmo.y, np.float64), np.asarray(gmo.z, np.float64))
ch, inten = np.asarray(gmo.channelId), np.asarray(gmo.scalar)
print(f"[rtx] coords {gmo.elementsCoordsType}, frame {gmo.frameOfReference}, points {gmo.numElements}, "
      f"scanComplete {gmo.scanComplete}")
if str(gmo.elementsCoordsType).endswith("SPHERICAL"):
    az, el, r = np.radians(x), np.radians(y), z  # (azimuth deg, elevation deg, range m)
    x, y, z = r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)
pts = np.stack([x, y, z], 1)
if str(gmo.frameOfReference).endswith("WORLD"):
    pts_s = pts - np.asarray(args.pos)
else:
    pts_s = pts
rng = np.linalg.norm(pts_s, axis=1)
elev = np.degrees(np.arcsin(pts_s[:, 2] / np.maximum(rng, 1e-6)))
print(f"[rtx] range {rng.min():.2f}..{rng.max():.2f} m, intensity {inten.min():.3f}..{inten.max():.3f}")
for c in np.unique(ch):
    m = ch == c
    print(f"[rtx]   channel {int(c):2d}: {m.sum():5d} pts, elevation {np.median(elev[m]):6.2f} deg")

# ideal ray cast along the same directions against the baked warehouse mesh
src = Usd.Stage.Open(WAREHOUSE_RAYCAST_USD)
mesh = UsdGeom.Mesh(src.GetPrimAtPath("/RaycastMesh/mesh"))
wp.init()
wmesh = wp.Mesh(points=wp.array(np.asarray(mesh.GetPointsAttr().Get(), np.float32), dtype=wp.vec3),
                indices=wp.array(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32), dtype=wp.int32))


@wp.kernel
def cast(mesh: wp.uint64, o: wp.vec3, d: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, o, d[i], 200.0)
    out[i] = wp.where(q.result, q.t, -1.0)


dirs = (pts_s / np.maximum(rng[:, None], 1e-6)).astype(np.float32)
out = wp.zeros(len(dirs), dtype=wp.float32)
wp.launch(cast, dim=len(dirs), inputs=[wmesh.id, wp.vec3(*args.pos), wp.array(dirs, dtype=wp.vec3), out])
ideal = out.numpy()
ok = ideal > 0
err = rng[ok] - ideal[ok]
print(f"[rtx] vs ideal ray cast ({ok.sum()} beams): error mean {100 * err.mean():+.1f} cm, std {100 * err.std():.1f} cm, "
      f"|err|>0.5 m {np.mean(np.abs(err) > 0.5):.2%}")
np.savez_compressed(os.path.join(SIM, "logs", "rtx_vlp16_scan.npz"), points=pts_s.astype(np.float32), channel=ch,
                    intensity=inten, ideal_range=ideal, pos=np.asarray(args.pos))
omni.timeline.get_timeline_interface().stop()
app.close()
