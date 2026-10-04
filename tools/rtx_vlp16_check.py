"""VLP-16 RTX lidar check with plain Isaac Sim (no Isaac Lab) -- meant for the official Isaac Sim container.

On the host (Ubuntu 26.04) every RTX lidar returns empty data, including NVIDIA's own example profile, so the RTX
profile is validated inside nvcr.io/nvidia/isaac-sim (supported OS):

    docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -v ~/Desktop/3DOG/sim:/sim \\
        --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 /sim/tools/rtx_vlp16_check.py [--scene cubes]

Scenes: "cubes" (four 2 m cubes 5 m away) or "warehouse" (assets/warehouse full_warehouse.usd).
Writes logs/rtx_vlp16_<scene>.npz with the scan and, for the warehouse, the ideal ray-cast range per beam.
"""

import argparse
import os

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--scene", choices=["cubes", "warehouse"], default="warehouse")
ap.add_argument("--profile", default=None, help="OmniLidar USD; default assets/vlp16/rtx/VLP16_rtx.usda, 'default' = schema defaults")
ap.add_argument("--pos", type=float, nargs=3, default=(-5.0, 0.0, 0.434))
ap.add_argument("--frames", type=int, default=90)
args = ap.parse_args()
app = SimulationApp({"headless": True})

import glob  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

carb.settings.get_settings().set("/renderer/raytracingMotion/enabled", True)
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True)
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data  # noqa: E402

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.DefinePrim("/World", "Xform")
if args.scene == "cubes":
    for i, (x, y) in enumerate([(5, 0), (-5, 0), (0, 5), (0, -5)]):
        c = UsdGeom.Cube.Define(stage, f"/World/cube{i}")
        c.CreateSizeAttr(2.0)
        UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(x, y, 0))
    pos = (0.0, 0.0, 0.5)
else:
    wh = glob.glob(os.path.join(SIM, "assets", "warehouse", "**", "full_warehouse.usd"), recursive=True)[0]
    stage.DefinePrim("/World/Warehouse", "Xform").GetReferences().AddReference(wh)
    pos = tuple(args.pos)

attrs = {"omni:sensor:Core:outputFrameOfReference": "SENSOR"}
if args.profile == "default":
    lidar = Lidar("/World/lidar", attributes=attrs)
else:
    profile = args.profile or os.path.join(SIM, "assets", "vlp16", "rtx", "VLP16_rtx.usda")
    lidar = Lidar.create(path="/World/lidar", usd_path=profile, attributes=attrs)
UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/lidar")).SetTranslate(Gf.Vec3d(*pos))
sensor = LidarSensor(lidar, annotators=["generic-model-output"])
for _ in range(5):
    app.update()
omni.timeline.get_timeline_interface().play()

best = None
timeline = omni.timeline.get_timeline_interface()
first = None
for i in range(args.frames):
    app.update()
    data, _ = sensor.get_data("generic-model-output")
    if i % 100 == 0:
        print(f"[check]   frame {i} timeline t={timeline.get_current_time():.2f}s data "
              f"{None if data is None else getattr(data, 'shape', None)}", flush=True)
    if data is None or getattr(data, "shape", (0,))[0] == 0:
        continue
    if first is None:
        first = i
        print(f"[check]   first data at frame {i}", flush=True)
    gmo = parse_generic_model_output_data(data)
    if best is None or gmo.numElements > best.numElements:
        best = gmo
print(f"[check] scene {args.scene}, profile {args.profile or 'VLP16'}: max elements "
      f"{0 if best is None else best.numElements}", flush=True)
if best is None or best.numElements == 0:
    app.close()
    raise SystemExit(1)

x, y, z = (np.asarray(best.x, np.float64), np.asarray(best.y, np.float64), np.asarray(best.z, np.float64))
if str(best.elementsCoordsType).endswith("SPHERICAL"):  # (azimuth deg, elevation deg, range m)
    az, el, r = np.radians(x), np.radians(y), z
    x, y, z = r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)
pts = np.stack([x, y, z], 1)
rng = np.linalg.norm(pts, axis=1)
elev = np.degrees(np.arcsin(np.clip(pts[:, 2] / np.maximum(rng, 1e-6), -1, 1)))
ch, inten = np.asarray(best.channelId), np.asarray(best.scalar)
print(f"[check] coords {best.elementsCoordsType}, frame {best.frameOfReference}, range {rng.min():.2f}..{rng.max():.2f} m, "
      f"intensity {inten.min():.3f}..{inten.max():.3f}", flush=True)
for c in np.unique(ch):
    m = ch == c
    print(f"[check]   channel {int(c):2d}: {m.sum():5d} pts, elevation {np.median(elev[m]):6.2f} deg", flush=True)

out = {"points": pts.astype(np.float32), "channel": ch, "intensity": inten, "pos": np.asarray(pos)}
if args.scene == "warehouse":
    import warp as wp

    src = Usd.Stage.Open(os.path.join(SIM, "assets", "warehouse", "warehouse_raycast.usdc"))
    mesh = UsdGeom.Mesh(src.GetPrimAtPath("/RaycastMesh/mesh"))
    wp.init()
    wmesh = wp.Mesh(points=wp.array(np.asarray(mesh.GetPointsAttr().Get(), np.float32), dtype=wp.vec3),
                    indices=wp.array(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32), dtype=wp.int32))

    @wp.kernel
    def cast(mesh: wp.uint64, o: wp.vec3, d: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        q = wp.mesh_query_ray(mesh, o, d[i], 200.0)
        out[i] = wp.where(q.result, q.t, -1.0)

    dirs = (pts / np.maximum(rng[:, None], 1e-6)).astype(np.float32)
    res = wp.zeros(len(dirs), dtype=wp.float32)
    wp.launch(cast, dim=len(dirs), inputs=[wmesh.id, wp.vec3(*pos), wp.array(dirs, dtype=wp.vec3), res])
    ideal = res.numpy()
    ok = ideal > 0
    err = rng[ok] - ideal[ok]
    print(f"[check] vs ideal ray cast ({ok.sum()} beams): mean {100 * err.mean():+.1f} cm, std {100 * err.std():.1f} cm, "
          f"median |err| {100 * np.median(np.abs(err)):.1f} cm, |err| > 0.5 m {np.mean(np.abs(err) > 0.5):.2%}", flush=True)
    out["ideal_range"] = ideal
os.makedirs(os.path.join(SIM, "logs"), exist_ok=True)
np.savez_compressed(os.path.join(SIM, "logs", f"rtx_vlp16_{args.scene}.npz"), **out)
omni.timeline.get_timeline_interface().stop()
app.close()
