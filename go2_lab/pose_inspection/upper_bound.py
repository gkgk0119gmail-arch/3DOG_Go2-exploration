"""Observable upper bound of a scene for one sensor configuration (our addition to the POSE metrics).

Surface: every reachable robot position on a lattice x every yaw x every body posture of the grid, one full VLP-16
sweep each; a GT surface voxel of object o is observable when some return labelled o lands in it within the object
range. Reported for the full posture grid and for the level posture only (what planar planning can reach at best).

Volume: the same sweep on a coarser lattice marks free / occupied voxels; the union is the observable volume that
Cs is normalised by (POSE: V minus V_res).
"""

from __future__ import annotations

import itertools
import math
import os

import numpy as np
import torch

from .metrics import SurfaceCoverage
from .navgrid import NavGrid
from .raycast import SceneMesh, VoxelGrid, mount, rot_ypr, sensor_poses, vlp16_dirs


def posture_grid(ub_cfg, robot_cfg):
    """(pitch, roll) [rad] of the grid, clipped to the policy limits; the level posture first."""
    out = [(0.0, 0.0)]
    for p, r in itertools.product(ub_cfg.pitch_deg, ub_cfg.roll_deg):
        if abs(p) <= robot_cfg.pitch_limit_deg + 1e-6 and abs(r) <= robot_cfg.roll_limit_deg + 1e-6 and (p, r) != (0, 0):
            out.append((math.radians(p), math.radians(r)))
    return out


def _poses(xy, yaws, postures, base_height):
    P = [(x, y, yw, p, r) for (x, y) in xy for yw in yaws for (p, r) in postures]
    a = torch.tensor(P, dtype=torch.float32)
    base = torch.stack([a[:, 0], a[:, 1], torch.full_like(a[:, 0], base_height)], -1)
    return base, rot_ypr(a[:, 2], a[:, 3], a[:, 4]), a


def compute(cfg, scene_meta: dict, tilt_deg: float, device: str = "cuda:0", log=print) -> dict:
    d = scene_meta["dir"]
    ub, rb, li = cfg.upper_bound, cfg.robot, cfg.lidar
    nav = NavGrid(os.path.join(d, "grid.npz"), rb.radius).restrict_to_region(scene_meta["robot_start"]["region"])
    mesh = SceneMesh(os.path.join(d, "gt.npz"), device)
    targets = [o["target"] for o in scene_meta["objects"]]
    surf_all = SurfaceCoverage(os.path.join(d, "surface.npz"), targets, 1, device)
    surf_lvl = SurfaceCoverage(os.path.join(d, "surface.npz"), targets, 1, device)
    m_pos, m_R = mount(li, tilt_deg)
    yaws = [math.radians(y) for y in ub.yaws_deg]
    postures = posture_grid(ub, rb)

    # --- surface -----------------------------------------------------------------------------------------------
    xy = nav.lattice(ub.grid_step)
    dirs = torch.as_tensor(vlp16_dirs(li.channels, li.vfov_deg, ub.az_res_deg), dtype=torch.float32, device=device)
    base, Rb, raw = _poses(xy, yaws, postures, rb.base_height)
    level = (raw[:, 3].abs() < 1e-9) & (raw[:, 4].abs() < 1e-9)
    log(f"    surface sweep: {len(xy)} positions x {len(yaws)} yaws x {len(postures)} postures = {len(base):,} poses")
    for s in range(0, len(base), ub.batch_poses):
        sl = slice(s, s + ub.batch_poses)
        o, R = sensor_poses(base[sl].to(device), Rb[sl].to(device), m_pos, m_R)
        t, _, obj = mesh.cast(o, R, dirs, li.max_range)
        hit = (obj >= 0) & (t <= cfg.metrics.object_range)
        pts = o[:, None, :] + (R @ dirs.T).transpose(1, 2) * t[..., None]
        env0 = torch.zeros(int(hit.sum()), dtype=torch.long, device=device)
        surf_all.mark(env0, pts[hit], obj[hit])
        lv = level[sl].to(device)[:, None] & hit
        surf_lvl.mark(torch.zeros(int(lv.sum()), dtype=torch.long, device=device), pts[lv], obj[lv])

    # --- volume ------------------------------------------------------------------------------------------------
    vg = VoxelGrid(scene_meta["bounds"], scene_meta["z_range"], cfg.metrics.scene_voxel, 1, device)
    xyv = nav.lattice(ub.volume_grid_step)
    dirs_v = torch.as_tensor(vlp16_dirs(li.channels, li.vfov_deg, ub.volume_az_res_deg), dtype=torch.float32,
                             device=device)
    base, Rb, _ = _poses(xyv, yaws, postures, rb.base_height)
    log(f"    volume sweep: {len(xyv)} positions -> {len(base):,} poses")
    for s in range(0, len(base), ub.batch_poses):
        sl = slice(s, s + ub.batch_poses)
        o, R = sensor_poses(base[sl].to(device), Rb[sl].to(device), m_pos, m_R)
        t, _, _ = mesh.cast(o, R, dirs_v, li.max_range)
        vg.insert(o, R, dirs_v, t, torch.zeros(len(o), dtype=torch.int32, device=device),
                  cfg.metrics.no_return_free_range)
    observable = (vg.grid[0] > 0).cpu().numpy()
    np.savez_compressed(os.path.join(d, f"observable_tilt{tilt_deg:g}.npz"), observable=observable,
                        lo=vg.lo, voxel=np.float32(vg.voxel))

    # per-voxel observability (same order as SurfaceCoverage.keys) for figures and per-voxel ratios
    np.savez_compressed(os.path.join(d, f"ub_surface_tilt{tilt_deg:g}.npz"), keys=surf_all.keys.cpu().numpy(),
                        obj=surf_all.obj.cpu().numpy(), posture=surf_all.seen[0].cpu().numpy(),
                        level=surf_lvl.seen[0].cpu().numpy())
    per_all = surf_all.per_object()[0].cpu().numpy()
    per_lvl = surf_lvl.per_object()[0].cpu().numpy()
    return {
        "tilt_deg": tilt_deg, "positions": int(len(xy)), "postures": len(postures),
        "reachable_area_m2": float(nav.reachable.sum() * nav.cell**2),
        "co_ub_posture": float(surf_all.co()[0]), "co_ub_level": float(surf_lvl.co()[0]),
        "co_ub_posture_n26": float(surf_all.co(True)[0]), "co_ub_level_n26": float(surf_lvl.co(True)[0]),
        "per_object": {o["name"]: {"posture": float(per_all[o["index"]]), "level": float(per_lvl[o["index"]])}
                       for o in scene_meta["objects"]},
        "observable_volume_fraction": float(observable.mean()),
        "observable_file": f"observable_tilt{tilt_deg:g}.npz",
    }
