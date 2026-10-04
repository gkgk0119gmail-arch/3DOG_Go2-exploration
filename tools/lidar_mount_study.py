"""How much of the warehouse surface can a VLP-16 on a Go2 see, and how does tilting the mount change it?

    python tools/lidar_mount_study.py logs/lidar_scans/<run> [--tilts 0 10 20 30 45] [--out logs/mount_study]

For every mount variant the VLP-16 pattern (16 ch, +-15 deg, 0.2 deg) is ray-cast on the GPU (warp) against the baked
warehouse mesh and scored with the same 0.1 m voxel GT as tools/map3d.py:

  replay  : the poses of a recorded exploration run (same path, only the mount changes)
            -> 3D coverage per height band + near-obstacle hits (0.15-1.2 m band within 4 m: what the 2D mapper needs),
               all around / ahead (+-60 deg) / ahead and low (< 0.45 m: pallets, sills)
  bound   : full scans from every free spot of the warehouse (1 m lattice, 0.4 m clearance, 8 headings)
            -> the most any policy could ever cover with that mount ("observable" surface, N_obs of the reward)

Variants: "pitch<deg>" tilts the sensor nose-up about its y axis (front beams look up, rear beams look down),
"vertical" rolls it 90 deg (the scan plane sweeps floor-ceiling as the robot moves).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import warp as wp
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map3d import BANDS, OFFS, gt_surface, keys_of  # noqa: E402

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH = os.path.join(SIM, "assets", "warehouse", "warehouse_raycast.usdc")
GT_MAP = os.path.join(SIM, "assets", "maps", "full_warehouse.npz")
BOUNDS = (-26.3, -23.6, 5.3, 30.4)


@wp.kernel
def cast(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), quats: wp.array(dtype=wp.quat),
         local: wp.array(dtype=wp.vec3), n_dirs: int, max_d: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    p = tid // n_dirs
    o = origins[p]
    d = wp.quat_rotate(quats[p], local[tid % n_dirs])
    q = wp.mesh_query_ray(mesh, o, d, max_d)
    if q.result:
        out[tid] = o + q.t * d
    else:
        out[tid] = wp.vec3(1.0e9, 1.0e9, 1.0e9)


def vlp16_dirs(h_res=0.2):
    el = np.deg2rad(np.linspace(-15, 15, 16))
    az = np.deg2rad(np.arange(0, 360, h_res))
    E, A = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)


def mount_rot(name):
    if name == "vertical":
        return R.from_euler("x", 90, degrees=True)
    return R.from_euler("y", -float(name.replace("pitch", "")), degrees=True)  # -pitch about y = nose up


class Caster:
    def __init__(self, device="cuda:0"):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(MESH)
        m = UsdGeom.Mesh(stage.GetPrimAtPath("/RaycastMesh/mesh"))
        pts = np.asarray(m.GetPointsAttr().Get(), np.float32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int32)
        self.device = device
        self.mesh = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=device),
                            indices=wp.array(idx, dtype=wp.int32, device=device))

    def hits(self, origins, quats_xyzw, local, chunk_rays=8_000_000):
        """Yield (pose index offset, hit points [n_poses, n_dirs, 3] torch, valid mask) chunk by chunk."""
        n_dirs = len(local)
        loc = wp.array(local.astype(np.float32), dtype=wp.vec3, device=self.device)
        step = max(1, chunk_rays // n_dirs)
        for s in range(0, len(origins), step):
            o = wp.array(origins[s:s + step].astype(np.float32), dtype=wp.vec3, device=self.device)
            q = wp.array(quats_xyzw[s:s + step].astype(np.float32), dtype=wp.quat, device=self.device)
            n = len(origins[s:s + step])
            out = wp.empty(n * n_dirs, dtype=wp.vec3, device=self.device)
            wp.launch(cast, dim=n * n_dirs, inputs=[self.mesh.id, o, q, loc, n_dirs, 100.0, out], device=self.device)
            p = wp.to_torch(out).view(n, n_dirs, 3)
            yield s, p, p[..., 0] < 1.0e8


class Scorer:
    def __init__(self, v=0.1):
        self.v = v
        b = BOUNDS
        self.origin = np.array([b[0] - 1, b[1] - 1, -1.0])
        gt = gt_surface(MESH, b, v / 2)
        ijk = np.floor((gt - self.origin) / v).astype(np.int64)
        self.gt_keys, first = np.unique(keys_of(ijk), return_index=True)
        self.gt_ijk = ijk[first]
        self.gt_z = self.origin[2] + (self.gt_ijk[:, 2] + 0.5) * v

    def keys(self, p):
        """torch hit points [N, 3] (already filtered) -> unique voxel keys (inside the warehouse box)."""
        b = BOUNDS
        p = p[(p[:, 0] >= b[0] - 0.2) & (p[:, 0] <= b[2] + 0.2) & (p[:, 1] >= b[1] - 0.2) & (p[:, 1] <= b[3] + 0.2)]
        o = torch.as_tensor(self.origin, dtype=p.dtype, device=p.device)
        ijk = torch.floor((p - o) / self.v).long()
        return torch.unique((ijk[:, 0] << 42) + (ijk[:, 1] << 21) + ijk[:, 2])

    def covered(self, scanned):
        scanned = np.sort(scanned)
        cov = np.zeros(len(self.gt_keys), bool)
        for off in OFFS:
            cov |= np.isin(keys_of(self.gt_ijk + off), scanned)
        return cov

    def report(self, cov, mask=None):
        m = np.ones_like(cov) if mask is None else mask
        out = {"all": float(cov[m].sum() / m.sum()),
               "below_8.5m": float(cov[m & (self.gt_z < 8.5)].sum() / (m & (self.gt_z < 8.5)).sum())}
        for lo, hi, name in BANDS:
            bm = m & (self.gt_z >= lo) & (self.gt_z < hi)
            out[name] = float(cov[bm].mean()) if bm.any() else None
        return out


def load_run(scans):
    files = sorted(glob.glob(os.path.join(scans, "*.npz")))
    pos = np.array([np.load(f)["pos_w"] for f in files], np.float64)
    quat = np.array([np.load(f)["quat_w"] for f in files], np.float64)
    return pos, quat


def free_lattice(z, step=1.0, clearance=0.4):
    d = np.load(GT_MAP)
    g, go, res = d["grid"], d["origin"], float(d["resolution"])  # indexed [x, y]; 0 occ, 255 free
    dist = distance_transform_edt(g != 0) * res
    b = BOUNDS
    xs = np.arange(b[0] + 0.5, b[2], step)
    ys = np.arange(b[1] + 0.5, b[3], step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ix = ((X - go[0]) / res).astype(int)
    iy = ((Y - go[1]) / res).astype(int)
    ok = (g[ix, iy] == 255) & (dist[ix, iy] >= clearance)
    return np.stack([X[ok], Y[ok], np.full(ok.sum(), z)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--tilts", type=float, nargs="*", default=[0, 10, 20, 30, 45])
    ap.add_argument("--vertical", action="store_true", default=True)
    ap.add_argument("--headings", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(SIM, "logs", "mount_study"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wp.init()
    caster, scorer = Caster(), Scorer()
    local0 = vlp16_dirs()
    pos, quat = load_run(args.scans)
    lattice = free_lattice(z=float(pos[:, 2].mean()))
    print(f"GT surface voxels {len(scorer.gt_keys):,} | run poses {len(pos)} | free lattice spots {len(lattice)}")
    variants = [f"pitch{int(t)}" for t in args.tilts] + (["vertical"] if args.vertical else [])
    results = {}
    for name in variants:
        local = mount_rot(name).apply(local0)
        # --- replay of the recorded path ------------------------------------------------------------
        keys, near, front, front_low = [], [], [], []
        yaw_run = R.from_quat(quat).as_euler("xyz")[:, 2]
        for s, p, ok in caster.hits(pos, quat, local):
            keys.append(scorer.keys(p[ok]))
            # near-obstacle evidence: distinct 0.2 m cells hit in the 0.15-1.2 m band within 4 m, per scan
            o = torch.as_tensor(pos[s:s + len(p)], dtype=p.dtype, device=p.device)[:, None, :]
            r = torch.linalg.norm(p[..., :2] - o[..., :2], dim=-1)
            band = ok & (p[..., 2] > 0.15) & (p[..., 2] < 1.2) & (r < 4.0)
            cell = torch.floor(p[..., :2] / 0.2).long()
            ck = (torch.arange(len(p), device=p.device)[:, None] << 40) + ((cell[..., 0] + 5000) << 20) + (cell[..., 1] + 5000)
            u = torch.unique(ck[band])
            near.append(torch.bincount(u >> 40, minlength=len(p)).float().cpu().numpy())
            # same, only ahead of the robot (+-60 deg of its heading): what it walks into
            yw = torch.as_tensor(yaw_run[s:s + len(p)], dtype=p.dtype, device=p.device)[:, None]
            bear = torch.atan2(p[..., 1] - o[..., 1], p[..., 0] - o[..., 0]) - yw
            ahead = torch.cos(bear) > 0.5
            for lst, m in ((front, band & ahead), (front_low, band & ahead & (p[..., 2] < 0.45))):
                lst.append(torch.bincount(torch.unique(ck[m]) >> 40, minlength=len(p)).float().cpu().numpy())
        cov = scorer.covered(torch.unique(torch.cat(keys)).cpu().numpy())
        rep = scorer.report(cov)
        rep["near_cells_per_scan"] = float(np.concatenate(near).mean())
        rep["front_cells_per_scan"] = float(np.concatenate(front).mean())
        rep["front_low_cells_per_scan"] = float(np.concatenate(front_low).mean())
        # --- upper bound from everywhere ------------------------------------------------------------
        heads = 1 if name == "pitch0" else args.headings
        yaw = np.repeat(np.arange(heads) * 2 * np.pi / heads, len(lattice))
        org = np.tile(lattice, (heads, 1))
        q = R.from_euler("z", yaw[:, None]).as_quat()  # xyzw, level body
        bkeys = [scorer.keys(p[ok]) for _, p, ok in caster.hits(org, q, local)]
        bcov = scorer.covered(torch.unique(torch.cat(bkeys)).cpu().numpy())
        bound = scorer.report(bcov)
        rep_obs = scorer.report(cov, mask=bcov)  # replay coverage of what is observable at all with this mount
        results[name] = {"replay": rep, "bound": bound, "replay_of_observable": rep_obs["all"],
                         "observable_voxels": int(bcov.sum())}
        print(f"{name:9s} replay {rep['all']:6.1%} (<8.5 m {rep['below_8.5m']:6.1%}, ceiling {rep['ceiling >8.5 m']:6.1%}) "
              f"| bound {bound['all']:6.1%} (ceiling {bound['ceiling >8.5 m']:6.1%}) | replay/observable "
              f"{rep_obs['all']:6.1%} | near cells/scan all {rep['near_cells_per_scan']:5.1f} front {rep['front_cells_per_scan']:5.1f} "
              f"front<0.45m {rep['front_low_cells_per_scan']:5.1f}", flush=True)
    with open(os.path.join(args.out, "mount_study.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}/mount_study.json")


if __name__ == "__main__":
    main()
