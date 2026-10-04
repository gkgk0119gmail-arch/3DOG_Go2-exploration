"""Merge fixed-joint child links of a URDF into their parents, combining mass and inertia exactly.

Links named in --keep (and joints marked dont_collapse="true") are preserved, so the Go2 feet stay
as separate bodies for contact sensing while rotor / calflower helper links are folded into their parents.

Usage:
    python merge_fixed_links.py in.urdf out.urdf [--keep FL_foot FR_foot ...]
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def mat_to_rpy(R):
    p = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(p)) > 1e-9:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


def origin_to_tf(elem):
    T = np.eye(4)
    o = elem.find("origin") if elem is not None else None
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = rpy_to_mat(*[float(v) for v in o.get("rpy", "0 0 0").split()])
    return T


def set_origin(elem, T):
    o = elem.find("origin")
    if o is None:
        o = ET.Element("origin")
        elem.insert(0, o)
    o.set("xyz", " ".join(f"{v:.9g}" for v in T[:3, 3]))
    o.set("rpy", " ".join(f"{v:.9g}" for v in mat_to_rpy(T[:3, :3])))


def read_inertial(link):
    """Return (mass, com [3], inertia about com expressed in link frame [3x3]) or None."""
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass = float(inertial.find("mass").get("value"))
    T = origin_to_tf(inertial)
    i = inertial.find("inertia")
    Ii = np.array([
        [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
        [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
        [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
    ])
    R = T[:3, :3]
    return mass, T[:3, 3], R @ Ii @ R.T


def write_inertial(link, mass, com, inertia):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.9g}" for v in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "inertia", **{
        k: f"{inertia[a, b]:.9g}" for k, (a, b) in
        {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2), "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}.items()
    })
    link.insert(0, inertial)


def combine(a, b):
    """Combine two (mass, com, inertia-about-com) tuples expressed in the same frame."""
    if a is None:
        return b
    if b is None:
        return a
    m = a[0] + b[0]
    com = (a[0] * a[1] + b[0] * b[1]) / m
    inertia = np.zeros((3, 3))
    for mi, ci, Ii in (a, b):
        d = ci - com
        inertia += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return m, com, inertia


def merge(robot, keep):
    links = {l.get("name"): l for l in robot.findall("link")}
    while True:
        joint = next(
            (j for j in robot.findall("joint")
             if j.get("type") == "fixed" and j.find("child").get("link") not in keep
             and j.get("dont_collapse") != "true"),
            None,
        )
        if joint is None:
            return
        parent = links[joint.find("parent").get("link")]
        child = links[joint.find("child").get("link")]
        T_pc = origin_to_tf(joint)

        # mass properties, child expressed in parent frame
        ci = read_inertial(child)
        if ci is not None:
            R = T_pc[:3, :3]
            ci = (ci[0], T_pc[:3, :3] @ ci[1] + T_pc[:3, 3], R @ ci[2] @ R.T)
        merged = combine(read_inertial(parent), ci)
        if merged is not None:
            write_inertial(parent, *merged)

        # visuals / collisions move to the parent with composed origins
        for tag in ("visual", "collision"):
            for e in child.findall(tag):
                set_origin(e, T_pc @ origin_to_tf(e))
                parent.append(e)

        # grandchildren re-attach to the parent
        for j in robot.findall("joint"):
            if j.find("parent").get("link") == child.get("name"):
                j.find("parent").set("link", parent.get("name"))
                set_origin(j, T_pc @ origin_to_tf(j))

        robot.remove(joint)
        robot.remove(child)
        del links[child.get("name")]
        print(f"merged {child.get('name')} -> {parent.get('name')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--keep", nargs="*", default=[])
    args = ap.parse_args()
    tree = ET.parse(args.input)
    robot = tree.getroot()
    total_before = sum(float(m.get("value")) for m in robot.iter("mass"))
    merge(robot, set(args.keep))
    total_after = sum(float(m.get("value")) for m in robot.iter("mass"))
    ET.indent(tree)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"links={len(robot.findall('link'))} joints={len(robot.findall('joint'))} "
          f"mass {total_before:.4f} -> {total_after:.4f} kg")


if __name__ == "__main__":
    main()
