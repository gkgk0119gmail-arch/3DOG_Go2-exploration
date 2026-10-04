"""Per-robot planning worker: one ARiADNE planner (+ A*) in its own Python process.

Each robot of a parallel exploration run owns one worker, so robots never share planner state and the
CPU-heavy graph updates run on separate cores instead of blocking the simulation loop.

Protocol: length-prefixed pickles over the worker's stdin/stdout.
    ("reset", params: dict)                                  -> ("reset", None)
    ("plan", req_id, grid, origin, pos_xy[, extra])          -> ("plan", req_id, waypoint|None, done, graph, seconds)
params["kind"] picks the planner: "ros" (AriadnePlanner, default) or "dense" (DensePlanner: the ariadne3d training
graph + checkpoint). One kind per process (the two code bases share module names). ``extra`` = dict(heading,
util_grid, seen_area) for planners with 3D inputs.
    ("path", req_id, grid, origin, cell, start, goal, kw)    -> ("path", req_id, path|None, seconds)
"""

from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time

_HEADER = struct.Struct("<Q")


def _send(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)) + data)
    stream.flush()


def _recv(stream):
    head = stream.read(_HEADER.size)
    if len(head) < _HEADER.size:
        return None
    (n,) = _HEADER.unpack(head)
    return pickle.loads(stream.read(n))


# --------------------------------------------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------------------------------------------


def _serve():
    # keep the protocol stream clean: anything printed by the planner goes to stderr
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    proto_in = os.fdopen(os.dup(0), "rb")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from go2_lab.exploration.astar import plan_path

    import traceback

    planner_cls = None

    def make(params):
        nonlocal planner_cls
        if planner_cls is None:
            if params.get("kind", "ros") == "dense":
                from go2_lab.exploration.dense_planner import DensePlanner as planner_cls
            else:
                from go2_lab.exploration.ariadne_planner import AriadnePlanner as planner_cls
        return planner_cls(**{k: v for k, v in params.items() if k != "kind"})

    planner, params, failures = None, {}, 0
    while (msg := _recv(proto_in)) is not None:
        kind = msg[0]
        if kind == "reset":
            params = msg[1] or {}
            planner = make(params)
            _send(proto_out, ("reset", None))
        elif kind == "plan":
            _, req_id, grid, origin, pos = msg[:5]
            extra = msg[5] if len(msg) > 5 else None
            t0 = time.time()
            try:
                planner.update_map(grid, origin)
                planner.update_location(pos)
                if extra is not None and hasattr(planner, "update_3d"):
                    planner.update_3d(**extra)
                wp = planner.plan()
                wp = None if wp is None else np.asarray(wp, dtype=np.float64)
                reply = ("plan", req_id, wp, planner.done, planner.graph(), time.time() - t0)
            except Exception:  # noqa: BLE001 -- rebuild the graph from the current pose and carry on
                failures += 1
                print(f"[planner_worker] plan failed ({failures}), rebuilding planner:\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                planner = make(params)
                reply = ("plan", req_id, None, False, (np.zeros((0, 2)), np.zeros(0), [], np.zeros((0, 2))),
                         time.time() - t0)
            _send(proto_out, reply)
        elif kind == "path":
            _, req_id, grid, origin, cell, start, goal, kw = msg
            t0 = time.time()
            path, _ = plan_path(grid, origin, cell, start, goal, **kw)
            _send(proto_out, ("path", req_id, path, time.time() - t0))


# --------------------------------------------------------------------------------------------------------------
# simulator side
# --------------------------------------------------------------------------------------------------------------


class PlannerWorker:
    """Handle to one worker process; requests are asynchronous, replies are collected with :meth:`poll`."""

    def __init__(self, params: dict | None = None):
        self._replies: queue.Queue = queue.Queue()
        self._params = params or {}
        self.busy = 0
        self._start()
        self.reset(params)

    def _start(self):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
        threading.Thread(target=self._read_loop, args=(self.proc,), daemon=True).start()

    def _read_loop(self, proc):
        while (msg := _recv(proc.stdout)) is not None:
            self._replies.put(msg)

    def _submit(self, msg):
        self.busy += 1
        try:
            _send(self.proc.stdin, msg)
        except (BrokenPipeError, OSError):
            # worker died: start a fresh one with the last parameters and resend
            print("[planner_worker] worker process died, restarting", file=sys.stderr, flush=True)
            self._start()
            _send(self.proc.stdin, ("reset", self._params))
            self.busy += 1
            if msg[0] != "reset":
                _send(self.proc.stdin, msg)

    def reset(self, params: dict | None = None, wait: bool = True):
        """Fresh ARiADNE planner (optionally with parameter overrides). Drops pending replies."""
        self._params = params or {}
        self._submit(("reset", self._params))
        if wait:
            while True:
                msg = self._replies.get(timeout=120)
                self.busy -= 1
                if msg[0] == "reset":
                    return

    def plan(self, req_id, grid, origin, pos_xy, extra: dict | None = None):
        self._submit(("plan", req_id, grid, origin, pos_xy) + ((extra,) if extra is not None else ()))

    def path(self, req_id, grid, origin, cell, start, goal, **kw):
        self._submit(("path", req_id, grid, origin, cell, start, goal, kw))

    def poll(self) -> list:
        out = []
        while True:
            try:
                msg = self._replies.get_nowait()
            except queue.Empty:
                return out
            self.busy -= 1
            out.append(msg)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


if __name__ == "__main__":
    _serve()
