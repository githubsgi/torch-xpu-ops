"""GPU-tile and build slots on the compute nodes, shared by every harness process.

A slot is a directory RUNS/slots/<kind>/<node>.<i>; mkdir is atomic on the
shared file system, so any number of processes (the three harnesses, an MCTS
with many workers, a human running ggemm-eval) can share the nodes without a
daemon. A slot older than its stale limit is reclaimed (crashed owner).
"""

import json
import os
import random
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import config

SLOT_ROOT = config.RUNS / "slots"


def _slots(kind):
    n = config.TILES_PER_NODE if kind == "gpu" else config.BUILDS_PER_NODE
    return [(node, i) for node in config.nodes() for i in range(n)]


def _path(kind, node, i):
    return SLOT_ROOT / kind / f"{config.short(node)}.{i}"


def _try(kind, node, i, label, stale_s):
    p = _path(kind, node, i)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.mkdir()
    except FileExistsError:
        try:
            if time.time() - p.stat().st_mtime > stale_s:
                shutil.rmtree(p, ignore_errors=True)
        except FileNotFoundError:
            pass
        return False
    (p / "owner.json").write_text(json.dumps(
        {"host": socket.gethostname(), "pid": os.getpid(), "thread": threading.get_ident(),
         "label": label, "since": time.time()}))
    return True


@contextmanager
def acquire(kind, label="", prefer_node=None, stale_s=None, poll_s=2.0):
    """Block until a slot is free; yields (node, index)."""
    stale_s = stale_s or (config.RUN_TIMEOUT_S if kind == "gpu" else config.BUILD_TIMEOUT_S) + 600
    slots = _slots(kind)
    if not slots:
        raise RuntimeError("no compute nodes: set GGEMM_NODES or GGEMM_NODEFILE")
    while True:
        order = slots[:]
        random.shuffle(order)
        if prefer_node:
            order.sort(key=lambda s: s[0] != prefer_node)
        for node, i in order:
            if _try(kind, node, i, label, stale_s):
                try:
                    yield node, i
                finally:
                    shutil.rmtree(_path(kind, node, i), ignore_errors=True)
                return
        time.sleep(poll_s + random.random())


def status():
    rows = []
    for kind in ("gpu", "build"):
        busy = []
        for node, i in _slots(kind):
            p = _path(kind, node, i)
            if not p.exists():
                continue
            try:
                o = json.loads((p / "owner.json").read_text())
                busy.append(f"  {config.short(node)}.{i:<2d} {o.get('label', '')} "
                            f"({time.time() - o.get('since', time.time()):.0f} s)")
            except (FileNotFoundError, json.JSONDecodeError):
                busy.append(f"  {config.short(node)}.{i:<2d} (starting)")
        rows.append(f"{kind}: {len(busy)}/{len(_slots(kind))} busy")
        rows.extend(busy)
    return "\n".join(rows)


def clear():
    shutil.rmtree(SLOT_ROOT, ignore_errors=True)
