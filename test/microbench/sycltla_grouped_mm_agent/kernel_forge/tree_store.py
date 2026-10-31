"""Persistent revision tree (Kernel Forge: one tree per operator variant).

A node is an admitted candidate (compiles, passes verification, was measured):
header + configs on disk in nodes/<id>/, metadata in tree.json. Failed
attempts (build / runtime / verify failures, duplicates, no-ops) never become
nodes; they are kept in `failures` so that later prompts can summarize them.
Valid-but-slower candidates stay in the tree.

Selection quantity is the normalized latency L = 1 / score (score = geomean
speedup vs production on the search suite), so production has L = 1 and a
candidate that is 5% faster has L = 0.952.

All methods that touch shared state take `tree.lock` (re-entrant).
"""

import hashlib
import json
import math
import os
import random
import threading
import time
from pathlib import Path


def content_hash(header_text, cfgs):
    """Dedupe key: header text + normalized configs (the evaluator's build key)."""
    return hashlib.sha256(header_text.encode() + b"\0" +
                          json.dumps(cfgs, sort_keys=True).encode()).hexdigest()[:16]


class Node:
    PERSIST = ("id", "parent", "children", "depth", "visits", "action", "notes", "auto_change", "chash",
               "score", "eval", "created", "cost_usd", "repairs", "seconds", "profile", "full_eval")

    def __init__(self, **kw):
        self.id = kw["id"]
        self.parent = kw.get("parent")
        self.children = list(kw.get("children", []))
        self.depth = kw.get("depth", 0)
        self.visits = kw.get("visits", 0)
        self.action = kw.get("action", "root")        # root | llm | knob
        self.notes = kw.get("notes", "")              # LLM summary or mutation description
        self.auto_change = kw.get("auto_change", "")  # config / header diff vs parent, computed
        self.chash = kw.get("chash", "")
        self.score = kw.get("score", 1.0)             # own score on the search suite
        self.eval = kw.get("eval", {})                # status, per_op, per_case, seconds
        self.created = kw.get("created", time.time())
        self.cost_usd = kw.get("cost_usd", 0.0)
        self.repairs = kw.get("repairs", 0)
        self.seconds = kw.get("seconds", 0.0)
        self.profile = kw.get("profile")              # cached profile digest of its weakest op
        self.full_eval = kw.get("full_eval")          # final re-evaluation, if any
        # runtime only (not persisted): in-flight bookkeeping
        self.virtual = 0     # in-flight expansions passing through this node (virtual visits)
        self.pending = 0     # in-flight expansions whose new child will hang directly under it

    @property
    def L(self):
        return 1.0 / self.score if self.score > 0 else float("inf")

    def to_json(self):
        return {k: getattr(self, k) for k in self.PERSIST}


class Tree:
    def __init__(self, workdir):
        self.wd = Path(workdir)
        self.lock = threading.RLock()
        self.nodes = {}
        self.failures = []          # dicts: id, parent, action, status, notes, error, chash, cost_usd
        self.hashes = {}            # chash -> node id | "failed:<id>" | "inflight:<id>"
        self.root = None
        self.attempts = 0           # attempt counter (= iterations launched)
        self.iterations_done = 0
        self.llm_cost = 0.0
        self.llm_calls = 0
        self.meta = {}              # run parameters of the first start
        self.profile_locks = {}

    # ------------------------------------------------------------ persistence
    @property
    def path(self):
        return self.wd / "tree.json"

    def node_dir(self, nid):
        return self.wd / "nodes" / nid

    def save(self):
        with self.lock:
            data = {"version": 1, "root": self.root, "attempts": self.attempts,
                    "iterations_done": self.iterations_done, "llm_cost": self.llm_cost,
                    "llm_calls": self.llm_calls, "meta": self.meta, "saved": time.strftime("%F %T"),
                    "nodes": {k: n.to_json() for k, n in self.nodes.items()}, "failures": self.failures}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.path)

    @classmethod
    def load(cls, workdir):
        t = cls(workdir)
        d = json.loads(t.path.read_text())
        t.root, t.attempts = d["root"], d["attempts"]
        t.iterations_done, t.llm_cost = d["iterations_done"], d["llm_cost"]
        t.llm_calls, t.meta, t.failures = d.get("llm_calls", 0), d.get("meta", {}), d["failures"]
        for k, v in d["nodes"].items():
            t.nodes[k] = Node(**v)
            t.hashes[v["chash"]] = k
        for f in t.failures:
            if f.get("chash"):
                t.hashes.setdefault(f["chash"], f"failed:{f['id']}")
        # attempts that were in flight when the previous run died never completed: forget them
        t.attempts = max(t.attempts, t.iterations_done)
        return t

    # ---------------------------------------------------------------- hashes
    def reserve_hash(self, chash, tag):
        """Atomically claim a content hash. False if already known (node, failure or in flight)."""
        with self.lock:
            if chash in self.hashes:
                return False
            self.hashes[chash] = f"inflight:{tag}"
            return True

    def mark_failed_hash(self, chash, tag):
        with self.lock:
            self.hashes[chash] = f"failed:{tag}"

    def owner_of(self, chash):
        with self.lock:
            return self.hashes.get(chash)

    # -------------------------------------------------------------- the math
    def alpha(self, a0, a1, anneal_steps):
        """Progressive-widening exponent, annealed linearly a0 -> a1 over the first
        `anneal_steps` root visits (paper: 0.5 -> 0.3 over 1000)."""
        root = self.nodes[self.root]
        p = min(1.0, root.visits / max(1, anneal_steps))
        return a0 - (a0 - a1) * p

    def best_sub_L(self, nid):
        """Lowest L over the subtree rooted at nid (including itself)."""
        best, stack = float("inf"), [nid]
        while stack:
            n = self.nodes[stack.pop()]
            best = min(best, n.L)
            stack.extend(n.children)
        return best

    def lcb(self, child, parent, C, vloss):
        """L_c - C * sqrt(ln N_p / N_c) with L_c the best-subtree latency; virtual visits
        count in both N and add `vloss` to L per in-flight pass (virtual loss)."""
        n_c = max(1, child.visits + child.virtual)
        n_p = max(1, parent.visits + parent.virtual)
        return self.best_sub_L(child.id) + vloss * child.virtual - C * math.sqrt(math.log(n_p) / n_c)

    # -------------------------------------------------------------- selection
    def select(self, rng, C, vloss, a0, a1, anneal_steps):
        """Walk from the root: a node with N visits may take another child while
        |children| <= floor(N^alpha); else descend to the child with the lowest LCB.
        Returns (path of node ids ending at the node to expand, info dict) and applies
        virtual loss along the path."""
        with self.lock:
            alpha = self.alpha(a0, a1, anneal_steps)
            cur, path, trace = self.nodes[self.root], [], []
            while True:
                path.append(cur.id)
                n_eff = max(1, cur.visits + cur.virtual)
                cap = math.floor(n_eff ** alpha)
                n_ch = len(cur.children) + cur.pending
                if n_ch <= cap or not cur.children:
                    trace.append(f"{cur.id}: N={n_eff} children={n_ch} <= floor(N^{alpha:.3f})={cap} -> expand")
                    break
                scored = sorted(((self.lcb(self.nodes[c], cur, C, vloss), c) for c in cur.children))
                trace.append(f"{cur.id}: N={n_eff} children={n_ch} > {cap}; LCBs " +
                             ", ".join(f"{c}={s:.3f}" for s, c in scored[:4]) + f" -> {scored[0][1]}")
                cur = self.nodes[scored[0][1]]
            for nid in path:
                self.nodes[nid].virtual += 1
            cur.pending += 1
            return path, {"alpha": round(alpha, 4), "trace": trace}

    def release(self, path):
        """Undo virtual loss for a finished (or abandoned) expansion."""
        with self.lock:
            for nid in path:
                self.nodes[nid].virtual = max(0, self.nodes[nid].virtual - 1)
            self.nodes[path[-1]].pending = max(0, self.nodes[path[-1]].pending - 1)

    def backprop(self, path, child_id=None):
        """Real visits along the path (+1 for the new child). Best-subtree latency is
        derived on demand from the nodes, so only visit counts are stored."""
        with self.lock:
            for nid in path:
                self.nodes[nid].visits += 1
            if child_id:
                self.nodes[child_id].visits = max(1, self.nodes[child_id].visits)

    # ------------------------------------------------------------ add nodes
    def add_node(self, node):
        with self.lock:
            self.nodes[node.id] = node
            self.hashes[node.chash] = node.id
            if node.parent:
                self.nodes[node.parent].children.append(node.id)
            else:
                self.root = node.id

    def add_failure(self, **f):
        with self.lock:
            self.failures.append(f)

    def new_attempt_id(self):
        with self.lock:
            self.attempts += 1
            return f"a{self.attempts:03d}"

    # ---------------------------------------------------------------- views
    def lineage(self, nid):
        out = []
        while nid:
            n = self.nodes[nid]
            out.append(n)
            nid = n.parent
        return out[::-1]

    def best(self, key=None):
        key = key or (lambda n: n.score)
        return max(self.nodes.values(), key=key)

    def lineage_text(self, nid, with_delta=True):
        L = []
        prev = None
        for n in self.lineage(nid):
            d = ""
            if prev is not None and with_delta:
                d = f" ({(n.score / prev.score - 1) * 100:+.1f}% vs parent)"
            what = n.notes.replace("\n", " ")[:300] if n.action != "root" else "production kernel"
            L.append(f"  {n.id} [{n.action}] {what} -> score {n.score:.4f}x{d}")
            if n.auto_change and n.action != "root":
                L.append(f"      diff: {n.auto_change}")
            prev = n
        return "\n".join(L)

    def tree_text(self, nid=None, indent=0, C=None):
        nid = nid or self.root
        n = self.nodes[nid]
        s = (f"{'  ' * indent}{n.id} [{n.action}] N={n.visits} score={n.score:.4f} "
             f"best-subtree={1 / self.best_sub_L(nid):.4f} {n.notes.replace(chr(10), ' ')[:90]}\n")
        for c in n.children:
            s += self.tree_text(c, indent + 1)
        return s
