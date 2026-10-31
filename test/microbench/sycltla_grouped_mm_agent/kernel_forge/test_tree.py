"""Offline check of selection / progressive widening / virtual loss (no GPU, no LLM):
    python -m kernel_forge.test_tree
"""
import random
import tempfile

from .tree_store import Node, Tree


def simulate(n_iter=40, workers=4, C=0.1, seed=0):
    rng = random.Random(seed)
    t = Tree(tempfile.mkdtemp())
    t.add_node(Node(id="n000", visits=1, score=1.0, chash="root"))
    inflight = []
    k = 0
    shape = []
    while k < n_iter or inflight:
        while len(inflight) < workers and k < n_iter:
            path, info = t.select(rng, C, 0.05, 0.5, 0.3, 1000)
            k += 1
            inflight.append((k, path))
            shape.append((k, path[-1], len(path)))
        i, path = inflight.pop(0)
        t.release(path)
        par = t.nodes[path[-1]]
        child = Node(id=f"n{i:03d}", parent=par.id, depth=par.depth + 1, score=par.score * rng.uniform(0.9, 1.08),
                     chash=str(i), visits=0)
        t.add_node(child)
        t.backprop(path, child.id)
    return t, shape


if __name__ == "__main__":
    t, shape = simulate()
    print(t.tree_text())
    print("expanded-node sequence:", [(k, n) for k, n, d in shape])
    assert all(len(n.children) <= max(1, int(n.visits ** 0.5) + 1) + 1 for n in t.nodes.values()), "widening violated"
    assert all(n.virtual == 0 and n.pending == 0 for n in t.nodes.values()), "virtual loss leak"
    print("ok: root children", len(t.nodes["n000"].children), "max depth", max(n.depth for n in t.nodes.values()))
