"""EVERY SnapshotPool in the repo keeps its snapshots on the CPU (`gen3_declared_slot_load_v1`).

A snapshot is a weight SOURCE: T2 copies it into a declared slot, and a pool loaded onto the card keeps
each one in its LRU beside that slot (sizing arm A, 2026-10-01: +~33 MiB of quiescent floor per
promotion). `277f318f` fixed the trainer's rust-core pool; the sizing harness's own pool still built on
cuda and died on the runtime refusal (`95af710e` fixed it) — so the rule is pinned REPO-WIDE here, one
AST scan over src/ and tools/ (tests included): a `SnapshotPool(...)` call either omits `device=` (the
default is "cpu") or passes the literal "cpu". There is NO declared exception since deletion pass U3
deleted the python env core's worker pool (the one pool that INFERRED on its snapshot)."""
from __future__ import annotations

import ast
from pathlib import Path

from utils.paths import repo_path, src_path

#: (file relative to the repo, the device expression's source) — the declared exceptions: a pool whose
#: snapshots are the INFERENCE model (no T2 slot), never a slot's weight source. None today.
DECLARED: dict = {}


def _calls():
    root = Path(repo_path())
    for top in (Path(src_path()), Path(repo_path("tools"))):
        for p in top.rglob("*.py"):
            if "SnapshotPool(" not in p.read_text(errors="replace"):
                continue
            for node in ast.walk(ast.parse(p.read_text(errors="replace"))):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if name == "SnapshotPool":
                        yield str(p.relative_to(root)), node


def test_the_default_device_is_the_CPU():
    import inspect
    from agents.training.snapshot_pool import SnapshotPool
    assert inspect.signature(SnapshotPool.__init__).parameters["device"].default == "cpu"


def test_every_SnapshotPool_keeps_its_snapshots_on_the_CPU():
    found, bad, used = 0, [], set()
    for path, call in _calls():
        found += 1
        kw = {k.arg: k.value for k in call.keywords}
        dev = kw.get("device")
        if dev is None or (isinstance(dev, ast.Constant) and dev.value == "cpu"):
            continue
        key = (path, ast.unparse(dev))
        if key in DECLARED:
            used.add(key)
            continue
        bad.append(f"{path}:{call.lineno} device={ast.unparse(dev)}")
    assert found >= 8, f"the scan found only {found} SnapshotPool calls — is it still looking?"
    assert not bad, ("a SnapshotPool that loads its snapshots onto a non-CPU device — a pool refresh is "
                     f"a DECLARED LOAD into a T2 slot, never a device copy: {bad}")
    assert used == set(DECLARED), f"a declared exception no longer exists: {set(DECLARED) - used}"
