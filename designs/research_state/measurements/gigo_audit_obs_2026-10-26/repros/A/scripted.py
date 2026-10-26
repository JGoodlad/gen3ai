"""A scripted two-sided battle on the env core (investigator A): both sides answered by fixed action lists
(fallback: move slot 1, else the lowest legal action). Returns each side's (row, mask) per decision.

Action layout (present/legal.rs): 0-5 switch to team slot i, 6-9 move slot (request order), 10 struggle."""
from __future__ import annotations

import numpy as np

from utils.rust_env import ffi
from utils.rust_env import protocol as P


def play(p1_team, p2_team, p1_plan, p2_plan, seed_words=(1, 2, 3, 4), max_steps=200):
    spec = P.spec_json(n=1, threads=1, teams=[p1_team, p2_team], names=("fixone", "fixtwo"), turn_limit=250,
                       refusal_budget=4, bank_dir=None, opponents=[{"kind": "external"}])
    core = ffi.FfiCore(spec, lib=ffi.load(ffi.default_path("release")))
    c = core.cols

    def stage():
        c["ep_team"][0] = [0, 1]
        c["ep_seed"][0] = list(seed_words)
        c["ep_opp"][0] = 0

    stage()
    core.reset()
    stage()
    out = {0: [], 1: []}
    plans = {0: list(p1_plan), 1: list(p2_plan)}
    script = None
    try:
        for _ in range(max_steps):
            for s in (0, 1):
                if not c["need"][0, s]:
                    continue
                m = c["mask"][0, s].astype(bool)
                out[s].append((c["obs"][0, s].copy(), m.copy(), int(c["turn"][0])))
                k = len(out[s]) - 1
                want = plans[s][k] if k < len(plans[s]) else None
                if want is not None and m[want]:
                    a = want
                else:
                    a = 6 if m[6] else int(np.flatnonzero(m)[0])
                c["action"][0, s] = a
            core.step()
            fin = core.finished()
            if fin:
                script = fin[0]["script"]
                break
    finally:
        core.close()
    return out, script


def trace_lines(script, side):
    """Each side's core-event record text for a finished script (``ffi.trace``)."""
    lib = ffi.load(ffi.default_path("release"))
    t = ffi.trace(lib, script, commit="audit", label="A")
    return t
