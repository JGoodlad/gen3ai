"""CHEAT probe (investigator A): two battles whose p2 teams differ ONLY in a hidden fact, identical seeds and a
policy that is a pure function of (decision index, mask) on both sides; diff p1's rows decision by decision.

p1: seeded draw over the mask (a fresh generator per decision keyed on (seed, k), so it depends on the mask only).
p2: move slot 1 when legal, else the lowest legal action (never voluntarily touches the varied fact).

Usage: python leak_diff.py  (prints, per variant, the first differing decision and the differing cells).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from utils.rust_env import ffi
from utils.rust_env import protocol as P
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings  # noqa: F401

ENC = Gen3ObservationEncoder(load_mappings())
LAYOUT = ENC.get_layout()


import agents.observation.constants as K

_SLOT_FIELDS = sorted([(getattr(K, n), n[len("POKEMON_"):-len("_OFFSET")].lower()) for n in dir(K)
                       if n.startswith("POKEMON_") and n.endswith("_OFFSET") and isinstance(getattr(K, n), int)])
_BLOCKS = sorted([(getattr(K, n), n[len("OFFSET_"):].lower()) for n in dir(K)
                  if n.startswith("OFFSET_") and isinstance(getattr(K, n), int)])


def cell_name(i: int) -> str:
    for lo, name in reversed(_BLOCKS):
        if i >= lo:
            break
    if name in ("our_team", "opp_team"):
        k = (i - lo) % K.POKEMON_FULL_DIM
        slot = (i - lo) // K.POKEMON_FULL_DIM
        for flo, fname in reversed(_SLOT_FIELDS):
            if k >= flo:
                return f"{name}[{slot}].{fname}+{k - flo}"
    if name == "event_window":
        k = i - lo
        return f"event[{k // K.EVENT_TOKEN_DIM}]+{k % K.EVENT_TOKEN_DIM}"
    if name == "context":
        k = i - lo
        return f"context[{k // K.ACTIVE_CONTEXT_DIM}]+{k % K.ACTIVE_CONTEXT_DIM}"
    return f"{name}+{i - lo}"


def play(teams, seed_words, n_dec=400, seed=7, turn_limit=250):
    spec = P.spec_json(n=1, threads=1, teams=teams, names=("fixone", "fixtwo"), turn_limit=turn_limit,
                       refusal_budget=4, bank_dir=None, opponents=[{"kind": "external"}])
    lib = ffi.load(ffi.default_path("release"))
    core = ffi.FfiCore(spec, lib=lib)
    c = core.cols

    def stage():
        c["ep_team"][0] = [0, 1]
        c["ep_seed"][0] = seed_words
        c["ep_opp"][0] = 0

    stage()
    core.reset()
    stage()
    rows, masks, scripts = [], [], []
    k1 = 0
    try:
        for _ in range(4000):
            for s in (0, 1):
                if not c["need"][0, s]:
                    continue
                m = c["mask"][0, s].astype(bool)
                if s == 0:
                    rows.append(c["obs"][0, 0].copy())
                    masks.append(m.copy())
                    rng = np.random.default_rng([seed, k1])
                    a = int(rng.choice(np.flatnonzero(m)))
                    k1 += 1
                else:
                    a = 6 if m[6] else int(np.flatnonzero(m)[0])
                c["action"][0, s] = a
            core.step()
            fin = core.finished()
            if fin:
                scripts.append(fin[0]["script"])
                break
            if len(rows) >= n_dec:
                break
    finally:
        core.close()
    return np.array(rows), np.array(masks), (scripts[0] if scripts else None)


def opp_hp_cells():
    base = LAYOUT["opp_team"][0] if "opp_team" in LAYOUT else None
    return base


def compare(name, teams_a, teams_b, seed_words):
    ra, ma, sa = play(teams_a, seed_words)
    rb, mb, sb = play(teams_b, seed_words)
    n = min(len(ra), len(rb))
    print(f"== {name}: decisions {len(ra)} vs {len(rb)}")
    for k in range(n):
        d = np.flatnonzero(ra[k].view(np.uint32) != rb[k].view(np.uint32))
        if len(d) or not np.array_equal(ma[k], mb[k]):
            print(f"   first difference at p1 decision {k}: {len(d)} cells; mask equal={np.array_equal(ma[k], mb[k])}")
            for i in d[:25]:
                print(f"     cell {i} {cell_name(int(i))}: {ra[k][i]!r} vs {rb[k][i]!r}")
            return k, ra, rb, sa, sb
    print("   IDENTICAL over", n, "decisions")
    return None, ra, rb, sa, sb


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in LAYOUT.items() if not isinstance(v, (dict,))}, default=str)[:3000])
