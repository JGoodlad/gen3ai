"""PHASE 2c (Amendment 6): the network's OWN per-mon move belief at a root, read from one CPU forward at the pin.

What it reads: the X5 op roster's ``move_w`` [B,6,M] (`hypothesis_tokens.OpRoster`), the per-opponent-mon FIXED-MASS
move presence the damage operator itself prices with: over the mon species' legal moves minus its revealed ones,
π_m = σ(a + τ) with Σ π_m = 4 − r (a = the composed move posterior `last_move_belief_logits`, the Smogon prior fused
with the network's learned correction; `slot_move_presence`), 1 on a revealed move. On a REVEALED slot it is that
species' move belief; on a HIDDEN slot it is the belief for the slot's HYPOTHESIS species (the top-k species).
The roster is not stashed by the extractor, so ``build_op_roster`` is wrapped for the duration of one call (the
wrapper returns the roster unchanged; the forward's outputs are byte-identical).

``root_move_beliefs(model, rows, masks)`` → per row, 6 dicts (opponent slots in obs order):
``{"num": species num, "hyp": bool (a hypothesis slot), "w": {move id: π} over the UNREVEALED candidates}``.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np

TEAM = 6


def move_ids(data_dir: Path) -> Dict[int, str]:
    d = json.loads((Path(data_dir) / "gen3_moves.json").read_text())
    out = {int(v["num"]): k for k, v in d.items() if v.get("num")}
    if len(out) != len(d):
        raise RuntimeError("gen3_moves.json: move nums are not one-to-one")
    return out


def species_nums(data_dir: Path) -> Dict[str, int]:
    d = json.loads((Path(data_dir) / "gen3_species.json").read_text())
    return {k: int(v["num"]) for k, v in d.items() if v.get("num")}


@contextmanager
def capture_roster():
    import agents.model.extractor_forward as XF

    box: Dict[str, object] = {}
    orig = XF.build_op_roster

    def wrap(hb, ctx, hctx, hs, fm, move_logits, *a, **k):
        ro, hs2 = orig(hb, ctx, hctx, hs, fm, move_logits, *a, **k)
        box["roster"] = ro
        box["ctx"] = ctx
        box["hs"] = hs2
        return ro, hs2

    XF.build_op_roster = wrap
    try:
        yield box
    finally:
        XF.build_op_roster = orig


def slots_from_box(box: dict, num2move: Dict[int, str]) -> List[List[dict]]:
    """The captured roster of ONE forward → per row, the 6 opponent slots' move beliefs (see the module doc)."""
    if "roster" not in box:
        raise RuntimeError("the forward built no X5 op roster (no move belief to read)")
    ro, ctx, hs = box["roster"], box["ctx"], box["hs"]
    w = ro.move_w.float().cpu().numpy()                              # [B,6,M]
    sp = ctx.species_ids[:, TEAM:].long().cpu().numpy()              # [B,6] revealed nums (0 hidden)
    hyp = hs.slot_is_hypothesis.cpu().numpy()                        # [B,6]
    hsp = hs.slot_species.long().cpu().numpy()                       # [B,6] hypothesis nums
    rev = ctx.all_move_ids[:, TEAM:, :].long().cpu().numpy()         # [B,6,4] revealed move nums
    out: List[List[dict]] = []
    for b in range(w.shape[0]):
        slots = []
        for s in range(TEAM):
            revealed = {int(x) for x in rev[b, s] if int(x) > 0}
            # a revealed move's own weight is 1 (dropped); a revealed typeless Hidden Power's typed channels carry
            # its TYPE distribution P(t), kept here and dropped by the sampler (the kept set already holds its HP)
            ww = {num2move[int(n)]: float(w[b, s, n]) for n in np.flatnonzero(w[b, s] > 0)
                  if int(n) not in revealed and int(n) in num2move}
            slots.append({"num": int(hsp[b, s] if hyp[b, s] else sp[b, s]), "hyp": bool(hyp[b, s]),
                          "revealed": sorted(num2move.get(n, str(n)) for n in revealed), "w": ww})
        out.append(slots)
    return out


def root_move_beliefs(model, rows: np.ndarray, masks: np.ndarray, num2move: Dict[int, str],
                      batch: int = 64) -> List[List[dict]]:
    from main.policy_spectrum.qhat import logits_and_values

    out: List[List[dict]] = []
    with capture_roster() as box:
        for i in range(0, len(rows), batch):
            box.clear()
            logits_and_values(model, rows[i:i + batch], masks[i:i + batch], batch=batch)
            out.extend(slots_from_box(box, num2move))
    return out
