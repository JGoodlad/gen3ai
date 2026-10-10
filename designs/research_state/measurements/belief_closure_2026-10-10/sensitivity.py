"""Does an opponent mon's UNREVEALED-move belief reach the model's outputs once all four of its moves are revealed?

The four-move fact: a gen-3 mon has at most four moves, so once four are revealed every other move's presence is 0.
A model that honours it EXACTLY gives the same policy and value whatever the move head says about the unrevealed
moves of such a mon.

Two reads, on real decision rows (the Rust env core's seeded fixture battles, or a probe-battery bank):

* ``bump`` (default): add ``±--bump`` to the published move posterior's logit of EVERY unrevealed move (num >= 1) of
  every opponent mon with four revealed moves, and report the largest change in the masked policy probabilities /
  log-probabilities and in the value — split into rows where the four-revealed mon is the opponent's ACTIVE (and no
  other mon shows four) and rows where an alive BENCHED mon has four. A change is a leak.
* ``--effect``: the outputs with the closure ON vs OFF on the same rows, no bump — how much a model's outputs move
  when the fact is enforced (on a model trained without it this is an out-of-distribution read: it says how much the
  model READS the leaked mass, not what a model trained with the fact would do).

The closure is applied by ``--move-set-closure`` (`move_set_closure` on the extractor) where the code has it, else
EMULATED (the pin before the flag existed): the reinjection's weights are replaced by every slot's fixed-mass presence
(`hypothesis_tokens.slot_move_presence`, the construction the op roster already reads) — the same values.

``--zero-reinject`` zeroes MoveBelief's reinjection projection (a localisation control: if the change vanishes, the
reinjection is the leak). ``--checkpoint`` reads a trained ``.zip`` (it must load at the imported tree; use ``--src``
for a PIN checkout's ``src``). ``--perturb s`` adds the K9 golden's name-keyed noise to a fresh build.

CPU only. Run from the repo root (it imports ``./src`` unless ``--src``; the fixture battles read ``data/`` relative
to the cwd):

    python designs/research_state/measurements/belief_closure_2026-10-10/sensitivity.py --battles 60 --perturb 0.02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def _src_root(argv) -> str:
    """``--src DIR`` (e.g. a read-only PIN checkout's ``src``), else ``./src``. Read before any project import."""
    for i, x in enumerate(argv):
        if x == "--src" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--src="):
            return x.split("=", 1)[1]
    return os.path.join(os.getcwd(), "src")


sys.path.insert(0, _src_root(sys.argv))

import numpy as np  # noqa: E402
import torch  # noqa: E402

TEAM = 6


def _rows(n_battles: int, seed: int):
    from utils.rust_env.fixture_battles import play_rows
    games = play_rows(n_battles, seed=seed, n_teams=None, n_envs=2)
    return np.concatenate([g[0] for g in games], 0), np.concatenate([g[1] for g in games], 0)


def _bank_rows(bank: str):
    """A probe-battery bank's decision rows (``rows.npy`` / ``masks.npy``) — at the obs layout the bank recorded."""
    return np.load(f"{bank}/rows.npy"), np.load(f"{bank}/masks.npy")


class Closure:
    """Switch the four-move closure on / off: the flag where the extractor has it, else the emulation."""

    def __init__(self, fe: Any) -> None:
        self.fe = fe
        self.native = hasattr(fe, "move_set_closure")
        self.on = False
        if not self.native:
            mb = fe.move_belief
            orig_logits, orig_reinject = mb.move_logits, mb.reinject_moves
            seen: Dict[str, Any] = {}

            def move_logits(opp_tokens, opp_species_ids=None, opp_move_ids=None, **kw):
                seen["species"], seen["moves"] = opp_species_ids, opp_move_ids
                return orig_logits(opp_tokens, opp_species_ids, opp_move_ids, **kw)

            def reinject_moves(opp_tokens, apply_mask, move_embedding, move_logits, weights=None):
                if self.on:
                    from agents.model.hypothesis_tokens import slot_move_presence
                    w, _r, _s = slot_move_presence(fe.hypothesis_builder, move_logits, seen["species"],
                                                   seen["moves"])
                    weights = w.to(move_logits.dtype)
                return orig_reinject(opp_tokens, apply_mask, move_embedding, move_logits, weights=weights)

            mb.move_logits, mb.reinject_moves = move_logits, reinject_moves

    def set(self, on: bool) -> None:
        self.on = on
        if self.native:
            self.fe.move_set_closure = "on" if on else "off"


def _forward(policy: Any, obs_t, mask_t, sel, chunk: int = 128):
    vs, lps = [], []
    with torch.no_grad():
        for c in sel.split(chunk):
            v, lp = policy.rollout_core({"observation": obs_t[c]}, mask_t[c])
            vs.append(v.reshape(-1))
            lps.append(lp)
    return torch.cat(vs), torch.cat(lps)


def _summ(dv: torch.Tensor, dpr: torch.Tensor, flip: torch.Tensor) -> dict:
    q = lambda t, p: float(torch.quantile(t, p)) if t.numel() else float("nan")  # noqa: E731
    return {"rows": int(dv.numel()),
            "dvalue_median": q(dv, 0.5), "dvalue_p90": q(dv, 0.9), "dvalue_max": float(dv.max()),
            "dprob_median": q(dpr, 0.5), "dprob_p90": q(dpr, 0.9), "dprob_max": float(dpr.max()),
            "argmax_flip_rate": float(flip.float().mean()),
            "rows_with_any_change": int(((dv > 0) | (dpr > 0)).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="the checkout's src/ to import (default ./src)")
    ap.add_argument("--battles", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--bank", default=None, help="read decision rows from a probe-battery bank instead of playing")
    ap.add_argument("--max-rows", type=int, default=2000, help="cap on the rows forwarded per category")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--bump", type=float, default=6.0)
    ap.add_argument("--closure", action="store_true", help="bump read with the closure ON")
    ap.add_argument("--effect", action="store_true", help="closure ON vs OFF instead of the bump read")
    ap.add_argument("--zero-reinject", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="add name-keyed N(0, s) noise to every parameter (the K9 golden's perturbation), so the "
                         "fresh build's zero-init heads (the pointer scorer among them) carry signal")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)

    if a.checkpoint:
        from agents.model.snapshot import load_checkpoint_strict
        model = load_checkpoint_strict(a.checkpoint, device="cpu")
    else:
        from main.fresh_checkpoint import build_fresh_model
        model, _args, _pk = build_fresh_model(a.model_seed)
    policy = model.policy
    policy.set_training_mode(False)
    fe = policy.features_extractor
    if a.perturb > 0:
        from agents.model.parity_probe import perturb_
        perturb_(policy, seed=1234, scale=a.perturb, keyed=True)
    if a.zero_reinject:
        with torch.no_grad():
            fe.move_belief.reinject.weight.zero_()
            fe.move_belief.reinject.bias.zero_()
    closure = Closure(fe)

    obs, mask = _bank_rows(a.bank) if a.bank else _rows(a.battles, a.seed)
    obs_t, mask_t = torch.as_tensor(obs), torch.as_tensor(mask)
    with torch.no_grad():
        ctx = fe.unpack({"observation": obs_t})
    mids, act = ctx.all_move_ids[:, TEAM:, :], ctx.opp_active_local                 # [B,6,4], [B]
    alive = ctx.hp_and_active[:, TEAM:, 0] > 0                                      # [B,6]
    four_any = (mids > 0).sum(-1) == 4                                              # [B,6] (fainted too)
    four = four_any & alive
    onehot = torch.nn.functional.one_hot(act, TEAM).bool()
    cats = {
        # the bump reaches every four-revealed slot, so the ACTIVE category excludes any other such slot
        "opp_ACTIVE_has_4_only": ((four & onehot).any(-1) & ~(four_any & ~onehot).any(-1)),
        "alive_BENCH_mon_has_4": (four & ~onehot).any(-1),
    }
    if a.effect:
        cats["no_opp_mon_has_4"] = ~four_any.any(-1)
    head = {"rows": int(len(obs)), "mode": "effect" if a.effect else "bump", "bump": a.bump,
            "closure": a.closure, "closure_mechanism": "flag" if closure.native else "emulated",
            "zero_reinject": a.zero_reinject, "bank": a.bank, "checkpoint": a.checkpoint, "perturb": a.perturb,
            "category_rows_available": {k: int(v.sum()) for k, v in cats.items()}}

    mb = fe.move_belief
    orig = mb.move_logits
    state = {"bump": 0.0}

    def bumped(opp_tokens, opp_species_ids=None, opp_move_ids=None, **kw):
        out = orig(opp_tokens, opp_species_ids, opp_move_ids, **kw)
        if state["bump"] == 0.0 or opp_move_ids is None:
            return out
        M = out.shape[-1]
        rev = torch.zeros_like(out, dtype=torch.bool)
        rev.scatter_(-1, opp_move_ids.clamp(0, M - 1), opp_move_ids > 0)
        slot4 = ((opp_move_ids > 0).sum(-1) == 4).unsqueeze(-1)                   # [B,6,1]
        hit = slot4 & ~rev & (torch.arange(M, device=out.device) >= 1)
        return torch.where(hit, out + state["bump"], out)

    mb.move_logits = bumped
    res: Dict[str, Any] = {}
    g = torch.Generator().manual_seed(0)
    for name, m in cats.items():
        idx = m.nonzero().flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() > a.max_rows:
            idx = idx[torch.randperm(idx.numel(), generator=g)[: a.max_rows]].sort().values
        legal = mask_t[idx].bool()
        if a.effect:
            closure.set(False)
            v0, l0 = _forward(policy, obs_t, mask_t, idx)
            closure.set(True)
            v1, l1 = _forward(policy, obs_t, mask_t, idx)
            closure.set(False)
            dv = (v1 - v0).abs()
            dpr = (l1.exp() - l0.exp()).abs().masked_fill(~legal, 0).amax(-1)
            flip = l1.masked_fill(~legal, -1e9).argmax(-1) != l0.masked_fill(~legal, -1e9).argmax(-1)
        else:
            closure.set(a.closure)
            state["bump"] = 0.0
            v0, l0 = _forward(policy, obs_t, mask_t, idx)
            dv = torch.zeros_like(v0)
            dpr = torch.zeros(len(idx))
            flip = torch.zeros(len(idx), dtype=torch.bool)
            for b in (a.bump, -a.bump):
                state["bump"] = b
                v1, l1 = _forward(policy, obs_t, mask_t, idx)
                dv = torch.maximum(dv, (v1 - v0).abs())
                dpr = torch.maximum(dpr, (l1.exp() - l0.exp()).abs().masked_fill(~legal, 0).amax(-1))
                flip |= l1.masked_fill(~legal, -1e9).argmax(-1) != l0.masked_fill(~legal, -1e9).argmax(-1)
            state["bump"] = 0.0
        res[name] = _summ(dv, dpr, flip)

    # The PUBLISHED posterior's residual on four-revealed alive mons (what the prober's move list shows): Σ and max of
    # sigmoid over every unrevealed num >= 1 but the typeless 237.
    idx = four.any(-1).nonzero().flatten()[: a.max_rows]
    closure.set(False)
    sums, maxs = [], []
    with torch.no_grad():
        for c in idx.split(128):
            policy.rollout_core({"observation": obs_t[c]}, mask_t[c])
            pub = fe.last_move_belief_logits
            M = pub.shape[-1]
            m4 = mids[c]
            rev = torch.zeros_like(pub, dtype=torch.bool)
            rev.scatter_(-1, m4.clamp(0, M - 1), m4 > 0)
            num = torch.arange(M)
            p = torch.sigmoid(pub) * (~rev & (num >= 1) & (num != 237))
            sums.append(p.sum(-1)[four[c]])
            maxs.append(p.amax(-1)[four[c]])
    rs, rm = torch.cat(sums), torch.cat(maxs)
    res["published_residual_on_alive_4_revealed_mons"] = {
        "mons": int(rs.numel()), "sum_sigmoid_unrevealed_mean": float(rs.mean()),
        "sum_sigmoid_unrevealed_median": float(rs.median()), "max_single_move_mean": float(rm.mean()),
        "frac_mons_with_an_unrevealed_move_ge_0.5": float((rm >= 0.5).float().mean())}
    print(json.dumps({**head, **res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
