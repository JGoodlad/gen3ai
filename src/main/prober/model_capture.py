"""ONE batched eager forward over a battle's recorded decisions, with read-only hooks — the raw
material of `/game`'s model panels (`designs/prober/battle_view_v2.md` §2–§4).

Part of the prober's TORCH BOUNDARY (`ProbeModel.capture_battle` is the only caller): it runs the
policy on the STORED observations (`states.npz`) and reads what the forward already computes —
nothing in the model changes and nothing is trained:

* the action distribution and the pointer head's RAW per-action scores (a forward hook on
  `pointer_head`, whose output is the [B,11] logits before the mask);
* the win-probability head (`last_win_prob_logits`);
* the flat opponent pointer (`last_flat_intent_logits` + `last_flat_intent`, X5) and the hypothesis
  set (`last_hypothesis`);
* the trunk's attention: a forward PRE-hook on every trunk round — the POST-LN `BiasedEncoderLayer`s and, under
  ``--trunk-layers 3/4``, the PRE-LN `IdentityInitRound`s appended after them (`agents.model.trunk_depth`) —
  receives the round's input and its additive bias and recomputes ``softmax(q kᵀ/√d_head + bias)`` from the
  round's own ``in_proj`` (over ``LN₁(x)`` for a pre-LN round, whose attention reads the normed input) — exactly
  what ``scaled_dot_product_attention`` consumes (its default scale is 1/√d_head; the bias carries the key-padding
  addend and every edge family). The rounds stack in EXECUTION order (``attention[:, l]`` is round ``l``).
  `PolicyStateQuery` (``--policy-readout trunk`` only) the same way;
* the DamageOperator's pre-gain stash (`damage_op.last_raw_block`) and, when the move-resolution family
  is built, its RAW facts (a pre-hook re-reads `MoveResolutionCell.raw` on the same ops).

Every hook is installed for one capture and removed in a ``finally``; with no capture running the
model carries none, so the cost when off is zero. Eager only: a compiled extractor would skip the
module hooks, so a caller must hand an uncompiled policy (the prober's default; `--compile` routes a
grad-free call to the compiled path, which is why `capture_battle` runs the EAGER forward explicitly).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

#: Rows per forward. Measured 2026-10-08 on the production arch, CPU: ~0.09 s per 64 rows.
CAPTURE_BATCH = 64
#: How many species the belief-evolution read keeps per decision (by presence).
SPECIES_TOP = 12
TAIL_TOP = 5
_MASKED = -1e8          # a key whose bias is below this on every query row is key-padded
#: The trunk's attention ROUNDS, by class name (this module imports no model code at import time): production's two
#: post-LN `BiasedEncoderLayer`s, then the identity-init pre-LN `IdentityInitRound`s that `--trunk-layers 3/4` append.
_ROUND_TYPES = ("BiasedEncoderLayer", "IdentityInitRound")
#: Of those, the rounds that attend over ``norm1(x)`` rather than ``x`` (pre-LN: ``x ← x + attn(LN₁(x))``).
_PRE_LN_ROUNDS = ("IdentityInitRound",)


def _np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _attn_weights(mod, x, bias):
    """softmax(q kᵀ/√d + bias) for one trunk round's input — its own `in_proj`, its own heads. A PRE-LN round
    (`IdentityInitRound`) projects ``norm1(x)``; a post-LN `BiasedEncoderLayer` projects ``x`` itself."""
    import torch

    B, n, d = x.shape
    H, hd = int(mod.n_heads), int(mod.head_dim)
    if type(mod).__name__ in _PRE_LN_ROUNDS:
        x = mod.norm1(x)
    qkv = mod.in_proj(x).reshape(B, n, 3, H, hd)
    q, k = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    if bias is not None:
        logits = logits + bias.to(logits.dtype)
    return torch.softmax(logits, dim=-1)                                    # [B,H,n,n]


def _query_weights(mod, tokens, pad, key_log_presence):
    """The `PolicyStateQuery`'s one learned query over its keys → [B,H,n]."""
    import torch

    B, n, d = tokens.shape
    H = int(mod.n_heads)
    hd = d // H
    q = mod.query.expand(B, 1, d).reshape(B, 1, H, hd).transpose(1, 2)
    k = mod.k_proj(tokens).reshape(B, n, H, hd).transpose(1, 2)
    bias = pad.to(tokens.dtype) * -1e9
    if key_log_presence is not None:
        bias = bias + torch.where(pad, torch.zeros_like(key_log_presence), key_log_presence).to(tokens.dtype)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + bias[:, None, None, :]
    return torch.softmax(logits, dim=-1)[:, :, 0, :]                          # [B,H,n]


def token_layout(extractor, n_tokens: int) -> Dict[str, Any]:
    """Which trunk seat is which, read off the BUILT extractor (never a literal): the base seats (our 6,
    their 6, the board token(s)), E3 (4), E4 (K), E5 (6 when built), OTHER_species (X5), then the event
    seats. ``ok`` is False when the counts do not add up to ``n_tokens`` — a layout this reader does not
    know, so a surface labels the seats by index instead of guessing."""
    tt = getattr(extractor, "team_transformer", None)
    es = getattr(extractor, "entity_seats", None)
    base = int(getattr(tt, "_total_tokens", 13) or 13)
    board = tuple(int(i) for i in (getattr(tt, "board_seats", None) or (12,)))
    if base == 13:
        board = (12,)
    n_e3 = 4 if es is not None else 0
    k_e4 = int(getattr(es, "topk_seats", 0) or 0) if es is not None else 0
    n_tail = 6 if (es is not None and getattr(es, "tail_seats", False)) else 0
    has_other = getattr(extractor, "hypothesis_builder", None) is not None
    rest = n_tokens - base - n_e3 - k_e4 - n_tail - (1 if has_other else 0)
    return {"n_tokens": int(n_tokens), "base": base, "board_seats": list(board), "n_e3": n_e3,
            "k_e4": k_e4, "n_tail": n_tail, "has_other": bool(has_other), "n_events": max(rest, 0),
            "ok": rest >= 0}


def capture(policy, obs: np.ndarray, masks: np.ndarray, *, batch: int = CAPTURE_BATCH) -> Dict[str, Any]:
    """The raw readout of every row of ``obs`` [N,D] / ``masks`` [N,11] — numpy arrays, keyed by name.
    A head the model does not build is ABSENT from the result (never zeros)."""
    import torch

    from agents.model.damage_op_layout import decode_damage_block

    fe = policy.features_extractor
    layers = [m for m in fe.modules() if type(m).__name__ in _ROUND_TYPES]
    pq = getattr(fe, "policy_query", None)
    mr = getattr(fe, "move_resolution_cell", None)
    ptr = getattr(policy, "pointer_head", None)
    op = getattr(fe, "damage_op", None)

    cur: Dict[str, List] = {"attn": [], "query": [], "ptr": [], "mr_m": [], "mr_s": [], "keymask": []}
    handles = []

    def layer_hook(mod, args, kwargs):      # one call per trunk round, in execution order
        x = args[0]
        bias = args[1] if len(args) > 1 else kwargs.get("bias")
        w = _attn_weights(mod, x, bias)
        cur["attn"].append(w)
        if bias is not None and not cur["keymask"]:
            cur["keymask"].append((bias[:, 0] <= _MASKED).all(dim=1))         # [B,n]

    def query_hook(mod, args, kwargs):
        tokens = args[0]
        pad = args[1] if len(args) > 1 else kwargs.get("pad")
        klp = args[2] if len(args) > 2 else kwargs.get("key_log_presence")
        cur["query"].append(_query_weights(mod, tokens, pad, klp))

    def ptr_hook(mod, args, out):
        cur["ptr"].append(out)

    def mr_hook(mod, args, kwargs):
        ops = args[0] if args else kwargs.get("ops")
        m, s = mod.raw(ops)
        cur["mr_m"].append(m)
        cur["mr_s"].append(s)

    out: Dict[str, List] = {}

    def put(k, v):
        out.setdefault(k, []).append(v)

    try:
        for L in layers:
            handles.append(L.register_forward_pre_hook(layer_hook, with_kwargs=True))
        if pq is not None:
            handles.append(pq.register_forward_pre_hook(query_hook, with_kwargs=True))
        if ptr is not None:
            handles.append(ptr.register_forward_hook(ptr_hook))
        if mr is not None:
            handles.append(mr.register_forward_pre_hook(mr_hook, with_kwargs=True))
        n = int(obs.shape[0])
        for s in range(0, n, batch):
            for v in cur.values():
                v.clear()
            ot = torch.as_tensor(np.asarray(obs[s:s + batch], dtype=np.float32))
            mt = torch.as_tensor(np.asarray(masks[s:s + batch]))
            with torch.no_grad():
                d = policy.get_distribution({"observation": ot, "action_mask": mt})
                lg = d.distribution.logits
                masked = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
                put("probs", _np(torch.softmax(masked, 1)))
                if cur["ptr"]:
                    put("scores", _np(cur["ptr"][-1]))
                wl = getattr(fe, "last_win_prob_logits", None)
                if wl is not None:
                    put("win_prob", _np(torch.sigmoid(wl.reshape(-1))))
                if cur["attn"]:
                    put("attention", np.stack([_np(a) for a in cur["attn"]], axis=1).astype(np.float16))
                    if cur["keymask"]:
                        put("key_masked", cur["keymask"][0].cpu().numpy())
                if cur["query"]:
                    put("query_attention", _np(cur["query"][-1]))
                fl = getattr(fe, "last_flat_intent_logits", None)
                fi = getattr(fe, "last_flat_intent", None)
                if fl is not None and fi is not None:
                    live = fi.live.bool()
                    z = torch.where(live, fl, torch.full_like(fl, float("-inf")))
                    p = torch.softmax(z, dim=-1)
                    p = torch.where(live.any(-1, keepdim=True), p, torch.zeros_like(p))
                    put("intent_p", _np(p))
                    put("intent_live", live.cpu().numpy())
                    put("intent_cand", fi.cand_ids.cpu().numpy())
                    put("intent_hp_seat", fi.hp_seat.cpu().numpy())
                    out["intent_k"] = [int(fi.k)]
                hs = getattr(fe, "last_hypothesis", None)
                if hs is not None:
                    put("slot_species", hs.slot_species.cpu().numpy())
                    put("slot_is_hyp", hs.slot_is_hypothesis.cpu().numpy())
                    if hs.slot_pi is not None:
                        put("slot_pi", _np(hs.slot_pi))
                    put("hyp_species", hs.hyp_species.cpu().numpy())
                    put("hyp_pi", _np(hs.hyp_pi))
                    put("hyp_live", hs.hyp_live.cpu().numpy())
                    put("other_mass", _np(hs.other_mass))
                    put("other_live", hs.other_live.cpu().numpy())
                    put("other_any", _np(hs.other_any))
                    tp, ti = torch.topk(hs.other_tail_probs.float(), k=min(TAIL_TOP, hs.other_tail_probs.shape[-1]), dim=-1)
                    put("tail_idx", ti.cpu().numpy())
                    put("tail_p", _np(tp))
                    sp, si = torch.topk(hs.species.pi.float(), k=min(SPECIES_TOP, hs.species.pi.shape[-1]), dim=-1)
                    put("species_idx", si.cpu().numpy())
                    put("species_pi", _np(sp))
                    if hs.moves is not None:
                        mv = hs.moves
                        put("seat_nums", mv.seat_nums.cpu().numpy())
                        put("seat_live", mv.seat_live.cpu().numpy())
                        put("seat_revealed", mv.seat_revealed.cpu().numpy())
                        put("seat_pi", _np(mv.seat_pi))
                        put("move_other_mass", _np(mv.other_mass))
                if op is not None and getattr(op, "last_raw_block", None) is not None:
                    raw = op.last_raw_block
                    # `op_drop_renders` (production) computes the matrices but does not SERIALIZE them
                    # into the row — decoding them would read past its end.
                    dropped = bool(getattr(op, "drop_renders", False))
                    kin = 0 if dropped else int(getattr(op, "matrices_incoming_k", 0))
                    mout = False if dropped else bool(getattr(op, "matrices_outgoing", False))
                    for r in range(raw.shape[0]):
                        put("op", decode_damage_block(raw[r].detach().cpu().numpy(),
                                                      outgoing=bool(getattr(op, "outgoing", False)),
                                                      matrices_outgoing=mout, matrices_incoming_k=kin))
                if cur["mr_m"]:
                    put("mr_move", _np(cur["mr_m"][-1]))
                    put("mr_switch", _np(cur["mr_s"][-1]))
    finally:
        for h in handles:
            h.remove()

    res: Dict[str, Any] = {}
    for k, v in out.items():
        if k == "op":
            res[k] = v
        elif k == "intent_k":
            res[k] = v[0]
        else:
            res[k] = np.concatenate(v, axis=0)
    if "attention" in res:
        res["layout"] = token_layout(fe, int(res["attention"].shape[-1]))
    if pq is not None:
        res["query_layout"] = {"n_keys": int(res["query_attention"].shape[-1])} if "query_attention" in res else None
    if mr is not None:
        from agents.model.move_resolution_rules import (MOVE_RESOLUTION_MOVE_COORDS,
                                                        MOVE_RESOLUTION_SWITCH_COORDS)
        res["mr_move_coords"] = list(MOVE_RESOLUTION_MOVE_COORDS)
        res["mr_switch_coords"] = list(MOVE_RESOLUTION_SWITCH_COORDS)
    res["n_layers"] = len(layers)
    return res


__all__ = ["CAPTURE_BATCH", "capture", "token_layout"]
