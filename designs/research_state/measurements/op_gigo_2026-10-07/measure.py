"""How often the damage operator's 2026-10-07 GIGO fired on REAL data — the Lane S bank replayed on the Rust core.

The defect: the op read an opponent's ability as REVEALED whenever its id was non-zero, but an UNREVEALED
opponent's id1 carries its species' top-1 Smogon-prior ability (`known` = 0). The fix (gen3_op_ability_known_v1)
reads the `known` flag and mixes the species prior; gen3_op_status_rules_v1 also folds Safeguard (both sides),
incoming Sleep / Freeze Clause, our Substitute and Yawn as a delayed sleep into status landing.

Over every answered decision of BOTH viewers (rows), with the production-config extractor (move-resolution off):
  (a) rows whose opponent ACTIVE has known = 0 with a non-zero top-1 prior id (the rows the old read asserted a
      guess as certain);
  (b) among (a): the OLD op output differs from the NEW one at all (the FULL op block, `type(op).forward`), and
      materially — any p_land of our status moves (op `MOVE_INFLICTS_STATUS` = 1 at the request slot) moving
      > 0.05 (`_status_landing`), or any KO/high/low cell of the outgoing per-move damage block moving > 0.05
      (`_outgoing_block`, cells [low, high, ·, ko] of each of our 4 request-order moves);
  (c) played status moves, resolved on the protocol exactly as the committed gate does
      (`agents/model/op_status_landing_bridge_integration_test.py`): OLD certain-zero claims (p_land 0, known 1)
      whose status LANDED, and the same for the NEW op;
  (d) the species / top-1 abilities driving (a)–(c);
  (e) each NEW status rule switched off on its own (everything else new): rows where the model outputs (pi / vf /
      pointer inputs / win-prob / α) or the op's flat block change; a REACH control zeroes the whole incoming
      status channel.

The extractor is FRESH INIT from `production_config.json` by default, or a TRAINED checkpoint (`--checkpoint`).
🚨 Read (b)'s model-output counts and (e) from a TRAINED extractor: the pair-outcome / conditional-threat /
intent cells that carry the op's incoming status channel deliver through ZERO-INIT projections, so at fresh init
they read exactly 0 and the channel looks unreachable (it is not). (a), (c) and the op-block / direct reads do not
depend on the weights.

The OLD read = THEIR six's ability `known` column set to `ability1_ids > 0` on the same observation (verified
equivalent to an actual revert of the op's two helpers on the test's 48 battles). Deterministic (a committed bank,
a deterministic replay, CPU, eager); a NOISE control re-runs the base op block and must differ on 0 rows.

    scripts/ops/mem_cap.sh 16 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \\
        designs/research_state/measurements/op_gigo_2026-10-07/measure.py [n_battles] [--checkpoint <zip|run dir>]
(from the worktree root with PYTHONPATH=<worktree>/src)
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import sys
import time
from typing import Any, Callable, Dict, Iterator, List

import torch

import agents.model.damage_op_blocks as dob
from agents import gen3_data
from agents.model import op_status_landing_bridge_integration_test as gate
from agents.model.damage_op_layout import _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE
from agents.model.damage_op_layout import _COND_FRZ_IDX, _COND_SLP_IDX, _SUBSTITUTE_CTX_IDX, _YAWN_CTX_IDX
from agents.model.extractor_ctx import POKEMON_ABILITY_KNOWN_OFFSET
from agents.model.move_resolution_tables import S_SG_OPP, S_SG_OURS
from agents.observation.constants import POKEMON_CONDITION_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, TEAM_SIZE

MATERIAL = 0.05
_YAWN_TABLES = ("MOVE_INFLICTS_STATUS", "MOVE_IS_SLEEP", "MOVE_BLOCKED_IF_STATUSED", "MOVE_STATUS_CAT",
                "MOVE_STATUS_IDENT", "MOVE_IS_YAWN", "MOVE_STATUS_TYPE_IMMUNE")


def _names() -> Dict[str, Dict[int, str]]:
    sp = {}
    for sid in gen3_data.species.base_form_ids():
        sd = gen3_data.species.get(sid)
        if sd is not None:
            sp[int(sd.num)] = sid
    ab = {}
    for aid in gen3_data.abilities.raw():
        ad = gen3_data.abilities.get(aid)
        if ad is not None:
            ab[int(ad.num)] = aid
    mv = {}
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        if md is not None:
            mv[int(md.num)] = mid
    return {"species": sp, "ability": ab, "move": mv}


def _with_ctx(ctx: Any, **cols: Any) -> Any:
    """A ctx copy with columns zeroed: ``screen=[idx]`` / ``opp_raw=[idx]`` / ``our_raw=[idx]``."""
    kw = {}
    for name, field in (("screen", "screen_feature"), ("opp_raw", "opp_ctx_raw"), ("our_raw", "our_ctx_raw")):
        if name in cols:
            t = getattr(ctx, field).clone()
            t[:, cols[name]] = 0.0
            kw[field] = t
    return dataclasses.replace(ctx, **kw)


@contextlib.contextmanager
def _instance_wrap(op: Any, method: str, ctx_fn: Callable[[Any], Any]) -> Iterator[None]:
    """Shadow ``op.<method>`` with one that sees ``ctx_fn(ctx)`` — every in-op caller goes through ``self.``."""
    orig = getattr(op, method)
    setattr(op, method, lambda ctx, *a, **k: orig(ctx_fn(ctx), *a, **k))
    try:
        yield
    finally:
        delattr(op, method)


@contextlib.contextmanager
def _old_ability(op: Any) -> Iterator[None]:
    """The pre-fix ability read on EVERY op path: `opp_ability_view` (the one reader) with known = ``id > 0``."""
    def view(ctx: Any) -> Any:
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        ids = ctx.ability1_ids[:, opp]
        return ids, (ids > 0).to(ctx.pokemon_part.dtype), ctx.species_ids[:, opp]
    op.opp_ability_view = view
    try:
        yield
    finally:
        del op.opp_ability_view


@contextlib.contextmanager
def _incoming_rule_off(rule: str) -> Iterator[None]:
    """Patch `damage_op_blocks.incoming_status_mask` (the op's one call site) with one input neutralised."""
    orig = dob.incoming_status_mask

    def patched(sg, slp, frz, alive, rest, sub, act):  # type: ignore[no-untyped-def]
        if rule == "our_safeguard":
            sg = torch.zeros_like(sg)
        elif rule == "incoming_sleep_clause":
            slp = torch.zeros_like(slp)          # `our_slp` feeds only the clause inside the mask
        elif rule == "freeze_clause":
            frz = torch.zeros_like(frz)          # `our_frz` feeds only the clause inside the mask
        elif rule == "our_substitute":
            sub = torch.zeros_like(sub)
        return orig(sg, slp, frz, alive, rest, sub, act)
    dob.incoming_status_mask = patched
    try:
        yield
    finally:
        dob.incoming_status_mask = orig


@contextlib.contextmanager
def _yawn_table_off(op: Any, yawn_num: int) -> Iterator[None]:
    """The pre-fix status tables at Yawn (no status category, not an inflictor) — in place, restored."""
    saved = {n: getattr(op, n)[yawn_num].clone() for n in _YAWN_TABLES}
    try:
        for n in _YAWN_TABLES:
            getattr(op, n)[yawn_num] = 0
        yield
    finally:
        for n, v in saved.items():
            getattr(op, n)[yawn_num] = v


@contextlib.contextmanager
def _incoming_status_zeroed(op: Any) -> Iterator[None]:
    """A REACH control: the op's whole INCOMING status channel (`_incoming_status_lands` and
    `_incoming_dedicated_land`, the two readers every incoming consumer goes through) returns zeros."""
    o1, o2 = op._incoming_status_lands, op._incoming_dedicated_land
    op._incoming_status_lands = lambda *a, **k: torch.zeros_like(o1(*a, **k))
    op._incoming_dedicated_land = lambda *a, **k: torch.zeros_like(o2(*a, **k))
    try:
        yield
    finally:
        del op._incoming_status_lands, op._incoming_dedicated_land


def variants(op: Any, yawn_num: int) -> Dict[str, Callable[[], Any]]:
    """Each NEW rule switched off on its own (everything else new), plus the OLD ability read."""
    return {
        "old_ability_read": lambda: _old_ability(op),
        "their_safeguard": lambda: _instance_wrap(op, "_outgoing_status_land",
                                                  lambda c: _with_ctx(c, screen=[S_SG_OPP])),
        "their_drowsy_yawn_fails": lambda: _instance_wrap(op, "_outgoing_status_land",
                                                          lambda c: _with_ctx(c, opp_raw=[_YAWN_CTX_IDX])),
        "our_drowsy_yawn_fails": lambda: _instance_wrap(op, "_incoming_status_mask",
                                                        lambda c: _with_ctx(c, our_raw=[_YAWN_CTX_IDX])),
        "our_safeguard": lambda: _incoming_rule_off("our_safeguard"),
        "incoming_sleep_clause": lambda: _incoming_rule_off("incoming_sleep_clause"),
        "freeze_clause": lambda: _incoming_rule_off("freeze_clause"),
        "our_substitute": lambda: _incoming_rule_off("our_substitute"),
        "yawn_as_sleep_inflictor": lambda: _yawn_table_off(op, yawn_num),
        "noise_control": lambda: contextlib.nullcontext(),
        "REACH_incoming_status_channel_zeroed": lambda: _incoming_status_zeroed(op),
    }


def _model_out(fe: Any, x: torch.Tensor, cap: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """One extractor forward: every per-row output the model's heads read (+ the op's flat block)."""
    pi, vf = fe({"observation": x})
    out = {"pi": pi, "vf": vf, "op_block": cap["block"]}
    pin = fe.last_pointer_inputs
    if pin is not None:
        for f in pin._fields:
            out[f"ptr_{f}"] = getattr(pin, f)
    for name in ("last_win_prob_logits", "last_alpha_logits"):
        t = getattr(fe, name, None)
        if isinstance(t, torch.Tensor):
            out[name] = t
    return {k: v.detach().clone() for k, v in out.items()}


def _row_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    B = a["pi"].shape[0]
    d = torch.zeros(B, dtype=torch.bool)
    for k in keys:
        d |= (a[k] != b[k]).reshape(B, -1).any(1)
    return d


def run(fe: Any, rows: Any, yawn_num: int, batch: int = 128) -> Dict[str, torch.Tensor]:
    op = fe.damage_op
    orig = op.forward
    cap: Dict[str, Any] = {}

    def hook(ctx, *args, **kwargs):  # type: ignore[no-untyped-def]
        block = orig(ctx, *args, **kwargs)
        cap["block"] = block
        if "ctx" not in cap:                 # the BASE forward's ctx + spread (the direct reads below)
            cap["ctx"] = ctx
            cap["spread"] = args[1] if len(args) > 1 else kwargs.get("spread_belief")
        return block
    op.forward = hook
    acc: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
    var = variants(op, yawn_num)
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                x = torch.as_tensor(rows[i:i + batch], dtype=torch.float32)
                cap.clear()
                base = _model_out(fe, x, cap)
                ctx, spread = cap["ctx"], cap["spread"]
                model_keys = [k for k in base if k != "op_block"]
                for name, cm in var.items():
                    with cm():
                        alt = _model_out(fe, x, cap)
                    acc[f"model_{name}"].append(_row_diff(base, alt, model_keys))
                    acc[f"block_{name}"].append(_row_diff(base, alt, ["op_block"]))
                # direct reads on the base ctx: the status channel + the outgoing damage block, new vs old
                ar = torch.arange(ctx.batch_size)
                old = gate.old_ability_ctx(ctx)
                sl_new, sl_old = op._status_landing(ctx), op._status_landing(old)
                inflicts = op.MOVE_INFLICTS_STATUS[ctx.our_active_req_move_ids]
                n_cells = _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE
                shape = (-1, _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE)
                ob_new = op._outgoing_block(ctx, spread)[:, :n_cells].reshape(shape)
                ob_old = op._outgoing_block(old, spread)[:, :n_cells].reshape(shape)
                opp_g = TEAM_SIZE + ctx.opp_active_local
                our_cond = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7]
                our_rest = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]
                our_alive = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
                got = {
                    "sl_new": sl_new, "sl_old": sl_old, "inflicts": inflicts,
                    "status_mat": (((sl_new[:, :4] - sl_old[:, :4]).abs() > MATERIAL) & (inflicts > 0.5)).any(1),
                    "known_flip": ((sl_new[:, 4:] != sl_old[:, 4:]) & (inflicts > 0.5)).any(1),
                    "dmg_mat": ((ob_new - ob_old)[:, :, [0, 1, 3]].abs() > MATERIAL).any(2).any(1),
                    "opp_known": ctx.pokemon_part[ar, opp_g, POKEMON_ABILITY_KNOWN_OFFSET],
                    "opp_ab": ctx.ability1_ids[ar, opp_g], "opp_sp": ctx.species_ids[ar, opp_g],
                    "has_opp": ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(1),
                    # their BENCH: a non-active slot with known = 0 and a non-zero top-1 prior id
                    "bench_unrev": ((ctx.pokemon_part[:, TEAM_SIZE:, POKEMON_ABILITY_KNOWN_OFFSET] < 0.5)
                                    & (ctx.ability1_ids[:, TEAM_SIZE:] > 0)
                                    & (torch.arange(TEAM_SIZE)[None, :] != ctx.opp_active_local[:, None])).any(1),
                    "cond_their_safeguard": ctx.screen_feature[:, S_SG_OPP] > 0.5,
                    "cond_our_safeguard": ctx.screen_feature[:, S_SG_OURS] > 0.5,
                    "cond_their_drowsy": ctx.opp_ctx_raw[:, _YAWN_CTX_IDX] > 0.5,
                    "cond_our_drowsy": ctx.our_ctx_raw[:, _YAWN_CTX_IDX] > 0.5,
                    "cond_our_frozen_any": our_cond[..., _COND_FRZ_IDX].sum(-1) > 0.5,
                    "cond_our_asleep_nonrest_alive": (our_cond[..., _COND_SLP_IDX] * our_alive
                                                      * (1.0 - our_rest)).sum(-1) > 0.5,
                    "cond_our_substitute": ctx.our_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5,
                }
                for k, v in got.items():
                    acc[k].append(v.detach())
                print(f"  batch {i // batch + 1}/{(len(rows) + batch - 1) // batch}", file=sys.stderr, flush=True)
    finally:
        op.forward = orig
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


def _top(c: collections.Counter, k: int = 12) -> List[List[Any]]:
    return [[key, n] for key, n in c.most_common(k)]


def trained_extractor(ref: str) -> Any:
    """A TRAINED extractor: ``ref`` (a ``.zip`` or a run dir → its last snapshot, `resolve_model_ref`) strict-loaded
    on CPU through `load_checkpoint_strict` — READ ONLY. Its kwargs are the checkpoint's own, not the mirror's."""
    from agents.model.snapshot import load_checkpoint_strict
    from agents.training.fixed_opponent_pool import resolve_model_ref
    zip_path = ref if ref.endswith(".zip") else resolve_model_ref(ref).zip_path
    fe = load_checkpoint_strict(zip_path, device="cpu").policy.features_extractor.eval()
    fe.checkpoint_zip = zip_path
    return fe


def main(n_battles: int, checkpoint: str = "") -> Dict[str, Any]:
    t0 = time.time()
    names = _names()
    yawn = gen3_data.moves.get("yawn")
    assert yawn is not None
    rows, meta = gate.collect(n_battles)
    t_collect = time.time() - t0
    fe = trained_extractor(checkpoint) if checkpoint else gate._extractor()
    torch.set_num_threads(4)
    r = run(fe, rows, int(yawn.num))
    t_ops = time.time() - t0 - t_collect
    N = len(meta)
    battles = len({d.label for d in meta})
    sp = lambda i: names["species"].get(int(r["opp_sp"][i]), f"#{int(r['opp_sp'][i])}")    # noqa: E731
    ab = lambda i: names["ability"].get(int(r["opp_ab"][i]), f"#{int(r['opp_ab'][i])}")    # noqa: E731

    a_mask = (r["opp_known"] < 0.5) & (r["opp_ab"] > 0) & r["has_opp"]
    na = int(a_mask.sum())
    a_drivers: collections.Counter = collections.Counter()
    b_any_drivers: collections.Counter = collections.Counter()
    b_mat_drivers: collections.Counter = collections.Counter()
    for i in torch.nonzero(a_mask).flatten().tolist():
        a_drivers[f"{sp(i)} [{ab(i)}]"] += 1
        if bool(r["model_old_ability_read"][i]):
            b_any_drivers[f"{sp(i)} [{ab(i)}]"] += 1
        if bool(r["status_mat"][i] | r["dmg_mat"][i]):
            b_mat_drivers[f"{sp(i)} [{ab(i)}]"] += 1
    with_status_move = a_mask & (r["inflicts"] > 0.5).any(1)

    # (c) played status moves on the protocol (the gate's resolution + exclusions)
    excl: collections.Counter = collections.Counter()
    played = landed = 0
    certain: Dict[str, List[int]] = {"new": [0, 0], "old": [0, 0]}
    contra: Dict[str, collections.Counter] = {"new": collections.Counter(), "old": collections.Counter()}
    contra_rows: Dict[str, List[str]] = {"new": [], "old": []}
    old_true_zero_moves: collections.Counter = collections.Counter()
    for i, d in enumerate(meta):
        if d.slot is None or d.chosen is None or d.chosen in gate._CALLERS:
            continue
        if float(r["inflicts"][i, d.slot]) < 0.5:
            continue
        o = gate.status_outcome(d.lines, d.me, d.chosen)
        key = "not_executed" if not o["executed"] else "switched" if o["switched"] else \
            "status_changed" if o["changed"] else None
        if key:
            excl[key] += 1
            continue
        played += 1
        landed += int(o["landed"])
        for which in ("new", "old"):
            sl = r[f"sl_{which}"]
            if float(sl[i, d.slot]) == 0.0 and float(sl[i, 4 + d.slot]) == 1.0:
                certain[which][0] += 1
                if o["landed"]:
                    certain[which][1] += 1
                    contra[which][f"{sp(i)} [{ab(i)}] {d.chosen}"] += 1
                    contra_rows[which].append(f"{d.label} {d.me} t{d.turn} {d.chosen} vs {sp(i)} -> {o['status']}")
                elif which == "old" and float(r["sl_new"][i, d.slot]) > 0.0:
                    old_true_zero_moves[f"{sp(i)} [{ab(i)}] {d.chosen}"] += 1

    rules = {k[6:]: {"model_outputs_change": int(r[k].sum()), "op_block_changes": int(r["block_" + k[6:]].sum())}
             for k in sorted(x for x in r if x.startswith("model_")) if k[6:] not in ("old_ability_read",)}
    conds = {k[5:]: int(r[k].sum()) for k in sorted(x for x in r if x.startswith("cond_"))}
    return {
        "extractor": (f"TRAINED {fe.checkpoint_zip}" if checkpoint
                      else "FRESH INIT (torch.manual_seed(0)), production_config.json kwargs"),
        "n_battles": battles, "n_rows": N, "wall_s": {"collect": round(t_collect, 1), "op_reads": round(t_ops, 1)},
        "a_unrevealed_prior_active": {"rows": na, "frac": round(na / max(N, 1), 4),
                                      "of_which_our_active_has_status_move": int(with_status_move.sum())},
        "b_among_a": {"old_differs_model_outputs": int((r["model_old_ability_read"] & a_mask).sum()),
                      "old_differs_op_block": int((r["block_old_ability_read"] & a_mask).sum()),
                      "material_any": int(((r["status_mat"] | r["dmg_mat"]) & a_mask).sum()),
                      "material_status_p_land": int((r["status_mat"] & a_mask).sum()),
                      "material_damage_cell": int((r["dmg_mat"] & a_mask).sum()),
                      "status_known_bit_flipped": int((r["known_flip"] & a_mask).sum())},
        "b_outside_a_old_differs_model_outputs": int((r["model_old_ability_read"] & ~a_mask).sum()),
        "bench_unrevealed_prior_rows": {"rows": int(r["bench_unrev"].sum()),
                                        "outside_a": int((r["bench_unrev"] & ~a_mask).sum())},
        "c_played_status_moves": {"played_resolved": played, "landed": landed, "excluded": dict(excl),
                                  "old_certain_zero": certain["old"][0], "old_contradicted": certain["old"][1],
                                  "new_certain_zero": certain["new"][0], "new_contradicted": certain["new"][1]},
        "d_drivers": {"a_rows": _top(a_drivers), "b_any_rows": _top(b_any_drivers),
                      "b_material_rows": _top(b_mat_drivers), "c_old_contradicted": _top(contra["old"]),
                      "c_old_certain_zero_new_says_possible_not_landed": _top(old_true_zero_moves)},
        "c_old_contradicted_rows": contra_rows["old"], "c_new_contradicted_rows": contra_rows["new"],
        "e_rule_off_rows_changed (noise_control = base vs base)": rules, "e_rule_condition_rows": conds,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n_battles", nargs="?", type=int, default=10**9)
    ap.add_argument("--checkpoint", default="", help="a .zip or a run dir (→ its last snapshot); default: fresh init")
    a = ap.parse_args()
    print(json.dumps(main(a.n_battles, a.checkpoint), indent=1), flush=True)
