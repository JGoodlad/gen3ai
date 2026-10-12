"""A2 FIELD DIFF AT THE ROOT (the decisive GIGO audit of the honest worlds; no outcome is read).

For every state with an ok det playout row and each of its first 8 (PLAYED) worlds, in det/ (Smogon worlds) and
det_b/ (belief-species worlds): replay the world prefix and the TRUE prefix through the pin's ``core_events``
(``run_core(..., trackers=True, obs=True)``) and compare, cell by cell, the OPPONENT viewer's own observation row at
its LAST obs-bearing decision in the prefix (the prefix stops at the searched side's open root request, so the
opponent can have no later decision; its last one is the root-turn decision when its root choice is open, else its
latest earlier one: the `after` gap to the root is recorded). Every differing cell is NAMED through the pin's
observation layout (``audit_lib.cell_names``: ``Gen3ObservationEncoder.get_layout()`` + ``constants``) and
classified by (block, field, the opponent team slot, whether that mon is SEEN by the searched side at the root):

  EXPECTED   a never-seen mon's cells (any); a seen mon's RE-DRAWN move slot cells (and the request-move cells of its
             active slot, and its Hidden Power type block only when the world and truth differ in holding a HP move)
  SEPARATE   the opponent's own last-action cells (counted apart; flagged when the state rewrote a past choice)
  UNEXPECTED everything else: a seen mon's exact HP, status / counters, boosts, volatiles, item, ability, spread,
             species, recency..., a kept move slot, the field / weather / hazards / screens / clock, the opponent's
             view of OUR side, the event window, the obs facts

Also compared (a stronger read of the hidden state than the row): the opponent's LAST ``|request|`` line in the
prefix, per mon by species: a SEEN mon's exact ``condition`` (HP / max HP + status), ``stats``, ``item``,
``baseAbility``, kept move ids and kept-slot PP (active). And the gate: the searched side's rows at every decision
up to the root byte-equal.

    run_at_pin.sh <cp> 1 8 2h audit_a2_field_diff.py [--every 1] [--limit N]
→ archive ``hidden_info_binding/<cp>/audit/a2_rows_<dir>.jsonl``, here ``audit_a2_field_diff_<cp>.json``.
"""
import argparse
import collections
import difflib
import json
import time

import numpy as np

import audit_lib as A

C, D = A.C, A.D
J = 8
DIRS = ("det", "det_b")
MOVE_NUM = {}
SLOT_SUBS = ["id", "power", "secondary", "recoil", "type", "category", "known", "current_pp", "max_pp", "accuracy",
             "never_miss"]
#: the pin's layout constants (never a literal index): filled by load_layout() from agents.observation.constants
K = {}


def load_layout():
    from agents.observation import constants as c
    K.update(FULL=int(c.POKEMON_FULL_DIM), OWN=int(c.OFFSET_OUR_TEAM), MOVES=int(c.POKEMON_MOVES_OFFSET),
             SLOT=int(c.MOVE_SLOT_DIM), ACTIVE=int(c.POKEMON_ACTIVE_OFFSET), SPECIES=int(c.POKEMON_SPECIES_OFFSET),
             FSEEN=int(c.OFFSET_OBS_FACTS) + int(c.FACTS_SEEN_OFFSET), FROW=int(c.FACTS_SEEN_ROW_DIM),
             PER=int(c.ACTIVE_REQ_MOVES_PER), TEAM=int(c.TEAM_SIZE), EW=int(c.OFFSET_EVENT_WINDOW),
             ETOK=int(c.EVENT_TOKEN_DIM), ECOL={int(m): m.name for m in c.EventCol},
             ESPECIES={int(c.EventCol.ACTOR_SPECIES), int(c.EventCol.TARGET_SPECIES), int(c.EventCol.REL_SPECIES)},
             EMOVE=int(c.EventCol.MOVE))


def gender_draws(team, P):
    """The turn-0 construction GENDER draws a packed team costs (rust_sim turn/driver.rs `draw_turn0_genders`: one
    PRNG draw per mon whose set leaves the gender empty AND whose species has no fixed gender)."""
    n = 0
    for e in team.split("]"):
        f = D.parse_set(e)
        if not f[7] and P.species.get(D.toid(f[1] or f[0]), {}).get("gender") is None:
            n += 1
    return n


def strip_gender(ln):
    f = ln.split("|")
    if len(f) > 3 and f[1] in ("switch", "drag", "replace", "detailschange"):
        f[3] = ", ".join(x for x in f[3].split(", ") if x not in ("M", "F"))
    return "|".join(f)


def protocol_diff_kinds(lt, lw):
    """The kinds of the lines that differ between two protocol streams: a switch / drag line differing ONLY in the
    gender token -> 'gender_in_switch_details'; any other line -> its protocol keyword."""
    dl = [x for x in difflib.unified_diff(lt, lw, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("---", "+++")]
    if not dl:
        return [], []
    gm = collections.Counter(strip_gender(x[1:]) for x in dl if x[0] == "-")
    gp = collections.Counter(strip_gender(x[1:]) for x in dl if x[0] == "+")
    kinds = []
    for x in dl:
        k = strip_gender(x[1:])
        if gm[k] and gp[k] and strip_gender(x[1:]) != x[1:]:
            kinds.append("gender_in_switch_details")
        else:
            kinds.append((x[1:].split("|") + ["", ""])[1] or "blank")
    return sorted(set(kinds)), dl


def opp_lines(res, o):
    return [ln for s, lines in res["chunks"] if s == o for ln in lines if not ln.startswith("|request|")]


def last_request(res, side_idx):
    req = [ln for s, lines in res["chunks"] if s == side_idx for ln in lines if ln.startswith("|request|")]
    return json.loads(req[-1][len("|request|"):]) if req else None


def set_info(team):
    sets = [D.parse_set(e) for e in team.split("]")]
    sids = [D.toid(f[1] or f[0]) for f in sets]
    moves = [f[4].split(",") for f in sets]
    return sids, moves


def kept_slots(moves, seen):
    return {k for k, m in enumerate(moves) if m in seen or (m.startswith("hiddenpower") and "hiddenpower" in seen)}


def has_hp(moves):
    return any(m.startswith("hiddenpower") for m in moves)


def compare_state(names, P, b, d, wr, det_dir):
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.truth import cmd_index

    side = d["side"]
    opp = "p2" if side == "p1" else "p1"
    v = 0 if side == "p1" else 1
    o = 1 - v
    at = cmd_index(b, side, d["n"])
    true_team = getattr(b, opp)["team"]
    t_sids, t_moves = set_info(true_team)
    revealed = {k: set(x) for k, x in wr["revealed"].items()}
    seen_slot = [s in revealed for s in t_sids]
    rewrites = wr["leak"]["rewritten_moves"] + wr["leak"]["rewritten_switches"]
    battles = [A.rec(b, "true", opp, None, b.commands[:at])]
    battles += [A.rec(b, f"w{j}", opp, wr["teams"][j], wr["cmds"][j]) for j in range(J)]
    res = run_core(battles, trackers=True, obs=True)
    rt = res[0]
    if not rt["ok"]:
        raise RuntimeError(f"true prefix refused: {rt.get('error')}")
    sv_t = A.obs_entries(rt, v)
    ov_t = A.obs_entries(rt, o)
    root_after = int(sv_t[d["n"]]["after"])
    row_t = A.row_of(ov_t[-1])
    req_t = last_request(rt, o)
    num = lambda s: int(P.species.get(s, {}).get("num") or -1)  # noqa: E731
    full, own0 = K["FULL"], K["OWN"]
    order_t = [int(row_t[own0 + r * full + K["SPECIES"]]) for r in range(K["TEAM"])]
    out = {"id": d["id"], "dir": det_dir, "turn": d["turn"], "n_seen": sum(seen_slot), "rewrites": rewrites,
           "opp_after_gap": root_after - int(ov_t[-1]["after"]), "opp_n_dec": len(ov_t), "worlds": []}
    row_order_ok_t = order_t == [num(s) for s in t_sids]
    active_t = [r for r in range(6) if row_t[own0 + r * full + K["ACTIVE"]] > 0.5]
    gd_t = gender_draws(true_team, P)
    lines_t = opp_lines(rt, o)
    slines_t = opp_lines(rt, v)
    out["gender_draws_true"] = gd_t
    for j in range(J):
        rw = res[1 + j]
        w = {"j": j, "cats": collections.Counter(), "unexpected": [], "notes": []}
        out["worlds"].append(w)
        if not rw["ok"]:
            w["notes"].append(f"world prefix refused: {str(rw.get('error'))[:200]}")
            w["cats"]["UNEXPECTED|replay|refused|-"] += 1
            continue
        w_sids, w_moves = set_info(wr["teams"][j])
        gd_same = gender_draws(wr["teams"][j], P) == gd_t
        w["gender_draws_equal"] = gd_same
        lw_ = opp_lines(rw, o)
        dl = [x for x in difflib.unified_diff(lines_t, lw_, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("---", "+++")]
        w["opp_protocol_differs"] = bool(dl)
        if dl:
            w["cats"][f"HIDDEN|opp_protocol_differs|gender_draws_{'equal' if gd_same else 'DIFFER'}"] += 1
            w["notes"].append("opp protocol: " + " ~ ".join(dl[:4])[:300])
        # THE OBSERVATION ITSELF: the searched side's protocol (its own requests excluded) world vs truth
        kinds, sdl = protocol_diff_kinds(slines_t, opp_lines(rw, v))
        w["searched_protocol_kinds"] = kinds
        for k in kinds:
            w["cats"][f"OBS_INCONSISTENT|searched_protocol|{k}"] += 1
        if sdl:
            w["notes"].insert(0, "SEARCHED protocol: " + " ~ ".join(sdl[:4])[:300])
        n_t = {num(t_sids[t]) for t in range(6) if not seen_slot[t]}
        n_w = {num(w_sids[t]) for t in range(6) if not seen_slot[t]}
        kept_all = {MOVE_NUM.get(t_moves[t][k], -1) for t in range(6) if seen_slot[t]
                    for k in kept_slots(t_moves[t], revealed[t_sids[t]])}
        st_diff = any(seen_slot[t] and (("sleeptalk" in t_moves[t]) != ("sleeptalk" in w_moves[t])) for t in range(6))
        sv_w, ov_w = A.obs_entries(rw, v), A.obs_entries(rw, o)
        gate = len(sv_w) > d["n"] and all(sv_w[k]["obs"]["b64"] == sv_t[k]["obs"]["b64"] for k in range(d["n"] + 1))
        if not gate:
            w["cats"]["UNEXPECTED|gate|searched_side_rows_differ|-"] += 1
        if len(ov_w) != len(ov_t):
            w["cats"]["UNEXPECTED|opp_decisions|count_differs|-"] += 1
            w["notes"].append(f"opp decisions {len(ov_w)} vs {len(ov_t)}")
            continue
        row_w = A.row_of(ov_w[-1])
        order_w = [int(row_w[own0 + r * full + K["SPECIES"]]) for r in range(K["TEAM"])]
        if not (row_order_ok_t and order_w == [num(s) for s in w_sids]):
            w["cats"]["UNEXPECTED|row_order|team_order_not_row_order|-"] += 1
            w["notes"].append(f"row order t={order_t} w={order_w}")
            continue
        # seen mons must be the SAME species in the same slot
        for t in range(6):
            if seen_slot[t] and w_sids[t] != t_sids[t]:
                w["cats"]["UNEXPECTED|world|seen_species_changed|-"] += 1
        redrawn = [set(range(len(t_moves[t]))) - kept_slots(t_moves[t], revealed[t_sids[t]]) if seen_slot[t] else None
                   for t in range(6)]
        # each KEPT (seen) move of a seen mon: its 11 cells in the world's row (found by move id) == truth's
        for t in range(6):
            if not seen_slot[t]:
                continue
            kept_nums = {MOVE_NUM.get(t_moves[t][k], -1) for k in kept_slots(t_moves[t], revealed[t_sids[t]])}
            bt_ = bw_ = own0 + t * full + K["MOVES"]
            S = K["SLOT"]
            slots_t = [row_t[bt_ + S * s: bt_ + S * s + S] for s in range(4)]
            slots_w = [row_w[bw_ + S * s: bw_ + S * s + S] for s in range(4)]
            for st in slots_t:
                if int(st[0]) not in kept_nums:
                    continue
                m = [sw for sw in slots_w if int(sw[0]) == int(st[0])]
                if not m:
                    w["cats"]["UNEXPECTED|own_team|kept_move_missing|seen"] += 1
                    continue
                w["cats"]["CHECKED|own_team|kept_move_content|seen"] += 1
                for q in np.flatnonzero(m[0].view(np.uint32) != st.view(np.uint32)):
                    sub = SLOT_SUBS[int(q)]
                    cat = f"UNEXPECTED|own_team|kept_move_content.{sub}|seen"
                    w["cats"][cat] += 1
                    if len(w["unexpected"]) < 6:
                        w["unexpected"].append({"name": f"own_team[{t}].move(id {int(st[0])}).{sub}", "true": float(st[q]),
                                                "world": float(m[0][q]), "species": t_sids[t], "cat": cat})
        diff = np.flatnonzero(row_w.view(np.uint32) != row_t.view(np.uint32))
        for i in diff:
            nm = names[int(i)]
            blk, fld, sub = nm["block"], nm["field"], nm["sub"]
            if blk == "own_team":
                t = nm["slot"]
                if not seen_slot[t]:
                    cat = f"EXPECTED|own_team|{fld}|never_seen"
                elif fld.startswith("move"):
                    # the row's 4 move slots are the moveset SORTED BY MOVE ID (encoder/slot.rs `moves`), so a
                    # re-drawn move re-sorts the kept ones: positional diffs are expected when the mon has a
                    # re-drawn slot; each KEPT move's 11 cells are compared by move id below (kept_move_content)
                    cat = (f"EXPECTED|own_team|move_block_resorted.{sub}|seen_with_redrawn" if redrawn[t]
                           else f"UNEXPECTED|own_team|move.{sub}|seen_no_redrawn_slot")
                elif fld == "hp_block" and has_hp(t_moves[t]) != has_hp(w_moves[t]):
                    cat = f"EXPECTED|own_team|hp_block.{sub}|seen_hp_move_presence_differs"
                elif fld == "last_action":
                    cat = f"SEPARATE|own_team|last_action|seen|rewrites={'yes' if rewrites else 'no'}"
                else:
                    cat = f"UNEXPECTED|own_team|{fld}.{sub}|seen"
            elif blk == "reactive" and fld == "active_req_moves":
                k = int(sub) % K["PER"]
                a = active_t[0] if len(active_t) == 1 else None
                ok = a is not None and seen_slot[a] and k in redrawn[a]
                cat = ("EXPECTED|reactive|active_req_moves|seen_redrawn_slot" if ok
                       else "UNEXPECTED|reactive|active_req_moves|kept_slot")
            elif blk == "obs_facts" and fld == "seen":
                # row r = the opponent's own team slot r: [revealed, move-seen bit x4 (positional), item_public,
                # ability_public] = what the searched side has seen of it
                q = int(sub)
                fr = K["FROW"]
                r_, c_ = q // fr, q % fr
                if not seen_slot[r_]:
                    cat = "EXPECTED|obs_facts|seen|own_never_seen"
                elif 1 <= c_ <= 4:
                    bt = sum(row_t[K["FSEEN"] + r_ * fr + 1: K["FSEEN"] + r_ * fr + 5] > 0.5)
                    bw = sum(row_w[K["FSEEN"] + r_ * fr + 1: K["FSEEN"] + r_ * fr + 5] > 0.5)
                    cat = ("EXPECTED|obs_facts|seen_move_bits_positional|seen_same_count" if (bt == bw and redrawn[r_])
                           else "UNEXPECTED|obs_facts|seen_move_bits|seen_count_differs")
                else:
                    cat = f"UNEXPECTED|obs_facts|seen.{['revealed', '', '', '', '', 'item_public', 'ability_public'][c_]}|seen"
            elif blk == "event_window":
                q = (int(i) - K["EW"]) % K["ETOK"]
                col = K["ECOL"].get(q, str(q))
                tv, wv = int(row_t[i]), int(row_w[i])
                rw_ = 'yes' if rewrites else 'no'
                if q in K["ESPECIES"] and (tv in n_t or wv in n_w):
                    cat = f"SEPARATE|event_window|{col}|refers_to_never_seen_mon|rewrites={rw_}"
                elif q == K["EMOVE"] and tv not in kept_all:
                    cat = f"SEPARATE|event_window|{col}|refers_to_redrawn_move|rewrites={rw_}"
                else:
                    cat = f"UNEXPECTED|event_window|{col}|gender_draws_{'equal' if gd_same else 'DIFFER'}"
            elif blk == "reactive" and fld == "turns_since_progress" and st_diff:
                # trackers/clock.rs: the rest-loop freeze reads the ACTIVE's own moveset (no Sleep Talk) -> move-derived
                cat = "EXPLAINED|reactive|turns_since_progress|seen_mon_sleeptalk_presence_differs"
            elif blk == "pair_history":
                jrow = int(fld[len("h[i,"):-1])
                cat = ("EXPECTED|pair_history|h|own_never_seen" if not seen_slot[jrow]
                       else "UNEXPECTED|pair_history|h|own_seen")
            else:
                cat = f"UNEXPECTED|{blk}|{fld}|gender_draws_{'equal' if gd_same else 'DIFFER'}"
            w["cats"][cat] += 1
            if cat.startswith(("UNEXPECTED", "SEPARATE", "EXPLAINED")) and len(w["unexpected"]) < 6:
                sl = nm["slot"]
                w["unexpected"].append({"cell": int(i), "name": f"{blk}[{sl}].{fld}.{sub}" if sl is not None else f"{blk}.{fld}.{sub}",
                                        "true": float(row_t[i]), "world": float(row_w[i]),
                                        "species": t_sids[sl] if (blk == "own_team" and sl is not None) else None,
                                        "cat": cat})
        # the opponent's own last request: the exact hidden state of its SEEN mons
        req_w = last_request(rw, o)
        if req_t is None or req_w is None:
            w["cats"]["NOTE|request|missing|-"] += 1
        else:
            by_t = {D.toid(p["details"].split(",")[0]): p for p in req_t["side"]["pokemon"]}
            by_w = {D.toid(p["details"].split(",")[0]): p for p in req_w["side"]["pokemon"]}
            act_t = (req_t.get("active") or [{}])[0].get("moves") if req_t.get("active") else None
            act_w = (req_w.get("active") or [{}])[0].get("moves") if req_w.get("active") else None
            for t in range(6):
                if not seen_slot[t]:
                    continue
                pt, pw = by_t.get(t_sids[t]), by_w.get(w_sids[t])
                if pt is None or pw is None:
                    w["cats"]["UNEXPECTED|request|mon_missing|seen"] += 1
                    continue
                for key in ("condition", "stats", "item", "baseAbility", "active"):
                    if pt.get(key) != pw.get(key):
                        cat = f"UNEXPECTED|request|{key}|seen"
                        w["cats"][cat] += 1
                        if len(w["unexpected"]) < 10:
                            w["unexpected"].append({"name": f"request.{t_sids[t]}.{key}", "true": pt.get(key),
                                                    "world": pw.get(key), "cat": cat})
                for k in range(len(t_moves[t])):
                    if k not in redrawn[t] and (pt["moves"][k] if k < len(pt["moves"]) else None) != (
                            pw["moves"][k] if k < len(pw["moves"]) else None):
                        w["cats"]["UNEXPECTED|request|kept_move_id|seen"] += 1
                if pt.get("active") and pw.get("active") and act_t and act_w:
                    for k, (mt, mw) in enumerate(zip(act_t, act_w)):
                        if k in redrawn[t]:
                            continue
                        for key in ("id", "pp", "maxpp", "disabled"):
                            if mt.get(key) != mw.get(key):
                                cat = f"UNEXPECTED|request|active_kept_move_{key}|seen"
                                w["cats"][cat] += 1
                                if len(w["unexpected"]) < 10:
                                    w["unexpected"].append({"name": f"request.active.{t_sids[t]}.move{k}.{key}",
                                                            "true": mt.get(key), "world": mw.get(key), "cat": cat})
                w["cats"]["CHECKED|request|seen_mon|-"] += 1
        w["cats"] = dict(w["cats"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    c = A.cp()
    names = A.cell_names()
    load_layout()
    import os
    from pathlib import Path
    MOVE_NUM.update({k: int(v["num"]) for k, v in json.loads((Path(os.environ["VDP_PIN"]) / "data" / "pokemon" / "gen3_moves.json").read_text()).items()})
    assert [names[K["OWN"] + K["MOVES"] + q]["sub"] for q in range(K["SLOT"])] == SLOT_SUBS, "slot layout drift"
    bk = A.bank(c)
    P = A.priors()
    for det_dir in DIRS:
        W = A.world_rows(c, det_dir)
        ids = sorted(A.played_ids(c, det_dir))[:: a.every][: a.limit]
        path = A.audit_dir(c) / f"a2_rows_{det_dir}.jsonl"
        done = {r["id"] for r in C.read_rows(path)}
        todo = [x for x in ids if x not in done]
        print(f"[a2] {c} {det_dir}: {len(ids)} played states, {len(todo)} to go", flush=True)
        t0 = time.monotonic()
        with open(path, "a") as f:
            for i, did in enumerate(todo):
                d, b = A.decision(bk, did)
                try:
                    row = compare_state(names, P, b, d, W[did], det_dir)
                    row["ok"] = True
                except Exception as exc:
                    row = {"id": did, "dir": det_dir, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                C.write_row(f, row)
                if (i + 1) % 50 == 0 or i + 1 == len(todo):
                    print(f"[a2] {det_dir} {i + 1}/{len(todo)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    summarize(c, a)


def summarize(c, a):
    out = {"audit": "A2 field diff at the root (opponent's own row + its last request)", "cp": c,
           "every": a.every, "compared": "the opponent viewer's LAST obs-bearing decision in the prefix (cut at the "
           "searched side's open root request); opp_after_gap = root chunk index - that decision's chunk index",
           "by_dir": {}}
    tot_unexp = 0
    for det_dir in DIRS:
        rows = C.read_rows(A.audit_dir(c) / f"a2_rows_{det_dir}.jsonl")
        ok = [r for r in rows if r.get("ok")]
        cats = collections.Counter()
        ex = collections.defaultdict(list)
        gap = collections.Counter()
        worlds = 0
        states_unexp = set()
        gstat = collections.Counter()
        for r in ok:
            gap[min(r["opp_after_gap"], 9)] += 1
            for w in r["worlds"]:
                worlds += 1
                if "gender_draws_equal" in w:
                    gstat[("gender_draws_equal" if w["gender_draws_equal"] else "gender_draws_DIFFER",
                           "opp_protocol_differs" if w["opp_protocol_differs"] else "opp_protocol_same")] += 1
                for k, n in w["cats"].items():
                    cats[k] += n
                    if k.startswith("UNEXPECTED"):
                        states_unexp.add(r["id"])
                for u in w["unexpected"]:
                    key = u["cat"]
                    if len(ex[key]) < 10:
                        ex[key].append({"id": r["id"], "world": w["j"], "turn": r["turn"], **u, "notes": w["notes"][:2]})
        unexp = {k: n for k, n in sorted(cats.items()) if k.startswith("UNEXPECTED")}
        tot_unexp += sum(unexp.values())
        out["by_dir"][det_dir] = {
            "states": len(rows), "states_ok": len(ok), "errors": [r for r in rows if not r.get("ok")][:10],
            "worlds_compared": worlds,
            "opp_after_gap_hist": {str(k): v for k, v in sorted(gap.items())},
            "unexpected_cells_by_category": unexp,
            "unexpected_total": sum(unexp.values()), "states_with_unexpected": len(states_unexp),
            "separate_intent_cells": {k: n for k, n in sorted(cats.items()) if k.startswith("SEPARATE")},
            "explained_move_derived": {k: n for k, n in sorted(cats.items()) if k.startswith("EXPLAINED")},
            "hidden_divergence_worlds": {k: n for k, n in sorted(cats.items()) if k.startswith("HIDDEN")},
            "worlds_whose_SEARCHED_side_protocol_differs_by_kind": {k: n for k, n in sorted(cats.items()) if k.startswith("OBS_INCONSISTENT")},
            "worlds_whose_SEARCHED_side_protocol_differs": sum(bool(w.get("searched_protocol_kinds")) for r in ok for w in r["worlds"]),
            "states_with_any_such_world": sum(any(w.get("searched_protocol_kinds") for w in r["worlds"]) for r in ok),
            "examples_searched_protocol": [{"id": r["id"], "world": w["j"], "kinds": w["searched_protocol_kinds"], "note": w["notes"][0]}
                                            for r in ok for w in r["worlds"] if w.get("searched_protocol_kinds")][:12],
            "worlds_by_gender_draws_x_opp_protocol": {" & ".join(k): n for k, n in sorted(gstat.items())},
            "expected_cells_by_category": {k: n for k, n in sorted(cats.items()) if k.startswith("EXPECTED")},
            "checks": {k: n for k, n in sorted(cats.items()) if k.startswith(("CHECKED", "NOTE"))},
            "examples": {k: v for k, v in ex.items()},
        }
    out["unexpected_total"] = tot_unexp
    out["pass"] = tot_unexp == 0 and all(v["states"] == v["states_ok"] for v in out["by_dir"].values())
    A.write_json(f"audit_a2_field_diff_{c}.json", out)
    print(json.dumps({d: {k: v[k] for k in ("states_ok", "worlds_compared", "unexpected_total", "unexpected_cells_by_category",
                                            "separate_intent_cells", "explained_move_derived", "worlds_by_gender_draws_x_opp_protocol", "worlds_whose_SEARCHED_side_protocol_differs_by_kind", "worlds_whose_SEARCHED_side_protocol_differs", "states_with_any_such_world")} for d, v in out["by_dir"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
