"""TEETH of A1-A3 (design doc §5: each check re-run once on a PLANTED defect must FAIL). No outcome is scored: Phase 1's
outcomes are only compared for equality, as in A1.

  A1-a  the builder's command template renders every rewritten move slot OFF BY ONE (``determinize.build_worlds``
        re-compiled in THIS process with ``[t[2]]`` -> ``[(t[2] + 1) % 4]``), every world = the true team; states with
        >= 1 rewritten move. Caught = the A1 checks (gate on try 1, both viewers' rows + protocol vs the true prefix,
        ``one_state`` outcomes vs Phase 1 on seeds 0..7) report >= 1 mismatch.
  A1-b  every world = the true team with ONE kept SEEN mon's item swapped (leftovers <-> choiceband). Same A1 checks
        (the trivial "team != true team" flag is NOT counted).
  A2-item / A2-evs  the archived played worlds (det/) with ONE seen mon's item (resp. EVs) swapped, through A2's
        ``compare_state``: caught per world = an UNEXPECTED own_team ``items`` (resp. ``spread``) cell or an UNEXPECTED
        request ``item`` (resp. ``stats``) field.
  A3    one archived world's rendered prefix with ONE opponent command perturbed (its first ``move m`` -> the next
        move of the same set): caught = the direct re-gate rows differ from the truth (or the core refuses).

    run_at_pin.sh <cp> 2 8 2h audit_teeth.py [--n 10]
→ here ``audit_teeth_<cp>.json``.
"""
import argparse
import collections
import inspect
import json

import audit_lib as A

C, D = A.C, A.D


def swap_item(entry):
    f = D.parse_set(entry)
    # a PASSIVE item (no protocol message, no choice lock) so the world stays as close to the truth as possible
    f[2] = "metalcoat" if D.toid(f[2]) == "dragonfang" else "dragonfang"
    return "|".join(f)


def swap_evs(entry):
    f = D.parse_set(entry)
    alt = "252,252,,,4," if f[6] != "252,252,,,4," else ",,4,252,,252"
    f[6] = alt
    return "|".join(f)


def replace_slot(team, t, fn):
    e = team.split("]")
    e[t] = fn(e[t])
    return "]".join(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only", choices=("a1a",), default=None, help="re-run one plant and merge into the JSON")
    a = ap.parse_args()
    c = A.cp()
    import audit_a2_field_diff as A2
    import run_det_playouts as R
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bk = A.bank(c)
    P = A.priors()
    p1 = A.phase1_rows(c)
    a1rows = [r for r in C.read_rows(A.audit_dir(c) / "a1_rows.jsonl") if r.get("ok")]
    ids_rw = [r["id"] for r in a1rows if r["leak"]["rewritten_moves"] > 0][: a.n]
    ids_any = [r["id"] for r in a1rows][: a.n]
    out = {"audit": "TEETH of A1-A3 (planted defects must be caught)", "cp": c}
    WDET = A.world_rows(c, A.AUDIT_DIRS[0])

    # the planted builder: the SAME source, the rewritten-slot render off by one (both expose the command template so
    # A1's TEXT invariant, ``audit_lib.text_invariant``, runs as part of the A1 checks)
    ns_bad = A.builder_ns(plant_off_by_one=True)
    ns_ok = A.builder_ns()

    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(R.J)]
    off = C.clock_offset()

    def a1_check(did, builder_ns, team_fn):
        d, b = A.decision(bk, did)
        side = d["side"]
        opp = "p2" if side == "p1" else "p1"
        at = cmd_index(b, side, d["n"])
        true_team = getattr(b, opp)["team"]
        sids = [D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]) for e in true_team.split("]")]
        world_team = team_fn(true_team, sids, d, b, at)
        if world_team is None:
            return None
        saved = builder_ns["sample_world"]
        builder_ns["sample_world"] = lambda rng, P_, team, rv, *x, **k: (world_team, {"new_slots": 0, "redrawn_move_slots": 0})
        try:
            w = builder_ns["build_worlds"](b, side, at, d["n"], did, P, R.J, root_choice_in_log=not p1[did]["other_open"])
        finally:
            builder_ns["sample_world"] = saved
        rep = {"id": did, "gate_try1": all(t is not None for t in w["teams"]) and all(x == 1 for x in w["tries"])}
        if w["teams"][0] is not None:
            ti = A.text_invariant(w, b.commands[:at])
            rep["text_invariant_ok"] = ti["ok"]
            rep["text_invariant_mismatches"] = ti.get("n_mismatch")
            rep["text_invariant_example"] = (ti.get("mismatches") or [None])[0]
        if not all(t is not None for t in w["teams"]):
            rep["caught_by"] = "gate (no world passed in 30 tries)"
            rep["caught"] = True
            return rep
        cm = w["cmds"][0]
        rep["cmd_text_diffs"] = sum(x != list(y) for x, y in zip(cm, b.commands[:at]))
        rt, rr = run_core([A.rec(b, "t", opp, None, b.commands[:at]), A.rec(b, "r", opp, world_team, cm)], trackers=True, obs=True)
        rep["rows_equal_both_viewers"] = bool(rr["ok"]) and all(
            [x["obs"]["b64"] for x in A.obs_entries(rt, v)] == [x["obs"]["b64"] for x in A.obs_entries(rr, v)] for v in (0, 1))
        rep["chunks_equal"] = bool(rr["ok"]) and rt["chunks"] == rr["chunks"]
        r = R.one_state(cores, fwd, bk, did, {"teams": [world_team] * R.J, "cmds": [cm] * R.J}, off)
        if not r.get("ok"):
            rep["outcome_mismatch"] = f"one_state refused: {r.get('error')}"
        else:
            rep["outcome_mismatch"] = sum(s != p1[did]["out"].get(act, "")[: R.J] for act, s in r["out"].items())
        rep["caught_without_text_invariant"] = ((not rep["gate_try1"]) or (not rep["rows_equal_both_viewers"])
                                                or (not rep["chunks_equal"]) or bool(rep["outcome_mismatch"]))
        rep["caught"] = rep["caught_without_text_invariant"] or not rep.get("text_invariant_ok", True)
        return rep

    def true_team_fn(true_team, sids, d, b, at):
        return true_team

    def item_swapped_fn(true_team, sids, d, b, at):
        rv = WDET.get(d["id"], {}).get("revealed") or {}
        seen = [t for t, s in enumerate(sids) if s in rv]
        return replace_slot(true_team, seen[0], swap_item) if seen else None

    with inference_globals(C.threads()):
        fwd = R.Fwd(C.load_model(c))
        plants = (("A1a_off_by_one_rewritten_slot", ids_rw, ns_bad, true_team_fn),
                  ("A1b_seen_mon_item_swapped", ids_any, ns_ok, item_swapped_fn))
        for key, ids, ns, fn in plants:
            if a.only == "a1a" and key != "A1a_off_by_one_rewritten_slot":
                continue
            reps = []
            for did in ids:
                try:
                    rp = a1_check(did, ns, fn)
                except Exception as exc:
                    rp = {"id": did, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "caught": None}
                    for k, core in enumerate(cores):
                        core.close()
                        cores[k] = SearchCore(lib=lib)
                if rp is not None:
                    reps.append(rp)
                    print(f"[teeth {key}]", rp, flush=True)
            out[key] = {"planted": len(reps), "caught": sum(bool(r.get("caught")) for r in reps),
                        "caught_without_text_invariant": sum(bool(r.get("caught_without_text_invariant", r.get("caught")))
                                                             for r in reps),
                        "errors": sum(r.get("caught") is None for r in reps), "states": reps}
    for core in cores:
        core.close()
    if a.only == "a1a":                  # merge into the existing teeth JSON (the other plants unchanged)
        pj = A.HERE / f"audit_teeth_{c}.json"
        prev = json.loads(pj.read_text())
        prev["A1a_off_by_one_rewritten_slot"] = out["A1a_off_by_one_rewritten_slot"]
        prev["summary"] = {k: ({kk: vv for kk, vv in v.items() if kk not in ("states", "worlds")}) for k, v in prev.items()
                           if isinstance(v, dict) and k != "summary"}
        A.write_json(pj.name, prev)
        print(json.dumps(prev["summary"]["A1a_off_by_one_rewritten_slot"]), flush=True)
        return

    # A2: seen mon's item / EVs swapped in the archived played worlds
    A2.load_layout()
    import os
    from pathlib import Path
    A2.MOVE_NUM.update({k: int(v["num"]) for k, v in json.loads(
        (Path(os.environ["VDP_PIN"]) / "data" / "pokemon" / "gen3_moves.json").read_text()).items()})
    names = A.cell_names()
    W = A.world_rows(c, A.AUDIT_DIRS[0])
    played = sorted(A.played_ids(c, A.AUDIT_DIRS[0]))[: a.n]
    for key, fn, cell_f, req_f in (("A2_seen_item_swapped", swap_item, "items", "item"),
                                   ("A2_seen_evs_swapped", swap_evs, "spread", "stats")):
        n_w, caught_w, caught_cell, caught_req, per, any_w, cmp_w = 0, 0, 0, 0, [], 0, 0
        for did in played:
            d, b = A.decision(bk, did)
            wr = dict(W[did])
            sids = [D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]) for e in getattr(b, "p2" if d["side"] == "p1" else "p1")["team"].split("]")]
            seen = [t for t, s in enumerate(sids) if s in wr["revealed"]]
            if not seen:
                continue
            wr["teams"] = [replace_slot(t, seen[0], fn) for t in wr["teams"][: A2.J]]
            row = A2.compare_state(names, P, b, d, wr, A.AUDIT_DIRS[0])
            for w in row["worlds"]:
                n_w += 1
                cats = w["cats"]
                hc = any(k.startswith(f"UNEXPECTED|own_team|{cell_f}") for k in cats)
                hr = any(k.startswith(f"UNEXPECTED|request|{req_f}") for k in cats)
                caught_cell += hc
                caught_req += hr
                caught_w += hc or hr
                any_w += any(k.startswith("UNEXPECTED") for k in cats)
                compared = not any(k.startswith(("UNEXPECTED|replay", "UNEXPECTED|opp_decisions", "UNEXPECTED|row_order")) for k in cats)
                cmp_w += compared
            per.append({"id": did, "slot": seen[0], "species": sids[seen[0]],
                        "unexpected": dict(collections.Counter(k for w in row["worlds"] for k, n in w["cats"].items()
                                                               if k.startswith("UNEXPECTED") for _ in range(n)))})
        out[key] = {"planted_worlds": n_w, "worlds_compared_cell_by_cell (prefix replayed)": cmp_w,
                    "caught_worlds_in_the_planted_field": caught_w, "caught_worlds_any_UNEXPECTED": any_w,
                    f"caught_by_own_team_{cell_f}_cells": caught_cell,
                    f"caught_by_request_{req_f}": caught_req, "states": per}
        print(f"[teeth {key}]", {k: v for k, v in out[key].items() if k != "states"}, flush=True)

    # A3: one archived world's rendered command perturbed
    reps = []
    for did in sorted(W)[:: max(1, len(W) // (3 * a.n))]:
        if len(reps) >= a.n:
            break
        wr = W[did]
        if not wr.get("ok") or wr["teams"][0] is None:
            continue
        d, b = A.decision(bk, did)
        side = d["side"]
        opp = "p2" if side == "p1" else "p1"
        v = 0 if side == "p1" else 1
        at = cmd_index(b, side, d["n"])
        team, cmds = wr["teams"][0], [list(x) for x in wr["cmds"][0]]
        sets = [D.parse_set(e)[4].split(",") for e in team.split("]")]
        # the first opponent MOVE command whose move sits in exactly ONE set (so the mon is unambiguous)
        idx = next((i for i, (s_, t_) in enumerate(cmds) if s_ == opp and t_.startswith("move ")
                    and sum(D.toid(t_[5:]) in mv for mv in sets) == 1), None)
        if idx is None:
            continue
        m = D.toid(cmds[idx][1][5:])
        own = next(mv for mv in sets if m in mv)
        cmds[idx][1] = f"move {own[(own.index(m) + 1) % len(own)]}"
        rt, rw = run_core([A.rec(b, "t", opp, None, b.commands[:at]), A.rec(b, "w", opp, team, cmds)], trackers=True, obs=True)
        _, ref = D.reveals(rt, v, d["n"])
        try:
            rows = D.reveals(rw, v, d["n"])[1] if rw["ok"] else None
        except RuntimeError:
            rows = None
        caught = rows != ref
        reps.append({"id": did, "cmd_index": idx, "from": f"move {m}", "to": cmds[idx][1], "refused": not rw["ok"],
                     "caught": bool(caught)})
        print("[teeth A3]", reps[-1], flush=True)
    out["A3_perturbed_command"] = {"planted": len(reps), "caught": sum(r["caught"] for r in reps), "worlds": reps}
    reps2 = []
    for did in sorted(W)[:: max(1, len(W) // (3 * a.n))]:
        if len(reps2) >= a.n:
            break
        wr = W[did]
        if not wr.get("ok") or wr["teams"][0] is None:
            continue
        d, b = A.decision(bk, did)
        side = d["side"]
        opp = "p2" if side == "p1" else "p1"
        v = 0 if side == "p1" else 1
        at = cmd_index(b, side, d["n"])
        team, cmds = wr["teams"][0], [list(x) for x in wr["cmds"][0]]
        names_w = [D.parse_set(e)[0] for e in team.split("]")]
        idx = next((i for i, (s_, t_) in enumerate(cmds) if s_ == opp and t_.startswith("switch ")), None)
        if idx is None:
            continue
        cur = cmds[idx][1][7:]
        alt = [nm for nm in names_w if D.toid(nm) != D.toid(cur)]
        cmds[idx][1] = f"switch {alt[-1]}"
        rt, rw = run_core([A.rec(b, "t", opp, None, b.commands[:at]), A.rec(b, "w", opp, team, cmds)], trackers=True, obs=True)
        _, ref = D.reveals(rt, v, d["n"])
        try:
            rows = D.reveals(rw, v, d["n"])[1] if rw["ok"] else None
        except RuntimeError:
            rows = None
        reps2.append({"id": did, "cmd_index": idx, "from": f"switch {cur}", "to": cmds[idx][1], "refused": not rw["ok"],
                      "caught": bool(rows != ref)})
        print("[teeth A3 switch]", reps2[-1], flush=True)
    out["A3_perturbed_switch_target"] = {"planted": len(reps2), "caught": sum(r["caught"] for r in reps2),
                                         "caught_without_a_refusal": sum(r["caught"] and not r["refused"] for r in reps2),
                                         "worlds": reps2}
    out["A3_perturbed_command"]["caught_without_a_refusal"] = sum(r["caught"] and not r["refused"] for r in reps)
    out["summary"] = {k: ({kk: vv for kk, vv in v.items() if kk not in ("states", "worlds")}) for k, v in out.items()
                      if isinstance(v, dict) and k != "summary"}
    A.write_json(f"audit_teeth_{c}.json", out)
    print(json.dumps(out["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
