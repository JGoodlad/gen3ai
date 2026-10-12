"""A1 BUILDER IDENTITY: teeth THROUGH the honest-world builder (not around it, as det_identity_check was).

For every 4th state (sorted ids) with a Phase 1 row: ``determinize.build_worlds`` is run with its sampler replaced
(in THIS process only) so that every "world" IS the TRUE opponent team. The true team then goes through the same
command-template rendering (slot / switch rewrites) and the same gate. Required:
  (a) the gate passes on try 1 for all 8 worlds;
  (b) the RENDERED prefix replays to the same obs rows (BOTH viewers, every decision up to the root) and the same
      per-side protocol chunks as the TRUE prefix;
  (c) ``run_det_playouts.one_state`` on those worlds (the rendered commands) reproduces Phase 1's outcomes on dice
      seeds 0..7 for every legal action exactly.
PASS = zero mismatches. No IMP is computed; Phase 1's outcomes are read only to be compared for byte-equality.

    run_at_pin.sh <cp> 2 8 3h audit_a1_builder_identity.py [--every 4] [--limit N]
→ archive ``hidden_info_binding/<cp>/audit/a1_rows.jsonl`` (resumable), here ``audit_a1_builder_identity_<cp>.json``.
"""
import argparse
import time

import audit_lib as A

C, D = A.C, A.D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--text-only", action="store_true", help="only the builder + the TEXT invariant (no playouts)")
    a = ap.parse_args()
    c = A.cp()
    if a.text_only:
        return text_only(c, a)
    import run_det_playouts as R
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bk = A.bank(c)
    P = A.priors()
    p1 = A.phase1_rows(c)
    ids = sorted(d["id"] for d in bk.decisions)[:: a.every][: a.limit]
    path = A.audit_dir(c) / "a1_rows.jsonl"
    done = {r["id"] for r in C.read_rows(path)}
    todo = [x for x in ids if x not in done]
    print(f"[a1] {c}: {len(ids)} states, {len(done)} done, {len(todo)} to go", flush=True)

    TRUE = {}

    def true_sampler(rng, P_, team, revealed, *args, **kw):     # the monkeypatch: every world IS the true team
        TRUE["calls"] = TRUE.get("calls", 0) + 1
        return team, {"new_slots": 0, "redrawn_move_slots": 0}

    NS = A.builder_ns()                   # the pilot's build_worlds, also returning its command template
    NS["sample_world"] = true_sampler
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(R.J)]
    off = C.clock_offset()
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        fwd = R.Fwd(C.load_model(c))
        with open(path, "a") as f:
            for i, did in enumerate(todo):
                d, b = A.decision(bk, did)
                side = d["side"]
                opp = "p2" if side == "p1" else "p1"
                at = cmd_index(b, side, d["n"])
                row = {"id": did, "turn": d["turn"], "side": side}
                c0 = C.cpu_seconds()
                try:
                    if did not in p1:
                        raise RuntimeError("no Phase 1 row")
                    true_team = getattr(b, opp)["team"]
                    w = NS["build_worlds"](b, side, at, d["n"], did, P, R.J, root_choice_in_log=not p1[did]["other_open"])
                    row["text_invariant"] = A.text_invariant(w, b.commands[:at])
                    row["tries"] = w["tries"]
                    row["fails"] = w["fails"]
                    row["leak"] = w["leak"]
                    row["gate_try1"] = all(t is not None for t in w["teams"]) and all(x == 1 for x in w["tries"])
                    row["teams_true"] = all(t == true_team for t in w["teams"])
                    tcmds = [list(x) for x in b.commands[:at]]
                    rcmds = w["cmds"][0]
                    row["cmds_identical_text"] = rcmds == tcmds
                    row["n_cmd_text_diff"] = sum(x != y for x, y in zip(rcmds, tcmds))
                    row["cmd_text_diff_examples"] = [[x, y] for x, y in zip(rcmds, tcmds) if x != y][:3]
                    rt, rr = run_core([A.rec(b, "t", opp, None, tcmds), A.rec(b, "r", opp, true_team, rcmds)],
                                      trackers=True, obs=True)
                    if not (rt["ok"] and rr["ok"]):
                        raise RuntimeError(f"replay refused: {rt.get('error')} / {rr.get('error')}")
                    rows_eq = {}
                    for v in (0, 1):
                        et, er = A.obs_entries(rt, v), A.obs_entries(rr, v)
                        rows_eq[v] = (len(et) == len(er) and all(x["obs"]["b64"] == y["obs"]["b64"] for x, y in zip(et, er)))
                    row["rows_equal_both_viewers"] = bool(rows_eq[0] and rows_eq[1])
                    row["chunks_equal"] = rt["chunks"] == rr["chunks"]
                    r = R.one_state(cores, fwd, bk, did, {"teams": [true_team] * R.J, "cmds": [rcmds] * R.J}, off)
                    if not r.get("ok"):
                        raise RuntimeError(f"one_state: {r.get('error')}")
                    mism = []
                    if set(r["out"]) != set(p1[did]["out"]):
                        mism.append({"action": "SET", "det": sorted(r["out"]), "phase1": sorted(p1[did]["out"])})
                    for act, s in r["out"].items():
                        ref = p1[did]["out"].get(act, "")[: R.J]
                        if s != ref:
                            mism.append({"action": act, "det": s, "phase1": ref})
                    row["n_actions"] = len(r["out"])
                    row["outcome_mismatch"] = mism
                    row["ok"] = True
                except Exception as exc:          # counted, never dropped
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                    for k, core in enumerate(cores):
                        core.close()
                        cores[k] = SearchCore(lib=lib)
                row["cpu_s"] = round(C.cpu_seconds() - c0, 2)
                C.write_row(f, row)
                el = time.monotonic() - t0
                print(f"[a1] {i + 1}/{len(todo)} {el / 60:.1f} min ok={row['ok']}", flush=True)
    for core in cores:
        core.close()
    summarize(c, ids, path)


def summarize(c, ids, path):
    rows = {r["id"]: r for r in C.read_rows(path)}
    rows = [rows[x] for x in ids if x in rows]
    ok = [r for r in rows if r.get("ok")]
    fail = {
        "error": [r for r in rows if not r.get("ok")],
        "gate_not_try1": [r for r in ok if not r["gate_try1"]],
        "builder_team_not_true": [r for r in ok if not r["teams_true"]],
        "rows_differ": [r for r in ok if not r["rows_equal_both_viewers"]],
        "chunks_differ": [r for r in ok if not r["chunks_equal"]],
        "outcome_mismatch": [r for r in ok if r["outcome_mismatch"]],
        "text_invariant": [r for r in ok if not r.get("text_invariant", {"ok": True})["ok"]],
    }
    out = {
        "audit": "A1 builder identity", "cp": c, "states_selected": len(ids), "states_done": len(rows),
        "states_ok": len(ok), "actions_compared": sum(r["n_actions"] for r in ok),
        "states_with_rewritten_cmds": sum(r["n_cmd_text_diff"] > 0 for r in ok),
        "cmd_text_diffs": sum(r["n_cmd_text_diff"] for r in ok),
        "rewrites": {"moves": sum(r["leak"]["rewritten_moves"] for r in ok),
                     "switches": sum(r["leak"]["rewritten_switches"] for r in ok),
                     "ambiguous_kept_by_name": sum(len(r["leak"]["moves"]) for r in ok),
                     "root_choice_in_log": sum(r["leak"]["root_choice"] is not None for r in ok)},
        "counts": {k: len(v) for k, v in fail.items()},
        "examples": {k: [{kk: vv for kk, vv in r.items() if kk not in ("tries",)} for r in v[:10]] for k, v in fail.items()},
        "cpu_s_total": round(sum(r["cpu_s"] for r in rows), 1),
    }
    out["pass"] = len(rows) == len(ids) and all(len(v) == 0 for v in fail.values())
    A.write_json(f"audit_a1_builder_identity_{c}.json", out)
    print({k: out[k] for k in ("states_done", "states_ok", "actions_compared", "counts", "pass")}, flush=True)


def text_only(c, a):
    """A1's TEXT-LEVEL invariant alone (``audit_lib.text_invariant``) over the SAME states; merged into the A1 JSON."""
    import json

    from main.policy_spectrum.truth import cmd_index

    bk = A.bank(c)
    P = A.priors()
    p1 = A.phase1_rows(c)
    ids = sorted(d["id"] for d in bk.decisions)[:: a.every][: a.limit]
    NS = A.builder_ns()
    NS["sample_world"] = lambda rng, P_, team, rv, *x, **k: (team, {"new_slots": 0, "redrawn_move_slots": 0})
    rows = []
    for did in ids:
        d, b = A.decision(bk, did)
        side = d["side"]
        at = cmd_index(b, side, d["n"])
        w = NS["build_worlds"](b, side, at, d["n"], did, P, 1, root_choice_in_log=not p1[did]["other_open"])
        rows.append({"id": did, **A.text_invariant(w, b.commands[:at])})
    bad = [r for r in rows if not r["ok"]]
    ti = {"rule": "true team: every rendered rewritten token == the original token up to toid; every other token "
                  "byte-equal; template rewritten counts == leak counts",
          "states": len(rows),
          "states_with_rewrites": sum(r.get("rewritten_moves", 0) + r.get("rewritten_switches", 0) > 0 for r in rows),
          "rewritten_moves": sum(r.get("rewritten_moves", 0) for r in rows),
          "rewritten_switches": sum(r.get("rewritten_switches", 0) for r in rows),
          "violations": len(bad), "examples": bad[:10], "pass": not bad}
    pj = A.HERE / f"audit_a1_builder_identity_{c}.json"
    out = json.loads(pj.read_text())
    out["text_invariant"] = ti
    out["pass_playouts_rows_gate"] = out.get("pass_playouts_rows_gate", out["pass"])
    out["pass"] = bool(out["pass_playouts_rows_gate"] and ti["pass"])
    A.write_json(pj.name, out)
    print({k: v for k, v in ti.items() if k != "examples"}, out["pass"], flush=True)


if __name__ == "__main__":
    main()
