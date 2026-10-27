"""Adversarial parity probe for the lean world screen (review of 2eb65b3d).

Per case: (1) the LEAN build, with every screened core_events batch ALSO replayed in full and the gate's verdict
(passed, matched length, hint) recomputed on both, battle by battle; (2) the REFERENCE build; (3) the two builds'
canonical records compared (``screens`` excluded, as in world_screen_parity_test).
Writes one JSON line per case to the out file (incremental, resumable by case id).
"""
import dataclasses
import json
import sys
import time

from agents.battle import core_replay as CR
from agents.info_levers import revealed as RV
from main.value_search import worlds as W

ORIG = CR.run_core


def gate_of(viewer, n, ref, ref_proto, opp_tag):
    def g(res):
        if not res["ok"]:
            return [False, -1, None, "notok"]
        try:
            _, rows = W.reveals(res, viewer, n)
        except W.WorldError:
            part = RV.viewer_lines(res, viewer)
            return [False, W.divergence_score(part, ref_proto), W.divergence_hint(part, ref_proto, opp_tag), "short"]
        proto = W.protocol_lines(res, viewer, n)
        ln = W.divergence_score(proto, ref_proto)
        hint = W.divergence_hint(proto, ref_proto, opp_tag)
        if rows != ref or proto != ref_proto:
            return [False, ln, hint, "diff"]
        return [True, ln, None, "ok"]
    return g


def canon(rec):
    return json.dumps({k: v for k, v in rec.items() if k != "screens"}, sort_keys=True, default=str)


def build(log, at, side, key, dice, lean, J=8, max_tries=60):
    s = W.WorldSampler(reveal="off", max_tries=max_tries, dice=dice, lean_screen=lean)
    try:
        return dataclasses.asdict(s.sample(log, at, side, J, key, root_choice_in_log=False))
    except W.WorldError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def one(case):
    log, at, side, key, dice = case["log"], case["at"], case["side"], case["key"], case["dice"]
    root = W.prepare_root(log, at, side, key, False)
    g = gate_of(root.viewer, root.n, root.ref, root.ref_proto, f"p{root.oi + 1}")
    st = {"batches": 0, "battles": 0, "stopped": 0, "full_notok": 0, "post_stop_refusal": 0, "mismatch": [],
          "lean_notok": 0}

    def wrapped(battles, **kw):
        lean = ORIG(battles, **kw)
        if kw.get("screen") is None:
            return lean
        assert kw["screen"]["lines"] == root.ref_proto
        kf = dict(kw, screen=None)
        full = ORIG(battles, **kf)
        st["batches"] += 1
        for b, rf, rl in zip(battles, full, lean):
            st["battles"] += 1
            st["stopped"] += bool(rl.get("stopped"))
            st["full_notok"] += not rf["ok"]
            st["lean_notok"] += not rl["ok"]
            if not rf["ok"] and rl["ok"]:
                st["post_stop_refusal"] += 1
            gf, gl = g(rf), g(rl)
            if gf[:3] != gl[:3]:
                st["mismatch"].append({"label": b.label, "full": gf, "lean": gl,
                                       "full_err": (rf.get("error") or "")[:300]})
        return lean

    CR.run_core = wrapped
    t0 = time.time()
    try:
        lean = build(log, at, side, key, dice, True)
    finally:
        CR.run_core = ORIG
    t1 = time.time()
    ref = build(log, at, side, key, dice, False)
    t2 = time.time()
    return {"id": case["id"], "side": side, "at": at, "dice": dice, "n": root.n,
            "lean_err": lean.get("error"), "ref_err": ref.get("error"),
            "tries": ref.get("tries"), "identical": canon(lean) == canon(ref),
            "screens_lean": lean.get("screens"), "screens_ref": ref.get("screens"),
            "t_lean_checked": round(t1 - t0, 1), "t_ref": round(t2 - t1, 1), **st}


def cases_from_bank(k, turn_lo, turn_hi, side_filter=None, dice="seed", skip=0, stride=37):
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    out, seen_b = [], set()
    ds = [d for d in bank.decisions if turn_lo <= d["turn"] <= turn_hi and d.get("kind") == "free"
          and (side_filter is None or d["side"] == side_filter)]
    for d in ds[skip::stride]:
        if d["battle"] in seen_b:
            continue
        seen_b.add(d["battle"])
        bat = bank.battles[bank.battle_index[d["battle"]]]
        out.append({"id": d["id"], "log": to_log(bat), "at": cmd_index(bat, d["side"], d["n"]), "side": d["side"],
                    "key": "lws_review|" + d["id"], "dice": dice})
        if len(out) >= k:
            break
    return out


def cases_from_fixture(turns, dice="seed", sides=("p1",)):
    from pathlib import Path
    fx = json.loads(Path("src/main/value_search/testdata/su_stage0_decisions.json").read_text())
    ds = fx["decisions"]
    big = max(ds, key=lambda d: d["at"])
    out = []
    for t in turns:
        for side in sides:
            at = 2 * t if side == "p1" else 2 * t        # honest_log: turn t's decision sees turns < t only
            out.append({"id": f"fixture_t{t}_{side}_{dice}", "log": big["log"], "at": at, "side": side,
                        "key": f"lws_review|fixture|t{t}|{side}", "dice": dice})
    return out


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    outp = sys.argv[2]
    budget = float(spec.get("budget_s", 480))
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(outp)}
    except FileNotFoundError:
        pass
    if spec["src"] == "bank":
        cases = cases_from_bank(spec["k"], spec["lo"], spec["hi"], spec.get("side"), spec.get("dice", "seed"),
                                spec.get("skip", 0), spec.get("stride", 37))
    else:
        cases = cases_from_fixture(spec["turns"], spec.get("dice", "seed"), tuple(spec.get("sides", ["p1"])))
    t0 = time.time()
    for c in cases:
        if c["id"] in done:
            continue
        if time.time() - t0 > budget:
            print("budget reached", flush=True)
            break
        try:
            r = one(c)
        except Exception as e:  # a case that cannot be prepared is reported, never hidden
            r = {"id": c["id"], "crash": f"{type(e).__name__}: {str(e)[:400]}"}
        with open(outp, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k != "mismatch"}), "mismatches", len(r.get("mismatch", [])),
              flush=True)
