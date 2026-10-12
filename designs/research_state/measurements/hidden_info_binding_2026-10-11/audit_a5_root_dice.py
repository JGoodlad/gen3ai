"""A5 ROOT DICE, the empirical half (the source half is the report: ``Game::branch`` -> ``BridgeSession::reseed``
REPLACES the battle PRNG with ``Prng::new(seed)`` at the branch point). No outcome is scored: texts are compared.

Two pairs of games are opened at the same root with the SAME playout seed and stepped in MIRRORED lockstep (both
games get the actions the network picks greedily on game A's rows, so a policy difference can never pass for a dice
difference), every legal action, the searched side's protocol text kept:

  part B (the TEETH)  the bank battle at the searched side's FIRST decision, under its own battle seed (A) and a
                      different battle seed (B): the board at the root is the same (checked: root rows byte-equal),
                      the PRNG state is not. Seeded: every branch's whole text must be byte-equal (post-root dice
                      depend only on the playout seed). CONTROL: the same with seed ``null`` (keep the battle's own
                      stream) must DIFFER somewhere, or the pair has no power and is not counted.
  part A (the brief's) the TRUE team (A) vs the true team with ONE never-seen bench slot replaced by the archived
                      Smogon world 0's set there (B), true prefix (the gate re-checked), seed = Phase 1's leaf seed 0:
                      the root turn's text (through the first ``|turn|`` line) byte-equal on every branch where the
                      swapped mon does not appear in it; and the whole text up to the swapped mon's first line.

    run_at_pin.sh <cp> 1 6 1h audit_a5_root_dice.py [--nb 10] [--na 20]
→ here ``audit_a5_root_dice_<cp>.json``.
"""
import argparse
import ctypes
import re
import json

import numpy as np

import audit_lib as A

C, D = A.C, A.D


def mirrored(cores, fwd, logs, at, side, seed):
    """Open both games, step them with game A's greedy actions; returns (results A, results B, diverged reason)."""
    from run_playouts import ACT, greedy
    from utils.rust_env.successors import _raise, production_stall

    st = []
    for core, log in zip(cores, logs):
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": [seed],
               "stall": production_stall(side), "max_turns": 999, "keep_cmds": True, "text": side}
        root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        nb = int(root["branches"])
        cap = 2 * nb
        st.append({"nb": nb, "cap": cap, "n": 0, "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                   "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                   "acts": np.zeros(cap, dtype=np.int32)})
    why = None
    if st[0]["nb"] != st[1]["nb"]:
        why = "branch count"
    while why is None:
        ks = []
        for core, s in zip(cores, st):
            with core._lock:
                k = core.lib.rust_env_playout_step(core._h, s["acts"].ctypes.data, s["n"], s["rows"].ctypes.data,
                                                   s["masks"].ctypes.data, s["who"].ctypes.data, s["cap"])
                if k == ctypes.c_size_t(-1).value:
                    _raise(core.lib)
            ks.append(int(k))
        if ks[0] != ks[1] or not np.array_equal(st[0]["who"][:ks[0]], st[1]["who"][:ks[0]]):
            why = "pending decisions differ"
            break
        k = ks[0]
        if k == 0:
            break
        a = greedy(fwd(st[0]["rows"][:k], st[0]["masks"][:k]), st[0]["masks"][:k])
        if not all(st[1]["masks"][i, a[i]] for i in range(k)):
            why = "an action illegal in game B"
            break
        for s in st:
            s["acts"][:k] = a
            s["n"] = k
    res = [core._cstr(core.lib.rust_env_playout_results) for core in cores]
    return res[0], res[1], why


_GENDER = re.compile(r", [MF](?=[\"|,]|$)")


def norm(text):
    """The text with every GENDER token removed (switch details and request JSON): a mon's gender is drawn at
    battle CONSTRUCTION from the battle seed (rust_sim turn/driver.rs `draw_turn0_genders`), not by post-root dice."""
    return [_GENDER.sub("", ln) for ln in text]


def through_turn(text):
    out = []
    for ln in text:
        out.append(ln)
        if ln.startswith("|turn|"):
            break
    return out


def until(text, name):
    out = []
    for ln in text:
        if name in ln:
            break
        out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--na", type=int, default=20)
    a = ap.parse_args()
    c = A.cp()
    import run_det_playouts as R
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bk = A.bank(c)
    lib = F.load(F.default_path("release"))
    out = {"audit": "A5 root dice (empirical)", "cp": c, "partB": [], "partA": []}
    with inference_globals(C.threads()):
        fwd = R.Fwd(C.load_model(c))
        # ---------------- part B: same root board, different prefix PRNG state (battle seed)
        seen_b = set()
        for d in sorted(bk.decisions, key=lambda x: x["id"]):
            if len(out["partB"]) >= a.nb:
                break
            if d["battle"] in seen_b:
                continue
            seen_b.add(d["battle"])
            _, b = A.decision(bk, d["id"])
            side = d["side"]
            at = cmd_index(b, side, 0)
            la = to_log(b)
            lb = dict(la, seed="sodium," + "5eed" * 8)
            rec = {"id": d["id"], "side": side, "at": at}
            rb_ = A.rec(b, "a", "p2", None, b.commands[:at])
            rb2 = A.rec(b, "b", "p2", None, b.commands[:at])
            rb2.seed = lb["seed"]
            ra, rbb = run_core([rb_, rb2], trackers=True, obs=True)
            v = 0 if side == "p1" else 1
            ea, eb = A.obs_entries(ra, v), A.obs_entries(rbb, v)
            rec["root_rows_equal"] = bool(ea and eb and ea[-1]["obs"]["b64"] == eb[-1]["obs"]["b64"])
            if not rec["root_rows_equal"]:
                rec["skipped"] = "root rows differ under the two battle seeds"
                out["partB"].append(rec)
                continue
            seed = turn_seeds(d["id"], 1)[0]
            for label, sd in (("seeded", seed), ("control_null", None)):
                cores = [SearchCore(lib=lib), SearchCore(lib=lib)]
                try:
                    r1, r2, why = mirrored(cores, fwd, [la, lb], at, side, sd)
                finally:
                    for core in cores:
                        core.close()
                eq = [norm(b1["text"]) == norm(b2["text"]) and b1["end"] == b2["end"] for b1, b2 in zip(r1["branches"], r2["branches"])]
                raw_eq = [b1["text"] == b2["text"] for b1, b2 in zip(r1["branches"], r2["branches"])]
                fd = []
                for b1, b2 in zip(r1["branches"], r2["branches"]):
                    n1, n2 = norm(b1["text"]), norm(b2["text"])
                    k = next((i for i, (x, y) in enumerate(zip(n1, n2)) if x != y), None)
                    if k is not None and len(fd) < 2:
                        fd.append({"action": b1["action"], "line_index": k, "a": [x[:200] for x in n1[max(0, k - 3):k + 2]],
                                   "b": [x[:200] for x in n2[max(0, k - 3):k + 2]]})
                rec[f"{label}_first_diffs"] = fd
                rec[label] = {"diverged": why, "branches": len(eq), "branches_identical": int(sum(eq)),
                              "branches_identical_incl_gender_tokens": int(sum(raw_eq)),
                              "first_turn_identical": int(sum(through_turn(norm(b1["text"])) == through_turn(norm(b2["text"]))
                                                              for b1, b2 in zip(r1["branches"], r2["branches"])))}
            out["partB"].append(rec)
            print("[a5 B]", rec, flush=True)
        # ---------------- part A: one never-seen bench slot swapped, same seed
        W = A.world_rows(c, "det")
        played = sorted(A.played_ids(c, "det"))
        for did in played:
            if len(out["partA"]) >= a.na:
                break
            d, b = A.decision(bk, did)
            side = d["side"]
            opp = "p2" if side == "p1" else "p1"
            at = cmd_index(b, side, d["n"])
            wr = W[did]
            true_team = getattr(b, opp)["team"]
            ent = true_team.split("]")
            sids = [D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]) for e in ent]
            never = [t for t, s in enumerate(sids) if s not in wr["revealed"]]
            if not never:
                continue
            t = never[0]
            w_ent = wr["teams"][0].split("]")[t]
            new_name = D.parse_set(w_ent)[0]
            if any(D.toid(new_name) in D.toid(x[1]) or D.toid(D.parse_set(ent[t])[0]) in D.toid(x[1])
                   for x in b.commands[:at]):
                continue
            team_b = "]".join(ent[:t] + [w_ent] + ent[t + 1:])
            v = 0 if side == "p1" else 1
            rt, rw = run_core([A.rec(b, "t", opp, None, b.commands[:at]), A.rec(b, "w", opp, team_b, b.commands[:at])],
                              trackers=True, obs=True)
            gate = rw["ok"] and D.reveals(rw, v, d["n"])[1] == D.reveals(rt, v, d["n"])[1]
            rec = {"id": did, "swapped_slot": t, "true": sids[t], "world": D.toid(new_name), "gate": bool(gate)}
            if not gate:
                rec["skipped"] = "the swap changed the searched side's rows"
                out["partA"].append(rec)
                continue
            la = to_log(b)
            lb = R.world_log(b, team_b, [list(x) for x in b.commands[:at]], opp)
            names = (D.parse_set(ent[t])[0], new_name)
            for label, sd in (("seeded", turn_seeds(did, 1)[0]), ("null_battle_stream", None)):
                cores = [SearchCore(lib=lib), SearchCore(lib=lib)]
                try:
                    r1, r2, why = mirrored(cores, fwd, [la, lb], at, side, sd)
                except Exception as exc:     # recorded, never dropped
                    rec[label] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    continue
                finally:
                    for core in cores:
                        core.close()
                ft, fu, n_ok, n_sw = 0, 0, 0, 0
                bad = []
                osw = (f"|switch|p{2 - v}a: ", f"|drag|p{2 - v}a: ")
                for b1, b2 in zip(r1["branches"], r2["branches"]):
                    t1, t2 = through_turn(norm(b1["text"])), through_turn(norm(b2["text"]))
                    if any(nm in ln for nm in names for ln in t1 + t2):
                        continue
                    # a mirrored opponent SWITCH index names a team position, and the swap changes which mon sits
                    # where in the request order: such a branch compares two different choices, not two dice streams
                    if any(ln.startswith(osw) for ln in t1 + t2):
                        n_sw += 1
                        continue
                    n_ok += 1
                    ft += t1 == t2
                    if t1 != t2 and len(bad) < 2:
                        bad.append({"action": b1["action"], "true": t1[-12:], "swapped": t2[-12:]})
                    fu += until(norm(b1["text"]), names[0]) == until(norm(b2["text"]), names[1])
                rec[label] = {"diverged": why, "branches_compared": n_ok, "root_turn_identical": ft,
                              "excluded_opponent_switch_in_root_turn": n_sw,
                              "identical_until_swapped_mon_appears": fu, "examples_differ": bad}
            out["partA"].append(rec)
            print("[a5 A]", {k: ({kk: vv for kk, vv in v_.items() if kk != "examples_differ"} if isinstance(v_, dict) else v_)
                            for k, v_ in rec.items()}, flush=True)
    pb = [r for r in out["partB"] if "seeded" in r]
    pb_power = [r for r in pb if r["control_null"]["branches_identical"] < r["control_null"]["branches"]]
    pa = [r for r in out["partA"] if "seeded" in r and "error" not in r["seeded"] and "error" not in r.get("null_battle_stream", {"error": 1})]
    out["partA_errors"] = [{"id": r["id"], **{k: r[k] for k in ("seeded", "null_battle_stream") if k in r and "error" in r[k]}}
                           for r in out["partA"] if any("error" in r.get(k, {}) for k in ("seeded", "null_battle_stream"))]
    out["summary"] = {
        "partB_pairs": len(pb), "partB_pairs_with_power (null control differs)": len(pb_power),
        "partB_seeded_all_identical": all(r["seeded"]["branches_identical"] == r["seeded"]["branches"] and not r["seeded"]["diverged"]
                                          for r in pb_power),
        "partB_seeded_branches": sum(r["seeded"]["branches"] for r in pb_power),
        "partB_seeded_identical": sum(r["seeded"]["branches_identical"] for r in pb_power),
        "partA_states": len(pa),
        **{f"partA_{lab}_{k}": sum(r[lab][k] for r in pa) for lab in ("seeded", "null_battle_stream")
           for k in ("branches_compared", "root_turn_identical", "identical_until_swapped_mon_appears")},
    }
    out["pass"] = (len(pb_power) > 0 and out["summary"]["partB_seeded_all_identical"]
                   and out["summary"]["partA_seeded_root_turn_identical"] == out["summary"]["partA_seeded_branches_compared"])
    A.write_json(f"audit_a5_root_dice_{c}.json", out)
    print(out["summary"], out["pass"], flush=True)


if __name__ == "__main__":
    main()
