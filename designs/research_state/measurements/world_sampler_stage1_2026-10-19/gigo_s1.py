"""STAGE 1's GIGO AUDIT (``PREREGISTRATION.md`` §7), run before the read: counts, provenance and DETERMINISTIC TEETH over
the world and play rows. No outcome is read (the play rows are opened for their ``ok`` / ``error`` / ``log_w`` /
provenance fields only).

  coverage     every state of the read holds exactly one world row and one play row per arm; no duplicate line
  provenance   every world row built at ONE builder commit, every play row at ONE play commit (X26's pin + the roll
               control), every row's ``driver`` that commit
  defects      a builder ``crash`` row or a play ``defect`` row, per arm (each a FINDING)
  rolls        every OK F_rolls world: 8 finite ``log_w`` and 8 force tables; its play row carries the SAME weights
  CRN tooth 1  C_ladder / C_forum_strict on a state where no corpus team fits (n = 0, λ = 0): its worlds ARE B's (the
               backoff takes B's own draws: ``samplers_s1.peek``)
  CRN tooth 2  F_off on a state where F_legal's legality rule rejected nothing: its worlds ARE F_legal's (the rule spends
               no randomness until it rejects: F2's identity, re-checked here on the bank)
  pool tooth   every state's TRUE team is a pool team as X26's pin packed it (C_pool's meta ``true_in_pool``)
  leak tooth   no C_forum_strict world shares >= 4 mon sets with the state's TRUE team (design §6.2)

    python gigo_s1.py [--ids-from-table]      -> gigo_s1.json (this folder); exits 1 on a failed tooth
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_common as E  # noqa: E402

if str(E.REPO / "src") not in sys.path:
    sys.path.insert(1, str(E.REPO / "src"))


def lines(path: Path) -> Counter:
    c: Counter = Counter()
    for r in E.read_rows(path):
        c[r["id"]] += 1
    return c


def audit(ids) -> dict:
    import score_s1 as SC

    out: dict = {"states": len(ids), "teeth": {}, "arms": {}}
    worlds = {a: E.read_by_id(E.worlds_path(a)) for a in E.ARMS}
    rows = {a: E.read_by_id(E.rows_path(a)) for a in E.ARMS}
    heads, pins, drivers = set(), set(), set()
    fail = []
    for a in E.ARMS:
        wl, rl = lines(E.worlds_path(a)), lines(E.rows_path(a))
        blk = {"world_dup": sum(1 for i in ids if wl[i] > 1), "play_dup": sum(1 for i in ids if rl[i] > 1),
               "world_missing": sum(1 for i in ids if i not in worlds[a]),
               "play_missing": sum(1 for i in ids if i not in rows[a]),
               "world_ok": sum(1 for i in ids if worlds[a].get(i, {}).get("ok")),
               "world_excluded": sum(1 for i in ids if i in worlds[a] and not worlds[a][i].get("ok")
                                     and not worlds[a][i].get("crash")),
               "world_defect": sum(1 for i in ids if worlds[a].get(i, {}).get("crash")),
               "play_ok": sum(1 for i in ids if rows[a].get(i, {}).get("ok")),
               "play_refused": sum(1 for i in ids if i in rows[a] and not rows[a][i].get("ok")
                                   and worlds[a].get(i, {}).get("ok"))}
        out["arms"][a] = blk
        for k in ("world_dup", "play_dup", "world_missing", "play_missing", "world_defect"):
            if blk[k]:
                fail.append(f"{a}: {k} {blk[k]}")
        for i in ids:
            if i in worlds[a]:
                heads.add(worlds[a][i].get("head"))
                drivers.add(worlds[a][i].get("driver"))
            if i in rows[a]:
                pins.add(rows[a][i].get("pin_head"))
                drivers.add(rows[a][i].get("driver"))
    out["provenance"] = {"builder_heads": sorted(map(str, heads)), "play_heads": sorted(map(str, pins - {None})),
                         "drivers": sorted(map(str, drivers))}
    if len(heads) != 1 or len(pins - {None}) != 1 or len(drivers) != 1:
        fail.append(f"provenance: {out['provenance']}")
    # rolls
    bad_rolls = 0
    for i in ids:
        w = worlds["F_rolls"].get(i, {})
        if not w.get("ok"):
            continue
        lw = w.get("log_w") or []
        if len(lw) != E.J or len(w.get("rolls") or []) != E.J or not all(abs(x) < 1e9 for x in lw):
            bad_rolls += 1
        elif rows["F_rolls"].get(i, {}).get("ok") and list(rows["F_rolls"][i].get("log_w") or []) != list(lw):
            bad_rolls += 1
    out["teeth"]["rolls"] = {"bad": bad_rolls}
    if bad_rolls:
        fail.append(f"rolls: {bad_rolls}")
    # CRN tooth 1: a corpus arm with no fitting corpus team IS B
    crn1 = Counter()
    for a in ("C_ladder", "C_forum_strict"):
        for i in ids:
            w, b = worlds[a].get(i, {}), worlds["B"].get(i, {})
            if not (w.get("ok") and b.get("ok")) or (w.get("prop") or {}).get("n", -1) != 0:
                continue
            crn1[f"{a}_checked"] += 1
            if w["teams"] != b["teams"]:
                crn1[f"{a}_differs"] += 1
    out["teeth"]["crn_corpus_is_B_when_n0"] = dict(crn1)
    if any(k.endswith("_differs") for k in crn1):
        fail.append(f"CRN tooth 1: {dict(crn1)}")
    # CRN tooth 2: F_off IS F_legal where the rule rejected nothing
    crn2 = Counter()
    for i in ids:
        w, f = worlds["F_off"].get(i, {}), worlds["F_legal"].get(i, {})
        if not (w.get("ok") and f.get("ok")) or any(k.startswith("illegal") for k in (f.get("fails") or {})):
            continue
        crn2["checked"] += 1
        if w["teams"] != f["teams"]:
            crn2["differs"] += 1
    out["teeth"]["crn_off_is_legal_when_untouched"] = dict(crn2)
    if crn2["differs"]:
        fail.append(f"CRN tooth 2: {dict(crn2)}")
    # the pool tooth: every bank opponent IS a pool team (C_pool is the closed-population ceiling only if so)
    pool_meta = [worlds["C_pool"][i].get("pool") for i in ids if i in worlds["C_pool"]]
    pool_meta = [m for m in pool_meta if m]
    miss = sum(1 for m in pool_meta if not m.get("true_in_pool"))
    out["teeth"]["true_team_in_pool"] = {"rows_with_meta": len(pool_meta), "missing": miss,
                                         "true_in_candidates": sum(1 for m in pool_meta if m.get("true_in_candidates"))}
    if miss:
        fail.append(f"pool tooth: {miss} states whose true team is not a pool team")
    # leak tooth
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import to_log

    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    leak = 0
    for i in ids:
        w = worlds["C_forum_strict"].get(i, {})
        if not w.get("ok"):
            continue
        b = bank.battles[bank.battle_index[dec[i]["battle"]]]
        true_team = to_log(b)["teams"][1 if dec[i]["side"] == "p1" else 0]
        if SC.world_meters(w, true_team)["near4"] > 0:
            leak += 1
    out["teeth"]["forum_strict_near_copies"] = leak
    if leak:
        fail.append(f"leak tooth: {leak} states")
    out["FAIL"] = fail
    out["verdict"] = "PASS" if not fail else "FAIL"
    return out


def main():
    import score_s1 as SC

    rows = {a: E.read_by_id(E.rows_path(a)) for a in E.ARMS}
    ids = SC.complete_ids(rows)
    res = audit(ids)
    (E.HERE / "gigo_s1.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("states", "verdict", "FAIL", "teeth", "provenance")}, indent=1))
    if res["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
