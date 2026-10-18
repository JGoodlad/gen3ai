"""E2's GIGO AUDIT of the archive, run BEFORE the read is believed (pure stdlib; read-only on every input; no outcome
is computed: counts, ids, provenance fields and finiteness only).

Per checkpoint, against the bank (`states_e2_<cp>.json` + the bank's own manifest / decisions):
  * the BANK: the manifest's content sha = the states file's; the decision ids = the states ids; 600 states, 120 per gap
    quintile, at most 2 per battle;
  * the TRUTH (rows/O): one row per bank id (RAW lines counted, so a duplicate the reader's last-row-wins `read_dir` would
    hide is caught; byte-identical vs differing duplicates told apart), no foreign id, the pin / set / cp fields, 40 seeds,
    the same action keys in out / crit / trace, and the shard file's row ORDER = sorted(bank ids). Both phase1_e2 versions
    select new work in sorted-id order (d5880c87 took sorted ids[:30] before dropping done ids; 6a1cf301 takes the first
    30 not-done ids), so the order check proves every post-fix row was the next not-done id and no id was re-picked;
  * per arm, the WORLDS (worlds/<arm>) and the PLAYOUT ROWS (rows/<arm>): the same count / duplicate / foreign-id checks,
    the build statuses (ok / excluded / defect = crash), the builder head and the playout pin head, world <-> row status
    consistency, 8 worlds per ok row, the arm's action keys = the truth's;
  * the identity validation set (V_oracle): exactly the 12 registered identity ids;
  * the pin RE-GATES (`pin_regate_e2_<cp>_<arm>.json`, this folder): the registered criterion re-checked (worlds > 0,
    fail = 0, the revealed-move plant caught every time it was tried), its states = the arm's ok worlds, 8 worlds each,
    its pin head = the pad pin;
  * NON-FINITE values: every float in every row, recursively (json reads NaN / Infinity as floats).

    python -I gigo_audit_e2.py [--out gigo_audit_e2.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

PRE_FIX = "d5880c879ad5690fc60397d92eb2d71b67d5c6ef"      # the E2 HEAD that built every world (and ran truth until 14:20)


def nonfinite(x, path: str, acc: Counter) -> None:
    if isinstance(x, float):
        if not math.isfinite(x):
            acc[path] += 1
    elif isinstance(x, dict):
        for v in x.values():
            nonfinite(v, path, acc)
    elif isinstance(x, list):
        for v in x:
            nonfinite(v, path, acc)


def scan(path: Path, label: str):
    """Every raw row of a JSONL file: (rows in file order, torn-line count, non-finite counts by top-level key)."""
    rows, torn, nf = [], 0, Counter()
    if not path.exists():
        return rows, torn, nf
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            for k, v in r.items():
                nonfinite(v, f"{label}.{k}", nf)
            rows.append((r, hashlib.sha256(line.rstrip(b"\n")).hexdigest()))
    return rows, torn, nf


def dup_report(rows) -> dict:
    by = {}
    for r, h in rows:
        by.setdefault(r["id"], []).append(h)
    dups = {i: hs for i, hs in by.items() if len(hs) > 1}
    return {"ids_with_more_than_one_row": len(dups),
            "of_which_byte_identical": sum(len(set(hs)) == 1 for hs in dups.values()),
            "examples": sorted(dups)[:5]}, by


def coverage(by: dict, bank: set) -> dict:
    return {"rows": sum(len(v) for v in by.values()), "distinct_ids": len(by),
            "bank_ids_missing": len(bank - set(by)), "foreign_ids": len(set(by) - bank),
            "missing_examples": sorted(bank - set(by))[:5], "foreign_examples": sorted(set(by) - bank)[:5]}


def bank_check(cp: str) -> dict:
    doc = json.loads(E.states_path(cp).read_text())
    st = doc["states"]
    man = json.loads((E.bank_path(cp) / "manifest.json").read_text())
    with gzip.open(E.bank_path(cp) / "decisions.jsonl.gz", "rt") as f:
        dec_ids = sorted(json.loads(x)["id"] for x in f if x.strip())
    per_q = Counter(int(e["quintile"]) for e in st.values())
    per_b = Counter(e["battle"] for e in st.values())
    return {"states": len(st),
            "manifest_sha_matches_states_file": man.get("content_sha256") == doc.get("bank_content_sha256"),
            "decision_ids_equal_states_ids": dec_ids == sorted(st), "per_quintile": dict(sorted(per_q.items())),
            "max_states_per_battle": max(per_b.values()), "battles": len(per_b),
            "k_counts": dict(sorted(Counter(int(e["k"]) for e in st.values()).items()))}


def audit_cp(cp: str, nf_all: Counter) -> dict:
    st = E.e2_states(cp)
    bank = set(st)
    res = {"bank": bank_check(cp)}
    # ---- truth
    rows, torn, nf = scan(E.ARCH / cp / "rows" / "O" / "shard_0_of_1.jsonl", f"{cp}.O")
    extra_files = sorted(p.name for p in (E.ARCH / cp / "rows" / "O").glob("*.jsonl") if p.name != "shard_0_of_1.jsonl")
    nf_all.update(nf)
    dr, by = dup_report(rows)
    order = [r["id"] for r, _ in rows]
    tr = {"files_other_than_shard_0_of_1": extra_files, "torn_lines": torn, **coverage(by, bank), "duplicates": dr,
          "file_order_equals_sorted_bank_ids": order == sorted(bank),
          "status": dict(Counter("ok" if r.get("ok") else "refused" for r, _ in rows)),
          "field_mismatch": {"cp": sum(r.get("cp") != cp for r, _ in rows),
                             "set": sum(r.get("set") != "e2" for r, _ in rows),
                             "pin": sum(r.get("pin") != E.CPS[cp]["pin"] for r, _ in rows),
                             "n_seeds_not_40": sum(r.get("n_seeds") != 40 for r, _ in rows if r.get("ok")),
                             "out_len_not_40": sum(any(len(s) != 40 for s in r["out"].values()) for r, _ in rows
                                                   if r.get("ok")),
                             "action_keys_differ_out_crit_trace": sum(
                                 not (set(r["out"]) == set(r["crit"]) == set(r["trace"])) for r, _ in rows if r.get("ok"))},
          "outcome_chars": sorted({c for r, _ in rows if r.get("ok") for s in r["out"].values() for c in s}),
          "rows_by_selecting_code": {"d5880c87 (pre-fix, the first 30-id call)": min(30, len(order)),
                                     "6a1cf301 (post-fix)": max(0, len(order) - 30)}}
    acts = {r["id"]: set(r["out"]) for r, _ in rows if r.get("ok")}
    res["truth"] = tr
    # ---- arms
    arms = {}
    for a in E.ARMS:
        sub = E.arm_dir(a)
        wrows, wtorn, wnf = scan(E.ARCH / cp / "worlds" / sub / "worlds_0_of_1.jsonl", f"{cp}.worlds.{sub}")
        rrows, rtorn, rnf = scan(E.ARCH / cp / "rows" / sub / "rows_0_of_1.jsonl", f"{cp}.rows.{sub}")
        nf_all.update(wnf)
        nf_all.update(rnf)
        wdr, wby = dup_report(wrows)
        rdr, rby = dup_report(rrows)
        W = {r["id"]: r for r, _ in wrows}
        R = {r["id"]: r for r, _ in rrows}
        wstat = Counter("ok" if r.get("ok") else ("defect" if r.get("crash") else "excluded") for r in W.values())
        rstat = Counter("ok" if r.get("ok") else ("defect" if r.get("defect") else "excluded" if r.get("excluded")
                                                  else "refused") for r in R.values())
        other = {"worlds": sorted(p.name for p in (E.ARCH / cp / "worlds" / sub).glob("*.jsonl")),
                 "rows": sorted(p.name for p in (E.ARCH / cp / "rows" / sub).glob("*.jsonl"))}
        arms[a] = {
            "files": other, "worlds": {"torn_lines": wtorn, **coverage(wby, bank), "duplicates": wdr,
                                       "status": dict(wstat),
                                       "builder_head": dict(Counter(r.get("head") for r in W.values())),
                                       "arm_label_mismatch": sum(r.get("arm") != a for r in W.values()),
                                       "ok_without_8_worlds": sum(len(r.get("teams") or []) != E.J or
                                                                  len(r.get("cmds") or []) != E.J
                                                                  for r in W.values() if r.get("ok")),
                                       "defect_examples": [f"{i}: {str(r.get('error'))[:160]}" for i, r in W.items()
                                                           if not r.get("ok") and r.get("crash")][:10]},
            "rows": {"torn_lines": rtorn, **coverage(rby, bank), "duplicates": rdr, "status": dict(rstat),
                     "pin_head": dict(Counter(r.get("pin_head") for r in R.values())),
                     "world_head": dict(Counter(r.get("world_head") for r in R.values())),
                     "arm_label_mismatch": sum(r.get("arm") != a for r in R.values()),
                     "n_worlds_not_8": sum(r.get("n_worlds") != E.J for r in R.values() if r.get("ok")),
                     "out_len_not_8": sum(any(len(s) != E.J for s in r["out"].values()) for r in R.values()
                                          if r.get("ok")),
                     "action_keys_differ_from_truth": sum(set(r["out"]) != acts.get(i) for i, r in R.items()
                                                          if r.get("ok") and i in acts),
                     "refused_examples": [f"{i}: {str(r.get('error'))[:160]}" for i, r in R.items()
                                          if not r.get("ok") and not r.get("excluded") and not r.get("defect")][:10]},
            "consistency": {"row_ok_world_not_ok": sum(1 for i, r in R.items()
                                                       if r.get("ok") and not W.get(i, {}).get("ok")),
                            "world_not_ok_row_ok": sum(1 for i, w in W.items()
                                                       if not w.get("ok") and R.get(i, {}).get("ok")),
                            "world_ok_row_not_ok": sum(1 for i, w in W.items() if w.get("ok") and i in R
                                                       and not R[i].get("ok")),
                            "row_world_head_differs_from_world_head": sum(
                                1 for i, r in R.items() if i in W and r.get("world_head") != W[i].get("head")),
                            "world_without_row": len(set(W) - set(R)), "row_without_world": len(set(R) - set(W))}}
    res["arms"] = arms
    # ---- the identity validation set
    want = set((E.ARCH / cp / "identity_ids.txt").read_text().split())
    vw, _, vwnf = scan(E.ARCH / cp / "worlds" / "V_oracle" / "worlds_0_of_1.jsonl", f"{cp}.worlds.V_oracle")
    vr, _, vrnf = scan(E.ARCH / cp / "rows" / "V_oracle" / "rows_0_of_1.jsonl", f"{cp}.rows.V_oracle")
    nf_all.update(vwnf)
    nf_all.update(vrnf)
    res["identity_set"] = {"registered_ids": len(want), "worlds_ids_equal": {r["id"] for r, _ in vw} == want,
                           "rows_ids_equal": {r["id"] for r, _ in vr} == want, "worlds_rows": len(vw), "rows_rows": len(vr),
                           "rows_ok": sum(bool(r.get("ok")) for r, _ in vr)}
    # ---- the pin re-gates
    rg = {}
    for a in E.ARMS:
        p = E.HERE / f"pin_regate_e2_{cp}_{E.arm_dir(a)}.json"
        if not p.exists():
            rg[a] = {"present": False}
            continue
        j = json.loads(p.read_text())
        mv = j.get("plant_move", {"tried": 0, "caught": 0})
        n_ok = arms[a]["worlds"]["status"].get("ok", 0)
        crit = bool(j["worlds"] and j["fail"] == 0 and mv["tried"] and mv["caught"] == mv["tried"])
        rg[a] = {"present": True, "recorded": j["verdict"], "criterion_recomputed": "PASS" if crit else "FAIL",
                 "agree": (j["verdict"] == "PASS") == crit, "states": j["states"], "ok_worlds_rows": n_ok,
                 "states_equal_ok_world_rows": j["states"] == n_ok,
                 "worlds_equal_8x_states": j["worlds"] == E.J * j["states"],
                 "pin_head_is_pad": str(j.get("pin_head", "")).startswith(E.CPS[cp]["pad"]), "fail": j["fail"],
                 "plant_move": mv, "plant_item": j.get("plant_item"), "plant_strip": j.get("plant_strip"),
                 "padded_worlds": j.get("padded_worlds")}
    res["pin_regates"] = rg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gigo_audit_e2.json")
    a = ap.parse_args()
    nf = Counter()
    doc = {cp: audit_cp(cp, nf) for cp in E.ORDER}
    doc["non_finite_values"] = {"total": sum(nf.values()), "by_source_field": dict(sorted(nf.items()))}
    doc["queue_done"] = (E.ARCH / "QUEUE_DONE").read_text().strip() if (E.ARCH / "QUEUE_DONE").exists() else None
    s = json.dumps(doc, indent=1, sort_keys=True)
    (E.HERE / a.out).write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
