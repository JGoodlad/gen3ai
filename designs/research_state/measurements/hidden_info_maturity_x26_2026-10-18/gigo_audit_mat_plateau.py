"""The GIGO AUDIT of THE PLATEAU STEP (amendment A1, PREREGISTRATION §9.7), written for the read (2026-10-25) and run
BEFORE `read_mat.py plateau`'s verdicts were looked at. It reads only the archive and prints / writes counts,
provenance and file hashes, never an outcome (the k = 6 identities print match COUNTS only).

Checks, on BOTH banks (each a PASS / FAIL line; any FAIL is reported before any verdict):
  1. the banks: the LEAN / FULL ids hashes = PREREGISTRATION §6 (interim) and §9.2 (plateau), each bank directory's
     content sha256 (recomputed by ``main.policy_spectrum.bank.load_bank``), the two banks disjoint in ids and battles;
     every anchor copy's sha256 = ckpts.json = this folder's checkpoints_x26.json = the run's own file (read only);
  2. COVERAGE per (bank, anchor, unit) at the anchor's mode on that bank, and per world arm: torn lines, DUPLICATE,
     FOREIGN and MISSING ids (§9.7);
  3. PROVENANCE: every world row's ``head`` = the builder 826bee89; every pin row at 9c033489; ``cp`` / ``step`` /
     ``arm`` / ``world_head`` each its own; every row carrying ``driver`` = the A1 driver 198fa91e and ``bank`` = its
     namespace; a row WITHOUT ``driver`` only where the LEAN run (before A1) wrote it;
  4. DEFECTS: 0 in every world arm and playout (the CRN tooth's "CRN broken" included); the 1 % defect flag and the
     10 % exclusion flag per (bank, arm) (§9.7);
  5. THE k = 6 IDENTITIES (§9.7; E2's ``identity_e2.tally`` with Tier 1's ``compare``): F+U ≡ F at every FULL anchor and
     O−U ≡ O at P*, on each bank, outcome for outcome;
  6. the shared F playout ok-set identical at every anchor of a bank (on the ids both read) and = the F world ok-set;
  7. NON-FINITES in every row; ROOT TIES per (bank, anchor);
  8. THE ROW MANIFEST: sha256, bytes and mtime of every archive row / world / root / belief file, and the newest mtime
     against ``QUEUE_DONE`` (nothing written after the queue finished).

    python -I gigo_audit_mat_plateau.py [--out gigo_audit_mat_plateau.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gigo_audit_mat as G  # noqa: E402  (raw / coverage / sha_file, the LEAN audit's)
import mat_common as E  # noqa: E402

E.add_paths()

DRIVER_SHA = "198fa91e2c831342c9175769042f9105492863b6"
IDS = {"interim": (G.LEAN_IDS_SHA, G.FULL_IDS_SHA, G.BANK_CONTENT_SHA),
       "plateau": ("4b63651ddc94256aa5992afc9f9f4c44d2af4a74ed520390a121c907a4ac544d",
                   "728dcdd44feff8ee240b7bd0285f2e5857e063f6584f2089a29007fd9f92da85",
                   "25d2e83ce0ce9a52f7e5af29fde9fd95393472703321c5d50e391f721b738a56")}
PSTAR = "x26_P280M"
DEFECT_FLAG, EXCL_FLAG = 0.01, 0.10


def ids_sha(ids) -> str:
    return hashlib.sha256("".join(ids).encode()).hexdigest()


def legacy_ok(bank: str, label, unit: str, did: str, lean_interim: set) -> bool:
    """May this row lack ``driver``? Only what the LEAN run wrote before A1: on the interim bank, the grid anchors' roots
    (all 800), their LEAN-state belief / O / F / FN and identity rows, and the LEAN-state shared F and FN worlds."""
    if bank != "interim" or label == PSTAR:
        return False
    if unit == "roots" or unit.startswith("V:") or unit == "worlds V:oracle":
        return True
    return did in lean_interim and unit in ("belief", "O", "F", "FN", "worlds F", "worlds FN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fails: list[str] = []
    doc: dict = {"checks": {}}

    def check(name: str, ok: bool, detail=None):
        doc["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)

    sys.path.insert(0, str(E.REPO / "src"))
    from identity_mat import e2_tally
    from identity_read import compare          # Tier 1's (pure)
    from main.policy_spectrum.bank import load_bank  # verifies the content sha on load

    tally = e2_tally()
    reg = E.registered_ckpts()
    folder = json.loads((E.HERE / "checkpoints_x26.json").read_text())
    check("checkpoints_x26.json = the archive's ckpts.json (anchors)", folder.get("anchors") == reg, sorted(reg))
    for lab in sorted(reg, key=lambda x: reg[x]["step"]):
        e = reg[lab]
        c = G.sha_file(Path(e["file"]))
        s = G.sha_file(Path(e["src"])) if Path(e["src"]).exists() else "the run's file is gone"
        check(f"{lab}: copy sha = ckpts.json = the run's file", c == e["sha256"] == s, c[:16])
    pin = E.series().pin
    with E.using_bank("interim"):
        lean_interim = set(E.mode_ids("lean"))
    ident = E.identity_ids()
    bank_battles = {}
    doc["banks"] = {}
    for bank in ("interim", "plateau"):
        with E.using_bank(bank):
            bd = doc["banks"][bank] = {"worlds": {}, "per_anchor": {}, "flags": {}, "k6": {}}
            lean, full = E.mode_ids("lean"), E.mode_ids("full")
            lsha, fsha, csha = IDS[bank]
            check(f"[{bank}] lean ids sha = the registration", ids_sha(lean) == lsha, len(lean))
            check(f"[{bank}] full ids sha = the registration", ids_sha(full) == fsha, len(full))
            check(f"[{bank}] LEAN subset of FULL, 320 / 800", set(lean) <= set(full) and len(lean) == 320
                  and len(full) == 800)
            bk = load_bank(E.bank_path())
            check(f"[{bank}] bank content sha256 (recomputed on load)", bk.manifest["content_sha256"] == csha,
                  bk.manifest["content_sha256"][:16])
            meta = E.bank_states()
            bank_battles[bank] = {m.get("battle", x.rsplit("#", 2)[0]) for x, m in meta.items()}
            anchors = sorted((x for x in reg if E.anchor_mode(x) is not None), key=lambda x: reg[x]["step"])
            modes = {lab: E.anchor_mode(lab) for lab in anchors}
            bd["modes"] = modes
            shared_need = sorted({arm for lab in anchors for arm in E.MODE_ARMS[modes[lab]] if E.ARMS[arm][1]})
            shared_want = {arm: (full if any(arm in E.MODE_ARMS[modes[lab]] and modes[lab] != "lean"
                                             for lab in anchors) else lean) for arm in shared_need}
            worlds = [(f"worlds {arm}", None, E.worlds_dir(None, arm), shared_want[arm]) for arm in shared_need]
            worlds += [("worlds FN", lab, E.worlds_dir(lab, "FN"), E.mode_ids(modes[lab])) for lab in anchors]
            if bank == "interim":
                worlds.append(("worlds V:oracle", None, E.worlds_dir(None, "V:oracle"), ident))
            world_ok = {}
            for unit, lab, dd, want in worlds:
                tag = f"[{bank}] {unit}" + (f"@{lab}" if lab else "")
                cv = G.coverage(sorted(dd.glob("worlds_*.jsonl")), want)
                rows = cv.pop("_rows")
                n = max(1, len(want))
                cv["ok"] = sum(bool(r.get("ok")) for r in rows)
                cv["excluded"] = sum((not r.get("ok")) and not r.get("crash") for r in rows)
                cv["defect"] = sum(bool(r.get("crash")) for r in rows)
                cv["crn_broken"] = sum("CRN broken" in str(r.get("error", "")) for r in rows)
                cv["heads"] = sorted({str(r.get("head")) for r in rows})
                cv["exclusion_reasons"] = sorted({str(r.get("error", ""))[:80] for r in rows if not r.get("ok")})[:6]
                drv = provenance(rows, bank, lab, unit, lean_interim)
                cv.update(drv)
                bd["worlds"][tag] = cv
                world_ok[(unit, lab)] = {r["id"] for r in rows if r.get("ok")}
                check(f"{tag}: one row per wanted id", cv["unique"] == len(want) and cv["duplicates"] == 0
                      and cv["n_foreign"] == 0 and cv["missing"] == 0 and cv["torn"] == 0,
                      {k: cv[k] for k in ("lines", "unique", "duplicates", "n_foreign", "missing", "torn")})
                check(f"{tag}: builder head = 826bee89", cv["heads"] == [E.BUILDER_SHA], cv["heads"])
                check(f"{tag}: driver / bank stamps", not cv["bad_driver"] and not cv["unstamped_new"],
                      {k: cv[k] for k in ("stamped", "legacy_unstamped", "unstamped_new", "bad_driver")})
                check(f"{tag}: 0 defects (CRN tooth included)", cv["defect"] == 0 and cv["crn_broken"] == 0,
                      {"defect": cv["defect"], "crn_broken": cv["crn_broken"]})
                check(f"{tag}: 0 non-finite", cv["nonfinite"] == 0, cv["nonfinite"])
                bd["flags"][tag] = {"exclusion_share": cv["excluded"] / n, "over_10pct": cv["excluded"] / n > EXCL_FLAG,
                                    "defect_share": cv["defect"] / n, "over_1pct": cv["defect"] / n > DEFECT_FLAG}

            f_ok: dict = {}
            rowsets: dict = {}
            for lab in anchors:
                step, mode = reg[lab]["step"], modes[lab]
                want = E.mode_ids(mode)
                units = {"roots": (E.d("roots"), f"{lab}.jsonl", full if bank == "interim" else want),
                         "belief": (E.d("fn_belief"), f"{lab}.jsonl", want),
                         "O": (E.rows_dir(lab, "O"), "shard_*.jsonl", want)}
                for arm in E.MODE_ARMS[mode]:
                    units[arm] = (E.rows_dir(lab, arm), "rows_*.jsonl", want)
                if bank == "interim":
                    units["V:oracle"] = (E.rows_dir(lab, "V:oracle"), "rows_*.jsonl", ident)
                bd["per_anchor"][lab] = {}
                for u, (dd, pat, w) in units.items():
                    tag = f"[{bank}] {lab} {u}"
                    cv = G.coverage(sorted(dd.glob(pat)), w)
                    rows = cv.pop("_rows")
                    rowsets[(lab, u)] = {r["id"]: r for r in rows}
                    n = max(1, len(w))
                    cv["not_ok"] = sum(r.get("ok") is False for r in rows)
                    cv["defect"] = sum(bool(r.get("defect")) for r in rows)
                    cv["excluded"] = sum(bool(r.get("excluded")) for r in rows)
                    pk = "pin" if u in ("roots", "belief", "O") else "pin_head"
                    cv["pins"] = sorted({str(r.get(pk)) for r in rows})
                    cv["cps"] = sorted({str(r.get("cp")) for r in rows})
                    cv.update(provenance(rows, bank, lab, u, lean_interim))
                    if u not in ("roots", "belief"):
                        cv["steps"] = sorted({r.get("step") for r in rows})
                    if u not in ("roots", "belief", "O"):
                        cv["world_heads"] = sorted({str(r.get("world_head")) for r in rows})
                        cv["arms"] = sorted({str(r.get("arm")) for r in rows})
                    if u == "roots":
                        cv["ties"] = sum(bool(r.get("tie")) for r in rows if r["id"] in set(want))
                    if u == "F":
                        f_ok[lab] = {r["id"] for r in rows if r.get("ok")}
                    bd["per_anchor"][lab][u] = cv
                    check(f"{tag}: one row per wanted id", cv["unique"] == len(w) and cv["duplicates"] == 0
                          and cv["n_foreign"] == 0 and cv["missing"] == 0 and cv["torn"] == 0,
                          {k: cv[k] for k in ("lines", "unique", "duplicates", "n_foreign", "missing", "torn")})
                    check(f"{tag}: pin 9c033489", cv["pins"] == [pin], cv["pins"])
                    check(f"{tag}: cp field = {lab}", cv["cps"] == [lab], cv["cps"])
                    check(f"{tag}: driver / bank stamps", not cv["bad_driver"] and not cv["unstamped_new"],
                          {k: cv[k] for k in ("stamped", "legacy_unstamped", "unstamped_new", "bad_driver")})
                    if "steps" in cv:
                        check(f"{tag}: step = {step}", cv["steps"] == [step], cv["steps"])
                    if "world_heads" in cv:
                        wh = cv["world_heads"]
                        check(f"{tag}: world head 826bee89", E.BUILDER_SHA in wh
                              and all(h in (E.BUILDER_SHA, "None") for h in wh), wh)
                        check(f"{tag}: arm field", cv["arms"] == [u], cv["arms"])
                    check(f"{tag}: 0 defects", cv["defect"] == 0, cv["defect"])
                    check(f"{tag}: 0 non-finite", cv["nonfinite"] == 0, cv["nonfinite"])
                    if u not in ("roots", "belief", "V:oracle"):
                        bd["flags"][tag] = {"exclusion_share": cv["excluded"] / n,
                                            "over_10pct": cv["excluded"] / n > EXCL_FLAG,
                                            "defect_share": cv["defect"] / n, "over_1pct": cv["defect"] / n > DEFECT_FLAG}
            # 6. the shared F: the same playout ok-set at every anchor (on the ids both read), = the world ok-set
            fw = world_ok[("worlds F", None)]
            for lab, s in f_ok.items():
                w = set(E.mode_ids(modes[lab]))
                check(f"[{bank}] {lab}: F playout ok-set = F world ok-set", s == (fw & w), {"playout": len(s),
                                                                                         "world": len(fw & w)})
            # 5. the k = 6 identities
            k6 = sorted(x for x, m in meta.items() if int(m["k"]) == 6)
            for lab in anchors:
                if modes[lab] == "lean":
                    continue
                F_, FU = rowsets[(lab, "F")], rowsets[(lab, "F+U")]
                fu = [(x, F_.get(x), FU.get(x)) for x in k6 if x in F_ and x in FU]
                t = tally(fu, compare, allow_both_excluded=True)
                bd["k6"][f"F+U = F at k = 6 @{lab}"] = t
                check(f"[{bank}] F+U ≡ F at k = 6 @{lab}", t["verdict"] == "PASS",
                      {k: t[k] for k in ("states", "both_not_ok", "refused", "actions", "out_mismatch",
                                         "trace_mismatch", "crit_mismatch")})
                if modes[lab] == "full_hide":
                    O_, OU = rowsets[(lab, "O")], rowsets[(lab, "O-U")]
                    ou = [(x, O_.get(x), OU.get(x)) for x in k6 if x in O_ and x in OU]
                    t = tally(ou, compare)
                    bd["k6"][f"O-U = O at k = 6 @{lab}"] = t
                    check(f"[{bank}] O−U ≡ O at k = 6 @{lab}", t["verdict"] == "PASS",
                          {k: t[k] for k in ("states", "refused", "actions", "out_mismatch", "trace_mismatch",
                                             "crit_mismatch")})
            bd["k6_states_in_bank"] = len(k6)
    with E.using_bank("interim"):
        ii = set(E.mode_ids("full"))
    with E.using_bank("plateau"):
        pp = set(E.mode_ids("full"))
    check("the two banks share 0 ids and 0 battles", not (ii & pp) and not (bank_battles["interim"]
                                                                          & bank_battles["plateau"]),
          {"ids": len(ii & pp), "battles": len(bank_battles["interim"] & bank_battles["plateau"])})
    # 8. the row manifest, and nothing written after QUEUE_DONE
    done = E.ARCH / "QUEUE_DONE"
    man = {}
    for p in sorted(E.ARCH.rglob("*.jsonl")):
        rel = p.relative_to(E.ARCH).as_posix()
        if rel.split("/")[0] in ("pilot", "traces", "logs") or rel.startswith("pilot"):
            continue
        st = p.stat()
        man[rel] = {"sha256": G.sha_file(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    newest = max(v["mtime_ns"] for v in man.values())
    check("no row file written after QUEUE_DONE", newest <= done.stat().st_mtime_ns,
          {"files": len(man), "QUEUE_DONE": done.read_text().strip()})
    doc["manifest"] = man
    doc["manifest_sha256"] = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()
    doc["fails"] = fails
    print(json.dumps({"checks": len(doc["checks"]), "fails": fails,
                      "flags_raised": sorted(k for b in doc["banks"].values() for k, v in b["flags"].items()
                                             if v["over_10pct"] or v["over_1pct"]),
                      "manifest_sha256": doc["manifest_sha256"]}, indent=1))
    if a.out:
        (E.HERE / a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


def provenance(rows, bank, lab, unit, lean_interim) -> dict:
    """Driver / bank stamps: every stamped row at the A1 driver and its own bank; unstamped only where legacy."""
    stamped = [r for r in rows if "driver" in r]
    bad = sum(r.get("driver") != DRIVER_SHA or r.get("bank") != bank for r in stamped)
    un = [r for r in rows if "driver" not in r]
    leg = sum(legacy_ok(bank, lab, unit, r["id"], lean_interim) for r in un)
    return {"stamped": len(stamped), "bad_driver": bad, "legacy_unstamped": leg, "unstamped_new": len(un) - leg}


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
