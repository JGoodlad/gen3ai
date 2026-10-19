"""The GIGO AUDIT of the first LEAN trajectory, written for the read (2026-10-19) and run BEFORE `read_mat.py`'s verdicts
were looked at. It reads only the archive and prints / writes counts and provenance, never an outcome.

Checks (each a PASS / FAIL line; any FAIL is reported before any verdict):
  1. the bank: the LEAN / FULL ids hashes = PREREGISTRATION §6, the bank directory's content sha256 (recomputed by
     ``main.policy_spectrum.bank.load_bank``), the anchors' copies' sha256 = ckpts.json = this folder's
     checkpoints_x26.json, and each copy still equal to its source file in the run directory (read only);
  2. COVERAGE per anchor x unit (roots, belief, O, F, FN, the identity's V_oracle) and per world arm: raw lines, torn
     lines, unique ids, DUPLICATE ids, FOREIGN ids, MISSING ids; the file order of every row file against the sorted
     wanted ids (the E2 ``--limit``-before-done class: a re-picked id shows as a duplicate, a skipped id as missing);
  3. PROVENANCE: every world row's builder head = the registration commit 826bee89; every root / belief / truth /
     playout row at the pin 9c033489; every playout row's ``cp`` / ``step`` / ``arm`` its own; every playout row's
     ``world_head`` the builder's;
  4. EXCLUSIONS / DEFECTS: world exclusions (``ok`` false, no crash) and defects (``crash``, the CRN tooth's
     "CRN broken" included) per arm; playout refusals; the F playout ok-set the SAME at every anchor (F is shared);
  5. NON-FINITES: every float in every row (json reads NaN / Infinity as floats);
  6. ROOT TIES per anchor (rule 8's exclusion).

    python -I gigo_audit_mat.py [--out gigo_audit_mat.json]      (the pin's or HEAD's src is not needed; the bank sha
                                                                  check imports main.policy_spectrum from THIS checkout)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E

HEAD_SHA = "826bee8919c0986fd3733d3d06f03803822a929a"
LEAN_IDS_SHA = "20b98bf42bdc52ced2d5b457bf2e88b6fc06d704687284f8ffbd60c598857626"
FULL_IDS_SHA = "036793b367612d08d32ebf9d05d5d36dfbeccf0b8ee5d0357378ff5d801ccebe"
BANK_CONTENT_SHA = "8da1613d1e28f41294f86689870fc26cad438fa50bd013dc9f8debcc6fde3fe2"


def raw(path: Path):
    """(rows, torn) over one JSONL file, every row kept (duplicates included)."""
    rows, torn = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return rows, torn


def nonfinite(x) -> int:
    if isinstance(x, float):
        return int(not math.isfinite(x))
    if isinstance(x, dict):
        return sum(nonfinite(v) for v in x.values())
    if isinstance(x, list):
        return sum(nonfinite(v) for v in x)
    return 0


def coverage(files: list[Path], want: list[str]) -> dict:
    rows, torn = [], 0
    for p in files:
        r, t = raw(p)
        rows += r
        torn += t
    ids = [r["id"] for r in rows]
    seen: dict[str, int] = {}
    for x in ids:
        seen[x] = seen.get(x, 0) + 1
    ws = set(want)
    return {"files": [p.name for p in files], "lines": len(rows), "torn": torn, "unique": len(seen),
            "duplicates": sum(1 for v in seen.values() if v > 1), "foreign": sorted(set(seen) - ws)[:5],
            "n_foreign": len(set(seen) - ws), "missing": len(ws - set(seen)),
            "file_order_is_sorted": ids == sorted(ids), "nonfinite": sum(nonfinite(r) for r in rows),
            "_rows": rows}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fails: list[str] = []
    doc: dict = {}

    def check(name: str, ok: bool, detail=None):
        doc.setdefault("checks", {})[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)

    # 1. the bank and the anchors
    lean, full = E.mode_ids("lean"), E.mode_ids("full")
    check("lean ids sha = PREREGISTRATION §6", hashlib.sha256("".join(lean).encode()).hexdigest() == LEAN_IDS_SHA
          or hashlib.sha256("\n".join(lean).encode()).hexdigest() == LEAN_IDS_SHA, len(lean))
    check("full ids sha = PREREGISTRATION §6", hashlib.sha256("".join(full).encode()).hexdigest() == FULL_IDS_SHA
          or hashlib.sha256("\n".join(full).encode()).hexdigest() == FULL_IDS_SHA, len(full))
    check("LEAN subset of FULL, 320 / 800", set(lean) <= set(full) and len(lean) == 320 and len(full) == 800)
    sys.path.insert(0, str(E.REPO / "src"))
    from main.policy_spectrum.bank import load_bank  # verifies the content sha on load

    bank = load_bank(E.bank_path())
    check("bank content sha256 (recomputed on load)", bank.manifest["content_sha256"] == BANK_CONTENT_SHA,
          bank.manifest["content_sha256"][:16])
    check("every LEAN id is a bank decision", set(lean) <= {d["id"] for d in bank.decisions})
    reg = E.registered_ckpts()
    folder = json.loads((E.HERE / "checkpoints_x26.json").read_text())
    check("checkpoints_x26.json = the archive's ckpts.json (anchors)",
          folder.get("anchors") == reg, sorted(reg))
    anchors = sorted(reg, key=lambda x: reg[x]["step"])
    for lab in anchors:
        e = reg[lab]
        c, s = sha_file(Path(e["file"])), sha_file(Path(e["src"]))
        check(f"{lab}: copy sha = ckpts.json = the run's file", c == e["sha256"] == s, c[:16])
        check(f"{lab}: mode LEAN", e["mode"] == "lean")
    pin = E.series().pin
    ident = E.identity_ids()

    # 2-5. coverage, provenance, exclusions, non-finites
    wcov = {}
    for arm, dd, want in [("F", E.worlds_dir(None, "F"), lean), ("V:oracle", E.worlds_dir(None, "V:oracle"), ident)] + \
            [(f"FN@{lab}", E.worlds_dir(lab, "FN"), lean) for lab in anchors]:
        cv = coverage(sorted(dd.glob("worlds_*.jsonl")), want)
        rows = cv.pop("_rows")
        cv["ok"] = sum(bool(r.get("ok")) for r in rows)
        cv["excluded"] = sum((not r.get("ok")) and not r.get("crash") for r in rows)
        cv["defect"] = sum(bool(r.get("crash")) for r in rows)
        cv["crn_broken"] = sum("CRN broken" in str(r.get("error", "")) for r in rows)
        cv["excluded_ids"] = sorted(r["id"] for r in rows if not r.get("ok") and not r.get("crash"))
        cv["exclusion_reasons"] = sorted({str(r.get("error", ""))[:80] for r in rows if not r.get("ok")})[:6]
        cv["heads"] = sorted({r.get("head") for r in rows})
        cv["cp_field"] = sorted({str(r.get("cp")) for r in rows})
        wcov[arm] = cv
        check(f"worlds {arm}: one row per wanted id", cv["unique"] == len(want) and cv["duplicates"] == 0
              and cv["n_foreign"] == 0 and cv["missing"] == 0 and cv["torn"] == 0,
              {k: cv[k] for k in ("lines", "unique", "duplicates", "n_foreign", "missing", "torn")})
        check(f"worlds {arm}: builder head = 826bee89", cv["heads"] == [HEAD_SHA], cv["heads"])
        check(f"worlds {arm}: 0 defects (CRN tooth included)", cv["defect"] == 0 and cv["crn_broken"] == 0,
              {"defect": cv["defect"], "crn_broken": cv["crn_broken"]})
        check(f"worlds {arm}: 0 non-finite", cv["nonfinite"] == 0, cv["nonfinite"])
    doc["worlds"] = wcov

    pcov: dict = {}
    f_ok_sets = {}
    for lab in anchors:
        step = reg[lab]["step"]
        units = {"roots": (E.d("roots"), f"{lab}.jsonl", full), "belief": (E.d("fn_belief"), f"{lab}.jsonl", lean),
                 "O": (E.rows_dir(lab, "O"), "shard_*.jsonl", lean), "F": (E.rows_dir(lab, "F"), "rows_*.jsonl", lean),
                 "FN": (E.rows_dir(lab, "FN"), "rows_*.jsonl", lean),
                 "V:oracle": (E.rows_dir(lab, "V:oracle"), "rows_*.jsonl", ident)}
        pcov[lab] = {}
        for u, (dd, pat, want) in units.items():
            cv = coverage(sorted(dd.glob(pat)), want)
            rows = cv.pop("_rows")
            cv["ok"] = sum(bool(r.get("ok", True)) for r in rows)
            cv["not_ok"] = sum(r.get("ok") is False for r in rows)
            cv["defect"] = sum(bool(r.get("defect")) for r in rows)
            cv["excluded"] = sum(bool(r.get("excluded")) for r in rows)
            cv["not_ok_reasons"] = sorted({str(r.get("error", ""))[:80] for r in rows if r.get("ok") is False})[:6]
            pk = "pin_head" if u in ("F", "FN", "V:oracle") else "pin"
            cv["pins"] = sorted({str(r.get(pk)) for r in rows})
            cv["cps"] = sorted({str(r.get("cp")) for r in rows})
            if u in ("F", "FN", "V:oracle", "O"):
                cv["steps"] = sorted({r.get("step") for r in rows})
            if u in ("F", "FN", "V:oracle"):
                cv["world_heads"] = sorted({str(r.get("world_head")) for r in rows})
                cv["arms"] = sorted({str(r.get("arm")) for r in rows})
            if u == "roots":
                cv["ties_lean"] = sum(bool(r.get("tie")) for r in rows if r["id"] in set(lean))
                cv["ties_full"] = sum(bool(r.get("tie")) for r in rows)
            if u == "F":
                f_ok_sets[lab] = {r["id"] for r in rows if r.get("ok")}
            pcov[lab][u] = cv
            tag = f"{lab} {u}"
            check(f"{tag}: one row per wanted id", cv["unique"] == len(want) and cv["duplicates"] == 0
                  and cv["n_foreign"] == 0 and cv["missing"] == 0 and cv["torn"] == 0,
                  {k: cv[k] for k in ("lines", "unique", "duplicates", "n_foreign", "missing", "torn")})
            check(f"{tag}: pin 9c033489", cv["pins"] == [pin], cv["pins"])
            check(f"{tag}: cp field = {lab}", cv["cps"] == [lab], cv["cps"])
            if "steps" in cv:
                check(f"{tag}: step = {step}", cv["steps"] == [step], cv["steps"])
            if "world_heads" in cv:
                wh_ok = all(h == HEAD_SHA or h == "None" for h in cv["world_heads"]) and HEAD_SHA in cv["world_heads"]
                check(f"{tag}: world head 826bee89", wh_ok, cv["world_heads"])
                check(f"{tag}: arm field", cv["arms"] == [{"V:oracle": "V:oracle"}.get(u, u)], cv["arms"])
            check(f"{tag}: 0 defects", cv["defect"] == 0, cv["defect"])
            check(f"{tag}: 0 non-finite", cv["nonfinite"] == 0, cv["nonfinite"])
    doc["per_anchor"] = pcov

    # the shared F's playout ok-set must be the same states at every anchor
    sets = list(f_ok_sets.values())
    check("F playout ok-set identical at every anchor", all(s == sets[0] for s in sets),
          {lab: len(s) for lab, s in f_ok_sets.items()})
    f_world_ok = {r for r in lean if r not in set(wcov["F"]["excluded_ids"])}
    check("F playout ok-set = F world ok-set", all(s == f_world_ok for s in sets), len(f_world_ok))
    doc["F_exclusion_rate"] = wcov["F"]["excluded"] / len(lean)
    doc["fails"] = fails
    print(json.dumps({"F_exclusion_rate": doc["F_exclusion_rate"], "fails": fails}, indent=1))
    if a.out:
        (E.HERE / a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
