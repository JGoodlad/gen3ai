"""AM-X47c-2's refreshed production B2 truth: compact the archive rows into this directory and summarise them beside
the 150M truth they replace (READ A's) and the toy's.

    PYTHONPATH=src python designs/research_state/measurements/x35_prod_b2_truth_2026-10-20/summarize.py <archive rows.jsonl>

Writes ``rows_X26at280M_S32.compact.jsonl.gz`` (the committed form, ``truth.compact``) and ``summary.json``: rows,
refusals, the continuation stamp (must be ``gc.TRUTH_BY_SIDE["production"]``), the decisive-turn count, and how the
decisive set and the near-best sets compare with the 150M truth (and the toy's) on the same 306 turns. Arithmetic on
files only."""
import hashlib
import json
import sys
from pathlib import Path

from main.policy_spectrum.truth import load_rows, turn_truth, value_summary, write_compact
from main.x35.gc import TRUTH_BY_SIDE

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "collapse_pilot_2026-10-17"
TOY = HERE.parent / "x35_toy_b2_truth_2026-10-20"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _compare(new: dict, old: dict, subset: list) -> dict:
    both = [i for i in subset if new.get(i, {}).get("ok") and old.get(i, {}).get("ok")]
    tn = {i: turn_truth(new[i]) for i in both}
    to = {i: turn_truth(old[i]) for i in both}
    dn = {i for i in both if tn[i]["dominated"].any()}
    do = {i for i in both if to[i]["dominated"].any()}
    same = [i for i in both if list(tn[i]["actions"]) == list(to[i]["actions"])
            and list(tn[i]["near"]) == list(to[i]["near"])]
    same_dec = [i for i in dn & do if i in set(same)]
    return {"turns_both_ok": len(both), "decisive_new": len(dn), "decisive_old": len(do), "decisive_both": len(dn & do),
            "decisive_only_new": len(dn - do), "decisive_only_old": len(do - dn),
            "same_near_best_set": len(same), "near_best_set_changed": len(both) - len(same),
            "decisive_both_same_near_best_set": len(same_dec)}


def main(src: str) -> int:
    dst = HERE / "rows_X26at280M_S32.compact.jsonl.gz"
    n = write_compact(Path(src), dst)
    new = {r["id"]: r for r in load_rows(dst)}
    old = {r["id"]: r for r in load_rows(PILOT / "rows_X26at150M_S32.compact.jsonl.gz")}
    toy = {r["id"]: r for r in load_rows(TOY / "rows_toyTstar290M_S32.compact.jsonl.gz")}
    conts = sorted({r["continuation"] for r in new.values()})
    subset = json.loads((PILOT / "subset_x26.json").read_text())["ids"]
    out = {"rows": n, "subset": len(subset), "missing": sorted(set(subset) - set(new)),
           "ok": sum(1 for r in new.values() if r.get("ok")), "refused": sum(1 for r in new.values() if not r.get("ok")),
           "continuations": conts, "stamp_is_registered": conts == [TRUTH_BY_SIDE["production"]],
           "seeds": sorted({len(r["seeds"]) for r in new.values() if r.get("ok")}),
           "compact_sha256": _sha(dst),
           "vs_150M": _compare(new, old, subset), "vs_toy": _compare(new, toy, subset),
           "value_summary_280M": value_summary(list(new.values())),
           "value_summary_150M": value_summary(list(old.values()))}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("value_summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
