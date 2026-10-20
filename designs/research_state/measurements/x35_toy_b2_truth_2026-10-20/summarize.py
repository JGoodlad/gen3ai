"""AM-X47b-2's toy B2 truth: compact the archive rows into this directory and summarise them beside production's.

    PYTHONPATH=src python designs/research_state/measurements/x35_toy_b2_truth_2026-10-20/summarize.py <archive rows.jsonl>

Writes ``rows_toyTstar290M_S32.compact.jsonl.gz`` (the committed form, ``truth.compact``) and ``summary.json``: rows,
refusals, the continuation stamp (must be ``gc.TRUTH_BY_SIDE["toy"]``), the decisive-turn count, and how the toy's
decisive set and near-best sets compare with production's truth on the same 306 turns. Arithmetic on files only."""
import hashlib
import json
import sys
from pathlib import Path

from main.policy_spectrum.truth import load_rows, turn_truth, value_summary, write_compact
from main.x35.gc import TRUTH_BY_SIDE

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "collapse_pilot_2026-10-17"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(src: str) -> int:
    dst = HERE / "rows_toyTstar290M_S32.compact.jsonl.gz"
    n = write_compact(Path(src), dst)
    toy = {r["id"]: r for r in load_rows(dst)}
    prod = {r["id"]: r for r in load_rows(PILOT / "rows_X26at150M_S32.compact.jsonl.gz")}
    conts = sorted({r["continuation"] for r in toy.values()})
    subset = json.loads((PILOT / "subset_x26.json").read_text())["ids"]
    both = [i for i in subset if toy.get(i, {}).get("ok") and prod.get(i, {}).get("ok")]
    dec_t = {i for i in both if turn_truth(toy[i])["dominated"].any()}
    dec_p = {i for i in both if turn_truth(prod[i])["dominated"].any()}
    same_near = sum(1 for i in both if set(turn_truth(toy[i])["actions"]) == set(turn_truth(prod[i])["actions"])
                    and list(turn_truth(toy[i])["near"]) == list(turn_truth(prod[i])["near"]))
    out = {"rows": n, "subset": len(subset), "missing": sorted(set(subset) - set(toy)),
           "ok": sum(1 for r in toy.values() if r.get("ok")), "refused": sum(1 for r in toy.values() if not r.get("ok")),
           "continuations": conts, "stamp_is_registered": conts == [TRUTH_BY_SIDE["toy"]],
           "seeds": sorted({len(r["seeds"]) for r in toy.values() if r.get("ok")}),
           "compact_sha256": _sha(dst),
           "decisive_toy": len(dec_t), "decisive_production": len(dec_p), "decisive_both": len(dec_t & dec_p),
           "turns_both_ok": len(both), "same_near_best_set": same_near,
           "value_summary_toy": value_summary(list(toy.values())),
           "value_summary_production": value_summary(list(prod.values()))}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("value_summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
