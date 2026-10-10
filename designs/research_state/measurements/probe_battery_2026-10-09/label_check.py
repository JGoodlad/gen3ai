"""LABEL VALIDITY of the battery's approximate damage physics (`main.probe_battery.facts._dmg_max`), DESCRIPTIVE.

For every banked decision where the opponent's ACTUAL choice was a damaging move X and our choice was a move (our
active stays in), the calculator's max-roll ratio r = dmg_max(X) / our active's HP is compared with what happened:
did our active faint before our next decision? Rows where the foe itself fainted before our next decision are dropped (its move may never have run). Bands: SURE KO (the MIN roll 0.85·r >= 1.03), NO KO (r <= 0.97),
in between. A sane calculator gives a high faint rate in SURE (below 1 only by misses, the foe being KO'd first, or a
switch-out effect) and a low one in NO (crits ~1/16, residual chip, Explosion-type self-KOs excluded).
Run: PYTHONPATH=src python designs/research_state/measurements/probe_battery_2026-10-09/label_check.py <bank dir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from main.probe_battery import facts as F
from main.probe_battery.bank import load_decisions


def main(bank: str) -> int:
    decs = load_decisions(Path(bank))
    by = defaultdict(list)
    for d in decs:
        by[(d["game"], d["side"])].append(d)
    # the bank is a SUBSAMPLE: use the full replay for the "next decision" read
    full = Path(bank) / "replay_all" / "decisions.jsonl.gz"
    if full.exists():
        from main.probe_battery.bank import iter_jsonl

        by = defaultdict(list)
        for d in iter_jsonl(full):
            by[(d["game"], d["side"])].append(d)
    bands = {"sure": [0, 0], "mid": [0, 0], "no": [0, 0]}
    for seq in by.values():
        seq.sort(key=lambda d: d["n"])
        for d, nxt in zip(seq, seq[1:]):
            ch, oc = d.get("choice") or "", d.get("opp_choice") or ""
            if not ch.startswith("move ") or not oc.startswith("move "):
                continue
            c = F.make_ctx(d)
            if c.oa is None or c.ta is None:
                continue
            mid = oc.split(" ", 1)[1].strip().lower().replace(" ", "").replace("-", "")
            true_ids = F._move_ids(c.ta)
            full_id = next((m for m in true_ids if F._base_id(m) == F._base_id(mid)), None)
            if full_id is None or F._base_id(mid) in ("explosion", "selfdestruct"):
                continue
            dm = F._dmg_max(c.ta, full_id, c.oa, weather=c.weather, screen=c.our_screen)
            if dm is None or dm <= 0:
                continue
            r = dm / max(1, int(c.oa["current_hp"]))
            sp = c.oa["species"]
            m2 = next((m for m in nxt["V"]["ours"]["mons"] if m["species"] == sp), None)
            fainted = m2 is not None and bool(m2.get("fainted") or float(m2.get("hp_fraction") or 0) <= 0)
            t2 = next((m for m in nxt["V"]["opp"]["mons"] if m["species"] == c.ta["species"]), None)
            if t2 is None or t2.get("fainted"):
                continue                     # the foe fell first: its move may never have run
            band = "sure" if 0.85 * r >= 1.03 else ("no" if r <= 0.97 else "mid")
            bands[band][0] += 1
            bands[band][1] += int(fainted)
    out = {k: {"n": v[0], "fainted": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None} for k, v in bands.items()}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
