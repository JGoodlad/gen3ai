"""``python -m main.info_audit`` — run the information-lever audit suite over decisions of the committed Lane S bank and
write one JSON report (``designs/endstate/design_information_levers.md`` §10).

    python -m main.info_audit --n 8 --worlds 4 --specs off,pilot,oracle,F+M --out /tmp/info_audit.json

For every decision: the template identity; for every spec, the worlds the SEARCH builder builds under it and the
checks over them (provenance, legality, genders, the re-gate, the oracle identity under ``oracle``) and the per-class
LEAK read (``checks.Leakage``); and the log source against the Rust reading over the decision's prefix. A spec whose
worlds cannot be built for a decision records it as EXCLUDED (the hidden-info study counts exclusions, §6.1). The
report's ``leakage`` tables put the pilot's builder (``pilot``) beside the honest one (``off``): the per-class
truth-match rates before and after the fix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Sequence

from agents.info_levers.spec import PRESETS
from main.info_audit import checks as C


def select(n: int, min_turn: int = 4, max_turn: int = 40, min_opp_alive: int = 2) -> List[dict]:
    """``n`` free decisions of the committed bank, spread evenly over its fixed subset order (deterministic)."""
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import select_subset

    bank = B.load_bank(BANK_V1)
    dd = {d["id"]: d for d in bank.decisions}
    ids = [i for i in select_subset(bank) if dd[i]["kind"] == "free" and min_turn <= dd[i]["turn"] <= max_turn
           and dd[i]["opp_alive"] >= min_opp_alive]
    step = max(1, len(ids) // max(1, n))
    out = []
    for i in ids[::step][:n]:
        d = dd[i]
        bat = bank.battles[bank.battle_index[d["battle"]]]
        out.append({"id": i, "decision": d, "battle": bat})
    return out


def run(decisions: Sequence[dict], specs: Sequence[str], n_worlds: int, z_max: float = 4.0,
        log=print) -> dict:
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    samplers = {s: W.WorldSampler(reveal=s) for s in specs}
    leak = {s: C.Leakage(z_max=z_max) for s in specs}
    per: Dict[str, List[dict]] = {}
    spec_stats: Dict[str, dict] = {s: {"built": 0, "excluded": [], "screens": 0, "seconds": 0.0} for s in specs}

    def add(v: dict) -> None:
        per.setdefault(v["check"], []).append(v)

    for item in decisions:
        d, bat = item["decision"], item["battle"]
        log_ = to_log(bat)
        at = cmd_index(bat, d["side"], d["n"])
        root = W.prepare_root(log_, at, d["side"], item["id"])
        add(C.template_identity(root))
        add(C.log_consistency(root.true, root.viewer, item["id"]))
        cache: dict = {}                    # one screen per decision, shared by every spec (common random numbers)
        for s in specs:
            t0 = time.time()
            try:
                w = samplers[s].sample(log_, at, d["side"], n_worlds, item["id"], screen_cache=cache)
            except W.WorldError as e:
                spec_stats[s]["excluded"].append({"id": item["id"], "why": str(e)[-300:]})
                spec_stats[s]["seconds"] += time.time() - t0
                log(f"  {s} {item['id']}: EXCLUDED ({str(e)[-160:]})")
                continue
            spec_stats[s]["seconds"] += time.time() - t0
            spec_stats[s]["built"] += 1
            spec_stats[s]["screens"] += w.screens
            spec = samplers[s].spec
            for v in (C.provenance(spec, w), C.legality(root, w), C.genders(root, w), C.regate(root, w)):
                add(dict(v, check=f"{v['check']}[{s}]"))
            if spec == PRESETS["oracle"]:
                add(C.oracle_identity(root, w))
            leak[s].add(w)
            log(f"  {s} {item['id']}: built, {w.screens} screening replays, {time.time() - t0:.1f}s")
    checks = {k: C.merge(v) for k, v in per.items()}
    for s in specs:
        for v in leak[s].verdicts():
            checks[f"{v['check']}[{s}]"] = v
    return {"decisions": len(decisions), "worlds_per_decision": n_worlds, "specs": spec_stats,
            "leakage": {s: {"table": leak[s].table(), "samplers": leak[s].samplers} for s in specs},
            "checks": checks, "pass": all(v["pass"] for v in checks.values())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.info_audit", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=8, help="decisions of the committed bank")
    ap.add_argument("--worlds", type=int, default=4, help="worlds per decision and spec")
    ap.add_argument("--specs", default="off,pilot,oracle,F+M", help="comma list of preset names")
    ap.add_argument("--min-turn", type=int, default=4)
    ap.add_argument("--max-turn", type=int, default=40)
    ap.add_argument("--out", default=None, help="the JSON report (default: stdout)")
    a = ap.parse_args(argv)
    specs = [s for s in a.specs.split(",") if s]
    rep = run(select(a.n, a.min_turn, a.max_turn), specs, a.worlds, log=lambda m: print(m, file=sys.stderr, flush=True))
    text = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    bad = [k for k, v in rep["checks"].items() if not v["pass"]]
    print(f"info_audit: {len(rep['checks'])} checks, {len(bad)} failed{(': ' + ', '.join(bad)) if bad else ''}",
          file=sys.stderr)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
