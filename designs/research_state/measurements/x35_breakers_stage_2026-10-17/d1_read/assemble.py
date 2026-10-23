"""D1's look-1 read JSON for ``python -m main.x35.readout decide-d1`` (``main.x35.d1.read_d1``), assembled from:

* H: ``python -m main.x35.kakuna d1h --on D1 --off C0`` of each Kakuna state (``h_a`` = C0 s3501-s3503 + D1 s3501,
  ``h_b`` = D1 s3502, ``h_c`` = D1 s3503: one plan per state, written once, as each final landed), merged; the seeds
  of the states are disjoint, so the merge is a union (a clash is a refusal);
* P: ``python -m main.x35.d1 p-input`` on the home72-panel ledger (``d1p_read.py``, ONE request);
* REPORTED beside P (no rule): the pool-ground S (G1's regime), D1 from ``d1s_read.py``, C0 from ``../g1/results.json``,
  with the same Welch 97.5 % interval R1 uses.

    assemble.py --archive ~/gen3ai_archive/x35_d1_read --out read_look1_input.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PY = sys.executable


def _run(argv: list[str]) -> dict:
    out = subprocess.run([PY, *argv], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arc = Path(a.archive).expanduser()
    h: dict = {"on": {}, "off": {}, "ties": {"on": {}, "off": {}}, "units": {"on": {}, "off": {}}}
    for st in ("h_a", "h_b", "h_c"):
        d = _run(["-m", "main.x35.kakuna", "d1h", "--state", str(arc / st), "--on", "D1", "--off", "C0"])
        for side in ("on", "off"):
            for seed, cells in d[side].items():
                if seed in h[side]:
                    raise SystemExit(f"{st}: {side} seed {seed} appears in two states")
                h[side][seed] = cells
                h["ties"][side][seed] = d["ties"][side][seed]
                h["units"][side][seed] = d["units"][side][seed]
    pst = json.loads((arc / "p" / "state.json").read_text())
    finals = [{"label": k, "arm": k[:2].upper(), "seed": int(k[-4:]), "path": v["copy"]}
              for k, v in sorted(pst["nodes"].items())]
    (arc / "finals_p.json").write_text(json.dumps(finals, indent=1) + "\n")
    p = _run(["-m", "main.x35.d1", "p-input", "--ledger", str(arc / "p" / "ledger"), "--family",
              "x35_d1_p_2026-10-23", "--request", "d1p_look1_b100", "--refpop",
              str(STAGE / "reference_population_production.json"), "--finals", str(arc / "finals_p.json"),
              "--on", "D1", "--off", "C0"])
    from main.x35 import readout as R

    s_d1 = {r["seed"]: r["s_pp"] for r in json.loads((arc / "s" / "results.json").read_text())["finals"]}
    g1 = json.loads((STAGE / "g1" / "results.json").read_text())
    s_c0 = {r["seed"]: r["s_pp"] for r in g1["finals"] if r["arm"] == "c0"}
    seeds = sorted(s_c0)
    w = R.welch_interval([s_d1[s] for s in seeds], [s_c0[s] for s in seeds], confidence=R.R1_CONFIDENCE)
    doc = {"arm": "D1", "look": 1, "h": h, "p": p,
           "reported_pool_s": {"on": s_d1, "off": s_c0, "welch975": w.as_dict() if hasattr(w, "as_dict") else str(w),
                               "source": "D1: d1_read/d1s_read.py (family x35_d1_s_2026-10-23); C0: g1/results.json; "
                                         "regime adfdbee824c9eb0a, 7 members x 500 mirrored pairs"}}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"h_on": h["on"], "h_off": h["off"], "p_on": p["on"], "p_off": p["off"],
                      "pool_s": doc["reported_pool_s"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
