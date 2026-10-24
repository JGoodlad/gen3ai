"""SPRT's (STAGED_PLAN §4.2) look-1 record ``../reads/sprt_look1.json``, assembled from:

* the registered VOID guards per seed: AM-6's K9(b) record (``../k9b/<run>.json`` must read PASS) and V-S1 / V-S2
  (``main.x35.batch2_void.read_void("sprt", run_dir)``, the code ``python -m main.x35.kit void`` prints); a VOID or
  non-PASS seed STOPS the assembly (exit 2), it never enters the read;
* R1: SPRT's three S from ``s_results.json`` (``s_read.py read``), C0's from ``../g1/results.json`` (G1's regime);
  ``main.x35.readout.read`` decides. The guards gate only a QUALIFYING R1 (the orchestrator's A1 ruling (a)); on a
  qualifying R1 this STOPS (exit 3) and lists the guard inputs that are missing, since none was played;
* REPORTED (no rule): the mechanism (accept fraction, N_prom, pairs per candidate, the final median pool age, each
  seed's test wall) from the run directories, and the dose (``../dose/<run>.json``: steps/s and GPU-h).

    assemble.py --s-results s_results.json --out ../reads/sprt_look1.json --input-out read_look1_input.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
MODELS = Path("/home/goodlad/dev/gen3ai/models")
FORK = 280_000_170
SEEDS = ("3501", "3502", "3503")
ON, OFF = "rb_x35_sprt_s{}", "rb_x35_c0_s{}"
#: the registered mechanism prediction (§4.2): the accept fraction over the arm's candidates
ACCEPT_BAND = (0.10, 0.90)


def _snapshots(run: str, final_step: int) -> dict:
    steps = sorted(int(re.search(r"(\d+)", p.name).group(1)) for p in (MODELS / run / "snapshots").glob("snapshot_*.zip"))
    return {"n_snapshots": len(steps), "n_prom": sum(s > FORK for s in steps),
            "median_age_m": statistics.median((final_step - s) / 1e6 for s in steps)}


def _sprt_log(run: str) -> dict:
    rows = [json.loads(ln) for ln in (MODELS / run / "sprt_promotion.jsonl").read_text().splitlines() if ln.strip()]
    v = [r for r in rows if r.get("event") == "verdict"]
    return {"candidates": len(v), "accepted": sum(bool(r.get("promoted")) for r in v),
            "pairs": [int(r["n_pairs"]) for r in v], "stops": [str(r.get("reason")) for r in v],
            "test_scores": [r.get("test_score_selected") for r in v],
            "test_wall_s": round(sum(float(r.get("wall_s", 0.0)) for r in v), 1)}


def main() -> int:
    from main.x35 import batch2_void as V
    from main.x35 import readout as R
    from main.x35.kit import _final_step

    ap = argparse.ArgumentParser()
    ap.add_argument("--s-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-out", required=True)
    a = ap.parse_args()

    void, k9b = {}, {}
    for s in SEEDS:
        run = ON.format(s)
        k = json.loads((STAGE / "k9b" / f"{run}.json").read_text())
        k9b[s] = k["verdict"]
        void[s] = V.read_void("sprt", str(MODELS / run)).as_dict()
        if k9b[s] != "PASS" or void[s]["verdict"] != "PASS":
            print(f"STOP: {run}: K9(b) {k9b[s]}, V-S1/V-S2 {void[s]['verdict']} -> not read", file=sys.stderr)
            return 2

    sr = json.loads(Path(a.s_results).read_text())
    on = {r["seed"]: r["s_pp"] for r in sr["finals"] if r["arm"] == "sprt" and r.get("se_pp") is not None
          and not r["inconclusive"]}
    g1 = json.loads((STAGE / "g1" / "results.json").read_text())
    off = {r["seed"]: r["s_pp"] for r in g1["finals"] if r["arm"] == "c0"}
    if set(on) != set(SEEDS) or set(off) != set(SEEDS):
        print(f"STOP: S missing (sprt {sorted(on)}, c0 {sorted(off)})", file=sys.stderr)
        return 2
    if sr["refpop_content_sha256"] != g1["refpop_content_sha256"]:
        print("STOP: the two reads used different R", file=sys.stderr)
        return 2

    dose = {s: {side: json.loads((STAGE / "dose" / f"{run.format(s)}.json").read_text())
                for side, run in (("on", ON), ("off", OFF))} for s in SEEDS}
    doc = {"arm": "SPRT", "look": 1, "r1": {"on": on, "off": off},
           "dose": {"on": [dose[s]["on"]["gpu_h"] for s in SEEDS], "off": [dose[s]["off"]["gpu_h"] for s in SEEDS]}}
    Path(a.input_out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    v = R.read(doc).as_dict()
    if v["r1"]["class"] == R.GAIN:
        print("STOP: R1 QUALIFIES (GAIN): the guards gate it and none of their inputs was played: "
              "G-F (g_f.h_pp), G-C (g_c, pooled), G-K (g_k home72 / away), G-E (R2 on / off)", file=sys.stderr)
        print(json.dumps(v["r1"], indent=1, default=str), file=sys.stderr)
        return 3
    why = f"not read: R1 is non-qualifying ({v['r1']['class']}), so registered guard semantics cannot change the class"
    for n in ("G-F", "G-C", "G-K"):
        v["guards"][n].update(reasons=[why], status=R.INCONCLUSIVE)
    v["guards"]["G-E"].update(reasons=["absent by design: the G-E readers launch only on an R1 GAIN"])

    mech = {}
    for s in SEEDS:
        fs_on = _final_step(str(MODELS / ON.format(s) / "final_model.zip"))
        fs_off = _final_step(str(MODELS / OFF.format(s) / "final_model.zip"))
        mech[s] = {"sprt": {**_sprt_log(ON.format(s)), **_snapshots(ON.format(s), fs_on)},
                   "c0": _snapshots(OFF.format(s), fs_off)}
    cand = sum(m["sprt"]["candidates"] for m in mech.values())
    acc = sum(m["sprt"]["accepted"] for m in mech.values())
    frac = acc / cand
    pairs = [p for m in mech.values() for p in m["sprt"]["pairs"]]
    np_on = sum(m["sprt"]["n_prom"] for m in mech.values())
    np_off = sum(m["c0"]["n_prom"] for m in mech.values())
    v["reported"]["mechanism"] = {
        "per_seed": mech, "candidates": cand, "accepted": acc, "accept_fraction": frac,
        "accept_fraction_prediction": list(ACCEPT_BAND),
        "accept_fraction_in_band": ACCEPT_BAND[0] <= frac <= ACCEPT_BAND[1],
        "pairs_per_candidate_median": statistics.median(pairs), "pairs_per_candidate_mean": statistics.fmean(pairs),
        "capped": sum(st == "cap" for m in mech.values() for st in m["sprt"]["stops"]),
        "n_prom_pooled": {"sprt": np_on, "c0": np_off}, "n_prom_ge_2x_c0": np_on >= 2 * np_off,
        "median_pool_age_m": {"sprt": [mech[s]["sprt"]["median_age_m"] for s in SEEDS],
                              "c0": [mech[s]["c0"]["median_age_m"] for s in SEEDS]},
        "how": "accept fraction / pairs from each seed's sprt_promotion.jsonl verdict lines; N_prom and age from the "
               "final's snapshots/ (steps > the fork 280,000,170; age = final step - snapshot step, median)"}
    v["reported"]["dose_steps_per_s"] = {
        "sprt": [dose[s]["on"]["steps_per_s"] for s in SEEDS], "c0": [dose[s]["off"]["steps_per_s"] for s in SEEDS],
        "ratio_of_means": statistics.fmean(dose[s]["on"]["steps_per_s"] for s in SEEDS)
        / statistics.fmean(dose[s]["off"]["steps_per_s"] for s in SEEDS),
        "source": "../dose/<run>.json (launcher summary 'FPS'); reported, never decides"}
    rec = {"arm": v["arm"], "look": 1, "verdict": v["verdict"], "r1": v["r1"], "r2": v["r2"], "guards": v["guards"],
           "reasons": v["reasons"], "reported": v["reported"],
           "inputs": {"r1": {"on": on, "off": off,
                             "source": "SPRT: sprt_read/s_read.py (family x35_sprt_read_2026-10-24); C0: g1/results.json;"
                                       " regime adfdbee824c9eb0a, 7 members x 500 mirrored pairs"},
                      "void": void, "k9b": k9b},
           "ruling": "guards gate only a qualifying R1 (the orchestrator's A1 ruling (a)); R1 is non-qualifying, so "
                     "G-F / G-C / G-K are not read and G-E is absent by design"}
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"verdict": rec["verdict"], "r1": {k: v["r1"][k] for k in ("class", "outcome", "final")},
                      "delta_pp": v["r1"]["stat"]["delta_pp"], "ci_pp": v["r1"]["stat"]["ci_pp"],
                      "accept_fraction": frac, "n_prom": [np_on, np_off]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
