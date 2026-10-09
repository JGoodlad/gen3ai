"""Static-token screen look 2: the DEVIATION check of every seed, read-only, from the runs' own records.

The brief's claim to verify: L4, L5, S4 and S5 (the look-2 seeds) ran with NO deviation — one launcher child, no crash
or interval restart, no K9(b) stop, no ``--behaviour-check warn`` — while S1 / S2 (one tie-only resume each,
amendment 1) and S3 (the amendment-2 warn switch) carry the deviations look 1 banked. Run over all ten.

    python deviations.py [--models M] [--jobs ~/.claude/jobs/st_screen] [--out deviations.json]

Per run it reads: the TensorBoard events files under ``tb/`` (one per launcher child); the ``===== child attached``
markers of ``launcher_child.full.log``; ``behaviour_violations.jsonl`` and ``crashes/``; whether ``metadata.json``
mentions ``behaviour-check warn``; the training ``pin_history``; the launcher log's ``Restarts`` summary, its crash and
auto-restart lines and its ``Training complete`` line (``<jobs>/<run>.launcher.log``); the chain's START / END lines
(``<jobs>/chain_status.txt``); and every K9(b) probe in the full child log (``s3_validity.parse``): the maximum
``max_abs_dlogp_current`` / ``_judged`` / ``_excluded`` and ``excluded_frac`` over them (bar 1e-4; the tie-share
ceiling 0.15). K9(b) is FATAL only on a JUDGED row (an excluded near-tie row can move log pi by a tie flip, F-ST-10),
so a run is CLEAN iff it has one child, zero restarts / crashes, no violation file, no warn, every pin 6c6d2e09, every
probe's JUDGED max under the bar and its excluded share under the ceiling, and a final_model.zip; the current-row max
(judged + excluded) is reported beside it."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look1_2026-10-08"))
import s3_validity as S3  # noqa: E402

ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
JOBS = Path.home() / ".claude/jobs/st_screen"
PIN = "6c6d2e0942e2111703a7e6d79bfadb31c8e51f01"
SEEDS = (1001, 1002, 1003, 1004, 1005)
LOOK2_RUNS = ("rb_st_legacy_s1004", "rb_st_legacy_s1005", "rb_st_static_s1004", "rb_st_static_s1005")
CEILING = 0.15


def check(models: Path, jobs: Path, run: str) -> Dict[str, Any]:
    rd = models / run
    meta = json.loads((rd / "metadata.json").read_text())
    pins = [str(p.get("git_hash", "")) for p in meta.get("pin_history") or [] if isinstance(p, dict)]
    full = rd / "launcher_child.full.log"
    lines = full.read_text(errors="replace").splitlines() if full.exists() else []
    attaches = sum(1 for ln in lines if S3.ATTACH.match(ln))
    probes = [b for b in S3.parse(lines) if b.get("rows_probed", 0) > 0]
    ll = jobs / f"{run}.launcher.log"
    ltxt = ll.read_text(errors="replace") if ll.exists() else ""
    restarts = [int(m.group(1)) for m in re.finditer(r"Restarts\s*:\s*(\d+)", ltxt)]
    chain = [ln for ln in (jobs / "chain_status.txt").read_text().splitlines() if f" {run} " in f"{ln} "] \
        if (jobs / "chain_status.txt").exists() else []
    bv = rd / "behaviour_violations.jsonl"
    r: Dict[str, Any] = {
        "run": run, "final_model": (rd / "final_model.zip").is_file(),
        "tb_files": len(list((rd / "tb").glob("events.out.tfevents.*"))),
        "child_attach_markers": attaches, "pin_history": pins,
        "behaviour_violations": len([x for x in bv.read_text().splitlines() if x.strip()]) if bv.exists() else 0,
        "crashes_dir_files": len(list((rd / "crashes").iterdir())) if (rd / "crashes").is_dir() else 0,
        "warn_in_metadata": "behaviour-check warn" in json.dumps(meta),
        "launcher_restarts_summary": restarts,
        "launcher_crash_lines": len(re.findall(r"Child crashed", ltxt)),
        "launcher_auto_restart_lines": len(re.findall(r"Auto-restart", ltxt)),
        "launcher_training_complete": "Training complete" in ltxt,
        "chain": chain,
        "k9b_probes": len(probes),
        "k9b_max_abs_dlogp_current": max((b.get("max_abs_dlogp_current", 0.0) for b in probes), default=None),
        "k9b_max_abs_dlogp_judged": max((b.get("max_abs_dlogp_judged", 0.0) for b in probes), default=None),
        "k9b_max_abs_dlogp_excluded": max((b.get("max_abs_dlogp_excluded", 0.0) for b in probes), default=None),
        "k9b_max_excluded_frac": max((b.get("excluded_frac", 0.0) for b in probes), default=None),
        "k9b_violations_total_max": max((b.get("violations_total_max", 0.0) for b in probes), default=None)}
    r["clean"] = (r["final_model"] and r["tb_files"] == 1 and attaches == 1 and bool(pins)
                  and all(p == PIN for p in pins) and r["behaviour_violations"] == 0 and r["crashes_dir_files"] == 0
                  and not r["warn_in_metadata"] and restarts == [0] and r["launcher_crash_lines"] == 0
                  and r["launcher_auto_restart_lines"] == 0 and r["launcher_training_complete"]
                  and r["k9b_probes"] > 0 and (r["k9b_max_abs_dlogp_judged"] or 0.0) < S3.BAR
                  and (r["k9b_max_excluded_frac"] or 0.0) <= CEILING)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--jobs", default=str(JOBS))
    ap.add_argument("--out", default=str(HERE / "deviations.json"))
    a = ap.parse_args()
    out = {"runs": [check(Path(a.models), Path(a.jobs), f"rb_st_{arm}_s{s}")
                    for arm in ("legacy", "static") for s in SEEDS]}
    out["look2_runs_clean"] = {r["run"]: r["clean"] for r in out["runs"] if r["run"] in LOOK2_RUNS}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for r in out["runs"]:
        print(f"{r['run']}: clean={r['clean']} children(tb/attach)={r['tb_files']}/{r['child_attach_markers']} "
              f"restarts={r['launcher_restarts_summary']} crashes={r['launcher_crash_lines']} "
              f"bv={r['behaviour_violations']} warn={r['warn_in_metadata']} pins={[p[:8] for p in r['pin_history']]} "
              f"probes={r['k9b_probes']} max|dlogp| cur/judged/excl={r['k9b_max_abs_dlogp_current']}/"
              f"{r['k9b_max_abs_dlogp_judged']}/{r['k9b_max_abs_dlogp_excluded']} "
              f"max_excl={r['k9b_max_excluded_frac']} final={r['final_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
