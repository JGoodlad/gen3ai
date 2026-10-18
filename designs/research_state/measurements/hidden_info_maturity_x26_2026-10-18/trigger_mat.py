"""THE TRIGGER and the INPUT COPIES of the maturity battery (PREREGISTRATION §1, §2.1). No model, no engine: any Python
3 with this checkout on disk. Every read of the run dir is READ-ONLY (``maturity_series.copy_checkpoint``: sha256 of the
source before and after, the copy made read-only, nothing written under ``models/``).

    python -I trigger_mat.py status                          # the run's 25M anchors: registered / pending / missing
    python -I trigger_mat.py register --labels x26_025M,x26_225M --mode lean     # copy + register named anchors
    python -I trigger_mat.py register --pending --mode lean   # every anchor not yet registered (the standing trigger)
    python -I trigger_mat.py traces                           # copy the bank's three eval cycles (PREREGISTRATION §2.1)

``register`` appends to ``<archive>/ckpts.json`` (the queue's work list; a running queue picks a new anchor up at its
next poll). A label already registered is never changed. ``--priority`` orders the queue (lower first; default = the
step).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

MS = E.src_module("maturity_series")


def run_dir() -> Path:
    return E.MODELS / E.series().run


def status() -> dict:
    s = E.series()
    files = MS.checkpoint_files(run_dir())
    reg = E.registered_ckpts()
    anc = MS.anchors(files, s.every, s.first)
    top = files[-1][0] if files else 0
    out = {"run": s.run, "pin": s.pin, "newest_checkpoint": top, "anchors": {}}
    m = s.first
    while m <= top:
        lab = MS.anchor_label(s.name, m)
        hit = anc.get(m)
        out["anchors"][lab] = ("registered" if lab in reg else "pending" if hit else "missing (no checkpoint in its "
                               "window)") + (f" @ {hit[0]}" if hit else "")
        m += s.every
    return out


def register(labels, mode: str, priority: dict) -> None:
    s = E.series()
    files = MS.checkpoint_files(run_dir())
    pend = MS.pending_anchors(files, s, E.registered_ckpts())
    unknown = [x for x in labels if x not in pend and x not in E.registered_ckpts()]
    if unknown:
        raise SystemExit(f"not an anchor of {s.run} with a checkpoint: {unknown}")
    doc = json.loads(E.ckpts_path().read_text()) if E.ckpts_path().exists() else {
        "series": s.name, "run": s.run, "pin": s.pin, "play_commit": s.play_commit(), "anchors": {}}
    mc_src = run_dir() / "model_config.json"
    mc_dst = E.d("ckpt") / "model_config.json"
    mc = MS.copy_checkpoint(mc_src, mc_dst, forbid_under=[E.MODELS])
    if doc.get("model_config_sha256", mc["sha256"]) != mc["sha256"]:
        raise SystemExit("the run's model_config.json changed since the first registration (a new series)")
    doc["model_config_sha256"] = mc["sha256"]
    for lab in labels:
        if lab in doc["anchors"]:
            print(f"{lab}: already registered, unchanged")
            continue
        step, src = pend[lab]
        dst = E.d("ckpt", lab) / src.name
        rec = MS.copy_checkpoint(src, dst, forbid_under=[E.MODELS])
        side = src.with_suffix(".json")
        if side.exists():
            MS.copy_checkpoint(side, dst.with_suffix(".json"), forbid_under=[E.MODELS])
        doc["anchors"][lab] = {"step": step, "file": str(dst), "src": str(src), "sha256": rec["sha256"],
                               "bytes": rec["bytes"], "mode": mode, "priority": int(priority.get(lab, step)),
                               "registered_at": time.strftime("%F %T %z")}
        print(f"{lab}: {src.name} -> {dst} sha256 {rec['sha256'][:16]} ({mode})")
    tmp = E.ckpts_path().with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(E.ckpts_path())


def copy_traces(cycles) -> dict:
    """Copy each named eval cycle (every file but the checkpoint hard link ``snapshot.zip``) with a per-file sha256."""
    s = E.series()
    out = {"run": s.run, "cycles": {}}
    for cyc in cycles:
        src = run_dir() / "eval_traces" / cyc
        dst = E.traces_root() / s.run / "eval_traces" / cyc
        man = json.loads((src / "eval_manifest.json").read_text())
        if man.get("git_hash") != s.pin:
            raise SystemExit(f"{src}: recorded at {man.get('git_hash')}, not the pin {s.pin}")
        files = {}
        for p in sorted(src.rglob("*")):
            if not p.is_file() or p.name == "snapshot.zip":
                continue
            rel = p.relative_to(src)
            rec = MS.copy_checkpoint(p, dst / rel, forbid_under=[E.MODELS])
            files[str(rel)] = rec["sha256"]
        out["cycles"][cyc] = {"files": len(files), "sha256_of_listing": MS.sha256_file(_write_listing(dst, files)),
                              "snapshot_sha256": (man.get("snapshot_storage") or {}).get("sha256")}
        print(f"{cyc}: {len(files)} files")
    (E.traces_root() / "traces_manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def _write_listing(dst: Path, files: dict) -> Path:
    p = dst.parent / f"{dst.name}.sha256.json"
    p.write_text(json.dumps(files, indent=0, sort_keys=True) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("register")
    r.add_argument("--labels", default="")
    r.add_argument("--pending", action="store_true")
    r.add_argument("--mode", choices=("lean", "full"), required=True)
    r.add_argument("--priority", default="", help="label:int,... (lower runs first)")
    t = sub.add_parser("traces")
    t.add_argument("--extension", action="store_true", help="also the pre-registered extension cycle (§2.4)")
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status(), indent=1))
    elif a.cmd == "register":
        if a.pending:
            s = E.series()
            labels = list(MS.pending_anchors(MS.checkpoint_files(run_dir()), s, E.registered_ckpts()))
        else:
            labels = [x for x in a.labels.split(",") if x]
        pr = {k: int(v) for k, v in (x.split(":") for x in a.priority.split(",") if x)}
        register(labels, a.mode, pr)
    else:
        copy_traces(E.DRAW_CYCLES + ((E.EXTENSION_CYCLE,) if a.extension else ()))


if __name__ == "__main__":
    main()
