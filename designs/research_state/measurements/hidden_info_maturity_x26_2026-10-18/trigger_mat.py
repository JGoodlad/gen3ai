"""THE TRIGGER and the INPUT COPIES of the maturity battery (PREREGISTRATION §1, §2.1). No model, no engine: any Python
3 with this checkout on disk. Every read of the run dir is READ-ONLY (``maturity_series.copy_checkpoint``: sha256 of the
source before and after, the copy made read-only, nothing written under ``models/``).

    python -I trigger_mat.py status                          # the run's 25M anchors: registered / pending / missing
    python -I trigger_mat.py register --labels x26_025M,x26_225M --mode lean     # copy + register named anchors
    python -I trigger_mat.py register --pending --mode lean   # every anchor not yet registered (the standing trigger)
    python -I trigger_mat.py traces                           # copy the bank's three eval cycles (PREREGISTRATION §2.1)
    # amendment A1 (2026-10-19), the plateau step:
    python -I trigger_mat.py register --plateau --mode full_hide --priority x26_P280M:10   # the DECLARED plateau P*
    python -I trigger_mat.py set-mode --labels x26_025M --bank interim --mode full --priority 20   # raise a mode
    python -I trigger_mat.py traces --bank plateau --from ~/gen3ai_archive/x26_eval_traces_plateau [--extension]

``register`` appends to ``<archive>/ckpts.json`` (the queue's work list; a running queue picks a new anchor up at its
next poll). A label already registered is never changed. ``--priority`` orders the queue (lower first; default = the
step). ``register --plateau`` registers the series' DECLARED plateau checkpoint (``maturity_series.plateau_anchor``:
the exact file, its sha256 = the declaration) on the interim bank. ``set-mode`` sets one anchor's mode on one bank
(A1's per-(anchor, bank) modes; a mode is only ever RAISED, lean < full < full_hide; each change is logged in the
anchor's ``mode_history``). ``traces --bank plateau --from <dir>`` copies P*'s cycles from an existing copy and checks
every file's sha256 against the run's own file while the run still holds it (a mismatch refuses).
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


def register(labels, mode: str, priority: dict, plateau: bool = False) -> None:
    s = E.series()
    files = MS.checkpoint_files(run_dir())
    pend = MS.pending_anchors(files, s, E.registered_ckpts())
    if plateau:
        lab, hit = MS.plateau_anchor(files, s)
        pend = {lab: hit}
        labels = [lab]
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
        if plateau and rec["sha256"] != s.plateau_sha256:
            raise SystemExit(f"{dst}: the copy's sha256 is not the declared plateau's")
        pr = int(priority.get(lab, step))
        doc["anchors"][lab] = {"step": step, "file": str(dst), "src": str(src), "sha256": rec["sha256"],
                               "bytes": rec["bytes"], "mode": mode, "priority": pr,
                               "registered_at": time.strftime("%F %T %z")}
        if plateau:
            doc["anchors"][lab].update({"plateau": True, "banks": {"interim": {"mode": mode, "priority": pr}}})
        print(f"{lab}: {src.name} -> {dst} sha256 {rec['sha256'][:16]} ({mode})")
    _write_ckpts(doc)


def _write_ckpts(doc: dict) -> None:
    tmp = E.ckpts_path().with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(E.ckpts_path())


def set_mode(labels, bank: str, mode: str, priority) -> None:
    """A1: one anchor's mode (and queue priority) on one bank; a mode is only ever RAISED."""
    if bank not in E.BANKS or mode not in E.MODE_RANK:
        raise SystemExit(f"unknown bank {bank!r} or mode {mode!r}")
    doc = json.loads(E.ckpts_path().read_text())
    for lab in labels:
        e = doc["anchors"].get(lab)
        if e is None:
            raise SystemExit(f"{lab}: not a registered anchor")
        banks = e.setdefault("banks", {"interim": {"mode": e["mode"], "priority": e["priority"]}})
        old = banks.get(bank)
        if old is not None and E.MODE_RANK[mode] < E.MODE_RANK[old["mode"]]:
            raise SystemExit(f"{lab}: {bank} is {old['mode']}; a mode is never lowered (its rows exist)")
        pr = int(priority if priority is not None else (old or {}).get("priority", e["step"]))
        banks[bank] = {"mode": mode, "priority": pr}
        e.setdefault("mode_history", []).append({"at": time.strftime("%F %T %z"), "bank": bank,
                                                 "from": None if old is None else old["mode"], "to": mode,
                                                 "priority": pr})
        print(f"{lab}: {bank} {None if old is None else old['mode']} -> {mode} (priority {pr})")
    _write_ckpts(doc)


def copy_traces(cycles, bank: str = "interim", source_root: Path | None = None) -> dict:
    """Copy each named eval cycle (every file but the checkpoint hard link ``snapshot.zip``) with a per-file sha256.
    ``source_root`` (A1): copy from an existing copy of ``<run>/eval_traces`` instead of the run dir; every file must
    then equal the run's own file (sha256) wherever the run still holds it (counted in the manifest)."""
    s = E.series()
    out = {"run": s.run, "bank": bank, "source_root": str(source_root or run_dir() / "eval_traces"), "cycles": {}}
    for cyc in cycles:
        src = (Path(source_root) if source_root else run_dir() / "eval_traces") / cyc
        run_src = run_dir() / "eval_traces" / cyc
        dst = E.traces_root() / s.run / "eval_traces" / cyc
        man = json.loads((src / "eval_manifest.json").read_text())
        if man.get("git_hash") != s.pin:
            raise SystemExit(f"{src}: recorded at {man.get('git_hash')}, not the pin {s.pin}")
        files, vs_run = {}, {"equal": 0, "absent_in_run": 0}
        for p in sorted(src.rglob("*")):
            if not p.is_file() or p.name == "snapshot.zip":
                continue
            rel = p.relative_to(src)
            rec = MS.copy_checkpoint(p, dst / rel, forbid_under=[E.MODELS])
            files[str(rel)] = rec["sha256"]
            if source_root:
                rp = run_src / rel
                if not rp.is_file():
                    vs_run["absent_in_run"] += 1
                elif MS.sha256_file(rp) != rec["sha256"]:
                    raise SystemExit(f"{p}: differs from the run's own {rp}")
                else:
                    vs_run["equal"] += 1
        if source_root and run_src.is_dir():
            missing = sorted(str(q.relative_to(run_src)) for q in run_src.rglob("*")
                             if q.is_file() and q.name != "snapshot.zip" and str(q.relative_to(run_src)) not in files)
            if missing:
                raise SystemExit(f"{cyc}: the copy lacks {len(missing)} of the run's files, e.g. {missing[:3]}")
        out["cycles"][cyc] = {"files": len(files), "sha256_of_listing": MS.sha256_file(_write_listing(dst, files)),
                              "snapshot_sha256": (man.get("snapshot_storage") or {}).get("sha256"),
                              **({"vs_run_dir": vs_run} if source_root else {})}
        print(f"{cyc}: {len(files)} files" + (f" ({vs_run})" if source_root else ""))
    name = "traces_manifest.json" if bank == "interim" else f"traces_manifest_{bank}.json"
    (E.traces_root() / name).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
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
    r.add_argument("--plateau", action="store_true", help="the series' DECLARED plateau checkpoint (A1)")
    r.add_argument("--mode", choices=tuple(E.MODE_RANK), required=True)
    r.add_argument("--priority", default="", help="label:int,... (lower runs first)")
    m = sub.add_parser("set-mode")
    m.add_argument("--labels", required=True)
    m.add_argument("--bank", choices=sorted(E.BANKS), required=True)
    m.add_argument("--mode", choices=tuple(E.MODE_RANK), required=True)
    m.add_argument("--priority", type=int, default=None)
    t = sub.add_parser("traces")
    t.add_argument("--extension", action="store_true", help="also the pre-registered extension cycle (§2.4)")
    t.add_argument("--bank", choices=sorted(E.BANKS), default="interim")
    t.add_argument("--from", dest="source", default=None, help="an existing copy of <run>/eval_traces to copy from")
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status(), indent=1))
    elif a.cmd == "register":
        if a.pending and a.plateau:
            raise SystemExit("--pending and --plateau are separate registrations")
        if a.pending:
            s = E.series()
            labels = list(MS.pending_anchors(MS.checkpoint_files(run_dir()), s, E.registered_ckpts()))
        else:
            labels = [x for x in a.labels.split(",") if x]
        pr = {k: int(v) for k, v in (x.split(":") for x in a.priority.split(",") if x)}
        register(labels, a.mode, pr, plateau=a.plateau)
    elif a.cmd == "set-mode":
        set_mode([x for x in a.labels.split(",") if x], a.bank, a.mode, a.priority)
    else:
        bk = E.BANKS[a.bank]
        copy_traces(bk.cycles + ((bk.extension,) if a.extension else ()), a.bank,
                    Path(a.source).expanduser() if a.source else None)


if __name__ == "__main__":
    main()
