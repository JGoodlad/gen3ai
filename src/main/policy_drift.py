"""POLICY DRIFT meter CLI — is training REFINING the current strategy or ADOPTING NEW ONES?

A DESCRIPTOR, not a test: every number is a property of a FROZEN probe-state set, and the verdict
words ("refining" / "shifting (…)" / "cycling?") are reading aids with uncalibrated thresholds
(`agents.training.policy_drift`). Read the series, not one row.

  # 1. freeze a probe set ONCE per lineage, from an early COMPETENT snapshot (pool seeded)
  python -m main.policy_drift collect models/<run> [--battles 40]
  # 2. follow the run: one durable row per NEW snapshot; resumable; read-only on models/
  nohup nice -n 15 python -m main.policy_drift watch models/<run> > <out>/watch.log 2>&1 < /dev/null &
  # 3. read it any time
  python -m main.policy_drift report <run>

Output (never under models/): ``$GEN3AI_ARCHIVE_DIR`` or ``~/gen3ai_archive``, then
``policy_drift/<run>/`` — ``probe.npz`` (+ ``probe.json`` provenance), ``meta.json`` (the probe's
sha256, pinned at the first watch; a different probe is REFUSED), ``rows.jsonl`` (one row per
snapshot), ``probs/<step>.npz`` (each snapshot's action probabilities on the probe — what makes a
pruned snapshot still usable as a 10M-back reference, and what a resume reuses).

RECOLLECTING the probe set invalidates every comparison across the change: move the old out dir
aside (or pass a new ``--out``) and re-run ``collect`` then ``watch``; the old rows stay readable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from agents.training import policy_drift as pd

DEFAULT_BACK = 10_000_000


def archive_root() -> Path:
    return Path(os.environ.get("GEN3AI_ARCHIVE_DIR") or (Path.home() / "gen3ai_archive"))


def resolve_run(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p.resolve()
    from utils.paths import main_models_dir
    md = main_models_dir()
    if md is not None and (md / run).is_dir():
        return (md / run).resolve()
    sys.exit(f"[policy_drift] no run directory {run!r} (not a path, not under the models/ archive)")


def default_out(run_name: str) -> Path:
    return archive_root() / "policy_drift" / run_name


def refuse_under_models(path: Path) -> None:
    from utils.paths import main_models_dir
    md = main_models_dir()
    rp = path.resolve()
    if md is not None and (rp == md.resolve() or md.resolve() in rp.parents):
        sys.exit(f"[policy_drift] REFUSED: {path} is under models/ — this meter never writes there")


def _step_of(zp: Path) -> Optional[int]:
    for rx in (pd._SNAP_RE, pd._CKPT_RE):
        m = rx.match(zp.name)
        if m:
            return int(m.group(1))
    return None


def _obs_layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    mappings = load_mappings()
    return mappings, Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()["layout"]


def make_probs_fn(device: str, obs: np.ndarray, mask: np.ndarray):
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.training.warmstart import masked_action_probs
    mappings, _ = _obs_layout()
    cv = current_model_version(mappings)

    def fn(zp: Path) -> np.ndarray:
        model, _ = load_foreign_opponent(str(zp), current_version=cv, device=device)
        try:
            return masked_action_probs(model, obs, mask)
        finally:
            del model
    return fn


# ── collect ───────────────────────────────────────────────────────────────────────────────────
def cmd_collect(a) -> int:
    src = Path(a.source)
    if src.is_file():
        zp, run_name, seeded = src.resolve(), (a.run_name or src.resolve().parent.parent.name), None
    else:
        run = resolve_run(a.source)
        run_name = run.name
        snaps = pd.list_run_models(run, "snapshots")
        if snaps:
            zp, seeded = snaps[max(snaps)], True
        else:
            ck = pd.list_run_models(run, "checkpoints")
            if not ck:
                sys.exit(f"[policy_drift] {run} has no snapshot or checkpoint yet")
            zp, seeded = ck[max(ck)], False
            print("[policy_drift] ⚠️  NO pool snapshot yet (self-play not engaged): using the latest "
                  "CHECKPOINT. RECOLLECT once the pool seeds — see the module docstring.", flush=True)
    out = Path(a.out) if a.out else default_out(run_name) / "probe.npz"
    refuse_under_models(out)
    if out.exists() and not a.force:
        sys.exit(f"[policy_drift] {out} exists — a probe set is FROZEN; --force (or a new --out) to recollect")
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch as th
    th.set_num_threads(a.threads)
    from agents.training.churn_probe import collect_probe_states
    tmp = out.with_suffix(".tmp.npz")
    n = asyncio.run(collect_probe_states(str(zp), None, str(tmp), battles=a.battles, seed_base=a.seed_base))
    os.replace(tmp, out)
    meta = {"source_ckpt": str(zp), "source_step": _step_of(zp), "pool_seeded": seeded,
            "battles": a.battles, "seed_base": a.seed_base, "n_states": n,
            "sha256": pd.file_sha256(out), "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[policy_drift] probe set: {out} ({n} states from step {meta['source_step']})")
    return 0


# ── watch ─────────────────────────────────────────────────────────────────────────────────────
def _pin_meta(out_dir: Path, probe: Path, run: Path, source: str) -> None:
    sha = pd.file_sha256(probe)
    mp = out_dir / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        if m["probe_sha256"] != sha:
            sys.exit(f"[policy_drift] REFUSED: {probe} is not the probe set this series was built on "
                     f"(sha {sha[:12]} vs {m['probe_sha256'][:12]}). Recollecting starts a NEW out dir.")
        if m["source"] != source:
            sys.exit(f"[policy_drift] REFUSED: series was built on source={m['source']!r}, not {source!r}")
        return
    mp.write_text(json.dumps({"run_dir": str(run), "probe": str(probe), "probe_sha256": sha,
                              "source": source, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                             indent=1))


def cmd_watch(a) -> int:
    run = resolve_run(a.run)
    out_dir = Path(a.out) if a.out else default_out(run.name)
    refuse_under_models(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = Path(a.probe) if a.probe else out_dir / "probe.npz"
    if not probe.exists():
        sys.exit(f"[policy_drift] no probe set at {probe} — run `collect` first")
    if a.nice:
        cur = os.nice(0)
        if cur < a.nice:
            os.nice(a.nice - cur)
    import torch as th
    th.set_num_threads(a.threads)
    _pin_meta(out_dir, probe, run, a.source)
    with np.load(probe) as z:
        obs, mask = z["obs"], z["mask"]
    _, layout = _obs_layout()
    nums = pd.req_move_nums(obs, layout)
    probs_fn = make_probs_fn(a.device, obs, mask)
    if a.anchor and not (out_dir / pd.ANCHOR_FILE).exists():
        azp = Path(a.anchor)
        astep = a.anchor_step if a.anchor_step is not None else _step_of(azp)
        if astep is None:
            sys.exit("[policy_drift] --anchor's step is not in its file name; pass --anchor-step")
        np.savez_compressed(out_dir / pd.ANCHOR_FILE, probs=probs_fn(azp), step=np.int64(astep),
                            path=str(azp))
    print(f"[policy_drift] watching {run} → {out_dir} (pid {os.getpid()}, nice {os.nice(0)}, "
          f"device {a.device}, {len(obs)} probe states)", flush=True)
    log = lambda s: print(s, flush=True)                  # noqa: E731
    while True:
        new = pd.process_pending(run, out_dir, obs, mask, nums, probs_fn, source=a.source,
                                 back_steps=a.back_steps, settle_s=a.settle_s, log=log)
        if a.once:
            return 0
        pending = set(pd.list_run_models(run, a.source)) - {int(r["step"]) for r in pd.load_rows(out_dir)}
        if not new and not pending and (run / "final_model.zip").exists():
            log("[policy_drift] run has final_model.zip and nothing is pending — done")
            return 0
        time.sleep(a.poll)


# ── report ────────────────────────────────────────────────────────────────────────────────────
def _f(x, fmt="{:.3f}"):
    return "   -  " if x is None else fmt.format(x)


def render(rows: list) -> str:
    labs = pd.bucket_labels()
    lines = [
        "POLICY DRIFT — a DESCRIPTOR, not a test (frozen probe set; uncalibrated verdict thresholds)",
        "KL = masked KL(current ‖ ref), mean/median nats · flips = greedy-action flip rate vs prev, "
        f"by the OLDER policy's top1−top2 margin ({' | '.join(labs)})",
        f"{'step':>12} {'KL prev':>13} {'KL back10M':>13} {'KL anchor':>13} {'flips vs prev':>20}  verdict",
    ]
    for r in sorted(rows, key=lambda r: r["step"]):
        if r.get("status") != "ok":
            lines.append(f"{r['step']:>12,} {r.get('status')}: {r.get('error', '')[:80]}")
            continue
        cells = []
        for k in ("prev", "back", "anchor"):
            b = r["refs"].get(k)
            cells.append(f"{_f(b and b['kl_mean'])}/{_f(b and b['kl_median'])}")
        pv = r["refs"].get("prev")
        fl = "/".join(_f(pv and pv["flip"]["buckets"][l]["rate"], "{:.2f}") for l in labs) if pv else "-"
        lines.append(f"{r['step']:>12,} {cells[0]:>13} {cells[1]:>13} {cells[2]:>13} {fl:>20}  "
                     f"{pd.verdict(r)}")
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        last = max(ok, key=lambda r: r["step"])
        lines.append("latest action mix (choice states): " + ", ".join(
            f"{c} {last['shares'][c] * 100:.0f}%" for c in pd.CLASSES if last["shares"].get(c, 0) > 0))
    return "\n".join(lines)


def cmd_report(a) -> int:
    out_dir = Path(a.out) if a.out else default_out(Path(a.run).name)
    rows = pd.load_rows(out_dir)
    if not rows:
        sys.exit(f"[policy_drift] no rows in {out_dir}")
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(render(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.policy_drift", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("collect", help="freeze a probe-state set (bridge battles, ckpt vs itself)")
    c.add_argument("source", help="a run dir/name (latest pool snapshot) or a .zip")
    c.add_argument("--out"); c.add_argument("--run-name")
    c.add_argument("--battles", type=int, default=40)
    c.add_argument("--seed-base", type=int, default=0)
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--force", action="store_true")
    w = sub.add_parser("watch", help="one durable row per new snapshot; resumable; read-only on models/")
    w.add_argument("run"); w.add_argument("--out"); w.add_argument("--probe")
    w.add_argument("--source", choices=("snapshots", "checkpoints", "both"), default="snapshots")
    w.add_argument("--back-steps", type=int, default=DEFAULT_BACK)
    w.add_argument("--anchor", help="an explicit anchor .zip (default: the first snapshot recorded)")
    w.add_argument("--anchor-step", type=int)
    w.add_argument("--device", default="cpu")
    w.add_argument("--threads", type=int, default=2)
    w.add_argument("--nice", type=int, default=15)
    w.add_argument("--poll", type=float, default=300.0)
    w.add_argument("--settle-s", type=float, default=60.0)
    w.add_argument("--once", action="store_true")
    r = sub.add_parser("report", help="compact table + a one-line verdict per snapshot")
    r.add_argument("run", help="run name (or path; only its basename is used)")
    r.add_argument("--out"); r.add_argument("--json", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"collect": cmd_collect, "watch": cmd_watch, "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
