"""X35's LAUNCH KIT, the Python half of ``launch_x35.sh`` (``design_compute_efficiency.md`` §5.6.1-§5.6.3, §5.6.8;
STAGED_PLAN §2-§4). It BUILDS argvs and CHECKS preconditions; it never launches anything.

EVERY ARM'S ARGV comes from the frozen call record (``main.x35.refpop``'s file for the arm's side: P* or T*, its
step, A1's archive with labels), so the launch cannot drift from what R froze. The forms are the staging's validated
stand-ins (``argv_*_standin.txt``, each ``checkargs`` / ``launcher --dry-run`` checked at ``9ebda190``), in one
canonical flag order (the order carries no meaning to the trainer's parser):

* production (``--behaviour-check warn``, F14; +20M): ``c0``; ``a1`` (the 8 archive nodes, share 0.5);
  ``a2`` (``--pfsp-scale 2.5 --n-sentinels 10``);
* the TWO-PHASE kits. Phase 1 (+5M, F16's ``--no-self-play`` where the parent self-plays): ``a3_mx`` -- a main
  exploiter of P* (seeds 3511-3513) -- and ``a6_lx`` -- a league exploiter of P* + the archive, share 1.0, stable
  PFSP (seeds 3521-3523). Between the phases, ``phase1-record`` writes ONE write-once record of the three
  exploiter finals (path, step, sha256), refusing a missing or short final. Phase 2 (+20M): ``a3`` / ``a6`` carry
  those finals as ``--stable-opponents`` at share 0.4 with ``--stable-opponent-pfsp`` (the registered round-1 B
  recipe), seeds 3501-3505, b-OFF = C0;
* toy (``--behaviour-check fatal``, as X34; +70M; the X34 toy extras): ``toy_c0``, ``toy_a1``, ``toy_a2``.

    python -m main.x35.kit describe --arm a1 --seed 3501 --call <R_production.json> [--phase1 <record>]
    python -m main.x35.kit phase1-record --arm a3 --call <R_production.json> --out <stage>/phase1_a3.json
    python -m main.x35.kit verify-phase1 <stage>/phase1_a3.json
    python -m main.x35.kit live                    # exit 1 while any training run is live (pre-check 1)
    python -m main.x35.kit disk --required-gib 20.68 [--thinning]   # pre-check 8 (F22)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.x35.refpop import RefPopError, sha256_file
from main.x35.refpop import verify as verify_call

L_PROD = 20_000_000
L_EXPLOITER = 5_000_000
L_TOY = 70_000_000
CKPT_EVERY = "1000000"
#: F22: the launcher's REQUIRED headroom for one 20M production fork (dry-run, 2026-10-17).
FORK_GIB = 20.68
PHASE1_SCHEMA = "x35_phase1_exploiters_v1"
TOY_EXTRAS = ["--snapshot-ladder-games", "0", "--supply-starve-cycles", "self_play_pool=0"]


@dataclass(frozen=True)
class Arm:
    name: str
    side: str            # the call record's side: production | toy
    length: int
    seeds: Tuple[int, ...]
    run: str             # run-name format
    phase: int           # 0 = single phase, 1 = exploiters, 2 = folded in
    behaviour: str
    phase1_of: Optional[str] = None  # phase 2: the phase-1 arm whose record it reads


PROD_SEEDS = tuple(range(3501, 3506))
TOY_SEEDS = tuple(range(3601, 3606))
ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("c0", "production", L_PROD, PROD_SEEDS, "rb_x35_c0_s{seed}", 0, "warn"),
    Arm("a1", "production", L_PROD, PROD_SEEDS, "rb_x35_a1arch_s{seed}", 0, "warn"),
    Arm("a2", "production", L_PROD, PROD_SEEDS, "rb_x35_a2pfsp_s{seed}", 0, "warn"),
    Arm("a3_mx", "production", L_EXPLOITER, (3511, 3512, 3513), "rb_x35_mx_s{seed}", 1, "warn"),
    Arm("a3", "production", L_PROD, PROD_SEEDS, "rb_x35_a3mx_s{seed}", 2, "warn", phase1_of="a3_mx"),
    Arm("a6_lx", "production", L_EXPLOITER, (3521, 3522, 3523), "rb_x35_lx_s{seed}", 1, "warn"),
    Arm("a6", "production", L_PROD, PROD_SEEDS, "rb_x35_a6lx_s{seed}", 2, "warn", phase1_of="a6_lx"),
    Arm("toy_c0", "toy", L_TOY, TOY_SEEDS, "rb_x35t_c0_s{seed}", 0, "fatal"),
    Arm("toy_a1", "toy", L_TOY, TOY_SEEDS, "rb_x35t_a1arch_s{seed}", 0, "fatal"),
    Arm("toy_a2", "toy", L_TOY, TOY_SEEDS, "rb_x35t_a2pfsp_s{seed}", 0, "fatal"),
)}
#: STAGED_PLAN §4's batch-1 forks (R0 and the readers have no registered argv yet: a T55 FINDING).
BATCH1: Tuple[Tuple[str, int], ...] = (
    ("a1", 3501), ("c0", 3501), ("c0", 3502), ("a1", 3502), ("a1", 3503), ("c0", 3503),
    ("a2", 3501), ("a2", 3502), ("a2", 3503),
    ("toy_a1", 3601), ("toy_c0", 3601), ("toy_c0", 3602), ("toy_a1", 3602), ("toy_a1", 3603), ("toy_c0", 3603),
    ("toy_a2", 3601), ("toy_a2", 3602), ("toy_a2", 3603),
)


class KitError(ValueError):
    """A precondition of the kit does not hold (named; the launch script HOLDs on it)."""


def _arm(name: str, seed: int) -> Arm:
    if name not in ARMS:
        raise KitError(f"arm {name!r} is not one of {sorted(ARMS)}")
    a = ARMS[name]
    if seed not in a.seeds:
        raise KitError(f"seed {seed} is not registered for {name} ({list(a.seeds)})")
    return a


def archive_spec(call: Mapping[str, Any]) -> str:
    """A1's ``--stable-opponents`` value: N_8 … N_1 as explicit ``<zip>:<label>`` (§5.6.3: 8 unique labels)."""
    arch = sorted(call["archive"], key=lambda e: -int(e["k"]))
    return ",".join(f"{e['path']}:{e['label']}" for e in arch)


def run_name(arm: str, seed: int) -> str:
    return _arm(arm, seed).run.format(seed=seed)


def build_argv(arm: str, seed: int, call: Mapping[str, Any], phase1: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The trainer argv (the launcher's own flags excluded) of one arm × seed."""
    a = _arm(arm, seed)
    if call.get("side") != a.side:
        raise KitError(f"arm {arm} reads the {a.side} call record, got side {call.get('side')!r}")
    p = call["pstar"]
    start, step = str(p["path"]), int(p["step"])
    head = ["--model", start]
    lever: List[str] = []
    if arm == "a3_mx":
        head += ["--exploiter", start, "--no-self-play"]
    elif arm == "a6_lx":
        prefix = str(p.get("label") or "x26_pstar")
        lever = ["--stable-opponents", f"{start}:{prefix}," + archive_spec(call),
                 "--stable-opponent-selfplay-share", "1.0", "--stable-opponent-pfsp"]
    elif arm in ("a1", "toy_a1"):
        lever = ["--stable-opponents", archive_spec(call), "--stable-opponent-selfplay-share", "0.5"]
    elif arm in ("a2", "toy_a2"):
        lever = ["--pfsp-scale", "2.5", "--n-sentinels", "10"]
    elif a.phase == 2:
        if phase1 is None:
            raise KitError(f"arm {arm} is phase 2: it needs the {a.phase1_of} phase-1 record")
        if phase1.get("arm") != a.phase1_of or phase1.get("pstar_sha256") != p["sha256"]:
            raise KitError(f"the phase-1 record is for {phase1.get('arm')} from P* "
                           f"{str(phase1.get('pstar_sha256'))[:12]}, not {a.phase1_of} from {str(p['sha256'])[:12]}")
        lever = ["--stable-opponents", ",".join(f"{e['path']}:{e['label']}" for e in phase1["exploiters"]),
                 "--stable-opponent-selfplay-share", "0.4", "--stable-opponent-pfsp"]
    out = head + ["--behaviour-check", a.behaviour, "--steps", str(step + a.length),
                  "--checkpoint-every-steps", CKPT_EVERY]
    if a.side == "toy":
        out += TOY_EXTRAS
    return out + ["--device", "cuda"] + lever + ["--seed", str(seed), "--run-name", run_name(arm, seed)]


def parent_run_dir(call: Mapping[str, Any]) -> str:
    """The run the fork's dry-run must name: the start node's run directory."""
    p = Path(str(call["pstar"]["path"]))
    return str(p.parent.parent if p.parent.name == "checkpoints" else p.parent)


def describe(arm: str, seed: int, call: Mapping[str, Any],
             phase1: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    a = _arm(arm, seed)
    argv = build_argv(arm, seed, call, phase1)
    return {"arm": arm, "seed": seed, "side": a.side, "phase": a.phase, "run_name": run_name(arm, seed),
            "parent_run_dir": parent_run_dir(call), "delta_steps": a.length, "delta_text": f"+{a.length:,} steps",
            "argv": argv, "argv_shell": " ".join(shlex.quote(x) for x in argv)}


# ------------------------------------------------------------------------------------------- the phase-1 record
def _final_step(zip_path: str) -> int:
    from agents.training.fixed_opponent_pool import resolve_model_ref

    try:
        r = resolve_model_ref(zip_path, warn=False)
    except (FileNotFoundError, ValueError) as exc:
        raise KitError(f"{zip_path}: does not resolve as a model ({exc})") from exc
    if r.num_timesteps is None:
        raise KitError(f"{zip_path}: no recorded step")
    return int(r.num_timesteps)


def phase1_record(arm: str, call: Mapping[str, Any], models_dir: str, out: str,
                  now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Freeze the three exploiter finals of a phase-1 arm (``a3_mx`` / ``a6_lx``); write-once."""
    a = ARMS.get(arm)
    if a is None or a.phase != 1:
        raise KitError(f"{arm!r} is not a phase-1 arm ({[n for n, x in ARMS.items() if x.phase == 1]})")
    if os.path.exists(out):
        raise KitError(f"{out} exists: a phase-1 record is written ONCE")
    p = call["pstar"]
    want = int(p["step"]) + a.length
    ex: List[Dict[str, Any]] = []
    finals = [(run_name(arm, s), Path(models_dir) / run_name(arm, s) / "final_model.zip") for s in a.seeds]
    for rn, z in finals:
        if not z.is_file():
            raise KitError(f"{rn}: no final_model.zip (an interrupted or running exploiter is not a phase-1 final)")
    for rn, z in finals:
        step = _final_step(str(z))
        if step < want:
            raise KitError(f"{rn}: final at {step:,} < the registered {want:,} (P* + {a.length:,})")
        ex.append({"label": rn.replace("rb_x35_", ""), "run": rn, "path": str(z.resolve()), "step": step,
                   "sha256": sha256_file(z)})
    t = now or dt.datetime.now(dt.timezone.utc)
    doc = {"schema": PHASE1_SCHEMA, "arm": arm, "pstar_sha256": p["sha256"], "pstar_step": int(p["step"]),
           "written_at": t.isoformat(), "exploiters": ex}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def verify_phase1(path: str) -> List[str]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: unreadable ({exc})"]
    probs = [] if doc.get("schema") == PHASE1_SCHEMA else [f"schema {doc.get('schema')!r}"]
    if len(doc.get("exploiters") or []) != 3:
        probs.append("a phase-1 record holds exactly 3 exploiters")
    for e in doc.get("exploiters") or []:
        if not os.path.exists(e["path"]):
            probs.append(f"{e['path']}: missing")
        elif sha256_file(e["path"]) != e["sha256"]:
            probs.append(f"{e['path']}: sha256 changed since the record")
    return probs


# ---------------------------------------------------------------------------------- the post-spawn pool check
#: Arms whose fork does NOT self-play, so nothing is seeded (F16: the main exploiter types --no-self-play).
NO_POOL_ARMS = frozenset({"a3_mx"})
POOL_SEED_SCHEMA = 2
_FILTER_RE = re.compile(r"\[step filter <= (\d+): (?:DROPPED (\d+) later snapshot\(s\) at steps \[([0-9, ]*)\]"
                        r"|none later)")
PENDING = "PENDING"


def check_pool_seed(arm: str, run_dir: str, call: Mapping[str, Any], log_text: str,
                    fork_num_timesteps: Optional[int]) -> Tuple[str, List[str]]:
    """T54's post-spawn check (``--dry-run`` cannot preview it: the seed runs in the child). ``(status, problems)``,
    status PASS / FAIL / PENDING (the record is not written yet). FAIL iff ``pool_seed.json`` is not schema 2, its
    ``fork_step`` is NULL (T54 only WARNS and seeds unfiltered when it cannot read the step: a HOLD here) or differs
    from P*'s ``num_timesteps`` (and the call record's step), a seeded snapshot is later than it, or the startup
    line "[step filter <= N: …]" disagrees with the record (or says the filter is OFF)."""
    from agents.training.pool_seed import SEED_RECORD_FILE, pool_dir_for, snapshot_step

    if arm in NO_POOL_ARMS:
        return "PASS", []
    p = Path(pool_dir_for(run_dir)) / SEED_RECORD_FILE
    if not p.is_file():
        return PENDING, [f"{p} not written yet"]
    try:
        rec = json.loads(p.read_text())
    except ValueError as exc:
        return PENDING, [f"{p}: not yet parseable ({exc})"]
    probs: List[str] = []
    if rec.get("schema") != POOL_SEED_SCHEMA:
        probs.append(f"pool_seed.json schema {rec.get('schema')!r} != {POOL_SEED_SCHEMA} (a pin without T54?)")
    fs = rec.get("fork_step")
    want = int(call["pstar"]["step"])
    if fs is None:
        probs.append("fork_step is NULL: T54 could not read the fork step and seeded UNFILTERED")
    else:
        if fork_num_timesteps is None or int(fs) != int(fork_num_timesteps):
            probs.append(f"fork_step {fs} != P*'s num_timesteps {fork_num_timesteps}")
        if int(fs) != want:
            probs.append(f"fork_step {fs} != the call record's P* step {want}")
        late = [n for n in rec.get("snapshots") or [] if (st := snapshot_step(n)) is None or st > int(fs)]
        if late:
            probs.append(f"seeded snapshot(s) later than fork_step {fs} (or unnamed): {late}")
    if "step filter OFF" in log_text:
        probs.append("the startup line says the step filter is OFF")
    m = _FILTER_RE.findall(log_text)
    if not m:
        if not probs:
            return PENDING, ["the startup '[step filter <= N: …]' line is not in the log yet"]
    else:
        n, k, steps = m[-1]
        logged = sorted(int(x) for x in steps.replace(" ", "").split(",") if x) if k else []
        if fs is not None and int(n) != int(fs):
            probs.append(f"the startup line's step filter {n} != the record's fork_step {fs}")
        if logged != sorted(rec.get("dropped_steps") or []) or (k and int(k) != len(logged)):
            probs.append(f"the startup line's dropped steps {logged} != the record's {rec.get('dropped_steps')}")
    return ("FAIL" if probs else "PASS"), probs


# ------------------------------------------------------------------------------------------------ pre-checks
def live_runs() -> Dict[str, int]:
    from utils.procfs import live_training_runs

    return live_training_runs()


def disk_check(models_dir: str, required_gib: float, *, thinning: bool,
               existing: Optional[Sequence[str]] = None, free_gib: Optional[float] = None) -> Tuple[bool, str]:
    """Pre-check 8 (F22): free disk covers EVERY not-yet-created batch-1 fork at ``required_gib`` each, or the
    operator declares ``thinning`` (finished arms are being thinned to their 10M nodes + final)."""
    have = set(existing) if existing is not None else {d for d in os.listdir(models_dir)
                                                        if d.startswith("rb_x35")}
    remaining = [run_name(a, s) for a, s in BATCH1 if run_name(a, s) not in have]
    free = free_gib if free_gib is not None else shutil.disk_usage(models_dir).free / 2 ** 30
    need = len(remaining) * required_gib
    msg = f"{free:.1f} GiB free; {len(remaining)} batch-1 fork(s) to create x {required_gib:.2f} = {need:.1f} GiB"
    if free >= need:
        return True, msg
    if thinning and free >= required_gib:
        return True, msg + " -- short for the whole batch, THINNING declared (F22): this launch alone fits"
    return False, msg + (" -- not even this launch fits" if free < required_gib else
                         " -- thin finished arms (F22) or pass --thinning")


def _load(path: str) -> Dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.kit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--arm", required=True, choices=sorted(ARMS))
    d.add_argument("--seed", required=True, type=int)
    d.add_argument("--call", required=True, help="the frozen R file of the arm's side (main.x35.refpop)")
    d.add_argument("--phase1", default=None, help="the phase-1 record (phase-2 arms)")
    d.add_argument("--field", default=None, help="print one field (argv = one token per line, argv_shell, "
                                                 "run_name, parent_run_dir, delta_text, side, phase)")
    r = sub.add_parser("phase1-record")
    r.add_argument("--arm", required=True, choices=("a3", "a6"))
    r.add_argument("--call", required=True)
    r.add_argument("--models-dir", default=None)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verify-phase1")
    v.add_argument("file")
    sub.add_parser("live")
    q = sub.add_parser("pool-seed-check", help="T54's post-spawn check: exit 0 PASS, 1 FAIL, 3 PENDING")
    q.add_argument("--arm", required=True, choices=sorted(ARMS))
    q.add_argument("--run-dir", required=True)
    q.add_argument("--call", required=True)
    q.add_argument("--log", action="append", default=[], help="a log holding the startup line (repeatable)")
    k = sub.add_parser("disk")
    k.add_argument("--required-gib", type=float, default=FORK_GIB)
    k.add_argument("--models-dir", default=None)
    k.add_argument("--thinning", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "describe":
            call = _load(a.call)
            probs = verify_call(a.call)
            if probs:
                raise KitError(f"the call record {a.call} does not verify: {probs}")
            doc = describe(a.arm, a.seed, call, _load(a.phase1) if a.phase1 else None)
            if a.field == "argv":
                print("\n".join(doc["argv"]))  # one token per line: the launch script's mapfile
            else:
                print(doc[a.field] if a.field else json.dumps(doc, indent=1))
            return 0
        if a.cmd == "phase1-record":
            from utils.paths import main_models_dir

            md = a.models_dir or str(main_models_dir() or "")
            doc = phase1_record(f"{a.arm}_{'mx' if a.arm == 'a3' else 'lx'}", _load(a.call), md, a.out)
            print(f"[x35 kit] phase-1 record {a.out}: " + ", ".join(e["label"] for e in doc["exploiters"]))
            return 0
        if a.cmd == "verify-phase1":
            probs = verify_phase1(a.file)
            for x in probs:
                print(f"[x35 kit] {x}", file=sys.stderr)
            return 0 if not probs else 1
        if a.cmd == "pool-seed-check":
            from agents.training.lineage import checkpoint_num_timesteps

            call = _load(a.call)
            text = "".join(Path(x).read_text(errors="replace") for x in a.log if os.path.isfile(x))
            status, probs = check_pool_seed(a.arm, a.run_dir, call, text,
                                            checkpoint_num_timesteps(str(call["pstar"]["path"])))
            print(f"[x35 kit] pool seed: {status}" + (f": {probs}" if probs else ""))
            return {"PASS": 0, "FAIL": 1}.get(status, 3)
        if a.cmd == "live":
            live = live_runs()
            for rd, pid in sorted(live.items()):
                print(f"LIVE {rd} pid {pid}")
            return 1 if live else 0
        from utils.paths import main_models_dir

        ok, msg = disk_check(a.models_dir or str(main_models_dir() or "."), a.required_gib, thinning=a.thinning)
        print(f"[x35 kit] disk: {msg}")
        return 0 if ok else 1
    except (KitError, RefPopError, FileNotFoundError, ValueError) as exc:
        print(f"[x35 kit] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
