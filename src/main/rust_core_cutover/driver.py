"""The cutover stress driver: `run` (the scheduler), `unit` (one unit, one process), `status`.

Durable layout under ``--out`` (never /tmp):

* ``registration.json`` — the plan's streams, targets and the sizes they were derived from, written
  once; a resume whose plan differs REFUSES (a new target is a new registration, not an edit).
* ``units/<stream>.<i>.json`` — one row per finished unit, written atomically (tmp + rename).
* ``divergences/`` — every divergent battle's input log (re-runnable alone), ``fuzz/`` — each fuzz
  unit's run dir (repros included), ``logs/`` — each unit process's stdout/stderr.
* ``control.json`` — the operator's knobs, re-read every loop: ``cap`` (max concurrent units),
  ``pause`` (start nothing), ``streams`` (an optional allow-list), ``stop`` (drain and exit).
* ``timeline.jsonl`` — ``(t, running, cap, why)`` at every change (the governor's classifier input).
* ``governor.json`` — the training-fps rows and every throttle event.
* ``pin.json`` — the commit the pin was exported from (every row is stamped with it).

Every unit process runs ``nice -n 19`` with one torch / BLAS thread.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from main.rust_core_cutover import plan as PL

DEFAULT_CAP = 6
POLL_S = 5.0
GOVERN_EVERY_S = 30.0
UNIT_TIMEOUT_S = 3 * 3600


# ---------------------------------------------------------------------------------------------
# registration, pin, rows
# ---------------------------------------------------------------------------------------------

def sizes() -> Dict[str, int]:
    from utils import ladder_corpus
    from utils.team_sources import team_list

    m = ladder_corpus.manifest()
    return {"pool_n": len(team_list("pool")), "ladder_full_n": m["tiers"][PL.LADDER_FULL_TIER]["n"],
            "ladder_policy_n": m["tiers"][PL.LADDER_POLICY_TIER]["n"]}


def registration(sz: Dict[str, int]) -> dict:
    return {"schema": PL.SCHEMA, "sizes": sz,
            "streams": [{"name": s.name, "kind": s.kind, "battles": s.battles, "per_unit": s.per_unit,
                         "units": s.units, "target": s.target, "params": s.params,
                         "requires": s.requires} for s in PL.streams(**sz)]}


def write_registration(out: Path, sz: Dict[str, int]) -> dict:
    reg = json.loads(json.dumps(registration(sz)))
    path = out / "registration.json"
    if path.exists():
        have = json.loads(path.read_text())
        if have != reg:
            raise SystemExit(f"{path} records a different plan — refusing to mix registrations")
        return have
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return reg


def pin_info() -> dict:
    """The commit this code was exported from (``PIN`` beside ``src/``), or git's HEAD."""
    from utils.paths import repo_root

    f = repo_root() / "PIN"
    if f.exists():
        return {"commit": f.read_text().strip(), "root": str(repo_root()), "kind": "export"}
    from utils.git import get_git_hash  # a worktree / checkout: stamped, but not a durable pin

    return {"commit": get_git_hash(), "root": str(repo_root()), "kind": "checkout"}


def capabilities() -> List[str]:
    caps = []
    try:
        from agents.training import gen3_env

        if "core" in getattr(gen3_env, "OBS_SOURCES", ()):
            caps.append("obs_source_core")
    except Exception:
        pass
    return caps


def row_path(out: Path, uid: str) -> Path:
    return out / "units" / f"{uid}.json"


def done_units(out: Path) -> set:
    d = out / "units"
    return {p.stem for p in d.glob("*.json")} if d.exists() else set()


def load_rows(out: Path) -> List[dict]:
    d = out / "units"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def write_row(out: Path, row: dict) -> None:
    path = row_path(out, row["unit"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, sort_keys=True, default=repr))
    os.replace(tmp, path)


def read_control(out: Path) -> dict:
    try:
        return json.loads((out / "control.json").read_text())
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------------------------
# the unit subcommand
# ---------------------------------------------------------------------------------------------

def run_one(out: Path, uid: str) -> int:
    from main.rust_core_cutover.units import run_unit

    reg = json.loads((out / "registration.json").read_text())
    name, i = PL.parse_unit_id(uid)
    stream = {s.name: s for s in PL.streams(**reg["sizes"])}[name]
    row = run_unit(stream, i, out)
    row["pin"] = pin_info()
    write_row(out, row)
    return 0


# ---------------------------------------------------------------------------------------------
# the scheduler
# ---------------------------------------------------------------------------------------------

def next_unit(streams: List[PL.Stream], done: set, running: set, allowed: Optional[List[str]],
              caps: List[str]) -> Optional[str]:
    """The next unit: from the runnable stream with the LOWEST fraction (done + running) / units,
    so every stream advances together and coverage grows everywhere at once."""
    best, best_frac = None, None
    for s in streams:
        if allowed is not None and s.name not in allowed:
            continue
        if s.requires is not None and s.requires not in caps:
            continue
        taken = [i for i in range(s.units) if PL.unit_id(s.name, i) in done | running]
        if len(taken) >= s.units:
            continue
        frac = len(taken) / s.units
        if best_frac is None or frac < best_frac:
            nxt = next(i for i in range(s.units) if PL.unit_id(s.name, i) not in done | running)
            best, best_frac = PL.unit_id(s.name, nxt), frac
    return best


def _unit_env() -> dict:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               POKESIM_EMISSION_SELFCHECK="1", PYTHONUNBUFFERED="1")
    return env


class _Adopted:
    """A unit process a PREVIOUS driver started that is still alive (units run in their own
    session, so killing the driver alone does not kill them): watched by pid, never re-spawned."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> Optional[int]:
        return None if Path(f"/proc/{self.pid}").exists() else 0

    def wait(self) -> int:
        while self.poll() is None:
            time.sleep(1)
        return 0


def _alive_unit(pid: int, uid: str) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"main.rust_core_cutover" in argv and uid.encode() in argv


class Driver:
    def __init__(self, out: Path, cap: int):
        from main.rust_core_cutover.governor import Governor

        self.out = out
        sz = sizes()
        self.reg = write_registration(out, sz)
        self.streams = PL.streams(**sz)
        self.caps = capabilities()
        self.governor = Governor(out)
        self.running: Dict[str, object] = {}
        self.started_at: Dict[str, float] = {}
        marks = out / "running"
        marks.mkdir(parents=True, exist_ok=True)
        for m in marks.glob("*.pid"):
            uid, pid = m.stem, int(m.read_text())
            if _alive_unit(pid, uid) and not row_path(out, uid).exists():
                self.running[uid], self.started_at[uid] = _Adopted(pid), m.stat().st_mtime
            else:
                m.unlink()
        ctl = read_control(out)
        if "cap" not in ctl:
            ctl["cap"] = cap
            self._write_control(ctl)
        self.timeline: List[tuple] = []
        tl = out / "timeline.jsonl"
        if tl.exists():
            self.timeline = [tuple(json.loads(x)[:2]) for x in tl.read_text().splitlines() if x.strip()]
        first = not (out / "pin.json").exists()
        (out / "pin.json").write_text(json.dumps({**pin_info(), "caps": self.caps}, indent=1) + "\n")
        from main.rust_core_cutover import governor as G

        now = time.time()
        # The FIRST start opens with the pre-stress OFF window; a resumed driver re-reads its
        # window from disk. A window is (length, the time the stress had drained, why).
        if first:
            self.off_len, self.off_started, self.off_why = float(G.FIRST_OFF_S), None, "pre-stress baseline"
        else:
            self.off_len = float(ctl.get("off_len", 0))
            self.off_started = ctl.get("off_started")
            self.off_why = ctl.get("off_why", "")
        self.next_off = float(ctl.get("next_off", now + G.OFF_EVERY_S))
        self._last_state = None

    def in_off(self, now: float) -> bool:
        if self.off_len <= 0:
            return False
        if self.off_started is not None and now >= self.off_started + self.off_len:
            self.log(f"OFF window closed: {self.off_why}")
            self.off_len, self.off_started = 0.0, None
            return False
        return True

    def open_off(self, length: float, why: str) -> None:
        self.off_len, self.off_started, self.off_why = float(length), None, why
        self.log(f"OFF window opened ({length:.0f}s once drained): {why}")

    def _write_control(self, ctl: dict) -> None:
        tmp = self.out / "control.json.tmp"
        tmp.write_text(json.dumps(ctl, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.out / "control.json")

    def log(self, msg: str) -> None:
        with open(self.out / "driver.log", "a") as fh:
            fh.write(f"{time.strftime('%F %T')} {msg}\n")

    def _mark(self, cap: int, why: str) -> None:
        state = (len(self.running), cap, why)
        if state == self._last_state:
            return
        self._last_state = state
        t = time.time()
        self.timeline.append((t, len(self.running)))
        with open(self.out / "timeline.jsonl", "a") as fh:
            fh.write(json.dumps([t, len(self.running), cap, why]) + "\n")

    def _spawn(self, uid: str) -> None:
        (self.out / "logs").mkdir(exist_ok=True)
        log = open(self.out / "logs" / f"{uid}.log", "w")
        argv = ["nice", "-n", "19", sys.executable, "-m", "main.rust_core_cutover", "unit",
                "--out", str(self.out), "--unit", uid]
        self.running[uid] = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, env=_unit_env(),
                                             start_new_session=True)
        self.started_at[uid] = time.time()
        (self.out / "running" / f"{uid}.pid").write_text(str(self.running[uid].pid))
        self.log(f"start {uid} pid={self.running[uid].pid}")

    def _reap(self) -> None:
        for uid, p in list(self.running.items()):
            rc = p.poll()
            if rc is None and time.time() - self.started_at[uid] > UNIT_TIMEOUT_S:
                self.log(f"TIMEOUT {uid} after {UNIT_TIMEOUT_S}s — killing pgid {p.pid}")
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
            if rc is None:
                continue
            del self.running[uid]
            (self.out / "running" / f"{uid}.pid").unlink(missing_ok=True)
            ok = row_path(self.out, uid).exists()
            self.log(f"exit {uid} rc={rc} row={'yes' if ok else 'NO'}")
            if not ok:  # a unit that died without its row: record it (a finding), never loop on it
                write_row(self.out, {"schema": PL.SCHEMA, "unit": uid, "stream": PL.parse_unit_id(uid)[0],
                                     "i": PL.parse_unit_id(uid)[1], "status": "error", "battles": 0,
                                     "errors": [{"error": f"unit process exited {rc} without a row "
                                                 f"(see logs/{uid}.log)"}],
                                     "divergent": [], "known": [], "pin": pin_info(),
                                     "finished": time.time()})

    def loop(self) -> None:
        from main.rust_core_cutover import governor as G

        self.log(f"driver pid={os.getpid()} pin={pin_info()['commit']} caps={self.caps} "
                 f"done={len(done_units(self.out))}")
        last_gov = 0.0
        while True:
            self._reap()
            ctl = read_control(self.out)
            cap = int(ctl.get("cap", DEFAULT_CAP))
            now = time.time()
            if now - last_gov >= GOVERN_EVERY_S:
                last_gov = now
                self.governor.poll(self.timeline)
                if not self.in_off(now):
                    newcap = self.governor.decide(cap, now)
                    if newcap != cap:
                        self.log(f"THROTTLE cap {cap} -> {newcap} ({self.governor.events[-1]})")
                        ctl["cap"] = cap = newcap
                    fresh = self.governor.needs_baseline()
                    if fresh:
                        self.open_off(G.OFF_S, f"new run(s) need a baseline: {fresh}")
                    elif now >= self.next_off:
                        self.open_off(G.OFF_S, "periodic control window")
                        self.next_off = now + G.OFF_EVERY_S
                self.governor.save()
                ctl.update(off_len=self.off_len, off_started=self.off_started, next_off=self.next_off,
                           off_why=self.off_why)
                self._write_control(ctl)
            if ctl.get("stop"):
                if not self.running:
                    self.log("stop requested; drained; exiting")
                    self._mark(0, "stopped")
                    return
            in_off = self.in_off(now)
            if in_off and not self.running and self.off_started is None:
                self.off_started = now   # the window's clock starts once the stress has drained
                self.log(f"OFF window drained, {self.off_len:.0f}s from now: {self.off_why}")
            done = done_units(self.out)
            if not (ctl.get("pause") or ctl.get("stop") or in_off):
                while len(self.running) < cap:
                    uid = next_unit(self.streams, done, set(self.running), ctl.get("streams"), self.caps)
                    if uid is None:
                        break
                    self._spawn(uid)
            why = "stop" if ctl.get("stop") else "pause" if ctl.get("pause") else \
                f"off: {self.off_why}" if in_off else "on"
            self._mark(cap, why)
            if not self.running and next_unit(self.streams, done, set(), ctl.get("streams"), self.caps) is None:
                self.log("every runnable unit is done; exiting")
                self._mark(0, "finished")
                return
            time.sleep(POLL_S)


# ---------------------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------------------

def status(out: Path) -> dict:
    from main.rust_core_cutover.governor import Governor

    reg = json.loads((out / "registration.json").read_text())
    rows = load_rows(out)
    by_stream: Dict[str, List[dict]] = {}
    for r in rows:
        by_stream.setdefault(r["stream"], []).append(r)
    streams = []
    for s in reg["streams"]:
        rs = by_stream.get(s["name"], [])
        dec = {k: sum(r.get("totals", {}).get(k, {}).get("decisions", 0) for r in rs) for k in ("V", "T", "O")}
        streams.append({
            "stream": s["name"], "units": f"{len(rs)}/{s['units']}",
            "battles": sum(r.get("battles", 0) for r in rs), "registered": s["battles"],
            "decisions_O": dec["O"], "divergent": sum(len(r.get("divergent", [])) for r in rs),
            "known": sum(len(r.get("known", [])) for r in rs),
            "errors": sum(len(r.get("errors", [])) for r in rs),
        })
    ladder_cov = {}
    for name in ("ladder_full_a", "ladder_full_b"):
        seen = set()
        for r in by_stream.get(name, []):
            for a, b in r.get("covered", []):
                seen.update((a, b))
        ladder_cov[name] = len(seen)
    from main.rust_core_cutover import verdict as VD

    div = [d for r in rows for d in r.get("divergent", []) if "classes" in d]
    soak = []
    for r in by_stream.get("soak_transport", []) + by_stream.get("soak_core_obs", []):
        smp = [x for x in r.get("rss_samples", []) if x[2] is not None]
        soak.append({"unit": r["unit"], "episodes": r.get("battles"), "replacements": r.get("child_replacements"),
                     "child_rss_kib": [smp[0][2], smp[-1][2]] if smp else None,
                     "env_rss_kib": [smp[0][3], smp[-1][3]] if smp else None,
                     "errors": len(r.get("errors", []))})
    fuzz_hard = sum(int(r.get("hard_failures") or 0) for r in rows if r.get("kind") == "fuzz")
    return {"pin": json.loads((out / "pin.json").read_text()) if (out / "pin.json").exists() else None,
            "verdict_census": VD.census(div), "fuzz_hard_failures": fuzz_hard, "soak": soak,
            "error_units": [r["unit"] for r in rows if r.get("status") == "error"][:50],
            "control": read_control(out), "streams": streams,
            "ladder_teams_covered": ladder_cov, "ladder_full_n": reg["sizes"]["ladder_full_n"],
            "training_fps": Governor(out).report(), "throttle_events": Governor(out).events[-10:]}


def render_status(s: dict) -> str:
    lines = [f"PROGRESS (not a verdict) — pin {s['pin']['commit'][:12] if s['pin'] else '?'}; "
             f"control {s['control']}"]
    lines.append(f"{'stream':22s} {'units':>11s} {'battles':>9s} {'of':>7s} {'O-decisions':>12s} "
                 f"{'divergent':>9s} {'known':>6s} {'errors':>6s}")
    for r in s["streams"]:
        lines.append(f"{r['stream']:22s} {r['units']:>11s} {r['battles']:>9d} {r['registered']:>7d} "
                     f"{r['decisions_O']:>12d} {r['divergent']:>9d} {r['known']:>6d} {r['errors']:>6d}")
    lines.append(f"ladder full-tier teams covered: A {s['ladder_teams_covered'].get('ladder_full_a')} / "
                 f"B {s['ladder_teams_covered'].get('ladder_full_b')} of {s['ladder_full_n']}")
    for cat, keys in sorted(s["verdict_census"].items()):
        lines.append(f"{cat}: {sum(v['battles'] for v in keys.values())} battle-classes")
        for k, v in sorted(keys.items(), key=lambda kv: -kv[1]["battles"])[:12]:
            lines.append(f"    {v['battles']:6d} battles {v['fields']:7d} fields  {k}   e.g. {v['example']}")
    lines.append(f"fuzz non-allowlisted failures: {s['fuzz_hard_failures']}; error units: {s['error_units'][:10]}")
    for x in s["soak"]:
        lines.append(f"soak {x}")
    for f in s["training_fps"]:
        lines.append(f"training fps {f}")
    for e in s["throttle_events"]:
        lines.append(f"THROTTLE {e}")
    return "\n".join(lines)


#: The self-check binaries a pin builds (every unit and fuzzer runs them), and the release
#: `sim_bridge` (the production build, for the throughput A/B and the soak's production row).
PIN_BINS = ("sim_bridge", "core_events", "search_driver", "ab_replay", "bridge_replay")


def make_pin(commit: str, root: Path) -> Path:
    """Export ``commit`` (``git archive``, NOT a worktree: nothing removes it when a worktree is
    cleaned up) under ``root/pins/<sha>``, borrow the MAIN checkout's Showdown submodule (refusing
    unless it is the commit the export pins), and build the self-check binaries into the export's
    own ``target/`` — their compile-time dex path is then the export's, which outlives every
    worktree."""
    from utils.paths import repo_root

    here = repo_root()
    sha = subprocess.run(["git", "-C", str(here), "rev-parse", commit], capture_output=True, text=True,
                         check=True).stdout.strip()
    dest = root / "pins" / sha[:12]
    if not (dest / "PIN").exists():
        dest.mkdir(parents=True, exist_ok=True)
        arch = subprocess.run(["git", "-C", str(here), "archive", sha], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=arch, check=True)
        want = subprocess.run(["git", "-C", str(here), "ls-tree", sha, "deps/pokemon-showdown"],
                              capture_output=True, text=True, check=True).stdout.split()[2]
        from utils.git import get_main_repo_root

        main_ps = Path(get_main_repo_root(cwd=str(here))) / "deps" / "pokemon-showdown"
        have = subprocess.run(["git", "-C", str(main_ps), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if have != want:
            raise SystemExit(f"main's deps/pokemon-showdown is {have}, the pin wants {want}")
        ps = dest / "deps" / "pokemon-showdown"
        if ps.exists() and not ps.is_symlink():
            ps.rmdir()
        ps.symlink_to(main_ps)
        (dest / "PIN").write_text(sha + "\n")
    crate = dest / "src" / "rust_sim"
    cargo = ["nice", "-n", "19", "cargo", "build", "--manifest-path", str(crate / "Cargo.toml")]
    subprocess.run(cargo + ["--profile", "selfcheck", "--features", "emission-selfcheck",
                            *[x for b in PIN_BINS for x in ("--bin", b)]], check=True)
    subprocess.run(cargo + ["--release", "--bin", "sim_bridge", "--bin", "core_events"], check=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The Rust core CUTOVER stress (program M6 §3).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pin", help="export a commit + build its binaries (run from a checkout)")
    pp.add_argument("--commit", default="HEAD")
    pp.add_argument("--root", type=Path, required=True)
    for name in ("run", "unit", "status", "register"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name == "run":
            p.add_argument("--cap", type=int, default=DEFAULT_CAP)
            p.add_argument("--detach", action="store_true")
        if name == "unit":
            p.add_argument("--unit", required=True)
        if name == "status":
            p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pin":
        print(make_pin(a.commit, a.root.expanduser().resolve()))
        return 0
    out = a.out.expanduser().resolve()
    if a.cmd == "unit":
        return run_one(out, a.unit)
    if a.cmd == "register":
        print(json.dumps(write_registration(out, sizes()), indent=1))
        return 0
    if a.cmd == "status":
        s = status(out)
        print(json.dumps(s, indent=1) if a.json else render_status(s))
        return 0
    if a.detach:
        out.mkdir(parents=True, exist_ok=True)
        pid_file = out / "driver.pid"
        if pid_file.exists():
            old = int(pid_file.read_text())
            if Path(f"/proc/{old}").exists():
                print(f"a driver is already running (pid {old})", file=sys.stderr)
                return 2
        argv2 = ["nice", "-n", "19", sys.executable, "-m", "main.rust_core_cutover", "run",
                 "--out", str(out), "--cap", str(a.cap)]
        with open(out / "driver.stdout", "a") as fh:
            p = subprocess.Popen(argv2, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        # `nice` execs the interpreter, so this pid IS the driver's
        pid_file.write_text(f"{p.pid}\n")
        print(f"detached driver pid {p.pid}; progress: python -m main.rust_core_cutover status --out {out}")
        return 0
    Driver(out, a.cap).loop()
    return 0
