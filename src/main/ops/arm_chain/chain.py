"""The chain's persistent state machine: one supervisor walks the items in order, one transition at a time.

ARM:     PENDING ─▶ PRECHECKED ─▶ GOING ─▶ LAUNCHED ─▶ SPAWNED ─▶ BANNER_OK ─▶ RUNNING ─▶ ENDED ─▶ COMPLETE
         ─▶ K9B_PASS ─▶ THINNED ─▶ RECORDED ─▶ DONE                                        (then the next item)
WINDOW:  PENDING ─▶ STARTED ─▶ DONE | DONE_FAILED (on_fail "continue"); a restart in STARTED HOLDs

Any step may raise a HOLD: the item keeps its phase, the hold is recorded with its reason, one ``HOLD:`` status line
is written, and the supervisor exits 3. Nothing after a HOLD runs. ``clear-hold --by WHO`` (the orchestrator's
decision, typed by the training session) removes it, and the next ``run`` RETRIES the step that held;
``abandon --by WHO`` marks a held item ``DONE_FAILED`` instead and the chain hands off past it.

Invariants the tests pin:

* **Never a second GO.** ``GOING`` is persisted BEFORE the spawn. A supervisor that resumes in ``GOING`` (it died
  mid-GO) looks for evidence of the launch (a live ``launch_x35.sh <arm> <seed> … GO`` process, the wrapper's START
  line past the armed offset, ``models/<run>``); with evidence it adopts the launch, without it HOLDs — it never
  re-issues the GO. Clearing
  that hold resets the item to ``PENDING`` (a fresh pre-check and dry run).
* **Triggers survive the supervisor.** The END / pool-seed / HOLD triggers and their offsets are persisted at GO,
  so a resumed supervisor sees an END written while nobody was watching (the 10-25 chain-swap gap), and never a
  line from before the GO.
* **One supervisor per chain** (an exclusive ``flock`` on the state file's lock for the supervisor's life): the
  10-26 duplicate-supervisor incident started from a waiter that read a live supervisor as dead.
* **A started item cannot change under the chain**: the state keeps each started item's spec snapshot; a chain
  file edited under a started item is refused (append new items freely).
* **The hand-off is the NEXT ITEM's own pin**, by construction: there is no chain-wide pin to inherit.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from utils import procfs

from main.ops.arm_chain import armend, postspawn, window
from main.ops.arm_chain.launch import Describe, Hold, X35Launcher
from main.ops.arm_chain.procwatch import ProcRef, alive, ref_from_pidfile
from main.ops.arm_chain.runner import Runner, SubprocessRunner
from main.ops.arm_chain.ship import Shipper, static_tier_runner
from main.ops.arm_chain.spec import ArmItem, ChainSpec, Item, WindowItem, as_dict
from main.ops.arm_chain.statuslog import StatusLog, Trigger, end_body, launch_tag

STATE_SCHEMA = "arm_chain_state_v1"
#: A dry run older than this is never GO'd on (only a supervisor restart between PRECHECKED and GO can age one).
PRECHECK_MAX_AGE_S = 600
ARM_PHASES = ("PENDING", "PRECHECKED", "GOING", "LAUNCHED", "SPAWNED", "BANNER_OK", "RUNNING", "ENDED", "COMPLETE",
              "K9B_PASS", "THINNED", "RECORDED", "DONE")
WINDOW_PHASES = ("PENDING", "STARTED", "DONE", "DONE_FAILED")
TERMINAL = ("DONE", "DONE_FAILED")
EXIT_OK, EXIT_HOLD, EXIT_BUSY = 0, 3, 4


class ChainError(RuntimeError):
    pass


def own_tag(spec: ChainSpec) -> str:
    return f"ARMCHAIN[{spec.name}]"


@dataclass
class Deps:
    """Every effect the chain has, as a seam (the tests replace each one)."""

    runner: Runner = field(default_factory=SubprocessRunner)
    base_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    proc: str = "/proc"
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], float] = time.time
    launcher: Optional[Callable[[ChainSpec], X35Launcher]] = None
    shipper: Optional[Callable[[ChainSpec], Shipper]] = None
    gpu_idle: window.Probe = window.nvidia_idle
    lease: Optional[Callable[[], str]] = None
    # the kit's GO re-runs its pre-checks (checkargs <= 10 min, the dry run <= 15 min), then waits 180 s, then the
    # pool-seed check (<= 30 min): a shorter bound HOLDs a healthy launch
    wrapper_spawn_timeout_s: float = 75 * 60
    end_timeout_s: float = 7 * 24 * 3600
    poll_s: float = 30.0


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


class Chain:
    def __init__(self, spec: ChainSpec, deps: Optional[Deps] = None):
        self.spec, self.d = spec, deps or Deps()
        os.makedirs(spec.work_dir, exist_ok=True)
        self.log = StatusLog(spec.status_file, own_tag(spec))
        self.state = self._load_state()
        self.launcher = (self.d.launcher or (lambda s: X35Launcher(s, self.d.runner, self.d.base_env)))(spec)

    # -- state -----------------------------------------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        p = self.spec.state_path
        if os.path.exists(p):
            st = json.loads(open(p).read())
            if st.get("schema") != STATE_SCHEMA or st.get("chain") != self.spec.name:
                raise ChainError(f"{p} is not this chain's state")
        else:
            st = {"schema": STATE_SCHEMA, "chain": self.spec.name, "items": {}, "history": []}
        for it in self.spec.items:
            rec = st["items"].setdefault(it.key, {"phase": "PENDING", "hold": None, "data": {}})
            if rec["phase"] != "PENDING" and rec.get("snapshot") is not None and rec["snapshot"] != as_dict(it):
                raise ChainError(f"{it.key} has started ({rec['phase']}) and its chain-file entry CHANGED — a started "
                                 "item never changes under the chain (append a new item instead)")
        return st

    def _save(self) -> None:
        p = self.spec.state_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)

    def rec(self, it: Item) -> Dict[str, Any]:
        return dict(self.state["items"][it.key])

    def _set(self, it: Item, phase: str, note: str = "", **data: Any) -> None:
        r = self.state["items"][it.key]
        old = r["phase"]
        r["phase"] = phase
        r["data"].update(data)
        if r.get("snapshot") is None:
            r["snapshot"] = as_dict(it)
        self.state["history"].append({"at": _iso(self.d.now()), "item": it.key, "from": old, "to": phase,
                                      "note": note[:300]})
        self._save()
        self.log.append(f"{it.key}: {old} -> {phase}" + (f": {note}" if note else ""))

    def _hold(self, it: Item, reason: str) -> int:
        r = self.state["items"][it.key]
        r["hold"] = {"reason": reason[:600], "at": _iso(self.d.now()), "phase": r["phase"]}
        self.state["history"].append({"at": _iso(self.d.now()), "item": it.key, "from": r["phase"], "to": "HOLD",
                                      "note": reason[:300]})
        self._save()
        self.log.append(f"HOLD: {it.key} at {r['phase']}: {reason}")
        return EXIT_HOLD

    def clear_hold(self, key: str, by: str, why: str) -> str:
        if not by.strip() or not why.strip():
            raise ChainError("clear-hold needs --by (whose decision) and --why")
        r = self.state["items"].get(key)
        if r is None or not r.get("hold"):
            raise ChainError(f"{key} holds nothing")
        old = r["hold"]
        r["hold"] = None
        reset = r["phase"] in ("GOING", "STARTED")
        if reset:
            r["phase"] = "PENDING"
            r["snapshot"] = None   # the item has not launched: an orchestrator's edit of it is taken, not refused
        self.state["history"].append({"at": _iso(self.d.now()), "item": key, "from": "HOLD",
                                      "to": r["phase"], "note": f"cleared by {by}: {why}"[:300]})
        self._save()
        msg = (f"{key}: hold cleared by {by} ({why}); was: {old['reason'][:160]}"
               + ("; reset to PENDING (a fresh pre-check, never a blind re-GO or re-run)" if reset else ""))
        self.log.append(msg)
        return msg

    def abandon(self, key: str, by: str, why: str) -> str:
        """The orchestrator's decision that a HELD item will not finish (a VOID arm it stopped, a window it drops):
        the item becomes ``DONE_FAILED`` and the chain hands off past it. Only a held item; nothing is signalled,
        nothing is thinned or shipped for it."""
        if not by.strip() or not why.strip():
            raise ChainError("abandon needs --by (whose decision) and --why")
        r = self.state["items"].get(key)
        if r is None or not r.get("hold"):
            raise ChainError(f"{key} is not HELD (only a held item can be abandoned)")
        it = next(i for i in self.spec.items if i.key == key)
        if isinstance(it, ArmItem) and "wrapper" in r["data"]:
            wref = ProcRef.from_json(r["data"]["wrapper"])
            lref = ref_from_pidfile(self.launcher.launcher_pidfile(it), self.d.proc)
            if alive(wref, self.d.proc) or (lref is not None and "main.launcher" in procfs.cmdline(lref.pid, self.d.proc)):
                raise ChainError(f"{key}: its launch wrapper or launcher is still ALIVE — abandoning it would let the "
                                 "next arm GO beside it; stop it first (the orchestrator's decision), then abandon")
        old = r["phase"]
        r["hold"], r["phase"] = None, "DONE_FAILED"
        self.state["history"].append({"at": _iso(self.d.now()), "item": key, "from": old, "to": "DONE_FAILED",
                                      "note": f"abandoned by {by}: {why}"[:300]})
        self._save()
        msg = f"{key}: ABANDONED at {old} by {by} ({why}); the chain hands off past it"
        self.log.append(msg)
        return msg

    # -- the supervisor ----------------------------------------------------------------------------------------------
    def run(self, max_items: Optional[int] = None) -> int:
        lock_path = self.spec.state_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return EXIT_BUSY
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            return self._run_locked(max_items)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _run_locked(self, max_items: Optional[int]) -> int:
        done_items = 0
        for it in self.spec.items:
            r = self.state["items"][it.key]
            if r["phase"] in TERMINAL:
                continue
            if r.get("hold"):
                self.log.append(f"{it.key} is HELD at {r['phase']} ({r['hold']['reason'][:120]}); nothing runs until "
                                "clear-hold")
                return EXIT_HOLD
            try:
                if isinstance(it, ArmItem):
                    while self.state["items"][it.key]["phase"] not in TERMINAL:
                        self._arm_step(it)
                else:
                    self._window(it)
            except Hold as h:
                return self._hold(it, str(h))
            except Exception as e:   # never die silently: the Monitor watches for HOLD lines, not tracebacks
                return self._hold(it, f"UNEXPECTED {type(e).__name__}: {e}")
            if self.state["items"][it.key].get("hold"):
                return EXIT_HOLD
            done_items += 1
            if max_items is not None and done_items >= max_items:
                return EXIT_OK
        self.log.append("chain COMPLETE: every item is DONE")
        return EXIT_OK

    # -- arm steps ---------------------------------------------------------------------------------------------------
    def _triggers(self, it: ArmItem) -> Dict[str, Trigger]:
        return {k: Trigger.from_json(v) for k, v in self.rec(it)["data"]["triggers"].items()}

    def _arm_step(self, it: ArmItem) -> None:
        phase = self.rec(it)["phase"]
        getattr(self, f"_arm_{phase.lower()}")(it)

    def _arm_pending(self, it: ArmItem) -> None:
        disk = armend.disk_ok(self.spec.models_dir, it.disk_need_gib, self.spec.disk_headroom_gib)
        d = self.launcher.prelaunch(it)
        m = re.search(r"checkpoint_(\d+)_steps\.zip$", d.argv[d.argv.index("--model") + 1]) if "--model" in d.argv else None
        if not m:
            raise Hold(f"{it.run}: the describe's --model is not a checkpoint (a chain arm is a FORK)")
        self._set(it, "PRECHECKED",
                  f"dry run PASS at pin {it.pin[:12]}: FORK of {d.parent_run_dir}, {d.delta_text}; {disk}",
                  prechecked_epoch=self.d.now(),
                  describe={"run_name": d.run_name, "parent_run_dir": d.parent_run_dir, "delta_text": d.delta_text,
                            "steps": d.steps, "argv": d.argv}, fork_step=int(m.group(1)))

    def _arm_prechecked(self, it: ArmItem) -> None:
        age = self.d.now() - float(self.rec(it)["data"].get("prechecked_epoch", 0))
        if age > PRECHECK_MAX_AGE_S:
            self._set(it, "PENDING", f"the dry run is {age / 60:.0f} min old (a supervisor restart): re-checking, never "
                                     "a GO on a stale dry run")
            return
        ok, why = self.d.gpu_idle()
        t0 = self.d.monotonic()
        while not ok:
            if self.d.monotonic() - t0 > 300:
                raise Hold(f"the GPU is not idle before the GO: {why} (a window step or the previous trainer left behind)")
            self.d.sleep(10)
            ok, why = self.d.gpu_idle()
        tag = launch_tag(it.arm, it.seed)
        trig = {"end": self.log.arm(f"END {it.run}", tag, end_body(it.run)),
                "start": self.log.arm(f"START {it.run}", tag, "START " + re.escape(it.run) + ": .*"),
                "spawned": self.log.arm(f"pool-seed PASS {it.run}", tag, r"post-spawn pool-seed check PASS: .*"),
                "held": self.log.arm(f"wrapper HOLD {it.run}", tag, r"HOLD: (.*)")}
        self._set(it, "GOING", f"GO at pin {it.pin[:12]} (GO by {it.go_by})",
                  triggers={k: t.to_json() for k, t in trig.items()}, go_epoch=self.d.now())
        ref = self.launcher.go(it)
        # the post-spawn budget starts at the SPAWN (the GO-time verify commands can take minutes before it)
        self._set(it, "LAUNCHED", f"launch wrapper pid {ref.pid}", wrapper=ref.to_json(), go_epoch=self.d.now())

    def _arm_going(self, it: ArmItem) -> None:
        """A supervisor died between persisting GOING and persisting LAUNCHED: adopt the launch, or HOLD."""
        tr = self._triggers(it)
        for t in tr.values():
            self.log.adopt(t)
        started = self.log.find(tr["start"]) is not None
        exists = os.path.exists(os.path.join(self.spec.models_dir, it.run))
        wref = self.launcher.find_go_wrapper(it, self.d.proc)
        lref = ref_from_pidfile(self.launcher.launcher_pidfile(it), self.d.proc)
        if wref is not None or started or exists:
            ref = wref or lref or ProcRef(0, -1)
            self._set(it, "LAUNCHED", f"adopted after a supervisor restart (GO wrapper "
                                      f"{'pid ' + str(wref.pid) if wref else 'gone'}, START line {started}, run dir "
                                      f"{exists}, launcher {'pid ' + str(lref.pid) if lref else 'not found'})",
                      wrapper=ref.to_json(), go_epoch=self.d.now())
            return
        raise Hold("the supervisor stopped mid-GO and finds no evidence of a launch (no GO wrapper process, no START "
                   "line, no run dir): verify by hand; clearing this hold resets the arm to PENDING (never a blind "
                   "re-GO)")

    def _wrapper(self, it: ArmItem) -> ProcRef:
        return ProcRef.from_json(self.rec(it)["data"]["wrapper"])

    def _adopt(self, it: ArmItem) -> Dict[str, Trigger]:
        tr = self._triggers(it)
        for t in tr.values():
            self.log.adopt(t)
        return tr

    def _arm_launched(self, it: ArmItem) -> None:
        tr = self._adopt(it)
        left = self.d.wrapper_spawn_timeout_s - (self.d.now() - self.rec(it)["data"]["go_epoch"])
        line = postspawn.wait_spawned(self.log, tr["spawned"], tr["held"], self._wrapper(it), max(left, 0.0),
                                      poll_s=min(self.d.poll_s, 15.0), proc=self.d.proc,
                                      clock=postspawn.Clock(self.d.sleep, self.d.monotonic))
        self._set(it, "SPAWNED", line.split("] ", 1)[-1][:200])

    def _arm_spawned(self, it: ArmItem) -> None:
        if it.banner is None:
            self._set(it, "BANNER_OK", "no banner registered for this arm")
            return
        run_dir = os.path.join(self.spec.models_dir, it.run)
        logs = [self.launcher.launcher_log(it), os.path.join(run_dir, "launcher_child.log")]
        line = postspawn.banner_guard(logs, it.banner, clock=postspawn.Clock(self.d.sleep, self.d.monotonic))
        self._set(it, "BANNER_OK", f"banner OK: {line}")

    def _arm_banner_ok(self, it: ArmItem) -> None:
        if it.throughput is None:
            self._set(it, "RUNNING", "no throughput gate registered for this arm")
            return
        data = self.rec(it)["data"]
        line = postspawn.throughput_gate(os.path.join(self.spec.models_dir, it.run), int(data["fork_step"]),
                                         it.throughput, clock=postspawn.Clock(self.d.sleep, self.d.monotonic),
                                         already_s=self.d.now() - data["go_epoch"])
        self._set(it, "RUNNING", line)

    def _arm_running(self, it: ArmItem) -> None:
        tr = self._adopt(it)
        rc = armend.wait_end(self.log, tr["end"], self._wrapper(it), self.launcher.launcher_pidfile(it),
                             self.d.end_timeout_s, poll_s=self.d.poll_s, proc=self.d.proc, sleep=self.d.sleep,
                             monotonic=self.d.monotonic)
        self._set(it, "ENDED", f"launcher_rc={rc}", rc=rc)

    def _describe(self, it: ArmItem) -> Describe:
        d = self.rec(it)["data"]["describe"]
        return Describe(run_name=d["run_name"], parent_run_dir=d["parent_run_dir"], delta_text=d["delta_text"],
                        steps=int(d["steps"]), argv=list(d["argv"]))

    def _launcher_log_text(self, it: ArmItem) -> str:
        try:
            return open(self.launcher.launcher_log(it), errors="replace").read()
        except OSError as e:
            raise Hold(f"no launcher log for {it.run}: {e}") from e

    def _arm_ended(self, it: ArmItem) -> None:
        data = self.rec(it)["data"]
        s = armend.parse_launcher_log(self._launcher_log_text(it))
        note = armend.completeness(os.path.join(self.spec.models_dir, it.run), self._describe(it).steps,
                                   int(data["rc"]), s)
        self._set(it, "COMPLETE", f"{note}; fps {s.fps}, restarts {s.restarts}")

    def _arm_complete(self, it: ArmItem) -> None:
        rec = armend.k9b_scratch(self.spec, it, self.d.runner, self.d.base_env)
        self._set(it, "K9B_PASS", f"K9(b) PASS (record in the scratch stage {rec})", k9b_record=rec)

    def _arm_k9b_pass(self, it: ArmItem) -> None:
        note = armend.thin(self.spec, it, self.d.runner, self.d.base_env, self._describe(it).steps, sleep=self.d.sleep)
        self._set(it, "THINNED", note)

    def _arm_thinned(self, it: ArmItem) -> None:
        data = self.rec(it)["data"]
        start = self._triggers(it)["start"]   # only THIS GO's segments: an earlier failed GO's START is not one
        dose = armend.dose_record(it, self.log.lines_after(start.offset), self._describe(it), self._launcher_log_text(it),
                                  int(data["rc"]), comparator=self.spec.extra.get("x_dose_comparator"))
        src_dir = os.path.join(self.spec.work_dir, "records", it.run)
        os.makedirs(src_dir, exist_ok=True)
        dose_src = os.path.join(src_dir, "dose.json")
        with open(dose_src, "w") as f:
            json.dump(dose, f, indent=1)
            f.write("\n")
        stage = self.spec.records_stage
        files = {os.path.join(stage, "k9b", f"{it.run}.json"): data["k9b_record"],
                 os.path.join(stage, "dose", f"{it.run}.json"): dose_src}
        shipper = (self.d.shipper or self._default_shipper)(self.spec)
        res = shipper.ship(files, f"measure(arm chain): arm-end records for {it.run} — K9(b) PASS (AM-6) + dose "
                                  f"({dose['wall_min']} min, {dose['steps_per_s']} steps/s)",
                           tag=f"{it.run}-{int(self.d.now())}")
        self._set(it, "RECORDED", f"shipped {len(res.files)} record(s) as {res.commit}", records_commit=res.commit)

    def _default_shipper(self, spec: ChainSpec) -> Shipper:
        return Shipper(spec.repo, spec.records_stage, spec.submodules, spec.work_dir,
                       os.path.join(os.path.dirname(spec.job_dir), "arm_chain_ship.lock"),
                       static_tier=static_tier_runner(self.d.runner, spec.python, spec.agent_cores, self.d.base_env,
                                                      spec.work_dir))

    def _arm_recorded(self, it: ArmItem) -> None:
        nxt = self._next(it)
        self._set(it, "DONE", f"hand-off to {nxt}")

    def _next(self, it: Item) -> str:
        i = self.spec.items.index(it)
        if i + 1 >= len(self.spec.items):
            return "nothing (the chain's last item)"
        n = self.spec.items[i + 1]
        return f"{n.key} at its own pin {n.pin[:12]}" if isinstance(n, ArmItem) else n.key

    # -- windows -----------------------------------------------------------------------------------------------------
    def _window(self, w: WindowItem) -> None:
        if self.rec(w)["phase"] == "STARTED":
            raise Hold(f"the supervisor stopped while window {w.name} ran: a step may still hold the GPU — verify by "
                       "hand (gpu_lock.sh --status), then clear-hold to re-run it or abandon it")
        self._set(w, "STARTED", "the window starts")
        lease = self.d.lease or (lambda: window.lease_status(self.spec.repo))
        res = window.run_window(self.spec, w, self.d.runner, self.d.base_env, self.d.gpu_idle, lease,
                                say=lambda m: self.log.append(m), monotonic=self.d.monotonic, sleep=self.d.sleep)
        if res.ok:
            self._set(w, "DONE", f"window PASS; hand-off to {self._next(w)}")
        elif w.on_fail == "continue":
            self._set(w, "DONE_FAILED", f"window FAILED ({'; '.join(res.lines)[:200]}); on_fail continue: hand-off "
                                        f"to {self._next(w)}")
        else:
            raise Hold(f"window {w.name} FAILED ({'; '.join(res.lines)[:300]}); on_fail hold")

    # -- reporting ---------------------------------------------------------------------------------------------------
    def status_lines(self) -> List[str]:
        out = []
        for it in self.spec.items:
            r = self.state["items"][it.key]
            h = f"  HOLD: {r['hold']['reason'][:160]}" if r.get("hold") else ""
            out.append(f"{it.key:<40} {r['phase']:<12}{h}")
        return out
