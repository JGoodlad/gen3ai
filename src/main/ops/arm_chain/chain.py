"""The chain's persistent state machine: one supervisor walks the items in order, one transition at a time.

ARM:     PENDING ─▶ PRECHECKED ─▶ GOING ─▶ LAUNCHED ─▶ SPAWNED ─▶ BANNER_OK ─▶ RUNNING ─▶ ENDED ─▶ COMPLETE
         ─▶ K9B_PASS ─▶ THINNED ─▶ RECORDED ─▶ DONE                                        (then the next item)
WINDOW:  PENDING ─▶ STARTED ─▶ DONE | DONE_FAILED (on_fail "continue"); a restart in STARTED HOLDs

Any step may raise a HOLD: the item keeps its phase, the hold is recorded with its reason, one ``HOLD:`` status line
is written, and the supervisor exits 3. Nothing after a HOLD runs. ``clear-hold --by WHO --resolution R`` (the
orchestrator's decision, typed by the training session) removes it, recorded with its resolution: ``retry`` re-runs
the step that held; ``relaunch`` resets an arm held at LAUNCHED whose kit never launched to PENDING; ``accept``
advances an arm past a post-spawn gate the orchestrator has checked by hand. ``abandon --by WHO`` marks a held item
``DONE_FAILED`` instead and the chain hands off past it.

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
  10-26 duplicate-supervisor incident started from a waiter that read a live supervisor as dead. ``clear-hold`` and
  ``abandon`` take the same lock, so they never write under a live supervisor.
* **A supervisor's liveness is readable** (``liveness``): it beats ``state["supervisor"]`` before every blocking step,
  declaring how long that step may take; ``status`` reads the lock holder and the heartbeat and prints ALIVE / STALE /
  DEAD, loudly when an item is mid-flight with no supervisor.
* **A started item cannot change under the chain**: the state keeps each started item's spec snapshot; a chain
  file edited under a started item is refused (append new items freely).
* **The hand-off is the NEXT ITEM's own pin**, by construction: there is no chain-wide pin to inherit.
* **A registered action, never a script's choice**: a VOID banner / a failed throughput gate applies the arm's
  ``on_void`` / ``on_throughput_fail`` (``stop``: SIGTERM the launcher through ``signal_if``, so it saves at its
  interrupt, record it, HOLD; ``page``: HOLD only).
* **The record ship is ASYNCHRONOUS once the kit's ``k9b-batch`` writes outside main** (``ship.kit_k9b_out_of_main``):
  a detached ``ship-records`` worker ships while the next item starts, so a ship HOLD pages (``SHIP-HOLD:``) instead
  of idling the GPU. Until that check passes, the ship stays synchronous. Either way main must be clean before and
  after every dry run, and before every GO.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, NoReturn, Optional, Sequence

from utils import procfs

from main.ops.arm_chain import armend, live, liveness, postspawn, window
from main.ops.arm_chain.launch import Describe, Hold, X35Launcher
from main.ops.arm_chain.procwatch import ProcRef, alive, ref_from_pidfile, ref_of, signal_if, wait_gone
from main.ops.arm_chain.runner import Runner, SubprocessRunner, agent_prefix, cpu_env
from main.ops.arm_chain.ship import Shipper, kit_k9b_out_of_main, main_dirty, static_tier_runner
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
#: ``clear-hold --resolution``: what the orchestrator decided (recorded in the history and the status line)
RESOLUTIONS = ("retry", "relaunch", "accept")
#: the phase an ``accept`` advances to: past the post-spawn gate that held
ACCEPT_NEXT = {"LAUNCHED": "SPAWNED", "SPAWNED": "BANNER_OK", "BANNER_OK": "RUNNING"}


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
    #: does the chain's lease token validate? ``(ok, why)``; default ``window.lease_token_valid``
    lease_token: Optional[Callable[[], "tuple[bool, str]"]] = None
    # the kit's GO re-runs its pre-checks (checkargs <= 10 min, the dry run <= 15 min), then waits 180 s, then the
    # pool-seed check (<= 30 min): a shorter bound HOLDs a healthy launch
    wrapper_spawn_timeout_s: float = 75 * 60
    end_timeout_s: float = 7 * 24 * 3600
    poll_s: float = 30.0
    #: the signal seam of the registered ``stop`` action (always behind ``signal_if``'s identity gate)
    kill: Callable[[int, int], None] = os.kill
    #: how long ``stop`` waits for the SIGTERM'd launcher (it saves at its next safe point; it SIGKILLs its child at
    #: 150 s, TRAINING_RUN_SOP §4)
    stop_wait_s: float = 300.0
    #: may the record ship go asynchronous? ``(ok, why)``; default :func:`ship.kit_k9b_out_of_main`
    kit_k9b_fixed: Optional[Callable[[ChainSpec], "tuple[bool, str]"]] = None
    #: starts the detached ``ship-records`` worker; default the CLI verb under nice 19 on the agent cores
    ship_spawner: Optional[Callable[[ChainSpec, ArmItem], ProcRef]] = None
    #: the chain's end waits this long for in-flight ships before COMPLETE / SHIP-HOLD
    ship_wait_s: float = 45 * 60
    #: the live launchers / trainers (``live.discover`` lines); the GO refuses beside any. Default: ``/proc``
    live_runs: Optional[Callable[[], List[str]]] = None


class _BeatingRunner:
    """The chain's runner, with a heartbeat before every blocking ``run`` that declares the step's own bound."""

    def __init__(self, inner: Runner, beat: Callable[[str, float], None]):
        self.inner, self.beat = inner, beat

    def run(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float, out_path: str) -> int:
        self.beat("running " + " ".join(os.path.basename(a) if a.startswith("/") else a for a in argv)[:100], timeout_s)
        return self.inner.run(argv, cwd=cwd, env=env, timeout_s=timeout_s, out_path=out_path)

    def spawn(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], out_path: str) -> ProcRef:
        return self.inner.spawn(argv, cwd=cwd, env=env, out_path=out_path)


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


class Chain:
    def __init__(self, spec: ChainSpec, deps: Optional[Deps] = None):
        self.spec, self.d = spec, deps or Deps()
        os.makedirs(spec.work_dir, exist_ok=True)
        self.log = StatusLog(spec.status_file, own_tag(spec))
        self.state = self._load_state()
        self._beating = False          # only the supervisor (``run``) beats; the CLI verbs never write a heartbeat
        self._doing = "starting"
        self._last_beat_saved = 0.0
        self.runner = _BeatingRunner(self.d.runner, self._beat)
        self.launcher = (self.d.launcher or (lambda s: X35Launcher(s, self.runner, self.d.base_env,
                                                                   sleep=self._sleep)))(spec)

    # -- state -----------------------------------------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        p = self.spec.state_path
        if os.path.exists(p):
            st = json.loads(open(p).read())
            if st.get("schema") != STATE_SCHEMA or st.get("chain") != self.spec.name:
                raise ChainError(f"{p} is not this chain's state")
        else:
            st = {"schema": STATE_SCHEMA, "chain": self.spec.name, "items": {}, "history": []}
        st.setdefault("ships", {})
        for it in self.spec.items:
            rec = st["items"].setdefault(it.key, {"phase": "PENDING", "hold": None, "data": {}})
            if rec["phase"] != "PENDING" and rec.get("snapshot") is not None and rec["snapshot"] != as_dict(it):
                raise ChainError(f"{it.key} has started ({rec['phase']}) and its chain-file entry CHANGED — a started "
                                 "item never changes under the chain (append a new item instead)")
        return st

    def _save(self) -> None:
        """Atomic and durable: a per-process tmp file (a CLI verb and the supervisor never share one), fsync'd before
        the rename, the directory fsync'd after it."""
        p = self.spec.state_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        dfd = os.open(os.path.dirname(p), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    # -- the heartbeat ---------------------------------------------------------------------------------------------
    def _beat(self, what: str, bound_s: float) -> None:
        """Persist ``state["supervisor"]``: alive, doing ``what``, and due to beat again within ``bound_s`` (+ slack).
        A beat that only moves the clock is saved at most every ``BEAT_EVERY_S``."""
        if not self._beating:
            return
        sup = self.state.setdefault("supervisor", {})
        now = self.d.now()
        doing = f"{self._doing}: {what}" if what else self._doing
        until = now + max(bound_s, 0.0) + liveness.BEAT_SLACK_S
        if (doing == sup.get("doing") and now - self._last_beat_saved < liveness.BEAT_EVERY_S
                and until <= float(sup.get("stale_after_epoch") or 0) + 60):
            return    # the persisted beat already covers this step
        sup.update(last_beat=_iso(now), last_beat_epoch=now, doing=doing, stale_after_epoch=until)
        self._last_beat_saved = now
        self._save()

    def _sleep(self, s: float) -> None:
        self._beat("", s)
        self.d.sleep(s)

    @contextlib.contextmanager
    def _cli_lock(self, verb: str) -> Iterator[None]:
        """A CLI verb that writes the state takes the supervisor's lock (never blocking): a live supervisor means the
        item is not held, and its next save would overwrite this one."""
        lock_path = self.spec.state_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ChainError(f"{verb}: a supervisor holds {lock_path} — it is running, so nothing is held; wait for "
                                 "its HOLD (it exits at once), then retry") from None
            self.state = self._load_state()     # what the last supervisor saved, not what this object read earlier
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

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

    def clear_hold(self, key: str, by: str, why: str, resolution: str = "retry") -> str:
        """The orchestrator's decision on a HELD item, recorded with its ``resolution`` (never silently):

        * ``retry`` — re-run the step that held. GOING / STARTED reset to PENDING (a fresh pre-check, never a blind
          re-GO or re-run). Elsewhere the step re-reads its evidence: right when the cause was transient and outside
          the persisted evidence (main dirty, a busy GPU, a ship), wrong at LAUNCHED after a wrapper HOLD (it reads the
          same line again: use ``relaunch``).
        * ``relaunch`` — LAUNCHED only (review 10-26, finding 3: the kit's GO-time pre-checks HELD, or no pool-seed
          PASS): back to PENDING for a fresh dry run and GO, with fresh triggers past today's lines. Refused unless the
          kit provably never launched: the GO wrapper is gone, no live launcher in the pid file, and no
          ``models/<run>`` (a run dir means the kit launched: the kit refuses it, and a human decides).
        * ``accept`` — the orchestrator checked the run by hand and accepts it: the arm ADVANCES past the post-spawn
          gate that held (LAUNCHED → SPAWNED, SPAWNED → BANNER_OK, BANNER_OK → RUNNING; finding 9's paged arm), and
          every later gate still runs. Refused for an arm its registered ``stop`` signalled, and at LAUNCHED without
          ``models/<run>`` (nothing launched to accept)."""
        if not by.strip() or not why.strip():
            raise ChainError("clear-hold needs --by (whose decision) and --why")
        if resolution not in RESOLUTIONS:
            raise ChainError(f"clear-hold --resolution must be one of {', '.join(RESOLUTIONS)}, not {resolution!r}")
        with self._cli_lock("clear-hold"):
            r = self.state["items"].get(key)
            if r is None or not r.get("hold"):
                raise ChainError(f"{key} holds nothing")
            it = next(i for i in self.spec.items if i.key == key)
            old, phase = r["hold"], r["phase"]
            if resolution == "relaunch":
                to, effect = self._resolve_relaunch(it, r)
            elif resolution == "accept":
                to, effect = self._resolve_accept(it, r, by, why)
            else:
                to, effect = self._resolve_retry(r)
            r["hold"], r["phase"] = None, to
            self.state["history"].append({"at": _iso(self.d.now()), "item": key, "from": "HOLD", "to": to,
                                          "note": f"cleared by {by} ({resolution} at {phase}): {why}"[:300]})
            self._save()
            msg = (f"{key}: hold cleared by {by} ({why}); resolution {resolution} at {phase}: {effect}; was: "
                   f"{old['reason'][:160]}")
            self.log.append(msg)
            return msg

    @staticmethod
    def _resolve_retry(r: Dict[str, Any]) -> "tuple[str, str]":
        if r["phase"] in ("GOING", "STARTED"):
            r["snapshot"] = None   # the item has not launched: an orchestrator's edit of it is taken, not refused
            return "PENDING", "reset to PENDING (a fresh pre-check, never a blind re-GO or re-run)"
        if r["phase"] == "LAUNCHED":
            return "LAUNCHED", ("the post-spawn wait re-reads the wrapper's lines since the GO (a wrapper HOLD there "
                                "holds again: --resolution relaunch or accept)")
        return r["phase"], f"{r['phase']}'s step runs again"

    def _resolve_relaunch(self, it: Item, r: Dict[str, Any]) -> "tuple[str, str]":
        if not isinstance(it, ArmItem) or r["phase"] != "LAUNCHED":
            raise ChainError(f"{it.key}: relaunch applies to an arm held at LAUNCHED, not {r['phase']} (at GOING, "
                             "retry already resets to PENDING)")
        data = r["data"]
        why_not = []
        if "wrapper" in data and alive(ProcRef.from_json(data["wrapper"]), self.d.proc):
            why_not.append(f"its GO wrapper pid {data['wrapper']['pid']} is ALIVE")
        w = self.launcher.find_go_wrapper(it, self.d.proc)
        if w is not None:
            why_not.append(f"a live GO wrapper for {it.arm} s{it.seed} (pid {w.pid})")
        lref = ref_from_pidfile(self.launcher.launcher_pidfile(it), self.d.proc)
        if lref is not None and "main.launcher" in procfs.cmdline(lref.pid, self.d.proc):
            why_not.append(f"its launcher pid {lref.pid} is ALIVE")
        if os.path.exists(os.path.join(self.spec.models_dir, it.run)):
            why_not.append(f"models/{it.run} EXISTS (the kit launched it, and refuses a GO onto it: a human decides — "
                           "accept, abandon, or move the run dir aside first)")
        if why_not:
            raise ChainError(f"{it.key}: relaunch REFUSED, the kit may have launched: {'; '.join(why_not)}")
        attempts = list(data.get("attempts", []))
        attempts.append({**{k: v for k, v in data.items() if k != "attempts"}, "held": r["hold"]})
        r["data"] = {"attempts": attempts}
        r["snapshot"] = None   # nothing launched: an orchestrator's edit of the arm is taken, not refused
        return "PENDING", (f"RELAUNCH: reset to PENDING (attempt {len(attempts) + 1}: a fresh dry run and GO, triggers "
                           "re-armed past today's lines; the wrapper, launcher and run dir were verified absent)")

    def _resolve_accept(self, it: Item, r: Dict[str, Any], by: str, why: str) -> "tuple[str, str]":
        phase, data = r["phase"], r["data"]
        if not isinstance(it, ArmItem) or phase not in ACCEPT_NEXT:
            raise ChainError(f"{it.key}: accept applies to an arm held at a post-spawn gate "
                             f"({', '.join(ACCEPT_NEXT)}), not {phase}")
        if data.get("stopped") or data.get("stopping"):
            raise ChainError(f"{it.key}: accept REFUSED — the registered stop SIGTERMed this run "
                             f"({(data.get('stopped') or data.get('stopping'))}): abandon it")
        if phase == "LAUNCHED" and not os.path.exists(os.path.join(self.spec.models_dir, it.run)):
            raise ChainError(f"{it.key}: accept REFUSED — no models/{it.run}: nothing launched to accept "
                             "(--resolution relaunch)")
        to = ACCEPT_NEXT[phase]
        data.setdefault("accepted", []).append({"phase": phase, "to": to, "by": by, "why": why[:200],
                                                "at": _iso(self.d.now())})
        return to, f"ACCEPTED: {phase} -> {to} without its gate (the orchestrator's evidence); every later gate runs"

    def abandon(self, key: str, by: str, why: str) -> str:
        """The orchestrator's decision that a HELD item will not finish (a VOID arm it stopped, a window it drops):
        the item becomes ``DONE_FAILED`` and the chain hands off past it. Only a held item; nothing is signalled,
        nothing is thinned or shipped for it."""
        if not by.strip() or not why.strip():
            raise ChainError("abandon needs --by (whose decision) and --why")
        with self._cli_lock("abandon"):
            return self._abandon_locked(key, by, why)

    def _abandon_locked(self, key: str, by: str, why: str) -> str:
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
        rc: Any = "died"
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            self.state = self._load_state()       # what the last writer (a clear-hold) saved under this lock
            me = ref_of(os.getpid(), self.d.proc) or ProcRef(os.getpid(), -1)
            self.state["supervisor"] = {"pid": me.pid, "start_ticks": me.start_ticks, "started_at": _iso(self.d.now()),
                                        "code": _code_version()}
            self._beating = True
            self._beat("", 0.0)
            rc = self._run_locked(max_items)
            return int(rc)
        except BaseException as e:
            rc = f"died: {type(e).__name__}"
            raise
        finally:
            self._beating = False
            sup = self.state.get("supervisor")
            if sup is not None and sup.get("pid") == os.getpid():
                sup.update(exit=rc, exited_at=_iso(self.d.now()))
                with contextlib.suppress(Exception):   # a full disk must not mask the exit's own exception
                    self._save()
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _run_locked(self, max_items: Optional[int]) -> int:
        done_items = 0
        for it in self.spec.items:
            r = self.state["items"][it.key]
            if r["phase"] in TERMINAL:
                continue
            try:
                self._reconcile_ships()   # inside the try: a failure here is a HOLD line, never a silent death
                if r.get("hold"):
                    self.log.append(f"HOLD: {it.key} is still HELD at {r['phase']} "
                                    f"({r['hold']['reason'][:120]}); nothing runs until clear-hold")
                    return EXIT_HOLD
                if isinstance(it, ArmItem):
                    while self.state["items"][it.key]["phase"] not in TERMINAL:
                        self._arm_step(it)
                else:
                    self._doing = f"{it.key} {r['phase']}"
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
        try:
            return self._finish_ships()
        except Exception as e:
            self.log.append(f"HOLD: the chain's end: UNEXPECTED {type(e).__name__}: {e}")
            return EXIT_HOLD

    # -- arm steps ---------------------------------------------------------------------------------------------------
    def _triggers(self, it: ArmItem) -> Dict[str, Trigger]:
        return {k: Trigger.from_json(v) for k, v in self.rec(it)["data"]["triggers"].items()}

    def _arm_step(self, it: ArmItem) -> None:
        phase = self.rec(it)["phase"]
        self._doing = f"{it.key} {phase}"
        self._beat("", 0.0)
        getattr(self, f"_arm_{phase.lower()}")(it)

    def _guard_main_clean(self, when: str) -> None:
        """Main is never dirty under a dry run or a GO: the kit's ``k9b-batch`` used to write a missing record into
        main's stage (F1), and an async ship lands records on main while the next arm pre-checks. Re-read a few
        times so a ship's own fast-forward (milliseconds) is never read as dirt; dirt that stays is a HOLD."""
        dirt = ""
        for i in range(3):
            dirt = main_dirty(self.spec.repo)
            if not dirt:
                return
            if i < 2:
                self._sleep(5)
        raise Hold(f"main is DIRTY {when} (never dry-run or GO on a dirty main; who wrote it? the kit's k9b-batch "
                   f"into main's stage is F1): {dirt[:200]}")

    def _arm_pending(self, it: ArmItem) -> None:
        disk = armend.disk_ok(self.spec.models_dir, it.disk_need_gib, self.spec.disk_headroom_gib)
        self._guard_main_clean("before the dry run")
        d = self.launcher.prelaunch(it)
        self._guard_main_clean("after the dry run")
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
        self._guard_main_clean("before the GO")
        ok, why = self.d.gpu_idle()
        t0 = self.d.monotonic()
        while not ok:
            if self.d.monotonic() - t0 > 300:
                raise Hold(f"the GPU is not idle before the GO: {why} (a window step or the previous trainer left behind)")
            self._sleep(10)
            ok, why = self.d.gpu_idle()
        self._refuse_a_second_go(it)
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
        self._rearm_cue(it)

    def _rearm_cue(self, it: ArmItem) -> None:
        """Review 10-26, finding 4: a ``tail -F`` Monitor armed before this GO follows none of the new run's logs.
        One filtered line names them (``monitor`` follows them itself and needs no re-arm)."""
        self.log.append(f"MONITOR-REARM {it.key}: {' '.join(watch_paths(self.spec, it))}")

    def _refuse_a_second_go(self, it: ArmItem) -> None:
        """Review 10-26, finding 7: never GO beside a live GO wrapper (any arm's) or a live launcher / trainer."""
        wrappers = self.launcher.live_go_wrappers(self.d.proc)
        runs = (self.d.live_runs or (lambda: [r.line() for r in live.discover(self.spec.models_dir, self.d.proc)]))()
        if wrappers or runs:
            raise Hold(f"refusing the GO of {it.run}: a launch is already live (GO wrapper(s) "
                       f"{[w.pid for w in wrappers]}; launcher / trainer {runs[:3]}) — never a second GO beside it")

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
                                      clock=postspawn.Clock(self._sleep, self.d.monotonic))
        self._set(it, "SPAWNED", line.split("] ", 1)[-1][:200])

    def _note(self, it: Item, msg: str, **data: Any) -> None:
        """Persist ``data`` on the item and write one status line, without a phase change."""
        self.state["items"][it.key]["data"].update(data)
        self._save()
        self.log.append(f"{it.key}: {msg}")

    def _act(self, it: ArmItem, field_name: str, action: str, why: str) -> NoReturn:
        """Apply the arm's REGISTERED action, then HOLD. ``page``: HOLD only. ``stop``: the SOP stop path — SIGTERM
        the launcher through the identity gate (its pid file's process, and only if its cmdline is THIS run's
        ``main.launcher``), so it saves at its interrupt; wait (bounded) for it to go; record it; HOLD."""
        if action == "page":
            raise Hold(f"{why} — {field_name} page: the run keeps training until the orchestrator decides")
        prior = self.rec(it)["data"].get("stopping")
        if prior is not None:
            # review 10-26, finding 8: a supervisor that died between the SIGTERM and its record must not send a
            # second one (it may escalate the launcher's shutdown past its interrupted save)
            lref = ProcRef.from_json(prior["launcher"])
            ok, msg = bool(prior.get("signalled", True)), f"already SIGTERMed at {prior['at']} (not re-sent)"
        else:
            lref = ref_from_pidfile(self.launcher.launcher_pidfile(it), self.d.proc)
            if lref is None:
                raise Hold(f"{why} — {field_name} stop: no live launcher in {self.launcher.launcher_pidfile(it)}; NOT "
                           "signalled (verify by hand)")
            self.state["items"][it.key]["data"]["stopping"] = {"by": field_name, "at": _iso(self.d.now()),
                                                               "launcher": lref.to_json()}
            self._save()
            ok, msg = signal_if(lref, launcher_ere(it.run), signal.SIGTERM, proc=self.d.proc, kill=self.d.kill)
            self.state["items"][it.key]["data"]["stopping"]["signalled"] = ok
            self._save()
        gone = wait_gone(lref, self.d.stop_wait_s, poll_s=min(self.d.poll_s, 10.0), proc=self.d.proc,
                         sleep=self._sleep, monotonic=self.d.monotonic) if ok else False
        interrupted = os.path.isfile(os.path.join(self.spec.models_dir, it.run, "final_model_interrupted.zip"))
        rec = {"by": field_name, "at": _iso(self.d.now()), "signalled": ok, "identity": msg[:200],
               "launcher": lref.to_json(), "gone": gone, "interrupted_save": interrupted}
        self._note(it, f"STOP ({field_name} stop): {msg}; launcher {'GONE' if gone else 'STILL ALIVE'} after "
                       f"{self.d.stop_wait_s:.0f} s; final_model_interrupted.zip {'present' if interrupted else 'absent'}",
                   stopped=rec)
        raise Hold(f"{why} — STOPPED by the registered {field_name} stop ({'SIGTERM sent' if ok else 'NOT signalled'}: "
                   f"{msg[:120]}; launcher {'gone' if gone else 'STILL ALIVE'}); the orchestrator decides: abandon, or "
                   "a relaunch")

    def _arm_spawned(self, it: ArmItem) -> None:
        if it.banner is None:
            self._set(it, "BANNER_OK", "no banner registered for this arm")
            return
        run_dir = os.path.join(self.spec.models_dir, it.run)
        logs = [self.launcher.launcher_log(it), os.path.join(run_dir, "launcher_child.log")]
        try:
            line = postspawn.banner_guard(logs, it.banner, clock=postspawn.Clock(self._sleep, self.d.monotonic),
                                          ended=self._ended(it))
        except postspawn.Void as v:
            self._act(it, "on_void", it.on_void, str(v))
        self._set(it, "BANNER_OK", f"banner OK: {line}")

    def _arm_banner_ok(self, it: ArmItem) -> None:
        if it.throughput is None:
            self._set(it, "RUNNING", "no throughput gate registered for this arm")
            return
        data = self.rec(it)["data"]
        run_dir = os.path.join(self.spec.models_dir, it.run)
        clock = postspawn.Clock(self._sleep, self.d.monotonic)
        ended = self._ended(it)
        try:
            if data.get("first_trainer_epoch") is None:
                left = it.throughput.first_line_within_min * 60 - (self.d.now() - data["go_epoch"])
                epoch, what = postspawn.wait_first_trainer_line(os.path.join(run_dir, "launcher_child.log"), run_dir,
                                                                int(data["fork_step"]), max(left, 0.0), now=self.d.now,
                                                                poll_s=min(self.d.poll_s, 30.0), clock=clock,
                                                                ended=ended)
                self._note(it, f"the throughput gate counts from {what}, {(epoch - data['go_epoch']) / 60:.0f} min "
                               f"after the GO", first_trainer_epoch=epoch)
                data = self.rec(it)["data"]
            line = postspawn.throughput_gate(run_dir, int(data["fork_step"]), it.throughput, clock=clock,
                                             poll_s=min(self.d.poll_s * 4, 120.0),
                                             already_s=self.d.now() - data["first_trainer_epoch"], ended=ended)
        except postspawn.ThroughputFail as f:
            self._act(it, "on_throughput_fail", it.on_throughput_fail, str(f))
        self._set(it, "RUNNING", line)

    def _ended(self, it: ArmItem) -> Optional[Callable[[], Optional[str]]]:
        """Review 10-26, finding 6: the post-spawn gates watch the wrapper's END line too, so a run that dies in its
        first hour fails its gate at once instead of at the gate's deadline."""
        trig = self.rec(it)["data"].get("triggers", {}).get("end")
        if trig is None:
            return None
        t = self._adopt(it)["end"]

        def ended() -> Optional[str]:
            m = self.log.find(t)
            return m.group(0).split("] ", 1)[-1] if m else None
        return ended

    def _arm_running(self, it: ArmItem) -> None:
        tr = self._adopt(it)
        rc = armend.wait_end(self.log, tr["end"], self._wrapper(it), self.launcher.launcher_pidfile(it),
                             self.d.end_timeout_s, poll_s=self.d.poll_s, proc=self.d.proc, sleep=self._sleep,
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
        rec = armend.k9b_scratch(self.spec, it, self.runner, self.d.base_env)
        self._set(it, "K9B_PASS", f"K9(b) PASS (record in the scratch stage {rec})", k9b_record=rec)

    def _arm_k9b_pass(self, it: ArmItem) -> None:
        note = armend.thin(self.spec, it, self.runner, self.d.base_env, self._describe(it).steps, sleep=self._sleep)
        self._set(it, "THINNED", note)

    def _arm_thinned(self, it: ArmItem) -> None:
        data = self.rec(it)["data"]
        start = self._triggers(it)["start"]   # only THIS GO's segments: an earlier failed GO's START is not one
        comparator, comparator_src = dose_comparator(self.spec)
        dose = armend.dose_record(it, self.log.lines_after(start.offset), self._describe(it), self._launcher_log_text(it),
                                  int(data["rc"]), comparator=comparator)
        src_dir = os.path.join(self.spec.work_dir, "records", it.run)
        os.makedirs(src_dir, exist_ok=True)
        dose_src = os.path.join(src_dir, "dose.json")
        with open(dose_src, "w") as f:
            json.dump(dose, f, indent=1)
            f.write("\n")
        stage = self.spec.records_stage
        files = {os.path.join(stage, "k9b", f"{it.run}.json"): data["k9b_record"],
                 os.path.join(stage, "dose", f"{it.run}.json"): dose_src}
        message = (f"measure(arm chain): arm-end records for {it.run} — K9(b) PASS (AM-6) + dose "
                   f"({dose['wall_min']} min, {dose['steps_per_s']} steps/s)")
        self.log.append(f"{it.key}: the dose record's c0_comparator: {comparator_src}")
        req = ship_request_path(self.spec, it)
        _write_json(req, {"item": it.key, "run": it.run, "files": files, "message": message})
        ok, why = (self.d.kit_k9b_fixed or kit_k9b_out_of_main)(self.spec)
        if ok and (self.d.ship_spawner is not None or self.spec.source):
            res_path = ship_result_path(self.spec, it)
            if os.path.exists(res_path):
                os.replace(res_path, res_path + f".prev{int(self.d.now())}")
            ref = (self.d.ship_spawner or self._spawn_ship_worker)(self.spec, it)
            self.state["ships"][it.key] = {"worker": ref.to_json(), "request": req, "result": res_path,
                                           "spawned": _iso(self.d.now()), "outcome": None}
            self._set(it, "RECORDED", f"records ship ASYNC (worker pid {ref.pid}; {why}): the chain hands off now, a "
                                      "failed ship pages SHIP-HOLD", records_ship="async")
            return
        if ok:
            why = "the chain file has no source path for a ship worker"
        self.log.append(f"{it.key}: the records ship is SYNCHRONOUS ({why})")
        shipper = (self.d.shipper or (lambda s: default_shipper(s, self.runner, self.d.base_env)))(self.spec)
        self._beat("the records ship (sync: its lock <= 30 min, then git and the static tier)", 2 * 1800.0)
        res = shipper.ship(files, message, tag=f"{it.run}-{int(self.d.now())}")
        self._set(it, "RECORDED", f"shipped {len(res.files)} record(s) as {res.commit}", records_commit=res.commit,
                  records_ship="sync")

    def _spawn_ship_worker(self, spec: ChainSpec, it: ArmItem) -> ProcRef:
        argv = [*agent_prefix(spec.agent_cores, 19), spec.python, "-m", "main.ops.arm_chain", "ship-records",
                spec.source, "--item", it.key]
        return self.runner.spawn(argv, cwd=spec.repo, env=cpu_env(self.d.base_env, spec.repo),
                                   out_path=os.path.join(spec.work_dir, "ships", f"{it.run}.worker.txt"))

    # -- async ships -------------------------------------------------------------------------------------------------
    def _ship_outcome(self, key: str) -> Optional[Dict[str, Any]]:
        """The worker's result, a synthetic failure for a worker that died without one, or None (in flight)."""
        sh = self.state["ships"][key]
        try:
            return dict(json.loads(open(sh["result"]).read()))
        except (OSError, ValueError):
            pass
        if alive(ProcRef.from_json(sh["worker"]), self.d.proc):
            return None
        try:   # it may have written the result between the two reads
            return dict(json.loads(open(sh["result"]).read()))
        except (OSError, ValueError):
            return {"ok": False, "reason": "the ship worker died without a result (see its worker.txt)", "synthetic": True}

    def _reconcile_ships(self) -> None:
        """Record each ship's outcome; a FAILED one is re-read every time, so a by-hand ``ship-records`` retry that
        succeeded is taken (the next ``run`` then completes the chain)."""
        for key, sh in self.state["ships"].items():
            old = sh.get("outcome")
            if old is not None and old.get("ok"):
                continue
            out = self._ship_outcome(key)
            if out is None or out == old:
                continue
            sh["outcome"] = out
            self._save()
            if out.get("synthetic") and old is None:   # the worker wrote no line of its own
                self.log.append(f"SHIP-HOLD: {key}: {out['reason']}; retry: {retry_hint(self.spec, key)}")

    def _finish_ships(self) -> int:
        """The chain's end: wait (bounded) for in-flight ships; COMPLETE only when every record landed."""
        t0 = self.d.monotonic()
        while True:
            self._reconcile_ships()
            pending = [k for k, sh in self.state["ships"].items() if sh.get("outcome") is None]
            if not pending or self.d.monotonic() - t0 > self.d.ship_wait_s:
                break
            self._sleep(min(self.d.poll_s, 15.0))
        failed = [k for k, sh in self.state["ships"].items()
                  if sh.get("outcome") is not None and not sh["outcome"].get("ok")]
        if pending or failed:
            self.log.append(f"SHIP-HOLD: every item is DONE but records are NOT all shipped (in flight {pending}, "
                            f"failed {failed}); retry each: {retry_hint(self.spec, '<item>')}")
            return EXIT_HOLD
        self.log.append("chain COMPLETE: every item is DONE")
        return EXIT_OK

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
        res = window.run_window(self.spec, w, self.runner, self.d.base_env, self.d.gpu_idle, lease,
                                say=lambda m: self.log.append(m), monotonic=self.d.monotonic, sleep=self._sleep,
                                token_valid=self.d.lease_token)
        if res.ok:
            self._set(w, "DONE", f"window PASS; hand-off to {self._next(w)}")
        elif w.on_fail == "continue":
            self._set(w, "DONE_FAILED", f"window FAILED ({'; '.join(res.lines)[:200]}); on_fail continue: hand-off "
                                        f"to {self._next(w)}")
        else:
            raise Hold(f"window {w.name} FAILED ({'; '.join(res.lines)[:300]}); on_fail hold")

    # -- reporting ---------------------------------------------------------------------------------------------------
    def liveness(self) -> liveness.Liveness:
        return liveness.probe(self.state, self.spec.state_path + ".lock", self.d.now(), self.d.proc,
                              {it.key: self.launcher.launcher_pidfile(it) for it in self.spec.arms()})

    def status_lines(self) -> List[str]:
        out = [self.liveness().line()]
        for it in self.spec.items:
            r = self.state["items"][it.key]
            h = f"  HOLD: {r['hold']['reason'][:160]}" if r.get("hold") else ""
            sh = ""
            if it.key in self.state["ships"]:
                rec = self.state["ships"][it.key].get("outcome")
                o = rec if rec is not None and rec.get("ok") else self._ship_outcome(it.key)
                sh = ("  ship: IN FLIGHT" if o is None else f"  ship: {o.get('commit')}" if o.get("ok")
                      else f"  ship: FAILED ({str(o.get('reason'))[:100]})")
            out.append(f"{it.key:<40} {r['phase']:<12}{h}{sh}")
        return out


def _code_version() -> str:
    """The commit this supervisor's code was started from (best-effort: the checkout holding this module)."""
    try:
        p = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short=12", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def watch_paths(spec: ChainSpec, it: ArmItem) -> List[str]:
    """The logs an arm's run writes, fixed per item: the launcher's stdout log and the trainer's
    ``launcher_child.full.log`` (it rotates by rename, which ``tail -F`` and ``monitor`` follow)."""
    return [os.path.join(spec.job_dir, f"{it.run}.launcher.log"),
            os.path.join(spec.models_dir, it.run, "launcher_child.full.log")]


def dose_comparator(spec: ChainSpec) -> "tuple[Optional[Dict[str, Any]], str]":
    """The dose record's ``c0_comparator`` block and where it came from (review 10-26, finding 13: a chain file with
    no ``x_dose_comparator`` silently dropped it). An explicit ``x_dose_comparator`` wins (``null`` omits it on
    purpose); otherwise the block every dose record already in MAIN's stage carries, when they all agree; otherwise
    none, and the status line says why."""
    if "x_dose_comparator" in spec.extra:
        c = spec.extra["x_dose_comparator"]
        return (dict(c) if c else None), ("the chain file's x_dose_comparator" if c else
                                          "none (the chain file's x_dose_comparator is null)")
    d = os.path.join(spec.repo, spec.records_stage, "dose")
    blocks = []
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        names = []
    for n in names:
        try:
            blocks.append(json.loads(open(os.path.join(d, n)).read()).get("c0_comparator"))
        except (OSError, ValueError, AttributeError):
            blocks.append(None)
    kinds = {json.dumps(b, sort_keys=True) for b in blocks}
    if len(kinds) == 1 and blocks[0]:
        return dict(blocks[0]), f"the block all {len(blocks)} dose record(s) in the stage carry"
    if not blocks:
        return None, "none (no x_dose_comparator, and the stage has no dose record to take it from)"
    return None, (f"none (no x_dose_comparator, and the stage's {len(blocks)} dose record(s) carry {len(kinds)} "
                  "different blocks)")


def default_shipper(spec: ChainSpec, runner: Runner, base_env: Mapping[str, str]) -> Shipper:
    """The real records ship. Built from the chain FILE only: the ship worker never loads the state (finding 11: a
    chain-file edit to a started item made the state load refuse, and the worker died with no result)."""
    return Shipper(spec.repo, spec.records_stage, spec.submodules, spec.work_dir,
                   os.path.join(os.path.dirname(spec.job_dir), "arm_chain_ship.lock"),
                   static_tier=static_tier_runner(runner, spec.python, spec.agent_cores, base_env, spec.work_dir))


def launcher_ere(run: str) -> str:
    """The identity a ``stop`` signals: THIS run's ``main.launcher`` (``--run-name <run>``, the name terminated)."""
    return r"main\.launcher .*--run-name[ =]" + re.escape(run) + r"( |$)"


def ship_request_path(spec: ChainSpec, it: ArmItem) -> str:
    return os.path.join(spec.work_dir, "ships", f"{it.run}.request.json")


def ship_result_path(spec: ChainSpec, it: ArmItem) -> str:
    return os.path.join(spec.work_dir, "ships", f"{it.run}.result.json")


def retry_hint(spec: ChainSpec, key: str) -> str:
    return f"python -m main.ops.arm_chain ship-records {spec.source or '<chain.json>'} --item {key}"


def _write_json(path: str, d: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def ship_worker(spec: ChainSpec, key: str, shipper: Shipper, now: Callable[[], float] = time.time) -> int:
    """The detached ``ship-records`` worker (and the by-hand retry): ship one arm's records from its request file,
    write the result file, and write ONE status line itself (the supervisor may be deep in the next arm's wait).
    Idempotent: a re-ship of records already on the remote is a success (``Shipper._already_shipped``)."""
    it = next((i for i in spec.arms() if i.key == key), None)
    if it is None:
        raise ChainError(f"{key} is not an arm of chain {spec.name}")
    log = StatusLog(spec.status_file, own_tag(spec))
    try:
        try:
            req = json.loads(open(ship_request_path(spec, it)).read())
        except (OSError, ValueError) as e:
            raise Hold(f"no readable ship request at {ship_request_path(spec, it)} ({e}): the arm has not reached "
                       "THINNED") from e
        res = shipper.ship(dict(req["files"]), str(req["message"]), tag=f"{it.run}-{int(now())}")
    except Exception as h:   # a Hold, or anything else: never die without a result and a page
        why = str(h) if isinstance(h, Hold) else f"UNEXPECTED {type(h).__name__}: {h}"
        _write_json(ship_result_path(spec, it), {"ok": False, "reason": why[:600], "at": _iso(now())})
        log.append(f"SHIP-HOLD: {key}: {why[:300]}; retry: {retry_hint(spec, key)}")
        return EXIT_HOLD
    _write_json(ship_result_path(spec, it), {"ok": True, "commit": res.commit, "files": res.files,
                                             "already": res.already, "at": _iso(now())})
    log.append(f"{key}: records shipped as {res.commit}{' (already on the remote)' if res.already else ''} (async)")
    return EXIT_OK
