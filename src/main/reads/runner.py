"""PLAN / RUN / STATUS: a read's games, on the box's two lanes (``designs/tools/reads.md`` §3).

* THE ANCHOR LANE plays :mod:`main.x35.kakuna` units, unchanged: 20-game units on the heavy slots' SHORT lane
  (``mem_cap.sh --max-s 600``; one killed past its bound, exit 88, is played once more as a long job), each on a 95xx
  port claimed with a held flock (``kakuna.claim_port``), each preflighted (the T28 halt marker, the committed team
  manifest / panel cell, the model's bytes). One kakuna state per (group, model), planned only once the model SETTLES
  (``main.reads.models``), so a not-yet-final model is never played. ``concurrency`` K (default 1) runs up to K units at
  once (:mod:`main.reads.anchor_pool`, scheduling only; K is capped at the box's heavy-slot count).
* THE S LANE plays S against the frozen R: ``main.h2h play-many`` cells of ``pairs`` mirrored pairs recorded in
  ``batch_pairs`` batches (the resume grain) into the group's own eval ledger and family; a UNIT is up to ``workers``
  engines inside ONE heavy slot (``mem_cap.sh --slot``), time-boxed to ``unit_s`` (<= 40 min), at nice 19 on the
  agent cores. R is re-verified (every member's sha256 and its ``content_sha256``) before every unit; a change STOPS
  the lane (exit 3), never a silent re-score.

Both lanes: resumable (a counted unit / a recorded batch is never replayed), one supervisor per lane per state (a
flock keyed by the state's PATH outside it, a ``/proc`` scan, and the dir's identity: :mod:`main.reads.supervisors`),
a REGISTERED stop (``stop_epoch``: an absolute epoch, never open-ended), and the playing checkout must stay at
the commit ``plan`` recorded (``code.json``; with a spec ``pin``, that commit and a clean tree), or one a ``repin``
accepted because the play path is byte-identical (:mod:`main.reads.repin`). Run them detached
(``setsid nohup python -m main.reads run --lane anchor ...``), one process per lane. A supervisor writes its OWN pid
to ``<state>/<lane>/supervisor.pid`` and honors a HOLD (:mod:`main.reads.lanectl`): at the top of every loop turn it looks
for ``<state>/<lane>/hold.json`` and, when it is there, starts no new unit, waits for those in flight and exits 0.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from main.reads import anchor_pool as AP
from main.reads import lanectl as LC
from main.reads import layout as LY
from main.reads.assemble import AssembleError
from main.reads import models as M
from main.reads import supervisors as SV
from main.reads.spec import ReadSpec, load as load_spec
from utils import git_sha
from utils.filelock import try_exclusive

CORES = "5-7,13-15"
S_CAP_GB = 12
S_UNIT_S = 2400
S_WORKERS = 2
S_THREADS = 2
POLL_S = 300
#: ``s-unit``'s exit when its time box ended a cell mid-way (the batches recorded so far stand).
UNIT_TIME_UP = 124
#: The S lane's exit when R moved under the read.
R_MOVED = 3
FAMILY_RULE = "main.reads S on the frozen R (population_score of design_compute_efficiency §5.6.4)"


class RunError(RuntimeError):
    """A precondition of a verb does not hold (named; nothing is played)."""


def child_env() -> Dict[str, str]:
    """A child's env: THIS checkout's ``src`` (a worktree's ``python`` would import main's), CPU only."""
    from utils.paths import src_path

    return dict(os.environ, PYTHONPATH=str(src_path()), CUDA_VISIBLE_DEVICES="")


def _log(msg: str) -> None:
    print(f"[reads {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _jl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}, default=str) + "\n")


def _atomic(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------------- the code
def checkout_code(root: Optional[Path] = None) -> Dict[str, Any]:
    """The playing checkout's commit and whether its tracked tree is dirty."""
    from utils.paths import repo_root

    r = str(root or repo_root())
    commit = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", r, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    diff = subprocess.run(["git", "-C", r, "diff", "HEAD"], capture_output=True, check=True).stdout
    # a dirty tree's EDITS are part of the code: a later edit keeps "dirty" but moves this digest
    return {"commit": commit, "dirty": dirty, "diff_sha": hashlib.sha256(diff).hexdigest()[:16], "root": r}


def check_code(spec: ReadSpec, code: Dict[str, Any]) -> None:
    if spec.pin is not None:
        if not code["commit"].startswith(git_sha.current(spec.pin, repo=code.get("root"))):
            raise RunError(f"the playing checkout is at {code['commit'][:12]}, not the spec's pin {spec.pin}")
        if code["dirty"]:
            raise RunError(f"the playing checkout {code['root']} has tracked changes (a pin plays a clean tree)")


def _code_holds(state: Path, now_code: Dict[str, Any]) -> None:
    """The playing checkout is the commit ``plan`` ran at, or one a ``repin`` accepted (its play path byte-identical),
    with the same clean-tree / diff-digest state."""
    from main.reads import repin as RP

    was = json.loads(LY.code_file(state).read_text())
    try:
        ok = RP.accepted_commits(state)
    except RP.RepinError as exc:
        raise RunError(f"the playing code cannot be checked: {exc}") from None
    if (now_code["commit"] not in [git_sha.current(c, repo=now_code.get("root")) for c in ok] or now_code["dirty"] != was["dirty"]
            or now_code.get("diff_sha") != was.get("diff_sha")):
        raise RunError(f"the playing code changed under the read: planned at {was['commit'][:12]}"
                       f"{' (dirty)' if was['dirty'] else ''}"
                       f"{', repinned to ' + ', '.join(c[:12] for c in ok[1:]) if len(ok) > 1 else ''}"
                       f", now {now_code['commit'][:12]}{' (dirty)' if now_code['dirty'] else ''}")


# -------------------------------------------------------------------------------------------- plan
def _load_models(state: Path) -> Dict[str, Dict[str, Any]]:
    p = LY.models_file(state)
    return json.loads(p.read_text()) if p.exists() else {}


def settle(spec: ReadSpec, state: Path, *, now: Optional[float] = None,
           step_of: Callable[[str], int] = M.recorded_step) -> Dict[str, str]:
    """Resolve and copy every model that has SETTLED and is not yet copied; plan its anchor units. Returns
    ``{label: why}`` for the models still not ready."""
    with open(state / "settle.lock", "w") as lk:       # the two lanes' supervisors both settle: one at a time
        fcntl.flock(lk, fcntl.LOCK_EX)
        return _settle(spec, state, now=now, step_of=step_of)


def _settle(spec: ReadSpec, state: Path, *, now: Optional[float], step_of: Callable[[str], int]) -> Dict[str, str]:
    from main.x35 import kakuna as KA

    have = _load_models(state)
    waiting: Dict[str, str] = {}
    for lab, m in spec.models.items():
        if lab in have:
            continue
        r = M.resolve(m, now=now, step_of=step_of)
        if isinstance(r, M.NotReady):
            waiting[lab] = r.why
            continue
        have[lab] = M.copy_resolved(r, LY.ckpt_dir(state, lab))
        _atomic(LY.models_file(state), have)
        _log(f"model {lab}: {r.source} step {r.step:,} sha {r.sha256[:12]} copied")
    from main.reads.spec import imported

    for g, a in spec.anchor.items():
        for lab in a.models:
            st = LY.anchor_state(state, g, lab)
            if lab not in have or lab in imported(a) or (st / "plan.json").exists():
                continue
            m = spec.models[lab]
            units = KA.plan_units([{"label": lab, "arm": m.arm, "seed": m.seed, "path": have[lab]["copy"],
                                    "belief_prior_mix": m.load.get("belief_prior_mix"),
                                    "search": m.load.get("search")}],
                                  teamsets=a.teamsets, unit_seeds=a.unit_seeds, opponent=a.opponent,
                                  config_dir=st / "configs", per_game=a.per_game)
            if any(u.model_sha256 != have[lab]["sha256"] for u in units):
                raise RunError(f"{lab}: the copy no longer hashes to its recorded sha256")
            st.mkdir(parents=True, exist_ok=True)
            _atomic(st / "plan.json", {"units": [u.__dict__ for u in units]})
            _log(f"anchor {g}/{lab}: planned {len(units)} units")
    return waiting


def plan(spec_path: Path, state: Path, *, code: Optional[Dict[str, Any]] = None, now: Optional[float] = None,
         step_of: Callable[[str], int] = M.recorded_step) -> Dict[str, Any]:
    """Write the read's state (``main.reads.layout``) from its spec; idempotent with the SAME spec bytes."""
    state = LY.check_state_dir(state)
    spec = load_spec(spec_path)
    sf = LY.spec_file(state)
    if not sf.exists():
        live = SV.live_supervisors(state, exclude=SV.self_and_ancestors())
        if live:
            # the D3 reader's duplicate: a state renamed aside while its supervisor lived, re-planned at the same path
            raise RunError(f"{state} holds no plan yet, but live supervisor(s) {live} name this path: they work a "
                           "state that was moved from it. Stop them (by pid) or plan another path")
    state.mkdir(parents=True, exist_ok=True)
    b = Path(spec_path).read_bytes()
    if sf.exists() and sf.read_bytes() != b:
        raise RunError(f"{sf} holds another spec: a state is ONE read (plan a new state for a new spec)")
    code = code or checkout_code()
    check_code(spec, code)
    if LY.code_file(state).exists():
        _code_holds(state, code)
    else:
        _atomic(LY.code_file(state), dict(code, pin=spec.pin))
    if not sf.exists():
        sf.write_bytes(b)
    from main.reads import imports as IM

    for lane, groups in (("anchor", spec.anchor), ("s", spec.s)):
        for g, x in groups.items():
            if x.imports is not None:
                IM.check(spec, lane, g, this_state=state)     # the earlier read, the same cell: before any play
    for g, x in spec.s.items():
        _s_prepare(spec, state, g)
    waiting = settle(spec, state, now=now, step_of=step_of)
    return {"state": str(state), "spec_sha256": spec.sha256, "waiting": waiting}


def load_state(state: Path) -> ReadSpec:
    state = LY.check_state_dir(state)
    if not LY.spec_file(state).exists():
        raise RunError(f"{state}: no spec.json (run `plan` first)")
    return load_spec(LY.spec_file(state))


# -------------------------------------------------------------------------------------------- S lane
def _s_members(spec: ReadSpec, g: str) -> List[Dict[str, Any]]:
    from main.reads.assemble import members_of

    x = spec.s[g]
    p = M.spec_path(x.refpop)
    members_of(p, want_content=x.refpop_content_sha256)       # every file re-hashed, the content checked
    return [dict(m) for m in json.loads(p.read_text())["members"]]


def _s_prepare(spec: ReadSpec, state: Path, g: str) -> Dict[str, Any]:
    """Copy R's members (sha-checked) and register the group's family; idempotent."""
    x = spec.s[g]
    d = LY.s_dir(state, g)
    stf = d / "state.json"
    st = json.loads(stf.read_text()) if stf.exists() else {}
    if "members" not in st:
        st["members"] = {}
        for m in _s_members(spec, g):
            dst, sha = M.copy_checked(Path(m["path"]), d / "members" / m["label"], m["sha256"])
            st["members"][m["label"]] = {"copy": str(dst), "sha256": sha, "source": m["path"]}
        st["refpop_content_sha256"] = x.refpop_content_sha256
        d.mkdir(parents=True, exist_ok=True)
        _atomic(stf, st)
    if not st.get("family_registered"):
        r = subprocess.run([sys.executable, "-m", "main.eval_ledger", "family-register", "--root",
                            str(LY.s_ledger(state, g)), "--id", x.family, "--decision-kind", "ab_verdict", "--rule",
                            FAMILY_RULE, "--protocol", "gen3_eval_protocol_v1_h2h"], check=False,
                           env=child_env(), capture_output=True, text=True)
        if r.returncode != 0:
            raise RunError(f"family-register {x.family} refused (exit {r.returncode}): {r.stderr.strip()[-800:]}")
        st["family_registered"] = True
        _atomic(stf, st)
    return st


def s_pairs(state: Path, spec: ReadSpec, g: str, label: str, sha: str) -> Dict[str, int]:
    """``{member: completed pairs}`` of ``label``'s cells in group ``g`` so far."""
    from main.reads.assemble import cell_pairs

    x = spec.s[g]
    st = json.loads((LY.s_dir(state, g) / "state.json").read_text())
    return cell_pairs(LY.s_ledger(state, g), x.family, x.request(label), sha,
                      {k: v["sha256"] for k, v in st["members"].items()})


def s_engines(pending: Sequence[Tuple[str, Sequence[str]]], workers: int) -> List[str]:
    """Split the pending ``(label, [members])`` cells over at most ``workers`` engines (``label:m1,m2``)."""
    engines: List[str] = []
    for label, left in pending:
        k = min(len(left), workers - len(engines))
        engines += [f"{label}:" + ",".join(left[w::k]) for w in range(k)]
        if len(engines) >= workers:
            break
    return engines


def s_unit_argv(state: Path, g: str, engines: Sequence[str], box_s: int, *, deadline_epoch: float,
                threads: int = S_THREADS, cap_gb: int = S_CAP_GB, cpus: str = CORES,
                py: str = sys.executable) -> List[str]:
    """ONE heavy-slot S unit (``mem_cap.sh --slot``: it waits for a slot and holds it for the box)."""
    from utils.paths import repo_path

    return [str(repo_path("scripts/ops/mem_cap.sh")), "--slot", "--name", "reads_s", str(cap_gb), "nice", "-n", "19",
            "taskset", "-c", cpus, py, "-m", "main.reads", "s-unit", "--state", str(state), "--group", g,
            "--box-s", str(int(box_s)), "--deadline-epoch", str(int(deadline_epoch)), "--threads", str(threads),
            *[x for e in engines for x in ("--engine", e)]]


def play_many_argv(spec: ReadSpec, state: Path, g: str, label: str, member_labels: Sequence[str],
                   models: Dict[str, Dict[str, Any]], members: Dict[str, Dict[str, Any]], *, threads: int,
                   py: str = sys.executable) -> List[str]:
    x = spec.s[g]
    a = [py, "-m", "main.h2h", "play-many", "--players", models[label]["copy"], "--opponents",
         *[members[m]["copy"] for m in member_labels], "--pairs", str(x.pairs), "--batch-pairs", str(x.batch_pairs),
         "--purpose", "ab", "--family", x.family, "--request", x.request(label), "--out", str(LY.s_ledger(state, g)),
         "--label", f"reads_{spec.read}"[:64], "--device", "cpu", "--torch-threads", "1", "--threads", str(threads)]
    if x.panel is not None:
        a += ["--panel", x.panel]
    return a


def s_unit(state: Path, g: str, engines: Sequence[str], box_s: int, *, deadline_epoch: Optional[float] = None,
           threads: int = S_THREADS) -> int:
    """The body of one S unit (inside its heavy slot): the engines' ``play-many`` children, killed at the box or
    the registered stop, whichever is first (the box starts AFTER the slot wait, so the stop is absolute)."""
    spec = load_state(state)
    models = _load_models(state)
    st = json.loads((LY.s_dir(state, g) / "state.json").read_text())
    for spec_e in engines:
        M.verify(models[spec_e.split(":", 1)[0]])
    env = child_env()
    deadline = time.time() + box_s
    if deadline_epoch is not None:
        deadline = min(deadline, float(deadline_epoch))
    procs = []
    logs = LY.s_dir(state, g) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for e in engines:
        label, labs = e.split(":", 1)
        argv = play_many_argv(spec, state, g, label, labs.split(","), models, st["members"], threads=threads)
        logf = open(logs / f"play_{label}.log", "a")  # noqa: SIM115 - closed when the child is reaped
        procs.append((e, subprocess.Popen(argv, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL), logf))
    while any(p.poll() is None for _e, p, _f in procs) and time.time() < deadline:
        time.sleep(min(5.0, max(0.05, deadline - time.time())))
    boxed = set()
    for e, p, _f in procs:
        if p.poll() is None:
            boxed.add(e)
            p.terminate()
    rc = 0
    for e, p, logf in procs:
        try:
            r = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            r = p.wait()
        logf.close()
        _log(f"s unit engine {e}: exit {r}{' (time box)' if e in boxed else ''}")
        if e not in boxed:
            rc = rc or r                                      # an engine that ENDED on its own reports its exit
    return UNIT_TIME_UP if boxed and rc == 0 else rc


def _progress(spec: ReadSpec, state: Path, g: str, engines: Sequence[str],
              models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    labs = sorted({e.split(":", 1)[0] for e in engines})
    return {lab: s_pairs(state, spec, g, lab, models[lab]["sha256"]) for lab in labs}


def _s_pending(spec: ReadSpec, state: Path, models: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, List[str]]]:
    from main.reads.spec import imported

    out = []
    for g, x in spec.s.items():
        st = json.loads((LY.s_dir(state, g) / "state.json").read_text())
        for lab in x.models:
            if lab not in models or lab in imported(x):
                continue
            have = s_pairs(state, spec, g, lab, models[lab]["sha256"])
            left = [m for m in st["members"] if have.get(m, 0) < x.pairs]
            if left:
                out.append((g, lab, left))
    return out


# -------------------------------------------------------------------------------------------- run
def _lock(state: Path, lane: str) -> Any:
    fh = try_exclusive(LY.lane_lock(state, lane), "w")
    if fh is None:
        raise RunError(f"another {lane} supervisor holds {LY.lane_lock(state, lane)}")
    return fh


def _anchor_todo(spec: ReadSpec, state: Path, fails: Dict[str, int]) -> List[Tuple[Path, Any]]:
    """The anchor lane's queue: unit seeds outermost, then the groups in spec order. Within a seed a group plays its
    models in spec order, each model's units in plan order; a ``per_game`` group (S-U) plays DEAL by deal instead: game
    ``g`` on every team set by every model (so a searched game follows its raw twin's) before game ``g + 1``."""
    from main.x35 import kakuna as KA

    out = []
    for s in range(max((len(a.unit_seeds) for a in spec.anchor.values()), default=0)):
        for g, a in spec.anchor.items():
            if s >= len(a.unit_seeds):
                continue
            todo = []
            for li, lab in enumerate(a.models):
                st = LY.anchor_state(state, g, lab)
                if not (st / "plan.json").exists():
                    continue
                for ui, u in enumerate(KA.load_plan(st)):
                    if (u.unit_seed == a.unit_seeds[s] and not KA.is_done(st, u)
                            and fails.get(u.name, 0) < KA.MAX_ATTEMPTS):
                        order = ((u.game, a.teamsets.index(u.teamset), li) if a.per_game and u.game is not None
                                 else (li, ui))
                        todo.append((order, st, u))
            out += [(st, u) for _o, st, u in sorted(todo, key=lambda t: t[0])]
    return out


def _check_prefix(spec: ReadSpec, state: Path, st: Path, u: Any, jl: Path) -> None:
    """G-PREFIX after a per-game unit of a searched group ends (:mod:`main.reads.prefix`): once both units of its deal
    are counted, their protocol through the search's first turn must be byte-identical; a mismatch HOLDS the lane (no
    new unit starts) for a root cause, and the group stays inconclusive (``assemble``) until it is resolved."""
    from main.reads import prefix as PX

    if getattr(u, "game", None) is None:
        return
    pair = PX.pair_of_unit(spec, state, st, u.name)
    if pair is None:
        return
    res = PX.check_pair(pair)
    if res is None:
        return
    _jl(jl, {"event": "g_prefix", "deal": res["deal"], "ok": res["ok"], "why": res["why"],
             "searched": res["searched"], "raw": res["raw"]})
    if not res["ok"]:
        _log(f"anchor: G-PREFIX MISMATCH on deal {res['deal']} ({res['why']}): holding the lane")
        LC.hold(state, "anchor", by="G-PREFIX", why=f"deal {res['deal']}: {res['why']}"[:600])


def anchor_concurrency(requested: int, lane: str) -> int:
    """The units the anchor lane runs at once: ``requested`` (>= 1), capped at the box's heavy-slot count. Only the
    anchor lane runs units concurrently (an S unit is already several engines inside ONE slot)."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise RunError(f"--concurrency {requested!r}: need an integer >= 1")
    if lane != "anchor":
        if requested != 1:
            raise RunError(f"--concurrency {requested} applies to the anchor lane only (an S unit is already "
                           "--workers engines in one heavy slot)")
        return 1
    cap = AP.slot_cap()
    if requested > cap:
        _log(f"anchor: --concurrency {requested} capped at the box's {cap} heavy slots")
    return min(requested, cap)


def run(state: Path, lane: str, stop_epoch: float, *, workers: int = S_WORKERS, threads: int = S_THREADS,
        unit_s: int = S_UNIT_S, run_unit: Optional[Callable[..., int]] = None,
        run_s: Optional[Callable[[List[str]], int]] = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time, concurrency: int = 1, max_units: Optional[int] = None) -> int:
    """Work ``lane`` (``anchor`` | ``s``) until done or ``stop_epoch``; resumable (module docstring). ``concurrency``:
    the anchor lane's units at once (:func:`anchor_concurrency`; 1 = one at a time, the default). ``max_units`` (anchor
    lane): start at most that many units in THIS supervisor's life, then let them finish and stop (event ``max_units``;
    S-U's stage 0 is the queue's first units). It is not a registered stop."""
    from main.x35 import kakuna as KA

    if lane not in ("anchor", "s"):
        raise RunError(f"lane {lane!r} is none of anchor, s")
    if max_units is not None and (lane != "anchor" or isinstance(max_units, bool) or not isinstance(max_units, int)
                                  or max_units < 1):
        raise RunError(f"--max-units {max_units!r}: an integer >= 1, on the anchor lane only")
    spec = load_state(state)
    if lane == "anchor" and not spec.anchor or lane == "s" and not spec.s:
        raise RunError(f"the spec has no {lane} group")
    k = anchor_concurrency(concurrency, lane)
    KA.check_stop_epoch(int(stop_epoch), now=clock())
    state = LY.check_state_dir(state)
    LC.refuse_if_held(state, lane)      # a held lane starts only through `resume`, which releases the hold
    sv = SV.Hold(state, lane)           # the path lock + the /proc scan + the dir's identity
    try:
        hold = _lock(state, lane)       # the in-state lock too: a supervisor of older code takes only this one
    except RunError:
        sv.close()
        raise
    run_unit = run_unit or KA.run_unit
    run_s = run_s or (lambda argv: subprocess.run(argv, check=False, env=child_env()).returncode)
    jl = LY.supervisor_log(state, lane)
    try:
        LC.write_pidfile(state, lane, stop_epoch=stop_epoch)   # THIS process's pid: `$!` of setsid nohup is the wrapper's
    except BaseException:
        hold.close()
        sv.close()
        raise
    _jl(jl, {"event": "start", "pid": os.getpid(), "stop_epoch": stop_epoch,
             **({"concurrency": k, "concurrency_requested": concurrency} if lane == "anchor" else {})})
    fails: Dict[str, int] = {}
    started = [0]

    def ended(st: Path, u: Any) -> None:
        if not KA.is_done(st, u):
            fails[u.name] = fails.get(u.name, 0) + 1
            return
        _check_prefix(spec, state, st, u, jl)

    pool = AP.UnitPool(k, run_unit, ended, log=_log)
    try:
        while True:
            sv.check()                  # the path still names the dir this supervisor started on
            held = LC.read_hold(state, lane)
            if held is not None:        # HOLD: start nothing new, let the units in flight finish, then stop
                _jl(jl, {"event": "held", "by": held["by"], "why": held["why"], "in_flight": len(pool)})
                _log(f"{lane}: HELD by {held['by']} ({held['why']}): {len(pool)} unit(s) in flight finish, then stop")
                pool.drain()
                _jl(jl, {"event": "held_stopped",
                         **({"left": len(_anchor_todo(spec, state, fails))} if lane == "anchor" else {})})
                return 0
            _code_holds(state, checkout_code())
            waiting = settle(spec, state)
            models = _load_models(state)
            if lane == "anchor":
                todo = [(st, u) for st, u in _anchor_todo(spec, state, fails) if not pool.busy(st, u)]
                if todo and max_units is not None and started[0] >= max_units:
                    pool.drain()                # the units this supervisor may start are started: finish, then stop
                    _jl(jl, {"event": "max_units", "started": started[0],
                             "left": len(_anchor_todo(spec, state, fails))})
                    return 0
                if todo and pool.free():
                    if clock() >= stop_epoch:
                        pool.drain()            # a lane never stops with a unit of its own in flight
                        _jl(jl, {"event": "stop_at", "left": len(_anchor_todo(spec, state, fails))})
                        return 0
                    n_new = pool.free() if max_units is None else min(pool.free(), max_units - started[0])
                    for st, u in todo[:n_new]:
                        if k > 1:
                            _log(f"anchor: start {u.name} ({len(pool) + 1}/{k} in flight)")
                        started[0] += 1
                        pool.submit(st, u)      # k == 1: played right here, to its end
                    continue
                if len(pool):                   # the pool is full, or nothing new to start: wait for a unit to end
                    pool.wait_one(AP.POLL_S)
                    continue
            else:
                try:
                    for g in spec.s:
                        _s_members(spec, g)                  # R re-verified before every unit
                except (ValueError, OSError, AssembleError) as exc:   # R moved under the read: stop, never re-score
                    _jl(jl, {"event": "r_moved", "why": f"{type(exc).__name__}: {exc}"})
                    return R_MOVED
                pending = _s_pending(spec, state, models)
                if pending:
                    if clock() + 60.0 >= stop_epoch:
                        _jl(jl, {"event": "stop_at", "left": [(g, lab) for g, lab, _ in pending]})
                        return 0
                    g = pending[0][0]
                    engines = s_engines([(lab, left) for gg, lab, left in pending if gg == g], workers)
                    box = int(min(unit_s, stop_epoch - clock()))
                    argv = s_unit_argv(state, g, engines, box, deadline_epoch=stop_epoch, threads=threads)
                    before = _progress(spec, state, g, engines, models)
                    _jl(jl, {"event": "unit", "group": g, "engines": engines, "box_s": box})
                    rc = run_s(argv)
                    _jl(jl, {"event": "unit_end", "group": g, "engines": engines, "rc": rc})
                    if rc not in (0, UNIT_TIME_UP):
                        return rc
                    if rc == 0 and _progress(spec, state, g, engines, models) == before:
                        # every engine ended cleanly and no pair was added: play-many has nothing left to play
                        raise RunError(f"S unit {engines} ended OK with no new pair: the cells cannot reach "
                                       f"{spec.s[g].pairs} pairs (incomplete batches?); read the ledger")
                    continue
            if not waiting:
                _jl(jl, {"event": "done", "not_counted": sorted(n for n, k in fails.items() if k >= KA.MAX_ATTEMPTS)})
                return 0
            if clock() + POLL_S >= stop_epoch:
                _jl(jl, {"event": "stop_at", "waiting": waiting})
                return 0
            _log(f"{lane}: waiting for {sorted(waiting)} to settle (poll {POLL_S} s)")
            LC.wait_unless_held(state, lane, POLL_S, sleep)       # looks for a hold between short steps
    except SV.SupervisorError:
        raise                           # the path names another dir now: write nothing there, not even a log line
    except Exception as exc:
        _jl(jl, {"event": "refused", "why": f"{type(exc).__name__}: {exc}"})
        raise
    except BaseException:
        pool.abandon()                  # an interrupt leaves the units (own sessions) running, as at k == 1
        raise
    finally:
        pool.close()                    # the claims below are held until the last unit of this lane has ended
        LC.remove_pidfile(state, lane)  # only if it still names this process (a moved state's new dir has none)
        hold.close()
        sv.close()


# -------------------------------------------------------------------------------------------- status
def _hold_view(state: Path, lane: str) -> Optional[Dict[str, Any]]:
    try:
        h = LC.read_hold(state, lane)
    except LC.LaneError as exc:
        return {"error": str(exc)}
    return None if h is None else {k: h.get(k) for k in ("by", "why", "time")}


def status(state: Path) -> Dict[str, Any]:
    """Every group's progress: models settled or waiting, anchor units counted / planned, S pairs per member."""
    from main.reads import repin as RP
    from main.x35 import kakuna as KA

    spec = load_state(state)
    models = _load_models(state)
    out: Dict[str, Any] = {"read": spec.read, "models": {lab: (models[lab]["sha256"][:12] if lab in models
                                                              else "waiting") for lab in spec.models},
                           "anchor": {}, "s": {},
                           # live supervisors from /proc (never kill -0), by lane: [] = none is running
                           "supervisors": {ln: SV.live_supervisors(state, lane=ln).get(ln, []) for ln in SV.LANES},
                           # each lane's supervisor as IT recorded itself (<state>/<lane>/supervisor.pid): the pid a
                           # watcher targets, with start ticks; alive is checked against /proc. null = none recorded
                           "supervisor_pids": {ln: LC.pid_status(state, ln) for ln in SV.LANES},
                           # a held lane (main.reads.lanectl): who, why, since when; null = not held
                           "holds": {ln: _hold_view(state, ln) for ln in SV.LANES},
                           # the playing commits a repin accepted beyond code.json's (main.reads.repin)
                           "repins": [{k: r[k] for k in ("to_commit", "by", "time")} for r in RP.load_repins(state)]}
    from main.reads.spec import imported

    for g, a in spec.anchor.items():
        for lab in a.models:
            st = LY.anchor_state(state, g, lab)
            if a.imports is not None and lab in imported(a):        # imported(a) is empty unless imports is set
                out["anchor"].setdefault(g, {})[lab] = f"imported from {a.imports.state} ({a.imports.group})"
                continue
            if not (st / "plan.json").exists():
                out["anchor"].setdefault(g, {})[lab] = "not planned"
                continue
            units = KA.load_plan(st)
            out["anchor"].setdefault(g, {})[lab] = {"counted": sum(KA.is_done(st, u) for u in units),
                                                    "planned": len(units)}
    for g, x in spec.s.items():
        for lab in x.models:
            out["s"].setdefault(g, {})[lab] = (
                f"imported from {x.imports.state} ({x.imports.group})"
                if x.imports is not None and lab in imported(x)
                else s_pairs(state, spec, g, lab, models[lab]["sha256"]) if lab in models else "waiting")
    return out
