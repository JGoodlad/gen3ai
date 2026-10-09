"""THE DISK-SPACE GUARD — a training run does not start without the disk it will fill, and does not
let a checkpoint write fail half-way (orchestrator, 2026-10-09).

WHY. On 2026-10-09 ~07:00 the root filesystem reached 100 % (2.2 GB free) while a pinned screen chain was
writing checkpoints; a failed checkpoint write mid-run would have cost a registered seed. One 15M-step run
fills ~4.2 GB (`models/rb_st_legacy_s1006`: 4.2 GB, ten sibling `rb_st_*` runs the same to 0.1 GB).

TWO HALVES, ONE MODULE.

1. THE PREFLIGHT (:func:`check_for_run`). Refuses (`FATAL_CONFIG`, so the launcher does not restart into the
   same refusal) when the free space on the run archive's filesystem is below REQUIRED. REQUIRED is derived
   from the run itself (:func:`requirement`), and the arithmetic is printed:

       periodic checkpoints  = (steps // interval  -  resume // interval)  x  checkpoint bytes
       + 3 extra checkpoint copies   (final_model, best_model, final_model_interrupted)
       + 1 copy on a fork            (the source --model is copied into the new dir)
       + eval cycles x eval-trace bytes per cycle       (traces are kept forever by default)
       + pool snapshots x checkpoint bytes              (capped by the pool window, 20)
       + the run's compile cache (what is not on disk yet)
       + value-sidecar / TensorBoard / log bytes per million steps
       = subtotal;   REQUIRED = subtotal x (1 + MARGIN_FRACTION) + RESERVE_BYTES

   Periodic checkpoints are NEVER pruned (`_TrackingCheckpointCallback`), so the count is the boundaries still
   to cross, not a retention window. The checkpoint size is read from the run's own newest checkpoint (a
   resume), else the fork's source `--model`, else the newest same-architecture run in the archive, else the
   biggest recent one (`find_checkpoint_bytes`, source named in the report). The per-unit ALLOWANCES are
   MEASURED from a finished run and declared in :class:`Allowances` with their provenance.

2. THE IN-RUN WATCH (:class:`InRunGuard`). At each checkpoint save (one `shutil.disk_usage`), free space
   below 2 x the next save (= the size of the one just written) prints a loud warning; below 1 x it asks for
   the graceful stop (`TrainExitCode.FATAL_DISK`, 8: the checkpoint just written stands, the process exits
   cleanly, the launcher does NOT restart). A save that fails with ENOSPC is the same stop (the previous
   checkpoint stays valid; `latest.txt` is written only after a save succeeds).

THE OPT-OUT. `--allow-low-disk` (dev / short runs): the preflight proceeds under a recorded
`cli_args._disk_guard` and the in-run STOP stands down (warnings still print).
`--debug` (CPU smoke) is exempt.

BOTH SURFACES. The trainer calls :func:`check_for_run` after its run dir is resolved (nothing created yet);
the launcher calls it before the pin / worktree / run dir exist and `--dry-run` calls the same function.
"""
from __future__ import annotations

import errno
import math
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

MiB = 1 << 20
GiB = 1 << 30

OPT_OUT_FLAG = "--allow-low-disk"

#: Verdict statuses.
EXEMPT_DEBUG = "exempt-debug"
PASS = "pass"
OPTED_OUT = "opted-out"            # would be REFUSED, proceeding under the flag
REFUSED_LOW = "refused-low"

#: The pool's window (`agents.training.snapshot_pool.DEFAULT_MAX_SNAPSHOTS`; a test pins the equality —
#: this module must not import the sb3-heavy pool just to read one integer).
POOL_WINDOW = 20

#: With no same-architecture run to read, the checkpoint size is the BIGGEST of this many newest runs'
#: checkpoints (recent runs are the nearest architecture; the biggest errs on the safe side without letting
#: one old, much larger model set the bar for every launch).
FALLBACK_RECENT_RUNS = 5


@dataclass(frozen=True)
class Allowances:
    """Per-unit bytes MEASURED from a finished run, rounded UP.

    Provenance: `models/rb_st_legacy_s1006` (15,000,181 steps, `--arch production`, ridealong heads, eval every
    2M, 2026-10-09), `du` per top-level entry — compile_cache 2.3 GB (inductor 1.2 + triton 1.2),
    checkpoints 899 MB (15 x 62.8 MB), eval_traces 484 MB over 7 cycles (69.1 MB each), snapshots 360 MB
    (6 x 60 MB), value_sidecar 71 MB, tb 4.1 MB, launcher logs 4.4 MB, best_model + final_model 120 MB, total
    4.2 GB; ten sibling `rb_st_*` runs `du` 4.2-4.3 GB. A run that exceeds a rate is a FINDING: re-measure."""
    eval_trace_bytes_per_cycle: int = 72 * MiB          # 69.1 MiB measured
    compile_cache_bytes: int = 2_450 * MiB              # 2.3 GB measured (2.4 GiB)
    value_sidecar_bytes_per_mstep: int = 5 * MiB        # 71 MB / 15M = 4.7 MB
    log_bytes_per_mstep: int = 1 * MiB                  # (tb 4.1 + logs 4.4 MB) / 15M = 0.57 MB
    fixed_bytes: int = 16 * MiB                         # metadata.json 324 KB, ledgers, sidecars
    extra_checkpoint_copies: int = 3                    # final_model, best_model, final_model_interrupted


DEFAULT_ALLOWANCES = Allowances()

#: The safety margin over the subtotal (estimation error: a ragged cadence, a forced checkpoint, a bigger
#: model than the reference) and the absolute reserve a run must LEAVE FREE — the 2026-10-09 near-miss had
#: 2.2 GB free at 100 %, and the filesystem is shared with logs, the desktop and every other agent.
MARGIN_FRACTION = 0.25
RESERVE_BYTES = 4 * GiB


def _fmt(n: float) -> str:
    n = float(n)
    if abs(n) >= GiB:
        return f"{n / GiB:.2f} GiB"
    return f"{n / MiB:.1f} MiB"


# --- reading the disk ----------------------------------------------------------------------------

def probe_path(path: str) -> str:
    """The nearest EXISTING ancestor of ``path`` (a run dir that is not created yet lives on its
    parent's filesystem)."""
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p or os.sep


def _read_free(path: str) -> int:
    return int(shutil.disk_usage(probe_path(path)).free)


def free_bytes(path: str) -> int:
    """Bytes available to this user on ``path``'s filesystem (`shutil.disk_usage`). The seam the suite
    pins (the root conftest replaces THIS with a roomy disk, so no test reads the box's real free space);
    the reader proper is :func:`_read_free`."""
    return _read_free(path)


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


# --- the checkpoint size -------------------------------------------------------------------------

def _newest_zip(directory: str, prefix: str = "") -> Optional[str]:
    try:
        names = [n for n in os.listdir(directory) if n.endswith(".zip") and n.startswith(prefix)]
    except OSError:
        return None
    best: Optional[Tuple[float, str]] = None
    for n in names:
        full = os.path.join(directory, n)
        try:
            m = os.path.getmtime(full)
        except OSError:
            continue
        if best is None or m > best[0]:
            best = (m, full)
    return best[1] if best else None


def _run_checkpoint_zip(run_dir: str) -> Optional[str]:
    """The newest resumable zip a run dir holds: its periodic checkpoints, else its final model."""
    z = _newest_zip(os.path.join(run_dir, "checkpoints"), "checkpoint_")
    if z:
        return z
    for name in ("final_model.zip", "final_model_interrupted.zip", "final_model_exception.zip"):
        full = os.path.join(run_dir, name)
        if os.path.isfile(full):
            return full
    return None


def _size(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return int(os.path.getsize(path))
    except OSError:
        return None


def _current_arch_signature() -> Optional[str]:
    try:
        from agents.model.model_version.constants import ARCH_SIGNATURE
        return str(ARCH_SIGNATURE)
    except Exception:                              # noqa: BLE001 — an estimate must not kill a launch
        return None


def _recorded_signature(run_dir: str) -> Optional[str]:
    import json
    try:
        with open(os.path.join(run_dir, "model_config.json")) as f:
            return str(json.load(f).get("arch_signature"))
    except (OSError, ValueError):
        return None


def find_checkpoint_bytes(run_dir: str, model_path: Optional[str], archive_dir: Optional[str],
                          *, scan_limit: int = 30) -> Tuple[Optional[int], str]:
    """``(bytes of ONE checkpoint, where it came from)``; ``(None, why)`` when nothing can be read.

    Order: this run's own newest checkpoint (a resume) -> the fork's source ``--model`` (a zip, or a run dir
    -> its newest zip) -> the archive: the newest run recording the CURRENT ``ARCH_SIGNATURE`` -> the BIGGEST
    newest-zip among the ``FALLBACK_RECENT_RUNS`` newest runs that have one (a different architecture)."""
    z = _run_checkpoint_zip(run_dir) if os.path.isdir(run_dir) else None
    if z and _size(z):
        return _size(z), f"this run's own newest checkpoint {os.path.basename(z)}"
    if model_path:
        mp = model_path
        if os.path.isdir(mp):
            mp = _run_checkpoint_zip(mp) or ""
        if mp and _size(mp):
            return _size(mp), f"the fork source {os.path.basename(mp)}"
    if archive_dir and os.path.isdir(archive_dir):
        try:
            runs = [os.path.join(archive_dir, n) for n in os.listdir(archive_dir)
                    if os.path.isdir(os.path.join(archive_dir, n)) and not n.startswith("_")]
        except OSError:
            runs = []
        runs.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0, reverse=True)
        runs = runs[:scan_limit]
        sig = _current_arch_signature()
        sized: List[Tuple[str, int]] = []
        for r in runs:
            zr = _run_checkpoint_zip(r)
            sz = _size(zr)
            if not sz:
                continue
            if sig is not None and _recorded_signature(r) == sig:
                return sz, f"archive run {os.path.basename(r)} (same ARCH_SIGNATURE) {os.path.basename(zr or '')}"
            sized.append((r, sz))
        if sized:
            recent = sized[:FALLBACK_RECENT_RUNS]           # newest first
            r, sz = max(recent, key=lambda t: t[1])
            return sz, (f"archive run {os.path.basename(r)} (the biggest of the {len(recent)} newest runs "
                        "with a checkpoint; no run records this ARCH_SIGNATURE)")
    return None, "no checkpoint of this run, no fork source and no sized run in the archive"


# --- the requirement -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPlan:
    """Everything the arithmetic reads, resolved by the caller (the trainer from its namespace, the launcher
    from the argv + the checkpoint)."""
    run_dir: str
    steps: int                       # --steps, the total the run trains to
    resume_steps: int                # num_timesteps the run starts from (0 fresh)
    checkpoint_interval: int         # env steps between periodic checkpoints
    eval_freq: int                   # env steps between eval cycles (<= 0: no cycles)
    ckpt_bytes: Optional[int]        # ONE checkpoint zip
    ckpt_source: str
    fork: bool = False               # a fork copies its source model into the new dir
    snapshots: bool = True           # pool snapshots are written (self-play); conservative default
    compile_cache: bool = True       # a CUDA run builds the inductor / triton cache
    existing_compile_cache_bytes: int = 0


@dataclass(frozen=True)
class Requirement:
    terms: Tuple[Tuple[str, int, str], ...]      # (label, bytes, the arithmetic)
    subtotal: int
    margin: int
    reserve: int
    ckpt_known: bool

    @property
    def required(self) -> int:
        return self.subtotal + self.margin + self.reserve

    def lines(self) -> List[str]:
        out = [f"      {label:<22} {_fmt(b):>11}   = {how}" for label, b, how in self.terms]
        out.append(f"      {'subtotal':<22} {_fmt(self.subtotal):>11}")
        out.append(f"      {'+ margin ' + format(MARGIN_FRACTION, '.0%'):<22} {_fmt(self.margin):>11}")
        out.append(f"      {'+ reserve left free':<22} {_fmt(self.reserve):>11}   (the 2026-10-09 near-miss had 2.2 GB free at 100 %)")
        out.append(f"      {'= REQUIRED':<22} {_fmt(self.required):>11}")
        return out


def requirement(plan: RunPlan, allow: Allowances = DEFAULT_ALLOWANCES) -> Requirement:
    interval = max(1, int(plan.checkpoint_interval))
    n_ckpt = max(0, int(plan.steps) // interval - int(plan.resume_steps) // interval)
    remaining = max(0, int(plan.steps) - int(plan.resume_steps))
    mstep = remaining / 1e6
    ck = int(plan.ckpt_bytes or 0)
    terms: List[Tuple[str, int, str]] = []
    terms.append(("periodic checkpoints", n_ckpt * ck,
                  f"{n_ckpt} x {_fmt(ck)}  (steps {plan.steps:,} // {interval:,} - resume {plan.resume_steps:,} // {interval:,})"))
    extra = allow.extra_checkpoint_copies + (1 if plan.fork else 0)
    terms.append(("final/best/aborted" + ("+fork" if plan.fork else ""), extra * ck,
                  f"{extra} x {_fmt(ck)}"))
    n_eval = math.ceil(remaining / plan.eval_freq) if plan.eval_freq and plan.eval_freq > 0 else 0
    terms.append(("eval traces", n_eval * allow.eval_trace_bytes_per_cycle,
                  f"{n_eval} cycles x {_fmt(allow.eval_trace_bytes_per_cycle)}  (ceil({remaining:,} / {plan.eval_freq:,}))"))
    n_snap = 0
    if plan.snapshots:
        n_snap = POOL_WINDOW if plan.fork else min(POOL_WINDOW, n_eval)
    terms.append(("pool snapshots", n_snap * ck, f"{n_snap} x {_fmt(ck)}  (cap {POOL_WINDOW}{', fork re-seeds the pool' if plan.fork else ''})"))
    cc = max(0, allow.compile_cache_bytes - int(plan.existing_compile_cache_bytes)) if plan.compile_cache else 0
    terms.append(("compile cache", cc,
                  f"{_fmt(allow.compile_cache_bytes)} - {_fmt(plan.existing_compile_cache_bytes)} already on disk"
                  if plan.compile_cache else "CPU run: none"))
    side = int(mstep * (allow.value_sidecar_bytes_per_mstep + allow.log_bytes_per_mstep))
    terms.append(("sidecar + tb + logs", side,
                  f"{mstep:.1f}M steps x {_fmt(allow.value_sidecar_bytes_per_mstep + allow.log_bytes_per_mstep)}/M"))
    terms.append(("fixed", allow.fixed_bytes, "metadata, ledgers, sidecar json"))
    subtotal = sum(b for _l, b, _h in terms)
    return Requirement(tuple(terms), subtotal, int(subtotal * MARGIN_FRACTION), RESERVE_BYTES,
                       ckpt_known=plan.ckpt_bytes is not None)


# --- the verdict ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskVerdict:
    status: str
    free: int = 0
    path: str = ""
    req: Optional[Requirement] = None
    ckpt_source: str = ""
    allow_flag: bool = False

    @property
    def refused(self) -> bool:
        return self.status == REFUSED_LOW

    def lines(self) -> List[str]:
        if self.status == EXEMPT_DEBUG:
            return ["disk space  : exempt — --debug (CPU smoke)"]
        assert self.req is not None
        r = self.req
        sized = ("" if r.ckpt_known else
                 "  ⚠️  checkpoint size UNKNOWN (no checkpoint, fork source or sized run to read) — "
                 "the checkpoint terms are 0; the in-run watch uses the real size from the first save")
        head = (f"{_fmt(self.free)} free on {self.path} vs {_fmt(r.required)} REQUIRED "
                f"(checkpoint {self.ckpt_source})")
        if self.status == PASS:
            out = [f"disk space  : ✓ {head}"]
        elif self.status == OPTED_OUT:
            out = [f"disk space  : ⚠️  tolerated under {OPT_OUT_FLAG} — {head}. Recorded in metadata.json "
                   "(cli_args._disk_guard); the in-run STOP is off, warnings stay on."]
        else:
            out = [f"disk space  : ✗ REFUSED — {head}.",
                   f"              SHORT by {_fmt(r.required - self.free)}. Free space (the run archive, "
                   "`du -sh models/* | sort -h`; retention: designs/research_state/models_retention_policy.md) "
                   f"or pass {OPT_OUT_FLAG} (dev / short runs only; recorded in metadata.json)."]
        if sized:
            out.append("              " + sized.strip())
        out.append("              the arithmetic:")
        out += [f"  {ln}" for ln in r.lines()]
        return out

    def to_record(self) -> dict:
        r = self.req
        return {"status": self.status, "allow_low_disk": self.allow_flag, "free_bytes": self.free,
                "required_bytes": r.required if r else None, "subtotal_bytes": r.subtotal if r else None,
                "checkpoint_source": self.ckpt_source, "path": self.path}


def check_for_run(plan: Optional[RunPlan], *, debug: bool, allow: bool,
                  free_fn: Optional[Callable[[str], int]] = None,
                  allowances: Allowances = DEFAULT_ALLOWANCES) -> DiskVerdict:
    """The one decision every surface reads. ``plan`` None only with ``debug`` (nothing to size)."""
    if debug or plan is None:
        return DiskVerdict(EXEMPT_DEBUG, allow_flag=bool(allow))
    req = requirement(plan, allowances)
    path = probe_path(plan.run_dir)
    free = int((free_fn or free_bytes)(plan.run_dir))
    ok = free >= req.required
    status = PASS if ok else (OPTED_OUT if allow else REFUSED_LOW)
    return DiskVerdict(status, free=free, path=path, req=req, ckpt_source=plan.ckpt_source,
                       allow_flag=bool(allow))


def plan_from_values(*, run_dir: str, model_path: Optional[str], steps: int, resume_steps: int,
                     checkpoint_every_steps: Optional[int], eval_freq: Optional[int], fork: bool,
                     self_play: bool = True, device: str = "auto",
                     archive_dir: Optional[str] = None) -> RunPlan:
    """Resolve a :class:`RunPlan` from flag VALUES (a namespace's, or an argv's), reading the cadence defaults
    from the code that applies them."""
    from agents.training.eval_schedule import EVAL_FREQ_STEPS
    from main.train.constants import checkpoint_interval_env_steps
    interval = checkpoint_interval_env_steps(checkpoint_every_steps)
    ef = EVAL_FREQ_STEPS if eval_freq is None else int(eval_freq)
    cb, src = find_checkpoint_bytes(run_dir, model_path, archive_dir)
    cc_dir = os.path.join(run_dir, "compile_cache")
    existing = _dir_bytes(cc_dir) if os.path.isdir(cc_dir) else 0
    return RunPlan(run_dir=run_dir, steps=int(steps), resume_steps=int(resume_steps),
                   checkpoint_interval=interval, eval_freq=ef, ckpt_bytes=cb, ckpt_source=src,
                   fork=bool(fork), snapshots=bool(self_play),
                   compile_cache=str(device or "auto").strip().lower() != "cpu",
                   existing_compile_cache_bytes=existing)


# --- the in-run watch ----------------------------------------------------------------------------

OK = "ok"
WARN = "warn"
STOP = "stop"

WARN_FACTOR = 2.0       # free below this x the next save: warn, loudly, at every save
STOP_FACTOR = 1.0       # free below this x the next save: the graceful stop


@dataclass(frozen=True)
class Decision:
    level: str
    free: int
    next_save: int
    message: str = ""


class InRunGuard:
    """The per-save disk check: ONE `shutil.disk_usage` per checkpoint, nothing between saves.

    ``next_save`` is the size of the checkpoint just written (the next one is the same model). ``stop_enabled``
    False (`--allow-low-disk`) keeps the warning and drops the stop."""

    def __init__(self, path: str, *, stop_enabled: bool = True,
                 free_fn: Optional[Callable[[str], int]] = None) -> None:
        self.path = path
        self.stop_enabled = bool(stop_enabled)
        self._free = free_fn

    def after_save(self, saved_bytes: int) -> Decision:
        nxt = max(0, int(saved_bytes))
        free = int((self._free or free_bytes)(self.path))
        if free < STOP_FACTOR * nxt and self.stop_enabled:
            return Decision(STOP, free, nxt, (
                f"[DiskGuard] 🛑 {_fmt(free)} free on {probe_path(self.path)} is below ONE more checkpoint "
                f"({_fmt(nxt)}) — stopping cleanly NOW (the checkpoint just written stands; exit "
                f"FATAL_DISK 8, no restart). Free disk space, then resume with --model."))
        if free < WARN_FACTOR * nxt:
            tail = "" if self.stop_enabled else f" ({OPT_OUT_FLAG}: the stop is off)"
            return Decision(WARN, free, nxt, (
                f"[DiskGuard] ⚠️⚠️⚠️ LOW DISK: {_fmt(free)} free on {probe_path(self.path)}, below TWO more "
                f"checkpoints ({_fmt(2 * nxt)}); the run stops cleanly below {_fmt(nxt)}.{tail}"))
        return Decision(OK, free, nxt)


def is_disk_full(exc: BaseException) -> bool:
    """An ENOSPC / EDQUOT write failure, however it was wrapped."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
