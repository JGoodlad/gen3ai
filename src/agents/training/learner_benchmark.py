"""LEARNER-ONLY PPO UPDATE BENCHMARK — where the wall time of one `train()` call goes (T13).

    python3 src/agents/training/learner_benchmark.py run --device cuda          # the real read
    python3 src/agents/training/learner_benchmark.py run --device cpu --tiny    # the code path, CPU

**WHAT IT MEASURES.** One production PPO update — the real `InstrumentedMaskablePPO.train()` of
arm C (`models/ai_v14_02_lbat_ctrl`, `final_model.zip` by default, resolved explicitly) on ONE real
98,304-sample rollout buffer, with C's own flags (its recorded `original_command`, compile-trainer
ON) — run K times from IDENTICAL state (weights, optimizer, the model's plain counters/EMAs, the
pristine pre-`train()` buffer, the RNG seeds), so every repeat is the same work. Reported:

  * `train/train_ms` (the trainer's own un-bracketed wall clock around the WHOLE call) per repeat;
  * a PHASE BREAKDOWN from a second set of K repeats with `torch.cuda.synchronize()`-bracketed
    segment marks (`instrumented_ppo.phase_hook` — None in production, see its docstring), each
    phase as seconds per update and % of that set's train_ms, plus the host time spent blocked in
    scalar reads (`.item()` / `float()`). Bracketing adds syncs, so BOTH train_ms sets are reported;
  * ABLATIONS, K repeats each: the per-term noise-scale probe OFF; all optional telemetry OFF;
    `diag_skipped` — an update `--diagnostics-every` skips, at the run's own flags (so the
    `--rank-tripwire` exemption still runs `rank/*`: the honest production saving);
    n_epochs halved; matmul precision 'high' (TF32) in a separate worker — only when the TF32
    compile-parity gate exists at HEAD (the real startup gate then decides), else SKIPPED + said;
  * one `torch.profiler` trace of a single epoch (CUDA + CPU) with GPU busy % (union of device
    intervals / wall), the top 10 kernels, and kernel launches + host syncs per micro-batch.

**THE BUFFER IS REAL, NEVER SYNTHETIC.** The masks and the aux labels drive branches in the fold,
so random obs would benchmark a different program. The worker runs the TRAINER ITSELF in-process
(`main.train_rl_agent.main()`, argv = C's recorded command re-pointed as a FORK into a scratch run
dir under `~/gen3ai_archive/learner_bench/`, never `models/`), lets the first `collect_rollouts`
and its `_on_rollout_end` label callbacks run, and intercepts the first `train()`: the pre-update
buffer is pickled beside the run dir and every repeat restores it. `--buffer` reuses a saved one.

**BENCHMARKS WARN, THEY NEVER STRETCH** (root `CLAUDE.md`): contention is REPORTED; a number is
never rescaled. A cuda run REFUSES a GPU that holds any compute process or a box running a trainer.

Worker exit: the driver writes its JSON, then terminates its own descendants (env workers, the
forkserver, bridges) by explicit PID and `os._exit`s — the trainer's own teardown (final save,
final eval) is never reached.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import gzip
import hashlib
import json
import os
import pickle
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The arm whose update this measures (T13's learner-only benchmark). Resolved against the MAIN
#: checkout's run archive — `models/` is not committed and a worktree has none.
DEFAULT_RUN = "ai_v14_02_lbat_ctrl"
DEFAULT_CHECKPOINT = "final_model.zip"
DEFAULT_OUT_ROOT = Path.home() / "gen3ai_archive" / "learner_bench"

#: Flags stripped from the recorded command before it is re-pointed: the launcher's own, the run
#: identity, and the three this script re-supplies. `--device` appears twice in C's command.
_STRIP_ALWAYS = {"--model", "--run-name", "--run-dir", "--steps", "--eval-freq", "--eval_freq",
                 "--tb-inherit", "--no-tb-inherit", "--matmul-precision", "--matmul_precision"}
#: An eval cycle must never start inside the measurement (it spawns CPU workers). 1e12 steps.
_NO_EVAL_FREQ = "1000000000000"

#: `--tiny` geometry — enough to take every branch (two epochs, a two-micro accumulation group).
TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "grad_accum_steps": 2, "n_epochs": 2,
        "k": 2, "warmup": 1}

#: The model-side stashes `train()` reads that a rollout callback fills (all cleared per rollout).
_MODEL_STASHES = ("_win_prob_lambda_metrics", "_dense_aux_metrics", "_win_prob_rollout_metrics",
                  "_fork_metrics", "_pbrs_metrics")

#: Tags read back after each call to show the repeats did the same work.
_WORK_TAGS = ("train/loss", "train/approx_kl", "train/clip_fraction", "train/value_loss",
              "train/policy_gradient_loss", "train/grad_norm", "train/entropy_loss")

_TRAINER_LITERAL = "train_rl_" + "agent"   # never spelled whole in an argv (watchers grep for it)


# ------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested in learner_benchmark_test.py)
# ------------------------------------------------------------------------------------------------

def split_flags(tokens: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """`['--a','1','--b']` -> `[('--a',['1']), ('--b',[])]`; a leading positional is dropped."""
    out: List[Tuple[str, List[str]]] = []
    for tok in tokens:
        if tok.startswith("--"):
            out.append((tok, []))
        elif out:
            out[-1][1].append(tok)
    return out


def build_trainer_argv(original_command: str, *, model_zip: str, run_dir: str, steps: int,
                       device: str, tiny: bool = False, matmul_precision: str = "highest",
                       launcher_only: Iterable[str] = ()) -> List[str]:
    """C's recorded launcher command, re-pointed as a FORK of ``model_zip`` into ``run_dir``.

    Keeps every training flag verbatim (so the fold, the arch and the compile are C's), strips the
    launcher-only flags and the run identity, and re-supplies ``--model/--run-dir/--steps``, an
    eval cadence that cannot fire, ``--no-tb-inherit`` and the matmul precision. ``device='cpu'``
    swaps both compile flags off (`--compile-trainer` refuses a CPU device by design); ``tiny``
    shrinks ``--n-envs``. The result never contains the trainer's module name.
    """
    toks = shlex.split(original_command)
    drop = set(_STRIP_ALWAYS) | set(launcher_only)
    cpu = device == "cpu"
    if cpu:
        drop |= {"--device", "--compile-trainer", "--no-compile-trainer", "--compile-opponents",
                 "--no-compile-opponents", "--compile-opponents-strict",
                 "--compile-opponents-preload", "--no-compile-opponents-preload"}
    else:
        drop |= {"--device"}
    if tiny:
        drop |= {"--n-envs", "--n_envs"}
    argv: List[str] = []
    for flag, vals in split_flags(toks):
        if flag in drop:
            continue
        argv.append(flag)
        argv.extend(vals)
    argv += ["--device", device]
    if cpu:
        argv += ["--no-compile-trainer", "--no-compile-opponents"]
    if tiny:
        argv += ["--n-envs", str(TINY["n_envs"])]
    argv += ["--model", model_zip, "--run-dir", run_dir, "--steps", str(int(steps)),
             "--eval-freq", _NO_EVAL_FREQ, "--no-tb-inherit", "--matmul-precision", matmul_precision]
    bad = [t for t in argv if _TRAINER_LITERAL in t]
    if bad:
        raise ValueError(f"trainer argv would carry the trainer's module name: {bad}")
    return argv


def recorded_steps(original_command: str) -> Optional[int]:
    """The ``--steps`` value of a recorded command, or None."""
    for flag, vals in split_flags(shlex.split(original_command)):
        if flag == "--steps" and vals:
            return int(float(vals[0]))
    return None


def gpu_busy_reason(compute_apps_csv: str, cmdlines: Dict[int, str], own_pid: int) -> Optional[str]:
    """Why the GPU is NOT idle for a measurement, or None. Pure over its two inputs.

    ``compute_apps_csv`` is `nvidia-smi --query-compute-apps=pid,process_name,used_memory
    --format=csv,noheader`; any row is a process holding a CUDA context. ``cmdlines`` maps pid ->
    /proc cmdline (NULs as spaces); any process other than ours naming the trainer module is a
    live training run, whether or not it has reached the card yet.
    """
    rows = [r.strip() for r in compute_apps_csv.splitlines() if r.strip()]
    trainers = sorted(pid for pid, cmd in cmdlines.items()
                      if pid != own_pid and _TRAINER_LITERAL in cmd)
    reasons = []
    if rows:
        reasons.append(f"{len(rows)} CUDA compute process(es) on the GPU: {rows[:4]}")
    if trainers:
        reasons.append(f"trainer process(es) running: pids {trainers[:8]}")
    return "; ".join(reasons) or None


def read_cmdlines() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                out[int(d)] = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
    return out


def descendants(pid: int) -> List[int]:
    """Every live descendant of ``pid`` (from /proc/*/stat), deepest first."""
    children: Dict[int, List[int]] = defaultdict(list)
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                stat = f.read().decode("utf-8", "replace")
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children[ppid].append(int(d))
    out: List[int] = []

    def _walk(p: int) -> None:
        for c in children.get(p, []):
            _walk(c)
            out.append(c)
    _walk(pid)
    return out


def terminate_pids(pids: Sequence[int], grace_s: float = 5.0) -> None:
    """SIGTERM each explicit pid, wait, then SIGKILL the survivors. Never a pattern kill."""
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(os.path.exists(f"/proc/{p}") for p in pids):
        time.sleep(0.1)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


class PhaseTimer:
    """The segment timer installed as `phase_hook.PHASE_HOOK`.

    Each mark books the wall time since the previous mark to the mark's NAME; ``start`` only resets
    the clock. ``sync`` (``torch.cuda.synchronize`` on cuda) runs BEFORE the clock is read, so a
    phase's number includes the device work it queued — the bracketing that makes a per-phase read
    meaningful on an async device, and the reason bracketed train_ms is reported separately.
    """

    def __init__(self, sync: Optional[Callable[[], None]] = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.sync, self.clock = sync, clock
        self.seconds: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.sequence: List[str] = []
        self._last: Optional[float] = None

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()
        self.sequence.clear()
        self._last = None

    def __call__(self, name: str) -> None:
        if self.sync is not None:
            self.sync()
        now = self.clock()
        self.sequence.append(name)
        if name == "start" or self._last is None:
            self._last = now
            return
        self.seconds[name] += now - self._last
        self.counts[name] += 1
        self._last = now


def interval_union(intervals: Iterable[Tuple[float, float]]) -> float:
    """Total length covered by a set of [start, end) intervals (overlaps counted once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _is_plain(v: Any, depth: int = 0) -> bool:
    """A value `copy.deepcopy` restores exactly and cheaply: scalars, strings, numpy, and
    lists/tuples/dicts of those. Tensors, modules, callables and objects are NOT plain."""
    if depth > 4:
        return False
    if v is None or isinstance(v, (bool, int, float, str, np.generic)):
        return True
    if isinstance(v, np.ndarray):
        return v.nbytes <= (64 << 20)
    if isinstance(v, (list, tuple)):
        return all(_is_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _is_plain(x, depth + 1) for k, x in v.items())
    return False


def _copy_arrays(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return {k: _copy_arrays(x) for k, x in v.items()}
    return copy.deepcopy(v)


#: Buffer attributes a restore must NOT overwrite: the spaces (live objects) and the device (the
#: RESTORING process's, which may differ from the one that saved the buffer).
_BUFFER_KEEP = ("observation_space", "action_space", "device")


def capture_buffer_state(buf: Any) -> Dict[str, Any]:
    """A deep copy of every rollout-buffer attribute except the live spaces/device."""
    return {k: _copy_arrays(v) for k, v in vars(buf).items() if k not in _BUFFER_KEEP}


def restore_buffer_state(buf: Any, state: Dict[str, Any]) -> None:
    """Put ``buf`` back into the captured state — arrays copied, so the pristine copy survives
    `get()`'s in-place flatten and `_align_opp_intent_labels`' in-place shift."""
    for k, v in state.items():
        setattr(buf, k, _copy_arrays(v))


def buffer_fingerprint(state: Dict[str, Any]) -> Dict[str, Any]:
    """Keys, shapes and dtypes — what a reuse must match."""
    fp: Dict[str, Any] = {}
    for k, v in sorted(state.items()):
        if isinstance(v, np.ndarray):
            fp[k] = [list(v.shape), str(v.dtype)]
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
            fp[k] = {kk: [list(x.shape), str(x.dtype)] for kk, x in sorted(v.items())}
    return fp


def capture_model_state(model: Any) -> Dict[str, Any]:
    """Weights, optimizer state and every PLAIN model attribute (counters, EMAs, dict stashes)."""
    return {
        "policy": {k: v.detach().clone() for k, v in model.policy.state_dict().items()},
        "optimizer": copy.deepcopy(model.policy.optimizer.state_dict()),
        "plain": {k: copy.deepcopy(v) for k, v in vars(model).items() if _is_plain(v)},
        "keys": set(vars(model)),
    }


def restore_model_state(model: Any, st: Dict[str, Any]) -> List[str]:
    """Restore a `capture_model_state`; returns the attrs `train()` ADDED since the capture.

    Added attrs are KEPT, deliberately: they are the lazily-created, once-per-PROCESS state (an
    announce-once flag, a lazily-built EMA dict) that a steady-state production update finds
    already present. Deleting them would make every repeat a first-update-in-process, which is not
    the update being measured. What `train()` computes is fixed by the weights, the optimizer, the
    buffer and the seeds — all restored — and the repeats' identical `train/loss` shows it."""
    model.policy.load_state_dict(st["policy"])
    model.policy.optimizer.load_state_dict(copy.deepcopy(st["optimizer"]))
    for k, v in st["plain"].items():
        setattr(model, k, copy.deepcopy(v))
    return sorted(k for k in vars(model) if k not in st["keys"])


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else None


def summarise_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median / min / max of train_ms and wall over the measured (non-warm-up) calls of a config,
    plus the per-phase medians when the calls were bracketed."""
    meas = [c for c in calls if not c.get("warmup")]
    out: Dict[str, Any] = {"n": len(meas)}
    for key in ("train_ms", "wall_s", "noise_per_term_ms", "scalar_read_s", "scalar_reads"):
        vals = [c.get(key) for c in meas if c.get(key) is not None]
        if vals:
            out[key] = {"median": _median(vals), "min": min(vals), "max": max(vals)}
    phases = [c["phases_s"] for c in meas if c.get("phases_s")]
    if phases:
        names = sorted({n for p in phases for n in p})
        med_ms = out.get("train_ms", {}).get("median")
        out["phases"] = {}
        for n in names:
            m = _median([p.get(n, 0.0) for p in phases])
            out["phases"][n] = {"s_per_update": m,
                                "pct_of_bracketed_train_ms": (100.0 * m * 1000.0 / med_ms)
                                if (m is not None and med_ms) else None}
        out["phase_counts"] = meas[0].get("phase_counts")
    # Same-work evidence: the spread of each work tag across repeats (0 on a deterministic device).
    spread = {}
    for tag in _WORK_TAGS + ("param_abs_sum",):
        vals = [c["work"].get(tag) for c in meas if c.get("work") and c["work"].get(tag) is not None]
        if len(vals) >= 2:
            spread[tag] = {"min": min(vals), "max": max(vals)}
    out["work_spread"] = spread
    return out


# ------------------------------------------------------------------------------------------------
# The worker: the trainer in-process, first `train()` intercepted
# ------------------------------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _exit_worker(code: int) -> None:
    """Leave without the trainer's teardown: stand the env-worker watchdog down FIRST (it would
    otherwise see the killed workers and `os._exit(1)` over our code), then terminate every
    descendant by explicit pid, then exit."""
    sys.stdout.flush()
    sys.stderr.flush()
    ev = _WORKER.get("_watchdog_event")
    if ev is not None:
        ev.set()
        time.sleep(1.5)                    # the watchdog polls on a 1 s wait of this event
    terminate_pids(descendants(os.getpid()))
    os._exit(code)


def save_buffer(path: Path, buf_state: Dict[str, Any], stash: Dict[str, Any],
                meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"buffer": buf_state, "stash": stash, "meta": meta}, f, protocol=5)
    tmp.replace(path)
    _write_json(path.with_suffix(".json"),
                {**meta, "fingerprint": buffer_fingerprint(buf_state), "bytes": path.stat().st_size})


def load_buffer(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


@contextmanager
def _scalar_read_meter(acc: Dict[str, float]):
    """Count + time host-blocking scalar reads (`Tensor.item`, `float(tensor)`) for the block."""
    import torch
    orig_item, orig_float = torch.Tensor.item, torch.Tensor.__float__

    def _item(self):
        t0 = time.perf_counter()
        try:
            return orig_item(self)
        finally:
            acc["s"] += time.perf_counter() - t0
            acc["n"] += 1

    def _float(self):
        t0 = time.perf_counter()
        try:
            return orig_float(self)
        finally:
            acc["s"] += time.perf_counter() - t0
            acc["n"] += 1
    torch.Tensor.item, torch.Tensor.__float__ = _item, _float
    try:
        yield
    finally:
        torch.Tensor.item, torch.Tensor.__float__ = orig_item, orig_float


@contextmanager
def _config(model: Any, name: str, base_epochs: int):
    """Apply one ablation for the block and undo it after. Every switch is reported by name."""
    from agents.training.instrumented_ppo import ppo as ppo_mod
    _missing = object()
    saved_attrs = {k: getattr(model, k, _missing)
                   for k in ("noise_scale_per_term", "n_epochs", "diagnostics_every",
                             "_diagnostics_ran_in_process")}
    saved_mod = {k: getattr(ppo_mod, k) for k in ("grad_balance_metrics", "rank_probe",
                                                  "edge_family_metrics", "cell_family_metrics")}
    try:
        if name in ("noise_probe_off", "telemetry_off"):
            model.noise_scale_per_term = False
        if name == "telemetry_off":
            # A NON-EMPTY dict: each probe site is guarded by `if not <metrics>`, so an empty
            # return would re-run the guard on every minibatch instead of switching the probe off.
            def _off(*_a, **_k):
                return {"bench/telemetry_off": 1.0}
            for k in saved_mod:
                setattr(ppo_mod, k, _off)
        if name == "diag_skipped":
            # gen3_diagnostics_cadence_v1: the update `--diagnostics-every N` SKIPS, at the run's
            # own flags — so a load-bearing exemption (`--rank-tripwire` keeps `rank/*`) still
            # runs, exactly as it would in production. Past the process's first update, and a
            # cadence no restored rollout index is a multiple of.
            model.diagnostics_every = 2 ** 31 - 1
            model._diagnostics_ran_in_process = True
        if name == "epochs_half":
            model.n_epochs = max(1, base_epochs // 2)
        if name == "profile":
            model.n_epochs = PROFILE_EPOCHS
        if name == "ridealong":
            # gen3_ridealong_heads_v1: the X26 baseline's four DETACHED heads at the baseline spec,
            # attached to the loaded policy exactly as a v126 build makes them (fresh, private RNG,
            # on the model's device) — rebuilt per call, so every repeat does the same work. The
            # checkpoint predates v126, so the heads cannot come from the zip itself.
            from agents.model.ridealong_heads import RideAlongSpec, build_ridealong
            fe = model.policy.features_extractor
            obs_dim = int(model.policy.observation_space["observation"].shape[0])
            model.policy.ridealong = build_ridealong(
                fe, obs_dim=obs_dim, spec=RideAlongSpec(ensemble=5, rnd=True, adv=5, opp=5)
            ).to(model.device)
            model._ridealong_opt = None
            model._ridealong_disabled = False
        yield
    finally:
        if name == "ridealong":
            model.policy.ridealong = None
            model._ridealong_opt = None
        for k, v in saved_attrs.items():
            if v is _missing:
                if k in vars(model):
                    delattr(model, k)
            else:
                setattr(model, k, v)
        for k, v in saved_mod.items():
            setattr(ppo_mod, k, v)


def _one_call(model: Any, orig_train: Callable, *, name: str, pristine: Dict[str, Any],
              state0: Dict[str, Any], seed: int, base_epochs: int, bracketed: bool,
              warmup: bool, profile_to: Optional[Path] = None) -> Dict[str, Any]:
    import torch
    from agents.training.instrumented_ppo import phase_hook
    cuda = str(model.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else None
    restore_buffer_state(model.rollout_buffer, pristine)
    added = restore_model_state(model, state0)
    seed_all(seed)
    model.logger.name_to_value.clear()
    timer = PhaseTimer(sync=sync)
    scal = {"s": 0.0, "n": 0}
    rec: Dict[str, Any] = {"config": name, "warmup": warmup, "bracketed": bracketed,
                           "restored_added_attrs": added}
    with _config(model, name, base_epochs):
        rec["n_epochs"] = int(model.n_epochs)
        if sync:
            sync()
        t0 = time.perf_counter()
        if profile_to is not None:
            rec["windows"] = _profile_windows(model, orig_train, profile_to, cuda, sync)
            rec["wall_s"] = time.perf_counter() - t0
        elif bracketed:
            with phase_hook.installed(timer), _scalar_read_meter(scal):
                orig_train(model)
            if sync:
                sync()
            rec["wall_s"] = time.perf_counter() - t0
        else:
            orig_train(model)
            if sync:
                sync()
            rec["wall_s"] = time.perf_counter() - t0
    ntv = model.logger.name_to_value
    rec["train_ms"] = float(ntv["train/train_ms"]) if "train/train_ms" in ntv else None
    rec["noise_per_term_ms"] = (float(ntv["train/noise_per_term_ms"])
                                if "train/noise_per_term_ms" in ntv else None)
    work = {t: float(ntv[t]) for t in _WORK_TAGS if t in ntv}
    with torch.no_grad():
        work["param_abs_sum"] = float(sum(p.double().abs().sum() for p in model.policy.parameters()))
    rec["work"] = work
    if bracketed:
        rec["phases_s"] = dict(timer.seconds)
        rec["phase_counts"] = dict(timer.counts)
        rec["scalar_read_s"], rec["scalar_reads"] = scal["s"], scal["n"]
    return rec


def _n_micro(model: Any) -> int:
    n = int(model.rollout_buffer.buffer_size) * int(model.rollout_buffer.n_envs)
    return -(-n // int(model.batch_size))


#: The profiled call runs this many epochs, one profiler WINDOW each (plus a tail window for the
#: post-loop logging). Window 0 holds the call's setup, the once-per-call probes and the per-term
#: noise sampler (epoch 0's first accumulation group); the LAST epoch window is the steady state.
PROFILE_EPOCHS = 3


def _profile_windows(model: Any, orig_train: Callable, out_dir: Path, cuda: bool,
                     sync: Optional[Callable[[], None]]) -> List[Dict[str, Any]]:
    """One `train()` under `torch.profiler`, a window per epoch: `prof.step()` is driven from the
    phase hook's ``epoch_end`` mark (no sync is added — the windows see the un-bracketed update).
    Traces are exported for window 0 and the last epoch window."""
    from torch.profiler import ProfilerActivity, profile, schedule
    from agents.training.instrumented_ppo import phase_hook
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    n_micro = _n_micro(model)
    windows: List[Dict[str, Any]] = []
    stamps = [time.perf_counter()]

    def _ready(p: Any) -> None:
        i = len(windows)
        label = f"epoch{i}" if i < PROFILE_EPOCHS else "tail"
        keep = i in (0, PROFILE_EPOCHS - 1)
        w = summarise_profile(p, n_micro if label != "tail" else 1,
                              out_dir / f"profile_{label}.trace.json.gz" if keep else None)
        w["window"] = label
        w["cpu_wall_s"] = stamps[-1] - stamps[-2] if len(stamps) >= 2 else None
        windows.append(w)

    with profile(activities=acts, record_shapes=False,
                 schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
                 on_trace_ready=_ready) as prof:
        def _mark(name: str) -> None:
            if name == "epoch_end":
                stamps.append(time.perf_counter())
                prof.step()
        with phase_hook.installed(_mark):
            orig_train(model)
        if sync:
            sync()
        stamps.append(time.perf_counter())
    return windows


def summarise_profile(prof: Any, n_micro: int, trace_path: Optional[Path]) -> Dict[str, Any]:
    """GPU busy %, top kernels, kernel launches and host syncs per micro-batch, from one window.

    GPU busy = the UNION of device-activity intervals (kernels + memcpy/memset) over the window's
    span (first to last event start/end, CPU or device) — how much of the window the card had any
    work. Kernel time alone (``kernel_time_s``) can exceed the union when streams overlap."""
    from torch.autograd import DeviceType
    out: Dict[str, Any] = {"n_micro": n_micro}
    if trace_path is not None:
        try:
            raw = trace_path.with_suffix("")
            prof.export_chrome_trace(str(raw))
            with open(raw, "rb") as fi, gzip.open(trace_path, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            raw.unlink()
            out["trace"] = str(trace_path)
        except Exception as e:  # noqa: BLE001 — the summary below does not depend on the file
            out["trace_error"] = repr(e)
    events = list(prof.events())
    dev = [e for e in events if e.device_type == DeviceType.CUDA]
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    t_all = [(e.time_range.start, e.time_range.end) for e in events]
    span_us = (max(b for _, b in t_all) - min(a for a, _ in t_all)) if t_all else 0.0
    out["profiled_span_s"] = span_us / 1e6
    if dev:
        busy_us = interval_union((e.time_range.start, e.time_range.end) for e in dev)
        kern = [e for e in dev if not e.name.startswith(("Memcpy", "Memset"))]
        out["gpu_busy_s"] = busy_us / 1e6
        out["gpu_busy_pct"] = 100.0 * busy_us / span_us if span_us else None
        by: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
        for e in kern:
            by[e.name][0] += 1
            by[e.name][1] += e.time_range.end - e.time_range.start
        top = sorted(by.items(), key=lambda kv: -kv[1][1])[:10]
        tot = sum(v[1] for v in by.values()) or 1.0
        out["top_kernels"] = [{"name": n[:160], "count": c, "total_ms": t / 1e3,
                               "pct_of_kernel_time": 100.0 * t / tot} for n, (c, t) in top]
        out["kernels"] = len(kern)
        out["memcpy_memset"] = len(dev) - len(kern)
        out["kernel_time_s"] = tot / 1e6
    else:
        out["gpu_busy_pct"] = None
        out["note"] = "no CUDA activity (cpu device)"
    names = defaultdict(int)
    for e in cpu:
        names[e.name] += 1
    launches = sum(v for k, v in names.items()
                   if k in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC",
                            "cuLaunchKernelEx"))
    syncs = {k: v for k, v in names.items() if "Synchronize" in k}
    scalar = names.get("aten::_local_scalar_dense", 0)
    memcpy = {k: v for k, v in names.items() if k.startswith(("cudaMemcpy", "cuMemcpy"))}
    per = float(max(1, n_micro))
    out["per_micro"] = {
        "kernel_launch_calls": launches / per,
        "device_kernels": out.get("kernels", 0) / per,
        "sync_calls": {k: v / per for k, v in syncs.items()},
        "scalar_reads_local_scalar_dense": scalar / per,
        "memcpy_calls": {k: v / per for k, v in memcpy.items()},
    }
    return out


def _bench_train(self) -> None:
    """Replaces `train()` in the worker: save/restore the buffer, run every config, exit."""
    cfg = _WORKER
    out = Path(cfg["out_dir"])
    res: Dict[str, Any] = {"worker": cfg["name"], "started": _dt.datetime.now().isoformat(),
                           "argv": cfg["trainer_argv"], "calls": []}
    try:
        import torch
        orig_train = cfg["_orig_train"]
        if cfg.get("tiny"):
            self.batch_size = TINY["batch_size"]
            self.grad_accum_steps = TINY["grad_accum_steps"]
            self.n_epochs = TINY["n_epochs"]
        buf = self.rollout_buffer
        pristine = capture_buffer_state(buf)
        if not cfg.get("buffer_in"):
            stash = {k: copy.deepcopy(getattr(self, k, None)) for k in _MODEL_STASHES}
            meta = {"checkpoint": cfg["model_zip"], "checkpoint_sha256": cfg["model_sha256"],
                    "num_timesteps": int(self.num_timesteps), "n_steps": int(buf.buffer_size),
                    "n_envs": int(buf.n_envs), "trainer_argv": cfg["trainer_argv"],
                    "git_head": cfg.get("git_head"), "created": _dt.datetime.now().isoformat()}
            save_buffer(Path(cfg["buffer_out"]), pristine, stash, meta)
            res["buffer_saved"] = cfg["buffer_out"]
        base_epochs = int(self.n_epochs)
        res["geometry"] = {"samples": int(buf.buffer_size) * int(buf.n_envs),
                           "n_steps": int(buf.buffer_size), "n_envs": int(buf.n_envs),
                           "batch_size": int(self.batch_size),
                           "grad_accum_steps": int(getattr(self, "grad_accum_steps", 1)),
                           "n_epochs": base_epochs, "micro_batches_per_epoch": _n_micro(self),
                           "device": str(self.device),
                           "matmul_precision": torch.get_float32_matmul_precision(),
                           # `--compile-trainer` patches the BOUND forward on the instance.
                           "compiled_extractor": "forward" in vars(self.policy.features_extractor),
                           "noise_scale_per_term": bool(getattr(self, "noise_scale_per_term", True)),
                           "capacity_telemetry": bool(getattr(self, "capacity_telemetry", False))}
        state0 = capture_model_state(self)
        seed = int(cfg.get("seed", 1234))
        common = dict(pristine=pristine, state0=state0, seed=seed, base_epochs=base_epochs)
        for name, bracketed in cfg["plan"]:
            reps = [True] * int(cfg["warmup"]) + [False] * int(cfg["k"])
            for i, wu in enumerate(reps):
                rec = _one_call(self, orig_train, name=name, bracketed=bracketed, warmup=wu,
                                **common)
                rec["index"] = i
                res["calls"].append(rec)
                print(f"[learner_bench] {cfg['name']}:{name}{'[bracketed]' if bracketed else ''} "
                      f"#{i}{' (warm-up)' if wu else ''}: train_ms={rec['train_ms']:.0f} "
                      f"wall={rec['wall_s']:.2f}s", flush=True)
                _write_json(out / f"result_{cfg['name']}.json", res)
        if cfg.get("profile"):
            rec = _one_call(self, orig_train, name="profile", bracketed=False, warmup=False,
                            profile_to=out, **common)
            res["profile"] = rec
            for w in rec.get("windows", []):
                print(f"[learner_bench] profile {w['window']}: busy {w.get('gpu_busy_pct')}% "
                      f"span {w.get('profiled_span_s')}s per-micro {json.dumps(w.get('per_micro'))}",
                      flush=True)
        res["finished"] = _dt.datetime.now().isoformat()
        res["status"] = "ok"
        _write_json(out / f"result_{cfg['name']}.json", res)
        _exit_worker(0)
    except BaseException as e:  # noqa: BLE001 — report, then leave by explicit PID
        res["status"] = "error"
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()
        try:
            _write_json(out / f"result_{cfg['name']}.json", res)
        finally:
            print(res["traceback"], flush=True)
            _exit_worker(3)


def _bench_collect(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
    """Replaces `collect_rollouts` when a saved buffer is reused: restore it, return True."""
    saved = load_buffer(Path(_WORKER["buffer_in"]))
    live = buffer_fingerprint(capture_buffer_state(rollout_buffer))
    want = buffer_fingerprint(saved["buffer"])
    live_keys = set(live.get("observations", {}))
    want_keys = set(want.get("observations", {}))
    if live_keys != want_keys:
        raise RuntimeError(f"saved buffer obs keys differ from this model's: "
                           f"missing={sorted(live_keys - want_keys)} extra={sorted(want_keys - live_keys)}")
    restore_buffer_state(rollout_buffer, saved["buffer"])
    for k, v in saved["stash"].items():
        setattr(self, k, v)
    print(f"[learner_bench] reused buffer {_WORKER['buffer_in']} "
          f"({saved['meta'].get('n_steps')}x{saved['meta'].get('n_envs')})", flush=True)
    return True


def _tiny_learn(orig_learn: Callable) -> Callable:
    def learn(self, *a, **k):
        # `--n-steps` is INERT on a resume (SB3 restores the checkpoint's), so the tiny rollout is
        # made by rebuilding the buffer at the tiny length with the live buffer's own class/args.
        b = self.rollout_buffer
        self.n_steps = TINY["n_steps"]
        self.rollout_buffer = type(b)(TINY["n_steps"], b.observation_space, b.action_space,
                                      device=b.device, gamma=b.gamma, gae_lambda=b.gae_lambda,
                                      n_envs=b.n_envs, **(getattr(self, "rollout_buffer_kwargs", None) or {}))
        return orig_learn(self, *a, **k)
    return learn


def worker_main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    _WORKER.update(cfg)
    os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)   # the ablation owns this switch
    import asyncio
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    _WORKER["_orig_train"] = InstrumentedMaskablePPO.train
    InstrumentedMaskablePPO.train = _bench_train
    if cfg.get("buffer_in"):
        InstrumentedMaskablePPO.collect_rollouts = _bench_collect
    elif cfg.get("tiny"):
        InstrumentedMaskablePPO.learn = _tiny_learn(InstrumentedMaskablePPO.learn)
    import importlib
    # `model_build` holds its OWN binding of the watchdog starter (a from-import), so the wrapper
    # goes on that module: it records the shutdown event `_exit_worker` must set.
    from main.train import model_build as _mb
    _orig_wd = _mb.start_subprocess_watchdog

    def _wd(env, label="env", shutdown_event=None):
        _WORKER["_watchdog_event"] = shutdown_event
        return _orig_wd(env, label=label, shutdown_event=shutdown_event)
    _mb.start_subprocess_watchdog = _wd
    trainer = importlib.import_module("main." + _TRAINER_LITERAL)
    sys.argv = ["learner_benchmark[worker]"] + list(cfg["trainer_argv"])
    asyncio.run(trainer.main())
    # Reaching here means `learn()` returned without calling `train()` — nothing was measured.
    _write_json(Path(cfg["out_dir"]) / f"result_{cfg['name']}.json",
                {"worker": cfg["name"], "status": "error",
                 "error": "the trainer returned without reaching train()"})
    _exit_worker(4)


# ------------------------------------------------------------------------------------------------
# The parent: preflight, the workers, the summary
# ------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tf32_gate_available() -> Tuple[bool, str]:
    """Is the reduced-precision compile-parity gate (gen3_tf32_parity_gate_v1) in this tree?"""
    from agents.model import compile_trainer as ct
    if hasattr(ct, "fp32_reference") and hasattr(ct, "_TF32_K"):
        return True, "gen3_tf32_parity_gate_v1 present"
    return False, ("the TF32 compile-parity gate (gen3_tf32_parity_gate_v1) has NOT landed at this "
                   "HEAD — the fp32 gate (1e-4) refuses a TF32 compile, as the ai_v14_04_lbat_t32 "
                   "launch showed; TF32 ablation SKIPPED")


def _run_worker(name: str, cfg: Dict[str, Any], out: Path, env: Dict[str, str],
                timeout_min: float) -> Dict[str, Any]:
    cfg_path = out / f"worker_{name}.json"
    _write_json(cfg_path, cfg)
    log = out / f"worker_{name}.log"
    # `-m`, not the script path: a script puts its own directory (src/agents/training) first on
    # sys.path, where a sibling module could shadow a top-level name.
    cmd = [sys.executable, "-u", "-m", "agents.training.learner_benchmark", "_worker", str(cfg_path)]
    print(f"[learner_bench] worker {name}: log {log}", flush=True)
    t0 = time.monotonic()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                cwd=cfg["cwd"])
        try:
            rc = proc.wait(timeout=timeout_min * 60.0)
        except subprocess.TimeoutExpired:
            pids = descendants(proc.pid) + [proc.pid]
            terminate_pids(pids)
            rc = -9
            print(f"[learner_bench] worker {name} exceeded {timeout_min} min — killed pids {pids}",
                  flush=True)
    res_path = out / f"result_{name}.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"status": "missing"}
    res["exit_code"] = rc
    res["worker_wall_s"] = time.monotonic() - t0
    if rc != 0 and res.get("status") == "ok":
        res["status"] = f"exit {rc}"
    return res


def summarise(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for res in results.values():
        for c in res.get("calls", []):
            key = c["config"] + ("[bracketed]" if c.get("bracketed") else "")
            by_cfg[key].append(c)
    configs = {k: summarise_calls(v) for k, v in by_cfg.items()}
    base = configs.get("baseline", {}).get("train_ms", {}).get("median")
    for k, s in configs.items():
        med = s.get("train_ms", {}).get("median")
        if base and med is not None:
            s["delta_vs_baseline_ms"] = med - base
            s["ratio_vs_baseline"] = med / base
    br = configs.get("baseline[bracketed]", {})
    if base and br.get("phases"):
        for p in br["phases"].values():
            if p["s_per_update"] is not None:
                p["pct_of_unbracketed_train_ms"] = 100.0 * p["s_per_update"] * 1000.0 / base
    fit = None
    half = configs.get("epochs_half", {})
    geo = next((r.get("geometry") for r in results.values() if r.get("geometry")), None)
    if base and half.get("train_ms") and geo:
        e1, e2 = geo["n_epochs"], max(1, geo["n_epochs"] // 2)
        if e1 != e2:
            per = (base - half["train_ms"]["median"]) / (e1 - e2)
            fit = {"per_epoch_ms": per, "fixed_ms": base - e1 * per,
                   "per_micro_ms": per / geo["micro_batches_per_epoch"],
                   "points": {e1: base, e2: half["train_ms"]["median"]}}
    prof = next((r["profile"].get("windows") for r in results.values()
                 if r.get("profile")), None)
    return {"configs": configs, "two_point_epoch_fit": fit, "geometry": geo, "profile": prof}


def print_summary(summary: Dict[str, Any]) -> None:
    cf = summary["configs"]
    print("\n=== learner-only PPO update — train_ms (median over K; min–max) ===")
    for k in sorted(cf):
        t = cf[k].get("train_ms")
        if not t:
            continue
        d = cf[k].get("delta_vs_baseline_ms")
        print(f"  {k:28s} {t['median'] / 1e3:8.2f} s  ({t['min'] / 1e3:.2f}–{t['max'] / 1e3:.2f})"
              + (f"   Δ {d / 1e3:+.2f} s" if d is not None and k != "baseline" else ""))
    br = cf.get("baseline[bracketed]", {}).get("phases")
    if br:
        print("\n=== phase breakdown (bracketed; s per update, % of bracketed / un-bracketed) ===")
        for n, p in sorted(br.items(), key=lambda kv: -(kv[1]["s_per_update"] or 0)):
            print(f"  {n:12s} {p['s_per_update']:8.3f} s  {p['pct_of_bracketed_train_ms'] or 0:5.1f}%"
                  f"  {p.get('pct_of_unbracketed_train_ms') or 0:5.1f}%")
    rb = cf.get("ridealong[bracketed]", {}).get("phases", {}).get("ridealong")
    if rb:
        print(f"\n  ride-along heads' own step (bracketed `ridealong` phase): "
              f"{rb['s_per_update']:.3f} s/update = {rb['pct_of_bracketed_train_ms'] or 0:.1f}% "
              "of that update")
    if summary.get("two_point_epoch_fit"):
        f = summary["two_point_epoch_fit"]
        print(f"\n  two-point fit: {f['per_epoch_ms'] / 1e3:.2f} s/epoch "
              f"({f['per_micro_ms']:.1f} ms/micro), fixed {f['fixed_ms'] / 1e3:.2f} s")
    for w in summary.get("profile") or []:
        print(f"\n  profile {w['window']}: GPU busy {w.get('gpu_busy_pct')}% of "
              f"{w.get('profiled_span_s')} s; per micro: {json.dumps(w.get('per_micro'))}")
        for kx in (w.get("top_kernels") or [])[:5]:
            print(f"      {kx['total_ms']:9.1f} ms  x{kx['count']:<6d} {kx['name'][:90]}")


def run_main(a: argparse.Namespace) -> int:
    from utils.git import get_git_hash
    from utils.paths import main_models_dir, repo_root
    from utils.contention import describe_contention, warn_if_contended
    from main.checkargs import LAUNCHER_ONLY

    device = a.device
    tiny = bool(a.tiny)
    if device == "cuda" and tiny:
        print("--tiny is the CPU code-path exercise; refusing it on cuda.")
        return 2
    env = dict(os.environ)
    pp = str(repo_root() / "src")
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""          # the build phase must never touch the card
    else:
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"REFUSED: cannot query nvidia-smi ({e!r}) — an idle GPU cannot be confirmed.")
            return 2
        why = gpu_busy_reason(apps, read_cmdlines(), os.getpid())
        if why:
            print(f"REFUSED: the GPU is not idle — {why}. This is a speed measurement; run it on "
                  f"an idle card.")
            return 2
    contended = warn_if_contended("learner_benchmark")

    models = main_models_dir()
    if a.model:
        model_zip = Path(a.model).resolve()
    else:
        if models is None:
            print("REFUSED: no models/ archive on this box and no --model given.")
            return 2
        model_zip = (models / a.run / DEFAULT_CHECKPOINT).resolve()
    if not model_zip.is_file():
        print(f"REFUSED: checkpoint not found: {model_zip}")
        return 2
    run_dir_src = model_zip.parent if model_zip.parent.name != "checkpoints" else model_zip.parent.parent
    meta = json.loads((run_dir_src / "metadata.json").read_text())
    original = meta.get("original_command")
    if not original:
        print(f"REFUSED: {run_dir_src}/metadata.json has no original_command.")
        return 2

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_root).expanduser().resolve() / f"{stamp}_{device}{'_tiny' if tiny else ''}"
    forbidden = [p for p in (models, repo_root() / "data") if p is not None]
    if any(str(out).startswith(str(Path(p).resolve())) for p in forbidden):
        print(f"REFUSED: --out-root {out} is under models/ or data/.")
        return 2
    out.mkdir(parents=True, exist_ok=False)

    steps = (recorded_steps(original) or 0) + 10 * 48 * 2048
    k = TINY["k"] if tiny and a.k is None else (a.k if a.k is not None else 5)
    warmup = TINY["warmup"] if tiny and a.warmup is None else (a.warmup if a.warmup is not None else 1)
    sha = _sha256(model_zip)
    head = get_git_hash()
    base_cfg = {"model_zip": str(model_zip), "model_sha256": sha, "git_head": head,
                "tiny": tiny, "k": k, "warmup": warmup, "seed": a.seed, "out_dir": str(out),
                "cwd": str(models.parent if models is not None else repo_root())}
    buffer_path = Path(a.buffer).resolve() if a.buffer else out / "rollout_buffer.pkl"

    plan_a: List[Tuple[str, bool]] = [("baseline", False), ("baseline", True)]
    if not a.no_ablations:
        plan_a += [("noise_probe_off", False), ("telemetry_off", False), ("diag_skipped", False),
                   ("epochs_half", False)]
    if a.ridealong:
        # the X26 ride-along heads' added update time: unbracketed (train_ms) and bracketed (the
        # `ridealong` phase), against the `baseline` pair above on the same buffer and state.
        plan_a += [("ridealong", False), ("ridealong", True)]
    argv_a = build_trainer_argv(original, model_zip=str(model_zip), run_dir=str(out / "run_main"),
                                steps=steps, device=device, tiny=tiny, launcher_only=LAUNCHER_ONLY)
    cfg_a = {**base_cfg, "name": "main", "plan": plan_a, "trainer_argv": argv_a,
             "profile": not a.no_profile,
             "buffer_in": str(buffer_path) if a.buffer else None,
             "buffer_out": None if a.buffer else str(buffer_path)}
    header = {"started": _dt.datetime.now().isoformat(), "git_head": head, "device": device,
              "tiny": tiny, "checkpoint": str(model_zip), "checkpoint_sha256": sha, "k": k,
              "warmup_per_config": warmup, "contention_at_start": describe_contention(),
              "contended_at_start": contended, "out_dir": str(out),
              "host": os.uname().nodename}
    results: Dict[str, Dict[str, Any]] = {}
    results["main"] = _run_worker("main", cfg_a, out, env, a.worker_timeout_min)
    print(f"[learner_bench] main worker: {results['main'].get('status')} "
          f"(exit {results['main'].get('exit_code')})", flush=True)

    tf32: Dict[str, Any] = {"ran": False}
    ok, why = tf32_gate_available()
    if device != "cuda":
        tf32["reason"] = "TF32 is a CUDA matmul mode; not applicable on cpu"
    elif a.no_ablations or a.skip_tf32:
        tf32["reason"] = "skipped by flag"
    elif not ok:
        tf32["reason"] = why
    elif not buffer_path.exists():
        tf32["reason"] = "no saved buffer from the main worker"
    else:
        argv_b = build_trainer_argv(original, model_zip=str(model_zip), run_dir=str(out / "run_tf32"),
                                    steps=steps, device=device, tiny=tiny,
                                    launcher_only=LAUNCHER_ONLY, matmul_precision="high")
        cfg_b = {**base_cfg, "name": "tf32", "plan": [("tf32", False)], "trainer_argv": argv_b,
                 "profile": False, "buffer_in": str(buffer_path), "buffer_out": None}
        results["tf32"] = _run_worker("tf32", cfg_b, out, env, a.worker_timeout_min)
        tf32 = {"ran": True, "status": results["tf32"].get("status"),
                "exit_code": results["tf32"].get("exit_code"), "gate": why}
        if results["tf32"].get("exit_code") not in (0, None):
            tf32["reason"] = (f"the TF32 worker exited {results['tf32'].get('exit_code')} — read "
                              f"worker_tf32.log (a FATAL_CONFIG there is the parity gate refusing)")
    summary = summarise(results)
    report = {**header, "finished": _dt.datetime.now().isoformat(),
              "contention_at_end": describe_contention(), "tf32": tf32,
              "workers": {n: {k2: r.get(k2) for k2 in ("status", "exit_code", "worker_wall_s",
                                                        "error", "buffer_saved", "geometry")}
                          for n, r in results.items()},
              "summary": summary}
    _write_json(out / "learner_bench.json", report)
    print_summary(summary)
    print(f"\n[learner_bench] report: {out / 'learner_bench.json'}")
    if a.publish:
        pub = Path(a.publish)
        pub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "learner_bench.json", pub / f"learner_bench_{stamp}_{device}.json")
        print(f"[learner_bench] copied to {pub}")
    bad = [n for n, r in results.items() if r.get("status") != "ok" and n == "main"]
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="preflight, the main worker, the TF32 worker, the report")
    r.add_argument("--device", choices=("cuda", "cpu"), required=True)
    r.add_argument("--tiny", action="store_true",
                   help=f"CPU code-path exercise: {TINY}")
    r.add_argument("--run", default=DEFAULT_RUN,
                   help=f"run under the main checkout's models/ (default {DEFAULT_RUN})")
    r.add_argument("--model", default=None,
                   help=f"explicit checkpoint .zip (default <run>/{DEFAULT_CHECKPOINT})")
    r.add_argument("--buffer", default=None, help="reuse a saved rollout_buffer.pkl")
    r.add_argument("--k", type=int, default=None, help="measured repeats per config (default 5)")
    r.add_argument("--warmup", type=int, default=None, help="warm-up calls per config (default 1)")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--no-ablations", action="store_true")
    r.add_argument("--ridealong", action="store_true",
                   help="also time the update with the X26 ride-along heads attached "
                        "(ensemble 5, rnd, adv 5, opp 5; gen3_ridealong_heads_v1)")
    r.add_argument("--skip-tf32", action="store_true")
    r.add_argument("--no-profile", action="store_true")
    r.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    r.add_argument("--publish", default=None,
                   help="also copy the report JSON into this directory (e.g. the measurements dir)")
    r.add_argument("--worker-timeout-min", type=float, default=180.0,
                   help="a hung worker is killed (by explicit PID) after this long")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("cfg")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "_worker":
        worker_main(a.cfg)
        return 0
    return run_main(a)


if __name__ == "__main__":
    sys.exit(main())
