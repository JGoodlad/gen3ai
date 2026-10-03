"""THE UPDATE FIT CHECK (`gen3_update_fit_v1`, 2026-10-01) — a MEASURED, PREDICTIVE startup gate on
the learner's CUDA memory: does the first real PPO update fit on this card, with the declared headroom?

WHY. `gen3_cuda_ledger_v1`'s device-batch check was a LOWER bound (the regions gate's one R1 step,
4.3 GiB at B = 2048) and had a measured false pass: N = 256 with the X26 ride-along heads passed it
and then ran OUT OF MEMORY in its first update (10.19 GiB allocated on an 11.63 GiB card). The real
update needs the R1 step PLUS the eager tail's forwards (ride-along heads, TD-aux, the cf block), the
first-update diagnostics, the micro-batch staging and the optimizer step. No sum of startup rows
predicts that, so this check RUNS it.

WHAT. At startup — after T2, the compiled regions and the declared optimizer state are up, i.e. with
everything the steady state holds on the card — `dry_update` runs ONE REAL `train()` (the production
code path, unchanged: R1's compiled micro-step at the production micro size, the eager tail, the
backward, the accumulation, `clip_grad_norm` and the optimizer steps, the device-batch staging, the
first-update diagnostics) for ONE epoch over a FIXTURE rollout of the buffer's full declared shape
(the K9 learner golden's real labelled rows when they fit this surface, else the committed real-obs
fixture — `compile_regions.r1_batch`, the regions gate's own source, tiled), then RESTORES the learner
exactly (`preserved_learner`): every module's state in place, every optimizer's state in place, the
python / numpy / torch / CUDA RNG streams, the model's and every module's instance attributes, the
logger. The run that follows is BIT-IDENTICAL to one without the check (`update_fit_test`, on the
learner golden). One epoch suffices: every epoch runs the same micro-batches through the same code,
and the first carries the once-per-update diagnostics (the cadence's "first update of the process").

THE VERDICT (deterministic: a pass or a typed FATAL, no warn tier). The allocator's PEAK RESERVED
during the dry update is the update's DEMAND. With the card this process can have = its reserved bytes
now + the device's free bytes now (other processes and the CUDA context already excluded):

    headroom = (reserved_now + device_free_now) - demand   must be >= FIT_HEADROOM_MIB (1,024 MiB)

1,024 MiB = K6's 512 MiB ceiling margin + the sizing study's D-6 rule (ceiling - demand >= 512 MiB), so
a pass here is exactly D-6's rule at the first update. An OUT-OF-MEMORY inside the dry update is the
same FATAL (the update cannot fit at all). Both raise `UpdateWontFit` (FATAL_CONFIG: a restart would
replay it), naming the numbers and the levers.

What it does NOT see: an eval cycle's or a pool refresh's transient (both are DECLARED at startup —
T2's eval slots and slot loads allocate nothing new — and K6's memory trend watches the run), and a
data-dependent eager branch whose size on real rows exceeds the fixture's (the eager tail's masked
selects scale with label presence; the golden rows are real labelled rows). The validation launch
compares the dry update's demand with the real first update's (`designs/training/learner_lifecycle.md`).

`GEN3AI_UPDATE_FIT_SNAPSHOT=<path.pickle>` additionally records the CUDA allocator's history during
the dry update and dumps `torch.cuda.memory._snapshot()` there (the per-op attribution read; heavy,
off by default).
"""
from __future__ import annotations

import contextlib
import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

MiB = 1 << 20
SCHEMA = "gen3_update_fit_v1"
#: Device headroom the first update must leave: K6's ceiling margin (512) + D-6's rule (512).
FIT_HEADROOM_MIB = 1024.0
SNAPSHOT_ENV = "GEN3AI_UPDATE_FIT_SNAPSHOT"


class UpdateWontFit(RuntimeError):
    """The first real PPO update does not fit on this card with the declared headroom: measured by a
    dry update at startup, refused there (FATAL_CONFIG), never discovered as an OOM at update 1."""


@dataclass
class FitReading:
    demand_mib: float            # peak reserved during the dry update
    peak_alloc_mib: float        # peak allocated during it
    reserved_mib: float          # reserved after it (the allocator keeps its cache)
    device_free_mib: float       # device free after it
    card_mib: float              # reserved + free: what this process can have
    headroom_mib: float          # card - demand
    floor_alloc_mib: float       # allocated before it (what the steady state holds)
    seconds: float
    rows: int
    source: str
    oom: Optional[str] = None
    #: the peak of each `train()` segment (`phase_hook` names): where the demand lives
    phase_peaks: Optional[Dict[str, Dict[str, float]]] = None

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ------------------------------------------------------------------------------ state preservation
_SKIP_KEYS = frozenset({"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
                        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                        "_backward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks"})
_PLAIN = (int, float, bool, str, bytes, type(None), np.generic, np.ndarray, np.random.Generator,
          np.random.RandomState)


def _is_plain(v: Any, depth: int = 3) -> bool:
    """Plain DATA (deep-copied on snapshot, so an in-place mutation is undone): numbers, strings,
    numpy values and generators, and containers of them. Anything else is restored by REBINDING."""
    if isinstance(v, _PLAIN):
        return not isinstance(v, np.ndarray) or v.nbytes <= (16 << 20)
    if depth <= 0:
        return False
    import collections
    if isinstance(v, (list, tuple, set, frozenset, collections.deque)):
        return all(_is_plain(x, depth - 1) for x in v)
    if isinstance(v, dict):
        return all(_is_plain(k, 0) and _is_plain(x, depth - 1) for k, x in v.items())
    return False


def _snap_dict(obj: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    d = vars(obj)
    rebind = {k: v for k, v in d.items() if k not in _SKIP_KEYS}
    deep = {k: copy.deepcopy(v) for k, v in rebind.items() if _is_plain(v)}
    return rebind, deep


def _restore_dict(obj: Any, snap: Tuple[Dict[str, Any], Dict[str, Any]]) -> None:
    rebind, deep = snap
    d = vars(obj)
    for k in [k for k in d if k not in _SKIP_KEYS and k not in rebind]:
        del d[k]                                          # an attribute the dry update created
    for k, v in rebind.items():
        d[k] = copy.deepcopy(deep[k]) if k in deep else v


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


def _snap_optimizer(opt: torch.optim.Optimizer) -> Any:
    groups = [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in opt.param_groups]
    states: List[List[Optional[Dict[str, Any]]]] = []
    for g in opt.param_groups:
        row: List[Optional[Dict[str, Any]]] = []
        for p in g["params"]:
            st = opt.state.get(p)
            row.append(None if st is None else
                       {k: (_cpu(v) if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()})
        states.append(row)
    return groups, states


def _restore_optimizer(opt: torch.optim.Optimizer, snap: Any) -> None:
    groups, states = snap
    with torch.no_grad():
        for g, gs, row in zip(opt.param_groups, groups, states):
            for k in [k for k in g if k != "params" and k not in gs]:
                del g[k]
            g.update(copy.deepcopy(gs))
            for p, saved in zip(g["params"], row):
                if saved is None:
                    opt.state.pop(p, None)
                    continue
                st = opt.state[p]
                for k in [k for k in st if k not in saved]:
                    del st[k]
                for k, v in saved.items():
                    if torch.is_tensor(v) and torch.is_tensor(st.get(k)) and st[k].shape == v.shape:
                        st[k].copy_(v)                    # in place: the declared state keeps its storage
                    else:
                        st[k] = v.to(p.device) if torch.is_tensor(v) else copy.deepcopy(v)


@contextlib.contextmanager
def preserved_learner(model: Any) -> Iterator[None]:
    """Run the body, then put the learner back EXACTLY: modules (in place), optimizers (in place),
    RNG streams, instance attributes of the model / its policy's modules / every optimizer holder,
    the logger's pending records, the policy's training mode. Restores on an exception too."""
    from agents.training.learner_lifecycle import learner_objects

    modules, opts = learner_objects(model)
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_rng)
    mod_state = {p: {k: _cpu(v) for k, v in m.state_dict().items()} for p, m in modules.items()}
    opt_state = {p: _snap_optimizer(o) for p, o in opts.items()}
    holders: List[Any] = [model]
    for m in modules.values():
        holders.extend(m.modules())
    dicts = [(h, _snap_dict(h)) for h in holders]
    logger = getattr(model, "_logger", None)
    log_snap = None
    if logger is not None:
        log_snap = {k: copy.copy(getattr(logger, k)) for k in ("name_to_value", "name_to_count",
                                                              "name_to_excluded") if hasattr(logger, k)}
    training = bool(getattr(model.policy, "training", False))
    try:
        yield
    finally:
        for h, snap in dicts:
            _restore_dict(h, snap)
        with torch.no_grad():
            for p, m in modules.items():
                live = m.state_dict()
                for k, v in mod_state[p].items():
                    live[k].copy_(v)
        for p, o in opts.items():
            _restore_optimizer(o, opt_state[p])
        for m in modules.values():
            for prm in m.parameters():
                prm.grad = None
        if logger is not None and log_snap is not None:
            for k, v in log_snap.items():
                getattr(logger, k).clear()
                getattr(logger, k).update(v)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        if hasattr(model.policy, "set_training_mode"):
            model.policy.set_training_mode(training)


# ------------------------------------------------------------------------------ the fixture rollout
def fixture_buffer(model: Any) -> Tuple[Any, str]:
    """A shallow copy of the model's rollout buffer with FRESH arrays of its full declared shape,
    filled by tiling the regions gate's micro-batch fixture (`compile_regions.r1_batch`: the learner
    golden's real labelled rows when they fit this surface). The live buffer is never touched."""
    from agents.model.compile_regions import r1_batch

    rb = model.rollout_buffer
    n_steps, n_envs = int(rb.buffer_size), int(rb.n_envs)
    rows = n_steps * n_envs
    fx = r1_batch(model, int(model.batch_size))
    b = int(fx.actions.shape[0])
    idx = np.arange(rows) % b

    def tile(t: torch.Tensor, like: np.ndarray) -> np.ndarray:
        a = t.detach().cpu().numpy().reshape(b, -1)[idx]
        return a.reshape(like.shape).astype(like.dtype, copy=False)

    out = copy.copy(rb)
    out.observations = {k: tile(fx.obs[k], v) for k, v in rb.observations.items()}
    out.actions = tile(fx.actions, rb.actions)
    out.action_masks = tile(fx.action_masks, rb.action_masks)
    out.log_probs = tile(fx.old_log_prob, rb.log_probs)
    out.values = tile(fx.old_values, rb.values)
    out.advantages = tile(fx.advantages, rb.advantages)
    out.returns = tile(fx.returns, rb.returns)
    out.rewards = np.zeros_like(rb.rewards)
    out.episode_starts = np.zeros_like(rb.episode_starts)
    out.full, out.pos, out.generator_ready = True, n_steps, False
    return out, fx.source


class _PhasePeaks:
    """A `phase_hook` that books the PEAK allocated / reserved bytes of each `train()` segment (the
    work between two marks — `phase_hook` docs) to the closing mark's name: WHERE the update's peak
    lives. Synchronizes at every mark (the dry update only; production installs no hook)."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.peaks: Dict[str, List[float]] = {}

    def mark(self, name: str) -> None:
        torch.cuda.synchronize(self.device)
        a = torch.cuda.max_memory_allocated(self.device) / MiB
        r = torch.cuda.max_memory_reserved(self.device) / MiB
        if name != "start":
            p = self.peaks.setdefault(name, [0.0, 0.0])
            p[0], p[1] = max(p[0], a), max(p[1], r)
        torch.cuda.reset_peak_memory_stats(self.device)

    def table(self) -> Dict[str, Dict[str, float]]:
        return {k: {"peak_alloc_mib": v[0], "peak_reserved_mib": v[1]} for k, v in self.peaks.items()}


def _is_rust_env(model: Any) -> bool:
    e, seen = getattr(model, "env", None), 0
    while e is not None and seen < 8:
        if type(e).__name__ == "RustVecEnv":
            return True
        e, seen = getattr(e, "venv", None), seen + 1
    return False


def _behaviour_probe_setup(model: Any, buf: Any, dump_dir: str) -> None:
    """The K9(b) behaviour check runs in the dry update exactly as in the first real one — under the
    Rust core that is Lane G's pre-loop PROBE, an eager grad-mode forward of one micro-batch, whose
    graph is part of the update's memory story (`consistency.release_autograd_stashes`). For it to
    judge the fixture CORRECTLY the fixture's stored log-probs are THIS policy's (an eager no-grad
    forward of the fixture micro-batch, tiled like the rows), every row is current (versions 0), the
    verdict only WARNS, and a violation's dump goes to a temporary directory that is deleted. All of
    it is restored with the learner."""
    import torch as th

    if str(getattr(model, "behaviour_check", "off") or "off") == "off":
        return
    pol = model.policy
    b = int(model.batch_size)
    rows = int(buf.buffer_size) * int(buf.n_envs)
    flat_obs = {k: th.as_tensor(v.reshape(rows, *v.shape[2:])[:b], device=model.device)
                for k, v in buf.observations.items()}
    acts = th.as_tensor(buf.actions.reshape(rows, -1)[:b, 0]).long().to(model.device)
    masks = th.as_tensor(buf.action_masks.reshape(rows, -1)[:b]).to(model.device)
    was = pol.training
    pol.set_training_mode(True)
    with th.no_grad():
        _v, logp, _e = pol.evaluate_actions(flat_obs, acts, action_masks=masks)
    pol.set_training_mode(was)
    lp = logp.detach().float().cpu().numpy()
    buf.log_probs = lp[np.arange(rows) % b].reshape(buf.log_probs.shape).astype(buf.log_probs.dtype)
    del _v, logp, _e, flat_obs
    model.behaviour_check = "warn"
    model.behaviour_dump_dir = dump_dir
    if _is_rust_env(model):
        model._rust_row_versions = np.zeros((int(buf.buffer_size), int(buf.n_envs)), dtype=np.int64)
        model._rust_version = 0


def dry_update(model: Any, *, epochs: int = 1) -> FitReading:
    """ONE real `train()` (``epochs`` epochs) on `fixture_buffer`, inside `preserved_learner`; returns
    the allocator's reading. An out-of-memory is CAUGHT and reported in ``oom`` (the caller refuses)."""
    dev = next(model.policy.parameters()).device
    cuda = dev.type == "cuda"
    snap_path = os.environ.get(SNAPSHOT_ENV) if cuda else None
    buf, source = fixture_buffer(model)
    live_buf = model.rollout_buffer
    import tempfile
    dump_dir = tempfile.mkdtemp(prefix="update_fit_dump_")
    oom: Optional[str] = None
    phases = _PhasePeaks(dev)
    if cuda:
        torch.cuda.synchronize(dev)
        floor = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    else:
        floor = 0
    t0 = time.perf_counter()
    with preserved_learner(model):
        if getattr(model, "_logger", None) is None:
            # at startup the learner's logger does not exist yet (`_setup_learn` makes it): a null one for
            # the dry update, removed with every other attribute the dry update created
            from agents.training.train_logger import configure
            model._logger = configure(None, [])
        model.rollout_buffer = buf
        model.n_epochs = int(epochs)
        _behaviour_probe_setup(model, buf, dump_dir)
        if snap_path:
            torch.cuda.memory._record_memory_history(max_entries=400_000)
        try:
            from agents.training.instrumented_ppo import phase_hook as PH
            with PH.installed(phases.mark) if cuda else contextlib.nullcontext():
                model.train()
        except torch.cuda.OutOfMemoryError as exc:
            oom = str(exc).splitlines()[0]
        finally:
            if snap_path:
                with contextlib.suppress(Exception):
                    torch.cuda.memory._dump_snapshot(snap_path)
                torch.cuda.memory._record_memory_history(enabled=None)
            model.rollout_buffer = live_buf
    del buf
    import shutil
    shutil.rmtree(dump_dir, ignore_errors=True)
    secs = time.perf_counter() - t0
    if not cuda:
        return FitReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, secs, int(live_buf.buffer_size * live_buf.n_envs),
                          source, oom)
    torch.cuda.synchronize(dev)
    # the hook resets the peak counters at every mark: the update's peak is the max over its segments
    # and whatever ran after the last mark
    demand = max([torch.cuda.max_memory_reserved(dev) / MiB] + [v[1] for v in phases.peaks.values()])
    peak = max([torch.cuda.max_memory_allocated(dev) / MiB] + [v[0] for v in phases.peaks.values()])
    reserved = torch.cuda.memory_reserved(dev) / MiB
    free = torch.cuda.mem_get_info(dev)[0] / MiB
    card = reserved + free
    return FitReading(demand, peak, reserved, free, card, card - demand, floor / MiB, secs,
                      int(live_buf.buffer_size * live_buf.n_envs), source, oom, phases.table())


def verdict(r: FitReading, mode: str = "?", run_dir: Optional[str] = None) -> str:
    """The deterministic verdict on one reading: the log line, or `UpdateWontFit` on an OOM or a
    headroom below `FIT_HEADROOM_MIB`. Writes `<run_dir>/update_fit.json` either way."""
    line = (f"🧮 [UpdateFit] dry first update ({r.rows:,} fixture rows, 1 epoch, device batch {mode}, "
            f"{r.seconds:.1f}s): demand (peak reserved) {r.demand_mib:,.0f} MiB, peak allocated "
            f"{r.peak_alloc_mib:,.0f}, floor {r.floor_alloc_mib:,.0f}; card for this process "
            f"{r.card_mib:,.0f} (reserved {r.reserved_mib:,.0f} + free {r.device_free_mib:,.0f}) -> "
            f"headroom {r.headroom_mib:,.0f} MiB vs the declared {FIT_HEADROOM_MIB:,.0f}"
            + (f"; peak segment '{max(r.phase_peaks, key=lambda k: r.phase_peaks[k]['peak_reserved_mib'])}'"
               if r.phase_peaks else ""))
    if r.oom:
        line = f"🧮 [UpdateFit] dry first update ran OUT OF MEMORY ({r.oom}); " + line.split(": ", 1)[1]
    if run_dir:
        import json
        with contextlib.suppress(OSError):
            with open(os.path.join(str(run_dir), "update_fit.json"), "w") as f:
                json.dump({"schema": SCHEMA, "headroom_required_mib": FIT_HEADROOM_MIB,
                           "device_batch_mode": mode, **r.as_dict()}, f, indent=1)
    if r.oom or r.headroom_mib < FIT_HEADROOM_MIB:
        raise UpdateWontFit(
            f"{line}: the FIRST PPO UPDATE does not fit on this card with the declared headroom "
            f"({SCHEMA}; measured by running it, so a refusal here is the OOM the first update would "
            f"have hit, or a run closer to the edge than D-6 allows). Levers, numerically identical "
            f"first: --device-batch staged (not resident); fewer T2 lanes / smaller opponent buckets "
            f"(--t2-lanes, --t2-buckets); fewer envs (--n-envs, the rollout held at ~98k rows). A "
            f"smaller learner micro-batch at the same effective batch changes the sum order — a "
            f"declared recipe change, not a fit lever.")
    return line


def check_update_fits(model: Any, run_dir: Optional[str] = None) -> Optional[str]:
    """The startup gate: `dry_update`, then `verdict`. None off CUDA (inert)."""
    if next(model.policy.parameters()).device.type != "cuda":
        return None
    return verdict(dry_update(model), str(getattr(model, "device_batch_mode", "?")), run_dir)
