"""THE UPDATE FIT CHECK (`gen3_update_fit_v1`, 2026-10-01) — a MEASURED, PREDICTIVE startup gate on
the learner's CUDA memory: does the first real PPO update fit on this card, with the declared headroom?

WHY. `gen3_cuda_ledger_v1`'s device-batch check was a LOWER bound (the regions gate's one R1 step,
4.3 GiB at B = 2048) and had a measured false pass: N = 256 with the X26 ride-along heads passed it
and then ran OUT OF MEMORY in its first update (10.19 GiB allocated on an 11.63 GiB card). The real
update needs the R1 step PLUS the eager tail's work (the ride-along heads' update, the probes), the
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

THE DEMAND (`gen3_update_fit_v2`, 2026-10-15) is what TRAINING will need, not what startup left behind. v1 read the
allocator's peak reserved during the dry update from whatever cache startup had left, so a startup TRANSIENT held
in the cache (the region gate's arms: 9,790 MiB reserved on `rb_xs_s2701`, whose update peaked at 6,579 allocated)
was read as the update's demand and refused a run that fit with ~3 GiB to spare. v2:

  1. starts from the steady state's FLOOR: the allocator's cached blocks are handed back first (`empty_cache`;
     the gate already releases its own, `compile_trainer.arm_compile_sentinel`) — so the reading is the update's;
  2. runs the dry update (below), then the MID-RUN CANARY's check once (`CompileCanary.measure`: the gate's
     arms on the same rows, every 100 updates in the run, with training state resident), in that order — the
     run's own order at update 10, the canary's blocks requested from the cache the update left;
  3. DEMAND = the PEAK RESERVED over both. Reserved, not allocated: the gap is the allocator's fragmentation
     over the sequence, MEASURED rather than assumed (`fragmentation_mib` in the record).

THE VERDICT (deterministic: a pass or a typed FATAL, no warn tier). With the card this process can have = its
reserved bytes now + the device's free bytes now (other processes and the CUDA context already excluded):

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
SCHEMA = "gen3_update_fit_v2"
#: Device headroom the first update must leave: K6's ceiling margin (512) + D-6's rule (512).
FIT_HEADROOM_MIB = 1024.0
SNAPSHOT_ENV = "GEN3AI_UPDATE_FIT_SNAPSHOT"


class UpdateWontFit(RuntimeError):
    """The first real PPO update does not fit on this card with the declared headroom: measured by a
    dry update at startup, refused there (FATAL_CONFIG), never discovered as an OOM at update 1."""


@dataclass
class FitReading:
    demand_mib: float            # peak reserved over the dry update + the dry canary (v2)
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
    #: every key the dry update put on the logger's bus — the startup metric-key check's input
    #: (`agents.training.metric_keys`, gen3_metric_key_collision_v1)
    metric_keys: Tuple[str, ...] = ()
    #: v2: the update's own peak reserved, the dry canary's (None: no compiled learner, so no canary), and the
    #: canary check's disagreement text when its comparison did not pass (reported, never a verdict here)
    update_demand_mib: float = 0.0
    canary_demand_mib: Optional[float] = None
    canary_peak_alloc_mib: Optional[float] = None
    canary_note: Optional[str] = None

    @property
    def fragmentation_mib(self) -> float:
        """Peak reserved minus peak allocated over the measured sequence: the allocator's measured overhead."""
        return self.demand_mib - self.peak_alloc_mib

    def as_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "fragmentation_mib": self.fragmentation_mib}


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
                                                              "name_to_excluded", "name_to_site")
                    if hasattr(logger, k)}
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
    # The rollout is served at fp32 whatever the learner TRAINS with (T2 / `trunk_precision`, design (b)), so the
    # stored μ is the fp32 function: under a bf16 learner (the production recipe since 2026-10-14) the live policy's
    # own forward would differ from the K9(b) probe's fp32 read by bf16's rounding (0.0279 on the A/B's first EB
    # leg) and the dry update would print a spurious BEHAVIOUR-POLICY MISMATCH.
    from agents.model.trunk_precision import fp32_scope
    with th.no_grad(), fp32_scope(pol):
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


def armed_canary(model: Any) -> Any:
    """The in-run parity canary armed for THIS model (`compile_control`'s, set by `arm_compile_sentinel`), or None
    (a learner that is not compiled has no canary)."""
    from agents.model import compile_control as cc
    ctl = getattr(cc, "_CONTROL", None)
    canary = getattr(ctl, "canary", None)
    return canary if canary is not None and getattr(canary, "model", None) is model else None


def release_cached_blocks(device: Any) -> None:
    """Hand the allocator's CACHED blocks back (gen3_update_fit_v2 step 1): the reading starts from what the steady
    state HOLDS, not from a startup transient's cache. Values never change: the cache only decides where a tensor
    lives."""
    import gc
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def dry_update(model: Any, *, epochs: int = 1, canary: Any = None) -> FitReading:
    """ONE real `train()` (``epochs`` epochs) on `fixture_buffer`, inside `preserved_learner`, then (``canary``, a
    `CompileCanary`) the in-run canary's check once on the restored learner; returns the allocator's reading over
    both. An out-of-memory is CAUGHT and reported in ``oom`` (the caller refuses)."""
    dev = next(model.policy.parameters()).device
    cuda = dev.type == "cuda"
    snap_path = os.environ.get(SNAPSHOT_ENV) if cuda else None
    buf, source = fixture_buffer(model)
    live_buf = model.rollout_buffer
    import tempfile
    dump_dir = tempfile.mkdtemp(prefix="update_fit_dump_")
    oom: Optional[str] = None
    recorded: Tuple[str, ...] = ()
    phases = _PhasePeaks(dev)
    if cuda:
        release_cached_blocks(dev)
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
        # X14 (--overlap-collect): an IDENTITY decoupled rewrite (w = 1) for a buffer no fill made; only the
        # overlap path reads it, and `preserved_learner` removes it
        model._overlap_prox = np.asarray(buf.log_probs, dtype=np.float64).copy()
        _behaviour_probe_setup(model, buf, dump_dir)
        if snap_path:
            torch.cuda.memory._record_memory_history(max_entries=400_000)
        try:
            from agents.training.instrumented_ppo import phase_hook as PH
            with PH.installed(phases.mark) if cuda else contextlib.nullcontext():
                model.train()
            recorded = tuple(getattr(model._logger, "name_to_value", {}) or {})
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
    note: Optional[str] = None
    if cuda:
        # the hook resets the peak counters at every mark: the update's tail (after its last mark) is closed here,
        # so the canary's segment below is its own
        phases.mark("update_tail")
    update_names = list(phases.peaks)
    if canary is not None and oom is None:
        # THE MID-RUN CANARY (gen3_update_fit_v2 step 2): its arms run with training state resident, after an
        # update, every 100 updates — so they must fit on the card the update leaves. Measured, not estimated.
        try:
            note = canary.measure()
        except torch.cuda.OutOfMemoryError as exc:
            oom = "the canary's check: " + str(exc).splitlines()[0]
        if cuda:
            phases.mark("canary")
    secs = time.perf_counter() - t0
    if not cuda:
        return FitReading(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, secs, int(live_buf.buffer_size * live_buf.n_envs),
                          source, oom, metric_keys=recorded, canary_note=note)
    torch.cuda.synchronize(dev)
    d = fit_demand(phases.peaks, update_names, (torch.cuda.max_memory_allocated(dev) / MiB,
                                                torch.cuda.max_memory_reserved(dev) / MiB))
    reserved = torch.cuda.memory_reserved(dev) / MiB
    free = torch.cuda.mem_get_info(dev)[0] / MiB
    card = reserved + free
    return FitReading(d.demand_mib, d.peak_alloc_mib, reserved, free, card, card - d.demand_mib, floor / MiB, secs,
                      int(live_buf.buffer_size * live_buf.n_envs), source, oom, phases.table(), recorded,
                      update_demand_mib=d.update_demand_mib, canary_demand_mib=d.canary_demand_mib,
                      canary_peak_alloc_mib=d.canary_peak_alloc_mib, canary_note=note)


@dataclass
class Demand:
    demand_mib: float
    peak_alloc_mib: float
    update_demand_mib: float
    canary_demand_mib: Optional[float]
    canary_peak_alloc_mib: Optional[float]


def fit_demand(peaks: Dict[str, List[float]], update_phases: List[str], tail: Tuple[float, float]) -> Demand:
    """THE DEMAND DEFINITION (gen3_update_fit_v2), pure: ``peaks`` = each measured segment's [peak allocated, peak
    reserved] (MiB; the update's `phase_hook` segments, its ``update_tail`` and the dry ``canary``), ``update_phases``
    the update's own segment names, ``tail`` = (allocated, reserved) peaks after the last mark. The demand is the PEAK
    RESERVED over EVERY segment — the update's AND the mid-run canary's, measured in the run's order from an emptied
    cache — so a canary whose check outgrows the update sets the demand, and its gap to the peak allocated is the
    measured fragmentation."""
    demand = max([tail[1]] + [v[1] for v in peaks.values()])
    peak = max([tail[0]] + [v[0] for v in peaks.values()])
    upd = max([0.0] + [peaks[k][1] for k in update_phases if k in peaks])
    can = peaks.get("canary")
    return Demand(demand, peak, upd, None if can is None else can[1], None if can is None else can[0])


def verdict(r: FitReading, mode: str = "?", run_dir: Optional[str] = None) -> str:
    """The deterministic verdict on one reading: the log line, or `UpdateWontFit` on an OOM or a
    headroom below `FIT_HEADROOM_MIB`. Writes `<run_dir>/update_fit.json` either way."""
    can = ("no compiled learner, so no canary" if r.canary_demand_mib is None else
           f"the canary's check {r.canary_demand_mib:,.0f} reserved / {r.canary_peak_alloc_mib or 0.0:,.0f} allocated")
    line = (f"🧮 [UpdateFit] dry first update ({r.rows:,} fixture rows, 1 epoch, device batch {mode}, "
            f"{r.seconds:.1f}s) + the mid-run canary's check, from an emptied cache: demand (peak reserved) "
            f"{r.demand_mib:,.0f} MiB (the update {r.update_demand_mib:,.0f}; {can}), peak allocated "
            f"{r.peak_alloc_mib:,.0f}, fragmentation {r.fragmentation_mib:,.0f}, floor {r.floor_alloc_mib:,.0f}; card for this process "
            f"{r.card_mib:,.0f} (reserved {r.reserved_mib:,.0f} + free {r.device_free_mib:,.0f}) -> "
            f"headroom {r.headroom_mib:,.0f} MiB vs the declared {FIT_HEADROOM_MIB:,.0f}"
            + (f"; peak segment '{max(r.phase_peaks, key=lambda k: r.phase_peaks[k]['peak_reserved_mib'])}'"
               if r.phase_peaks else ""))
    if r.oom:
        line = f"🧮 [UpdateFit] dry first update ran OUT OF MEMORY ({r.oom}); " + line.split(": ", 1)[1]
    if r.canary_note:
        line += (f". NOTE: the dry canary's comparison did not pass on the weights the startup gate just passed "
                 f"({r.canary_note[:300]}) — memory only here; the in-run canary judges")
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
    """The startup gate: `dry_update`, the METRIC-KEY check over every key it recorded (+ the network's
    derived families; `agents.training.metric_keys` — `MetricKeyCollisionError`, FATAL_CONFIG), then
    `verdict`. Off CUDA: the key check over the derived families only, then None (no dry update)."""
    from agents.training.metric_keys import check_startup_metric_keys
    if next(model.policy.parameters()).device.type != "cuda":
        check_startup_metric_keys(model)
        return None
    reading = dry_update(model, canary=armed_canary(model))
    check_startup_metric_keys(model, reading.metric_keys)
    return verdict(reading, str(getattr(model, "device_batch_mode", "?")), run_dir)
