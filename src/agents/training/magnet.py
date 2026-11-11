"""THE TRAILING MAGNET — a reverse-KL regulariser toward a frozen copy of the policy (`gen3_magnet_kl_v1`; CE-3d, task
T56, X35-A4; `designs/training/magnet_kl.md`).

    loss += β · mean_rows KL(π_θ(·|s) ‖ π_mag(·|s))        β = `--magnet-kl-coef` (0 = OFF, the default)

``π_mag`` (the MAGNET) is a frozen copy of the policy's own weights, replaced by the current weights at the first update
boundary at or after `--magnet-refresh-steps` env steps since it was last set. It sits BESIDE the entropy bonus, which
is unchanged.

WHERE IT COMES FROM. Magnetic Mirror Descent (Sokota et al., ICLR 2023) regularises each policy step toward a magnet
with the REVERSE KL, KL(π ‖ ρ): a uniform magnet is the entropy bonus PPO already has, and a magnet that TRAILS the
policy moves the fixed point from a regularised (quantal-response) equilibrium toward a Nash equilibrium. R-NaD /
DeepNash (Perolat et al., 2022) is the same idea as a reward transform, −η · log(π / π_reg), with π_reg replaced every
Δm steps. This module is MMD's deep form: PPO plus a reverse-KL loss term. What it does NOT take from R-NaD: the
interpolation between two consecutive regularisation policies, and the opponent-side term of the reward transform.

THE DIRECTION is KL(π ‖ π_mag), the expectation under the LEARNER's distribution — MMD's and R-NaD's. Over the legal
actions

    KL(π ‖ π_mag) = −H(π) − Σ_a π(a) · log π_mag(a)

so β also adds β to the weight of the entropy bonus's own gradient (an entropy-matched control is the screen's, not this
module's).

LEGAL ACTIONS ONLY. Both distributions are the MASKED ones over the row's own legal set (the mask the row was played
under): the learner's from R1's own forward (`masked_categorical`), the magnet's from T2 on the same observation and
mask. The sum runs over legal entries only; an illegal entry contributes exactly 0 to the value and to the gradient.

WHERE THE MAGNET'S LOG-PROBS COME FROM. T2, at the FILL: one extra slot of the trainee's slot group holds the magnet's
weights, and the collector serves every row of the filled buffer through it once (`RustCollector._fill` →
`fill_column`), writing `RolloutBuffer.magnet_logp` — ``[n_steps, n_envs, n_actions]`` float32, 0.0 on illegal entries.
The learner reads the column like any other per-row array; R1 never runs a second forward. (An eager frozen-copy
forward per micro-batch costs a whole extra forward of the update; one T2 pass per rollout is the cheap spelling.)

THE STATE. One host (CPU) copy of the magnet's served weights and the env step it was set at (`Magnet`). It is NOT a
module of the policy (every reader and strict load would carry a second network) and NOT in the checkpoint zip: it is a
SIDECAR, ``<checkpoint>.magnet.pt``, written by `run_io.commit_checkpoint` between the zip and ``latest.txt`` — so every
resumable checkpoint of a magnet run carries its magnet. A `--model` launch reads it (`attach`): a same-run resume of a
run that recorded the magnet ON refuses a checkpoint without one; a FORK (or a run that turns the lever on) seeds the
magnet from the loaded weights at the loaded step, and says so.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch as th

from agents.model import masked_categorical as _mc
from agents.training.lifecycle_decl import startup_builder
from main.exit_codes import FatalConfigError, NonFiniteLearnerError

#: The sidecar beside a checkpoint zip: ``checkpoint_N_steps.zip`` -> ``checkpoint_N_steps.magnet.pt``.
SIDECAR_SUFFIX = ".magnet.pt"
SIDECAR_FORMAT = 1
#: `--magnet-refresh-steps`' default: the backlog card's 2M env steps.
REFRESH_STEPS_DEFAULT = 2_000_000
#: The TensorBoard tags (one producer each): the term's mean KL over the update's micro-batches (`ppo.train`), and the
#: collector's three facts about the column it filled (`fill_column` / `RustCollector`).
TAG_KL = "train/magnet_kl"
TAG_AGE = "magnet/age_steps"
TAG_FORWARD_MS = "magnet/forward_ms"
TAG_REFRESHES = "magnet/refreshes"


class MagnetError(FatalConfigError):
    """A magnet that cannot be what the run declared: a missing or foreign sidecar, a missing buffer column. A restart
    meets the same files and the same declaration, so it exits `FATAL_CONFIG` (never restarted)."""


# ----------------------------------------------------------------------------------------------- the loss term
def reverse_kl_rows(logp: th.Tensor, masks_bool: th.Tensor, magnet_logp: th.Tensor) -> th.Tensor:
    """KL(π ‖ π_mag) per row over the LEGAL actions: ``Σ_legal π · (log π − log π_mag)``.

    ``logp`` is the learner's masked, normalised log-probabilities (`masked_categorical.masked_logits`: an illegal
    entry is finite, about −1e8, and its probability is exactly 0); ``magnet_logp`` the buffer column (finite on legal
    entries). Two ``where``s make an illegal entry contribute exactly 0 to the value AND to the gradient whatever the
    column holds there (0.0 by `fill_column`'s contract; T2 itself serves −inf, and ``0 · inf`` in a backward is NaN):
    the inner one reads the column on legal entries only, the outer one sums legal entries only — as
    `masked_categorical.entropy` does. Static shapes, no host read: it runs inside R1."""
    zero = logp.new_zeros(())
    p = _mc.probs(logp)        # the entropy bonus's own probabilities (one kernel, shared under the trace)
    return th.where(masks_bool, p * (logp - th.where(masks_bool, magnet_logp, zero)), zero).sum(-1)


def masked_uniform_logp(masks: th.Tensor) -> th.Tensor:
    """A magnet column for a batch that has none (the compile gate's fixture rows): the masked-uniform distribution,
    ``−log(n_legal)`` on legal entries and 0.0 on illegal ones — the column's own layout, a deterministic function of
    the mask."""
    m = masks.to(th.bool)
    n = m.sum(-1, keepdim=True).clamp(min=1).to(th.float32)
    return th.where(m, -th.log(n).expand(m.shape), th.zeros((), dtype=th.float32, device=m.device))


# ----------------------------------------------------------------------------------------------- the state
def sidecar_path(zip_path: "str | os.PathLike[str]") -> str:
    """``<checkpoint>.magnet.pt`` for a checkpoint named with or without ``.zip``."""
    p = os.fspath(zip_path)
    return (p[:-4] if p.endswith(".zip") else p) + SIDECAR_SUFFIX


def _served(policy: Any) -> Dict[str, th.Tensor]:
    """The weights T2 serves (the policy's state dict minus the ride-along heads): what a magnet IS."""
    from agents.inference.service.slots import served_state_dict
    return served_state_dict(policy)


class Magnet:
    """The magnet's weights (a host copy of the served state dict) and the env step they were set at."""

    # @startup_builder: built ONCE, by `attach`, before the collector and the freeze; `refresh` copies in place.
    @startup_builder
    def __init__(self, policy: Any, *, step: int, refresh_steps: int, origin: str) -> None:
        if int(refresh_steps) <= 0:
            raise MagnetError(f"--magnet-refresh-steps must be positive, not {refresh_steps}")
        self.state: Dict[str, th.Tensor] = {k: v.detach().to("cpu", copy=True) for k, v in _served(policy).items()}
        self.step = int(step)
        self.refresh_steps = int(refresh_steps)
        self.origin = str(origin)
        self.refreshes = 0

    # ---- the cadence: total env steps, never a call count
    def age(self, num_timesteps: int) -> int:
        return int(num_timesteps) - self.step

    def due(self, num_timesteps: int) -> bool:
        """True at the first update boundary at or after ``refresh_steps`` env steps since the magnet was set."""
        return self.age(num_timesteps) >= self.refresh_steps

    def refresh(self, policy: Any, num_timesteps: int) -> None:
        """The magnet becomes ``policy``'s current weights (an in-place host copy: nothing is allocated)."""
        live = _served(policy)
        self._check_keys(live, "refresh")
        with th.no_grad():
            for k, dst in self.state.items():
                dst.copy_(live[k].detach())
        self.step = int(num_timesteps)
        self.refreshes += 1

    def model_id(self) -> str:
        return f"magnet@{self.step}"

    def _check_keys(self, live: Dict[str, th.Tensor], where: str) -> None:
        bad = sorted(set(live) ^ set(self.state)) or sorted(
            k for k in self.state if tuple(self.state[k].shape) != tuple(live[k].shape)
            or self.state[k].dtype != live[k].dtype)
        if bad:
            raise MagnetError(f"magnet {where}: the magnet's weights are not this policy's ({len(bad)} keys differ, "
                              f"e.g. {bad[:3]}); a magnet is a copy of the SAME network")

    def equals(self, policy: Any) -> bool:
        """Bit-equality of the magnet with ``policy``'s served weights (tests, the banner)."""
        live = _served(policy)
        return set(live) == set(self.state) and all(
            th.equal(self.state[k], live[k].detach().to("cpu")) for k in self.state)

    @contextlib.contextmanager
    def swapped(self, policy: Any) -> Iterator[None]:
        """``policy`` holds the MAGNET's weights inside the block and its own, bit for bit, after it (also on an
        exception). STARTUP ONLY — it allocates its scratch — for the one load that needs it: handing T2's magnet slot
        the magnet's weights through `InferenceService.load`, which takes a policy. Parameters are written in place,
        so the optimizer's state (keyed by parameter object) is untouched."""
        live = policy.state_dict(keep_vars=True)
        self._check_keys({k: live[k] for k in self.state if k in live}, "swap")
        with th.no_grad():
            scratch: List[th.Tensor] = [live[k].detach().clone() for k in self.state]
            for k, src in self.state.items():
                live[k].copy_(src)
        try:
            yield
        finally:
            with th.no_grad():
                for k, s in zip(self.state, scratch):
                    live[k].copy_(s)

    # ---- the sidecar
    def save(self, path: "str | os.PathLike[str]") -> str:
        """Write the sidecar atomically (`utils.atomic_io`): a killed write leaves the previous file or none."""
        from utils.atomic_io import atomic_write

        payload = {"format": SIDECAR_FORMAT, "step": self.step, "refresh_steps": self.refresh_steps,
                   "state": self.state}
        return atomic_write(path, lambda tmp: th.save(payload, tmp))

    @classmethod
    def load(cls, path: "str | os.PathLike[str]", policy: Any, *, refresh_steps: int) -> "Magnet":
        """The magnet a checkpoint's sidecar holds, checked against ``policy``'s served weights key by key. The
        cadence is the RUN's (`refresh_steps`, the resolved flag), never the sidecar's record of the old one."""
        try:
            payload = th.load(os.fspath(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise MagnetError(f"the magnet sidecar {os.fspath(path)} cannot be read: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != SIDECAR_FORMAT or "state" not in payload:
            raise MagnetError(f"the magnet sidecar {os.fspath(path)} is not a format-{SIDECAR_FORMAT} magnet")
        mag = cls(policy, step=int(payload["step"]), refresh_steps=refresh_steps, origin=f"sidecar {os.fspath(path)}")
        state = payload["state"]
        mag._check_keys(state, f"sidecar {os.fspath(path)}")
        with th.no_grad():
            for k, dst in mag.state.items():
                dst.copy_(state[k])
        return mag


def save_sidecar(model: Any, zip_path: "str | os.PathLike[str]") -> Optional[str]:
    """`run_io.commit_checkpoint`'s step: the magnet beside the zip just written. None (and no file) when the run has
    no magnet."""
    mag: Optional[Magnet] = getattr(model, "_magnet", None)
    if mag is None:
        return None
    return mag.save(sidecar_path(zip_path))


def attach(model: Any, *, model_path: Optional[str], same_run: bool, saved_coef: float,
           say: Any = print) -> Optional[Magnet]:
    """Build the run's magnet at STARTUP (after the load and `apply_training_hparams`, before the collector) and hold
    it as ``model._magnet``; None when `--magnet-kl-coef` is 0.

    * a fresh run: the magnet is θ₀ at step 0;
    * ``model_path`` with a sidecar: the checkpoint's own magnet and its age;
    * ``model_path`` without one: the loaded weights at the loaded step — the lever turned on at this launch, or a
      fork of a file the commit protocol did not write (a snapshot). REFUSED on a same-run resume whose checkpoint
      recorded the magnet ON (``saved_coef > 0``): every checkpoint such a run commits carries its sidecar, so a
      missing one is a lost file, and re-seeding would silently change the regulariser mid-run.
    """
    coef = float(getattr(model, "magnet_kl_coef", 0.0) or 0.0)
    if coef < 0.0:
        raise MagnetError(f"--magnet-kl-coef must be >= 0, not {coef}")
    if coef == 0.0:
        model._magnet = None
        return None
    refresh = int(getattr(model, "magnet_refresh_steps", None) or REFRESH_STEPS_DEFAULT)
    step = int(getattr(model, "num_timesteps", 0) or 0)
    side = sidecar_path(model_path) if model_path else None
    if side is not None and os.path.exists(side):
        mag = Magnet.load(side, model.policy, refresh_steps=refresh)
        if mag.step > step:
            raise MagnetError(f"the magnet sidecar {side} was set at step {mag.step:,}, after the checkpoint's own "
                              f"step {step:,}: it is not this checkpoint's")
    elif model_path and same_run and float(saved_coef or 0.0) > 0.0:
        raise MagnetError(
            f"--magnet-kl-coef {coef:g}: {model_path} recorded the magnet ON (coefficient {float(saved_coef):g}) but "
            f"has no sidecar {side}. Every checkpoint a magnet run commits carries one (`run_io.commit_checkpoint`), "
            f"so the file was lost; resuming would silently restart the magnet at the current weights. Restore the "
            f"sidecar, resume from a checkpoint that has one, or FORK the checkpoint (a fork seeds a new magnet).")
    else:
        mag = Magnet(model.policy, step=step, refresh_steps=refresh,
                     origin=("the initial weights" if not model_path else f"the loaded weights of {model_path}"))
    model._magnet = mag
    say(f"🧲 [MAGNET] --magnet-kl-coef {coef:g}: loss += {coef:g} · KL(π ‖ π_mag) over the legal actions, beside the "
        f"entropy bonus (ent_coef {float(getattr(model, 'ent_coef', 0.0)):g}; the KL's own entropy part adds {coef:g} "
        f"to that weight). The magnet: {mag.origin}, set at step {mag.step:,} (age {mag.age(step):,}); refreshed at "
        f"the first update boundary at or after {mag.refresh_steps:,} steps of age (--magnet-refresh-steps).")
    return mag


# ----------------------------------------------------------------------------------------------- the collector's side
def fill_column(svc: Any, slot: int, buf: Any) -> np.ndarray:
    """The magnet's masked log-probabilities of every row of the filled buffer, ``[n_steps, n_envs, n_actions]``
    float32 with 0.0 on illegal entries (T2 serves −inf there) — served by T2's magnet ``slot`` on the rows' own
    observations and masks, in flushes of at most the declared arena. Called at the FILL, on the main thread."""
    from agents.inference.service.spec import Priority
    from agents.training.rust_rollout.store import KEY_OBSERVATION

    n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
    n = n_steps * n_envs
    obs = buf.observations[KEY_OBSERVATION].reshape(n, -1)
    mask = buf.action_masks.reshape(n, -1)
    legal = mask.astype(bool)
    step = max(1, int(svc.spec.max_rows_per_flush))
    out = np.zeros((n, legal.shape[1]), dtype=np.float32)
    for s in range(0, n, step):
        e = min(n, s + step)
        t = svc.submit(int(slot), obs[s:e], mask[s:e], Priority.ROLLOUT)
        svc.flush()
        lp, _v, _g = t.host()
        np.copyto(out[s:e], lp, where=legal[s:e])
    if not np.isfinite(out).all():
        raise NonFiniteLearnerError("the magnet's log-probabilities are not finite on a legal action (T2's magnet "
                                    "slot): the KL term would poison the update")
    return out.reshape(n_steps, n_envs, -1)


def fill(svc: Any, slot: int, model: Any) -> None:
    """`RustCollector._fill`'s step: write the buffer's magnet column and record the collector's facts about it."""
    import time

    mag: Optional[Magnet] = getattr(model, "_magnet", None)
    if mag is None:
        raise MagnetError("the collector declares a magnet slot but the learner holds no magnet (`attach` did not run)")
    t0 = time.perf_counter()
    model.rollout_buffer.magnet_logp = fill_column(svc, slot, model.rollout_buffer)
    logger = getattr(model, "_logger", None)
    if logger is not None:
        logger.record(TAG_FORWARD_MS, 1000.0 * (time.perf_counter() - t0))
        logger.record(TAG_AGE, float(mag.age(int(model.num_timesteps))))
        logger.record(TAG_REFRESHES, float(mag.refreshes))


def refresh_if_due(svc: Any, slot: int, model: Any) -> bool:
    """`RustCollector.after_update`'s step: at the first update boundary at or after the cadence, the magnet becomes
    the weights the learner just stepped to (the same weights T2's trainee slot was just loaded with), and T2's magnet
    slot is loaded with them. True iff it refreshed."""
    mag: Optional[Magnet] = getattr(model, "_magnet", None)
    if mag is None:
        raise MagnetError("the collector declares a magnet slot but the learner holds no magnet (`attach` did not run)")
    if not mag.due(int(model.num_timesteps)):
        return False
    mag.refresh(model.policy, int(model.num_timesteps))
    svc.load(int(slot), model.policy, mag.model_id())
    return True


def load_slot(svc: Any, slot: int, model: Any) -> bool:
    """STARTUP: hand T2's magnet slot the magnet's weights. The slot's template is the trainee's policy, so a magnet
    that IS the current weights (a fresh run, a fork's seed) needs no load; a resumed one is loaded through `swapped`.
    True iff a load ran."""
    mag: Optional[Magnet] = getattr(model, "_magnet", None)
    if mag is None:
        raise MagnetError("the collector declares a magnet slot but the learner holds no magnet (`attach` did not run)")
    if mag.equals(model.policy):
        return False
    with mag.swapped(model.policy):
        svc.load(int(slot), model.policy, mag.model_id())
    return True


def validate_decl(*, magnet: bool, fork: Any, overlap: bool) -> None:
    """Refuse a collector declaration the magnet's column cannot serve (`RustEnvDecl.__post_init__`)."""
    if not magnet:
        return
    if fork is not None:
        raise ValueError("RustEnvDecl: --magnet-kl-coef > 0 cannot combine with the fork arm (--fork-fraction > 0): "
                         "the magnet's column is defined for the buffer's played rows only")
    if overlap:
        raise ValueError("RustEnvDecl: --magnet-kl-coef > 0 cannot combine with --overlap-collect: the magnet's pass "
                         "runs on the main thread at the fill, while the overlap's worker owns the T2 flush")
