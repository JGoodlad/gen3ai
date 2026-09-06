"""cf_producer_snapshot — WHICH weights the producer is running, and the object that runs them.

Two questions, one module, because the second is meaningless without the first: which checkpoint
on disk is the freshest (:func:`resolve_latest_checkpoint`), and what a loaded one can do
(:class:`Snapshot` — it scores candidate decisions AND builds the players that roll them out).
Those two jobs live behind one object because they must use the SAME weights; a producer that
ranked with one snapshot and rolled out with another would be labelling states chosen by a policy
that is not the one being measured.

Almost everything non-obvious in here is a `torch.compile` SHAPE/DTYPE fact rather than an
arithmetic one — B=1 scoring under a compiled graph, float32 masks on both paths, and the
two-key warm-up that pays the first re-trace up front instead of charging it to whichever record
happened to be first. Each carries its own measurement; none of them is tidiness.

Extracted verbatim from ``cf_producer.py`` (2026-09-06, the file-size ratchet's third cut of the
1,000-2,000 band) — both were already their own banner sections there. ``cf_producer`` re-imports
every name below, so ``from agents.training.cf_producer import load_snapshot`` still resolves, and
nothing it computes moved: ``cf_producer_test.py``'s extraction-parity golden was captured BEFORE
the move and reproduces byte-for-byte after it.
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from agents.training.cf_producer_sampler import normalized_entropy
from utils.bridge.reconstruction import ReconstructionRecord


#: The TWO names a resumable checkpoint is written under. `_TrackingCheckpointCallback` writes the
#: PERIODIC `checkpoint_<step>_steps.zip`; `train.lifecycle._forced_checkpoint` (SIGUSR1 — the
#: launcher TUI's `c` key) writes `checkpoint_forced_<step:010d>_<HHMMSS>.zip` into the same
#: directory. Reading only the first is not a cosmetic gap: an unparseable name scores `(0, 0, …)`
#: in `resolve_latest_checkpoint`'s key, so ANY periodic checkpoint outranks EVERY forced one —
#: a forced save taken AFTER the last periodic one would be silently passed over in favour of an
#: older snapshot, against this module's own "the highest step wins" contract.
_CKPT_STEP_RES = (
    re.compile(r"checkpoint_(\d+)_steps\.zip$"),
    re.compile(r"checkpoint_forced_(\d+)_\d+\.zip$"),
)


def step_from_checkpoint_name(path: str) -> Optional[int]:
    """The step a checkpoint FILENAME declares — periodic or forced — else None."""
    base = os.path.basename(path)
    for rx in _CKPT_STEP_RES:
        m = rx.search(base)
        if m:
            return int(m.group(1))
    return None


def resolve_latest_checkpoint(run_dir: str) -> "Optional[tuple[str, Optional[int]]]":
    """The freshest ``(path, step)`` under ``run_dir``, or None when there is not one yet.

    Considers ``latest.txt`` (which holds a run-RELATIVE path) AND the checkpoint glob, and picks
    the highest STEP among them — not simply whatever ``latest.txt`` names. The two disagree
    exactly when a checkpoint landed between the zip write and the pointer update, and in that
    window the higher step is the one whose weights are on disk.

    BOTH resumable checkpoint names count (see ``_CKPT_STEP_RES``): the periodic
    ``checkpoint_<step>_steps.zip`` and the FORCED ``checkpoint_forced_<step>_<HHMMSS>.zip`` that
    SIGUSR1 writes. Globbing only the first made a forced save reachable solely through
    ``latest.txt`` and then, because its step did not parse, rank BELOW every periodic zip — so an
    operator forcing a checkpoint mid-run moved the producer BACKWARDS to an older snapshot.
    """
    cands: "list[str]" = []
    latest = os.path.join(run_dir, "latest.txt")
    try:
        rel = Path(latest).read_text().strip()
    except OSError:
        rel = ""
    if rel:
        p = rel if os.path.isabs(rel) else os.path.join(run_dir, rel)
        if os.path.exists(p):
            cands.append(p)
    for root in (os.path.join(run_dir, "checkpoints"), run_dir):
        cands += glob.glob(os.path.join(root, "checkpoint_*_steps.zip"))
        cands += glob.glob(os.path.join(root, "checkpoint_forced_*.zip"))
    if not cands:
        return None
    # Highest declared step wins; an unparseable name falls back to mtime so a hand-placed
    # checkpoint is still reachable rather than invisible.
    def _key(p: str):
        s = step_from_checkpoint_name(p)
        return (0 if s is None else 1, s or 0, os.path.getmtime(p))
    best = max(sorted(set(cands)), key=_key)
    return best, step_from_checkpoint_name(best)


class Snapshot:
    """A loaded checkpoint: it scores decisions and it builds the players that roll them out.

    Both jobs live behind one object because they must use the SAME weights — a producer that
    ranked with one snapshot and rolled out with another would be labelling states chosen by a
    policy that is not the one being measured. The `snapshot_loader` seam in :func:`main` exists so
    the end-to-end test can run the REAL bridge rollouts without conjuring a current-architecture
    checkpoint (the same substitution `cf_audit_integration_test` makes, and the same one only).
    """

    def __init__(self, path: str, step: int, model, mappings, *, compiled: bool = False) -> None:
        self.path = path
        self.step = int(step)
        self.model = model
        self.mappings = mappings
        #: True when this snapshot's extractor went through `torch.compile`. It changes ONE thing
        #: here — the batch size :meth:`score` forwards at — and the reason is in that method.
        self.compiled = bool(compiled)
        self._player_seq = 0

    # -- scoring ---------------------------------------------------------------------
    def score(self, obs: np.ndarray, masks: np.ndarray):
        """``(win_probs | None, entropies)`` for a batch of decisions.

        ``win_probs`` is None when the checkpoint trained no win-prob head; the caller must fall
        back to entropy-only ranking and SAY so, rather than reading a missing head as a
        confident 0.0.

        ⚠️ **UNDER `torch.compile` THIS FORWARDS ONE ROW AT A TIME, on purpose.** The rollouts —
        99% of this process's work — forward at **B=1**, and dynamo specializes a compiled graph on
        the batch dimension: a single `B=29` scoring call therefore forces a second trace, and
        coming back to `B=1` a third. Measured 2026-08-23 on the live `ai_v9_29_rev1_0823`
        checkpoint: with a batched score in front of them, the FIRST label's 8 rollouts cost
        **79.4 s** against **3.0 s** for the second — ~76 s of pure recompilation, once per record
        shape. Row-wise scoring keeps exactly ONE shape alive in the whole process. It costs
        ~0.12 s per record against ~0.04 s batched (29 candidates × 4.1 ms), i.e. it trades 80 ms
        for the recompiles, and it is a pure win the moment a second batch size would have appeared.
        EAGER snapshots have no such guard to keep, so they still take the single batched forward.
        """
        import torch as th

        policy = self.model.policy
        obs_a = np.asarray(obs, dtype=np.float32)
        # ⚠️ float32 IS THE POINT, not tidiness. A materialized decision's mask arrives as `int8`
        # while the live rollout path's mask (straight out of `embed_battle`) is float32 — and a
        # compiled graph guards on DTYPE as hard as it does on shape. Measured 2026-08-23: with the
        # int8 mask, the first scored row cost **19.5 s** (one full re-trace) against a 3.8 ms
        # steady state. Same reason as the B=1 chunking below: keep exactly one signature alive.
        mask_a = np.asarray(masks, dtype=np.float32)
        step = 1 if self.compiled else len(obs_a)
        probs_rows: "List[np.ndarray]" = []
        wp_rows: "List[np.ndarray]" = []
        have_wp = True
        with th.no_grad():
            for lo in range(0, len(obs_a), max(1, step)):
                ot = th.as_tensor(obs_a[lo:lo + step])
                mt = th.as_tensor(mask_a[lo:lo + step])
                dist = policy.get_distribution({"observation": ot, "action_mask": mt})
                logits = dist.distribution.logits
                masked = th.where(mt.bool(), logits, th.full_like(logits, -1e8))
                probs_rows.append(th.softmax(masked, 1).cpu().numpy())
                # The win-prob head's output is a STASH written by the forward we just ran, so it
                # is read off the same pass rather than paid for with a second one.
                fe = getattr(policy, "features_extractor", None)
                wp_logits = getattr(fe, "last_win_prob_logits", None) if fe is not None else None
                if wp_logits is None:
                    have_wp = False
                else:
                    wp_rows.append(th.sigmoid(wp_logits[:, 0]).cpu().numpy())
        probs = np.concatenate(probs_rows, axis=0)
        win_probs = np.concatenate(wp_rows, axis=0) if (have_wp and wp_rows) else None
        ent = np.asarray([normalized_entropy(row) for row in probs], dtype=float)
        return win_probs, ent

    # -- players ---------------------------------------------------------------------
    def make_player(self, record: ReconstructionRecord, side: str, *, role: str):
        """A stochastic ``RLPlayer`` on ``side`` of ``record`` — see *THE ECOLOGY DECISION*.

        ``stochastic=True`` is the load-bearing half: a greedy copy of a net is strictly stronger
        than a temp-1.0 sample of it, so a greedy rollout would bias every label LOW against the
        regime the training actor actually plays (measured +0.037 [+0.007, +0.066] over 477
        sentinel states when this was got wrong on the prober's path).
        """
        from poke_env.ps_client import AccountConfiguration
        from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
        from agents.inference.player import RLPlayer

        self._player_seq += 1
        return RLPlayer(
            model=self.model, team=record.packed_team(side), battle_format=record.format_id,
            server_configuration=LocalhostServerConfiguration, mappings=self.mappings,
            account_configuration=AccountConfiguration(
                f"Cf{role}{self._player_seq % 100000}", "pw"),
            max_concurrent_battles=1, stochastic=True, start_listening=False)


def _warm_the_compiled_graph(model) -> float:
    """Force the re-trace the FIRST REAL decision would otherwise pay for. Returns seconds spent.

    `maybe_compile_extractor` warms the graph with ``{"observation": zeros(1, D)}`` — one key. Every
    call this process actually makes arrives through ``policy.get_distribution`` with **two**
    (``observation`` + ``action_mask``), and dynamo guards on the dict's KEY SET exactly as hard as
    it does on shape and dtype. So the compile looks warm and the first live forward silently
    re-traces the whole extractor.

    MEASURED 2026-08-23 on the live `ai_v9_29_rev1_0823` checkpoint: that first forward cost
    **19.5 s** against a 3.8 ms steady state — 17 s of a 27 s six-label pass, charged to whichever
    record happened to be first. Paying it HERE makes it a startup cost that announces itself
    rather than a mystery in the first cycle's heartbeat. Never fatal: a warm-up that raises has
    cost nothing but the warm-up (the real call would simply re-trace as before).
    """
    import torch as th

    t0 = time.perf_counter()
    try:
        space = model.observation_space
        dim = int(space["observation"].shape[0])
        n_act = int(space["action_mask"].shape[0])
        obs = {"observation": th.zeros(1, dim, dtype=th.float32),
               "action_mask": th.ones(1, n_act, dtype=th.float32)}
        with th.no_grad():
            model.policy.get_distribution(obs)
    except Exception as exc:                                            # noqa: BLE001
        print(f"[cf_producer] compiled-graph warm-up skipped ({type(exc).__name__}: "
              f"{str(exc)[:160]}) — the first real decision will re-trace instead", flush=True)
    return time.perf_counter() - t0


def load_snapshot(path: str, step: Optional[int], *, device: str = "cpu",
                  compile_extractor: bool = True) -> Snapshot:
    """Load a checkpoint for scoring + rollouts.

    Uses the prober's `sanitized_load_custom_objects` (drop extractor kwargs the CURRENT
    constructor no longer accepts) rather than a bare `MaskablePPO.load`, for the same reason
    every other rollout path does: a checkpoint written one flag-deletion ago otherwise TypeErrors.
    The step is taken from the model's own ``num_timesteps`` — the authoritative number — with the
    filename as the fallback for a checkpoint that does not carry one.
    """
    from sb3_contrib import MaskablePPO
    from agents.observation.state_encoder import load_mappings
    from main.prober.model import sanitized_load_custom_objects

    custom_objects, _dropped = sanitized_load_custom_objects(path, device)
    model = MaskablePPO.load(path, env=None, device=device, custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    # A `--log-level periodic` checkpoint carries an ObservationDebugger that print()s a DEEP TRACE
    # banner on every forward; it would drown the heartbeat this process communicates through.
    for mod in model.policy.modules():
        if hasattr(mod, "_debugger"):
            mod._debugger = None
    compiled = False
    if compile_extractor:
        from agents.model.compile_opponents import maybe_compile_extractor
        t0 = time.perf_counter()
        compiled = bool(maybe_compile_extractor(
            model, True, label=f"cf_producer:{os.path.basename(path)}", hide_cuda=True))
        if compiled:
            warm = _warm_the_compiled_graph(model)
            print(f"[cf_producer] extractor compiled in {time.perf_counter() - t0 - warm:.0f}s "
                  f"(+{warm:.0f}s warming the live call signature) — paid ONCE PER PROCESS, not "
                  f"per checkpoint: dynamo keys on the CODE object and the weights are graph "
                  f"inputs, so the NEXT refresh reuses this graph (measured 1.1 s for a whole "
                  f"second load). It is ~6.4x on every rollout decision", flush=True)
    resolved = int(getattr(model, "num_timesteps", 0) or 0) or int(step or 0)
    return Snapshot(path, resolved, model, load_mappings(), compiled=compiled)
