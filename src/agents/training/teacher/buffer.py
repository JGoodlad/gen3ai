"""``Correction`` + ``CorrectionBuffer`` — the side DAGGER dataset of verified-better search results.

A ``Correction`` is one searched state's ``(obs, mask, A*, confirmed advantage)``. The buffer is a
bounded recency RING (deque, oldest dropped on overflow → stale corrections age out, §9) that the
AWR aux loss samples — with its OWN policy forward — on EACH rollout minibatch inside ``train()`` (the
terms accumulate into that minibatch's loss). It is **standalone**, NOT the PPO rollout buffer: the
searched states are off-policy (from older eval traces) and must never enter GAE / the clip objective
(§2). It lives on the model (``model._correction_buffer``) like ``_win_terminal_scratch``, filled by
the :class:`SearchTeacherCallback` from worker shards.
"""

from __future__ import annotations

import collections
import threading
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Correction:
    """One verified-better search correction (the AWR training target for a single searched state)."""

    obs: np.ndarray          # [obs_dim] float32 — the searched state's one-sided policy obs
    action_mask: np.ndarray  # [n_actions] int8/bool — the legal mask at that state
    better_action: int       # A* — the divergence-ply action index (PPO indexing; action 6+k == move slot k)
    advantage: float         # confirmed_dwin = Wilson-LB(A* win-rate) − played realized rate (> 0, gate-passed)
    confirmed_value: float   # the rollout-confirmed return of s (only used by the off-by-default value term)
    step_produced: int       # the eval step that produced the source trace (recency aging + provenance)
    opponent: str            # the resolved opponent label (per-opponent metrics + provenance)

    def as_record(self) -> dict:
        """JSON-friendly scalars (obs/mask travel as a sibling .npz in a worker shard, not here)."""
        d = asdict(self)
        d.pop("obs"); d.pop("action_mask")
        return d


class CorrectionBuffer:
    """A bounded recency ring of :class:`Correction`s, sampled by the AWR aux loss each ``train()``."""

    def __init__(self, capacity: int, rng: Optional[np.random.Generator] = None):
        if capacity <= 0:
            raise ValueError(f"CorrectionBuffer capacity must be > 0, got {capacity}")
        self._ring: "collections.deque[Correction]" = collections.deque(maxlen=int(capacity))
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lock = threading.Lock()   # the callback fills from _on_step; train() samples — both on the
        #                                 main thread, but the lock keeps it safe if a future thread fills.

    def add(self, c: Correction) -> None:
        with self._lock:
            self._ring.append(c)

    def extend(self, cs: List[Correction]) -> None:
        with self._lock:
            self._ring.extend(cs)

    def __len__(self) -> int:
        return len(self._ring)

    def sample(self, batch_size: int) -> Optional[List[Correction]]:
        """Uniform sample of ``batch_size`` corrections (without replacement when enough exist, else the
        whole buffer). ``None`` when empty — the loss None-guards on this (mirrors ``_win_prob_loss``)."""
        with self._lock:
            n = len(self._ring)
            if n == 0:
                return None
            k = min(int(batch_size), n)
            idx = self._rng.choice(n, size=k, replace=False)
            items = list(self._ring)
        return [items[i] for i in idx]

    @staticmethod
    def to_tensors(corrections: List[Correction], device) -> dict:
        """Stack a sampled minibatch into the Dict-obs + targets the AWR loss consumes. ``obs_dict`` is
        the ``{observation, action_mask}`` Dict the dual-head policy reads (``obs["observation"]``)."""
        import torch as th

        obs = th.as_tensor(np.stack([c.obs for c in corrections]).astype(np.float32), device=device)
        mask = th.as_tensor(np.stack([c.action_mask for c in corrections]), device=device)
        return {
            "obs_dict": {"observation": obs, "action_mask": mask},
            "action_mask": mask,
            "better_action": th.as_tensor(
                np.asarray([c.better_action for c in corrections], dtype=np.int64), device=device),
            "advantage": th.as_tensor(
                np.asarray([c.advantage for c in corrections], dtype=np.float32), device=device),
            "confirmed_value": th.as_tensor(
                np.asarray([c.confirmed_value for c in corrections], dtype=np.float32), device=device),
        }
