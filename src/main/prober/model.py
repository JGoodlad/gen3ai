"""Thin torch wrapper for the prober engine.

Isolates the only non-pure pieces — loading a MaskablePPO checkpoint and running
forward / backward passes — behind a small object the pure ``engine`` functions
call. Tests substitute a fake exposing the same three members
(``action_dist``, ``logit_grad``, ``offsets``), so the engine is exercised with
zero torch.

We deliberately use raw ``MaskablePPO.load`` (no env, no
``ModelVersion.check_compatible``) to match the legacy ``probe_replay.py`` CLI: a
checkpoint whose architecture no longer matches the current obs layout surfaces
as a torch shape error on the first ``action_dist`` call, which the TUI catches
and renders as an analysis error rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsOffsets:
    """Semantic obs-vector offsets the analysis depends on.

    Resolved from the live encoder layout so an obs-layout change moves these
    automatically; ``engine_test.py`` pins the resolved values as a regression
    guard so a silent layout shift is caught loudly.
    """

    mm_off: int            # active-move type multipliers (4 dims) vs current opp
    om_off: int            # our_matchups block (144 dims)
    active_block_dim: int  # our active-pokemon block span [0:active_block_dim)
    turn_history_offset: int
    turn_history_dim: int  # n_history_turns * turn_delta_dim

    @classmethod
    def from_encoder(cls, enc) -> "ObsOffsets":
        import agents.observation.constants as C

        lay = enc.get_layout()
        rl = lay["reactive_layout"]
        return cls(
            mm_off=C.OFFSET_REACTIVE + rl["move_multiplier"]["offset"],
            om_off=C.OFFSET_REACTIVE + rl["our_matchups"]["offset"],
            active_block_dim=99,  # the launcher CLI's "our active pokemon block(99)"
            turn_history_offset=lay["turn_history_offset"],
            turn_history_dim=lay["n_history_turns"] * lay["turn_delta_dim"],
        )

    @classmethod
    def resolve(cls) -> "ObsOffsets":
        from agents.observation.state_encoder import (
            Gen3ObservationEncoder,
            load_mappings,
        )

        return cls.from_encoder(Gen3ObservationEncoder(load_mappings()))


class ProbeModel:
    """Loaded policy + resolved offsets; the engine's torch boundary."""

    def __init__(self, policy, offsets: ObsOffsets,
                 global_encoder=None, global_off: int = 0, global_dim: int = 0) -> None:
        self._policy = policy
        self.offsets = offsets
        # For decoding the field state (weather/spikes/screens) out of the obs.
        self._global_encoder = global_encoder
        self._global_off = global_off
        self._global_dim = global_dim

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "ProbeModel":
        from sb3_contrib import MaskablePPO
        from agents.observation.state_encoder import (
            Gen3ObservationEncoder,
            load_mappings,
        )

        model = MaskablePPO.load(ckpt_path, device=device)
        policy = model.policy
        policy.set_training_mode(False)
        # A checkpoint trained with --log-level periodic carries an ObservationDebugger
        # that print()s a "DEEP TRACE" banner on forward passes. That noise pollutes the
        # CLI and would corrupt the Textual screen, so silence it on the probed model
        # (printing only; never affects the computed distribution / gradients).
        for m in policy.modules():
            if hasattr(m, "_debugger"):
                m._debugger = None
        enc = Gen3ObservationEncoder(load_mappings())
        gp = enc.get_layout()["parts"]["global"]
        return cls(policy=policy, offsets=ObsOffsets.from_encoder(enc),
                   global_encoder=enc.global_env_encoder,
                   global_off=gp["start"], global_dim=gp["dim"])

    def describe_global(self, obs: np.ndarray) -> "dict | None":
        """Decode the field state (weather, spikes, screens, turn) from the obs."""
        if self._global_encoder is None:
            return None
        g = np.asarray(obs)[self._global_off:self._global_off + self._global_dim]
        return self._global_encoder.describe_vector(g)

    def action_dist(self, obs: np.ndarray, mask: np.ndarray):
        """Return (masked-softmax probs, raw logits) for a single obs/mask."""
        import torch

        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        d = self._policy.get_distribution({"observation": ot, "action_mask": mt})
        lg = d.distribution.logits.clone()
        masked = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
        probs = torch.softmax(masked, 1)[0].detach().numpy()
        return probs, lg[0].detach().numpy()

    def value(self, obs: np.ndarray, mask: np.ndarray) -> float:
        """The critic's V(s) for a single obs/mask (dual-head policy value path)."""
        import torch

        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            v = self._policy.predict_values({"observation": ot, "action_mask": mt})
        return float(v.reshape(-1)[0])

    def logit_grad(self, obs: np.ndarray, mask: np.ndarray, action_idx: int) -> np.ndarray:
        """Return |d logit(action_idx) / d obs| as a per-dim array."""
        import torch

        ot = torch.as_tensor(obs).unsqueeze(0).requires_grad_(True)
        mt = torch.as_tensor(mask).unsqueeze(0)
        d = self._policy.get_distribution({"observation": ot, "action_mask": mt})
        d.distribution.logits[0, action_idx].backward()
        return ot.grad[0].abs().numpy()
