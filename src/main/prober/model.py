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
    om_off: int            # our_matchups block (144 dims): our moves' eff vs their mons
    tm_off: int            # their_matchups block (144 dims): their moves' eff vs OUR mons (incoming threat)
    active_block_dim: int  # our active-pokemon block span [0:active_block_dim)
    turn_history_offset: int
    turn_history_dim: int  # n_history_turns * turn_delta_dim
    turn_delta_dim: int = 0  # one TurnDelta slot's width — slices turn_history into per-turn saliency
    # incoming-damage / OHKO belief block (incoming_damage_v1): per-our-slot P(KO)/expected-chip/
    # P(outspeed) + opp recovery scalars — the calibrated DAMAGE belief (vs ThreatView's raw
    # type-effectiveness). Defaults keep the synthetic-test ObsOffsets construction valid; a real
    # `resolve()`/`from_encoder()` always fills them, and the decoder no-ops when dim==0.
    incoming_off: int = 0
    incoming_dim: int = 0
    incoming_per_mon: int = 5
    incoming_recovery: int = 3
    pokemon_full_dim: int = 107   # per-our-mon obs block width (active flag = its last dim)
    total_dim: int = 0            # the current encoder's full obs dim — a guard so a wrong-length
    #                               (e.g. archived old-arch) trace is REFUSED, not silently mis-sliced

    @classmethod
    def from_encoder(cls, enc) -> "ObsOffsets":
        import agents.observation.constants as C

        lay = enc.get_layout()
        rl = lay["reactive_layout"]
        inc = rl.get("incoming_damage", {})
        return cls(
            mm_off=C.OFFSET_REACTIVE + rl["move_multiplier"]["offset"],
            om_off=C.OFFSET_REACTIVE + rl["our_matchups"]["offset"],
            tm_off=C.OFFSET_REACTIVE + rl["their_matchups"]["offset"],
            active_block_dim=99,  # the launcher CLI's "our active pokemon block(99)"
            turn_history_offset=lay["turn_history_offset"],
            turn_history_dim=lay["n_history_turns"] * lay["turn_delta_dim"],
            turn_delta_dim=lay["turn_delta_dim"],
            incoming_off=C.OFFSET_REACTIVE + inc.get("offset", 0) if inc else 0,
            incoming_dim=inc.get("dim", 0),
            incoming_per_mon=inc.get("per_mon", 5),
            incoming_recovery=inc.get("recovery", 3),
            pokemon_full_dim=C.POKEMON_FULL_DIM,
            total_dim=lay.get("total_dim", 0),
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

    def action_probs_batch(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Masked-softmax action probs for a BATCH of (obs, mask). Shape (N, 11).

        One transformer forward over N decisions instead of N single-row passes — the
        right call for offline sweeps (e.g. the human-agreement probe) where thousands of
        saved/reconstructed states are scored against a frozen policy."""
        import torch

        ot = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        mt = torch.as_tensor(np.asarray(masks))
        with torch.no_grad():
            d = self._policy.get_distribution({"observation": ot, "action_mask": mt})
            lg = d.distribution.logits
            masked = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
            probs = torch.softmax(masked, 1).cpu().numpy()
        return probs

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

    def features(self, obs: np.ndarray, mask: np.ndarray) -> "dict[str, np.ndarray]":
        """The model's INTERNAL post-projection features — what the policy/value MLPs read.

        Returns ``{'pi': [PROJECTION_DIM], 'vf': [PROJECTION_DIM]}``. This is the probe boundary:
        a linear probe on these activations tells us whether a derived quantity (is-faster,
        damage, faint-soon) is ALREADY in the representation. The feature extractor reads only
        ``obs['observation']`` (never ``action_mask``), so the mask is inert here — but we pass
        the real one for parity with the distribution/value paths."""
        import torch

        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            pi, vf = self._policy.extract_features({"observation": ot, "action_mask": mt})
        return {"pi": pi[0].detach().numpy(), "vf": vf[0].detach().numpy()}

    def value_grad(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return |d V(s) / d obs| as a per-dim array — the CRITIC's input sensitivity.

        The policy-logit saliency (``logit_grad``) shows what the actor reads; this shows what the
        VALUE head reads, which is the relevant lens for critic tail-blindness (does the critic's
        V(s) actually move with the incoming-damage / OHKO belief block?)."""
        import torch

        ot = torch.as_tensor(obs).unsqueeze(0).requires_grad_(True)
        mt = torch.as_tensor(mask).unsqueeze(0)
        v = self._policy.predict_values({"observation": ot, "action_mask": mt})
        v.reshape(-1)[0].backward()
        return ot.grad[0].abs().numpy()
