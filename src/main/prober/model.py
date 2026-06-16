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
    pokemon_full_dim: int = 110   # per-our-mon obs block width (active flag = its last dim)
    # gen3_wish_wired_v1: the two pending-Wish "floating heal" reactive scalars (our / opp side).
    # 0 when not present in the layout (synthetic-test / old-arch traces) → the decoder no-ops.
    wish_our_off: int = 0
    wish_opp_off: int = 0
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
            wish_our_off=(C.OFFSET_REACTIVE + rl["wish_floating_our"]["offset"]) if "wish_floating_our" in rl else 0,
            wish_opp_off=(C.OFFSET_REACTIVE + rl["wish_floating_opp"]["offset"]) if "wish_floating_opp" in rl else 0,
            total_dim=lay.get("total_dim", 0),
        )

    @classmethod
    def resolve(cls) -> "ObsOffsets":
        from agents.observation.state_encoder import (
            Gen3ObservationEncoder,
            load_mappings,
        )

        return cls.from_encoder(Gen3ObservationEncoder(load_mappings()))


_BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "acc", "eva")   # TurnDelta boost-delta order


def _boost_delta_str(arr) -> str:
    """A boost-stage change array (7, raw stages) → 'atk+1 spe+2' (only the non-zero stats)."""
    if arr is None:
        return ""
    out = []
    for name, v in zip(_BOOST_STATS, arr):
        iv = int(round(float(v)))
        if iv:
            out.append(f"{name}{iv:+d}")
    return " ".join(out)


class ProbeModel:
    """Loaded policy + resolved offsets; the engine's torch boundary."""

    def __init__(self, policy, offsets: ObsOffsets,
                 global_encoder=None, global_off: int = 0, global_dim: int = 0,
                 pokemon_encoder=None, our_team_off: int = 0, opp_team_off: int = 0,
                 turn_delta_encoder=None) -> None:
        self._policy = policy
        self.offsets = offsets
        # For decoding the field state (weather/spikes/screens) out of the obs.
        self._global_encoder = global_encoder
        self._global_off = global_off
        self._global_dim = global_dim
        # For decoding per-mon items + movesets (incl. the opponent's REVEALED ones) from the team blocks.
        self._pokemon_encoder = pokemon_encoder
        self._our_team_off = our_team_off
        self._opp_team_off = opp_team_off
        # For decoding the most-recent TurnDelta (crit + couldn't-move reason) from the history block.
        self._turn_delta_encoder = turn_delta_encoder

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "ProbeModel":
        from sb3_contrib import MaskablePPO
        from agents.observation.state_encoder import (
            Gen3ObservationEncoder,
            load_mappings,
        )

        import agents.observation.constants as C

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
                   global_off=gp["start"], global_dim=gp["dim"],
                   pokemon_encoder=enc.pokemon_encoder,
                   our_team_off=C.OFFSET_OUR_TEAM, opp_team_off=C.OFFSET_OPP_TEAM,
                   turn_delta_encoder=enc.turn_delta_encoder)

    def describe_global(self, obs: np.ndarray) -> "dict | None":
        """Decode the field state (weather, spikes, screens, turn, pending Wish) from the obs."""
        if self._global_encoder is None:
            return None
        arr = np.asarray(obs)
        g = arr[self._global_off:self._global_off + self._global_dim]
        out = self._global_encoder.describe_vector(g)
        # gen3_wish_wired_v1: a pending Wish (heals the slot mon ~50% at end of THIS turn) reads as a
        # non-zero reactive scalar per side — surface it so the human/agent sees the floating heal.
        o = self.offsets
        if o.wish_our_off and o.wish_our_off < arr.shape[0]:
            out["wish_our"] = bool(arr[o.wish_our_off] > 1e-4)
        if o.wish_opp_off and o.wish_opp_off < arr.shape[0]:
            out["wish_opp"] = bool(arr[o.wish_opp_off] > 1e-4)
        return out

    def describe_team(self, obs: np.ndarray) -> "dict[str, dict]":
        """species → {item, moves} for every mon on both sides, decoded from the team blocks.
        Unlike the summary's teams block (our side only, end-of-battle), this reads the per-turn
        obs, so it surfaces the OPPONENT's item + moves the moment they're revealed (unrevealed
        items decode to ``ITM-UNKN`` → dropped; an opp mon's unrevealed moves simply aren't listed).
        Species keys are matched leniently downstream. One decode pass per call (12 mons)."""
        if self._pokemon_encoder is None:
            return {}
        arr = np.asarray(obs)
        stride = self.offsets.pokemon_full_dim
        out: "dict[str, dict]" = {}
        for base in (self._our_team_off, self._opp_team_off):
            for i in range(6):
                s = base + i * stride
                block = arr[s:s + stride]
                if block.shape[0] < stride:
                    continue
                d = self._pokemon_encoder.describe_vector(block)
                species = d.get("species") or ""
                if not species:
                    continue
                item = d.get("item") or ""
                moves = tuple(m for m in (d.get("moves") or []) if m and m != "none")
                out[species] = {
                    "item": item if item and item != "ITM-UNKN" else "",
                    "moves": moves,
                }
        return out

    def describe_turn_outcome(self, obs: np.ndarray) -> "dict":
        """Decode what actually happened on the most-recent TurnDelta in an obs: each side's
        **crit** (critical hit) and **cant** reason (couldn't move — slp/par/flinch/recharge/…).
        The history block's LAST slot is the newest turn (see observation CLAUDE), so for decision
        T's outcome the engine reads decision T+1's obs. Empty dict when no slot / no decoder."""
        from agents.observation.turn_delta_encoder import OFFSET_OPP_CRIT, OFFSET_OUR_CRIT

        off = self.offsets
        td = off.turn_delta_dim
        if td <= 0 or off.turn_history_dim < td:
            return {}
        arr = np.asarray(obs)
        newest = off.turn_history_offset + (off.turn_history_dim // td - 1) * td   # last slot = newest
        slot = arr[newest:newest + td]
        if slot.shape[0] < td:
            return {}
        out = {"our_crit": bool(slot[OFFSET_OUR_CRIT] > 0.5),
               "opp_crit": bool(slot[OFFSET_OPP_CRIT] > 0.5)}
        if self._turn_delta_encoder is not None:
            d = self._turn_delta_encoder.describe_vector(slot)
            out["our_cant"] = d.get("our_cant")   # e.g. 'slp'/'par'/'flinch'/'recharge' or None
            out["opp_cant"] = d.get("opp_cant")
            out["our_boost"] = _boost_delta_str(d.get("our_boost_delta"))   # e.g. "atk+1" (this turn)
            out["opp_boost"] = _boost_delta_str(d.get("opp_boost_delta"))
            out["move_order"] = d.get("move_order")   # "we_first" / "opp_first" / None — who moved first
            # Move type-effectiveness ('immune'/'resisted'/'normal'/'super-effective' or None) — lets
            # the RESULT line explain a 0-damage attack (e.g. 'no effect (immune)' for Normal vs Ghost).
            out["our_effectiveness"] = d.get("our_effectiveness")
            out["opp_effectiveness"] = d.get("opp_effectiveness")
            # Resolved move outcome ('hit'/'miss'/'fail' or None) from gen3_move_outcome_v1 — the
            # RECORDED fate of each side's move, so 'missed' is a fact, not an accuracy guess.
            out["our_move_outcome"] = d.get("our_move_outcome")
            out["opp_move_outcome"] = d.get("opp_move_outcome")
        return out

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

    def belief(self, obs: np.ndarray, mask: np.ndarray):
        """The hidden-opponent SPECIES belief the loaded model computes for THIS obs — the belief
        head's per-slot species logits + which opp slots are believed (un-revealed). Runs ONE clean
        forward (the intervention-sweep / saliency passes overwrite the extractor's stash, so reading
        a leftover would be wrong) and returns ``(species_logits[6, n_species], believed_mask[6] bool)``
        as numpy, or ``None`` when the checkpoint has no belief head (the run trained belief-off). The
        decode → top-k species + the Hungarian truth-match live in the pure engine."""
        import torch

        extractor = getattr(self._policy, "features_extractor", None)
        if extractor is None:
            return None
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        logits = getattr(extractor, "last_belief_logits", None)
        bmask = getattr(extractor, "last_opp_believed_mask", None)
        if logits is None or bmask is None or "species" not in logits:
            return None
        return (logits["species"][0].detach().cpu().numpy(),
                bmask[0].detach().cpu().numpy().astype(bool))

    def damage_op_view(self, obs: np.ndarray, mask: np.ndarray):
        """The unified DamageOperator's view for THIS obs: per-our-mon incoming threat
        ``[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance`` (slot 0 = active, 1-5 = the
        SAFE-SWITCH bench reads), the opp-active effect scalars, and (on a ``--unified-damage both`` run)
        our 4 moves' OUTGOING damage. Runs ONE clean forward and decodes the operator's PRE-gain physics
        stash (`last_raw_block`); ``None`` when the checkpoint has no damage operator (``--damage-op`` off)."""
        import torch
        from agents.model.features_extractor import decode_damage_block

        extractor = getattr(self._policy, "features_extractor", None)
        op = getattr(extractor, "damage_op", None) if extractor is not None else None
        if op is None:
            return None
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        raw = getattr(extractor, "last_raw_block", None)
        if raw is None:
            return None
        return decode_damage_block(raw[0].detach().cpu().numpy(), outgoing=bool(op.outgoing))

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
