"""Thin torch wrapper for the prober engine.

Isolates the only non-pure pieces — loading a MaskablePPO checkpoint and running
forward / backward passes — behind a small object the pure ``engine`` functions
call. Tests substitute a fake exposing the same three members
(``action_dist``, ``logit_grad``, ``offsets``), so the engine is exercised with
zero torch.

We deliberately use raw ``MaskablePPO.load`` (no env, no
``ModelVersion.check_compatible``) to match the legacy ``probe_replay.py`` CLI —
this tool's job is to open ARCHIVED runs, and a compatibility gate that refuses
them is the opposite of what a forensic tool wants.

**But "no gate" was not the same as "a good failure", and that gap is what
`ArchDriftError` closes.** This project iterates the architecture continuously
(root `CLAUDE.md`: *"checkpoint compatibility is not a concern"*), so a
checkpoint older than the last arch change cannot be re-run under current code.
Measured across the 84 archived runs on this box, that surfaced three ways, none
of which said what was wrong:

1. a flag the code DELETED is still baked into the zip's
   ``features_extractor_kwargs`` → ``TypeError: __init__() got an unexpected
   keyword argument 'spread_belief_nature_marginalize'`` (59 runs). **Recovered
   here** — unknown kwargs are dropped, and which ones is reported, because a
   dropped flag means the rebuilt extractor is not the one that played.
2. a value the code now VALIDATES → e.g. ``move_candidate_floor=0.0``, legal
   when trained, rejected since the v65 legality guard (14 runs). Not recovered:
   the guard is correct, and silently relaxing it for a probe would answer with a
   different model than the one being probed.
3. weight SHAPES that no longer fit → ``RuntimeError: mat1 and mat2 shapes cannot
   be multiplied (12x380 and 386x256)``. Not recoverable in principle.

All three now raise ``ArchDriftError`` naming the drift (saved obs dim vs the
encoder's, saved ``arch_signature`` vs the code's, the dropped kwargs, and the
``git_hash`` to check out), instead of a raw torch error from four frames inside
SB3. The failure is the same; the reader's next step is no longer a mystery.
"""

from __future__ import annotations

import inspect
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np


class ArchDriftError(RuntimeError):
    """A checkpoint that cannot be re-run under the CURRENT code.

    Carries the drift itself (`dropped_kwargs`, `saved_obs_dim`, `current_obs_dim`,
    `saved_arch`, `current_arch`, `git_hash`) so a surface can render it as a
    diagnosis rather than re-parsing the message."""

    def __init__(self, message: str, *, dropped_kwargs=(), saved_obs_dim=None,
                 current_obs_dim=None, saved_arch=None, current_arch=None, git_hash=None):
        super().__init__(message)
        self.dropped_kwargs = tuple(dropped_kwargs)
        self.saved_obs_dim = saved_obs_dim
        self.current_obs_dim = current_obs_dim
        self.saved_arch = saved_arch
        self.current_arch = current_arch
        self.git_hash = git_hash


def peek_checkpoint(ckpt_path: str) -> dict:
    """The checkpoint's saved ARCH fingerprint, without deserializing it.

    SB3 stores its `data` block as a plain JSON zip member, so this reads in ~5 ms where
    `MaskablePPO.load` spends seconds on ~27 MB — which is what lets a drift be DIAGNOSED before
    (and explained after) a failed load. Best-effort by design: a checkpoint this cannot parse just
    yields `{}` and the caller proceeds to the normal load path.

    NOTE the returned `extractor_kwargs` is the JSON *rendering* of the saved kwargs — fine for
    asking "which names are in here", useless for reconstruction, since SB3 renders class refs as
    `:serialized:` markers. Rebuilding real kwargs needs `load_from_zip_file` (see `_real_policy_kwargs`).
    """
    out: dict = {}
    try:
        with zipfile.ZipFile(ckpt_path) as z:
            data = json.loads(z.read("data").decode())
    except Exception:  # noqa: BLE001 — a peek NEVER blocks the real load
        return out
    pk = data.get("policy_kwargs")
    if isinstance(pk, str):
        try:
            pk = json.loads(pk)
        except Exception:  # noqa: BLE001
            pk = None
    fek = (pk or {}).get("features_extractor_kwargs")
    if isinstance(fek, dict):
        out["extractor_kwargs"] = fek
        layout = fek.get("layout")
        parts = (layout or {}).get("parts") if isinstance(layout, dict) else None
        if isinstance(parts, dict):
            ends = [p.get("end") for p in parts.values() if isinstance(p, dict) and p.get("end")]
            if ends:
                out["obs_dim"] = int(max(ends))
    return out


def _accepted_extractor_kwargs() -> "set[str] | None":
    """The kwarg names the CURRENT `Gen3FeaturesExtractor.__init__` accepts — or `None` if it takes
    `**kwargs` (then nothing can be called unknown and no filtering is safe)."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    params = inspect.signature(Gen3FeaturesExtractor.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params) - {"self"}


def _sidecar(ckpt_path: str, name: str) -> dict:
    """A checkpoint's `model_config.json` / `metadata.json`, searching its own dir then UP to the run
    root. `{}` when absent (an archived run may have neither).

    THREE levels, not the two `load_model_snapshot` needs: an eval snapshot sits at
    `<run>/eval_traces/step_<N>/snapshot.zip`, so the run-level `metadata.json` — the only place the
    `git_hash` lives, i.e. the one thing that tells a reader which commit to check out — is a full
    grandparent away. Measured: with two levels the drift message silently lost the git hash on
    every retained eval snapshot, which is exactly the case the message exists for."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(3):
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            try:
                with open(cand) as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001 — provenance is best-effort
                return {}
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return {}


def _current_obs_dim() -> "int | None":
    try:
        from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
        return int(Gen3ObservationEncoder(load_mappings()).dimension)
    except Exception:  # noqa: BLE001
        return None


def _arch_drift_error(ckpt_path: str, peek: dict, dropped: "tuple[str, ...]",
                      exc: Exception) -> ArchDriftError:
    """Turn a failed load into a DIAGNOSIS — what drifted, and what to do about it."""
    saved_obs = peek.get("obs_dim")
    cur_obs = _current_obs_dim()
    cfg = _sidecar(ckpt_path, "model_config.json")
    meta = _sidecar(ckpt_path, "metadata.json")
    saved_arch = cfg.get("arch_signature")
    try:
        from agents.model.model_version import ARCH_SIGNATURE as cur_arch
    except Exception:  # noqa: BLE001
        cur_arch = None
    git_hash = meta.get("git_hash") or cfg.get("git_hash")

    lines = [f"This checkpoint cannot be re-run under the current code:\n  {ckpt_path}", ""]
    if saved_obs and cur_obs and saved_obs != cur_obs:
        lines.append(f"  obs dim        trained {saved_obs}  ·  code builds {cur_obs}")
    if saved_arch and cur_arch and saved_arch != cur_arch:
        lines.append(f"  arch_signature trained {saved_arch}  ·  code is {cur_arch}")
    if dropped:
        lines.append("  dropped flags  " + ", ".join(dropped)
                     + "   (deleted since; the rebuilt extractor is NOT the one that played)")
    lines += [
        # Collapsed to ONE line: these carry embedded newlines (the legality guard's message is a
        # paragraph), and truncating a multi-line string mid-sentence produced a ragged fragment
        # that read as a broken message rather than a quoted cause.
        f"  underlying     {type(exc).__name__}: {' '.join(str(exc).split())[:180]}…",
        "",
        "The prober re-runs the model under CURRENT code, so a run trained before an",
        "architecture change is not re-runnable — by design (see the root CLAUDE.md:",
        '"checkpoint compatibility is not a concern"). Options:',
    ]
    if git_hash:
        lines.append(f"  · probe from the commit it was trained on:  git checkout {git_hash}")
    else:
        lines.append("  · probe from the commit it was trained on (metadata.json records git_hash)")
    lines += [
        "  · or probe a run trained at the current architecture",
        "",
        "MODEL-FREE views need none of this and work on every archived run:",
        "  scan · triage · turns · overview · find (except `disagree`) · falsify · calibration",
    ]
    return ArchDriftError("\n".join(lines), dropped_kwargs=dropped, saved_obs_dim=saved_obs,
                          current_obs_dim=cur_obs, saved_arch=saved_arch, current_arch=cur_arch,
                          git_hash=git_hash)


@dataclass(frozen=True)
class ObsOffsets:
    """Semantic obs-vector offsets the analysis depends on.

    Resolved from the live encoder layout so an obs-layout change moves these
    automatically; ``engine_test.py`` pins the resolved values as a regression
    guard so a silent layout shift is caught loudly.
    """

    om_off: int            # our_matchups block (144 dims); 0 = absent (gen3_entity_rehome_v1 deleted it)
    tm_off: int            # their_matchups block (144 dims); 0 = absent (gen3_entity_rehome_v1 deleted it)
    active_block_dim: int  # our active-pokemon block span [0:active_block_dim)
    turn_history_offset: int
    turn_history_dim: int  # n_history_turns * turn_delta_dim
    turn_delta_dim: int = 0  # one TurnDelta slot's width — slices turn_history into per-turn saliency
    # gen3_cpu_damage_deleted_v1: the 4 active-move type-multiplier scalars were DELETED from the obs
    # (the DamageOperator's outgoing per-move block carries real damage instead). 0 = absent, and every
    # consumer no-ops — kept as a field so old archived traces still decode.
    mm_off: int = 0
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
            mm_off=(C.OFFSET_REACTIVE + rl["move_multiplier"]["offset"]) if "move_multiplier" in rl else 0,
            # gen3_entity_rehome_v1: the matchup matrices are DELETED from the live layout —
            # 0 = absent (the mm_off convention); every consumer no-ops / returns None.
            om_off=(C.OFFSET_REACTIVE + rl["our_matchups"]["offset"]) if "our_matchups" in rl else 0,
            tm_off=(C.OFFSET_REACTIVE + rl["their_matchups"]["offset"]) if "their_matchups" in rl else 0,
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
                 turn_delta_encoder=None, dropped_kwargs=()) -> None:
        self._policy = policy
        self.offsets = offsets
        # Extractor kwargs the CURRENT code no longer accepts, dropped so the load could proceed
        # (see the module docstring). NON-EMPTY means the rebuilt extractor is not bit-identical to
        # the one that played, so faithfulness is approximate — a surface must SAY so, exactly as
        # it does for `obs_mismatch`. Empty on a clean load, which is the common case.
        self.dropped_kwargs = tuple(dropped_kwargs)
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

        # ARCH-DRIFT RECOVERY (see the module docstring). Peek at the saved kwargs (~5 ms) and drop
        # any the current extractor no longer accepts, so a checkpoint whose only problem is a
        # DELETED flag still loads. Anything else — a value the code now rejects, weight shapes that
        # no longer fit — is re-raised as an ArchDriftError that names the drift.
        peek = peek_checkpoint(ckpt_path)
        dropped: "tuple[str, ...]" = ()
        custom_objects = None
        accepted = _accepted_extractor_kwargs()
        saved_kwargs = peek.get("extractor_kwargs")
        if accepted is not None and isinstance(saved_kwargs, dict):
            dropped = tuple(sorted(k for k in saved_kwargs if k not in accepted))
            if dropped:
                # Only NOW pay the full deserialize: the JSON peek renders class refs as
                # `:serialized:` markers, so real kwargs have to come from SB3's own loader.
                from stable_baselines3.common.save_util import load_from_zip_file
                data, _, _ = load_from_zip_file(ckpt_path, device=device)
                pk = dict(data.get("policy_kwargs") or {})
                fek = dict(pk.get("features_extractor_kwargs") or {})
                pk["features_extractor_kwargs"] = {k: v for k, v in fek.items() if k in accepted}
                custom_objects = {"policy_kwargs": pk}

        try:
            model = MaskablePPO.load(ckpt_path, device=device, custom_objects=custom_objects)
        except Exception as exc:  # noqa: BLE001 — every failure becomes a DIAGNOSIS, not a stack trace
            raise _arch_drift_error(ckpt_path, peek, dropped, exc) from exc
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
                   turn_delta_encoder=enc.turn_delta_encoder,
                   dropped_kwargs=dropped)

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
        right call for offline sweeps (e.g. the human-agreement probe, the better-line beam)
        where many saved/reconstructed states are scored against a frozen policy."""
        import torch

        obs = np.asarray(obs, dtype=np.float32)
        self._check_obs_dim(obs)
        ot = torch.as_tensor(obs)
        mt = torch.as_tensor(np.asarray(masks))
        with torch.no_grad():
            d = self._policy.get_distribution({"observation": ot, "action_mask": mt})
            lg = d.distribution.logits
            masked = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
            probs = torch.softmax(masked, 1).cpu().numpy()
        return probs

    def values_batch(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """The critic's V(s) for a BATCH of (obs, mask). Shape (N,).

        The batched counterpart of :meth:`value` — one critic forward over the whole search
        frontier instead of N single-row passes (the better-line beam scores every depth-d leaf
        in one pass). ``predict_values`` already takes a batched Dict obs, so this just stacks
        and reshapes; same de-normalized real-return scale as :meth:`value`."""
        import torch

        obs = np.asarray(obs, dtype=np.float32)
        self._check_obs_dim(obs)
        ot = torch.as_tensor(obs)
        mt = torch.as_tensor(np.asarray(masks))
        with torch.no_grad():
            v = self._policy.predict_values({"observation": ot, "action_mask": mt})
        return v.reshape(-1).cpu().numpy()

    def _expected_obs_dim(self) -> "int | None":
        """The obs dim the loaded policy was TRAINED on (its ``observation_space`` 'observation' key)."""
        try:
            return int(self._policy.observation_space["observation"].shape[0])
        except (KeyError, AttributeError, TypeError, IndexError):
            return None

    def _check_obs_dim(self, obs: np.ndarray) -> None:
        """Fail LOUD on an obs-version mismatch (a re-rolled/materialized obs built by the CURRENT
        encoder fed to an OLDER checkpoint) — a clear error instead of a confusing mid-forward torch
        shape error or a silently wrong V(s′). The recorded-state analyze path also flags `obs_mismatch`."""
        exp = self._expected_obs_dim()
        n = int(np.asarray(obs).shape[-1])
        if exp is not None and n != exp:
            raise ValueError(
                f"obs-version mismatch: obs is {n} dims but the loaded model expects {exp} — probe a "
                "model trained on the current obs layout (use --ckpt / the exact tier / --keep-eval-snapshots)")

    def value(self, obs: np.ndarray, mask: np.ndarray) -> float:
        """The critic's V(s) for a single obs/mask (dual-head policy value path)."""
        import torch

        self._check_obs_dim(obs)
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            v = self._policy.predict_values({"observation": ot, "action_mask": mt})
        return float(v.reshape(-1)[0])

    def popart_stats(self) -> "tuple[float, float] | None":
        """The PopArt running ``(mu, sigma)`` of the value targets, or ``None`` when the run
        trains without PopArt. ``value()`` returns the DE-normalized real-return value; the
        PopArt-normalized value ``(V - mu) / sigma`` is the critic's OWN learning scale (the
        ~[-1, 1] space the value loss optimizes, comparable across the run's return-scale drift).
        ``sigma`` is floored well above 0 by the normalizer, so callers can divide safely."""
        pa = getattr(self._policy, "popart", None)
        if pa is None:
            return None
        try:
            return float(pa.mu), float(pa.sigma)
        except (AttributeError, TypeError, ValueError):
            return None

    def value_dist_at(self, obs: np.ndarray, mask: np.ndarray) -> "np.ndarray | None":
        """The distributional value head's per-atom return distribution for an ARBITRARY obs — one
        clean forward, then ``softmax(last_value_dist_logits)``. ``None`` when the checkpoint trained no
        value-dist head (``--value-dist-mode none``). This is the COUNTERFACTUAL analog of the trace's
        recorded ``value_dist`` array: a re-rolled one-ply successor state has no saved row, so the
        lookahead reads the resulting state's distribution HERE. Mirrors ``belief`` / ``damage_op_view``
        (the sweep/saliency passes clobber the stash, so a fresh forward is required)."""
        import torch

        extractor = getattr(self._policy, "features_extractor", None)
        if extractor is None or getattr(extractor, "value_dist_mode", "none") == "none":
            return None
        self._check_obs_dim(obs)
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        logits = extractor.last_value_dist_logits
        if logits is None:
            return None
        return torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

    def win_prob_at(self, obs: np.ndarray, mask: np.ndarray) -> "float | None":
        """The win-probability head's calibrated P(win|s) for an ARBITRARY obs — one clean forward, then
        ``sigmoid(last_win_prob_logits)``. ``None`` when the checkpoint trained no win-prob head
        (``--win-prob-mode none``). The COUNTERFACTUAL analog of the trace's recorded ``win_probs`` array
        for a re-rolled one-ply successor state the lookahead evaluates."""
        import torch

        extractor = getattr(self._policy, "features_extractor", None)
        if extractor is None:
            return None
        self._check_obs_dim(obs)
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        logits = extractor.last_win_prob_logits
        if logits is None:
            return None
        return float(torch.sigmoid(logits[0, 0]).item())

    def value_dist_support(self) -> "tuple[float, float, int] | None":
        """The distributional value head's atom support ``(vmin, vmax, bins)``, or ``None`` when the run
        trained no value-dist head (``--value-dist-mode none``). The trace stores the per-atom probs; the
        prober maps atoms → return units via ``linspace(vmin, vmax, bins)`` to render the histogram + the
        E[Z]/std/percentile reads. On a ``--use-popart`` run the support is in the critic's normalized
        space (the loss normalizes the return target), so ``ValueDistView`` denormalizes E[Z] for display."""
        ex = getattr(self._policy, "features_extractor", None)
        if ex is None or getattr(ex, "value_dist_mode", "none") == "none":
            return None
        try:
            return float(ex.value_dist_vmin), float(ex.value_dist_vmax), int(ex.value_dist_bins)
        except (AttributeError, TypeError, ValueError):
            return None

    def architecture(self) -> "list[dict]":
        """Describe the loaded extractor's FORWARD PIPELINE as ordered plain-dict phases — so the
        prober can DRAW the model instead of asking the reader to imagine it. Reflects the CURRENT
        (config v27) extractor, in true forward order. Each phase carries ``name`` · ``active``
        (optional phases are flag-gated, so this reflects THIS checkpoint) · ``optional`` ·
        ``stage`` (``trunk`` shared body / ``fork`` = the CLSPool split / ``shared`` = post-fork,
        feeds BOTH heads / ``side`` = a stashed readout that does NOT feed pi/vf / ``policy`` /
        ``value`` heads) · ``role`` (what-it-does + key dims, live-interpolated). Empty when the
        extractor can't be introspected (a stub/fake model). Pure data out (torch lives in model.py)."""
        ex = getattr(self._policy, "features_extractor", None)
        if ex is None:
            return []
        from agents.model import features_extractor as F

        def on(attr: str) -> bool:          # an optional sub-module is built (≠ None) ⇒ that phase is active
            return getattr(ex, attr, None) is not None

        d = getattr(F, "D_MODEL", "?")
        layers = getattr(F, "TRANSFORMER_N_LAYERS", "?")
        hist = getattr(F, "N_HISTORY_TURNS", "?")
        tokens = 12 + (hist if isinstance(hist, int) else 0) + 1
        proj = getattr(ex, "projection_dim", getattr(F, "PROJECTION_DIM", "?"))
        pin = getattr(ex, "projection_input_dim", "?")
        vin = getattr(ex, "value_projection_input_dim", "?")
        mb = getattr(ex, "move_belief_mode", "off")
        wp = getattr(ex, "win_prob_mode", "none")
        vd = getattr(ex, "value_dist_mode", "none")     # v29 distributional value head (side readout)
        # move_latent_encoder is ABSENT (not None) when off → its own getattr guard, not on().
        has_latent = getattr(getattr(ex, "pokemon_encoder", None), "move_latent_encoder", None) is not None
        prior_fusion = bool(getattr(ex, "move_prior_fusion", False))    # MoveBelief Smogon-prior posterior
        dmg_out = bool(getattr(ex, "damage_outgoing", False))           # DamageOperator outgoing direction
        # (name, active, optional, stage, role, attn) — TRUE forward order (forward_internal). `attn`
        # flags the ATTENTION layers (self-/cross-attention) so the prober can mark where the network
        # attends. gen3_tiered_pipeline_v1 order: Embeddings → ObsUnpack → PokemonEncoder[+MoveLatent]
        # → [BeliefSlots] → [MoveBelief] → [SpreadBelief] → [DamageOperator] → TeamTransformer →
        # [BeliefHead·side] → CLSPool(fork) → [WinProbHead·side] → [HiddenOppBeliefPool] →
        # ProjectionAssembler → π / V. The belief + physics are T0/T1: they all run PRE-transformer,
        # unconditionally — there is no longer a POST placement.
        rows = [
            ("Embeddings", True, False, "trunk",
             "shared tables: species/move/item/ability/type emb + HP soft-type + TurnDelta embedder", False),
            ("ObsUnpack", True, False, "trunk", "flat obs → per-mon tensors + global/reactive scalars + masks", False),
            ("PokemonEncoder", True, False, "trunk",
             f"6+6 mons → role tokens ({d}d) · within-mon move self-attn" + ("  + MoveLatent(v24)" if has_latent else ""),
             True),
            ("BeliefSlots", on("belief_slots"), True, "trunk",
             "fill hidden-opp slots with learned tokens (in-lineup)", False),
            ("MoveBelief", on("move_belief"), True, "trunk",
             f"predict + reinject opp moves ({mb})" + (" + prior-fusion" if prior_fusion else "")
             + " · T0 RESOLVE, PRE-transformer", False),
            ("SpreadBelief", on("spread_belief"), True, "trunk",
             "predict + reinject opp hidden stat-spread → DamageOperator (v25)"
             + " · T0 RESOLVE, PRE-transformer", False),
            ("DamageOperator", on("damage_op"), True, "shared",
             "differentiable gen3 damage: incoming P(KO)" + (" + outgoing per-move" if dmg_out else "")
             + " → the trunk (prefuse_proj) + both heads · T1 REASON, PRE-transformer, once",
             False),
            ("TeamTransformer", True, False, "trunk",
             f"SELF-ATTENTION over {tokens} tokens (12 mon + {hist} hist + 1 global) · {layers} layers ({d}d)", True),
            ("BeliefHead", on("belief_head"), True, "side",
             "aux: predict hidden species/moves (side readout)", False),
            ("CLSPool", True, False, "fork",
             "CLS queries CROSS-ATTEND the team tokens → our/their/value pools — FORKS → π · V", True),
            ("WinProbHead", wp != "none", True, "side", f"P(win) readout off value_pooled ({wp})", False),
            ("ValueDistHead", on("value_dist_head"), True, "side",
             f"return-distribution readout off value_pooled ({vd})", False),
            ("HiddenOppBeliefPool", on("hidden_opp_belief"), True, "shared",
             "k belief queries CROSS-ATTEND the 12 tokens → both heads", True),
            ("ProjectionAssembler", True, False, "shared",
             "→ (pi_combined, vf_combined): pools + ctx + belief + damage", False),
            ("π policy head", True, False, "policy", f"pi_combined → norm → proj({proj})  [in {pin}]", False),
            ("V value head", True, False, "value", f"vf_combined → norm → proj({proj})  [in {vin}]", False),
        ]
        return [{"name": n, "active": bool(a), "optional": o, "stage": s, "role": r, "attn": at}
                for (n, a, o, s, r, at) in rows]

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
        logits = extractor.last_belief_logits
        bmask = extractor.last_opp_believed_mask
        if logits is None or bmask is None or "species" not in logits:
            return None
        return (logits["species"][0].detach().cpu().numpy(),
                bmask[0].detach().cpu().numpy().astype(bool))

    def damage_op_view(self, obs: np.ndarray, mask: np.ndarray):
        """The unified DamageOperator's view for THIS obs: per-our-mon incoming threat
        ``[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance`` in TEAM-SLOT order (the active is
        whichever slot holds the active flag; the bench slots are the SAFE-SWITCH reads), the opp-active
        Choice-Band tail, and (on a ``--unified-damage both`` run) our 4 moves' OUTGOING damage. On a
        ``--damage-matrices incoming/outgoing`` run it carries `incoming_matrix` (per opp-move × per OUR-mon
        FULL cell `[low,high,crit,pko,type_mult,status_lands]` + a per-move header of belief/acc/is_phys/
        effect/secondary, each move with its decoded NAME — exact, from the op's stashed candidate indices,
        typed HP rendered as ``hiddenpower(type)``) and/or `outgoing_matrix` (our 4 moves × the opp's 6
        mons). Runs ONE clean forward and decodes the operator's PRE-gain physics stash; ``None`` when the
        checkpoint has no damage operator (``--damage-op`` off).

        NOTE: the stash lives on the **DamageOperator submodule** (`op.last_raw_block`, set inside its
        forward), NOT on the extractor — reading `extractor.last_raw_block` always returned None, silently
        hiding the whole op view in the prober (regression-guarded by `model_test.py`)."""
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
        raw = op.last_raw_block                              # the op stashes on ITSELF, not the extractor
        if raw is None:
            return None
        # The discrete incoming move-space is decoded at the op's `matrices_incoming_k` (the lean top-K
        # block that used to share the `topk_k` knob was deleted — gen3_op_block_trim_v1). Likewise pass
        # `matrices_outgoing` for the outgoing matrix. (Missing decode flags were the cause of the nonsense
        # "acc-580" render on matrix runs.)
        matrices_in_k = int(getattr(op, "matrices_incoming_k", 0))
        matrices_out = bool(getattr(op, "matrices_outgoing", False))
        view = decode_damage_block(
            raw[0].detach().cpu().numpy(), outgoing=bool(op.outgoing),
            matrices_outgoing=matrices_out, matrices_incoming_k=matrices_in_k)
        # Resolve each candidate move to its EXACT name from the op's stashed indices (the op knows which
        # candidate it picked — better than a nearest-latent decode); `last_topk_idx` is set by
        # `_incoming_matrix`.
        idx = op.last_topk_idx
        if idx is not None:
            names = self._topk_move_names(op, [int(c) for c in idx[0].detach().cpu().tolist()])
            blk = view.get("incoming_matrix")
            if blk and blk.get("moves"):
                for k, mv in enumerate(blk["moves"]):
                    mv["move"] = names[k] if k < len(names) else None
        # gen3_op_move_align_v1: the op's OUTGOING per-move blocks (outgoing / status_landing /
        # outgoing_matrix) now read the request-ordered obs slice (`ctx.our_active_req_move_*`), so they are
        # ACTION-ordered (action 6+k) — the prober labels them with `a.matchups.move_labels` (the recorded,
        # action-index-ordered labels), the same axis as the matchups table + faithfulness. No per-mon
        # (sorted-by-id) relabel is needed anymore; the old `our_moves` decode + the "op move order ≠ action
        # order" caveat are gone with the fix.
        return view

    def _topk_move_names(self, op, cand_indices):
        """Resolve the DamageOperator's top-K CANDIDATE indices → move-id strings. gen3_opp_hp_typed_candidates_v1:
        the candidate axis is C = n_moves, and the opponent's Hidden Power is the 16 ORDINARY TYPED move-nums
        355-370 — so each decodes via the normal num→id path with its TYPE preserved (``hiddenpower(ice)``); the
        bare typeless 237 (the masked presence token, never selected) maps to bare ``hiddenpower``. No HP-special
        index→type collapse (the source of the old prober HP ambiguity)."""
        from agents import gen3_data
        if getattr(self, "_topk_num_to_id", None) is None:
            m = {}
            for mid, v in gen3_data.moves.raw().items():
                if not v.get("num"):
                    continue
                num = int(v["num"])
                if mid == "hiddenpower":
                    m.setdefault(num, "hiddenpower")                        # the bare typeless presence token
                elif mid.startswith("hiddenpower"):
                    m[num] = f"hiddenpower({mid[len('hiddenpower'):]})"     # typed 355-370 → typed display
                else:
                    m[num] = mid
            self._topk_num_to_id = m
        return [self._topk_num_to_id.get(int(c), f"#{int(c)}") for c in cand_indices]

    def move_belief(self, obs: np.ndarray, mask: np.ndarray):
        """The model's MOVE belief for THIS obs — per opp slot the multi-label move-prediction posterior
        ``sigmoid(last_move_belief_logits)`` (a --move-prior-fusion run's logits are already prior⊕learned),
        PLUS each side's per-slot species/revealed-moves/active decoded from the obs team blocks. Lets the
        engine show 'what the model thinks the REVEALED opponent's still-UNSEEN moves are' and label the
        per-our-mon damage-op rows (team-slot order). Runs ONE clean forward (the sweep/saliency passes
        clobber the stash). Returns ``None`` when the checkpoint has no move-belief head
        (``--move-belief-mode off``). The decode → top-k believed moves lives in the pure engine."""
        import torch
        import agents.observation.constants as C

        extractor = getattr(self._policy, "features_extractor", None)
        if extractor is None or self._pokemon_encoder is None:
            return None
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        logits = extractor.last_move_belief_logits
        if logits is None:
            return None
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()        # [6, n_moves] P(move in slot's set)
        arr = np.asarray(obs)
        stride = self.offsets.pokemon_full_dim

        def _slots(base: int) -> list:
            out = []
            for i in range(6):
                block = arr[base + i * stride: base + (i + 1) * stride]
                if block.shape[0] < stride:
                    out.append({"species": "", "revealed_moves": (), "known": False, "active": False})
                    continue
                # The AUTHORITATIVE revealed flag is the species_known obs bit — NOT the decoded species
                # string (an UN-revealed opp slot decodes to a placeholder like "Unknown(0)", which must
                # NOT count as revealed). Our team is always known; an opp slot is known iff revealed.
                known = bool(block[C.POKEMON_SPECIES_KNOWN_OFFSET] > 0.5)
                d = self._pokemon_encoder.describe_vector(block) if known else {}
                species = (d.get("species") or "").strip() if known else ""
                moves = tuple(m for m in (d.get("moves") or []) if m and m != "none")
                active = bool(block[stride - 1] > 0.5)
                out.append({"species": species, "revealed_moves": moves, "known": known, "active": active})
            return out

        return {"opp_probs": probs,
                "opp_slots": _slots(self._opp_team_off),
                "our_slots": _slots(self._our_team_off)}

    def spread_belief_view(self, obs: np.ndarray, mask: np.ndarray):
        """The SpreadBelief's predicted opp DERIVED stats for THIS obs — per opp slot
        ``[atk, def, spa, spd, spe]`` (the L100 stat VALUES the `DamageOperator` consumes instead of its
        hand-coded de-timid constants, so a wrong spread is an otherwise-invisible damage root-cause). Runs
        ONE clean forward (the sweep/saliency passes clobber the stash) and reads ``last_spread_belief``.

        The head predicts spreads for the REVEALED opp mons (species known, EV spread unknown — the
        believed mask is `~opp_believed_mask`), so the view keys off the per-slot revealed species (decoded
        from the opp team obs block, obs-slot order — the SAME slot order as the spread tensor). Returns a
        dict ``{spread[6,5], believed_mask[6] bool, opp_species[6] (id or ''), opp_active}`` or ``None`` when
        the run trained ``--spread-belief`` off. The true-derived-stat join + the Smogon prior live in the
        pure engine (which matches each revealed slot to its true mon by species)."""
        import torch
        import agents.observation.constants as C

        extractor = getattr(self._policy, "features_extractor", None)
        if extractor is None:
            return None
        ot = torch.as_tensor(obs).unsqueeze(0)
        mt = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            self._policy.extract_features({"observation": ot, "action_mask": mt})
        sb = extractor.last_spread_belief
        if sb is None:
            return None
        bmask = extractor.last_opp_believed_mask
        bm = bmask[0].detach().cpu().numpy().astype(bool) if bmask is not None else None
        # Per opp slot: the REVEALED species id (obs-slot order; "" when hidden) so the engine can match the
        # believed spread row → the true mon. Mirrors `move_belief`'s _slots decode (authoritative known bit).
        opp_species, opp_active = [], -1
        stride = self.offsets.pokemon_full_dim
        arr = np.asarray(obs)
        for i in range(6):
            base = self._opp_team_off + i * stride
            block = arr[base: base + stride]
            if block.shape[0] < stride:
                opp_species.append("")
                continue
            known = bool(block[C.POKEMON_SPECIES_KNOWN_OFFSET] > 0.5)
            sid = ""
            if known and self._pokemon_encoder is not None:
                sid = (self._pokemon_encoder.describe_vector(block).get("species") or "").strip()
            opp_species.append(sid)
            if block[stride - 1] > 0.5:
                opp_active = i
        return {"spread": sb[0].detach().cpu().numpy(), "believed_mask": bm,
                "opp_species": opp_species, "opp_active": opp_active}

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
