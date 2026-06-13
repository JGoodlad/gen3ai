import numpy as np
from gymnasium import spaces
from typing import Optional

from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from poke_env.data.normalize import to_id_str

from agents.observation.state_encoder import get_observation_encoder
from agents.observation.base import ObservationEncoder
from agents.observation.constants import (
    TEAM_SIZE, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET,
)
from agents.observation.belief_labels import (
    build_belief_labels, zero_belief_labels, BELIEF_MOVE_SLOTS,
)
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.model.features_extractor import N_HISTORY_TURNS
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.battle.live_view import LegalActions
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_function import RewardFunction
from agents.training.turn_delta import TurnDelta
from agents.training.episode_tracker import EpisodeTracker
from agents.training.stall import StallConfig, StallLogger
from agents.battle.gen3_battle import Gen3Battle
from utils.logging.levels import LogLevel


class Gen3Env(SinglesEnv):
    def __init__(self, mappings, reward_fn: Optional[RewardFunction] = None,
                 log_level=LogLevel.QUIET, stall_config: Optional[StallConfig] = None,
                 *args, battle_class=Gen3Battle, emit_belief_labels: bool = False, **kwargs):
        self.log_level = log_level
        self._stall_logger = StallLogger(stall_config)
        super().__init__(*args, **kwargs)
        # poke-env's PokeEnv builds its two _EnvPlayer agents internally without a
        # battle_class seam. _battle_class is read per-battle at _create_battle time
        # (no battle has started yet here), so setting it on the agents post-init
        # makes every battle a Gen3Battle (event log + live_view) with zero edits to
        # poke-env's env. The trainee (battle1) is what obs/reward/replay read.
        self.agent1._battle_class = battle_class
        self.agent2._battle_class = battle_class
        self.observation_encoder = get_observation_encoder(mappings)

        obs_dim = self.observation_encoder.dimension
        self.vector_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        # Hidden-opponent belief AUX labels (TRAINING-ONLY): when on, the obs Dict carries two
        # PRIVILEGED int64 keys (the opponent's still-hidden mons) consumed ONLY by the PPO aux loss.
        # The model forward reads only obs["observation"], so they never leak into the acting path;
        # eval/self-play/inference run with this off and never declare/need them. Single source of the
        # enable flag is --opp-belief-aux-coef>0 (threaded as emit_belief_labels from train_rl_agent).
        self._emit_belief_labels = emit_belief_labels
        # Precompute id -> embedding-NUM maps once (keyed by gen3_data species/move id) for the labeller.
        self._species_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items() if "num" in rec}
        self._move_num = {mid: rec["num"] for mid, rec in mappings.get("moves", {}).items() if "num" in rec}
        base_obs = {
            "observation": self.vector_space,
            "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8),
        }
        if self._emit_belief_labels:
            _imax = np.iinfo(np.int64).max
            # low=-1 keeps the PAD / not-scored sentinel in-space; Box(int64) (NOT Discrete, which
            # rejects -1 and the rollout buffer special-cases it).
            base_obs["belief_species"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE,), dtype=np.int64)
            base_obs["belief_moves"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE, BELIEF_MOVE_SLOTS), dtype=np.int64)
        # SB3 reads the SINGULAR observation_space (threaded as the VecEnv space); the PLURAL
        # observation_spaces is intercepted + rewrapped by PokeEnv.__setattr__ (it would drop the
        # belief keys) and is not the SB3-facing space, so it can stay minimal.
        self.observation_space = spaces.Dict(base_obs)
        self.observation_spaces = {
            self.agent1.username: self.observation_space,
            self.agent2.username: self.observation_space
        }

        self.reward_manager: RewardFunction = reward_fn or Gen3RewardManager(log_level=self.log_level)
        self._tracker = EpisodeTracker(history_cap=N_HISTORY_TURNS)
        # Share ONE ProgressClock between obs and reward (design §5.1): the tracker owns it (updated
        # at embed time so the obs is fresh); the reward READS its stashed last_penalty. Set the
        # clock's per-run penalty magnitude once from the reward config (single source of truth).
        if hasattr(self.reward_manager, "progress_clock"):
            self.reward_manager.progress_clock = self._tracker.progress_clock
            cfg = getattr(self.reward_manager, "config", None)
            if cfg is not None:
                self._tracker.progress_clock.no_progress_penalty = cfg.no_progress_penalty
        self._pending_delta = None   # delta folded once at embed time, reused by calc_reward
        self._turn_delta_encoder = TurnDeltaEncoder(
            mappings.get("moves", {}),
            mappings.get("species", {}),
        )

    def embed_battle(self, battle):
        # Record FIRST so the tracker's HP-candidate state reflects the just-fired
        # HP (if any) before we encode the obs. The observation at turn N then
        # carries the narrowing from turns 1..N-1.
        legal = None
        if battle is self.battle1:
            # Clear any prior cached delta so a non-recording embed (terminal / no legal action)
            # forces calc_reward to re-fold rather than reuse a stale window.
            self._pending_delta = None
        if battle is self.battle1 and not battle.strict_view().finished:
            # Capture the server-authoritative legality snapshot ONCE this decision and
            # thread it to the mask, the recorded context, AND the obs encoder (its
            # trapped / maybe_trapped reactive bits), so the mapper later decodes the chosen
            # action against the exact same immutable surface and the encoder doesn't rebuild it.
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)
            if mask.sum() > 0:
                self._tracker.record(battle, mask, legal=legal)
                # Advance the shared ProgressClock for the JUST-COMPLETED window BEFORE encode reads
                # it (design §5.1). Cache the returned delta so calc_reward reuses it (no double fold).
                self._pending_delta = self._tracker.update_progress_clock(battle, legal)

        if battle is self.battle1:
            obs = self.observation_encoder.encode(
                battle, hp_tracker=self._tracker.hidden_power_tracker, legal=legal,
                progress_clock=self._tracker.progress_clock,
            )
            prev_mask = self._tracker.prev_mask
            history_vecs = self._tracker.prev_N_delta_vecs(N_HISTORY_TURNS, self._turn_delta_encoder, battle=battle)
        else:
            obs = self.observation_encoder.encode(battle)
            prev_mask = np.ones(11, dtype=np.float32)
            history_vecs = np.zeros((N_HISTORY_TURNS, self._turn_delta_encoder.dimension), dtype=np.float32)

        return np.concatenate([obs, prev_mask, history_vecs.flatten()])

    def action_masks(self) -> np.ndarray:
        ctx = self._tracker.last_ctx
        if ctx is not None:
            return ctx.mask
        import sys
        sys.stderr.write(
            "[WARN] action_masks() called before any BattleContext was built — "
            "returning all-valid fallback. This should only happen before the first reset.\n"
        )
        return np.ones(11, dtype=np.int8)

    def get_action_mask(self, battle):
        if battle is self.battle1 and self._tracker.last_ctx is not None:
            return self._tracker.last_ctx.mask
        return Gen3ActionMasker.get_mask(battle).astype(np.int8)

    def _belief_labels(self, obs_vec) -> dict:
        """Build the privileged hidden-opponent belief labels for the trainee's CURRENT decision.

        Source of truth for the opponent's FULL team is `battle2.team` — agent2's OWN battle view, so
        it knows all six of its mons (the trainee's `battle1.opponent_team` only holds the revealed
        ones). The believed-slot mask is read DIRECTLY from `obs_vec` — the SAME per-slot `species_known`
        the model's `BeliefSlots` keys its injection on — so the label's believed slots can NEVER diverge
        from where the model fills unknown-mon tokens (single source of truth, not a second derivation).
        Returns {"belief_species": int64[6], "belief_moves": int64[6,4]}; all-PAD until both battles
        exist (early reset / pre-team)."""
        b1 = getattr(self, "battle1", None)
        b2 = getattr(self, "battle2", None)
        if b1 is None or b2 is None:
            bs, bm = zero_belief_labels()
            return {"belief_species": bs, "belief_moves": bm}
        # Per-opp-slot species_known straight from the obs the model reads (NOT re-derived from a count).
        species_known = [
            float(obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET])
            for i in range(TEAM_SIZE)
        ]
        # FAIL LOUD on a broken structural invariant: revealed opp slots MUST be a leading-contiguous
        # block (the encoder packs revealed-first, believed trailing). A gap means the encoder's slot
        # packing changed and the label↔BeliefSlots alignment is silently corrupt — crash rather than
        # train the belief head on mis-slotted supervision. (A legit data edge — a hidden mon whose
        # name doesn't map — is handled gracefully inside build_belief_labels; this is structural.)
        n_known = sum(1 for s in species_known if s >= 0.5)
        if any(species_known[i] >= 0.5 for i in range(n_known, TEAM_SIZE)):
            raise RuntimeError(
                f"belief labels: opp species_known {species_known} is not leading-contiguous — the "
                "encoder's revealed-first opp-slot packing changed, breaking the believed-slot "
                "alignment with the model's BeliefSlots. Fix the slot order or the label builder."
            )
        revealed = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m is not None]
        revealed_species = [m.species for m in revealed]
        full = list(b2.team.values())
        team_species = [m.species for m in full]
        team_moves = [[mv.id for mv in m.moves.values()] for m in full]
        bs, bm = build_belief_labels(
            team_species, team_moves, revealed_species, species_known,
            self._species_num, self._move_num, to_id_str,
        )
        return {"belief_species": bs, "belief_moves": bm}

    def action_to_order(self, action, battle, **kwargs):
        if isinstance(action, BattleOrder):
            return action
        if battle is self.battle1:
            if battle.strict_view().turn >= self._stall_logger.threshold:
                self._stall_logger.log_once(battle, suffix="STALL")
                return ForfeitBattleOrder()
            ctx = self._tracker.last_ctx
            Gen3ActionMapper.assert_decision_current(ctx, battle)
            return Gen3ActionMapper.action_to_order(
                action, battle, legal=ctx.legal, mask=ctx.mask,
            )
        return super().action_to_order(action, battle)

    def calc_reward(self, battle):
        if battle is self.battle1:
            # Reuse the delta embed_battle already folded for the ProgressClock (same window); fall
            # back to a fresh fold if embed didn't run (e.g. terminal with no decision recorded).
            delta = self._pending_delta if self._pending_delta is not None else self._tracker.build_delta(battle=battle)
            self._pending_delta = None
            return self.reward_manager.process_turn_reward(battle, delta)
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def step(self, action):
        try:
            battle = getattr(self, "_battle", None) or self.battle1
            trainee_idx = action.get(self.agent1.username, -1) if isinstance(action, dict) else action
            if battle is self.battle1 and self._tracker.last_ctx is not None:
                self._tracker.advance(trainee_idx)
                self.reward_manager.record_action(self._tracker.last_ctx, trainee_idx)
            out = super().step(action)
            if self._emit_belief_labels:
                agent_obs = out[0].get(self.agent1.username)
                if agent_obs is not None:
                    agent_obs.update(self._belief_labels(agent_obs["observation"]))
            return out
        except Exception as e:
            import traceback
            print(f"ERROR IN STEP: {e}")
            traceback.print_exc()
            raise e

    def reset(self, *args, **kwargs):
        self.reward_manager.report_episode(getattr(self, "battle1", None))
        self._tracker.reset()
        try:
            if hasattr(self, "agent1"):
                self.agent1.save_replays = None
            self.reward_manager.reset()
            self._stall_logger.reset()
            out = super().reset(*args, **kwargs)
            if self._emit_belief_labels:
                obs, info = out
                agent_obs = obs.get(self.agent1.username)
                if agent_obs is not None:
                    agent_obs.update(self._belief_labels(agent_obs["observation"]))
                return obs, info
            return out
        except Exception as e:
            import traceback
            print(f"ERROR IN RESET: {e}")
            traceback.print_exc()
            raise e
