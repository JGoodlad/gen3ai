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
    build_belief_labels, build_known_move_labels, build_belief_target_index,
    zero_belief_labels, zero_known_moves, BELIEF_MOVE_SLOTS,
    build_known_spread_labels, zero_spread_labels, SPREAD_STAT_ORDER, N_SPREAD_STATS,
    build_known_nature_ev_labels, zero_nature_ev_labels,
    build_hp_type_labels, zero_hp_type_labels, hp_type_idx_from_move_id, N_HP_TYPES_LABEL,
)
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.model.features_extractor import N_HISTORY_TURNS
from agents.model.damage_tables import invert_nature_evs
from agents import gen3_data
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


# gen3_defensive_entropy_v1: the active mon counts as having a productive HP-recovery opportunity (for the
# defensive-exploration entropy boost) only when it has taken at least this much chip — below the threshold a
# heal restores too little to be worth exploring (a Wish cast at full HP for a teammate is the accepted miss).
_DEFENSIVE_HEAL_HP = 0.85


class Gen3Env(SinglesEnv):
    def __init__(self, mappings, reward_fn: Optional[RewardFunction] = None,
                 log_level=LogLevel.QUIET, stall_config: Optional[StallConfig] = None,
                 *args, battle_class=Gen3Battle, emit_belief_labels: bool = False,
                 move_belief_mode: str = "off", emit_belief_target: bool = False,
                 emit_win_target: bool = False, emit_spread_labels: bool = False,
                 emit_hp_type_labels: bool = False, emit_defensive_opportunity: bool = False,
                 emit_pubval_target: bool = False, distill_team_species=None,
                 opponent_team=None, **kwargs):
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
        # OPPONENT-side teambuilder (the same post-init injection seam as _battle_class above).
        # PokeEnv passes its single `team=` kwarg to BOTH internal _EnvPlayers — and the opponent
        # Players the wrapper rotates per episode are pure DECISION functions over battle2 (agent2
        # does the networking), so THEIR teambuilders are dead weight: agent2's `_team` decides the
        # opponent's real team. Without this seam, a `--trainee-team` pin silently pinned the
        # OPPONENTS to the trainee's team too — every "specialist vs diverse field" run was actually
        # a single-team MIRROR vs bot pilots (the root cause of the inflated ~100% training win
        # rates: bots piloting an expertise-gated stall team are trivially beatable, so the wins
        # were genuine but the curriculum was fake). None → both sides keep `team=` (the pre-fix
        # behavior, correct for non-pinned runs where both draw the same pool).
        if opponent_team is not None:
            self.agent2._team = opponent_team
        self.observation_encoder = get_observation_encoder(mappings)

        # Stage-3 generator adoption: the vector space is GENERATED from the declarative schema
        # (one source for the obs shape — the schema's tiling proof ran against the same layout
        # this encoder built, so a dim drift here is structurally impossible). Byte-identical to
        # the old inline Box(-inf, inf, (dimension,), float32) — pinned by schema_test.
        from agents.observation.schema import build_schema
        self.vector_space = build_schema(self.observation_encoder.get_layout()).gym_space()
        self.action_space = spaces.Discrete(11)
        # Hidden-opponent belief AUX labels (TRAINING-ONLY): when on, the obs Dict carries two
        # PRIVILEGED int64 keys (the opponent's still-hidden mons) consumed ONLY by the PPO aux loss.
        # The model forward reads only obs["observation"], so they never leak into the acting path;
        # eval/self-play/inference run with this off and never declare/need them. Single source of the
        # enable flag is --opp-belief-aux-coef>0 (threaded as emit_belief_labels from train_rl_agent).
        # The MOVE belief (--move-belief-mode) also needs labels: belief_moves (its 'unknown' slots, via
        # the same Hungarian) + known_moves (its 'known' slots — revealed mons' FULL privileged moveset).
        # Either feature ON ⇒ emit the label keys; known_moves is only consumed by the move loss.
        self._move_belief_mode = move_belief_mode
        self._emit_known_moves = move_belief_mode in ("revealed", "both")
        self._emit_belief_labels = emit_belief_labels or move_belief_mode != "off"
        # Latent-belief target (TRAINING-ONLY): when on, the obs Dict also carries `belief_target_slots`
        # [6,107] — the FRESH per-mon obs encode of each hidden mon at its believed slot (same canonical
        # assignment as belief_species). The extractor runs the model's own pokemon_encoder over it
        # (stop-grad) to get the encoder-role-token the latent head regresses toward. Enabled by
        # --opp-belief-latent-coef>0 (which requires --opp-belief-aux-coef>0, so _emit_belief_labels is
        # already True). Read ONLY by the latent aux loss — never enters the policy/value forward.
        self._emit_belief_target = emit_belief_target
        # SPREAD-belief label key (TRAINING-ONLY, gen3_unified_spread_belief_v1): when on, the obs Dict
        # carries `belief_spread` [6,5] (the TRUE derived stats {atk,def,spa,spd,spe} of each REVEALED opp
        # mon, from agent2's own team) + `belief_spread_mask` [6]. Consumed ONLY by the spread-belief loss
        # (instrumented_ppo._spread_belief_loss); the SpreadBelief head learns the opponent's hidden EV
        # spread instead of sitting at the usage-mean prior, so the DamageOperator prices damage against the
        # opponent's REAL bulk/offense/speed. Enabled by --spread-belief-coef>0 (requires --spread-belief).
        # Read ONLY by the loss — never enters the policy/value forward (the believed stats the op consumes
        # are the model's own prediction, not this label).
        self._emit_spread_labels = emit_spread_labels
        # gen3_nature_ev_belief_v1: the inverted (species -> nature_num, EVs) map is FIXED per battle (agent2's
        # team doesn't change), but the inversion is ~expensive (25 natures × 64 EVs / mon), so cache it keyed
        # by the team's species set and recompute only on a new battle. Read by _spread_labels.
        self._nature_ev_cache_key = None
        self._nature_ev_cache = {}
        # HP-TYPE-belief label key (TRAINING-ONLY, gen3_opp_hp_type_belief_v1): when on, the obs Dict carries
        # `hp_type_label` [6] (the TRUE Hidden Power type index 0..15 of each REVEALED opp mon that runs HP,
        # from agent2's own team's typed move id) + `hp_type_mask` [6]. Consumed ONLY by the HP-type CE loss
        # (instrumented_ppo._hp_type_belief_loss); the HPTypeBelief head learns which HP the opponent has so
        # the DamageOperator prices the right typed-HP threat (Gen 3 NEVER reveals the opp HP type, so this is
        # a privileged label — never in the obs vector / pi-vf forward). Enabled by --hp-type-belief learned
        # + --hp-type-belief-coef>0 (threaded as emit_hp_type_labels from train_rl_agent).
        self._emit_hp_type_labels = emit_hp_type_labels
        # WIN-PROBABILITY label keys (TRAINING-ONLY): when on, the obs Dict carries `win_target` [1] and
        # `win_mask` [1] (float32). The env emits PLACEHOLDER zeros each step; the WinProbLabelCallback
        # OVERWRITES them post-collection with the Monte-Carlo episode outcome (win=1/loss=0) + a known
        # mask (the outcome is a FUTURE quantity, so it can't be a real per-step obs like the belief
        # labels). Read ONLY by the win-prob aux loss; the model forward reads only obs["observation"].
        # Enabled by --win-prob-mode != none (threaded as emit_win_target from train_rl_agent).
        self._emit_win_target = emit_win_target
        # DEFENSIVE-EXPLORATION flag (TRAINING-ONLY, gen3_defensive_entropy_v1): when on, the obs Dict carries
        # `defensive_opportunity` [1] = 1.0 on decisions where the active mon has a PRODUCTIVE defensive option
        # (a legal HP-recovery move with HP to restore, OR a legal self/team status-cure with a status to
        # clear), else 0.0. Read ONLY by the state-conditioned entropy boost (the PPO loss weights the entropy
        # bonus up on these decisions so the policy keeps exploring defensive moves instead of collapsing to
        # attacking) — never enters the policy/value forward. Enabled by --defensive-entropy-boost > 1.0.
        self._emit_defensive_opportunity = emit_defensive_opportunity
        # PUBLIC-VALUE aux target (TRAINING-ONLY, gen3_pubval_aux_v1): when on, the obs Dict carries
        # `pubval_target` [1] = the FROZEN human-replay-calibrated V_pub = P(win | PUBLIC board) evaluated
        # on the current board (a REAL per-step value like win_margin, NOT a placeholder — V_pub is a pure
        # function of present public state) + `pubval_mask` [1] (0 only before the battle objects exist).
        # Read ONLY by the pubval aux loss (the PubValHead regresses toward it); never enters the forward,
        # never enters GAE. The artifact loads ONCE here — a missing/stale data/gen3_pubval.json fails the
        # run at construction (fail-loud), not mid-rollout. Enabled by --pubval-mode != none.
        self._emit_pubval_target = emit_pubval_target
        self._pubval_model = None
        if emit_pubval_target:
            from agents.training.pubval import PubValModel
            self._pubval_model = PubValModel.load()
        # gen3_exploiter_distill_v1: `distill_mask` [1] = 1.0 iff the trainee's CURRENT team IS the frozen
        # distillation teacher's team (the exploiter's pinned team). Read ONLY by the exploiter-distillation
        # KL in the PPO loss, which masks the teacher's advice to these states (elsewhere the specialist is
        # off-distribution and would corrupt the other teams). `distill_team_species` is the teacher team's
        # species id-set (frozenset), matched against the trainee's own team (`battle1.team`). None → the key
        # is not emitted. Cached per battle (the team is fixed for a battle) in `_distill_active`.
        # N teachers: distill_team_species is a LIST of species id-sets (one per teacher). The distill_mask
        # obs key holds the INTEGER team-id (0=none, k=teacher k, 1-indexed). N=1 → id ∈ {0,1}, obs space
        # unchanged vs the single-teacher form (Box high = len(list) = 1) → running single-teacher runs resume.
        # A teacher may own MANY teams (a multi-team `--trainee-teams` z-cluster exploiter), so each entry
        # is a LIST of species-sets and teacher k's mask fires on ANY of them. A BARE set (the older
        # one-team-per-teacher form) is wrapped, so both shapes work and old callers are unaffected.
        self._emit_distill_mask = bool(distill_team_species)
        self._distill_team_species = [
            [sp] if isinstance(sp, (set, frozenset)) else list(sp)
            for sp in (distill_team_species or [])
        ]
        self._distill_team_id = None  # per-battle cache (0=none, k=teacher k)
        # Per-battle cache of the fresh per-mon identity encodes (keyed by species id). A hidden mon is
        # untouched (full HP, no status) until revealed, at which point it leaves the believed set, so a
        # mon's fresh encode is stable while it is a target — caching is exact. Cleared on reset().
        self._target_encode_cache = {}
        # Precompute id -> embedding-NUM maps once (keyed by gen3_data species/move id) for the labeller.
        self._species_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items() if "num" in rec}
        # gen3_typed_hp_belief_v1 — the OPPONENT's move-belief labels use the TRUE TYPED Hidden Power num
        # (355-370), not the typeless 237.
        #
        # This used to fold every HP onto 237 to match the OBSERVED (bare) form. But the belief posterior
        # is now composed into the 16 typed channels before anything reads it, and 237 is driven hard-off
        # — so a 237-keyed label supervised a channel nothing consumes while leaving the typed channels as
        # NEGATIVES in the multi-label BCE. The head was being actively trained toward "this opponent has
        # no Hidden Power of any type", fighting the very composition that is supposed to make HP real.
        #
        # Leak-safety is unchanged: these labels are TRAINING-ONLY Dict keys read by the loss and never by
        # the forward, and the true HP type is exactly the privileged fact `hp_type_label` (v38) already
        # supervises. Keying the move BCE on it merges those two supervisions into one signal on one
        # posterior instead of two heads pulling at the same quantity through different channels. The
        # OBSERVATION still shows the opponent's HP as bare 237 — the model must still guess the type.
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
            if self._emit_known_moves:
                # KNOWN-mode move belief: the revealed mons' FULL privileged movesets, at the revealed
                # slots (so the move head learns each seen mon's still-UNREVEALED moves). Only declared
                # when 'known'/'both' so an 'unknown'-only run keeps the buffer minimal.
                base_obs["known_moves"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE, BELIEF_MOVE_SLOTS), dtype=np.int64)
            if self._emit_belief_target:
                # LATENT-belief target: the fresh per-mon obs encode (107-d, the same per-Pokémon slot
                # width as obs["observation"]) of each hidden mon at its believed slot; PAD slots zeros.
                # float32 (real obs features); only declared when --opp-belief-latent-coef>0.
                base_obs["belief_target_slots"] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(TEAM_SIZE, POKEMON_FULL_DIM), dtype=np.float32)
        if self._emit_spread_labels:
            # SPREAD-belief label (gen3_unified_spread_belief_v1): the TRUE derived stats {atk,def,spa,spd,spe}
            # of each REVEALED opp mon + a per-slot mask (1 = supervised). float32 (real stat VALUES, the same
            # scale the SpreadBelief head outputs + the op consumes). Only declared when --spread-belief-coef>0.
            base_obs["belief_spread"] = spaces.Box(
                low=0.0, high=np.inf, shape=(TEAM_SIZE, N_SPREAD_STATS), dtype=np.float32)
            base_obs["belief_spread_mask"] = spaces.Box(
                low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
            # NATURE/EV labels (gen3_nature_ev_belief_v1) — the generative spread belief's privileged targets,
            # INVERTED from agent2's known derived stats. belief_nature [6] (nature num 0..24) + belief_ev [6,5]
            # (EVs in {atk,def,spa,spd,spe} order) + per-slot masks. Ride the SAME _emit_spread_labels gate; read
            # ONLY by the nature/EV loss (--spread-belief-nature). low=0 keeps the not-scored sentinel in-space.
            base_obs["belief_nature"] = spaces.Box(low=0, high=24, shape=(TEAM_SIZE,), dtype=np.int64)
            base_obs["belief_nature_mask"] = spaces.Box(low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
            base_obs["belief_ev"] = spaces.Box(low=0.0, high=252.0, shape=(TEAM_SIZE, N_SPREAD_STATS), dtype=np.float32)
            base_obs["belief_ev_mask"] = spaces.Box(low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
        if self._emit_hp_type_labels:
            # HP-TYPE-belief label (gen3_opp_hp_type_belief_v1): the TRUE HP type index (0..15) of each
            # REVEALED opp mon that runs Hidden Power + a per-slot mask (1 = supervised). int64 (a class
            # index for CE); low=-1 keeps the PAD / not-scored sentinel in-space. Only declared when
            # --hp-type-belief learned + --hp-type-belief-coef>0.
            base_obs["hp_type_label"] = spaces.Box(
                low=-1, high=N_HP_TYPES_LABEL - 1, shape=(TEAM_SIZE,), dtype=np.int64)
            base_obs["hp_type_mask"] = spaces.Box(
                low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
        if self._emit_win_target:
            # Win-probability MC label + known-mask (placeholders here; back-filled post-collection).
            base_obs["win_target"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            base_obs["win_mask"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            # Normalized material margin ∈ [−1,1] (Φ_mat-derived) — a REAL per-step value (not a
            # placeholder), used by the win-prob loss to stratify P(win) skill by how decided the game
            # is (value lives in close games, |margin|≈0) + a material-baseline skill score.
            base_obs["win_margin"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._emit_defensive_opportunity:
            # gen3_defensive_entropy_v1: 1.0 = a productive defensive move (recovery/cure) is legal this
            # decision. A REAL per-step value; read ONLY by the state-conditioned entropy boost in the PPO loss.
            base_obs["defensive_opportunity"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._emit_pubval_target:
            # gen3_pubval_aux_v1: the frozen human-replay V_pub of the current PUBLIC board (a REAL
            # per-step value) + a validity mask. Read ONLY by the pubval aux loss.
            base_obs["pubval_target"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            base_obs["pubval_mask"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if self._emit_distill_mask:
            # gen3_exploiter_distill_v1: INTEGER team-id (0=none, k=teacher k) of the trainee's current team
            # among the N distillation-teacher teams. Read ONLY by the exploiter-distillation KL in the PPO
            # loss (gates which teacher's advice is on-distribution). A REAL per-step value (const per battle).
            base_obs["distill_mask"] = spaces.Box(
                low=0.0, high=float(len(self._distill_team_species)), shape=(1,), dtype=np.float32)
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
            out = {"belief_species": bs, "belief_moves": bm}
            if self._emit_known_moves:
                out["known_moves"] = zero_known_moves()
            if self._emit_belief_target:
                out["belief_target_slots"] = np.zeros((TEAM_SIZE, POKEMON_FULL_DIM), dtype=np.float32)
            return out
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
        out = {"belief_species": bs, "belief_moves": bm}
        if self._emit_known_moves:
            # Revealed slots are the leading-contiguous block (guarded above); `revealed` is in encoder
            # slot order, so it aligns 1-1 with that block. Each gets its species' FULL privileged moveset.
            out["known_moves"] = build_known_move_labels(
                revealed_species, team_species, team_moves, species_known,
                self._move_num, to_id_str,
            )
        if self._emit_belief_target:
            out["belief_target_slots"] = self._build_belief_target_slots(
                b2, full, team_species, revealed_species, species_known)
        return out

    def _spread_labels(self, obs_vec) -> dict:
        """SPREAD-belief label (gen3_unified_spread_belief_v1) — INDEPENDENT of the species/move belief so
        --spread-belief works standalone. For each REVEALED opp slot, the TRUE derived stats
        {atk,def,spa,spd,spe} of that mon (agent2's OWN team's computed `mon.stats` — the privileged ground
        truth Gen 3 hides from the trainee) + a per-slot mask. All-zero (mask 0) until both battles exist.
        Read ONLY by the spread-belief loss; never enters the policy/value forward."""
        b1 = getattr(self, "battle1", None)
        b2 = getattr(self, "battle2", None)
        if b1 is None or b2 is None:
            sp, spm = zero_spread_labels()
            nat, nmask, ev, evmask = zero_nature_ev_labels()
            return {"belief_spread": sp, "belief_spread_mask": spm, "belief_nature": nat,
                    "belief_nature_mask": nmask, "belief_ev": ev, "belief_ev_mask": evmask}
        # Per-opp-slot species_known straight from the obs the model reads (same source as _belief_labels).
        species_known = [
            float(obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET])
            for i in range(TEAM_SIZE)
        ]
        revealed = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m is not None]
        revealed_species = [m.species for m in revealed]
        # TRUE derived stats per opp species, from agent2's OWN team's computed stats (Gen 3 hides the opp's
        # EVs from the trainee even once the species is revealed). Incomplete stats → species omitted (mask 0).
        species_to_spread = {}
        for m in b2.team.values():
            st = getattr(m, "stats", None) or {}
            vals = [st.get(k) for k in SPREAD_STAT_ORDER]
            if all(v is not None for v in vals):
                species_to_spread[to_id_str(m.species)] = [float(v) for v in vals]
        sp, spm = build_known_spread_labels(revealed_species, species_to_spread, species_known, to_id_str)
        # gen3_nature_ev_belief_v1: the NATURE/EV decomposition (inverted from the same derived stats, cached).
        nat, nmask, ev, evmask = build_known_nature_ev_labels(
            revealed_species, self._nature_ev_map(b2), species_known, to_id_str)
        return {"belief_spread": sp, "belief_spread_mask": spm, "belief_nature": nat,
                "belief_nature_mask": nmask, "belief_ev": ev, "belief_ev_mask": evmask}

    def _nature_ev_map(self, b2) -> dict:
        """``{to_id_str(species) -> (nature_num, [ev×5])}`` for agent2's team, INVERTED from each mon's known
        derived stats + base stats (`damage_tables.invert_nature_evs`, gen3_nature_ev_belief_v1). Cached per
        battle keyed by the team's species set (the team is fixed; the inversion is ~expensive). A mon whose
        stats don't invert to a valid nature/EV spread is omitted (its slot stays mask 0)."""
        key = frozenset(to_id_str(m.species) for m in b2.team.values())
        if key == self._nature_ev_cache_key:
            return self._nature_ev_cache
        out = {}
        for m in b2.team.values():
            st = getattr(m, "stats", None) or {}
            derived = [st.get(k) for k in SPREAD_STAT_ORDER]
            if any(v is None for v in derived):
                continue
            sid = to_id_str(m.species)
            sd = gen3_data.species.get(sid)
            if sd is None:
                continue
            base = [float(sd.base_stats.get(k, 0)) for k in SPREAD_STAT_ORDER]
            res = invert_nature_evs([float(v) for v in derived], base, species_id=sid)
            if res is not None:
                out[sid] = res
        self._nature_ev_cache_key = key
        self._nature_ev_cache = out
        return out

    def _hp_type_labels(self, obs_vec) -> dict:
        """HP-TYPE-belief label (gen3_opp_hp_type_belief_v1) — INDEPENDENT of the other belief legs. For each
        REVEALED opp slot whose species runs a Hidden Power, the TRUE HP type index (0..15) from agent2's OWN
        team's typed move id (Gen 3 NEVER reveals the opp HP type, so this is privileged) + a per-slot mask.
        All-PAD (mask 0) until both battles exist. Read ONLY by the HP-type CE loss; never enters the forward."""
        b1 = getattr(self, "battle1", None)
        b2 = getattr(self, "battle2", None)
        if b1 is None or b2 is None:
            lab, msk = zero_hp_type_labels()
            return {"hp_type_label": lab, "hp_type_mask": msk}
        # Per-opp-slot species_known straight from the obs the model reads (same source as _spread_labels).
        species_known = [
            float(obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET])
            for i in range(TEAM_SIZE)
        ]
        revealed = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m is not None]
        revealed_species = [m.species for m in revealed]
        # TRUE HP type per opp species, from agent2's OWN team's typed move id (e.g. 'hiddenpowerice' → ICE).
        # poke-env keeps the type suffix on an own mon's Move._id, so the type is recoverable here. A species
        # with no Hidden Power is simply absent → that slot stays mask 0.
        species_to_hp_type = {}
        for m in b2.team.values():
            for mv in m.moves.values():
                t = hp_type_idx_from_move_id(mv.id)
                if t is not None:
                    species_to_hp_type[to_id_str(m.species)] = t
                    break
        lab, msk = build_hp_type_labels(revealed_species, species_to_hp_type, species_known, to_id_str)
        return {"hp_type_label": lab, "hp_type_mask": msk}

    def _build_belief_target_slots(self, b2, full, team_species, revealed_species, species_known):
        """Fresh per-mon obs encode [TEAM_SIZE, POKEMON_FULL_DIM] of each hidden mon at its believed
        slot — the SAME canonical assignment as belief_species (both via `assign_hidden_to_slots`), so
        the latent target and the species-CE label can never name a different mon for a slot. Each mon
        is encoded as agent2's OWN mon (full spread + full moveset), untouched (full HP / no status,
        since a still-hidden mon never entered battle) → a clean fresh-identity POKEMON_FULL_DIM slot the
        model's pokemon_encoder turns into the role-token the latent head regresses toward. Cached per species
        (cleared each reset). PAD / non-target slots stay zeros."""
        target = np.zeros((TEAM_SIZE, POKEMON_FULL_DIM), dtype=np.float32)
        idx = build_belief_target_index(
            team_species, revealed_species, species_known, self._species_num, to_id_str)
        for slot in range(TEAM_SIZE):
            team_idx = int(idx[slot])
            if team_idx < 0:
                continue
            target[slot] = self._encode_fresh_mon(full[team_idx], b2)
        return target

    def _encode_fresh_mon(self, mon, b2):
        """One POKEMON_FULL_DIM fresh-identity slot for a hidden mon (is_own=True → spread populated),
        cached by species id. Mirrors `state_encoder`'s per-slot pack: the POKEMON_VECTOR_DIM per-mon
        encode + a trailing active flag (0.0 — a bench/identity encode)."""
        sp = mon.species
        cached = self._target_encode_cache.get(sp)
        if cached is not None:
            return cached
        enc = self.observation_encoder.pokemon_encoder.encode(mon, b2, is_own=True)
        slot = np.zeros(POKEMON_FULL_DIM, dtype=np.float32)
        slot[: len(enc)] = enc   # active flag (last dim) left 0.0
        self._target_encode_cache[sp] = slot
        return slot

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

    def _merge_training_keys(self, agent_obs):
        """Merge TRAINING-ONLY label keys into the trainee obs. Belief labels are real per-step
        privileged info; `win_target`/`win_mask` are PLACEHOLDERS (zeros) the WinProbLabelCallback
        overwrites post-collection with the Monte-Carlo episode outcome (the outcome is a FUTURE
        quantity, so it can't be a real per-step value like the belief labels)."""
        if self._emit_belief_labels:
            agent_obs.update(self._belief_labels(agent_obs["observation"]))
        if self._emit_spread_labels:
            agent_obs.update(self._spread_labels(agent_obs["observation"]))
        if self._emit_hp_type_labels:
            agent_obs.update(self._hp_type_labels(agent_obs["observation"]))
        if self._emit_win_target:
            agent_obs["win_target"] = np.zeros(1, dtype=np.float32)
            agent_obs["win_mask"] = np.zeros(1, dtype=np.float32)
            # The material margin _compute_phi_mat stashed this turn (calc_reward runs before this in
            # step(); 0.0 at reset). A REAL value (present-state), unlike the back-filled win_target.
            agent_obs["win_margin"] = np.array(
                [float(getattr(self.reward_manager, "_last_material_margin", 0.0))], dtype=np.float32)
        if self._emit_defensive_opportunity:
            agent_obs["defensive_opportunity"] = np.array([self._defensive_opportunity()], dtype=np.float32)
        if self._emit_pubval_target:
            v, m = self._pubval_target()
            agent_obs["pubval_target"] = np.array([v], dtype=np.float32)
            agent_obs["pubval_mask"] = np.array([m], dtype=np.float32)
        if self._emit_distill_mask:
            agent_obs["distill_mask"] = np.array([self._distill_mask()], dtype=np.float32)

    def _distill_mask(self) -> float:
        """gen3_exploiter_distill_v1 (N teachers): the INTEGER team-id of the trainee's CURRENT team among
        the distillation teachers' teams — 0.0 = not any teacher's team, k = teacher k (1-indexed). Only
        these states get teacher k's KL. Cached per battle (`battle1.team` is fixed for a battle); 0.0
        (uncached) until the full 6-mon team is known (early reset). Both sides use the SAME
        `to_id_str(species)` id-set (the teacher teams' sets are precomputed in train_rl_agent)."""
        if self._distill_team_id is not None:
            return float(self._distill_team_id)
        b1 = getattr(self, "battle1", None)
        team = getattr(b1, "team", None) if b1 is not None else None
        if not team or len(team) < TEAM_SIZE:
            return 0.0  # team not fully known yet — don't cache a partial set
        cur = frozenset(to_id_str(m.species) for m in team.values())
        self._distill_team_id = 0
        for k, sp_list in enumerate(self._distill_team_species, start=1):
            if cur in sp_list:          # teacher k owns >=1 team; its KL fires on ANY of them
                self._distill_team_id = k
                break
        return float(self._distill_team_id)

    def _pubval_target(self) -> "tuple[float, float]":
        """gen3_pubval_aux_v1: evaluate the FROZEN human-replay public value on the current board →
        ``(V_pub, mask)``. Reads the vetted LiveView (public info only: revealed mons' HP/status,
        hazards, active boosts, weather) and folds each side through the SAME ``PubSide`` /
        ``features()`` the calibration corpus used — parity is structural (guarded end-to-end by
        ``pubval_parity_fuzz_test``). Mask 0 (target 0.5, ignored by the loss) only before the battle
        exists; a genuine computation error RAISES (fail-loud — a silently wrong target would train the
        aux head toward garbage)."""
        b1 = getattr(self, "battle1", None)
        if b1 is None or self._pubval_model is None:
            return 0.5, 0.0
        from agents.training.pubval import features, pub_side_from_live
        live = b1.live_view()
        f = features(pub_side_from_live(live.ours), pub_side_from_live(live.opp),
                     int(live.turn), live.weather.weather)
        return self._pubval_model.predict(f), 1.0

    def _defensive_opportunity(self) -> float:
        """gen3_defensive_entropy_v1: 1.0 if the trainee's ACTIVE mon has a PRODUCTIVE defensive option this
        decision — a legal HP-recovery move (`is_heal`) with the active below `_DEFENSIVE_HEAL_HP`, OR a legal
        self-cure (Refresh) while statused, OR a legal team-cure (Heal Bell/Aromatherapy) while any team member
        is statused — else 0.0. Cheap (a few move lookups); never raises (it rides the per-decision emit path).
        Read ONLY by the entropy boost; never enters the forward. On a forced switch `available_moves` is empty
        → 0.0 (you can't heal when forced to replace)."""
        b1 = getattr(self, "battle1", None)
        if b1 is None:
            return 0.0
        try:
            active = b1.active_pokemon
            moves = b1.available_moves or []
            if active is None or not moves:
                return 0.0
            hp = active.current_hp_fraction
            self_statused = active.status is not None
            team_statused = any(getattr(m, "status", None) is not None for m in b1.team.values())
            for mv in moves:
                md = gen3_data.moves.get(mv.id)
                if md is None:
                    continue
                if md.is_heal and hp is not None and hp < _DEFENSIVE_HEAL_HP:
                    return 1.0
                if md.cures_self_status and self_statused:
                    return 1.0
                if md.cures_team_status and team_statused:
                    return 1.0
        except Exception:
            return 0.0
        return 0.0

    def step(self, action):
        try:
            battle = getattr(self, "_battle", None) or self.battle1
            trainee_idx = action.get(self.agent1.username, -1) if isinstance(action, dict) else action
            if battle is self.battle1 and self._tracker.last_ctx is not None:
                self._tracker.advance(trainee_idx)
                self.reward_manager.record_action(self._tracker.last_ctx, trainee_idx)
            out = super().step(action)
            if (self._emit_belief_labels or self._emit_win_target or self._emit_spread_labels
                    or self._emit_hp_type_labels or self._emit_defensive_opportunity
                    or self._emit_pubval_target or self._emit_distill_mask):
                agent_obs = out[0].get(self.agent1.username)
                if agent_obs is not None:
                    self._merge_training_keys(agent_obs)
            return out
        except Exception as e:
            import traceback
            print(f"ERROR IN STEP: {e}")
            traceback.print_exc()
            raise e

    def reset(self, *args, **kwargs):
        self.reward_manager.report_episode(getattr(self, "battle1", None))
        self._tracker.reset()
        self._target_encode_cache = {}   # new battle → new opponent team → drop the fresh-encode cache
        self._distill_team_id = None     # new battle → recompute which teacher's team (if any) the trainee is on
        try:
            if hasattr(self, "agent1"):
                self.agent1.save_replays = None
            self.reward_manager.reset()
            self._stall_logger.reset()
            out = super().reset(*args, **kwargs)
            if (self._emit_belief_labels or self._emit_win_target or self._emit_spread_labels
                    or self._emit_hp_type_labels or self._emit_defensive_opportunity
                    or self._emit_pubval_target or self._emit_distill_mask):
                obs, info = out
                agent_obs = obs.get(self.agent1.username)
                if agent_obs is not None:
                    self._merge_training_keys(agent_obs)
                return obs, info
            return out
        except Exception as e:
            import traceback
            print(f"ERROR IN RESET: {e}")
            traceback.print_exc()
            raise e
