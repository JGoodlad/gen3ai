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
    build_belief_labels, build_known_move_labels,
    zero_belief_labels, zero_known_moves, BELIEF_MOVE_SLOTS,
    build_known_spread_labels, zero_spread_labels, SPREAD_STAT_ORDER, N_SPREAD_STATS,
    build_known_nature_ev_labels, zero_nature_ev_labels,
    build_hp_type_labels, zero_hp_type_labels, hp_type_idx_from_move_id, N_HP_TYPES_LABEL,
    build_item_labels, zero_item_labels,
)
from agents.model.damage_tables import invert_nature_evs, _hp_typed_nums, HIDDEN_POWER_NUM
from agents import gen3_data
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.battle.live_view import LegalActions
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_function import RewardFunction
from agents.training.episode_tracker import EpisodeTracker
from agents.training.stall import StallConfig, StallLogger
from agents.battle.gen3_battle import Gen3Battle
# gen3_bait_entropy_v1: ONE zero-damage predicate, shared with the scripted BaitBot opponent, so the
# training flag fires on exactly the boards BaitBot exploits (and both resolve the type chart + the
# gen-3 ability immunities from `data/` via `effective_multiplier` — never a hand-copied table).
from agents.baitbot import blocks as _blocks_zero_damage
from utils.logging.levels import LogLevel


# gen3_defensive_entropy_v1: the active mon counts as having a productive HP-recovery opportunity (for the
# defensive-exploration entropy boost) only when it has taken at least this much chip — below the threshold a
# heal restores too little to be worth exploring (a Wish cast at full HP for a teammate is the accepted miss).
_DEFENSIVE_HEAL_HP = 0.85


def _bait_candidate_attack(active, moves):
    """gen3_bait_entropy_v1: the damaging move this decision is most likely to spend, or None.

    The RE-CLICK is the sharpest form of the pathology (32% of gen-15 whiffs re-took a decision the
    board had already answered), so a still-legal `last_move` that deals damage wins; otherwise the
    highest-base-power legal attack stands in for "the attack we would click". A proxy for the argmax
    on purpose — the env has no policy to ask (see `Gen3Env._bait_opportunity`). Status moves are never
    candidates: firing one into an immune arrival is a different, much cheaper error (the BaitBot
    predicate makes the same cut)."""
    attacks = [m for m in moves if m.base_power and m.base_power > 0]
    if not attacks:
        return None
    last = getattr(active, "last_move", None)
    if last is not None and getattr(last, "base_power", None):
        for m in attacks:
            if m.id == last.id:
                return m
    return max(attacks, key=lambda m: m.base_power)


# gen3_frame_deletion_v1: the deepest context read left is `build_delta`'s `_history[-2]`.
# 1 gives that (the tracker keeps cap+1 contexts) with no slack for a deleted feature.
_TRACKER_HISTORY_CAP = 1


class Gen3Env(SinglesEnv):
    def __init__(self, mappings, reward_fn: Optional[RewardFunction] = None,
                 log_level=LogLevel.QUIET, stall_config: Optional[StallConfig] = None,
                 *args, battle_class=Gen3Battle, emit_belief_labels: bool = False,
                 move_belief_mode: str = "off",
                 emit_win_target: bool = False, emit_spread_labels: bool = False,
                 emit_opp_intent_labels: bool = False,
                 emit_hp_type_labels: bool = False, emit_item_labels: bool = False,
                 emit_defensive_opportunity: bool = False,
                 emit_bait_opportunity: bool = False,
                 distill_team_species=None,
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
        # SPREAD-belief label key (TRAINING-ONLY, gen3_unified_spread_belief_v1): when on, the obs Dict
        # carries `belief_spread` [6,5] (the TRUE derived stats {atk,def,spa,spd,spe} of each REVEALED opp
        # mon, from agent2's own team) + `belief_spread_mask` [6]. Consumed ONLY by the spread-belief loss
        # (instrumented_ppo._spread_belief_loss); the SpreadBelief head learns the opponent's hidden EV
        # spread instead of sitting at the usage-mean prior, so the DamageOperator prices damage against the
        # opponent's REAL bulk/offense/speed. Enabled by --spread-belief-coef>0 (requires --spread-belief).
        # Read ONLY by the loss — never enters the policy/value forward (the believed stats the op consumes
        # are the model's own prediction, not this label).
        self._emit_spread_labels = emit_spread_labels
        # OPPONENT-INTENT labels (TRAINING-ONLY, gen3_opp_intent_v1): what the opponent ACTUALLY did.
        # UNLIKE every other label here, this one describes the PREVIOUS decision — the delta folded at
        # embed time closes the window that just ended, so the opponent's action at decision t is only
        # observable when building the obs for t+1. The one-row shift back onto the predictions happens
        # in the PPO loss (`align_labels_to_predictions`), NOT here; this emits what the delta says and
        # nothing more. Enabled by --opp-intent-coef>0. Read ONLY by the loss.
        self._emit_opp_intent_labels = emit_opp_intent_labels
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
        # ITEM-belief label key (TRAINING-ONLY, gen3_item_belief_v1): when on, the obs Dict carries
        # `item_label` [6] (the TRUE item NUM of each revealed opp mon, from agent2's own team —
        # privileged, Gen 3 never reveals a Choice Band) + `item_mask` [6]. Consumed ONLY by the
        # BeliefBank's item CE row; never enters the forward. On iff --item-belief + coef>0.
        self._emit_item_labels = emit_item_labels
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
        # BAIT-EXPLORATION flag (TRAINING-ONLY, gen3_bait_entropy_v1): when on, the obs Dict carries
        # `bait_opportunity` [1] = 1.0 on decisions where the attack we are most likely to click is
        # ZERO-damage against an alive, revealed opponent BENCH mon — i.e. the board the bait loop is
        # fired from (they pivot that mon in, our attack does nothing, and gen-15 measured us
        # re-clicking it at p≈0.96). Read ONLY by the state-conditioned entropy boost in the PPO loss
        # (the sampling-side test of the "exploration starvation at a saturated action" mechanism);
        # never enters the policy/value forward. Enabled by --bait-entropy-boost > 1.0.
        self._emit_bait_opportunity = emit_bait_opportunity
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
        if self._emit_opp_intent_labels:
            # OPPONENT-INTENT labels (gen3_opp_intent_v1) — what they DID at the PREVIOUS decision.
            # Three int64 scalars, not a seat index: the seats are `w.topk(K)` built by the MODEL
            # mid-forward and they PERMUTE every turn, so the env cannot name them. It emits the
            # canonical move NUM and the loss locates it among the seats (`match_seats_to_move_num`).
            # An index-based label would silently point at a different move whenever the belief re-sorted.
            base_obs["opp_action_kind"] = spaces.Box(low=0, high=2, shape=(1,), dtype=np.int64)
            base_obs["opp_action_num"] = spaces.Box(low=0, high=_imax, shape=(1,), dtype=np.int64)
            # beta's target: which of THEIR team slots came in (SWITCH_SLOT_NONE = masked).
            base_obs["opp_switch_slot"] = spaces.Box(low=-100, high=TEAM_SIZE, shape=(1,), dtype=np.int64)
            # The CONTENT-ADDRESSED key for a still-HIDDEN switch-in: its species num, resolved at
            # loss time against the model's own believed-slot posterior (there is no valid slot
            # index for an anonymous query — see opp_intent_labels).
            base_obs["opp_switch_species"] = spaces.Box(low=0, high=_imax, shape=(1,), dtype=np.int64)
            base_obs["opp_class"] = spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64)
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
        if self._emit_item_labels:
            # ITEM-belief label (gen3_item_belief_v1): the TRUE item NUM of each REVEALED opp mon +
            # a per-slot mask (1 = supervised). int64 class index for CE over the item-num axis
            # (num 0 = "nothing" IS a class); low=-1 keeps the PAD sentinel in-space. The high bound
            # is the encoder's item axis (`max_items`), the same axis ItemBelief's logits span.
            base_obs["item_label"] = spaces.Box(
                low=-1, high=self.observation_encoder.get_layout()["max_items"] - 1,
                shape=(TEAM_SIZE,), dtype=np.int64)
            base_obs["item_mask"] = spaces.Box(
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
        if self._emit_bait_opportunity:
            # gen3_bait_entropy_v1: 1.0 = a revealed, alive opponent BENCH mon is immune to the attack we
            # are most likely to click. A REAL per-step value; read ONLY by the bait entropy boost.
            base_obs["bait_opportunity"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
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
        # gen3_frame_deletion_v1: the lag frames are gone, so the tracker only needs the
        # one-step window `build_delta` folds over (see EpisodeTracker.__init__).
        self._tracker = EpisodeTracker(history_cap=_TRACKER_HISTORY_CAP)
        # Share ONE ProgressClock between obs and reward (design §5.1): the tracker owns it (updated
        # at embed time so the obs is fresh); the reward READS its stashed last_penalty. Set the
        # clock's per-run penalty magnitude once from the reward config (single source of truth).
        if hasattr(self.reward_manager, "progress_clock"):
            self.reward_manager.progress_clock = self._tracker.progress_clock
            cfg = getattr(self.reward_manager, "config", None)
            if cfg is not None:
                self._tracker.progress_clock.no_progress_penalty = cfg.no_progress_penalty
        self._pending_delta = None   # delta folded once at embed time, reused by calc_reward
        self._opp_slot_map = {}      # species -> opp slot as of the CURRENT decision
        # ...and the PREVIOUS decision's copy, which is the one beta's label must read. The label
        # for decision t is built during step() for t+1, and embed_battle(t+1) has ALREADY
        # refreshed the current map by then — so reading it would describe the board AFTER the
        # switch, where the switch-in is revealed and its believed slot no longer exists.
        # Measured: `opp_intent/beta_wanted_content` read 0.0 (no row even asked for
        # content-addressing) until this one-step delay was added. Same timing trap as the delta.
        self._opp_slot_map_prev = {}
        self._intent_delta = None    # same delta, NOT consumed — the alpha/beta label reads this

    def embed_battle(self, battle):
        # Record FIRST so the tracker's HP-candidate state reflects the just-fired
        # HP (if any) before we encode the obs. The observation at turn N then
        # carries the narrowing from turns 1..N-1.
        legal = None
        if battle is self.battle1:
            # Clear any prior cached delta so a non-recording embed (terminal / no legal action)
            # forces calc_reward to re-fold rather than reuse a stale window.
            self._pending_delta = None
            # gen3_opp_intent_v1: a SEPARATE, non-consumed copy for the alpha/beta label. The
            # reward path CONSUMES `_pending_delta` (calc_reward nulls it), and calc_reward runs
            # INSIDE super().step() — i.e. BEFORE the label keys are merged — so reading the
            # pending slot there always found None and every row was masked. Measured: a smoke
            # reported alpha_mask_rate 1.0 / n_supervised 0, the exact "the head logs a loss and
            # learns nothing" shape. This slot is written here and read there; nothing consumes it.
            self._intent_delta = None
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
                self._intent_delta = self._pending_delta

        if battle is self.battle1:
            obs = self.observation_encoder.encode(
                battle, hp_tracker=self._tracker.hidden_power_tracker, legal=legal,
                progress_clock=self._tracker.progress_clock,
                recency=self._tracker.recency,
                pair_history=self._tracker.pair_history,
                event_window=self._tracker.event_window,
                # gen3_obs_assembler_v1: the incremental obs cache, owned by the tracker (so it
                # is reset with the episode and deep-copied with a counterfactual arm). Only the
                # TRAINEE's battle threads it — battle2's encode below is a plain full rebuild,
                # which is also why the two can never share cache state.
                assembler=self._tracker.obs_assembler(self.observation_encoder.dimension),
            )
            if self._emit_opp_intent_labels:
                # beta's coordinate frame, snapshotted from the SAME obs vector the model reads
                # (so `species_known` here is byte-identical to what BeliefSlots keyed on).
                self._snapshot_opp_slot_map(obs)
        else:
            obs = self.observation_encoder.encode(battle)

        # gen3_frame_deletion_v1: the obs IS the encoder's output. The prev-turn action mask and
        # the N-turn TurnDelta lag frames used to be concatenated here; both are deleted, and the
        # H-B event window (built inside `encode`) is what carries "what happened" now.
        return obs

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

    def _snapshot_opp_slot_map(self, obs_vec) -> None:
        """Cache `species -> opp slot` AS OF THIS DECISION — beta's target coordinate frame.

        `β` predicts at decision t and the label is only observable at t+1, and the encoder packs
        opp slots REVEALED-FIRST, so a mon that switches in is in a DIFFERENT slot at t+1 than the
        one it occupied at t. Resolving the target on the t+1 board therefore names the wrong slot
        (this is what produced `beta_loss = inf` — the target landed on a masked index).

        For a REVEALED mon the slot is its encoder position. For a still-HIDDEN mon it is its
        BELIEVED slot from `assign_hidden_to_slots` — the same canonical assignment the species
        head and the latent target are built from, so `β` pointing at slot j and the species head
        naming slot j always refer to the same mon. That is what lets `β` learn a switch to a mon
        we had not yet seen: the target is the SLOT (privileged, exact), independent of whether the
        species prediction for that slot happens to be right.
        """
        b1 = getattr(self, "battle1", None)
        b2 = getattr(self, "battle2", None)
        if b1 is None or b2 is None:
            self._opp_slot_map_prev = self._opp_slot_map
            self._opp_slot_map = {}
            return
        try:
            revealed = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True)
                        if m is not None]
            slot_map = {}
            for i, m in enumerate(revealed):
                slot_map[to_id_str(m.species)] = i
            # REVEALED mons ONLY. A hidden mon is deliberately absent, so the label emits
            # SWITCH_SLOT_NONE and the loss resolves it by CONTENT against the model's own
            # believed-slot posterior. Adding hidden mons here via `assign_hidden_to_slots` would
            # hand back its Pokedex-sorted index — the exact index-based target content-addressing
            # exists to replace — and the content path would never execute. It did exactly that:
            # `opp_intent/beta_believed_targets` read 0.0 until this loop was removed.
            self._opp_slot_map_prev = self._opp_slot_map
            self._opp_slot_map = slot_map
        except Exception:
            # Never break the obs path for a label; an empty map just masks this decision.
            self._opp_slot_map_prev = self._opp_slot_map
            self._opp_slot_map = {}

    def _opp_intent_labels(self) -> dict:
        """OPPONENT-INTENT label (gen3_opp_intent_v1) — what they did at the PREVIOUS decision.

        Sourced from `self._pending_delta`, the delta `embed_battle` already folded for the
        ProgressClock (same window, no second fold). That delta closes the window that just ENDED,
        so it reports decision t-1 while we are building the obs for t — the consumer shifts.

        `β`'s slot is resolved against the OPPONENT's team as the model sees it (encoder slot order),
        so the target indexes the same six tokens `BetaSwitchHead` points at. An unresolvable
        species is masked rather than guessed: a pointer cannot address what is not there.
        """
        from agents.training.opp_intent_labels import build_opp_intent_label, zero_opp_intent_label
        delta = getattr(self, "_intent_delta", None)
        if delta is None:
            kind, num, slot, sp = zero_opp_intent_label()
        else:
            def _slot_of(species):
                # The CACHED map from decision t — NOT the live board, which has already re-packed
                # now that the switch-in is revealed. See _snapshot_opp_slot_map.
                return getattr(self, "_opp_slot_map_prev", {}).get(
                    to_id_str(species) if species else "")
            kind, num, slot, sp = build_opp_intent_label(
                delta, self._intent_move_num_resolver(delta), _slot_of,
                lambda species: self._species_num.get(to_id_str(species)) if species else None)
        return {"opp_action_kind": np.array([kind], dtype=np.int64),
                "opp_action_num": np.array([num], dtype=np.int64),
                "opp_switch_slot": np.array([slot], dtype=np.int64),
                "opp_switch_species": np.array([sp], dtype=np.int64),
                # gen3_opp_class_v1: WHICH KIND of opponent produced this label (bot / pool /
                # stable / exploiter). Set by the wrapper at episode reset; 0 when unwrapped, which
                # is the honest default since a bare env has no opponent rotation to report.
                "opp_class": np.array([getattr(self, "_opponent_class", 0)], dtype=np.int64)}

    def _intent_move_num_resolver(self, delta):
        """`move_id -> num` for the intent label, with the opponent's Hidden Power resolved to its
        TYPED num (`gen3_typed_hidden_power_ids_v1`, 355-370) instead of the bare typeless 237.

        WHY THIS IS NOT OPTIONAL. The alpha seats come from `refine_candidates`, which masks the bare
        237 out of the candidate axis entirely (`HP_CAND_MASK[237] = 0.0`) because 237 is the
        PRESENCE channel at BP 0 — the op reasons only over the typed nums. The label, meanwhile, is
        whatever poke-env saw the opponent click, and gen 3 NEVER reveals an opponent's HP type, so
        it arrives as bare `hiddenpower` -> 237. `match_seats_to_move_num` compares raw ints, so a
        bare 237 label can match NO seat, ever. That silently deletes every opponent Hidden Power
        from alpha's supervision — 43.2% of `data/teams/` mons carry one — and deletes precisely the
        surprise-coverage move alpha exists to anticipate. It also shows up as a higher
        `alpha_mask_rate` attributable to nothing.

        The type comes from the PRIVILEGED source, not from our own belief: agent2's own team keeps
        the type suffix on `Move._id` (`hiddenpowerice`), exactly as `_hp_type_labels` reads it.
        Typing the label off `HPTypeBelief`'s argmax instead would supervise alpha against the
        model's own guess — circular, and it would bake this belief's 9% error straight into the
        target. A label must be ground truth or absent.

        Falls back to the raw num when the true type cannot be recovered (no b2 yet, or the attacker
        genuinely runs no HP), so an unresolvable case stays masked rather than becoming a wrong seat.
        """
        raw = lambda mid: self._move_num.get(to_id_str(mid))
        b2 = getattr(self, "battle2", None)
        if b2 is None:
            return raw
        attacker = getattr(delta, "opp_prev_active", None)   # the mon that MADE this decision
        if not attacker:
            return raw

        def _resolve(mid):
            num = raw(mid)
            if num != HIDDEN_POWER_NUM:
                return num
            key = to_id_str(attacker)
            for m in b2.team.values():
                if to_id_str(m.species) != key:
                    continue
                for mv in m.moves.values():
                    t = hp_type_idx_from_move_id(mv.id)
                    if t is not None:
                        return int(_hp_typed_nums()[t])
                break
            return num                                        # no HP found → leave bare (stays masked)

        return _resolve

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

    def _item_labels(self, obs_vec) -> dict:
        """ITEM-belief label (gen3_item_belief_v1) — the _hp_type_labels shape over the item axis.
        For each REVEALED opp slot, the TRUE item NUM from agent2's OWN team (matched by species —
        privileged: Gen 3 reveals an item only when it acts, and NEVER a Choice Band). A mon holding
        nothing labels num 0 ("nothing" is a class, not PAD). Read ONLY by the item CE loss."""
        b1 = getattr(self, "battle1", None)
        b2 = getattr(self, "battle2", None)
        if b1 is None or b2 is None:
            lab, msk = zero_item_labels()
            return {"item_label": lab, "item_mask": msk}
        species_known = [
            float(obs_vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET])
            for i in range(TEAM_SIZE)
        ]
        revealed = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m is not None]
        revealed_species = [m.species for m in revealed]
        # TRUE item num per opp species from agent2's own team. poke-env keeps an own mon's `item`
        # as the id string ("leftovers", "" / None when empty); num 0 is the "nothing" class.
        from agents import gen3_data
        species_to_item_num = {}
        for m in b2.team.values():
            item_id = getattr(m, "item", None)
            if item_id:
                data = gen3_data.items.get(to_id_str(item_id))
                if data is not None:
                    species_to_item_num[to_id_str(m.species)] = int(data.num)
            else:
                species_to_item_num[to_id_str(m.species)] = 0
        lab, msk = build_item_labels(revealed_species, species_to_item_num, species_known, to_id_str)
        return {"item_label": lab, "item_mask": msk}

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
        if self._emit_item_labels:
            agent_obs.update(self._item_labels(agent_obs["observation"]))
        if self._emit_opp_intent_labels:
            agent_obs.update(self._opp_intent_labels())
        if self._emit_win_target:
            agent_obs["win_target"] = np.zeros(1, dtype=np.float32)
            agent_obs["win_mask"] = np.zeros(1, dtype=np.float32)
            # The material margin _compute_phi_mat stashed this turn (calc_reward runs before this in
            # step(); 0.0 at reset). A REAL value (present-state), unlike the back-filled win_target.
            agent_obs["win_margin"] = np.array(
                [float(getattr(self.reward_manager, "_last_material_margin", 0.0))], dtype=np.float32)
        if self._emit_defensive_opportunity:
            agent_obs["defensive_opportunity"] = np.array([self._defensive_opportunity()], dtype=np.float32)
        if self._emit_bait_opportunity:
            agent_obs["bait_opportunity"] = np.array([self._bait_opportunity()], dtype=np.float32)
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

    def _bait_opportunity(self) -> float:
        """gen3_bait_entropy_v1: 1.0 if the attack this decision is most likely to spend does ZERO damage
        to an alive, REVEALED opponent BENCH mon — the board a bait loop is fired FROM.

        The pathology (ledger 2026-08-19 / `designs/research_state/bait_loop_hunt.md`): the opponent
        voluntarily pivots a mon our attack cannot touch and we fire anyway at p≈0.96. In gen 3 the switch
        resolves first, so the decision that whiffs is taken while the immune mon is still on their BENCH —
        which is why this is a bench predicate, not an active one, and why it lines up with the offline
        detector's whiff states (`main.prober.loops.bait_events`).

        SCOPE, deliberately (see `training/CLAUDE.md` → bait-exploration entropy):
          * REVEALED bench only. `opponent_team` holds the mons we have seen; an unrevealed arrival is a
            real bait the flag cannot call. Using agent2's true team was available (this key is privileged
            and never enters the forward) and REFUSED: boosting entropy on a distinction the policy cannot
            make adds sampling noise with no learnable signal, and gen-15 settled that perception is not
            the gap.
          * ABILITY immunities count once the ability is revealed (`effective_multiplier` reads `mon.ability`
            and poke-env leaves it unset until then) — the same information the policy holds. TYPE immunity
            (Earthquake into a bench Salamence, the canonical loop) always counts.
          * The α half of the proposed predicate is NOT here: α is published by the extractor inside the
            LEARNER's forward, and this runs in the env worker before any forward exists (the eval-time
            capture reads it off an in-process `RLPlayer`, a seam training does not have). Documented as
            v1 rather than approximated.

        Cheap (one effectiveness lookup per revealed bench mon, memoized in `gen3_mechanics`); never
        raises (it rides the per-decision emit path). Forced switch (`available_moves` empty) → 0.0.
        Read ONLY by the entropy boost; never enters the forward."""
        b1 = getattr(self, "battle1", None)
        if b1 is None:
            return 0.0
        try:
            active = b1.active_pokemon
            moves = b1.available_moves or []
            if active is None or not moves:
                return 0.0
            candidate = _bait_candidate_attack(active, moves)
            if candidate is None:
                return 0.0
            opp_active = b1.opponent_active_pokemon
            for mon in (b1.opponent_team or {}).values():
                if mon is opp_active or mon.active or mon.fainted:
                    continue
                if _blocks_zero_damage(candidate, mon):
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
                    or self._emit_hp_type_labels or self._emit_item_labels
                    or self._emit_defensive_opportunity
                    or self._emit_bait_opportunity
                    or self._emit_distill_mask
                    or self._emit_opp_intent_labels):
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
        self._distill_team_id = None     # new battle → recompute which teacher's team (if any) the trainee is on
        try:
            if hasattr(self, "agent1"):
                self.agent1.save_replays = None
            self.reward_manager.reset()
            self._stall_logger.reset()
            out = super().reset(*args, **kwargs)
            if (self._emit_belief_labels or self._emit_win_target or self._emit_spread_labels
                    or self._emit_hp_type_labels or self._emit_item_labels
                    or self._emit_defensive_opportunity
                    or self._emit_bait_opportunity
                    or self._emit_distill_mask
                    or self._emit_opp_intent_labels):
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
