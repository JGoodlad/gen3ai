"""THE TRAINEE'S SPACES — ``(observation_space, action_space)`` from the run's args, with no env built.

Deletion pass U2 (``designs/ops/deletion_pass_manifest.md`` §6 finding 1): production startup used to
instantiate a ``Gen3Env`` (a poke-env ``SinglesEnv`` that never connects) just to read its two spaces
(``rust_rollout.build.trainee_spaces``), which made the Python env core a production dependency of the
Rust env core. The spaces are a pure function of the resolved args and the observation layout, so they
live here, and ``Gen3Env`` builds ITS space through this module too — one declaration, so the two env
cores cannot disagree on a key, a shape, a dtype or a bound while both exist.

* ``resolved_obs_source`` / ``trainee_env_kwargs`` — the trainee's per-run label switches as a pure
  function of the args (moved here from ``main.train.env_factory``, which re-exports them until it goes).
* ``label_gates`` — the derived per-key gates (``Gen3Env`` reads the same ones at step time).
* ``trainee_observation_space`` — the Dict space, key by key.
* ``trainee_spaces`` — both spaces for a run's args.

Imports no poke-env env, no ``Gen3Env`` / ``wrappers`` / ``env_factory`` / bridge session
(pinned by ``trainee_spaces_test``).
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from agents.model.dense_aux_head import DENSE_AUX_DIM_OUT
from agents.observation.belief_labels import BELIEF_MOVE_SLOTS, N_HP_TYPES_LABEL, N_SPREAD_STATS
from agents.observation.constants import TEAM_SIZE
from agents.observation.true_team import TRUE_TEAM_KEY, TRUE_TEAM_SHAPE
from agents.training.dense_aux import AUX_MASK_KEY, AUX_TARGET_KEY, AUX_TURN_KEY
from agents.training.fork_arm import PG_MASK_KEY as FORK_PG_MASK_KEY

#: The trainee's action space: 4 moves + 5 switches + 2 reserved slots (``Gen3ActionMapper``).
N_ACTIONS = 11


def resolved_obs_source(args: Any) -> str:
    """The trainee's obs source for ``args`` (gen3_core_obs_source_v1): the typed value, else
    ``core`` on the rust bridge (the production default since the M6 cutover) and ``python`` on
    any other transport (the core lives in the rust ``sim_bridge`` child)."""
    typed = getattr(args, "obs_source", None)
    if typed is not None:
        return typed
    return "core" if getattr(args, "use_bridge", "rust") == "rust" else "python"


def trainee_env_kwargs(args: Any) -> dict:
    """The TRAINEE ``Gen3Env``'s per-run keyword arguments — which training-only label keys it
    emits (ARCHITECTURE.md §7) and where its observation row comes from — as a pure function of
    the resolved ``args``, so every builder of a training-shaped env (this factory, the cutover
    stress's slice N) derives the SAME surface from one place."""
    return dict(
        # TRAINING-only privileged belief labels (only the trainee env; the model side
        # gates the BeliefHead on the same coef>0 signal). Eval/self-play opponents play
        # via RLPlayer, not Gen3Env, so they never emit them.
        emit_belief_labels=(args.opp_belief_aux_coef > 0.0),
        move_belief_mode=args.move_belief_mode,
        emit_win_target=(args.win_prob_mode != "none"),
        # gen3_winprob_rollout_weight_v1: the per-row BCE WEIGHT key, declared only when
        # there is something to weigh — the weight above 1.0 AND a rollout fraction above
        # 0.0 (the two are already bound to each other by `combination_checks`, and this
        # is the same predicate spelled where the obs space is decided rather than
        # inferred from one half of it).
        emit_win_row_weight=(
            float(getattr(args, "win_prob_rollout_weight", 1.0) or 1.0) > 1.0
            and float(getattr(args, "win_prob_rollout_target", 0.0) or 0.0) > 0.0),
        # gen3_fork_v1: the per-row POLICY-TERM mask key (`fork_pg_m`), declared only when
        # the arm is on. It is the carrier for THE MASK RULE — the fork step is out of the
        # policy term for every branch — and an injected row is the only row that ever
        # holds anything but the 1.0 placeholder, so an unflagged run's observation space,
        # its rollout buffer and its policy loss are all untouched.
        emit_fork_pg_mask=(float(getattr(args, "fork_fraction", 0.0) or 0.0) > 0.0),
        # PRIVILEGED TRUE-TEAM channel (gen3_value_true_team_v1): emit the opponent's
        # actual party only when the value route that reads it was built. Emitting it
        # unconditionally would put a key in the observation_space that no consumer reads
        # and that every non-local path would then have to fabricate.
        emit_opp_true_team=bool(getattr(args, "value_true_team", False)),
        # DENSE AUXILIARY labels (gen3_dense_aux_v1): emit the 25 end-of-battle targets
        # and their two-part mask only when the head that consumes them was built — the
        # same coef>0 signal `extractor_arch._DERIVED` turns into the `dense_aux` toggle,
        # so the key set and the module cannot disagree.
        emit_dense_aux=(float(getattr(args, "win_prob_dense_aux", 0.0) or 0.0) > 0.0),
        # SPREAD-belief supervision (gen3_unified_spread_belief_v1): emit the privileged
        # true-spread label only when the loss will consume it (coef>0; the CLI guards that
        # --spread-belief-coef requires --spread-belief, so the head is present to supervise).
        emit_spread_labels=(args.spread_belief and args.spread_belief_coef > 0.0),
        emit_opp_intent_labels=(getattr(args, 'opp_intent_coef', 0.0) > 0.0),
        # HP-TYPE-belief supervision (gen3_typed_hp_belief_v1): emit the privileged true-HP-type
        # label only when the CE will consume it (the head itself is unconditional under a move
        # belief; the CLI guards that the coef implies one).
        emit_hp_type_labels=(args.move_belief_mode != "off" and args.hp_belief_mode == "composed"
                             and args.hp_type_belief_coef > 0.0),
        # ITEM-belief supervision (gen3_item_belief_v1): emit the privileged true-item
        # label only when the head exists AND the CE will consume it.
        emit_item_labels=(args.item_belief and args.item_belief_coef > 0.0),
        # DEFENSIVE-exploration flag (gen3_defensive_entropy_v1): emit only when the boost is on, so
        # the state-conditioned entropy term in the PPO loss can read it. Off = no key, no cost.
        emit_defensive_opportunity=(args.defensive_entropy_boost > 1.0),
        # BAIT-exploration flag (gen3_bait_entropy_v1): same gate, same reason — emit only when the
        # boost is on, so the flag costs nothing on every run that is not taking the probe.
        emit_bait_opportunity=(args.bait_entropy_boost > 1.0),
        # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1): the teacher team's species id-set
        # (None unless --distill-coef>0). The env emits `distill_mask`=1 on states where the
        # trainee pilots this team — the only states the distillation KL folds. None → no key.
        distill_team_species=getattr(args, "_distill_species", None),
        # gen3_core_obs_source_v1: the trainee's row from the Rust core (`--obs-source core`)
        # or from the Python encoder (the default). Only the trainee env; opponents unchanged.
        obs_source=resolved_obs_source(args),
    )


class LabelGates(NamedTuple):
    """Which training-only label keys the trainee's observation carries — derived ONCE from the raw
    switches (``trainee_env_kwargs``' ``emit_*`` values), read by the space builder and by ``Gen3Env``."""
    belief_labels: bool
    known_moves: bool
    opp_intent_labels: bool
    win_target: bool
    spread_labels: bool
    hp_type_labels: bool
    item_labels: bool
    win_row_weight: bool
    fork_pg_mask: bool
    opp_true_team: bool
    dense_aux: bool
    defensive_opportunity: bool
    bait_opportunity: bool
    distill_mask: bool


def label_gates(*, emit_belief_labels: bool = False, move_belief_mode: str = "off",
                emit_win_target: bool = False, emit_win_row_weight: bool = False,
                emit_fork_pg_mask: bool = False, emit_opp_true_team: bool = False,
                emit_dense_aux: bool = False, emit_spread_labels: bool = False,
                emit_opp_intent_labels: bool = False, emit_hp_type_labels: bool = False,
                emit_item_labels: bool = False, emit_defensive_opportunity: bool = False,
                emit_bait_opportunity: bool = False,
                distill_team_species: Optional[Sequence[Any]] = None) -> LabelGates:
    """The per-key gates. Two are DERIVED: the move belief needs the belief labels whatever
    ``--opp-belief-aux-coef`` says, and the row weight exists only beside a win target."""
    return LabelGates(
        belief_labels=bool(emit_belief_labels or move_belief_mode != "off"),
        known_moves=move_belief_mode in ("revealed", "both"),
        opp_intent_labels=bool(emit_opp_intent_labels),
        win_target=bool(emit_win_target),
        spread_labels=bool(emit_spread_labels),
        hp_type_labels=bool(emit_hp_type_labels),
        item_labels=bool(emit_item_labels),
        win_row_weight=bool(emit_win_row_weight and emit_win_target),
        fork_pg_mask=bool(emit_fork_pg_mask),
        opp_true_team=bool(emit_opp_true_team),
        dense_aux=bool(emit_dense_aux),
        defensive_opportunity=bool(emit_defensive_opportunity),
        bait_opportunity=bool(emit_bait_opportunity),
        distill_mask=bool(distill_team_species),
    )


def trainee_observation_space(layout: dict, vector_space: Any, g: LabelGates,
                              distill_species: Sequence[Any] = ()) -> spaces.Dict:
    """The trainee's ``spaces.Dict``: the flat observation + the action mask + every label key ``g``
    switches on. ``layout`` is the encoder's ``get_layout()``,
    ``vector_space`` its schema's gym space, ``distill_species`` one entry per distillation teacher."""
    base_obs = {
        "observation": vector_space,
        "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8),
    }
    # The int64 labels' upper bound. Hoisted out of the belief block: the intent keys read it
    # too, and an intent-only env (no belief labels) raised UnboundLocalError at construction
    # (found by `rust_env_label_inventory_test`, M5 Lane C, 2026-09-29).
    _imax = np.iinfo(np.int64).max
    if g.belief_labels:
        # low=-1 keeps the PAD / not-scored sentinel in-space; Box(int64) (NOT Discrete, which
        # rejects -1 and the rollout buffer special-cases it).
        base_obs["belief_species"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE,), dtype=np.int64)
        base_obs["belief_moves"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE, BELIEF_MOVE_SLOTS), dtype=np.int64)
        if g.known_moves:
            # KNOWN-mode move belief: the revealed mons' FULL privileged movesets, at the revealed
            # slots (so the move head learns each seen mon's still-UNREVEALED moves). Only declared
            # when 'known'/'both' so an 'unknown'-only run keeps the buffer minimal.
            base_obs["known_moves"] = spaces.Box(low=-1, high=_imax, shape=(TEAM_SIZE, BELIEF_MOVE_SLOTS), dtype=np.int64)
    if g.opp_intent_labels:
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
    # gen3_opp_class_v1 — WHICH KIND of opponent this episode faces (bot / pool / stable /
    # exploiter). Declared for TWO consumers, which is why it is not inside either gate:
    #   * the opponent-intent losses, which it SPLITS (one pooled intent accuracy over random
    #     bots, heuristics and frozen selves cannot be read — see `_select_episode_opponent`);
    #   * the training-side value sidecar (`gen3_value_sidecar_v1`), whose by-opponent-class
    #     calibration slice is otherwise EMPTY on the exact runs it exists for. A win-prob arm
    #     normally runs with no intent loss at all, so gating this key on the intent labels made
    #     the sidecar's opponent slice unreachable in practice.
    # It is a LABEL key: the network never reads it, so widening the gate cannot change a
    # forward pass. `train()`'s one-ahead intent SHIFT stays gated on `opp_intent_coef > 0` and
    # runs AFTER every callback's `_on_rollout_end`, so the sidecar reads the env's own
    # per-episode value, unshifted, either way.
    if g.opp_intent_labels or g.win_target:
        base_obs["opp_class"] = spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64)
    if g.spread_labels:
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
    if g.hp_type_labels:
        # HP-TYPE-belief label (gen3_opp_hp_type_belief_v1): the TRUE HP type index (0..15) of each
        # REVEALED opp mon that runs Hidden Power + a per-slot mask (1 = supervised). int64 (a class
        # index for CE); low=-1 keeps the PAD / not-scored sentinel in-space. Only declared when
        # --hp-type-belief learned + --hp-type-belief-coef>0.
        base_obs["hp_type_label"] = spaces.Box(
            low=-1, high=N_HP_TYPES_LABEL - 1, shape=(TEAM_SIZE,), dtype=np.int64)
        base_obs["hp_type_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
    if g.item_labels:
        # ITEM-belief label (gen3_item_belief_v1): the TRUE item NUM of each REVEALED opp mon +
        # a per-slot mask (1 = supervised). int64 class index for CE over the item-num axis
        # (num 0 = "nothing" IS a class); low=-1 keeps the PAD sentinel in-space. The high bound
        # is the encoder's item axis (`max_items`), the same axis ItemBelief's logits span.
        base_obs["item_label"] = spaces.Box(
            low=-1, high=layout["max_items"] - 1,
            shape=(TEAM_SIZE,), dtype=np.int64)
        base_obs["item_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(TEAM_SIZE,), dtype=np.float32)
    if g.win_target:
        # Win-probability MC label + known-mask (placeholders here; back-filled post-collection).
        base_obs["win_target"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        base_obs["win_mask"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # Normalized material margin ∈ [−1,1] (Φ_mat-derived) — a REAL per-step value (not a
        # placeholder), used by the win-prob loss to stratify P(win) skill by how decided the game
        # is (value lives in close games, |margin|≈0) + a material-baseline skill score.
        base_obs["win_margin"] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    if g.win_row_weight:
        # gen3_winprob_rollout_weight_v1: the per-row BCE weight. A PLACEHOLDER of 1.0 (not 0.0
        # like the two above) because it is a MULTIPLIER, not a label — a path that somehow
        # reached the loss before `WinProbLabelCallback._on_rollout_end` overwrote it would
        # zero the entire win-prob term on a zero placeholder, and read as a dead head rather
        # than as a plumbing break. `high` is unbounded because the ceiling is the flag's own
        # value divided by a normaliser, and coupling the space to the flag would make an obs
        # space that a resume at a different weight could not reload.
        base_obs["win_row_w"] = spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
    if g.fork_pg_mask:
        # gen3_fork_v1: 1.0 = this row is in the clipped policy term, 0.0 = it is not.
        # Bounded in [0, 1] because it is a MASK and not a dose — the fork arm does not
        # re-weight the policy gradient, it excludes exactly the fork step (see
        # `agents.training.fork_buffer`, THE MASK RULE).
        base_obs[FORK_PG_MASK_KEY] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    if g.opp_true_team:
        # gen3_value_true_team_v1: the opponent's TRUE party in the obs's own per-mon layout.
        # The bounds are the per-mon block's own: it carries embedding NUMS (up to the species
        # axis) alongside normalised scalars, exactly like the opp-team slice of the flat
        # vector, so the Box is bounded by the widest of those axes rather than by 1.0.
        base_obs[TRUE_TEAM_KEY] = spaces.Box(
            low=0.0, high=float(max(layout["max_species"],
                                    layout["max_moves"])),
            shape=TRUE_TEAM_SHAPE, dtype=np.float32)
    if g.dense_aux:
        # gen3_dense_aux_v1: the DENSE AUXILIARY targets + their two-part mask. `aux_target` is
        # a placeholder (back-filled post-collection); `aux_mask` and `aux_turn` are REAL
        # present-state values. All three are LABEL keys — the network never reads them.
        base_obs[AUX_TARGET_KEY] = spaces.Box(
            low=0.0, high=1.0, shape=(DENSE_AUX_DIM_OUT,), dtype=np.float32)
        base_obs[AUX_MASK_KEY] = spaces.Box(
            low=0.0, high=1.0, shape=(DENSE_AUX_DIM_OUT,), dtype=np.float32)
        base_obs[AUX_TURN_KEY] = spaces.Box(
            low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
    if g.defensive_opportunity:
        # gen3_defensive_entropy_v1: 1.0 = a productive defensive move (recovery/cure) is legal this
        # decision. A REAL per-step value; read ONLY by the state-conditioned entropy boost in the PPO loss.
        base_obs["defensive_opportunity"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    if g.bait_opportunity:
        # gen3_bait_entropy_v1: 1.0 = a revealed, alive opponent BENCH mon is immune to the attack we
        # are most likely to click. A REAL per-step value; read ONLY by the bait entropy boost.
        base_obs["bait_opportunity"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    if g.distill_mask:
        # gen3_exploiter_distill_v1: INTEGER team-id (0=none, k=teacher k) of the trainee's current team
        # among the N distillation-teacher teams. Read ONLY by the exploiter-distillation KL in the PPO
        # loss (gates which teacher's advice is on-distribution). A REAL per-step value (const per battle).
        base_obs["distill_mask"] = spaces.Box(
            low=0.0, high=float(len(distill_species)), shape=(1,), dtype=np.float32)

    return spaces.Dict(base_obs)


def trainee_spaces(args: Any, mappings: Any = None) -> Tuple[spaces.Dict, spaces.Discrete]:
    """``(observation_space, action_space)`` of the trainee ``args`` declare — what the Rust env core's
    ``RustVecEnv`` and the learner build against, and what a checkpoint records."""
    from agents.observation.schema import build_schema
    from agents.observation.state_encoder import get_observation_encoder, load_mappings

    layout = get_observation_encoder(mappings if mappings is not None else load_mappings()).get_layout()
    kw = trainee_env_kwargs(args)
    kw.pop("obs_source", None)
    species = kw.get("distill_team_species") or ()
    obs = trainee_observation_space(layout, build_schema(layout).gym_space(), label_gates(**kw),
                                    distill_species=list(species))
    return obs, spaces.Discrete(N_ACTIONS)
