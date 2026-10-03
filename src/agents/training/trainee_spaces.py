"""THE TRAINEE'S SPACES — ``(observation_space, action_space)`` from the run's args, with no env built.

Deletion pass U2 (``designs/ops/deletion_pass_manifest.md`` §6 finding 1): production startup used to
instantiate a ``Gen3Env`` (a poke-env ``SinglesEnv`` that never connects) just to read its two spaces
(``rust_rollout.build.trainee_spaces``), which made the Python env core a production dependency of the
Rust env core. The spaces are a pure function of the resolved args and the observation layout, so they
live here — and, since the Python core was deleted (U3), this is the ONE declaration of them.

* ``trainee_env_kwargs`` — the trainee's per-run label switches as a pure function of the args.
* ``label_gates`` — the derived per-key gates (``Gen3Env`` reads the same ones at step time).
* ``trainee_observation_space`` — the Dict space, key by key.
* ``trainee_spaces`` — both spaces for a run's args.

Imports no poke-env env (the deleted Python core's modules cannot come back unseen: ``trainee_spaces_test``).
"""
from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import numpy as np
from gymnasium import spaces

from agents.observation.belief_labels import BELIEF_MOVE_SLOTS, N_HP_TYPES_LABEL, N_SPREAD_STATS
from agents.observation.constants import TEAM_SIZE
from agents.training.fork_arm import PG_MASK_KEY as FORK_PG_MASK_KEY

#: The trainee's action space: 4 moves + 5 switches + 2 reserved slots (``Gen3ActionMapper``).
N_ACTIONS = 11


def trainee_env_kwargs(args: Any) -> dict:
    """The trainee's per-run label switches — which training-only label keys its observation emits
    (ARCHITECTURE.md §7) — as a pure function of the resolved ``args``, so every builder of a
    training-shaped env derives the SAME surface from one place. (The ``emit_*`` names are the
    ``label_gates`` keywords; they were the ``Gen3Env`` constructor's before U3.)"""
    return dict(
        # TRAINING-only privileged belief labels (only the trainee env; the model side
        # gates the BeliefHead on the same coef>0 signal). Eval/self-play opponents play
        # via RLPlayer, not Gen3Env, so they never emit them.
        emit_belief_labels=(args.opp_belief_aux_coef > 0.0),
        move_belief_mode=args.move_belief_mode,
        emit_win_target=(args.win_prob_mode != "none"),
        # gen3_fork_v1: the per-row POLICY-TERM mask key (`fork_pg_m`), declared only when
        # the arm is on. It is the carrier for THE MASK RULE — the fork step is out of the
        # policy term for every branch — and an injected row is the only row that ever
        # holds anything but the 1.0 placeholder, so an unflagged run's observation space,
        # its rollout buffer and its policy loss are all untouched.
        emit_fork_pg_mask=(float(getattr(args, "fork_fraction", 0.0) or 0.0) > 0.0),
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
    fork_pg_mask: bool


def label_gates(*, emit_belief_labels: bool = False, move_belief_mode: str = "off",
                emit_win_target: bool = False,
                emit_fork_pg_mask: bool = False, emit_spread_labels: bool = False,
                emit_opp_intent_labels: bool = False, emit_hp_type_labels: bool = False,
                emit_item_labels: bool = False) -> LabelGates:
    """The per-key gates. One is DERIVED: the move belief needs the belief labels whatever
    ``--opp-belief-aux-coef`` says."""
    return LabelGates(
        belief_labels=bool(emit_belief_labels or move_belief_mode != "off"),
        known_moves=move_belief_mode in ("revealed", "both"),
        opp_intent_labels=bool(emit_opp_intent_labels),
        win_target=bool(emit_win_target),
        spread_labels=bool(emit_spread_labels),
        hp_type_labels=bool(emit_hp_type_labels),
        item_labels=bool(emit_item_labels),
        fork_pg_mask=bool(emit_fork_pg_mask),
    )


def trainee_observation_space(layout: dict, vector_space: Any, g: LabelGates) -> spaces.Dict:
    """The trainee's ``spaces.Dict``: the flat observation + the action mask + every label key ``g``
    switches on. ``layout`` is the encoder's ``get_layout()``,
    ``vector_space`` its schema's gym space."""
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
    if g.fork_pg_mask:
        # gen3_fork_v1: 1.0 = this row is in the clipped policy term, 0.0 = it is not.
        # Bounded in [0, 1] because it is a MASK and not a dose — the fork arm does not
        # re-weight the policy gradient, it excludes exactly the fork step (see
        # `agents.training.fork_buffer` and `rust_rollout/fork_test.py`, THE MASK RULE).
        base_obs[FORK_PG_MASK_KEY] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    return spaces.Dict(base_obs)


def trainee_spaces(args: Any, mappings: Any = None) -> Tuple[spaces.Dict, spaces.Discrete]:
    """``(observation_space, action_space)`` of the trainee ``args`` declare — what the Rust env core's
    ``RustVecEnv`` and the learner build against, and what a checkpoint records."""
    from agents.observation.schema import build_schema
    from agents.observation.state_encoder import get_observation_encoder, load_mappings

    layout = get_observation_encoder(mappings if mappings is not None else load_mappings()).get_layout()
    kw = trainee_env_kwargs(args)
    obs = trainee_observation_space(layout, build_schema(layout).gym_space(), label_gates(**kw))
    return obs, spaces.Discrete(N_ACTIONS)
