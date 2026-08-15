"""Gates for `α`/`β` (design_opponent_intent.md §6 — G3b, G4, G5 in module form).

The two load-bearing ones:

  * EQUIVARIANCE — permuting their moves must permute `α` exactly, and permuting their bench must
    permute `β`. A flat `Linear(ctx, K)` would pass every shape test here and silently learn "seat
    0 is usually right" from the belief's own sort order, memorising the very ordering `α` exists
    to correct.
  * MATCHING BY CANONICAL ID — seats are `w.topk(K)` and permute every turn, so a target built
    from a seat INDEX is wrong the moment the belief re-sorts. The label is a move NUM and the
    match happens at loss time.
"""
import pytest
import torch

from agents.model.opp_intent import (INTENT_IGNORE, AlphaIntentHead, BetaSwitchHead,
                                     intent_losses, match_seats_to_move_num, render_alpha)


def test_alpha_is_equivariant_under_permuting_their_moves():
    torch.manual_seed(0)
    head = AlphaIntentHead(seat_dim=8, ctx_dim=6).eval()
    seats = torch.randn(3, 5, 8)
    ctx = torch.randn(3, 6)
    with torch.no_grad():
        base = head(seats, ctx)
        perm = torch.tensor([3, 0, 4, 1, 2])
        permd = head(seats[:, perm], ctx)
    assert torch.allclose(base[:, perm], permd[:, :5], atol=1e-6), \
        "alpha must permute WITH the seats — the scorer has become position-indexed"
    assert torch.allclose(base[:, 5], permd[:, 5], atol=1e-6), \
        "the SWITCH logit reads board context only and must be permutation-INVARIANT"


def test_beta_is_equivariant_under_permuting_their_bench():
    torch.manual_seed(0)
    head = BetaSwitchHead(token_dim=12, ctx_dim=6).eval()
    tok = torch.randn(2, 6, 12)
    ctx = torch.randn(2, 6)
    mask = torch.ones(2, 6)
    with torch.no_grad():
        base = head(tok, ctx, mask)
        perm = torch.tensor([5, 4, 3, 2, 1, 0])
        permd = head(tok[:, perm], ctx, mask[:, perm])
    assert torch.allclose(base[:, perm], permd, atol=1e-6)


def test_padding_and_illegal_targets_are_unrepresentable_not_merely_unlikely():
    torch.manual_seed(0)
    a = AlphaIntentHead(seat_dim=4, ctx_dim=4).eval()
    seats, ctx = torch.randn(1, 3, 4), torch.randn(1, 4)
    valid = torch.tensor([[1.0, 0.0, 1.0]])
    with torch.no_grad():
        p = a(seats, ctx, seat_valid=valid).softmax(-1)
    assert float(p[0, 1]) == 0.0, "a padded seat must receive exactly zero mass"

    b = BetaSwitchHead(token_dim=4, ctx_dim=4).eval()
    tok, c = torch.randn(1, 6, 4), torch.randn(1, 4)
    cand = torch.tensor([[1.0, 0, 0, 1.0, 0, 0]])
    with torch.no_grad():
        pb = b(tok, c, cand).softmax(-1)
    assert float(pb[0, 1]) == 0.0 and float(pb[0, 4]) == 0.0
    assert pytest.approx(float(pb.sum()), abs=1e-6) == 1.0


def test_matching_is_by_id_and_survives_a_seat_permutation():
    """The same clicked move must yield the same SEAT, whatever order the belief sorted them in."""
    seats = torch.tensor([[101, 205, 33, 7]])
    chosen = torch.tensor([205])
    kind = torch.tensor([0])
    assert int(match_seats_to_move_num(seats, chosen, kind, 4)[0]) == 1
    resorted = torch.tensor([[7, 205, 101, 33]])
    assert int(match_seats_to_move_num(resorted, chosen, kind, 4)[0]) == 1, \
        "an index-based target would have moved with the sort; an id-based one must not"


def test_a_belief_miss_is_masked_and_a_switch_is_its_own_class():
    seats = torch.tensor([[101, 205, 33, 7], [1, 2, 3, 4], [1, 2, 3, 4]])
    chosen = torch.tensor([999, 3, 0])
    kind = torch.tensor([0, 0, 1])          # miss, hit, switch
    out = match_seats_to_move_num(seats, chosen, kind, 4)
    assert int(out[0]) == INTENT_IGNORE, "a move outside the seats must be MASKED, not smeared"
    assert int(out[1]) == 2
    assert int(out[2]) == 4, "SWITCH is class n_seats"


def test_unnameable_actions_are_masked():
    out = match_seats_to_move_num(torch.tensor([[1, 2]]), torch.tensor([1]), torch.tensor([2]), 2)
    assert int(out[0]) == INTENT_IGNORE


def test_loss_skips_masked_rows_and_reports_the_mask_rate():
    torch.manual_seed(0)
    logits = torch.randn(4, 5, requires_grad=True)
    tgt = torch.tensor([1, INTENT_IGNORE, 4, INTENT_IGNORE])
    loss, m = intent_losses(logits, tgt, None, None)
    assert m["opp_intent/alpha_n_supervised"] == 2.0
    assert m["opp_intent/alpha_mask_rate"] == pytest.approx(0.5)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_all_masked_is_a_finite_zero_not_a_nan():
    """An early rollout can legitimately contain no usable label; that must not poison the loss."""
    logits = torch.randn(3, 5)
    tgt = torch.full((3,), INTENT_IGNORE)
    loss, m = intent_losses(logits, tgt, None, None)
    assert float(loss) == 0.0 and torch.isfinite(loss)
    assert m["opp_intent/alpha_mask_rate"] == pytest.approx(1.0)


def test_switch_and_move_accuracy_are_reported_separately():
    """A head that only learns 'they attack' must not hide behind the attack-heavy base rate."""
    logits = torch.full((4, 3), -10.0)
    logits[:, 0] = 10.0                       # always predicts seat 0
    tgt = torch.tensor([0, 0, 2, 2])          # two moves, two switches
    _, m = intent_losses(logits, tgt, None, None)
    # Always predicting seat 0 gets the KIND right on every move row and every switch row wrong.
    assert m["opp_intent/alpha_move_kind_recall"] == pytest.approx(1.0)
    assert m["opp_intent/alpha_move_recall_top1"] == pytest.approx(1.0)
    assert m["opp_intent/alpha_switch_recall"] == pytest.approx(0.0)
    assert m["opp_intent/alpha_acc"] == pytest.approx(0.5)


def test_render_alpha_names_every_option_and_never_invents_one():
    """G3b as a test: mass may only ever point at something with a name."""
    probs = torch.tensor([0.5, 0.2, 0.0, 0.3])
    seats = torch.tensor([101, 205, 0, 0])     # third seat unfilled
    rows = render_alpha(probs, seats, lambda n: {101: "thunderbolt", 205: "icebeam"}.get(n))
    names = [r["name"] for r in rows]
    assert names[0] == "thunderbolt" and "SWITCH" in names
    assert all(n for n in names), "every rendered option must carry a name"
    assert len(rows) == 3, "the unfilled seat must not appear at all"


# ---------------------------------------------------------------- integration (v67 wiring)

def _intent_kwargs(**over):
    """The minimum config the heads need: alpha POINTS AT the E4 seats, which need the prefuse op."""
    base = dict(opp_intent=True, entity_topk_seats=6, entity_tail_seats=True, damage_op=True,
                move_latent=True, move_belief_mode="revealed",
                attend_unrevealed_opponents=True)
    base.update(over)
    return base


def test_off_builds_no_heads_and_adds_no_state_dict_keys():
    from agents.model.identity_init_test import _build_real_policy
    off, _ = _build_real_policy(**_intent_kwargs(opp_intent=False))
    on, _ = _build_real_policy(**_intent_kwargs())
    fo, fn = off.policy.features_extractor, on.policy.features_extractor
    assert fo.alpha_head is None and fo.beta_head is None
    assert fn.alpha_head is not None and fn.beta_head is not None
    new = set(fn.state_dict()) - set(fo.state_dict())
    assert new and all(k.startswith(("alpha_head.", "beta_head.")) for k in new), sorted(new)[:4]
    assert not (set(fo.state_dict()) - set(fn.state_dict())), "OFF must not have keys ON lacks"


def test_a_real_policy_emits_a_normalized_alpha_over_seats_plus_switch():
    from agents.model.identity_init_test import _build_real_policy
    m, _ = _build_real_policy(**_intent_kwargs())
    fe = m.policy.features_extractor.eval()
    dim = m.observation_space["observation"].shape[0]
    with torch.no_grad():
        fe({"observation": torch.rand(3, dim), "action_mask": torch.ones(3, 11)})
    assert fe.last_alpha_logits.shape == (3, 7), "K=6 seats + SWITCH"
    assert fe.last_beta_logits.shape == (3, 6)
    p = torch.softmax(fe.last_alpha_logits, dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
    assert fe.last_alpha_seat_nums.shape == (3, 6)


def test_enabling_without_entity_seats_fails_loud():
    """Alpha is a POINTER over the E4 seats — with none there is nothing to point at."""
    from agents.model.identity_init_test import _build_real_policy
    with pytest.raises(ValueError, match="requires entity_topk_seats"):
        _build_real_policy(**_intent_kwargs(entity_topk_seats=0, entity_tail_seats=False))


def test_pre_v68_configs_are_below_the_floor():
    """The v68 opp_intent default branch is pre-floor since gen3_ctx_dedup_v1 raised
    MIGRATION_FLOOR: any config old enough to lack the field is a pre-generation checkpoint
    and is refused outright, and a config at the floor already records it."""
    from agents.model.model_version import (
        _migrate_config, MIGRATION_FLOOR, MODEL_CONFIG_VERSION, ModelVersionError)
    assert MODEL_CONFIG_VERSION >= 68
    assert _migrate_config({"config_version": MIGRATION_FLOOR,
                            "opp_intent": True})["opp_intent"] is True
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 67})


def test_version_gate_rejects_a_toggle_flip():
    import dataclasses
    from agents.model.model_version import ModelVersionError
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version
    base = current_model_version(load_mappings())
    on = dataclasses.replace(base, opp_intent=True)
    with pytest.raises(ModelVersionError, match="opp_intent"):
        on.check_compatible(base)
    with pytest.raises(ModelVersionError, match="opp_intent"):
        base.check_compatible(on)
    on.check_compatible(on)


def test_a_row_with_no_legal_switch_in_does_not_produce_nan():
    """THE bug a smoke caught: their last mon is active ⇒ every beta candidate masked ⇒ all -inf
    ⇒ log_softmax NaN ⇒ cross_entropy NaN for the WHOLE batch, even though such a row is always
    IGNORE'd. Those rows must come back finite."""
    torch.manual_seed(0)
    b = BetaSwitchHead(token_dim=4, ctx_dim=4).eval()
    tok, c = torch.randn(2, 6, 4), torch.randn(2, 4)
    mask = torch.zeros(2, 6)
    mask[0, 2] = 1.0                       # row 0 has one legal target; row 1 has NONE
    with torch.no_grad():
        out = b(tok, c, mask)
    assert torch.isfinite(out[1]).all(), "an all-masked row must be finite, not all -inf"
    assert float(out[0, 0]) == float("-inf"), "a masked slot in a LIVE row is still unrepresentable"
    tgt = torch.tensor([2, INTENT_IGNORE])
    loss, m = intent_losses(None, None, out, tgt)
    assert torch.isfinite(loss), f"beta loss went non-finite: {loss}"
    assert m["opp_intent/beta_n_supervised"] == 1.0


def test_a_target_on_a_masked_slot_would_be_inf_which_is_why_the_fold_masks_it():
    """The SECOND non-finite case (measured: beta_loss=inf). beta's label slot is resolved on the
    board at t+1; its logits come from the board at t. A switch-in that was UNREVEALED at t has no
    addressable slot there, so the target lands on a -inf logit. This pins WHY the PPO fold must
    drop unreachable targets — if this ever stops being inf, that guard can be revisited."""
    logits = torch.tensor([[0.5, float("-inf"), 0.2]])
    loss, _ = intent_losses(None, None, logits, torch.tensor([1]))
    assert not torch.isfinite(loss), "an unreachable target must be non-finite — hence the guard"
    loss2, _ = intent_losses(None, None, logits, torch.tensor([2]))
    assert torch.isfinite(loss2)


# ------------------------------------------- content-addressed believed-slot resolution

def _species_logits(n_slots=6, n_species=400):
    return torch.full((1, n_slots, n_species), -9.0)


def test_content_addressing_points_at_the_slot_the_MODEL_believes_holds_the_mon():
    """Not the label's Pokedex-sorted index — the slot the species posterior actually chose."""
    from agents.model.opp_intent import resolve_believed_slot_by_content
    lg = _species_logits(); lg[0, 4, 77] = 6.0        # the model thinks slot 4 is species 77
    mask = torch.tensor([[0., 0, 0, 1, 1, 1]])        # slots 3,4,5 believed
    out = resolve_believed_slot_by_content(lg, mask, torch.tensor([77]))
    assert int(out[0]) == 4


def test_content_addressing_never_picks_a_non_believed_slot():
    from agents.model.opp_intent import resolve_believed_slot_by_content
    lg = _species_logits(); lg[0, 1, 77] = 9.0        # a REVEALED slot claims it — must be ignored
    lg[0, 5, 77] = 1.0
    mask = torch.tensor([[0., 0, 0, 0, 0, 1]])
    assert int(resolve_believed_slot_by_content(lg, mask, torch.tensor([77]))[0]) == 5


def test_a_belief_miss_is_masked_rather_than_pointed_somewhere():
    """If the model does not believe the mon is anywhere, there is nothing coherent to point at —
    supervising would train beta toward the argmax of a near-uniform posterior, i.e. noise."""
    from agents.model.opp_intent import resolve_believed_slot_by_content
    lg = _species_logits()                             # flat: p(any species) ~ 1/400
    mask = torch.tensor([[0., 0, 0, 1, 1, 1]])
    assert int(resolve_believed_slot_by_content(lg, mask, torch.tensor([77]))[0]) == INTENT_IGNORE


def test_a_non_switch_row_is_masked():
    from agents.model.opp_intent import resolve_believed_slot_by_content
    lg = _species_logits(); lg[0, 4, 77] = 6.0
    mask = torch.tensor([[0., 0, 0, 1, 1, 1]])
    assert int(resolve_believed_slot_by_content(lg, mask, torch.tensor([0]))[0]) == INTENT_IGNORE


def test_content_addressing_is_INVARIANT_to_permuting_the_believed_slots():
    """THE equivariance property the index-based target could not have. Permute the believed slots
    and the resolved target follows the CONTENT, so a positional shortcut earns nothing."""
    from agents.model.opp_intent import resolve_believed_slot_by_content
    lg = _species_logits(); lg[0, 4, 77] = 6.0
    mask = torch.tensor([[0., 0, 0, 1, 1, 1]])
    base = int(resolve_believed_slot_by_content(lg, mask, torch.tensor([77]))[0])
    perm = torch.tensor([0, 1, 2, 5, 3, 4])
    moved = int(resolve_believed_slot_by_content(lg[:, perm], mask[:, perm], torch.tensor([77]))[0])
    assert perm[moved] == base, "the target must track the mon, not the index"


def test_move_accuracy_is_not_charged_for_the_switch_decision_its_baseline_never_faces():
    """`alpha_acc_move` vs `alpha_acc_move_baseline_argmax_w` is NOT a like-for-like comparison, and
    reading it as one produced a wrong conclusion on a live run ("alpha is below its own baseline").

    `pred` is an argmax over all K+1 classes, so on a MOVE row where alpha's top mass sits on SWITCH
    it scores zero. The baseline is `(tgt == 0)` — a property of the TARGETS that never chooses
    between moving and switching, so it cannot make that error at all.

    This plants the pathological case: alpha ranks the moves PERFECTLY but always prefers SWITCH.
    The raw metric reads 0.0 (looks like a head that cannot rank moves); the restricted metric reads
    1.0 (the truth); and the switch-rate diagnostic reads 1.0, naming which defect it actually is.
    """
    k = 4                                     # 4 move seats + SWITCH at index 4
    n = 6
    logits = torch.full((n, k + 1), -5.0)
    tgt = torch.tensor([0, 1, 2, 3, 0, 1])    # every row is a MOVE row
    for i, t in enumerate(tgt):
        logits[i, int(t)] = 1.0               # correct move is the best MOVE...
    logits[:, k] = 9.0                        # ...but SWITCH always outranks it
    _, m = intent_losses(logits, tgt, None, None)

    assert m["opp_intent/alpha_move_kind_recall"] == pytest.approx(0.0), \
        "axis 2 must show the pathology: it never once said 'a move'"
    assert m["opp_intent/alpha_move_recall_top1"] == pytest.approx(1.0), \
        "axis 4 must show the truth: given they moved, it ranks the moves perfectly"
    assert m["opp_intent/alpha_pred_switch_rate"] == pytest.approx(1.0), \
        "and this names the defect: it over-predicts switching"
    # THE POINT: a single fused number cannot say both of those at once, which is why the old
    # `alpha_acc_move` (kind AND target together) read 0.0 here and was mistaken for a head that
    # could not rank moves.


def test_the_restricted_metric_equals_the_raw_one_when_switch_is_never_predicted():
    """No double-counting: with SWITCH ranked last the two metrics must agree, so `_restricted` is
    a strict refinement rather than a different measurement."""
    k = 4
    logits = torch.full((5, k + 1), -5.0)
    tgt = torch.tensor([0, 2, 1, 3, 2])
    for i, t in enumerate(tgt):
        logits[i, int(t)] = 1.0
    logits[:, k] = -9.0                       # SWITCH never wins
    _, m = intent_losses(logits, tgt, None, None)
    # With SWITCH never predicted the kind decision is trivially right on every move row, so the
    # two axes decouple cleanly: kind recall saturates and the target metric carries all the signal.
    assert m["opp_intent/alpha_move_kind_recall"] == pytest.approx(1.0)
    assert m["opp_intent/alpha_move_recall_top1"] == pytest.approx(1.0)
    assert m["opp_intent/alpha_pred_switch_rate"] == pytest.approx(0.0)


def test_all_four_prediction_axes_are_reported():
    """The opponent's action is a KIND (move vs switch) and a TARGET given that kind, so there are
    four things to be right about. Each needs its own number — a fused metric hides which one broke,
    which is precisely what `alpha_acc_move` did before it was split."""
    k = 4
    logits = torch.randn(8, k + 1)
    tgt = torch.tensor([0, 1, k, 2, k, 3, 0, k])          # a mix of moves and switches
    beta_logits = torch.randn(8, 6)
    beta_tgt = torch.tensor([INTENT_IGNORE, INTENT_IGNORE, 2, INTENT_IGNORE,
                             4, INTENT_IGNORE, INTENT_IGNORE, 1])
    _, m = intent_losses(logits, tgt, beta_logits, beta_tgt)
    for axis, key in (("they switched -> KIND", "opp_intent/alpha_switch_recall"),
                      ("they moved    -> KIND", "opp_intent/alpha_move_kind_recall"),
                      ("they switched -> TARGET", "opp_intent/beta_recall_top1"),
                      ("they moved    -> TARGET", "opp_intent/alpha_move_recall_top1")):
        assert key in m, f"no metric for the axis '{axis}' ({key})"
        assert 0.0 <= m[key] <= 1.0


def test_move_kind_and_switch_prediction_rate_are_complements():
    """`alpha_acc_move_kind` and `alpha_pred_switch_on_move_rows` partition the move rows, so they
    must sum to 1 — a cheap invariant that catches either being computed on the wrong subset."""
    k = 4
    torch.manual_seed(3)
    logits = torch.randn(12, k + 1)
    tgt = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])   # all MOVE rows
    _, m = intent_losses(logits, tgt, None, None)
    assert (m["opp_intent/alpha_move_kind_recall"]
            + m["opp_intent/alpha_pred_switch_rate"]) == pytest.approx(1.0)


def test_switch_precision_is_not_the_same_as_switch_recall():
    """A head that shouts SWITCH at everything has PERFECT switch recall and terrible precision.
    Only logging recall makes those indistinguishable — and the consumers act on what alpha ASSERTS,
    so precision is the number that decides whether acting on it helps."""
    k = 4
    logits = torch.full((10, k + 1), -5.0)
    logits[:, k] = 9.0                                   # always predicts SWITCH
    tgt = torch.tensor([k, k, 0, 1, 2, 3, 0, 1, 2, 3])   # only 2 of 10 are real switches
    _, m = intent_losses(logits, tgt, None, None)
    assert m["opp_intent/alpha_switch_recall"] == pytest.approx(1.0), "caught every switch"
    assert m["opp_intent/alpha_switch_precision"] == pytest.approx(0.2), "but cried switch 5x too often"
    assert m["opp_intent/alpha_pred_switch_rate"] == pytest.approx(1.0)


def test_opp_addressable_distinguishes_hidden_from_dead():
    """gen3_opp_addressable_v1: `hp == 0` means UNKNOWN, not dead, on an unrevealed opp slot
    (measured 1033/1033) — the single-sourced addressability mask must keep hidden slots
    pointable while excluding revealed-and-fainted ones. Pinned on ObsUnpack directly so the
    rule cannot silently drift back into a consumer's inline derivation."""
    import numpy as np
    import gymnasium as gym
    import torch
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.constants import (
        OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_HP_OFFSET,
        POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_ACTIVE_OFFSET, OFFSET_OUR_TEAM)
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings)
    obs = torch.zeros(1, layout["total_dim"])
    # our active (slot 0) + opp active (slot 0) alive so locate_active_slot is well-defined
    for base in (OFFSET_OUR_TEAM, OFFSET_OPP_TEAM):
        obs[0, base + POKEMON_HP_OFFSET] = 1.0
        obs[0, base + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
        obs[0, base + POKEMON_ACTIVE_OFFSET] = 1.0
    # opp slot 1: revealed and FAINTED (species_known=1, hp=0)  -> NOT addressable
    obs[0, OFFSET_OPP_TEAM + 1 * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    # opp slot 2: UNREVEALED (species_known=0, hp=0)            -> addressable
    # opp slot 3: revealed and alive                             -> addressable
    obs[0, OFFSET_OPP_TEAM + 3 * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0
    obs[0, OFFSET_OPP_TEAM + 3 * POKEMON_FULL_DIM + POKEMON_HP_OFFSET] = 0.6
    ctx = fe.unpack({"observation": obs})
    a = ctx.opp_addressable[0]
    assert bool(a[0]) is True,  "alive active is addressable"
    assert bool(a[1]) is False, "revealed-and-fainted must NOT be addressable"
    assert bool(a[2]) is True,  "an UNREVEALED slot (hp encodes 0 = unknown) stays addressable"
    assert bool(a[3]) is True,  "revealed alive bench is addressable"
