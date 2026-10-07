"""Gates for `α`/`β` (design_opponent_intent.md §6 — G3b, G4, G5 in module form).

The blob path's two heads (`AlphaIntentHead` / `BetaSwitchHead`) and their equivariance / padding / NaN-row unit
tests are DELETED with them (the X5 version break, config v144, part 2); the flat pointer's own are in
`flat_intent_test.py`. What stays:

  * MATCHING BY CANONICAL ID — seats are `w.topk(K)` and permute every turn, so a target built
    from a seat INDEX is wrong the moment the belief re-sorts. The label is a move NUM and the
    match happens at loss time.
"""
import pytest
import torch

from agents.model.opp_intent import INTENT_IGNORE, match_seats_to_move_num, render_alpha


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
    """Since the X5 version break (v144) `opp_intent` builds X5's FLAT pointer with the hypothesis tokens (the
    opponent-belief family is one family). OFF (both toggles off) builds none of it; ON adds exactly the X5
    modules — the blob path's α / β heads are DELETED (part 2), so they hold no state_dict key."""
    from agents.model.identity_init_test import _build_real_policy
    off, _ = _build_real_policy(**_intent_kwargs(opp_intent=False, opp_belief_slots=False))
    on, _ = _build_real_policy(**_intent_kwargs())
    fo, fn = off.policy.features_extractor, on.policy.features_extractor
    assert fo.flat_intent_head is None and fo.hypothesis_builder is None
    assert not hasattr(fo, "alpha_head") and not hasattr(fo, "beta_head")
    assert fn.flat_intent_head is not None and fn.hypothesis_builder is not None
    assert not hasattr(fn, "alpha_head") and not hasattr(fn, "beta_head")   # deleted
    new = set(fn.state_dict()) - set(fo.state_dict())
    assert new and {k.split(".")[0] for k in new} == {"flat_intent_head", "hypothesis_builder", "belief_head"}, \
        sorted(new)[:4]
    assert not any(k.startswith(("alpha_head.", "beta_head.")) for k in fn.state_dict())
    assert not (set(fo.state_dict()) - set(fn.state_dict())), "OFF must not have keys ON lacks"


def test_a_real_policy_emits_a_normalized_alpha_over_seats_plus_switch():
    """The consumers' α is the flat pointer's RE-EXPRESSION (`FlatConsumerOps.alpha`, logits over the K seats ·
    OTHER_move · SWITCH) — one normalised distribution; β the six slots · OTHER_species. The blob α's stash is
    deleted."""
    from agents.model.identity_init_test import _build_real_policy
    m, _ = _build_real_policy(**_intent_kwargs())
    fe = m.policy.features_extractor.eval()
    dim = m.observation_space["observation"].shape[0]
    with torch.no_grad():
        fe({"observation": torch.rand(3, dim), "action_mask": torch.ones(3, 11)})
    assert not hasattr(fe, "last_alpha_logits") and not hasattr(fe, "last_beta_logits")
    assert fe.last_flat_intent_logits.shape == (3, 6 + 8), "K=6 seats + OTHER_move + six slots + OTHER_species"
    ops = fe.stash.flat_consumer_ops
    assert ops.alpha.shape == (3, 6 + 2), "K=6 seats + OTHER_move + SWITCH"
    assert ops.beta.shape == (3, 7), "six slots + OTHER_species"
    # α IS the flat distribution re-expressed: its softmax puts the pointer's own mass on each seat and
    # OTHER_move, and the whole switch mass (six slots + OTHER_species) on SWITCH — normalised over seats+switch
    pf = torch.softmax(fe.last_flat_intent_logits.float(), dim=-1)
    pa = torch.softmax(ops.alpha.float(), dim=-1)
    assert torch.allclose(pa.sum(-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(pa[:, :7], pf[:, :7], atol=1e-5)
    assert torch.allclose(pa[:, 7], pf[:, 7:].sum(-1), atol=1e-5)
    assert fe.last_flat_intent.seat_nums.shape == (3, 6)


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
