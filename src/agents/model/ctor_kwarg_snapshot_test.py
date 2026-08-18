"""A kwarg may not leave `Gen3FeaturesExtractor.__init__` without a dead-list verdict.

WHY THIS EXISTS — it is the failure the dead-kwarg machinery cannot detect about itself.
`snapshot.sanitize_dead_extractor_kwargs` works off a CURATED list, so it is exactly as complete as
whoever last deleted a flag remembered to make it. Nothing anywhere read the constructor's real
signature, so a forgotten name produced no error at deletion time, no failing test, and no warning
— only a bare `TypeError: got an unexpected keyword argument` months later, on the paths (training
resume, frozen pool opponents, eval workers) whose entire job is to sanitize-or-refuse WITH
JUDGMENT.

MEASURED 2026-08-17 over the 89 runs under `models/` carrying a checkpoint (via the prober's
`_dropped_extractor_kwargs`, which is pure set math over the live signature): 23 distinct saved
kwarg names the current constructor rejects, of which **five were in neither dead list** —
`mask_incoming_damage_obs` (61 runs), `mask_active_move_scalars_obs` / `mask_move_effects_obs` (58),
`spread_belief_nature_marginalize` (55) and `hp_type_belief_mode` (51). **70 of 89 runs carried at
least one**, and 7 of those (`ai_v9_01`..`ai_v9_07`) actually reached the TypeError — the other 63
were masked only because a DIFFERENT curated entry refused them first, which is luck, not coverage.
Every one of the five predates MIGRATION_FLOOR, which is why nobody noticed: the
CONFIG half of the rule collapsed into the blanket floor refusal, and the ZIP half — the pickled
`features_extractor_kwargs` SB3 actually splats into the constructor — carries no `config_version`
to fall under any floor.

So the snapshot below is not documentation of the current signature; it is a TRIPWIRE. It converts
"someone forgot" from silent into red, the same way the delivery-graph gate does for a phase that
stops being reachable. Adding a kwarg is a one-line snapshot update; REMOVING one is a decision, and
the failure message states it.
"""
import inspect

import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.snapshot import _DEAD_FEK_INERT, _DEAD_FEK_JUDGED

# Not architecture: SB3's construction contract + our two injected data handles. They are exempt
# from the dead-list rule because they were never flags and can never be recorded in a saved
# `features_extractor_kwargs` as a toggle.
_NON_FLAG_PARAMS = frozenset({"observation_space", "layout", "mappings", "log_level"})

# THE SNAPSHOT — every parameter of `Gen3FeaturesExtractor.__init__` except `self`, as of v89.
# Sorted, so a diff reads as one line added or one line removed.
CTOR_KWARGS_V89 = frozenset({
    "attend_unrevealed_opponents", "belief_grad_mode", "consequence_topk", "damage_candidate_k",
    "damage_matrices_incoming", "damage_matrices_outgoing", "damage_op", "damage_outgoing",
    "damage_topk_k", "edge_bias_families", "entity_tail_seats", "entity_topk_seats",
    "history_events", "hp_belief_mode", "intent_conditional", "intent_move_cell",
    "intent_threshold", "intent_value_reduce", "item_belief", "layout", "log_level", "mappings",
    "move_belief_mode", "move_candidate_floor", "move_latent", "move_prior_fusion",
    "observation_space", "op_believed_lean", "op_drop_renders", "opp_belief_cls_k",
    "pair_outcome_cell", "pair_outcome_switch", "switch_branch_cell",
    "conditional_threat_cell", "pair_value_route",
    "opp_belief_slots", "opp_intent", "opp_intent_grad_mode", "species_prior_fusion",
    "spread_belief", "spread_belief_nature", "t0_species_prior", "threat_prob_outspeed",
    "value_clock", "value_dist_bins", "value_dist_mode", "value_dist_vmax", "value_dist_vmin",
    "value_entity_pool", "value_entity_pool_full", "value_intent", "value_threat_inject",
    "win_prob_mode",
})

# The five names MEASURED as uncovered on 2026-08-17, with the run counts that made the case. Pinned
# by name so the measurement cannot quietly be undone by an edit to the curated lists — the numbers
# are the reason those entries exist, and a bare list would not carry it.
MEASURED_UNCOVERED_2026_08_17 = (
    ("mask_incoming_damage_obs", 61),
    ("mask_active_move_scalars_obs", 58),
    ("mask_move_effects_obs", 58),
    ("spread_belief_nature_marginalize", 55),
    ("hp_type_belief_mode", 51),
)


def _live_kwargs() -> frozenset:
    return frozenset(inspect.signature(Gen3FeaturesExtractor.__init__).parameters) - {"self"}


def _dead_names() -> frozenset:
    return frozenset(_DEAD_FEK_INERT) | {k for k, _ in _DEAD_FEK_JUDGED}


def test_constructor_kwargs_match_the_snapshot():
    """The tripwire. A REMOVED kwarg is the load-bearing half — read the message, not the diff."""
    live = _live_kwargs()
    # A non-flag param leaving is a refactor of SB3 plumbing, not a dead-flag decision — it still
    # fails (the snapshot is the contract) but it does not get the dead-list instruction.
    removed = sorted((CTOR_KWARGS_V89 - live) - _NON_FLAG_PARAMS)
    plumbing = sorted((CTOR_KWARGS_V89 - live) & _NON_FLAG_PARAMS)
    added = sorted(live - CTOR_KWARGS_V89)
    assert not plumbing, (
        f"{plumbing} left the constructor. These are SB3's construction contract / our injected "
        "data handles, not flags, so no dead-list entry applies — but every caller that builds an "
        "extractor by keyword passes them. Update CTOR_KWARGS_V89 once the call sites agree.")
    assert not removed, (
        f"{len(removed)} kwarg(s) left Gen3FeaturesExtractor.__init__: {removed}\n"
        "Every archived checkpoint still has them pickled in its "
        "policy_kwargs['features_extractor_kwargs'], and SB3 splats that dict straight into the "
        "constructor — so a name deleted here TypeErrors every resume, frozen pool opponent and "
        "eval worker that touches such a checkpoint.\n"
        "BEFORE updating this snapshot, for EACH name:\n"
        "  1. Judge it. INERT = no value of it selected anything in the surviving forward (it only "
        "sized/initialised a deleted module, was training-only, or its branch was unreachable in "
        "production). JUDGED = some value fed a forward this code can no longer reproduce — either "
        "because the state_dict is byte-identical across its values (nothing shape-based catches "
        "the swap) or because an ON value named PARAMETERS.\n"
        "  2. Add it to snapshot._DEAD_FEK_INERT, or to _DEAD_FEK_JUDGED as "
        "(name, the_one_still_reproducible_value), with a one-line rationale in that file's style.\n"
        "  3. If it also lives in model_config.json AND the config could be at or above "
        "MIGRATION_FLOOR, give model_version._migrate_config the matching entry; below the floor "
        "the blanket PRE-GENERATION refusal already covers the config half.\n"
        "  4. THEN remove it from CTOR_KWARGS_V89 here.")
    assert not added, (
        f"new constructor kwarg(s) {added} — add them to CTOR_KWARGS_V89 in this file. (Adding is "
        "safe: an OLD checkpoint simply does not record the name, and SB3 uses the default.)")


def test_no_name_is_both_alive_and_dead():
    """A resurrection would make the sanitizer silently strip a LIVE argument, reverting the flag to
    its default on every load — the same class of silent-wrong the dead lists exist to prevent."""
    clash = sorted(_dead_names() & _live_kwargs())
    assert not clash, (
        f"{clash} are in snapshot._DEAD_FEK_* but ALSO accepted by the constructor. If the flag came "
        "back, delete its dead-list entry; the sanitizer would otherwise strip it from every "
        "loaded checkpoint and silently re-default it.")


@pytest.mark.parametrize("name,n_runs", MEASURED_UNCOVERED_2026_08_17)
def test_the_measured_gap_is_covered(name, n_runs):
    """The five names measured on 89 archived runs stay covered by a curated verdict."""
    assert name in _dead_names(), (
        f"{name} was measured on {n_runs} of the 89 archived runs under models/ and is no longer "
        "in either dead list — the load paths would TypeError on those checkpoints again.")
