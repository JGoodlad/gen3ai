"""Dex-IDENTITY facts on the national-dex ``num`` axis — the layer BOTH the physics and the beliefs key on.

Split out of `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06), the second round of the cut
`belief_tables.py` began. That first round moved the belief PRIORS out; this one moves the small set
of facts the two halves SHARE, so the belief priors could follow without dragging dex identity into a
belief module. What lives here answers only *which num is this thing, and how often is it seen* —

  * ``HIDDEN_POWER_NUM`` / `_belief_num` / `_hp_typed_nums` — the Hidden-Power num identity: the bare
    typeless 237 the opponent's belief keys its PRESENCE channel on, the id→num fold that produces it,
    and the 16 DISTINCT typed nums 355-370 (`gen3_typed_hidden_power_ids_v1`). The op's physics reads
    the first and third (typed-candidate expansion, the BP guard); `build_move_prior_logits` reads all
    three. Neither owns them.
  * `build_species_usage_prior` — the normalized gen3ou species usage share per num. Read by
    `build_damage_buffers` as the op's ``SPECIES_USAGE_PRIOR`` (the expected-latent defender for the
    OUTGOING kernel's unrevealed columns) AND by `belief_tables.build_species_cooccur_prior` as its
    marginal. A pure `gen3_data` read with no coupling to either consumer, which is what makes it
    placeable here rather than in one of them.

The LAYERING is the reason this module exists, and it runs strictly one way:

    ``damage_tables`` → ``belief_tables`` → ``dex_ids``,  and  ``damage_tables`` → ``dex_ids``

Nothing here may import `belief_tables` or `damage_tables` — an import back closes a cycle Python
resolves only for whichever module was imported first, so it would work in the normal import order
and raise in every other. `belief_tables_test.py` AST-scans both edges to keep it so.

`damage_tables` re-exports every name below, so `from agents.model.damage_tables import
HIDDEN_POWER_NUM` (and the `_hp_typed_nums` / `_belief_num` / `build_species_usage_prior` spellings
in `gen3_env`, `belief_heads` and five test modules) still resolves.
"""
from __future__ import annotations

from functools import lru_cache
from typing import cast, Tuple

import torch

from agents import gen3_data
from agents.gen3_data.moves import MoveData
from agents.gen3_data.species import SpeciesData
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER

# Hidden Power: all variants share this num in the OPPONENT's belief (gen3 never reveals the type).
# `damage_tables` keeps the physics siblings `HIDDEN_POWER_BP` / `N_HP_TYPES` beside the op that reads them.
HIDDEN_POWER_NUM = 237


def _belief_num(move_id: str, md: MoveData) -> int:
    """The move num the OPPONENT's move-belief PRIOR keys on. Every Hidden Power — bare or typed —
    aggregates to the typeless num ``237``, which under `gen3_typed_hp_belief_v1` is the belief's
    **PRESENCE channel**: ``prior[species, 237] = Σ_t usage(hiddenpower<t>) = P(species runs SOME HP)``.
    That is exactly the quantity the presence×type factorisation needs, and it is well-defined for the
    opponent (Gen 3 never reveals the HP type, so an opp HP is always observed bare and pins 237).

    The per-TYPE half of the factorisation is `build_hp_type_prior` (the conditional
    ``P(type | has HP)`` from the same Smogon data), and the two are multiplied back into the 16 typed
    nums 355-370 by ``HPTypeBelief.compose_typed_hp`` — which reconstructs the typed usage exactly,
    since ``P(has HP)·P(t | has HP) == usage(hiddenpower<t>)``. So no prior information is lost by
    keying presence here; the typed prior CELLS at 355-370 are overwritten by that composition and are
    never read. Non-HP moves pass through unchanged.

    See designs/ai_v6/design_typed_hidden_power_ids.md.

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06)."""
    return HIDDEN_POWER_NUM if move_id.startswith("hiddenpower") else md.num


@lru_cache(maxsize=1)
def _hp_typed_nums() -> Tuple[int, ...]:
    """The 16 DISTINCT typed-Hidden-Power dex nums (355-370) in ``HIDDEN_POWER_TYPE_ORDER`` order —
    the same axis as ``HP_TYPE_IDX`` / the obs ``hp_probs`` / ``belief_labels.HP_TYPE_NAMES``.
    Data-derived (never hardcoded) so a num remap can't silently misalign the type axis; the throwing
    GIGO guard in `build_damage_buffers` pins that alignment.

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06)."""
    return tuple(cast(MoveData, gen3_data.moves.get("hiddenpower" + t.name.lower())).num
                 for t in HIDDEN_POWER_TYPE_ORDER)


# gen3_unrevealed_outgoing_prior_v1: the FLOOR a real species with no usage entry gets, applied on the
# NORMALIZED usage scale (so it means "1-in-a-million teams", not "1e-6 raw sets" — the raw counts run to
# millions and a raw-scale floor would be indistinguishable from the hard zero it exists to prevent).
_USAGE_PRIOR_FLOOR = 1e-6


def build_species_usage_prior(n_species: int) -> torch.Tensor:
    """``[n_species]`` the normalized gen3ou species USAGE distribution over dex nums —
    ``P(an unrevealed opp slot is species s)`` before Species-Clause filtering
    (gen3_unrevealed_outgoing_prior_v1: the expected-latent defender for the OUTGOING kernel's
    unrevealed columns, marginalized through ``SPECIES_EXP_MULT`` / ``SPECIES_SPREAD_PRIOR``).

    Sourced from `gen3_data.priors.species_usage()` (the Smogon ``Raw count`` weights). The sentinel
    species (num 0) gets EXACTLY 0; every real base form absent from the usage data gets the tiny
    `_USAGE_PRIOR_FLOOR` (never a hard zero — in-battle Species-Clause renormalization must always
    be able to fall back to *something*), then the whole vector is renormalized to sum 1. BASE forms
    only (a forme shares its base's num — iterating `raw()` would double-write rows). Fail-loud on
    the canonical carrier (Tyranitar, the #1 gen3ou mon) so a key-normalization drift can't silently
    flatten the prior to the floor.

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06)."""
    usage = gen3_data.priors.species_usage()
    total = sum(usage.values())
    if total <= 0.0:
        raise ValueError("build_species_usage_prior: no species usage data — "
                         "gen3_smogon_stats.json empty/malformed?")
    prior = torch.zeros(n_species, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 < sd.num < n_species):                  # sentinel num 0 stays exactly 0
            continue
        prior[sd.num] = max(float(usage.get(sid, 0.0)) / total, _USAGE_PRIOR_FLOOR)
    tt = gen3_data.species.get("tyranitar")
    if tt is None or not (0 < tt.num < n_species) or float(prior[tt.num]) < 0.01:
        raise ValueError(
            "build_species_usage_prior: Tyranitar did not resolve to a dominant usage share — the "
            "species-usage prior is empty/misaligned (id normalization drift?). GIGO guard.")
    return prior / prior.sum()
