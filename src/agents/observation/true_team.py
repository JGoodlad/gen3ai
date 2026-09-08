"""Pure builder for the PRIVILEGED true-opponent-team block (`gen3_value_true_team_v1`).

The critic ladder's arm 5 (`designs/research_state/winprob_critic_ladder_2026-09-08.md` §L1) asks
one question: *how much of the win-prob critic's residual error is irreducible uncertainty about
the opponent's team?* Answering it needs a channel that carries the opponent's ACTUAL six mons —
species, moves, item, ability, derived stats, current HP and status — to the VALUE path and to
nothing else.

**It is not a new encoding.** The block this module builds is `TEAM_SIZE × POKEMON_FULL_DIM` — the
exact per-mon layout `state_encoder` writes for the opp-team slice of the 2501-dim vector — and it
is produced by the SAME `PokemonEncoder.encode`, called with `is_own=True` against the OPPONENT's
own battle view. That is the whole trick: from the opponent's side of the sim every one of its mons
is a fully-known own mon, so the privileged block is byte-for-byte the encoding the trainee would
see if the opponent's team were its own.

**Canonical order: species num ascending**, the same deterministic order
`belief_labels.assign_hidden_to_slots` uses for the hidden-mon assignment. Slot position carries no
meaning on this channel (the consumer pools over the six rows with permutation-invariant attention),
so any stable order works — sharing the belief labels' order is what keeps the two privileged views
of the same team describable in one sentence.

**Off the hot path.** Like `belief_labels`, this module lives here for cohesion with the obs layer
but is NOT called by `Gen3ObservationEncoder.encode`; `Gen3Env` / `RLPlayer` invoke it only when
`--value-true-team` is on, so the default obs build pays nothing.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from agents.observation.constants import (
    POKEMON_ACTIVE_OFFSET,
    POKEMON_FULL_DIM,
    POKEMON_VECTOR_DIM,
    TEAM_SIZE,
)

#: The obs key the block rides on. Training-only + eval-only: absent at ladder play, where no
#: local sim exists to read the opponent's team from (see `src/main/play.py`).
TRUE_TEAM_KEY = "opp_true_team"

#: `[TEAM_SIZE, POKEMON_FULL_DIM]` — the shape of the emitted block.
TRUE_TEAM_SHAPE = (TEAM_SIZE, POKEMON_FULL_DIM)


def empty_true_team_block() -> np.ndarray:
    """The all-zero block — the honest encoding of "no privileged view available".

    Zero is not a sentinel invented here: `POKEMON_SPECIES_KNOWN_OFFSET` is 0.0 in a zero row, which
    is exactly how `state_encoder` writes an ABSENT slot, so the consumer reads "unknown mon"
    through the same channel it already understands.
    """
    return np.zeros(TRUE_TEAM_SHAPE, dtype=np.float32)


def _species_num(mon: Any, species_to_num: Optional[dict]) -> int:
    """Sort key: the species embedding num, or a large stable fallback for an unmapped name."""
    if species_to_num is None:
        return 0
    from poke_env.data.normalize import to_id_str  # local: keeps this module import-light
    return int(species_to_num.get(to_id_str(mon.species), 1 << 30))


def build_true_team_block(
    team: Sequence[Any],
    battle: Any,
    pokemon_encoder: Any,
    species_to_num: Optional[dict] = None,
    active_mon: Any = None,
) -> np.ndarray:
    """`(the opponent's OWN team, the opponent's OWN battle view)` → `[TEAM_SIZE, POKEMON_FULL_DIM]`.

    `team` is the opponent player's `battle.team.values()` — real poke-env `Pokemon` objects on the
    side that owns them, so every hidden-in-Gen-3 fact (item, ability, EV spread, the fourth move,
    the Hidden Power type) is populated. `battle` MUST be that same opponent-side battle: `encode`
    reads side conditions and the active mon from it, and passing the trainee's view would silently
    describe the wrong board.

    Rows beyond the opponent's team size stay ZERO (`species_known == 0`), which is how the encoder
    already writes an absent slot. Never raises: it rides the per-decision emit path, and a label
    channel that can kill a rollout is worse than one that degrades to "unknown".
    """
    block = empty_true_team_block()
    if not team:
        return block
    mons = sorted(team, key=lambda m: _species_num(m, species_to_num))
    live = None
    try:
        live = battle.live_view().ours if battle is not None else None
    except Exception:
        live = None
    for i, mon in enumerate(mons[:TEAM_SIZE]):
        try:
            live_mon = live.get(mon.species) if live is not None else None
        except Exception:
            live_mon = None
        try:
            vec = pokemon_encoder.encode(mon, battle, is_own=True, live_mon=live_mon)
        except Exception:
            continue
        block[i, :POKEMON_VECTOR_DIM] = vec
        # The ACTIVE flag is written by the CALLER in `state_encoder` too (it is hoisted out of
        # `encode` there), so it is written here for the same reason: it is a board fact, not a
        # per-mon one. `trapped` / `maybe_trapped` stay 0 — they are OUR-side legality facts that
        # have no meaning on a privileged view of the other side's party.
        _act = active_mon if active_mon is not None else getattr(battle, "active_pokemon", None)
        if _act is not None and mon is _act:
            block[i, POKEMON_ACTIVE_OFFSET] = 1.0
    return block
