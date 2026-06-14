"""Pure builder for the hidden-opponent belief aux LABELS.

Given the opponent's FULL team (privileged self-play info) and which slots the encoder left
un-revealed, produce the per-opp-slot supervision target for the in-place belief head:

    belief_species [TEAM_SIZE]      int  — species NUM of the hidden mon assigned to each believed
                                           slot (canonical order); -1 for revealed / non-target slots.
    belief_moves   [TEAM_SIZE, MAX] int  — up to MAX move NUMs of that hidden mon; -1 pad.

Canonical assignment (must match `BeliefSlots`' slot order in the model):
- believed slots = opp slots with species_known==0 (the trailing zero-placeholder slots in the
  encoder's revealed-first opp iteration).
- hidden mons = the opponent's full team minus the already-revealed species, sorted by species num
  ascending (deterministic, learnable — the slots are interchangeable so any stable order works).
- the j-th sorted hidden mon labels the j-th believed slot.

This is the ONLY hindsight-privileged signal; it never enters the policy/value forward (it rides as
a separate Dict-obs key consumed only by the aux loss), so it cannot leak into the acting path.

Pure + dependency-free (caller injects the name->num maps and a normaliser) so it unit-tests with
synthetic data and stays off the obs hot path unless belief is enabled.
"""
from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np

from agents.observation.constants import TEAM_SIZE

BELIEF_MOVE_SLOTS = 4  # moves labelled per hidden mon (Gen 3 mons have <= 4 moves)
PAD = -1


def assign_hidden_to_slots(
    team_species: Sequence[str],
    revealed_species: Sequence[str],
    species_known: Sequence[float],
    species_to_num: Dict[str, int],
    normalize: Callable[[str], str],
) -> List[Tuple[int, int]]:
    """The CANONICAL believed-slot → hidden-mon assignment, the SINGLE source of truth shared by
    `build_belief_labels` (species/moves int labels) and `build_belief_target_index` (the latent
    target's fresh per-mon encode). Returning one assignment guarantees the species-CE label and the
    encoder-role-token latent target can NEVER name a different mon for the same believed slot.

    Returns a list of (encoder_opp_slot, team_index) pairs in believed-slot order — `team_index` is
    the index into `team_species` (and the caller's parallel team/mon lists) of the hidden mon
    assigned to that slot. Mirrors the prior in-line logic exactly: hidden = full team minus revealed
    (skipping species not in the num map), sorted by species num ascending; believed slots = the
    trailing un-revealed opp slots (species_known < 0.5) in encoder order; the j-th sorted hidden mon
    fills the j-th believed slot. Slots beyond a clean 1-1 assignment are dropped (left to the caller
    to PAD)."""
    revealed = {normalize(s) for s in revealed_species}
    hidden: List[Tuple[int, int]] = []   # (species_num, team_index)
    for idx, sp in enumerate(team_species):
        sp_norm = normalize(sp)
        if sp_norm in revealed:
            continue
        num = species_to_num.get(sp_norm)
        if num is None:
            continue
        hidden.append((num, idx))
    hidden.sort(key=lambda t: t[0])

    believed_slots = [i for i in range(TEAM_SIZE) if i < len(species_known) and species_known[i] < 0.5]
    assignment: List[Tuple[int, int]] = []
    for j, slot in enumerate(believed_slots):
        if j >= len(hidden):
            break  # fewer hidden mons than believed slots (parse mismatch) — leave the rest PAD
        assignment.append((slot, hidden[j][1]))
    return assignment


def build_belief_target_index(
    team_species: Sequence[str],
    revealed_species: Sequence[str],
    species_known: Sequence[float],
    species_to_num: Dict[str, int],
    normalize: Callable[[str], str],
) -> np.ndarray:
    """Per opp-slot index (int64[TEAM_SIZE]) into the FULL team of the hidden mon assigned to that
    believed slot; PAD (-1) for revealed / non-target slots. Same assignment as `build_belief_labels`
    (both call `assign_hidden_to_slots`), so the latent-belief target encode and the species-CE label
    reference the identical mon per slot. The caller encodes `team[idx]` fresh for the latent target."""
    target_idx = np.full(TEAM_SIZE, PAD, dtype=np.int64)
    for slot, team_idx in assign_hidden_to_slots(
        team_species, revealed_species, species_known, species_to_num, normalize
    ):
        target_idx[slot] = team_idx
    return target_idx


def build_belief_labels(
    team_species: Sequence[str],
    team_moves: Sequence[Sequence[str]],
    revealed_species: Sequence[str],
    species_known: Sequence[float],
    species_to_num: Dict[str, int],
    move_to_num: Dict[str, int],
    normalize: Callable[[str], str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (belief_species[TEAM_SIZE] int64, belief_moves[TEAM_SIZE, BELIEF_MOVE_SLOTS] int64).

    team_species / team_moves : the opponent's FULL team (parallel; up to TEAM_SIZE entries).
    revealed_species          : species (names) already revealed this battle.
    species_known             : per encoder opp-slot 0/1 — 1 revealed, 0 believed (len TEAM_SIZE).
    species_to_num / move_to_num : name(normalised) -> embedding-index num.
    normalize                 : id-normaliser applied to every name before map lookup.

    Unknown names (not in a map) are skipped/padded rather than raising — robustness on the hot path.
    Slots/mons beyond a clean 1-1 assignment stay PAD (not scored)."""
    belief_species = np.full(TEAM_SIZE, PAD, dtype=np.int64)
    belief_moves = np.full((TEAM_SIZE, BELIEF_MOVE_SLOTS), PAD, dtype=np.int64)

    # Single source of truth for which hidden mon fills which believed slot (shared with the latent
    # target via `assign_hidden_to_slots`). The species num + moveset are then derived from the
    # assigned team index — byte-identical to the prior inline build.
    for slot, team_idx in assign_hidden_to_slots(
        team_species, revealed_species, species_known, species_to_num, normalize
    ):
        belief_species[slot] = species_to_num[normalize(team_species[team_idx])]
        m = 0
        for mv in team_moves[team_idx]:
            if m >= BELIEF_MOVE_SLOTS:
                break
            mv_num = move_to_num.get(normalize(mv))
            if mv_num is None:
                continue
            belief_moves[slot, m] = mv_num
            m += 1

    return belief_species, belief_moves


def build_known_move_labels(
    revealed_species_in_slot_order: Sequence[str],
    team_species: Sequence[str],
    team_moves: Sequence[Sequence[str]],
    species_known: Sequence[float],
    move_to_num: Dict[str, int],
    normalize: Callable[[str], str],
) -> np.ndarray:
    """Build known_moves[TEAM_SIZE, BELIEF_MOVE_SLOTS] int64 — the privileged label for the MOVE
    belief's 'known' slots.

    Each REVEALED opp slot (species_known==1, the leading-contiguous block, in encoder order) gets the
    FULL privileged moveset of the species revealed there. The move head then learns to predict that
    mon's still-UNREVEALED moves (the surprise-OHKO "it had a move we hadn't seen" gap) — defensible
    supervision (we have SEEN this species; we're guessing its remaining moves). Believed / pad slots
    stay PAD (scored only by the 'unknown'-slot path).

    revealed_species_in_slot_order : species names at each revealed slot, encoder order (parallel to
                                     the revealed block of species_known).
    team_species / team_moves      : the opponent's FULL privileged team (parallel) — the source of
                                     each revealed species' complete moveset.
    species_known                  : per encoder opp-slot 0/1 (1 revealed, 0 believed).
    move_to_num / normalize        : name(normalised) -> embedding-num map + id-normaliser.

    Unknown names are skipped/padded (never raises — hot path)."""
    known_moves = np.full((TEAM_SIZE, BELIEF_MOVE_SLOTS), PAD, dtype=np.int64)
    # privileged species_norm -> full moveset (first occurrence; Gen-3 OU species-clause ⇒ unique).
    full_by_species: Dict[str, List[str]] = {}
    for sp, moves in zip(team_species, team_moves):
        full_by_species.setdefault(normalize(sp), list(moves))
    revealed_slots = [i for i in range(TEAM_SIZE) if i < len(species_known) and species_known[i] >= 0.5]
    for slot, sp in zip(revealed_slots, revealed_species_in_slot_order):
        moves = full_by_species.get(normalize(sp))
        if moves is None:
            continue
        m = 0
        for mv in moves:
            if m >= BELIEF_MOVE_SLOTS:
                break
            mv_num = move_to_num.get(normalize(mv))
            if mv_num is None:
                continue
            known_moves[slot, m] = mv_num
            m += 1
    return known_moves


def zero_known_moves() -> np.ndarray:
    """All-PAD known_moves — off / pre-battle / parse-failure path."""
    return np.full((TEAM_SIZE, BELIEF_MOVE_SLOTS), PAD, dtype=np.int64)


def zero_belief_labels() -> Tuple[np.ndarray, np.ndarray]:
    """All-PAD labels — used for the off / pre-battle / parse-failure path so the Dict key always
    has a consistent shape (every slot PAD ⇒ nothing scored)."""
    return (
        np.full(TEAM_SIZE, PAD, dtype=np.int64),
        np.full((TEAM_SIZE, BELIEF_MOVE_SLOTS), PAD, dtype=np.int64),
    )
