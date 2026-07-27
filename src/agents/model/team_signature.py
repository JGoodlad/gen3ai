"""Team IDENTITY signature — the key the per-team LUT (`--zarch-lut`, v46) looks a team up by.

`gen3_zarch_lut_v1`. The FiLM conditioning path reads a team-static latent z; the LUT adds a FREE
per-team code on top, which means the extractor must answer "which of my N pinned teams is this?"
from the observation alone — no env plumbing, so eval / frozen opponents / the prober all work
unchanged.

**The signature is species(6) ⊕ moves(24), each block sorted.** Two properties earn each half:

  * PERMUTATION-INVARIANT — sorting makes the code independent of team order and move-slot order,
    matching `ZArchEncoder`'s DeepSets treatment of a team as a SET.
  * INVARIANT WITHIN A BATTLE — species never changes in Gen 3, and OUR OWN moveset never changes
    (the only Gen-3 exceptions are Mimic / Transform / Sketch, which `build_roster_table` REJECTS).
    Deliberately NOT in the signature: item (Knock Off / consumption mutate it mid-battle) and HP /
    status / boosts (per-decision). A signature that moved mid-battle would silently re-condition
    the policy mid-game — the GIGO class this module exists to make impossible.

**Species alone is NOT enough** — measured on the def-20 cluster, 5 of 20 teams share a species
roster with another (the same 6 mons on different sets). Adding moves gave 20/20 unique. Adding
ability gave nothing. Hence species ⊕ moves.

Typed Hidden Power resolves to its DISTINCT num (355-370, `gen3_typed_hidden_power_ids_v1`), which
is exactly what OUR side's obs carries — so the signature matches `ctx.all_move_ids` as-is.
"""

from typing import Any, Dict, List, Sequence, Tuple

from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

TEAM_SIGNATURE_SPECIES = 6
TEAM_SIGNATURE_MOVES = 24
TEAM_SIGNATURE_DIM = TEAM_SIGNATURE_SPECIES + TEAM_SIGNATURE_MOVES   # 30

#: Gen-3 moves that REWRITE a mon's own move slots mid-battle. A team running one of these breaks
#: the within-battle invariance the signature relies on, so it is rejected loudly at build time
#: rather than mis-conditioning the policy on some later turn.
_MOVE_SET_MUTATORS = frozenset({"mimic", "transform", "sketch"})


class _ParseOnly(Teambuilder):
    """poke-env's Showdown-export parser without a team-yielding role."""

    def yield_team(self) -> str:  # pragma: no cover - never used
        raise NotImplementedError


def team_signature(team_str: str, mappings: Dict[str, Any]) -> Tuple[int, ...]:
    """The 30-int identity signature of one Showdown-export team.

    Raises on a team that is not exactly 6 mons, carries an unknown species/move, or runs a
    move-set mutator — every failure is loud, never a silently-degraded code.
    """
    species_dex = mappings["species"]
    moves_dex = mappings["moves"]
    mons = _ParseOnly().parse_showdown_team(team_str)
    if len(mons) != TEAM_SIGNATURE_SPECIES:
        raise ValueError(
            f"team_signature expects a {TEAM_SIGNATURE_SPECIES}-mon team, parsed {len(mons)}")

    species_nums: List[int] = []
    move_nums: List[int] = []
    for mon in mons:
        # poke-env puts the species in `.nickname` when the export line carries no nickname
        # (`Salamence @ Leftovers`) and in `.species` when it does (`Nick (Salamence) @ ...`).
        name = to_id_str(mon.species or mon.nickname or "")
        if name not in species_dex:
            raise ValueError(f"team_signature: unknown species {name!r}")
        species_nums.append(int(species_dex[name]["num"]))
        for move in mon.moves:
            move_id = to_id_str(move)
            if move_id in _MOVE_SET_MUTATORS:
                raise ValueError(
                    f"team_signature: {name} runs {move_id!r}, which rewrites its own move slots "
                    "mid-battle — the team-identity signature would not be invariant within a "
                    "battle. Drop the team or extend the signature before using --zarch-lut.")
            if move_id not in moves_dex:
                raise ValueError(f"team_signature: unknown move {move_id!r} on {name}")
            move_nums.append(int(moves_dex[move_id]["num"]))

    # A mon may carry <4 moves; pad to a fixed width with the 0 sentinel (matching the obs's
    # zero-padded `all_move_ids`) so every signature is the same length.
    move_nums.extend([0] * (TEAM_SIGNATURE_MOVES - len(move_nums)))
    if len(move_nums) != TEAM_SIGNATURE_MOVES:
        raise ValueError(f"team_signature: {len(move_nums)} moves, expected <= {TEAM_SIGNATURE_MOVES}")
    return tuple(sorted(species_nums)) + tuple(sorted(move_nums))


def build_roster_table(team_strs: Sequence[str], mappings: Dict[str, Any]) -> List[List[int]]:
    """Signatures for the LUT's pinned teams, in order — row i ↔ LUT row i+1 (row 0 = unknown).

    **Throws on a duplicate signature.** Two indistinguishable teams would share a LUT row, so the
    per-team code the whole experiment rests on would silently be a per-PAIR code. Fail at launch,
    not in the metrics.
    """
    table: List[List[int]] = []
    seen: Dict[Tuple[int, ...], int] = {}
    for i, team_str in enumerate(team_strs):
        sig = team_signature(team_str, mappings)
        if sig in seen:
            raise ValueError(
                f"--zarch-lut: teams {seen[sig]} and {i} have the SAME species+moves signature, so "
                "the LUT cannot tell them apart (they would share one per-team code). Drop one, or "
                "extend the signature.")
        seen[sig] = i
        table.append(list(sig))
    return table
