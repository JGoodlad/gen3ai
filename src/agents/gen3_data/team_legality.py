"""TEAM VALIDATION against the gen3ou format spec (``agents.gen3_data.format_spec``) — poke-env-free.

Every pool we play (the training pool, the Smogon sample teams, the specialist bot teams) is checked
against the LADDER's rules: the bans (species, items, moves, abilities, a move banned for one species, the
per-set combos), Accuracy Trap Clause, Species Clause, One Boost Passer Clause and Speed Pass Clause. This
is the team-building half of the spec's stories; it REPORTS, it never edits a team file
(``python -m main.team_legality``).

What it does NOT check is the rest of ``Obtainable`` (learnsets, EVs, IVs, levels, event-only
combinations): Showdown's own ``TeamValidator`` owns that, and the pinned engine plays whatever it is
given. A problem caused only by a MASTER-only rule (:data:`format_spec.PINNED_DIFFERENCES`: Quick Claw,
Recycle as a One Boost Passer effect) is tagged ``ladder_only`` — the pinned engine accepts the team, the
ladder does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from . import format_spec as fs
from . import species as species_dex


@dataclass(frozen=True)
class TeamSet:
    """One set as the validator reads it: ids only (``format_spec.to_id``)."""
    species: str
    item: str
    ability: str
    moves: Tuple[str, ...]

    def has(self) -> FrozenSet[str]:
        """Every id the set holds (Showdown's per-set ``setHas`` for the combo check)."""
        return frozenset({self.species, self.item, self.ability, *self.moves} - {""})


@dataclass(frozen=True)
class Problem:
    """One violation: the ``rule`` that rejects it, the set (0-based ``slot``, -1 = the team), a message, and
    ``ladder_only`` (True = only a master-only rule rejects it; the pinned engine plays it)."""
    rule: str
    slot: int
    message: str
    ladder_only: bool = False


_GENDER = re.compile(r"\s+\((?:M|F)\)\s*$")


def parse_paste(text: str) -> List[TeamSet]:
    """A Showdown export (``Nick (Species) (M) @ Item`` / ``Ability:`` / ``- Move`` …), blank-line separated."""
    sets: List[TeamSet] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[0]
        name, _, item = head.partition(" @ ")
        name = _GENDER.sub("", name.strip())
        m = re.search(r"\(([^()]+)\)\s*$", name)
        species = m.group(1) if m else name
        ability = ""
        moves: List[str] = []
        for ln in lines[1:]:
            if ln.lower().startswith("ability:"):
                ability = ln.split(":", 1)[1]
            elif ln.startswith("-"):
                moves.append(ln[1:].strip())
        sets.append(TeamSet(fs.to_id(species), fs.to_id(item), fs.to_id(ability),
                            tuple(fs.to_id(mv) for mv in moves if fs.to_id(mv))))
    return sets


def parse_packed(packed: str) -> List[TeamSet]:
    """A Showdown PACKED team (``nick|species|item|ability|move,move|nature|evs|…``, sets joined by ``]``;
    an empty species field means the nickname IS the species) — the procedural generator's and the ladder
    corpus's form."""
    sets: List[TeamSet] = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        if len(f) < 5:
            continue
        species = f[1] or f[0]
        sets.append(TeamSet(fs.to_id(species), fs.to_id(f[2]), fs.to_id(f[3]),
                            tuple(fs.to_id(m) for m in f[4].split(",") if fs.to_id(m))))
    return sets


def _passable(s: TeamSet, effects: FrozenSet[str]) -> int:
    return sum(1 for mv in s.moves if mv in effects) + (s.item in effects) + (s.ability in effects)


def _one_boost_passer(team: Sequence[TeamSet], effects: FrozenSet[str]) -> Optional[Tuple[int, str]]:
    """`data/rulesets.ts` oneboostpasserclause.onValidateTeam, in team order (first problem only, as there)."""
    passers = 0
    for i, s in enumerate(team):
        if "batonpass" not in s.moves:
            continue
        n = _passable(s, effects)
        if n == 1:
            passers += 1
        if n > 1:
            return i, f"{s.species} has Baton Pass and multiple ways to boost its stats"
        if passers > 1:
            return i, "multiple Pokemon have Baton Pass and a way to boost their stats"
    return None


def validate_team(team: Sequence[TeamSet], spec: Optional[fs.FormatSpec] = None) -> List[Problem]:
    """Every format-spec violation of ``team`` (empty = legal under the spec's rules)."""
    spec = spec or fs.active()
    out: List[Problem] = []
    ladder_only_items = {b.ids[0] for b in spec.bans if not b.on_pinned and b.kind == "item"}
    for i, s in enumerate(team):
        if s.species in spec.banned_species:
            ban = next(b for b in spec.bans if b.kind == "species" and b.ids[0] == s.species)
            out.append(Problem(ban.via, i, f"species {s.species} is banned ({ban.via})"))
        if s.item in spec.banned_items:
            out.append(Problem("banlist", i, f"{s.species}'s item {s.item} is banned",
                               ladder_only=s.item in ladder_only_items))
        if s.ability in spec.banned_abilities:
            out.append(Problem("banlist", i, f"{s.species}'s ability {s.ability} is banned"))
        for mv in s.moves:
            if spec.is_banned("move", mv, species=s.species):
                out.append(Problem("banlist", i, f"{s.species}'s move {mv} is banned"))
        has = s.has()
        for b in spec.combo_bans:
            if b.kind == "combo" and set(b.ids) <= has:
                out.append(Problem(b.name, i, f"{s.species} has the banned combination {b.name}"))
        if spec.has_rule("accuracytrapclause"):
            acc = set(s.moves) & fs.ACCURACY_TRAP_ACCURACY_MOVES
            trap = (s.ability in fs.ACCURACY_TRAP_TRAPPING) or bool(set(s.moves) & fs.ACCURACY_TRAP_TRAPPING)
            if acc and trap:
                out.append(Problem("Accuracy Trap Clause", i,
                                   f"{s.species} has an accuracy-lowering move {sorted(acc)} and a trap"))
        if spec.has_rule("speedpassclause") and "batonpass" in s.moves and _passable(s, fs.SPEED_PASS_EFFECTS):
            out.append(Problem("Speed Pass Clause", i, f"{s.species} has Baton Pass and a way to boost its Speed"))
    if spec.has_rule("speciesclause"):
        seen: Dict[int, int] = {}
        for i, s in enumerate(team):
            sd = species_dex.get(s.species)
            num = sd.num if sd is not None else -1 - i          # an unknown id never collides
            if num in seen:
                out.append(Problem("Species Clause", i, f"more than one {s.species} (slot {seen[num]})"))
            seen.setdefault(num, i)
    if spec.has_rule("oneboostpasserclause"):
        hit = _one_boost_passer(team, fs.ONE_BOOST_PASSER_EFFECTS)
        if hit is not None:
            pinned_effects = fs.ONE_BOOST_PASSER_EFFECTS - {
                item for key, item in fs.PINNED_DIFFERENCES if key == "oneboostpasserclause.boostingEffects"}
            out.append(Problem("One Boost Passer Clause", hit[0], hit[1],
                               ladder_only=_one_boost_passer(team, pinned_effects) is None))
    return out
