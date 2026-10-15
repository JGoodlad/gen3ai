"""Two changes to the replica's builder, installed into its modules at run time (the replica's files are unchanged).

1. THE LEARNSET FIX. The replica's move prior and its legality check read ``agents.gen3_data.learnset`` at the PIN,
   whose table predates ``64a695dc`` (2026-10-14) and lacks 1,500 legal (species, move) pairs: every move a species
   inherits from its pre-evolution chain (Tyranitar Dragon Dance, Breloom Spore, Swampert Hydro Pump, ...). Here
   ``replica.sampling.learnset`` is replaced by the CURRENT table (``data/pokemon/gen3_learnset.json`` at a commit at
   or after ``64a695dc``, read with the facade's own semantics: an unknown species is unconstrained). Every arm
   (F, F+M, O-M, O-U, and F+U through F) draws from it. The table file's sha256 is checked against
   ``LEARNSET_SHA256`` so a run can never silently use another table.

2. THE NEVER-SEEN DRAW, PLACED UNIFORMLY. The replica draws the never-seen slots one by one in slot order, each
   species from the teammate prior minus the bases already on the team. That sequential draw is not exchangeable
   (the earlier slots get the more popular species more often). The hand-over spec says: draw the never-seen
   species, then "place the drawn mons on the never-seen slots in a uniformly random order". So the k' mons are
   drawn exactly as the replica draws them (species, four moves, item, ability, spread, attack-IV variant, a
   gender independent of the species), then permuted uniformly onto the never-seen slots. Installed as
   ``replica.builder.draw_never_seen``, so F, F+M (and their screens' contexts) and O-U all use it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np

#: the current table (``64a695dc``'s prevo-chain learnset; unchanged since on main when this unit was built)
LEARNSET_SHA256 = "c90202a18dcafe728a9fddfb81393fb55f1c45a4cacb8c06a4742170fa456596"
HERE = Path(__file__).resolve().parent.parent


def learnset_path() -> Path:
    env = os.environ.get("HIU_LEARNSET")
    if env:
        return Path(env)
    local = HERE / "gen3_learnset.json"            # the frozen copy beside a code snapshot
    if local.exists():
        return local
    return HERE.parents[3] / "data" / "pokemon" / "gen3_learnset.json"   # the checkout's own data/


class CurrentLearnset:
    """``agents.gen3_data.learnset``'s API over the current table."""

    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != LEARNSET_SHA256:
            raise RuntimeError(f"learnset table {path} sha256 {self.sha256} != the registered {LEARNSET_SHA256}")
        self.path = str(path)
        self._dex: Dict[str, FrozenSet[str]] = {s: frozenset(m) for s, m in json.loads(raw).items()}

    def get_legal_moves(self, species: str) -> Optional[FrozenSet[str]]:
        return self._dex.get(species)

    def is_legal(self, species: str, move_id: str) -> bool:
        moves = self._dex.get(species)
        return True if moves is None else move_id in moves


_INSTALLED: Optional[CurrentLearnset] = None


def draw_never_seen(prep, world: List, rng: np.random.Generator) -> None:
    """The replica's never-seen draw (``builder.draw_never_seen``) with the drawn mons placed on the never-seen
    slots in a uniformly random order."""
    from replica import builder as B
    from replica import sampling as S
    free = [i for i in range(6) if world[i] is None]
    if not free:
        return
    revealed = [m.species for m in prep.seen.order()]
    present = [w for w in world if w is not None]
    drawn = []
    for _ in free:
        bases = [S.base_of(w.species) for w in present + drawn]
        tp = S.teammate_prior(revealed, bases)
        if not tp:
            raise RuntimeError(f"{prep.sid}: empty teammate prior for a never-seen slot")
        sp = S.weighted_pick(tp, rng)
        mv = S.draw_moves(sp, [], 4, rng, allow_hp=True)
        hp = next((S.hp_type(m) for m in mv if S.is_hp(m)), None)
        nature, evs, _ = B._pick_spread(sp, rng)
        variant = S.ATK_IV_VARIANTS[int(rng.integers(2))]
        gender = B.gender_field(sp, "M" if rng.random() < 0.5 else "F")
        drawn.append(S.MonSet(name=B.species_name(sp), species_name=B.species_name(sp), species=sp,
                              item=S.weighted_pick(S.item_prior(sp), rng),
                              ability=S.weighted_pick(S.ability_prior(sp), rng), moves=mv, nature=nature, evs=evs,
                              ivs=S.ivs_for(hp, variant), gender=gender, tail="||", origin={"U": "drawn"}))
    perm = rng.permutation(len(free))
    for slot, k in zip(free, perm):
        world[slot] = drawn[int(k)]


def install() -> CurrentLearnset:
    """Install both changes (idempotent). Call AFTER ``replica.env.setup(cp)``."""
    global _INSTALLED
    from replica import builder as B
    from replica import sampling as S
    if _INSTALLED is None:
        _INSTALLED = CurrentLearnset(learnset_path())
    S.learnset = _INSTALLED            # move_ok / legality read ``learnset.is_legal`` at call time
    S._CP_CACHE.clear()
    B.draw_never_seen = draw_never_seen
    # the fix is live: a move the pinned table lacked is now in the prior (Tyranitar Dragon Dance)
    if not S.move_ok("tyranitar", "dragondance") or "dragondance" not in S.move_prior("tyranitar", [], True):
        raise RuntimeError("the learnset fix did not take: tyranitar dragondance is still filtered out")
    return _INSTALLED
