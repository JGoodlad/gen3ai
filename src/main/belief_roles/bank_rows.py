"""The Lane S bank as the readers' substrate: re-encoded rows + every TRUTH a belief read needs.

- **Rows** are re-encoded through this checkout's Rust core (``policy_spectrum.reader.reencode_with_labels``,
  gate ① byte-checked) in the bank's FIXED decision order.
- **Both full teams sit on every battle record** (G-4, VERIFIED by X5 Tier 0): the opponent's true
  species (nums, formes folded into the base num) and true movesets come from the OTHER side's packed
  team, so the presence labels and N_ρ are exact.
- **The opponent-intent label** of a banked decision is the trackers' ``IntentLabel`` at the viewer's
  NEXT entry (what the opponent did at this decision, the label training aligns to the row;
  ``opp_intent_labels`` holds the move / switch / masked semantics). It is mapped onto the COMMON EVENT
  SPACE (:func:`event_of`): a move by num, with every Hidden Power collapsed into ONE event (237: the
  opponent clicked Hidden Power; its type is a set property, never a choice), or a switch-in by species.
- **On-pool / off-pool**: a battle is on-pool iff the opponent's species set is one of the pool's
  archetype teams (``data/teams/gen3_team_archetypes.json``). That is a MEASUREMENT of the bank (the
  pool may measure, never ship as a prior); the bank v1 is all on-pool (Tier 0 F1), so off-pool is
  reported with n = 0 until a ladder bank exists (X8 / X9).
"""
from __future__ import annotations

import ast
import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from main.belief_roles.roles import HP_NUM

#: Switch events live above every move num: event = SWITCH_BASE + species num.
SWITCH_BASE = 1000
#: No supervised label at this decision (UNKNOWN kind, or the viewer's last entry).
NO_EVENT = -1
KIND_MOVE, KIND_SWITCH = 0, 1


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def species_num(name: str) -> int:
    from agents import gen3_data

    sp = gen3_data.species.get(to_id(name))
    if sp is None or int(sp.num) <= 0:
        raise KeyError(f"species {name!r} has no dex num (F-X5-3: never a silent 0)")
    return int(sp.num)


def move_event_num(move_id: str) -> int:
    """A move's num on the common event space: every Hidden Power (typeless or typed) → 237."""
    from agents import gen3_data

    mid = to_id(move_id)
    mid = _move_aliases().get(mid, mid)          # a packed team may spell `wisp` (Showdown's alias table)
    if mid.startswith("hiddenpower"):
        return HP_NUM
    md = gen3_data.moves.get(mid)
    if md is None:
        raise KeyError(f"move {move_id!r} has no dex num (F-X5-3: never a silent UNKNOWN)")
    return int(md.num)


@functools.lru_cache(maxsize=1)
def _move_aliases() -> Dict[str, str]:
    """``data/pokemon/gen3_move_aliases.json`` (the dex alias table the Rust port reads too)."""
    from agents.gen3_data import _base

    return {str(k): str(v) for k, v in _base.load_json("gen3_move_aliases.json").items()}


def event_of(label: Optional[dict]) -> int:
    """The trackers' intent label → an event id (:data:`NO_EVENT` when unsupervised)."""
    if not label:
        return NO_EVENT
    kind = int(label.get("kind", 2))
    if kind == KIND_MOVE:
        mid = label.get("move_id")
        return move_event_num(mid) if mid else NO_EVENT
    if kind == KIND_SWITCH:
        sp = label.get("switch_species")
        if not sp:
            raise KeyError("a SWITCH label with no switch-in species (F-X5-3)")
        return SWITCH_BASE + species_num(sp)
    return NO_EVENT


def packed_mons(packed: str) -> List[Tuple[str, List[str]]]:
    """``[(species, [move ids])]`` of a packed team (field 1 = species, or field 0 when empty)."""
    out = []
    for mon in packed.split("]"):
        f = mon.split("|")
        sp = f[1] if len(f) > 1 and f[1] else f[0]
        moves = [to_id(m) for m in (f[4].split(",") if len(f) > 4 and f[4] else []) if m]
        out.append((sp, moves))
    return out


def _as_dict(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def pool_species_sets() -> Set[FrozenSet[int]]:
    """The pool archetypes' species sets, as num sets (a MEASUREMENT reference only)."""
    from utils.paths import repo_path

    d = json.loads(repo_path("data", "teams", "gen3_team_archetypes.json").read_text())
    return {frozenset(species_num(s) for s in t["species"]) for t in d["teams"].values()}


@dataclass
class BankRows:
    bank: object                    # main.policy_spectrum.bank.Bank
    rows: np.ndarray                # [N, obs_dim] float32
    masks: np.ndarray               # [N, 11] bool
    gate: dict                      # gate ① result
    battle_index: np.ndarray        # [N] int — the decision's battle (cluster) index
    opp_class: List[str]            # [N] bot / pool_snapshot / exploiter
    on_pool: np.ndarray             # [N] bool
    true_species: np.ndarray        # [N, 6] int — the OPPONENT's true team nums (bank team order)
    true_moves: List[List[Set[int]]]  # [N][6] — each opponent mon's true move nums (HP → 237)
    event: np.ndarray               # [N] int — the opponent's realised action (NO_EVENT: unsupervised)

    @property
    def n(self) -> int:
        return int(self.rows.shape[0])

    def subset(self, idx: Sequence[int]) -> "BankRows":
        ix = np.asarray(idx, dtype=np.int64)
        return BankRows(bank=self.bank, rows=self.rows[ix], masks=self.masks[ix], gate=self.gate,
                        battle_index=self.battle_index[ix], opp_class=[self.opp_class[i] for i in ix],
                        on_pool=self.on_pool[ix], true_species=self.true_species[ix],
                        true_moves=[self.true_moves[i] for i in ix], event=self.event[ix])


def load_bank_rows(bank_dir: Path, workers: int = 2) -> BankRows:
    """Load the bank at ``bank_dir``, re-encode it and attach the truths (module docstring)."""
    from main.policy_spectrum.bank import load_bank

    return bank_rows_of(load_bank(Path(bank_dir)), workers=workers)


def bank_rows_of(bank, workers: int = 2) -> BankRows:
    """:func:`load_bank_rows` on an already-loaded ``Bank`` (or a slice of one — the tests)."""
    from main.policy_spectrum.reader import reencode_with_labels

    rows, masks, gate, labels = reencode_with_labels(bank, workers=workers)
    pool = pool_species_sets()
    team_of: Dict[Tuple[str, str], Tuple[List[int], List[Set[int]]]] = {}
    bidx: Dict[str, int] = {}
    for i, b in enumerate(bank.battles):
        bidx[b.battle_id] = i
        for side, other in (("p1", "p2"), ("p2", "p1")):
            mons = packed_mons(_as_dict(getattr(b, other))["team"])
            team_of[(b.battle_id, side)] = ([species_num(sp) for sp, _ in mons],
                                            [{move_event_num(m) for m in mv} for sp, mv in mons])
    n = len(bank.decisions)
    true_species = np.zeros((n, 6), dtype=np.int64)
    true_moves: List[List[Set[int]]] = []
    on_pool = np.zeros(n, dtype=bool)
    battle_index = np.zeros(n, dtype=np.int64)
    event = np.full(n, NO_EVENT, dtype=np.int64)
    opp_class = []
    for i, d in enumerate(bank.decisions):
        sp, mv = team_of[(d["battle"], d["side"])]
        if len(sp) != 6:
            raise ValueError(f"{d['id']}: the opponent's packed team holds {len(sp)} mons, not 6")
        true_species[i] = sp
        true_moves.append(mv)
        on_pool[i] = frozenset(sp) in pool
        battle_index[i] = bidx[d["battle"]]
        event[i] = event_of(labels[i])
        opp_class.append(str(d["opp_class"]))
    return BankRows(bank=bank, rows=rows, masks=masks, gate=gate, battle_index=battle_index,
                    opp_class=opp_class, on_pool=on_pool, true_species=true_species,
                    true_moves=true_moves, event=event)
