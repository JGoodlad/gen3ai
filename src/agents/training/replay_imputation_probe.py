"""OWN-SIDE IMPUTATION METER — what human-replay training would cost us, in obs units.

The question (``designs/research_state/metamon_replay_feasibility.md`` §2.7, the memo's #1
risk). A public Showdown replay hides the IMITATED player's own unrevealed details: EVs and
nature are never shown, the item only when it activates, a move only when it is used. Metamon
fills all of that from priors *with hindsight* and **cannot measure the cost** — they have no
ground truth. We do: on our own bridge battles the TRUE observation is known at every
decision, so we can build the imputed variant beside it and read the error off directly.

**This is the meter, and ONLY the meter.** No spectator transcoder, no ``|request|``
synthesis. We take a real battle, and at each of our decisions we ask one question: *if the
only thing we knew about our own side were what a spectator had been shown by now, how far
would the observation move?* Everything else about the decision — the opponent, the board,
the log, the legality snapshot — is held at truth, so the number that comes out is
attributable to own-side imputation alone.

What gets imputed (the memo's "Naive-equivalent", i.e. Metamon's ``NaiveUsagePredictor``)
---------------------------------------------------------------------------------------
Per own mon, per decision, given the reveals visible in the protocol SO FAR:

* **moves** — revealed moves keep their true ``Move`` objects (and therefore their true
  remaining PP, which a replay can count); the remaining slots are filled with the highest
  Smogon ``move_priors`` candidates for the species, gated by the gen-3 learnset.
* **item** — held at truth once revealed; otherwise the top ``item_priors`` candidate.
* **EVs + nature** — NEVER revealed, so always imputed: the top ``spread_priors`` entry.
  IVs go to the competitive-standard 31s. Stats and ``max_hp`` are recomputed from the
  imputed spread, and ``current_hp`` is re-derived to hold the HP FRACTION fixed (the
  fraction is what the protocol shows; the integer is not).

Species are held at TRUTH. See "Limitations" below — that is a deliberate scope cut and it
makes the early-game number a LOWER bound.

Why mutate-encode-restore rather than re-materialize
----------------------------------------------------
The obs encoder is read-only over the battle, so the cheapest exact meter is to snapshot the
own-side fields, overwrite them with the imputed values, bump ``Gen3Battle._state_epoch`` so
the ``live_view`` memo rebuilds, encode, then put everything back and bump again. Both arms
encode with ``assembler=None`` (a full cold rebuild) so neither can inherit the other's
incremental cache, and the LIVE decision path is untouched — the probe's own play proceeds
on the true observation.

Reveal tracking mirrors what a REPLAY shows, not what poke-env knows
--------------------------------------------------------------------
:func:`track_own_reveals` reads the raw protocol lines and nothing else, binding nicknames to
species off the switch/drag/replace details the way
``main.search_dividend.determinize.used_moves_by_species`` does (this pool carries localized
nicknames, so a nickname-keyed read would name nothing). The move-call rules are the sharp
part and each has a unit test: Sleep Talk's callee IS in the user's set, Metronome's and
Mirror Move's are NOT, Struggle is never a set move.

Limitations (read these before quoting a number)
------------------------------------------------
1. **Species are held at truth.** A real replay does not know which six mons the imitated
   player brought until they appear — an unrevealed SLOT would be imputed too (Smogon
   ``teammate_priors``). That is a strictly larger error and it is largest exactly early,
   which is the direction that matters here, so the early-game figures below are a **LOWER
   bound**. The probe reports ``unrevealed_own_slots`` per phase so the size of the gap is
   at least visible.
2. **The ``|request|`` is held at truth**, because synthesizing one is the memo's Gap 1 and
   explicitly out of scope. That leaks into exactly one block: the reactive block's 12-dim
   ``active_req_moves`` group reads move slot *i* off the TRUE request and then looks it up
   in the (imputed) moveset, so a slot naming a move the imputation dropped resolves to
   neutral zeros. A real pipeline would synthesize the request from the imputed set and put
   the IMPUTED move's id there — still wrong, but wrong differently. That block is therefore
   printed with a ``*`` and must be read as a DIRECTION, not a magnitude. The honest
   companion figure the probe does report is ``active_set_mismatch_rate``: how often the
   imputed set differs from the true set on the ACTIVE mon at all.
3. **Ability is not imputed.** Our own ability is effectively prior-determined in gen 3
   (most species have one) and the memo scopes the risk to moves/item/spread.
4. **This is imputation error, not label error.** It says nothing about whether a human's
   ACTION is recoverable — that is the memo's Gap 3.
5. Absolute mean|Δ| is not comparable across blocks with different natural scales; the
   ``frac_dims`` column (how much of the block moved at all) is the scale-free read.

Run directly (no server needed; battles run in-process via the local BattleStream bridge):
    python src/agents/training/replay_imputation_probe.py [n_battles] [--json out.json]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from poke_env import AccountConfiguration
from poke_env.battle.move import Move, MoveSet
from poke_env.data.gen_data import GenData
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.stats import compute_raw_stats

from agents import gen3_data
from agents.battle.battle_event import from_clause_move_source
from agents.inference.player import Gen3Player
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import (
    OFFSET_EVENT_WINDOW,
    OFFSET_GLOBAL,
    OFFSET_OPP_TEAM,
    OFFSET_OUR_TEAM,
    OFFSET_PAIR_HISTORY,
    OFFSET_REACTIVE,
    POKEMON_FULL_DIM,
    REACTIVE_DIM,
    TEAM_SIZE,
)
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from main.search_dividend.determinize import to_id
from utils.bridge.local_battle_runner import run_local_battles
from utils.contention import warn_if_contended
from utils.team_loader import TeamLoader

BATTLE_FORMAT = "gen3ou"

# The three protocol tags by which a mon appears on the field. Same set
# `determinize.reveal_events` keys on — a mon on the field is a mon a spectator has seen.
_REVEAL_TAGS = ("switch", "drag", "replace")

# Move-call `[from]` sources whose CALLEE is drawn from the user's own moveset (so the callee
# is revealed) versus drawn from somewhere else (so it is not).
#
# gen3: Sleep Talk picks a random move from the USER's own set → the callee is revealed.
# Metronome picks from the whole move pool and Mirror Move copies the TARGET's last move →
# neither says anything about the user's set. `lockedmove` is the continuation marker for a
# multi-turn move the user genuinely selected, so it reveals.
_CALLERS_NOT_FROM_OWN_SET = frozenset({"metronome", "mirrormove", "sleeptalk"})
_CALLERS_REVEALING_CALLEE = frozenset({"sleeptalk"})

# Never a member of a declared set.
_NON_SET_MOVES = frozenset({"struggle"})

# Phase buckets. Turn 1-5 is the "before reveals" window the memo's §2.7 names as the place
# where imputation error and label informativeness are anti-correlated.
_PHASES: Tuple[Tuple[str, int, int], ...] = (
    ("turns_1_5", 1, 5),
    ("turns_6_15", 6, 15),
    ("turns_16plus", 16, 10_000),
)


def phase_of(turn: int) -> str:
    for name, lo, hi in _PHASES:
        if lo <= turn <= hi:
            return name
    return _PHASES[0][0]


# ---------------------------------------------------------------------------
# Reveal tracking — the pure, unit-tested half
# ---------------------------------------------------------------------------


@dataclass
class OwnSideReveals:
    """What a SPECTATOR has been shown about one side, so far.

    ``moves[species]`` is the set of move ids seen used by that species; ``items`` is the set
    of species whose held item has shown itself. Spreads never appear, so there is no field
    for them — that absence IS the model."""

    moves: Dict[str, Set[str]] = field(default_factory=dict)
    items: Set[str] = field(default_factory=set)
    seen: Set[str] = field(default_factory=set)  # species that have appeared on the field

    def moves_of(self, species: str) -> Set[str]:
        return self.moves.get(species, set())

    def item_shown(self, species: str) -> bool:
        return species in self.items


def _actor_species(token: str, nick_to_species: Dict[str, str]) -> Optional[str]:
    """Species behind a ``p1a: Nickname`` position token, via the switch-line binding."""
    if ":" not in token:
        return None
    return nick_to_species.get(token.split(":", 1)[1].strip())


def _mentions_item_cause(parts: Sequence[str]) -> bool:
    """True if any ``[from]`` clause on this line names an ITEM as the cause.

    Both spellings occur in the wild (`[from] item: Leftovers` from the bundled gen3 sim,
    `[from]item: Leftovers` from modern Showdown), so the check is whitespace-insensitive."""
    for tok in parts:
        t = str(tok).strip()
        if not t.startswith("[from]"):
            continue
        if t[len("[from]"):].strip().lower().startswith("item:"):
            return True
    return False


def track_own_reveals(lines: Sequence[str], our_tag: str) -> OwnSideReveals:
    """Fold the protocol into "what a spectator has been shown about ``our_tag``'s side".

    ``our_tag`` is ``"p1"`` / ``"p2"``. ``lines`` are raw ``|``-delimited protocol lines in
    order; only the prefix a decision has actually seen should be passed.

    The three rules, each pinned by a test in ``replay_imputation_probe_test.py``:

    * a **move** is revealed when the side USES it — with the move-call carve-outs above;
    * an **item** is revealed when it activates, is removed, or is named as a ``[from]``
      cause on any line whose actor is ours (Leftovers healing, a popped Berry, Trick,
      Knock Off);
    * a **spread** is never revealed, at all, ever.
    """
    out = OwnSideReveals()
    nick_to_species: Dict[str, str] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        p = line.split("|")
        if len(p) < 3:
            continue
        tag = p[1]
        pos = p[2]
        ours = pos.startswith(our_tag)

        if tag in _REVEAL_TAGS and len(p) >= 4:
            nick = pos.split(":", 1)[1].strip() if ":" in pos else ""
            sp = to_id(p[3].split(",")[0])
            if nick:
                nick_to_species[nick] = sp
            if ours:
                out.seen.add(sp)
            continue

        if not ours:
            # An opponent-actor line can still name OUR mon in an `[of]` clause (Knock Off:
            # `|-enditem|p1a: X|Leftovers|[from] move: Knock Off|[of] p2a: Y`) — but there the
            # ACTOR is the mon losing the item, so the actor test already covers it. Nothing
            # on an opponent-actor line reveals our side.
            continue

        sp = _actor_species(pos, nick_to_species)
        if sp is None:
            continue

        if tag == "move" and len(p) >= 4:
            caller = from_clause_move_source(p)
            used = to_id(p[3])
            if caller and caller != used and caller in _CALLERS_NOT_FROM_OWN_SET:
                # The CALLER is a real slot on the set; the callee only counts for Sleep Talk.
                out.moves.setdefault(sp, set()).add(caller)
                if caller in _CALLERS_REVEALING_CALLEE and used not in _NON_SET_MOVES:
                    out.moves.setdefault(sp, set()).add(used)
            elif used not in _NON_SET_MOVES:
                out.moves.setdefault(sp, set()).add(used)
                if caller and caller != used:
                    out.moves.setdefault(sp, set()).add(caller)
            continue

        if tag in ("-enditem", "-item"):
            out.items.add(sp)
            continue

        if tag == "-activate" and len(p) >= 4 and str(p[3]).lower().startswith("item:"):
            out.items.add(sp)
            continue

        if _mentions_item_cause(p):
            out.items.add(sp)

    return out


# ---------------------------------------------------------------------------
# The Naive-equivalent imputation
# ---------------------------------------------------------------------------


def impute_moves(species: str, revealed: Set[str], n_slots: int) -> List[str]:
    """The move ids a Naive prior would put in the ``n_slots`` slots not yet revealed.

    Ranked by Smogon ``move_priors`` and gated by the gen-3 learnset — the same hard legality
    gate the move-belief prior uses, so we can never impute a move the species cannot learn.

    ⚠️ This inherits Metamon's own ``score_pokemon`` caveat verbatim: a marginal per-move
    usage prior ranks *frequently-possible* above *frequently-used*, and their gen1 Tauros
    example is the canonical failure. Fixing it needs a candidate-set corpus (their
    ``ReplayPredictor``), which is out of scope for a meter — so the number this probe
    reports is the error of the NAIVE filler, and a better filler can only lower it."""
    if n_slots <= 0:
        return []
    prior = gen3_data.priors.moves(species)
    if not prior:
        return []
    # `None` from the facade means "no learnset entry" ⇒ NO constraint, never "no legal moves".
    legal = gen3_data.learnset.get_legal_moves(species)
    out: List[str] = []
    for mid, _p in sorted(prior.items(), key=lambda kv: (-kv[1], kv[0])):
        if mid in revealed or mid in out:
            continue
        if legal and mid not in legal:
            continue
        if gen3_data.moves.get(mid) is None:
            continue  # not in our movedex → the encoder would raise; skip rather than crash
        out.append(mid)
        if len(out) == n_slots:
            break
    return out


def impute_item(species: str) -> Optional[str]:
    """Top Smogon item prior for the species, or ``None`` when it has no entry."""
    prior = gen3_data.priors.items(species)
    if not prior:
        return None
    best = max(sorted(prior.items()), key=lambda kv: kv[1])[0]
    if best in ("", "nothing", "noitem", "none"):
        return None
    return best


def impute_spread(species: str) -> Optional[Tuple[str, List[int]]]:
    """``(nature, evs)`` of the top Smogon spread prior, or ``None`` with no entry."""
    spreads = gen3_data.priors.spreads(species)
    if not spreads:
        return None
    nature, evs, _w = max(spreads, key=lambda r: r[2])
    return str(nature).lower(), [int(e) for e in evs]


@dataclass
class _MonSnapshot:
    """Everything :func:`apply_imputation` may overwrite on one ``Pokemon``, so the restore
    is total rather than best-effort."""

    mon: Any
    moves: Any
    item: Optional[str]
    consumed_item: Optional[str]
    ivs: Optional[list]
    evs: Optional[list]
    nature: Optional[str]
    stats: Dict[str, Optional[int]]
    max_hp: Optional[int]
    current_hp: Optional[int]


def _snapshot(mon: Any) -> _MonSnapshot:
    return _MonSnapshot(
        mon=mon,
        moves=mon._moves,
        item=mon._item,
        consumed_item=mon._consumed_item,
        ivs=list(mon._ivs) if mon._ivs is not None else None,
        evs=list(mon._evs) if mon._evs is not None else None,
        nature=mon._nature,
        stats=dict(mon._stats),
        max_hp=mon._max_hp,
        current_hp=mon._current_hp,
    )


def _restore(snap: _MonSnapshot) -> None:
    m = snap.mon
    m._moves = snap.moves
    m._item = snap.item
    m._consumed_item = snap.consumed_item
    m._ivs = snap.ivs
    m._evs = snap.evs
    m._nature = snap.nature
    m._stats = snap.stats
    m._max_hp = snap.max_hp
    m._current_hp = snap.current_hp


@dataclass
class ImputationReport:
    """What the imputation actually changed, per decision — the provenance beside the Δ."""

    n_mons: int = 0
    n_move_slots_imputed: int = 0
    n_move_slots_total: int = 0
    n_items_imputed: int = 0
    n_spreads_imputed: int = 0
    active_set_mismatch: bool = False
    unrevealed_own_slots: int = 0


def apply_imputation(battle: Any, reveals: OwnSideReveals) -> Tuple[List[_MonSnapshot], ImputationReport]:
    """Overwrite our own side's not-yet-revealed details with their prior top-1 candidates.

    Returns ``(snapshots, report)``; the caller MUST pass the snapshots to
    :func:`undo_imputation` (and both sides bump the battle's state epoch, so the ``live_view``
    memo cannot serve a stale board across the swap)."""
    gen_data = GenData.from_gen(3)
    snaps: List[_MonSnapshot] = []
    rep = ImputationReport()
    active = battle.active_pokemon

    for mon in battle.team.values():
        sp = to_id(mon.species)
        snaps.append(_snapshot(mon))
        rep.n_mons += 1
        if sp not in reveals.seen:
            rep.unrevealed_own_slots += 1

        # --- moves -----------------------------------------------------------------
        true_ids = set(mon.moves.keys())
        # A typed Hidden Power is keyed typed on our own mons; the protocol only ever says
        # "Hidden Power", so match it the way `determinize.norm_move` does.
        shown = set()
        for rid in reveals.moves_of(sp):
            if rid == "hiddenpower":
                shown |= {m for m in true_ids if m.startswith("hiddenpower")}
            elif rid in true_ids:
                shown.add(rid)
            else:
                shown.add(rid)
        kept = {mid: mv for mid, mv in mon.moves.items() if mid in shown}
        n_slots = max(0, len(true_ids) - len(kept))
        filler = impute_moves(sp, {m for m in shown} | set(kept), n_slots)
        rep.n_move_slots_total += len(true_ids)
        rep.n_move_slots_imputed += n_slots
        new_moves = MoveSet(dict(kept))
        for mid in filler:
            mv = Move(Move.retrieve_id(mid), gen=3, raw_id=mid)
            new_moves[mv.id] = mv
        mon._moves = new_moves
        if active is not None and mon is active:
            rep.active_set_mismatch = set(new_moves.moves.keys()) != true_ids

        # --- item ------------------------------------------------------------------
        if not reveals.item_shown(sp):
            guess = impute_item(sp)
            mon._item = guess
            mon._consumed_item = None
            rep.n_items_imputed += 1

        # --- spread (never revealed) ------------------------------------------------
        guess_spread = impute_spread(sp)
        if guess_spread is not None:
            nature, evs = guess_spread
            ivs = [31] * 6
            frac = float(mon.current_hp_fraction)
            mon._evs = evs
            mon._ivs = ivs
            mon._nature = nature
            stats = compute_raw_stats(mon._species, evs, ivs, mon._level, nature, gen_data)
            mon._stats = dict(zip(["hp", "atk", "def", "spa", "spd", "spe"], stats))
            new_max = int(stats[0])
            mon._max_hp = new_max
            mon._stats["hp"] = new_max
            # The protocol shows a FRACTION, so that is what is held fixed; the integer HP
            # follows the imputed max_hp. The rounding residual is a real (small) error and
            # is deliberately left in the measurement rather than papered over.
            if snaps[-1].current_hp is not None:
                mon._current_hp = int(round(frac * new_max))
            rep.n_spreads_imputed += 1

    battle._state_epoch += 1
    return snaps, rep


def undo_imputation(battle: Any, snaps: Sequence[_MonSnapshot]) -> None:
    for s in snaps:
        _restore(s)
    battle._state_epoch += 1


# ---------------------------------------------------------------------------
# Block accounting
# ---------------------------------------------------------------------------


def obs_blocks(encoder: Any) -> List[Tuple[str, int, int]]:
    """``(name, start, end)`` for every top-level obs block, plus the our-team sub-blocks.

    Top-level ranges come from ``get_layout()['parts']``, except that ``reactive`` is taken as
    ``start + dim``: the layout's ``reactive.end`` is ``self.dimension``, which spans the two
    blocks that were later appended after it. Sub-block ranges are the per-mon layout, unioned
    across the six our-team slots — that is what answers "WHICH part of our team block"."""
    layout = encoder.get_layout()
    parts = layout["parts"]
    blocks: List[Tuple[str, int, int]] = [
        ("our_team", OFFSET_OUR_TEAM, OFFSET_OPP_TEAM),
        ("opp_team", OFFSET_OPP_TEAM, int(parts["context"]["start"])),
        ("context", int(parts["context"]["start"]), OFFSET_GLOBAL),
        ("global", OFFSET_GLOBAL, OFFSET_REACTIVE),
        ("reactive", OFFSET_REACTIVE, OFFSET_REACTIVE + REACTIVE_DIM),
        ("pair_history", OFFSET_PAIR_HISTORY, OFFSET_EVENT_WINDOW),
        ("event_window", OFFSET_EVENT_WINDOW, int(layout["total_dim"])),
    ]
    return blocks


def our_team_subblocks(encoder: Any) -> Dict[str, np.ndarray]:
    """``{sub_block_name: index array}`` over the whole obs, for the our-team slots only.

    Every dim of the our-team block lands in exactly one bucket: the named sub-blocks the
    per-mon layout declares, plus an explicit ``our_team.unnamed`` residual for the per-slot
    dims the layout does not name (the sleep-belief / recency / last-action / trapped /
    active tail). A residual bucket rather than a silent gap, so the sub-block rows are
    guaranteed to partition the parent and cannot quietly lose an error."""
    pk = encoder.get_layout()["pokemon"]
    out: Dict[str, np.ndarray] = {}
    claimed: Set[int] = set()
    for name, spec in pk.items():
        if not isinstance(spec, dict):
            continue  # scalars like `pokemon_vector_dim` are metadata, not ranges
        off, dim = int(spec["offset"]), int(spec["dim"])
        idx: List[int] = []
        for slot in range(TEAM_SIZE):
            base = OFFSET_OUR_TEAM + slot * POKEMON_FULL_DIM + off
            idx.extend(range(base, base + dim))
        out[f"our_team.{name}"] = np.asarray(idx, dtype=np.int64)
        claimed.update(idx)
    residual = [i for i in range(OFFSET_OUR_TEAM, OFFSET_OPP_TEAM) if i not in claimed]
    if residual:
        out["our_team.unnamed"] = np.asarray(residual, dtype=np.int64)
    return out


def reactive_subblocks(encoder: Any) -> Dict[str, np.ndarray]:
    """``{sub_block_name: index array}`` for the reactive block.

    Split out because the two halves mean DIFFERENT things here. The five board scalars are
    opponent/log-derived and cannot move under own-side imputation. The 12-dim
    ``active_req_moves`` group is the one place where holding the ``|request|`` at truth
    (limitation 2) is visible: the request still names the TRUE move in slot *i*, the imputed
    moveset may not contain it, and ``_request_slot_moves`` then resolves the slot to ``None``
    → neutral zeros. A real replay pipeline would synthesize the request FROM the imputed set,
    so slot *i* would carry the imputed move's id instead of a zero. The slot still differs
    from truth either way, so the direction is right; the magnitude is NOT a measurement and
    must be reported as an artifact of the scope cut."""
    lay = encoder.get_layout()["reactive_layout"]
    out: Dict[str, np.ndarray] = {}
    claimed: Set[int] = set()
    for name, spec in lay.items():
        if not isinstance(spec, dict):
            continue
        off, dim = int(spec["offset"]), int(spec["dim"])
        idx = list(range(OFFSET_REACTIVE + off, OFFSET_REACTIVE + off + dim))
        out[f"reactive.{name}"] = np.asarray(idx, dtype=np.int64)
        claimed.update(idx)
    residual = [i for i in range(OFFSET_REACTIVE, OFFSET_REACTIVE + REACTIVE_DIM)
                if i not in claimed]
    if residual:
        out["reactive.unnamed"] = np.asarray(residual, dtype=np.int64)
    return out


@dataclass
class _Acc:
    """|Δ| accumulator for one (block, phase) cell."""

    n_decisions: int = 0
    sum_abs: float = 0.0        # Σ over decisions of mean|Δ| within the block
    max_abs: float = 0.0
    sum_frac: float = 0.0       # Σ over decisions of (dims changed / block width)
    sum_rel_l2: float = 0.0     # Σ over decisions of ‖Δ‖₂ / ‖true‖₂
    n_touched: int = 0          # decisions where ANY dim in the block moved

    def add(self, d: np.ndarray, true: np.ndarray) -> None:
        self.n_decisions += 1
        a = np.abs(d)
        self.sum_abs += float(a.mean()) if a.size else 0.0
        if a.size:
            self.max_abs = max(self.max_abs, float(a.max()))
        changed = int(np.count_nonzero(d))
        self.sum_frac += changed / a.size if a.size else 0.0
        # Scale-free companion to mean|Δ|: blocks carry raw embedding ids (hundreds) beside
        # normalized scalars (0-1), so an absolute mean is not comparable ACROSS rows. A zero
        # true-norm block (nothing populated) can only have a zero Δ here, so 0/0 → 0.
        den = float(np.linalg.norm(true))
        self.sum_rel_l2 += (float(np.linalg.norm(d)) / den) if den > 0.0 else 0.0
        if changed:
            self.n_touched += 1

    def summary(self) -> Dict[str, float]:
        n = max(1, self.n_decisions)
        return {
            "n_decisions": self.n_decisions,
            "mean_abs": self.sum_abs / n,
            "max_abs": self.max_abs,
            "frac_dims": self.sum_frac / n,
            "rel_l2": self.sum_rel_l2 / n,
            "frac_decisions_touched": self.n_touched / n,
        }


class ImputationMeter:
    """Accumulates the per-(block, phase) error over every decision of every battle."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.blocks: Dict[str, np.ndarray] = {
            name: np.arange(start, end, dtype=np.int64)
            for name, start, end in obs_blocks(encoder)
        }
        self.blocks.update(our_team_subblocks(encoder))
        self.blocks.update(reactive_subblocks(encoder))
        self.blocks["ALL"] = np.arange(int(encoder.get_layout()["total_dim"]), dtype=np.int64)
        self.cells: Dict[Tuple[str, str], _Acc] = {}
        self.phase_meta: Dict[str, Dict[str, float]] = {}

    def add(self, true_obs: np.ndarray, imputed_obs: np.ndarray, turn: int,
            rep: ImputationReport) -> None:
        ph = phase_of(turn)
        delta = imputed_obs - true_obs
        for name, idx in self.blocks.items():
            self.cells.setdefault((name, ph), _Acc()).add(delta[idx], true_obs[idx])
        m = self.phase_meta.setdefault(ph, {
            "n": 0.0, "move_slots_imputed": 0.0, "move_slots_total": 0.0,
            "items_imputed": 0.0, "spreads_imputed": 0.0,
            "active_set_mismatch": 0.0, "unrevealed_own_slots": 0.0,
        })
        m["n"] += 1
        m["move_slots_imputed"] += rep.n_move_slots_imputed
        m["move_slots_total"] += rep.n_move_slots_total
        m["items_imputed"] += rep.n_items_imputed
        m["spreads_imputed"] += rep.n_spreads_imputed
        m["active_set_mismatch"] += 1.0 if rep.active_set_mismatch else 0.0
        m["unrevealed_own_slots"] += rep.unrevealed_own_slots

    def result(self) -> Dict[str, Any]:
        phases = [p for p, _lo, _hi in _PHASES]
        out: Dict[str, Any] = {"phases": phases, "blocks": {}, "coverage": {}}
        for name in self.blocks:
            out["blocks"][name] = {
                ph: self.cells[(name, ph)].summary()
                for ph in phases if (name, ph) in self.cells
            }
        for ph, m in self.phase_meta.items():
            n = max(1.0, m["n"])
            out["coverage"][ph] = {
                "n_decisions": int(m["n"]),
                "frac_move_slots_imputed": (m["move_slots_imputed"] / m["move_slots_total"]
                                            if m["move_slots_total"] else 0.0),
                "mean_items_imputed_per_decision": m["items_imputed"] / n,
                "mean_spreads_imputed_per_decision": m["spreads_imputed"] / n,
                "active_set_mismatch_rate": m["active_set_mismatch"] / n,
                "mean_unrevealed_own_slots": m["unrevealed_own_slots"] / n,
            }
        return out


# ---------------------------------------------------------------------------
# The probe player
# ---------------------------------------------------------------------------


class ImputationProbePlayer(Gen3Player):
    """Plays a seeded-random policy on the TRUE obs and measures the imputed one beside it.

    The live decision is made on truth — the imputation is a read-only side measurement, so
    the trajectory this probe walks is the same one the true-obs agent would walk. That is
    deliberate: we want the error measured on the state DISTRIBUTION a real player visits,
    not on one an imputation-confused player would."""

    def __init__(self, *, meter: ImputationMeter, rng_seed: int, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._meter = meter
        self._rng = np.random.RandomState(rng_seed)
        self._lines: Dict[str, List[str]] = {}
        self.n_decisions = 0

    async def _handle_battle_message(self, split_messages):
        """Archive the raw protocol exactly as a spectator would see it, BEFORE poke-env
        parses it into state — the canonical fuzz-test interception point."""
        tag = split_messages[0][0].lstrip(">")
        buf = self._lines.setdefault(tag, [])
        for msg in split_messages[1:]:
            if len(msg) >= 2:
                buf.append("|".join(msg))
        await super()._handle_battle_message(split_messages)

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "IMPUTATION_PROBE_STALL")
        if forfeit:
            return forfeit
        obs_dict = self.embed_battle(battle)
        mask = obs_dict["action_mask"]
        if int(mask.sum()) == 0:
            return self.choose_default_move()

        self._measure(battle)

        idx = int(self._rng.choice(np.flatnonzero(mask)))
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

    def _measure(self, battle) -> None:
        tracker = self._get_tracker(battle)
        ctx = tracker.last_ctx
        legal = ctx.legal if ctx is not None else None
        enc = self.observation_encoder
        kw = dict(hp_tracker=tracker.hidden_power_tracker, legal=legal,
                  progress_clock=tracker.progress_clock, recency=tracker.recency,
                  pair_history=tracker.pair_history, event_window=tracker.event_window,
                  assembler=None)

        # Arm A — TRUTH. Re-encoded cold (not reused from embed_battle) so both arms come
        # out of the same code path and a cache can never explain a difference.
        true_obs = np.asarray(enc.encode(battle, **kw), dtype=np.float64)

        tag = battle.strict_view().battle_tag
        reveals = track_own_reveals(self._lines.get(tag, ()), battle.player_role)

        # Arm B — IMPUTED.
        snaps, rep = apply_imputation(battle, reveals)
        try:
            imputed_obs = np.asarray(enc.encode(battle, **kw), dtype=np.float64)
        finally:
            undo_imputation(battle, snaps)

        # THE INTEGRITY GATE. The meter mutates the live battle and puts it back; if the
        # restore ever leaked, every LATER decision's "truth" would silently be a previous
        # decision's imputation and the whole measurement would be of nothing in particular.
        # So it is checked, every decision, rather than believed — this is one extra ~1 ms
        # cold encode against a multi-second battle.
        if not np.array_equal(np.asarray(enc.encode(battle, **kw), dtype=np.float64),
                              true_obs):
            raise AssertionError(
                f"imputation restore LEAKED at turn {battle.turn} of "
                f"{battle.strict_view().battle_tag}: the re-encoded truth differs from the "
                f"pre-imputation truth, so every later decision would be contaminated"
            )

        self._meter.add(true_obs, imputed_obs, int(battle.turn), rep)
        self.n_decisions += 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_probe(n_battles: int, *, impl: str = "rust") -> Dict[str, Any]:
    """Play ``n_battles`` REPRODUCIBLE bridge battles and return the meter's result.

    Reproducibility follows ``obs_roundtrip_fuzz_test.record_fixture_battle`` exactly — teams
    pinned by key, a per-player RNG on each side (never the global module), and a fixed sim
    seed — because a meter you cannot re-run is a meter whose number cannot be checked."""
    from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer

    encoder = get_observation_encoder(load_mappings())
    meter = ImputationMeter(encoder)
    pool = TeamLoader().get_all_teams()
    assert pool, "no gen3ou teams under data/teams"

    for key in range(n_battles):
        t1, t2 = pool[key % len(pool)], pool[(key + 1) % len(pool)]
        probe = ImputationProbePlayer(
            meter=meter, rng_seed=1000 + key, battle_format=BATTLE_FORMAT, team=t1,
            account_configuration=AccountConfiguration(f"Imp{key}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=1)
        probe.observation_encoder = encoder
        opp = SeededRandomPlayer(
            rng_seed=2000 + key, battle_format=BATTLE_FORMAT, team=t2,
            account_configuration=AccountConfiguration(f"ImpO{key}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=1)
        await run_local_battles(probe, opp, 1,
                                seed=[11 + key, 22 + key, 33 + key, 44 + key], impl=impl)
        print(f"  battle {key + 1}/{n_battles}: {probe.n_decisions} decisions measured",
              flush=True)

    res = meter.result()
    res["n_battles"] = n_battles
    return res


_HEADLINE = ("ALL", "our_team", "our_team.moves", "our_team.items", "our_team.spread",
             "opp_team", "context", "global", "reactive*", "pair_history", "event_window")

# Blocks whose number is an ARTIFACT of a scope cut, not a measurement (see limitation 2 and
# `reactive_subblocks`). Marked with a `*` everywhere they are printed so a reader cannot
# lift the figure out of the table without the caveat travelling with it.
_ARTIFACT_BLOCKS = frozenset({"reactive", "reactive.active_req_moves"})


def _row(name: str, cells: Dict[str, Any], phases: Sequence[str]) -> str:
    label = f"{name}*" if name in _ARTIFACT_BLOCKS else name
    row = f"{label:<26}"
    for p in phases:
        c = cells.get(p)
        row += (f"{c['mean_abs']:>11.5f}{c['max_abs']:>10.2f}{c['frac_dims']:>10.4f}"
                f"{c['rel_l2']:>9.4f}" if c else f"{'-':>40}")
    return row


def print_report(res: Dict[str, Any]) -> None:
    phases = res["phases"]
    print("\n=== OWN-SIDE IMPUTATION ERROR, per obs block x phase ===")
    print(f"{'block':<26}" + "".join(f"{p:>40}" for p in phases))
    print(f"{'':<26}" + "".join(f"{'mean|d|    max|d| frac_dims   relL2':>40}" for _ in phases))
    for name in _HEADLINE:
        key = name.rstrip("*")
        print(_row(key, res["blocks"].get(key, {}), phases))

    print("\n=== sub-blocks ===")
    shown = {n.rstrip("*") for n in _HEADLINE}
    for name in sorted(res["blocks"]):
        if "." not in name or name in shown:
            continue
        print(_row(name, res["blocks"][name], phases))
    print("\n  * ARTIFACT, not a measurement: the |request| is held at TRUTH (limitation 2),")
    print("    so a request slot naming a move the imputed set lacks resolves to neutral")
    print("    zeros. A synthesized request would put the IMPUTED move's id there instead —")
    print("    still != truth, but a different number. Read the direction, not the size.")

    print("\n=== what was imputed (coverage) ===")
    keys = ["n_decisions", "frac_move_slots_imputed", "mean_items_imputed_per_decision",
            "mean_spreads_imputed_per_decision", "active_set_mismatch_rate",
            "mean_unrevealed_own_slots"]
    print(f"{'':<36}" + "".join(f"{p:>16}" for p in phases))
    for k in keys:
        row = f"{k:<36}"
        for p in phases:
            v = res["coverage"].get(p, {}).get(k)
            row += f"{v:>16.4f}" if isinstance(v, float) else f"{v if v is not None else '-':>16}"
        print(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("n_battles", nargs="?", type=int, default=20)
    ap.add_argument("--impl", default="rust", choices=("node", "rust"))
    ap.add_argument("--json", default=None, help="write the full result dict here")
    args = ap.parse_args(list(argv) if argv is not None else None)

    warn_if_contended()
    print(f"Own-side imputation meter — {args.n_battles} reproducible bridge battles "
          f"(impl={args.impl})", flush=True)
    res = asyncio.run(run_probe(args.n_battles, impl=args.impl))
    print_report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
