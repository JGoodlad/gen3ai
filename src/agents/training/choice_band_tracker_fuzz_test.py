"""Escalating bridge fuzz for :class:`ChoiceBandTracker` (System A).

Runs real ``gen3ou`` battles in-process via the local BattleStream bridge (no server). The fuzz
*tracker* player (p1) instantiates a ``ChoiceBandTracker`` and feeds it the real battle's events —
exactly as ``EpisodeTracker.record()`` will once wired in — so the tracker is exercised through the
production parse → event-log → LiveView pipeline WITHOUT being wired into the obs encoder (the
Step-1 no-op contract).

Two layers of validation at every p1 decision, for every revealed opponent mon:

  1. **CONTRACT** — recompute the expected ``p_cb`` from an INDEPENDENT raw-protocol reconstruction
     of the per-stint distinct *selected* moves (a second parser, distinct from the tracker's
     BattleEvent-log walk) + poke-env's revealed item state, and assert the tracker matches. A bug
     in the tracker's stint walk produces a different distinct-move count and is caught here.
  2. **GROUND-TRUTH INVARIANTS** (curated battles, where the true items are known from the team
     spec + the protocol's item-change lines):
       * a mon TRULY holding Choice Band must NEVER read ``p_cb == 0`` (never false-eliminated);
       * a mon truly NOT holding Choice Band must NEVER read ``p_cb == 1`` (never false-asserted).
     These use the true item (independent of the tracker) — the load-bearing checks.

Plus a **coverage report**: how many times each required edge case fired (a zero-fire edge case is
a coverage FAILURE). Curated scripted matchups drive the hard edges deterministically; a random
full-pool sweep adds breadth across the whole species/item space.

Run (no server needed; in-process via the local bridge). Argument is a TIME BUDGET in MINUTES:
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/choice_band_tracker_fuzz_test.py [minutes]   # default 1
Tiers: 1 (dev) / 5 (pre-merge) / 20 (pre-ship; MUST hit zero violations + full coverage).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents import gen3_data
from agents.battle.gen3_battle import Gen3Battle
from agents.training.choice_band_tracker import CHOICE_BAND, ChoiceBandTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Required edge cases — each MUST fire at least once per tier (zero-fire == coverage failure).
REQUIRED_EDGES = (
    "cb_locked_one_move",        # a true CB holder stayed locked into 1 move; p_cb never wrongly 0
    "cb_first_move_status",      # a true CB holder whose locked (first) move is a STATUS move
    "noncb_revealed_leftovers",  # passive Leftovers/heal item reveal → p_cb 0
    "noncb_revealed_berry",      # Berry consumed (-enditem [eat]) → p_cb 0 (consumed_item set)
    "two_distinct_one_stint",    # 2 distinct selected moves in one stint → p_cb 0
    "two_distinct_across_switch",# 2 distinct moves split across a switch → NOT eliminated
    "sleep_talk_delegation",     # Sleep Talk → called move; the called move not a 2nd selection
    "struggle_excluded",         # Struggle (0 PP) not counted as a selection
    "encore_no_false_elim",      # an Encored mon (forced repeat) never false-eliminated
    "thief_knockoff_itemless",   # Knock Off / Thief → opponent itemless → p_cb 0
    "trick_hands_cb",            # Trick hands the opp Choice Band → p_cb rises to 1
    "trick_swaps_cb_away",       # Trick swaps a CB holder's band away → p_cb 0
)


# ===========================================================================
# Curated teams — known items (the ground-truth dict). Validated legal gen3ou.
# ===========================================================================

# p2 = the OBSERVED side (the one whose Choice-Band belief we infer).
OBSERVED_TEAM = """\
Aerodactyl @ Choice Band
Ability: Pressure
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Double-Edge
- Hidden Power [Flying]

Heracross @ Choice Band
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Swords Dance
- Megahorn
- Rock Slide
- Earthquake

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Pursuit

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 240 SpD
Careful Nature
- Rest
- Sleep Talk
- Body Slam
- Earthquake

Metagross @ Sitrus Berry
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Alakazam @ Choice Band
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Trick
- Psychic
- Fire Punch
- Ice Punch
"""

# Same six mons, reordered to LEAD with Snorlax — so its Rest → sleep → Sleep Talk delegation
# (otherwise buried in slot 4) reliably surfaces. Rotated with OBSERVED_TEAM per battle.
OBSERVED_TEAM_SNORLAX_LEAD = """\
Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 240 SpD
Careful Nature
- Rest
- Sleep Talk
- Body Slam
- Earthquake

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Pursuit

Heracross @ Choice Band
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Swords Dance
- Megahorn
- Rock Slide
- Earthquake

Metagross @ Sitrus Berry
Ability: Clear Body
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Rock Slide
- Explosion

Aerodactyl @ Choice Band
Ability: Pressure
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Double-Edge
- Hidden Power [Flying]

Alakazam @ Choice Band
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Trick
- Psychic
- Fire Punch
- Ice Punch
"""

# Item the OBSERVED mons START with (id form). The dynamic true item is this, adjusted by the
# protocol's item-change lines (Trick / Knock Off / Thief), reconstructed in _OppGroundTruth.
OBSERVED_START_ITEM = {
    "aerodactyl": CHOICE_BAND,
    "heracross": CHOICE_BAND,
    "tyranitar": "leftovers",
    "snorlax": "leftovers",
    "metagross": "sitrusberry",
    "alakazam": CHOICE_BAND,
}

# p1 = the TRACKER + DISRUPTOR side. Carries Knock Off / Thief / Trick(+CB) / Encore so it can
# drive the item-change + forced-move edge cases against p2.
TRACKER_TEAM = """\
Hariyama @ Leftovers
Ability: Guts
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Knock Off
- Brick Break
- Rock Slide
- Earthquake

Gengar
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thief
- Shadow Ball
- Thunderbolt
- Ice Punch

Alakazam @ Choice Band
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Trick
- Psychic
- Fire Punch
- Ice Punch

Azumarill @ Leftovers
Ability: Huge Power
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Encore
- Return
- Hidden Power [Ghost]
- Surf

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Calm Mind
- Surf
- Rest
- Sleep Talk

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Thunder Wave
- Ice Beam
"""

# A dedicated Struggle matchup: a CB mon locked into a single low-PP move (Hyper Beam, 5 PP) vs a
# passive Pressure wall that drains its PP without KOing it → forced Struggle.
STRUGGLE_OBSERVED_TEAM = """\
Snorlax @ Choice Band
Ability: Thick Fat
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Hyper Beam

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Thunder Wave
- Ice Beam
"""
STRUGGLE_START_ITEM = {"snorlax": CHOICE_BAND, "blissey": "leftovers"}

STRUGGLE_TRACKER_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Calm Mind
- Surf
- Rest
- Sleep Talk

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 232 Def / 24 Spe
Impish Nature
- Spikes
- Drill Peck
- Roar
- Protect
"""


# ===========================================================================
# Independent (raw-protocol) reconstruction of the opponent's belief inputs
# ===========================================================================

_STRUGGLE_ID = "struggle"

# Moves that CALL another move (so the called move on a |move| line is not a free selection). This
# is a DELIBERATELY-different delegation predicate from the tracker's "[from] names a different
# move" rule — both must agree on real battles, but a regression in either (e.g. the Pursuit-on-
# switch `[from] Pursuit` bug, which tags the move with ITSELF) makes them disagree so the contract
# check catches it instead of sharing the blind spot.
_DELEGATOR_CALLERS = frozenset({
    "sleeptalk", "metronome", "mirrormove", "assist", "naturepower", "snatch", "magiccoat",
    "copycat",
})


def _from_move_sources(sm) -> list:
    """Move-ids named by ``[from] <move>`` / ``[from]move: <move>`` clauses on a |move| line
    (item:/ability: provenance excluded)."""
    out = []
    for t in sm[4:]:
        t = t.strip()
        if not t.startswith("[from]"):
            continue
        s = t[len("[from]"):].strip()
        low = s.lower()
        if low.startswith(("item:", "ability:")):
            continue
        if low.startswith("move:"):
            s = s.split(":", 1)[-1].strip()
        out.append(to_id_str(s))
    return out


@dataclass
class _MonTruth:
    start_item: Optional[str]
    held: Optional[str]            # current revealed/true held item id (None == itemless/unknown)
    consumed: bool = False         # an item was consumed/removed from it
    item_revealed: bool = False    # the protocol disclosed a held item (so 'held' is trustworthy)
    stint_moves: Set[str] = field(default_factory=set)  # distinct selected moves THIS stint
    ever_moves: Set[str] = field(default_factory=set)   # distinct selected moves over the battle
    first_stint_move: Optional[str] = None


class _OppGroundTruth:
    """Per-battle, independent reconstruction of each opponent mon's belief inputs from the RAW
    protocol stream — a second code path that does NOT touch the tracker or its BattleEvent log.

    Tracks, per opponent species: the current revealed/true held item (start item adjusted by
    item-change lines), whether an item was consumed, and the distinct *selected* moves in the
    current stint (reset on switch/drag and on any item-change line, excluding delegated executions
    and Struggle). Also flips per-battle coverage flags as the relevant lines stream by.
    """

    def __init__(self, opp_role: str, start_items: Dict[str, str], coverage: Counter):
        self.opp_role = opp_role  # 'p1' or 'p2' — the OPPONENT's side from p1's perspective
        self.coverage = coverage
        self.mons: Dict[str, _MonTruth] = {}
        self._start_items = dict(start_items)
        self._active: Optional[str] = None
        self._nick_to_species: Dict[str, str] = {}

    def _mon(self, species: str) -> _MonTruth:
        if species not in self.mons:
            self.mons[species] = _MonTruth(
                start_item=self._start_items.get(species),
                held=self._start_items.get(species),
            )
        return self.mons[species]

    @staticmethod
    def _ident_side(ident: str) -> str:
        return ident[:2] if len(ident) >= 2 else ""

    def _species_of(self, ident: str) -> str:
        # ident like 'p2a: Nick' → resolve nickname→species (filled from switch detail lines).
        nick = ident.split(": ", 1)[-1]
        return self._nick_to_species.get(nick, to_id_str(nick))

    def ingest(self, sm: Tuple[str, ...]) -> None:
        """Consume one split protocol line (tuple of fields)."""
        if len(sm) < 2:
            return
        kw = sm[1]
        if kw in ("switch", "drag") and len(sm) >= 4:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            nick = ident.split(": ", 1)[-1]
            species = to_id_str(sm[3].split(",")[0])
            self._nick_to_species[nick] = species
            self._active = species
            m = self._mon(species)
            m.stint_moves = set()                 # new stint
            if kw == "drag":
                self.coverage["_drag_seen"] += 1
        elif kw == "move" and len(sm) >= 4:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            species = self._species_of(ident)
            m = self._mon(species)
            move_id = to_id_str(sm[3])
            if move_id == _STRUGGLE_ID:
                self.coverage["struggle_excluded"] += 1
                return
            # Independent delegation predicate: the move was CALLED iff a [from] source is a known
            # caller (Sleep Talk / Snatch / …). Pursuit-on-switch tags itself ([from] Pursuit, not a
            # caller) → a real selection; lockedmove continuations ([from] lockedmove) → a duplicate
            # of the already-counted selection turn, harmless to count.
            from_srcs = _from_move_sources(sm)
            if any(s in _DELEGATOR_CALLERS for s in from_srcs):
                if "sleeptalk" in from_srcs:
                    self.coverage["sleep_talk_delegation"] += 1
                return
            before = len(m.stint_moves)
            if m.first_stint_move is None or not m.stint_moves:
                if not m.stint_moves:
                    m.first_stint_move = move_id
            m.stint_moves.add(move_id)
            m.ever_moves.add(move_id)
            if len(m.stint_moves) >= 2 and before < 2:
                self.coverage["two_distinct_one_stint"] += 1
        elif kw == "-enditem" and len(sm) >= 4:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            species = self._species_of(ident)
            m = self._mon(species)
            item = to_id_str(sm[3])
            suffix = " ".join(sm[4:]).lower()
            had_cb = m.held == CHOICE_BAND
            m.consumed = True
            m.held = None
            m.item_revealed = True
            m.stint_moves = set()  # item change resets the move-window
            if "[eat]" in suffix or "berry" in item:
                self.coverage["noncb_revealed_berry"] += 1
            if "knock off" in suffix or "thief" in suffix:
                self.coverage["thief_knockoff_itemless"] += 1
            if "trick" in suffix and had_cb:
                self.coverage["trick_swaps_cb_away"] += 1
        elif kw == "-item" and len(sm) >= 4:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            species = self._species_of(ident)
            m = self._mon(species)
            item = to_id_str(sm[3])
            suffix = " ".join(sm[4:]).lower()
            had_cb = m.held == CHOICE_BAND
            m.held = item
            m.item_revealed = True
            m.consumed = False
            m.stint_moves = set()  # item change resets the move-window
            if "trick" in suffix or "switcheroo" in suffix:
                if item == CHOICE_BAND:
                    self.coverage["trick_hands_cb"] += 1
                elif had_cb:
                    self.coverage["trick_swaps_cb_away"] += 1
        elif kw == "-heal" and len(sm) >= 5:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            species = self._species_of(ident)
            suffix = " ".join(sm[4:]).lower()
            if "[from] item:" in suffix or "[from]item:" in suffix:
                item = to_id_str(suffix.split("item:")[-1])
                m = self._mon(species)
                # poke-env sets mon.item from a heal reveal ONLY for non-berry/herb items.
                if "berry" not in item and "herb" not in item:
                    m.held = item
                    m.item_revealed = True
                    self.coverage["noncb_revealed_leftovers"] += 1
        elif kw in ("-start", "-activate") and len(sm) >= 4:
            ident = sm[2]
            if self._ident_side(ident) != self.opp_role:
                return
            if "encore" in " ".join(sm[3:]).lower():
                self.coverage["encore_no_false_elim"] += 1


def expected_pcb(held: Optional[str], consumed: bool, n_distinct: int, prior: float) -> float:
    """The ChoiceBandTracker contract, recomputed independently for the assertion."""
    if held == CHOICE_BAND:
        return 1.0
    if held is not None:
        return 0.0
    if consumed:
        return 0.0
    if n_distinct >= 2:
        return 0.0
    return prior


# ===========================================================================
# Players
# ===========================================================================

@dataclass
class FuzzStats:
    battles: int = 0
    decisions: int = 0
    contract_checks: int = 0
    invariant_checks: int = 0
    violations: List[str] = field(default_factory=list)
    species_seen: Set[str] = field(default_factory=set)
    items_seen: Set[str] = field(default_factory=set)
    coverage: Counter = field(default_factory=Counter)


class CBTrackerPlayer(Player):
    """p1 — runs a ChoiceBandTracker per battle, asserts the contract + ground-truth invariants
    against an independent raw-protocol reconstruction, and (when ``disrupt``) plays Knock Off /
    Thief / Trick / Encore against the opponent to drive item-change + forced-move edges."""

    def __init__(self, *args, stats: FuzzStats, start_items: Dict[str, str],
                 disrupt: bool = False, curated: bool = False, passive: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = stats
        self._start_items = start_items
        self._disrupt = disrupt
        self._curated = curated
        self._passive = passive
        self._trackers: Dict[str, ChoiceBandTracker] = {}
        self._truth: Dict[str, _OppGroundTruth] = {}
        self._proto: Dict[str, List[tuple]] = defaultdict(list)
        self._item_priors = gen3_data.priors

    async def _handle_battle_message(self, split_messages):
        tag = None
        for sm in split_messages:
            if sm and sm[0].startswith(">"):
                tag = sm[0][1:]
                continue
            if tag is not None and len(sm) > 1:
                self._proto[tag].append(tuple(sm))
                self._truth_for(tag).ingest(tuple(sm))
        return await super()._handle_battle_message(split_messages)

    def _opp_role(self, battle) -> str:
        role = getattr(battle, "player_role", None) or battle._player_role
        return "p2" if role == "p1" else "p1"

    def _truth_for(self, tag: str) -> _OppGroundTruth:
        if tag not in self._truth:
            # Role is resolved lazily once a battle object exists; default p2 (p1's usual role).
            b = self._battles.get(tag)
            opp_role = self._opp_role(b) if b is not None else "p2"
            self._truth[tag] = _OppGroundTruth(opp_role, self._start_items, self.stats.coverage)
        return self._truth[tag]

    def _tracker(self, tag: str) -> ChoiceBandTracker:
        if tag not in self._trackers:
            self._trackers[tag] = ChoiceBandTracker()
        return self._trackers[tag]

    def _prior(self, species: str) -> float:
        return float(self._item_priors.items(species).get(CHOICE_BAND, 0.0))

    def choose_move(self, battle):
        try:
            self._observe_and_check(battle)
            return self._pick_move(battle)
        except AssertionError:
            raise
        except Exception as e:  # pragma: no cover - diagnostic
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)

    def _observe_and_check(self, battle):
        tag = battle.battle_tag
        tracker = self._tracker(tag)
        truth = self._truth_for(tag)
        truth.opp_role = self._opp_role(battle)  # fix role now the battle object exists
        live = battle.strict_view().live
        item_state = {m.species: (m.item, m.consumed_item) for m in live.opp.mons}
        events = battle.events_since(0)
        tracker.observe(item_state, events)

        self.stats.decisions += 1
        for m in live.opp.mons:
            sp = m.species
            self.stats.species_seen.add(sp)
            if m.item:
                self.stats.items_seen.add(m.item)
            t = truth.mons.get(sp)
            n_distinct = len(t.stint_moves) if t else 0
            prior = self._prior(sp)
            got = tracker.p_cb(sp)

            # --- CONTRACT: tracker p_cb == expected from independent stint count + live item. ---
            exp = expected_pcb(m.item, m.consumed_item is not None, n_distinct, prior)
            self.stats.contract_checks += 1
            if abs(got - exp) > 1e-6:
                self._violation(
                    battle, tag, sp,
                    f"CONTRACT: p_cb={got:.4f} expected={exp:.4f} "
                    f"(held={m.item} consumed={m.consumed_item} n_distinct={n_distinct} "
                    f"prior={prior:.4f})",
                )

            # --- GROUND-TRUTH INVARIANTS (curated: true item known). ---
            if self._curated and t is not None and t.item_revealed_or_start():
                true_held = t.held
                self.stats.invariant_checks += 1
                if true_held == CHOICE_BAND and got == 0.0:
                    self._violation(battle, tag, sp,
                                    "INV-A: TRUE Choice-Band holder false-eliminated (p_cb=0)")
                if true_held != CHOICE_BAND and got == 1.0:
                    self._violation(battle, tag, sp,
                                    f"INV-B: non-CB mon false-asserted as CB (p_cb=1, true={true_held})")

            # --- coverage that depends on tracker state, not just the protocol. ---
            if t is not None and t.start_item == CHOICE_BAND and t.held == CHOICE_BAND \
                    and not t.item_revealed and n_distinct >= 1 and got != 0.0:
                self.stats.coverage["cb_locked_one_move"] += 1
                if t.first_stint_move and _is_status_move(t.first_stint_move):
                    self.stats.coverage["cb_first_move_status"] += 1
            # across-switch: ≥2 distinct over the battle but ≤1 in the current stint, not eliminated
            if t is not None and len(t.ever_moves) >= 2 and n_distinct < 2 and got != 0.0 \
                    and m.item is None and m.consumed_item is None:
                self.stats.coverage["two_distinct_across_switch"] += 1

    def _violation(self, battle, tag, species, msg):
        full = f"{tag} turn {battle.turn} [{species}]: {msg}"
        self.stats.violations.append(full)
        plog = self._proto.get(tag, [])
        print(f"\n[FUZZ VIOLATION] {full}", flush=True)
        print("[FUZZ VIOLATION] last 40 protocol lines:", flush=True)
        for sm in plog[-40:]:
            if len(sm) >= 2 and sm[1] != "request":
                print(f"  {sm}", flush=True)
        raise AssertionError(full)

    # ---- move selection ----
    def _pick_move(self, battle):
        if self._passive and battle.available_moves and not battle.force_switch:
            return self._passive_move(battle)
        if self._disrupt and battle.available_moves and not battle.force_switch:
            opp = battle.opponent_active_pokemon
            opp_has_item = bool(opp and opp.item)
            me = battle.active_pokemon
            my_item = me.item if me else None
            for mv in battle.available_moves:
                mid = mv.id
                if mid == "knockoff" and opp_has_item:
                    return self.create_order(mv)
                if mid == "trick" and my_item == CHOICE_BAND and opp and opp.item != CHOICE_BAND:
                    return self.create_order(mv)
                if mid == "thief" and not my_item and opp_has_item:
                    return self.create_order(mv)
                if mid == "encore" and opp and (opp.last_move is not None):
                    return self.create_order(mv)
        return self.choose_random_move(battle)

    def _passive_move(self, battle):
        """Struggle-scenario p1: stay alive and drain the opponent's PP without KOing it for the
        first dozen turns (Pressure + Calm Mind / Rest), then attack so the battle still ends."""
        avail = {m.id: m for m in battle.available_moves}
        mon = battle.active_pokemon
        hp = (mon.current_hp_fraction if mon else 1.0) or 1.0
        if battle.turn < 14:
            if hp < 0.55 and "rest" in avail:
                return self.create_order(avail["rest"])
            for mid in ("calmmind", "protect", "spikes"):
                if mid in avail:
                    return self.create_order(avail[mid])
        for mid in ("surf", "drillpeck", "seismictoss", "shadowball"):
            if mid in avail:
                return self.create_order(avail[mid])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle) -> None:
        self.stats.battles += 1
        tag = battle.battle_tag
        self._trackers.pop(tag, None)
        self._truth.pop(tag, None)
        self._proto.pop(tag, None)


def _is_status_move(move_id: str) -> bool:
    md = gen3_data.moves.get(move_id)
    if md is None:
        return False
    cat = getattr(md, "category", None)
    name = getattr(cat, "name", str(cat)) if cat is not None else ""
    return name.lower() == "status"


# Add a tiny helper method onto _MonTruth (kept here so the dataclass stays declarative).
def _item_revealed_or_start(self) -> bool:
    """True item is trustworthy: either the protocol revealed it, or we know its start item and no
    change has been seen."""
    return self.item_revealed or self.start_item is not None


_MonTruth.item_revealed_or_start = _item_revealed_or_start


class CBObservedPlayer(Player):
    """p2 — the observed side. On curated teams a light script drives the edge behaviours
    (alternate non-CB moves, Swords-Dance-first on the CB-SD mon, Rest→Sleep Talk, Trick the CB
    away); otherwise it plays randomly."""

    def __init__(self, *args, scripted: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._scripted = scripted
        self._stint_used: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
        self._switch_budget: Dict[str, int] = defaultdict(int)

    def choose_move(self, battle):
        try:
            if not self._scripted:
                return self.choose_random_move(battle)
            return self._scripted_move(battle)
        except Exception:  # pragma: no cover
            return self.choose_random_move(battle)

    def _scripted_move(self, battle):
        mon = battle.active_pokemon
        if mon is None or not battle.available_moves:
            return self.choose_random_move(battle)
        species = mon.species
        avail = {m.id: m for m in battle.available_moves}

        # Alakazam (CB) → Trick its band away on the first opportunity.
        if species == "alakazam" and "trick" in avail:
            return self.create_order(avail["trick"])
        # Heracross (CB) → Swords Dance first so the lock is a STATUS move.
        if species == "heracross" and "swordsdance" in avail and mon.boosts.get("atk", 0) == 0:
            return self.create_order(avail["swordsdance"])
        # Snorlax (Lefties) → Rest while awake & hurt (→ sleep), then deliberately Sleep Talk
        # while asleep (an asleep mon can still SELECT any move; only Sleep Talk delegates — picking
        # an attack just yields |cant|slp, never the delegated line we need to exercise).
        if species == "snorlax" and "sleeptalk" in avail:
            from poke_env.battle.status import Status as _S
            if mon.status == _S.SLP:
                return self.create_order(avail["sleeptalk"])
            if "rest" in avail and (mon.current_hp_fraction or 1.0) < 1.0:
                return self.create_order(avail["rest"])

        # Non-CB alternator: prefer a move not yet used this stint (→ 2 distinct in a stint).
        key = (battle.battle_tag, id(mon))
        used = self._stint_used[key]
        fresh = [m for mid, m in avail.items() if mid not in used and mid != "explosion"]
        choice = fresh[0] if fresh else next(iter(avail.values()))
        used.add(choice.id)
        return self.create_order(choice)

    def _battle_finished_callback(self, battle) -> None:
        for k in [k for k in self._stint_used if k[0] == battle.battle_tag]:
            self._stint_used.pop(k, None)


class StruggleObservedPlayer(Player):
    """p2 — the Struggle scenario: a CB mon locked into a single low-PP move. Just play what's
    legal (it's choice-locked / forced), letting PP drain to Struggle."""

    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_random_move(battle)


# ===========================================================================
# Runner
# ===========================================================================

def _make_players(stats: FuzzStats, ts: int):
    """Build the curated, struggle, and random player pairs (all share ``stats``)."""
    observed_tb = Gen3Teambuilder([OBSERVED_TEAM, OBSERVED_TEAM_SNORLAX_LEAD])
    tracker_tb = Gen3Teambuilder([TRACKER_TEAM])
    p1 = CBTrackerPlayer(
        battle_format=BATTLE_FORMAT, team=tracker_tb, stats=stats,
        start_items=OBSERVED_START_ITEM, disrupt=True, curated=True,
        account_configuration=AccountConfiguration(f"CBt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)
    p2 = CBObservedPlayer(
        battle_format=BATTLE_FORMAT, team=observed_tb, scripted=True,
        account_configuration=AccountConfiguration(f"CBo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)

    sp1 = CBTrackerPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder([STRUGGLE_TRACKER_TEAM]), stats=stats,
        start_items=STRUGGLE_START_ITEM, disrupt=False, curated=True, passive=True,
        account_configuration=AccountConfiguration(f"CBst{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)
    sp2 = StruggleObservedPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder([STRUGGLE_OBSERVED_TEAM]),
        account_configuration=AccountConfiguration(f"CBso{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)

    pool = (TeamLoader().get_all_teams())
    if not pool:
        raise RuntimeError("no gen3ou teams under data/teams for the random sweep")
    rp1 = CBTrackerPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool), stats=stats,
        start_items={}, disrupt=False, curated=False,
        account_configuration=AccountConfiguration(f"CBr1{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)
    rp2 = CBObservedPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool), scripted=False,
        account_configuration=AccountConfiguration(f"CBr2{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle, max_concurrent_battles=4)
    return (p1, p2), (sp1, sp2), (rp1, rp2)


async def main(minutes: float = 1.0) -> None:
    ts = int(time.time()) % 100000
    stats = FuzzStats()
    (p1, p2), (sp1, sp2), (rp1, rp2) = _make_players(stats, ts)
    deadline = time.monotonic() + minutes * 60.0
    print(f"ChoiceBandTracker fuzz — gen3ou — budget {minutes:g} min", flush=True)

    # Interleave: curated disruptor matchup, the Struggle scenario, and a random-pool sweep.
    round_no = 0
    while time.monotonic() < deadline:
        round_no += 1
        await run_local_battles(p1, p2, 3)
        if time.monotonic() >= deadline:
            break
        await run_local_battles(sp1, sp2, 1)
        if time.monotonic() >= deadline:
            break
        await run_local_battles(rp1, rp2, 4)

    _report(stats, minutes)


def _report(stats: FuzzStats, minutes: float) -> None:
    s = stats
    print(f"\n{'=' * 68}")
    print(f"Budget                        : {minutes:g} min")
    print(f"Battles                       : {s.battles}")
    print(f"Decisions                     : {s.decisions}")
    print(f"Contract checks               : {s.contract_checks}")
    print(f"Invariant checks (curated)    : {s.invariant_checks}")
    print(f"Distinct opp species seen     : {len(s.species_seen)}")
    print(f"Distinct opp items seen       : {len(s.items_seen)}  {sorted(s.items_seen)[:12]}")
    print("\nEdge-case coverage (each MUST be > 0):")
    missing = []
    for edge in REQUIRED_EDGES:
        n = s.coverage.get(edge, 0)
        flag = "ok " if n > 0 else "MISS"
        if n == 0:
            missing.append(edge)
        print(f"  [{flag}] {edge:28s} {n}")
    extra = {k: v for k, v in s.coverage.items() if not k.startswith("_") and k not in REQUIRED_EDGES}
    if extra:
        print(f"  (other: {dict(extra)})")
    print(f"{'=' * 68}")

    if s.violations:
        print(f"FAIL — {len(s.violations)} tracker violation(s). First 5:")
        for v in s.violations[:5]:
            print(f"  {v}")
        sys.exit(1)
    if missing:
        print(f"FAIL — coverage gap: {missing} never fired this tier.")
        sys.exit(1)
    if s.decisions == 0:
        print("FAIL — no decisions recorded; bridge/team setup broken.")
        sys.exit(1)
    print("PASS — zero violations; every required edge case fired.")


if __name__ == "__main__":
    mins = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    asyncio.run(main(mins))
