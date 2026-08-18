"""Replay a spectator Showdown ``.log`` into per-decision ``(obs, action, mask)``.

This is the shared **Phase-1 data pipeline** for both the human-agreement probe
(measure-before-you-build) and behavioural cloning (``designs/ai_v6/impl_step2_bc.md``).
It drives a :class:`~agents.battle.gen3_battle.Gen3Battle` through a saved spectator
log the way poke-env's own client would, and at each turn where the chosen side made
a *voluntary* decision it reconstructs the full observation that side's agent would
have seen, plus the action the human actually took.

Why this reuses the live stack instead of re-deriving anything
--------------------------------------------------------------
``encode`` only needs a populated :class:`~agents.battle.live_view.LegalActions`
snapshot to run; everything else (current board, event log, turn-history, trackers)
folds from the protocol stream the log already contains. So we **synthesise** the
legal set from known state and hand it to the exact ``embed_battle`` call sequence
the inference player uses — same encoder, same tracker, same turn-history.

Faithful reconstruction (the foundation)
----------------------------------------
A naive single pass only knows our own team as it is *revealed*, which impoverishes
the obs and makes a switch to a not-yet-seen own mon unrepresentable. We fix that the
way a real player's knowledge would: a **pre-scan** collects our side's full revealed
team + per-mon moveset across the whole game, and at turn 1 we inject one synthesised
``|request|`` (``battle.parse_request``) — poke-env's *own* native mechanism — which
creates the full bench and adds every mon's known moveset. From then on the normal
replay manages HP / status / active. So the obs own-team block, the move mask, and the
switch options are all faithful to what that player knew, not just to what a spectator
had seen so far.

Residual limits — unrecoverable from a SPECTATOR log (they apply to BC too, so we
surface them rather than hide them):

* **EV / IV / nature spreads are absent** (a real own-team request carries computed
  stats; a spectator log never does). Base stats are known from species data; the
  spread block stays at defaults. Abilities / items are passed only when revealed.
* **Moves never used all game stay hidden** — the reconstructed moveset is the union
  of what that mon used across the battle (typically 3-4 of 4), not its true kit.
* **No PP tracking** — every known move is treated as available (PP inference from
  move-use counts is a documented refinement).
* **Voluntary decisions only.** The first action our active takes each turn is the
  voluntary choice; forced post-faint replacements (a second action in the same turn)
  and ``|cant|`` (sleep / recharge / flinch) turns are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from poke_env.data.normalize import to_id_str

from agents.action.constants import MOVE_START, N_MOVE_SLOTS
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.battle_event import UnknownMessageType, UnsupportedMessageType
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions, LegalMove, LegalSwitch
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.training.episode_tracker import EpisodeTracker

_LOG = logging.getLogger("bc.log_reader")

# Connection / chat / cosmetic lines a spectator log carries that a live battle stream
# never routes to the battle object. Skipped outright so the Gen3Battle classify
# tripwire only ever sees genuine gen3 protocol keywords.
_SKIP_KEYWORDS = frozenset({
    "", "init", "title", "j", "J", "l", "L", "c", "c:", "n", "N", "chat", "join",
    "leave", "player", "inactive", "inactiveoff", "raw", "html", "uhtml",
    "uhtmlchange", "t:", "expire", "badge", "spectator", "spectatorleave",
    "message", "-message", "debug", "error", "bigerror", "askreg", "unlink",
    "showteam", "seed", "gen", "tier", "rule", "rated", "clearpoke", "teampreview",
})


@dataclass(frozen=True)
class Decision:
    """One reconstructed voluntary decision from the chosen side's perspective."""

    obs: np.ndarray            # full observation vector (gen3_frame_deletion_v1: == encode output)
    mask: np.ndarray           # (11,) int8 synthesised action mask
    human_action: int          # action index the human actually took
    action_type: str           # "move" | "switch"
    turn: int
    our_species: str
    opp_species: str
    n_legal: int               # number of legal actions in the synthesised mask


@dataclass
class ReadStats:
    """Per-replay bookkeeping — how much was reconstructable vs structurally lost."""

    decisions: int = 0
    # ~0 with turn-1 full-team injection; a residual hit means a mon appeared only
    # AFTER turn 1 (e.g. the scan missed it) so it wasn't in the injected team.
    excluded_switch_unrevealed: int = 0
    excluded_move_unmapped: int = 0       # move we failed to map (should be ~0; a parse bug if not)
    excluded_empty_mask: int = 0          # synth legal set was empty
    turns: int = 0


@dataclass
class ReplayDecisions:
    decisions: List[Decision]
    stats: ReadStats
    winner_role: Optional[str]   # "p1"/"p2" or None
    our_role: str                # which side we reconstructed
    won: Optional[bool]          # did our side win?


def _split_lines(text: str) -> List[List[str]]:
    out: List[List[str]] = []
    for raw in text.split("\n"):
        if not raw:
            continue
        if raw[0] == ">":            # room framing — not a protocol line
            continue
        out.append(raw.split("|"))
    return out


def _species_of(detail_field: str) -> str:
    """'Skarmory, F' / 'Tyranitar, L100, M' -> id form 'skarmory'/'tyranitar'."""
    return to_id_str(detail_field.split(",")[0])


def _resolve_player_roles(lines: List[List[str]]) -> Dict[str, str]:
    """Map 'p1'/'p2' -> username from the |player| lines (first, rated, occurrence)."""
    roles: Dict[str, str] = {}
    for parts in lines:
        if len(parts) > 3 and parts[1] == "player" and parts[2] and parts[3]:
            roles.setdefault(parts[2], parts[3])
    return roles


@dataclass
class _TeamScan:
    order: List[str]                       # ident-name (nickname) in first-reveal order
    details: Dict[str, str]                # name -> switch details ("Skarmory, M")
    movesets: Dict[str, List[str]]         # name -> move ids used across the whole game
    abilities: Dict[str, str]              # name -> revealed ability id ("" if never)
    items: Dict[str, str]                  # name -> revealed item id ("" if never)


def _prescan_team(lines: List[List[str]], our_role: str) -> _TeamScan:
    """Collect our side's FULL team + per-mon revealed moveset across the whole game.

    This is the knowledge a real player had about their own side that a progressive
    spectator view lacks; it backs the turn-1 full-team request injection.
    """
    prefix = our_role + "a"
    scan = _TeamScan(order=[], details={}, movesets={}, abilities={}, items={})
    for p in lines:
        if len(p) < 3:
            continue
        kw, actor = p[1], p[2]
        if not actor.startswith(prefix):
            continue
        name = actor.split(": ", 1)[1] if ": " in actor else actor
        if kw in ("switch", "drag"):
            if name not in scan.movesets:
                scan.order.append(name)
                scan.movesets[name] = []
            scan.details.setdefault(name, p[3] if len(p) > 3 else name)
        elif kw == "move" and len(p) > 3:
            mid = to_id_str(p[3])
            scan.movesets.setdefault(name, [])
            if mid and mid != "struggle" and mid not in scan.movesets[name]:
                scan.movesets[name].append(mid)
        elif kw == "-ability" and len(p) > 3:
            scan.abilities.setdefault(name, to_id_str(p[3]))
        elif kw in ("-item", "-enditem") and len(p) > 3:
            scan.items.setdefault(name, to_id_str(p[3]))
    return scan


def _full_team_request(battle: Gen3Battle, our_role: str,
                       our_username: str, scan: _TeamScan) -> dict:
    """Build a complete own-side request from the pre-scan (all mons at full HP — the
    state at turn 1, before anything is damaged). Injected once via ``parse_request``."""
    side_pokemon = []
    for name in scan.order:
        ident = f"{our_role}: {name}"
        mon = battle.team.get(ident)
        moves = scan.movesets.get(name, [])[:N_MOVE_SLOTS]
        side_pokemon.append({
            "ident": ident,
            "details": scan.details.get(name, name),
            "condition": "100/100",
            "active": bool(mon.active) if mon is not None else False,
            "item": scan.items.get(name, ""),
            "baseAbility": scan.abilities.get(name, ""),
            "moves": moves,
        })
    return {"side": {"name": our_username, "id": our_role, "pokemon": side_pokemon}}


class SpectatorLogReader:
    """Replays one spectator ``.log`` and yields the chosen side's decisions."""

    def __init__(self, mappings: dict, encoder: Optional[Gen3ObservationEncoder] = None):
        self._mappings = mappings
        self._encoder = encoder or Gen3ObservationEncoder(mappings)

    # -- legal-set synthesis -------------------------------------------------
    def _synth_legal(self, battle: Gen3Battle, used_move_id: Optional[str]) -> LegalActions:
        active = battle.active_pokemon
        move_ids: List[str] = list(active.moves.keys()) if active is not None else []
        if used_move_id is not None and used_move_id not in move_ids:
            # A move revealed *this* turn isn't in the dict yet — union it so the human's
            # action is representable. Hidden Power is keyed bare in protocol but typed in
            # poke-env's move dict; treat any hiddenpower* as already present.
            if not (used_move_id.startswith("hiddenpower")
                    and any(m.startswith("hiddenpower") for m in move_ids)):
                move_ids.append(used_move_id)
        move_ids = move_ids[:N_MOVE_SLOTS]
        move_slots = tuple(
            LegalMove(id=mid, current_pp=1, max_pp=1, disabled=False, target=None)
            for mid in move_ids
        )
        switches = tuple(
            LegalSwitch(species=mon.species, slot=i)
            for i, mon in enumerate(battle.team.values())
            if (not mon.active and not mon.fainted)
        )
        return LegalActions(
            move_slots=move_slots, switches=switches, force_switch=False,
            trapped=False, maybe_trapped=False, wait=False, struggle=False,
            last_request=None,
        )

    @staticmethod
    def _map_action(parts: List[str], legal: LegalActions) -> Tuple[Optional[int], str]:
        kw = parts[1]
        if kw == "move":
            used = to_id_str(parts[3])
            for i, m in enumerate(legal.move_slots):
                if m.id == used or (
                    used.startswith("hiddenpower") and m.id.startswith("hiddenpower")
                ):
                    return MOVE_START + i, "move"
            return None, "move_unmapped"
        # switch / drag
        species = _species_of(parts[3]) if len(parts) > 3 else ""
        for sw in legal.switches:
            if sw.species == species:
                return sw.slot, "switch"
        return None, "switch_unrevealed"

    def _build_obs(self, battle: Gen3Battle, legal: LegalActions,
                   tracker: EpisodeTracker) -> Tuple[Optional[np.ndarray], np.ndarray]:
        mask = Gen3ActionMasker.mask_from_legal(legal).astype(np.int8)
        if int(mask.sum()) == 0:
            return None, mask
        if not battle.strict_view().finished:
            tracker.record(battle, mask, legal=legal)
            tracker.update_progress_clock(battle, legal)
        obs = self._encoder.encode(
            battle, hp_tracker=tracker.hidden_power_tracker, legal=legal,
            progress_clock=tracker.progress_clock,
        )
        full = obs.astype(np.float32)
        return full, mask

    # -- main entry ----------------------------------------------------------
    def read(self, log_text: str, our_username: str) -> ReplayDecisions:
        lines = _split_lines(log_text)
        roles = _resolve_player_roles(lines)
        our_role = next(
            (r for r, name in roles.items() if to_id_str(name) == to_id_str(our_username)),
            None,
        )
        if our_role is None:
            raise ValueError(f"username {our_username!r} not found in log players {roles}")
        our_prefix = our_role + "a"  # e.g. 'p2a'

        battle = Gen3Battle(
            battle_tag="battle-gen3ou-bcoffline",
            username=our_username,
            logger=_LOG,
            save_replays=False,
            gen=3,
        )
        battle._player_role = our_role
        battle._format = "gen3ou"

        # Pre-scan our full team + movesets so the turn-1 injection makes the obs faithful.
        scan = _prescan_team(lines, our_role)

        tracker = EpisodeTracker()
        stats = ReadStats()
        decisions: List[Decision] = []
        winner_role: Optional[str] = None

        current_turn = 0
        decided_this_turn = False
        injected = False

        for parts in lines:
            if len(parts) < 2:
                continue
            kw = parts[1]

            if kw == "win":
                name = parts[2] if len(parts) > 2 else ""
                winner_role = next(
                    (r for r, n in roles.items() if to_id_str(n) == to_id_str(name)), None
                )
                battle.won_by(name)
                break
            if kw == "tie":
                battle.tied()
                break
            if kw == "turn":
                self._safe_parse(battle, parts)
                current_turn = battle.turn
                decided_this_turn = True if current_turn < 1 else False
                stats.turns = max(stats.turns, current_turn)
                # At turn 1 both leads are out and nothing is damaged — inject the full
                # own team (creates the bench, adds every known moveset) exactly once.
                if current_turn >= 1 and not injected:
                    injected = True
                    try:
                        battle.parse_request(
                            _full_team_request(battle, our_role, our_username, scan)
                        )
                    except Exception as e:  # malformed scan — fall back to revealed-only
                        _LOG.debug("full-team injection failed: %s", e)
                continue

            # Our active's voluntary action this turn → reconstruct the decision
            # BEFORE applying it to state (we want the pre-decision board).
            is_our_action = (
                kw in ("move", "switch")
                and len(parts) > 2 and parts[2].startswith(our_prefix)
            )
            if is_our_action and current_turn >= 1 and not decided_this_turn:
                decided_this_turn = True
                used_move = to_id_str(parts[3]) if (kw == "move" and len(parts) > 3) else None
                legal = self._synth_legal(battle, used_move)
                action_id, atype = self._map_action(parts, legal)
                if action_id is None:
                    if atype == "switch_unrevealed":
                        stats.excluded_switch_unrevealed += 1
                    else:
                        stats.excluded_move_unmapped += 1
                else:
                    full, mask = self._build_obs(battle, legal, tracker)
                    if full is None:
                        stats.excluded_empty_mask += 1
                    elif mask[action_id] != 1:
                        stats.excluded_move_unmapped += 1
                    else:
                        opp = battle.opponent_active_pokemon
                        our = battle.active_pokemon
                        decisions.append(Decision(
                            obs=full, mask=mask, human_action=int(action_id),
                            action_type=atype, turn=current_turn,
                            our_species=our.species if our else "",
                            opp_species=opp.species if opp else "",
                            n_legal=int(mask.sum()),
                        ))
                        stats.decisions += 1
                        tracker.advance(int(action_id))
                self._safe_parse(battle, parts)
                continue

            # Everything else: drive state forward (skipping spectator/chat junk).
            if kw in _SKIP_KEYWORDS:
                continue
            self._safe_parse(battle, parts)

        won = None if winner_role is None else (winner_role == our_role)
        return ReplayDecisions(
            decisions=decisions, stats=stats, winner_role=winner_role,
            our_role=our_role, won=won,
        )

    @staticmethod
    def _safe_parse(battle: Gen3Battle, parts: List[str]) -> None:
        try:
            battle.parse_message(parts)
        except (UnknownMessageType, UnsupportedMessageType) as e:
            # Offline tooling is the documented use of the classify sentinel — a
            # spectator-only keyword we haven't listed. Skip it, but log once.
            _LOG.debug("skip unclassified protocol line %r: %s", parts[:2], e)
