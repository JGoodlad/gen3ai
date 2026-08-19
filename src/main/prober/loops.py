"""BAIT-LOOP detection — "the opponent pivoted something immune in, and we fired anyway".

The pathology this measures (gen-15, 843-battle sentinel sweep + a causal injection probe,
`designs/research_state/ledger.md`, the two 2026-08-19 sections): the opponent VOLUNTARILY pivots
a mon our attack cannot touch, we attack it anyway at a median chosen-probability of 0.963, and the
exchange REPEATS — one battle ran 9 cycles of Earthquake into a switched-in Salamence. Perception
is not the gap (α called SWITCH on 76% of loop steps, β named the right slot on 82%); the injection
probe found no CHANNEL by which "your ordinary attack does nothing to the arrival" can reach the
policy at all. Gen-16 enables the switch-branch cell — that channel — and this module is the
instrument that says whether the behaviour died.

**Everything here is derived from the raw Showdown PROTOCOL, never from the prober's rendered
timeline.** That is not a preference: the rendered `— no effect` deliberately collapses an
immunity, a full-paralysis `cant`, and an unpriced small hit into one phrase, and a detector built
on it would count all three (verified on the calibration battle — its T54 `we surf — no effect` is
a `|cant|p1a: Suicune|par` and its T40 `rapidspin — no effect` is a real 1% resisted hit; neither
is a bait).

Definitions, fixed HERE so every surface means the same thing:

  * **voluntary pivot** — a ``|switch|`` for a side, in a turn block where that side had NOT
    fainted earlier in the block, and not a ``|drag|`` (Whirlwind/Roar are the opponent's choice
    about as much as a coin flip is). The turn-0 leads are not pivots.
  * **moved into** — the other side then used a move, AFTER the arrival, TARGETING the pivoting
    side. A self-targeting move (Protect / Recover / Refresh) is not a bait: nothing was fired at
    the arrival. This is the DENOMINATOR — a pivot we answered with a switch of our own is not an
    opportunity to whiff.
  * **whiff** — that move did nothing: ``immune`` (a literal ``|-immune|``), ``fail`` (a ``|-fail|``
    with no external ``[from]`` cause), or ``near_zero`` (≤ ``near_zero_frac`` of the target's HP).
    A ``miss`` is counted SEPARATELY and is never a whiff — it is luck, and taxing it would make
    the metric partly a dice reading.
  * **loop** — the same ``(our move, arriving species)`` pair whiffing ≥2 times in ONE battle.
  * **re-click** — a whiff whose pair had ALREADY whiffed earlier in the same battle. An immunity
    is deterministic and fully observable once seen, so every re-click is a decision taken with
    the answer already on the board. This is the sharpest of the three rates and the one the
    gen-16 bar is written against.

Pure: no torch, no session, no file IO. The caller supplies the parsed protocol lines
(`engine.parse_protocol_log`) and the trace's recorded decisions; everything else is derived.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field, replace
from typing import Sequence

#: What gen-15 measured, carried WITH every live reading so a number is never quoted without its
#: reference point (the `awareness.AWARENESS_BASELINES` pattern). A baseline is a reference, NOT a
#: target — the pre-registered gen-16 BARS live in `designs/research_state/bait_loop_hunt.md`.
#: Source: the 843-battle gen-15 sentinel sweep, `designs/research_state/ledger.md` (2026-08-19).
LOOP_BASELINES = {
    "generation": "gen-15",
    "measured": "2026-08-19",
    "run": "ai_v9_18_gen15_v8rewards_0818",
    "scope": "sentinel_* eval opponents, every step",
    "n_battles": 843,
    "moved_into_pivots": 4923,
    "whiffs": 820,
    "whiff_rate_per_moved_into_pivot": 0.167,
    "loop_battles": 117,
    "loop_battle_rate": 0.139,
    "loop_ge3_battles": 52,
    "loop_ge3_rate": 0.062,
    "reclicks": 264,
    "reclick_rate": 0.322,
    "median_chosen_prob_on_loop_steps": 0.963,
    "loop_step_median_delta_v": -4.31,
    "loop_step_median_delta_win_prob": -0.096,
    "whiff_kinds": {"immune": 769, "fail": 41, "near_zero": 10},
    # battles-with-a-loop by training step — it got WORSE before it plateaued, so a single
    # end-of-run number read alone would understate the behaviour's persistence.
    "loop_battle_rate_by_step": {"4M": 0.050, "20M": 0.211, "24M": 0.152},
    "beta_slot_accuracy": {"first_time": 0.520, "repeat": 0.659, "loop_step": 0.821},
    "alpha_switch_top1_on_loop_steps": 0.762,
    "alpha_switch_p_median_on_loop_steps": 0.60,
    "mirror_whiff_rate": 0.145,       # THEY whiff into OUR pivots — the opponent-side control
    "source": "designs/research_state/ledger.md (2026-08-19); "
              "designs/research_state/bait_loop_hunt.md",
}

#: The three ways a move can do nothing on purpose. A MISS is deliberately absent — see the header.
WHIFF_KINDS = ("immune", "fail", "near_zero")

#: Default "near zero" band, as a fraction of the target's max HP. 1% is the spec bar: below it a
#: gen-3 attack is doing rounding, not damage.
NEAR_ZERO_FRAC = 0.01

#: Float slack on the near-zero comparison — see `bait_events`.
_HP_EPS = 1e-9

_SWITCH_RE = re.compile(r"^\|(switch|drag)\|(p[12])a: ([^|]*)\|([^|]*)\|(.*)$")
_MOVE_RE = re.compile(r"^\|move\|(p[12])a: ([^|]*)\|([^|]*)\|(.*)$")
_DAMAGE_RE = re.compile(r"^\|-(damage|heal)\|(p[12])a: ([^|]*)\|([^|]*)$")
_FAINT_RE = re.compile(r"^\|faint\|(p[12])a: (.*)$")
_PLAYER_RE = re.compile(r"^\|player\|(p[12])\|([^|]*)")

#: Lines that CLOSE a move's result window. Everything between a `|move|` and one of these belongs
#: to that move; after it, a `-damage` is end-of-turn residual (sand/burn/Leftovers), not the hit.
_MOVE_WINDOW_END = ("|upkeep", "|-weather", "|turn|")


def norm_id(s: object) -> str:
    """`"Rapid Spin"` / `"rapidspin"` → `rapidspin`. The project's `to_id_str` convention, local so
    this module stays import-free."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def species_of(detail: str) -> str:
    """`"Salamence, F"` / `"Ditto"` → the canonical species id. The protocol's details field carries
    gender/level/shiny after the species."""
    return norm_id(str(detail).split(",")[0])


def hp_frac(s: str) -> "float | None":
    """`"224/404 par"` → 0.554, `"0 fnt"` → 0.0, `"55/100"` → 0.55. None when unparseable.

    Showdown emits either absolute HP or (under HP Percentage Mod, which every eval battle runs)
    `cur/100`; both are `a/b`, so one parse covers them."""
    tok = (s or "").strip().split(" ")[0] if s else ""
    if not tok:
        return None
    if tok in ("0", "0.0"):
        return 0.0
    if "/" not in tok:
        return None
    a, _, b = tok.partition("/")
    try:
        num, den = float(a), float(b)
    except ValueError:
        return None
    return (num / den) if den else None


def players_from_protocol(lines: Sequence[str]) -> "dict[str, str]":
    """`{"p1": "<username>", "p2": "<username>"}` from the ``|player|`` lines."""
    out: "dict[str, str]" = {}
    for ln in lines or ():
        m = _PLAYER_RE.match(ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def split_turns(lines: Sequence[str]) -> "list[tuple[int, list[str]]]":
    """Protocol lines → ordered ``(turn, lines)`` blocks. Turn 0 is everything before the first
    ``|turn|1``: team preview, the rules, and the LEADS — which are switches but not pivots."""
    blocks: "collections.OrderedDict[int, list[str]]" = collections.OrderedDict()
    cur = 0
    blocks[0] = []
    for ln in lines or ():
        if ln.startswith("|turn|"):
            try:
                cur = int(ln.split("|")[2])
            except (IndexError, ValueError):
                pass
            blocks.setdefault(cur, [])
            continue
        blocks.setdefault(cur, []).append(ln)
    return list(blocks.items())


@dataclass(frozen=True)
class ProtocolEvent:
    """One ordered protocol fact. `i` is the line index WITHIN the turn block, which is what makes
    "the move came after the arrival" decidable — the sim emits lines in execution order."""
    turn: int
    i: int
    kind: str                       # "switch" | "faint" | "move"
    side: str                       # "p1" | "p2"
    species: str = ""               # switch: the arriving mon
    move: str = ""                  # move: the move id
    forced: bool = False            # switch: post-faint replacement
    drag: bool = False              # switch: Whirlwind/Roar — not the side's choice
    hp: "float | None" = None       # switch: the arrival's HP fraction
    target_side: "str | None" = None    # move: the side named as the target
    immune: bool = False
    miss: bool = False
    fail: bool = False
    hp_after: "float | None" = None     # move: the defender's HP after the hit
    hp_before: "float | None" = None    # move: the defender's HP as of the move


def parse_events(lines: Sequence[str]) -> "tuple[ProtocolEvent, ...]":
    """Protocol lines → the ordered switch / faint / move facts, each move carrying its OUTCOME.

    A move's outcome lines (`|-immune|`, `|-miss|`, `|-fail|`, `|-damage|`) are the lines between it
    and the next move / end-of-turn marker. `hp_before` is tracked RUNNING (from every `-damage` /
    `-heal` on that side) rather than taken from the switch line, so entry hazards landing between
    the arrival and our attack are not silently charged to the attack.
    """
    out: "list[ProtocolEvent]" = []
    for turn, blk in split_turns(lines):
        fainted: "set[str]" = set()
        hp_now: "dict[str, float | None]" = {}
        pending: "dict | None" = None

        def _flush() -> None:
            if pending is not None:
                out.append(ProtocolEvent(**pending))

        for idx, ln in enumerate(blk):
            m = _SWITCH_RE.match(ln)
            if m:
                _flush()
                pending = None
                kind, side, _nick, detail, hp = m.groups()
                hp_now[side] = hp_frac(hp)
                out.append(ProtocolEvent(
                    turn=turn, i=idx, kind="switch", side=side, species=species_of(detail),
                    forced=(side in fainted) or kind == "drag", drag=(kind == "drag"),
                    hp=hp_now[side]))
                continue
            m = _FAINT_RE.match(ln)
            if m:
                side = m.group(1)
                fainted.add(side)
                hp_now[side] = 0.0
                out.append(ProtocolEvent(turn=turn, i=idx, kind="faint", side=side))
                continue
            m = _MOVE_RE.match(ln)
            if m:
                _flush()
                side, _nick, move, rest = m.groups()
                tgt = rest.split("|")[0].strip()
                tgt_side = tgt[:2] if tgt[:2] in ("p1", "p2") else None
                defender = "p2" if side == "p1" else "p1"
                pending = {
                    "turn": turn, "i": idx, "kind": "move", "side": side, "move": norm_id(move),
                    "target_side": tgt_side, "hp_before": hp_now.get(defender),
                }
                continue
            if pending is not None:
                defender = "p2" if pending["side"] == "p1" else "p1"
                if ln.startswith("|-immune|" + defender + "a:"):
                    pending["immune"] = True
                elif ln.startswith("|-miss|"):
                    pending["miss"] = True
                elif ln.startswith("|-fail|") and "[from]" not in ln:
                    # A `[from]` cause means something ELSE failed the move (an ability, a
                    # protect); the bait question is whether OUR move did nothing on its own.
                    pending["fail"] = True
                else:
                    md = _DAMAGE_RE.match(ln)
                    if md and md.group(2) == defender and "[from]" not in ln:
                        pending["hp_after"] = hp_frac(md.group(4))
            md = _DAMAGE_RE.match(ln)
            if md:
                hp_now[md.group(2)] = hp_frac(md.group(4))
            if any(ln.startswith(p) for p in _MOVE_WINDOW_END):
                _flush()
                pending = None
        _flush()
    return tuple(out)


def active_by_turn(events: Sequence[ProtocolEvent]) -> "dict[int, dict[str, str]]":
    """`{turn: {"p1": species, "p2": species}}` — each side's active as of the START of that turn,
    i.e. after every earlier turn's switches and before this turn's. That is the board a decision
    was recorded against, which is what makes it a join key."""
    out: "dict[int, dict[str, str]]" = {}
    cur: "dict[str, str]" = {}
    turns = sorted({e.turn for e in events})
    for t in turns:
        out[t] = dict(cur)
        for e in events:
            if e.turn == t and e.kind == "switch":
                cur[e.side] = e.species
    return out


def identify_our_side(events: Sequence[ProtocolEvent],
                      our_actives: "dict[int, str]") -> "tuple[str | None, str]":
    """WHICH protocol side is the trainee — decided STRUCTURALLY, never from a username.

    Returns ``(side, reason)``; ``side`` is None when it cannot be settled, and the caller must then
    SKIP the battle rather than assume. Assuming p1 is the whole class of bug this avoids: eval
    seats the trainee on either side, and a mirror match makes the species names useless as a tell.

    The evidence is the recorded board itself: for each turn the trace says which mon WE had active,
    and the protocol says which mon each side had active. The side that agrees more often is ours.
    A tie (no decisions, or a true mirror that never diverged) returns None.
    """
    if not our_actives:
        return None, "no recorded actives to match against"
    board = active_by_turn(events)
    score = {"p1": 0, "p2": 0}
    for turn, species in our_actives.items():
        row = board.get(int(turn))
        if not row or not species:
            continue
        for side in ("p1", "p2"):
            if row.get(side) == species:
                score[side] += 1
    if score["p1"] == score["p2"]:
        return None, f"side undecidable (active-species agreement tied at {score['p1']})"
    side = "p1" if score["p1"] > score["p2"] else "p2"
    return side, f"active-species agreement {score[side]} vs {score['p1' if side == 'p2' else 'p2']}"


@dataclass(frozen=True)
class BaitEvent:
    """One voluntary pivot that we MOVED into — whiff or not. The non-whiffs are kept because they
    are the denominator: a rate reported over whiffs alone cannot fall by getting better."""
    turn: int
    arrival: str
    move: str
    kind: str                       # immune | fail | near_zero | miss | hit | no_damage
    whiff: bool
    loop_step: bool = False         # this (move, arrival) pair whiffed ≥2× in this battle
    reclick: bool = False           # this pair had ALREADY whiffed earlier in this battle
    inv: "int | None" = None        # the decision index this turn's move_selection sits at
    chosen_prob: "float | None" = None
    delta_v: "float | None" = None
    delta_win_prob: "float | None" = None


@dataclass(frozen=True)
class PivotRead:
    """The α/β readout on one voluntary opponent pivot — did the model SEE it coming?

    Kept beside the whiffs on purpose: perception and action are separate failures, and the gen-15
    finding was that only the second one was broken. A gen-16 run whose whiff rate falls while these
    also fall has not fixed the policy, it has lost the belief."""
    turn: int
    arrival: str
    first_time: bool                # first VOLUNTARY pivot of this species in this battle
    arrival_revealed: bool          # the mon was already revealed ⇒ its obs slot is decidable
    slot_true: "int | None"
    beta_top_slot: "int | None"
    slot_correct: "bool | None"
    species_correct: "bool | None"
    alpha_top_is_switch: "bool | None"
    alpha_switch_p: "float | None"
    loop_step: bool = False


@dataclass(frozen=True)
class BattleLoops:
    """One battle's bait/loop fold. `skipped` non-None means NOTHING here was judged."""
    skipped: "str | None" = None
    our_side: "str | None" = None
    side_reason: str = ""
    outcome: str = "?"
    n_turns: "int | None" = None
    # `move_selection` decisions in this battle — the PER-DECISION denominator. Carried because
    # every per-battle rate here is confounded by game LENGTH (a longer game has more chances to
    # loop), and a rate needs a denominator that grows with the exposure to be read alongside one
    # that does not.
    n_decisions: int = 0
    # our-side (the pathology): THEY pivot, WE fire
    opp_voluntary_pivots: int = 0       # every voluntary opp pivot after turn 0
    moved_into_pivots: int = 0          # …of which we answered with a move at the arrival
    whiffs: int = 0
    whiff_kinds: "dict[str, int]" = field(default_factory=dict)
    misses: int = 0                     # excluded from whiffs by construction — luck, not a bait
    reclicks: int = 0
    loop_battle: bool = False           # ≥1 pair whiffed ≥2×
    worst_loop: int = 0                 # the largest single-pair repeat count
    loops: "tuple[dict, ...]" = ()      # [{move, arrival, count, turns}]
    baits: "tuple[BaitEvent, ...]" = ()
    reads: "tuple[PivotRead, ...]" = ()
    # EVERY turn's critic deltas bucketed `loop_step` / `other_bait` / `other`. The third bucket is
    # the whole point: "loop turns cost ΔV −4.31" means nothing without the ordinary turn it is
    # being compared to, and the comparison has to come from the SAME battles.
    turn_deltas: "tuple[dict, ...]" = ()
    # mirror (the control): WE pivot, THEY fire. Same code path, sides swapped.
    our_voluntary_pivots: int = 0
    mirror_moved_into: int = 0
    mirror_whiffs: int = 0
    mirror_loop_battle: bool = False


def bait_events(events: Sequence[ProtocolEvent], *, pivot_side: str,
                near_zero_frac: float = NEAR_ZERO_FRAC) -> "tuple[BaitEvent, ...]":
    """Every voluntary pivot by ``pivot_side`` that the OTHER side then moved into, classified.

    Returns one `BaitEvent` per moved-into pivot (`loop_step` / `reclick` are filled by
    `mark_loops`, which needs the whole battle). A pivot answered with a switch, with a
    self-targeting move, or with nothing at all yields no event — it was not an opportunity to
    whiff, so counting it in the denominator would dilute the rate with turns that cannot fail.
    """
    atk_side = "p1" if pivot_side == "p2" else "p2"
    by_turn: "dict[int, list[ProtocolEvent]]" = collections.defaultdict(list)
    for e in events:
        by_turn[e.turn].append(e)
    out: "list[BaitEvent]" = []
    for turn in sorted(by_turn):
        if turn == 0:
            continue                      # the LEADS are not a pivot
        evs = by_turn[turn]
        pivots = [e for e in evs
                  if e.kind == "switch" and e.side == pivot_side and not e.forced and not e.drag]
        if not pivots:
            continue
        arrival = pivots[0]
        moves = [e for e in evs if e.kind == "move" and e.side == atk_side
                 and e.i > arrival.i and e.target_side == pivot_side]
        if not moves:
            continue
        m = moves[0]
        if m.miss:
            kind = "miss"
        elif m.immune:
            kind = "immune"
        elif m.fail:
            kind = "fail"
        elif m.hp_after is None:
            kind = "no_damage"            # a status/utility move, or no damage line at all
        else:
            before = m.hp_before if m.hp_before is not None else (
                arrival.hp if arrival.hp is not None else 1.0)
            lost = max(0.0, before - m.hp_after)
            # `+ _HP_EPS`: HP arrives as `k/100` under HP Percentage Mod, so an exactly-1% hit
            # computes as 0.010000000000000009 and a bare `<=` would call it a real hit. That is
            # not hypothetical — it is the calibration battle's T40 Rapid Spin (91% → 90%).
            kind = "near_zero" if lost <= near_zero_frac + _HP_EPS else "hit"
        out.append(BaitEvent(turn=turn, arrival=arrival.species, move=m.move, kind=kind,
                             whiff=kind in WHIFF_KINDS))
    return tuple(out)


def mark_loops(baits: Sequence[BaitEvent]) -> "tuple[tuple[BaitEvent, ...], tuple[dict, ...]]":
    """Fill `loop_step` / `reclick` and return the loop groups.

    A LOOP is a `(move, arrival)` pair that whiffed ≥2× in this battle — it is symmetric over the
    whole battle, so the FIRST click of a pair that later repeats is a loop step too. A RE-CLICK is
    ordered: the 2nd..Nth click only. They answer different questions ("did this battle contain a
    cycle" vs "how often did it re-take a decision the board had already answered"), so both ship.
    """
    counts = collections.Counter((b.move, b.arrival) for b in baits if b.whiff)
    loops = {k: c for k, c in counts.items() if c >= 2}
    turns: "dict[tuple, list[int]]" = collections.defaultdict(list)
    for b in baits:
        if b.whiff and (b.move, b.arrival) in loops:
            turns[(b.move, b.arrival)].append(b.turn)
    seen: "set[tuple]" = set()
    marked: "list[BaitEvent]" = []
    for b in sorted(baits, key=lambda x: x.turn):
        key = (b.move, b.arrival)
        reclick = b.whiff and key in seen
        if b.whiff:
            seen.add(key)
        marked.append(replace(b, loop_step=(b.whiff and key in loops), reclick=reclick))
    groups = tuple({"move": k[0], "arrival": k[1], "count": c, "turns": sorted(turns[k])}
                   for k, c in sorted(loops.items(), key=lambda kv: (-kv[1], kv[0])))
    return tuple(marked), groups


def pivot_reads(events: Sequence[ProtocolEvent], *, opp_side: str,
                dec_by_turn: "dict[int, dict]", loop_turns: "set[int]") -> "tuple[PivotRead, ...]":
    """The α/β readout on each voluntary opponent pivot.

    ⚠ The β SLOT is compared STRUCTURALLY, never by the rendered species name: obs slot *k* of the
    opponent-team block is the *k*-th REVEALED opponent mon (`get_team_list` iterates
    `battle.opponent_team`, a dict in first-seen order). So the true slot is the arrival's index in
    the reveal order AS OF the start of this turn — decidable only when the arrival was already
    revealed. β's `species` label is a decode of the belief head and can name a mon that is not on
    the team at all; using it as the ground truth would grade the readout against itself.
    """
    reveal_order: "list[str]" = []
    seen_voluntary: "set[str]" = set()
    out: "list[PivotRead]" = []
    for e in events:
        if e.kind != "switch" or e.side != opp_side:
            continue
        sp = e.species
        was_revealed = sp in reveal_order
        slot_true = reveal_order.index(sp) if was_revealed else None
        if not e.forced and not e.drag and e.turn > 0:
            inv = dec_by_turn.get(e.turn) or {}
            oi = inv.get("opp_intent") or {}
            beta = oi.get("beta") or []
            alpha = oi.get("alpha") or []
            top_slot = int(beta[0]["slot"]) if beta and beta[0].get("slot") is not None else None
            top_species = norm_id(beta[0].get("species")) if beta else None
            out.append(PivotRead(
                turn=e.turn, arrival=sp,
                first_time=sp not in seen_voluntary,
                arrival_revealed=was_revealed,
                slot_true=slot_true,
                beta_top_slot=top_slot,
                slot_correct=((top_slot == slot_true)
                              if (beta and slot_true is not None) else None),
                species_correct=((top_species == sp) if beta else None),
                alpha_top_is_switch=(bool(alpha[0].get("name") == "SWITCH") if alpha else None),
                alpha_switch_p=next((float(a["p"]) for a in alpha
                                     if a.get("name") == "SWITCH"), None),
                loop_step=e.turn in loop_turns,
            ))
            seen_voluntary.add(sp)
        if sp not in reveal_order:
            reveal_order.append(sp)
    return tuple(out)


def chosen_prob(inv: dict) -> "float | None":
    """The recorded probability of the action actually taken (`"96.3%"` → 0.963)."""
    row = (inv.get("actions") or {}).get(inv.get("chosen"))
    if not isinstance(row, dict):
        return None
    try:
        return float(str(row.get("prob", "")).rstrip("%")) / 100.0
    except ValueError:
        return None


def _deltas_by_turn(invocations: Sequence[dict],
                    series: "Sequence[float] | None") -> "dict[int, float]":
    """`{turn: worst per-decision delta}` for a per-decision series (V or P(win)).

    A turn can hold more than one decision (a faint puts a `move_selection` and the `forced_switch`
    it caused on the same turn), so the WORST is kept — a turn is charged with the damage done on
    it, never with an average that a benign second decision could dilute.
    """
    if not series:
        return {}
    out: "dict[int, float]" = {}
    for k, inv in enumerate(invocations):
        t = inv.get("turn")
        if t is None or k + 1 >= len(series):
            continue
        d = float(series[k + 1]) - float(series[k])
        t = int(t)
        if t not in out or d < out[t]:
            out[t] = d
    return out


def analyze_battle(lines: Sequence[str], invocations: Sequence[dict], *,
                   outcome: str = "?", n_turns: "int | None" = None,
                   values: "Sequence[float] | None" = None,
                   win_probs: "Sequence[float] | None" = None,
                   near_zero_frac: float = NEAR_ZERO_FRAC) -> BattleLoops:
    """One battle's whole fold. Never raises and never silently drops: an unusable battle comes back
    with `skipped` set and every count zero, so a run-level scan can report its own coverage."""
    if not lines:
        return BattleLoops(skipped="empty_protocol", outcome=outcome, n_turns=n_turns)
    events = parse_events(lines)
    if not events:
        return BattleLoops(skipped="no_parsable_events", outcome=outcome, n_turns=n_turns)

    dec_by_turn: "dict[int, dict]" = {}
    our_actives: "dict[int, str]" = {}
    inv_by_turn: "dict[int, int]" = {}
    for k, inv in enumerate(invocations or ()):
        t = inv.get("turn")
        if t is None or inv.get("phase") != "move_selection":
            continue
        t = int(t)
        dec_by_turn.setdefault(t, inv)
        inv_by_turn.setdefault(t, k)
        sp = norm_id(((inv.get("our") or {}).get("species")))
        if sp:
            our_actives.setdefault(t, sp)

    our_side, reason = identify_our_side(events, our_actives)
    if our_side is None:
        return BattleLoops(skipped=f"side_undetermined: {reason}", side_reason=reason,
                           outcome=outcome, n_turns=n_turns)
    opp_side = "p2" if our_side == "p1" else "p1"

    baits, loops = mark_loops(bait_events(events, pivot_side=opp_side,
                                          near_zero_frac=near_zero_frac))
    mirror, mirror_loops = mark_loops(bait_events(events, pivot_side=our_side,
                                                  near_zero_frac=near_zero_frac))

    dv = _deltas_by_turn(list(invocations or ()), values)
    dwp = _deltas_by_turn(list(invocations or ()), win_probs)
    joined = tuple(replace(b, inv=inv_by_turn.get(b.turn),
                           chosen_prob=chosen_prob(dec_by_turn.get(b.turn) or {}),
                           delta_v=dv.get(b.turn), delta_win_prob=dwp.get(b.turn))
                   for b in baits)

    kinds = collections.Counter(b.kind for b in baits if b.whiff)
    loop_turns = {t for g in loops for t in g["turns"]}
    bait_turns = {b.turn for b in baits if b.whiff}
    turn_deltas = tuple(
        {"turn": t,
         "delta_v": dv.get(t), "delta_win_prob": dwp.get(t),
         "bucket": ("loop_step" if t in loop_turns
                    else "other_bait" if t in bait_turns else "other")}
        for t in sorted(set(dv) | set(dwp)))
    reads = pivot_reads(events, opp_side=opp_side, dec_by_turn=dec_by_turn,
                        loop_turns=loop_turns)
    n_opp_voluntary = sum(1 for e in events if e.kind == "switch" and e.side == opp_side
                          and not e.forced and not e.drag and e.turn > 0)
    n_our_voluntary = sum(1 for e in events if e.kind == "switch" and e.side == our_side
                          and not e.forced and not e.drag and e.turn > 0)
    return BattleLoops(
        skipped=None, our_side=our_side, side_reason=reason, outcome=outcome,
        n_turns=n_turns if n_turns is not None else (max(e.turn for e in events) or None),
        n_decisions=len(dec_by_turn),
        opp_voluntary_pivots=n_opp_voluntary,
        moved_into_pivots=len(baits),
        whiffs=sum(1 for b in baits if b.whiff),
        whiff_kinds=dict(kinds),
        misses=sum(1 for b in baits if b.kind == "miss"),
        reclicks=sum(1 for b in joined if b.reclick),
        loop_battle=bool(loops),
        worst_loop=max((g["count"] for g in loops), default=0),
        loops=loops, baits=joined, reads=reads, turn_deltas=turn_deltas,
        our_voluntary_pivots=n_our_voluntary,
        mirror_moved_into=len(mirror),
        mirror_whiffs=sum(1 for b in mirror if b.whiff),
        mirror_loop_battle=bool(mirror_loops),
    )


def median(xs: Sequence[float]) -> "float | None":
    """Median without a numpy import (this module stays dependency-free); None on empty."""
    s = sorted(float(x) for x in xs if x is not None)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
