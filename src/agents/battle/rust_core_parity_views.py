"""The Rust Core parity harness — slice V, the TRUTH AUDIT (``gen3_core_parity_views_v1``).

At EVERY decision of a recorded battle, for BOTH viewers, the ``LiveView`` + ``LegalActions``
TRAINING builds (a ``Gen3Battle`` fed the viewer's per-side text through
:mod:`agents.battle.offline_feed`, read at the exact chunk ``Player._handle_battle_message``
would dispatch the decision on) against the simulator's own projection of the same board — the
port's ``one_sided_view`` captured at that board by ``core_events --views``, and the ENGINE truth
beside it. Field by field, **no allowlist**.

**Why ``one_sided_view`` is the projection and not the M1 core.** M1's ``CoreEvent`` stream is
the EVENT reading; it carries no board. ``view.rs::one_sided_view`` is the only projection of the
engine board onto one side that exists today, and it is exactly what M2's ``present(board, events,
side)`` replaces — so M2 swaps the producer under this slice and keeps the comparison. That is
why the slice rides ``rust_core_parity`` (one harness, one corpus, one core call per battle)
rather than a second harness.

Every compared field carries a CLASS (:data:`MON_FIELDS`, :data:`SIDE_FIELDS`,
:data:`VIEW_FIELDS`, :data:`LEGAL_FIELDS`):

* **SIM-FACT** — the projection reads the ENGINE (boosts, HP, status, active, fainted, types,
  side conditions, weather, our own moves / PP / item / ability / stats, team size). A
  divergence is a READING bug on one side or the other, and the truth decides which.
* **PRESENTATION** — the value is a poke-env rule, not a sim fact (an opponent's PP as a sighting
  count, the volatile fold, the sleep/toxic counter, the protect streak, reveal gating, slot
  order). The projection reproduces the rule, each one NAMED in :data:`RULES` with the poke-env
  line it mirrors, and the comparison is still exact.

Beside the field comparison, the TRUTH checks that no projection can make — the reading of a
REVEALED opponent fact against the engine's own value (the other viewer's ``ours`` block, which
is engine-sourced): a revealed item / ability / move must be the engine's, a consumed item must
no longer be held, and every sim-state volatile in :data:`TRUTH_VOLATILES` must be present on
the reading exactly when the engine holds it (both sides).

🚨 **The motivating class is a READING bug about a sim fact that every other gate is blind to**:
poke-env dropped every Baton-Passed stat stage for the fork's whole life (fixed 2026-08-23),
and every read-model / obs gate agreed with it, because they all read the same ``Pokemon``. Here
the other side of the comparison is the ENGINE, so that class fails on the first decision after
the pass (``rust_core_parity_test.py::test_the_view_slice_catches_a_dropped_baton_pass``).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions, LivePokemon, LiveView
from agents.battle.offline_feed import feed_line, new_battle, player_names
from agents.battle.poke_env_findings import explain

SIM = "SIM-FACT"
RULE = "PRESENTATION"

#: The named PRESENTATION rules the projection reproduces — ``id: (the rule, the poke-env line
#: it mirrors, where the projection applies it)``. A divergence on a PRESENTATION field is fixed
#: by making the projection reproduce the rule, never by relaxing the comparison.
RULES: Dict[str, Tuple[str, str, str]] = {
    "V1-reveal": ("an opposing mon has a row only once a |switch|/|drag| line showed it; own "
                  "`revealed` is set by the same line", "abstract_battle.py switch → "
                  "Pokemon.switch_in `_revealed = True`", "view.rs SideObservation.order/own_seen"),
    "V2-slot-order": ("own mons in the FIRST |request|'s roster order; opponents in REVEAL order",
                      "Battle._update_team_from_request (dict never reordered); "
                      "opponent_team dict insertion", "view.rs own_order / order"),
    "V3-opp-pp": ("an opposing move's PP = max_pp − sightings, a sighting against a Pressure "
                  "holder costing two", "Pokemon.moved → Move.use(pressure); "
                  "abstract_battle._pressure_on", "view.rs MoveObs + view_adapter._move"),
    "V4-volatiles": ("the |-start|/|-end|/|-activate|/|-singleturn|/|-singlemove| fold, cleared on "
                     "switch-out and faint, `ends_on_turn` dropped at |turn|, countable effects "
                     "counted", "Pokemon.start_effect/end_effect/end_turn/_clear_effects",
                     "view.rs fold_volatiles + view_adapter._volatiles"),
    "V5-status-counter": ("poke-env's own sleep counter (per |move|/|cant| line) and the toxic "
                          "STAGE (per residual `[from] psn` chip since the switch-in — the sim's "
                          "since `gen3_pe_reading_fixes_v1`) — see `designs/rust_sim/"
                          "one_sided_view.md` §4b for its rules and its truth",
                          "pokemon.py moved/cant_move/note_residual_chip/cure_status/switch_out, "
                          "the `status` setter, battle.py switch", "view.rs fold_status"),
    "V6-protect-counter": ("a plain consecutive-stall-move count", "Pokemon.moved/cant_move/"
                           "switch_out", "view.rs fold_status"),
    "V7-item-disclosure": ("an opposing item is known from |-item|/|-enditem| or a `[from] item:` "
                           "clause on |-damage|/|-heal| only; `consumed_item` from |-enditem|",
                           "abstract_battle -item/-enditem/_check_*_message_for_item; "
                           "Pokemon.end_item", "view.rs observe"),
    "V8-ability-slots": ("the two-slot ability (base + temporary), the single-possible-ability "
                         "inference, the Trace double assignment", "Pokemon.ability setter, "
                         "_update_from_pokedex, switch_out", "view_adapter._ability"),
    "V9-opp-hidden": ("an opponent's spread / stats / exact HP are unknown: ivs/evs/nature None, "
                      "stats all-None, HP as the ceil-% fold", "LivePokemon.from_pokemon "
                      "(is_own=False)", "view.rs mon_json(own=false)"),
    "V11-screens": ("a timed screen is stored as the TURN it started, Spikes as its layer count",
                    "abstract_battle._side_start", "view.rs side_conditions_json"),
    "V12-weather-turns": ("`turns_active` = now − the |-weather| set turn", "LiveView._fold_weather",
                          "view.rs weather_json"),
    "V13-legal-flags": ("the legality flags are poke-env's parse of the request", "Battle.parse_request",
                        "view_adapter.legal_actions_from_payload"),
}

#: ``LivePokemon`` field → (class on OUR side, class on THEIR side, rule id or None).
MON_FIELDS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "species": (SIM, SIM, None),
    "active": (SIM, SIM, None),
    "fainted": (SIM, SIM, None),
    "revealed": (RULE, RULE, "V1-reveal"),
    "hp_fraction": (SIM, SIM, None),
    "status": (SIM, SIM, None),
    "types": (SIM, SIM, None),
    "moves": (SIM, RULE, "V3-opp-pp"),
    "item": (SIM, RULE, "V7-item-disclosure"),
    "ability": (SIM, RULE, "V8-ability-slots"),
    "boosts": (SIM, SIM, None),       # V10 retired: the fork clears them at the faint (PE-V10)
    "volatiles": (RULE, RULE, "V4-volatiles"),
    "base_stats": (SIM, SIM, None),
    "ivs": (SIM, RULE, "V9-opp-hidden"),
    "evs": (SIM, RULE, "V9-opp-hidden"),
    "nature": (SIM, RULE, "V9-opp-hidden"),
    "spread_known": (SIM, RULE, "V9-opp-hidden"),
    "consumed_item": (RULE, RULE, "V7-item-disclosure"),
    "status_counter": (RULE, RULE, "V5-status-counter"),
    "protect_counter": (RULE, RULE, "V6-protect-counter"),
    "stats": (SIM, RULE, "V9-opp-hidden"),
    "current_hp": (SIM, RULE, "V9-opp-hidden"),
    "max_hp": (SIM, RULE, "V9-opp-hidden"),
}
SIDE_FIELDS: Dict[str, Tuple[str, Optional[str]]] = {
    "team_size": (SIM, None),
    "side_conditions": (SIM, "V11-screens"),
    "mons[order]": (RULE, "V2-slot-order"),
    "active": (SIM, None),
}
VIEW_FIELDS: Dict[str, Tuple[str, Optional[str]]] = {
    "turn": (SIM, None), "finished": (SIM, None), "won": (SIM, None), "lost": (SIM, None),
    "weather.weather": (SIM, None), "weather.is_permanent": (SIM, None),
    "weather.turns_active": (RULE, "V12-weather-turns"),
}
LEGAL_FIELDS: Dict[str, Tuple[str, Optional[str]]] = {
    "move_slots": (SIM, None), "switches": (SIM, None), "force_switch": (SIM, None),
    "trapped": (RULE, "V13-legal-flags"), "maybe_trapped": (RULE, "V13-legal-flags"),
    "wait": (SIM, None), "struggle": (SIM, None), "own_hp_typed_id": (SIM, None),
}

#: Engine volatile (``search::volatile_names``) → the ``LivePokemon.volatiles`` key poke-env
#: holds for it. The TRUTH check is two-way on this set: the reading has the key iff the engine
#: holds the volatile. Only conditions the protocol ANNOUNCES belong here — a condition the sim
#: never prints (choice lock, the stall counter) has no reading to check.
TRUTH_VOLATILES: Dict[str, str] = {
    "confusion": "confusion", "curse": "curse", "disable": "disable", "encore": "encore",
    "focusenergy": "focusenergy", "leechseed": "leechseed", "substitute": "substitute",
    "taunt": "taunt", "yawn": "yawn", "attract": "attract",
}


def field_census() -> Dict[str, int]:
    """How many compared fields sit in each class (the report's classification count)."""
    c: collections.Counter = collections.Counter()
    for own, opp, _ in MON_FIELDS.values():
        c[f"mon/ours {own}"] += 1
        c[f"mon/theirs {opp}"] += 1
    for table in (SIDE_FIELDS, VIEW_FIELDS, LEGAL_FIELDS):
        for cls, _ in table.values():
            c[cls] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------

@dataclass
class ViewCensus:
    battles: int = 0
    viewers: int = 0
    decisions: int = 0
    fields: int = 0
    truth_checks: collections.Counter = field(default_factory=collections.Counter)
    #: The core's TRUTH AUDIT checks (``present::audit::check_view`` against the engine).
    board_checks: collections.Counter = field(default_factory=collections.Counter)
    #: Per named rule the audit applies (V15), the facts the view held differently from the
    #: engine, legally. Printed, never a divergence.
    rules_fired: collections.Counter = field(default_factory=collections.Counter)
    #: The KNOWN poke-env READING findings (``agents.battle.poke_env_findings``): per finding id,
    #: the core-vs-reading field differences it explains (``known``) and the DECISIONS it touched
    #: (``known_decisions``, the per-1,000 numerator). Printed, never a divergence — and never a
    #: blanket tolerance: a difference no finding's predicate explains is a divergence.
    known: collections.Counter = field(default_factory=collections.Counter)
    known_decisions: collections.Counter = field(default_factory=collections.Counter)
    known_examples: Dict[str, Any] = field(default_factory=dict)
    _known_seen: set = field(default_factory=set)
    divergences: collections.Counter = field(default_factory=collections.Counter)
    examples: Dict[str, Any] = field(default_factory=dict)
    refused: List[str] = field(default_factory=list)
    #: Per battle label → decisions with at least one divergence (the per-1,000 denominator).
    bad_decisions: collections.Counter = field(default_factory=collections.Counter)
    #: Battles OUTSIDE the slice's scope, by format (printed on every run, never silent).
    out_of_scope: collections.Counter = field(default_factory=collections.Counter)

    def diverge(self, key: str, example: Any) -> None:
        self.divergences[key] += 1
        self.examples.setdefault(key, example)

    def render(self) -> str:
        head = (f"{self.battles} battles, {self.viewers} viewers, {self.decisions} decisions, "
                f"{self.fields} field comparisons, truth checks {dict(self.truth_checks)}, "
                f"core board checks {dict(self.board_checks)}, "
                f"named rules {dict(self.rules_fired)}, "
                f"KNOWN poke-env findings (facts) {dict(self.known)} "
                f"(decisions) {dict(self.known_decisions)}")
        if self.out_of_scope:
            head += (f"; OUT OF SCOPE (not gen3ou — the training obs path is gen3ou-only): "
                     f"{dict(self.out_of_scope)}")
        if not self.divergences and not self.refused:
            return f"✅ slice V: 0 divergences — {head}"
        bad = sum(self.bad_decisions.values())
        out = [f"❌ slice V: {sum(self.divergences.values())} divergences in "
               f"{len(self.divergences)} classes at {bad} decisions "
               f"({1000.0 * bad / max(self.decisions, 1):.2f} per 1,000), "
               f"{len(self.refused)} refused — {head}"]
        for k, n in self.divergences.most_common():
            out.append(f"   {n:7d}  {k}\n            e.g. {self.examples[k]}")
        for r in self.refused[:10]:
            out.append(f"   REFUSED {r}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# the decision points — where TRAINING reads the board
# ---------------------------------------------------------------------------

def decision_points(chunks: Sequence[Tuple[str, str]], viewer: str,
                    battle: Optional[Gen3Battle] = None,
                    packed_team: Optional[str] = None) -> Iterator[Tuple[int, Gen3Battle]]:
    """``(chunk index, battle)`` at every decision ``viewer``'s player would take, in order.

    A DISPATCH MIRROR of ``Player._handle_battle_message``'s ``should_request``: a chunk that
    carried a non-empty ``|request|`` dispatches the decision AFTER the whole chunk is parsed,
    and ``_handle_battle_request`` returns without choosing on a ``wait`` request. The battle is
    the SAME object across yields (it is the live reading), so read it before resuming."""
    lines = [ln for side, c in chunks if side == viewer for ln in c.split("\n")]
    b = battle if battle is not None else new_battle(viewer, player_names(lines),
                                                     packed_team=packed_team)
    for i, (side, text) in enumerate(chunks):
        if side != viewer:
            continue
        should = False
        for line in text.split("\n"):
            feed_line(b, line)
            if line.startswith("|request|") and len(line) > len("|request|"):
                should = True
        if should and not b._wait and not b.finished:
            yield i, b


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def _mon_value(m: LivePokemon, name: str) -> Any:
    v = getattr(m, name)
    if name in ("boosts", "volatiles", "base_stats", "stats"):
        return dict(v)
    if name == "moves":
        return [(x.id, x.current_pp, x.max_pp) for x in v]
    return v


def _same(x: Any, y: Any) -> bool:
    """TYPE-strict equality (an int is not a float, whatever their values) — the core column's
    rule, the same as slice E's: the core renders the reading, so there is no rounding to excuse."""
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    return x == y


def _cmp_mon(p: LivePokemon, v: LivePokemon, side: str, census: ViewCensus, where: str,
             column: str = "") -> int:
    bad = 0
    for f in dc_fields(LivePokemon):
        own_cls, opp_cls, rule = MON_FIELDS[f.name]
        cls = own_cls if side == "ours" else opp_cls
        census.fields += 1
        x, y = _mon_value(p, f.name), _mon_value(v, f.name)
        if column:
            same = _same(x, y)
        elif f.name == "hp_fraction":
            same = abs(float(x) - float(y)) <= 1e-9
        else:
            same = x == y
        if not same and column == "core":
            # The core reads the TRUTH; a difference a KNOWN poke-env finding explains, value-aware,
            # is that finding — counted, never a divergence (``poke_env_findings``).
            fid = explain(f.name, p, v)
            if fid is not None:
                census.known[fid] += 1
                census.known_examples.setdefault(fid, (where, side, p.species, x, y))
                if (fid, where) not in census._known_seen:
                    census._known_seen.add((fid, where))
                    census.known_decisions[fid] += 1
                continue
        if not same:
            tag = f"{cls}{'/' + rule if cls == RULE and rule else ''}"
            census.diverge(f"{_col(column)}[{tag}] {side}.{f.name}",
                           (where, p.species, "reading", x, column or "projection", y))
            bad += 1
    return bad


def _col(column: str) -> str:
    return f"[{column}]" if column else ""


def compare_decision(live_p: LiveView, legal_p: Optional[LegalActions], live_v: LiveView,
                     legal_v: Optional[LegalActions], census: ViewCensus, where: str,
                     column: str = "") -> int:
    """The reading (``live_p`` / ``legal_p``) against a column — the port's projection (the view
    road's, ``column=""``) or the core's ``present()`` (``column="core"``, compared TYPE-strict) —
    field by field. Returns the number of divergent fields."""
    bad = 0
    eq = _same if column else (lambda x, y: x == y)

    def note(key: str, x: Any, y: Any, cls: str, rule: Optional[str]) -> None:
        nonlocal bad
        census.fields += 1
        if not eq(x, y):
            tag = f"{cls}{'/' + rule if cls == RULE and rule else ''}"
            census.diverge(f"{_col(column)}[{tag}] {key}", (where, "reading", x, column or "projection", y))
            bad += 1

    for k, (cls, rule) in VIEW_FIELDS.items():
        if k.startswith("weather."):
            a = getattr(live_p.weather, k.split(".")[1])
            b = getattr(live_v.weather, k.split(".")[1])
        else:
            a, b = getattr(live_p, k), getattr(live_v, k)
        note(k, a, b, cls, rule)
    for side in ("ours", "opp"):
        sp, sv = getattr(live_p, side), getattr(live_v, side)
        note(f"{side}.team_size", sp.team_size, sv.team_size, *SIDE_FIELDS["team_size"])
        note(f"{side}.side_conditions", dict(sp.side_conditions), dict(sv.side_conditions),
             *SIDE_FIELDS["side_conditions"])
        note(f"{side}.active", sp.active.species if sp.active else None,
             sv.active.species if sv.active else None, *SIDE_FIELDS["active"])
        order_p = [m.species for m in sp.mons]
        order_v = [m.species for m in sv.mons]
        note(f"{side}.mons[order]", order_p, order_v, *SIDE_FIELDS["mons[order]"])
        by_v = {m.species: m for m in sv.mons}
        for m in sp.mons:
            if m.species in by_v:
                bad += _cmp_mon(m, by_v[m.species], side, census, where, column)
    if (legal_p is None) != (legal_v is None):
        note("legal[presence]", legal_p is None, legal_v is None, SIM, None)
    elif legal_p is not None and legal_v is not None:
        for k, (cls, rule) in LEGAL_FIELDS.items():
            note(f"legal.{k}", getattr(legal_p, k), getattr(legal_v, k), cls, rule)
    return bad


def core_checks(cap: Mapping[str, Any], vi: int, live_p: LiveView,
                legal_p: Optional[LegalActions], census: ViewCensus, where: str) -> int:
    """The CORE column (M2, ``gen3_core_present_v1``): the core's ``present()`` view and legality —
    built from the viewer's stream ALONE — against the reading, TYPE-strict, every field; its
    11-bit action mask against ``Gen3ActionMasker.mask_from_legal`` of the reading's legality; and
    the core's own TRUTH AUDIT of that view against the ENGINE (``present::audit::check_view``).
    The core reads the TRUTH: a field where poke-env is KNOWN to be wrong is counted under its
    finding (``agents.battle.poke_env_findings``), value-aware, never as a divergence. Returns
    divergent checks."""
    from agents.action.mask_generator import Gen3ActionMasker
    from agents.battle.core_view import legal_actions_from_core, live_view_from_core

    live_c = live_view_from_core(cap["core"][vi], battle_tag=live_p.battle_tag)
    legal_json = cap["legal"][vi]
    legal_c = legal_actions_from_core(legal_json, None)
    bad = compare_decision(live_p, legal_p, live_c, legal_c, census, where, column="core")
    if legal_p is not None and legal_json is not None:
        census.fields += 1
        want = [int(x) for x in Gen3ActionMasker.mask_from_legal(legal_p)]
        if legal_json["mask"] != want:
            census.diverge("[core] mask", (where, "reading", want, "core", legal_json["mask"]))
            bad += 1
    audit = cap["audit"][vi]
    census.board_checks["BOARD"] += int(audit["checks"])
    census.rules_fired.update(audit.get("rules_fired", {}))
    for cls, detail in audit["divergences"]:
        census.diverge(f"[BOARD] {cls}", (where, detail))
        bad += 1
    return bad


def _bare(move_id: str) -> str:
    return "hiddenpower" if move_id.startswith("hiddenpower") else move_id


def truth_checks(live_p: LiveView, other_payload: Mapping[str, Any],
                 truth: Sequence[Sequence[Mapping[str, Any]]], viewer_idx: int,
                 census: ViewCensus, where: str) -> int:
    """The READING of every revealed opposing fact, and every sim-state volatile on both sides,
    against the ENGINE. ``other_payload`` is the OTHER viewer's one-sided view: its ``ours``
    block is engine-sourced (item, ability, moveset, types), which is exactly the truth a
    revealed-opponent reading has to equal."""
    bad = 0

    def check(kind: str, ok: bool, example: Any) -> None:
        nonlocal bad
        census.truth_checks[kind] += 1
        if not ok:
            census.diverge(f"[TRUTH] {kind}", (where,) + tuple(example))
            bad += 1

    eng_opp = {str(r["species"]): r for r in other_payload["ours"]["mons"]}
    for m in live_p.opp.mons:
        e = eng_opp.get(m.species)
        if e is None:
            check("opp.species-exists", False, (m.species, "not on the engine's team"))
            continue
        held = str(e.get("item") or "")
        if m.item is not None:
            check("opp.item", m.item == held, (m.species, "reading", m.item, "engine", held))
        elif m.consumed_item is not None:
            check("opp.consumed-item-not-held", held != m.consumed_item,
                  (m.species, "consumed", m.consumed_item, "engine holds", held))
        if m.ability is not None:
            check("opp.ability", m.ability == e.get("ability"),
                  (m.species, "reading", m.ability, "engine", e.get("ability")))
        eng_moves = {_bare(str(x["move_id"])) for x in e.get("moves", ())}
        for mv in m.moves:
            check("opp.move-in-moveset", _bare(mv.id) in eng_moves,
                  (m.species, "reading", mv.id, "engine", sorted(eng_moves)))
        if not m.fainted:
            check("opp.types", list(m.types) == list(e.get("types", ())),
                  (m.species, "reading", m.types, "engine", e.get("types")))
    # sim-state volatiles, both sides, active mons only (a benched mon holds none)
    for side, eng_side in (("ours", truth[viewer_idx]), ("opp", truth[1 - viewer_idx])):
        by_species = {str(r["species"]): r for r in eng_side}
        for m in getattr(live_p, side).mons:
            e = by_species.get(m.species)
            if e is None or not m.active or m.fainted:
                continue
            eng = {TRUTH_VOLATILES[v] for v in e.get("vol", ()) if v in TRUTH_VOLATILES}
            rd = {k for k in m.volatiles if k in TRUTH_VOLATILES.values()}
            check(f"{side}.volatiles", eng == rd,
                  (m.species, "reading", sorted(rd), "engine", sorted(eng)))
    return bad


# ---------------------------------------------------------------------------
# one battle
# ---------------------------------------------------------------------------

#: The formats slice V audits. The TRAINING observation path is gen3ou-only (the encoder is
#: gen3ou-scoped and fail-loud outside it); the `gen3customgame` corpora the event slice runs
#: (protocol scenarios, byte-fuzz shapes, the Forecast sweep) carry ability-less and
#: species-clause-breaking hacked teams and no HP-percentage mod — inputs training never sees.
SCOPE_FORMATS = frozenset({"gen3ou"})


def check_views(label: str, chunks: Sequence[Tuple[str, str]], caps: Sequence[Mapping[str, Any]],
                census: ViewCensus, teams: Optional[Mapping[str, str]] = None,
                battle_factory=None, format_id: str = "gen3ou") -> None:
    """Every decision of both viewers of one battle. ``caps`` is ``core_events --views``'
    ``views`` list; ``teams`` the packed team per viewer (what each ``Player`` was built with);
    ``battle_factory(viewer)`` (a test seam) builds the reading's battle. A battle whose
    ``format_id`` is outside :data:`SCOPE_FORMATS` is COUNTED as out of scope, never compared."""
    from agents.battle.view_adapter import read_models_from_payload

    if format_id not in SCOPE_FORMATS:
        census.out_of_scope[format_id] += 1
        return
    census.battles += 1
    for vi, viewer in enumerate(("p1", "p2")):
        census.viewers += 1
        mine = [c for c in caps if c["new_request"][vi]]
        want = [c for c in mine if not (c[viewer].get("request") or {}).get("wait")]
        start = battle_factory(viewer) if battle_factory is not None else None
        pts = decision_points(chunks, viewer, start, (teams or {}).get(viewer))
        prev_after = 0
        k = 0
        for i, battle in pts:
            # the capture whose write carried this request chunk
            while k < len(mine) and mine[k]["after"] <= i:
                prev_after = mine[k]["after"]
                k += 1
            if k >= len(mine) or not (prev_after <= i < mine[k]["after"]):
                census.diverge("[ALIGN] reading decided where the core shipped no request",
                               (label, viewer, i))
                continue
            cap = mine[k]
            later = [j for j in range(i + 1, cap["after"]) if chunks[j][0] == viewer]
            if later:
                # More of the viewer's stream follows this decision inside the capture's write,
                # so the captured board is not the one it decided on. Never skipped: a capture the
                # core failed to take lands here too (the refusal test drops one to prove it).
                census.diverge("[ALIGN] the reading decided on a board the core captured no "
                               "view of", (label, viewer, i, cap["after"]))
                continue
            if cap not in want:
                census.diverge("[ALIGN] reading decided on a `wait` request", (label, viewer, i))
                continue
            census.decisions += 1
            where = f"{label}/{viewer}@t{battle.turn}#c{i}"
            sv = battle.strict_view()
            live_v, legal_v, _ = read_models_from_payload(cap[viewer], battle_tag=battle.battle_tag)
            bad = compare_decision(sv.live, sv.legal, live_v, legal_v, census, where)
            other = "p2" if viewer == "p1" else "p1"
            bad += truth_checks(sv.live, cap[other], cap["truth"], vi, census, where)
            if "core" in cap:
                bad += core_checks(cap, vi, sv.live, sv.legal, census, where)
            if bad:
                census.bad_decisions[label] += 1
            want.remove(cap)
        for c in want:
            census.diverge("[ALIGN] the core shipped a decision the reading never took",
                           (label, viewer, c["after"]))
