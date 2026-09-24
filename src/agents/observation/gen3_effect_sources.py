"""Source-derive EVERY effect id the volatile encoder can meet in gen3 — the whole class.

Why this module exists
----------------------
``gen3_effects.encode_volatiles`` is crash-don't-drop: an effect id it has not classified
RAISES. That is right for a closed local sim and fatal on an open one, and the curated list
kept being completed one crash at a time (``doomdesire``, ``immunity``, ``magmaarmor``,
``waterveil``, then ``healbell`` — found by the belief-calibration read on 2026-09-24, on a
Metamon ladder team, 4 of 1,200 battles; the 719-team pool has no Heal Bell, so training never
met it). Each earlier fix derived ONE path from source; this derives all of them.

What reaches ``LivePokemon.volatiles``
--------------------------------------
``LiveView`` reads ``mon.effects`` verbatim, and poke-env puts an ``Effect`` on a mon from four
protocol lines (``abstract_battle.py``): ``|-start|``, ``|-activate|`` (target non-empty, minus
the special branches that never call ``start_effect``), ``|-singleturn|`` and ``|-singlemove|``,
plus two silent ones (a ``|move|`` of Minimize, a ``|-prepare|`` of Sky Drop). So the class is:

    every ``add('-start' | '-activate' | '-singleturn' | '-singlemove', …)`` the gen3 sim
    can EXECUTE, turned into the effect id poke-env records for it.

Two halves, deliberately different in kind:

1. **The emission scan is STATIC** (:func:`scan_emissions`): every such ``add(`` call in the
   Showdown source the gen3 format executes — ``sim/*.ts``, ``data/{moves,abilities,items,
   conditions}.ts`` and the mod chain gen3 inherits (gen3 → gen4 → … → gen8 → base; a mod entry
   may override one handler and inherit the rest, so the scan takes the UNION, a superset).
   Entries are kept iff their id is gen3-legal (``agents.gen3_data``); a ``sim/`` line naming
   ``move: X`` / ``ability: X`` / ``item: X`` is kept iff X is. A computed effect argument
   (``'move: ' + this.effectState.sourceEffect``) cannot be read statically, so it must appear in
   :data:`DYNAMIC_EFFECT_EXPANSIONS` with the concrete strings it takes in gen3 — an unlisted one
   is an ERROR, never a skip.
2. **The id mapping is EXECUTED** (:func:`derive_encoder_ids`): each concrete emission is fed as
   a real protocol line into a real ``Gen3Battle`` and the ids that land in ``mon.effects`` are
   read back through ``LiveView``'s own id function. poke-env's branch logic (Skill Swap, Leppa
   Berry, Trick, Mimic never start an effect; everything else does) is therefore MEASURED, not
   restated, and an ``Effect.UNKNOWN`` surfaces as the id ``unknown`` exactly as it would live.

Consumers: ``gen3_effects_test.py`` (fails on drift) and ``main/ladder_drift_scan.py``
(``--effects`` re-runs this against the Showdown tree the public server runs).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from agents.battle.gen3_battle import Gen3Battle

KEYWORDS: Tuple[str, ...] = ("-start", "-activate", "-singleturn", "-singlemove")

#: gen3's Showdown mod chain, most-derived first (``inherit:`` in each ``scripts.ts``; gen8 has
#: none, so it inherits the base ``data/``). Verified by :func:`mod_chain` against the tree.
GEN3_MOD_CHAIN: Tuple[str, ...] = ("gen3", "gen4", "gen5", "gen6", "gen7", "gen8")
_DATA_KINDS: Tuple[str, ...] = ("moves", "abilities", "items", "conditions", "scripts")

#: Conditions whose entry exists in ``conditions.ts`` but that no gen3 battle can start. Each
#: carries its reason. A condition NOT listed here is kept (the conservative direction).
GEN_GATED_CONDITIONS: Dict[str, str] = {
    "dynamax": "gen8 Dynamax — `Dynamax` is not a gen3 mechanic (no Dynamax Band, no rule)",
    "healreplacement": "gen7 Z-move side effect (Z-Memento / Z-Parting Shot); no Z-moves in gen3",
}

#: Items ``data/pokemon/gen3_items.json`` lists that no gen3ou team can hold. Each is verified by
#: the test against Showdown (base entry ``gen: 2`` + ``isNonstandard: "Past"``).
NOT_GEN3_OBTAINABLE_ITEMS: Dict[str, str] = {
    "mysteryberry": "the gen-2 name of Leppa Berry: base `gen: 2, isNonstandard: 'Past'`, "
                    "defined for real only in data/mods/gen2 — the gen3 validator rejects it",
}

#: ``sim/`` lines whose engine guard excludes gen3, keyed ``(file, the effect argument and every
#: argument after it, whitespace-collapsed)`` — the full argument list, because two lines can
#: share an effect and differ only in what follows it (Skill Swap).
SIM_LINE_GATES: Dict[Tuple[str, str], str] = {
    ("sim/battle.ts", "'typechange', realTypeString, '[silent]'"):
        "the switch-in type re-announcement, `if (this.gen >= 7 && !pokemon.terastallized)` "
        "(gen3's typechange comes from Conversion / Conversion 2 / Camouflage / Color Change, "
        "which the data scan keeps)",
    ("sim/battle.ts", "'typeadd', pokemon.addedType, '[silent]'"):
        "the same gen7+ re-announcement; `typeadd` is Forest's Curse / Trick-or-Treat (gen6)",
    ("sim/battle.ts",
     "'Skill Swap', targetAbility.name, sourceAbility.name, `[of] ${target}`"):
        "the `else` of `if (this.gen <= 4 || source.isAlly(target))` — gen3 takes the branch that "
        "names no abilities, which poke-env's Skill Swap branch folds without starting an effect",
}

#: Data lines the gen3 OU FORMAT never executes because a RULE gates them, keyed ``(file, entry,
#: effect expression)``. The test verifies the rule is in gen3 OU's ruleset, so dropping the rule
#: upstream fails CI (and ``ladder_drift_scan --effects`` against the public server).
RULE_GATED_LINES: Dict[Tuple[str, str, str], Tuple[str, str]] = {
    ("data/mods/gen3/moves.ts", "beatup", "'move: Beat Up'"): (
        "Beat Up Nicknames Mod",
        "`if (!this.ruleTable.has('beatupnicknamesmod'))` — gen3's `Standard AG` (data/mods/gen3/"
        "rulesets.ts) includes the mod, so gen3 OU never announces the ally. Without the rule the "
        "line would land as `unknown` (poke-env has no Effect.BEAT_UP) and RAISE"),
}

#: Derived lines that land as ``unknown`` (no poke-env ``Effect`` member) and are PERSISTENT
#: states with no slot: the owner's layout decision. ``(keyword, effect string)`` → why.
PENDING_OWNER_LINES: Dict[Tuple[str, str], str] = {
    ("-start", "Mud Sport"): "gen3 Mud Sport: a volatile on the user (gen5 mod `volatileStatus: "
                             "'mudsport'`) halving Electric moves while it is active",
    ("-start", "move: Water Sport"): "gen3 Water Sport: the same shape for Fire moves",
}

_TRAP_MOVE_LINES: Tuple[str, ...] = (
    "move: Bind", "move: Clamp", "move: Fire Spin", "move: Sand Tomb", "move: Whirlpool",
    "move: Wrap")
_STOCKPILE_LINES: Tuple[str, ...] = ("stockpile1", "stockpile2", "stockpile3")

#: The computed effect arguments the static scan cannot read, each with the concrete strings it
#: takes in a gen3 battle and WHY. Key: ``(file relative to the Showdown root, entry id or ''
#: for a sim/ line, the whitespace-collapsed argument expression)``.
DYNAMIC_EFFECT_EXPANSIONS: Dict[Tuple[str, str, str], Tuple[Tuple[str, ...], str]] = {
    # partiallytrapped.onStart announces the trapping move: every gen3 partial-trap move
    # (`move: X` → Effect.X — the `_TRAP_VARIANTS` in gen3_effects).
    ("data/conditions.ts", "partiallytrapped", "'move: ' + this.effectState.sourceEffect"): (
        _TRAP_MOVE_LINES, "the gen3 partial-trap moves (Bind/Clamp/Fire Spin/Sand Tomb/"
                          "Whirlpool/Wrap)"),
    ("data/mods/gen5/conditions.ts", "partiallytrapped",
     "'move: ' + this.effectState.sourceEffect"): (
        _TRAP_MOVE_LINES, "the same line in the gen5 override gen3 inherits"),
    # perishsong's residual: `perish${duration}`, the count 3 → 1 (perish0 is a literal onEnd).
    ("data/moves.ts", "perishsong", "`perish${duration}`"): (
        ("perish1", "perish2", "perish3"), "Perish Song's residual count, 3 → 1"),
    # stockpile's condition: `'stockpile' + this.effectState.layers`, 1..3 layers.
    ("data/moves.ts", "stockpile", "'stockpile' + this.effectState.layers"): (
        _STOCKPILE_LINES, "Stockpile holds 1..3 layers"),
    ("data/mods/gen3/moves.ts", "stockpile", "'stockpile' + this.effectState.layers"): (
        _STOCKPILE_LINES, "Stockpile holds 1..3 layers (the gen3 override)"),
    ("data/mods/gen6/moves.ts", "stockpile", "'stockpile' + this.effectState.layers"): (
        _STOCKPILE_LINES, "Stockpile holds 1..3 layers (the gen6 override gen3 inherits)"),
    # --- generic engine lines whose computed argument is unreachable in gen3 (EMPTY expansion,
    # the reason is the proof) ---
    ("sim/battle-actions.ts", "", "`move: ${move.name}`"): (
        (), "a protect BROKEN by a `breaksProtect` move (Feint / Shadow Force / Hyperspace "
            "Hole / …): no gen3 move breaks protection"),
    ("sim/battle.ts", "", "this.effect.fullname"): (
        (), "checkMoveMakesContact's Protective Pads announcement: a gen7 item"),
    ("sim/pokemon.ts", "", "sourceEffect.fullname"): (
        (), "setAbility's Mummy / Lingering Aroma case: gen5 / gen9 abilities"),
}


@dataclass(frozen=True)
class Emission:
    """One ``add(<keyword>, <target>, <effect>, …)`` call site."""
    keyword: str
    file: str            # relative to the Showdown root
    entry: str           # the top-level data entry id ('' for a sim/ line)
    line: int
    target: str          # the target argument expression, verbatim
    effect_expr: str     # the effect argument expression, whitespace-collapsed
    literal: Optional[str]  # the effect string when the argument is a literal, else None
    extra: Tuple[str, ...]  # the arguments after the effect, as replayed (see _placeholder)
    args_expr: str = ""     # the effect + following arguments, whitespace-collapsed (gate key)


def _to_id(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def mod_chain(showdown_root: Path) -> Tuple[str, ...]:
    """Walk ``inherit:`` from gen3 and return the chain — asserted equal to
    :data:`GEN3_MOD_CHAIN` by the test, so a re-parented mod cannot go unnoticed."""
    chain: List[str] = []
    cur: Optional[str] = "gen3"
    while cur:
        chain.append(cur)
        txt = (showdown_root / "data/mods" / cur / "scripts.ts").read_text(encoding="utf-8")
        m = re.search(r"\binherit:\s*'([a-z0-9]+)'", txt)
        cur = m.group(1) if m else None
    return tuple(chain)


def _split_args(src: str, start: int) -> Tuple[List[str], int]:
    """Split the argument list of a call whose ``(`` is just before ``start``; returns the
    top-level argument strings and the index of the closing ``)``. Respects quotes, template
    literals (incl. ``${…}``) and nesting."""
    args: List[str] = []
    depth = 0
    cur: List[str] = []
    i = start
    n = len(src)
    while i < n:
        c = src[i]
        if c in "'\"`":
            q = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == q:
                    break
                j += 1
            cur.append(src[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    raise ValueError("unterminated call")


def _literal(expr: str) -> Optional[str]:
    """The string value of a literal argument, else ``None``."""
    if len(expr) >= 2 and expr[0] == expr[-1] and expr[0] in "'\"":
        return expr[1:-1]
    if len(expr) >= 2 and expr[0] == expr[-1] == "`" and "${" not in expr:
        return expr[1:-1]
    return None


_ADD_RE = re.compile(r"\badd\(\s*(['\"])(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\1\s*,")
_ENTRY_RE = re.compile(r"\n\t([a-z0-9]+): (\{|null,)")
_KEY2_RE = re.compile(r"\n\t\t(?:async\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:(]")
_KEY3_RE = re.compile(r"\n\t\t\t(?:async\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:(]")


@dataclass
class _Entry:
    """One top-level data entry in one layer of the mod chain."""
    start: int
    end: int
    null: bool                       # `id: null,` — the mod deletes the entry
    inherit: bool                    # `inherit: true` — shallow-merge onto the parent's
    keys: Dict[str, Tuple[int, int]]  # depth-2 key → (start, end) span in the file
    cond_inherit: bool               # `condition: { inherit: true, … }` — merge one level down
    cond_keys: Dict[str, Tuple[int, int]]


def _parse_entries(txt: str) -> Dict[str, _Entry]:
    heads = list(_ENTRY_RE.finditer(txt))
    out: Dict[str, _Entry] = {}
    for i, h in enumerate(heads):
        start = h.start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(txt)
        if h.group(2) == "null,":
            out[h.group(1)] = _Entry(start, end, True, False, {}, False, {})
            continue
        body = txt[start:end]
        k2 = [(m.group(1), start + m.start()) for m in _KEY2_RE.finditer(body)]
        keys: Dict[str, Tuple[int, int]] = {}
        for j, (k, pos) in enumerate(k2):
            keys[k] = (pos, k2[j + 1][1] if j + 1 < len(k2) else end)
        cond_keys: Dict[str, Tuple[int, int]] = {}
        cond_inherit = False
        if "condition" in keys:
            cs, ce = keys["condition"]
            k3 = [(m.group(1), cs + m.start()) for m in _KEY3_RE.finditer(txt[cs:ce])]
            for j, (k, pos) in enumerate(k3):
                cond_keys[k] = (pos, k3[j + 1][1] if j + 1 < len(k3) else ce)
            cond_inherit = re.search(r"\n\t\t\tinherit: true", txt[cs:ce]) is not None
        out[h.group(1)] = _Entry(start, end, False, "inherit" in keys, keys, cond_inherit,
                                 cond_keys)
    return out


def _key_at(spans: Dict[str, Tuple[int, int]], pos: int) -> Optional[str]:
    for k, (s, e) in spans.items():
        if s <= pos < e:
            return k
    return None


def _effective(layers: List[Optional[_Entry]], idx: int, pos: int) -> bool:
    """Is the code at ``pos`` in layer ``idx``'s entry what the gen3 Dex EXECUTES? Mirrors
    ``sim/dex.ts`` loadData: walking from gen3 toward base, a layer that defines the entry
    WITHOUT ``inherit: true`` (or as ``null``) ends the walk; with it, its keys shadow every
    parent's same key (``onFractionalPriority: undefined`` shadows too), and a ``condition``
    carrying its own ``inherit: true`` is merged one level down the same way."""
    ent = layers[idx]
    assert ent is not None
    k2 = _key_at(ent.keys, pos)
    if k2 is None:
        return False
    k3 = _key_at(ent.cond_keys, pos) if k2 == "condition" else None
    for j in range(idx):  # every MORE-derived layer
        child = layers[j]
        if child is None:
            continue  # transparent
        if child.null or not child.inherit:
            return False  # replaced wholesale
        if k2 in child.keys:
            if k2 != "condition" or not child.cond_inherit:
                return False  # the child's key replaces ours
            if k3 is None or k3 in child.cond_keys:
                return False
    return True


def _placeholder(expr: str) -> str:
    """A protocol value for a COMPUTED argument after the effect, so the executed line has the
    real arity: an ``[of]`` source names the other active mon, a type names a type, anything
    else names a move the target has revealed (Mimic / Leppa Berry index ``mon.moves``)."""
    if "[of]" in expr:
        return "[of] p1a: Zappy"
    lit = re.match(r"'(\[[a-z]+\][^']*)'", expr)
    if lit:
        return lit.group(1)
    if "type" in expr.lower():
        return "Water"
    return "Tackle"


def _emissions_in(root: Path, rel: str) -> List[Tuple[int, Emission]]:
    path = root / rel
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    entries = sorted(((e.start, eid) for eid, e in _parse_entries(txt).items())) \
        if rel.startswith("data/") else []
    out: List[Tuple[int, Emission]] = []
    for m in _ADD_RE.finditer(txt):
        args, _ = _split_args(txt, m.end())
        if len(args) < 2:
            continue
        entry = ""
        for pos, eid in entries:
            if pos > m.start():
                break
            entry = eid
        extra = tuple(v if v is not None else _placeholder(a)
                      for a, v in ((a, _literal(a)) for a in args[2:]))
        out.append((m.start(), Emission(
            keyword=m.group(2), file=rel, entry=entry, line=txt.count("\n", 0, m.start()) + 1,
            target=args[0], effect_expr=" ".join(args[1].split()), literal=_literal(args[1]),
            extra=extra, args_expr=" ".join(", ".join(args[1:]).split()))))
    return out


def _gen3_legal() -> Dict[str, FrozenSet[str]]:
    from agents import gen3_data
    return {
        "moves": frozenset(gen3_data.moves.raw()),
        "abilities": frozenset(gen3_data.abilities.raw()),
        "items": frozenset(gen3_data.items.raw()) - frozenset(NOT_GEN3_OBTAINABLE_ITEMS),
    }


def _prefixed_legal(literal: str, legal: Dict[str, FrozenSet[str]]) -> bool:
    """For a ``sim/`` line: a ``move:``/``ability:``/``item:`` literal is reachable iff the
    named thing is gen3-legal; an unprefixed literal is kept (conservative)."""
    for prefix, kind in (("move:", "moves"), ("ability:", "abilities"), ("item:", "items")):
        if literal.lower().startswith(prefix):
            return _to_id(literal[len(prefix):]) in legal[kind]
    return True


def scan_emissions(showdown_root: Path) -> List[Emission]:
    """Every gen3-REACHABLE ``add(<keyword>, …)`` call (the static half; see the module doc)."""
    legal = _gen3_legal()
    kept: List[Emission] = []
    # sim/ — generic engine code, not keyed; the gates are explicit.
    for p in sorted((showdown_root / "sim").glob("*.ts")):
        for _, e in _emissions_in(showdown_root, f"sim/{p.name}"):
            if e.literal is not None and not _prefixed_legal(e.literal, legal):
                continue
            if (e.file, e.args_expr) in SIM_LINE_GATES:
                continue
            kept.append(e)
    # data/ — resolved through the mod chain exactly as the gen3 Dex merges it.
    for kind in ("moves", "abilities", "items", "conditions"):
        rels = [f"data/mods/{m}/{kind}.ts" for m in GEN3_MOD_CHAIN] + [f"data/{kind}.ts"]
        parsed = []
        for rel in rels:
            path = showdown_root / rel
            parsed.append(_parse_entries(path.read_text(encoding="utf-8", errors="replace"))
                          if path.exists() else {})
        for idx, rel in enumerate(rels):
            for pos, e in _emissions_in(showdown_root, rel):
                if kind in legal and e.entry not in legal[kind]:
                    continue
                if kind == "conditions" and e.entry in GEN_GATED_CONDITIONS:
                    continue
                if (rel, e.entry, e.effect_expr) in RULE_GATED_LINES:
                    continue
                layers = [p.get(e.entry) for p in parsed]
                if not _effective(layers, idx, pos):
                    continue  # shadowed by a more-derived mod: gen3 never runs this line
                if e.literal is not None and not _prefixed_legal(e.literal, legal):
                    continue
                kept.append(e)
    return [e for e in kept if e.target.strip() not in ("''", '""')]  # poke-env: `target != ""`


class UnresolvedDynamicEffect(Exception):
    """A computed effect argument is not in :data:`DYNAMIC_EFFECT_EXPANSIONS`."""


def concrete_lines(emissions: Iterable[Emission]) -> Dict[Tuple[str, str, Tuple[str, ...]],
                                                          List[Emission]]:
    """(keyword, effect string, literal extras) → the emissions that produce it. Raises
    :class:`UnresolvedDynamicEffect` naming EVERY computed argument that has no expansion."""
    out: Dict[Tuple[str, str, Tuple[str, ...]], List[Emission]] = {}
    unresolved: List[str] = []
    for e in emissions:
        if e.literal is not None:
            values: Tuple[str, ...] = (e.literal,)
        else:
            exp = DYNAMIC_EFFECT_EXPANSIONS.get((e.file, e.entry, e.effect_expr))
            if exp is None:
                unresolved.append(f"{e.file}:{e.line} [{e.entry or '-'}] "
                                  f"add('{e.keyword}', {e.target}, {e.effect_expr})")
                continue
            values = exp[0]
        for v in values:
            out.setdefault((e.keyword, v, e.extra), []).append(e)
    if unresolved:
        raise UnresolvedDynamicEffect(
            "computed effect arguments with no entry in gen3_effect_sources."
            "DYNAMIC_EFFECT_EXPANSIONS (add each with the concrete strings it takes in gen3):\n  "
            + "\n  ".join(unresolved))
    return out


def _fresh_battle() -> "Gen3Battle":
    from agents.battle.gen3_battle import Gen3Battle
    quiet = logging.getLogger("gen3_effect_sources")
    b = Gen3Battle("battle-gen3ou-effectsources", "p1user", quiet, gen=3)
    for line in (
        ["", "player", "p1", "p1user", "", ""], ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"], ["", "teamsize", "p2", "6"], ["", "gametype", "singles"],
        ["", "gen", "3"], ["", "start"],
        ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
        ["", "switch", "p2a: Snorlax", "Snorlax, L100, M", "100/100"],
        ["", "turn", "1"],
        # reveal a move, so a line that indexes the target's moves (Leppa Berry) can resolve it
        ["", "move", "p2a: Snorlax", "Tackle", "p1a: Zappy"],
    ):
        b.parse_message(line)
    return b


def effect_ids_for_line(keyword: str, effect: str, extra: Tuple[str, ...] = ()) -> Set[str]:
    """EXECUTE one protocol line on a fresh ``Gen3Battle`` and return the ids it adds to the
    target's ``LiveView`` volatiles. The arguments after the effect are the source's own (a
    computed one gets a :func:`_placeholder`), so the branches that read them (Trick's ``[of]``
    partner, Mimic's move, Skill Swap's abilities) run their real code."""
    from agents.battle.live_view import _id
    b = _fresh_battle()
    mon = b.get_pokemon("p2a: Snorlax")
    before = {str(_id(e)) for e in mon.effects}
    line = ["", keyword, "p2a: Snorlax", effect, *extra]
    prev = logging.root.manager.disable
    logging.disable(logging.WARNING)  # Effect.UNKNOWN's warning; the id itself is the signal
    try:
        b.parse_message(line)
    finally:
        logging.disable(prev)
    return {str(_id(e)) for e in mon.effects} - before


#: The two silent start_effect sites (no ``-start``/``-activate`` line): a ``|move|`` of
#: Minimize and a ``|-prepare|`` of Sky Drop (gen4+, not gen3-legal).
SILENT_EFFECT_SOURCES: Dict[str, str] = {
    "minimize": "abstract_battle `|move|` handler: start_effect('MINIMIZE') on a Minimize use",
}


def derive_encoder_ids(showdown_root: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """Every id ``encode_volatiles`` can meet in gen3 → the ``(keyword, effect, file:line)``
    lines that put it there. The union of the executed emissions and
    :data:`SILENT_EFFECT_SOURCES`."""
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for (kw, eff, extra), ems in sorted(concrete_lines(scan_emissions(showdown_root)).items()):
        for vid in effect_ids_for_line(kw, eff, extra):
            for e in ems:
                out.setdefault(vid, []).append((kw, eff, f"{e.file}:{e.line}"))
    for vid, why in SILENT_EFFECT_SOURCES.items():
        out.setdefault(vid, []).append(("(silent)", vid, why))
    return out


def unclassified(derived: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, List[str]]:
    """The derived ids the encoder would RAISE on, minus the owner-pending ``unknown`` lines —
    empty means every effect the gen3 sim can announce is classified. Shared by the test and
    ``ladder_drift_scan --effects``."""
    from agents.observation.gen3_effects import GEN3_VOLATILE_TO_SLOT, NOT_A_VOLATILE
    bad: Dict[str, List[str]] = {}
    for vid, srcs in derived.items():
        if vid in GEN3_VOLATILE_TO_SLOT or vid in NOT_A_VOLATILE:
            continue
        rest = [f"{kw} {eff!r} ({where})" for kw, eff, where in srcs
                if not (vid == "unknown" and (kw, eff) in PENDING_OWNER_LINES)]
        if rest:
            bad[vid] = rest
    return bad


