"""THE gen3ou FORMAT SPEC — the one declaration of the rules every game we play is under.

Every game the project creates is ``format_id: "gen3ou"``; the model never sees another format. This
module states that format's rules ONCE, each with its SOURCE, and gives every rule exactly ONE handling
STORY (``designs/endstate/design_format_spec.md``, the always-current doc for it):

* :attr:`Story.PRIOR_ZERO` — a BANNED entity (species, item, move, ability, or a species-conditional
  move). Its Smogon prior mass is 0 (filtered at acquisition, renormalised, a throwing guard at the
  facade), every model prior table gives it the ILLEGAL value instead of a liftable floor, and team
  validation flags it in any pool we play.
* :attr:`Story.BOARD_STATE` — a clause the engine enforces during the battle (Sleep Clause Mod, Freeze
  Clause Mod). The observation carries the state it reads, and the damage-op / move-resolution rules
  ask THIS module whether the clause is in force (:func:`sleep_clause_mod`, :func:`freeze_clause_mod`)
  instead of hard-coding it.
* :attr:`Story.TEAM_BUILDING` — a rule about which TEAMS are legal (Species Clause, the Baton Pass
  clauses and combos, Accuracy Trap). Enforced by the engine's validator and by our team validation
  (``agents.gen3_data.team_legality``); the model needs nothing more.
* :attr:`Story.ENGINE` — a battle mechanic the engine applies whose consequence the model already reads
  through the observation (HP Percentage Mod, Switch Priority Clause Mod, Endless Battle Clause, …).
* :attr:`Story.COSMETIC` — no effect on a battle the model can observe (Nickname Clause, Cancel Mod).
* :attr:`Story.RULESET` — a container (Standard, Standard AG); its parts are rules of their own.

WHICH Showdown: the spec is the LIVE LADDER's format (Showdown master, the server a rated game is played
on). Our engine is pinned (``deps/pokemon-showdown`` @ :data:`PINNED_COMMIT`), and where the two differ the
entry says so (:attr:`Ban.on_pinned`, :data:`PINNED_DIFFERENCES`): a pool team the pinned engine plays may
still be illegal on the ladder, which is exactly what team validation reports. ``main.format_drift``
compares this spec against master (a committed snapshot, offline; ``ladder_drift_scan`` re-runs it
against a fresh clone) and FAILS on any new or removed ban or clause.

Poke-env-free (``src/poke_env_import_gate_test.py``); ids are Showdown ids (lowercase alphanumerics).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

#: The one format every game is created with (``battle_format``; the Rust core's ``format_id``).
FORMAT_ID = "gen3ou"
#: ``deps/pokemon-showdown``'s commit — the engine every training / eval game runs.
PINNED_COMMIT = "e0551883ff8c676937a39ae8f4d6c0caf9de1613"
#: The Showdown master commit the committed drift snapshot was taken at (``main/format_drift_snapshot/``).
MASTER_SNAPSHOT_COMMIT = "c046106cbe075931b1ff8d8b800ff5be47a85f96"
_M = f"master@{MASTER_SNAPSHOT_COMMIT[:8]}"


def to_id(name: str) -> str:
    """A Showdown display name as its id (``toID``: lowercase, alphanumerics only)."""
    return "".join(c for c in name.lower() if c.isalnum())


class Story(enum.Enum):
    """How a rule reaches (or deliberately does not reach) the model. Exactly one per rule."""
    PRIOR_ZERO = "prior_zero"
    BOARD_STATE = "board_state"
    TEAM_BUILDING = "team_building"
    ENGINE = "engine"
    COSMETIC = "cosmetic"
    RULESET = "ruleset"


@dataclass(frozen=True)
class Ban:
    """One banned thing. ``kind``: ``species`` / ``item`` / ``move`` / ``ability``; ``species_move`` (a per-set
    combo of one species and one move, ``ids = (species, move)``: the move is banned FOR that species, so its
    prior there is 0); ``combo`` (any other per-SET combination, Showdown's ``A + B``: one set holding every
    part is banned — no single entity is, so it stays with team validation; ``A ++ B`` would be per team and
    gen3ou has none). ``ids``: the entity id, or the parts. ``via``: the rule that bans it. ``on_pinned``:
    the pinned engine's format bans it too (False = a master-only ban)."""
    kind: str
    name: str
    ids: Tuple[str, ...]
    via: str
    source: str
    on_pinned: bool = True


@dataclass(frozen=True)
class Rule:
    """One rule of the resolved gen3ou rule table (the format's own ruleset entries and everything they
    include), with its ONE story, the reason for it, where the rule is defined, and the bans it carries."""
    name: str
    parent: str
    story: Story
    reason: str
    source: str
    bans: Tuple[Ban, ...] = ()

    @property
    def id(self) -> str:
        return to_id(self.name)


# --------------------------------------------------------------------------------------------------
# The bans. Sources: the format's banlist line (pinned `config/formats.ts:4421`; master `:4699`), the
# clause banlists in `data/rulesets.ts`, and the Uber entries of `data/mods/gen3/formats-data.ts`.
# --------------------------------------------------------------------------------------------------
_FMT = "pinned:config/formats.ts:4421"
_FMT_M = f"{_M}:config/formats.ts:4699"

#: The gen-3 Uber tier (pinned `data/mods/gen3/formats-data.ts`, each entry's line; identical on master).
UBER_SPECIES: Tuple[Tuple[str, int], ...] = (
    ("mewtwo", 506), ("mew", 509), ("wynaut", 638), ("wobbuffet", 641), ("lugia", 752), ("hooh", 755),
    ("latias", 1145), ("latios", 1148), ("kyogre", 1151), ("groudon", 1154), ("rayquaza", 1157),
    ("deoxys", 1163), ("deoxysattack", 1166), ("deoxysdefense", 1169), ("deoxysspeed", 1172),
)

#: The format banlist's ENTITY bans (and the one species + move combo): prior 0.
FORMAT_BANS: Tuple[Ban, ...] = (
    *(Ban("species", sid, (sid,), "Uber", f"pinned:data/mods/gen3/formats-data.ts:{line} (tier: \"Uber\")")
      for sid, line in UBER_SPECIES),
    Ban("species_move", "Smeargle + Ingrain", ("smeargle", "ingrain"), "banlist", _FMT),
    Ban("ability", "Sand Veil", ("sandveil",), "banlist", _FMT),
    Ban("ability", "Soundproof", ("soundproof",), "banlist", _FMT),
    Ban("item", "Quick Claw", ("quickclaw",), "banlist", _FMT_M, on_pinned=False),
    Ban("move", "Assist", ("assist",), "banlist", _FMT),
    Ban("move", "Swagger", ("swagger",), "banlist", _FMT),
)
#: The format banlist's per-set COMBOS that ban no single entity: team validation only.
FORMAT_COMBO_BANS: Tuple[Ban, ...] = (
    Ban("combo", "Baton Pass + Block", ("batonpass", "block"), "banlist", _FMT),
    Ban("combo", "Baton Pass + Mean Look", ("batonpass", "meanlook"), "banlist", _FMT),
    Ban("combo", "Baton Pass + Spider Web", ("batonpass", "spiderweb"), "banlist", _FMT),
)

#: OHKO Clause bans every move with the `ohko` flag (`data/rulesets.ts:921-937`, `onValidateSet`); the
#: gen-3-legal ones, each `ohko:` line in pinned `data/moves.ts` (verified by `format_spec_test`).
OHKO_BANS: Tuple[Ban, ...] = tuple(
    Ban("move", name, (to_id(name),), "OHKO Clause", f"pinned:data/moves.ts:{line} (ohko)")
    for name, line in (("Fissure", 5522), ("Guillotine", 8008), ("Horn Drill", 8986), ("Sheer Cold", 16209)))
EVASION_ITEM_BANS: Tuple[Ban, ...] = tuple(
    Ban("item", name, (to_id(name),), "Evasion Items Clause", "pinned:data/rulesets.ts:961 (banlist)")
    for name in ("Bright Powder", "Lax Incense"))
#: `Acupressure` is on the clause's banlist too (gen 4+, no gen-3 dex entry) — kept so the drift check sees
#: the whole list; it can never match a gen-3 set.
EVASION_MOVE_BANS: Tuple[Ban, ...] = tuple(
    Ban("move", name, (to_id(name),), "Evasion Moves Clause", "pinned:data/rulesets.ts:970 (banlist)")
    for name in ("Acupressure", "Minimize", "Double Team"))

#: Species with NO legal ability left: every gen-3 ability they can have is banned (Sand Veil / Soundproof),
#: and gen3ou's `Obtainable Abilities` requires one (`sim/team-validator.ts:738-750`: "needs to have an
#: ability" / the banned-ability problem at :2018). Showdown does not name them, the validator rejects every
#: set of theirs; the spec names them so the species prior can be 0. Derived from the gen-3 dex (abilities of
#: gen <= 3 only: Mr. Mime's Filter is gen 4) and re-derived by `format_spec_test`.
ABILITY_LOCKED_SPECIES: Tuple[Tuple[str, str], ...] = (
    ("sandshrew", "sandveil"), ("sandslash", "sandveil"), ("cacnea", "sandveil"), ("cacturne", "sandveil"),
    ("whismur", "soundproof"), ("loudred", "soundproof"), ("exploud", "soundproof"), ("mrmime", "soundproof"),
)
ABILITY_LOCKED_BANS: Tuple[Ban, ...] = tuple(
    Ban("species", sid, (sid,), "Obtainable Abilities",
        f"derived: every gen-3 ability of {sid} is banned ({ab}); sim/team-validator.ts:738-750, :2018")
    for sid, ab in ABILITY_LOCKED_SPECIES)

# --------------------------------------------------------------------------------------------------
# The team-building clauses' own lists (verbatim from the clause bodies; the drift check compares them).
# --------------------------------------------------------------------------------------------------
#: One Boost Passer Clause's `boostingEffects` — MASTER's list (`data/rulesets.ts:1164-1168` @ master), which
#: added `recycle` (line 1167) since the pin; the pinned list lacks it (:data:`PINNED_DIFFERENCES`).
ONE_BOOST_PASSER_EFFECTS: FrozenSet[str] = frozenset({
    "acidarmor", "agility", "amnesia", "apicotberry", "barrier", "bellydrum", "bulkup", "calmmind",
    "cosmicpower", "curse", "defensecurl", "dragondance", "ganlonberry", "growth", "harden", "howl",
    "irondefense", "liechiberry", "meditate", "petayaberry", "recycle", "salacberry", "sharpen", "speedboost",
    "starfberry", "swordsdance", "tailglow", "withdraw"})
#: Speed Pass Clause's `boostingEffects` (`data/rulesets.ts:1239-1241`, identical on master).
SPEED_PASS_EFFECTS: FrozenSet[str] = frozenset({
    "agility", "dragondance", "ancientpower", "silverwind", "salacberry", "speedboost", "starfberry"})
#: Accuracy Trap Clause's `trapping` list (`data/rulesets.ts:991-993`, identical on master).
ACCURACY_TRAP_TRAPPING: FrozenSet[str] = frozenset({
    "arenatrap", "magnetpull", "shadowtag", "block", "meanlook", "spiderweb", "anchorshot", "jawlock",
    "octolock", "spiritshackle", "thousandwaves"})
#: Accuracy Trap Clause's `accuracy` list as computed for gen 3 (`data/rulesets.ts:994-997`: a move whose own
#: `boosts.accuracy < 0`, or a 100 %-chance secondary lowering accuracy) — re-derived from pinned
#: `data/moves.ts` by `format_spec_test`.
ACCURACY_TRAP_ACCURACY_MOVES: FrozenSet[str] = frozenset({
    "flash", "kinesis", "mudslap", "sandattack", "smokescreen"})


# --------------------------------------------------------------------------------------------------
# The resolved rule table — every rule gen3ou's ruleset reaches, ONE story each.
# --------------------------------------------------------------------------------------------------
_R = "pinned:data/rulesets.ts"
RULES: Tuple[Rule, ...] = (
    # --- the format entry itself (pinned config/formats.ts:4418-4422; master :4695-4700) ---
    Rule("Banlist entities", "format", Story.PRIOR_ZERO,
         "A banned species / item / move / ability (or a move banned for one species) can never be on a legal "
         "team, so the model's prior gives it nothing.",
         "pinned:config/formats.ts:4421; master:config/formats.ts:4699", FORMAT_BANS),
    Rule("Banlist combos", "format", Story.TEAM_BUILDING,
         "Baton Pass with a trapping move on ONE set: each part is legal alone, so no prior changes; the "
         "validator rejects the set and the model sees only legal teams. (The belief's per-move marginals can "
         "still put a little mass on the illegal pair: a stated limitation, design_format_spec.md §8.)",
         "pinned:config/formats.ts:4421; master:config/formats.ts:4699", FORMAT_COMBO_BANS),
    Rule("Standard", "format", Story.RULESET,
         "A ruleset (its parts are listed below, each with its own story).", "pinned:data/mods/gen3/rulesets.ts:2-10"),
    Rule("One Boost Passer Clause", "format", Story.TEAM_BUILDING,
         "Limits Baton Passers with a boost; a team property the validator enforces. The model sees a legal "
         "team and needs nothing more.", f"{_R}:1156-1193 (master adds `recycle`, :1167)"),
    Rule("Accuracy Trap Clause", "format", Story.TEAM_BUILDING,
         "A per-set combination (trapper + a guaranteed accuracy drop); validator-enforced, nothing for the "
         "model.", f"{_R}:986-1004"),
    Rule("Freeze Clause Mod", "format", Story.BOARD_STATE,
         "In-battle: no foe is frozen while one of its side is frozen. The engine enforces it; each mon's "
         "status is in the observation; the op / move-resolution rules ask the spec whether it is in force.",
         f"{_R}:1451-1471"),
    Rule("Speed Pass Clause", "format", Story.TEAM_BUILDING,
         "No Baton Passer with a Speed boost; validator-enforced, nothing for the model.", f"{_R}:1232-1262"),
    # --- Standard (data/mods/gen3/rulesets.ts:2-10) ---
    Rule("Standard AG", "Standard", Story.RULESET,
         "A ruleset (gen 3 inherits gen 4's: no Team Preview).", "pinned:data/mods/gen4/rulesets.ts:2-7"),
    Rule("Sleep Clause Mod", "Standard", Story.BOARD_STATE,
         "In-battle: no foe is put to sleep while one of its side is asleep from a non-Rest source. The engine "
         "enforces it; the observation carries each mon's sleep and the Rest flag (and the board token's "
         "`sleep_clause_used`); the op / move-resolution rules ask the spec whether it is in force.",
         f"{_R}:1378-1402"),
    Rule("Switch Priority Clause Mod", "Standard", Story.ENGINE,
         "The faster mon switches first on a double switch: turn order the engine applies; nothing to model "
         "beyond the speed facts the observation already carries.", f"{_R}:1424-1431"),
    Rule("Species Clause", "Standard", Story.TEAM_BUILDING,
         "One of each species per team; validator-enforced. Its CONSEQUENCE (a revealed species is not also "
         "hidden) is already a rule of the hidden-team belief (`belief_tables.SPECIES_CLAUSE_LOGIT`, the op's "
         "Species-Clause-filtered prior), so the model needs nothing more.", f"{_R}:788-805"),
    Rule("Nickname Clause", "Standard", Story.COSMETIC, "Nicknames never reach a battle decision.",
         f"{_R}:806-825"),
    Rule("OHKO Clause", "Standard", Story.PRIOR_ZERO, "Reduces to banned moves: prior 0.", f"{_R}:921-938",
         OHKO_BANS),
    Rule("Evasion Items Clause", "Standard", Story.PRIOR_ZERO, "Reduces to banned items: prior 0.",
         f"{_R}:957-965", EVASION_ITEM_BANS),
    Rule("Evasion Moves Clause", "Standard", Story.PRIOR_ZERO, "Reduces to banned moves: prior 0.",
         f"{_R}:966-974", EVASION_MOVE_BANS),
    # --- Standard AG, gen-4 mod (data/mods/gen4/rulesets.ts:2-7) ---
    Rule("Obtainable", "Standard AG", Story.PRIOR_ZERO,
         "Learnset / ability / forme legality. The move prior already gives an unlearnable move the ILLEGAL "
         "value (`gen3_data.learnset`); the one consequence for the format spec is the ability-locked species "
         "(every gen-3 ability banned), whose species prior is 0.", f"{_R}:166-218", ABILITY_LOCKED_BANS),
    Rule("HP Percentage Mod", "Standard AG", Story.ENGINE,
         "The opponent's HP is shown as a percentage; the observation encodes what the protocol shows.",
         f"{_R}:1352-1369"),
    Rule("Cancel Mod", "Standard AG", Story.COSMETIC, "A UI affordance (undo a choice before the turn runs).",
         f"{_R}:1370-1377"),
    Rule("Beat Up Nicknames Mod", "Standard AG", Story.ENGINE,
         "Beat Up never announces the ally; `observation/gen3_effect_sources.RULE_GATED_LINES` relies on it.",
         f"{_R}:826-840"),
    Rule("Endless Battle Clause", "Standard AG", Story.ENGINE,
         "The engine ends a provably endless battle; our own stall forfeit (`agents.training.stall`) ends a "
         "game far earlier.", f"{_R}:1060-1075"),
)

#: Where the pinned engine's format differs from the ladder's (master): ``(comparison key, entry) → why``,
#: each a gap ``main.format_drift check --pinned`` must find EXACTLY (a pin bump that closes one fails there
#: until the entry is deleted).
PINNED_DIFFERENCES: Dict[Tuple[str, str], str] = {
    ("format_banlist", "Quick Claw"):
        "Quick Claw is on master's gen3ou banlist (config/formats.ts:4699) and absent from the pinned one "
        "(:4421): a pool team holding it PLAYS in training and is ILLEGAL on the ladder.",
    ("oneboostpasserclause.boostingEffects", "recycle"):
        "master's One Boost Passer Clause counts `recycle` as a boosting effect (data/rulesets.ts:1167); the "
        "pinned list lacks it.",
}


@dataclass(frozen=True)
class FormatSpec:
    """The resolved spec of ONE format: its rules (:data:`RULES`) and the lookups derived from them."""
    format_id: str
    rules: Tuple[Rule, ...]
    rule_ids: FrozenSet[str] = field(init=False)
    bans: Tuple[Ban, ...] = field(init=False)

    def __post_init__(self) -> None:
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"format spec {self.format_id}: a rule is declared twice")
        object.__setattr__(self, "rule_ids", frozenset(ids))
        object.__setattr__(self, "bans", tuple(b for r in self.rules for b in r.bans))

    def has_rule(self, rule_id: str) -> bool:
        return rule_id in self.rule_ids

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(f"{self.format_id} has no rule {rule_id!r}")

    def _banned(self, kind: str) -> FrozenSet[str]:
        return frozenset(b.ids[0] for b in self.bans if b.kind == kind)

    @property
    def banned_species(self) -> FrozenSet[str]:
        return self._banned("species")

    @property
    def banned_items(self) -> FrozenSet[str]:
        return self._banned("item")

    @property
    def banned_moves(self) -> FrozenSet[str]:
        return self._banned("move")

    @property
    def banned_abilities(self) -> FrozenSet[str]:
        return self._banned("ability")

    @property
    def combo_bans(self) -> Tuple[Ban, ...]:
        """Every per-set combination, ``species_move`` included (team validation checks them all)."""
        return tuple(b for b in self.bans if b.kind in ("combo", "species_move"))

    def species_move_bans(self) -> Dict[str, FrozenSet[str]]:
        """``{species: {move}}`` — the moves banned FOR one species (Smeargle + Ingrain)."""
        out: Dict[str, set] = {}
        for b in self.bans:
            if b.kind == "species_move":
                out.setdefault(b.ids[0], set()).add(b.ids[1])
        return {k: frozenset(v) for k, v in out.items()}

    def is_banned(self, kind: str, entity_id: str, species: Optional[str] = None) -> bool:
        """``entity_id`` (of ``kind``) is banned outright — or, with ``species`` and ``kind == "move"``,
        banned for that species (a ``species_move`` combo)."""
        if entity_id in self._banned(kind):
            return True
        return kind == "move" and species is not None and entity_id in self.species_move_bans().get(species, ())


GEN3OU = FormatSpec(FORMAT_ID, RULES)
_ACTIVE: FormatSpec = GEN3OU


def active() -> FormatSpec:
    """The spec in force — gen3ou, the only format the project plays. Read at CALL time, so a test can swap
    ``_ACTIVE`` for a planted spec and see every reader move."""
    return _ACTIVE


def sleep_clause_mod() -> bool:
    """Sleep Clause Mod is in force (the op / move-resolution sleep gates read this, never a constant)."""
    return active().has_rule("sleepclausemod")


def freeze_clause_mod() -> bool:
    """Freeze Clause Mod is in force (the op's incoming freeze gate reads this, never a constant)."""
    return active().has_rule("freezeclausemod")


def without_rule(spec: FormatSpec, rule_id: str) -> FormatSpec:
    """``spec`` minus one rule — the planted-change helper the reader tests use."""
    return FormatSpec(spec.format_id, tuple(r for r in spec.rules if r.id != rule_id))


def stories() -> Dict[str, Story]:
    """``{rule id: story}`` of the active spec (one story per rule, by construction)."""
    return {r.id: r.story for r in active().rules}


def iter_bans(kinds: Iterable[str] = ("species", "item", "move", "ability", "species_move", "combo")) -> Iterable[Ban]:
    ks = set(kinds)
    return (b for b in active().bans if b.kind in ks)
