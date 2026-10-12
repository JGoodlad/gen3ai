"""THE INFORMATION-LEVER SPEC: one declarative description of the EXTRA information a consumer is given about the
opponent's hidden team, beyond what a real player would have. ONE taxonomy for every consumer:

* the TRAINING observation (``--oracle-reveal <spec>``: the Rust encoder writes the told facts into the opponent
  block, ``pokesim::encoder::oracle``);
* the SEARCH world builder (``main.value_search.worlds``: a world's hidden facts are the truth's, a wrong plausible
  draw, or the honest draw);
* the MODEL-INPUT injection (``agents.info_levers.inject``: the network's belief inputs pushed toward the target).

Design and how to read ceilings with it: ``designs/endstate/design_information_levers.md``.

THE TAXONOMY (one :class:`Lever` per (class, part)):

=========  ===============================================================  =================
class      the hidden fact                                                   parts
=========  ===============================================================  =================
species    the identity of a never-seen mon (a seen mon's species is public)  unseen
moves      the move slots play has not revealed                               seen, unseen
item       the held item, until play names it                                 seen, unseen
ability    the ability, until play names it (two-ability species)             seen, unseen
spread     nature, EVs, IVs (→ the exact stats and the Hidden Power type)     seen, unseen
hp         the exact HP inside the displayed 1 % bucket                       seen
counters   hidden durations: sleep / confusion / Encore / Disable / lock turns seen
=========  ===============================================================  =================

``seen`` = the opponent mons the viewer has seen (switched / dragged in); ``unseen`` = the ones it has not. An
``unseen`` lever applies where the consumer's mon at that slot IS a true opponent mon (its species is right: told by a
``species`` lever, or a hypothesis / world draw that happens to be right). A never-seen mon is at full HP with no
counters, so ``hp`` / ``counters`` have no ``unseen`` part.

Each lever has a MASS p ∈ (0, 1] and a SOURCE:

* ``truth``: the consumer's belief about the fact becomes p·δ(true value) + (1 − p)·(its honest belief) — the
  owner's "high mass to the known facts, but not 100 %" (the dose-response family; p = 1 is the oracle);
* ``placebo``: the same with the TRUE value replaced by an equally confident WRONG but plausible one (drawn from the
  honest prior, excluding the truth and everything play revealed): does the content help, or any confidence?
* ``map``: the same with the target the honest belief's OWN mode (its top choice): an HONEST sharpening — no truth is
  read — that separates "fewer, sharper worlds / beliefs" from "the right ones".

Each consumer REALISES the mixture its natural way and declares what it cannot honour (its support table refuses
it by name, never silently): the search builder DRAWS each world's fact from the mixture; the injection WRITES the
mixed marginals; the observation can only state a fact or not, so it TELLS the target with probability p per
(episode, side, mon, class).

TEXT. A spec is a preset name (:data:`PRESETS`) or a comma-separated list of clauses ``class[:scope][@mass][~source]``
(defaults: the class's widest scope, mass 1, source truth). The CANONICAL text (:attr:`InfoReveal.canonical`) is
what the Rust core parses and what :attr:`InfoReveal.hash` is taken over: clauses in class order, each with its scope
explicit (a class whose seen and unseen parts are identical is written ``both``), a mass below 1 as ``@`` + up to
four decimals (masses are whole multiples of 1/10,000), a non-truth source as ``~source``; the empty spec is
``off``. The LABEL (:attr:`InfoReveal.label`) is ``off`` / ``species`` / ``full`` when the spec equals that X32
level (``--oracle-reveal``'s three original values, recorded in ``model_config.json`` exactly as before), else the
canonical text.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

CLASSES: Tuple[str, ...] = ("species", "moves", "item", "ability", "spread", "hp", "counters")
PARTS: Tuple[str, ...] = ("seen", "unseen")
SCOPES: Tuple[str, ...] = ("seen", "unseen", "both")
SOURCES: Tuple[str, ...] = ("truth", "placebo", "map")

#: The parts a class can carry: a seen mon's species is public; a never-seen mon is at full HP with no counters.
CLASS_PARTS: Mapping[str, Tuple[str, ...]] = {
    "species": ("unseen",), "moves": PARTS, "item": PARTS, "ability": PARTS, "spread": PARTS,
    "hp": ("seen",), "counters": ("seen",),
}
#: The SET classes (a fact of the packed set): what a ``full`` reveal states.
SET_CLASSES: Tuple[str, ...] = ("moves", "item", "ability", "spread")

#: A mass is an integer number of 1/10,000ths: exact canonical text, the same integer in Rust.
MASS_DENOM = 10_000

#: The X32 ``--oracle-reveal`` levels, whose LABEL stays the level name (recorded in ``model_config.json``).
LEVEL_PRESETS: Tuple[str, ...] = ("off", "species", "full")


class InfoRevealError(ValueError):
    """A spec text that does not parse, or a lever a consumer cannot honour (raised with the consumer's name)."""


@dataclass(frozen=True, order=True)
class Lever:
    """One (class, part) of the opponent's hidden facts, told at ``mass_bp`` / 10,000 from ``source``."""

    cls: str
    part: str
    mass_bp: int = MASS_DENOM
    source: str = "truth"

    def __post_init__(self) -> None:
        if self.cls not in CLASSES:
            raise InfoRevealError(f"info lever class {self.cls!r} is not one of {CLASSES}")
        if self.part not in CLASS_PARTS[self.cls]:
            raise InfoRevealError(f"info lever {self.cls}: part {self.part!r} is not one of {CLASS_PARTS[self.cls]} "
                                  "(a seen mon's species is public; a never-seen mon is at full HP with no counters)")
        if not isinstance(self.mass_bp, int) or not 0 < self.mass_bp <= MASS_DENOM:
            raise InfoRevealError(f"info lever {self.cls}:{self.part}: mass {self.mass_bp}/{MASS_DENOM} is not in "
                                  "(0, 1] (a mass-0 lever is no lever: leave it out)")
        if self.source not in SOURCES:
            raise InfoRevealError(f"info lever {self.cls}:{self.part}: source {self.source!r} is not one of {SOURCES}")

    @property
    def mass(self) -> float:
        return self.mass_bp / MASS_DENOM

    def key(self) -> Tuple[int, int]:
        return CLASSES.index(self.cls), PARTS.index(self.part)


def _mass_text(bp: int) -> str:
    if bp == MASS_DENOM:
        return ""
    return "@0." + f"{bp:04d}".rstrip("0")


def _parse_mass(text: str, clause: str) -> int:
    m = re.fullmatch(r"(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)", text)
    if not m:
        raise InfoRevealError(f"info lever {clause!r}: mass {text!r} is not a decimal in [0, 1]")
    frac = text.split(".", 1)[1] if "." in text else ""
    whole = text.split(".", 1)[0] or "0"
    if len(frac.rstrip("0")) > 4:
        raise InfoRevealError(f"info lever {clause!r}: mass {text!r} has more than four decimals (masses are whole "
                              f"multiples of 1/{MASS_DENOM})")
    return int(whole) * MASS_DENOM + int((frac + "0000")[:4])


@dataclass(frozen=True)
class InfoReveal:
    """A set of :class:`Lever` s, at most one per (class, part), kept in canonical order. Immutable and hashable."""

    levers: Tuple[Lever, ...] = ()

    def __post_init__(self) -> None:
        seen = set()
        for lv in self.levers:
            if (lv.cls, lv.part) in seen:
                raise InfoRevealError(f"info levers: {lv.cls}:{lv.part} is given twice")
            seen.add((lv.cls, lv.part))
        object.__setattr__(self, "levers", tuple(sorted(self.levers, key=Lever.key)))

    # ---------------------------------------------------------------- queries
    def lever(self, cls: str, part: str) -> Optional[Lever]:
        """The lever on (``cls``, ``part``), or None (that fact is the consumer's HONEST belief)."""
        if cls not in CLASSES or part not in PARTS:
            raise InfoRevealError(f"info levers: no such fact {cls}:{part}")
        for lv in self.levers:
            if lv.cls == cls and lv.part == part:
                return lv
        return None

    @property
    def is_off(self) -> bool:
        return not self.levers

    def classes(self) -> Tuple[str, ...]:
        return tuple(c for c in CLASSES if any(lv.cls == c for lv in self.levers))

    def sources(self) -> Tuple[str, ...]:
        return tuple(s for s in SOURCES if any(lv.source == s for lv in self.levers))

    # ---------------------------------------------------------------- algebra (presets are built with these)
    def plus(self, other: "InfoReveal") -> "InfoReveal":
        """Every lever of both; a (class, part) both name must agree."""
        mine = {(lv.cls, lv.part): lv for lv in self.levers}
        for lv in other.levers:
            if (lv.cls, lv.part) in mine and mine[(lv.cls, lv.part)] != lv:
                raise InfoRevealError(f"info levers: {lv.cls}:{lv.part} is given two different ways")
            mine[(lv.cls, lv.part)] = lv
        return InfoReveal(tuple(mine.values()))

    def minus(self, other: "InfoReveal") -> "InfoReveal":
        """These levers less every (class, part) ``other`` names."""
        drop = {(lv.cls, lv.part) for lv in other.levers}
        return InfoReveal(tuple(lv for lv in self.levers if (lv.cls, lv.part) not in drop))

    # ---------------------------------------------------------------- text, hash, record
    @property
    def canonical(self) -> str:
        if not self.levers:
            return "off"
        out: List[str] = []
        for cls in CLASSES:
            mine = {lv.part: lv for lv in self.levers if lv.cls == cls}
            if not mine:
                continue
            s, u = mine.get("seen"), mine.get("unseen")
            if s is not None and u is not None and (s.mass_bp, s.source) == (u.mass_bp, u.source):
                parts = [("both", s)]
            else:
                parts = [(p, lv) for p, lv in (("seen", s), ("unseen", u)) if lv is not None]
            for scope, lv in parts:
                out.append(f"{cls}:{scope}{_mass_text(lv.mass_bp)}" + ("" if lv.source == "truth" else f"~{lv.source}"))
        return ",".join(out)

    @property
    def label(self) -> str:
        """The X32 level name when the spec IS that level, else the canonical text (module docstring)."""
        for name in LEVEL_PRESETS:
            if PRESETS[name] == self:
                return name
        return self.canonical

    @property
    def hash(self) -> str:
        return hashlib.sha256(("info_reveal_v1:" + self.canonical).encode()).hexdigest()[:16]

    def record(self) -> Dict[str, object]:
        """The block a run / a measurement records: label, canonical text, hash and the levers."""
        return {"label": self.label, "canonical": self.canonical, "hash": self.hash,
                "levers": [{"class": lv.cls, "part": lv.part, "mass": lv.mass, "source": lv.source}
                           for lv in self.levers]}

    def __str__(self) -> str:
        return self.label


def _clauses(text: str) -> List[Lever]:
    out: List[Lever] = []
    for raw in text.split(","):
        clause = raw.strip()
        if not clause:
            raise InfoRevealError(f"info levers {text!r}: an empty clause")
        m = re.fullmatch(r"([a-z]+)(?::([a-z]+))?(?:@([0-9.]+))?(?:~([a-z]+))?", clause)
        if not m:
            raise InfoRevealError(f"info lever {clause!r} is not `class[:scope][@mass][~source]` "
                                  f"(classes {CLASSES}, scopes {SCOPES}, sources {SOURCES})")
        cls, scope, mass, source = m.group(1), m.group(2), m.group(3), m.group(4) or "truth"
        if cls not in CLASSES:
            raise InfoRevealError(f"info lever {clause!r}: class {cls!r} is not one of {CLASSES}")
        domain = CLASS_PARTS[cls]
        if scope is None:
            parts = list(domain)
        elif scope == "both":
            parts = list(domain)                       # `both` on a one-part class is that part
        elif scope in PARTS:
            parts = [scope]
        else:
            raise InfoRevealError(f"info lever {clause!r}: scope {scope!r} is not one of {SCOPES}")
        bp = MASS_DENOM if mass is None else _parse_mass(mass, clause)
        if bp == 0:
            continue                                   # a mass-0 lever is the honest belief: no lever
        out.extend(Lever(cls, p, bp, source) for p in parts)
    return out


def parse(text: "str | InfoReveal") -> InfoReveal:
    """A preset name (:data:`PRESETS`, :func:`preset`) or a clause list → :class:`InfoReveal` (typed refusal on
    anything else). An ``InfoReveal`` passes through."""
    if isinstance(text, InfoReveal):
        return text
    if not isinstance(text, str):
        raise InfoRevealError(f"info levers: {text!r} is not a spec text")
    t = text.strip()
    if t in PRESETS:
        return PRESETS[t]
    p = _param_preset(t)
    if p is not None:
        return p
    return InfoReveal(tuple(_clauses(t)))


def canonical(text: "str | InfoReveal") -> str:
    return parse(text).canonical


def label(text: "str | InfoReveal") -> str:
    return parse(text).label


def same(a: "str | InfoReveal", b: "str | InfoReveal") -> bool:
    """Do two spec texts mean the same levers (``full`` and its clause list do)?"""
    return parse(a) == parse(b)


# ------------------------------------------------------------------------------------------------
# PRESETS: the X32 levels, the oracle, the pilot's builder, and the hidden-info study's arms
# (`design_hidden_info_binding.md` §6.2 / §7.2 / §8.3; the mapping table is in design_information_levers.md)
# ------------------------------------------------------------------------------------------------

def _spec(text: str) -> InfoReveal:
    return InfoReveal(tuple(_clauses(text))) if text else InfoReveal()


_ALL = "species:unseen,moves:both,item:both,ability:both,spread:both,hp:seen,counters:seen"
#: The study's five-class partition of the hidden facts (§6.1): M, U, I, A, S. S is a seen mon's spread AND its exact
#: HP (the spread determines the exact HP the replayed history leaves); `counters` belongs to no class (the study's
#: H-COUNTERS: the pre-root dice) and is told by O alone.
STUDY_CLASSES: Mapping[str, InfoReveal] = {
    "M": _spec("moves:seen"),
    "U": _spec("species:unseen,moves:unseen,item:unseen,ability:unseen,spread:unseen"),
    "I": _spec("item:seen"),
    "A": _spec("ability:seen"),
    "S": _spec("spread:seen,hp:seen"),
}

PRESETS: Dict[str, InfoReveal] = {
    # the X32 levels (`--oracle-reveal`'s original values; the observation must stay byte-identical to them)
    "off": _spec(""),
    "species": _spec("species:unseen"),
    "full": _spec("species:unseen,moves:both,item:both,ability:both,spread:both"),
    # everything: the information ceiling (search: the TRUE battle)
    "oracle": _spec(_ALL),
    # the value-discrimination pilot's builder (v1 / v2): a seen mon's item, ability, spread TRUE (hence its exact
    # HP) and the pre-root dice TRUE (hence its hidden counters); moves of seen mons and never-seen mons drawn
    "pilot": _spec("item:seen,ability:seen,spread:seen,hp:seen,counters:seen"),
    # the study's base and its one-class sufficiency / necessity arms (§6.2)
    "F": _spec(""),
    "O": _spec(_ALL),
    "F+Usp": _spec("species:unseen"),
    # the move dose, the wrong-moves placebo, the honest sharpening (§6.4)
    "D25": _spec("moves:seen@0.25"), "D50": _spec("moves:seen@0.5"), "D75": _spec("moves:seen@0.75"),
    "D90": _spec("moves:seen@0.9"),
    "W50": _spec("moves:seen@0.5~placebo"), "W90": _spec("moves:seen@0.9~placebo"), "W100": _spec("moves:seen~placebo"),
    "MAP90": _spec("moves:seen@0.9~map"), "MAP100": _spec("moves:seen~map"),
    # the knock-in forks' levers (§8.3, mass 0.95), here on the OBSERVATION channel (design_information_levers.md §6)
    "ki_moves": _spec("moves:seen@0.95"), "ki_items": _spec("item:seen@0.95"),
    "ki_spreads": _spec("spread:seen@0.95"), "ki_wrongmoves": _spec("moves:seen@0.95~placebo"),
}
for _c, _x in STUDY_CLASSES.items():
    PRESETS[f"F+{_c}"] = PRESETS["F"].plus(_x)
    PRESETS[f"O-{_c}"] = PRESETS["O"].minus(_x)

#: The model-input axis's parametrised arms (§7.2): FAMILY@p, p a mass.
_PARAM_FAMILIES: Mapping[str, str] = {
    "TRUE": "moves:seen@{p}", "PLACEBO": "moves:seen@{p}~placebo", "SELFMAP": "moves:seen@{p}~map",
    "TRUEHYP": "moves:both@{p}",
}


def _param_preset(t: str) -> Optional[InfoReveal]:
    m = re.fullmatch(r"([A-Z]+)@([0-9.]+)", t)
    if not m or m.group(1) not in _PARAM_FAMILIES:
        return None
    return InfoReveal(tuple(_clauses(_PARAM_FAMILIES[m.group(1)].format(p=m.group(2)))))


def preset(name: str) -> InfoReveal:
    """A named preset (:data:`PRESETS`, or ``TRUE@p`` / ``PLACEBO@p`` / ``SELFMAP@p`` / ``TRUEHYP@p``)."""
    if name in PRESETS:
        return PRESETS[name]
    p = _param_preset(name)
    if p is None:
        raise InfoRevealError(f"info levers: no preset {name!r} (presets: {sorted(PRESETS)} and "
                              f"{sorted(_PARAM_FAMILIES)}@p)")
    return p


def preset_table() -> List[Dict[str, str]]:
    """Every named preset with its canonical text and hash (the design doc's table is checked against this)."""
    return [{"name": k, "canonical": v.canonical, "hash": v.hash} for k, v in PRESETS.items()]


# ------------------------------------------------------------------------------------------------
# CONSUMER SUPPORT: what each consumer honours. A lever outside it is REFUSED by name (never silently dropped).
# ------------------------------------------------------------------------------------------------

#: per consumer, (class, part) → the sources it honours. A (class, part) absent from the table cannot be told.
Support = Mapping[Tuple[str, str], Tuple[str, ...]]


@dataclass(frozen=True)
class Consumer:
    """A consumer's declared support: ``tells`` = (class, part) → sources it honours; ``residual`` = (class, part)
    facts it can only ever give TRUE (a named leak it cannot remove, reported with every honest spec it runs);
    ``needs`` = (class, part) → the (class, part) lever without which that one can never apply (refused, so a
    spec cannot be a silent no-op)."""

    name: str
    tells: Support
    residual: Tuple[Tuple[str, str], ...] = ()
    needs: Mapping[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    why: Mapping[Tuple[str, str], str] = field(default_factory=dict)

    def validate(self, spec: "str | InfoReveal") -> InfoReveal:
        """``spec`` parsed and checked against this consumer: :class:`InfoRevealError` names the first lever it
        cannot honour and why."""
        s = parse(spec)
        for lv in s.levers:
            k = (lv.cls, lv.part)
            if k not in self.tells:
                why = self.why.get(k, "this consumer cannot state that fact")
                raise InfoRevealError(f"{self.name}: cannot honour the lever {lv.cls}:{lv.part} ({why})")
            if lv.source not in self.tells[k]:
                raise InfoRevealError(f"{self.name}: the lever {lv.cls}:{lv.part}~{lv.source} needs a source it does "
                                      f"not support (it honours {self.tells[k]} there)")
            need = self.needs.get(k)
            if need is not None and s.lever(*need) is None:
                raise InfoRevealError(f"{self.name}: the lever {lv.cls}:{lv.part} can never apply without "
                                      f"{need[0]}:{need[1]} (no slot of the consumer holds a never-seen mon otherwise)")
        return s

    def residual_of(self, spec: "str | InfoReveal") -> Tuple[Tuple[str, str], ...]:
        """The facts this consumer gives TRUE although ``spec`` does not tell them (its named residual leaks)."""
        s = parse(spec)
        return tuple(k for k in self.residual if s.lever(*k) is None)


def facts() -> Iterable[Tuple[str, str]]:
    """Every (class, part) of the taxonomy, in canonical order."""
    for c in CLASSES:
        for p in CLASS_PARTS[c]:
            yield c, p


def dumps(spec: "str | InfoReveal") -> str:
    return json.dumps(parse(spec).record(), sort_keys=True)
