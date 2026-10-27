"""THE HELD-OUT PANEL REGISTRY (B-D3, ``gen3_heldout_panels_v1``) and its near-neighbour rule.

Design: ``designs/endstate/design_diversity_teams_opponents.md`` §5.6 ("Freeze, then widen"). The registry file is
``designs/research_state/heldout_panels.json``; it was committed BEFORE any pool-widening corpus existed, and every
widening corpus (``main.team_corpora``) and the ``--team-mix`` startup guard read it.

**What a panel is.** A frozen, committed directory of team pastes, identified by CONTENT
(``main.anchors.session.team_set_digest``, ``gen3_team_set_digest_v1``: names and order never enter it), with the rule
that drew it, its seed, its manifest's sha256, and ``held_out_from``: the list of things that must never train on it.
A SEALED panel (:data:`SEALED`) is registered and excluded like any held-out panel but played by no meter until its
read (``forum216b``: X54). ``home72`` and ``away`` are IN-POOL reference panels (``held_out_from == []``: our pool already holds them); the others
are held out from every training team source. A DIAGNOSTIC panel (``rc72``, the H-CTX read's recombined pool sets)
lives in its own registry, ``designs/research_state/diagnostic_panels.json`` (:func:`load_diagnostic`): played and
verified by name like any panel, but outside the exclusion rule below (:func:`panel_names` never names it).

**The near-neighbour rule** (``NEAR_NEIGHBOUR_MIN_SHARED = 4``, matcher ``gen3_team_overlap_v1``): a team is a near
neighbour of a panel iff it shares at least 4 of its 6 MON SETS (species + sorted moves, Hidden Power typed) with ONE
team of the panel. An exact copy shares 6. A widening corpus excludes every registered panel's teams AND their near
neighbours, the in-pool panels included (so a corpus never re-weights the sample ground); the training guard refuses a
team source that hits a HELD-OUT panel (:func:`contamination`).

**A panel holds DISTINCT teams** (``gen3_panel_duplicate_guard_v1``): :func:`panel_team_texts` raises
:class:`PanelDuplicateError` for one team under two labels (byte-identical, or the same once packed) unless
:data:`KNOWN_DUPLICATES` names that group with a reason. ``home72`` (X37b finding 1) and ``away`` are named: both are
frozen, so their duplicates are excused, not removed.

Why by content and with a radius: ``away`` was described as "never trained on" for a month while 19 of its 20 teams
were pool copies (F-DV-5), and ``hl_05_26`` holds repeat teams from the same players, so an exact-only exclusion leaks.
"""
from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path

SCHEMA = "gen3_heldout_panels_v1"
REGISTRY_REL = "designs/research_state/heldout_panels.json"
#: The DIAGNOSTIC panels' own registry (:func:`load_diagnostic`): played by name, outside the exclusion rule.
DIAGNOSTIC_REL = "designs/research_state/diagnostic_panels.json"
DIAGNOSTIC_SCHEMA = "gen3_diagnostic_panels_v1"
NEAR_NEIGHBOUR_MIN_SHARED = 4
MATCHER = "gen3_team_overlap_v1"
#: ``held_out_from`` tokens. ``training`` = no training team source (pool, corpus, pin) may hold the panel's teams or
#: their near neighbours.
HELD_OUT_FROM_TRAINING = "training"


class PanelRegistryError(RuntimeError):
    """The registry does not match the committed panels (a digest, a count or a manifest changed)."""


class PanelDuplicateError(PanelRegistryError):
    """A panel holds one team under two labels (byte-identical, or the same team once packed) that no
    :data:`KNOWN_DUPLICATES` entry names. A read that resamples the panel's FILES double-weights that team, and the
    scheduled builder refuses one team under two identities (X37b, F-DV-25)."""


#: ``gen3_panel_duplicate_guard_v1``. A panel is a set of DISTINCT teams; :func:`panel_team_texts` refuses one that is
#: not, unless the duplicate is NAMED here with the reason it is tolerated. Teams are the same iff they pack to the same
#: string (``Teambuilder.join_team`` over the parsed paste -- the criterion X37b's ``canonical()`` used, a byte-identical
#: paste included). An entry matches a duplicate GROUP exactly (the set of its labels): a third copy, or a different pair,
#: is a new duplicate and refused. A stale entry (its group no longer duplicates) is refused as well, so the list cannot
#: outlive the defect it excuses. The panels' files, the manifests and the registered digests are FROZEN (every X35 / X37
#: readout names them); this list exists so those users keep working, not to bless a duplicate in a NEW panel.
KNOWN_DUPLICATES: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "home72": ({
        "labels": ("63eda9d8d491d6a4.txt", "dbbfac7bd5.txt"),
        "reason": "X37b finding 1 (F-DV-25): one team twice, 71 distinct teams in 72 files; the panel stays frozen so "
                  "X35's readouts stay comparable, and X37b plays the duplicate as the sorted-first id "
                  "(`canonical()` in its scripts/x37b.py)",
    },),
    "away": ({
        "labels": ("team_13.gen3ou_team", "team_9.gen3ou_team"),
        "reason": "found by this guard 2026-10-19, the same class as X37b finding 1 (F-DV-25): two BYTE-IDENTICAL "
                  "files, 19 distinct teams in 20 files; the panel is Metamon's `competitive` copy and stays frozen",
    },),
}


def duplicate_groups(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, ...]]:
    """The groups of labels in ``items`` (``(label, paste)`` pairs) that are ONE team (same packed string), each group
    sorted, the groups sorted. Pure: no registry, no Node (the pack is :class:`utils.team_packing.Teambuilder`'s)."""
    from utils.team_packing import Teambuilder

    by_pack: Dict[str, List[str]] = {}
    for label, text in items:
        by_pack.setdefault(Teambuilder.join_team(Teambuilder.parse_showdown_team(text)), []).append(label)
    return sorted(tuple(sorted(v)) for v in by_pack.values() if len(v) > 1)


def check_distinct(name: str, items: Sequence[Tuple[str, str]],
                   allowed: Optional[Dict[str, Tuple[Dict[str, Any], ...]]] = None) -> None:
    """Raise :class:`PanelDuplicateError` for a duplicate group of panel ``name`` that ``allowed`` (default
    :data:`KNOWN_DUPLICATES`) does not name, or for an allow entry that no longer matches a duplicate group."""
    allowed = KNOWN_DUPLICATES if allowed is None else allowed
    entries = allowed.get(name, ())
    for e in entries:
        if not str(e.get("reason", "")).strip():
            raise PanelDuplicateError(f"panel {name}: the allow entry for {tuple(e['labels'])} carries no reason")
    named = {tuple(sorted(e["labels"])) for e in entries}
    found = set(duplicate_groups(items))
    new = sorted(found - named)
    if new:
        raise PanelDuplicateError(
            f"panel {name}: {len(new)} team(s) appear under two or more labels (one team once packed): "
            + "; ".join(" = ".join(g) for g in new)
            + ". A resample of the panel's files double-weights them (X37b F-DV-25). Fix the panel, or -- only for a "
              "frozen panel that is already in use -- name the group in KNOWN_DUPLICATES with the reason.")
    stale = sorted(named - found)
    if stale:
        raise PanelDuplicateError(f"panel {name}: KNOWN_DUPLICATES names {stale}, which is no longer a duplicate group "
                                  "of the panel -- delete the entry")


class PanelSealedError(PanelRegistryError):
    """A SEALED panel (:data:`SEALED`) was asked for by a surface that PLAYS or READS it."""


#: SEALED panels (2026-10-20, beat-Metamon FINDING 2): REGISTERED, so every widening corpus and the ``--team-mix``
#: training guard exclude them and their near neighbours, but NEVER READ -- no meter plays them -- until the named read
#: is registered. Unsealing is a commit that deletes the entry and cites that read's registration. Only the surfaces
#: that play a panel by name refuse (:func:`check_unsealed`; ``main.h2h.panel``): :func:`panel_team_texts` stays open,
#: because the exclusion rule needs the content. A read that names the team DIRECTORY instead (``main.anchors``
#: configs) is outside this guard; the panel is installed nowhere such a config could reach.
SEALED: Dict[str, str] = {
    "forum216b": "X54, the Metamon goal's confirmation read (design_beat_metamon_path.md §6.6): carved from w_forum_v1's "
                 "tier AB and registered BEFORE X35's D1 arm launched, so the goal has a panel no lever was chosen on",
}


def check_unsealed(name: str) -> None:
    """Raise :class:`PanelSealedError` if ``name`` is SEALED (:data:`SEALED`)."""
    if name in SEALED:
        raise PanelSealedError(f"panel {name} is SEALED until {SEALED[name]}; it is registered so that training "
                               "excludes it, and no meter may play it before that read is registered")


@dataclass(frozen=True)
class Hit:
    """One team of a candidate source that collides with a registered panel."""

    label: str              # the candidate team's label
    panel: str              # the panel it hits
    shared: int             # mon sets shared with the panel's nearest team (6 = exact)
    nearest: Optional[str]  # that panel team's label


def registry_path() -> Path:
    return repo_path(REGISTRY_REL)


@functools.lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    reg = json.loads(registry_path().read_text())
    if reg.get("schema") != SCHEMA:
        raise PanelRegistryError(f"{REGISTRY_REL}: schema {reg.get('schema')!r} != {SCHEMA!r}")
    if int(reg["near_neighbour"]["min_shared_mon_sets"]) != NEAR_NEIGHBOUR_MIN_SHARED:
        raise PanelRegistryError("the registry's near-neighbour radius differs from the code's")
    return reg


def panel_names(*, held_out_only: bool = False) -> List[str]:
    panels = load_registry()["panels"]
    return [n for n, p in panels.items()
            if not held_out_only or HELD_OUT_FROM_TRAINING in p.get("held_out_from", [])]


def panel_entry(name: str) -> Dict[str, Any]:
    """The registry entry of ``name``: a held-out / in-pool panel, else a DIAGNOSTIC one (:func:`load_diagnostic`)."""
    panels = load_registry()["panels"]
    if name in panels:
        return panels[name]
    diag = load_diagnostic()["panels"]
    if name in diag:
        return diag[name]
    raise KeyError(f"{name!r} is not a registered panel ({', '.join(list(panels) + list(diag))}); register it in "
                   f"{REGISTRY_REL} (or, for a diagnostic panel, {DIAGNOSTIC_REL})")


@functools.lru_cache(maxsize=1)
def load_diagnostic() -> Dict[str, Any]:
    """THE DIAGNOSTIC PANEL REGISTRY (``gen3_diagnostic_panels_v1``, :data:`DIAGNOSTIC_REL`): panels a meter plays by
    name (``main.x35.kakuna``'s cells, ``main.h2h.panel``) that are NOT part of the exclusion rule. A diagnostic panel
    is neither held out nor a sample ground: :func:`panel_names` (the list every widening corpus and the training
    guard read) never names it, so registering one moves no corpus, no guard and no ``registry_sha256``. Its teams
    and their count / content digest / manifest are verified exactly like a registered panel's (:func:`verify`). A
    name in both registries is refused."""
    reg = json.loads(repo_path(DIAGNOSTIC_REL).read_text())
    if reg.get("schema") != DIAGNOSTIC_SCHEMA:
        raise PanelRegistryError(f"{DIAGNOSTIC_REL}: schema {reg.get('schema')!r} != {DIAGNOSTIC_SCHEMA!r}")
    both = sorted(set(reg["panels"]) & set(load_registry()["panels"]))
    if both:
        raise PanelRegistryError(f"{DIAGNOSTIC_REL}: {both} are also in {REGISTRY_REL}; a panel has one registry")
    return reg


def diagnostic_panel_names() -> List[str]:
    """The DIAGNOSTIC panels (:func:`load_diagnostic`); never part of :func:`panel_names`."""
    return list(load_diagnostic()["panels"])


def panel_dir(name: str) -> Path:
    return repo_path(panel_entry(name)["team_dir"])


@functools.lru_cache(maxsize=None)
def panel_team_texts(name: str) -> Tuple[Tuple[str, str], ...]:
    """``(label, text)`` for every team of ``name``, after checking its count and content digest."""
    from main.anchors.session import load_team_texts, team_set_digest

    ent = panel_entry(name)
    # A SEALED panel is read only for the exclusion rule (never played) and holds one team the label path cannot
    # encode (`sfv2_0810`, Level 99; pinned by a test); unsealing removes the entry, so the guard then applies to it.
    items = load_team_texts(panel_dir(name), label_guard=name not in SEALED)
    if len(items) != int(ent["n"]):
        raise PanelRegistryError(f"panel {name}: {len(items)} team files, registered {ent['n']}")
    got = team_set_digest([t for _, t in items])
    if got != ent["content_digest"]:
        raise PanelRegistryError(f"panel {name}: content digest {got} != registered {ent['content_digest']}")
    check_distinct(name, items)
    return tuple(items)


def verify() -> Dict[str, int]:
    """Every panel's count, content digest and manifest sha256, the DIAGNOSTIC ones included; returns ``{panel: n}`` or
    raises."""
    out = {}
    for name, ent in list(load_registry()["panels"].items()) + list(load_diagnostic()["panels"].items()):
        out[name] = len(panel_team_texts(name))
        man = ent.get("manifest")
        if man:
            got = hashlib.sha256(repo_path(man["path"]).read_bytes()).hexdigest()
            if got != man["sha256"]:
                raise PanelRegistryError(f"panel {name}: manifest {man['path']} sha256 {got} != {man['sha256']}")
    return out


@functools.lru_cache(maxsize=None)
def _reference(name: str):
    from main.anchors.team_overlap import Reference, parse_items

    return Reference(parse_items(panel_team_texts(name)))


def nearest(text: str, panels: Optional[Sequence[str]] = None) -> List[Tuple[str, int, Optional[str]]]:
    """``(panel, shared mon sets with its nearest team, that team's label)`` for every panel (default: all)."""
    from main.anchors.team_overlap import parse_team

    team = parse_team(text)
    return [(p, *_reference(p).max_shared(team)) for p in (panels or panel_names())]


def hits(items: Sequence[Tuple[str, str]], panels: Optional[Sequence[str]] = None,
         min_shared: int = NEAR_NEIGHBOUR_MIN_SHARED) -> List[Hit]:
    """Every ``(label, text)`` of ``items`` that is an exact copy or a near neighbour (≥ ``min_shared`` shared mon
    sets with ONE team) of a panel in ``panels`` (default: every registered panel)."""
    out = []
    for label, text in items:
        for panel, shared, near in nearest(text, panels):
            if shared >= min_shared:
                out.append(Hit(label=label, panel=panel, shared=int(shared), nearest=near))
    return out


def contamination(items: Sequence[Tuple[str, str]]) -> List[Hit]:
    """The TRAINING GUARD's question: which teams of a training source hit a panel held out from training?"""
    return hits(items, panel_names(held_out_only=True))


def registry_sha256() -> str:
    """The registry file's sha256 (stamped on every corpus manifest and every read that relies on it)."""
    return hashlib.sha256(registry_path().read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Verify the held-out panel registry (counts, digests, manifests).")
    ap.add_argument("cmd", choices=("verify",))
    ap.parse_args(argv)
    for name, n in verify().items():
        ent = panel_entry(name)
        print(f"{name:18s} n={n:3d} digest={ent['content_digest'][:12]}… held_out_from={ent['held_out_from']}")
    print(f"registry sha256 {registry_sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
