"""THE HELD-OUT PANEL REGISTRY (B-D3, ``gen3_heldout_panels_v1``) and its near-neighbour rule.

Design: ``designs/endstate/design_diversity_teams_opponents.md`` §5.6 ("Freeze, then widen"). The registry file is
``designs/research_state/heldout_panels.json``; it was committed BEFORE any pool-widening corpus existed, and every
widening corpus (``main.team_corpora``) and the ``--team-mix`` startup guard read it.

**What a panel is.** A frozen, committed directory of team pastes, identified by CONTENT
(``main.anchors.session.team_set_digest``, ``gen3_team_set_digest_v1``: names and order never enter it), with the rule
that drew it, its seed, its manifest's sha256, and ``held_out_from``: the list of things that must never train on it.
``home72`` and ``away`` are IN-POOL reference panels (``held_out_from == []``: our pool already holds them); the others
are held out from every training team source.

**The near-neighbour rule** (``NEAR_NEIGHBOUR_MIN_SHARED = 4``, matcher ``gen3_team_overlap_v1``): a team is a near
neighbour of a panel iff it shares at least 4 of its 6 MON SETS (species + sorted moves, Hidden Power typed) with ONE
team of the panel. An exact copy shares 6. A widening corpus excludes every registered panel's teams AND their near
neighbours, the in-pool panels included (so a corpus never re-weights the sample ground); the training guard refuses a
team source that hits a HELD-OUT panel (:func:`contamination`).

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
NEAR_NEIGHBOUR_MIN_SHARED = 4
MATCHER = "gen3_team_overlap_v1"
#: ``held_out_from`` tokens. ``training`` = no training team source (pool, corpus, pin) may hold the panel's teams or
#: their near neighbours.
HELD_OUT_FROM_TRAINING = "training"


class PanelRegistryError(RuntimeError):
    """The registry does not match the committed panels (a digest, a count or a manifest changed)."""


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
    panels = load_registry()["panels"]
    if name not in panels:
        raise KeyError(f"{name!r} is not a registered panel ({', '.join(panels)}); register it in {REGISTRY_REL}")
    return panels[name]


def panel_dir(name: str) -> Path:
    return repo_path(panel_entry(name)["team_dir"])


@functools.lru_cache(maxsize=None)
def panel_team_texts(name: str) -> Tuple[Tuple[str, str], ...]:
    """``(label, text)`` for every team of ``name``, after checking its count and content digest."""
    from main.anchors.session import load_team_texts, team_set_digest

    ent = panel_entry(name)
    items = load_team_texts(panel_dir(name))
    if len(items) != int(ent["n"]):
        raise PanelRegistryError(f"panel {name}: {len(items)} team files, registered {ent['n']}")
    got = team_set_digest([t for _, t in items])
    if got != ent["content_digest"]:
        raise PanelRegistryError(f"panel {name}: content digest {got} != registered {ent['content_digest']}")
    return tuple(items)


def verify() -> Dict[str, int]:
    """Every panel's count, content digest and manifest sha256; returns ``{panel: n}`` or raises."""
    out = {}
    for name, ent in load_registry()["panels"].items():
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
