"""THE REGISTERED-PANEL TEAM SOURCE of the head-to-head (B-D7, ``design_diversity_teams_opponents.md`` §5.6, §6.2).

``main.h2h play | play-many --panel <name>``: BOTH sides of every mirrored pair draw their teams from ONE frozen,
REGISTERED team panel (``designs/research_state/heldout_panels.json``, reader :mod:`main.heldout_panels`) instead of
the player's eval team builder (the default pool with its 10 % sample-team bias). It is GEN-M's engine and D1's
co-primary P instrument (each seed final against X35's frozen reference population, mirrored pairs on ``home72``).

* **Refused unless registered.** An unknown name is an :class:`PanelRefused` naming the registered ones; a panel whose
  files on disk no longer match the registry (count, ``gen3_team_set_digest_v1`` content digest) is refused too
  (:func:`main.heldout_panels.panel_team_texts` checks both), and so is a panel the gen3ou validator does not pack
  WHOLE (``Gen3Teambuilder`` silently SKIPS an invalid paste, which would make the read a different panel). A SEALED
  panel (:data:`main.heldout_panels.SEALED`, e.g. ``forum216b`` until X54) is refused by name.
* **The draw.** A plain ``Gen3Teambuilder`` over the panel's teams, no bias: each side's team is a uniform draw from
  the panel, re-seeded per (game key, side) by the eval core (``rust_eval.seeds.draw_team``), so a run reproduces
  game for game, exactly as the pool path does; the mirror (game ``2k + 1`` hands the two teams over) is the
  executor's, unchanged. The opponent (a greedy SENTINEL) draws from the trainee's builder, which is this one.
* **The regime.** ``team_set`` = ``ts:`` + 16 hex over ``{kind: registered_panel, panel, content_digest, registry
  schema, bias 0}`` (:func:`panel_team_set_id`) -- the panel's CONTENT is identity, so its rows never pool with the
  pool's or another panel's (the readers key on ``regime_id``, and the ledger's seed block includes it); the label
  ``team_source`` names the panel and its full content digest (:func:`panel_team_source`). The game PROTOCOL is
  unchanged (``gen3_eval_protocol_v1_h2h``): the team set is a regime field, as the team-strength meter's scheduled
  builder already uses it (``H2HEngine(team_builder=...)``).
* **Byte-identical without it.** ``panel=None`` builds today's regime and engine, unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.training import eval_ledger as L
from main.h2h.errors import H2HError
from utils.teambuilder import Gen3Teambuilder

#: The ``team_set_id`` parts' kind (bump the tag if the draw rule ever changes).
KIND = "registered_panel"
DRAW_RULE = "gen3_h2h_panel_draw_v1: uniform Gen3Teambuilder draw per (game key, side), no bias"


class PanelRefused(H2HError):
    """A panel that is not registered, whose files differ from the registry, or that does not pack whole."""


def panel_texts(name: str) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """``(registry entry, [(label, text)])`` of a REGISTERED panel whose count and content digest hold on disk."""
    from main import heldout_panels as HP

    try:
        ent = HP.panel_entry(name)
        HP.check_unsealed(name)
    except KeyError as exc:
        raise PanelRefused(f"--panel {name!r}: {exc.args[0]}") from None
    except HP.PanelSealedError as exc:
        raise PanelRefused(f"--panel {name!r}: {exc}") from None
    try:
        items = list(HP.panel_team_texts(name))
    except HP.PanelRegistryError as exc:
        raise PanelRefused(f"--panel {name!r}: the files on disk are not the registered panel: {exc}") from None
    return dict(ent), items


def panel_team_set_id(name: str) -> str:
    """The regime's ``team_set`` of panel ``name`` (module docs); checks the panel on disk first."""
    from main import heldout_panels as HP

    ent, _items = panel_texts(name)
    return L.team_set_id({"builder": "Gen3Teambuilder", "kind": KIND, "panel": name,
                          "content_digest": ent["content_digest"], "n": int(ent["n"]), "registry_schema": HP.SCHEMA,
                          "draw": DRAW_RULE})


def panel_team_source(name: str) -> str:
    """The regime's ``team_source`` LABEL: the panel, its full content digest and the draw (not identity)."""
    from main import heldout_panels as HP

    ent = HP.panel_entry(name)
    return (f"registered panel {name} ({HP.SCHEMA}; {ent['n']} teams, gen3_team_set_digest_v1 "
            f"{ent['content_digest']}): both sides draw from the panel ({DRAW_RULE})")


def panel_regime(turn_limit: int, name: str, mode: str = "off") -> Dict[str, Any]:
    """The v2 regime of a panel read: :func:`main.h2h.play.builder_regime` on the panel's team set and label -- the
    very function :class:`main.h2h.play.H2HEngine` stamps a :class:`PanelTeams` builder with, so plan and engine
    agree by construction (and the engine's ``regime_id`` is re-checked against the plan's before any batch)."""
    from types import SimpleNamespace

    from main.h2h.play import builder_regime

    return builder_regime(int(turn_limit), SimpleNamespace(team_set_id=panel_team_set_id(name),
                                                           team_source=panel_team_source(name)), mode)


class PanelTeams(Gen3Teambuilder):
    """A ``Gen3Teambuilder`` over a registered panel, with the ``team_set_id`` / ``team_source`` the engine stamps
    (module docs). REFUSES a panel the validator does not pack whole."""

    _block_episodes = 1

    def __init__(self, name: str):
        ent, items = panel_texts(name)
        super().__init__([t for _, t in items])
        if len(self.packed_teams) != len(items):
            raise PanelRefused(f"--panel {name!r}: {len(items) - len(self.packed_teams)} of {len(items)} teams fail the "
                               "gen3ou validator -- a panel that does not pack whole is not the registered panel")
        self.panel = name
        self.content_digest = str(ent["content_digest"])
        self.labels = [lab for lab, _ in items]
        self.team_set_id = panel_team_set_id(name)
        self.team_source = panel_team_source(name)


def check_panel(name: Optional[str]) -> Optional[str]:
    """``None`` passes (the pool path); a name must be a registered panel that holds on disk. Returns the name."""
    if name is not None:
        panel_texts(name)
    return name
