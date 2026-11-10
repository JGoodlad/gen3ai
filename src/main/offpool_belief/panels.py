"""Which panels the meter may read, and what a number from one may be called.

Every panel comes from the registry (``main.heldout_panels``) BY NAME: its teams are digest-checked on every load, a
SEALED panel raises before a team is read, and a diagnostic or unregistered panel has no entry. ``OFF-POOL`` is a
property the registry states (``held_out_from`` names ``training``), never one this module infers from a name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from main.offpool_belief import OffPoolBeliefError


@dataclass(frozen=True)
class PanelRef:
    """One readable panel: its registry name, content digest, whether it is held out from training, and its teams
    as ``(label, paste)`` in the registry's order (the order the bank's team index means)."""

    name: str
    content_digest: str
    held_out: bool
    teams: Tuple[Tuple[str, str], ...]

    @property
    def kind(self) -> str:
        return "OFF-POOL" if self.held_out else "IN-POOL reference"


def panel_ref(name: str) -> PanelRef:
    """The panel ``name`` as the meter may read it; a sealed, unregistered, retired or moved panel is refused."""
    from main import heldout_panels as HP

    try:
        HP.check_unsealed(name)
        entry = HP.panel_entry(name)
        teams = HP.panel_team_texts(name)
    except (HP.PanelRegistryError, KeyError) as e:
        raise OffPoolBeliefError(f"the off-pool belief meter may not read panel {name!r}: {e}") from e
    if str(entry.get("role", "")).startswith("RETIRED"):
        raise OffPoolBeliefError(f"panel {name!r} is RETIRED as a read panel ({entry['role']})")
    return PanelRef(name=name, content_digest=str(entry["content_digest"]),
                    held_out=HP.HELD_OUT_FROM_TRAINING in entry.get("held_out_from", ()), teams=tuple(teams))


def registry_sha256() -> str:
    from main import heldout_panels as HP

    return HP.registry_sha256()


def refuse_under_models(path: Path, what: str) -> None:
    """``models/`` is read-only (standing rule 6): the meter writes its bank and its scores elsewhere."""
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is None:
        return
    p, root = Path(path).resolve(), Path(md).resolve()
    if p == root or root in p.parents:
        raise OffPoolBeliefError(f"{what} {p} is under {root}: models/ is read-only, name a path outside it")


def refuse_inside_repo(path: Path, what: str) -> None:
    """A bank is held-out material (decision rows of games against a held-out panel, with the true sets as labels);
    it never lands in a checkout or a worktree, where a commit, a corpus builder or a grep would find it."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            raise OffPoolBeliefError(f"{what} {p} is inside the checkout {d}: a bank is held-out material, name a "
                                     "path outside every checkout (the archive)")
