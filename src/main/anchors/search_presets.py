"""The REGISTERED configurations of the live searched player (:mod:`main.anchors.search_side`), named so a read's spec
can register one by name (``main.reads`` ``models.<label>.load.search``) and a unit can stamp it into ``our_regime``.

Pure and light (no torch, no engine): ``main.reads.spec`` validates a spec with it.

``r0s4`` is R0 §4's search, which S-U plays unchanged (``designs/endstate/design_search_u_headroom.md`` §0;
``measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md`` §4): L1's anchored one-ply search
(``search_l1_l5_2026-10-20``) —
* 8 fully HONEST worlds F: ``WorldSampler(reveal="off", max_tries=60)``, ``dice="seed"`` (the maturity battery's F
  builder, ``hidden_info_maturity_x26_2026-10-18/build_mat_worlds.py``);
* the TRUNC-4-K8 leaf: every legal action played once in each of the 8 worlds (world k on dice seed k), 4 turns deep,
  then the critic V, the checkpoint greedy on BOTH sides (σ̂ = π);
* piKL at L1's fitted λ = 0.015: argmax Q(a) + λ·log π(a).

A searched decision is one of OUR side's decisions at turn >= ``from_turn`` with >= 2 legal actions; every other
decision is the checkpoint's greedy choice, exactly the raw arm's.

The spelling a spec, a CLI flag and a stamp share is ``<preset>@<from_turn>`` (``r0s4@10``); a bare ``<preset>`` means
``@10`` (S-U's turn). The stamp is ``+search=<preset>@<from_turn>`` after ``our_regime``'s other parts.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

#: S-U searches from turn 10 (its §2: the forensics put the U gap there, and the shared nine-turn prefix pairs the arms).
DEFAULT_FROM_TURN = 10
_SPEC = re.compile(r"^([a-z0-9_]+)(?:@(\d{1,3}))?$")


class SearchSpecError(ValueError):
    """A search spelling that names no registered preset, or a turn out of range."""


@dataclass(frozen=True)
class SearchPreset:
    """One registered configuration (module docstring). ``per_action`` playouts per legal action under the uniform
    allocation, each in its own world (``n_worlds`` of them); ``depth`` turns then the critic; the acting ``rule``
    (``pikl`` at ``pikl_lambda``); the world builder's information spec ``reveal`` (``off`` = honest), its ``dice`` and
    its ``max_tries`` per world."""

    name: str
    n_worlds: int
    per_action: int
    depth: Optional[int]
    rule: str
    pikl_lambda: float
    reveal: str
    dice: str
    max_tries: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


PRESETS: Dict[str, SearchPreset] = {
    "r0s4": SearchPreset(name="r0s4", n_worlds=8, per_action=8, depth=4, rule="pikl", pikl_lambda=0.015,
                         reveal="off", dice="seed", max_tries=60),
}


def parse(text: Any) -> Tuple[SearchPreset, int]:
    """``"r0s4@10"`` / ``"r0s4"`` -> ``(preset, from_turn)``; anything else is a :class:`SearchSpecError`."""
    if not isinstance(text, str):
        raise SearchSpecError(f"search {text!r}: a string '<preset>@<from turn>' (e.g. 'r0s4@10')")
    m = _SPEC.match(text.strip())
    if m is None:
        raise SearchSpecError(f"search {text!r}: '<preset>@<from turn>' (e.g. 'r0s4@10')")
    name, turn = m.group(1), m.group(2)
    if name not in PRESETS:
        raise SearchSpecError(f"search preset {name!r} is none of {sorted(PRESETS)}")
    t = DEFAULT_FROM_TURN if turn is None else int(turn)
    if not 1 <= t <= 999:
        raise SearchSpecError(f"search {text!r}: the first searched turn must be in 1..999")
    return PRESETS[name], t


def canonical(text: Any) -> str:
    """The one spelling of a search option: ``<preset>@<from_turn>``."""
    p, t = parse(text)
    return f"{p.name}@{t}"


def stamp(text: Optional[str]) -> str:
    """The suffix a search option adds to ``our_regime`` (``""`` for none)."""
    return "" if text is None else f"+search={canonical(text)}"
