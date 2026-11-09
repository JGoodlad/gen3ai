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

``r0s4_rolls`` is ``r0s4`` with ONE lever changed, the world builder's dice: ``dice="rolls"`` (F1 tier A,
``main.value_search.worlds`` THE DICE), so a world may reproduce each printed damage number on ANY consistent roll
rather than the truth's own. Its worlds stay uniformly weighted (``SearchConfig(weights="uniform")``, as every live
preset). It is the S-U gate's leak-reduced re-gate (``measurements/su_regate_rolls_2026-10-28/``; the gate review's M1 / M3).

``r0s4_honest`` is ``r0s4_rolls`` with its worlds built by the LEAK-HONEST builder
(``main.value_search.honest_worlds``, ``designs/endstate/design_sprime_honest.md`` D1-D9): ``dice="fresh"``. The search
is a function of the searched side's information set I alone — no battle seed, no opponent team, no opponent command
enters — its worlds replayed on fresh dice matched segment by segment (``pokesim::fresh_dice``), every draw keyed by I.
Same worlds count, leaf, rule and uniform weights as ``r0s4_rolls``; ``max_tries`` is unused (the builder's budgets are
its own). ``r0s4`` and ``r0s4_rolls`` are unchanged.

``r0s4_honest_inc`` is ``r0s4_honest`` with its worlds CARRIED through a battle's searched decisions
(``main.value_search.honest_inc``, ``design_sprime_honest.md`` §9): ``dice="fresh_inc"``. Each world's trajectory is
replayed and only the new segments matched; the oldest two of eight retire each decision and the shortfall comes from
a warm chain. Still a function of I alone (the carry at a decision is a function of I's earlier decisions, each a
prefix of it). Its first searched decision of a battle is ``r0s4_honest``'s, byte for byte.

THE ENVELOPE PRESETS (the cost read's DRAFT, ``measurements/searched_cost_2026-11-09/DRAFT_PREREG.md`` §3 / §7 R1;
``designs/endstate/design_searched_player.md`` §3), each ``r0s4_honest_inc`` with ONE cost lever plus the SAFETY NET
(``net_reserve_s``: the bank-aware deadline of :mod:`main.reads.timer_model`, applied by the live searcher to the
PICK only, never inside the search, so an offline re-score of a decision is the same with or without it):

* ``r0s4_honest_inc_k4`` (K4): the world count, ``n_worlds`` 4 and ``per_action`` 4 (a world serves ⌈4 / 2⌉ = 2
  decisions; an exclusion is < 4 worlds);
* ``r0s4_honest_inc_cap`` (CAP): the builder's COUNT budget profile ``cap`` (``honest_inc.BUDGETS``: stage 1's base
  budget wherever ``r0s4_honest_inc`` builds at B3 — the cold first build, a battle's re-cold build, the warm chain
  with no survivor — and the shortfall escalation at 256 tries instead of 1,024).

The new fields default to today's values (``budget`` None = ``honest_inc.DEFAULT_BUDGET``, ``net_reserve_s`` None = no
net) and :meth:`SearchPreset.as_dict` leaves a default field out, so the four earlier presets' dicts, stamps and
records are byte-identical (``search_presets_test``).

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
    its ``max_tries`` per world. ``dice="fresh"`` selects the leak-honest builder (``r0s4_honest``), ``"fresh_inc"`` its
    incremental form (``r0s4_honest_inc``). ``budget``: the incremental builder's named COUNT budget profile
    (``honest_inc.BUDGETS``; None = today's); ``net_reserve_s``: the live safety net's reserve (None = no net)."""

    name: str
    n_worlds: int
    per_action: int
    depth: Optional[int]
    rule: str
    pikl_lambda: float
    reveal: str
    dice: str
    max_tries: int
    budget: Optional[str] = None
    net_reserve_s: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """The registered dict; a field of :data:`ENVELOPE_FIELDS` at its default None is left out, so a preset that
        sets none of them has the dict it had before they existed."""
        d = asdict(self)
        for k in ENVELOPE_FIELDS:
            if d[k] is None:
                del d[k]
        return d


#: the fields the envelope presets added (left out of :meth:`SearchPreset.as_dict` at their default None)
ENVELOPE_FIELDS: Tuple[str, ...] = ("budget", "net_reserve_s")
#: the budget profiles a preset may name (``main.value_search.honest_inc.BUDGETS``' keys beside the default, pinned
#: equal by a test: this module stays light)
BUDGET_NAMES: Tuple[str, ...] = ("cap",)


PRESETS: Dict[str, SearchPreset] = {
    "r0s4": SearchPreset(name="r0s4", n_worlds=8, per_action=8, depth=4, rule="pikl", pikl_lambda=0.015,
                         reveal="off", dice="seed", max_tries=60),
    "r0s4_rolls": SearchPreset(name="r0s4_rolls", n_worlds=8, per_action=8, depth=4, rule="pikl", pikl_lambda=0.015,
                               reveal="off", dice="rolls", max_tries=60),
    "r0s4_honest": SearchPreset(name="r0s4_honest", n_worlds=8, per_action=8, depth=4, rule="pikl",
                                pikl_lambda=0.015, reveal="off", dice="fresh", max_tries=60),
    "r0s4_honest_inc": SearchPreset(name="r0s4_honest_inc", n_worlds=8, per_action=8, depth=4, rule="pikl",
                                    pikl_lambda=0.015, reveal="off", dice="fresh_inc", max_tries=60),
    # the envelope presets (module docstring): one cost lever each against r0s4_honest_inc, and the safety net
    "r0s4_honest_inc_k4": SearchPreset(name="r0s4_honest_inc_k4", n_worlds=4, per_action=4, depth=4, rule="pikl",
                                       pikl_lambda=0.015, reveal="off", dice="fresh_inc", max_tries=60,
                                       net_reserve_s=10.0),
    "r0s4_honest_inc_cap": SearchPreset(name="r0s4_honest_inc_cap", n_worlds=8, per_action=8, depth=4, rule="pikl",
                                        pikl_lambda=0.015, reveal="off", dice="fresh_inc", max_tries=60, budget="cap",
                                        net_reserve_s=10.0),
}


def _check_presets() -> None:
    """A budget profile is the incremental builder's and a known name; a reserve is >= 0."""
    for p in PRESETS.values():
        if p.budget is not None and (p.dice != "fresh_inc" or p.budget not in BUDGET_NAMES):
            raise SearchSpecError(f"preset {p.name}: a budget profile is the incremental builder's, one of "
                                  f"{list(BUDGET_NAMES)}")
        if p.net_reserve_s is not None and not p.net_reserve_s >= 0:
            raise SearchSpecError(f"preset {p.name}: net_reserve_s must be >= 0")


_check_presets()


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
