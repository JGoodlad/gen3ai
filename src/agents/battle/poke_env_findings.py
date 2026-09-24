"""The KNOWN poke-env READING findings — where poke-env's view of the battle is WRONG about a sim fact.

``gen3_poke_env_findings_v1`` (the Rust Core Program's M2, owner directive 2026-09-23). The Rust
core's ``present()`` (``src/rust_sim/src/present/``) implements the TRUE reading; where poke-env —
and so the ``LiveView`` training builds today — disagrees, and the authoritative source (the Rust
board, the pinned Showdown source, the request JSON) says poke-env is the side that is wrong, the
disagreement is a FINDING, registered here. **Never a rule that reproduces the mistake, and never
a blanket tolerance**: each entry names ONE field and a VALUE-AWARE predicate that must hold for a
difference to count as that finding; any other difference in the same field is still a divergence.

Every comparison of the core's reading against poke-env's (slice V's core column,
``rust_core_present_test.py``, ``one_sided_view_parity_fuzz_test.py``'s core road) routes a field
difference through :func:`explain`. **When the poke-env fix lands (a TRAINING-INPUT change —
``designs/CHANGELOG.md``, the owner's call), delete the entry: the check tightens by itself.**

Each entry records the field, a minimal reproduction, what poke-env reads and the truth (with how
the truth is established), whether it reaches the TRAINING INPUT, and its measured rate. The rates
and the procedure live in ``designs/rust_sim/present.md`` §3 and
``designs/research_state/measurements/rust_core_m2_2026-09-23/``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Finding:
    id: str
    #: The ``LivePokemon`` field the finding lives in.
    field: str
    title: str
    poke_env_reads: str
    truth: str
    #: How the truth was established (the authoritative source).
    source: str
    #: A minimal protocol excerpt that reproduces it (one side's text, after a two-mon-a-side start).
    reproduce: str
    #: Whether a difference reaches the 2501-dim observation, and where.
    reaches_obs: str
    #: ``fnmatch`` patterns over the obs block names a difference may touch
    #: (``one_sided_view_parity_fuzz_test._block_of``: ``context``, ``opp_team[3].status_counters``).
    obs_blocks: Tuple[str, ...]
    #: ``(reading LivePokemon, core LivePokemon) -> bool`` — True iff THIS finding explains the
    #: difference in :attr:`field` exactly (value-aware).
    predicate: Callable[[Any, Any], bool]


def _pe_v10(reading: Any, core: Any) -> bool:
    return bool(core.fainted and reading.fainted and not dict(core.boosts) and dict(reading.boosts))


def _pe_r1b(reading: Any, core: Any) -> bool:
    # poke-env ticks at `|turn|`, the sim at the residual chip: exactly ONE apart — ahead for a mon
    # that entered after the residual, behind between the residual and the next `|turn|` (an
    # end-of-turn forced replacement's decision). While the mon is active and badly poisoned, and
    # after it FAINTS, where poke-env freezes the count it had and the toxic slot no longer reads it.
    one_apart = abs(reading.status_counter - core.status_counter) == 1
    return one_apart and ((reading.status == core.status == "tox" and bool(core.active))
                          or (reading.status == core.status == "fnt"))


def _pe_v16(reading: Any, core: Any) -> bool:
    r, c = dict(reading.volatiles), dict(core.volatiles)
    return "flashfire" in c and "flashfire" not in r and {k: v for k, v in c.items() if k != "flashfire"} == r


FINDINGS: Dict[str, Finding] = {f.id: f for f in (
    Finding(
        id="PE-V10", field="boosts",
        title="a fainted mon keeps its stat stages",
        poke_env_reads="the stages it fainted with, until it is switched out (`Pokemon.faint` "
                       "does not clear boosts; `switch_out` does)",
        truth="none — the sim's faint `clearVolatile` zeroes `boosts`",
        source="the Rust board (`MonState::boosts` zeroed at the faint, the stages kept only in "
               "the observation-only `faint_boosts`) and `sim/pokemon.ts` clearVolatile",
        reproduce="|-boost|p2a: Zapdos|spa|1  ·  |faint|p2a: Zapdos",
        reaches_obs="YES — the fainted ACTIVE mon's stages are encoded by `active_context` at the "
                    "forced-switch decision (e.g. +1 SpA is byte 4 of that block)",
        obs_blocks=("context",),
        predicate=_pe_v10),
    Finding(
        id="PE-R1b", field="status_counter",
        title="the toxic count ticks at `|turn|`, the sim's stage at the residual chip",
        poke_env_reads="+1 per `|turn|` while active: ONE AHEAD of the sim's stage for a mon that "
                       "switched in after the residual (a post-faint replacement), ONE BEHIND at a "
                       "decision taken between the residual and the next `|turn|` (an end-of-turn "
                       "forced replacement)",
        truth="the sim's toxic STAGE: the residual `[from] psn` chips since its switch-in "
              "(`tox.onSwitchIn` resets the stage, `onResidual` ramps it before chipping)",
        source="the Rust board (`Status::Toxic(stage)`) and the stream's residual chips",
        reproduce="|-status|p2a: Zapdos|tox · |-damage|p2a: Zapdos|94/100 tox|[from] psn · |turn|2 · "
                  "|switch|p2a: Snorlax|… · |switch|p2a: Zapdos|Zapdos|94/100 tox · |turn|3  "
                  "(poke-env 1, the sim 0); or …|turn|2 · |-damage|p2a: Zapdos|82/100 tox|[from] psn · "
                  "a forced-switch |request| (poke-env 1, the sim 2)",
        reaches_obs="YES while it is active — the per-mon `status_counters` toxic slot (min(ctr, 8) / 8); "
                    "the count poke-env freezes at a faint does not (the slot reads it only under `tox`)",
        obs_blocks=("our_team[[]*[]].status_counters", "opp_team[[]*[]].status_counters"),
        predicate=_pe_r1b),
    Finding(
        id="PE-V16", field="volatiles",
        title="Flash Fire is ended by its holder's own Fire move",
        poke_env_reads="`Pokemon.moved` silently ends FLASH_FIRE when the holder uses a damaging "
                       "Fire move, whatever the gen",
        truth="the `flashfire` volatile lasts until the holder leaves the field (the ability's "
              "`onEnd` / `clearVolatile`) and boosts every Fire move ×1.5",
        source="the Rust board (`MonState::flash_fire`, probe-verified vs the resolved gen-3 dist, "
               "`harness/probe_flashfire_rng.js`) and `data/abilities.ts` flashfire",
        reproduce="|-start|p2a: Houndoom|ability: Flash Fire  ·  |move|p2a: Houndoom|Flamethrower|p1a: …",
        reaches_obs="YES — the `flashfire` binary volatile slot of `active_context` (byte 25)",
        obs_blocks=("context",),
        predicate=_pe_v16),
)}


def explain(field: str, reading: Any, core: Any) -> Optional[str]:
    """The id of the registered finding that explains the difference in ``field`` between the
    reading's ``LivePokemon`` and the core's, or ``None`` (a divergence)."""
    for f in FINDINGS.values():
        if f.field == field and f.predicate(reading, core):
            return f.id
    return None


def obs_block_explained(block: str, fired: Iterable[str]) -> bool:
    """Whether an obs ``block`` that differs between the core's successor and poke-env's is one a
    finding that FIRED on this successor (``fired`` — ids from :func:`explain`) may touch."""
    return any(fnmatch.fnmatchcase(block, pat) for fid in fired for pat in FINDINGS[fid].obs_blocks)
