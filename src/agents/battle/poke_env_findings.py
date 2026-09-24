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

**Today the registry is EMPTY** (``gen3_pe_reading_fixes_v1`` fixed all three M2 findings in the
fork), so every comparison above is exact.
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
    #: (``one_sided_view_parity_fuzz_test._block_of``: ``context[0]``, ``opp_team[3].status_counters``).
    obs_blocks: Tuple[str, ...]
    #: ``(reading LivePokemon, core LivePokemon) -> bool`` — True iff THIS finding explains the
    #: difference in :attr:`field` exactly (value-aware).
    predicate: Callable[[Any, Any], bool]


#: The registry. **EMPTY since ``gen3_pe_reading_fixes_v1``** (2026-09-24, a TRAINING-INPUT change):
#: the three M2 findings — PE-V10 (a fainted mon kept its stat stages until switch-out), PE-R1b (the
#: toxic count ticked at ``|turn|`` instead of at the residual chip) and PE-V16 (Flash Fire ended by
#: its holder's own Fire move) — were FIXED in the fork (``src/poke_env/battle/pokemon.py``,
#: ``abstract_battle.py``, ``battle.py``; pins in ``poke_env/battle/reading_fixes_test.py``), and
#: their entries deleted so every check that routes through :func:`explain` tightened to exact
#: equality. Their record (reproductions, rates) is ``designs/rust_sim/present.md`` §3 and
#: ``designs/research_state/measurements/rust_core_m2_2026-09-23/``. A NEW finding is registered
#: here as a :class:`Finding` with a value-aware predicate — never a blanket tolerance.
FINDINGS: Dict[str, Finding] = {}


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
