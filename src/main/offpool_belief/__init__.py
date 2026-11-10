"""THE OFF-POOL BELIEF METER — ``python -m main.offpool_belief`` (MF-3; generalisation review R4).

**The question it answers.** On a team the network never trained against, is its belief about the opponent's
hidden team better or worse than the Smogon prior it starts from? The three learned belief legs (the hidden species,
the unrevealed moves of a revealed mon, a revealed mon's unknown item) are each ``Smogon prior ⊕ learned delta``, so
the meter reads every leg twice on the SAME decision rows: the checkpoint's own posterior, and the prior alone (the
deltas zeroed, through ``agents.model.belief_prior_mix`` at λ = 1, the tested D0f path).

**The rows are a frozen BANK** (:mod:`.bank`): one fixed pilot (by default the ``pstar_plateau`` baseline) plays
greedy against itself on the Rust env core; seat 1 pilots each team of a REGISTERED panel
(``main.heldout_panels``), seat 2 a seeded draw of the in-pool ``home72``. The bank keeps seat 2's decision rows,
the seat FACING the panel team, with the core's privileged labels (the true hidden species, the revealed mons'
true moves and items). Every checkpoint is scored on the same rows, so two checkpoints differ only by their
weights.

**The numbers** (:mod:`.metrics`), each for the posterior, the prior and any ``--mix λ`` blend:

* ``species_logp`` — per decision, the mean log presence of the TRUE hidden species (0 is perfect; presences sum to
  the hidden count); ``species_hit`` — the share of the true hidden species among the top-k presences;
* ``moves_logp`` / ``moves_prec`` — per revealed mon with unrevealed true moves, the mean log probability of those
  moves, and precision at their count (the Hidden Power family excluded: its type is its own head);
* ``items_logp`` / ``items_top1`` — per revealed mon whose item the observation does not yet state.

**Team-clustered** (:mod:`.stats`): a row belongs to the panel team its seat faces; a panel's number is the mean
over teams of each team's mean, and every interval is a seeded bootstrap over TEAMS (a row is not a sample).

**What it refuses** (:mod:`.panels`): a SEALED panel, an unregistered or moved panel (the registry's content
digest), a bank whose rows or panel no longer match what its manifest recorded, an ``--out`` under ``models/``.
Only a panel the registry marks ``held_out_from: training`` is reported as OFF-POOL; an in-pool panel (``home72``)
is the FAMILIAR reference column and is labelled so. A bank holds a held-out panel's teams in the clear (labels and
observations): it is written outside the repository, is stamped ``held_out_material``, and is never a training
input.

Rule: ``designs/tools/offpool_belief.md``. The frozen probe this replaces as a tool:
``designs/research_state/measurements/arch_generalisation_2026-10-26/`` (P5).
"""
from __future__ import annotations

BANK_SCHEMA = "gen3_offpool_belief_bank_v1"
SCORE_SCHEMA = "gen3_offpool_belief_score_v1"

#: The in-pool panel every bank game's OTHER seat draws from (and the familiar reference panel when scored).
HOME_PANEL = "home72"
#: The three unfamiliar panels of the standing read, in report order: H (ladder replays), F (forum), P72 (procedural).
OFFPOOL_PANELS = ("heldout72_hpfix", "forum72", "p72")
#: The pilot of the standing bank, a NAMED baseline (``designs/baselines.json``).
DEFAULT_PILOT = "pstar_plateau"


class OffPoolBeliefError(RuntimeError):
    """The meter cannot run as asked (a panel it may not read, a bank that moved, an output it may not write)."""
