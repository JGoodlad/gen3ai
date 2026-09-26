"""Value-neutrality linchpin: the per-decision observation vector must not change.

Replays the fixed, deterministic battle set from ``golden_obs_capture`` and asserts every
decision's full obs vector is byte-identical to the golden fixture captured on HEAD before the
``gen3_data`` refactor. Because the obs vector is the entire downstream-visible product of the
data layer, a green run proves the refactor (data facade, type-chart/natures sourced from
``data/`` instead of poke-env) changed **no observed value** — i.e. no ``ARCH_SIGNATURE`` bump
is warranted.

Bridge-backed (real in-process battles, no server), so it carries the ``sim`` marker — but NOT
``slow`` (measured 4.2 s), which is exactly why it stays in the routine gate. If an intentional
obs-value change lands (with an ``ARCH_SIGNATURE`` bump), regenerate the fixture:
    python src/agents/training/golden_obs_capture.py --write

**FIXTURE REGENERATED 2026-08-04 for v48** (`gen3_cpu_damage_deleted_v1`, obs 2992 -> 2889).
⚠️ THIS TEST WAS RED ON MAIN FOR THREE DAYS and nobody noticed, because the routine gate is
``-m "not integration and not e2e"`` — which excludes it. The v48 deletion landed in ``2a660e9``
(2026-08-01) without regenerating the fixture, whose previous capture was ``96c0ea0``
(2026-06-26, v42's 3469 -> 2992). If you add an obs-affecting change, run the regen in the SAME
commit, and remember the unit gate will not tell you otherwise.

Regenerating a value-neutrality linchpin bakes in whatever is live, so this regen was NOT done
on the dim mismatch alone — it was PROVEN safe first, and the proof is reproducible: capture the
full vectors at the pre-v48 commit (``e56212b``) and at HEAD, then align them COLUMN-wise (each
column = that dim across all 991 decisions, so a coincidental match is essentially impossible).
Result: HEAD is EXACTLY the pre-v48 capture with 103 columns deleted, bit-for-bit over all 991
decisions, the deleted runs being ``1454..1461`` (the 8 active-move scalars) and ``1473..1567``
(95 = the adjacent 51-dim incoming-damage + 44-dim move-effect blocks) — precisely v48's
documented removal, starting at 1454, the documented start of the reactive block. The decision
count was unchanged (991), so the trajectory did not branch differently either. ⇒ nothing beyond
v48 had altered the obs, so the regen encodes only the intended deletion.

Regen 2026-08-08 (`gen3_entity_rehome_v1`, v60): the Stage-3 entity re-home — matchup matrices +
6 reactive scalars deleted, protect/trapped/maybe_trapped re-homed per-mon, obs 2925 -> 2667.
ARCH_SIGNATURE bumped in the same commit; the obs-roundtrip fuzz (627 decisions, bit-for-bit) and
the trapping/protect fuzz gates all passed on the new layout before this regen was taken.

**Regen 2026-08-19** (event-window eff side-flip fix; obs dim unchanged at 2501): the tracker
read IMMUNE/RESISTED/SUPEREFFECTIVE events as defender-tagged while the producer tags the
MOVER, so the four `EFF_*` cells were DEAD (all-neutral) on every live battle — the fix
populates them. Proven confined before this regen was taken, by the column-alignment method:
pre-fix vs post-fix vectors over all 991 decisions — decision count unchanged (no branching),
128 changed columns ALL inside the event window, and the changed row-columns exactly the four
`EFF_*` cells. Regression: `event_window_test::
test_effectiveness_from_a_one_sided_turn_lands_on_the_movers_row` (verified red on revert).

**Regen 2026-08-14 for v65** (`gen3_deadline_clock_v1`, obs 2667 -> 2669). ⚠️ AND IT HAPPENED A
THIRD TIME: the clock landed in `cbb0413` without regenerating the fixture, so this test was RED
ON MAIN again — caught only because a full `pytest src/` was run before a ship, exactly as the v48
note above predicted. That is now addressed structurally rather than by another warning: this file
carries the `sim` marker (it plays 6 bridge battles via `golden_obs_capture`), and `sim` is IN the
default developer gate — see the tier table in the root CLAUDE.md. The old gate,
`-m "not integration and not e2e"`, is what let all three regressions hide.

PROOF for this regen, by the same column-alignment method the v48 note establishes (a
value-neutrality linchpin must never be regenerated on a dim mismatch alone — regen bakes in
whatever is live): full vectors captured at `cbb0413^` (pre-clock) and at HEAD, aligned
column-wise over all 991 decisions. Decision count UNCHANGED at 991, so the trajectory did not
branch. Exactly 2 columns inserted, at HEAD indices **1518-1519** — inside the global-env block
(`OFFSET_GLOBAL` 1508, `GLOBAL_ENV_DIM` 20), matching `CLOCK_DIM` 1 -> 3. Deleting precisely those
two columns from the HEAD capture reproduces the pre-clock baseline **bit-for-bit across all 991 x
2667 cells**. So nothing beyond v65 had altered the obs, and this regen encodes only the intended
insertion.

**Regen 2026-09-23** (the TRUTH AUDIT's poke-env READING fixes R1–R3 — a new status starts its own
counter; `-copyboost`'s first ident receives; our active PP is the request's; obs dim unchanged at
2501). Proven confined before this regen, by capturing full vectors at the base commit (which
reproduced the committed fixture 991/991) and with the fixes: decision count UNCHANGED at 991 (no
branching), **32 / 991 decisions** differ, and every differing cell is one of TWO columns, both in
our_team slot 3, both resolved from the layout — `POKEMON_COUNTER_OFFSET` (the SLEEP counter; the
toxic one never moves) and `POKEMON_SLEEP_BELIEF_OFFSET + 1` (`p_wake`). The values are R1's: the
first decision after a Rest read counter 1/4 (carried from the previous status) where the sim has
slept 0 turns, and `p_wake` 1.0 where Rest's 2-turn schedule gives 0.0. R2 and R3 do not occur in
this battle set. Regressions: `poke_env/battle/reading_fixes_test.py` (fails on upstream).

**Regen 2026-09-24** (`gen3_pe_reading_fixes_v1` — the Rust core M2's poke-env findings PE-V10 /
PE-R1b / PE-V16 fixed in the fork; obs dim unchanged at 2501). Proven confined before this regen by
`designs/research_state/measurements/pe_reading_fixes_2026-09-24/golden_diff_census.py`: the base
capture reproduced the committed fixture 991/991; decision count UNCHANGED at 991; **4 / 991
decisions** differ, every cell the TOXIC half of `status_counters` of an opposing badly-poisoned
mon (indices 802 / 924 / 1046 = opp slots 0 / 1 / 2, resolved from the layout), each 0 → 1/8 —
PE-R1b's "one BEHIND" shape, a decision between the residual chip and the next `|turn|`. PE-V10
and PE-V16 do not occur in this battle set (the same harness over 80 battles / 9,987 decisions
shows PE-V10 at 55 decisions, `context[*].boosts` of a FAINTED active only; nothing else moves).
Regressions: `poke_env/battle/reading_fixes_test.py` + `poke_env_gaps/pe_reading_fixes_obs_integration_test.py`.

**Regen 2026-09-26** (`gen3_event_record_v2`, the observation-architecture batch: obs 2501 -> 2761 —
+2 field-sport volatile slots per active context, event rows 22 -> 30 columns, a DENIED row type).
Proven confined before this regen by
`designs/research_state/measurements/obs_arch_batch_2026-09-26/golden_confinement_census.py`: the
base capture (56837827) reproduced the committed fixture 991/991; decision count UNCHANGED at 991;
mapping the HEAD vectors back to the old layout (dropping the four sport columns, which are 0 in
every decision, and the event window) reproduces the base **bit-for-bit on every non-event cell**;
**500 / 991** decisions' event windows differ in the old 22 columns, and every one is explained by
the batch's intended changes alone (inserted DENIED rows, a switch-in row's entry chip and causing
move, an item transfer's RECEIVED direction, the destinybond / perishsong causes, the E4 target) —
0 unexplained.
"""
import json
import os

import pytest

from agents.training.golden_obs_capture import capture_vectors, vector_hashes

# gen3 test tiers (MEASURED 2026-08-14): 4.2 s — battle-backed but CHEAP, which is the whole
# reason `sim` cannot be the marker that decides routine cost. It plays 6 bridge battles and
# BELONGS in the routine gate; excluding it is what let three obs regressions reach main.
pytestmark = pytest.mark.sim

_FIXTURE = os.path.join(os.path.dirname(__file__), "golden_obs_fixture.json")


@pytest.mark.integration
def test_obs_vectors_match_golden():
    with open(_FIXTURE) as f:
        golden = json.load(f)

    vecs = capture_vectors()
    got = vector_hashes(vecs)

    assert vecs, "no vectors captured"
    assert len(vecs[0]) == golden["obs_dim"], (
        f"obs dim changed: fixture {golden['obs_dim']} vs got {len(vecs[0])}"
    )

    assert len(got) == golden["n_decisions"], (
        f"decision count changed ({golden['n_decisions']} -> {len(got)}): the battle "
        f"trajectory diverged. Either determinism broke or an obs change altered branching. "
        f"If the change is intentional (ARCH_SIGNATURE bumped), regenerate the fixture."
    )

    expected = golden["hashes"]
    first_diff = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
    assert first_diff is None, (
        f"obs vector changed at decision {first_diff} of {len(got)} "
        f"(got {got[first_diff][:12]}…, expected {expected[first_diff][:12]}…). "
        f"The data refactor altered an observed value — this is retrain-class. "
        f"If intentional, bump ARCH_SIGNATURE and regenerate the golden fixture."
    )
