"""Value-neutrality linchpin: the per-decision observation vector must not change.

Replays the fixed, deterministic battle set from ``golden_obs_capture`` and asserts every
decision's full obs vector is byte-identical to the golden fixture captured on HEAD before the
``gen3_data`` refactor. Because the obs vector is the entire downstream-visible product of the
data layer, a green run proves the refactor (data facade, type-chart/natures sourced from
``data/`` instead of poke-env) changed **no observed value** — i.e. no ``ARCH_SIGNATURE`` bump
is warranted.

Bridge-backed (real in-process battles, no server), so it's an integration test. If an
intentional obs-value change lands (with an ``ARCH_SIGNATURE`` bump), regenerate the fixture:
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
"""
import json
import os

import pytest

from agents.training.golden_obs_capture import capture_vectors, vector_hashes

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
