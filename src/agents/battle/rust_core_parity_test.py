"""The Rust Core parity gate, slice E (events) — ``gen3_core_parity_events_v1``.

The core's READING projection (``src/rust_sim/src/core_events/``) against ``Gen3Battle``'s event log
of the same per-side text, per viewer, per event, ``seq · turn · kind · side · actor · target ·
value · raw`` — type-strict, NO allowlist. The corpus, the replay and the comparison live in
``agents.battle.rust_core_parity``; this file is the two tiers of ``program_rust_core.md`` §3:

* **COMMIT** (routine gate, seconds): the 8 recorded battles of
  ``rust_core_parity_fixtures/commit_tier.json.gz`` + the six byte-fuzz fixtures carrying the four
  ambiguity-prone shapes + the first battle of each of the 22 protocol capture scenarios.
* **MILESTONE** (``slow``; its verdicts land in ``designs/ops/slow_tier_status.json``): 2 × 200
  seeded-random and 2 × 50 production-policy battles PLAYED live — the live logs must ALSO equal
  the offline feed's — plus the protocol corpus × 2 seeds and every byte-fuzz fixture, under the
  committed manifest (the tier refuses on a mismatch).

    python3 -m pytest src/agents/battle/rust_core_parity_test.py -q                # COMMIT
    python3 -m pytest src/agents/battle/rust_core_parity_test.py -m slow -q -n 2   # MILESTONE
"""

from __future__ import annotations

import copy
import logging
from typing import Iterable

import pytest

from agents.battle import rust_core_parity as P

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _assert_clean(census: P.Census, min_events: int, min_kinds: int) -> None:
    print("\n" + census.render())
    assert not census.refused, census.render()
    assert not census.divergences, census.render()
    assert census.events >= min_events, f"only {census.events} events compared — the gate is vacuous"
    assert len(census.kinds) >= min_kinds, f"only {len(census.kinds)} event kinds: {dict(census.kinds)}"


# ---------------------------------------------------------------------------
# COMMIT tier
# ---------------------------------------------------------------------------

def test_commit_tier_core_readings_equal_gen3battle():
    census = P.check_battles(P.commit_corpus(), P.Census())
    _assert_clean(census, min_events=12_000, min_kinds=20)


def test_the_gate_refuses_a_battle_the_core_no_longer_reproduces():
    """The recorded-bytes check has teeth: a fixture whose digest no longer matches is REFUSED."""
    b = copy.deepcopy(P.load_commit_fixture()[0])
    b.chunks_sha = "0" * 64
    census = P.check_battles([b], P.Census())
    assert census.refused and "no longer reproduces" in census.refused[0]


def test_the_comparison_has_teeth():
    """An injected one-field difference, an int-for-float, and a dropped event are all caught."""
    b = P.load_commit_fixture()[0]
    res = P.run_core([b])[0]
    chunks = P.core_chunks(res)
    ref = P.reference(chunks, "p1")
    events = res["viewers"][0]

    def census_of(evs) -> P.Census:
        c = P.Census()
        P.compare_viewer(evs, ref, "inject", c)
        return c

    assert not census_of(events).divergences
    tampered = copy.deepcopy(events)
    dmg = next(r for ev in tampered for r in ev["readings"] if r["kind"] == "DAMAGE")
    dmg["value"]["hp_after"] = dmg["value"]["hp_after"] + 1e-12
    assert "DAMAGE.value" in census_of(tampered).divergences
    tampered = copy.deepcopy(events)
    mv = next(r for ev in tampered for r in ev["readings"] if r["kind"] == "MOVE")
    mv["value"]["target_status"] = "PAR" if mv["value"]["target_status"] != "PAR" else None
    assert "MOVE.value" in census_of(tampered).divergences
    tampered = copy.deepcopy(events)
    dmg = next(r for ev in tampered for r in ev["readings"] if r["kind"] == "DAMAGE")
    dmg["value"]["amount"] = int(dmg["value"]["amount"])            # a float read as an int
    assert "DAMAGE.value" in census_of(tampered).divergences
    tampered = copy.deepcopy(events)
    next(ev for ev in tampered if ev["readings"])["readings"].pop()
    assert "[event COUNT]" in census_of(tampered).divergences


def test_the_golden_records_round_trip_and_reparse():
    """The persisted RECORD (`gen3_core_event_v1`, `core_events::record`): every golden record —
    the COMMIT tier's battles + one battle per protocol scenario, both viewers — reads, re-writes
    BYTE-IDENTICALLY, and re-parses from its stored text to its stored typed stream (the
    migrate-by-reparse path a schema change must pass)."""
    records = sorted(P.RECORD_GOLDEN_DIR.glob("*.jsonl.gz"))
    assert len(records) >= 60, f"only {len(records)} golden records"
    assert P.check_golden_records() == []


# ---------------------------------------------------------------------------
# MILESTONE tier
# ---------------------------------------------------------------------------

def _played(keys: Iterable[int], policy=None) -> P.Census:
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    census = P.Census()
    lives = [P.play(k, policy=policy) for k in keys]
    for lv in lives:
        P.compare_live(lv, census)
    return P.check_battles([lv.recorded for lv in lives], census)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["keys_0_199", "keys_5000_5199"])
def test_milestone_seeded_random_battles(seed):
    P.check_manifest()
    census = _played(P.MILESTONE_RANDOM_KEYS[seed])
    _assert_clean(census, min_events=100_000, min_kinds=24)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["policy_keys_100_149", "policy_keys_6000_6049"])
def test_milestone_production_policy_battles(seed):
    P.check_manifest()
    model = P.load_production_policy()
    census = _played(P.MILESTONE_POLICY_KEYS[seed], policy=model)
    _assert_clean(census, min_events=5_000, min_kinds=15)


@pytest.mark.slow
def test_milestone_protocol_and_byte_fuzz_corpora():
    P.check_manifest()
    census = P.check_battles(P.protocol_battles(2) + P.byte_fuzz_battles(), P.Census())
    _assert_clean(census, min_events=40_000, min_kinds=25)
