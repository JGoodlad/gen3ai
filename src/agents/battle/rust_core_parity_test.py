"""The Rust Core parity gate — slice E (events, ``gen3_core_parity_events_v1``) and slice V (views
+ legality, the TRUTH AUDIT, ``gen3_core_parity_views_v1`` — :mod:`rust_core_parity_views`).

Slice V's tests below: the COMMIT-tier gate, its TEETH (a re-introduced Baton Pass drop and a
misread Spikes layer each FAIL; a dropped capture FAILS as ``[ALIGN]``), and the classification
completeness check; the MILESTONE tests run both slices on the same played battles.

The core's READING projection (``src/rust_sim/src/core_events/``) against ``Gen3Battle``'s event log
of the same per-side text, per viewer, per event, ``seq · turn · kind · side · actor · target ·
value · raw`` — type-strict, NO allowlist. The corpus, the replay and the comparison live in
``agents.battle.rust_core_parity``; this file is the two tiers of ``program_rust_core.md`` §3:

* **COMMIT** (routine gate, seconds): the 10 recorded battles of
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
from typing import Iterable, Tuple

import pytest

from agents.battle import rust_core_parity as P
from agents.battle import rust_core_parity_views as V

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _assert_clean(census: P.Census, min_events: int, min_kinds: int) -> None:
    print("\n" + census.render())
    assert not census.refused, census.render()
    assert not census.divergences, census.render()
    assert census.events >= min_events, f"only {census.events} events compared — the gate is vacuous"
    assert len(census.kinds) >= min_kinds, f"only {len(census.kinds)} event kinds: {dict(census.kinds)}"
    _assert_selfcheck_ran(census)


def _assert_selfcheck_ran(census: P.Census) -> None:
    """The corpus ran through the EMISSION SELF-CHECK build (`gen3_core_emission_selfcheck_v1`):
    every core process reported its counts, and every line was checked. A production
    `core_events` (no counts) makes this gate refuse rather than pass without the check."""
    sc = census.selfcheck
    assert not sc.get("runs_without"), (
        f"{sc['runs_without']} core_events run(s) had NO emission self-check — the binary is a "
        "production build; unset POKESIM_CORE_EVENTS_BIN or point it at target/selfcheck/")
    assert sc.get("omniscient", 0) > 0 and sc.get("per_viewer", 0) >= 2 * sc["omniscient"] * 0.9, dict(sc)


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
# slice V — the TRUTH AUDIT: every decision's LiveView + LegalActions, both viewers
# ---------------------------------------------------------------------------

def _assert_views_clean(views: V.ViewCensus, min_decisions: int, min_truth: int) -> None:
    print("\n" + views.render())
    assert not views.refused, views.render()
    assert not views.divergences, views.render()
    assert views.decisions >= min_decisions, f"only {views.decisions} decisions — vacuous"
    assert sum(views.truth_checks.values()) >= min_truth, dict(views.truth_checks)


def _baton_pass_battles():
    fx = {b.label: b for b in P.load_commit_fixture()}
    return [fx[f"random_{k}"] for k in P.COMMIT_BATON_PASS_KEYS]


def test_commit_tier_views_equal_liveview():
    """Slice V at the COMMIT tier: the projection (``one_sided_view`` + the named reading rules)
    and the engine truth against the LiveView training builds, at every decision of every
    in-scope battle, both viewers — the recorded battles INCLUDING the two Baton Pass ones."""
    views = V.ViewCensus()
    P.check_battles(P.commit_corpus(), P.Census(), views=views)
    _assert_views_clean(views, min_decisions=1_700, min_truth=35_000)
    assert views.battles >= 13, f"only {views.battles} in-scope battles"


def test_the_view_slice_catches_a_dropped_baton_pass(monkeypatch):
    """TEETH, the motivating class: re-introduce poke-env's pre-2026-08-23 behaviour (a Baton
    Pass carries NOTHING to the entrant) and the gate must FAIL on the SIM-FACT boosts of the
    entrant AND on the engine-truth volatiles (the passed Substitute)."""
    from poke_env.battle.pokemon import Pokemon

    monkeypatch.setattr(Pokemon, "apply_baton_pass", lambda self, snapshot: None)
    views = V.ViewCensus()
    P.check_battles(_baton_pass_battles(), P.Census(), views=views)
    keys = set(views.divergences)
    print(views.render())
    assert any(k.startswith("[SIM-FACT]") and k.endswith(".boosts") for k in keys), keys
    assert any(k.startswith("[TRUTH]") and k.endswith(".volatiles") for k in keys), keys


def test_the_view_slice_catches_a_misread_hazard_layer(monkeypatch):
    """TEETH, a second sim-fact class: poke-env stores a Spikes stack as the TURN it started
    (as for a screen) instead of its layer count — the gate must FAIL on side_conditions."""
    from poke_env.battle.abstract_battle import AbstractBattle
    from poke_env.battle.side_condition import SideCondition

    real = AbstractBattle._side_start

    def misread(self, side, condition_str):
        conds = self.side_conditions if side[:2] == self._player_role else self.opponent_side_conditions
        if SideCondition.from_showdown_message(condition_str) is SideCondition.SPIKES:
            conds[SideCondition.SPIKES] = self.turn
            return
        real(self, side, condition_str)

    monkeypatch.setattr(AbstractBattle, "_side_start", misread)
    views = V.ViewCensus()
    P.check_battles(P.load_commit_fixture(), P.Census(), views=views)
    assert any(k.startswith("[SIM-FACT]") and k.endswith("side_conditions")
               for k in views.divergences), views.render()


def test_the_view_slice_refuses_a_decision_it_cannot_align():
    """A decision the reading takes where the core shipped no request (or the reverse) is an
    [ALIGN] divergence, never a silent skip."""
    b = P.load_commit_fixture()[0]
    res = P.run_core([b], views=True)[0]
    caps = [c for i, c in enumerate(res["views"]) if i != 5]
    views = V.ViewCensus()
    V.check_views(b.label, P.core_chunks(res), caps, views,
                  teams={"p1": b.p1["team"], "p2": b.p2["team"]})
    assert any(k.startswith("[ALIGN]") for k in views.divergences), views.render()


def test_every_read_model_field_is_classified():
    """A field added to ``LivePokemon`` / ``LegalActions`` must be classified SIM-FACT or
    PRESENTATION (with a named rule) before the gate will run — no field rides unclassified."""
    from dataclasses import fields

    from agents.battle.live_view import LegalActions, LivePokemon

    assert {f.name for f in fields(LivePokemon)} == set(V.MON_FIELDS)
    assert {f.name for f in fields(LegalActions)} - {"last_request"} == set(V.LEGAL_FIELDS)
    for table in (V.SIDE_FIELDS, V.VIEW_FIELDS, V.LEGAL_FIELDS):
        for cls, rule in table.values():
            assert cls in (V.SIM, V.RULE) and (rule is None or rule in V.RULES), (cls, rule)
    for own, opp, rule in V.MON_FIELDS.values():
        assert {own, opp} <= {V.SIM, V.RULE} and (rule is None or rule in V.RULES)
        assert V.RULE not in (own, opp) or rule is not None, "a PRESENTATION field needs a rule"
    print(V.field_census())


# ---------------------------------------------------------------------------
# MILESTONE tier (slices E + V on the same played battles)
# ---------------------------------------------------------------------------

def _played(keys: Iterable[int], policy=None) -> Tuple[P.Census, V.ViewCensus]:
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    census, views = P.Census(), V.ViewCensus()
    lives = [P.play(k, policy=policy) for k in keys]
    for lv in lives:
        P.compare_live(lv, census)
    P.check_battles([lv.recorded for lv in lives], census, views=views)
    return census, views


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["even_keys_whole_pool", "odd_keys_whole_pool"])
def test_milestone_seeded_random_battles(seed):
    P.check_manifest()
    census, views = _played(P.MILESTONE_RANDOM_KEYS[seed])
    _assert_clean(census, min_events=180_000, min_kinds=24)
    _assert_views_clean(views, min_decisions=55_000, min_truth=1_200_000)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["policy_keys_100_149", "policy_keys_6000_6049"])
def test_milestone_production_policy_battles(seed):
    P.check_manifest()
    model = P.load_production_policy()
    census, views = _played(P.MILESTONE_POLICY_KEYS[seed], policy=model)
    _assert_clean(census, min_events=5_000, min_kinds=15)
    _assert_views_clean(views, min_decisions=3_000, min_truth=50_000)


@pytest.mark.slow
def test_milestone_protocol_and_byte_fuzz_corpora():
    P.check_manifest()
    census = P.check_battles(P.protocol_battles(2) + P.byte_fuzz_battles(), P.Census())
    _assert_clean(census, min_events=40_000, min_kinds=25)
