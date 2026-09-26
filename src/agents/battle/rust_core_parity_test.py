"""The Rust Core parity gate — slice E (events, ``gen3_core_parity_events_v1``), slice V (views
+ legality, the TRUTH AUDIT, ``gen3_core_parity_views_v1`` — :mod:`rust_core_parity_views`) and slice
T (the trackers, the α/β label and the reward, ``gen3_core_parity_trackers_v1`` —
:mod:`rust_core_parity_trackers`) and slice O (the 2501-dim observation row, BYTE-equal,
``gen3_core_parity_obs_v1`` — :mod:`rust_core_parity_obs`).

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
* **MILESTONE** (``slow``; its verdicts land in ``designs/ops/slow_tier_status.json``): 2 × 360
  seeded-random and 2 × 50 POLICY battles PLAYED live — the live logs must ALSO equal
  the offline feed's — 2 × 150 LADDER battles (the LADDER-USAGE corpus; its NAMED known
  divergences run in their own test and must still fire), plus the protocol corpus × 2 seeds and
  every byte-fuzz fixture, under the committed manifest (the tier refuses on a mismatch).
  The POLICY battles are driven by a FRESHLY BUILT, seeded, untrained current-architecture model
  (``main.fresh_checkpoint``), not the ``production`` baseline: the observation-architecture
  batch (v121, MIGRATION_FLOOR 121) put every named baseline behind the pre-generation wall, and
  this slice needs a policy that DRIVES battles, not a strong one. It returns to ``production``
  once a v121 production node exists (the ``ai_v14_01_base`` lineage).

    python3 -m pytest src/agents/battle/rust_core_parity_test.py -q                # COMMIT
    python3 -m pytest src/agents/battle/rust_core_parity_test.py -m slow -q -n 2   # MILESTONE
"""

from __future__ import annotations

import copy
import logging
from typing import Iterable, Tuple

import pytest

from agents.battle import rust_core_parity as P
from agents.battle import rust_core_parity_obs as O
from agents.battle import rust_core_parity_trackers as T
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

def _assert_views_clean(views: V.ViewCensus, min_decisions: int, min_board: int) -> None:
    print("\n" + views.render())
    assert not views.refused, views.render()
    assert not views.divergences, views.render()
    assert views.decisions >= min_decisions, f"only {views.decisions} decisions — vacuous"
    assert sum(views.board_checks.values()) >= min_board, dict(views.board_checks)


def _baton_pass_battles():
    fx = {b.label: b for b in P.load_commit_fixture()}
    return [fx[f"random_{k}"] for k in P.COMMIT_BATON_PASS_KEYS]


def test_commit_tier_views_equal_liveview():
    """Slice V at the COMMIT tier: the core's ``present()`` view + legality (and its board audit
    against the engine) against the LiveView training builds, at every decision of every
    in-scope battle, both viewers — the recorded battles INCLUDING the two Baton Pass ones."""
    views = V.ViewCensus()
    P.check_battles(P.commit_corpus(), P.Census(), views=views)
    _assert_views_clean(views, min_decisions=1_700, min_board=300_000)
    assert views.battles >= 13, f"only {views.battles} in-scope battles"


def test_the_view_slice_catches_a_dropped_baton_pass(monkeypatch):
    """TEETH, the motivating class: re-introduce poke-env's pre-2026-08-23 behaviour (a Baton
    Pass carries NOTHING to the entrant) and the gate must FAIL on the SIM-FACT boosts of the
    entrant AND on its volatiles (the passed Substitute) — against the CORE column, whose own view
    the board audit holds to the engine (the port's projection + reading-vs-engine TRUTH checks
    that used to catch this directly are deleted, program §4 M2)."""
    from poke_env.battle.pokemon import Pokemon

    monkeypatch.setattr(Pokemon, "apply_baton_pass", lambda self, snapshot: None)
    views = V.ViewCensus()
    P.check_battles(_baton_pass_battles(), P.Census(), views=views)
    keys = set(views.divergences)
    print(views.render())
    assert any(k.startswith("[core][SIM-FACT]") and k.endswith(".boosts") for k in keys), keys
    assert any(k.startswith("[core]") and k.endswith(".volatiles") for k in keys), keys


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
    assert any(k.startswith("[core][SIM-FACT]") and k.endswith("side_conditions")
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
# slice T — the TRACKERS, the α/β label and the reward, every decision, both viewers (M3)
# ---------------------------------------------------------------------------

def _assert_trackers_clean(t: T.TrackerCensus, min_decisions: int, min_rows: int) -> None:
    print("\n" + t.render())
    assert not t.refused, t.render()
    assert not t.divergences, t.render()
    assert t.decisions >= min_decisions, f"only {t.decisions} decisions — vacuous"
    assert t.window_rows >= min_rows, f"only {t.window_rows} event-window rows compared"
    assert all(t.labels[k] > 0 for k in (0, 1, 2)), f"a label kind never fired: {dict(t.labels)}"
    assert t.terminal_rewards > 0, "no terminal reward compared"


def test_commit_tier_trackers_equal_episode_tracker():
    """Slice T at the COMMIT tier: the core's trackers (folded on the version from the viewer's
    stream) == the EpisodeTracker training drives, at every decision, both viewers, field by field
    — plus the α/β label and the win-indicator reward."""
    t, views = T.TrackerCensus(), V.ViewCensus()
    # slice V rides the SAME replay: with the trackers on, a decision hands its `present()` view to
    # the version's memo, and slice V is what holds that view to the reading
    P.check_battles(P.commit_corpus(), P.Census(), trackers=t, views=views)
    _assert_trackers_clean(t, min_decisions=1_700, min_rows=40_000)
    _assert_views_clean(views, min_decisions=1_700, min_board=300_000)


def _commit_trackers_with(monkeypatch, target, name, fn) -> T.TrackerCensus:
    monkeypatch.setattr(target, name, fn)
    t = T.TrackerCensus()
    P.check_battles(P.load_commit_fixture()[:4], P.Census(), trackers=t)
    return t


def test_the_tracker_slice_catches_a_misattributed_residual(monkeypatch):
    """TEETH, the v81 class: fold residual damage into the attacking move's `hp_delta` again (the
    event window's `[from]` guard dropped) — the gate must FAIL on the window rows."""
    from agents.battle.battle_event import BattleEvent

    t = _commit_trackers_with(monkeypatch, BattleEvent, "from_clause", property(lambda self: None))
    assert any(k.startswith("[TRACKER] window") for k in t.divergences), t.render()


def test_the_tracker_slice_catches_a_clock_that_never_resets(monkeypatch):
    """TEETH: a progress clock whose PROGRESS clause never fires — the gate must FAIL on `clock`."""
    from agents.training.progress_clock import ProgressClock

    t = _commit_trackers_with(monkeypatch, ProgressClock, "_is_progress", staticmethod(lambda *a, **k: False))
    assert any(k.startswith("[TRACKER] clock") for k in t.divergences), t.render()


def test_the_tracker_slice_catches_a_phaze_labelled_as_a_choice(monkeypatch):
    """TEETH, the label: an intent labeller that forgets the PHAZE mask (a switch that co-occurs with
    a resolved move is OUR Roar, not their choice) — the gate must FAIL on `label`."""
    import agents.training.opp_intent_labels as L

    real = L.build_opp_intent_label

    def no_phaze_mask(delta, move_num_of, opp_slot_of_species, species_num_of=None):
        if delta is not None and delta.opp_switch_to and not delta.opp_fainted \
                and not delta.phase_is_forced_switch:
            return (L.KIND_SWITCH, 0, L.SWITCH_SLOT_NONE,
                    0 if species_num_of is None else (species_num_of(delta.opp_switch_to) or 0))
        return real(delta, move_num_of, opp_slot_of_species, species_num_of)

    t = _commit_trackers_with(monkeypatch, L, "build_opp_intent_label", no_phaze_mask)
    assert any(k.startswith("[TRACKER] label") for k in t.divergences), t.render()


def test_the_information_boundary_holds_on_the_core_record_and_the_check_has_teeth(monkeypatch):
    """ACTION DENIAL's information boundary at the SLICE level: every denied / refused action in
    the core's native record over the COMMIT corpus carries ``"opp"`` for an opponent (the viewer
    never saw the opponent's choice) and ``{"own": …}`` for our own — and the check FAILS on a real
    opponent denial whose choice is made to leak."""
    seen = {"opp": [], "ours": []}

    def take(_b, res):
        for viewer in res.get("trackers") or []:
            for cap in viewer:
                for a in cap.get("window") or ():
                    if a["kind"] in ("denied", "cant"):
                        seen[(a["actor"] if a["kind"] == "denied" else a["mon"])[0]].append(a)

    t = T.TrackerCensus()
    P.check_battles(P.commit_corpus(), P.Census(), trackers=t, on_result=take)
    assert not [k for k in t.divergences if k.startswith("[BOUNDARY]")], t.render()
    assert len(seen["opp"]) >= 20 and len(seen["ours"]) >= 20, {k: len(v) for k, v in seen.items()}
    assert T.boundary_violations(seen["opp"] + seen["ours"]) == []
    leak = dict(seen["opp"][0], choice={"own": "move 1"})
    assert T.boundary_violations([leak]) == [(0, leak)]
    hidden = dict(seen["ours"][0], choice="opp")
    assert T.boundary_violations([hidden]) == [(0, hidden)]
    # and slice T itself runs the check: a record that leaks FAILS the gate
    t = _commit_trackers_with(monkeypatch, T, "boundary_violations", lambda w: [(0, leak)])
    assert any(k.startswith("[BOUNDARY]") for k in t.divergences), t.render()


# ---------------------------------------------------------------------------
# slice O — the OBSERVATION ROW, every decision, both viewers, byte-equal (M4)
# ---------------------------------------------------------------------------

#: The obs blocks every tier must see NONZERO somewhere (the gate cannot be green on zeros).
_OBS_BLOCKS = ("our_team", "opp_team", "context", "global", "board", "pair_history", "event_window")


def _assert_obs_clean(o: O.ObsCensus, min_decisions: int) -> None:
    print("\n" + o.render())
    assert not o.refused, o.render()
    assert not o.divergences, o.render()
    assert o.decisions >= min_decisions, f"only {o.decisions} decisions — vacuous"
    assert o.rows_equal == o.decisions, o.render()
    missing = [b for b in _OBS_BLOCKS if o.nonzero_blocks[b] == 0]
    assert not missing, f"obs blocks never nonzero: {missing} ({dict(o.nonzero_blocks)})"


def test_commit_tier_obs_rows_equal_the_python_encoder():
    """Slice O at the COMMIT tier: the Rust encoder's row (``core_events --obs``, the self-check
    build's NaN-poisoned prefill — an unwritten cell reads NaN and fails) == the row
    ``Gen3Env.embed_battle`` encodes, BYTE for byte, plus the 11-dim mask, at every decision of
    every in-scope battle, both viewers."""
    o, t = O.ObsCensus(), T.TrackerCensus()
    P.check_battles(P.commit_corpus(), P.Census(), trackers=t, obs=o)
    _assert_obs_clean(o, min_decisions=1_700)
    _assert_trackers_clean(t, min_decisions=1_700, min_rows=40_000)


def test_the_obs_golden_is_reproduced_by_the_core():
    """Every obs GOLDEN (``training/golden_obs_fixture.json`` — the per-decision sha256 of the
    trainee's row over the fixed deterministic battle set): the core, replaying the same battles'
    input logs, writes rows with EXACTLY the committed hashes, in order — and equal, byte for byte,
    to the vectors the Python capture recorded in the same run."""
    import json

    from agents.training.golden_obs_capture import vector_hashes
    from utils.paths import repo_path

    golden = json.loads(repo_path("src", "agents", "training", "golden_obs_fixture.json").read_text())
    py_vectors, battles = O.golden_battles()
    rows = O.core_rows(battles)
    assert len(py_vectors) == golden["n_decisions"], "the golden battles no longer replay (determinism)"
    assert len(rows) == len(py_vectors), (len(rows), len(py_vectors))
    first = next((i for i, (a, b) in enumerate(zip(rows, py_vectors)) if a.tobytes() != b.tobytes()), None)
    assert first is None, f"core row != the capture's Python row at decision {first}"
    got = vector_hashes(rows)
    first = next((i for i, (a, b) in enumerate(zip(got, golden["hashes"])) if a != b), None)
    assert first is None, f"core row != the committed golden at decision {first} of {len(got)}"


def test_the_obs_slice_catches_a_changed_python_encoder(monkeypatch):
    """TEETH, the day-it-lands property: a Python encoder change the core does not mirror (the
    move slot's PP normaliser) FAILS slice O on exactly that field."""
    import agents.observation.moves as M

    monkeypatch.setattr(M, "MAX_PP", 32)
    o = O.ObsCensus()
    P.check_battles(P.load_commit_fixture()[:2], P.Census(), obs=o)
    keys = set(o.divergences)
    assert any(k.startswith("[OBS] our_team moves+") for k in keys), o.render()
    assert any(k.startswith("[OBS] opp_team moves+") for k in keys), o.render()


def test_the_obs_slice_catches_an_unwritten_cell_and_a_signed_zero():
    """TEETH, the NaN poison: a core row with ONE NaN cell (a cell the encoder never wrote, as the
    test build's prefill leaves it) FAILS, as does a ``-0.0`` where Python wrote ``0.0`` — the gate
    compares BYTES, not ``==``."""
    import base64

    import numpy as np

    b = P.load_commit_fixture()[0]
    res = P.run_core([b], trackers=True, obs=True)[0]
    cap = next(c for c in res["trackers"][0] if "obs" in c)
    row = np.frombuffer(base64.b64decode(cap["obs"]["b64"]), dtype="<f4").copy()
    mask = np.array(cap["mask"])

    def with_row(r):
        return dict(cap, obs=dict(cap["obs"], b64=base64.b64encode(r.astype("<f4").tobytes()).decode()))

    clean = O.ObsCensus()
    O.compare_row("clean", cap, row, mask, clean)
    assert clean.rows_equal == 1 and not clean.divergences
    zero_at = int(np.flatnonzero(row == 0)[0])
    poisoned = row.copy()
    poisoned[zero_at] = np.nan
    o = O.ObsCensus()
    O.compare_row("nan", with_row(poisoned), row, mask, o)
    assert o.nan_cells == 1 and o.divergences, o.render()
    signed = row.copy()
    signed[zero_at] = -0.0
    o = O.ObsCensus()
    O.compare_row("signed", with_row(signed), row, mask, o)
    assert o.byte_only_cells == 1 and o.divergences, o.render()
    # the CHOICE TOKENS (what search branches on): a different token FAILS
    tokens = dict(cap["tokens"])
    assert tokens, "fixture: a decision with legal actions"
    o = O.ObsCensus()
    O.compare_row("tokens", cap, row, mask, o, py_tokens=tokens)
    assert not o.divergences and o.tokens == len(tokens)
    k = next(iter(tokens))
    o = O.ObsCensus()
    O.compare_row("tokens", cap, row, mask, o, py_tokens=dict(tokens, **{k: tokens[k] + "x"}))
    assert "[TOKENS] the choice string per legal action" in o.divergences, o.render()


# ---------------------------------------------------------------------------
# MILESTONE tier (slices E + V on the same played battles)
# ---------------------------------------------------------------------------

def _played(keys: Iterable[int], policy=None,
            source=None) -> Tuple[P.Census, V.ViewCensus, T.TrackerCensus, O.ObsCensus]:
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    census, views, trackers, obs = P.Census(), V.ViewCensus(), T.TrackerCensus(), O.ObsCensus()
    lives = [P.play(k, policy=policy, source=source) for k in keys]
    for lv in lives:
        P.compare_live(lv, census)
    P.check_battles([lv.recorded for lv in lives], census, views=views, trackers=trackers, obs=obs)
    return census, views, trackers, obs


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["even_keys_whole_pool", "odd_keys_whole_pool"])
def test_milestone_seeded_random_battles(seed):
    P.check_manifest()
    census, views, trackers, obs = _played(P.MILESTONE_RANDOM_KEYS[seed])
    _assert_clean(census, min_events=180_000, min_kinds=24)
    _assert_views_clean(views, min_decisions=55_000, min_board=1_200_000)
    _assert_trackers_clean(trackers, min_decisions=55_000, min_rows=1_500_000)
    _assert_obs_clean(obs, min_decisions=55_000)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["policy_keys_100_149", "policy_keys_6000_6049"])
def test_milestone_production_policy_battles(seed, tmp_path):
    """Named for the ``production`` policy it will play again once a v121 production node exists;
    until then a fresh seeded v121 model (seed fixed per parametrization) drives the battles."""
    from main.fresh_checkpoint import load_fresh_policy

    P.check_manifest()
    model = load_fresh_policy(tmp_path / "fresh_v121", seed=20260926 + seed)
    census, views, trackers, obs = _played(P.MILESTONE_POLICY_KEYS[seed], policy=model)
    _assert_clean(census, min_events=5_000, min_kinds=15)
    _assert_views_clean(views, min_decisions=3_000, min_board=50_000)
    _assert_trackers_clean(trackers, min_decisions=3_000, min_rows=80_000)
    _assert_obs_clean(obs, min_decisions=3_000)


@pytest.mark.slow
def test_milestone_protocol_and_byte_fuzz_corpora():
    P.check_manifest()
    census = P.check_battles(P.protocol_battles(2) + P.byte_fuzz_battles(), P.Census())
    _assert_clean(census, min_events=40_000, min_kinds=25)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1], ids=["ladder_keys_0_149", "ladder_keys_150_299"])
def test_milestone_ladder_battles(seed):
    """The LADDER-USAGE corpus (``gen3_ladder_usage_corpus_v1``): real public-ladder teams, both
    slices, every decision — the surface the Heal Bell crash hid on. Battle ``key`` plays corpus
    teams ``2·key`` / ``2·key+1`` of the MILESTONE tier, so the two ranges play 600 teams once each.
    The NAMED known divergences (``P.LADDER_KNOWN_DIVERGENCES``) run in their own test."""
    P.check_manifest()
    keys = [k for k in P.MILESTONE_LADDER_KEYS[seed] if k not in P.LADDER_KNOWN_DIVERGENCES]
    census, views, trackers, obs = _played(keys, source="ladder")
    _assert_clean(census, min_events=120_000, min_kinds=24)
    _assert_views_clean(views, min_decisions=20_000, min_board=500_000)
    _assert_trackers_clean(trackers, min_decisions=20_000, min_rows=500_000)
    _assert_obs_clean(obs, min_decisions=20_000)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(P.LADDER_KNOWN_DIVERGENCES))
def test_milestone_ladder_known_divergences_still_fire(key):
    """Each NAMED exclusion still diverges in EXACTLY its named census keys and nothing else — so
    the day its class is fixed this fails, and the entry must go (an allowlist entry that outlives
    its fix misleads every reader after it). Slice E stays fully clean on these battles."""
    P.check_manifest()
    name, want, _where = P.LADDER_KNOWN_DIVERGENCES[key]
    census, views, trackers, obs = _played([key], source="ladder")
    print("\n" + census.render() + "\n" + views.render() + "\n" + trackers.render() + "\n" + obs.render())
    assert not census.divergences and not census.refused, census.render()
    assert not trackers.divergences and not trackers.refused, trackers.render()
    assert not obs.divergences and not obs.refused, obs.render()
    assert not views.refused, views.render()
    assert set(views.divergences) == set(want), (name, dict(views.divergences))
