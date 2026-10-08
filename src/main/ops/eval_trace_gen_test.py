"""Tests for the OFFLINE eval-cycle generator and the read-side provenance gate — the PURE half.

The provenance vocabulary, the refusals, the seed rule's statement and the read-root spellings are unmarked
and run in every gate — they are where the rules that stop a wrong number from being reported live, and a
rule that only runs in the slow tier is a rule that rides main RED. The cycle itself (played on the Rust eval
core: the readers accept it, a rerun at the same seed is identical, the manifest's regime is the run's) is
``eval_trace_gen_integration_test.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.ops import eval_trace_gen as ETG


# --------------------------------------------------------------------------- the vocabulary

def _live_manifest(**over):
    m = {"step": 10_000_032, "n_games": 100, "selection_schema": 2,
         "opponents": ["random", "heuristic", "staller", "sentinel_0", "sentinel_1"],
         "selection": {"opponents": {}}}
    m.update(over)
    return m


def _gen_manifest(**over):
    m = _live_manifest(n_games=400)
    m[ETG.GENERATED_KEY] = {
        "tool": ETG.GENERATOR_TAG, "schema": ETG.GENERATED_SCHEMA,
        "games_per_opponent": 400, "sentinels_used": 2, "capture": "ALL",
        "eval_sentinel_greedy": True, "seed": 1, "workers": 4, "concurrency": 1,
        "bots": ["random", "heuristic", "staller"],
        "population": "OFFLINE-GENERATED cycle: 5 opponents x 400 games, ALL battles traced",
        "battles_played": 2000, "battles_expected": 2000, "complete": True, "shortfall": 0,
    }
    m.update(over)
    return m


def _partial_manifest(played=1425):
    """A cycle whose workers died part-way — the shape the 2026-09-09 worktree-removal produced."""
    m = _gen_manifest()
    m[ETG.GENERATED_KEY] = {**m[ETG.GENERATED_KEY], "battles_played": played,
                            "battles_expected": 2000, "complete": False,
                            "shortfall": 2000 - played}
    return m


def test_is_generated_separates_the_two_kinds():
    assert ETG.is_generated(_gen_manifest()) is True
    assert ETG.is_generated(_live_manifest()) is False
    # 🚨 A missing or malformed block must read as LIVE-shaped only in the sense of "not
    # generated" — never as a silent True, which would let a cross-population delta through.
    assert ETG.is_generated(None) is False
    assert ETG.is_generated({ETG.GENERATED_KEY: "eval_trace_gen"}) is False


def test_population_is_stated_for_a_live_cycle_too():
    """A header that describes only the unusual side makes the other read as the neutral default."""
    live = ETG.population_of(_live_manifest())
    assert "LIVE" in live and "100 games" in live
    assert "QUOTA" in live, "the live cycle's traces are the outcome quota's slice, not a sample"
    assert "OFFLINE-GENERATED" in ETG.population_of(_gen_manifest())
    assert "UNKNOWN" in ETG.population_of({})


def test_spec_ignores_the_seed_and_keeps_the_frame():
    """Two seeds are two draws from ONE population; two game counts are not."""
    a = ETG.spec_of(_gen_manifest())
    b = _gen_manifest()
    b[ETG.GENERATED_KEY] = {**b[ETG.GENERATED_KEY], "seed": 999, "workers": 1}
    assert ETG.spec_of(b) == a

    c = _gen_manifest(n_games=200)
    assert ETG.spec_of(c) != a


def test_parse_run_ref_splits_the_step(tmp_path):
    run = tmp_path / "some_run"
    run.mkdir()
    d, step = ETG.parse_run_ref(f"{run}@10000032")
    assert d == run.resolve() and step == 10_000_032
    d, step = ETG.parse_run_ref(str(run))
    assert d == run.resolve() and step is None


# --------------------------------------------------------------------------- the refusals

def _fake_run(tmp_path: Path, *, regime=True, snapshots=(4_000_000, 6_000_000, 8_000_000),
              cli_regime="absent"):
    run = tmp_path / "fake_run"
    (run / "snapshots").mkdir(parents=True)
    cfg = {"arch_signature": "x", "config_version": 1}
    if regime:
        cfg["eval_sentinel_greedy"] = True
    (run / "model_config.json").write_text(json.dumps(cfg))
    meta = {"git_hash": "deadbeef"}
    if cli_regime != "absent":
        meta["cli_args"] = {"eval_sentinel_greedy": cli_regime}
    (run / "metadata.json").write_text(json.dumps(meta))
    for s in snapshots:
        (run / "snapshots" / f"snapshot_{s:012d}.zip").write_bytes(b"not-a-real-zip")
    return run


def test_a_run_without_a_recorded_eval_regime_is_refused(tmp_path):
    """🚨 `eval_sentinel_greedy` names an OPPONENT-REGIME BOUNDARY worth ~8.9 pp to the trainee.

    There is no default for it that is not a lie, so a run that recorded none is refused rather
    than measured under a guess.
    """
    run = _fake_run(tmp_path, regime=False)
    with pytest.raises(SystemExit) as exc:
        ETG.read_regime(run)
    assert exc.value.code == 2

    assert ETG.read_regime(_fake_run(tmp_path / "b", regime=True))["eval_sentinel_greedy"] is True


def test_a_preboundary_regime_may_be_DECLARED_but_never_defaulted(tmp_path, capsys):
    """A pre-boundary run is not unreadable — it is UNDECLARED, and the two are different.

    ``model_config.json`` only gained ``eval_sentinel_greedy`` at the 2026-09-07 boundary, so a
    run from before it records the regime nowhere the generator looks. Refusing is still right:
    there is no default that is not a lie. But the refusal must NAME the value recoverable from
    the run's own ``metadata.json`` so the caller can DECLARE it knowingly, and the declared
    cycle must be marked as declared — an assumed regime and a stated one are not the same
    evidence, and `spec_of` bounds every delta the cycle can enter either way.
    """
    run = _fake_run(tmp_path, regime=False, cli_regime=False)
    assert ETG.recoverable_regime(run) is False

    with pytest.raises(SystemExit) as exc:
        ETG.read_regime(run)
    assert exc.value.code == 2
    msg = capsys.readouterr().out + capsys.readouterr().err
    # the refusal names the recovered value AND the flag that would declare it
    assert "--no-eval-sentinel-greedy" in msg
    assert "cli_args.eval_sentinel_greedy" in msg

    declared = ETG.read_regime(run, declared=False)
    assert declared["eval_sentinel_greedy"] is False
    assert declared["eval_sentinel_greedy_source"] == "declared"

    # a run that records the regime keeps saying so, and a CONTRADICTING declaration is refused:
    # the run's own record wins over a caller who disagrees with it.
    recorded = ETG.read_regime(_fake_run(tmp_path / "b", regime=True))
    assert recorded["eval_sentinel_greedy"] is True
    assert recorded["eval_sentinel_greedy_source"] == "recorded"
    with pytest.raises(SystemExit) as exc2:
        ETG.read_regime(_fake_run(tmp_path / "c", regime=True), declared=False)
    assert exc2.value.code == 2


def test_a_run_with_no_regime_anywhere_is_refused_with_no_flag_offered(tmp_path, capsys):
    """Nothing on disk answers the question ⇒ the refusal offers no declaration to make.

    The declaration exists to let a caller state a regime they can VERIFY from the run's own
    record. A run that records it in neither place gives them nothing to verify, so the refusal
    must say "do not generate this cycle" rather than hand over a flag.
    """
    run = _fake_run(tmp_path, regime=False)          # cli_args absent too
    assert ETG.recoverable_regime(run) is None
    with pytest.raises(SystemExit):
        ETG.read_regime(run)
    out = capsys.readouterr().out
    assert "nothing on disk answers the question" in out
    assert "--no-eval-sentinel-greedy" not in out


def test_a_declared_regime_still_bounds_the_delta(tmp_path):
    """🚨 A DECLARATION IS NOT A LICENCE TO DIFFERENCE ACROSS THE BOUNDARY.

    Declaring the regime tells the generator which population to PLAY; it does not make a
    stochastic-sentinel frame comparable to a greedy-sentinel one. `spec_of` keys on the regime
    VALUE, not on how it was learned, so a pre-boundary run's declared cycle is still refused
    against a post-boundary control — which is the whole reason the 75M run cannot be read
    against the 10M ladder's controls.
    """
    pre = _gen_manifest()
    pre[ETG.GENERATED_KEY]["eval_sentinel_greedy"] = False
    pre[ETG.GENERATED_KEY]["eval_sentinel_greedy_source"] = "declared"
    post = _gen_manifest()                            # recorded greedy=True
    assert ETG.spec_of(pre) != ETG.spec_of(post)
    assert ETG.spec_of(pre)["eval_sentinel_greedy"] is False
    # and how it was learned is NOT part of the spec: two declared cycles at one regime compare
    pre2 = _gen_manifest()
    pre2[ETG.GENERATED_KEY]["eval_sentinel_greedy"] = False
    assert ETG.spec_of(pre) == ETG.spec_of(pre2)


def test_sentinels_are_clamped_never_padded(tmp_path):
    """A short run simply has fewer opponent cells. Repeating one to reach the requested count
    would inflate the between-opponent spread with a duplicated cell — which is the very
    quantity the high-power read exists to measure."""
    run = _fake_run(tmp_path, snapshots=(4_000_000, 6_000_000, 8_000_000))
    got, notes = ETG.pick_sentinels(run, 10_000_032, 6)
    assert len(got) == 3, "3 snapshots exist below the step; 6 were asked for"
    assert [s["step"] for s in got] == [4_000_000, 6_000_000, 8_000_000]
    assert any("CLAMPED" in n for n in notes)
    assert len({s["path"] for s in got}) == 3, "no snapshot is repeated to pad the count"


def test_a_snapshot_at_or_above_the_read_step_is_excluded_by_default(tmp_path):
    """A live cycle's pool holds only snapshots OLDER than the trainee. A self-mirror is a
    50%-by-construction cell and would change what the pool row means."""
    run = _fake_run(tmp_path, snapshots=(4_000_000, 10_000_032))
    got, _ = ETG.pick_sentinels(run, 10_000_032, 6)
    assert [s["step"] for s in got] == [4_000_000]
    got, _ = ETG.pick_sentinels(run, 10_000_032, 6, include_current=True)
    assert [s["step"] for s in got] == [4_000_000, 10_000_032]


def test_spacing_keeps_both_endpoints(tmp_path):
    """Evenly spaced across the step range = across the rating range; the weakest and the
    strongest pool opponent are both measured."""
    steps = tuple(range(1_000_000, 9_000_001, 1_000_000))
    run = _fake_run(tmp_path, snapshots=steps)
    got, _ = ETG.pick_sentinels(run, 10_000_032, 3)
    assert [s["step"] for s in got] == [1_000_000, 5_000_000, 9_000_000]


# --------------------------------------------------------------------------- seeding

def test_a_drawn_seed_is_a_62_bit_integer_and_varies():
    """A run with no --seed DRAWS one and records it; the Rust core has no unseeded mode."""
    seeds = {ETG.draw_seed() for _ in range(8)}
    assert len(seeds) == 8 and all(0 <= x < (1 << 62) for x in seeds)


def test_the_reproducibility_note_says_what_the_seed_does_and_does_not_pin():
    note = ETG.REPRODUCIBILITY_NOTE
    assert "gen3_eval_game_seed_v1" in note and "NOT pinned by the seed" in note and "near_ties" in note


def test_the_spec_keeps_the_two_ENGINES_apart():
    """🚨 A cycle generated on the poke-env eval worker (before the Rust port: no `transport`) and one generated on the
    Rust eval core are different engines and encoders — two populations, never differenced."""
    old = _gen_manifest()
    new = _gen_manifest()
    new[ETG.GENERATED_KEY] = {**new[ETG.GENERATED_KEY], "transport": ETG.TRANSPORT}
    assert ETG.spec_of(old)["transport"] == "python_bridge"
    assert ETG.spec_of(new)["transport"] == "rust_eval"
    assert ETG.spec_of(old) != ETG.spec_of(new)
    assert ETG.spec_of(_live_manifest())["transport"] is None


def test_an_odd_game_count_is_refused_under_the_mirrored_pair_regime(tmp_path):
    run = _fake_run(tmp_path)
    cfg = json.loads((run / "model_config.json").read_text())
    (run / "model_config.json").write_text(json.dumps({**cfg, "eval_mirrored_pairs": True}))
    (run / "eval_traces" / "step_9000000").mkdir(parents=True)
    (run / "eval_traces" / "step_9000000" / "snapshot.zip").write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        ETG.main([f"{run}@9000000", "--games", "3", "--out", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists(), "refused before anything was written"


# --------------------------------------------------------------------------- the read-side gate

def _readout(manifest):
    return {"manifest": manifest}


def test_critic_read_refuses_an_offline_vs_live_delta():
    """🚨 The rule the whole provenance block exists for. The two frames differ in games, in
    opponent count, and in whether the traced battles are a sample or the outcome quota's
    loss-enriched slice — every conditioning row is a statistic OF that frame."""
    from main.ops.critic_read import check_comparable
    with pytest.raises(SystemExit) as exc:
        check_comparable(_readout(_gen_manifest()), _readout(_live_manifest()))
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        check_comparable(_readout(_live_manifest()), _readout(_gen_manifest()))
    # both live, and both offline at one spec, are fine
    check_comparable(_readout(_live_manifest()), _readout(_live_manifest()))
    check_comparable(_readout(_gen_manifest()), _readout(_gen_manifest()))


def test_two_offline_frames_must_share_the_spec_but_not_the_seed():
    from main.ops.critic_read import check_comparable
    a = _gen_manifest()
    b = _gen_manifest()
    b[ETG.GENERATED_KEY] = {**b[ETG.GENERATED_KEY], "seed": 7, "workers": 1}
    check_comparable(_readout(a), _readout(b))  # a different seed is two draws from one population

    c = _gen_manifest(n_games=200)
    c[ETG.GENERATED_KEY] = {**c[ETG.GENERATED_KEY], "games_per_opponent": 200}
    with pytest.raises(SystemExit) as exc:
        check_comparable(_readout(a), _readout(c))
    assert exc.value.code == 2


def test_an_incomplete_cycle_is_refused_and_says_so_in_its_population():
    """🚨 THE CASE THE SPEC CHECK CANNOT CATCH ALONE, and the one that actually happened.

    On 2026-09-09 the worktree a generation was running out of was removed under it by
    `land.sh`; all four workers died with `failed to make path absolute`, and the cycle landed at
    71% of its plan with a perfectly well-formed manifest — nominal `n_games` 400, all 12
    opponents present, `selection` recorded, capture rates 1.0. It compares EQUAL to a complete
    cycle under every other field, while carrying two thirds of the battles. Frame size moves
    every fitted conditioning row on its own, which is the whole content of the RETRACTION.
    """
    from main.ops.critic_read import check_comparable
    partial = _partial_manifest()

    assert ETG.completeness(partial)["complete"] is False
    assert ETG.completeness(partial)["shortfall"] == 575
    assert ETG.completeness(_gen_manifest())["complete"] is True
    # a LIVE cycle carries no provenance block and is not judged incomplete by its absence
    assert ETG.completeness(_live_manifest())["complete"] is True

    # the population string LEADS with it, so it cannot be read as the nominal cycle
    text = ETG.population_of(partial)
    assert text.startswith("INCOMPLETE"), text
    assert "1,425 of 2,000" in text

    # and the read refuses, whichever side it is on
    for pair in ((partial, _gen_manifest()), (_gen_manifest(), partial)):
        with pytest.raises(SystemExit) as exc:
            check_comparable(_readout(pair[0]), _readout(pair[1]))
        assert exc.value.code == 2
    # two COMPLETE cycles at one spec still pass
    check_comparable(_readout(_gen_manifest()), _readout(_gen_manifest()))


def _strip_completeness(man):
    man = json.loads(json.dumps(man))
    man[ETG.GENERATED_KEY] = {k: v for k, v in man[ETG.GENERATED_KEY].items()
                              if k not in ("complete", "battles_expected", "shortfall",
                                           "battles_played")}
    return man


def test_completeness_is_DERIVED_when_it_was_not_recorded():
    """A cycle written before the field existed still records everything the number is made of.

    The plan is `n_games` games against each of `opponents`; the `selection` block counts what was
    played. Deriving beats answering UNKNOWN — the same arithmetic would have caught the 71%
    frame — and it is arithmetic on the manifest's OWN fields, not an assumption about them.
    """
    from main.ops.critic_read import check_comparable
    man = _strip_completeness(_gen_manifest())
    # 5 opponents x 400 games = 2000 expected; give the selection block exactly that many played
    man["selection"] = {"opponents": {o: {"battles_played": 400} for o in man["opponents"]}}
    comp = ETG.completeness(man)
    assert comp["battles_expected"] == 2000 and comp["battles_played"] == 2000
    assert comp["complete"] is True and comp["shortfall"] == 0
    check_comparable(_readout(man), _readout(man))

    # ... and the same derivation CATCHES a truncated one, which is the point
    short = _strip_completeness(_gen_manifest())
    short["selection"] = {"opponents": {o: {"battles_played": 285} for o in short["opponents"]}}
    comp = ETG.completeness(short)
    assert comp["complete"] is False and comp["shortfall"] == 2000 - 1425
    assert ETG.population_of(short).startswith("INCOMPLETE")
    with pytest.raises(SystemExit):
        check_comparable(_readout(short), _readout(man))


def test_a_manifest_that_cannot_answer_is_UNKNOWN_not_complete():
    """🚨 The third value, and why `bool()` of it must be the safe answer.

    When neither the recorded fields nor the derivable ones are there, nothing on disk can certify
    the cycle finished. Reading that as True would be the flattering guess.
    """
    from main.ops.critic_read import check_comparable
    man = _strip_completeness(_gen_manifest())
    man.pop("n_games", None)
    man["selection"] = {"opponents": {}}
    comp = ETG.completeness(man)
    assert comp["complete"] is None, "UNKNOWN — never True, and never a bare False either"
    assert bool(comp["complete"]) is False, "a caller that forgets gets the SAFE answer"
    assert "COMPLETENESS UNKNOWN" in ETG.population_of(man)

    with pytest.raises(SystemExit) as exc:
        check_comparable(_readout(man), _readout(_gen_manifest()))
    assert exc.value.code == 2


def test_read_root_accepts_both_spellings(tmp_path):
    """The shadow RUN dir and the eval_traces/step_<N> inside it are both natural things to
    paste, and both normalise to the run-shaped root the offline readers need."""
    from main.ops.critic_read import resolve_read_root
    run = tmp_path / "real_run"
    run.mkdir()
    shadow = tmp_path / "generated"
    step_dir = shadow / "eval_traces" / "step_10000032"
    step_dir.mkdir(parents=True)

    assert resolve_read_root(run, None, role="arm") == run, "no override → the run itself"
    assert resolve_read_root(run, str(shadow), role="arm") == shadow.resolve()
    assert resolve_read_root(run, str(step_dir), role="arm") == shadow.resolve()

    with pytest.raises(SystemExit) as exc:
        resolve_read_root(run, str(tmp_path / "nope"), role="arm")
    assert exc.value.code == 2
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit):
        resolve_read_root(run, str(bare), role="control")


def test_the_read_root_is_part_of_the_cache_key():
    """🚨 Without this, an offline read of (run, step) fingerprints identically to the LIVE read
    of the same (run, step), and the second invocation silently serves the first one's
    artifacts — a 400-game cycle reported under a 100-game readout."""
    from main.ops.critic_read import _fingerprint, _cond_fingerprint
    import argparse
    args = argparse.Namespace(seed=0, bins=10, cond_boot=10, cond_ladder="off")
    live = {"run_dir": "/models/r", "step": 10, "read_root": "/models/r", "manifest": {}}
    off = {"run_dir": "/models/r", "step": 10, "read_root": "/tmp/gen/r", "manifest": {}}
    assert _fingerprint(live, args) != _fingerprint(off, args)
    assert _cond_fingerprint(live, args) != _cond_fingerprint(off, args)


def test_the_report_header_says_the_population_and_warns_off_the_100_game_table():
    from main.ops.critic_read_render import _population_block
    gen = _gen_manifest()[ETG.GENERATED_KEY]
    doc = {"arm": {"population": ETG.population_of(_gen_manifest()), "generated": True,
                   "generated_by": {**gen, "reproducibility_note": "seeded, concurrency 1"}},
           "control": {"population": ETG.population_of(_gen_manifest()), "generated": True,
                       "generated_by": gen}}
    md = _population_block(doc)
    assert "OFFLINE-GENERATED" in md
    assert "never sit in one table" in md, \
        "the hp400 rows must not be pasted beside the 100-game deltas as if comparable"
    assert "seeded, concurrency 1" in md

    live = {"arm": {"population": ETG.population_of(_live_manifest()), "generated": False},
            "control": {"population": ETG.population_of(_live_manifest()), "generated": False}}
    md = _population_block(live)
    assert "LIVE" in md and "OFFLINE-GENERATED" not in md


def test_the_ledger_quote_marks_an_offline_read():
    """A quote of a 400-game replay pasted into the ledger without a marker would be read as the
    registered live read — the same failure the CROSS-STEP marker already exists to stop."""
    from main.ops.critic_read_render import ledger_line
    gen = _gen_manifest()[ETG.GENERATED_KEY]
    doc = {"deltas": [], "arm": {"run": "arm", "step": 10_000_032, "generated": True,
                                 "generated_by": gen},
           "control": {"run": "ctl", "step": 10_000_032}}
    assert "OFFLINE-GENERATED" in ledger_line(doc)
    doc["arm"]["generated"] = False
    assert "OFFLINE-GENERATED" not in ledger_line(doc)


