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


def _strip_completeness(man):
    man = json.loads(json.dumps(man))
    man[ETG.GENERATED_KEY] = {k: v for k, v in man[ETG.GENERATED_KEY].items()
                              if k not in ("complete", "battles_expected", "shortfall",
                                           "battles_played")}
    return man


# --------------------------------------------------------------------------- the SPECIALIST's team pin

def _specialist_run(tmp_path: Path, teams, *, multi: bool, record_shas=True, spec_kind=None):
    """A run that pinned its trainee: the pin lives ONLY in metadata.json (cli_args + the declared matchup)."""
    run = _fake_run(tmp_path)
    files = []
    for i, text in enumerate(teams):
        f = tmp_path / f"team_{i}.txt"
        f.write_text(text, encoding="utf-8")
        files.append(str(f))
    shas = [ETG._pin_sha(t) for t in teams]
    ets = {"kind": spec_kind or ("pin_multi" if multi else "pinned"), "pin_file": files[0], "bias_prob": 0.0,
           "pin_sha": shas[0] if record_shas else None}
    if multi and record_shas:
        ets["pin_shas"] = shas
    cli = {"eval_sentinel_greedy": True, "_matchup_spec": {"eval_trainee_teams": ets},
           ("trainee_teams" if multi else "trainee_team"): ",".join(files) if multi else files[0]}
    (run / "metadata.json").write_text(json.dumps({"git_hash": "deadbeef", "cli_args": cli}))
    return run, files


def test_a_specialist_cycle_pilots_the_runs_OWN_pin(tmp_path):
    """🚨 The pin is in metadata.json, never model_config.json (whose parser has no such field).

    The old read took `trainee_team_str` from model_config.json — a key that cannot legally be there —
    so EVERY specialist cycle silently piloted random pool teams. Reverting the read makes both reads
    below None.
    """
    run, _ = _specialist_run(tmp_path, ["Starmie @ Leftovers\n"], multi=False)
    reg = ETG.read_regime(run)
    assert reg["trainee_team_str"] == "Starmie @ Leftovers\n"
    assert reg["trainee_pin"]["kind"] == "pinned" and reg["trainee_pin"]["verified"] is True

    run2, files = _specialist_run(tmp_path / "m", ["A @ B\n", "C @ D\n"], multi=True)
    reg2 = ETG.read_regime(run2)
    assert reg2["trainee_team_str"] == ["A @ B\n", "C @ D\n"]     # the LIST the trainer hands eval
    assert reg2["trainee_pin"]["files"] == files


def test_a_pool_run_reads_as_the_pool(tmp_path):
    reg = ETG.read_regime(_fake_run(tmp_path))
    assert reg["trainee_team_str"] is None and reg["trainee_pin"]["files"] == []


@pytest.mark.parametrize("breakage", ["no_file_named", "file_missing", "file_changed"])
def test_an_unrecoverable_specialist_pin_is_REFUSED_never_defaulted_to_the_pool(tmp_path, capsys, breakage):
    run, files = _specialist_run(tmp_path, ["Starmie @ Leftovers\n"], multi=False)
    meta = json.loads((run / "metadata.json").read_text())
    if breakage == "no_file_named":
        del meta["cli_args"]["trainee_team"]
        (run / "metadata.json").write_text(json.dumps(meta))
    elif breakage == "file_missing":
        Path(files[0]).unlink()
    else:
        Path(files[0]).write_text("Gengar @ Leftovers\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ETG.read_regime(run)
    assert exc.value.code == 2
    assert "RANDOM POOL TEAMS" in capsys.readouterr().out


def test_a_pre_matchupspec_pin_is_used_but_marked_unverified(tmp_path):
    run, _ = _specialist_run(tmp_path, ["Starmie @ Leftovers\n"], multi=False)
    meta = json.loads((run / "metadata.json").read_text())
    del meta["cli_args"]["_matchup_spec"]
    (run / "metadata.json").write_text(json.dumps(meta))
    reg = ETG.read_regime(run)
    assert reg["trainee_team_str"] == "Starmie @ Leftovers\n"
    assert reg["trainee_pin"]["verified"] is False
