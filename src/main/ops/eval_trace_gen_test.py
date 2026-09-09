"""Tests for the OFFLINE eval-cycle generator and the read-side provenance gate.

Two tiers, deliberately. The PURE half (the provenance vocabulary, the refusals, the seed
derivation, the read-root spellings) is unmarked and runs in every gate — it is where the rules
that stop a wrong number from being reported live, and a rule that only runs in the slow tier is a
rule that rides main RED. The one test that actually PLAYS battles is marked ``slow`` + ``sim``,
takes a real saved checkpoint out of the run archive, and SKIPS when there is none (a linked
worktree, or a box that has never trained).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pytest

from main.ops import eval_trace_gen as ETG
from utils.paths import main_models_dir


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
    }
    m.update(over)
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

def _fake_run(tmp_path: Path, *, regime=True, snapshots=(4_000_000, 6_000_000, 8_000_000)):
    run = tmp_path / "fake_run"
    (run / "snapshots").mkdir(parents=True)
    cfg = {"arch_signature": "x", "config_version": 1}
    if regime:
        cfg["eval_sentinel_greedy"] = True
    (run / "model_config.json").write_text(json.dumps(cfg))
    (run / "metadata.json").write_text(json.dumps({"git_hash": "deadbeef"}))
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

def test_the_unit_seed_does_not_depend_on_the_worker():
    """🚨 Work-stealing decides WHICH worker plays a shard, and that is a race. Keying the dice
    off the worker would make a seeded cycle depend on the worker count, which is the one thing a
    seed exists to remove."""
    from main.eval_worker import unit_seed
    a = unit_seed(20260909, "sentinel_0", 3)
    assert a == unit_seed(20260909, "sentinel_0", 3)
    assert a != unit_seed(20260909, "sentinel_0", 4)
    assert a != unit_seed(20260909, "sentinel_1", 3)
    assert a != unit_seed(20260910, "sentinel_0", 3)
    assert 0 <= a < (1 << 62)


def test_per_battle_sim_seeds_are_varied_and_reproducible():
    """A single fixed `seed` runs N copies of ONE battle, which is not a sample of N. `seed_base`
    keeps the dice varied within a call and identical across calls."""
    from utils.bridge.local_battle_runner import _LocalBattleRunner
    r = _LocalBattleRunner.__new__(_LocalBattleRunner)
    r.seed, r.seed_base = None, 4242
    seeds = [r._seed_for(i) for i in range(20)]
    assert len({tuple(s) for s in seeds}) == 20, "every battle gets its own dice"
    assert seeds == [r._seed_for(i) for i in range(20)], "and the same ones next time"
    assert all(len(s) == 4 and all(0 <= w <= 0xFFFF for w in s) for s in seeds), \
        "the [m,n,o,p] form utils.bridge.seed_spec validates"

    # Adjacent seed_base values must not give adjacent streams — an LCG seeded with n and n+1 is
    # not two independent battles, and adjacent shards of one cycle differ by exactly that.
    r2 = _LocalBattleRunner.__new__(_LocalBattleRunner)
    r2.seed, r2.seed_base = None, 4243
    assert r2._seed_for(0) != seeds[0]

    r3 = _LocalBattleRunner.__new__(_LocalBattleRunner)
    r3.seed, r3.seed_base = [1, 2, 3, 4], None
    assert r3._seed_for(7) == [1, 2, 3, 4], "no seed_base → the fixed seed, unchanged"
    r4 = _LocalBattleRunner.__new__(_LocalBattleRunner)
    r4.seed, r4.seed_base = None, None
    assert r4._seed_for(7) is None, "neither → the child mints and reports its own"


def test_seed_and_seed_base_together_are_refused():
    """Two different dice regimes — one stream for every battle, or a derived stream per battle.
    Silently preferring one would make the label a lie."""
    from utils.bridge.local_battle_runner import _LocalBattleRunner
    with pytest.raises(ValueError, match="seed_base"):
        _LocalBattleRunner(object(), object(), "gen3ou", [1, 2, 3, 4], seed_base=9)
    # either alone is fine
    _LocalBattleRunner(object(), object(), "gen3ou", [1, 2, 3, 4])
    _LocalBattleRunner(object(), object(), "gen3ou", None, seed_base=9)


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


def test_read_root_accepts_both_spellings(tmp_path):
    """The shadow RUN dir and the eval_traces/step_<N> inside it are both natural things to
    paste, and both normalise to the run-shaped root cf_audit needs."""
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
    args = argparse.Namespace(states=1, anchors=1, rollouts=1, impl="rust", seed=0,
                              anchor_tolerance=0.9, bins=10, cond_boot=10, cond_ladder="off")
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


# --------------------------------------------------------------------------- the real thing

def _a_run_with_a_checkpoint():
    """A saved run with a cycle whose snapshot is on disk, a RECORDED eval regime, and at least
    one pool snapshot BELOW that step to serve as a sentinel.

    All three are needed and none can be assumed: a run may have been groomed of its eval
    snapshots, may predate the regime key, or — like the currently-LIVE arm, which is the newest
    thing in the archive and therefore the first candidate — may simply not have reached its
    second snapshot yet.
    """
    models = main_models_dir()
    if models is None:
        return None
    for snap in sorted(glob.glob(str(models / "*" / "eval_traces" / "step_*" / "snapshot.zip")),
                       reverse=True):
        run = Path(snap).parents[2]
        m = re.search(r"step_(\d+)", snap)
        if m is None:
            continue
        step = int(m.group(1))
        try:
            cfg = json.loads((run / "model_config.json").read_text())
        except (OSError, ValueError):
            continue
        if "eval_sentinel_greedy" not in cfg:
            continue
        pool = [int(mm.group(1)) for p in glob.glob(str(run / "snapshots" / "snapshot_*.zip"))
                if (mm := re.search(r"snapshot_(\d+)\.zip$", os.path.basename(p)))]
        if any(s < step for s in pool):
            return run, step
    return None


@pytest.mark.slow
@pytest.mark.sim
def test_a_generated_cycle_is_shaped_like_a_live_one(tmp_path):
    """The contract, on a real (tiny) cycle: the npz keys a live recorder writes, a manifest at
    selection_schema 2 with per-opponent capture rates so rule 17 holds, and the provenance block.

    Two games against two opponents — the smallest thing that still exercises the whole worker
    path, since the point of the tool is that it is NOT a re-implementation.
    """
    found = _a_run_with_a_checkpoint()
    if found is None:
        pytest.skip("no run archive with a snapshot + recorded eval regime (worktree, or a box "
                    "that has never trained)")
    run, step = found
    out = tmp_path / "cycle"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rc = ETG.main([f"{run}@{step}", "--games", "2", "--sentinels", "1",
                   "--opponents", "random,heuristic", "--out", str(out),
                   "--workers", "1", "--concurrency", "1", "--seed", "13",
                   "--shard-games", "2", "--nice", "15", "--force"])
    assert rc == 0

    step_dir = out / "eval_traces" / f"step_{step}"
    man = json.loads((step_dir / "eval_manifest.json").read_text())

    # --- the provenance block
    assert ETG.is_generated(man), "a generated cycle MUST be self-identifying"
    gen = man[ETG.GENERATED_KEY]
    assert gen["tool"] == ETG.GENERATOR_TAG
    assert gen["games_per_opponent"] == 2 and gen["sentinels_used"] == 1
    assert gen["capture"] == "ALL" and gen["reproducible"] is True
    assert gen["checkpoint_sha"] and gen["source_run"] == run.name
    assert gen["eval_sentinel_greedy"] == json.loads(
        (run / "model_config.json").read_text())["eval_sentinel_greedy"], \
        "the regime is READ from the run, never assumed"

    # --- rule 17: the selection block, with per-opponent capture rates
    assert man["selection_schema"] == 2
    per = man["selection"]["opponents"]
    assert set(per) == {"random", "heuristic", "sentinel_0"}
    for key, rec in per.items():
        assert rec["battles_played"] == 2, key
        for field in ("battles_won", "battles_drawn", "traces_written", "traces_won",
                      "traces_drawn"):
            assert field in rec, f"{key} is missing {field}"
        # FULL capture is the whole reason to generate a read cycle: every battle traced.
        assert rec["traces_written"] == rec["battles_played"], key
        decided = [rec[f"capture_rate_{k}"] for k in ("win", "loss", "draw")
                   if rec[f"capture_rate_{k}"] is not None]
        assert decided and all(abs(r - 1.0) < 1e-9 for r in decided), \
            f"{key}: full capture must read as a capture rate of 1.0, got {rec}"

    # --- the sentinel is PINNED, which a live cycle leaves empty
    assert gen["sentinel_steps"] and gen["sentinel_steps"][0] < step

    # --- the npz keys a live BattleRecorder writes (opp_true_team is NOT among them, same as live)
    import numpy as np
    npzs = sorted(glob.glob(str(step_dir / "*" / "*_states.npz")))
    assert len(npzs) == 6, f"2 games x 3 opponents, all traced; got {len(npzs)}"
    expected = {"obs", "logits", "values", "win_probs", "has_state", "actions", "action_mask",
                "move_logits", "spread_belief"}
    for path in npzs:
        with np.load(path) as d:
            keys = set(d.keys())
        assert expected <= keys, f"{path} missing {expected - keys}"
        assert "opp_true_team" not in keys, "not recorded live, and must not appear here"
        assert Path(path.replace("_states.npz", "_summary.json")).exists()

    # --- the checkpoint is where cf_audit looks for it, and the manifest points at it
    assert (step_dir / "snapshot.zip").exists()
    assert man["snapshot"] == "snapshot.zip"
    assert (out / "model_config.json").exists() and (out / "metadata.json").exists()
    assert (out / "snapshots").is_symlink(), "read-only reach back into models/, never a copy"

    # --- and nothing was written under models/
    assert not str(out.resolve()).startswith(str(run.resolve()))
