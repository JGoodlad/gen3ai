"""`--grow FIELD=VALUE` (B-GROW, `main.train.grow_fork`, `gen3_grow_fork_v1`, config v159): the launch side of a FORK's
declared, function-preserving growth. The model side (the identity rounds, the optimizer rule, the bit-identity gate)
is `agents/model/growth_test.py`; here, each surface the flag crosses, and each refusal by name:

* the parser takes only BUILT growable fields (``trunk_layers``), and names the build an unbuilt one needs;
* `resolve_config` sets the grown field to the CHILD's value after the parent's was inherited, so the child builds,
  checks and records the grown architecture; refused: no ``--model``, the field's own flag typed beside it, a shrink;
* `decide`: a genuine FORK is stamped; a same-run resume never grows; a fork that grows nothing is meaningless; a
  same-run RESTART of a grown fork (the launcher re-launches the original argv) is a no-op;
* the stamp reaches the lineage block (`run_io._run_lineage`) and `model_config.json` (`growth`, v159);
* `main.checkargs` and `main.launcher --dry-run` resolve the grow-fork and print the declared growth;
* END TO END on real checkpoints: the trainee loader (`load_model_snapshot(grow=…)` + the by-name optimizer realignment
  + `apply_growth`) and the SEEDED POOL (`grow_seeded_pool`): every grown file is its parent's function bit for bit, and
  reloads strictly at the child's version (the pool's opponents through `load_opponent_snapshot`).

Every assertion fails on the revert it names.
"""
from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from agents.model.model_version import ModelVersion, ModelVersionError
from main.train import grow_fork as GF
from main.train.derived_toggle_resume_test import _resolve, _saved_version
from main.train.parser import build_parser
from main.exit_codes import TrainExitCode
# the dry run's fixtures (a temp repo + archive) and F-BI-2's toy parent, requested by name (`getfixturevalue`)
from main.launcher.dry_run_test import isolated, repo  # noqa: F401
from main.launcher.fork_arch_dry_run_test import X34_FORK, toy_parent  # noqa: F401

_FRESH = ["--arch", "production", "--steps", "1000"]


def _version(trunk_layers: int) -> ModelVersion:
    return dataclasses.replace(_saved_version(_resolve(list(_FRESH))), trunk_layers=trunk_layers)


def _run(tmp_path, version, name="run"):
    from main.train.derived_toggle_resume_test import _run_dir
    from main.train.recipe_testkit import recorded_cli_args
    base = tmp_path / name
    base.mkdir(parents=True, exist_ok=True)
    return _run_dir(base, version, cli_args=recorded_cli_args())


# ── the parser ────────────────────────────────────────────────────────────────────────────────────
def test_the_parser_takes_only_built_growable_fields(capsys):
    p = build_parser()
    assert p.parse_args([]).grow is None
    assert p.parse_args(["--grow", "trunk_layers=4"]).grow == [("trunk_layers", 4)]
    assert p.parse_args(["--grow", "trunk-layers=3"]).grow == [("trunk_layers", 3)]
    assert p.parse_args(["--grow", "trunk_ffn_by_kind=on"]).grow == [("trunk_ffn_by_kind", "on")]
    for bad, why in (("trunk_layers", "FIELD=VALUE"), ("trunk_layers=9", "must be one of"),
                     ("trunk_ffn_by_kind=sideways", "must be one of"), ("opp_intent_grad_mode=shaping", "B-INT"),
                     ("damage_op=on", "not a growable field")):
        with pytest.raises(SystemExit):
            p.parse_args(["--grow", bad])
        assert why in capsys.readouterr().err, bad


# ── the namespace step (resolve_config) ───────────────────────────────────────────────────────────
def test_a_grow_fork_resolves_the_CHILD_value_and_records_it(tmp_path, capsys):
    """Without `apply_to_namespace` the child inherits the parent's depth (3) and builds the parent."""
    _run_d, ckpt = _run(tmp_path, _version(3))
    plain = _resolve(["--model", ckpt, "--run-name", "child_plain", "--steps", "2000"])
    assert plain.trunk_layers == 3 and plain._grow_declared == {}
    grown = _resolve(["--model", ckpt, "--run-name", "child_grow", "--steps", "2000", "--grow", "trunk_layers=4"])
    assert grown.trunk_layers == 4
    assert grown._grow_declared == {"trunk_layers": (3, 4)}
    assert _saved_version(grown).trunk_layers == 4                    # model_config.json records the grown depth
    assert "[GrowFork] declared growth: trunk_layers 3 -> 4" in capsys.readouterr().out


@pytest.mark.parametrize("extra, why", [
    (["--trunk-layers", "4"], "both typed"),                       # name the child's value ONCE
    (["--grow", "trunk_layers=4", "--grow", "trunk_layers=4"], "twice"),
])
def test_namespace_refusals(tmp_path, capsys, extra, why):
    _run_d, ckpt = _run(tmp_path, _version(3))
    argv = ["--model", ckpt, "--run-name", "child", "--steps", "2000", "--grow", "trunk_layers=4"]
    if extra[0] == "--grow":
        argv = argv[:-2]
    with pytest.raises(SystemExit) as e:
        _resolve(argv + extra)
    assert e.value.code == 2 and why in capsys.readouterr().err


def test_a_shrink_and_a_fresh_grow_are_refused(tmp_path, capsys):
    _run_d, ckpt = _run(tmp_path, _version(4))
    with pytest.raises(SystemExit):
        _resolve(["--model", ckpt, "--run-name", "child", "--steps", "2000", "--grow", "trunk_layers=3"])
    assert "only ADDS rounds" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _resolve(_FRESH + ["--grow", "trunk_layers=4"])
    assert "without --model is meaningless" in capsys.readouterr().err
    ns = types.SimpleNamespace(grow=[("trunk_layers", 4)], model="x.zip", _explicit_flags=frozenset())
    with pytest.raises(GF.GrowForkRefused, match="could not be read"):
        GF.apply_to_namespace(ns, None)


# ── the decision ──────────────────────────────────────────────────────────────────────────────────
def test_decide_stamps_a_fork_and_refuses_a_same_run_growth_and_a_meaningless_one(tmp_path):
    run_d, ckpt = _run(tmp_path, _version(3), name="parent")
    child = str(tmp_path / "child")
    stamp = GF.decide(grown={"trunk_layers": (3, 4)}, model_path=ckpt, model_dir=child, now="2026-10-19T00:00:00+00:00")
    assert stamp["schema"] == "gen3_grow_fork_v1" and stamp["date"].startswith("2026-10-19")
    assert stamp["fields"] == {"trunk_layers": {"parent": 3, "child": 4}}
    assert stamp["parent_model"] == ckpt and "no state" in stamp["optimizer"]
    assert "trunk_layers 3 -> 4" in GF.describe(stamp)
    assert GF.decide(grown={}, model_path=ckpt, model_dir=child) is None
    with pytest.raises(GF.GrowForkRefused, match="SAME-RUN resume never grows"):
        GF.decide(grown={"trunk_layers": (3, 4)}, model_path=ckpt, model_dir=run_d)
    with pytest.raises(GF.GrowForkRefused, match="grows nothing"):
        GF.decide(grown={"trunk_layers": (4, 4)}, model_path=ckpt, model_dir=child)


def test_a_restart_of_a_grown_fork_is_a_no_op_and_keeps_its_recorded_growth(tmp_path):
    """The launcher restarts a run with its ORIGINAL argv, which still says --grow: the run's own checkpoint already
    records the child depth, and its lineage records the growth — nothing grows again, model_config.json keeps it."""
    from main.launcher.checkpoint import resume_child_args

    _p, parent_ckpt = _run(tmp_path, _version(3), name="parent")
    stamp = GF.decide(grown={"trunk_layers": (3, 4)}, model_path=parent_ckpt, model_dir=str(tmp_path / "x"))
    run_d, ckpt = _run(tmp_path, dataclasses.replace(_version(4), growth=stamp), name="grown")
    meta = json.loads((tmp_path / "grown" / "run" / "metadata.json").read_text())
    meta["lineage"] = {"schema": "x", "role": "fork", "fork_parent": None, "grow": stamp}
    (tmp_path / "grown" / "run" / "metadata.json").write_text(json.dumps(meta))
    argv, _stripped = resume_child_args(
        ["--model", parent_ckpt, "--run-name", "grown", "--steps", "2000", "--grow", "trunk_layers=4"], ckpt, run_d)
    args = _resolve(argv)
    assert args.trunk_layers == 4 and args._grow_declared == {"trunk_layers": (4, 4)}
    assert GF.enforce(args, run_d) is None and args._grow_fork is None
    assert args._grow_record == stamp                                  # model_config.json keeps the run's growth
    # ...and a same-run resume whose lineage records NO growth: the flag is meaningless
    plain_d, plain_ckpt = _run(tmp_path, _version(4), name="plain")
    with pytest.raises(GF.GrowForkRefused, match="records no growth"):
        GF.decide(grown={"trunk_layers": (4, 4)}, model_path=plain_ckpt, model_dir=plain_d)


def test_the_stamp_reaches_the_lineage_block(tmp_path):
    from main.train.run_io import _run_lineage

    _p, ckpt = _run(tmp_path, _version(3), name="parent")
    child = tmp_path / "child"
    stamp = GF.decide(grown={"trunk_layers": (3, 4)}, model_path=ckpt, model_dir=str(child))
    args = types.SimpleNamespace(exploiter=None, _grow_fork=stamp)
    block = _run_lineage(args, str(child), model_path=ckpt, fork_step=1024)
    assert block["grow"] == stamp
    assert _run_lineage(args, str(child), model_path=str(child / "checkpoints" / "c.zip"), fork_step=0) is None
    # main.lineage shows it
    from main import lineage as LN
    child.mkdir()
    (child / "metadata.json").write_text(json.dumps({"lineage": block}))
    assert LN.read_run(str(child))["grow"] == stamp
    assert "GROWN AT ITS FORK: --grow trunk_layers 3 -> 4" in LN.render(LN.read_run(str(child)))


# ── checkargs and the dry run ─────────────────────────────────────────────────────────────────────
def test_checkargs_resolves_the_grow_fork_and_prints_the_declared_growth(tmp_path, capsys):
    from main.checkargs import check, main as checkargs_main

    _p, ckpt = _run(tmp_path, _version(3), name="parent")
    argv = ["--model", ckpt, "--run-name", "child_grow", "--steps", "2000", "--grow", "trunk_layers=4"]
    res = check(argv)
    assert res["ns"].trunk_layers == 4 and "trunk_layers" not in res["resolution"]["inherited"]
    assert checkargs_main(["--argv", " ".join(argv)]) == 0
    out = capsys.readouterr().out
    assert "declared growth (--grow)       : trunk_layers 3 -> 4" in out and "STAMPED" in out
    bad = ["--model", ckpt, "--run-name", "child_grow", "--steps", "2000", "--grow", "trunk_layers=3"]
    assert checkargs_main(["--argv", " ".join(bad)]) != 0
    assert "grows nothing" in capsys.readouterr().out


def test_the_launcher_dry_run_prints_the_declared_growth(request, monkeypatch, capsys):
    from main.launcher.dry_run_test import _dry_run, _make_run

    _root, (first, _second), work = request.getfixturevalue("isolated")
    _parent_dir, ckpt = _make_run(work, first, name="parent")
    (work / "models" / "parent" / "model_config.json").write_text(json.dumps(dataclasses.asdict(_version(2))))
    _dry_run(["--model", ckpt, "--run-name", "grow_x", "--steps", "1000", "--grow", "trunk_layers=4"], monkeypatch)
    out = capsys.readouterr().out
    assert "FORK of" in out and "grow        : DECLARED trunk_layers 2 -> 4" in out
    assert "a declared growth's fork-time gate" in out                  # the child-only half, named
    assert not (work / "models" / "grow_x").exists()
    _dry_run(["--model", ckpt, "--run-name", "grow_y", "--steps", "1000", "--grow", "trunk_layers=2"], monkeypatch,
             expect=int(TrainExitCode.FATAL_CONFIG))
    assert "✗ REFUSED (--grow)" in capsys.readouterr().out


# ── end to end on real checkpoints ────────────────────────────────────────────────────────────────
def _toy_checkpoint(run_dir, *, step: bool = True):
    """The toy parent (depth 2) after one AdamW step, SAVED as a real run (`final_model.zip` + `model_config.json`)."""
    from agents.model.growth_test import _parent, _toy
    from agents.model.snapshot import recorded_model_version, save_model_snapshot

    model, obs, mask = _parent() if step else (_toy(2), None, None)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "final_model"))
    version = recorded_model_version(model)
    save_model_snapshot(str(run_dir), version, git_hash="test")
    return model, version, run_dir / "final_model.zip"


@pytest.fixture(scope="module")
def _rows():
    from agents.model.growth_test import _rows as rows
    return rows()


def test_the_trainee_loader_grows_a_saved_fork_parent_and_the_child_reloads_strictly(tmp_path, _rows):
    """The model_build sequence: `load_model_snapshot(grow=…)` (the parent loads STRICTLY at depth 2), the by-name
    optimizer realignment, `apply_growth`; then a save + strict reload at the child's version, no growth declared."""
    from agents.model import growth as G
    from agents.model.snapshot import load_model_snapshot, save_model_snapshot
    from main.train.checkpoint_state import _validate_or_reset_optimizer_state
    from utils.torch_state_guard import torch_globals

    obs, mask = _rows
    fields = {"trunk_layers": {"parent": 2, "child": 4}}
    with torch_globals(num_threads=1):
        parent, version, zp = _toy_checkpoint(tmp_path / "parent")
        ref = G.forward_record(parent.policy, obs, mask)
        child_version = dataclasses.replace(version, trunk_layers=4)
        with pytest.raises(ModelVersionError, match="trunk_layers mismatch"):          # undeclared: refused
            load_model_snapshot(str(zp), env=None, current_version=child_version, device="cpu")
        model = load_model_snapshot(str(zp), env=None, current_version=child_version, device="cpu", grow=fields)
        assert G.trunk_layers_of(model.policy) == 2                                      # loaded AS the parent
        _validate_or_reset_optimizer_state(model, str(zp))
        report = G.apply_growth(model, fields, rows=(obs, mask))
        assert report["function_check"]["bit_identical"]
        assert report["optimizer_state"]["carried"] == len(parent.policy.optimizer.param_groups[0]["params"])
        model.policy.set_training_mode(False)
        assert np.array_equal(G.forward_record(model.policy, obs, mask)["logits"], ref["logits"])
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        model.save(str(child_dir / "final_model"))
        save_model_snapshot(str(child_dir), dataclasses.replace(child_version, growth={"fields": fields}),
                            git_hash="test")
        again = load_model_snapshot(str(child_dir / "final_model.zip"), env=None, current_version=child_version,
                                    device="cpu")
        again.policy.set_training_mode(False)
        got = G.forward_record(again.policy, obs, mask)
        saved = ModelVersion.from_json_file(str(child_dir / "model_config.json"))
    assert G.trunk_layers_of(again.policy) == 4
    assert np.array_equal(got["logits"], ref["logits"]) and np.array_equal(got["values"], ref["values"])
    # the optimizer state survived the round trip, the new rounds' included as "no state yet"
    assert len(again.policy.optimizer.state) == report["optimizer_state"]["carried"]
    assert saved.trunk_layers == 4 and saved.growth == {"fields": fields}


def test_the_seeded_pool_grows_to_the_childs_architecture_and_plays_the_parents_function(tmp_path, _rows):
    """A fork's pool is seeded from its parent's (PARENT-architecture snapshots); `grow_seeded_pool` grows each and
    rewrites the arch record. Without it, `load_opponent_snapshot` at the child's version refuses every snapshot."""
    import shutil

    from agents.model import growth as G
    from agents.model.snapshot import load_opponent_snapshot
    from agents.training.pool_seed import read_seed_record, write_seed_record
    from utils.torch_state_guard import torch_globals

    obs, mask = _rows
    fields = {"trunk_layers": {"parent": 2, "child": 4}}
    with torch_globals(num_threads=1):
        parent, version, zp = _toy_checkpoint(tmp_path / "parent")
        ref = G.forward_record(parent.policy, obs, mask)
        pool = tmp_path / "child" / "snapshots"
        pool.mkdir(parents=True)
        for step in (1000, 2000):
            shutil.copy2(zp, pool / f"snapshot_{step:012d}.zip")
        (pool / "model_config.json").write_text(version.to_json())
        write_seed_record(str(pool), {"schema": 2, "n_snapshots": 2})
        child_version = dataclasses.replace(version, trunk_layers=4)
        with pytest.raises(ModelVersionError, match="trunk_layers mismatch"):
            load_opponent_snapshot(str(pool / "snapshot_000000001000.zip"), child_version)
        out = GF.grow_seeded_pool(str(pool), fields, child_version)
        assert out["snapshots"] == ["snapshot_000000001000.zip", "snapshot_000000002000.zip"] and not out["skipped"]
        assert (pool / "model_config.json").read_text() == child_version.to_json()      # what _record_arch compares
        assert read_seed_record(str(pool))["grown"]["snapshots"] == out["snapshots"]
        assert sorted(p.name for p in pool.iterdir()) == [
            "model_config.json", "pool_seed.json", "snapshot_000000001000.zip", "snapshot_000000002000.zip"]
        for z in sorted(pool.glob("snapshot_*.zip")):
            opp = load_opponent_snapshot(str(z), child_version)
            assert G.trunk_layers_of(opp.policy) == 4
            opp.policy.set_training_mode(False)
            got = G.forward_record(opp.policy, obs, mask)
            assert np.array_equal(got["logits"], ref["logits"]) and np.array_equal(got["values"], ref["values"])
        assert GF.grow_seeded_pool(str(pool), fields, child_version) is None              # idempotent


# ── F-BI-2's fork-architecture verdict (`main.train.fork_arch`) honours the declared growth ──────────


def test_the_fork_arch_verdict_passes_a_declared_growth_and_judges_it_with_the_trainers_gate(request):
    """`fork_arch.check_against_saved` routes a `--grow` argv through `check_compatible_grown` (the load's gate): the
    declared depth passes; an UNDECLARED typed depth and any OTHER arch difference beside `--grow` stay refused."""
    from main.checkargs import check
    from main.train import fork_arch

    parent = request.getfixturevalue("toy_parent")
    assert fork_arch.DECLARED_GROWTH == {"grow": ("trunk_ffn_by_kind", "trunk_layers")}
    ok = check(["--model", parent, "--grow", "trunk_layers=4", *X34_FORK])["fork_arch"]
    assert ok.checked and not ok.refused, ok
    typed = check(["--model", parent, "--trunk-layers", "4", *X34_FORK])["fork_arch"]
    assert typed.refused and "trunk_layers mismatch" in typed.reason
    other = check(["--model", parent, "--grow", "trunk_layers=4", "--mon-hazard-cost", "on", *X34_FORK])["fork_arch"]
    assert other.refused and "mon_hazard_cost mismatch" in other.reason, other


# ── B-KFFN: `--grow trunk_ffn_by_kind=on` (design_growth_builds_kffn_int.md §1.8: K-10, K-12, K-13) ──────────────
def _kversion(kffn: str) -> ModelVersion:
    return dataclasses.replace(_saved_version(_resolve(list(_FRESH))), trunk_ffn_by_kind=kffn)


def test_kffn_a_grow_fork_resolves_on_and_a_restart_keeps_it(tmp_path, capsys):
    """K-12 (the launch side). The fork resolves, builds and records ``on`` (the CLI tier: a config_only field would
    BUILD its frozen default, F-GB-1); its stamp records Q-K3's copy rule; a flagless restart of the grown run
    inherits ``on`` from its own model_config.json (the derived_toggle_resume round trip)."""
    _d, ckpt = _run(tmp_path, _kversion("off"), name="parent")
    plain = _resolve(["--model", ckpt, "--run-name", "child_plain", "--steps", "2000"])
    assert plain.trunk_ffn_by_kind == "off"
    grown = _resolve(["--model", ckpt, "--run-name", "child", "--steps", "2000", "--grow", "trunk_ffn_by_kind=on"])
    assert grown.trunk_ffn_by_kind == "on" and grown._grow_declared == {"trunk_ffn_by_kind": ("off", "on")}
    assert _saved_version(grown).trunk_ffn_by_kind == "on"
    assert "[GrowFork] declared growth: trunk_ffn_by_kind 'off' -> 'on'" in capsys.readouterr().out
    stamp = GF.decide(grown={"trunk_ffn_by_kind": ("off", "on")}, model_path=ckpt, model_dir=str(tmp_path / "child"))
    assert stamp["fields"] == {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
    assert "trunk_ffn_by_kind: each new expert parameter is a COPY" in stamp["optimizer"]
    run_d, own = _run(tmp_path, _kversion("on"), name="grown")
    again = _resolve(["--model", own, "--steps", "3000"])
    assert again.trunk_ffn_by_kind == "on" and _saved_version(again).trunk_ffn_by_kind == "on"


def test_kffn_refusals_on_the_launch_side(tmp_path, capsys):
    """K-10 (the launch side). The typed flag beside ``--grow``; ``on -> off``; a fork that TYPES ``--trunk-ffn-by-kind
    on`` without ``--grow``: `checkargs` refuses it (the F-BI-2 fork-arch verdict) and the text names the --grow."""
    from main.checkargs import check, main as checkargs_main

    _d, ckpt = _run(tmp_path, _kversion("off"), name="parent")
    base = ["--model", ckpt, "--run-name", "child", "--steps", "2000"]
    with pytest.raises(SystemExit) as e:
        _resolve(base + ["--grow", "trunk_ffn_by_kind=on", "--trunk-ffn-by-kind", "on"])
    assert e.value.code == 2 and "both typed" in capsys.readouterr().err
    _d2, ckpt_on = _run(tmp_path, _kversion("on"), name="parent_on")
    with pytest.raises(SystemExit):
        _resolve(["--model", ckpt_on, "--run-name", "c2", "--steps", "2000", "--grow", "trunk_ffn_by_kind=off"])
    assert "not function-preserving" in capsys.readouterr().err
    typed = check(base + ["--trunk-ffn-by-kind", "on"])["fork_arch"]
    assert typed.refused and "trunk_ffn_by_kind mismatch" in typed.reason
    assert "--grow trunk_ffn_by_kind=on" in typed.reason
    assert checkargs_main(["--argv", " ".join(base + ["--trunk-ffn-by-kind", "on"])]) != 0
    ok = check(base + ["--grow", "trunk_ffn_by_kind=on"])["fork_arch"]
    assert ok.checked and not ok.refused, ok


def test_kffn_the_launcher_dry_run_refuses_a_typed_kffn_fork(request, monkeypatch, capsys):
    """K-10: the dry run judges the same verdict (F-BI-2's path): the typed flag on a fork is REFUSED, the declared
    growth would launch."""
    from main.launcher.dry_run_test import _dry_run, _make_run

    _root, (first, _second), work = request.getfixturevalue("isolated")
    _parent_dir, ckpt = _make_run(work, first, name="parent")
    (work / "models" / "parent" / "model_config.json").write_text(json.dumps(dataclasses.asdict(_kversion("off"))))
    _dry_run(["--model", ckpt, "--run-name", "k_typed", "--steps", "1000", "--trunk-ffn-by-kind", "on"], monkeypatch,
             expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert "trunk_ffn_by_kind mismatch" in out and "--grow trunk_ffn_by_kind=on" in out
    _dry_run(["--model", ckpt, "--run-name", "k_grow", "--steps", "1000", "--grow", "trunk_ffn_by_kind=on"],
             monkeypatch)
    assert "grow        : DECLARED trunk_ffn_by_kind 'off' -> 'on'" in capsys.readouterr().out


def _prod_checkpoint(run_dir):
    """A PRODUCTION parent (kffn off) after one AdamW step, saved as a real run (`final_model.zip` + config)."""
    from agents.model.growth_test import _step
    from agents.model.growth_test import _rows as rows
    from agents.model.snapshot import recorded_model_version, save_model_snapshot
    from main.fresh_checkpoint import build_fresh_model

    model, _a, _pk = build_fresh_model(0)
    obs, mask = rows()
    _step(model, obs, mask)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "final_model"))
    version = recorded_model_version(model)
    save_model_snapshot(str(run_dir), version, git_hash="test")
    return model, version, run_dir / "final_model.zip"


@pytest.mark.slow  # ship gate (2026-10-20): 29 s beside a live run (saves + reloads three production policies)
def test_kffn_the_seeded_pool_grows_its_experts_and_a_relaunch_mid_pool_is_idempotent(tmp_path, _rows):
    """K-13. `grow_seeded_pool` on a PRODUCTION pool, ``trunk_ffn_by_kind off -> on``: every snapshot loads at the
    child's version as its parent snapshot's function, the record is the child's; a relaunch that died MID-POOL (one
    snapshot grown, the record not yet rewritten) grows only the other and skips the grown one. Without the field-generic
    pool path (§3) the skip test read ``trunk_layers`` and raised KeyError."""
    import shutil

    from agents.model import growth as G
    from agents.model.snapshot import load_opponent_snapshot
    from agents.training.pool_seed import write_seed_record
    from utils.torch_state_guard import torch_globals

    obs, mask = _rows
    fields = {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}
    with torch_globals(num_threads=1):
        parent, version, zp = _prod_checkpoint(tmp_path / "parent")
        parent.policy.set_training_mode(False)
        ref = G.forward_record(parent.policy, obs, mask)
        pool = tmp_path / "child" / "snapshots"
        pool.mkdir(parents=True)
        names = [f"snapshot_{s:012d}.zip" for s in (1000, 2000)]
        for n in names:
            shutil.copy2(zp, pool / n)
        (pool / "model_config.json").write_text(version.to_json())
        write_seed_record(str(pool), {"schema": 2, "n_snapshots": 2})
        child_version = dataclasses.replace(version, trunk_ffn_by_kind="on")
        with pytest.raises(ModelVersionError, match="trunk_ffn_by_kind mismatch"):
            load_opponent_snapshot(str(pool / names[0]), child_version)
        out = GF.grow_seeded_pool(str(pool), fields, child_version)
        assert out["snapshots"] == names and not out["skipped"]
        assert (pool / "model_config.json").read_text() == child_version.to_json()
        for z in names:
            opp = load_opponent_snapshot(str(pool / z), child_version)
            assert opp.policy.features_extractor.trunk_ffn_by_kind == "on"
            opp.policy.set_training_mode(False)
            got = G.forward_record(opp.policy, obs, mask)
            assert np.array_equal(got["logits"], ref["logits"]) and np.array_equal(got["values"], ref["values"])
        assert GF.grow_seeded_pool(str(pool), fields, child_version) is None              # idempotent
        # a relaunch that died mid-pool: snapshot 2 back at the parent, the record still the parent's
        shutil.copy2(zp, pool / names[1])
        (pool / "model_config.json").write_text(version.to_json())
        out2 = GF.grow_seeded_pool(str(pool), fields, child_version)
        assert out2["snapshots"] == [names[1]] and out2["skipped"] == [names[0]]
        opp = load_opponent_snapshot(str(pool / names[1]), child_version)
        opp.policy.set_training_mode(False)
        assert np.array_equal(G.forward_record(opp.policy, obs, mask)["logits"], ref["logits"])
