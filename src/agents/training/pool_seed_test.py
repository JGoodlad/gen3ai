"""THE FORK POOL-SEED GUARD — the rule, the refusal, and the startup line they exist to preserve.

Every test here is written against the DEFECT rather than against the code: a fork that starts with
an empty ``snapshots/`` trains at ~0% self-play while its argv, its banner and every metric say it
is a self-play run. The tests that matter most are the two that a count-shaped check cannot see —
that a seeded pool reproduces the PARENT's ``self_play_fraction`` (the zips alone do not), and that
a non-empty pool is left byte-identical (the training session's hand-seeded arms must survive this
code arriving under them).
"""
from __future__ import annotations

import hashlib
import json
import os
import types

import pytest

from agents.training import pool_seed
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction


# --------------------------------------------------------------------------------------------
# fixtures — a synthetic parent run dir, and an args namespace
# --------------------------------------------------------------------------------------------
def _args(**kw):
    ns = types.SimpleNamespace(
        model=None, self_play=True, snapshot_dir=None,
        fork_pool_seed=True, allow_empty_pool=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _make_parent(root, *, n_snapshots=3, win_rate=0.901250, metadata=True) -> str:
    """A parent run dir shaped exactly like a real one: a checkpoint to fork from, a run-level
    ``metadata.json``/``model_config.json``, and a ``snapshots/`` pool with its metadata files."""
    run = os.path.join(str(root), "parent_run")
    pool = os.path.join(run, "snapshots")
    os.makedirs(pool, exist_ok=True)
    open(os.path.join(run, "final_model.zip"), "wb").write(b"parent-weights")
    if metadata:
        json.dump({"git_hash": "deadbeef"}, open(os.path.join(run, "metadata.json"), "w"))
        json.dump({"arch_signature": "sig"}, open(os.path.join(run, "model_config.json"), "w"))
    for i in range(n_snapshots):
        step = (i + 1) * 1_000_000
        open(os.path.join(pool, f"snapshot_{step:012d}.zip"), "wb").write(f"snap{i}".encode())
    if win_rate is not None:
        json.dump({"win_rate_vs_bots": win_rate, "self_play_fraction": 0.9,
                   "last_eval_step": 24_000_000, "seeded": True, "pool_generation": 7},
                  open(os.path.join(pool, "summary.json"), "w"))
        open(os.path.join(pool, "win_rate_vs_bots.txt"), "w").write(f"{win_rate:.6f}\n")
        json.dump({"arch_signature": "sig"}, open(os.path.join(pool, "model_config.json"), "w"))
    return run


def _fingerprint(d: str) -> dict:
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out[name] = hashlib.sha1(open(p, "rb").read()).hexdigest()
    return out


# --------------------------------------------------------------------------------------------
# THE AUDIT — the file set the pool's loader actually reads
# --------------------------------------------------------------------------------------------
class TestTheAuditedFileSet:
    def test_the_metadata_list_is_exactly_what_SnapshotPool_reads(self):
        """The two names are CLASS ATTRIBUTES of the pool, so a rename there breaks this test
        rather than silently un-copying the file that carries the ramp's starting point."""
        assert SnapshotPool._SUMMARY_FILE in pool_seed.POOL_METADATA_FILES
        assert SnapshotPool._WIN_RATE_FILE in pool_seed.POOL_METADATA_FILES
        # model_config.json is not read by SnapshotPool itself — load_model_snapshot looks for it
        # beside the zip and then one dir up — but it IS what makes the pool's arch check the
        # pool's own rather than the run root's.
        assert "model_config.json" in pool_seed.POOL_METADATA_FILES
        assert len(pool_seed.POOL_METADATA_FILES) == 3

    def test_the_snapshot_glob_is_the_pools_own(self):
        assert pool_seed.SNAPSHOT_GLOB == "snapshot_*.zip"


# --------------------------------------------------------------------------------------------
# THE RULE — decide() is pure, so every branch is testable without a run
# --------------------------------------------------------------------------------------------
class TestTheRule:
    def _decide(self, tmp_path, **kw):
        base = dict(self_play=True, model_path="/elsewhere/parent/final_model.zip",
                    model_dir=str(tmp_path / "fork"), pool_dir=str(tmp_path / "fork" / "snapshots"),
                    parent_run_dir=None, seed_enabled=True, allow_empty=False)
        base.update(kw)
        return pool_seed.decide(**base)

    def test_self_play_off_does_nothing(self, tmp_path):
        d = self._decide(tmp_path, self_play=False)
        assert not d.seed and not d.refuse

    def test_a_FRESH_run_never_seeds_and_NEVER_refuses(self, tmp_path):
        """A fresh run legitimately starts poolless and grows one once it clears the win-rate
        gate. Refusing it would break every fresh launch."""
        d = self._decide(tmp_path, model_path=None)
        assert not d.seed and not d.refuse and not d.is_fork
        assert "fresh" in d.reason

    def test_a_RESTART_never_seeds(self, tmp_path):
        """`--model` inside the run dir is a launcher restart. Re-seeding there would overwrite
        the run's own grown pool with the parent's stale one every few hours."""
        fork = tmp_path / "fork"
        d = self._decide(tmp_path, model_path=str(fork / "checkpoints" / "checkpoint_9_steps.zip"))
        assert not d.seed and not d.refuse
        assert "restart" in d.reason

    def test_a_restart_from_the_LEGACY_root_layout_is_still_a_restart(self, tmp_path):
        fork = tmp_path / "fork"
        d = self._decide(tmp_path, model_path=str(fork / "final_model.zip"))
        assert not d.seed and not d.refuse

    def test_a_fork_with_a_NON_EMPTY_pool_is_untouched(self, tmp_path):
        pool = tmp_path / "fork" / "snapshots"
        pool.mkdir(parents=True)
        (pool / "snapshot_000000001000.zip").write_bytes(b"x")
        d = self._decide(tmp_path)
        assert not d.seed and not d.refuse and d.is_fork
        assert "1 snapshot" in d.reason

    def test_a_fork_with_an_empty_pool_and_a_parent_pool_SEEDS(self, tmp_path):
        parent = _make_parent(tmp_path)
        d = self._decide(tmp_path, parent_run_dir=parent)
        assert d.seed and not d.refuse and d.is_fork

    def test_a_fork_whose_parent_has_no_pool_REFUSES(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        d = self._decide(tmp_path, parent_run_dir=parent)
        assert not d.seed and d.refuse
        assert "no pool" in d.reason

    def test_allow_empty_pool_turns_that_refusal_into_a_no_op(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        d = self._decide(tmp_path, parent_run_dir=parent, allow_empty=True)
        assert not d.seed and not d.refuse

    def test_no_fork_pool_seed_REFUSES_rather_than_silently_training_on_bots(self, tmp_path):
        parent = _make_parent(tmp_path)
        d = self._decide(tmp_path, parent_run_dir=parent, seed_enabled=False)
        assert not d.seed and d.refuse
        assert "--no-fork-pool-seed" in d.reason

    def test_no_fork_pool_seed_plus_allow_empty_pool_is_the_explicit_opt_out(self, tmp_path):
        parent = _make_parent(tmp_path)
        d = self._decide(tmp_path, parent_run_dir=parent, seed_enabled=False, allow_empty=True)
        assert not d.seed and not d.refuse

    def test_the_refusal_message_names_all_THREE_ways_out(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        msg = pool_seed.refusal_message(self._decide(tmp_path, parent_run_dir=parent))
        assert "--allow-empty-pool" in msg
        assert "--self-play" in msg
        assert "cp " in msg, "the manual path must be spelled out, not merely alluded to"
        assert "summary.json" in msg, "the zips-alone half-fix is the trap; name the metadata"


# --------------------------------------------------------------------------------------------
# THE COPY — and the startup line it exists to reproduce
# --------------------------------------------------------------------------------------------
class TestTheCopy:
    def test_every_zip_and_every_metadata_file_lands(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=3)
        pool = str(tmp_path / "fork" / "snapshots")
        rec = pool_seed.seed_pool(parent, pool)
        assert rec["n_snapshots"] == 3
        assert sorted(rec["files"]) == sorted(pool_seed.POOL_METADATA_FILES)
        for name in pool_seed.POOL_METADATA_FILES:
            assert os.path.isfile(os.path.join(pool, name))
        assert len(pool_seed.snapshot_zips(pool)) == 3

    def test_the_copies_are_byte_identical(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=2)
        pool = str(tmp_path / "fork" / "snapshots")
        pool_seed.seed_pool(parent, pool)
        src = _fingerprint(os.path.join(parent, "snapshots"))
        dst = _fingerprint(pool)
        for name, digest in src.items():
            assert dst[name] == digest, name

    def test_a_parent_with_zips_but_NO_metadata_seeds_and_says_the_win_rate_is_unknown(self, tmp_path):
        """The half-fix, made visible rather than silent: this is the pool that reads 0%."""
        parent = _make_parent(tmp_path, n_snapshots=2, win_rate=None)
        rec = pool_seed.seed_pool(parent, str(tmp_path / "fork" / "snapshots"))
        assert rec["n_snapshots"] == 2
        assert rec["files"] == []
        assert rec["win_rate_vs_bots"] is None

    def test_THE_SEEDED_POOL_REPRODUCES_THE_PARENTS_STARTUP_LINE(self, tmp_path):
        """🚨 The claim the whole feature is for. The zips carry the pool SIZE; the METADATA
        carries the starting `self_play_fraction`, and the 2026-09-02 manual fix needed both."""
        parent = _make_parent(tmp_path, n_snapshots=14, win_rate=0.901250)
        pool_dir = str(tmp_path / "fork" / "snapshots")
        pool_seed.seed_pool(parent, pool_dir)

        def line(d):
            pool = SnapshotPool(pool_dir=d, current_version=object())
            wr = pool.load_persisted_win_rate()
            frac = 1.0 - heuristic_fraction(wr)
            return (f"Pool has {len(pool)} snapshots, win_rate_vs_bots={wr:.2%} "
                    f"→ self_play_fraction={frac:.0%}")

        assert line(pool_dir) == line(os.path.join(parent, "snapshots"))
        assert line(pool_dir) == ("Pool has 14 snapshots, win_rate_vs_bots=90.12% "
                                  "→ self_play_fraction=90%")

    def test_the_ZIPS_ALONE_do_NOT_reproduce_it(self, tmp_path):
        """The negative control for the test above — this is exactly what the first attempt at
        the manual fix produced, and it read `self_play_fraction=0%` with 14 snapshots."""
        parent = _make_parent(tmp_path, n_snapshots=14, win_rate=0.901250)
        pool_dir = str(tmp_path / "fork" / "snapshots")
        os.makedirs(pool_dir)
        for z in pool_seed.snapshot_zips(os.path.join(parent, "snapshots")):
            open(os.path.join(pool_dir, os.path.basename(z)), "wb").write(open(z, "rb").read())
        pool = SnapshotPool(pool_dir=pool_dir, current_version=object())
        assert len(pool) == 14
        assert pool.load_persisted_win_rate() == 0.0
        assert 1.0 - heuristic_fraction(0.0) == 0.0

    def test_the_record_is_written_into_the_pool_and_reads_back(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=2)
        pool = str(tmp_path / "fork" / "snapshots")
        rec = pool_seed.seed_pool(parent, pool)
        assert pool_seed.read_seed_record(pool) == rec
        assert rec["parent_run_name"] == "parent_run"
        assert rec["schema"] == pool_seed.SEED_RECORD_SCHEMA

    def test_read_seed_record_is_total(self, tmp_path):
        assert pool_seed.read_seed_record(str(tmp_path)) is None
        os.makedirs(tmp_path / "p")
        (tmp_path / "p" / pool_seed.SEED_RECORD_FILE).write_text("{not json")
        assert pool_seed.read_seed_record(str(tmp_path / "p")) is None


# --------------------------------------------------------------------------------------------
# THE ENTRY POINT — prepare_pool, end to end over a real directory
# --------------------------------------------------------------------------------------------
class _Refused(Exception):
    pass


def _exit_fn(code):
    raise _Refused(code)


class TestPreparePool:
    def test_a_genuine_fork_is_seeded_and_says_so(self, tmp_path, capsys):
        parent = _make_parent(tmp_path, n_snapshots=14, win_rate=0.901250)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        lines = []
        rec = pool_seed.prepare_pool(
            _args(model=os.path.join(parent, "final_model.zip")), fork,
            emit_fn=lines.append, exit_fn=_exit_fn)
        assert rec is not None and rec["n_snapshots"] == 14
        assert len(lines) == 1
        assert "seeded 14 snapshots + metadata from parent_run" in lines[0]
        assert "win_rate_vs_bots=90.12%" in lines[0]

    def test_it_is_IDEMPOTENT_a_hand_seeded_pool_is_left_BYTE_IDENTICAL(self, tmp_path):
        """The training session hand-seeded the running dose arms. When a later arm syncs main,
        this code must not touch what is already there."""
        parent = _make_parent(tmp_path, n_snapshots=3)
        fork = str(tmp_path / "fork")
        pool = os.path.join(fork, "snapshots")
        os.makedirs(pool)
        open(os.path.join(pool, "snapshot_000000042000.zip"), "wb").write(b"hand-seeded")
        open(os.path.join(pool, "summary.json"), "w").write('{"win_rate_vs_bots": 0.5}')
        before = _fingerprint(pool)
        lines = []
        rec = pool_seed.prepare_pool(
            _args(model=os.path.join(parent, "final_model.zip")), fork,
            emit_fn=lines.append, exit_fn=_exit_fn)
        assert rec is None and lines == []
        assert _fingerprint(pool) == before

    def test_a_RESTART_never_seeds(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=3)
        fork = str(tmp_path / "fork")
        os.makedirs(os.path.join(fork, "checkpoints"))
        ckpt = os.path.join(fork, "checkpoints", "checkpoint_1000_steps.zip")
        open(ckpt, "wb").write(b"own")
        lines = []
        assert pool_seed.prepare_pool(_args(model=ckpt), fork,
                                      emit_fn=lines.append, exit_fn=_exit_fn) is None
        assert lines == []
        assert not os.path.isdir(os.path.join(fork, "snapshots"))
        assert parent  # the parent existed and was still not consulted

    def test_a_FRESH_run_never_seeds_and_never_refuses(self, tmp_path):
        fork = str(tmp_path / "fresh")
        os.makedirs(fork)
        lines = []
        assert pool_seed.prepare_pool(_args(model=None), fork,
                                      emit_fn=lines.append, exit_fn=_exit_fn) is None
        assert lines == []

    def test_a_poolless_fork_REFUSES_with_FATAL_CONFIG(self, tmp_path, capsys):
        from main.exit_codes import TrainExitCode
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        with pytest.raises(_Refused) as exc:
            pool_seed.prepare_pool(_args(model=os.path.join(parent, "final_model.zip")), fork,
                                   emit_fn=lambda _m: None, exit_fn=_exit_fn)
        assert exc.value.args[0] == int(TrainExitCode.FATAL_CONFIG)
        err = capsys.readouterr().err
        assert "--allow-empty-pool" in err and "FATAL" in err

    def test_allow_empty_pool_lets_the_poolless_fork_through(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        assert pool_seed.prepare_pool(
            _args(model=os.path.join(parent, "final_model.zip"), allow_empty_pool=True),
            fork, emit_fn=lambda _m: None, exit_fn=_exit_fn) is None

    def test_no_fork_pool_seed_refuses_even_though_the_parent_HAS_a_pool(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=5)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        with pytest.raises(_Refused):
            pool_seed.prepare_pool(
                _args(model=os.path.join(parent, "final_model.zip"), fork_pool_seed=False),
                fork, emit_fn=lambda _m: None, exit_fn=_exit_fn)
        assert not pool_seed.snapshot_zips(os.path.join(fork, "snapshots"))

    def test_self_play_OFF_does_nothing_at_all(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=5)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        lines = []
        assert pool_seed.prepare_pool(
            _args(model=os.path.join(parent, "final_model.zip"), self_play=False),
            fork, emit_fn=lines.append, exit_fn=_exit_fn) is None
        assert lines == []
        assert not os.path.isdir(os.path.join(fork, "snapshots"))

    def test_an_explicit_snapshot_dir_is_honoured(self, tmp_path):
        parent = _make_parent(tmp_path, n_snapshots=2)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        elsewhere = str(tmp_path / "pool_elsewhere")
        pool_seed.prepare_pool(
            _args(model=os.path.join(parent, "final_model.zip"), snapshot_dir=elsewhere),
            fork, emit_fn=lambda _m: None, exit_fn=_exit_fn)
        assert len(pool_seed.snapshot_zips(elsewhere)) == 2
        assert not os.path.isdir(os.path.join(fork, "snapshots"))

    def test_pool_dir_for_matches_the_trainers_own_resolution(self, tmp_path):
        assert pool_seed.pool_dir_for(_args(), "/runs/x") == os.path.join("/runs/x", "snapshots")
        assert pool_seed.pool_dir_for(_args(snapshot_dir="/p"), "/runs/x") == "/p"


# --------------------------------------------------------------------------------------------
# THE PROVENANCE RECORD — lineage.pool_seeded_from
# --------------------------------------------------------------------------------------------
class TestTheLineageRecord:
    def test_the_seed_record_rides_into_the_lineage_block(self, tmp_path):
        from main.train.run_io import _run_lineage
        parent = _make_parent(tmp_path, n_snapshots=4)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        model = os.path.join(parent, "final_model.zip")
        rec = pool_seed.prepare_pool(_args(model=model), fork,
                                     emit_fn=lambda _m: None, exit_fn=_exit_fn)
        block = _run_lineage(_args(model=model, exploiter=None, distill_teacher=None),
                             fork, model_path=model, fork_step=1234)
        assert block is not None
        assert block["pool_seeded_from"] == rec
        assert block["pool_seeded_from"]["n_snapshots"] == 4
        assert block["fork_parent"] is not None, "it is a SIBLING key, not an edit to fork_parent"

    def test_an_unseeded_fork_records_no_such_key(self, tmp_path):
        from main.train.run_io import _run_lineage
        parent = _make_parent(tmp_path, n_snapshots=0, win_rate=None)
        fork = str(tmp_path / "fork")
        os.makedirs(fork)
        model = os.path.join(parent, "final_model.zip")
        block = _run_lineage(_args(model=model, exploiter=None, distill_teacher=None),
                             fork, model_path=model, fork_step=0)
        assert block is not None and "pool_seeded_from" not in block

    def test_a_restart_contributes_no_block_at_all(self, tmp_path):
        from main.train.run_io import _run_lineage
        fork = str(tmp_path / "fork")
        os.makedirs(os.path.join(fork, "checkpoints"))
        ckpt = os.path.join(fork, "checkpoints", "checkpoint_9_steps.zip")
        open(ckpt, "wb").write(b"own")
        assert _run_lineage(_args(model=ckpt, exploiter=None, distill_teacher=None),
                            fork, model_path=ckpt, fork_step=9) is None


# --------------------------------------------------------------------------------------------
# THE PREDICATE — imported, never re-derived
# --------------------------------------------------------------------------------------------
def test_the_fork_predicate_is_the_ONE_in_fork_lr():
    """A second predicate for the same question is a second answer waiting to disagree."""
    import inspect
    src = inspect.getsource(pool_seed.decide)
    assert "from main.train.fork_lr import is_same_run_checkpoint" in src
    assert "startswith" not in src, "no hand-rolled path arithmetic in the decision"
