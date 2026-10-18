"""THE FORK POOL SEED'S STEP FILTER (T54, `gen3_pool_seed_step_filter_v1`, finding F18).

The defect: a parent that is still TRAINING when a fork is cut from one of its checkpoints holds
snapshots from AFTER the fork point, and copying its whole pool makes the fork train against its
parent's FUTURE selves. The boundary is ``snapshot step <= the fork checkpoint's num_timesteps``,
exact integers (rule 8): a snapshot AT the fork step is kept, one a single step later is dropped.

Every directory here is a tmp dir; no test reads a real run.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
import zipfile

import pytest

from agents.training import pool_seed

STEPS = (1_000_000, 2_000_000, 3_000_000, 4_000_000)


def _args(**kw):
    ns = types.SimpleNamespace(model=None, self_play=True, fork_pool_seed=True,
                               allow_empty_pool=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _sb3_zip(path: str, num_timesteps: int) -> str:
    """A checkpoint shaped as SB3 writes it: a plain-JSON ``data`` member carrying num_timesteps."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": num_timesteps}))
    return path


def _make_parent(root, *, steps=STEPS, pfsp=True) -> str:
    run = os.path.join(str(root), "parent_run")
    pool = os.path.join(run, "snapshots")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    os.makedirs(pool, exist_ok=True)
    json.dump({"git_hash": "deadbeef"}, open(os.path.join(run, "metadata.json"), "w"))
    json.dump({"arch_signature": "sig"}, open(os.path.join(run, "model_config.json"), "w"))
    for s in steps:
        open(os.path.join(pool, f"snapshot_{s:012d}.zip"), "wb").write(f"snap{s}".encode())
    summary = {"win_rate_vs_bots": 0.9, "self_play_fraction": 0.9, "last_eval_step": steps[-1],
               "seeded": True, "pool_generation": 7}
    if pfsp:
        summary["pfsp_win_rates"] = {str(s): round(0.4 + i * 0.01, 4) for i, s in enumerate(steps)}
    # Deliberately NOT json.dumps(indent=2): a byte-for-byte copy must not be a re-serialisation.
    open(os.path.join(pool, "summary.json"), "w").write(json.dumps(summary))
    open(os.path.join(pool, "win_rate_vs_bots.txt"), "w").write("0.900000\n")
    json.dump({"arch_signature": "sig"}, open(os.path.join(pool, "model_config.json"), "w"))
    return run


def _ckpt(parent: str, step: int) -> str:
    return _sb3_zip(os.path.join(parent, "checkpoints", f"checkpoint_{step}_steps.zip"), step)


def _fingerprint(d: str, *, skip=(pool_seed.SEED_RECORD_FILE,)) -> dict:
    return {n: hashlib.sha1(open(os.path.join(d, n), "rb").read()).hexdigest()
            for n in sorted(os.listdir(d)) if n not in skip and os.path.isfile(os.path.join(d, n))}


def _steps_on_disk(pool: str) -> list:
    return [pool_seed.snapshot_step(z) for z in pool_seed.snapshot_zips(pool)]


class _Refused(Exception):
    pass


def _exit_fn(code):
    raise _Refused(code)


def _prepare(parent: str, fork_root, model: str, **kw):
    fork = str(fork_root)
    os.makedirs(fork, exist_ok=True)
    lines: list = []
    rec = pool_seed.prepare_pool(_args(model=model, **kw), fork,
                                 emit_fn=lines.append, exit_fn=_exit_fn)
    return rec, lines, pool_seed.pool_dir_for(fork)


# --------------------------------------------------------------------------------------------
# THE DEFECT — fails on revert
# --------------------------------------------------------------------------------------------
def test_a_fork_of_a_LIVE_parent_never_seeds_its_FUTURE_selves(tmp_path):
    """🚨 F18. The parent has trained on to 4M; the fork is cut at 2M. Before T54 all four
    snapshots landed in the fork's pool."""
    parent = _make_parent(tmp_path)
    rec, lines, pool = _prepare(parent, tmp_path / "fork", _ckpt(parent, 2_000_000))
    assert _steps_on_disk(pool) == [1_000_000, 2_000_000]
    assert rec is not None
    assert rec["fork_step"] == 2_000_000
    assert rec["dropped_steps"] == [3_000_000, 4_000_000]
    assert pool_seed.read_seed_record(pool) == rec
    assert len(lines) == 1
    assert "step filter <= 2000000: DROPPED 2 later snapshot(s) at steps [3000000, 4000000]" in lines[0]


# --------------------------------------------------------------------------------------------
# THE BOUNDARY — exact integers, equality KEPT (rule 8)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fork_step,kept", [
    (2_000_000, [1_000_000, 2_000_000]),          # equality: the fork point's own weights, KEPT
    (1_999_999, [1_000_000]),                     # one step before: dropped
    (2_000_001, [1_000_000, 2_000_000]),
    (0, []),
])
def test_the_boundary_is_step_LE_fork_step(tmp_path, fork_step, kept):
    parent = _make_parent(tmp_path)
    pool = str(tmp_path / "fork" / "snapshots")
    rec = pool_seed.seed_pool(parent, pool, fork_step=fork_step)
    assert _steps_on_disk(pool) == kept
    assert rec["dropped_steps"] == [s for s in STEPS if s not in kept]
    assert rec["n_snapshots"] == len(kept)


# --------------------------------------------------------------------------------------------
# THE PER-SNAPSHOT STATE — pfsp_win_rates pruned to the same boundary
# --------------------------------------------------------------------------------------------
def test_pfsp_win_rates_are_pruned_to_the_kept_steps_and_nothing_else_changes(tmp_path):
    parent = _make_parent(tmp_path)
    src_summary = json.load(open(os.path.join(parent, "snapshots", "summary.json")))
    src_summary["pfsp_win_rates"]["not_a_step"] = 0.5      # the callback's reader skips it; so do we
    open(os.path.join(parent, "snapshots", "summary.json"), "w").write(json.dumps(src_summary))
    pool = str(tmp_path / "fork" / "snapshots")
    rec = pool_seed.seed_pool(parent, pool, fork_step=2_000_000)
    got = json.load(open(os.path.join(pool, "summary.json")))
    assert got["pfsp_win_rates"] == {"1000000": 0.4, "2000000": 0.41, "not_a_step": 0.5}
    for k, v in src_summary.items():
        if k != "pfsp_win_rates":
            assert got[k] == v, k
    assert rec["pruned_summary_keys"] == ["pfsp_win_rates"]
    # The other metadata files are byte-for-byte copies.
    for name in ("win_rate_vs_bots.txt", "model_config.json"):
        assert (open(os.path.join(pool, name), "rb").read()
                == open(os.path.join(parent, "snapshots", name), "rb").read())


def test_the_pruned_map_is_what_the_callback_reads_back(tmp_path):
    """`SelfPlayCallback.__init__` reads `load_summary()["pfsp_win_rates"]`; the pruned file must
    still be the pool's own summary."""
    from agents.training.snapshot_pool import SnapshotPool
    parent = _make_parent(tmp_path)
    pool = str(tmp_path / "fork" / "snapshots")
    pool_seed.seed_pool(parent, pool, fork_step=3_000_000)
    summary = SnapshotPool(pool_dir=pool, current_version=object()).load_summary()
    assert sorted(int(k) for k in summary["pfsp_win_rates"]) == [1_000_000, 2_000_000, 3_000_000]
    assert summary["win_rate_vs_bots"] == 0.9


def test_a_summary_with_no_later_pfsp_key_is_copied_byte_for_byte(tmp_path):
    """A dropped zip whose PFSP entry was already pruned upstream: no rewrite, a plain copy."""
    parent = _make_parent(tmp_path, pfsp=False)
    pool = str(tmp_path / "fork" / "snapshots")
    rec = pool_seed.seed_pool(parent, pool, fork_step=2_000_000)
    assert rec["dropped_steps"] == [3_000_000, 4_000_000]
    assert rec["pruned_summary_keys"] == []
    assert (open(os.path.join(pool, "summary.json"), "rb").read()
            == open(os.path.join(parent, "snapshots", "summary.json"), "rb").read())


# --------------------------------------------------------------------------------------------
# A FINISHED parent — byte-identical to the pre-filter behaviour
# --------------------------------------------------------------------------------------------
def test_a_fork_of_a_FINISHED_parent_is_byte_identical_to_the_unfiltered_copy(tmp_path):
    parent = _make_parent(tmp_path)
    final = _sb3_zip(os.path.join(parent, "final_model.zip"), 4_000_123)
    rec, lines, pool = _prepare(parent, tmp_path / "fork", final)
    assert rec is not None and rec["dropped_steps"] == [] and rec["fork_step"] == 4_000_123
    assert rec["pruned_summary_keys"] == []
    # Every file the pool's loader reads == the parent's, byte for byte ...
    assert _fingerprint(pool) == _fingerprint(os.path.join(parent, "snapshots"))
    # ... and == what the unfiltered copy (fork_step unknown, i.e. pre-T54) writes.
    legacy = str(tmp_path / "legacy" / "snapshots")
    pool_seed.seed_pool(parent, legacy, fork_step=None)
    assert _fingerprint(pool) == _fingerprint(legacy)
    assert "step filter <= 4000123: none later" in lines[0]


# --------------------------------------------------------------------------------------------
# the edges: an unreadable step, an all-later pool
# --------------------------------------------------------------------------------------------
def test_an_UNREADABLE_fork_step_copies_all_and_SAYS_SO(tmp_path):
    parent = _make_parent(tmp_path)
    junk = os.path.join(parent, "final_model.zip")
    open(junk, "wb").write(b"not a zip")            # no `data`, no `<N>_steps` in the name
    rec, lines, pool = _prepare(parent, tmp_path / "fork", junk)
    assert rec is not None and rec["fork_step"] is None and rec["dropped_steps"] == []
    assert _steps_on_disk(pool) == list(STEPS)
    assert "step filter OFF" in lines[0] and "UNREADABLE" in lines[0]


def test_the_fork_step_falls_back_to_the_checkpoint_FILENAME(tmp_path):
    p = tmp_path / "checkpoint_2500000_steps.zip"
    p.write_bytes(b"groomed")
    assert pool_seed.fork_step_of(str(p)) == 2_500_000
    assert pool_seed.fork_step_of(_sb3_zip(str(tmp_path / "x.zip"), 77)) == 77
    assert pool_seed.fork_step_of(None) is None


def test_a_fork_EARLIER_than_every_parent_snapshot_REFUSES(tmp_path):
    """The window slid past the fork point: nothing is left to seed, which is the poolless fork
    the module exists to refuse — not a silent bot fallback."""
    parent = _make_parent(tmp_path)
    with pytest.raises(_Refused) as exc:
        _prepare(parent, tmp_path / "fork", _ckpt(parent, 999_999))
    from main.exit_codes import TrainExitCode
    assert exc.value.args[0] == int(TrainExitCode.FATAL_CONFIG)


def test_allow_empty_pool_lets_the_all_later_fork_through(tmp_path):
    parent = _make_parent(tmp_path)
    rec, _lines, pool = _prepare(parent, tmp_path / "fork", _ckpt(parent, 999_999),
                                 allow_empty_pool=True)
    assert rec is not None and rec["n_snapshots"] == 0
    assert pool_seed.is_pool_empty(pool)


def test_snapshot_step_parses_like_SnapshotPool_scan():
    assert pool_seed.snapshot_step("/x/snapshot_000002000000.zip") == 2_000_000
    assert pool_seed.snapshot_step("snapshot_badname.zip") is None
