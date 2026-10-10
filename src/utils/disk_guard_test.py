"""THE DISK GUARD (`utils.disk_guard`): the REQUIRED arithmetic, the preflight verdict, the in-run thresholds.

Every test here fails on revert of the part it names. The box's free space is never read: the root
conftest replaces `disk_guard.free_bytes`, and a test names the number it wants (`free_fn=` or its own
`monkeypatch`); the reader proper is `_read_free`.
"""
from __future__ import annotations

import errno
import os
import shutil
from types import SimpleNamespace

from utils import disk_guard as dg
from utils.disk_guard import GiB, MiB

CKPT = 60 * MiB


def _plan(tmp_path, **kw):
    base = dict(run_dir=str(tmp_path / "rb_new"), steps=15_000_000, resume_steps=0,
                checkpoint_interval=1_000_000, eval_freq=2_000_000, ckpt_bytes=CKPT, ckpt_source="test",
                fork=False, snapshots=True, compile_cache=True, existing_compile_cache_bytes=0)
    base.update(kw)
    return dg.RunPlan(**base)


def _terms(req):
    return {label: b for label, b, _how in req.terms}


# --- the arithmetic on a constructed run --------------------------------------------------------

def test_the_requirement_is_the_declared_sum_for_a_fresh_15M_run(tmp_path):
    a = dg.DEFAULT_ALLOWANCES
    req = dg.requirement(_plan(tmp_path))
    t = _terms(req)
    assert t["periodic checkpoints"] == 15 * CKPT                       # 15 boundaries, never pruned
    assert t["final/best/aborted"] == a.extra_checkpoint_copies * CKPT
    assert t["eval traces"] == 8 * a.eval_trace_bytes_per_cycle         # ceil(15M / 2M)
    assert t["pool snapshots"] == 8 * CKPT                              # one per cycle, under the cap
    assert t["compile cache"] == a.compile_cache_bytes
    assert t["sidecar + tb + logs"] == int(15.0 * (a.value_sidecar_bytes_per_mstep + a.log_bytes_per_mstep))
    assert t["fixed"] == a.fixed_bytes
    assert req.subtotal == sum(t.values())
    assert req.margin == int(req.subtotal * dg.MARGIN_FRACTION)
    assert req.required == req.subtotal + req.margin + dg.RESERVE_BYTES


def test_the_allowances_cover_the_measured_run_they_were_taken_from(tmp_path):
    """rb_st_legacy_s1006 (15M steps, 4.2 GB on disk, checkpoint 62.8 MB): the estimate must sit ABOVE what
    it actually wrote, and not absurdly far above (a guard that always refuses is switched off)."""
    req = dg.requirement(_plan(tmp_path, ckpt_bytes=62_816_554))
    measured_total = 4.2e9
    assert req.subtotal > measured_total
    assert req.required < 4 * measured_total


def test_a_resume_counts_only_the_boundaries_still_to_cross(tmp_path):
    p = _plan(tmp_path, resume_steps=9_400_000, steps=15_000_000)
    t = _terms(dg.requirement(p))
    assert t["periodic checkpoints"] == (15 - 9) * CKPT                 # 10M..15M: 6 boundaries
    assert t["eval traces"] == 3 * dg.DEFAULT_ALLOWANCES.eval_trace_bytes_per_cycle   # ceil(5.6M / 2M)


def test_the_compile_cache_already_on_disk_is_not_required_again(tmp_path):
    a = dg.DEFAULT_ALLOWANCES
    half = _terms(dg.requirement(_plan(tmp_path, existing_compile_cache_bytes=a.compile_cache_bytes // 2)))
    full = _terms(dg.requirement(_plan(tmp_path, existing_compile_cache_bytes=a.compile_cache_bytes * 2)))
    assert half["compile cache"] == a.compile_cache_bytes - a.compile_cache_bytes // 2
    assert full["compile cache"] == 0
    assert _terms(dg.requirement(_plan(tmp_path, compile_cache=False)))["compile cache"] == 0


def test_a_fork_adds_the_source_copy_and_reseeds_the_whole_pool_window(tmp_path):
    t = _terms(dg.requirement(_plan(tmp_path, fork=True)))
    assert t["final/best/aborted+fork"] == (dg.DEFAULT_ALLOWANCES.extra_checkpoint_copies + 1) * CKPT
    assert t["pool snapshots"] == dg.POOL_WINDOW * CKPT
    assert _terms(dg.requirement(_plan(tmp_path, snapshots=False)))["pool snapshots"] == 0


def test_the_pool_window_constant_is_the_pools():
    from agents.training.snapshot_pool import DEFAULT_MAX_SNAPSHOTS
    assert dg.POOL_WINDOW == DEFAULT_MAX_SNAPSHOTS


def test_the_snapshot_term_is_capped_by_the_window(tmp_path):
    t = _terms(dg.requirement(_plan(tmp_path, steps=500_000_000, eval_freq=2_000_000)))
    assert t["pool snapshots"] == dg.POOL_WINDOW * CKPT


def test_the_report_prints_the_arithmetic(tmp_path):
    v = dg.check_for_run(_plan(tmp_path), debug=False, allow=False, free_fn=lambda p: 1 * GiB)
    text = "\n".join(v.lines())
    for needle in ("periodic checkpoints", "15 x 60.0 MiB", "eval traces", "compile cache", "= REQUIRED"):
        assert needle in text


# --- the checkpoint size ------------------------------------------------------------------------

def _zip(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * n)


def test_the_size_comes_from_this_runs_own_newest_checkpoint_first(tmp_path):
    run = tmp_path / "run"
    _zip(str(run / "checkpoints" / "checkpoint_1_steps.zip"), 100)
    other = tmp_path / "archive" / "other"
    _zip(str(other / "checkpoints" / "checkpoint_1_steps.zip"), 999)
    n, src = dg.find_checkpoint_bytes(str(run), "/some/fork/model.zip", str(tmp_path / "archive"))
    assert n == 100 and "own newest checkpoint" in src


def test_a_fork_reads_its_source_model_then_the_archive(tmp_path):
    src_zip = tmp_path / "parent" / "checkpoints" / "checkpoint_7_steps.zip"
    _zip(str(src_zip), 321)
    n, src = dg.find_checkpoint_bytes(str(tmp_path / "new"), str(src_zip), None)
    assert n == 321 and "fork source" in src
    n2, _ = dg.find_checkpoint_bytes(str(tmp_path / "new"), str(tmp_path / "parent"), None)   # a bare run dir
    assert n2 == 321


def test_a_fresh_run_reads_the_same_architecture_run_in_the_archive(tmp_path, monkeypatch):
    arch = tmp_path / "archive"
    _zip(str(arch / "old_other" / "final_model.zip"), 900)
    _zip(str(arch / "mine" / "final_model.zip"), 400)
    (arch / "mine" / "model_config.json").write_text('{"arch_signature": "SIG-X"}')
    monkeypatch.setattr(dg, "_current_arch_signature", lambda: "SIG-X")
    n, src = dg.find_checkpoint_bytes(str(tmp_path / "new"), None, str(arch))
    assert n == 400 and "same ARCH_SIGNATURE" in src


def test_with_no_same_architecture_run_the_biggest_of_the_newest_few_stands_in(tmp_path, monkeypatch):
    arch = tmp_path / "archive"
    for i, n in enumerate([100, 700, 300]):
        z = arch / f"r{i}" / "final_model.zip"
        _zip(str(z), n)
        os.utime(z, (1000 + i, 1000 + i))
        os.utime(arch / f"r{i}", (1000 + i, 1000 + i))
    monkeypatch.setattr(dg, "_current_arch_signature", lambda: "NOBODY-HAS-THIS")
    n, src = dg.find_checkpoint_bytes(str(tmp_path / "new"), None, str(arch))
    assert n == 700 and "biggest" in src


def test_no_size_anywhere_is_said_out_loud_not_guessed(tmp_path):
    n, why = dg.find_checkpoint_bytes(str(tmp_path / "new"), None, str(tmp_path / "empty"))
    assert n is None and "no checkpoint" in why
    v = dg.check_for_run(_plan(tmp_path, ckpt_bytes=None, ckpt_source=why), debug=False, allow=False,
                         free_fn=lambda p: 1000 * GiB)
    assert v.status == dg.PASS and "UNKNOWN" in "\n".join(v.lines())


# --- the preflight verdict ----------------------------------------------------------------------

def test_the_preflight_refuses_below_the_bar_and_allows_above_it(tmp_path):
    plan = _plan(tmp_path)
    need = dg.requirement(plan).required
    short = dg.check_for_run(plan, debug=False, allow=False, free_fn=lambda p: need - 1)
    assert short.refused and short.status == dg.REFUSED_LOW
    assert "REFUSED" in short.lines()[0] and "SHORT by" in short.lines()[1]
    assert dg.OPT_OUT_FLAG in "\n".join(short.lines())
    ok = dg.check_for_run(plan, debug=False, allow=False, free_fn=lambda p: need)
    assert ok.status == dg.PASS and not ok.refused and ok.lines()[0].startswith("disk space  : ✓")


def test_the_verdict_reads_the_seam_the_conftest_pins(tmp_path, monkeypatch):
    """`check_for_run` reads `disk_guard.free_bytes` at call time (the stub reaches it)."""
    monkeypatch.setattr(dg, "free_bytes", lambda path: 1 * GiB)
    assert dg.check_for_run(_plan(tmp_path), debug=False, allow=False).refused
    monkeypatch.setattr(dg, "free_bytes", lambda path: 500 * GiB)
    assert not dg.check_for_run(_plan(tmp_path), debug=False, allow=False).refused


def test_the_opt_out_tolerates_and_records(tmp_path):
    v = dg.check_for_run(_plan(tmp_path), debug=False, allow=True, free_fn=lambda p: 1 * GiB)
    assert v.status == dg.OPTED_OUT and not v.refused
    assert "tolerated under --allow-low-disk" in v.lines()[0]
    rec = v.to_record()
    assert rec["status"] == "opted-out" and rec["allow_low_disk"] is True and rec["free_bytes"] == GiB


def test_debug_is_exempt_and_never_reads_the_disk(tmp_path):
    def boom(_p):
        raise AssertionError("--debug must not read the disk")
    v = dg.check_for_run(None, debug=True, allow=False, free_fn=boom)
    assert v.status == dg.EXEMPT_DEBUG and not v.refused
    assert dg.check_for_run(_plan(tmp_path), debug=True, allow=False, free_fn=boom).status == dg.EXEMPT_DEBUG


def test_the_measured_path_is_the_nearest_existing_ancestor(tmp_path):
    deep = tmp_path / "does" / "not" / "exist" / "yet"
    assert dg.probe_path(str(deep)) == str(tmp_path)
    v = dg.check_for_run(_plan(tmp_path, run_dir=str(deep)), debug=False, allow=False, free_fn=lambda p: 1)
    assert v.path == str(tmp_path)


def test_the_reader_proper_is_shutil_disk_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(total=9, used=4, free=5))
    assert dg._read_free(str(tmp_path / "nope")) == 5


# --- the in-run guard's two thresholds -----------------------------------------------------------

def _guard(free, **kw):
    return dg.InRunGuard("/", free_fn=lambda p: free, **kw)


def test_in_run_levels_are_ok_warn_stop_at_two_and_one_next_saves():
    s = 100 * MiB
    assert _guard(2 * s).after_save(s).level == dg.OK                      # exactly 2x: not below
    assert _guard(2 * s - 1).after_save(s).level == dg.WARN
    assert _guard(s).after_save(s).level == dg.WARN                        # exactly 1x: not below the stop bar
    d = _guard(s - 1).after_save(s)
    assert d.level == dg.STOP and "FATAL_DISK" in d.message and "stopping cleanly" in d.message
    w = _guard(int(1.5 * s)).after_save(s)
    assert w.level == dg.WARN and "LOW DISK" in w.message


def test_the_opt_out_keeps_the_warning_and_drops_the_stop():
    s = 100 * MiB
    d = _guard(s // 2, stop_enabled=False).after_save(s)
    assert d.level == dg.WARN and dg.OPT_OUT_FLAG in d.message


def test_the_in_run_guard_reads_the_disk_once_per_call(tmp_path):
    calls = []
    g = dg.InRunGuard(str(tmp_path), free_fn=lambda p: calls.append(p) or 10 * GiB)
    g.after_save(60 * MiB)
    assert calls == [str(tmp_path)]


def test_a_full_disk_write_error_is_recognised_through_its_wrapper():
    full = OSError(errno.ENOSPC, "No space left on device")
    assert dg.is_disk_full(full)
    try:
        try:
            raise full
        except OSError as inner:
            raise RuntimeError("torch.save wrapper") from inner
    except RuntimeError as outer:
        assert dg.is_disk_full(outer)
    assert not dg.is_disk_full(OSError(errno.EACCES, "denied"))
    assert not dg.is_disk_full(ValueError("x"))


# --- the SHARED compile root (gen3_shared_compile_cache_v1; the shared cache's FINDING 3) ---------

def test_the_shared_roots_growth_is_COUNTED_on_the_run_archives_filesystem(tmp_path):
    sc = dg.SharedCachePlan(root=str(tmp_path / "shared"), cap=20 * GiB, used=3 * GiB, same_fs=True)
    base = dg.requirement(_plan(tmp_path))
    req = dg.requirement(_plan(tmp_path, shared_cache=sc))
    assert _terms(req)["shared compile cache"] == 17 * GiB          # cap - in use
    assert req.subtotal == base.subtotal + 17 * GiB
    need = req.required
    assert dg.check_for_run(_plan(tmp_path, shared_cache=sc), debug=False, allow=False,
                            free_fn=lambda p: need - 1).refused


def test_on_ANOTHER_filesystem_it_is_checked_there_and_REPORTED_never_counted(tmp_path):
    sc = dg.SharedCachePlan(root=str(tmp_path / "elsewhere"), cap=20 * GiB, used=0, same_fs=False)
    plan = _plan(tmp_path, shared_cache=sc)
    assert _terms(dg.requirement(plan))["shared compile cache"] == 0
    need = dg.requirement(plan).required

    def free(p):                         # the run archive is roomy, the shared root's filesystem is not
        return 1 * GiB if p == sc.root else need
    v = dg.check_for_run(plan, debug=False, allow=False, free_fn=free)
    assert v.status == dg.PASS
    assert "shared compile cache: ⚠️  LOW" in v.shared_line and "REPORTED, not refused" in v.shared_line
    assert v.shared_line in "\n".join(v.lines())
    roomy = dg.check_for_run(plan, debug=False, allow=False, free_fn=lambda p: 500 * GiB)
    assert roomy.shared_line.startswith("shared compile cache: ✓")


def test_the_plan_reads_the_shared_root_only_when_sharing_is_on_and_creates_nothing(tmp_path, monkeypatch):
    from agents.model import compile_cache_shared as SH
    root = tmp_path / "never_created"
    monkeypatch.setenv(SH.ENV_DIR, str(root))
    monkeypatch.setenv(SH.ENV_CAP, "2")
    monkeypatch.setenv(SH.ENV_OPT, "0")
    assert dg.shared_cache_plan(str(tmp_path / "run"), compiles=True) is None       # opted out
    monkeypatch.setenv(SH.ENV_OPT, "1")
    assert dg.shared_cache_plan(str(tmp_path / "run"), compiles=False) is None      # a CPU run compiles nothing
    sc = dg.shared_cache_plan(str(tmp_path / "run"), compiles=True)
    assert sc is not None and sc.root == str(root) and sc.cap == 2 * GiB and sc.used == 0 and sc.same_fs
    assert not root.exists()                                                         # the preflight changes nothing
    plan = dg.plan_from_values(run_dir=str(tmp_path / "run"), model_path=None, steps=98_304, resume_steps=0,
                               checkpoint_every_steps=None, eval_freq=None, fork=False, device="cuda")
    assert plan.shared_cache == sc


def test_is_disk_full_takes_none_and_survives_a_cyclic_cause_chain():
    """The mypy fix (the cursor is `BaseException | None`): `None` is not a disk-full error, and a
    `__cause__` / `__context__` cycle terminates (the `seen` set) instead of looping."""
    assert not dg.is_disk_full(None)
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert not dg.is_disk_full(a)
    full = OSError(errno.ENOSPC, "No space left on device")
    b.__cause__ = full
    assert dg.is_disk_full(a)
