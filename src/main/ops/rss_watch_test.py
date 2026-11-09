"""``main.ops.rss_watch``: the per-arm RSS watch's BASELINE RULE, pinned against the incident that broke it.

The incident (2026-11-06, ``rb_x60t_s3501``, resume #2; ``status.txt`` lines 1636-1639): the old session script took
its baseline at the resumed child's START-UP sample, because ``launcher_child.full.log`` is appended across lives and
already held rows from the earlier lives::

    10:32:50  baseline (first sample after the first update): 4 procs, total 1.70 GiB, largest 0.63 GiB
    10:37:50  PAGE: total RSS 14.20 GiB > 125 % of the baseline 1.70 GiB - re-baselining

The rule now: the baseline waits for a ``total_timesteps`` row that is NEW since the watcher armed. Every test of the
rule below FAILS on the old rule (``test_the_old_rule_is_the_incident`` shows it doing so, by swapping it back in).

Unmarked and ~free except the one ``/proc`` smoke (``integration``: it spawns a ``sleep`` and kills it by PID).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator, List

import pytest

from main.ops import rss_watch as W

GIB = 2**30


def gb(x: float) -> int:
    return int(x * GIB)


def sample(total_gib: float, rows: int, *, n: int = 15, step: int = 280_000_170, avail_gib: float = 70.0,
           largest_gib: float = 9.27, pid: int = 686620) -> W.Sample:
    return W.Sample(step=step, nproc=n, total_bytes=gb(total_gib), max_bytes=gb(largest_gib), max_pid=pid,
                    avail_bytes=gb(avail_gib), log_rows=rows)


def feed(watch: W.RssWatch, *samples: W.Sample) -> List[List[str]]:
    return [watch.observe(s) for s in samples]


# ------------------------------------------------------------------------------------------ the baseline rule


def test_a_fresh_launch_baselines_at_the_first_update_not_at_start_up() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=0)
    out = feed(w, sample(1.70, 0, n=4), sample(12.0, 0), sample(14.17, 1), sample(14.40, 2))
    assert out[0] == [] and out[1] == []                       # start-up and the compile ramp: not a baseline
    assert len(out[2]) == 1 and out[2][0].startswith("baseline (first sample after the first update): 15 procs, total 14.17 GiB")
    assert w.baseline == gb(14.17)
    assert out[3] == []                                        # 14.40 is 101.6 % of 14.17: no page


def test_a_resume_baselines_after_this_lifes_first_update_and_never_false_pages() -> None:
    """THE INCIDENT, replayed: the log holds 215 rows from the earlier lives when the watcher arms."""
    w = W.RssWatch(280_000_170, rows_at_arm=215)
    out = feed(w, sample(1.70, 215, n=4, largest_gib=0.63), sample(14.20, 215), sample(14.17, 216), sample(14.30, 217))
    assert out[0] == [] and out[1] == [], "a resumed child's start-up ramp must neither baseline nor page"
    assert len(out[2]) == 1 and "total 14.17 GiB" in out[2][0] and out[2][0].startswith("baseline (")
    assert out[3] == []
    assert w.baseline == gb(14.17)
    assert not any("PAGE" in line for lines in out for line in lines)


def test_the_old_rule_is_the_incident(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the old rule swapped back in (any row in the log = training), the same replay baselines at 1.70 GiB and pages."""
    monkeypatch.setattr(W, "phase", lambda log_rows, rows_at_arm: "train" if log_rows > 0 else "startup")
    w = W.RssWatch(280_000_170, rows_at_arm=215)
    out = feed(w, sample(1.70, 215, n=4, largest_gib=0.63), sample(14.20, 215))
    assert out[0][0].startswith("baseline (first sample after the first update): 4 procs, total 1.70 GiB")
    assert out[1] == ["PAGE: total RSS 14.20 GiB > 125 % of the baseline 1.70 GiB — re-baselining"]


def test_a_watcher_armed_after_the_first_update_waits_for_the_next_one() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=216)
    out = feed(w, sample(14.1, 216), sample(14.2, 217))
    assert out[0] == []
    assert out[1] and out[1][0].startswith("baseline (")


def test_a_replaced_shorter_log_makes_all_its_rows_new() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=500)
    out = feed(w, sample(1.0, 0, n=3), sample(14.0, 1))
    assert out[0] == []
    assert out[1] and out[1][0].startswith("baseline (")


def test_the_baseline_line_keeps_the_old_scripts_text() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=0)
    (line,) = w.observe(sample(14.17, 1, largest_gib=9.27, pid=686620, avail_gib=73.45))
    assert line == ("baseline (first sample after the first update): 15 procs, total 14.17 GiB, largest 9.27 GiB "
                    "(pid 686620), MemAvailable 73.45 GiB")


# ------------------------------------------------------------------------------------------ the other events


def test_growth_over_the_baseline_pages_once_and_rebaselines() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=0)
    w.observe(sample(10.0, 1))
    assert w.observe(sample(12.5, 2)) == []                    # exactly 125 % (integer bytes): strictly greater pages
    out = w.observe(sample(13.0, 3))
    assert out == ["PAGE: total RSS 13.00 GiB > 125 % of the baseline 10.00 GiB — re-baselining"]
    assert w.baseline == gb(13.0)
    assert w.observe(sample(14.0, 4)) == []                    # 107.7 % of the new baseline


def test_low_mem_available_pages_below_8_gib_only() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=0)
    w.observe(sample(10.0, 1))
    assert w.observe(sample(10.0, 2, avail_gib=8.0)) == []
    (line,) = w.observe(sample(10.0, 3, avail_gib=7.5))
    assert line == "PAGE: MemAvailable 7.50 GiB < 8 GiB (total 10.00 GiB, 15 procs)"


def test_the_summary_fires_at_the_fork_step_plus_10m_and_ends_the_watch() -> None:
    w = W.RssWatch(280_000_170, rows_at_arm=0)
    w.observe(sample(14.0, 1, step=289_999_999))
    assert not w.done
    out = w.observe(sample(14.2, 3, step=290_000_170))
    assert w.done and out[-1].startswith("SUMMARY at 10M (step 290000170): baseline 14.00 GiB, last 14.20 GiB (15 procs, largest 9.27 GiB), max 14.20 GiB")


# ------------------------------------------------------------------------------------------ the readers


def test_count_update_rows_counts_the_trainers_table_rows(tmp_path: Path) -> None:
    log = tmp_path / "launcher_child.full.log"
    assert W.count_update_rows(log) == 0                       # a missing log has no rows
    log.write_text("===== child attached 2026-11-06 09:34:07 (pid 469464) =====\n"
                   "|    total_timesteps                | 280105805 |\n"
                   "noise\n"
                   "|    total_timesteps                             | 280204000  |\n")
    assert W.count_update_rows(log) == 2


def test_newest_step_is_the_largest_checkpoint(tmp_path: Path) -> None:
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in ("checkpoint_280000170_steps.zip", "checkpoint_290000170_steps.zip", "checkpoint_9000_steps.zip", "other.txt"):
        (ck / n).write_text("x")
    assert W.newest_step(ck) == 290_000_170
    assert W.newest_step(tmp_path / "absent") == 0


# ------------------------------------------------------------------------------------------ the loop, end to end


def test_the_loop_on_a_resumed_run_logs_one_baseline_after_the_new_update_and_no_page(tmp_path: Path) -> None:
    """The real log file and the real status file: the log holds three old rows; the fourth arrives mid-watch."""
    run_dir = tmp_path / "models" / "rb_demo"
    run_dir.mkdir(parents=True)
    child = run_dir / "launcher_child.full.log"
    child.write_text("|    total_timesteps | 1 |\n" * 3)
    status, rows_dir = tmp_path / "status.txt", tmp_path / "rss"
    totals: Iterator[float] = iter([1.70, 14.20, 14.17, 14.30])
    sleeps: List[float] = []

    def take() -> W.Sample:
        return sample(next(totals), W.count_update_rows(child), n=4 if len(sleeps) == 0 else 15)

    def nap(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) == 2:                                   # the resumed child finishes its first update
            with open(child, "a") as fh:
                fh.write("|    total_timesteps | 2 |\n")

    alive_calls = {"n": 0}

    def alive(_pid: int) -> bool:
        alive_calls["n"] += 1
        return alive_calls["n"] <= 4                           # four samples, then the launcher is gone

    cfg = W.Config(launcher=4242, run="rb_demo", fork_step=280_000_170, run_dir=run_dir, status_file=status,
                   rows_dir=rows_dir, interval_s=0.0, max_samples=36)
    assert W.run(cfg, sample=take, alive=alive, sleep=nap) == 0
    lines = status.read_text().splitlines()
    assert all("RSSWATCH[rb_demo]: " in line for line in lines)
    msgs = [line.split("RSSWATCH[rb_demo]: ", 1)[1] for line in lines]
    assert msgs[0].startswith("armed (pid ") and "launcher 4242, fork 280000170 → summary at 290000170" in msgs[0]
    assert msgs[1].startswith("baseline (first sample after the first update): 15 procs, total 14.17 GiB")
    assert msgs[2] == "SUMMARY (launcher 4242 gone): baseline 14.17 GiB, max 14.30 GiB"
    assert len(msgs) == 3, msgs                                # no PAGE
    phases = [r.split("\t")[-1] for r in (rows_dir / "rb_demo.tsv").read_text().splitlines()[1:]]
    assert phases == ["startup", "startup", "train", "train"]


def test_the_loop_stops_at_the_sample_bound(tmp_path: Path) -> None:
    status, rows_dir = tmp_path / "status.txt", tmp_path / "rss"
    cfg = W.Config(launcher=1, run="r", fork_step=280_000_170, run_dir=tmp_path / "none", status_file=status, rows_dir=rows_dir,
                   interval_s=0.0, max_samples=3)
    assert W.run(cfg, sample=lambda: sample(10.0, 0), alive=lambda _p: True, sleep=lambda _s: None) == 0
    last = status.read_text().splitlines()[-1]
    assert "SUMMARY (3-h bound): baseline 0.00 GiB, max 10.00 GiB" in last


# ------------------------------------------------------------------------------------------ the CLI and /proc


def test_the_cli_documents_itself_and_refuses_a_bare_call(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ok:
        W.main(["--help"])
    assert ok.value.code == 0 and "rss_watch" in capsys.readouterr().out
    with pytest.raises(SystemExit) as bare:
        W.main([])
    assert bare.value.code == 2                                # no default run, no default status file


def test_a_dead_launcher_pid_is_refused(tmp_path: Path) -> None:
    p = subprocess.Popen(["true"])
    p.wait()
    assert W.main(["--launcher-pid", str(p.pid), "--run", "r", "--fork-step", "0", "--status-file", str(tmp_path / "s"),
                   "--rows-dir", str(tmp_path / "rows"), "--run-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "s").exists()


@pytest.mark.integration
def test_tree_rss_reads_a_real_process_tree(tmp_path: Path) -> None:
    p = subprocess.Popen(["sleep", "30"])
    try:
        n, total, mx, mxp = W.tree_rss(p.pid)
        assert n == 1 and mxp == p.pid and 0 < mx == total
        s = W.take_sample(p.pid, tmp_path)
        assert s.nproc == 1 and s.avail_bytes > 0 and s.step == 0 and s.log_rows == 0
    finally:
        p.kill()
        p.wait()
