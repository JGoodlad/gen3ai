"""Regression tests for the canary logger (canary.py): rows, seen-keys, alerts, the watch loop."""
import json
import os
import time

from main.ops.arm_chain import canary, live


def rec(verdict="PASS", compiled=0.6, eager=0.3, sp=0.01, update=7, steps=1000):
    return {"verdict": verdict, "update": update, "num_timesteps": steps,
            "watch": {"scheduled": {"clip_margin": {"max_abs_dlogp": {"compiled": compiled, "eager": eager},
                                                    "same_precision": {"max_abs_dlogp": sp}}}}}


def write_rows(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "canary_verdicts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_row(run_dir, r):
    with open(os.path.join(run_dir, "canary_verdicts.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")


def test_format_row_readings_and_alert_rule():
    """One rule, no family bar: a non-PASS above 0.54 alerts; the same reading under PASS does not."""
    line, hot = canary.format_row("runA", rec("FAIL", compiled=0.6, eager=0.3), 0.54)
    assert hot is True
    assert line == "runA u7 @1000 FAIL compiled 0.6 eager 0.3 ratio 2.0 same-precision 0.01"
    assert canary.format_row("runA", rec("PASS", compiled=0.6), 0.54)[1] is False
    assert canary.format_row("runA", rec("FAIL", compiled=0.5, eager=0.3), 0.54)[1] is False


def test_format_row_non_pass_without_readings_alerts():
    """A non-PASS record with no readable reading cannot be shown safe: it alerts."""
    line, hot = canary.format_row("r", {"verdict": "ERROR"}, 0.54)
    assert hot is True and "ratio None" in line


def test_scan_logs_only_new_rows_across_calls(tmp_path):
    """The seen file bounds each scan to rows not yet logged."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec(update=1), rec(update=2)])
    seen, out = canary.Seen(str(tmp_path / "seen.json")), []
    assert canary.scan([rd], seen, out.append, lambda k, m: None) == 2
    assert canary.scan([rd], seen, out.append, lambda k, m: None) == 0
    append_row(rd, rec(update=3))
    assert canary.scan([rd], seen, out.append, lambda k, m: None) == 1
    assert [ln.split()[1] for ln in out] == ["u1", "u2", "u3"]


def test_recreated_run_dir_with_fewer_rows_is_logged_from_row_one(tmp_path):
    """H-10: the seen key is the file's inode, not the run name; a re-created dir with fewer rows is not skipped."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec(update=1), rec(update=2), rec(update=3)])
    seen, out = canary.Seen(str(tmp_path / "seen.json")), []
    canary.scan([rd], seen, out.append, lambda k, m: None)
    os.rename(rd, rd + ".gone")                    # keep the old inode allocated so the new file cannot reuse it
    write_rows(rd, [rec(update=101)])
    assert canary.scan([rd], seen, out.append, lambda k, m: None) == 1
    assert out[-1].split()[1] == "u101"


def test_unparsable_row_is_logged_and_alerts(tmp_path):
    """A corrupt canary row is reported, not skipped."""
    rd = str(tmp_path / "runA")
    os.makedirs(rd)
    with open(os.path.join(rd, "canary_verdicts.jsonl"), "w") as f:
        f.write("{not json\n")
    out, alerts = [], []
    canary.scan([rd], canary.Seen(str(tmp_path / "s.json")), out.append, lambda k, m: alerts.append(k))
    assert "UNPARSABLE" in out[0] and len(alerts) == 1


def test_scan_alerts_on_hot_row_only(tmp_path):
    """scan routes only a hot row (non-PASS above the bar) to the alert callback."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec("PASS", compiled=0.9), rec("FAIL", compiled=0.9)])
    alerts = []
    canary.scan([rd], canary.Seen(str(tmp_path / "s.json")), lambda s: None, lambda k, m: alerts.append((k, m)))
    assert len(alerts) == 1 and alerts[0][0].endswith(" 1") and "FAIL" in alerts[0][1]


def test_alerter_fires_a_key_once_even_across_instances(tmp_path):
    """Each alert case fires once (keys persist in a file), so a 5-minute loop does not page every pass."""
    got, path = [], str(tmp_path / "alerted.txt")
    a = canary.Alerter(path, got.append)
    a("k1", "first")
    a("k1", "again")
    a("k2", "other")
    canary.Alerter(path, got.append)("k1", "new process, same key")
    assert got == ["first", "other"]


def test_watch_emits_expiring_on_last_pass_and_writes_heartbeat(tmp_path):
    """CB-13: v9 lapsed silently after its bounded loop; the last pass must say EXPIRING and each pass a heartbeat."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("100.0 0\n")
    out_dir, emitted, sleeps = tmp_path / "out", [], []
    rc = canary.watch(str(tmp_path / "models"), str(out_dir), emitted.append, interval_s=300, passes=2, proc=str(proc),
                      sleep=sleeps.append)
    assert rc == 0 and sleeps == [300]
    assert len(emitted) == 1 and "EXPIRING after 2 passes" in emitted[0]
    assert (out_dir / "canary_heartbeat").read_text().split()[-2:] == ["pass", "2/2"]


def test_watch_logs_a_live_run_discovered_without_a_glob(tmp_path):
    """watch uses live.discover (launcher argv), so a never-seen family's canary rows are logged."""
    proc, models = tmp_path / "proc", tmp_path / "models"
    proc.mkdir()
    (proc / "uptime").write_text("100.0 0\n")
    write_rows(str(models / "rb_brandnew_family_s1"), [rec(update=5)])
    d = proc / "10"
    d.mkdir()
    (d / "fd").mkdir()
    (d / "stat").write_text("10 (python) S 1 10 10 " + " ".join(["0"] * 8) + " 0 0 0 0 20 0 1 0 9000 0 0\n")
    (d / "cmdline").write_bytes(b"python\0-m\0main.launcher\0--run-name\0rb_brandnew_family_s1\0")
    out_dir = tmp_path / "out"
    canary.watch(str(models), str(out_dir), lambda m: None, passes=1, proc=str(proc), sleep=lambda s: None)
    log = (out_dir / "canary_log.txt").read_text()
    assert "rb_brandnew_family_s1 u5" in log


# -- liveness_alerts ----------------------------------------------------------------------------------------

def mkrun(run_dir, elapsed_s, pid=11):
    return live.LiveRun(pid=pid, start_ticks=5, elapsed_s=elapsed_s, run_dir=run_dir, source="argv:--run-name")


def collect():
    got = []
    return got, (lambda key, msg: got.append((key, msg)))


def test_unresolved_run_alerts():
    """A live launcher whose run is UNRESOLVED has canaries nobody is logging: alert."""
    got, alert = collect()
    r = live.LiveRun(pid=12, start_ticks=6, elapsed_s=10, run_dir=None, source="", reason="why not")
    canary.liveness_alerts([r], alert, now=time.time())
    assert len(got) == 1 and "UNRESOLVED" in got[0][1] and "why not" in got[0][1]


def test_run_up_90_min_without_canary_file_alerts_and_30_min_does_not(tmp_path):
    """H-12: 'NO canary record' after the 60-min default (a cold start reaches its first record at ~34 min)."""
    rd = str(tmp_path / "runA")
    os.makedirs(rd)
    got, alert = collect()
    canary.liveness_alerts([mkrun(rd, 90 * 60)], alert, now=time.time())
    assert len(got) == 1 and "NO canary record" in got[0][1]
    got2, alert2 = collect()
    canary.liveness_alerts([mkrun(rd, 30 * 60)], alert2, now=time.time())
    assert got2 == []


def test_canary_file_older_than_launcher_start_counts_as_none(tmp_path):
    """A canary file left by an earlier life of the same run dir is not this launcher's record."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec()])
    now = time.time()
    f = os.path.join(rd, "canary_verdicts.jsonl")
    os.utime(f, (now - 7200, now - 7200))
    got, alert = collect()
    canary.liveness_alerts([mkrun(rd, 90 * 60)], alert, now=now)
    assert len(got) == 1 and "NO canary record" in got[0][1]


def test_fresh_canary_file_is_quiet_and_stale_one_alerts(tmp_path):
    """A recent record is quiet; a newest record older than stale_h alerts once per mtime."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec()])
    now = time.time()
    f = os.path.join(rd, "canary_verdicts.jsonl")
    os.utime(f, (now - 60, now - 60))
    got, alert = collect()
    canary.liveness_alerts([mkrun(rd, 5 * 3600)], alert, now=now)
    assert got == []
    os.utime(f, (now - 4 * 3600, now - 4 * 3600))
    canary.liveness_alerts([mkrun(rd, 5 * 3600)], alert, now=now)
    assert len(got) == 1 and "old" in got[0][1]


def test_watch_files_alerts_when_stdout_is_not_a_regular_file(tmp_path):
    """The alert watch can follow only regular files; a pipe/tty stdout alerts, an orphan trainer is skipped."""
    log = tmp_path / "l.log"
    log.write_text("x")
    ok = live.LiveRun(1, 1, 1, "/r", "child", stdout=str(log))
    pipe = live.LiveRun(2, 1, 1, "/r", "child", stdout="pipe:[123]")
    orphan = live.LiveRun(3, 1, 1, "/r", "orphan-trainer", stdout="pipe:[9]")
    got, alert = collect()
    assert canary.watch_files([ok, pipe, orphan], alert) == [str(log)]
    assert len(got) == 1 and "pipe:[123]" in got[0][1]


def test_a_half_written_last_row_waits_for_the_next_pass(tmp_path):
    """Review 10-26: a row still being written was parsed half, paged as UNPARSABLE and marked seen for ever."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec(update=1)])
    with open(os.path.join(rd, "canary_verdicts.jsonl"), "a") as f:
        f.write(json.dumps(rec(update=2))[:20])
    seen, out, alerts = canary.Seen(str(tmp_path / "seen.json")), [], []
    assert canary.scan([rd], seen, out.append, lambda k, m: alerts.append(m)) == 1
    with open(os.path.join(rd, "canary_verdicts.jsonl"), "a") as f:
        f.write(json.dumps(rec(update=2))[20:] + "\n")
    assert canary.scan([rd], seen, out.append, lambda k, m: alerts.append(m)) == 1
    assert [ln.split()[1] for ln in out] == ["u1", "u2"] and alerts == []
