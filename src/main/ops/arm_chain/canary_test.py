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


def test_format_row_readings_and_every_non_pass_alerts():
    """10-27: N1-P s3501's u200 UNCONFIRMED read compiled 0.1675 — under the old 0.54 value bar, so nobody was paged.
    EVERY non-PASS verdict alerts, whatever its readings; a PASS never does."""
    line, hot = canary.format_row("rb_x35_n1p_s3501", rec("unconfirmed", compiled=0.1675, eager=0.12))
    assert hot is True
    assert line == "rb_x35_n1p_s3501 u7 @1000 unconfirmed compiled 0.1675 eager 0.12 ratio 1.396 same-precision 0.01"
    assert canary.format_row("runA", rec("fatal", compiled=0.01, eager=0.3))[1] is True
    assert canary.format_row("runA", rec("PASS", compiled=0.9, eager=0.1))[1] is False
    assert canary.format_row("runA", rec("pass", compiled=0.9, eager=0.1))[1] is False


def test_format_row_non_pass_without_readings_alerts():
    """A non-PASS record with no readable reading alerts too."""
    line, hot = canary.format_row("r", {"verdict": "ERROR"})
    assert hot is True and "ratio None" in line


def test_a_ratio_above_the_family_bar_is_a_descriptive_tag_that_never_alerts():
    """The family bars (canary_decide: C0 2.64, X47 2.40) tag the line; the resume decision stays canary_decide's."""
    line, hot = canary.format_row("rb_x35_c0_s3505", rec("PASS", compiled=0.9, eager=0.3))
    assert line.endswith(" [descriptive: ratio>2.64 (c0 bar)]") and hot is False
    line, _ = canary.format_row("rb_x35t_x47d4_s3602", rec("PASS", compiled=0.5, eager=0.2))
    assert line.endswith(" [descriptive: ratio>2.4 (x47 bar)]")
    assert "descriptive" not in canary.format_row("rb_x35_c0_s3505", rec("PASS", compiled=0.6, eager=0.3))[0]
    assert "descriptive" not in canary.format_row("rb_x35_r0_s3531", rec("PASS", compiled=0.9, eager=0.1))[0]


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


def test_scan_alerts_on_every_non_pass_row_only(tmp_path):
    """scan routes every non-PASS row (whatever its value) to the alert callback, and no PASS row."""
    rd = str(tmp_path / "runA")
    write_rows(rd, [rec("PASS", compiled=0.9), rec("FAIL", compiled=0.01)])
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
    rc = canary.watch(str(tmp_path / "models"), str(out_dir), emitted.append, str(tmp_path / "archive"), interval_s=300,
                      passes=2, proc=str(proc),
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
    canary.watch(str(models), str(out_dir), lambda m: None, str(tmp_path / "archive"), passes=1, proc=str(proc),
                 sleep=lambda s: None)
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
    assert canary.watch_files([ok, pipe, orphan], alert) == [str(log), "/r/launcher_child.full.log",
                                                             "/r/launcher_child.full.log"]
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


# -- the launcher-log alerts and the Monitor filter (operator review 10-27) -----------------------------------

ALERT_LINES = ["Traceback (most recent call last):", "🛑 Child crashed (exit 1) · crash #1",
               "[contention] WEDGE: no progress for 600 s", "UpdateWontFit: the update needs 31.2 GiB",
               "BEHAVIOUR-POLICY MISMATCH at update 812", "StampMismatch: the Rust extension stamp differs",
               "[canary] compiled DISAGREES with eager at u200"]


def test_the_monitor_filter_pages_every_launcher_alert_and_the_chain_and_canary_holds():
    import re
    f = re.compile(canary.monitor_filter("x35-oct27"))
    for ln in ALERT_LINES + ["[2026-10-27T01:00:00-07:00] ARMCHAIN[x35-oct27]: HOLD: arm:r at RUNNING: x",
                             "[2026-10-27T01:00:00-07:00] ARMCHAIN[x35-oct27]: SHIP-HOLD: arm:r: push failed",
                             "[2026-10-27T01:00:00-07:00] ARMCHAIN[x35-oct27]: chain COMPLETE: every item is DONE",
                             "[2026-10-27T01:00:00-07:00] CANARYLOG: HOLD: CANARY non-PASS: r u200 unconfirmed"]:
        assert f.search(ln), ln
    for ln in ["[2026-10-27T01:00:00-07:00] ARMCHAIN[x35-oct27]: arm:r: RUNNING -> ENDED: launcher_rc=0",
               "[2026-10-27T01:00:00-07:00] ARMCHAIN[other]: HOLD: arm:q at RUNNING: x",
               "[PERIODIC] update 812 fps 1,402 wedge-free"]:
        assert not f.search(ln), ln


def test_the_monitor_filter_works_under_grep_without_a_warning(tmp_path):
    """The Monitor runs it through ``grep -E``: it must match there, and print no stray-backslash warning."""
    import subprocess
    lines = "\n".join(ALERT_LINES + ["[t] ARMCHAIN[x35-oct27]: SHIP-HOLD: arm:r: x", "a quiet line"]) + "\n"
    p = subprocess.run(["grep", "-E", canary.monitor_filter("x35-oct27")], input=lines, capture_output=True, text=True)
    assert p.stderr == "" and len(p.stdout.splitlines()) == len(ALERT_LINES) + 1


def test_the_design_doc_carries_the_monitor_filter_the_code_prints():
    """§2 and §7 step 7 quote the filter; a filter changed in one place only fails here."""
    from utils.paths import repo_path
    doc = open(repo_path("designs", "ops", "arm_chain.md")).read()
    assert canary.monitor_filter("<name>") in doc
    assert canary.LAUNCHER_ALERT_ERE in doc


# -- canary weight dumps archived (operator review 10-27) --------------------------------------------------------

def test_every_unconfirmed_fatal_and_near_dump_is_archived_with_its_sha256(tmp_path):
    import hashlib
    rd = tmp_path / "models" / "rb_x35_n1p_s3501"
    rd.mkdir(parents=True)
    names = ["canary_u200_s1048576_unconfirmed_policy.pt", "canary_u200_s1048576_r2_fatal_policy.pt",
             "canary_u310_s2000000_near_policy.pt"]
    for i, n in enumerate(names):
        (rd / n).write_bytes(b"weights %d" % i)
    for n in ["canary_u7_s1_pass_policy.pt", ".canary_u200_s1_unconfirmed_policy.pt.42.tmp", "canary_verdicts.jsonl"]:
        (rd / n).write_bytes(b"not a dump")
    arch, out, alerts = tmp_path / "archive", [], []
    assert canary.archive_dumps([str(rd)], str(arch), out.append, lambda k, m: alerts.append(m)) == 3
    for i, n in enumerate(names):
        u = n.split("_")[1]
        d = arch / f"rb_x35_n1p_s3501_canary_{u}"
        assert (d / n).read_bytes() == b"weights %d" % i
        assert (d / (n + ".sha256")).read_text() == f"{hashlib.sha256(b'weights %d' % i).hexdigest()}  {n}\n"
    assert sorted(p.name for p in arch.iterdir()) == ["rb_x35_n1p_s3501_canary_u200", "rb_x35_n1p_s3501_canary_u310"]
    assert alerts == [] and len(out) == 3
    # a second pass copies nothing; a source that changed under an archived name is NEVER copied over it
    assert canary.archive_dumps([str(rd)], str(arch), out.append, lambda k, m: alerts.append(m)) == 0
    (rd / names[0]).write_bytes(b"other bytes")
    (arch / "rb_x35_n1p_s3501_canary_u200" / (names[0] + ".sha256")).unlink()
    assert canary.archive_dumps([str(rd)], str(arch), out.append, lambda k, m: alerts.append(m)) == 0
    assert (arch / "rb_x35_n1p_s3501_canary_u200" / names[0]).read_bytes() == b"weights 0"
    assert len(alerts) == 1 and "differs" in alerts[0]


def test_the_watch_loop_archives_a_live_runs_dump(tmp_path):
    proc, models = tmp_path / "proc", tmp_path / "models"
    proc.mkdir()
    (proc / "uptime").write_text("100.0 0\n")
    write_rows(str(models / "rb_a_s1"), [rec("unconfirmed", update=200)])
    (models / "rb_a_s1" / "canary_u200_s9_unconfirmed_policy.pt").write_bytes(b"w")
    d = proc / "10"
    d.mkdir()
    (d / "fd").mkdir()
    (d / "stat").write_text("10 (python) S 1 10 10 " + " ".join(["0"] * 8) + " 0 0 0 0 20 0 1 0 9000 0 0\n")
    (d / "cmdline").write_bytes(b"python\0-m\0main.launcher\0--run-name\0rb_a_s1\0")
    emitted = []
    canary.watch(str(models), str(tmp_path / "out"), emitted.append, str(tmp_path / "arch"), passes=1, proc=str(proc),
                 sleep=lambda s: None)
    assert (tmp_path / "arch" / "rb_a_s1_canary_u200" / "canary_u200_s9_unconfirmed_policy.pt").read_bytes() == b"w"
    assert any("CANARY non-PASS: rb_a_s1 u200 @1000 unconfirmed" in m for m in emitted)
