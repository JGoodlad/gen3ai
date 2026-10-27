"""Regression tests for the status file (statuslog.py): anchored, offset-bounded, inode-bound triggers."""
import json
import os

import pytest

from main.ops.arm_chain import statuslog as sl

TS = "2026-10-26T12:00:00+00:00"
CHAIN = "arm_chain"


def clock():
    return TS


def mk(tmp_path, tag=CHAIN):
    return sl.StatusLog(str(tmp_path / "status.txt"), tag, clock=clock)


def wrapper(tmp_path, arm="n1p", seed=3501):
    """Another writer (the launch wrapper) on the same file, under its own tag."""
    return sl.StatusLog(str(tmp_path / "status.txt"), sl.launch_tag(arm, seed), clock=clock)


def test_stale_line_before_arming_never_fires(tmp_path):
    """CB-25: a matching line already in the log when the trigger was armed is history, not an event."""
    log = mk(tmp_path)
    tag = sl.launch_tag("n1p", 3501)
    wrapper(tmp_path).append("END rb_x35_n1p_s3501 launcher_rc=0")
    t = log.arm("end", tag, sl.end_body("rb_x35_n1p_s3501"))
    assert log.find(t) is None
    wrapper(tmp_path).append("END rb_x35_n1p_s3501 launcher_rc=0")
    m = log.find(t)
    assert m is not None and m.group(1) == "0"


def test_quoting_armed_line_does_not_fire_trigger_and_trigger_tag_append_refused(tmp_path):
    """CB-24 (X34 chain 10-19): the chain's 'armed' line quoting the trigger text must neither fire nor be writable."""
    log = mk(tmp_path)
    tag = sl.launch_tag("n1p", 3501)
    body = "END rb_x35_n1p_s3501 launcher_rc=0"
    t = log.arm("end", tag, sl.end_body("rb_x35_n1p_s3501"))
    log.append(f"armed: waiting for '{tag}: {body}'")          # quotes the trigger text under the chain's own tag
    assert log.find(t) is None
    with pytest.raises(sl.SelfMatchError):
        log.append(body, tag=tag)                                 # a line the trigger WOULD match is refused
    assert log.find(t) is None
    assert not any(line.endswith(": " + body) for line in log.lines())


def test_trigger_on_writers_own_tag_refused(tmp_path):
    """CB-24: a trigger on the chain's own tag would fire on the chain's own lines."""
    log = mk(tmp_path)
    with pytest.raises(sl.SelfMatchError):
        log.arm("x", CHAIN, ".*")


def test_run_name_prefix_does_not_fire_longer_run(tmp_path):
    """H-03: END of rb_x35_c0_s350 must not fire on the END line of rb_x35_c0_s3501."""
    log = mk(tmp_path)
    tag = sl.launch_tag("c0", 350)
    t = log.arm("end", tag, sl.end_body("rb_x35_c0_s350"))
    wrapper(tmp_path, "c0", 350).append("END rb_x35_c0_s3501 launcher_rc=0")
    assert log.find(t) is None
    wrapper(tmp_path, "c0", 350).append("END rb_x35_c0_s350 launcher_rc=7")
    m = log.find(t)
    assert m is not None and m.group(1) == "7"


def test_run_name_prefix_does_not_fire_longer_run_s35010(tmp_path):
    """H-04: END of rb_x35_n1p_s3501 must not fire on rb_x35_n1p_s35010."""
    log = mk(tmp_path)
    tag = sl.launch_tag("n1p", 3501)
    t = log.arm("end", tag, sl.end_body("rb_x35_n1p_s3501"))
    wrapper(tmp_path).append("END rb_x35_n1p_s35010 launcher_rc=0")
    assert log.find(t) is None


def test_start_body_is_anchored_to_the_run_name(tmp_path):
    """H-03 for START lines: a prefix run name does not match, a dot in a run name is literal."""
    log = mk(tmp_path)
    tag = sl.launch_tag("n1p", 3501)
    t = log.arm("start", tag, sl.start_body("rb_x35.n1p"))
    w = wrapper(tmp_path)
    w.append("START rb_x35Xn1p: forked")
    assert log.find(t) is None
    w.append("START rb_x35.n1p: forked at pin")
    assert log.find(t) is not None


def test_other_writers_line_with_same_body_under_other_tag_does_not_fire(tmp_path):
    """A trigger on tag 'd3 s3501' ignores another writer's identical body under a different tag."""
    log = mk(tmp_path)
    t = log.arm("end", sl.launch_tag("d3", 3501), sl.end_body("rb_x35_d3mut_s3501"))
    wrapper(tmp_path, "n1p", 3501).append("END rb_x35_d3mut_s3501 launcher_rc=0")
    assert log.find(t) is None
    wrapper(tmp_path, "d3", 3501).append("END rb_x35_d3mut_s3501 launcher_rc=0")
    assert log.find(t) is not None


def test_truncated_file_raises(tmp_path):
    """H-02: a file that shrank below the armed offset is an error, never a silent re-read from the top."""
    log = mk(tmp_path)
    for i in range(3):
        log.append(f"line {i}")
    t = log.arm("end", "d3 s3501", ".*")
    assert t.offset == 3
    with open(log.path, "w") as f:         # same inode, now 1 line
        f.write("[x] a: b\n")
    with pytest.raises(sl.StatusError, match="truncated or rotated"):
        log.find(t)


def test_replaced_file_raises(tmp_path):
    """H-02: a file replaced (rename over, new inode) under a live trigger is refused even if long enough."""
    log = mk(tmp_path)
    log.append("one")
    t = log.arm("end", "d3 s3501", ".*")
    assert t.inode == os.stat(log.path).st_ino
    new = tmp_path / "new.txt"
    new.write_text("[x] a: b\n[x] a: c\n[x] a: d\n")
    os.replace(new, log.path)
    assert os.stat(log.path).st_ino != t.inode
    with pytest.raises(sl.StatusError, match="REPLACED"):
        log.find(t)


def test_persisted_trigger_fires_on_line_written_between_supervisors(tmp_path):
    """CB-16 (chain-swap gap): a trigger armed by supervisor 1 and adopted by supervisor 2 sees the line written between."""
    path = str(tmp_path / "status.txt")
    sup1 = sl.StatusLog(path, CHAIN, clock=clock)
    sup1.append("hello")
    tag = sl.launch_tag("n1p", 3501)
    t = sup1.arm("end", tag, sl.end_body("rb_x35_n1p_s3501"))
    saved = json.loads(json.dumps(t.to_json()))
    del sup1                                                       # supervisor 1 dies
    wrapper(tmp_path).append("END rb_x35_n1p_s3501 launcher_rc=0")  # the gap
    sup2 = sl.StatusLog(path, CHAIN, clock=clock)
    fresh = sup2.arm("end-fresh", tag, sl.end_body("rb_x35_n1p_s3501"))
    assert sup2.find(fresh) is None                                # a re-armed trigger MISSES the gap line
    adopted = sup2.adopt(sl.Trigger.from_json(saved))
    assert adopted == sl.Trigger.from_json(saved)
    assert sup2.find(adopted) is not None
    sup2.adopt(adopted)
    assert sup2._armed.count(adopted) == 1                         # adopt is idempotent


def test_newline_in_message_is_folded_never_forges_a_line_and_lines_are_timestamped(tmp_path):
    """One O_APPEND write per line: a newline would let a message forge a second line, so it is folded to ' | '.
    It is never refused: a refusal inside the HOLD handler killed the supervisor with no HOLD line (2026-10-26)."""
    log = mk(tmp_path)
    got = log.append("a\n[x] d3 s3501: END r launcher_rc=0\r\nb")
    assert got == f"[{TS}] {CHAIN}: a | [x] d3 s3501: END r launcher_rc=0 | b"
    assert log.append("ok") == f"[{TS}] {CHAIN}: ok"
    assert log.lines() == [got, f"[{TS}] {CHAIN}: ok"]                # exactly two lines: nothing forged


def test_a_hold_whose_reason_is_multiline_dirt_writes_its_hold_line(tmp_path):
    """The 2026-10-26 BLOCKER: `git status --porcelain` with two stray files as a HOLD reason."""
    log = mk(tmp_path)
    reason = "main is dirty: ?? a.json\n?? b.json"
    line = log.append(f"HOLD: d3 3504 at PRECHECKED: {reason}")
    assert "HOLD: d3 3504 at PRECHECKED: main is dirty: ?? a.json | ?? b.json" in line
    assert len(log.lines()) == 1
