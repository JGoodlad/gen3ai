"""Regression tests for the chain file (spec.py): every default an ad hoc script used to take is a required field."""
import copy
import subprocess

import pytest

from main.ops.arm_chain import spec

PIN_A = "a" * 40
PIN_B = "b" * 40
TRAINER = "train_rl_agent" + ".py"  # concatenated: root rule 3, this file never holds the literal name


def minimal_arm(**over):
    d = {"kind": "arm", "arm": "n1p", "seed": 3501, "pin": PIN_A, "run": "rb_x35_n1p_s3501", "go_by": "orchestrator",
         "thin": "planner", "disk_need_gib": 50}
    d.update(over)
    return d


def minimal_window(**over):
    d = {"kind": "window", "name": "pre-flight", "worktree": "/tmp/wt", "require_commit": "abcdef1",
         "budget_min": 30, "on_fail": "hold", "preempt_marker": "PREEMPT",
         "steps": [{"name": "s1", "argv": ["echo", "hi"], "mem_gb": 8, "timeout_s": 60}]}
    d.update(over)
    return d


def minimal_chain(**over):
    d = {"schema": spec.SCHEMA, "name": "x35-chain", "job_dir": "/tmp/job", "repo": "/tmp/repo", "python": "/usr/bin/python3",
         "backend": {"kind": "x35", "launch_script": "/tmp/launch.sh", "stage": "/tmp/stage", "lease_owner": "trainer",
                     "lease_token_file": "/tmp/token"},
         "records": {"stage": "/tmp/records"}, "items": [minimal_arm()]}
    d.update(over)
    return d


def with_arm(**over):
    return minimal_chain(items=[minimal_arm(**over)])


def test_minimal_chain_parses():
    """Baseline: the helper chain is valid, so every refusal below is the refusal under test."""
    s = spec.parse(minimal_chain())
    assert [a.run for a in s.arms()] == ["rb_x35_n1p_s3501"]
    assert s.arms()[0].pin == PIN_A
    assert s.status_file == "/tmp/job/status.txt"


def test_arm_without_pin_refused():
    """CB-05: there is no chain-wide default pin; an arm without one is refused."""
    arm = minimal_arm()
    del arm["pin"]
    with pytest.raises(spec.SpecError, match="'pin' is required"):
        spec.parse(minimal_chain(items=[arm]))


def test_abbreviated_pin_refused():
    """CB-05: an 8-hex abbreviation is not a pin (the launch must name one full commit)."""
    with pytest.raises(spec.SpecError, match="40-hex"):
        spec.parse(with_arm(pin="a" * 8))


def test_thin_none_without_reason_refused():
    """H-05: keep-all is a per-arm field with a reason, not a case on the run name's glob."""
    with pytest.raises(spec.SpecError, match="thin_reason"):
        spec.parse(with_arm(thin="none"))
    s = spec.parse(with_arm(thin="none", thin_reason="K9 reads every checkpoint"))
    assert s.arms()[0].thin_reason == "K9 reads every checkpoint"


def test_duplicate_run_refused():
    """A run appears once per chain (a duplicated arm would be launched twice under one name)."""
    with pytest.raises(spec.SpecError, match="duplicate"):
        spec.parse(minimal_chain(items=[minimal_arm(), minimal_arm(arm="other")]))


def test_go_in_flags_refused():
    """The chain adds GO itself; a flag list carrying it would launch without the dry-run re-verify."""
    with pytest.raises(spec.SpecError, match="GO"):
        spec.parse(with_arm(flags=["--x", "GO"]))


def test_trainer_script_name_in_verify_argv_refused():
    """Root rule 3: the literal trainer script name never appears in a backgrounded argv."""
    with pytest.raises(spec.SpecError, match="trainer script"):
        spec.parse(with_arm(verify=[["python", TRAINER, "--debug"]]))


def test_trainer_script_name_in_window_step_refused():
    """Root rule 3 applies to window steps as well as verify argv."""
    w = minimal_window(steps=[{"name": "s", "argv": ["python", TRAINER], "mem_gb": 1, "timeout_s": 5}])
    with pytest.raises(spec.SpecError, match="trainer script"):
        spec.parse(minimal_chain(items=[w]))


def test_window_without_on_fail_refused():
    """A window must say whether the chain continues past its failure (X59-S: continue; others: hold)."""
    w = minimal_window()
    del w["on_fail"]
    with pytest.raises(spec.SpecError, match="on_fail"):
        spec.parse(minimal_chain(items=[w]))
    assert spec.parse(minimal_chain(items=[minimal_window()])).items[0].on_fail == "hold"


def test_wrong_schema_refused():
    """A file of another schema version is not silently read."""
    with pytest.raises(spec.SpecError, match="schema"):
        spec.parse(minimal_chain(schema="arm_chain_v0"))


def test_two_arms_keep_their_own_pins():
    """CB-05 hand-off: the next arm launches at ITS pin, never the first arm's."""
    s = spec.parse(minimal_chain(items=[minimal_arm(), minimal_arm(run="rb_x35_d3mut_s3503", pin=PIN_B)]))
    assert [a.pin for a in s.arms()] == [PIN_A, PIN_B]


def test_input_dict_is_not_mutated():
    """Parsing reads the dict only."""
    d = minimal_chain()
    before = copy.deepcopy(d)
    spec.parse(d)
    assert d == before


# -- check_pins (git) ------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_check_pins_real_commit_passes_and_fake_sha_is_reported(tmp_path):
    """CB-05: a pin that is not a commit in the main checkout is reported, a real one is not."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True, env=env).stdout

    git("init", "-q")
    (repo / "f").write_text("x")
    git("add", "f")
    git("commit", "-q", "-m", "c1")
    real = git("rev-parse", "HEAD").strip()
    good = spec.parse(minimal_chain(repo=str(repo), items=[minimal_arm(pin=real)]))
    assert spec.check_pins(good) == []
    fake = spec.parse(minimal_chain(repo=str(repo), items=[minimal_arm(pin="0" * 40)]))
    problems = spec.check_pins(fake)
    assert len(problems) == 1 and "0" * 40 in problems[0]


def test_a_status_file_the_kit_does_not_write_is_refused():
    """Review 10-26: the x35 kit writes <job_dir>/status.txt; a chain listening elsewhere would never see END."""
    d = minimal_chain()
    d["status_file"] = "/elsewhere/status.txt"
    with pytest.raises(spec.SpecError, match="status_file"):
        spec.parse(d)
