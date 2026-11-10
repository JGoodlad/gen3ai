"""``utils.core_split``: the ONE owner of the trainer / agent CPU lists, its state file and its two refusals.

This file and the owner are the only places that spell a list (``core_split_literal_gate_test.py``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils import core_split as CS


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "state" / "core_split.json"
    monkeypatch.setenv(CS.STATE_ENV, str(p))
    return p


def _nobody() -> dict:
    return {}


def _a_trainer() -> dict:
    return {4242: "launcher", 4250: "trainer"}


def test_the_default_split_is_byte_identical_to_the_lists_every_site_used_to_spell() -> None:
    """Before the owner, a dozen files spelled these two strings. No state file must give exactly them."""
    assert CS.DEFAULT_SPLIT == "5+3"
    assert CS.DEFAULT.trainer_list == "0-4,8-12" and CS.DEFAULT.agent_list == "5-7,13-15"
    assert CS.DEFAULT.trainer_ids == (0, 1, 2, 3, 4, 8, 9, 10, 11, 12)
    assert CS.DEFAULT.agent_ids == (5, 6, 7, 13, 14, 15)


def test_the_suite_is_sealed_onto_the_default_split() -> None:
    """``conftest.py`` names a state file that cannot exist, so no test reads the operator's split."""
    p = CS.state_path()
    assert os.environ[CS.STATE_ENV] == str(p) and not p.exists() and "nonexistent" in str(p)
    assert (CS.trainer_cpus(), CS.agent_cpus()) == ("0-4,8-12", "5-7,13-15")
    assert (CS.trainer_cpu_ids(), CS.agent_cpu_ids()) == (CS.DEFAULT.trainer_ids, CS.DEFAULT.agent_ids)
    assert CS.launch_trainer_cpus(live=_nobody) == ("0-4,8-12", "")          # a plain launch read creates nothing


def test_every_split_is_whole_physical_cores_and_the_two_sides_partition_the_box() -> None:
    lists = {"5+3": ("0-4,8-12", "5-7,13-15"), "4+4": ("0-3,8-11", "4-7,12-15"), "3+5": ("0-2,8-10", "3-7,11-15")}
    assert set(lists) == set(CS.SPLITS)
    for name, (trainer, agents) in lists.items():
        s = CS.parse_split(name)
        assert (s.name, s.trainer_list, s.agent_list) == (name, trainer, agents)
        assert sorted(s.trainer_ids + s.agent_ids) == list(range(2 * CS.PHYSICAL_CORES))
        for ids in (s.trainer_ids, s.agent_ids):                           # CPU n comes with its sibling n + 8
            assert {c + CS.PHYSICAL_CORES for c in ids if c < CS.PHYSICAL_CORES} == {c for c in ids if c >= CS.PHYSICAL_CORES}
    for bad in ("6+2", "2+6", "5+4", "4", "", "5-3"):
        with pytest.raises(CS.CoreSplitError):
            CS.parse_split(bad)


def test_cpu_lists_round_trip() -> None:
    assert CS.format_cpus([0, 2, 3]) == "0,2-3" and CS.format_cpus([]) == "" and CS.format_cpus([7, 5, 6, 6]) == "5-7"
    assert CS.parse_cpus("0-2,5") == [0, 1, 2, 5] and CS.parse_cpus(" 3 \n") == [3] and CS.parse_cpus("") == []
    for bad in ("a", "3-1", "1-"):
        with pytest.raises(ValueError):
            CS.parse_cpus(bad)
    for s in CS.SPLITS:
        assert CS.format_cpus(CS.parse_cpus(CS.parse_split(s).agent_list)) == CS.parse_split(s).agent_list


def test_the_ab_arms_drop_the_highest_physical_cores_of_the_control() -> None:
    control = list(CS.DEFAULT.trainer_ids)
    assert CS.narrowed(control, 0) == control
    assert CS.format_cpus(CS.narrowed(control, 1)) == CS.parse_split("4+4").trainer_list
    assert CS.format_cpus(CS.narrowed(control, 2)) == CS.parse_split("3+5").trainer_list
    assert set(CS.narrowed(control, 2)) < set(CS.narrowed(control, 1)) < set(control)
    with pytest.raises(ValueError, match="whole physical cores"):
        CS.narrowed([0, 1, 2, 8, 9], 1)
    with pytest.raises(ValueError, match="cannot drop"):
        CS.narrowed([0, 8], 1)


def test_set_is_refused_while_a_trainer_is_live_and_changes_nothing(state: Path) -> None:
    with pytest.raises(CS.Refused, match="4242.*--next-launch"):
        CS.set_split("4+4", live=_a_trainer)
    assert not state.exists() and CS.agent_cpus() == "5-7,13-15"
    assert "ACTIVE" in CS.set_split("4+4", live=_nobody)
    assert (CS.trainer_cpus(), CS.agent_cpus()) == ("0-3,8-11", "4-7,12-15")
    d = json.loads(state.read_text())
    assert d["schema"] == CS.SCHEMA and d["split"] == "4+4" and d["next_launch"] is None and d["log"][-1]["event"] == "set"
    with pytest.raises(CS.Refused):                                         # back to the default is a change too
        CS.set_split("5+3", live=_a_trainer)
    assert CS.active().name == "4+4"


def test_next_launch_only_queues_and_a_launch_read_with_no_trainer_live_applies_it(state: Path) -> None:
    """core_split.md §7's order: the agents never widen under a live tree. Fails on a revert of either refusal."""
    msg = CS.set_split("4+4", next_launch=True, live=_a_trainer)
    assert "QUEUED" in msg and CS.active().name == "5+3" and CS.agent_cpus() == "5-7,13-15"
    # a trainer is still live: the launch read gives the ACTIVE list and the queue stands
    cpus, note = CS.launch_trainer_cpus(live=_a_trainer)
    assert cpus == "0-4,8-12" and "stays QUEUED" in note and CS.read_state()["next_launch"] == "4+4"
    # a dry run reports the list a launch would get and writes nothing: not the state, not even the lock file
    lock = Path(str(state) + ".lock")
    lock.unlink()
    before = state.read_bytes()
    assert CS.launch_trainer_cpus(dry_run=True, live=_nobody)[0] == "0-3,8-11"
    cpus, note = CS.launch_trainer_cpus(dry_run=True, live=_a_trainer)
    assert cpus == "0-4,8-12" and CS.STAYS_QUEUED in note and CS.STAYS_QUEUED == "stays QUEUED"
    assert state.read_bytes() == before and not lock.exists() and sorted(x.name for x in state.parent.iterdir()) == [state.name]
    assert CS.active().name == "5+3" and CS.read_state()["next_launch"] == "4+4"
    # the launch: the new tree starts narrow and the agents widen with the same write
    cpus, note = CS.launch_trainer_cpus(live=_nobody)
    assert cpus == "0-3,8-11" and "now ACTIVE" in note
    assert (CS.active().name, CS.agent_cpus(), CS.read_state()["next_launch"]) == ("4+4", "4-7,12-15", None)
    assert [e["event"] for e in CS.read_state()["log"]] == ["set-next-launch", "applied-at-launch"]
    assert CS.launch_trainer_cpus(live=_nobody) == ("0-3,8-11", "")
    # queueing the active split clears a queue
    CS.set_split("3+5", next_launch=True)
    assert "nothing is queued" in CS.set_split("4+4", next_launch=True) and CS.read_state()["next_launch"] is None


def test_an_unreadable_state_is_an_error_never_a_silent_default(state: Path) -> None:
    state.parent.mkdir(parents=True)
    for bad in ("{", "[]", json.dumps({"schema": CS.SCHEMA, "split": "9+9"}), json.dumps({"schema": "x", "split": "5+3"}),
                json.dumps({"schema": CS.SCHEMA, "split": "5+3", "next_launch": "nope"})):
        state.write_text(bad)
        with pytest.raises(CS.CoreSplitError, match="unreadable"):
            CS.agent_cpus()
        assert CS.main(["agent-cpus"]) == CS.EXIT_REFUSED
    state.write_text(json.dumps({"schema": CS.SCHEMA, "split": "3+5"}))
    assert CS.agent_cpus() == "3-7,11-15"


def test_agents_beside_a_live_trainer_never_take_a_cpu_of_the_active_trainer_list(state: Path) -> None:
    """The ship gate's rule. A trainer narrowed for 8 s by the A/B driver frees nothing: before the union a gate that
    started inside a narrow arm took the freed cores and kept them after the driver restored the list."""
    online = list(range(16))
    assert CS.agent_ids_beside(CS.DEFAULT.trainer_ids, online) == list(CS.DEFAULT.agent_ids)
    assert CS.agent_ids_beside(CS.narrowed(CS.DEFAULT.trainer_ids, 2), online) == list(CS.DEFAULT.agent_ids)
    assert CS.agent_ids_beside(list(range(14)), online) == [14, 15]           # a WIDER trainer is respected
    assert CS.agent_ids_beside(online, online) == []
    CS.set_split("3+5", live=_nobody)
    assert CS.agent_ids_beside(CS.parse_split("3+5").trainer_ids, online) == list(CS.parse_split("3+5").agent_ids)
    assert CS.agent_ids_beside(CS.DEFAULT.trainer_ids, online) == list(CS.DEFAULT.agent_ids)   # an old, wider tree


def test_a_trainer_is_what_python_runs_never_a_word_in_an_argv() -> None:
    """A false "live trainer" refuses every ``set`` and keeps a queued split queued at a real launch; a missed one
    lets the agents widen under a live tree. Fails on a revert to the substring match."""
    script = "src/main/" + CS._TRAINER_SCRIPT
    py = "/opt/conda/envs/e/bin/python3.11"
    yes = {"launcher": [["python3", "-m", "main.launcher", "--steps", "5"], [py, "-u", "-m", "main.launcher"],
                        ["python", "-mmain.launcher"], ["python3", "-um", "main.launcher"],
                        ["python3", "-X", "dev", "-W", "error", "-m", "main.launcher", "--arch", "production"],
                        ["python3", "-Xdev", "-Werror", "-m", "main.launcher"],
                        ["python3", "-m", "main.launcher", "--note", "not a --dry-run-like word"],
                        ["python3", "-m", "main.launcher.__main__", "--steps", "5"], ["python3", "src/main/launcher"],
                        [py, "-u", "/srv/gen3ai/src/main/launcher/__main__.py", "--arch", "production"]],
           "trainer": [["python3", script, "--debug"], [py, "/tmp/launcher-7fe15726-x/" + script, "--model", "m.zip"],
                       ["python3", "-X", "dev", "-u", script], ["python3", "--", script],
                       ["python3", "-m", "main." + CS._TRAINER_SCRIPT[:-3]],
                       ["python3", "-m", "main." + CS._TRAINER_SCRIPT[:-3] + ".__main__"]]}
    for kind, argvs in yes.items():
        for argv in argvs:
            assert CS.is_trainer_argv(argv) == kind, argv
    no = [[], ["python3"], ["python3", "-m"], ["python3", "--"], ["python3", "-"],
          ["grep", "-rn", "-m", "main.launcher", "src"], ["vim", script], ["tail", "-f", script],
          ["bash", "-c", "python3 -m main.launcher"], ["nice", "-n", "19", "python3", "-m", "main.launcher"],
          ["python3", "-c", "import x", "-m", "main.launcher"], ["python3", "-uc", "x", script],
          ["python3", "-m", "pytest", "-m", "main.launcher"], ["python3", "-m", "main.checkargs", script],
          ["python3", "tools/x.py", script], ["python3", "tools/x.py", "-m", "main.launcher"],
          ["python3", "-m", "main.launcher", "--dry-run"], ["python3", "-m", "main.launcher", "--help"],
          ["python3", "-m", "main.launcher", "-h"], ["python3", "-X", "-m", "main.launcher"],
          ["python3", "src/main/not_" + CS._TRAINER_SCRIPT], ["python3", script + ".bak"],
          ["python3", "src/main/launcher", "--dry-run"], ["python3", "src/main/launcher/config.py"],
          ["python3", "src/other/launcher"], ["python3", "launcher"], ["python3", "__main__.py"],
          ["python3", "-m", "other.launcher.__main__"]]
    for argv in no:
        assert CS.is_trainer_argv(argv) is None, argv


def test_the_state_file_is_the_users_whatever_the_environment_says(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two readers with a different ``$HOME`` (a sandboxed shell, a service) must not read two different splits."""
    import pwd
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    want = home / ".local" / "state" / "gen3ai" / "core_split.json"
    for env in ({}, {"HOME": str(tmp_path)}, {"XDG_STATE_HOME": str(tmp_path)}, {"HOME": "", "XDG_STATE_HOME": "x"}):
        assert CS.state_path(env) == want, env
    assert CS.state_path({CS.STATE_ENV: str(tmp_path / "s.json"), "HOME": "/x"}) == tmp_path / "s.json"
    monkeypatch.delenv(CS.STATE_ENV)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert CS.state_path() == want and CS.ab_marker_path() == want.parent / CS.AB_MARKER


def test_a_state_write_is_synced_before_it_replaces_the_file(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    CS.set_split("4+4", live=_nobody)
    assert order == ["fsync", "replace"] and CS.active().name == "4+4"


def test_a_launcher_and_its_trainer_are_live_and_a_dry_run_is_not(tmp_path: Path) -> None:
    trainer = "train_rl_agent" + ".py"
    argvs = {10: ["python3", "-m", "main.launcher", "--steps", "5"],
             11: ["python3", f"/repo/src/main/{trainer}", "--run-dir", "/m/rb_x"],
             12: ["python3", "-m", "main.launcher", "--dry-run", "--steps", "5"],
             13: ["python3", "-m", "pytest", "src/main/launcher"],
             14: ["bash", "-c", "echo -m main.launcher is quoted, not run"],
             15: ["python3", "-m", "main.ops.core_split_ab", "run", "--launcher", "10"]}
    for pid, argv in argvs.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    (tmp_path / "self").mkdir()
    assert CS.live_trainers(str(tmp_path)) == {10: "launcher", 11: "trainer"}
    assert CS.is_trainer_argv(["python3", "-m", "main.launcher"]) == "launcher"
    assert CS.is_trainer_argv(["python3", "-m", "main.launcher_tools"]) is None

    # a CPU-only run on the agent cores is agent work: it blocks no change that keeps its CPUs with the agents
    five, four = CS.parse_split("5+3"), CS.parse_split("4+4")
    aff = {10: list(five.trainer_ids), 11: list(five.agent_ids)}
    assert CS.blocking_trainers(five, four, str(tmp_path), lambda pid: aff[pid]) == {10: "launcher"}
    assert CS.blocking_trainers(five, four, str(tmp_path), lambda pid: list(five.agent_ids)) == {}
    assert set(CS.blocking_trainers(four, five, str(tmp_path), lambda pid: list(four.agent_ids))) == {10, 11}, \
        "a rollback would put the trainer on two of its CPUs"
    assert set(CS.blocking_trainers(five, four, str(tmp_path), lambda pid: None)) == {10, 11}, "unreadable counts"


def test_show_names_the_live_tree_and_flags_one_on_the_agent_cpus(state: Path) -> None:
    quiet = CS.status(live=_nobody)
    assert quiet["split"] == "5+3" and not quiet["state_file_exists"] and quiet["trainers"] == []
    assert "the default: no state file" in CS.render(quiet) and "no training launcher" in CS.render(quiet)
    aff = {4242: list(CS.DEFAULT.trainer_ids), 4250: list(range(16))}
    s = CS.status(live=_a_trainer, affinity=lambda pid: aff[pid])
    assert [(t["pid"], t["is_trainer_list"], t["on_agent_cpus"]) for t in s["trainers"]] == \
        [(4242, True, ""), (4250, False, "5-7,13-15")]
    text = CS.render(s)
    assert "pid 4242: cpus 0-4,8-12: the trainer list" in text and "pid 4250" in text and "MISMATCH" in text
    narrow = CS.status(live=lambda: {7: "trainer"}, affinity=lambda pid: [0, 1, 8, 9])
    assert "narrower than the trainer list" in CS.render(narrow)
    cpu_run = CS.render(CS.status(live=lambda: {7: "trainer"}, affinity=lambda pid: list(CS.DEFAULT.agent_ids)))
    assert "agent work" in cpu_run and "MISMATCH" not in cpu_run
    CS.set_split("3+5", next_launch=True)
    CS.ab_marker_path().write_text(json.dumps({"launcher": 4242}))
    text = CS.render(CS.status(live=_nobody))
    assert "QUEUED for the next launch: 3+5" in text and "A/B MARKER present" in text and "core_split_ab restore" in text


def test_the_cli_is_what_a_shell_script_reads(state: Path, capsys: pytest.CaptureFixture) -> None:
    def run(*argv: str) -> tuple:
        rc = CS.main(list(argv))
        out = capsys.readouterr()
        return rc, out.out.strip(), out.err.strip()

    assert run("agent-cpus") == (0, "5-7,13-15", "") and run("trainer-cpus") == (0, "0-4,8-12", "")
    assert run("trainer-cpus", "--launch") == (0, "0-4,8-12", "")
    assert run("trainer-cpus", "--dry-run")[0] == CS.EXIT_USAGE and run("set", "6+2")[0] == CS.EXIT_USAGE
    rc, out, _ = run("set", "4+4", "--next-launch", "--note", "the A/B passed 4")
    assert rc == 0 and "QUEUED" in out and CS.read_state()["log"][-1]["note"] == "the A/B passed 4"
    rc, out, _ = run("show")
    assert rc == 0 and out.startswith("core split 5+3: trainer 0-4,8-12, agents 5-7,13-15") and "QUEUED" in out
    assert json.loads(run("show", "--json")[1])["next_launch"] == "4+4"


def test_the_module_runs_as_a_script_and_prints_only_the_list_on_stdout(tmp_path: Path) -> None:
    """``$(python -m utils.core_split agent-cpus)`` in a shell: stdout is the list and nothing else."""
    env = dict(os.environ, **{CS.STATE_ENV: str(tmp_path / "s.json")})
    for cmd, want in (("agent-cpus", "5-7,13-15"), ("trainer-cpus", "0-4,8-12")):
        r = subprocess.run([sys.executable, "-m", "utils.core_split", cmd], capture_output=True, text=True, env=env, timeout=60)
        assert (r.returncode, r.stdout, r.stderr) == (0, want + "\n", "")
    assert not list(tmp_path.iterdir())
