"""The launch-diff gate: an unacknowledged behavioural difference must FAIL the launch."""
import json
import pytest
from main.launchdiff import parse_flags, report, load_reference


def _ref(tmp_path, cmd):
    d = tmp_path / "run"; d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"launcher_command": cmd}))
    return str(d)


def test_parses_bare_and_valued_flags():
    f = parse_flags(["--self-play", "--n-envs", "48", "--all-shaping-pbrs"])
    assert f == {"--self-play": "ON", "--n-envs": "48", "--all-shaping-pbrs": "ON"}


def test_a_dropped_flag_is_reported_and_gates(capsys, tmp_path):
    ref = _ref(tmp_path, "x --all-shaping-pbrs --n-envs 48")
    rc = report(load_reference(ref), parse_flags(["--n-envs", "48"]), acked=set())
    assert rc == 1, "dropping --all-shaping-pbrs must FAIL the launch"
    assert "all-shaping-pbrs" in capsys.readouterr().out


def test_acknowledging_it_passes(tmp_path):
    ref = _ref(tmp_path, "x --all-shaping-pbrs --n-envs 48")
    assert report(load_reference(ref), parse_flags(["--n-envs", "48"]),
                  acked={"all-shaping-pbrs"}) == 0


def test_launcher_owned_and_per_run_flags_do_not_gate(tmp_path):
    ref = _ref(tmp_path, "x --n-envs 48 --run-name a --steps 100 --nice 10")
    assert report(load_reference(ref),
                  parse_flags(["--n-envs", "48", "--run-name", "b", "--steps", "200"]),
                  acked=set()) == 0


def test_a_changed_VALUE_gates_too(tmp_path):
    ref = _ref(tmp_path, "x --draw-penalty -35")
    assert report(load_reference(ref), parse_flags(["--draw-penalty", "-30"]), acked=set()) == 1
    assert report(load_reference(ref), parse_flags(["--draw-penalty", "-30"]),
                  acked={"draw-penalty"}) == 0


def test_identical_commands_have_nothing_to_acknowledge(tmp_path):
    ref = _ref(tmp_path, "x --n-envs 48 --self-play")
    assert report(load_reference(ref), parse_flags(["--n-envs", "48", "--self-play"]),
                  acked=set()) == 0


def test_a_reference_without_launcher_command_is_refused(tmp_path):
    d = tmp_path / "r"; d.mkdir(); (d / "metadata.json").write_text("{}")
    with pytest.raises(SystemExit):
        load_reference(str(d))
