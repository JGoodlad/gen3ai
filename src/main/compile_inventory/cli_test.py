"""The compile inventory's CLI names its run and its buffer: no default points at a retired arm."""
import pytest

from main.compile_inventory import __main__ as cli


def test_the_run_stage_requires_a_buffer():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--stage", "trace", "--device", "cpu"])


def test_a_bare_run_is_REFUSED_not_defaulted_to_a_retired_arm(capsys):
    """The old default (`ai_v14_02_lbat_ctrl`, config v123) is pre-era and died at the version gate."""
    a = cli.build_parser().parse_args(["run", "--stage", "trace", "--device", "cpu", "--buffer", "/nowhere.pkl"])
    assert a.run is None and a.model is None
    assert not hasattr(cli, "DEFAULT_RUN") and not hasattr(cli, "DEFAULT_BUFFER")
    assert cli.cmd_run(a) == 2
    assert "no default run" in capsys.readouterr().out
