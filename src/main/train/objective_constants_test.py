"""The objective's CONSTANTS (deletion pass P11b, and the no-progress clock's two switches in P11d): flags that had one legal value, now a namespace default.

Pinned here, each failing on a revert of the piece it names:

* every constant is on EVERY namespace `build_parser()` parses (`RewardConfig.from_args` falls back to the
  dataclass's shaped-era defaults for an attribute that is absent, so a missing one would silently train the
  wrong reward — the reason the constants are not "readers use a literal");
* none of them is an OPTION any more (nothing can type it), and each typed spelling is refused WITH its
  reason, read from `designs/deleted_flags.md`;
* a typed value never reaches the namespace, and a recorded checkpoint cannot move one (`config` never
  inherits them) — what an old checkpoint recorded is judged where it is read (D4 for the critic).
"""
from __future__ import annotations

import pytest

from agents.model.critic_mode import CRITIC_DEFAULT
from main.train.parser import build_parser
from main.train.parser.objective import OBJECTIVE_CONSTANTS
from utils.paths import repo_path

#: constant -> the deleted flag that used to type it (the census / `deleted_flags.md` name)
_FLAG = {"critic": "--critic", "gamma": "--gamma", "victory_value": "--victory-value", "draw_penalty": "--draw-penalty",
         "terminal_indicator": "--terminal-indicator", "progress_decision_tense": "--progress-decision-tense",
         "progress_switch_freeze": "--progress-switch-freeze"}


def test_every_constant_is_on_every_namespace():
    ns = build_parser().parse_args(["--steps", "1"])
    for dest, value in OBJECTIVE_CONSTANTS.items():
        assert getattr(ns, dest) == value, dest
    assert ns.critic == CRITIC_DEFAULT == "winprob"


def test_no_constant_is_an_option_and_each_names_a_deleted_flag_row():
    options = {o for a in build_parser()._actions for o in a.option_strings}
    text = repo_path("designs", "deleted_flags.md").read_text(encoding="utf-8")
    for dest in OBJECTIVE_CONSTANTS:
        flag = _FLAG[dest]
        assert flag not in options, f"{flag} is an option again — a constant must not be typeable"
        # section 1 (`| `--flag` |`), or section 1c for a trainer flag another tool's parser still declares
        # (`| trainer --flag |`; `--gamma` sat there until P6 slice 6d-1 deleted the `cf_producer` parser that still declared it)
        assert f"| `{flag}` |" in text or f"| trainer {flag} |" in text, f"{flag} has no designs/deleted_flags.md row"
    assert set(_FLAG) == set(OBJECTIVE_CONSTANTS), "a constant with no declared deleted flag"


@pytest.mark.parametrize("dest", sorted(OBJECTIVE_CONSTANTS))
def test_a_typed_flag_is_refused_with_its_reason_and_leaves_the_constant_alone(dest, capsys):
    flag = _FLAG[dest]
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--steps", "1", flag, "1"])
    assert e.value.code == 2
    assert f"{flag} was DELETED" in capsys.readouterr().err


def test_the_reward_config_a_launch_builds_is_the_winprob_one_never_the_dataclass_fallback():
    """The silent-wrong-objective guard: absent attributes would read the shaped-era dataclass defaults
    (victory 30.0, draw -35.0, signed terminal) — the constants must keep `from_args` on production's."""
    from agents.training.reward_config import RewardConfig

    rc = RewardConfig.from_args(build_parser().parse_args(["--steps", "1"]))
    assert (rc.victory_value, rc.draw_penalty, rc.terminal_indicator) == (1.0, 0.0, True)
