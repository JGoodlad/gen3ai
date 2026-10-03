"""The win-prob critic's CONFIG surface — the implication, the refusal, and the constants.

`gen3_winprob_critic_mode_v1`; deletion pass P11b deleted `--critic` and the whole reward cluster (`--gamma`,
`--victory-value`, `--draw-penalty`, `--terminal-indicator`): the win-prob critic is the only trainable critic, so
they are CONSTANTS of the namespace — `main/train/parser/objective.py`. The rule
this file exists to keep is the one
`main.train.combination_checks`' own docstring states: **checkargs printing "✓ this command still
launches" on a command `resolve_config` then kills is the whole defect.** So every claim here is
made on the namespace BOTH surfaces build — `resolve_critic_mode` then `desugar_umbrella_flags` —
and the per-rule agreement is already covered by `combination_checks_test`'s parametrized table,
which this change extends rather than duplicates.

What is specific to the critic mode, and therefore lives here:

* the **constants**: no argv and no recorded checkpoint can move the critic, the discount or the terminal off
  the win-prob values on a namespace (a recorded SHAPED checkpoint is REFUSED instead — D4,
  `env_core_switch_test`; a recorded non-production REWARD is refused by `check_reward_config`,
  `reward_defaults_test`);
* the **one implied flag**: `--win-prob-mode` is IMPLIED to `shaping` when untyped, and `none` is refused.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from main.train.combination_checks import failing_checks


def _ns(argv, saved=None):
    """The argv as BOTH surfaces see it before the checks run: parsed, marked, critic-resolved,
    desugared — the same sequence `resolve_config` and `checkargs` run, in the same order."""
    from main.train.config import desugar_umbrella_flags, resolve_critic_mode
    from main.train_rl_agent import build_parser
    parser = build_parser()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        args = parser.parse_args(["--steps", "100"] + argv)
        args._explicit_flags = frozenset(d for d, v in vars(args).items() if v is not None)
        args._saved_config_present = saved is not None
        resolve_critic_mode(args)
        desugar_umbrella_flags(args)
    return args


def _hits(argv, saved=None):
    return [c.name for c in failing_checks(_ns(argv, saved))]


# --------------------------------------------------------------------------------------------
# OFF
# --------------------------------------------------------------------------------------------

def test_the_flagless_namespace_is_the_WINPROB_critic():
    """THE BARE-ARGV DEFAULT (deletion pass D2, 2026-10-02): a fresh argv that types nothing is the
    win-prob critic with its three REQUIRED reward values and its three implied tri-states — the
    production critic, on the production env core."""
    a = _ns([])
    assert a.critic == "winprob"
    assert not hasattr(a, "hand_shaping")   # deleted with the shaped reward path
    assert a.terminal_indicator is True
    assert a.victory_value == 1.0
    assert a.draw_penalty == 0.0
    assert a.win_prob_mode == "shaping"
    assert a.gamma == 1.0
    assert not hasattr(a, "env_core")       # the only core is not a namespace attribute (P11b)


def test_the_critic_is_a_constant_no_argv_or_parent_can_move():
    """`--critic` is deleted (P11b): the namespace carries `winprob` whatever else is typed, a `--model`
    resume included (a recorded shaped parent is refused before any of this runs — D4)."""
    class _SavedShaped:
        critic = "shaped"
    for argv, saved in (([], None), (["--win-prob-mode", "read_only"], None), (["--model", "x.zip"], _SavedShaped())):
        assert _ns(argv, saved=saved).critic == "winprob", argv


def test_a_flagless_run_trips_no_critic_check():
    assert not [h for h in _hits([]) if "winprob" in h or "frozen" in h]


# --------------------------------------------------------------------------------------------
# the IMPLIED flag
# --------------------------------------------------------------------------------------------

def test_winprob_implies_the_head():
    a = _ns([])
    assert a.win_prob_mode == "shaping", "the head must EXIST to be the critic"


def test_an_explicit_win_prob_mode_survives_the_implication():
    """An implication that overwrote a typed flag would make the refusal unreachable and the operator's
    choice (`read_only`: the critic's gradient does not reach the trunk) invisible."""
    assert _ns(["--win-prob-mode", "read_only"]).win_prob_mode == "read_only"


def test_a_recorded_critic_record_reads_shaped_when_it_says_so_or_says_nothing(tmp_path):
    """What a recorded `model_config.json` MEANS did not move with the flag's deletion: a record that says
    `shaped`, and one written before `critic` existed (the key absent), both read `shaped` — so the resume
    meets its refusal (D4) or its pin rather than silently becoming a probability critic."""
    import json

    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from main.train.rust_env_setup import recorded_critic
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    data = json.loads(ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512], "critic": "shaped"}).to_json())
    for name, mutate in (("recorded", lambda d: d), ("absent", lambda d: d.pop("critic") and d)):
        cfg = dict(data)
        mutate(cfg)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(cfg))
        saved = ModelVersion.from_json_file(str(path))
        assert recorded_critic(None, saved_ver=saved) == "shaped", name


# --------------------------------------------------------------------------------------------
# the terminal's CONSTANTS (they were REQUIRED by four combination rows until P11b)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [["--terminal-indicator"], ["--no-terminal-indicator"], ["--victory-value", "1.0"],
                                  ["--victory-value", "30"], ["--draw-penalty", "0"], ["--draw-penalty", "-35"],
                                  ["--gamma", "1.0"], ["--gamma", "0.99"], ["--critic", "winprob"]])
def test_no_terminal_critic_or_discount_value_can_be_typed(argv, capsys):
    """Each used to be refused (or required) by its own `winprob_critic_*` row; the flags are gone, so the
    value is fixed by construction and a typed one is refused with the reason."""
    from main.train_rl_agent import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "100"] + argv)
    assert "was DELETED" in capsys.readouterr().err


def test_the_winprob_terminal_rows_are_gone_from_the_combination_list():
    from main.train.combination_checks import BY_NAME
    for name in ("winprob_critic_needs_the_indicator_terminal", "winprob_critic_needs_unit_victory_value",
                 "winprob_critic_refuses_draw_penalty", "winprob_critic_needs_unit_gamma",
                 "winprob_strata_needs_the_winprob_critic", "fork_needs_the_winprob_critic"):
        assert name not in BY_NAME, name
    assert "winprob_critic_needs_a_head" in BY_NAME


# --------------------------------------------------------------------------------------------
# the refusals that are about a missing head
# --------------------------------------------------------------------------------------------

def test_winprob_refuses_a_missing_head():
    assert "winprob_critic_needs_a_head" in _hits(["--win-prob-mode", "none"])
