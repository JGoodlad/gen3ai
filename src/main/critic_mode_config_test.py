"""The win-prob critic's CONFIG surface — the implications, the refusals, and the constant.

`gen3_winprob_critic_mode_v1`; `--critic` itself was deleted in deletion pass P11b (the win-prob critic is the
only trainable critic, so `critic` is a CONSTANT of the namespace — `main/train/parser/objective.py`). The rule
this file exists to keep is the one
`main.train.combination_checks`' own docstring states: **checkargs printing "✓ this command still
launches" on a command `resolve_config` then kills is the whole defect.** So every claim here is
made on the namespace BOTH surfaces build — `resolve_critic_mode` then `desugar_umbrella_flags` —
and the per-rule agreement is already covered by `combination_checks_test`'s parametrized table,
which this change extends rather than duplicates.

What is specific to the critic mode, and therefore lives here:

* the **asymmetry between implied and required**. Two flags are implied because their argparse
  default is `None`, so "unset" is representable; three reward flags are REQUIRED because theirs is
  concrete, so an implication would silently overwrite a typed value and the refusal meant to
  catch a conflicting one could never fire. That asymmetry is a property of the flag surface and
  will read as an inconsistency to anyone who does not know why — pin it with the reason.
* the **constant**: no argv and no recorded checkpoint can move `critic` off `winprob` on a namespace
  (a recorded SHAPED checkpoint is REFUSED instead — D4, `env_core_switch_test`).
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
# the IMPLIED two
# --------------------------------------------------------------------------------------------

def test_winprob_implies_the_two_tristate_flags():
    a = _ns([])
    assert a.win_prob_mode == "shaping", "the head must EXIST to be the critic"
    assert a.gamma == 1.0, "V(s) == P(win|s) holds exactly only at gamma 1"


@pytest.mark.parametrize("argv,dest,value", [
    (["--win-prob-mode", "read_only"], "win_prob_mode", "read_only"),
    (["--gamma", "0.99"], "gamma", 0.99),
])
def test_an_explicit_value_survives_the_implication(argv, dest, value):
    """An implication that overwrote a typed flag would make the refusals unreachable and the
    operator's choice invisible — which is exactly why the three concrete-default reward flags are
    NOT implied (see the test below)."""
    assert getattr(_ns(argv), dest) == value


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
# the REQUIRED three, and why they are not implied
# --------------------------------------------------------------------------------------------

_REQUIRED = {
    "winprob_critic_needs_the_indicator_terminal": ["--terminal-indicator"],
    "winprob_critic_needs_unit_victory_value": ["--victory-value", "1.0"],
    "winprob_critic_refuses_draw_penalty": ["--draw-penalty", "0"],
}


#: The SIGNED terminal's values, typed — what each REQUIRED check refuses under winprob.
_SIGNED = {
    "winprob_critic_needs_the_indicator_terminal": ["--no-terminal-indicator"],
    "winprob_critic_needs_unit_victory_value": ["--victory-value", "30"],
    "winprob_critic_refuses_draw_penalty": ["--no-terminal-indicator", "--draw-penalty", "-35"],
}


@pytest.mark.parametrize("check", sorted(_SIGNED))
def test_each_required_reward_flag_is_refused_when_the_signed_value_is_typed(check):
    """These three have CONCRETE argparse defaults — since the bare-argv flip (2026-10-02) exactly
    winprob's values, so a bare argv passes — and `resolve_critic_mode` cannot tell "left alone" from
    "typed the default", so it never implies them. A typed signed-terminal value is REFUSED by its own
    check, whose message names the flag to pass."""
    assert check in _hits(_SIGNED[check])
    assert check not in _hits([])


def test_the_full_required_set_launches_clean():
    """The command the design's §5.4 launch line is made of must trip NOTHING."""
    argv = []
    for flags in _REQUIRED.values():
        argv += flags
    assert not [h for h in _hits(argv) if "winprob" in h or "frozen" in h]


def test_none_of_the_three_is_silently_overwritten():
    """The positive half: a typed value reaches the checks unchanged, so a conflicting one is
    REPORTED rather than replaced."""
    a = _ns(["--victory-value", "7.5", "--draw-penalty", "-3"])
    assert a.victory_value == 7.5 and a.draw_penalty == -3.0
    hits = _hits(["--victory-value", "7.5", "--draw-penalty", "-3"])
    assert "winprob_critic_needs_unit_victory_value" in hits
    assert "winprob_critic_refuses_draw_penalty" in hits


# --------------------------------------------------------------------------------------------
# the refusals that are about a missing head
# --------------------------------------------------------------------------------------------

def test_winprob_refuses_a_missing_head():
    base = ["--terminal-indicator", "--victory-value", "1.0", "--draw-penalty", "0"]
    assert "winprob_critic_needs_a_head" in _hits(base + ["--win-prob-mode", "none"])
