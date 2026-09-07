"""`--critic`'s CONFIG surface — the implications, the refusals, and the OFF path.

`gen3_winprob_critic_mode_v1`. The rule this file exists to keep is the one
`main.train.combination_checks`' own docstring states: **checkargs printing "✓ this command still
launches" on a command `resolve_config` then kills is the whole defect.** So every claim here is
made on the namespace BOTH surfaces build — `resolve_critic_mode` then `desugar_umbrella_flags` —
and the per-rule agreement is already covered by `combination_checks_test`'s parametrized table,
which this change extends rather than duplicates.

What is specific to the critic mode, and therefore lives here:

* the **asymmetry between implied and required**. Three flags are implied because their argparse
  default is `None`, so "unset" is representable; four reward flags are REQUIRED because theirs is
  concrete, so an implication would silently overwrite a typed value and the refusal meant to
  catch a conflicting one could never fire. That asymmetry is a property of the flag surface and
  will read as an inconsistency to anyone who does not know why — pin it with the reason.
* the **inheritance ORDER**. `resolve_critic_mode` runs before the `_resolve` sweep, so a fork of
  a `shaped` parent cannot inherit that parent's `use_popart=True` / `win_prob_mode='none'` into a
  `winprob` run. Off by one call and the mode is broken by a value nobody typed.
* **OFF is byte-identical at the namespace**, which is the cheapest place to catch an implication
  that leaked into the default path.
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
        resolve_critic_mode(args, saved)
        desugar_umbrella_flags(args)
    return args


def _hits(argv, saved=None):
    return [c.name for c in failing_checks(_ns(argv, saved))]


# --------------------------------------------------------------------------------------------
# OFF
# --------------------------------------------------------------------------------------------

def test_the_flagless_namespace_is_unchanged():
    """A run that does not type `--critic` must be byte-identical at the namespace: same critic,
    same reward composition, same discount, same PopArt. This is where an implication that leaked
    out of the `winprob` branch shows up first."""
    a = _ns([])
    assert a.critic == "shaped"
    assert a.hand_shaping is True
    assert a.terminal_indicator is False
    assert a.no_progress_tax_armed is False
    assert a.victory_value == 30.0
    assert a.draw_penalty == -35.0
    assert a.win_prob_mode is None      # still the sentinel; `_resolve` fills it to 'none'
    assert a.use_popart is None
    assert a.gamma is None


def test_an_explicit_shaped_is_the_same_namespace_as_no_flag():
    a, b = _ns([]), _ns(["--critic", "shaped"])
    ignore = {"_explicit_flags"}
    assert {k: v for k, v in vars(a).items() if k not in ignore} == \
           {k: v for k, v in vars(b).items() if k not in ignore | {"critic"}} | {"critic": "shaped"}


def test_a_flagless_run_trips_no_critic_check():
    assert not [h for h in _hits([]) if "winprob" in h or "frozen" in h]


# --------------------------------------------------------------------------------------------
# the IMPLIED three
# --------------------------------------------------------------------------------------------

def test_winprob_implies_the_three_tristate_flags():
    a = _ns(["--critic", "winprob"])
    assert a.win_prob_mode == "shaping", "the head must EXIST to be the critic"
    assert a.gamma == 1.0, "V(s) == P(win|s) holds exactly only at gamma 1"
    assert a.use_popart is False


@pytest.mark.parametrize("argv,dest,value", [
    (["--win-prob-mode", "read_only"], "win_prob_mode", "read_only"),
    (["--gamma", "0.99"], "gamma", 0.99),
])
def test_an_explicit_value_survives_the_implication(argv, dest, value):
    """An implication that overwrote a typed flag would make the refusals unreachable and the
    operator's choice invisible — which is exactly why the four concrete-default reward flags are
    NOT implied (see the test below)."""
    assert getattr(_ns(["--critic", "winprob"] + argv), dest) == value


def test_the_implication_runs_BEFORE_inheritance():
    """A fork of a `shaped` parent must not inherit `use_popart=True` into a `winprob` run.

    `resolve_critic_mode` sets the value while it is still the `None` sentinel, so the later
    `_resolve` sweep finds it filled and inherits nothing. One call later in the order and the
    mode would be broken by a value nobody typed, on the very command shape (a fork) the mode is
    most likely to be launched as."""
    class _Saved:
        use_popart = True
        win_prob_mode = "none"
        critic = "shaped"
    a = _ns(["--critic", "winprob", "--model", "x.zip"], saved=_Saved())
    assert a.use_popart is False
    assert a.win_prob_mode == "shaping"


def test_the_critic_itself_is_INHERITED_on_a_flagless_resume():
    """It is STRUCTURAL and resume-immutable, so a flagless resume must keep the parent's mode —
    otherwise every restart of a winprob run would FATAL at check_compatible."""
    class _Saved:
        critic = "winprob"
    a = _ns(["--model", "x.zip"], saved=_Saved())
    assert a.critic == "winprob"


# --------------------------------------------------------------------------------------------
# the REQUIRED four, and why they are not implied
# --------------------------------------------------------------------------------------------

_REQUIRED = {
    "winprob_critic_needs_no_hand_shaping": ["--no-hand-shaping"],
    "winprob_critic_needs_the_indicator_terminal": ["--terminal-indicator"],
    "winprob_critic_needs_unit_victory_value": ["--victory-value", "1.0"],
    "winprob_critic_refuses_draw_penalty": ["--draw-penalty", "0"],
}


@pytest.mark.parametrize("check", sorted(_REQUIRED))
def test_each_required_reward_flag_is_refused_when_missing(check):
    """These four have CONCRETE argparse defaults, so `resolve_critic_mode` cannot tell "left
    alone" from "typed the default" and refuses to guess. Each is required by its own check, and
    each message names the flag to pass — this tree's standing preference for a
    composition-changing combination (`--use-popart` requires an explicit `--clip-range-vf none`
    for exactly the same reason)."""
    assert check in _hits(["--critic", "winprob"])


def test_the_full_required_set_launches_clean():
    """The command the design's §5.4 launch line is made of must trip NOTHING."""
    argv = ["--critic", "winprob"]
    for flags in _REQUIRED.values():
        argv += flags
    assert not [h for h in _hits(argv) if "winprob" in h or "frozen" in h]


def test_none_of_the_four_is_silently_overwritten():
    """The positive half: a typed value reaches the checks unchanged, so a conflicting one is
    REPORTED rather than replaced."""
    a = _ns(["--critic", "winprob", "--victory-value", "7.5", "--draw-penalty", "-3"])
    assert a.victory_value == 7.5 and a.draw_penalty == -3.0
    hits = _hits(["--critic", "winprob", "--victory-value", "7.5", "--draw-penalty", "-3"])
    assert "winprob_critic_needs_unit_victory_value" in hits
    assert "winprob_critic_refuses_draw_penalty" in hits


# --------------------------------------------------------------------------------------------
# the refusals that are about a SECOND critic, or a subsumed knob
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv,check", [
    (["--win-prob-mode", "none"], "winprob_critic_needs_a_head"),
    (["--use-popart"], "winprob_critic_refuses_popart"),
    (["--value-from-dist"], "winprob_critic_refuses_value_from_dist"),
    (["--win-prob-coef", "1.0"], "winprob_critic_refuses_win_prob_coef"),
    (["--value-tail-weight", "0.3"], "winprob_critic_refuses_value_tail_weight"),
    (["--win-prob-pbrs-coef", "0.5"], "winprob_critic_refuses_self_phi_pbrs"),
    (["--win-prob-pbrs-source", "models/p.zip"], "winprob_critic_refuses_self_phi_source"),
    (["--value-dist-mode", "read_only", "--value-dist-bins", "51",
      "--value-dist-vmin", "-12", "--value-dist-vmax", "12"],
     "winprob_critic_refuses_value_dist"),
])
def test_each_incompatible_flag_is_refused(argv, check):
    base = ["--critic", "winprob", "--no-hand-shaping", "--terminal-indicator",
            "--victory-value", "1.0", "--draw-penalty", "0"]
    assert check in _hits(base + argv)


def test_the_value_dist_refusal_is_on_the_RESOLVED_mode_not_a_typed_flag():
    """A fork whose PARENT recorded `value_dist_mode='shaping'` inherits it, and the PPO CE gate /
    the grad-balance value term / the prober's awareness stack all read that MODE STRING rather
    than the head — so a run that merely never typed the flag would still mis-state its config to
    all three. The check therefore reads the resolved value, not `_typed`."""
    class _Saved:
        value_dist_mode = "shaping"
        critic = "winprob"
    a = _ns(["--critic", "winprob", "--model", "x.zip"], saved=_Saved())
    # `resolve_config`'s `_resolve` sweep is what inherits it; emulate that one line here.
    a.value_dist_mode = "shaping"
    assert "winprob_critic_refuses_value_dist" in [c.name for c in failing_checks(a)]


def test_win_prob_pbrs_frozen_is_BUILDABLE_under_the_winprob_critic():
    """gen3_frozen_phi_actor_only_v1 lifted the hold. The rung the owner amendment kept ONE EDIT
    away is now that edit: under `winprob` the flag must raise no refusal of its own, so a
    FROZEN-phi arm is a launch rather than a message."""
    hits = _hits(["--critic", "winprob", "--no-hand-shaping", "--terminal-indicator",
                  "--victory-value", "1.0", "--draw-penalty", "0",
                  "--win-prob-pbrs-frozen", "models/p.zip"])
    assert "win_prob_pbrs_frozen_needs_the_winprob_critic" not in hits
    assert "win_prob_pbrs_frozen_needs_a_head" not in hits


def test_win_prob_pbrs_frozen_is_REFUSED_under_the_shaped_critic():
    """The surviving refusal is a ROUTING answer, not a deferral: under `shaped` the critic
    predicts a PopArt-normalized shaped return, so phi is in different units and the dose is a real
    question the shaped ladder's own coefficient exists to ask."""
    hits = _hits(["--win-prob-pbrs-frozen", "models/p.zip"])
    assert "win_prob_pbrs_frozen_needs_the_winprob_critic" in hits


def test_the_shaped_refusal_ROUTES_rather_than_deferring():
    from main.train.combination_checks import BY_NAME
    text = BY_NAME["win_prob_pbrs_frozen_needs_the_winprob_critic"].text(_ns([]))
    assert "--win-prob-pbrs-coef" in text and "--win-prob-pbrs-source" in text, (
        "the refusal must name the flags that DO work under `shaped`")
    assert "ACTOR-ONLY" in text
    assert "1.0" in text, "the currency-matched coefficient must be stated, not left to be derived"


def test_the_frozen_flag_still_needs_a_win_prob_head():
    hits = _hits(["--win-prob-pbrs-frozen", "models/p.zip", "--win-prob-mode", "none"])
    assert "win_prob_pbrs_frozen_needs_a_head" in hits


def test_the_self_phi_refusal_states_the_double_counting():
    from main.train.combination_checks import BY_NAME
    text = BY_NAME["winprob_critic_refuses_self_phi_pbrs"].text(_ns([]))
    assert "no coefficient" in text and "currency-matched" in text
    assert "DOUBLE COUNTING" in text
