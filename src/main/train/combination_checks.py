"""COMBINATION CHECKS — the value-conditional refusals, in ONE place, read by BOTH surfaces.

WHY THIS MODULE EXISTS. `agents.model.flag_registry`'s `requires` graph expresses one shape of
dependency — *flag A must be ENABLED for flag B to be enabled* — and `main.checkargs` reads it, so
an unsatisfiable structural combination is reported offline instead of crashing inside
`Gen3FeaturesExtractor.__init__`. But some launch-time refusals are not that shape: they are
value-conditional (*`--distill-target action` requires `--distill-coef` > 0*), they lived in
`main.train.config.resolve_config` as `parser.error` lines, and nothing outside that function knew
them.

That gap has now cost two launches, and the SECOND one is why this module is no longer a list of
four. C1 (2026-09-01) forked a parent whose recorded config carried `distill_target="action"`,
passed `--distill-coef 0`, and did NOT name `--distill-target` — so `_resolve` inherited `action`,
the check fired, and the run died at launch while `checkargs` had said "this command still
launches". The fix moved *that* rule here and left every sibling behind, so the module's own
premise — "one list, both readers" — held only for the four rules that had been migrated. G5
(2026-09-06) then died three times in a row on the ones that had not: `--distill-anchor-monitor`
at `--distill-coef 0`, `--distill-team-bias` with no teacher, and finally the inherited
`--distill-target action`. A partial single-source is a single-source nobody can trust, because
nothing about the output distinguishes "checked and clean" from "never asked".

So the list is now EXHAUSTIVE over its class, and that is enforced rather than intended:
`combination_checks_test.py` walks `config.py`'s own AST — resolving local aliases, which is how
`_anchor_wanted` and `_items` hid three refusals from an earlier reading — and FAILS, naming
file:line, if any `parser.error` remains whose guard reads a second flag's value. The allowlist
carries a reason string per entry.

THE CONTRACT. A check is a pure predicate over an args-shaped namespace plus the message the LAUNCH
path prints. `resolve_config` calls `refuse_first`, which renders that message through the check's
own exit style; `main.checkargs` calls `failing_checks` on the EFFECTIVE namespace (argv overlaid
on the fork parent's recorded config) and reports every one. Neither owns the rule.

WHAT BELONGS HERE: a refusal that reads two or more RESOLVED values and says one combination is
incoherent — including a range check that only applies in a mode (`--exploiter-ladder-gate` under
`--exploiter-ladder`), because "which flag turns this on" is itself a cross-flag fact. What does
NOT: a range check on a single value (`--distill-topk >= 1`), which argparse's caller can answer
from the one value it has; and anything needing the parser, a torch import, an env var, or a
teacher spec's filesystem — those stay in `resolve_config`, which has them, and are listed with
their reason in the test's allowlist.

ORDERING. `resolve_config` evaluates the whole list at ONE point, late in validation, and refuses
on the first failure in DECLARATION order — which is source order as the checks stood before the
migration. For an argv that trips exactly one rule (every real launch failure so far) the message
and the exit path are byte-identical to what shipped. For an argv broken two ways, WHICH message
comes first can differ from the pre-migration order, because the single-value range checks now all
run before the sweep; the test table pins the one-defect case, which is the contract.

EXPLICITNESS. Several refusals fire only on a value the operator actually TYPED (`--distill-team-bias`
with no teacher; the anchor knobs that do nothing). On the launch path that is captured before
`_resolve` fills defaults, as `args._explicit_flags`; `main.checkargs` snapshots the same set before
it inherits from the parent config. `_typed` reads that marker and falls back to `is not None` when
it is absent, so a bare namespace still gets an honest answer.
"""
from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

# The rule is DECLARED in the resolver that also applies it; referenced here, never re-typed.
from main.train.compile_flags import _PRELOAD_WITHOUT_OPPONENTS


class CombinationCheck(NamedTuple):
    """One value-conditional refusal. `predicate` is TRUE when the combination is BROKEN."""

    name: str
    dests: Tuple[str, ...]          # the args attributes it reads — checkargs prints their provenance
    predicate: Callable[[Any], bool]
    #: The launch path's message: a literal, or a renderer for the ones that quote a value.
    message: Union[str, Callable[[Any], str]]
    #: How `resolve_config` refuses. "parser" = `parser.error` (exit 2); "exit1" = print + exit 1;
    #: "fatal_config" = print to stderr + exit `TrainExitCode.FATAL_CONFIG`, for a config a restart
    #: would hit identically.
    exit_style: str = "parser"
    #: Information only the LAUNCH path is guaranteed to have. A surface that lacks it reports the
    #: finding as ADVISORY rather than dropping it silently. Currently one value: "saved_config"
    #: (the resumed checkpoint's recorded `model_config.json` was read).
    needs: Tuple[str, ...] = ()

    def text(self, args) -> str:
        """The message as the launch path prints it."""
        return self.message(args) if callable(self.message) else self.message


# --------------------------------------------------------------------------------------------
# Predicate helpers. Each reads only the namespace, so both surfaces get the same answer.
# --------------------------------------------------------------------------------------------

def _positive(value: Any) -> bool:
    """The `args.distill_coef and args.distill_coef > 0` idiom the launch path uses, None-safe."""
    return bool(value) and float(value) > 0.0


def _typed(args, dest: str) -> bool:
    """Did the operator actually TYPE this flag? (vs. inherit it, or land on a default.)

    `resolve_config` stamps `_explicit_flags` before `_resolve` fills anything; `main.checkargs`
    stamps it before it inherits from the parent's recorded config. With no marker the honest
    fallback is the tri-state sentinel the flags themselves use.
    """
    marker = getattr(args, "_explicit_flags", None)
    if marker is None:
        return getattr(args, dest, None) is not None
    return dest in marker


def _val(args, dest: str, default: Any) -> Any:
    """The value a launch would RESOLVE for `dest` — the default when the flag is still unset.

    Every one of these mirrors a `_resolve(name, default)` line in `main.train.config`. On the
    LAUNCH path the resolve has already run, so this returns the value unchanged; on `checkargs`'
    namespace an unset flag is still `None`, and reading `None != "none"` as "the mode is on" is a
    FALSE POSITIVE that reported two nonexistent problems the first time this list was widened.
    """
    value = getattr(args, dest, None)
    return default if value is None else value


def _winprob(args) -> bool:
    """Is this the WIN-PROB critic? Read through the ONE predicate, never a string compare here."""
    from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
    return is_winprob(_val(args, "critic", CRITIC_DEFAULT))


def _dist_mode(args) -> str:
    return _val(args, "value_dist_mode", "none")


def _belief_mode(args) -> str:
    return _val(args, "move_belief_mode", "off")


def _teacher_items(args) -> List[str]:
    """The `--distill-teacher` spec split the way `resolve_config` splits it, before parsing."""
    return [x.strip() for x in (getattr(args, "distill_teacher", None) or "").split(",") if x.strip()]


def _grad_project(args) -> bool:
    return getattr(args, "distill_anchor_mode", None) == "grad_project"


def _anchor_wanted(args) -> bool:
    """The anchor machinery is attached: a live coefficient, the monitor, or `grad_project`.

    An UNSET monitor is not the same as an off one — a fold defaults it ON — so the launch path's
    own `default_anchor_monitor` decides that case, rather than a second copy of the rule here.
    """
    from main.train.config import default_anchor_monitor          # local: config imports us
    monitor = getattr(args, "distill_anchor_monitor", None)
    if monitor is None:
        monitor = default_anchor_monitor(args)
    return bool(_positive(getattr(args, "distill_anchor_coef", None))
                or monitor
                or _grad_project(args))


def _stop_on(args) -> bool:
    stop = getattr(args, "distill_stop", None)
    return bool(stop) and stop != "off"


def _dual_on(args) -> bool:
    return _positive(getattr(args, "distill_anchor_target_kl", None))


def _q_live(args) -> bool:
    return ((getattr(args, "q_winprob_coef", None) or 0.0) > 0.0
            or (getattr(args, "q_winprob_onpolicy_coef", None) or 0.0) > 0.0)


def _adaptive_on(args) -> bool:
    return _val(args, "adaptive_batch", "off") != "off"


def _exploiter_temp_on(args) -> bool:
    return getattr(args, "exploiter_temp_start", None) is not None


def _edge_families(args) -> Optional[set]:
    """The `--edge-bias-families` set, or None when the flag is off.

    The VOCABULARY check (is `q` a real family?) stays in `resolve_config`: it reads
    `agents.model.features_extractor._EDGE_FAMILIES`, which imports torch, and `main.checkargs`
    promises not to. Every CROSS-FLAG requirement of a family is here, and needs no vocabulary.
    """
    ebf = getattr(args, "edge_bias_families", None)
    if not ebf or ebf == "off":
        return None
    return {"d1", "d3"} if ebf == "d" else set(str(ebf).split(","))


def _fams(args, wanted: set) -> bool:
    fams = _edge_families(args)
    return bool(fams and (fams & wanted))


def _anchor_knob_typed(args) -> bool:
    return bool(_typed(args, "distill_anchor_mode")
                or _typed(args, "distill_anchor_ref")
                or _typed(args, "distill_anchor_ema_tau")
                or _typed(args, "distill_anchor_refresh_every")
                or getattr(args, "distill_anchor_parent", None) is not None)


def _dual_knob_typed(args) -> bool:
    return bool(_typed(args, "distill_anchor_dual_lr")
                or _typed(args, "distill_anchor_coef_min")
                or _typed(args, "distill_anchor_coef_max"))


def _stop_knob_typed(args) -> bool:
    return bool(_typed(args, "distill_stop_window")
                or _typed(args, "distill_stop_eps")
                or _typed(args, "distill_stop_kl_slope")
                or _typed(args, "distill_stop_persist")
                or _typed(args, "distill_stop_anneal_factor"))


def _cf_duty_cycle_starved(args) -> bool:
    """The counterfactual label path is starved BY CONSTRUCTION — see `_announce_cf_duty_cycle`."""
    on = (_positive(getattr(args, "cf_twin_coef", None))
          or _positive(getattr(args, "cf_winprob_coef", None)))
    if not (on and getattr(args, "cf_records", None)) or getattr(args, "debug", False):
        return False
    from main.train.constants import (CF_DUTY_CYCLE_FLOOR, cf_label_duty_cycle,
                                      checkpoint_interval_env_steps)
    interval = checkpoint_interval_env_steps(getattr(args, "checkpoint_every_steps", None),
                                             int(args.n_envs))
    return cf_label_duty_cycle(args.cf_label_lag_steps, interval) < CF_DUTY_CYCLE_FLOOR


def _cf_duty_cycle_message(args) -> str:
    from main.train.constants import (CF_DUTY_CYCLE_FLOOR, cf_label_duty_cycle,
                                      checkpoint_interval_env_steps,
                                      checkpoint_save_freq_vec_calls)
    n_envs = int(args.n_envs)
    every = getattr(args, "checkpoint_every_steps", None)
    vec_calls = checkpoint_save_freq_vec_calls(every, n_envs)
    interval = checkpoint_interval_env_steps(every, n_envs)
    duty = cf_label_duty_cycle(args.cf_label_lag_steps, interval)
    shown = ("unbounded (--cf-label-lag-steps 0 = labels never expire)" if duty == float("inf")
             else f"{duty:.1%}")
    return (
        f"\n[CF] FATAL: the counterfactual label path is STARVED BY CONSTRUCTION.\n"
        f"  --cf-label-lag-steps         : {args.cf_label_lag_steps:,} env steps\n"
        f"  checkpoint interval          : {interval:,} env steps "
        f"({vec_calls:,} vec-calls x {n_envs} envs)\n"
        f"  --checkpoint-every-steps     : "
        f"{'(unset — the 50000-vec-call default)' if every is None else format(every, ',')}\n"
        f"  => DUTY CYCLE                : {shown}  (floor {CF_DUTY_CYCLE_FLOOR:.0%})\n"
        f"  The producer stamps every label with the newest checkpoint's step, so outside that\n"
        f"  window EVERY label it writes is expired by the buffer on arrival. Two remedies, and\n"
        f"  either alone is enough:\n"
        # Both remedies are printed WITHOUT thousands separators: they are copy-pasteable argv
        # values, and `--checkpoint-every-steps 600,000` is an argparse error.
        f"    * checkpoint MORE OFTEN: --checkpoint-every-steps "
        f"{max(1, int(args.cf_label_lag_steps / CF_DUTY_CYCLE_FLOOR))} or less\n"
        f"    * widen the staleness bound: --cf-label-lag-steps "
        f"{max(1, int(CF_DUTY_CYCLE_FLOOR * interval))} or more (a label then supervises a\n"
        f"      policy further from the one that produced it — the cost this bound exists to cap)\n")


# --------------------------------------------------------------------------------------------
# THE LIST. Declaration order is the source order these refusals had inside `resolve_config`.
# --------------------------------------------------------------------------------------------

COMBINATION_CHECKS: Tuple[CombinationCheck, ...] = (

    # ---- the --adaptive-batch family: range checks that only exist in a mode -------------------
    CombinationCheck(
        "adaptive_batch_target_positive", ("adaptive_batch", "adaptive_batch_target"),
        lambda a: _adaptive_on(a) and a.adaptive_batch_target <= 0.0,
        "--adaptive-batch-target must be > 0 (it is a noise-scale RATIO setpoint)"),
    CombinationCheck(
        "adaptive_batch_band_above_one", ("adaptive_batch", "adaptive_batch_band"),
        lambda a: _adaptive_on(a) and a.adaptive_batch_band <= 1.0,
        "--adaptive-batch-band must be > 1 — it is a MULTIPLICATIVE no-op band "
        "[target/band, target*band]; 1.0 would make every reading out of band"),
    CombinationCheck(
        "adaptive_batch_min_accum", ("adaptive_batch", "adaptive_batch_min_accum"),
        lambda a: _adaptive_on(a) and a.adaptive_batch_min_accum < 1,
        "--adaptive-batch-min-accum must be >= 1"),
    CombinationCheck(
        "adaptive_batch_max_ge_min",
        ("adaptive_batch", "adaptive_batch_max_accum", "adaptive_batch_min_accum"),
        lambda a: _adaptive_on(a) and a.adaptive_batch_max_accum < a.adaptive_batch_min_accum,
        "--adaptive-batch-max-accum must be >= --adaptive-batch-min-accum"),
    CombinationCheck(
        "adaptive_batch_every_min", ("adaptive_batch", "adaptive_batch_every"),
        lambda a: _adaptive_on(a) and a.adaptive_batch_every < 1,
        "--adaptive-batch-every must be >= 1 (it counts ROLLOUTS between K moves)"),

    # ---- gen3_winprob_critic_mode_v1: THE CRITIC MODE ---------------------------------------
    # `--critic winprob` makes the win-prob head the value function. Everything below either
    # CONTRADICTS that (a second critic, a normalizer with no scale to track, a reward whose
    # currency is not P(win)) or is a knob the mode SUBSUMES. Each one is refused rather than
    # ignored, because a silently-inert flag on a critic-route change is the failure this whole
    # module exists to end. `config.resolve_critic_mode` IMPLIES the coherent value for each of
    # them first, so a refusal here means the operator TYPED something incompatible.
    CombinationCheck(
        "winprob_critic_needs_a_head", ("critic", "win_prob_mode"),
        lambda a: _winprob(a) and _val(a, "win_prob_mode", "none") == "none",
        "--critic winprob requires --win-prob-mode read_only|shaping: the win-prob HEAD is the "
        "critic, and 'none' builds no head at all, so there would be no value function. "
        "(An unset --win-prob-mode is implied to 'shaping' — this fires only on an explicit "
        "'none'.) 'read_only' is the arm where the critic's gradient does not reach the trunk."),
    CombinationCheck(
        "winprob_critic_refuses_popart", ("critic", "use_popart"),
        lambda a: _winprob(a) and bool(_val(a, "use_popart", False)),
        "--critic winprob is incompatible with --use-popart. PopArt's JOB does not exist here: "
        "the payoff set is fixed at {win, not-win}, so the return is bounded and stationary for "
        "the life of the run and there is no scale to track; the BCE's gradient w.r.t. the logit "
        "is already O(1). Worse, `_denorm` would take V out of [0,1], and PopArt's POP surgery "
        "only ever corrected `value_net`, which this critic does not read. Pass --no-use-popart."),
    CombinationCheck(
        "winprob_critic_refuses_value_dist", ("critic", "value_dist_mode"),
        lambda a: _winprob(a) and _dist_mode(a) != "none",
        lambda a: ("--critic winprob is incompatible with --value-dist-mode "
                   f"{_dist_mode(a)!r}: that is a SECOND critic. Under a terminal-only objective "
                   "the return takes two values, so a categorical over that support IS a "
                   "Bernoulli and the 51-atom head is the same parameterization with 50 redundant "
                   "degrees of freedom. It also mis-states the run's config to every consumer "
                   "that gates on the MODE string rather than on the head (the PPO CE gate, the "
                   "grad-balance value term, the prober's awareness/PIT stack). Pass "
                   "--value-dist-mode none.")),
    CombinationCheck(
        "winprob_critic_refuses_value_from_dist", ("critic", "value_from_dist"),
        lambda a: _winprob(a) and bool(getattr(a, "value_from_dist", False)),
        "--critic winprob is incompatible with --value-from-dist: both name WHICH readout is the "
        "critic, and they name different ones. `_critic_value` cannot route to two places."),
    CombinationCheck(
        "winprob_critic_refuses_win_prob_coef", ("critic", "win_prob_coef"),
        lambda a: _winprob(a) and _typed(a, "win_prob_coef"),
        "--critic winprob is incompatible with an explicit --win-prob-coef: the head's BCE is now "
        "THE VALUE LOSS and is weighted by --vf-coef. One critic, one coefficient — two on one "
        "loss is the ambiguity the distributional critic's `_ce_w` conditional existed to resolve. "
        "NOTE --vf-coef now multiplies a BCE rather than an MSE over a shaped return, so 0.5 does "
        "not transfer between the two loss families; re-tune it on this arm."),
    CombinationCheck(
        "winprob_critic_refuses_value_tail_weight", ("critic", "value_tail_weight"),
        lambda a: _winprob(a) and float(_val(a, "value_tail_weight", 0.0) or 0.0) != 0.0,
        "--critic winprob is incompatible with --value-tail-weight > 0. It weights the SCALAR "
        "MSE, whose term is dropped under this critic, so it would be silently INERT — and its "
        "shape is the banned one anyway: at the decision boundary relevance and label NOISE "
        "arrive together, so 'care more' must be MORE SAMPLES, never a larger per-sample weight "
        "on a Bernoulli likelihood."),
    CombinationCheck(
        # See `--terminal-indicator`. A [0,1] critic cannot represent "worse than a loss", so the
        # ordering `--draw-penalty` exists to set is not merely unused here — it is unrepresentable.
        "winprob_critic_refuses_draw_penalty", ("critic", "draw_penalty", "terminal_indicator"),
        lambda a: _winprob(a) and float(_val(a, "draw_penalty", 0.0) or 0.0) != 0.0,
        lambda a: ("--critic winprob is incompatible with --draw-penalty "
                   f"{float(_val(a, 'draw_penalty', 0.0)):g}. Under this critic the terminal is "
                   "the WIN INDICATOR (+victory_value on a win, 0.0 on a loss, a tie AND a "
                   "250-turn timeout alike), so there is no separate draw magnitude to set, and a "
                   "critic bounded in [0,1] cannot represent 'a timeout is worse than a loss' at "
                   "all. The anti-stall pressure comes from the obs deadline clock and, if the "
                   "stall rate rises, from --arm-no-progress-tax. Pass --draw-penalty 0.")),
    CombinationCheck(
        "winprob_critic_needs_the_indicator_terminal", ("critic", "terminal_indicator"),
        lambda a: _winprob(a) and not bool(_val(a, "terminal_indicator", False)),
        "--critic winprob requires --terminal-indicator. The critic is sigmoid(logit) in [0,1] "
        "and GAE mixes the REWARD with it, so a +V/-V terminal would put the return and the "
        "critic in different scales and every terminal TD error would carry a systematic, "
        "state-dependent offset (a loss reads `-V - V` against a truth of `0 - V`). The indicator "
        "terminal is what makes V(s) == E[return] hold."),
    CombinationCheck(
        "winprob_critic_needs_unit_victory_value", ("critic", "victory_value"),
        lambda a: _winprob(a) and float(_val(a, "victory_value", 30.0) or 0.0) != 1.0,
        lambda a: ("--critic winprob requires --victory-value 1.0 (got "
                   f"{float(_val(a, 'victory_value', 30.0)):g}). With the indicator terminal the "
                   "undiscounted return is `victory_value * 1{win}` while the critic is "
                   "sigmoid(logit) in [0,1], so the two agree at exactly one scale. At 1.0 the "
                   "return IS the win indicator and V(s) == P(win|s) with no approximation term "
                   "-- the identity the whole mode rests on.")),
    CombinationCheck(
        "winprob_critic_needs_no_hand_shaping", ("critic", "hand_shaping"),
        lambda a: _winprob(a) and bool(_val(a, "hand_shaping", True)),
        "--critic winprob requires --no-hand-shaping. The identity V(s) = P(win|s) rests on a "
        "TERMINAL-ONLY reward: under PBRS with Phi(terminal)=0 the return from s telescopes to "
        "`R_T - Phi(s)`, so a critic minimizing its loss learns `V_game(s) - Phi(s)` -- not "
        "P(win), and broken by a KNOWN function rather than approximately. Every PBRS term is "
        "policy-INVARIANT, so deleting them costs learning SPEED, never correctness. NOTE it also "
        "drops `no_progress_tax`; re-arm it with --arm-no-progress-tax if the stall rate rises."),
    CombinationCheck(
        # Owner amendment, 2026-09-06 (design_winprob_only_critic.md §3.7). The SELF-phi shape,
        # refused for a REASON rather than deferred: with V == phi the shaping term IS the
        # advantage, so route 1 adds the advantage to the reward and takes the advantage of that.
        "winprob_critic_refuses_self_phi_pbrs", ("critic", "win_prob_pbrs_coef"),
        lambda a: _winprob(a) and _typed(a, "win_prob_pbrs_coef"),
        "--critic winprob is incompatible with --win-prob-pbrs-coef: no coefficient -- the "
        "potential is currency-matched; the dose ladder belonged to the shaped critic. And the "
        "SELF-phi form is DOUBLE COUNTING: phi is the win-prob head, which under this critic IS "
        "V, so `gamma*phi(s') - phi(s)` is precisely the TD residual GAE already turns into the "
        "advantage. Its Ng shield is also at its structurally weakest here (the theorem assumes a "
        "FIXED phi; ours is the head being trained). The FROZEN-phi rung is a separate flag, "
        "--win-prob-pbrs-frozen."),
    CombinationCheck(
        "winprob_critic_refuses_self_phi_source", ("critic", "win_prob_pbrs_source"),
        lambda a: _winprob(a) and getattr(a, "win_prob_pbrs_source", None) is not None,
        "--critic winprob is incompatible with --win-prob-pbrs-source: that flag is the SHAPED "
        "critic's frozen-phi path and is driven by --win-prob-pbrs-coef, which this mode refuses "
        "(the potential is currency-matched, coefficient exactly 1.0). Under this critic the "
        "frozen rung is --win-prob-pbrs-frozen, which takes no coefficient."),
    CombinationCheck(
        # Owner amendment, 2026-09-06 (design §3.7): "self path to be deleted at the default flip;
        # frozen path kept refused for one generation." Refused, NOT deleted -- the code path
        # behind --win-prob-pbrs-source is intact, so lifting this is one edit.
        "win_prob_pbrs_frozen_is_held", ("critic", "win_prob_pbrs_frozen"),
        lambda a: getattr(a, "win_prob_pbrs_frozen", None) is not None,
        "--win-prob-pbrs-frozen is declared but HELD in this build. Under --critic winprob it is "
        "deferred to a later FROZEN-phi ablation, not judged wrong: exact Ng invariance DOES hold "
        "for a fixed phi -- the critic then learns `P(win) - phi_frozen`, recoverable at inference "
        "by adding phi back -- and the currency-matched coefficient is exactly 1.0, since phi is "
        "already in the value currency (the terminal is the win indicator and V is P(win)). Under "
        "--critic shaped, use the existing --win-prob-pbrs-coef / --win-prob-pbrs-source pair, "
        "whose meaning is unchanged."),

    # ---- the distributional critic --------------------------------------------------------
    CombinationCheck(
        "value_from_dist_needs_shaping", ("value_from_dist", "value_dist_mode"),
        lambda a: a.value_from_dist and _val(a, "value_dist_mode", "none") != "shaping",
        lambda a: ("--value-from-dist requires --value-dist-mode shaping (the distributional head "
                   "must be a live critic that shapes the trunk; got value_dist_mode="
                   f"{_dist_mode(a)!r}).")),
    CombinationCheck(
        # The launch path AUTO-CLEARS an inherited PopArt's clip on a resume, so the refusal only
        # reaches a run that TYPED --use-popart (or has no parent to have inherited it from).
        "popart_needs_explicit_clip_off", ("use_popart", "clip_range_vf"),
        lambda a: bool(a.use_popart) and a.clip_range_vf is not None
        and (_typed(a, "use_popart") or not getattr(a, "_saved_config_present", False)),
        "--use-popart requires an explicit '--clip-range-vf none' (it defaults to 0.5). PopArt "
        "normalizes the value targets so value clipping is unnecessary — and an active clip "
        "would clip in un-normalized units and cripple the critic. Pass --clip-range-vf none.",
        needs=("saved_config",)),

    # ---- exploiter mode ------------------------------------------------------------------
    CombinationCheck(
        "exploiter_excludes_self_play", ("exploiter", "self_play"),
        lambda a: bool(a.exploiter) and bool(a.self_play),
        "--exploiter trains vs ONE fixed target as the sole opponent — it is mutually "
        "exclusive with --self-play. Drop --self-play (the exploiter needs no pool)."),
    CombinationCheck(
        "exploiter_keep_bots_needs_exploiter", ("exploiter_keep_bots", "exploiter"),
        lambda a: bool(a.exploiter_keep_bots) and not a.exploiter,
        "--exploiter-keep-bots only applies in exploiter mode — pass --exploiter <target> "
        "too (it mixes the bots in ALONGSIDE that target)."),
    CombinationCheck(
        "warmstart_consensus_needs_exploiter", ("warmstart_consensus", "exploiter"),
        lambda a: bool(a.warmstart_consensus) and not a.exploiter,
        "--warmstart-consensus builds an EXPLOITER init (a disagreement-gated consensus of "
        "teacher exploiters, sharp-on-agree / flat-on-disagree) and only applies in exploiter "
        "mode — pass --exploiter <target>. It is deliberately NOT available for "
        "generalist / self-play training, whose objective is to ABSORB per-team divergence "
        "(--distill-teacher), the OPPOSITE of distilling the consensus."),
    CombinationCheck(
        "exploiter_temp_start_needs_exploiter", ("exploiter_temp_start", "exploiter"),
        lambda a: _exploiter_temp_on(a) and not a.exploiter,
        "--exploiter-temp-start only applies in exploiter mode — pass --exploiter "
        "<target> too (it anneals THAT target's play temperature)."),
    CombinationCheck(
        "exploiter_temp_positive", ("exploiter_temp_start", "exploiter_temp_end"),
        lambda a: _exploiter_temp_on(a) and (a.exploiter_temp_start <= 0.0
                                             or a.exploiter_temp_end <= 0.0),
        "--exploiter-temp-start / --exploiter-temp-end must be > 0 (a softmax "
        "temperature; the opponent's logits are divided by it)."),
    CombinationCheck(
        "exploiter_temp_anneal_frac", ("exploiter_temp_start", "exploiter_temp_anneal_frac"),
        lambda a: _exploiter_temp_on(a) and not 0.0 <= a.exploiter_temp_anneal_frac <= 1.0,
        "--exploiter-temp-anneal-frac must be a fraction in [0, 1]"),
    CombinationCheck(
        "exploiter_temp_ratchet_factor",
        ("exploiter_temp_start", "exploiter_temp_mode", "exploiter_temp_ratchet_factor"),
        lambda a: _exploiter_temp_on(a) and a.exploiter_temp_mode == "ratchet"
        and not 0.0 < a.exploiter_temp_ratchet_factor < 1.0,
        "--exploiter-temp-ratchet-factor must be in (0, 1) (it multiplies the "
        "temperature DOWN each ratchet)."),
    CombinationCheck(
        "exploiter_temp_ratchet_wr",
        ("exploiter_temp_start", "exploiter_temp_mode", "exploiter_temp_ratchet_wr"),
        lambda a: _exploiter_temp_on(a) and a.exploiter_temp_mode == "ratchet"
        and not 0.0 < a.exploiter_temp_ratchet_wr < 1.0,
        "--exploiter-temp-ratchet-wr must be a win-rate in (0, 1)."),
    CombinationCheck(
        "exploiter_temp_ratchet_games",
        ("exploiter_temp_start", "exploiter_temp_mode", "exploiter_temp_ratchet_games"),
        lambda a: _exploiter_temp_on(a) and a.exploiter_temp_mode == "ratchet"
        and a.exploiter_temp_ratchet_games < 1,
        "--exploiter-temp-ratchet-games must be >= 1."),
    CombinationCheck(
        "exploiter_temp_ratchet_start_above_end",
        ("exploiter_temp_start", "exploiter_temp_mode", "exploiter_temp_end"),
        lambda a: _exploiter_temp_on(a) and a.exploiter_temp_mode == "ratchet"
        and a.exploiter_temp_start <= a.exploiter_temp_end,
        "--exploiter-temp-mode ratchet needs --exploiter-temp-start > "
        "--exploiter-temp-end (it ratchets the temp DOWN from start toward end)."),
    CombinationCheck(
        "exploiter_ladder_needs_exploiter", ("exploiter_ladder", "exploiter"),
        lambda a: bool(a.exploiter_ladder) and not a.exploiter,
        "--exploiter-ladder only applies in exploiter mode — pass --exploiter "
        "<target> too (the ladder's TERMINAL rung IS that target; without it the "
        "curriculum has no destination)."),
    CombinationCheck(
        "exploiter_ladder_gate_range", ("exploiter_ladder", "exploiter_ladder_gate"),
        lambda a: bool(a.exploiter_ladder) and not 0.0 < a.exploiter_ladder_gate < 1.0,
        "--exploiter-ladder-gate must be a win-rate in (0, 1)."),
    CombinationCheck(
        "exploiter_ladder_window_min", ("exploiter_ladder", "exploiter_ladder_window"),
        lambda a: bool(a.exploiter_ladder) and a.exploiter_ladder_window < 1,
        "--exploiter-ladder-window must be >= 1."),
    CombinationCheck(
        "exploiter_ladder_rungs_min", ("exploiter_ladder", "exploiter_ladder_rungs"),
        lambda a: bool(a.exploiter_ladder) and a.exploiter_ladder_rungs < 1,
        "--exploiter-ladder-rungs must be >= 1 (the number of auto: rungs drawn "
        "BEFORE the --exploiter target is appended)."),
    CombinationCheck(
        "exploiter_ladder_rungs_min_no_ladder", ("exploiter_ladder", "exploiter_ladder_rungs"),
        lambda a: not a.exploiter_ladder and a.exploiter_ladder_rungs < 1,
        "--exploiter-ladder-rungs must be >= 1."),
    CombinationCheck(
        "exploiter_temp_ratchet_needs_start", ("exploiter_temp_start", "exploiter_temp_mode"),
        lambda a: a.exploiter_temp_start is None and a.exploiter_temp_mode == "ratchet",
        "--exploiter-temp-mode ratchet requires --exploiter-temp-start (the initial/max "
        "temperature to ratchet down from — set it HIGH, e.g. 5.0)."),

    # ---- gen3_fork_lr_pin_v1 ---------------------------------------------------------------
    CombinationCheck(
        "fork_lr_is_resume_only", ("fork_lr", "model"),
        lambda a: getattr(a, "fork_lr", None) is not None and not a.model,
        "--fork-lr is RESUME-ONLY: it pins the LR of a checkpoint being FORKED, and a "
        "fresh run has no inherited LR to override. Use --lr on a fresh run."),
    CombinationCheck(
        "fork_lr_freeze_needs_fork_lr", ("fork_lr_freeze", "fork_lr"),
        lambda a: bool(getattr(a, "fork_lr_freeze", False))
        and getattr(a, "fork_lr", None) is None,
        "--fork-lr-freeze needs --fork-lr: it freezes the KL controller AT the pinned "
        "rate, and without a pin there is no rate to freeze at (pass --fork-lr <value>)."),

    # ---- the value-distribution head's support ---------------------------------------------
    CombinationCheck(
        "value_dist_mode_needs_bins", ("value_dist_mode", "value_dist_bins"),
        lambda a: _val(a, "value_dist_mode", "none") != "none"
        and not (a.value_dist_bins and a.value_dist_bins > 0),
        "--value-dist-mode requires --value-dist-bins > 0 (the atom count; recommended 32)"),
    CombinationCheck(
        "value_dist_mode_needs_support",
        ("value_dist_mode", "value_dist_vmax", "value_dist_vmin"),
        lambda a: _val(a, "value_dist_mode", "none") != "none"
        and not (a.value_dist_vmax > a.value_dist_vmin),
        "--value-dist-mode requires --value-dist-vmax > --value-dist-vmin (the atom support)"),
    CombinationCheck(
        "value_dist_bins_without_mode", ("value_dist_mode", "value_dist_bins"),
        lambda a: _val(a, "value_dist_mode", "none") == "none" and bool(a.value_dist_bins),
        "--value-dist-bins is set but --value-dist-mode is none — pass a mode, or drop the bins"),

    # ---- the win-prob head and its PBRS ------------------------------------------------------
    CombinationCheck(
        "win_prob_pbrs_coef_needs_mode", ("win_prob_pbrs_coef", "win_prob_mode"),
        lambda a: _positive(a.win_prob_pbrs_coef)
        and _val(a, "win_prob_mode", "none") == "none",
        "--win-prob-pbrs-coef > 0 requires --win-prob-mode read_only|shaping — the PBRS "
        "potential φ(s) IS the win-prob head's output, and --win-prob-mode none builds "
        "no head. Pass a mode, or drop the shaping coefficient."),
    CombinationCheck(
        "win_prob_pbrs_source_needs_coef", ("win_prob_pbrs_source", "win_prob_pbrs_coef"),
        lambda a: bool(getattr(a, "win_prob_pbrs_source", None))
        and not _positive(a.win_prob_pbrs_coef),
        "--win-prob-pbrs-source names the FROZEN potential for the win-prob PBRS, so it "
        "requires --win-prob-pbrs-coef > 0. With no coefficient the source would be a "
        "frozen network loaded, forwarded once per rollout, and multiplied by zero."),

    # ---- the search teacher and OPD ----------------------------------------------------------
    CombinationCheck(
        "opd_coef_needs_search_teacher", ("opd_coef", "search_teacher"),
        lambda a: _positive(a.opd_coef) and not a.search_teacher,
        "--opd-coef > 0 requires --search-teacher (OPD distils the search-teacher's "
        "correction buffer; its workers build the π' targets)"),
    CombinationCheck(
        "search_teacher_mode_needs_teacher", ("search_teacher_mode", "search_teacher"),
        lambda a: a.search_teacher_mode != "crater" and not a.search_teacher,
        lambda a: (f"--search-teacher-mode {a.search_teacher_mode} requires "
                   "--search-teacher — the mode selects which teacher fills the correction "
                   "buffer, and without the flag no teacher runs at all.")),
    CombinationCheck(
        "search_teacher_mode_needs_win_prob", ("search_teacher_mode", "win_prob_mode"),
        lambda a: a.search_teacher_mode != "crater"
        and _val(a, "win_prob_mode", "none") == "none",
        lambda a: (f"--search-teacher-mode {a.search_teacher_mode} requires "
                   "--win-prob-mode read_only|shaping — the one-ply RANKING *is* the win-prob "
                   "head, and --win-prob-mode none builds no head. Falling back to the critic's "
                   "shaped-return ranking would run a DIFFERENT teacher under the same flag.")),

    # ---- the counterfactual label family -----------------------------------------------------
    CombinationCheck(
        "cf_winprob_coef_needs_win_prob_mode", ("cf_winprob_coef", "win_prob_mode"),
        lambda a: _positive(a.cf_winprob_coef)
        and _val(a, "win_prob_mode", "none") == "none",
        "--cf-winprob-coef > 0 requires --win-prob-mode read_only|shaping — the "
        "counterfactual labels supervise the WIN-PROB head, which 'none' does not build"),
    CombinationCheck(
        "cf_evidential_coef_needs_head", ("cf_evidential_coef", "cf_evidential"),
        lambda a: _positive(a.cf_evidential_coef) and not a.cf_evidential,
        "--cf-evidential-coef > 0 requires --cf-evidential — the evidential term "
        "supervises a head that flag BUILDS, and it is a structural (version-gated) "
        "toggle that cannot be turned on mid-run"),
    CombinationCheck(
        "cf_twin_coef_needs_heads", ("cf_twin_coef", "cf_twin_heads"),
        lambda a: _positive(a.cf_twin_coef) and not a.cf_twin_heads,
        "--cf-twin-coef > 0 requires --cf-twin-heads — the twin heads are a "
        "state_dict change (v99, version-gated) and cannot be added to a run that "
        "did not start with them."),
    CombinationCheck(
        "cf_twin_heads_need_win_prob_mode", ("cf_twin_heads", "win_prob_mode"),
        lambda a: bool(a.cf_twin_heads) and _val(a, "win_prob_mode", "none") == "none",
        "--cf-twin-heads requires --win-prob-mode read_only|shaping — the twins "
        "mirror head A's on-policy BCE, and --win-prob-mode none builds no head A, so "
        "the arm's control arm would not exist."),
    CombinationCheck(
        "cf_shadow_coef_needs_critic", ("cf_shadow_coef", "cf_shadow_critic"),
        lambda a: _positive(a.cf_shadow_coef) and not a.cf_shadow_critic,
        "--cf-shadow-coef > 0 requires --cf-shadow-critic — the shadow head is a "
        "state_dict change (v99, version-gated) and cannot be added to a run that "
        "did not start with it."),
    CombinationCheck(
        "q_winprob_coef_needs_mode",
        ("q_winprob_coef", "q_winprob_onpolicy_coef", "q_winprob_mode"),
        lambda a: _q_live(a) and _val(a, "q_winprob_mode", "none") == "none",
        "--q-winprob-coef / --q-winprob-onpolicy-coef > 0 requires --q-winprob-mode "
        "read_only — the term supervises a head that flag BUILDS, and it is a "
        "structural (version-gated) toggle that cannot be turned on mid-run."),
    CombinationCheck(
        "cf_records_needs_bridge", ("cf_records", "use_bridge"),
        lambda a: bool(a.cf_records) and _val(a, "use_bridge", "rust") == "off",
        "--cf-records requires the in-process bridge (--use-bridge node|rust) — the "
        "reconstruction record is a bridge frame; a websocket run emits none"),
    CombinationCheck(
        # gen3_cf_label_duty_cycle_v1 — a quantity nobody was computing. FATAL_CONFIG, not
        # parser.error: a restart would hit the identical config, so the launcher must give up.
        "cf_label_duty_cycle_floor",
        ("cf_twin_coef", "cf_winprob_coef", "cf_records", "cf_label_lag_steps",
         "checkpoint_every_steps", "n_envs"),
        _cf_duty_cycle_starved, _cf_duty_cycle_message, exit_style="fatal_config"),

    # ---- distillation: the coefficient gates the companions ----------------------------------
    CombinationCheck(
        "distill_value_coef_needs_distill_coef", ("distill_value_coef", "distill_coef"),
        lambda a: _positive(a.distill_value_coef) and not _positive(a.distill_coef),
        "--distill-value-coef > 0 requires --distill-coef > 0 — the value distillation is "
        "coherent only because the policy KL drives π_student→π_teacher on those states, "
        "making V_teacher the right target (V^π is policy-relative)."),
    CombinationCheck(
        "distill_value_feat_coef_needs_distill_coef", ("distill_value_feat_coef", "distill_coef"),
        lambda a: _positive(a.distill_value_feat_coef) and not _positive(a.distill_coef),
        "--distill-value-feat-coef > 0 requires --distill-coef > 0 — the FitNets value-feature "
        "match is coherent only because the policy KL drives π_student→π_teacher on those states, "
        "making the teacher's value_pooled the right target (V^π is policy-relative)."),

    # ---- gen3_distill_offslice_anchor_v1 — the OFF-SLICE trust region -------------------------
    CombinationCheck(
        "anchor_proj_samples_needs_grad_project",
        ("distill_anchor_proj_samples", "distill_anchor_mode"),
        lambda a: _typed(a, "distill_anchor_proj_samples") and not _grad_project(a),
        "--distill-anchor-proj-samples only applies to --distill-anchor-mode "
        "grad_project — it sizes that mode's constraint set and nothing else reads it."),
    CombinationCheck(
        # G5 (2026-09-06), refusal #1. The slice IS the `distill_mask` obs key, and the env emits
        # it only for a run with a live distill term — so a control arm at coef 0 cannot carry the
        # instrument, and `checkargs` used to say the command launched.
        "anchor_needs_live_distill",
        ("distill_anchor_coef", "distill_anchor_monitor", "distill_anchor_mode", "distill_coef"),
        lambda a: _anchor_wanted(a) and not _positive(a.distill_coef),
        "--distill-anchor-coef / --distill-anchor-monitor require --distill-coef > 0 — "
        "the anchor's OFF-SLICE split reads the `distill_mask` obs key, which the env "
        "emits only for a run with a live exploiter-distillation term."),
    CombinationCheck(
        "anchor_knobs_need_anchor",
        ("distill_anchor_mode", "distill_anchor_ref", "distill_anchor_parent",
         "distill_anchor_coef", "distill_anchor_monitor"),
        lambda a: _anchor_knob_typed(a) and not _anchor_wanted(a),
        "--distill-anchor-mode / --distill-anchor-ref / --distill-anchor-ema-tau / "
        "--distill-anchor-refresh-every / --distill-anchor-parent do nothing without "
        "--distill-anchor-coef > 0 or --distill-anchor-monitor — pass one of those, or "
        "drop these."),
    CombinationCheck(
        "anchor_target_kl_needs_coef", ("distill_anchor_target_kl", "distill_anchor_coef"),
        lambda a: _dual_on(a) and not _positive(a.distill_anchor_coef),
        "--distill-anchor-target-kl requires --distill-anchor-coef > 0: the dual "
        "update is MULTIPLICATIVE (coef <- coef * exp(...)), so a coefficient of 0 "
        "is a fixed point and the controller could never move it. Give the dual a "
        "starting coefficient to scale."),
    CombinationCheck(
        "anchor_dual_lr_positive", ("distill_anchor_target_kl", "distill_anchor_dual_lr"),
        lambda a: _dual_on(a) and a.distill_anchor_dual_lr is not None
        and a.distill_anchor_dual_lr <= 0,
        "--distill-anchor-dual-lr must be > 0 (it is the dual's step size)."),
    CombinationCheck(
        "anchor_coef_min_nonnegative", ("distill_anchor_target_kl", "distill_anchor_coef_min"),
        lambda a: _dual_on(a) and (a.distill_anchor_coef_min or 0.0) < 0,
        "--distill-anchor-coef-min must be >= 0 (it clamps a KL weight)."),
    CombinationCheck(
        "anchor_coef_max_ge_min",
        ("distill_anchor_target_kl", "distill_anchor_coef_max", "distill_anchor_coef_min"),
        lambda a: _dual_on(a) and a.distill_anchor_coef_max is not None
        and a.distill_anchor_coef_max < (a.distill_anchor_coef_min or 0.0),
        "--distill-anchor-coef-max must be >= --distill-anchor-coef-min."),
    CombinationCheck(
        "dual_knobs_need_target_kl",
        ("distill_anchor_target_kl", "distill_anchor_dual_lr", "distill_anchor_coef_min",
         "distill_anchor_coef_max"),
        lambda a: _dual_knob_typed(a) and not _dual_on(a)
        and not (a.distill_anchor_target_kl is not None and a.distill_anchor_target_kl < 0),
        "--distill-anchor-dual-lr / --distill-anchor-coef-min / "
        "--distill-anchor-coef-max do nothing without --distill-anchor-target-kl > 0 "
        "— pass that, or drop these."),

    # ---- gen3_distill_stop_rule_v1 -------------------------------------------------------------
    CombinationCheck(
        "distill_stop_needs_anchor_monitor",
        ("distill_stop", "distill_anchor_monitor", "distill_anchor_coef", "distill_anchor_mode"),
        lambda a: _stop_on(a) and not _anchor_wanted(a),
        "--distill-stop requires the anchor MONITOR: pass --distill-anchor-monitor "
        "(or --distill-anchor-coef > 0, or --distill-anchor-mode grad_project). "
        "The rule's RISE half reads distill/collateral_kl_vs_parent, which only "
        "exists when the frozen fold parent is attached — without it the AND-gate "
        "could never close and the flag would be a silent no-op."),
    CombinationCheck(
        "distill_stop_window_min", ("distill_stop", "distill_stop_window"),
        lambda a: _stop_on(a) and a.distill_stop_window is not None and a.distill_stop_window < 2,
        "--distill-stop-window must be >= 2: the rise test is an OLS slope over "
        "window+1 points and needs at least one residual degree of freedom for its "
        "standard error to exist."),
    CombinationCheck(
        "distill_stop_persist_min", ("distill_stop", "distill_stop_persist"),
        lambda a: _stop_on(a) and a.distill_stop_persist is not None and a.distill_stop_persist < 1,
        "--distill-stop-persist must be >= 1."),
    CombinationCheck(
        "distill_stop_anneal_factor_range", ("distill_stop", "distill_stop_anneal_factor"),
        lambda a: _stop_on(a) and a.distill_stop_anneal_factor is not None
        and not 0.0 < a.distill_stop_anneal_factor < 1.0,
        "--distill-stop-anneal-factor must be in (0, 1) — it is the per-rollout "
        "geometric decay of --distill-coef after the rule fires."),
    CombinationCheck(
        "stop_knobs_need_distill_stop",
        ("distill_stop", "distill_stop_window", "distill_stop_eps", "distill_stop_kl_slope",
         "distill_stop_persist", "distill_stop_anneal_factor"),
        lambda a: _stop_knob_typed(a) and not _stop_on(a),
        "--distill-stop-window / --distill-stop-eps / --distill-stop-kl-slope / "
        "--distill-stop-persist / --distill-stop-anneal-factor do nothing without "
        "--distill-stop {warn,anneal,abort} — pass one, or drop these."),

    # ---- the teacher spec's CROSS-FLAG requirements (the grammar itself stays in config.py) ----
    CombinationCheck(
        "distill_coef_needs_teacher", ("distill_coef", "distill_teacher"),
        lambda a: _positive(a.distill_coef) and not _teacher_items(a),
        "--distill-coef > 0 requires --distill-teacher (as 'TEACHER:TEAM[,TEAM...]' groups)"),
    CombinationCheck(
        # G5 (2026-09-06), refusal #2.
        "distill_team_bias_needs_teacher", ("distill_team_bias", "distill_teacher"),
        lambda a: _typed(a, "distill_team_bias")
        and (getattr(a, "distill_team_bias", None) or 0) > 0 and not _teacher_items(a),
        "--distill-team-bias > 0 requires --distill-teacher — the bias points at the "
        "TEACHER TEAMS ('TEACHER:TEAM[,TEAM...]' groups) and there is nothing to bias "
        "toward without one; the flag would be a silent no-op. Drop it, pass "
        "--distill-team-bias 0, or name the teacher(s)."),

    # ---- gen3_distill_target_gate_v1 (design §7.5): the action-form family's dependency graph ---
    CombinationCheck(
        # G5 (2026-09-06), refusal #3 — and the C1 (2026-09-01) launch this module was built for.
        "distill_target_needs_coef", ("distill_target", "distill_coef"),
        lambda a: getattr(a, "distill_target", None) == "action"
        and not _positive(getattr(a, "distill_coef", None)),
        "--distill-target action requires --distill-coef > 0 — the target form is a "
        "property of the distill term; without the term there is nothing to shape"),
    CombinationCheck(
        "distill_topk_needs_action", ("distill_topk", "distill_target"),
        lambda a: getattr(a, "distill_topk", 1) not in (1, None)
        and getattr(a, "distill_target", None) not in ("action", None),
        "--distill-topk requires --distill-target action — the top-K dial "
        "parameterizes the action-form target; the 'kl' path has no K"),
    CombinationCheck(
        "distill_gate_needs_action", ("distill_gate", "distill_target"),
        lambda a: getattr(a, "distill_gate", "none") not in ("none", None)
        and getattr(a, "distill_target", None) not in ("action", None),
        "--distill-gate requires --distill-target action (design §7.5: the gate "
        "rides the action-form term)"),
    CombinationCheck(
        "distill_gate_tau_needs_advantage", ("distill_gate_tau", "distill_gate"),
        lambda a: (getattr(a, "distill_gate_tau", 0.0) or 0.0) != 0.0
        and getattr(a, "distill_gate", "none") not in ("advantage", None),
        "--distill-gate-tau requires --distill-gate advantage — tau is the advantage "
        "gate's threshold"),
    CombinationCheck(
        "distill_teacher_excludes_trainee_pin",
        ("distill_teacher", "trainee_team", "trainee_teams"),
        lambda a: bool(_teacher_items(a)) and bool(a.trainee_team or a.trainee_teams),
        "--distill-teacher is mutually exclusive with --trainee-team/--trainee-teams: "
        "distillation biases the trainee toward the teacher teams via --distill-team-bias "
        "while keeping the pool for rehearsal; a hard pin would remove the rehearsal (and "
        "cause forgetting), and the bias would override the pin anyway"),

    # ---- the belief stack ---------------------------------------------------------------------
    CombinationCheck(
        "move_belief_hidden_needs_species_belief", ("move_belief_mode", "opp_belief_aux_coef"),
        lambda a: _val(a, "move_belief_mode", "off") in ("unrevealed", "both")
        and not _positive(_val(a, "opp_belief_aux_coef", 0.0)),
        lambda a: (f"--move-belief-mode {_belief_mode(a)} scores the opponent's HIDDEN slots, "
                   "which are only filled with learned unknown-mon tokens when the species-belief "
                   "head is on. Add --opp-belief-aux-coef <coef> (>0), or use --move-belief-mode "
                   "revealed (seen mons only).")),
    CombinationCheck(
        "damage_op_needs_revealed_move_belief", ("damage_op", "move_belief_mode"),
        lambda a: bool(a.damage_op)
        and _val(a, "move_belief_mode", "off") not in ("revealed", "both"),
        "--damage-op requires --move-belief-mode revealed (or both): the operator is fed the opp "
        "active's predicted moves, which are only supervised for a revealed mon. Set "
        "--move-belief-mode revealed, or drop --damage-op."),
    CombinationCheck(
        "move_prior_fusion_needs_move_belief", ("move_prior_fusion", "move_belief_mode"),
        lambda a: bool(a.move_prior_fusion) and _val(a, "move_belief_mode", "off") == "off",
        "--move-prior-fusion requires --move-belief-mode != off (revealed|unrevealed|both): the "
        "prior fuses into the move-belief head's logits. Set --move-belief-mode revealed, or drop "
        "--move-prior-fusion."),
    CombinationCheck(
        "species_prior_fusion_needs_belief_coef", ("species_prior_fusion", "opp_belief_aux_coef"),
        lambda a: bool(a.species_prior_fusion)
        and not _positive(_val(a, "opp_belief_aux_coef", 0.0)),
        "--species-prior-fusion requires --opp-belief-aux-coef > 0: the team-composition prior "
        "fuses into the BeliefHead's species head, which is only built under the hidden-opponent "
        "belief slots. Set --opp-belief-aux-coef, or drop --species-prior-fusion."),

    # ---- the damage operator and its blocks ----------------------------------------------------
    CombinationCheck(
        "damage_candidate_k_needs_op", ("damage_candidate_k", "damage_op"),
        lambda a: bool(a.damage_candidate_k) and not a.damage_op,
        "--damage-candidate-k requires --damage-op (it caps the damage operator's incoming "
        "candidate sweep, which only exists when the op is built). Add --damage-op / "
        "--unified-damage, or drop --damage-candidate-k."),
    CombinationCheck(
        "damage_outgoing_needs_op", ("damage_outgoing", "damage_op"),
        lambda a: bool(a.damage_outgoing) and not a.damage_op,
        "--damage-outgoing requires --damage-op (the outgoing block is part of the damage operator). "
        "Use --unified-damage both, or add --damage-op."),
    CombinationCheck(
        "entity_topk_seats_need_op_and_latent",
        ("entity_topk_seats", "damage_op", "move_latent"),
        lambda a: bool(a.entity_topk_seats) and a.entity_topk_seats > 0
        and not (a.damage_op and a.move_latent),
        "--entity-topk-seats > 0 requires --damage-op AND --move-latent (--unified-moves): "
        "the E4 threat seats gather the op's pre-transformer candidate weights + move latents. "
        "Add those flags, or set --entity-topk-seats 0 (E3-only)."),
    CombinationCheck(
        "entity_tail_seats_need_topk", ("entity_tail_seats", "damage_op", "entity_topk_seats"),
        lambda a: bool(a.entity_tail_seats)
        and not (a.damage_op and a.entity_topk_seats and a.entity_topk_seats > 0),
        "--entity-tail-seats requires --damage-op AND --entity-topk-seats > 0 "
        "(the tail is defined relative to the E4 seats' truncation)."),
    CombinationCheck(
        "edge_families_d1s1c1c2_need_outgoing",
        ("edge_bias_families", "damage_op", "damage_outgoing"),
        lambda a: _fams(a, {"d1", "s1", "c1", "c2"}) and not (a.damage_op and a.damage_outgoing),
        "--edge-bias-families d1/s1/c1/c2 require --damage-op AND --damage-outgoing "
        "(--unified-damage both / --unified-moves both)."),
    CombinationCheck(
        "edge_families_x_needs_op", ("edge_bias_families", "damage_op"),
        lambda a: _fams(a, {"x"}) and not a.damage_op,
        "--edge-bias-families x requires --damage-op "
        "(the Pursuit belief comes from the op's pre-transformer posterior)."),
    CombinationCheck(
        "edge_families_kernels_need_op", ("edge_bias_families", "damage_op"),
        lambda a: _fams(a, {"d2", "d4", "v", "t", "g", "c4", "c3", "c5"}) and not a.damage_op,
        "--edge-bias-families d2/d4/v/t/g/c4/c3/c5 require --damage-op (the op's kernels/buffers)."),
    CombinationCheck(
        "edge_families_d3s3_need_seats", ("edge_bias_families", "entity_topk_seats"),
        lambda a: _fams(a, {"d3", "s3"})
        and not (a.entity_topk_seats and a.entity_topk_seats > 0),
        "--edge-bias-families d3/s3 require --entity-topk-seats > 0 (the bias rows "
        "ARE the E4 threat seats)."),
    CombinationCheck(
        "move_candidate_floor_needs_fusion", ("move_candidate_floor", "move_prior_fusion"),
        lambda a: _floor_is_non_default(a) and not a.move_prior_fusion,
        "--move-candidate-floor requires --move-prior-fusion (it sets the floor of the FUSED move "
        "prior, which only exists under fusion). Enable --move-prior-fusion (or --unified-damage), "
        "or drop --move-candidate-floor."),
    CombinationCheck(
        "damage_topk_needs_op", ("damage_topk_k", "damage_op"),
        lambda a: bool(a.damage_topk_k) and a.damage_topk_k > 0 and not a.damage_op,
        "--damage-topk requires --damage-op (the discrete incoming block extends the damage operator). "
        "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-topk 0."),
    CombinationCheck(
        "damage_topk_needs_move_latent", ("damage_topk_k", "move_latent"),
        lambda a: bool(a.damage_topk_k) and a.damage_topk_k > 0 and not a.move_latent,
        "--damage-topk requires --move-latent (the block gathers each move's identity latent "
        "from the MoveLatentEncoder). Use --unified-moves, or add --move-latent, or set --damage-topk 0."),
    CombinationCheck(
        "damage_topk_needs_incoming_matrix", ("damage_topk_k", "damage_matrices_incoming"),
        # Only reachable when --damage-matrices was passed EXPLICITLY as off/outgoing: with
        # the flag unset, `resolve_config` AUTO-ENABLES the incoming matrix at K>0.
        lambda a: bool(a.damage_topk_k) and a.damage_topk_k > 0
        and _typed(a, "damage_matrices") and not a.damage_matrices_incoming,
        lambda a: (f"--damage-topk {a.damage_topk_k} contradicts --damage-matrices "
                   f"{a.damage_matrices}: K is "
                   "the INCOMING matrix's width, and the lean top-K block it used to select was deleted "
                   "(gen3_op_block_trim_v1). Use --damage-matrices incoming/both, or set --damage-topk 0.")),
    CombinationCheck(
        "damage_matrices_outgoing_needs_op", ("damage_matrices_outgoing", "damage_op"),
        lambda a: bool(getattr(a, "damage_matrices_outgoing", False)) and not a.damage_op,
        "--damage-matrices outgoing requires --damage-op (the matrix is emitted by the damage operator). "
        "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-matrices off."),
    CombinationCheck(
        "damage_matrices_incoming_needs_op", ("damage_matrices_incoming", "damage_op"),
        lambda a: bool(getattr(a, "damage_matrices_incoming", False)) and not a.damage_op,
        "--damage-matrices incoming requires --damage-op (the matrix is emitted by the damage "
        "operator). Use --unified-damage / --unified-moves, or add --damage-op."),
    CombinationCheck(
        "damage_matrices_incoming_needs_move_latent",
        ("damage_matrices_incoming", "move_latent"),
        lambda a: bool(getattr(a, "damage_matrices_incoming", False)) and not a.move_latent,
        "--damage-matrices incoming requires --move-latent (the matrix header gathers each move's "
        "identity latent). Use --unified-moves, or add --move-latent."),
    CombinationCheck(
        "move_belief_latent_coef_needs_latent", ("move_belief_latent_coef", "move_latent"),
        lambda a: bool(a.move_belief_latent_coef) and not a.move_latent,
        "--move-belief-latent-coef requires --move-latent (the grading reads its per-move latent "
        "table). Enable --move-latent (or --unified-moves), or set --move-belief-latent-coef 0."),
    CombinationCheck(
        "move_belief_latent_coef_needs_revealed",
        ("move_belief_latent_coef", "move_belief_mode"),
        lambda a: bool(a.move_belief_latent_coef)
        and _val(a, "move_belief_mode", "off") not in ("revealed", "both"),
        "--move-belief-latent-coef requires --move-belief-mode revealed (or both): it grades the "
        "move belief on revealed slots. Set --move-belief-mode revealed (or --unified-moves), or set "
        "--move-belief-latent-coef 0."),
    CombinationCheck(
        "spread_belief_coef_needs_head", ("spread_belief_coef", "spread_belief"),
        lambda a: bool(a.spread_belief_coef) and not a.spread_belief,
        "--spread-belief-coef requires --spread-belief (it supervises the believed opp spread). "
        "Enable --spread-belief, or set --spread-belief-coef 0."),
    CombinationCheck(
        "spread_belief_nature_needs_head", ("spread_belief_nature", "spread_belief"),
        lambda a: bool(a.spread_belief_nature) and not a.spread_belief,
        "--spread-belief-nature requires --spread-belief (it reparameterises the SpreadBelief head). "
        "Enable --spread-belief, or drop --spread-belief-nature."),
    CombinationCheck(
        "hp_type_belief_coef_needs_move_belief",
        ("hp_type_belief_coef", "move_belief_mode"),
        lambda a: _typed(a, "hp_type_belief_coef") and bool(a.hp_type_belief_coef)
        and _val(a, "move_belief_mode", "off") == "off",
        "--hp-type-belief-coef requires a move belief (--move-belief-mode != off / --unified-moves): "
        "the HP-type head composes P(HP present) out of the move posterior. Enable the move belief, "
        "or set --hp-type-belief-coef 0."),

    # ---- the LR anneal pair: print + exit 1, never parser.error ---------------------------------
    CombinationCheck(
        "anneal_start_needs_min_lr", ("anneal_lr_start_steps", "anneal_min_lr"),
        lambda a: a.anneal_lr_start_steps is not None and a.anneal_min_lr is None,
        "[AnnealLR] ERROR: --anneal-min-lr is required when --anneal-lr-start-steps is set",
        exit_style="exit1"),
    CombinationCheck(
        "anneal_start_below_steps", ("anneal_lr_start_steps", "steps"),
        lambda a: a.anneal_lr_start_steps is not None and a.anneal_lr_start_steps >= a.steps,
        lambda a: (f"[AnnealLR] ERROR: --anneal-lr-start-steps ({a.anneal_lr_start_steps:,}) "
                   f"must be less than --steps ({a.steps:,})"),
        exit_style="exit1"),

    # ---- the compile pair: the rule itself lives in compile_flags, referenced not re-typed ------
    CombinationCheck(
        "compile_preload_needs_compile_opponents",
        ("compile_opponents_preload", "compile_opponents"),
        lambda a: bool(a.compile_opponents_preload) and not a.compile_opponents,
        _PRELOAD_WITHOUT_OPPONENTS),
)


def _floor_is_non_default(args) -> bool:
    """A NON-DEFAULT `--move-candidate-floor`; the default is not flagged, it is just the default."""
    from agents.model.damage_tables import _PRIOR_FLOOR
    return _val(args, "move_candidate_floor", _PRIOR_FLOOR) != _PRIOR_FLOOR


BY_NAME = {c.name: c for c in COMBINATION_CHECKS}


def failing_checks(args) -> List[CombinationCheck]:
    """Every check whose combination is broken on `args`, in declaration order.

    A predicate that cannot READ a value (an argv the parser never filled, a namespace missing the
    attribute) is skipped rather than guessed at: "unknown" is not a verdict, and under-reporting is
    the right failure direction for a tool whose warnings are meant to be worth acting on.
    """
    out: List[CombinationCheck] = []
    for check in COMBINATION_CHECKS:
        try:
            if check.predicate(args):
                out.append(check)
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def refuse_first(args, parser) -> None:
    """THE LAUNCH PATH'S half: refuse on the first broken combination, in its own exit style.

    `parser.error` exits 2 the way argparse does; `exit1` reproduces the two `--anneal-lr-*`
    refusals, which print to stdout and exit 1; `fatal_config` reproduces the CF duty-cycle floor,
    which exits `TrainExitCode.FATAL_CONFIG` so the launcher gives up instead of restarting into
    the identical config.
    """
    import sys

    from main.exit_codes import TrainExitCode

    for check in failing_checks(args):
        text = check.text(args)
        if check.exit_style == "exit1":
            print(text)
            sys.exit(1)
        if check.exit_style == "fatal_config":
            print(text, file=sys.stderr, flush=True)
            sys.exit(int(TrainExitCode.FATAL_CONFIG))
        parser.error(text)
