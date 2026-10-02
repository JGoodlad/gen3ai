"""COMBINATION CHECKS — the value-conditional refusals, in ONE place, read by BOTH surfaces.

WHY THIS MODULE EXISTS. `agents.model.flag_registry`'s `requires` graph expresses one shape of
dependency — *flag A must be ENABLED for flag B to be enabled* — and `main.checkargs` reads it, so
an unsatisfiable structural combination is reported offline instead of crashing inside
`Gen3FeaturesExtractor.__init__`. But some launch-time refusals are not that shape: they are
value-conditional (*`--q-winprob-coef` > 0 requires `--q-winprob-mode`*), they lived in
`main.train.config.resolve_config` as `parser.error` lines, and nothing outside that function knew
them.

That gap cost launches, and the SECOND round is why this module is no longer a short list. C1
(2026-09-01) forked a parent whose recorded config carried an action-form distillation target, passed
`--distill-coef 0`, and did NOT name the target — so `_resolve` inherited it, the check fired, and the
run died at launch while `checkargs` had said "this command still launches". The fix moved *that* rule
here and left every sibling behind, so the module's own premise — "one list, both readers" — held only
for the rules that had been migrated. G5 (2026-09-06) then died three times in a row on the ones that
had not. (The distillation family those rules guarded was deleted with deletion pass L3; the lesson
stands.) A partial single-source is a single-source nobody can trust, because nothing about the output
distinguishes "checked and clean" from "never asked".

So the list is now EXHAUSTIVE over its class, and that is enforced rather than intended:
`combination_checks_test.py` walks `config.py`'s own AST — resolving local aliases, which is how
two helper predicates once hid three refusals from an earlier reading — and FAILS, naming
file:line, if any `parser.error` remains whose guard reads a second flag's value. The allowlist
carries a reason string per entry.

THE CONTRACT. A check is a pure predicate over an args-shaped namespace plus the message the LAUNCH
path prints. `resolve_config` calls `refuse_first`, which renders that message through the check's
own exit style; `main.checkargs` calls `failing_checks` on the EFFECTIVE namespace (argv overlaid
on the fork parent's recorded config) and reports every one. Neither owns the rule.

WHAT BELONGS HERE: a refusal that reads two or more RESOLVED values and says one combination is
incoherent — including a range check that only applies in a mode (`--exploiter-ladder-gate` under
`--exploiter-ladder`), because "which flag turns this on" is itself a cross-flag fact. What does
NOT: a range check on a single value (`--rank-tripwire-drop` in (0, 1)), which argparse's caller can
answer from the one value it has; and anything needing the parser, a torch import, an env var, or a
filesystem — those stay in `resolve_config`, which has them, and are listed with
their reason in the test's allowlist.

ORDERING. `resolve_config` evaluates the whole list at ONE point, late in validation, and refuses
on the first failure in DECLARATION order — which is source order as the checks stood before the
migration. For an argv that trips exactly one rule (every real launch failure so far) the message
and the exit path are byte-identical to what shipped. For an argv broken two ways, WHICH message
comes first can differ from the pre-migration order, because the single-value range checks now all
run before the sweep; the test table pins the one-defect case, which is the contract.

EXPLICITNESS. Several refusals fire only on a value the operator actually TYPED (a knob that does
nothing without its master flag). On the launch path that is captured before
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
    #: The argv option(s) this check REFUSES ON A RESUME — a FRESH-run-only flag. Declared here so
    #: the launcher's restart path strips exactly what the trainer refuses (`fresh_only_flags()`),
    #: rather than keeping a second list that drifts (2026-09-26: an interval restart of a
    #: `--arch production` run re-passed `--arch` beside `--model` and crash-looped out). Every
    #: option named here takes exactly ONE value (`combination_checks_test.py` pins that against
    #: the real parser), which is what the launcher's stripper assumes.
    fresh_only: Tuple[str, ...] = ()

    def text(self, args) -> str:
        """The message as the launch path prints it."""
        return self.message(args) if callable(self.message) else self.message


# --------------------------------------------------------------------------------------------
# Predicate helpers. Each reads only the namespace, so both surfaces get the same answer.
# --------------------------------------------------------------------------------------------

def _positive(value: Any) -> bool:
    """The `value and value > 0` idiom the launch path uses, None-safe."""
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


def _starve_spec_error(args) -> "ValueError | None":
    """The parse error of `--supply-starve-cycles`, or None (`agents.training.lever_supply`)."""
    from agents.training.lever_supply import parse_starve_overrides
    try:
        parse_starve_overrides(getattr(args, "supply_starve_cycles", None))
    except ValueError as e:
        return e
    return None


def _val(args, dest: str, default: Any) -> Any:
    """The value a launch would RESOLVE for `dest` — the default when the flag is still unset.

    Every one of these mirrors a `_resolve(name, default)` line in `main.train.config`. On the
    LAUNCH path the resolve has already run, so this returns the value unchanged; on `checkargs`'
    namespace an unset flag is still `None`, and reading `None != "none"` as "the mode is on" is a
    FALSE POSITIVE that reported two nonexistent problems the first time this list was widened.
    """
    value = getattr(args, dest, None)
    return default if value is None else value


def _winprob_gamma() -> float:
    from agents.model.critic_mode import CRITIC_WINPROB, critic_gamma
    return critic_gamma(CRITIC_WINPROB)


def _winprob(args) -> bool:
    """Is this the WIN-PROB critic? Read through the ONE predicate, never a string compare here."""
    from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
    return is_winprob(_val(args, "critic", CRITIC_DEFAULT))


def _belief_mode(args) -> str:
    return _val(args, "move_belief_mode", "off")


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


#: The cf-buffer consumers — the SAME tuple `agents.training.cf_supply` guards (pinned by
#: `cf_supply_test`); a literal here keeps this module import-light for `checkargs`.
_CF_CONSUMER_COEFS: Tuple[str, ...] = (
    "cf_winprob_coef", "cf_evidential_coef", "cf_twin_coef", "cf_shadow_coef",
    "q_winprob_coef", "q_winprob_onpolicy_coef")


def _cf_duty_cycle_starved(args) -> bool:
    """The counterfactual label path is starved BY CONSTRUCTION — see `_announce_cf_duty_cycle`."""
    on = (_positive(getattr(args, "cf_twin_coef", None))
          or _positive(getattr(args, "cf_winprob_coef", None)))
    if not (on and getattr(args, "cf_records", None)) or getattr(args, "debug", False):
        return False
    from main.train.constants import (CF_DUTY_CYCLE_FLOOR, cf_label_duty_cycle,
                                      checkpoint_interval_env_steps)
    interval = checkpoint_interval_env_steps(getattr(args, "checkpoint_every_steps", None))
    return cf_label_duty_cycle(args.cf_label_lag_steps, interval) < CF_DUTY_CYCLE_FLOOR


def _cf_duty_cycle_message(args) -> str:
    from main.train.constants import (CF_DUTY_CYCLE_FLOOR, cf_label_duty_cycle,
                                      checkpoint_interval_env_steps)
    n_envs = int(args.n_envs)
    every = getattr(args, "checkpoint_every_steps", None)
    interval = checkpoint_interval_env_steps(every)
    duty = cf_label_duty_cycle(args.cf_label_lag_steps, interval)
    shown = ("unbounded (--cf-label-lag-steps 0 = labels never expire)" if duty == float("inf")
             else f"{duty:.1%}")
    return (
        f"\n[CF] FATAL: the counterfactual label path is STARVED BY CONSTRUCTION.\n"
        f"  --cf-label-lag-steps         : {args.cf_label_lag_steps:,} env steps\n"
        f"  checkpoint interval          : {interval:,} env steps "
        f"(TOTAL env steps over all {n_envs} envs)\n"
        f"  --checkpoint-every-steps     : "
        f"{'(unset — the 2,400,000-env-step default)' if every is None else format(every, ',')}\n"
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

# ---- `--env-core rust` (M5 Lane G) -------------------------------------------------------------

def _rust_core(args) -> bool:
    return _val(args, "env_core", "rust") == "rust"     # the parser default since D2 (2026-10-02)


#: What `--env-core rust` does not serve yet, as (dest, predicate, reason). Each is a path the Python
#: env or a Python-only callback owns today; the collector refuses rather than silently dropping it.
_ENV_CORE_UNPORTED: Tuple[Tuple[str, Callable[[Any], bool], str], ...] = (
    ("cf_records", lambda a: bool(_val(a, "cf_records", False)), "--cf-records (a bridge reconstruction tap)"),
    ("team_pfsp", lambda a: _val(a, "team_pfsp", "off") != "off", "--team-pfsp (per-worker PFSP pulls)"),
    ("exploiter_ladder", lambda a: bool(_val(a, "exploiter_ladder", None)),
     "--exploiter-ladder (the rung loader is not wired to T2 yet)"),
    ("async_rollout", lambda a: bool(_val(a, "async_rollout", False)),
     "--async-rollout (a SubprocVecEnv scheduling mode)"),
)
_ENV_CORE_UNPORTED_DESTS: Tuple[str, ...] = ("env_core",) + tuple(d for d, _, _ in _ENV_CORE_UNPORTED)

#: Flags that act only under `--env-core rust`.
_ENV_CORE_ONLY_DESTS: Tuple[str, ...] = (
    "rollout_trigger", "rollout_target_samples", "rollout_target_band", "version_pinning", "trainee_slots",
    "t2_buckets", "t2_lanes", "t2_opponent_bucket_cap", "t2_backend", "rust_env_front", "rust_env_threads", "rust_env_profile",
    "rust_env_refusal_budget", "rust_env_respawn_budget", "opponent_sampling", "rust_eval_envs")


def _env_core_unported(args) -> List[str]:
    out: List[str] = []
    for _dest, pred, why in _ENV_CORE_UNPORTED:
        try:
            if pred(args):
                out.append(why)
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _quantum(args) -> int:
    from math import gcd

    b, n = int(args.batch_size), int(args.n_envs)
    return b * n // gcd(b, n)


def _target_off_quantum(args) -> bool:
    t = int(_val(args, "rollout_target_samples", 0) or 0)
    return t > 0 and t % _quantum(args) != 0


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

    # ---- gen3_arch_surface_guard_v1: THE ARCH UMBRELLA ---------------------------------------
    CombinationCheck(
        "arch_umbrella_is_fresh_only", ("arch", "model"),
        lambda a: getattr(a, "arch", None) is not None and bool(getattr(a, "model", None)),
        "--arch production is a FRESH-run flag and is refused on a resume. A fork or restart "
        "INHERITS its parent's architecture surface from the recorded model_config.json "
        "(main.train.config's `_resolve`), which is the ONLY value that can load its weights; "
        "writing production's values over that would either FATAL at check_compatible or train a "
        "network the parent never was. Drop --arch, and read the ARCH SURFACE block that "
        "`--dry-run` / `python -m main.checkargs` print for a fork — it is INFO there, not a gate.",
        # A restart would hit the IDENTICAL refusal, so the launcher must give up, not crash-loop.
        exit_style="fatal_config",
        fresh_only=("--arch",)),

    # ---- gen3_winprob_critic_mode_v1: THE CRITIC MODE ---------------------------------------
    # `--critic winprob` makes the win-prob head the value function. Everything below either
    # CONTRADICTS that (a second critic, a normalizer with no scale to track, a reward whose
    # currency is not P(win)) or is a knob the mode SUBSUMES. Each one is refused rather than
    # ignored, because a silently-inert flag on a critic-route change is the failure this whole
    # module exists to end. `config.resolve_critic_mode` IMPLIES the coherent value for each of
    # them first, so a refusal here means the operator TYPED something incompatible.
    CombinationCheck(
        "winprob_critic_needs_a_head", ("critic", "win_prob_mode"),
        # unset resolves to 'shaping' under this critic (`config.resolve_critic_mode`'s implication)
        lambda a: _winprob(a) and _val(a, "win_prob_mode", "shaping") == "none",
        "--critic winprob requires --win-prob-mode read_only|shaping: the win-prob HEAD is the "
        "critic, and 'none' builds no head at all, so there would be no value function. "
        "(An unset --win-prob-mode is implied to 'shaping' — this fires only on an explicit "
        "'none'.) 'read_only' is the arm where the critic's gradient does not reach the trunk."),
    CombinationCheck(
        # The one check in this family pointing the OTHER way: not "winprob refuses X" but
        # "X requires winprob". Refused rather than ignored because the flag's whole premise is
        # that the BCE it reweights IS the value loss. Under `--critic shaped` that BCE is an
        # auxiliary readout, so a stratified weight there would re-price a
        # DIAGNOSTIC and leave the actual critic untouched — a silent no-op wearing the name of
        # the experiment, which is exactly what this module exists to end.
        "winprob_strata_needs_the_winprob_critic",
        ("win_prob_strata_weight", "critic"),
        lambda a: float(_val(a, "win_prob_strata_weight", 0.0) or 0.0) > 0.0 and not _winprob(a),
        "--win-prob-strata-weight > 0 requires --critic winprob. It reweights the win-prob BCE "
        "per OPPONENT CLASS so the between-class share of the objective rises (the head refit's "
        "§6 mechanism: only ~10-14% of the terminal label's variance lies between (cycle, "
        "opponent) cells). Under --critic shaped that BCE is an AUXILIARY readout and the critic "
        "is the scalar value net, so the weight would re-price a diagnostic and change nothing "
        "about the value function — and the `opp_class` label key it strata-fies on is declared "
        "under the win-prob label gate. Pass --critic winprob, or drop the flag."),
    CombinationCheck(
        # gen3_ridealong_rnd_variants_v1 (v127): the observation variants share base's frozen
        # target and normalisation, and base is the reference every variant is compared with —
        # the extractor raises the same thing (`flag_requires_test` pins it); this is the
        # launch-time copy `main.checkargs` can see.
        "rnd_variants_need_the_base_rnd_head",
        ("ridealong_rnd_variants", "ridealong_rnd"),
        lambda a: (str(_val(a, "ridealong_rnd_variants", "off") or "off") not in ("off", "none", "")
                   and not bool(_val(a, "ridealong_rnd", False))),
        "--ridealong-rnd-variants requires --ridealong-rnd: the observation variants share base's "
        "frozen target and normalisation, and base is the reference every variant is compared "
        "with. Pass --ridealong-rnd, or drop the variants."),
    # --- gen3_fork_v1: the FORK ARM's three refusals. ---------------------------------------
    CombinationCheck(
        # 🚨 THE ONE THAT IS NOT A CONVENTION. Under `winprob` the reward stream is the TERMINAL
        # WIN INDICATOR alone, so a branch's ENTIRE reward sequence is reconstructible from its
        # outcome bit — which is the only reason `fork_buffer.branch_rewards` can build one outside
        # the env. Under `shaped` the terminal is the SIGNED one, which the builder does not make.
        "fork_needs_the_winprob_critic", ("fork_fraction", "critic"),
        lambda a: float(_val(a, "fork_fraction", 0.0) or 0.0) > 0.0 and not _winprob(a),
        "--fork-fraction > 0 requires --critic winprob. A forked branch's transitions are built "
        "OUTSIDE the env, and only under this critic is a branch's reward sequence reconstructible "
        "from its outcome as the WIN INDICATOR (--terminal-indicator, --victory-value 1.0), which "
        "is what `fork_buffer.branch_rewards` builds. Under `shaped` the terminal is SIGNED "
        "(+V / -V / --draw-penalty), which the branch builder does not reproduce. Pass --critic "
        "winprob, or drop the flag."),
    CombinationCheck(
        # THE EXPENSIVE SILENT NO-OP: without the ring there is no replayable episode.
        # PYTHON CORE ONLY: on --env-core rust a fork replays the core's own finished input log
        # (`rust_rollout/fork.py`, forks.md §14), and --cf-records is refused there.
        "fork_needs_cf_records", ("fork_fraction", "cf_records"),
        lambda a: (float(_val(a, "fork_fraction", 0.0) or 0.0) > 0.0 and not _rust_core(a)
                   and not bool(_val(a, "cf_records", False))),
        "--fork-fraction > 0 requires --cf-records. A fork REPLAYS its episode to the forked turn "
        "and diverges there, and the replayable record lives in the `<run>/cf_records/` ring that "
        "--cf-records switches on; training otherwise keeps a single-slot stash it overwrites "
        "every episode. Without the ring every fork would fail to resolve and the run would be the "
        "unforked one under a forked name. ⚠️ Raise --cf-records-keep too: the ring is pruned "
        "GLOBALLY to the newest N while a production rollout finishes ~2,400 episodes, so at the "
        "default 512 the forks that DO resolve are the rollout's LATE ones — a selection bias, not "
        "just a shortfall. Pass --cf-records, or drop the flag."),
    CombinationCheck(
        "fork_refuses_strata_weight", ("fork_fraction", "win_prob_strata_weight"),
        lambda a: (float(_val(a, "fork_fraction", 0.0) or 0.0) > 0.0
                   and float(_val(a, "win_prob_strata_weight", 0.0) or 0.0) != 0.0),
        "--fork-fraction > 0 is incompatible with --win-prob-strata-weight != 0. The strata weight "
        "prices rows by `win_margin`, the normalised material margin the env's reward manager "
        "computes; a branch has no env, so every injected row carries the 0.0 FILL and would land "
        "in one stratum. The delivered strata dose would then be a function of the fork rate "
        "rather than of the flag. Drop one of the two."),
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
                   "all. The anti-stall pressure comes from the obs deadline clock. "
                   "Pass --draw-penalty 0.")),
    CombinationCheck(
        "winprob_critic_needs_the_indicator_terminal", ("critic", "terminal_indicator"),
        lambda a: _winprob(a) and not bool(_val(a, "terminal_indicator", True)),
        "--critic winprob requires --terminal-indicator. The critic is sigmoid(logit) in [0,1] "
        "and GAE mixes the REWARD with it, so a +V/-V terminal would put the return and the "
        "critic in different scales and every terminal TD error would carry a systematic, "
        "state-dependent offset (a loss reads `-V - V` against a truth of `0 - V`). The indicator "
        "terminal is what makes V(s) == E[return] hold."),
    CombinationCheck(
        "winprob_critic_needs_unit_victory_value", ("critic", "victory_value"),
        lambda a: _winprob(a) and float(_val(a, "victory_value", 1.0) or 0.0) != 1.0,
        lambda a: ("--critic winprob requires --victory-value 1.0 (got "
                   f"{float(_val(a, 'victory_value', 1.0)):g}). With the indicator terminal the "
                   "undiscounted return is `victory_value * 1{win}` while the critic is "
                   "sigmoid(logit) in [0,1], so the two agree at exactly one scale. At 1.0 the "
                   "return IS the win indicator and V(s) == P(win|s) with no approximation term "
                   "-- the identity the whole mode rests on.")),
    CombinationCheck(
        # Cutover loose end 2 (2026-09-30): the critic -> discount PAIRING (`critic_mode.critic_gamma`).
        # `resolve_critic_mode` IMPLIES 1.0 and its docstring hands a TYPED value to this module to
        # judge — which had no gamma row, so `--critic winprob --gamma 0.99` launched with the
        # identity broken. Every winprob run in models/ trained at 1.0; the shaped critic's gamma is
        # its own tunable (no row here). An UNTYPED mismatch is `resolve_config`'s pairing guard.
        "winprob_critic_needs_unit_gamma", ("critic", "gamma"),
        lambda a: _winprob(a) and getattr(a, "gamma", None) is not None
        and float(a.gamma) != _winprob_gamma(),
        lambda a: (f"--critic winprob requires --gamma {_winprob_gamma():g} (got {float(a.gamma):g}). "
                   "The discount is PAIRED with the critic (agents.model.critic_mode.critic_gamma): "
                   "with the terminal-only indicator reward and the 250-turn hard cap, V(s) == "
                   "P(win|s) holds exactly only at gamma 1 -- at 0.9999 over 250 turns the return is "
                   "discounted by 0.975, the same order as the calibration error this critic exists "
                   "to remove. Drop --gamma (it is implied) or pass --gamma 1.0.")),

    # ---- the distributional critic --------------------------------------------------------

    # ---- gen3_supply_guard_v2: a PFSP lever with no possible supply ---------------------------
    CombinationCheck(
        # `--pfsp-scale` weights the self-play POOL by measured sentinel win-rates. Without
        # --self-play there is no pool and no sentinel, so the weighting is uniform forever while
        # the argv and the recorded config both say PFSP. FATAL_CONFIG: identical on every restart.
        "pfsp_scale_needs_self_play", ("pfsp_scale", "self_play"),
        lambda a: _positive(_val(a, "pfsp_scale", 0.0)) and not bool(_val(a, "self_play", False)),
        lambda a: (f"\n[SUPPLY] FATAL: --pfsp-scale {_val(a, 'pfsp_scale', 0.0):g} weights the "
                   f"self-play POOL by measured sentinel win-rates, but --self-play is off — there "
                   f"is no pool to weight, so sampling would stay uniform for the whole run. Pass "
                   f"--self-play, or drop --pfsp-scale."),
        exit_style="fatal_config"),
    CombinationCheck(
        # `--team-pfsp` measures per-team win-rates ONLY on self-play POOL battles or EXPLOITER
        # target battles (wrappers._maybe_record_team_pfsp — bots are excluded by design). With
        # neither on, not one game is ever counted and team sampling stays uniform.
        "team_pfsp_needs_self_play_or_exploiter", ("team_pfsp", "self_play", "exploiter"),
        lambda a: (_val(a, "team_pfsp", "off") != "off"
                   and not bool(_val(a, "self_play", False)) and not getattr(a, "exploiter", None)),
        lambda a: (f"\n[SUPPLY] FATAL: --team-pfsp {_val(a, 'team_pfsp', 'off')} counts per-team "
                   f"games ONLY on self-play pool battles or exploiter-target battles (bots are "
                   f"excluded), and this run has neither --self-play nor --exploiter — no team "
                   f"win-rate would ever be measured and sampling would stay uniform. Pass "
                   f"--self-play, or drop --team-pfsp."),
        exit_style="fatal_config"),
    CombinationCheck(
        "supply_starve_cycles_parses", ("supply_starve_cycles",),
        lambda a: _starve_spec_error(a) is not None,
        lambda a: str(_starve_spec_error(a))),

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
        "generalist / self-play training, whose objective is to ABSORB per-team divergence, "
        "the OPPOSITE of distilling the consensus."),
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
        "obs_source_core_needs_rust_bridge", ("obs_source", "use_bridge"),
        lambda a: _val(a, "obs_source", None) == "core" and _val(a, "use_bridge", "rust") != "rust",
        "--obs-source core requires --use-bridge rust — the core that builds the row lives in "
        "the rust sim_bridge child (gen3_core_obs_source_v1)"),
    CombinationCheck(
        "cf_records_needs_bridge", ("cf_records", "use_bridge"),
        lambda a: bool(a.cf_records) and _val(a, "use_bridge", "rust") == "off",
        "--cf-records requires the in-process bridge (--use-bridge node|rust) — the "
        "reconstruction record is a bridge frame; a websocket run emits none"),
    CombinationCheck(
        # gen3_supply_guard_v1 — THE CLASS `ai_v12_12_ladder_cflabels` fell into (10M steps at
        # --cf-winprob-coef 0.5, ZERO labels): a live cf-buffer coefficient whose supplier cannot
        # exist. The trainer-spawned producer labels the `cf_records/` ring; without the tap it has
        # nothing to label, so the coefficient would fold nothing for the whole run.
        "cf_consumer_needs_label_supply",
        ("cf_winprob_coef", "cf_evidential_coef", "cf_twin_coef", "cf_shadow_coef",
         "q_winprob_coef", "q_winprob_onpolicy_coef", "cf_records", "cf_label_supply"),
        lambda a: (any(_positive(_val(a, c, 0.0)) for c in _CF_CONSUMER_COEFS)
                   and (_val(a, "cf_label_supply", "producer") or "producer") == "producer"
                   and not bool(_val(a, "cf_records", False))),
        lambda a: (
            f"\n[SUPPLY] FATAL: {', '.join(c for c in _CF_CONSUMER_COEFS if _positive(_val(a, c, 0.0)))}"
            f" is live, but its label SUPPLY cannot exist: the cf label producer (which "
            f"--cf-label-supply producer starts at launch) labels the reconstruction records "
            f"--cf-records rings into <run>/cf_records/, and --cf-records is off. The coefficient "
            f"would fold NOTHING for the whole run (ai_v12_12_ladder_cflabels: 10M steps, 0 labels). "
            f"Pass --cf-records, or set the coefficient(s) to 0."),
        exit_style="fatal_config"),
    CombinationCheck(
        # gen3_cf_label_duty_cycle_v1 — a quantity nobody was computing. FATAL_CONFIG, not
        # parser.error: a restart would hit the identical config, so the launcher must give up.
        "cf_label_duty_cycle_floor",
        ("cf_twin_coef", "cf_winprob_coef", "cf_records", "cf_label_lag_steps",
         "checkpoint_every_steps", "n_envs"),
        _cf_duty_cycle_starved, _cf_duty_cycle_message, exit_style="fatal_config"),

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

    # ---- the env core (M5 Lane G): what `--env-core rust` serves, refused by name -------------
    CombinationCheck(
        "env_core_rust_needs_the_winprob_critic", ("env_core", "critic"),
        lambda a: _rust_core(a) and not _winprob(a),
        "--env-core rust requires --critic winprob: the Rust env produces NO terminal observation "
        "(Lane D's decision), and a shaped critic bootstraps a truncation from one (F-LD-2)",
        exit_style="fatal_config"),
    CombinationCheck(
        "env_core_rust_unported_paths", _ENV_CORE_UNPORTED_DESTS,
        lambda a: _rust_core(a) and bool(_env_core_unported(a)),
        lambda a: ("--env-core rust does not serve these paths yet: " + "; ".join(_env_core_unported(a))
                   + ". Turn them off, or run --env-core python (designs/training/rust_collector.md, "
                   "'What --env-core rust refuses')"),
        exit_style="fatal_config"),
    CombinationCheck(
        # gen3_fork_rust_v1 (forks.md §14.3): a branch replays the parent's draws BY KEY; a per-env
        # torch.Generator stream cannot be replayed per branch.
        "fork_rust_needs_keyed_opponent_sampling", ("fork_fraction", "env_core", "opponent_sampling"),
        lambda a: (_rust_core(a) and float(_val(a, "fork_fraction", 0.0) or 0.0) > 0.0
                   and _val(a, "opponent_sampling", "keyed") != "keyed"),
        "--fork-fraction > 0 under --env-core rust requires --opponent-sampling keyed: a branch is the "
        "parent's game under the PARENT's draw keys (common random numbers, designs/training/forks.md "
        "§14.3), and the generator mode's per-env stream cannot be replayed per branch",
        exit_style="fatal_config"),
    CombinationCheck(
        # gen3_fork_rust_v1 (forks.md §14.2): branch games join the completed-game FIFO.
        "fork_rust_needs_complete_game_trigger", ("fork_fraction", "env_core", "rollout_trigger"),
        lambda a: (_rust_core(a) and float(_val(a, "fork_fraction", 0.0) or 0.0) > 0.0
                   and _val(a, "rollout_trigger", "complete_game") != "complete_game"),
        "--fork-fraction > 0 under --env-core rust requires --rollout-trigger complete_game: a branch is "
        "a complete game that joins the completed-game FIFO beside its parent (designs/training/forks.md "
        "§14.2); the window fill is the parity schedule",
        exit_style="fatal_config"),
    CombinationCheck(
        "env_core_flags_need_the_rust_core", _ENV_CORE_ONLY_DESTS,
        lambda a: not _rust_core(a) and any(_typed(a, d) for d in _ENV_CORE_ONLY_DESTS),
        "the rollout-collector flags (--rollout-trigger / --rollout-target-samples / --rollout-target-band / "
        "--version-pinning / --trainee-slots / --t2-* / --rust-eval-envs) act only under --env-core rust — typed on the python "
        "env core they would be silently inert"),
    CombinationCheck(
        "rollout_target_on_the_quantum", ("env_core", "rollout_target_samples", "batch_size", "n_envs"),
        lambda a: _rust_core(a) and _target_off_quantum(a),
        lambda a: (f"--rollout-target-samples {int(_val(a, 'rollout_target_samples', 0) or 0):,} must be a "
                   f"multiple of lcm(--batch-size {a.batch_size}, --n-envs {a.n_envs}) = {_quantum(a):,}: no "
                   "ragged micro-batch may reach the compiled learner graph, and the buffer keeps its "
                   "[n_steps, n_envs] shape"),
        exit_style="fatal_config"),

    # ---- T17 MIRRORED TEAM PAIRS (gen3_mirrored_pairs_v1): the Python eval path's prerequisites -----
    # A mirrored pair's two games share ONE battle seed, so the Python eval path plays every game under
    # the per-GAME seed rule — which only the in-process bridge, one game in flight, can honour.
    CombinationCheck(
        "mirrored_pairs_need_the_bridge", ("eval_mirrored_pairs", "env_core", "use_bridge"),
        lambda a: bool(_val(a, "eval_mirrored_pairs", False)) and not _rust_core(a)
        and _val(a, "use_bridge", "rust") == "off",
        "--eval-mirrored-pairs needs the in-process bridge (--use-bridge rust|node): a mirrored pair shares "
        "one battle seed, and a Showdown server mints its own dice"),
    CombinationCheck(
        "mirrored_pairs_need_one_game_in_flight",
        ("eval_mirrored_pairs", "env_core", "eval_concurrency_per_worker"),
        lambda a: bool(_val(a, "eval_mirrored_pairs", False)) and not _rust_core(a)
        and int(_val(a, "eval_concurrency_per_worker", 1) or 1) != 1,
        "--eval-mirrored-pairs needs --eval-concurrency-per-worker 1: every game is seeded, and "
        "overlapping games would consume the seeded streams out of order"),

    # ---- T6 SPRT PROMOTION (gen3_sprt_promotion_v1) ------------------------------------------------
    CombinationCheck(
        "promotion_sprt_needs_self_play", ("promotion_sprt", "self_play"),
        lambda a: bool(_val(a, "promotion_sprt", False)) and not bool(_val(a, "self_play", False)),
        "--promotion-sprt decides POOL promotion — without --self-play there is no pool and nothing to promote"),
    CombinationCheck(
        "promotion_sprt_ignores_the_threshold", ("promotion_sprt", "promote_threshold"),
        lambda a: bool(_val(a, "promotion_sprt", False)) and _typed(a, "promote_threshold"),
        "--promote-threshold does nothing under --promotion-sprt (the SPRT decides; a typed threshold "
        "would be silently inert)"),
    CombinationCheck(
        "promotion_sprt_needs_the_bridge", ("promotion_sprt", "env_core", "use_bridge"),
        lambda a: bool(_val(a, "promotion_sprt", False)) and not _rust_core(a)
        and _val(a, "use_bridge", "rust") == "off",
        "--promotion-sprt plays seeded mirrored pairs: the Python eval path needs the in-process bridge "
        "(--use-bridge rust|node)"),
    CombinationCheck(
        "promotion_sprt_needs_one_game_in_flight",
        ("promotion_sprt", "env_core", "eval_concurrency_per_worker"),
        lambda a: bool(_val(a, "promotion_sprt", False)) and not _rust_core(a)
        and int(_val(a, "eval_concurrency_per_worker", 1) or 1) != 1,
        "--promotion-sprt plays seeded mirrored pairs: the Python eval path needs "
        "--eval-concurrency-per-worker 1"),
)


def _floor_is_non_default(args) -> bool:
    """A NON-DEFAULT `--move-candidate-floor`; the default is not flagged, it is just the default."""
    from agents.model.damage_tables import _PRIOR_FLOOR
    return _val(args, "move_candidate_floor", _PRIOR_FLOOR) != _PRIOR_FLOOR


BY_NAME = {c.name: c for c in COMBINATION_CHECKS}


def fresh_only_flags() -> Tuple[str, ...]:
    """Every argv option the trainer REFUSES on a resume (`--model` set), in declaration order.

    The launcher's same-run restart builds its child argv from the RESUME role, and strips these
    first: the resumed run INHERITS what they applied from its recorded `model_config.json`, so
    nothing is lost — and leaving them in makes every restart die on the refusal.
    """
    out: List[str] = []
    for check in COMBINATION_CHECKS:
        for flag in check.fresh_only:
            if flag not in out:
                out.append(flag)
    return tuple(out)


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
