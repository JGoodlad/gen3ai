"""Phase 1 — CONFIG RESOLUTION: turn a parsed argv into a coherent, validated run configuration.

Three jobs, in order, and the order is load-bearing:

* **DESUGAR** the umbrella flags (`--unified-moves` -> `--unified-damage` -> the component
  toggles, `--damage-matrices` -> its two bools) BEFORE `_resolve`, so they are not None-filled
  from a saved checkpoint.
* **`_resolve`** every version-checked structural toggle: `None` (not passed) INHERITS the saved
  value, so the documented flagless resume (`--model X --steps N`) keeps the architecture it was
  trained with instead of falling back to OFF and FATAL-ing at `check_compatible`.
* **VALIDATE** — for a training-only coefficient, the `parser.error` here is the ONLY gate there
  is (nothing version-checks it), which is why the checks are exhaustive rather than a sample.

`args` is MUTATED in place; the handful of values that are not attributes of `args` come back in
`ResolvedRunConfig`.
"""
import dataclasses
from typing import Any

from agents.model.damage_tables import _MIN_PRIOR_FLOOR, _PRIOR_FLOOR
from agents.training.watchdog import start_orphan_watchdog
from main.launcher.ipc import emit
from main.train.checkpoint_state import _load_saved_version
from main.train.combination_checks import refuse_first
from main.train.compile_flags import (
    resolve_compile_opponents_preload, resolve_compile_trainer_auto,
)
from main.train.constants import (
    CF_DUTY_CYCLE_FLOOR, DEFAULT_DISTILL_TEAM_BIAS, cf_label_duty_cycle,
    checkpoint_interval_env_steps, checkpoint_save_freq_vec_calls,
)
from poke_env import LocalhostServerConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration
from utils.logging.levels import LogLevel


@dataclasses.dataclass(frozen=True)
class ResolvedRunConfig:
    """The three resolved values that are NOT attributes of `args`."""

    server_config: Any
    annealing_mode: bool
    log_level: LogLevel


#: A TIMEOUT this many times a clean loss stops being a tie-breaker and starts being the objective.
#: The validated composition is 35/30 = 1.17x; the clean-world ruling is 1/1 = 1.0x. 3.0 is loose
#: enough that no composition anyone has actually launched trips it.
_DRAW_SCALE_RATIO = 3.0
#: How many value-dist ATOMS the achievable raw-return range must span before the critic can be
#: said to resolve it. HL-Gauss smooths each target with sigma = 0.75*delta, so one target already
#: occupies ~3 atoms; below ~8 the whole return range holds fewer than three distinguishable
#: levels, and under `--value-from-dist` that quantized E[Z] IS the critic feeding GAE.
_MIN_SUPPORT_ATOMS = 8.0


def _terminal_scale_guards(args) -> None:
    """The two SCALE questions `gen3_clean_world_config_v1` opened by making the terminal a flag.

    `--victory-value` (v105) is the first flag that can change the RETURN SCALE, and two other,
    older flags are quietly denominated in that same scale. Neither pairing had a check, because
    each half was validated on its own — the `value_from_dist` (M2) shape exactly.

    1. **`--draw-penalty` vs `--victory-value`.** The v105 guard above tests the ORDERING (a draw
       must not beat a loss) and passes the far more likely mistake: typing `--victory-value 1.0`
       and inheriting the -35.0 default, i.e. a timeout 35x a clean loss. The composition is then
       not "1 TERMINAL" at all — it is a stall-avoidance objective with a win bonus, and no metric
       downstream distinguishes the two.

    2. **`--value-dist-{vmin,vmax,bins}` vs the terminal.** With PopArt ON the HL-Gauss target is
       `popart.normalize(returns)`, so the support is in units of standard deviations and the raw
       terminal says nothing about it — the guard is skipped, which is every run ever launched.
       With PopArt OFF (the registered clean/sparse arms, ledger 2d38a4a) the target is the RAW
       return and the two ARE in the same units, so the support either brackets the reachable
       returns with resolution to spare or it silently destroys the critic: too WIDE quantizes the
       whole range into a handful of atoms, too NARROW saturates the edge atoms that absorb the
       out-of-support tails.

    Warnings, never refusals: a wide support may be a deliberate choice ahead of a reward change,
    and a launch that works today must not become a `FATAL_CONFIG`. But they are stated at launch,
    because both defects train correctly toward the wrong thing.
    """
    victory = getattr(args, "victory_value", None)
    if victory is None or float(victory) <= 0.0:
        return                                   # refused above; nothing to say about a bad scale
    if bool(getattr(args, "terminal_indicator", False)):
        # gen3_winprob_critic_mode_v1: under the WIN INDICATOR every non-win pays exactly 0.0, so
        # `draw_penalty` is not merely unused — it is inapplicable, and both guards below are
        # statements ABOUT it. The ORDERING one would fire on every such run and say the opposite
        # of the truth ("running the clock out is the best non-winning outcome") when a timeout and
        # a loss are the SAME payoff by construction; the SCALE one would divide by a magnitude
        # that is not in the stream. A warning that is false on a supported configuration teaches
        # the reader to skip the whole family, so it is suppressed rather than reworded.
        return
    victory = float(victory)
    draw = float(args.draw_penalty) if getattr(args, "draw_penalty", None) is not None else -victory
    if abs(draw) > _DRAW_SCALE_RATIO * victory:
        print(f"[Reward] ⚠️ TERMINAL SCALE: --draw-penalty {draw:g} is {abs(draw) / victory:.0f}x a "
              f"clean loss (-{victory:g}). The ordering is right, but the MAGNITUDE makes the "
              f"250-turn timeout — not the win — the dominant term in the reward stream, so a run "
              f"advertised as '1 TERMINAL' is really a stall-avoidance objective. The validated "
              f"pairing is 30/-35 (1.2x) and the clean-world ruling is draw = loss: pass "
              f"--draw-penalty {-victory:g} with --victory-value {victory:g}.")
    if str(getattr(args, "value_dist_mode", "none")) == "none" or getattr(args, "use_popart", False):
        return
    bins = int(getattr(args, "value_dist_bins", 0) or 0)
    vmin, vmax = float(args.value_dist_vmin), float(args.value_dist_vmax)
    if bins < 2 or not (vmax > vmin):
        return                                   # already a parser.error above
    reach = max(victory, abs(draw))              # the largest |return| the terminal can produce
    delta = (vmax - vmin) / (bins - 1)
    atoms = (2.0 * reach) / delta
    outside = (reach > vmax) or (-reach < vmin)
    if not outside and atoms >= _MIN_SUPPORT_ATOMS:
        return
    why = ("the support does NOT BRACKET them — HL-Gauss absorbs the out-of-support mass into the "
           "EDGE atoms, so the critic cannot represent that outcome at all"
           if outside else
           f"they span only {atoms:.1f} of {bins} atoms (bin width {delta:.3g}), so the critic is "
           f"quantized to ~{delta:.3g} on a +-{reach:g} scale")
    print(f"[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE: PopArt is OFF, so the HL-Gauss target "
          f"is the RAW return and the atom support is in the SAME units. Returns reach +-{reach:g} "
          f"(--victory-value {victory:g}, --draw-penalty {draw:g}) and {why}. Size "
          f"--value-dist-vmin/--value-dist-vmax to the terminal (a few times +-{reach:g}), or turn "
          f"PopArt on. This matters most under --value-from-dist, where E[Z] IS the critic feeding "
          f"GAE and nothing downstream distinguishes a resolution-starved critic from a fitted one "
          f"(value_dist/mean_abs_err looks BETTER as the support widens).")


def _adaptive_batch_guards(args, parser) -> None:
    """Validate the `--adaptive-batch` family — the ONLY gate on it (training-only, never recorded).

    The one non-obvious refusal is the last: `--adaptive-batch policy` steers by
    `train/noise_scale_ratio_policy`, which only the PER-TERM probe produces. With the probe
    switched off by `$GEN3AI_NOISE_SCALE_PER_TERM=0` that series never exists, so the controller
    would sit in its `unavailable` branch for the whole run — a loop that silently does nothing,
    which is the failure mode hardest to notice and cheapest to refuse.
    """
    import os
    mode = getattr(args, "adaptive_batch", "off")
    if mode == "off":
        return
    env = os.environ.get("GEN3AI_NOISE_SCALE_PER_TERM")
    if mode == "policy" and env is not None and env.strip().lower() in ("", "0", "false", "off", "no"):
        parser.error("--adaptive-batch policy steers by train/noise_scale_ratio_policy, which only "
                     "the PER-TERM noise-scale probe emits — and $GEN3AI_NOISE_SCALE_PER_TERM is "
                     f"set to {env!r}, which disables it. Unset it, or use --adaptive-batch total.")


def _announce_cf_duty_cycle(args) -> None:
    """PRINT the counterfactual label DUTY CYCLE, and REFUSE a starved one.

    THE DEFECT THIS MAKES UNREPRESENTABLE (`ai_v9_29_rev1_0823`, 2026-08-23). The label producer
    can only stamp labels with the step of the newest `checkpoints/` zip, and `cf_label_buffer`
    expires a row more than `--cf-label-lag-steps` behind the live policy. So the two flags define
    a fraction — and NOBODY WAS COMPUTING IT. At the hardcoded 50 000 VEC-CALL cadence and
    `--n-envs 48` the checkpoint interval is 2 400 000 env steps against a 150 000-step bound: a
    6.25% duty cycle, observed as **6 labels ingested against 255 expired in two hours**, with
    every counter on both sides reading healthy (the producer was producing; the buffer was
    expiring; neither knew about the other's number).

    So the number is now PRINTED on every launch that has both halves on, healthy or not — a
    quantity nobody computes is how this shipped — and refused below the floor. A refusal exits
    `FATAL_CONFIG` rather than `parser.error`, because restarting would hit the identical config
    every time and the launcher must give up rather than loop.

    `--debug` is exempt: a smoke has one env and runs for thousands of steps, so its duty cycle is
    an artifact of the smoke rather than a statement about the recipe.
    """
    on = bool((args.cf_twin_coef and args.cf_twin_coef > 0)
              or (args.cf_winprob_coef and args.cf_winprob_coef > 0))
    if not (on and args.cf_records):
        return
    n_envs = 1 if args.debug else int(args.n_envs)
    every = getattr(args, "checkpoint_every_steps", None)
    vec_calls = checkpoint_save_freq_vec_calls(every, n_envs)
    interval = checkpoint_interval_env_steps(every, n_envs)
    duty = cf_label_duty_cycle(args.cf_label_lag_steps, interval)
    shown = "unbounded (--cf-label-lag-steps 0 = labels never expire)" if duty == float("inf") \
        else f"{duty:.1%}"
    line = (f"🧾 [CF] label DUTY CYCLE {shown} — --cf-label-lag-steps "
            f"{args.cf_label_lag_steps:,} / {interval:,} env-steps between checkpoints "
            f"({vec_calls:,} vec-calls x {n_envs} envs)")
    if args.debug:
        emit(line + "  [--debug: the floor is not enforced]")
        return
    if duty >= CF_DUTY_CYCLE_FLOOR:
        emit(line)
        return


def inherit_saved_flag(args, saved_ver, name, default) -> bool:
    """THE RESUME INHERITANCE RULE, in one function: `None` on the CLI means INHERIT.

    `resolve_config`'s `_resolve` is this, bound to the process's own `args` + saved `ModelVersion`.
    It is module-level so `main.checkargs` can build the SAME effective namespace a launch would —
    argv overlaid on the checkpoint's recorded config — and run the combination checks on THAT,
    instead of on the argv alone. Absence on the command line carries information only because this
    rule exists, so the rule cannot live inside the closure that consumes it.

    Returns True when the value came from the SAVED config (what checkargs reports as inherited);
    False when it was already set on the CLI, or fell through to `default`.
    """
    if getattr(args, name) is not None:
        return False
    if saved_ver is not None and hasattr(saved_ver, name):
        setattr(args, name, getattr(saved_ver, name))
        return True
    setattr(args, name, default)
    return False


def is_fold(args) -> bool:
    """A fold is actually RUNNING: at least one `--distill-teacher` AND `--distill-coef > 0`."""
    return bool(getattr(args, "distill_teacher", None)
                and getattr(args, "distill_coef", None)
                and args.distill_coef > 0)


def default_anchor_monitor(args) -> bool:
    """Would `--distill-anchor-monitor` default ON for this config? (Pure — no mutation, no print.)

    Module-level and pure for the same reason `inherit_saved_flag` is: `main.checkargs` has to
    reach the same verdict a launch reaches. It could not, and the cost was a FALSE POSITIVE in the
    other direction from the C1/G5 family — a perfectly good fold argv reported as "--distill-stop
    requires the anchor MONITOR", because offline the monitor still read as unset.

    `coef_on` / `proj_on` already attach the frozen parent and already emit every collateral meter,
    so defaulting the monitor on beside them would be a second name for one thing. And WILL a fold
    parent resolve? `resolve_anchor_parent` tries an explicit `--distill-anchor-parent`, then the
    run dir's `lineage` block, then `--model`; the lineage route only ever names a parent for a run
    that was ITSELF launched from one, so on this launch those two flags decide it.
    """
    coef_on = bool(getattr(args, "distill_anchor_coef", None)
                   and args.distill_anchor_coef > 0)
    proj_on = getattr(args, "distill_anchor_mode", None) == "grad_project"
    parent_available = bool(getattr(args, "distill_anchor_parent", None)
                            or getattr(args, "model", None))
    return bool(is_fold(args) and not coef_on and not proj_on and parent_available)


def _resolve_fold_instruments(args) -> str:
    """THE TWO PURE INSTRUMENTS DEFAULT ON FOR A FOLD (`gen3_distill_instruments_default_v1`).

    Returns the default `--distill-stop` mode; sets `args.distill_anchor_monitor` and the two
    `*_source` provenance strings that ride into `metadata.json`'s `cli_args`.

    WHY A DEFAULT AND NOT A FLAG. `--distill-anchor-monitor` (the off-slice collateral meters) and
    `--distill-stop warn` (the plateau-AND-rise detector in its log-only mode) are pure
    INSTRUMENTS: the monitor attaches no loss term and changes no parameter, and `warn` only emits
    a launcher event plus `distill/stop_signal`. As opt-ins they were carried on three of seven
    fold arms in one batch and omitted on the other four, so the pre-registered cross-check could
    not be run on the arms that mattered — and an ABSENT series in a column of numbers reads like a
    zero. An instrument that costs nothing and whose absence is unreadable belongs on by default.

    THE CONDITION is "a fold is actually running": at least one `--distill-teacher` AND
    `--distill-coef > 0`. Both halves are load-bearing rather than cautious — the anchor's
    off-slice split reads the `distill_mask` obs key, which the env emits only for a run with a
    live distill term, and `resolve_config` refuses the anchor without one. A teacher named beside
    `--distill-coef 0` (the distillation-free arm) is therefore NOT a fold here, and stays exactly
    as it is today.

    THE DEFAULT YIELDS; AN EXPLICIT FLAG REFUSES — the `--compile-trainer` rule, one flag over. A
    fold whose parent cannot be resolved WARNS and leaves the instrument off (recorded as
    `default-no-parent`, so the absence is visible rather than silent); an explicit
    `--distill-anchor-monitor` still reaches the FATAL in `build_callbacks`, because there the
    operator asked for something that cannot be delivered.
    """
    fold = is_fold(args)
    coef_on = bool(args.distill_anchor_coef and args.distill_anchor_coef > 0)
    proj_on = args.distill_anchor_mode == "grad_project"
    if args.distill_anchor_monitor is None:
        args.distill_anchor_monitor = default_anchor_monitor(args)
        if args.distill_anchor_monitor:
            args.distill_anchor_monitor_source = "default"
            emit("📏 --distill-anchor-monitor ON by default (a fold is running: --distill-teacher "
                 "with --distill-coef > 0). Attaches the FROZEN fold parent and emits "
                 "distill/collateral_kl_vs_parent + the off-slice meters — no loss term, no "
                 "parameter changed, one frozen no_grad forward per minibatch. "
                 "--no-distill-anchor-monitor turns it off.")
        elif fold and not coef_on and not proj_on:
            args.distill_anchor_monitor_source = "default-no-parent"
            emit("⚠️ --distill-anchor-monitor would be ON by default here (a fold is running), but "
                 "no fold parent can be resolved — no --distill-anchor-parent and no --model. "
                 "Leaving the instrument OFF rather than refusing to launch; the collateral meters "
                 "will not exist for this run (cli_args records "
                 "distill_anchor_monitor_source=default-no-parent). Pass "
                 "--distill-anchor-parent to get them.")
        else:
            args.distill_anchor_monitor_source = "default-off"
    else:
        args.distill_anchor_monitor = bool(args.distill_anchor_monitor)
        args.distill_anchor_monitor_source = "cli"
    anchor_on = coef_on or proj_on or args.distill_anchor_monitor
    stop_default = "warn" if (fold and anchor_on) else "off"
    if args.distill_stop is not None:
        args.distill_stop_source = "cli"
    elif stop_default != "off":
        args.distill_stop_source = "default"
        emit("🛑 --distill-stop warn by default (a fold is running with the frozen parent "
             "attached). LOG-ONLY: a launcher event + distill/stop_signal when "
             "distill/teacher_agreement_on_slice has plateaued AND "
             "distill/collateral_kl_vs_parent is rising. Nothing is annealed and nothing stops; "
             "pass --distill-stop off to silence it, or anneal|abort to give it teeth.")
    else:
        args.distill_stop_source = "default-off"
    return stop_default


def desugar_umbrella_flags(args) -> None:
    """The UMBRELLA desugars, in one place: `--unified-moves` -> `--unified-damage` -> the
    component toggles, `--predict-unrevealed-mon-moves`, and `--damage-matrices` -> its two bools.

    Module-level for the same reason `inherit_saved_flag` is: `main.checkargs` has to build the
    SAME namespace a launch builds before it can read `combination_checks` on it. A flagless run
    resolves `--unified-moves` to 'both', which turns `--damage-op` / `--move-latent` ON — so a
    checker that skipped this would report every damage-family dependency as unsatisfied on a
    command that launches. Mutates `args` in place; prints the two operator notes it always did.
    """
    # --unified-moves is the umbrella over the WHOLE move system: it sets --unified-damage to the same
    # level (so the op/belief/outgoing desugar below runs) AND turns on the move latent + its grading.
    # Applied BEFORE the --unified-damage desugar so the level flows through. v24.
    #
    # DEFAULT-ON (2026-08-04, owner decision): the unified move system is the model — every production
    # config since v24 runs it, and the off path is an ablation baseline, not a supported configuration.
    # A None (flagless) invocation resolves to:
    #   * FRESH run → 'both' (the full system), with a printed note;
    #   * RESUME (--model) → NO desugar — the component toggles stay None and _resolve below inherits the
    #     checkpoint's saved arch verbatim (the same flagless-resume contract every structural toggle
    #     follows), so a resume can never be version-FATALed by a default. A launcher restart that
    #     forwarded the original explicit flag is likewise unchanged.
    # An EXPLICIT 'off' still works (fresh ablation baselines need it) but is DEPRECATED and warns.
    if args.unified_moves is None:
        if args.model:
            args.unified_moves = "off"     # no desugar — inherit the saved component toggles via _resolve
        else:
            args.unified_moves = "both"
            print("[Arch] --unified-moves defaults to 'both' (the unified move system is the model; "
                  "pass --unified-moves off explicitly for the DEPRECATED ablation baseline).")
    elif args.unified_moves == "off":
        print("[Arch] DEPRECATED: --unified-moves off — the non-unified path is an ablation baseline "
              "only (no move belief, no damage op, no discrete move-space). It keeps working, but new "
              "features target the unified system.")
    if getattr(args, "unified_moves", "off") != "off":
        if getattr(args, "unified_damage", "off") == "off":
            args.unified_damage = args.unified_moves
        if args.move_latent is None:
            args.move_latent = True
        if args.move_belief_latent_coef is None:
            args.move_belief_latent_coef = 0.05
        # gen3_unified_topk_incoming_v1: the umbrella also turns on the DISCRETE top-K incoming block at the
        # default K (the deps — damage_op + move_latent — are satisfied above/below). An explicit
        # --damage-topk wins (incl. --damage-topk 0 to A/B it off under --unified-moves).
        if args.damage_topk_k is None:
            from agents.model.features_extractor import _DMG_TOPK_DEFAULT_K
            args.damage_topk_k = _DMG_TOPK_DEFAULT_K


    # --unified-damage desugars into the component flags BEFORE _resolve (so they aren't None-filled from a
    # saved version). When not 'off' it forces damage_op + prior fusion + (for 'both') the outgoing block,
    # and defaults the move-belief mode to 'revealed' unless the user set it explicitly (so
    # `--unified-damage both --move-belief-mode both` still guesses unrevealed mons' moves).
    if getattr(args, "unified_damage", "off") != "off":
        if args.move_belief_mode is None:
            args.move_belief_mode = "revealed"
        args.damage_op = True
        args.move_prior_fusion = True
        args.damage_outgoing = (args.unified_damage == "both")

    # Explicit CLARITY knob: "predict the moves of mons we haven't even SEEN". OFF
    # (--no-predict-unrevealed-mon-moves) zeros BOTH hidden-mon move-prediction paths — the
    # hidden-opponent BeliefHead's moves-BCE (`opp_belief_moves_weight` → 0) AND any MoveBelief
    # unrevealed leg (`move_belief_mode` 'unrevealed'/'both' → 'revealed'). The REVEALED-mon move belief
    # (predict a SEEN mon's unseen slots) and the SPECIES belief on hidden mons are UNTOUCHED. A desugar
    # into existing fields (no new version field); unset/True preserves the current behavior.
    if getattr(args, "predict_unrevealed_mon_moves", None) is False:
        args.opp_belief_moves_weight = 0.0
        if args.move_belief_mode in ("unrevealed", "both"):
            args.move_belief_mode = "revealed"

    # gen3_per_move_matrices_v1: --damage-matrices desugars to the two bool toggles BEFORE _resolve (so a
    # resume inherits them). None ⇒ let _resolve inherit/default; an explicit value wins. The INCOMING matrix
    # is the ENRICHED top-K — it REUSES --damage-topk K as its K (the one "how many opp moves" knob) and
    # REPLACES the lean top-K block at that K. Default the K to _DMG_TOPK_DEFAULT_K if unset (so it works
    # standalone); an explicit --damage-topk (or --unified-moves' default) wins.
    if getattr(args, "damage_matrices", None) is not None:
        args.damage_matrices_outgoing = args.damage_matrices in ("outgoing", "both")
        args.damage_matrices_incoming = args.damage_matrices in ("incoming", "both")
        if args.damage_matrices_incoming and not args.damage_topk_k:
            from agents.model.features_extractor import _DMG_TOPK_DEFAULT_K   # local: needed without --unified-moves
            args.damage_topk_k = _DMG_TOPK_DEFAULT_K     # the matrix's K = --damage-topk (default 5)
    else:
        if not hasattr(args, "damage_matrices_outgoing"):
            args.damage_matrices_outgoing = None
        if not hasattr(args, "damage_matrices_incoming"):
            args.damage_matrices_incoming = None


def resolve_critic_mode(args, saved_ver=None) -> None:
    """Resolve `--critic` and imply the three tri-state flags the `winprob` value settles, in place.

    Module-level, and called before the `_resolve` sweep, for `desugar_umbrella_flags`' exact
    reason: `main.checkargs` has to build the SAME effective namespace a launch builds before it
    can read `combination_checks` on it, and every refusal in the critic family reads a value this
    function fills. A checker that skipped it would report a `winprob` command's implied
    `--win-prob-mode shaping` as a missing dependency on a command that launches.

    Under `shaped` (the default) nothing beyond the mode itself is assigned, so a run that does not
    type the flag is byte-identical.

    **IMPLIED under `winprob` — exactly the flags whose "unset" is REPRESENTABLE:**

    ``win_prob_mode``  'shaping'  the head must EXIST to be the critic ('none' is refused)
    ``gamma``          1.0        V(s) is then EXACTLY P(win|s) (see the flag's help)
    ``use_popart``     False      a bounded stationary Bernoulli payoff has no scale to track

    All three carry an argparse default of `None`, so an unset flag is distinguishable from a
    typed one and the implication can never overwrite an operator's choice — it is then judged by
    `combination_checks`. Running BEFORE the inheritance sweep is load-bearing: a fork of a
    `shaped` parent would otherwise inherit that parent's `use_popart=True` / `win_prob_mode='none'`
    from its recorded config, and the mode would be broken by a value nobody typed.

    🚨 **NOT IMPLIED, and that is a decision rather than an omission:** `--no-hand-shaping`,
    `--terminal-indicator`, `--victory-value 1.0` and `--draw-penalty 0`. Those four are
    resume-immutable REWARD fields with concrete argparse defaults (True / False / 30.0 / −35.0),
    so "the operator left it alone" and "the operator typed the default" are indistinguishable —
    an implication there would silently overwrite a typed value, and the refusal meant to catch a
    conflicting one could never fire. They are instead REQUIRED, each by its own
    `combination_checks` entry naming the flag to pass. That is this tree's standing preference for
    a composition-changing combination (`--use-popart` requires an explicit `--clip-range-vf none`
    for the same reason): a self-documenting config beats a silent override, and the reward
    composition a run trained under is exactly the thing the v8→v9 drift proved must be stated.
    """
    from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob

    inherit_saved_flag(args, saved_ver, "critic", CRITIC_DEFAULT)
    if not is_winprob(args.critic):
        return
    for name, value in (("win_prob_mode", "shaping"), ("gamma", 1.0), ("use_popart", False)):
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def resolve_config(args, parser) -> ResolvedRunConfig:
    """Desugar, inherit and validate `args` in place. Returns the values that live outside it."""
    # WHICH FLAGS WERE ACTUALLY TYPED — captured FIRST, before one default is filled, because after
    # `_resolve` an unset tri-state flag is indistinguishable from a typed default and several
    # refusals ("you passed a knob that does nothing") are only honest about a typed value.
    # `main.checkargs` stamps the same marker before it inherits from the parent config, so
    # `combination_checks._typed` gives both surfaces the same answer. See that module's docstring.
    args._explicit_flags = frozenset(d for d, v in vars(args).items() if v is not None)

    # --- Resolve `--use-bridge` into the two internal fields ------------------------------------
    # ONE knob now: `--use-bridge {off,node,rust}`, defaulting to `rust` (serverless training AND
    # eval). It splits into `args.use_showdown_bridge` (a plain bool = "bridge enabled?", read at
    # every transport site) + `args.bridge_impl` (the "node"|"rust" child selector, read only at
    # spawn). `off` keeps a bridge_impl of "node" so a websocket run still has a well-formed value
    # for the offline/search paths that take one.
    #
    # The DEPRECATED `--use-showdown-bridge` boolean alias is DELETED. It meant `--use-bridge=node`,
    # which is no longer the default, so keeping it would have made "the legacy flag" silently mean
    # "the slower impl" — pass `--use-bridge=node` explicitly for that.
    _use_bridge = getattr(args, "use_bridge", "rust")
    args.bridge_impl = "node" if _use_bridge == "off" else _use_bridge
    args.use_showdown_bridge = _use_bridge != "off"

    # --- Resolve resumable structural toggles (None sentinel = "not passed on the CLI") ---
    # Each version-checked structural toggle defaults to None so a FLAGLESS resume can INHERIT the
    # saved value (the documented `--model … --steps …` command), instead of falling back to OFF and
    # FATALing at check_compatible (saved-ON vs current-default-OFF). An EXPLICIT flag that flips a
    # toggle still FATALs at load (desirable). A fresh run (no --model) → the toggle's OFF default.
    _saved_ver = _load_saved_version(args.model) if args.model else None
    if args.model and _saved_ver is None:
        print("[Resume] WARNING: saved model_config.json unreadable — structural toggles fall back to "
              "their OFF defaults and may FATAL at the version check; pass them explicitly if needed.")
    # Whether a recorded parent config was READ. `combination_checks` needs it for the one check
    # whose launch-path behaviour depends on it (an inherited PopArt has its clip auto-cleared just
    # below, so the refusal must not fire there); `main.checkargs` reports that check as ADVISORY
    # when it could not read the parent.
    args._saved_config_present = _saved_ver is not None
    _popart_explicit = args.use_popart is not None
    _coef_explicit = args.opp_belief_aux_coef is not None

    desugar_umbrella_flags(args)

    # --- gen3_winprob_critic_mode_v1: THE CRITIC MODE, and the composition it implies ------------
    # Resolved BEFORE `_resolve` so the implications below land on the same tri-state sentinels
    # every other flag is inherited through — an implied value must look exactly like a typed one
    # to `_resolve`, or a fork would inherit the parent's `shaped` composition under a `winprob`
    # argv. `--critic` itself is STRUCTURAL + resume-immutable, so it inherits like `win_prob_mode`.
    resolve_critic_mode(args, _saved_ver)

    def _resolve(name, default):
        inherit_saved_flag(args, _saved_ver, name, default)
    # gen3_winprob_critic_mode_v1: `--gamma` is now a FLAG. Its shaped-critic default is the
    # historical hardcoded 0.9999, read from `reward_weights.PBRS_GAMMA` rather than retyped —
    # PBRS is policy-invariant only when the two agree, so a second copy of the number is a second
    # place for them to disagree. (`reward_weights` is pure constants + one stall import, so this
    # costs `main.checkargs` no torch.) Under `--critic winprob` `resolve_critic_mode` already
    # implied 1.0 above, so this line does not fire there.
    from agents.training.reward_weights import PBRS_GAMMA as _PBRS_GAMMA_DEFAULT
    _resolve("gamma", _PBRS_GAMMA_DEFAULT)
    _resolve("use_popart", False)
    _resolve("opp_belief_cls_k", 0)
    _resolve("opp_belief_aux_coef", 0.0)
    _resolve("move_belief_mode", "off")        # v17 structural (version-checked, fresh-only)
    _resolve("move_belief_coef", 0.0)          # training-only (inherited like opp_belief_aux_coef)
    _resolve("damage_op", False)               # v19 structural (version-checked, fresh-only)
    _resolve("damage_outgoing", False)         # v23 structural (version-checked, fresh-only)
    _resolve("move_candidate_floor", _PRIOR_FLOOR)  # v65 forward-behavior (version-checked, fresh-only)
    _resolve("move_latent", False)             # v24 structural (version-checked, fresh-only)
    _resolve("move_belief_latent_coef", 0.0)   # training-only (inherited like move_belief_coef)
    _resolve("spread_belief", False)           # v25 structural (version-checked, fresh-only)
    _resolve("spread_belief_nature", False)    # v40 structural (version-checked, fresh-only)
    _resolve("spread_belief_coef", 0.0)        # training-only (inherited like move_belief_coef)
    _resolve("move_prior_fusion", False)       # v20 forward-behavior (version-checked, fresh-only)
    _resolve("damage_candidate_k", 0)          # v49 forward-behavior (version-checked, fresh-only)
    _resolve("entity_topk_seats", 0)           # v54 structural int (version-checked, fresh-only)
    _resolve("consequence_topk", 6)            # v59 forward-behavior int (version-checked)
    _resolve("edge_bias_families", "off")      # v56 structural str (version-checked, fresh-only)
    _resolve("entity_tail_seats", False)       # v57 structural bool (version-checked, fresh-only)
    _resolve("win_prob_mode", "none")          # v22 structural + resume-immutable (version-checked)
    _resolve("win_prob_coef", 1.0)             # training-only (inherited like opp_belief_aux_coef)
    _resolve("value_dist_mode", "none")        # v29 structural + resume-immutable (version-checked)
    _resolve("value_dist_bins", 0)             # v29 structural (atom count; version-checked)
    _resolve("value_dist_vmin", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_vmax", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_coef", 1.0)           # training-only (inherited like win_prob_coef)
    _resolve("td_aux_coef", 0.0)               # v90 training-only (inherited like win_prob_coef)
    _resolve("win_prob_pbrs_coef", 0.0)        # v104 training-only (inherited like td_aux_coef)
    # v105 training-only PATH, inherited WITH the coefficient above: a flagless resume that dropped
    # it would silently swap the FROZEN potential back to the live, drifting head — a change of
    # objective mid-run with nothing in any metric saying so.
    _resolve("win_prob_pbrs_source", None)
    _resolve("policy_grad_coef", 1.0)                 # v102 training-only (inherited like td_aux_coef; 1.0 = upstream)
    _resolve("value_threat_inject", False)     # v64 structural bool (version-checked, fresh-only)
    _resolve("opp_intent_coef", 0.0)           # v67 training-only coef; the HEADS are structural
    _resolve("beta_setvalued_coef", 0.0)       # training-only coef; no module, no version gate
    _resolve("intent_label_bot_weight", 1.0)   # v97 training-only (inherited like win_prob_coef)
    # (`opp_intent_grad_mode` had a `_resolve` here until 2026-08-23. It is config_only now —
    #  no argparse dest to inherit FROM, so a resolve line would be dead. Frozen "detached".)
    _resolve("intent_move_cell", False)        # v77 structural, version-checked (G3)
    _resolve("value_entity_pool", False)       # v80 structural, version-checked (Stage-3 T3)
    _resolve("history_events", False)          # v81 structural, version-checked (Tier H-B)
    _resolve("value_entity_pool_full", False)  # v82 structural, version-checked (full row set)
    _resolve("item_belief", False)             # v83 structural, version-checked (gen3_item_belief_v1)
    _resolve("pair_outcome_cell", False)   # v93 structural, version-checked (gen3_pair_outcome_v1)
    _resolve("pair_outcome_switch", False)  # v94 structural, version-checked (gen3_pair_outcome_switch_v1)
    _resolve("switch_branch_cell", False)   # v94 structural, version-checked (gen3_switch_branch_v1)
    _resolve("conditional_threat_cell", False)  # v95 structural, version-checked (gen3_conditional_threat_v1)
    _resolve("pair_value_route", False)     # v95 structural, version-checked (gen3_pair_value_route_v1)
    _resolve("intent_threshold", False)        # v84 structural, version-checked (gen3_intent_threshold_v1)
    _resolve("intent_conditional", False)      # v85 structural, version-checked (gen3_intent_conditional_v1)
    _resolve("op_drop_renders", False)         # v86 structural, version-checked (gen3_op_lean_forward_v1)
    _resolve("op_believed_lean", False)        # v86 structural, version-checked (gen3_op_lean_forward_v1)
    # THE COUNTERFACTUAL FAMILY — structural heads AND their coefficients, all inherited.
    # The three STRUCTURAL bools are version-checked; the ten coefficients below are the
    # td_aux_coef class (config v100, gen3_cf_coef_provenance_v1): recorded for provenance,
    # never gated, and read back here so a flagless resume keeps the arm it was launched as.
    # ⚠️ Every one of these argparse entries MUST default to None or the `_resolve` line is a
    # no-op — that is exactly how these three sat here reading False for two versions.
    # `flag_registry_test.test_cli_flags_argparse_default_is_none` is now that gate.
    _resolve("cf_evidential", False)           # v98 structural, version-checked (gen3_cf_evidential_head_v1)
    _resolve("cf_twin_heads", False)           # v99 structural, version-checked (gen3_cf_twin_heads_v1)
    _resolve("cf_shadow_critic", False)        # v99 structural, version-checked (gen3_cf_twin_heads_v1)
    _resolve("cf_records", False)              # v100 training-only (inherited like td_aux_coef)
    _resolve("cf_records_keep", 512)           # v100 training-only
    _resolve("cf_winprob_coef", 0.0)           # v100 training-only
    _resolve("cf_head_only", True)             # v100 training-only
    _resolve("cf_label_lag_steps", 150_000)    # v100 training-only
    _resolve("cf_label_likelihood", "binomial")  # v100 training-only
    _resolve("cf_evidential_coef", 0.0)        # v100 training-only
    _resolve("cf_evidential_reg", 1e-3)        # v100 training-only
    _resolve("cf_twin_coef", 0.0)              # v100 training-only
    _resolve("cf_shadow_coef", 0.0)            # v100 training-only
    # gen3_q_winprob_head_v1 (v107) — the per-action Q head. The MODE is structural and
    # version-checked; the two coefficients are the td_aux_coef class (recorded, never gated) and
    # are read back here so a flagless resume keeps the arm it was launched as.
    _resolve("q_winprob_mode", "none")         # v107 structural, version-checked
    _resolve("q_winprob_coef", 0.0)            # v107 training-only
    _resolve("q_winprob_onpolicy_coef", 0.0)   # v107 training-only
    # gen3_capacity_telemetry_v1 — the live saturation early-warnings. The td_aux_coef class:
    # recorded for provenance, never gated, and read back here so a flagless resume (or a
    # hand-typed one between launcher restarts) keeps logging the run's own `capacity/*` series.
    _resolve("capacity_telemetry", False)      # v101 training-only diagnostic (no loss, no grad)
    _resolve("canary_reset_steps", 1_000_000)  # v101 training-only
    _resolve("capacity_cosine_every", 50)      # v101 training-only
    _resolve("capacity_velocity_every", 50)    # v101 training-only
    _resolve("species_prior_fusion", False)    # v68 structural bool (version-checked, fresh-only)
    _resolve("t0_species_prior", False)        # v72 structural bool (version-checked, fresh-only)
    _resolve("search_teacher_coef", 0.0)       # training-only AWR weight (inherited on flagless resume)
    _resolve("search_teacher_value_coef", 0.0)  # training-only off-policy value term (default OFF)
    _resolve("search_teacher_beta", 1.0)       # training-only AWR temperature
    _resolve("search_teacher_batch_size", 256)  # training-only per-train() correction sample
    _resolve("opd_coef", 0.0)                  # training-only OPD KL weight (inherited on flagless resume)
    _resolve("distill_coef", 0.0)              # training-only exploiter-distillation KL weight (inherited on resume)
    _resolve("distill_value_coef", 0.0)        # training-only exploiter VALUE-distillation MSE weight (inherited on resume)
    _resolve("distill_value_feat_coef", 0.0)   # training-only FitNets value-FEATURE distill cosine weight (inherited on resume)
    # gen3_distill_offslice_anchor_v1 — the OFF-SLICE trust region to the frozen fold parent. The
    # distill_coef class: training-only, never gated, argparse default None so an unset flag lands
    # on the byte-identical 0.0 / "off_slice" here rather than in three separate places.
    _resolve("distill_anchor_coef", 0.0)       # training-only OFF-SLICE anchor KL weight (0.0 = off)
    _resolve("distill_anchor_mode", "off_slice")  # training-only: which rows the anchor applies to
    # WHICH policy the anchor is measured against. "parent" (the FIXED fold parent) is the default
    # and is byte-identical to what the anchor shipped with; "ema"/"periodic" are the moving-
    # reference arms. Same class as the two above: training-only, never gated, resolved here so an
    # unset flag lands on the byte-identical default in ONE place.
    _resolve("distill_anchor_ref", "parent")
    _resolve("distill_anchor_ema_tau", 0.99)   # training-only Polyak tau (~1/(1-tau) train() calls)
    _resolve("distill_anchor_refresh_every", 8)  # training-only periodic cadence (0 = never = parent)
    # gen3_distill_grad_project_v1 — m, the off-slice rows that constrain each step's DISTILL
    # gradient under `--distill-anchor-mode grad_project`. Same class as the three above:
    # training-only, never gated, argparse default None so an unset flag lands on the one default.
    _resolve("distill_anchor_proj_samples", 16)
    # gen3_distill_stop_rule_v1 — the DUAL-ASCENT budget on the anchor coefficient, and the FOLD
    # STOP RULE. Same class as everything above: training-only, never gated, argparse default None
    # so an unset flag lands on the byte-identical OFF default in one place and a flagless resume
    # keeps the arm it was launched as.
    _resolve("distill_anchor_target_kl", 0.0)   # training-only dual budget (0.0 = off)
    _resolve("distill_anchor_dual_lr", 0.1)     # training-only dual step eta
    _resolve("distill_anchor_coef_min", 0.0)    # training-only lower clamp (0 = no floor)
    # NOTE: `distill_anchor_coef_max` gets NO `_resolve` line, deliberately — its default is None,
    # which MEANS "10x the starting coefficient" and is computed inside `AnchorDualAscent` where the
    # starting coefficient is known. A `_resolve(..., None)` would be a line that does nothing and
    # reads as if it did something.
    _stop_default = _resolve_fold_instruments(args)
    _resolve("distill_stop", _stop_default)
    _resolve("distill_stop_window", 8)          # training-only detector look-back, in rollouts
    _resolve("distill_stop_eps", 0.005)         # training-only PLATEAU threshold (absolute)
    _resolve("distill_stop_kl_slope", 2.0)      # training-only RISE threshold, in slope-SEs
    _resolve("distill_stop_persist", 3)         # training-only AND-gate persistence count
    _resolve("distill_stop_anneal_factor", 0.7)  # training-only per-rollout decay of --distill-coef
    # gen3_distill_target_gate_v1 (config v103) — the action-form/top-K distill target, the
    # advantage gate, and the rank tripwire. The td_aux_coef class: recorded for provenance,
    # never gated, read back here so a flagless resume keeps the arm it was launched as.
    _resolve("distill_target", "kl")           # v103 training-only TARGET FORM ("kl" = byte-identical)
    _resolve("distill_topk", 1)                # v103 training-only top-K (1 = argmax CE)
    _resolve("distill_gate", "none")           # v103 training-only JUDGE (rung a)
    _resolve("distill_gate_tau", 0.0)          # v103 training-only gate threshold (normalized-adv units)
    _resolve("distill_beta", 1.0)              # v103 training-only AWR |adv| temperature
    _resolve("rank_tripwire", "warn")          # v103 training-only diagnostic (§4.1; no loss, no grad)
    _resolve("rank_tripwire_drop", 0.20)       # v103 training-only TRIP threshold (fractional drop)
    _resolve("opd_beta", 1.0)                  # training-only OPD softmax temperature β
    _resolve("damage_topk_k", 0)               # v30 structural int (top-K incoming; version-checked, fresh-only)
    _resolve("damage_matrices_outgoing", False)  # v32 structural (outgoing damage matrix; version-checked, fresh-only)
    _resolve("damage_matrices_incoming", False)  # v33 structural (incoming damage matrix; version-checked, fresh-only)
    # gen3_op_block_trim_v1: --damage-topk K now sizes the INCOMING MATRIX and nothing else — the v30 LEAN
    # top-K block it used to select is DELETED (a strict subset of the matrix, which already suppressed it
    # in every production config; the ledger-P1 cProfile measured it at 0 calls/forward). So K>0 implies the
    # matrix. When the user gave no explicit --damage-matrices (the --unified-moves path, which auto-sets
    # K=5) turn the incoming matrix ON rather than let K>0 mean "emit nothing"; an EXPLICIT
    # --damage-matrices off/outgoing next to K>0 is a contradiction and errors below.
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.damage_matrices_incoming:
        if getattr(args, "damage_matrices", None) is None:
            args.damage_matrices_incoming = True
            print("[Arch] --damage-topk implies the INCOMING per-move damage matrix (gen3_op_block_trim_v1: "
                  f"the lean top-K block was deleted) — enabling it at K={args.damage_topk_k}.")
    _resolve("belief_grad_mode", "shaping")    # v41 resume-immutable training hparam (vf_coef class; flagless resume inherits)
    _resolve("value_from_dist", False)         # v45 Phase B: dist head is the critic (resume-immutable; flagless resume inherits)
    _resolve("hp_belief_mode", "composed")     # v53 STRUCTURAL (version-checked, fresh-only)
    _resolve("hp_type_belief_coef", 0.05)      # training-only (inherited like spread_belief_coef)
    _resolve("item_belief_coef", 0.05)         # training-only (inherited like hp_type_belief_coef)
    # Phase B (v45): the dist head can only BE the critic if it's a live, trunk-shaping head.
    # PopArt INHERITED on a flagless resume → adopt its required `--clip-range-vf none` (the saved
    # popart run necessarily used it), so the explicit-config check below doesn't block the resume.
    if args.use_popart and not _popart_explicit and _saved_ver is not None and args.clip_range_vf is not None:
        args.clip_range_vf = None
    # Friendly belief-resume notes (inheriting vs an explicit flip).
    if args.model and _saved_ver is not None:
        _sc = getattr(_saved_ver, "opp_belief_aux_coef", 0.0) or 0.0
        if not _coef_explicit and _sc > 0.0:
            print(f"[Belief] resume: inheriting saved --opp-belief-aux-coef {_sc:g} (pass it explicitly to override).")
        elif _coef_explicit and (_sc > 0.0) != (args.opp_belief_aux_coef > 0.0):
            print(f"[Belief] WARNING: --opp-belief-aux-coef {args.opp_belief_aux_coef:g} flips the belief head "
                  f"vs the saved checkpoint (coef {_sc:g}); a weight-shape change → will FATAL on load.")

    if not 0.0 <= args.stable_opponent_selfplay_share <= 1.0:
        parser.error("--stable-opponent-selfplay-share must be a fraction in [0, 1]")
    if not 0.0 <= args.exploiter_bot_fraction <= 1.0:
        parser.error("--exploiter-bot-fraction must be a fraction in [0, 1]")
    # gen3_fork_lr_pin_v1 — `--fork-lr` is RESUME-ONLY. On a fresh run the optimizer starts at
    # `--lr` and nothing overrides it, so a pin there is either a no-op or a second spelling of
    # `--lr`, and the second reading is the dangerous one: a fresh run pinned to a value its own
    # `--lr` contradicts records a `dose` block naming a rate it never used. Refuse and say which
    # flag to use. `--fork-lr-freeze` alone is likewise refused — a freeze with nothing to freeze at.
    if getattr(args, "fork_lr", None) is not None and args.fork_lr <= 0:
        parser.error("--fork-lr must be > 0 (it is a learning rate, not a switch).")

    if args.opp_belief_cls_k < 0:
        parser.error("--opp-belief-cls-k must be >= 0 (0 = off)")
    if args.opp_belief_aux_coef < 0.0:
        parser.error("--opp-belief-aux-coef must be >= 0 (0 = off)")
    if args.move_belief_coef is not None and args.move_belief_coef < 0.0:
        parser.error("--move-belief-coef must be >= 0 (0 = off)")
    if args.win_prob_coef is not None and args.win_prob_coef < 0.0:
        # A negative coef would INVERT the BCE gradient (train the head/trunk to MAXIMISE error).
        # win_prob_coef is training-only (not version-locked), so guard it here — the only gate.
        parser.error("--win-prob-coef must be >= 0 (0 = off; the mode controls on/off)")
    if args.value_dist_coef is not None and args.value_dist_coef < 0.0:
        # A negative coef would INVERT the CE gradient. value_dist_coef is training-only (not
        # version-locked), so guard it here — the only gate.
        parser.error("--value-dist-coef must be >= 0 (0 = off; the mode controls on/off)")
    if args.td_aux_coef is not None and args.td_aux_coef < 0.0:
        # A negative coef would INVERT the consistency gradient (train the critic to MAXIMISE its own
        # Bellman residual). td_aux_coef is training-only (not version-locked), so guard it here.
        parser.error("--td-aux-coef must be >= 0 (0 = off)")
    if args.win_prob_pbrs_coef is not None and args.win_prob_pbrs_coef < 0.0:
        # A negative coef INVERTS the potential — the policy would be rewarded for driving its own
        # P(win) DOWN. The invariance theorem still holds (φ' = −φ is a valid potential), which is
        # exactly why this cannot be caught downstream: it would train, converge, and be wrong.
        parser.error("--win-prob-pbrs-coef must be >= 0 (0 = off)")
    # --- gen3_clean_world_config_v1: the TERMINAL magnitude + the outcome ORDERING it implies ---
    if getattr(args, "victory_value", 30.0) is not None and args.victory_value <= 0.0:
        # A non-positive victory value inverts win/loss (or flattens them), which trains correctly
        # toward the wrong objective and no metric names it.
        parser.error("--victory-value must be > 0 (a win scores +V, a loss -V; 30.0 = the default, "
                     "1.0 = the clean-world ±1 terminal)")
    if (getattr(args, "victory_value", None) is not None
            and not bool(getattr(args, "terminal_indicator", False))   # see _terminal_scale_guards
            and args.draw_penalty is not None and args.draw_penalty > -float(args.victory_value)):
        # NOT an error — "a draw is better than a loss" is a legitimate thing to want, and a fresh
        # arm may deliberately choose it. It IS the single largest hazard in the clean-world arm,
        # so it is stated once, loudly, at launch (probe N §5.2/B3).
        print(f"[Reward] ⚠️ ORDERING: --draw-penalty {args.draw_penalty:g} is BETTER than a clean "
              f"loss (-{float(args.victory_value):g}), so running the 250-turn clock out is the best "
              f"non-winning outcome and a losing agent's optimal play is to stall. The validated "
              f"composition keeps draw_penalty <= -victory_value. If this is deliberate, make "
              f"stall rate + mean game length a PRIMARY endpoint.")
    _terminal_scale_guards(args)
    if args.policy_grad_coef is not None and args.policy_grad_coef < 0.0:
        # A negative coef would ASCEND the PPO surrogate — train the policy to be maximally wrong.
        # 0.0 (arm F's pure-distill/aux phase) is the intended floor. policy_grad_coef is training-only
        # (not version-locked), so guard it here — the only gate.
        parser.error("--policy-grad-coef must be >= 0 (1 = upstream PPO; 0 = no policy-gradient term)")
    _adaptive_batch_guards(args, parser)
    if args.intent_label_bot_weight is not None and args.intent_label_bot_weight < 0.0:
        # A negative weight would train alpha/beta to be MAXIMALLY wrong about bots — the opposite
        # of "train on them less". 0.0 (ignore bot rows entirely) is the intended floor.
        # Training-only (not version-locked), so this parser check is the only gate.
        parser.error("--intent-label-bot-weight must be >= 0 (0 = train on no bot rows; 1 = off)")
    if args.opd_coef is not None and args.opd_coef < 0.0:
        parser.error("--opd-coef must be >= 0 (0 = off)")
    # gen3_winprob_oneply_teacher_v1 (ai_v12 routes 2+3). The mode selects WHICH teacher fills the
    # correction buffer; `crater` is the default and needs no gate. Two ways `winprob_oneply` can be
    # asked for and be unable to run, both silent otherwise (the callback is simply never built, or
    # every candidate is skipped for want of a head):
    if not (0.0 < args.winprob_teacher_band <= 0.5):
        # 0 admits nothing; > 0.5 admits every decision and the gate stops being a gate.
        parser.error("--winprob-teacher-band must be in (0, 0.5] (the |P(win) - 0.5| half-width)")
    if not (0.0 <= args.winprob_teacher_margin < 1.0):
        parser.error("--winprob-teacher-margin must be in [0, 1) — it is a win-PROBABILITY gap")
    # gen3_cf_label_plumbing_v1 — training-only, so these parser checks are the ONLY gate.
    if args.cf_winprob_coef is not None and args.cf_winprob_coef < 0.0:
        parser.error("--cf-winprob-coef must be >= 0 (0 = off)")
    if args.cf_label_lag_steps is not None and args.cf_label_lag_steps < 0:
        parser.error("--cf-label-lag-steps must be >= 0 (0 = never expire)")
    if args.cf_records_keep is not None and args.cf_records_keep < 1:
        parser.error("--cf-records-keep must be >= 1")
    # gen3_capacity_telemetry_v1 — training-only diagnostics, so these parser checks are the ONLY
    # gate. A reset interval of 0 would re-seed a target on EVERY minibatch, which is not a slower
    # canary but a different (and meaningless) instrument, so it is refused rather than clamped.
    if args.canary_reset_steps is not None and args.canary_reset_steps < 1:
        parser.error("--canary-reset-steps must be >= 1 (it is the ENV-step interval between "
                     "plasticity-canary resets; there is no 'off' value — use "
                     "--no-capacity-telemetry to turn the whole instrument off)")
    if args.capacity_cosine_every is not None and args.capacity_cosine_every < 0:
        parser.error("--capacity-cosine-every must be >= 0 (0 = skip the half-batch cosine)")
    if args.capacity_velocity_every is not None and args.capacity_velocity_every < 0:
        parser.error("--capacity-velocity-every must be >= 0 (0 = skip the feature-velocity probe)")
    # gen3_cf_evidential_head_v1 — the coefficients are training-only, so (as above) these parser
    # checks are the ONLY gate. `--cf-evidential` itself IS version-gated, hence not checked here.
    if args.cf_evidential_coef is not None and args.cf_evidential_coef < 0.0:
        parser.error("--cf-evidential-coef must be >= 0 (0 = off)")
    if args.cf_evidential_reg is not None and args.cf_evidential_reg < 0.0:
        parser.error("--cf-evidential-reg must be >= 0 (0 = no KL pull toward Beta(1,1))")
    # gen3_cf_twin_heads_v1 — the coefficients are training-only (parser checks are the ONLY gate);
    # the two structural flags are version-gated, so only their CROSS-flag requirements land here.
    if args.cf_twin_coef is not None and args.cf_twin_coef < 0.0:
        parser.error("--cf-twin-coef must be >= 0 (0 = off)")
    if args.cf_shadow_coef is not None and args.cf_shadow_coef < 0.0:
        parser.error("--cf-shadow-coef must be >= 0 (0 = off)")
    # gen3_q_winprob_head_v1 (v107) — the two coefficients are training-only, so these parser
    # checks are their ONLY gate. `--q-winprob-mode` itself is version-gated, so only its
    # cross-flag requirements land here.
    if args.q_winprob_coef is not None and args.q_winprob_coef < 0.0:
        parser.error("--q-winprob-coef must be >= 0 (0 = off)")
    if args.q_winprob_onpolicy_coef is not None and args.q_winprob_onpolicy_coef < 0.0:
        parser.error("--q-winprob-onpolicy-coef must be >= 0 (0 = off)")
    if getattr(args, "checkpoint_every_steps", None) is not None and args.checkpoint_every_steps < 1:
        parser.error("--checkpoint-every-steps must be >= 1 (it is an ENV-STEP interval; there is "
                     "no 'off' value — omit the flag for the historical 50000-vec-call cadence)")
    _announce_cf_duty_cycle(args)
    if args.distill_coef is not None and args.distill_coef < 0.0:
        parser.error("--distill-coef must be >= 0 (0 = off)")
    if args.distill_value_coef is not None and args.distill_value_coef < 0.0:
        parser.error("--distill-value-coef must be >= 0 (0 = off)")
    if args.distill_value_feat_coef is not None and args.distill_value_feat_coef < 0.0:
        parser.error("--distill-value-feat-coef must be >= 0 (0 = off)")
    # gen3_distill_offslice_anchor_v1 — the OFF-SLICE trust region. The dependency is not a style
    # rule: the anchor's slice IS the `distill_mask` obs key, and the env emits that key only when
    # `_distill_species` is populated, which `apply_distill_team_bias` gates on --distill-coef > 0.
    # Without a live distill there is no slice, so an anchor would either anchor everything or
    # nothing — either way not the thing the flag names.
    if args.distill_anchor_coef is not None and args.distill_anchor_coef < 0.0:
        parser.error("--distill-anchor-coef must be >= 0 (0 = off; with --distill-anchor-monitor, "
                     "0 is the pure-instrument arm)")
    if args.distill_anchor_proj_samples is not None and args.distill_anchor_proj_samples < 1:
        parser.error("--distill-anchor-proj-samples must be >= 1 — it is the number of off-slice "
                     "rows that constrain each step's distill gradient, and 0 constraints is "
                     "--distill-anchor-mode off_slice with no projection at all.")
    # The two moving-reference knobs. tau is a convex-combination weight, so anything outside [0, 1]
    # is not an average at all — it EXTRAPOLATES away from the student (tau > 1) or overshoots past
    # it (tau < 0), and either would still train and still read as ON.
    if args.distill_anchor_ema_tau is not None and not (0.0 <= args.distill_anchor_ema_tau <= 1.0):
        parser.error("--distill-anchor-ema-tau must be in [0, 1] — it is the Polyak weight in "
                     "ref <- tau*ref + (1-tau)*student. 1.0 IS --distill-anchor-ref parent (the "
                     "reference never moves); 0.0 makes the reference the current student, so the "
                     "anchor loss goes to ~0.")
    if args.distill_anchor_refresh_every is not None and args.distill_anchor_refresh_every < 0:
        parser.error("--distill-anchor-refresh-every must be >= 0 (0 = never refreshed = "
                     "--distill-anchor-ref parent).")
    # ---- gen3_distill_stop_rule_v1 refusals ----------------------------------------------------
    # The DUAL is multiplicative, so a zero starting coefficient is a FIXED POINT: the controller
    # would run every rollout and move nothing, while every startup line and every series said it
    # was on. Refuse rather than ship a silent no-op — the same principle as the anchor's
    # unresolvable-parent FATAL.
    # The STOP RULE's AND-gate reads `distill/collateral_kl_vs_parent`, which exists only when the
    # frozen fold parent is attached. Without it the rise half is permanently silent and the rule
    # would never fire, while reading as ON — the exact silent-no-op class the anchor's loud
    # startup line was written against.
    # gen3_exploiter_distill_v1: parse --distill-teacher into (teacher_path, [team_files]) GROUPS once,
    # stored on args for the teambuilder + model-setup to reuse. Preferred form =
    # 'TEACHER:TEAM[,TEAM...][;TEACHER2:...]' — ';' separates TEACHERS, ',' separates that teacher's TEAMS,
    # so ONE multi-team teacher (a --trainee-teams z-cluster exploiter) binds to all its teams without being
    # repeated N times (which would cost N identical teacher forwards per batch). The legacy comma-separated
    # pair form ('T1:a.txt,T2:b.txt') still parses (a comma segment containing ':' starts a new teacher).
    #
    # `--distill-team-bias` carries a None argparse default so a TYPED value is distinguishable from
    # the unset flag; the guard below refuses a typed bias with no teacher, and could not exist if
    # every flagless run arrived carrying 0.4. Resolved here, before any reader.
    if args.distill_team_bias is None:
        args.distill_team_bias = DEFAULT_DISTILL_TEAM_BIAS
    #
    # THE COEFFICIENT GATES THE LOSS, NOT THE BOOKKEEPING (gen3_distill_bias_at_coef0_v1). The pairs
    # are parsed whenever --distill-teacher is given, at ANY coefficient — because `--distill-team-bias`
    # (the trainee's team distribution) reads them, and a CONTROL arm is precisely "the same teacher
    # teams, the same bias, no loss". Gating the parse on the coefficient made that arm silently
    # UNBIASED: run `ai_v9_58_R2CTRL_0827` recorded `--distill-teacher <5> --distill-coef 0
    # --distill-team-bias 0.4` and trained at an EFFECTIVE bias of 0.0, so the capstone's
    # "team-bias constancy" design was violated by the config layer, invisibly, in both metadata and
    # argv. What DOES stay coefficient-gated is everything that costs something or changes a tensor:
    # the teacher model LOADING (main.train.model_build), the loss fold (instrumented_ppo), and the
    # training-only `distill_mask` obs key (main.train.matchup_setup — emitting it at coef 0 would
    # change the observation SPACE of a run that folds no distill term).
    args._distill_pairs = []
    _items = [x.strip() for x in (args.distill_teacher or "").split(",") if x.strip()]
    if _items:
        if ":" not in _items[0]:
            # The bare-list + parallel --distill-teacher-team form is DELETED (no run ever passed it;
            # verified across every models/*/metadata.json 2026-08-16). One form, no misalignment.
            #
            # The check is on the FIRST segment, which is the one that distinguishes the two forms.
            # It used to be `all(":" in x for x in _items)`, which also rejected the DOCUMENTED
            # multi-team group 'T1:a.txt,b.txt' (a teacher's 2nd and later teams are comma segments
            # with no colon by construction) — unless another ';'-joined teacher happened to follow,
            # which is the only reason the multi-team form was ever seen to work. A later bare
            # segment with no preceding teacher is still refused, by `parse_distill_teacher_spec`.
            parser.error("--distill-teacher takes 'TEACHER:TEAM[,TEAM...]' colon groups — the bare "
                         "teacher list (with the deleted --distill-teacher-team) is no longer accepted")
        from agents.training.distill_spec import check_teacher_spec, parse_distill_teacher_spec
        from agents.training.matchup_spec import read_recorded_trainee_teams

        def _resolve_teacher_teams(_run_dir):
            """The ``'TEACHER:*'`` resolver. ``require_teams`` because THIS caller's whole request
            is "the teams that run trained on" — a run that recorded none is a refusal here, not
            the empty list a generalist legitimately reads as elsewhere. The `@step` suffix is
            already off (`distill_spec` splits it); a path that does not exist RAISES."""
            return read_recorded_trainee_teams(_run_dir, require_teams=True)

        try:
            # 'TEACHER:*' → EXACTLY the teams that teacher trained on, from its own recorded
            # provenance (single source of truth — a hand-typed list could mismatch and fire the
            # distill mask where the teacher is off-distribution, silently).
            args._distill_pairs = parse_distill_teacher_spec(
                args.distill_teacher, resolve_wildcard=_resolve_teacher_teams)
        except (ValueError, FileNotFoundError) as _e:
            parser.error(str(_e))
        # gen3_run_spec_split_v1 — THE TEACHER-ASSEMBLY GUARD. A teacher that resolves to ZERO
        # teams folds no loss and biases no team draw while every log line still reads as a running
        # fold; the only witness was the team count in the `🧪 [DISTILL]` startup banner. The rule
        # lives in ONE place (`distill_spec.check_teacher_spec`) and `main.checkargs` reads the same
        # function offline, so the two cannot drift.
        #
        # `check_paths=False`: the PATH questions already have loud answers downstream on a real
        # launch (`model_build` exits FATAL_CONFIG naming a teacher it cannot load;
        # `apply_distill_team_bias` raises on a team file it cannot open), and re-asking them here
        # would newly refuse a coef-0 CONTROL arm whose teacher run has since been archived.
        # `main.checkargs` passes True, because offline there is no downstream to answer.
        for _finding in check_teacher_spec(args.distill_teacher,
                                           resolve_wildcard=_resolve_teacher_teams,
                                           check_paths=False):
            parser.error(_finding)
    if args.distill_topk < 1:
        parser.error("--distill-topk must be >= 1 (1 = argmax CE; K >= n_actions recovers the KL)")
    # gen3_distill_target_gate_v1 (design §7.5): the action-form family's dependency graph.
    # Checked on the RESOLVED values (after `_resolve`), so an incoherent combination is refused
    # whether it was typed on this launch or inherited from the checkpoint's recorded config.
    #
    # The four rules themselves live in `main.train.combination_checks`, which `main.checkargs`
    # reads too — that is the whole point of the module. C1 (2026-09-01) forked a parent recording
    # `distill_target="action"`, passed `--distill-coef 0`, named no target, and died HERE while
    # checkargs had said the command still launches. One declaration, both readers.
    if args.distill_beta <= 0.0:
        parser.error("--distill-beta must be > 0 (an AWR temperature)")
    if not (0.0 < args.rank_tripwire_drop < 1.0):
        parser.error("--rank-tripwire-drop must be in (0, 1) — a fractional drop from baseline")
    if args.damage_candidate_k and args.damage_candidate_k < 0:
        parser.error("--damage-candidate-k must be >= 0 (0 = the full candidate sweep).")
    _ebf = args.edge_bias_families
    if _ebf and _ebf != "off":
        # The family vocabulary is the EXTRACTOR'S, single-sourced — a hand-copied set here
        # silently rejected the v79 `h` family at launch (caught by the flag-on bridge smoke:
        # the extractor knew `h`, the CLI did not, so a `,h` launch died in argparse).
        from agents.model.features_extractor import _EDGE_FAMILIES as _valid
        _fams = {"d1", "d3"} if _ebf == "d" else set(_ebf.split(","))
        if _fams - set(_valid):
            parser.error(f"--edge-bias-families: unknown families {sorted(_fams - set(_valid))} "
                         f"(valid: off, d [= d1,d3 frozen], or a comma list of {sorted(_valid)})")
    if not (_MIN_PRIOR_FLOOR <= args.move_candidate_floor < 1.0):
        # gen3_unconditional_move_legality_v1: the floor is the LEGAL-BUT-UNOBSERVED base, and a value at
        # or below the "impossible" probability collapses the legality distinction it exists to preserve.
        # 0.0 in particular is what a pre-v65 resume carries — it used to mean "legality OFF".
        parser.error(
            f"--move-candidate-floor {args.move_candidate_floor} is out of range: it is the "
            f"LEGAL-BUT-UNOBSERVED base of the move prior and must satisfy "
            f"{_MIN_PRIOR_FLOOR} <= value < 1.0 (default {_PRIOR_FLOOR}).\n"
            "Move legality is unconditional and has no off switch; 0.0 is no longer meaningful. "
            "If this came from resuming a pre-v65 checkpoint, that model's belief is incompatible — "
            "start a fresh run."
        )
    # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed).

    # THE ONE COMBINATION SWEEP. Every value-conditional refusal — the cross-flag dependency
    # graph, the mode-scoped ranges, the two --anneal-lr exits and the CF duty-cycle floor —
    # is DECLARED in `main.train.combination_checks` and evaluated here, once, on the resolved
    # namespace. `main.checkargs` reads the same list on the effective (argv + inherited)
    # namespace, so "checkargs says it launches" now means it launches, not merely that it
    # parses. G5 (2026-09-06) died three times on rules that had never been migrated;
    # `combination_checks_test.py` AST-scans this file and fails if a cross-flag
    # `parser.error` reappears outside the list.
    refuse_first(args, parser)

    # One server config, built from --showdown-port and threaded to every Showdown client
    # (training-env players in spawn workers, eval, and self-play). Default port: 8000.
    server_config = (
        LocalhostServerConfiguration
        if args.showdown_port is None
        else localhost_server_configuration(args.showdown_port)
    )
    if args.use_showdown_bridge:
        emit(f"🌉 Transport: in-process BattleStream bridge [{args.bridge_impl}] for BOTH training "
             "and eval (no Showdown server needed — --showdown-port ignored)")
        if args.bridge_impl == "rust":
            # One-time startup warning naming the Rust bridge's honest remaining scope limits (the
            # offline search/replay drivers are still Node-only; an INCOMPLETE modeled move set that
            # fail-louds) — resolve/build the binary NOW so a missing toolchain fails loudly at
            # startup, not deep inside the first env reset.
            from utils.bridge.sim_bridge_bin import (
                warn_rust_deferrals, resolve_and_publish_sim_bridge_bin)
            warn_rust_deferrals(emit)
            # Build ONCE here and PUBLISH the path (POKESIM_SIM_BRIDGE_BIN) so every
            # SubprocVecEnv env worker / eval-worker subprocess inherits a ready binary
            # instead of racing its own `cargo build` on first spawn.
            _rust_bin = resolve_and_publish_sim_bridge_bin()
            emit(f"🦀 [BRIDGE=rust] sim_bridge binary (prebuilt, published to children): {_rust_bin}")
            # The search-TEACHER used to be hard-blocked here. That guard is GONE
            # (`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1`): the Rust
            # `search_driver` binary now serves BOTH offline verb families, and
            # `SearchTeacherCallback(impl=args.bridge_impl)` threads this run's engine into the
            # worker subprocesses, so a rust run's teacher no longer silently falls back to node.
            #
            # For the record, since it cost someone an investigation: the guard's ORIGINAL reason —
            # that the search-teacher needs the sim's own byte-identical `input_log` — was simply
            # FALSE. Nothing reads the record's committed-choice lines. The only readers are
            # `replay_kernels.js::writeStart` and `ReconstructionRecord.start_options()` /
            # `.players()`, all of which touch only the `>start` / `>player` lines, which the rust
            # record renders exactly. The real blocker was always the missing DRIVER, and that is
            # what got built.
            if getattr(args, "search_teacher", False) or getattr(args, "teacher_persistent", False):
                emit("🦀 [BRIDGE=rust] search-teacher on the RUST offline drivers "
                     "(search_driver binary serves open_root/expand_many + replay/reroll/"
                     "reroll_many). Gated by: better_line node≡rust candidate values bit-identical, "
                     "search_clone_parity (clone ≡ reroll_many at the obs), and the counterfactual "
                     "confirm leg — each run on rust. NOT yet gated: a full multi-cycle teacher run "
                     "end-to-end on rust. Fall back with --use-bridge=node if a cycle misbehaves.")
    else:
        emit(f"🔌 Showdown server: {server_config.websocket_url}")

    annealing_mode = args.anneal_lr_start_steps is not None

    if args.hp_type_belief_coef and args.move_belief_mode == "off":
        # The CE supervises the HPTypeBelief head's posterior (last_hp_type_logits), and the head is built
        # only alongside a move belief (it composes P(HP present) from the move posterior's 237 channel).
        # EXPLICIT coef + no belief = a real contradiction → error. But the coef DEFAULTS to 0.05
        # (_resolve), so on the DEPRECATED `--unified-moves off` ablation baseline the un-passed default
        # would make the flag fail out of the box — the same shape as the `--hp-belief-mode flat` case
        # below, resolved the same way: AUTO-ZERO with a loud note.
        print("[HPBelief] no move belief (--unified-moves off): auto-zeroing the default "
              "--hp-type-belief-coef (the HP-type head is built only alongside a move belief).")
        args.hp_type_belief_coef = 0.0
    if args.hp_type_belief_coef and args.hp_belief_mode == "flat":
        # The `flat` ablation builds NO HPTypeBelief head, so there is no posterior for the CE to
        # supervise. AUTO-ZERO with a loud note rather than erroring:
        # --hp-type-belief-coef defaults to 0.05, so erroring would make
        # `--hp-belief-mode flat` fail out of the box — a hostile flag to run an ablation with. The
        # note keeps it from being a SILENT no-op, which is the failure that actually matters here.
        print("[HPBelief] --hp-belief-mode flat: auto-zeroing --hp-type-belief-coef (the ablation "
              "builds no HP-type head, so there is no posterior for the CE to supervise). The 16 "
              "typed HP channels are still predicted + supervised by the move-belief BCE.")
        args.hp_type_belief_coef = 0.0
    # The preload IS the opponent compile, moved into the forkserver — so it FOLLOWS
    # --compile-opponents by default (both ship ON). Only an EXPLICIT --compile-opponents-preload
    # alongside --no-compile-opponents is a contradiction worth erroring on; the same pairing
    # arrived at by defaults must silently resolve to "no compile at all", or --no-compile-opponents
    # (the documented fallback) would itself become a usage error.
    try:
        args.compile_opponents_preload = resolve_compile_opponents_preload(
            args.compile_opponents_preload, args.compile_opponents)
    except ValueError as exc:
        parser.error(str(exc))
    if args.item_belief_coef and not args.item_belief:
        # The CE supervises the ItemBelief head's posterior (last_item_logits) and the head exists
        # only under --item-belief; a coef with no head would be a silent no-op (the bank's row
        # sees no stash and returns None), so make the config honest instead of quietly inert.
        print("[ItemBelief] --item-belief off: auto-zeroing --item-belief-coef (the head the CE "
              "supervises is not built; pass --item-belief to enable it).")
        args.item_belief_coef = 0.0
    log_level = LogLevel[args.log_level.upper()]

    # Automatically enable deep traces if --debug is set
    if args.debug:
        log_level = LogLevel.DEBUG
        # Default a smoke run to CPU so it never contends with a live GPU training run.
        # Only the "auto" default is overridden — an explicit --device cpu|cuda still wins.
        # Set before any args.device consumer (pool/opponent/model build are all downstream).
        if args.device == "auto":
            args.device = "cpu"
        # A --debug smoke run is a short-lived child of the launching shell/agent and
        # uses DummyVecEnv (no SubprocVecEnv worker watchdog). If its parent dies it gets
        # orphaned, and a hung smoke (e.g. a vanished 9XXX server) then lingers for days as
        # a zombie. Exit if reparented. Started here — before team/env/server setup — so a
        # hang anywhere in startup is covered too. Real (launcher-managed) runs keep a live
        # parent and are unaffected.
        start_orphan_watchdog(label="debug-smoke")

    # --compile-trainer's AUTO default, resolved AFTER --debug has had its say on the device (a
    # --debug run with no --device is cpu, and must stay a pure-CPU minute-long smoke). An explicit
    # --compile-trainer / --no-compile-trainer always wins; the auto value only fills the None.
    #
    # ⚠️ THE SECOND HALF IS NOT OPTIONAL. `check_shape_stability` REFUSES two configs outright
    # (--async-rollout, and a rollout that does not divide by --batch-size), so a device-only
    # default would convert two classes of command that work today into a FATAL_CONFIG exit —
    # the same failure the cpu conditioning above exists to avoid, one flag over. A DEFAULT yields
    # to the config the user actually typed and says why; an EXPLICIT --compile-trainer still hits
    # the refusal, loudly, because there the user asked for something impossible.
    if args.compile_trainer is None:
        args.compile_trainer, _ct_why = resolve_compile_trainer_auto(
            device=args.device, debug=args.debug, n_steps=args.n_steps, n_envs=args.n_envs,
            batch_size=args.batch_size, async_rollout=bool(getattr(args, "async_rollout", False)))
        if _ct_why:
            emit("⚡ --compile-trainer would be ON by default here, but this config cannot take "
                 f"it — leaving it OFF rather than refusing to launch. Reason: {_ct_why} "
                 "(pass --compile-trainer explicitly to make this a hard error instead.)")
        if args.compile_trainer:
            # LOUD, and at STARTUP rather than only after the compile succeeds: with the default
            # ON, every plain cuda run now trades away the ObservationDebugger, and a trade nobody
            # typed a flag for is exactly the kind that has to announce itself.
            emit("⚡ --compile-trainer ON by default (device=cuda) — ~1.75x on the PPO train step. "
                 "⚠️ this DROPS the ObservationDebugger (dynamo cannot trace its numpy asserts); "
                 "pass --no-compile-trainer to keep it. Compile failure is FATAL by design.")

    return ResolvedRunConfig(server_config=server_config, annealing_mode=annealing_mode,
                             log_level=log_level)
