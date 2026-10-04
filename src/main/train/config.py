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
import sys

from agents.model.damage_tables import _MIN_PRIOR_FLOOR, _PRIOR_FLOOR
from agents.training.watchdog import start_orphan_watchdog
from main.launcher.ipc import emit
from main.train import arch_surface
from main.train.checkpoint_state import _load_saved_version
from main.train.combination_checks import refuse_first
from main.train.compile_flags import resolve_compile_trainer_auto
from utils.logging.levels import LogLevel


@dataclasses.dataclass(frozen=True)
class ResolvedRunConfig:
    """The two resolved values that are NOT attributes of `args`."""

    annealing_mode: bool
    log_level: LogLevel


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


def enforce_not_shaped_parent(model_path: str) -> None:
    """The LAUNCH-path wrapper over `model_version.shaped_reward.check_not_shaped` AND
    `model_version.retired_levers.check_no_retired_levers`: print the typed refusal and exit
    `FATAL_CONFIG` (restarting would hit the identical checkpoint every time, so the launcher must
    give up rather than loop). `main.checkargs` reads the same predicates."""
    from agents.model.model_version.retired_levers import (
        RetiredLeverCheckpointError, check_no_retired_levers)
    from agents.model.model_version.shaped_reward import (
        ShapedRewardCheckpointError, check_not_shaped, saved_config_path)
    from main.exit_codes import TrainExitCode
    try:
        check_not_shaped(saved_config_path(model_path))
        check_no_retired_levers(saved_config_path(model_path))
    except (ShapedRewardCheckpointError, RetiredLeverCheckpointError) as e:
        print(f"\n[ModelVersion] FATAL: {e}", flush=True)
        sys.exit(int(TrainExitCode.FATAL_CONFIG))


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


def _recorded_cli_arg(model_path, dest):
    """``dest`` from the RESOLVED namespace the checkpoint's run recorded (``metadata.json:cli_args``),
    or None. JSON only — no ``.zip`` is opened."""
    import json
    import os

    from agents.training.lineage import run_dir_of

    try:
        run_dir = run_dir_of(model_path) if model_path else None
        with open(os.path.join(run_dir, "metadata.json")) as f:
            cli = json.load(f).get("cli_args")
    except (OSError, TypeError, ValueError):
        return None
    return cli.get(dest) if isinstance(cli, dict) else None


class UnrecordedEnableCoef(ValueError):
    """A resume of a checkpoint whose derived toggle is ON but whose enabling dose was never recorded
    (a pre-v125 ``model_config.json``) and cannot be recovered from the run's ``metadata.json``.
    Re-running cannot change it — ``FATAL_CONFIG``."""


def inherit_derived_enable_coefs(args, saved_ver, model_path, *, announce: bool = True) -> dict:
    """The MIGRATION for a pre-v125 checkpoint whose derived toggle is ON but whose dose is unrecorded.

    A DERIVED structural toggle (``flag_registry`` ``derived=True``) is switched on by a training
    COEFFICIENT (``source_arg`` > 0). Every such coefficient is now a ``ModelVersion`` field, so a
    flagless resume inherits it through the plain ``_resolve`` sweep like any other surface value —
    ``opp_intent_coef`` became one at v125. Before that ``model_config.json`` recorded only the
    ``opp_intent`` BOOL, and a launcher RESTART of a fresh ``--arch production`` run (the restart strips
    the FRESH-only ``--arch``) resolved the coefficient to 0.0 and died at ``check_compatible``:
    ``[ModelVersion] FATAL: opp_intent mismatch: saved=True, current=False`` (F-LG-6 run, 2026-09-30).

    The v125 migration leaves that dose UNRECORDED (``None``) rather than invent one. Here, for each
    such row the argv left untyped (a TYPED value always wins): take the dose the run recorded in its
    ``metadata.json:cli_args`` (the resolved namespace of the launch that trained it), announced as a
    migration; else raise :class:`UnrecordedEnableCoef` naming the flag — never the registry's
    ``on_value``, which would be a training dose nobody chose. Returns ``{dest: (value, source)}``.
    Module-level so ``main.checkargs`` resolves the same namespace.
    """
    from agents.model.flag_registry import REGISTRY, is_enabled

    out: dict = {}
    if saved_ver is None:
        return out
    for f in REGISTRY:
        if not (f.derived and f.source_arg) or getattr(args, f.source_arg, None) is not None:
            continue
        if getattr(saved_ver, f.source_arg, None) is not None or not is_enabled(getattr(saved_ver, f.name, None)):
            continue                     # recorded ⇒ the plain sweep inherits it; OFF ⇒ nothing to keep
        flag = "--" + f.source_arg.replace("_", "-")
        value = _recorded_cli_arg(model_path, f.source_arg)
        if not is_enabled(value):
            raise UnrecordedEnableCoef(
                f"the checkpoint's model_config.json records {f.name}=True but not the dose that enables it "
                f"({f.source_arg}; a config older than the field), and its run's metadata.json:cli_args has "
                f"none either. Pass {flag} <the value the run trained with> — it is a training dose, so it "
                f"is never guessed.")
        setattr(args, f.source_arg, value)
        out[f.source_arg] = (value, "metadata.json:cli_args")
        if announce:
            emit(f"[Resume] MIGRATION: {f.name} is ON but this checkpoint's model_config.json predates "
                 f"{f.source_arg}; inheriting {flag} {value!r} from the run's metadata.json:cli_args")
    return out


def desugar_umbrella_flags(args) -> None:
    """The UMBRELLA desugars, in one place: `--unified-moves` -> `--unified-damage` -> the
    component toggles, and `--damage-matrices` -> its two bools.

    Module-level for the same reason `inherit_saved_flag` is: `main.checkargs` has to build the
    SAME namespace a launch builds before it can read `combination_checks` on it. A flagless run
    resolves `--unified-moves` to 'both', which turns `--damage-op` / `--move-latent` ON — so a
    checker that skipped this would report every damage-family dependency as unsatisfied on a
    command that launches. Mutates `args` in place; prints the two operator notes it always did.
    """
    # --- gen3_arch_surface_guard_v1: `--arch production`, FIRST -------------------------------
    # Before every other desugar, because the others read the values it writes: `--unified-moves`
    # only fills `move_latent` / `damage_topk_k` when they are still None, and `--damage-matrices`'
    # else-branch preserves whatever is already set. Running it first therefore makes the umbrella a
    # DEFAULT that the sugar and every explicit flag alike still override — precedence in one
    # direction, top to bottom, with no special cases.
    #
    # A resume is REFUSED (`combination_checks`' `arch_umbrella_is_fresh_only`, which both surfaces
    # read) rather than handled here: a fork INHERITS its parent's surface through `_resolve`, and
    # writing production's values over that would replace inheritance with a mirror the parent may
    # never have matched — a check_compatible FATAL at best, a silently different network at worst.
    if getattr(args, "arch", None) == "production" and not getattr(args, "model", None):
        from main.train.arch_surface import apply_production_arch, arch_source_tag
        applied = apply_production_arch(args)
        args.arch_source = arch_source_tag()
        print(f"[Arch] --arch production: applied {len(applied)} ARCH-surface flag(s) from "
              f"designs/production_config.json ({args.arch_source}). An explicitly-typed flag "
              f"still wins.")
        # K10(a) THE RECIPE SURFACE, right after the architecture and before every other desugar
        # (and before `resolve_critic_mode`, which then implies the rest of the win-prob critic's
        # settings). Typed tokens win: `recipe_surface.typed_dests` is the parser's record.
        from main.train.recipe_surface import apply_production_recipe
        recipe = apply_production_recipe(args)
        print(f"[Recipe] --arch production: applied {len(recipe)} RECIPE knob(s) from "
              f"designs/production_config.json's recipe.fresh ({args.recipe_source}). An "
              f"explicitly-typed flag still wins.")
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


def resolve_critic_mode(args) -> None:
    """Imply the ONE tri-state flag the win-prob critic settles, in place — `--win-prob-mode shaping` when
    untyped. (Deletion pass P11b deleted `--critic`, `--gamma`, `--victory-value`, `--draw-penalty` and
    `--terminal-indicator`: the critic, the discount and the whole terminal are CONSTANTS of the namespace,
    `parser/objective.py`, so nothing else is left to imply or to refuse.)

    Module-level, and called before the `_resolve` sweep, for `desugar_umbrella_flags`' exact reason:
    `main.checkargs` has to build the SAME effective namespace a launch builds before it can read
    `combination_checks` on it. A checker that skipped it would report the implied value as a missing
    dependency on a command that launches.

    ``win_prob_mode``  'shaping'  the head must EXIST to be the critic ('none' is refused)

    Its argparse default is the `None` sentinel, so an unset flag is distinguishable from a typed one and the
    implication can never overwrite an operator's choice — it is then judged by `combination_checks`. Running
    BEFORE the inheritance sweep is load-bearing: the sweep would otherwise inherit a parent's recorded
    `win_prob_mode` over the implication (a shaped parent's `'none'` is refused outright, D4, so the only
    live case is a parent that recorded another legal mode).
    """
    if getattr(args, "win_prob_mode", None) is None:
        args.win_prob_mode = "shaping"


def resolve_eval_sentinel_regime(args, saved_ver, *, announce: bool = True) -> None:
    """THE EVAL OPPONENT REGIME and the promotion gate it derives, resolved in place.

    RULE OF EVIDENCE 15: a windowed statistic never crosses an opponent-regime boundary. Whether
    the pool SENTINEL a cycle measures the trainee against plays greedy — and draws its team the
    way the trainee does — decides what `win_rate_vs_pool` and `eval/elo` MEAN; the asymmetric
    regime reads **+8.9 pp [+7.0, +10.7]** in the trainee's favour on the same frozen pair the
    dense ladder plays symmetrically. So `--eval-sentinel-greedy` carries an argparse default of
    `None` and is INHERITED here: a run recorded stochastic stays stochastic across every launcher
    restart and every flagless resume, and only a TYPED flag moves it. A FRESH run (no `--model`)
    gets `EVAL_SENTINEL_GREEDY_DEFAULT` = True.

    🚨 THE 49-RUN PRECEDENT is why the inheritance is not optional: `--eval-sentinel-greedy` was ON
    for v5.5 through v8 and was dropped, UNRECORDED, at the v9 launch — 164 stochastic runs were
    then compared against greedy ones with nothing on disk saying the regime had moved.

    THE GATE FOLLOWS THE REGIME, and the ORDER of the three branches is the rule:

    * ``argv`` — an explicit ``--promote-threshold`` always wins.
    * ``regime typed on THIS argv`` — the operator MOVED the regime, so the gate is re-derived from
      the NEW regime rather than inherited from the old one. Inheriting 0.65 into a freshly-greedy
      run would freeze the pool, which is the exact failure the auto-lowering exists to prevent.
    * otherwise — inherit the checkpoint's own gate (which preserves a parent's EXPLICIT value),
      else the regime-derived default.

    Module-level and callable from `main.checkargs` for `resolve_critic_mode`'s reason: a checker
    that re-implemented this would report a *different* effective config than the launch resolves,
    and "what is the baseline?" is answered off that report. `announce=False` suppresses the launch
    line for the offline surfaces.
    """
    from agents.training.snapshot_pool import (
        EVAL_SENTINEL_GREEDY_DEFAULT, promote_threshold_default)

    greedy_typed = args.eval_sentinel_greedy is not None
    greedy_inherited = inherit_saved_flag(args, saved_ver, "eval_sentinel_greedy",
                                          EVAL_SENTINEL_GREEDY_DEFAULT)
    args.eval_sentinel_greedy = bool(args.eval_sentinel_greedy)
    args.eval_sentinel_greedy_source = ("argv" if greedy_typed
                                        else "inherited" if greedy_inherited else "default")

    thr_default = promote_threshold_default(args.eval_sentinel_greedy)
    if args.promote_threshold is not None:
        args.promote_threshold_source = "argv"
    elif greedy_typed:
        args.promote_threshold = thr_default
        args.promote_threshold_source = "default"
    else:
        args.promote_threshold_source = (
            "inherited" if inherit_saved_flag(args, saved_ver, "promote_threshold", thr_default)
            else "default")
    if announce:
        emit(f"⚖️  [EVAL REGIME] eval sentinels "
             f"{'GREEDY + symmetric teams' if args.eval_sentinel_greedy else 'STOCHASTIC @ --self-play-temp'} "
             f"(--{'' if args.eval_sentinel_greedy else 'no-'}eval-sentinel-greedy, "
             f"source={args.eval_sentinel_greedy_source}); "
             f"promote_threshold={args.promote_threshold:g} "
             f"(source={args.promote_threshold_source})")


def resolve_config(args, parser) -> ResolvedRunConfig:
    """Desugar, inherit and validate `args` in place. Returns the values that live outside it."""
    # WHICH FLAGS WERE ACTUALLY TYPED — captured FIRST, before one default is filled, because after
    # `_resolve` an unset tri-state flag is indistinguishable from a typed default and several
    # refusals ("you passed a knob that does nothing") are only honest about a typed value.
    # `main.checkargs` stamps the same marker before it inherits from the parent config, so
    # `combination_checks._typed` gives both surfaces the same answer. See that module's docstring.
    args._explicit_flags = frozenset(d for d, v in vars(args).items() if v is not None)

    # --- Resolve resumable structural toggles (None sentinel = "not passed on the CLI") ---
    # Each version-checked structural toggle defaults to None so a FLAGLESS resume can INHERIT the
    # saved value (the documented `--model … --steps …` command), instead of falling back to OFF and
    # FATALing at check_compatible (saved-ON vs current-default-OFF). An EXPLICIT flag that flips a
    # toggle still FATALs at load (desirable). A fresh run (no --model) → the toggle's OFF default.
    # gen3_shaped_reward_deletion_v1: a RESUME or FORK of a checkpoint trained with the DELETED
    # shaped reward is REFUSED here — before the inheritance sweep reads one recorded flag, and
    # before the migration pops the deleted fields that prove it. Never a silent switch to the
    # terminal alone (owner decision, 2026-09-26).
    if args.model:
        enforce_not_shaped_parent(args.model)
    _saved_ver = _load_saved_version(args.model) if args.model else None
    if args.model and _saved_ver is None:
        print("[Resume] WARNING: saved model_config.json unreadable — structural toggles fall back to "
              "their OFF defaults and may FATAL at the version check; pass them explicitly if needed.")
    # Whether a recorded parent config was READ. `combination_checks`' `needs=("saved_config",)`
    # mechanism reads it (no check declares that need today); `main.checkargs` reports such a check
    # as ADVISORY when it could not read the parent.
    args._saved_config_present = _saved_ver is not None
    _coef_explicit = args.opp_belief_aux_coef is not None

    desugar_umbrella_flags(args)
    # K10(a): a launcher RESTART strips `--arch` (fresh-only). On a same-run restart of an
    # `--arch production` run, every untyped recipe knob is resolved by exactly one route
    # (`recipe_surface.restart_route`: INERT / the resume's own inheritance / the checkpoint's
    # model_config.json / the run's metadata.json:cli_args), announced; a MISSING value REFUSES by
    # name — never a parser default. A fork into a new run dir is untouched.
    from main.train.recipe_surface import RecipeRestartError, inherit_on_restart
    try:
        _restored = inherit_on_restart(args, getattr(args, "run_dir", None), _saved_ver)
    except RecipeRestartError as e:
        from main.exit_codes import TrainExitCode
        print(f"\n[Recipe] FATAL: {e}", file=sys.stderr, flush=True)
        emit(f"[Recipe] FATAL: {e}")
        sys.exit(int(TrainExitCode.FATAL_CONFIG))
    for _d, _v, _src in _restored:
        emit(f"[Recipe] same-run restart of an --arch production run: {_d}={_v!r} from {_src}")
    # Deletion pass D4: a checkpoint that trained the SHAPED critic is REFUSED (run it pinned) — read off the
    # checkpoint's RECORD (there is no `--env-core` / `--critic` to type any more, deletion pass P11b). A
    # python-era checkpoint that trained winprob moves onto the Rust core, announced as a CORE SWITCH
    # (`env_core_switch_line`, below). Before the combination sweep.
    from main.train.rust_env_setup import PythonEraShapedCheckpoint, refuse_python_era_checkpoint
    try:
        refuse_python_era_checkpoint(args.model, saved_ver=_saved_ver)
    except PythonEraShapedCheckpoint as e:
        from main.exit_codes import TrainExitCode
        print(f"\n{e}", file=sys.stderr, flush=True)
        emit(str(e))
        sys.exit(int(TrainExitCode.FATAL_CONFIG))
    # ...and the run's PROVENANCE tags with it: `arch_source` (model_config.json) and `recipe_source`
    # (metadata.json:cli_args) are stamped only by the `--arch` branch the restart stripped, and every
    # later save records the namespace's — so a first restart used to null them.
    from main.train.arch_surface import inherit_arch_source_on_restart
    _arch_tag = inherit_arch_source_on_restart(args, getattr(args, "run_dir", None), _saved_ver)
    if _arch_tag:
        emit(f"[Arch] same-run restart: arch_source={_arch_tag!r} kept from model_config.json")

    # --- gen3_winprob_critic_mode_v1: THE CRITIC MODE, and the composition it implies ------------
    # Resolved BEFORE `_resolve` so the implications below land on the same tri-state sentinels
    # every other flag is inherited through — an implied value must look exactly like a typed one
    # to `_resolve`. The critic itself is a CONSTANT of the namespace (`parser/objective.py`): a recorded
    # shaped parent was REFUSED above (D4), so no parent can hand a different critic to this sweep.
    resolve_critic_mode(args)
    # A pre-v125 checkpoint with opp_intent ON records no dose: migrate it from metadata.json or
    # REFUSE (never guess). Before the sweep, so `_resolve` sees the migrated coefficient as set.
    try:
        inherit_derived_enable_coefs(args, _saved_ver, args.model)
    except UnrecordedEnableCoef as e:
        from main.exit_codes import TrainExitCode
        print(f"\n[Resume] FATAL: {e}", file=sys.stderr, flush=True)
        emit(f"[Resume] FATAL: {e}")
        sys.exit(int(TrainExitCode.FATAL_CONFIG))

    def _resolve(name, default):
        inherit_saved_flag(args, _saved_ver, name, default)
    # gen3_policy_gae_lambda_v1: the PPO POLICY's GAE λ. 0.80 is the value both model_build sites
    # hardcoded for every run to date, so an unset flag on a fresh run is byte-identical; a flagless
    # resume inherits the parent's recorded value (a pre-v123 config migrates to 0.80 — the only
    # possible past).
    _resolve("policy_gae_lambda", 0.80)
    # gen3_diagnostics_cadence_v1: the optional learner telemetry's cadence. A FRESH run takes the
    # production default; a flagless resume keeps the parent's recorded regime (a pre-v124 parent
    # migrates to 1 — every update, what it actually ran), so a live run's TB series never change
    # cadence at a restart unless the flag is NAMED.
    from agents.training.instrumented_ppo.diagnostics_cadence import DIAGNOSTICS_EVERY_DEFAULT
    _resolve("diagnostics_every", DIAGNOSTICS_EVERY_DEFAULT)
    # T17 (gen3_mirrored_pairs_v1): the in-loop eval's PAIRING REGIME. A fresh run takes OFF (the
    # orchestrator flips it at an era start); a flagless resume keeps the regime its checkpoint recorded,
    # so a live run's win_rate_vs_pool / eval/elo never cross the boundary unless the flag is NAMED.
    _mp_typed = args.eval_mirrored_pairs is not None
    _mp_inherited = inherit_saved_flag(args, _saved_ver, "eval_mirrored_pairs", False)
    args.eval_mirrored_pairs = bool(args.eval_mirrored_pairs)
    args.eval_mirrored_pairs_source = ("argv" if _mp_typed else "inherited" if _mp_inherited else "default")
    # T6 (gen3_sprt_promotion_v1): the PROMOTION regime, resolved and inherited exactly like the pairing.
    _sp_typed = args.promotion_sprt is not None
    _sp_inherited = inherit_saved_flag(args, _saved_ver, "promotion_sprt", False)
    args.promotion_sprt = bool(args.promotion_sprt)
    args.promotion_sprt_source = ("argv" if _sp_typed else "inherited" if _sp_inherited else "default")
    emit(f"⚖️  [EVAL REGIME] promotion: "
         f"{'SPRT on fresh mirrored pairs (T6; --promote-threshold not read)' if args.promotion_sprt else 'first win_rate_vs_pool > --promote-threshold'} "
         f"(--{'' if args.promotion_sprt else 'no-'}promotion-sprt, source={args.promotion_sprt_source})")
    emit(f"⚖️  [EVAL REGIME] in-loop eval pairing: "
         f"{'MIRRORED TEAM PAIRS (each pairing from both sides, one seed; the pair is the unit)' if args.eval_mirrored_pairs else 'unpaired games'} "
         f"(--{'' if args.eval_mirrored_pairs else 'no-'}eval-mirrored-pairs, source={args.eval_mirrored_pairs_source})")
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
    _resolve("policy_grad_coef", 1.0)               # v102 training-only (inherited like opp_belief_aux_coef; 1.0 = upstream)
    _resolve("value_threat_inject", False)     # v64 structural bool (version-checked, fresh-only)
    _resolve("opp_intent_coef", 0.0)           # v67 training-only coef; the HEADS are structural
    _resolve("beta_setvalued_coef", 0.0)       # training-only coef; no module, no version gate
    _resolve("intent_label_bot_weight", 1.0)   # v97 training-only (inherited like opp_belief_aux_coef)
    # gen3_fork_v1 — the FORK ARM. TRAINING-only and inherited, for the sharpest version of that
    # reason yet: the fraction is what the run COSTS, and a flagless restart that dropped it would
    # halve the run's simulation bill mid-arm while every argv, model_config and ledger line still
    # said it was the forked arm. The four knobs below it are inherited so a restart cannot change
    # WHICH decisions are forked or WHAT the branches share — either would be a distribution shift
    # inside a single registered arm.
    _resolve("fork_fraction", 0.0)
    _resolve("fork_branches", 3)
    _resolve("fork_contested_gap", 0.40)
    _resolve("fork_contested_absv", 0.0)
    _resolve("fork_max_per_battle", 1)
    _resolve("fork_crn", "dice_and_draws")
    # gen3_ridealong_heads_v1 (v126): four STRUCTURAL toggles, so a flagless resume must inherit
    # them — dropping one would fail `check_compatible`, which is the loud half working.
    _resolve("ridealong_ensemble", 0)
    _resolve("ridealong_rnd", False)
    _resolve("ridealong_adv", 0)
    _resolve("ridealong_opp", 0)
    _resolve("ridealong_rnd_variants", "off")      # v127 structural str (canonical comma list)
    _resolve("belief_tokens", "blob")              # v136 structural str (X5 U2; version-checked, fresh-only)
    _resolve("oracle_reveal", "off")               # v137 RESUME-IMMUTABLE str (the diagnostic observation mode; flagless resume inherits)
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
    _resolve("intent_threshold", False)        # v84 structural, version-checked (gen3_intent_threshold_v1)
    _resolve("intent_conditional", False)      # v85 structural, version-checked (gen3_intent_conditional_v1)
    _resolve("op_drop_renders", False)         # v86 structural, version-checked (gen3_op_lean_forward_v1)
    _resolve("op_believed_lean", False)        # v86 structural, version-checked (gen3_op_lean_forward_v1)
    # gen3_capacity_telemetry_v1 — the live saturation early-warnings. The training-only provenance class:
    # recorded for provenance, never gated, and read back here so a flagless resume (or a
    # hand-typed one between launcher restarts) keeps logging the run's own `capacity/*` series.
    _resolve("capacity_telemetry", False)      # v101 training-only diagnostic (no loss, no grad)
    _resolve("canary_reset_steps", 1_000_000)  # v101 training-only
    _resolve("capacity_cosine_every", 50)      # v101 training-only
    _resolve("capacity_velocity_every", 50)    # v101 training-only
    _resolve("species_prior_fusion", False)    # v68 structural bool (version-checked, fresh-only)
    _resolve("t0_species_prior", False)        # v72 structural bool (version-checked, fresh-only)
    # gen3_distill_target_gate_v1 (config v103) — the rank tripwire. The training-only provenance class: recorded
    # for provenance, never gated, read back here so a flagless resume keeps the arm it was launched as.
    _resolve("rank_tripwire", "warn")          # v103 training-only diagnostic (§4.1; no loss, no grad)
    _resolve("rank_tripwire_drop", 0.20)       # v103 training-only TRIP threshold (fractional drop)
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
    # gen3_eval_sentinel_greedy_default_v1 — THE EVAL OPPONENT REGIME, and the gate it implies.
    # Module-level for the same reason `inherit_saved_flag` and `resolve_critic_mode` are:
    # `main.checkargs` has to reach the same resolved values a launch reaches.
    resolve_eval_sentinel_regime(args, _saved_ver)
    _resolve("belief_grad_mode", "shaping")    # v41 resume-immutable training hparam (vf_coef class; flagless resume inherits)
    _resolve("hp_belief_mode", "composed")     # v53 STRUCTURAL (version-checked, fresh-only)
    _resolve("hp_type_belief_coef", 0.05)      # training-only (inherited like spread_belief_coef)
    _resolve("item_belief_coef", 0.05)         # training-only (inherited like hp_type_belief_coef)
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
    if args.policy_grad_coef is not None and args.policy_grad_coef < 0.0:
        # A negative coef would ASCEND the PPO surrogate — train the policy to be maximally wrong.
        # 0.0 is the intended floor. policy_grad_coef is training-only
        # (not version-locked), so guard it here — the only gate.
        parser.error("--policy-grad-coef must be >= 0 (1 = upstream PPO; 0 = no policy-gradient term)")
    _adaptive_batch_guards(args, parser)
    if args.intent_label_bot_weight is not None and args.intent_label_bot_weight < 0.0:
        # A negative weight would train alpha/beta to be MAXIMALLY wrong about bots — the opposite
        # of "train on them less". 0.0 (ignore bot rows entirely) is the intended floor.
        # Training-only (not version-locked), so this parser check is the only gate.
        parser.error("--intent-label-bot-weight must be >= 0 (0 = train on no bot rows; 1 = off)")
    if args.diagnostics_every < 1:
        parser.error("--diagnostics-every must be >= 1 (1 = the optional telemetry every update)")
    if not (0.0 <= args.policy_gae_lambda <= 1.0):
        # A single-value RANGE check. GAE's λ weights the n-step advantage estimators by
        # (1-λ)λ^(n-1): outside [0, 1] that is not an average of estimators at all.
        parser.error("--policy-gae-lambda must be in [0, 1] "
                     "(0.80 = the default every run to date used; 1 = Monte-Carlo; 0 = one-step TD)")
    for _ra in ("ridealong_ensemble", "ridealong_adv", "ridealong_opp"):
        _k = getattr(args, _ra, None)
        if _k is not None and not 0 <= int(_k) <= 12:
            parser.error(f"--{_ra.replace('_', '-')} must be in [0, 12] members (0 = off)")
    if args.fork_fraction is not None and not (0.0 <= args.fork_fraction <= 1.0):
        parser.error("--fork-fraction must be in [0, 1] "
                     "(0 = off; it is a fraction of the buffer's decisions)")
    if args.fork_contested_gap is not None and not (0.0 < args.fork_contested_gap <= 1.0):
        # A single-value RANGE check (the cross-flag half is in `combination_checks`). It is a
        # QUANTILE, so 0 would select nothing and a value above 1 is not a quantile at all — both
        # are refused rather than clamped, because a clamp would let a fat-fingered "40" run as
        # "fork everything" under a flag whose registered value is 0.40.
        parser.error("--fork-contested-gap must be in (0, 1] — it is the QUANTILE of the top-2 "
                     "logit gap, not an absolute gap (0.40 is the registered value)")
    if args.fork_contested_absv is not None and not (0.0 <= args.fork_contested_absv <= 0.5):
        parser.error("--fork-contested-absv must be in [0, 0.5] (|V - 0.5| cannot exceed 0.5; "
                     "0 = off, and off is what the registered endpoint assumes)")
    if args.fork_max_per_battle is not None and args.fork_max_per_battle < 1:
        parser.error("--fork-max-per-battle must be >= 1 (forks per episode slice)")
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
    if getattr(args, "checkpoint_every_steps", None) is not None and args.checkpoint_every_steps < 1:
        parser.error("--checkpoint-every-steps must be >= 1 (it is an ENV-STEP interval; there is "
                     "no 'off' value — omit the flag for the 2,400,000-env-step default)")
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
    # M5 Lane G: the collector flags' defaults, AFTER the sweep (which must see "untyped" as None).
    from main.train.rust_env_setup import env_core_switch_line, resolve_env_core_args
    resolve_env_core_args(args)
    _switch = env_core_switch_line(args)
    if _switch:
        emit(_switch)

    # --- gen3_arch_surface_guard_v1: IS THIS THE ARCHITECTURE YOU MEANT? -----------------------
    # AFTER the combination sweep, deliberately: a broken flag combination is a bug in the command
    # and must be reported before a question about the command's INTENT.
    #
    # 🚨 THIS SURFACE REPORTS AND RECORDS; IT DOES NOT REFUSE, and that is a placement decision
    # rather than an omission. `resolve_config` runs in the CHILD, by which time the launcher has
    # already resolved the pin, created the isolated worktree and made the run dir — so a refusal
    # here is both late and a duplicate of one already made. The three surfaces that DO refuse all
    # answer before anything exists: `_prepare_session` (before the worktree and the `makedirs`),
    # `--dry-run`, and `python -m main.checkargs`. All four read the same `arch_surface.report`.
    #
    # The residue is a DIRECT `python src/main/train_rl_agent.py …`, which is the smoke/debug path
    # (`--debug` is non-production by construction and would trip this on every run). There the
    # block is printed and `arch_tables_test::test_production_config_matches_newest_run` remains
    # the gate, exactly as before.
    _arch_report = arch_surface.report(
        args,
        fresh=not bool(args.model),
        allowed=bool(getattr(args, "allow_nonproduction_arch", False)),
        umbrella=getattr(args, "arch", None),
    )
    if not getattr(args, "arch_source", None) and _arch_report.allowed and _arch_report.diffs:
        args.arch_source = (f"nonproduction ({arch_surface.ALLOW_FLAG}, "
                            f"{len(_arch_report.diffs)} key(s) vs {_arch_report.source_tag})")
    if _arch_report.fresh and not getattr(args, "debug", False):
        for _line in arch_surface.report_lines(_arch_report):
            emit(_line)
    # K10(a) THE RECIPE SURFACE — reported and recorded here, refused (like the ARCH surface) only
    # by the surfaces that answer before anything exists: checkargs, --dry-run, the launcher gate.
    from main.train import recipe_surface
    try:
        _recipe_report = recipe_surface.report(
            args, fresh=not bool(args.model),
            restart=bool(getattr(args, "_recipe_restart_inherited", ())),
            allowed=bool(getattr(args, "allow_nonproduction_recipe", False)),
            umbrella=getattr(args, "arch", None))
    except recipe_surface.RecipeError as _e:    # reported, never a crash in the child
        emit(f"⚠️  RECIPE SURFACE unavailable: {_e}")
    else:
        if ((_recipe_report.fresh or _recipe_report.restart_inherited)
                and not getattr(args, "debug", False)):
            for _line in recipe_surface.report_lines(_recipe_report):
                emit(_line)

    emit("🌉 Transport: in-process BattleStream bridge [rust] for BOTH training "
         "and eval (no Showdown server needed)")
    # One-time startup warning naming the Rust bridge's honest remaining scope limits (an INCOMPLETE
    # modeled move set that fail-louds) — resolve/build the binary NOW so a missing toolchain fails
    # loudly at startup, not deep inside the first env reset.
    from utils.bridge.sim_bridge_bin import (
        warn_rust_deferrals, resolve_and_publish_sim_bridge_bin)
    warn_rust_deferrals(emit)
    # Build ONCE here and PUBLISH the path (POKESIM_SIM_BRIDGE_BIN) so every eval-worker
    # subprocess inherits a ready binary instead of racing its own `cargo build` on first spawn.
    _rust_bin = resolve_and_publish_sim_bridge_bin()
    emit(f"🦀 [BRIDGE=rust] sim_bridge binary (prebuilt, published to children): {_rust_bin}")
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
    # ⚠️ THE SECOND HALF IS NOT OPTIONAL. `check_shape_stability` REFUSES an update (the real size:
    # --rollout-target-samples, else n_steps * n_envs) that does not divide by --batch-size, so a
    # device-only default would convert a class of command that works
    # today into a FATAL_CONFIG exit — the same failure the cpu conditioning above exists to avoid,
    # one flag over. A DEFAULT yields
    # to the config the user actually typed and says why; an EXPLICIT --compile-trainer still hits
    # the refusal, loudly, because there the user asked for something impossible.
    if args.compile_trainer is None:
        args.compile_trainer, _ct_why = resolve_compile_trainer_auto(
            device=args.device, debug=args.debug, n_steps=args.n_steps, n_envs=args.n_envs,
            batch_size=args.batch_size, rollout_target_samples=getattr(args, "rollout_target_samples", 0))
        if _ct_why:
            emit("⚡ --compile-trainer would be ON by default here, but this config cannot take "
                 f"it — leaving it OFF rather than refusing to launch. Reason: {_ct_why} "
                 "(pass --compile-trainer explicitly to make this a hard error instead.)")
        if args.compile_trainer:
            # LOUD, and at STARTUP: a default nobody typed a flag for announces itself.
            emit("⚡ --compile-trainer ON by default (device=cuda) — ~1.75x on the PPO train step. "
                 "--no-compile-trainer opts out. Compile failure is FATAL by design.")

    return ResolvedRunConfig(annealing_mode=annealing_mode, log_level=log_level)
