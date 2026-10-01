"""THE RECIPE SURFACE — "is this the TRAINING RECIPE you meant?", the ARCH SURFACE's twin (K10(a)).

**WHY IT EXISTS.** `--arch production` applied the production ARCHITECTURE and nothing else: five
parser defaults differed from the live recipe (`--n-envs` 32 vs 48, `--batch-size` 4096 vs 2048,
`--n-epochs` 5 vs 10, `--ent-coef` 0.02 vs 0.05, `--clip-range-vf` 0.5 vs none), and so did
`--grad-accum-steps` (1 vs 32 — an effective batch of 4,096 instead of 65,536), `--self-play`,
`--critic` (`shaped` vs `winprob`) with its three REQUIRED reward values, and the supervision
doses. A fresh argv that omitted them parsed, resolved, dry-ran and launched — and trained a
different recipe: the recipe-side twin of the 2026-09-06 stripped-architecture incident
(`arch_surface`'s docstring), closed the same way — a DECLARED surface, a mirror, one applier, one
report, read by every surface that launches.

**WHERE THE VALUES LIVE** — `designs/production_config.json`'s nested ``recipe`` block (stripped from
`agents.training.baselines.production_config()`, so no arch consumer ever sees it):

  * ``recipe.fresh`` — N0's MEASURED fresh recipe (`models/ai_v14_01_base`, the lineage's fresh
    launch), every knob it launched with, including the critic and its reward values, plus
    ``kl_controller``: the controller constants it ran with (not flags — pinned against the
    callbacks' own defaults by `recipe_surface_test`). A key that is ALSO a recorded top-level
    mirror field must EQUAL it (`production_recipe` refuses otherwise): one truth, two readers.
  * ``recipe.sizing`` — THE ONE PLACE the M5 SIZING study's verdict fills (`SIZING_ROWS`): the env core
    (`rust` — THE M5 SWITCH, `gen3_env_core_switch_v1`), N, the n_steps maximum, the collector's update
    size and T2's slots / buckets / lanes, plus ``verdict`` (null = N* PENDING). `recipe_blocks` merges it
    into the fresh recipe, so every reader treats a sizing row like any other row; a collector-only row
    (`COLLECTOR_ROWS`) whose value is null is left untyped, and none is applied on the python core.
  * ``recipe.fork`` — what a generalist FORK changes: the E5 verdict (5 epochs at a FROZEN 5.6e-5,
    the same dose as 10 at 2.8e-5). `--fork-lr` is refused on a fresh run, so `--arch production`
    never applies this; a fork's argv is compared with it as INFO. 🚨 E5's 5 epochs are NEVER paired
    with the fresh LR — that pairing was never measured (design_learner_recipe.md Decision record).

Sources of every value: `designs/endstate/design_learner_recipe.md` §3.22; the doc and the block are
held together by `src/recipe_doc_gate_test.py`.

**WHAT `--arch production` DOES.** Writes every ``recipe.fresh`` row the argv did not TYPE, as if
typed, after the ARCH surface and before `resolve_critic_mode` (which then implies the rest of an
applied `--critic winprob`). "Typed" is a fact the parser records (`record_typed_recipe_flags`).

**THE REFUSAL.** A FRESH argv that differs on an UNTYPED knob refuses (`checkargs`, `--dry-run`, the
launcher); a TYPED difference is the arm's lever (INFO); `--allow-nonproduction-recipe` consents.

**THE RESTART** (`inherit_on_restart`). A launcher restart strips the fresh-only `--arch`. Converged
with the general rule that a resume inherits its surface from the CHECKPOINT: on a same-run restart
of an `--arch production` run, every untyped row is resolved by exactly one route —

  1. `--lr`, `--batch-size`, `--n-steps`, `--gamma`: INERT on a resume (SB3 restores the
     checkpoint's own values) — never re-applied here, whatever the argv says;
  2. a recorded tri-state field (`critic`, the doses, `opp_intent_coef` from config v125,
     `policy_gae_lambda`, …): the general restart mechanism (`_resolve` / `resolve_critic_mode` /
     `config.inherit_derived_enable_coefs`, `68850f27`) inherits it — untouched here;
  3. a recorded field with a concrete parser default (`terminal_indicator`, `victory_value`,
     `draw_penalty`, `vf_coef` — value-CHECKED, so the default would FATAL): from the checkpoint's
     `model_config.json`, ANNOUNCED;
  4. a knob recorded nowhere else (`n_envs`, `n_epochs`, `ent_coef`, …): from the run's own
     `metadata.json:cli_args`, ANNOUNCED.

A value MISSING from its route (no `cli_args`, no key, no field) REFUSES by name
(`RecipeRestartError`, `FATAL_CONFIG`) — never a parser or registry default. A fork into a new run
dir inherits nothing from here: its recipe is its argv, as always, and the report says so.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

#: The namespace attribute the recording actions write: the dests the argv TYPED.
TYPED_ATTR = "_recipe_typed"

#: The consent flag — launcher-forwarded, recorded in `metadata.json`'s `cli_args`.
ALLOW_FLAG = "--allow-nonproduction-recipe"

FRESH_KEY = "fresh"
FORK_KEY = "fork"
#: ``recipe.sizing`` — THE ONE PLACE the M5 SIZING study's verdict fills (order constraint 5 of
#: `program_rust_core.md`): the env core and every SIZE the run declares at startup (N, the n_steps
#: maximum, the collector's update size, the inference service's slots / buckets / lanes). Merged into
#: the fresh recipe by `recipe_blocks`, so every reader (the applier, the diff, the restart, the doc
#: gate) treats a sizing row exactly like any other recipe row.
SIZING_KEY = "sizing"
#: ``recipe.sizing.verdict`` — the Decision-record pointer of the sizing verdict that set the block;
#: ``null`` = PENDING (the values are the pre-sizing N = 48 shape). Not a flag.
VERDICT_KEY = "verdict"
KL_KEY = "kl_controller"
#: The controller constants `recipe.fresh.kl_controller` records (constructor values, not flags).
KL_CONSTANTS = ("target_kl", "kl_factor", "lr_factor")

#: ``RecipeRow.unset`` for a row whose parser default is CONCRETE: read the default off the parser.
PARSER_DEFAULT = "<parser default>"
#: ``RecipeRow.unset`` for `--gamma`: an unset one resolves to its CRITIC's declared discount
#: (`agents.model.critic_mode.critic_gamma`), so it is read off the resolved critic, never a constant.
CRITIC_PAIRED = "<the --critic pairing>"

#: INERT on a resume — SB3 restores the checkpoint's own values (root CLAUDE.md). Never re-applied
#: on a restart, so nothing here fights that restoration.
INERT_ON_RESUME = frozenset({"lr", "batch_size", "n_steps", "gamma"})


class RecipeError(ValueError):
    """The mirror and the declared rows disagree — a rename, a dropped block, a stray key, or a
    block value that contradicts the same recorded field at the mirror's top level."""


class RecipeRestartError(ValueError):
    """A same-run restart whose recipe value cannot be found where it must come from. Re-running
    cannot change it — `FATAL_CONFIG`."""


class RecipeRow(NamedTuple):
    dest: str
    flag: str
    #: What an UNTYPED value resolves to on a FRESH run: `PARSER_DEFAULT`, or — for a tri-state
    #: (`default=None`) flag — the value `resolve_config` fills (pinned against `config.py`).
    unset: Any
    note: str


#: ``recipe.sizing`` — the env core and the run's declared SIZES (`SIZING_KEY`). Order is the report's.
SIZING_ROWS: Tuple[RecipeRow, ...] = (
    RecipeRow("env_core", "--env-core", PARSER_DEFAULT, "THE M5 SWITCH — the Rust env core (§1 row 1a)"),
    RecipeRow("n_envs", "--n-envs", PARSER_DEFAULT, "§1 row 1 — the SIZING study's to change"),
    RecipeRow("n_steps", "--n-steps", PARSER_DEFAULT, "§1 row 2 — the SIZING study's / Lane G's (the "
              "rust core's n_steps MAXIMUM: the buffer is preallocated at it)"),
    RecipeRow("rollout_target_samples", "--rollout-target-samples", 0,
              "the collector's update size; null / 0 = n_envs x n_steps"),
    RecipeRow("trainee_slots", "--trainee-slots", 1, "T2's trainee slots; null = the collector's (1, or 3 "
              "under per-game pinning)"),
    RecipeRow("t2_buckets", "--t2-buckets", None, "T2's opponent buckets; null = derived at startup"),
    RecipeRow("t2_lanes", "--t2-lanes", 0, "T2's lanes; null / 0 = derived (min(slots, 8))"),
)
#: The sizing rows that act ONLY under `--env-core rust` (`combination_checks._ENV_CORE_ONLY_DESTS`):
#: a ``null`` value means "the collector derives it" and the row is left UNTYPED; a value is applied
#: and compared only when the resolved env core is ``rust`` (on python it would be silently inert).
COLLECTOR_ROWS = frozenset({"rollout_target_samples", "trainee_slots", "t2_buckets", "t2_lanes"})

#: ``recipe.fresh`` — N0's launch (the sizing rows above come from ``recipe.sizing``).
FRESH_ROWS: Tuple[RecipeRow, ...] = (
    RecipeRow("batch_size", "--batch-size", PARSER_DEFAULT, "§1 row 3 — the micro-batch"),
    RecipeRow("grad_accum_steps", "--grad-accum-steps", PARSER_DEFAULT, "§1 row 4 — K"),
    RecipeRow("n_epochs", "--n-epochs", PARSER_DEFAULT, "§1 row 6 — N0's measured 10"),
    RecipeRow("lr", "--lr", PARSER_DEFAULT, "§1 row 9 — the fresh KL controller's seed"),
    RecipeRow("min_lr", "--min-lr", PARSER_DEFAULT, "§1 row 8 — the controller's floor"),
    RecipeRow("max_lr", "--max-lr", None, "§1 row 8 — unset ⇒ 2 × lr"),
    RecipeRow("anneal_lr_start_steps", "--anneal-lr-start-steps", None, "§1 row 9 — no cosine phase"),
    RecipeRow("weight_decay", "--weight-decay", PARSER_DEFAULT, "§1 row 18"),
    RecipeRow("clip_range", "--clip-range", PARSER_DEFAULT, "§1 row 7"),
    RecipeRow("clip_range_vf", "--clip-range-vf", PARSER_DEFAULT, "§1 row 14 — none (INERT under winprob)"),
    RecipeRow("ent_coef", "--ent-coef", PARSER_DEFAULT, "§1 row 11"),
    RecipeRow("gamma", "--gamma", CRITIC_PAIRED, "§1 row 12 — the critic's declared discount "
              "(critic_mode.critic_gamma): 1.0 under winprob; a TYPED --critic shaped gets 0.9999"),
    RecipeRow("policy_gae_lambda", "--policy-gae-lambda", 0.80, "§1 row 13"),
    RecipeRow("self_play", "--self-play", PARSER_DEFAULT, "§1 row 22"),
    RecipeRow("critic", "--critic", "shaped", "§1 row 14 — the win-prob critic"),
    RecipeRow("terminal_indicator", "--terminal-indicator", PARSER_DEFAULT, "§1 row 14 — REQUIRED by winprob"),
    RecipeRow("victory_value", "--victory-value", PARSER_DEFAULT, "§1 row 14 — REQUIRED by winprob"),
    RecipeRow("draw_penalty", "--draw-penalty", PARSER_DEFAULT, "§1 row 14 — REQUIRED by winprob"),
    RecipeRow("vf_coef", "--vf-coef", PARSER_DEFAULT, "§1 row 14"),
    RecipeRow("opp_belief_aux_coef", "--opp-belief-aux-coef", 0.0, "§1 row 16"),
    RecipeRow("opp_intent_coef", "--opp-intent-coef", 0.0, "§1 row 16"),
    RecipeRow("move_belief_coef", "--move-belief-coef", 0.0, "§1 row 16"),
    RecipeRow("move_belief_latent_coef", "--move-belief-latent-coef", 0.0, "§1 row 16"),
    RecipeRow("spread_belief_coef", "--spread-belief-coef", 0.0, "§1 row 16"),
    RecipeRow("hp_type_belief_coef", "--hp-type-belief-coef", 0.05, "§1 row 16"),
    RecipeRow("item_belief_coef", "--item-belief-coef", 0.05, "§1 row 16"),
    RecipeRow("beta_setvalued_coef", "--beta-setvalued-coef", 0.0, "§1 row 16"),
    RecipeRow("intent_label_bot_weight", "--intent-label-bot-weight", 1.0, "§1 row 16"),
)
#: Every row the fresh recipe declares — ``recipe.sizing`` first, then ``recipe.fresh``.
ROWS: Tuple[RecipeRow, ...] = SIZING_ROWS + FRESH_ROWS

#: ``recipe.fork`` — the FORK-only rows; a fork's other knobs are the fresh values it inherits
#: from the recipe unless ``recipe.fork`` overrides them (today: `n_epochs`).
FORK_ROWS: Tuple[RecipeRow, ...] = (
    RecipeRow("fork_lr", "--fork-lr", None, "§1 row 9 — E5: a frozen 5.6e-5"),
    RecipeRow("fork_lr_freeze", "--fork-lr-freeze", PARSER_DEFAULT, "§1 rows 8-9"),
)
#: ``recipe.fork`` may override these fresh rows (E5's 5 epochs).
FORK_OVERRIDES = frozenset({"n_epochs"})

ALL_ROWS: Tuple[RecipeRow, ...] = ROWS + FORK_ROWS
_ROW: Dict[str, RecipeRow] = {r.dest: r for r in ALL_ROWS}


# ------------------------------------------------------------------------ "was it TYPED?"
_RECORDING: Dict[type, type] = {}


def _recording(cls: type) -> type:
    """A subclass of an argparse Action class that also records its dest as TYPED. argparse calls an
    action only for a token ON the command line, so this is the exact fact — including a typed value
    that EQUALS the default, which no comparison of values can recover."""
    sub = _RECORDING.get(cls)
    if sub is None:
        def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[no-untyped-def]
            cls.__call__(self, parser, namespace, values, option_string)
            typed = set(getattr(namespace, TYPED_ATTR, None) or ())
            typed.add(self.dest)
            setattr(namespace, TYPED_ATTR, frozenset(typed))
        sub = type(f"RecipeTyped{cls.__name__}", (cls,), {"__call__": __call__})
        _RECORDING[cls] = sub
    return sub


def record_typed_recipe_flags(parser: argparse.ArgumentParser) -> None:
    """Make every RECIPE row's action record that it was typed. Called once, by `build_parser`."""
    dests = {r.dest for r in ALL_ROWS}
    for action in parser._actions:
        if action.dest in dests:
            action.__class__ = _recording(type(action))


def typed_dests(ns: Any) -> FrozenSet[str]:
    return frozenset(getattr(ns, TYPED_ATTR, None) or ())


# --------------------------------------------------------------------------------- the mirror
def _raw_mirror() -> Dict[str, Any]:
    from agents.training import baselines
    doc = baselines.production_config()
    doc[baselines.RECIPE_BLOCK_KEY] = baselines.production_recipe_block()
    return doc


def recipe_blocks(mirror: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(fresh, fork)`` — validated. Raises `RecipeError` on any key-set or agreement defect."""
    from agents.training.baselines import RECIPE_BLOCK_KEY
    doc = _raw_mirror() if mirror is None else dict(mirror)
    block = doc.get(RECIPE_BLOCK_KEY)
    if not isinstance(block, dict) or set(block) != {FRESH_KEY, FORK_KEY, SIZING_KEY}:
        raise RecipeError("the production mirror's `recipe` block must hold exactly "
                          f"`{FRESH_KEY}`, `{FORK_KEY}` and `{SIZING_KEY}` "
                          f"(got {sorted(block) if isinstance(block, dict) else block!r})")
    fresh, fork, sizing = dict(block[FRESH_KEY]), dict(block[FORK_KEY]), dict(block[SIZING_KEY])
    s_declared = {r.dest for r in SIZING_ROWS}
    if set(sizing) != s_declared | {VERDICT_KEY}:
        raise RecipeError(f"`recipe.{SIZING_KEY}` must hold exactly {sorted(s_declared | {VERDICT_KEY})} "
                          f"(got {sorted(sizing)}) — the sizing verdict fills ONE declared block")
    both = sorted(s_declared & set(fresh))
    if both:
        raise RecipeError(f"{both} are in both `recipe.{FRESH_KEY}` and `recipe.{SIZING_KEY}` — one place")
    if sizing["env_core"] not in ("python", "rust"):
        raise RecipeError(f"`recipe.{SIZING_KEY}.env_core` must be 'python' or 'rust' (got {sizing['env_core']!r})")
    fresh.update({k: v for k, v in sizing.items() if k != VERDICT_KEY})
    declared = {r.dest for r in ROWS}
    missing = sorted(declared - set(fresh))
    stray = sorted(set(fresh) - declared - {KL_KEY})
    if missing or stray:
        raise RecipeError(f"`recipe.fresh` does not match the declared rows "
                          f"(main.train.recipe_surface.ROWS): missing {missing}, undeclared {stray} "
                          "— renamed or dropped?")
    kl = fresh.get(KL_KEY)
    if not isinstance(kl, dict) or set(kl) != set(KL_CONSTANTS):
        raise RecipeError(f"`recipe.fresh.{KL_KEY}` must hold exactly {list(KL_CONSTANTS)}")
    fork_allowed = {r.dest for r in FORK_ROWS} | FORK_OVERRIDES
    if not {r.dest for r in FORK_ROWS} <= set(fork) or set(fork) - fork_allowed:
        raise RecipeError(f"`recipe.fork` must hold {sorted(r.dest for r in FORK_ROWS)} and may "
                          f"override only {sorted(FORK_OVERRIDES)} (got {sorted(fork)})")
    from agents.model.critic_mode import critic_gamma
    if not _agree(fresh["gamma"], critic_gamma(fresh["critic"])):
        raise RecipeError(f"`recipe.fresh.gamma` {fresh['gamma']!r} is not its critic's declared discount "
                          f"(critic_mode.critic_gamma({fresh['critic']!r}) = {critic_gamma(fresh['critic'])!r})"
                          " — the discount is PAIRED with the critic; change the pairing, not one side")
    top = {k: v for k, v in doc.items() if k != RECIPE_BLOCK_KEY}
    clash = sorted(k for k in declared if k in top and not _agree(fresh[k], top[k]))
    if clash:
        raise RecipeError("`recipe.fresh` contradicts the mirror's recorded field(s) "
                          + ", ".join(f"{k} (recipe {fresh[k]!r} vs field {top[k]!r})" for k in clash)
                          + " — one truth: change both, together")
    return fresh, fork


def production_recipe(mirror: Optional[Dict[str, Any]] = None, *, kind: str = FRESH_KEY) -> Dict[str, Any]:
    """`{dest: value}`: the FRESH recipe (`ROWS`), or — `kind="fork"` — the fresh recipe overlaid
    with ``recipe.fork`` plus the fork-only rows (`ALL_ROWS`)."""
    fresh, fork = recipe_blocks(mirror)
    out = {r.dest: fresh[r.dest] for r in ROWS}
    if kind == FORK_KEY:
        out.update(fork)
    elif kind != FRESH_KEY:
        raise ValueError(kind)
    return out


def sizing_block(mirror: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``recipe.sizing`` raw, the ``verdict`` included (validated through `recipe_blocks`)."""
    from agents.training.baselines import RECIPE_BLOCK_KEY
    recipe_blocks(mirror)
    doc = _raw_mirror() if mirror is None else dict(mirror)
    return dict(doc[RECIPE_BLOCK_KEY][SIZING_KEY])


def production_env_core(mirror: Optional[Dict[str, Any]] = None) -> str:
    """The PRODUCTION env core (``recipe.sizing.env_core``) — what an untyped `--env-core` resolves to
    on a fresh `--arch production` launch (a `--model` launch inherits its checkpoint's core instead)."""
    return str(production_recipe(mirror)["env_core"])


def kl_controller(mirror: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return dict(recipe_blocks(mirror)[0][KL_KEY])


def source_tag() -> str:
    """Which mirror CONTENT the recipe came from (the file's git blob hash, as `--arch` records)."""
    from main.train.arch_surface import arch_source_tag
    return arch_source_tag()


# ----------------------------------------------------------------------------------- the umbrella
def apply_production_recipe(ns: Any, mirror: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
    """`--arch production`'s recipe half: every UNTYPED ``recipe.fresh`` row set as if typed.
    Stamps `ns.recipe_source` (recorded in `metadata.json`'s `cli_args`)."""
    from agents.model.critic_mode import critic_gamma
    want = production_recipe(mirror)
    typed = typed_dests(ns)
    # `--gamma` is PAIRED with the critic: an untyped one takes the discount of the critic this launch
    # will train — a TYPED `--critic shaped` gets the shaped critic's 0.9999, never recipe.fresh's
    # winprob 1.0 (a pairing no run trained). With the critic untyped this IS recipe.fresh's gamma
    # (`recipe_blocks` refuses a block whose gamma is not its critic's).
    critic = ns.critic if "critic" in typed else want["critic"]
    core = ns.env_core if "env_core" in typed else want["env_core"]
    applied: List[Tuple[str, Any]] = []
    for r in ROWS:
        if r.dest in typed:
            continue
        if r.dest in COLLECTOR_ROWS and (want[r.dest] is None or core != "rust"):
            continue                       # derived by the collector / inert on the python core
        value = critic_gamma(critic) if r.dest == "gamma" else want[r.dest]
        setattr(ns, r.dest, value)
        applied.append((r.dest, value))
    ns.recipe_source = source_tag()
    return applied


# ---------------------------------------------------------------------------- the restart path
def _is_production_launch(original_command: Optional[str]) -> bool:
    from main.train.arch_surface import argv_umbrella
    if not original_command:
        return False
    try:
        return argv_umbrella(shlex.split(original_command)) == "production"
    except ValueError:
        return False


def _model_version_fields() -> FrozenSet[str]:
    from agents.model.model_version import ModelVersion
    return frozenset(f.name for f in dataclasses.fields(ModelVersion))


def restart_route(dest: str, parser_default: Any, fields: FrozenSet[str]) -> str:
    """WHERE a same-run restart takes an untyped row from — exactly one route per row:
    ``inert`` (SB3 restores it; never re-applied), ``resume`` (a recorded tri-state field —
    `_resolve` / `resolve_critic_mode` inherit it), ``model_config`` (a recorded field with a
    concrete default — value-checked, so its default would FATAL) or ``cli_args`` (recorded nowhere
    else)."""
    if dest in INERT_ON_RESUME:
        return "inert"
    if dest in fields:
        return "resume" if parser_default is None else "model_config"
    return "cli_args"


def inherit_on_restart(ns: Any, run_dir: Optional[str], saved_ver: Any = None,
                       parser_defaults: Optional[Dict[str, Any]] = None,
                       *, model: Optional[str] = None) -> List[Tuple[str, Any, str]]:
    """On a SAME-RUN restart of an `--arch production` run, resolve every untyped row by its one
    route (`restart_route`), ANNOUNCED through the return value `[(dest, value, source), …]`.

    `[]` unless `--model` is a checkpoint INSIDE `run_dir` (`fork_lr.is_same_run_checkpoint`) and
    the run's immutable `original_command` carried `--arch production`. A value MISSING from its
    route raises `RecipeRestartError` naming the flag — never a default. `model` overrides
    `ns.model` for a caller (`main.checkargs`) that resolved a relative `models/…` path."""
    from main.train.fork_lr import is_same_run_checkpoint
    model = model or getattr(ns, "model", None)
    if not model or not run_dir or not is_same_run_checkpoint(model, run_dir):
        return []
    meta_path = os.path.join(run_dir, "metadata.json")
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        meta = None
    if not isinstance(meta, dict) or not _is_production_launch(meta.get("original_command")):
        return []
    if parser_defaults is None:
        from main.train.parser import build_parser
        p = build_parser()
        parser_defaults = {r.dest: p.get_default(r.dest) for r in ROWS}
    fields = _model_version_fields()
    cli = meta.get("cli_args")
    typed = typed_dests(ns)
    out: List[Tuple[str, Any, str]] = []
    missing: List[str] = []
    for r in ROWS:
        if r.dest in typed:
            continue
        route = restart_route(r.dest, parser_defaults.get(r.dest), fields)
        if route in ("inert", "resume"):
            continue
        if route == "model_config":
            if saved_ver is None or not hasattr(saved_ver, r.dest):
                missing.append(f"{r.flag} (the checkpoint's model_config.json)")
                continue
            value, source = getattr(saved_ver, r.dest), "model_config.json"
        else:
            if not isinstance(cli, dict) or r.dest not in cli:
                if r.dest in COLLECTOR_ROWS:
                    continue               # untyped there too (a run from before the flag): derived
                if r.dest == "env_core":
                    # a run recorded before `--env-core` existed ran the only core there was
                    setattr(ns, r.dest, "python")
                    out.append((r.dest, "python", "predates --env-core"))
                    continue
                missing.append(f"{r.flag} ({meta_path}:cli_args)")
                continue
            value, source = cli[r.dest], "metadata.json:cli_args"
        setattr(ns, r.dest, value)
        out.append((r.dest, value, source))
    if missing:
        raise RecipeRestartError(
            "a same-run RESTART of an --arch production run (the restart strips the fresh-only "
            "--arch) cannot find the recipe value it trained with for: " + ", ".join(missing)
            + ". Pass each flag with the value the run trained with — a training recipe value is "
              "never guessed, and a parser default would silently change the run mid-flight.")
    ns._recipe_restart_inherited = tuple(out)
    # The PROVENANCE tag rides with the knobs: `recipe_source` is stamped only by the `--arch` branch
    # the restart stripped, and `cli_args` is re-recorded from the namespace at every save, so the
    # first restart used to drop it. Kept from the run's own record; absent there (a run launched
    # before K10(a)) it stays unset — a provenance string gates nothing, so nothing is refused.
    if not getattr(ns, "recipe_source", None) and isinstance(cli, dict) and cli.get("recipe_source"):
        ns.recipe_source = cli["recipe_source"]
    return out


# ------------------------------------------------------------------------------- the comparison
class RecipeDiff(NamedTuple):
    dest: str
    flag: str
    resolved: Any
    production: Any
    #: "argv" (TYPED — a deliberate deviation), "default" (the silent kind), "restart" (resolved
    #: by `inherit_on_restart`), "inherited" (a fork's value from its parent's config) or "paired"
    #: (`--gamma` following a TYPED `--critic` — the deviation is the critic's, already typed).
    source: str

    def line(self) -> str:
        how = {"argv": "TYPED", "default": "untyped default", "restart": "restored at restart",
               "inherited": "inherited", "paired": "paired with the TYPED --critic"
               }.get(self.source, self.source)
        return f"{self.dest:<24} {self.resolved!r:<10} ({how})   production: {self.production!r}"


class RecipeReport(NamedTuple):
    diffs: Tuple[RecipeDiff, ...]
    fresh: bool
    allowed: bool
    umbrella: Optional[str]
    source_tag: str
    advisory: bool = False
    #: "fresh" or "fork" — which recipe the argv was compared with.
    kind: str = FRESH_KEY
    restart_inherited: Tuple[Tuple[str, Any, str], ...] = ()

    @property
    def silent(self) -> Tuple[RecipeDiff, ...]:
        return tuple(d for d in self.diffs if d.source == "default")

    @property
    def refuses(self) -> bool:
        """A FRESH argv with an UNTYPED difference REFUSES, unless consented to or pinned elsewhere
        (the mirror is THIS tree's — the ARCH surface's `advisory` rule)."""
        return bool(self.silent) and self.fresh and not self.allowed and not self.advisory


def _parser_defaults() -> Dict[str, Any]:
    from main.train.parser import build_parser
    p = build_parser()
    return {r.dest: p.get_default(r.dest) for r in ALL_ROWS}


def _resolved(r: RecipeRow, ns: Any, defaults: Dict[str, Any]) -> Any:
    """The value a launch trains with. A CONCRETE-default row always holds a value on a parsed
    namespace, so a `None` there is REAL (`--clip-range-vf none`); only a tri-state row's `None`
    means "unset", and resolves to what `resolve_config` fills (`RecipeRow.unset`)."""
    if r.unset == PARSER_DEFAULT:
        return getattr(ns, r.dest, defaults.get(r.dest))
    v = getattr(ns, r.dest, None)
    if v is None and r.unset == CRITIC_PAIRED:
        from agents.model.critic_mode import critic_gamma
        return critic_gamma(_resolved(_ROW["critic"], ns, defaults))
    return r.unset if v is None else v


def _agree(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return bool(a == b)


def diff_against_production(ns: Any, mirror: Optional[Dict[str, Any]] = None,
                            *, kind: str = FRESH_KEY,
                            inherited: FrozenSet[str] = frozenset()) -> List[RecipeDiff]:
    want = production_recipe(mirror, kind=kind)
    rows = ALL_ROWS if kind == FORK_KEY else ROWS
    defaults = _parser_defaults()
    typed = typed_dests(ns)
    restored = {d for d, *_ in getattr(ns, "_recipe_restart_inherited", ()) or ()}
    from agents.model.critic_mode import critic_gamma
    #: the discount a TYPED `--critic` pairs with (None when the critic was not typed)
    typed_critic_gamma = (critic_gamma(_resolved(_ROW["critic"], ns, defaults))
                          if "critic" in typed else None)
    out: List[RecipeDiff] = []
    core = _resolved(_ROW["env_core"], ns, defaults)
    for r in rows:
        if r.dest in COLLECTOR_ROWS and (want[r.dest] is None or core != "rust"):
            continue                       # derived by the collector / inert on the python core
        have = _resolved(r, ns, defaults)
        if _agree(have, want[r.dest]):
            continue
        src = ("argv" if r.dest in typed else "restart" if r.dest in restored
               else "paired" if (r.dest == "gamma" and typed_critic_gamma is not None
                                 and _agree(have, typed_critic_gamma))
               else "inherited" if r.dest in inherited else "default")
        out.append(RecipeDiff(r.dest, r.flag, have, want[r.dest], src))
    return out


def report(ns: Any, *, fresh: bool, restart: bool = False, allowed: bool = False,
           umbrella: Optional[str] = None, advisory: bool = False,
           mirror: Optional[Dict[str, Any]] = None,
           inherited: FrozenSet[str] = frozenset()) -> RecipeReport:
    """THE one entry point — `main.checkargs`, `--dry-run`, the launcher and `resolve_config`.
    A fresh argv, or a restart of an `--arch production` run, is compared with ``recipe.fresh``;
    any other `--model` argv (a fork, or a restart of one) with ``recipe.fork``."""
    restored = tuple(getattr(ns, "_recipe_restart_inherited", ()) or ())
    kind = FRESH_KEY if (fresh or (restart and restored)) else FORK_KEY
    return RecipeReport(
        diffs=tuple(diff_against_production(ns, mirror, kind=kind, inherited=inherited)),
        fresh=bool(fresh), allowed=bool(allowed), umbrella=umbrella, source_tag=source_tag(),
        advisory=bool(advisory), kind=kind, restart_inherited=restored)


def report_for_child_argv(child_args: Sequence[str], *, advisory: bool = False) -> Optional[RecipeReport]:
    """`report()` for a CHILD ARGV (the launcher's gate); `None` when the argv does not parse."""
    from main.checkargs import check
    try:
        return check(list(child_args), advisory=advisory).get("recipe")
    except Exception:                                # noqa: BLE001 — a guard never crashes a launch
        return None


def report_lines(rep: RecipeReport) -> List[str]:
    """The printed block, identical on every surface."""
    out = [f"RECIPE SURFACE vs designs/production_config.json recipe.{rep.kind}  [{rep.source_tag}]"]
    if rep.umbrella:
        out.append(f"  --arch {rep.umbrella} applied recipe.fresh + recipe.{SIZING_KEY} (every knob this "
                   "argv did not type; explicit flags still win)")
    try:
        sz = sizing_block()
        verdict = sz.get(VERDICT_KEY)
        out.append(f"  sizing      : env core {sz['env_core']}, N {sz['n_envs']}, n_steps max {sz['n_steps']} — "
                   + (f"verdict {verdict}" if verdict else
                      "verdict PENDING (N* not set: the pre-sizing N = 48 shape)"))
    except Exception:                                # noqa: BLE001 — a report line never breaks a launch
        pass
    if rep.restart_inherited:
        out.append("  ♻️  same-run RESTART of an --arch production run (--arch stripped): "
                   "resolved from the run's own record — "
                   + "  ".join(f"{d}={v!r} [{s}]" for d, v, s in rep.restart_inherited))
        out.append("      (--lr / --batch-size / --n-steps / --gamma are INERT on a resume — SB3 "
                   "restores them — and are never re-applied here)")
    n = len(ROWS) if rep.kind == FRESH_KEY else len(ALL_ROWS)
    if not rep.diffs:
        out.append(f"  ✓ every RECIPE knob matches recipe.{rep.kind} ({n} knobs)")
        return out
    verb = "differ" if len(rep.diffs) != 1 else "differs"
    out.append(f"  {len(rep.diffs)} of {n} RECIPE knob(s) {verb} from recipe.{rep.kind}:")
    out += [f"      {d.line()}" for d in rep.diffs]
    if not rep.fresh:
        out.append("  ℹ️  INFO only — a FORK/RESTART. A fork's recipe is its ARGV plus what it "
                   "inherits from its parent's model_config.json; a same-run restart of an "
                   "--arch production run resolves its knobs from the run's own record (above).")
        return out
    if not rep.silent:
        out.append("  ℹ️  every differing knob was TYPED — a deliberate deviation, recorded in "
                   "metadata.json's cli_args.")
        return out
    if rep.allowed:
        out.append(f"  ℹ️  {ALLOW_FLAG} — the untyped drift above is an EXPLICIT choice.")
        return out
    if rep.advisory:
        out.append("  ℹ️  ADVISORY — the child is PINNED to another commit, whose own mirror is the "
                   "authority. Reported, not gated.")
        return out
    names = ", ".join(d.flag for d in rep.silent)
    out += [
        f"  ✗ REFUSED: a FRESH run that silently trains a non-production recipe — UNTYPED: {names}.",
        "     Those values are the PARSER's defaults, not a choice (design_learner_recipe.md §3.22:",
        "     the recipe-side twin of the 2026-09-06 stripped-architecture incident).",
        "     FIX — either:",
        "       * pass `--arch production` (fills every untyped knob from recipe.fresh), or",
        "       * TYPE the value you mean (a typed value is a deliberate deviation), or",
        f"       * pass `{ALLOW_FLAG}` to assert the untyped drift is deliberate.",
    ]
    return out
