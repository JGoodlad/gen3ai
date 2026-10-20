"""THE RECIPE SURFACE — "is this the TRAINING RECIPE you meant?", the ARCH SURFACE's twin (K10(a)).

**WHY IT EXISTS.** `--arch production` applied the production ARCHITECTURE and nothing else: five
parser defaults differed from the live recipe (`--n-envs` 32 vs 48, `--batch-size` 4096 vs 2048,
`--n-epochs` 5 vs 10, `--ent-coef` 0.02 vs 0.05, `--clip-range-vf` 0.5 vs none), and so did
`--grad-accum-steps` (1 vs 32 — an effective batch of 4,096 instead of 65,536), `--self-play`,
the critic (`shaped` vs `winprob`) with its three REQUIRED reward values and its discount (those
five are CONSTANTS of the namespace now, `parser/objective.py` — deletion pass P11b), and the supervision
doses. A fresh argv that omitted them parsed, resolved, dry-ran and launched — and trained a
different recipe: the recipe-side twin of the 2026-09-06 stripped-architecture incident
(`arch_surface`'s docstring), closed the same way — a DECLARED surface, a mirror, one applier, one
report, read by every surface that launches.

**WHERE THE VALUES LIVE** — `designs/production_config.json`'s nested ``recipe`` block (stripped from
`agents.training.baselines.production_config()`, so no arch consumer ever sees it):

  * ``recipe.fresh`` — N0's MEASURED fresh recipe (`models/ai_v14_01_base`, the lineage's fresh
    launch), every knob it launched with, including its reward values, plus the one knob ADOPTED since:
    ``trunk_precision`` ``"bf16"`` (the bf16 strength A/B PASSED, 2026-10-14; ``--trunk-precision fp32``
    is the explicit opt-out), and one knob RECORDED since: ``overlap_collect`` ``false`` (X14 stays OFF; a row so a
    resume or fork inherits it, core review 2026-10-20 C-2), plus
    ``kl_controller``: the controller constants it ran with (not flags — pinned against the
    callbacks' own defaults by `recipe_surface_test`). A key that is ALSO a recorded top-level
    mirror field must EQUAL it (`production_recipe` refuses otherwise): one truth, two readers.
  * ``recipe.sizing`` — THE ONE PLACE the M5 SIZING study's verdict fills (`SIZING_ROWS`): N, the n_steps
    maximum, the collector's update size and T2's slots / buckets / lanes, plus ``verdict`` (null = N*
    PENDING). The env core is not a row: `rust` is the only core (deletion pass P11b deleted
    `--env-core`). `recipe_blocks` merges it into the fresh recipe, so every reader treats a sizing row
    like any other row; a collector-only row (`COLLECTOR_ROWS`) whose value is null is left untyped.
  * ``recipe.fork`` — what a generalist FORK changes: the E5 verdict (5 epochs at a FROZEN 5.6e-5,
    the same dose as 10 at 2.8e-5). `--fork-lr` is refused on a fresh run, so `--arch production`
    never applies this; a fork's argv is compared with it as INFO. 🚨 E5's 5 epochs are NEVER paired
    with the fresh LR — that pairing was never measured (design_learner_recipe.md Decision record).

Sources of every value: `designs/endstate/design_learner_recipe.md` §3.22; the doc and the block are
held together by `src/recipe_doc_gate_test.py`.

**WHAT `--arch production` DOES.** Writes every ``recipe.fresh`` row the argv did not TYPE, as if
typed, after the ARCH surface and before `resolve_critic_mode` (which then implies the rest of the
win-prob critic's settings). "Typed" is a fact the parser records (`record_typed_recipe_flags`).

**THE REFUSAL.** A FRESH argv that differs on an UNTYPED knob refuses (`checkargs`, `--dry-run`, the
launcher); a TYPED difference is the arm's lever (INFO); `--allow-nonproduction-recipe` consents.

**EVERY `--model` LAUNCH** (`inherit_recorded_recipe`) — a same-run restart (a launcher restart strips the
fresh-only `--arch`), a FORK, a fork's own resume — resolves every UNTYPED row by exactly one route from the
record of the run that WROTE the checkpoint (the run itself on a restart, the PARENT on a fork):

  1. `--lr`, `--batch-size`, `--n-steps`: INERT on a resume (SB3 restores the
     checkpoint's own values) — never re-applied here, whatever the argv says;
  2. a recorded tri-state field (the doses, `opp_intent_coef` from config v125,
     `policy_gae_lambda`, `trunk_precision`, …): the general resume mechanism (`_resolve` /
     `config.inherit_derived_enable_coefs`, `68850f27`) inherits it — untouched here;
  3. a recorded field with a concrete parser default (`vf_coef` — value-CHECKED, so the default would
     FATAL): from the checkpoint's `model_config.json`, ANNOUNCED;
  4. a knob recorded nowhere else (`n_envs`, `n_epochs`, `ent_coef`, `self_play`, …): from the source
     run's `metadata.json:cli_args`, ANNOUNCED.

A value MISSING from its route (no `cli_args`, no key, no field) REFUSES by name
(`RecipeRecordError`, `FATAL_CONFIG`) — never a parser or registry default. Until 2026-10-14 this ran only on
a same-run restart of an `--arch production` run, and a fork (or a manual resume of one) trained the PARSER's
recipe (`--n-envs` 32, `--self-play` False, `--grad-accum-steps` 1, …) — the class this closes. The LAUNCHER
writes the restored rows into a `--model` child's argv as TYPED tokens (`materialize_recorded_recipe`), so a
child pinned to a commit from before the fix trains the same recipe. ``recipe.fork`` is never applied: a
fork's recipe is its argv plus its parent's record, compared with ``recipe.fork`` as INFO.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
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
#: INERT on a resume — SB3 restores the checkpoint's own values (root CLAUDE.md). Never re-applied
#: on a restart, so nothing here fights that restoration.
INERT_ON_RESUME = frozenset({"lr", "batch_size", "n_steps"})


class RecipeError(ValueError):
    """The mirror and the declared rows disagree — a rename, a dropped block, a stray key, or a
    block value that contradicts the same recorded field at the mirror's top level."""


class RecipeRecordError(ValueError):
    """A `--model` launch (a same-run restart, a fork, a fork's resume) whose untyped recipe value cannot be
    found in the record it must come from. Re-running cannot change it — `FATAL_CONFIG`."""


class RecipeRow(NamedTuple):
    dest: str
    flag: str
    #: What an UNTYPED value resolves to on a FRESH run: `PARSER_DEFAULT`, or — for a tri-state
    #: (`default=None`) flag — the value `resolve_config` fills (pinned against `config.py`).
    unset: Any
    note: str


#: ``recipe.sizing`` — the run's declared SIZES (`SIZING_KEY`). Order is the report's.
SIZING_ROWS: Tuple[RecipeRow, ...] = (
    RecipeRow("n_envs", "--n-envs", PARSER_DEFAULT, "§1 row 1 — the SIZING study's to change"),
    RecipeRow("n_steps", "--n-steps", PARSER_DEFAULT, "§1 row 2 — the SIZING study's / Lane G's (the "
              "rust core's n_steps MAXIMUM: the buffer is preallocated at it)"),
    RecipeRow("rollout_target_samples", "--rollout-target-samples", 0,
              "the collector's update size; null / 0 = n_envs x n_steps"),
    RecipeRow("trainee_slots", "--trainee-slots", 1, "T2's trainee slots; null = the collector's (1)"),
    RecipeRow("t2_buckets", "--t2-buckets", None, "T2's opponent buckets; null = derived at startup"),
    RecipeRow("t2_lanes", "--t2-lanes", 0, "T2's lanes; null / 0 = derived (min(slots, 8))"),
)
#: The sizing rows the Rust collector alone reads: a ``null`` value means "the collector derives it" and
#: the row is left UNTYPED.
COLLECTOR_ROWS = frozenset({"rollout_target_samples", "trainee_slots", "t2_buckets", "t2_lanes"})
#: The collector rows that size the INFERENCE SERVICE only (T2's slots / buckets / lanes) — derived at startup when
#: unrecorded; they change throughput, never the rows a learner update sees.
SERVING_ROWS = frozenset({"trainee_slots", "t2_buckets", "t2_lanes"})

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
    RecipeRow("policy_gae_lambda", "--policy-gae-lambda", 0.80, "§1 row 13"),
    RecipeRow("self_play", "--self-play", PARSER_DEFAULT, "§1 row 22"),
    RecipeRow("vf_coef", "--vf-coef", PARSER_DEFAULT, "§1 row 14"),
    RecipeRow("opp_belief_aux_coef", "--opp-belief-aux-coef", 0.0, "§1 row 16"),
    RecipeRow("opp_intent_coef", "--opp-intent-coef", 0.0, "§1 row 16"),
    RecipeRow("move_belief_coef", "--move-belief-coef", 0.0, "§1 row 16"),
    RecipeRow("move_belief_latent_coef", "--move-belief-latent-coef", 0.0, "§1 row 16"),
    RecipeRow("spread_belief_coef", "--spread-belief-coef", 0.0, "§1 row 16"),
    RecipeRow("hp_type_belief_coef", "--hp-type-belief-coef", 0.05, "§1 row 16"),
    RecipeRow("item_belief_coef", "--item-belief-coef", 0.05, "§1 row 16"),
    RecipeRow("intent_label_bot_weight", "--intent-label-bot-weight", 1.0, "§1 row 16"),
    # The learner's trunk-round compute precision (gen3_trunk_precision_flag_v1, config v155): ADOPTED into the
    # fresh recipe 2026-10-14 (the bf16 strength A/B PASSED). A tri-state flag (`default=None`), so an UNTYPED
    # value resolves to "fp32" (`config.resolve_config`) and a flagless FORK / restart inherits its parent's
    # recorded value (`_resolve`) — recipe.fork deliberately carries no row for it (§3.22 "Left out").
    RecipeRow("trunk_precision", "--trunk-precision", "fp32", "§1 row 25 — bf16 trunk rounds (A/B PASS 2026-10-14)"),
    # X14 (`--overlap-collect`, gen3_overlap_collect_v1): it changes the OBJECTIVE (one-version-stale rows, decoupled
    # PPO), so a resume or a fork inherits it from the record like any knob, and `checkargs` shows it (core review
    # 2026-10-20 C-2: it was no row, and a fork of an ON arm silently trained serial PPO). A tri-state flag resolved
    # by `rust_env_setup.resolve_env_core_args` (False when untyped). Absent from a record, the run PREDATES the flag
    # (`PRE_FLAG_ROWS`): it trained OFF, which is what the unset flag resolves to.
    RecipeRow("overlap_collect", "--overlap-collect", False, "X14 — default OFF (stale rows, decoupled PPO)"),
)
#: Every row the fresh recipe declares — ``recipe.sizing`` first, then ``recipe.fresh``.
ROWS: Tuple[RecipeRow, ...] = SIZING_ROWS + FRESH_ROWS

#: ``recipe.fork`` — the FORK-only rows; a fork's other knobs are the fresh values it inherits
#: from the recipe unless ``recipe.fork`` overrides them (today: `n_epochs`).
FORK_ROWS: Tuple[RecipeRow, ...] = (
    RecipeRow("fork_lr", "--fork-lr", None, "§1 row 9 — E5: a frozen 5.6e-5"),
    RecipeRow("fork_lr_freeze", "--fork-lr-freeze", PARSER_DEFAULT, "§1 rows 8-9"),
)
#: Rows a record may LACK because the source run predates the flag: a `cli_args` dict without the key means the run
#: trained with the flag's unset value, so the untyped flag is left to resolve to it — never refused, never
#: materialized (a child pinned before the flag would not parse it). A run with NO `cli_args` still refuses.
PRE_FLAG_ROWS = frozenset({"overlap_collect"})

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
    want = production_recipe(mirror)
    typed = typed_dests(ns)
    applied: List[Tuple[str, Any]] = []
    for r in ROWS:
        if r.dest in typed:
            continue
        if r.dest in COLLECTOR_ROWS and want[r.dest] is None:
            continue                       # derived by the collector
        value = want[r.dest]
        setattr(ns, r.dest, value)
        applied.append((r.dest, value))
    ns.recipe_source = source_tag()
    return applied


# ---------------------------------------------------------------- the record path (every --model launch)
def _model_version_fields() -> FrozenSet[str]:
    from agents.model.model_version import ModelVersion
    return frozenset(f.name for f in dataclasses.fields(ModelVersion))


def restart_route(dest: str, parser_default: Any, fields: FrozenSet[str]) -> str:
    """WHERE a `--model` launch (a same-run restart, a fork, a fork's own resume) takes an UNTYPED row
    from — exactly one route per row: ``inert`` (SB3 restores it from the zip; never re-applied),
    ``resume`` (a recorded tri-state field — `_resolve` inherits it from the checkpoint's
    `model_config.json`), ``model_config`` (a recorded field with a concrete default — value-checked,
    so its default would FATAL) or ``cli_args`` (recorded nowhere else: the source run's
    `metadata.json:cli_args`)."""
    if dest in INERT_ON_RESUME:
        return "inert"
    if dest in fields:
        return "resume" if parser_default is None else "model_config"
    return "cli_args"


class RecordSource(NamedTuple):
    """Where a `--model` launch's recipe record lives: the run that WROTE the checkpoint."""
    role: str                 # "restart" (the checkpoint is this run's own) or "fork" (another run's)
    run_dir: Optional[str]    # the checkpoint's run dir (`agents.training.lineage.run_dir_of`)
    meta_path: Optional[str]
    meta: Optional[Dict[str, Any]]


def record_source(model: str, run_dir: Optional[str]) -> RecordSource:
    """The run whose record a `--model` launch inherits from — ONE predicate for the role
    (`fork_lr.is_same_run_checkpoint`) and ONE resolver for the checkpoint's run
    (`agents.training.lineage.run_dir_of`, `load_model_snapshot`'s own convention)."""
    from agents.training.lineage import run_dir_of
    from main.train.fork_lr import is_same_run_checkpoint
    same = bool(run_dir) and is_same_run_checkpoint(model, str(run_dir))
    src = str(run_dir) if same else run_dir_of(model)
    meta_path = os.path.join(src, "metadata.json") if src else None
    meta: Optional[Dict[str, Any]] = None
    if meta_path:
        try:
            with open(meta_path) as fh:
                loaded = json.load(fh)
            meta = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            meta = None
    return RecordSource("restart" if same else "fork", src, meta_path, meta)


def inherit_recorded_recipe(ns: Any, run_dir: Optional[str], saved_ver: Any = None,
                            parser_defaults: Optional[Dict[str, Any]] = None,
                            *, model: Optional[str] = None,
                            partial: bool = False) -> List[Tuple[str, Any, str]]:
    """EVERY `--model` launch — a same-run restart, a FORK, a fork's own resume — resolves every UNTYPED
    recipe row by its one route (`restart_route`) from the record of the run that WROTE the checkpoint
    (`record_source`), ANNOUNCED through the return value `[(dest, value, source), …]`.

    THE CLASS IT CLOSES (2026-10-14, P1 GIGO). This used to run only on a same-run restart of a run whose
    `original_command` carried `--arch production`; a FORK (`--model <parent ckpt> --fork-lr …`) and a manual
    resume of a fork took every untyped recipe knob from the PARSER — `--n-envs` 32 (parent 256),
    `--grad-accum-steps` 1 (32), `--ent-coef` 0.02 (0.05), `--self-play` False (True),
    `--rollout-target-samples` 0 (98304), `--clip-range-vf` 0.5 (none) — against the documented contract
    ("with `--model`, every flag you do not name is INHERITED"). Now a fork trains its PARENT's recipe and a
    resume its OWN, unless a knob is TYPED (`recipe.fork` stays an INFO comparison, never applied).

    A value MISSING from its route (no `metadata.json`, no `cli_args`, no key, no config field) raises
    `RecipeRecordError` naming the flag — never a parser or registry default. `[]` without a `--model`.
    `model` overrides `ns.model` for a caller (`main.checkargs`) that resolved a relative `models/…` path.

    `partial=True` is for a reader that could not parse the checkpoint's `model_config.json` under THIS tree
    (a run pinned to an older commit: `saved_ver` is `None`) and for the launcher's materialization: the
    `model_config`-route rows are SKIPPED rather than reported missing (the child reads the config itself,
    and a wrong `vf_coef` FATALs there), every other row is still restored."""
    model = model or getattr(ns, "model", None)
    if not model:
        return []
    src = record_source(model, run_dir)
    if parser_defaults is None:
        from main.train.parser import build_parser
        p = build_parser()
        parser_defaults = {r.dest: p.get_default(r.dest) for r in ROWS}
    fields = _model_version_fields()
    meta = src.meta
    cli = meta.get("cli_args") if isinstance(meta, dict) else None
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
            if saved_ver is None and partial:
                continue
            if saved_ver is None or not hasattr(saved_ver, r.dest):
                missing.append(f"{r.flag} (the checkpoint's model_config.json)")
                continue
            value, source = getattr(saved_ver, r.dest), "model_config.json"
        else:
            if not isinstance(cli, dict) or r.dest not in cli:
                if r.dest in PRE_FLAG_ROWS and isinstance(cli, dict):
                    continue                       # the run predates the flag: it trained the unset value
                if r.dest in COLLECTOR_ROWS and (isinstance(cli, dict) or r.dest in SERVING_ROWS):
                    # untyped there too (a run from before the flag): derived. The SERVING sizes are derived at
                    # startup whatever the record says; the update size (`rollout_target_samples`) is not, so
                    # with NO record at all it is named below like any training knob.
                    continue
                missing.append(f"{r.flag} ({src.meta_path or 'no run dir'}:cli_args)")
                continue
            value, source = cli[r.dest], f"{'parent ' if src.role == 'fork' else ''}metadata.json:cli_args"
        setattr(ns, r.dest, value)
        out.append((r.dest, value, source))
    if missing:
        who = "its own run" if src.role == "restart" else f"its PARENT run ({src.run_dir})"
        raise RecipeRecordError(
            f"a --model launch ({src.role}: {model}) cannot find the recipe value {who} trained with "
            "for: " + ", ".join(missing)
            + ". Pass each flag with the value you mean — a training recipe value is never guessed, and a "
              "parser default would silently train a different recipe (the 2026-10-14 fork-recipe GIGO class).")
    ns._recipe_record_inherited = tuple(out)
    ns._recipe_record_role = src.role
    # The PROVENANCE tag rides with the knobs: `recipe_source` is stamped only by the `--arch` branch (fresh-only),
    # and `cli_args` is re-recorded from the namespace at every save, so a first restart used to drop it. Kept from
    # the source run's record (a fork's knobs ARE its parent's); absent there it stays unset — it gates nothing.
    if not getattr(ns, "recipe_source", None) and isinstance(cli, dict) and cli.get("recipe_source"):
        ns.recipe_source = cli["recipe_source"]
    return out


def _token(action: Any, value: Any) -> Optional[str]:
    """`value` spelled for `action`'s parser, or None when it cannot be spelled (a None the type cannot parse)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none" if getattr(getattr(action, "type", None), "__name__", "") == "optional_float" else None
    return str(value)


def materialize_recorded_recipe(child_args: Sequence[str], run_dir: Optional[str]
                                ) -> Tuple[List[str], List[Tuple[str, Any, str]]]:
    """THE LAUNCHER'S HALF of `inherit_recorded_recipe`: the child argv with every UNTYPED `cli_args`-route
    recipe row of a `--model` launch appended as a TYPED `--flag value`, plus the rows it added.

    Why the launcher writes them rather than trusting the child: a resume or fork runs PINNED, by default to
    the checkpoint's own commit, and every commit before this fix resolves an untyped knob to the PARSER
    default. A typed token is honoured by every trainer, so the recipe a `--model` launch trains is decided
    HERE, by this tree's one resolver, whatever the pinned child's age. INERT rows (`--lr`, `--batch-size`,
    `--n-steps`; SB3 restores them) and `model_config`-route rows (the child's own value check) are never
    added; a fresh argv, or a `--model` launch with nothing untyped to restore, comes back unchanged.
    Raises `RecipeRecordError` naming every knob it cannot resolve (`FATAL_CONFIG` at the launcher)."""
    from main.train.parser import build_parser
    args = list(child_args)
    p = build_parser()
    ns, _rest = p.parse_known_args(args)
    if not getattr(ns, "model", None):
        return args, []
    restored = inherit_recorded_recipe(ns, run_dir, None, model=ns.model, partial=True)
    by_dest = {a.dest: a for a in p._actions}
    added: List[Tuple[str, Any, str]] = []
    unspellable: List[str] = []
    for dest, value, source in restored:
        r = _ROW[dest]
        if value is None and p.get_default(dest) is None:
            continue                       # the parser's own None: nothing to write
        tok = _token(by_dest.get(dest), value)
        if tok is None:
            unspellable.append(f"{r.flag} (recorded {value!r})")
            continue
        args += [r.flag, tok]
        added.append((dest, value, source))
    if unspellable:
        raise RecipeRecordError("the recorded recipe value cannot be passed to the child on its command line for: "
                                + ", ".join(unspellable) + ". Pass each flag with the value you mean.")
    return args, added


# ------------------------------------------------------------------------------- the comparison
class RecipeDiff(NamedTuple):
    dest: str
    flag: str
    resolved: Any
    production: Any
    #: "argv" (TYPED — a deliberate deviation), "default" (the silent kind), "debug" (the
    #: `--debug` smoke shape, `main.train.debug_shape`), "recorded" (restored from the source run's
    #: record by `inherit_recorded_recipe`), "inherited" (a fork's value from its parent's config) or "inert"
    #: (an untyped `--lr` / `--batch-size` / `--n-steps` on a `--model` launch: the zip's value trains, not this).
    source: str

    def line(self) -> str:
        how = {"argv": "TYPED", "default": "untyped default", "recorded": "restored from the record",
               "inert": "INERT on a resume: SB3 restores the checkpoint's own",
               "debug": "--debug smoke shape",
               "inherited": "inherited"}.get(self.source, self.source)
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
    #: The untyped rows `inherit_recorded_recipe` restored, `(dest, value, source)`, and the launch's role.
    record_inherited: Tuple[Tuple[str, Any, str], ...] = ()
    record_role: Optional[str] = None
    #: The trunk-round precision this argv RESOLVES to (``"bf16"`` / ``"fp32"``): printed on every surface, so the
    #: adopted bf16 default is visible in the report and not only as a refusal when it is missing.
    trunk_precision: Optional[str] = None

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
    restored = {d for d, *_ in getattr(ns, "_recipe_record_inherited", ()) or ()}
    from main.train.debug_shape import OVERRIDE_ATTR
    debug = frozenset(getattr(ns, OVERRIDE_ATTR, None) or ())
    out: List[RecipeDiff] = []
    for r in rows:
        if r.dest in COLLECTOR_ROWS and want[r.dest] is None:
            continue                       # derived by the collector
        have = _resolved(r, ns, defaults)
        if _agree(have, want[r.dest]):
            continue
        src = ("argv" if r.dest in typed else "debug" if r.dest in debug
               else "recorded" if r.dest in restored
               else "inherited" if r.dest in inherited
               else "inert" if (r.dest in INERT_ON_RESUME and getattr(ns, "model", None)) else "default")
        out.append(RecipeDiff(r.dest, r.flag, have, want[r.dest], src))
    return out


def report(ns: Any, *, fresh: bool, restart: bool = False, allowed: bool = False,
           umbrella: Optional[str] = None, advisory: bool = False,
           mirror: Optional[Dict[str, Any]] = None,
           inherited: FrozenSet[str] = frozenset()) -> RecipeReport:
    """THE one entry point — `main.checkargs`, `--dry-run`, the launcher and `resolve_config`.
    A fresh argv, or a same-run restart (its recipe IS the run's own), is compared with ``recipe.fresh``;
    a fork, or a fork's own resume, with ``recipe.fork``."""
    restored = tuple(getattr(ns, "_recipe_record_inherited", ()) or ())
    role = getattr(ns, "_recipe_record_role", None)
    kind = FRESH_KEY if (fresh or (restart and role == "restart" and restored)) else FORK_KEY
    return RecipeReport(
        diffs=tuple(diff_against_production(ns, mirror, kind=kind, inherited=inherited)),
        fresh=bool(fresh), allowed=bool(allowed), umbrella=umbrella, source_tag=source_tag(),
        advisory=bool(advisory), kind=kind, record_inherited=restored, record_role=role,
        trunk_precision=str(_resolved(_ROW["trunk_precision"], ns, _parser_defaults())))


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
        out.append(f"  sizing      : N {sz['n_envs']}, n_steps max {sz['n_steps']} — "
                   + (f"verdict {verdict}" if verdict else
                      "verdict PENDING (N* not set: the pre-sizing N = 48 shape)"))
    except Exception:                                # noqa: BLE001 — a report line never breaks a launch
        pass
    if rep.trunk_precision is not None:
        if rep.trunk_precision == "bf16":
            why = ("the learner's trunk rounds run under bf16 autocast in R1 (T2 / eval / the K9(b) probe stay "
                   "fp32); `--trunk-precision fp32` is the explicit opt-out")
        elif rep.fresh:
            why = "the learner's trunk rounds run in fp32 — NOT the production recipe (bf16); a TYPED fp32 is the opt-out"
        else:
            why = ("the learner's trunk rounds run in fp32 — a fork / restart KEEPS its parent's recorded "
                   "precision; type `--trunk-precision bf16` to train it at the production precision")
        out.append(f"  trunk       : {rep.trunk_precision} — {why}")
    if rep.record_inherited:
        whose = "the PARENT run's" if rep.record_role == "fork" else "the run's own"
        out.append(f"  ♻️  --model launch ({rep.record_role or '?'}): every untyped recipe knob restored from "
                   f"{whose} record — "
                   + "  ".join(f"{d}={v!r} [{s}]" for d, v, s in rep.record_inherited))
        out.append("      (--lr / --batch-size / --n-steps are INERT on a resume — SB3 "
                   "restores them — and are never re-applied here)")
    n = len(ROWS) if rep.kind == FRESH_KEY else len(ALL_ROWS)
    if not rep.diffs:
        out.append(f"  ✓ every RECIPE knob matches recipe.{rep.kind} ({n} knobs)")
        return out
    verb = "differ" if len(rep.diffs) != 1 else "differs"
    out.append(f"  {len(rep.diffs)} of {n} RECIPE knob(s) {verb} from recipe.{rep.kind}:")
    out += [f"      {d.line()}" for d in rep.diffs]
    if not rep.fresh:
        out.append("  ℹ️  INFO only — a FORK/RESTART. Its recipe is its ARGV plus the record of the run that "
                   "wrote the checkpoint (a fork: its PARENT's model_config.json + metadata.json:cli_args; a "
                   "restart: its own) — recipe.fork is never applied.")
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
