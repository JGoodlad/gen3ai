"""THE ARCH SURFACE — "is this the architecture you MEANT?", asked at validation time.

**"IT LAUNCHES" AND "IT IS THE EXPERIMENT" ARE INDEPENDENT CHECKS, AND ONLY THE RESOLVED-CONFIG
DIFF TESTS THE SECOND.** That sentence is the whole module.

WHY IT EXISTS (2026-09-06 incident, ~7 GPU-hours; ledger `2026-09-06 · INCIDENT`). The first
win-prob-critic arm was launched from a 38-token argv copied out of a design document's command
block: the critic flags and the PPO knobs, and NONE of the production feature flags. Every
architecture flag silently took its OFF default. Three gates were run before launch and all three
passed — `python -m main.checkargs` exit 0, `resolve_config` accepted, `--dry-run` clean. All three
were RIGHT, because none of them was asked the second question. The run trained a near-bare network
for **25,131 s / 24.4M steps / 6 checkpoints**, and was still holding the GPU when it was discovered
a SECOND time. Measured on its own `model_config.json`: **31 keys differ from
`designs/production_config.json`** — every edge family off, zero entity seats, no belief slots, no
event window, no intent heads. Every number taken off that run measures a different model.
`arch_tables_test`'s drift gate would have gone red, but only for whoever next ran the suite.

The gap was not a missing check. It was a missing QUESTION. A launch-time answer is the only one
that arrives before the GPU-hours do.

🚨 **THIS IS NOT THE SAME FAILURE AS A REFUSED FLAG COMBINATION, AND THE TWO MUST NEVER SHARE A
MESSAGE OR A CLOSING LINE.** Rebuilding that same arm from an older generation's recorded
`original_command` also fails — on nine flags the win-prob critic SUBSUMES and therefore refuses
(`--use-popart`, `--value-from-dist`, the four `--value-dist-*`, `--value-dist-coef`,
`--win-prob-coef`, `--value-tail-weight`). That failure is **LOUD and PRE-launch**: `checkargs`
names it, nothing starts, the operator fixes it in a minute. Architecture drift is **SILENT and
POST-launch**: everything parses, the run starts, and seven GPU-hours later the config diff is the
only thing that would have told you. A guard that catches the first is no protection against the
second — so `combination_checks` and this module keep separate blocks, separate summary lines and
separate refusal paths on every surface that prints them.

WHAT IT COMPARES. The key set is DERIVED from `agents.model.flag_registry`
(`arch_surface_flags()`) — never hand-listed, because a hand list goes stale the first time a
toggle lands and then quietly under-reports. It is the `structural` × `family=arch` rows: the
toggles whose mismatch means a DIFFERENT NETWORK. Excluded, each by its own declaration rather than
by an exception here:

  * `training_coef`      — the belief-supervision doses and every other loss weight. NOT
                           architecture; see `unapplied_production_keys()`, which reports them so a
                           reader is not misled into thinking `--arch production` covered them.
  * `runtime`            — perf knobs, never recorded.
  * `resume_immutable`   — the forward is identical (`belief_grad_mode`, the value-dist bounds).
  * `family=critic`      — the readouts an experiment deliberately VARIES. `--critic winprob`
                           IMPLIES `win_prob_mode` and REFUSES `--value-dist-mode` /
                           `--value-from-dist`, so a guard that demanded these match production
                           would refuse every critic arm — the exact class of arm the incident was.

WHAT IT DOES WITH THE ANSWER. On a FRESH argv (no `--model`) a non-empty diff REFUSES, naming every
differing key with both values, unless the argv carries `--allow-nonproduction-arch`. On a FORK or a
RESTART it is INFO only: a resume INHERITS its parent's surface through `config.inherit_saved_flag`,
so the argv's silence there is not a bare architecture — it is the parent's.

`--arch production` is the other half: it writes the whole surface as if typed, from
`designs/production_config.json`, before `_resolve` — so an explicit flag still wins and the C1-class
inheritance rules are untouched. `arch_source_tag()` records WHICH mirror, by content hash, into
`model_config.json`.

ONE function, three readers. `main.checkargs`, `main.launcher.dry_run` and the launcher's
`_prepare_session` all call `report()`. Three copies of a guard is three things to keep in step, and
this project has paid for that shape twice already (the pre-`combination_checks` `parser.error`
lines; the pre-registry `ARCH_ARG_KEYS`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from agents.model.flag_registry import (
    REGISTRY,
    Klass,
    ModelFlag,
    Tier,
    arch_surface_flags,
    is_enabled,
)
from agents.training import baselines

#: The mirror every arm is judged against — resolved through the BASELINES REGISTRY
#: (`designs/baselines.json`'s `production` entry declares it as its `config_mirror`), never
#: hardcoded here. `arch_tables` / `delivery_graph` / the compile gate key on the same file, so the
#: guard cannot disagree with the generated architecture docs — or with the registry — about what
#: "production" is. Resolved at import; the registry is a committed file, not a live lookup.
PRODUCTION_CONFIG_PATH = baselines.production_config_path()

#: The umbrella flag's only value today. A string rather than a bool because the next mirror
#: (`--arch gen16`, a named archived config) is a value here, not a second flag.
UMBRELLA_VALUES = ("production",)

#: The consent flag. Launcher-recognised, FORWARDED to the child, and recorded — it lands in
#: `metadata.json`'s `cli_args` like every other flag, and additionally stamps `arch_source` in
#: `model_config.json`, so "we meant this" is a fact on disk rather than a memory.
ALLOW_FLAG = "--allow-nonproduction-arch"


class ArchDiff(NamedTuple):
    """One surface key on which the resolved config and the production mirror disagree."""

    name: str
    cli_flag: str
    resolved: Any
    production: Any
    #: Where the resolved value came from: "argv" (the operator typed it, or a desugar filled it)
    #: or "default" (nothing set it, so the registry's fresh-run default applies).
    source: str

    def line(self) -> str:
        return (f"{self.name:<28} {self.resolved!r:<14} "
                f"(this argv, {self.source})   production: {self.production!r}")


class ArchReport(NamedTuple):
    """The whole verdict — what differs, whether that refuses, and the lines to print."""

    diffs: Tuple[ArchDiff, ...]
    #: True on a FRESH argv (no `--model`); a fork/restart inherits its parent's surface.
    fresh: bool
    #: Did the argv carry `--allow-nonproduction-arch`?
    allowed: bool
    #: Did the argv carry `--arch production`?
    umbrella: Optional[str]
    #: The mirror's content hash, for the record.
    source_tag: str
    #: Is the child pinned to a commit OTHER than this tree's HEAD? Then the comparison is
    #: informational — see `refuses`.
    advisory: bool = False

    @property
    def refuses(self) -> bool:
        """A FRESH argv that drifts from production and did not say so REFUSES.

        Except when `advisory`: `designs/production_config.json` is THIS tree's mirror, and a run
        pinned to another commit is built by that commit's registry, its flags and its mirror — so
        refusing it on today's would be the same false POSITIVE `gen3_pinned_argv_parser_v1` fixed
        for the parser (a `--sync-to-main` fork refused for every HEAD-only flag it legitimately
        carries). `--arch` itself does not exist before 2026-09-06, so a pinned older argv could
        not even take the remedy the message offers. The diff is still computed and printed —
        never dropped, which is the other half of that lesson.
        """
        return bool(self.diffs) and self.fresh and not self.allowed and not self.advisory


# ------------------------------------------------------------------------- the production mirror
def load_production_config(path: Optional[str] = None) -> Dict[str, Any]:
    """The mirror, through the registry's own accessor.

    `baselines.production_config()` raises `BaselineError` naming the path when the mirror cannot
    be read, rather than returning `{}` — a guard that compares against an empty mirror reports "0
    keys differ", which is exactly the false clean it exists to prevent. An explicit `path` is
    honoured for the tests that build a synthetic mirror.
    """
    if path is None:
        return baselines.production_config()
    with open(path) as fh:
        return dict(json.load(fh))


def production_blob_sha(path: str = PRODUCTION_CONFIG_PATH) -> str:
    """git's own blob hash of the mirror, computed WITHOUT git.

    The same 40 hex chars `git rev-parse HEAD:designs/production_config.json` prints, from the bytes
    on disk — so it identifies the CONTENT a run was launched against even in a source tarball, a
    container, or a worktree whose index is mid-rebase. A commit sha would name when the file was
    last touched; this names what it SAID.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def arch_source_tag(path: str = PRODUCTION_CONFIG_PATH) -> str:
    """The `arch_source` string recorded in `model_config.json` by `--arch production`."""
    return f"production_config@{production_blob_sha(path)[:12]}"


# ------------------------------------------------------------------------------- the comparison
_MISSING = object()


def resolved_value(flag: ModelFlag, ns: Any) -> Tuple[Any, str]:
    """`(value, where it came from)` for one surface row on a resolved-ish namespace.

    Three cases, and every one of them is a real row in this registry:

    * a DERIVED row (`opp_belief_slots`, `opp_intent`) whose CLI surface is a coefficient — the
      surface value is the BOOL the extractor gets, i.e. `is_enabled(coef)`, because that is what
      `model_config.json` records and therefore the only thing comparable to the mirror;
    * a `config_only` row with no argparse dest at all — absent from the namespace, so its FROZEN
      registry default is the value, which is exactly what a CLI-launched run builds;
    * an unset tri-state flag — still `None` after parsing, so the registry `default` is what a
      FRESH run resolves to (`main.train.config`'s `_resolve` fills precisely that).

    A derived row is read from its COEFFICIENT first and from its own NAME second, and that order
    matters in both directions. An argv namespace carries only `opp_intent_coef` (there is no
    `--opp-intent` flag); a RECORDED `model_config.json` carries only `opp_intent` (the bool is
    what the extractor was built from, and the coefficient is a training field the config does not
    keep). Reading one shape only made the guard report `opp_intent False` against a config whose
    recorded value is `True` — a false finding on the live win-prob run, and the reason this
    function is usable against a recorded config as well as an argv.
    """
    raw = getattr(ns, flag.arg, _MISSING)
    if flag.derived and (raw is _MISSING or raw is None):
        recorded = getattr(ns, flag.name, _MISSING)
        if recorded is not _MISSING and recorded is not None:
            return bool(recorded), "recorded"
    if raw is _MISSING or raw is None:
        # `flag.default` is already the SURFACE value for a derived row (the bool), not a coef.
        return (bool(flag.default) if flag.derived else flag.default), "default"
    return (is_enabled(raw) if flag.derived else raw), "argv"


def diff_against_production(ns: Any, production: Optional[Dict[str, Any]] = None) -> List[ArchDiff]:
    """Every ARCH-surface key on which `ns` and the mirror disagree, in registry order.

    A key the mirror does not carry is SKIPPED rather than guessed at — that is the schema delta
    between a mirror and the live code (`arch_tables_test` treats it the same way), and "the mirror
    has not been regenerated since this toggle landed" is not a finding about the operator's argv.
    """
    prod = load_production_config() if production is None else production
    out: List[ArchDiff] = []
    for flag in arch_surface_flags():
        if flag.name not in prod:
            continue
        want = prod[flag.name]
        have, source = resolved_value(flag, ns)
        if have != want:
            out.append(ArchDiff(flag.name, flag.cli_flag, have, want, source))
    return out


def surface_partition() -> Dict[str, int]:
    """`{arch, critic, non_structural, total}` — WHY the guard compares fewer rows than exist.

    A guard that compares fewer keys than a reader's own count is the failure mode that matters:
    the reader cannot tell an excluded row from a forgotten one, so the difference must be
    ARITHMETIC and printed, not merely documented. The three buckets partition `REGISTRY` exactly
    (`arch_surface_partition_is_exhaustive` pins that), so a new row that lands without a `family`
    decision breaks the identity rather than quietly widening or narrowing the surface.

    Measured 2026-09-06: 39 arch + 7 structural critic + 3 non-structural = 49 registry rows. The
    hand-rolled check that validated the corrected win-prob relaunch compared all 49 and found 0
    differing; this guard compares the 39 and finds 0 on the same config. Both are right — they
    answer different questions, and the block says which one it asked.
    """
    surface = set(arch_surface_flags())
    arch = len(surface)
    critic = len([f for f in REGISTRY
                  if f not in surface and f.klass is Klass.STRUCTURAL
                  and f.tier is not Tier.CONSTRUCTOR_ONLY])
    return {"arch": arch, "critic": critic,
            "non_structural": len(REGISTRY) - arch - critic, "total": len(REGISTRY)}


def production_surface_keys(production: Optional[Dict[str, Any]] = None) -> List[str]:
    """The ARCH-surface keys the mirror actually carries — what `--arch production` can set."""
    prod = load_production_config() if production is None else production
    return [f.name for f in arch_surface_flags() if f.name in prod]


def unapplied_production_keys(production: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
    """Mirror values `--arch production` deliberately does NOT write, each with the flag to type.

    Two sources, both DECLARED in the registry row rather than listed here:

    * the non-surface rows — the CRITIC readouts (`family=critic`, what an experiment varies) and
      `resume_immutable` (`belief_grad_mode`);
    * the SUPERVISION DOSES attached to a surface toggle (`ModelFlag.coef_arg`). These are the
      dangerous ones: `--arch production` builds the production network with `move_belief_coef` and
      `spread_belief_coef` at 0.0 where production trains them at 0.05, so an umbrella that printed
      only what it applied would read as coverage of what it did not — the very mechanism this
      module exists to end, one layer down.

    Reported on EVERY `--arch` block, never suppressed when short: an operator must be able to see
    the boundary of the umbrella without going to look for it.
    """
    prod = load_production_config() if production is None else production
    surface = {f.name for f in arch_surface_flags()}
    out = [(f.cli_flag, prod[f.name]) for f in REGISTRY
           if f.name in prod and f.name not in surface
           and f.tier is Tier.CLI and f.klass is not Klass.RUNTIME]
    out += [(f"--{f.coef_arg.replace('_', '-')}", prod[f.coef_arg])
            for f in arch_surface_flags()
            if f.coef_arg and f.coef_arg in prod]
    return out


# ----------------------------------------------------------------------------------- the umbrella
def apply_production_arch(ns: Any, production: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
    """`--arch production`: write the whole ARCH surface onto `ns` as if it had been typed.

    Returns `[(flag, value), …]` for what it actually applied, so `--dry-run` can print it.

    PRECEDENCE is the tri-state sentinel every structural toggle already uses: a value that is not
    `None` was set by the operator (or by a desugar that ran first) and is left alone. So
    `--arch production --entity-topk-seats 0` really does build zero seats — and the guard below
    then reports that one key, which is the honest outcome: the umbrella is a default, not a lock.

    An attribute a `config_only` row would need is never CREATED. Those two rows have no argparse
    dest, and inventing one would put a value where `arch_toggles_from_args` expects the frozen
    default; both already sit AT production's value, so there is nothing to apply anyway. A `cli`
    row IS written even when the attribute is absent — `damage_matrices_outgoing` /
    `damage_matrices_incoming` have no flag of their own (the `--damage-matrices` MODE flag
    desugars into them) and so do not exist on the namespace yet. That desugar's else-branch is
    `if not hasattr(...)`, so writing them here is exactly the "already set" case it preserves, and
    an explicit `--damage-matrices off` afterwards still overrides. Without this the umbrella left
    the two per-move matrices OFF on a command that asked for production.
    """
    prod = load_production_config() if production is None else production
    applied: List[Tuple[str, Any]] = []
    for flag in arch_surface_flags():
        if flag.name not in prod:
            continue
        if flag.tier is not Tier.CLI:
            continue
        if getattr(ns, flag.arg, None) is not None:
            continue
        want = prod[flag.name]
        if flag.derived:
            # The mirror records the BOOL; the flag takes a coefficient. Prefer the coefficient the
            # mirror itself carries (`opp_belief_aux_coef` is a recorded training field); fall back
            # to the row's declared `on_value` for the one — `opp_intent` — whose coefficient
            # `model_config.json` does not record at all.
            if not is_enabled(want):
                continue
            want = prod.get(flag.arg, flag.on_value)
            if want is None:
                continue
        setattr(ns, flag.arg, want)
        # Keyed by the REGISTRY NAME, not the flag: two rows share `--damage-matrices` and two more
        # are set through a coefficient, so a flag-keyed list would print duplicates and hide which
        # surface key was actually written.
        applied.append((flag.name, want))
    return applied


# ------------------------------------------------------------------------------------ the verdict
def report(ns: Any, *, fresh: bool, allowed: bool = False,
           umbrella: Optional[str] = None, advisory: bool = False,
           production: Optional[Dict[str, Any]] = None) -> ArchReport:
    """THE one entry point. `main.checkargs`, `--dry-run` and the launcher all call this."""
    return ArchReport(
        diffs=tuple(diff_against_production(ns, production)),
        fresh=bool(fresh),
        allowed=bool(allowed),
        umbrella=umbrella,
        source_tag=arch_source_tag(),
        advisory=bool(advisory),
    )


def report_for_child_argv(child_args: Sequence[str],
                          *, advisory: bool = False) -> Optional[ArchReport]:
    """`report()` for a CHILD ARGV — the launcher's entry point, and `--dry-run`'s.

    Builds the effective namespace through `main.checkargs.check` (which parses with the REAL
    trainer parser, runs the launch path's own desugars — the `--arch production` umbrella
    included — and overlays a `--model` parent's recorded config), then returns the report that
    `check` already computed. `None` when the argv does not parse: a parse failure is the caller's
    louder finding and is reported by its own path.

    The import is function-local because `main.checkargs` imports THIS module at module scope.
    """
    from main.checkargs import check
    try:
        return check(list(child_args), advisory=advisory).get("arch")
    except Exception:                                # noqa: BLE001 — a guard never crashes a launch
        return None


def report_lines(rep: ArchReport) -> List[str]:
    """The printed block, identical on all three surfaces."""
    out = [f"ARCH SURFACE vs designs/production_config.json  [{rep.source_tag}]"]
    if rep.umbrella:
        out.append(f"  --arch {rep.umbrella} applied the production surface "
                   f"(explicit flags still win)")
        skipped = unapplied_production_keys()
        if skipped:
            out.append("  ⚠️  NOT applied — `--arch` writes the ARCHITECTURE only. The mirror's "
                       "CRITIC readouts and its SUPERVISION DOSES are outside that surface by "
                       "class (an experiment varies the first; the second is a training dose, not "
                       "a network). Type them yourself if you want production's values:")
            out.append("      " + "  ".join(f"{f} {v!r}" for f, v in skipped))
    part = surface_partition()
    scope = (f"{part['arch']} of {part['total']} registry toggles are the ARCH surface; "
             f"{part['critic']} critic readouts + {part['non_structural']} non-structural rows "
             f"are excluded by their own declaration")
    if not rep.diffs:
        out.append(f"  ✓ every ARCH-surface key matches the production mirror ({scope})")
        return out
    verb = "differ" if len(rep.diffs) != 1 else "differs"
    out.append(f"  {len(rep.diffs)} of {part['arch']} ARCH-surface key(s) {verb} from production "
               f"({scope}):")
    out += [f"      {d.line()}" for d in rep.diffs]
    if not rep.fresh:
        out.append("  ℹ️  INFO only — this is a FORK/RESTART, which INHERITS its parent's arch "
                   "surface from the recorded model_config.json. The diff above is against "
                   "PRODUCTION, not against the parent.")
        return out
    if rep.allowed:
        out.append(f"  ℹ️  {ALLOW_FLAG} — the drift above is an EXPLICIT choice and is recorded "
                   "in model_config.json's arch_source.")
        return out
    if rep.advisory:
        out.append("  ℹ️  ADVISORY — the child is PINNED to another commit, and this mirror is the "
                   "CURRENT tree's. That commit has its own registry, its own flags and its own "
                   "production_config.json, so today's is not the authority on it (and `--arch` "
                   "may not exist there at all). Reported, not gated.")
        return out
    out += [
        "  ✗ REFUSED: a FRESH run whose architecture is not production's, and did not say so.",
        "     'it launches' and 'it is the experiment' are INDEPENDENT checks, and only the diff",
        "     above tests the second. 2026-09-06: an arm launched from a document's 38-token",
        "     command block trained a near-bare network for ~7 GPU-hours / 24.4M steps. checkargs",
        "     said 'still launches'; resolve_config accepted; --dry-run said 'would launch'; all",
        "     three were right, and none of them had been asked this.",
        "     FIX — either:",
        "       * pass `--arch production` (applies every key above; explicit flags still win), or",
        f"       * pass `{ALLOW_FLAG}` to assert the drift is deliberate.",
    ]
    return out


# -------------------------------------------------------------------------------- argv helpers
def argv_has_allow(argv: Sequence[str]) -> bool:
    """Is `--allow-nonproduction-arch` in this argv? (Launcher-side; no parser needed.)"""
    return ALLOW_FLAG in argv or ALLOW_FLAG.replace("-", "_") in argv


def argv_umbrella(argv: Sequence[str]) -> Optional[str]:
    """The `--arch <value>` in this argv, in either spelling and either `=` form."""
    for i, tok in enumerate(argv):
        if tok in ("--arch",):
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--arch="):
            return tok.split("=", 1)[1]
    return None


def argv_model(argv: Sequence[str]) -> Optional[str]:
    """The `--model` value, for the FRESH-vs-fork split, without parsing the whole argv."""
    for i, tok in enumerate(argv):
        if tok in ("--model", "--model="):
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--model="):
            return tok.split("=", 1)[1]
    return None
