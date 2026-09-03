"""COMBINATION CHECKS — the value-conditional refusals, in ONE place, read by BOTH surfaces.

WHY THIS MODULE EXISTS. `agents.model.flag_registry`'s `requires` graph expresses one shape of
dependency — *flag A must be ENABLED for flag B to be enabled* — and `main.checkargs` reads it, so
an unsatisfiable structural combination is reported offline instead of crashing inside
`Gen3FeaturesExtractor.__init__`. But some launch-time refusals are not that shape: they are
value-conditional (*`--distill-target action` requires `--distill-coef` > 0*), they live in
`main.train.config.resolve_config` as `parser.error` lines, and nothing outside that function knew
them.

That gap has now cost a launch. C1 (2026-09-01) forked a parent whose recorded config carried
`distill_target="action"`, passed `--distill-coef 0`, and did NOT name `--distill-target` — so
`_resolve` inherited `action`, the check below fired, and the run died at launch. `checkargs` had
said "this command still launches", because it saw neither the inherited value nor the rule.

THE CONTRACT. A check here is a pure predicate over an args-shaped namespace plus the message the
LAUNCH path prints. `resolve_config` calls `failing_checks` and `parser.error`s the first message;
`main.checkargs` calls the same function on the EFFECTIVE namespace (argv overlaid on the fork
parent's recorded config) and reports every one. Neither owns the rule, so they cannot drift.

WHAT BELONGS HERE: a refusal that reads two or more RESOLVED values and says one combination is
incoherent. What does NOT: a range check on a single value (`--distill-topk >= 1`), anything that
needs the parser, the filesystem, or a teacher spec — those stay in `resolve_config`, which has
them.
"""
from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Tuple


class CombinationCheck(NamedTuple):
    """One value-conditional refusal. `predicate` is TRUE when the combination is BROKEN."""

    name: str
    dests: Tuple[str, ...]          # the args attributes it reads — checkargs prints their provenance
    predicate: Callable[[Any], bool]
    message: str


def _positive(value: Any) -> bool:
    """The `args.distill_coef and args.distill_coef > 0` idiom the launch path uses, None-safe."""
    return bool(value) and float(value) > 0.0


# gen3_distill_target_gate_v1 (design_advantage_gated_distillation.md §7.5): the action-form
# family's dependency graph. Lifted VERBATIM out of `resolve_config` — same order, same messages.
COMBINATION_CHECKS: Tuple[CombinationCheck, ...] = (
    CombinationCheck(
        "distill_target_needs_coef",
        ("distill_target", "distill_coef"),
        lambda a: getattr(a, "distill_target", None) == "action"
        and not _positive(getattr(a, "distill_coef", None)),
        "--distill-target action requires --distill-coef > 0 — the target form is a "
        "property of the distill term; without the term there is nothing to shape",
    ),
    CombinationCheck(
        "distill_topk_needs_action",
        ("distill_topk", "distill_target"),
        lambda a: getattr(a, "distill_topk", 1) not in (1, None)
        and getattr(a, "distill_target", None) not in ("action", None),
        "--distill-topk requires --distill-target action — the top-K dial "
        "parameterizes the action-form target; the 'kl' path has no K",
    ),
    CombinationCheck(
        "distill_gate_needs_action",
        ("distill_gate", "distill_target"),
        lambda a: getattr(a, "distill_gate", "none") not in ("none", None)
        and getattr(a, "distill_target", None) not in ("action", None),
        "--distill-gate requires --distill-target action (design §7.5: the gate "
        "rides the action-form term)",
    ),
    CombinationCheck(
        "distill_gate_tau_needs_advantage",
        ("distill_gate_tau", "distill_gate"),
        lambda a: (getattr(a, "distill_gate_tau", 0.0) or 0.0) != 0.0
        and getattr(a, "distill_gate", "none") not in ("advantage", None),
        "--distill-gate-tau requires --distill-gate advantage — tau is the advantage "
        "gate's threshold",
    ),
)


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
