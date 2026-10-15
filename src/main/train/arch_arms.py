"""NAMED ARMS for `--arch` — an experiment arm DECLARED as the production surface plus an overlay.

`--arch production` applies `designs/production_config.json` (the ARCH surface) and its training recipe. A NAMED ARM
is the same thing with a small, declared OVERLAY of ARCH-surface keys on top: `--arch <arm>` applies production ⊕ the
overlay as if every key had been typed (an explicitly typed flag still wins), applies production's RECIPE unchanged
(`main.train.recipe_surface`: both arms of a comparison train on the same recipe), and the ARCH-surface guard then
compares a fresh argv against production ⊕ the overlay instead of production alone — so the arm launches without
`--allow-nonproduction-arch`, and any drift FROM THE ARM is still refused, naming the key (`main.train.arch_surface`).

Why a declaration and not a copied flag list: "it launches" and "it is the experiment" are independent checks
(`arch_surface`'s module docstring, the 2026-09-06 incident). A bundle typed from a document's command block is one
forgotten flag away from a different network; a declared arm is checked by `python -m main.checkargs`, `--dry-run` and
the launcher against ONE table, and its provenance tag (`arch_source`) names the arm and both contents by hash.

Each arm says what it is FOR and where it is designed; an arm is a closed bundle, never widened ad hoc (a new lever is
a new arm or a bisection of this one). Torch-free and import-light: the parser reads `NAMED_ARMS` for its choices.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Tuple


class ArmDecl(NamedTuple):
    """One named arm: the overlay over the production ARCH surface (registry name -> value) and why."""

    overlay: Tuple[Tuple[str, Any], ...]
    purpose: str
    design: str


#: THE PRE-ADOPTION PRODUCTION SURFACE (until 2026-10-12): the value each of the nineteen END-STATE keys had in
#: `designs/production_config.json` before the closing test's PASS made the end state production. ONE declared table,
#: read by tests that build the LEGACY network on purpose (`legacy_production_config`; the `static_recovery` arm that
#: also read it was deleted in legacy-deletion U0). Deleted with the legacy paths it names (`designs/ops/legacy_deletion_manifest_2026-10-12.md`).
PRE_ADOPTION_PRODUCTION: Tuple[Tuple[str, Any], ...] = (
    ("token_encoding", "legacy"),
    ("mon_hazard_cost", "off"),
    ("move_actor_state", "off"),
    ("trunk_layers", 2),
    ("switch_hazard_cost", "off"),
    ("eot_residual", "off"),
    ("move_resolution", "off"),
    ("speed_physics", "off"),
    ("value_threat_inject", True),
    ("op_reduction", "max"),
    ("obs_facts", "off"),
    ("move_resolution_facts", "off"),
    ("status_facts", "off"),
    ("ko_ramp", "ramp"),
    ("drop_progress_clock", "off"),
    ("g_ledger", "coarse"),
    ("effective_stats", "off"),
    ("move_target_state", "off"),
    ("move_set_closure", "off"),
)

#: --arch values that were DELETED, with the reason the parser refuses them by (never a bare "invalid choice").
#: `designs/deleted_flags.md` §1b carries the same rows.
DELETED_ARMS: Dict[str, str] = {
    # legacy deletion U0 (designs/ops/legacy_deletion_manifest_2026-10-12.md G1 / D-LD3, 2026-10-15): the arm was
    # DEFINED by OFF values (`move_resolution off`, `value_threat_inject True`, `op_reduction max`, ...) that the
    # deletion units remove, so it could not survive the first of them.
    "static_recovery": (
        "deleted by the legacy deletion's U0 (designs/ops/legacy_deletion_manifest_2026-10-12.md, D-LD3): the arm "
        "was the static encoding + the six static-recovery levers WITHOUT the end state's other levers, i.e. defined "
        "by OFF values the deletion removes; the closing test PASSED without needing it (its bisection rung C1). "
        "Run it PINNED at the tag `pre-legacy-deletion` (0a0d9e1d)"),
}

#: The named arms. Keys are `--arch` values; `production` is not one of them (it is the mirror itself).
NAMED_ARMS: Dict[str, ArmDecl] = {
    # gen3_endstate_facts_v1 (owner 2026-10-09: "run all of them speculatively together, then bisect"): the END-STATE
    # arm — `static_recovery` + the bundle's four levers (move resolution, speed physics, the principled reductions,
    # the OBS-FACTS block) with the critic's threat injection off, + every fact-completion lever (v152), the probe
    # battery's two facts (v153) and the four-move closure (v154). ADOPTED AS PRODUCTION 2026-10-12 (the closing test's
    # PASS, `design_endstate_closing_test.md` §9): `designs/production_config.json` now carries every key below at
    # these values, so `--arch endstate` is an ALIAS of `--arch production` (the same network and recipe; only its
    # provenance tag `arch_source` names the arm), kept so the closing test's and the bf16 A/B's argvs still resolve.
    "endstate": ArmDecl(
        overlay=(
            ("token_encoding", "static"),
            ("mon_hazard_cost", "on"),
            ("move_actor_state", "on"),
            ("trunk_layers", 3),
            ("switch_hazard_cost", "on"),
            ("eot_residual", "on"),
            ("move_resolution", "on"),
            ("speed_physics", "on"),
            ("value_threat_inject", False),
            ("op_reduction", "principled"),
            ("obs_facts", "v1"),
            ("move_resolution_facts", "full"),
            ("status_facts", "exact"),
            ("ko_ramp", "exact"),
            ("drop_progress_clock", "on"),
            ("g_ledger", "eot"),
            ("effective_stats", "on"),
            ("move_target_state", "on"),
            # gen3_move_set_closure_v1 (config v154; owner 2026-10-10 "Do A", a PRE-DATA amendment of the closing
            # test): the four-move FACT in every opponent slot's MoveBelief reinjection.
            ("move_set_closure", "on"),
        ),
        purpose="ALIAS of production since the 2026-10-12 adoption: static_recovery + move resolution, speed "
                "physics, the principled op reductions, the OBS-FACTS block, no critic threat injection, + every "
                "fact-completion lever (the restored move-resolution facts, the exact status facts and cure flags, the "
                "exact P(KO), no progress clock, one end-of-turn rule, the active mons' stage-applied stats, their "
                "active's state on our move seats, the four-move closure in the move belief's reinjection); the end "
                "state, which PASSED its closing test and is production",
        design="designs/endstate/design_hand_computed_features.md §4 / §5 (and design_static_tokens.md §13)",
    ),
}


def arch_value(value: str) -> str:
    """The `--arch` argparse TYPE: a DELETED arm is refused WITH its reason (argparse prints it after
    `argument --arch:`); every other value goes on to the parser's own `choices` check."""
    import argparse

    if value in DELETED_ARMS:
        raise argparse.ArgumentTypeError(f"--arch {value} was DELETED — {DELETED_ARMS[value]}")
    return value


def arm_overlay(name: str) -> Dict[str, Any]:
    """The overlay of arm ``name`` as a dict (KeyError naming the known arms for an unknown name)."""
    if name in DELETED_ARMS:
        raise KeyError(f"--arch {name!r} was DELETED: {DELETED_ARMS[name]}")
    if name not in NAMED_ARMS:
        raise KeyError(f"unknown --arch arm {name!r}; the named arms are {sorted(NAMED_ARMS)}")
    return dict(NAMED_ARMS[name].overlay)


def legacy_production_config(mirror: Dict[str, Any]) -> Dict[str, Any]:
    """``mirror`` (a production-config dict) with the nineteen end-state keys at their PRE-ADOPTION values — the
    LEGACY production network, for code and tests that build it on purpose (a lever's "production ⊕ lever" test
    written against the legacy base). Every other key is the mirror's."""
    out = dict(mirror)
    for k, v in PRE_ADOPTION_PRODUCTION:
        if k not in out:
            raise KeyError(f"the production mirror carries no {k!r}; PRE_ADOPTION_PRODUCTION is out of step with it")
        out[k] = v
    return out
