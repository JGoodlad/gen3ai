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


#: The named arms. Keys are `--arch` values; `production` is not one of them (it is the mirror itself).
NAMED_ARMS: Dict[str, ArmDecl] = {
    # gen3_static_recovery_v1 (owner 2026-10-09: "run all of them speculatively together and then we can bisect them
    # out"): the static token encoding with EVERY static-recovery lever on, on the production recipe. The closing-test
    # candidate against legacy; bisected lever by lever afterwards.
    "static_recovery": ArmDecl(
        overlay=(
            ("token_encoding", "static"),
            ("mon_hazard_cost", "on"),
            ("move_actor_state", "on"),
            ("trunk_layers", 3),
            ("switch_hazard_cost", "on"),
            ("eot_residual", "on"),
        ),
        purpose="the static encoding + every static-recovery lever (trunk depth 3, the per-mon Spikes fact, the "
                "actor's state on its move seats, the switch-cell hazard, the end-of-turn residual); the closing-test "
                "candidate against legacy, bisected lever by lever",
        design="designs/endstate/design_static_tokens.md §13",
    ),
    # gen3_endstate_facts_v1 (owner 2026-10-09: "run all of them speculatively together, then bisect"): the END-STATE
    # arm — `static_recovery` + the bundle's four levers (move resolution, speed physics, the principled reductions,
    # the OBS-FACTS block) with the critic's threat injection off, + every fact-completion lever. Bisected afterwards.
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
        ),
        purpose="static_recovery + move resolution, speed physics, the principled op reductions, the OBS-FACTS block, "
                "no critic threat injection, + every fact-completion lever (the restored move-resolution facts, the "
                "exact status facts and cure flags, the exact P(KO), no progress clock, one end-of-turn rule); the "
                "end-state candidate, bisected lever by lever",
        design="designs/endstate/design_hand_computed_features.md §4 / §5 (and design_static_tokens.md §13)",
    ),
}


def arm_overlay(name: str) -> Dict[str, Any]:
    """The overlay of arm ``name`` as a dict (KeyError naming the known arms for an unknown name)."""
    if name not in NAMED_ARMS:
        raise KeyError(f"unknown --arch arm {name!r}; the named arms are {sorted(NAMED_ARMS)}")
    return dict(NAMED_ARMS[name].overlay)
