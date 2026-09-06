"""THE COMPOSITION ANNOUNCER — what a config's reward is MADE OF, as a census and one line.

Split out of `reward_manager.py` (2026-09-06, alongside `gen3_winprob_critic_mode_v1`), which was
11 lines short of the file-size gate's 2,000-line hard bound. It is a natural seam rather than an
arbitrary cut: everything here is **stateless and duck-typed over a CONFIG** — it reads field NAMES
off any config-shaped object (a `RewardConfig`, a recorded `ModelVersion`, an argparse namespace)
and never touches a manager, a battle or a turn. `reward_manager.py`'s subject is the per-decision
FOLD; this module's subject is the STATIC question *"which terms can this config emit at all?"*,
and the two never needed to share a file.

⚠️ **THE GATES HERE ARE THE FOLDS' OWN, AND THAT IS THE WHOLE POINT.**
`Gen3RewardManager._hand_pbrs_on` delegates to `_pbrs_term_active`, and
`_apply_pbrs_suppression` / `_apply_bias_drops` / `__init__`'s `_active_bias` fast path all read
`_bias_term_active`. They were two hand-maintained copies of the same conditions until 2026-08-29,
which is exactly how a census can advertise a composition the folds do not implement. Keep the
delegation: a rename here must break three call sites loudly rather than silently un-gate a term.

**Imports of `reward_manager` are FUNCTION-LOCAL, deliberately.** `reward_manager` re-exports every
public name below (so `from agents.training.reward_manager import reward_class_composition` — and
`_pbrs_term_active` / `_bias_term_active` / `_rc`, which the tests read — still resolves), which
makes the dependency mutual. Deferring the three names this module needs to call time is what keeps
that from being an import cycle, and costs one dict lookup on a path that runs once per launch.
"""
import hashlib
from dataclasses import fields


def _rc(config, name, default):
    """Read a reward field off any config-shaped object (RewardConfig / ModelVersion / namespace)."""
    return getattr(config, name, default)


def _pbrs_term_active(config, name: str) -> bool:
    """Is PBRS term `name` folded under `config`? THE gate — every `_fold_*_pbrs` calls this through
    ``Gen3RewardManager._hand_pbrs_on``, so the census below and the folds cannot drift apart (they
    were two hand-maintained copies of the same conditions until 2026-08-29)."""
    if not bool(_rc(config, "hand_shaping", True)):
        return False                   # --no-hand-shaping: every hand potential off, TERMINAL alone
    asp = bool(_rc(config, "all_shaping_pbrs", True))
    if name == "pbrs_material":        # _fold_material_pbrs — its OWN flag, not asp's (see RewardConfig)
        return bool(_rc(config, "pbrs_material", True))
    if name == "pbrs_belief":          # _fold_belief_pbrs — likewise
        return bool(_rc(config, "pbrs_belief", True))
    if name == "pbrs_status":          # _fold_status_pbrs
        return bool(_rc(config, "bias_redesign", False)) or asp
    if name == "pbrs_progress":        # _fold_progress_pbrs
        return bool(_rc(config, "stall_pbrs", False))
    if name in ("pbrs_hazard", "pbrs_boost", "pbrs_opp_boosts", "pbrs_roar"):
        return asp
    return True


def _bias_term_active(config, name: str) -> bool:
    """Is BIAS term `name` reachable under `config`? Mirrors `_apply_pbrs_suppression`,
    `_apply_bias_drops`, `_apply_progress_clock` and the three weight-gated terms."""
    if not bool(_rc(config, "hand_shaping", True)):
        # --no-hand-shaping zeroes the WHOLE BIAS class, tilt included — UNLESS the anti-stall
        # tilt was explicitly re-armed (gen3_winprob_critic_mode_v1, design gap B4). The re-armed
        # term still has to satisfy its OWN gate below, so `--arm-no-progress-tax` re-arms the
        # tilt rather than reviving the other 24 BIAS terms.
        if not (name == "no_progress_tax" and bool(_rc(config, "no_progress_tax_armed", False))):
            return False
    asp = bool(_rc(config, "all_shaping_pbrs", True))
    stall = bool(_rc(config, "stall_pbrs", False))
    if name == "no_progress_tax":
        # Charged only under --bias-redesign OR --all-shaping-pbrs; --stall-pbrs then zeroes it
        # (Φ_progress carries the anti-stall signal policy-invariantly instead).
        return (bool(_rc(config, "bias_redesign", False)) or asp) and not stall
    if asp:
        return False                   # everything-but-stall → every other BIAS term is zeroed
    if name == "stall_tax":
        return not (stall or bool(_rc(config, "drop_redundant_bias", False)))
    if name == "matchup_penalty":
        return not bool(_rc(config, "drop_redundant_bias", False))
    from agents.training.reward_manager import SWITCH_BIAS_DROP_FAMILY
    if name in SWITCH_BIAS_DROP_FAMILY:
        return not bool(_rc(config, "drop_switch_bias", False))
    if name in ("stay_risk_tax", "escape_risk_bonus"):
        return float(_rc(config, "switch_bias_weight", 0.0)) > 0.0
    if name == "self_ko_penalty":
        return float(_rc(config, "self_ko_hp_penalty", 0.0)) > 0.0
    return True


def reward_class_composition(config) -> dict:
    """The per-class ACTIVE-term census of `config` — what this run's reward is MADE OF.

    Returns ``{"terminal": n, "pbrs": n, "bias": n, "bias_terms": [names], "pbrs_terms": [names],
    "terminal_terms": [names]}``. `bias_terms` is the one a reader acts on: the BIAS class is the
    only one that biases the converged optimum, so naming its members is naming the run's
    hand-coded incentives. `terminal_terms` is ADDITIVE (`gen3_reward_term_export_v1`) — the
    counts and the two older lists are unchanged, and the `reward/` live export derives its
    tracked set from all three so the exported terms cannot disagree with the census.
    """
    from agents.training.reward_manager import RewardBreakdown, RewardClass
    reg = RewardBreakdown._REGISTRY
    pbrs = [n for n, c in reg.items() if c is RewardClass.PBRS and _pbrs_term_active(config, n)]
    bias = [n for n, c in reg.items() if c is RewardClass.BIAS and _bias_term_active(config, n)]
    terminal = [n for n, c in reg.items() if c is RewardClass.TERMINAL]
    return {"terminal": len(terminal), "pbrs": len(pbrs), "bias": len(bias),
            "bias_terms": bias, "pbrs_terms": pbrs, "terminal_terms": terminal}


def reward_config_digest(config) -> str:
    """A stable sha1 over EVERY field of a `RewardConfig` — the identity of a reward function.

    `gen3_cf_twin_heads_v1`. A shaped RETURN is a fact about a board *under a reward composition*,
    so a Monte-Carlo return label manufactured by an offline producer is only a label for THIS run
    if the producer used THIS run's reward. There is no other way to tell: the number is a float,
    and a return computed under a different composition is not a noisier sample of ours — it is a
    measurement of a different value function, and averaging it in is silent GIGO.

    Stable across processes and Python versions: the fields are sorted by name and rendered with
    `repr`, so it depends on the VALUES and not on dataclass declaration order or dict iteration.
    Floats go through `repr` deliberately — two configs that differ in the 15th decimal of a weight
    ARE different rewards, and rounding here would hide exactly the drift the digest exists to
    catch. Duck-typed (`fields()` when available, else `vars()`) like everything else that consumes
    a reward config.
    """
    try:
        items = {f.name: getattr(config, f.name) for f in fields(config)}
    except TypeError:                                    # not a dataclass — best effort
        items = dict(vars(config))
    body = ";".join(f"{k}={items[k]!r}" for k in sorted(items))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def format_reward_composition(config) -> str:
    """One human line: ``[Reward] composition: 1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)``.

    Printed at startup so a launch STATES its reward composition instead of implying it. With no
    BIAS terms the tail reads ``(none — fully policy-invariant)``; with many it truncates, because
    the count is the signal and the long additive list is the pathology, not the detail.
    """
    comp = reward_class_composition(config)
    names = comp["bias_terms"]
    if not names:
        tail = "none — fully policy-invariant"
    elif len(names) <= 6:
        tail = ", ".join(names)
    else:
        tail = ", ".join(names[:6]) + f", … +{len(names) - 6} more"
    return (f"[Reward] composition: {comp['terminal']} TERMINAL + {comp['pbrs']} PBRS "
            f"+ {comp['bias']} BIAS ({tail})")
