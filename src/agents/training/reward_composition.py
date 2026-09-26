"""THE COMPOSITION ANNOUNCER — what a config's reward is MADE OF, as a census and one line.

Stateless and duck-typed over a CONFIG: it reads field NAMES off any config-shaped object (a
`RewardConfig`, a recorded `ModelVersion`, an argparse namespace) and never touches a manager, a
battle or a turn.

**Since the shaped-reward deletion (2026-09-26, program_rust_core §4 M3 row) every config's census
is the same: `1 TERMINAL + 0 PBRS + 0 BIAS`.** The announcer is kept, with its output format and its
`metadata.json` block unchanged, because it exists for the v8→v9 lesson — a run STATES its reward
composition rather than implying it — and because its readers (the startup line, the `reward/`
export's tracked set, the `metadata.json` `reward_composition` block, `reward_config_digest`'s
cf-label stamp) are unchanged. The PBRS and BIAS keys stay in the census (always 0 / empty) so a
reader of an old and a new `metadata.json` reads one schema.
"""
import hashlib
from dataclasses import fields

from agents.training.reward_config import RewardBreakdown, RewardClass


def _rc(config, name, default):
    """Read a reward field off any config-shaped object (RewardConfig / ModelVersion / namespace)."""
    return getattr(config, name, default)


def reward_class_composition(config) -> dict:
    """The per-class ACTIVE-term census of `config` — what this run's reward is MADE OF.

    Returns ``{"terminal": n, "pbrs": n, "bias": n, "bias_terms": [names], "pbrs_terms": [names],
    "terminal_terms": [names]}``. Every term in the registry is TERMINAL now, so the PBRS and BIAS
    halves are always empty; `config` is still taken so the signature (and every caller) is stable.
    """
    reg = RewardBreakdown._REGISTRY
    terminal = [n for n, c in reg.items() if c is RewardClass.TERMINAL]
    return {"terminal": len(terminal), "pbrs": 0, "bias": 0,
            "bias_terms": [], "pbrs_terms": [], "terminal_terms": terminal}


def reward_config_digest(config) -> str:
    """A stable sha1 over EVERY field of a `RewardConfig` — the identity of a reward function.

    `gen3_cf_twin_heads_v1`. A Monte-Carlo return label manufactured by an offline producer is only
    a label for THIS run if the producer used THIS run's reward, and the number itself cannot say
    so. Stable across processes and Python versions: fields sorted by name, rendered with `repr`.
    ⚠️ The shaped-reward deletion REMOVED fields, so a digest stamped by a pre-deletion producer
    does not match a post-deletion consumer's for the same (terminal-only) reward — a loud
    mismatch, never a silent one.
    """
    try:
        items = {f.name: getattr(config, f.name) for f in fields(config)}
    except TypeError:                                    # not a dataclass — best effort
        items = dict(vars(config))
    body = ";".join(f"{k}={items[k]!r}" for k in sorted(items))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def format_reward_composition(config) -> str:
    """One human line: ``[Reward] composition: 1 TERMINAL + 0 PBRS + 0 BIAS (none — fully
    policy-invariant)``, printed at startup so a launch STATES its reward composition."""
    comp = reward_class_composition(config)
    names = comp["bias_terms"]
    tail = ", ".join(names) if names else "none — fully policy-invariant"
    return (f"[Reward] composition: {comp['terminal']} TERMINAL + {comp['pbrs']} PBRS "
            f"+ {comp['bias']} BIAS ({tail})")


def inert_reward_flags(config) -> list:
    """The recorded reward flags this config makes INERT — sorted, possibly empty.

    One source survives the deletion: `draw_penalty` under `--terminal-indicator`. The term is
    still emitted; the flag's number is simply not read, because the indicator pays
    `+victory_value` on a win and `0.0` on a loss, a tie AND a 250-turn timeout alike. It is
    DOCUMENTATION written beside the fields (`snapshot.save_model_snapshot`), never in place of one.
    """
    return ["draw_penalty"] if bool(_rc(config, "terminal_indicator", False)) else []


def reward_composition_block(config) -> dict:
    """The `reward_composition` block `metadata.json` records — the census PLUS the announcer's own
    line, each class's share of the active terms, and `inert_reward_flags`."""
    comp = dict(reward_class_composition(config))
    total = comp["terminal"] + comp["pbrs"] + comp["bias"]
    comp["composition_line"] = format_reward_composition(config)
    comp["class_shares"] = ({k: comp[k] / total for k in ("terminal", "pbrs", "bias")}
                            if total else {"terminal": 0.0, "pbrs": 0.0, "bias": 0.0})
    comp["inert_reward_flags"] = inert_reward_flags(config)
    return comp
