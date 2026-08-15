"""ONE place that turns parsed CLI args into `Gen3FeaturesExtractor` kwargs.

WHY THIS MODULE EXISTS. `train_rl_agent.py` used to assemble this dict TWICE — once on the fresh-run
path (`extractor_kwargs`) and once on the resume path (`_load_extractor_kwargs`) — as two ~45-line
blocks of `d["flag"] = args.flag`. They were verified identical at the time this was factored out (42
keys, same set, same sources), but nothing enforced that: adding a v51 toggle to one and not the other
would mean a resume silently version-checks against a DIFFERENT arch than it builds. That is the
`gen3_resume_optimizer_realign_bug` failure shape — a mismatch that surfaces as a confusing tensor
error much later, or not at all.

There is now a third consumer that made the duplication untenable: the forkserver preload
(`agents.model.compile_preload`) must build the SAME extractor the env workers will, in a fresh
interpreter that never parsed argv, so it needs this mapping as DATA rather than as inline statements.

`ARCH_ARG_KEYS` is that data: extractor-kwarg name -> the `args` attribute it reads. It is now
**GENERATED from `agents.model.flag_registry`** rather than hand-kept — the registry is the single
declaration of every model-relevant toggle and of which of the five hand-synced surfaces it belongs
on (`flag_registry_test.py` validates the other four). Anything needing a derivation (a coef>0 enable
signal, a private computed field) lives in `_DERIVED`, which stays hand-written because its values
are callables; the registry still declares those rows (`derived=True`) so the key SET is checked.

`FROZEN_ARCH_KWARGS` is the config-only tier: toggles with no argparse entry at all, frozen at the
registry's `default` for every CLI-launched run. They are still recorded in `model_config.json` and
still resume-gated — only the SELECT role is gone (see `flag_registry`'s tier table). The extractor
constructor kwarg survives, so each stays reachable for an experiment.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agents.model.flag_registry import REGISTRY, Tier

# extractor kwarg -> args attribute. Straight pass-throughs, GENERATED from the registry in
# declaration order (which is `since` order, so a new toggle appends).
ARCH_ARG_KEYS: Dict[str, str] = {
    f.name: f.arg for f in REGISTRY if f.tier is Tier.CLI and not f.derived
}

# Config-only kwarg -> its FROZEN value. No argparse entry reads these; the registry default IS
# the value every CLI-launched run gets.
FROZEN_ARCH_KWARGS: Dict[str, Any] = {
    f.name: f.default for f in REGISTRY if f.tier is Tier.CONFIG_ONLY
}

# Kwargs that are NOT a plain attribute read. Each is a callable over `args`. The registry declares
# these rows with `derived=True`; `flag_registry_test` pins the key set to it, so one can neither
# appear here without a registry row nor be dropped from here while the row survives.
_DERIVED = {
    # coef>0 is the enable signal for these two; the COEF itself is a training hparam set on the
    # model, but the BOOL is the version-checked arch toggle.
    "opp_belief_slots": lambda a: getattr(a, "opp_belief_aux_coef", 0.0) > 0.0,
    # v67 gen3_opp_intent_v1 — same shape: the COEF is the training hparam, the BOOL builds the heads.
    "opp_intent": lambda a: getattr(a, "opp_intent_coef", 0.0) > 0.0,
}


def build_extractor_arch_kwargs(args, base: Optional[Dict[str, Any]] = None,
                                log_level: Any = None) -> Dict[str, Any]:
    """Layout kwargs (`base`) + every version-checked architecture toggle read off `args`.

    `base` is normally `Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()` — the obs
    layout half. Pass None to get only the arch half (what the compile pre-warm wants when it builds
    its own layout). `log_level` is threaded only on the fresh-run path; it is a diagnostic, not an
    arch field, so it is omitted when None rather than being written as a null.
    """
    kwargs: Dict[str, Any] = dict(base) if base else {}
    for kwarg, attr in ARCH_ARG_KEYS.items():
        kwargs[kwarg] = getattr(args, attr)
    for kwarg, fn in _DERIVED.items():
        kwargs[kwarg] = fn(args)
    # The config-only tier LAST and unconditionally: there is no `args` attribute to read, and a
    # frozen value must not be overridable by one that happens to be lying around on the namespace.
    kwargs.update(FROZEN_ARCH_KWARGS)
    if log_level is not None:
        kwargs["log_level"] = log_level
    return kwargs


def arch_toggles_from_args(args) -> Dict[str, Any]:
    """Every registry toggle's value for THIS run, keyed by name — the `current_model_version`
    half of `build_extractor_arch_kwargs`.

    Same three sources, no layout: a plain attribute read, a `_DERIVED` callable, or the frozen
    config-only value. `train_rl_agent._run_arch_toggles` builds on this so the version gate and
    the extractor cannot be fed a different toggle set (they were two hand-kept lists).
    """
    toggles: Dict[str, Any] = {kwarg: getattr(args, attr) for kwarg, attr in ARCH_ARG_KEYS.items()}
    toggles.update({kwarg: fn(args) for kwarg, fn in _DERIVED.items()})
    toggles.update(FROZEN_ARCH_KWARGS)
    return toggles


def arch_kwargs_to_plain(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-serialisable subset, for handing an arch across a process boundary.

    The forkserver preload receives its config through the environment, so anything unpicklable or
    huge (the layout's numpy mapping tables) is dropped — the preload only needs the toggles that
    change the traced GRAPH. Dropping a key means the preload builds a slightly different extractor
    and the worker's guards miss, which costs a recompile but is never incorrect, so this errs
    toward dropping.
    """
    plain: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in ("log_level",):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            plain[k] = v
    return plain
