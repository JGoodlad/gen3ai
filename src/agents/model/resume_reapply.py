"""Re-apply the RESUME-IMMUTABLE run-level attributes to a freshly LOADED extractor (gen3_reveal_provenance_v1).

SB3's load rebuilds the policy -- and so the extractor -- from the ZIP's saved ``policy_kwargs``. On a resume those are
the run's own, but on a FORK they are the PARENT's, so any run-level value the extractor carries as an attribute reads
the parent's until the trainer says otherwise. Two flags are of that class: the registry's ``resume_immutable`` rows
(``flag_registry.Klass.RESUME_IMMUTABLE``) -- ``belief_grad_mode`` (a mode a resume may MIGRATE) and ``oracle_reveal``
(a mode a diagnostic fork may CHANGE under its consent). Every other registry row is ``structural``: a mismatch is
refused by ``ModelVersion.check_compatible`` before the load returns, so a fork cannot differ there.

``belief_grad_mode`` was re-applied after every ``--model`` load (the 2026-07-21 incident) and ``oracle_reveal`` was
not, so ``metadata.json`` -- which copied the extractor's attribute -- recorded the PARENT's ``off`` for every knock-in
fork. This module is the ONE table of "which flags are re-applied, by which setter", and a test fails when a
``resume_immutable`` row exists with no setter here: a third such flag cannot be forgotten the same way.

The resolved value comes from the RESOLVED run config (``args``: the typed flag, else the checkpoint run's
``model_config.json``), the same source the env core is declared from and ``model_config.json`` is written from.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

#: registry flag name -> the extractor setter that applies the resolved value (``agents.model.extractor_api``).
SETTERS: Dict[str, str] = {
    "belief_grad_mode": "set_belief_grad_mode",
    "oracle_reveal": "set_oracle_reveal",
    # gen3_tau_fp32_alt_v1 (config v160): NOT resume-immutable (a fork may switch it, nothing is gated) but carried on the
    # extractor (its hypothesis builder) the same way, so a fork that inherits or types a solve its parent's zip does
    # not carry needs the same post-load application (`REAPPLIED_NUMERICS`).
    "tau_solve": "set_tau_solve",
}
#: Re-applied rows that are NOT `resume_immutable` (`reveal_provenance_test` pins SETTERS = those + this).
REAPPLIED_NUMERICS = frozenset({"tau_solve"})


def resolved_run_level(args: Any) -> Dict[str, Any]:
    """The resolved value of every re-applied flag, read off the trainer's namespace (``resolve_config`` has already
    filled an untyped flag from the checkpoint run's record). An unset ``oracle_reveal`` is ``off``, as the env core is
    declared (``rust_env_setup``: ``getattr(args, "oracle_reveal", None) or "off"``)."""
    out = {name: getattr(args, name) for name in SETTERS}
    out["oracle_reveal"] = out["oracle_reveal"] or "off"
    out["tau_solve"] = out["tau_solve"] or "fp32_alt"          # unset = the default solve (`_resolve` fills it)
    return out


def reapply_resume_immutable(extractor: Any, resolved: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Apply each resolved value to `extractor` through its setter. Returns ``{flag: (loaded value, applied value)}``
    for every flag (a no-op where they agree). A flag missing from `resolved` is an error, never skipped."""
    applied: Dict[str, Tuple[Any, Any]] = {}
    for name, setter in SETTERS.items():
        if name not in resolved:
            raise KeyError(f"reapply_resume_immutable: no resolved value for {name!r} (every resume-immutable flag "
                           f"is re-applied after a load; resolved = {sorted(resolved)})")
        before = getattr(extractor, name, None)
        getattr(extractor, setter)(resolved[name])
        applied[name] = (before, getattr(extractor, name, None))
    return applied
