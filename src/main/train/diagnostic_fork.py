"""THE DIAGNOSTIC-FORK CONSENT: the one way a FORK may change ``--oracle-reveal`` from its parent's recorded spec.

``--oracle-reveal`` is RESUME-IMMUTABLE (``ModelVersion.check_oracle_reveal``): a network trained under one
observation mode is a different function of the observation under another, so a resume that flips it is refused
with no escape hatch. The hidden-information study's KNOCK-IN forks (``designs/endstate/design_hidden_info_binding.md``
§8) need exactly such a flip — a reveal (``ki_moves`` = ``moves:seen@0.95``, …) on a FORK of a non-reveal checkpoint —
so this module is the explicit, RECORDED consent for it (``designs/endstate/design_information_levers.md`` §6):

* ``--diagnostic-fork-consent "<reason>"`` carries a REQUIRED non-empty reason. Without it every ``--model`` load whose
  requested spec differs in MEANING from the saved one is refused exactly as before (``check_oracle_reveal``).
* With it, the change is allowed ONLY on a genuine FORK: a ``--model`` that is not a checkpoint of the run directory
  being written (``main.train.fork_lr.is_same_run_checkpoint`` — the ONE predicate the lineage block, the arch
  surface, the disk preflight and ``main.launcher --dry-run``'s ``FORK of`` / ``RESTART of`` all read). A same-run
  resume that flips the spec stays refused, consent or not.
* The consent with NO spec change is refused as MEANINGLESS — except on a same-run RESTART of a fork whose own lineage
  block already records a consent: the launcher restarts the run with its original argv, which still carries the flag.
* NEVER a production lineage. A consented fork INTO a run directory the baselines registry names is refused here; a
  run whose lineage (its own block or any ancestor's) carries the stamp can never be registered as a baseline
  (``main.baselines set`` refuses it by name, ``main.baselines check`` flags one already there), so it can never become
  the ``production`` entry ``--arch production`` and the production mirror are resolved from. A registered PARENT is
  allowed: the knock-ins fork the adopted arm's own seed, and the child — not the parent — is the diagnostic.
* STAMPED: the decision is recorded in the child's IMMUTABLE ``metadata.json`` lineage block as ``diagnostic_fork``
  (schema, the reason, the date, the parent's recorded spec and the child's spec as ``InfoReveal.record()`` blocks, the
  parent checkpoint), read back through ``agents.training.lineage.diagnostic_fork_stamps`` by ``main.lineage``,
  ``main.sidecar_audit``, ``main.checkargs`` and the launch banner.

The fork arm (``--fork-fraction > 0``) stays refused with any non-``off`` spec (``combination_checks
.oracle_reveal_vs_fork_arm``: its successor rows are encoded without the reveal); a fork inherits ``fork_fraction``
from its parent's ``model_config.json`` like every unnamed flag.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: The stamp's schema (the lineage block's ``diagnostic_fork`` key).
SCHEMA = "gen3_diagnostic_fork_consent_v1"
#: The flag (its argparse ``dest`` is ``diagnostic_fork_consent``).
FLAG = "--diagnostic-fork-consent"


class DiagnosticForkRefused(ValueError):
    """A ``--diagnostic-fork-consent`` the launch must not honour (or a consented fork it must refuse), with why."""


def consent_reason(text: str) -> str:
    """The flag's argparse type: a REQUIRED non-empty reason (whitespace alone is no reason)."""
    import argparse

    reason = (text or "").strip()
    if not reason:
        raise argparse.ArgumentTypeError(f"{FLAG} needs a non-empty reason (what the diagnostic fork measures)")
    return reason


def _registry_runs() -> Dict[str, List[str]]:
    """``{run_name: [baseline names]}`` of every run the baselines registry names (entries and list members)."""
    from agents.training import baselines as reg

    out: Dict[str, List[str]] = {}
    doc = reg.load_registry()
    for name, raw in (doc.get("baselines") or {}).items():
        run = (raw or {}).get("run")
        if run:
            out.setdefault(str(run), []).append(name)
    return out


def registry_names_for_run(run_dir: str) -> List[str]:
    """The baseline names whose entry names ``run_dir``'s run (by its directory name)."""
    name = os.path.basename(os.path.normpath(run_dir or ""))
    return sorted(_registry_runs().get(name, []))


def decide(*, consent: Optional[str], model_path: Optional[str], model_dir: Optional[str], requested: Optional[str],
           saved: Optional[str] = None, now: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The consent's verdict for one launch. Returns the STAMP to record on a consented FORK whose spec changes, or
    ``None`` (no consent in play, or a restart of a consented fork); raises :class:`DiagnosticForkRefused` naming why.

    ``requested`` = the EFFECTIVE ``--oracle-reveal`` (after inheritance: a flagless fork inherits the parent's);
    ``saved`` = the parent's recorded spec (read from ``model_path``'s run config when not given)."""
    from agents.model import oracle_reveal as OR
    from main.train.fork_lr import is_same_run_checkpoint

    if consent is None:
        return None
    if not str(consent).strip():
        raise DiagnosticForkRefused(f"{FLAG} needs a non-empty reason")
    if not model_path:
        raise DiagnosticForkRefused(
            f"{FLAG} without --model is meaningless: a FRESH run chooses its own --oracle-reveal; the consent exists "
            "only to let a FORK change its parent's recorded spec")
    saved_label = OR.label(saved) if saved is not None else OR.recorded_oracle_reveal(model_path)
    req_label = OR.label(requested or "off")
    changed = not OR.same(saved_label, req_label)
    if model_dir and is_same_run_checkpoint(model_path, model_dir):
        if changed:
            raise DiagnosticForkRefused(
                f"{model_path} is a checkpoint of the run being written ({model_dir}): a SAME-RUN resume can never "
                f"change --oracle-reveal ({saved_label} -> {req_label}), consent or not — the run's rows would change "
                "meaning mid-run. Fork into a NEW run directory (--run-name) to change it.")
        from agents.training.lineage import diagnostic_fork_stamps

        own = [s for s in diagnostic_fork_stamps(model_dir) if s.get("self")]
        if own:
            return None     # the launcher restarting a consented fork with its original argv
        raise DiagnosticForkRefused(
            f"{FLAG} on a same-run resume of {model_dir} with no --oracle-reveal change is meaningless (the run's "
            f"spec stays {saved_label}) and its lineage records no consent to restart under — drop the flag")
    if not changed:
        raise DiagnosticForkRefused(
            f"{FLAG} with no --oracle-reveal change is meaningless: the fork's spec {req_label} means the same as the "
            f"parent's recorded {saved_label} — drop the flag, or name the reveal the diagnostic fork is for")
    names = registry_names_for_run(model_dir or "")
    if names:
        raise DiagnosticForkRefused(
            f"a diagnostic fork INTO {model_dir} is refused: the baselines registry names that run ({', '.join(names)}), "
            "and a run carrying the diagnostic-fork stamp can never be a PRODUCTION lineage — fork into a new run name")
    return {
        "schema": SCHEMA,
        "reason": str(consent).strip(),
        "date": now or datetime.now(timezone.utc).isoformat(),
        "parent_model": os.path.abspath(model_path),
        "parent_spec": OR.record(saved_label),
        "child_spec": OR.record(req_label),
    }


def decide_for_args(args: Any, model_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """:func:`decide` over the trainer's resolved namespace."""
    return decide(consent=getattr(args, "diagnostic_fork_consent", None), model_path=getattr(args, "model", None),
                  model_dir=model_dir, requested=getattr(args, "oracle_reveal", None))


def enforce(args: Any, model_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """The LAUNCH-path wrapper (``train_rl_agent.main``, the moment ``model_dir`` is known and BEFORE it is created, so
    a refusal leaves nothing behind): the stamp, stored on ``args._diagnostic_fork`` for the loader's gate and the
    lineage block; ``os._exit(FATAL_CONFIG)`` on a refusal (the launcher must not restart into it)."""
    try:
        stamp = decide_for_args(args, model_dir)
    except DiagnosticForkRefused as e:
        import sys

        from main.exit_codes import TrainExitCode

        print(f"\n🛑 [DiagnosticFork] FATAL: {e}", flush=True)
        sys.stdout.flush()
        os._exit(int(TrainExitCode.FATAL_CONFIG))
    args._diagnostic_fork = stamp
    return stamp


def describe(stamp: Dict[str, Any]) -> str:
    """One line for the banner / checkargs / the lineage readers."""
    p, c = stamp.get("parent_spec") or {}, stamp.get("child_spec") or {}
    return (f"--oracle-reveal {p.get('label')} -> {c.get('label')} (canonical {c.get('canonical')}, hash "
            f"{c.get('hash')}), consented {str(stamp.get('date') or '?')[:10]}: \"{stamp.get('reason')}\"")
