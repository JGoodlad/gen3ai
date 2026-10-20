"""`--grow FIELD=VALUE` — a FORK's DECLARED, function-preserving growth (B-GROW, `gen3_grow_fork_v1`).

The model side (what grows, how it starts as the identity, the optimizer rule, the fork-time gate) is
`agents.model.growth`; this module is the LAUNCH side, the one place every surface reads it from (the trainer's
`resolve_config` and `train_rl_agent`, `main.checkargs`, `main.launcher --dry-run`):

* **The flag.** ``--grow FIELD=VALUE``, repeatable, one per field. ``VALUE`` is the CHILD's value; the PARENT's is read
  from the ``--model`` checkpoint's ``model_config.json``. Only `growth.GROWABLE` fields parse, and only BUILT ones
  (``trunk_layers`` today; ``trunk_ffn_by_kind`` / ``opp_intent_grad_mode`` are refused naming B-KFFN / B-INT).
* **The namespace** (:func:`apply_to_namespace`, after the inheritance sweep). Each declared field is SET to its child
  value — so the child builds, version-checks and records the grown architecture. Refused: ``--grow`` without
  ``--model``, a parent config that cannot be read, a field also TYPED by its own flag (``--trunk-layers``: name the
  depth once), and a shrink.
* **The decision** (:func:`decide`, once the run dir is known). Only a genuine FORK grows
  (`main.train.fork_lr.is_same_run_checkpoint`, the one predicate). A fork whose parent already records every child
  value is refused as MEANINGLESS; a same-run resume never grows. A same-run RESTART of a grown fork (the launcher
  re-launches the original argv, which still says ``--grow``) is a no-op when the run's OWN lineage records the
  growth. The verdict is a STAMP: ``{"schema", "parent_model", "date", "fields": {f: {"parent", "child"}},
  "optimizer"}``; the load adds the fork-time gate's report to it.
* **The records.** The stamp is the ``grow`` key of the child's IMMUTABLE ``metadata.json`` lineage block, and the
  ``growth`` field of its ``model_config.json`` (config v159; a same-run restart keeps the recorded one).
* **The pool** (:func:`grow_seeded_pool`). A fork's self-play pool is seeded from its parent's (`pool_seed`), whose
  snapshots are at the PARENT's architecture, and the pool is single-architecture by construction (its arch record,
  the inference service's slot groups). So each seeded snapshot is grown the same way (strict load at the parent's
  architecture, the same identity rounds, the same fork-time gate) and the pool's arch record becomes the child's:
  every opponent plays exactly the function its parent snapshot played.
"""
from __future__ import annotations

import argparse
import os
import types
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: The flag (its argparse ``dest`` is ``grow``: a list of ``(field, value)``).
FLAG = "--grow"


class GrowForkRefused(ValueError):
    """A ``--grow`` the launch must refuse, with why."""


def grow_spec(text: str) -> Tuple[str, Any]:
    """The flag's argparse type: ``FIELD=VALUE`` -> ``(field, typed child value)``."""
    from agents.model.growth import GrowthError, parse_value

    field, sep, value = str(text or "").partition("=")
    field = field.strip().replace("-", "_")
    if not sep or not field or not value.strip():
        raise argparse.ArgumentTypeError(f"{FLAG} takes FIELD=VALUE (e.g. {FLAG} trunk_layers=4), got {text!r}")
    try:
        return field, parse_value(field, value)
    except GrowthError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def declared(ns: Any) -> Dict[str, Any]:
    """``{field: child value}`` from ``ns.grow`` (empty without the flag); a field named twice is refused."""
    out: Dict[str, Any] = {}
    for field, value in getattr(ns, "grow", None) or []:
        if field in out:
            raise GrowForkRefused(f"{FLAG} names {field} twice ({out[field]!r}, {value!r}) — name each field once")
        out[field] = value
    return out


def _cli_flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def apply_to_namespace(ns: Any, saved: Any) -> Dict[str, Tuple[Any, Any]]:
    """Set every declared field to its CHILD value on ``ns`` (after the inheritance sweep put the parent's there).
    Returns ``{field: (parent, child)}`` (also ``ns._grow_declared``); ``{}`` without the flag. ``saved`` is the
    ``--model`` checkpoint's recorded `ModelVersion` (None = unreadable)."""
    from agents.model.growth import GrowthError, validate_step

    d = declared(ns)
    ns._grow_declared = {}
    if not d:
        return {}
    if not getattr(ns, "model", None):
        raise GrowForkRefused(f"{FLAG} without --model is meaningless: a FRESH run chooses its own architecture "
                              f"(type {', '.join(_cli_flag(f) for f in d)} instead)")
    if saved is None:
        raise GrowForkRefused(f"{FLAG}: the --model checkpoint's model_config.json could not be read, so the parent's "
                              "value of a grown field is unknown — refused rather than guessed")
    typed = getattr(ns, "_explicit_flags", frozenset())
    out: Dict[str, Tuple[Any, Any]] = {}
    for field, child in d.items():
        if field in typed:
            raise GrowForkRefused(
                f"{FLAG} {field}={child} and {_cli_flag(field)} {getattr(ns, field)!r} are both typed — name the "
                f"child's value once ({FLAG} {field}={child}); the parent's comes from its model_config.json")
        parent = getattr(saved, field)
        try:
            validate_step(field, parent, child)
        except GrowthError as e:
            raise GrowForkRefused(str(e)) from None
        setattr(ns, field, child)
        out[field] = (parent, child)
    ns._grow_declared = out
    return out


def growth_fields(ns: Any) -> Dict[str, Dict[str, Any]]:
    """``{field: {"parent": p, "child": c}}`` for every declared field that CHANGES (``ns._grow_declared``, set by
    :func:`apply_to_namespace`) — the lineage stamp's ``fields`` block and the argument of
    ``ModelVersion.check_compatible_grown``, for the trainer's load and `main.train.fork_arch` alike."""
    return {f: {"parent": p, "child": c} for f, (p, c) in (getattr(ns, "_grow_declared", None) or {}).items()
            if p != c}


def _own_stamp(model_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    from agents.training.lineage import read_block

    stamp = (read_block(model_dir or "") or {}).get("grow")
    return stamp if isinstance(stamp, dict) else None


def decide(*, grown: Dict[str, Tuple[Any, Any]], model_path: Optional[str], model_dir: Optional[str],
           now: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The verdict for one launch: the STAMP of a genuine growth, ``None`` (no ``--grow``, or a restart of a grown
    fork), or `GrowForkRefused`. ``grown`` is :func:`apply_to_namespace`'s ``{field: (parent, child)}``."""
    from agents.model.growth import SCHEMA, optimizer_rule
    from main.train.fork_lr import is_same_run_checkpoint

    if not grown:
        return None
    changes = {f: pc for f, pc in grown.items() if pc[0] != pc[1]}
    if model_path and model_dir and is_same_run_checkpoint(model_path, model_dir):
        if changes:
            raise GrowForkRefused(
                f"{model_path} is a checkpoint of the run being written ({model_dir}): a SAME-RUN resume never grows "
                f"({', '.join(f'{f} {p} -> {c}' for f, (p, c) in changes.items())}) — the run's own checkpoints "
                "would change architecture mid-run. Fork into a NEW run directory (--run-name) to grow it.")
        if _own_stamp(model_dir):
            return None      # the launcher restarting a grown fork with its original argv
        raise GrowForkRefused(
            f"{FLAG} on a same-run resume of {model_dir} grows nothing (it already records "
            f"{', '.join(f'{f}={c}' for f, (_p, c) in grown.items())}) and its lineage records no growth to restart "
            "under — drop the flag")
    same = [f for f, (p, c) in grown.items() if p == c]
    if same:
        raise GrowForkRefused(
            f"{FLAG} {', '.join(f'{f}={grown[f][1]}' for f in same)} grows nothing: the parent already records that "
            "value — drop it (a plain fork inherits it)")
    return {
        "schema": SCHEMA,
        "date": now or datetime.now(timezone.utc).isoformat(),
        "parent_model": os.path.abspath(model_path or ""),
        "fields": growth_fields(types.SimpleNamespace(_grow_declared=grown)),
        "optimizer": optimizer_rule(changes),
    }


def enforce(args: Any, model_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """The LAUNCH-path wrapper (``train_rl_agent.main``, once ``model_dir`` is known, before it is created): sets
    ``args._grow_fork`` (the stamp, or None) and ``args._grow_record`` (what ``model_config.json``'s ``growth`` records:
    this fork's stamp, the run's own recorded block on a same-run restart, else None); ``os._exit(FATAL_CONFIG)`` on
    a refusal (the launcher must not restart into it)."""
    from main.train.fork_lr import is_same_run_checkpoint

    model = getattr(args, "model", None)
    try:
        stamp = decide(grown=getattr(args, "_grow_declared", None) or {}, model_path=model, model_dir=model_dir)
    except GrowForkRefused as e:
        import sys

        from main.exit_codes import TrainExitCode

        print(f"\n🛑 [GrowFork] FATAL: {e}", flush=True)
        sys.stdout.flush()
        os._exit(int(TrainExitCode.FATAL_CONFIG))
    args._grow_fork = stamp
    record = None
    if stamp is None and model and model_dir and is_same_run_checkpoint(model, model_dir):
        from main.train.checkpoint_state import _load_saved_version
        saved = _load_saved_version(model)
        record = getattr(saved, "growth", None) if saved is not None else None
    args._grow_record = record
    return stamp


def describe(stamp: Dict[str, Any]) -> str:
    """One line for the banner / checkargs / the dry run / the lineage readers."""
    fields = stamp.get("fields") or {}
    parts = ", ".join(f"{f} {v.get('parent')!r} -> {v.get('child')!r}" for f, v in fields.items())
    fc = stamp.get("function_check") or {}
    proof = (f"; the parent's function bit for bit on {fc.get('rows')} real rows (fp32 eager CPU)"
             if fc.get("bit_identical") else "")
    return f"--grow {parts} from {os.path.basename(str(stamp.get('parent_model') or '?'))}{proof}"


def describe_declared(grown: Dict[str, Tuple[Any, Any]]) -> str:
    return ", ".join(f"{f} {p!r} -> {c!r}" for f, (p, c) in grown.items())


# ------------------------------------------------------------------------------------------ the seeded pool
def _snapshot_zips(pool_dir: str) -> List[str]:
    from agents.training.pool_seed import snapshot_zips
    return snapshot_zips(pool_dir)


def grow_seeded_pool(pool_dir: str, fields: Dict[str, Dict[str, Any]], child_version: Any, *,
                     emit_fn: Any = None) -> Optional[Dict[str, Any]]:
    """Grow a fork's SEEDED pool to the child's architecture (module docstring, "The pool"). No-op (None) when the
    pool has no arch record or its record already holds every child value (grown before: an idempotent relaunch).
    Each snapshot: a STRICT load at the record's (parent's) architecture on the CPU, `growth.apply_growth` (the same
    rounds, the same optimizer rule, the same bit-identity gate), a save to a temporary file and an atomic replace. A
    snapshot whose every declared field is already BUILT at its child value (each row's ``current``; a relaunch that
    died mid-pool) is left alone. The record is rewritten LAST,
    as the child version's JSON — byte-identical to what the run's `SnapshotPool._record_arch` compares. Returns
    ``{"fields", "snapshots", "skipped"}``, also added to ``pool_seed.json`` as ``grown``."""
    from agents.model.growth import GROWABLE, GrowthError, apply_growth
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import load_model_snapshot
    from agents.training.pool_seed import read_seed_record, write_seed_record

    cfg = os.path.join(pool_dir, "model_config.json")
    if not os.path.isfile(cfg):
        return None
    recorded = ModelVersion.from_json_file(cfg)
    if all(getattr(recorded, f) == step["child"] for f, step in fields.items()):
        return None
    for f, step in fields.items():
        if getattr(recorded, f) != step["parent"]:
            raise GrowthError(f"seeded pool {pool_dir}: its arch record says {f}={getattr(recorded, f)!r}, neither the "
                              f"declared parent {step['parent']!r} nor the child {step['child']!r}")
    grown: List[str] = []
    skipped: List[str] = []
    for z in _snapshot_zips(pool_dir):
        model = load_model_snapshot(z, env=None, current_version=recorded, device="cpu")
        if all(GROWABLE[f].current(model.policy) == step["child"]                   # type: ignore[misc]
               for f, step in fields.items()):
            skipped.append(os.path.basename(z))
            continue
        apply_growth(model, fields)
        tmp = os.path.join(pool_dir, ".growing_" + os.path.basename(z)[:-4])   # outside the pool's glob
        model.save(tmp)                                # SB3 appends .zip
        os.replace(tmp + ".zip", z)
        grown.append(os.path.basename(z))
        if emit_fn is not None:
            emit_fn(f"🌱 [GrowFork] pool snapshot grown ({describe_declared({f: (s['parent'], s['child']) for f, s in fields.items()})}), "
                    f"function bit-identical: {os.path.basename(z)}")
    with open(cfg, "w") as fh:
        fh.write(child_version.to_json())
    summary = {"fields": fields, "snapshots": grown, "skipped": skipped}
    rec = read_seed_record(pool_dir)
    if rec is not None:
        rec["grown"] = summary
        write_seed_record(pool_dir, rec)
    return summary


def pool_lines(stamp: Optional[Dict[str, Any]], *, self_play: bool) -> Iterable[str]:
    """The dry run's pool line for a grow-fork (only with --self-play: no pool is seeded otherwise)."""
    if stamp and self_play:
        yield ("the self-play pool seeded from the parent is GROWN the same way at the child's start (each snapshot: "
               "strict load, identity rounds, the bit-identity gate), and its arch record becomes the child's")
