"""THE BASELINE REGISTRY — ``designs/baselines.json``, and the ONE accessor over it.

A baseline is the thing a result is read AGAINST. This programme has half a dozen of them and,
until ``gen3_baselines_registry_v1``, not one was a first-class object:

===========================  =========================================================
what it was                  where it lived
===========================  =========================================================
"production"                 a hand-copied ``designs/production_config.json`` that
                             nothing consumes at launch, guarded by a NEWEST-RUN
                             heuristic that fires on whichever research arm trained
                             most recently
the untaught meter's         a string literal in
opponent + config            ``agents/training/untaught_meter.py``
the famine comparator        a sentence in one ledger entry
the curated TensorBoard set  decided by asking somebody
the parser's defaults        a near-bare model, and nothing said so
===========================  =========================================================

Each is a place where "what is the stable baseline?" is answered from memory. On 2026-09-06 that
cost two incidents in one day: a win-prob arm was launched from a design-doc command block WITHOUT
the production architecture surface (81 keys differed from the production config), and, separately,
the FOLD PARENT was nearly written into ``production_config.json`` as if it were the production
run. Neither is a coding error — both are the same missing object.

**So a baseline is now NAMED, and every consumer reads it by name.** ``designs/baselines.json``
records, per name: the run, an EXPLICIT checkpoint (a ``.zip`` or an ``@step`` — never a bare run
directory, so ``gen3_last_snapshot_resolution_v1``'s last-snapshot rule cannot silently move it),
the commit that run was pinned to, the config version and arch signature, the file's sha256, a
one-sentence purpose, the date it was set and the LEDGER ENTRY that set it.

Three rules make it a registry rather than a second place to be wrong:

1. **It is VALIDATED, by a test that runs in the routine suite.** Every referenced file exists (or
   the entry is marked ``era_checkout_only`` and carries the commit its weights are readable
   from), every sha matches, and every ``config_version`` / ``arch_signature`` is re-read from the
   run's OWN ``model_config.json`` rather than trusted.
2. **The ``production`` entry OWNS the mirror, and it declares HOW the mirror is CONSTRUCTED.**
   ``designs/production_config.json`` is not a copy of any one run: since 2026-09-06 it is gen-17's
   SURFACE migrated ``v97 → v109`` with an explicit 13-key critic override block
   (``designs/production_config.README.md``, ``CHANGELOG`` → *production_config_2026-09-06*). So the
   entry records the surface RUN, the mirror's schema version, and every deliberate override — and
   the check is that the mirror equals the run on every shared key **outside** that declared block.
   That replaces the newest-run heuristic, which compared against whatever arm was most recently
   written: on 2026-09-06 that was the mis-launched arm whose 38-flag argv had reverted 31
   architecture fields to their OFF defaults, and mirroring it would have redefined the production
   architecture BY OMISSION with every derived artifact agreeing.
3. **Changing a baseline is a PROCEDURE**: ``python -m main.baselines set <name> <run>[@step]
   --reason "<ledger entry title>"``. The tool rewrites the entry with a freshly computed
   sha/commit/version and PRINTS the ledger line to append. It never edits the ledger — the ledger
   is append-only and is the record of WHY, which no tool can author.

This module is deliberately **torch-free and offline**: it reads JSON and (for ``resolve``) the
run-spec choke point. ``models/`` lives only in the MAIN checkout, so every file-touching call
resolves through :func:`utils.paths.main_models_dir` and every caller must treat a ``None``
archive as a skip, never as a failure.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from utils.paths import main_models_dir, repo_path

#: The registry itself. One file, committed, human-readable, machine-validated.
REGISTRY_PATH = str(repo_path("designs", "baselines.json"))

#: The schema tag the file must carry. A file that does not is REFUSED rather than read
#: leniently — a registry read under the wrong schema is exactly the failure it exists to stop.
SCHEMA = "gen3_baselines_registry_v1"

#: Every field an entry must carry. ``notes`` / ``era_checkout_only`` / ``config_overrides`` /
#: ``floor_elo`` / ``pending`` / ``num_timesteps`` are optional.
REQUIRED_FIELDS = ("kind", "run", "checkpoint", "commit", "config_version", "arch_signature",
                   "purpose", "set_on", "set_by", "sha256")

#: Entry kinds. ``checkpoint`` names a model ``.zip`` (or an ``@step``); ``config`` names a
#: ``model_config.json`` a series of models is LOADED AGAINST, which is a baseline in exactly the
#: same sense and must be pinned in exactly the same way.
KINDS = ("checkpoint", "config")

_CHUNK = 1 << 20


class BaselineError(RuntimeError):
    """A registry name that does not exist, or a registry that cannot be read.

    Always names the registry path and what was available — the whole point of the object is that
    nobody has to remember, so its failure mode must not require remembering either.
    """


# --------------------------------------------------------------------------------------------
# The objects
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Baseline:
    """One named baseline, exactly as recorded. Nothing here is inferred at read time."""

    name: str
    kind: str
    run: str
    checkpoint: str
    commit: str
    config_version: int
    arch_signature: str
    purpose: str
    set_on: str
    set_by: str
    sha256: str
    notes: str = ""
    era_checkout_only: bool = False
    num_timesteps: Optional[int] = None
    floor_elo: Optional[float] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    #: ``production`` only: the repo-relative path of the CONSTRUCTED mirror this entry owns.
    #: Declared in the registry rather than hardcoded by each consumer, so :func:`production_config`
    #: and the ARCH-SURFACE guard read the same file the entry says it constructs.
    config_mirror: Optional[str] = None
    #: ``production`` only: the CONSTRUCTED mirror's own schema version, when it differs from the
    #: surface run's. ``None`` means "the mirror is a straight copy" and a key-set delta is then
    #: drift rather than a migration.
    config_mirror_version: Optional[int] = None
    pending: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- addressing

    @property
    def is_step_spec(self) -> bool:
        """``@<step>`` form — the run dir plus a step, rather than a file inside the run."""
        return self.checkpoint.startswith("@")

    @property
    def step(self) -> Optional[int]:
        return int(self.checkpoint[1:]) if self.is_step_spec else None

    @property
    def spec(self) -> str:
        """The run spec a consumer passes to the choke point — run-relative, resolved under
        ``models/``. EXPLICIT by construction: either ``<run>@<step>`` or ``<run>/<file>``, so the
        bare-directory rung (``latest.txt`` → highest checkpoint → …) is never reached and the
        file a name points at cannot move when a run gains a checkpoint."""
        return f"{self.run}{self.checkpoint}" if self.is_step_spec \
            else f"{self.run}/{self.checkpoint}"

    @property
    def rel_path(self) -> Optional[str]:
        """The run-relative FILE, when the entry names one (``None`` for an ``@step`` spec).

        This is what the archive-grooming keep-set needs: a registry-named file must survive every
        retention tier, and a tier plan is expressed in run-relative paths.
        """
        return None if self.is_step_spec else self.checkpoint

    def path(self) -> Optional[str]:
        """The absolute path of the named file, or ``None`` when there is no archive on this box.

        Does NOT check existence — :func:`validate` is what reports a missing file, with a reason.
        """
        models = main_models_dir()
        if models is None or self.rel_path is None:
            return None
        return os.path.join(str(models), self.run, self.rel_path)

    # ---------------------------------------------------------------- announcing

    def describe(self) -> str:
        """The one line EVERY consumer prints, so a reader always sees which run was meant.

        ``baseline <name> = <run>@<step> (set <date>, <ledger title>)`` — the step is the recorded
        ``num_timesteps`` where the entry carries one (so this works with no archive at all), and
        the checkpoint's file name otherwise.
        """
        if self.num_timesteps is not None:
            where = f"@{self.num_timesteps:,}"
        elif self.is_step_spec:
            where = self.checkpoint
        else:
            where = f"/{self.checkpoint}"
        return (f"baseline {self.name} = {self.run}{where} "
                f"(set {self.set_on}, {self.set_by})")

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind, "run": self.run, "checkpoint": self.checkpoint,
            "commit": self.commit, "config_version": self.config_version,
            "arch_signature": self.arch_signature, "num_timesteps": self.num_timesteps,
            "sha256": self.sha256, "purpose": self.purpose,
            "set_on": self.set_on, "set_by": self.set_by,
        }
        if self.notes:
            out["notes"] = self.notes
        if self.era_checkout_only:
            out["era_checkout_only"] = True
        if self.floor_elo is not None:
            out["floor_elo"] = self.floor_elo
        if self.config_overrides:
            out["config_overrides"] = dict(self.config_overrides)
        if self.config_mirror:
            out["config_mirror"] = self.config_mirror
        if self.config_mirror_version is not None:
            out["config_mirror_version"] = self.config_mirror_version
        if self.pending:
            out["pending"] = dict(self.pending)
        return out


@dataclass(frozen=True)
class BaselineList:
    """A named SET of baselines — e.g. the curated TensorBoard runs.

    Members are registry NAMES, never run strings: a list that named runs directly would be a
    second place the run identity lives, which is the class of defect this file exists to close.
    """

    name: str
    members: Sequence[str]
    purpose: str
    set_on: str
    set_by: str
    notes: str = ""

    def describe(self) -> str:
        return (f"baseline-list {self.name} = [{', '.join(self.members)}] "
                f"(set {self.set_on}, {self.set_by})")


# --------------------------------------------------------------------------------------------
# Loading + access
# --------------------------------------------------------------------------------------------

def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    """The raw registry document, schema-checked. Never cached — it is one small JSON, and a
    stale cache in a tool that EDITS the file would be its own defect class."""
    p = path or REGISTRY_PATH
    try:
        with open(p) as fh:
            doc = json.load(fh)
    except FileNotFoundError as exc:
        raise BaselineError(f"no baseline registry at {p!r} — this file is committed; a checkout "
                            "without it is broken, not merely unconfigured.") from exc
    except (OSError, ValueError) as exc:
        raise BaselineError(f"baseline registry {p!r} is unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise BaselineError(f"baseline registry {p!r}: top level is {type(doc).__name__}, not an "
                            "object.")
    if doc.get("schema") != SCHEMA:
        raise BaselineError(f"baseline registry {p!r} declares schema {doc.get('schema')!r}, "
                            f"expected {SCHEMA!r} — refusing to read it under the wrong schema.")
    return doc


def _entries(doc: Dict[str, Any]) -> Dict[str, Any]:
    b = doc.get("baselines")
    return b if isinstance(b, dict) else {}


def _lists(doc: Dict[str, Any]) -> Dict[str, Any]:
    li = doc.get("lists")
    return li if isinstance(li, dict) else {}


def names(path: Optional[str] = None) -> List[str]:
    """Every baseline name, sorted."""
    return sorted(_entries(load_registry(path)))


def list_names(path: Optional[str] = None) -> List[str]:
    """Every baseline-LIST name, sorted."""
    return sorted(_lists(load_registry(path)))


def is_name(name: str, path: Optional[str] = None) -> bool:
    """Is this string a registry name? Used by consumers that accept a NAME **or** a raw ref, so
    the two can never be confused: a name is a member of this set, everything else is a ref."""
    try:
        doc = load_registry(path)
    except BaselineError:
        return False
    return name in _entries(doc) or name in _lists(doc)


def _build(name: str, raw: Dict[str, Any]) -> Baseline:
    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise BaselineError(f"baseline {name!r} in {REGISTRY_PATH} is missing required field(s): "
                            f"{', '.join(missing)}")
    return Baseline(
        name=name,
        kind=str(raw["kind"]),
        run=str(raw["run"]),
        checkpoint=str(raw["checkpoint"]),
        commit=str(raw["commit"]),
        config_version=int(raw["config_version"]),
        arch_signature=str(raw["arch_signature"]),
        purpose=str(raw["purpose"]),
        set_on=str(raw["set_on"]),
        set_by=str(raw["set_by"]),
        sha256=str(raw["sha256"]),
        notes=str(raw.get("notes", "")),
        era_checkout_only=bool(raw.get("era_checkout_only", False)),
        num_timesteps=(None if raw.get("num_timesteps") is None
                       else int(raw["num_timesteps"])),
        floor_elo=(None if raw.get("floor_elo") is None else float(raw["floor_elo"])),
        config_overrides=dict(raw.get("config_overrides") or {}),
        config_mirror=(None if raw.get("config_mirror") is None
                       else str(raw["config_mirror"])),
        config_mirror_version=(None if raw.get("config_mirror_version") is None
                               else int(raw["config_mirror_version"])),
        pending=dict(raw.get("pending") or {}),
    )


def get(name: str, path: Optional[str] = None) -> Baseline:
    """THE accessor. ``baselines.get("production")``.

    Raises :class:`BaselineError` naming the registry path AND every available name — a missing
    baseline must never send the reader back to their memory, which is the failure mode the
    registry exists to remove.
    """
    doc = load_registry(path)
    entries = _entries(doc)
    if name not in entries:
        if name in _lists(doc):
            raise BaselineError(
                f"{name!r} is a baseline LIST, not a single baseline — call "
                f"agents.training.baselines.get_list({name!r}) instead "
                f"({path or REGISTRY_PATH}).")
        raise BaselineError(
            f"no baseline named {name!r} in {path or REGISTRY_PATH}. Available: "
            f"{', '.join(sorted(entries)) or '(none)'}"
            + (f" · lists: {', '.join(sorted(_lists(doc)))}" if _lists(doc) else ""))
    return _build(name, entries[name])


def get_list(name: str, path: Optional[str] = None) -> BaselineList:
    """A named SET of baselines, e.g. ``tb_curated``."""
    doc = load_registry(path)
    lists = _lists(doc)
    if name not in lists:
        raise BaselineError(
            f"no baseline list named {name!r} in {path or REGISTRY_PATH}. Available lists: "
            f"{', '.join(sorted(lists)) or '(none)'}")
    raw = lists[name]
    for key in ("members", "purpose", "set_on", "set_by"):
        if key not in raw:
            raise BaselineError(f"baseline list {name!r} is missing required field {key!r}")
    return BaselineList(name=name, members=list(raw["members"]), purpose=str(raw["purpose"]),
                        set_on=str(raw["set_on"]), set_by=str(raw["set_by"]),
                        notes=str(raw.get("notes", "")))


def spec(name: str, path: Optional[str] = None) -> str:
    """The run spec for a named baseline — what you pass to any flag that takes a model ref."""
    return get(name, path).spec


def describe(name: str, path: Optional[str] = None) -> str:
    """The one-line provenance every consumer prints."""
    return get(name, path).describe()


def announce(name: str, path: Optional[str] = None, *, prefix: str = "") -> str:
    """Print (and return) :meth:`Baseline.describe` — the line a consumer emits so the reader
    always sees WHICH RUN was meant, rather than a bare path they have to recognise."""
    line = prefix + describe(name, path)
    print(line)
    return line


# --------------------------------------------------------------------------------------------
# Resolution (the only part that touches models/)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedBaseline:
    """A named baseline, resolved to the file on this box, with the choke point's provenance."""

    baseline: Baseline
    zip_path: str
    config_path: str
    run_dir: str
    rung: str
    rule: str
    num_timesteps: Optional[int]

    def describe(self) -> str:
        steps = (f"{self.num_timesteps:,} steps" if self.num_timesteps is not None
                 else "steps unknown")
        return f"{self.baseline.describe()} -> {self.zip_path} @{steps} [rung={self.rung}]"


def candidate_paths(ref: str) -> List[str]:
    """``<run>/<file>`` and ``models/<run>/<file>`` both work, from a worktree too.

    ``models/`` is NOT committed and exists only in the MAIN checkout, so a run-relative spec is
    additionally tried under :func:`utils.paths.main_models_dir`, the accessor that reaches across
    from a linked worktree.
    """
    out = [ref]
    models = main_models_dir()
    if models is not None:
        base = ref.split("@", 1)[0]
        if not os.path.isabs(base) and not os.path.exists(base):
            out.append(os.path.join(str(models), ref))
            if ref.startswith("models/"):
                out.append(os.path.join(str(models.parent), ref))
    return out


def resolve(name: str, path: Optional[str] = None) -> ResolvedBaseline:
    """Resolve a named baseline to its file through the ONE choke point.

    ``fixed_opponent_pool.resolve_model_ref`` is CALLED, not re-implemented, so the file a name
    resolves to is by construction the file a launch would load. Because every registry spec is
    EXPLICIT (a ``.zip`` or an ``@step``), the rung is always ``explicit_zip`` / ``explicit_step``
    and the last-snapshot rungs are unreachable — which is the property that makes a name stable
    while its run keeps training.
    """
    b = get(name, path)
    if b.kind == "config":
        raise BaselineError(
            f"baseline {name!r} is kind={b.kind!r} — it names a model_config.json, not a model. "
            f"Use agents.training.baselines.config_path({name!r}).")
    from agents.training.fixed_opponent_pool import resolve_model_ref
    last: Exception = BaselineError(f"baseline {name!r}: nothing tried")
    for cand in candidate_paths(b.spec):
        try:
            r = resolve_model_ref(cand, warn=False)
        except (FileNotFoundError, ValueError) as exc:
            last = exc
            continue
        return ResolvedBaseline(baseline=b, zip_path=r.zip_path, config_path=r.config_path,
                                run_dir=r.run_dir, rung=r.rung, rule=r.rule,
                                num_timesteps=r.num_timesteps)
    raise BaselineError(
        f"baseline {name!r} ({b.spec}) does not resolve on this box: {last}"
        + ("  This entry is marked era_checkout_only — its weights are readable from commit "
           f"{b.commit} only." if b.era_checkout_only else ""))


def config_path(name: str, path: Optional[str] = None) -> str:
    """The absolute path of a ``kind="config"`` baseline (or a checkpoint entry's sibling config).

    Raises rather than returning a non-existent path, because a config that silently is not there
    becomes "each model resolved its own", which is a different measurement.
    """
    b = get(name, path)
    if b.kind == "config":
        for cand in candidate_paths(b.spec):
            if os.path.isfile(cand):
                return cand
        raise BaselineError(
            f"baseline {name!r} names {b.spec} but no such file is on this box"
            + (f" (models/ is {main_models_dir()})" if main_models_dir() else
               " — there is no models/ archive in this checkout."))
    return resolve(name, path).config_path


def run_dir(name: str, path: Optional[str] = None) -> Optional[str]:
    """The baseline's run directory on this box, or ``None`` with no archive."""
    models = main_models_dir()
    return None if models is None else os.path.join(str(models), get(name, path).run)


# --------------------------------------------------------------------------------------------
# The CONSTRUCTED production mirror — one reader
# --------------------------------------------------------------------------------------------

#: Where the ``production`` entry's mirror lives when the entry declares no ``config_mirror``.
#: A fallback, not the authority: the registry entry is.
DEFAULT_PRODUCTION_CONFIG = "designs/production_config.json"


def production_config_path(path: Optional[str] = None) -> str:
    """The absolute path of the CONSTRUCTED architecture mirror the ``production`` entry owns.

    Read this rather than hardcoding ``designs/production_config.json``. Every consumer that
    hardcodes it is a second opinion about what "production" is, and the registry exists because
    this project has twice discovered two such opinions disagreeing — the newest-run drift
    heuristic that would have blessed a 38-flag research argv, and the fold parent that was nearly
    written into the mirror as if it were the production surface.
    """
    rel = get("production", path).config_mirror or DEFAULT_PRODUCTION_CONFIG
    return rel if os.path.isabs(rel) else str(repo_path(*rel.split("/")))


def production_config(path: Optional[str] = None) -> Dict[str, Any]:
    """The mirror itself, as a dict. Raises :class:`BaselineError` naming the path when unreadable.

    A mirror that silently is not there would let the ARCH-SURFACE guard report "0 keys differ",
    which is precisely the false clean the guard exists to prevent.
    """
    p = production_config_path(path)
    doc = _read_json(p)
    if doc is None:
        raise BaselineError(
            f"the `production` baseline names {p} as its architecture mirror, but that file "
            f"cannot be read. Every derived architecture artifact keys on it — regenerate it, or "
            f"fix the entry's `config_mirror` in {path or REGISTRY_PATH}.")
    return dict(doc)


# --------------------------------------------------------------------------------------------
# What the retention policy must never delete
# --------------------------------------------------------------------------------------------

def protected_files(path: Optional[str] = None) -> Dict[str, List[str]]:
    """``{run_name: [run-relative file, …]}`` — every file the registry NAMES.

    The archive-grooming policy reads this: a registry-named checkpoint must be in EVERY tier's
    keep set, in every era. A retention tier that can delete the file a baseline names would make
    the baseline unresolvable — and the registry's whole claim is that a name still means
    something a year later.

    ``@step`` entries contribute both checkpoint layouts (``checkpoints/`` and the legacy run
    root), because which one exists is a property of the run's age, not of the baseline.
    """
    out: Dict[str, List[str]] = {}
    for name in names(path):
        b = get(name, path)
        rels = out.setdefault(b.run, [])
        if b.rel_path is not None:
            if b.rel_path not in rels:
                rels.append(b.rel_path)
        else:
            step = b.step
            for cand in (os.path.join("checkpoints", f"checkpoint_{step}_steps.zip"),
                         f"checkpoint_{step}_steps.zip"):
                if cand not in rels:
                    rels.append(cand)
    return out


def protected_runs(path: Optional[str] = None) -> List[str]:
    """Every run the registry names — the tier-1 REFERENCED set contributed by the registry."""
    return sorted(protected_files(path))


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------

@dataclass
class Finding:
    """One validation result. ``level`` is ``ok`` / ``warn`` / ``error``."""

    level: str
    name: str
    message: str

    def line(self) -> str:
        mark = {"ok": "✓", "warn": "⚠", "error": "✗"}.get(self.level, "?")
        return f"{mark} {self.name}: {self.message}"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(p: str) -> Optional[Dict[str, Any]]:
    try:
        with open(p) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def validate(path: Optional[str] = None, *, verify_sha: bool = True,
             production_config: Optional[str] = None) -> List[Finding]:
    """Every check the registry makes about itself. Returns findings; raises only on a registry
    that cannot be parsed at all.

    Two halves, deliberately separated by what they need:

    * **Structural** — schema, required fields, kinds, an EXPLICIT checkpoint, list members that
      resolve to names, the ``floor_elo``/``notes`` agreement. Needs no archive, so it runs
      everywhere including CI.
    * **Archive-backed** — the file exists, the sha matches, ``config_version`` /
      ``arch_signature`` / ``commit`` equal what the run itself records, and the ``production``
      entry's run config equals ``designs/production_config.json`` key-for-key. SKIPPED (one
      ``warn``, never an ``error``) when :func:`utils.paths.main_models_dir` is ``None``, because a
      fresh clone has no ``models/`` and an absent archive is not drift.
    """
    doc = load_registry(path)
    out: List[Finding] = []
    entries = _entries(doc)
    if not entries:
        out.append(Finding("error", "registry", "no 'baselines' object — the registry is empty."))
        return out

    models = main_models_dir()
    if models is None:
        out.append(Finding("warn", "registry",
                           "no models/ archive on this box — the file / sha / config-version "
                           "checks are SKIPPED. Structural checks still ran."))

    for name in sorted(entries):
        try:
            b = get(name, path)
        except BaselineError as exc:
            out.append(Finding("error", name, str(exc)))
            continue
        out.extend(_validate_entry(b, models))

    for lname in sorted(_lists(doc)):
        try:
            li = get_list(lname, path)
        except BaselineError as exc:
            out.append(Finding("error", lname, str(exc)))
            continue
        unknown = [m for m in li.members if m not in entries]
        if unknown:
            out.append(Finding("error", lname,
                               f"member(s) {', '.join(unknown)} are not baseline names. A list "
                               "names BASELINES, never runs — a run string here would be a second "
                               "place the identity lives."))
        elif not li.members:
            out.append(Finding("error", lname, "list is EMPTY — an empty curated set reads as "
                                               "'nothing is curated', which is never what is meant."))
        else:
            out.append(Finding("ok", lname, li.describe()))

    if models is not None:
        out.extend(_validate_production_mirror(path, production_config))
    if verify_sha and models is not None:
        out.extend(_validate_shas(path))
    return out


def _validate_entry(b: Baseline, models: Optional[Any]) -> List[Finding]:
    out: List[Finding] = []
    if b.kind not in KINDS:
        out.append(Finding("error", b.name, f"kind={b.kind!r} is not one of {KINDS}"))
    if not b.checkpoint or b.checkpoint.endswith("/"):
        out.append(Finding("error", b.name,
                           f"checkpoint={b.checkpoint!r} is a bare directory. A registry entry "
                           "must be EXPLICIT (a .zip, a .json, or @<step>) so the last-snapshot "
                           "rule cannot move what the name points at."))
    elif b.is_step_spec:
        if not b.checkpoint[1:].isdigit():
            out.append(Finding("error", b.name, f"checkpoint={b.checkpoint!r} is not @<step>"))
    elif b.kind == "checkpoint" and not b.checkpoint.endswith(".zip"):
        out.append(Finding("error", b.name,
                           f"kind='checkpoint' but checkpoint={b.checkpoint!r} is not a .zip"))
    elif b.kind == "config" and not b.checkpoint.endswith(".json"):
        out.append(Finding("error", b.name,
                           f"kind='config' but checkpoint={b.checkpoint!r} is not a .json"))
    if not b.set_by.strip():
        out.append(Finding("error", b.name, "set_by is empty — every baseline must name the "
                                            "ledger entry that set it."))
    if b.floor_elo is not None and str(int(b.floor_elo)) not in b.notes:
        out.append(Finding("error", b.name,
                           f"floor_elo={b.floor_elo} but the notes do not mention {int(b.floor_elo)} "
                           "— the machine-readable floor and the prose that explains it must not "
                           "be able to drift apart."))
    if models is None:
        return out

    run_path = os.path.join(str(models), b.run)
    if not os.path.isdir(run_path):
        level = "warn" if b.era_checkout_only else "error"
        out.append(Finding(level, b.name,
                           f"run dir {run_path} does not exist"
                           + (f" (era_checkout_only, commit {b.commit})" if b.era_checkout_only
                              else " — a baseline whose run was groomed away is not a baseline.")))
        return out
    fpath = b.path()
    if fpath is not None and not os.path.isfile(fpath):
        level = "warn" if b.era_checkout_only else "error"
        out.append(Finding(level, b.name, f"file {fpath} does not exist"))
        return out

    cfg = _read_json(os.path.join(run_path, "model_config.json"))
    if cfg is None:
        out.append(Finding("error", b.name, f"{run_path}/model_config.json is unreadable"))
    else:
        if int(cfg.get("config_version", -1)) != b.config_version:
            out.append(Finding("error", b.name,
                               f"config_version {b.config_version} but the run records "
                               f"{cfg.get('config_version')!r}"))
        if str(cfg.get("arch_signature")) != b.arch_signature:
            out.append(Finding("error", b.name,
                               f"arch_signature {b.arch_signature!r} but the run records "
                               f"{cfg.get('arch_signature')!r}"))
    meta = _read_json(os.path.join(run_path, "metadata.json"))
    if meta is not None:
        recorded = str(meta.get("git_hash") or "")
        if recorded and recorded != b.commit:
            out.append(Finding("warn", b.name,
                               f"commit {b.commit[:8]} but the run's metadata now records "
                               f"{recorded[:8]} — a restart across code, or a re-pin. Re-set the "
                               "entry if the newer commit is the one you mean."))
    out.append(Finding("ok", b.name, b.describe()))
    return out


def _validate_shas(path: Optional[str]) -> List[Finding]:
    out: List[Finding] = []
    for name in names(path):
        b = get(name, path)
        p = b.path()
        if p is None or not os.path.isfile(p):
            continue
        got = sha256_file(p)
        if got != b.sha256:
            out.append(Finding("error", name,
                               f"sha256 DRIFT — recorded {b.sha256[:12]}…, file is {got[:12]}…. "
                               "The bytes a name points at changed; re-set the entry with "
                               f"`python -m main.baselines set {name} …` naming the ledger entry "
                               "that authorises it."))
    return out


def _validate_production_mirror(path: Optional[str],
                                production_config: Optional[str]) -> List[Finding]:
    """``designs/production_config.json`` must equal the ``production`` baseline's run config.

    This REPLACES the newest-run heuristic. A heuristic that reads "the run whose directory was
    written most recently" answers a question nobody asked: on 2026-09-06 the most recent run was
    a research arm launched WITHOUT the production architecture surface, and the mirror it would
    have been compared against is not the mirror anybody intends.
    """
    out: List[Finding] = []
    try:
        b = get("production", path)
        # The entry declares its own mirror (`config_mirror`); the explicit argument still wins,
        # because a caller validating a candidate file must be able to name it.
        prod_path = production_config or production_config_path(path)
    except BaselineError as exc:
        return [Finding("error", "production", str(exc))]
    models = main_models_dir()
    if models is None:
        return out
    run_cfg = _read_json(os.path.join(str(models), b.run, "model_config.json"))
    mirror = _read_json(prod_path)
    if run_cfg is None:
        return [Finding("error", "production", f"cannot read {b.run}/model_config.json")]
    if mirror is None:
        return [Finding("error", "production", f"cannot read {prod_path}")]
    findings = compare_production(run_cfg, mirror, b)
    out.extend(Finding("error", "production", m) for m in findings)
    if not findings:
        built = (f"CONSTRUCTED from it (v{b.config_version} → v{b.config_mirror_version})"
                 if b.config_mirror_version is not None
                 and b.config_mirror_version != b.config_version else "mirrors it")
        out.append(Finding("ok", "production",
                           f"{os.path.basename(prod_path)} {built}: {len(mirror)} keys, surface "
                           f"run {b.run}"
                           + (f", {len(b.config_overrides)} declared override(s)"
                              if b.config_overrides else "")))
    return out


#: Never compared: the schema version is what the declared migration MOVES, and the signature is
#: checked on its own (against the run) by :func:`_validate_entry`.
_MIRROR_EXEMPT_KEYS = frozenset({"config_version"})


def compare_production(run_cfg: Dict[str, Any], mirror: Dict[str, Any],
                       b: Baseline) -> List[str]:
    """Does ``designs/production_config.json`` match the CONSTRUCTION the registry declares?

    Pure data in, messages out, so it is testable with no archive at all. The construction has
    three parts and each is checked as a separate claim:

    1. **The SURFACE.** Every key both sides carry, outside the declared override block, must be
       EQUAL. That is the whole architecture surface, and it is the half the 2026-09-06 incident
       destroyed: the mis-launched arm's argv reverted 31 architecture fields to their OFF defaults,
       and a mirror copied from it would have redefined production by omission.
    2. **The MIGRATION.** A key on one side only is legitimate exactly when the entry declares a
       ``config_mirror_version`` above the run's — a v97 → v109 migration adds the v98-v109 fields
       and drops what v108 deleted. With no declared migration a key-set delta is DRIFT and fails.
    3. **The OVERRIDES.** Every declared override must be present in the mirror carrying the value
       the entry declares — and, where the run carries that key too, must actually DIFFER from it.
       A stale override is not harmless: it exempts a key from the only check that guards it.
    """
    msgs: List[str] = []
    overrides = dict(b.config_overrides)
    exempt = set(overrides) | _MIRROR_EXEMPT_KEYS

    shared = (set(run_cfg) & set(mirror)) - exempt
    diffs = {k: {"run": run_cfg[k], "mirror": mirror[k]}
             for k in sorted(shared) if run_cfg[k] != mirror[k]}
    if diffs:
        msgs.append(
            f"designs/production_config.json disagrees with the `production` baseline's SURFACE "
            f"run ({b.run}) on {len(diffs)} shared field(s) that the registry does NOT declare as "
            f"an override:\n{json.dumps(diffs, indent=2)}\n"
            "Either re-derive the mirror from that run's config, or — if the difference is "
            "deliberate — declare it in the entry's config_overrides so it is a stated part of the "
            "construction rather than drift.")

    only_run = sorted(set(run_cfg) - set(mirror) - exempt)
    only_mirror = sorted(set(mirror) - set(run_cfg) - exempt)
    if only_run or only_mirror:
        migrating = (b.config_mirror_version is not None
                     and b.config_mirror_version > b.config_version)
        if not migrating:
            msgs.append(
                "the mirror and the `production` run's config do not carry the same KEYS "
                f"(run-only: {only_run or '—'}; mirror-only: {only_mirror or '—'}) and the entry "
                "declares no `config_mirror_version` above the run's. Without a declared migration "
                "a key-set delta is drift: re-sync the mirror, or record the version the mirror "
                "was constructed at.")

    for key, want in overrides.items():
        if key not in mirror:
            msgs.append(f"config_overrides declares {key!r}, but designs/production_config.json "
                        "does not carry that key at all.")
            continue
        if mirror[key] != want:
            msgs.append(f"config_overrides says {key!r} should be {want!r} in the mirror, but the "
                        f"mirror carries {mirror[key]!r}.")
        if key in run_cfg and run_cfg[key] == mirror[key]:
            msgs.append(f"config_overrides declares {key!r} as a deliberate difference, but the "
                        f"mirror and the surface run AGREE on it ({run_cfg[key]!r}). A stale "
                        "override exempts a key from the only check that guards it — delete it.")
    return msgs


def worst_level(findings: Sequence[Finding]) -> str:
    for level in ("error", "warn", "ok"):
        if any(f.level == level for f in findings):
            return level
    return "ok"
