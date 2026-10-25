"""The READ SPEC: what a registered read plays, written once before any counted game (``designs/tools/reads.md`` §2).

A spec is ONE JSON file (schema :data:`SCHEMA`), committed with the read's PREREG. It names:

* ``models`` -- ``{label: {arm, seed, ref}}``; ``ref`` is a :class:`ModelRef` (``pinned``: a frozen file and its
  sha256; ``final``: a run's ``final_model.zip`` under the SETTLED rule of :mod:`main.reads.models`);
* ``anchor`` -- ``{group: {opponent, teamsets, unit_seeds, models}}``: the external-anchor cells, played as
  :mod:`main.x35.kakuna` units on the short lane;
* ``s`` -- ``{group: {refpop, refpop_content_sha256, panel, pairs, batch_pairs, family, models, regime_id}}``: S
  against a frozen reference population R, played on the heavy slots;
* ``decision`` -- ``{rule, params}``: a rule of :data:`main.reads.rules.RULES` and its registered parameters;
* ``pin`` (optional) -- the commit the PLAYING checkout must be at (a run from another commit is refused).

:func:`load` validates every field and refuses anything else (a typo is a refusal, never a default). Pure; no file
but the spec itself is read here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SCHEMA = "gen3_read_spec_v1"
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
#: The model-reference kinds (``main.reads.models`` owns their rules).
REF_KINDS = ("pinned", "final")
#: The anchor opponents a group may name: the cells :mod:`main.x35.kakuna` plays (its committed configs).
ANCHOR_OPPONENTS = ("metamon:Kakuna",)


class SpecError(ValueError):
    """A spec that is not a spec (named field, named problem)."""


@dataclass(frozen=True)
class ModelRef:
    """Where a model's bytes come from. ``pinned``: ``path`` + ``sha256`` (the file must hash to it). ``final``: the
    run dir ``run``; its ``final_model.zip`` once SETTLED (step >= ``min_step``, untouched ``quiet_s`` seconds, and,
    when named, ``done_marker``'s line present). A ``final`` NEVER falls back to a checkpoint (x26c FINDING 2)."""

    kind: str
    path: Optional[str] = None
    sha256: Optional[str] = None
    run: Optional[str] = None
    min_step: Optional[int] = None
    quiet_s: int = 300
    done_marker: Optional[Tuple[str, str]] = None   # (file, the exact line substring)


@dataclass(frozen=True)
class Model:
    label: str
    arm: str
    seed: int
    ref: ModelRef


@dataclass(frozen=True)
class AnchorGroup:
    name: str
    opponent: str
    teamsets: Tuple[str, ...]
    unit_seeds: Tuple[int, ...]
    models: Tuple[str, ...]


@dataclass(frozen=True)
class SGroup:
    name: str
    refpop: str
    refpop_content_sha256: str
    panel: Optional[str]
    pairs: int
    batch_pairs: int
    family: str
    models: Tuple[str, ...]
    regime_id: Optional[str] = None   # the regime the read registers; assemble refuses another

    def request(self, label: str) -> str:
        """The eval-ledger request of ``label``'s cells in this group (one request per node, the resume grain)."""
        return f"{self.name}_{label}_b{self.batch_pairs}"


@dataclass(frozen=True)
class ReadSpec:
    read: str
    models: Dict[str, Model]
    anchor: Dict[str, AnchorGroup]
    s: Dict[str, SGroup]
    rule: str
    params: Dict[str, Any]
    prereg: Optional[str] = None
    pin: Optional[str] = None
    sha256: str = ""                  # of the spec file's bytes
    raw: Dict[str, Any] = field(default_factory=dict)


def _need(d: Mapping[str, Any], key: str, where: str, typ: Any) -> Any:
    if key not in d:
        raise SpecError(f"{where}: missing {key!r}")
    v = d[key]
    if typ is int and isinstance(v, bool) or not isinstance(v, typ):
        raise SpecError(f"{where}.{key}: {v!r} is not {getattr(typ, '__name__', typ)}")
    return v


def _only(d: Mapping[str, Any], allowed: Tuple[str, ...], where: str) -> None:
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise SpecError(f"{where}: unknown field(s) {extra} (allowed {sorted(allowed)})")


def _label(x: Any, where: str) -> str:
    if not isinstance(x, str) or not _LABEL.match(x):
        raise SpecError(f"{where}: label {x!r} must match {_LABEL.pattern}")
    return x


def parse_unit_seeds(spec: Any, where: str) -> Tuple[int, ...]:
    """``"0-19"`` / ``"0,1,2"`` / ``[0, 1]`` -> sorted distinct seeds (the kakuna runner's grammar)."""
    from main.x35.kakuna import KakunaError, parse_unit_seeds as _p

    try:
        return _p(",".join(str(s) for s in spec) if isinstance(spec, list) else str(spec))
    except KakunaError as exc:
        raise SpecError(f"{where}: {exc}") from None


def _ref(d: Any, where: str) -> ModelRef:
    if not isinstance(d, dict):
        raise SpecError(f"{where}: an object")
    kind = _need(d, "kind", where, str)
    if kind == "pinned":
        _only(d, ("kind", "path", "sha256"), where)
        sha = _need(d, "sha256", where, str)
        if not _SHA256.match(sha):
            raise SpecError(f"{where}.sha256: 64 lowercase hex")
        return ModelRef(kind=kind, path=_need(d, "path", where, str), sha256=sha)
    if kind == "final":
        _only(d, ("kind", "run", "min_step", "quiet_s", "done_marker"), where)
        q = d.get("quiet_s", 300)
        if isinstance(q, bool) or not isinstance(q, int) or q < 60:
            raise SpecError(f"{where}.quiet_s: an int >= 60 (the end chain's write must be over)")
        dm = d.get("done_marker")
        if dm is not None and not (isinstance(dm, list) and len(dm) == 2 and all(isinstance(x, str) and x for x in dm)):
            raise SpecError(f"{where}.done_marker: [file, line substring]")
        ms = _need(d, "min_step", where, int)
        if ms <= 0:
            raise SpecError(f"{where}.min_step: > 0")
        return ModelRef(kind=kind, run=_need(d, "run", where, str), min_step=ms, quiet_s=q,
                        done_marker=(dm[0], dm[1]) if dm else None)
    raise SpecError(f"{where}.kind: {kind!r} is none of {REF_KINDS}")


def _group_models(d: Mapping[str, Any], where: str, models: Mapping[str, Model]) -> Tuple[str, ...]:
    ms = _need(d, "models", where, list)
    if not ms or len(set(ms)) != len(ms):
        raise SpecError(f"{where}.models: a non-empty list without repeats")
    for m in ms:
        if m not in models:
            raise SpecError(f"{where}.models: {m!r} is not a declared model")
    return tuple(ms)


def parse(doc: Mapping[str, Any], *, sha256: str = "") -> ReadSpec:
    """Validate a spec document (module docstring); every problem is a :class:`SpecError`."""
    from main.reads.rules import RULES

    _only(doc, ("schema", "read", "prereg", "pin", "models", "anchor", "s", "decision", "note"), "spec")
    if doc.get("schema") != SCHEMA:
        raise SpecError(f"spec.schema: {doc.get('schema')!r} is not {SCHEMA!r}")
    read = _need(doc, "read", "spec", str)
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$", read):
        raise SpecError(f"spec.read: {read!r} is not a read id")
    pin = doc.get("pin")
    if pin is not None and not (isinstance(pin, str) and _COMMIT.match(pin)):
        raise SpecError(f"spec.pin: {pin!r} is not a commit sha")
    models: Dict[str, Model] = {}
    for lab, m in _need(doc, "models", "spec", dict).items():
        w = f"models.{_label(lab, 'models')}"
        if not isinstance(m, dict):
            raise SpecError(f"{w}: an object")
        _only(m, ("arm", "seed", "ref"), w)
        models[lab] = Model(label=lab, arm=_need(m, "arm", w, str), seed=_need(m, "seed", w, int),
                            ref=_ref(m.get("ref"), f"{w}.ref"))
    if not models:
        raise SpecError("spec.models: none declared")
    anchor: Dict[str, AnchorGroup] = {}
    for g, d in (doc.get("anchor") or {}).items():
        w = f"anchor.{_label(g, 'anchor')}"
        if not isinstance(d, dict):
            raise SpecError(f"{w}: an object")
        _only(d, ("opponent", "teamsets", "unit_seeds", "models"), w)
        opp = _need(d, "opponent", w, str)
        if opp not in ANCHOR_OPPONENTS:
            raise SpecError(f"{w}.opponent: {opp!r} is none of {ANCHOR_OPPONENTS} (the kakuna runner's cells)")
        from main.x35.kakuna import KakunaError, parse_teamsets

        try:
            ts = parse_teamsets(",".join(_need(d, "teamsets", w, list)))
        except KakunaError as exc:
            raise SpecError(f"{w}.teamsets: {exc}") from None
        anchor[g] = AnchorGroup(name=g, opponent=opp, teamsets=ts,
                                unit_seeds=parse_unit_seeds(_need(d, "unit_seeds", w, (str, list)), f"{w}.unit_seeds"),
                                models=_group_models(d, w, models))
    s: Dict[str, SGroup] = {}
    families: Dict[str, str] = {}
    for g, d in (doc.get("s") or {}).items():
        w = f"s.{_label(g, 's')}"
        if not isinstance(d, dict):
            raise SpecError(f"{w}: an object")
        _only(d, ("refpop", "refpop_content_sha256", "panel", "pairs", "batch_pairs", "family", "models",
                  "regime_id"), w)
        rc = _need(d, "refpop_content_sha256", w, str)
        if not _SHA256.match(rc):
            raise SpecError(f"{w}.refpop_content_sha256: 64 lowercase hex (the frozen R's content_sha256)")
        panel = d.get("panel")
        if panel is not None and not isinstance(panel, str):
            raise SpecError(f"{w}.panel: a registered panel name or null (the eval team builder)")
        pairs, bp = _need(d, "pairs", w, int), _need(d, "batch_pairs", w, int)
        if pairs <= 0 or bp <= 0 or pairs % bp:
            raise SpecError(f"{w}: pairs {pairs} must be a positive multiple of batch_pairs {bp}")
        fam = _need(d, "family", w, str)
        if fam in families:
            # one family holds ONE regime (eval_ledger: a family read never mixes regimes)
            raise SpecError(f"{w}.family: {fam!r} is also group {families[fam]!r}'s; one family per S group")
        families[fam] = g
        rid = d.get("regime_id")
        if rid is not None and not (isinstance(rid, str) and re.match(r"^[0-9a-f]{16}$", rid)):
            raise SpecError(f"{w}.regime_id: 16 lowercase hex (an eval-ledger regime id) or absent")
        s[g] = SGroup(name=g, refpop=_need(d, "refpop", w, str), refpop_content_sha256=rc, panel=panel, pairs=pairs,
                      batch_pairs=bp, family=fam, models=_group_models(d, w, models), regime_id=rid)
    if not anchor and not s:
        raise SpecError("spec: neither an anchor nor an S group: nothing to play")
    dec = _need(doc, "decision", "spec", dict)
    _only(dec, ("rule", "params"), "decision")
    rule = _need(dec, "rule", "decision", str)
    if rule not in RULES:
        raise SpecError(f"decision.rule: {rule!r} is none of {sorted(RULES)}")
    params = dict(dec.get("params") or {})
    probs = RULES[rule].check_params(params, models=models, anchor=anchor, s=s)
    if probs:
        raise SpecError(f"decision.params ({rule}): {probs}")
    prereg = doc.get("prereg")
    if prereg is not None and not isinstance(prereg, str):
        raise SpecError("spec.prereg: a path")
    return ReadSpec(read=read, models=models, anchor=anchor, s=s, rule=rule, params=params, prereg=prereg, pin=pin,
                    sha256=sha256, raw=dict(doc))


def load(path: "str | Path") -> ReadSpec:
    b = Path(path).read_bytes()
    try:
        doc = json.loads(b)
    except ValueError as exc:
        raise SpecError(f"{path}: not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: not a JSON object")
    return parse(doc, sha256=hashlib.sha256(b).hexdigest())


def model_groups(spec: ReadSpec) -> List[Tuple[str, str, str]]:
    """``[(lane, group, label)]``: every (cell group, model) the read plays, anchor groups first."""
    return ([("anchor", g, m) for g, a in spec.anchor.items() for m in a.models]
            + [("s", g, m) for g, x in spec.s.items() for m in x.models])
