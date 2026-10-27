"""The READ SPEC: what a registered read plays, written once before any counted game (``designs/tools/reads.md`` §1).

A spec is ONE JSON file (schema :data:`SCHEMA`), committed with the read's PREREG. It names:

* ``models`` -- ``{label: {arm, seed, ref, load?}}``; ``ref`` is a :class:`ModelRef` (``pinned``: a frozen file and its
  sha256; ``final``: a run's ``final_model.zip`` under the SETTLED rule of :mod:`main.reads.models`); ``load`` is
  the model-LOAD options its units pass (:data:`LOAD_OPTIONS`; two labels may play one file under different ones);
* ``anchor`` -- ``{group: {opponent, teamsets, unit_seeds, models, per_game?}}``: the external-anchor cells, played as
  :mod:`main.x35.kakuna` units on the short lane; ``per_game: true`` plans each seed's 20-game series as 20 ONE-GAME
  units (S-U: a searched game is 25-75 min, so no 20-game searched unit fits a lane), and a group holding a SEARCHED
  model (``load.search``) must be ``per_game`` and name its RAW TWIN (the same ``ref``, the same other load options),
  whose unit of the same deal is G-PREFIX's other half (:mod:`main.reads.prefix`);
* ``s`` -- ``{group: {refpop, refpop_content_sha256, panel, pairs, batch_pairs, family, models, regime_id}}``: S
  against a frozen reference population R, played on the heavy slots;
* ``decision`` -- ``{rule, params, allow_partial_at_stop?}``: a rule of :data:`main.reads.rules.RULES` and its registered
  parameters; ``allow_partial_at_stop: true`` registers that the PREREG accepts a reduced read at the registered stop (the
  record then carries the ``INCOMPLETE_BY_STOP`` class, ``main.reads.record``);
* ``pin`` (optional) -- the commit the PLAYING checkout must be at (a run from another commit is refused);
* ``rederived_guards`` (optional) -- ``count`` / ``report``: how assemble acts on the anchor guards it re-derives from
  the captures and peer logs (:mod:`main.reads.rederive`); absent = ``report`` for a spec registered before them, else
  ``count``;
* ``choices_guard`` (optional) -- ``require`` (default) / ``five_tuple``: a game-for-game guard whose expected capture
  is missing is INCONCLUSIVE, unless the spec registers that its banks lack captures.

An anchor or S group may carry ``import: {state, spec_sha256, group, models}``: those models' cells are NOT played
here but taken from group ``group`` of an EARLIER read's state (its ``spec.json`` must hash to ``spec_sha256``), sha-
and regime-checked by :mod:`main.reads.imports` (N1-P FINDING 6: look 2 re-played look 1's finals).

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

from main.x35.kakuna import OPPONENTS as KAKUNA_OPPONENTS

SCHEMA = "gen3_read_spec_v1"
#: ``choices_guard`` (review H2): ``require`` (the default) makes a game-for-game guard INCONCLUSIVE when a capture its
#: choice comparison expects is missing; ``five_tuple`` registers that the banks are known to lack captures, so the
#: 5-tuple decides alone there (recorded per unit).
CHOICES_REQUIRE, CHOICES_FIVE_TUPLE = "require", "five_tuple"
CHOICES_GUARDS = (CHOICES_REQUIRE, CHOICES_FIVE_TUPLE)
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
#: The model-reference kinds (``main.reads.models`` owns their rules).
REF_KINDS = ("pinned", "final")
#: The anchor opponents a group may name: the Metamon agents :mod:`main.x35.kakuna` plays on its committed cells, each
#: with the registry's weight pins (an agent a cell's config lacks gets a derived config, ``kakuna.opponent_config``).
ANCHOR_OPPONENTS = KAKUNA_OPPONENTS
#: A model's LOAD options (``models.<label>.load``), each an inference-time option of OUR checkpoint that the anchor
#: units pass on: ``belief_prior_mix`` = λ in (0, 1] (``main.anchors --our-belief-prior-mix``; omit it for λ = 0);
#: ``search`` = a registered search ``<preset>@<turn>`` (``main.anchors --our-search``, :mod:`main.anchors.search_presets`;
#: S-U's ``r0s4@10``), allowed only in a ``per_game`` anchor group; ``input_log`` = a turn window ``FROM-TO`` whose
#: decisions' honest input logs a RAW label's units keep (``main.anchors --keep-input-logs``, the S-U gate test's
#: states; it changes no play, so it stamps nothing and never with ``search``).
LOAD_OPTIONS = ("belief_prior_mix", "search", "input_log")


def load_stamp(load: Mapping[str, Any]) -> str:
    """The suffix a model's load options add to the ``our_regime`` its units stamp (``""`` for none): exactly what
    ``main.anchors`` writes (``greedy+belief_prior_mix=0.25``, ``greedy+search=r0s4@10``; ``runner.py``'s ``our_regime``).
    It is what lets one anchor group hold models with DIFFERENT load options and still carry a correct regime id per
    (model, load) (``main.reads.assemble.anchor_group``)."""
    from main.anchors.search_presets import stamp as search_stamp

    lam = load.get("belief_prior_mix")
    return ("" if lam is None else f"+belief_prior_mix={float(lam):g}") + search_stamp(load.get("search"))


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
    load: Dict[str, Any] = field(default_factory=dict)    # model-load options (:data:`LOAD_OPTIONS`)


@dataclass(frozen=True)
class Import:
    """Cells of ``models`` taken from group ``group`` of an EARLIER read's ``state`` (never re-played here)."""

    state: str
    spec_sha256: str
    group: str
    models: Tuple[str, ...]


@dataclass(frozen=True)
class AnchorGroup:
    name: str
    opponent: str
    teamsets: Tuple[str, ...]
    unit_seeds: Tuple[int, ...]
    models: Tuple[str, ...]
    imports: Optional[Import] = None
    #: one-game units (``main.x35.kakuna.plan_units(per_game=True)``); a searched model's group must be one
    per_game: bool = False


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
    imports: Optional[Import] = None

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
    allow_partial_at_stop: bool = False   # the PREREG accepts a reduced read at the registered stop (INCOMPLETE_BY_STOP)
    sha256: str = ""                  # of the spec file's bytes
    raw: Dict[str, Any] = field(default_factory=dict)
    rederived_guards: str = "count"   # main.reads.rederive.MODES: a re-derived guard hit voids the unit, or HOLDS decide
    choices_guard: str = "require"    # CHOICES_GUARDS: a game-for-game guard missing an expected capture is INCONCLUSIVE


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


def _load_opts(d: Any, where: str) -> Dict[str, Any]:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise SpecError(f"{where}: an object")
    _only(d, LOAD_OPTIONS, where)
    out: Dict[str, Any] = {}
    if "belief_prior_mix" in d:
        from agents.model.belief_prior_mix import BeliefPriorMixError, check_lambda

        v = d["belief_prior_mix"]
        try:
            lam = check_lambda(v) if not isinstance(v, bool) else None
        except BeliefPriorMixError as exc:
            raise SpecError(f"{where}.belief_prior_mix: {exc}") from None
        if lam is None or lam <= 0.0:
            raise SpecError(f"{where}.belief_prior_mix: {v!r} must be in (0, 1] (omit it for the unchanged model)")
        out["belief_prior_mix"] = lam
    if "search" in d:
        from main.anchors.search_presets import SearchSpecError, canonical

        try:
            out["search"] = canonical(d["search"])
        except SearchSpecError as exc:
            raise SpecError(f"{where}.search: {exc}") from None
    if "input_log" in d:
        from main.anchors.input_log import InputLogError, canonical as window_canonical

        if "search" in d:
            raise SpecError(f"{where}.input_log: a RAW label's option (a searched label logs its own decisions)")
        try:
            out["input_log"] = window_canonical(d["input_log"])
        except InputLogError as exc:
            raise SpecError(f"{where}.input_log: {exc}") from None
    return out


def search_twins(models: Mapping[str, Model], labels: Tuple[str, ...], where: str) -> Dict[str, str]:
    """``{searched label: its RAW TWIN}`` in one anchor group: the twin is THE label of the group with the same ``ref``
    and the same load options but ``search`` (the raw arm on the same bytes, whose game of each deal G-PREFIX compares).
    None or two such labels is a :class:`SpecError`."""
    out: Dict[str, str] = {}
    for lab in labels:
        m = models[lab]
        if "search" not in m.load:
            continue
        rest = {k: v for k, v in m.load.items() if k != "search"}
        twins = [o for o in labels if o != lab and models[o].ref == m.ref and models[o].load == rest]
        if len(twins) != 1:
            raise SpecError(f"{where}: searched model {lab!r} needs exactly ONE raw twin in the group (the same ref, its "
                            f"load without 'search'), found {twins}")
        out[lab] = twins[0]
    return out


def _group_models(d: Mapping[str, Any], where: str, models: Mapping[str, Model]) -> Tuple[str, ...]:
    ms = _need(d, "models", where, list)
    if not ms or len(set(ms)) != len(ms):
        raise SpecError(f"{where}.models: a non-empty list without repeats")
    for m in ms:
        if m not in models:
            raise SpecError(f"{where}.models: {m!r} is not a declared model")
    return tuple(ms)


def _import(d: Any, where: str, group_models: Tuple[str, ...]) -> Optional[Import]:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise SpecError(f"{where}: an object {{state, spec_sha256, group, models}}")
    _only(d, ("state", "spec_sha256", "group", "models"), where)
    st = _need(d, "state", where, str)
    if not st.strip():
        raise SpecError(f"{where}.state: the earlier read's state dir")
    sha = _need(d, "spec_sha256", where, str)
    if not _SHA256.match(sha):
        raise SpecError(f"{where}.spec_sha256: 64 lowercase hex (the earlier state's spec.json)")
    ms = _need(d, "models", where, list)
    if not ms or len(set(ms)) != len(ms):
        raise SpecError(f"{where}.models: a non-empty list without repeats")
    for m in ms:
        if m not in group_models:
            raise SpecError(f"{where}.models: {m!r} is not one of this group's models")
    return Import(state=st, spec_sha256=sha, group=_label(_need(d, "group", where, str), f"{where}.group"),
                  models=tuple(ms))


def parse(doc: Mapping[str, Any], *, sha256: str = "", rule_params: bool = True) -> ReadSpec:
    """Validate a spec document (module docstring); every problem is a :class:`SpecError`. ``rule_params=False``
    skips only the rule's parameter check (an EARLIER read's spec, read for an import, was valid under the rules of
    its own day)."""
    from main.reads.rules import RULES

    _only(doc, ("schema", "read", "prereg", "pin", "models", "anchor", "s", "decision", "note", "rederived_guards",
                "choices_guard"), "spec")
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
        _only(m, ("arm", "seed", "ref", "load"), w)
        models[lab] = Model(label=lab, arm=_need(m, "arm", w, str), seed=_need(m, "seed", w, int),
                            ref=_ref(m.get("ref"), f"{w}.ref"), load=_load_opts(m.get("load"), f"{w}.load"))
    if not models:
        raise SpecError("spec.models: none declared")
    anchor: Dict[str, AnchorGroup] = {}
    for g, d in (doc.get("anchor") or {}).items():
        w = f"anchor.{_label(g, 'anchor')}"
        if not isinstance(d, dict):
            raise SpecError(f"{w}: an object")
        _only(d, ("opponent", "teamsets", "unit_seeds", "models", "import", "per_game"), w)
        opp = _need(d, "opponent", w, str)
        if opp not in ANCHOR_OPPONENTS:
            raise SpecError(f"{w}.opponent: {opp!r} is none of {ANCHOR_OPPONENTS} (the kakuna runner's cells)")
        from main.x35.kakuna import KakunaError, parse_teamsets

        try:
            ts = parse_teamsets(",".join(_need(d, "teamsets", w, list)))
        except KakunaError as exc:
            raise SpecError(f"{w}.teamsets: {exc}") from None
        gm = _group_models(d, w, models)
        per_game = d.get("per_game", False)
        if not isinstance(per_game, bool):
            raise SpecError(f"{w}.per_game: {per_game!r} is not true / false")
        searched = [m for m in gm if "search" in models[m].load]
        if searched and not per_game:
            raise SpecError(f"{w}: searched model(s) {searched} need per_game: true (a 20-game searched unit fits no "
                            "lane; S-U plays one game a unit)")
        if searched and d.get("import") is not None:
            raise SpecError(f"{w}: a group with a searched model imports nothing (G-PREFIX pairs games of THIS state)")
        search_twins(models, gm, w)
        anchor[g] = AnchorGroup(name=g, opponent=opp, teamsets=ts,
                                unit_seeds=parse_unit_seeds(_need(d, "unit_seeds", w, (str, list)), f"{w}.unit_seeds"),
                                models=gm, imports=_import(d.get("import"), f"{w}.import", gm), per_game=per_game)
    s: Dict[str, SGroup] = {}
    families: Dict[str, str] = {}
    for g, d in (doc.get("s") or {}).items():
        w = f"s.{_label(g, 's')}"
        if not isinstance(d, dict):
            raise SpecError(f"{w}: an object")
        _only(d, ("refpop", "refpop_content_sha256", "panel", "pairs", "batch_pairs", "family", "models",
                  "regime_id", "import"), w)
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
        gm = _group_models(d, w, models)
        s[g] = SGroup(name=g, refpop=_need(d, "refpop", w, str), refpop_content_sha256=rc, panel=panel, pairs=pairs,
                      batch_pairs=bp, family=fam, models=gm, regime_id=rid,
                      imports=_import(d.get("import"), f"{w}.import", gm))
    if not anchor and not s:
        raise SpecError("spec: neither an anchor nor an S group: nothing to play")
    dec = _need(doc, "decision", "spec", dict)
    _only(dec, ("rule", "params", "allow_partial_at_stop"), "decision")
    rule = _need(dec, "rule", "decision", str)
    if rule not in RULES:
        raise SpecError(f"decision.rule: {rule!r} is none of {sorted(RULES)}")
    partial = dec.get("allow_partial_at_stop", False)
    if not isinstance(partial, bool):
        raise SpecError(f"decision.allow_partial_at_stop: {partial!r} is not true / false")
    if partial and RULES[rule].x35_record:
        raise SpecError(f"decision.allow_partial_at_stop: rule {rule!r} writes an X35-stage record, which "
                        "main.x35.batch (look licensing, final_class) loads without reading the INCOMPLETE_BY_STOP class")
    if partial and not anchor:
        raise SpecError("decision.allow_partial_at_stop: the reduced read is of the ANCHOR lane's unplayed units, and the "
                        "spec has no anchor group (an S node short of its pairs has no S at all)")
    params = dict(dec.get("params") or {})
    probs = RULES[rule].check_params(params, models=models, anchor=anchor, s=s) if rule_params else []
    if probs:
        raise SpecError(f"decision.params ({rule}): {probs}")
    prereg = doc.get("prereg")
    if prereg is not None and not isinstance(prereg, str):
        raise SpecError("spec.prereg: a path")
    from main.reads import rederive as RD

    rederived = doc.get("rederived_guards", RD.default_mode(sha256))
    if rederived not in RD.MODES:
        raise SpecError(f"spec.rederived_guards: {rederived!r} is none of {RD.MODES}")
    choices = doc.get("choices_guard", CHOICES_REQUIRE)
    if choices not in CHOICES_GUARDS:
        raise SpecError(f"spec.choices_guard: {choices!r} is none of {CHOICES_GUARDS}")
    return ReadSpec(read=read, models=models, anchor=anchor, s=s, rule=rule, params=params, prereg=prereg, pin=pin,
                    allow_partial_at_stop=partial, sha256=sha256, raw=dict(doc), rederived_guards=rederived,
                    choices_guard=choices)


def load(path: "str | Path", *, rule_params: bool = True) -> ReadSpec:
    b = Path(path).read_bytes()
    try:
        doc = json.loads(b)
    except ValueError as exc:
        raise SpecError(f"{path}: not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: not a JSON object")
    return parse(doc, sha256=hashlib.sha256(b).hexdigest(), rule_params=rule_params)


def imported(group: Any) -> Tuple[str, ...]:
    """The labels of an anchor / S group whose cells are imported (never played in this state)."""
    return group.imports.models if group.imports is not None else ()


def model_groups(spec: ReadSpec) -> List[Tuple[str, str, str]]:
    """``[(lane, group, label)]``: every (cell group, model) the read plays, anchor groups first."""
    return ([("anchor", g, m) for g, a in spec.anchor.items() for m in a.models]
            + [("s", g, m) for g, x in spec.s.items() for m in x.models])
