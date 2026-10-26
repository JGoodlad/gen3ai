"""Does the always-current HUMAN-DATA architecture still say what is true? The static gate for
`designs/endstate/human_data_architecture.md`.

Sits beside `mode_flag_doc_gate_test.py` and `recipe_doc_gate_test.py` at the `src/` root: like them it pins a
hand-written always-current doc to the artefacts it describes, here the snapshot registry, the data card and the code.

**Why it exists (REVIEW C F2, 2026-10-25).** The doc is read as the answer to "what human data is in the system, and
which result used which data". Its first draft declared itself always-current and nothing held it there: the snapshot
table was a hand copy of the registry, the card's headline numbers were hand copies of `results.json`, and the moment
the switch `--human-data` is built the words "NOT BUILT" would go on being believed. A doc that answers a
contamination question wrongly is worse than no doc.

**The extraction is DECLARED.** The doc keeps its pinned facts in two machine-readable places: the STATUS code block of
§0 (`  <key> : <value>` lines) and the snapshot table of §5 (one row per `hr_…` id). `PINNED` below maps each status key
to the artefact it must equal; a key renamed in the doc fails loudly (its line is missing), never quietly.

**Five checks** (each a pure function with a planted-mismatch meta test, the freshness gate's pattern):

- **T1 snapshots ↔ registry.** The §5 table's ids equal the registry's `snapshots`; per id the file count, manifest
  sha256, status and "durable copy yes / no" equal the registry's; the status box's `snapshot <id>` lines agree; at
  most one `reference` and one `candidate`; every `hr_…` token in `designs/endstate/*.md` and `UNDERSTANDING.md` is a
  registered snapshot, rowset or model id, or a placeholder (it contains `<` or `*`).
- **T2 build-state lockstep.** While `agents.info_levers.human_data` does not exist: the status box says
  `switch --human-data : NOT BUILT` and `designs/deleted_flags.md` lists `--human-data` as PROPOSED. Once it exists:
  `BUILT`, no PROPOSED row, and the module's `LEVERS` equal the lever heads of the doc's one grammar line.
- **T3 pinned numbers.** Each `PINNED` status line equals its value computed from the registry or the card's
  `results.json`, formatted by the declared rule (exact string equality: rounding is part of the declaration).
- **T4 production is off.** Pre-build the production mirror has no `human_data` key; post-build it resolves `off`.
- **T5 rowsets track the panels.** Every rowset not `retired` records the held-out panel set it excluded, and that set
  equals the panel registry's current one (a new panel registration retires it).

**Cost: milliseconds** (three JSON files, a handful of markdown files; no torch). Opt out explicitly:

    GEN3AI_SKIP_HUMAN_DATA_DOC_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

_SKIP = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_HUMAN_DATA_DOC_GATE") == "1",
    reason="GEN3AI_SKIP_HUMAN_DATA_DOC_GATE=1",
)

DOC = "designs/endstate/human_data_architecture.md"
REGISTRY = "designs/research_state/human_replay_snapshots.json"
CARD = "designs/research_state/measurements/human_replay_quality_2026-10-25/results/results.json"
DELETED = "designs/deleted_flags.md"
MODULE = "agents.info_levers.human_data"
TOKEN_DOCS = ("designs/research_state/UNDERSTANDING.md",)      # plus every designs/endstate/*.md

_STRATA_GE1500 = ("1500-1599", "1600-1699", "1700+")
_UNSEEN = "acting account UNSEEN on the train dates"


# ------------------------------------------------------------------------------------------ parsing
def status_lines(doc: str) -> List[Tuple[str, str]]:
    """The ``  key : value`` lines of the §0 STATUS code block, in order (duplicates kept)."""
    m = re.search(r"^## 0\. STATUS.*?\n```\n(.*?)\n```", doc, re.S | re.M)
    if not m:
        return []
    out: List[Tuple[str, str]] = []
    for ln in m.group(1).splitlines():
        if " : " in ln:
            k, v = ln.split(" : ", 1)
            out.append((k.strip(), v.strip()))
    return out


def status_box(doc: str) -> Dict[str, str]:
    """The STATUS block as a map (a duplicated key keeps its LAST value; :func:`check_numbers` refuses duplicates)."""
    return dict(status_lines(doc))


def snapshot_rows(doc: str) -> Dict[str, Dict[str, str]]:
    """The §5 snapshot table: | `id` | population | files | dates | `sha` | durable | status | use |."""
    out: Dict[str, Dict[str, str]] = {}
    for m in re.finditer(r"^\| `(hr_[^`]+)` \| ([^|]+) \| ([\d,]+) \| [^|]+ \| `([0-9a-f]{64})` \| (yes|no) \| (\w+) \|",
                         doc, re.M):
        out[m.group(1)] = {"population": m.group(2).strip(), "files": m.group(3).replace(",", ""),
                           "sha": m.group(4), "durable": m.group(5), "status": m.group(6)}
    return out


def grammar_levers(doc: str) -> Set[str]:
    """The lever heads of the doc's ONE grammar line (``**The grammar, …:** `--human-data off` | `bc:…` | …``)."""
    m = re.search(r"\*\*The grammar[^\n]*?\*\*(.*?)(?:\n\n|$)", doc, re.S)
    if not m:
        return set()
    return {h for h in re.findall(r"`(?:--human-data )?([a-z]+)(?::[^`]*)?`", m.group(1)) if h != "off"}


def hr_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"\bhr_[A-Za-z0-9_<>*-]*(?:\.[A-Za-z0-9<>*_-]+)*", text)
            if "<" not in t and "*" not in t and not t.endswith("_")}


# ------------------------------------------------------------------------------------------ checks
def check_snapshots(doc: str, reg: Dict[str, Any], other_texts: Dict[str, str]) -> List[str]:
    """T1."""
    out: List[str] = []
    rows, snaps = snapshot_rows(doc), reg.get("snapshots", {})
    if set(rows) != set(snaps):
        out.append(f"{DOC} §5 lists snapshots {sorted(rows)}; the registry holds {sorted(snaps)}")
    box = status_box(doc)
    for sid in set(rows) & set(snaps):
        r, s = rows[sid], snaps[sid]
        durable = "yes" if s.get("durable") else "no"
        for what, mine, theirs in (("files", r["files"], str(s["files"])), ("manifest sha256", r["sha"],
                                   s["manifest_sha256"]), ("status", r["status"], s.get("status")),
                                   ("durable copy", r["durable"], durable)):
            if mine != theirs:
                out.append(f"{DOC} §5 {sid}: {what} {mine!r} != registry {theirs!r}")
        want = f"{s['files']:,} logs; durable copy {durable}"
        if box.get(f"snapshot {sid}") != want:
            out.append(f"{DOC} §0 STATUS 'snapshot {sid}' is {box.get(f'snapshot {sid}')!r}, want {want!r}")
    stray = [k for k in box if k.startswith("snapshot ") and k[len("snapshot "):] not in snaps]
    out += [f"{DOC} §0 STATUS names an unregistered '{k}'" for k in stray]
    for st in ("reference", "candidate"):
        n = sum(1 for s in snaps.values() if s.get("status") == st)
        if n > 1:
            out.append(f"{REGISTRY}: {n} snapshots are '{st}' (at most one is live)")
    known = set(snaps) | set(reg.get("rowsets", {})) | set(reg.get("models", {}))
    for path, text in other_texts.items():
        for t in sorted(hr_tokens(text) - known):
            out.append(f"{path} names {t!r}, which is not a registered snapshot / rowset / model id")
    return out


def check_switch(doc: str, deleted: str, built: bool, levers: Optional[Set[str]],
                 prod: Dict[str, Any]) -> List[str]:
    """T2 + T4."""
    out: List[str] = []
    state = status_box(doc).get("switch --human-data")
    want = "BUILT" if built else "NOT BUILT"
    if state != want:
        out.append(f"{DOC} §0 STATUS 'switch --human-data' is {state!r}; {MODULE} "
                   f"{'exists' if built else 'does not exist'}, so it must read {want!r}")
    row = next((ln for ln in deleted.splitlines() if ln.startswith("| `--human-data` |")), None)
    if not built and (row is None or "PROPOSED" not in row):
        out.append(f"{DELETED}: '--human-data' must be a PROPOSED row while {MODULE} does not exist")
    if built and row is not None and "PROPOSED" in row:
        out.append(f"{DELETED}: '--human-data' is built; drop its PROPOSED row")
    heads = grammar_levers(doc)
    if not heads:
        out.append(f"{DOC} §6: the one grammar line ('**The grammar, …:**') is missing or names no lever")
    if built and levers is not None and set(levers) != heads:
        out.append(f"{DOC} §6 grammar levers {sorted(heads)} != {MODULE}.LEVERS {sorted(levers)}")
    if not built and "human_data" in prod:
        out.append(f"the production mirror has 'human_data' = {prod['human_data']!r} but {MODULE} is not built")
    if built and prod.get("human_data") != "off":
        out.append(f"the production mirror's human_data is {prod.get('human_data')!r}; production must be 'off'")
    return out


#: status key -> how its value is computed from (registry, results). The FORMAT is part of the declaration.
PINNED: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], str]] = {
    "informative rows, actor >= 1500": lambda reg, res: "{:.2f}M".format(
        sum(res["labels_x50_train"]["by_stratum"][s]["informative"] for s in _STRATA_GE1500) / 1e6),
    ">= 1500 full-six sides unfamiliar to the pool (<= 2 fits; a lower bound)": lambda reg, res: "{:.1f} %".format(
        100 * sum(res["team_distance"]["humans_ge1500_vs_pool"]["set_fits"][str(k)] for k in (0, 1, 2))),
    "HOM - P* NLL on accounts unseen in training": lambda reg, res: "{:+.3f} nats".format(
        res["hom_join"]["strata"][_UNSEEN]["d"]["mean"]),
    "rowsets / models registered": lambda reg, res: f"{len(reg.get('rowsets', {}))} / {len(reg.get('models', {}))}",
}


def check_numbers(doc: str, reg: Dict[str, Any], res: Dict[str, Any]) -> List[str]:
    """T3."""
    out: List[str] = []
    box = status_box(doc)
    if not box:
        return [f"{DOC}: the §0 STATUS code block is missing"]
    keys = [k for k, _ in status_lines(doc)]
    for k in sorted({k for k in keys if keys.count(k) > 1}):
        out.append(f"{DOC} §0 STATUS repeats the key '{k}' (REVIEW D m1: a stale duplicate must not shadow the "
                   "real line)")
    for key, fn in PINNED.items():
        want = fn(reg, res)
        if box.get(key) != want:
            out.append(f"{DOC} §0 STATUS '{key}' is {box.get(key)!r}; the registry / card say {want!r}")
    return out


def check_rowsets(reg: Dict[str, Any], panels: Set[str]) -> List[str]:
    """T5."""
    out: List[str] = []
    for rid, row in reg.get("rowsets", {}).items():
        if row.get("status") == "retired":
            continue
        got = set(row.get("panel_exclusion_set", ()))
        if got != panels:
            out.append(f"{REGISTRY} rowset {rid}: excluded panels {sorted(got)} != the held-out panels "
                       f"{sorted(panels)}; retire it (status 'retired') or rebuild it")
    return out


# ------------------------------------------------------------------------------------------ the real inputs
def _read(rel: str) -> str:
    return repo_path(*rel.split("/")).read_text()


def _inputs() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    return _read(DOC), json.loads(_read(REGISTRY)), json.loads(_read(CARD))


def _token_texts() -> Dict[str, str]:
    paths = sorted(p.relative_to(repo_path()).as_posix() for p in repo_path("designs", "endstate").glob("*.md"))
    return {p: _read(p) for p in (*paths, *TOKEN_DOCS)}


def _built() -> Tuple[bool, Optional[Set[str]]]:
    spec = importlib.util.find_spec(MODULE)
    if spec is None:
        return False, None
    mod = importlib.import_module(MODULE)
    return True, set(getattr(mod, "LEVERS", ()))


@_SKIP
def test_t1_snapshots_match_the_registry():
    doc, reg, _ = _inputs()
    assert snapshot_rows(doc), f"{DOC} §5: no snapshot rows parsed (the table's shape changed?)"
    assert check_snapshots(doc, reg, _token_texts()) == []


@_SKIP
def test_t2_t4_switch_state_and_production_default():
    from agents.training.baselines import production_config

    doc, _, _ = _inputs()
    built, levers = _built()
    assert check_switch(doc, _read(DELETED), built, levers, production_config()) == []


@_SKIP
def test_t3_pinned_status_numbers():
    doc, reg, res = _inputs()
    assert check_numbers(doc, reg, res) == []


@_SKIP
def test_t5_rowsets_track_the_held_out_panels():
    from main import heldout_panels

    _, reg, _ = _inputs()
    assert check_rowsets(reg, set(heldout_panels.panel_names(held_out_only=True))) == []


# ------------------------------------------------------------------------------------------ meta: planted mismatches
_SHA = "a" * 64
_REG = {"snapshots": {"hr_x_v1": {"files": 10, "manifest_sha256": _SHA, "status": "reference", "durable": {"a": 1}}},
        "rowsets": {}, "models": {}}
_DOC = (
    "## 0. STATUS (pinned)\n\n```\nSTATUS\n  switch --human-data   : NOT BUILT\n  snapshot hr_x_v1  : 10 logs; durable "
    "copy yes\n  informative rows, actor >= 1500 : 1.80M\n  >= 1500 full-six sides unfamiliar to the pool (<= 2 fits; a "
    "lower bound) : 15.9 %\n"
    "  HOM - P* NLL on accounts unseen in training : -0.580 nats\n  rowsets / models registered : 0 / 0\n```\n\n"
    "## 5. Snapshots\n\n| id | p | files | dates | sha | durable | status | use |\n|---|---|---|---|---|---|---|---|\n"
    f"| `hr_x_v1` | spectator | 10 | a → b | `{_SHA}` | yes | reference | x |\n\n"
    "**The grammar, the one place it is defined:** `--human-data off` | `bc:<rowset>@<coef>` | `teams:<rowset>@<s>`.\n\n"
)
_RES = {"labels_x50_train": {"by_stratum": {"1500-1599": {"informative": 1_260_000}, "1600-1699": {"informative":
        440_000}, "1700+": {"informative": 100_000}}},
        "team_distance": {"humans_ge1500_vs_pool": {"set_fits": {"0": 0.05, "1": 0.05, "2": 0.059}}},
        "hom_join": {"strata": {_UNSEEN: {"d": {"mean": -0.58}}}}}
_DELETED = "| `--human-data` | PROPOSED in the architecture doc | the switch; not built |\n"


def test_meta_the_planted_world_is_clean():
    assert check_snapshots(_DOC, _REG, {"x.md": "cites `hr_x_v1` and `hr_sub_<tag>` and `hr_full_*`"}) == []
    assert check_switch(_DOC, _DELETED, False, None, {}) == []
    assert check_numbers(_DOC, _REG, _RES) == []
    assert grammar_levers(_DOC) == {"bc", "teams"}


def test_meta_t1_catches_each_snapshot_mismatch():
    import copy

    for mutate in (lambda r: r["snapshots"]["hr_x_v1"].__setitem__("files", 11),
                   lambda r: r["snapshots"]["hr_x_v1"].__setitem__("manifest_sha256", "b" * 64),
                   lambda r: r["snapshots"]["hr_x_v1"].__setitem__("status", "candidate"),
                   lambda r: r["snapshots"]["hr_x_v1"].__setitem__("durable", None),
                   lambda r: r["snapshots"].__setitem__("hr_y_v1", dict(r["snapshots"]["hr_x_v1"]))):
        reg = copy.deepcopy(_REG)
        mutate(reg)
        assert check_snapshots(_DOC, reg, {}), "a registry change the doc does not follow must fail"
    assert check_snapshots(_DOC, _REG, {"x.md": "trained on hr_spec_v9"}), "an unregistered id in a doc must fail"
    two = {"snapshots": {"hr_x_v1": _REG["snapshots"]["hr_x_v1"], "hr_y_v1": dict(_REG["snapshots"]["hr_x_v1"])}}
    assert any("at most one is live" in p for p in check_snapshots(_DOC, two, {}))


def test_meta_t2_t4_catch_a_built_switch_the_doc_does_not_admit():
    assert check_switch(_DOC, _DELETED, True, {"bc", "teams"}, {"human_data": "off"})       # doc still NOT BUILT
    built_doc = _DOC.replace(": NOT BUILT", ": BUILT")
    assert check_switch(built_doc, "", True, {"bc", "teams"}, {"human_data": "off"}) == []
    assert check_switch(built_doc, "", True, {"bc", "teams", "kl"}, {"human_data": "off"})   # grammar != module
    assert check_switch(built_doc, _DELETED, True, {"bc", "teams"}, {"human_data": "off"})   # PROPOSED row left
    assert check_switch(built_doc, "", True, {"bc", "teams"}, {"human_data": "bc:x@1"})      # production not off
    assert check_switch(_DOC, "| `--human-data` | the switch |\n", False, None, {})          # row not PROPOSED
    assert check_switch(_DOC, _DELETED, False, None, {"human_data": "off"})                  # key before the build
    assert check_switch(_DOC.replace("**The grammar", "The grammar"), _DELETED, False, None, {})


def test_meta_t3_catches_a_moved_number():
    import copy

    res = copy.deepcopy(_RES)
    res["labels_x50_train"]["by_stratum"]["1700+"]["informative"] += 10_000
    assert check_numbers(_DOC, _REG, res)
    res = copy.deepcopy(_RES)
    res["hom_join"]["strata"][_UNSEEN]["d"]["mean"] = -0.5
    assert check_numbers(_DOC, _REG, res)
    reg = {**_REG, "rowsets": {"hr_x_v1.r1": {}}}
    assert check_numbers(_DOC, reg, _RES)
    assert check_numbers("no status block", _REG, _RES)
    stale = _DOC.replace("  informative rows, actor >= 1500 : 1.80M\n",
                         "  informative rows, actor >= 1500 : 9.99M\n  informative rows, actor >= 1500 : 1.80M\n")
    assert any("repeats the key" in p for p in check_numbers(stale, _REG, _RES))


def test_meta_t5_catches_a_rowset_built_before_a_panel_was_registered():
    reg = {"rowsets": {"hr_x_v1.r1": {"panel_exclusion_set": ["a", "b"]},
                       "hr_x_v1.r0": {"panel_exclusion_set": ["a"], "status": "retired"}}}
    assert check_rowsets(reg, {"a", "b"}) == []
    assert check_rowsets(reg, {"a", "b", "c"})
