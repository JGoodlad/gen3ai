"""Does every test stub actually STUB something? The static gate against VACUOUS patch targets.

Sits beside `ruff_gate_test.py`, `file_size_gate_test.py` and `claude_md_freshness_gate_test.py` at
the `src/` root because, like those, its subject is the whole tree rather than any one package. The
scan engine is `src/stub_vacuity_scan.py`; this file is the POLICY over it — what fails, what may be
allow-listed, and the self-checks that pin both.

**The bug class.** A test installs a stub and asserts an outcome. If the stub never reaches the code
under test, the assertion is about the REAL code path and the test passes for a reason that has
nothing to do with what it claims to cover. `ccd08003` (the `instrumented_ppo/ppo.py` decomposition,
2026-09-06) found four such sites in one file family, one of them holding up a byte-identity test
that was comparing two IDENTICAL arms and reporting a pass. Nothing in the suite could see it.

**The shapes are created by a MOVE, not by writing a bad test.** Every one of those four sites was
correct on the day it was written and went vacuous when the symbol it named moved to another module.
That is why this has to be a standing gate rather than a one-off sweep: the next decomposition
(`reward_manager.py`, 1,990 lines, patched at 32 sites from `reward_tracker_test.py` alone) creates
the same hazard, and the sweep would have to be remembered.

**Two vacuity shapes, named in the failure message because the fixes differ.**

* **(1) DEFINITION-SITE PATCH, consumer holds its own reference** (verdict `unread`). The module
  named HAS the attribute — it defines it — but never reads it, and no non-test module reaches it as
  `<module>.<attr>` either. The consumer wrote a MODULE-LEVEL `from x import func`, so it took its
  copy before any test ran and the stub cannot reach it. Silent at runtime: the patch installs
  cleanly and the real code runs. **Fix:** repoint at the module that actually calls it.
* **(2) THE NAME IS NOT THERE** (verdict `missing`). The module named does not bind the attribute at
  all, usually because it moved. `monkeypatch.setattr` and `mock.patch` raise on this *if the line
  runs*, so it is loud — but it is silent under `raising=False` / `create=True`, silent for every
  test the routine gate never selects, and **silent always for the hand-rolled `mod.name = stub`
  shape**, which simply creates the attribute. Two of `ccd08003`'s three patch-target sites were
  that shape. **Fix:** repoint, or delete the stub.

**A DEFINITION-SITE patch is not automatically vacuous, and that is where a naive gate dies.**
`main.launcher.ipc` defines `emit` and never calls it, yet patching `ipc.emit` works perfectly
because its consumers hold the MODULE and resolve the attribute at call time. Measured 2026-09-07:
**44 of 407 sites** are exactly that pattern. A gate without the consumer search would open with a
44-finding false-positive storm and be excluded inside a week, so `unread` is only reported after
searching every non-test module for a qualified read — and the failure message names what it
searched.

**What is OUT of scope, and why none of it is a gap** (139 of 407 sites, each recorded with its
reason — a scanner that cannot say what it declined to look at is indistinguishable from one that
found nothing):

* a target that is not a module of ours — `sys.argv`, `torch`, `subprocess`, `sb3_contrib`, or a
  local object (97). Not ours to reason about.
* a CLASS attribute (34). Safe by construction: `patch.object(SomeClass, "method")` mutates the one
  class object, so a consumer that did `from x import SomeClass` sees the stub anyway.
* a computed target — a name loop or an expression (8). **This is the gate's one real blind spot.**
  `for name in (...): monkeypatch.setattr(mod, name, ...)` is invisible to it. Prefer the spelled-out
  form in a new test so the gate can keep auditing it.

**Cost (measured 2026-09-07): 3.6 s** for 407 sites across 70 test files — one AST parse per test
file plus one per non-test module for the consumer index. No cost marker; runs in the routine gate.
Opt out explicitly:

    GEN3AI_SKIP_STUB_GATE=1 pytest src/ -q
"""
import ast
import os
import textwrap
from pathlib import Path
from typing import Dict

import pytest

from stub_vacuity_scan import Site, consumer_index, scan_tree
from utils.paths import src_root

_SRC_ROOT = src_root()

# ---------------------------------------------------------------------------------------------
# THE ALLOWLIST — a TEMPORARY list that must shrink, never a blanket exclude.
#
# **EMPTY, as of 2026-09-07, and that is a MEASUREMENT.** The full-tree census recorded ONE finding
# in 407 sites (`designs/research_state/measurements/stub_vacuity_audit_2026-09-07.md`) and it was
# FIXED at the source rather than listed: `main/launcher/dry_run_test.py` trapped `_launch_child` on
# `main.launcher.child` while `run.py` calls its own module-level import of it, so the "--dry-run
# must never spawn a child" booby trap — the most consequential of that fixture's three — was not
# armed. Verified at runtime: with the old spelling installed, `run._launch_child` was still the
# real function.
#
# Key: `"<repo-relative test path>:<dotted module>.<attr>"`. Value: THE REASON, which is required
# and is checked for substance — an entry without one fails the gate rather than passing silently.
# `test_meta_the_allowlist_may_not_go_stale` fails an entry that no longer names a live finding, so
# the list cannot outlive its own fix the way the file-size grandfather list was designed not to.
#
# ⚠️ Reach for a FIX before an entry. The two legitimate reasons to list something are a symbol read
# through machinery this AST scan cannot see (a registry table built at import time, a `getattr` on
# a computed name) and a genuine cross-module indirection being actively refactored. "The test
# passes" is not a reason — a vacuous stub's test passing is the entire failure mode.
# ---------------------------------------------------------------------------------------------
ALLOWLIST: Dict[str, str] = {}


def key_of(site: Site) -> str:
    return f"{site.path}:{site.module}.{site.attr}"


_SHAPES = {
    "unread": (
        "SHAPE (1) — DEFINITION-SITE PATCH, and the consumer holds its own reference.\n"
        "    `{module}` HAS `{attr}` but never READS it, and NO non-test module reaches it as\n"
        "    `{module}.{attr}` either. Whoever calls it took a module-level `from … import {attr}`,\n"
        "    so it holds a copy taken before this stub was installed and the stub reaches nothing.\n"
        "    The test is asserting about the REAL code path.\n"
        "    FIX: repoint the patch at the module that actually calls `{attr}` (name it as a module\n"
        "    object, not through a loop variable, so this gate can keep auditing it), or delete the\n"
        "    stub if the test does not need it."
    ),
    "missing": (
        "SHAPE (2) — THE NAME IS NOT THERE. `{module}` does not bind `{attr}` at all, under any\n"
        "    binding. Usually the symbol MOVED and the patch did not follow it.\n"
        "    FIX: repoint at whichever module holds it now, or delete the stub. Do not assume this\n"
        "    one would have raised: it is silent under `raising=False`/`create=True`, silent in any\n"
        "    tier the routine gate does not select, and ALWAYS silent for a `mod.name = stub`\n"
        "    assignment, which simply creates the attribute."
    ),
}


def describe(site: Site) -> str:
    body = _SHAPES[site.verdict].format(module=site.module, attr=site.attr)
    return (f"  {site.path}:{site.lineno}  {site.call}({site.target})\n"
            f"    -> {site.module}.{site.attr}\n    {body}")


_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_STUB_GATE") == "1",
                           reason="GEN3AI_SKIP_STUB_GATE=1")


@pytest.fixture(scope="module")
def sites():
    return scan_tree(_SRC_ROOT)


@_SKIP
def test_the_scan_actually_walked_the_tree(sites):
    """A scan that found nothing is a GAP IN COVERAGE, not a pass.

    Same rule as the ruff gate's missing-tool branch: a check that silently opts out reads exactly
    like one that found nothing, so the empty result FAILS and says what is broken.
    """
    assert len(sites) > 200, (
        f"the stub-vacuity scan found only {len(sites)} patch sites in {_SRC_ROOT} — the census "
        f"measured 407 on 2026-09-07, so this is a BROKEN SCAN, not a clean tree. Check that "
        f"PYTHONPATH points at THIS checkout's src/ (in a worktree the export is mandatory; "
        f"src/packaging_gate_test.py covers the other direction) and that "
        f"src/stub_vacuity_scan.py still parses."
    )
    assert any(s.verdict == "ok" for s in sites), "no site resolved at all — the scan is broken"


@_SKIP
def test_no_test_stub_is_vacuous(sites):
    findings = [s for s in sites if s.verdict in _SHAPES and key_of(s) not in ALLOWLIST]
    assert not findings, (
        f"{len(findings)} VACUOUS PATCH TARGET(S) — a stub that stubs NOTHING, so whatever the "
        f"test asserts is about the real code path and it passes for the wrong reason:\n\n"
        + "\n\n".join(describe(s) for s in findings)
        + "\n\nIf a finding is genuinely reached by machinery this AST scan cannot see (an "
          "import-time registry, a `getattr` on a computed name), add it to ALLOWLIST in "
          f"{os.path.basename(__file__)} WITH ITS REASON — and expect to take it back off. "
          "That list is temporary and may only shrink; 'the test passes' is not a reason."
    )


@_SKIP
def test_the_allowlist_carries_a_reason_and_has_not_gone_stale(sites):
    """Every entry must be substantive AND still name a live finding.

    The c-family lesson the file-size gate records: an allowlist entry can outlive its own fix and
    then mislead every reader after it, including agents briefed from it.
    """
    thin = [k for k, why in ALLOWLIST.items() if len(why.strip()) < 30]
    assert not thin, (
        f"ALLOWLIST entries with no real reason: {thin}. A reason is REQUIRED — it is what lets the "
        f"next reader decide whether the entry is still true."
    )
    live = {key_of(s) for s in sites if s.verdict in _SHAPES}
    stale = sorted(set(ALLOWLIST) - live)
    assert not stale, (
        f"STALE ALLOWLIST ENTRIES — these no longer name a finding, so REMOVE them: {stale}.\n"
        f"  The list may only shrink, and this is the shrink."
    )


# ---------------------------------------------------------------------------------------------
# SELF-CHECKS — a synthetic tree per case, so every rule in the docstring is pinned against a
# planted fault rather than against whatever the real tree happens to contain today. A gate whose
# detection is never exercised is a gate nobody can trust once the tree is clean, and the tree IS
# clean: one finding in 407 sites.
# ---------------------------------------------------------------------------------------------

def _tree(tmp_path: Path, files: Dict[str, str]) -> Path:
    root = tmp_path / "src"
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body))
    return root


def _verdicts(root: Path):
    return {(s.module, s.attr): s.verdict for s in scan_tree(root) if s.module}


def test_selfcheck_shape_1_definition_site_patch_with_a_direct_import_consumer(tmp_path):
    """The classic no-op: the consumer did `from x import func`, the test patched `x.func`."""
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/helpers.py": "def helper():\n    return 'real'\n",
        "pkg/consumer.py": """
            from pkg.helpers import helper       # module-level: takes its copy ONCE

            def run():
                return helper()
        """,
        "pkg/consumer_test.py": """
            import pkg.helpers as helpers

            def test_it(monkeypatch):
                monkeypatch.setattr(helpers, "helper", lambda: "stub")
        """,
    })
    assert _verdicts(root)[("pkg.helpers", "helper")] == "unread"


def test_selfcheck_shape_2_the_name_moved_out_of_the_module(tmp_path):
    """The `ccd08003` shape, reduced: a decomposition moved the symbol, the patch did not follow.

    Both spellings that commit had to repoint are planted — the `monkeypatch.setattr` call AND the
    hand-rolled `mod.name = stub` assignment, which raises NOTHING at runtime and is therefore the
    one a live suite can never surface on its own.
    """
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/setup_mixin.py": """
            def shared_trunk_parameters(fe):
                return []

            def prepare(fe):
                return shared_trunk_parameters(fe)
        """,
        "pkg/ppo.py": """
            from pkg.setup_mixin import prepare      # the name MOVED to setup_mixin

            def train(fe):
                return prepare(fe)
        """,
        "pkg/ppo_test.py": """
            import pkg.ppo as ppo_mod

            def test_call_form(monkeypatch):
                monkeypatch.setattr(ppo_mod, "shared_trunk_parameters", lambda fe: [])

            def test_assignment_form():
                ppo_mod.live_gauge_metrics = lambda v, z: {}
        """,
    })
    v = _verdicts(root)
    assert v[("pkg.ppo", "shared_trunk_parameters")] == "missing"
    assert v[("pkg.ppo", "live_gauge_metrics")] == "missing", (
        "the hand-rolled `mod.name = stub` shape must be judged too — it is SILENT at runtime, and "
        "two of ccd08003's three patch-target sites were exactly that spelling"
    )


def test_selfcheck_a_legitimate_definition_site_patch_is_NOT_a_finding(tmp_path):
    """The `main.launcher.ipc.emit` pattern — 44 of 407 real sites. This is the false positive that
    would get the gate excluded, so it is pinned."""
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/ipc.py": "def emit(msg):\n    print(msg)\n",
        "pkg/consumer.py": """
            from pkg import ipc                  # holds the MODULE, resolves at CALL time

            def run():
                ipc.emit("hi")
        """,
        "pkg/consumer_test.py": """
            from pkg import ipc

            def test_it(monkeypatch):
                monkeypatch.setattr(ipc, "emit", lambda m: None)
        """,
    })
    assert _verdicts(root)[("pkg.ipc", "emit")] == "ok"


def test_selfcheck_a_deferred_import_in_the_consumer_is_a_real_reach(tmp_path):
    """`from x import f` INSIDE a function re-executes per call, so it DOES see the stub.

    Two real sites depend on this and would be false positives without it
    (`utils.team_loader.TeamLoader`, `obs_materializer.materialize_branches`).
    """
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/heavy.py": "class Loader:\n    pass\n",
        "pkg/consumer.py": """
            def run():
                from pkg.heavy import Loader     # deferred: re-read on every call
                return Loader()
        """,
        "pkg/consumer_test.py": """
            def test_it(monkeypatch):
                monkeypatch.setattr("pkg.heavy.Loader", object)
        """,
    })
    assert _verdicts(root)[("pkg.heavy", "Loader")] == "ok"


def test_selfcheck_the_string_form_resolves_the_same_as_the_object_form(tmp_path):
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/helpers.py": "def helper():\n    return 1\n",
        "pkg/consumer.py": "from pkg.helpers import helper\n\ndef run():\n    return helper()\n",
        "pkg/consumer_test.py": """
            from unittest import mock

            def test_string_form():
                with mock.patch("pkg.helpers.helper"):
                    pass
        """,
    })
    assert _verdicts(root)[("pkg.helpers", "helper")] == "unread"


def test_selfcheck_a_dynamic_namespace_module_is_never_a_finding(tmp_path):
    """A module that reaches its own namespace by computation defeats any static read claim, so it
    clears everything. Conservative on purpose — it can only suppress, never invent."""
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/registry.py": """
            def thing():
                return 1

            def dispatch(name):
                return globals()[name]()
        """,
        "pkg/registry_test.py": """
            import pkg.registry as reg

            def test_it(monkeypatch):
                monkeypatch.setattr(reg, "thing", lambda: 2)
        """,
    })
    assert _verdicts(root)[("pkg.registry", "thing")] == "ok"


def test_selfcheck_out_of_scope_targets_are_skipped_with_a_reason(tmp_path):
    """Never silently dropped: every declined site says why."""
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/thing_test.py": """
            import sys

            def test_it(monkeypatch):
                monkeypatch.setattr("sys.argv", ["x"])
                monkeypatch.setattr(sys, "path", [])
        """,
    })
    skipped = [s for s in scan_tree(root) if s.verdict == "skipped"]
    assert len(skipped) == 2 and all(s.reason for s in skipped), (
        f"expected 2 skipped sites each carrying a reason, got {[(s.verdict, s.reason) for s in skipped]}"
    )


def test_selfcheck_a_local_object_attribute_assignment_is_not_a_patch_site(tmp_path):
    """`self.x = 1` and every ordinary attribute write must not be mistaken for a stub."""
    root = _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/thing_test.py": """
            class Holder:
                def __init__(self):
                    self.value = 1

            def test_it():
                h = Holder()
                h.value = 2
        """,
    })
    assert [s for s in scan_tree(root) if s.module] == []


# ---------------------------------------------------------------------------------------------
# META — the POLICY, pinned against synthetic sites so a future softening fails loudly.
# ---------------------------------------------------------------------------------------------

def _site(verdict, path="src/a_test.py", module="pkg.m", attr="f"):
    return Site(path, 7, "monkeypatch.setattr", f"{module}, '{attr}'", module, attr, verdict)


def test_meta_both_shapes_produce_a_message_that_names_the_shape_and_the_fix():
    for verdict, marker in (("unread", "SHAPE (1)"), ("missing", "SHAPE (2)")):
        msg = describe(_site(verdict))
        assert marker in msg and "FIX:" in msg
        assert "pkg.m" in msg and "f" in msg, "the message must name the module and the attribute"


def test_meta_an_ok_or_skipped_site_is_never_a_finding():
    assert "ok" not in _SHAPES and "skipped" not in _SHAPES


def test_meta_a_thin_allowlist_reason_is_rejected():
    assert len("too messy".strip()) < 30, "the substance bar is what the gate asserts"


def test_meta_the_allowlist_is_keyed_per_SITE_not_per_module():
    """A blanket per-module exclude would take unrelated real findings down with it — the ruff
    gate's rule, for the same reason."""
    assert key_of(_site("unread")) == "src/a_test.py:pkg.m.f"


def test_meta_the_scan_engine_has_no_import_of_the_code_under_test():
    """PURE AST is load-bearing, not a style choice: importing every test module would cost minutes,
    would break on any module whose architecture drifted past current code, and could execute an
    entry point (`src/main/entry_point_guard_test.py` exists because that has been a real hazard)."""
    src = (_SRC_ROOT / "stub_vacuity_scan.py").read_text()
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "ast", "dataclasses", "pathlib", "collections", "sys",
                        "utils"}, f"the scan engine grew a runtime dependency: {sorted(imported)}"


@_SKIP
def test_meta_the_consumer_search_is_actually_finding_consumers():
    """If the index came back empty the gate would still pass — and would be reporting on nothing.

    This is the same failure direction as `test_the_scan_actually_walked_the_tree`: the expensive
    half of the check has to prove it ran.
    """
    index = consumer_index(_SRC_ROOT)
    assert len(index) > 1000, f"the consumer index holds only {len(index)} qualified reads"
    assert "utils.paths.repo_root" in index, "a known cross-module read is missing from the index"
