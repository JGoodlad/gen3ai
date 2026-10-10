"""`utils.showdown_deps` - what a training child needs from `deps/pokemon-showdown`, and the preflight over it.

The launcher's pinned worktree links `deps/pokemon-showdown` to the LAUNCHING checkout's submodule. From a checkout
whose submodule was never initialised or built that is an empty placeholder, and the child died minutes after the
launch in team validation (`gpu_checks_endstate_2026-10-09` F-GE-4). This file pins the pure half: the closed list of
files, the verdict, the fix text, and - against the REAL checkout and a REAL `node` run - that the list is what team
validation actually reads (so it can neither go stale nor hold a phantom path).

`main/launcher/submodule_gate_test.py` pins the wiring (the launcher refuses, `--dry-run` prints).
"""
import json
import os
import subprocess

import pytest

from utils import showdown_deps as sd
from utils.contention import scale_timeout
from utils.paths import repo_path


def _node(_name):
    return "/usr/bin/node"


def _no_node(_name):
    return None


# ---------------------------------------------------------------------------------------
# the verdict, over throwaway trees
# ---------------------------------------------------------------------------------------

def test_a_complete_tree_passes(tmp_path):
    root = sd.write_stub_checkout(str(tmp_path))
    v = sd.check(root, which=_node)
    assert not v.refused and v.missing == () and v.fix() == []
    assert v.lines()[0].startswith("showdown deps : ✓") and root in v.lines()[0]


def test_an_empty_placeholder_is_the_uninitialised_submodule_and_the_fix_is_named(tmp_path):
    """The exact 2026-10-09 shape: `git worktree add` leaves `deps/pokemon-showdown` an EMPTY directory."""
    root = os.path.join(str(tmp_path), "deps", "pokemon-showdown")
    os.makedirs(root)
    v = sd.check(root, which=_node)
    assert v.refused and len(v.missing) == len(sd.NEEDS)
    assert v.kinds == (sd.KIND_CHECKOUT, sd.KIND_BUILD, sd.KIND_MODULES)
    text = "\n".join(v.lines())
    assert "REFUSED" in text and "package.json" in text
    assert "git submodule update --init" in text and "./scripts/bootstrap.sh --skip-env" in text
    assert str(tmp_path) in text, "the fix must name the checkout it applies to"
    assert "Cannot find module" in text, "name the symptom the operator would otherwise meet minutes later"


def test_a_missing_directory_is_refused_the_same_way(tmp_path):
    v = sd.check(os.path.join(str(tmp_path), "deps", "pokemon-showdown"), which=_node)
    assert v.refused and v.kinds[0] == sd.KIND_CHECKOUT


def test_a_checked_out_but_unbuilt_submodule_is_a_BUILD_problem_not_a_checkout_one(tmp_path):
    root = sd.write_stub_checkout(str(tmp_path))
    for n in sd.NEEDS:
        if n.kind != sd.KIND_CHECKOUT:
            os.remove(os.path.join(root, *n.rel.split("/")))
    v = sd.check(root, which=_node)
    assert v.kinds == (sd.KIND_BUILD, sd.KIND_MODULES)
    text = "\n".join(v.lines())
    assert "git submodule update --init" not in text, "the checkout is fine - do not send them to re-init it"
    assert "bootstrap.sh --skip-env" in text and "node build" in text and "npm ci" in text


def test_a_dangling_dist_link_reads_as_missing(tmp_path):
    """A linked worktree reaches `dist/` through a link to main's; a link whose target is gone is NOT a dist."""
    root = sd.write_stub_checkout(str(tmp_path))
    dist = os.path.join(root, "dist")
    for dirpath, _dirs, files in os.walk(dist, topdown=False):
        for f in files:
            os.remove(os.path.join(dirpath, f))
        os.rmdir(dirpath)
    os.symlink(os.path.join(str(tmp_path), "gone"), dist)
    v = sd.check(root, which=_node)
    assert v.refused and v.kinds == (sd.KIND_BUILD,)
    assert all(n.rel.startswith("dist/") for n in v.missing)


def test_a_missing_node_refuses_even_with_a_complete_tree(tmp_path):
    root = sd.write_stub_checkout(str(tmp_path))
    v = sd.check(root, which=_no_node)
    assert v.refused and v.missing == () and v.node is None
    assert "`node` is not on PATH" in v.lines()[0]
    assert any("node" in f for f in v.fix())


def test_a_single_missing_file_is_named(tmp_path):
    root = sd.write_stub_checkout(str(tmp_path))
    os.remove(os.path.join(root, "dist", "data", "mods", "gen3", "scripts.js"))
    v = sd.check(root, which=_node)
    assert [n.rel for n in v.missing] == ["dist/data/mods/gen3/scripts.js"]
    assert "dist/data/mods/gen3/scripts.js" in v.lines()[0]


def test_every_need_is_declared_once_with_a_known_kind_and_a_reason():
    rels = [n.rel for n in sd.NEEDS]
    assert len(rels) == len(set(rels))
    assert {n.kind for n in sd.NEEDS} == {sd.KIND_CHECKOUT, sd.KIND_BUILD, sd.KIND_MODULES}
    assert all(n.why.strip() for n in sd.NEEDS)


# ---------------------------------------------------------------------------------------
# the list is what team validation really reads - against the REAL checkout and a REAL node
# ---------------------------------------------------------------------------------------

#: Trace every module `require` resolves and every `fs.readFileSync` across ONE real validation of a legal team.
_TRACE_JS = r"""
const Module = require('module');
const fs = require('fs');
const seen = new Set();
const orig = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  const r = orig.call(this, request, parent, ...rest); seen.add(r); return r;
};
const ofs = fs.readFileSync;
fs.readFileSync = function (p, ...a) { if (typeof p === 'string') seen.add(p); return ofs.call(this, p, ...a); };
const { Teams, TeamValidator } = require(process.argv[1]);
const v = new TeamValidator('gen3ou');
const team = Teams.import('Tyranitar @ Leftovers\nAbility: Sand Stream\nEVs: 252 HP / 4 Atk / 252 SpA\n' +
  'Quiet Nature\n- Rock Slide\n- Crunch\n- Pursuit\n- Fire Blast\n');
process.stdout.write(JSON.stringify({ errors: v.validateTeam(team),
  read: [...seen].filter(s => s.startsWith('/')).map(s => fs.realpathSync(s)) }));
"""


def test_the_real_checkout_satisfies_the_preflight():
    root = sd.source_dir(str(repo_path()))
    v = sd.check(root)
    assert not v.refused, "\n".join(v.lines())


@pytest.mark.integration
def test_the_list_is_exactly_what_team_validation_reads():
    """Fails if a listed file is no longer read (a phantom path would refuse a healthy checkout) - and proves the
    control team validates, so the trace is of a run that WORKED."""
    root = sd.source_dir(str(repo_path()))
    proc = subprocess.run(["node", "-e", _TRACE_JS, root], capture_output=True, text=True,
                          timeout=scale_timeout(120))
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    assert not out["errors"], f"the positive control team did not validate: {out['errors']}"
    read = set(out["read"])
    unread = []
    for n in sd.NEEDS:
        if n.rel == "package.json":
            continue                    # read by node's own package resolver, which no JS hook sees (asserted below)
        if os.path.realpath(os.path.join(root, *n.rel.split("/"))) not in read:
            unread.append(n.rel)
    assert not unread, f"showdown_deps.NEEDS lists files team validation does not read: {unread}"
    pkg = json.loads(open(os.path.join(root, "package.json"), encoding="utf-8").read())
    assert pkg["main"] == "dist/sim/index.js", "NEEDS names the package's `main` - it moved"
    assert any(n.rel == pkg["main"] for n in sd.NEEDS)
