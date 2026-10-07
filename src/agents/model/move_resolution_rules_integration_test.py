"""gen3_move_resolution_v1 — every flag SET in `move_resolution_rules` re-derived from the simulator's own gen-3 dex.

The sets are declared in code because `data/` carries no move flags (and was frozen when the family was built), so
this is what stops them from silently drifting from Showdown: node loads the vendored `deps/pokemon-showdown`
(`Dex.mod('gen3')`, the gen-3 mod with its inheritance resolved) and every set must match EXACTLY over the gen-3
move range (num 1..354). Out-of-process (node), hence `integration`.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agents.model import move_resolution_rules as R
from utils.paths import repo_path

pytestmark = pytest.mark.integration

_JS = r"""
const {Dex} = require('./dist/sim');
const d = Dex.mod('gen3');
const all = d.moves.all().filter(m => m.num > 0 && m.num <= 354);
const foe = new Set(%s);
const ids = f => all.filter(f).map(m => m.id).sort();
console.log(JSON.stringify({
  FOE_NO_PROTECT: ids(m => foe.has(m.target) && !m.flags.protect),
  BYPASSSUB: ids(m => m.flags.bypasssub),
  SOUND: ids(m => m.flags.sound),
  REFLECTABLE: ids(m => m.flags.reflectable),
  DEFROST: ids(m => m.flags.defrost),
  FAILENCORE: ids(m => m.flags.failencore),
  STATUS_TYPE_IMMUNITY: ids(m => m.category === 'Status' && m.ignoreImmunity === false),
  priority: Object.fromEntries(all.map(m => [m.id, m.priority])),
}));
"""


@pytest.fixture(scope="module")
def dex():
    node = shutil.which("node")
    sd = repo_path("deps", "pokemon-showdown")
    if node is None:
        pytest.fail("node is not on PATH — the gen-3 dex cannot be loaded (a missing tool FAILS, never skips)")
    js = _JS % json.dumps(sorted(R.FOE_TARGETS))
    out = subprocess.run([node, "-e", js], cwd=str(sd), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


@pytest.mark.parametrize("name", ["FOE_NO_PROTECT", "BYPASSSUB", "SOUND", "REFLECTABLE", "DEFROST",
                                  "FAILENCORE", "STATUS_TYPE_IMMUNITY"])
def test_every_declared_flag_set_is_the_simulators(dex, name):
    declared, derived = set(getattr(R, name)), set(dex[name])
    assert declared == derived, (f"{name} drifted from the gen-3 dex: only declared {sorted(declared - derived)}, "
                                 f"only in the dex {sorted(derived - declared)}")


def test_every_kind_moves_priority_matches_the_facade(dex):
    """The family reads priority from the data facade; it must agree with the simulator for every kind's move."""
    from agents import gen3_data
    for ids in R.KINDS.values():
        for mid in ids:
            assert gen3_data.moves.get(mid).priority == dex["priority"][mid], mid
