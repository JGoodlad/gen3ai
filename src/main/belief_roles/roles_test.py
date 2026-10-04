"""The role set and substitute pairs come from SMOGON data only, and follow §4.1 / §4.2 R3's rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from main.belief_roles import roles as R
from utils.paths import repo_path

#: Runs the derivation in a FRESH interpreter under an `open` audit hook (nothing cached from an
#: earlier test can hide a read) and prints every file it opened under `data/`.
_PROBE = r"""
import json, sys
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(args[0] if isinstance(args[0], str) else args[0].decode())
sys.addaudithook(hook)
from main.belief_roles.roles import derive
rs = derive()
print(json.dumps({"opened": opened, "n_roles": len(rs.roles), "n_pairs": len(rs.pairs)}))
"""


def test_role_derivation_reads_smogon_data_only():
    data = str(repo_path("data")) + os.sep
    env = dict(os.environ, PYTHONPATH=str(repo_path("src")))
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env,
                         cwd=str(repo_path()), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    under_data = sorted({os.path.relpath(p, data) for p in rec["opened"] if p.startswith(data)})
    assert rec["n_roles"] > 0
    # Nothing from the team pool (data/teams/...), and nothing outside the declared Smogon/dex files.
    assert not [p for p in under_data if not p.startswith("pokemon" + os.sep)], under_data
    assert {os.path.basename(p) for p in under_data} <= set(R.SOURCES), under_data
    # The Smogon move priors and teammate lifts ARE the sources (a derivation reading neither is not this one).
    assert {"gen3_move_priors.json", "gen3_teammate_priors.json"} <= {os.path.basename(p) for p in under_data}


def test_role_bar_and_hidden_power_collapse():
    """Every role clears 0.25 expected carriers per team, every other move does not; typed Hidden
    Powers are ONE role (237), recomputed here independently of `derive`'s loop."""
    from agents import gen3_data
    from agents.model.dex_ids import build_species_usage_prior

    rs = R.derive()
    usage = build_species_usage_prior(R.N_SPECIES).double().numpy() * R.TEAM
    expect = {}
    for sid in gen3_data.species.base_form_ids():
        sd = gen3_data.species.get(sid)
        if sd is None or not (0 < sd.num < R.N_SPECIES):
            continue
        for mid, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(mid)
            if md is None:
                continue
            num = R.HP_NUM if mid.startswith("hiddenpower") else int(md.num)
            expect[num] = expect.get(num, 0.0) + float(usage[sd.num]) * float(p)
    want = sorted(m for m, e in expect.items() if e >= R.ROLE_BAR)
    assert rs.nums == want
    assert R.HP_NUM in rs.nums
    assert not [m for m in rs.nums if 355 <= m <= 370], "a typed Hidden Power is its own role"
    for r in rs.roles:
        assert abs(r.carriers - expect[r.num]) < 1e-9


def test_substitute_pairs_follow_r3():
    from agents.model.belief_tables import build_species_cooccur_prior

    rs = R.derive()
    carry = R.smogon_carry_table()
    _, ll = build_species_cooccur_prior(R.N_SPECIES)
    assert rs.pairs, "no substitute pair at all — the R3 read would be empty"
    for p in rs.pairs:
        assert p.role in rs.nums and p.s1 < p.s2
        assert carry[p.s1][p.role] >= R.SUBSTITUTE_CARRY and carry[p.s2][p.role] >= R.SUBSTITUTE_CARRY
        assert float(ll[p.s2, p.s1]) < 0 and float(ll[p.s1, p.s2]) < 0
    # deterministic, and the sha stamps the set
    assert R.derive().sha256() == rs.sha256()
