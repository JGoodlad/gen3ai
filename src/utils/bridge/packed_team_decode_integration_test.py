"""Exhaustive packed-team decode cross-validation — Python vs the sim itself.

A reconstruction record's ``>player`` lines carry both sides' packed teams
verbatim (the strings the teambuilder produced — the replay byte-identity fuzz
guards that round-trip). Review tooling decodes them with
``reconstruction.decode_packed_team`` (poke-env's ``parse_packed_team`` under a
normalized dict shape). This test pins that decode against the GROUND TRUTH —
the sim's own ``Teams.unpack`` — for **every team in the project pool** (the
full universe of strings a record can ever contain: training, eval, and bias
draws all come from this pool), field by field: species, item, ability, move
order, nature, EVs, IVs, level.

Exhaustive-by-construction beats a sampling fuzz here: the input space is the
finite, fixed team pool, so we just sweep all of it. The gen3-critical cases
(Hidden Power IV spreads, explicit Atk-0 IVs, omission-defaults) are covered by
whatever the pool actually plays. Needs the Node bridge; no battles, no server.
"""

import json
import subprocess

import pytest

from utils.contention import scale_timeout

from utils.bridge.reconstruction import _STAT_ORDER, decode_packed_team
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# Unpack every packed string with the sim's own parser, in ONE node call.
# Defaults mirror the sim's interpretation: a missing/blank EV is 0, a
# missing/blank IV is 31, level omitted is 100 (Teams.unpack leaves omitted
# fields undefined — the packed format's omission convention).
_NODE_UNPACK = """
const path = require('path');
const { Teams } = require(path.resolve('deps/pokemon-showdown/dist/sim/teams'));
const toId = (s) => ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]+/g, '');
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
let input = '';
process.stdin.on('data', (c) => { input += c; });
process.stdin.on('end', () => {
  const teams = JSON.parse(input);
  const out = teams.map((packed) => {
    const team = Teams.unpack(packed);
    if (!team) return null;
    return team.map((m) => ({
      species: toId(m.species || m.name),
      item: toId(m.item),
      ability: toId(m.ability),
      moves: m.moves.map(toId),
      nature: toId(m.nature),
      evs: Object.fromEntries(STATS.map((s) => [s, (m.evs && m.evs[s]) || 0])),
      ivs: Object.fromEntries(STATS.map((s) => [s, (m.ivs && m.ivs[s] !== undefined) ? m.ivs[s] : 31])),
      level: m.level || 100,
    }));
  });
  process.stdout.write(JSON.stringify(out));
});
"""


@pytest.mark.integration
def test_every_pool_team_decodes_identically_in_python_and_sim():
    packed_teams = Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams
    assert packed_teams, "empty team pool"

    proc = subprocess.run(
        ["node", "-e", _NODE_UNPACK],
        input=json.dumps(packed_teams).encode(),
        capture_output=True, timeout=scale_timeout(120),
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-1500:]
    sim_teams = json.loads(proc.stdout.decode())

    n_mons = 0
    mismatches = []
    for t_idx, (packed, sim_team) in enumerate(zip(packed_teams, sim_teams)):
        assert sim_team is not None, f"sim failed to unpack pool team {t_idx}"
        py_team = decode_packed_team(packed)
        assert len(py_team) == len(sim_team), f"team {t_idx}: mon count differs"
        for m_idx, (py, sm) in enumerate(zip(py_team, sim_team)):
            n_mons += 1
            for field in ("species", "item", "ability", "moves", "nature",
                          "evs", "ivs", "level"):
                if py[field] != sm[field]:
                    mismatches.append(
                        f"team {t_idx} mon {m_idx} ({py['species']}) {field}: "
                        f"python={py[field]!r} sim={sm[field]!r}")

    assert not mismatches, (
        f"{len(mismatches)} decode mismatches across {n_mons} mons:\n  "
        + "\n  ".join(mismatches[:20]))
    # Belt-and-braces: the sweep covered real Hidden Power IV spreads (the
    # gen3 case where IV decode errors would silently change a move's type).
    assert any(
        any(v != 31 for v in m["ivs"].values())
        for packed in packed_teams[:50] for m in decode_packed_team(packed)
    ), "pool sample contains no non-default IVs — sweep lost its teeth?"
    print(f"\n{len(packed_teams)} teams / {n_mons} mons: python decode == sim decode")
