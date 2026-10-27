"""``metamon_side --team-skip`` (S-U's per-game units, ``main.anchors --deal-game``): the peer's SEEDED team draw makes
and discards its first N draws, so its first battle pilots the team the same seed's draw N gave game N of a whole
series; the play is untouched. Run on the script loaded by PATH in a fresh stdlib interpreter, against a fake
``metamon.env.wrappers.TeamSet`` (the script never imports metamon at load time)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from utils.paths import src_path

SCRIPT = src_path("main", "anchors", "peer_scripts", "metamon_side.py")


def _draws(tmp_path, seed: int, skip: int, n: int, blocked=()) -> dict:
    teams = tmp_path / "teams"
    teams.mkdir(exist_ok=True)
    for i in range(9):
        (teams / f"t{i}.gen3ou_team").write_text(f"team {i}")
    code = textwrap.dedent(f"""
        import importlib.util, json, os, sys, types
        wrappers = types.ModuleType("metamon.env.wrappers")

        class TeamSet:
            def __init__(self):
                self.team_files = [os.path.join({str(teams)!r}, f) for f in os.listdir({str(teams)!r})]
            def block_team(self, t):
                return t in {list(blocked)!r}
        wrappers.TeamSet = TeamSet
        for name, mod in (("metamon", types.ModuleType("metamon")), ("metamon.env", types.ModuleType("metamon.env")),
                          ("metamon.env.wrappers", wrappers)):
            sys.modules[name] = mod
        sys.modules["metamon"].env = sys.modules["metamon.env"]
        sys.modules["metamon.env"].wrappers = wrappers
        spec = importlib.util.spec_from_file_location("metamon_side", {str(SCRIPT)!r})
        ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
        # the peer yields only VERIFIED Showdown packs (install_showdown_pack); here each file's pack is its text
        ms._PACKS.update({{os.path.abspath(f): open(f).read() for f in TeamSet().team_files}})
        log = []
        ms.install_seeded_team_draw({seed}, log, skip={skip})
        ts = TeamSet()
        got = [ts.yield_team() for _ in range({n})]
        print(json.dumps({{"got": got, "log": log}}))
    """)
    proc = subprocess.run([sys.executable, "-c", code], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_skip_yields_the_whole_streams_later_draws_and_logs_only_the_played_ones(tmp_path):
    whole = _draws(tmp_path, seed=5, skip=0, n=6)
    for k in (1, 3, 5):
        part = _draws(tmp_path, seed=5, skip=k, n=6 - k)
        assert part["got"] == whole["got"][k:]
        assert part["log"] == whole["log"][k:]


def test_a_skip_counts_blocked_redraws_exactly_as_the_whole_stream_does(tmp_path):
    whole = _draws(tmp_path, seed=8, skip=0, n=6, blocked=("team 2", "team 4"))
    assert not {"team 2", "team 4"} & set(whole["got"])
    part = _draws(tmp_path, seed=8, skip=4, n=2, blocked=("team 2", "team 4"))
    assert part["got"] == whole["got"][4:]
