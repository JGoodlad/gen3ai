"""The three core seams M5 Lane H added (``src/rust_env``), on real battles:

* a bot route declared ``"streams": "episode"`` re-seeds its bot from the game's battle seed, so the
  SAME staged game plays the SAME battle on any env after any history (the default ``"env"`` rule does
  not — the teeth);
* ``Core::finished`` reports every ended episode once, in env order, with its winner, end turn,
  forfeit and input log — identically through the FFI and the process front end;
* ``rust_env_trace_json`` replays that log into records whose text the reconstruction record
  reproduces through the offline replay driver (``replay_battle(impl="rust")``), and ends where the
  core ended.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def lib():
    from agents.training.rust_rollout.testkit import build_selfcheck
    from utils.rust_env import ffi

    build_selfcheck()
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


@pytest.fixture(scope="module")
def teams():
    from agents.training.rust_rollout.testkit import pool_builder

    return list(pool_builder(12).packed_teams)


def _core(lib, teams, n, streams, front="ffi"):
    from utils.rust_env import ffi, proc
    from utils.rust_env import protocol as P

    spec = P.spec_json(n=n, threads=min(n, 2), teams=teams, names=("rhone", "rhtwo"), decision_tense=False,
                       switch_freeze=False, turn_limit=250, refusal_budget=4, bank_dir=None,
                       opponents=[{"kind": "bot", "bot": "staller", "seed": 11, "streams": streams},
                                  {"kind": "bot", "bot": "random", "seed": 12, "streams": streams}])
    if front == "ffi":
        return ffi.FfiCore(spec, lib=lib)
    return proc.ProcCore(spec, binary=proc.default_path("selfcheck"), nan_poison=True)


def _play(core, games, env_of):
    """Play ``games`` [(teams, seed, route)] — game k on env ``env_of[k]``, the others FILLER (the same
    route, other seeds) — p1 its highest legal action. Returns {game: finished row} + every row."""
    c = core.cols
    n = int(core.n)
    queue = {e: [k for k in range(len(games)) if env_of[k] == e] for e in range(n)}
    cur, staged = {}, {}
    filler = ((0, 1), [9, 9, 9, 9], 1)

    def nxt(e):
        return queue[e].pop(0) if queue[e] else None

    def stage(e, k):
        t, s, r = games[k] if k is not None else filler
        c["ep_team"][e] = t
        c["ep_seed"][e] = s
        c["ep_opp"][e] = r
        staged[e] = k

    for e in range(n):
        stage(e, nxt(e))
    core.reset()
    last = c["episode"].copy()
    for e in range(n):
        cur[e] = staged[e]
        stage(e, nxt(e))
    out, rows = {}, []
    while len(out) < len(games):
        for e in np.flatnonzero(c["need"][:, 0] == 1):
            c["action"][e, 0] = int(np.flatnonzero(c["mask"][e, 0]).max())
        core.step()
        fin = core.finished()
        rows.append(fin)
        for f in fin:
            if cur[f["env"]] is not None:
                out[cur[f["env"]]] = f
        ep = c["episode"].copy()
        for e in np.flatnonzero(ep != last):
            cur[e] = staged[e]
            stage(e, nxt(e))
        last = ep
    return out, rows


def _games():
    return [((2 * k % 12, (2 * k + 5) % 12), [100 + k, 200 + k, 300 + k, 400 + k], k % 2) for k in range(4)]


def test_a_per_episode_bot_route_plays_the_same_game_on_any_env_after_any_history(lib, teams):
    g = _games()
    core_a, core_b = _core(lib, teams, 1, "episode"), _core(lib, teams, 4, "episode")
    try:
        a, _ = _play(core_a, g, [0, 0, 0, 0])                 # one env, in order
        b, _ = _play(core_b, list(reversed(g)), [3, 3, 1, 0])  # other envs, other histories
    finally:
        core_a.close()
        core_b.close()
    for k in range(4):
        fa, fb = a[k], b[3 - k]
        assert fa["script"].split("\n", 1)[1] == fb["script"].split("\n", 1)[1], f"game {k} differs"
        assert (fa["winner"], fa["end_turn"]) == (fb["winner"], fb["end_turn"])


def test_the_default_env_stream_rule_depends_on_history(lib, teams):
    """Teeth: without ``"streams": "episode"`` the same staged games replay differently."""
    g = _games()
    core_a, core_b = _core(lib, teams, 1, "env"), _core(lib, teams, 4, "env")
    try:
        a, _ = _play(core_a, g, [0, 0, 0, 0])
        b, _ = _play(core_b, list(reversed(g)), [3, 3, 1, 0])
    finally:
        core_a.close()
        core_b.close()
    assert any(a[k]["script"].split("\n", 1)[1] != b[3 - k]["script"].split("\n", 1)[1] for k in range(4))


def test_finished_is_the_same_through_both_front_ends(lib, teams):
    g = _games()
    got = {}
    for front in ("ffi", "proc"):
        core = _core(lib, teams, 2, "episode", front)
        try:
            _out, rows = _play(core, g, [0, 1, 0, 1])
        finally:
            core.close()
        got[front] = rows
        for fin in rows:
            envs = [f["env"] for f in fin]
            assert envs == sorted(envs) and len(set(envs)) == len(envs), "one row per env per op, env order"
    assert got["ffi"] == got["proc"]


def test_the_trace_replays_to_the_cores_end_and_its_records_are_the_replay_drivers_text(lib, teams, tmp_path):
    from utils.bridge.reconstruction import ReconstructionRecord, replay_battle
    from utils.rust_env import ffi

    core = _core(lib, teams, 2, "episode")
    try:
        out, _ = _play(core, _games()[:2], [0, 1])
    finally:
        core.close()
    for k, f in out.items():
        t = ffi.trace(lib, f["script"], commit="test", label=f"g{k}")
        assert (t["winner"], t["turn"]) == (f["winner"], f["end_turn"])
        header = json.loads(t["records"]["p1"].split("\n", 1)[0])
        assert header["event_schema"] == "gen3_core_event_v1" and header["viewer"] == "p1"
        texts = [json.loads(line)["text"] for line in t["records"]["p1"].splitlines()[1:]]
        rec = ReconstructionRecord.from_dict(dict(t["recon"], battle_tag=f"g{k}", trainee_username="rhone"))
        rep = replay_battle(rec, impl="rust")
        lines = [ln for ch in rep.p1_chunks for ln in ch.split("\n") if ln.startswith("|") and not ln.startswith("|t:|")]
        assert lines == [x for x in texts if not x.startswith("|t:|")]
        p = tmp_path / f"g{k}.p1.jsonl.gz"
        with gzip.open(p, "wt") as fh:
            fh.write(t["records"]["p1"])
    with pytest.raises(Exception, match="trace"):
        ffi.trace(lib, "START {}\n", commit="x", label="bad")
