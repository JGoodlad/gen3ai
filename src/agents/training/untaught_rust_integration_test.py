"""``untaught_meter.play_cells`` on the REAL Rust eval core (CPU, tiny): poke-env retirement P2.

What the move off the poke-env bridge must keep, each a test that FAILS on revert:

* the games run on the Rust EVAL CORE — every cell is stamped ``transport = "rust_eval"`` with the core build, and
  a fresh process that plays a cell never imports the poke-env PLAYER path (``agents.inference.player`` /
  ``utils.bridge.local_battle_runner``, what the old ``play_cells`` played through);
* CRN across refs — two refs see the SAME opponent team and battle seed game by game (the cycle seed is a function
  of (seed, team index) only), and the pilot plays its PINNED team in every game;
* the TRAINING regime samples the PILOT (the executor's opt-in ``trainee_temp``, the keyed draw on
  ``STREAM_TRAINEE``): some pilot decisions leave the argmax; the EVAL regime plays the argmax (no draw recorded);
* a cell replays bit for bit at one compute;
* a game at the turn limit is a DRAW, never a timeout: ``attempted == finished``.

Seeded PERTURBED-fresh checkpoints (``rust_eval.offline.build_models``), T2 eager, the in-process
(ffi) core, the emission self-check build. The heavy plays are MODULE fixtures (the tier budget is per test call).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agents.training import untaught_meter as um

pytestmark = [pytest.mark.sim, pytest.mark.integration]

GAMES = 6
SEED = 3


def _compute():
    from main.h2h.play import Compute

    return Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                   profile="selfcheck")


@pytest.fixture(scope="module")
def built():
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    return True


@pytest.fixture(scope="module")
def setup(built, tmp_path_factory):
    from agents.training.rust_eval import offline as PAR

    dst = tmp_path_factory.mktemp("untaught_rust") / "run_untaught_rust"
    with PAR.declared_torch_state(1):
        a, (b,), _cfg = PAR.build_models(dst, n_sentinels=1)
    ref_a = um.resolve_ref(a, label="A")
    ref_b = um.resolve_ref(b, label="B")
    opp = um.resolve_ref(b, label="OPPONENT", role="opponent")
    teams = um.load_team_manifest(str(um.DEFAULT_TEAMS_MANIFEST))[:2]
    return ref_a, ref_b, opp, teams


@pytest.fixture(scope="module")
def stochastic(setup):
    ref_a, ref_b, opp, teams = setup
    log, info = {}, {}
    cells = um.play_cells([ref_a, ref_b], teams, opp, games_per_team=GAMES, seed=SEED, compute=_compute(),
                          game_log=log, info=info)
    again_log = {}
    again = um.play_cells([ref_a], teams[:1], opp, games_per_team=GAMES, seed=SEED, compute=_compute(),
                          game_log=again_log)
    return cells, log, info, again, again_log


@pytest.fixture(scope="module")
def greedy(setup):
    ref_a, _ref_b, opp, teams = setup
    log, info = {}, {}
    cells = um.play_cells([ref_a], teams[:1], opp, games_per_team=GAMES, seed=SEED, compute=_compute(),
                          stochastic=False, game_log=log, info=info)
    return cells, log, info


def test_every_cell_is_stamped_rust_eval_and_a_turn_limit_game_is_a_draw_not_a_timeout(stochastic, setup):
    cells, _log, info, _a, _al = stochastic
    _ra, _rb, _o, teams = setup
    assert sorted(cells) == ["A", "B"] and all(sorted(t) == sorted(x.key for x in teams) for t in cells.values())
    for t in cells.values():
        for c in t.values():
            assert c.transport == "rust_eval"
            assert c.attempted == c.finished == GAMES and c.timeouts == 0
            assert c.wins + c.ties + c.losses == GAMES
            assert c.turn_limit_draws is not None and c.turn_limit_draws <= c.ties
            assert len(c.opp_teams) == GAMES
    assert info["transport"] == "rust_eval" and info["encoder"] == "rust" and info["core_stamp"]
    assert info["eval_regime"]["opponent_item_kind"] == "sentinel"
    assert um.cells_transport(cells) == "rust_eval"


def test_CRN_two_refs_see_the_same_opponent_team_and_battle_seed_game_by_game(stochastic, setup):
    _cells, log, _info, _a, _al = stochastic
    _ra, _rb, _o, teams = setup
    for t in teams:
        a, b = log[("A", t.key)], log[("B", t.key)]
        assert [g["game"] for g in a] == [g["game"] for g in b] == list(range(GAMES))
        assert [g["teams"][1] for g in a] == [g["teams"][1] for g in b], "the opponent's team differs by ref"
        assert [g["seed"] for g in a] == [g["seed"] for g in b], "the battle seed differs by ref"
        assert len({g["teams"][0] for g in a + b}) == 1, "the pilot did not play its ONE pinned team"
    seeds0 = [g["seed"] for g in log[("A", teams[0].key)]]
    seeds1 = [g["seed"] for g in log[("A", teams[1].key)]]
    assert seeds0 != seeds1, "two teams must not share a schedule (the cycle seed moves with the team)"


def test_the_training_regime_SAMPLES_the_pilot_and_the_eval_regime_plays_its_argmax(stochastic, greedy):
    _cells, log, _info, _a, _al = stochastic
    off = [g["sampled_off_argmax"] for g in log[("A", "U_61590463")]]
    assert all(x is not None for x in off), "a sampled pilot logs its argmax beside every action"
    assert sum(off) > 0, "at T = 1 the pilot never left its argmax: it is not sampling"
    _gc, glog, ginfo = greedy
    assert all(g["sampled_off_argmax"] is None for g in glog[("A", "U_61590463")])
    assert ginfo["eval_regime"]["opponent_item_kind"] == "fixed"
    assert ginfo["eval_regime"]["pilot_play"] == "greedy"


def test_a_cell_replays_bit_for_bit_at_one_compute(stochastic):
    cells, log, _info, again, again_log = stochastic
    assert again["A"]["U_61590463"].to_json() == cells["A"]["U_61590463"].to_json()
    assert again_log[("A", "U_61590463")] == log[("A", "U_61590463")]


def test_a_fresh_process_plays_a_cell_without_a_poke_env_player_or_the_bridge(setup):
    """No poke-env ``Player`` is CONSTRUCTED (the old path built two ``RLPlayer`` s per cell) and the bridge's battle
    runner is never imported. (Whether ``agents.inference.player`` is IMPORTED is no signal here — the test plants
    a hook on poke-env's ``Player`` itself, which imports the package.)"""
    ref_a, _rb, opp, teams = setup
    code = f"""
import json, sys
import poke_env.player.player as PP
built = []
_init = PP.Player.__init__
def _counting(self, *a, **k):
    built.append(type(self).__name__)
    return _init(self, *a, **k)
PP.Player.__init__ = _counting
from agents.training import untaught_meter as um
from main.h2h.play import Compute
ref = um.resolve_ref({ref_a.zip_path!r}, label="A")
opp = um.resolve_ref({opp.zip_path!r}, label="O", role="opponent")
teams = um.load_team_manifest({str(um.DEFAULT_TEAMS_MANIFEST)!r})[:1]
info = {{}}
cells = um.play_cells([ref], teams, opp, games_per_team=2, seed=0, info=info,
                      compute=Compute(device="cpu", backend="eager", n_envs=2, threads=1, torch_threads=1,
                                      front="ffi", profile="selfcheck"))
print(json.dumps({{"transport": cells["A"][teams[0].key].transport, "stamp": info["core_stamp"],
                  "players_built": built,
                  "bridge": "utils.bridge.local_battle_runner" in sys.modules}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["transport"] == "rust_eval" and got["stamp"]
    assert got["players_built"] == [], f"a Rust-core play built poke-env players: {got['players_built']}"
    assert not got["bridge"], "the poke-env bridge battle runner was imported by a Rust-core play"
