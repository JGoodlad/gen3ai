"""Unit guard for the --eval-sentinel-greedy thread: the eval worker's ``_play_unit`` builds the
pool sentinel STOCHASTIC + flat-pool-teams under ``--no-eval-sentinel-greedy`` (its
training-opponent regime) and GREEDY (argmax) + **the trainee's own teambuilder** under the
default, while the measured trainee is always greedy (built via ``build_eval_players``, whose
stochastic=False contract is tested separately). The heavy bits (model load, trainee build, the
battle gather, the turn-count read) are stubbed so this stays a pure construction test.

🚨 THE TEAM HALF IS AS LOAD-BEARING AS THE GREEDY HALF (gen3_eval_sentinel_greedy_default_v1,
2026-09-07). The trainee draws from ``Gen3Teambuilder(all_teams, bias_teams=sample_teams,
bias_prob=0.1)`` while the sentinel used to draw from the flat ``Gen3Teambuilder(all_teams)`` — a
10% tilt toward the curated sample teams that the trainee got and its opponent did not. Together
with the temperature handicap that made an eval sentinel edge a DIFFERENT experiment from the
dense ladder's edge for the same frozen pair, worth +8.9 pp [+7.0, +10.7] to the trainee. One
switch moves both, because the pair of them is exactly the condition the ladder reuses a pair
under."""

from unittest.mock import MagicMock

import main.eval_worker as ew
from agents.training.eval_sharding import EvalItem, ShardedEvalPool, SENTINEL


def _run_sentinel_unit(monkeypatch, sentinel_greedy, *, self_play_temp=0.7):
    captured = {}
    trainee_tb, opp_tb = MagicMock(name="trainee_tb"), MagicMock(name="opp_tb")
    monkeypatch.setattr(ew, "load_model_snapshot", lambda *a, **k: MagicMock(name="sentinel_model"))

    # The trainee is built by build_eval_players (greedy by contract); stub it to a numeric mock so
    # _play_unit can read its raw counters and assemble a ShardResult without a real battle.
    trainee = MagicMock(name="trainee")
    trainee.n_won_battles, trainee.n_finished_battles = 1, 2
    trainee.episode_reward_sum, trainee.n_reward_episodes = 0.0, 2
    trainee.td_residuals = lambda: []
    monkeypatch.setattr(ew, "build_eval_players", lambda model, names, *a, **k: {names[0]: trainee})
    monkeypatch.setattr(ew, "episode_length_sum", lambda p: 0.0)

    def fake_opponent(*a, **k):
        captured["sentinel_stochastic"] = k.get("stochastic")
        captured["sentinel_temperature"] = k.get("temperature")
        captured["sentinel_team"] = k.get("team")
        return MagicMock(name="opponent")
    monkeypatch.setattr(ew, "RLPlayer", fake_opponent)

    async def fake_play(*a, **k):
        return None
    monkeypatch.setattr(ew, "_play", fake_play)

    item = EvalItem("sentinel_0", SENTINEL, 2, path="snap.zip", step=1000)
    pool = ShardedEvalPool([item], shard_games=2)
    res = ew._play_unit(
        pool.units[0], pool, MagicMock(name="model"), {}, None,
        trainee_tb, opp_tb, None, MagicMock(), 1, "cpu",
        None, 1000, "t", 0, False, 0.99, self_play_temp, sentinel_greedy,
        MagicMock(name="reward_factory"))
    captured["trainee_tb"], captured["opp_tb"] = trainee_tb, opp_tb
    return captured, res


def test_sentinel_stochastic_by_default(monkeypatch):
    cap, res = _run_sentinel_unit(monkeypatch, sentinel_greedy=False)
    assert cap["sentinel_stochastic"] is True          # default: samples at temperature
    assert cap["sentinel_temperature"] == 0.7
    assert res.item_key == "sentinel_0" and res.n_finished == 2


def test_sentinel_greedy_when_flagged(monkeypatch):
    cap, _ = _run_sentinel_unit(monkeypatch, sentinel_greedy=True)
    assert cap["sentinel_stochastic"] is False          # best-vs-best: sentinel plays argmax


def test_sentinel_temperature_threaded(monkeypatch):
    cap, _ = _run_sentinel_unit(monkeypatch, sentinel_greedy=False, self_play_temp=1.3)
    assert cap["sentinel_temperature"] == 1.3           # --self-play-temp reaches the opponent


def test_sentinel_draws_the_TRAINEES_teams_when_greedy(monkeypatch):
    """The symmetry half: greedy ⇒ the sentinel gets the trainee's own (sample-biased) builder, so
    the eval edge and the dense ladder's edge for that frozen pair are the SAME experiment."""
    cap, _ = _run_sentinel_unit(monkeypatch, sentinel_greedy=True)
    assert cap["sentinel_team"] is cap["trainee_tb"]


def test_sentinel_keeps_the_flat_pool_builder_when_stochastic(monkeypatch):
    """--no-eval-sentinel-greedy is byte-identical to the pre-2026-09-07 behaviour on BOTH axes."""
    cap, _ = _run_sentinel_unit(monkeypatch, sentinel_greedy=False)
    assert cap["sentinel_team"] is cap["opp_tb"]


def test_the_two_halves_move_together(monkeypatch):
    """One switch, two properties — a run can never be greedy-but-asymmetric (or the reverse) from
    the CLI. The ladder still READS the two independently off the eval row rather than assuming
    they agree, so a future flag split needs no ladder change."""
    for greedy in (True, False):
        cap, _ = _run_sentinel_unit(monkeypatch, sentinel_greedy=greedy)
        assert cap["sentinel_stochastic"] is (not greedy)
        assert (cap["sentinel_team"] is cap["trainee_tb"]) is greedy
