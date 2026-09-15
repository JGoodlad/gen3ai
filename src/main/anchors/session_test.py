"""THE WATCHDOG — the half of this tool that exists because a dead peer looks like silence.

Hazard H-B, measured 2026-09-14: when our side forfeits at turn 250 the battle ends while Metamon's
env is mid-step; its next `step` raises "Battle is already finished", its long-tail handler
force-resets (sending a fresh challenge) and then **calls itself**, never converging — ~985
recursion levels, then a `RecursionError` that kills the process. A harness without a watchdog then
sits with a live client, no opponent, and nothing at all to report. 4 forfeits in 800 games hit 3 of
16 half-cells.

Every other way a head-to-head dies has the same symptom, and none of them is an exception on our
side:

* a team Showdown REJECTS — the challenge never becomes a battle (de-risk H1, matched-regime H-A);
* a 19-character username — `action.php` returns a refusal the peer accepts as an assertion, logs
  "Successfully logged in", and blocks forever (foul-play H1);
* a challenge aimed at a user who is not yet online — Showdown drops it silently.

So the gate is: **four NAMED causes, each raised as a `SeriesFailure`, never a silent zero.** Every
test here drives the failing branch rather than asserting the happy one, because a watchdog that
cannot fire is indistinguishable from a series that never stalls.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from main.anchors.peers import PeerPlan
from main.anchors.session import (
    BattleRecord,
    OurSideState,
    install_our_side,
    SeriesFailure,
    await_peer_ready,
    build_team_source,
    load_team_texts,
    log_has,
    log_tail,
    watch,
)


class FakePeer:
    """A stand-in for `subprocess.Popen` with only what the watchdog reads.

    A local object, not a patched module symbol: the watchdog takes the process as a PARAMETER, so
    there is nothing to stub and no way for this to reach the wrong code path.
    """

    def __init__(self, returncode=None) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def die(self, rc: int = 1) -> None:
        self.returncode = rc


def _plan(tmp_path: Path, text: str = "") -> PeerPlan:
    log = tmp_path / "peer.log"
    log.write_text(text)
    return PeerPlan(label="metamon:SmallRL", argv=["true"], cwd=tmp_path, env={},
                    ready_pattern=r"Made Challenge Env", log_path=log)


def _record(i: int) -> BattleRecord:
    return BattleRecord(battle_tag=f"battle-gen3ou-{i}", turns=40, won=True, finished=True,
                        hit_forfeit_limit=False, our_team="t", n_decisions=38, n_defaults=0,
                        n_redecides=0, t_finished=time.time())


# ------------------------------------------------------------------------- cause: peer_exited
async def test_a_peer_that_dies_mid_series_is_named_not_silence(tmp_path: Path) -> None:
    """THE H-B CASE. The failure must name the peer, its rc and how far the series got."""
    plan = _plan(tmp_path, "RecursionError: maximum recursion depth exceeded\n")
    proc = FakePeer()
    state = OurSideState()
    state.records.extend(_record(i) for i in (1, 2, 3))
    state.last_progress = time.time()

    async def kill_soon() -> None:
        await asyncio.sleep(0.1)
        proc.die(1)

    asyncio.ensure_future(kill_soon())
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(proc, plan, state, expected=100,
                    first_game_timeout_s=60.0, progress_timeout_s=60.0, poll_s=0.02)
    assert excinfo.value.cause == "peer_exited"
    detail = excinfo.value.detail
    assert "3/100" in detail
    assert "rc=1" in detail
    assert "RecursionError" in detail, "the peer's own log tail must be in the failure"
    assert "H-B" in detail, "the named hazard belongs in the message a human reads"


async def test_a_peer_exit_after_the_expected_count_is_NOT_a_failure(tmp_path: Path) -> None:
    """The control. The peer normally exits first at the end of a series; that is not a death."""
    plan = _plan(tmp_path)
    state = OurSideState()
    state.records.extend(_record(i) for i in range(1, 5))
    await watch(FakePeer(returncode=0), plan, state, expected=4,
                first_game_timeout_s=60.0, progress_timeout_s=60.0, poll_s=0.02)


# ------------------------------------------------------- causes: no_first_game / no_progress
async def test_nothing_finishing_at_all_is_no_first_game(tmp_path: Path) -> None:
    """A rejected team and a dropped challenge both look exactly like this: a live pair, no games."""
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(FakePeer(), _plan(tmp_path, "waiting...\n"), OurSideState(), expected=10,
                    first_game_timeout_s=0.05, progress_timeout_s=600.0, poll_s=0.02)
    assert excinfo.value.cause == "no_first_game"
    assert "0/10" in excinfo.value.detail
    assert "HANG" in excinfo.value.detail


async def test_games_stopping_part_way_is_no_progress_and_carries_the_count(tmp_path: Path) -> None:
    state = OurSideState()
    state.records.extend(_record(i) for i in (1, 2))
    state.last_progress = time.time() - 100.0
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(FakePeer(), _plan(tmp_path), state, expected=10,
                    first_game_timeout_s=600.0, progress_timeout_s=1.0, poll_s=0.02)
    assert excinfo.value.cause == "no_progress"
    assert "2/10" in excinfo.value.detail


async def test_a_series_that_keeps_finishing_games_is_never_failed(tmp_path: Path) -> None:
    """THE POSITIVE CONTROL. A watchdog that fired on a healthy series would be worse than none."""
    state = OurSideState()
    plan = _plan(tmp_path)
    proc = FakePeer()

    async def play() -> None:
        for i in range(1, 5):
            await asyncio.sleep(0.05)
            state.records.append(_record(i))
            state.last_progress = time.time()

    asyncio.ensure_future(play())
    await watch(proc, plan, state, expected=4, first_game_timeout_s=30.0,
                progress_timeout_s=30.0, poll_s=0.02)
    assert len(state.records) == 4


# ------------------------------------------------------------------- cause: peer_never_ready
async def test_a_peer_that_never_prints_its_banner_is_named(tmp_path: Path) -> None:
    """The ACCEPTOR must be online before the challenger's first `/challenge` — Showdown drops a
    challenge aimed at a user who is not online and the challenger then waits forever."""
    with pytest.raises(SeriesFailure) as excinfo:
        await await_peer_ready(FakePeer(), _plan(tmp_path, "loading weights\n"), timeout_s=0.1,
                               poll_s=0.02)
    assert excinfo.value.cause == "peer_never_ready"
    assert "Made Challenge Env" in excinfo.value.detail


async def test_a_peer_that_dies_before_coming_online_is_peer_exited(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        await await_peer_ready(FakePeer(returncode=2), _plan(tmp_path, "AssertionError: flash\n"),
                               timeout_s=30.0, poll_s=0.02)
    assert excinfo.value.cause == "peer_exited"
    assert "rc=2" in excinfo.value.detail


async def test_readiness_is_the_peers_own_banner_and_nothing_else(tmp_path: Path) -> None:
    await await_peer_ready(FakePeer(), _plan(tmp_path, "...\nMade Challenge Env\n"),
                           timeout_s=5.0, poll_s=0.02)


# ------------------------------------------------------------------------------ the log helpers
def test_log_helpers_answer_honestly_when_there_is_no_log(tmp_path: Path) -> None:
    missing = tmp_path / "nope.log"
    assert log_has(missing, "anything") is False
    assert "no log" in log_tail(missing)


# ------------------------------------------------------------------------------- the team source
def test_load_team_texts_strips_the_trailing_blank_lines_that_hung_a_series(
        tmp_path: Path) -> None:
    """The third line of defence behind the fork fix and the pack guard. Cheap, so it stays."""
    (tmp_path / "a.gen3ou_team").write_text("Skarmory @ Leftovers\n- Spikes\n\n\n")
    (tmp_path / "index.csv").write_text("filename\na.gen3ou_team\n")
    items = load_team_texts(tmp_path)
    assert [name for name, _ in items] == ["a.gen3ou_team"], (
        "index.csv is bookkeeping, not a team — counting it reports a team-source asymmetry that "
        "does not exist"
    )
    assert not items[0][1].endswith("\n")


def test_an_empty_team_directory_is_a_named_failure_not_an_empty_pool(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        build_team_source({"kind": "dir", "path": str(tmp_path)}, seed=1, draw_log=[])
    assert excinfo.value.cause == "no_teams"


def test_an_unknown_team_source_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        build_team_source({"kind": "telepathy"}, seed=1, draw_log=[])
    assert excinfo.value.cause == "bad_team_source"


def test_a_battle_record_reads_a_tie_as_a_tie_and_not_a_loss(tmp_path: Path) -> None:
    """poke-env's own three flags: `won is None` with `finished` true is a TIE. Metamon's own CSV
    cannot express that (its `won` is a boolean, de-risk H3), which is why OUR side is the
    authority on every result."""
    rec = _record(1)
    assert rec.result == "win"
    assert BattleRecord(**{**rec.__dict__, "won": None}).result == "tie"
    assert BattleRecord(**{**rec.__dict__, "won": False}).result == "loss"


# ───────────────────────────── our side is a BOT, or a snapshot that needs the foreign loader ───
# Both of these were added for the 2026-09-14 external-anchor CALIBRATION, and both exist because
# the failure they prevent is silent or near-silent:
#
#   * the per-battle observer assumed `battle.strict_view()`, which is our vendored fork's
#     `Gen3Battle`. A roster bot is a plain poke-env `Player` and gets a plain `Battle`, so the
#     observer raised inside the finished-callback and the read came back "0/4 games completed"
#     while the server log showed `finished=2`. Games that happened, recorded as games that did not.
#   * a cross-run frozen snapshot FAILS a bare `MaskablePPO.load` — the extractor is rebuilt from
#     the zip's own `policy_kwargs` and handed to the CURRENT `ExtractorBuild`. Every
#     `ai_v9_29_rev1_0823` node dies with `unexpected keyword argument 'threat_prob_outspeed'`
#     while every `ai_v12`/`ai_v13` node loads fine, so a campaign that spans eras hits it on
#     SOME cells only — the worst shape of bug to find by running the whole thing.
class _PlainBattle:
    """A poke-env `Battle` as a roster bot sees it: NO `strict_view`."""

    def __init__(self, tag: str = "battle-gen3ou-1", turn: int = 17) -> None:
        self.battle_tag = tag
        self.turn = turn
        self.won = True
        self.finished = True


class _ForkBattle(_PlainBattle):
    """Our fork's `Gen3Battle`: the observer must prefer the strict view when it exists."""

    def strict_view(self):
        return _PlainBattle(self.battle_tag, self.turn)


def test_the_observer_records_a_BOT_battle_which_has_no_strict_view() -> None:
    """🚨 THE BOT-SIDE REGRESSION, in the exact pairing that produced it: a roster bot instance
    and a plain poke-env `Battle`. Before the fix the observer raised inside the finished-callback
    and the read came back `0/4 games completed` while the server had played them all."""
    from agents.training.eval_callback import eval_opponent_class

    cls = eval_opponent_class("staller")
    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object(), our_side="bot:staller")
    try:
        bot = object.__new__(cls)
        cls._battle_finished_callback(bot, _PlainBattle("battle-gen3ou-42", 23))
    finally:
        undo()
    assert len(state.records) == 1
    rec = state.records[0]
    assert (rec.battle_tag, rec.turns, rec.result) == ("battle-gen3ou-42", 23, "win")
    assert rec.hit_forfeit_limit is False


def test_the_observer_still_prefers_the_strict_view_for_our_own_player() -> None:
    """The other half of the same pairing — the fix must not have quietly stopped using the fork's
    read-model for `RLPlayer`, which is where the turn count is server-authoritative."""
    from agents.inference.player import RLPlayer

    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object())
    try:
        dummy = object.__new__(RLPlayer)
        dummy._stall_loggers = {}
        dummy._trackers = {}
        dummy._battles = {}
        seen = {}
        battle = _ForkBattle("battle-gen3ou-9", 251)
        battle.strict_view = lambda: seen.setdefault("v", _PlainBattle("battle-gen3ou-9", 251))
        RLPlayer._battle_finished_callback(dummy, battle)
    finally:
        undo()
    assert "v" in seen, "the strict view was not consulted for an RLPlayer battle"
    assert state.records[0].turns == 251
    # 251 >= the 250-turn trainer forfeit limit: a cap forfeit reads as an ordinary loss to the
    # server, and only the turn count separates it.
    assert state.records[0].hit_forfeit_limit is True


def test_the_observer_is_installed_on_the_bot_class_and_removed_again() -> None:
    """The patch must reach the class the BOT instances resolve through, and the undo must restore
    the INHERITED lookup — a bot subclass defines no callback of its own, so writing the base
    method onto it would leave a permanent attribute behind."""
    import main.play as play
    from agents.training.eval_callback import eval_opponent_class

    cls = eval_opponent_class("staller")
    before_owns = "_battle_finished_callback" in cls.__dict__
    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object(),
                            our_side="bot:staller")
    try:
        assert "_battle_finished_callback" in cls.__dict__
        assert play.build_model_player.__name__ == "build_bot_player"
    finally:
        undo()
    assert ("_battle_finished_callback" in cls.__dict__) is before_owns


@pytest.mark.parametrize("mode, expected", [("bare", "bare"), ("auto", "bare")])
def test_a_successful_bare_load_is_recorded_as_bare(monkeypatch, mode, expected) -> None:
    import main.play as play

    monkeypatch.setattr(play, "load_policy", lambda path, device: ("model", path, device))
    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object(), model_loader=mode)
    try:
        out = play.load_policy("/some/snapshot.zip", "cpu")
    finally:
        undo()
    assert out == ("model", "/some/snapshot.zip", "cpu")
    assert state.model_loader == expected


def test_auto_falls_back_to_the_foreign_loader_and_says_which_one_ran(monkeypatch, capsys) -> None:
    """🚨 A fallback nobody can see is a cell whose policy nobody can identify. The row carries
    `model_loader`, and this is the test that it is set from what HAPPENED, not from the flag."""
    import main.play as play
    from main.anchors import session as session_mod

    def bare(path, device):
        raise TypeError("ExtractorBuild.__init__() got an unexpected keyword argument "
                        "'threat_prob_outspeed'")

    monkeypatch.setattr(play, "load_policy", bare)
    monkeypatch.setattr(session_mod, "foreign_loader", lambda path, device: "foreign-model")
    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object(), model_loader="auto")
    try:
        assert play.load_policy("/some/snapshot.zip", "cpu") == "foreign-model"
    finally:
        undo()
    assert state.model_loader == "foreign"
    out = capsys.readouterr().out
    assert "bare load failed" in out and "threat_prob_outspeed" in out


def test_bare_mode_does_NOT_silently_fall_back(monkeypatch) -> None:
    """`--model-load bare` is the LADDER's semantics: play the model you were handed, or refuse."""
    import main.play as play
    from main.anchors import session as session_mod

    def bare(path, device):
        raise TypeError("unexpected keyword argument 'threat_prob_outspeed'")

    called = []
    monkeypatch.setattr(play, "load_policy", bare)
    monkeypatch.setattr(session_mod, "foreign_loader",
                        lambda path, device: called.append(path) or "nope")
    state = OurSideState()
    undo = install_our_side(state, {"kind": "pool"}, 7, 250, object(), model_loader="bare")
    try:
        with pytest.raises(TypeError):
            play.load_policy("/some/snapshot.zip", "cpu")
    finally:
        undo()
    assert called == []


# ─────────────────────────────────── the per-game view when NEITHER side is ours to observe ───
_PEER_CSV = (
    "Player Username, Team File, Opponent Username, Result, Turn Count, Battle ID\n"
    "MetaSynthRLV21,/teams/aaa.gen3ou_team,MetaSmallRL1,WIN,20,1969304393\n"
    "MetaSynthRLV21,/teams/bbb.gen3ou_team,MetaSmallRL1,LOSS,30,0968496524\n"
    "MetaSynthRLV21,/teams/ccc.gen3ou_team,MetaSmallRL1,WIN,24,9600367016\n"
)


def test_a_peer_pair_cell_reads_its_games_from_metamons_own_battle_log(tmp_path: Path) -> None:
    from main.anchors.session import read_peer_battles

    (tmp_path / "battle_log_MetaSynthRLV21_gen3ou.csv").write_text(_PEER_CSV)
    recs = read_peer_battles(tmp_path)
    assert [r.result for r in recs] == ["win", "loss", "win"]
    assert [r.turns for r in recs] == [20, 30, 24]
    assert recs[0].our_team == "aaa.gen3ou_team"
    assert recs[0].battle_tag == "battle-1969304393"


def test_an_absent_battle_log_is_zero_games_and_not_an_exception(tmp_path: Path) -> None:
    """The watchdog polls this file while the pair is still coming up; an empty answer must mean
    'nothing yet', so that the NAMED no_first_game failure is what fires rather than a traceback."""
    from main.anchors.session import read_peer_battles

    assert read_peer_battles(tmp_path) == []


@pytest.mark.asyncio
async def test_a_peer_pair_that_both_exit_short_is_named_not_silence(tmp_path: Path) -> None:
    """Both peers can exit rc=0 having played fewer games than asked — a dropped challenge ends
    the acceptor's wait cleanly. That must be `short_series`, never a quiet partial n."""
    from main.anchors.session import watch_peer_pair

    (tmp_path / "battle_log_x_gen3ou.csv").write_text(_PEER_CSV)
    plans = [PeerPlan(label=f"p{i}", argv=[], cwd=tmp_path, env={}, ready_pattern="x",
                      log_path=tmp_path / f"p{i}.log") for i in (1, 2)]
    procs = [_DeadProc(0), _DeadProc(0)]
    state = OurSideState()
    with pytest.raises(SeriesFailure) as excinfo:
        await watch_peer_pair(procs, plans, tmp_path, state, expected=10,
                              first_game_timeout_s=60, progress_timeout_s=60, poll_s=0.01)
    assert excinfo.value.cause == "short_series"
    assert "3/10" in excinfo.value.detail


@pytest.mark.asyncio
async def test_a_peer_pair_that_reaches_the_count_is_never_failed(tmp_path: Path) -> None:
    from main.anchors.session import watch_peer_pair

    (tmp_path / "battle_log_x_gen3ou.csv").write_text(_PEER_CSV)
    plans = [PeerPlan(label=f"p{i}", argv=[], cwd=tmp_path, env={}, ready_pattern="x",
                      log_path=tmp_path / f"p{i}.log") for i in (1, 2)]
    state = OurSideState()
    await watch_peer_pair([_LiveProc(), _LiveProc()], plans, tmp_path, state, expected=3,
                          first_game_timeout_s=60, progress_timeout_s=60, poll_s=0.01)
    assert len(state.records) == 3


@pytest.mark.asyncio
async def test_a_dead_peer_in_a_pair_is_peer_exited_with_its_rc(tmp_path: Path) -> None:
    from main.anchors.session import watch_peer_pair

    plans = [PeerPlan(label=f"p{i}", argv=[], cwd=tmp_path, env={}, ready_pattern="x",
                      log_path=tmp_path / f"p{i}.log") for i in (1, 2)]
    state = OurSideState()
    with pytest.raises(SeriesFailure) as excinfo:
        await watch_peer_pair([_LiveProc(), _DeadProc(1)], plans, tmp_path, state, expected=10,
                              first_game_timeout_s=60, progress_timeout_s=60, poll_s=0.01)
    assert excinfo.value.cause == "peer_exited"
    assert "rc=1" in excinfo.value.detail


class _DeadProc:
    def __init__(self, rc: int) -> None:
        self.returncode = rc

    def poll(self):
        return self.returncode


class _LiveProc:
    returncode = None

    def poll(self):
        return None
