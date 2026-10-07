"""The IN-LOOP eval producers on the COUNT ledger (eval unit U2, ``cycle_ledger``): the eval cycle of both callbacks and
the T6 SPRT promotion write ``gen3_eval_count_row_v2`` rows — readable through a declared ``ReaderDecl``, draws
counted, one request per (cycle x regime), SPRT batches under their request id and batch index with a duplicate
refused, a decision row per candidate — and an aborted or failed cycle leaves the ledger consistent (``audit``
passes). Each test FAILS on a revert of the producer change (no ``_cycle_ledger`` plumbing ⇒ no rows).

The evaluator here is a FAKE with the executor's two outputs (one published ``ShardResult`` per unit and one
``game_log`` row per finished game); the real core is ``cycle_ledger_integration_test`` (a sim-tier test)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agents.training import cycle_ledger as CL
from agents.training import eval_ledger as L
from agents.training import mirrored_pairs as MP
from agents.training.eval_ledger import audit as AU
from agents.training.trace_result import DRAW, LOSS, WIN

TEAMS = [f"team-{i}" for i in range(6)]
CYCLE_ROWS = L.ReaderDecl(
    name="cycle_ledger_test.rows", purposes=frozenset({"cycle"}),
    regime=L.RegimeFilter(protocol=CL.PROTOCOL, play="greedy", seat_rule="fixed_p1"),
    requests="any", selection="include", flags_ok=frozenset(), inference="conditional")
SPRT_ROWS = L.ReaderDecl(
    name="cycle_ledger_test.sprt", purposes=frozenset({"promotion"}),
    regime=L.RegimeFilter(protocol=CL.PROTOCOL, play="greedy", mirrored=True, seat_rule="fixed_p1"),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")


class _Abort(BaseException):
    """What a stop honoured at a safe point does inside the cycle: the process leaves it and never returns."""


def _result(key: str, g: int, pair: bool) -> str:
    """A deterministic outcome per (opponent, game) with every bucket present; a mirrored pair's games differ."""
    k = (g // 2 if pair else g) + len(key)
    return [WIN, LOSS, DRAW, WIN][(k + (g % 2 if pair else 0)) % 4]


class _FakeEvaluator:
    """The executor's outputs for any plan (module docs). ``abort_after`` raises :class:`_Abort` after that many
    units; ``fail`` raises the core's own ``EvalCoreError``; ``lie`` publishes a wrong win count for one item."""

    def __init__(self, *, abort_after: Optional[int] = None, fail: bool = False, lie: Optional[str] = None,
                 all_wins: bool = False):
        self.turn_limit, self.n = 250, 4
        self.trainee_builder = SimpleNamespace(packed_teams=TEAMS, bias_packed_teams=TEAMS[:2], bias_prob=0.1)
        self.opp_builder = SimpleNamespace(packed_teams=TEAMS, bias_packed_teams=[], bias_prob=0.0)
        self.team_table = SimpleNamespace(teams=TEAMS)
        self.abort_after, self.fail, self.lie, self.all_wins = abort_after, fail, lie, all_wins
        self.cycles: List[Dict[str, Any]] = []

    def opponent_builder(self, item, sentinel_greedy):
        from agents.training.eval_sharding import SENTINEL

        return self.trainee_builder if (item.kind == SENTINEL and sentinel_greedy) else self.opp_builder

    def run_cycle(self, pool, run_dir, **kw):
        from agents.training.eval_sharding import ShardResult
        from agents.training.eval_sharding.units import game_range
        from agents.training.rust_eval.executor import EvalCoreError

        self.cycles.append(kw)
        if self.fail:
            raise EvalCoreError("a battle was QUARANTINED")
        mirrored = bool(getattr(pool, "mirrored", False))
        log = kw.get("game_log")
        for i, u in enumerate(pool.units):
            if self.abort_after is not None and i >= self.abort_after:
                raise _Abort("stop honoured at the safe point")
            games = list(game_range(u, pool.shard_games, mirrored))
            res = {g: (WIN if self.all_wins else _result(u.item.key, g, mirrored)) for g in games}
            for g in games:
                a, b = (g // 2) % len(TEAMS), (g // 2 + 1) % len(TEAMS)
                sw = mirrored and g % 2 == 1
                if log is not None:
                    log.append({"item": u.item.key, "kind": u.item.kind, "game": g, "result": res[g],
                                "end_turn": 20 + g, "teams": [b, a] if sw else [a, b], "swapped": sw,
                                "margins": [0.5, 1e-4 if g == 1 else 0.3], "opp": [], "script": "x" * 1000})
            won = sum(r == WIN for r in res.values()) + (1 if u.item.key == self.lie else 0)
            pool.publish(run_dir, ShardResult(
                unit_id=u.unit_id, item_key=u.item.key, worker_id=0, n_won=won, n_finished=len(games),
                sum_reward=0.0, n_episodes=len(games), sum_ep_len=20.0 * len(games), duration_sec=1.0,
                td_residuals=[], traces_written=0, traces_won=0, n_drawn=sum(r == DRAW for r in res.values()),
                pair_counts=MP.pair_counts([MP.result_points(res[g]) for g in games]) if mirrored else None))
        return _Stats(len(pool.units))


class _Stats:
    def __init__(self, n):
        self.n = n

    def as_dict(self):
        return {"games": self.n, "units": self.n, "host_steps": 1, "trainee_decisions": 10 * self.n,
                "p2_policy_decisions": 0, "traces": 0, "near_ties": 0,
                "seconds": {k: 0.0 for k in ("load", "stage", "submit", "drain", "act", "core", "finish", "trace",
                                             "total")}, "lifecycle": {}}


def _model(ev):
    m = MagicMock()
    m.save = lambda base: Path(base + ".zip").write_bytes(b"trainee weights")
    m.gamma = 0.99
    m.policy = SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    m._rust_collector = SimpleNamespace(evaluator=ev, cfg=SimpleNamespace(run_seed=5))
    m.get_env.return_value.env_method.return_value = [(100, 5, 2)]
    return m


def _ledger(tmp_path, **writer_kw) -> CL.CycleLedger:
    w = L.LedgerWriter(tmp_path / "ledger", producer=CL.PRODUCER, **writer_kw)
    return CL.CycleLedger(None, run_label="run_test", commit="c0ffee", writer=w)


def _bot_cb(tmp_path, ev, led, *, mirrored=False):
    from agents.training.eval_callback import PerOpponentEvalCallback

    cb = PerOpponentEvalCallback(model_dir=str(tmp_path / "run"), eval_games=4, eval_mirrored_pairs=mirrored,
                                 cycle_ledger=led)
    cb.model = _model(ev)
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000
    cb._init_callback()
    return cb


def _sentinel_files(tmp_path, n=2):
    from agents.training.snapshot_pool import SnapshotEntry

    out = []
    for i in range(n):
        p = tmp_path / "pool" / f"snapshot_{(i + 1) * 1_000_000:012d}.zip"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"sentinel {i}".encode())
        out.append(SnapshotEntry(path=p, step=(i + 1) * 1_000_000))
    return out


@pytest.fixture
def no_snapshot_loads(monkeypatch):
    """The fake evaluator plays no policy: the sentinels' snapshot LOADS are stubbed (their FILES stay real — the
    ledger digests them)."""
    from agents.training.eval_sharding import SENTINEL
    from agents.training.rust_eval import launch

    monkeypatch.setattr(launch, "load_sentinels",
                        lambda pool, model, safe_point=None: {it.key: object() for it in pool.items if it.kind == SENTINEL})


def _selfplay_cb(tmp_path, ev, led, *, mirrored=False, sprt=False):
    from agents.training.selfplay_callback import SelfPlayCallback

    pool = MagicMock()
    pool.load_persisted_win_rate.return_value = 0.0
    pool.is_empty.return_value = False
    entries = _sentinel_files(tmp_path)
    pool.__len__.return_value = len(entries)
    pool.sentinel_entries.return_value = list(reversed(entries))
    pool.entry_weight.return_value = 1.0
    pool.load_summary.return_value = {}
    cb = SelfPlayCallback(pool=pool, model_dir=str(tmp_path / "run"), eval_games=4, n_sentinels=2,
                          eval_sentinel_greedy=True, eval_mirrored_pairs=mirrored, promotion_sprt=sprt,
                          snapshot_ladder_games=0, cycle_ledger=led)
    cb.model = _model(ev)
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000
    cb._init_callback()
    return cb


def _audit_ok(led):
    rep = AU.audit(led.root)
    assert rep.ok, rep.problems
    return rep


# ---------------------------------------------------------------------------------------------- the cycle
def test_a_bot_cycle_writes_valid_v2_rows_readable_through_a_declared_read_beside_eval_results(tmp_path):
    from agents.training.eval_callback import eval_opponent_names

    led = _ledger(tmp_path)
    cb = _bot_cb(tmp_path, _FakeEvaluator(), led)
    cb._on_step()
    got = L.read(CYCLE_ROWS, root=led.root)
    bots = list(eval_opponent_names())
    assert sorted(r["opponent"]["id"] for r in got.rows) == sorted(f"bot:{b}" for b in bots)
    for r in got.rows:
        assert L.validate_row(r) == [], r
        assert r["purpose"] == "cycle" and r["request"]["kind"] == "cycle" and r["request"]["batch"] == 0
        assert r["request"]["id"] == f"run_test:cycle:2000000:{r['regime']['regime_id']}"
        reg = r["regime"]
        assert (reg["protocol"], reg["seat_rule"], reg["play"], reg["player_temp"]) == \
            (CL.PROTOCOL, "fixed_p1", "greedy", None)
        assert (reg["opponent_play"], reg["opponent_temp"], reg["mirrored"]) == ("sampled", L.BOT_NATIVE_TEMP, False)
        c = r["counts"]
        assert c["w"] + c["l"] + c["d"] == 4 and c["d"] >= 1 and c["aborted"] == 0, c   # draws COUNTED
        assert r["pairs"] is None and r["player"]["kind"] == "checkpoint" and r["opponent"]["kind"] == "bot"
        assert r["compute"]["near_tie_games"] == [1] and r["compute"]["digest_margin"] == CL.DIGEST_MARGIN
        assert r["compute"]["outcome_digest_all"] != r["compute"]["outcome_digest"]     # game 1 is in one only
    # the DUAL write: eval_results.jsonl still carries the cycle, with the same W / finished per bot
    row = [json.loads(x) for x in (tmp_path / "run" / "eval_results.jsonl").read_text().splitlines()][-1]
    for r in got.rows:
        name = r["opponent"]["id"][4:]
        assert list(row["counts"][name]) == [r["counts"]["w"], r["counts"]["w"] + r["counts"]["l"] + r["counts"]["d"]]
    rep = _audit_ok(led)
    assert rep.counts["live_rows"] == len(bots)
    s = AU.show(led.root)
    assert s["requests_open"] == [] and len(s["requests_closed"]) == 1          # one regime ⇒ one request, done


def test_a_selfplay_cycle_splits_its_requests_by_regime_and_mirrored_rows_carry_the_pentanomial(tmp_path,
                                                                                             no_snapshot_loads):
    led = _ledger(tmp_path)
    cb = _selfplay_cb(tmp_path, _FakeEvaluator(), led, mirrored=True)
    cb._on_step()
    by = list(L.read_by_regime(CYCLE_ROWS, root=led.root).values())
    kinds = {}
    for rd in by:
        ids = {r["request"]["id"] for r in rd.rows}
        assert len(ids) == 1, ids                                   # one request per regime
        kinds[rd.rows[0]["regime"]["opponent_play"]] = [r["opponent"]["kind"] for r in rd.rows]
    assert sorted(kinds) == ["greedy", "sampled"] and set(kinds["greedy"]) == {"checkpoint"}
    for rd in by:
        for r in rd.rows:
            assert L.validate_row(r) == [] and r["regime"]["mirror_rule"] == MP.SCHEMA
            assert r["pairs"]["n_pairs"] == 2 and sum(r["pairs"]["counts"]) == 2
    sent = [r for rd in by for r in rd.rows if r["opponent"]["kind"] == "checkpoint"]
    assert sorted(r["opponent"]["step"] for r in sent) == [1_000_000, 2_000_000]
    assert all(r["player"]["sha256"] == sent[0]["player"]["sha256"] for r in sent)
    _audit_ok(led)


def test_a_cycle_that_fails_on_the_core_writes_no_row_and_cancels_its_requests(tmp_path):
    led = _ledger(tmp_path)
    cb = _bot_cb(tmp_path, _FakeEvaluator(fail=True), led)
    cb._on_step()
    assert len(L.read(CYCLE_ROWS, root=led.root).rows) == 0
    s = AU.show(led.root)
    assert s["requests_open"] == [] and len(s["requests_closed"]) == 1
    _audit_ok(led)


def test_an_abort_mid_cycle_writes_nothing_and_a_restart_replays_the_same_seed_block_cleanly(tmp_path):
    dead = 4_000_000                                             # the aborted process's pid (dead after the abort)
    led = _ledger(tmp_path, pid=dead)
    cb = _bot_cb(tmp_path, _FakeEvaluator(abort_after=3), led)
    with pytest.raises(_Abort):
        cb._on_step()
    assert len(L.read(CYCLE_ROWS, root=led.root).rows) == 0     # nothing partial
    rep = _audit_ok(led)
    assert rep.counts["live_claims"] == 9                         # its claims are left for the void rule
    # the restart: a new process re-evaluates the same step; the dead writer's claims are voided, then replayed
    led2 = _ledger(tmp_path, alive=lambda pid: pid != dead)
    cb2 = _bot_cb(tmp_path, _FakeEvaluator(), led2)
    cb2._on_step()
    got = L.read(CYCLE_ROWS, root=led2.root)
    assert len(got.rows) == 9 and {r["row_id"].rsplit(":", 1)[0] for r in got.rows} == {led2.writer.writer_id}
    _audit_ok(led2)


def test_a_game_log_that_disagrees_with_the_published_results_is_refused_and_writes_no_row(tmp_path):
    led = _ledger(tmp_path)
    cb = _bot_cb(tmp_path, _FakeEvaluator(lie="random"), led)
    with pytest.raises(CL.CycleLedgerError, match="random: the game log disagrees"):
        cb._on_step()
    assert len(L.read(CYCLE_ROWS, root=led.root).rows) == 0       # every cell is checked before any row lands
    _audit_ok(led)


def test_the_trainer_builds_one_ledger_at_startup_and_hands_it_to_the_eval_callback(tmp_path, run_archive):
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser

    p = build_parser()
    args = p.parse_args(["--steps", "15000000"])
    resolve_config(args, p)
    bundle = build_callbacks(
        args=args, model_dir=str(tmp_path / "run_x"), annealing_mode=False, _pool=None, _fixed_opponents=None,
        _bot_weight_vec=None, OPPONENT_NAMES=(), _specialist_team_str=None, _promote_threshold=0.6,
        _heuristic_floor=0.0, _sp_start_wr=0.5, _sp_full_wr=0.9)
    led = bundle.eval_callback._cycle_ledger
    assert isinstance(led, CL.CycleLedger) and led.root == run_archive / "_ledger" and led.run_label == "run_x"


# ---------------------------------------------------------------------------------------------- the SPRT
def _sprt_pending(cb, tmp_path):
    entries = cb._pool.sentinel_entries()
    snap = tmp_path / "cand.zip"
    snap.write_bytes(b"candidate weights")
    return {"snapshot": str(snap),
            "sentinels": [{"label": f"sentinel_{i}", "path": str(e.path), "step": e.step} for i, e in enumerate(entries)]}


def test_sprt_batches_carry_their_request_and_batch_index_a_duplicate_is_refused_and_one_decision_lands(
        tmp_path, no_snapshot_loads):
    from agents.training import sprt as S

    led = _ledger(tmp_path)
    cb = _selfplay_cb(tmp_path, _FakeEvaluator(all_wins=True), led, sprt=True)
    cb._sprt_cfg = S.SprtConfig(min_pairs=0, batch_pairs=4, max_pairs=40)
    cb._eval_shard_games = 4
    cb._sprt_begin(6_000_000, _sprt_pending(cb, tmp_path))
    rid = "run_test:sprt:6000000"
    got = L.read(SPRT_ROWS, root=led.root, request_id=rid)
    assert len(got.rows) >= 2
    batches = sorted({r["request"]["batch"] for r in got.rows})
    assert batches == list(range(len(batches)))                  # every batch, in order, under ONE request
    for r in got.rows:
        assert L.validate_row(r) == [] and r["purpose"] == "promotion" and r["request"]["kind"] == "sprt"
        assert r["request"]["id"] == rid and r["seed"]["schedule_key"] == "inloop:sprt:6000000"
        assert r["pairs"] is not None and r["counts"]["l"] == 0
    # per batch: one row per sentinel, and the pairs the test folded
    assert {(r["request"]["batch"], r["opponent"]["id"]) for r in got.rows} == \
        {(b, f"run_test:pool@{s}") for b in batches for s in (1_000_000, 2_000_000)}
    # the decided request is CLOSED: no further batch can be claimed into it (the append-level duplicate refusal is
    # the next test)
    first = got.rows[0]
    with pytest.raises(L.RequestSpecError, match="is closed"):
        led.writer.claim(rid, batch=first["request"]["batch"], player=first["player"]["sha256"],
                         opponent=first["opponent"]["sha256"], regime_id=first["regime"]["regime_id"],
                         expected_wall_s=1)
    dec = [json.loads(x) for f in (led.root / "decisions").glob("decisions.*.jsonl") for x in f.read_text().splitlines()]
    assert len(dec) == 1 and dec[0]["kind"] == "promotion" and dec[0]["verdict"] == "accept"
    assert dec[0]["request_id"] == rid and dec[0]["subject"] == first["player"]["sha256"]
    assert dec[0]["consumed"]["count"] == len(got.rows) and dec[0]["consumed"]["digest"] == got.digest
    assert AU.show(led.root)["requests_open"] == []
    _audit_ok(led)


def test_a_duplicate_sprt_row_is_refused_at_the_append(tmp_path, no_snapshot_loads):
    """The append's own refusal, independent of the claim: a second row on a recorded batch key."""
    from agents.training import sprt as S

    led = _ledger(tmp_path)
    cb = _selfplay_cb(tmp_path, _FakeEvaluator(all_wins=True), led, sprt=True)
    cb._sprt_cfg = S.SprtConfig(min_pairs=0, batch_pairs=4, max_pairs=40)
    cb._eval_shard_games = 4
    cb._sprt_begin(6_000_000, _sprt_pending(cb, tmp_path))
    rid = "run_test:sprt:6000000"
    r0 = dict(L.read(SPRT_ROWS, root=led.root, request_id=rid).rows[0])
    unit = (rid, r0["request"]["batch"], r0["player"]["sha256"], r0["opponent"]["sha256"], r0["regime"]["regime_id"])
    claim = next(c for c in led.writer._state().claims.values() if c.unit == unit)   # the row's own, live claim
    r0["row_id"] = led.writer.next_row_id()
    with pytest.raises(L.DuplicateBatchError):
        led.writer.append_row(r0, claim)
    _audit_ok(led)
