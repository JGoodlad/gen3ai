"""A CONTRACT test per producer (design_evaluation.md §0b.4 item 7): each producer's row builder emits a row that
passes ``validate_row`` and appends under a claim. ``main.h2h``'s builder is pinned in ``main/h2h/play_test.py``;
this file pins the second migrated writer, the bot round robin (a committed measurement script, loaded by path)."""
from __future__ import annotations

import importlib.util

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import testkit as K
from utils.paths import repo_path

BOT_RR = repo_path("designs", "research_state", "measurements", "bot_base_ratings_2026-10-03", "bot_rr.py")


@pytest.fixture(scope="module")
def bot_rr():
    spec = importlib.util.spec_from_file_location("bot_rr_contract", BOT_RR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_the_bot_round_robin_writes_valid_v2_rows_under_claims(bot_rr, tmp_path):
    reg = bot_rr.regime()
    assert reg["protocol"] == bot_rr.PROTOCOL == "gen3_eval_protocol_v1_bot_rr" and reg["seat_rule"] == "balanced"
    assert reg["player_temp"] == L.BOT_NATIVE_TEMP and reg["team_set"].startswith("ts:")
    pa, pb = bot_rr.bot_block("random"), bot_rr.bot_block("heuristic")
    assert pa["kind"] == "bot" and pa["id"] == "bot:random"
    w = L.LedgerWriter(tmp_path / "ledger", producer=bot_rr.PRODUCER)
    req = w.open_request(bot_rr.default_request_id(0), kind="anchor_read", purpose="anchor", regime_id=reg["regime_id"],
                         protocol=bot_rr.PROTOCOL, spec={"producer": "bot_rr", "batch_pairs": 2, "schedule_seed": 0})
    c = w.claim(req["request_id"], batch=0, player=pa["sha256"], opponent=pb["sha256"], regime_id=reg["regime_id"],
                expected_wall_s=10)
    pc = [0, 0, 1, 0, 1]
    cnt = K.counts_of(pc)
    t1, t2 = L.team_id("x"), L.team_id("y")
    teams = {t1: {"p": [4, cnt["w"]], "o": [0, 0]}, t2: {"p": [0, 0], "o": [4, cnt["l"]]}}
    outcomes = [(0, "W", 20), (1, "L", 21), (2, "W", 30), (3, "W", 31)]
    row = bot_rr.build_row(writer=w, request=req, commit="c0ffee", player=pa, opponent=pb, regime=reg, batch=0,
                           n_pairs=2, schedule_seed=0, key=bot_rr.item_key("random", "heuristic"), cseed=123,
                           w=cnt["w"], l=cnt["l"], d=cnt["d"], pc=pc, teams=teams, outcomes=outcomes,
                           t_start=L.utc_now(), wall=1.0, a_p1_games=2, timeouts=0)
    assert L.validate_row(row) == []
    assert row["compute"]["outcome_digest"] == L.outcome_digest(outcomes) and row["compute"]["digest_margin"] is None
    w.append_row(row, c)
    assert bot_rr.done_units(str(tmp_path / "ledger"), req["request_id"], reg["regime_id"]) == \
        {(bot_rr.item_key("random", "heuristic"), 0): 2}


def test_the_banked_v1_rows_of_both_writers_read_upgraded():
    """The 120 P0 rows and the 1,296 bot round-robin rows stay v1 on disk and read as v2 (§0b.2), each under its
    writer's protocol — the verbatim copies a later backfill would carry."""
    decl = L.ReaderDecl(name="contract.banked", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                        selection="include", flags_ok=frozenset({"digest_unrecorded"}), inference="conditional")
    meas = repo_path("designs", "research_state", "measurements")
    p0 = L.read(decl, root=meas / "x5_p0_h2h_2026-10-03" / "rows")
    rr = L.read(decl, root=meas / "bot_base_ratings_2026-10-03" / "ledger")
    assert len(p0) == 120 and {r["regime"]["protocol"] for r in p0} == {"gen3_eval_protocol_v1_h2h"}
    assert len(rr) == 1296 and {r["regime"]["seat_rule"] for r in rr} == {"balanced"}
    assert all(r["flags"] == ["digest_unrecorded"] and r["request"] is None for r in [*p0, *rr])
