"""V-RANK's driver (:mod:`main.value_search.v_rank`) on a synthetic raw state and branch states: the stratified draw,
the scorer's parts and its per-world tap (a fake search), the plan (every other legal action of every searched state
as a forced branch, one null a raw unit, shards, extension), the classifier, the outcome-free stage 0 and the read: a
planted slope is read back, and every tooth VOIDs by name. No game is played."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from main.value_search import v_rank as V
from main.value_search import v_rank_stat as ST
from main.x35 import kakuna as KA

ENGINE = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
TOKENS = {"2": "move earthquake", "3": "move protect", "6": "switch Skarmory"}
PANELS = ("heldout72_hpfix", "home72")
PARAMS = {"u_panels": ["heldout72_hpfix"], "home_panel": "home72", "look": 2, "null_bar": 0.25, "margin": 0.25,
          "rule8_eps": 0.0005, "bootstrap": {"B": 200, "seed": 20261027}, "max_drop_frac": 0.1, "engine": ENGINE,
          "preset": "r0s4_honest"}
GAMES = 6
#: Q̂ of the three legal actions; action 2 is π's
Q = {"2": 0.5, "3": 0.7, "6": 0.3}


def _log(deal: str, k: int) -> Dict[str, Any]:
    return {"format_id": "gen3ou", "seed": f"1,2,3,{k}", "names": ["A", "B"], "teams": [deal, "t"],
            "cmds": [f"CHOOSE p1 move m{k}"]}


def _capture(battle: str, tail: str) -> Dict[str, Any]:
    start = json.dumps({"formatid": "gen3ou", "seed": [1], "p1": {"name": "Gen3AIAnchor1", "team": "x"},
                        "p2": {"name": "MetaKakuna1"}})
    body = "|t:|<N>\n|turn|11\n|move|p1a: X|Earthquake\n|turn|12\n|turn|14\n|turn|20\n" + tail
    return {"tag": battle, "commands": [f"START {start}"], "chunks": [["p1", body], ["p2", body]]}


def _kept(panel: str, name: str, g: int) -> List[Dict[str, Any]]:
    deal, battle = f"{panel[:4]}{g}", f"battle-gen3ou-{g + 1}"
    out = []
    # turn 11 has one legal action, turn 40 is outside both strata: neither is ever drawn
    for k, (turn, n_legal) in enumerate(((11, 1), (12, 3), (14, 3), (20, 3), (40, 3))):
        n = turn + 1
        out.append({"schema": "gen3_input_log_v1", "key": f"su_live_v1|{deal}|t{turn}|n{n}", "index": g, "deal": deal,
                    "turn": turn, "n": n, "n_legal": n_legal, "battle": battle, "pi_argmax": 2, "tokens": TOKENS,
                    "frame_sha256": "c" * 64, "log": _log(deal, k), "at": 3, "side": "p1", "mask": [0, 0, 1, 1]})
    return out


def _raw_state(root: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Two raw 20-game units (one a panel), ``GAMES`` games each, five kept decisions a game."""
    st = root / "anchor" / "u" / "pstar"
    st.mkdir(parents=True)
    units = []
    for panel in PANELS:
        name = f"pstar__{panel}__200"
        units.append(KA.Unit(name=name, label="pstar", arm="pstar", seed=2601, teamset=panel, unit_seed=200,
                             model="/m.zip", model_sha256="ab", input_log="10-30").__dict__)
        cell = st / "cells" / name
        (cell / "capture").mkdir(parents=True)
        rows, kept = [], []
        for g in range(GAMES):
            battle = f"battle-gen3ou-{g + 1}"
            # a real unit deals each of its teams twice (game i and game i + half)
            rows.append({"battle_tag": battle, "result": "loss", "turns": 30, "engine_stamp": ENGINE,
                         "our_team": f"team_{g % (GAMES // 2)}.txt"})
            (cell / "capture" / f"{battle}.json").write_text(json.dumps(_capture(battle, "|win|B")))
            kept += _kept(panel, name, g)
        (cell / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (cell / "input_logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
        with open(st / "units.jsonl", "a") as f:
            f.write(json.dumps({"event": "end", "unit": name, "counted": True}) + "\n")
    (st / "plan.json").write_text(json.dumps({"units": units}))
    return st, V.select_states([st])


MODEL_SHA = "ab"


def _scored(sel: List[Dict[str, Any]], *, exclude: int = 0) -> List[Dict[str, Any]]:
    """Every selected state searched with Q̂ = ``Q`` (the first ``exclude`` excluded instead)."""
    out = []
    for i, r in enumerate(sorted(sel, key=V.order_key)):
        row = {k: r[k] for k in V._ROW_KEYS}
        row.update(schema=V.STATE_SCHEMA, pi_argmax=2, model_sha256=MODEL_SHA)
        if i < exclude:
            row["search"] = {"searched": False, "why": "excluded", "played": 2, "cpu_s": 1.0}
        else:
            row["search"] = {"searched": True, "played": 3, "deviation": True, "q": dict(Q), "cpu_s": 4.0,
                             "log_pi": {"2": -0.2, "3": -2.0, "6": -3.0}, "config": {"name": "r0s4_honest"},
                             "species_hit": [1.0, 0.5]}
            row["worlds"] = {a: [v - 0.1, v + 0.1] * 4 for a, v in Q.items()}
            row["referee"] = {"preset": V.ORACLE, "seeds": V.ORACLE_SEEDS, "cpu_s": 9.0,
                              "out": {"2": "+-+-+-+-", "3": "++++++++", "6": "--------"}}
        out.append(row)
    return out


def _play(bst: Path, results: Dict[int, str], *, null_result: str = "loss", g_fail: str = "", capture_tail: str = "",
          engine: str = ENGINE, drop: str = "", wrong_action: str = "") -> None:
    """Write each planned unit of one shard as a COUNTED one-game unit: the branch forcing action ``a`` scores
    ``results[a]``, a null replays the raw game (a loss); the keyword arguments plant one failure each."""
    for m in [json.loads(x) for x in (bst / "branch_plan.jsonl").read_text().splitlines()]:
        cell = bst / "cells" / m["unit"]
        if cell.exists():
            continue
        (cell / "capture").mkdir(parents=True)
        with open(bst / "units.jsonl", "a") as f:
            f.write(json.dumps({"event": "end", "unit": m["unit"], "cpu_s": 23.0}) + "\n")
        if drop and m["unit"].endswith(drop):
            with open(bst / "units.jsonl", "a") as f:
                f.write(json.dumps({"event": "end", "unit": m["unit"]}) + "\n")
            continue
        null = m["kind"] == "null"
        res = null_result if null else results[m["action"]]
        (cell / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 1, "regime_verified_decisions": True, "peer_clean": True, "non_battle_endings": [],
            "forced_end_guard": "armed", "team_source_asymmetry": False}))
        tag = "battle-gen3ou-1"
        (cell / "games.jsonl").write_text(json.dumps({"battle_tag": tag, "result": res, "turns": 30,
                                                      "engine_stamp": engine}) + "\n")
        ok = not (g_fail and m["unit"].endswith(g_fail))
        forced = 6 if wrong_action and m["unit"].endswith(wrong_action) else m["action"]
        (cell / "branch.jsonl").write_text(json.dumps({"g_branch": "pass" if ok else "fail", "forced": forced,
                                                       "index": m["game"],
                                                       "mismatch": [] if ok else ["log_sha256"]}) + "\n")
        cap = _capture(tag, "|win|B" if null or res != "win" else "|win|A")
        if capture_tail and m["unit"].endswith(capture_tail):
            cap["chunks"][0][1] = cap["chunks"][0][1].replace("Earthquake", "Protect")
        (cell / "capture" / f"{tag}.json").write_text(json.dumps(cap))


@pytest.fixture()
def world(tmp_path):
    raw, sel = _raw_state(tmp_path / "raw")
    return tmp_path, raw, sel


def _shards(out: Path) -> List[Path]:
    return sorted(out.glob("s[0-9]*"))


def _planned(world, results: Dict[int, str], shards: int = 2, **play) -> Tuple[List[Dict[str, Any]], Any, Any]:
    tmp, raw, sel = world
    sc = _scored(sel)
    V.plan_branches(sel, sc, [raw], tmp / "br", shards)
    for s in _shards(tmp / "br"):
        _play(s, results, **play)
        (s / V.PEER_DROPS_NAME).write_text("".join(json.dumps(r) + "\n" for r in V.classify_state(s)))
    rows, peer, _arts = V.load_branches(_shards(tmp / "br"))
    return rows, sc, peer


# ------------------------------------------------------------------------------------------------------ select
def test_the_draw_is_one_state_a_stratum_a_deal_by_the_smallest_hash(world) -> None:
    _tmp, raw, sel = world
    assert len(sel) == 2 * GAMES * 2 and {r["schema"] for r in sel} == {V.SELECT_SCHEMA}
    for r in sel:
        assert r["select_hash"] == V.select_hash(r["deal"], r["turn"], r["n"]) and r["raw_state"] == str(raw)
        assert r["stratum"] == V.stratum_of(r["turn"]) and r["n_legal"] >= 2
        assert r["team"] == f"team_{r['index'] % (GAMES // 2)}.txt"       # the team we piloted in ITS game
        if r["stratum"] == "A":                     # turns 12 and 14 competed: the smaller hash won
            other = 14 if r["turn"] == 12 else 12
            assert r["select_hash"] < V.select_hash(r["deal"], other, other + 1)
        else:
            assert r["turn"] == 20
    assert {(r["deal"], r["stratum"]) for r in sel} == {(f"{p[:4]}{g}", s) for p in PANELS for g in range(GAMES)
                                                         for s in "AB"}
    assert [V.stratum_of(t) for t in (9, 10, 19, 20, 30, 31)] == [None, "A", "A", "B", "B", None]
    assert len(V.select_states([raw], games=(0, 1))) == 2 * 2 * 2
    with pytest.raises(V.VRankError, match="kept twice"):
        V.select(_kept("home72", "c", 0) + _kept("home72", "c", 0))
    with pytest.raises(V.VRankError, match="two raw states"):
        V.select_states([raw, raw])


def test_a_game_that_names_no_team_of_ours_refuses_the_draw(world) -> None:
    _tmp, raw, _sel = world
    games = raw / "cells" / "pstar__home72__200" / V.GAMES_NAME
    rows = [json.loads(x) for x in games.read_text().splitlines()]
    rows[1].pop("our_team")
    games.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(V.VRankError, match="names no team of ours"):
        V.select_states([raw])


def test_the_parts_split_the_states_and_the_filters_narrow_them(world) -> None:
    _tmp, _raw, sel = world
    with pytest.raises(V.VRankError, match=r"lacks \['team'\]"):           # an older draw: refused before any search
        V.to_score([{k: v for k, v in sel[0].items() if k != "team"}, *sel[1:]], [])
    parts = [V.to_score(sel, [], part=(i, 3)) for i in range(3)]
    assert sorted(r["key"] for p in parts for r in p) == sorted(r["key"] for r in sel)
    assert [V.order_key(r) for r in parts[0]] == sorted(V.order_key(r) for r in parts[0])
    assert {r["index"] for r in V.to_score(sel, [], games=(0, 1))} == {0, 1}
    assert V.to_score(sel, [], unit_seeds=[7]) == [] and len(V.to_score(sel, [], unit_seeds=[200])) == len(sel)
    assert len(V.to_score(sel, [sel[0]["key"]])) == len(sel) - 1
    for bad in ("3/3", "x/2", "1"):
        with pytest.raises(V.VRankError, match="part"):
            V.parse_part(bad)
    with pytest.raises(V.VRankError, match="range"):
        V.parse_range("5-2")


# ------------------------------------------------------------------------------------------------------- score
def test_the_per_world_values_are_exactly_what_the_search_averaged() -> None:
    taps = [(w, a, 0.1 * a + 0.01 * w) for a in (2, 3) for w in (0, 1, 2)]
    q = {"2": 0.21, "3": 0.31}
    assert V.world_values(taps, q, "k") == {"2": [0.2, 0.21, 0.22], "3": [0.3, 0.31, 0.32]}
    with pytest.raises(V.VRankError, match="is not the search's"):
        V.world_values(taps, {"2": 0.25, "3": 0.31}, "k")
    with pytest.raises(V.VRankError, match="the leaves valued actions"):
        V.world_values(taps, {**q, "6": 0.1}, "k")
    with pytest.raises(V.VRankError, match="valued twice"):
        V.world_values(taps + [taps[0]], q, "k")
    with pytest.raises(V.VRankError, match="same worlds"):
        V.world_values(taps[:-1] + [(7, 3, 0.32)], q, "k")
    with pytest.raises(V.VRankError, match="needs >= 2"):
        V.world_values([(0, 2, 0.2), (0, 3, 0.3)], {"2": 0.2, "3": 0.3}, "k")


class _FakeEngine:
    lib = "the-search-lib"

    def leaves(self, logs, at, side, root_turn, requests, depth=None):
        return [0.1 * r.action + 0.01 * r.world for r in requests]


class _FakeSearcher:
    """``_search`` values every legal action in 2 worlds through ``self.engine.leaves`` (so the tap sees them)."""

    def __init__(self, exclude: bool = False) -> None:
        self.engine, self.forward, self.exclude = _FakeEngine(), "fwd", exclude

    def _search(self, log, at, side, row, mask, d, key, argmax):
        if self.exclude:
            return {"searched": False, "why": "excluded", "played": argmax, "cpu_s": 0.1}
        acts = sorted(int(a) for a in d["tokens"])
        reqs = [SimpleNamespace(world=w, action=a) for a in acts for w in (0, 1)]
        vals = self.engine.leaves(None, at, side, d["turn"], reqs, depth=4)
        q = {str(a): round(sum(v for r, v in zip(reqs, vals) if r.action == a) / 2, 6) for a in acts}
        return {"searched": True, "played": 6, "deviation": True, "q": q, "log_pi": {str(a): -1.0 for a in acts},
                "config": {"name": "r0s4_honest"}, "cpu_s": 0.5}


class _FakeGate:
    def __init__(self, **kw: Any) -> None:
        self.s, self.closed = _FakeSearcher(**kw), False

    def root(self, rec):
        return None, {"turn": rec["turn"], "n": rec["n"], "tokens": dict(rec["tokens"]), "frame": [0.0, 1.0]}

    def close(self) -> None:
        self.closed = True


def test_the_scorer_records_q_its_worlds_and_the_oracle_and_restores_the_engine(world, monkeypatch) -> None:
    _tmp, _raw, sel = world
    monkeypatch.setattr("agents.battle.core_obs.wrap_row", lambda frame: frame)
    seen = {}

    def fake_referee(forward, log, at, side, seeds, *, actions=None, **_kw):
        seen.update(forward=forward, seeds=list(seeds), actions=list(actions))
        return SimpleNamespace(out={a: "+-" * (len(seeds) // 2) for a in actions})

    monkeypatch.setattr("main.value_search.playout.referee", fake_referee)
    sc = V.Scorer("unused", "r0s4_honest@10", gate=_FakeGate(), model_sha256=MODEL_SHA)
    leaves = sc.g.s.engine.leaves
    row = sc.score(sel[0])
    assert sc.g.s.engine.leaves == leaves                       # the tap is gone after every state
    assert row["worlds"] == {"2": [0.2, 0.21], "3": [0.3, 0.31], "6": [0.6, 0.61]} and row["schema"] == V.STATE_SCHEMA
    assert row["referee"]["preset"] == "oracle_leaf8" and row["referee"]["seeds"] == 8
    assert row["model_sha256"] == MODEL_SHA
    assert row["referee"]["out"] == {a: "+-+-+-+-" for a in "236"} and row["referee"]["q_self"]["3"] == 0.5
    assert seen["actions"] == [2, 3, 6] and len(seen["seeds"]) == 8 and seen["forward"] == "fwd"
    assert all(row[k] == sel[0][k] for k in V._ROW_KEYS) and row["pi_argmax"] == 2
    ex = V.Scorer("unused", "r0s4_honest@10", gate=_FakeGate(exclude=True)).score(sel[0])
    assert ex["search"]["why"] == "excluded" and "worlds" not in ex and "referee" not in ex
    sc.close()
    assert sc.g.closed


def test_the_scorer_refuses_another_engine_before_any_state(world, monkeypatch, tmp_path) -> None:
    _tmp, _raw, sel = world
    (tmp_path / "sel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    monkeypatch.setattr("main.anchors.server.engine_stamp", lambda: "port=x/1;env=y/2")
    rc = V.main(["score", "--selected", str(tmp_path / "sel.jsonl"), "--model", "/m.zip", "--search",
                 "r0s4_honest@10", "--engine", ENGINE, "--out-dir", str(tmp_path / "sc")])
    assert rc == 2 and not (tmp_path / "sc").exists()


def test_a_state_scored_in_two_part_files_is_refused(world, tmp_path) -> None:
    _tmp, _raw, sel = world
    sc = _scored(sel)
    (tmp_path / "a.jsonl").write_text(json.dumps(sc[0]) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(sc[0]) + "\n" + json.dumps(sc[1]) + "\n")
    with pytest.raises(V.VRankError, match="scored twice"):
        V.load_scored(V.scored_files(tmp_path))


# -------------------------------------------------------------------------------------------------------- plan
def test_the_plan_branches_every_other_legal_action_and_one_null_a_raw_unit(world) -> None:
    tmp, raw, sel = world
    sc = _scored(sel, exclude=1)
    rep = V.plan_branches(sel, sc, [raw], tmp / "br", shards=2)
    searched = len(sel) - 1
    assert rep["units"] == rep["new"] == 2 * searched + 2 and set(rep["per_shard"]) == {"s0", "s1"}
    assert sum(c["null"] for c in rep["per_panel"].values()) == 2
    metas = [json.loads(x) for s in _shards(tmp / "br") for x in (s / "branch_plan.jsonl").read_text().splitlines()]
    br = [m for m in metas if m["kind"] == "branch"]
    assert {(m["key"], m["action"]) for m in br} == {(r["key"], a) for r in sc if r["search"]["searched"]
                                                      for a in (3, 6)}
    excluded = next(r["key"] for r in sc if not r["search"]["searched"])
    assert excluded not in {m["key"] for m in br}
    for m in (m for m in metas if m["kind"] == "null"):         # the raw unit's selected state with the smallest hash
        own = [r for r in sel if r["cell"] == m["cell"]]
        assert m["key"] == min(own, key=lambda r: r["select_hash"])["key"] and m["action"] == m["pi_argmax"] == 2
    assert len({m["shard"] for m in metas}) == 2 and len({(m["cell"], m["shard"]) for m in metas}) == 2
    u = next(x for x in KA.load_plan(tmp / "br" / "s0") if x.name == br[0]["unit"])
    argv = KA.unit_argv(u, tmp / "out", 9555)
    assert argv[argv.index("--deal-game") + 1] == str(br[0]["game"]) and "--our-force" in argv
    assert "--keep-input-logs" not in argv
    spec = json.loads(Path(u.force).read_text())
    assert spec["kind"] == "branch" and spec["pi_argmax"] == 2 and spec["action"] == br[0]["action"]


def test_a_plan_is_extended_never_changed(world) -> None:
    tmp, raw, sel = world
    sc = _scored(sel)
    first = [r for r in sc if r["index"] == 0]                  # stage 0: game 0 only
    one = V.plan_branches(sel, first, [raw], tmp / "br", shards=2)
    before = {s.name: (s / "branch_plan.jsonl").read_text().splitlines() for s in _shards(tmp / "br")}
    again = V.plan_branches(sel, first, [raw], tmp / "br", shards=2)
    assert again["new"] == 0 and again["units"] == one["units"]
    more = V.plan_branches(sel, sc, [raw], tmp / "br", shards=2)
    assert more["new"] == more["units"] - one["units"] > 0
    for s in _shards(tmp / "br"):                              # every unit planned first is still there, unchanged
        assert set(before[s.name]) <= set((s / "branch_plan.jsonl").read_text().splitlines())
        assert len(KA.load_plan(s)) == len((s / "branch_plan.jsonl").read_text().splitlines())
    with pytest.raises(V.VRankError, match="only ever extended"):
        V.plan_branches(sel, first, [raw], tmp / "br", shards=2)           # units would vanish
    moved = [dict(r, search=dict(r["search"], q={"2": 0.5, "3": 0.7})) if i == 0 else r for i, r in enumerate(sc)]
    with pytest.raises(V.VRankError, match="only ever extended"):
        V.plan_branches(sel, moved, [raw], tmp / "br", shards=2)
    with pytest.raises(V.VRankError, match="another shard count"):
        V.plan_branches(sel, sc, [raw], tmp / "br", shards=3)


def test_a_plan_refuses_while_its_supervisor_runs(world) -> None:
    from utils.filelock import try_exclusive

    tmp, raw, sel = world
    sc = _scored(sel)
    V.plan_branches(sel, [r for r in sc if r["index"] == 0], [raw], tmp / "br", shards=1)
    held = try_exclusive(tmp / "br" / "s0" / "supervisor.lock", "w")
    try:
        with pytest.raises(V.VRankError, match="supervisor is running"):
            V.plan_branches(sel, sc, [raw], tmp / "br", shards=1)
    finally:
        held.close()
    assert V.plan_branches(sel, sc, [raw], tmp / "br", shards=1)["new"] > 0


# ---------------------------------------------------------------------------------------------- classify, read
def test_a_planted_slope_is_read_back_and_every_null_is_classified(world) -> None:
    # Q̂ = .5 / .7 / .3 for actions 2 / 3 / 6; y = 0 (the raw game) / 1 / 0: Σ q_c·y_c = 0.2, Σ q_c² = 0.08, slope 2.5
    rows, sc, peer = _planned(world, {3: "win", 6: "loss"})
    assert peer["pooled"]["nulls"] == 2 and peer["pooled"]["null_peer_drops"] == 0
    assert peer["classified_attempts"] == 2                     # the nulls; no branch failed
    res = V.read(world[2], sc, rows, PARAMS)
    assert res["verdict"] == res["class"] == ST.HOLDS and res["primary"]["beta_u"]["est"] == pytest.approx(2.5)
    assert res["primary"]["delta"]["est"] == pytest.approx(0.0) and res["positions"] == {p: 2 * GAMES for p in PANELS}
    assert res["counts"]["home72"]["complete"] == 2 * GAMES and res["counts"]["home72"]["exclusion_rate"] == 0.0
    ex = res["extras"]["heldout72_hpfix"]
    assert ex["share_with_stratum_B"] == 1.0 and ex["nulls_counted"] == 1 and ex["species_hit_mean"] == 0.75
    assert ex["branch_cpu_s"] == pytest.approx(23.0 * (4 * GAMES + 1))
    assert res["reported"]["pikl_deviation_rate"]["u"]["est"] == 1.0
    assert res["reported"]["imp_pikl_pick"]["u"]["est"] == 1.0   # S′ picked action 3, which won
    flat = V.read(world[2], sc, rows, dict(PARAMS, look=1))
    assert flat["verdict"] == ST.CONTINUE and flat["delta_class"] is None


def test_a_value_that_ranks_nothing_reads_null(world) -> None:
    rows, sc, _peer = _planned(world, {3: "loss", 6: "loss"})   # every action loses: y has no spread
    res = V.read(world[2], sc, rows, dict(PARAMS, look=1))
    assert res["verdict"] == ST.NULL and res["primary"]["beta_u"]["hi"] == 0.0


def test_stage_0_reads_the_spread_and_the_costs_and_no_outcome(world) -> None:
    tmp, raw, sel = world
    sc = _scored(sel, exclude=2)
    V.plan_branches(sel, sc, [raw], tmp / "br", 2)
    rows, peer, _arts = V.load_branches(_shards(tmp / "br"), need_peer=False)      # nothing played yet
    assert peer is None and {r["status"] for r in rows} == {"pending"}
    doc = V.stage0(sel, sc, rows)
    assert doc["reads_no_outcome"] and doc["searched_positions"] == len(sel) - 2
    assert doc["sizing"]["s_bar"] == pytest.approx(0.08) and doc["sizing"]["final_blocks"] == 200
    per = doc["per_panel"]
    assert sum(p["scored"] - p["searched"] for p in per.values()) == 2
    assert per["home72"]["legal_actions_mean"] == 3.0 and per["home72"]["referee_cpu_s_mean"] == 9.0
    assert "y" not in json.dumps(doc).replace("legal", "").replace("only", "")     # no outcome field at all
    assert V.sizing(0.030)["final_blocks"] == 200 and V.sizing(0.020)["final_blocks"] == 300
    assert V.sizing(0.0151)["final_blocks"] == 400 and V.sizing(0.010) == dict(V.sizing(0.010), final_blocks=400,
                                                                                 cap_binds=True)
    assert not V.sizing(0.016)["cap_binds"]
    with pytest.raises(V.VRankError, match="not positive"):
        V.sizing(0.0)


@pytest.mark.parametrize("plant,want", [
    ({"g_fail": "__g01__branch__t20n21a3"}, "G-BRANCH"),
    ({"null_result": "win"}, "null control"),
    ({"capture_tail": "__g02__branch__t20n21a6"}, "capture"),
    ({"engine": "port=x;env=y"}, "engine"),
    ({"wrong_action": "__g03__branch__t20n21a3"}, "forced action"),
])
def test_every_branch_tooth_voids_by_name(world, plant: Dict[str, Any], want: str) -> None:
    rows, sc, _peer = _planned(world, {3: "win", 6: "loss"}, **plant)
    res = V.read(world[2], sc, rows, PARAMS)
    assert res["verdict"] == "VOID" and any(w.startswith(want) for w in res["void"]), res["void"]


def test_every_shadow_and_selection_tooth_voids_by_name(world) -> None:
    rows, sc, _peer = _planned(world, {3: "win", 6: "loss"})
    sel = world[2]

    def void(sel2=sel, sc2=sc, rows2=rows, **over: Any) -> List[str]:
        res = V.read(sel2, sc2, rows2, dict(PARAMS, **over))
        assert res["verdict"] == "VOID"
        return res["void"]

    def edit(i: int, **kv: Any) -> List[Dict[str, Any]]:
        return [dict(r, **kv) if j == i else r for j, r in enumerate(sc)]

    assert any(w.startswith("search:") for w in void(preset="r0s4"))
    assert any(w.startswith("shadow arms:") for w in void(sc2=edit(0, referee=dict(sc[0]["referee"], preset="x"))))
    short = dict(sc[0]["referee"], out={"2": "+-", "3": "++", "6": "--"})
    assert any(w.startswith("shadow arms:") for w in void(sc2=edit(0, referee=short)))
    assert any(w.startswith("shadow arms:") for w in void(sc2=[{k: v for k, v in r.items() if k != "worlds"}
                                                                for r in sc]))
    fewer = edit(0, search=dict(sc[0]["search"], q={"2": 0.5, "3": 0.7}, log_pi={"2": -1.0, "3": -1.0}),
                 worlds={a: sc[0]["worlds"][a] for a in "23"},
                 referee=dict(sc[0]["referee"], out={a: sc[0]["referee"]["out"][a] for a in "23"}))
    assert any(w.startswith("plan:") for w in void(sc2=fewer))
    assert any(w.startswith("plan:") for w in void(rows2=[r for r in rows if r is not rows[0]]))
    assert any(w.startswith("selection:") for w in void(sel2=[dict(sel[0], select_hash="0" * 64), *sel[1:]]))
    assert any(w.startswith("selection:") for w in void(sel2=[dict(sel[0], stratum="B"), *sel[1:]]))
    assert any("without the team" in w for w in void(sel2=[dict(sel[0], team=""), *sel[1:]]))
    two = [dict(r, team="other.txt") if r["deal"] == sel[0]["deal"] and r["stratum"] == "B" else r for r in sel]
    assert any("a deal on two teams" in w for w in void(sel2=two))
    assert any("on another team" in w for w in void(sc2=edit(0, team="other.txt")))
    assert any(w.startswith("panels:") for w in void(u_panels=["forum72"]))
    stray = sc + [dict(sc[0], key="su_live_v1|zz|t12|n13")]
    assert any(w.startswith("scored:") for w in void(sc2=stray))


def test_a_dropped_branch_leaves_its_position_out_and_counts_against_the_cap(world) -> None:
    rows, sc, _peer = _planned(world, {3: "win", 6: "loss"}, drop="hpfix__200__g01__branch__t20n21a3")
    res = V.read(world[2], sc, rows, PARAMS)
    assert res["verdict"] == ST.HOLDS and res["counts"]["heldout72_hpfix"]["incomplete"] == 1
    assert res["positions"]["heldout72_hpfix"] == 2 * GAMES - 1
    res = V.read(world[2], sc, rows, dict(PARAMS, max_drop_frac=0.05))
    assert res["verdict"] == "VOID" and any(w.startswith("drops:") for w in res["void"])


def test_an_unscored_state_or_a_pending_branch_refuses_the_read(world) -> None:
    rows, sc, _peer = _planned(world, {3: "win", 6: "loss"})
    with pytest.raises(V.VRankError, match="not yet scored"):
        V.read(world[2], sc[1:], rows, PARAMS)
    pend = [dict(r, status="pending") if i == 0 else r for i, r in enumerate(rows)]
    with pytest.raises(V.VRankError, match="not yet played"):
        V.read(world[2], sc, pend, PARAMS)


def test_the_read_needs_every_shards_classification(world) -> None:
    tmp, raw, sel = world
    sc = _scored(sel)
    V.plan_branches(sel, sc, [raw], tmp / "br", 2)
    for s in _shards(tmp / "br"):
        _play(s, {3: "win", 6: "loss"})
    with pytest.raises(V.VRankError, match="peer_drops.jsonl: missing"):
        V.load_branches(_shards(tmp / "br"))
    assert V.main(["classify", *[x for s in _shards(tmp / "br") for x in ("--branch-state", str(s))]]) == 0
    (tmp / "sel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    (tmp / "sc").mkdir()
    (tmp / "sc" / "part_0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc))
    (tmp / "p.json").write_text(json.dumps(PARAMS))
    args = ["--selected", str(tmp / "sel.jsonl"), "--scored-dir", str(tmp / "sc"),
            *[x for s in _shards(tmp / "br") for x in ("--branch-state", str(s))]]
    assert V.main(["read", *args, "--params", str(tmp / "p.json"), "--out", str(tmp / "v.json")]) == 0
    assert json.loads((tmp / "v.json").read_text())["verdict"] == ST.HOLDS
    assert V.main(["stage0", *args, "--out", str(tmp / "s0.json")]) == 0
    # a plan ROOT names every shard of it (what the rule's ``branch_states`` take)
    root = ["--selected", str(tmp / "sel.jsonl"), "--scored-dir", str(tmp / "sc"), "--branch-state", str(tmp / "br")]
    assert V.expand_shards([tmp / "br"]) == _shards(tmp / "br") and V.main(["classify", *root[4:]]) == 0
    assert V.main(["read", *root, "--params", str(tmp / "p.json"), "--out", str(tmp / "v2.json")]) == 0
    assert json.loads((tmp / "v2.json").read_text()) == json.loads((tmp / "v.json").read_text())
    assert V.main(["select", "--raw-state", str(raw), "--out", str(tmp / "sel.jsonl")]) == 2      # written once
