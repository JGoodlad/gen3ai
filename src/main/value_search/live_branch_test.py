"""The deviation-only live read's driver and estimator (:mod:`main.value_search.live_branch`) on a synthetic raw state
and branch state: the registered order and quota, the branch plan (one forced branch per S′ deviation, one null control
per raw unit, each a ``--deal-game`` unit with ``--our-force``), and the read: a planted effect reads GO, no effect
reads STOP on rule 8's boundary, and every tooth (G-BRANCH, the capture prefix, the null control, the forced action,
the engine, the plan, the drop cap) VOIDs by name. No game is played."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from main.value_search import live_branch as LB
from main.value_search.sign_gate import select_hash
from main.x35 import kakuna as KA

ENGINE = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
TOKENS = {"2": "move earthquake", "3": "move protect", "6": "switch Skarmory"}
PANELS = ("heldout72_hpfix", "home72")
PARAMS = {"u_panels": ["heldout72_hpfix"], "home_panel": "home72", "bar": 0.0, "rule8_eps": 0.0005,
          "bootstrap": {"B": 1000, "seed": 20261028}, "max_drop_frac": 0.1, "engine": ENGINE, "preset": "r0s4_rolls"}


def _log(deal: str, k: int) -> Dict[str, Any]:
    return {"format_id": "gen3ou", "seed": f"1,2,3,{k}", "names": ["A", "B"], "teams": [deal, "t"],
            "cmds": [f"CHOOSE p1 move m{k}"]}


def _capture(battle: str, tail: str) -> Dict[str, Any]:
    start = json.dumps({"formatid": "gen3ou", "seed": [1], "p1": {"name": "A", "team": "x"}, "p2": {"name": "B"}})
    body = "|t:|<N>\n|turn|11\n|move|p1a: X|Earthquake\n|turn|12\n|turn|20\n" + tail
    return {"tag": battle, "commands": [f"START {start}"], "chunks": [["p1", body], ["p2", body]]}


def _raw_state(root: Path) -> Path:
    """Two raw units (one a panel), 3 games each, 2 kept decisions a game at turns 12 and 20."""
    st = root / "anchor" / "u" / "pstar"
    units, sel = [], []
    for panel in PANELS:
        name = f"pstar__{panel}__200"
        units.append(KA.Unit(name=name, label="pstar", arm="pstar", seed=2601, teamset=panel, unit_seed=200,
                             model="/m.zip", model_sha256="ab", input_log="10-30").__dict__)
        cell = st / "cells" / name
        (cell / "capture").mkdir(parents=True)
        rows = []
        for g in range(3):
            battle = f"battle-gen3ou-{g + 1}"
            rows.append({"battle_tag": battle, "result": "loss", "turns": 30, "engine_stamp": ENGINE})
            (cell / "capture" / f"{battle}.json").write_text(json.dumps(_capture(battle, "|win|B")))
            deal = f"{panel[:4]}{g}"
            for k, turn in enumerate((12, 20)):
                n = turn + 1
                sel.append({"key": f"su_live_v1|{deal}|t{turn}|n{n}", "panel": panel, "cell": name, "unit_seed": 200,
                            "index": g, "deal": deal, "turn": turn, "n": n, "select_hash": select_hash(deal, turn, n),
                            "battle": battle, "pi_argmax": 2, "tokens": TOKENS, "frame_sha256": "c" * 64,
                            "log": _log(deal, k), "raw_state": str(st)})
        (cell / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    st.mkdir(parents=True, exist_ok=True)
    (st / "plan.json").write_text(json.dumps({"units": units}))
    return st, sel


def _scored(sel: List[Dict[str, Any]], dev_every: int = 1) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(sorted(sel, key=LB.order_key)):
        dev = i % dev_every == 0
        out.append({"key": r["key"], "panel": r["panel"], "deal": r["deal"], "pi_argmax": 2,
                    "search": {"searched": True, "deviation": dev, "played": 3 if dev else 2,
                               "config": {"name": "r0s4_rolls"}, "leak": {}},
                    "referee": {"T_pick": 0.6, "T_top": 0.5} if dev else None,
                    "speed_flip": {"ok": True, "speed_mons": 1, "speed_pinned": 0}})
    return out


def _play(bst: Path, y: Dict[str, str], *, null_result: str = "loss", g_fail: str = "", capture_tail: str = "",
          engine: str = ENGINE, drop: str = "") -> None:
    """Write each planned unit's cell as a COUNTED one-game unit: branches score ``y[kind]``, nulls replay the raw
    game; the keyword arguments plant one failure each."""
    for m in [json.loads(x) for x in (bst / "branch_plan.jsonl").read_text().splitlines()]:
        cell = bst / "cells" / m["unit"]
        (cell / "capture").mkdir(parents=True)
        with open(bst / "units.jsonl", "a") as f:
            f.write(json.dumps({"event": "end", "unit": m["unit"], "cpu_s": 28.0}) + "\n")
        if drop and m["unit"].endswith(drop):
            with open(bst / "units.jsonl", "a") as f:
                f.write(json.dumps({"event": "end", "unit": m["unit"]}) + "\n")
            continue
        null = m["kind"] == "null"
        res = null_result if null else y["branch"]
        (cell / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 1, "regime_verified_decisions": True, "peer_clean": True, "non_battle_endings": [],
            "forced_end_guard": "armed", "team_source_asymmetry": False}))
        tag = "battle-gen3ou-1"                  # a one-game unit's tag is its process's first, whatever its game
        (cell / "games.jsonl").write_text(json.dumps({"battle_tag": tag, "result": res, "turns": 30,
                                                      "engine_stamp": engine}) + "\n")
        ok = not (g_fail and m["unit"].endswith(g_fail))
        (cell / "branch.jsonl").write_text(json.dumps({"g_branch": "pass" if ok else "fail", "forced": m["action"],
                                                       "index": m["game"],
                                                       "mismatch": [] if ok else ["log_sha256"]}) + "\n")
        tail = "|win|B" if null else ("|win|A" if res == "win" else "|win|B")
        early = capture_tail and m["unit"].endswith(capture_tail)
        cap = _capture(tag, tail)
        if early:
            cap["chunks"][0][1] = cap["chunks"][0][1].replace("Earthquake", "Protect")
        (cell / "capture" / f"{tag}.json").write_text(json.dumps(cap))


@pytest.fixture()
def world(tmp_path):
    raw, sel = _raw_state(tmp_path / "raw")
    return tmp_path, raw, sel


def test_the_registered_order_stops_each_panel_at_its_quota(world) -> None:
    _tmp, _raw, sel = world
    quota = {"heldout72_hpfix": 2, "home72": 1}
    sc = _scored(sel, dev_every=2)                # every other state deviates, in the registered order
    todo = LB.to_score(sel, [], quota)
    assert [LB.order_key(r) for r in todo] == sorted(LB.order_key(r) for r in sel)
    # the prefix: H needs its 1st and 3rd own states (2 deviations), home72 its 1st
    by_panel = {p: [r for r in sc if r["panel"] == p] for p in PANELS}
    pref = [r for r in sc if r["key"] in {x["key"] for x in by_panel["home72"][:1] + by_panel["heldout72_hpfix"][:3]}]
    got = LB.quota_prefix(sel, pref, quota)
    assert got["heldout72_hpfix"]["deviations"] == 2 and got["home72"]["deviations"] == 1
    with pytest.raises(LB.LiveBranchError, match="past"):
        LB.quota_prefix(sel, pref + [by_panel["home72"][1]], quota)
    with pytest.raises(LB.LiveBranchError, match="gap"):
        LB.quota_prefix(sel, [r for r in pref if r["key"] != by_panel["heldout72_hpfix"][1]["key"]], quota)
    with pytest.raises(LB.LiveBranchError, match="quota"):
        LB.parse_quota("home72=0")


def test_the_plan_is_one_forced_branch_per_deviation_and_one_null_per_raw_unit(world) -> None:
    tmp, raw, sel = world
    quota = {"heldout72_hpfix": 6, "home72": 6}
    sc = _scored(sel)
    meta = LB.plan_branches(sel, sc, quota, [raw], tmp / "br")
    assert sum(m["kind"] == "branch" for m in meta) == 12 and sum(m["kind"] == "null" for m in meta) == 2
    units = KA.load_plan(tmp / "br")
    u = next(x for x in units if x.name == meta[0]["unit"])
    argv = KA.unit_argv(u, tmp / "out", 9555)
    assert argv[argv.index("--deal-game") + 1] == str(meta[0]["game"]) and "--our-force" in argv
    assert "--keep-input-logs" not in argv and argv[argv.index("--games") + 1] == "20"
    spec = json.loads(Path(u.force).read_text())
    assert spec["action"] == 3 and spec["pi_argmax"] == 2 and spec["kind"] == "branch"
    nul = [m for m in meta if m["kind"] == "null"]
    assert all(m["action"] == m["pi_argmax"] for m in nul)
    with pytest.raises(LB.LiveBranchError, match="written once"):
        LB.plan_branches(sel, sc, quota, [raw], tmp / "br")


def _planned(world, **play):
    tmp, raw, sel = world
    sc = _scored(sel)
    LB.plan_branches(sel, sc, {"heldout72_hpfix": 6, "home72": 6}, [raw], tmp / "br")
    _play(tmp / "br", **play)
    return LB.collect(tmp / "br"), sc


def test_a_planted_gain_reads_go_and_no_gain_reads_stop_on_the_boundary(world) -> None:
    rows, sc = _planned(world, y={"branch": "win"})
    res = LB.read(rows, sc, PARAMS)
    assert res["verdict"] == "GO" and res["primary"]["est"] == 1.0 and res["primary"]["n"] == 6
    assert res["reported"]["t_imp_u"]["est"] == pytest.approx(0.1)
    assert res["reported"]["live_minus_t_u"]["est"] == pytest.approx(0.9)
    assert res["reported"]["per_set"]["heldout72_hpfix"]["nulls_counted"] == 1


def test_no_gain_is_stop_flagged_on_rule_8s_boundary(tmp_path) -> None:
    raw, sel = _raw_state(tmp_path / "raw")
    rows, sc = _planned((tmp_path, raw, sel), y={"branch": "loss"})
    res = LB.read(rows, sc, PARAMS)
    assert res["verdict"] == "STOP" and res["boundary"] is True and res["primary"]["lo"] == 0.0


@pytest.mark.parametrize("plant,want", [
    ({"g_fail": "__g01__branch__t12n13"}, "G-BRANCH"),
    ({"null_result": "win"}, "null control"),
    ({"capture_tail": "__branch__t20n21"}, "capture"),
    ({"engine": "port=x;env=y"}, "engine"),
])
def test_every_tooth_voids_by_name(world, plant: Dict[str, Any], want: str) -> None:
    rows, sc = _planned(world, y={"branch": "win"}, **plant)
    res = LB.read(rows, sc, PARAMS)
    assert res["verdict"] == "VOID" and any(want in w for w in res["void"]), res["void"]


def test_a_dropped_branch_counts_against_the_cap_and_a_pending_one_refuses(world) -> None:
    rows, sc = _planned(world, y={"branch": "win"}, drop="hpfix__200__g01__branch__t12n13")
    res = LB.read(rows, sc, PARAMS)
    assert res["verdict"] == "VOID" and any("drops" in w for w in res["void"])
    res = LB.read(rows, sc, dict(PARAMS, max_drop_frac=0.5))
    assert res["verdict"] == "GO" and res["reported"]["per_set"]["heldout72_hpfix"]["dropped"] == 1
    pend = [dict(r, status="pending") if i == 0 else r for i, r in enumerate(rows)]
    with pytest.raises(LB.LiveBranchError, match="not yet played"):
        LB.read(pend, sc, PARAMS)


def test_a_branch_that_forced_another_action_than_the_scored_pick_voids(world) -> None:
    rows, sc = _planned(world, y={"branch": "win"})
    sc2 = [dict(r, search=dict(r["search"], played=6)) if r["search"]["deviation"] else r for r in sc]
    res = LB.read(rows, sc2, PARAMS)
    assert res["verdict"] == "VOID" and any("S′'s scored pick" in w for w in res["void"])
    sc3 = [dict(r, search=dict(r["search"], config={"name": "r0s4"})) for r in sc]
    assert any("preset" in w for w in LB.read(rows, sc3, PARAMS)["void"])


def test_the_scorer_refuses_another_engine_before_any_state(world, monkeypatch, tmp_path) -> None:
    _tmp, _raw, sel = world
    (tmp_path / "sel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    monkeypatch.setattr("main.anchors.server.engine_stamp", lambda: "port=x/1;env=y/2")
    rc = LB.main(["score", "--selected", str(tmp_path / "sel.jsonl"), "--model", "/m.zip", "--search",
                  "r0s4_rolls@10", "--quota", "home72=1", "--engine", ENGINE, "--out", str(tmp_path / "sc.jsonl")])
    assert rc == 2 and not (tmp_path / "sc.jsonl").exists() and not (tmp_path / "engine.txt").exists()
