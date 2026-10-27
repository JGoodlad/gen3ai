"""Team identity, team sets, the reference draw and the matchup plan of the team-strength meter (pure, no Node)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.team_strength import panel as PN
from main.team_strength import plan as PLN
from main.team_strength import register as RG
from utils.paths import repo_path

HOME72 = repo_path("designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72")


def _texts(n: int) -> list:
    return [f.read_text() for f in sorted(HOME72.iterdir())[:n]]


def test_team_id_ignores_layout_and_nicknames_but_not_the_build():
    t = _texts(1)[0]
    blocks = t.strip().split("\n\n")
    first = blocks[0].split("\n")
    nick = "\n".join([f"Bob ({first[0].split(' @ ')[0]}) @ {first[0].split(' @ ')[1]}"] + first[1:])
    relaid = "\n\n\n".join([nick] + blocks[1:]) + "\n\n\n"
    reordered = "\n\n".join(blocks[::-1])
    assert PN.team_id(t) == PN.team_id(relaid) == PN.team_id(reordered)
    changed = t.replace("252 SpA", "248 SpA", 1)
    assert changed != t and PN.team_id(changed) != PN.team_id(t)
    with pytest.raises(PN.PanelError):
        PN.team_id("\n\n")


def _set_dir(tmp: Path, name: str, texts: list) -> Path:
    d = tmp / name
    d.mkdir()
    for i, t in enumerate(texts):
        (d / f"{name}_{i:02d}.txt").write_text(t)
    return d


def test_a_set_reads_byte_for_byte_and_refuses_a_moved_file(tmp_path):
    d = _set_dir(tmp_path, "a", _texts(5))
    ts = PN.read_set("a", d)
    blk = ts.block()
    assert blk["n_teams"] == 5 and len(blk["files"]) == 5
    PN.check_set(PN.read_set("a", d), blk)
    f = sorted(d.iterdir())[2]
    f.write_text(f.read_text() + "\n")             # a trailing newline: other BYTES, same team
    with pytest.raises(PN.PanelError):
        PN.check_set(PN.read_set("a", d), blk)


def test_the_reference_draw_is_deterministic_distinct_and_meets_each_quota(tmp_path):
    texts = _texts(20)
    sets = {"a": PN.read_set("a", _set_dir(tmp_path, "a", texts[:12])),
            "b": PN.read_set("b", _set_dir(tmp_path, "b", texts[6:20]))}   # 6 teams shared with a
    q = {"a": 5, "b": 5}
    one, two = PN.draw_reference(sets, q, 3800), PN.draw_reference(sets, q, 3800)
    assert [m.team_id for m in one] == [m.team_id for m in two]
    assert len({m.team_id for m in one}) == 10
    assert [m.set for m in one] == ["a"] * 5 + ["b"] * 5
    assert PN.panel_digest(one) == PN.panel_digest(two)
    assert [m.team_id for m in PN.draw_reference(sets, q, 3801)] != [m.team_id for m in one]
    with pytest.raises(PN.PanelError):
        PN.draw_reference(sets, {"a": 13}, 1)
    # a team in two sets is ONE identity in the membership map
    mem = PN.membership(sets)
    assert sum(1 for v in mem.values() if sorted(v) == ["a", "b"]) == 6


def test_the_plan_never_pairs_a_team_with_itself_and_is_deterministic():
    subjects = [f"t{i}" for i in range(30)] + ["r0", "r1"]
    refs = [f"r{i}" for i in range(6)]
    pairs = PLN.subject_pairs(subjects, refs, 5, 3801)
    assert all(p.subject != p.ref for p in pairs)
    by = {}
    for p in pairs:
        by.setdefault(p.subject, []).append(p.ref)
    assert all(len(v) == 5 and len(set(v)) == 5 for v in by.values())      # distinct refs while pairs < refs
    u1 = PLN.plan_units(subjects, refs, pairs_per_team=5, unit_pairs=40, seed=3801, key="k")
    u2 = PLN.plan_units(subjects, refs, pairs_per_team=5, unit_pairs=40, seed=3801, key="k")
    assert u1 == u2 and sum(len(u.pairs) for u in u1) == 32 * 5
    assert len({u.seed for u in u1}) == len(u1)
    assert PLN.plan_units(subjects, refs, pairs_per_team=5, unit_pairs=40, seed=3801, key="k2")[0].seed != u1[0].seed
    ks = PLN.kakuna_units(["a", "b"], 3, 7)
    assert [k[0] for k in ks[:2]] == ["kak_a_000", "kak_b_000"]
    assert all(0 <= s < 2 ** 31 for _, _, s in ks) and len({s for _, _, s in ks}) == 6


def test_the_registration_is_frozen_by_its_digest(tmp_path):
    texts = _texts(16)
    da, db = _set_dir(tmp_path, "a", texts[:8]), _set_dir(tmp_path, "b", texts[8:])
    spec = {"name": "t", "sets": {"a": str(da), "b": str(db)}, "reference": {"quota": {"a": 2, "b": 2}, "seed": 1},
            "pilots": {"x26": {"model": "m.zip", "sha256": "0" * 64, "pairs_per_team": 2, "unit_pairs": 8, "seed": 2},
                       "kakuna": {"subset_size": 4, "subset_seed": 3, "subset_sets": {"a": "mm_a", "b": "mm_b"},
                                  "ref_set": "mm_ref", "units_per_set": 1, "seed": 4, "port": 9563}},
            "fit": {"prior_sd": 2.0, "B": 10, "seed": 5}}
    reg = RG.build(spec, check_packing=False)
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(reg))
    assert RG.load(p)["digest"] == reg["digest"]
    assert all(len(v) == 4 for v in reg["pilots"]["kakuna"]["subsets"].values())
    reg["reference"]["seed"] = 2
    p.write_text(json.dumps(reg))
    with pytest.raises(PN.PanelError):
        RG.load(p)


def test_pilot_b_games_cluster_by_matchup_and_self_games_are_dropped(tmp_path, monkeypatch):
    """The anchors' two role halves replay one team sequence, so a unit's games with the same two teams are ONE
    cluster (a seat-swapped pair); a team drawn on both sides carries no information and is dropped, counted."""
    from types import SimpleNamespace

    from main.anchors import results as AR
    from main.team_strength import cli
    from main.team_strength import kakuna_lane as KL

    reg = {"pilots": {"kakuna": {"subset_sets": {"s": "mm_s"}, "subsets": {"s": ["a1", "a2"]}, "units_per_set": 1,
                                 "seed": 1}},
           "reference": {"members": [{"team_id": "b1"}, {"team_id": "a2"}]}}
    unit = PLN.kakuna_units(["s"], 1, 1)[0][0]
    (KL.unit_dir(tmp_path, unit)).mkdir(parents=True)
    (KL.unit_dir(tmp_path, unit) / "summary.json").write_text("{}")
    rows = [{"tag": "t1", "a": "a1", "b": "b1", "s": 1.0}, {"tag": "t2", "a": "a1", "b": "b1", "s": 0.0},
            {"tag": "t3", "a": "a2", "b": "a2", "s": 1.0}, {"tag": "t4", "a": "a2", "b": "b1", "s": 1.0}]
    monkeypatch.setattr(cli, "kakuna_digests", lambda _reg: {"s": "x", "ref": "y"})
    monkeypatch.setattr(PN, "load_registered", lambda _reg: {})
    monkeypatch.setattr(KL, "parse_unit", lambda *a, **k: (rows, []))
    monkeypatch.setattr(AR, "unit_counts", lambda *a, **k: SimpleNamespace(reasons=[]))
    kk = cli.load_kakuna(reg, tmp_path, tmp_path)
    assert kk["self_games"] == 1 and len(kk["games"]) == 3
    assert len({g.cluster for g in kk["games"]}) == 2                       # (a1, b1) twice = one cluster
    assert len({g.cluster for g in cli.load_kakuna(reg, tmp_path, tmp_path, cluster="game")["games"]}) == 3
    with pytest.raises(ValueError):
        cli.load_kakuna(reg, tmp_path, tmp_path, cluster="pair")


def test_a_unit_claims_its_port_takes_the_short_lane_and_retries_long_once_on_an_overrun(tmp_path, monkeypatch):
    """Pilot B's unit claims a 95xx port in the Kakuna lanes' shared lock dir and holds it for the child's life; it
    plays on the heavy slots' short lane, and an overrun (exit 88) is played ONCE more as a long job, after which the
    unit is never short again."""
    from main.team_strength import kakuna_lane as KL

    tree = tmp_path / "tree"
    argv_short = KL.argv_for("py", tree, tmp_path / "o", 7, 9550)
    argv_long = KL.argv_for("py", tree, tmp_path / "o", 7, 9550, short=False)
    assert argv_short[argv_short.index("--max-s") + 1] == str(KL.UNIT_MAX_S)
    assert "--max-s" not in argv_long and argv_long[argv_long.index("timeout") + 1] == str(KL.UNIT_LONG_S)
    assert argv_short[argv_short.index("--port") + 1] == "9550"

    closed = []
    calls = []

    class _Claim:
        def close(self):
            closed.append(True)

    class _Proc:
        pid = 1

        def __init__(self, argv, **_kw):
            calls.append(argv)

        def wait(self):
            return KL.SHORT_LANE_OVERRUN if "--max-s" in calls[-1] else 0

    monkeypatch.setattr(KL, "claim_port", lambda: (9551, _Claim()))
    monkeypatch.setattr(KL.subprocess, "Popen", _Proc)
    monkeypatch.setattr(KL, "env_for", lambda *_a: {})
    rc = KL.run_unit(py="py", tree=tree, state=tmp_path, unit="kak_s_000", config=tmp_path / "c.json", seed=7)
    assert rc == 0 and len(calls) == 2 and "--max-s" in calls[0] and "--max-s" not in calls[1]
    assert len(closed) == 2 and all(c[c.index("--port") + 1] == "9551" for c in calls)
    calls.clear()
    KL.run_unit(py="py", tree=tree, state=tmp_path, unit="kak_s_000", config=tmp_path / "c.json", seed=7)
    assert len(calls) == 1 and "--max-s" not in calls[0]           # it overran before: long from the start


def test_a_forced_end_in_any_capture_of_a_pair_cell_is_a_rederived_hit(tmp_path):
    """Neither side of a pair cell is ours, so EVERY FORCELOSE is a hit, found by scanning the capture files even when
    no games.jsonl row names them; a clean unit with its peer logs is clean."""
    from main.team_strength import kakuna_lane as KL

    u = tmp_path / "kak_s_000"
    (u / "capture").mkdir(parents=True)
    (u / "ours_challenge").mkdir()
    (u / "ours_challenge" / "peer_ours.log").write_text("ok\n")
    (u / "games.jsonl").write_text(json.dumps({"index": 0, "half": "ours_challenge", "battle_tag": "battle-1",
                                               "result": "win"}) + "\n")
    (u / "capture" / "battle-1.json").write_text(json.dumps({"commands": ["START {}"], "chunks": []}))
    summ = {"cell": {"our_side": "metamon:Kakuna", "forfeit_turn_limit": 250}}
    assert KL.rederived_guards(u, summ)["status"] == "clean"
    (u / "capture" / "battle-gen3ou-9.json").write_text(json.dumps({"commands": ["START {}", "FORCELOSE p2"]}))
    got = KL.rederived_guards(u, summ)
    assert got["status"] == "hit" and [h["tag"] for h in got["hits"]] == ["battle-gen3ou-9"]
