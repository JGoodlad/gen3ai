"""``main.human_data.x59s_ops`` (X59-S's read operations, PREREG Amendment 4) on constructed inputs: the floor unit's
argv differs from the Kakuna unit's ONLY in the our-side, VOID 3 and the learned-nothing guard, m5's flag, the
identity compare, and the in-house pairing rates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.human_data import x59s_ops as O


def test_the_floor_argv_swaps_only_the_our_side():
    from main.x35 import kakuna as KA

    u = O.floor_units()[0]
    base = KA.unit_argv(u, Path("/tmp/out"), 9555, py="PY")
    got = O.bot_argv(base, "random")
    i = base.index("--model")
    assert got[:i] == base[:i] and got[i:i + 2] == ["--our-side", "bot:random"] and got[i + 2:] == base[i + 2:]
    assert len(O.floor_units()) == len(O.FLOOR_BOTS) * len(O.PANELS) * len(O.FLOOR_SEEDS) == 24


def _side(seed, init="i", steps=O.STEPS, ok=True):
    return {"seed": seed, "init_sha256": init, "steps": steps, "learned_guard": {"pass": ok}}


def test_void3_names_each_failure():
    good = {"s_h_5901": _side(5901), "s_s_5901": _side(5901)}
    assert O.void3(good) == []
    assert any("init_sha256" in r for r in O.void3({**good, "s_p_5901": _side(5901, init="j")}))
    assert any("steps" in r for r in O.void3({**good, "s_p_5901": _side(5901, steps=7040)}))
    assert any("learned" in r for r in O.void3({**good, "s_p_5901": _side(5901, ok=False)}))


def test_m5_reads_the_last_epoch():
    h = [[i, 2.0 - 0.1 * i] for i in range(9)]
    assert not O.last_epoch_rose(h)
    assert O.last_epoch_rose(h[:-1] + [[8, 1.65]])         # 1.65 > the value one epoch (4 evals) before, 1.6


def test_identity_compares_game_for_game(tmp_path):
    rows = tmp_path / "x51.jsonl"
    games = [{"read_cell": "pstar_home72", "unit_seed": 0, "half": h, "index": i, "our_team": f"t{i}.txt",
              "result": "win" if i % 2 else "loss", "turns": 30 + i} for h in ("a", "b") for i in range(1, 11)]
    rows.write_text("\n".join(json.dumps(g) for g in games))
    cell = tmp_path / "cells" / "pstar__home72__00"
    cell.mkdir(parents=True)
    (cell / "games.jsonl").write_text("\n".join(json.dumps(g) for g in games))
    assert O.identity(tmp_path, "pstar__home72__00", rows, "pstar_home72", 0)["pass"]
    games[3]["turns"] += 1
    (cell / "games.jsonl").write_text("\n".join(json.dumps(g) for g in games))
    res = O.identity(tmp_path, "pstar__home72__00", rows, "pstar_home72", 0)
    assert not res["pass"] and len(res["differing"]) == 1


def test_inhouse_rates_are_the_equal_weight_panel_mean(tmp_path):
    side = {"students": {O.label(a, k): {"zip_path": f"/z/{a}_{k}.zip"} for a in O.ARMS for k in O.SEEDS}}
    for c in O.inhouse_cells(side):
        sc = {"heldout72_hpfix": 0.5, "forum72": 0.6, "p72": 0.7}[c["panel"]]
        (tmp_path / f"{c['name']}.log").write_text("noise\n" + json.dumps({"pairs": O.PAIRS, "score": sc}) + "\n")
    out = O.inhouse_read(tmp_path, side)
    assert out["short"] == [] and out["rates"]["delta_data"] == pytest.approx([0.6] * 5)
    (tmp_path / "delta_labels_5903_p72.log").write_text(json.dumps({"pairs": 100, "score": 0.9}))
    out = O.inhouse_read(tmp_path, side)
    assert out["short"] == ["delta_labels_5903_p72"] and out["rates"]["delta_labels"][2] is None


def test_ledger_engines_reads_the_gzipped_count_rows(tmp_path):
    import gzip

    d = tmp_path / "rows" / "h2h"
    d.mkdir(parents=True)
    row = {"compute": {"core_stamp": "src=95b45f104119bdb2;nfiles=129"}}
    (d / "ledger.x.jsonl.gz").write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
    (tmp_path / "requests.jsonl").write_text(json.dumps({"event": "x"}) + "\n")       # no compute: not a row
    got = O.ledger_engines(tmp_path)
    assert len(got) == 1 and got[0].startswith("env=")
