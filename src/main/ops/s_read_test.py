"""``scripts/ops/s_read.py`` — the paired speed benchmark's cost read — on synthetic blocks.

THE DEFECT THIS PINS (2026-10-08, the static-token screen): a block pair in which one side had ZERO quiet cycles
(every window contended) made the reader divide ``None`` by ``None`` — a TypeError that lost every pair that
DID read. An insufficient pair is now a REPORTED result (``status: "insufficient quiet cycles"``), the pooled
``s`` is ``null`` with its reason when an arm has no data, and the exit status says which.

Pure unit tests over ``classify_cycles`` / ``summarize`` plus one end-to-end ``main`` on scratch files (the
TensorBoard read is the one stubbed seam: ``tb_walls``, which ``read_block`` calls).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from utils.paths import repo_path

_SCRIPT = repo_path("scripts", "ops", "s_read.py")
_spec = importlib.util.spec_from_file_location("s_read", _SCRIPT)
assert _spec is not None and _spec.loader is not None
S = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S
_spec.loader.exec_module(S)

N_RECORDS = 14                       # cycles k = 1..13; updates 10 and 11 (the canary pair) are excluded -> 11 candidates


def _walls(start=1000.0, cycle_s=60.0):
    return [(i, start + cycle_s * i, 1000.0 * cycle_s) for i in range(N_RECORDS)]


def _cpu(recs, factor):
    rows = []
    for k in range(1, len(recs)):
        w0, w1 = recs[k - 1][1], recs[k][1]
        t = w0 + 1.0
        while t <= w1:
            rows.append({"t": t, "dt": 1.0, "factor": factor})
            t += 1.0
    return rows


def _block(tag, cycle_s, factor=1.0):
    recs = _walls(cycle_s=cycle_s)
    return S.block_summary(tag, recs, S.classify_cycles(recs, [], _cpu(recs, factor), float("inf")), N_RECORDS - 1)


def _arm(b):
    return "static" if "_static_" in b["tag"] else "legacy"


def test_a_quiet_block_keeps_every_cycle_but_the_canary_pair():
    b = _block("sB_legacy_1", 60.0)
    assert b["kept_n"] == 11 and b["median"] == 60.0
    assert {c["update"] for c in b["cycles"] if c["excluded"]} == {10, 11}


def test_a_contended_block_has_no_median_and_that_is_not_an_error():
    b = _block("sB_static_1", 60.0, factor=1.3)
    assert b["kept_n"] == 0 and b["median"] is None
    assert all("contended" in c["excluded"] or "canary" in c["excluded"] for c in b["cycles"])


def test_an_insufficient_pair_is_reported_and_the_other_pairs_still_read():
    """THE CRASH: pair 1's static side has 0 quiet cycles. It used to raise TypeError for the whole read."""
    blocks = [_block("sB_legacy_1", 60.0), _block("sB_static_1", 66.0),          # pair 0: s = +10 %
              _block("sB_legacy_2", 60.0), _block("sB_static_2", 99.0, factor=1.4)]   # pair 1: static all contended
    r = S.summarize(blocks, _arm, "legacy", "static")
    assert [p["status"] for p in r["pairs"]] == ["ok", S.INSUFFICIENT]
    assert r["pairs"][0]["s"] == pytest.approx(0.10) and r["pairs"][1]["s"] is None
    assert r["pairs"][1]["kept_n"] == [11, 0]
    assert r["s_per_block_pair"] == [pytest.approx(0.10)] and r["n_pairs_read"] == 1 and r["n_pairs_insufficient"] == 1
    assert r["s"] == pytest.approx(0.10)                        # the pooled arms both have data
    assert r["s_status"] == "ok"
    json.dumps(r)                                                # JSON-clean (no NaN / None arithmetic left)


def test_an_arm_with_no_quiet_cycles_makes_s_null_with_its_reason():
    blocks = [_block("sB_legacy_1", 60.0), _block("sB_static_1", 66.0, factor=2.0)]
    r = S.summarize(blocks, _arm, "legacy", "static")
    assert r["s"] is None and r["matched_wall_time_checkpoint"] is None and r["matched_wall_time_steps"] is None
    assert S.INSUFFICIENT in r["s_status"] and "static" in r["s_status"]
    assert r["s_block_range"] is None and r["n_pairs_read"] == 0


def test_min_cycles_raises_the_bar_for_a_pair_and_an_arm():
    blocks = [_block("sB_legacy_1", 60.0), _block("sB_static_1", 66.0)]
    assert S.summarize(blocks, _arm, "legacy", "static", min_cycles=11)["s"] == pytest.approx(0.10)
    r = S.summarize(blocks, _arm, "legacy", "static", min_cycles=12)
    assert r["s"] is None and r["pairs"][0]["status"] == S.INSUFFICIENT


def test_the_matched_wall_time_checkpoint_is_floored_to_a_million():
    blocks = [_block("sB_legacy_1", 60.0), _block("sB_static_1", 66.0)]
    r = S.summarize(blocks, _arm, "legacy", "static")
    assert r["matched_wall_time_steps"] == pytest.approx(15_000_000 / 1.1)
    assert r["matched_wall_time_checkpoint"] == 13_000_000


def test_the_stop_and_eval_windows_are_excluded():
    recs = _walls()
    cpu = _cpu(recs, 1.0)
    evals = [(recs[2][1] - 5, recs[2][1] - 1)]                  # inside cycle k=2
    rows = S.classify_cycles(recs, evals, cpu, t_stop=recs[13][1])    # the last record is the abort dump
    why = {c["k"]: c["excluded"] for c in rows}
    assert "eval" in why[2] and "stop" in why[13] and why[1] is None


def test_main_end_to_end_reports_an_insufficient_pair_with_exit_3(tmp_path, monkeypatch, capsys):
    """Scratch files for the four sidecars of each block; ``tb_walls`` (the TensorBoard read) is the stubbed seam."""
    plan = {"sB_legacy_1": (60.0, 1.0), "sB_static_1": (66.0, 1.0), "sB_legacy_2": (60.0, 1.0), "sB_static_2": (70.0, 1.6)}
    walls = {}
    for tag, (cyc, fac) in plan.items():
        recs = _walls(cycle_s=cyc)
        walls[tag] = recs
        (tmp_path / f"{tag}.phases.jsonl").write_text(json.dumps({"phase": "update", "t0": 1.0, "t1": 2.0}) + "\n")
        (tmp_path / f"{tag}.cpu.jsonl").write_text(
            "\n".join(["{}"] + [json.dumps(r) for r in _cpu(recs, fac)]) + "\n")
        (tmp_path / f"{tag}.drive.log").write_text("no stop here\n")
        tb = tmp_path / "models" / tag / "tb"
        tb.mkdir(parents=True)
        (tb / "events.out.tfevents.0").write_text("")
    monkeypatch.setattr(S, "tb_walls", lambda tb_dir, tag="train/train_ms": walls[Path(tb_dir).parent.name])
    out = tmp_path / "s.json"
    rc = S.main(["--dir", str(tmp_path), "--blocks", *plan, "--out", str(out)])
    assert rc == 0                                              # the pooled arms both have data
    res = json.loads(out.read_text())
    assert [p["status"] for p in res["pairs"]] == ["ok", S.INSUFFICIENT]
    # …and when a whole arm is contended the exit status says so, and the file is still written
    plan2 = {k: v for k, v in plan.items() if k in ("sB_legacy_1", "sB_static_2")}
    out2 = tmp_path / "s2.json"
    assert S.main(["--dir", str(tmp_path), "--blocks", *plan2, "--out", str(out2)]) == 3
    assert json.loads(out2.read_text())["s"] is None
    assert "insufficient" in capsys.readouterr().out
