"""Gate ④'s subset, mapping and readout math (unit; no core)."""

from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum import truth as T
from main.policy_spectrum.bank_test import BANK_V1


@pytest.fixture(scope="module")
def v1():
    return B.load_bank(BANK_V1)


def test_subset_is_fixed_stratified_and_covers_categories(v1):
    a = T.select_subset(v1)
    assert a == T.select_subset(v1)
    dd = {d["id"]: d for d in v1.decisions}
    free = [i for i in a if dd[i]["kind"] == "free"]
    assert len(free) >= 200 and sum(dd[i]["kind"] == "forced_switch" for i in a) == 20
    for c in ("hazard", "setup", "recovery", "status"):
        assert sum(c in dd[i]["cats"].values() for i in free) >= 30
    per_battle = {}
    for i in a:
        per_battle[dd[i]["battle"]] = per_battle.get(dd[i]["battle"], 0) + 1
    assert max(per_battle.values()) <= 2
    assert {dd[i]["opp_class"] for i in a} == {"bot", "pool_snapshot", "exploiter"}


def test_cmd_index_is_the_sides_nth_choose_and_matches_the_played_token(v1):
    b = v1.battles[0]
    ds = [d for d in v1.decisions if d["battle"] == b.battle_id]
    for d in ds:
        at = T.cmd_index(b, d["side"], d["n"])
        assert b.commands[at][0] == d["side"] and b.commands[at][1] == d["played"]
    with pytest.raises(IndexError):
        T.cmd_index(b, b.banked_side, 10_000)


def test_to_log_equals_lane_i_record_to_log(v1):
    from utils.bridge.reconstruction import ReconstructionRecord
    from utils.rust_env.successors import record_to_log

    b = v1.battles[3]
    import json
    rec = ReconstructionRecord(
        format_id=b.format_id, prng_seed=b.seed,
        input_log=(f">start {json.dumps({'formatid': b.format_id, 'seed': b.seed})}",
                   f">player p1 {json.dumps(b.p1)}", f">player p2 {json.dumps(b.p2)}"),
        commands=tuple(tuple(c) for c in b.commands))
    assert T.to_log(b) == record_to_log(rec)


def test_seeds_are_shared_format_and_per_turn():
    s = T.turn_seeds("x#p1#3", 4)
    assert s == T.turn_seeds("x#p1#3", 4) and len(set(s)) == 4
    assert all(x.startswith("sodium,") and len(x) == 7 + 32 for x in s)
    assert s != T.turn_seeds("x#p1#4", 4)


def _row(did, outcomes):
    return {"id": did, "ok": True, "outcomes": {str(a): v for a, v in outcomes.items()}}


def test_turn_truth_classes():
    # action 6 wins everywhere; 7 is CRN-identical to 6; 8 is clearly worse; 9 is noisy-close
    r = _row("x", {6: [1, 1, 1, 1], 7: [1, 1, 1, 1], 8: [-1, -1, -1, -1], 9: [1, 1, 1, -1]})
    t = T.turn_truth(r, eps=0.1)
    assert t["actions"] == [6, 7, 8, 9] and t["vstar"] == 1.0
    assert t["near"].tolist() == [True, True, False, False]
    assert t["strict_near"].tolist() == [True, True, False, False]
    assert t["dominated"].tolist() == [False, False, True, False]     # 9: gap 0.5, SE 0.5 -> uncertain


def test_readout_starvation_regret_and_guess(v1):
    # two synthetic turns on real bank ids (their categories are read from the bank)
    ids = [d["id"] for d in v1.decisions if d["kind"] == "free" and d["n_legal"] >= 3][:2]
    dd = {d["id"]: d for d in v1.decisions}
    rows, probs = [], np.zeros((len(v1.decisions), 11))
    pos = {d["id"]: i for i, d in enumerate(v1.decisions)}
    for k, did in enumerate(ids):
        legal = sorted(int(a) for a in dd[did]["tokens"])
        a0, a1, a2 = legal[:3]
        out = {a: [-1, -1] for a in legal}
        out[a0] = [1, 1]
        out[a1] = [1, 1]
        rows.append(_row(did, out))
        p = np.zeros(11)
        if k == 0:
            p[a0], p[a1], p[a2] = 0.995, 0.004, 0.001      # a1 near-best but starved
        else:
            p[a0], p[a1], p[a2] = 0.5, 0.3, 0.2
        probs[pos[did]] = p
    rows.append({"id": ids[0], "ok": False, "error": "refused"})
    r = T.readout(v1, rows, probs)
    assert r["turns"] == 2 and r["refused"] == 1
    assert r["starved"]["mean"] == pytest.approx(0.5)
    assert r["near"]["mean"] == pytest.approx((0.999 + 0.8) / 2)
    assert r["dom"]["mean"] == pytest.approx((0.001 + 0.2) / 2)
    assert r["regret"]["mean"] == pytest.approx(((1 - (0.999 - 0.001)) + (1 - (0.8 - 0.2))) / 2)
    flat = next(d["id"] for d in v1.decisions if d["kind"] == "free" and d["id"] not in ids)
    rows.append(_row(flat, {int(a): [1, 1] for a in dd[flat]["tokens"]}))     # every action wins
    assert T.readout(v1, rows, probs)["turns"] == 3
    dec = T.readout(v1, rows, probs, decisive_only=True)
    assert dec["turns"] == 2 and dec["starved"]["mean"] == pytest.approx(0.5)
    s = T.value_summary(rows[:2])
    assert s["turns"] == 2 and s["mean_near_best"] == 2.0 and s["vstar_eq_+1"] == 2


def test_micro_batcher_merges_concurrent_calls_and_returns_each_callers_slice():
    import threading
    import time as _t

    seen = []

    def fn(rows, masks):
        seen.append(len(rows))
        _t.sleep(0.05)
        return rows[:, :11] * 2

    mb = T.MicroBatcher(fn)
    outs = {}

    def call(k):
        r = np.full((k, 20), float(k), dtype=np.float32)
        outs[k] = mb(r, np.ones((k, 11), np.uint8))

    ths = [threading.Thread(target=call, args=(k,)) for k in range(1, 7)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for k in range(1, 7):
        assert outs[k].shape == (k, 11) and np.all(outs[k] == 2 * k)
    assert mb.calls == 6 and mb.forwards < 6 and sum(seen) == 21


def test_first_seeds_nests():
    r = {"id": "x", "ok": True, "seeds": ["a", "b", "c", "d"], "outcomes": {"6": [1, -1, 1, 0]},
         "ends": {"6": [{}, {}, {}, {}]}}
    q = T.first_seeds(r, 2)
    assert q["seeds"] == ["a", "b"] and q["outcomes"] == {"6": [1, -1]}
    assert T.turn_seeds("id", 64)[:16] == T.turn_seeds("id", 16)
    with pytest.raises(ValueError):
        T.first_seeds(r, 8)


def test_compact_round_trips_and_regenerates_the_seeds(tmp_path):
    import json

    r = {"schema": T.TRUTH_SCHEMA, "id": "b#p1#3", "ok": True, "continuation": "c", "at": 5, "side": "p1",
         "stall": "production", "max_turns": 999, "seeds": T.turn_seeds("b#p1#3", 3),
         "outcomes": {"6": [1.0, -1.0, 0.0], "7": [1.0, 1.0, 1.0]}, "ends": {"6": [{}, {}, {}]}}
    c = T.compact(r)
    assert c["outcomes"] == {"6": "+-0", "7": "+++"} and "seeds" not in c
    e = T.expand(json.loads(json.dumps(c)))
    assert e["seeds"] == r["seeds"] and e["outcomes"] == r["outcomes"]
    src = tmp_path / "rows.jsonl"
    src.write_text(json.dumps(r) + "\n")
    assert T.write_compact(src, tmp_path / "rows.c.jsonl.gz") == 1
    back = T.load_rows(tmp_path / "rows.c.jsonl.gz")[0]
    assert back["outcomes"] == r["outcomes"] and back["seeds"] == r["seeds"]


class _NoCore:
    def close(self) -> None:
        pass


def _run_with_lock(monkeypatch, tmp_path, lock):
    monkeypatch.setattr(T, "branch_turn", lambda bank, did, *a, **k: {"id": did, "ok": True})
    return T.run(None, [f"d{i}" for i in range(5)], None, "fake", tmp_path / "rows.jsonl", s=2,
                 log=lambda m: None, core_factory=_NoCore, chunk=2, lock_path=lock)


def test_the_per_chunk_lock_is_reentrant_under_an_outer_hold(monkeypatch, tmp_path):
    """``gpu_lock.sh … truth run --lock <the same file>`` used to be a self-deadlock: the per-chunk
    ``flock`` opened the file again and waited on the outer holder. Through ``utils.gpu_lock`` the
    per-chunk take is a verified no-op. (Run in a thread so a revert FAILS instead of hanging.)"""
    import threading

    from utils.gpu_lock import HELD_ENV, gpu_lock

    monkeypatch.delenv(HELD_ENV, raising=False)
    lock = tmp_path / "gpu.lock"
    out: list = []
    with gpu_lock(lock, timeout_s=5):
        t = threading.Thread(target=lambda: out.append(_run_with_lock(monkeypatch, tmp_path, lock)), daemon=True)
        t.start()
        t.join(timeout=30)
    assert out == [5], "truth.run blocked on a lock this process already holds"


def test_the_per_chunk_lock_is_taken_and_released(monkeypatch, tmp_path):
    import fcntl
    import os

    from utils.gpu_lock import HELD_ENV

    monkeypatch.delenv(HELD_ENV, raising=False)
    lock = tmp_path / "gpu.lock"
    assert _run_with_lock(monkeypatch, tmp_path, lock) == 5
    assert lock.exists()
    fd = os.open(str(lock), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)      # released after the last chunk
    finally:
        os.close(fd)
