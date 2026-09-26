"""Unit tests for the POLICY DRIFT meter's pure part and its resumable driver — no model, no bridge.

Each guards one rule that fails on revert: the margin bucketing (incl. the edge and the
single-legal exclusion), the dex action-class mapping (self-KO before attack), the cycling rule, the
10M-back reference pick, and resume-from-rows (a recorded step is never re-forwarded; cached probs
are reused; a torn last line is redone; a still-settling zip waits)."""
import json
import os
import time
from pathlib import Path

import numpy as np

from agents.training import policy_drift as pd

A = 11


def _p(rows):
    p = np.zeros((len(rows), A), np.float32)
    for i, r in enumerate(rows):
        for a, v in r.items():
            p[i, a] = v
    return p


def _mask_of(p):
    return (p > 0).astype(np.float32)


# ── flip rate by margin ───────────────────────────────────────────────────────────────────────
def test_flip_rate_buckets_by_older_policys_margin():
    old = _p([{6: 0.52, 7: 0.48},            # margin 0.04 → <0.1   (flips)
              {6: 0.54, 7: 0.46},            # margin 0.08 → <0.1   (stays)
              {6: 0.6, 7: 0.4},              # margin 0.2  → 0.1-0.3 (stays)
              {6: 0.8, 7: 0.2},              # margin 0.6  → >0.3   (flips)
              {6: 0.55, 7: 0.45}])           # margin 0.1 exactly → the HIGHER bucket (0.1-0.3)
    new = _p([{6: 0.4, 7: 0.6},
              {6: 0.6, 7: 0.4},
              {6: 0.7, 7: 0.3},
              {6: 0.3, 7: 0.7},
              {6: 0.4, 7: 0.6}])             # flips
    mask = _mask_of(old)
    r = pd.flip_by_margin(old, new, mask)
    b = r["buckets"]
    assert (b["<0.1"]["n"], b["<0.1"]["flips"]) == (2, 1)
    assert (b["0.1-0.3"]["n"], b["0.1-0.3"]["flips"]) == (2, 1)
    assert (b[">0.3"]["n"], b[">0.3"]["flips"]) == (1, 1)
    assert (r["n"], r["flips"]) == (5, 3)


def test_flip_rate_uses_the_OLDER_margin_not_the_newer():
    old = _p([{6: 0.9, 7: 0.1}])             # confident old → >0.3 bucket
    new = _p([{6: 0.49, 7: 0.51}])           # the new one is uncertain; must not decide the bucket
    r = pd.flip_by_margin(old, new, _mask_of(old))
    assert r["buckets"][">0.3"]["flips"] == 1 and r["buckets"]["<0.1"]["n"] == 0


def test_single_legal_states_cannot_flip_and_are_excluded():
    old = _p([{0: 1.0}, {6: 0.52, 7: 0.48}])
    new = _p([{0: 1.0}, {6: 0.48, 7: 0.52}])
    r = pd.flip_by_margin(old, new, _mask_of(old))
    assert r["n"] == 1 and r["buckets"][">0.3"]["n"] == 0


def test_greedy_ignores_illegal_mass():
    p = _p([{6: 0.3, 7: 0.2, 8: 0.5}])
    mask = np.zeros_like(p); mask[0, 6] = mask[0, 7] = 1
    assert pd.greedy_actions(p, mask)[0] == 6
    assert np.isclose(pd.top2_margin(p, mask)[0], 0.1)


# ── action classes from the dex ───────────────────────────────────────────────────────────────
def _num(mid):
    from agents.gen3_data import moves
    return moves.move_data(mid).num


def test_move_classes_from_the_dex():
    t = pd.move_class_by_num()
    want = {"explosion": "self_ko", "selfdestruct": "self_ko", "memento": "self_ko",
            "earthquake": "attack", "rapidspin": "attack", "toxic": "status", "protect": "status",
            "swordsdance": "setup", "dragondance": "setup", "bellydrum": "setup", "curse": "setup",
            "spikes": "hazard", "recover": "recovery", "softboiled": "recovery", "rest": "recovery",
            "roar": "phazing", "whirlwind": "phazing"}
    got = {m: t[_num(m)] for m in want}
    assert got == want


def test_action_classes_switch_move_slot_and_struggle():
    t = pd.move_class_by_num()
    nums = np.array([[_num("explosion"), _num("spikes"), _num("earthquake"), _num("recover")]] * 6)
    acts = np.array([3, 6, 7, 8, 9, 10])
    assert pd.action_classes(acts, nums, t) == ["switch", "self_ko", "hazard", "attack", "recovery", "other"]
    # an empty / unknown request slot is "other", never silently an attack
    assert pd.action_classes(np.array([6]), np.array([[0, 0, 0, 0]]), t) == ["other"]


def test_class_shares_and_deltas():
    s = pd.class_shares(["switch", "attack", "attack", "setup"])
    assert s["attack"] == 0.5 and s["switch"] == 0.25 and s["hazard"] == 0.0
    assert set(s) == set(pd.CLASSES)
    d = pd.share_deltas(s, pd.class_shares(["attack"] * 4))
    assert np.isclose(d["attack"], -0.5) and np.isclose(d["setup"], 0.25)


# ── references + cycling ──────────────────────────────────────────────────────────────────────
def test_pick_back_ref_is_latest_at_least_back_steps_earlier():
    steps = [2_000_000, 4_000_000, 9_000_000, 12_000_000]
    assert pd.pick_back_ref(steps, 20_000_000, 10_000_000) == 9_000_000
    assert pd.pick_back_ref(steps, 14_000_000, 10_000_000) == 4_000_000
    assert pd.pick_back_ref(steps, 11_000_000, 10_000_000) is None
    assert pd.pick_prev(steps, 12_000_000) == 9_000_000 and pd.pick_prev(steps, 2_000_000) is None


def test_cycling_flags_only_a_move_BACK_toward_the_reference():
    assert pd.cycling_refs({"anchor": 0.10}, {"anchor": 0.20}) == ["anchor"]      # closer → flag
    assert pd.cycling_refs({"anchor": 0.30}, {"anchor": 0.20}) == []              # farther → normal
    assert pd.cycling_refs({"anchor": 0.198}, {"anchor": 0.20}) == []             # within 5% → wobble
    assert pd.cycling_refs({"back": None}, {"back": 0.2}) == []
    assert pd.cycling_refs({"back": 0.05, "anchor": 0.4}, {"back": 0.2, "anchor": 0.3}) == ["back"]


def test_verdict_words():
    zero = {c: 0.0 for c in pd.CLASSES}
    flat = {"n": 10, "flips": 0, "rate": 0.0,
            "buckets": {l: {"n": 3, "flips": 0, "rate": 0.0} for l in pd.bucket_labels()}}
    ref = {"step": 1, "kl_mean": .1, "kl_median": .1, "flip": flat, "share_delta": dict(zero)}
    row = {"status": "ok", "cycling": [], "refs": {"prev": ref, "back": None, "anchor": ref}}
    assert pd.verdict(row) == "refining"
    shifted = dict(ref, share_delta=dict(zero, setup=0.08))
    assert pd.verdict(dict(row, refs={"prev": ref, "back": None, "anchor": shifted})).startswith("shifting (setup +8pp")
    assert pd.verdict(dict(row, cycling=["anchor"])).startswith("cycling?")


# ── the resumable driver ──────────────────────────────────────────────────────────────────────
def _fake_run(tmp: Path, steps):
    d = tmp / "run" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    old = time.time() - 3600
    for s in steps:
        f = d / f"snapshot_{s:012d}.zip"
        f.write_bytes(b"x")
        os.utime(f, (old, old))
    return tmp / "run"


def _probe(n=40, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, A), np.float32)
    mask[:, [0, 1, 6, 7, 8]] = 1
    return rng.normal(size=(n, 20)).astype(np.float32), mask, np.zeros((n, 4), np.int64)


class _Probs:
    def __init__(self, mask):
        self.mask, self.calls = mask, []

    def __call__(self, zp):
        self.calls.append(zp.name)
        s = int(zp.name.split("_")[1].split(".")[0])
        rng = np.random.default_rng(s)
        p = rng.random(self.mask.shape).astype(np.float32) * self.mask
        return p / p.sum(-1, keepdims=True)


def test_resume_skips_recorded_steps_and_reuses_cached_probs(tmp_path):
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [4_000_000, 6_000_000])
    out = tmp_path / "out"
    fn = _Probs(mask)
    rows = pd.process_pending(run, out, obs, mask, nums, fn, table={}, log=lambda s: None)
    assert [r["step"] for r in rows] == [4_000_000, 6_000_000] and len(fn.calls) == 2
    assert rows[1]["refs"]["prev"]["step"] == 4_000_000 and rows[1]["refs"]["anchor"]["step"] == 4_000_000

    # a restart with nothing new forwards NOTHING and writes nothing
    fn2 = _Probs(mask)
    assert pd.process_pending(run, out, obs, mask, nums, fn2, table={}, log=lambda s: None) == []
    assert fn2.calls == []

    # a new snapshot → exactly one forward, and one new row
    _fake_run(tmp_path, [16_000_000])
    fn3 = _Probs(mask)
    rows = pd.process_pending(run, out, obs, mask, nums, fn3, table={}, log=lambda s: None)
    assert fn3.calls == ["snapshot_000016000000.zip"]
    assert rows[0]["refs"]["back"]["step"] == 6_000_000        # latest ≥ 10M earlier
    assert [r["step"] for r in pd.load_rows(out)] == [4_000_000, 6_000_000, 16_000_000]


def test_resume_after_a_kill_between_probs_and_row(tmp_path):
    """The probs are cached but the row was torn mid-write: the step is REDONE without a forward."""
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [4_000_000, 6_000_000])
    out = tmp_path / "out"
    pd.process_pending(run, out, obs, mask, nums, _Probs(mask), table={}, log=lambda s: None)
    lines = (out / "rows.jsonl").read_text().splitlines()
    (out / "rows.jsonl").write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2])   # torn
    assert [r["step"] for r in pd.load_rows(out)] == [4_000_000]
    fn = _Probs(mask)
    rows = pd.process_pending(run, out, obs, mask, nums, fn, table={}, log=lambda s: None)
    assert fn.calls == [] and [r["step"] for r in rows] == [6_000_000]
    assert json.loads((out / "rows.jsonl").read_text().splitlines()[-1])["step"] == 6_000_000


def test_a_pruned_snapshot_still_serves_as_a_reference(tmp_path):
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [4_000_000])
    out = tmp_path / "out"
    pd.process_pending(run, out, obs, mask, nums, _Probs(mask), table={}, log=lambda s: None)
    (run / "snapshots" / "snapshot_000004000000.zip").unlink()          # the pool window slid
    _fake_run(tmp_path, [15_000_000])
    rows = pd.process_pending(run, out, obs, mask, nums, _Probs(mask), table={}, log=lambda s: None)
    assert rows[0]["refs"]["back"]["step"] == 4_000_000


def test_a_settling_zip_waits_for_the_next_poll(tmp_path):
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [4_000_000])
    (run / "snapshots" / "snapshot_000006000000.zip").write_bytes(b"x")    # mtime = now
    fn = _Probs(mask)
    rows = pd.process_pending(tmp_path / "run", tmp_path / "out", obs, mask, nums, fn,
                              table={}, log=lambda s: None)
    assert [r["step"] for r in rows] == [4_000_000] and fn.calls == ["snapshot_000004000000.zip"]


def test_cycling_is_recorded_when_the_policy_returns_toward_the_anchor(tmp_path):
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [2_000_000, 4_000_000, 6_000_000])
    base = _Probs(mask)(Path("snapshot_000002000000.zip"))
    far = _Probs(mask)(Path("snapshot_000009999999.zip"))
    table = {2_000_000: base, 4_000_000: far, 6_000_000: 0.9 * base + 0.1 * far}
    fn = lambda zp: table[int(zp.name.split("_")[1].split(".")[0])]      # noqa: E731
    rows = pd.process_pending(run, tmp_path / "out", obs, mask, nums, fn, table={}, log=lambda s: None)
    assert rows[1]["cycling"] == []
    assert rows[2]["cycling"] == ["anchor"] and pd.verdict(rows[2]).startswith("cycling?")


def test_render_report_marks_it_a_descriptor(tmp_path):
    from main.policy_drift import render
    obs, mask, nums = _probe()
    run = _fake_run(tmp_path, [4_000_000, 6_000_000])
    rows = pd.process_pending(run, tmp_path / "out", obs, mask, nums, _Probs(mask), table={},
                              log=lambda s: None)
    txt = render(rows)
    assert "DESCRIPTOR, not a test" in txt and "6,000,000" in txt


def test_a_margin_exactly_on_an_edge_goes_to_the_higher_bucket():
    old = _p([{6: 0.625, 7: 0.375}])         # margin 0.25 exactly (binary-exact)
    new = _p([{6: 0.375, 7: 0.625}])
    r = pd.flip_by_margin(old, new, _mask_of(old), edges=(0.25, 0.5))
    assert r["buckets"]["0.25-0.5"]["n"] == 1 and r["buckets"]["<0.25"]["n"] == 0
