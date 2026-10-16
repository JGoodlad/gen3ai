"""T38, the plateau meter's PARALLEL BATCHES: the decision rule is unchanged, proved on RECORDED Tier-1 checks.

``plateau_tent_replay.json`` holds three real ``plateau_t1`` requests of the T-ENT toy (T33, 2026-10-15, the archive
ledger): their per-batch pentanomials, cycle seeds and outcome digests, and the decision on record. A REPLAY drives
each through :func:`main.plateau.drive_check` at several in-flight counts, with a stand-in for the engine that
writes each wave's rows under their claims in REVERSE batch order (arrival order is not batch order) and whose
batches PAST the recorded stop are ADVERSARIAL — every pair a double loss for the newest node. The decision must be
the recorded one (verdict, stop batch, pairs, LLR to the last bit) at every in-flight count: the rule folds batches
0, 1, 2, … and stops at the first crossing, so a later batch can never decide, delay or reverse an earlier look.
``verify`` re-derives it from the rows (surplus included). The engine path — the same games under a wave — is
``main/h2h/plateau_integration_test.py``."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T
from agents.training import sprt as SP
from agents.training.eval_ledger import audit as A
from agents.training.eval_ledger import testkit as K
from main import plateau as P

FIXTURE = Path(__file__).with_name("plateau_tent_replay.json")
#: an adversarial batch: all 40 pairs lost twice by the newest node (pentanomial category 0)
ADVERSARIAL = (40, 0, 0, 0, 0)
INFLIGHT = (1, 2, 3, 4, 8, 150)
CLOCK = K.Clock()


def recorded():
    return json.loads(FIXTURE.read_text())["checks"]


class WaveFake:
    """A stand-in for ``play_upto`` under parallel batches: claims every missing batch ``< n`` first, then appends
    them in REVERSE batch order (a wave's rows need not arrive in batch order). A batch the recording holds plays its
    recorded pentanomial; one past it is :data:`ADVERSARIAL`."""

    def __init__(self, writer, req, script):
        self.w, self.req, self.script = writer, req, list(script)
        self.asked, self.played = [], []

    def __call__(self, n):
        self.asked.append(n)
        have = {int(r["request"]["batch"]) for r in L.read(P.TIER1_ROWS, root=self.w.root,
                                                            request_id=self.req["request_id"]).rows}
        reg = K.regime()
        wave = [b for b in range(n) if b not in have]
        claims = {b: self.w.claim(self.req["request_id"], batch=b, player=K.SHA_A, opponent=K.SHA_B,
                                  regime_id=reg["regime_id"], expected_wall_s=60.0) for b in wave}
        for b in reversed(wave):
            self.w.clock.advance(2)
            pc = self.script[b] if b < len(self.script) else ADVERSARIAL
            row = K.make_row(self.w.next_row_id(), request=self.req, batch=b, pc=pc, purpose="plateau",
                             ts=self.w.clock.iso(), cycle_seed=900 + b,
                             schedule_key="plateau_t1:" + self.req["request_id"])
            self.w.append_row(row, claims[b])
            self.played.append(b)


def test_the_fixture_is_what_the_rule_decided_on_record():
    for chk in recorded():
        rule = T.Tier1Rule.parse(chk["decision"]["rule"])
        assert rule.registered, "the recorded checks ran the REGISTERED rule"
        bs = chk["batches"]
        assert [b["batch"] for b in bs] == list(range(len(bs)))
        assert all(b["counts"]["aborted"] == 0 and sum(b["pair_counts"]) == rule.batch_pairs for b in bs)
        pc = [0] * 5
        for b in bs:
            pc = [x + y for x, y in zip(pc, b["pair_counts"])]
        rec = chk["recorded"]
        assert sum(pc) == rec["n_pairs"] and round(SP.llr(pc, rule.p0, rule.p1), 9) == rec["llr"]
        assert rec["stop_batch"] == len(bs) - 1 and rec["surplus_batches"] == []
        assert chk["decision"]["consumed_count"] == len(bs)


@pytest.mark.parametrize("inflight", INFLIGHT)
@pytest.mark.parametrize("idx", range(3))
def test_a_recorded_check_replays_to_the_same_decision_under_parallel_batches(tmp_path, idx, inflight):
    chk = recorded()[idx]
    rule = T.Tier1Rule.parse(chk["decision"]["rule"])
    rec, script = chk["recorded"], [tuple(b["pair_counts"]) for b in chk["batches"]]
    root = tmp_path / "ledger"
    w = L.LedgerWriter(root, producer="h2h", clock=CLOCK)
    req = w.open_request(chk["request_id"], kind=T.REQUEST_KIND, purpose=T.PURPOSE, protocol=K.PROTO,
                         spec={"replay_of": chk["decision"]["decision_id"]})
    fake = WaveFake(w, req, script)
    d = P.drive_check(w, root, req["request_id"], rule, K.SHA_A, fake, lambda _m: None, inflight=inflight)
    w.close()
    stop, cap = rec["stop_batch"], rule.max_pairs // rule.batch_pairs
    # the waves: K, 2K, ... batches asked for, never past the cap, and the last one covers the recorded stop
    waves = math.ceil((stop + 1) / inflight)
    assert fake.asked == [min(cap, inflight * (k + 1)) for k in range(waves)]
    played = min(cap, inflight * waves)
    assert sorted(fake.played) == list(range(played))
    # THE DECISION IS THE RECORDED ONE, to the last bit, whatever arrived after its stop
    rows = list(L.read(P.TIER1_ROWS, root=root, request_id=req["request_id"]).rows)
    res = T.evaluate(rows, rule)
    assert d["verdict"] == res.verdict == chk["decision"]["verdict"]
    assert (res.stop_batch, res.n_pairs, round(res.llr, 9)) == (stop, rec["n_pairs"], rec["llr"])
    assert res.surplus_batches == list(range(stop + 1, played))
    assert [t["batch"] for t in res.trail] == list(range(stop + 1)), "folded in batch order, stopped at the crossing"
    assert d["consumed"]["count"] == played
    v = A.verify(root, d["decision_id"])
    assert v.ok and v.verdict_rederived == chk["decision"]["verdict"], (v.problems, v.note)
    assert A.audit(root).ok


def test_the_adversarial_surplus_would_have_reversed_a_peeking_fold():
    """The replay's surplus has teeth: had a fold read the batches past the stop, the 90M check (stop at batch 29)
    with one full wave of 8 would no longer sit above the upper bound."""
    chk = recorded()[2]
    rule = T.Tier1Rule.parse(chk["decision"]["rule"])
    pc = [0] * 5
    for b in chk["batches"]:
        pc = [x + y for x, y in zip(pc, b["pair_counts"])]
    surplus = 8 * math.ceil(len(chk["batches"]) / 8) - len(chk["batches"])
    assert surplus > 0
    peeked = [pc[0] + surplus * ADVERSARIAL[0]] + pc[1:]
    _lo, hi = rule.bounds
    assert SP.llr(pc, rule.p0, rule.p1) > hi > SP.llr(peeked, rule.p0, rule.p1)


def test_inflight_never_asks_past_the_cap_and_refuses_zero(tmp_path):
    rule = T.Tier1Rule(batch_pairs=2, min_pairs=2, max_pairs=6)
    root = tmp_path / "ledger"
    w = L.LedgerWriter(root, producer="h2h", clock=CLOCK)
    req = w.open_request("plateau_t1:r:1:w1", kind=T.REQUEST_KIND, purpose=T.PURPOSE, protocol=K.PROTO, spec={})
    fake = WaveFake(w, req, [(0, 0, 2, 0, 0)] * 3)                      # never crosses: UNDECIDED at the cap
    d = P.drive_check(w, root, req["request_id"], rule, K.SHA_A, fake, lambda _m: None, inflight=8)
    assert fake.asked == [3] and d["verdict"] == T.UNDECIDED
    with pytest.raises(P.PlateauError):
        P.drive_check(w, root, req["request_id"], rule, K.SHA_A, fake, lambda _m: None, inflight=0)
    w.close()


def test_the_cli_knob_defaults_to_the_declared_value():
    a = P._parser().parse_args(["tick", "rb_x"])
    assert a.inflight_batches == P.DEFAULT_INFLIGHT == 4
    assert P._parser().parse_args(["tick", "rb_x", "--inflight-batches", "1"]).inflight_batches == 1


def test_the_fixture_matches_the_archive_ledger_when_it_is_here():
    """Provenance: where the run archive exists, the fixture is the archive ledger's rows and decisions verbatim."""
    from utils.paths import main_models_dir

    mm = main_models_dir()
    if mm is None or not (Path(mm) / "_ledger").is_dir():
        pytest.skip("no run archive ledger on this checkout")
    root = Path(mm) / "_ledger"
    for chk in recorded():
        rows = sorted(L.read(P.TIER1_ROWS, root=root, request_id=chk["request_id"]).rows,
                      key=lambda r: int(r["request"]["batch"]))
        if not rows:
            pytest.skip(f"{chk['request_id']} is not in this archive's ledger")
        assert [(r["row_id"], r["pairs"]["counts"], r["seed"]["cycle_seed"]) for r in rows] == \
            [(b["row_id"], b["pair_counts"], b["cycle_seed"]) for b in chk["batches"]]
        ds = L.read_decisions(root=root, kind="plateau", request_ids=[chk["request_id"]])
        assert [(d["decision_id"], d["verdict"]) for d in ds] == \
            [(chk["decision"]["decision_id"], chk["decision"]["verdict"])]
