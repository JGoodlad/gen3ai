"""`recording_agreement` — the reader's teeth — on a synthetic bank (no archive, no model, no games).

WHY THIS FILE EXISTS. The only test of `recording_agreement` on real data was
`policy_spectrum_integration_test.test_reader_reproduces_the_recording_policy_and_is_deterministic`,
which read the banked policy `ai_v14_01_base/eval_traces/step_74000016/snapshot.zip`. That file went
with every pre-Rustboro run's `eval_traces/` in the owner-approved 2026-10-09 cleanup (and the policy
was a pre-X5 checkpoint, refused at HEAD with the pinned-commit message since the version break, so the
test could only ever skip after asserting the refusal). The function is still live: the next banked
policy that is an `rb_` eval snapshot is read through it. These tests pin its arithmetic on a bank built
in memory, so the check survives the artifact.
"""
from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum import spectrum as S
from main.policy_spectrum.bank import Bank, BankBattle
from main.policy_spectrum.reader import recording_agreement

A = 11
BANKED = "rb_x/eval_traces/step_1000/snapshot.zip"


def _battle(bid="b0", policy=BANKED, banked="p1", traced="p1"):
    return BankBattle(battle_id=bid, source={"label": "x", "banked_policy": policy},
                      format_id="gen3ou", seed="1", p1={}, p2={}, commands=[],
                      banked_side=banked, traced_side=traced, outcome="win")


def _decisions(bid, logits):
    return [{"id": f"{bid}:{i}", "battle": bid, "rec_logits": list(map(float, row)),
             "mask": "1" * A} for i, row in enumerate(logits)]


def _logits(n=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, A)).astype(np.float32)


def _probs(logits):
    return S.masked_probs(logits, np.ones(logits.shape, dtype=bool))


def _root(tmp_path):
    (tmp_path / "rb_x" / "eval_traces" / "step_1000").mkdir(parents=True)
    zip_path = tmp_path / BANKED
    zip_path.write_bytes(b"x")
    return tmp_path, zip_path


def test_a_read_that_reproduces_the_recording_agrees_exactly(tmp_path):
    root, zip_path = _root(tmp_path)
    lg = _logits()
    bank = Bank({}, [_battle()], _decisions("b0", lg))
    r = recording_agreement(bank, _probs(lg), zip_path, root)
    assert r == {"decisions": 6, "max_abs_dp": 0.0, "argmax_agree": 1.0, "argmax_flip_max_margin": None}


def test_a_flipped_argmax_is_reported_with_its_margin_and_a_real_dp(tmp_path):
    root, zip_path = _root(tmp_path)
    lg = _logits()
    probs = _probs(lg)
    swapped = probs.copy()
    top2 = np.argsort(swapped[2])[-2:]
    swapped[2, top2[0]], swapped[2, top2[1]] = swapped[2, top2[1]], swapped[2, top2[0]]   # row 2 flips
    bank = Bank({}, [_battle()], _decisions("b0", lg))
    r = recording_agreement(bank, swapped, zip_path, root)
    assert r["decisions"] == 6 and r["argmax_agree"] == pytest.approx(5 / 6)
    assert r["max_abs_dp"] > 0.0
    margin = float(np.sort(probs[2])[-1] - np.sort(probs[2])[-2])
    assert r["argmax_flip_max_margin"] == pytest.approx(margin)


def test_only_decisions_the_checkpoint_itself_recorded_count(tmp_path):
    """A banked decision from another policy, or one banked from the OTHER side of the trace, has no
    recorded logits of THIS checkpoint to be held to."""
    root, zip_path = _root(tmp_path)
    lg = _logits(4)
    own = _battle("own")
    other_policy = _battle("elsewhere", policy="rb_y/eval_traces/step_1/snapshot.zip")
    other_side = _battle("flipped", banked="p2", traced="p1")
    bank = Bank({}, [own, other_policy, other_side],
                _decisions("own", lg) + _decisions("elsewhere", lg) + _decisions("flipped", lg))
    probs = np.concatenate([_probs(lg)] * 3)
    r = recording_agreement(bank, probs, zip_path, root)
    assert r["decisions"] == 4


def test_it_answers_none_when_it_cannot_judge(tmp_path):
    root, zip_path = _root(tmp_path)
    lg = _logits(3)
    bank = Bank({}, [_battle()], _decisions("b0", lg))
    probs = _probs(lg)
    assert recording_agreement(bank, probs, zip_path, None) is None          # no archive root
    outside = tmp_path.parent / "elsewhere.zip"
    assert recording_agreement(bank, probs, outside, root) is None           # not under the root
    unbanked = root / "rb_x" / "final_model.zip"
    unbanked.write_bytes(b"x")
    assert recording_agreement(bank, probs, unbanked, root) is None          # no battle names it
