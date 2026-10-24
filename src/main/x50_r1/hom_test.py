"""The HOM trainer's pure parts (design §4): the split, the row filter, the batch plan, the schedule and the stop
rule. Each test fails when its rule is reverted."""
from __future__ import annotations

import hashlib

import numpy as np

from main.x50_r1 import hom as H


def test_the_validation_split_is_sha256_mod_50_of_the_battle_id():
    ids = [f"battle-gen3ou-{i}" for i in range(5000)]
    val = [i for i in ids if H.is_val(i)]
    assert val == [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 50 == 0]
    assert 60 <= len(val) <= 140


def test_the_rows_the_hom_reads():
    z = {"mode": np.array([2, 0, 2, 2, 2, 2]), "own_action": np.array([3, 3, -1, 7, 7, 7]),
         "n_legal": np.array([4, 4, 4, 1, 4, 4]), "rating_viewer": np.array([1500, 1600, 1600, 1600, 1499, 1720])}
    assert H.eligible(z).tolist() == [True, False, False, False, False, True]


def test_the_plan_is_pure_covers_every_shard_each_epoch_and_quarter_evals():
    sizes = [16384] * 19 + [700]
    a, b = H.plan(sizes), H.plan(list(sizes))
    assert a == b
    for e in range(H.EPOCHS):
        ids = sorted(i for ep, _, sh, _ in a if ep == e for i in sh)
        assert ids == list(range(len(sizes)))
    assert [sh for ep, _, sh, _ in a if ep == 0] != [sh for ep, _, sh, _ in a if ep == 1]
    total = sum(nb for *_, nb in a)
    ev = H.eval_steps(a)
    assert len(ev) == 4 * H.EPOCHS and ev[-1] == total and ev == sorted(ev)


def test_the_cosine_schedule_and_the_stop_rule():
    assert H.lr_at(0, 1000) == H.LR_MAX and abs(H.lr_at(1000, 1000) - H.LR_MIN) < 1e-15
    assert abs(H.lr_at(500, 1000) - (H.LR_MAX + H.LR_MIN) / 2) < 1e-15
    assert not H.stop_now([2.0, 1.9, 1.95])
    assert H.stop_now([2.0, 1.9, 1.95, 1.97])
    assert not H.stop_now([2.0, 1.9, 1.95, 1.94, 1.96])


def test_r_b1s_rule_takes_the_conservative_side_of_the_band():
    from main.x50_r1.read_rb1 import learned_rule

    assert learned_rule(-0.0525) == "LEARNED"
    assert learned_rule(-0.0515) == "NOT LEARNED"           # within 0.002 of -0.05: conservative
    assert learned_rule(-0.03) == "NOT LEARNED"
