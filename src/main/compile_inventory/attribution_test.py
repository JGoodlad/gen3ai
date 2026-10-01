"""The component rules and the static fold scan (M5 Lane K8)."""
from __future__ import annotations

from main.compile_inventory.attribution import (Frame, classify, loss_detail, parse_stack,
                                                scan_fold_host_syncs)

SB3 = "/env/site-packages/stable_baselines3/common/policies.py"
DIST = "/env/site-packages/sb3_contrib/common/maskable/distributions.py"
FE = "/repo/src/agents/model/features_extractor.py"
POL = "/repo/src/agents/model/policy.py"
PPO = "/repo/src/agents/training/instrumented_ppo/ppo.py"
VT = "/repo/src/agents/training/instrumented_ppo/value_terms.py"


def test_extractor_beats_loss_when_a_loss_term_runs_its_own_forward():
    stack = [Frame(PPO, 1, "train"), Frame(VT, 2, "_td_aux_term"), Frame(FE, 3, "forward")]
    assert classify(stack) == "extractor"


def test_masking_under_evaluate_actions_is_the_distribution():
    stack = [Frame(PPO, 1, "train"), Frame(POL, 2, "evaluate_actions"),
             Frame(DIST, 3, "apply_masking")]
    assert classify(stack) == "distribution+masking"


def test_heads_and_the_extractor_under_the_policy():
    head = [Frame(PPO, 1, "train"), Frame(POL, 2, "evaluate_actions")]
    assert classify(head) == "heads"
    assert classify(head + [Frame(SB3, 3, "extract_features")]) == "extractor"


def test_a_loss_helper_is_named_by_its_own_method():
    stack = [Frame(PPO, 1, "train"), Frame(VT, 2, "_value_loss_from_se")]
    assert classify(stack) == "loss+diagnostics"
    assert loss_detail(stack) == "agents/training/instrumented_ppo/value_terms.py:_value_loss_from_se"


def test_parse_stack_reads_both_traceback_shapes():
    tb = '  File "/a/b.py", line 3, in f\n    x\n  File "/a/c.py", line 9, in g\n'
    assert parse_stack(tb) == [Frame("/a/b.py", 3, "f"), Frame("/a/c.py", 9, "g")]
    fs = "<FrameSummary file /a/b.py, line 3 in f>"
    assert parse_stack(fs) == [Frame("/a/b.py", 3, "f")]


def _fold(self, buf, logger):
    import numpy as np
    for epoch in range(2):
        for rollout_data in buf:
            loss = rollout_data.sum()
            vals = [loss.item()]                       # host read
            if bool(loss > 0):                         # branch on a value
                loss = loss * 2
            for k in ("a", "b"):                       # inner loop
                logger.record(k, float(loss))          # logging + host read
            np.mean(vals)                              # numpy
            x = loss.cpu()                             # host copy
    return x


def test_the_fold_scan_finds_each_construct_at_its_line():
    import inspect
    r = scan_fold_host_syncs(_fold)
    start = inspect.getsourcelines(_fold)[1]
    kinds = {(s["kind"], s["line"] - start) for s in r["sites"]}
    assert ("host read (.item)", 5) in kinds
    assert ("python branch on a value", 6) in kinds
    assert ("host read (float/int/bool of a value)", 6) in kinds
    assert ("inner python loop", 8) in kinds
    assert ("logging (logger.record)", 9) in kinds
    assert ("host read (float/int/bool of a value)", 9) in kinds
    assert ("numpy call", 10) in kinds
    assert ("host copy (.cpu/.numpy/.tolist)", 11) in kinds
    assert r["loop_line"] == start + 3
    # the outer `for epoch` loop and the `for rollout_data` loop itself are not sites
    assert not any(s["line"] - start in (2, 3) for s in r["sites"])


def test_a_chained_host_copy_is_one_site():
    def f(buf):
        for rollout_data in buf:
            x = rollout_data.cpu().numpy()
        return x
    r = scan_fold_host_syncs(f)
    assert r["counts"] == {"host copy (.cpu/.numpy/.tolist)": 1}
