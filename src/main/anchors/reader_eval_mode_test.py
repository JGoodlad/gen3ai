"""Every reader of a checkpoint reads the policy in EVAL mode (X59-S CUDA pre-flight check 5, 2026-10-26).

The anchors harness's ``bare`` load (``load_checkpoint_strict``) returned the policy in TRAINING mode, where
``nn.MultiheadAttention`` skips its eval-mode fused fast path: a different kernel, 4.8e-7 off the student's own
logits on 18 of 256 real rows, so the student's bit-equal export check refused it. Fails on revert of the
``set_training_mode(False)`` in ``load_checkpoint_strict`` or in ``main.anchors.session.foreign_loader``. The
logit-level consequence on real rows is ``main.human_data.student_test``'s real-rows export test. The mode each load
leaves is READ BACK into ``policy_mode``, stamped on every row (``main.reads.side_identity``)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("loader", ["bare", "foreign"])
def test_the_anchors_loaders_return_an_eval_mode_policy(tmp_path, loader, restore_torch_globals):
    from main.anchors.core_side import OurSideState, load_policy
    from main.fresh_checkpoint import save_fresh_checkpoint

    zp = save_fresh_checkpoint(tmp_path / "run", 5901)
    st = OurSideState()
    model = load_policy(str(zp), "cpu", loader, st)
    assert st.model_loader == loader
    training = sorted(n for n, m in model.policy.named_modules() if m.training)
    assert not training, f"modules left in training mode by the {loader} load: {training[:5]}"
    assert st.policy_mode == "eval", "the mode stamped on every row is READ BACK from the loaded policy"


def test_the_policy_mode_is_read_back_from_the_modules_never_from_the_loader():
    """TECH_DEBT P1 (b): ``policy_mode`` (stamped per cell, in the regime id unless the banked "train") is what the
    loaded modules say: all eval, all training, or a mix."""
    import types

    import torch

    from main.anchors.core_side import policy_mode

    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.MultiheadAttention(4, 2))
    model = types.SimpleNamespace(policy=net)
    assert policy_mode(model) == "train"
    net.eval()
    assert policy_mode(model) == "eval"
    net[1].train()
    assert policy_mode(model) == "mixed"
