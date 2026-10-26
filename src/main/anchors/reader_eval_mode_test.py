"""Every reader of a checkpoint reads the policy in EVAL mode (X59-S CUDA pre-flight check 5, 2026-10-26).

The anchors harness's ``bare`` load (``load_checkpoint_strict``) returned the policy in TRAINING mode, where
``nn.MultiheadAttention`` skips its eval-mode fused fast path: a different kernel, 4.8e-7 off the student's own
logits on 18 of 256 real rows, so the student's bit-equal export check refused it. Fails on revert of the
``set_training_mode(False)`` in ``load_checkpoint_strict`` or in ``main.anchors.session.foreign_loader``. The
logit-level consequence on real rows is ``main.human_data.student_test``'s real-rows export test."""
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
