"""Unit tests for the self-contained distilled opponent student + RLPlayer adapter.

Pure unit tests (no server, no teacher snapshot): a fresh untrained student is enough to verify
the architecture is self-contained, produces valid logits, the adapter is RLPlayer-shaped, save/load
round-trips, and the head is trainable. Faithfulness is validated separately (the recipe + h2h gate).
"""
import os
import tempfile

import numpy as np
import torch

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.distill.student import (
    DistilledStudent, DistilledOpponentModel, save_distilled, load_distilled,
)


def _layout():
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _obs(layout, B=4):
    # zeros => valid embedding indices (categorical-id fields); the additive token-type /
    # positional pieces keep the net non-degenerate, same trick the extractor's roundtrip uses.
    return {"observation": torch.zeros(B, layout["total_dim"]),
            "action_mask": torch.ones(B, 11, dtype=torch.int8)}


def test_forward_shape_and_finite():
    layout = _layout()
    s = DistilledStudent(layout).eval()
    with torch.no_grad():
        out = s(_obs(layout, B=4))
    assert out.shape == (4, 11), out.shape
    assert torch.isfinite(out).all()


def test_self_contained_no_teacher_reference():
    # the student must own its unpack + embeddings (no external teacher needed at inference)
    layout = _layout()
    s = DistilledStudent(layout)
    names = dict(s.named_children())
    assert "emb" in names and "unpack" in names and "encoder" in names and "head" in names
    # embeddings are real parameters carried by the student (so the saved .pt is self-sufficient)
    assert any(p.requires_grad for p in s.emb.parameters())


def test_adapter_is_rlplayer_shaped():
    layout = _layout()
    s = DistilledStudent(layout)
    model = DistilledOpponentModel(s, device="cpu")
    assert hasattr(model, "device") and hasattr(model.policy, "get_distribution")
    dist = model.policy.get_distribution(_obs(layout, B=3))
    logits = dist.distribution.logits
    assert logits.shape == (3, 11)
    # predict_values is benign (opponent never uses it) but must not crash
    assert model.policy.predict_values(_obs(layout, B=3)).shape == (3, 1)


def test_save_load_roundtrip_identical():
    layout = _layout()
    s = DistilledStudent(layout, enc_hidden=128, head_hidden=256).eval()
    with torch.no_grad():
        before = s(_obs(layout, B=2))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "student.pt")
        save_distilled(s, p)
        s2 = load_distilled(p, layout)
    assert s2.config == s.config
    with torch.no_grad():
        after = s2(_obs(layout, B=2))
    assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()


def test_head_is_trainable():
    layout = _layout()
    s = DistilledStudent(layout)
    s.freeze_embeddings()
    opt = torch.optim.Adam([p for p in s.parameters() if p.requires_grad], lr=1e-3)
    obs = {"observation": torch.zeros(8, layout["total_dim"])}
    target = torch.zeros(8, 11); target[:, 0] = 1.0  # arbitrary
    before = float(((torch.softmax(s(obs), 1) - target) ** 2).mean())
    for _ in range(20):
        loss = ((torch.softmax(s(obs), 1) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = float(((torch.softmax(s(obs), 1) - target) ** 2).mean())
    assert after < before, (before, after)


def test_capacity_ladder_sizes():
    # the manager escalates capacity; bigger config => more params, same I/O contract
    layout = _layout()
    small = DistilledStudent(layout, enc_hidden=128, head_hidden=256)
    big = DistilledStudent(layout, enc_hidden=384, head_hidden=512)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small
    with torch.no_grad():
        assert small(_obs(layout, 2)).shape == big(_obs(layout, 2)).shape == (2, 11)
