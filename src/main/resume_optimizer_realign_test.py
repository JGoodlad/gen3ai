"""Regression test for the resume optimizer-state realignment guard
(`_validate_or_reset_optimizer_state`).

The bug it pins (gen3_nature_ev_belief_v1 / the v40 SpreadBelief reorder): SB3/torch save+load the
Adam optimizer state BY PARAMETER POSITION. A refactor that reorders a module's parameters between a
checkpoint save and a resume silently misassigns the saved per-param momentum to the WRONG params —
weights still load (name-keyed) so the arch check passes, then `AdamW.step()` crashes
("size of tensor a (128) must match b (5)") the moment a misassigned param of a different shape first
gets a gradient. The guard treats ANY param↔state shape mismatch as proof the whole state is
misaligned and resets the momentum (fresh zero-init), so the resume proceeds instead of crashing.

The newer name-keyed remap (`_remap_optimizer_state_by_name` / `_read_saved_optimizer_state`) SUPERSEDES
that shape-only reset whenever the checkpoint zip is readable: it re-keys the saved momentum to the
current params by NAME, so a reorder — including a SAME-shape one the shape check cannot see — is
CORRECTED rather than reset or scrambled. The shape-only path remains as the no-zip fallback."""
import collections
import types

import torch
from torch import nn

from main.train_rl_agent import _validate_or_reset_optimizer_state


def _fake_model_with_populated_optimizer():
    """A tiny two-Linear policy whose AdamW has real, correctly-shaped momentum after one step."""
    policy = nn.Module()
    policy.a = nn.Linear(5, 128)    # weight [128, 5]
    policy.b = nn.Linear(128, 5)    # weight [5, 128]  (distinct shape — a reorder is shape-detectable)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    loss = policy.b(policy.a(torch.randn(4, 5))).sum()
    loss.backward()
    opt.step()                      # populates exp_avg / exp_avg_sq with correct shapes
    model = types.SimpleNamespace(policy=policy)
    model.policy.optimizer = opt
    return model, policy, opt


def test_aligned_state_is_left_untouched():
    # A normal resume: every momentum buffer matches its param → no reset.
    model, _policy, opt = _fake_model_with_populated_optimizer()
    n_before = len(opt.state)
    assert n_before == 4            # a.weight, a.bias, b.weight, b.bias
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == n_before   # untouched — clean resume is unchanged


def test_misaligned_state_is_reset():
    # Simulate the reorder: give a.weight ([128,5]) the momentum SHAPE meant for b.weight ([5,128]).
    model, policy, opt = _fake_model_with_populated_optimizer()
    opt.state[policy.a.weight]["exp_avg"] = torch.zeros(5, 128)     # wrong shape for a.weight
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == 0      # whole state dropped → AdamW reinitialises fresh momentum


def test_reset_preserves_lr_and_param_groups():
    # The reset must clear momentum ONLY — LR / param_groups (where the resume LR lives) stay intact.
    model, policy, opt = _fake_model_with_populated_optimizer()
    opt.param_groups[0]["lr"] = 7e-5
    opt.state[policy.b.bias]["exp_avg_sq"] = torch.zeros(128)       # wrong shape for b.bias ([5])
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == 0
    assert opt.param_groups[0]["lr"] == 7e-5    # LR survives the momentum reset


def test_no_optimizer_is_noop():
    # An inference-only load (env=None) may have no optimizer — must not raise.
    _validate_or_reset_optimizer_state(types.SimpleNamespace(policy=types.SimpleNamespace(optimizer=None)))
    _validate_or_reset_optimizer_state(types.SimpleNamespace(policy=None))


# ---------------------------------------------------------------------------
# Name-keyed remap — the SAME-SHAPE reorder the shape-only guard cannot see.
# ---------------------------------------------------------------------------
import io          # noqa: E402
import zipfile     # noqa: E402

from main.train_rl_agent import (   # noqa: E402
    _read_saved_optimizer_state,
    _remap_optimizer_state_by_name,
)


class _Pair(nn.Module):
    """Two SAME-SHAPE Linear(4,4) submodules; `first` controls registration ORDER (a-then-b vs
    b-then-a) so we can simulate a parameter-reorder refactor while keeping the names a.*/b.* fixed."""
    def __init__(self, first="a"):
        super().__init__()
        if first == "a":
            self.a = nn.Linear(4, 4); self.b = nn.Linear(4, 4)
        else:
            self.b = nn.Linear(4, 4); self.a = nn.Linear(4, 4)


def _opt_order_named(opt, module):
    """(name, param) in the EXACT optimizer index order, mirroring how the real guard builds it."""
    id_to_name = {id(p): n for n, p in module.named_parameters()}
    return [(id_to_name[id(p)], p) for g in opt.param_groups for p in g["params"]]


def _saved_opt_with_marked_momentum():
    """Build a saved optimizer whose a.weight momentum is all-1.0 and b.weight all-2.0 (so we can tell
    by VALUE whose momentum a param ends up with), plus the saved param-name order (a before b)."""
    m = _Pair("a")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m.b(m.a(torch.randn(4, 4))).sum().backward()
    opt.step()
    opt.state[m.a.weight]["exp_avg"] = torch.ones(4, 4)         # marker: a -> 1.0
    opt.state[m.b.weight]["exp_avg"] = torch.full((4, 4), 2.0)  # marker: b -> 2.0
    saved_opt = opt.state_dict()
    saved_param_names = [n for n, _ in m.named_parameters()]    # ['a.weight','a.bias','b.weight','b.bias']
    return saved_opt, saved_param_names


def test_same_shape_reorder_momentum_follows_name():
    """THE headline gap: a.weight and b.weight are identical-shape, so a reorder is INVISIBLE to a
    shape check and would silently scramble momentum. The name-keyed remap must make each param's
    momentum follow its NAME across the swap — a.weight keeps 1.0, b.weight keeps 2.0."""
    saved_opt, saved_param_names = _saved_opt_with_marked_momentum()

    cur = _Pair("b")   # REORDERED: b registered before a (same names, swapped positions)
    opt = torch.optim.AdamW(cur.parameters(), lr=1e-3)
    counts = _remap_optimizer_state_by_name(opt, _opt_order_named(opt, cur), saved_opt, saved_param_names)

    assert counts["carried"] == 4 and counts["reordered"] == 4 and counts["dropped_shape"] == 0
    # momentum followed the NAME, not the position (position-keyed would have given b.weight=1.0):
    assert torch.equal(opt.state[cur.a.weight]["exp_avg"], torch.ones(4, 4))
    assert torch.equal(opt.state[cur.b.weight]["exp_avg"], torch.full((4, 4), 2.0))


def test_aligned_order_carries_momentum_unchanged():
    """When the order already matches (the normal resume), every entry is carried verbatim and nothing
    is flagged as reordered."""
    saved_opt, saved_param_names = _saved_opt_with_marked_momentum()
    cur = _Pair("a")   # same order as saved
    opt = torch.optim.AdamW(cur.parameters(), lr=1e-3)
    counts = _remap_optimizer_state_by_name(opt, _opt_order_named(opt, cur), saved_opt, saved_param_names)
    assert counts["carried"] == 4 and counts["reordered"] == 0
    assert torch.equal(opt.state[cur.a.weight]["exp_avg"], torch.ones(4, 4))


def test_remap_drops_momentum_on_shape_change():
    """A name reused at a DIFFERENT shape (a real arch change) must drop that param's momentum (fresh
    zero-init), not crash — the name-keyed analogue of the old shape guard."""
    saved_opt, saved_param_names = _saved_opt_with_marked_momentum()
    cur = nn.Module()
    cur.a = nn.Linear(8, 8)   # a.weight is now [8,8] and a.bias [8] — both differ from saved [4,4]/[4]
    cur.b = nn.Linear(4, 4)
    opt = torch.optim.AdamW(cur.parameters(), lr=1e-3)
    counts = _remap_optimizer_state_by_name(opt, _opt_order_named(opt, cur), saved_opt, saved_param_names)
    assert counts["dropped_shape"] == 2            # BOTH a.weight and a.bias changed shape → dropped
    assert cur.a.weight not in opt.state and cur.a.bias not in opt.state   # no scrambled momentum left
    assert torch.equal(opt.state[cur.b.weight]["exp_avg"], torch.full((4, 4), 2.0))  # b unaffected


def _write_fake_checkpoint_zip(path, policy_sd, saved_opt):
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO(); torch.save(policy_sd, buf); z.writestr("policy.pth", buf.getvalue())
        buf = io.BytesIO(); torch.save(saved_opt, buf); z.writestr("policy.optimizer.pth", buf.getvalue())


def test_read_saved_optimizer_state_filters_buffers_and_orders_params(tmp_path):
    """`_read_saved_optimizer_state` must recover the saved PARAM order from policy.pth (skipping
    buffers, like hp_type_idx_map) and the optimizer state."""
    saved_opt, saved_param_names = _saved_opt_with_marked_momentum()
    policy_sd = collections.OrderedDict()
    policy_sd["some.buffer"] = torch.zeros(3)            # a buffer — must be skipped
    for nm in saved_param_names:
        policy_sd[nm] = torch.zeros(1)                  # value irrelevant; only the KEY ORDER matters
    zp = str(tmp_path / "ckpt.zip")
    _write_fake_checkpoint_zip(zp, policy_sd, saved_opt)
    got_opt, got_names = _read_saved_optimizer_state(zp, set(saved_param_names))
    assert got_names == saved_param_names               # buffer skipped, param order preserved
    assert len(got_opt["state"]) == len(saved_opt["state"])


def test_read_saved_optimizer_state_raises_on_count_mismatch(tmp_path):
    """If the filtered saved param-name count != the saved optimizer's param count, raise so the
    caller falls back to the shape-only guard rather than remapping on an inconsistent mapping."""
    saved_opt, saved_param_names = _saved_opt_with_marked_momentum()
    policy_sd = collections.OrderedDict((nm, torch.zeros(1)) for nm in saved_param_names[:-1])  # drop one
    zp = str(tmp_path / "ckpt.zip")
    _write_fake_checkpoint_zip(zp, policy_sd, saved_opt)
    try:
        _read_saved_optimizer_state(zp, set(saved_param_names))
    except ValueError:
        return
    assert False, "expected ValueError on param-count mismatch"
