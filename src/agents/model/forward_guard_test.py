"""`gen3_extractor_forward_guard_v1` — one extractor, two threads, no corruption.

🚨 **The defect this stands for is a CRASH only by luck.** Two concurrent forwards through one
`Gen3FeaturesExtractor` raise when their batch sizes differ (that is the
`ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)` that killed the 2026-09-22
playoff repro) and corrupt each other SILENTLY when they do not — and in the search battery the
mirror opponent's live decision and a playoff rollout player's decision are both B=1. So the
unguarded case is asserted on the crash (fast, deterministic enough to gate) and the guarded case
is asserted on the OUTPUT BYTES against a single-threaded control, which is the claim that
actually matters.
"""
import threading

import gymnasium as gym
import numpy as np
import pytest
import torch


def _extractor():
    from agents.model.damage_op_test import _make_layout
    from agents.model.features_extractor import Gen3FeaturesExtractor
    layout = _make_layout()
    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (layout["total_dim"],), np.float32),
        "action_mask": gym.spaces.Box(0, 1, (11,), np.float32)})
    torch.manual_seed(0)
    fe = Gen3FeaturesExtractor(space, layout=layout, move_belief_mode="revealed", damage_op=True,
                               attend_unrevealed_opponents=True, value_threat_inject=True)
    fe.eval()
    return fe, layout


def _obs(layout, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, layout["total_dim"], generator=g),
            "action_mask": torch.ones(batch, 11)}


def _hammer(fe, obs_a, obs_b, rounds):
    """Run the two observations in two threads; return (errors, a_outputs, b_outputs)."""
    errs: list = []
    outs: dict = {0: [], 1: []}

    def run(which, obs):
        for _ in range(rounds):
            try:
                with torch.no_grad():
                    pi, _vf = fe(obs)
                outs[which].append(pi.clone())
            except Exception as e:                           # noqa: BLE001
                errs.append(f"{type(e).__name__}: {e}")
                return

    ts = [threading.Thread(target=run, args=(0, obs_a)), threading.Thread(target=run, args=(1, obs_b))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errs, outs[0], outs[1]


def test_an_UNGUARDED_extractor_is_corrupted_by_a_second_thread():
    """The measurement that made the guard necessary. 2,400 interleaved forwards produced 1,063
    failures in seven classes; this asserts only that the unguarded case fails AT ALL, because
    the rate is a property of the box and the CLAIM is that the race exists."""
    from agents.model.forward_guard import forward_guard_for

    fe, layout = _extractor()
    assert forward_guard_for(fe) is None, "no guard is the DEFAULT — training must not pay for one"
    torch.set_num_threads(1)
    errs, _, _ = _hammer(fe, _obs(layout, 1, 1), _obs(layout, 9, 2), rounds=400)
    if not errs:
        pytest.skip("the race did not schedule on this box in 800 forwards — see the guarded "
                    "test below, which is the one that carries the contract")
    assert any("ValueThreatInject" in e or "out of bounds" in e or "reduced rows" in e
               or "Sizes of tensors" in e for e in errs), errs[:5]


def test_a_GUARDED_extractor_is_byte_identical_to_the_SINGLE_THREADED_control():
    """🚨 FAILS ON REVERT of the guard. The contract is not "it does not crash" — a corrupted
    forward that happens to have the right shapes returns WRONG NUMBERS, so the assertion is on
    the output bytes against the same forwards run alone."""
    from agents.model.forward_guard import forward_guard_for, install_forward_guard

    fe, layout = _extractor()
    obs_a, obs_b = _obs(layout, 1, 1), _obs(layout, 9, 2)
    with torch.no_grad():
        want_a = fe(obs_a)[0].clone()
        want_b = fe(obs_b)[0].clone()

    assert install_forward_guard(fe) is forward_guard_for(fe)
    assert install_forward_guard(fe) is forward_guard_for(fe), "installation is idempotent"
    torch.set_num_threads(1)
    errs, got_a, got_b = _hammer(fe, obs_a, obs_b, rounds=60)
    assert not errs, errs[:5]
    assert got_a and got_b
    for t in got_a:
        assert torch.equal(t, want_a), "a guarded B=1 forward drifted from its own control"
    for t in got_b:
        assert torch.equal(t, want_b), "a guarded B=9 forward drifted from its own control"


def test_the_guard_is_REENTRANT_because_the_atomic_unit_is_forward_PLUS_the_stash_reads():
    """`RLPlayer` reads the α publication and the win-prob logits AFTER `forward` returns, so a
    caller holds the guard across both and `forward`'s own acquire inside it must not deadlock."""
    from agents.model.forward_guard import install_forward_guard

    fe, layout = _extractor()
    guard = install_forward_guard(fe)
    with guard:                       # the caller's span
        with torch.no_grad():
            fe(_obs(layout, 2, 3))    # forward acquires the same lock again
    assert guard.acquire(blocking=False), "the guard was not released"
    guard.release()


def test_the_guard_is_not_an_ATTRIBUTE_on_the_module():
    """A `threading.RLock` set on an `nn.Module` makes `copy.deepcopy(policy)` — which SB3 does —
    raise a TypeError far from anything that asked for a lock. The registry is weak-keyed and
    external precisely so that cannot happen."""
    import copy

    from agents.model.forward_guard import install_forward_guard

    fe, _layout = _extractor()
    install_forward_guard(fe)
    copy.deepcopy(fe)                 # must not raise
    assert not any("lock" in k.lower() for k in vars(fe))
