"""Pins for the keyed draw (`gen3_keyed_draw_v1`): the hash is the documented chain, a row's action
does not depend on its batch, the draw is the inverse CDF (legal only, distribution-correct), the
temperature path matches ``RLPlayer``'s ``logits / T``, and the margin names the near-boundary rows."""
from __future__ import annotations

import numpy as np
import pytest

from agents.training import keyed_draw as K


def _splitmix_ref(z: int) -> int:
    m = 0xFFFFFFFFFFFFFFFF
    z = (z + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return z ^ (z >> 31)


def _key_ref(seed, stream, env, ep, dec) -> int:
    h = _splitmix_ref(seed)
    for part in (stream, env, ep, dec):
        h = _splitmix_ref(h ^ part)
    return h


def test_the_hash_is_the_documented_splitmix_chain():
    rng = np.random.default_rng(0)
    for _ in range(200):
        seed, stream, env, ep, dec = (int(x) for x in rng.integers(0, 2**31, 5))
        got = int(K.draw_keys(seed, stream, env, ep, dec))
        assert got == _key_ref(seed, stream, env, ep, dec)
    # the golden value of the Rust / Lane E twin (`opponents::stream_seed`'s splitmix64)
    from agents.training.rust_env_opponents import _splitmix64

    assert int(K.splitmix64(np.uint64(12345))) == _splitmix64(12345)


def test_uniforms_lie_in_the_unit_interval_and_differ_per_key():
    u = K.keyed_uniforms(7, K.STREAM_TRAINEE, np.arange(48), 3, np.arange(48))
    assert ((u >= 0) & (u < 1)).all()
    assert len(set(u.tolist())) == 48
    # the streams are independent names for the same (env, episode, decision)
    v = K.keyed_uniforms(7, K.STREAM_OPPONENT, np.arange(48), 3, np.arange(48))
    assert not np.array_equal(u, v)


def _random_logp(rng, b, a=11):
    raw = rng.normal(size=(b, a)).astype(np.float32) * 2.0
    mask = rng.random((b, a)) < 0.6
    mask[np.arange(b), rng.integers(0, a, b)] = True
    lp = raw - np.log(np.exp(raw.astype(np.float64)).sum(1, keepdims=True)).astype(np.float32)
    return np.where(mask, lp, -np.inf).astype(np.float32), mask


def test_a_rows_action_and_margin_do_not_depend_on_its_batch():
    rng = np.random.default_rng(1)
    logp, _ = _random_logp(rng, 257)
    u = rng.random(257)
    a_all, m_all = K.keyed_actions(logp, u)
    for i in range(257):
        a1, m1 = K.keyed_actions(logp[i:i + 1], u[i:i + 1])
        assert a1[0] == a_all[i] and m1[0] == m_all[i], i
    # and inside differently-sized, differently-offset batches
    for lo, hi in ((3, 40), (100, 101), (0, 256), (129, 257)):
        a_b, m_b = K.keyed_actions(logp[lo:hi], u[lo:hi])
        assert np.array_equal(a_b, a_all[lo:hi]) and np.array_equal(m_b, m_all[lo:hi])


def test_the_draw_is_always_legal_and_is_the_inverse_cdf():
    rng = np.random.default_rng(2)
    logp, mask = _random_logp(rng, 2000)
    u = rng.random(2000)
    a, _ = K.keyed_actions(logp, u)
    assert mask[np.arange(2000), a].all()
    p = np.where(mask, np.exp(logp.astype(np.float64)), 0.0)
    c = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
    ref = (c <= u[:, None]).sum(1)
    near = np.abs(c[np.arange(2000), np.minimum(ref, 10)] - u) < 1e-9
    assert (a == np.minimum(ref, 10))[~near].all()
    # u -> 0 picks the first legal action, u -> 1 the last
    first = np.argmax(mask, axis=1)
    last = 10 - np.argmax(mask[:, ::-1], axis=1)
    assert np.array_equal(K.keyed_actions(logp, np.zeros(2000))[0], first)
    assert np.array_equal(K.keyed_actions(logp, np.full(2000, 1 - 2**-53))[0], last)


def test_the_draw_follows_the_policy_distribution():
    with np.errstate(divide="ignore"):
        logp = np.log(np.array([[0.5, 0.0, 0.2, 0.3] + [0.0] * 7], dtype=np.float64))
    logp = np.where(np.isfinite(logp), logp, -np.inf).astype(np.float32)
    n = 60_000
    keys = K.draw_keys(99, 0, 0, 0, np.arange(n))
    a, _ = K.keyed_actions(np.repeat(logp, n, axis=0), K.uniforms(keys))
    freq = np.bincount(a, minlength=11) / n
    assert abs(freq[0] - 0.5) < 0.01 and abs(freq[2] - 0.2) < 0.01 and abs(freq[3] - 0.3) < 0.01
    assert freq[1] == freq[4:].sum() == 0


def test_temperature_divides_the_logits_as_rlplayer_does():
    rng = np.random.default_rng(4)
    logp, mask = _random_logp(rng, 500)
    u = rng.random(500)
    a_t, _ = K.keyed_actions(logp, u, 0.5)
    # the reference: softmax(logp / T) over the legal entries, then the same inverse CDF
    x = np.where(mask, logp.astype(np.float64) / 0.5, -np.inf)
    p = np.where(mask, np.exp(x - np.max(x, 1, keepdims=True)), 0.0)
    c = np.cumsum(p, 1)
    ref = np.minimum((c <= (u * c[:, -1])[:, None]).sum(1), 10 - np.argmax(mask[:, ::-1], 1))
    assert np.array_equal(a_t, ref)
    # per-row temperatures
    temps = np.where(np.arange(500) % 2 == 0, 1.0, 0.5)
    a_mix, _ = K.keyed_actions(logp, u, temps)
    a_1, _ = K.keyed_actions(logp, u, 1.0)
    assert np.array_equal(a_mix[::2], a_1[::2]) and np.array_equal(a_mix[1::2], a_t[1::2])


def test_the_margin_names_the_rows_a_last_bit_change_can_flip():
    with np.errstate(divide="ignore"):
        logp = np.log(np.array([[0.25, 0.25, 0.5] + [0.0] * 8]))
    logp = np.where(np.isfinite(logp), logp, -np.inf).astype(np.float32)
    a, m = K.keyed_actions(logp, np.array([0.25 + 1e-12]))
    assert a[0] == 1 and m[0] < 1e-9
    a, m = K.keyed_actions(logp, np.array([0.4]))
    assert a[0] == 1 and m[0] == pytest.approx(0.1, abs=1e-6)


def test_refusals():
    lp = np.full((1, 11), -np.inf, dtype=np.float32)
    with pytest.raises(ValueError, match="no legal"):
        K.keyed_actions(lp, np.array([0.5]))
    ok = np.zeros((1, 11), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        K.keyed_actions(ok, np.array([1.0]))
    with pytest.raises(ValueError, match="temperature"):
        K.keyed_actions(ok, np.array([0.5]), 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        K.draw_keys(1, 0, -1, 0, 0)
