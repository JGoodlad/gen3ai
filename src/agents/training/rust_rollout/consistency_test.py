"""Pins for K9(b) BEHAVIOUR-POLICY CONSISTENCY and the STALENESS probe (M5 Lane G).

On a buffer whose stored behaviour log-probs are the policy's own (real obs rows, the committed
parity fixture): the probe passes (|Δ| at float rounding on CPU); a stored log-prob moved by 1e-3 on ONE current
row FAILS it (`fatal` raises, `warn` prints); rows of older versions are NOT held to the bar — they are
bucketed by age with their ratio / clip fraction / KL recorded; the row choice takes current rows first
and every present age; `off` does nothing."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch as th

from agents.training.rust_rollout import consistency as K


def _model(n_steps=16, n_envs=4):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, layout, pk = _production_policy_kwargs()
    dim = layout["total_dim"]
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                             "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kw):
            return {"observation": np.zeros(dim, np.float32), "action_mask": np.ones(11, np.int8)}, {}

        def step(self, a):
            return self.reset()[0], 0.0, False, False, {}

    th.manual_seed(0)
    m = MaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E] * n_envs), n_steps=n_steps, batch_size=32,
                    policy_kwargs=pk, verbose=0, device="cpu", seed=0)
    perturb_(m.policy, seed=77, scale=PERTURB_SCALE)
    obs, mask = load_parity_rows(dim)
    n = n_steps * n_envs
    idx = np.arange(n) % len(obs)
    buf = m.rollout_buffer
    buf.reset()
    buf.observations["observation"][...] = obs[idx].reshape(n_steps, n_envs, dim)
    buf.observations["action_mask"][...] = mask[idx].reshape(n_steps, n_envs, 11)
    buf.action_masks[...] = mask[idx].reshape(n_steps, n_envs, 11).astype(np.float32)
    rng = np.random.default_rng(1)
    acts = np.array([rng.choice(np.flatnonzero(mask[i])) for i in idx])
    buf.actions[...] = acts.reshape(n_steps, n_envs, 1)
    m.policy.set_training_mode(False)
    with th.no_grad():
        _v, lp, _e = m.policy.evaluate_actions(
            {"observation": th.as_tensor(obs[idx]), "action_mask": th.as_tensor(mask[idx])},
            th.as_tensor(acts), action_masks=th.as_tensor(mask[idx]))
    buf.log_probs[...] = lp.numpy().reshape(n_steps, n_envs)
    buf.full = True
    m.ep_info_buffer = deque(maxlen=10)
    m._current_progress_remaining = 1.0
    m.num_timesteps = 1234
    return m


def test_off_does_nothing():
    m = _model()
    m.behaviour_check = "off"
    assert K.behaviour_probe(m) is None


def test_current_rows_match_on_cpu_to_float_rounding():
    m = _model()
    m.behaviour_check = "fatal"
    out = K.behaviour_probe(m)
    # the stored log-probs came from a 64-row batch, the probe runs 32: the CPU matmul's batch-size
    # rounding (~6e-7) is the whole difference, 170x under the bar
    assert out["behaviour/rows_current"] == 32 and out["behaviour/max_abs_dlogp_current"] < 5e-6
    assert out["staleness/probe_age_0_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["staleness/probe_age_0_clip_frac"] == 0.0


def test_one_current_row_off_by_1e_3_fails_fatal_and_warns_under_warn(capsys):
    m = _model()
    m.behaviour_check = "fatal"
    m.rollout_buffer.log_probs += 1e-3          # every stored behaviour log-prob moved: the probe sees it
    with pytest.raises(K.BehaviourMismatch, match="CURRENT policy"):
        K.behaviour_probe(m)
    m.behaviour_check = "warn"
    out = K.behaviour_probe(m)
    assert out["behaviour/max_abs_dlogp_current"] == pytest.approx(1e-3, rel=1e-3)
    assert "BEHAVIOUR-POLICY MISMATCH" in capsys.readouterr().out


def test_older_rows_are_bucketed_by_age_not_held_to_the_bar():
    m = _model()
    m.behaviour_check = "fatal"
    versions = np.full((16, 4), 5, dtype=np.int64)
    versions[8:, :] = 3                           # half the buffer played two versions ago
    m._rust_row_versions, m._rust_version = versions, 5
    m.rollout_buffer.log_probs[8:, :] -= 0.5      # those rows' behaviour policy differed
    out = K.behaviour_probe(m)
    assert out["behaviour/max_abs_dlogp_current"] < 5e-6
    assert out["staleness/probe_age_0_rows"] == 16 and out["staleness/probe_age_2_rows"] == 16
    assert out["staleness/probe_age_2_ratio_mean"] == pytest.approx(np.exp(0.5), rel=1e-5)
    assert out["staleness/probe_age_2_clip_frac"] == 1.0
    assert out["staleness/probe_age_2_approx_kl"] == pytest.approx(np.exp(0.5) - 1 - 0.5, rel=1e-5)


def test_the_row_choice_takes_current_rows_first_and_every_present_age():
    rng = np.random.default_rng(0)
    ages = np.array([0] * 10 + [1] * 50 + [4] * 50 + [20] * 3)
    pick = K.choose_rows(ages, 32, rng)
    assert len(pick) == 32 and len(set(pick.tolist())) == 32
    got = ages[pick]
    assert (got == 0).sum() == 10 and {1, 4, 20} <= set(got.tolist())
    assert K.choose_rows(np.zeros(5, int), 32, rng).tolist() == [0, 1, 2, 3, 4]


def test_the_probe_reads_the_precision_keyed_gate():
    """Lane G's probe and the python-core in-loop gate share ONE table (K9, M5 Lane K): fp32 = max < 1e-4;
    TF32 (`--matmul-precision high`) = the current rows' p99 < 3.6e-3 AND max < 0.071. A uniform shift
    between the fp32 bar and the TF32 p99 bar FAILS at fp32 and PASSES under TF32; above the p99 bar it
    fails under both. (The localized half — p99 clean, max fires — is `learner_gates_test`'s, on a
    production-size micro-batch; this probe's 32 rows make its p99 ~ its max.)"""
    assert K.BEHAVIOUR_GATES["highest"] == (K.GateCondition("max", 1e-4, 4, K.FP32_ROW_RATE_UPPER, K.FP32_TIE_EPS),)
    c99, cmax = K.BEHAVIOUR_GATES["high"]
    (p99_name, p99_bar, p99_k), (max_name, max_bar, max_k) = c99[:3], cmax[:3]
    assert c99.row_rate == cmax.row_rate == c99.tie_eps == cmax.tie_eps == 0.0     # the tie rule is fp32's
    assert (p99_name, max_name, p99_k, max_k) == ("p99", "max", 1, K.TF32_MAX_PERSISTENCE)
    assert 1e-4 < p99_bar < max_bar
    prev = th.get_float32_matmul_precision()
    try:
        for shift, precision, fails in ((0.5 * p99_bar, "highest", True), (0.5 * p99_bar, "high", False),
                                        (2.0 * p99_bar, "high", True)):
            th.set_float32_matmul_precision(precision)
            m = _model()
            m.behaviour_check = "fatal"
            m.rollout_buffer.log_probs += shift
            if fails:
                with pytest.raises(K.BehaviourMismatch, match=precision):
                    K.behaviour_probe(m)
            else:
                assert K.behaviour_probe(m)["behaviour/bar_p99"] == p99_bar
    finally:
        th.set_float32_matmul_precision(prev)


def _with_provenance(m):
    """The rust fill's per-row provenance for `_model`'s buffer: the stored distribution is the
    policy's own (so stored == recomputed on every untouched row)."""
    buf = m.rollout_buffer
    n_steps, n_envs = buf.log_probs.shape
    n = n_steps * n_envs
    obs = {k: th.as_tensor(v.reshape(n, *v.shape[2:])) for k, v in buf.observations.items()}
    m.policy.set_training_mode(False)
    with th.no_grad():
        m.policy.evaluate_actions(obs, th.as_tensor(buf.actions.reshape(-1)).long(),
                                  action_masks=th.as_tensor(buf.action_masks.reshape(n, -1)))
    full = K._stashed_logp(m.policy).astype(np.float32).reshape(n_steps, n_envs, -1)
    grid = np.arange(n).reshape(n_steps, n_envs)
    m._rust_row_provenance = {"env": grid % n_envs, "episode": 100 + grid // 7, "dec_n": grid % 7,
                              "version": np.zeros_like(grid), "slot": np.full_like(grid, 2),
                              "u": np.full(grid.shape, 0.25), "margin": np.full(grid.shape, 0.5),
                              "logp_all": full}
    return m


def test_a_single_row_violation_dumps_both_distributions_its_provenance_the_scan_and_the_artifacts(tmp_path, capsys):
    """The A2 shape (2026-10-01): ONE row of the probe off by 0.0389, every other row exact. The dump must
    let it be root-caused offline: both log-probs and both full distributions (entropy, margin, p), the
    row's env / episode / decision index / slot, the full-buffer scan (one row or many?), the row's obs
    and the weights that played it."""
    import json

    m = _with_provenance(_model())
    m.behaviour_check = "warn"
    m.behaviour_dump_dir = str(tmp_path)
    buf = m.rollout_buffer
    rng = np.random.default_rng([int(m.seed or 0), int(m.num_timesteps)])
    f = int(K.choose_rows(np.zeros(buf.log_probs.shape, np.int64), m.batch_size, rng)[0])
    t, e = divmod(f, buf.n_envs)
    buf.log_probs[t, e] += 0.0389
    out = K.behaviour_probe(m)
    assert out["behaviour/scan_rows_over_bar"] == 1.0
    rec = json.loads((tmp_path / K.VIOLATION_DUMP).read_text().splitlines()[-1])
    r = rec["rows"][0]
    assert (r["buffer_t"], r["buffer_e"]) == (t, e) and r["abs_dlogp"] == pytest.approx(0.0389, rel=1e-3)
    assert r["logp_stored"] - r["logp_recomputed"] == pytest.approx(0.0389, rel=1e-3)
    assert len(r["dist_stored"]) == 11 and len(r["dist_recomputed"]) == 11
    legal = [c == "1" for c in r["mask"]]                 # illegal entries are null on BOTH sides
    assert [v is not None for v in r["dist_recomputed"]] == legal == [v is not None for v in r["dist_stored"]]
    for side in ("stored", "recomputed"):
        assert r[f"entropy_{side}"] >= 0 and r[f"margin_{side}"] >= 0 and 0 < r[f"p_action_{side}"] <= 1
    assert (r["env"], r["episode"], r["dec_n"], r["slot"]) == (f % buf.n_envs, 100 + f // 7, f % 7, 2)
    assert rec["scan"]["over_bar"] == 1 and rec["scan"]["rows"] == buf.log_probs.size
    assert rec["scan"]["worst"][0]["abs_dlogp"] == pytest.approx(0.0389, rel=1e-3)
    z = np.load(tmp_path / rec["artifacts"]["rows_npz"])
    assert z["obs__observation"].shape[0] == 1
    assert np.array_equal(z["obs__observation"][0], buf.observations["observation"][t, e])
    pol = th.load(tmp_path / rec["artifacts"]["policy_pt"])
    assert set(pol) == set(m.policy.state_dict())
    assert "env" in capsys.readouterr().out


def test_a_clean_update_builds_no_details_unless_the_driver_asks_for_the_scan():
    m = _with_provenance(_model())
    m.behaviour_check = "fatal"
    out = K.behaviour_probe(m)
    assert "behaviour/scan_rows_over_bar" not in out
    m.behaviour_scan_all = True
    out = K.behaviour_probe(m)
    assert out["behaviour/scan_rows_over_bar"] == 0.0 and out["behaviour/scan_max_abs_dlogp"] < 5e-6


# ------------------------------------------------------------- the fp32 TIE RULE (gen3_behaviour_tie_rule_v1)
def _probe_rows_with_gaps(m):
    from agents.training.rust_rollout.tie_margins import selection_gaps

    b = m.rollout_buffer
    rng = np.random.default_rng([int(m.seed or 0), int(m.num_timesteps)])
    f = K.choose_rows(np.zeros(b.log_probs.shape, np.int64), m.batch_size, rng)
    t, e = f // b.n_envs, f % b.n_envs
    g, _s = selection_gaps(m.policy, {k: v[t, e] for k, v in b.observations.items()}, b.actions[t, e],
                           b.action_masks[t, e], m.device)
    return t, e, g


def test_the_fp32_rule_numbers_are_derived_from_the_banked_measurement():
    """k, the row rate and epsilon are re-derived from `TAIL_MEASUREMENT` by the declared criteria — a
    changed constant or a changed measurement FAILS here."""
    import json

    from utils.paths import repo_path

    r = json.loads(open(repo_path(*K.TAIL_MEASUREMENT.split("/"))).read())
    n_events, n_rows = r["rate"]["events"], r["rate"]["rows"]
    assert (n_events, n_rows) == (4, 3_670_016)
    upper = r["rate"]["poisson_upper95_events"] / n_rows                 # 9.154 / 3,670,016
    assert upper <= K.FP32_ROW_RATE_UPPER < 1.01 * upper + 1e-7
    p = K.per_update_rate(K.FP32_ROW_RATE_UPPER, 2048)
    k = next(k for k in range(1, 10) if K.UPDATES_HORIZON * p ** k <= K.FALSE_FATAL_TARGET)
    assert k == K.FP32_MAX_PERSISTENCE == 4
    flipped = max(v["fp64_gap"] for v in r["violators"])
    assert K.FP32_TIE_EPS >= 50 * flipped and all(v["fp32_gap"] <= K.FP32_TIE_EPS for v in r["violators"])
    assert r["tie_rate"]["frac_at_or_below"]["1e-05"] < 0.2               # (iii) keeps its teeth
    assert K.count_limit(K.FP32_ROW_RATE_UPPER, 98_304, 1024) == 7       # production N = 48 x 2,048


def test_a_single_violating_row_AT_a_tie_warns_and_FATALs_only_on_its_4th_consecutive_update(capsys):
    m = _with_provenance(_model())
    m.behaviour_check = "fatal"
    t, e, g = _probe_rows_with_gaps(m)
    tied = np.flatnonzero((g > 0) & (g <= K.FP32_TIE_EPS))
    assert tied.size, "the fixture's probe must hold a row at a (non-exact) selection tie"
    i = int(tied[0])
    m.rollout_buffer.log_probs[t[i], e[i]] += 0.0389          # A2's jump, on a row that sits at a tie
    for n in range(1, 4):
        out = K.behaviour_probe(m)
        assert out["behaviour/streak_max"] == float(n) and out["behaviour/violations_total_max"] == float(n)
    assert "NOT fatal YET" in capsys.readouterr().out
    with pytest.raises(K.BehaviourMismatch, match=r"\(i\) max violated on 4 consecutive"):
        K.behaviour_probe(m)


def test_a_single_violating_row_at_NO_tie_is_a_FAULT_and_FATAL_at_once():
    m = _with_provenance(_model())
    m.behaviour_check = "fatal"
    t, e, g = _probe_rows_with_gaps(m)
    i = int(np.argmax(g))
    assert g[i] > 100 * K.FP32_TIE_EPS
    m.rollout_buffer.log_probs[t[i], e[i]] += 0.0389
    with pytest.raises(K.BehaviourMismatch, match=r"\(iii\) 1 violating row\(s\) sit at NO selection tie"):
        K.behaviour_probe(m)


def test_many_rows_over_the_bar_in_the_full_buffer_scan_are_a_FAULT_and_FATAL_at_once():
    m = _with_provenance(_model())
    m.behaviour_check = "fatal"
    t, e, g = _probe_rows_with_gaps(m)
    tied = np.flatnonzero((g > 0) & (g <= K.FP32_TIE_EPS))
    assert tied.size
    lp = m.rollout_buffer.log_probs
    lp[t[tied[0]], e[tied[0]]] += 0.0389                      # the probe's one row sits at a tie ...
    chosen = set(zip(t.tolist(), e.tolist()))
    others = [(a, b) for a in range(lp.shape[0]) for b in range(lp.shape[1]) if (a, b) not in chosen][:8]
    for a, b in others:                                       # ... but 8 rows the probe did not pick moved too
        lp[a, b] += 1e-3
    assert K.count_limit(K.FP32_ROW_RATE_UPPER, lp.size, m.batch_size) <= 9
    with pytest.raises(K.BehaviourMismatch, match=r"\(ii\) 9 of 64 current rows over the bar"):
        K.behaviour_probe(m)
