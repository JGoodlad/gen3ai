"""SLICE N at the ENV LEVEL (M5's gate; ``slice_n``): N envs on T threads in the Rust env core, both
front ends, against the production-surface ``Gen3Env`` — obs, both masks the learner reads, every
built label key, reward and ``terminated`` / ``truncated``. No allowlist.

COMMIT (routine): 4 envs, 3 / 2 threads, pool teams, the production threshold AND a low stall
threshold (the forfeit through the auto-reset path). MILESTONE (``slow``; its verdict lands in
``designs/ops/slow_tier_status.json``): N = 48, T = 8 / 5, pool + ladder + procedural at the
production threshold, and ladder at a low threshold. Teeth: a moved mask, row, label and outcome
each fail on their own key.
"""
import numpy as np
import pytest

from main.rust_core_m5 import slice_n as S

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def built():
    return S.build("selfcheck")


def _commit(built, **kw):
    t = S.TIERS["commit"]
    kw.setdefault("turn_limit", None)
    kw.setdefault("n", 4)
    n = kw.pop("n")
    return S.run_part(built, "pool", n, n_envs=t.n_envs, threads=t.threads, threads_b=t.threads_b,
                      key_base=t.key_base + 500, **kw)


def test_commit_tier_the_rust_env_at_n_equals_gen3env(built):
    res = S.run_tier("commit", built=built)
    print(S.render(res))
    assert res["ok"], [(p["divergences"], p["examples"], p["front_end_diffs"]) for p in res["parts"]]
    prod, low = res["parts"]
    # non-vacuity: several episodes per env (the AUTO-RESET path), labels and masks compared,
    # natural endings at the production threshold and stall forfeits at the low one
    assert prod["census"]["auto_resets"] >= prod["n_envs"] and prod["counts"]["envs"] == prod["n_envs"]
    assert prod["counts"]["label_compares"] == prod["counts"]["decisions"] * len(S.label_keys(S.families()))
    assert prod["counts"]["mask_compares"] == 2 * prod["counts"]["decisions"] > 0
    assert prod["counts"]["terminated"] >= 4
    assert low["counts"]["forfeits"] == low["counts"]["truncated"] == low["n_episodes"]


def test_the_mask_comparison_has_teeth(built, monkeypatch):
    """A Python mask that differs in one cell from the core's fails ``action_masks()`` at every decision."""
    from agents.training import gen3_env

    real = gen3_env.Gen3Env.action_masks

    def moved(self):
        m = np.array(real(self), copy=True)
        m[-1] = 1 - m[-1]
        return m

    monkeypatch.setattr(gen3_env.Gen3Env, "action_masks", moved)
    res = _commit(built, n=1)
    assert res["divergences"].get("action_masks()") == res["counts"]["decisions"] > 0, res["divergences"]
    assert "action_mask" not in res["divergences"]


@pytest.mark.parametrize("what", ["row", "label", "end", "p2"])
def test_the_slice_has_teeth(built, what):
    """One moved recorded cell fails the slice on its own key (and a moved row stops the episode)."""
    def mutate(recs):
        ep = recs[0]
        dec_n, row, mask, labels = ep.rows[1]
        if what == "row":
            row = row.copy()
            row.view(np.uint8)[77] ^= 1
        elif what == "label":
            labels = dict(labels)
            labels["belief_moves"] = labels["belief_moves"].copy()
            labels["belief_moves"].reshape(-1)[-1] += 1
        elif what == "end":
            r, term, trunc = ep.end
            ep.end = (r + 0.5, term, trunc)
        elif what == "p2":
            ep.p2.append(ep.p2[-1])
        ep.rows[1] = (dec_n, row, mask, labels)

    res = _commit(built, n=2, mutate=mutate)
    key = {"row": "observation", "label": "belief_moves", "end": "end", "p2": "[p2 decisions]"}[what]
    assert res["divergences"].get(key) == 1 and not res["ok"], res["divergences"]


def test_the_front_end_comparison_has_teeth():
    a = {0: S.Episode(p1=[1, 2], p2=[3], rows=[(0, np.zeros(3, np.float32), np.ones(11, np.uint8), {})], end=(1.0, 1, 0))}
    b = {0: S.Episode(p1=[1, 2], p2=[3], rows=[(0, np.zeros(3, np.float32), np.ones(11, np.uint8), {})], end=(1.0, 1, 0))}
    assert S.recordings_equal(a, b) == []
    b[0].rows[0][2][4] = 0
    assert S.recordings_equal(a, b) == ["episode 0 decision 0: a column differs"]


@pytest.mark.slow
def test_milestone_the_rust_env_at_n48_equals_gen3env(built):
    res = S.run_tier("milestone", built=built)
    print(S.render(res))
    assert res["ok"], [(p["source"], p["divergences"], p["examples"], p["front_end_diffs"]) for p in res["parts"]]
    for p in res["parts"]:
        assert p["n_envs"] == 48 and p["counts"]["envs"] == 48, p["counts"]
        assert p["census"]["auto_resets"] >= 48
