"""SLICE N (``gen3_core_parity_env_v1``, program M6): the training env under ``--obs-source core``
against the same env under ``--obs-source python``, two ``Gen3Env``s in LOCKSTEP on the same seed,
teams, opponent RNG and trainee actions — every obs key (bytes, incl. every training-only label
key the PRODUCTION config emits), the reward, ``terminated`` and ``truncated`` equal at reset and
every step. **No allowlist.**

* COMMIT (routine gate): a handful of pool episodes, seeded-random trainee.
* MILESTONE (``slow``; its verdict lands in ``designs/ops/slow_tier_status.json``): pool
  seeded-random, the ``production`` policy, ladder full-tier and procedural episodes.

The cutover stress runs the same harness (``envs.run_envn``) at the registered counts.
"""
import pytest

from main.rust_core_cutover import envs as E
from main.rust_core_cutover import plan as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _stream(name, battles, per_unit, **params):
    return PL.Stream(name, "envn", battles, per_unit, "test", dict(params))


def _assert_clean(row, min_episodes, min_core):
    """No difference at all — no allowlist, no named class. (Finding F1, the live env's PHANTOM
    decision on a step the trainee was not asked to move, was FIXED on the Python side:
    gen3_no_phantom_decision_v1; its named class is retired.)"""
    print({k: row[k] for k in ("battles", "steps", "phantom_steps", "core_obs_counts", "label_keys")})
    assert not row["errors"], row["errors"]
    n = row["totals"]["N"]
    assert not n["divergences"], (n["divergences"], n["examples"])
    assert row["phantom_steps"] > 0, "no phantom step exercised — the F1 regression would go unseen"
    assert row["battles"] >= min_episodes
    assert row["core_obs_counts"]["core"]["core"] >= min_core, row["core_obs_counts"]
    assert row["core_obs_counts"]["python"]["core"] == 0
    # the production surface's training-only label keys are all compared (ARCHITECTURE.md §7)
    for k in ("belief_moves", "win_target", "opp_class", "item_label", "hp_type_label"):
        assert any(key.startswith(k) for key in row["label_keys"]), (k, row["label_keys"])


def test_commit_tier_the_core_obs_env_equals_the_python_env(tmp_path):
    row = E.run_envn(_stream("envn_commit", 6, 6, source="pool", policy=False, key_base=48_000), 0, tmp_path)
    _assert_clean(row, min_episodes=6, min_core=150)


def test_the_env_slice_has_teeth(tmp_path, monkeypatch):
    """A core row that differs in ONE cell from the Python encoder's must fail slice N."""
    from agents.battle import core_obs

    real = core_obs.frame_for_decision

    def perturbed(battle, raw, n, mask):
        f = real(battle, raw, n, mask)
        row = f.row.copy()
        row[5] += 1.0
        return core_obs.CoreObsFrame(f.tag, f.side, f.n, row, f.mask, f.turn, f.rqid)

    monkeypatch.setattr(core_obs, "frame_for_decision", perturbed)
    row = E.run_envn(_stream("envn_teeth", 1, 1, source="pool", policy=False, key_base=48_100), 0, tmp_path)
    assert row["totals"]["N"]["divergences"], "a changed core row went undetected"
    assert any(k.startswith("observation") for k in row["totals"]["N"]["divergences"])


@pytest.mark.slow
@pytest.mark.parametrize("source,policy,n", [("pool", False, 60), ("pool", True, 15),
                                             ("ladder", False, 30), ("procedural", False, 20)],
                         ids=["pool_random", "pool_policy", "ladder_full", "procedural"])
def test_milestone_env_slice(tmp_path, source, policy, n):
    row = E.run_envn(_stream(f"envn_ms_{source}{'_p' if policy else ''}", n, n, source=source,
                             policy=policy, key_base=49_000 + 100 * ["pool", "ladder", "procedural"].index(source)
                             + (50 if policy else 0)), 0, tmp_path)
    _assert_clean(row, min_episodes=n, min_core=20 * n)
