"""The ride-along step benchmark runs the learner's own step end to end (CPU, tiny batch), per
configuration, with every optimizer acquired before the timing."""
from agents.training.ridealong_step_benchmark import run


def test_the_step_benchmark_times_every_config_and_scales_a_pass_to_one_epoch(monkeypatch):
    import agents.training.ridealong_step_benchmark as B

    monkeypatch.setattr(B, "ROLLOUT_ROWS", 16)        # 2 minibatches of 8: a whole pass, fast
    res = run("cpu", batch=8, k=2, warmup=1, configs=("core", "+fast", "+all"), passes=1)
    assert res["minibatches_per_epoch"] == 2
    for cfg in ("core", "+fast", "+all"):
        c = res["configs"][cfg]
        assert c["step_ms"]["median"] > 0 and c["added_s_per_update"] > 0
    assert set(res["per_variant_minus_core"]) == {"+fast", "+all"}
    assert "contention_at_start" in res
