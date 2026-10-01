"""The ride-along step benchmark runs the learner's own step end to end (CPU, tiny batch)."""
from agents.training.ridealong_step_benchmark import run


def test_the_step_benchmark_times_the_real_step_and_scales_it_to_one_epoch():
    res = run("cpu", batch=8, k=2, warmup=1)
    assert res["step_ms_epoch0"]["median"] > 0
    assert res["minibatches_per_epoch"] == 98_304 // 8
    assert res["added_s_per_update"] == res["step_ms_epoch0"]["median"] / 1000.0 \
        * res["minibatches_per_epoch"] * res["ridealong_epochs"]
    assert "contention_at_start" in res
