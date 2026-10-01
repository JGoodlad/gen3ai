"""The T2 fan-out read's pure parts (routine, no GPU): rows dealt round-robin over the slots, and the
derived reads (the grouped-forward upper bound, the per-slot marginal) from a synthetic table."""
from main.rust_core_m5.fanout import build_parser, derive, plan_rows


def test_rows_are_dealt_round_robin_over_the_slots():
    parts = plan_rows([7, 9, 11], 8)
    assert parts == [[0, 3, 6], [1, 4, 7], [2, 5]]
    assert sum(len(p) for p in plan_rows(list(range(18)), 40)) == 40


def test_the_derived_reads_come_from_the_table():
    ms = {"trainee48": 2.0, "opp_S1": 1.5, "opp_S4": 2.4, "both_S1": 3.0, "both_S4": 4.5}
    res = {"ms_per_flush": {k: {"mean": v} for k, v in ms.items()}}
    d = derive(res, [1, 4])
    assert d["trainee_forward_ms"] == 2.0 and d["opponent_fanout_ms_at_S4"] == 2.5
    assert d["grouped_forward_upper_bound_saving_ms"] == 1.5
    assert abs(d["marginal_ms_per_extra_slot"] - 0.3) < 1e-12


def test_the_cli_needs_its_inputs():
    a = build_parser().parse_args(["--out", "x.json", "--pool", "p", "--ckpt", "c.zip"])
    assert (a.opp_rows, a.slots, a.device) == (40, "1,2,4,8,12,18", "cuda")


def test_a_real_flush_is_split_into_the_trainee_and_the_opponent_parts_by_slot():
    import numpy as np

    from main.rust_core_m5.fanout import flush_parts

    n = 6
    cols = {"need": np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 1], [0, 0]], dtype=np.int8),
            "opp_slot": np.array([3, -1, 4, 3, -1, 4], dtype=np.int64),
            "obs": np.arange(n * 2 * 2, dtype=np.float32).reshape(n, 2, 2),
            "mask": np.ones((n, 2, 3), dtype=bool)}
    env_slot = np.array([0, 0, 0, 0, 1, 0])
    p = flush_parts(cols, env_slot)
    assert [(s, o.shape[0]) for s, o, _m in p["trainee"]] == [(0, 3), (1, 1)]
    # opponent rows: need p2 AND an in-T2 slot (env 4 is an external / in-core p2, env 5 needs nothing)
    assert [(s, o.shape[0]) for s, o, _m in p["opponent"]] == [(3, 2), (4, 1)]
    assert (p["opponent"][0][1] == cols["obs"][[0, 3], 1]).all()          # side 1 for p2
    cols["obs"][:] = -1
    assert p["trainee"][0][1].min() >= 0                                   # copied, not views


def test_the_split_is_the_full_flush_minus_its_trainee_part():
    from main.rust_core_m5.fanout import split_summary

    s = split_summary([10.0, 12.0], [4.0, 4.0], [40, 44], [18, 20], [48, 48], seed=1)
    assert abs(s["opponent_ms"]["mean"] - 7.0) < 1e-12
    assert abs(s["opponent_share_of_flush"] - 14.0 / 22.0) < 1e-12
    assert s["n_flushes"] == 2 and s["opponent_slots_mean"] == 19.0
