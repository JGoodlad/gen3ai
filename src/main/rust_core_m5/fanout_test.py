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
