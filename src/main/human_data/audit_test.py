"""The Kakuna-family audit's registered rules on planted data (no Metamon import, no archive read)."""
from __future__ import annotations

import numpy as np
import pytest

from main.human_data import audit as A


def test_sample_is_pure_and_order_free():
    bk = list(range(1000, 1040))
    s1 = A.sample_games(123456789012, bk)
    s2 = A.sample_games(123456789012, list(reversed(bk)) + bk[:5])      # order and duplicates do not matter
    assert s1 == s2 and len(s1) == A.N_SAMPLE and s1 == sorted(s1) and len(set(s1)) == A.N_SAMPLE
    assert set(s1) <= set(bk)
    assert A.sample_games(123456789012 + 1, bk) != s1                   # keyed per account
    assert A.sample_games(123456789012 + 2 ** 32, bk) == s1             # acct % 2**32 is the key


def test_sample_takes_everything_at_or_below_n():
    assert A.sample_games(7, [5, 3, 9]) == [3, 5, 9]
    assert A.sample_games(7, range(A.N_SAMPLE)) == list(range(A.N_SAMPLE))


def test_sample_matches_the_registered_formula():
    bk = np.arange(50, dtype=np.int64) * 3
    rng = np.random.default_rng([A.SEED, 99 % 2 ** 32])
    want = sorted(int(x) for x in bk[rng.choice(len(bk), size=A.N_SAMPLE, replace=False)])
    assert A.sample_games(99, bk.tolist()) == want


def test_chat_speaker_rule():
    log = "\n".join([
        "|player|p1|Some Name|1|1500",
        "|c|☆Some Name|hi",              # rank prefix stripped, toid equal
        "|c| some_name|gl",              # a space rank, then toid
        "|c|+SomeName|hf",               # voice rank
        "|c|Other|gg",                   # another speaker
        "|c:|1779151468|@Some Name|wp",  # the timestamped form
        "|raw|Some Name said",           # not a chat line
        "|c|Some Name2|no",              # a different account
    ])
    assert A.chat_lines(log, "Some Name") == 4
    assert A.chat_lines(log, "Other") == 1
    assert A.chat_lines(log, "") == 0


def test_chat_strips_only_one_non_alnum_prefix():
    assert A.chat_lines("|c|aSomeName|x", "SomeName") == 0             # an alnum first char is part of the name
    assert A.chat_lines("|c|☆★SomeName|x", "SomeName") == 1             # toid removes the rest anyway


def test_reference_falls_back_below_100():
    few = {a: (1.0 if a < 99 else 0.5) for a in range(300)}
    assert A.reference_mode(few) == ("all_scored", [])
    enough = {a: (1.0 if a < 100 else 0.9) for a in range(300)}
    mode, members = A.reference_mode(enough)
    assert mode == "chat" and members == list(range(100))


def test_account_score_is_decision_weighted_and_needs_five():
    g = {"k_n": 10, "k_ll": -10.0, "k_hits": 5, "p_n": 20, "p_ll": -40.0, "h_ll": -30.0}
    big = {"k_n": 30, "k_ll": -90.0, "k_hits": 15, "p_n": 20, "p_ll": -20.0, "h_ll": -20.0}
    assert A.account_score([g] * 4) is None
    s = A.account_score([g] * 4 + [big])
    assert s["ll_kakuna"] == pytest.approx((-40.0 - 90.0) / 70)
    assert s["ll_pstar"] == pytest.approx((-160.0 - 20.0) / 100)
    assert s["S"] == pytest.approx(s["ll_kakuna"] - s["ll_pstar"])
    assert s["S_HOM"] == pytest.approx(s["ll_kakuna"] - (-140.0) / 100)
    assert s["scored_games"] == 5 and s["kakuna_argmax"] == pytest.approx(35 / 70)


def test_flag_is_strictly_above_the_reference_p99():
    ref = list(range(200))
    scores = {a: float(a) for a in ref}
    scores.update({1000: 500.0, 1001: 197.0})
    p99, flagged, boundary = A.flag_accounts(scores, ref)
    assert p99 == pytest.approx(np.percentile(np.arange(200, dtype=float), 99))
    assert flagged == sorted(a for a, s in scores.items() if s > p99)
    assert 1000 in flagged and 1001 not in flagged and boundary == []


def test_flag_boundary_account_is_reported_not_flagged():
    ref = list(range(101))                          # p99 of 0..100 is exactly 99.0
    scores = {a: float(a) for a in ref}
    scores[555] = 99.0 + 1e-12
    p99, flagged, boundary = A.flag_accounts(scores, ref)
    assert p99 == 99.0
    assert 555 in boundary and 99 in boundary
    assert flagged == [100]


def test_drop_threshold_tie_does_not_fire():
    assert not A.drop_fires(100, 10_000)            # exactly 1.0 %
    assert A.drop_fires(101, 10_000)
    assert not A.drop_fires(0, 10_000)
    assert not A.drop_fires(1_342_177, 134_217_700)  # exactly 1.0 % at rowset scale, no float rounding
    assert A.drop_fires(1_342_178, 134_217_700)


def test_power_rule():
    ref = [-2.0 + 0.001 * i for i in range(500)]
    p = A.power([-0.5, -0.6, -0.7], ref)
    assert p["verdict"] == "POWER" and p["pseudo_above_p99"] == 3
    q = A.power([-2.5, -0.1, -2.4], ref)
    assert q["verdict"] == "NO-POWER"
    assert A.power([], ref)["verdict"] == "NO-POWER"


def test_amendment_detection():
    assert A.has_amendment("# X\n\n## Amendment 2 — the audit\nbody")
    assert not A.has_amendment("# X\n\n## Amendment 1\nsee ## Amendment 2 later")


def test_capture_log_takes_the_public_view_of_each_side():
    cap = {"tag": "t", "chunks": [
        ["p1", "|t:|<NORMALIZED>\n|player|p1|Gen3AIAnchor1||"],
        ["p2", "|t:|<NORMALIZED>\n|player|p1|Gen3AIAnchor1||"],
        ["p1", '|request|{"x":1}'],
        ["p2", '|request|{"x":2}'],
        ["p1", "|error|[Unavailable choice]"],
        ["p1", "|switch|p1a: Skarmory|Skarmory, M|333/333\n|switch|p2a: Zapdos|Zapdos|100/100"],
        ["p2", "|switch|p1a: Skarmory|Skarmory, M|100/100\n|switch|p2a: Zapdos|Zapdos|318/318"],
    ]}
    log = A.capture_spectator_log(cap)
    assert "333/333" not in log and "318/318" not in log
    assert "|switch|p1a: Skarmory|Skarmory, M|100/100" in log and "|switch|p2a: Zapdos|Zapdos|100/100" in log
    assert "|request|" not in log and "|error|" not in log and "|t:|" not in log


def test_capture_log_drops_a_private_own_side_line():
    cap = {"tag": "t", "chunks": [
        ["p1", "|switch|p1a: Suicune|Suicune|404/404\n|-ability|p1a: Suicune|Pressure|[silent]\n|turn|2"],
        ["p2", "|switch|p1a: Suicune|Suicune|100/100\n|turn|2"],
    ]}
    assert A.capture_spectator_log(cap) == "|init|battle\n|switch|p1a: Suicune|Suicune|100/100\n|turn|2\n"
    bad = {"tag": "t", "chunks": [["p1", "|-ability|p2a: Suicune|Pressure\n|turn|2"], ["p2", "|turn|2"]]}
    with pytest.raises(A.AuditError):
        A.capture_spectator_log(bad)                     # a one-sided line about the OTHER side


def test_capture_log_refuses_unpaired_streams():
    with pytest.raises(A.AuditError):
        A.capture_spectator_log({"tag": "t", "chunks": [["p1", "|turn|1"], ["p1", "|turn|2"], ["p2", "|turn|1"]]})
    with pytest.raises(A.AuditError):
        A.capture_spectator_log({"tag": "t", "chunks": [["p1", "|-message|a"], ["p2", "|-message|b"]]})


def test_ll_rows_softmax_over_the_legal_mask():
    logits = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 3.0]])
    mask = np.array([[1, 1, 0], [1, 1, 1]], np.int8)
    ll, top1 = A.ll_rows(logits, mask, np.array([0, 2]))
    assert ll[0] == pytest.approx(np.log(0.5))
    assert ll[1] == pytest.approx(3.0 - np.log(np.exp([1.0, 2.0, 3.0]).sum()))
    assert top1.tolist() == [True, True]                 # row 0's top-1 is among the LEGAL actions (0 ties 1)


def test_control_rating_fills_only_an_empty_rating():
    log = "|player|p1|Gen3AIAnchor1||\n|player|p2|MetaKakuna1||\n|player|p1|Human|avatar|1450\n"
    got = A._UNRATED_PLAYER.sub(lambda m: m.group(1) + str(A.CONTROL_RATING), log)
    assert got == "|player|p1|Gen3AIAnchor1||1400\n|player|p2|MetaKakuna1||1400\n|player|p1|Human|avatar|1450\n"


def test_the_committed_view_keeps_the_flagged_rows_and_archives_the_rest(tmp_path):
    from main.human_data import audit as AU

    res = {"flag": {"reference_p99_S": 0.0}, "rule": "r",
           "accounts": [{"acct": "1", "S": 0.5}, {"acct": "2", "S": -0.5}, {"acct": "3", "S": None}],
           "controls": {"Kakuna": {"power": {"verdict": "POWER"},
                                   "pseudo_accounts": [{"argmax": 0.8}, {"argmax": 0.6}]}, "export_counts": {"a": 1}}}
    v = AU.committed_view(res, tmp_path / "full.json")
    assert [x["acct"] for x in v["accounts"]["flagged"]] == ["1"] and v["accounts"]["n"] == 3
    assert (tmp_path / "full.json").exists() and len(v["accounts"]["full_table_sha256"]) == 64
    assert "pseudo_accounts" not in v["controls"]["Kakuna"] and v["controls"]["Kakuna"]["pseudo_median_argmax"] == 0.7
    assert v["controls"]["export_counts"] == {"a": 1} and v["rule"] == "r"
