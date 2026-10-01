"""Unit tests of the STATE-level RND reads (synthetic data, CPU, seconds)."""

from __future__ import annotations

import numpy as np
import pytest

from main.ridealong_read import rnd_states as S


def _decisions(n_battles=40, rows_per=4):
    out = []
    for b in range(n_battles):
        src = "N0@36M" if b % 2 else "K2@90M"
        oc = "bot" if b % 4 < 2 else "pool_snapshot"
        for n in range(rows_per):
            out.append({"battle": f"b{b}", "source": src, "opp_class": oc, "turn": n + 1,
                        "our_alive": 6, "opp_alive": 6})
    return out


def test_split_is_battle_level_within_every_cell_and_deterministic():
    d = _decisions()
    tr = S.split_within_cells(d)
    np.testing.assert_array_equal(tr, S.split_within_cells(d))
    battles = np.array([x["battle"] for x in d])
    assert not set(battles[tr]) & set(battles[~tr])               # no battle straddles
    for cell in {(x["source"], x["opp_class"]) for x in d}:
        m = np.array([(x["source"], x["opp_class"]) == cell for x in d])
        n = len(set(battles[m]))
        assert len(set(battles[m & tr])) == round(0.7 * n)        # 70 % of EACH cell's battles


def test_split_refuses_a_battle_in_two_cells():
    d = _decisions()
    d[1] = dict(d[1], opp_class="exploiter")
    with pytest.raises(ValueError, match="two cells"):
        S.split_within_cells(d)


def _vector(L, our=(3, 151, 0.40), opp=(1, 248, 1.0), turn=17, weather=3, spikes=(0, 2)):
    """A synthetic observation: our slot ``our[0]`` active (species, HP), theirs likewise; one
    fainted revealed opponent in slot 5."""
    from agents.observation.constants import MAX_SPIKES

    v = np.zeros(L.total_dim, np.float32)
    for start, (slot, sp, hp) in ((L.our_start, our), (L.opp_start, opp)):
        base = start + slot * L.slot_dim
        v[base + L.active_off] = 1.0
        v[base + L.species_off] = sp
        v[base + L.hp_off] = hp
    for i in range(L.team_size):                                  # our bench: alive, slot 0 fainted
        if i != our[0]:
            v[L.our_start + i * L.slot_dim + L.species_off] = 10 + i
            v[L.our_start + i * L.slot_dim + L.hp_off] = 0.0 if i == 0 else 0.5
    v[L.opp_start + 5 * L.slot_dim + L.species_off] = 77           # a revealed, fainted opponent
    v[L.clock_off] = np.log1p(turn) / L.log_max_turns
    v[L.weather_off + weather] = 1.0
    v[L.hazards_off] = spikes[0] / MAX_SPIKES
    v[L.hazards_off + 1] = spikes[1] / MAX_SPIKES
    return v


def test_decode_state_reads_what_the_encoders_own_describe_reads():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    L = S.state_layout()
    v = _vector(L)
    d = S.decode_state(v[None, :], L)
    assert d["our_species"][0] == 151 and d["opp_species"][0] == 248
    assert d["our_hp"][0] == pytest.approx(0.40) and d["opp_hp"][0] == pytest.approx(1.0)
    assert d["turn"][0] == 17 and d["weather"][0] == 3
    assert not d["our_spikes"][0] and d["opp_spikes"][0]
    assert d["our_alive"][0] == 5 and d["opp_alive"][0] == 5
    # the encoder's own describe_vector agrees on the active flag, HP and the turn
    desc = Gen3ObservationEncoder(load_mappings()).describe_vector(v)
    act = [m for m in desc["our_team"] if m["active"]]
    assert len(act) == 1 and act[0]["hp"] == "40.0%"
    assert round(desc["world"]["turn"]) == 17
    key = S.state_keys(d)[0]
    assert key == "151|248|2|2|4|1|0|1"


def test_buckets_edges():
    np.testing.assert_array_equal(S.turn_bucket(np.array([1, 3, 4, 10, 11, 60, 61, 300])), [0, 0, 1, 1, 2, 4, 5, 5])
    np.testing.assert_array_equal(S.hp_bucket(np.array([0.0, 0.01, 0.25, 0.26, 0.75, 0.76, 1.0])),
                                  [0, 1, 1, 2, 3, 4, 4])


def test_train_counts_count_train_rows_only():
    keys = np.array(["a", "a", "b", "c", "a"])
    train = np.array([True, True, True, False, False])
    np.testing.assert_array_equal(S.train_counts(keys, train), [2, 2, 1, 0, 2])


def test_decile_and_bin_profiles_see_a_monotone_signal():
    rng = np.random.default_rng(0)
    count = rng.integers(0, 64, 2000)
    err = 1.0 / (1.0 + count) + rng.normal(0, 1e-4, 2000)
    dp = S.decile_profile(np.log1p(count), err)
    assert dp["decreasing_steps"] == 9
    cb = S.count_bin_profile(count, err)
    assert cb["bins"][0]["count"] == [0, 0] and cb["decreasing_steps"] == cb["steps"]
    flat = S.decile_profile(np.log1p(count), rng.normal(0, 1, 2000))
    assert flat["decreasing_steps"] < 9


def test_read_visitation_spearman_is_negative_for_a_working_signal():
    d = _decisions(60, 10)
    battles = np.array([x["battle"] for x in d])
    train = S.split_within_cells(d)
    rng = np.random.default_rng(1)
    count = rng.integers(0, 30, len(d))
    err = {10: 1.0 / (1.0 + count) + rng.normal(0, 1e-3, len(d))}
    r = S.read_visitation(err, count, train, battles)
    sp = r["10"]["spearman_err_vs_log1p_count"]
    assert sp["rho"] < -0.9 and sp["ci"][1] < 0
    assert r["heldout_count0"] == int((count[~train] == 0).sum())


def _turn(i, cls):
    return {"id": f"t{i}", "battle": f"b{i % 7}", "row": i, "argmax": 0, "cls": cls}


def test_read_starvation_pairs_within_turns_and_skips_ended_branches():
    turns = [_turn(i, {0: "argmax", 1: "starved_near", 2: "starved_far", 3: "fed"}) for i in range(30)]
    turns.append(_turn(99, {0: "argmax", 1: "fed"}))                 # no starved near-best: not read
    ids, acts, ended, err = [], [], [], []
    for T in turns:
        for a, c in T["cls"].items():
            ids.append(T["id"])
            acts.append(a)
            ended.append(T["id"] == "t0" and a == 1)                   # one game-ending branch
            err.append({"argmax": 1.0, "starved_near": 3.0, "starved_far": 2.0, "fed": 1.5}[c])
    succ = {"ids": np.array(ids), "acts": np.array(acts), "ended": np.array(ended)}
    r = S.read_starvation(turns, succ, np.array(err), scale=1.0)
    assert r["starved_turns"] == 30 and r["ended_branches"]["starved_near"] == 1
    sn = r["starved_near_vs_argmax"]
    assert sn["turns"] == 29 and sn["auroc"]["auroc"] == 1.0
    assert sn["paired_diff"]["mean"] == pytest.approx(2.0)
    assert sn["share_turns_diff_positive"]["mean"] == 1.0
    assert sn["paired_log_ratio"]["mean"] == pytest.approx(np.log(3.0), abs=1e-4)
    assert S.read_starvation(turns, succ, np.array(err))["starved_near_vs_argmax"]["paired_diff"]["mean"] == 2000.0
    assert r["fed_vs_argmax"]["paired_diff"]["mean"] == pytest.approx(0.5)
    assert r["starved_near_minus_starved_far"]["paired_diff"]["mean"] == pytest.approx(1.0)
    only = S.read_starvation(turns, succ, np.array(err), heldout_battles={"b1"}, log_ratio=False)
    assert "paired_log_ratio" not in only["starved_near_vs_argmax"]
    assert only["starved_turns"] == sum(T["battle"] == "b1" and "starved_near" in T["cls"].values() for T in turns)


def test_train_snapshots_score_reproduces_and_learns():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (512, 12)).astype(np.float32)
    far = rng.normal(4, 1, (128, 12)).astype(np.float32)
    s = S.train_snapshots(x, epochs_read=(1, 4))
    assert sorted(s["snap"]) == [1, 4] and s["n_train"] == 512
    a = S.score(s["snap"][4], s["in_dim"], x)
    np.testing.assert_array_equal(a, S.score(s["snap"][4], s["in_dim"], x))
    assert S.score(s["snap"][4], s["in_dim"], far).mean() > a.mean()
    assert S.score(s["snap"][4], s["in_dim"], x).mean() < S.score(s["snap"][1], s["in_dim"], x).mean()


def test_within_turn_share_separates_root_variance_from_action_variance():
    ids = np.repeat(np.arange(50), 4).astype(str)
    roots = np.repeat(np.arange(50, dtype=float), 4)
    assert S.within_turn_share(ids, roots) == 0.0                     # only the root differs
    acts = np.tile(np.arange(4, dtype=float), 50)
    assert S.within_turn_share(ids, acts) == 1.0                      # only the action differs
