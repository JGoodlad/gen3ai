"""`maturity_bank`: the k-stratified, opponent-stratified frozen-bank draw and its nested LEAN subset. Pure."""
from __future__ import annotations

import importlib.util
from collections import Counter

import pytest

from main.hidden_info_study import maturity_bank as MB
from utils.paths import repo_path

SALT = "test_salt_v1:"


def _cands(n_battles=60, strata=("snap", "bot")):
    out = []
    for b in range(n_battles):
        st = strata[b % len(strata)]
        for n, k in enumerate((1, 2, 3, 4, 5, 6, 6, 3)):        # 8 decisions per battle, every k bin present
            out.append(MB.Candidate(id=f"bat{b:03d}#p1#{n}", battle=f"bat{b:03d}", k=k, stratum=st))
    return out


def test_stage_buckets_equal_tier1s():
    p = repo_path("designs", "research_state", "measurements", "hidden_info_tier1_2026-10-13", "t1_common.py")
    spec = importlib.util.spec_from_file_location("t1_common_for_bank_test", p)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    assert (MB.K_BINS, MB.TURN_BINS, MB.MAT_BINS) == (T.K_BINS, T.TURN_BINS, T.MAT_BINS)
    for k in range(1, 7):
        assert MB.k_bin(k) == T.k_bin(k)
    for t in range(1, 60):
        assert MB.turn_bin(t) == T.turn_bin(t)
    for m in range(2, 13):
        assert MB.mat_bin(m) == T.mat_bin(m)
    with pytest.raises(ValueError):
        MB.k_bin(0)


def test_draw_fills_cells_in_hash_order_under_the_battle_caps():
    cands = _cands()
    targets = {(b, s): 8 for b in MB.K_BINS for s in ("snap", "bot")}
    sel = MB.draw(cands, targets, SALT, strata=("snap", "bot"))
    by_id = {c.id: c for c in cands}
    per_bin_bat, per_bat = Counter(), Counter()
    for (b, s), ids in sel.items():
        assert len(ids) == 8
        for x in ids:
            c = by_id[x]
            assert MB.k_bin(c.k) == b and c.stratum == s
            per_bin_bat[(b, c.battle)] += 1
            per_bat[c.battle] += 1
    assert max(per_bin_bat.values()) == 1 and max(per_bat.values()) <= 3
    # the first selected id of the first cell is the cell's first candidate in sha256(salt + id) order
    first = min((c for c in cands if MB.k_bin(c.k) == "1-2" and c.stratum == "snap"),
                key=lambda c: MB.order_key(SALT, c.id))
    assert sel[("1-2", "snap")][0] == first.id
    # a different salt draws a different bank
    assert MB.draw(cands, targets, "other:", strata=("snap", "bot")) != sel


def test_draw_refuses_a_short_cell_and_bad_input():
    cands = _cands(n_battles=10)
    with pytest.raises(MB.BankShort):
        MB.draw(cands, {("6", "snap"): 50}, SALT, strata=("snap", "bot"))
    with pytest.raises(ValueError, match="two strata"):
        MB.draw([MB.Candidate("a", "b1", 3, "snap"), MB.Candidate("c", "b1", 4, "bot")], {("3", "snap"): 1},
                SALT, strata=("snap", "bot"))
    with pytest.raises(ValueError, match="not in"):
        MB.draw(cands, {("3", "elite"): 1}, SALT, strata=("snap", "bot"))


def test_lean_is_nested_in_full():
    cands = _cands(n_battles=120)
    full = MB.draw(cands, {(b, s): 16 for b in MB.K_BINS for s in ("snap", "bot")}, SALT, strata=("snap", "bot"))
    lean = MB.nested_subset(full, 6)
    assert set(MB.flat(lean)) < set(MB.flat(full))
    assert all(lean[c] == full[c][:6] for c in full)
    assert len(MB.flat(lean)) == 6 * 10
    assert MB.ids_sha(MB.flat(lean)) == MB.ids_sha(reversed(MB.flat(lean)))
    with pytest.raises(MB.BankShort):
        MB.nested_subset(full, 17)


def test_draw_never_takes_an_excluded_battle():
    cands = _cands(80)
    targets = {(b, s): 4 for b in MB.K_BINS for s in ("snap", "bot")}
    base = MB.draw(cands, targets, SALT, ("snap", "bot"))
    taken = {c.battle for c in cands if c.id in set(MB.flat(base))}
    sup = MB.draw(cands, targets, SALT, ("snap", "bot"), exclude_battles=taken)
    sup_battles = {c.battle for c in cands if c.id in set(MB.flat(sup))}
    assert sup_battles and not (sup_battles & taken)            # the supplement shares no battle with the first bank
    with pytest.raises(MB.BankShort):                            # exclusion is never relaxed to fill a cell
        MB.draw(cands, targets, SALT, ("snap", "bot"), exclude_battles={c.battle for c in cands[: 8 * 75]})
