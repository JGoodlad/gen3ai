"""Unit tests for the bot-vs-bot game-count store (`bot_elo_store.py`).

Pure — no bridge/poke-env. The subtle property is **order-consistency**: counts must sum the
same regardless of which side a caller (or an independent fleet process) passed as `a`/`b`.
"""
import json

from agents.training import bot_elo_store as store


def test_accumulate_is_order_consistent():
    s = store.new_store("gh", ["random", "heuristic"])
    # Same matchup, opposite (a, b) order — must land in one entry with wins attributed right.
    store.accumulate(s, "heuristic", "random", wins_a=8, wins_b=2, n_games=10)
    store.accumulate(s, "random", "heuristic", wins_a=3, wins_b=7, n_games=10)  # flipped
    assert store.games(s, "random", "heuristic") == 20
    assert store.games(s, "heuristic", "random") == 20  # symmetric lookup
    e = s["pairs"][store.pair_key("heuristic", "random")]
    # heuristic is the lexicographically-lower name → wins_a tracks heuristic: 8 + 7 = 15.
    assert e["a"] == "heuristic" and e["b"] == "random"
    assert e["wins_a"] == 15 and e["wins_b"] == 5 and e["games"] == 20


def test_results_and_matrix():
    s = store.new_store("gh", ["random", "heuristic"])
    store.accumulate(s, "heuristic", "random", 18, 2, 20)
    results, wm = store.results_and_matrix(s, ["random", "heuristic"])
    assert results == [("heuristic", "random", 18, 20)]
    assert wm["heuristic"]["random"] == 0.9
    assert wm["random"]["heuristic"] == 0.1


def test_merge_sums_across_parts(tmp_path):
    names = ["random", "heuristic", "aggressive"]
    p1 = store.new_store("gh", names)
    store.accumulate(p1, "heuristic", "random", 9, 1, 10)
    p2 = store.new_store("gh", names)
    store.accumulate(p2, "random", "heuristic", 2, 8, 10)        # overlapping pair, flipped
    store.accumulate(p2, "aggressive", "random", 6, 4, 10)       # disjoint pair
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(p1))
    b.write_text(json.dumps(p2))

    merged = store.merge([str(a), str(b)], names, "gh")
    assert store.games(merged, "heuristic", "random") == 20      # 10 + 10
    e = merged["pairs"][store.pair_key("heuristic", "random")]
    assert e["wins_a"] == 17 and e["wins_b"] == 3                # heuristic: 9 + 8
    assert store.games(merged, "aggressive", "random") == 10     # disjoint carried over


def test_load_save_roundtrip_and_git_warning(tmp_path, capsys):
    names = ["random", "heuristic"]
    s = store.new_store("hash_a", names)
    store.accumulate(s, "heuristic", "random", 7, 3, 10)
    path = str(tmp_path / "games.json")
    store.save(path, s)

    # Resume with a DIFFERENT git hash → keeps counts, warns.
    reloaded = store.load(path, names, "hash_b")
    assert store.games(reloaded, "heuristic", "random") == 10
    assert "git_hash" in capsys.readouterr().out

    # --reset starts fresh.
    fresh = store.load(path, names, "hash_b", reset=True)
    assert fresh["pairs"] == {}


def test_missing_file_inits_empty(tmp_path):
    s = store.load(str(tmp_path / "nope.json"), ["random"], "gh")
    assert s["pairs"] == {} and s["git_hash"] == "gh"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
