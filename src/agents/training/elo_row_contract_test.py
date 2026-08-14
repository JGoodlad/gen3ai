"""The ELO reader must read the rows the eval pipeline actually WRITES.

The regression this exists to prevent, found on a live run: `eval_results.jsonl` rows are written as

    {"step": ..., "bots": {"heuristic": 0.94, ...}, "sentinels": [{"step":..., "win_rate":...}]}

while `_row_from_block` read bot results from `block["opponents"]` as `{name: {"win_rate": ...}}`
and sentinels from `block["pool"]["sentinels"]`. Every jsonl row therefore produced an EMPTY bot
dict, `fit_from_block` returned None, and the caller swallowed it in a best-effort `try` — so a
whole run reported no live `eval/elo` and a blank TUI badge, with nothing failing.

Two shapes genuinely exist and both must keep working:

* the metadata `latest_eval` block (`opponents` / `pool.sentinels`), and
* the `eval_results.jsonl` row (`bots` / top-level `sentinels`).

`_rows_from_log` already understood the second, which is why the OFFLINE ladder kept working and
masked the problem — the two readers had drifted apart and only one was exercised by anything.
"""
import pytest

from agents.training.elo import _row_from_block, fit_from_block


def _jsonl_shape():
    """A row exactly as the eval pipeline appends it (fields copied from a real run)."""
    return {
        "step": 12000000,
        "n_games": 100,
        "bots": {"random": 1.0, "heuristic": 0.94, "heuristic2": 0.87, "staller": 0.92,
                 "staller_v2": 0.87, "aggressive": 0.9, "aggressive_v2": 0.88,
                 "setup_sweep": 0.89, "setup_sweep_v2": 0.88},
        "sentinels": [{"step": 10000032, "win_rate": 0.63}, {"step": 8000016, "win_rate": 0.79}],
        "counts": {"random": [99, 100]},
        "evaluated_at": "2026-08-13T00:00:00",
    }


def _metadata_shape():
    """The other real shape — the `latest_eval` block inside metadata.json."""
    return {
        "step": 12000000,
        "opponents": {"random": {"win_rate": 1.0}, "heuristic": {"win_rate": 0.94}},
        "pool": {"sentinels": [{"step": 8000016, "win_rate": 0.79}]},
    }


@pytest.mark.parametrize("block,label", [(_jsonl_shape(), "jsonl"),
                                         (_metadata_shape(), "metadata")])
def test_both_written_shapes_yield_bot_win_rates(block, label):
    """THE contract. An empty `bots` is what silently disabled ELO for an entire run."""
    row = _row_from_block(block)
    assert row is not None, f"{label} row rejected outright"
    assert row.bots, (
        f"the {label} shape produced NO bot win rates — `fit_from_block` will return None and the "
        f"caller will swallow it, which is exactly how a run reports no ELO while nothing fails")
    assert row.sentinels, f"the {label} shape produced no sentinels"


def test_the_jsonl_shape_actually_fits():
    """End of the chain: a written row must produce a rating, not merely parse."""
    fit = fit_from_block(_jsonl_shape())
    assert fit is not None, "a row the pipeline writes must be fittable"
    elo, se = fit.rating_for_step(12000000)
    assert 1000.0 < elo < 3000.0, elo
    assert se > 0.0


def test_bare_float_and_dict_bot_values_are_both_accepted():
    """The jsonl uses bare floats; the metadata block nests under `win_rate`."""
    bare = _row_from_block({"step": 1, "bots": {"a": 0.75}})
    nested = _row_from_block({"step": 1, "bots": {"a": {"win_rate": 0.75}}})
    assert bare.bots == nested.bots == {"a": 0.75}


def test_the_rows_own_game_count_beats_the_callers_default():
    """`n_games=100` is an assumption; a row that records what it played must not be re-weighted."""
    assert _row_from_block({"step": 1, "bots": {"a": 0.5}, "n_games": 40}, n_games=100).n_games == 40
    assert _row_from_block({"step": 1, "bots": {"a": 0.5}}, n_games=100).n_games == 100


def test_a_block_with_no_step_is_still_rejected():
    """The one legitimate None: without a step the row cannot be placed on the ladder."""
    assert _row_from_block({"bots": {"a": 0.9}}) is None


def test_the_two_readers_agree_on_the_same_jsonl_row():
    """`_rows_from_log` and `_row_from_block` drifted apart once; pin them together.

    That drift is the whole bug: the offline ladder kept working through one reader while the live
    path was dead through the other, so nothing looked broken.
    """
    import json
    import os
    import tempfile

    from agents.training.elo import _rows_from_log
    row = _jsonl_shape()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "eval_results.jsonl"), "w") as f:
            f.write(json.dumps(row) + "\n")
        via_jsonl = _rows_from_log(d)[0]
    via_block = _row_from_block(row)
    assert via_jsonl.bots == via_block.bots
    assert via_jsonl.sentinels == via_block.sentinels
    assert via_jsonl.n_games == via_block.n_games
