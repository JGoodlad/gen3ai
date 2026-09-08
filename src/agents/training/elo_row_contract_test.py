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


# ── the row's OPPONENT-REGIME stamp + exact sentinel counts (gen3_eval_sentinel_greedy_default_v1)
# `snapshot_ladder.eval_measured_pairs` is a SECOND reader of these rows, and it consumes exactly
# the two fields added on 2026-09-07. Pin them at the WRITER, so a rename cannot leave the ladder
# silently reusing nothing (or, far worse, silently reusing an asymmetric edge).

def _written_row(tmp_path, **kw):
    import json

    from agents.model.snapshot import append_eval_result_row
    append_eval_result_row(str(tmp_path), 12_000_000, 100, {"random": 1.0}, **kw)
    return json.loads((tmp_path / "eval_results.jsonl").read_text().splitlines()[0])


def test_the_regime_block_and_sentinel_counts_are_written(tmp_path):
    row = _written_row(
        tmp_path,
        sentinels=[{"step": 8_000_000, "win_rate": 0.6, "counts": [60, 100]}],
        sentinel_regime={"greedy": True, "symmetric_teams": True})
    assert row["sentinel_regime"] == {"greedy": True, "symmetric_teams": True}
    assert row["sentinels"] == [{"step": 8_000_000, "win_rate": 0.6, "counts": [60, 100]}]


def test_a_row_written_without_the_new_fields_is_byte_identical_to_the_old_shape(tmp_path):
    """A caller that supplies neither (the bot-only eval path, and every pre-2026-09-07 writer)
    must produce exactly the row it always produced — an absent stamp means UNKNOWN, and the ladder
    reads unknown as not-reusable."""
    row = _written_row(tmp_path, sentinels=[{"step": 8_000_000, "win_rate": 0.6}])
    assert "sentinel_regime" not in row
    assert row["sentinels"] == [{"step": 8_000_000, "win_rate": 0.6}]


def test_the_elo_readers_ignore_the_new_fields(tmp_path):
    """Additive, both ways: the rating fit still reads the same edges off an enriched row."""
    row = _written_row(
        tmp_path,
        sentinels=[{"step": 8_000_000, "win_rate": 0.6, "counts": [60, 100]}],
        sentinel_regime={"greedy": True, "symmetric_teams": True})
    assert _row_from_block(row).sentinels == [(8_000_000, 0.6)]


def test_the_ladder_reads_the_stamp_the_writer_wrote(tmp_path):
    """The end-to-end join: writer → row → `snapshot_ladder.eval_measured_pairs`. This is the pair
    of names that must not drift, and the only test that fails if either side is renamed."""
    from agents.training import snapshot_ladder as sl

    _written_row(tmp_path,
                 sentinels=[{"step": 8_000_000, "win_rate": 0.6, "counts": [60, 100]}],
                 sentinel_regime={"greedy": True, "symmetric_teams": True})
    assert sl.eval_measured_pairs(str(tmp_path)) == {(8_000_000, 12_000_000): [40, 100]}
