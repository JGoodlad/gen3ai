"""The JOIN between a rollout-buffer row and its reconstruction record on disk.

`record_key` / `index_records` live beside the ring whose filename is the join key (they moved out
of the deleted `win_prob_rollout` in deletion pass L2). The fork arm (`fork_callback`) resolves a
sampled row's episode through them, and `MaskableAgentWrapper` / the async collector capture the
handle — so the join, and the DECISION-TIME capture order, stay pinned here.
"""
from __future__ import annotations

import inspect
import os

from agents.training.cf_records import CfRecordRing, index_records, record_key


def test_the_handle_and_the_ring_filename_use_the_SAME_sanitiser(tmp_path):
    """Two spellings of the sanitiser would make the join silently miss on exactly the tags that
    needed sanitising."""
    ring = CfRecordRing(tmp_path)
    tag = "battle-gen3ou-7/odd"
    path = ring.write_record(tag, {"log": []})
    assert path is not None
    idx = index_records(tmp_path)
    assert idx[record_key(os.getpid(), tag)] == str(path)


def test_the_newest_record_WINS_for_a_reused_key(tmp_path):
    from utils.bridge.reconstruction import RECON_SUFFIX
    for ns in ("0000000000000000001", "0000000000000000002"):
        (tmp_path / f"{ns}_99_battle-x{RECON_SUFFIX}").write_text("{}")
    idx = index_records(tmp_path)
    assert idx["99_battle-x"].endswith(f"0000000000000000002_99_battle-x{RECON_SUFFIX}")


def test_a_missing_records_dir_is_an_EMPTY_index_not_a_crash(tmp_path):
    assert index_records(tmp_path / "nope") == {}


def test_the_wrapper_publishes_the_handle_only_when_the_flag_threaded_it_on():
    """A run without `--fork-fraction` must not even build the tuple, and the capture must happen
    BEFORE the step: the buffer row holds the observation the decision was made FROM, so the handle
    must name the turn we were ASKED at, not the turn the step landed on."""
    from agents.training.wrappers import MaskableAgentWrapper

    src = inspect.getsource(MaskableAgentWrapper.step)
    assert "_emit_wp_rollout_handle" in src
    i_capture = src.index("_wp_handle = record_key")
    i_step = src.index("obs, reward, term, trunc, info = super().step(action)")
    assert i_capture < i_step
    assert src.index('info["wp_handle"]') > i_step, "it is published onto the info the step returned"


def test_the_async_collector_records_the_handle_on_EVERY_row_not_only_a_done_one():
    """The async collector owns the per-env buffer row, so it records inline — and unlike
    `win_outcome`, a handle exists at every decision, not only at a terminal."""
    from agents.training import async_vec_env

    src = inspect.getsource(async_vec_env)
    i_handle = src.index('_wp_keys[t, i] = str(info["wp_handle"])')
    i_done = src.index('_win_scr[t, i] = float(info["win_outcome"])')
    assert i_handle < i_done, "the handle capture must sit OUTSIDE the `if done:` block"
