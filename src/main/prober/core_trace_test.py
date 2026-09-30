"""`main.prober.core_trace` without a bridge: the cross-check's refusal, the room tag, and the
session's meta-only read (a core trace is never expanded to answer `run_summary`).

The end-to-end expansion — and its equality with the live Python recorder — is
`core_trace_integration_test.py` (`sim`). Milliseconds; unmarked.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from main.prober.core_trace import (
    CoreTraceError, CoreTraceMismatch, _check_protocol, _masked_probs, expand, protocol_lines,
)


def test_the_cross_check_ignores_timestamps_and_names_the_first_differing_line():
    rec = ["|t:|<NORMALIZED>", "|turn|1", "|move|p1a: Jirachi|Wish|p1a: Jirachi", "|upkeep"]
    rep = ["|t:|1727", "", "|turn|1", "|move|p1a: Jirachi|Wish|p1a: Jirachi", "|upkeep"]
    _check_protocol(rep, rec, "r")                        # equal modulo |t:| and blanks
    assert protocol_lines(rep) == protocol_lines(rec)
    with pytest.raises(CoreTraceMismatch, match=r"protocol line 1: record='\|move\|p1a: Jirachi\|Wish"):
        _check_protocol(["|turn|1", "|move|p1a: Jirachi|Protect|p1a: Jirachi"], rec, "r")
    with pytest.raises(CoreTraceMismatch, match="record has 1 more, first '\\|upkeep'"):
        _check_protocol(rep[:-1], rec, "r")


def test_probs_are_the_live_masked_softmax():
    logits = np.array([-1e9, -0.5, -1.2] + [-1e9] * 8, dtype=np.float32)
    mask = np.array([0, 1, 1] + [0] * 8, dtype=np.int8)
    p = _masked_probs(logits, mask)
    assert p.dtype == np.float32 and p[mask == 0].sum() == 0
    assert np.isclose(p.sum(), 1.0) and np.isclose(p[1] / p[2], np.exp(0.7), rtol=1e-5)


def test_a_core_battle_id_becomes_a_room_tag_poke_env_accepts():
    """`Player._create_battle` only opens a room whose segment 1 is the format; a core trace's
    battle_id (`core-<step>-<opp>-g<k>`) is not one — every replay of it read 0 decisions."""
    from agents.training.obs_materializer import _next_tag

    t = _next_tag("core-1000-heuristic-g3", "gen3ou")
    assert t.split("-")[1] == "gen3ou" and "core-1000-heuristic-g3" in t
    ok = _next_tag("battle-gen3ou-42", "gen3ou")
    assert ok.startswith("battle-gen3ou-42-recon")


def test_a_non_core_summary_passes_through_untouched():
    s = {"meta": {"step": 1}, "invocations": []}
    assert expand("/nowhere_summary.json", s) is s


def _core_meta_only(tmp_path) -> str:
    d = tmp_path / "eval_traces" / "step_1000" / "heuristic"
    d.mkdir(parents=True)
    sp = d / "loss_s0_001_summary.json"
    sp.write_text(json.dumps({"meta": {
        "step": 1000, "battle_id": "core-1000-heuristic-g0", "result": "LOSS", "turns": 9,
        "invocations": 7, "result_vocabulary": "gen3_trace_result_v2",
        "trace_source": {"schema": "gen3_core_trace_v1", "kind": "gen3_core_event_v1",
                         "env_core": "rust", "trainee_side": "p1",
                         "records": {"p1": "loss_s0_001.p1.jsonl.gz", "p2": "loss_s0_001.p2.jsonl.gz"},
                         "reconstruction": "loss_s0_001_reconstruction.json",
                         "states": "loss_s0_001_states.npz"}}}))
    return str(sp)


def test_the_session_reads_a_core_meta_without_expanding_and_refuses_a_broken_trace(tmp_path):
    from main.prober.session import ProbeSession

    sp = _core_meta_only(tmp_path)
    sess = ProbeSession(str(tmp_path))
    (b,) = sess.tree.all_battles()
    assert sess._meta(b)["battle_id"] == "core-1000-heuristic-g0"
    summ = sess.run_summary()                       # meta-only: never touches the missing siblings
    assert summ["result_vocabulary"] == ["gen3_trace_result_v2"]
    with pytest.raises(CoreTraceError, match="missing record, reconstruction, states"):
        sess._summary(b)
    assert os.listdir(os.path.dirname(sp)) == [os.path.basename(sp)]   # nothing written
