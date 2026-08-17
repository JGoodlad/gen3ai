"""Tests for ProbeModel (the torch boundary).

Most of the engine is exercised through a FakeProbeModel (no torch); these pin the few real
ProbeModel attribute reads a fake can't catch — specifically WHERE each forward stash lives, which
is invisible to a fake that returns the decoded view directly. (The DamageOperator stashes on the OP
submodule, not the extractor — a read of the wrong object silently returned None and hid the entire
op view in the prober, with nothing to catch it until this test.)
"""

import json
import os
import zipfile

import numpy as np
import pytest
import torch

from main.prober.model import (
    ArchDriftError, ObsOffsets, ProbeModel, _accepted_extractor_kwargs, _arch_drift_error,
    _sidecar, peek_checkpoint,
)

_OFF = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5,
                  turn_history_offset=0, turn_history_dim=0)


class _Op:
    """Stand-in DamageOperator submodule: it stashes ``last_raw_block`` on ITSELF (as the real op does)."""
    def __init__(self, row, outgoing=False):
        self.outgoing = outgoing
        self.last_raw_block = torch.as_tensor(row).unsqueeze(0)
        # gen3_extractor_stashes_v1: the reader now uses the op's TYPED property surface
        # directly (no getattr default), so the stand-in must carry the whole surface it
        # touches — as the real op always does (OpStashes fields default None).
        self.last_topk_idx = None


class _Ext:
    """Extractor carrying the op submodule but NO ``last_raw_block`` of its own (the bug read it here)."""
    def __init__(self, op):
        self.damage_op = op


class _Pol:
    def __init__(self, ext):
        self.features_extractor = ext

    def extract_features(self, obs):   # no-op — the stash is pre-set, no real forward needed
        return None


def test_damage_op_view_reads_op_submodule_stash():
    """REGRESSION: the DamageOperator stashes ``last_raw_block`` on the OP submodule, not the extractor.
    ``damage_op_view`` must read ``op.last_raw_block`` — reading ``extractor.last_raw_block`` (the old bug)
    always returned None and silently hid the ENTIRE op view (incoming + outgoing damage) in the prober."""
    width = 6 * 12 + 13                   # incoming per-mon + choice_band (outgoing off)
    row = (np.arange(width, dtype=np.float32) + 1.0) / width
    pm = ProbeModel(policy=_Pol(_Ext(_Op(row, outgoing=False))), offsets=_OFF)
    out = pm.damage_op_view(np.zeros(8, dtype=np.float32), np.ones(11, dtype=np.int8))
    assert out is not None                                  # was None before the fix (read the wrong object)
    assert len(out["incoming"]) == 6 and out["outgoing"] is None
    assert set(out["incoming"][0]["phys"]) == {"low", "high", "crit", "pko", "acc"}


def test_damage_op_view_none_when_no_op():
    """No DamageOperator submodule (``--damage-op`` off) → None, cleanly (before any forward)."""
    class _ExtNoOp:
        damage_op = None

    pm = ProbeModel(policy=_Pol(_ExtNoOp()), offsets=_OFF)
    assert pm.damage_op_view(np.zeros(8, dtype=np.float32), np.ones(11, dtype=np.int8)) is None


# ── Architecture-drift loading (ArchDriftError + the kwarg-drop recovery) ─────
# This project changes the architecture continuously, so most archived checkpoints CANNOT be re-run
# under current code. That was always true; what these pin is that it now fails as a DIAGNOSIS —
# what drifted and which commit to check out — rather than as a raw torch error thrown four frames
# inside SB3, and that a checkpoint whose ONLY problem is a deleted flag still loads.

def _fake_ckpt(path, extractor_kwargs, obs_end=2669):
    """A zip shaped like an SB3 checkpoint's `data` member — enough for the peek, which is the whole
    point of the peek: it must never deserialize 27MB to answer "what arch is this"."""
    data = {"policy_kwargs": {"features_extractor_kwargs": dict(
        extractor_kwargs, layout={"parts": {"our_team": {"start": 0, "end": 696},
                                            "reactive": {"start": 700, "end": obs_end}}})}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps(data))
    return str(path)


def test_peek_reads_the_arch_without_deserializing(tmp_path):
    ckpt = _fake_ckpt(tmp_path / "m.zip", {"spread_belief": True, "gone_flag": 1}, obs_end=2669)
    peek = peek_checkpoint(ckpt)
    assert peek["obs_dim"] == 2669
    assert set(peek["extractor_kwargs"]) == {"spread_belief", "gone_flag", "layout"}


def test_peek_never_raises_on_garbage(tmp_path):
    """A raising peek would turn a merely-unreadable checkpoint into a crash BEFORE the real load
    got its chance to produce a proper error."""
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert peek_checkpoint(str(bad)) == {}
    assert peek_checkpoint(str(tmp_path / "nope.zip")) == {}


def test_accepted_kwargs_excludes_self():
    accepted = _accepted_extractor_kwargs()
    assert accepted is None or ("self" not in accepted and "observation_space" in accepted)


def test_sidecar_reaches_the_run_root_from_an_eval_snapshot(tmp_path):
    """REGRESSION: an eval snapshot lives at `<run>/eval_traces/step_<N>/snapshot.zip`, so the
    run-level metadata.json — the ONLY source of the git_hash the drift message tells you to check
    out — is a GRANDPARENT away. A two-level search lost it on every retained snapshot, silently."""
    step = tmp_path / "run" / "eval_traces" / "step_42"
    os.makedirs(step)
    (tmp_path / "run" / "metadata.json").write_text(json.dumps({"git_hash": "cafe1234"}))
    ckpt = step / "snapshot.zip"
    ckpt.write_bytes(b"")
    assert _sidecar(str(ckpt), "metadata.json")["git_hash"] == "cafe1234"
    assert _sidecar(str(ckpt), "model_config.json") == {}      # absent → {}, never a raise


def test_drift_error_names_the_drift_and_the_commit(tmp_path):
    """The message has exactly one job: replace "mat1 and mat2 shapes cannot be multiplied" with
    what drifted and what to do next."""
    run = tmp_path / "run"
    os.makedirs(run)
    (run / "metadata.json").write_text(json.dumps({"git_hash": "deadbeef"}))
    (run / "model_config.json").write_text(json.dumps({"arch_signature": "gen3_old_v1"}))
    ckpt = _fake_ckpt(run / "m.zip", {"gone": 1}, obs_end=2992)

    err = _arch_drift_error(ckpt, peek_checkpoint(ckpt), ("gone",),
                            RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    text = str(err)
    assert "2992" in text                       # what it was trained on
    assert "gen3_old_v1" in text                # its arch signature
    assert "gone" in text                       # the flag the code deleted since
    assert "git checkout deadbeef" in text      # the actionable next step
    assert "scan" in text and "triage" in text  # and what DOES still work on this run
    assert err.saved_obs_dim == 2992 and err.git_hash == "deadbeef"
    assert err.dropped_kwargs == ("gone",)


def test_load_drops_unknown_kwargs_and_records_the_drop(tmp_path, monkeypatch):
    """The recovery: a checkpoint whose only problem is a DELETED flag still loads — and the drop is
    RECORDED, because a dropped flag means the rebuilt extractor is not the one that played and a
    surface has to be able to say so."""
    import sb3_contrib
    from stable_baselines3.common import save_util

    accepted = _accepted_extractor_kwargs()
    assert accepted, "signature introspection failed — the recovery path cannot be tested"
    keep = sorted(accepted)[0]
    ckpt = _fake_ckpt(tmp_path / "m.zip", {keep: 1, "deleted_flag": True})
    monkeypatch.setattr(save_util, "load_from_zip_file", lambda *a, **k: (
        {"policy_kwargs": {"features_extractor_kwargs": {keep: 1, "deleted_flag": True}}}, {}, {}))

    seen = {}
    op_row = np.zeros(6 * 12 + 13, dtype=np.float32)

    class _StubPolicy(_Pol):
        def set_training_mode(self, mode):
            pass

        def modules(self):
            return []

    def fake_load(path, device="cpu", custom_objects=None):
        seen["kwargs"] = custom_objects["policy_kwargs"]["features_extractor_kwargs"]
        return type("M", (), {"policy": _StubPolicy(_Ext(_Op(op_row)))})()

    monkeypatch.setattr(sb3_contrib.MaskablePPO, "load", staticmethod(fake_load))

    pm = ProbeModel.load(ckpt)
    assert "deleted_flag" not in seen["kwargs"] and keep in seen["kwargs"]
    assert pm.dropped_kwargs == ("deleted_flag",)


def test_load_turns_any_failure_into_a_diagnosis(tmp_path, monkeypatch):
    """Walls 2 and 3 (a value the code now rejects; weight shapes that no longer fit) are NOT
    recoverable — but they must still arrive as an ArchDriftError, with the original preserved."""
    import sb3_contrib

    ckpt = _fake_ckpt(tmp_path / "m.zip", {"whatever": 1})

    def boom(*a, **k):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (12x380 and 386x256)")

    monkeypatch.setattr(sb3_contrib.MaskablePPO, "load", staticmethod(boom))
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(ckpt)
    assert "cannot be re-run" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)   # the cause is chained, not swallowed
