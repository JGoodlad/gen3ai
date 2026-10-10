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

from agents.model import snapshot
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

def _fake_ckpt(path, extractor_kwargs, obs_end=None):
    """A zip shaped like an SB3 checkpoint's `data` member — enough for the peek, which is the whole
    point of the peek: it must never deserialize 27MB to answer "what arch is this". Its observation is
    the CURRENT width unless a test says otherwise, so the pre-load verdict has nothing to object to."""
    if obs_end is None:
        from main.prober.arch_status import current_obs_dim
        obs_end = current_obs_dim()
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
    """ASSERTED, not tolerated (gen3_vacuity_hunt_v1).

    This was `accepted is None or (...)`. `None` means "the extractor's constructor takes
    `**kwargs`, so nothing can be judged undroppable" — a fact about OUR OWN code in THIS tree,
    not an environment we might not have. That is the arranged-vs-encountered line: under the
    old disjunction, adding `**kwargs` to `Gen3FeaturesExtractor.__init__` would have turned
    this test and the one below into unconditional passes, silently, with the load sanitizer
    they cover no longer doing anything.
    """
    accepted = _accepted_extractor_kwargs()
    assert accepted is not None, (
        "Gen3FeaturesExtractor.__init__ now takes **kwargs — nothing is droppable, so the load "
        "sanitizer cannot flag a deleted flag and every checkpoint predating a flag deletion "
        "will TypeError in the offline readers. This is a real regression, not an exemption.")
    assert "self" not in accepted and "observation_space" in accepted


def test_dropped_extractor_kwargs_flags_deleted_flags_and_keeps_live_ones():
    """The load sanitizer's decision layer: a saved kwarg the current constructor no longer accepts
    must be flagged for dropping, while live kwargs and `layout` survive. This is what lets an
    offline reader (analyze, and the replay/counterfactual/lookahead rollouts) load a checkpoint
    written before a flag was deleted or demoted (v78 value_active_readout /
    damage_matrices_outgoing_all, v88 pubval_mode) instead of TypeError-ing on it."""
    from main.prober.model import _dropped_extractor_kwargs

    # ASSERTED, not skipped (gen3_vacuity_hunt_v1) — see `test_accepted_kwargs_excludes_self`.
    # A `**kwargs` constructor is a regression in the thing this test covers, not a reason the
    # test does not apply; the old `pytest.skip` would have retired this coverage in silence.
    accepted = _accepted_extractor_kwargs()
    assert accepted is not None, (
        "Gen3FeaturesExtractor.__init__ takes **kwargs — the load sanitizer can no longer drop "
        "a deleted flag, which is exactly the failure this test exists to catch")
    # a definitely-live kwarg + `layout` + a definitely-dead one
    saved = {"damage_op": True, "layout": {}, "__deleted_flag__": 7}
    dropped = _dropped_extractor_kwargs(saved)
    assert dropped == ("__deleted_flag__",), dropped
    assert _dropped_extractor_kwargs({"damage_op": True, "layout": {}}) == ()
    assert _dropped_extractor_kwargs(None) == ()


def test_sanitized_load_custom_objects_skips_the_deserialize_when_nothing_drops(tmp_path, monkeypatch):
    """The fast path is load-bearing, not a micro-optimization: a checkpoint trained at the CURRENT
    arch drops nothing, and it must not pay a full ~27 MB `load_from_zip_file` to learn that. So
    `None` custom_objects (SB3's own default) and an untouched zip."""
    from stable_baselines3.common import save_util
    from main.prober.model import sanitized_load_custom_objects

    accepted = _accepted_extractor_kwargs()
    assert accepted, "signature introspection failed — the sanitizer cannot be tested"
    keep = sorted(accepted)[0]

    def _boom(*a, **k):
        raise AssertionError("paid the full deserialize with nothing to drop")

    monkeypatch.setattr(save_util, "load_from_zip_file", _boom)
    assert sanitized_load_custom_objects(_fake_ckpt(tmp_path / "clean.zip", {keep: 1})) == (None, ())


def test_every_prober_checkpoint_load_goes_through_the_sanitizer():
    """REGRESSION: the prober rebuilds an extractor at TWO sites — `ProbeModel.load` and
    `replay_counterfactual`'s rollout-player loader — and a bare `MaskablePPO.load` at either one
    TypeErrors on any checkpoint written before a flag was deleted (measured over `models/`: 70 of
    89 runs carry at least one such kwarg). The counterfactual site was bare until this gate, which
    is why the failure only showed up on the rollout paths. A new load site must sanitize too.
    (Every load is `load_checkpoint_strict` since P10 follow-up F1; a bare `MaskablePPO.load` is a
    failure of `src/strict_checkpoint_load_gate_test.py`.)"""
    import pathlib
    import main.prober as pkg

    root = pathlib.Path(pkg.__file__).parent
    loads = [(p, i, line) for p in sorted(root.rglob("*.py")) if not p.name.endswith("_test.py")
             for i, line in enumerate(p.read_text().splitlines(), 1)
             if "load_checkpoint_strict(" in line and "import" not in line]
    assert len(loads) >= 2, f"the scan found {len(loads)} strict checkpoint loads in the prober"
    bare = [f"{p.relative_to(root)}:{i}" for p, i, line in loads if "custom_objects" not in line]
    assert not bare, ("checkpoint load without `custom_objects=` from "
                      f"`sanitized_load_custom_objects`: {bare}")


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

    def fake_load(path, *, device="cpu", custom_objects=None):
        seen["kwargs"] = custom_objects["policy_kwargs"]["features_extractor_kwargs"]
        return type("M", (), {"policy": _StubPolicy(_Ext(_Op(op_row)))})()

    monkeypatch.setattr(snapshot, "load_checkpoint_strict", fake_load)

    pm = ProbeModel.load(ckpt)
    assert "deleted_flag" not in seen["kwargs"] and keep in seen["kwargs"]
    assert pm.dropped_kwargs == ("deleted_flag",)


def test_load_turns_any_failure_into_a_diagnosis(tmp_path, monkeypatch):
    """Walls 2 and 3 (a value the code now rejects; weight shapes that no longer fit) are NOT
    recoverable — but they must still arrive as an ArchDriftError, with the original preserved."""
    ckpt = _fake_ckpt(tmp_path / "m.zip", {"whatever": 1})

    def boom(*a, **k):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (12x380 and 386x256)")

    monkeypatch.setattr(snapshot, "load_checkpoint_strict", boom)
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(ckpt)
    assert "cannot be re-run" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)   # the cause is chained, not swallowed


# ── EVERY incompatibility is ONE typed diagnosis (2026-10-09) ────────────────────────────────────────
# The owner: "it doesn't even detect the new error correctly". The classes below are the ways a checkpoint
# can be unusable under HEAD; each must arrive as an `ArchDriftError` carrying its `kind` and ONE plain
# sentence — never a raw StrictLoadError / RuntimeError / FileNotFoundError — and the ones decidable from
# the RECORD must be refused BEFORE the load (the silent class, "the weights fit but the observation changed
# meaning", would otherwise load fine and answer about inputs the model never trained on).

from agents.model.model_version import (  # noqa: E402
    ARCH_SIGNATURE, MODEL_CONFIG_VERSION, OBS_SEMANTICS_VERSION,
)
from agents.training.instrumented_ppo.strict_load import StrictLoadError  # noqa: E402
from main.prober import arch_status as A  # noqa: E402
from main.prober.model import NoCheckpointError  # noqa: E402

_HEAD = {"config_version": MODEL_CONFIG_VERSION, "arch_signature": ARCH_SIGNATURE}


def _run_ckpt(tmp_path, record, *, obs_end=None):
    """`<tmp>/run/m.zip` (a fake checkpoint at the current width unless told otherwise) beside the run's
    `model_config.json` (`record`; None = the run recorded nothing)."""
    run = tmp_path / "run"
    os.makedirs(run)
    if record is not None:
        (run / "model_config.json").write_text(json.dumps(record))
    return _fake_ckpt(run / "m.zip", {"k": 1}, obs_end=obs_end)


def _loader_must_not_run(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the loader was reached: the recorded identity should have refused this first")
    monkeypatch.setattr(snapshot, "load_checkpoint_strict", refuse)


def _loader_succeeds(monkeypatch, calls):
    class _StubPolicy(_Pol):
        def set_training_mode(self, mode):
            pass

        def modules(self):
            return []

    def ok(path, *, device="cpu", custom_objects=None):
        calls.append(path)
        return type("M", (), {"policy": _StubPolicy(_Ext(_Op(np.zeros(6 * 12 + 13, dtype=np.float32))))})()

    monkeypatch.setattr(snapshot, "load_checkpoint_strict", ok)


@pytest.mark.parametrize("record,obs_end,kind", [
    ({"config_version": 143, "arch_signature": "gen3_event_record_v2"}, None, A.KIND_ARCH_SIGNATURE),
    (dict(_HEAD), 2761, A.KIND_OBS_DIM),
    (dict(_HEAD, config_version=OBS_SEMANTICS_VERSION - 1), None, A.KIND_OBS_SEMANTICS),
    (dict(_HEAD, config_version=MODEL_CONFIG_VERSION + 1), None, A.KIND_NEWER),
])
def test_a_recorded_incompatibility_is_refused_typed_before_any_load(tmp_path, monkeypatch, record, obs_end, kind):
    ckpt = _run_ckpt(tmp_path, record, obs_end=obs_end)
    _loader_must_not_run(monkeypatch)
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(ckpt)
    e = ei.value
    assert e.kind == kind
    assert e.plain and str(e).startswith(e.plain), "the one plain sentence leads the full diagnosis"
    assert e.plain not in e.detail, "…and the card that folds the detail under it does not say it twice"
    assert "cannot be re-run under the current code" in str(e)


def test_the_semantics_only_class_is_caught_though_the_weights_would_load(tmp_path, monkeypatch):
    """THE SILENT CLASS, end to end: a checkpoint recorded one version below `OBS_SEMANTICS_VERSION` has the
    current signature and the current width, so every shape check passes and the (stubbed) strict load
    SUCCEEDS — yet it is refused, and the checkpoint one version later is not."""
    calls: list = []
    _loader_succeeds(monkeypatch, calls)
    below = _run_ckpt(tmp_path, dict(_HEAD, config_version=OBS_SEMANTICS_VERSION - 1))
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(below)
    assert ei.value.kind == A.KIND_OBS_SEMANTICS and not calls, "refused WITHOUT being loaded"
    assert f"v{OBS_SEMANTICS_VERSION}" in ei.value.plain

    (tmp_path / "ok").mkdir()
    at = _run_ckpt(tmp_path / "ok", dict(_HEAD, config_version=OBS_SEMANTICS_VERSION))
    assert isinstance(ProbeModel.load(at), ProbeModel) and calls == [at]


@pytest.mark.parametrize("raised,kind", [
    (StrictLoadError("the checkpoint's state dict does not match this model's (strict load): Missing key(s)"),
     A.KIND_STATE_DICT),
    (RuntimeError("Error(s) in loading state_dict for Gen3DualHeadMaskablePolicy: size mismatch for x"),
     A.KIND_STATE_DICT),
    (RuntimeError("mat1 and mat2 shapes cannot be multiplied (12x380 and 386x256)"), A.KIND_STATE_DICT),
    (TypeError("__init__() got an unexpected keyword argument 'deleted_flag'"), A.KIND_CONFIG_VALUE),
    (ValueError("move_candidate_floor=0.0 is not legal"), A.KIND_CONFIG_VALUE),
    (RuntimeError("something nobody anticipated"), A.KIND_LOAD_FAILED),
])
def test_a_failed_load_is_classified_never_raw(tmp_path, monkeypatch, raised, kind):
    """A run with NO record (so nothing to refuse it on) whose load then fails: the failure is mapped onto a
    kind by its evidence, the cause is chained, and the catch-all is `load_failed` — nothing escapes untyped."""
    ckpt = _run_ckpt(tmp_path, None)

    def boom(*a, **k):
        raise raised

    monkeypatch.setattr(snapshot, "load_checkpoint_strict", boom)
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(ckpt)
    assert ei.value.kind == kind and ei.value.plain
    assert ei.value.__cause__ is raised


def test_a_missing_or_unreadable_checkpoint_is_typed_not_a_raw_oserror(tmp_path, monkeypatch):
    _loader_must_not_run(monkeypatch)
    with pytest.raises(NoCheckpointError) as ei:
        ProbeModel.load(str(tmp_path / "gone.zip"))
    assert ei.value.kind == A.KIND_NO_CHECKPOINT and "no loadable checkpoint" in ei.value.plain
    assert isinstance(ei.value, ArchDriftError) and isinstance(ei.value, FileNotFoundError)

    empty = tmp_path / "empty.zip"
    empty.write_bytes(b"")
    with pytest.raises(ArchDriftError) as ei2:
        ProbeModel.load(str(empty))
    assert ei2.value.kind == A.KIND_UNREADABLE and not isinstance(ei2.value, NoCheckpointError)


def test_a_record_that_vouches_for_head_still_gets_a_typed_error_when_the_load_fails(tmp_path, monkeypatch):
    """The recorded identity is HEAD's, the weights are not what the code builds (a state_dict mismatch the
    record could not foresee): the pre-load check passes and the CLASSIFIER catches it, with the recorded
    version in the diagnosis rather than a guess."""
    ckpt = _run_ckpt(tmp_path, dict(_HEAD))

    def boom(*a, **k):
        raise StrictLoadError("Missing key(s) in state_dict")

    monkeypatch.setattr(snapshot, "load_checkpoint_strict", boom)
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(ckpt)
    assert ei.value.kind == A.KIND_STATE_DICT and ei.value.saved_version == MODEL_CONFIG_VERSION
