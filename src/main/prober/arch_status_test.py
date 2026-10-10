"""`arch_status` — every way a run can be incompatible with HEAD is ONE typed, plain-language verdict.

Pure (no torch, no checkpoint opened): the verdict is read from RECORDS (`model_config.json`, the eval manifest),
so these tests build constructed run directories. What they pin:

* each incompatibility class maps to its kind — including the SILENT one, a checkpoint whose weights fit and whose
  observation changed meaning (`OBS_SEMANTICS_VERSION`), which nothing but the recorded config version can see;
* the plain sentence says what the owner asked for, in those words;
* the picker's cheap per-run classification (cached by mtime) and the tier it earns.
"""
import json
import os
import time

import pytest

from agents.model.model_version import (
    ARCH_SIGNATURE,
    MIGRATION_FLOOR,
    MODEL_CONFIG_VERSION,
    OBS_SEMANTICS_VERSION,
)
from main.prober import arch_status as A


def _cur() -> dict:
    return {"config_version": MODEL_CONFIG_VERSION, "arch_signature": ARCH_SIGNATURE,
            "total_dim": A.current_obs_dim()}


def _write(path, doc) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


# -- the verdict from a record ---------------------------------------------------------------------------

def test_a_record_at_head_is_current():
    v = A.verdict_from_record(MODEL_CONFIG_VERSION, ARCH_SIGNATURE, A.current_obs_dim())
    assert v.status == A.CURRENT and v.kind is None and v.current and v.model_views


def test_the_first_version_with_the_new_observation_meaning_is_current_and_the_one_before_is_not():
    """THE SILENT CLASS. v150 and v151 share a signature, a state_dict and an observation WIDTH; they differ in
    what the Toxic cell means. Only the recorded version tells them apart, and the marker is the boundary — so the
    check is pinned at exactly that line, both sides."""
    assert OBS_SEMANTICS_VERSION > MIGRATION_FLOOR, (
        "precondition: a same-signature version BELOW the marker must exist to test the boundary; if the floor "
        "caught up with the marker (a new ARCH_SIGNATURE), this class has nothing to say — rewrite the test")
    at = A.verdict_from_record(OBS_SEMANTICS_VERSION, ARCH_SIGNATURE, A.current_obs_dim())
    below = A.verdict_from_record(OBS_SEMANTICS_VERSION - 1, ARCH_SIGNATURE, A.current_obs_dim())
    assert at.status == A.CURRENT
    assert below.status == A.INCOMPATIBLE and below.kind == A.KIND_OBS_SEMANTICS
    assert not below.model_views
    assert f"v{OBS_SEMANTICS_VERSION}" in below.plain and "never trained on" in below.plain


@pytest.mark.parametrize("version,signature,obs_dim,kind", [
    (143, "gen3_event_record_v2", 2761, A.KIND_ARCH_SIGNATURE),           # the rb_st_* runs: another network family
    (MODEL_CONFIG_VERSION, "gen3_something_else_v9", None, A.KIND_ARCH_SIGNATURE),
    (MIGRATION_FLOOR - 1, None, None, A.KIND_ARCH_SIGNATURE),             # below the floor, signature unrecorded
    (MODEL_CONFIG_VERSION, ARCH_SIGNATURE, 2761, A.KIND_OBS_DIM),         # same family, another observation width
    (MODEL_CONFIG_VERSION + 1, ARCH_SIGNATURE, None, A.KIND_NEWER),       # recorded by newer code than this
    (OBS_SEMANTICS_VERSION - 1, ARCH_SIGNATURE, None, A.KIND_OBS_SEMANTICS),
])
def test_every_record_class_maps_to_its_kind(version, signature, obs_dim, kind):
    v = A.verdict_from_record(version, signature, obs_dim)
    assert v.status == A.INCOMPATIBLE and v.kind == kind and v.kind in A.KINDS
    assert v.plain and v.plain == A.plain_sentence(kind, config_version=version, arch_signature=signature,
                                                   obs_dim=obs_dim)


def test_the_deepest_cause_wins_when_a_record_is_wrong_several_ways():
    """Family first, then newer-than-code, then width, then meaning: the report names what is most
    fundamentally wrong, not whichever check happened to run first."""
    v = A.verdict_from_record(143, "gen3_event_record_v2", 2761)
    assert v.kind == A.KIND_ARCH_SIGNATURE            # not obs_dim, though the width differs too
    v = A.verdict_from_record(OBS_SEMANTICS_VERSION - 1, ARCH_SIGNATURE, 2761)
    assert v.kind == A.KIND_OBS_DIM                   # not obs_semantics


def test_a_record_with_nothing_is_unrecorded_never_current_and_never_incompatible():
    v = A.verdict_from_record(None, None, None)
    assert v.status == A.UNRECORDED and v.model_views and not v.current
    sig_only = A.verdict_from_record(None, ARCH_SIGNATURE, None)
    assert sig_only.status == A.UNRECORDED, "a matching signature cannot vouch for the observation's MEANING"


def test_the_older_sentence_is_the_one_the_owner_asked_for():
    v = A.verdict_from_record(143, "gen3_event_record_v2", 2761)
    assert v.plain.startswith("This run's architecture (config v143, signature gen3_event_record_v2) is older "
                              f"than the code (config v{MODEL_CONFIG_VERSION}, signature {ARCH_SIGNATURE}); "
                              "model views need a current-architecture checkpoint.")


def test_every_kind_has_a_plain_sentence_and_the_drift_ones_share_the_opening():
    for kind in A.KINDS:
        s = A.plain_sentence(kind, config_version=143, arch_signature="gen3_event_record_v2", obs_dim=2761,
                             cause="x")
        assert s.endswith("."), (kind, s)
        if kind in (A.KIND_ARCH_SIGNATURE, A.KIND_OBS_DIM, A.KIND_OBS_SEMANTICS, A.KIND_STATE_DICT,
                    A.KIND_CONFIG_VALUE, A.KIND_LOAD_FAILED):
            assert "model views need a current-architecture checkpoint" in s, kind
    assert "no loadable checkpoint" in A.plain_sentence(A.KIND_NO_CHECKPOINT)
    assert "NEWER" in A.plain_sentence(A.KIND_NEWER, config_version=MODEL_CONFIG_VERSION + 1)


# -- reading the records ----------------------------------------------------------------------------------

def test_a_steps_manifest_outranks_the_run_level_config(tmp_path):
    run = str(tmp_path / "run")
    _write(os.path.join(run, "model_config.json"), _cur())
    _write(os.path.join(run, "eval_traces", "step_4000", "eval_manifest.json"),
           {"config_version": 143, "arch_signature": "gen3_event_record_v2"})
    v = A.verdict_for_step(run, 4000)
    assert v.kind == A.KIND_ARCH_SIGNATURE and v.source == "eval_manifest" and v.config_version == 143


def test_a_manifest_without_a_version_falls_back_to_the_run_config(tmp_path):
    run = str(tmp_path / "run")
    _write(os.path.join(run, "model_config.json"), _cur())
    _write(os.path.join(run, "eval_traces", "step_4000", "eval_manifest.json"), {"step": 4000})
    v = A.verdict_for_step(run, 4000)
    assert v.current and v.source == "model_config"


def test_the_newest_step_is_the_default_step(tmp_path):
    run = str(tmp_path / "run")
    _write(os.path.join(run, "eval_traces", "step_2000", "eval_manifest.json"),
           {"config_version": 143, "arch_signature": "gen3_event_record_v2"})
    _write(os.path.join(run, "eval_traces", "step_30000", "eval_manifest.json"),
           {"config_version": MODEL_CONFIG_VERSION, "arch_signature": ARCH_SIGNATURE})
    assert [s for s, _ in A.step_dirs(run)] == [2000, 30000]            # numeric, not lexicographic
    assert A.verdict_for_step(run).current


def test_a_run_with_no_record_at_all_is_unrecorded(tmp_path):
    run = str(tmp_path / "run")
    os.makedirs(run)
    assert A.verdict_for_step(run).status == A.UNRECORDED


def test_has_traces_means_a_step_directory_not_an_empty_eval_traces(tmp_path):
    a, b, c = (str(tmp_path / n) for n in "abc")
    os.makedirs(os.path.join(a, "eval_traces", "step_100"))
    os.makedirs(os.path.join(b, "eval_traces"))                          # exists, holds nothing
    os.makedirs(os.path.join(c, "eval_traces", "not_a_step"))
    assert A.run_has_traces(a) is True
    assert A.run_has_traces(b) is False and A.run_has_traces(c) is False
    assert A.run_has_traces(str(tmp_path / "missing")) is False


def test_the_tier_is_the_pickers_three_way_split(tmp_path):
    cur = A.verdict_from_record(MODEL_CONFIG_VERSION, ARCH_SIGNATURE, A.current_obs_dim())
    old = A.verdict_from_record(143, "gen3_event_record_v2", 2761)
    unk = A.verdict_from_record(None, None)
    assert A.tier_of(True, cur) == A.TIER_CURRENT
    assert A.tier_of(True, old) == A.TIER_OLDER
    assert A.tier_of(True, unk) == A.TIER_OLDER, "an unrecorded run is never counted as current"
    for v in (cur, old, unk):
        assert A.tier_of(False, v) == A.TIER_NO_TRACES


def test_the_run_verdict_is_cached_by_mtime_and_rereads_when_the_run_is_resaved(tmp_path):
    run = str(tmp_path / "run")
    cfg = os.path.join(run, "model_config.json")
    _write(cfg, {"config_version": 143, "arch_signature": "gen3_event_record_v2"})
    os.makedirs(os.path.join(run, "eval_traces", "step_1"))
    first = A.run_verdict(run)
    assert first.kind == A.KIND_ARCH_SIGNATURE
    assert A.run_verdict(run) is first, "an untouched run must be served from the cache"
    _write(cfg, _cur())
    later = time.time() + 5
    os.utime(cfg, (later, later))                                        # a re-save moves model_config's mtime
    again = A.run_verdict(run)
    assert again is not first and again.current


# -- a checkpoint file ------------------------------------------------------------------------------------

def test_a_missing_checkpoint_is_no_checkpoint_and_a_non_zip_is_unreadable(tmp_path):
    v = A.checkpoint_verdict(str(tmp_path / "gone.zip"))
    assert v.status == A.INCOMPATIBLE and v.kind == A.KIND_NO_CHECKPOINT
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"")
    u = A.checkpoint_verdict(str(bad))
    assert u.kind == A.KIND_UNREADABLE and not u.model_views


def _zip(path) -> str:
    import zipfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", "{}")
    return str(path)


def test_an_eval_snapshot_is_judged_by_the_manifest_beside_it(tmp_path):
    run = tmp_path / "run"
    _write(str(run / "model_config.json"), _cur())                       # the run root says current…
    snap = _zip(str(run / "eval_traces" / "step_9" / "snapshot.zip"))
    _write(str(run / "eval_traces" / "step_9" / "eval_manifest.json"),
           {"config_version": 143, "arch_signature": "gen3_event_record_v2"})    # …the cycle's own record doesn't
    assert A.checkpoint_verdict(snap).kind == A.KIND_ARCH_SIGNATURE


def test_a_periodic_checkpoint_is_judged_by_the_run_config_three_levels_up(tmp_path):
    run = tmp_path / "run"
    _write(str(run / "model_config.json"), {**_cur(), "config_version": OBS_SEMANTICS_VERSION - 1})
    ckpt = _zip(str(run / "checkpoints" / "checkpoint_1_steps.zip"))
    v = A.checkpoint_verdict(ckpt)
    assert v.kind == A.KIND_OBS_SEMANTICS and v.source == "model_config"


def test_the_width_the_zip_declares_outranks_the_sidecars(tmp_path):
    run = tmp_path / "run"
    _write(str(run / "model_config.json"), _cur())
    ckpt = _zip(str(run / "m.zip"))
    assert A.checkpoint_verdict(ckpt).current
    assert A.checkpoint_verdict(ckpt, peek_obs_dim=2761).kind == A.KIND_OBS_DIM


def test_the_worse_verdict_wins_and_a_recorded_reason_outranks_the_state_of_the_file():
    cur = A.verdict_from_record(MODEL_CONFIG_VERSION, ARCH_SIGNATURE, A.current_obs_dim())
    unk = A.verdict_from_record(None, None)
    old = A.verdict_from_record(143, "gen3_event_record_v2", 2761)
    gone = A.no_checkpoint_verdict("x.zip")
    assert A.worst(cur, unk) is unk and A.worst(unk, cur) is unk
    assert A.worst(cur, old) is old and A.worst(old, cur) is old
    assert A.worst(cur, cur) is cur, "a tie keeps the first"
    # two incompatible verdicts: the recorded reason explains more than the missing/unreadable file…
    assert A.worst(gone, old) is old and A.worst(old, gone) is old
    # …and two recorded reasons keep the first
    sem = A.verdict_from_record(OBS_SEMANTICS_VERSION - 1, ARCH_SIGNATURE, None)
    assert A.worst(old, sem) is old
