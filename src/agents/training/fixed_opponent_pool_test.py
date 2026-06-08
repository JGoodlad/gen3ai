import dataclasses
import json
import os
import tempfile

import pytest

from agents.model.model_version import ModelVersion, ModelVersionError
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.fixed_opponent_pool import (
    EXT_PREFIX,
    is_external,
    parse_stable_opponents,
    resolve_stable_opponents,
)


@pytest.fixture(scope="module")
def version():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})


def _write_run(tmpdir, version, *, name="ai_v5_5_popart_N_0607", zip_in_best_model=True):
    """Create a fake run dir: model_config.json (top level) + a (touched) weights zip."""
    run_dir = os.path.join(tmpdir, name)
    if zip_in_best_model:
        os.makedirs(os.path.join(run_dir, "best_model"))
        zip_path = os.path.join(run_dir, "best_model", "best_model.zip")
    else:
        os.makedirs(run_dir)
        zip_path = os.path.join(run_dir, "final_model.zip")
    open(zip_path, "w").close()  # the resolver only reads the config, not the weights
    with open(os.path.join(run_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())
    return run_dir


# ---------------------------------------------------------------------------
# parse_stable_opponents
# ---------------------------------------------------------------------------

def test_parse_path_only():
    specs = parse_stable_opponents("/models/run_a")
    assert specs == [{"path": "/models/run_a", "step": None, "label": None}]


def test_parse_full_grammar():
    specs = parse_stable_opponents("/models/run_a@500:champ")
    assert specs == [{"path": "/models/run_a", "step": 500, "label": "champ"}]


def test_parse_multiple_and_whitespace():
    specs = parse_stable_opponents(" /a@10 , /b:foo , ")
    assert len(specs) == 2
    assert specs[0]["path"] == "/a" and specs[0]["step"] == 10
    assert specs[1]["path"] == "/b" and specs[1]["label"] == "foo"


def test_parse_weight_rejected():
    """Per-opponent weights are not supported yet (training-mix is Stage 2) — rejected, not ignored."""
    with pytest.raises(ValueError, match="weights"):
        parse_stable_opponents("/a=2")


def test_parse_bad_step_raises():
    with pytest.raises(ValueError, match="step"):
        parse_stable_opponents("/a@notanint")


def test_is_external():
    assert is_external("ext_foo")
    assert not is_external("heuristic2")
    assert not is_external("sentinel_0")
    assert not is_external("external")  # the aggregate key is NOT an external label


# ---------------------------------------------------------------------------
# resolve_stable_opponents — the FATAL arch gate
# ---------------------------------------------------------------------------

def test_resolve_none_and_empty(version):
    assert resolve_stable_opponents(None, version) == []
    assert resolve_stable_opponents("", version) == []


def test_resolve_same_arch_run_dir(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version)
        entries = resolve_stable_opponents(run, version)
    assert len(entries) == 1
    e = entries[0]
    assert e.label == "ext_ai_v5_5_popart_N_0607"
    assert e.arch_signature == version.arch_signature
    assert e.zip_path.endswith("best_model/best_model.zip")


def test_resolve_arch_mismatch_is_fatal(version):
    """A different arch_signature (= different obs family) must raise ModelVersionError —
    the caller surfaces this as a startup FATAL."""
    foreign = dataclasses.replace(version, arch_signature="gen3_some_old_arch_v1")
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, foreign, name="old_run")
        with pytest.raises(ModelVersionError) as exc:
            resolve_stable_opponents(run, version)
    assert "gen3_some_old_arch_v1" in str(exc.value)
    assert "observation layout" in str(exc.value)


def test_resolve_label_and_temp(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="run_x")
        entries = resolve_stable_opponents(f"{run}:champ", version, default_temperature=0.7)
    e = entries[0]
    assert e.label == "ext_champ"      # user label gets the ext_ prefix
    assert e.temperature == 0.7


def test_resolve_final_model_zip(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="run_fm", zip_in_best_model=False)
        entries = resolve_stable_opponents(run, version)
    assert entries[0].zip_path.endswith("final_model.zip")


def test_resolve_missing_zip_raises(version):
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, "empty_run")
        os.makedirs(empty)
        with open(os.path.join(empty, "model_config.json"), "w") as f:
            f.write(version.to_json())
        with pytest.raises(FileNotFoundError, match="no model .zip"):
            resolve_stable_opponents(empty, version)


def test_resolve_duplicate_label_raises(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="dup_run")
        with pytest.raises(ValueError, match="duplicate label"):
            resolve_stable_opponents(f"{run}:same,{run}:same", version)


def test_resolve_config_in_parent_of_zip(version):
    """A direct best_model/best_model.zip path finds the run-level model_config.json (parent)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="run_parent")
        zip_path = os.path.join(run, "best_model", "best_model.zip")
        entries = resolve_stable_opponents(zip_path, version)
    assert len(entries) == 1
    assert entries[0].arch_signature == version.arch_signature


def test_resolve_label_is_run_name_not_best_model(version):
    """A direct best_model/best_model.zip path is labelled by the RUN name, NOT 'best_model'
    (the namespace folder the zip lives in)."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="my_cool_run")
        zip_path = os.path.join(run, "best_model", "best_model.zip")
        entries = resolve_stable_opponents(zip_path, version)
    assert entries[0].label == "ext_my_cool_run"


def test_resolve_prefers_colocated_best_model_config(version):
    """When best_model/ has its OWN model_config.json (the unified location), it is used."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="run_unified")
        # Co-locate the sidecar in best_model/ (what copy_run_config_to_best_model does).
        with open(os.path.join(run, "best_model", "model_config.json"), "w") as f:
            f.write(version.to_json())
        entries = resolve_stable_opponents(os.path.join(run, "best_model", "best_model.zip"), version)
    assert entries[0].config_path.endswith("best_model/model_config.json")
    assert entries[0].label == "ext_run_unified"


def test_copy_run_config_to_best_model(version):
    """copy_run_config_to_best_model makes best_model/ self-contained (zip + sidecar co-located)."""
    from agents.training.eval_callback import copy_run_config_to_best_model
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "a_run")
        best = os.path.join(run, "best_model")
        os.makedirs(best)
        with open(os.path.join(run, "model_config.json"), "w") as f:
            f.write(version.to_json())
        copy_run_config_to_best_model(run, best)
        assert os.path.isfile(os.path.join(best, "model_config.json"))
    # No-op (no raise) when there's nothing to copy.
    with tempfile.TemporaryDirectory() as tmp:
        copy_run_config_to_best_model(tmp, os.path.join(tmp, "best_model"))  # no run config → silent
    copy_run_config_to_best_model(None, None)  # None args → silent


# ── source_elo: the opponent's OWN recorded ELO (best_model.json sidecar / metadata fallback) ──

def _write_run_colocated(tmp, version, *, name, sidecar_elo=None, run_meta_elo=None):
    """A run dir with the config CO-LOCATED in best_model/ (the real backfilled layout), optionally
    with a best_model/best_model.json sidecar and/or a run-level metadata.json carrying an elo."""
    run = os.path.join(tmp, name)
    bm = os.path.join(run, "best_model")
    os.makedirs(bm)
    open(os.path.join(bm, "best_model.zip"), "w").close()
    with open(os.path.join(bm, "model_config.json"), "w") as f:
        f.write(version.to_json())
    if sidecar_elo is not None:
        with open(os.path.join(bm, "best_model.json"), "w") as f:
            json.dump({"lr": 1e-4, "latest_eval": {"elo": sidecar_elo}}, f)
    if run_meta_elo is not None:
        with open(os.path.join(run, "metadata.json"), "w") as f:
            json.dump({"latest_eval": {"elo": run_meta_elo}}, f)
    return run


def test_source_elo_prefers_best_model_sidecar(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run_colocated(tmp, version, name="r", sidecar_elo=1850.0, run_meta_elo=1700.0)
        e = resolve_stable_opponents(run, version)[0]
    assert e.source_elo == pytest.approx(1850.0)   # best_model.json wins over run-level metadata


def test_source_elo_falls_back_to_run_metadata(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run_colocated(tmp, version, name="r", sidecar_elo=None, run_meta_elo=1700.0)
        e = resolve_stable_opponents(run, version)[0]
    assert e.source_elo == pytest.approx(1700.0)


def test_source_elo_none_when_absent(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run_colocated(tmp, version, name="r")   # no elo anywhere
        e = resolve_stable_opponents(run, version)[0]
    assert e.source_elo is None
