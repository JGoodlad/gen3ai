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
    register_exploiter_for_eval,
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


def _write_run_with_step_checkpoint(tmpdir, version, *, name, step):
    """A run whose @step checkpoint lives in <run>/checkpoints/ (current layout)."""
    run_dir = os.path.join(tmpdir, name)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    open(os.path.join(run_dir, "checkpoints", f"checkpoint_{step}_steps.zip"), "w").close()
    with open(os.path.join(run_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())
    return run_dir


def test_resolve_at_step_label_is_run_name_not_checkpoints(version):
    """@step now resolves under <run>/checkpoints/ — the DEFAULT label must still be the run
    NAME (ext_<run>@<step>), not the literal subfolder ext_checkpoints@<step>."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run_with_step_checkpoint(tmp, version, name="run_at_step", step=3200000)
        entries = resolve_stable_opponents(f"{run}@3200000", version)
    assert len(entries) == 1
    assert entries[0].label == "ext_run_at_step@3200000"
    assert entries[0].zip_path.endswith("checkpoints/checkpoint_3200000_steps.zip")


def test_resolve_two_runs_same_step_do_not_collide(version):
    """Two DIFFERENT runs at the same @step must keep distinct labels (run names) — the old
    mis-derived ext_checkpoints@<step> would collide and trip the duplicate-label FATAL."""
    with tempfile.TemporaryDirectory() as tmp:
        run_a = _write_run_with_step_checkpoint(tmp, version, name="run_a", step=50000)
        run_b = _write_run_with_step_checkpoint(tmp, version, name="run_b", step=50000)
        entries = resolve_stable_opponents(f"{run_a}@50000,{run_b}@50000", version)
    labels = sorted(e.label for e in entries)
    assert labels == ["ext_run_a@50000", "ext_run_b@50000"]


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


# ---------------------------------------------------------------------------
# fold-back: the opponent's OWN pinned team (_read_trainee_pin via resolve)
# ---------------------------------------------------------------------------

def _write_pin_run(tmp, version, *, name="spec_run", pin_content="Skarmory @ Leftovers\n",
                   record_sha=True, delete_pin_file=False, sha_override=None):
    """A fake SPECIALIST run: model_config + weights zip + metadata.json recording a
    --trainee-team pin (and optionally the MatchupSpec pin_sha fingerprint)."""
    import hashlib
    run = _write_run(tmp, version, name=name)
    pin_file = os.path.join(tmp, f"{name}_team.txt")
    with open(pin_file, "w") as f:
        f.write(pin_content)
    sha = sha_override or hashlib.sha1(pin_content.encode()).hexdigest()[:10]
    meta = {"cli_args": {"trainee_team": pin_file}}
    if record_sha:
        meta["cli_args"]["_matchup_spec"] = {"trainee_teams": {"pin_sha": sha}}
    with open(os.path.join(run, "metadata.json"), "w") as f:
        json.dump(meta, f)
    if delete_pin_file:
        os.remove(pin_file)
    return run, pin_file


def test_resolve_reads_opponent_pin_from_metadata(version):
    with tempfile.TemporaryDirectory() as tmp:
        run, pin_file = _write_pin_run(tmp, version, pin_content="Blissey @ Leftovers\n")
        e = resolve_stable_opponents(run, version)[0]
    assert e.team_str == "Blissey @ Leftovers\n"
    assert e.team_file == pin_file
    assert e.to_cfg()["team_str"] == e.team_str   # threaded to the eval worker


def test_resolve_no_pin_is_none(version):
    with tempfile.TemporaryDirectory() as tmp:
        run = _write_run(tmp, version, name="generalist")   # no metadata at all
        e = resolve_stable_opponents(run, version)[0]
    assert e.team_str is None and e.team_file is None


def test_resolve_missing_pin_file_fails_loud(version):
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _write_pin_run(tmp, version, delete_pin_file=True)
        with pytest.raises(FileNotFoundError, match="no longer exists"):
            resolve_stable_opponents(run, version)


def test_resolve_pin_sha_mismatch_fails_loud(version):
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _write_pin_run(tmp, version, sha_override="badf00dbad")
        with pytest.raises(ValueError, match="pin_sha"):
            resolve_stable_opponents(run, version)


def test_resolve_pin_without_sha_is_accepted(version):
    # pre-MatchupSpec specialist runs (ai_v7_07/08/09) record only cli_args.trainee_team.
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _write_pin_run(tmp, version, record_sha=False)
        e = resolve_stable_opponents(run, version)[0]
    assert e.team_str == "Skarmory @ Leftovers\n"


# ---------------------------------------------------------------------------
# exploiter auto-eval registration (opponent-parity Proposal A)
# ---------------------------------------------------------------------------

def _entry(label, zip_path):
    from agents.training.fixed_opponent_pool import FixedOpponentEntry
    return FixedOpponentEntry(label=label, zip_path=zip_path, config_path="c.json",
                              arch_signature="sig")


def test_exploiter_auto_registers_for_eval():
    fixed, appended = register_exploiter_for_eval([], _entry("ext_target", "/m/t.zip"))
    assert appended and [e.label for e in fixed] == ["ext_target"]


def test_exploiter_registration_dedups_same_zip():
    # the historical both-flags recipe (--exploiter X --stable-opponents X) stays byte-identical
    explicit = [_entry("ext_target", "/m/t.zip")]
    fixed, appended = register_exploiter_for_eval(explicit, _entry("ext_other_label", "/m/t.zip"))
    assert not appended and fixed is explicit


def test_exploiter_registration_dedups_label_collision():
    explicit = [_entry("ext_target", "/m/other.zip")]
    fixed, appended = register_exploiter_for_eval(explicit, _entry("ext_target", "/m/t.zip"))
    assert not appended and fixed is explicit


def test_exploiter_registration_none_is_noop():
    fixed, appended = register_exploiter_for_eval([], None)
    assert not appended and fixed == []


def _write_multi_pin_run(tmp, version, *, name="multi_run", contents=("A @ Leftovers\n", "B @ Leftovers\n")):
    """A fake MULTI-team specialist run (--trainee-teams / pin_multi) + its pin_shas fingerprints."""
    import hashlib
    run = _write_run(tmp, version, name=name)
    files = []
    for i, c in enumerate(contents):
        f = os.path.join(tmp, f"{name}_t{i}.txt")
        with open(f, "w") as fh:
            fh.write(c)
        files.append(f)
    meta = {"cli_args": {"trainee_teams": ",".join(files)},
            "matchup_history": [{"spec": {"trainee_teams": {
                "kind": "pin_multi",
                "pin_shas": [hashlib.sha1(c.encode()).hexdigest()[:10] for c in contents]}}}]}
    with open(os.path.join(run, "metadata.json"), "w") as f:
        json.dump(meta, f)
    return run, files, list(contents)


def test_resolve_reads_a_MULTI_team_opponent_pin(version):
    """The fold-back contract for a multi-team z-cluster exploiter: it must carry ALL its teams, so
    as an opponent it samples among them (piloting the shared pool would evaporate its pressure)."""
    with tempfile.TemporaryDirectory() as tmp:
        run, files, contents = _write_multi_pin_run(tmp, version)
        e = resolve_stable_opponents(run, version)[0]
    assert list(e.team_strs) == contents          # ALL teams carried
    assert list(e.team_files) == files
    assert e.team_str == contents[0]              # back-compat mirror of element 0
    assert e.to_cfg()["team_strs"] == contents    # and it reaches the eval worker


def test_multi_pin_sha_mismatch_fails_loud(version):
    with tempfile.TemporaryDirectory() as tmp:
        run, files, _ = _write_multi_pin_run(tmp, version)
        with open(files[1], "w") as f:            # a member changed since that run trained on it
            f.write("MUTATED @ Leftovers\n")
        with pytest.raises(ValueError, match="pin_sha"):
            resolve_stable_opponents(run, version)
