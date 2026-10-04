"""`--oracle-reveal` (gen3_oracle_reveal_v1, config v137): the RECORD, the RESUME GATE and the offline REFUSALS.

The reveal itself is the Rust encoder's (`src/rust_env/tests/oracle_reveal_test.rs`, the differential over real
battles; `src/utils/rust_env/oracle_reveal_integration_test.py`, the real core read from Python). This file pins
the Python plumbing a wrong default would break silently:

* the mode is RECORDED in `model_config.json` (a fresh save writes it; a pre-v137 config migrates to `off`);
* a resume that FLIPS it is refused (`check_oracle_reveal`), a matching one passes, and `check_compatible` — which
  gates every load incl. a frozen opponent of the same run — does NOT read it (resume-immutable class);
* an offline tool that builds its observations without the reveal REFUSES an oracle checkpoint, by name;
* the extractor stores the mode (inert in the forward) and rejects an unknown level.
"""
import dataclasses
import json

import pytest

from agents.model import flag_registry as FR
from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config
from agents.model.oracle_reveal import OracleRevealRefused, recorded_oracle_reveal, refuse_if_revealed
from agents.model.snapshot import save_model_snapshot
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def version():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})


def test_the_flag_is_one_resume_immutable_registry_row_off_by_default():
    f = FR.BY_NAME["oracle_reveal"]
    assert (f.default, f.klass, f.tier, f.cli_flag) == ("off", FR.Klass.RESUME_IMMUTABLE, FR.Tier.CLI, "--oracle-reveal")
    assert f.since == MODEL_CONFIG_VERSION, "the field arrived with the config version that is current"
    assert FR.is_enabled("species") and not FR.is_enabled("off")
    assert f not in FR.arch_surface_flags(), "a diagnostic observation mode is never on the production ARCH surface"


def test_a_fresh_save_records_the_mode_and_the_round_trip_keeps_it(tmp_path, version):
    v = dataclasses.replace(version, oracle_reveal="species")
    assert ModelVersion.from_json_file  # the reader exists
    save_model_snapshot(str(tmp_path), v, git_hash="oracle-test")
    written = json.loads((tmp_path / "model_config.json").read_text())
    assert written["oracle_reveal"] == "species"
    assert ModelVersion.from_json_file(str(tmp_path / "model_config.json")).oracle_reveal == "species"
    assert recorded_oracle_reveal(str(tmp_path)) == "species"


def test_a_pre_v137_config_migrates_to_off():
    out = _migrate_config({"config_version": 136})
    assert out["oracle_reveal"] == "off" and out["config_version"] >= 137


def test_the_resume_gate_refuses_a_flip_names_both_modes_and_passes_a_match(version):
    saved = dataclasses.replace(version, oracle_reveal="species")
    saved.check_oracle_reveal("species")                        # a match
    with pytest.raises(ModelVersionError) as e:
        saved.check_oracle_reveal("off")                        # the flip back
    msg = str(e.value)
    assert "oracle_reveal mismatch" in msg and "'species'" in msg and "'off'" in msg and "--oracle-reveal species" in msg
    with pytest.raises(ModelVersionError):
        version.check_oracle_reveal("species")                   # off -> species too


def test_check_compatible_does_not_gate_a_frozen_opponent_on_the_mode(version):
    """Resume-immutable, not structural: a pool snapshot of the SAME run loads whatever the requested mode."""
    version.check_compatible(dataclasses.replace(version, oracle_reveal="species"))
    dataclasses.replace(version, oracle_reveal="species").check_compatible(version)


def test_an_offline_tool_refuses_an_oracle_checkpoint_by_name_and_accepts_a_production_one(tmp_path, version):
    run = tmp_path / "rb_x5ab_oracle_sp_s1"
    run.mkdir()
    (run / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(run), dataclasses.replace(version, oracle_reveal="species"), git_hash="t")
    with pytest.raises(OracleRevealRefused) as e:
        refuse_if_revealed(str(run / "final_model.zip"), tool="main.h2h", reason="The engine's eval cores are built at off.")
    assert "main.h2h" in str(e.value) and "species" in str(e.value) and "deferred" in str(e.value)
    plain = tmp_path / "rb_plain"
    plain.mkdir()
    (plain / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(plain), version, git_hash="t")
    refuse_if_revealed(str(plain / "final_model.zip"), tool="main.h2h")   # off: fine
    assert recorded_oracle_reveal(str(tmp_path / "no_such_run.zip")) == "off", "a legacy / missing config reads off"


def test_a_corrupt_recorded_mode_is_refused_not_read_as_off(tmp_path, version):
    (tmp_path / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(tmp_path), version, git_hash="t")
    cfg = tmp_path / "model_config.json"
    data = json.loads(cfg.read_text())
    data["oracle_reveal"] = "everything"
    cfg.write_text(json.dumps(data))
    with pytest.raises(OracleRevealRefused, match="not one of"):
        recorded_oracle_reveal(str(tmp_path / "final_model.zip"))


def test_the_h2h_host_load_takes_the_refusal(tmp_path, version):
    """The refusal is wired into the engine's checkpoint load, before any weight is read."""
    from main.h2h.arch import _load_host

    run = tmp_path / "oracle_run"
    run.mkdir()
    (run / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(run), dataclasses.replace(version, oracle_reveal="species"), git_hash="t")

    class Ref:
        zip_path = str(run / "final_model.zip")

    with pytest.raises(OracleRevealRefused, match="main.h2h"):
        _load_host(Ref())
