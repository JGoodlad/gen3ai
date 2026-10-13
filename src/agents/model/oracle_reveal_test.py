"""`--oracle-reveal` (gen3_oracle_reveal_v1, config v137): the RECORD, the RESUME GATE and the offline REFUSALS.

The reveal itself is the Rust encoder's (`src/rust_env/tests/oracle_reveal_test.rs`, the differential over real
battles; `src/utils/rust_env/oracle_reveal_integration_test.py`, the real core read from Python). This file pins
the Python plumbing a wrong default would break silently:

* the mode is RECORDED in `model_config.json` (a fresh save writes it; a pre-v137 config migrates to `off`);
* a resume that FLIPS it is refused (`check_oracle_reveal`), a matching one passes, and `check_compatible` — which
  gates every load incl. a frozen opponent of the same run — does NOT read it (resume-immutable class);
* an offline tool that builds its observations without the reveal REFUSES an oracle checkpoint, by name (`main.h2h`
  plays one through its per-side reveal and refuses it under its `off` mode);
* the extractor stores the mode (inert in the forward) and rejects an unknown level;
* the information-lever spec (``designs/endstate/design_information_levers.md`` §6): the flag takes any spec the
  training observation honours and records its LABEL (``ki_moves`` -> ``moves:seen@0.95``; the X32 levels keep their
  names whichever text they were given as), refuses every lever it cannot honour BY NAME, and the resume gate compares
  MEANING (``full`` vs its clause text is not a flip).
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
    # The field arrived with config v137 (a later bump — v138, `policy_readout` — must not move it).
    assert f.since == 137 <= MODEL_CONFIG_VERSION, "the field arrived with config v137"
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


def test_a_pre_v137_config_is_refused_and_a_recorded_mode_migrates_verbatim():
    """The v137 branch (a pre-v137 config defaults to `off`) is unreachable since the X5 version break raised
    MIGRATION_FLOOR to 144: such a config is refused at the floor. A RECORDED mode migrates verbatim."""
    from agents.model.model_version.migrations import MIGRATION_FLOOR
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 136})
    for mode in ("off", "species", "full"):
        out = _migrate_config({"config_version": MIGRATION_FLOOR, "oracle_reveal": mode})
        assert out["oracle_reveal"] == mode and out["config_version"] >= MIGRATION_FLOOR


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
        refuse_if_revealed(str(run / "final_model.zip"), tool="main.anchors", reason="The session builds without it.")
    assert "main.anchors" in str(e.value) and "species" in str(e.value) and "--oracle-reveal-mode" in str(e.value)
    plain = tmp_path / "rb_plain"
    plain.mkdir()
    (plain / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(plain), version, git_hash="t")
    refuse_if_revealed(str(plain / "final_model.zip"), tool="main.anchors")   # off: fine
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


def test_the_h2h_engine_refuses_an_oracle_checkpoint_under_its_off_mode_before_loading(tmp_path, version):
    """`main.h2h` no longer takes `refuse_if_revealed` (it PLAYS an oracle checkpoint through its per-side reveal,
    `main.h2h.reveal`), but under its default `off` mode it refuses one itself, typed, before any weight is read
    (the rest of the per-side matrix: `src/main/h2h/reveal_test.py`)."""
    from main.h2h.arch import declare_engine
    from main.h2h.reveal import RevealModeError

    run = tmp_path / "oracle_run"
    run.mkdir()
    (run / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(run), dataclasses.replace(version, oracle_reveal="species"), git_hash="t")

    class Ref:
        id = "oracle_run"
        sha256 = "a" * 64
        zip_path = str(run / "final_model.zip")

    def host(_ref):
        raise AssertionError("a checkpoint was loaded before the reveal mode was checked")

    with pytest.raises(RevealModeError, match="one_sided"):
        declare_engine([(Ref(), Ref())], host)


# ------------------------------------------------------------------ the information-lever spec

def _parse(argv):
    from main.train.parser import build_parser

    return build_parser().parse_args(argv)


def test_the_flag_takes_any_spec_the_observation_honours_and_stores_its_label():
    assert _parse([]).oracle_reveal is None, "default None: the _resolve line supplies 'off' (and a resume inherits)"
    for typed, want in (("off", "off"), ("species", "species"), ("full", "full"), ("species:unseen", "species"),
                        ("species:unseen,moves:both,item:both,ability:both,spread:both", "full"),
                        ("ki_moves", "moves:seen@0.95"), ("ki_items", "item:seen@0.95"),
                        ("ki_spreads", "spread:seen@0.95"), ("moves:seen@0.950", "moves:seen@0.95"),
                        ("F+M", "moves:seen"), ("D50", "moves:seen@0.5"), ("species,moves:unseen", "species:unseen,moves:unseen")):
        assert _parse(["--oracle-reveal", typed]).oracle_reveal == want, typed


@pytest.mark.parametrize("typed,named", [
    ("ki_wrongmoves", "placebo"), ("W90", "placebo"), ("MAP90", "map"), ("hp:seen", "hp:seen"),
    ("counters", "counters:seen"), ("oracle", "hp:seen"), ("O-M", "hp:seen"), ("moves:unseen", "species:unseen"),
    ("everything", "everything"),
])
def test_the_flag_refuses_every_lever_the_observation_cannot_honour_by_name(typed, named, capsys):
    with pytest.raises(SystemExit):
        _parse(["--oracle-reveal", typed])
    err = capsys.readouterr().err
    assert "--oracle-reveal" in err and named in err, err


def test_the_resume_gate_compares_meaning_not_text(version):
    full = dataclasses.replace(version, oracle_reveal="full")
    full.check_oracle_reveal("species:unseen,moves:both,item:both,ability:both,spread:both")   # the same reveal
    full.check_oracle_reveal("full")
    ki = dataclasses.replace(version, oracle_reveal="moves:seen@0.95")
    ki.check_oracle_reveal("ki_moves")
    for flip in ("off", "moves:seen@0.9", "moves:seen", "item:seen@0.95"):
        with pytest.raises(ModelVersionError, match="oracle_reveal mismatch"):
            ki.check_oracle_reveal(flip)
    with pytest.raises(ModelVersionError):
        version.check_oracle_reveal("ki_moves")


def test_a_recorded_spec_label_round_trips_and_its_record_names_the_levers(tmp_path, version):
    from agents.model import oracle_reveal as OR

    (tmp_path / "final_model.zip").write_bytes(b"")
    save_model_snapshot(str(tmp_path), dataclasses.replace(version, oracle_reveal="moves:seen@0.95"), git_hash="t")
    assert json.loads((tmp_path / "model_config.json").read_text())["oracle_reveal"] == "moves:seen@0.95"
    assert recorded_oracle_reveal(str(tmp_path / "final_model.zip")) == "moves:seen@0.95"
    with pytest.raises(OracleRevealRefused, match="moves:seen@0.95"):
        refuse_if_revealed(str(tmp_path / "final_model.zip"), tool="main.anchors")
    rec = OR.record("ki_moves")
    assert rec["label"] == rec["canonical"] == "moves:seen@0.95" and len(rec["hash"]) == 16
    assert rec["levers"] == [{"class": "moves", "part": "seen", "mass": 0.95, "source": "truth"}]
    assert OR.record("full")["label"] == "full" and OR.record("off")["levers"] == []
    assert "TRUE MOVES of the seen mons" in OR.describe("ki_moves") and "95.00%" in OR.describe("ki_moves")


def test_the_checkpoint_hparams_carry_the_label_and_the_reveal_record():
    """``metadata.json``'s per-checkpoint hparams (``run_io._model_hparams``) carry the extractor's LABEL and the spec's
    record block (label, canonical text, hash, levers)."""
    import types

    from main.train.run_io import _model_hparams

    def saveable(mode):
        return types.SimpleNamespace(
            gamma=1.0, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5, batch_size=64, n_steps=2048, grad_accum_steps=1,
            clip_range=lambda _: 0.2, clip_range_vf=None,
            policy=types.SimpleNamespace(optimizer=types.SimpleNamespace(param_groups=[{"weight_decay": 0.0}]),
                                         features_extractor=types.SimpleNamespace(oracle_reveal=mode)))

    h = _model_hparams(saveable("moves:seen@0.95"))
    assert h["oracle_reveal"] == "moves:seen@0.95"
    assert h["oracle_reveal_record"]["canonical"] == "moves:seen@0.95" and h["oracle_reveal_record"]["levers"]
    off = _model_hparams(saveable("off"))
    assert off["oracle_reveal"] == "off" and off["oracle_reveal_record"]["levers"] == []

