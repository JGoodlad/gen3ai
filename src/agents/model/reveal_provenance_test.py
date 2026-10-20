"""THE REVEAL PROVENANCE CONTRACT (gen3_reveal_provenance_v1, GIGO FIX 2026-10-15).

The incident: every diagnostic knock-in fork (`rb_hib_*`, `--oracle-reveal moves:seen@0.95 --diagnostic-fork-consent`)
recorded `oracle_reveal: off` / `oracle_reveal_record: off` in `metadata.json` (and in every sidecar) beside a
`model_config.json` that said `moves:seen@0.95`. SB3 rebuilds the extractor from the PARENT zip's `policy_kwargs`;
`model_build` re-applied `belief_grad_mode` after the load but not `oracle_reveal`; `run_io._model_hparams` wrote the
record from the extractor attribute. The OBSERVATION was never wrong (the env core is declared from the resolved run
config), so it was provenance only -- but every reader of the per-checkpoint field got the parent's mode.

What a revert breaks (each test names its own):

* the post-load re-apply (`resume_reapply`, wired in `model_build`) -> the loaded extractor keeps the parent's `off`;
* `_model_hparams` reading the env core's declaration first -> a stale attribute is recorded;
* the save guards (`save_model_snapshot`, `record_checkpoint`, the startup `require_reveal_provenance`) -> a wrong
  record is written again, silently;
* the readers (`oracle_reveal.run_provenance`, `main.lineage`, `main.sidecar_audit`) preferring `model_config.json` ->
  the 11 existing forks read as `off`.

Run: python -m pytest src/agents/model/reveal_provenance_test.py -q
"""
from __future__ import annotations

import ast
import dataclasses
import json
import types

import pytest

from agents.model import oracle_reveal as OR
from agents.model.model_version import ModelVersion
from agents.model.resume_reapply import SETTERS, reapply_resume_immutable, resolved_run_level

SPEC = "moves:seen@0.95"


# ============================== the readers (synthetic run dirs: JSON only, no model) ==============================
def _run(tmp_path, name, *, config, meta, sidecars=None):
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "model_config.json").write_text(json.dumps({"oracle_reveal": config}))
    (d / "metadata.json").write_text(json.dumps(meta))
    for stem, entry in (sidecars or {}).items():
        (d / "checkpoints" / f"{stem}.zip").write_text("")
        (d / "checkpoints" / f"{stem}.json").write_text(json.dumps(entry))
    return d


def _stale_fork(tmp_path, name="rb_hib_moves_s3101"):
    """A run exactly as the 11 existing knock-in forks are on disk."""
    off = OR.record("off")
    return _run(tmp_path, name, config=SPEC,
                meta={"git_hash": "aaaa1111", "oracle_reveal": "off", "oracle_reveal_record": off},
                sidecars={"checkpoint_100_steps": {"git_hash": "aaaa1111", "oracle_reveal": "off",
                                                   "oracle_reveal_record": off}})


def test_the_reader_prefers_model_config_and_flags_the_existing_forks_as_known_bad(tmp_path):
    prov = OR.run_provenance(str(_stale_fork(tmp_path)))
    assert prov["label"] == SPEC and prov["source"] == "model_config.json"
    assert prov["known_bad"] is True
    assert prov["disagrees"] == ["metadata oracle_reveal", "metadata oracle_reveal_record.label"]
    assert OR.REVEAL_PROVENANCE_FIX in prov["note"] and SPEC in prov["note"] and "KNOWN-BAD" in prov["note"]


def test_a_correct_record_and_a_plain_run_are_not_flagged(tmp_path):
    rec = OR.record(SPEC)
    ok = _run(tmp_path, "rb_fixed_fork", config=SPEC,
              meta={"oracle_reveal": SPEC, "oracle_reveal_record": rec, "env_core": {"oracle_reveal": SPEC}})
    plain = _run(tmp_path, "rb_plain", config="off", meta={"oracle_reveal": "off"})
    legacy = _run(tmp_path, "rb_legacy", config="off", meta={"git_hash": "x"})   # a run that states no mode at all
    for d in (ok, plain, legacy):
        assert OR.run_provenance(str(d))["known_bad"] is False
    assert OR.run_provenance(str(ok))["label"] == SPEC and OR.run_provenance(str(legacy))["label"] == "off"


def test_a_spelling_of_the_same_spec_is_not_a_mismatch(tmp_path):
    """`ki_moves` and `moves:seen@0.95` mean the same reveal; `full` and its clause list do."""
    d = _run(tmp_path, "rb_spelling", config=SPEC, meta={"oracle_reveal": "ki_moves"})
    assert OR.run_provenance(str(d))["known_bad"] is False


def test_main_lineage_and_sidecar_audit_flag_the_known_bad_record(tmp_path, capsys):
    from main import lineage as L
    from main import sidecar_audit as SA

    d = _stale_fork(tmp_path)
    row = L.read_run(str(d))
    assert row["reveal"]["known_bad"] is True and row["reveal"]["label"] == SPEC
    text = L.render(row)
    assert f"reveal mode: {SPEC}" in text and "KNOWN-BAD reveal record" in text and OR.REVEAL_PROVENANCE_FIX in text

    rec = SA.audit_run(str(d))
    assert rec["reveal"]["known_bad"] is True and rec["n_reveal_mismatched"] == 1
    assert rec["sidecars"][0]["oracle_reveal"] == "off" and rec["sidecars"][0]["reveal_matches_config"] is False
    assert SA.main([str(d)]) == 0                      # a historical defect is reported, never an error exit
    out = capsys.readouterr().out
    assert "KNOWN-BAD reveal record" in out and "1 run(s) with a KNOWN-BAD reveal record" in out

    clean = _run(tmp_path, "rb_clean", config="off", meta={"oracle_reveal": "off"})
    assert "reveal mode" not in L.render(L.read_run(str(clean)))
    assert SA.audit_run(str(clean))["reveal"]["known_bad"] is False


# ================================== the guard (save refuses a wrong record, typed) =================================
def _version(reveal):
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return dataclasses.replace(ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}),
                               oracle_reveal=reveal)


def _hparams(reveal, env=None):
    return {"oracle_reveal": reveal, "oracle_reveal_record": OR.record(reveal),
            "env_core": {"env_core": "rust", **({"oracle_reveal": env} if env is not None else {})}}


def test_save_refuses_a_metadata_that_disagrees_with_the_config_and_writes_nothing(tmp_path):
    from agents.model.snapshot import save_model_snapshot

    with pytest.raises(OR.RevealProvenanceError, match="metadata oracle_reveal = 'off'"):
        save_model_snapshot(str(tmp_path / "run"), _version(SPEC), git_hash="t", hparams=_hparams("off"))
    assert not (tmp_path / "run" / "model_config.json").exists(), "a refused save must leave the run untouched"
    with pytest.raises(OR.RevealProvenanceError, match="env_core.oracle_reveal = 'off'"):
        save_model_snapshot(str(tmp_path / "run"), _version(SPEC), git_hash="t", hparams=_hparams(SPEC, env="off"))
    save_model_snapshot(str(tmp_path / "ok"), _version(SPEC), git_hash="t", hparams=_hparams(SPEC, env=SPEC))
    assert OR.run_provenance(str(tmp_path / "ok"))["known_bad"] is False
    # a save that states no mode (no hparams: the fresh-checkpoint tools) is not judged
    save_model_snapshot(str(tmp_path / "bare"), _version(SPEC), git_hash="t")


def test_the_sidecar_write_is_judged_against_the_runs_model_config(tmp_path):
    from agents.model.snapshot import record_checkpoint, save_model_snapshot

    run = tmp_path / "run"
    save_model_snapshot(str(run), _version(SPEC), git_hash="t", hparams=_hparams(SPEC, env=SPEC))
    ckpt = run / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.parent.mkdir()
    ckpt.write_text("")
    with pytest.raises(OR.RevealProvenanceError, match="checkpoint"):
        record_checkpoint(str(run), str(ckpt), 1e-4, 4, hparams=_hparams("off"), git_hash="t")
    assert not (run / "checkpoints" / "checkpoint_100_steps.json").exists()
    record_checkpoint(str(run), str(ckpt), 1e-4, 4, hparams=_hparams(SPEC, env=SPEC), git_hash="t")
    assert json.loads((run / "checkpoints" / "checkpoint_100_steps.json").read_text())["oracle_reveal"] == SPEC


def test_the_env_core_stamp_declares_the_reveal_it_was_built_at():
    from agents.training.rust_rollout.build import RustEnvDecl
    from main.train.rust_env_setup import env_core_stamp

    col = types.SimpleNamespace(core=types.SimpleNamespace(stamp="s"), cfg=types.SimpleNamespace(
        trigger=types.SimpleNamespace(target=0, describe=lambda: "t")))
    off = env_core_stamp(RustEnvDecl(n_envs=256), col)
    on = env_core_stamp(RustEnvDecl(n_envs=256, oracle_reveal="ki_moves"), col)
    assert off["oracle_reveal"] == "off" and on["oracle_reveal"] == SPEC           # the LABEL, whatever it was typed as
    assert "oracle_reveal" not in off["summary"] and f"oracle_reveal {SPEC}" in on["summary"]


def _saveable(*, extractor, stamp=None):
    class _Opt:
        param_groups = [{"lr": 1e-4, "weight_decay": 0.0}]

    m = types.SimpleNamespace(
        policy=types.SimpleNamespace(optimizer=_Opt(), features_extractor=types.SimpleNamespace(oracle_reveal=extractor)),
        batch_size=64, n_steps=8, clip_range=lambda _p: 0.15, clip_range_vf=None, n_epochs=5, ent_coef=0.05,
        vf_coef=0.5, gamma=1.0, gae_lambda=0.8, max_grad_norm=0.5, learning_rate=1e-4)
    if stamp is not None:
        m._env_core_stamp = stamp
    return m


def test_the_record_comes_from_the_env_declaration_not_the_extractor_attribute():
    from main.train.run_io import _model_hparams, run_reveal

    stale = _saveable(extractor="off", stamp={"env_core": "rust", "oracle_reveal": SPEC})
    assert run_reveal(stale) == SPEC
    h = _model_hparams(stale)
    assert h["oracle_reveal"] == SPEC and h["oracle_reveal_record"]["canonical"] == SPEC
    assert h["env_core"]["oracle_reveal"] == SPEC
    # no collector (a fresh-checkpoint tool): the extractor, built from the same config, is the source
    assert run_reveal(_saveable(extractor=SPEC)) == SPEC and run_reveal(_saveable(extractor="off")) == "off"


def test_the_startup_guard_exits_fatal_config_on_a_stale_extractor_and_passes_when_they_agree(monkeypatch):
    from main.exit_codes import TrainExitCode
    from main.train.run_io import require_reveal_provenance

    codes = []

    def _exit(code):
        codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr("os._exit", _exit)
    stamp = {"env_core": "rust", "oracle_reveal": SPEC}
    with pytest.raises(SystemExit):                       # the extractor still carries the PARENT's `off`
        require_reveal_provenance(_saveable(extractor="off", stamp=stamp), _version(SPEC))
    assert codes == [int(TrainExitCode.FATAL_CONFIG)]
    with pytest.raises(SystemExit):                       # the env was declared at another mode than the config
        require_reveal_provenance(_saveable(extractor=SPEC, stamp={"env_core": "rust", "oracle_reveal": "off"}),
                                  _version(SPEC))
    require_reveal_provenance(_saveable(extractor=SPEC, stamp=stamp), _version(SPEC))          # agree: silent
    require_reveal_provenance(_saveable(extractor="off", stamp={"env_core": "rust", "oracle_reveal": "off"}),
                              _version("off"))


# ================================= the class: every resume-immutable flag is re-applied ============================
def test_every_resume_immutable_flag_has_a_post_load_setter():
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.flag_registry import REGISTRY, Klass

    from agents.model.resume_reapply import REAPPLIED_NUMERICS

    immutable = {f.name for f in REGISTRY if f.klass is Klass.RESUME_IMMUTABLE}
    assert not (REAPPLIED_NUMERICS & immutable) and REAPPLIED_NUMERICS <= {f.name for f in REGISTRY}
    assert set(SETTERS) == immutable | REAPPLIED_NUMERICS, (
        "a resume_immutable flag the loaded extractor carries as an attribute is rebuilt from the PARENT's zip on a "
        "fork: add its setter to agents.model.resume_reapply.SETTERS (and extractor_api) in the same pass")
    for name, setter in SETTERS.items():
        assert callable(getattr(Gen3FeaturesExtractor, setter)), (name, setter)


def test_the_trainer_applies_the_table_after_every_model_load_and_guards_both_first_saves():
    """FAILS ON REVERT of the wiring: `build_and_train` must re-apply through the table (not a lone
    `set_belief_grad_mode`) and run the startup guard before the first snapshot of BOTH the resume and the fresh path."""
    from utils.paths import src_path

    tree = ast.parse(src_path("main", "train", "model_build.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "build_and_train")
    calls = [n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))]
    assert calls.count("reapply_resume_immutable") == 1 and calls.count("resolved_run_level") == 1, calls
    assert "set_belief_grad_mode" not in calls, "the lone setter is the incident: go through the table"
    assert calls.count("require_reveal_provenance") == 2, calls


# ====================== the real fork: a plain parent, a consented flip, every record agrees =======================
@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    from main.fresh_checkpoint import save_fresh_checkpoint

    run = tmp_path_factory.mktemp("reveal_parent") / "rb_parent"
    return run, save_fresh_checkpoint(run, 11)


def _fork_load(parent, run_archive):
    """What `model_build` does for `--model <parent> --oracle-reveal SPEC --diagnostic-fork-consent ...`: the resolved
    namespace, the version built from ITS policy kwargs, the consent stamp, and the learner's loader."""
    from agents.model.snapshot import load_model_snapshot
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train import diagnostic_fork as DF
    from main.train.production_args import production_args

    run, zip_path = parent
    args = production_args()
    args.oracle_reveal = OR.label(SPEC)
    _, layout, pk = _production_policy_kwargs(args)
    version = ModelVersion.from_layout_and_policy_kwargs(layout, pk, vf_coef=float(args.vf_coef))
    child = run_archive / "rb_child"
    stamp = DF.decide(consent="reveal-provenance test", model_path=str(zip_path), model_dir=str(child),
                      requested=SPEC, now="2026-10-15T00:00:00+00:00")
    model = load_model_snapshot(str(zip_path), env=None, current_version=version, device="cpu",
                                enforce_oracle_reveal=args.oracle_reveal, oracle_reveal_fork=stamp)
    return args, version, model, child


@pytest.mark.slow  # ship gate (2026-10-19): 20 s beside a live run
def test_a_fork_of_a_plain_parent_with_a_reveal_records_it_everywhere(parent, run_archive):
    from agents.model.snapshot import record_checkpoint, save_model_snapshot
    from main import sidecar_audit as SA
    from main.train.run_io import _model_hparams, require_reveal_provenance

    args, version, model, child = _fork_load(parent, run_archive)
    ext = model.policy.features_extractor
    # THE MECHANISM, asserted rather than assumed: SB3 rebuilt the extractor from the PARENT zip's kwargs
    assert ext.oracle_reveal == "off" and model.policy_kwargs["features_extractor_kwargs"]["oracle_reveal"] == "off"
    assert version.oracle_reveal == SPEC

    # the trainer's post-load call (the SAME one that applies belief_grad_mode): the attribute follows the run config
    applied = reapply_resume_immutable(ext, resolved_run_level(args))
    assert applied["oracle_reveal"] == ("off", SPEC) and ext.oracle_reveal == SPEC
    # ...and ONLY the attribute: the policy's constructor kwargs (the forward identity `forward_fingerprint` and
    # main.h2h's EngineArch read) stay the parent's, so a fork shares a slot group with its plain control
    assert model.policy_kwargs["features_extractor_kwargs"]["oracle_reveal"] == "off"

    # with the env core's declaration on the model (as `_start_rust_env` leaves it) every witness agrees
    model._env_core_stamp = {"env_core": "rust", "oracle_reveal": args.oracle_reveal}
    require_reveal_provenance(model, version)
    hp = _model_hparams(model)
    save_model_snapshot(str(child), version, hparams=hp, git_hash="t")
    ckpt = child / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.parent.mkdir(exist_ok=True)
    ckpt.write_text("")
    record_checkpoint(str(child), str(ckpt), 1e-4, 4, hparams=hp, git_hash="t")

    meta = json.loads((child / "metadata.json").read_text())
    cfg = json.loads((child / "model_config.json").read_text())
    side = json.loads((child / "checkpoints" / "checkpoint_100_steps.json").read_text())
    assert cfg["oracle_reveal"] == meta["oracle_reveal"] == side["oracle_reveal"] == SPEC
    assert meta["oracle_reveal_record"]["label"] == side["oracle_reveal_record"]["label"] == SPEC
    assert meta["oracle_reveal_record"]["levers"] and meta["env_core"]["oracle_reveal"] == SPEC
    assert meta["snapshot_history"]["checkpoint_100_steps.zip"]["oracle_reveal"] == SPEC
    assert OR.run_provenance(str(child))["known_bad"] is False
    rec = SA.audit_run(str(child))
    assert rec["reveal"]["known_bad"] is False and rec["n_reveal_mismatched"] == 0


def test_without_the_reapply_the_stale_attribute_is_refused_not_recorded(parent, run_archive):
    """FAILS ON REVERT of the re-apply: a save with no env stamp reads the attribute, which is the parent's `off`."""
    from agents.model.snapshot import save_model_snapshot
    from main.train.run_io import _model_hparams

    _, version, model, child = _fork_load(parent, run_archive)
    assert model.policy.features_extractor.oracle_reveal == "off"
    with pytest.raises(OR.RevealProvenanceError, match="metadata oracle_reveal = 'off'"):
        save_model_snapshot(str(child), version, hparams=_model_hparams(model), git_hash="t")


def test_the_attribute_is_provenance_only_the_forward_does_not_read_it(parent, run_archive):
    """THE IMPACT VERDICT, executed: re-applying the reveal changes no weight and no output bit."""
    import torch

    from agents.model.extra_obs_keys import synthetic_obs

    args, _, model, _ = _fork_load(parent, run_archive)
    ext = model.policy.features_extractor.eval()
    obs = synthetic_obs(ext, ext.layout["total_dim"], device="cpu", action_mask=True)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    with torch.no_grad():
        pi0, vf0 = ext(obs)
    assert torch.isfinite(pi0).all() and torch.isfinite(vf0).all()     # so `equal` below is not a NaN != NaN
    reapply_resume_immutable(ext, resolved_run_level(args))
    with torch.no_grad():
        pi1, vf1 = ext(obs)
    assert torch.equal(pi0, pi1) and torch.equal(vf0, vf1)
    assert all(torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
