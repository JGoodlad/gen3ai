"""A launcher RESTART of a fresh `--arch production` run resolves back to the surface it saved.

The restart strips the FRESH-only `--arch` (`launcher.checkpoint.resume_child_args`) and trusts the
checkpoint's `model_config.json` to carry everything the umbrella wrote. Until config v125 it did not
carry `opp_intent_coef` — the dose that ENABLES the derived `opp_intent` toggle — so the restarted child
resolved it to 0.0, built no intent heads and died at `check_compatible`
(`[ModelVersion] FATAL: opp_intent mismatch: saved=True, current=False`; the F-LG-6 launcher run,
2026-09-30, `~/gen3ai_archive/cutover_prep/fresh2`). v125 records it. A pre-v125 checkpoint used to take
the dose from its run's `metadata.json:cli_args` as a MIGRATION (`config.inherit_derived_enable_coefs`);
since the X5 version break raised MIGRATION_FLOOR to 144 every such checkpoint is REFUSED before that
recovery could run (a launch exits FATAL_CONFIG with the pinned fix, a typed dose included).

(a) is the class guard: every key `apply_production_arch` can write onto the namespace is a
`ModelVersion` field. (b) is the end-to-end restart. Every test here fails on revert.
"""
from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from agents.model.model_version import ModelVersion
from main.train.parser import build_parser

_FRESH = ["--arch", "production", "--steps", "1000"]


def _umbrella_keys():
    """(registry name, the namespace dest `apply_production_arch` writes) for every surface row."""
    from agents.model.flag_registry import Tier, arch_surface_flags
    from main.train.arch_surface import load_production_config

    prod = load_production_config()
    return [(f.name, f.arg) for f in arch_surface_flags() if f.tier is Tier.CLI and f.name in prod]


# ── (a) the class guard ─────────────────────────────────────────────────────────────────────────
def test_every_key_the_arch_umbrella_writes_is_a_recorded_ModelVersion_field():
    """A value the umbrella writes that `model_config.json` does not record is LOST at the first
    launcher restart (which strips `--arch`) — the resume reads that key's OFF default."""
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    missing = [(name, dest) for name, dest in _umbrella_keys() if dest not in fields]
    assert not missing, f"--arch production writes these, model_config.json does not record them: {missing}"
    assert ("opp_intent", "opp_intent_coef") in _umbrella_keys()     # the derived row that motivated this


def test_the_field_is_recorded_by_every_launch_save_site():
    import main.train.lifecycle as lc
    import main.train.model_build as mb

    assert inspect.getsource(mb).count("opp_intent_coef=float(args.opp_intent_coef or 0.0)") == 2
    assert "opp_intent_coef=float(getattr(model, \"opp_intent_coef\", 0.0) or 0.0)" in inspect.getsource(lc)


# ── (b) end to end: fresh → save → the launcher's resume argv → resolve ─────────────────────────
def _saved_version(args) -> ModelVersion:
    """The ModelVersion a launch records, built the way `model_build` builds it (extractor + coef)."""
    from agents.model.extractor_arch import build_extractor_arch_kwargs
    from agents.model.features_extractor import NET_ARCH, Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    ek = build_extractor_arch_kwargs(args, base=Gen3ObservationEncoder(load_mappings()).get_features_extractor_kwargs())
    # (a `dict(...)` call, not a literal: `policy_activation_pin_test` counts the entry point's
    # policy_kwargs LITERALS, and this is a test double of the recorded fields, not a policy build)
    pk = dict(features_extractor_class=Gen3FeaturesExtractor, features_extractor_kwargs=ek, net_arch=NET_ARCH,
              critic=args.critic)
    # Every recorded training value `model_build` passes as `<name>=args.<name>` (the same-named dest);
    # the save-site test above pins that `opp_intent_coef` is among them at both sites.
    ctor = inspect.signature(ModelVersion.from_layout_and_policy_kwargs).parameters
    kw = {k: getattr(args, k) for k in ctor if k not in ("layout", "policy_kwargs")
          and getattr(args, k, None) is not None}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk, **kw)


def _run_dir(tmp_path, version, cli_args=None):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "checkpoint_1024_steps.zip"
    ckpt.write_bytes(b"")
    data = dataclasses.asdict(version) if isinstance(version, ModelVersion) else version
    (run / "model_config.json").write_text(json.dumps(data))
    if cli_args is not None:
        (run / "metadata.json").write_text(json.dumps({"cli_args": cli_args}))
    return str(run), str(ckpt)


def _resolve(argv):
    from main.train.config import resolve_config

    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


def test_a_launcher_restart_of_a_fresh_arch_production_run_resolves_back_to_its_surface(tmp_path):
    from agents.model.flag_registry import BY_NAME
    from main.launcher.checkpoint import resume_child_args

    fresh = _resolve(_FRESH)
    assert fresh.opp_intent_coef and fresh.opp_intent_coef > 0          # the umbrella turned it on
    run, ckpt = _run_dir(tmp_path, _saved_version(fresh))                # NO metadata.json: the field alone
    argv, stripped = resume_child_args(list(_FRESH), ckpt, run)
    assert stripped and "--arch" not in argv
    restarted = _resolve(argv)
    drift = {dest: (getattr(fresh, dest, None), getattr(restarted, dest, None)) for _n, dest in _umbrella_keys()
             if getattr(fresh, dest, None) != getattr(restarted, dest, None)}
    assert not drift, f"a launcher restart changed what --arch production set: {drift}"
    from agents.model.extractor_arch import _DERIVED
    assert _DERIVED["opp_intent"](restarted) is True
    assert BY_NAME["opp_intent"].source_arg == "opp_intent_coef"


# ── a pre-v125 checkpoint: below the X5 version break's floor, REFUSED ──────────────────────────
def _pre_v125(version: ModelVersion) -> dict:
    d = dataclasses.asdict(version)
    d.pop("opp_intent_coef")
    d["config_version"] = 124
    return d


def test_a_pre_v125_config_is_refused_at_the_floor():
    """The v125 branch (OFF ⇒ 0.0, ON ⇒ unrecorded) is unreachable since MIGRATION_FLOOR rose to 144."""
    from agents.model.model_version import ModelVersionError
    from agents.model.model_version.migrations import _migrate_config

    for on in (False, True):
        d = dict(_pre_v125(_saved_version(_resolve(_FRESH))), opp_intent=on)
        with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
            _migrate_config(d)


def _restart_argv(ckpt, run):
    """What the launcher hands a restarted child: the fresh argv minus `--arch`, plus --model/--run-dir."""
    from main.launcher.checkpoint import resume_child_args

    return resume_child_args(list(_FRESH), ckpt, run)[0]


@pytest.mark.parametrize("cli_args,typed", [
    ({"opp_intent_coef": 0.07}, []),                  # the dose the old migration recovered from cli_args
    (None, []),                                        # no recorded dose (the old UnrecordedEnableCoef case)
    (None, ["--opp-intent-coef", "0.05"]),             # a TYPED dose does not rescue a pre-break parent
])
def test_a_pre_v125_resume_is_REFUSED_with_the_pinned_fix(tmp_path, capsys, cli_args, typed):
    from agents.model.model_version.version_break import LAST_BLOB_COMMIT
    from main.exit_codes import TrainExitCode

    fresh = _resolve(_FRESH + ["--opp-intent-coef", "0.07"])
    run, ckpt = _run_dir(tmp_path, _pre_v125(_saved_version(fresh)), cli_args=cli_args)
    with pytest.raises(SystemExit) as e:
        _resolve(_restart_argv(ckpt, run) + typed)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    out = capsys.readouterr().out
    assert "PRE-GENERATION" in out and LAST_BLOB_COMMIT[:12] in out
    assert "MIGRATION" not in out                     # the cli_args recovery never ran
