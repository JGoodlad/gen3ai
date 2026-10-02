"""A launcher RESTART of a fresh `--arch production` run resolves back to the surface it saved.

The restart strips the FRESH-only `--arch` (`launcher.checkpoint.resume_child_args`) and trusts the
checkpoint's `model_config.json` to carry everything the umbrella wrote. Until config v125 it did not
carry `opp_intent_coef` — the dose that ENABLES the derived `opp_intent` toggle — so the restarted child
resolved it to 0.0, built no intent heads and died at `check_compatible`
(`[ModelVersion] FATAL: opp_intent mismatch: saved=True, current=False`; the F-LG-6 launcher run,
2026-09-30, `~/gen3ai_archive/cutover_prep/fresh2`). v125 records it; a pre-v125 checkpoint takes the
dose from its run's `metadata.json:cli_args` as a MIGRATION, or is REFUSED — never guessed.

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

_FRESH = ["--arch", "production", "--critic", "winprob", "--terminal-indicator", "--victory-value", "1.0",
          "--draw-penalty", "0", "--steps", "1000"]


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


# ── the pre-v125 migration: cli_args, else REFUSE ───────────────────────────────────────────────
def _pre_v125(version: ModelVersion) -> dict:
    d = dataclasses.asdict(version)
    d.pop("opp_intent_coef")
    d["config_version"] = 124
    return d


def test_a_pre_v125_config_migrates_OFF_to_0_and_leaves_ON_unrecorded():
    from agents.model.model_version.migrations import _migrate_config

    base = dataclasses.asdict(_saved_version(_resolve(_FRESH)))
    for on, want in ((False, 0.0), (True, None)):
        d = dict(base, opp_intent=on, config_version=124)
        d.pop("opp_intent_coef")
        out = _migrate_config(d)
        assert out["config_version"] >= 125 and "opp_intent_coef" in out and out["opp_intent_coef"] == want


def _restart_argv(ckpt, run):
    """What the launcher hands a restarted child: the fresh argv minus `--arch`, plus --model/--run-dir."""
    from main.launcher.checkpoint import resume_child_args

    return resume_child_args(list(_FRESH), ckpt, run)[0]


def test_a_pre_v125_resume_takes_the_dose_from_cli_args(tmp_path, capsys):
    fresh = _resolve(_FRESH + ["--opp-intent-coef", "0.07"])
    run, ckpt = _run_dir(tmp_path, _pre_v125(_saved_version(fresh)), cli_args={"opp_intent_coef": 0.07})
    restarted = _resolve(_restart_argv(ckpt, run))
    assert restarted.opp_intent_coef == 0.07
    assert "MIGRATION" in capsys.readouterr().out


def test_a_pre_v125_resume_with_no_recorded_dose_is_REFUSED_and_a_typed_one_wins(tmp_path):
    from main.exit_codes import TrainExitCode

    fresh = _resolve(_FRESH)
    run, ckpt = _run_dir(tmp_path, _pre_v125(_saved_version(fresh)))          # no metadata.json at all
    with pytest.raises(SystemExit) as e:
        _resolve(_restart_argv(ckpt, run))
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    typed = _resolve(_restart_argv(ckpt, run) + ["--opp-intent-coef", "0.05"])
    assert typed.opp_intent_coef == 0.05
