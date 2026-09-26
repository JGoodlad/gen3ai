"""EVERY NAMED BASELINE LOADS, OR FAILS LOUD — the contract `BaselineLoadError` carries.

The defect this file pins (tech-debt P1, 2026-09-14). An external-anchor de-risk asked the
registry for the name `production`, loaded it with a bare `MaskablePPO.load`, got

    TypeError: ExtractorBuild.__init__() got an unexpected keyword argument 'threat_prob_outspeed'

read it as "ordinary arch drift for that one entry", substituted a different checkpoint, and
published the numbers. MEASURED 2026-09-22, both halves of that reading were wrong:

* `production` LOADS at HEAD through this project's own loader
  (`agents.model.snapshot.load_foreign_opponent`, which runs the deleted-kwarg sanitizer), and
* the identical `TypeError` comes out of a bare load for **all five** current-generation entries
  (`production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`,
  `untaught_meter_opponent`) — so re-pointing the name would have moved the failure, not fixed it.

So the fix was never a re-point. It is that a by-name load goes through ONE function which either
returns a model or raises a typed `BaselineLoadError` whose message names the fix — and that the
registry's `era_checkout_only` flag is VALIDATED against the generation each entry records, so an
unmarked pre-generation node cannot sit there looking loadable.

Two tiers. The structural half needs neither torch nor `models/` and runs everywhere; the
`integration` half actually loads every checkpoint in the archive (~2.5 s each).
"""
from __future__ import annotations

import pytest

from agents.training import baselines
from utils.paths import main_models_dir, models_skip_reason


# ---------------------------------------------------------------- structural (no archive, no torch)

def _entry(**over):
    """A synthetic entry — the registry's own shape, so the era logic is testable with no archive.

    It records THIS tree's generation by default (the live ARCH_SIGNATURE at a version above the
    floor), so every test that does not name an era is about a CURRENT-generation entry — which is
    what they were written about. gen3_event_record_v2 moved the signature; a hard-coded one here
    silently turned every such test into a pre-generation one."""
    from agents.model.model_version import ARCH_SIGNATURE
    raw = {"kind": "checkpoint", "run": "ai_vX_fake", "checkpoint": "final_model.zip",
           "commit": "deadbeefcafe1234", "config_version": 999,
           "arch_signature": ARCH_SIGNATURE, "purpose": "p", "set_on": "2026-01-01",
           "set_by": "s", "sha256": "0" * 64}
    raw.update(over)
    return baselines._build("fake", raw)


def _save_current_generation_checkpoint(run_dir, *, pickled_extra_fek=None) -> str:
    """A LOADABLE current-generation checkpoint, built fresh and saved through the project's own
    path (a real MaskablePPO `.save` + `save_model_snapshot`'s model_config.json/metadata.json).

    `pickled_extra_fek` is written into the zip's PICKLED `features_extractor_kwargs` only — the
    exact shape of the 2026-09-14 incident's bytes (a constructor flag deleted since the save),
    which the recorded model_config.json does not carry."""
    import gymnasium as gym
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.model_version import ModelVersion
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.model.snapshot import save_model_snapshot
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (total_dim,), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = obs_space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kwargs):
            return {"observation": np.zeros(total_dim, np.float32),
                    "action_mask": np.ones(11, np.int8)}, {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    pk = {"features_extractor_class": Gen3FeaturesExtractor,
          "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
          "net_arch": [512, 512]}
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E]), policy_kwargs=pk,
                        verbose=0, device="cpu")
    if pickled_extra_fek:
        model.policy_kwargs["features_extractor_kwargs"].update(pickled_extra_fek)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "final_model"))
    save_model_snapshot(str(run_dir),
                        ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}),
                        git_hash="test")
    return str(run_dir / "final_model.zip")


def test_the_generation_verdict_is_registry_data_only():
    """`is_pre_generation` reads the ENTRY, never the archive — same answer in a fresh clone."""
    from agents.model.model_version import ARCH_SIGNATURE, MIGRATION_FLOOR

    assert not baselines.is_pre_generation(_entry(config_version=MIGRATION_FLOOR,
                                                  arch_signature=ARCH_SIGNATURE))
    assert baselines.is_pre_generation(_entry(config_version=MIGRATION_FLOOR - 1))
    # A signature bump with no floor move must still read as pre-generation.
    assert baselines.is_pre_generation(_entry(arch_signature="gen3_some_other_family_v1"))


def test_every_registry_entry_declares_its_era_honestly():
    """`era_checkout_only` is the registry's public claim that a by-name load works here.

    An unmarked pre-generation entry is what sends a reader hunting for a stand-in checkpoint; a
    stale mark on a loadable one sends them to a pinned checkout they do not need. Both are
    errors, and `validate()` says so with no archive at all.
    """
    for name in baselines.names():
        b = baselines.get(name)
        assert baselines.is_pre_generation(b) == b.era_checkout_only, (
            f"{name}: records {baselines.era_of(b)} but era_checkout_only={b.era_checkout_only}")
    errors = [f for f in baselines.validate(verify_sha=False) if f.level == "error"]
    assert not errors, "\n".join(f.line() for f in errors)


def test_a_pre_generation_name_raises_the_typed_error_naming_the_era_and_the_fix(monkeypatch):
    monkeypatch.setattr(baselines, "get", lambda name, path=None: _entry(
        config_version=45, arch_signature="gen3_critic_route_wave_v1", era_checkout_only=True))
    monkeypatch.setattr(baselines, "names", lambda path=None: [])
    with pytest.raises(baselines.BaselineLoadError) as exc:
        baselines.check_era("v8_line")
    err = exc.value
    assert err.reason == "pre_generation"
    assert err.era == "v45/gen3_critic_route_wave_v1"
    assert "deadbeefcafe1234" in str(err), "the message must name the commit it IS readable from"
    assert "FIX:" in str(err)
    # It is still a BaselineError, so every existing `except BaselineError` keeps working.
    assert isinstance(err, baselines.BaselineError)


def test_an_UNMARKED_pre_generation_entry_is_reported_as_an_error():
    findings = baselines._validate_era(_entry(config_version=45, era_checkout_only=False))
    assert [f.level for f in findings] == ["error"]
    assert "era_checkout_only" in findings[0].message
    assert "main.baselines set" in findings[0].message


def test_a_STALE_era_mark_on_a_loadable_entry_is_reported():
    findings = baselines._validate_era(_entry(era_checkout_only=True))
    assert [f.level for f in findings] == ["error"]
    assert "stale era mark" in findings[0].message.lower() or "drop the flag" in findings[0].message


def test_a_config_kind_baseline_is_refused_by_load_with_the_call_that_works(monkeypatch):
    monkeypatch.setattr(baselines, "get",
                        lambda name, path=None: _entry(kind="config",
                                                       checkpoint="model_config.json"))
    with pytest.raises(baselines.BaselineLoadError) as exc:
        baselines.load("untaught_meter_config")
    assert exc.value.reason == "not_a_model"
    assert "config_path(" in str(exc.value)


def test_load_goes_through_the_SANITIZING_loader_never_a_bare_MaskablePPO_load(monkeypatch):
    """THE regression. `load_foreign_opponent` is the loader that runs the deleted-kwarg
    sanitizer (`_patch_historical_floor`); a bare `MaskablePPO.load` rebuilds the extractor from
    the zip's own pickled kwargs and dies on any constructor flag deleted since. Swapping this
    call for a bare load is the mistake the whole file exists to stop, so it is pinned here.
    """
    import agents.model.snapshot as snap

    seen = {}

    def _fake(zip_path, current_version, device="cpu", config_path=None):
        seen["zip"] = zip_path
        seen["config"] = config_path
        return ("MODEL", "VERSION")

    monkeypatch.setattr(snap, "load_foreign_opponent", _fake)
    monkeypatch.setattr(baselines, "get", lambda name, path=None: _entry())
    monkeypatch.setattr(
        baselines, "resolve",
        lambda name, path=None: baselines.ResolvedBaseline(
            baseline=_entry(), zip_path="/tmp/x.zip", config_path="/tmp/model_config.json",
            run_dir="/tmp", rung="explicit_zip", rule="r", num_timesteps=1))
    assert baselines.load("production") == "MODEL"
    assert seen["zip"] == "/tmp/x.zip" and seen["config"] == "/tmp/model_config.json"


def test_a_loader_refusal_on_a_CURRENT_generation_entry_is_arch_drift_not_a_licence(monkeypatch):
    """A current-generation entry the loader still refuses is a DEFECT, and the message says so —
    it must never read as permission to swap in a stand-in checkpoint."""
    import agents.model.snapshot as snap

    def _boom(*a, **k):
        raise TypeError("unexpected keyword argument 'threat_prob_outspeed'")

    monkeypatch.setattr(snap, "load_foreign_opponent", _boom)
    monkeypatch.setattr(baselines, "get", lambda name, path=None: _entry())
    monkeypatch.setattr(
        baselines, "resolve",
        lambda name, path=None: baselines.ResolvedBaseline(
            baseline=_entry(), zip_path="/tmp/x.zip", config_path="/tmp/model_config.json",
            run_dir="/tmp", rung="explicit_zip", rule="r", num_timesteps=1))
    with pytest.raises(baselines.BaselineLoadError) as exc:
        baselines.load("production")
    assert exc.value.reason == "arch_drift"
    assert "not something to route around" in str(exc.value)
    assert "FIX:" in str(exc.value)


# ---------------------------------------------------------------- archive-backed (torch + models/)


@pytest.mark.integration
def test_EVERY_named_baseline_either_loads_or_raises_the_typed_error_naming_the_fix():
    """The row's acceptance test: iterate the whole registry, load every checkpoint entry."""
    if main_models_dir() is None:
        pytest.skip(models_skip_reason())
    checked = 0
    for name in baselines.names():
        b = baselines.get(name)
        if b.kind != "checkpoint":
            continue
        checked += 1
        try:
            model = baselines.load(name)
        except baselines.BaselineLoadError as exc:
            assert exc.reason in baselines.LOAD_FAILURE_REASONS, f"{name}: {exc.reason}"
            assert "FIX:" in str(exc), f"{name}: a refusal must name the fix\n{exc}"
            assert b.commit[:8] in str(exc), f"{name}: the fix must name the entry's commit"
            # Only a pre-generation node may refuse, and only when the registry SAYS so.
            assert exc.reason == "pre_generation" and b.era_checkout_only, (
                f"{name}: refused with reason={exc.reason!r} — a current-generation baseline that "
                f"does not load is a defect, not a fact about the entry.\n{exc}")
            continue
        assert model is not None
        assert not b.era_checkout_only, f"{name} loads here but is marked era_checkout_only"
    assert checked >= 5, "the registry should hold at least the five current-generation entries"


@pytest.mark.integration
def test_the_bare_load_that_started_this_STILL_fails_where_baselines_load_succeeds(
        tmp_path, monkeypatch):
    """Why `baselines.load` exists, pinned against real bytes.

    The original form loaded `production` and `v9_long_baseline` from the archive. Since
    gen3_event_record_v2 (the observation-architecture batch, v121) both are PRE-GENERATION and
    marked `era_checkout_only`, so on this tree they refuse by name at the era wall BEFORE any
    loader runs — pinned first, because a by-name load that got past the wall would be the bug.
    The loader contrast is then re-measured on a CURRENT-generation checkpoint built fresh and
    saved through the project's own path, whose pickled extractor kwargs carry the incident's
    exact deleted flag (`threat_prob_outspeed`): the bare `MaskablePPO.load` must still die on it,
    and `baselines.load` — resolving a synthetic registry entry through the real `resolve` choke
    point and the real sanitizing loader — must still return a model. No stub stands in for either
    loader.
    """
    for name in ("production", "v9_long_baseline"):
        with pytest.raises(baselines.BaselineLoadError) as exc:
            baselines.load(name)
        assert exc.value.reason == "pre_generation", (name, exc.value.reason)
        assert baselines.get(name).era_checkout_only, name

    from sb3_contrib import MaskablePPO

    zip_path = _save_current_generation_checkpoint(
        tmp_path / "ai_vX_current_gen", pickled_extra_fek={"threat_prob_outspeed": False})
    run_dir = str(tmp_path / "ai_vX_current_gen")
    entry = _entry(run=run_dir, checkpoint="final_model.zip")
    assert not baselines.is_pre_generation(entry)
    monkeypatch.setattr(baselines, "get", lambda name, path=None: entry)

    with pytest.raises(TypeError) as bare:
        MaskablePPO.load(zip_path, env=None, device="cpu")
    assert "unexpected keyword argument" in str(bare.value)
    assert "threat_prob_outspeed" in str(bare.value)
    assert baselines.resolve("current_gen").zip_path == zip_path
    assert baselines.load("current_gen") is not None, "must load through the sanitizing loader"
