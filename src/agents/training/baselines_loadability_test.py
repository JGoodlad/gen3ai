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
    """A synthetic entry — the registry's own shape, so the era logic is testable with no archive."""
    raw = {"kind": "checkpoint", "run": "ai_vX_fake", "checkpoint": "final_model.zip",
           "commit": "deadbeefcafe1234", "config_version": 999,
           "arch_signature": "gen3_critic_route_wave_v1", "purpose": "p", "set_on": "2026-01-01",
           "set_by": "s", "sha256": "0" * 64}
    raw.update(over)
    return baselines._build("fake", raw)


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
    monkeypatch.setattr(baselines, "get", lambda name, path=None: _entry(config_version=45,
                                                                        era_checkout_only=True))
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
def test_the_bare_load_that_started_this_STILL_fails_where_baselines_load_succeeds():
    """Why `baselines.load` exists, pinned against the real bytes.

    `production` is the entry the backlog row named; `v9_long_baseline` is there to record the
    measurement that killed the re-pointing hypothesis — the bare path fails identically on a
    DIFFERENT entry, so the trap is the loader, not the name.
    """
    if main_models_dir() is None:
        pytest.skip(models_skip_reason())
    from sb3_contrib import MaskablePPO

    for name in ("production", "v9_long_baseline"):
        r = baselines.resolve(name)
        with pytest.raises(TypeError) as exc:
            MaskablePPO.load(r.zip_path, env=None, device="cpu")
        assert "unexpected keyword argument" in str(exc.value)
        assert baselines.load(name) is not None, f"{name} must load through the sanitizing loader"
