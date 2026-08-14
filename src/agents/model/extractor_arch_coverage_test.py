"""Every version-checked extractor toggle must actually REACH the extractor.

The bug this exists to prevent, caught in the v72 smoke: `--t0-species-prior` was added to argparse,
to `ModelVersion`, to the migration, to the resume check, and to `current_model_version` — but NOT
to `extractor_arch.ARCH_ARG_KEYS`, the mapping that builds the real
`policy_kwargs["features_extractor_kwargs"]`. So the flag parsed, was recorded, was version-checked,
and the module was never built. Training ran happily with the feature silently off.

Nothing else can catch this shape of mistake:

* no shape check fires — the state_dict is identical whether the module exists or not;
* the unit suite passes — every test that exercises the feature constructs the extractor DIRECTLY
  with the kwarg, bypassing the mapping entirely;
* the version check passes — it compares the recorded value against itself.

Only an end-to-end run reveals it, and only if someone thinks to inspect the saved config. So this
test asserts the connection structurally instead: a toggle that `ModelVersion` version-checks and
`Gen3FeaturesExtractor.__init__` accepts MUST be reachable from `args`.
"""
import dataclasses
import inspect

import pytest

from agents.model.extractor_arch import ARCH_ARG_KEYS, _DERIVED
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import ModelVersion


def _extractor_params() -> set:
    return set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters) - {"self"}


def _reachable_from_args() -> set:
    """Kwargs `build_extractor_arch_kwargs` can produce — plain reads plus derived callables."""
    return set(ARCH_ARG_KEYS.values()) | set(_DERIVED)


def test_every_arch_mapping_target_is_a_real_extractor_kwarg():
    """A mapping entry naming a kwarg the extractor does not take would raise at build time."""
    unknown = _reachable_from_args() - _extractor_params()
    assert not unknown, (
        f"{sorted(unknown)} are mapped into features_extractor_kwargs but "
        f"Gen3FeaturesExtractor.__init__ does not accept them"
    )


def test_every_version_checked_toggle_reaches_the_extractor():
    """The load-bearing direction: a recorded arch field the extractor takes must be plumbed.

    Skips fields the extractor does not take as a constructor argument (pure training hparams like
    `vf_coef`, and the reward-config block) — those are recorded for the resume checks and are not
    supposed to reach the extractor at all.
    """
    recorded = {f.name for f in dataclasses.fields(ModelVersion)}
    should_be_plumbed = recorded & _extractor_params()
    missing = should_be_plumbed - _reachable_from_args()
    assert not missing, (
        f"{sorted(missing)} are recorded in model_config.json AND accepted by "
        f"Gen3FeaturesExtractor.__init__, but no path from `args` produces them — so the flag would "
        f"be silently ignored while still being version-checked. Add them to "
        f"extractor_arch.ARCH_ARG_KEYS (or _DERIVED)."
    )


@pytest.mark.parametrize("flag", ["t0_species_prior", "species_prior_fusion", "value_threat_inject"])
def test_named_structural_flags_are_plumbed(flag):
    """Explicit regression pins for the three no-state_dict-delta toggles.

    These are the highest-risk ones: adding or removing the module changes no tensor shape, so a
    broken plumb is invisible everywhere except a saved config nobody reads.
    """
    assert flag in _reachable_from_args(), f"--{flag.replace('_', '-')} never reaches the extractor"


def test_the_mapping_actually_produces_the_kwarg_from_args():
    """End-to-end through the REAL builder, not just the table — a namespace in, a kwarg out.

    The table tests above prove the entry exists; this proves the builder honours it. Both matter:
    the v72 bug was an absent entry, but a builder that skipped an entry would look identical from
    the saved config's point of view.
    """
    from argparse import Namespace

    from agents.model.extractor_arch import build_extractor_arch_kwargs
    # `build_extractor_arch_kwargs` reads EVERY mapped attr off args, so the namespace must be
    # complete — populate from the table itself rather than hand-listing (which would rot).
    ns = Namespace(**{attr: False for attr in ARCH_ARG_KEYS.values()})
    for attr in _DERIVED:
        setattr(ns, attr, False)
    for extra in ("opp_belief_aux_coef", "move_belief_coef",
                  "opp_intent_coef", "seed_quantile_coef", "damage_refine_rounds"):
        setattr(ns, extra, 0.0)
    ns.t0_species_prior = True
    out = build_extractor_arch_kwargs(ns, base={})
    assert out.get("t0_species_prior") is True

    ns.t0_species_prior = False
    assert build_extractor_arch_kwargs(ns, base={}).get("t0_species_prior") is False
