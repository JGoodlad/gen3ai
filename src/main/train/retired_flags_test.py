"""The Python env core's flags (deletion pass U3, manifest R1 / R6): deleted, or reduced to ONE legal value.
(P11b then deleted `--env-core`, `--use-bridge` and `--critic` outright: they are in `census_deleted_flags_test`,
refused WITH their reason, and `critic` is a constant of the namespace, `parser/objective.py`.)

A silent no-op is the failure this file exists to rule out. Each case below fails on a revert of the piece it
names.
"""
from __future__ import annotations

import inspect

import pytest

from main.train.parser import build_parser

#: Spelled by concatenation: `claude_md_freshness_gate_test` treats a bare `"--flag"` constant in a
#: non-test module as live CLI surface; these are deleted flags, kept here only to prove they stay deleted.
_DELETED = ["--" + n for n in ("async-rollout", "obs-source", "compile-opponents",
                               "compile-opponents-preload", "compile-opponents-strict")]


@pytest.mark.parametrize("flag", _DELETED + ["--no-" + _DELETED[0][2:], "--no-" + _DELETED[2][2:]])
def test_a_deleted_flag_is_unrecognised(flag, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "1", flag])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_the_one_core_and_transport_are_no_namespace_attribute_and_the_critic_is_a_constant():
    """The one env core and transport are not namespace attributes at all (nothing can type them, nothing reads
    them); the one critic is a CONSTANT of every namespace (`parser/objective.py`), not an option."""
    ns = build_parser().parse_args(["--steps", "1"])
    assert not hasattr(ns, "env_core") and not hasattr(ns, "use_bridge")      # deleted outright (P11b)
    assert ns.critic == "winprob"
    assert "--critic" not in {o for a in build_parser()._actions for o in a.option_strings}


def test_shaped_stays_LOADABLE_but_the_trainer_trains_only_the_default():
    """`shaped` stays LOADABLE (every pre-v109 checkpoint, every shaped-era opponent) but the one critic the
    trainer trains is `CRITIC_DEFAULT` (winprob) — there is no trainable-set constant any more (P11b)."""
    from agents.model import critic_mode
    from agents.model.critic_mode import CRITIC_DEFAULT, CRITIC_MODES, CRITIC_SHAPED, CRITIC_UNRECORDED

    assert CRITIC_DEFAULT in CRITIC_MODES and CRITIC_SHAPED in CRITIC_MODES and CRITIC_DEFAULT != CRITIC_SHAPED
    assert not hasattr(critic_mode, "CRITIC_TRAINABLE_MODES")
    assert CRITIC_UNRECORDED == CRITIC_SHAPED, "an ABSENT record still means shaped: old checkpoints must load as such"


def test_the_policy_still_loads_a_shaped_critic_checkpoint():
    from agents.model.policy import Gen3DualHeadMaskablePolicy

    assert "critic" in inspect.signature(Gen3DualHeadMaskablePolicy.__init__).parameters
    # the policy's own validation takes the LOADABLE set, never the trainable one
    src = inspect.getsource(Gen3DualHeadMaskablePolicy.__init__)
    assert "CRITIC_MODES" in src


def test_the_rollout_has_one_source_the_rust_collector():
    """`RolloutProbes.collect_rollouts` refuses a learner with no `_rust_collector` rather than falling into an
    upstream loop that would step a VecEnv that does not step (the Python collect was deleted)."""
    from types import SimpleNamespace

    from agents.training.instrumented_ppo.rollout_probes import RolloutProbes

    me = SimpleNamespace(num_timesteps=0, _rust_collector=None)
    with pytest.raises(RuntimeError, match="no Rust collector"):
        RolloutProbes.collect_rollouts(me, env=None, callback=None, rollout_buffer=None, n_rollout_steps=1)


def test_the_compile_opponents_trim_left_the_offline_core_and_the_pool_cache():
    """R6 deleted the TRAINER's opponent compile (flags, forkserver preload, prewarm, the revert quorum, the
    pool's compile wiring) — and KEPT `maybe_compile_extractor` for the offline readers and the pool's LRU."""
    from agents.model import compile_opponents as co
    from agents.training.snapshot_pool import SnapshotPool

    assert list(inspect.signature(co.maybe_compile_extractor).parameters) == ["model", "enabled", "label", "hide_cuda"]
    assert not any(hasattr(co, n) for n in ("arm_compile_quorum", "COMPILE_QUORUM_ENV", "CompileExtractorError"))
    params = inspect.signature(SnapshotPool.__init__).parameters
    assert "lru_cache_size" in params
    assert not {"compile_extractor", "compile_hide_cuda", "compile_strict"} & set(params)
