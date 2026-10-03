"""The Python env core's flags (deletion pass U3, manifest R1 / R6): deleted, or reduced to ONE legal value
that REFUSES the deleted one at PARSE time with the reason.

A silent no-op is the failure this file exists to rule out. After U3 the trainer has one env core, so a
typed `--env-core python` / `--use-bridge node` / `--critic shaped` would otherwise parse, change nothing,
and read as a run on the core the command named. Each case below fails on a revert of the piece it names.
"""
from __future__ import annotations

import inspect

import pytest

from main.train.parser import build_parser

#: Spelled by concatenation: `claude_md_freshness_gate_test` treats a bare `"--flag"` constant in a
#: non-test module as live CLI surface; these are deleted flags, kept here only to prove they stay deleted.
_DELETED = ["--" + n for n in ("async-rollout", "obs-source", "compile-opponents",
                               "compile-opponents-preload", "compile-opponents-strict")]


@pytest.mark.parametrize("flag,value", [("--env-core", "python"), ("--use-bridge", "node"),
                                        ("--use-bridge", "off"), ("--critic", "shaped")])
def test_a_deleted_value_is_refused_at_parse_time_with_the_reason(flag, value, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--steps", "1", flag, value])
    assert e.value.code == 2
    err = " ".join(capsys.readouterr().err.split())
    assert f"{flag} '{value}' was DELETED" in err and "only legal value is" in err, err
    assert "U3" in err, err                                    # names the pass, so the reader can find why


@pytest.mark.parametrize("flag,value", [("--env-core", "rust"), ("--use-bridge", "rust"), ("--critic", "winprob")])
def test_the_one_legal_value_still_parses(flag, value):
    ns = build_parser().parse_args(["--steps", "1", flag, value])
    assert getattr(ns, flag[2:].replace("-", "_")) == value


@pytest.mark.parametrize("flag", _DELETED + ["--no-" + _DELETED[0][2:], "--no-" + _DELETED[2][2:]])
def test_a_deleted_flag_is_unrecognised(flag, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "1", flag])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_the_defaults_are_the_one_core():
    ns = build_parser().parse_args(["--steps", "1"])
    assert ns.env_core == "rust" and ns.use_bridge == "rust" and ns.critic is None   # None = resolve_critic_mode


def test_the_trainable_critics_are_a_strict_subset_of_the_loadable_ones():
    """`shaped` stays LOADABLE (every pre-v109 checkpoint, every shaped-era opponent) but is not trainable."""
    from agents.model.critic_mode import (CRITIC_MODES, CRITIC_SHAPED, CRITIC_TRAINABLE_MODES, CRITIC_UNRECORDED)

    assert set(CRITIC_TRAINABLE_MODES) < set(CRITIC_MODES) and CRITIC_SHAPED in CRITIC_MODES
    assert CRITIC_SHAPED not in CRITIC_TRAINABLE_MODES
    assert CRITIC_UNRECORDED == CRITIC_SHAPED, "an ABSENT record still means shaped: old checkpoints must load as such"


def test_the_policy_still_loads_a_shaped_critic_checkpoint():
    from agents.model.policy import Gen3DualHeadMaskablePolicy

    assert "critic" in inspect.signature(Gen3DualHeadMaskablePolicy.__init__).parameters
    # the policy's own validation takes the LOADABLE set, never the trainable one
    src = inspect.getsource(Gen3DualHeadMaskablePolicy.__init__)
    assert "CRITIC_MODES" in src and "CRITIC_TRAINABLE_MODES" not in src


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
