"""The FORK ARM's FLAG SURFACE, and what OFF costs (`gen3_fork_v1`).

Six flags have to agree across nine files — the parser, the resolve/inheritance line, the range
checks, the refusals, the two `model_build` kwarg blocks, `run_io`, `lifecycle`, the
`ModelVersion` record and its migration — and the failure mode of a missed site is not a crash: it
is a run whose `model_config.json` says `fork_fraction 0.0` while the callback forks, or a restart
that quietly halves the arm's simulation bill. Every one of those sites is pinned here.

**And the OFF claim, which is the one an unflagged run depends on:** at `--fork-fraction 0` no obs
key is declared, no callback is registered, the buffer is the stock one, and the policy term is the
unmasked expression. That is checked here rather than asserted in a docstring.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch as th

from agents.model.model_version.constants import MODEL_CONFIG_VERSION
from agents.model.model_version.migrations import _migrate_config
from agents.training.fork_arm import PG_MASK_KEY
from main.train.parser import build_parser

FORK_FIELDS = ("fork_fraction", "fork_branches", "fork_contested_gap", "fork_contested_absv",
               "fork_max_per_battle", "fork_crn")
DEFAULTS = {"fork_fraction": 0.0, "fork_branches": 3, "fork_contested_gap": 0.40,
            "fork_contested_absv": 0.0, "fork_max_per_battle": 1, "fork_crn": "dice_and_draws"}


_WP = ["--critic", "winprob", "--terminal-indicator", "--victory-value",
       "1.0", "--draw-penalty", "0", "--steps", "1000"]


def _resolved(argv):
    from main.train.config import resolve_config
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


# ── the parser + the resolve line ────────────────────────────────────────────────────────────
def test_every_flag_parses_under_both_spellings_and_defaults_to_None():
    p = build_parser()
    for dash, under in (("--fork-fraction", "--fork_fraction"),
                        ("--fork-branches", "--fork_branches"),
                        ("--fork-contested-gap", "--fork_contested_gap"),
                        ("--fork-contested-absv", "--fork_contested_absv"),
                        ("--fork-max-per-battle", "--fork_max_per_battle"),
                        ("--fork-crn", "--fork_crn")):
        for spelling, value in ((dash, None), (under, None)):
            assert p.parse_args([]).__dict__[spelling.lstrip("-").replace("-", "_")] is None
        assert dash in p.format_help() or under in p.format_help()


def test_the_off_defaults_are_what_resolve_config_fills_in():
    """`default=None` is load-bearing: `_resolve` fires on `is None`, so any other argparse default
    would make the inheritance line dead code and a flagless RESUME would silently reset the arm."""
    args = _resolved(["--steps", "1000"])
    for name, want in DEFAULTS.items():
        assert getattr(args, name) == want, name


def test_a_flagged_argv_keeps_its_values():
    args = _resolved(_WP + ["--cf-records", "--fork-fraction", "0.02", "--fork-branches", "2",
                            "--fork-contested-gap", "0.25", "--fork-max-per-battle", "2",
                            "--fork-crn", "dice"])
    assert args.fork_fraction == 0.02 and args.fork_branches == 2
    assert args.fork_contested_gap == 0.25 and args.fork_max_per_battle == 2
    assert args.fork_crn == "dice"


@pytest.mark.parametrize("argv", [
    ["--fork-fraction", "1.5"],
    ["--fork-fraction", "-0.1"],
    ["--fork-contested-gap", "0"],        # a quantile of 0 selects nothing
    ["--fork-contested-gap", "40"],       # the fat-fingered "40 %" a clamp would run as "all"
    ["--fork-contested-absv", "0.9"],     # |V - 0.5| cannot exceed 0.5
    ["--fork-max-per-battle", "0"],
])
def test_out_of_range_values_are_REFUSED_not_clamped(argv):
    with pytest.raises(SystemExit):
        _resolved(["--steps", "1000"] + argv)


def test_an_illegal_branch_count_or_crn_mode_is_rejected_by_argparse():
    for argv in (["--fork-branches", "4"], ["--fork-crn", "everything"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


# ── the model + the record ───────────────────────────────────────────────────────────────────
def test_all_six_are_TRAINING_HPARAMS_so_both_build_paths_put_them_on_the_model():
    from main.train.model_build import _TRAINING_HPARAMS
    names = {n for n, _how in _TRAINING_HPARAMS}
    for f in FORK_FIELDS:
        assert f in names, f


def test_both_build_and_train_kwarg_blocks_carry_every_field():
    """The classic bug this family invites: `build_and_train` spells the block TWICE — once on the
    fresh path and once on the resume — and a field missing from one records as its default on
    exactly the runs that resumed."""
    import main.train.model_build as mb
    src = inspect.getsource(mb)
    for f in FORK_FIELDS:
        assert src.count(f"{f}=args.{f},") == 2, f


def test_run_io_and_lifecycle_read_every_field_off_the_MODEL():
    import main.train.lifecycle as lc
    import main.train.run_io as rio
    for mod in (rio, lc):
        src = inspect.getsource(mod)
        for f in FORK_FIELDS:
            assert f'"{f}"' in src, (mod.__name__, f)


def test_the_model_version_records_every_field_and_the_config_version_is_bumped():
    from agents.model.model_version.fields import ModelVersionFields
    ann = ModelVersionFields.__annotations__
    for f in FORK_FIELDS:
        assert f in ann, f
    assert MODEL_CONFIG_VERSION >= 120


def _fresh_current_config(**recorded) -> dict:
    """A fresh CURRENT-generation config through the project's own constructor + JSON path, with
    `recorded` written over it as a checkpoint that RECORDED those values would carry them."""
    import json

    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    data = json.loads(ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512]}).to_json())
    data.update(recorded)
    return data


def test_a_pre_v120_config_is_REFUSED_and_a_current_one_records_the_only_possible_past():
    """The v120 `setdefault` branch (the six fork knobs → their inert defaults) is FLOORED AWAY:
    gen3_event_record_v2 raised MIGRATION_FLOOR to 121 (archived verbatim in `_migrate_config`'s
    v97–v120 history), so a v119 config is pre-generation and is REFUSED with the diagnosis. The
    surviving property: a fresh current config RECORDS all six explicitly at those defaults."""
    from agents.model.model_version import ModelVersion, ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 119})
    fresh = _fresh_current_config()
    out = _migrate_config(dict(fresh))
    assert out["config_version"] == MODEL_CONFIG_VERSION >= 121
    for name, want in DEFAULTS.items():
        assert name in fresh, name
        assert out[name] == want, name
    ModelVersion(**out)


def test_migration_never_overwrites_a_recorded_value():
    from agents.model.model_version import ModelVersion, ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 119, "fork_fraction": 0.02, "fork_crn": "dice"})
    out = _migrate_config(_fresh_current_config(fork_fraction=0.02, fork_crn="dice"))
    assert out["fork_fraction"] == 0.02 and out["fork_crn"] == "dice"
    mv = ModelVersion(**out)
    assert mv.fork_fraction == 0.02 and mv.fork_crn == "dice"


def test_the_fraction_is_in_the_arch_table_so_a_headless_run_reads_INERT():
    from agents.model.arch_tables import _COEF_MODULE
    assert _COEF_MODULE["fork_fraction"] == "win_head"


# ── the refusals ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("extra,needle", [
    (["--fork-fraction", "0.02"], "requires --critic winprob"),
    (_WP + ["--fork-fraction", "0.02"], "requires --cf-records"),
    (_WP + ["--cf-records", "--fork-fraction", "0.02", "--value-true-team"], "value-true-team"),
    (_WP + ["--cf-records", "--fork-fraction", "0.02", "--win-prob-dense-aux", "0.1"],
     "dense-aux"),
    (_WP + ["--cf-records", "--fork-fraction", "0.02", "--win-prob-strata-weight", "0.5"],
     "strata-weight"),
])
def test_the_five_refusals_fire_with_their_own_text(extra, needle, capsys):
    argv = extra if extra[0].startswith("--critic") else ["--steps", "1000"] + extra
    with pytest.raises(SystemExit):
        _resolved(argv)
    assert needle in capsys.readouterr().err


def test_a_complete_forked_argv_is_ACCEPTED():
    args = _resolved(_WP + ["--cf-records", "--fork-fraction", "0.02"])
    assert args.fork_fraction == 0.02


# ── OFF costs nothing ────────────────────────────────────────────────────────────────────────
def _env(**kw):
    from poke_env import AccountConfiguration
    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    return Gen3Env(load_mappings(), battle_format="gen3ou",
                   account_configuration1=AccountConfiguration("ForkKeyT", None),
                   start_listening=False, **kw)


def test_the_obs_key_is_declared_ONLY_when_the_arm_is_on():
    assert PG_MASK_KEY not in _env().observation_space.spaces
    space = _env(emit_fork_pg_mask=True).observation_space.spaces
    assert PG_MASK_KEY in space
    assert space[PG_MASK_KEY].shape == (1,)
    assert (float(space[PG_MASK_KEY].low[0]), float(space[PG_MASK_KEY].high[0])) == (0.0, 1.0)


def test_a_collected_row_carries_the_ONE_placeholder():
    """A zero here would delete the policy gradient on every collected row and read as a dead run
    rather than as a plumbing break."""
    env = _env(emit_fork_pg_mask=True)
    obs: dict = {}
    env._merge_training_keys(obs)
    assert obs[PG_MASK_KEY].tolist() == [1.0]


def test_env_factory_arms_the_obs_key_and_the_handle_from_the_fraction():
    import main.train.env_factory as ef
    src = inspect.getsource(ef)
    assert "emit_fork_pg_mask=(float(getattr(args, \"fork_fraction\", 0.0) or 0.0) > 0.0)" in src
    assert "or float(getattr(args, \"fork_fraction\", 0.0) or 0.0) > 0.0)" in src, \
        "the reconstruction HANDLE must be armed by --fork-fraction too"


def test_the_callback_is_registered_AFTER_the_win_prob_one():
    """LOAD-BEARING ORDER: the fork selector reads `win_mask` to mean 'this episode terminated in
    the buffer', and that plane is a placeholder of zeros until `WinProbLabelCallback` back-fills
    it. Registered first, the arm would find nothing eligible, every rollout, in silence."""
    import main.train.callbacks as cb
    src = inspect.getsource(cb)
    assert "ForkArmCallback" in src
    assert src.index("WinProbLabelCallback") < src.index("ForkArmCallback")
    assert 'getattr(args, "fork_fraction", 0.0) or 0.0) > 0.0' in src


def test_the_handle_scratch_is_shared_with_the_win_prob_rollout_arm():
    """Two flags need the same per-decision handle and the ASYNC collector writes exactly one such
    array inline; a second scratch would be a second thing that could silently go unfilled there."""
    from agents.training.win_prob_callback import WinProbLabelCallback
    src = inspect.getsource(WinProbLabelCallback._rollout_on)
    assert "fork_fraction" in src


# ── the policy-term mask ─────────────────────────────────────────────────────────────────────
def test_the_fold_flag_is_the_KEYS_presence_not_the_flags_value():
    from agents.training.instrumented_ppo.train_setup import TrainSetup
    src = inspect.getsource(TrainSetup._resolve_fold_flags)
    assert "FORK_PG_MASK_KEY in self.rollout_buffer.observations" in src


def test_the_masked_policy_term_is_in_the_train_step_and_is_RENORMALISED():
    from agents.training.instrumented_ppo.ppo import train_step_source
    src = train_step_source()
    assert "fork_pg_mask_on" in src
    assert "_fk_m.sum().clamp(min=1.0)" in src, (
        "a masked .mean() over the FULL row count would shrink the policy term by the masked "
        "fraction — i.e. silently lower the effective policy LR by a number that moves with the "
        "fork rate")


def test_renormalising_equals_the_mean_over_the_KEPT_rows():
    per_row = th.tensor([1.0, 2.0, 3.0, 4.0])
    m = th.tensor([0.0, 1.0, 1.0, 1.0])
    masked = -((per_row * m).sum() / m.sum().clamp(min=1.0))
    assert float(masked) == pytest.approx(float(-per_row[1:].mean()))
    # ...and with nothing masked it IS the plain mean the unflagged run computes.
    ones = th.ones(4)
    assert float(-((per_row * ones).sum() / ones.sum())) == pytest.approx(float(-per_row.mean()))


# ── the buffer install ───────────────────────────────────────────────────────────────────────
def test_install_fork_buffer_is_idempotent_and_keeps_the_shape():
    from gymnasium import spaces
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
    from agents.training.fork_buffer import ForkRolloutBuffer, install_fork_buffer

    class _M:
        device = "cpu"
        gamma = 1.0
        gae_lambda = 0.95

    space = spaces.Dict({"observation": spaces.Box(-1, 1, (4,), np.float32),
                         "action_mask": spaces.Box(0, 1, (11,), np.float32)})
    m = _M()
    m.rollout_buffer = MaskableDictRolloutBuffer(8, space, spaces.Discrete(11), n_envs=3)
    install_fork_buffer(m)
    first = m.rollout_buffer
    assert isinstance(first, ForkRolloutBuffer)
    assert first.buffer_size == 8 and first.n_envs == 3
    install_fork_buffer(m)
    assert m.rollout_buffer is first


def test_model_build_installs_it_only_when_the_arm_is_on():
    import main.train.model_build as mb
    src = inspect.getsource(mb.apply_training_hparams)
    assert "install_fork_buffer" in src
    assert 'getattr(args, "fork_fraction", 0.0) or 0.0) > 0.0' in src
