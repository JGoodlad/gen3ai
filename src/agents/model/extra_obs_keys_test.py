"""The DECLARED extra-obs-key registry, and the crash it exists to make unrepeatable.

THE DEFECT THIS FILE PINS (`ai_v12_14_ladder_truevalue`, launched at 377a5aa1, child dead two
minutes in at env init, exit 1). Two independently-correct facts had never been exercised
together:

* `gen3_value_true_team_v1`'s value route reads its own Dict key, `opp_true_team`, and **RAISES**
  on a missing one — deliberately, because a silent skip reads exactly like a route that learned
  nothing (the gen-12 dead-tail bug).
* `gen3_forkserver_preload_v1` traced the compiled extractor once, on a hand-built **one-key**
  obs, so 48 workers could inherit the graph.

`--value-true-team` + the preload therefore killed the forkserver bootstrap, and `SubprocVecEnv`
construction failed in the parent. The same launch's `--compile-trainer`, `--compile-opponents`
and `--warmstart-battles` carried the identical hand-built dict, so fixing only the preload would
have moved the crash rather than removed it.

The fix is not "add the key at four sites" — it is that the (attribute -> key, shape) mapping is
DECLARED once in `extra_obs_keys` and every synthetic-obs caller builds from it. These tests hold
both ends: the declaration matches what the forward actually reads (the AST gate), and what the
declaration produces matches what a REAL `Gen3Env` emits (the key-set gate).
"""
import ast
import inspect

import pytest
import torch
from poke_env import AccountConfiguration

from agents.model import extra_obs_keys as EOK
from agents.model.compile_preload import build_preload_extractor, preload_trace_obs
from agents.model.extra_obs_keys import (
    BASE_OBS_KEYS, EXTRA_OBS_KEYS, required_extra_obs_keys, synthetic_obs,
)
from agents.observation.state_encoder import load_mappings
from agents.observation.true_team import TRUE_TEAM_KEY
from agents.training.gen3_env import Gen3Env

# Every flag combination that ADDS an obs key. One row per configuration the preload can be
# handed; `value_true_team` is the only such flag today, so the table is the pair — the point of
# the table is that a second one appends here rather than forking a test.
KEY_ADDING_CONFIGS = [
    pytest.param({}, set(), id="no-extra-keys"),
    pytest.param({"value_true_team": True}, {TRUE_TEAM_KEY}, id="value_true_team"),
]

#: flag name -> the `Gen3Env` kwarg that makes the env EMIT the same key. Declared rather than
#: derived: the env's emit flags are named independently of the extractor's, and a test that
#: guessed the mapping would pass on a rename that broke the run.
ENV_EMIT_KWARG = {"value_true_team": "emit_opp_true_team"}


def _fe(**cfg):
    """The extractor the FORKSERVER builds, through the preload's own entry point."""
    return build_preload_extractor(cfg)


def _env(**kw):
    return Gen3Env(load_mappings(), battle_format="gen3ou",
                   account_configuration1=AccountConfiguration("ExtraObsKeyT", None),
                   start_listening=False, **kw)


# ------------------------------------------------------------------ the declaration itself
def test_every_declared_row_names_a_REAL_extractor_attribute_that_the_flag_toggles():
    """`ExtraObsKey.attr` is the enable condition the forward tests. If it is not built by the
    flag — or not `None` without it — the registry answers for a different question than the seam
    asks, and `required_extra_obs_keys` silently under- or over-builds."""
    for entry in EXTRA_OBS_KEYS:
        off, _ = _fe()
        on, _ = _fe(**{entry.flag: True})
        assert getattr(off, entry.attr, "missing") is None, (
            f"{entry.key}: {entry.attr} is not None with --{entry.flag.replace('_', '-')} OFF, so "
            "required_extra_obs_keys would demand a key no forward reads")
        assert getattr(on, entry.attr, None) is not None, (
            f"{entry.key}: --{entry.flag.replace('_', '-')} did not build {entry.attr}")


def test_the_AST_of_the_forward_reads_no_UNDECLARED_obs_key():
    """THE DRIFT GATE. Every `obs.get(X)` / `obs[X]` in `extractor_forward` must resolve to a key
    this registry declares (or a base key). A new route that reads a new key therefore cannot
    reach a launch without a row here — which is the whole mechanism, since the four synthetic-obs
    callers build from the rows and nothing else."""
    from agents.model import extractor_forward as EF

    src = inspect.getsource(EF)
    tree = ast.parse(src)
    declared = set(BASE_OBS_KEYS) | {e.key for e in EXTRA_OBS_KEYS}
    found, unresolved = set(), []

    def _keyname(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            val = getattr(EF, node.id, None)   # a module-level constant like TRUE_TEAM_KEY
            if isinstance(val, str):
                return val
        return None

    for node in ast.walk(tree):
        arg = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "obs" and node.args):
            arg = node.args[0]
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "obs"):
            arg = node.slice
        if arg is None:
            continue
        name = _keyname(arg)
        if name is None:
            unresolved.append(ast.dump(arg))
        else:
            found.add(name)

    assert not unresolved, (
        "extractor_forward reads an obs key this gate cannot resolve to a literal — spell the key "
        f"out as a module-level constant so the registry can be checked against it: {unresolved}")
    assert found, "the AST gate found NO obs key reads at all — it has stopped checking anything"
    undeclared = found - declared
    assert not undeclared, (
        f"extractor_forward reads obs key(s) {sorted(undeclared)} that agents.model.extra_obs_keys "
        "does not declare. Every synthetic-obs caller (the forkserver preload, the round-trip "
        "smoke, compile_trainer, compile_opponents, warmstart) builds its dict from that table, "
        "so an undeclared key is a crash at the next launch that turns the flag on.")
    # …and the converse: a row for a key nothing reads would make every caller build dead weight.
    assert {e.key for e in EXTRA_OBS_KEYS} <= found, (
        "extra_obs_keys declares a key extractor_forward never reads: "
        f"{sorted({e.key for e in EXTRA_OBS_KEYS} - found)}")


# ------------------------------------------------------------------ (a) the forward runs
@pytest.mark.parametrize("cfg,extra", KEY_ADDING_CONFIGS)
def test_the_synthetic_preload_obs_FORWARDS(cfg, extra):
    """(a) For every key-adding configuration, the obs the preload traces on runs the real
    forward. Eager, not compiled: the failure is in the obs DICT, not in the codegen, and a
    30-second regression test is one nobody runs."""
    fe, layout = _fe(**cfg)
    obs = preload_trace_obs(fe, layout)
    assert set(obs) == {"observation"} | extra
    with torch.no_grad():
        pi, vf = fe(obs)
    assert pi.shape[0] == vf.shape[0] == 1


def test_the_batch_and_device_arguments_are_honoured():
    """`compile_trainer` warms at a real batch and `warmstart` tops up a chunk of REAL obs — both
    would produce a broadcast error, not a crash, if the block came back at batch 1."""
    fe, layout = _fe(value_true_team=True)
    obs = synthetic_obs(fe, int(layout["total_dim"]), batch=5, action_mask=True)
    assert obs["observation"].shape[0] == 5
    assert obs["action_mask"].shape == (5, 11)
    assert obs[TRUE_TEAM_KEY].shape == (5,) + tuple(EOK.BY_KEY[TRUE_TEAM_KEY].shape)
    with torch.no_grad():
        pi, _ = fe(obs)
    assert pi.shape[0] == 5


# ------------------------------------------------------------------ (b) it matches the real env
@pytest.mark.parametrize("cfg,extra", KEY_ADDING_CONFIGS)
def test_the_synthetic_key_set_matches_what_a_REAL_env_EMITS(cfg, extra):
    """(b) KEY SETS, not values. The synthetic obs must carry exactly the keys a real `Gen3Env`
    puts in its Dict space under the same configuration — restricted to the keys the EXTRACTOR
    reads, because the env also emits privileged LABEL keys (belief_species, win_target, …) that
    only the PPO loss consumes and no forward touches.

    Both directions are checked, and both have bitten: a key the extractor reads and the env does
    not emit is a live-run crash on the first decision; a key the env emits and the synthetic obs
    omits is a dynamo re-trace at best and this file's launch crash at worst."""
    fe, layout = _fe(**cfg)
    env_kw = {ENV_EMIT_KWARG[flag]: True for flag in cfg}
    env = _env(**env_kw)
    env_keys = set(env.observation_space.spaces)
    readable = set(BASE_OBS_KEYS) | {e.key for e in EXTRA_OBS_KEYS}
    synthetic = set(synthetic_obs(fe, int(layout["total_dim"]), action_mask=True))

    assert synthetic == env_keys & readable, (
        f"synthetic {sorted(synthetic)} != the env's extractor-readable keys "
        f"{sorted(env_keys & readable)} (env space: {sorted(env_keys)})")
    # every key the extractor DEMANDS is one the env actually supplies…
    demanded = {e.key for e in required_extra_obs_keys(fe)}
    assert demanded == extra and demanded <= env_keys, (
        f"the extractor demands {sorted(demanded)}; the env emits {sorted(env_keys)}")
    # …at the same shape, so the declared block is the one the env writes.
    for entry in required_extra_obs_keys(fe):
        assert tuple(env.observation_space[entry.key].shape) == tuple(entry.shape)


# ------------------------------------------------------------------ the regression
def test_the_PRE_FIX_one_key_trace_input_is_exactly_the_launch_crash():
    """THE REGRESSION. Reverting `preload_trace_obs` to the literal it replaced —

        obs = {"observation": torch.zeros(1, layout["total_dim"])}

    — reproduces `ai_v12_14_ladder_truevalue`'s death: the privileged route raises, which inside
    the forkserver kills the bootstrap. The second half is what fails on a revert of the fix."""
    fe, layout = _fe(value_true_team=True)
    pre_fix = {"observation": torch.zeros(1, int(layout["total_dim"]))}
    with pytest.raises(RuntimeError, match=TRUE_TEAM_KEY):
        with torch.no_grad():
            fe(pre_fix)
    # …and the shipped trace input does not.
    obs = preload_trace_obs(fe, layout)
    assert TRUE_TEAM_KEY in obs
    with torch.no_grad():
        fe(obs)


def test_the_round_trip_smoke_and_the_three_warmups_all_build_from_the_registry():
    """The four TRAINING-RUN synthetic-obs sites are the ones whose crash costs a GPU-hour. Pinned
    by SOURCE because there is no cheap way to run `compile_trainer` / `compile_opponents` /
    `warmstart` / the round-trip smoke in a unit test — and a source check is exactly strong
    enough for the claim, which is 'this site does not hand-build its dict any more'."""
    from agents.model import compile_opponents, compile_trainer
    from agents.training import warmstart
    from main.train import lifecycle

    for mod, fn in ((compile_trainer, "zero_extra_obs"), (compile_opponents, "zero_extra_obs"),
                    (warmstart, "zero_extra_obs"), (lifecycle, "synthetic_obs")):
        assert fn in inspect.getsource(mod), (
            f"{mod.__name__} no longer builds its synthetic obs from agents.model.extra_obs_keys "
            "— it is one obs-key flag away from the ai_v12_14_ladder_truevalue crash")
