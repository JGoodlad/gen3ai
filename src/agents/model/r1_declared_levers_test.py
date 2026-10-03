"""K8 — R1's DECLARED LEVERS (`gen3_r1_declared_levers_v1`): every config that adds a key to the
learner's batch, or turns on a term R1 branches on, is DECLARED at startup.

THE HAZARD (the K6+K8 lane's final FINDING, 2026-10-01). The compiled learner micro-step (region R1)
is compiled `fullgraph=True` at its startup signature and LOCKED; anything new after the lock is a
FATAL. The startup gate and prewarm built R1's static flags with the win-prob ROW-WEIGHT levers
OFF (`_micro_static(f, None, False)`), while `train()` turned them on from the config — and
the strata lever only on a rollout whose labelled rows hold two or more opponent classes, i.e. a
DATA-dependent signature that flips the first time the self-play pool seeds. Every such run reached
its first (or its first post-seed) update and died.

THE FIX IS A DECLARATION, never a relaxed lock: ONE predicate (`TrainSetup._r1_levers`) resolves the
levers from the config and the buffer's KEY set for `train()` AND for the startup declaration
(`compile_regions.r1_declaration`), the strata weights get a NEUTRAL default (ones — bit-identical,
proven below) when a rollout yields none, and every update is held to the declaration
(`check_r1_declared`, a typed `CompileSentinelError`).

THE TEST, per inventory row: build the learner FROM THAT CONFIG (the real parser + `resolve_config`,
its spaces, policy kwargs and training hparams), fill a CPU rollout (the K9 golden's real rows plus
the lever's keys), run the real startup lifecycle (install, gate, prewarm, LOCK — dynamo's `eager`
backend, CPU) and two REAL updates; 0 compiles / rejections after the lock, and the declared
signature carries the lever. Reverting the declaration (the gate's lever back to OFF) fails the
`strata` row with the sentinel's FATAL. (The rollout-weight lever, the entropy boosts and the
dense-aux / true-team keys were rows here until deletion pass L2.)
"""
from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr


N_ENVS, BATCH = 4, 16

#: THE INVENTORY — every config that puts a non-production key in the learner's batch or turns on a
#: lever R1 reads. (name, extra argv, what the declaration must carry).
LEVERS: Dict[str, Any] = {
    "production": ([], {}),
    "strata": (["--win-prob-strata-weight", "0.5"], {"strata": True}),
    # gen3_fork_rust_v1 (forks.md §14): on the Rust core the branch rows ride the complete-game FIFO
    # inside the FIXED update (no ragged tail) — the mask key is still a lever.
    # (The python-core fork row left with L4/L5: the python arm is deleted and a fork on the python core is
    # refused, `combination_checks` `fork_python_core_unavailable`.)
    "fork_rust": (["--env-core", "rust", "--fork-fraction", "0.01"], {"fork_pg_mask": True, "key": "fork_pg_m"}),
}


def _lever_args(name: str) -> Any:
    from main.rust_core_cutover.envs import PRODUCTION_ARGV
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    parser = build_parser()
    extra = list(LEVERS[name][0])
    a = parser.parse_args(list(PRODUCTION_ARGV) + extra)
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, parser)
    a.use_bridge, a.bridge_impl, a.use_showdown_bridge = "rust", "rust", True
    return a


def _fill(model: Any, *, mixed_classes: bool, seed: int) -> None:
    """The K9 golden's real rows for every key it holds; the lever's own keys at plausible values
    (multipliers in [1, 2] or {0, 1}, flags {0, 1}); `opp_class` ONE class or two."""
    from agents.training import learner_golden as LG
    from agents.training.fork_arm import PG_MASK_KEY
    rb = model.rollout_buffer
    rb.reset()
    rng = np.random.default_rng(seed)
    with np.load(LG.BUFFER_PATH) as z:
        data = {k: z[k] for k in z.files}
    for k, arr in rb.observations.items():
        g = data.get("obs:" + k)
        if g is not None and g.shape == arr.shape:
            arr[...] = g
        elif k == PG_MASK_KEY:
            arr[...] = rng.integers(0, 2, arr.shape)
        # anything else stays at zeros: an unsupervised label (mask 0)
    oc = rb.observations["opp_class"]
    oc[...] = 0
    if mixed_classes:
        oc[:, 1::2] = 1                                     # bots on even envs, the pool on odd ones
    for f in LG._BUFFER_FIELDS:
        getattr(rb, f)[...] = data[f]
    rb.full, rb.pos = True, LG.N_STEPS
    model._current_progress_remaining = 1.0


@pytest.fixture
def lifecycle():
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    made: List[Any] = []
    try:
        yield made
    finally:
        for m in made:
            cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _locked_learner(name: str, made: List[Any]) -> Any:
    from agents.training import learner_golden as LG
    model = LG.build_learner(args=_lever_args(name))
    made.append(model)
    _fill(model, mixed_classes=False, seed=0)
    ctl = cc.control()
    ctl.install()
    cr.install(model, backend="eager")
    cr.gate_regions(model, n_envs=N_ENVS, batch_size=BATCH, say=lambda _m: None)
    ctl.prewarm(cr.prewarm_calls(model, n_envs=N_ENVS, batch_size=BATCH))
    ctl.lock("test: the end of startup")
    return model


@pytest.mark.parametrize("name", list(LEVERS))
def test_the_startup_declaration_COVERS_every_update_the_lever_reaches(name, lifecycle):
    model = _locked_learner(name, lifecycle)
    ctl = cc.control()
    tags: List[set] = []
    # update 1: ONE opponent class (any run before the pool seeds — the strata lever yields no
    # weights); update 2: two classes (it does). The SAME declared signature both times.
    from agents.model import region_calls
    for i, mixed in enumerate((False, True)):
        _fill(model, mixed_classes=mixed, seed=i)
        region_calls.take()          # the per-update route window (`compile_control.attach` resets it)
        with ctl.guard("update end"):
            model.train()
        tags.append({k for k in model.logger.name_to_value if "strata_row_w_mean" in k})
    ctl.check("end")
    assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0, ctl.after_lock_frames
    # non-vacuous: the declaration CARRIES the lever (and its key is in the batch R1 compiled for)
    want, decl = LEVERS[name][1], model._r1_declared
    for field in ("strata", "fork_pg_mask"):
        assert getattr(decl.static, field) is bool(want.get(field, False)), (name, field, decl.static)
    if "key" in want:
        assert want["key"] in decl.obs_keys, (name, decl.obs_keys)
    if name == "strata":
        # the lever is LIVE on update 2 and idle on update 1 — its diagnostics say so, as before
        assert not tags[0] and tags[1], tags


def test_an_UNDECLARED_lever_or_key_after_the_lock_is_the_typed_FATAL(lifecycle):
    """A lever turned on after startup, or a key that appears in the buffer, is a typed
    `CompileSentinelError` naming it — never a silent recompile. Fails if `check_r1_declared` is not
    wired into `train()` (the dynamo sentinel's guard dump is the only backstop then)."""
    model = _locked_learner("production", lifecycle)
    model.win_prob_strata_weight = 0.5                      # a lever the config did not declare
    _fill(model, mixed_classes=True, seed=0)
    with pytest.raises(cc.CompileSentinelError, match=r"strata: declared False, this update True"):
        model.train()
    model.win_prob_strata_weight = 0.0
    _fill(model, mixed_classes=True, seed=0)
    model.rollout_buffer.observations["late_key"] = np.zeros((16, N_ENVS, 1), dtype=np.float32)
    with pytest.raises(cc.CompileSentinelError, match=r"observation keys: undeclared \['late_key'\]"):
        model.train()


def test_the_NEUTRAL_strata_weights_are_BIT_IDENTICAL_to_the_unweighted_expression():
    """The strata lever declared but idle (a one-class rollout): R1 runs with ones and
    `strata_active` False. Loss, every gradient and the PRESENT metric set equal the undeclared
    expression's bit for bit — so the neutral default changes no number and no TB tag."""
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.micro_step import micro_step, pack, unpack
    model = LG.build_learner()
    LG.load_buffer_into(model)
    b = cr.r1_batch(model, BATCH)
    f = model._resolve_fold_flags()

    def run(strata: bool) -> Dict[str, Any]:
        st = model._micro_static(f, strata)
        var = model._micro_var(st, None)
        assert ("strata_w" in var) is strata
        model.policy.set_training_mode(True)
        for p in model.policy.parameters():
            p.grad = None
        torch.manual_seed(0)
        out = micro_step(model.policy, b.obs, b.actions, b.action_masks, b.old_log_prob,
                         b.old_values, b.advantages, b.returns, var, st)
        out.loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else None
                 for p in model.policy.parameters()]
        vals, pres, finite = unpack(*pack(out))
        return {"loss": out.loss.detach().clone(), "grads": grads, "vals": vals, "pres": pres}

    off, idle = run(False), run(True)
    assert torch.equal(off["loss"], idle["loss"])
    assert len(off["grads"]) == len(idle["grads"])
    for g0, g1 in zip(off["grads"], idle["grads"]):
        assert (g0 is None) == (g1 is None)
        if g0 is not None:
            assert torch.equal(g0, g1)
    assert off["vals"] == idle["vals"] and off["pres"] == idle["pres"]
    assert any(k.startswith("win_prob/") for k in off["vals"])      # the BCE is live: not vacuous
