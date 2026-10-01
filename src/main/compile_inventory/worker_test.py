"""The two worker helpers that decide WHAT is measured: the buffer slice (real rows, the
[n_steps, n_envs] structure kept) and the trainer argv (few env workers, the learner compile ON
only for the time stage)."""
from __future__ import annotations

import numpy as np

from main.compile_inventory.__main__ import WORKER_N_ENVS, adjust_argv
from main.compile_inventory.worker import slice_buffer_state


def test_slice_keeps_the_first_steps_of_every_rollout_array():
    n, e = 8, 3
    state = {"buffer_size": n, "n_envs": e, "pos": n, "full": True, "generator_ready": True,
             "advantages": np.arange(n * e, dtype=np.float32).reshape(n, e),
             "observations": {"observation": np.ones((n, e, 5), np.float32),
                              "win_mask": np.zeros((n, e, 1), np.float32)},
             "gamma": 1.0}
    out = slice_buffer_state(state, 2)
    assert out["buffer_size"] == 2 and out["pos"] == 2 and out["generator_ready"] is False
    assert out["advantages"].shape == (2, e)
    assert (out["advantages"] == state["advantages"][:2]).all()
    assert out["observations"]["observation"].shape == (2, e, 5)
    assert out["n_envs"] == e and out["gamma"] == 1.0
    assert state["advantages"].shape == (n, e)             # the source is untouched


def test_slice_is_a_no_op_at_or_above_the_buffer():
    state = {"buffer_size": 4, "advantages": np.zeros((4, 2))}
    assert slice_buffer_state(state, 4) is state


def test_argv_drops_env_count_and_compile_flags_then_sets_them_once():
    argv = ["--n-envs", "48", "--compile-trainer", "--compile-opponents",
            "--compile-opponents-strict", "--lr", "0.0003", "--device", "cuda"]
    t = adjust_argv(argv, compile_trainer=True)
    assert t.count("--n-envs") == 1 and t[t.index("--n-envs") + 1] == WORKER_N_ENVS
    assert "--compile-trainer" in t and "--no-compile-trainer" not in t
    assert "--compile-opponents" not in t and "--no-compile-opponents" in t
    assert t[t.index("--lr") + 1] == "0.0003"
    f = adjust_argv(argv, compile_trainer=False)
    assert "--no-compile-trainer" in f and "--compile-trainer" not in f
