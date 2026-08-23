"""Regression guard: BLAS threads must be pinned for env workers, on EVERY entry point.

THE MEASUREMENT THIS DEFENDS (2026-08-02, 16-core box, 8 neural-opponent envs, identical config):

    threads unpinned :   6 fps,  load average 110
    threads pinned   : 231 fps,  load average   3

Each SubprocVecEnv worker runs a full CPU opponent forward; at the library default of one thread per
core, N workers spawn N×cores competing threads and the box thrashes. `launcher/child.py` has always
exported OMP/MKL=1, so production under the launcher was fine — but `python src/main/train_rl_agent.py`
is a documented entry point (root CLAUDE.md, "Training — run directly") and had no protection, so the
~38× cliff was one forgotten `export` away on any direct run.

Two independent guards, both pinned here because they fail differently:
  * module-level env vars in train_rl_agent (inherited by `spawn`ed workers) — the primary;
  * `torch.set_num_threads(1)` inside the worker `_init` — survives an explicit OMP override that a
    user sets for the LEARNER, which must not silently un-pin every worker.
"""
import ast
import os
import pathlib

from main.train import entry_source

_TRAIN = pathlib.Path(__file__).with_name("train_rl_agent.py")   # the HUB (where the pin lives)
_LAUNCHER_CHILD = pathlib.Path(__file__).with_name("launcher") / "child.py"
_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")

# What the pin has to run BEFORE. A bare `import torch` is the obvious one, but it is no longer the
# only one and since the 2026-08-22 decomposition it is not even present in the hub: `import
# stable_baselines3…` and the `main.train.*` phase modules all pull torch in TRANSITIVELY, and BLAS
# reads its thread count the moment it initialises regardless of which import got it there. Naming
# only `torch` would have left this guard inert the day the hub stopped importing it directly —
# which is exactly what happened, and is why the list is by EFFECT rather than by spelling.
_TORCH_BEARING_ROOTS = ("torch", "stable_baselines3", "sb3_contrib", "agents", "main.train",
                        "utils.bridge")


def _pulls_in_torch(module_name: str) -> bool:
    return any(module_name == r or module_name.startswith(r + ".") for r in _TORCH_BEARING_ROOTS)


def test_launcher_child_still_pins_threads():
    """The production path's guard — the one that made this invisible for so long."""
    src = _LAUNCHER_CHILD.read_text()
    for var in _VARS:
        assert f'env["{var}"] = "1"' in src, f"launcher/child.py no longer pins {var}"


def test_train_rl_agent_pins_threads_at_import_time():
    """The direct-run guard. It must be at MODULE level and BEFORE torch is imported: BLAS reads these
    when it initialises, so setting them after `import torch` is a no-op."""
    tree = ast.parse(_TRAIN.read_text())
    pin_line = torch_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in _VARS and pin_line is None:
            pin_line = node.lineno
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])] + [getattr(node, "module", "") or ""]
            if any(_pulls_in_torch(n) for n in names):
                torch_line = node.lineno if torch_line is None else min(torch_line, node.lineno)
    assert pin_line is not None, (
        "train_rl_agent.py no longer sets OMP/MKL_NUM_THREADS at import — a direct run will thrash "
        "(measured 6 fps vs 231)."
    )
    if torch_line is not None:
        assert pin_line < torch_line, (
            f"thread pinning at line {pin_line} runs AFTER torch is imported at line {torch_line}; "
            f"BLAS has already read its thread count by then, so the pin is a no-op."
        )


def test_pinning_uses_setdefault_not_hard_assignment():
    """An explicit user value must still win — the pin is a floor for the unset case, not a policy."""
    src = _TRAIN.read_text()
    assert "setdefault" in src.split("import torch")[0], (
        "the import-time pin should use os.environ.setdefault so an explicit override is honoured"
    )


def test_env_worker_pins_torch_threads_independently():
    """Because the env vars are setdefault-only, a learner-side override would otherwise un-pin every
    worker. The worker `_init` must call torch.set_num_threads(1) itself.

    Reads the whole entry point rather than the hub: `_init` moved into
    `main/train/env_factory.py` with the 2026-08-22 decomposition.
    """
    src = entry_source()
    init_idx = src.index("def _init():")
    body = src[init_idx:init_idx + 2000]
    assert "set_num_threads(1)" in body, (
        "the env-worker _init no longer pins torch.set_num_threads(1) — an explicit OMP_NUM_THREADS "
        "for the learner would silently make every worker's B=1 opponent forward multi-threaded."
    )


def test_importing_train_rl_agent_actually_sets_them():
    """End-to-end: importing the module (as a spawned worker does) leaves the vars set."""
    prev = {v: os.environ.pop(v, None) for v in _VARS}
    try:
        import importlib
        import main.train_rl_agent as t
        importlib.reload(t)
        for var in _VARS:
            assert os.environ.get(var) == "1", f"{var} not set after importing train_rl_agent"
    finally:
        for var, val in prev.items():
            if val is not None:
                os.environ[var] = val
