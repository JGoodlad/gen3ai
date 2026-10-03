"""Regression guard: BLAS threads must be pinned for env workers, on EVERY entry point.

THE MEASUREMENT THIS DEFENDS (2026-08-02, 16-core box, 8 neural-opponent envs, identical config):

    threads unpinned :   6 fps,  load average 110
    threads pinned   : 231 fps,  load average   3

Each SubprocVecEnv worker runs a full CPU opponent forward; at the library default of one thread per
core, N workers spawn N×cores competing threads and the box thrashes. `launcher/child.py` has always
exported OMP/MKL=1, so production under the launcher was fine — but `python src/main/train_rl_agent.py`
is a documented entry point (root CLAUDE.md, "Training — run directly") and had no protection, so the
~38× cliff was one forgotten `export` away on any direct run.

The module-level env vars in train_rl_agent (inherited by `spawn`ed children) are the guard. (The second,
`torch.set_num_threads(1)` inside the Python env worker's `_init`, went with the Python env core — deletion
pass U3.)
"""
import ast
import os
import pathlib


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
    # ASSERT THE ANCHOR, don't walk past it. The ordering check below is the whole point of this
    # test and it used to sit under `if torch_line is not None:` — so the day the hub's imports
    # stopped matching `_TORCH_BEARING_ROOTS` (a rename, a new indirection) the guard would go
    # quietly inert while still reporting green. The hub cannot NOT pull torch in: it builds and
    # trains the policy.
    assert torch_line is not None, (
        f"no torch-bearing import found in {_TRAIN.name} — either the hub stopped building the "
        f"model (impossible) or _TORCH_BEARING_ROOTS {_TORCH_BEARING_ROOTS} no longer names how "
        f"torch arrives. Until it does, the ordering assertion below is inert."
    )
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
