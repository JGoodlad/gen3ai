"""poke_env module init — LAZY (PEP 562), and the laziness is load-bearing.

This used to import `poke_env.player` (and through it `ps_client` → `concurrency`) eagerly, and
`concurrency` starts the GLOBAL asyncio loop thread the moment it is imported. Because importing
ANY submodule (`poke_env.battle.status`, `poke_env.data.normalize`, …) first executes this
package `__init__`, every consumer of a mere enum or string util inherited that thread — which is
what made `agents.model.features_extractor` unsafe to `fork()` from and killed the forkserver
compile preload (the hang is documented in `src/agents/training/CLAUDE.md` → Compiled CPU
opponents).

The public surface is UNCHANGED: `from poke_env import Player` still works — it just resolves on
first access (module `__getattr__`) instead of at package import. The loop thread now starts
exactly when something that actually needs it is imported (`poke_env.player` / `ps_client` /
`environment`), which is what every training-side consumer does anyway. The thread-free subtrees
(`poke_env.battle`, `poke_env.data`, `poke_env.exceptions`, `poke_env.stats`) stay thread-free.

`agents/model/compile_prewarm.py::extractor_import_is_fork_safe()` is the executable statement of
the invariant this buys; `compile_prewarm_test.py` pins it.
"""

import logging

__logger = logging.getLogger("poke-env")
__stream_handler = logging.StreamHandler()
__formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
__stream_handler.setFormatter(__formatter)
__logger.addHandler(__stream_handler)
logging.addLevelName(25, "PS_ERROR")

# public name -> the submodule that provides it. Resolution imports the submodule on FIRST
# attribute access; names never asked for never load (and never start threads).
_LAZY = {
    "AccountConfiguration": "poke_env.ps_client",
    "LocalhostServerConfiguration": "poke_env.ps_client.server_configuration",
    "MaxBasePowerPlayer": "poke_env.player",
    "Player": "poke_env.player",
    "RandomPlayer": "poke_env.player",
    "ServerConfiguration": "poke_env.ps_client.server_configuration",
    "ShowdownException": "poke_env.exceptions",
    "ShowdownServerConfiguration": "poke_env.ps_client.server_configuration",
    "SimpleHeuristicsPlayer": "poke_env.player",
    "compute_raw_stats": "poke_env.stats",
    "cross_evaluate": "poke_env.player",
    "gen_data": "poke_env.data",
    "to_id_str": "poke_env.data",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    import importlib

    target = _LAZY.get(name)
    if target is None:
        # `import poke_env; poke_env.<submodule>.X` used to work because the eager init imported
        # everything; resolve a submodule asked for as an attribute the same lazy way.
        try:
            value = importlib.import_module(f"poke_env.{name}")
        except ImportError:
            raise AttributeError(f"module 'poke_env' has no attribute {name!r}") from None
        globals()[name] = value
        return value

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value          # cache: subsequent accesses skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
