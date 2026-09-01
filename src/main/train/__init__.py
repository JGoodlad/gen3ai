"""The training entry point, decomposed.

`main.train_rl_agent` is the HUB: it keeps its path, its `main()` and `build_parser()`, and
re-exports every name that used to live in it, so no import path and no recorded
`launcher_command` changed. The phases live here, one module per concern:

    constants.py        BATTLE_FORMAT / the smoke-eval scale / the abort drain bound
    parser/             `build_parser()` behind a hub, one module per FLAG FAMILY in `--help`
                        order (base + operational, hyperparameters, reward, clean_world,
                        teacher, cf_grounding, value_heads, capacity, distillation,
                        eval_subprocess); `base.py` holds the three custom argparse pieces
    compile_flags.py    the `--compile-opponents` / `--compile-trainer` default resolvers
    checkpoint_state.py reading a checkpoint's saved arch; the by-NAME optimizer realign
    run_io.py           the run directory, latest.txt, the TB logger, the checkpoint callback
    lifecycle.py        grad checkpointing, the trainer compile, the round-trip smoke, signals
    config.py           phase 1 — desugar / `_resolve` / validate
    matchup_setup.py    phase 2 — teams, the matchup, every opponent source
    env_factory.py      phase 3 — the per-worker training-env `_init` closure
    callbacks.py        phase 4 — everything that runs during `learn()`
    model_build.py      phase 5 — the resume + fresh model paths, and `learn()` itself
    final_eval.py       the post-training win-rate evaluation

**`entry_source()` is the seam for source-scanning gates.** Several tests assert about the
training entry point by READING it (the flag-registry surface check, the `--edge-bias-families`
validator, the policy-kwargs AST pin, the `learn()`-budget pin). Those gates are about the entry
point, not about one file, so they read the whole package through here — otherwise a
decomposition silently empties them, which is the exact failure mode the file-size gate's
allowlist rule warns about.
"""
import pathlib
from typing import List

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_HUB = _PKG_DIR.parent / "train_rl_agent.py"


def entry_source_files() -> List[pathlib.Path]:
    """Every file the training entry point is made of: the hub, then the phase modules by name.

    RECURSIVE, because a phase may itself be a PACKAGE — `parser/` became one, and its
    `__init__.py` carries `build_parser()` itself. Only THIS package's own `__init__.py` is
    skipped (it is the map, not a phase); a sub-package's is a phase module like any other.
    A non-recursive glob here would have silently dropped all 229 `add_argument` calls out of
    every source-scanning gate.
    """
    return [_HUB] + sorted(
        p for p in _PKG_DIR.rglob("*.py")
        if p != _PKG_DIR / "__init__.py" and "__pycache__" not in p.parts
    )


def entry_source_modules() -> List[str]:
    """The same files as DOTTED module names, for gates that import rather than read them."""
    out = []
    for p in entry_source_files():
        if p == _HUB:
            continue
        rel = p.relative_to(_PKG_DIR).with_suffix("")
        parts = [q for q in rel.parts if q != "__init__"]
        out.append(".".join(["main", "train", *parts]))
    return out


def entry_source() -> str:
    """The concatenated source of the whole entry point (hub + phases)."""
    return "\n".join(p.read_text(encoding="utf-8") for p in entry_source_files())
