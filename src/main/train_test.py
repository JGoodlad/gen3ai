"""The `main/train/` package's CONTRACT with the `train_rl_agent.py` hub that re-exports it.

The 2026-08-22 decomposition took a 4,667-line entry point down to a ~350-line orchestrator over
one module per concern. Two things make that safe rather than merely tidier, and neither is
self-enforcing, so both are pinned here:

**1. The hub still exports everything it used to.** `from main.train_rl_agent import <anything>`
has to keep resolving — for the tests that import `_write_latest_txt` and
`_TrackingCheckpointCallback`, for `main.checkargs`' `build_parser`, and for anyone reading a
years-old command. The list below is the module-level name set of the file as it stood at the
commit before the split, transcribed once. It may GROW; a name leaving it is a deliberate deletion
that should fail here first.

**2. No phase module imports the hub.** The hub imports every phase; a phase importing the hub back
closes a cycle whose symptom is a partially-initialised module at import time — an
`AttributeError` on a name that plainly exists, from a traceback that points at the wrong file.
It cannot happen by accident today, which is exactly why it needs a gate: the shape is one
convenience import away and nothing else in the tree would notice.

Milliseconds — a name lookup and an AST walk. No marker, runs in the fast inner loop.
"""
import ast

import pytest

from main.train import entry_source, entry_source_files

_PKG = "main.train"
_HUB = "main.train_rl_agent"

# Every module-level name `src/main/train_rl_agent.py` defined at 586e682, the commit before the
# decomposition. Recovered by AST, not by hand, and transcribed here verbatim.
_PRE_SPLIT_EXPORTS = (
    "BATTLE_FORMAT", "BoolFlag", "CLIP_RANGE_DEFAULT", "DEFAULT_EVAL_BATTLES",
    "SMOKE_EVAL_BATTLES", "SMOKE_STEPS", "_ABORT_EVAL_DRAIN_SEC", "_BOOL_FALSE", "_BOOL_TRUE",
    "_HparamLogCallback", "_PRELOAD_WITHOUT_OPPONENTS", "_TrackingCheckpointCallback",
    "_apply_grad_checkpointing", "_attach_run_tb_logger", "_load_saved_version",
    "_maybe_compile_trainer", "_model_hparams", "_read_saved_optimizer_state",
    "_remap_optimizer_state_by_name", "_resolve_fresh_model_dir", "_run_arch_toggles",
    "_run_roundtrip_test", "_setup_signal_handlers", "_shape_only_reset_optimizer_state",
    "_validate_or_reset_optimizer_state", "_write_latest_txt", "build_parser", "main",
    "optional_float", "resolve_compile_opponents_preload", "resolve_compile_trainer_auto",
    "resolve_compile_trainer_default", "str2bool",
)


@pytest.mark.parametrize("name", _PRE_SPLIT_EXPORTS)
def test_the_hub_still_exports_every_pre_split_name(name):
    import main.train_rl_agent as hub

    assert hasattr(hub, name), (
        f"`main.train_rl_agent.{name}` no longer resolves. The hub's whole job is that the "
        f"decomposition changed no import path — re-export it from whichever `main/train/` module "
        f"now owns it, or delete this entry deliberately if the name is genuinely gone."
    )


def test_no_phase_module_imports_the_hub():
    """The cycle guard. `main/train/*` may import each other; none may import `train_rl_agent`."""
    offenders = []
    for path in entry_source_files():
        if path.name == "train_rl_agent.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            if mod and ("main.train_rl_agent" in mod or "train_rl_agent" in mod.split(".")):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"phase module(s) import the hub back: {offenders}. That closes an import cycle — the hub "
        f"imports every phase — and its symptom is an AttributeError on a name that plainly "
        f"exists. Move the shared value into `main/train/constants.py` (or the module that owns "
        f"it) instead."
    )


def test_every_phase_module_imports_on_its_own():
    """Each phase must stand up without the hub having been imported first.

    A module that only works once `train_rl_agent` has run its imports is a cycle that has not
    bitten yet. Imported here in one process, which is enough: the failure mode is a module-level
    reference resolved at import time, and that fires on the first import regardless of order.
    """
    import importlib

    for path in entry_source_files():
        if path.name == "train_rl_agent.py":
            continue
        importlib.import_module(f"{_PKG}.{path.stem}")


def test_entry_source_is_the_whole_entry_point():
    """The seam the source-scanning gates read. A silently-shrinking one re-opens every gate."""
    names = {p.name for p in entry_source_files()}
    assert "train_rl_agent.py" in names, "entry_source_files() must include the hub"
    assert len(names) >= 10, f"only {len(names)} entry-point files found: {sorted(names)}"
    src = entry_source()
    # One marker per gate that reads this: the parser, the `_resolve` inheritance helper, the
    # edge-family validator, the two learn() sites, and the worker thread pin. If a phase module
    # stops being reachable from here, at least one of these disappears.
    for marker in ('parser.add_argument("--model"', 'def _resolve(name, default)',
                   "--edge-bias-families: unknown families", "reset_num_timesteps=False",
                   "set_num_threads(1)"):
        assert marker in src, (
            f"entry_source() no longer contains {marker!r} — a phase module has dropped out of "
            f"the seam, and the gate that reads it is now vacuous rather than red.")


def test_every_training_hparam_row_names_a_real_parser_dest():
    """`_TRAINING_HPARAMS` rows must correspond to actual `--flag` dests.

    The table replaced ~120 lines that existed VERBATIM on both build paths (resume and fresh).
    That structural change makes the old failure — a coefficient added to one branch and not the
    other — unrepresentable, because there is now one list instead of two. But it introduces one
    NEW failure the old form could not have: a misspelled row.

    A typo'd `_PLAIN` row fails loudly on its own (`getattr(args, name)` raises), so the real
    exposure is the two `_F0_OPT` rows, which are `getattr(..., 0.0)` by design and would silently
    pin the coefficient at zero forever — a run that trains with the term OFF while the command
    line says it is on, reporting nothing. This checks every row against the parser, so the
    tolerant rows get the same spelling guarantee as the strict ones.
    """
    from main.train.model_build import _TRAINING_HPARAMS
    from main.train_rl_agent import build_parser

    dests = {a.dest for a in build_parser()._actions}
    unknown = [n for n, _how in _TRAINING_HPARAMS if n not in dests]
    assert not unknown, (
        f"_TRAINING_HPARAMS rows name no parser dest: {unknown}. A `_F0_OPT` row spelled wrong "
        f"pins its coefficient at 0.0 silently — the term reads OFF while the command says on.")
    names = [n for n, _how in _TRAINING_HPARAMS]
    assert len(names) == len(set(names)), "a duplicated row in _TRAINING_HPARAMS"
