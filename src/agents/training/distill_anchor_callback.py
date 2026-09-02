"""WHERE THE OFF-SLICE ANCHOR'S PARENT COMES FROM (`gen3_distill_offslice_anchor_v1`).

The anchor (`instrumented_ppo/distill_anchor.py`) regularises the student toward the policy it
STARTED the fold as. That makes "which checkpoint is the parent?" the one question this feature can
get silently, catastrophically wrong — so it is answered here, once, and pinned by tests.

🚨 **THE RESTART RULE.** A launcher run is restarted every few hours, and an idempotent FORK swaps
its `--model` to the fork's OWN latest checkpoint on every relaunch
(`main.launcher.checkpoint.resolve_fork_resume_model`) so progress is not discarded. Anchoring to
"the checkpoint we just loaded" would therefore re-anchor to a DRIFTED policy at every restart: the
trust region would follow the student around, ratchet by ratchet, and after a handful of restarts it
would constrain nothing while still reading as ON. So the parent is always **re-loaded from the
ORIGINAL FORK-PARENT PATH**, resolved in this order:

  1. ``--distill-anchor-parent PATH`` — an explicit pin, and the only route that needs no inference.
  2. ``<run_dir>/metadata.json`` → ``original_command`` → its ``--model`` value. `original_command`
     is IMMUTABLE by construction (`agents.model.snapshot.save_model_snapshot`: the existing value
     always wins), so it still names the fork parent after any number of restarts. This is the route
     that carries a live run.
  3. this process's own ``--model`` — correct on a fork's FIRST launch, where the run dir is fresh
     and holds no `metadata.json` yet (`resolve_launch_run_dir` refuses to fork onto an existing
     run, so route 2 cannot fire early with the wrong answer).

The resolved path and the route it came from are PRINTED at startup, because a wrong parent is a
silent no-op rather than a crash, and the only defence against a silent no-op is a loud statement of
what was chosen.

The frozen parent is never persisted (`_excluded_save_params`): re-loading from a path is the whole
correctness argument, and a pickled copy would be a second, wrong answer that survives restarts.
"""
import json
import os
import shlex
from typing import Callable, Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback

#: The resolution routes, in precedence order — the value printed at startup and asserted in tests.
ANCHOR_PARENT_ROUTES = ("explicit", "original_command", "cli_model")

_MODEL_FLAGS = ("--model", "--model_path", "--model-path")


def parse_model_arg(command: str) -> Optional[str]:
    """The ``--model`` value inside a recorded shell command, or ``None``.

    Handles both spellings argparse accepts (``--model X`` and ``--model=X``) and the underscore
    alias, via `shlex` so a quoted path with spaces survives. Total: a malformed command yields
    ``None`` rather than raising, because a resolution FALL-THROUGH is recoverable and an exception
    inside callback assembly is not.
    """
    try:
        toks = shlex.split(command or "")
    except ValueError:
        return None
    for i, tok in enumerate(toks):
        for flag in _MODEL_FLAGS:
            if tok == flag and i + 1 < len(toks):
                return toks[i + 1]
            if tok.startswith(flag + "="):
                return tok[len(flag) + 1:]
    return None


def read_original_command(run_dir: str) -> Optional[str]:
    """``<run_dir>/metadata.json``'s immutable ``original_command``, or ``None`` if absent/unreadable."""
    path = os.path.join(run_dir or "", "metadata.json")
    try:
        with open(path, encoding="utf-8") as f:
            val = json.load(f).get("original_command")
    except (OSError, ValueError):
        return None
    return val if isinstance(val, str) and val.strip() else None


def resolve_anchor_parent(*, explicit: Optional[str], run_dir: Optional[str],
                          cli_model: Optional[str]) -> Tuple[Optional[str], str]:
    """``(parent_path, route)`` per the precedence documented at the top of this module.

    Pure but for one `metadata.json` read. Returns ``(None, "unresolved")`` when nothing names a
    parent — the caller turns that into a refusal, never into "anchor off": an anchor that silently
    did not attach is exactly the failure the loud startup line exists to prevent.
    """
    if explicit:
        return explicit, "explicit"
    cmd = read_original_command(run_dir) if run_dir else None
    if cmd:
        got = parse_model_arg(cmd)
        if got:
            return got, "original_command"
    if cli_model:
        return cli_model, "cli_model"
    return None, "unresolved"


class DistillAnchorCallback(BaseCallback):
    """Attach the FROZEN fold parent (and the anchor hparams) before `learn()` takes its first step.

    A callback rather than a line in `apply_training_hparams` for one reason that is also the
    feature's correctness argument: `_on_training_start` runs on EVERY launch, including every
    launcher restart, which is exactly the cadence at which the parent must be re-read from its own
    path. `load_parent(path)` is injected so this module needs no `mappings` and no `main.train`
    import, and so the resolution logic can be tested without a checkpoint on disk.

    A failed load is FATAL (`TrainExitCode.FATAL_CONFIG`), matching the `--distill-teacher` load: a
    run that silently trains with no trust region while its command says otherwise produces a
    result that is worse than no result, because it looks like a measurement.
    """

    def __init__(self, *, parent_path: str, route: str, coef: float, mode: str, monitor: bool,
                 load_parent: Callable[[str], object], verbose: int = 0):
        super().__init__(verbose)
        self.parent_path = parent_path
        self.route = route
        self.coef = float(coef or 0.0)
        self.mode = str(mode)
        self.monitor = bool(monitor)
        self._load_parent = load_parent

    def _on_training_start(self) -> None:
        from main.launcher.ipc import emit
        # The hparams first: they are read by `train()` off the model, and a load failure below
        # exits the process, so there is no state in which they are set and the parent is not.
        self.model.distill_anchor_coef = self.coef
        self.model.distill_anchor_mode = self.mode
        self.model.distill_anchor_monitor = self.monitor
        try:
            parent = self._load_parent(self.parent_path)
        except Exception as e:  # noqa: BLE001 — a bad/incompatible parent must not crash-restart-loop
            from main.exit_codes import TrainExitCode
            import sys
            print(f"\n[DistillAnchor] FATAL: could not load the fold parent {self.parent_path!r} "
                  f"(resolved via {self.route}): {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        self.model._distill_anchor_parent = parent
        emit(f"⚓ [DISTILL-ANCHOR] parent = {self.parent_path} (via {self.route}) | "
             f"coef={self.coef:g} mode={self.mode}"
             f"{' | MONITOR-ONLY (no loss term)' if self.coef == 0.0 else ''} — "
             f"re-loaded from the PARENT PATH on every restart, never from the current checkpoint. "
             f"Watch distill/collateral_kl (damage) against distill/teacher_agreement_on_slice "
             f"(content).")

    def _on_step(self) -> bool:
        return True
