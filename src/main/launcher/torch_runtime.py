"""INTERPRETER ↔ RUN SAFETY — a resume or fork never silently switches torch under a run.

Owner, 2026-09-30: torch 2.8 (`gen3ai_torch28`) is the DEFAULT interpreter; torch 2.5.1
(`gen3ai_stable`) is LEGACY, kept so an old pinned run resumes on its own torch. Two torches
compile the same code to different kernels (the 2.5.1 trunk split, `compile_control`'s per-version
rows, the learner golden's per-version entries), so the torch a run trains under is part of the
run, and switching it at a restart would change the experiment with nothing saying so.

THE RECORD. Every save writes `metadata.json`'s `torch_version` (and a `torch` key on each
`pin_history` span — `agents.model.snapshot`). A run whose metadata has NO record predates it
(2026-09-30) and every such run trained on `gen3ai_stable`, so an absent record READS AS 2.5.1.
Versions are compared by RELEASE (`2.5.1+cu121` ≡ `2.5.1`): the CUDA tag is a build of the same
torch, not a different one.

THE RULE, decided ONCE per launcher session (`resolve_for_launch`) and held for every restart:
* FRESH launch: the child runs `resolve_child_python()` (`$GEN3AI_PYTHON`, else the launcher's own
  interpreter) — whatever torch that has becomes the run's record at its first save.
* RESUME / FORK (`--model`): the torch the checkpoint's RUN recorded is REQUIRED.
  - `$GEN3AI_PYTHON` set: it must carry that torch.
  - otherwise the launcher's own interpreter if it carries it; else the sibling conda env that
    `KNOWN_ENVS` names for that torch (looked up beside the launcher's env — names, never a
    machine path), selected automatically and announced.
  - nothing carries it: a typed `FATAL_CONFIG` refusal, unless `--allow-torch-switch` (the
    explicit consent — the new torch is then recorded at the child's first save).
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import Dict, Optional

from main.launcher.checkpoint import run_dir_for_checkpoint
from main.launcher.child import PYTHON_ENV_VAR, resolve_child_python

#: The launcher flag that consents to running a resumed/forked run under a torch other than the
#: one it recorded. Launcher-owned: stripped before the child's argv.
ALLOW_FLAG = "--allow-torch-switch"

#: What an UNRECORDED run trained on: every run before the record (2026-09-30) ran `gen3ai_stable`.
LEGACY_TORCH = "2.5.1"

#: torch RELEASE → the conda env NAME that carries it on this project's boxes. Names only — the
#: env is looked up beside the launcher's own env (`_sibling_python`), never at a machine path.
KNOWN_ENVS: Dict[str, str] = {
    "2.5.1": "gen3ai_stable",     # LEGACY (owner 2026-09-30): old pinned runs resume here
    "2.8.0": "gen3ai_torch28",    # the DEFAULT for new runs, gates and tooling
}

_PROBE = "import importlib.metadata as m; print(m.version('torch'))"
_probe_cache: Dict[str, Optional[str]] = {}


def release(version: Optional[str]) -> Optional[str]:
    """`2.8.0+cu126` → `2.8.0` (None stays None)."""
    return None if version is None else str(version).split("+", 1)[0].strip()


def interpreter_torch(python: str) -> Optional[str]:
    """The torch version installed under `python` (its package METADATA — torch is not imported,
    so this costs ~50 ms, not seconds). None when the interpreter cannot be run or has no torch."""
    if python in _probe_cache:
        return _probe_cache[python]
    ver: Optional[str] = None
    try:
        out = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            ver = out.stdout.strip().splitlines()[-1]
    except (OSError, subprocess.SubprocessError):
        ver = None
    _probe_cache[python] = ver
    return ver


def recorded_torch(model_path: str) -> "tuple[str, str]":
    """(torch version the checkpoint's RUN recorded, where that came from).

    Reads `<run>/metadata.json`'s `torch_version`; a run without one is the legacy 2.5.1."""
    run_dir = model_path if os.path.isdir(model_path) else run_dir_for_checkpoint(model_path)
    meta_path = os.path.join(run_dir, "metadata.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict) and meta.get("torch_version"):
        return str(meta["torch_version"]), f"recorded in {meta_path}"
    why = "no torch_version in" if isinstance(meta, dict) else "no readable"
    return LEGACY_TORCH, f"{why} {meta_path} — an unrecorded run is the legacy torch {LEGACY_TORCH}"


def _sibling_python(env_name: str) -> Optional[str]:
    """`<conda envs dir>/<env_name>/bin/python3`, the envs dir found from the launcher's OWN
    prefix (`…/envs/<x>` → `…/envs`; a base install → `<base>/envs`). None if it does not exist."""
    prefix = os.path.abspath(sys.prefix)
    parent = os.path.dirname(prefix)
    envs_dir = parent if os.path.basename(parent) == "envs" else os.path.join(prefix, "envs")
    cand = os.path.join(envs_dir, env_name, "bin", "python3")
    return cand if os.access(cand, os.X_OK) else None


@dataclasses.dataclass(frozen=True)
class TorchResolution:
    python: str                      # the interpreter every child of this session runs
    torch: Optional[str]             # its torch version (None: could not be read)
    required: Optional[str]          # the run's recorded torch (None on a FRESH launch)
    required_source: str             # where `required` came from
    how: str                         # one line: why this interpreter
    refusal: Optional[str] = None    # set ⇒ the launch must exit FATAL_CONFIG

    def lines(self) -> "list[str]":
        out = [f"interpreter : {self.python}  (torch {self.torch or 'UNREADABLE'}) — {self.how}"]
        if self.required is not None:
            out.append(f"run torch   : {self.required}  ({self.required_source})")
        if self.refusal:
            out.append(f"✗ REFUSED (torch): {self.refusal}")
        return out


def resolve_for_launch(model_path: Optional[str], *, allow_switch: bool = False) -> TorchResolution:
    """Decide the child interpreter for this launcher session (module docstring has the rule)."""
    override = os.environ.get(PYTHON_ENV_VAR, "").strip()
    default = resolve_child_python()
    default_torch = interpreter_torch(default)
    pinned = f" (pinned by ${PYTHON_ENV_VAR})" if override else ""
    if not model_path:
        return TorchResolution(default, default_torch, None, "", f"FRESH launch{pinned}; this "
                               "torch is recorded at the run's first save")
    required, source = recorded_torch(model_path)
    want = release(required)
    if release(default_torch) == want:
        return TorchResolution(default, default_torch, required, source,
                               f"carries the run's recorded torch{pinned}")
    if not override:
        env_name = KNOWN_ENVS.get(want or "")
        sib = _sibling_python(env_name) if env_name else None
        if sib is not None and release(interpreter_torch(sib)) == want:
            return TorchResolution(sib, interpreter_torch(sib), required, source,
                                   f"SELECTED: the `{env_name}` env carries the run's recorded "
                                   f"torch (the launcher's own has {default_torch})")
    cause = (f"the run trained on torch {required} but the child interpreter {default}{pinned} "
             f"has torch {default_torch}")
    if allow_switch:
        return TorchResolution(default, default_torch, required, source,
                               f"TORCH SWITCH CONSENTED ({ALLOW_FLAG}): {cause}; the new torch is "
                               "recorded at the child's first save")
    hint = (f"set ${PYTHON_ENV_VAR} to an interpreter with torch {want}"
            + (f" (the `{KNOWN_ENVS[want]}` env)" if want in KNOWN_ENVS else "")
            + f", or pass {ALLOW_FLAG} to switch this run's torch deliberately")
    return TorchResolution(default, default_torch, required, source, "MISMATCH",
                           refusal=f"{cause}. A resume/fork never switches torch silently — {hint}.")
