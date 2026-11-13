"""THE EFFECTIVE COMPILE SURFACE — the values a run COMPILES AT, for the shared compile cache's key.

`agents.model.compile_cache_shared.run_fields` keys a run on its compile-shaping trainer flags. The key is
declared before the model exists (`lifecycle._declare_compile_cache`), so it used to read the ARGV — and on a
`--model` launch (a restart, a fork, a fork's own resume) some argv values are INERT: SB3 restores the
checkpoint's own (`recipe_surface.INERT_ON_RESUME`). The argv's untyped `--batch-size` is the parser default
there while the learner compiles at the checkpoint's micro-batch, so ONE graph was filed under TWO keys and a
run's first restart compiled cold (B1 quit probe, 2026-11-12: T2 up in 1041 s against 155 s).

This module resolves those values to what the learner will really hold:

* a `--model` launch: the checkpoint zip's own SB3 `data` member — the very values `load` restores — read with
  `zipfile` + `json`, no torch and no model load. An unreadable checkpoint RAISES: the caller then shares
  nothing (a per-run cache), because a guessed value could file two different graphs under one key;
* a fresh launch: the argv's, after the same rollout-capacity cap `model_build` applies before it constructs
  the learner (`fresh_batch_size`), so a capped fresh run and its resumes agree too.

The closed list is `compile_cache_shared.EFFECTIVE_COMPILE_ARG_KEYS`; `compile_surface_test` pins it to
`COMPILE_ARG_KEYS & INERT_ON_RESUME`, so a new inert-on-resume flag that enters the key forces a decision here.
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict

from agents.model.compile_cache_shared import EFFECTIVE_COMPILE_ARG_KEYS


def resolve_model_arg(model: str) -> str:
    """`--model` as the loader takes it: the path itself, or — for a bare NAME that is not a path and not a
    `.zip` — the golden of that name under the cwd's `models/goldens/` (the trainer's long-standing lookup)."""
    if os.path.exists(model) or model.endswith(".zip"):
        return model
    for p in (os.path.join("models", "goldens", model),
              os.path.join("models", "goldens", model, "final_model"),
              os.path.join("models", "goldens", model, "final_model.zip")):
        if os.path.exists(p) or os.path.exists(p + ".zip"):
            return p
    return model


def restored_compile_args(model: str) -> Dict[str, int]:
    """`EFFECTIVE_COMPILE_ARG_KEYS` (argparse dests == the SB3 `data` member's keys) exactly as the checkpoint `--model` names will restore them. Raises
    (`OSError` / `KeyError` / `ValueError` / `zipfile.BadZipFile`) when the zip or a value cannot be read."""
    path = resolve_model_arg(model)
    # the loader's candidates (`agents.model.snapshot._resolve_paths`), FILES only: a zip is what gets restored
    zip_path = next((c for c in (path, path + ".zip", os.path.join(path, "final_model.zip"),
                                 os.path.join(path, "best_model.zip")) if os.path.isfile(c)), None)
    if zip_path is None:
        raise FileNotFoundError(f"no checkpoint zip at {path!r}")
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data"))
    out: Dict[str, int] = {}
    for k in EFFECTIVE_COMPILE_ARG_KEYS:
        v = data.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{zip_path}: the checkpoint's `data` holds no usable {k!r} (got {v!r})")
        out[k] = v
    return out


def fresh_batch_size(*, batch_size: int, n_steps: int, n_envs: int) -> int:
    """The micro-batch a FRESH learner is built with: the argv's, capped at the rollout capacity
    (`n_steps * n_envs`). ONE rule, shared by `model_build`'s fresh path and the cache key."""
    return min(int(batch_size), int(n_steps) * int(n_envs))


def effective_compile_args(args: Any) -> Dict[str, int]:
    """`EFFECTIVE_COMPILE_ARG_KEYS` at the value this launch's learner will hold (module docstring)."""
    model = getattr(args, "model", None)
    if model:
        return restored_compile_args(model)
    n_envs = 1 if getattr(args, "debug", False) else args.n_envs      # `train_rl_agent.main`'s own rule
    return {"batch_size": fresh_batch_size(batch_size=args.batch_size, n_steps=args.n_steps, n_envs=n_envs)}
