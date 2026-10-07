"""`--debug` is SAFE BY CONSTRUCTION: a fresh smoke never declares an update its ONE env cannot fill.

**WHY.** `--debug` runs ONE env on CPU (`train_rl_agent`: ``n_envs = 1 if args.debug``), but the
update size is the collector's ``--rollout-target-samples`` — and `--arch production` writes the
production recipe's 98,304 rows (``recipe.sizing``) into it. One CPU env needs HOURS to complete
98,304 rows of finished games, so `--arch production --debug --steps 10000` never reached its first
update: three agents stalled on it (2026-10-07) with nothing printed that said why.

**WHAT.** On a FRESH `--debug` run whose UNTYPED update size exceeds `DEBUG_MAX_UPDATE_ROWS`, every
UNTYPED knob of the rollout shape below is overridden to the smoke shape, printed loudly:

  * ``--rollout-target-samples`` → `DEBUG_TARGET_ROWS`, rounded UP to a multiple of
    lcm(``--batch-size``, ``--n-envs``) — the parse-time quantum (`combination_checks`'
    ``rollout_target_on_the_quantum``); the run-time quantum (one env) is ``--batch-size`` alone,
    which divides it;
  * ``--batch-size`` → `DEBUG_BATCH_SIZE`;
  * ``--n-epochs`` → `DEBUG_N_EPOCHS`.

A TYPED knob always wins (`recipe_surface.typed_dests` is the parser's record); a typed target that
is off the quantum is still refused by the combination check, as on any run. A RESUME is untouched:
`--batch-size` is INERT there (SB3 restores the checkpoint's), and a resume's untyped target is
``n_steps x 1`` from the checkpoint. A default `--debug` (no `--arch`) updates every
``--n-steps`` (2,048) rows and is under the ceiling, so it is unchanged.

The recipe surface reports an overridden knob as source ``debug`` (never the silent ``default``
kind), so `main.checkargs` on a `--debug` argv does not refuse it as a recipe drift.
"""
from __future__ import annotations

from math import gcd
from typing import Any, List, Tuple

#: An untyped update larger than this on a fresh `--debug` run is overridden (the default smoke's
#: is 2,048 = ``--n-steps`` x one env; production's is 98,304).
DEBUG_MAX_UPDATE_ROWS = 4096
#: The smoke update: 6 micro-batches of `DEBUG_BATCH_SIZE`, ~4 updates in a 10,000-step smoke.
DEBUG_TARGET_ROWS = 2304
DEBUG_BATCH_SIZE = 384
DEBUG_N_EPOCHS = 1

#: The namespace attribute naming the dests this module overrode (read by `recipe_surface`).
OVERRIDE_ATTR = "_debug_shape_override"


def _untyped_update_rows(args: Any) -> int:
    t = int(getattr(args, "rollout_target_samples", 0) or 0)
    return t or int(args.n_steps)          # one env: the default target is n_steps x 1


def apply_debug_rollout_shape(args: Any) -> List[Tuple[str, Any]]:
    """Override the untyped rollout shape of a fresh `--debug` run (module docstring). Returns
    ``[(dest, value), …]`` and records the dests on ``args.<OVERRIDE_ATTR>``; ``[]`` when nothing
    applies."""
    from main.train.recipe_surface import typed_dests

    if not getattr(args, "debug", False) or getattr(args, "model", None):
        return []
    typed = typed_dests(args)
    before = _untyped_update_rows(args)
    if "rollout_target_samples" in typed or before <= DEBUG_MAX_UPDATE_ROWS:
        return []
    out: List[Tuple[str, Any]] = []
    if "batch_size" not in typed:
        args.batch_size = DEBUG_BATCH_SIZE
        out.append(("batch_size", DEBUG_BATCH_SIZE))
    if "n_epochs" not in typed:
        args.n_epochs = DEBUG_N_EPOCHS
        out.append(("n_epochs", DEBUG_N_EPOCHS))
    b, n = int(args.batch_size), int(args.n_envs)
    q = b * n // gcd(b, n)
    target = -(-DEBUG_TARGET_ROWS // q) * q
    args.rollout_target_samples = target
    out.append(("rollout_target_samples", target))
    setattr(args, OVERRIDE_ATTR, frozenset(d for d, _ in out))
    print("🧪 [DEBUG SHAPE] --debug runs ONE env, which cannot fill this run's update of "
          f"{before:,} rows — overriding the UNTYPED rollout shape so the smoke "
          "reaches its updates: " + ", ".join(f"--{d.replace('_', '-')} {v}" for d, v in out)
          + ". A typed flag still wins (main.train.debug_shape).", flush=True)
    return out

