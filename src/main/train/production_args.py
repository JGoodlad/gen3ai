"""The resolved training namespace of a real FRESH ``--arch production`` launch — what the tests, the
K9 learner golden and the offline tools build "the production surface" from.

Re-homed from ``main/rust_core_cutover/envs.py`` (deletion pass U3 / R10, which deleted the Python-vs-Rust
cutover harness): ``production_args`` was the one piece of it that LIVE code kept calling. The parser, then
``main.train.config.resolve_config`` — the function the trainer itself runs, so the ARCH surface, the RECIPE
surface (``recipe.fresh``, K10(a)), the critic implications and every ``_resolve`` default arrive by the
launch's own code, not a second copy of it. ``production_args_test`` holds the two together on every
mirror key, the recipe included.

It used to rebuild the surface by hand (``apply_production_arch`` + a ``hasattr`` copy of every top-level
mirror key), which silently skipped the nested ``recipe`` block — so the harness's "production" trained 32 envs
x 4096 x 5 epochs at ent 0.02, no self-play, no grad accumulation, against a launch's 48 x 2048 x 10 at
0.05 with K=32.
"""
from __future__ import annotations

import functools

#: The argv `production_args` resolves: a FRESH `--arch production` launch and nothing else typed.
PRODUCTION_ARGV = ("--steps", "1", "--arch", "production")


def production_args():
    """The resolved training namespace of a real FRESH ``--arch production`` launch.

    Resolved once per process (``resolve_config`` imports and prints a launch's worth); every call
    returns a fresh deep copy, so a caller that mutates its namespace cannot reach the next one."""
    import copy

    a = copy.deepcopy(_resolved_production_args())
    a.use_bridge = "rust"
    return a


@functools.lru_cache(maxsize=1)
def _resolved_production_args():
    import contextlib
    import io

    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    a = parser.parse_args(list(PRODUCTION_ARGV))
    with contextlib.redirect_stdout(io.StringIO()):    # the launch banner; a refusal still exits
        resolve_config(a, parser)
    return a
