"""THE canonical `path[@step]` splitter — one place that knows a run spec is not a filesystem path.

WHY THIS MODULE EXISTS (gen3_run_spec_split_v1, 2026-09-05). Several flags take a *run spec*
(`models/<run>@<step>`) rather than a bare path: `--stable-opponents`, `--exploiter`, and — as of
this module — `--distill-teacher`, `--win-prob-pbrs-source`, `--distill-anchor-parent` and
`--warmstart-consensus`. Only ONE of them ever split the `@step` off, and every other consumer
handed the whole string to a reader that takes a *directory*::

    read_recorded_trainee_teams('models/ai_v9_92_R5F00_0831')            -> 2 teams
    read_recorded_trainee_teams('models/ai_v9_92_R5F00_0831@26267760')   -> 0 teams   # WRONG

The second answer is a WRONG ANSWER ON A SUCCESS PATH — the same class as the era's `--steps`
no-op: the reader could not find a `metadata.json` beside a directory that does not exist, and
"no metadata" was indistinguishable from "a generalist run recorded no trainee teams". A fold
written the obvious way therefore reported its teachers as teaching NOTHING, and the only symptom
was a team count in the `[DISTILL]` startup line.

The class fix is this function plus two throwing guards:

  * every consumer of a run spec routes the string through :func:`split_run_spec` BEFORE it reaches
    a filesystem reader — `fixed_opponent_pool._resolve_zip_and_config` does it at ITS entry, so
    every caller that passes `step=None` (the four above) is fixed at once;
  * `matchup_spec.read_recorded_trainee_teams` RAISES on a path that does not exist instead of
    reading as "a generalist run", so a spec that escapes the splitter cannot be silent again.

`src/agents/training/run_spec_test.py` holds the census that fails when a new consumer re-derives
the split locally.
"""
from __future__ import annotations


def split_run_spec(spec: str, *, what: str = "run spec") -> "tuple[str, int | None]":
    """``'models/run@123'`` → ``('models/run', 123)``; ``'models/run'`` → ``('models/run', None)``.

    The grammar is ``path[@step]`` — the path half must not contain ``@`` (filesystem paths in this
    tree do not, and a run dir never has). Splits on the FIRST ``@``, matching the
    ``--stable-opponents`` parser this generalizes.

    FAIL-LOUD on a non-integer step (``models/run@best``): that is a spec the reader would
    otherwise resolve to a path that does not exist, which is exactly the silence this module was
    written to remove. A bare trailing ``@`` carries no step and is accepted as the path it names.

    Pure — no filesystem access. ``what`` names the flag in the error message.
    """
    text = str(spec)
    path, sep, step_s = text.partition("@")
    path = path.strip()
    if not path:
        raise ValueError(f"{what} {text!r} has no path")
    step_s = step_s.strip()
    if not sep or not step_s:
        return path, None
    try:
        step = int(step_s)
    except ValueError:
        raise ValueError(
            f"{what} {text!r}: step {step_s!r} is not an integer — the grammar is "
            "'path[@step]', where step is a checkpoint's num_timesteps "
            "(<run>/checkpoints/checkpoint_<step>_steps.zip)") from None
    return path, step


def run_dir_of(spec: str, *, what: str = "run spec") -> str:
    """Just the path half of ``path[@step]`` — for a reader that takes a run DIRECTORY.

    The one-liner every provenance reader's caller needs; named so a call site reads as the intent
    ("give me the run dir this spec names") rather than as tuple bookkeeping.
    """
    return split_run_spec(spec, what=what)[0]
