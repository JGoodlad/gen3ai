"""EVAL ON THE RUST ENV CORE (M5 Lane H) — ``--env-core rust``'s eval cycle.

Module map (design of record: ``designs/training/eval_and_rating.md`` → "Eval on the Rust core";
progress: ``designs/research_state/measurements/m5_laneH/PROGRESS.md``):

* ``seeds``    — the per-GAME seed rule both eval paths share (``gen3_eval_game_seed_v1``).
* ``traces``   — the core trace (``gen3_core_trace_v1``): records + reconstruction + states + meta.
"""
