---
description: Investigate a trained Gen3 RL model's behaviour/outcomes using the prober (the JSON CLI `python -m main.prober.query` + `ProbeSession`, or the web app `python -m main.prober`). Use whenever the user wants to understand WHY the model did something — why it lost a battle, chose a move, mis-valued a state, disagreed with itself across checkpoints, or where a battle went wrong. Triggers on "investigate the model", "why did the model lose/choose X", "probe the model", "debug the policy/critic", "analyze this battle/trace", "forensic", "model behaviour", or a path to an eval_traces summary.json.
---

# /gen3ai-probe

Investigate why the model behaved as it did, with the prober: the forensic layer over a run's saved `eval_traces`. One engine, two surfaces: the JSON CLI `python -m main.prober.query` (yours) and the browser app `python -m main.prober` (the human's). Read `src/main/prober/CLAUDE.md` before extending anything; the recipe's detail is `designs/prober/session_scans.md` "The investigation recipe".

## Investigate

`<run_dir>` is `models/<run>`, which exists only in the MAIN checkout (`/home/goodlad/dev/gen3ai/models/...`); a `<battle_id>` is a `*_summary.json` path or the `short_id` from `list` / `summary`. `python -m main.prober.query --help` lists every command.

1. `triage <run_dir>`: loss categories ranked by recoverable win rate, the lever order.
2. `summary <run_dir>`, then `list <run_dir> --outcome loss [--opponent X] [--step N]` or `scan`: orient and pick battles.
3. `overview <battle_id>`: model-free digest; read `notable` (faints, switches, `biggest_value_drops`) first, since it shows where the battle turned.
4. `find <battle_id> value_drop|low_value|high_value|switch|uncertain|faint|disagree`, then `analyze <battle_id> <inv>`.
5. `falsify` the worst turn (dice or mistake), `falsify-scan` for the run-level split, `calibration` for the rest.

Read `model_resolution` in the JSON before trusting a model number: tier `exact` is bit-faithful, `nearest` / `recent` is a different model (drift expected); exact replay needs `--keep-eval-snapshots` at train time. Read each cycle's `eval_manifest.json` first (the trace quota prefers losses). Report findings grounded in the JSON: inv indices, values, saliency blocks, reward components.

## When the prober lacks what you need

If you hand-computed from raw npz/JSON or wished for a query, add it. The analysis goes in `engine/` (the single source, so both surfaces inherit it), a `find` criterion or method in `session/`, a subcommand in `query.py`, and its human view in `web/`, reading the same engine output. Add a `*_test.py`, and update `src/main/prober/CLAUDE.md` or `designs/prober/` in the same change.

## Guardrails

- Edit in a worktree, never the main checkout; commit only via `/gen3ai-ship` on its terms.
- Test: `pytest src/main/prober -q` with `PYTHONPATH=$PWD/src`; keep the engine pure (no web or printing).
