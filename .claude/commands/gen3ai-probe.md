---
description: Investigate a trained Gen3 RL model's behaviour/outcomes using the prober (the JSON CLI `python -m main.prober.query` + `ProbeSession`, or the Textual TUI `python -m main.prober`). Use whenever the user wants to understand WHY the model did something — why it lost a battle, chose a move, mis-valued a state, disagreed with itself across checkpoints, or where a battle went wrong. Triggers on "investigate the model", "why did the model lose/choose X", "probe the model", "debug the policy/critic", "analyze this battle/trace", "forensic", "model behaviour", or a path to an eval_traces summary.json. This skill ALSO carries a continuous self-improvement loop: extend the CLI when you lack data, and port proven CLI diagnostics to the TUI for human parity.
---

# /gen3ai-probe

Investigate why the model behaved as it did, using the **prober** — the forensic
layer over saved `eval_traces`. There are two front-ends over **one engine**
(`src/main/prober/engine/`):

- **CLI / API** (your lens, agents): `python -m main.prober.query …` and
  `ProbeSession` (`src/main/prober/session/`) — emit JSON.
- **TUI** (the human's lens): `python -m main.prober <run_dir>` — the same
  analysis, navigable.

Internals, the model-resolution ladder, and the panel/flag vocabulary are in
`src/main/prober/CLAUDE.md` — skim it before extending anything.

## Part 1 — Investigate (the recipe)

Drive the JSON CLI; parse stdout. `<run_dir>` is a `models/run_*/` (gitignored —
lives in the **main checkout** `/home/goodlad/dev/gen3ai/models/...`, not the
worktree). A `<battle_id>` is the `id` (a `*_summary.json` path) or `short_id`
from list/summary output.

1. **Orient** — `query summary <run_dir>`: steps, opponents, win/loss, per-step
   model identity (git/arch/snapshot), checkpoints, γ.
2. **Pick battles** — `query list <run_dir> --outcome loss [--opponent X] [--step N]`.
3. **Digest (model-free, cheap)** — `query overview <battle_id>`: per-decision
   rows (chosen, top_prob, V(s), ΔV, TD residual, reward, events, flags) and a
   `notable` block (faints, switches, `biggest_value_drops`). Read `notable`
   first — it points at where the battle turned.
4. **Zoom** — `query find <battle_id> <criterion> [--limit N]`:
   `value_drop`/`low_value`/`high_value` (model-free, ranked), flags
   (`switch`/`uncertain`/`faint`), or `disagree` (loads the model — decisions the
   resolved model wouldn't repeat).
5. **Deep-dive** — `query analyze <battle_id> <inv> [--tier auto|nearest|recent]`:
   faithfulness, matchups, intervention sweep, saliency, value+TD, model
   disagreement. Mind the **tier**: `exact` = bit-faithful; `nearest`/`recent` =
   a different model, so drift is expected (the JSON `model_resolution` says
   which). Want bit-exact replay of old traces? They need `--keep-eval-snapshots`
   at train time.

Report findings to the user grounded in the JSON (quote inv indices, values,
saliency blocks, reward components).

## Part 2 — Continuous self-improvement (DO THIS — it's the point)

The prober must get sharper every time it's used. Two reflexes, run at the end of
every investigation:

### Reflex A — data gap → extend the CLI/engine
If you found yourself **wishing for a number, view, or query the CLI/API doesn't
expose** (or you worked around its absence by hand-computing from raw npz/JSON),
that is a signal to **add it**, not to move on:
- New per-decision metric or analysis → add it in **`engine.py`**
  (`InvocationAnalysis` / a new analysis unit). The engine is the **single source
  of truth**, so both front-ends inherit it.
- New query / ranking / filter → add a `find` criterion or a `ProbeSession`
  method in **`session.py`**, surfaced as a subcommand/flag in **`query.py`**.
- Always: add/extend a `*_test.py` and update `src/main/prober/CLAUDE.md`.

### Reflex B — CLI insight → TUI parity (so a human can follow along)
If a CLI field/query was **decisive in cracking the case**, the human's lens must
show it too. Port it to the **TUI** (`app.py`) — a panel row, a flag glyph, a
keybinding, a column — reading the **same engine output** (no recomputation).
Goal: **parity** — anything the agent can see in JSON, a human can see by
navigating the TUI, and vice-versa. When you add an engine field for the CLI,
add its TUI surfacing in the same change unless there's a reason not to.

This is a virtuous cycle: investigation reveals a gap → the engine grows → the
CLI exposes it for agents → the TUI mirrors it for humans → the next
investigation starts from a richer baseline.

## Guardrails (when you extend)

- **Worktree only.** Edit under the worktree path; never the bare repo. `models/`
  is in the main checkout — reference it by absolute path. (See
  [[feedback_edit_in_worktree_path]].)
- **Test before claiming done:**
  `export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/main/prober src/main/tui -q`
  TUI changes: also drive a real battle via `App.run_test()` (see existing CUJ
  patterns) — assertions, not just imports.
- **Keep docs current** (`src/main/prober/CLAUDE.md`, root `CLAUDE.md` if a flag/
  entry point changes) — same change, per the repo's auto-doc rule.
- **Never commit.** Surface what you added; landing on `main` is `/gen3ai-ship`
  only, on the user's explicit say-so.
- **Keep the engine pure** (no Textual/printing) so the CLI/TUI parity holds;
  γ-dependent or presentation logic lives in `session.py`/`app.py`, not the engine.
