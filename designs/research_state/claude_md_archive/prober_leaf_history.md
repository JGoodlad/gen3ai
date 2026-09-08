# `src/main/prober/CLAUDE.md` — history lifted out of the leaf

HISTORY, additive only. Created 2026-09-08 when the prober leaf was cut from 1,624 lines / 137 KB.
Everything here is dated narrative whose RULE, where it still has one, is stated in
`src/main/prober/CLAUDE.md` or in a `designs/prober/` topic doc. **Nothing here is current** — the
Textual TUI it describes no longer exists, and its key bindings, panels and CSS ids are gone with
it. Do not re-derive a plan from anything on this page without checking the code first.

## The Textual TUI's retirement, and what was dropped with it

Lifted from the leaf's header block. The live fact — `python -m main.prober` starts the web app,
`web/` is the only human-facing surface — is in the leaf.

**⚠ THE TEXTUAL TUI IS GONE** (`app.py`, `prober.tcss`, `review.py` — deleted, ~4,400 lines). It was
a *third* renderer over the same engine, which meant every new signal had to be drawn twice for a
single reader — the v67 α/β read was, days before this. `python -m main.prober` now starts the web
app, and its two TUI-only flags (`--ckpt`, `--inv`) print what replaced them instead of failing.
Dropped with it, each on evidence rather than taste:

| dropped | why |
|---|---|
| **Flow** (box-art dataflow) | `model.g5d.io` already draws the architecture, interactively, with the measured-dependence overlay |
| **Team** / **Board** panels | subsets of `/battle`'s board; the field line (weather/hazards/screens) survives on `/analyze` |
| **Review mode** (flags + notes) | 5 of 84 runs, unused since June, and the only thing that would have made the web read-write. Notes exported to `<run>/review_notes.md` first |
| **refine rounds** (axis A) | needs `--damage-refine-rounds`, which the production config does not run |

Everything else was ported: the per-decision **`analyze`** view and the counterfactual tier
(`lookahead` / `better_line` / `replay_counterfactual`, as password-gated background jobs off
`/analyze`). See `web/CLAUDE.md`.

## The TUI's model-tier keys

Lifted from § *Per-battle model resolution*. `--ckpt`, the tier badge and the per-path model cache
survive and are stated in the leaf; `m` and `R` were TUI keys.

the prober deliberately loads without a version check.

`--ckpt` forces an override. The badge shows the active tier + the trace's
`git_hash`/`arch_signature` from the manifest, so any faithfulness drift is
explained. `m` cycles the preference (`auto` → `nearest` → `recent`) to view a
trace under a different model; `R` reloads. Loaded models are cached by path
(`_model_cache`), so revisiting a step is instant. Even for runs that predate the
manifest (no snapshots), the ladder picks the **nearest checkpoint** — strictly
better than always using `best_model`.


## Manual review mode, and why it was retired

Lifted from § *What one decision's analysis CONTAINS*.


**Manual review mode is RETIRED** (`review.py`, `<run>/review_notes.json`). It let you flag a
decision *funky* and append timestamped notes while stepping through a battle — and it was the only
thing that would have forced the web front end to become read-write, with an auth story for
anonymous writes on a box that trains. The usage said it was not worth that: **5 of 84 runs, 12
annotated decisions, none flagged, nothing since mid-June**. Every note was exported to
`<run>/review_notes.md` before the code was deleted, so the content outlives the feature.

The EXPECTED → DID → HAPPENED story it framed was never review-specific — it is what `/analyze`'s
decision + outcome + critic panels show for any decision, now with α/β as the "expected" half.

## Two TUI-only gotchas

Lifted from § *Gotchas*: the `y` yank onto the replay-path bar, and the Textual scroll-gutter fix.
Both describe widgets that no longer exist.

- **`y`** yanks a precise pointer to the CURRENT decision — **`<replay.html path> inv<N>`** (N =
  the highlighted invocation) — onto a dedicated full-width **`#replay-path-bar`**, the ref on its OWN
  line so it's cleanly selectable under **`v`** copy mode (the portable path; the old toast wrapped it
  with its label). It also best-effort `copy_to_clipboard`s the ref (OSC-52 terminals only —
  kitty/iTerm2/WezTerm; dead on Terminal.app, hence the bar) **and RECORDS the ref in this decision's
  review notes** (deduped — a re-yank doesn't pile up), so the exact issue is one paste away to hand a
  model (`query analyze <battle> <N>`). The bar is hidden until used and cleared when a new battle is
  selected; warns if the file is missing.
- **Scroll stability:** the three scroll regions (`#trace-tree`, `#invocation-list`, `#analysis-scroll`)
  set **`scrollbar-gutter: stable`** so the 1-cell scrollbar column is always reserved — content no
  longer shifts a cell ("off by a pixel") the moment a scrollbar appears on scroll.


## The one line REWRITTEN rather than moved

The `web/` bullet's closing cross-reference pointed at a section of the leaf that is now a topic
doc, so the pointer was re-aimed at `designs/prober/beliefs_and_threats.md`. Preserved here verbatim
so the line-level audit of the split reports zero unaccounted lines:

```
  DamageOperator. See `web/CLAUDE.md`, and *Beliefs / Threats (GPU-first observability)* below for
```
