# The trace RESULT vocabulary — `win` · `loss` · `draw`, as the prober reads it

Always-current topic doc of `src/main/prober/CLAUDE.md`. The declaration is `agents/training/trace_result.py`.

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### THE RESULT VOCABULARY — `win` · `loss` · `draw`, and what an old tree's zero means

One declaration, `agents/training/trace_result.py` (`gen3_trace_result_v2`, pure stdlib — the
prober imports it without pulling in the training stack, exactly like `trace_selection`). The
recorder writes it, the filename prefix carries it and every filter here is built from it.

| `meta.result` | `meta.draw_kind` | how the battle layer reports it |
|---|---|---|
| `WIN` | — | `won` |
| `LOSS` | — | `lost`, before the turn cap |
| `DRAW` | `timeout` | `lost` **and** `turn >= MAX_TURNS` (250) — the trainee FORFEITED at the deadline |
| `DRAW` | `tie` | `won`/`lost` both falsy, finished — the sim's `\|tie\|` |

**A TIMEOUT ARRIVES WEARING A LOSS'S FLAGS.** The trainee forfeits at the cap
(the stall forfeit: `agents.training.stall.StallConfig`; the Python `inference/player._handle_stall` that issued it was
deleted in T27 P6), so the stream reports `lost=True`. The training reward never
agreed — the terminal fold (`reward_config.terminal_breakdown`, the Rust env core's terminal) pays `draw_penalty`
for exactly that state, keyed on the TURN COUNT — and `classify_result` now tests the cap **before** the loss, on the same constant
(`reward_weights._TIMEOUT_TURN_CAP` == `MAX_TURNS`; parity pinned by
`trace_result_test.test_the_classification_matches_the_training_rewards_own_timeout_rule`).

🚨 **A PRE-DRAW-BUCKET TREE'S `draw: 0` IS NOT A MEASUREMENT.** Every trace written before
2026-09-07 carries no `meta.result_vocabulary`, and that ABSENCE is what dates it. In that era a
timeout was written as an ordinary `loss_*` (separable only by `meta.turns >= MAX_TURNS`, which is
what the G7 kill clause has always done) and a tie was **dropped before the summary was written** —
no file, no count. So such a tree can estimate a STALL rate but a TIE rate is **NOT MEASURABLE**
there. `run_summary()` reports `result_vocabulary` (the eras present, sampled) and
`result_vocabulary_note` (`trace_result.era_note`, `None` when the tree is all-current); `/` prints
the note above the outcome chart, and the chart omits the draw series entirely rather than drawing
a flat zero line. A MIXED tree — a run that restarted onto new code mid-flight — gets the note too.

🚨 **AN UNKNOWN RESULT IS REFUSED, NEVER RENDERED.** `battle_overview` / `battle_turns` call
`trace_result.result_of`, which raises `UnknownTraceResult` on a token outside the vocabulary
(`"TIE"` included — it is the string the *previous* writer would have used). A forensic tool that
displays an outcome it cannot classify is how a mislabelled bucket survives being looked at. An
ABSENT result is not an unknown one and passes through.

**Where draws sit in the CAPTURE QUOTA: their own bucket** (`_FORENSIC_DRAW_QUOTA` = 5, beside
win 5 / loss 10). Folding them into the loss quota — which is what the old code did — lets a stall
storm evict the decisive losses the prober exists to study. `calibration` EXCLUDES draws (no binary
realized label) and reports `n_draw_excluded` rather than dropping them silently; `awareness_scan`'s
`cap_loss` accepts `loss` **or** `draw` at the cap so the row means the same thing across the whole
archive.
