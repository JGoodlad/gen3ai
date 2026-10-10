# `eval_traces/` retention — the trainer's own grooming and the manual `groom.py`

Always-current topic doc of `src/main/prober/CLAUDE.md`.

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### Retention / grooming (`groom.py`)

Training writes a trace pair per sampled eval battle (+ a ~62MB snapshot per cycle
when `--keep-eval-snapshots`, default 10, is on — a hard link to the same-step checkpoint where one is
byte-identical, so ~0 extra), so `eval_traces/` grows. The
groomer prunes it — **scoped strictly to `eval_traces/`**:

```bash
python -m main.prober.groom <run_dir> [--keep-trace-steps 10] [--keep-snapshots 10] [--apply]
```

Keeps full traces for the K most-recent eval steps (deletes older step dirs) and
`snapshot.zip` for the N most-recent. **Dry-run by default** — it prints a JSON
report (`removed_steps`, `dropped_snapshots`, `mb_reclaimed`, `bytes_hardlinked_not_freed` — a
hard-linked snapshot whose checkpoint stands frees no blocks and is NOT in `mb_reclaimed`); pass `--apply` to
delete.

This CLI is a **manual fallback**. The producer grooms its own data: the **trainer**
(rl_agent eval callback) prunes after every cycle — `_prune_eval_snapshots`
(`--keep-eval-snapshots`, default 10) — so a live run's ~62 MB weight snapshots stay bounded on
its own. 🚨 **`--keep-eval-trace-steps` now defaults to `0` = KEEP ALL** (2026-09-08): the old
cap of 20 groomed arm A's 10M-step traces off disk and made the win-prob ladder's registered
A@10M comparator uncomputable, so the TRACES are no longer auto-pruned at all. The
prober is read-only and **never** grooms. Use this CLI for finished runs, a
different retention, or a one-off deep clean.
