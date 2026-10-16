# T38 — the plateau meter's parallel batches: replay identity and speed-up (2026-10-16)

Unit T38 (`designs/ops/TASK_BACKLOG.md` X26-8; `designs/endstate/design_x26_readiness.md` F14 / §2.6). The code:
`main.plateau tick --inflight-batches K` (default 4) → `main.h2h.play.play_waves` → `H2HEngine.play_batches` → the
executor's opt-in `seed_blocks` (`gen3_eval_seed_blocks_v1`). Everything ran on the agent cores under rule 5b
(`taskset -c 5-7,13-15`, `nice -n 19`, `OMP_NUM_THREADS=1`, `scripts/ops/mem_cap.sh`), CPU only
(`CUDA_VISIBLE_DEVICES=""`), into SCRATCH ledgers under `/tmp` (never the archive's), beside the live X26
(`rb_x26_s2601`, ≈ 2M steps). Load average ≈ 5–6 throughout, quieter than T33's 11–15: read the RATIOS, not the
absolute rates. Compute flags as T33: `--threads 4 --torch-threads 2 --n-envs 64`, front `proc`, release build.

## Files

- `extract.py` — pulls three RECORDED `plateau_t1` requests of the T-ENT toy (T33, the archive ledger) into the
  collected test's fixture `src/main/plateau_tent_replay.json` (per batch: pentanomial, counts, cycle seed, outcome
  digest; the decision on record).
- `run_toy.sh` → `toy_runs.txt` — the recorded check `rb_toy_ent_s4002` 65M vs 45M (GAIN at 680 pairs = 17 batches on
  record) ticked at K = 1, 4, 2, 8, sequentially, each into a fresh scratch ledger (a scratch run dir symlinking the
  two nodes, `model_config.json` and `metadata.json`). `compare.py` → `toy_compare.jsonl` — each replay against the
  record.
- `prod_rate.py` → `prod_rate.jsonl` (summary) and `prod_rate_detail.json` (per-batch pentanomials, digests,
  near-tie lists) — the production net (`rb_bf_eb_s2001` 15M vs 10M, the plateau's schedule namespace), the SAME 8
  batches of 40 pairs at K = 1, 4, 8 on ONE engine, one fresh scratch ledger per K.

## Results

| network | K = 1 | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|
| production, games/s (CPU-s per game, the Python process) | 1.031 (1.88) | — | **2.070 (0.94)** | 2.619 (0.75) |
| toy, games/s per wave | 8.2–9.0 | 12.3–15.2 | 18.1–23.3 | 26.4–29.1 |
| toy, wall of the whole tick (startup included) | 146.1 s (13.1 s of it a one-time env-core build) | 96.5 s | 79.4 s | 76.0 s |

- **Identity, production:** `same_games_every_k: true` — every batch's outcome digest and pentanomial identical at
  K = 1, 4, 8.
- **Identity, toy, against the RECORD:** at every K all 17 recorded batches were re-played with the recorded cycle
  seeds, pentanomials and `outcome_digest_all` (every game), and the decision was GAIN at stop batch 16, 680 pairs,
  LLR +3.071067199 — the recorded decision to the last bit. K = 2 / 4 / 8 added 1 / 3 / 7 surplus batches, never folded.
- Peak RSS 0.65–0.83 GB per tick (toy), one process at every K.
