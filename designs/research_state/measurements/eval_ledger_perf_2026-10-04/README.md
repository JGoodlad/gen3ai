# F-ED-22: the eval ledger's cost per claim / append, before and after the persisted indexes (2026-10-04)

**Question.** Eval U2 (`ecf9eeca`) measured that every `claim` / `append_row` of the ledger writer re-folded the WHOLE
archive's requests stream under the lock (~7 µs per event), and every `read` scanned every row shard — 11.9 ms per claim at
1.7k events and 37 ms at 5.1k, ~0.34 s per production cycle per run already in the archive. Does the incremental, persisted
fold (`agents/training/eval_ledger/event_index.py`, `row_index.py`) make the cost flat in the archive's size?

**Verdict: YES (MEASURED, `results.jsonl`).** A claim or an append at 50,000 events costs what it does at 1,700, within noise
(one `stat` per events file is the only term that still grows).

## What was run

`run_all.sh <before_src> <after_src> results.jsonl` → `perfbench.py` at 1.7k / 5.1k / 50k events, `--reps 5` (3 for the
slow BEFORE at 50k), on **tmpfs** (`/tmp`: CPU shape, fsync is free) and on the **NVMe** the archive lives on
(`~/.cache/gen3ai/tmp`: fsync-bound, ~5 ms per fsync).

- **BEFORE** = the main checkout at `ecf9eeca` (`PYTHONPATH=/home/goodlad/dev/gen3ai/src`, no index): the code F-ED-22 was
  measured on. **AFTER** = this unit's tree.
- The archive is SYNTHESIZED by writing the streams directly (a claim-by-claim build is itself quadratic in the old code), in
  the in-loop producers' shape: a cycle = 3 opens + 15 claims + 15 `row` events + 3 dones = 36 events and 15 rows in 3
  regimes; 40 cycles per writer file (~1,440 events: one 75M run). 1,692 / 5,076 / 49,968 events in 2 / 4 / 35 files; 705 /
  2,115 / 20,820 rows. (`audit` of the 1.7k archive is OK: the synthesis is a valid ledger.)
- Timed on a FRESH `LedgerWriter` (a fresh trainer process) with the index warm on disk: `fresh_first_claim` (the first claim
  after construction), `steady_claim`, `append` (`append_row` under its claim: the row, then its `row` event), `open`,
  `done`, `read_own` (a request-scoped `read`, 15 rows), and `cold_first_op` (the first operation on an archive that has
  never been indexed: for BEFORE that is just a fold).
- The box was NOT quiet (load average 3.9–4.8: other agents' jobs); the BEFORE / AFTER gap is two to three orders of
  magnitude, the AFTER rows are flat to ±0.2 ms on tmpfs.

## The numbers (milliseconds; claim = `fresh_first_claim`)

| events | files | fs | claim before → after | append before → after | read(own) before → after | cold first op after |
|---|---|---|---|---|---|---|
| 1,692 | 2 | tmpfs | 12.8 → 0.43 | 12.8 → 0.58 | 36.7 → 2.1 | 28.6 (builds the index) |
| 5,076 | 4 | tmpfs | 37.1 → 0.43 | 37.1 → 0.60 | 107.7 → 2.1 | 76.1 |
| 49,968 | 35 | tmpfs | 473 → 0.63 | 489 → 0.80 | 1,241 → 2.3 | 842 |
| 1,692 | 2 | NVMe | 18.0 → 5.8 | 23.5 → 11.0 | 36.9 → 2.6 | 63.8 |
| 5,076 | 4 | NVMe | 43.0 → 5.3 | 48.6 → 11.0 | 108.2 → 2.4 | 118.1 |
| 49,968 | 35 | NVMe | 501 → 6.1 | 522 → 11.2 | 1,259 → 3.5 | 904 |

On the NVMe the AFTER claim is ONE fsync (the event's own, ~5 ms) and the append two (the row, then its `row` event): the
index itself adds ~0.4 ms of CPU and no fsync (WAL + `synchronous=NORMAL`, and one idle "keeper" connection per process so
that closing an operation's connection does not checkpoint the WAL — without it each close fsynced the database: 20 ms per
claim instead of 6, measured on this disk).

## Reading it

- **Flat.** Claim and append are 0.4–0.6 ms at 1.7k and 5.1k events and 0.6–0.8 ms at 50k (tmpfs); 5.3–6.1 ms and 11 ms on
  the NVMe. The 0.2 ms drift at 50k is the 35 `stat`s (one per events file) plus reading the 35 cursor rows; it grows with the
  NUMBER OF WRITER PROCESSES, not with events.
- **A production cycle** (≈ 36 operations): the fold CPU (≈ 0.34 s × the runs in the archive: 0.34 s at one run, 10 s at 30) is gone; what remains is the fsyncs (~0.3 s per cycle on this NVMe: 15 claims × 5.8 + 15 appends × 11 + 6 opens and dones), flat.
- **The first operation on an unindexed archive builds the index once**: ~17 µs per event (0.84 s at 50k), under the lock
  (bounded at 60 s: that is ~3.5M events). A rebuild (a stale or corrupt index) costs the same.
- **A request-scoped read** (the SPRT's decision, a per-request resume, an X5 look) costs the request's rows. An `any` read
  and `audit` still consume the archive, by design.
- **Not measured here:** the in-trainer wall of a cycle (the GPU owner's real-launch gate, U2's hand-off); contention between
  several writer processes beyond the correctness test (`index_test::test_concurrent_writer_processes_leave_one_consistent_index`).

## Files

- `perfbench.py`, `run_all.sh`, `results.jsonl` — the benchmark, its driver and every row (labels `before-tmpfs`,
  `after-tmpfs`, `before-nvme`, `after-nvme`).
- The correctness half is the test file `src/agents/training/eval_ledger/index_test.py`: the index equals the full fold after
  every step of a scripted life; the same life with and without the index (`GEN3AI_LEDGER_INDEX=0`) writes byte-identical
  streams and the same refusals; stale / truncated / rewritten / vanished / out-of-order / corrupt / foreign-format indexes are
  rebuilt; `audit` finds a tampered index; a seed block recorded 12 cycles earlier is still refused; the indexed read equals the
  scan for every scope and every `as_of` (boundaries included); and the SHAPE tests that count what is parsed.
