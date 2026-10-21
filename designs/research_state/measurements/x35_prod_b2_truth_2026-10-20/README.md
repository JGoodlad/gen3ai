# Production's B2 truth refreshed under P\* (AM-X47c-2, 2026-10-20): READ A's turns re-branched under X26@280M

The orchestrator ruled on F-5 (AM-X47c-2, [`BATCH1_LAUNCH_SHEET.md`](../x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md)
§0) that §5.6.5's own truth-refresh rule fires for PRODUCTION. P\* (280M) is ≥ 50M past the old truth's 150M
continuation, so production's G-C B2 is read against this record, registered before any production X35 read.
AM-X47b-2's "production keeps its production truth" meant only that production does not read the toy's truth.

| side | B2 truth | rows | continuation stamp (`main.x35.gc.TRUTH_BY_SIDE`) |
|---|---|---|---|
| production (A1 / A2 / D1 vs C0) | **this record**, P\* greedy on both sides | [`rows_X26at280M_S32.compact.jsonl.gz`](rows_X26at280M_S32.compact.jsonl.gz) | `X26@280M greedy (both sides)` |
| toy (X47, toy A1 / A2 vs toy C0) | T\* greedy on both sides | [`../x35_toy_b2_truth_2026-10-20/`](../x35_toy_b2_truth_2026-10-20/README.md) | `toyTstar@290M greedy (both sides)` |
| *retired for G-C:* READ A | X26@150M greedy on both sides | [`../collapse_pilot_2026-10-17/rows_X26at150M_S32.compact.jsonl.gz`](../collapse_pilot_2026-10-17/rows_X26at150M_S32.compact.jsonl.gz) | `X26@150M greedy (both sides)`: now refused by `gc pool --side production` |

`gc.TRUTH_ROWS_REL` names both committed files. `gc_test.py::test_am_x47c_2_each_sides_committed_truth_carries_its_sides_stamp`
pins each file's stamp to its side, and checks that the 150M stamp is refused.

## What was run

This is READ A's recipe ([`../collapse_pilot_2026-10-17/README.md`](../collapse_pilot_2026-10-17/README.md)) and the toy
refresh's procedure, with ONE input changed: the continuation.
- **The same bank and subset:** `bank_x26` (content sha `46233db2…`) and the same 306-turn subset `subset_x26.json`.
- **The same branching:** every legal action × **32** shared dice seeds (CRN), played to the end in the TRUE world.
- **The continuation is P\*** = `rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256
  `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed` (R's frozen call). It was re-hashed from a
  read-only copy in the archive, and it plays greedy on both sides.

The command is [`run_truth.sh`](run_truth.sh) `<archive dir>`, then [`summarize.py`](summarize.py) `<rows.jsonl>`.
- **Where:** CPU only, at worktree commit `d6d567eb`. The settings were `CUDA_VISIBLE_DEVICES=""`,
  `nice -n 19 taskset -c 5-7,13-15`, one OpenMP thread and 3 workers. The Rust env library was built in that worktree
  at that commit; main's library was not used (see FINDING 2).
- **Heavy slot:** the runner ran under `scripts/ops/mem_cap.sh` (8 GB cap). Each ≤ 45-min chunk took one heavy slot,
  and the runner resumed between chunks.
- **Time:** 7 chunks, 306 / 306 turns, ≈ 5.1 h of run wall time (six 45-min chunks plus 33 min), so ≤ ≈ 15 CPU-h
  with 3 workers at nice 19. That is ≈ 41 turns per chunk against the toy's 306 in 22 min. Elapsed time was
  10:41 → 23:59 on 2026-10-20, including the slot queue, a failed first slot (FINDING 2) and a worktree build.
- **Memory and exit:** peak 2.86 GB; the last chunk exited 0.
- **Outputs:**
  - the full rows: `~/gen3ai_archive/x35_prod_b2_truth_2026-10-20/rows_X26at280M_S32.jsonl`, sha256
    `f33fc336347b28f4b7f35b16c51633fc623365f011f3ff9b9b417962fb07cd84`;
  - the committed compact form (`truth.compact`): sha256 `2ed100d8…c9211b`.

## What it shows ([`summary.json`](summary.json))

- **Completeness:** 306 rows, 306 ok, 0 refused, 0 missing, 32 seeds each. The stamp is the registered one.
- **Decisive turns** (some action separably worse than the best by more than ε = 0.1):

  | | 280M truth | 150M truth | in both | only in 280M | only in 150M |
  |---|---|---|---|---|---|
  | decisive turns | **169** | 179 | 135 | 34 | 44 |

  So 78 of the 306 turns change decisive status.
- **Near-best sets:** the action-wise near-best set is identical to the 150M truth's on only **80 of 306** turns, so
  **226 change**. On the 135 turns decisive in both truths, the near-best set is identical on only 24.
- **Against the toy truth:** 135 decisive turns are shared, and the near-best set is identical on 76 of 306 turns. P\*
  is about as far from X26@150M as it is from T\*.
- **Values:** mean V\* 0.249 at 280M against 0.295 at 150M. V\* = +1 on 60 turns against 71. The mean near-best count
  is 2.90 against 2.86.
- **Reading:** the near-best set is unstable across continuations. Any two continuations, whether 150M against 280M or
  production against toy, agree on it on only ≈ 25 % of turns, so B2 measures starvation against its continuation's
  own idea of near-best. This is why the refresh rule exists, and why the refresh must come before the first read.

## FINDINGS

1. **The cost was ≈ 4–10× the planned ~1–1.5 CPU-h** (≈ 5.1 h of run wall time, ≤ ≈ 15 CPU-h, in 7 chunks). The
   plan was extrapolated from the toy refresh, whose forward is cheap. Production's forward is not, so READ A's
   ≈ 3.5 h wall is the right prior. Each chunk also re-queued for a heavy slot.
2. **Main's built `src/rust_env/target/release/libpokesim_env.so` was STALE** (built 2026-10-19 19:18). It predates
   `dfeb3d92d` and `33e1753bf`, which changed `src/rust_env/src/search/game.rs` and `src/rust_sim/src/present/*`. The
   load-time signature check covers only the layout table, so a run on it would not have been refused. The worktree
   built its own library instead. Ruled with the orchestrator, who will handle main's library.
