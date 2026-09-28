# TASK BACKLOG — build and engineering work

The one ranked list of BUILD tasks (owner, 2026-09-27): infrastructure, features, tools and fixes that
are not experiments and not tech debt. Experiments live in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md); tech debt in
[`TECH_DEBT_BACKLOG.md`](TECH_DEBT_BACKLOG.md) (its dispatch rules are unchanged). A task that an
experiment needs says so in "unblocks".

**Rows move** to §3 with the commit that finished them. Rank = what unblocks the most important
experiments soonest, within the week's quota.

---

## 1. SCHEDULED

| # | task | unblocks | when |
|---|---|---|---|
| T1 | **Rust core M5, Lane 0** — the shared env core boundary (`src/rust_env/`: generated column schema, refusal policy, build stamp, mimalloc/jemalloc global allocator, reused per-env buffers) | every other M5 lane | Mon 09-28 ~22:00 |
| T2 | **M5 T2 inference service** — one GPU service for trainee, opponent and eval forwards: fixed weight SLOTS (stacked, same-arch), slot-tagged requests, fixed-width (or compiled dynamic) buckets, priority classes (training rollout > eval filler), measured by the learner's end-to-end throughput; GPU-over-CPU bias | X4, X7, X14, X19; background eval | with T1 |

## 2. PROPOSED — ranked

| rank | # | task | unblocks | size |
|---|---|---|---|---|
| 1 | T3 | **M5 remaining lanes** (Phase A plan, `program_rust_core.md` M5: FFI + process front ends over one core, training labels incl. the DELAYED-LABEL BUFFER, episode end/reward, opponent routing, bots ported (Lane F, owner yes), training integration behind a flag, eval on the core, in-process `successors()`, the M5 gate harness) | X4–X7, X14; the remaining deletion rows | ~24–33 agent-days |
| 2 | T4 | **Opponent intent: OTHER instead of masking** — belief misses become an OTHER label (today masked; `opp_intent/alpha_mask_rate`) | X4, X5 | S |
| 3 | T5 | **Ladder recipe v3 backfill for N0** — `--backfill-fresh` replaces 54 eval-cycle pairs with 200-game fresh ones (~10,800 games, ~1.5 CPU-h) | a clean N0 ladder under v3 | S, after the N0 read queue (~Tue) |
| 4 | T6 | **Promotion by SPRT + rate every candidate on fresh games** (M5 eval lane; background eval) | unbiased learning curves | M |
| 5 | T7 | **Memory-triggered restarts** (launcher: RSS over baseline + a ~24 h backstop; env child respawn without the learner) | fewer killed games | S |
| 6 | T8 | **Metamon HP repair** (backlog (f) P1 spec, `6779b1bd`): synergy with the spread, Smogon per-species HP usage, validator IV template | X9, X10 | M, next week |
| 7 | T9 | **Rust core M7** — the ladder client on the core; `play.py`, anchors and the prober read core rows; the Python encoder retires | live ladder; one-language obs changes | L |
| 8 | T10 | **Deletion pass, part 2** — the rows blocked on M5/M7 (Python trackers, obs speed-up layers, JSON search protocol + node drivers, `turn_delta_legacy.py`) | less to keep in sync | M |
| 9 | T11 | **Transform + Hidden Power belief crash** (both trackers; TECH_DEBT (a) P1) | live ladder | S |
| 10 | T12 | **Imprison livelock — verify it is gone** (TECH_DEBT (a) P1; likely fixed by `0896b7d9`) | live ladder | S |

## 3. DONE

| task | commit |
|---|---|
| Rust core M6 cutover (`--obs-source core` default) | `ac0b6469` |
| Deletion pass part 1; shaped reward path deleted | `43712881` … `e3ef16db` |
| Observation-architecture batch (v121) | `b0a28b5b` |
| Learner flags (`--policy-gae-lambda`, `--matmul-precision`, per-epoch KL/clip) | `2cc83080` |
| Policy drift meter (+ eligibility-conditioned rates) | `58229ace`, `a24c3db6` |
| Launcher restart fix (FRESH-only flags stripped; FRESH into an existing dir refused) | `ab27425f` |
| Ladder recipe v3 (fresh 200-game promotion baselines; relative-to-reference column) | `d3efa93e` |
| Anchors opponent-vs-opponent pair mode | `dcddac0b`, `d4799e8a` |
