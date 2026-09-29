# TASK BACKLOG — build and engineering work

The one ranked list of BUILD tasks (owner, 2026-09-27): infrastructure, features, tools and fixes that
are not experiments and not tech debt. Experiments live in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md); tech debt in
[`TECH_DEBT_BACKLOG.md`](TECH_DEBT_BACKLOG.md) (its dispatch rules are unchanged). A task that an
experiment needs says so in "unblocks".

**Rows move** to §3 with the commit that finished them. Rank = what unblocks the most important
experiments soonest, within the week's quota.

**NORTH STAR (owner, 2026-09-27), now → ~2 weeks:** the Rust end state (M5 lanes; deletion part 2) + architectural tech-debt pay-down, including DISCRETE mons with OTHER tokens (`design_q_head.md` §1) and the opponent pointer's OTHER label (T4), landed as ONE retrain boundary so a fresh lineage starts on the Rust many-envs env. Next north star: search enablement.

---

## 1. SCHEDULED

| # | task | unblocks | when |
|---|---|---|---|
| T1 | **Rust core M5, Lane 0** — the shared env core boundary (`src/rust_env/`: generated column schema, refusal policy, build stamp, mimalloc/jemalloc global allocator, reused per-env buffers) | every other M5 lane | Mon 09-28 ~22:00 |
| T2 | **M5 T2 inference service** — BUILDING 2026-09-29 (`src/agents/inference/service/`; design + backend decision in `designs/endstate/program_rust_core.md` §2 "T2 DESIGN"; resume point `designs/research_state/measurements/m5_t2/PROGRESS.md`) — one GPU service for trainee, opponent and eval forwards: fixed weight SLOTS (stacked, same-arch), slot-tagged requests, fixed-width (or compiled dynamic) buckets, priority classes (training rollout > eval filler), measured by the learner's end-to-end throughput; GPU-over-CPU bias | X4, X7, X14, X19; background eval | with T1 |

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
| 11 | T13 | **GPU-upgrade benchmark** (owner, 2026-09-27: mini-ITX, so ONE card — a newer-generation replacement, not a second GPU) — after T2 lands, run a learner-only PPO update benchmark (one 98,304-sample rollout; BUILT 2026-09-28 as `python3 -m agents.training.learner_benchmark run --device cuda` — phases, ablations, per-epoch profiler; its first read is `designs/research_state/measurements/learner_bench_2026-09-28/`) + a batched-inference benchmark at M5 bucket sizes on the 3080 Ti and on rented cards (the two candidates, 5070 Ti and 5080, from a GPU marketplace for a like-for-like buy decision; a 4090/5090 and GCP L4/A100 as architecture references). **Attribute the bottleneck on the 3080 Ti FIRST** (no rental needed): (a) is the GPU the limit at all — GPU busy % vs the CPU env lane during a real M5 iteration (nsys timeline); (b) clock sweep — lock core clock down ~20% (`nvidia-smi -lgc`), then memory clock down ~20% (`-lmc`), one at a time, and read the throughput drop: ~proportional to core ⇒ compute-bound, to memory ⇒ bandwidth-bound, neither ⇒ launch/latency or CPU-bound; (c) Nsight Compute on the top kernels of the PPO update and of one inference bucket (SM % vs DRAM % of peak — the roofline side); (d) VRAM headroom at peak (`max_memory_allocated`) — capacity only matters if it caps env count, slots or batch. Report the verdict per phase (inference / PPO update), then speedup per card, plus power (SFX PSU) and thermals (small-case throttling). Bandwidth-bound ⇒ no candidate helps much (5070 Ti 896, 5080 960 vs 3080 Ti 912 GB/s) | the buy decision | S |
| 12 | T14 | **Torch upgrade 2.5.1 → ≥ 2.8 — MOVED INTO M5 as Lane K1** (owner, 2026-09-28; `designs/endstate/program_rust_core.md` §2 M5):  2.5.1's Inductor miscompiles the fused TeamTransformer learner graph (ledger 2026-09-28; worked around by a graph break at `6521f420`); 2.8.0+cu126 compiles the unsplit graph correctly (eval graph checked; train graph and 2.6/2.7 NOT checked). Needs: cu12x local builds replacing the cu121 pins in `environment.yml` (+ torchvision/torchaudio), the real-obs parity gate green at fp32 and TF32, a speed A/B, then drop the split. A natural boundary: with M5's T2 or the next lineage | remove the split; maybe faster kernels | S–M |

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
