# Evaluation end state — the league, the meters, the budget

**Status: ALWAYS-CURRENT SKELETON (owner, 2026-10-02: "put the skeleton in and the TODOs, and we fill it out
with our implementation over time").** It says how we evaluate: which questions eval answers, with what statistics,
on what infrastructure and at what budget. A build or decision that differs updates this doc and its Decision
record in the same commit. Era context: [`era_plan_post_m5.md`](era_plan_post_m5.md).

## 0. Principles
1. **Every eval DECISION is deterministic, with a declared error rate.** A sequential test with stated α/β, or a
   margin a measured noise floor justifies. Never a threshold we hope the noise stays on the right side of.
2. **Mirrored team pairs, with the PAIR as the unit,** for every SYMMETRIC comparison: pool eval, promotion, the
   pool matrix, the plateau test, anchors. Pinned-team meters (untaught, `best_response_gap --play`) stay unmirrored,
   because mirroring would mix piloting with response (`eval_and_rating.md`).
3. **The regime is recorded on every row** (greedy/sampled, mirrored, the eval core). Readers refuse to mix regimes.
4. **A cheap method is trusted only after it reaches the dense method's decisions on our own data.**
5. **Re-ground week-one choices** (§6). Several eval parameters were set by what one CPU could afford in week
   one, not by the precision a question needs.

## 1. The questions eval answers

| question | meter / test | status |
|---|---|---|
| Is plain training still paying? (**plateau**) | head-to-head GSPRT vs the snapshot ~7 GPU-h back (H1 0.52) **and** an outside panel, once per 10M steps; < 2 Elo per training GPU-hour = plateau | DESIGNED (era plan) — TODO build |
| Does a candidate join the pool? (**promotion**) | GSPRT on mirrored pairs, H0 0.50 / H1 0.55, α = β = 0.05, cap 1,680 pairs (`sprt_promotion.py`) | BUILT, default OFF (`e9c5ab2d`) — flips ON at X26 |
| Who stays in the pool? (**eviction / cycling**) | max-entropy Nash averaging on the pool matrix; HodgeRank cyclic width vs noise = the cycling meter | DESIGNED (TASK_BACKLOG T20) — TODO literature review + build |
| How strong is it? (**rating**) | `snapshot_ladder/ladder.json` (dense BT, recipe-stamped) at run end, matched snapshot count | BUILT (`main.elo`) |
| Is the loop working? (**exploiter gap**) | `main.best_response_gap`: must FALL round over round | BUILT |
| Against the outside world? (**anchors**) | `main.anchors` (SmallRL at milestones; Kakuna/Foul Play at run end), greedy vs greedy, mirrored | BUILT; cadence TODO (TASK_BACKLOG T18) |
| Does the value tell states apart? (**discrimination**) | TODO: one registered within-game discrimination meter, by phase and opponent class | TODO (era step 3) |

## 2. The evidence tiers (pool matrix and, where they apply, every pairwise read)

| tier | what | cost | status |
|---|---|---|---|
| T0 | reuse games already played as a DISCOUNTED prior (power prior): the trainee's games around a snapshot's birth approximate its row; regime-flagged | free | TODO |
| T1 | sparse probe: a new member vs a spanning set on the rating spine; HodgeRank ratings (the spine is well-posed sparse; CYCLES are NOT) | small | TODO |
| T2 | targeted games only where they decide something (residual edges, members near the eviction boundary), ResponseGraphUCB-style | proportional to cycling | TODO |
| T3 | dense audit: the full matrix (20 members ≈ 190 pairs × 200 games ≈ 38k games ≈ 9 min background) = ground truth | periodic | TODO |
| gate | pre-registered: T0–T2 must reach T3's eviction decisions with materially fewer games, else play dense (affordable to ~40 members) | — | TODO |

## 3. Statistics in use

| method | used for | reference | status |
|---|---|---|---|
| GSPRT on the pentanomial, λ bisection, minimum pair count, Wald bounds | promotion; the plateau test | Fishtest (Van den Bergh); Wald 1945 | BUILT (`sprt.py`) |
| CUSUM | catching plateau onset sooner (refinement) | Page 1954 | TODO, optional |
| HodgeRank | the transitive spine plus cyclic width | Jiang, Lim, Yao, Ye 2011 | BUILT for trainee × bots (`hodge.py`); TODO pool × pool |
| Nash averaging (max-entropy) | eviction | Balduzzi et al. 2018 | TODO |
| α-Rank + ResponseGraphUCB | decisions on an incomplete, noisy matrix | Omidshafiei et al. 2019; Rowland et al. 2019 | TODO, in reserve |
| Active ranking | game allocation | Heckel et al. 2019 | TODO |
| Power prior | discounted reuse of old games | Ibrahim & Chen 2000 | TODO |
| Population size vs cyclicity | the pool cap | Czarnecki et al. 2020 (spinning tops); AlphaStar league (Vinyals et al. 2019); PSRO (Lanctot et al. 2017) | TODO |

## 4. Infrastructure
- **Best-effort (background) eval** (TASK_BACKLOG T19): run eval in the training cycle's idle capacity. During the
  update the CPU is about 15/16 idle and the GPU saturated, so decide where eval INFERENCE runs and measure the cost
  to training. Design it together with X14 (rollout/update overlap). TODO.
- **Slot hot-swap:** T2 slots load weights by an in-place copy (`slots.copy_in`: no allocation, captured graphs
  unchanged, parity-verified at load). With the reserved eval slots (6 today), any pool member can be evaluated
  without growing GPU memory. BUILT; TODO wire it to an eval scheduler.
- **Pool larger than the GPU's active set:** snapshots already live on the CPU, so the pool may grow while PFSP picks
  the per-rollout active set (the AlphaStar league pattern). TODO.
- TODO, verify: whether the Rust eval core's cycle BLOCKS training today. The `--eval-freq` help text says
  non-blocking and self-skipping; the sizing O9 read says blocking. Settle it before T19's design.

## 5. The eval budget
- **Today:** an eval cycle every 2M steps (`EVAL_FREQ_STEPS`), **100 games per opponent per cycle**
  (`EVAL_GAMES`). That is a week-one figure, set by what the CPU could afford then.
- **Capacity now:** the Rust eval core plays about 1,000 games in about 14 s (sizing). At N = 256, 2M steps is
  about 17 min of training wall.
- **Precision, not affordability, should set it.** Per-opponent standard error is √(p(1−p)/n): 100 games ≈ ±5 pp,
  400 ≈ ±2.5 pp, 1,600 ≈ ±1.25 pp. Mirrored pairs reduce these by the share of variance that is team luck, which
  is TODO to measure.
- TODO: derive games per opponent per 2M steps from each question's needed precision. Inputs:
  - the plateau test's band;
  - the promotion SPRT's expected pairs;
  - T3's dense-audit cadence;
  - the panel reads.

  Then check it fits T19's measured background capacity. Record the derivation here.

## 6. Week-one assumptions to RE-GROUND (owner, 2026-10-02: "encourage the agent to re-evaluate key assumptions")

| parameter | today | origin | re-grounding question |
|---|---|---|---|
| games per opponent per eval cycle | 100 | week-one CPU budget | what precision does each question need (§5)? |
| eval cadence | every 2M steps | week one | per question: plateau once per 10M; promotion per candidate; matrix per promotion |
| pool size | 20 (`DEFAULT_MAX_SNAPSHOTS`) | — | what does spinning tops say for our measured cyclic width? |
| eviction | oldest-first (or spread retention) | — | Nash averaging (§3) |
| promotion | first threshold crossing (0.55 / 0.65) | — | SPRT (built), flips at X26 |
| bots in the opponent mix (`heuristic_fraction`) | a few % | week one | still useful once the pool is strong? |
| the eval team set and its size | — | — | representative of the ladder teams (era step 5)? |
| the trace quota preferring losses | — | prober needs | still the right default for the readers? |
| per-pair games on the snapshot ladder | — | — | enough for its ±10 Elo claim with mirrored pairs? |

## 7. TODO (fill in as implemented)
- [ ] Literature review (owner, required before the T20 build): the references in §3.
- [ ] T19 background eval: design, capacity measurement, build.
- [ ] T20 tiered pool matrix + Nash eviction + cycling meter, with the T0–T2 vs T3 validation gate.
- [ ] Plateau test build (era plan).
- [ ] The discrimination meter (era step 3).
- [ ] §5 budget derivation; §6 re-grounding, one row at a time with evidence.
- [ ] Settle the blocking question in §4.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | Mirrored pairs **(owner)** | symmetric evals mirror; pinned-team meters refuse | mirroring everything | piloting vs response (`04aa4a45`) |
| 2026-10-02 | Promotion **(owner)** | GSPRT on mirrored pairs (Fishtest's method + a minimum pair count), Wald bounds | overshoot-corrected bounds (saved ~21 % of pairs, false promotion 5.07 %) | `measurements/sprt_promotion/` |
| 2026-10-02 | Performance checks **(owner)** | deterministic performance-SHAPE tests + an on-demand benchmark | a wall-clock perf guard | `48265bf8` |
| 2026-10-02 | Pool defence against cycling **(owner)** | tiered matrix + Nash averaging + Hodge meter, validated against dense; grow the pool rather than evict harder when cycles are wide | hope; recency-only eviction | TASK_BACKLOG T20 |
