# EXPERIMENT BACKLOG — the ranked research queue

The one ranked list of EXPERIMENTS and RESEARCH questions (owner, 2026-09-27). Build and engineering
work lives in [`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md); tech debt in
[`../ops/TECH_DEBT_BACKLOG.md`](../ops/TECH_DEBT_BACKLOG.md); observation CONTENT proposals in
[`../endstate/obs_enrichment_backlog.md`](../endstate/obs_enrichment_backlog.md). Results are banked in
[`ledger.md`](ledger.md) and summarized in [`UNDERSTANDING.md`](UNDERSTANDING.md); this file only ranks
what to run next and why.

**How rows move.** A row is **RUNNING** once it has a registration on main; **DONE** when its read is
banked (it moves to §3 with the ledger commit). Every experiment is pre-registered before its first
game, carries one meter and a decision rule, and states its prerequisites. Rank = expected progress
toward the goal per unit of GPU / quota, with prerequisites respected.

**The priority framing (owner, 2026-09-27).** Getting **Q right** (a value that DISCRIMINATES within a
game, enabling search and expert iteration) outranks root-causing belief memorization: if the beliefs
memorize the pool and Q succeeds, that is major progress; a root cause without Q is not. Memorization
is still chipped at in parallel, because Q's opponent columns are BUILT FROM the beliefs — memorization
becomes Q's generalization problem the day opponents switch to ladder teams. Pool memorization is a
SUCCESS milestone, not a failure: report on-pool first, off-pool second.

---

## 1. RUNNING / QUEUED (have a registration or a launch slot)

| # | experiment | question | status | prerequisite |
|---|---|---|---|---|
| X1 | **N0 end-of-run reads** (`measurements/n0_endofrun_2026-09-27/`, registered `79615d33`) | Is the new lineage's head a better search LEAF than the six prior heads? Where does N0 sit vs SyntheticRLV2 and Foul Play (shared 72-team set)? Foul Play vs SyntheticRLV2 head to head | queue waiting on N0's final (~Sun 23:15) | — |
| X2 | **Learner battery** (`learner_battery_2026-09-26.md`, `6df5160f`; decisions `bfde6db5`) | n_epochs 5 vs 10 (dose-matched), TF32, policy GAE λ 0.95 vs 0.80 — strength per GPU-hour | queued after N0: C → E5 → T32 → L95, ~Mon 11:00–16:30 | N0 final |

## 2. PROPOSED — ranked

| rank | # | experiment | question / hypothesis | meter(s) | prerequisite | cost |
|---|---|---|---|---|---|---|
| 1 | X3 | **Poor man's leaf** — our policy's top-N as the prior + Foul Play's hand eval (poke-engine `gen3/evaluate.rs`, MIT) as the leaf, root-relative | Is a within-game DISCRIMINATING leaf the missing piece? Does our prior + their leaf beat their brute force? | mirror battery vs unsearched self; vs Foul Play | N0 final; Tue quota reset | ~1 agent-day, CPU |
| 2 | X4 | **The Q-head experiment** — joint Q(s,a,b) = V + A + B + I (probability space, centred under π × α); sibling-readout MLP on shared tokens; flat opponent pointer (one softmax α over concrete candidates, OTHER as entities, masked when impossible); branch labels (Gumbel top-k rows, α columns, CRN dice, racing, fresh labels, stored inclusion probabilities) | Does a paired-comparison-trained A/I give the leaf quality six win-prob heads lacked? | leaf rows (mirror L2, separation, overrules); V calibration non-inferiority; strength guard; trunk gradient cosines (Q vs V, Q vs policy, per A/B/I); detached probe vs shared; corr(policy logit, A); α(OTHER) calibration; V-vs-table gap | M5 successors + T2 inference service | several agent-days |
| 3 | X5 | **Fixed-mass hypothesis tokens** (arch arm, bundled with X4) — team: 6 tokens (revealed + species hypotheses) + OTHER_species, mass 6; moves: 6 seats + OTHER_move, mass 4; per-slot categorical q = Smogon prior ⊕ learned delta, presence = min(1, k·q), OTHER = leftover; log-weight attention bias; replaces the blob belief slots | Do concrete, fixed-mass beliefs beat blob tokens for intent prediction and the Q interaction term? | intent NLL/calibration; OTHER rates falling; I-term quality; on-pool belief metrics (X8); strength guard | with X4 | in X4 |
| 4 | X6 | **Branch successors train V too** (AlphaGo-style: one alternative action then on-policy to terminal) | Does off-path coverage improve V as a search leaf without hurting on-trajectory calibration? | leaf rows; NEW off-path calibration meter; on-trajectory ECE non-inferiority | X4's label machinery | small add-on |
| 5 | X7 | **Env-scale** — N ∈ {48, 256, 1024} envs at matched samples/update, `--adaptive-batch policy`, delayed-label buffer for the critic, policy λ from X2 | Where do gains from more envs stop (critical batch)? Does the delayed-label buffer keep the critic fed at short segments? | strength per GPU-hour; `train/noise_scale_ratio_policy`; label lag; game-length histogram match (no short-game bias) | M5 | GPU-days |
| 6 | X8 | **Belief memorization — HOW and WHY** (chip away in parallel; CPU probes) — on-pool metrics first: mass on truth by reveal count, exact-team identification and at which reveal it locks on, OTHER calibration, learned gain over the Smogon prior, the belief win-rate A/B on pool; then the mechanism: per-team exposure, frozen-trunk linear vs MLP probes (k = 5), aux weight 0.05, prior fusion | Is memorization healthy in-distribution, and what mechanism will fail on ladder teams? | on-pool primary, off-pool secondary | none (rows from the paused 09-24 root-cause work reusable) | CPU, low quota |
| 7 | X9 | **Ladder-teams arm** — reproduce the lineage recipe with Metamon ladder teams as opponents (and candidate training teams) | Do ladder opponents carry the belief/Q gains off the pool? | the off-pool columns become primary | Metamon HP repair (TASK T8) | GPU |
| 8 | X10 | **Archetype exploiters** from repaired ladder teams inside the new lineage's population loop (pooled read, 200-game readers) | Does a wider exploiter set find ladder-relevant holes? | pooled Δ; untaught / SmallRL guards | X9's teams | GPU |
| 9 | X11 | **Omniscient twin** — OFFLINE only: detached privileged network (own params) on training-only `obs_true` from the Rust core; opponent's true mons through our full-info encoder | Where is the public model's error REDUCIBLE (modelling) vs IRREDUCIBLE (hidden info)? Where is information worth gathering (scouting)? | V − E_belief[V*] vs Var_belief(V*) by state type; value-of-information map | Rust core `obs_true`; determinized worlds | ~1–2 agent-days, background |
| 10 | X12 | **Learned E10 set mixture + learned roster mixture** (prototypes, exact reveal updates) | Do correlated set/team beliefs sharpen α and the hypothesis masses? | intent NLL; mass on truth; OTHER rates | X8's understanding | M |
| 11 | X13 | **Capacity arm** — 2× width at matched GPU-hours | Is the plateau capacity? (evidence so far favours optimization: multi-team distill +69 Elo) | strength per GPU-hour | M5 (cheap GPU) | GPU |
| 12 | X14 | **Off-policy overlap** — collect while updating (1 version lag), decoupled PPO clip + truncated IS + V-trace | Does overlap buy throughput without hurting learning? | strength per GPU-hour; IS effective sample size; lag distribution | M5 async collector | M |
| 13 | X15 | **Expert iteration** — search-improved play distilled into the policy | Does search + distillation raise the ceiling? | strength; leaf rows | a leaf that passes (X3/X4) | L |
| 14 | X16 | **Bot floor 5% vs 10%** | Is the permanent bot slice worth its compute late in training? | untaught meter; bot WR | a fork point after the floor binds | GPU, small |
| 15 | X17 | **Tactics probe analysis** (14,370 durable rows at `~/gen3ai_archive/tactics_probe_2026-09-24`) | (see its PREDICTION on main) | registered | none | CPU |
| 16 | X18 | **E1 / E11 observation arms** (what the opponent knows about us; `[of]` attribution) | Do those facts move strength? | registered per arm | obs batch pattern | M each |
| 17 | X19 | **bf16 opponent/eval inference fidelity** (T2) | Is bf16 equivalent for opponents? (greedy flips, paired battles, ±2 pp bar) | agreement; paired equivalence | T2 | S |

## 3. DONE (moved here with the ledger commit)

| # | experiment | verdict | ledger |
|---|---|---|---|
| — | Population loop round 2 | NOT DETECTED, N+ FINAL; side-check row S | `aa8d56ea`, `3d866fea`, `c301dcf3` |
| — | Belief win-rate A/B | branch 1, DiD +4.8 pp | `423bff1e` |
