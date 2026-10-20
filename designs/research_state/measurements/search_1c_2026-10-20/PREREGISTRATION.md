# STAGE 1c: more (or team-level) proposals, importance-resampled by the opponent's own choices — are the worlds nearer the truth, and does honest search then pay? PREREGISTRATION (2026-10-20)

**Written and committed BEFORE any world of a registered state is built for this unit.** Spec:
[`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §4 (f), §6.2 (stage 1c's row); the L3 read
([`../search_l3_2026-10-20/README.md`](../search_l3_2026-10-20/README.md)) and its registered consequence; the literature
review [`hidden_info_search_review_2026-10-20.md`](../../literature/hidden_info_search_review_2026-10-20.md) §3, §10
(L3: "the exact-world mass after resampling 8 of 64 proposals by weight"). Nothing below changes after a registered
state is built; a deviation is reported as a FINDING in the README, never edited here. The read script
[`read_1c.py`](read_1c.py) and the facts meter [`c1c_common.py`](c1c_common.py) are part of this registration (their
sha256 are in [`c1c_registration.json`](c1c_registration.json)).

**What was touched before registration** (no world of a registered state was built; no meter against the truth was printed or read for any state):
- The module code (below) and its tests, on the committed test bank (`BANK_V1`).
- An ENGINEERING SMOKE of the driver at N = 64 per set on 4 states of the maturity interim bank that are NOT LEAN and
  NOT among the 287 (sha256 order, salt `search_1c_smoke_v1:`; rows in `~/gen3ai_archive/search_1c_smoke/`), in one
  heavy slot. It printed only the costs, the exclusions and each weighted arm's effective sample size; no meter against
  the truth was printed or read. Result: 4 / 4 states built in both sets, 0 defects, 3.4 wall-min in all (9–113 s a
  state; F's build 6–89 CPU-s, the corpus set 1–3, the likelihood of 129 worlds 1–19), peak 3.6 GB. The ESS of the
  likelihood-weighted 64 F worlds was 2.8–20 (the dice alone: 40–58). Two engineering facts read from the rows: on a
  state with every opponent mon seen and no consistent corpus team, the C set IS the F set (stage 1's common random
  numbers); on a turn-2 state no opponent choice is scored yet, so the likelihood is flat.
- Because the smoke's cost allowed it (the brief: "N = 64, or more if the cost allows"), N was raised to 128 per set,
  with the brief's N = 64 kept as nested SECONDARY arms, and a fixed tempered weighting (L^0.5) added as a
  DESCRIPTIVE read of the likelihood's sharpness; the memory cap was raised to 16 GB.
- Two FUNCTIONAL checks of the code path (N = 4 and N = 12, on 1 and 2 other non-LEAN states, OUTSIDE a heavy slot as
  light jobs: ≤ 1.2 GB, ≤ 35 s each, `nice -n 19`, SCHED_IDLE, an address-space cap). They printed only row shapes,
  costs and the read's gate results (G5: 16 / 16 resamples recomputed exactly).
- The read script's selfcheck on synthetic states.
- Stage 1's registered ids and order, its corpus cache (a read-only copy) and its fitted α; L1's fitted anchors.
- Stage 1's own outputs were NOT read (it is running; its read is due at its stop, 2026-10-20 18:27 PDT).

Abbreviations: F = the fully honest world builder (every hidden fact drawn from Smogon usage, kept only if it replays
what our side saw); C = stage 1's team-level corpus sampler on the ladder corpus (C_ladder, design §4 (c)); L = the
opponent-choice likelihood (L3; design §4 (f)); IMP = decision improvement over the policy's own top choice on the
true world's held-out referee seeds; CI = a battle-clustered percentile bootstrap interval (2,000 draws, seed 20261020,
states weighted equally).

## 0. The question

L3 found the likelihood a strong JUDGE: among the truth and 8 honest worlds it gives the truth 0.249 of the weight
(AUC 0.692), the dose curve's range. But re-weighting today's 8 honest worlds moves the share of the seen mons' correct
unrevealed moves only +0.005: no proposal is near the truth, so there is nothing to pick. Stage 1c gives the judge more
and better candidates. **Does a larger honest proposal set (128 F worlds), a team-level one (128 C worlds: whole real
teams), or both (256), importance-resampled by each world's damage-roll weight × L, hold worlds clearly NEARER the
truth than today's 8 F worlds? And where it does, does one-ply honest search over the resampled 8, with L1's anchored
acting rule, beat the policy on these held-out states?**

## 1. The bank, the anchor, the opponent model

- **States:** the world sampler's stage-1 bank, unchanged (L3's): the maturity interim LEAN states at X26 225M that
  the banked F could build, **287 states** (`world_sampler_stage1_2026-10-19/s1_common.py` `common_ids()`, sha256
  `cf8d8c6f…dfb`), processed in **stage 1's registered order** (sha256(`ws_stage1_order_v1:` + id)). Each carries its
  opponent stratum (**bot**: a scripted eval opponent; **snapshot**: one of X26's own earlier snapshots). Read-only from
  `~/gen3ai_archive/hidden_info_maturity`.
- **The opponent model** is L3's, unchanged: X26's **225M** checkpoint (the maturity archive's registered copy, sha256
  re-checked at every load), CPU, one torch thread, `main.value_search.opp_likelihood.world_logliks` (the opponent's own
  rows, its seat; unobserved and pending choices excluded; the guards raise).
- **Held out:** no parameter of this unit is fitted on these states. L1's anchors were fitted on Tier 1 (15M
  policies); N, the resample rule and the arms are fixed here.

## 2. The proposals (per state; the builder at the DRIVER commit, `main` after this unit's code ship)

One `WorldSampler(reveal="off", dice="rolls", max_tries=60)` with F2's legality rule ON, `key` = the decision id, ONE
screen cache per state:
- **P_F: N_F = 128 honest F worlds** (`sample(log, at, side, 128, key=id)`).
- **P_C: N_C = 128 C worlds**: the same builder with stage 1's C_ladder departures swapped in for that one call
  (`samplers_s1.CorpusProposal` on the ladder corpus, `corpus_s1` normalised and validator-checked, **α = 89.125**
  from stage 1's `alpha_fit.json`, the backoff stage 1's B; imported unchanged). With probability λ = n / (n + α) a
  world is a consistent ladder team (its never-seen mons WHOLE, its seen mons' unrevealed moves), else a B world.
  Built AFTER P_F on the same screen cache, so a C world's seen mons take P_F's screened item, ability and spread
  (common random numbers).
- Every world carries its damage-roll log-weight (`Worlds.log_w`) and the rolls it took (`Worlds.rolls`, its log's
  `"rolls"`), and every consumer below carries them.
- **An exclusion:** a `WorldError` building P_F (resp. P_C) excludes the state from every arm built on that set
  (counted, named). Any other exception is a DEFECT.

## 3. The arms (each a weighting of a proposal set; the weights are the module's: `opp_likelihood.posterior_log_weights`)

| arm | proposals | weight of proposal j | role |
|---|---|---|---|
| **F8** | the first 8 of P_F | uniform | **the baseline of record** (today's honest builder under `rolls`, J = 8) |
| **F128L** | P_F (128) | exp(log_w_j + log L_j), self-normalised | PRIMARY (i) |
| **C128L** | P_C (128) | exp(log_w_j + log L_j) | PRIMARY (ii) |
| **M256L** | P_F ∪ P_C (256) | exp(log_w_j + log L_j) | PRIMARY (iii) |
| F64L, C64L, M128L | the first 64 of P_F, of P_C, and their union | exp(log_w_j + log L_j) | SECONDARY: the brief's N = 64, and the N-scaling (128 − 64) |
| F128d, C128d | P_F, P_C | exp(log_w_j) (the dice only) | SECONDARY: the larger set without the likelihood |
| C8 | the first 8 of P_C | uniform | SECONDARY: the corpus without the likelihood (stage 1's C_ladder, under `rolls`) |

The first n of a set is itself an n-world draw (each world j has its own key-seeded random stream), so the 64-world
arms are nested in the 128-world ones. **DESCRIPTIVE:** each primary arm with the likelihood TEMPERED, exp(log_w_j +
0.5·log L_j) (fixed here, never fitted), for whether the network's sharpness helps or hurts.

**The resample** (what a search would play): each weighted arm's 8 worlds are drawn by SYSTEMATIC resampling on its
weights (`opp_likelihood.resample`: one uniform per key `search_1c:<arm>:<id>`; a heavy proposal kept more than once),
the kept worlds equally weighted, each with its own rolls. The phase-A meters use the WEIGHTED expectation over all the
arm's proposals (the resample's expectation, without its sampling noise); the playouts use the resampled 8.

## 4. PHASE A: the meters (zero playouts)

**The truth's hidden facts** (what the searched side has not seen at the root): each true never-seen species; each
seen mon's true UNREVEALED moves (a Hidden Power one move whatever its type); each seen mon's item when play did not
reveal it; each seen mon's spread (nature, EVs, IVs); each never-seen mon's move set. Per world j
(`c1c_common.world_facts`):
- **H_j = the share of the BINDING facts the world holds**: (true never-seen species held + true unrevealed moves held)
  / their number (L2's fact set, stage 1 Amendment A1). Undefined on a state with no binding fact.
- **M_hit**, **M_exact** (every seen mon's unrevealed set exact), **U_hit**, **U_exact** (the never-seen species SET
  exact), **q_SM** = U_exact ∧ M_exact (exact on both binding classes: the dose curve's exact-world unit), **item**,
  **spread**, **U_moves** (the true never-seen mons whose same-species world mon holds their exact move set), **near4**
  (the world shares ≥ 4 whole mon sets with the truth: a near copy).

Per state and arm, each meter is its expectation under the arm's weights: E_X[x] = Σ_j w_j x_j.
- **PRIMARY: ΔH_X = E_X[H] − E_F8[H]**, per primary arm X, pooled.
- **SECONDARY** (each with its CI, pooled and per stratum):
  - the same contrast vs F8 for every meter and arm;
  - the exact-world mass E_X[q_SM] itself;
  - **the likelihood's own selection** on the same proposals: F128L − F128d and C128L − C128d, every meter;
  - **the proposals' own effect**: F128d − F8, C8 − F8, C128d − C8;
  - **the N-scaling**: F128L − F64L, C128L − C64L, M256L − M128L (H, M_hit, U_hit, q_SM), and the 64-world arms'
    verdicts by the same rule as the primaries;
  - **ESS** = (Σ w)² / Σ w² of each weighted arm (and ESS / N, and the share of states with ESS < 2);
  - **NW_lift** = the weight on the NEAREST proposals (the maximal H in the set) minus their uniform share.
- **DESCRIPTIVE:** the near-copy rate of the C proposals (F3's leak class), the corpus λ, the costs.

## 5. PHASE A's rules, THE BAR, and the registered consequences (deterministic; rule 8)

On each primary arm's pooled **ΔH_X** 95 % CI [lo, hi], in this order:

| verdict | rule | meaning |
|---|---|---|
| **NEARER** | lo > +0.0005 | the weighted proposals hold worlds clearly nearer the truth than today's 8 |
| **FARTHER** | hi < −0.0005 | they hold worlds clearly farther |
| **NULL** | lo > −0.0095 and hi < +0.0095 (inside ±0.01) | no move larger than a hundredth of the binding facts |
| **NOT-DETECTED** | otherwise | inconclusive at this n |

Two flags on a NEARER arm: **MATERIAL** iff lo > 0.0505 (at least 5 points more of the binding facts held: ten times
L3's +0.005 reweighting lift), and **DOSE** iff the lower bound of E_X[q_SM] > 0.1005 (the exact-world mass reaches
half of the 8-rollout leaf's p\* ≈ 0.2, `search_ledger.md` §5.1).

**THE BAR for phase B:** at least one primary arm reads NEARER. Then phase B plays **F8, every NEARER primary arm,
and C8 when C128L or M256L is NEARER** (the corpus without the likelihood, so the likelihood's share of any gain is
read). The bar is DIRECTION, not size: a playout costs about 1 CPU-h per arm on the 287 (less than this phase's
builds), and the size question is the one the playout answers. MATERIAL and DOSE are reported beside it.

**Registered consequences of phase A alone:**
- **No arm NEARER:** among 128–256 honest or corpus proposals, the opponent's own choices find nothing nearer the truth
  than today's 8 worlds. The (f) route needs proposals that hold near-true worlds BY CONSTRUCTION (a corpus that
  covers the opponent population, or a far larger N), or it is closed for search at this maturity. No playout.
- **NEARER without MATERIAL:** a real but small move; phase B tests whether it is search-sized.
- **DOSE:** the first sampler on this bank to reach half the dose curve's exact-world weight.

## 6. PHASE B: the playouts (run only when the bar is met)

- **Where:** X26's pin `9c033489` + ONE cherry-pick, the damage-roll control `fc010d7d3` (stage 1's play recipe;
  `play_s1.check_pin` checks the Rust diff), in this unit's own checkout. The pilot's driver
  (`run_det_playouts.one_state`, imported unchanged): every legal action × the arm's 8 worlds played to the END (world
  j on dice seed j), the network greedy on both sides; the honest one-ply critic on seeds 40 + j. A world's rolls ride
  on its log (`play_s1.RollsLog`, unchanged); a world whose root tokens differ from the banked root's is REFUSED (a
  defect, never an exclusion).
- **THE IDENTITY TOOTH (before any play; must pass):** 12 fixed states (sha256 order, salt `search_1c_identity_v1:`):
  their banked F worlds re-played in this checkout reproduce the banked 225M F rows exactly (`out`, `trace`, `crit`,
  `other_open`).
- **The referee:** the banked TRUE world's held-out seeds 16..39 at 225M; **IMP = T(the rule's pick) − T(the policy's
  top choice)**. Scorers: LEAF-K8 (the mean of the 8 world outcomes), TRUNC-4-K8 (4 turns, then the critic), CRITIC.
- **The acting rule:** **PIKL** (L1's anchor: argmax Q(a) + λ·log π(a)) at L1's FITTED values, λ = 0.1 (LEAF-K8),
  0.015 (TRUNC-4-K8), 0.015 (CRITIC), from `../search_l1_l5_2026-10-20/fit_l1.json` (re-checked at the read). ARGMAX,
  GUMBEL (c 0.1 / 0.2 / 0.05) and THRESH (δ 5/32 / 0.02 / 0.02) are SECONDARY. L1's own `pick` (imported unchanged); ties
  to the policy's order (1e-6).

**Rules, per played PRIMARY arm X** (PIKL; 95 % CIs; rule 8):

| verdict | rule |
|---|---|
| **SEARCH-PAYS** | lower95[IMP_X] > +0.0005 at BOTH LEAF-K8 and TRUNC-4-K8: honest search beats the policy |
| **HURTS** | upper95[IMP_X] < −0.0005 at both |
| **NULL** | the LEAF-K8 interval inside ±0.01 |
| **NOT-DETECTED** | otherwise |

Flag **BEATS-F8** iff lower95[IMP_X − IMP_F8] > +0.0005 at both scorers. SECONDARY: the other rules and the CRITIC
scorer; X − C8 for the corpus arms; per stratum. No multiplicity correction across the (at most three) primary arms:
each verdict is quoted with its arm.

**Registered consequences:**
- **SEARCH-PAYS (any arm):** the first honest search on a held-out bank to beat the policy. Before anything is built on
  it: a replication on a second bank (E2's competent-opponent states, or the maturity plateau bank), and the owner's
  questions Q1 (a corpus sampler at search time) and Q3 (the opponent's choices as evidence) for any live use.
- **NEARER but no arm SEARCH-PAYS:** the worlds moved but not enough to pay; the gap is stated in the dose curve's units
  (phase A's q_SM and H against p\*), and the next lever is a proposal set that holds the truth's team (coverage), not
  the judge.
- **HURTS:** the likelihood-weighted worlds mislead search; the opponent model is the suspect (read per stratum).

## 7. The sample, the order and THE STOP

- **Phase A** processes the states in the registered order in CHUNKS, each in its own heavy slot of ≤ 45 min (no new
  state after 30 min; the call capped by a 45-min timeout); every state's row is fsync'd, so the done states are
  always a PREFIX of the order (a uniform random sample of the 287). A state started twice without a row is PARKED (a
  FINDING; it counts toward G2's defect budget).
- **THE STOP (phase A):** no state starts after **STOP_AT = the first launch + 30 h** (written once; a relaunch keeps it).
  The read is made ONCE, on the done prefix, at `A_DONE` or at the stop. **An INTERIM read** (`read_1c.py a --interim`)
  is made about 12 h after launch if the run is still going: descriptive, labelled INTERIM, no verdict and no bar; it
  changes no rule, no n and no stop.
- **Phase B** (if the bar is met) plays the same prefix in the same order, each state in every played arm before the
  next; its stop is its first launch + 24 h; read once at `B_DONE` or the stop.
- **Power** (disclosed): phase A's meters are deterministic functions of each state's worlds, so their CIs are tight
  (a ΔH of ≈ ±0.005 is resolvable at 287). Phase B's between-arm IMP differences carry about ±0.021 at 287 (design
  §6.2): it detects only a search-sized gain (the dose curve's p = 0.2 is worth ≈ +0.03 at LEAF-K8).

## 8. GIGO gates (a failure STOPS the read; reported, never patched)

- **G1.** The rows' ids are a prefix of the registered order, each once.
- **G2.** At most 2 DEFECT or PARKED rows (a guard of the likelihood, the world guard, a hook, or a crash).
- **G3.** Determinism: 6 fixed states (sha256 order, salt `search_1c_recheck_v1:`) rebuilt and rescored in a fresh
  process reproduce the stored weights, likelihoods, facts and resample indices exactly.
- **G4** (descriptive, not a stop). The truth's log-likelihood equals L3's stored value (same model, same root,
  `8fe4fb89`'s module) on the common states; a difference is a commit's encoder or core change, counted.
- **G5.** Every resample index, recomputed at the read from the stored weights by the module's own functions, equals the
  stored one; every set holds N finite log-weights. At build time every kept world IS its proposal (log, team and
  rolls).
- **Phase B:** the identity tooth (12 / 12); every refusal is a defect, counted per arm (more than 2 in an arm stops its
  read).

## 9. Hazards declared in advance (each reported as a FINDING with the read)

1. **Pool-opponent bank (optimistic).** The opponents are training-pool teams. The ladder corpus may hold near copies
   of them (F3's class); the near4 rate of the C proposals is reported. The network's own opponent model is optimistic
   on the snapshot stratum and mis-specified on the bot stratum (L3).
2. **The mixture M128L** treats the two proposal sets as equal-mass components; the proper mixture posterior would
   weight each set by its prior mass on the consistent worlds, which is not estimated.
3. **Correlated proposals.** The seen mons' screened facts come from one shared Metropolis-Hastings chain per mon
   (thinned), and P_C reuses P_F's screen: the 128 proposals are not 128 independent draws, so an ESS overstates the
   independent information.
4. **Exclusions** are not missing at random, and a 128-world build fails if any one world fails in 60 tries: P_F may
   exclude states that today's J = 8 builds.
5. **The world residuals:** the truth's dice except the damage rolls, and the hidden counters (shared by every world).
6. **Encoder and core:** the worlds are built and scored at the driver commit; the playouts run at X26's pin (stage 1's
   arrangement). G4 counts the truth-likelihood differences from L3.
10. **The likelihood is sharp** (the smoke: ESS 3–20 of 64): the resampled 8 may hold only a few distinct worlds, each
    played on several dice seeds. That is the posterior's verdict, not a defect; the tempered read shows its effect.
7. **One model, one bank, one anchor** (X26 225M, ≤ 287 states).
8. **Owner questions Q1 and Q3** (a corpus sampler; the opponent's choices as evidence) are OPEN: this is an OFFLINE
   measurement only; nothing enters `data/`, a network input or a live search.
9. **C here is not stage 1's C_ladder arm:** it is the same sampler under `dice="rolls"` and N = 128; stage 1's arm is
   under `seed` dice and J = 8.

## 10. Cost and run

**Phase A** (extrapolated from the N = 64 smoke, × 2 for the build and the likelihood): ≈ 100 CPU-s and ≈ 2 wall-min a
state, ≈ 8 slot-hours for the 287 at one single-thread worker; with the heavy-slot queue, ≈ 12–20 wall-h (the 30-h stop
bounds it; the read states its n). Memory: 3.6 GB measured at N = 64; the cap is 16 GB. **Phase B** (if the bar is met):
≈ 13 CPU-s per state per arm (stage 1's measured play cost), ≈ 1 CPU-h per arm on the 287, 3–5 arms.
CPU only (`CUDA_VISIBLE_DEVICES=""`), SCHED_IDLE, `nice -n 19`, `taskset -c 5-7,13-15`, one heavy slot per chunk
(`mem_cap.sh --slot`; rule 5b). Rows go to `~/gen3ai_archive/search_1c/`; nothing is written under `models/`, `data/`
or another unit's archive. The runner: [`launch_1c.sh`](launch_1c.sh) → [`loop_1c.sh`](loop_1c.sh) →
[`run_1c.py`](run_1c.py) (phase A) and [`play_1c.py`](play_1c.py) via [`run_play.sh`](run_play.sh) (phase B).
