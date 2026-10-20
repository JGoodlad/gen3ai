# The OFF-POOL BANK, and L3 and the decoy test replicated on it at P\* — PREREGISTRATION (2026-10-20)

**Written and committed BEFORE any battle of the bank is played and before any likelihood of a bank state is computed.**
Nothing below changes after that; a deviation is a FINDING in the README, never an edit here. The read script
[`offpool_read.py`](offpool_read.py) and every driver are part of this registration (sha256 in
[`offpool_registration.json`](offpool_registration.json)).

**Why.** L3 (`../search_l3_2026-10-20/`) found that the likelihood of the opponent's own choices ranks the true world
above today's honest worlds (AUC 0.692). Its decoy counter-test (`counter_test/`) read NOT-DETECTED: a pool team that is
not the truth already beats the honest worlds (0.575 [0.517, 0.631]), and against bots decoys match the truth. So L3 is
PROVISIONAL-POOL, and by stage 1c's A4 neither L3 nor any 1c NEARER / SEARCH-PAYS steers anything until it replicates on
an OFF-POOL opponent bank (held-out or `hl_05_26` teams, `design_world_sampler.md` F4) AND at the plateau checkpoint
(≥ 280M). This unit builds that bank and runs the two cheapest reads on it.

**What was touched before registration (no likelihood, no AUC, no outcome of a bank battle):**
- **The off-pool proof** (§2) ran: [`offpool_proof.json`](offpool_proof.json). It reads team files only.
- **The off-pool decoy corpus** (§6) was built: `<archive>/decoy_corpus.json`, 22,606 teams, sha256 `d2d531b5…f707`. Only
  its counts were read.
- **Engineering smokes**, all on seed 2026102099, never in the bank:
  - 28, 90 and 30 battles: timings, and the per-battle off-pool gate (all 120 + 28 battles passed);
  - the bank pipeline on the 120 smoke2 battles: the census counts per cell, gate ① (0 failures), a 20-state smoke bank;
  - the scoring driver on that smoke bank: build times, decoy counts and wall time per state only (20 states: 0 defects,
    8 held a decoy, G4 byte-identical on 20 of 20, 8.9 s a state).
- **A CPU forward benchmark of P\*** (rows per second).

Abbreviations: P\* = X26's plateau checkpoint, 280,000,170 steps; AUC = the probability that one candidate's likelihood
beats another's; CI = a 95 % battle-clustered percentile bootstrap interval (L3's: 2,000 draws, seed 20261020, states
weighted equally); F = the fully honest world builder.

## 1. The models

- **The acting model and the anchor: P\*** (`rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256
  `814823c5…91ed`), read-only, copied to `<archive>/ckpt/` and sha-checked at every load. P\* plays the TRACED side
  of every bank battle. L3 seated its anchor (the 225M checkpoint the bank was scored at) as the opponent model; here the
  anchor is P\*, so P\* is also the opponent model of the likelihood (§5).
- **The opponents** are the live plateau cycle's roster: the nine scripted bots and the five sentinels the 280M cycle met
  (`eval_results.jsonl`: X26's checkpoints at 268,000,171 / 236,000,155 / 194,000,195 / 160,000,085 / 134,000,128,
  each sha-checked, `offpool_common.SENTINELS`), sentinels greedy as live.

## 2. OFF-POOL: the definition and the proof

**Off-pool = a team X26's lineage never met in training.** X26 is a fresh run (`lineage.role` fresh, no ancestry, no
teacher). It trained at its pin `9c033489` from step 0 to the end (`pin_history`). Its trainee and every opponent (its
self-play snapshots and the scripted bots) drew teams from the pool `data/teams`
(`matchup_history`: trainee `default_biased`, opponents `pool`; no `team_mix`). The pin predates
`gen3_pinned_data_root_v1`, so X26 read MAIN's `data/teams`; no commit after the pin touches `data/teams`, so that tree is
the pin's.

**The opponent teams: `heldout72_hpfix`** (`../heldout72_hpfix_kakuna_2026-10-18/`). They are 72 `hl_05_26` ladder teams,
drawn by a pre-registered rule against our pool, with Hidden Power repaired from Smogon usage. Each file is sha-checked
against its manifest.

**The proof** (`offpool_refs.py proof`, run): with `gen3_team_overlap_v1` (a mon set = species + sorted moves, Hidden Power
typed), against
- R_PIN, every one of the 824 files under `data/teams` at the pin (a superset of the 719 teams the loader keeps), and
- R_COMP, Metamon's `competitive` set:

**0 exact teams (loose and strict), and 0 teams sharing ≥ 4 mon sets with any one reference team.** The most any heldout
team shares with one pool team is 3 sets, for 1 team; 68 of 72 share ≤ 1. Against R_PACKED (stage 1's `pool_pin.json`, the
719 teams exactly as the pin's trainer packed them): 0 packed teams in common by sha256, and at most 2 identical packed
mon entries per team. Every bank battle is then gated (§4 G-OFF).

## 3. The battles

One generated eval cycle of P\* (`gen_cycle.py`). It is the production offline Rust eval (`main.ops.eval_trace_gen`, CPU,
at L3's code commit `8fe4fb89`) with three patches:
1. the sentinels are the five above;
2. the opponents' team pool is the 72 heldout teams;
3. every opponent, bots and sentinels alike, draws from it.

The trainee draws its team as live (the pool, with its 10 % sample-team bias). The cycle is played in three PARTS, each in
one heavy slot so the forwards batch:

| part | opponents | games each | seed |
|---|---|---|---|
| bots | the nine bots | 18 | 2026102011 |
| sentA | sentinels 268M, 236M, 194M | 30 | 2026102012 |
| sentB | sentinels 160M, 134M | 30 | 2026102013 |

That is 162 bot and 150 sentinel battles. Every game is a pure function of (seed, opponent label, game index). Capture
is ALL: every battle is traced. A part that times out (INCOMPLETE) is discarded whole and replayed on the same seed. The
bank refuses a part whose games played differ from the table.

## 4. The bank: the maturity battery's draw (`make_bank.py`)

The draw of `hidden_info_maturity_x26_2026-10-18/make_mat_bank.py`, unchanged except the source and the size:
- **Pool:** every trace of the cycle, replayed through `core_events --trackers --obs` at `8fe4fb89`. Eligible = a free
  move round, turn 2–40, ≥ 3 legal actions.
- **Census:** k (opponent species seen), turn, material, and the stratum (`sentinel_*` → snapshot, a bot → bot).
- **Cells:** (k bin 1–2 / 3 / 4 / 5 / 6) × (snapshot, bot), processed k bin by k bin, snapshot first. **32 per cell (the
  maturity battery's LEAN size): 320 states, 160 per stratum.**
- **Order and caps:** candidates in `sha256("search_offpool_bank_v1:" + id)` order; ≤ 1 state per battle per k bin and
  ≤ 3 per battle.
- **A short cell STOPS the draw.** The ONE pre-registered extension adds cycle `cyc2` (the same three parts, seeds
  2026102021–23) and draws again by the same rule. A cell still short then STOPS the unit.

**Bank gates (a failure STOPS):**
- **G-OFF:** every battle's opponent team is one of the 72 heldout teams as the eval core packed them, and the traced
  side's is a pool team.
- **Gate ①:** the replay reproduces the recorded obs, mask and action (> 1 % failing battles refuses).
- **The clock-decode turn gate.**

The ids, the bank's content sha256 and the census go to `states_offpool.json` and are committed BEFORE any likelihood.

## 5. The candidates and the likelihood (L3's, unchanged)

Per state, at L3's code commit `8fe4fb89` (a detached checkout; `offpool_run.py`):
- **The truth and J = 8 honest worlds:** L3's call, `WorldSampler(reveal="off", dice="rolls", max_tries=60)`, F2's
  legality rule on, `sample(log, at, side, 8, key=<id>, screen_cache={})`. A state the builder cannot fill is an
  EXCLUSION (counted), never a defect.
- **The likelihood:** `main.value_search.opp_likelihood` (L3 PREREGISTRATION §3): log L(w) = Σ_t log π(a_t | o_t^opp(w))
  over the opponent's past observed choices, with the same scored / unobserved / pending rules. π comes from **P\***, on
  CPU, 1 torch thread. All candidates of a state (truth, 8 worlds, decoys) go through ONE deduplicated forward. Two
  log-likelihoods within **1e-6 nats** are a TIE. The familiarity control `conf` rides along.

## 6. The off-pool decoys (the counter-test's, with an off-pool source)

Up to **D = 4** per state, built and gated exactly as `counter_test/decoy_run.py` builds and gates a pool decoy:
- Tier 1's `pool_worlds.build_world`, the `_roll_match` damage-roll gate, the playout-open check and the THROWING
  consistency guard;
- candidates tried in `sha256("search_offpool_decoy_v1:<id>")` permutation order, in batches of 60, at most 1,000 gated.

Two things change:
- **The source: the off-pool decoy corpus** (`offpool_refs.py corpus`). It holds every `hl_05_26` team the engine plays
  (`utils.ladder_corpus`'s kept files), with:
  - the 72 heldout teams removed;
  - Hidden Power repaired exactly as heldout72_hpfix was (`hp_repair`, seed 20261018 + team id, heldout72's sanitiser),
    so a decoy's Hidden Power is drawn like the truth's, not the corpus's Dark artifact;
  - DROPPED if it equals, or shares ≥ 4 mon sets with, any R_PIN or R_COMP team, or its packed text is in R_PACKED;
  - validated and packed by the trainer's `Gen3Teambuilder`.

  22,606 teams. Counts: 49 engine-unplayable, 72 heldout, 1 pool-exact, 134 pool near copies, 0 validator drops.
- **One more near-copy rule.** A ladder player reuses a team across replays with different reconstructed fills, so a
  candidate is dropped if it shares ≥ 4 identical `set_key`s with the truth (the counter-test's rule) OR ≥ 4 mon sets
  with the same species and move set, Hidden Power untyped.

A candidate must hold every species the searched side saw and every move it saw (`pool_worlds.candidates`).

## 7. READ 2: L3 replicated off-pool at P\* (L3's metric)

`offpool_read.py` cuts each row to L3's shape (the truth + the 8 honest worlds) and calls **L3's registered read**
(`l3_read.py`, sha256 `036f6260…`, loaded by path and checked): A_s, the AUC, S1–S6, and the verdict rule on the pooled
AUC's CI [lo, hi] (±0.005 band per rule 8): WORKS lo > 0.555; WEAK lo > 0.505; REVERSED hi < 0.495; NO-EVIDENCE
hi < 0.545; else NOT-DETECTED.

**THE REPLICATION LABEL (this unit's PRIMARY for L3; A4):**

| label | rule | consequence |
|---|---|---|
| **REPLICATED** | pooled WORKS or WEAK, and the bot stratum WORKS or WEAK, or the snapshot stratum does not reach WEAK either | L3 holds off-pool at the plateau. Its PROVISIONAL-POOL label is lifted for L3 itself. A 1c verdict still needs its own off-pool replication |
| **REPLICATED-SELF-PLAY-ONLY** | pooled WORKS / WEAK, the bot stratum not, the snapshot stratum WORKS / WEAK | the signal holds off-pool only where the opponent model nearly is the opponent (A4's SELF-PLAY-ONLY) |
| **NOT-REPLICATED** | pooled NO-EVIDENCE or REVERSED | L3's separation does not survive off-pool: it is withdrawn as evidence about hidden facts, and candidate (f) with our network as the opponent model is dropped at the plateau |
| **INCONCLUSIVE** | pooled NOT-DETECTED | L3 stays PROVISIONAL-POOL |

Every verdict line states the bot stratum's verdict beside the pooled one.

## 8. READ 3: the off-pool decoy (pool memory or real-team coherence?)

The counter-test's registered read (`decoy_read.py`, sha256 from `decoy_registration.json`, loaded by path and checked)
computes, per state with ≥ 1 decoy:
- A_td, the truth vs its decoys;
- A_dh, the decoys vs the honest worlds;
- A_th, the truth vs the honest worlds;

plus the AUCs, their CIs and the conf controls. Its labels are mapped to this unit's meaning: here neither the truth nor
a decoy is a pool team, so pool memory cannot lift either.

**PRIMARY (b), AUC_dh's CI:**

| label | rule | meaning |
|---|---|---|
| **COHERENCE-LIFT** | lo > 0.505 | an OFF-POOL real team beats Smogon-usage worlds with no pool memory: the pool decoy's lift (0.575) is, at least in part, real-team COHERENCE |
| **NO-LIFT** | hi < 0.545 | a real ladder team gets no material lift off-pool: the pool decoy's lift was POOL MEMORY (with the hazards of §10 3) |
| INCONCLUSIVE | otherwise | |

Beside it (descriptive): whether the CI holds the pool decoy's 0.575.

**CO-PRIMARY (a), AUC_td's CI:** TRUTH-ABOVE lo > 0.505 (MATERIAL lo > 0.555); DECOY-ABOVE hi < 0.495; TIED hi < 0.545;
else INCONCLUSIVE.

**(c) The combined verdict** (the counter-test's table, relabelled):

| verdict | rule |
|---|---|
| UNDERPOWERED | fewer than 60 states hold a decoy |
| TRUTH-SPECIFIC | TRUTH-ABOVE and NO-LIFT |
| TRUTH-PLUS-COHERENCE | TRUTH-ABOVE and COHERENCE-LIFT |
| TRUTH-ABOVE-COHERENCE-UNCLEAR | TRUTH-ABOVE and INCONCLUSIVE |
| **COHERENCE-ONLY** | TIED / DECOY-ABOVE and COHERENCE-LIFT: a real non-truth team scores like the truth, so the likelihood measures coherence, not the truth's hidden facts |
| NO-SEPARATION-ON-SUBSET / INCONSISTENT | as the counter-test |
| NOT-DETECTED | (a) INCONCLUSIVE |

**Registered consequences:**
- **COHERENCE-LIFT:** pool-ness in the counter-test is not evidence of memorisation. Any corpus proposal (1c's C arms)
  inherits a likelihood advantage from coherence alone, so a corpus NEARER must be read against coherent non-truth teams
  (STRICT, placebo), never as truth evidence.
- **NO-LIFT:** the pool-bank reads (L3 0.692, 1c on the pool bank) are inflated by pool memory; only off-pool reads steer.
- **COHERENCE-ONLY:** L3's truth-specific part is not shown off-pool; reweighting by L (stage 1c) is a coherence prior,
  and its value is phase B's question on this bank.

The bot stratum's labels are stated beside the pooled ones (A4).

## 9. Gates of the reads (a failure STOPS; reported, never patched)

- **G1:** the rows' ids are the bank's ids (`states_offpool.json` `states_sha256`).
- **G2:** at most 2 DEFECT rows (any likelihood guard or the decoy consistency guard raising).
- **G3:** determinism: 10 fixed states (salt `search_offpool_recheck_v1:`), re-scored in a fresh process, reproduce the
  stored log L and conf exactly, decoys included.
- **G4 (descriptive):** the truth's opponent rows at the scored steps are byte-compared at X26's pin's `core_events`. If any
  differ, the primary is also read on the identical states, beside it.

**Descriptive additions (no verdict):**
- **Intervals clustered by OPPONENT TEAM** (72 teams recur across battles). L3's battle clustering is the registered one.
- **The truth-vs-honest AUC reweighted to L3's bank outcome mix per stratum.** This bank traces every battle; L3's
  traces were the live loss-enriched quota.
- **The AUC by outcome.**

## 10. Hazards declared in advance (each reported as a FINDING)

1. **The truth's own hidden parts are a reconstruction.** Metamon filled what a replay did not reveal from usage
   statistics (F7), and Hidden Power was filled from Smogon usage. Parts of the true team therefore look like an honest
   world's draws, which biases AUC_th toward 0.5 against L3's pool truths. It is conservative for READ 2.
2. **Outcome mix.** This bank traces every battle; L3's bank came from the live outcome quota (loss-enriched, most of
   all against bots). The draw rule is the same; the trace population is not. The reweighted AUC (§9) bounds it.
3. **Cross-bank comparison in READ 3.** The pool decoy's 0.575 was measured at 225M on the pool bank. Here it is P\* on
   the off-pool bank. A difference can come from maturity, the bank or the source. The PRIMARY's labels are absolute
   (vs 0.5), never a difference from 0.575.
4. **72 opponent teams.** About 4 states per team. The team-clustered interval is reported beside the registered one.
5. **One regime change in the snapshot stratum.** Live, a greedy sentinel draws the trainee's distribution; here it
   draws the heldout set. Each sentinel is an earlier X26 and the opponent model is P\*: the stratum stays the optimistic
   case (L3 hazard 1).
6. **The encoder.** The battles were played and the rows encoded at `8fe4fb89`; P\* trained at `9c033489`. G4 measures it.
7. **The decoy corpus is not a model of the ladder.** Its unrevealed parts are reconstructions too (F7), the same
   population as the truths. That is what makes truth vs decoy a fair test of truth-specificity, and it bounds what
   "coherence" means: coherence of a Metamon-reconstructed ladder team.
8. **One model, one anchor, one bank** (P\*, 320 states). No replication at another checkpoint is part of this unit.

## 11. Cost and run

The smokes on the contended box: a 90-battle bot part took 8.8 min, and a 30-battle sentinel part took 14 min; the
three parts are each one heavy-slot chunk of ≤ 45 min (`timeout 45m`; `--timeout-min 42`). Scoring: about 9 s a state
(the scoring smoke, G4 included), ≈ 1 slot-hour for the 320 in chunks of ≤ 45 min (`--budget-min 38`), then G3. CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh --slot`: [`run_bank.sh`](run_bank.sh), then
[`run_score.sh`](run_score.sh). Nothing is written under `models/` or `data/`. Archive: `~/gen3ai_archive/search_offpool_bank/`. The read writes `result_offpool.json` into this folder.

**Out of scope:** stage 1c phase B on this bank. Its cost is estimated in the README as a recommendation.
