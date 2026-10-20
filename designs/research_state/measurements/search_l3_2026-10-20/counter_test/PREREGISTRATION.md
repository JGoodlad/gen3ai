# L3's familiarity counter-test: does the likelihood pick the truth because it is TRUE, or because it is a POOL team? — PREREGISTRATION (2026-10-20)

**Written and committed BEFORE any likelihood of a decoy is computed on a registered state.** It answers the L3 review's
finding F2 ([`../REVIEW.md`](../REVIEW.md)). The read script [`decoy_read.py`](decoy_read.py) is part of this
registration; its sha256, and the driver's, are in [`decoy_registration.json`](decoy_registration.json). Nothing below
changes after the read. A deviation is reported as a FINDING in the README, never edited here.

**What was touched before registration:**
- The driver's engineering SMOKE (`decoy_run.py --smoke 8`) ran on 8 maturity-bank states that are NOT among the 287.
  It built the honest worlds and the decoys and printed only build counts and timings: 4 of 8 states had at least one
  decoy, 0 to 64 candidates per state, 3.5 to 27 s per state. It computed no likelihood.
- The read script's selfcheck ran on synthetic states.
- L3's own rows and result (`result_l3.json`) were read; they are published.

## 0. The question

L3 found that the likelihood of the opponent's observed choices, with X26's 225M network seated as the opponent, ranks
the TRUE world above today's honest worlds (AUC 0.692). Every opponent in that bank plays a TRAINING-POOL team, which
X26 played for about 225M steps. The confidence control (AUC_conf 0.481) shows the network is not SURER on the truth's
rows. It does not show the network does not PREDICT BETTER on a pool team for reasons that vanish off-pool: its policy
can encode team-specific habits for the team it is told it holds.

**The counter-test.** Score a DECOY in the truth's place: a pool team that is NOT the truth but is consistent with
everything the searched side saw. Two outcomes:
- **Familiarity explains L3:** the decoy scores like the truth (truth vs decoy ≈ 0.5), and above the honest worlds.
- **Evidence:** the truth clearly beats the decoy.

## 1. The bank and the code

L3's bank unchanged: the **287** registered states (`world_sampler_stage1_2026-10-19/s1_common.py` `ordered_ids()`, ids
sha256 `cf8d8c6f…dfb`), each with its stratum (bot / snapshot). The opponent model is X26's 225M checkpoint (sha256
`bc93a321…19ce`), loaded by L3's own `l3_run.load_model` (sha-checked) on CPU, 1 torch thread.

**The code is L3's own commit, `8fe4fb89`** (a detached checkout; `decoy_run.py --code`). The honest worlds, the
template, the likelihood and the replays are then exactly L3's, and gate G5 checks that the truth's and the honest
worlds' values reproduce L3's archived rows.

## 2. The candidates per state

**The truth and L3's 8 honest worlds**, rebuilt by L3's call: `WorldSampler(reveal="off", dice="rolls", max_tries=60)`,
legality on, `sample(log, at, side, 8, key=<id>, screen_cache={})`. A state the builder cannot fill is an EXCLUSION, as
in L3.

**Up to D = 4 POOL DECOYS:**
- **The pool:** X26's PIN packing of the training pool (`~/gen3ai_archive/world_sampler_stage1/pool/pool_pin.json`,
  sha-checked: stage 1's C_pool pool, the population the bank's opponents came from). Read-only.
- **Consistent with the reveals:** the pool teams whose species include every species the searched side saw and whose
  sets hold every move it saw (Tier 1's `pool_worlds.candidates`).
- **Not the truth, and not a near copy:** drop a candidate whose `team_key` equals the truth's. Also drop one sharing
  **≥ 4 identical mon sets** with the truth (`set_key`: species, item, ability, move set, nature, EVs, IVs; the design's
  near-copy rule). A near copy shares most of the truth's facts, so a tie with it would be misread as familiarity.
  Duplicate teams (the same `team_key`) are kept once.
- **Built as Tier 1's POOL builds a world** (`pool_worlds.build_world`): the candidate's sets at the seen mons' TRUE
  slots (moves aligned to the true slots), its other mons at the never-seen slots, the engine's gender placement, and
  the clause check.
- **Gated like an honest world:** the SAME damage-roll MATCH control the `dice="rolls"` honest worlds pass
  (`WorldSampler._roll_match` on this root). The searched side's obs rows and protocol up to the root must be
  byte-equal to the truth's under some feasible rolls (`pool_worlds.strict_verdict`), and the playout tool must open the
  world. The decoy replays the rolls its gate accepted. The truth replays on its own dice, as in L3.
- **A fixed seed:** the eligible candidates are tried in the order of a sha256 permutation keyed
  `search_l3_decoy_v1:<id>`, in batches of 60. The first D distinct teams that pass are the decoys, which is a uniform
  draw without replacement among the passers. At most 1,000 candidates are gated per state.
- **The world consistency guard** (`world_guard.check_teams`) runs on the decoys and THROWS: a raise is a DEFECT.

A state with no passing decoy is read for the gates but carries no counter-test quantity. It is counted, split into "no
eligible candidate" and "none passed".

## 3. The likelihood

L3's, unchanged (`main.value_search.opp_likelihood` at `8fe4fb89`; L3 PREREGISTRATION §3). The truth, the 8 honest
worlds and the decoys go through ONE deduplicated forward per state over the same scored steps. Two log-likelihoods
within **1e-6 nats** are a TIE (rule 8). The familiarity control `conf` rides along.

## 4. The metrics (states weighted equally; 95 % battle-clustered bootstrap, 2,000 draws, seed 20261020: L3's `Boot`)

Per state s with D_s ≥ 1 decoys, with score(x) = 1 if x > 1e-6, ½ if |x| ≤ 1e-6, else 0:
- **A_td(s)** = mean over decoys d of score(log L(truth) − log L(d)): the truth vs its decoys.
- **A_dh(s)** = mean over decoys d and honest worlds j of score(log L(d) − log L(j)): the decoys vs the honest worlds.
- **A_th(s)** = L3's A_s on the same state: the truth vs the honest worlds.

**PRIMARY: AUC_td** = the mean of A_td over the states holding a decoy, pooled over both strata.
**CO-PRIMARY: AUC_dh**, pooled.

**SECONDARY / DESCRIPTIVE:**
- **AUC_th** on the same states. The states with a decoy are not a random half of the 287 (fewer reveals, more pool
  teams fit), so L3's number is re-read on this subset, beside AUC_th on all OK states.
- **The pool share of L3:** (AUC_dh − ½) / (AUC_th − ½) on the subset, with its bootstrap interval. 0 means pool-ness
  explains none of L3's separation, 1 means all of it.
- The same three AUCs with `conf` in place of log L.
- Every quantity per stratum (bot, snapshot), with the same verdict rules.
- AUC_td by the decoys' mean shared mon sets with the truth (< 2 vs ≥ 2).

## 5. The decision rule (deterministic; rule 8: ±0.005 around each boundary is not clear)

**(a) The truth vs its decoys, on AUC_td's 95 % interval [lo, hi], in this order:**

| label | rule |
|---|---|
| TRUTH-ABOVE | lo > 0.505 (TRUTH-ABOVE-MATERIAL when also lo > 0.555) |
| DECOY-ABOVE | hi < 0.495 |
| TIED | hi < 0.545: no material truth advantage (≈ 0.5 within the interval) |
| INCONCLUSIVE | otherwise |

**(b) The decoys vs the honest worlds, on AUC_dh's interval:** POOL-LIFT if lo > 0.505; NO-POOL-LIFT if hi < 0.545;
otherwise INCONCLUSIVE.

**(c) THE VERDICT (pooled; the same rule per stratum, secondary), in this order:**

| verdict | rule | meaning |
|---|---|---|
| **UNDERPOWERED** | fewer than 60 states hold a decoy | no claim |
| **TRUTH-SPECIFIC** | (a) TRUTH-ABOVE and (b) NO-POOL-LIFT | pool familiarity does not explain L3: the likelihood picks the truth, not pool-ness |
| **TRUTH-PLUS-POOL** | (a) TRUTH-ABOVE and (b) POOL-LIFT | part of L3 is pool-ness and part is truth; the pool share (§4) says how much |
| TRUTH-ABOVE-POOL-UNCLEAR | (a) TRUTH-ABOVE and (b) INCONCLUSIVE | evidence beyond pool-ness; pool-ness not settled |
| **FAMILIARITY** | (a) TIED or DECOY-ABOVE, and (b) POOL-LIFT | **familiarity explains L3** on this subset: a pool decoy does as well as the truth |
| NO-SEPARATION-ON-SUBSET | (a) TIED / DECOY-ABOVE, (b) not POOL-LIFT, and AUC_th's lo ≤ 0.505 | L3 does not separate on this subset either; no conclusion about familiarity |
| INCONSISTENT | (a) TIED / DECOY-ABOVE, (b) not POOL-LIFT, AUC_th clearly > 0.5 | reported, investigated, no claim |
| NOT-DETECTED | (a) INCONCLUSIVE | inconclusive at this n |

**Registered consequences:**
- **FAMILIARITY:** L3's WORKS is withdrawn as evidence about hidden facts. Stage 1c's NEARER on a pool bank is then
  uninterpretable without the off-pool bank (the review's A4), and it must not unlock anything.
- **TRUTH-PLUS-POOL:** L3 stands, discounted by the pool share. The off-pool replication stays required.
- **TRUTH-SPECIFIC:** F2's familiarity mechanism is not supported on this bank. L3 stays PROVISIONAL-POOL until the
  off-pool bank and the plateau anchor (F1 and F5 are untouched by this test).

## 6. GIGO gates (a failure STOPS the read; reported, never patched)

- **G1.** The rows' ids are the 287 registered ids (sha256 `cf8d8c6f…dfb`).
- **G2.** At most 2 DEFECT rows: any likelihood guard (L3's G2) or the decoy consistency guard raising.
- **G3.** Determinism: 10 fixed states (salt `search_l3_decoy_recheck_v1:`), re-scored in a fresh process, reproduce
  the stored log L and conf exactly, decoys included.
- **G5.** Reproduction: on every state OK in both, the truth's and the 8 honest worlds' log L and conf equal L3's
  archived rows within **1e-4 nats** (the forward batch now holds the decoys too, so the last bits may move). More than
  2 failing or missing states STOP the read.

## 7. Hazards declared in advance (each reported as a FINDING with the read)

1. **The subset is selected.** Decoys exist only where some other pool team fits every reveal: mostly early states with
   few reveals, at about half the states in the smoke. The verdict is about that subset; AUC_th is re-read on it.
2. **Pool-ness vs real-team coherence.** A decoy is a coherent real team; an honest world is a draw from Smogon usage.
   A POOL-LIFT could come from coherence (the network has learned how real teams fit together) rather than from
   training familiarity. An off-pool real-team decoy (the ladder corpus) would separate the two. It is NOT part of this
   unit.
3. **The decoy's seen-mon facts are the pool team's own.** Its item, ability and spread must fit the log under some
   rolls. That differs from the honest worlds' screened spreads, so decoy vs honest differs in non-binding facts too
   (the review's F4).
4. **The near-copy rule drops the decoys most like the truth.** That is by design (§2), and it means a TIED verdict is
   about teams that differ from the truth on at least 3 of 6 sets.
5. **No validator on the decoys.** Pool teams were validated when the pool was packed, and the clause check runs. The
   truth's names, genders and levels are kept on the seen mons, as Tier 1's POOL builder keeps them.
6. **One model, one anchor, one pool bank** (X26 225M), as in L3.

## 8. Cost and run

About 10 to 20 CPU-s per state (the smoke), ≈ 1.5 CPU-h for the 287 plus G3. It runs in chunks of at most 45 minutes,
each in ONE heavy slot (rule 5b: `nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh --slot`), CPU only
(`CUDA_VISIBLE_DEVICES=""`). The run is [`run.sh`](run.sh). Rows go to `~/gen3ai_archive/search_l3_decoy/`. The read
writes `result_decoy.json` into this folder.
