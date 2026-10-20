# L1 + L5: an ANCHORED acting rule for honest search, and Long et al.'s PIMC properties — PREREGISTRATION (2026-10-20)

**Written and committed BEFORE any anchored rule is evaluated on any bank.** Spec: the literature review
[`hidden_info_search_review_2026-10-20.md`](../../literature/hidden_info_search_review_2026-10-20.md) §10 (L1, L5) and
[`search_ledger.md`](../../../endstate/search_ledger.md) §1 / §7 Q13. Nothing below changes after the read; a deviation
is reported as a FINDING in the README, never edited here. The scripts in [`scripts/`](scripts/) are part of this
registration (their sha256 are in [`l1l5_registration.json`](l1l5_registration.json)).

**What was touched before registration (no outcome of any anchored rule):** the archives' formats were inspected; the
loader (`scripts/l1l5_data.py`) was written against them and its cache BUILT (sha256 `9f63442d…80c4`): G1–G3 below
PASS (0 ARGMAX mismatches on 1,477 Tier-1 and 1,148 X26 rows; E2's pooled ARGMAX means and n equal the registered ones);
the rules' selfcheck ran on synthetic states. The registered ARGMAX IMPs (already published by Tier 1, E2 and the maturity battery)
were seen before, as everyone has.

Abbreviations: IMP = decision improvement, T(the rule's pick) − T(the policy's top choice), T the mean outcome over the
TRUE world's 24 referee seeds (16..39), which no scorer reads; F = the fully honest world arm (every hidden fact drawn
from Smogon usage and kept only if it replays what our side saw); O = the true world; FN = the network's own belief
worlds; CI = a battle-clustered percentile bootstrap interval (2,000 draws, seed 20261011, `t1_stats.Boot`).

## 0. The questions

**L1.** Honest search's plain argmax over 8 honest worlds does not beat the policy (Tier 1 IMP_F +0.006
[−0.004, +0.017]; X26 IMP_F ≤ 0 at every anchor). The closest precedent (Ataraxos, Stratego) played 362 Elo BELOW its own
policy without a KL anchor to it, and +123 above it with one. **Does an acting rule ANCHORED to the policy make honest
search pay, on data the anchor's one parameter was not fitted on?** PAYS ⇒ an anchored rule becomes the honest search's
default, and every honest "search does not beat the policy" read is re-stated as "the argmax does not". NULL ⇒ the honest
leaf carries no decision information beyond the policy: the sampler binds, not the acting rule.

**L5 (descriptive only).** Where does Gen 3 OU sit on Long, Sturtevant, Buro & Furtak's (AAAI 2010) map of when
perfect-information Monte Carlo works: leaf correlation, bias, disambiguation factor?

## 1. Banks (all BANKED; zero playouts; read-only)

| role | bank | policy | states (F ok) | strata of the bootstrap |
|---|---|---|---|---|
| **FIT** | Tier 1 (`hidden_info_tier1_2026-10-13`; registered 600 + supplement per checkpoint) | end2 + prod at 15M | 728 + 749 = **1,477** | battles within checkpoint |
| **HELD OUT 1** | E2, competent opponents (`hidden_info_e2_competent_2026-10-16`) | end2 + prod at 15M (fresh states) | **1,108** (registered n) | battles within checkpoint |
| **HELD OUT 2** | X26 maturity, interim bank, LEAN (`hidden_info_maturity_x26_2026-10-18`) | X26 at 25M / 100M / 175M / 225M | **287** common states × 4 anchors = **1,148** rows | battles (one cluster per battle across the four anchors) |

- Inclusion is each bank's REGISTERED one: Tier 1 = truth ok and F ok in the frozen blinded table (sha256 committed
  beside it); E2 = `read_e2.records`' rules (truth ok, F's world ok, F's row ok); X26 = the registered read's common states
  (truth ok, F ok, no root tie at every anchor; the ids' sha256 and the table's must equal `result_mat.json`'s, else the
  loader refuses).
- The root policy π is the masked softmax of the stored root logits over the legal actions (Tier 1 / E2: the states
  files; X26: each anchor's own `roots/<anchor>.jsonl`). The X26 rows are read from the archive of a RUNNING queue (the
  plateau step); only LEAN ids at the four finished anchors are read, and every state's ARGMAX IMP must equal the frozen
  LEAN table's (G1), so a row the queue re-wrote would be caught.
- **GIGO gates (G1–G3; a failure stops the read):** G1 the recomputed ARGMAX IMP equals the registered per-state IMP at
  LEAF-K8, TRUNC-4-K8 and CRITIC on every Tier-1 and X26 state (|Δ| ≤ 1e-9), and E2's pooled mean and n equal
  `result_e2.json`'s (|Δmean| ≤ 5e-6, the file's rounding); G2 the policy order from the stored logits reproduces every
  state's registered top choice (`t1_data.truth_arrays` raises otherwise); G3 the counts above.

## 2. The scorers and the metric

Q(a) per legal action from the 8 honest (F) worlds: **LEAF-K8** (the mean of the 8 playout outcomes) and **TRUNC-4-K8**
(the mean of the 8 critic values 4 turns on, the terminal where the game ended first) are PRIMARY (the review's rule
requires both); **CRITIC** (the one-ply critic over the 8 worlds) is secondary. The metric is each bank's honest-search
gain metric unchanged: IMP as defined above, the policy's top choice the baseline, every state weighted equally
(pooled over the checkpoints / anchors of a bank).

## 3. The rules and their ONE parameter each (fitted on Tier 1)

Every rule picks among the legal actions in the POLICY's order; a tie within 1e-6 goes to the earliest (the registered
rule). `log π` is the masked log-softmax.

| rule | pick | parameter grid (least → most anchored) | limits |
|---|---|---|---|
| ARGMAX (registered) | argmax Q | — | — |
| **PIKL(λ)** | argmax Q(a) + λ·log π(a), the MODE of π′ ∝ π·exp(Q/λ) | 0, .001, .0025, .005, .0075, .01, .015, .02, .03, .04, .05, .075, .1, .15, .2, .3, .5, 1, ∞ | λ = 0 ARGMAX; ∞ the policy |
| **GUMBEL(c)** | argmax logit(a) + (50 + 8)·c·q̃(a), q̃ = Q min-max rescaled (`main.value_search.rules`, no noise) | 5, 2, 1, .5, .2, .1, .05, .02, .01, .005, .002, .001, .0005, 0 | c = 1 the module default; 0 the policy |
| **THRESH(δ)** | ARGMAX's pick a\* only if Q(a\*) − Q(top1) > δ (+1e-9), else top1 (SPARTA-style deviation) | 0, .005, .01, .02, 1/32, 3/32, 5/32, 7/32, 9/32, 11/32, 13/32, 15/32, 19/32, 23/32, ∞ | δ = 0 ARGMAX; ∞ the policy |

THRESH's grid sits on odd multiples of 1/32, between the values a LEAF-K8 margin can take (multiples of 1/16), so no
margin lies on a threshold (rule 8).

**The fit (§3.1).** For each rule and each scorer separately: the grid value that MAXIMISES the Tier-1 pooled mean IMP of
arm F. A tie within 1e-12 goes to the MORE anchored value. The fitted values are frozen, written to `result_l1.json`
before any held-out number is computed in the same run, and applied unchanged to both held-out banks.

## 4. The PRIMARY read and its decision rule

4.1 For each rule R and held-out bank B ∈ {E2, X26}: the pooled mean held-out IMP at the fitted value, at LEAF-K8 and at
TRUNC-4-K8.

4.2 Intervals for the verdict: **98.33 %** (Bonferroni over the three rules; 95 % reported beside). Rule 8: a bound
within 0.0005 of its boundary is not clear (the zero edge is written "> +0.0005", as in the study).

4.3 The verdict, evaluated in this order:
- **PAYS** iff some rule R has, on BOTH held-out banks, lower bound > +0.0005 at BOTH LEAF-K8 and TRUNC-4-K8.
- **PAYS-SCOPED** iff not PAYS, but some (rule, bank, primary scorer) cell has lower bound > +0.0005 (the cells are
  named; read as a hint to replicate, not as the default).
- **NULL** iff every (rule, bank, primary scorer) cell has upper bound < +0.005 (clear: < +0.0045): whatever an anchor
  buys is bounded below half a point of win probability per decision.
- **NOT DETECTED** otherwise.

Decision: PAYS ⇒ propose the paying rule (at its fitted value) as the honest search's default (a code change is a
separate unit); PAYS-SCOPED ⇒ a registered replication on the next bank (the plateau bank, when its queue is done);
NULL ⇒ the acting rule is not what binds honest search, the sampler is; NOT DETECTED ⇒ the banks cannot tell.

## 5. Secondaries (DESCRIPTIVE; no verdict)

S1 per checkpoint / anchor; S2 each rule minus ARGMAX (held out); S3 the whole held-out grid curve per rule (an UPPER
ENVELOPE: choosing on it is fitting on the test set); S4 anchored rollout search minus anchored critic search (the CRITIC
scorer fitted the same way); S5 by reveal-count k and opponent class; S6 the SAMPLED piKL policy's expected IMP (λ fitted
on Tier 1 the same way) and the sampled policy's own; S7 the ORACLE world (O, leaf seeds 0..7): the rule fitted on
Tier 1's O rows (the review predicts a small λ in the oracle world, consistent with N5) and read held out; S8 the
network-belief worlds FN (fitted on Tier 1's FN, read on X26's FN).

## 6. L5, Long et al.'s properties (DESCRIPTIVE; `scripts/l5_props.py`)

Definitions VERIFIED in the paper (pp. 136, 139): leaf correlation = the share of PRE-TERMINAL nodes (every move leads
directly to a terminal) whose move values are all the same; bias = the share of those correlated nodes won by a fixed
player; disambiguation = the change in the number of consistent worlds since the player last moved (in the synthetic
model each information set is halved with probability df at each of the player's moves, recursively).

Our analogues (each a PROXY, said so in the read):
- **lc, b** on the TRUTH rows of all three banks (every legal action played out in the true world on 40 seeds, the
  network on both sides): a state is pre-terminal when every playout of every action ends in the decision's own turn;
  correlated when every (action, seed) outcome is the same; lc = correlated / pre-terminal; b = the correlated states won
  by the searched side / correlated. Also lc_seed (per seed: are all actions' outcomes the same) and the same two
  quantities for EVERY state by the median turns its playouts take to end (0, 1, 2–3, 4–7, ≥ 8).
- **df** on the X26 interim bank's 513 battles, from the traced side's own protocol view (the copied eval traces): per
  own turn t with true opponent facts still hidden, df_event = P(≥ 1 new true fact revealed during turn t), which in
  Long's generative model is exactly the probability of at least one halving per move; and the revealed FRACTION. Facts:
  species + the true sets' moves (primary); + items and abilities (secondary).

Long et al. place Skat and Hearts at correlation 0.8 to ~1.0 and df ≈ 0.6. L5 places us beside them; no rule reads it.

## 7. Compute and hazards known in advance

- CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19 taskset -c 5-7,13-15`. MEASURED light (the cache build: 1.3 GB peak,
  31 s, one core), so it runs under an address-space cap (`ulimit -v 8000000`) rather than queueing for a heavy slot
  behind 13 hour-long jobs (a FINDING in the read). The cache (`~/gen3ai_archive/search_l1_l5/`) holds per-state arrays
  only.
- The Tier-1 fit and the E2 read share the two 15M policies (E2's states are fresh). X26 is a different run at four
  maturities, so it is also held out across POLICIES.
- The X26 rows are four anchors of the SAME 287 states: one bootstrap cluster per battle carries their dependence.
- L5's banks were sampled with quotas (by reveal count, after a tie exclusion), not as Long's random walk to the leaves;
  its pre-terminal states are the bank states that happen to sit at the end of a game.
