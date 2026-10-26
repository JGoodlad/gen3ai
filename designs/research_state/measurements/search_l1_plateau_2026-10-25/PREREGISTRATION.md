# L1 REPLICATION on the maturity battery's plateau bank: REGISTRATION OF THE READ (2026-10-25)

**Written and committed BEFORE the replication is evaluated on the plateau bank.** The replication itself was registered
by L1's pre-registration ([`../search_l1_l5_2026-10-20/PREREG.md`](../search_l1_l5_2026-10-20/PREREG.md) §4.3: "PAYS-SCOPED
⇒ a registered replication on the next bank (the plateau bank, when its queue is done)"; ledger `search_ledger.md` §7
Q13). That text names the bank and the trigger but not the cell, the statistic or the pass bar, so this file writes
them down from L1's own rules (§4.1–§4.3) with no new choice of substance, and says where it reads the registration
literally (FINDING 1 of the README). Nothing below changes after the read; a deviation is a FINDING, never an edit here.
The script [`l1_plateau.py`](l1_plateau.py) is part of this registration.

**Touched before registration:** nothing of this bank under L1's rules. The plateau bank's own READ
(`../hidden_info_maturity_x26_2026-10-18/` README §7; amendment A1) was seen, which is G_F / TODAY / R3 under the
maturity rules and no anchored rule. L1's fitted values (`fit_l1.json`) and its held-out results were seen, as everyone has.

## 0. The question

L1 found that an acting rule anchored to the policy (piKL, Gumbel's scale, a deviation threshold) removes the plain argmax's
harm over 8 honest worlds but beats the policy on one cell only: **piKL at the 4-turn leaf (TRUNC-4-K8) on E2**, +0.012
[98.33 %: +0.001, +0.023], where the anchor's own gain over the argmax was +0.002 [−0.005, +0.008]. **Does that cell
replicate on a fresh bank, with the parameter frozen from Tier 1?**

## 1. Bank, rule, parameter (all frozen)

- **Bank:** the maturity battery's PLATEAU bank (`<archive>/plateau/`, drawn from P\*'s own newest cycles, 0 shared battles
  with the interim bank), at its two FULL anchors **25M and P\***, on the registered record's **719 common states**
  (truth ok, F ok, no root tie at both; ids sha256 `ffb3e84a…551ed` and table sha256 `cd443030…bd9f0e` must equal
  `result_mat_plateau.json`'s, else the read refuses) = **1,438 rows**, one bootstrap cluster per battle (464) across the
  two anchors, 2,000 draws, seed 20261011 (`t1_stats.Boot`, as L1's X26 read).
- **Rules and parameters:** L1's three rules and scorers exactly as in L1 §2–§3, at the values **FITTED ON TIER 1 and frozen
  in `fit_l1.json`** (sha256 recorded in the result); NOT refitted on this bank. The honest arm F only; the policy's top
  choice is the baseline; the metric is IMP = T(pick) − T(top1), T the true world's referee seeds 16..39.
- **Referee:** every cell is a **SELF-REFEREE read** (the truth T is played to the end by the anchor's own greedy policy on
  both sides). E1-0 (`../search_e1_0_2026-10-26/`, `015f00e2`) found the bot-stratum cells UNDECIDED, a gross
  self-referee misreading not ruled out, so the bot-stratum cells are labelled SELF-REFEREE wherever quoted.
- **GIGO gates (a failure stops the read):** G1 the recomputed ARGMAX IMP equals the plateau table's per-state F IMP at
  LEAF-K8, TRUNC-4-K8, CRITIC and the O row's LEAF-K8 (|Δ| ≤ 1e-9) on every state and anchor; G3 719 states per anchor;
  the two sha256 above; G2 (the policy order from the stored logits reproduces the registered top choice) is raised inside
  `t1_data.truth_arrays`.

## 2. The primary read and the decision rule

- **THE REGISTERED CELL: PIKL | TRUNC-4-K8 | plateau bank**, the pooled mean held-out IMP, 98.33 % battle-clustered
  interval (L1 §4.2's level, kept so the bar is the one that produced the cell; 95 % reported beside).
- **REPLICATES** iff that cell's lower bound > +0.0005 (L1 §4.3's clear-cell bar; a bound within 0.0005 of it is NOT
  clear, rule 8). **NOT REPLICATED** otherwise.
- The other five (rule × primary scorer) cells, the plain ARGMAX and the single-bank form of L1 §4.3 (PAYS: some rule
  clear at both primary scorers; PAYS-SCOPED: some cell clear; NULL: every upper bound < +0.0045; else NOT DETECTED)
  are reported as the same table, DESCRIPTIVE beside the registered cell. One bank cannot give L1's two-bank PAYS.
- **What a REPLICATION would and would not mean:** it would add a second bank on which the piKL anchor at the 4-turn leaf
  is above the policy. It would NOT make the anchor the source of the gain unless the anchor's gain over the plain argmax
  (IMP(piKL) − IMP(ARGMAX), 95 %) is also clear; the E2 cell was the leaf's own signal (+0.002 [−0.005, +0.008]).
  Neither outcome adopts an anchored rule as the honest default (a code change is a separate unit).

## 3. Secondaries (DESCRIPTIVE; no verdict)

By anchor, by opponent stratum (snapshot / bot; SELF-REFEREE), by reveal count k; the held-out grid curves per rule
(an upper envelope: choosing on it is fitting on the test set); anchored rollout minus anchored critic; the oracle
world's plain argmax IMP (context).

## 4. Compute

CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh --max-s 600`; read-only on
`models/` and the archive; the only new files are `result_l1_plateau.json` and this folder's.
