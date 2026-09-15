# PREDICTION — the CHEAPEST route to SIBLING DISCRIMINATION in the win-prob head

**Registered 2026-09-14, before any number in this directory exists.** Nothing below may be
edited after the first result lands; the README scores it verbatim.

The owner's question: *what is the cheapest way to give the win-prob value head SIBLING
DISCRIMINATION — the ability to rank two successor states one move apart?* That is the property
the one-ply search test consumes and every head measured so far lacks: the mirror battery's
separation-of-raced reads **11.9–15.0 %** on six win-prob heads against the shaped critic's
**45.4 %** at the identical 3 s operating point
([`search_dividend_winprob_heads_2026-09-11`](../search_dividend_winprob_heads_2026-09-11/README.md)),
and no trained lever in the critic ladder has moved either leaf row.

Three parts. **A** reads three arms nobody has put on the leaf instrument. **B** is the
experiment: a PAIRED SIBLING dataset built by forking a frozen checkpoint's recorded decisions,
and a head refit with a pairwise RANKING term on the frozen trunk. **C** puts the refit head back
on the leaf instrument.

---

## 0. The frame, fixed in advance

| | |
|---|---|
| frozen checkpoint (B, C, and A's anchor) | `models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip` — step **10,000,032**, `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `win_prob_mode=shaping` |
| the head being refit | `WinProbHead` = LayerNorm(128) → Linear(128,128) → ReLU → Linear(128,1) off `stash.value_pooled`. **The trunk is FROZEN**: only these four tensors move. |
| Part A arms (each @10000032) | `ai_v12_23_ladder_rollout` · `ai_v12_28_ladder_ent05` · `ai_v12_29_ladder_vf025` |
| Part A anchor | `ai_v12_11_ladder_ctrl10M` @10000032, **in the same window** (rule 23) |
| regime | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice`, rust sim bridge, `models/` read-only, nothing written under `models/` |

🚨 **The task named `ai_v12_22_ladder_rollout`. No such run exists** — `models/ai_v12_22_*` is
`ai_v12_22_ladder_lambda095` and the rollout-target arm is **`ai_v12_23_ladder_rollout`**. The
rollout arm read here is 23; recorded as a finding, not silently substituted.

---

## 1. PART A — the mirror battery on three unread arms

**Cells**, exactly the registered operating point of the 2026-09-11 battery so every number is
comparable to the six heads already on it:

```
--arm honest --budget 1 --opponents self --games-seed 7 --side-swap (auto)
rung B: --root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 \
        --defensive-confirm 0 --defensive-contested-deadline-s 3.0
grid  : --root-strategy grid
```

100 pairs per cell per head is a SCREEN, not a resolving width (the 2026-09-11 battery needed 800
pairs to resolve ±0.015 on L2). Every head plays the SAME game indices, so every head-to-head
contrast is paired at the battle level.

### Readings, registered

| row | what is reported | the bar |
|---|---|---|
| **L2** — paired mirror win rate, rung B | Wilson CI over pairs. Null is **0.50 by construction** (the unsearched side plays the same network with search structurally off). | **A DIVIDEND** = Wilson lower bound > 0.50. At 100 pairs the half-width is ≈ ±0.10, so anything short of ~0.60 cannot clear it — this cell is registered as a SCREEN that can only find a LARGE effect or fail to. |
| **`grid`** — unguarded paired mirror win rate | the sensitive leaf row (rule 23 / §11.3 of the battery) | reported against the six heads already measured (0.1875–0.3150) and against its OWN contemporaneous `ctrl10M` anchor, paired on shared indices. |
| **L1** — separation-of-raced | pooled AND width-matched by realized K (`l1_width_matched.py`) | **NO FIXED BAR** (rule 23). Read only against the contemporaneous anchor inside a matched K band. A pooled L1 quoted without its K is not a reading. |
| overrule rate, forced fraction, action-changed | descriptive | — |

### Registered predictions (A)

* **A1.** None of the three arms clears the L2 bar. All three land in **[0.45, 0.55]**.
  *(Six heads are on the structural null; none of these three levers touches successor
  discrimination. `vf025` is a value-coefficient dose, `ent05` an entropy-floor lever, `rollout`
  a target-form lever — the closest of the three to a discrimination objective.)*
* **A2.** `rollout` is the best `grid` cell of the three (it is the only one whose target is a
  bootstrapped successor value rather than the terminal label), and lands in **0.25–0.38**.
* **A3.** No arm's `grid` differs from its contemporaneous `ctrl10M` anchor with a paired CI clear
  of zero at 100 pairs.
* **A4.** L1 pooled for every arm lands within the anchor's own width band once matched, i.e.
  **no head-level L1 difference survives width matching** — L1 reproduces as a width meter.

**BRANCH 4** (of the four the task names) fires iff any arm clears the L2 bar or beats its
contemporaneous anchor's `grid` with a paired CI clear of zero: *an existing arm already pays*.

---

## 2. PART B — the paired sibling refit

### 2.1 The dataset

Source: `ai_v12_11_ladder_ctrl10M` @10000032's **offline full-capture eval traces**
(`/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M/`, which carries
per-battle `*_reconstruction.json` — the resolved seed, both packed teams and every committed
choice — and `*_states.npz`). More are generated with `main.ops.eval_trace_gen` only if needed.

**CONTESTED-state selection, declared here:** a recorded decision qualifies iff it is a MOVE round
(not a forced switch), turn ≥ 2, has **≥ 3 legal actions**, and the policy's **top-2 masked logit
gap is below the 40th percentile** of that quantity over all qualifying decisions in the frame.
The realized selection rate is reported. `|V − 0.5|` is recorded per decision but is **not** the
selector — it selects on the head being refit, which would make the held-out read partly a
measurement of the selector.

**The fork, per selected decision — THREE branches** (the owner's 2026-09-14 amendment, preferred
over the two two-branch designs and run in its place if budget forces one):

1. **`top1`** — the policy's argmax action (what was recorded, where it matches).
2. **`top2`** — the policy's runner-up action.
3. **`rand`** — one uniformly-random legal alternative, drawn from the legal set minus {top1, top2}
   with a decision-keyed RNG (so it is reproducible and independent of the policy's ordering).

All three replay the recorded prefix to the divergence turn and then continue **live to a
terminal** with the SAME frozen checkpoint on our side and the RELOADED real opponent, under
**COMMON RANDOM NUMBERS**: the branches share the record's own resolved `>start` seed and no
`post_t_seed` reseed, so they draw from ONE dice stream that diverges only because the action did.
Continuation policy: **greedy (argmax)** on both sides, stated here so the residual torch-RNG
variance that `cf_q_labels` records for its temperature-1.0 rollouts **does not exist** in this
dataset — greedy continuations make the CRN pairing exact in both the dice and the policy draws.
Verification, run before the dataset is accepted: a **prefix replay check** — for a sample of
forks, the three branches' protocol streams are identical up to the divergence turn, and a
`divergence_turn=None` full replay reproduces the recorded winner.

Each branch records: terminal outcome (win=1 / loss=0; a **draw at the stall-forfeit cap is
EXCLUDED**, never scored 0.5 — it is decided by seat), the SUCCESSOR state's `value_pooled` [128]
and the original head's V at that successor, plus turn, opponent, battle and team ids.

**Target ~20,000 forks (≈60,000 rollouts).** The box carries a live training arm; the realized
count is reported with the load it was measured under, and a scale-down is stated, never hidden.

### 2.2 The refits

80/20 split **by fork** (no fork straddles), grouped so that a battle's forks stay on one side.
Both refits: the **FROZEN trunk**, only `WinProbHead`'s four tensors trained, Adam lr 1e-3,
batch 1024, budgeted in STEPS not epochs (the N-curve's rule), early stopping on a grouped
validation split of the training forks.

* **(i) CONTROL refit** — ordinary BCE on each branch's own outcome. Every branch is one row.
* **(ii) RANKING refit** — the same BCE **plus** a pairwise term over the branch pairs within a
  fork: `−log σ(logit_a − logit_b)` where branch *a* won and *b* lost, **zero weight on tied
  pairs** (both won or both lost). Coefficient swept over **{0.1, 0.3, 1.0}**; the swept value is
  chosen on the VALIDATION split and the held-out read is taken once, on the test forks.

### 2.3 The readings, registered

| meter | definition |
|---|---|
| **PAIRWISE ACCURACY** (the headline) | over held-out **non-tied** branch pairs: the fraction where `sign(V_a − V_b)` agrees with `sign(outcome_a − outcome_b)`. Ties in V count as ½. Bootstrap CI **over FORKS** (never over pairs — pairs inside a fork share a prefix). |
| weighted variant | the same, weighted by `|Δoutcome|` (which is 1 for every non-tied pair here, so it is reported and expected to coincide — a guard against a coding error, not a second statistic) |
| **SEPARATION** | mean `|V_a − V_b|` on non-tied pairs vs on tied pairs, and their ratio. A head that ranks by widening every gap is separating on nothing. |
| **CALIBRATION** | ECE (15 equal-mass bins) and Brier on held-out branch rows. The guard: **ranking must not be bought with calibration.** |
| **CONDITIONING GUARD** | `cond.opp_class_auc.t4_10` on the refit head — `main.ops.critic_read` if it accepts an external head, else the N-curve's `frame_check` decode applied to the refit head's outputs on the same frame. |
| **blind-spot rate** | how often the `rand` branch beats BOTH policy candidates (outcome 1 where both top1 and top2 are 0) — a property of the POLICY, reported whatever the refit does. |

### Registered predictions (B)

* **B0 — the baseline, and a finding on its own.** The ORIGINAL online head's held-out pairwise
  accuracy is **0.52–0.58**. *(It is trained on a terminal label with no sibling term at all; it
  is not at chance because a genuinely losing successor is often visibly losing.)* **A value below
  0.52 with its CI excluding 0.55 would itself be the headline: the promoted critic cannot rank
  siblings at all.**
* **B1.** The CONTROL refit (i) does **not** beat the original head by more than the bootstrap CI
  — the same data, the same loss, no new information. `|Δ| < 0.02`, CI covering zero.
* **B2.** The RANKING refit (ii) **beats** (i) and the original head, by **+0.03 to +0.10**, with
  the delta's own CI clear of zero at the best swept coefficient. *(This is the "the trunk already
  holds it, the LOSS is the fix" prediction; it is the one this measurement exists to test.)*
* **B3.** Separation rises with the ranking term: the non-tied/tied `|ΔV|` ratio goes from
  **≈ 1.0–1.2** on the original head to **≥ 1.5** on (ii).
* **B4.** ECE of (ii) at coefficient 1.0 is **worse** than (i) by 0.01–0.04 (ranking is bought
  partly with calibration at the high coefficient), and at 0.1–0.3 it is within 0.01.
* **B5.** The conditioning guard `cond.opp_class_auc.t4_10` on (ii) is **not detected below**
  the original head's value (which reads ~0.72–0.73 on the N-curve's frame).
* **B6.** The blind-spot rate — `rand` beats both policy candidates — is **8–20 %** of non-tied
  forks. *(A uniformly random legal action is usually bad; a rate above ~25 % would say the
  policy's own top-2 is not where the value is.)*

---

## 3. PART C — the refit head on the leaf instrument

The best (ii) head is loaded into the mirror battery **as the leaf only**: the win head is never
concatenated into pi/vf (it is a leak-safe SIDE readout), so replacing its weights changes the
SEARCH LEAF and nothing about how either side plays unsearched. Mechanism: a new
`--leaf-head <path>` on `python -m main.search_dividend` that overwrites
`policy.features_extractor.win_head`'s `state_dict` after load, refusing on a shape or key
mismatch, with a unit test. **The ORIGINAL head runs as the contemporaneous control in the same
window, on the same game indices**, so the contrast is paired and width-matched.

Cells: rung B (the registered defensive operating point) and `grid`, 100–200 pairs.

| bar | |
|---|---|
| **A DIVIDEND** | rung-B L2's Wilson **lower bound > 0.50** |
| the separation row | separation-of-raced vs the control **at matched K** (`l1_width_matched.py`), never pooled |
| the `grid` row | paired against the control on shared indices |

### Registered predictions (C)

* **C1.** The refit head's rung-B L2 does **not** clear 0.50 at 100–200 pairs
  (**0.47–0.55**). *(Six heads on the null; one refit head's ranking gain has to be large to show
  through a gate that refuses to search 75–80 % of decisions and overrules ~1 %.)*
* **C2.** The refit head's **`grid`** cell beats the control's `grid` by **+0.03 to +0.12**,
  and this is the row where a real ranking gain shows first, because `grid` has no gate to hide
  behind. NOT DETECTED at 100 pairs is the likely outcome even if the point estimate moves.
* **C3.** Separation-of-raced at matched K rises against the control by a factor **1.2–2.0** —
  smaller than the 3–4× that would be needed to reach the shaped critic's 45 %.

---

## 4. The four branches, named in advance

| | fires when | what it means | what follows |
|---|---|---|---|
| **1 — the LOSS is the fix** | B2 holds AND (C1 clears **or** C2 detected) | the frozen trunk already carries sibling information and the terminal BCE simply does not extract it | a cheap training arm: the pairwise term added to the win-prob head's loss on labels the cf machinery already produces |
| **2 — COVERAGE, not loss** | B2 holds AND C shows nothing | ranking learned on RECORDED states does not transfer to the states search actually queries (search queries successors of *every* candidate, most of which the policy never visits) | forks must go INTO training — the label factory feeds the head the states it will be consulted on |
| **3 — the TRUNK lacks it** | B2 fails (the ranking refit does not beat the control by more than its CI) | `value_pooled` does not linearly-or-nearly separate siblings; no head-side loss can recover it | the lever is the TRUNK/representation (or the search leaf is the wrong consumer), and the cheap-fix question is answered NO |
| **4 — an arm already pays** | Part A finds an arm clearing L2 or beating its anchor's `grid` with a CI clear of zero | the property exists in a trained arm nobody read | read that arm's lever first; B/C become the mechanism study |

Branches 1 and 2 are not exclusive of 4; if 4 fires it is reported first.

## 5. Hazards that would VOID a row, declared now

1. **CRN not actually shared.** If the prefix-replay check fails on any sampled fork, the paired
   design is void and the dataset is REFUSED, not reported with a caveat.
2. **Selection on the head.** The contested filter uses the POLICY's logit gap, not V. If it is
   ever changed to a V-based rule the held-out read is void.
3. **A draw at the stall cap is not an outcome** (rule 12); excluded and counted.
4. **L1 without its K is not a reading** (rule 23). Every L1 in this directory carries its
   realized K and a contemporaneous control.
5. **The box carries a live training arm.** Every wall-clock-budgeted number here (L1, realized K,
   arms scored) is contention-coupled; the load average at the start and end of each cell is
   recorded.
6. **100 pairs resolves ±0.10.** Every Part A and Part C cell is a SCREEN. "NOT DETECTED" at this
   width is not "no effect"; the registered language is *not detected at ±0.10*.
7. **One checkpoint, one seed.** Part B is a single-arm study on one frozen checkpoint (rule 2 /
   rule 22): a positive B2 is a CANDIDATE, not a family claim, until it replicates on a second
   checkpoint.
