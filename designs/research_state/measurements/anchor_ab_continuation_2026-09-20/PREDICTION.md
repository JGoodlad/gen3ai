# PRE-REGISTRATION — CAN THE CONTINUATION'S GAIN BE SEEN EXTERNALLY?

**Committed before the first battle.** Owner's question, verbatim: *"Can you get the A/B
continuation against metamon teams to see if we can see the improvement externally."*

The continuation control `ai_v13_09_wcont` (arm W + 12M steps, **no teachers**, frozen dose) read
**+15.50 pp** over its frozen parent on the untaught-8 meter and **+0.140 / +0.210** over the parent
/ the fold path on a **100-game** `SmallRL` greedy-away cell
([`wcont_control_read_2026-09-20/`](../wcont_control_read_2026-09-20/README.md) §4). Both anchor
contrasts were **WITHIN the 0.090 run-level floor at n = 100** — the cell resolves ~±0.10 and cannot
separate the arms whatever they did. **This read scales that cell to n = 400 per arm per team set so
a +0.10 difference is resolvable**, and adds the `home` team set, which the n = 100 read never took.

---

## 1. THE ARMS — resolved paths and steps, fixed here

| arm | label | `.zip` (explicit, never a bare run dir) | step |
|---|---|---|---:|
| **W** | **arm W** — the FROZEN PARENT, the 75M win-prob flywheel arm | `models/ai_v13_02_flywheel_winprob/final_model.zip` | **75,005,952** |
| **C** | **the CONTINUATION** `ai_v13_09_wcont` — arm W + 12M, NO teachers | `models/ai_v13_09_wcont/final_model.zip` | **87,097,344** |
| **F** | **the FOLD PATH** `ai_v13_08_fold1_cont` — era-1 fold, same depth/dose/endpoint | `models/ai_v13_08_fold1_cont/final_model.zip` | **87,097,344** |

`num_timesteps` read from each zip's own SB3 `data` blob. **C and F end on the same step, to the
step**, which is what makes the C − F contrast a matched one.

## 2. THE CELLS

`python -m main.anchors` — the **tool of record**, per
[`designs/ops/EXTERNAL_ANCHORS_SOP.md`](../../../ops/EXTERNAL_ANCHORS_SOP.md). Greedy-vs-greedy
(SOP rule 2), verified **per decision on both sides**. CPU only (`CUDA_VISIBLE_DEVICES=""`),
`OMP_NUM_THREADS=1` (hazard H13), `nice 15`, `--out` always explicit (never the cwd), own Showdown
server on a **9521–9540** port, stopped by its own PID. A GPU training arm owns the box.

| # | opponent | regime | team set | games/arm | arms |
|---|---|---|---|---:|---|
| 1 | `metamon:SmallRL` | greedy | **away** (Metamon's own 20 teams — the owner's "metamon teams") | **400** | W, C, F |
| 2 | `metamon:SmallRL` | greedy | **home** (our 719-team pool) | **400** | W, C, F |
| 3 | `metamon:SyntheticRLV2` | greedy | away | **200** | W, C, F | *(only if wall clock allows after 1 and 2)* |

### 2.1 Sub-cells and the SEEDS — 🚨 DIFFERENT from the 2026-09-20 read

The 2026-09-20 read ran at the tool's **default `--team-seed 20260914`**. Every cell here is
**four 100-game sub-cells** at

> **`--team-seed` ∈ {20260919, 20260929, 20260939, 20260949}**

(the `SyntheticRLV2` cell: the first two only). Spacing is 10 because the peer draws at `seed + 1`;
no our-seed collides with a their-seed. Four reasons for the split, all registered in advance:

1. **Hazard H5 is a 0.5 %/game event that kills a whole invocation.** A forfeit at the 250-turn cap
   can crash the Metamon peer (`RecursionError`) *after* its last decision. A 400-game monolith
   loses everything past the crash; four sub-cells lose at most 100 games, and the harness resumes.
2. **The same four seeds are used for all three arms**, so the three arms face the **same four
   draws** — a matched-draw design. The registered primary CI is nonetheless the **unpaired**
   Newcombe interval (conservative); a paired game-index contrast is reported only as a
   **descriptor**, and only after the game-index alignment is verified post hoc.
3. It bounds the team-draw component instead of hiding it: the four sub-cell rates per arm are
   reported.
4. SOP amendment 2026-09-16 (batched anchor runs killed by a session harness's memory accounting)
   is respected: **at most two cells in flight**, each a detached single-cell subtree.

## 3. THE CONTRASTS AND THE BAR

Three **paired contrasts per cell**, each on win rate (ours; ties count in the denominator only):

| contrast | meaning | sign convention |
|---|---|---|
| **C − W** | 🚨 THE OWNER'S QUESTION — is the continuation's gain visible externally? | + = continuation ahead of its frozen parent |
| **C − F** | continuation vs the era-1 fold at the same depth | + = the plain continuation beats the fold path |
| **F − W** | did the fold path move off its parent at all? | + = fold path ahead of the parent |

### 🚨 THE BAR, fixed here

**PRIMARY (the owner's bar):** the **95 % CI on the DIFFERENCE excludes zero** — Newcombe's
hybrid-score interval on the two pooled proportions, cross-checked by a 20,000-draw paired
bootstrap over sub-cells. **A CI that straddles zero is NOT DETECTED — never "equivalent",
never "no difference"** (rule 6; memory: *equivalence needs the DELTA's own CI inside the bar*).

**SECONDARY, stricter, reported beside it:** the difference also clears the **0.090 run-level
floor** on this cell (`|arm W − W_b|`, `flywheel_wb_floor_read_2026-09-18` §6.1) by BOTH clauses —
(a) `|Δ| > 0.090` **and** (b) the Δ's CI excludes the floor POINT. ⚠️ **That floor is a
BETWEEN-SEED quantity** (two independent re-runs of the same recipe) and C − W is a WITHIN-LINEAGE
contrast (a parent against its own continuation), so the floor is the wrong null for it in kind; it
is reported because no better floor has been measured, and clearing it is strictly harder.

**Power at n = 400.** se(p̂) ≈ 0.025 at p ≈ 0.5–0.65, so se(Δ) ≈ 0.035 and the 95 % half-width is
**≈ 0.069**. A true Δ = +0.10 gives z ≈ 2.9 (power ≈ 0.82); a true Δ = +0.14 gives z ≈ 4.1
(power ≈ 0.98). **n = 400 resolves +0.10, which is what was asked for; it does NOT resolve +0.05.**

## 4. THE PRIOR, and what I expect

From the n = 100 away read (point estimates, all WITHIN floor there): **C − W = +0.140**,
**C − F = +0.210**, **F − W = −0.070**. Arm W's own banked away rate was **0.500**, C's **0.640**,
F's **0.430**. On `home`, the only banked number is arm W's era-mate `ai_v12_02` at **0.520**; the
SOP's pooled home−away effect for `SmallRL` was **−0.030 [−0.125, +0.066]**, NOT DETECTED, so I
carry no home/away prior and expect the two team sets to agree in sign.

**Filed probabilities, before the first game:**

| branch | statement | P |
|---|---|---:|
| **(a)** | **C − W detected (CI excludes 0) on BOTH team sets** | **0.55** |
| **(b)** | detected on exactly ONE team set | **0.30** |
| **(c)** | **NOT DETECTED on either at n = 400** | **0.15** |

Also filed: **P(C − F detected on away) = 0.75** (the larger contrast); **P(F − W detected on
away) = 0.20**; **P(the C − W point estimate on away lands in [+0.05, +0.25]) = 0.70**.

### What each branch would MEAN — written before the data

* **(a)** The +15.50 pp the untaught meter saw **transfers to an opponent nobody here trained**, on
  both our pool and Metamon's own teams. That makes the continuation's gain a competence gain and
  not a meter artefact, and it makes the external anchor a live instrument for this lever — the
  first time an in-run competence movement of this era has been confirmed off-distribution.
* **(b)** The gain is **team-set conditional**. If it appears on `away` only, the continuation
  improved on ground we do not train on (the stronger claim, and the one the owner asked for); if
  on `home` only, it improved inside our own pool's distribution and the off-distribution claim is
  unsupported. Either way the single-cell result is reported as such and NOT pooled into a headline.
* **(c)** 🚨 **The internal +15.50 pp did NOT transfer to an external opponent at a resolvable
  size.** That is a finding about the METERS, not only about the arm: the untaught meter, the Big-5
  and DDTar slices and the in-run promotion row would all be moving together on an axis that a
  third-party agent cannot see. It would NOT license "the continuation did not improve" — `SmallRL`
  is one opponent at one skill level and a ceiling/floor effect against it is a live alternative —
  but it would demote every internal number of this era from "competence" to "competence **against
  our own lineage**" until an external read says otherwise, and it would put the burden on the next
  era gate (`SyntheticRLV2`, which is AHEAD of us) rather than on `SmallRL`.

## 5. INTEGRITY — recorded per cell, before any win rate is read

1. **`status == "OK"`.** A `FAILED` sub-cell's games are reported separately and **excluded from
   the pooled n**; the pooled n and the intended n are both printed. A cell that loses more than one
   of its four sub-cells is reported as **PARTIAL**, with its reduced n and widened CI.
2. **`regime_verified` per cell — BOTH the composite flag AND the per-decision rates.** The
   2026-09-20 read carried a **FALSE composite flag with per-decision `argmax_match_rate` = 1.0000
   on both halves** (hazard W-J: a post-game `RecursionError` in the challenger half's peer marks
   `regime_check_ok` false). Both are recorded for every sub-cell and the distinction is never
   collapsed. A composite-false cell with any per-decision rate **below 1.0000** in a greedy cell is
   a REGIME FAILURE and its games are void.
3. **Forfeit-at-turn-limit counts per cell** (`hit_forfeit_limit`), and the timeout/stall rate.
   🚨 **A cell with > 25 % timeouts is INCONCLUSIVE, not a result** (rule 12).
4. `team_source_asymmetry == false`; `distinct_our_teams`; `model_step`, `model_rung`
   (`explicit_zip` expected on all three arms) and `model_loader` on every row.
5. Showdown pin `e0551883f`, Metamon `@0a00a759`, `SmallRL@ckpt40` — stamped per row by the tool.

## 6. WHAT THIS READ CANNOT SAY

SOP §6 applies unchanged, plus two specific to this design: it cannot attribute the C − F gap to any
single lever (the fold path differs from the control in the distill LOSS, in `--distill-team-bias
0.4` **and** in its opponent ecology — three levers, one contrast), and it cannot turn a `SmallRL`
result into a statement about strength in general.

## 7. WHAT WILL NOT BE TOUCHED

`ledger.md`, `UNDERSTANDING.md` and every design note are **read-only** for this job. The
deliverable is this directory: `PREDICTION.md`, `README.md`, the per-cell `main.anchors` output
dirs, and the scripts.
