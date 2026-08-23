> ## ✅ RESULTS (2026-08-20) — full battery in
> `measurements/ai_v9_19_gen16_mechanics_0819_endofrun.{json,md}`
>
> | § | verdict |
> |---|---|
> | **1** | **INFERIOR** — direct arena Δ **−41.57** [−50.15, −33.00], 6,400 cross-generation games. Bot-mediation VALIDATED (offset 10.81 < 15). Hodge clears (6.19 ELO, p=0.39, 0 cross-lineage cycles, read on the MERGED graph). |
> | **2** | cell liveness **PASS**; B1 **FAIL** (0.374→0.203, sig) · B2 **PASS** · B3 **FAIL** · B4 **MIXED** (belief reallocated, not lost). Whiff rate per pivot ROSE — the substrate bought repetition-suppression, not bait-avoidance. |
> | injection | **LEARNED BUT INSUFFICIENT** — β 1,526× stronger on loop steps, now 86% of α, and still 0 argmax flips against p≈0.97. `switch_branch` = all margin, no mass. |
> | order 4 | B1 is a **MEMORY** effect (in-window re-clicks 0.20×, out-of-window 1.26×, p=0.0042). |
> | ADDITION 1 | TD-aux **FALSIFIED** as the bait lever — B3 rose monotonically with λ. |
> | order 3 | PFSP frame-bias variant **DEAD**; capability variant prime suspect. |
>
> §6 did not cover this outcome (cells live AND §1 inferior). Owner pre-commitment executed →
> gen-17 `ai_v9_21_gen17_pfspoff_0820`, single-variable PFSP-off, fresh init.

# RUNBOOK — gen-16, THE MECHANICS GENERATION

**Pre-registered 2026-08-19, BEFORE gen-16 launches.** Every rule below is fixed while the number
it governs does not yet exist.

Gen-16 turns the conditional-mechanics SUBSTRATE on in the BASE. Fresh weights, pinned to current
main (the v96/v97 `gen3_critic_route_wave_v1` wall forbids a warm start); fresh pools, fresh
sentinels.

**The change list** (all nine acknowledged through the launch-diff gate vs gen-15):
`--pair-outcome-cell` · `--pair-outcome-switch` · `--switch-branch-cell` (OA2) ·
`--conditional-threat-cell` (OA1) · status-economy (in-place, no flag) · `--pfsp-scale 2.0` +
`--pool-spread` · `--intent-label-bot-weight 0.25` · and the two reward flags DROPPED because the
v8 composition is now the DEFAULT (verified in both the parser and the `RewardConfig` dataclass:
`all_shaping_pbrs=True`, `draw_penalty=-35.0`).

**Deliberately NOT in:** `--pair-value-route` (PV owes the C4-style offline gate, ledger C6 — no
exceptions), `--td-aux-coef` (rung 2 undecided; it is training-coefficient class, so a passing
rung 2 can join mid-run or ride gen-17), any fingerprint/flywheel machinery, and the α-batch's
grad-mode ladder / coef probe / B-move decisions (those are probes, not base changes).

## 1. Primary gate — non-inferiority vs gen-15

Dense `snapshot_ladder/ladder.json` tail-4, matched snapshot COUNT, at run END, SE from the
**paired refit** (`c'Σc`) — never the naive diagonal, never `main.elo`'s sparse fit.

- **NON_INFERIOR** iff Δ ≥ −15.0 AND CI95-low > −40.0. **INFERIOR** iff the whole CI sits below −15.
- Tie-break: more games per pair on the frozen ladder (`--backfill` cannot do this and now says so).
  Size against the **variance decomposition**, not the game count.

### 1a. AMENDMENT (2026-08-19) — the ladder gets a Hodge read, and it can qualify the gate

Added after registration because the instrument did not exist when §1 was written; it **adds a
check on the gate, it does not move the bar**. `python -m main.elo <run>` and `main.endofrun`'s §1
block now print the HodgeRank split of the ladder graph (`agents/training/hodge.py`): the transitive
**spine** the BT gate models, and the cyclic **width** it is structurally blind to, each measured
against the binomial noise floor its own game counts imply.

**Pre-registered use.** Take the read on the gen-16 ladder and on gen-15's — and on the merged
graph if a cross-lineage ladder is measured:

- If the **cross-lineage** cyclic content clears its noise floor materially — excess width ≥ the
  §1 non-inferiority margin (15 ELO) on the gen-16↔gen-15 edges, or a significant 3-cycle spanning
  both lineages — then Δ is not a clean scalar contrast: **re-fit the generation contrast with the
  cyclic term acknowledged** (an mElo-class two-component fit: spine + a low-rank cyclic part) and
  report BOTH numbers. The BT verdict is then reported *with* that qualification, never silently.
- Otherwise the BT gate is recorded as **VALIDATED against cycle contamination** — an explicit,
  dated statement that the transitive projection was checked rather than assumed.

**Baselines already on file** (whole-run ladders, 21 players / 174 edges / 814 triangles, 300
bootstrap reps): gen-15 spine 939 ELO, excess width **46 ELO** (p = 0.005), 3.8% null-adjusted
cyclic energy, 3 significant snapshot 3-cycles; gen-14 spine 765, excess **26 ELO** (p = 0.0033),
2.2%, 0 individually-significant cycles. Both are real width — so "there are no cycles here" is
already falsified and is not an available conclusion.

⚠️ The live `eval/hodge_width_elo` / `eval/hodge_cyclic_fraction` scalars are the WEAK counterpart
(a cycle's own games are a star, so the triangles come from the static bot round-robin, and at ~100
games/edge gen-15's 12 cycles read p = 0.13-0.93 — never significant alone). **The dense offline
read is the instrument here**, and the ELO-reading rules apply to it unchanged: no mid-run width
narration.

**Non-inferiority is the right bar and this is why:** the substrate is zero-init, and the BASE's job
is teaching the cells to be TRUE. Teaching the policy to USE them is the exploiter gates' job,
afterward. A base generation that merely holds serve while the cells come alive has done its job;
demanding a ladder gain here would be demanding the wrong thing from the wrong instrument.

## 2. THE BAIT/LOOP HUNT — the reason this generation exists

Registration of record: [`bait_loop_hunt.md`](bait_loop_hunt.md). Read at **matched scope**
(`--opponent 'sentinel_*'`) and **matched battle count**. All four together; any one alone has a
cheap way to be satisfied.

| # | bar | gen-15 | passes if |
|---|---|---|---|
| **B1 (primary)** | within-battle re-click rate | 32.2% | **< 16%** |
| **B2** | loop-battle rate | 13.9% | **< 7%** |
| **B3 (the honest one)** | median chosen-prob on residual loop steps | 0.963 | **< 0.85** |
| **B4 (a GUARD, not a goal)** | β slot acc · α SWITCH on loop steps | 82.1% · 76.2% | **flat or up** |

B4 inverted is the failure that would otherwise read as success: if the whiff rates fall while β and
α also fall, the run **lost the belief** rather than fixing the policy. Neither a repetition tax nor
a hand-coded immunity mask is a permitted response to a red number here — gen-14 had a repetition
tax and looped anyway.

**Launch-window liveness check (~5M), and it gates interpretation of everything else:** the new
`cell/<name>_{weight,grad}_norm` TB metrics MUST come off zero. The substrate is zero-init, so a
cell whose weight/grad norms never leave zero was never in the graph — and every downstream reading
about it would be a reading about nothing. Check this FIRST; a dead arrival channel invalidates §2's
interpretation before it invalidates anything else.

**REPEAT the α/β injection probe on gen-16.** On gen-15 it settled the mechanism by intervention:
forcing α/β to certainty produced 0 argmax flips in 40 arm-decisions and a bit-exactly zero β arm,
while the same intervention moved P(explosion) by 41.4 points — the signal existed and the channel
was multiplied by `is_boom`. `switch_branch` (OA2) is exactly the missing channel. If the injection
probe still reads ~0 flips on gen-16, the channel did not arrive, whatever B1 does.

## 3. Stall watch — SHAPE, not just rate

gen-15 read 0.73% cap-length episodes vs gen-14's 0.22% (Fisher p = 0.104, **not significant**), but
the distribution changed shape: mean turns 69.9 → 46.7, i.e. **bimodal — faster typical games with a
slightly fatter cap tail**. Re-read both the rate AND the turn distribution. A confirmed stall
regression argues for `--stall-pbrs` in gen-17, never for re-adding bias terms.

## 4. Fresh §4 route/family baseline — NOT a comparison

Run `critic_route_audit` + `edge_ablation_audit` at ≥12,000 states. This is a **fresh baseline at
gen-16's architecture**, not a delta against gen-15: v96 deleted routes and the substrate adds
cells, so past the surviving families the comparison is apples-to-oranges. Expect **eff-driven
shifts in the event-seat rows** — gen-16 is the first generation training with the event window's
EFF columns live (they were DEAD through gen-15; fixed `f05764e`). **That is a feature of this run,
not noise**, and must not be read as a substrate effect.

**AMENDMENT (owner, 2026-08-19) — the consequence families get an EXPOSURE-CONDITIONED read.**
A family gated on a mechanic reads dead by DILUTION when the audited decisions don't carry the
mechanic: c1's cell only exists when a boost move sits on an E3 seat, c5 needs Baton Pass, x needs
a Pursuit carrier. For each of c1/c2/c3/c4/c5/x report BOTH:
- **EXPOSURE** — the fraction (and raw count) of audited decisions where the family's gate is
  OPEN: a boost move offered (c1) · a status move (c2) · recovery (c3) · Protect (c4) · Baton
  Pass (c5) · a believed Pursuit carrier opposite (x);
- **KL / flips / dV CONDITIONED on exposure**, alongside the pooled numbers (continuity with the
  gen-4/gen-14 tables).
**A deletion license requires the CONDITIONED read at noise — a diluted zero licenses nothing.**
**AND it must be the CONTENT-ONLY arm** (`report[fam]["content"]`, `gen3_content_only_ablation_v1`):
the legacy full ablation charges every family for a bias constant shared bit-identically across
co-writing families (97% of c5's and 70% of c3's historical pooled KL was that artifact — ledger
2026-08-19). The full arm is continuity-only.
Where the exposed n is too small for a stable read, say so and defer that family's verdict to the
EXPLOITER GATES, whose teams were chosen carrier-first (boom teams, CMPass, TSS) and are the
designed bench for exactly these mechanics.

**OWNER REFINEMENT (2026-08-19): exposure is defined at the SCENARIO level, not the carrier
level** — a Pursuit carrier facing a Swampert has the move offered and the fact worthless. The
canonical scenario per family (these are the audit's conditioning states; mine the eval traces
for matches first, and where matched n is too small use CONSTRUCTED bridge scenarios in the
`damage_op_probe_fuzz_test` pattern):
- **x** — a Pursuit carrier (Tyranitar) opposite a Pursuit-vulnerable frail Ghost/Psychic
  (Gengar-class) with α leaning SWITCH. Carrier-only exposure does not count.
- **c1** — the MARGINAL-boost decision: setup move offered AND current stages ≥ +1 (Tyranitar at
  +1 deciding on the second Dragon Dance — boost-again vs attack). Stage-0 first-boost decisions
  are a separate, easier stratum; report both.
- **c5** — Baton Pass on the team AND stages ≥ +2 AND an alive receiver that inherits usefully
  (the Celebi two-Calm-Minds question). Expected tiny n in traces ⇒ constructed scenarios likely
  required.
  🚨 **c5 CANNOT be read on any snapshot trained before 2026-08-23.** The client dropped every
  Baton-Passed stat stage (`Battle.switch` cleared them and the `[from] Baton Pass` tag was sliced
  away before anyone read it — ledger 2026-08-23), so the "inherits usefully" half of this gate was
  *absent from the observation* and the boost PBRS term *penalised* a successful pass. A pre-fix c5
  number measures a missing fact, not indifference to one; it licenses nothing in either direction.
  Re-read c5 only on a snapshot trained at or after the fix.
- **c3** — a DAMAGED recovery carrier (Milotic-class) where heal-rate vs opponent
  damage-per-turn is the live margin, conditioned on LOW boom threat (recovery value collapses
  against Explosion — a Metagross-without-boom opposite is the clean case).
- **c4** — owner: deprioritized ("don't care"); its pooled number may stand as its verdict, no
  scenario construction effort.
- **c2** — the post-STATUS world (Toxic/Thunder Wave/sleep consequence): status move offered
  against an unstatused, non-Substitute target. (NB the owner's "second Dragon Dance" scenario
  belongs to c1, not c2 — recorded here so the label mix-up cannot redirect the audit.) Context: on gen-14's pooled table the entire c-block
spans KL 0.00001–0.00233 against d1's 0.062 — the conditioned read decides whether that is
irrelevance or dilution, and per the standing dependence-vs-coverage rule, families whose facts
lack a substrate successor (c1's post-setup hypothetical, c3's heal-vs-KO flip, c5's inheritance)
additionally need a per-fact coverage ruling before any code is deleted.

## 5. α/β `_pool` readouts + the switch-coverage matrix

Baselines in the ledger's sweep section. Report both; they are the belief-side companions to B4.

## 6. What would make this generation a mistake

Stated now so it cannot be rationalised later: if gen-16 is **non-inferior on §1 but the cells are
live and B1/B2/B3 do not move**, then the substrate arrived and the policy still will not use it —
which sends the question to the exploiter gates (where elicitation is the named confound), NOT to
more base training. If instead the **cells never came off zero**, the generation tested nothing and
the correct response is to fix the arrival channel and re-run, not to reinterpret §1.
