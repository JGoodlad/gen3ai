# Adversarial review of X61 / N1 (PREREG at `de269480` + `ae5c8dcb`), 2026-10-25

**Scope.** `PREREG.md` (this directory), the generator `src/utils/team_corpora/build_n1_proc.py`, the corpus
`n1_proc_v1` (12,000 teams, 0.7 share per side), and the arm builder `src/main/x61/arms.py`. Context:
`literature/sota_survey_2026-10-26.md` §5 N1, `model/generalisation_review_2026-10-26.md`,
`goal_b_panel_validity_2026-10-26/README.md`, and REVIEW A's revealed-fit rule (`human_replay_quality_2026-10-25/REVIEW_A_data.md` B2).

**Method.** I read the PREREG, the generator, the arm builder and its transcripts, and the context docs. I ran two CPU
checks (`review/`, under rule 5b's cores, nice 19, no GPU), each deterministic:
- `review/coverage.py` → `coverage.log`: how much of each panel the corpus and the pool already hold, per team, per
  mon set and per full set.
- `review/power.py` → `power.log`: a Monte Carlo of §5's registered two-look rule, with K0's σ estimate included.
  With K0 switched off it reproduces the PREREG's own figures (GO 0.30 / 0.52 at +5 pp, 0.63 / 0.87 at +8), which
  validates the model.

I read the K0 seeds' `metadata.json` (read-only). `rb_toy_ent_s4001b` is a same-pin (`3899277a`) continuation of
`s4001` from step 72,000,094, so it is a legitimate third seed.

## Verdict: **NOT-READY**

The build is sound:
- the generator is deterministic and Smogon-only;
- the validator, the engine prefilter and the soak are real;
- the two arms differ by the lever alone (`arms_test.py`; the transcripts' token diff).

The design is not. Two blockers mean that no class the design can return licenses what its table says:
- the primary's H half is fed by the same generator as the corpus (B1);
- the GO / KILL bars are in production units, but they are applied to a toy whose whole familiarity premium sits
  below the GO bar (B2).

Under a true null, the registered rule ends UNDECIDED 73–79 % of the time (M1).

**Recommended design** (§ Recommendation): **N1-P, a production fork of D3's kind with the N1 corpus, read against
the shared C0. ≈ 12.5 GPU-h at look 1 (3 seeds × 20M), ≈ 21 GPU-h with look 2.** The primary is Δ on F. If the
from-step-0 claim must be tested on the toy, use toy Option T (≈ 23 GPU-h), never the registered one.

## Findings

| # | severity | finding | fix |
|---|---|---|---|
| B1 | **BLOCKER** | H is generator-adjacent: 62 % of its full sets sit verbatim in the corpus (the pool holds 10 %) | primary Δ_F; move H next to P72 |
| B2 | **BLOCKER** | The thresholds are production-scaled; the toy's effect ceiling is below GO, and BELOW-GO / KILL wrongly license "no production follow-up" | toy-scaled screen; no toy licence over production |
| M1 | MAJOR | Power is overstated once K0 is drawn (+5 pp: GO 0.23 / 0.41, not 0.30 / 0.53); under a null, KILL fires 6–11 % and UNDECIDED 73–79 %; false GO reaches 8.6 % at σ = 4 pp | ≥ 2 seeds per arm at look 1, or the production design |
| M2 | MAJOR | K0 cannot size σ_s (2 df), and the 2 pp floor comes from forks, which share a parent | own-seed σ, a fresh-run floor of 3 pp, home72 in K0 |
| M3 | MAJOR | A cheaper and more interpretable read exists at the same GPU-h: a production fork against the already-paid C0 | N1-P (§ Recommendation) |
| M4 | MAJOR | Downstream: an N1-trained model's goal-(b) U is inflated on 2 of its 3 panels (H, P72) | a set-level familiarity line in the goal read |
| m1–m9 | MINOR | the guard radius; paired dice; the look-1 Δ̂ ≤ 0 path; the smoke and throughput; the single read point; the share and coherence; V-DOSE; the Kakuna procedural familiarity; the pin predates the GIGO fixes | below |

### B1 (BLOCKER). H is fed by the corpus's own generator family: the primary carries the contamination §5 excludes P72 for

**Evidence** (`review/coverage.log`). "Mon set" is species + sorted moves (Hidden Power typed); "full set" adds the
item, ability, nature and EVs (`gen3_team_overlap_v1` `key` / `strict_key`).

| panel | mon sets in corpus (≥ 5 copies) | in pool | full sets in corpus | in pool | max shared with one corpus team |
|---|---|---|---|---|---|
| **H** `heldout72_hpfix` | **94.9 %** (82.4 %) | 38.0 % | **62.3 %** | 10.4 % | 2 → 58 teams, 1 → 14 |
| **F** `forum72` | 83.1 % (70.1 %) | 70.1 % | 19.4 % | 23.1 % | 2 → 57, 1 → 15 |
| P72 | 85.6 % (70.4 %) | 30.8 % | 56.5 % | 8.1 % | 2 → 55, 1 → 17 |

- **H behaves like P72, not like F.** Under N1, at least 62 % of H's full sets become trained sets: +52 pp over
  C0f's 10 %. For F the corpus adds at most 19 pp of full sets, and F's mon sets were already 70 % in the pool.
- **The reason is H's provenance.** H's unrevealed fields are filled from usage (hl_05_26's NaiveUsagePredictor fill;
  panel-validity G8: "H's sets are partly invented"). That is the same population N1 draws its moves, items and
  spreads from.
- §5 removes P72 from the primary because "a gain on P72 is partly 'learned the generator'". By the PREREG's own
  argument, H carries that defect too.
- **The team-level exclusion holds, and binds.**
  - 58 / 72 H teams sit exactly at 2 shared sets: the corpus is packed against the radius.
  - The ≤ 2 rule keeps the panels unfamiliar **as teams**, the registered definition. It does not keep them
    unfamiliar **as sets**, which is the channel a toy with no belief stack pilots through.

**Fix:**
- Make the primary **Δ_F**, with F at 60 units (1,200 games) per arm per seed (unit seeds 0–59). Its binomial
  variance per arm equals U_HF's at 600 + 600, so the power is unchanged.
- Move H into the generator-adjacent secondaries with P72.
- Register **Δ_H − Δ_F** as the "learned the generator" line.
- Stamp the coverage table in §1.

### B2 (BLOCKER). The decision thresholds are production-scaled; the toy cannot reach GO by the mechanism tested, and a toy null cannot license "no production follow-up"

- **The toy's whole premium is smaller than the GO bar.**
  - T\* at 290M: home72 − F = **3.5 pp** (0.140 logits; P\* 6.8 pp; X51 README).
  - At 100M the pool dose per team is about a third of T\*'s, so the toy's premium is likely smaller. K0 does not
    measure it (it plays H and F only).
  - N1 works, by the survey's mechanism, by closing that premium. On the gap-closing route the toy effect is
    bounded by about +3.5 pp.
  - GO needs Δ̂ ≥ +4, and in practice Δ̂ ≳ +6.8 at look 1 (the widened LB > 0).
- **The PREREG's own model expects a sub-threshold toy effect.**
  - §7 predicts Δ_HF +3.0 [−2, +8]: "halved in logits for the toy's smaller familiarity premium" from the survey's
    production +3 to +8.
  - The survey's risk line says the same: "the toy's own familiarity premium is about half P\*'s … which lowers the
    test's sensitivity".
  - So a toy at +2 to +3 is exactly what a production +4 to +6 predicts.
  - Yet **BELOW-GO (UB < +4) licenses "no production follow-up"**, and KILL (UB < +2) "no production follow-up on
    this family". Those licences reject the hypothesis on the evidence the hypothesis predicts.
- **What the toy can license.** A toy GO is evidence for production. A toy null is evidence only that "the toy, at
  its premium, cannot show it". FINDING 11 (no belief stack) widens that gap further: production's premium lives
  partly in the belief channel (FACING), which the toy lacks.

**Fix:**
- Turn the toy read into a **one-sided screen**: SCREEN-GO if the widened LB > 0 on Δ_F (no Δ̂ floor, or Δ̂ ≥ +2 to
  match the halving model). Every other outcome reads "the toy is uninformative" and licenses nothing about
  production.
- Delete the KILL and BELOW-GO licences.
- Add home72 to K0 (3 × 200 games), and report the old toy's premium at 100M as the sensitivity ceiling.

### M1 (MAJOR). Power and error rates, recomputed with K0 drawn

From `review/power.log`: 100k draws per cell. The base U is 0.40. Each figure is the final class probability after
the registered CONTINUE.

| true seed SD σ | Δ = 0: GO / KILL / UNDECIDED | Δ = +2 GO | Δ = +4 GO | Δ = +5 GO (look 1 / final) | Δ = +8 GO (look 1 / final) |
|---|---|---|---|---|---|
| 2 pp | 0.027 / 0.062 / **0.79** | 0.11 | 0.29 | **0.23 / 0.41** | 0.51 / 0.76 |
| 3 pp | 0.061 / 0.097 / **0.74** | 0.14 | 0.27 | 0.23 / 0.35 | 0.42 / 0.61 |
| 4 pp | **0.086** / 0.114 / **0.73** | 0.15 | 0.24 | 0.22 / 0.30 | 0.36 / 0.50 |

- **The PREREG's 0.30 / 0.53 assumes σ_s equals the floor exactly.** With K0's estimate drawn, `max(σ̂, 2)` can only
  widen the interval when the true σ is 2, so the power at +5 pp falls to 0.23 / 0.41.
- When the true σ is 4, K0 under-reads it often (M2), and the false GO rate climbs to **8.6 %**: 3.4× the nominal
  one-sided 2.5 %. Two unspent looks add to that.
- **Under a true null, the design almost never decides.** KILL fires 6–11 % of the time and UNDECIDED 73–79 %. Its
  §7 forecast (UNDECIDED 0.50) is optimistic.
- **"Is 2 seeds per arm the minimum sound design?"** Var(Δ̂) ≈ (4 + 2σ²) / m pp² at 600 + 600 games, where m is the
  number of seeds per arm.

  | m per arm | half-width at σ = 2 | half-width at σ = 3 | P(GO) at +3 pp, σ = 2 | P(GO) at +5 pp, σ = 2 |
  |---|---|---|---|---|
  | 2 | 4.8 pp | 6.5 pp | 0.23 | 0.53 |
  | 3 | 3.9 pp | 5.3 pp | 0.32 | 0.70 |
  | 4 | 3.4 pp | 4.6 pp | 0.41 | 0.82 |

  Two seeds are the minimum that can estimate σ at all. Even two are weak at the toy's plausible +2 to +3 pp
  (B2). Adding games beyond 600 per panel buys little, because seed variance dominates (8 of 12 pp² at σ = 2).

### M2 (MAJOR). K0 cannot size the spread it is asked for, and the floor comes from the wrong regime

- **K0's estimate has 2 degrees of freedom.** K0 plays 3 seeds × 200 games per panel, so each seed's U_HF carries
  ≈ 6 pp² of binomial variance.
  - At σ = 2 the estimate floors to 0 with probability ≈ 0.45.
  - At σ = 4 it reads below 2 pp (so the floor binds) with probability ≈ 0.36, and 0 with ≈ 0.24 (χ²₂
    arithmetic).
  - It cannot tell 2 from 4.
- **The 2 pp floor is a fork number.** D1's five production fork seeds read per-seed Δ of +4.3 / 0 / +4.0 / 0 / +2.1:
  a sample SD of 2.1 pp, against ≈ 2.4 pp of binomial noise alone, so the overdispersion is ≈ 0. Forks share a
  parent. Two FRESH runs from step 0 diverge from the first team draw, and their spread is unmeasured (**UNVERIFIED**
  in either direction).
- **K0 also measures another pin's toy.** It plays the old toy at `3899277a`; the arms run at the new pin
  (FINDING 8: bf16 trunk, the recipe.fresh sizing).

**Fix:**
- Estimate σ from the look's own seeds (≥ 2 per arm), floored at 3 pp for fresh runs.
- Keep K0 only as the level check: the HOLD at 0.15, plus home72 (B2).

### M3 (MAJOR). A read that is cheaper to interpret exists at the same GPU-h

**N1-P**: a production fork with the N1 corpus, built on D3's pattern (pin = C0's production pin + one commit
confined to `src/utils/team_corpora/`, `kit pin-check`), read against the **shared C0** by `main.x35.d1`'s
statistics.

- **The control is already paid for.** C0 s3501–s3505 are run and read against Kakuna.
- **Fork seeds overdisperse by ≈ 0** (M2), so 3 seeds at look 1 give a half-width of ≈ 3.9 pp. That is the D1
  look-1 width: +2.78 [−1.12, +6.67].
- **The read is on the system goal (b) measures,** whose premium is 10–13 pp (against the toy's ≈ 3.5).
- **Cost:** ≈ 12.5 GPU-h at look 1 (3 × 20M at ≈ 1,333 steps/s; D3's look-1 figure 12.1), plus ≈ 8.3 GPU-h for look
  2 (s3504–s3505).
- **What it does not test is "from step 0".** A NULL is confounded with plateau plasticity (the named D1 risk), so it
  kills only "N1 at the plateau".
- **What a GO says.** Breadth at 10⁴ moves the real system even at the plateau. That is a stronger licence for a fresh
  production N1 (X53 with this corpus, ≈ 21 GPU-h to 100M) than any toy result.
- **It also gives a dose read beside D1** (1,981 forum teams at 0.5, +2.1 [−0.9, +5.1]) if run at D1's share.

**A mixture with D3's corpus** needs a multi-corpus `--team-mix` (a build), and it would be two levers in one A/B
(owner rule: one lever per A/B). Not now.

### M4 (MAJOR, downstream of X61). Goal (b)'s meter would be inflated by any N1-trained model

- U = H + F + P72. Two of its three panels are now generator-adjacent to this corpus (B1).
- The panel-validity rule ("unfamiliar = ≤ 3 shared sets with our training", per TEAM) cannot see set-level coverage.

**Fix:** before any production N1 is read for goal (b):
- add a per-panel full-set coverage line, by the model's training corpora, to `goal_b_panel_validity`'s
  recommendations;
- lean the goal claim on F and the sealed `forum216b`.

The X54 owner should know this before forum216b's unsealing is spent on an N1 model.

### Minor findings

- **m1. The guard mismatch (radius 4 at startup, 3 in the build) is NOT a live hazard for X61.**
  - The build and `n1_proc_test.py` (`test_no_team_is_near_a_held_out_panel_at_radius_three…`, a non-slow test)
    enforce 3 on the committed bytes.
  - V-MIX pins the tier sha, so a rebuilt corpus cannot slip in.
  - It is latent for the next corpus: `matchup_setup` refuses only at `HP.NEAR_NEIGHBOUR_MIN_SHARED` = 4.
  - **Fix:** the guard reads the corpus manifest's own `panel_registry.near_neighbour_min_shared` and refuses at
    min(that, 4), or the registry moves to 3 (with a re-stamp unit).
  - Also noted: the guard checks `mixes[0]` only. That is harmless while both sides name one corpus, but a
    per-side mix would escape it.
- **m2. "Paired dice" give no variance reduction.** With probability 0.7 the first team draw already differs, so the
  trajectories diverge at once. §5 correctly credits no pairing; §2's wording implies it does. Say so.
- **m3. Look 1 UNDECIDED with Δ̂ ≤ 0 has no stated path.** Presumably it ends as the final UNDECIDED. Write it down.
- **m4. The smoke (FINDING 10) and the throughput at the pin are unverified.**
  - Add to §3: the `--debug --steps 10000` smoke with the lever under `mem_cap.sh` at launch time, CPU.
  - Add a first-hour steps/s guard: HOLD if below 0.7 × 4,880.
  - Both arms' GPU-h rest on an UNMEASURED throughput.
- **m5. One read point conflates "generalises better" with "learns slower".** Procgen's many-level agents train
  slower early. Register the 50M checkpoints' F read (CPU only) as a trajectory secondary.
- **m6. The 0.7 share has no dose rationale.** The survey's pool share of "e.g. 0.3" is an example. D1 ran 0.5 and D3
  0.3, so the three breadth levers cannot be compared at one share.
  - With procedural opponents 70 % of the time, the trainee learns opponents whose move / item / spread
    correlations are absent: the sets are drawn independently given the species (FINDING 6). F's human-coherent
    sets are where that would cost.
  - A KILL then cannot separate "breadth is null" from "incoherence costs". The KILL wording ("this family") is
    scoped correctly; Δ_H − Δ_F (B1) is the read that separates them.
- **m7. V-DOSE is likely UNVERIFIED by construction** (corpus draws carry no pool index). A per-side corpus-draw
  counter in the TEAM-MIX log line would make it a real guard (a small build).
- **m8. Kakuna's possible familiarity with procedural style** (`paper_variety`, panel-validity G4) is common to both
  arms, so it cancels in Δ to first order. It touches only the P72 / n1p72 secondaries' levels. FINDING 5 is right
  to leave it uncontrolled.
- **m9. The pin predates the 10-26 GIGO audit's fixes** (`d8d674b99`, landed after `de269480`).
  - The toy is exposed to only one of them: the silent volatile ends (`gen3_volatile_silent_ends_v1`: Destiny Bond
    and trapped cells). The op fix is off under `--no-damage-op`, and the intent label sits at coefficient 0.
  - The pool's two packer defects are 3 mons, common to both arms.
  - The audit's pack oracle found all 12,000 `n1_proc_v1` packs identical.
  - The contrast is clean. A re-pin (B1 / B2 force a re-registration anyway) should take the fixed tree and re-run
    `arms check --tree`.

## Answers to the brief's questions, in one line each

1. **Construct.**
   - A toy at 100M, one seed per arm, tests "≥ 10⁴ procedural teams from step 0 move a no-belief toy's piloting of
     F-like teams". It is from step 0, yes: both arms are fresh at one pin.
   - A GO is evidence for production. A KILL or BELOW-GO licenses nothing about production (B2).
2. **Power.**
   - The PREREG's numbers reproduce only with σ_s fixed at the floor. Drawn, they fall to 0.23 / 0.41 at +5 pp
     (M1).
   - Two seeds per arm are the minimum, not a sound design, at the toy's plausible effect.
3. **The corpus.**
   - Incoherence is a real risk, confounded into KILL (m6).
   - The ≤ 2 exclusion keeps the panels unfamiliar as teams, not as sets (B1).
   - The sibling generator makes H familiar, not only P72 (B1).
   - Kakuna's procedural familiarity cancels in Δ (m8).
4. **The control.**
   - toy_c0f is matched token for token. The 17 dropped flags are dropped from both arms and are inert under
     `--no-damage-op`.
   - K0 on the old toy can mis-size the spread, mostly by under-reading a larger σ (M2).
5. **The guard.** Not live for X61; latent for the next corpus (m1).
6. **The alternatives.** N1-P (a production fork) wins per GPU-h on interpretability; it does not test step 0 (M3).
   A D3 mixture is two levers.

## Recommendation

1. **Run N1-P first.**
   - **The lever:** `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both`. 0.5 is D1's share, so N1-P − D1 is a
     corpus-scale contrast at one share.
   - **The arm:** a D3-pattern kit arm forked from C0's parent at C0's production pin + one `team_corpora` commit
     (a small build plus a PREREG).
   - **Seeds and length:** s3501–s3503 at 20M (≈ 12.5 GPU-h); look 2 adds s3504–s3505 (≈ 8.3 GPU-h).
   - **The primary:** Δ_F against the shared C0, D1's statistics, F at 30 units per seed on both arms. C0's extra F
     units are CPU only.
   - **Reported:** U, Δ_H, Δ_P72 and Δ_H − Δ_F (B1), plus home72's in-pool cost.
   - **GO:** LB > 0 → register X53-N1 (fresh production with this corpus, ≈ 21 GPU-h to 100M).
   - **NULL:** kills only N1 at the plateau.
2. **Toy Option T,** only if the orchestrator wants the from-step-0 screen as well:
   - B1 and B2 fixed: Δ_F primary, a one-sided SCREEN-GO, no licence over production;
   - **2 seeds per arm at one look** (s6101 + s6102, ≈ 23 GPU-h), σ from the look's own seeds, floored at 3 pp;
   - the in-house unfamiliar meter for the toy: each arm against the frozen toy R, mirrored pairs on `--panel
     forum72` and `heldout72_hpfix` (≥ 500 pairs per panel), reported beside Kakuna (X37b: never instead of it);
   - **a cheaper variant,** if K0 (extended to the old toys' 50M checkpoints, CPU only) shows a 50M toy already
     above the 0.15 floor and within ≈ 3 pp of its 100M level: 4 seeds per arm at 50M for the same ≈ 23 GPU-h
     (half-width ≈ 3.4 pp at σ = 2, P(GO) 0.41 at +3 pp, against 0.23).
3. **Do not launch X61 as registered** (≈ 12 GPU-h for a design that, under its own forecast, mostly returns
   UNDECIDED, and whose two informative non-GO classes would wrongly close production).
