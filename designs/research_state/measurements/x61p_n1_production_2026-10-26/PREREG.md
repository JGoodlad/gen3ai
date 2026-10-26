# X61P / N1-P — Procgen-scale procedural team breadth at the production plateau: PRE-REGISTRATION (2026-10-26)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED** (no `rb_x35_n1p_*` run exists). The build
is CPU only. The launch is the orchestrator's GO. Nothing below changes after the first arm launches except by a dated
amendment, written before the read, that says why.

**Why this design.** The adversarial review of X61 ([`../x61_n1_breadth_2026-10-25/REVIEW.md`](../x61_n1_breadth_2026-10-25/REVIEW.md),
`aed145fd`) found the toy design NOT-READY: H is fed by the corpus's own generator family (B1), and the toy's whole
familiarity premium sits below its GO bar (B2). It recommended **N1-P**: a production fork of D3's kind with the N1
corpus, read against the shared C0 by D1's statistics, with the primary on F alone. This file registers exactly that.
X61's toy registration is not run as written.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170; C0 = X35's plain-continuation control; F = `forum72`; H = `heldout72_hpfix`; P72 = `p72`; U = H, F and
P72 at equal weight (goal (b)'s meter); S = in-house strength against the frozen R; GUESS / ESTIMATE as in
`design_beat_metamon_path.md`.

## 1. Question and mechanism

Does mixing **12,000 procedural teams** (10⁴, the scale every system that generalises across teams trained on;
`sota_survey_2026-10-26.md` §5 N1) into the production plateau's training raise how well it pilots **human-built teams
it has never seen** against Kakuna? Every breadth test so far (D1, D3, X52) added 10³ teams. D1 (1,981 forum teams at
0.5) read H +2.10 [−0.92, +5.11] at its final, with its gain on its own forum population.

**What it does not test is "from step 0".** A NULL here is confounded with plateau plasticity (D1's named risk), so it
rules out only "N1 at the plateau" (§6).

## 2. The arm (kit arm `n1p`, `python -m main.x35.kit describe --arm n1p --seed <s> --call reference_population_production.json`)

| item | value |
|---|---|
| start | P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, R's frozen call), as C0 |
| the ONE lever | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both`: with probability 0.5 per episode and side the team is a corpus team, else exactly C0's draw (the pool with its 10 % sample bias). 0.5 is D1's share, so N1-P − D1 is a corpus-scale contrast at one share (descriptive only: two arms, two corpora). The in-loop eval stays on the pool |
| everything else | C0's argv token for token (`kit_test.test_n1p_is_c0_plus_the_procedural_mix_and_nothing_else`): `--behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --seed <s> --run-name rb_x35_n1p_s<s>`, the recipe inherited from X26's record |
| pin | **N1P_PIN `5f8f6cfa37de8e283ba553c0d415680cf51bff18`** (`batch.N1P_PIN`; annotated tag `x61p-n1p-pin-5f8f6cfa` on origin, local branch `x35-n1p-pin`) = C0's production pin `f4b25238` + ONE commit with four files, all under `src/utils/team_corpora/`: the corpus's `manifest.json`, `soak_selfcheck.json` and `teams.jsonl.gz`, byte for byte main's `de269480`, and the one-line `CORPORA` registration in `__init__.py` (corpus tree `b273cfe1ff11`). `kit pin-check --arm n1p` checks the parent, the paths and the tree; `batch_test.test_the_registered_n1p_pin_adds_only_the_corpus_files_main_landed_and_its_registry_line` checks that the corpus subtree is main's and that `__init__.py` differs by that line alone. The trainer, env core, model, launcher and `data/` are C0's byte for byte, and `f4b25238` carries the pinned-data root (`3d6780ee`) |
| **NOT in the pin: the 10-26 GIGO fixes** (`d8d674b9`) | C0 ran without them (its runs predate them, at `f4b25238`). Mixing them in would put an obs change (the silent volatile ends, obs golden 36/991), an op-input change (config v162) and a label change into the arm and not the control: a second lever. The fixes are common-cause defects of both arms, so the contrast stays clean. Not the builder, D3's corpus, or the AM-D1-FIX label guard either (§10 FINDING 3) |
| seeds | look 1: 3501–3503; look 2: 3504–3505 only on look 1 = CONTINUE (`batch.N1P_STAGES`, `kit stage-check`) |
| control | **C0 s3501–s3505**, already run at `f4b25238` (each K9(b) PASS): shared, 0 GPU-h. Its finals are pinned by sha256 in the read specs (§4) |
| GPU | **≈ 12.3 GPU-h at look 1** (3 × ≈ 4.1; ESTIMATE from the 11 sibling forks measured at this pin and length, 3.84–4.43 GPU-h each: C0 ×5, D1 ×5, D3 s3501; `x35_breakers_stage_2026-10-17/dose/`) + ≈ 8.2 at look 2 = **≈ 20.5 GPU-h** for the final read |

**The corpus `n1_proc_v1`** (X61's PREREG §1 has the full card; `utils.team_corpora.build_n1_proc`, BUILD_SEED 6101):
12,000 whole teams from the Smogon facade alone (`agents.gen3_data.priors`), the pinned `TeamValidator('gen3ou')`, level
100 (0 of 12,000 carry another `Level:`, checked here with main's `label_path_defects`), tier ALL sha256
`d59f44f0…5314f`, data sha256 `99e3425a…0bad6`. No team shares ≥ 3 mon sets with any team of H, F, P72, `heldout72`
or the sealed `forum216b`. The 10-26 audit's pack oracle found all 12,000 packs identical under the fixed packer
(review m9), so the packer defects do not touch the corpus.

**The soak** (`n1_proc_v1/soak_selfcheck.json`, `gen3_team_corpus_soak_v2`): **PASS**, 120,000 battles, 4,457,859 p1
decisions, 0 core errors. It is NOT re-run: the pin carries the same bytes, and the soak's env build is the pin's by
source (`git diff f4b25238 de269480` over `src/rust_env`, `src/rust_sim/src`, `src/utils/teambuilder.py` and
`src/utils/team_packing.py` is EMPTY; over `src/utils/rust_env` it touches only the fixture-battle and build-stamp tools).

**The pinned trainer's startup on this argv** ([`startup_team_phase_at_pin.txt`](startup_team_phase_at_pin.txt): a
detached checkout of the pin, its own `src/` and `data/`, the real Node validator, CPU):
`🧬 [TEAM-MIX] n1_proc_v1@ALL (12000 teams, tier sha d59f44f0b33b) share 0.50 on both sides; eval stays on the pool`,
then `held-out panels: 0 hits over 5 panel(s) (registry sha ea3917c2f1e6); 0 corpus team(s) pack identically to a
pool team`. The team phase took 18.5 s at 0.87 GB peak.

**Dose (ESTIMATE, D1 doc §4.1's rule: ≈ 570k games in 20M).** Per side ≈ 285k corpus draws, ≈ 24 per corpus team,
against ≈ 790 per pool team in C0. Each pool team's exact draws fall to ≈ 395 (−50 %, as D1's). Against P\*'s ≈ 9,300
cumulative pilots per pool team, that removes ≈ 4 % of a team's exact exposure.

## 3. Pre-launch checks (a failure is a HOLD, never a launch)

`launch_x35.sh n1p <seed> --pin 5f8f6cfa37de8e283ba553c0d415680cf51bff18 --lease-owner <owner>` runs every pre-check
(2b `pin-check --arm n1p`, 2d `corpus-check --arm n1p`, 5 `checkargs --pin`, 6 the dry-run with the card visible, 9
`stage-check`) and launches only with `GO`. Before the first GO, also:

1. **The smoke** (review m4): the PIN's trainer, `--debug --steps 10000 --team-mix n1_proc_v1@ALL:0.5 --team-mix-sides
   both`, CPU, `GEN3AI_MODELS_DIR` a scratch dir, in the foreground under `mem_cap.sh` (root rule 3). It must finish
   with the TEAM-MIX banner and no FATAL. NOT run at this registration (§10 FINDING 6).
2. The tag `x61p-n1p-pin-5f8f6cfa` dereferences to `batch.N1P_PIN` on origin.

## 4. PRIMARY F: D1's rule on `forum72` alone (`main.reads`, rule `d1`, arm label `X61P`)

- **The read spec:** [`read_spec_look1.json`](read_spec_look1.json) and [`read_spec_look2.json`](read_spec_look2.json)
  (`python -m main.reads check-spec` exit 0 on both; `rules_test.test_x61ps_committed_specs_parse_…` pins them). The
  n1p finals are `final` refs (`models/rb_x35_n1p_s<s>/final_model.zip`, step ≥ 300,000,170, quiet 300 s); the C0
  finals are `pinned` refs by sha256. Both arms play in ONE state, so `main.reads` holds them to one playing commit.
- **Cell:** Kakuna ckpt 34, greedy against greedy, `forum72`'s committed config, role-balanced 20-game units, **60 units
  per seed final (unit seeds 0–59, 1,200 games)** on BOTH arms (C0's units are CPU only), anchor group `primary`.
- **Statistic** (`main.x35.d1.read_d1` with `ARM_PANELS["X61P"] = ("forum72",)`, `ARM_UNITS["X61P"] = 60`): Δ_F =
  N1-P − C0 of the per-arm pooled F score (K = 1, so the plain proportion), Wilson per arm, Newcombe method 10, AM-5's
  √max(1, φ̂) widening from the look's own seeds; 95 % at both looks; the sign rule ≥ 2 of 3 seeds (look 1), ≥ 3 of 5
  (look 2).
- **Classes** (rule 8: a bound within 1e-9 of 0 or ±3.0 is UNDECIDED): **GAIN** iff LB > 0 and the sign rule;
  **LOSS** iff UB < 0; **NULL** iff the interval lies inside [−3.0, +3.0]; else **UNDECIDED**. At look 1, UNDECIDED
  with Δ̂ > 0 → CONTINUE to look 2 (final); **UNDECIDED with Δ̂ ≤ 0 at look 1 is FINAL** (review m3: no look 2).
- **Floor:** each arm counts ≥ 1,200 · seeds − 20 F games (at most one uncounted unit per arm); else INCONCLUSIVE.
- **Power** (`power/power.py` → [`power/power.log`](power/power.log): a Monte Carlo of THIS rule through
  `main.x35.d1.d1_primary`, 4,000 draws a cell, base F 0.49 = C0's own F at D1's read, 342 / 700). σ is the true
  between-seed SD of a final's F score; D1's five fork seeds overdispersed by ≈ 0 (review M2).

  | true Δ_F | σ = 0: look-1 GAIN / final GAIN | σ = 1 pp | σ = 2 pp |
  |---|---|---|---|
  | 0 (false GAIN) | 0.024 / 0.035 | 0.031 / 0.049 | 0.049 / 0.074 |
  | +2 pp | 0.33 / 0.56 | 0.30 / 0.48 | 0.22 / 0.35 |
  | +3 pp | 0.64 / 0.88 | 0.54 / 0.78 | 0.38 / 0.57 |
  | +4 pp | 0.88 / 0.99 | 0.76 / 0.95 | 0.54 / 0.77 |

  Under a true null the final is NULL 0.53 (σ = 0) and 0.38 (σ = 1), and UNDECIDED 0.42 and 0.54.

## 5. CO-PRIMARY P: D1's rule, verbatim

S of each seed final against the frozen R (`reference_population_production.json`, content `cef0960e…`), 7 members ×
500 mirrored pairs on `home72` (regime `d4de0fd7c16a2658`, D1's), S group `p` of the same specs; Δ_P = mean N1-P S −
mean C0 S, Welch two-sided 97.5 %. **NON-INFERIOR** iff LB > −2.0 pp. **C0 is re-played, not imported.** D1's banked
P rows carry no regime id (`regime_ids: [""]`), and `main.reads` decides no group without one. Declared: home72's
teams are not in the corpus, but the corpus halves their exact exposure (§2 Dose); P reads that cost.

## 6. Verdicts and what each licenses (D1's table, then the follow-up)

| verdict (`d1` rule) | rule | what it licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | F GAIN, P NON-INFERIOR, every guard PASS | **GO:** register **X53-N1**, a FRESH `--arch production` run with this corpus from step 0 and a 100M look against X26 @100M (≈ 21 GPU-h to the look; beat-Metamon §6.5). Breadth at 10⁴ moves the real system even at the plateau. Nothing joins the production recipe on this read alone |
| **TRADE** | F GAIN, and P's inferiority not excluded or a guard FAILs | **GO** as above, with the exchange rate (Δ̂_F / −Δ̂_P) handed to the owner; X53-N1 then carries P as its co-primary |
| **NULL** | F NULL | rules out **"N1 at the plateau"** (this share, 20M) ONLY. It does not rule out N1 from step 0 (the plasticity confound, §1), the toy screen (the review's Option T) or a human-derived corpus (X61 PREREG §10) |
| **LOSS** | F LOSS | procedural breadth hurt human-team piloting at the plateau (the incoherent-set risk, review m6): report it; no N1 family follow-up at the plateau |
| **UNDECIDED** (final) | otherwise | nothing licensed; the orchestrator rules |

**Guards, read only if F is GAIN** (D1's / D3's): G-F, G-E (the reader `ge_n1p`, a fresh best responder of N1-P
s3501's final, against the shared `ge_c0`; `main.x35.ge`), G-C (`main.x35.gc`). Otherwise the spec's
`guards_not_read` records each INCONCLUSIVE with the A1 ruling's reason.

## 7. REPORTED, never decided (B1: the generator-adjacent panels)

The anchor group `reported` of the same specs: **H, P72 and home72**, 7 units per seed final (unit seeds 0–6, D1's), both
arms. The `d1` rule's `reported_group` writes them into the record's `reported.reported_panels`
(`main.x35.d1.reported_panels`):

- per panel, the pooled N1-P − C0 Newcombe difference: **Δ_H**, **Δ_P72** and **Δ_home72** (Kakuna's in-pool cost);
- **U** per arm and ΔU, unwidened and φ̂-widened: goal (b)'s meter, reported only (§8);
- **Δ_H − Δ_F**: the "learned the generator" line. 62 % of H's full sets sit verbatim in the corpus against F's 19 %
  (review B1), so a gain concentrated on H over F reads as set familiarity, not piloting.

Also reported, by hand from the banked records: C0's F and reported units at unit seeds 0–6 against D1's banked C0 units
(a harness-drift line), and N1-P − D1 on F at the shared share 0.5 (the corpus-scale contrast; D1's F was its own
population).

## 8. The M4 hazard: any N1-trained model is inflated on H and P72

U = H + F + P72, and two of its panels are generator-adjacent to this corpus (B1: 62 % and 56.5 % of their full sets are
in it, against F's 19 %). The panel-validity rule ("unfamiliar" = ≤ 3 shared sets per TEAM) cannot see set-level
coverage. Therefore:

- **No N1-trained model's U is read as goal (b)** (this arm, or X53-N1) without a per-panel full-set coverage line by
  its training corpora beside it (`goal_b_panel_validity_2026-10-26`'s recommendations get that line first).
- A goal claim by an N1-trained model leans on F and on the sealed `forum216b`. The X54 owner is told before
  `forum216b`'s unsealing is spent on an N1 model.
- This read's own decision is on F alone, so the hazard does not touch §6.

## 9. VOID guards (a VOID seed is never read; re-run at the same pin and seed after the cause is fixed)

1. its K9(b) at arm end is not PASS (`kit k9b`, AM-6's ceilings);
2. its startup log lacks `🧬 [TEAM-MIX] n1_proc_v1@ALL (12000 teams, tier sha d59f44f0b33b) share 0.50 on both sides;
   eval stays on the pool`, or its held-out-panel line names any hit, or it reports any pool copy other than `0 corpus
   team(s) pack identically to a pool team`;
3. its recorded pin is not N1P_PIN (`kit pin-check`), or `kit corpus-check --arm n1p` fails at it;
4. the T54 pool-seed post-spawn check fails (`kit pool-seed-check`);
5. it stopped on a corpus team the env cannot play or label (a `FATAL` naming an `n1_proc_v1` team, ids `corpus_<n>`):
   the corpus is fixed by a rebuild under a new name and registration.

**Not VOID, a HOLD for the Training session:** first-hour steps/s below 1,000 (0.7 × the siblings' ≈ 1,420): pause and
report (review m4; the 12,000-team table's throughput is UNMEASURED at production).
**INCONCLUSIVE** (no class): an arm below the floor (§4), φ̂ undefined, or any `cells_inconclusive` in the record.
`main.reads` writes no record while any cell is short, the REPORTED group's included, so every registered cell is
played before the primary is recorded.

## 10. Predictions (GUESS, registered before any N1-P game)

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F (primary)** | **+1.5 pp** | **[−1.5, +4.5]** |
| Δ_H (reported) | +3.0 | [−1.0, +7.0] |
| Δ_P72 (reported) | +3.5 | [−0.5, +7.5] |
| Δ_H − Δ_F | +1.5 | [−3.0, +6.0] |
| Δ on home72 vs Kakuna | −1.5 | [−5.0, +2.0] |
| **Δ_P (co-primary)** | **−1.5 pp** | **[−3.0, 0.0]** |

The basis: D1's F +4.0 came on its own population and its P −1.61 at the same share. N1's teams are procedural and
set-incoherent (X61 FINDING 6), so less transfer is expected to human-built F than to the generator-adjacent H / P72.
Final classes: **GAIN 0.20 · NULL 0.30 · LOSS 0.03 · UNDECIDED 0.47**; look 1 CONTINUE 0.45; P NON-INFERIOR 0.40.
Goal (b) met by this arm's U: probability 0.05 (and inflated by §8 if it were).

## 11. Cost

| item | GPU-h | CPU lane-h | note |
|---|---:|---:|---|
| look 1: n1p s3501–s3503 | **≈ 12.3** | — | ESTIMATE (§2) |
| look 2 (on CONTINUE): s3504–s3505 | ≈ 8.2 | — | |
| the look-1 read: Kakuna | 0 | ≈ 27 | 6 finals × (1,200 F + 420 reported) = 9,720 games at ≈ 10 s |
| the look-1 read: P | 0 | ≈ 17 | 6 × 3,500 pairs (D1: 8.6 per arm) |
| the look-2 read | 0 | ≈ 45 + 29 | its spec re-plays the look-1 finals (FINDING 7) |

## 12. Validation at this registration (CPU, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15)

- The argv (`kit describe --arm n1p --seed 3501`): [`argv_n1p_s3501.txt`](argv_n1p_s3501.txt).
- `main.checkargs --pin 5f8f6cfa…` on it: **exit 0**, 9 flags accepted by the PINNED parser @5f8f6cfa (so the pin's
  `CORPORA` holds `n1_proc_v1`), 0 unrecognized, 71 inherited from X26's record, the fork arch compatible
  ([`checkargs_n1p_s3501.txt`](checkargs_n1p_s3501.txt)).
- `main.launcher --dry-run --restart-interval-hours 24 --pin-commit 5f8f6cfa… <argv>`
  ([`dryrun_n1p_s3501.txt`](dryrun_n1p_s3501.txt)): **role FORK of `rb_x26_s2601`**, +20,000,000 steps (280,000,170 →
  300,000,170), run dir `rb_x35_n1p_s3501` [would be created], pin `5f8f6cfa` (source `pin_commit`), 12.49 GiB
  REQUIRED vs 387 GiB free, the Showdown deps present. **Its one refusal is the CUDA-visibility preflight**
  (`CUDA_VISIBLE_DEVICES=''` hides every GPU for a `--device cuda` launch), because this unit runs without a lease
  (rule 5): exit 3, "would NOT launch", no other check refused, as D3's. `launch_x35.sh`'s pre-check 6 re-runs it with
  the card visible at GO.
- `kit pin-check --arm n1p` PASS (the side pin + one commit, 4 files, all under `src/utils/team_corpora/`, tree
  `b273cfe1ff11`; 22 side-pinned production runs checked); `kit corpus-check --arm n1p` PASS at the pin (12,000 teams,
  120,000 soak battles) and FAIL at `f4b25238`; `kit stage-check` s3501 ALLOWED, s3504 PENDING
  ([`kit_checks_n1p.txt`](kit_checks_n1p.txt)).

## 13. FINDINGS at registration (standing rule 7)

1. **Two looks make the false-GAIN rate exceed the nominal 2.5 %.** Under a true null the registered rule ends GAIN
   3.5 % (σ = 0), 4.9 % (σ = 1 pp) and 7.4 % (σ = 2 pp) of the time (§4 Power). The rule is D1's, kept as the review
   asked, so this is a property of D1's / D3's reads too.
2. **The read decides on F alone, so its reach is narrow by design.** F is 72 human-built forum teams; a GO says N1 moves
   the plateau's piloting of THEM. H and P72 are reported only (B1), and U is hazarded (§8).
3. **The pin omits the AM-D1-FIX label guard** (`label_path_defects`, a refusal at corpus load). D1's and D3's pins
   carry it, because their lever commits' subtree held it. It is vacuous for this corpus (0 of 12,000 teams carry a
   non-100 level, checked here), so the brief's "corpus files only" pin costs no protection for N1-P. A later corpus
   at this pin would load unguarded.
4. **The startup guard's radius is 4 and it reads `mixes[0]` only** (review m1). Neither is live here: the corpus was
   built and tested at radius 3, and both sides name the one corpus.
5. **The procedural sets are incoherent by construction** (X61 FINDING 6; review m6). At 0.5 per side the trainee
   meets them half the time. A LOSS or NULL cannot separate "breadth is null" from "incoherence costs"; Δ_H − Δ_F (§7)
   is the read that separates them.
6. **The `--debug` smoke with the lever was NOT run** (§3.1). The heavy slots were queued 3–4 deep throughout this
   unit (D3 s3502 is live), and the smoke may only run in the foreground, inside the 10-minute tool bound. What stands
   in for it: the pinned trainer's own startup team phase on the real argv (§2), the soak, and the `--team-mix`
   plumbing tests. The first launch's log is the first end-to-end run (VOID guard 2 reads it).
7. **Look 2's spec re-plays the look-1 finals** (≈ 60 % of look 2's CPU). `main.reads` has no cross-state import for
   the `d1` rule. Kakuna play is deterministic per model and unit seed (X26c), so the re-played counts reproduce; S
   does not (its dice), which is how D1 look 2 was read too.
8. **The `ge_n1p` reader's argv has no validation transcript** (the four readers validated at `dfeb3d92` do). It is the
   same builder with N1-P's final as target, as `ge_d3`.
9. **The held-out registry at the pin is not main's** (`ea3917c2f1e6` at the pin). The corpus manifest stamps main's
   registry of `de269480`; the pin's startup guard re-derived 0 hits on its own registry, which is what the run reads.
10. **The X61 toy registration stands unrun.** If the orchestrator wants the from-step-0 screen as well, it is the
    review's Option T (2 seeds per arm, F primary, a one-sided SCREEN-GO), by a new registration.
