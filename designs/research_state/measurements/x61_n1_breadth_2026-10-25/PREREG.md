# X61 — N1 on the toy: Procgen-scale team breadth from step 0 — PREREGISTRATION (2026-10-25)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED.** An adversarial review of this file
comes first; the launch is then the orchestrator's GO. Nothing below may change after the first arm launches except
by a dated amendment that says why and was written before the read.

Abbreviations: pp = percentage points; CI = 95 % interval; LB / UB = its lower / upper bound; H = `heldout72_hpfix`;
F = `forum72`; U_HF = the equal-weight mean of H and F; U = the goal's pooled meter over H, F and P72; S = in-house
strength against the frozen toy reference population R; GUESS / ESTIMATE as in `design_beat_metamon_path.md`.

## 0. The question

Does training on ≥ 10⁴ teams FROM STEP 0 make the policy pilot teams it has never seen better, against an opponent it
did not train with? The survey's diagnosis (`designs/research_state/literature/sota_survey_2026-10-26.md` §0, §5 N1):
every system that generalises across teams trained on 10⁴–10⁵ of them (Metamon: 11k procedural + 107k replay teams;
random-battle bots: a fresh team every game), and Procgen's held-out gap closes at ≈ 10⁴ levels. We have trained on
719 teams, and every breadth test so far (D1, D3, X52) added breadth at a PLATEAU, at 10³ teams.

## 1. The lever and its corpus

- **The lever:** `--team-mix n1_proc_v1@ALL:0.7 --team-mix-sides both` (`gen3_team_mix_v1`, built for D1; default
  absent = byte-identical). On each side, every fresh team draw is a corpus team with probability 0.7, else the
  side's usual pool draw, so the pool keeps the survey's minority share 0.3 (trainee: 70 % corpus / 27 % pool / 3 %
  sample-bias). Eval stays on the pool (`designs/training/team_curriculum.md`).
- **The corpus `n1_proc_v1`** (`python -m utils.team_corpora.build_n1_proc [--check]`, `gen3_n1_procedural_team_v1`,
  BUILD_SEED 6101; manifest `src/utils/team_corpora/n1_proc_v1/manifest.json`, data sha256 `99e3425a…0bad6`, tier
  ALL sha256 `d59f44f0b33b…`):
  - **12,000 whole teams from Smogon usage only** (`agents.gen3_data.priors`): the lead by usage, each next mon by
    the summed Smogon teammate conditionals of the chosen mons; per mon 4 moves by P(move in set) (typed Hidden
    Powers as one candidate), the item, the ability, and nature + EVs as one usage spread; Hidden Power typed by
    `gen3_hp_repair_v1`; level 100. Prefix tiers K1 (1,000) ⊂ K3 (3,000) ⊂ K10 (10,000) ⊂ ALL (12,000), for a later
    dose read on one draw.
  - **Not human data.** Smogon aggregates only (the priors rule; `human_data_architecture.md` §2 "human
    AGGREGATES"): no replay, no human-built team, no human decision.
  - **Legal:** the pinned `TeamValidator('gen3ou')` (1,628 of 13,689 corpus candidates refused: 864 egg-move
    combinations, 432 Baton Pass clauses, 320 incompatible event moves, 12 other), Species Clause by construction,
    the engine's coverage oracle as a prefilter (42 moves and 4 items never drawn; the largest dropped move mass is
    Ninetales' 0.12 per slot) and a re-check of every id the output carries (0 unplayable).
  - **Breadth:** 11,543 distinct species sextets, 15,655 distinct mon sets (9,372 seen once); 190 of 216 eligible
    species drawn (the 26 never drawn are each below 0.1 % usage); team frequency vs Smogon usage % Pearson 0.997
    (Tyranitar 0.437 vs its usage 0.518, the largest gap: Species Clause and the validator thin it).
  - **Distance to the panels (REVIEW A's revealed-fit rule, B2):** a candidate sharing ≥ 3 mon sets with any team
    of `heldout72`, `heldout72_hpfix`, `forum72`, `p72` or the SEALED `forum216b` (and, for the corpus, `n1p72`) is
    refused: 62 corpus candidates were. The kept corpus's max shared mon sets with any held-out panel team: 0 → 310
    teams, 1 → 8,952, 2 → 2,738 (the panels' own "unfamiliar" region, ≤ 2). With the pool: 0 → 1,689, 1 → 7,952,
    2 → 2,319, 3 → 40; 0 exact pool copies.
  - **The soak** (`gen3_team_corpus_soak_v2`: the label path with every core label family, the selfcheck build,
    every team 10 times in each seat, in two 5-round chunks merged by `soak merge`): **PASS**, 120,000 battles,
    4,457,859 p1 decisions, 0 core errors, 373 stall forfeits (0.3 %), median 25 turns, 797.9 s; every team packed
    by the training teambuilder with 0 drops (`n1_proc_v1/soak_selfcheck.json`).
  - **The trainer's startup on it** (the toy_n1 argv parsed by the trainer's parser, `build_matchup_and_opponents`
    with the real Node validator, CPU): the matchup echo names the corpus on both sides, and the startup guard prints
    `🧬 [TEAM-MIX] n1_proc_v1@ALL (12000 teams, tier sha d59f44f0b33b) share 0.70 on both sides`, 0 held-out panel
    hits over 5 panels, 0 corpus teams packing as a pool team, ≈ 20 s for the whole phase. The full `--debug`
    smoke did not run (FINDING 10).
- **The held-out PROCEDURAL panel `n1p72`** (`teams_n1p72/`, `n1p72_manifest.json`): 72 teams of the SAME generator,
  stream `panel`, drawn BEFORE the corpus; the corpus excludes its ≥ 3-fit neighbours. It is Procgen's "test levels":
  generalisation to new draws of the training distribution. It is NOT in `heldout_panels.json` (FINDING 3).

## 2. Arms, pin, seeds, steps

| arm | run | argv (`python -m main.x61.arms describe --arm <arm> --seed <s>`) | look 1 | look 2 (on CONTINUE) |
|---|---|---|---|---|
| **toy_c0f** — the FRESH toy C0 | `rb_x61t_c0f_s<seed>` | the toy recipe from step 0, the pool only | s6101 | s6102 |
| **toy_n1** — N1 | `rb_x61t_n1_s<seed>` | toy_c0f's argv token for token + the lever | s6101 | s6102 |

- **The toy recipe** is `rb_toy_ent_s4002`'s `original_command` (the T-ENT toy, the run behind T\*) with its run
  tokens re-typed and the 17 flags the 2026-10-15 legacy deletions removed dropped. Every one of the 17 was the OFF
  value of a damage-operator family that `--no-damage-op` makes vacuous (`designs/deleted_flags.md` U1 / U3), so the
  drop changes no module the toy builds. Plus `--behaviour-check warn` under X34's hand-read ceilings
  (`main.x35.kit.K9B_TOY_CEILINGS`), the orchestrator's C3 ruling for toy arms (2026-10-19); both arms carry it.
  `main.x61.arms_test` pins the derivation and that the two arms differ by the lever alone.
- **Not the old toy C0.** X35's toy C0 (`rb_x35t_c0_*`, forks of T\* at `4609c155`, fp64 τ) is a different pin, a
  fork and another step range; it is NOT this control. Both arms here are fresh at ONE pin.
- **The pin: `de2694804a67086be56c7ae7d8f827f09ec604fa`** (`main.x61.arms.PIN`; annotated tag `x61-n1-pin-de269480`
  on origin) = the commit that landed the corpus, so the trainer's `utils.team_corpora` at the pin holds
  `n1_proc_v1`. It carries the pinned-data root (`gen3_pinned_data_root_v1`). Pin check (a detached worktree of the
  pin, `python -m main.x61.arms check --tree`): `n1_proc_v1` ALL, n = 12,000, data sha256 `99e3425a…0bad6`, tier
  sha256 `d59f44f0…5314f`, manifest sha256 `4db822ba…e067`.
- **Steps:** `--steps 100000000` each, `--restart-interval-hours 6`, checkpoints every 1M. The read uses each run's
  first checkpoint at ≥ 100,000,000 steps (the final).
- **Paired dice:** the same seed per pair. Look 2 adds seed 6102 to BOTH arms.
- **N4 (a wider network from step 0) is NOT an arm** (FINDING 1).
- **The validation transcripts at the pin** (CPU, `CUDA_VISIBLE_DEVICES=""`, run from a checkout AT the pin, so
  `checkargs` reads the pin's own parser): `argv_<arm>_s6101.txt`; `checkargs_<arm>_s6101.txt`: **exit 0** for both
  arms (46 / 50 flags, 0 unrecognized, 2 launcher-owned; K9(b) `warn`, typed); `dryrun_<arm>_s6101.txt`: role FRESH,
  pin `de269480` (source `pin_commit`), `--steps 100,000,000` from 0, desktop-GPU / disk (20.35 GiB required, 380 free)
  / Showdown deps ✓, refused ONLY by the CUDA-visibility preflight, because this unit hides the GPU (rule 5): exit 3,
  as D3's. The two argvs differ by `--team-mix n1_proc_v1@ALL:0.7 --team-mix-sides both` and the run name alone.

## 3. Pre-launch checks (a failure is a HOLD, never a launch)

1. This PREREG has passed an adversarial review (the orchestrator dispatches it).
2. The tag `x61-n1-pin-de269480` is on origin and dereferences to `main.x61.arms.PIN`; `python -m main.x61.arms check
   --tree <a detached worktree of the pin>` prints `n1_proc_v1` ALL, n = 12,000, data sha256 `99e3425a…`: DONE
   2026-10-25 (§2); re-run at launch.
3. `checkargs` exit 0 and `launcher --dry-run` "would launch" for both arms, re-run with the GPU visible at launch.
4. The committed `soak_selfcheck.json` reads PASS for this corpus's data sha256 (10 rounds, in two 5-round chunks):
   DONE 2026-10-25 (§1).
5. **K0, the toy-at-100M calibration (CPU, ≈ 2.4 lane-h, before either launch).** The three existing toy seeds at
   100M (`rb_toy_ent_s4001b` @100,000,169, `rb_toy_ent_s4002` @100,000,118, `rb_toy_ent_s4003` @100,000,043;
   pin `3899277a`, read-only copies) each play 10 units (200 games) on H and on F against Kakuna (§4's instrument).
   It answers two things the read needs: (a) whether the toy at 100M plays Kakuna at a level a difference can be read
   at (X51 shows the toy CAN: T\* at 290M scored 0.493 on home72 and 0.458 on F); (b) σ̂_s, the between-seed SD of a
   100M toy's U_HF (the sample SD of the three seeds' U_HF, minus their mean binomial variance, floored at 0).
   **HOLD rule:** if the three seeds' pooled U_HF < 0.15, the toy read is not resolvable near the floor: X61 is not
   launched as registered; the fallback is the in-house unfamiliar meter (§6 S7) on the toy plus a production
   follow-up, by a new registration.
6. No training run in `models/` uses the names `rb_x61t_*`.

## 4. The read: instruments and cells

- **Kakuna cells:** `python -m main.x35.kakuna plan --finals <finals.json> --teamsets <cells> --unit-seeds <range>`
  then `run` (`metamon:Kakuna` ckpt 34, greedy vs greedy, our side greedy on CPU, 20-game role-balanced units,
  symmetric cells on each panel's committed config, the regime and counting rule of X51; rule 5b's cores, one lane,
  ≤ 45-min chunks under `mem_cap.sh`).
- **Per final, look 1:** H and F 30 units each (unit seeds 0–29, 600 games per panel); P72 and home72 20 units each
  (seeds 0–19, 400 games). Look 2 plays the same cells for the s6102 finals.
- **In-house:** `main.h2h play-many` mirrored pairs (§6 S6, S7).
- **Counting:** a unit counts iff `main.x35.kakuna`'s rule passes (OK, n = 20, regime verified, both content digests,
  no asymmetry, peer clean). A timeout is never an outcome; > 25 % of an arm's planned units uncounted ⇒
  INCONCLUSIVE.

## 5. The PRIMARY and the decision rule

**Primary: Δ_HF = U_HF(toy_n1) − U_HF(toy_c0f), in pp.** Per arm a and panel k ∈ {H, F}: p_ak = wins / counted
games (ties are non-wins), pooled over the arm's look seeds. p̄_a = (p_aH + p_aF) / 2; V_a = (1/4) Σ_k p_ak (1 − p_ak) /
n_ak; n_eff,a = p̄_a (1 − p̄_a) / V_a; each arm's Wilson interval at (p̄_a, n_eff,a) and Δ_HF's Newcombe method-10
hybrid of the two (`main.x35.d1.newcombe_strat` with K = 2, z = `main.anchors.results.Z95`).

**Why P72 is NOT in the primary.** N1 trains on P72's POPULATION: P72 is `ou_random_teams.js --emit 200 --seed 3721`,
a sibling generator over the same Smogon tables. A gain on P72 is partly "learned the generator", the procedural
twin of `human_data_architecture.md`'s Q-H13. U (the goal's meter) and P72 are reported (§6), never decided on.

**The seed widening (one seed per arm at look 1).** The pooled counts treat a seed as a binomial sample, which hides
run-to-run variance (D1's review F-RV-5: 8–21 % false GAIN at a seed SD of 2–4 pp). Each Newcombe half-width is
multiplied by √φ, φ = 1 + 2 σ_s² / (V_N1 + V_C0f) · (1 / m), where m is the number of seeds per arm in the look and
σ_s = max(σ̂_s from K0, **2.0 pp**) (a declared floor at the low end of F-RV-5's range; K0's three seeds cannot rule a
smaller spread in). At look 2, σ_s = max(that value, D1's overdispersion estimate from the look's own seeds,
`main.x35.d1.overdispersion`, expressed as an SD). Every class reads the WIDENED interval; the unwidened one is
recorded beside it.

**Classes**, in this order, on the widened interval of Δ_HF:

| class | rule | what it licenses |
|---|---|---|
| **GO** | LB > 0 AND Δ̂ ≥ +4.0 | register the production follow-up (a fresh `--arch production` arm with this lever from step 0 and a 100M look, `design_beat_metamon_path.md` §6.5b): breadth at 10⁴ is the lever the field says it is |
| **KILL** | UB < +2.0 | breadth-at-scale by procedural teams does not move the toy: no production follow-up on this family; the human-repaired arm (§10) is a different population and is not killed by this |
| **BELOW-GO** | +2.0 ≤ UB < +4.0 | the 4-pp effect goal (b) needs is excluded on the toy: no production follow-up on this evidence |
| **UNDECIDED** | otherwise | at look 1 with Δ̂ > 0: CONTINUE to look 2 (s6102, both arms; ≈ 12 more GPU-h) on the orchestrator's GO; look 2 is final; an UNDECIDED final licenses nothing alone |

RULE 8: a bound within 1e-9 of 0, +2.0 or +4.0, or Δ̂ within 1e-9 of +4.0 or of 0, makes the class UNDECIDED (and
"Δ̂ > 0" false). A missing cell, a cell below 90 % of its planned games, or zero variance in an arm: INCONCLUSIVE.

**Power (ESTIMATE, at the σ_s floor).** At look 1 the widened half-width is ≈ 6.8 pp (binomial ≈ 4.0 pp × √2.9), so
GO needs Δ̂ ≳ +6.8; at look 2 ≈ 4.8 pp. A true +5 pp reaches GO with probability ≈ 0.30 at look 1 and ≈ 0.53 by the
final; a true +8 pp ≈ 0.63 and ≈ 0.90. **The design resolves a large N1 effect, not a modest one** (FINDING 2).

## 6. Secondaries (REPORTED; no rule reads them)

- **S1. U** (H, F, P72 equal weight; D1's H statistic), the goal's meter, per arm and N1 − C0f, with U − H (F + P72)
  beside it as the panel-validity read's sensitivity line (`goal_b_panel_validity_2026-10-26/`).
- **S2. P72 alone**, the sibling-generator procedural panel.
- **S3. n1p72**, the own-generator held-out procedural panel, 400 games per final, IF it is registered before the
  read (FINDING 3); else not read.
- **S4. home72** (400 per final): the in-pool cost (VGC-Bench's per-team cost), and the familiarity premium
  home72 − U_HF per arm, with its N1 − C0f difference (the survey's secondary).
- **S5. The training curves:** the bot evals and the self-play pool win rates, both arms (the toy's own logs).
- **S6. In-house S** of each final against the frozen toy R (`reference_population_toy.json`), mirrored pairs on
  `home72` (`main.h2h play-many --panel home72`, ≥ 500 pairs per member), `main.x35.readout.population_score`;
  Δ_S = N1 − C0f.
- **S7. The in-house unfamiliar-team meter:** toy_n1 vs toy_c0f head to head, 500 mirrored pairs each on `--panel
  forum72`, `heldout72_hpfix` and `p72` (both pilot the same unseen teams). X37b found in-house meters blind to P\*'s
  piloting deficit, so this is reported beside the Kakuna read, never instead of it.

## 7. Predictions (GUESS, registered before any run)

| quantity | point | 80 % interval |
|---|---|---|
| U_HF(toy_c0f) at 100M | 0.40 | [0.33, 0.47] |
| **Δ_HF (primary)** | **+3.0 pp** | **[−2.0, +8.0]** |
| Δ on P72 | +4.0 | [−1.0, +9.0] |
| Δ on n1p72 (if read) | +7.0 | [+1.0, +13.0] |
| Δ on home72 | −2.0 | [−6.0, +2.0] |
| Δ_S (in-house vs R) | −2.0 | [−5.0, +1.0] |
| the familiarity premium home72 − U_HF | C0f ≈ +5 pp; N1 smaller by ≈ 4 | — |

**Final class probabilities** (after look 2 where it runs): GO 0.20, BELOW-GO 0.25, KILL 0.05, UNDECIDED 0.50.
The basis: the survey's +3 to +8 pp on U for production, halved in logits for the toy's smaller familiarity premium
(X51: T\*'s home72 → F drop is 3.5 pp, P\*'s 6.8), against D1's +2.1 at 10³ teams on a plateau.

## 8. VOID guards (a VOID arm is re-run, never read)

- **V-PIN.** Each run's `metadata.json` pin is `main.x61.arms.PIN`.
- **V-ARGV.** Each run's `original_command` after the entry point equals `build_argv(arm, seed)` exactly
  (`main.lineage`).
- **V-MIX.** toy_n1's log prints `🧬 [TEAM-MIX] n1_proc_v1@ALL (12000 teams, tier sha d59f44f0b33b…) share 0.70 on
  both sides` and 0 pool copies, and its MatchupSpec record's mix block equals the corpus's identity; toy_c0f's log
  has no TEAM-MIX line and its spec has no mix block.
- **V-STEPS.** Both finals at ≥ 100,000,000 steps; crash restarts are allowed and counted; a run that resumed from a
  checkpoint of the OTHER arm is VOID.
- **V-FAULT.** A core error or crash loop naming a corpus team VOIDs toy_n1 (fix: rebuild the corpus under a new
  name, re-soak, amend this file before the re-run).
- **V-K9.** A K9(b) ceiling crossing (judged max ≥ 1e-3, p99 ≥ 1e-4, a non-finite value; read by `main.ops.tb_read`
  at each restart and at the end) STOPs that arm: VOID until investigated.
- **V-KAKUNA.** Both arms' cells share one regime stamp (Kakuna ckpt 34, greedy, the harness commit, each panel
  config's sha256); any difference VOIDs the read (not the runs).
- **V-DOSE (reported, a guard only where measurable).** toy_n1's realised trainee corpus share, from the per-team
  tracker (corpus draws carry no pool index), must lie in [0.60, 0.80] if the tracker's counts can be read; else it is
  recorded UNVERIFIED.

## 9. Cost

| item | GPU-h | CPU lane-h | note |
|---|---:|---:|---|
| look 1: toy_c0f s6101 + toy_n1 s6101 to 100M | **≈ 12.0** | — | ESTIMATE: `rb_toy_ent_s4002` ran 2M → 100M in 5.57 h wall on 2026-10-15 at its pin (≈ 4,880 steps/s); today's pin's toy is UNMEASURED; the corpus adds startup validation only |
| look 2 (on CONTINUE): s6102 both arms | ≈ 12.0 | — | |
| K0 calibration | 0 | ≈ 2.4 | 1,200 games at the toy's 7.1 s per game (X51) |
| the Kakuna read, look 1 | 0 | ≈ 7.9 | 2 × 2,000 games; + ≈ 1.6 if n1p72 is read |
| in-house S6 + S7 | 0 | UNVERIFIED (≈ 2–4) | |
| **look 1 in all** | **≈ 12** | **≈ 12–15** | the survey estimated ≈ 13 toy GPU-h for two arms |

## 10. The LATER, separately-toggled human-derived arm (specified, NOT BUILT)

The survey's corpus mixes procedural teams with forum teams and Hidden-Power-repaired REPLAY teams. A replay team is
HUMAN DATA under the owner's rule (`human_data_architecture.md`): it must be its own corpus, its own toggle, and cite
its snapshot. This unit builds only the procedural corpus. The human arm, when it is built:

- **The corpus `n1_hrep_v1`** (a separate `utils.team_corpora` entry; never merged into `n1_proc_v1`): FULL-SIX sides
  of ladder replays at actor rating ≥ 1500 from ONE registered snapshot (`hr_spec_v1` or `hr_arch_2026-10-24`), the
  snapshot id and its manifest sha256 stamped in the corpus manifest with `"human_data": true`; the panel exclusion
  at ≥ 3 MOVE-REVEALED fits BEFORE any fill (REVIEW A B2), against every held-out panel; REVIEW D's information floor
  (D-M7); hidden facts filled from Smogon usage (the C-GAME fill) and Hidden Power by `gen3_hp_repair_v1`; level 100;
  the pinned validator; the label-path soak.
- **The switch:** the arm names the corpus (`--team-mix n1_hrep_v1…`), so it is off unless typed, and it is listed in
  the human-data registry once that registry is built (`human_data_architecture.md` §6, PROPOSED).
- **The arm:** `toy_n1h` = toy_c0f's argv + `--team-mix n1_hrep_v1@ALL:0.7 --team-mix-sides both`, read by §5's rule
  against toy_c0f AND toy_n1 at the same pin and seeds (a mixture of both corpora needs a multi-corpus `--team-mix`,
  a build). Its read must say that H is the human ladder's population (Q-H13): a gain on H alone is "learned the
  ladder", so F, P72 and n1p72 carry the transfer claim.
- **Prerequisites:** the human-data architecture's re-review, and this read (an N1 KILL does not kill it: a
  different population).

## 11. FINDINGS (standing rule 7)

1. **N4 cannot be a third arm today.** The trunk width is the architecture constant `D_MODEL` = 128
   (`agents/model/arch_constants.py`) with no flag, so a 2×-wide toy needs a width flag first (a build: the constant
   feeds many module widths and the checkpoint loader). Out of this unit's scope; the survey's ≈ 10–15 toy GPU-h for
   N4 assumed the flag. A depth arm (`--trunk-layers 4` from step 0) is launchable but is not width and was not
   registered here.
2. **The read is underpowered for a modest effect** (§5 Power). One seed per arm cannot see run-to-run spread, so
   the interval is widened by a declared floor; a true +5 pp GOes about half the time even after look 2. Two seeds
   per arm at look 1 (≈ 24 GPU-h) would be the cleaner design; the brief's ≈ 13 GPU-h budget chose one.
3. **`n1p72` is not registered in `heldout_panels.json`.** Registering a panel changes the registry's sha256, which
   `w_forum_v1`'s and `d3_setmut_v1`'s manifests stamp and their tests pin, so registration means re-stamping those two
   manifests (their data unchanged) while D3 is LIVE at its own pin (main's copy is not what D3 trains on, but the
   change deserves its own unit). Until then `main.x35.kakuna` and `main.h2h --panel` cannot play it, so S3 is
   conditional. The corpus already excludes its ≥ 3-fit neighbours, so registration later changes no training data.
4. **The registry's own radius is 4; this corpus used 3.** The `--team-mix` startup guard checks 4, so it cannot
   see a 3-fit neighbour; the build and `n1_proc_test.py` enforce 3. A future corpus built at 4 would pass the guard
   and break REVIEW A's rule silently.
5. **P72 and the corpus share their source tables** (§5): Metamon's own 11k `paper_variety` teams were usage-generated
   too, and whether Kakuna trained on them is UNVERIFIED (`design_diversity_teams_opponents.md` §5), so Kakuna's own
   familiarity with procedural teams is uncontrolled on P72 and n1p72.
6. **Procedural sets are incoherent by construction.** Moves, item and spread are drawn independently given the
   species (D3's rule; e.g. a Brave 252 Atk Claydol with Rapid Spin and Shadow Ball). P72's teams rate ≈ −146 Elo
   against forum teams (TSR). The survey's "the policy may learn to pilot bad teams" risk is live; S4 / S6 measure the
   in-pool cost.
7. **38.5 % of drawn sets carry Hidden Power** (27,721 of 72,000; D3's 39 %), because the 16 typed HP keys' summed
   usage mass is drawn as one candidate. It matches Smogon's per-species P(HP in set); the type comes from the repair
   rule, not the set's own coverage logic.
8. **The toy's architecture at the new pin is not T-ENT's bit for bit.** The recipe is T-ENT's command, but the code
   between `3899277a` and the pin moved (config versions, bf16 trunk); the parameter count at the pin is printed by
   the child at startup, not by `--dry-run` (UNVERIFIED here). Both arms share it, so the contrast is clean; K0's
   seeds are the OLD toy, so σ̂_s and the 100M level carry that difference (the same caveat as X51's T\* cells).
9. **`--behaviour-check warn`** follows C3 for toy arms; a fresh run's K9(b) under `fatal` is not what any toy has
   run, and a fatal stop would cost the arm's GPU hours. The ceilings are read by hand (V-K9).
10. **The trainer's `--debug --steps 10000` smoke with the lever was NOT run.** The box's two heavy slots were
    queued 9–12 deep for the whole unit, and the smoke may only run in the foreground (root rule 3), where the
    10-minute tool bound expired in the queue. What stands in for it: the trainer's own Phase 2 on the real argv
    (§1), the soak (the env core with the 12,000-team table, the label path), and the existing `--team-mix` plumbing
    tests (`main/train/team_mix_setup_test.py`). The first launch's log is the first end-to-end run (V-MIX reads it).
11. **The toy has no belief stack** (`--no-item-belief`, `--move-belief-mode off`, …), so X61 tests breadth on the
    PILOTING channel alone. The 2026-10-26 architecture review found production's belief deltas memorise the pool
    (FACING; `model/generalisation_review_2026-10-26.md`); a production follow-up carries that channel too, and a toy
    GO does not say what breadth does to it.
