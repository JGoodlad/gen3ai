# X41 / D3 — SET MUTATION at the plateau: PRE-REGISTRATION (2026-10-25)

Written and shipped **before any D3 launch** (no `rb_x35_d3mut_*` run exists; nothing was launched by this unit). The
build is CPU only. Whether and when D3 runs is the orchestrator's GO (the 2026-10-24 ruling queued the build; the X26c
read of 2026-10-25 says its own D3 trigger did not fire, so the GO is a fresh ruling, `design_beat_metamon_path.md`
§6.4, §9).

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170; C0 = X35's plain-continuation control; H = the PRIMARY (below; also the panel `heldout72_hpfix` when
written `heldout72_hpfix`); F = `forum72`; P72 = `p72`; U = the three unfamiliar panels pooled at equal weight.

## 1. Question and mechanism

Does training on our own pool teams with **unfamiliar SETS on familiar species cores** raise our piloting of unfamiliar
teams against Kakuna (goal (b))? D3 is the diversity doc's L5 (a) (§4.5): domain randomisation of the sets, "so that
exact identification fails while human team structure survives". It targets the CORE-stratum deficit (unfamiliar sets
on familiar species already cost ≈ 11 pp: X37a reverse CORE 0.463 vs home72 0.575), which D1's breadth (new human
teams) did not isolate. D1's final (`d1_read/`) put its point gain on its own forum population (+4.0) and none on P72
(+0.1); D3's training teams belong to NO panel's population, so a gain on all three panels is the generalisation reading.

## 2. The arm (kit arm `d3`, `python -m main.x35.kit describe --arm d3 --seed <s> --call reference_population_production.json`)

| item | value |
|---|---|
| start | P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, R's frozen call) |
| the ONE lever | `--team-mix d3_setmut_v1@ALL:0.3 --team-mix-sides both`: with probability 0.3 per episode and side the team is drawn from the set-mutation corpus, else exactly as C0 (the pool with its 10 % sample bias); the in-loop eval stays on the pool |
| everything else | C0's argv token for token (`kit_test.test_d3_is_c0_plus_the_set_mutation_mix_and_nothing_else`): `--behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --seed <s> --run-name rb_x35_d3mut_s<s>`, the recipe inherited from X26's record |
| pin | **D3_PIN `9e289ba44eb0015f6fa9c7815fc2e94f296d8f4f`** (`batch.D3_PIN`, tag `x35-d3-pin-9e289ba4` on origin) = the production pin `f4b25238` + ONE commit confined to `src/utils/team_corpora/` (D1's AM-D1-FIX pattern; `kit pin-check --arm d3` checks parent, paths and tree). The trainer, env core, model, launcher and `data/` are C0's byte for byte; `f4b25238` carries the pinned-data root (`3d6780ee`, `gen3_pinned_data_root_v1`), so the run reads `data/` as of its pin |
| seeds | look 1: 3501–3503; look 2: 3504–3505 only on look 1 = CONTINUE (`batch.D3_STAGES`, `stage-check`) |
| control | **C0 s3501–s3505**, already run at `f4b25238` (each K9(b) PASS): shared, 0 GPU-h |
| GPU | ≈ **12.1 GPU-h at look 1** (3 × ≈ 4.04, D1's measured 3.87–4.31 per seed, `dose/rb_x35_d1wide_*.json`; ESTIMATE) + ≈ 8.1 at look 2 = **≈ 20.2 GPU-h** for the final read |

**The corpus** (`src/utils/team_corpora/d3_setmut_v1/`, built by `python -m utils.team_corpora.build_d3_setmut`,
deterministic, `--check` reproduces the bytes): **2,876 teams** = 719 pool teams × 4 variants (0 exhausted), tier
`ALL` sha256 `625d664a…bcd3`, data sha256 `fa367429…6778`. Each variant keeps the source's six species, abilities and
untouched mons byte for byte, and redraws **k ∈ {1, 2}** mons (1,494 / 1,382) from Smogon usage only
(`agents.gen3_data.priors`): 4 moves by P(move in set), the item by usage, nature + EVs as one usage spread; Hidden Power
typed by `gen3_hp_repair_v1`. Every mutated set is a NEW mon set (`gen3_team_overlap_v1`), so a variant shares
exactly 6 − k sets with its source. Rules: pinned `TeamValidator('gen3ou')` (86 draws refused and redrawn: egg-move /
event combinations, the Baton Pass clauses), no exact pool copy, no held-out panel neighbour (≥ 4 shared sets), no
duplicate, level 100 only; the engine's coverage oracle prefilters the draw and re-checks the corpus (0 unplayable).

**The soak** (`src/utils/team_corpora/d3_setmut_v1/soak_selfcheck.json`, schema `gen3_team_corpus_soak_v2`, every core
label family declared, the emission self-check build): **PASS, 10 rounds, 28,760 battles (every team 10 times in each seat), 951,264 p1 decisions, 0 core errors, 32 stall forfeits, 121.9 s; every one of the 2,876 teams packed by the training teambuilder with 0 drops**.

**Dose (ESTIMATE, D1 doc §4.1's rule: ≈ 570k games in 20M).** Per side ≈ 171k corpus draws, ≈ 59 per variant and
≈ 238 per source core across its 4 variants; each pool team's exact draws fall from ≈ 790 (C0) to ≈ 555. Against P\*'s
≈ 9,300 cumulative pilots per pool team (beat-Metamon §4) that removes ≈ 2.5 % of a team's exact exposure.

## 3. PRIMARY H — D1's rule, verbatim (`main.x35.d1`, `python -m main.x35.readout decide-d1`, arm label `X41`)

- **Cell:** Kakuna ckpt 34, greedy vs greedy, the symmetric unfamiliar panels `heldout72_hpfix`, `forum72`, `p72`
  (`python -m main.x35.kakuna plan --teamsets heldout72_hpfix,forum72,p72` + `run`), 7 role-balanced 20-game units per
  panel per seed final, **unit seeds 0–6** (D1's), so 140 games per panel per final, 420 per final.
- **Statistic:** Δ_H = D3 − C0 of the equal-weight panel mean, Kish n_eff, Wilson per arm, Newcombe method 10, AM-5's
  √max(1, φ̂) widening; 95 % at both looks; the sign rule ≥ 2 of 3 seeds (look 1), ≥ 3 of 5 (look 2).
- **Classes** (rule 8: a bound within 1e-9 of 0 or ±3.0 is UNDECIDED): **GAIN** iff LB > 0 and the sign rule;
  **LOSS** iff UB < 0; **NULL** iff the interval lies inside [−3.0, +3.0]; else **UNDECIDED** (look 1 with Δ̂ > 0 →
  CONTINUE to look 2, final).
- **C0's rows are REUSED** from D1's read (`../x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json`, the same
  finals, panels and unit seeds), on one deterministic condition checked BEFORE any D3 unit is played: C0 s3501's 7
  `heldout72_hpfix` units re-played at the D3 read's harness reproduce the banked units' per-game outcomes exactly
  (X26c showed Kakuna-cell play is deterministic per model and unit seed across harness versions). Any mismatch ⇒
  every C0 unit is re-played at the D3 read's harness (+1,260 / 2,100 games), and the banked rows are not used.

## 4. CO-PRIMARY P — D1's rule, verbatim

S of each seed final against the frozen R (`reference_population_production.json`, content `cef0960e…`), 7 members ×
500 mirrored pairs on the registered `home72` panel (`main.h2h play-many --panel home72`, regime `d4de0fd7c16a2658`,
D1's), `python -m main.x35.d1 p-input`; Δ_P = mean D3 S − mean C0 S, Welch two-sided 97.5 %. **NON-INFERIOR** iff
LB > −2.0 pp. C0's P rows are D1's banked ones (`d1_read/p_home72_results.json`) iff the regime id matches; else C0 is
re-played. **Declared:** the corpus holds mutated versions of the home72 teams (D3 perturbs the pool, home72 included),
so home72 is not "untouched" for D3: P reads what the lever does to the teams we bring, which is the owner's question.

## 5. Guards (D1's) and VOID guards

- **Guards, read only if H is GAIN** (as D1's table): G-F, G-E (the reader `ge_d3`, a fresh best responder of D3
  s3501's final, against the shared `ge_c0`; `main.x35.ge`), G-C (`main.x35.gc`). G-K's home72 / away Kakuna read and
  the dose confound are REPORTED.
- **VOID (a seed is void, never read; re-run at the same pin and seed after the cause is fixed):**
  1. its K9(b) at arm end is not PASS (`kit k9b`, AM-6's ceilings);
  2. its startup log lacks `🧬 [TEAM-MIX] d3_setmut_v1@ALL (2876 teams, tier sha 625d664aff21) share 0.30 on both
     sides; eval stays on the pool`, or its held-out-panel line names any hit, or it reports any pool copy other than
     `0 corpus team(s) pack identically to a pool team`;
  3. its recorded pin is not D3_PIN (`kit pin-check`), or `kit corpus-check --arm d3` fails at it;
  4. the T54 pool-seed post-spawn check fails (`kit pool-seed-check`);
  5. it stopped on a corpus team the env cannot play or label (a `FATAL` naming a `d3m_*` team): the corpus is fixed
     by a rebuild under a new registration, as AM-D1-FIX did.
- **INCONCLUSIVE** (no class): fewer than 140·seeds − 20 counted games on a panel for an arm, or φ̂ undefined (D1's).

## 6. Verdicts (D1's table) and what each licenses

| verdict | rule | consequence for goal (b) |
|---|---|---|
| **ADOPT-CANDIDATE** | H GAIN, P NON-INFERIOR, every guard PASS | D3 joins X53's recipe (beat-Metamon §6.5: "add D3 only if D3 has read GAIN"); X54's read becomes reachable only if D3's own U clears ≈ 0.53 |
| **TRADE** | H GAIN and P inferiority not excluded, or a guard FAILs | the exchange rate (Δ̂_H / −Δ̂_P) goes to the owner; D3 + D1 in one arm is NOT licensed by this read |
| **NULL** | H NULL | exact-set identification on familiar cores is not the binding channel at this dose; the CORE-stratum story (§1) is weakened; the next lever is the orchestrator's |
| **LOSS** | H LOSS | the perturbation hurt unfamiliar piloting; report it and stop the family |
| **UNDECIDED** (final) | otherwise | as D1's: no lever shown; the orchestrator rules |

Nothing is adopted by this rule alone.

## 7. Predictions (GUESS, registered before any D3 game)

- **H:** Δ_H = **+1.5 pp, 80 % [−1.5, +4.5]**. Reasoning: D1's +2.10 [−0.92, +5.11] came at a 0.5 share of new
  human teams, with its gain on its own population; D3 is a 0.3 share and trains on no panel's population, but aims
  at the CORE deficit that every panel carries. X26c's FLAT (Δ +0.08 over 100M) says plain training adds ≈ 0, so any
  D3 gain is the lever's.
- **Per panel (reported, no rule):** no ordering predicted with confidence; if D3 works by CORE-set generalisation the
  gain should be spread over all three panels rather than concentrated on F (D1's pattern). Predicted |F − H| < 3 pp.
- **P:** Δ_P = **−0.8 pp, 80 % [−2.5, +0.9]** (the pool's exact share drops 30 %, smaller than D1's 50 %, and the
  mutated teams keep the pool's cores; D1 read −1.61).
- **Classes at the FINAL look:** GAIN 0.15 · NULL 0.15 · LOSS 0.03 · UNDECIDED 0.67. P NON-INFERIOR with probability
  0.55. Look 1 CONTINUE with probability 0.6.
- **Goal (b) by D3 alone** (D3's own U LB > 0.50 at 2,100 games): probability 0.05.

## 8. Cost and order

GPU ≈ 12.1 GPU-h at look 1, ≈ 20.2 with look 2 (C0 shared). CPU per look-1 read: Kakuna 3 × 420 = 1,260 games
(≈ 3.5 lane-h) + the C0 spot check (140 games), P 3 × 3,500 pairs (≈ 8.6 CPU-h, D1's). The orchestrator places it in the
GPU queue; `launch_x35.sh d3 <seed> --pin <D3_PIN> --lease-owner <owner>` runs every pre-check (2b pin-check `--arm d3`,
2d corpus-check `--arm d3`, 9 stage-check) and launches only with `GO`.

## 9. Validation at this registration (CPU, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15)

- The argv (`kit describe --arm d3 --seed 3501`): [`argv_d3_s3501.txt`](argv_d3_s3501.txt).
- `main.checkargs --pin 9e289ba4…` on that argv: **exit 0**, 9 flags accepted by the PINNED parser @9e289ba4 (so
  the pin's `CORPORA` holds `d3_setmut_v1`), 71 flags inherited from X26's record, the fork arch compatible
  ([`checkargs_d3_s3501.txt`](checkargs_d3_s3501.txt)).
- `main.launcher --dry-run --restart-interval-hours 24 --pin-commit 9e289ba4… <argv>`
  ([`dryrun_d3_s3501.txt`](dryrun_d3_s3501.txt)): role FORK of `rb_x26_s2601`, +20,000,000 steps (280,000,170 →
  300,000,170), run dir `rb_x35_d3mut_s3501` [would be created], 12.48 GiB REQUIRED vs 394.77 GiB free, the
  Showdown deps present. **Its one refusal is the CUDA-visibility preflight** (`CUDA_VISIBLE_DEVICES=''` hides every
  GPU for a `--device cuda` launch), because this unit runs without a lease under rule 5. So it prints "would NOT
  launch". No other check refused. `launch_x35.sh`'s pre-check 6 re-runs the dry-run with the card visible at GO.
- `kit pin-check --arm d3` PASS (the side pin + one commit, 12 files, all under `src/utils/team_corpora/`, tree
  `2156dd46c78e`); `kit corpus-check --arm d3` PASS at the pin and FAIL at `f4b25238`; `kit stage-check` s3501
  ALLOWED, s3504 PENDING ([`kit_checks_d3.txt`](kit_checks_d3.txt)).

## 10. FINDINGS at registration

1. **The design leaves the corpus size open.** §4.5 specifies k, q and the draw, not how many variants an OFFLINE corpus
   holds. Chosen: 4 per pool team (2,876), D1's corpus scale; every variant is drawn ≈ 59 times per side, so the corpus
   is memorisable in principle, as D1's was (F-DV-22). Per-episode resampling (the design's literal "per episode") would
   need a trainer build and a new pin beyond `src/utils/team_corpora/`.
2. **Both sides** (D1's frame and §2.5's self-play coupling); §4.5 names no side.
3. **Ability kept.** §4.5 lists moves, item, EVs and nature; the ability stays the source's. IVs are reset to 31 except
   Hidden Power's.
4. **home72 / away variants are IN the corpus** (§4): a deliberate difference from `w_forum_v1`, whose builder excludes
   the in-pool panels' neighbours.
5. **The soak ran at this checkout's build, not at the pin's** (as D1's did). Checked: `git diff f4b25238 HEAD` over
   `src/rust_env`, `src/rust_sim/src`, `src/utils/teambuilder.py` and `src/utils/team_packing.py` is EMPTY, and over
   `src/utils/rust_env` touches only the build-stamp tool (`stamp.py`), so the env core, its label writers and the
   training teambuilder the soak drove are the pin's by source.
6. **The `ge_d3` reader's argv has no validation transcript** (the four readers validated at `dfeb3d92` do); it is the
   same builder with D3's final as target.
7. **X26c's registered D3-revisit trigger did not fire** (in-house S did not rise); this build was the orchestrator's
   standing queue item, and the GO is a separate ruling.
