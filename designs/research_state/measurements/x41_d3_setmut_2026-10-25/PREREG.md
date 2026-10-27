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

## AMENDMENT AM-D3-ENG (2026-10-26): look 2 is played on ONE engine, E1. No decision-rule change

**Why.** A Kakuna or S game runs on our in-process Rust engine, and the regime id did not name it (REVIEW F F-M2;
`designs/tools/reads.md` §4, "The engine"). Look 1 was played across two engines:
- D3 ON on **E2** (port `806f97bd…`, env `9ecb4249…`): anchor at `2cc864f2`, S at `1ed71c9dd`;
- C0's banked H and P rows on **E1** (port `7d932165…`, env `7c532805…`): D1's read, at `6c428453` / `5e389ddac`.

The determinism guard reproduced 140 / 140 games. That bounds divergence below about 2.1 % of games (95 %), against
look 1's Δ̂ = +1.27 pp. Look 1's CONTINUE stands as recorded. Re-decided by `main.reads` at or after the engine change,
its `import_off` refuses (an import never crosses engines).

**The ruling (orchestrator, 2026-10-26).** Look 2's read plays EVERY game on **E1**, the engine both arms trained on
(`f4b25238`, D3_PIN `9e289ba4`):
1. D3 ON's look-1 units (s3501–s3503, both lanes) are RE-PLAYED on E1.
2. The look-2 units are played on E1.
3. C0's banked rows (E1, `d1_read/read_look2_input.json`) then import cleanly through `import_off`. Its anchor and S
   engines are checked against the read's own groups, and so is the guard group's.

The rule of §3–§6 (D1's, verbatim), the panels, the unit seeds and the look schedule are unchanged.

**What the look-2 reader must do.**
- Play from a pin whose engine is E1: `python -c "from main.reads.engine import engine_at, E1; assert
  engine_at('<pin>') == E1"`.
- No existing commit carries both E1's Rust and `import_off`, which landed at `2cc864f2`, already on E2. So the pin is
  a commit made for this read: `main.reads` at or after the engine change, with `src/rust_sim` and `src/rust_env`
  equal to `f4b25238`'s. It also needs whatever Python column table that core expects (the env core's stamp check
  refuses a mismatched `schema`).
- Register the pin in the look-2 spec.
- Leave look 1's state untouched: its re-played ON units are a new group of the look-2 state, not a repin.

## AMENDMENT AM-D3-L2 (2026-10-27): the look-2 play pin, its identity check, the look-2 spec. Before any look-2 game. No decision-rule change

**The pin: `c4a5c37b6b3749b93157048169e9c2fb348c995a`** (annotated tag `x41-d3-look2-play-c4a5c37b`, on origin;
NOT on main). It carries E1's game under the newest read harness that predates `b0441623e`:
- **Base `a9c09711d` = `b0441623e^`.** `b0441623e` packs the Kakuna peer's teams through Showdown
  (`gen3_peer_showdown_pack_v1`), which makes forum72 a new regime. C0's banked forum72 rows then refuse to import.
- **Restored byte for byte to `f4b25238d`** (the arms' training pin; each tree object equals `f4b25238d`'s):
  `src/rust_sim/`, `src/rust_env/`, `src/agents/model/`, `src/agents/observation/`, `src/agents/inference/`,
  `src/utils/bridge/`, `src/utils/team_loader/`, `src/utils/team_packing.py`, `src/utils/teambuilder.py`,
  `src/utils/gen3_utils.py`. `data/` and `deps/` are the same at both commits.
- **Kept from `a9c09711d`:** the read harness. That is `main.reads` (`import_off`, the engine stamp, the VOID gate,
  `--concurrency`), `main.anchors` and `main.x35.kakuna` (registry weight pins, offline peer, the engine stamp), and
  the heavy-slot, port and lock helpers.

**Why more than the Rust.** AM-D3-ENG asked for E1's Rust plus the column table its core expects. The column table
is unchanged from `f4b25238` to `a9c09711` (`columns.py`, `protocol.py` and `label_inventory.py` are identical, schema
`0a6035e172e514b3`). But between those commits the PYTHON play path also changed what our side plays, and the engine
key cannot see a Python change (`kakuna_determinism_2026-10-27/` §7):
- the damage operator's consumed-item rule (`gen3_op_held_item_v1`, `d8d674b99`): a knocked-off Choice Band or Quick
  Claw;
- the eval-mode load (`141f36f7f`): `nn.MultiheadAttention`'s fused path, logits moved by up to 4.8e-7;
- our packer (`d8d674b99`): Frustration's `Happiness: 0`, and a `Hidden Power: <Type>` line's IVs.

So the pin takes the training pin's bytes for every model, observation, inference, bridge and team-packing file. Our
side plays what C0's bank played. The bank's playing commits `6c428453a` and `5e389ddac` differ from the pin in **0**
of those files.

**Engine = E1, at three levels:**
- `engine_at('c4a5c37b6') == E1` (git objects; also under main's `main.reads.engine`);
- the playing worktree's `main.anchors.server.engine_stamp()` = `port=7d932165eced1c79/90;env=7c53280566d666ae/125`;
- its built artifacts' stamps: `sim_bridge --build-stamp` `src=7d932165eced1c79;nfiles=90`; `rust_env_proc` and the
  cdylib `src=7c53280566d666ae;nfiles=125`, schema `0a6035e172e514b3`. `ffi.load` passes its source, schema and
  data-dir checks. The untracked `Cargo.lock` files are the legacy blobs `676e633e…` / `3d00802e…`, and `cargo build
  --locked` left them unchanged.

**Play path vs the bank** (`main.reads.playpath`): 205 / 201 of 1,683 files differ from `6c428453a` / `5e389ddac`.
All are harness:
- `main.anchors` (weight pins, the engine stamp; the unused `--our-belief-prior-mix` option);
- `main.x35.kakuna` and `main.reads` `__init__` / `supervisors`;
- `mem_cap`, heavy slots, ports, locks;
- the anchors registry (its Kakuna weights are the banked ones) and the panel registries;
- `team_corpora`'s level-100 guard;
- training-only modules in the import closure (`agents.training`, `main.train`, `main.launcher`), which a unit
  imports but does not run;
- the stamp module's CLI, and the rc72 cell files.

**The identity check: PASS, 80 / 80 games identical choice by choice** ([`read/look2_identity.json`](read/look2_identity.json),
by [`read/look2_identity_compare.py`](read/look2_identity_compare.py)). The pin's own queue (`python -m main.x35.kakuna
run`, its `mem_cap` short lane, 95xx port claims, `HF_HUB_OFFLINE=1`) re-played four of C0's banked E1 units from the
bank's model bytes, with captures. The state is `~/gen3ai_archive/x41_d3_read/look2_identity/`. Each was compared with
the banked unit game for game on five levels: the 5-tuple the guard compares, the seed and START (both packed teams),
OUR choices in order, Kakuna's choices in order, and every per-side protocol chunk (timestamps removed).

| re-played unit (bank) | why this unit | 5-tuple | our choices | Kakuna's | protocol bytes | regime id | engine |
|---|---|---|---|---|---|---|---|
| c0_s3501 forum72 u04 (`6c428453a`) | game 8: E1 plays Destiny Bond (a loss in 64), E3 Ice Punch (a win in 105) | 20 / 20 | 20 / 20 | 20 / 20 | 20 / 20 | `2118812c27885de3` = bank | E1 = bank |
| c0_s3502 p72 u01 (`6c428453a`) | game 9: E1 Thunderbolt, E3 Thunder Wave (the divergence the determinism study could not attribute) | 20 / 20 | 20 / 20 | 20 / 20 | 20 / 20 | `b7547f904af56acc` = bank | E1 = bank |
| c0_s3503 forum72 u02 (`6c428453a`) | game 19 after an Endure; game 9's choice that never ran | 20 / 20 | 20 / 20 | 20 / 20 | 20 / 20 | `2118812c27885de3` = bank | E1 = bank |
| c0_s3505 forum72 u00 (`5e389ddac`) | the second bank commit (D1's h_e) | 20 / 20 | 20 / 20 | 20 / 20 | 20 / 20 | `2118812c27885de3` = bank | E1 = bank |

The first three are positive controls. On engine E3 they diverge from the bank (`kakuna_determinism_2026-10-27/`), so a
pin carrying any post-E1 observation semantics would have failed them. The pin plays E1's game in every one, and the
unattributed p72 divergence is gone too. Its cause is therefore among the restored paths. The read's registered guard
(C0 s3501's 7 banked `heldout72_hpfix` units, §3) re-checks this inside the look-2 state.

**Look 2's data (AM-D3-ENG, made concrete).** Spec [`read/read_spec_look2.json`](read/read_spec_look2.json); state
`~/gen3ai_archive/x41_d3_read/look2/`. Look 1's state and record are untouched.
1. **D3 ON s3501–s3505**: every H unit (3 panels × unit seeds 0–6) and every P cell is played at the pin.
   - s3501–s3503 are re-played from the same bytes as look 1 (`pinned` refs, look 1's sha256s).
   - Look 1's E2 rows are not read at look 2. They stay look 1's record, and look 1's CONTINUE stands.
2. **C0 s3501–s3505** are imported from D1's committed `d1_read/read_look2_input.json` (sha256 `8c4884aa…`; E1 in
   both lanes by `LEGACY_BANK_ENGINE`) through `import_off`.
   - The registered regimes are `09f297883701a35a` (anchor) and `d4de0fd7c16a2658` (S). The pin's engine must equal
     the bank's in both lanes.
   - H is pooled only if the guard group re-plays C0 s3501's 7 banked `heldout72_hpfix` units game for game (§3,
     unchanged; banked file `be989806…`).
3. **The rule** is §3–§6 verbatim at look 2 (`look: 2`): 95 %, AM-5's widening, the sign rule ≥ 3 of 5. The look is
   final. The INCONCLUSIVE floor is 140 · 5 − 20 = 680 counted games per panel per arm.
4. **The VOID gate is registered** (`params.void`): each D3 run's AM-6 K9(b) record must read PASS, or `decide`
   refuses. §5's other VOID guards were checked by hand before any look-2 game, as look 1's were. Every seed passes:

   | seed | K9(b) at arm end | `[TEAM-MIX]` banner; held-out hits; pool copies | `pin_history` / corpus-check | pool-seed check | FATAL on a `d3m_*` team |
   |---|---|---|---|---|---|
   | 3501–3503 | PASS (look 1's table) | as look 1 | `9e289ba4…` / PASS | PASS | none |
   | 3504 | PASS (`k9b/rb_x35_d3mut_s3504.json`, `a9c09711d`) | the exact line; 0 hits over 5 panels; `0 corpus team(s) pack identically to a pool team` | `9e289ba4…` / PASS | PASS | none |
   | 3505 | PASS (`k9b/rb_x35_d3mut_s3505.json`, `28711af06`) | as s3504 | `9e289ba4…` / PASS | PASS | none |

   The sources are the X35 launcher journal (`~/.claude/jobs/x35/rb_x35_d3mut_s350{4,5}.launcher.log`,
   `*.go.pin.txt`, `*.go.corpus.txt`, `*.pool_seed_check.txt`), each run's `metadata.json` `pin_history`, and a
   FATAL search of those files and each `launcher_child.full.log`. The only FATAL-bearing lines are the
   flag banners and K9(b)'s per-update AM-6 warnings.
5. **Guards** (G-F, G-E, G-C): `guards_not_read` is registered as at look 1. It applies only to a non-GAIN H. On a
   GAIN the unplayed guards stay INCONCLUSIVE until they are played (§5), so the verdict reads INCONCLUSIVE until
   then. G-E needs a GPU-trained best responder: the orchestrator's call.
6. **Play order and cost.** `main.reads run --lane anchor --concurrency 1` and one `--lane s` job. Both start when
   the H-CTX rc72b read's anchor supervisor exits (the orchestrator's ruling, 2026-10-27). N1-P look 1 keeps its
   K = 2, and the heavy-slot queue interleaves the reads.
   - Anchor: 105 H + 7 guard units = 2,240 games. At look 1's measured 8.5 min a unit, about 16 lane-h, more on a
     contended box.
   - S: 5 × 3,500 mirrored pairs. Look 1 ran 1.5 h a seed, so about 7.5 h in the second heavy slot.
7. **Not at the pin** (FINDING): three harness guards that landed after `b0441623e`.
   - The forfeit-win guard (`bf5f1a8e7`).
   - H5's fail-fast at the first long-tail line, and H21's "any peer fault voids the unit" (`e28318c82`).
   - The choice-level guard comparison (`e28318c82`). The pin's guard compares the 5-tuple; the identity check above
     compared choices.

   At the pin, H5 is covered by `peer_clean`, which drops the whole unit. Every banked unit this read imports was
   played without these guards too, and `e28318c82` found the long-tail line in no counted archived unit. **Registered
   here:** before `record`, every counted unit of the look-2 state has its peer logs scanned for `Force resetting due
   to long-tail error` (H21's rule). A unit with one is moved aside and re-played at its seed before `decide`.

### AM-D3-L2 addendum (2026-10-27): the pre-data review's rulings ([`REVIEW_look2.md`](REVIEW_look2.md), `71d74e5c1`). Before any look-2 game

Every look-2 state file so far was a plan file. That state is moved aside (`look2.void_plan_k9b_<epoch>`) and
re-planned from the spec below, so no look-2 game has been played under the old registration.

1. **B1 (the BLOCKER): the VOID gate reads main's K9(b) records, and the PIN decides.**
   - `params.void.k9b_dir` is now ABSOLUTE:
     `/home/goodlad/dev/gen3ai/designs/research_state/measurements/x35_breakers_stage_2026-10-17/k9b`. s3505's
     record (`28711af06`) is not in the pin's tree, so a repo-relative path read it MISSING at the pin.
   - The spec is [`read/read_spec_look2.json`](read/read_spec_look2.json), sha256 `f67c56d4…` (was `2e98bf74…`; only
     this field changed).
   - **The deciding checkout is the pin's playing worktree** (`c4a5c37b6`, clean): `record` / `decide` run there,
     never from main.
   - Verified with the pin's own code (`rules.void_gate` on this spec): d3_s3501–s3505 all `PASS`,
     `void_failures` = `{}`.
   - `check-spec` at the pin: `play_path_differs_at_head` 0, `sha256_final` true.
2. **M1: NULL is unreachable at look 2.** At 700 games per panel per arm and panel rates of 0.45–0.50, H's
   unwidened half-width is ≥ 3.0 pp (3.006–3.022). Even Δ̂ = 0 exactly reads [−3.015, +3.015] → UNDECIDED. The
   band [−3.0, +3.0] is NOT changed (D1's rule verbatim). **Registered:** the record reports the final 95 % interval
   as a bound on D3's H effect, as a reported line beside the class: "D3's effect on H lies within [LB, UB] pp". An
   UNDECIDED whose interval lies inside [−3.1, +3.1] is reported as "no effect beyond ≈ 3 pp". No class reads it.
3. **M2: look 2 is TERMINAL.** No further D3 units or seeds will ever be pooled with these rows. Any later D3 read is
   a new registration on new seeds. The expected outcome is registered with the review's table: P(GAIN | what is
   known) ≈ 3–7 %, otherwise UNDECIDED with an interval of about ±3 pp. The orchestrator ruled to proceed, with
   the Kakuna lane deferred (item 4).
4. **Schedule (the orchestrator's ruling; it replaces item 6's start condition).**
   - The **S lane** (one job) starts when the H-CTX rc72b read's anchor supervisor exits.
   - The **Kakuna anchor lane** (`--concurrency 1`) is DEFERRED until N1-P look 1's Kakuna lane finishes: its
     supervisor has exited AND either every planned unit there is counted, or no anchor supervisor of that state has
     been alive for 30 min. The latter is logged as "stopped with units left".
   - Both are started by the detached, bounded queue starter, whose log is `<state>/queue_starter.jsonl`.
   - **m7:** a lane that stops at its registered stop with units left is RESUMED (`main.reads resume`) at the pin.
     This is pre-approved.
5. **m1: the guard's ordering.** The 80 / 80 identity check above was run before any D3 game, and it covers §3's
   "checked BEFORE any D3 unit" at look 2. The in-state guard group plays interleaved by unit seed, as the pin's
   runner orders it, and still gates H's pooling at `decide`.
6. **m2 and m3: record-time checks, run BEFORE `record`.** Their outputs are committed with the record.
   - `read/look2_identity_compare.py` on the 7 guard units against the banked units (choice level). **Any choice
     difference is a FINDING that stops pooling:** the banked C0 H rows are not used, and every C0 unit is re-played
     at the pin (§3's mismatch branch).
   - Every counted unit of the state (h and guard) must stamp `model_loader: bare` (the strict loader, restored,
     training mode), as the bank and look 1 did. A unit loaded any other way is moved aside and re-played.
   - **Wording correction to AM-D3-L2:** only `snapshot.py`'s half of the eval-mode change (`141f36f7f`) is restored.
     The pin's `session.foreign_loader` (a harness file) loads in eval mode, which is why the `bare` check is
     required.
   - H21's long-tail scan (item 7 above) runs at the same time.
7. **m5: external state.** The Metamon checkout (`/home/goodlad/dev/metamon`: HEAD, tracked changes, untracked) and
   both conda envs (`metamon`, `gen3ai_torch28`: `conda list --explicit --md5` and `pip freeze --all` digests) are
   snapshotted at each lane's start and at the end. The snapshots go in `<state>/external_snapshots.jsonl` and are
   recorded. Any change between start and end is a FINDING named in the record.
8. **m6: two descriptive rows, no rule reads them.**
   - D3 s3501–s3503 on E1 (this state) against look 1's E2 units, game for game: the engine effect on the ON arm.
   - Look 1's Δ̂ recomputed with its ON rows replaced by their E1 re-plays. AM-D3-ENG's CONTINUE stands whatever it
     reads.
9. **m4 (the S lane's identity) is accepted as declared:** `main.h2h`, `agents/{model,observation,inference}`,
   `rust_env` and the packer are identical to the bank's. **m8:** no H game draws the forum72 Frustration set at unit
   seeds 0–6.
10. **(2026-10-27, the orchestrator's ruling; registered after the S lane started and before any look-2 Kakuna
    game.)** The pin decides and carries no re-derived anchor guards, so a REPORT-only step runs before `record`:
    - main's G14 re-derivation (`src/main/reads/rederive.py`, `29b90161f`: `unit_check` on every counted unit of
      both anchor groups) runs READ-ONLY over the look-2 state from main's checkout;
    - it checks the forced ends (`FORCELOSE`) in the captures, the front end's `disconnected mid-battle` lines, and
      the peers' long-tail lines;
    - its summary is a descriptive row in the record. Counting is unchanged;
    - any HIT on a counted unit STOPS the record: the orchestrator is told before `record` runs.
