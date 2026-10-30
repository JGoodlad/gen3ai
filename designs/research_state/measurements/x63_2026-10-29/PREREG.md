# X63 — a long breadth continuation of the plateau (X63-U against X26c and X63-C): PRE-REGISTRATION (2026-10-29)

**Status: REGISTERED before any X63-U launch, any X63 game and any read. NOT LAUNCHED** (no `rb_x63u_*` run exists).
The launch is the Training session's, through the arm chain (kit arm `x63u`), on the orchestrator's GO; the chain file
is a DRAFT, not armed (§10). Nothing below changes after the arm launches except by a dated amendment, written before
the read, that says why. The design and its reasons are
[`../../../endstate/design_x63_long_breadth_continuation.md`](../../../endstate/design_x63_long_breadth_continuation.md)
(revised in place 2026-10-29); this file wins any disagreement with it.

**It also registers when X63-C is read.** X63-C's PREREG
([`../x63c_control_replicate_2026-10-28/PREREG.md`](../x63c_control_replicate_2026-10-28/PREREG.md)) defers its
games, engine and cell to "the cell and engine X63's read registers"; §5 here is that registration.

Abbreviations follow the design: pp; LB / UB; X26@380M (sha256 `7e0ebe942eaa…`); X26c = `rb_x26c_s2601`; X63-C =
`rb_x63c_s6302`; X63-U = `rb_x63u_s<seed>`; F = `forum72`; H = `heldout72_hpfix`; P72 = `p72`; home72; S; σ; σ̂_CC.

## 1. Question

Does continuing the plateau network (X26@380M) for +100M steps with 80 % of each side's teams drawn uniformly from the
17,876-team union `x63_union_v1` make it score higher on F against Kakuna than the same network continued +100M on
the pool alone (X26c, X63-C)? The lever is "a long union continuation at share 0.8 under the production KL
controller". It is NOT a dose read and no verdict names dose (design §4, review M2).

## 2. The arm (kit arm `x63u`: `python -m main.x35.kit describe --arm x63u --seed 2601 --call designs/research_state/measurements/x35_breakers_stage_2026-10-17/reference_population_production.json`)

| field | value |
|---|---|
| run | `rb_x63u_s2601` (Stage A). Stage B `rb_x63u_s6302`, Stage C `rb_x63u_s6303` + `rb_x63c_s6303`: each added to the kit by a dated amendment after its gate (§7) |
| start | X26@380M, `models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip`, sha256 `7e0ebe942eaaac69f1a5ef123d236fbb69f1e7a9fe32011f4d7e0e833dfbf6fd` (the kit's registered start node, X63-C's; `describe` refuses a file that differs) |
| the ONE lever | `--team-mix x63_union_v1@ALL:0.8 --team-mix-sides both` |
| everything else | X26c's recorded `original_command` token for token, run name changed: `--model <X26@380M> --behaviour-check warn --steps 480000195 --checkpoint-every-steps 1000000 --device cuda --team-mix x63_union_v1@ALL:0.8 --team-mix-sides both --seed 2601 --run-name rb_x63u_s2601`, with the launcher's `--restart-interval-hours 24 --pin-commit <X63U_PIN>` |
| pin | **X63U_PIN** `39eaa326736376a5ec3dde41af4afbe8a76a9015` (`batch.X63U_PIN`, amendment label `AM-X63U`) = the production pin `f4b25238def57900c0305bb2a3a5a8f9edaef285` + ONE commit with four files, all under `src/utils/team_corpora/`: `x63_union_v1/{manifest.json, soak_selfcheck.json, teams.jsonl.gz}` and the `CORPORA` line, byte for byte main's. Corpus tree `ca0845cfcb22f70e7a808871f57cff480ad621db` (`batch.X63U_CORPORA_TREE`). Tag `x63u-pin-39eaa326` on origin; local branch `x63u-pin` |
| not in the pin | the builders, the components' own corpora, the AM-D1-FIX label guard (vacuous: every union team is level 100, checked by main's `load_dir`), the 10-26 / 10-27 GIGO fixes. The trainer, env core, model, launcher and pinned data are X26c's and X63-C's byte for byte |
| control | X26c (`final_model.zip` 480,081,209, sha256 `cba7a1f835930e618871cfcafcc78d95320bd0c3b20210d4f423f0b4ff72fc4c`) and X63-C (`final_model.zip` 480,080,819, sha256 `e778c4a46e3bade7bfcf4d439ee85a0a80c16e5c4bb5dbeecb2e64b343826f0b`), both trained at `f4b25238`, both arm-end K9(b) PASS: 0 GPU-h |
| GPU | ≈ 19.9 GPU-h (X26c 19.92 at 1,402 steps/s; X63-C 20.7 h at 1,348 steps/s) |
| thin | the planner's (its 10M nodes and the final are kept, so 400M / 430M / 480M survive) |

Transcripts at this registration (CPU, card hidden): [`argv_x63u_s2601.txt`](argv_x63u_s2601.txt),
[`checkargs_x63u_s2601.txt`](checkargs_x63u_s2601.txt) (against X26c's record: the four mix tokens and the run name
only), [`dryrun_x63u_s2601.txt`](dryrun_x63u_s2601.txt) (`FORK of …/rb_x26_s2601`, +100,000,000 steps; the only
failing check is the hidden card), [`kit_checks_x63u.txt`](kit_checks_x63u.txt) (describe, pin-check, pool-check
against X26c's recorded seed, stage-check).

## 3. The corpus `x63_union_v1` (built and soaked at this registration)

`python -m utils.team_corpora.build_x63_union` (deterministic; `--check` rebuilds and compares). Components, through
the verifying reader at their registered tiers, in this order:

| component | tier | teams | tier sha256 | share of all draws |
|---|---|---:|---|---:|
| `x62_recomb_v1` (built by this unit: X62's G0, `build_x62_recomb`, X62 design §2 exactly) | ALL | 12,000 | `f20326386208…` | 0.537 |
| `n1_proc_v1` | K3 | 3,000 | `8335d302a47e…` | 0.134 |
| `d3_setmut_v1` | ALL | 2,876 | `625d664aff21…` | 0.129 |
| the pool (trainee: 0.18 pool + 0.02 sample teams; opponent: the uniform pool) | | 719 | | 0.200 |

Union tier ALL: 17,876 teams, tier sha256 `f324b6b10fcae17567a285d7ecdf4ff98e439d403a9d3a1131568b3fbc7d3f5e`, data sha256
`f5cdc121ab6d8ec9b13dff60570183add14f30379ddaa68863d36529007128a7`. Both builders reproduce byte for byte under
`--check`. `x62_recomb_v1`: data sha256 `60a4d2e02b68…`, tiers K1 `4e613d9f575b…` ⊂ K3 `2e0a595c5c76…` ⊂ ALL
`f20326386208…`.

**Measured at the build** (the manifest; [`corpus/`](corpus/)):
- **x62 against its prototype log, on target:** 3,575 usable pool mons (739 removed), 572 of 658 sets drawn (the 86
  withdrawn listed), 63 species, 12,000 accepted of 12,412 candidates, species / set-exposure TVD 0.046 / 0.171,
  Hidden Power on 0.370 of slots (pool 0.443), every team ≤ 2 shared with any pool team and < 3 with any held-out
  panel, n1p72 or rc72b team. Refusals: pool species six 275, STRICT Hidden Power 118, within-corpus cap 18,
  validator 1 (the One Boost Passer Clause). One declared deviation from X62 §2.1's wording: the species profile counts
  ALL the pool's mons of a species, as the prototype (and every number X62 §2.2 quotes) did; counting only STRICT-class
  mons moved the species TVD to 0.109 (FINDING 2).
- **rc72b at radius 3, refused by the builder (review m3):** the union's max shared with ONE rc72b team is
  {0: 1,030, 1: 6,693, 2: 10,153}: **all ≤ 2**; a test plants a 6- and a 3-shared copy and fails when the check is
  removed.
- **Every panel's max shared** (all / by component): forum72 {0: 1,148, 1: 11,739, 2: 4,930, 3: 59}, H
  {…, 3: 6}, heldout72 {…, 3: 2}, p72 {…, 3: 1}, `forum216b` {…, 3: 221}, n1p72 {…, 3: 11}. **Every 3 is a D3 team**
  (x62 and n1 are ≤ 2 everywhere): D3 was built at radius 4. The pool itself holds 31 teams at 3 with a forum72 team
  (4.3 %) and 115 with a `forum216b` team, so D3's 59 (2.1 % of its teams; 0.33 % of the union) inherit the pool's
  overlap below the pool's own rate (FINDING 3).
- **No duplicate six; cross-component near-duplicates:** none at ≥ 5 shared sets (the worst is 4, 2 x62–d3 pairs).
- **The material pack-regime share** (the x35 pack audit's method, packed under `f4b25238`'s own `src`): **39 of
  17,876 teams, 0.218 %** (x62 15, n1 16, d3 8), every one a Frustration at base power 1. Per side ≈ 0.8 × 0.218 % ≈
  0.17 % of draws from the union, against C0's 0.125 % from the pool: under the 2 % rule, no backport pin.
- **What the components cover of each Kakuna panel** ([`corpus/f_coverage.log`](corpus/f_coverage.log); share of the
  panel's mon slots whose set, species + moves, occurs in the source; the strict key adds item, ability, nature and
  EVs): F — pool 0.701 / strict 0.232, **x62 0.699** / 0.222 (STRICT withdraws almost none of F's pool sets: the M3
  re-measure), n1@K3 0.762 / 0.155, d3 0.778 / 0.255, the union 0.850 / 0.294. H — pool 0.380 / 0.104, n1@K3 0.866 /
  **0.477** (n1@ALL 0.623), the union 0.900 / 0.505. P72 — pool 0.308 / 0.081, n1@K3 0.755 / **0.407**, the union
  0.787 / 0.435. H and P72 stay hazarded by n1 at K3, as the design states.

**The soak** (`x63_union_v1/soak_selfcheck.json`, main's `gen3_team_corpus_soak_v2`, the label path driven):
**PASS**, 178,760 battles (10 a team, both seats; two
chunks of 5 rounds at seeds 3802 and 3803, merged by `soak merge`), every label family, **0 core errors**, 335
forfeits, 926.5 s of play. `x62_recomb_v1` has no soak record of its own: all 12,000 of its teams are union rows,
and a test ties its data sha256 to the union's soak record.

**At the pin** ([`pin_checks/`](pin_checks/)): a detached checkout of X63U_PIN with its own `src/` and `data/`, its env
core built from its own tree, CUDA hidden.
- **The pin's own crash soak** (`f4b25238`'s `utils.team_corpora.soak`, `gen3_team_corpus_soak_v1`, the selfcheck
  build of the pin's env core, one round: every team once in each seat): **PASS**, 17,876 battles, 0 core errors, 29
  forfeits, 94 s; peak 0.62 GB for the whole process with the full 17,876-team table in its env core
  ([`pin_checks/pin_soak_x63_union_v1.json`](pin_checks/pin_soak_x63_union_v1.json)).
- **The pinned trainer's startup team phase on this argv** ([`pin_checks/startup_team_phase.py`](pin_checks/startup_team_phase.py),
  the BARE argv as N1-P's harness): `🧬 [TEAM-MIX] x63_union_v1@ALL (17876 teams, tier sha f324b6b10fca) share 0.80 on
  both sides; eval stays on the pool`, then `held-out panels: 0 hits over 5 panel(s) (registry sha ea3917c2f1e6); 0
  corpus team(s) pack identically to a pool team`. **Team phase 31.0 s at 0.71 GB peak RSS**, against 2.5 s and
  0.62 GB for X26c's argv in the same harness (N1-P's 12,000 teams: 18.5 s). Linear in the table, it adds ≈ 30 s per
  launch.
- **Not measured:** the per-worker RSS of a full training launch with the table (a `--debug` smoke at the pin could not
  get a heavy slot inside its bound and was stopped before it started; FINDING 5). The chain's first-hour throughput
  gate (§8) is the guard.

## 4. Pre-launch checks (each a HOLD on failure, never a launch)

1. The pin check (`kit pin-check --arm x63u`): X63U_PIN = `f4b25238` + one commit, every changed path under
   `src/utils/team_corpora/`, corpus tree = `X63U_CORPORA_TREE`.
2. `checkargs` of the kit's argv against X26c's record differs by the four mix tokens and the run name only.
3. The launcher `--dry-run`: `FORK of …/rb_x26_s2601`, +100,000,000 steps, the pin above.
4. The kit's start-node sha256 and pool check against X26c's recorded `pool_seed.json` PASS.
5. The material pack-regime share ≤ 2 % of the union's teams (§3; STATUS §7 risk 4).
6. The GPU lease and the arm chain (the Training session's).

## 5. The read: engine, cell, games (this also registers X63-C's read)

- **Playing commit: the X63 successor of the X35-frame shared play pin** `65b998daffffaa1669b0b4ef840874e23edc12d3` (tag
  `x63-play-65b998da` on origin; local branch `x63-play-pin`) = `86cfdf7efbe21b4ff3382541e6d78390df7d9966` (tag `x35-shared-play-86cfdf7e`, FROZEN;
  `designs/tools/reads.md` §4) + ONE commit adding only the `x63_kvk` rule to `src/main/reads/`. The frozen pin's spec
  parser refuses a rule it does not hold, and none of its rules does §6–§7 (re-review blocker 1), so the rule must be
  AT the playing commit; the play path is byte-identical (`main.reads.playpath`: 1,715 files at both commits, one digest
  `afedc505f0e2…`; `git diff --name-only 86cfdf7ef 65b998daf` = `src/main/reads/rules.py`, `rules_test.py`,
  `rules_x63.py`, `rules_x63_test.py`; `check-spec` at the pin accepts the draft spec, `play_path_differs_at_head: 0`). Its regime is the frozen pin's: engine E1,
  our side's legacy pack, Kakuna's post-H22 pack, peer code `9cc9caacde85f8ea`, policy mode `train`. Every game of
  every model in §5–§7 is played there; record and decide run from main (the X48 pattern). The draft spec is
  [`read_spec.draft.json`](read_spec.draft.json); the spec of record is registered by amendment before the first X63
  game (controls included).
- **Cell:** opponent `metamon:Kakuna` (checkpoint 34), greedy against greedy, CPU, role-balanced 20-game units.
- **Unit seed blocks** (disjoint from X48's forum72 0–49 and from each other):

| group | panel | unit seeds | games a model | models |
|---|---|---|---:|---|
| `primary` | forum72 | 1000–1199 | 4,000 | X63-U s2601, X26c, X63-C (Stage B adds X63-U s6302; Stage C adds X63-U s6303, X63-C s6303) |
| `home` | home72 | 1000–1099 | 2,000 | the same models |
| `traj` (REPORTED) | forum72 | 1200–1259 | 1,200 | X63-U s2601, X26c, X63-C at their 400M and 430M nodes |
| `reported` (REPORTED) | heldout72_hpfix, p72, rc72b | 1000–1019 | 400 each | the +100M models |
| `pstar` (REPORTED) | forum72 | 1200–1259 | 1,200 | P\* (`pstar_plateau`) |

- **Every model is pinned by sha256 in the read spec** (X26c's final is the 10-25 file, never a later extension's).
  X63-U's ref is `kind: final` with `min_step 480000195`.
- **Order on the lane.** The controls' `primary` and `home` units (12,000 games) may play as soon as this file is
  registered, before X63-U finishes; their numbers are recorded and not read against any arm until X63-U's own 6,000
  have played. Every REPORTED group plays only after the deciding groups of its stage, and never ahead of another
  registered read's deciding games. The lane order is otherwise the orchestrator's.
- **X63-C's registered quantity** d = F(X63-C) − F(X26c) and σ̂_CC² = max(0, d² − p̂₁q̂₁/G₁ − p̂₂q̂₂/G₂) / 2 are
  computed from the `primary` group exactly as X63-C's PREREG §3 states. X63-C's home72 count is 2,000 here, not the
  1,200 its PREREG named as "X63 Stage B's"; σ̂_CC is F only, so nothing it registered changes.

## 6. The statistic and the seed term

For k_U arm seeds and k_C control seeds at +100M, each model's score a plain proportion of its games:
- Δ_F = mean_U F − mean_C F; binomial part B = Σ_U p̂q̂/G / k_U² + Σ_C p̂q̂/G / k_C².
- σ̂² = mean over the sides with ≥ 2 seeds of (that side's sample variance, ddof 1 − that side's mean p̂q̂/G);
  σ̂ = √max(0, σ̂²); df = Σ (seeds − 1) over the sides (1 at Stage A, 2 at B, 4 at C).
- **σ_eff = max(0.010, σ̂)**; SE² = B + σ_eff² (1/k_U + 1/k_C).
- **q = 1.959964 if σ̂ ≤ 0.010, else the Student-t 0.975 quantile on df** (4.302653 at df 2, 2.776445 at df 4).
- LB / UB = Δ_F ∓ q · SE. Rule 8: any comparison of a statistic with its threshold within 1e-9 takes the
  conservative side (not CONTINUE, not GAIN, not RETIRE, not WITHIN).
- The same arithmetic on home72 gives Δ_home72, σ̂_home and SE_home (§7.4).
- A 72-team cluster bootstrap of Δ_F is REPORTED beside it.

## 7. The decision rule

1. **Stage A** (X63-U s2601; X26c, X63-C): **CONTINUE iff Δ̂_A ≥ +0.010.** Else **NOT PURSUED**: the record prints
   σ̂_CC and P(NOT PURSUED) at a true +2 / +3 pp from `power/power_x63_v2.log`, and claims nothing for the era.
2. **Stage B** (k_U = k_C = 2; + X63-U s6302, licensed only by CONTINUE): **GAIN iff LB > 0 and σ̂ ≤ 0.030**;
   **RETIRE iff UB < +0.015 and σ̂ ≤ 0.030**; **INCONCLUSIVE iff σ̂ > 0.030**; else UNDECIDED.
3. **Stage C** (k = 3; + X63-U s6303 and X63-C s6303), only if B is UNDECIDED with Δ̂_B ≥ +0.015: the same classes,
   final.
4. **On a GAIN only** (k as at the GAIN):
   - **FLOOR PASS iff the arm seeds' pooled home72 Wilson 95 % LB > 0.50.**
   - **net = Δ_home72 + r · Δ_F**; SE_net² = SE_home² + r² SE_F²; **PRICE WITHIN iff net ≥ 0**; **PRICE EXCEEDED iff
     net + 1.959964 · SE_net < 0**.
   - **r = 1, recorded 2026-10-29 by AM-X63-R (§14), before X63-U s2601 launches.** At registration r was UNSET; the owner writes it (STATUS §6; recommended default 1) and the orchestrator
     records it here as a dated amendment **before X63-U s2601 launches** (the chain's GO; re-review blocker 2: once
     the arm trains, its pool win rate and arm-end records preview the in-pool cost that r prices). If none is
     recorded by then, the verdict on a GAIN with FLOOR PASS is UNPRICED and the record prints
     r\* = −Δ̂_home72 / Δ̂_F.

On a GAIN the verdict is taken in this order (re-review blocker 3: one answer for every case):
1. FLOOR fails ⇒ **OVER-RATE**, whatever r;
2. else r UNSET ⇒ **UNPRICED**;
3. else PRICE WITHIN ⇒ **ADOPT-CANDIDATE** (and no VOID);
4. else PRICE EXCEEDED ⇒ **OVER-RATE**;
5. else **TRADE**.

| verdict | rule |
|---|---|
| **NOT PURSUED** / **RETIRE** / **UNDECIDED** / **INCONCLUSIVE** | §7 items 1–3 |

What each licenses is the design's §7 table. An ADOPT-CANDIDATE's lineage hazards H and P72 (n1's coverage); a
baseline proposal from it needs replacement held-out panels. **Power** (the design §7; `power/power_x63_v2.log` A,
row "floor 1.0 t-when-above, INC 3.0 (REGISTERED)"): false GAIN 0.001 / 0.023 / 0.042 / 0.050 / 0.049 at σ 0 / 1 /
1.5 / 2 / 3 pp (0.059 with the arm's spread twice the control's); GAIN at a true +3 pp 0.91 / 0.64 / 0.45 / 0.32 /
0.19.

**The rule is `main.reads` rule `x63_kvk`** (`src/main/reads/rules_x63.py`, at main and at the playing commit; 56
tests, a hand computation of every class, rule 8 on both sides of every bar, the z-to-t switch at the floor, and
Stage A's σ̂ equal to X63-C's σ̂_CC). Its conventions, registered here: a tie counts as a non-win (D1's); every
rule-8 side not named above takes the side that licenses less (σ̂ at the floor uses t; at 0.030 it is INCONCLUSIVE;
Stage C does not fire on a near Δ̂; a near EXCEEDED is not EXCEEDED; a near floor LB does not pass); at Stage A the
`home` group is REPORTED only; a deciding group with an unplayed unit, two engines, or a model with no counted game is
INCONCLUSIVE (inputs); the cluster bootstrap resamples our team per game (10,000 draws, seed 0, one draw of teams
shared by every model).

## 8. VOID guards (a VOID seed is never read; it is re-run at the same pin and seed after the cause is fixed)

- **The banner** within 20 min of the start: `[TEAM-MIX] x63_union_v1@ALL (17876 teams, tier sha f324b6b10fca) share
  0.80 on both sides; eval stays on the pool`, then `held-out panels: 0 hits over 5 panel(s)` and `0 corpus team(s)
  pack identically to a pool team`. Anything else, or `[ModelVersion] FATAL`, or `does NOT inherit`: VOID.
- **The pin:** the launcher log names X63U_PIN; the kit pin-check passes; the dry run was `FORK of …/rb_x26_s2601`.
- **The pool seed:** `fork_step` 380,000,195 and the seeded set equal to X26c's recorded `pool_seed.json`.
- **A FATAL naming a `corpus_<n>` team** (a corpus team the env cannot play): VOID, and the union is rebuilt.
- **Not finished:** the arm chain's completeness check (rc 0, `Last step` ≥ 480,000,195, `final_model.zip`, the
  newest checkpoint within 98,304 steps of it).
- **K9(b) not PASS at the arm's end** (AM-6).
- **More than one restart.** The 24-h interval admits one (a resume restores the optimizer's rate from the
  checkpoint: `main.dose`'s note on `main.train.model_build`); a second restart is a PAGE, and the seed is VOID
  unless the orchestrator rules otherwise before the read.
- **The read:** two engine stamps or two `peer_pack` values in one group is INCONCLUSIVE for that group; a model whose
  sha256 differs from the spec's is refused per unit.

**Throughput < 1,200 steps/s over the first hour ⇒ PAGE** (HOLD and keep training; the orchestrator decides). It is not
VOID: 100M at 1,200 steps/s is 23.1 h, inside the 24-h interval; below that, one restart is expected and allowed.

## 9. AMENDMENT AM-X63-N1L2, written now, before N1-P look 2's verdict (s3504–s3505 train 10-29)

| N1-P look-2 verdict | when it lands | X63 |
|---|---|---|
| GAIN / NULL / UNDECIDED-final | any time | unchanged |
| LOSS | before X63-U s2601 launches | HOLD the launch; rebuild the union without n1 (`x62_recomb_v1@ALL` + `d3_setmut_v1@ALL`, 14,876 teams), a new pin, a dated amendment; the GUESS row "no-n1" |
| LOSS | after X63-U s2601 launches | X63-U trains and is read as registered; every later arm seed keeps the same union. Every verdict carries "n1, 0.134 of draws, read LOSS at 20M / share 0.5"; on NOT PURSUED or RETIRE the no-n1 union is the named follow-up, its own registration |

## 10. The GPU chain (DRAFT, not armed)

`~/.claude/jobs/x35/chains/drafts/x63u-stagea-oct29.json` (the `arm_chain_v1` format of `x35-n1p-oct29.json`): one
item, kit arm `x63u` s2601 at X63U_PIN, banner expectation and VOID strings as §8, throughput `min_steps_per_s 1200`
within 60 min (`on_throughput_fail: page`), `on_void: stop`, disk 22.75 GiB (the dry run's REQUIRED).
`python -m main.ops.arm_chain plan <draft> --check-pins` reads PLAN OK. The Training session ("Training Oct") arms and launches
it after N1-P s3505 ends, on the orchestrator's GO.

## 11. Predictions (GUESS, registered before any game; the design §6.4)

Δ_F at +100M +1.5 pp (−1.5, +4.5); Δ_home72 −2.0 (−5.0, +1.0); Δ_S −2.0 (−4.5, 0.0); Δ_H / Δ_P72 +1.5 / +1.5
(−1.5, +4.5); σ 1.25 pp (0.5, 3.0); the arm's median learning rate / the controls' 0.85× (0.6×, 1.0×).
P(GAIN) ≈ 0.28 over the σ prior; E[GPU-h] ≈ 41.

## 12. FINDINGS at registration (standing rule 7)

1. **σ is unmeasured until Stage A's games.** Every error rate in §7 is conditional on σ; the rule holds false GAIN
   ≤ 0.05 to σ = 3 pp (0.059 if the arm's spread is twice the control's), not beyond.
2. **X62 §2.1's profile wording** ("the pool's STRICT-class mons") disagrees with its prototype and every number §2.2
   quotes (all of a species' pool mons). The build follows the prototype, declares it in the manifest
   (`params.profile`) and pins it by test; X62 §2.1 is corrected in the same change.
3. **D3 teams sit at radius 3 from held-out panels** (forum72 59, `forum216b` 221, H 6, heldout72 2, p72 1, n1p72 11):
   D3 was built at radius 4 and the pin's guard is radius 4. The overlap is the pool's own, inherited at a lower rate
   than the pool carries (forum72: pool 4.3 % of teams, D3 2.1 %). Not refused; F's cleanliness against the union is
   no worse than against the pool every control trains on.
4. **The price is measured on the protected teams** (home72 = the trainee's sample-bias set).
5. **Per-worker RSS of a full launch is unmeasured** (startup and a whole-table env core are, §3). The first-hour
   throughput gate is the guard.
6. **The kit holds Stage A's seed only.** Stage B (`rb_x63u_s6302`) and Stage C (`rb_x63u_s6303`, `rb_x63c_s6303`)
   join by dated amendments after their gates.
7. **r was UNSET at registration.** It is the owner's (STATUS §6 item 4); the chain draft's `go_by` says so. Resolved
   by AM-X63-R (§14): r = 1, before launch.
8. **The two pins are local branches until the ship**: `x63u-pin` (39eaa326) and `x63-play-pin` (65b998da); their
   tags `x63u-pin-39eaa326` and `x63-play-65b998da` go to origin with the ship.
9. **The successor play pin fails two tests at itself by design** (`package_doc_test`: `main/reads/__init__.py` is on
   the play path and is not touched; two `belief_prior_mix` tests: that module does not exist there), and its
   worktree needs `./scripts/bootstrap.sh` before any play.
10. **No GPU, no game.** Every check ran on CPU at nice 19 on the agent cores; the soaks and builds under
    `mem_cap.sh`.

## 13. Re-review (blockers only, 2026-10-29)

A fresh reviewer (opus-high, blockers only) read the design, this file, the power script, review 1, the rulings, X63-C's
PREREG, `reads.md` §4, STATUS and the kit. It reproduced `power_x63_v2.log` byte for byte, confirmed the kit's argv is
X26c's plus the four mix tokens, both control finals' sha256s, X63-C's deferrals honoured, and every review-1 fix
present. It found **three blockers**, each folded in before registration:

| # | blocker | handled |
|---|---|---|
| 1 | The frozen play pin's spec parser refuses a rule it does not hold, so no X63 game (controls included) could play there | FIXED: rule `x63_kvk` built with its tests; the successor play pin `65b998da` = `86cfdf7e` + the rule only, play path byte-identical; `check-spec` accepts the draft spec at it (§5, §7) |
| 2 | r's deadline "before the first X63-U game" left ≈ 21 h in which the arm's in-pool signals preview the cost r prices | FIXED: r must be recorded before X63-U s2601 launches (§7 item 4; STATUS §6) |
| 3 | A GAIN with no r and a failed floor matched both UNPRICED and OVER-RATE | FIXED: a precedence order, FLOOR fails ⇒ OVER-RATE whatever r (§7) |

Its non-blocking checks: seed sharing (X63-U s2601 with X26c) makes the test more cautious (false GAIN at σ = 2 pp
0.049 / 0.026 / 0.004 at a pair correlation of 0 / 0.5 / 0.9); n1 at K3 is consistent with ruling 1; a resume at
`f4b25238` restores the optimizer's rate and re-sends the mix; and the pin's `CORPORA` line must not name the component
corpora absent there (it names only `x63_union_v1`).

## 14. AMENDMENT AM-X63-R (2026-10-29, before X63-U s2601 launches): r = 1

- **r = 1.** It is written before any X63-U step, game or arm-end record, and before the chain's GO.
- **Who set it:** the owner was asked for r, recommended default 1 (STATUS §6 item 4). The owner replied "You choose"
  on 2026-10-29, so the orchestrator set it, using the default's reasoning from STATUS §6: both goals are measured as
  a win rate against the same opponent, and goal 2a has about 5 pp of margin while goal 2b is about 4 pp short.
- **What is fixed:** §7 step 4 now reads: on a GAIN with FLOOR PASS, net = Δ_home72 + 1 · Δ_F.
  - net ≥ 0 ⇒ ADOPT-CANDIDATE;
  - net + 1.959964 · SE_net < 0 ⇒ OVER-RATE;
  - otherwise TRADE.
  - The UNPRICED class can no longer occur for this arm.
- **Scope:** r applies to X63 at every stage (A, B and C). It also stands as X62's r unless X62's own registration
  sets another.
- **Changing it:** r is never changed after launch. A different r would be a new, separately labelled analysis,
  never the verdict.

## 15. AMENDMENT AM-X63-CANARY (2026-10-29 ≈ 19:30, before any X63-U canary disagreement)

- **Written while:** X63-U s2601 was training. Its only canary so far is u10, which passed (ratio 1.619), and it has
  logged no disagreement.
- **Basis:** the slice-2 diagnosis (`6b3966dc6`, `measurements/canary_slice2_2026-10-29/`).
  - Six of seven production disagreements at margin 0.72 were per-parameter bf16-rule refusals on the fixed
    fixture's rows 0–31 (slice 2), with none on slice 1.
  - Their ratio divides by one eager bf16 draw that sits 5–50× below the function's own rounding sensitivity.
  - A real compiled-numerics cause is still open, and a GPU replay window after X63-U decides it.
- **The rule.** Take a canary FATAL whose every disagreement is a per-parameter bf16-rule refusal on slice 0 or
  slice 2 only. That means it was never confirmed on both halves, and it has no margin, yardstick, drift or
  non-finite refusal.
  - Such a FATAL, resumed once from the newest checkpoint at X63U_PIN with the registered argv and its dump
    archived, **does not count toward §8's restart limit**.
  - At most two such resumes per seed. A third FATAL, or a FATAL of any other class, follows §8 unchanged.
  - The resume never raises `--canary-shift-max`, because relocating at 1.2 increases this class's exposure.
- **Disclosure:** each such resume is named in the record beside the verdict. If the replay window later reads the
  class REAL, the record says so.
- **Prior:** P(FATAL of this class during X63-U) ≈ 1.3 % (0.3–6 %), assuming independence.
