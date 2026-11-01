# PREREG: N4, the toy capacity screen (2026-10-27)

**Registered before any N4 or C0′ run exists, before the build (T69) and before any number of either arm.** The
design and its reasons are [`../../../endstate/design_n4_capacity_screen.md`](../../../endstate/design_n4_capacity_screen.md).
That doc wins on reasons; this file wins on what is read and how it is classed. The parent is
[`../../../endstate/strategy_if_levers_null_2026-10-26.md`](../../../endstate/strategy_if_levers_null_2026-10-26.md)
§4: T3 takes the capacity branch on the plasticity check's NO-LOSS (`8bebdfe3`).

## 1. The question and the predictions

**Q.** At the toy's plateau and at equal data, does a trunk with 4× the FFN width score higher than the same fork
without it? This is the screen of "is the network too small to escape the plateau?".

**Predictions, written before any data** (MY JUDGEMENT, for the record; they decide nothing):
- **P(GO) ≈ 0.25.** X47's depth growth read NO-GO-S under this rule.
- **P(NO-GO) ≈ 0.50.**
- **P(UNDECIDED) ≈ 0.25.**
- **The engagement descriptor E1 reads "engaged" on all three seeds.** X47's new rounds engaged.

## 2. The lever (built by T69; PREREG of the function, not the code)

- **The field.** `trunk_ffn_dim` (default 256 = `TRANSFORMER_FFN_DIM`). A fork grows it with
  `--grow trunk_ffn_dim=1024`.
- **What changes in each trunk round:** its FFN becomes `linear2(relu(linear1(h))) + linear2_x(relu(linear1_x(h)))`.
  - `linear1_x` maps 128 → 768, with a bias, orthogonal gain √2, from a private seed;
  - `linear2_x` maps 768 → 128, with no bias, and starts at ZERO.
- **The fork-time gate.** The grown child must be bit-identical to the parent on the 64 real rows (logits and
  values, CPU fp32 eager), or the launch is refused (`FATAL_CONFIG`).
- **The optimizer.** New parameters start with fresh AdamW state.
- **The pool.** The seeded pool grows the same way.
- **Size.** +394,752 parameters on the toy (2 rounds).

## 3. Arms, argvs, seeds, dose, pin

- **`N4_PIN`.** The first `main` commit carrying T69 (the field, the kit arms `toy_c0n4` / `toy_n4ffn`, the
  descriptor readers, the `n4_screen` rule). It is recorded in §9 by a NOTE before the first GO. Both arms launch at
  it, and only there.
- **The parent.** T\* = `models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip`, sha256
  `cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7` (R's `toy_pstar`).
- **Seeds.** 3641, 3642, 3643. No look 2 is registered.

| arm | run | argv (through `launch_x35.sh <arm> <seed>`, which adds `--pin-commit <N4_PIN> --restart-interval-hours 24`) |
|---|---|---|
| C0′ `toy_c0n4` | `rb_x35t_c0n4_s<seed>` | `--model <T*> --behaviour-check warn --steps 360000087 --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 --supply-starve-cycles self_play_pool=0 --device cuda --seed <seed> --run-name rb_x35t_c0n4_s<seed>` |
| N4 `toy_n4ffn` | `rb_x35t_n4ffn_s<seed>` | the same + `--grow trunk_ffn_dim=1024` (and its own `--run-name`) |

- **The ONE lever** is `--grow trunk_ffn_dim=1024`. Every other knob is inherited from T\*'s record, identically on
  both arms (`main.checkargs` shows the inheritance).
- **The order:** C0′ 3641, N4 3641, N4 3642, C0′ 3642, C0′ 3643, N4 3643.
- **Thinning.** N4 arms are NOT thinned before the read is recorded.

## 4. VOID (a seed enters no read; fix and re-run it)

- **Not finished.** `final_model.zip` is missing, or `metadata.json` `num_timesteps` < 360,000,087.
- **K9(b).** Its AM-6 arm-end record is not PASS.
- **The pin.** Any of the run's `pin_history` spans is not `N4_PIN`.
- **The growth stamps.**
  - An N4 run's `metadata.json` lineage lacks a `grow` stamp with `fields.trunk_ffn_dim` = {parent 256, child 1024}
    and `function_check` max |Δ| 0.0.
  - A C0′ run carries any `grow` stamp.
- **The engine.** The Kakuna rows of the read carry two engine stamps (`server_version` or the Rust sim binary's
  sha256): the whole K read is VOID (`main.reads` engine rule).

## 5. The read and its classes

### 5.1 Inputs

- **S per final.** `readout.population_score` of each final against R (`../x35_breakers_stage_2026-10-17/reference_population_toy.json`,
  its six members):
  - 500 mirrored pairs (1,000 games) per (final, member) cell, greedy against greedy, schedule seed 0;
  - a new eval-ledger family `x35_n4` registered at `N4_PIN`;
  - the player is the final, the opponent the member;
  - a grown final, the plain toys and P\* are three architectures, so they are split across engines as in X47's
    read plan (launch sheet §4.5).
- **K per final.** `main.anchors` / `main.reads` anchor group:
  - opponent `metamon:Kakuna`, teamset `forum72` alone, unit seeds 0–49;
  - so 1,000 games per final: 6,000 games;
  - greedy, CPU, role-balanced 20-game units, the H22 pack regime;
  - every arm in one read at one engine, the arms interleaved unit by unit.
- **"Final"** = `final_model.zip` (AM-X47-11).

### 5.2 Statistics

- **Δ_S** = mean over N4's three finals of S − mean over C0′'s three. The interval is
  `main.x35.readout.welch_interval(on, off, confidence=0.90)`: Welch, two-sided 90 %, so each bound is one-sided
  95 %.
- **Δ_K** is the `d1` estimators on the one panel `forum72`, z = 1.6449:
  - `strat_score(counts, panels=("forum72",), z=1.6449)` per arm, with `counts` the arm's wins / games pooled over
    its three finals (equal games per final, so equal weight);
  - `newcombe_strat(on, off)` gives (Δ̂_K, lo, hi);
  - `overdispersion(on_cells, off_cells, panels=("forum72",))` gives φ̂ from the 3 + 3 per-seed scores;
  - `widen(Δ̂_K, lo, hi, √max(1, φ̂))` gives the interval (LB_K, UB_K). Each bound is one-sided ≈ 95 %.

### 5.3 Classes (rule 8: a bound within 1e-9 of 0 or of its bar is NOT past it; that meter is UNDECIDED)

| meter | GO | NO-GO | else |
|---|---|---|---|
| S | **GO-S** iff LB_S > 0 | **NO-GO-S** iff UB_S < +2.0 pp | UNDECIDED-S |
| K | **GO-K** iff LB_K > 0 | **NO-GO-K** iff UB_K < +1.5 pp | UNDECIDED-K |

- **GO** iff GO-S or GO-K. Its label is `GO (both)`, `GO (in-house)` or `GO (unfamiliar)`.
- **NO-GO** iff NO-GO-S and not GO-K. The record adds NO-GO-K or UNDECIDED-K, and the line "no unfamiliar-only
  effect ≥ ≈ 4 pp detected; smaller ones are not excluded".
- **UNDECIDED** otherwise. The record prints both intervals and the screen's power at the observed σ.
- **INCONCLUSIVE** when a registered input is missing: a cell below 500 pairs, a K unit short of 20 games beyond
  the `main.reads` refusal budget, or a seed VOID and not re-run.

### 5.4 No look 2, no amendment of the classes after any N4 number

An UNDECIDED goes to the orchestrator and the owner (strategy doc §5, T5) with the intervals. More seeds would be a
NEW registration, with its own alpha and its own reason, never this one's look 2.

## 6. Reported, never deciding

1. **The X35 R1 class** at look 1 (`readout.decide_r1(1, …)`, 97.5 %, ±2.0 pp) on the same finals.
2. **ρ**: the mean wall per step of N4 over C0′, from the dose records.
3. **G-F**: the 3 × 3 head-to-head (N4 finals × C0′ finals, `cross.matrix_from_cells(..., min_pairs=500)`).
4. **Per-member Δ_S** (Welch at 90 %), with `pstar_x26` on its own.
5. **The pin-drift line**: C0′'s mean S against the banked toy C0 finals' 52.428 (`toy_x47_look1.json`). It is a
   Welch interval on 3 + 3, and the S cells are played at `N4_PIN`.
6. **home72 against Kakuna**, 400 games per final, only if the anchor lane has room after F: the arms' home72 − F
   drop.
7. **The mechanism read** (design doc §5):
   - E1: per-round ‖`linear2_x`‖_F / ‖`linear2`‖_F at every N4 node; "engaged" iff ≥ 0.01 at the final;
   - E2: the mean KL(π ‖ π with `linear2_x` zeroed) and the argmax-move share on `bank_x26`, at every 10th node and
     the final;
   - the capacity battery v3 on the six finals and T\*, on C0′ s3641's eval-trace states (`--max-states 2000 --seed
     0`), lines (a), (b), (e);
   - the registered reading table of design doc §5 is applied and printed beside the class.

## 7. The read spec (written into `read_spec.json` by T69, at `N4_PIN`; its content is fixed here)

- **The flow.** `gen3_read_spec_v1`, `pin` = `N4_PIN`, following the registration flow in
  `designs/tools/reads.md` §1.
- **`models`:** six `final` refs (`min_step` 360000087, `quiet_s` 300), labels `c0n4_s364x` / `n4ffn_s364x`.
- **`s` group `pool`:**
  - `refpop` and its `refpop_content_sha256`;
  - `panel` null, `pairs` 500, `batch_pairs` 100;
  - `family` `x35_n4`;
  - all six models.
- **`anchor` group `unfamiliar`:** opponent `metamon:Kakuna`, `teamsets` `["forum72"]`, `unit_seeds` `0-49`, all
  six models.
- **`decision`:** rule `n4_screen` with `params` {on, off label lists, `s_group`, `anchor_group`, `s_bar` 2.0,
  `k_bar` 0.015, `confidence` 0.90, `z` 1.6449, `rule8_eps` 1e-9, `void` {`k9b_dir`, runs}}.
- **A spec that differs from this list is refused by this PREREG.** A difference forced by the code (a field name)
  is a NOTE in §9 before any game.

## 8. The hand-off (the training session's chain; the GO is the orchestrator's)

- **The chain file.** A new `arm_chain_v1` file (`designs/ops/arm_chain.md` §2), backend `x35`, the six items of §3
  in order, each with:
  - `pin` = `N4_PIN`;
  - `run` as in §3;
  - `thin`: `none` for N4 (reason: §3), `planner` for C0′;
  - `on_void: stop`, `on_throughput_fail: page`;
  - `throughput.min_steps_per_s` 2,500;
  - on N4, a `banner.expect` = the growth line T69 prints, and on C0′ that line in `banner.void`.
- **When.** After the current chain `x35-lrf-oct26` (LR-F s3502 RUNNING since 07:24 PDT 10-27; s3503 PENDING; the
  card frees ≈ 16:00 PDT 10-27), and after T69 lands with its GPU smoke (design doc §7).
- **Cost.** ≈ 30 GPU-h (C0′ ≈ 3 × 4.6, N4 ≈ 3 × 5.5 at a GUESSED ρ 1.2), ≈ 26–38 lane-h of Kakuna, ≈ 3 CPU-h of
  S cells and mechanism reads.

## 9. Power (`power.py` → `power.log`, deterministic) and NOTES

**Power** (design doc §4.3 has the table):
- **S:** GO-S 0.04 at a true 0; 0.53–0.79 at +2 pp; 0.83–0.98 at +3 pp. NO-GO-S 0.52–0.77 at 0. σ 1.07 / 0.75.
- **K at 1,000 F games a final:** GO-K 0.04–0.06 at 0; 0.61–0.69 at +3; 0.81–0.89 at +4 pp.

**NOTES** (appended before the first GO, never after any N4 number):

**NOTE 1 (2026-10-27, before the first GO; no N4 run exists).** Ruling 2 of the precommit memo `endstate/precommit_reads_2026-10-27.md` §4, accepted by the orchestrator: LR-F look 2 (8.6 GPU-h) runs after X48's arms and before N4 launches. Nothing registered above changes.

**AMENDMENT AM-N4-PIN (2026-10-31, before the first GO; no N4 or C0′ run, game or number exists).** No class, bar,
estimator, seed, dose, argv or VOID reason changes.
1. **`N4_PIN` = `1553f8bd838a13d3709239c43b3ee98d27cac359`**, the first `main` commit carrying T69's width field
   (`c35ffab9e`), the `n4_screen` rule (`a05a0ca2c`) and the kit arms `toy_c0n4` / `toy_n4ffn`
   (`main.x35.batch.N4_PIN`, checked by `n4_pin_check`). §3 also named the descriptor readers; they are not built.
   They read reported lines only (§6 item 7) and never touch training, so `N4_PIN` is the TRAINING pin: all six
   forks launch there and only there, and §4's pin VOID reads against it.
2. **The read's pin and S driver are fixed by a NOTE before any read game**, under one constraint: every K and S row
   plays on ONE engine. Verified 2026-10-31 (CPU): one `main.h2h` engine refuses three architectures
   (`CellArchMismatch`), and the runner stripes R's members across two engines, so at `N4_PIN` the S lane would
   refuse. The NOTE takes one of two routes:
   - `N4_PIN` with an architecture-pure driver (X48 NOTE 5's pattern), with its identity check: one banked toy C0
     cell replayed equals its banked rows;
   - or one later `main` commit carrying the runner's grouping fix, with the same check.

   Either commit must also carry the width descriptor readers (E1 / E2) and the battery's width worker, both
   missing at `N4_PIN`. The frozen X35-frame shared pin `86cfdf7ef` cannot load a grown toy (its model code is
   config v161), so it is not a route.
3. **The rule's decision parameters, fixed NOW as registered.** Training writes in-loop eval rows within minutes of
   the first GO, so these cannot wait for the read. §7 does not list four of them, and the `n4_screen` rule needs
   them:
   - `mixed_veto`: `false`. §5.3 registers no MIXED class.
   - `near_scope`: `"meter"`. §5.3: a bound near its bar reads that meter UNDECIDED. The rule's default, `"bound"`,
     is X48's wording.
   - the S group's `pairs`: 500 (§5.1).
   - `nogo_scope`: `{"lever": "--grow trunk_ffn_dim=1024", "dose": "70M", "start": "T*"}`. These are descriptive
     strings for the NO-GO record's text, and no class reads them.

   The M4 "LR-confounded" / "LR-elevated" labels the rule prints are labels, never classes. Adopting any of the
   REVIEW's fixes that touch the classes or their inputs needs its own amendment BEFORE THE FIRST GO, never after
   any N4 or C0′ number (§5.4, §9's header): M2's harm veto, M3's 2,000 pairs, m3's Δ̂ floor, M5's NO-GO wording.
   That is the orchestrator's ruling.
4. **The GPU smoke (§8; design doc §7 item 3) is not a separate launch.** It needs a lease this unit does not hold.
   Its checks become guards on the chain's arms:
   - the fork-time gate's line is the N4 banner's `expect`, "(394,752 parameters); optimizer state carried for". The
     trainer prints it only after the bit-identity gate passes (`main.train.model_build`);
   - `[GrowFork] FATAL` and `[ModelVersion] FATAL` are `void` lines on both arms, and any `[GrowFork]` line is `void`
     on C0′ (§8). C0′'s `expect` is "[CompileRegions] parity PASS", which every healthy toy C0 child printed. All arms
     run with `on_void: stop`;
   - the pool growth line ("seeded pool grown") is checked by hand at N4 s3641's start. A failure there is already
     `[GrowFork] FATAL`;
   - ρ is measured from the dose records (§6 item 2; it decides nothing).

   C0′ s3641 runs first, so a GPU-only failure of the grown net stops the chain after one control arm.
5. **Verified before this amendment** (CPU, `CUDA_VISIBLE_DEVICES=""`, 2026-10-31):
   - `checkargs` exits 0 on both argvs, and the only difference between them is the lever;
   - `launcher --dry-run` shows FORK of `rb_toy_ent_s4002_deep`, +70,000,000 steps, "grow : DECLARED trunk_ffn_dim
     256 -> 1024" on N4 only, 15.93 GiB REQUIRED, and fork arch compatible. Its one refusal is the CUDA-visibility
     gate, which the empty `CUDA_VISIBLE_DEVICES` trips by design;
   - the kit's pool, corpus and K9(b) checks PASS;
   - a grown T\* (offline, gate max |Δ| 0.0, 394,752 parameters) plays 2 pairs against T\* and 2 against P\* in
     `main.h2h`, and one counted 20-game forum72 unit against Kakuna in `main.anchors`.
6. **K units play as finals settle** (the runner's settle loop). "Arms interleaved unit by unit" (§5.1) then holds
   among the settled finals, and later finals catch up from unit seed 0. Unit seeds are fixed per unit, so no
   outcome depends on the order.
7. **NOTE 1's condition is moot.** LR-F look 1 read UNDECIDED, final (`1355d2d3f`), so look 2 does not run (the kit's
   stage-check refuses it). With X48's NO-GO (`b01dc048b`), the precommit memo's §2.2 row governs ("as for not
   GAIN"): N4 or N4-P-direct is the orchestrator's choice, and it chose the N4 toy (brief, 2026-10-31).

## 10. Files

- `power.py`, `power.log`: the screen's and the X35 R1 rule's power.
- `battery_baseline.log`: the capacity battery v3 on T\*, toy C0 s3601's final and X47 s3601's final, on ONE state
  set (2,000 states of C0 s3601's eval traces). Reported, never deciding.

**AMENDMENT AM-N4-FIXES (2026-10-31, before the first GO; no N4 or C0′ run, game or number exists).** The
orchestrator's ruling (2026-10-31): adopt the REVIEW's M2, M3, m3 and M5, exactly as the REVIEW proposed them. This
REPLACES AM-N4-PIN item 3's values where they differ. The training side does not change: the argvs, seeds, dose,
pin, order, thinning and VOID reasons all stand, and so does the chain draft.
1. **M2, the harm veto: a MIXED class.** It applies both ways, as the REVIEW wrote it (M2 (a)):
   - GO-S with UB_K < 0 reads **MIXED**;
   - so does GO-K with UB_S < 0.

   MIXED is classed UNDECIDED, is never GO, and is printed with both intervals. The class order is INCONCLUSIVE >
   MIXED > GO > NO-GO > UNDECIDED. Spec: `mixed_veto: true`. M2 (b): α stays 0.05 one-sided per meter, and the
   record states the family false GO (item 5), never "Bonferroni".
2. **M3: 2,000 mirrored pairs per (final, member) S cell**, in place of §5.1's 500. Spec: the S group's `pairs`
   2000, which is also the INCONCLUSIVE floor for a cell. G-F's 3 × 3 stays at 500 pairs (§6 item 3, reported).
3. **m3: GO-S also needs the point estimate Δ̂_S ≥ +1.0 pp.** This is the REVIEW's own value, "the screen's minimum
   interesting effect".
   - A Δ̂_S within 1e-9 of +1.0 is not past it (rule 8). With `near_scope` `"meter"` it reads S UNDECIDED.
   - NO-GO-S is unchanged: UB_S < +2.0 pp.
   - GO-K has no floor. The REVIEW's m3 is S-only.

   Spec: `go_s_min_delta: 1.0`. It is a new optional parameter of the `n4_screen` rule, built in this amendment's
   commit; absent, the rule is unchanged, so X48's banked record does not move.
4. **M5: the NO-GO wording.** Every NO-GO reads "no ≥ 2 pp in-house gain from a 4× trunk FFN width (`--grow
   trunk_ffn_dim=1024`) within 70M of T\*". It never reads "capacity does not bind on the toy". Spec: `nogo_scope`
   `{"lever": "a 4x trunk FFN width (--grow trunk_ffn_dim=1024)", "dose": "70M", "start": "T*"}`.
   - M5 (a), the three-node slope, is NOT adopted: the ruling names the wording only. The planner's thinning keeps
     C0′'s 320M and 340M nodes, so a later registration could still read it.
5. **The error rates under the amended rule** (`power_am_fixes.py` → `power_am_fixes.log`, seeded and
   deterministic; the REVIEW's model; K 1,000 forum72 games a final, base 0.42). σ_S is the per-final SD of S.
   - **The planning σ_S is 0.56–0.65 pp:** the REVIEW's ESTIMATE at 2,000 pairs (ESTIMATE, not measured).
   - **The pessimistic σ_S is 1.07 pp:** §9's registered value at 500 pairs, in case the evaluation-noise share is
     smaller than the REVIEW's upper bound.

   | true (Δ_S, Δ_K) pp | P(GO), σ_S 0.56–0.65 | P(GO), σ_S 1.07 | P(NO-GO), σ_S 0.56–0.65 | P(MIXED), σ_S 0.56–0.65 |
   |---|---|---|---|---|
   | (0, 0): the family false GO | **0.048–0.068** | 0.071–0.084 | 0.83–0.90 | ≤ 0.003 |
   | (+1, 0) (GO-S alone) | 0.34–0.39 | 0.20 | 0.34–0.39 | — |
   | (+2, 0) | 0.83–0.91 | 0.52 | 0.009–0.016 | 0.035–0.053 |
   | (+3, 0) | 0.94–0.96 | 0.79 | 0.000 | 0.039–0.057 |
   | (0, +4) | 0.78–0.86 | 0.78–0.86 | 0.10–0.18 | 0.025–0.035 |
   | (+2, −3), the harm case | 0.26–0.37 | 0.16–0.24 | 0.009–0.016 | **0.52–0.66** |

   - Each range spans σ_s(K) 0 and 1 pp, and seed-effect correlations ρ 0 and 0.7 between the meters.
   - The S meter alone, at σ_S 0.56 / 0.65 / 1.07: GO-S 0.010 / 0.017 / 0.035 at a true 0; 0.94 / 0.88 / 0.53 at
     +2 pp. NO-GO-S at a true 0: 0.94 / 0.87 / 0.53.
   - Against the rule as registered (`power.log`, `review_power.log`):
     - the family false GO falls from 0.090–0.093 to 0.048–0.068;
     - GO at +2 pp rises from 0.53–0.79 to 0.83–0.91 at the planning σ;
     - the floor costs GO at a true +1 pp (0.34–0.39 at the planning σ);
     - in the harm case (+2, −3), P(GO) falls from 0.79 to 0.26–0.37: most of that probability moves to MIXED.
6. **The read's pin.** The spec's pin must carry this commit's rule (`go_s_min_delta`), and an older pin's rule
   refuses the key. So the read plays at ONE later `main` commit, never at `N4_PIN`; AM-N4-PIN item 2's first route
   is withdrawn. That commit carries the read-side tools (the S lane's architecture-pure engines, the width E1 / E2
   readers, the battery's width worker), with X48 NOTE 5 (a)'s identity check. It is named by a NOTE before any read
   game.
7. **The spec's decision parameters, now complete:**
   - `mixed_veto` true;
   - `near_scope` "meter";
   - `go_s_min_delta` 1.0;
   - `s_bar` 2.0, `k_bar` 0.015, `confidence` 0.90, `z` 1.6449, `rule8_eps` 1e-9;
   - S `pairs` 2000;
   - `nogo_scope` as in item 4;
   - `void` {`k9b_dir`, runs, and the pin VOID against `N4_PIN`}.
8. **The S lane's cost** rises to about 6–12 lane-h (36 cells × 2,000 pairs), from 1.5–3. CPU only.
