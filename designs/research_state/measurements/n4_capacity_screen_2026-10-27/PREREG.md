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

**NOTES** (appended before the first GO, never after any N4 number): none yet.

## 10. Files

- `power.py`, `power.log`: the screen's and the X35 R1 rule's power.
- `battery_baseline.log`: the capacity battery v3 on T\*, toy C0 s3601's final and X47 s3601's final, on ONE state
  set (2,000 states of C0 s3601's eval traces). Reported, never deciding.
