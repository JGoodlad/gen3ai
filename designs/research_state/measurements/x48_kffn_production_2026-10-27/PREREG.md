# X48 — per-token-kind FFN experts at the production plateau: PRE-REGISTRATION (2026-10-27)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED** (no `rb_x35_x48kffn_*` run exists). The
build is CPU only; the launch is the orchestrator's GO through the Training session's arm chain. Nothing below changes
after the first X48 arm launches except by a dated amendment, written before the read, that says why.

**Why this rule.** The N4 review ([`../n4_capacity_screen_2026-10-27/REVIEW.md`](../n4_capacity_screen_2026-10-27/REVIEW.md)
§4, `7660d717`) found that X48 under X35's R1 (the backlog's BI-2 line) cannot reach GAIN at three seeds and mostly
cannot reach NULL either: 13 GPU-h to read UNDECIDED. It recommended running X48 first, as the GPU plan orders, under
**N4's one-look screen rule** ([`../n4_capacity_screen_2026-10-27/PREREG.md`](../n4_capacity_screen_2026-10-27/PREREG.md)
§5) with its fixes M2 and M3. This file registers exactly that, with M4 and M5 as reported lines and the canary family
ruled (§7). R1 is reported, never deciding (§8).

Abbreviations: pp = percentage points; LB / UB = lower / upper bound; P\* = `rb_x26_s2601` @280,000,170; C0 = X35's
plain-continuation control (P\* + 20M at the production pin `f4b25238`); S = in-house strength, the population score
against the frozen R (`design_compute_efficiency.md` §5.6.4); K = the score against Kakuna on `forum72` (goal (b)'s
unfamiliar panel); GUESS / ESTIMATE / MY JUDGEMENT as in
[`../../../endstate/strategy_if_levers_null_2026-10-26.md`](../../../endstate/strategy_if_levers_null_2026-10-26.md).

## 1. Question, mechanism, predictions

**Q.** At production's plateau and at equal data (+20M), does untying each trunk round's shared FFN into five
per-token-kind experts score higher than the same fork without it, in-house (S) or against Kakuna on unfamiliar teams
(K)? This is the first production read of the capacity branch the strategy tree took on the plasticity probe's
NO-LOSS (`8bebdfe3`).

**Mechanism** ([`../../../endstate/design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §2.2,
H-KFFN): one FFN per round transforms all 64 tokens of five very different kinds (Pokémon, board, our move seats,
threat seats, events). Untying gives each kind its own 256-unit transform at the same FLOPs per token. It is
specialisation at the SAME per-token width, so a NO-GO does not close FFN width (N4) or the residual width
(`D_MODEL`; review §1, M6).

**Predictions** (GUESS, registered before any X48 number; they decide nothing): Δ_S **+0.3 pp, 80 % [−1.5, +2.0]**;
Δ_K **+0.3 pp, 80 % [−2.0, +2.5]**. Basis: X47's depth growth read Δ̂ ≤ 0 on the toy (−0.88 [−3.20, +1.44]); the
plasticity probe's NO-LOSS keeps capacity open; the copies inherit AdamW state (Q-K3), so no fresh-state shock.
Classes, from [`power.log`](power.log) (4) over this GUESS: **GO 0.28–0.36 · NO-GO 0.36–0.50 · UNDECIDED 0.15–0.36**
(σ_S 1.07 / 0.60). Mean LR within 0.9× of C0's on every seed (M4): 0.8.

## 2. The arm (kit arm `x48`: `python -m main.x35.kit describe --arm x48 --seed <s> --call reference_population_production.json`)

| item | value |
|---|---|
| start | P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, R's frozen call), as C0 |
| the ONE lever | `--grow trunk_ffn_by_kind=on` (B-KFFN, T66): a DECLARED function-preserving growth at the fork. Each round's shared FFN becomes five experts routed by fixed seat-role segments, each a COPY of the shared FFN carrying its AdamW moments (Q-K3); +791,040 parameters (2,609,183 → 3,400,223 on the fresh production policy, F-GB-8). The child refuses to start unless it is P\*'s function bit for bit on the 64-row gate (`[GrowFork] FATAL`) |
| everything else | C0's argv token for token (`kit_test.test_x48_is_c0_plus_the_grow_lever_and_nothing_else`): `--model <P*> --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --grow trunk_ffn_by_kind=on --seed <s> --run-name rb_x35_x48kffn_s<s>` ([`argv_x48_s3501.txt`](argv_x48_s3501.txt)); every untyped knob is inherited from X26's record, as C0's |
| pin | **the production side pin `f4b25238def57900c0305bb2a3a5a8f9edaef285`**, C0's. B-KFFN `3f69e6e2e` is its ancestor, so X48 needs no own pin (`batch.X48_ARMS` is side-pinned; `kit pin-check` holds it there) |
| seeds | **3501, 3502, 3503. One look; no look 2** (N4 §5.4). More seeds are a NEW registration with its own α; the kit refuses s3504+ (`kit_test.test_x48_refuses_every_seed_but_the_one_registered_look`) |
| control | **C0 s3501–s3503**, already run at `f4b25238` (each K9(b) PASS), finals pinned by sha256 (§5): 0 GPU-h. They are **re-played** in X48's read on its one engine (§5.1), never imported |
| GPU | **≈ 13 GPU-h** (3 × ≈ 4.3; the measured 20M production fork, D3 s3505 245 min at 1,397 steps/s; ρ ≈ 1.00–1.05 ESTIMATED, B-KFFN's five smaller matmuls, UNVERIFIED until the first 1M) |
| stage | `batch.X48_STAGES` (`X48`: s3501–s3503); `kit stage-check` gives it the AM-8 side gate and nothing else |

## 3. Pre-launch checks (a failure is a HOLD, never a launch)

`launch_x35.sh x48 <seed> --pin f4b25238def57900c0305bb2a3a5a8f9edaef285 --lease-owner <owner>` runs every pre-check
(2b `pin-check --arm x48`, 2d `corpus-check` (no corpus: PASS), 5 `checkargs --pin`, 6 the dry run with the card
visible, 9 `stage-check`) and launches only with `GO`. `kit validate-run --arm x48` passes a transcript pair only if
BOTH tools print the declared growth `trunk_ffn_by_kind 'off' -> 'on'` (`kit.ARM_GROWTH`), and fails it on C0 or with
X47's growth line (`kit_test.test_the_verdict_fails_x48_transcripts_read_as_c0_or_as_x47s_growth`).

## 4. VOID (a VOID seed enters no read; fix the cause and re-run it at the same pin and seed)

1. **Not finished:** no `final_model.zip`, or `metadata.json` `num_timesteps` < 300,000,170.
2. **K9(b):** its AM-6 arm-end record (`kit k9b`) is not PASS.
3. **The pin:** any `pin_history` span is not `f4b25238` (`kit pin-check`).
4. **The growth stamp:** the run's `model_config.json` `growth` (and the lineage's `grow`) lacks `fields.trunk_ffn_by_kind`
   = {parent `off`, child `on`}, or `function_check.bit_identical` is not true, or `optimizer_state` has no `copied`
   count (Q-K3's copied moments). A launcher or child log with `[GrowFork] FATAL`, `[ModelVersion] FATAL` or `does NOT
   inherit` (the chain's `banner.void`, which also stops it).
5. **The seeded pool:** the T54 post-spawn check fails (`kit pool-seed-check`), or `pool_seed.json` lacks its `grown`
   block (every seeded snapshot grown the same way).
6. **The KL** (LR-F's bar, STAGED_PLAN §4.1.2): a trailing-2M mean approx KL > 0.08, rule 8 at 1e-9, read by hand with
   `main.ops.tb_read` at every restart and at arm end. Copied experts add 30 % parameters that each move at about η.
7. **The engine:** a meter whose rows span two engines (`main.reads` engine rule) is INCONCLUSIVE, never pooled.

**Not VOID, a HOLD:** first-hour throughput below 1,000 steps/s (ρ > 1.4, far outside the estimate). The chain item
`page`s, not `stop`s (§11): slowness may be the lever's real cost, and the orchestrator decides.

## 5. The read and its classes (N4 PREREG §5, with the review's M2 and M3)

### 5.1 Inputs: ONE read spec, ONE playing commit, ONE engine, both arms re-played

- **Why C0 is re-played.** C0's banked S (G1, `~/gen3ai_archive/x35_g1/results.json`) is at 500 pairs, and its banked
  Kakuna rows are on E1 with the legacy pack, before `b0441623e` split forum72's regime (`designs/tools/reads.md` §4).
  The read cannot import either. Both arms are played in X48's read state, on its playing commit, so engine, pack and
  play path are one by construction.
- **C0's finals, pinned by sha256:** s3501 `09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303`, s3502
  `7e9a9890e2b397de6d8b87588c597d7f53e39b84efd3d1080305f77dbf22feec`, s3503
  `99096be8eccfc1cef4268ab4d6de15063a386cda475e72340212fdbf3a1b7e26` (`models/rb_x35_c0_s350x/final_model.zip`).
  X48's finals are `final` refs (`min_step` 300000170, `quiet_s` 300).
- **S per final** (M3): `readout.population_score` against R (`../x35_breakers_stage_2026-10-17/reference_population_production.json`,
  content `cef0960e…4519`, 7 members), **2,000 mirrored pairs (4,000 games) per (final, member) cell**, greedy vs
  greedy, the eval team builder (no panel), schedule seed 0, a new eval-ledger family `x35_x48`. One device for every
  S cell of the read (the regime id carries it).
- **K per final:** `main.reads` anchor group, opponent `metamon:Kakuna`, teamset `forum72` alone, unit seeds 0–49
  (1,000 games a final, 6,000 in all), greedy, CPU, role-balanced 20-game units, the H22 pack regime
  (`gen3_peer_showdown_pack_v1`), the arms interleaved unit by unit.

### 5.2 Statistics (N4's, verbatim)

- **Δ_S** = mean of X48's three S − mean of C0's three; `main.x35.readout.welch_interval(on, off, confidence=0.90)`
  (Welch, two-sided 90 %: each bound one-sided 95 %).
- **Δ_K** by the `d1` estimators on `forum72` alone at z = 1.6449: `strat_score` per arm (wins / games pooled over its
  three finals), `newcombe_strat(on, off)`, φ̂ = `overdispersion(on_cells, off_cells, panels=("forum72",))`, and
  `widen(Δ̂_K, lo, hi, √max(1, φ̂))` → (LB_K, UB_K). φ̂ is printed with the class (review m2).

### 5.3 Classes (rule 8: a bound within 1e-9 of 0 or of its bar is NOT past it)

| meter | GO | NO-GO | else |
|---|---|---|---|
| S | **GO-S** iff LB_S > 0 | **NO-GO-S** iff UB_S < +2.0 pp | UNDECIDED-S |
| K | **GO-K** iff LB_K > 0 | **NO-GO-K** iff UB_K < +1.5 pp | UNDECIDED-K |

In this order:
1. **INCONCLUSIVE** when a registered input is missing: an S cell below 2,000 pairs, a K unit short beyond the
   `main.reads` refusal budget, a VOID seed not re-run, or a meter on two engines.
2. **MIXED (M2), classed UNDECIDED:** GO-S with UB_K < 0, or GO-K with UB_S < 0. Printed with both intervals. Never GO.
3. **GO** iff GO-S or GO-K. Labelled `GO (both)`, `GO (in-house)` or `GO (unfamiliar)`, with Δ̂_S and Δ̂_K (review m3).
   Its text reads "growth at the plateau helped" (review m4).
4. **NO-GO** iff NO-GO-S and not GO-K. Its text reads **"no ≥ 2 pp in-house gain from per-kind FFN experts within 20M
   of P\*"** (M5), never "capacity does not bind", plus NO-GO-K or UNDECIDED-K and "no unfamiliar-only effect ≥ ≈ 4 pp
   detected; smaller ones are not excluded". It is labelled **"LR-confounded"** (a weak NO-GO) when X48's mean LR is
   < 0.9× C0's (M4, §8.3).
5. **UNDECIDED** otherwise, with both intervals and the screen's power at the observed σ.

**The family false GO is ≈ 0.09, not 0.05** (M2): α = 0.05 one-sided per meter, two meters, GO on either. Under a true
null on both meters this rule reads GO 0.088–0.089 ([`power.log`](power.log) (3), ρ 0 and 0.7). The record states
this number; it is a screen whose GO buys a production re-read, not an adoption.

**No look 2, and no amendment of §5 after any X48 number** (N4 §5.4). An UNDECIDED or MIXED goes to the orchestrator
and the owner with the intervals.

## 6. Power ([`power.py`](power.py) → [`power.log`](power.log), CPU, deterministic, seconds)

The N4 review's simulation ([`../n4_capacity_screen_2026-10-27/review_power.py`](../n4_capacity_screen_2026-10-27/review_power.py))
at production's inputs, with M2's veto. σ_S at 2,000 pairs: production C0's three G1 finals at 500 pairs read SD
0.48 pp against a per-final measurement SE of 0.55 pp, so the seed SD itself is small; at 4× pairs the measurement SE
is ≈ 0.27 pp. **σ_S ≈ 0.4–0.6 pp is the expected range (ESTIMATE); 1.07 (the toy's registered σ) is the pessimistic
end.**

| | true 0 | +1 pp | +2 pp | +3 pp |
|---|---|---|---|---|
| GO-S / NO-GO-S at σ_S 0.60 | 0.039 / 0.888 | 0.452 / 0.279 | **0.920** / 0.007 | 0.997 / 0.000 |
| GO-S / NO-GO-S at σ_S 1.07 | 0.039 / 0.519 | 0.205 / 0.188 | 0.527 / 0.030 | 0.824 / 0.002 |
| GO-K / NO-GO-K at σ_k 1 pp (+0 / +2 / +3 / +4) | 0.056 / 0.255 | 0.365 / 0.030 | 0.604 / 0.006 | 0.803 / 0.001 |

The joint rule (σ_S 0.60, σ_k 1 pp): GO 0.089 at (0, 0); 0.87 at (+2, 0); 0.60 at (0, +3). The harm case (+2, −3)
reads MIXED 0.55 and GO 0.36 (the GOs there are draws where K's UB did not fall below 0). K is the weak meter: 1,000
F games a final detect +3 pp about 0.6 of the time.

## 7. The canary family: X48 takes the C0 bar (2.64)

A canary FATAL on a run that no family matches HOLDs until a human comes (`main.ops.canary_decide`; the GPU plan's
FINDING 3). **Ruled here: `rb_x35_x48kffn_s*` is in the C0 family** (`canary_decide.FAMILY_NAME_PATTERNS`, tested by
`canary_decide_test.test_every_production_x35_arm_is_judged_at_the_c0_bar`). Reasons:
1. X48 is a P\*-fork at the production pin, the definition every production arm in the C0 family meets.
2. The bar judges the compiled / eager `max_abs_dlogp` ratio of ONE micro-step. X48's experts are the same `linear` /
   activation ops at the same per-token width, applied to row segments; no new op class enters the step. Whether the
   compiler fuses the five segments differently is **UNVERIFIED**.
3. The one architecture-change precedent read LOWER, not higher: X47 (deeper) had a healthy-PASS maximum of 1.893
   against toy C0's 2.108.
4. The C0 bar is itself a toy number on every production arm (TECH_DEBT P1 row). X48 adds no new uncertainty class.
   When W0's CUDA census re-rules the production bar, X48 takes the re-ruled C0 bar. Every other HOLD reason
   (clip-branch switches, a missing field, nothing judged) still HOLDs.

## 8. Reported, never deciding (printed under that heading, with no adjectives; review m7)

1. **X35 R1** at look 1 (`readout.decide_r1(1, …)`, 97.5 %, ±2.0 pp) on the same S finals.
2. **ρ**: X48's mean wall per step over C0's, from the dose records (C0's ran 10-21 to 10-24 on the same card; box load
   differs by day).
3. **The learning rate (M4):** each arm's mean LR over its 20 nodes (`metadata.json` `snapshot_history`) and its mean
   approx KL (dose records). **If X48's three-seed mean LR is < 0.9× C0's, a NO-GO is labelled "LR-confounded".** The
   copied moments (Q-K3) predict little controller response; X47's zero-init rounds ran at 0.91×.
4. **The slope (M5):** S at the +10M node of every arm (C0's `checkpoint_290000xxx_steps.zip` survives thinning; X48's
   too), 500 pairs a cell; per arm S(final) − S(+10M), and Δ_slope = X48's − C0's, Welch at 90 %.
5. **Per-member Δ_S** (Welch at 90 %), with `x26_pstar` on its own; **G-F**, the 3 × 3 head to head (X48 finals × C0
   finals, 500 pairs a cell).
6. **The pin-drift line:** the re-played C0 S at 2,000 pairs against G1's banked C0 S (56.98 mean at 500 pairs, E1),
   descriptive (two engines, two pair counts).
7. **The experts' divergence** (design_bot_ideas_screen §2.2's descriptor, with M1's lesson): per round and kind
   k ≠ Pokémon, D_k = ‖W_k − W_Pokémon‖_F / ‖W_Pokémon‖_F for `linear1` and `linear2` at the +10M node and the final;
   beside it the null √2 × C0's own FFN drift ‖W_C0(final) − W_P\*‖_F / ‖W_P\*‖_F (two copies drifting independently
   by C0's amount). Reported as the ratio D_k / null.
8. **home72 against Kakuna**, 400 games a final, only if the anchor lane has room after F: each arm's home72 − F drop.

## 9. What each class licenses (the N4 review §4 table, applied)

| X48 | then |
|---|---|
| **GO** (any label) | **skip the toy N4.** The next capacity read is production width directly (N4-P), which is also X48's registered equal-parameter attribution arm (Q-K4: a wider shared FFN with X48's parameter count), at `N4_PIN` with a production C0′, read by the PR rule. Nothing joins the recipe on a screen GO |
| **NO-GO** | closes per-kind FFN untying at the plateau for production, "within 20M". It does NOT close FFN width, `D_MODEL` or from-scratch capacity. N4 (with the review's M1–M5) or N4-P-direct is the orchestrator's choice. "LR-confounded" makes it a weak NO-GO |
| **MIXED / UNDECIDED** | nothing licensed; the orchestrator and the owner rule with the intervals |
| any, with **LR-F look 1 GAIN** | the capacity reads are re-designed with the LR floor on both arms (review §4) before any further width read |

## 10. The read machine (content fixed here)

- **The rule** is `main.reads` rule `n4_screen` (N4 PREREG §7, built with T69), with M2's MIXED class. **No X48 game
  is played until that rule exists with MIXED and a spec with the content below passes `python -m main.reads
  check-spec`.** If the rule as built names a field differently, that is a NOTE in §13 before any game.
- **The spec:** `gen3_read_spec_v1`, `pin` = the read's playing commit (any `main` commit carrying the rule; one for
  the whole read); `models`: three X48 `final` refs (labels `x48kffn_s350x`) and three C0 `pinned` refs by the
  sha256s of §5.1 (labels `c0_s350x`), plus the six +10M nodes for §8.4; `s` group `pool` (the refpop and its content
  sha, `panel` null, `pairs` **2000**, `batch_pairs` 100, family `x35_x48`, the six finals); `s` group `slope`
  (`pairs` 500, the six +10M nodes); anchor group `unfamiliar` (`metamon:Kakuna`, `["forum72"]`, unit seeds `0-49`,
  the six finals); `decision`: rule `n4_screen`, `params` {on, off, `s_group` `pool`, `anchor_group` `unfamiliar`,
  `s_bar` 2.0, `k_bar` 0.015, `confidence` 0.90, `z` 1.6449, `rule8_eps` 1e-9, `mixed_veto` true, `void` {`k9b_dir`,
  the three X48 run dirs}}.
- **The record** is `../x35_breakers_stage_2026-10-17/reads/x48_look1.json`, arm label `X48` (`batch.READ_LABEL`).

## 11. The hand-off: the draft chain items

The three items are in **`~/.claude/jobs/x35/chains/drafts/x35-x48-oct27.items.json`** (NOT a chain file, NOT
appended to `x35-lrf-oct26.json`), for a new chain file the Training session writes (GPU plan §4.3). Each item:
`"arm": "x48"`, `"pin": "f4b25238def57900c0305bb2a3a5a8f9edaef285"`, `"run": "rb_x35_x48kffn_s350x"`, `"go_by"`
naming this PREREG's commit, `"thin": "planner"` (it keeps the 10M nodes §8.4 reads and the final; no s_eq read is
registered), `"banner": {"expect": "[GrowFork] grown:", "void": ["[GrowFork] FATAL", "[ModelVersion] FATAL", "does NOT
inherit"], "within_min": 20}`, `"throughput": {"min_steps_per_s": 1000, "within_min": 60, "first_line_within_min":
90}`, `"on_void": "stop"`, `"on_throughput_fail": "page"`. The `[GrowFork] grown:` line has not been seen in a
production launcher log yet; the first GO is watched by hand (GPU plan §4.3).

## 12. Cost, and P(GO) per full cost (the strategy doc's ranking unit)

| item | GPU-h | CPU lane-h | note |
|---|---:|---:|---|
| X48 s3501–s3503 | **≈ 13** | — | §2 |
| S: 6 finals × 7 × 4,000 games + 6 nodes × 7 × 1,000 (§8.4) = 210,000 games | (≈ 0.5–1 in a GPU window) | **≈ 31 S-lane-h** | ESTIMATE from G1: 42,000 games in 6.25 h wall (`x35_g1/driver.log`); a GPU window at 118 games/s pooled is UNVERIFIED for this net |
| K: 6,000 forum72 games | 0 | **≈ 26–38 Kakuna lane-h** | N4 review §3's rate |
| G-F (§8.5): 9 × 1,000 games | 0 | ≈ 1.5 | |

Full cost ≈ 13 GPU-h + ≈ 26–38 Kakuna lane-h ≈ **1.6–2.1 resource-days** (the strategy's unit, the S lane apart).
**P(GO) 0.28–0.36 over the GUESS** (§1), so **≈ 0.13–0.22 GO per resource-day**. A screen GO is not the strategy
table's P(GAIN) on the PR rule (0.06–0.12 per resource-day there): it buys a production re-read, not an adoption.

## 13. Validation at this registration, and NOTES

CPU only, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15, from the main checkout (nothing created):
- `kit describe --arm x48 --seed 3501` → [`argv_x48_s3501.txt`](argv_x48_s3501.txt), exactly the GPU plan §4.2's argv.
- `main.checkargs --pin f4b25238…` → [`checkargs_x48_s3501.txt`](checkargs_x48_s3501.txt): **exit 0**, 8 flags
  accepted by the PINNED parser, 0 unrecognized, 70 inherited from X26's record, `declared growth (--grow) :
  trunk_ffn_by_kind 'off' -> 'on'`, the only ARCH-surface difference from production `trunk_ffn_by_kind 'on'`, `✓ fork
  arch`.
- `main.launcher --dry-run --restart-interval-hours 24 --pin-commit f4b25238… <argv>` →
  [`dryrun_x48_s3501.txt`](dryrun_x48_s3501.txt): **role FORK of `rb_x26_s2601`**, +20,000,000 steps (280,000,170 →
  300,000,170), pin `f4b25238` (source `pin_commit`), `grow : DECLARED trunk_ffn_by_kind 'off' -> 'on'` with the
  seeded pool grown the same way, `✓ fork arch`, 12.67 GiB REQUIRED vs 346.64 GiB free. **Its one refusal is the
  CUDA-visibility preflight** (`CUDA_VISIBLE_DEVICES=''`, rule 5: this unit holds no lease): exit 3, "would NOT
  launch", as D3's and N1-P's registrations. The chain's own dry run re-runs it with the card visible before every GO.
- `kit_test`: the kit's argv equals the validated file token for token, and `check_validation` PASSes these two
  transcripts as X48's (the hidden-card line alone modelled as visible) and FAILs them read as C0's or with X47's
  growth line.
- `python -m main.ops.arm_chain plan <draft> --check-pins --describe` on a scratch chain file holding the three items →
  [`arm_chain_plan_x48.txt`](arm_chain_plan_x48.txt).

**NOTES** (appended before the first GO, never after any X48 number):

**NOTE 1 (2026-10-27, before the first GO; no X48 run exists).** The pre-launch review [`REVIEW.md`](REVIEW.md)
(`77d33ed2`) §7 item 2, adopted whole. It tightens one VOID guard, corrects one stated reason and adds two labels.
§5's classes, bars, α, seeds, pin and power are UNCHANGED; there is no look 2.

- **(a) VOID 4 made exact (review M2).** A seed is VOID unless its run's `model_config.json` `growth` has
  `optimizer_state` == {`carried` 280, `no_state` 0, `fresh` 0, `copied` 48}, `new_parameters` == {`tensors` 48,
  `elements` 791,040}, and `function_check.bit_identical` true with `max_abs_d_logits` and `max_abs_d_value` both
  0.0. These are the values MEASURED on P\* at the pin (review §1). Any other value is VOID: in particular `copied` 0
  (a reset optimizer state, Q-K3's rejected fresh-moments alternative), which "has no `copied` count" did not catch.
  It is checked **by a script at arm end, beside `kit k9b`**, before the seed enters any read; never by hand. The
  chain's banner now expects `fresh for 0, COPIED for 48` (the Training session's chain file, on the coordinator's
  word), so a different growth line STOPs the run within 20 min.
- **(b) "experts not engaged", a weak NO-GO (review M1).** A NO-GO is labelled "experts not engaged" when, at the
  final, §8.7's D_k / null ≤ 1.0 for every non-Pokémon kind in rounds 0–1, averaged over `linear1` and `linear2` and
  over the three seeds (rule 8: a value within 1e-9 of 1.0 counts as ≤ 1.0). Round 2 is excluded: its board, threat
  and event experts get no policy or value gradient by construction. The read then states that the experts did not
  diverge more than two copies of C0's own FFN drift would. §8.7's script runs before the class is published.
- **(c) VOID 6's reason corrected, and the NO-GO text (review M1).** §4 item 6's last sentence is wrong. It now
  reads: "Copied experts add 30 % parameters, which move SLOWER than η at first: each inherits the shared FFN's
  second moment, built from all 64 tokens' gradient (Q-K3; review §2: ≈ 0.35–0.65 of their own-calibrated rate over
  20M for the board, threat and event experts)." The 0.08 bar itself stays (C0's trailing approx KL is 0.018). Every
  NO-GO text gains "with the experts' AdamW state copied from the shared FFN (Q-K3)".
- **(d) "LR-elevated", a weak GO (review M3).** A GO is labelled "LR-elevated" when X48's three-seed mean LR is
  > 1.1× C0's (rule 8 at 1e-9), from the same `snapshot_history` read as §8.3. It is the mirror of §8.3's
  "LR-confounded" NO-GO: throttled experts may lower the per-update KL and keep the controller one ×1.2 rung higher,
  which is LR-F's own lever. C0's mean LR is 1.02–1.05e-4 (review §7).
- (b) and (d) are labels, not classes, like "LR-confounded": the `n4_screen` rule's params or post-rule lines.

**NOTE 2 (2026-10-27, before the first GO; no X48 run exists).** Ruling 3 of the precommit memo `endstate/precommit_reads_2026-10-27.md` §4, accepted by the orchestrator: on an LR-F GAIN, X48 is not stopped. It finishes and is read as registered (§5 unchanged), and its record carries the label "read at the controller's learning rate"; N4 / N4-P re-register with the floor.

**NOTE 3 (2026-10-27, before the first GO; no X48 run exists): the rule as built.** `main.reads` rule `n4_screen`
(`src/main/reads/rules_n4_screen.py`) exists with M2's MIXED; [`read_spec.draft.json`](read_spec.draft.json) passes
`check-spec --pin-pending`. Where it differs from §10's list, and how NOTEs 1-2 land:
- `on` / `off` are model-label lists (N4 §7's wording). Added params: `nogo_scope` (M5's text: lever "per-kind FFN
  experts", dose "20M", start "P\*", and NOTE 1 (c)'s `qualifier`), `pin_drift` (§8.6: 56.98 and its source) and, at
  registration, `slope` (§8.4: the six +10M nodes as `pinned` refs by sha256, 500 pairs; X48's do not exist yet, so the
  draft omits them). `void` also registers the run-dir guards `pin` (VOID 3) and `pool_seed` (VOID 5's schema and
  fork step; its `grown` block is not read by the guard).
- `near_scope` is `bound`, §5.3's "a bound within 1e-9 ... is NOT past it" (N4 §5.3's "that meter is UNDECIDED" is
  `meter`, for N4's own spec).
- §8.3's mean LR is each run's mean `snapshot_history` `lr` (`metadata.json`), averaged over the three finals. The rule
  applies "LR-confounded" (NO-GO, < 0.9×) and NOTE 1 (d)'s "LR-elevated" (GO, > 1.1×); a ratio within 1e-9 of a bar is
  not past it. NOTE 1 (b) "experts not engaged" and NOTE 2's "read at the controller's learning rate" need inputs the
  cells do not hold (§8.7's script; LR-F look 1's class): they are post-rule lines, named in the decision's
  `not_reported`. So are the approx KL, ρ, G-F, the experts' divergence, home72 and K's power at φ̂. The UNDECIDED
  line's power is S's, a seeded Monte Carlo at the observed σ.
- "A K unit short beyond the `main.reads` refusal budget" is `d1`'s floor: at most one uncounted 20-game unit per arm
  (2,980 of 3,000 games); an unplayed unit refuses the record.
- A VOID seed refuses the decision at the registered gate (K9(b), pin, pool seed) instead of reading INCONCLUSIVE;
  NOTE 1 (a)'s exact growth stamp and VOID 6's KL stay the arm-end script's and the operator's.
- The record carries R1 under `reported.r1_look1`, never at the top level, so `batch.final_class` cannot read R1 as
  X48's class.

**NOTE 4 (2026-10-28, the read's registration: after the three X48 finals exist, before any X48 read game; no S or
K number of X48 has been seen).** The pre-read gates, the pin and the registered spec. §5's classes, bars, α and seeds
are UNCHANGED; there is no look 2. All of it was run CPU only (`CUDA_VISIBLE_DEVICES=""`, one torch thread, nice 19 on
cores 5-7,13-15), read-only on `models/`.

- **(a) VOID 4, exact (NOTE 1 (a)): PASS on all three seeds.** [`void4_growth_stamp.py`](void4_growth_stamp.py) →
  [`void4_growth_stamp.json`](void4_growth_stamp.json). Both stamps (`model_config.json` `growth` and the lineage's
  `grow`) are equal and hold exactly carried 280 / no_state 0 / fresh 0 / copied 48, 48 tensors / 791,040 elements,
  `bit_identical` true with both max |Δ| 0.0, parent P\*. Each `launcher_child.full.log` holds the banner `fresh for 0,
  COPIED for 48` once and no void line. (`launcher_child.log` is a ~1 MiB ring buffer whose head is trimmed, so the
  banner is read from the full log.) VOID 1: `num_timesteps` 300,061,281 / 300,061,657 / 300,061,319. The registered
  VOID gate (K9(b), pin, pool seed), pre-run with main's `rules.void_gate` on the spec: PASS for all six ON labels
  ([`void_gate_preread.json`](void_gate_preread.json)). VOID 6: trailing-21-rollout approx KL max 0.0202 / 0.0205 /
  0.0199, median 0.019 (bar 0.08; [`pre_read_kl.txt`](pre_read_kl.txt)).
- **(b) "Experts engaged" (§8.7, NOTE 1 (b)): the label WOULD apply.** [`expert_divergence.py`](expert_divergence.py)
  → [`expert_divergence.json`](expert_divergence.json). Fixed here before any game: the null for each (round, linear)
  is √2 × the mean over C0's three seeds of C0's own FFN drift from P\* at the same node. R_k is the mean over the
  three X48 seeds and both linears of D_k / null. At the final, R_k for rounds 0–1 spans **0.72 (r1 event) to 0.93
  (r0 move)**: every one is ≤ 1.0, so **a NO-GO is labelled "experts not engaged"**. A seed-matched null gives the
  same (min 0.72). At +10M the span is 0.68–0.82. Drift from P\* over 20M, as a fraction of the mon FFN's own: move
  experts 0.78–0.82×; board, threat and event 0.24–0.43× in rounds 0–1; round 2's threat and event experts 0.04×
  (no policy or value gradient, review M1). This is the throttle review M1 predicted.
- **(c) Mean LR (M4, NOTE 1 (d)): neither label can apply.** By the rule's own read (`rules_n4_screen.mean_lr`,
  [`pre_read_lr.json`](pre_read_lr.json)), X48 has 1.0482e-4 (nodes at 1.2056e-4: 6, 6, 1 of 20) and C0 has
  1.0348e-4 (5, 2, 2 of 20). The ratio is **1.013**, inside [0.9, 1.1]. `main.dose`
  ([`pre_read_dose.md`](pre_read_dose.md)): median LR 1.005e-4 and dose rate 2.044e-8 on all six runs.
- **(d) NOTE 2's label does not apply.** LR-F look 1 is UNDECIDED, final (`reads/lrfloor_look1.json`), not GAIN
  (precommit memo §2.2, its 2026-10-28 line).
- **(e) The pin: the X35-frame shared play pin `86cfdf7efbe21b4ff3382541e6d78390df7d9966`** (tag
  `x35-shared-play-86cfdf7e`; `designs/tools/reads.md` §4). It is not a main commit, so this departs from §10's "any
  `main` commit carrying the rule". The reasons:
  1. It carries the rule: `a05a0ca2c` is its ancestor, and `rules_n4_screen.py` is byte-identical to main's.
  2. It plays both arms on the game they trained on: E1, with `f4b25238`'s model, observation, inference and Rust
     trees. A main pin would play on an unbanked engine and across `d8d674b99` (an observation fix that changes OUR
     side's inputs), and with our post-H24 pack, unlike the arms' training.
  3. Kakuna plays the post-H22 pack (§5.1).
  4. It is the frozen pin reads.md assigns to "X48's C0 replay", so X48's C0 rows are the C0 bank the other X35-frame
     reads share (review m6).
  5. G1's banked C0 S is on E1 and regime `adfdbee824c9eb0a`, as is this pin's S lane (LR-F, and (g) below). So
     §8.6's pin-drift line now differs by pair count and harness only, not by engine.
- **(f) The spec: [`read_spec.json`](read_spec.json), sha256 `94bb64dcc6c3409ac9581860dd488a7c26713d066b6c72c4b8e82e3a70d3478a`.**
  `check-spec` exits 0 at the pin and at main. Against the draft:
  - `pin` added.
  - The **slope group** (§8.4, family `x35_x48_slope`, 500 pairs) plays the six +10M nodes, each a `pinned` ref by
    sha256: X48 `checkpoint_290000046` / `_290000083` / `_290000182`, C0 `_290000009` / `_290000046` / `_290000131`.
    Thinning kept all six.
  - **`regime_id` `adfdbee824c9eb0a` is registered on both S groups.** The pair count does not enter it.
  - `void.runs` gains the three +10M labels, on the same run dirs, because the rule gates every ON label.
  - **`rederived_guards` is absent, and it resolves to `count`.** The pin's spec parser predates the field
    (`29b90161f`) and refuses it. Main's `check-spec` prints `rederived_guards: count`, because this spec's sha256 is
    not in `rederive.REGISTERED_BEFORE`. G14's re-derivation runs at assemble, and `record` / `decide` run from a main
    checkout (reads.md §1: operator verbs from the newer checkout).
  - The X48 finals stay `final` refs, which settle to sha256 `85c74453…f9446` / `2463a178…409ce8` / `2b8c3a4e…290e`.
- **(g) The S-lane identity at the pin, before any X48 read game** (as LR-F did): C0 s3501's G1 batch 0 re-played
  against `x26_pstar` and `tstar` from this read's play worktree ([`run_s_identity.sh`](run_s_identity.sh),
  [`s_identity_compare.py`](s_identity_compare.py) → [`s_identity.json`](s_identity.json)). **IDENTICAL**: both cells equal G1's rows on regime (`adfdbee824c9eb0a`), seed block,
  counts (109-90-1 and 121-79-0) and both outcome digests, with env core `7c532805…/125`.
- **(h) Where it plays.** The play worktree is `/home/goodlad/dev/gen3ai-reads/x48_play_86cfdf7e`: a detached checkout
  of the pin, `bootstrap.sh --skip-env`, and E1's untracked `Cargo.lock` blobs and Rust build copied from LR-F's play
  worktree. `main.anchors --dry-run` prints `port=7d932165eced1c79/90;env=7c53280566d666ae/125` (E1). The state is
  `~/gen3ai_archive/x35_x48_read/look1/`. Both lanes run at `--concurrency 1` with a registered stop of `--hours 168`,
  resumable. The record goes to `../x35_breakers_stage_2026-10-17/reads/x48_look1.json` (§10).
- **(i) Not played** (reported lines the spec does not hold): G-F (§8.5) and home72 (§8.8). Each is its own read, run
  only if the lanes have room after the record. ρ (§8.2) is read from the dose records at the record.

**NOTE 5 (2026-10-29 ≈ 03:00, before any X48 S number exists; no X48 S row was written).** The S lane cannot play an
X48 cell at the pin, so X48's S cells play through [`x48_s_driver.py`](x48_s_driver.py).
- **What happened.** The lane's first X48 unit (02:23) refused at preflight, before any game:
  `CellArchMismatch: cell opponent tstar@290000087: its architecture matches neither of the engine's 2 slot groups`.
  An h2h engine at the pin serves at most `main.h2h.arch.MAX_GROUPS` = 2 architectures. A C0 final shares one
  architecture with six of R's members (production, `x26_*`), and `tstar` is the second, so every C0 engine had two.
  The X48 final is a third (per-kind FFN experts). `runner.s_engines` stripes a label's members over engines
  (`left[w::k]`), so it puts `tstar` beside the others, and the supervisor exited rc 1.
- **Why not a code fix.** The runner is off the play path, but a pinned spec cannot `repin`. Neither the registration
  nor §5 moves.
- **The driver.** It is the S lane's own loop with one change: it groups each label's pending members by architecture
  before it puts them on engines. Engines are `x48kffn_s350x:production,x26_*` and `x48kffn_s350x:tstar`. Everything
  else is the runner's own code at the pin:
  - R is re-verified before every unit, and the pending cells come from the runner;
  - `s_unit_argv` gives the heavy slot, nice 19 and the agent cores;
  - the `s-unit` verb re-hashes every model and plays `play_many_argv` with the same pairs, batches, family, request
    and ledger;
  - the 2,400 s box, 2 workers and the registered stop are unchanged.

  The pairs, seeds, regime and rows are the ones the runner would write; only which process plays which cell differs.
  The S lane is HELD, so the runner cannot start on it, and the driver also plays the C0 slope cells.
- **No change to the measurement.** Grouping cells into processes was already free under the runner: C0's units
  regrouped cells from unit to unit.

## 14. FINDINGS at registration (standing rule 7)

1. **The read's rule does not exist yet.** `n4_screen` is N4's, to be built with T69, and needs M2's MIXED class. X48's
   finals land ≈ 10-28 midday and its read waits behind the Kakuna lane anyway (GPU plan FINDING 6), so the build is
   not on the GPU's critical path. Until it exists, no X48 read game is played (§10).
2. **K is the weak meter.** At 1,000 F games a final, GO-K at a true +3 pp is ≈ 0.6. Goal (b)'s question is answered
   with low power here; a sharper unfamiliar read is the PR rule's (≈ 107 lane-h), registered only on a GO.
3. **σ_S at 2,000 pairs is an ESTIMATE** from three C0 finals at 500 pairs (2 df). If it comes out near 1.07, the
   screen reads UNDECIDED often (§6).
4. **ρ and the LR comparison cross days.** C0 trained 10-21 to 10-24; X48 trains from ≈ 10-27 22:00 on the same card
   under a different box load. ρ is reported only, and the screen reads finals at matched steps.
5. **The canary ruling's mechanism claim (§7.2) is UNVERIFIED** on the compiled graph. W0's census is the fix.
6. **checkargs' growth note says "new parameters start fresh"** while B-KFFN's stamp records the copies' moments as
   COPIED (Q-K3). The note is generic to every growth; VOID guard 4 reads the stamp, not the note.
7. **An X48 NO-GO is narrow by design:** per-kind specialisation at the same width, 20M, the controller's LR. It does
   not veto N4-P (review M6).
8. **The descriptor (§8.7) needs a small script** at read time (weights only; `pin_worker depth` refuses kffn
   checkpoints, F-GB-10).
