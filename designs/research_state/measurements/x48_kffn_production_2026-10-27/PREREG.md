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
