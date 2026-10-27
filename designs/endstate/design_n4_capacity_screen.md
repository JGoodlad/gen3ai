# N4: the toy capacity screen. Is the network too small to escape the plateau?

**Status: DRAFT (2026-10-27). REGISTERED as
[`../research_state/measurements/n4_capacity_screen_2026-10-27/PREREG.md`](../research_state/measurements/n4_capacity_screen_2026-10-27/PREREG.md).
T69's MODEL SIDE is built (2026-10-27, `--grow trunk_ffn_dim`); nothing is launched.** This doc makes runnable the capacity branch of
[`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md) §4 (T3). The plasticity check
(`8bebdfe3`) read NO-LOSS, so T3 takes this branch
([`../research_state/measurements/plasticity_check_2026-10-26/README.md`](../research_state/measurements/plasticity_check_2026-10-26/README.md)).
N4 needs one build first, T69 B-FFNW (§2), and then about 30 GPU-h (§3).

**Abbreviations.**
- **T\*** is the toy's plateau checkpoint: X34 (`rb_toy_ent_s4002_deep`) at 290,000,087 steps.
- **P\*** is production's plateau checkpoint: X26 at 280M.
- **C0** is a plain-continuation control. The banked toy C0 is s3601–s3603 at the toy pin `4609c155`. **C0′** is
  the new one N4 runs at its own pin.
- **S** is in-house strength: `main.x35.readout.population_score`, the equal-weight mean over the six members of
  the toy's frozen reference population R (`reference_population_toy.json` in the X35 stage).
- **K** is the Kakuna read: our model against Metamon's Kakuna on `forum72` (F, the unfamiliar panel).
- pp = percentage points. LB / UB = lower / upper bound.
- **GPU-h** = hours of our card. **lane-h** = one wall-hour of the Kakuna anchor lane at 160–230 F games an hour
  (strategy doc, abbreviations).
- **GUESS** = an unmeasured effect. **ESTIMATE** = a cost from measured rates. **MY JUDGEMENT** = an inference no
  read states.

---

## 0. One screen

- **The question.** If the trunk had more room, would the plateau move? Concretely: at equal data (matched steps),
  does a toy plateau fork with a wider trunk score higher than the same fork without it?
- **The arms. One lever, a growth fork against a same-pin control:**
  - **N4:** T\* + `--grow trunk_ffn_dim=1024`, +70M steps. Each trunk round's feed-forward block (FFN) grows from
    256 to 1,024 hidden units. The 768 new units start with zero output weights, so the child computes the
    parent's function bit for bit at the fork.
  - **C0′:** T\* with the same argv minus `--grow`.
  - Both run at ONE new pin, seeds 3641–3643.
- **Why not the strategy doc's sketch.** It sketched a fresh toy at twice `D_MODEL`, at matched wall clock. That
  design answers a different question, because the toy's own record reverses between equal compute and the
  plateau (§1.1).
- **The dose.** +70M from T\* (to 360,000,087), as X47 and the banked toy C0.
- **Cost: ≈ 30 GPU-h** (ESTIMATE):
  - C0′ ≈ 3 × 4.6 GPU-h, measured at the toy pin;
  - N4 ≈ 3 × 5.5 GPU-h, at a GUESSED 1.2× wall per step.
  - It is ≈ 2× the strategy doc's 13–15 GPU-h: the banked toy C0 cannot serve as control (§3.2, FINDING 1).
  - Plus ≈ 26–38 lane-h of Kakuna and ≈ 3 CPU-h of S cells and mechanism reads.
- **The GPU window.** The current arm chain `x35-lrf-oct26`:
  - LR-F s3502 RUNNING since 07:24 PDT 10-27; s3503 PENDING;
  - ≈ 4.1 h a 20M arm, so the card frees ≈ 16:00 PDT 10-27.
  - N4 launches in the first window after T69 lands (≈ 1–1.5 agent-days, so ≈ 10-28 to 10-29), one fork at a
    time, ≈ 1.3 card-days.
- **The read (§4): a SCREEN, one look, two meters.** Each class is set by one-sided 95 % bounds on the 3 + 3 seed
  finals:
  - **S:** GO-S iff the LB of N4 − C0′ is > 0; NO-GO-S iff the UB is < +2.0 pp (the X35 bar); else UNDECIDED-S.
  - **K:** GO-K iff its LB is > 0; NO-GO-K iff its UB is < +1.5 pp (PR-1's bar); else UNDECIDED-K.
  - **Overall:** GO iff GO-S or GO-K. NO-GO iff NO-GO-S and not GO-K. Else UNDECIDED.
  - **Power** (`power.log`): a true +3 pp on S reads GO-S 0.83–0.98, and a true 0 reads GO-S 0.04 and NO-GO-S
    0.52–0.77. K is a large-effect detector only: +4 pp reads GO-K 0.81–0.89 at 1,000 F games a final.
- **What it implies for production (§6).**
  - **GO** licenses a production width fork, read by the PR rule against Kakuna. That fork needs its own pin
    decision (FINDING 3).
  - **NO-GO** retires "more trunk FFN width at the plateau" for the toy. It transfers WEAKLY, because production asks
    the same trunk widths to carry more input.
  - **X48** (per-kind FFN experts, built, at the banked production C0's pin) is the production-side capacity read
    that needs no toy and no build. It is the complement, whatever N4 reads.
- **The mechanism read (§5)** is reported, never deciding:
  - **engagement:** the new units' output-weight norm, and the policy shift when the new branch is zeroed;
  - **the capacity battery v3** (`python -m main.capacity`): rank, trainability and tanh saturation at every final,
    on shared states.
  - It splits a NO-GO three ways: the width was used and bought nothing (the strongest NO-GO), the optimizer never
    recruited it (a weak NO-GO), or GO.

---

## 1. Why this design

### 1.1 Four candidate designs, one kept

| design | what it measures | why kept or dropped |
|---|---|---|
| **A. Fresh toy at 2× `D_MODEL`, matched wall clock** (the strategy doc's sketch) | sample efficiency per GPU-h, short of the plateau | **DROPPED.** The toy's own record reverses between the two regimes. At production's 15M budget and equal wall clock, the 1.42M toy beats production by +6.5 pp [+4.5, +8.4]. At the plateaus, P\* is 7.3–10.5 pp above the toy's T\* against Kakuna (UNDERSTANDING §4.9). So a matched-clock fresh pair is biased against the wider, slower net and answers "which learns faster per hour", not "which plateaus higher". Taking both fresh to the plateau costs ≈ 3 seeds × 2 arms × 18–27 GPU-h ≈ 110–160 GPU-h. That is the scale bet's price, not a screen's |
| **B. Growth fork at T\*, depth** | does a deeper trunk lift the plateau? | **DONE as X47** (2 → 4 rounds, `d6e57f2e9`): UNDECIDED with Δ̂ ≤ 0, and at matched steps −0.41 pp [−2.34, +1.51]. Under N4's screen rule (§4), X47's matched-step finals class **NO-GO-S** (90 % two-sided [−1.56, +0.73], computed from its record) |
| **C. Growth fork at T\*, per-token-kind FFN experts** (`--grow trunk_ffn_by_kind=on`, B-KFFN) | parameters without FLOPs | **REFUSED on the toy.** `main.checkargs` at HEAD `90708bff` (2026-10-27, CPU): "unsatisfiable combinations: 4 ✗ WOULD FAIL IN THE EXTRACTOR". The experts need `entity_tail_seats`, `entity_topk_seats`, `history_events` and `opp_intent`, which the toy turns off. It is production's X48 (§6) |
| **D. Growth fork at T\*, FFN width** (this doc) | at equal data, does more trunk capacity lift the plateau? | **KEPT.** The width axis is the one not yet tried. It is function-preserving at the fork, so it starts at the plateau exactly, as X47 did. Its dose and control frame are X47's, so the two growth reads sit side by side (§4.3) |

**Why the FFN, not `D_MODEL` or the policy MLP.**
- **`D_MODEL` cannot grow function-preservingly bit for bit.**
  - Duplicating every residual dimension keeps LayerNorm exact in real arithmetic, but halved weights summed in a
    longer reduction are not bit-identical in fp32. The growth gate admits no tolerance (ruling Q-K2,
    `../model/versioning.md` "Declared growth at a fork").
  - Duplicated units also stay symmetric until noise breaks them.
- **The FFN is where a transformer's parameters scale.**
  - The convention is an FFN 4× `D_MODEL` wide (Vaswani et al. 2017; from memory). Ours is 2× (`TRANSFORMER_FFN_DIM`
    256 over `D_MODEL` 128). N4 takes it to 8×.
  - At fixed parameter count, shape matters little for a language model's loss (Kaplan et al. 2020; from memory,
    UNVERIFIED for RL), so FFN width is a fair proxy for "more capacity in the trunk".
- **The policy MLP was not chosen.**
  - Its tanh units do saturate across the plateau, but that did not slow refitting (plasticity check F3).
  - The piloting deficit goes with the TEAM context, which the trunk builds (piloting-gap forensics, STATUS §3).
  - Widening it touches SB3's `net_arch` and `latent_pi`'s consumers, which is a larger build.
  - The battery's tanh probe watches it anyway (§5).
- **The literature for width on generalisation.** In Procgen, ×2 / ×4 IMPALA width improves train and test returns
  (Cobbe et al. 2020, VERIFIED in
  [`../research_state/literature/context_generalisation_2026-10-27.md`](../research_state/literature/context_generalisation_2026-10-27.md)).
  There it was from scratch, at many levels: a growth fork at a 719-team plateau is a narrower test (§6.2).

### 1.2 What `--grow` does on the toy today (checked 2026-10-27, CPU, no launch)

All three were checked on T\*'s checkpoint `checkpoint_290000087_steps.zip`, at HEAD `90708bff`, with
`CUDA_VISIBLE_DEVICES=""`:

| argv (the toy C0 argv + …) | `main.checkargs` | `main.launcher --dry-run` |
|---|---|---|
| `--grow trunk_layers=3` | "declared growth: trunk_layers 2 -> 3"; ✓ fork arch compatible; ✓ this command still launches | "FORK of …/rb_toy_ent_s4002_deep", +70,000,000 steps, "grow: DECLARED trunk_layers 2 -> 3", 14.74 GiB disk needed (351.67 free); no run dir created |
| `--grow trunk_ffn_by_kind=on` | ✗ 4 unsatisfiable combinations (the extractor would fail) | not run |
| `--grow trunk_ffn_dim=1024` | ✗ refused: "not a growable field (growable: opp_intent_grad_mode, trunk_ffn_by_kind, trunk_layers)" | not run |

So the growth machinery works on the toy at HEAD, and the width field is the build (§2).

**After T69's model side (2026-10-27, `gen3_trunk_ffn_width_v1`, config v164):** `main.checkargs` on T\* with
`--grow trunk_ffn_dim=1024` prints "declared growth (--grow) : trunk_ffn_dim 256 -> 1024" and "✓ fork arch :
compatible". A bare `checkargs` on T\* judges the argv against T\*'s OWN recorded commit `ae2d0901`, which predates
`--grow` (for `trunk_layers=3` too), so the kit must name the pin (`--pin-commit N4_PIN`).

---

## 2. The lever and its build (T69, B-FFNW)

**The function.**
- Every trunk round's FFN position computes `ffn(h) = linear2(relu(linear1(h))) + ffn_extra(h)`, where
  `ffn_extra(h) = linear2_x(relu(linear1_x(h)))`.
- `linear1_x` is `D_MODEL → (trunk_ffn_dim − 256)` with a bias; `linear2_x` is `(trunk_ffn_dim − 256) → D_MODEL`
  with NO bias.
- The sum is exactly an FFN of width `trunk_ffn_dim`: two FFNs over a split hidden layer, sharing the base's output
  bias.
- It is built as a BRANCH, not by resizing `linear1` / `linear2`, so the fork-time gate holds bit for bit:
  - a zero `linear2_x` adds an exact 0.0;
  - a longer GEMM reduction is not guaranteed bit-identical (the B-KFFN finding F-GB-5).
- It is called from the one place every round runs its FFN, `round.ffn(h)` (`agents/model/trunk_kffn.py`'s
  docstring). So the bf16 trunk region covers it, as it covers B-KFFN's experts.

**The init.**
- `linear1_x` gets the init the existing rounds' `linear1` get: orthogonal, gain √2, bias 0. It is an
  `IsolatedLinear` built inside `torch.random.fork_rng` from a new private seed in `arch_constants.py`, the
  `TRUNK_EXTRA_INIT_SEED` pattern.
- `linear2_x` is ZERO.
- A fresh build at `trunk_ffn_dim = D` builds the same modules, so a grown state dict equals a fresh child's.
- A fresh `trunk_ffn_dim` build is otherwise untested here (no N4 arm uses it).

**The field.**
- `trunk_ffn_dim`, an integer ≥ 256, default `TRANSFORMER_FFN_DIM` (256). The default stays in `arch_constants.py`
  (root rule: architecture constants live only there).
- It is a `ModelVersion` field with a config-version bump and a migration that reads its absence as 256.
- A `growth.GROWABLE` row (`parse`, `validate`: child > parent, `current`, `grow`, no copy map).
- **Refused in this build** (typed): together with `trunk_ffn_by_kind on`. Neither arm needs the pair, and the
  experts would have to decide whether to copy the branch.
- New parameters get fresh AdamW state (the B-GROW rule, `growth.regrow_optimizer`). The seeded self-play pool grows
  the same way (`grow_fork.grow_seeded_pool`).

**Sizes** (counted from the shapes):

| | per round | toy (2 rounds) | production (3 rounds) |
|---|---|---|---|
| new parameters | 99,072 (`linear1_x`) + 98,304 (`linear2_x`) = 197,376 | **394,752**: the extractor 848,897 → 1,243,649 (+46 %), the policy ≈ 1.42M → ≈ 1.81M (+28 %) | 592,128: 2,609,183 → 3,201,311 (+22.7 %; X48 adds 791,040) |
| FFN MACs per token | 65,536 → 262,144 (×4) | the trunk's per-token matmul MACs ≈ 2.5× | the same per round |

**The wall per step (ρ, a GUESS).**
- X47's two extra rounds added ≈ 262k trunk MACs per token and measured ρ = 1.136 (`toy_x47_look1.json`).
- N4 adds ≈ 393k, so a linear extrapolation gives ρ ≈ 1.2.
- The rollout's CPU side does not grow, so this is an upper-side GUESS. Measure it in the build's GPU smoke.
- **A CPU hint (T69's smoke, 2026-10-27; not a GPU number).** Two `--debug` forks of a 4,096-step toy parent per arm,
  +16,384 steps each, interleaved (C0, grown, grown, C0), on 3 shared agent cores beside a live run and other agents'
  reads. Updates 2–7 of each: the grown arm's update took 1.19× (train) and its collection 1.24× the control's, 1.23×
  per update pooled; but the two seed pairs read 1.05× and 1.36× per update, so contention dominates. It does not
  contradict ρ ≈ 1.2, and it does not measure it.

**Build list (T69, ≈ 1–1.5 agent-days, opus-high: a model-version change with a bit-identity gate):**
1. The field, the branch, the migration and the `GROWABLE` row, with tests that fail on revert:
   - the gate: parent vs grown child bit-identical on the 64 real rows after one AdamW step;
   - the grown state dict equals a fresh child build's;
   - the pool grows;
   - the refused pair.
2. **Kit arms** in `main.x35.kit` / `batch`: `toy_c0n4` (`rb_x35t_c0n4_s{seed}`) and `toy_n4ffn`
   (`rb_x35t_n4ffn_s{seed}`), seeds 3641–3643.
   - Both arms are judged against ONE arm-level pin `N4_PIN`, not the toy side pin (AM-1 / AM-8 keep `4609c155`
     for batch 1).
   - `toy_n4ffn` is the `toy_c0n4` argv token for token + `--grow trunk_ffn_dim=1024`.
3. **The descriptor readers**, generalising `main.x35.x47 norms` / `policy-diff` to any declared growth through a
   per-row `zero_new` hook (§5).
4. **The read rule** `n4_screen` in `main.reads.rules` (PREREG §5):
   - S from an `s` group;
   - K from an `anchor` group (forum72);
   - `welch_interval` at confidence 0.90;
   - the `d1` estimators (`strat_score`, `overdispersion`, `widen`) at z = 1.6449;
   - the VOID gate (K9(b) + the growth stamps).
   - It ships with a 2-pair smoke of a grown toy against T\* and against P\*, through `main.h2h` and through
     `main.anchors` against Kakuna (the X47 smoke's pattern: three architectures need split engines).

---

## 3. Arms, dose, seeds, cost, the window

### 3.1 The arms (PREREG §3 holds the argvs)

| arm | kit arm / run | the argv | role |
|---|---|---|---|
| **C0′** | `toy_c0n4` / `rb_x35t_c0n4_s3641..3643` | the banked toy C0's argv (`--model <T*> --behaviour-check warn --steps 360000087 --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 --supply-starve-cycles self_play_pool=0 --device cuda --seed <s>`) at `N4_PIN` | the same-pin, same-dose control |
| **N4** | `toy_n4ffn` / `rb_x35t_n4ffn_s3641..3643` | C0′'s argv + `--grow trunk_ffn_dim=1024` | the ONE lever |

- **The order**, one fork at a time, interleaved (X47's C2 ruling): C0′ s3641, N4 s3641, N4 s3642, C0′ s3642,
  C0′ s3643, N4 s3643. A stop at any point leaves the arms within one run of each other.
- **Thinning.** N4 arms keep every 1M node until the read is complete (the descriptors read them; X47's AM-X47-12,
  ≈ 1.5 GiB an arm). C0′ arms take the planner's thinning (read at finals only).

### 3.2 Why a new control and not the banked toy C0

The banked toy C0 (s3601–s3603) trained at the toy pin `4609c155` (config v159, T62's float64 τ solve). The width
field cannot reach that pin:
- B-KFFN's field-generic `GROWABLE` refactor is NOT an ancestor of `4609c155` (checked: `3f69e6e2e` is in the
  production pin `f4b25238` and not in the toy pin).
- So the D1 / D3 / N1-P "own pin = side pin + one cherry-picked commit, patch-identical to main" route (launch sheet
  AM-D1-FIX-d) has no clean commit to cherry-pick.
- An N4 arm at HEAD against the banked C0 would differ in a week of trainer and engine code, besides the lever.

So both arms run at ONE new pin `N4_PIN`: the first `main` commit that carries T69 and the kit arms.
- **C0′ is reusable.** It becomes the toy's same-pin control for any later toy arm at `N4_PIN`.
- **It gives a pin-drift line for free.** C0′ against the banked C0 (REPORTED, PREREG §6) says how far one week of
  code moved the toy frame.

### 3.3 GPU-h (ESTIMATE)

| item | basis | GPU-h |
|---|---|---|
| C0′ × 3 | the banked toy C0: 4.17–4.27 h from the first own checkpoint to the final (69M) at 4,607 steps/s (TASK_BACKLOG state board), + ≈ 0.3 h start-up and pool seed. At `N4_PIN` the τ solve is FP32_ALT, which may be faster (UNVERIFIED) | ≈ 13.8 |
| N4 × 3 | C0′ × ρ ≈ 1.2 (§2, GUESS) | ≈ 16.5 |
| **look 1 = the whole screen** | no look 2 is registered (PREREG §5.4) | **≈ 30** (≈ 28–33) |
| reads | S cells on the agent cores (CPU); Kakuna on the anchor lane (CPU) | 0 |

**CPU.**
- **Kakuna:** 6 finals × 1,000 F games = 6,000 games ≈ 26–38 lane-h at the measured 160–230 games an hour.
- **S cells:** 6 finals × 6 R members = 36 cells × 500 mirrored pairs, plus G-F's 3 × 3, reported. X47's 63 cells
  took ≈ 50 min on 3 lanes, so this is ≈ 2–3 lane-h of the S lane.
- **The battery and descriptors:** ≈ 1–2 CPU-h.

### 3.4 The GPU window (cited 2026-10-27 08:00 PDT)

- **The arm chain** `x35-lrf-oct26` (`~/.claude/jobs/x35/arm_chain/x35-lrf-oct26.state.json`):
  - D3 s3504 / s3505 are DONE;
  - LR-F `rb_x35_lrfloor_s3502` is RUNNING (GO 07:24 PDT; throughput gate 1,470 steps/s);
  - `rb_x35_lrfloor_s3503` is PENDING.
- **When the card frees.** At D3 s3505's measured dose (245.2 min for 20M at 1,397 steps/s, `28711af06`), s3502
  ends ≈ 11:30 and s3503 ≈ 15:45 PDT, so the card frees ≈ 16:00 PDT 10-27.
- **N4 is not ready then.** T69 must land first. N4's six forks are the first GPU item after T69 ships, unless the
  orchestrator orders otherwise; the queue order is the orchestrator's (memory: "GPU queue order is my call").
- **The hand-off** is a new `arm_chain_v1` file, backend `x35`, six arm items at `N4_PIN` (PREREG §8):
  - `on_void: stop`, `on_throughput_fail: page`;
  - a throughput floor of 2,500 steps/s (UNVERIFIED for N4; C0′ measured 4,607 at the old pin);
  - a banner guard on the growth line T69 prints.

---

## 4. The read

### 4.1 Why a screen rule and not X35 R1

The X35 R1 rule (`main.x35.readout`) classes GAIN only when a two-sided 97.5 % LB clears +2.0 pp, with a look 2 at
five seeds. Its own power on the toy (`power.log`, the registered sequential rule):
- a true +3 pp reads GAIN by look 2 only 0.16–0.32;
- a true 0 reads NULL only 0.16–0.48;
- look 2 runs 0.41–0.48 of the time under a null, and 0.85–0.98 at +1 to +3 pp. Each look 2 costs ≈ +20 GPU-h.

A screen should instead be cheap, rarely say GO under a null, and usually say GO at an effect worth a production
fork. So N4 registers a one-look screen at one-sided 95 % per meter. The R1 class on the same finals is REPORTED
beside it, so the read stays comparable with X47's.

### 4.2 The classes (exact formulas: PREREG §5)

- **The primary estimand is matched steps.** Capacity is a question at equal data, so N4 reads every final at
  360M.
- **ρ is reported.** The equal-learn-time view is a compute question the production fork answers at its own
  cost.

| meter | statistic | GO | NO-GO | otherwise |
|---|---|---|---|---|
| **S** | Δ_S = mean S(N4 finals) − mean S(C0′ finals), Welch, two-sided 90 % (`readout.welch_interval(..., confidence=0.90)`) | **GO-S** iff LB > 0 | **NO-GO-S** iff UB < +2.0 pp | UNDECIDED-S |
| **K** | Δ_K = the N4 finals' pooled F score − C0′'s against Kakuna: the `d1` estimators on `forum72` alone (`strat_score` at z = 1.6449, `newcombe_strat`, then `widen` by √max(1, φ̂) from `overdispersion` on the 3 + 3 finals) | **GO-K** iff LB > 0 | **NO-GO-K** iff UB < +1.5 pp | UNDECIDED-K |

- **Overall.** **GO** iff GO-S or GO-K. **NO-GO** iff NO-GO-S and not GO-K. **UNDECIDED** otherwise.
- **The family false-GO rate** under a global null is ≤ 1 − 0.95² ≈ 0.10 (two meters at one-sided 0.05 each,
  correlated through shared seeds).
- **Rule 8.** A bound within 1e-9 of 0 or of a bar is not past it; that meter is UNDECIDED.
- **Why NO-GO does not need NO-GO-K.**
  - K at 6,000 games reads NO-GO-K only ≈ 0.26 of the time under a null. Requiring it would make the screen
    almost never close.
  - So K can only OPEN the gate, as a large-effect detector, and the record says so.
  - **This is a declared asymmetry.** A NO-GO excludes a ≥ 2 pp in-house effect. It does NOT exclude an
    unfamiliar-only effect below ≈ 4 pp.

### 4.3 Power (`power.py` → `power.log`; 200,000 / 100,000 draws, seeded)

| true Δ | GO-S, σ 0.75 / 1.07 | NO-GO-S, σ 0.75 / 1.07 | | true Δ_K | GO-K at 1,000 F a final, σ_s 0 / 1 pp |
|---|---|---|---|---|---|
| 0 | 0.040 / 0.040 | 0.767 / 0.520 | | 0 | 0.040 / 0.057 |
| +1 | 0.332 / 0.206 | 0.256 / 0.189 | | +2 | 0.402 / 0.367 |
| +2 | 0.791 / 0.528 | 0.016 / 0.030 | | +3 | 0.692 / 0.607 |
| +3 | 0.976 / 0.825 | 0.000 / 0.002 | | +4 | 0.891 / 0.809 |

- **σ** is the per-seed SD of a final's S. The toy C0 finals measured 0.73 and X47's finals 0.53 (s_eq nodes 0.86);
  the X35 registration uses 1.07.
- **The K base** is 0.42: P\*'s F ≈ 0.49 (the strategy doc's power base) less T\*'s 7.3 pp F deficit (`3010c81ba`).
- **K's simulation** uses a normal binomial SE widened by √max(1, φ̂); the registered estimator is `d1`'s Newcombe
  hybrid of Wilson intervals (PREREG §5.2), which agrees with it to well under 0.1 pp at these counts (MY JUDGEMENT).
- **The meters share seeds**, so their errors are positively correlated. The family rate above treats them as
  independent, which is the conservative side for the false GO (MY JUDGEMENT).
- **What X47 reads under this rule.** Its matched-step finals (S 52.014 vs 52.428) give 90 % [−1.56, +0.73]:
  NO-GO-S. N4's rule would have closed depth on the toy too, and the two growth reads compare directly.

### 4.4 Reported, never deciding

- The X35 R1 class at look 1 (97.5 %) on the same finals.
- ρ and the dose records.
- **G-F:** the head-to-head 3 × 3 (N4 finals × C0′ finals, `cross_stat`).
- **Per-member Δ_S**, with P\* (`pstar_x26`) on its own (AM-X47-7's pattern).
- **The pin-drift line:** C0′ finals' S against the banked toy C0 finals' (52.428).
- **home72 against Kakuna**, 400 games a final, ONLY if the lane has room after F. It is the familiarity premium's
  mechanism line: capacity-as-memorisation predicts home72 rises more than F (strategy doc §2, the 0.14 vs 0.28
  logit drop).

---

## 5. The mechanism read (reported, never deciding)

**Engagement (T69's descriptor readers):**
- **(E1)** At each N4 node: ‖`linear2_x`‖_F / ‖`linear2`‖_F per round. The unit is "engaged" when the ratio is
  ≥ 0.01 at the final (AM-X47-10's convention).
- **(E2)** The policy shift on `bank_x26` (7,028 decisions) between each node and the same weights with every
  `linear2_x` zeroed, which restores the narrow function exactly: the mean KL(π_node ‖ π_zeroed) and the share of
  argmax moves, at every 10th node and the final (AM-X47b-3's pattern).

**The capacity battery v3** (`python -m main.capacity <final> --states <C0′ s3641's eval-trace states> --out
<stage>/battery/<run>.json --max-states 2000`):
- Run on all six finals and on T\*, on ONE shared state set, so the readings difference (the battery's own rule 2).
- The `--out` keeps it out of the run dirs.
- The lines read:
  - (a) `team_tokens` / `pi_features` PR and srank99;
  - (b) the trainability ratio at `team_tokens`;
  - (e) the policy MLP's tanh saturation per layer (`f10b2de8`).
- None decides (the battery's validity rule 1).

**The three registered readings** (MY JUDGEMENT; written before any N4 number):

| pattern | E1 / E2 | battery | S / K | reads as |
|---|---|---|---|---|
| **capacity binds** | engaged; KL rising | `team_tokens` srank99 above C0′'s (mean of seeds) | GO | the trunk was short of room at the toy plateau |
| **used, buys nothing** | engaged; KL rising (X47's new rounds moved the argmax on 25–29 % of decisions) | any | NO-GO | the strongest NO-GO: the network spends the room and gains nothing. This was X47's pattern for depth |
| **not recruited** | ratio < 0.01 at the finals; KL ≈ 0 | ≈ C0′ | NO-GO | a weak NO-GO: in 70M the optimizer did not use the new units. A growth-recipe limit (a zero-init branch at an annealed LR), not a capacity verdict |

**What the tanh probe adds.** The plasticity check found the policy MLP's layer-1 saturation rising across X26's
plateau (0.84 → 0.92 of activations; units saturated on ≥ 99 % of rows 6 % → 18 %). If a wider trunk relieves the
head, N4's layer-1 saturated-unit share falls below C0′'s. Either way it is a line, not a verdict.

**The baseline, measured today** (`main.capacity`, CPU, battery v3; ONE set of 2,000 states from toy C0 s3601's
eval traces; the PREREG's `battery_baseline.log`):

| checkpoint | `team_tokens` PR / srank99 | `pi_features` PR / srank99 | trainability ratio, `team_tokens` | tanh l1: sat. acts / sat. units | tanh l2 |
|---|---|---|---|---|---|
| T\* (290,000,087) | 16.68 / 111 | 23.72 / 366 | 1.295 | 0.868 / 0.066 | 0.353 / 0.008 |
| toy C0 s3601 final (+70M) | 15.15 / 109 | 22.06 / 358 | 1.274 | 0.881 / 0.092 | 0.375 / 0.010 |
| X47 s3601 final (depth 4, +70M) | 16.65 / 109 | 23.19 / 362 | 1.286 | 0.886 / 0.104 | 0.382 / 0.008 |

- **The plain continuation saturates the head further.** C0's layer-1 units saturated on ≥ 99 % of rows go
  6.6 % → 9.2 % in 70M, as X26's did across its plateau.
- **Depth did not relieve it** (X47: 10.4 %).
- **X47 kept `team_tokens`' PR at T\*'s level** where C0 drifted down (16.65 vs 15.15), with no S gain.
- **Caveat.** One seed each, no interval. Descriptive only, which is why N4 reads these lines on all six finals.

---

## 6. What each class implies for production: the bridge

### 6.1 What transfers

- **The mechanism transfers by construction.**
  - The same `GROWABLE` row grows production's three rounds (`trunk_layers` 3, `D_MODEL` 128, FFN 256, the same as
    the toy's).
  - **N4-P** = P\* + `--grow trunk_ffn_dim=1024`, +592,128 parameters, in the X35 frame against C0, read by the PR
    rule (strategy doc §3.2: every T2 fork reads by it).
- **The size of an effect does not transfer.**
  - The toy's familiarity premium is about half P\*'s (X51).
  - The toy plateaus 7–10 pp lower against Kakuna.
  - Toy-to-production transfer of a lever has never been calibrated (the X35 toy calibration stayed NOT SHOWN,
    AM-RP).
- **The direction transfers asymmetrically** (MY JUDGEMENT).
  - The toy runs the SAME trunk widths on far less input: its extractor is 0.85M against production's 2.03M, with
    the damage operator, beliefs, entity seats and history events all off.
  - So production asks more of the same width than the toy does. A toy GO is good evidence that production is
    short too. A toy NO-GO is weak evidence that production is not.

### 6.2 What each class does

| class | the capacity branch (T3) | production | the scale bet (rank 12) |
|---|---|---|---|
| **GO** (either meter) | OPEN | **N4-P is licensed**: ≈ 5 seeds × ≈ 4.4 GPU-h × ρ_P + ≈ 107 lane-h at the PR rule. Its pin: production's C0 is banked at `f4b25238` (config v161), which T69 cannot reach by a patch-identical cherry-pick either (FINDING 3). The orchestrator chooses: a backported own pin with a declared-diff check, or a new production C0 at `N4_PIN` (+≈ 22 GPU-h) | gains its toy support at the owner question (T5) |
| **NO-GO**, "used, buys nothing" | **CLOSED for growth-at-the-plateau** on the toy: depth (X47) and FFN width both NO-GO-S | no N4-P on toy evidence. **X48 still runs on its own case** (production-only, built, banked C0 at its pin) | loses its cheap support; NOT refuted (from-scratch width on 10⁴ teams is Procgen's regime, not a plateau fork's) |
| **NO-GO**, "not recruited" | weakly closed; the record names the recipe limit | as above | as above |
| **UNDECIDED** | open, unlicensed | no N4-P | the interval goes to the owner at T5; a look 2 is a NEW registration (≈ +20 GPU-h), never an amendment after seeing look 1 |
| **VOID** | — | — | fix the named input, re-run the seed |

**X48 as the production complement, whatever N4 reads.**
- X48 (per-token-kind FFN experts at P\*, `--grow trunk_ffn_by_kind=on`) is a capacity lever:
  - +0.79M parameters at zero extra FLOPs;
  - BUILT;
  - its control is the banked production C0 at its own pin (B-KFFN `3f69e6e2e` is an ancestor of `f4b25238`);
  - READY TO LAUNCH, ≈ 15 GPU-h (EXPERIMENT_BACKLOG BI-2).
- It reads production capacity directly, which the toy cannot settle (§6.1).
- **Ranking it is outside this unit (rule 9):** FINDING 4.

---

## 7. What the training session needs, in order

1. **T69 landed** on main (§2): the field, the kit arms, the descriptor readers and the read rule, each tested. Its
   commit is `N4_PIN`, recorded in the PREREG by a NOTE (not an amendment: the pin is a registered unknown, §3.2).
2. **The dry runs at `N4_PIN`** (`launch_x35.sh toy_c0n4 3641` / `toy_n4ffn 3641`, no GO). Each transcript reads:
   - "FORK of …/rb_toy_ent_s4002_deep";
   - "+70,000,000 steps";
   - "pin : `N4_PIN`";
   - N4 only: "grow : DECLARED trunk_ffn_dim 256 -> 1024".
3. **A GPU smoke** under the lease, ≤ 15 min: one N4 fork to its first 1M checkpoint in a scratch models dir. It
   checks the fork-time gate's line, the pool growth and the measured ρ, which replaces §2's GUESS in the PREREG
   NOTE.
4. **The chain file and the GO**, the orchestrator's. Six arms in §3.1's order.
5. **The reads** after the six finals settle:
   - `main.reads` with the spec PREREG §7 fixes, pinned to `N4_PIN`: S lane and anchor lane, `--concurrency 1`;
   - then the battery and the descriptors;
   - then `record`.

---

## 8. FINDINGS (standing rule 7)

1. **The cost is ≈ 2× the strategy doc's 13–15 GPU-h.** That figure assumed a reusable toy control. The toy pin
   `4609c155` predates the field-generic growth table (B-KFFN `3f69e6e2e`), so no clean own-pin cherry-pick
   exists, and N4 needs its own C0′ (§3.2). C0′ is reusable by later toy arms at `N4_PIN`.
2. **The strategy doc's sketch (fresh 2× `D_MODEL`, matched wall clock) is replaced, with a reason.** The toy's
   record reverses between equal compute and the plateau (§1.1). The strategy doc's §2 / §3.2 rows now point here.
3. **The production bridge has the same pin problem.** Production's C0 is at `f4b25238` (config v161). HEAD is
   v162, and T69 adds a version, so N4-P needs either a backported own pin with a declared-diff check (a new
   precedent: patch-identity cannot hold) or a new production C0 (≈ +22 GPU-h). It is the orchestrator's choice, on
   a GO only.
4. **X48 is a ready production capacity read, unranked against N4** (out of scope, rule 9). It needs no build and no
   new control, at ≈ 15 GPU-h.
5. **ρ for N4 is a GUESS (≈ 1.2)**, extrapolated from X47's trunk MACs; T69's CPU smoke read 1.23× pooled (pairs 1.05× and
   1.36×, a contended box: a hint only, §2). The GPU smoke (§7 item 3) measures it, and
   the FP32_ALT solve at `N4_PIN` may also make C0′ faster than the banked toy C0's 4,607 steps/s (UNVERIFIED).
6. **The battery baseline is one seed per row** (T\*, toy C0 s3601, X47 s3601 on one state set). It describes the
   frame and decides nothing.
7. **Grown-toy LOADING in `main.anchors` is VERIFIED; PLAYING is not.** A grown toy (T69's CPU smoke) loads through
   the bare loader (`load_checkpoint_strict`) and the foreign loader at width 1024 with the same logits, and
   `arch_status` reads it CURRENT (2026-10-27). No grown toy has played a game yet: the 2-pair `main.h2h` +
   `main.anchors` smoke is still open (T69's remaining items).
8. **The K meter is weak at screen scale**: 6,000 games give GO-K 0.61–0.69 at +3 pp. It is registered as a
   large-effect detector only, and a NO-GO says so (§4.2).
9. **This unit launched nothing and used no GPU** (`CUDA_VISIBLE_DEVICES=""`).
10. **The battery's depth worker REFUSES a wide trunk** (T69, 2026-10-27). `pin_worker.trunk_rounds` raises
    `UnsupportedTrunk` on a round with a width branch, because its layer replica, FFN statistics, truncation and spectra
    read only the base FFN (the B-KFFN precedent). The §5 battery on the N4 arms needs that worker taught the branch
    first; it is not in T69's model side.
   - Its CPU work ran under `nice -n 19 taskset -c 5-7,13-15`, the battery under `mem_cap.sh --max-s 600`:
     `main.checkargs` × 3, `main.launcher --dry-run` × 1, `main.capacity` and `power.py`.
   - `models/` was only read. The battery's output went to `/tmp`.

---

## 9. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **N4 DESIGNED AND REGISTERED (DRAFT)** (N4 designer, orchestrator brief; CPU only) | A growth fork at T\*: `--grow trunk_ffn_dim=1024` (a zero-init FFN branch per round, bit-identical at the fork) vs a same-pin C0′, 3 + 3 seeds × 70M at a new pin `N4_PIN`, ≈ 30 GPU-h. A one-look SCREEN on matched-step finals: S at one-sided 95 % (GO LB > 0, NO-GO UB < +2.0 pp) and Kakuna F as a large-effect detector (GO LB > 0). The mechanism read (engagement + battery v3) is reported. The bridge: GO licenses N4-P at the PR rule; NO-GO transfers weakly; X48 is the production complement | the fresh 2× `D_MODEL` toy at matched wall clock (the T-ENT reversal); fresh to the plateau (≈ 110–160 GPU-h); KFFN on the toy (refused by `checkargs`); depth (X47 done); reusing the banked toy C0 (no clean pin); X35 R1 as the decider (+3 pp → GAIN 0.16–0.32); a look 2 (≈ +20 GPU-h, 0.4–0.5 likely under a null) | §1.2's dry runs at `90708bff`; `power.log`; UNDERSTANDING §4.9; `toy_x47_look1.json`; the plasticity check (`8bebdfe3`); the battery on T\*, toy C0 s3601 and X47 s3601 (`battery_baseline.log`) |
