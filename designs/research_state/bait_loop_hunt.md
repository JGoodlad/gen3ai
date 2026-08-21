> ## ⚖️ VERDICT (2026-08-21) — **HABIT, not hedging-blindness.** Full data:
> `measurements/bait_programme_habit_verdict.json`
>
> Four instruments converge and **none of them finds an information deficit**: α/β know the switch
> (α top-1 **1.000** on loop steps) · injection to certainty is a 1,526×-amplified, **86%-of-α**
> channel that flips **ZERO** decisions · the immunity coordinate `e_mult_switch` is present but
> carries the 2nd-largest signal on the **8th**-largest weight · **the critic already ranks an
> alternative above the whiff in 21/23 loop decisions** (median +1.02 V, max +12.2).
>
> **Mechanism: exploration starvation at saturated actions.** The whiff sits at p≈0.97, so the
> alternatives at p≈0.01–0.03 are never sampled and their advantages are never realized — the 0.97
> is self-sealing. The levers are POLICY-side: a **deliberate-bait exploiter** (raises the cost) and
> **search-as-teacher** (raises the sampling). `repetition_tax` and a hand-coded immunity mask
> remain ruled out.
>
> **B3 is immobile across three generations — 0.985 / 0.970 / 0.972 against a <0.85 bar.** That is
> this programme's most stable number and its headline.
>
> **TD-aux is falsified as the bait lever, empirically** (registered λ>0 ⇒ B3 falls; observed B3
> rose monotonically on both cells). §7's insufficiency branch named it; it is now closed.
>
> ⚠️ **B-bars are OPPONENT-CONDITIONAL.** B1's in-window stratum reads 0.253 / 0.051 / 0.139 across
> gen-15/16/17: the eff fix owns the mechanism (out-of-window is flat in all three), the opponent
> population modulates the magnitude. Compare via arena games at matched opponents or state the
> confound. The matched read is NOT TAKEN (the arena emits no eval_traces) and is carried as an
> open caveat.

# The bait-loop hunt — PRE-REGISTERED for gen-16

**Status:** pre-registered 2026-08-19, before gen-16 launches. Edit only with new evidence, and say
what the evidence was. The bars below were written when nobody knew the answer; that is the only
thing that makes them worth anything.

**Instruments (all shipped, all model-free):**

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.prober.query loops <run_dir> --opponent 'sentinel_*'     # the detector
```
plus `cell/<name>_{weight,grad}_norm` in TensorBoard (the launch-window liveness check, §5).

---

## 1. The pathology

The opponent VOLUNTARILY pivots a mon our attack cannot touch; we fire anyway, and the exchange
repeats — one gen-15 battle ran **nine** cycles of Earthquake into a switched-in Salamence. Measured
over 843 gen-15 sentinel battles (`ledger.md` → *Sentinel sweep, gen-15*): 16.7% of moved-into
pivots whiff, 13.9% of battles carry a repeated pair, and **32% of whiffs re-click a pair the model
had already watched whiff IN THE SAME BATTLE**, at a median chosen-probability of **0.963**, with
another legal move available 91% of the time.

It is not a perception failure. On loop steps α called SWITCH 76.2% of the time and β named the
right slot 82.1% — both heads right, at the moment the wrong move is fired at p≈0.96. The causal
injection probe (`ledger.md` → *α/β injection probe, gen-15*) then settled the mechanism by
intervention rather than correlation: forcing α/β to certainty produced **0 argmax flips in 40
arm-decisions and a bit-exactly zero β arm**, while the same intervention on an Explosion decision
moved P(explosion) by 41.4 points. The op already holds "Earthquake KOs their Salamence with
p=0.000" and β already points at that slot — and **the one channel contracting the two is
multiplied by `is_boom`**. The number exists, the pointer is right, the product is ×0.

**So the gap is a missing CHANNEL, not a missing signal**, and gen-16's `switch_branch` (OA2) is
exactly that channel: E[our move | SWITCH] over β-weighted arrivals, un-gated by mechanic.

## 2. Why this class and not this metric

The owner's framing: *bait-and-switch is a core concept of the metagame*. Baiting an immunity,
pivoting on a locked Choice user, luring a Pursuit — these are the same skill, and a model that
cannot represent "what does my move do to the thing that is ARRIVING" cannot play any of them. The
goal is to kill the CLASS. The three rates below are a **thermometer for one instance of it**, and
the ways to move them without curing anything are known and cheap: penalise repetition (gen-14 had
a repetition tax and looped anyway — falsified), or hand-code an immunity mask into the action
space. **Neither is a permitted response to a red number here.** If the rates fall and nothing else
moves, that is a result about the rates, not about the model.

## 3. The gen-15 baselines

Carried in code as `main.prober.loops.LOOP_BASELINES`, so the CLI prints them beside every live
reading and neither surface keeps a copy. Measured 2026-08-19 on
`ai_v9_18_gen15_v8rewards_0818`, `sentinel_*`, all steps, 843 battles.

| quantity | gen-15 |
|---|---|
| moved-into pivots | 4923 over 843 battles |
| **whiff rate** (per moved-into pivot) | **16.7%** (820/4923) |
| whiff kinds | immune 769 · fail 41 · near-zero 10 |
| **loop-battle rate** (a pair whiffed ≥2×) | **13.9%** (117/843) |
| loop ≥3× | 6.2% (52/843) |
| **within-battle re-click rate** | **32.2%** (264/820) |
| median chosen-prob on loop steps | **0.963** |
| loop-step median ΔV / ΔP(win) | −4.31 / −0.096 |
| loop-battle rate by step | 5.0% @4M → 21.1% @20M → 15.2% @24M |
| β top-1 slot accuracy | 52.0% first-time → 65.9% repeat → **82.1% on loop steps** |
| α top-1 = SWITCH on loop steps | 76.2% (median P(SWITCH) 0.60) |
| mirror (THEY whiff into OUR pivots) | 14.5% |

The detector reproduces this run to within the two definitional tightenings it ships (`near_zero`
at ≤1% rather than ≤3%; a `-fail` carrying an external `[from]` cause is no longer our move
failing): 843/843 battles matched, 4923 moved-into pivots, 117 loop battles, 52 at ≥3×, 264
re-clicks, median loop-step chosen-prob 0.9625, ΔV −4.31, ΔP(win) −0.0958, β 52.0/65.9/82.1, α
76.3%, trend 5.0 → 21.1 → 15.2. Whiffs read 810 rather than 820 — the ten events the tightenings
remove.

## 4. The PRE-REGISTERED bars

Read at **matched scope** (`--opponent 'sentinel_*'`) and at **matched battle count**, on a gen-16
run at a comparable step. All three must be read together: any one of them alone has a cheap way to
be satisfied.

| # | bar | gen-15 | gen-16 passes if |
|---|---|---|---|
| **B1** | within-battle re-click rate | 32.2% | **< 16%** (halves) |
| **B2** | loop-battle rate | 13.9% | **< 7%** |
| **B3** | median chosen-prob on residual loop steps | 0.963 | **< 0.85** |
| **B4** | β slot accuracy on loop steps · α SWITCH on loop steps | 82.1% · 76.2% | **flat or up** |

**B1 is the primary.** A re-click is a decision taken with the answer already on the board — the
sharpest form of the pathology and the one least confounded by how often the opponent pivots.
**B3 is the honest one**: it is entirely possible for the behaviour to survive at lower confidence,
which is a real (if partial) result and must be reportable as one rather than rounded into B1.
**B4 is a GUARD, not a goal.** Perception was never the broken half; if the whiff rates fall while
β and α also fall, the run lost the belief rather than fixing the policy, and that reads as a
FAILURE of this hunt regardless of B1–B3.

## 5. The confounds, and how the detector conditions for them

Two are registered, both measured on gen-15, both printed as `caveats` on every result:

1. **Length.** Loop rate rises with game length — a 200-turn game has more chances to repeat a
   pair. So the per-battle rate (B2) never travels alone: `whiff_rate_per_pivot` and
   `whiff_rate_per_decision` normalize the exposure and are reported beside it. Read the per-pivot
   rate first; a B2 that moves while the per-pivot rate does not is a length result.
2. **Winning positions.** The loops concentrate in games we were WINNING (gen-15: 23.1% loop-battle
   rate in wins vs 7.0% in losses; 84.6% of the worst loopers are wins) — a won position is where a
   free turn is affordable. `by_outcome` is therefore always reported, and **the comparison is
   win-arm to win-arm**. An overall rate that moves with the win rate has moved for two reasons.

A third is structural and worth stating: `beta_slot_accuracy` is decidable only when the arriving
mon was already revealed, so its denominator skews toward repeat pivots by construction. The
`first_time` / `repeat` / `loop_step` split is reported precisely so that skew is visible rather
than averaged away. And the `mirror` block is a CONTROL, not a target — it measures the opponent's
policy, which in a sentinel matchup is a frozen self.

## 6. The launch-window liveness check

gen-16 turns on four zero-init pointer cells at once. Each contributes exactly zero to every action
logit at init — deliberately, so ON-at-init is byte-identical to OFF — which means a cell that never
learns is indistinguishable from one that works. **Without this check, "the behaviour did not
change" and "the cell never came off zero" are the same observation**, and the hunt cannot tell
them apart.

In TensorBoard, from the first `train()` call:

```
cell/switch_branch_weight_norm      cell/switch_branch_grad_norm
cell/pair_outcome_move_*            cell/pair_outcome_switch_*
cell/conditional_threat_*
```

Read the PAIR: both ~0 = dead · weight ~0 with grad > 0 = still climbing off the init (expected in
the first hours) · weight > 0 with grad ~0 = converged and contributing. **Check within the first
launch window.** `weight_norm` is a parameter magnitude, not an effect size — it answers *is this
alive*, never *is this important*; only an ablation answers the second.

## 7. The end-of-run protocol

Run BOTH; neither alone is sufficient.

**(a) The detector, gen-16 vs gen-15, at matched battle counts.**

```bash
python -m main.prober.query loops models/<gen16> --opponent 'sentinel_*' > gen16_loops.json
python -m main.prober.query loops models/ai_v9_18_gen15_v8rewards_0818 --opponent 'sentinel_*' > gen15_loops.json
```
Compare B1–B4, per outcome arm, on the `by_step` rows nearest matched steps. The gen-15 numbers are
also embedded in every gen-16 result's `baseline` block, so a single run is self-comparing for a
quick read; the explicit re-run is for matched *counts*, which the baseline cannot supply.

**(b) REPEAT the injection probe on gen-16.** This is the load-bearing half. Forcing β to the immune
arrival must now MOVE the policy — under gen-15 that arm was bit-exactly zero. The probe is the
only thing that distinguishes "the channel exists and carries gradient" from "the rates happened to
fall".

Three outcomes, with the fork pre-committed:

| outcome | signature | fork |
|---|---|---|
| **installed and learned** | cell weight norms off zero (§6) · injection MOVES the policy · B1–B3 pass · B4 flat-or-up | Done. Write it up, and re-run the detector on the exploiters that fork the gen-16 base — the base teaches the cell to be TRUE, the exploiter teaches the policy to USE it. |
| **installed, never learned** | cell weight norms ~0 with grad ~0 · injection still ≈zero · rates unchanged | The channel is present but carries no gradient. Levers: the cell's INPUTS (does it actually receive `out_pko` against the β-weighted arrival?), the delivery route (policy-side vs the critic-side twin that DID learn — `IntentThresholdValue` ‖W‖₁ 15.4 vs the move cell's 0.48), and supervision. **NOT** a bigger cell. |
| **learned but insufficient** | weights alive · injection moves the policy · rates fall short of B1–B3 | A credit-assignment problem, not a representation one: the channel delivers the fact and the policy still prefers the whiff. Levers: value-target / TD-aux on the loop steps (they already cost median ΔV −4.31, so the critic prices them — check whether that price reaches the policy), search-as-teacher on bait decisions, exploiter pressure from an opponent that baits deliberately. |

**Two responses are ruled out in advance, in every branch:**

- **NOT `repetition_tax`.** gen-14 shipped one and looped anyway — falsified, and re-deriving it
  from a red B1 would be re-running a dead experiment.
- **NEVER a hand-coded immune mask.** It would satisfy B1 and B2 outright while teaching the model
  nothing, and it does not generalize one inch beyond type immunity — not to Choice-locked pivots,
  not to Pursuit, not to any other member of the class this hunt exists to kill.

## 8. Where the instruments live

| piece | file |
|---|---|
| detection logic + definitions + `LOOP_BASELINES` (pure, no torch, no session) | `src/main/prober/loops.py` |
| unit tests incl. the calibration-battle shape | `src/main/prober/loops_test.py` |
| the run-level scan + aggregation + caveats | `ProbeSession.loops` (`src/main/prober/session.py`) |
| the CLI | `python -m main.prober.query loops` |
| cell liveness metrics | `agents.training.grad_balance.cell_family_metrics`, emitted by `instrumented_ppo` |

**Calibration:** on `step_22000032/sentinel_0/win_s0_001` the detector must find one loop —
`earthquake` → `salamence`, count 9, turns 3, 7, 11, 15, 19, 23, 27, 31, 35, all `immune` — plus
exactly two whiffs NOT in it (T40 `rapidspin` → `metagross`, near-zero; T44 `earthquake` →
`claydol`, immune), and 8 re-clicks. Pinned as a shape in `loops_test.py` (that trace lives under
gitignored `models/`, so the test cannot read it) and verified against the real trace 2026-08-19.
