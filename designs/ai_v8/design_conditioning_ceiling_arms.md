# The N=20 conditioning-ceiling experiment — a 2×2 over LUT × team-diversity

**Status: ARMS 1+2 ANSWERED (2026-07-27) — NEITHER the LUT nor team-diversity closed the gap.
Arm 3 (within-team-set LUT isolation) running; arm 4 (zero-init) queued.**

## The question

The multi-team exploiter count sweep distils cleanly at N=1 (0.84), N=3 (0.835) and N=10 (0.825),
then **stalls at N=20 (0.653)**. Ledger **D4**. Two rival explanations, both consistent with
everything measured so far:

- **Conditioning-SIGNAL starvation.** FiLM is LINEAR in z, so its per-team conditioning rank is
  capped by the RANK of the z cloud. The DeepSets z is *compositional*, so z-similar teams sit at
  `z̄ + ε_i` with tiny ε, and the generator's gradient `∂L/∂J ∝ δ ⊗ ε` is proportional to that tiny
  residual — ill-conditioned. Under this story the machinery is fine and the *codes* are the problem.
- **A real capacity/interference ceiling.** 20 conflicting strategies simply don't fit one shared
  head, however they're addressed.

They make opposite predictions, so the experiment is worth running.

## The design

Every arm forks from the SAME checkpoint (`ai_v8_04`, ~277.2M), evals every 2M steps against the
SAME target with n=200, and is otherwise **byte-identical** to the `ai_v8_12` baseline command
(diff-verified: only the named field changes). So each arm moves exactly one variable.

| arm | run | teams | LUT | isolates |
|---|---|---|---|---|
| baseline | `ai_v8_12_defensive20_exploiter_0724` | def-20 (z-clustered) | ❌ | — (plateau **0.653**) |
| 1 | `ai_v8_16_def20_lut_0726` | def-20 | ✅ | **the LUT effect** (teams held fixed) |
| 2 | `ai_v8_17_rand20_nolut_0726` | random-20 | ❌ | **the diversity effect** (LUT held off) |
| 3 | `ai_v8_18_rand20_lut_0726` | random-20 | ✅ | **do they compose?** |

**Why arm 2 exists.** Arm 1 alone cannot separate *spacing* from *freedom*: a LUT both spreads the
codes AND removes the constraint that a code be a function of the roster. Arm 2 widens spacing while
keeping composition intact. If arm 2 alone clears the stall, spacing was the whole story.

### The random-20 set (the diversity control)

Difficulty must be held fixed — the per-team win rate is confounded by team STRENGTH, so an easier
random set would "win" for reasons unrelated to conditioning. Selected by
`tmp/pick_random20_matched.py` from the 676 measured pool teams (`--team-pfsp measure` data, no new
battles), sampled inside a ±0.05 win-rate band around the cluster's own mean, excluding cluster
members:

| | def-20 | random-20 |
|---|---|---|
| mean win rate | 0.547 | **0.548** (Δ+0.001) |
| z participation ratio | 6.30 / 32 dims | **9.59** (×1.52) |
| mean pairwise cos-distance | 0.281 | **0.385** (×1.37) |
| archetypes | stall 6, semi-stall 5 (11/16 defensive) | balance 10, offense 4, hyper-off 3, stall 2, semi 1 |

Same difficulty, ~1.5× the code rank. Geometry measured by `tmp/z_spread_compare.py`.

## The decision gate (`tmp/lut_verdict.py`)

Reference trajectories, measured from `eval_results.jsonl`:

```
ai_v8_12  def-20 no LUT   .495 .57 .60 .69 .595 .635 .675 .65 .68 .66 .64 .655   -> plateau 0.653
ai_v8_13  def-10 no LUT   .465 .605 .685 .705 .75 .705 .72                        -> plateau 0.72
```

The gate pools **≥4 post-climb cycles** (≥800 games → 95% CI ≈ ±0.033) and returns:

| verdict | condition | meaning |
|---|---|---|
| **DECISIVE POSITIVE** | CI lower ≥ 0.72 | the count gap is CLOSED — conditioning signal was the limiter |
| **PARTIAL** | CI lower > 0.683 | signal is part of the story, not all of it |
| **DECISIVE NULL** | CI inside 0.653 ± 0.03 | a FREE code moved nothing ⇒ **the ceiling is not conditioning signal**. Do NOT climb to LoRA/MoE on this theory |
| **REGRESSION** | CI upper < 0.623 | the added conditioning is hurting |

**Why a pooled mean, not the best cycle.** At n=200 a single cycle carries ±0.069, and the baseline
itself printed a 0.69 cycle while plateauing at 0.653. Reading a lucky cycle as a win is precisely
how this experiment would fool us.

## Unattended operation

`tmp/experiment_supervisor.sh` (started detached; log `tmp/supervisor.log`, state
`tmp/supervisor_state.txt`) runs the arms **one at a time** — the box is CPU-saturated, so two
concurrent arms would just halve each other — and advances when the gate settles OR the arm passes a
25M-step cap (the baseline plateaued by +7M and ran to +22M; a cap stops a non-converging arm from
eating the window). An arm stopped on the cap is recorded as CAPPED, **not** as a verdict.

Safety: one arm at a time; never re-runs a finished arm (idempotent across supervisor restarts);
SIGTERM so the child saves a checkpoint; a **3-launch attempt cap** per arm so a persistent bug
parks the arm instead of looping forever; a hard deadline after which no NEW arm starts.

## RESULT — arm 2 (random-20, no LUT), 2026-07-27

The diversity control: same difficulty (win rate 0.548 vs the cluster's 0.547), **×1.52 the z code
rank** (participation ratio 9.59 vs 6.30), LUT **off**.

Per-cycle: `+0M .555 · +2M .600 · +4M .580 · +6M .545 · +8M .655 · +10M .560 · +12M .650 · +14M .625`

| | plateau | 95% CI | n | vs the 0.7250 ceiling |
|---|---|---|---|---|
| baseline def-20, no LUT | 0.6488 | ±0.0234 | 1600 | −0.076 |
| arm 1 def-20 + LUT | 0.6725 | ±0.0325 | 800 | −0.053 |
| **arm 2 random-20, no LUT** | **0.6225** | ±0.0336 | 800 | **−0.103** |

**Diversity effect = −0.026, 95% CI [−0.067, +0.015]** — not significant, and pointing the WRONG
way. Verdict **GAP NOT CLOSED**.

### What arms 1+2 together say

Two INDEPENDENT routes to "more conditioning signal" — a free unconstrained per-team code, and
genuinely well-spread compositional codes — and **neither moved the ceiling**. That is the core
prediction of the ill-conditioning story, twice falsified.

⇒ **The conditioning-signal theory is unsupported. Do NOT climb to LoRA/MoE on it.**

### Three limits that must travel with this claim

1. **Neither effect is SIGNIFICANT.** The honest statement is *"no route to more conditioning signal
   produced a detectable gain"* — **not** *"conditioning provably does nothing."* The first says stop
   investing here; the second is stronger than the data supports.
2. **Both residuals are UNRESOLVED.** The ±0.03 null band needs ~5,000 games (≈25 cycles, ≈50M steps)
   to emit a true DECISIVE NULL; an arm caps at ~2,400. `GAP NOT CLOSED` means *the arm's question is
   answered, the effect SIZE is not*.
3. **Arm 2 vs baseline is CROSS-team-set.** The match was on the generalist's self-play win rate =
   team STRENGTH, which is not the same as how exploitable the frozen target is when piloting those
   teams. So the −0.026 may be "these teams are harder to exploit with", not "diversity hurts".
   **Arm 3 is what removes this** — random-20 +LUT vs arm 2 is a WITHIN-team-set comparison.

### If arm 3 also lands ~0.62–0.65

Then the ceiling is not a conditioning problem in any form reachable by this architecture, and the
next hypothesis has to come from elsewhere: capacity, gradient interference between conflicting
per-team strategies, or simply that 20 strategies exceed what one shared head can hold. Write it up
as a kill rather than keep poking the same theory. Note the programme is NOT blocked either way —
see the N=10 implication above.

## Honest caveat: `--zarch-lut add` is NOT identity-at-init

Most toggles here are byte-identical when switched on. This one is not, and the interpretation
depends on knowing that. The per-team codes are deliberately **random**-init (large and ~orthogonal
from step 0 is the entire intervention), and the checkpoint's FiLM generators are already TRAINED —
so a changed `z` perturbs the head output immediately. Only an UNMATCHED team is unperturbed, via
the zero-init row 0.

Consequence: **a LUT arm starts with a small self-inflicted handicap it must first recover.**
Observed at the fork point: arm 1 opened at 0.455 vs the def-20 baseline's 0.495 (inside the ±0.069
single-cycle noise, but in the direction the mechanism predicts). Early cycles at or slightly below
baseline are EXPECTED and are not evidence the LUT failed. This is why the gate reads a pooled
**plateau** and ignores the climb — but do not let the plateau rule hide a genuine early collapse:
a REGRESSION verdict is a real branch of the gate.

**CORRECTION (2026-07-27).** An earlier version of this note claimed a zero-init LUT "would
reproduce the exact ill-conditioned geometry". **That was wrong.** At init, yes, `z = LN(z_deepsets)`
— the same compositional geometry. But the codes are FREE per-team parameters: `∂L/∂code_i` is the
full `∂L/∂z` restricted to team `i`'s samples, *not* something scaled by a tiny compositional
residual, and it flows from step 1 because the forked checkpoint's FiLM generators are already
trained. So zero-init does **not** inherit the ill-conditioning — it merely *starts* neutral.

That makes zero-init the better-controlled arm, and `--zarch-lut-init-std 0` now exists for it:
identity at init (pinned by `zarch_lut_test.test_zero_init_lut_leaves_the_forward_unperturbed`, close
up to the `zarch_lut_norm` LayerNorm eps), so the arm starts at parity and any divergence is purely
learned conditioning. Arm 1's −0.040 fork handicap biased its measured effect **downward**, so a
zero-init rerun of the arm-1 setup is the cleaner read of the same question. The init scale is
TRAINING-only and deliberately NOT version-gated: module shapes are identical and a resume loads
saved weights, so it only ever matters at the initial fork.

Watch `zarch/lut_code_norm` (new) alongside `zarch/lut_code_dist` on a zero-init arm — normalizing
all-zero rows would otherwise print `code_dist = 1.0` (maximum spread) for codes with no spread at
all, which is exactly the moment we would be watching them grow.

## RESULT — arm 1 (def-20 + LUT), 2026-07-27

| | plateau WR | 95% CI | n |
|---|---|---|---|
| baseline def-20, no LUT (`ai_v8_12`) | 0.6488 | ±0.0234 | 1600 |
| **arm 1 def-20 + LUT (`ai_v8_16`)** | **0.6725** | ±0.0325 | 800 |
| N=10 ceiling (`ai_v8_13`) | 0.7250 | ±0.0357 | 600 |

Per-cycle: `+0M 0.455 · +2M 0.550 · +4M 0.655 · +6M 0.610 · +8M 0.655 · +10M 0.680 · +12M 0.675 · +14M 0.680`
(the first four are the climb + the identity-at-init handicap; the gate pools from +7M).

**LUT effect = +0.024, 95% CI [−0.016, +0.064] — NOT distinguishable from zero.** It recovers ~31%
of the 0.076 count gap, but the CI on that fraction spans −21% to +84%. **DECISIVE POSITIVE is ruled
out**: arm 1's CI upper (0.705) is below the 0.72 target.

### What this kills

A free, unconstrained per-team code is the **maximum conditioning signal this architecture can
receive** — large, ~orthogonal codes from step 0 (`lut_code_dist` 1.0), every decision correctly
addressed (`lut_hit_frac` 1.0, `lut_teams_seen` 20/20). The ill-conditioning story predicted that
would clear the stall. It did not.

⇒ **Do NOT climb to LoRA / MoE / higher-rank conditioning on this theory.** Those are more expensive
ways to deliver the same signal that just failed to help. The `project_code_rank_ceiling` fix-order
("stop clustering → per-team LUT → covariance term → search teacher") should be re-read: rung 2 is
now spent, and rungs 3-4 inherit the same premise.

### Honest limits of this result

- **The gate could not resolve the residual.** The ±0.03 null band needed ~5,000 games (≈25 cycles,
  ≈50M steps) to emit DECISIVE NULL, double the arm's cap — so the verdict token said INCONCLUSIVE
  and the call was made on the *ruled-out* branch (CI upper < 0.72), which the design does support.
  The +0.024 is genuinely unresolved: it is NOT established as zero, only as "too small to close the
  gap". A follow-up wanting that distinction must budget ~25 cycles, not 4.
- **One team set.** Arm 1 tested the LUT on the z-CLUSTERED def-20. Arm 3 (random-20 + LUT) asks
  whether a free code helps when the codes are already well-spread — a different regime.
- The arm ran 14M of its 25M cap; stopped early because the decisive branch had already resolved and
  arm 2 was the better use of the box.

### The implication that actually matters for the programme

**You do not need N=20 to work.** N=10 distils cleanly (0.825) and the retention ablation (ledger
**D2**) showed the skill STICKS without teachers (~76% retained at equilibrium). So two N=10
exploiters cover the same 20 teams with a mechanism that is already proven end-to-end. The N=20
question is about **efficiency (fewer exploiter runs), not capability** — which lowers its priority
now that the cheap fix has failed.

## Known trap (cost 4 crashes)

`write_eval_manifest` called `.encode()` on `trainee_team_str`, which is a **list** for a `pin_multi`
trainee. Eval fires MID-ROLLOUT, so the `AttributeError` took the whole run down into a restart
loop. It was a regression from the multi-team eval work (`eval_worker` was taught to accept a list;
this consumer was not) and it fires only on the **first eval cycle** (~70 min in) — invisible to
smoke tests. Fixed in `dd01bae` with a regression test. It would have hit every multi-team arm
identically, LUT or not.

## See also

- `designs/research_state/ledger.md` → D4 (this is the experiment that resolves it)
- `designs/learning/conditioning_architectures.md` §5b — the FiLM/SNR diagnosis
- `src/agents/model/CLAUDE.md` → Per-team LUT (v46) — the mechanism + the `zarch/lut_hit_frac` canary
- Memory: `project_code_rank_ceiling`, `project_sampling_snr_analysis`
