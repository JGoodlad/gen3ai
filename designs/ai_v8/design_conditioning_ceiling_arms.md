# The N=20 exploiter ceiling — a 5-arm factorial over COUNT × DIVERSITY × CONDITIONING

**Status: COMPLETE (2026-07-26 → 07-28). All 5 arms settled.**

## TL;DR

The multi-team exploiter ceiling is a **TEAM-COUNT** problem, not a conditioning problem.

| effect | estimate | 95% CI | |
|---|---|---|---|
| **COUNT** (20 → 10 teams) | **+0.077** | [+0.046, +0.108] | **significant** |
| **CONDITIONING** (per-team LUT) | **+0.028** | [+0.001, +0.055] | *marginally* significant |
| DIVERSITY (random vs z-clustered teams) | −0.022 | [−0.053, +0.008] | n.s. |
| LUT INIT (zero vs random codes) | +0.004 | [−0.042, +0.050] | n.s. |

Count alone accounts for essentially the whole 0.076 gap. Conditioning is real but **2.8× smaller**.
And because zero-init and random-init codes produce the *same* result from *opposite* geometries,
FiLM's small gain comes from a **shared modulation, not per-team specialisation** — so higher-rank
conditioning (LoRA/MoE) would not help either. **Stop raising N; run N≤10 exploiters and distil.**

## The question

The count sweep distils cleanly at N=1 (0.84), N=3 (0.835), N=10 (0.825), then **stalls at N=20**.
Ledger **D4**. The standing explanation was **conditioning-signal starvation**: FiLM is LINEAR in z,
so its per-team conditioning rank is capped by the rank of the z cloud; the DeepSets z is
*compositional*, so z-similar teams sit at `z̄ + ε_i` with tiny ε and the generator's gradient
`∂L/∂J ∝ δ ⊗ ε` is proportional to that residual (`designs/learning/conditioning_architectures.md`
§5b). The rival explanation was a plain capacity/interference ceiling.

Both make predictions about interventions that raise conditioning signal. Neither survived.

## Design

Every arm forks from the SAME checkpoint (`ai_v8_04`, ~277.2M), evals every 2M steps against the SAME
frozen target at n=200/cycle, and is **byte-identical to the `ai_v8_12` baseline command except the
named field** (diff-verified at launch). So each arm moves exactly one variable.

| arm | run | teams | LUT | isolates |
|---|---|---|---|---|
| baseline | `ai_v8_12` | def-20 (clustered) | ❌ | reference |
| ref | `ai_v8_13` | def-10 (clustered) | ❌ | the N=10 reference |
| 1 | `ai_v8_16` | def-20 | ✅ random-init | the LUT effect, teams fixed |
| 2 | `ai_v8_17` | random-20 | ❌ | the diversity effect, LUT fixed |
| 3 | `ai_v8_18` | random-20 | ✅ random-init | LUT within one team set (no team confound) |
| 4 | `ai_v8_20` | **random-10** | ❌ | **COUNT at matched diversity** (nested subset of arm 2) |
| 5 | `ai_v8_19` | def-20 | ✅ **zero-init** | arm 1 without its fork handicap |

**The random team sets are win-rate matched** so difficulty is held fixed — the per-team win rate is
confounded by team STRENGTH, and an easier random set would "win" for reasons unrelated to the
hypothesis. random-20 mean WR **0.548** vs the cluster's **0.547**; z participation ratio **9.59 vs
6.30** (×1.52 the code rank). random-10 is a **nested subset** of random-20 (mean WR 0.552), which
makes arm 4 vs arm 2 a pure count comparison with zero team-identity confound.

## Results

Plateau win-rate vs the frozen target, ≥4 pooled post-climb cycles each:

| run | arm | plateau | 95% CI | n |
|---|---|---|---|---|
| `ai_v8_13` | clustered-10 | **0.7250** | ±0.036 | 600 |
| `ai_v8_20` | **random-10** | **0.7000** | ±0.032 | 800 |
| `ai_v8_19` | def-20 +LUT (zero-init) | 0.6763 | ±0.032 | 800 |
| `ai_v8_16` | def-20 +LUT (random-init) | 0.6725 | ±0.033 | 800 |
| `ai_v8_18` | random-20 +LUT | 0.6550 | ±0.033 | 800 |
| `ai_v8_12` | def-20 baseline | 0.6488 | ±0.023 | 1600 |
| `ai_v8_17` | random-20 | 0.6225 | ±0.034 | 800 |

### The count × diversity grid — no interaction

| | clustered | random | diversity effect |
|---|---|---|---|
| **N=10** | 0.7250 | 0.7000 | −0.025 n.s. |
| **N=20** | 0.6488 | 0.6225 | −0.026 n.s. |
| **count (20→10)** | **+0.076 SIG** | **+0.078 SIG** | |

The count benefit is nearly identical on clustered and random teams (+0.076 / +0.078); the diversity
penalty is nearly identical at N=10 and N=20 (−0.025 / −0.026). The two effects simply **add**.

### The mechanism (arm 5 — the most informative result)

Random-init codes are orthogonal by construction (`lut_code_dist` **1.0**, flat). Zero-init codes grow
from a common origin and *converge* in direction (`code_dist` 0.24 → **0.146**, while `code_norm` grew
0 → 0.32 — so they engaged, they just did not separate).

**Opposite geometry, same outcome: 0.6725 vs 0.6763, difference +0.004 (n.s.).**

⇒ FiLM extracts its ~+0.028 from a **SHARED modulation**, not per-team specialisation. This explains
why every attempt to improve the *per-team* signal did nothing — the mechanism was never using that
part. It also reproduces the "lazy mode" of `project_sampling_snr_analysis` (two-thirds of z's energy
in one shared direction) from a clean zero start, where nothing forced it.

## Honest limits and corrections

**1. A mid-experiment correction on significance.** With the first two LUT estimates the conditioning
effect read **not significant**; the fifth arm moved it to **marginally significant**. The honest claim
is *"conditioning is a real but small lever"*, **not** *"conditioning does nothing"*.

**2. The first pooling was statistically invalid.** Two of the three LUT estimates share the `clust20`
baseline, so naive inverse-variance pooling double-counted that comparator and overstated precision
(gave [+0.003, +0.052]). Merging the two def-20 LUT arms into ONE condition — justified, since their
difference is +0.004 n.s. — yields two genuinely independent comparisons and the honest interval
**[+0.001, +0.055]**. It clears zero only barely; treat it as *small and probably real*, not established.

**3. Effect SIZES are under-resolved by design.** The gate's ±0.03 null band needs ~5,000 games
(≈25 cycles, ≈50M steps) to emit a true DECISIVE NULL; an arm caps near 2,400. `GAP NOT CLOSED` means
*the arm's question is answered, the effect size is not*. Four arms returned it.

**4. Arm 1 carried a self-inflicted handicap.** `--zarch-lut add` with random-init codes is NOT
identity-at-init — it perturbs an already-trained FiLM head, and arm 1 opened −0.040 below baseline and
spent ~6M steps recovering. That biased its measured effect DOWNWARD. Arm 5 removed it (and found the
initialisation does not matter at all).

**5. Nothing here measures whether the diversity cost COMPOUNDS** across successive distilled batches.
That is the assumption the "many N≤10 exploiters" plan now rests on.

## Implications

1. **The N=20 ceiling is a COUNT problem.** Do NOT build LoRA/MoE on the conditioning theory — and note
   the mechanism finding says higher-rank conditioning would not help anyway, because the benefit is not
   coming from the per-team direction.
2. **N=10 generalizes off the clustered set** (0.700 random vs 0.725 clustered, n.s.). This was the
   untested assumption behind "two N=10 exploiters cover 20 teams" — every prior N=10 datapoint came
   from a *clustered* set. The plan HOLDS, now with evidence.
3. **Team diversity is a small consistent cost** (~−0.025) — credible from its consistency across two
   independent comparisons, but never individually significant. Not a reason to cluster; not free.
4. **Practical:** stop trying to raise N. Run **N≤10** exploiters and distil — proven end-to-end
   (`project_multiteam_distill_payoff`) and the skill STICKS without teachers
   (`project_distill_retention_ablation`, ~76% retained at equilibrium).
5. **Open:** where the count cliff actually sits (N=5 / N=3 unmeasured on random sets).

## Method notes (what made this readable)

- **Verdicts are COMPUTED, never eyeballed** (`tmp/lut_verdict.py`): ≥4 pooled post-climb cycles, an
  explicit CI, and one of DECISIVE POSITIVE / PARTIAL / DECISIVE NULL / GAP NOT CLOSED / REGRESSION.
  Single cycles swung by up to **0.135** between adjacent evals; **three separate times** an early
  "effect" evaporated on the next cycle. Any eyeballed read would have produced a false positive.
- **One variable per arm, diff-verified against the baseline command at launch.**
- **Difficulty matched before comparing team sets**, because per-team win rate tracks team strength.
- **A GIGO canary per mechanism** — `zarch/lut_hit_frac` had to sit at 1.0 (a missed signature lookup
  silently routes to the unknown row and makes the whole arm a no-op), and `zarch/lut_code_norm` had to
  grow off zero on arm 5 (otherwise a null would mean "never engaged", not "did not help").

## Reproduce

```
python tmp/lut_verdict.py models/ai_v8_1{2,3,6,7,8,9}_* models/ai_v8_20_*   # per-cycle + verdicts
tmp/pick_random20_matched.py    # the win-rate-matched random set
tmp/z_spread_compare.py         # z geometry (participation ratio, pairwise cos-distance)
tmp/experiment_supervisor.sh    # the unattended arm sequencer; log tmp/supervisor.log
```

Raw data: `models/<run>/eval_results.jsonl` (one row per eval cycle, exact `counts`).

## See also

- `designs/research_state/ledger.md` → **D4**
- `src/agents/model/CLAUDE.md` → Per-team LUT (v46, `gen3_zarch_lut_v1`)
- Memory: `project_count_dominates_conditioning`, `project_lut_conditioning_ceiling_result`,
  `project_code_rank_ceiling` (superseded at N=20), `project_sampling_snr_analysis`

## Appendix — infrastructure bugs this surfaced

1. **Multi-team eval manifest crash.** `write_eval_manifest` called `.encode()` on `trainee_team_str`,
   which is a LIST for a `pin_multi` trainee. Eval fires MID-ROLLOUT, so it took the whole run down
   into a restart loop, and it only fires on the FIRST eval cycle (~70 min in) — invisible to smokes.
   Would have hit every multi-team arm, LUT or not. Fixed `dd01bae`.
2. **The supervisor died at both early handoffs — deterministically, not flakily.** `grep -c` PRINTS
   the count AND exits 1 on no matches, so `$(grep -c … || echo 0)` captured `"0\n0"` and the following
   `$((tries+1))` was an arithmetic syntax error that killed the launch. Signature: `TRY <arm>` written
   to the state file with `STARTING` never logged. After the fix it completed three handoffs unattended.
3. **A monitor that cried wolf.** The arm list was hardcoded; when the queue was reordered it reported
   `NO ARM RUNNING` while an arm ran fine. Now derived from the supervisor's own `ARMS` array.
4. **LUT fork plumbing.** Attaching the LUT to a forked checkpoint needed `policy_kwargs` repointed
   (else every checkpoint written was unreloadable — caught by the startup roundtrip smoke) and the new
   params APPENDED to the existing optimizer group (a second group cannot be reproduced by a fresh
   build, breaking restarts).
