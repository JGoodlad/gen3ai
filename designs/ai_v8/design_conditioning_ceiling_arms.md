# The N=20 conditioning-ceiling experiment — a 2×2 over LUT × team-diversity

**Status: RUNNING (arm 1 live 2026-07-26). Supervised unattended by `tmp/experiment_supervisor.sh`.**

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
