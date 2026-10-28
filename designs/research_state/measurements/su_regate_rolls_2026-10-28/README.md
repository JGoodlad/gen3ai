# S-U gate, leak-reduced re-gate (`dice="rolls"`) — READ: **GO** (2026-10-28)

Registered in [`PREREG.md`](PREREG.md) (`3626ef6e5`, before any state was scored). The read is
[`scripts/regate_read.py`](scripts/regate_read.py) and the record is [`regate_result.json`](regate_result.json).

**The question.** The S-U gate ([`../su_gate_2026-10-27/`](../su_gate_2026-10-27/README.md)) read GO at 0.635
[0.578, 0.687]. Its review ([`../su_headroom_2026-10-27/REVIEW_gate.md`](../su_headroom_2026-10-27/REVIEW_gate.md)
M1) found that GO carried by the states exposed to the true-seed leak. This re-gate asks: with the damage-roll truth
filter removed, do S's deviations still beat chance on U?

**The setup.** S′ = `r0s4_rolls@10` is S (`r0s4@10`) with ONE lever changed: `dice="rolls"`. Everything else is the
gate's:
- the same 1,560 frozen states;
- the same 24-seed self-referee;
- the same rule, `deviation_sign`;
- the same engine `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129` (the gate's, not S-U's registered `bdc5d2be`:
  G-1 carried);
- the same code under the read, at the gate's pin.

**This verdict SUPERSEDES the gate's leak-exposed GO for licensing V-RANK / S-U.**

**Verdict:** `GO: pooled-U sign accuracy 0.599 [0.543, 0.649] (n 362; GO iff LB > 0.5)`, not at the boundary.
- **Checks.** Every VOID check passed.
  - Speed-pin tooth: the marker equals the gate's on all 1,560 states.
  - Referee tooth: on all 398 same-pick deviations, S′'s referee result equals the gate's exactly.
- **Artefacts.** The state is `~/gen3ai_archive/su_regate_rolls_2026-10-28/`: `scored.jsonl` sha256 `6868773a…e453`,
  plus `pipeline.sh` and its logs.

## Result

| | home72 | H | F | P72 | pooled U |
|---|---|---|---|---|---|
| states (games) | 400 (200) | 360 (180) | 400 (200) | 400 (200) | 1,160 |
| excluded (gate's) | 20 (5.0 %) (39) | 2 (0.6 %) (5) | 20 (5.0 %) (43) | 3 (0.8 %) (6) | |
| deviation rate | 0.397 [0.349, 0.448] | 0.433 [0.379, 0.485] | 0.408 [0.357, 0.464] | 0.401 [0.350, 0.448] | |
| countable deviations | 127 | 129 | 115 | 118 | 362 |
| **sign accuracy** | **0.551 [0.462, 0.638]** | 0.620 [0.525, 0.711] | 0.583 [0.495, 0.667] | 0.593 [0.504, 0.680] | **0.599 [0.543, 0.649]** |
| gate's sign accuracy (S) | 0.537 | 0.640 | 0.660 | 0.605 | 0.635 |
| IMP per searched state | +0.010 [−0.002, +0.024] | +0.021 [+0.006, +0.037] | +0.010 [−0.000, +0.021] | +0.012 [−0.001, +0.025] | |
| search CPU-s a state (median / mean) | 4.5 / 5.3 | 3.9 / 4.2 | 4.7 / 6.1 | 4.0 / 4.6 | |

- **REPORTED:** U − home72 = +0.047 [−0.061, +0.143]. The gate's was +0.098.
- **Cost:** 2.2 CPU-h of search plus 2.0 CPU-h of referee, ≈ 4.3 CPU-h in all. Wall time was 5 h 04 m at nice 19 +
  SCHED_IDLE, behind the N1-P and X59-S reads' heavy slots.

**Strata (REPORTED, never decided; same estimator and seed as the review's):**

| pooled sign accuracy | U, S′ (this read) | U, S (the gate) | home72, S′ |
|---|---|---|---|
| Speed pinned | 0.602 [0.494, 0.712], n 73 | 0.711 [0.603, 0.815], n 69 | 0.533, n 15 |
| Speed not pinned | 0.598 [0.534, 0.656], n 289 | 0.618 [0.554, 0.677], n 277 | 0.554, n 112 |
| leak-exposed (a rewrite or a pinned Speed) | 0.632 [0.560, 0.702], n 180 | 0.716 [0.643, 0.782], n 172 | 0.571, n 49 |
| leak-clean | 0.564 [0.478, 0.639], n 182 | 0.554 [0.469, 0.629], n 174 | 0.538, n 78 |

The exposed share of searched states is unchanged (home72 0.47; U 0.49–0.55). It is a property of the true root.

**How S′ moved against S** (U, states searched under both):
- **Same deviation:** 312 of S's 444 deviations (0.70).
- **Changed:** 63 deviated to another action, 69 deviations were lost, and 79 gained.
- **Exclusions fell:** 43 → 20 on F and 39 → 20 on home72. On U, 43 states excluded under S were searched under S′;
  14 went the other way.
- **Paired IMP S′ − S:** 0.000 [−0.004, +0.005] on U and +0.001 [−0.009, +0.012] on home72.

## Reading

- **GO holds on the leak-reduced S, by a smaller margin.** Pooled U falls 0.635 → 0.599, and every U panel stays
  above 0.58. Under the gate's rule, the licence the gate gave stands, now on S′.
- **The damage-roll leak carried part of the exposed states' excess.**
  - Exposed U falls 0.716 → 0.632, and the Speed-pinned stratum falls 0.711 → 0.602, level with not-pinned (0.598).
  - Clean U is unchanged (0.554 → 0.564).
  - The exposed − clean gap on U shrinks from +0.16 to +0.07.
  - This fits the review's mechanism (the true-roll screen pinning hidden spreads toward the truth) without proving it:
    the strata are reported, post hoc in origin, and their intervals overlap.
- **The leak-clean stratum alone still has an LB below 0.50** (0.478, n 182). The pooled GO rests on both halves.
  - A deviation-only live read sized for the contrast (review M3 step 3) is where the clean half gets its power.
  - S′ still replays the prefix on the true seed: speed ties, crits and hidden durations, plus the template's rewrites
    and untyped Hidden Power. It is leak-REDUCED, not leak-free.
- **U − home72 halves** (+0.098 → +0.047, interval across 0). Nothing here attributes a U-specific gain to V.
- **The self-referee confound (review M2) is untouched.** T and S′ share a P\*-shaped opponent. A GO licenses the next
  stage only, never a gain.
- **The next step is the orchestrator's call.** The review's M3 orders a deviation-only live branching read against
  Kakuna (≈ 1,800 U deviations, ≈ 20–25 CPU-h, built on V-RANK's forced-branch builds B-V1–B-V4) before or instead of
  V-RANK's all-sibling slope. Any amendment that cites this re-gate registers S′ (`r0s4_rolls`) and this engine.
- **Predictions** (PREREG §8): pooled U ≈ 0.58 (0.52–0.64) HIT at 0.599; home72 ≈ 0.53 (0.551); exclusions fall HIT;
  ≈ 70 % of S's deviations kept HIT (0.70); P(GO) ≈ 0.55.

## FINDINGs

- **R-1 (a defect of the registered script, fixed after scoring, before the verdict was read).** The Speed-pin tooth
  compared the whole `speed_flip` block, including its `cpu_s` timing, and refused 1,539 rows. The marker's own fields
  (`ok`, `speed_mons`, `speed_pinned`) match on all 1,560. The comparison is narrowed to those fields. The `--control`
  run re-reproduces the gate's digits. The `--control` tooth could not catch this: the gate's rows equal themselves.
- **R-2.** S′ keeps the gate's G-1: its engine is not S-U's registered `bdc5d2be`.
- **R-3.** Tier A's likelihood weights are not read: S′'s worlds stay uniform. This keeps one lever. A
  likelihood-weighted S′ is unmeasured here.
