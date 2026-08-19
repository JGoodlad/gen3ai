# RUNBOOK — gen-16, THE MECHANICS GENERATION

**Pre-registered 2026-08-19, BEFORE gen-16 launches.** Every rule below is fixed while the number
it governs does not yet exist.

Gen-16 turns the conditional-mechanics SUBSTRATE on in the BASE. Fresh weights, pinned to current
main (the v96/v97 `gen3_critic_route_wave_v1` wall forbids a warm start); fresh pools, fresh
sentinels.

**The change list** (all nine acknowledged through the launch-diff gate vs gen-15):
`--pair-outcome-cell` · `--pair-outcome-switch` · `--switch-branch-cell` (OA2) ·
`--conditional-threat-cell` (OA1) · status-economy (in-place, no flag) · `--pfsp-scale 2.0` +
`--pool-spread` · `--intent-label-bot-weight 0.25` · and the two reward flags DROPPED because the
v8 composition is now the DEFAULT (verified in both the parser and the `RewardConfig` dataclass:
`all_shaping_pbrs=True`, `draw_penalty=-35.0`).

**Deliberately NOT in:** `--pair-value-route` (PV owes the C4-style offline gate, ledger C6 — no
exceptions), `--td-aux-coef` (rung 2 undecided; it is training-coefficient class, so a passing
rung 2 can join mid-run or ride gen-17), any fingerprint/flywheel machinery, and the α-batch's
grad-mode ladder / coef probe / B-move decisions (those are probes, not base changes).

## 1. Primary gate — non-inferiority vs gen-15

Dense `snapshot_ladder/ladder.json` tail-4, matched snapshot COUNT, at run END, SE from the
**paired refit** (`c'Σc`) — never the naive diagonal, never `main.elo`'s sparse fit.

- **NON_INFERIOR** iff Δ ≥ −15.0 AND CI95-low > −40.0. **INFERIOR** iff the whole CI sits below −15.
- Tie-break: more games per pair on the frozen ladder (`--backfill` cannot do this and now says so).
  Size against the **variance decomposition**, not the game count.

**Non-inferiority is the right bar and this is why:** the substrate is zero-init, and the BASE's job
is teaching the cells to be TRUE. Teaching the policy to USE them is the exploiter gates' job,
afterward. A base generation that merely holds serve while the cells come alive has done its job;
demanding a ladder gain here would be demanding the wrong thing from the wrong instrument.

## 2. THE BAIT/LOOP HUNT — the reason this generation exists

Registration of record: [`bait_loop_hunt.md`](bait_loop_hunt.md). Read at **matched scope**
(`--opponent 'sentinel_*'`) and **matched battle count**. All four together; any one alone has a
cheap way to be satisfied.

| # | bar | gen-15 | passes if |
|---|---|---|---|
| **B1 (primary)** | within-battle re-click rate | 32.2% | **< 16%** |
| **B2** | loop-battle rate | 13.9% | **< 7%** |
| **B3 (the honest one)** | median chosen-prob on residual loop steps | 0.963 | **< 0.85** |
| **B4 (a GUARD, not a goal)** | β slot acc · α SWITCH on loop steps | 82.1% · 76.2% | **flat or up** |

B4 inverted is the failure that would otherwise read as success: if the whiff rates fall while β and
α also fall, the run **lost the belief** rather than fixing the policy. Neither a repetition tax nor
a hand-coded immunity mask is a permitted response to a red number here — gen-14 had a repetition
tax and looped anyway.

**Launch-window liveness check (~5M), and it gates interpretation of everything else:** the new
`cell/<name>_{weight,grad}_norm` TB metrics MUST come off zero. The substrate is zero-init, so a
cell whose weight/grad norms never leave zero was never in the graph — and every downstream reading
about it would be a reading about nothing. Check this FIRST; a dead arrival channel invalidates §2's
interpretation before it invalidates anything else.

**REPEAT the α/β injection probe on gen-16.** On gen-15 it settled the mechanism by intervention:
forcing α/β to certainty produced 0 argmax flips in 40 arm-decisions and a bit-exactly zero β arm,
while the same intervention moved P(explosion) by 41.4 points — the signal existed and the channel
was multiplied by `is_boom`. `switch_branch` (OA2) is exactly the missing channel. If the injection
probe still reads ~0 flips on gen-16, the channel did not arrive, whatever B1 does.

## 3. Stall watch — SHAPE, not just rate

gen-15 read 0.73% cap-length episodes vs gen-14's 0.22% (Fisher p = 0.104, **not significant**), but
the distribution changed shape: mean turns 69.9 → 46.7, i.e. **bimodal — faster typical games with a
slightly fatter cap tail**. Re-read both the rate AND the turn distribution. A confirmed stall
regression argues for `--stall-pbrs` in gen-17, never for re-adding bias terms.

## 4. Fresh §4 route/family baseline — NOT a comparison

Run `critic_route_audit` + `edge_ablation_audit` at ≥12,000 states. This is a **fresh baseline at
gen-16's architecture**, not a delta against gen-15: v96 deleted routes and the substrate adds
cells, so past the surviving families the comparison is apples-to-oranges. Expect **eff-driven
shifts in the event-seat rows** — gen-16 is the first generation training with the event window's
EFF columns live (they were DEAD through gen-15; fixed `f05764e`). **That is a feature of this run,
not noise**, and must not be read as a substrate effect.

## 5. α/β `_pool` readouts + the switch-coverage matrix

Baselines in the ledger's sweep section. Report both; they are the belief-side companions to B4.

## 6. What would make this generation a mistake

Stated now so it cannot be rationalised later: if gen-16 is **non-inferior on §1 but the cells are
live and B1/B2/B3 do not move**, then the substrate arrived and the policy still will not use it —
which sends the question to the exploiter gates (where elicitation is the named confound), NOT to
more base training. If instead the **cells never came off zero**, the generation tested nothing and
the correct response is to fix the arrival channel and re-run, not to reinterpret §1.
