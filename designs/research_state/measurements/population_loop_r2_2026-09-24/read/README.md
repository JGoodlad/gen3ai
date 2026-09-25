# Population loop round 2 — THE REGISTERED PRIMARY READ (2026-09-25) · VERDICT: NOT DETECTED → branch N+ (PROVISIONAL: the §4.3 KILL guards are not yet read)

The registration is [`../../../population_loop_round2_2026-09-24.md`](../../../population_loop_round2_2026-09-24.md)
(committed `3265ec83` before any B2 eval; §4 the read, §5 the branches, §6 the hazards). The launch kit is
[`../README.md`](../README.md). Method precedent: [round 1's read](../../population_loop_r1_2026-09-23/read/README.md).
Read at `origin/main` `0896b7d9` (the meter only reads run files), from a worktree (finding K-1), 10:52 PT 09-25.

**The codes, each time:** **B2** = `ai_v13_27_popr2_loop`, the round-2 LOOP generalist (B, the round-1
loop generalist, continued +8M against the stable set {A, `ai_v13_13`, RB} at share 0.40). **C2** =
`ai_v13_28_popr2_ctrl`, the round-2 NO-EXPLOITER CONTROL (C continued +8M, B2's argv at share 0.0).
**RB2** = `ai_v13_29_popr2_read_loop`, the fresh offense best responder ("reader") trained against B2;
**RC2** = `ai_v13_30_popr2_read_ctrl`, the one trained against C2. **A** = `ai_v13_18_teach5_offense_hidose`,
the round-0 offense exploiter of the plateau parent G0 (`ai_v13_12_plateau`), whose recipe every reader
copies. **RB** / **RC** = `ai_v13_24_popr1_read_loop` / `ai_v13_25_popr1_read_ctrl`, round 1's readers of
B / C. `ai_v13_13` = `ai_v13_13_exploit5_offense`, the second (0.39×) offense exploiter of G0.
**RB+** / **RC+** = `ai_v13_31_popr1_read_loop_ext` / `ai_v13_32_popr1_read_ctrl_ext`, RB / RC forked +50 %
(the §4.4 convergence side-check).

## Status

| piece | state |
|---|---|
| §5 V: `--check` gate (§4.2.1) | **0 mismatches**, 3 runs, budget 8,060,928 · dose 3.815e-08 · offense on all three (`brgap_check.txt`) |
| §5 V: STOP-list from files | **clean** — B2 TB `train/stable_fraction` 0.36 on all 4 post-fork points, C2 0.00; every B2 / C2 eval row `matchup_hash` `ef5242cffd`; RB2 and RC2 at 119,341,056 (`stoplist.json`, `stoplist_tb.txt`) |
| §5 V: `plateau_b1` reproduction (963/1600) | **NOT RUN** — it belongs to G-U |
| §4.1 manipulation check M2 | **ABSORBED**, M2 = +8.67 pp [+3.06, +14.19] (`manipulation_check.json`) |
| §4.2 PRIMARY Δ2 = gap(RB2) − gap(RC2) | **−8.25 pp [−15.01, −1.38]**, bar 5.0 → **NOT DETECTED** (`primary_verdict.json`) |
| §4.2.3 round-over-round (descriptor) | loop chain r3 − r2 **−4.50 [−11.33, +2.39]**; control chain r3 − r2 **−6.25 [−12.83, +0.41]** → the registered "loop falls and the control does not" pattern is **NOT met** (the control fell too) |
| §4.3 G-U (untaught 8) and G-A (SmallRL anchors) | **NOT READ** — no driver is running and none was asked of this read. Branch K comes BEFORE N+ in §5's order, so the branch below is PROVISIONAL until both are read |
| §4.4 convergence side-check | **PENDING** — RB+ is training (ends ≈ 12:35 PT), then RC+ (≈ 14:30). One-liner: `scripts/side_check.sh` (refuses until both have finished) |
| §5 branch | **N+ (provisional)** — Δ2 NOT DETECTED, ABSORBED. If G-U and G-A come back clean: counts **2 of 3** toward the stopping rule, and round 3 is the LAST round at this configuration |

## Deviations from the registration

| # | deviation | reason |
|---|---|---|
| D1 | §4.3 guards (G-U, G-A) not run, so branch K and V's reproduction clause are OPEN | out of this read's scope (primary only); CPU-hours of work on a box with a live training arm. The branch is labelled PROVISIONAL, not issued |
| D2 | §4.2 secondary (`--play 400 --greedy` on RB2 / RC2) not run | its registered use is only "run only if the primary is OUTSIDE" (§4.2) |
| D3 | read at `origin/main` `0896b7d9`, not the arms' pin | the registration's commands are main's meter (as round 1's read); it reads run files only and reproduced round 1's primary exactly at registration time |

## 1. §4.1 manipulation check — **ABSORBED**

`scripts/manipulation_check.py` (round 1's, with the third specialist added). Greedy-vs-greedy
`externals`, 100 games per specialist per cycle, `sentinel_regime {greedy: true, symmetric_teams: true}`.

| cycle step | B2 vs A | B2 vs `ai_v13_13` | B2 vs RB | C2 vs A | C2 vs `ai_v13_13` | C2 vs RB |
|---:|---:|---:|---:|---:|---:|---:|
| 104,000,016 | 42 | 44 | 52 | 35 | 34 | 37 |
| 106,000,032 | 41 | 38 | 53 | 32 | 37 | 34 |
| 108,000,000 | 44 | 46 | 48 | 42 | 42 | 39 |
| 110,000,016 | 42 | 45 | 59 | 37 | 38 | 34 |

**M2 = B2's last two cycles (284/600 = 0.473) − C2's (232/600 = 0.387) = +8.67 pp [+3.06, +14.19] →
ABSORBED.** Per specialist (n = 200 each, descriptors): vs A +3.5 [−6.1, +13.0]; vs `ai_v13_13` +5.5
[−4.2, +15.0]; **vs RB, the one B2 had not seen, +17.0 [+7.3, +26.3]**. So the loop engaged at depth 2
on the NEW specialist, not only on the old ones. B2 was already at 0.52 against RB at its first cycle
(RB beat B at 0.60 at its endpoint, i.e. B scored 0.40), and gained little further against A and
`ai_v13_13` (0.42–0.46, round 1's B ended at 0.41).

## 2. §4.2 PRIMARY — Δ2 = gap(RB2) − gap(RC2)

```
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_29_popr2_read_loop ai_v13_30_popr2_read_ctrl --check
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_30_popr2_read_ctrl ai_v13_29_popr2_read_loop \
    --rounds ai_v13_30_popr2_read_ctrl=2 ai_v13_29_popr2_read_loop=3 --json brgap_r2.json --md brgap_r2.md
python scripts/primary_rule.py .
```

| reader | target | pooled | gap pp | 95 % CI | per-cycle vs-target | cycles 3–4 − 1–2 |
|---|---|---:|---:|---|---|---:|
| A (round 0) | G0 | 256/400 | +14.00 | [+9.18, +18.55] | 0.65 / 0.58 / 0.67 / 0.66 | +5.0 |
| **RC2** | C2 (control) | 243/400 | **+10.75** | [+5.88, +15.41] | 0.64 / 0.60 / 0.66 / 0.53 | −2.5 |
| **RB2** | B2 (loop) | 210/400 | **+2.50** | [−2.39, +7.35] | 0.50 / 0.53 / 0.52 / 0.55 | +2.0 |

**Δ2 = −8.25 pp, Newcombe 95 % [−15.01, −1.38]** (the `round 3 − round 2` row; the tool's VERDICT line
reads "UNREADABLE — fewer than two archetypes", finding F1, and is not the verdict). **The rule, F = 2.00
reused, bar 5.0:** |Δ2| = 8.25 > 5.0 (clause 1 holds), but the CI's upper end −1.38 does NOT exclude −5.0
(clause 2 fails) → not OUTSIDE; the CI is not inside [−5.0, +5.0] → not EQUIVALENT. **VERDICT: NOT
DETECTED.** It sits where the registration's power statement said a real effect of this size would sit:
detection needed Δ2 ≲ −11.7.

**Convergence descriptor (H-1's read, §4.2):** neither reader rose ≥ 5 pp from cycles 1–2 to 3–4 (RB2
+2.0, RC2 −2.5), so **no convergence caveat attaches to Δ2**. Unlike round 1's RB (0.52 → 0.60), RB2 is
flat at ≈ 0.52.

**Descriptors, never folded:**
- gap(RC2) − gap(RC) = **−6.25 [−12.83, +0.41]** (prior: inside the bar — the point is outside it, the CI
  straddles it). gap(RB2) − gap(RB) = **−4.50 [−11.33, +2.39]**.
- Endpoint stat (the LAST cycle only, the banked convention; NOT the registered stat): RB2 0.55 − RC2
  0.53 = **+2.0 [−11.6, +15.5]**. The sign flips because RC2's last cycle (0.53) is its lowest.
- Late window (cycles 3–4 only): **−6.0 [−15.5, +3.7]**.
- Pooled over rounds 1 and 2 (RB + RB2 vs RC + RC2, 800 games each) — **includes data seen before
  registration, NOT a registered test** (§4.2.3): **−9.13 [−13.88, −4.31]**; its upper end would not clear
  −5.0 either.

## 3. §4.2.3 round over round (descriptor)

`brgap_loop_chain.{txt,json,md}` (A → RB → RB2) and `brgap_ctrl_chain.{txt,json,md}` (A → RC → RC2):

| chain | r2 − r1 | r3 − r2 |
|---|---|---|
| loop (G0 → B → B2) | −7.00 [−13.68, −0.23] (as registered) | **−4.50 [−11.33, +2.39]** |
| control (G0 → C → C2) | +3.00 [−3.58, +9.54] (as registered) | **−6.25 [−12.83, +0.41]** |

The registered pattern ("the loop chain's r3 − r2 below zero AND the control chain's not") is **not met**:
both chains fell in round 2, the control by more (point). The loop's gap has fallen monotonically,
+14.0 → +7.0 → +2.5; the control's went +14.0 → +17.0 → +10.75. The contrast that governs is §2's Δ2.

## 4. The §5 branch (read in order; the first that applies governs)

| branch | status |
|---|---|
| **V** | `--check` 0 mismatches; STOP-list clean. **The `plateau_b1` reproduction clause is OPEN** (G-U not run) |
| **K** | **OPEN** — G-U and G-A not read |
| **R** | impossible — Δ2 < 0 |
| **T** | impossible — the CI's upper end −1.38 does not clear −5.0 |
| **N+** | **FIRES, PROVISIONALLY** — NOT DETECTED at bar 5.0; ABSORBED (M2 lower bound +3.06) |
| M, E | ruled out (ABSORBED; the CI is not inside ±5.0) |

**If the guards come back clean**, the registered consequence of N+ applies: *B2 beat what it saw, and a
fresh best responder is not measurably weaker at this power, for the second round running.* It counts
**2 of 3** toward the stopping rule. Round 3 is one round deeper (RB2 joins B2's set, share 0.40, the same
dose) and is the LAST round at this configuration. **Before round 3 is registered, the orchestrator
decides whether to buy the power the question needs** (`--eval-battles 200`, or a pooled 3-round read
registered BEFORE round 3's numbers exist), because a third NOT DETECTED at n = 400 ends the loop on power
alone. **If G-U or G-A fires, the branch is K** and B2 is not a round-3 parent.

## 5. Findings from this read

- **F2 reproduced again, LIVE.** The unassisted `--check` listing put RB2 in round 2 and RC2 in round 3, the
  reverse of §4.2's mapping (`brgap_check.txt`). The registered `--rounds` fixed it.
- **Both readers exited 1 AFTER "Training complete"** (`crashes/restart_err_*.txt` in each run dir: a
  `train_env` worker died with exitcode −15 during teardown). The launcher restarted, printed "Training
  already complete (119,341,056 / 119,280,128 steps)" and ended. `final_model.zip` was written before the
  crash (06:46 / 10:36), every eval cycle is present, and the landing step is as registered. It cost nothing
  here, but it is a teardown defect.
- **The watcher logs "FAILURE: launcher pid … GONE"** at each normal exit (`watch_ai_v13_{29,30}_*.txt`) —
  a false alarm reading, for the Training Run session's watcher.
- **The launch-kit concurrency guard false-positived** (chain_status 03:00: "kit guard false-positive on a
  non-python shell"); the Training Run session overrode it with `ALLOW_CONCURRENT=1` after checking no real
  trainer was live. C2 also had one blocked start (rc 5, 22:22) and was relaunched at 22:29, so B2 and C2
  were 7 minutes from adjacent.
- **RC2's gap fell 6.25 pp below RC's** on a plain continuation block (the prior was "inside the bar"). A
  second 8M block at share 0.0 may itself lower exploitability, or this is reader noise (F = 2.00 is one
  pair at one target). This is why the within-round contrast, not the round-over-round chain, governs.
- **The JSON `_meta.cwd` / `teamsets` fields name this read's worktree path**, which is removed after the
  commit; the teamsets file is the committed `designs/research_state/exploiter_teamsets_2026-09-22.json`.

## Files

| file | what |
|---|---|
| `brgap_check.txt` | §4.2.1 gate |
| `brgap_r2.{txt,json,md}` · `primary_verdict.json` · `scripts/primary_rule.py` | §4.2 the registered read + the rule |
| `brgap_loop_chain.*` · `brgap_ctrl_chain.*` | §4.2.3 both chains |
| `manipulation_check.json` · `scripts/manipulation_check.py` | §4.1 |
| `stoplist.json` · `stoplist_tb.txt` · `scripts/stoplist_check.py` | §5 V STOP-list facts from files |
| `scripts/side_check.sh` | §4.4, the pending one-liner (run from a worktree root; refuses until RB+ and RC+ have finished; writes `brgap_ext_check.txt`, `brgap_ext.{txt,json,md}`, `convergence_rule.{txt,json}` here) |
