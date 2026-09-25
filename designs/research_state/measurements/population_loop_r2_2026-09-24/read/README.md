# Population loop round 2 — THE REGISTERED READ (2026-09-25) · VERDICT: NOT DETECTED → branch N+ (FINAL: both KILL guards read clean; reproduction check EXACT)

The registration is [`../../../population_loop_round2_2026-09-24.md`](../../../population_loop_round2_2026-09-24.md)
(committed `3265ec83` before any B2 eval; §4 the read, §5 the branches, §6 the hazards). The launch kit is
[`../README.md`](../README.md). Method precedent: [round 1's read](../../population_loop_r1_2026-09-23/read/README.md).
Primary read at `origin/main` `0896b7d9` (the meter only reads run files), from a worktree (finding K-1), 10:52 PT
09-25. KILL guards played 11:13–12:47 PT 09-25: G-U on the arms' pin `6eb9c776`, G-A on `7c511161` (§§4–5).

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
| §5 V: `plateau_b1` reproduction (963/1600) | **PASSED EXACTLY** — 963/1600 = 60.19 pp, and every per-team rate identical to round 1's rows (`gu_guard.json`) |
| §4.1 manipulation check M2 | **ABSORBED**, M2 = +8.67 pp [+3.06, +14.19] (`manipulation_check.json`) |
| §4.2 PRIMARY Δ2 = gap(RB2) − gap(RC2) | **−8.25 pp [−15.01, −1.38]**, bar 5.0 → **NOT DETECTED** (`primary_verdict.json`) |
| §4.2.3 round-over-round (descriptor) | loop chain r3 − r2 **−4.50 [−11.33, +2.39]**; control chain r3 − r2 **−6.25 [−12.83, +0.41]** → the registered "loop falls and the control does not" pattern is **NOT met** (the control fell too) |
| §4.3 G-U (untaught 8) | **does NOT fire** — B2 − C2 = −1.31 pp [−4.19, +1.19] vs the 3.69 floor; 4,800 battles, 0 timeouts (§4) |
| §4.3 G-A (SmallRL anchors, greedy vs greedy) | **does NOT fire** — B2 − C2 = +1.75 pp [−2.13, +5.62] vs the 11.0 floor; 24 units / 2,400 games, all OK, regime verified (§5) |
| §4.4 convergence side-check | **PENDING** — RB+ finished its training ≈ 12:35 PT, RC+ runs to ≈ 14:30. One-liner: `scripts/side_check.sh` (refuses until both have finished). It changes no branch and no count |
| §5 branch | **N+ — FINAL.** Δ2 NOT DETECTED, ABSORBED, guards clean, not VOID. Counts **2 of 3** toward the stopping rule. **Power decision RESOLVED by the owner (2026-09-25): NO round 3 on this lineage** — the loop carries into the NEW lineage with both power levers registered up front (§6) |

## Deviations from the registration

| # | deviation | reason |
|---|---|---|
| D1 | §4.3 guards run AFTER the primary was banked (ledger `aa8d56ea` carried the branch as PROVISIONAL) | the primary read was scoped first; the guards were then dispatched and ran as registered. The guard rule and floors were fixed before either was played |
| D4 | G-U at **3** workers (registration: 4) and **nice 19** (round 1: 15); G-A at `--nice 19` + outer `nice -n 19` (registration: `--nice 15`) | coordinator's load constraint (a live arm, the M6 stress and a gate on the box). Workers split units and niceness schedules CPU; neither changes a number at concurrency 1, and the reproduction check came back EXACT |
| D5 | G-A units at `--team-seed` = `--seed-base` ∈ {0,…,50} (100-game units) with `--server rust` named explicitly | round 1's D3, carried as the registration directs (§4.3: "the same S for B2 and C2 (round 1's D3)"); `--server rust` is the tree's default, named so the argv says it |
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

## 4. §4.3 G-U — the untaught 8. **Does NOT fire. Reproduction check PASSED EXACTLY.**

Tree `6eb9c776` (the arms' pin), its own worktree (`gen3ai-wt/pin-6eb9c776-popr2`, submodule `e0551883`,
`dist` / `node_modules` linked — H-9 did not bite), its own release `sim_bridge` (`strings` shows only that
worktree's dex path), cwd = that tree, CPU only, one thread, nice 19. Round 1's driver with the refs changed
(`scripts/gu_driver.py`, `scripts/gu_env.sh`): `untaught_meter_opponent`, `--seed 0`, concurrency 1, 200 games
per team, units of (ref × team × 25 battles), **4,800 battles, 0 timeouts**. The paired team bootstrap is
round 1's, verbatim (20,000 draws, seed 20260915). `data/` is byte-identical between `6eb9c776`, `7c511161`
and main. The aggregate was exercised first on round 1's banked rows relabelled (it reprints round 1's B − C
−2.44 [−6.31, +0.88] exactly), and the first `plateau_b1` unit matched round 1's rows battle for battle.

| ref | untaught-8 | wins/finished |
|---|---:|---:|
| `popr2_loop` (B2) | **60.75 pp** | 972/1600 |
| `popr2_ctrl` (C2) | **62.06 pp** | 993/1600 |
| `plateau_b1` (G0) | **60.19 pp** ✅ | **963/1600, EXACT**; all 8 per-team rates identical to round 1's |

| contrast | Δ pp | 95 % CI | reading |
|---|---:|---|---|
| **B2 − C2 (the guard)** | **−1.31** | **[−4.19, +1.19]** | \|Δ\| < 3.69 and the CI contains −3.69 → **WITHIN; KILL does NOT fire** |
| B2 − B (banked B, descriptor) | +1.94 | [−1.75, +4.69] | within |
| C2 − C (banked C, descriptor) | +0.81 | [−1.38, +3.12] | within |
| B2 − G0 (descriptor) | +0.56 | [−1.44, +2.25] | within |
| C2 − G0 (descriptor) | +1.88 | [−0.19, +4.25] | within |

Per team, B2 − C2: +0.005 / +0.035 / −0.050 / −0.020 / −0.020 / **−0.090** / +0.010 / +0.025 (team order
of the manifest). The registration's stated worry ("a second −2.4 on top would put B2 − C2 near −4.9") did
NOT happen: the cumulative two-round contrast is −1.31, SMALLER than round 1's −2.44. B2 recovered
(+1.94 over B) more than C2 moved (+0.81 over C). Within the floor is not "no cost".

## 5. §4.3 G-A — SmallRL anchors (greedy vs greedy). **Does NOT fire.**

Tree `7c511161` (not the pin; §4.3), its own worktree (`gen3ai-wt/pin-7c511161-popr2`) and release
`sim_bridge`. Round 1's driver with the models changed (`scripts/ga_driver.sh`): `python -m main.anchors
--model <final_model.zip> --opponent metamon:SmallRL --server rust --regime greedy --teamset {away,home}
--games 100 --team-seed S --seed-base S --device cpu --nice 19`, `OMP_NUM_THREADS=1`, one unit at a time,
S ∈ {0,10,20,30,40,50}, the same S for B2 and C2. Each unit started its own in-repo websocket front end on
a free 9500–9599 port and stopped it itself (no Node server; nothing left listening afterwards). **2,400 games;
all 24 units status OK, n = 100, `regime_verified_decisions` true, `argmax_match_rate` 1.0, every model
loaded `bare`, no `team_source_asymmetry`.** 14 games hit the 250-turn forfeit, 3 ties. The aggregate
(`scripts/ga_aggregate.py`) was exercised first on round 1's units relabelled: it reprints round 1's B − C
−0.0175 [−0.0562, +0.0213] exactly.

| model | pooled 1,200 | rate | Wilson 95 % | away 600 | home 600 |
|---|---:|---:|---|---:|---:|
| B2 (the loop) | 754 | **0.628** | [0.601, 0.655] | 367 | 387 |
| C2 (the control) | 733 | **0.611** | [0.583, 0.638] | 357 | 376 |
| B (banked, round 1) | 733 | 0.611 | [0.583, 0.638] | 348 | 385 |
| C (banked, round 1) | 754 | 0.628 | [0.601, 0.655] | 352 | 402 |
| G0 (banked) | 744 | 0.620 | [0.592, 0.647] | 341 | 403 |

| contrast | Δ | Newcombe 95 % | reading |
|---|---:|---|---|
| **B2 − C2 (the guard)** | **+0.0175** | **[−0.0213, +0.0562]** | Δ > 0; not OUTSIDE BELOW the 0.110 floor → **KILL does NOT fire** |
| B2 − C2, away / home (descriptor) | +0.0167 / +0.0183 | [−0.039, +0.072] / [−0.036, +0.073] | — |
| B2 − B (descriptor) | +0.0175 | [−0.0213, +0.0562] | — |
| C2 − C (descriptor) | −0.0175 | [−0.0562, +0.0213] | — |
| B2 − G0 / C2 − G0 (descriptor) | +0.0083 / −0.0092 | [−0.030, +0.047] / [−0.048, +0.030] | — |

⚠️ **A COINCIDENCE, CHECKED:** B2's total (754) equals round 1's C, and C2's (733) equals round 1's B. It is
NOT a label swap: every new unit's `summary.json` names its own checkpoint (`ai_v13_27_…` / `ai_v13_28_…`),
the away/home splits differ (367/387 vs 352/402; 357/376 vs 348/385), and the per-unit counts differ
(e.g. `away_s0`: B2 67, C2 64, B 58, C 48).

**`peer_clean` false on 2 of 24 units, both COMPLETE halves, COUNTED as registered (H-3):**
`B2_loop_home_s50` (Metamon's post-game `RecursionError` in the `ours_challenge` half, where Metamon is the
ACCEPTOR — H-3 again) and `C2_ctrl_home_s30` (in the `peer_challenge` half, the SOP's H17 case).

## 6. The §5 branch — **N+, FINAL**

Read in the registered order (the first that applies governs):

| branch | status |
|---|---|
| **V** | **ruled out** — `--check` 0 mismatches; STOP-list clean; `plateau_b1` 963/1600 EXACT |
| **K** | **ruled out** — G-U B2 − C2 −1.31 [−4.19, +1.19] vs 3.69; G-A +1.75 [−2.13, +5.62] pp vs 11.0 |
| **R** | impossible — Δ2 < 0 |
| **T** | impossible — the CI's upper end −1.38 does not clear −5.0 |
| **N+** | **✅ GOVERNS** — NOT DETECTED at bar 5.0; ABSORBED (M2 lower bound +3.06) |
| M, E | ruled out (ABSORBED; the CI is not inside ±5.0) |

**The registered consequence:** *B2 beat what it saw — including the new specialist RB — and a fresh best
responder is not measurably weaker at this power, for the second round running.* It counts **2 of 3**
toward the stopping rule.

**The power decision (§5 N+'s pre-condition) — RESOLVED by the owner, 2026-09-25:** there is **NO round 3
on this lineage at this configuration.** The population loop carries into the NEW lineage with BOTH power
levers registered up front, before that lineage's round 1: readers at `--eval-battles 200`, and a pooled
multi-round read registered in advance. The stopping count on this lineage therefore ends at 2 of 3,
un-exhausted; it neither turned (no T) nor was stopped (no third non-detection). The §4.4 side-check is still
read as a descriptor when RC+ finishes; it changes no branch.

## 7. Findings from this read

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
- **G-A's totals coincide across rounds** (B2 = C's 754, C2 = B's 733) — checked, not a label swap (§5).
- **H-3 fired again**: one of the two Metamon `RecursionError`s was in the ACCEPTOR half (§5). The SOP's H17
  wording and `main.anchors`' `peer_recursion_upstream` stamp remain unfixed (anchors owner).
- **The guards ran on a loaded box** (load 21–31 on 16 cores: RB+ training, the M6 stress, an M6 gate) at nice
  19 with no measurable cost: G-U 1 h 23 m (3 workers), G-A 5–6 min per 100-game unit.
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
| `gu_rows/` · `scripts/gu_driver.py` · `scripts/gu_env.sh` | G-U: 4,800 per-battle rows (the `plateau_b1` rows are the reproduction), the resumable driver, its environment |
| `untaught_popr2.{json,md}` · `gu_guard.json` · `gu_aggregate.txt` | G-U aggregate (the tool's own) + the guard rule + the reproduction check |
| `ga_rows/` · `scripts/ga_driver.sh` · `scripts/ga_aggregate.py` · `ga_guard.json` · `ga_aggregate.txt` | G-A: 24 units (`summary.json`, `games.jsonl.gz`, both peer reports; the full trees with server / peer logs are in `~/gen3ai_archive/popr2_read_2026-09-25/ga_rows_full/`), driver, aggregate + guard rule |
| `scripts/side_check.sh` | §4.4, the pending one-liner (run from a worktree root; refuses until RB+ and RC+ have finished; writes `brgap_ext_check.txt`, `brgap_ext.{txt,json,md}`, `convergence_rule.{txt,json}` here) |
