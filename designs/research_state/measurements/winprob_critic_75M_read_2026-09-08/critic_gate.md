# 75M READ · TEST 2 — THE CRITIC GATE (G1–G4 + the G5 controls, G7), at run END

`ai_v12_02_winprob_critic` — the SPARSE win-prob-critic arm — finished at 75,005,952 steps
(`final_model.zip` 2026-09-08 09:41, launcher exit 09:44). This is the calibration half of **D2**
(week plan `week_plan_2026-09-07.md` §1, ledger 2026-09-07 · *REGISTRATION … THREE VALUE-FUNCTION
TESTS at 75M*, `44e3a7f5`). The strength half was already read at n = 12 on 2026-09-07 (ledger · *THE
n = 12 STRENGTH READ*); it is re-run here at run end because the tool refits both sides.

**No refusal.** The tool ran to completion, exit **1** (`VERDICT: MIXED`, criteria not met G1–G4).
A `GateRefusal` is exit 2 and did not occur.

**Every number below is a QUOTE of `main.critic_gate`. No verdict is written here** — tests 1 and 2
are read together by the orchestrator.

---

## 1. Summary table

| endpoint | bar (verbatim from the registration) | reading at the run's last measured step | vs the bar |
|---|---|---|---|
| **G1 — resolution (PRIMARY)** | "the head's RESOLUTION strictly beats the matched-stratum baseline from `winprob_critic_baseline_2026-09-06/`" | 74,000,016 · `all` **0.0452** [0.0331, 0.0615] vs 0.0618 · `bot` **0.0257** [0.0162, 0.0425] vs 0.0337 · `pool` **0.0500** [0.0350, 0.0696] vs 0.0711 | **FAILS on every stratum at every step — 60 of 60 rows read `n`** (20 trace steps × 3 strata). At 74M the `all` and `pool` arm CIs sit ENTIRELY BELOW the bar (strictly worse); the `bot` CI STRADDLES it ⇒ NOT DETECTED on that stratum |
| **G2 — reliability, per-stratum non-inferiority** | "G2/G3 per-stratum non-inferiority" (bot / pool separately) | 74,000,016 · `bot` 0.0028 [0.0017, 0.0085] vs 0.0012 → **n** (CI above base) · `pool` 0.0045 [0.0031, 0.0228] vs 0.0103 → **Y** (point ≤ base) · `all` 0.0023 [0.0017, 0.0085] vs 0.0020 → Y (not gated) | **`bot` fails 19 of 20 steps; `pool` passes 19 of 20** |
| **G3 — ECE, per-stratum non-inferiority** | same | 74,000,016 · `bot` 0.0364 [0.0237, 0.0720] vs 0.0228 → **n** · `pool` 0.0475 [0.0324, 0.1369] vs 0.0875 → **Y** · `all` 0.0263 [0.0225, 0.0748] vs 0.0349 → Y (not gated) | **`bot` fails 11 of 20 steps; `pool` passes 20 of 20** |
| **G4 — skill > 0** | "G4 skill > 0" (cluster CI clears zero) | 74,000,016 · `all` **+0.240** [+0.175, +0.294] **Y** · `bot` **+0.186** [+0.098, +0.256] **Y** · `pool` **+0.213** [+0.115, +0.274] **Y** | **all three strata clear zero at the final step**; over the run `all` 20/20 Y, `bot` 12/20 Y, `pool` 15/20 Y |
| **G7 — stall / episode length** | READ AMENDMENT 6: "the arm's OWN first 2M window of the self-play regime … bar 1.25× of that, SUSTAINED over two consecutive snapshots"; ep_pool DESCRIPTIVE (6b), ep_bots keeps the bar | tool: **`kill: false`, `measured: true`, all 37 cycles `OK`.** 74,000,016 stall 0.0041, ep_bots **21.889**, ep_pool 35.602. Max ep_bots over the run **24.089 (68M)**; max stall rate 0.0123 (66M) | **no breach on either reference** — see the reference note below |
| **Sufficiency — anchored ladder vs `famine_comparator`** | "**NOT INFERIOR by more than the 38-Elo floor at n = 12** (the comparator's count)" | `delta_elo` **−51.0** ELO, 95% CI **[−81.5, −20.5]** (SE 15.56), matched fit size, both sides refit from their own `games.jsonl`. `delta_elo_final_fits` **−47.0** (run 58,000,032 = 2051.4 se 6.8; parent 24,000,000 = 2098.4 se 10.6) — the tool publishes NO interval for that variant | tool sentence, verbatim: *"the arm TRAILS the comparator by 51 ELO at 12 snapshots (floor 38) — the trail EXCEEDS the floor: the ladder half of the famine gate is MET"* → **INFERIOR** on the point estimate. 🚨 **The DELTA's CI straddles the −38 floor**, so the exceedance is **NOT DETECTED** at 95%. And see the node-set hazard below — this n = 12 is **not the registered object** |
| **Untaught meter** | DESCRIPTIVE (state it, never a bar) | **NOT RUN** | `--skip-meter`; see §5 |

🚨 **The `pool` stratum is played under the greedy-vs-stochastic HANDICAP.** On this arm an eval
cycle plays the GREEDY trainee against a STOCHASTIC sentinel with an asymmetric teambuilder
(`eval_worker`: `stochastic = not sentinel_greedy`), worth **+8.9 pp [+7.0, +10.7]** to the newer
snapshot on the 60 pairs both sources cover (ledger 2026-09-07 · *THE LADDER-FIT FIX LANDED*,
`3e6875a5`). The ladder fit drops those sentinel edges; the CALIBRATION strata do not — the pool
traces ARE those games. So `pool` G1/G2/G3/G4 are read on a handicapped population, and the bot
stratum is the unhandicapped one. Bot and pool are never pooled, per the design.

**The tool's own aggregate:** `VERDICT: MIXED — criteria not met: G1, G2, G3, G4` (an across-steps
aggregate; the per-step truth is in the rows above and in the appended rendering).

---

## 2. 🚨 What contradicted the registration — the n = 12 node set is NOT the arm's first 12

The registration reads: *"A is 75M and rev-1 was 25M, so at n = 12 A is at ~24M — the comparison is
at matched COUNT, per the ELO rules."* **That is not what the tool compared, and it could not be.**

At run end the arm's ladder rates only its **20 surviving pool snapshots, 36,000,000 → 74,000,016**.
The self-play pool grooms older snapshots off disk, and `main.critic_gate` re-slices the COMMITTED
ladder's own rated steps (`snapshot_ladder.fit_ladder(..., steps=…)`, the documented path for "a
committed ladder whose pool has since been groomed"). So the arm's **first 12 snapshots are
36M → 58M**, while the comparator's first 12 are its true **2M → 24M**:

| | node #1 | node #12 |
|---|---|---|
| arm (`ai_v12_02_winprob_critic`) | 36,000,000 · 1967.4 | 58,000,032 · **2002.3** (se 10.7) |
| `famine_comparator` (`ai_v9_29_rev1_0823`) | 2,000,016 · 1723.0 | 24,000,000 · **2053.3** (se 11.3) |

Consequences, stated rather than adjudicated:

1. **The comparison is no longer like-for-like in the registered sense.** The arm's 12-node fit
   spans a late, narrow band of its own training (1967 → 2002, +35 Elo) while the comparator's spans
   its whole run (1723 → 2053, +330). Both are 12-node fits anchored to the same pinned bots, so the
   *fit-size* control the rule exists for is satisfied; the *training-position* match is not.
2. **The direction of the asymmetry favours the arm** — its side of the comparison is drawn from
   36–58M rather than 4–26M — and it still trails by 51.
3. **This −51 is NOT comparable to the 26M read's −21** (ledger 2026-09-07 · *THE LADDER-FIT FIX
   LANDED*, `3e6875a5`). That read's n = 12 was the arm's 4M → 26M, a node set no longer on disk.
   Two different objects with the same name; do not narrate a "worsening from −21 to −51".
4. `main.critic_gate` did not silently substitute — it prints the node table and the refit note. The
   substitution is a consequence of pool retention meeting a rule written for a live ladder.

The same caveat applies to §1 of the tool's report (the ladder vs `v9_fold_parent`): **Δ at 14
snapshots +49 ELO [+13, +85]**, with the arm's 14 nodes being 36M → 62M.

---

## 3. The arm's full 20-node ladder, and its final node

`models/ai_v12_02_winprob_critic/snapshot_ladder/ladder.json` (20 rated, converged, 494 frozen pairs
measured, written 09:22 by the run's own PINNED updater — i.e. the PRE-FIX path, which folds the eval
cycles' greedy-vs-stochastic sentinel edges into the fit). Beside it, a read-only refit of the SAME
20 steps with this tree's `fit_ladder` (`write=False`; **47 sentinel edges dropped**):

| step | committed `ladder.json` | refit (sentinel edges dropped) |
|---|---|---|
| 36,000,000 | 1996.6 (se 6.2) | 1954.5 (se 8.4) |
| 38,000,016 | 2013.8 | 1957.8 |
| 40,000,032 | 2014.2 | 1966.9 |
| 42,000,000 | 2013.0 | 1961.4 |
| 44,000,016 | 2008.1 | 1969.9 |
| 46,000,032 | 2011.3 | 1962.0 |
| 48,000,000 | 2030.5 | 1977.2 |
| 50,000,016 | 2025.1 | 1969.3 |
| 52,000,032 | 2047.0 | 1993.5 |
| 54,000,000 | 2044.4 | 1995.0 |
| 56,000,016 | 2022.1 | 1965.4 |
| 58,000,032 | 2051.4 | 1989.8 |
| 60,000,000 | 2036.2 | 1983.3 |
| 62,000,016 | 2046.4 | 1985.2 |
| 64,000,032 | **2061.9** (committed peak) | **2000.2** (refit peak) |
| 66,000,000 | 2037.6 | 1969.6 |
| 68,000,016 | 2049.2 | 1987.3 |
| 70,000,032 | 2060.4 | 1995.0 |
| 72,000,000 | 2047.1 | 1980.3 |
| **74,000,016 (final node)** | **2057.3 (se 7.7)** | **1984.2 (se 8.5)** |

**The final node is now FINAL** — the run is over, no snapshot will be added, so the newest-node
inflation the matched-count rule controls has stopped moving for this ladder. Two caveats that do
not go away: the committed file reads **~+73 high at the final node** against the sentinel-dropped
refit, so the headline `2057.3` is the pre-fix number; and the ladder never rates the 75,005,952
`final_model.zip` itself — its last rated object is the 74,000,016 snapshot.

---

## 4. The G7 reference note (amendment 6 / 6b)

The tool's era reference on this run is **the parent's own recorded eval, worst cycle** —
`ai_v9_59_R2ACTION_0827`, ep_bots **29.4775**, ep_pool **36.046** (bars 1.25× = 36.85 / 45.06). That
is not the arm-own reference READ AMENDMENT 6 registered (the arm's 4–6M self-play window,
ep_bots ≈ 21.911, bar 27.39, sustained two snapshots). **Neither reference is breached:**

- against the tool's reference, all 37 cycles print `OK`, `kill: false`, `measured: true`;
- against amendment 6's arm-own reference, the arm's maximum ep_bots over the whole run is
  **24.089 at 68,000,016** — below 27.39, so no single snapshot breaches, let alone two consecutive;
- the 32M / 34M cycles that fired the standing kill on the old 18.28 reference (ledger 2026-09-07 ·
  *INCIDENT + OWNER RULING*) read **23.791** and **23.487** here and print `OK` on both references.

**ep_pool is DESCRIPTIVE** (amendment 6b): it rises 25.89 (6M) → 36.94 (58M) → 35.60 (74M). The
stall-rate half is clean throughout — maximum **0.0123 at 66M** (3 of 243 captured traces), against
the 0.05 bar, and the tool's own caveat stands: the stall rate is read off the **loss-enriched
capture quota**, so it is an upper-ish bound and never a population rate.

---

## 5. Endpoints NOT computed here

- **Untaught meter (§4 of the gate) — NOT RUN.** `--skip-meter`. The registered D2 command does not
  carry that flag; it was added because the meter's plan is **8,000 battles at concurrency 1** across
  five refs (arm + `v9_fold_parent` + the three G5 controls, 200 games × 8 untaught teams) and arm C
  (`ai_v12_04_pfsp_fork25M`) began training on this box mid-read. A multi-hour CPU battle load beside
  a live run was refused. The column is **DESCRIPTIVE** by registration (a fresh arm against a 28M
  parent), so nothing gated is missing — but it is a deviation from the registered argv and is
  recorded as one. The 26M read (`winprob_critic_26M_n12_read_2026-09-07/`) skipped it identically.
  The three G5 controls were still passed and all five refs RESOLVED (`--dry-run` output).
- **G5 / G6 / G8**, per the tool: not runnable from traces (gaps M1 / M2 / M3). **G9** is runnable
  but by `python -m main.capacity`.

**Falsification clause (design §5.5), which the tool prints and which this read must carry:** *"What
would falsify the design, stated before the data: G1 flat (resolution unmoved) with G2–G4 passing
means the promotion bought calibration this head already had and nothing else — the wrong-meter trap,
and the target/readout diagnosis of §2 would survive intact while *this* remedy for it would not.
That must be reported as loudly as a pass."*

---

## 6. The tool's markdown rendering, verbatim

# CRITIC GATE — `ai_v12_02_winprob_critic` vs `ai_v9_59_R2ACTION_0827`

The pre-registered read of `designs/ai_v12/design_winprob_only_critic.md` §5.5 (endpoints) / §4.3 (bars). Generated 2026-09-08T10:08:02.

| input | spec | resolved file | rung |
|---|---|---|---|
| run | `ai_v12_02_winprob_critic` | `/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic/final_model.zip` | `latest_txt` |
| parent | `v9_fold_parent` | `/home/goodlad/dev/gen3ai/models/ai_v9_59_R2ACTION_0827/final_model.zip` | `explicit_zip` |
| control | `models/ai_v9_195_G5PLAINA_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_195_G5PLAINA_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_196_G5PLAINB_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_196_G5PLAINB_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_197_G5PLAINC_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_197_G5PLAINC_0906/final_model.zip` | `latest_txt` |

## VERDICT — **MIXED**

> criteria not met: G1, G2, G3, G4

## 1. Anchored ladder, at matched SNAPSHOT COUNT

**rating final — the run wrote a final model**

| # | ai_v12_02_winprob_critic step | elo ±95% | ai_v9_59_R2ACTION_0827 step | elo ±95% |
|---|---|---|---|---|
| 1 | 36,000,000 | 1964 ± 19 | 2,000,016 | 1588 ± 81 |
| 2 | 38,000,016 | 1966 ± 19 | 4,000,032 | 1837 ± 57 |
| 3 | 40,000,032 | 1979 ± 19 | 6,000,000 | 1817 ± 58 |
| 4 | 42,000,000 | 1977 ± 19 | 8,000,016 | 1888 ± 55 |
| 5 | 44,000,016 | 1974 ± 19 | 10,000,032 | 1913 ± 55 |
| 6 | 46,000,032 | 1969 ± 19 | 12,000,000 | 1916 ± 55 |
| 7 | 48,000,000 | 1992 ± 20 | 14,000,016 | 1965 ± 55 |
| 8 | 50,000,016 | 1982 ± 19 | 16,000,032 | 1937 ± 55 |
| 9 | 52,000,032 | 2008 ± 20 | 18,000,000 | 1962 ± 55 |
| 10 | 54,000,000 | 2005 ± 20 | 20,000,016 | 1934 ± 74 |
| 11 | 56,000,016 | 1980 ± 19 | 22,000,032 | 1934 ± 74 |
| 12 | 58,000,032 | 1999 ± 20 | 24,000,000 | 1983 ± 74 |
| 13 | 60,000,000 | 1992 ± 20 | 26,000,016 | 1948 ± 30 |
| 14 | 62,000,016 | 1996 ± 20 | 28,000,032 | 1947 ± 30 |

**Δ at 14 snapshots: +49 ELO [+13, +85]** — matched SNAPSHOT COUNT, never matched step. Both fits are anchored to the same pinned bots, so the delta is meaningful; its SE combines two independent fits and carries NO term for the anchor uncertainty they share.

Fit size: BOTH sides were REFIT from their own games.jsonl on their first n snapshots (strict: every snapshot endpoint inside the prefix) with THIS tree's fit_ladder — because the newest node of a longer fit is inflated, AND because a committed ladder.json written before 2026-09-07 folded the eval cycles' greedy-vs-stochastic SENTINEL edges into the fit (+8.9 pp to the newer snapshot; +21..+29 Elo on the newest nodes) — refit: run=True, parent=True; the committed-ladder delta would have read +89.

## 2. Calibration gate (G1–G4) — RESOLUTION is primary

Bars read from `/home/goodlad/dev/gen3ai-wt/read75-gate/designs/research_state/measurements/winprob_critic_baseline_2026-09-06/selection_reweighted.json` (`ai_v9_59_R2ACTION_0827`, steps [26000016, 28000032]; G1 reduce=`max`, G2/G3 reduce=`last`), selection-reweighted. the committed baseline publishes no interval for `resolution`, so G1 compares the ARM's cluster CI against the baseline as a FIXED bar. G2/G3 inherit the same asymmetry: the baseline value they are compared against is a point, and only the ARM carries an interval.

### G1 (resolution, primary) + G4 (skill)

| step | stratum | gated | **resolution** [95% CI] | baseline | Δ | skill [95% CI] | G1 | G4 |
|---|---|---|---|---|---|---|---|---|
| 36,000,000 | `all` | no | **0.0315** [0.0214, 0.0452] | 0.0618 | -0.0304 | +0.144 [+0.089, +0.189] | ❌ | ✅ |
| 36,000,000 | `bot` | yes | **0.0134** [0.0070, 0.0267] | 0.0337 | -0.0203 | +0.108 [+0.011, +0.184] | ❌ | ✅ |
| 36,000,000 | `pool` | yes | **0.0472** [0.0303, 0.0692] | 0.0711 | -0.0239 | +0.085 [-0.078, +0.177] | ❌ | ❌ |
| 38,000,016 | `all` | no | **0.0290** [0.0164, 0.0472] | 0.0618 | -0.0329 | +0.159 [+0.044, +0.265] | ❌ | ✅ |
| 38,000,016 | `bot` | yes | **0.0211** [0.0132, 0.0348] | 0.0337 | -0.0125 | +0.122 [+0.006, +0.229] | ❌ | ✅ |
| 38,000,016 | `pool` | yes | **0.0310** [0.0153, 0.0627] | 0.0711 | -0.0401 | +0.132 [-0.038, +0.262] | ❌ | ❌ |
| 40,000,032 | `all` | no | **0.0351** [0.0183, 0.0600] | 0.0618 | -0.0267 | +0.204 [+0.058, +0.308] | ❌ | ✅ |
| 40,000,032 | `bot` | yes | **0.0133** [0.0063, 0.0263] | 0.0337 | -0.0204 | +0.072 [-0.056, +0.203] | ❌ | ❌ |
| 40,000,032 | `pool` | yes | **0.0459** [0.0209, 0.0798] | 0.0711 | -0.0252 | +0.217 [+0.061, +0.339] | ❌ | ✅ |
| 42,000,000 | `all` | no | **0.0436** [0.0321, 0.0586] | 0.0618 | -0.0182 | +0.266 [+0.194, +0.320] | ❌ | ✅ |
| 42,000,000 | `bot` | yes | **0.0184** [0.0106, 0.0317] | 0.0337 | -0.0153 | +0.175 [+0.027, +0.298] | ❌ | ✅ |
| 42,000,000 | `pool` | yes | **0.0578** [0.0395, 0.0840] | 0.0711 | -0.0133 | +0.230 [+0.107, +0.315] | ❌ | ✅ |
| 44,000,016 | `all` | no | **0.0393** [0.0251, 0.0577] | 0.0618 | -0.0225 | +0.215 [+0.102, +0.305] | ❌ | ✅ |
| 44,000,016 | `bot` | yes | **0.0183** [0.0099, 0.0322] | 0.0337 | -0.0154 | +0.110 [-0.046, +0.241] | ❌ | ❌ |
| 44,000,016 | `pool` | yes | **0.0470** [0.0292, 0.0730] | 0.0711 | -0.0241 | +0.189 [+0.062, +0.285] | ❌ | ✅ |
| 46,000,032 | `all` | no | **0.0408** [0.0241, 0.0615] | 0.0618 | -0.0210 | +0.242 [+0.162, +0.307] | ❌ | ✅ |
| 46,000,032 | `bot` | yes | **0.0185** [0.0099, 0.0310] | 0.0337 | -0.0152 | +0.181 [+0.069, +0.256] | ❌ | ✅ |
| 46,000,032 | `pool` | yes | **0.0455** [0.0268, 0.0711] | 0.0711 | -0.0256 | +0.179 [+0.054, +0.272] | ❌ | ✅ |
| 48,000,000 | `all` | no | **0.0421** [0.0307, 0.0615] | 0.0618 | -0.0197 | +0.220 [+0.132, +0.299] | ❌ | ✅ |
| 48,000,000 | `bot` | yes | **0.0166** [0.0098, 0.0287] | 0.0337 | -0.0171 | +0.121 [-0.004, +0.216] | ❌ | ❌ |
| 48,000,000 | `pool` | yes | **0.0527** [0.0344, 0.0817] | 0.0711 | -0.0184 | +0.171 [+0.011, +0.289] | ❌ | ✅ |
| 50,000,016 | `all` | no | **0.0410** [0.0309, 0.0530] | 0.0618 | -0.0208 | +0.224 [+0.171, +0.271] | ❌ | ✅ |
| 50,000,016 | `bot` | yes | **0.0269** [0.0181, 0.0403] | 0.0337 | -0.0067 | +0.247 [+0.182, +0.307] | ❌ | ✅ |
| 50,000,016 | `pool` | yes | **0.0433** [0.0296, 0.0621] | 0.0711 | -0.0278 | +0.151 [+0.041, +0.233] | ❌ | ✅ |
| 52,000,032 | `all` | no | **0.0397** [0.0268, 0.0582] | 0.0618 | -0.0221 | +0.254 [+0.179, +0.310] | ❌ | ✅ |
| 52,000,032 | `bot` | yes | **0.0133** [0.0080, 0.0232] | 0.0337 | -0.0203 | +0.132 [+0.007, +0.233] | ❌ | ✅ |
| 52,000,032 | `pool` | yes | **0.0451** [0.0306, 0.0710] | 0.0711 | -0.0260 | +0.218 [+0.134, +0.274] | ❌ | ✅ |
| 54,000,000 | `all` | no | **0.0464** [0.0335, 0.0642] | 0.0618 | -0.0154 | +0.259 [+0.166, +0.335] | ❌ | ✅ |
| 54,000,000 | `bot` | yes | **0.0186** [0.0104, 0.0304] | 0.0337 | -0.0150 | +0.189 [+0.068, +0.288] | ❌ | ✅ |
| 54,000,000 | `pool` | yes | **0.0542** [0.0354, 0.0814] | 0.0711 | -0.0169 | +0.194 [+0.024, +0.304] | ❌ | ✅ |
| 56,000,016 | `all` | no | **0.0422** [0.0291, 0.0602] | 0.0618 | -0.0197 | +0.241 [+0.132, +0.309] | ❌ | ✅ |
| 56,000,016 | `bot` | yes | **0.0332** [0.0218, 0.0492] | 0.0337 | -0.0005 | +0.268 [+0.143, +0.351] | ❌ | ✅ |
| 56,000,016 | `pool` | yes | **0.0395** [0.0224, 0.0677] | 0.0711 | -0.0316 | +0.156 [+0.024, +0.253] | ❌ | ✅ |
| 58,000,032 | `all` | no | **0.0395** [0.0266, 0.0592] | 0.0618 | -0.0223 | +0.222 [+0.146, +0.289] | ❌ | ✅ |
| 58,000,032 | `bot` | yes | **0.0130** [0.0054, 0.0279] | 0.0337 | -0.0206 | +0.118 [-0.061, +0.241] | ❌ | ❌ |
| 58,000,032 | `pool` | yes | **0.0447** [0.0311, 0.0678] | 0.0711 | -0.0264 | +0.172 [+0.071, +0.258] | ❌ | ✅ |
| 60,000,000 | `all` | no | **0.0361** [0.0236, 0.0525] | 0.0618 | -0.0257 | +0.202 [+0.131, +0.264] | ❌ | ✅ |
| 60,000,000 | `bot` | yes | **0.0205** [0.0127, 0.0360] | 0.0337 | -0.0131 | +0.154 [-0.009, +0.262] | ❌ | ❌ |
| 60,000,000 | `pool` | yes | **0.0321** [0.0187, 0.0522] | 0.0711 | -0.0390 | +0.126 [+0.025, +0.205] | ❌ | ✅ |
| 62,000,016 | `all` | no | **0.0390** [0.0273, 0.0536] | 0.0618 | -0.0229 | +0.209 [+0.138, +0.273] | ❌ | ✅ |
| 62,000,016 | `bot` | yes | **0.0239** [0.0147, 0.0381] | 0.0337 | -0.0098 | +0.166 [+0.050, +0.270] | ❌ | ✅ |
| 62,000,016 | `pool` | yes | **0.0377** [0.0246, 0.0581] | 0.0711 | -0.0334 | +0.131 [+0.014, +0.217] | ❌ | ✅ |
| 64,000,032 | `all` | no | **0.0261** [0.0158, 0.0411] | 0.0618 | -0.0357 | +0.139 [+0.053, +0.215] | ❌ | ✅ |
| 64,000,032 | `bot` | yes | **0.0174** [0.0094, 0.0323] | 0.0337 | -0.0163 | +0.143 [+0.034, +0.246] | ❌ | ✅ |
| 64,000,032 | `pool` | yes | **0.0263** [0.0123, 0.0477] | 0.0711 | -0.0448 | +0.078 [-0.048, +0.181] | ❌ | ❌ |
| 66,000,000 | `all` | no | **0.0411** [0.0284, 0.0589] | 0.0618 | -0.0208 | +0.184 [+0.088, +0.261] | ❌ | ✅ |
| 66,000,000 | `bot` | yes | **0.0152** [0.0083, 0.0305] | 0.0337 | -0.0185 | +0.085 [-0.021, +0.206] | ❌ | ❌ |
| 66,000,000 | `pool` | yes | **0.0514** [0.0316, 0.0756] | 0.0711 | -0.0197 | +0.163 [+0.002, +0.268] | ❌ | ✅ |
| 68,000,016 | `all` | no | **0.0358** [0.0241, 0.0536] | 0.0618 | -0.0260 | +0.196 [+0.093, +0.292] | ❌ | ✅ |
| 68,000,016 | `bot` | yes | **0.0133** [0.0073, 0.0305] | 0.0337 | -0.0203 | +0.065 [-0.208, +0.221] | ❌ | ❌ |
| 68,000,016 | `pool` | yes | **0.0489** [0.0312, 0.0750] | 0.0711 | -0.0222 | +0.200 [+0.076, +0.302] | ❌ | ✅ |
| 70,000,032 | `all` | no | **0.0437** [0.0296, 0.0663] | 0.0618 | -0.0182 | +0.217 [+0.128, +0.306] | ❌ | ✅ |
| 70,000,032 | `bot` | yes | **0.0313** [0.0209, 0.0539] | 0.0337 | -0.0024 | +0.262 [+0.155, +0.375] | ❌ | ✅ |
| 70,000,032 | `pool` | yes | **0.0384** [0.0183, 0.0685] | 0.0711 | -0.0327 | +0.096 [-0.092, +0.236] | ❌ | ❌ |
| 72,000,000 | `all` | no | **0.0276** [0.0178, 0.0409] | 0.0618 | -0.0342 | +0.136 [+0.057, +0.213] | ❌ | ✅ |
| 72,000,000 | `bot` | yes | **0.0137** [0.0062, 0.0296] | 0.0337 | -0.0199 | +0.040 [-0.143, +0.180] | ❌ | ❌ |
| 72,000,000 | `pool` | yes | **0.0337** [0.0186, 0.0543] | 0.0711 | -0.0374 | +0.104 [-0.037, +0.212] | ❌ | ❌ |
| 74,000,016 | `all` | no | **0.0452** [0.0331, 0.0615] | 0.0618 | -0.0166 | +0.240 [+0.175, +0.294] | ❌ | ✅ |
| 74,000,016 | `bot` | yes | **0.0257** [0.0162, 0.0425] | 0.0337 | -0.0079 | +0.186 [+0.098, +0.256] | ❌ | ✅ |
| 74,000,016 | `pool` | yes | **0.0500** [0.0350, 0.0696] | 0.0711 | -0.0211 | +0.213 [+0.115, +0.274] | ❌ | ✅ |

### G2 / G3 — PER-STRATUM NON-INFERIORITY vs the same-stratum baseline

> OWNER RULING 2026-09-06: §4.3's absolute G2 (reliability <= 0.005) and G3 (ECE <= 0.05) bars are ALREADY BREACHED by the committed baseline on the POOL stratum (reliability 0.0064 / 0.0103, ECE 0.0667 / 0.0875 at 26M / 28M), while §4.3 called G3 a 'no-regression clause' the reweighted baseline already passes — true pooled and on bot, FALSE on pool, and both gates are registered over 'both classes'. As written the arm had to clear a bar its predecessor never cleared. G2 and G3 are therefore PER-STRATUM RELATIVE bars: no worse than the baseline's SAME-stratum value. The absolute numbers stay printed as aspirational targets.

Rule: PASS if the arm's point estimate is <= the baseline's same-stratum value, OR the arm's cluster-bootstrap CI CONTAINS that value (non-inferiority — never a direction claim). FAIL only when the arm's whole CI sits ABOVE it. The baseline value is its own row **at the matched checkpoint** where the artifact carries that step, else its steps reduced with `last` — the cell says which.

| step | stratum | gated | gate | metric | arm [95% CI] | baseline (step) | Δ | verdict | **decided by** | §4.3 absolute (aspirational) |
|---|---|---|---|---|---|---|---|---|---|---|
| 36,000,000 | `all` | no | G2 | reliability | **0.0088** [0.0026, 0.0233] | 0.0020 (@28,000,032, reduced `last`) | +0.0068 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 36,000,000 | `all` | no | G3 | ECE | **0.0827** [0.0418, 0.1324] | 0.0349 (@28,000,032, reduced `last`) | +0.0477 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 36,000,000 | `bot` | yes | G2 | reliability | **0.0029** [0.0015, 0.0094] | 0.0012 (@28,000,032, reduced `last`) | +0.0017 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 36,000,000 | `bot` | yes | G3 | ECE | **0.0395** [0.0205, 0.0698] | 0.0228 (@28,000,032, reduced `last`) | +0.0167 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 36,000,000 | `pool` | yes | G2 | reliability | **0.0309** [0.0110, 0.0751] | 0.0103 (@28,000,032, reduced `last`) | +0.0206 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 36,000,000 | `pool` | yes | G3 | ECE | **0.1579** [0.0866, 0.2553] | 0.0875 (@28,000,032, reduced `last`) | +0.0704 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 38,000,016 | `all` | no | G2 | reliability | **0.0036** [0.0018, 0.0132] | 0.0020 (@28,000,032, reduced `last`) | +0.0016 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 38,000,016 | `all` | no | G3 | ECE | **0.0499** [0.0318, 0.0896] | 0.0349 (@28,000,032, reduced `last`) | +0.0150 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 38,000,016 | `bot` | yes | G2 | reliability | **0.0074** [0.0044, 0.0142] | 0.0012 (@28,000,032, reduced `last`) | +0.0062 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 38,000,016 | `bot` | yes | G3 | ECE | **0.0597** [0.0387, 0.0965] | 0.0228 (@28,000,032, reduced `last`) | +0.0369 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 38,000,016 | `pool` | yes | G2 | reliability | **0.0051** [0.0028, 0.0270] | 0.0103 (@28,000,032, reduced `last`) | -0.0052 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 38,000,016 | `pool` | yes | G3 | ECE | **0.0599** [0.0425, 0.1305] | 0.0875 (@28,000,032, reduced `last`) | -0.0276 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 40,000,032 | `all` | no | G2 | reliability | **0.0023** [0.0018, 0.0090] | 0.0020 (@28,000,032, reduced `last`) | +0.0003 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 40,000,032 | `all` | no | G3 | ECE | **0.0432** [0.0304, 0.0793] | 0.0349 (@28,000,032, reduced `last`) | +0.0082 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 40,000,032 | `bot` | yes | G2 | reliability | **0.0052** [0.0028, 0.0139] | 0.0012 (@28,000,032, reduced `last`) | +0.0040 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 40,000,032 | `bot` | yes | G3 | ECE | **0.0555** [0.0297, 0.0999] | 0.0228 (@28,000,032, reduced `last`) | +0.0327 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 40,000,032 | `pool` | yes | G2 | reliability | **0.0037** [0.0027, 0.0192] | 0.0103 (@28,000,032, reduced `last`) | -0.0066 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline MISSES |
| 40,000,032 | `pool` | yes | G3 | ECE | **0.0517** [0.0327, 0.1217] | 0.0875 (@28,000,032, reduced `last`) | -0.0357 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 42,000,000 | `all` | no | G2 | reliability | **0.0035** [0.0022, 0.0081] | 0.0020 (@28,000,032, reduced `last`) | +0.0014 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 42,000,000 | `all` | no | G3 | ECE | **0.0248** [0.0217, 0.0607] | 0.0349 (@28,000,032, reduced `last`) | -0.0101 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 42,000,000 | `bot` | yes | G2 | reliability | **0.0049** [0.0023, 0.0106] | 0.0012 (@28,000,032, reduced `last`) | +0.0037 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 42,000,000 | `bot` | yes | G3 | ECE | **0.0566** [0.0311, 0.0811] | 0.0228 (@28,000,032, reduced `last`) | +0.0338 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 42,000,000 | `pool` | yes | G2 | reliability | **0.0116** [0.0048, 0.0391] | 0.0103 (@28,000,032, reduced `last`) | +0.0013 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 42,000,000 | `pool` | yes | G3 | ECE | **0.0763** [0.0400, 0.1715] | 0.0875 (@28,000,032, reduced `last`) | -0.0111 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 44,000,016 | `all` | no | G2 | reliability | **0.0020** [0.0013, 0.0095] | 0.0020 (@28,000,032, reduced `last`) | +0.0000 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 44,000,016 | `all` | no | G3 | ECE | **0.0306** [0.0223, 0.0736] | 0.0349 (@28,000,032, reduced `last`) | -0.0043 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 44,000,016 | `bot` | yes | G2 | reliability | **0.0057** [0.0025, 0.0134] | 0.0012 (@28,000,032, reduced `last`) | +0.0045 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 44,000,016 | `bot` | yes | G3 | ECE | **0.0559** [0.0316, 0.0961] | 0.0228 (@28,000,032, reduced `last`) | +0.0331 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 44,000,016 | `pool` | yes | G2 | reliability | **0.0056** [0.0025, 0.0230] | 0.0103 (@28,000,032, reduced `last`) | -0.0048 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 44,000,016 | `pool` | yes | G3 | ECE | **0.0642** [0.0381, 0.1297] | 0.0875 (@28,000,032, reduced `last`) | -0.0233 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 46,000,032 | `all` | no | G2 | reliability | **0.0019** [0.0009, 0.0101] | 0.0020 (@28,000,032, reduced `last`) | -0.0001 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 46,000,032 | `all` | no | G3 | ECE | **0.0332** [0.0167, 0.0777] | 0.0349 (@28,000,032, reduced `last`) | -0.0017 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 46,000,032 | `bot` | yes | G2 | reliability | **0.0022** [0.0013, 0.0063] | 0.0012 (@28,000,032, reduced `last`) | +0.0010 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 46,000,032 | `bot` | yes | G3 | ECE | **0.0255** [0.0174, 0.0517] | 0.0228 (@28,000,032, reduced `last`) | +0.0027 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 46,000,032 | `pool` | yes | G2 | reliability | **0.0084** [0.0027, 0.0365] | 0.0103 (@28,000,032, reduced `last`) | -0.0019 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 46,000,032 | `pool` | yes | G3 | ECE | **0.0796** [0.0410, 0.1746] | 0.0875 (@28,000,032, reduced `last`) | -0.0079 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 48,000,000 | `all` | no | G2 | reliability | **0.0035** [0.0018, 0.0146] | 0.0020 (@28,000,032, reduced `last`) | +0.0015 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 48,000,000 | `all` | no | G3 | ECE | **0.0467** [0.0227, 0.1058] | 0.0349 (@28,000,032, reduced `last`) | +0.0118 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 48,000,000 | `bot` | yes | G2 | reliability | **0.0052** [0.0024, 0.0116] | 0.0012 (@28,000,032, reduced `last`) | +0.0040 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 48,000,000 | `bot` | yes | G3 | ECE | **0.0471** [0.0275, 0.0732] | 0.0228 (@28,000,032, reduced `last`) | +0.0243 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 48,000,000 | `pool` | yes | G2 | reliability | **0.0144** [0.0046, 0.0481] | 0.0103 (@28,000,032, reduced `last`) | +0.0041 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 48,000,000 | `pool` | yes | G3 | ECE | **0.1124** [0.0484, 0.2096] | 0.0875 (@28,000,032, reduced `last`) | +0.0249 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 50,000,016 | `all` | no | G2 | reliability | **0.0058** [0.0030, 0.0140] | 0.0020 (@28,000,032, reduced `last`) | +0.0037 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 50,000,016 | `all` | no | G3 | ECE | **0.0554** [0.0281, 0.1008] | 0.0349 (@28,000,032, reduced `last`) | +0.0205 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 50,000,016 | `bot` | yes | G2 | reliability | **0.0023** [0.0016, 0.0069] | 0.0012 (@28,000,032, reduced `last`) | +0.0011 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 50,000,016 | `bot` | yes | G3 | ECE | **0.0133** [0.0111, 0.0508] | 0.0228 (@28,000,032, reduced `last`) | -0.0095 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 50,000,016 | `pool` | yes | G2 | reliability | **0.0136** [0.0045, 0.0430] | 0.0103 (@28,000,032, reduced `last`) | +0.0033 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 50,000,016 | `pool` | yes | G3 | ECE | **0.1002** [0.0470, 0.1932] | 0.0875 (@28,000,032, reduced `last`) | +0.0127 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 52,000,032 | `all` | no | G2 | reliability | **0.0006** [0.0006, 0.0054] | 0.0020 (@28,000,032, reduced `last`) | -0.0014 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 52,000,032 | `all` | no | G3 | ECE | **0.0126** [0.0137, 0.0545] | 0.0349 (@28,000,032, reduced `last`) | -0.0223 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 52,000,032 | `bot` | yes | G2 | reliability | **0.0043** [0.0021, 0.0091] | 0.0012 (@28,000,032, reduced `last`) | +0.0031 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 52,000,032 | `bot` | yes | G3 | ECE | **0.0413** [0.0276, 0.0669] | 0.0228 (@28,000,032, reduced `last`) | +0.0185 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 52,000,032 | `pool` | yes | G2 | reliability | **0.0013** [0.0009, 0.0164] | 0.0103 (@28,000,032, reduced `last`) | -0.0090 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline MISSES |
| 52,000,032 | `pool` | yes | G3 | ECE | **0.0272** [0.0202, 0.1080] | 0.0875 (@28,000,032, reduced `last`) | -0.0603 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline MISSES |
| 54,000,000 | `all` | no | G2 | reliability | **0.0026** [0.0012, 0.0120] | 0.0020 (@28,000,032, reduced `last`) | +0.0006 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 54,000,000 | `all` | no | G3 | ECE | **0.0374** [0.0190, 0.0908] | 0.0349 (@28,000,032, reduced `last`) | +0.0025 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 54,000,000 | `bot` | yes | G2 | reliability | **0.0024** [0.0009, 0.0080] | 0.0012 (@28,000,032, reduced `last`) | +0.0012 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 54,000,000 | `bot` | yes | G3 | ECE | **0.0333** [0.0154, 0.0588] | 0.0228 (@28,000,032, reduced `last`) | +0.0105 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 54,000,000 | `pool` | yes | G2 | reliability | **0.0120** [0.0032, 0.0460] | 0.0103 (@28,000,032, reduced `last`) | +0.0016 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 54,000,000 | `pool` | yes | G3 | ECE | **0.0976** [0.0386, 0.1973] | 0.0875 (@28,000,032, reduced `last`) | +0.0101 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 56,000,016 | `all` | no | G2 | reliability | **0.0006** [0.0006, 0.0063] | 0.0020 (@28,000,032, reduced `last`) | -0.0014 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 56,000,016 | `all` | no | G3 | ECE | **0.0178** [0.0142, 0.0611] | 0.0349 (@28,000,032, reduced `last`) | -0.0171 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 56,000,016 | `bot` | yes | G2 | reliability | **0.0051** [0.0026, 0.0110] | 0.0012 (@28,000,032, reduced `last`) | +0.0039 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 56,000,016 | `bot` | yes | G3 | ECE | **0.0424** [0.0217, 0.0727] | 0.0228 (@28,000,032, reduced `last`) | +0.0196 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 56,000,016 | `pool` | yes | G2 | reliability | **0.0060** [0.0025, 0.0274] | 0.0103 (@28,000,032, reduced `last`) | -0.0043 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 56,000,016 | `pool` | yes | G3 | ECE | **0.0663** [0.0356, 0.1419] | 0.0875 (@28,000,032, reduced `last`) | -0.0212 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 58,000,032 | `all` | no | G2 | reliability | **0.0016** [0.0010, 0.0079] | 0.0020 (@28,000,032, reduced `last`) | -0.0004 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 58,000,032 | `all` | no | G3 | ECE | **0.0361** [0.0247, 0.0771] | 0.0349 (@28,000,032, reduced `last`) | +0.0012 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 58,000,032 | `bot` | yes | G2 | reliability | **0.0030** [0.0016, 0.0132] | 0.0012 (@28,000,032, reduced `last`) | +0.0018 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 58,000,032 | `bot` | yes | G3 | ECE | **0.0369** [0.0261, 0.0693] | 0.0228 (@28,000,032, reduced `last`) | +0.0141 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 58,000,032 | `pool` | yes | G2 | reliability | **0.0076** [0.0035, 0.0307] | 0.0103 (@28,000,032, reduced `last`) | -0.0027 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 58,000,032 | `pool` | yes | G3 | ECE | **0.0708** [0.0482, 0.1616] | 0.0875 (@28,000,032, reduced `last`) | -0.0167 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 60,000,000 | `all` | no | G2 | reliability | **0.0010** [0.0007, 0.0063] | 0.0020 (@28,000,032, reduced `last`) | -0.0010 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 60,000,000 | `all` | no | G3 | ECE | **0.0236** [0.0175, 0.0595] | 0.0349 (@28,000,032, reduced `last`) | -0.0113 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 60,000,000 | `bot` | yes | G2 | reliability | **0.0054** [0.0018, 0.0134] | 0.0012 (@28,000,032, reduced `last`) | +0.0041 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 60,000,000 | `bot` | yes | G3 | ECE | **0.0565** [0.0226, 0.0906] | 0.0228 (@28,000,032, reduced `last`) | +0.0337 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 60,000,000 | `pool` | yes | G2 | reliability | **0.0054** [0.0024, 0.0293] | 0.0103 (@28,000,032, reduced `last`) | -0.0049 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 60,000,000 | `pool` | yes | G3 | ECE | **0.0655** [0.0401, 0.1627] | 0.0875 (@28,000,032, reduced `last`) | -0.0219 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 62,000,016 | `all` | no | G2 | reliability | **0.0021** [0.0012, 0.0072] | 0.0020 (@28,000,032, reduced `last`) | +0.0001 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 62,000,016 | `all` | no | G3 | ECE | **0.0387** [0.0246, 0.0700] | 0.0349 (@28,000,032, reduced `last`) | +0.0038 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 62,000,016 | `bot` | yes | G2 | reliability | **0.0068** [0.0032, 0.0141] | 0.0012 (@28,000,032, reduced `last`) | +0.0055 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 62,000,016 | `bot` | yes | G3 | ECE | **0.0623** [0.0383, 0.0900] | 0.0228 (@28,000,032, reduced `last`) | +0.0395 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 62,000,016 | `pool` | yes | G2 | reliability | **0.0082** [0.0034, 0.0319] | 0.0103 (@28,000,032, reduced `last`) | -0.0021 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 62,000,016 | `pool` | yes | G3 | ECE | **0.0722** [0.0464, 0.1565] | 0.0875 (@28,000,032, reduced `last`) | -0.0153 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 64,000,032 | `all` | no | G2 | reliability | **0.0035** [0.0019, 0.0111] | 0.0020 (@28,000,032, reduced `last`) | +0.0015 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 64,000,032 | `all` | no | G3 | ECE | **0.0515** [0.0289, 0.0965] | 0.0349 (@28,000,032, reduced `last`) | +0.0166 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 64,000,032 | `bot` | yes | G2 | reliability | **0.0023** [0.0014, 0.0077] | 0.0012 (@28,000,032, reduced `last`) | +0.0011 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 64,000,032 | `bot` | yes | G3 | ECE | **0.0411** [0.0210, 0.0755] | 0.0228 (@28,000,032, reduced `last`) | +0.0183 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 64,000,032 | `pool` | yes | G2 | reliability | **0.0114** [0.0057, 0.0335] | 0.0103 (@28,000,032, reduced `last`) | +0.0011 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 64,000,032 | `pool` | yes | G3 | ECE | **0.0958** [0.0573, 0.1700] | 0.0875 (@28,000,032, reduced `last`) | +0.0083 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 66,000,000 | `all` | no | G2 | reliability | **0.0052** [0.0020, 0.0173] | 0.0020 (@28,000,032, reduced `last`) | +0.0032 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline meets |
| 66,000,000 | `all` | no | G3 | ECE | **0.0677** [0.0342, 0.1163] | 0.0349 (@28,000,032, reduced `last`) | +0.0328 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 66,000,000 | `bot` | yes | G2 | reliability | **0.0041** [0.0018, 0.0182] | 0.0012 (@28,000,032, reduced `last`) | +0.0028 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 66,000,000 | `bot` | yes | G3 | ECE | **0.0569** [0.0237, 0.1203] | 0.0228 (@28,000,032, reduced `last`) | +0.0341 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 66,000,000 | `pool` | yes | G2 | reliability | **0.0138** [0.0044, 0.0463] | 0.0103 (@28,000,032, reduced `last`) | +0.0035 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 66,000,000 | `pool` | yes | G3 | ECE | **0.1088** [0.0498, 0.2013] | 0.0875 (@28,000,032, reduced `last`) | +0.0213 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 68,000,016 | `all` | no | G2 | reliability | **0.0030** [0.0014, 0.0114] | 0.0020 (@28,000,032, reduced `last`) | +0.0010 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 68,000,016 | `all` | no | G3 | ECE | **0.0469** [0.0256, 0.0844] | 0.0349 (@28,000,032, reduced `last`) | +0.0119 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 68,000,016 | `bot` | yes | G2 | reliability | **0.0070** [0.0012, 0.0349] | 0.0012 (@28,000,032, reduced `last`) | +0.0058 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 68,000,016 | `bot` | yes | G3 | ECE | **0.0387** [0.0197, 0.0877] | 0.0228 (@28,000,032, reduced `last`) | +0.0159 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 68,000,016 | `pool` | yes | G2 | reliability | **0.0071** [0.0025, 0.0302] | 0.0103 (@28,000,032, reduced `last`) | -0.0032 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 68,000,016 | `pool` | yes | G3 | ECE | **0.0732** [0.0384, 0.1534] | 0.0875 (@28,000,032, reduced `last`) | -0.0143 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 70,000,032 | `all` | no | G2 | reliability | **0.0040** [0.0014, 0.0175] | 0.0020 (@28,000,032, reduced `last`) | +0.0020 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 70,000,032 | `all` | no | G3 | ECE | **0.0554** [0.0285, 0.1160] | 0.0349 (@28,000,032, reduced `last`) | +0.0205 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 70,000,032 | `bot` | yes | G2 | reliability | **0.0027** [0.0023, 0.0090] | 0.0012 (@28,000,032, reduced `last`) | +0.0015 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 70,000,032 | `bot` | yes | G3 | ECE | **0.0223** [0.0203, 0.0654] | 0.0228 (@28,000,032, reduced `last`) | -0.0005 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 70,000,032 | `pool` | yes | G2 | reliability | **0.0169** [0.0061, 0.0550] | 0.0103 (@28,000,032, reduced `last`) | +0.0066 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 70,000,032 | `pool` | yes | G3 | ECE | **0.1191** [0.0665, 0.2237] | 0.0875 (@28,000,032, reduced `last`) | +0.0316 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 72,000,000 | `all` | no | G2 | reliability | **0.0041** [0.0019, 0.0123] | 0.0020 (@28,000,032, reduced `last`) | +0.0021 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 72,000,000 | `all` | no | G3 | ECE | **0.0517** [0.0332, 0.0907] | 0.0349 (@28,000,032, reduced `last`) | +0.0168 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 72,000,000 | `bot` | yes | G2 | reliability | **0.0097** [0.0039, 0.0226] | 0.0012 (@28,000,032, reduced `last`) | +0.0084 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 72,000,000 | `bot` | yes | G3 | ECE | **0.0654** [0.0428, 0.0983] | 0.0228 (@28,000,032, reduced `last`) | +0.0426 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 72,000,000 | `pool` | yes | G2 | reliability | **0.0121** [0.0044, 0.0402] | 0.0103 (@28,000,032, reduced `last`) | +0.0018 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 72,000,000 | `pool` | yes | G3 | ECE | **0.1033** [0.0539, 0.1764] | 0.0875 (@28,000,032, reduced `last`) | +0.0158 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 74,000,016 | `all` | no | G2 | reliability | **0.0023** [0.0017, 0.0085] | 0.0020 (@28,000,032, reduced `last`) | +0.0003 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 74,000,016 | `all` | no | G3 | ECE | **0.0263** [0.0225, 0.0748] | 0.0349 (@28,000,032, reduced `last`) | -0.0086 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 74,000,016 | `bot` | yes | G2 | reliability | **0.0028** [0.0017, 0.0085] | 0.0012 (@28,000,032, reduced `last`) | +0.0016 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 74,000,016 | `bot` | yes | G3 | ECE | **0.0364** [0.0237, 0.0720] | 0.0228 (@28,000,032, reduced `last`) | +0.0136 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 74,000,016 | `pool` | yes | G2 | reliability | **0.0045** [0.0031, 0.0228] | 0.0103 (@28,000,032, reduced `last`) | -0.0059 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline MISSES |
| 74,000,016 | `pool` | yes | G3 | ECE | **0.0475** [0.0324, 0.1369] | 0.0875 (@28,000,032, reduced `last`) | -0.0400 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline MISSES |

_`all` is reported for context and NEVER gated: it averages two populations whose measured calibration bias has opposite sign._

## 2b. FAMINE pre-test — does terminal-only learn at the incumbent's rate?

Comparator `famine_comparator` → `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip`; floor **38 ELO** (from registry `famine_comparator`.floor_elo).

**the arm TRAILS the comparator by 51 ELO at 12 snapshots (floor 38) — the trail EXCEEDS the floor: the ladder half of the famine gate is MET** (signed trail +51, positive = behind).

> RULE: at ~5M: trailing the comparator by more than the floor at matched SNAPSHOT COUNT **AND** win_rate_vs_bots not rising ⇒ terminal-only starves ⇒ kill the arm and launch FROZEN-φ.
>
> This computes the LADDER half only — win_rate_vs_bots is the AND-gate's other half.
>
> ⚠️ the incumbent had PBRS AND PopArt AND the shaped critic, so this is a rate comparison ACROSS RECIPES and the floor is the incumbent's own run-to-run noise, not a replicate of this arm. A trail inside the floor is NOT evidence the two recipes are equivalent — only that starvation has not been demonstrated.

## 3. G7 — stall rate + episode length (KILL condition)

Thresholds: stall rate ≤ 0.05 (a battle at ≥ 250 turns is a stall), episode length ≤ 1.25× the era's. (the design registers G7 as 'no increase over the era, pre-registered threshold' and names no number; this threshold is main.critic_gate's default, not the design's.)

| step | stall rate (captured) | mean turns | ep_len bots | ep_len pool | verdict |
|---|---|---|---|---|---|
| 2,000,016 | — | — | 20.70 | — | OK |
| 4,000,032 | — | — | 21.99 | — | OK |
| 6,000,000 | — | — | 21.84 | 25.89 | OK |
| 8,000,016 | — | — | 21.80 | 28.35 | OK |
| 10,000,032 | — | — | 22.57 | 31.05 | OK |
| 12,000,000 | — | — | 22.34 | 31.10 | OK |
| 14,000,016 | — | — | 21.94 | 32.62 | OK |
| 16,000,032 | — | — | 22.32 | 32.03 | OK |
| 18,000,000 | — | — | 21.47 | 31.75 | OK |
| 20,000,016 | — | — | 21.93 | 31.87 | OK |
| 22,000,032 | — | — | 22.78 | 32.07 | OK |
| 24,000,000 | — | — | 21.54 | 32.08 | OK |
| 26,000,016 | — | — | 21.44 | 31.33 | OK |
| 28,000,032 | — | — | 21.61 | 31.50 | OK |
| 30,000,000 | — | — | 21.83 | 31.58 | OK |
| 32,000,016 | — | — | 23.79 | 33.14 | OK |
| 34,000,032 | — | — | 23.49 | 31.69 | OK |
| 36,000,000 | 0.0000 | 26.7 | 22.58 | 33.00 | OK |
| 38,000,016 | 0.0044 | 28.4 | 22.94 | 31.96 | OK |
| 40,000,032 | 0.0044 | 27.2 | 23.12 | 31.51 | OK |
| 42,000,000 | 0.0000 | 26.6 | 22.76 | 31.85 | OK |
| 44,000,016 | 0.0042 | 26.9 | 23.64 | 31.95 | OK |
| 46,000,032 | 0.0000 | 26.6 | 22.83 | 32.44 | OK |
| 48,000,000 | 0.0000 | 28.0 | 23.46 | 32.70 | OK |
| 50,000,016 | 0.0000 | 27.4 | 22.82 | 33.04 | OK |
| 52,000,032 | 0.0000 | 28.2 | 22.88 | 35.60 | OK |
| 54,000,000 | 0.0000 | 27.1 | 23.22 | 35.60 | OK |
| 56,000,016 | 0.0000 | 28.8 | 23.85 | 36.03 | OK |
| 58,000,032 | 0.0000 | 29.6 | 23.93 | 36.94 | OK |
| 60,000,000 | 0.0000 | 26.7 | 23.17 | 34.83 | OK |
| 62,000,016 | 0.0000 | 28.1 | 23.34 | 34.78 | OK |
| 64,000,032 | 0.0000 | 27.9 | 23.69 | 35.42 | OK |
| 66,000,000 | 0.0123 | 28.7 | 23.50 | 33.80 | OK |
| 68,000,016 | 0.0000 | 28.7 | 24.09 | 35.54 | OK |
| 70,000,032 | 0.0043 | 27.9 | 23.27 | 34.69 | OK |
| 72,000,000 | 0.0000 | 27.0 | 22.32 | 35.60 | OK |
| 74,000,016 | 0.0041 | 26.0 | 21.89 | 35.60 | OK |

_the captured eval traces' per-battle summary meta.turns / meta.result — the CAPTURE QUOTA, which is loss-enriched by design (agents.training.trace_selection), so read it as an upper-ish bound, never as a population rate_

## 4. Untaught meter — with a CONTINUATION control

_skipped (`--skip-meter`)_

## Not runnable here

| # | criterion | why |
|---|---|---|
| G5 | sd_true_excess, floor-subtracted, per population | gap M1 — not runnable from traces |
| G6 | the MIRROR TABLE (no cell crossing 0.50) | gap M2 — not runnable from traces |
| G8 | win_mask coverage >= a pre-registered floor | gap M3 — the run must record it |
| G9 | capacity value_pooled participation ratio | runnable, but by `python -m main.capacity` |

*Falsification clause (design §5.5, verbatim): What would falsify the design, stated before the data: G1 flat (resolution unmoved) with G2–G4 passing means the promotion bought calibration this head already had and nothing else — the wrong-meter trap, and the target/readout diagnosis of §2 would survive intact while *this* remedy for it would not. That must be reported as loudly as a pass.*

---

## 7. The command, and the paths

Run 2026-09-08 ~10:1x–10:2x PT from the worktree `/home/goodlad/dev/gen3ai-wt/read75-gate`
(branch `read75-gate-0908`, off `main` @ `b697e82a`), CPU only, `nice -n 10`. `models/` was read by
absolute path from the MAIN checkout and **nothing was written under `models/`**.

```bash
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=""        # the GPU is arm C's
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/gate_75M
nice -n 10 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate \
  ai_v12_02_winprob_critic \
  --parent v9_fold_parent \
  --famine-comparator famine_comparator \
  --control models/ai_v9_195_G5PLAINA_0906 \
            models/ai_v9_196_G5PLAINB_0906 \
            models/ai_v9_197_G5PLAINC_0906 \
  --skip-meter \
  --json $OUT/critic_gate_75M_nometer.json \
  --md   $OUT/critic_gate_75M_nometer.md
# exit 1 — VERDICT MIXED (a GateRefusal would be exit 2; none occurred)
```

`--dry-run` was run first with the identical argv minus `--skip-meter`/`--json`/`--md`: every input
resolved, including all three G5 controls and the meter's eight untaught teams.

**Resolved inputs** (as the tool printed them):

| role | resolved to | rung / rule |
|---|---|---|
| run | `models/ai_v12_02_winprob_critic/final_model.zip` @75,005,952 | `latest_txt` / `last_snapshot` |
| `--parent` | baseline `v9_fold_parent` = `ai_v9_59_R2ACTION_0827@28,115,184` | `explicit_zip` |
| `--famine-comparator` | baseline `famine_comparator` = `ai_v9_29_rev1_0823@25,067,760`, floor **38.0** ELO | `explicit_zip` |
| `--control` ×3 (the G5 arms) | `ai_v9_195_G5PLAINA_0906` / `ai_v9_196_G5PLAINB_0906` / `ai_v9_197_G5PLAINC_0906`, each @29,294,832 | `latest_txt` / `last_snapshot` |
| calibration bars | `designs/research_state/measurements/winprob_critic_baseline_2026-09-06/selection_reweighted.json` — `ai_v9_59_R2ACTION_0827` steps [26,000,016 · 28,000,032], strata `all` / `bot` / `pool`, `reduce=max` for G1 and `last` for G2/G3 | committed artifact |
| traces | 20 steps, 4,665 battles; reweight resolved for 37 steps | — |

The G5 control arms are **not** in `designs/baselines.json`, so they are named by run directory
exactly as the 10M and 26M reads named them (`winprob_critic_10M_read_2026-09-07/`,
`winprob_critic_26M_n12_read_2026-09-07/`). The three baselines that ARE registry names were passed
by name.

**Artifacts in this directory:**

| file | what |
|---|---|
| `critic_gate.md` | this report — summary first, the tool's own markdown appended verbatim in §6 |
| `critic_gate.json` | the tool's `--json` output, unmodified |
| `critic_gate_stdout.txt` | the console rendering (the G1/G4, G2/G3 and G7 tables as printed) |
