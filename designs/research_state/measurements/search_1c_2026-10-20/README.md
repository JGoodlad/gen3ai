# STAGE 1c: more or team-level proposals, importance-resampled by the opponent's own choices (2026-10-20)

Stage 1c of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.3, registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any world of a registered state was built. The numbers and their
reading go to the design doc, `search_ledger.md`, `UNDERSTANDING.md` and the ledger once read.

**Status: READ (§5 phase A, §6 phase B). THE UNIT: NEARER BUT NOT PAYING.** The opponent's choices pick slightly nearer worlds (snapshot opponents only), and honest search over them still does not beat the policy.

| file | what |
|---|---|
| `PREREGISTRATION.md`, `c1c_registration.json` | the arms, the meters, the rules, the bar, the stops; the registered files' sha256 |
| `c1c_common.py` | the constants, the arms as (set, first n) slices, the per-world facts meter (registered) |
| `run_1c.py` | PHASE A's driver: per state 128 F + 128 C_ladder worlds (`dice="rolls"`), the opponent-choice likelihood, the facts, each weighted arm's resample (`opp_likelihood.resample`); `--one-chunk`, `--recheck` (G3), `--smoke`, `--status` |
| `play_idx_1c.py`, `play_1c.py`, `run_play.sh` | PHASE B (only when the bar is met): the play indices computed at the DRIVER (`play_idx_1c.py`: the pin's `main` predates `opp_likelihood`), the identity tooth, then every played arm's 8 worlds through the pilot's driver at the pin checkout |
| `loop_1c.sh`, `launch_1c.sh` | the detached runner: one heavy slot per bounded chunk; the registered stops |
| `read_1c.py` | the registered read: `a` (phase A, `--interim`), `b` (phase B), `bcheck` (phase B's plumbing), `selfcheck` |

## 1. Checkouts and archive

| | |
|---|---|
| DRIVER | `~/.cache/gen3ai/search_1c/driver`: a detached checkout of the commit that ships this unit's code (every row's `head`). Keep it until the read |
| PIN (phase B) | `~/.cache/gen3ai/search_1c/pin_rolls`: X26's pin `9c033489` + the damage-roll control `fc010d7d3` cherry-picked (its tree equals stage 1's play commit `13ffd55c`), its own `src/rust_env` release build |
| archive | `~/gen3ai_archive/search_1c/` (`rows/`, `worlds/`, `recheck/`, `play/<arm>/`, `ws1/` (the corpus cache, a read-only copy of stage 1's), `logs/`, `STOP_AT`, `A_DONE` / `A_STOPPED`, `STOP_B`, `B_DONE` / `B_STOPPED`) |
| inputs (read only) | `~/gen3ai_archive/hidden_info_maturity/` (the bank, the 225M checkpoint copy, roots, truth and banked F rows); `~/gen3ai_archive/search_l3/` (G4); stage 1's corpus cache and `alpha_fit.json` |
| smoke | `~/gen3ai_archive/search_1c_smoke/` (4 non-LEAN states at N = 64; never read as output) |

## 2. Operate

```bash
D=~/.cache/gen3ai/search_1c/driver/designs/research_state/measurements/search_1c_2026-10-20
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $D/run_1c.py --status                   # progress counts only (no meter)
tail -f ~/gen3ai_archive/search_1c/logs/queue.log
bash $D/launch_1c.sh a                          # (re)start phase A from the DRIVER checkout; STOP_AT kept
cd $D && $PY -I read_1c.py a --out result_1c_a.json        # THE READ, once A_DONE / A_STOPPED and the recheck exist
bash $D/launch_1c.sh play                       # phase B, only when result_1c_a.json's bar is met
cd $D && $PY -I read_1c.py b --out result_1c_b.json
```

CPU only (`CUDA_VISIBLE_DEVICES=""`), SCHED_IDLE, `nice -n 19`, `taskset -c 5-7,13-15`, one heavy slot per ≤ 45-min
chunk (root rule 5b).

## 3. The run (operational log; none of it changes the design)

- **Launched** 2026-10-20 04:41 PDT from `664385776`; `STOP_AT` = 2026-10-21 10:41 PDT (registered, kept by every
  relaunch). Stopped by its owner at 05:22 while still waiting for its first heavy slot (no state built) to register
  AMENDMENT A; **relaunched** 05:29 from `a144482b0`.
- **First chunk** (06:26–06:59): 27 states, ≈ 90 s a state (27–245 s), peak 6.3 GB of the 16-GB cap, 0 defects, 0
  exclusions, 0 parked.
- **THE ORCHESTRATOR'S RULING (b), 2026-10-20 08:36 PDT: the chunks run BLOCKING** (`GEN3AI_BLOCKING="stage 1c
  (P0.1)"`, bypassing the heavy-slot queue, then 17 deep). Stage 1c is the owner's P0.1 study, and most of the queue
  was lower-priority work. It is still `nice -n 19`, `taskset -c 5-7,13-15`, SCHED_IDLE, under `mem_cap`, ONE chunk at a
  time, with the 45-min chunk shape and the registered stop. It changes throughput, not the design: the done states are
  still a prefix of the registered order. The waiting loop was stopped between chunks (no state in flight) and
  relaunched with the mark at 08:36.
- **10:24: the chunks' scheduling class, SCHED_IDLE → nice 19 (SCHED_OTHER).** `loop_1c.sh` wraps each chunk in
  `chrt --idle` (stage 1's habit), which yields even to other agents' nice-19 jobs: wall time ran 2.5–3× CPU time
  (≈ 4–6 min a state). The ruling asks for `nice -n 19` on the agent cores, still far below the trainer (nice 10). The
  loop's bash was stopped between states (its chunk left to finish, still SCHED_IDLE: an unprivileged process cannot
  leave it), and an operational wrapper, `~/gen3ai_archive/search_1c/ops/loop_a_nice19.sh` (the same loop and recheck,
  the same driver checkout's `run_1c.py`, so every row's head stays `a144482b0`), runs the next chunks at nice 19.
- **THE ORCHESTRATOR'S RULING, 2026-10-20 ≈ 16:55 PDT: no longer BLOCKING.** Per A4 and the decoy result
  (`3d14e51d`), nothing stage 1c finds can steer until the off-pool replication, so it is not the GPU's next work. The
  remaining chunks run in the normal FIFO heavy slots (nice 19, cores 5-7,13-15, ≤ 45-min chunks); a later end is
  accepted. The wrapper's bash was stopped between chunks' starts (the chunk in flight left to finish), and
  `ops/loop_a_fifo.sh` (the same loop, without the mark) continues after it. The read is unchanged.
- **Phase B, 21:00:** launched through `ops/loop_b_fifo.sh` (loop_1c.sh's play branch at nice 19, normal FIFO slots;
  `STOP_B` = 2026-10-21 21:00 PDT). The identity tooth passed 12 / 12 (21:45). **The first play chunk crashed before
  any play (a DEFECT of this unit's script, a FINDING):** `play_1c.py` runs at the pin checkout, whose `main` predates
  `main.value_search.opp_likelihood`, so `c1c_common.play_indices` could not import. The wrapper was stopped while it
  waited for a slot (no play row existed). FIX: `play_idx_1c.py` computes the same registered indices at the DRIVER
  checkout into `play/play_idx.json`, and `play_1c.py` reads them. `read_1c.py b` recomputes every played index with
  `play_indices` and stops on a mismatch, so the check is end to end. The registered stop is unchanged.

## 4. THE INTERIM READ (registered §7: about 12 h after launch, descriptive; NO verdict, NO bar; it changes no rule, n or stop)

Made 2026-10-20 16:42 PDT on the done prefix: **236 of 287 rows, 234 read** (2 exclusions, 0 defects, 0 parked).
[`result_1c_a_interim.json`](result_1c_a_interim.json), by `read_1c.py a --interim` from the driver checkout. Gates:
G1 (a prefix) and G2 pass; G5 recomputes all 1,872 stored resamples exactly; G3 (the recheck) runs after phase A.
**G4 (descriptive): 0 of 232 truth likelihoods equal L3's exactly, and all differ negligibly** (median |Δ| 4e-6 nats,
max 6e-5; the same scored steps). That is the expected trace of the fp32_alt τ solve the driver commit inherited (config
v160, `d4af9f1e2`; X26's pre-v160 checkpoint migrates to it). The worlds' own log-likelihoods differ by ≈ 1 nat.

Pooled, the amended reading (corpus arms STRICT), Δ against F8 (95 % CI, n = 234):

| arm | ΔH (binding facts held) | ΔM_hit (unrevealed moves) | ΔU_hit (never-seen species) | q_SM (exact-world mass) |
|---|---|---|---|---|
| F128L | +0.007 [−0.000, +0.015] | +0.010 [−0.000, +0.019] | +0.007 [−0.008, +0.023] | 0.0000 |
| C128L | +0.018 [+0.010, +0.027] | +0.003 [−0.007, +0.013] | +0.100 [+0.075, +0.125] | 0.0001 |
| M256L | +0.013 [+0.006, +0.020] | +0.007 [−0.002, +0.016] | +0.051 [+0.032, +0.070] | 0.0000 |

- **The corpus proposals carry the species, not the likelihood:** C8 − F8 alone (8 uniform corpus worlds) is ΔH
  +0.019 [+0.012, +0.025].
- **The likelihood's own selection is small:**
  - F128L − F128d +0.007 [+0.001, +0.013];
  - C128L − C128d +0.004 [−0.002, +0.010].
- **The exact-world mass stays ≈ 0 in every arm** (≤ 0.0003). The every-move-exact share rises 0.012 (F8) → 0.018
  (F128L).
- **The ESS of the likelihood-weighted 128** is ≈ 18–19 (9–13 % of states below 2).
- **The N-scaling (128 − 64) is ≈ 0.**
- **By stratum, every lift is in the SNAPSHOT stratum:**

  | | snapshot | bot |
  |---|---|---|
  | F128L | +0.019 | −0.005 |
  | C128L | +0.035 | +0.002 |
  | the likelihood's selection, F | +0.014 [+0.008, +0.020] | −0.000 |
- **The leak (A1) is small:**
  - 10 % of states hold a near4 C proposal;
  - L puts 0.0014 of its weight on them;
  - C128L's ΔH without those states is +0.017.
- **The A3 placebo** reads not NEARER on every arm (F128L's +0.004 [−0.001, +0.009] is the largest).

Read as an interim only: the registered verdict and the bar are made once, at `A_DONE` or the stop, after G3.

## 5. PHASE A: THE REGISTERED READ (2026-10-20 21:00 PDT; `read_1c.py a`, once, after `A_DONE` and the recheck)

[`result_1c_a.json`](result_1c_a.json).

**Gates: all pass.**
- G1: 287 rows, the registered order. G2: 0 defects, 0 parked.
- Exclusions: 2 states, in both sets. One is a 128-world build with one world never passing the joint gate in 60
  tries (hazard 4). The other is a seen Tyranitar with no honest screen fit.
- G3: 6 / 6 fresh-process rechecks identical. G5: 2,280 / 2,280 stored resamples recomputed exactly.
- G4 (descriptive): 1 / 282 truth likelihoods equal L3's exactly; the rest differ by at most 6e-5 nats (median 4e-6;
  the same scored steps), from the fp32_alt τ solve the driver commit inherited.
- **285 states read** (bot 143, snapshot 142).

**THE VERDICT (§5 and AMENDMENT A): all three primaries NEARER; none MATERIAL, none DOSE; each PROVISIONAL-POOL and
SELF-PLAY-ONLY** (bot NOT-DETECTED, snapshot NEARER). Pooled, against F8 (the corpus arms STRICT):

| arm | ΔH (binding facts held) | ΔM_hit | ΔU_hit | q_SM (exact-world mass) | bot ΔH | snapshot ΔH |
|---|---|---|---|---|---|---|
| **F128L** | **+0.007** [+0.001, +0.014] NEARER | +0.011 [+0.002, +0.019] | +0.004 | 0.0000 | −0.006 | +0.021 |
| **C128L** | **+0.019** [+0.012, +0.027] NEARER | +0.006 [−0.003, +0.015] | +0.095 [+0.072, +0.118] | 0.0001 | −0.000 | +0.038 |
| **M256L** | **+0.013** [+0.007, +0.019] NEARER | +0.008 [+0.000, +0.016] | +0.048 [+0.031, +0.066] | 0.0000 | −0.003 | +0.030 |

- **Non-strict beside STRICT:** C128L +0.019, M256L +0.013, both NEARER; their q_SM is 0.0003 / 0.0001.
- **The 64-world arms:** F64L NOT-DETECTED (+0.006); C64L NEARER (+0.020); M128L NEARER (+0.014).
- **THE ATTRIBUTION CO-PRIMARIES (A2):**
  - F128L − F128d: +0.008 [+0.002, +0.013], NEARER, SELF-PLAY-ONLY (bot −0.000, snapshot +0.016);
  - C128L − C128d: +0.005 [−0.000, +0.011], NOT-DETECTED (bot −0.008, snapshot +0.018).
- **A3, the permuted-L placebos:** NULL on all three (F128L +0.003 [−0.002, +0.007]).
- **THE BAR IS MET.** Phase B plays F8, F128L, C128L, M256L, C8, F128d and C128d.

**What the worlds hold** (the levels; descriptive):

| | F8 | F128L | C8 | C128L | M256L |
|---|---|---|---|---|---|
| H (share of the binding facts) | 0.442 | 0.450 | 0.459 | 0.461 | 0.455 |
| M_hit (seen mons' unrevealed moves) | 0.506 | 0.516 | 0.507 | 0.511 | 0.514 |
| U_hit (never-seen species) | 0.151 | 0.155 | 0.246 | 0.246 | 0.199 |
| M_exact (every seen mon's set exact) | 0.011 | 0.016 | 0.013 | 0.012 | 0.015 |
| U_exact (the never-seen species set) | 0.017 | 0.017 | 0.064 | 0.065 | 0.041 |
| q_SM (exact on both) | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |

- **The proposals and the likelihood, separately:**
  - C8 − F8 = +0.017 [+0.011, +0.023] on H (+0.095 on U_hit): the corpus brings the species by itself.
  - F128d − F8 = −0.000: more proposals alone change nothing.
  - The likelihood's own selection on the F proposals: +0.008 on H (+0.008 on M_hit, +0.005 [+0.000, +0.012] on
    M_exact).
- **The N-scaling (128 − 64) is ≈ 0 on every arm** (F +0.002 [−0.002, +0.006]). The ESS of the likelihood-weighted
  128 is ≈ 18–20 (≈ 107 for the dice alone). 10–13 % of states fall below 2.
- **NW_lift** (extra weight on the nearest proposals): C128L +0.012, F128L +0.003.
- **The tempered likelihood (L^0.5):** about the same (F128L +0.005, C128L +0.018).
- **A1's leak report:** 11 % of states hold a near4 corpus proposal, and L puts 0.0018 of its weight on them.
  C128L's ΔH without those states is +0.017. The honest base rate (F128d's near4 share) is 0.001.
- **On the BOT stratum the likelihood works AGAINST the corpus.** C8 − F8 is +0.014 there, but C128L − C128d is −0.008:
  the network, seated as a scripted bot, picks the wrong teams.

**Reading (descriptive):**
- The opponent's own choices do select nearer worlds, but only where the opponent IS our network (the snapshot
  stratum). Even there the selection is small: about 0.8 points more of the binding facts held over 128 honest
  proposals, and the every-move-exact share 0.011 → 0.016.
- Against the dose curve's target (an exact world at weight ≈ 0.2), the exact-world mass stays ≈ 0 in every arm, more
  than 100× short.
- The team-level corpus moves the never-seen species a lot (exact set 0.017 → 0.065), and the likelihood adds little to
  it. Doubling the proposals to 128 adds nothing. There is still nothing near the truth for the likelihood to pick.
- Phase B tests whether these slightly nearer worlds pay in playouts. The dose curve and the power (±0.02) predict NOT
  DETECTED.

## 6. PHASE B: THE REGISTERED READ (2026-10-21 10:01 PDT; `read_1c.py b`, once, after `B_DONE`)

[`result_1c_b.json`](result_1c_b.json).

**Gates: all pass.**
- The identity tooth: 12 / 12.
- 287 states × 7 arms played. 0 refused worlds; the 2 phase-A exclusions are excluded in every arm.
- Every played world is the registered one: `read_b` recomputes each played index with `play_indices` (0
  mismatches).
- **285 states read** in every arm.

**THE VERDICT (§6 and AMENDMENT A): every played arm is NOT-DETECTED, and Holm rejects none.** "Honest search pays" is
NOT met. The anchored rule is piKL at L1's fitted λ (0.1 / 0.015). IMP is the gain over the policy's own top choice on
the true world's referee:

| arm | LEAF-K8 | TRUNC-4-K8 | − F8 at LEAF-K8 | Holm p |
|---|---|---|---|---|
| F8 (baseline) | +0.001 [−0.009, +0.012] | −0.002 [−0.015, +0.010] | — | — |
| **F128L** | −0.001 [−0.011, +0.008] | −0.001 [−0.013, +0.010] | −0.003 [−0.014, +0.010] | 0.63 |
| **C128L** (STRICT) | +0.007 [−0.003, +0.017] | +0.002 [−0.010, +0.015] | +0.006 [−0.006, +0.018] | 0.37 |
| **M256L** | −0.002 [−0.015, +0.010] | −0.005 [−0.016, +0.008] | −0.004 [−0.019, +0.011] | 0.80 |
| F128d | −0.002 [−0.014, +0.009] | −0.002 [−0.015, +0.011] | −0.004 | 0.68 |
| C128d (STRICT) | +0.001 [−0.011, +0.013] | −0.004 [−0.015, +0.008] | −0.000 | 0.74 |
| C8 (STRICT) | −0.001 [−0.014, +0.013] | +0.007 [−0.005, +0.019] | −0.002 | — |

- **What NOT-DETECTED means (A6):** no gain of about +0.03 or more (search-sized) was detected (n = 285). It neither
  reopens the route nor licenses a re-run on this bank.
- **Each stratum:** NOT-DETECTED on every arm.
- **No other acting rule pays:**
  - GUMBEL and THRESH stay within ±0.012 on every arm;
  - the plain ARGMAX sits clearly below the policy on F8 (−0.020 [−0.035, −0.006] at LEAF-K8), as L1 found, and the
    anchor removes that harm.
- **A7, the ESS split:** no subset pays. F128L's states with ESS < 2 read +0.011 [−0.001, +0.025] at LEAF-K8.

**THE UNIT (§5–§6): NEARER BUT NOT PAYING.** The likelihood-weighted proposals hold slightly more of the truth's hidden
facts (phase A, snapshot opponents only). Honest one-ply search over the resampled worlds, with the anchor, plays like
the policy: within about ±0.01 of it, and no better than today's 8 F worlds. That is what the dose curve predicts for
worlds that stay ≈ 0 % exact.

## 7. FINDINGS (hazards, deviations, and what could not be verified)

1. **Two DEFECTS of this unit's phase-B scripts, both caught before any play.**
   - `play_1c.py` runs at the pin, whose `main` predates `opp_likelihood`, so it could not compute the play indices
     (crash at the first chunk).
   - `play_indices` named a uniform arm's first 8 on the 2 states whose sets were never built.
   - Both fixed (`46ce84db9`, `a2a3624b3`): the indices are computed at the DRIVER by `play_idx_1c.py`, and `read_b`
     re-checks every one.
2. **G4:** the truth's likelihood differs from L3's by ≤ 6e-5 nats on all 282 compared states (1 exactly equal). The driver commit inherited the
   fp32_alt τ solve (`d4af9f1e2`, rebased in at the code ship) that X26's pre-v160 checkpoint migrates to. It is
   negligible against the ≈ 1-nat spread between worlds, and declared rather than re-run.
3. **A1's flag is conservative.** `near4` counts revealed sets too, so the STRICT read also zeroes rare honest
   near-exact worlds (the honest base rate is 0.001). It moved nothing: STRICT and non-strict verdicts agree.
4. **The familiarity confound (L3's review and decoy test):** the likelihood partly rewards pool-like teams. Here it
   cannot explain a gain, because the corpus-side selection is NOT-DETECTED and the F-side selection is among honest
   draws, which are not pool teams. It is also why everything is PROVISIONAL-POOL.
5. **M256L has no proper mixture weight** (hazard 2): it is read descriptively in spirit; it changed no conclusion.
6. **The run's operations** (§3) changed throughput, never the design.
   - Phase A was stopped twice between states (to register Amendment A, and for the BLOCKING rulings), relaunched with
     the registered stop each time; no state was lost or rerun.
   - Chunks moved from SCHED_IDLE to nice 19 through operational wrappers outside the repo
     (`~/gen3ai_archive/search_1c/ops/`).
   - Every phase-A row's head is `a144482b0`; phase B's scripts are at `a2a3624b3`, and only the play-index code
     changed.
7. **Light jobs outside a heavy slot:** two functional checks of the driver (N = 4 / 12, ≤ 1.2 GB, ≤ 35 s) and the
   reads (≤ 0.5 GB, seconds) ran under `nice -n 19`, SCHED_IDLE and an address-space cap.
8. **One model, one bank, one pre-plateau anchor, pool opponents.** Nothing here is read off-pool or at the plateau
   (A4). The NOT-DETECTED also holds only at this power (≈ ±0.02 between arms).
9. **The design doc's §6.3 named the draft's 64 + 64 proposals** in the code ship (`664385776`) until the phase-A read
   corrected it (`d4d9c6149`); the registration itself always said 128 + 128.
