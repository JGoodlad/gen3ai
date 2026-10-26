# The MATURITY BATTERY on X26: the interim bank and the LEAN trajectory (2026-10-18)

**Status.**
- **The harness** is T35 (`design_hidden_info_binding.md` §12.13.7; backlog X26-6). It shipped before the
  registration.
- **The registration** is [`PREREGISTRATION.md`](PREREGISTRATION.md), shipped with the bank's ids
  (`states_mat_x26.json`) before any trajectory output existed.
- **The queue** crawls detached (§3), launched 2026-10-18 04:45 PDT from `826bee89`. The read is `read_mat.py`, committed with the harness, before any output.
- **Results:** READ 2026-10-19 (§5), after `QUEUE_DONE` (2026-10-19 04:30 PDT, nothing parked). GAP-SHRINKS NOT
  DETECTED, BELIEF-RISES NOT DETECTED, NOT CONVERGED, FRONTIER NOT READ (LEAN). PROVISIONAL: the interim bank.
- **THE PLATEAU STEP** (§6, amendment A1, PREREGISTRATION §9): `QUEUE_DONE` 2026-10-23 01:07 PDT, nothing parked.
  **READ 2026-10-25 (§7), exactly as registered.** The record (the plateau bank, [25M, P\*], 719 states): GAP-SHRINKS
  NOT DETECTED, BELIEF-RISES NOT DETECTED, FRONTIER-STATIC, STILL-MOVING NOT DETECTED, CONVERGED NOT READ. Its
  replication (the interim FULL pair) reads GAP-SHRINKS. R3 at P\* is SHARED on both banks. Every IMP is a
  SELF-REFEREE read (§7.5).

**What it asks** (owner, 2026-10-17: "what does it take to enable search? Are we getting closer, not closer?"). It
re-runs §12.13.7's stage and dose reads at X26's 25M, 100M, 175M and 225M checkpoints on ONE frozen bank, and reads the
TRAJECTORY:
- does honest one-ply search's gap to the oracle shrink (GAP-SHRINKS)?
- does the network's effective belief quality rise toward the truth (BELIEF-RISES)?
- is the gap still moving between the last two anchors (CONVERGED)?

**Vocabulary.**
- IMP = decision improvement: the win probability gained per decision by playing a scorer's pick instead of the
  policy's top-1, judged on held-out referee seeds of the true world.
- O = search in the true world; F = search in fully honest worlds; FN = F with the network's own species and move
  belief.
- G_F = IMP_O − IMP_F. TODAY = FN's position on the dose curve, in "weight on the true world" units.
- LEAF-K8 = the mean outcome over 8 worlds played to the end.

Bulk lives in `~/gen3ai_archive/hidden_info_maturity/` (layout: `mat_common.py`'s docstring). This folder holds the
scripts, the committed ids, the anchors' sha256 (`checkpoints_x26.json`) and, later, the read.

## 1. Pipeline

| step | script | where (`run_mat.sh <where>`) | output |
|---|---|---|---|
| trigger + copies | `trigger_mat.py status / register / traces` (`maturity_series`: the first checkpoint in [m, m + 25M), the read-only sha256 copy) | any Python | `ckpt/`, `ckpts.json`, `traces/` |
| bank | `make_mat_bank.py` (`maturity_bank.draw`; no model) | `pin` | `bank/`, `census.npz`, `states_mat_x26.json` |
| roots | `roots_mat.py` (each anchor's top-1, V, tie) | `pin` | `roots/<anchor>.jsonl` |
| truth (O) | `phase1_mat.py` (the pilot's `run_playouts.one_state`) | `pin` | `rows/<anchor>/O/` |
| belief (FN) | `fn_belief_mat.py` (the pilot's `det_worlds.root_belief`) | `pin` | `fn_belief/<anchor>.jsonl` |
| worlds | `build_mat_worlds.py` (`WorldSampler`, the guard on; FN = Tier 1's `fn_worlds`; F re-sampled for CRN and checked) | `head` | `worlds/shared/<arm>/`, `worlds/<anchor>/FN/` |
| playouts | `play_mat_worlds.py` (the pilot's `run_det_playouts.one_state`) | `pin` | `rows/<anchor>/<arm>/` |
| runner | `queue_mat.py` (launched by `launch_queue.sh`) | the HEAD checkout | `logs/queue_mat.log`, `QUEUE_DONE` |
| identity | `identity_mat.py` (oracle worlds at HEAD, played at the pin = Phase 1) | pure | `identity_mat.json` |
| table | `score_mat.py` (Tier 1's `t1_data` + `score_blind.dk_family`) | pure | `score/table.pkl` (+ sha256) |
| read | `read_mat.py` (+ `read_mat_test.py`, planted effects; rules in `maturity_rules.py`) | pure | `result_mat.json` |
| GIGO audit (written for the read, 2026-10-19) | `gigo_audit_mat.py` (coverage, duplicates, provenance, defects, CRN, exclusions, non-finites; counts only) | pure | `gigo_audit_mat.json` |
| post-hoc companions (after the read; descriptive) | `posthoc_mat.py` (the gap's change split into IMP_O / IMP_F, per stratum, FN − F, the dose curves and P-STAR) | pure | `posthoc_mat.json` |
| **A1 (the plateau step):** the plateau anchor, modes | `trigger_mat.py register --plateau`, `set-mode --labels L --bank B --mode M` | any Python | `ckpt/x26_P280M/`, `ckpts.json` `banks` |
| A1: the plateau bank | `trigger_mat.py traces --bank plateau --from …`, then `make_mat_bank.py --bank plateau` | `pin` | `plateau/bank/`, `plateau/census.npz`, `states_mat_x26_plateau.json` |
| A1: every driver | `--bank interim\|plateau` (default interim); mode `full_hide` adds O−M / O−U | as above | `plateau/…` for the plateau bank |
| A1: the read | `score_mat.py --bank <b> --tag plateau_step` (both banks), then `read_mat.py plateau` | pure | `<bank>/score/plateau_step.pkl`, `result_mat_plateau.json` |
| A1: the GIGO audit (§9.7; written for the read, 2026-10-25) | `gigo_audit_mat_plateau.py` (both banks: coverage, provenance + driver / bank stamps, defects, CRN, the k = 6 identities, the flags, a row sha256 manifest; counts only) | pure | `gigo_audit_mat_plateau.json` |
| A1: the tests | `read_mat_test.py` (7), `queue_mat_test.py` (5): planted, no archive | pure | — |

**CPU (root rule 5b):**
- every worker runs one thread, under `nice -n 19` and SCHED_IDLE;
- workers are pinned to `taskset -c 5-7,13-15`, each in its own `mem_cap` scope;
- `queue_mat.py --slots 2` keeps at most 2 workers.

No GPU: `CUDA_VISIBLE_DEVICES=""`. The run directory is only ever read, and only to copy.

## 2. Reproduce / operate

```bash
F=designs/research_state/measurements/hidden_info_maturity_x26_2026-10-18
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $F/trigger_mat.py status                               # the run's 25M anchors: registered / pending
$PY -I $F/trigger_mat.py register --pending --mode lean       # THE TRIGGER: copy + register every new anchor
bash $F/launch_queue.sh                                       # the detached runner (from the HEAD checkout)
$PY -I $F/queue_mat.py --status                               # PROGRESS: per anchor and unit, done / wanted
tail -f ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log
$PY -I $F/identity_mat.py --out identity_mat.json             # after QUEUE_DONE: the identity per anchor
$PY -I $F/read_mat_test.py && $PY -I $F/score_mat.py && $PY -I $F/read_mat.py   # the registered read
```

## 3. The queue (DONE 2026-10-19 04:30 PDT, nothing parked)

**LAUNCHED 2026-10-18 04:45 PDT** (runner pid in `~/gen3ai_archive/hidden_info_maturity/queue.pid`). It is detached
(`setsid nohup`), so it survives the launching session.

| | |
|---|---|
| runner | `queue_mat.py --slots 2`, launched by `launch_queue.sh` from the HEAD checkout |
| HEAD checkout (the world builder; §12.13.7 (f)) | `~/.cache/gen3ai/hib_mat/head`, detached at `826bee89` (the registration commit). **Keep it** until the read |
| pin checkout (bank, roots, truth, belief, playouts) | `~/.cache/gen3ai/hib_mat/pin_x26`, detached at `9c033489` (X26's pin), with its own Rust release builds. **Keep it** until the read |
| archive | `~/gen3ai_archive/hidden_info_maturity/` (checkpoint copies, copied traces, bank, every row) |
| progress (counts only, no outcome) | `python -I <this folder>/queue_mat.py --status` (done / wanted per anchor and unit); `tail ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log` |
| done marker | `~/gen3ai_archive/hidden_info_maturity/QUEUE_DONE` (it lists any PARKED unit: a FINDING) |
| order | the shared F worlds, then 25M, 225M, 100M, 175M (roots, identity, truth, belief, F, FN worlds, FN) |
| cost | ≈ 30 CPU-h for the four LEAN anchors, at the pilot's measured 77–92 CPU-s per state (PREREGISTRATION §5) |
| **ETA** | **≈ 15–20 wall-hours on the 2 slots, i.e. ≈ 2026-10-18 20:00 to 2026-10-19 01:00 PDT.** It runs at SCHED_IDLE beside the live trainer, so it stretches whenever the trainer takes the cores (budget to 2026-10-19 midday). The first-vs-last pair (25M, 225M) completes first, about 60 % of the way in |

- **Restart after a crash or a reboot:** `bash <this folder>/launch_queue.sh` from the HEAD checkout. Every unit is
  resumable, and `--limit` caps only NEW work.
- **A new anchor** (the trigger): `trigger_mat.py register --pending --mode lean`, then relaunch if `QUEUE_DONE` has
  appeared.
- **The read** (once `QUEUE_DONE` exists): §2's last three lines. Their outputs, `identity_mat.json` and
  `result_mat.json`, land in this folder.

## 4. Notes fixed before any output

- **Not blinded.** The arms carry their labels. Every rule is fixed as code before any output exists, and the runner
  reads no outcome (E2's reasoning: Tier 1's blinding protected the frozen table and scripts, not the author's
  knowledge).
- **One pin for the whole series.** Every X26 checkpoint was saved at `9c033489`. The observation encoding, the engine
  and the critic's interface are therefore the same at every anchor, and only the weights differ.
- **F is network-free.** It is built once and played at every anchor, so the anchors' F rows differ only by the network.
  FN is built per anchor from that anchor's own belief.
- **The pilot** ran on 12 eligible states outside the bank (`<archive>/pilot/`). It is engineering only: GIGO and
  cost, never read as trajectory output.

## 5. RESULTS: the first LEAN trajectory, READ 2026-10-19 (PROVISIONAL: the interim bank)

**How it was read.** Exactly as registered, after `QUEUE_DONE` (2026-10-19 04:30 PDT, `parked = []`), in this order:
1. `read_mat_test.py`, the planted-effect self-test: 3 / 3 pass (a planted shrink and belief rise are found; a null
   series reads nothing; LEAN never reads a frontier).
2. `gigo_audit_mat.py` (written for this read, counts and provenance only; `gigo_audit_mat.json`): 200 / 200 checks pass.
3. `identity_mat.py` (`identity_mat.json`): PASS at every anchor.
4. `score_mat.py` (the frozen table, `<archive>/score/table.pkl`, sha256 `43db5719…46ddee`), then `read_mat.py`
   (`result_mat.json`).

No rule, threshold, arm or seed was changed, and no arm was re-run. `posthoc_mat.py` (`posthoc_mat.json`) was written
AFTER the read. It holds DESCRIPTIVE companions on the same table and the same bootstrap, and it carries no verdict.

### 5.1 The GIGO audit (first)

| check | result |
|---|---|
| the bank | LEAN 320 / FULL 800 ids hashes = PREREGISTRATION §6; the bank's content sha256 `8da1613d…` recomputed on load; every LEAN id a bank decision |
| the anchors | each copy's sha256 = `ckpts.json` = this folder's `checkpoints_x26.json` = the run's own file (read only); all four LEAN |
| coverage | every unit holds EXACTLY one row per wanted id: roots 800 per anchor; belief, truth (O), F and FN 320 per anchor; the identity's 12 per anchor; F worlds 320 (once), FN worlds 320 per anchor. **0 duplicates, 0 foreign ids, 0 missing, 0 torn lines** |
| the `--limit`-before-done class (E2's) | every driver caps NEW work after dropping done ids (`maturity_series.select_new_work`, or the builder's own after-done slice); every row file is in sorted-id order with no id twice, so no id was re-picked or skipped |
| provenance | every world row built at the registration commit `826bee89`; every root, belief, truth and playout row at the pin `9c033489`; every playout row's `cp`, `step`, `arm` and `world_head` its own |
| defects | **0** in every world arm and every playout (the guard's included) |
| common random numbers | the F re-sample reproduced the stored F worlds on every FN build (0 "CRN broken" rows). The tooth was live: the shared F build finished (06:00) before the first FN build (07:40) |
| honest-world exclusions | F: **33 / 320 = 10.3 %** (`WorldError`: no honest draw fits what our side saw), below the early batches' 12.5 %; FN: 32 / 320 at each anchor. **The same 287 F states play at every anchor** (F is network-free and built once) |
| the identity (oracle worlds built at HEAD and played at the pin = Phase 1) | 12 / 12 states, 87 actions, 0 outcome / trace / critic mismatches, max \|ΔV\| 1e-5, at each of the four anchors |
| root ties (rule 8's exclusion) | 0 at every anchor |
| non-finite values | 0 in every row |

**Common states:** 287 of 320 (all 33 drops are F exclusions), in 248 battles; 286 also hold FN at every anchor (the
TODAY read). Common-ids sha256 `cf8d8c6f…`.

### 5.2 The registered read (`result_mat.json`; LEAF-K8; paired; 2,000 battle-clustered draws, seed 20261011)

| anchor | G_F = IMP_O − IMP_F | IMP_O (oracle) | IMP_F (honest) | IMP_FN (network belief) | TODAY (p on the true world) |
|---|---|---|---|---|---|
| 25M | **+0.0747** [+0.0547, +0.0959] | +0.063 [+0.044, +0.082] | −0.012 [−0.031, +0.005] | −0.018 [−0.038, +0.002] | 0.000 [0, 0.039] |
| 100M | **+0.0618** [+0.0418, +0.0813] | +0.040 [+0.023, +0.058] | −0.022 [−0.039, −0.005] | −0.017 [−0.032, −0.002] | 0.011 [0, 0.107] |
| 175M | **+0.0549** [+0.0332, +0.0761] | +0.051 [+0.032, +0.072] | −0.004 [−0.021, +0.014] | −0.006 [−0.024, +0.011] | 0.000 [0, 0.045] |
| 225M | **+0.0528** [+0.0335, +0.0710] | +0.042 [+0.023, +0.061] | −0.011 [−0.026, +0.003] | −0.007 [−0.023, +0.009] | 0.011 [0, 0.181] |

| rule (R-MATURITY, §12.13.4) | the contrast | verdict |
|---|---|---|
| **GAP-SHRINKS** (all states, PRIMARY) | G_F(225M) − G_F(25M) = **−0.0219 [−0.0494, +0.0045]** | **NOT DETECTED** (the upper bound is not below −0.0005) |
| GAP per k bin (k = opponent Pokémon seen) 1–2 / 3 / 4 / 5 / 6 | −0.062 [−0.130, +0.004] / −0.020 [−0.088, +0.047] / −0.013 [−0.052, +0.027] / +0.006 [−0.052, +0.067] / −0.017 [−0.060, +0.027] | NOT DETECTED in every bin |
| GAP per turn bucket ≤ 7 / 8–14 / 15–21 / ≥ 22 | −0.019 / −0.022 / +0.001 / **−0.050 [−0.102, −0.002]** | **GAP-SHRINKS at turn ≥ 22 only**; NOT DETECTED elsewhere |
| GAP per material bucket (mons alive) 12 / 10–11 / 8–9 / ≤ 7 | −0.005 / −0.043 / −0.041 / +0.006 | NOT DETECTED in every bucket |
| **BELIEF-RISES** | TODAY(225M) − TODAY(25M) = **+0.011 [−0.034, +0.178]** | **NOT DETECTED** |
| **CONVERGED** (175M → 225M) | −0.0022 [−0.0236, +0.0190] | **NOT CONVERGED** (the interval is not inside ±0.01; see FINDING 2) |
| **FRONTIER** | — | **NOT READ** (LEAN: every bucket holds < 150 states), as registered |
| the per-bucket GATE (descriptive in LEAN) | all 13 buckets at all 4 anchors | **GATE-CLOSED everywhere**: no bucket is even GATE-WEAK |

**Per opponent type (stratum), descriptive as registered.** G_F at 25M / 225M: snapshot (competent play) +0.078 / +0.055,
bot +0.071 / +0.050. Change: snapshot −0.023 [−0.064, +0.017], bot −0.021 [−0.057, +0.016]; their difference −0.001
[−0.055, +0.052]. The two populations move together.

**The t₀ references** (different states and networks, unpaired): Tier 1's 15M G +0.070 [+0.060, +0.081] (end2 + prod,
pooled) and E2's +0.072 [+0.060, +0.083]. X26's 25M reads +0.075, the same size.

### 5.3 Descriptive companions (POST HOC, `posthoc_mat.py`; no verdict)

- **Where the gap's change comes from.** G_F = IMP_O − IMP_F, so its change splits into the two:
  - 25M → 225M at LEAF-K8: ΔIMP_O **−0.021 [−0.046, +0.006]**, ΔIMP_F **+0.001 [−0.023, +0.026]**.
  - So the point decline of the gap is the ORACLE's edge over the policy falling, not honest search rising. The
    policy's own pick moves closer in true value to the oracle's pick. Honest one-ply search stays at or below the
    policy at every anchor (IMP_F −0.012 / −0.022 / −0.004 / −0.011).
  - **On the continuous 4-turn leaf (TRUNC-4-K8) the gap does not move at all:** ΔG_F +0.005 [−0.020, +0.032]. The
    LEAF-K8 decline is scorer-dependent.
- **The network belief's value over Smogon usage (IMP_FN − IMP_F)**, LEAF-K8: −0.006 / +0.005 / −0.002 / +0.004, none
  clear. At TRUNC-4-K8: +0.003 / +0.006 / +0.005 / **+0.018 [−0.0002, +0.038]** at 225M, a boundary, not clear.
- **The belief-dose curve and p\*** (M8: the true world at weight p, F's worlds at 1 − p; on the 286 FN-common states).
  P-STAR here is the R-DOSE definition against the POLICY only: the smallest 0.05-grid p whose lower bound clears 0 at p
  and at every larger p.

  | anchor | LEAF-K8: IMP at p = 0 / 0.2 / 0.35 / 1 | P-STAR (LEAF-K8) | TODAY (LEAF-K8) | P-STAR (TRUNC-4-K8) | TODAY on the TRUNC-4-K8 curve (not registered) |
  |---|---|---|---|---|---|
  | 25M | −0.012 / +0.011 / +0.035 / +0.063 | 0.25 | 0.00 [0, 0.04] | 0.35 | 0.02 [0, 0.26] |
  | 100M | −0.022 / +0.009 / +0.025 / +0.040 | 0.30 | 0.01 [0, 0.11] | 0.35 | 0.10 [0, 0.30] |
  | 175M | −0.004 / +0.026 / +0.046 / +0.051 | 0.20 | 0.00 [0, 0.05] | 0.30 | 0.08 [0, 0.21] |
  | 225M | −0.011 / +0.014 / +0.028 / +0.042 | 0.30 | 0.01 [0, 0.18] | 0.50 | 0.18 [0, 0.36] |

  - The p 0 → 0.05 step is again Tier 1's tie-breaking step (the 8-rollout leaf ties).
  - At LEAF-K8, today's belief sits at ≈ 0.01, below that step, at every anchor. p\* stays at 0.2–0.3: a gap of
    ≈ 0.2–0.3 in "weight on the true world" units that does not close.
  - On the continuous leaf, TODAY's point drifts up (0.02 → 0.18) while p\* drifts out (0.35 → 0.50), because the
    oracle's ceiling falls. The intervals are wide (upper bounds 0.21–0.36) and nothing is registered on this curve.

### 5.4 What it says ("are we getting closer?")

- **Registered answer: UNDECIDED.** The gap's point estimate falls monotonically, +0.075 → +0.062 → +0.055 → +0.053
  (−29 %), but the 25M → 225M change is not clear of zero: −0.022 [−0.049, +0.004]. The belief does not rise
  detectably, and nothing has converged.
- **Descriptively, "closer" is the wrong frame for what moved.** The policy became stronger relative to the oracle
  (IMP_O fell by about a third). Honest one-ply search did NOT become more useful: it is at or below the policy at every
  anchor, and the per-bucket gate is closed everywhere. On the continuous leaf the oracle-honest gap is flat. The prize
  that perfect information would buy is shrinking slowly. What honest search captures of it is still ≈ none.
- **What enables search is unchanged by maturity so far:** a world sampler whose belief puts ≈ 0.2–0.3 of weight on the
  true world (at the 8-rollout leaf). The network's belief is worth ≈ 0.01 of that at every anchor. Training the policy
  for another 200M steps did not move it there.
- **PROVISIONAL.** The interim bank (PREREGISTRATION §2.5). The plateau bank, when scored, is the read of record, and a
  direction that differs between the two is quoted BANK-DEPENDENT.

### 5.5 FINDINGS

1. **The one bucket verdict is a multiplicity hazard.** GAP-SHRINKS at turn ≥ 22 (−0.050 [−0.102, −0.002]) is the only
   clear cell among 13 bucket reads at 95 %, where chance alone gives ≈ 0.65 false positives. The registration has no
   multiplicity correction on the bucket GAP reads (the gate has its Bonferroni level, the GAP does not). It is quoted
   as read and should steer nothing alone.
2. **CONVERGED cannot be reached at the registered sizes.** The rule needs the change's whole 95 % interval inside
   ±0.01. On the 287 common states the measured half-widths are ±0.021 between adjacent anchors (175M → 225M) and
   ±0.027 between the first and the last (the registration's LEAN precision claim, now VERIFIED for X26). FULL's
   claimed ±0.017 is also above 0.01. An interval inside ±0.01 needs ≈ 1,450 common states (≈ 1,600 bank states at
   10 % exclusion): the interim and the plateau banks together, which the registration reads separately. As
   registered, CONVERGED is unreachable in either mode. That is a design defect for the orchestrator, not a result.
3. **The LEAF-K8 gap decline is scorer-dependent** (§5.3: flat at TRUNC-4-K8). R-MATURITY is registered on LEAF-K8
   only, so this changes no verdict.
4. **Mixing modes shrinks the common set.** `read_mat.py` reads the states valid at EVERY anchor named. A read over a
   LEAN anchor and a FULL anchor therefore runs on the 320 LEAN states. A FULL-precision trajectory needs every anchor
   in it to be FULL, or an explicit `--cp` list of FULL anchors.
5. **R3's hide-only half (O−M / O−U) is still not in the battery's arms** (PREREGISTRATION §4, as registered). The
   competent-opponent states are in the bank (144 of the 287 common states).
6. No hazard hit: CPU only (`CUDA_VISIBLE_DEVICES=""`, the 3-core pin, `nice -n 19`, `mem_cap`), nothing written under
   `models/` or `data/`, both checkouts kept.

### 5.6 The recommended plateau step (DECIDED 2026-10-19: steps 1, 2, 3 and 5 run, 4 deferred, 6 declined; §6)

**The plateau checkpoint P\*** = `rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…`). It is
at the series' pin `9c033489` (its sidecar `git_hash`; `pin_history` covers 0 → 320M). Its three newest eval cycles,
`step_276000077`, `step_278000056` and `step_280000170`, are all recorded at the pin.

**Costs,** from this run's MEASURED per-state CPU seconds (the rows' own `cost`; one thread; contended box):
- O ≈ 49–56 s;
- one 8-world arm played ≈ 15 s;
- an F world build ≈ 14 s, assumed the same for F+M, F+U, O−M and O−U (UNVERIFIED);
- a FULL anchor (800 states; O, F, FN, F+M, F+U) ≈ **25 CPU-h**, plus the shared worlds once per bank (≈ 9 CPU-h);
- a LEAN anchor ≈ 7.5 CPU-h.

| step | why | CPU-h |
|---|---|---|
| **1. P\* FULL on the interim bank** | the plateau point; first-vs-plateau on 800 states | ≈ 25 |
| **2. 25M upgraded to FULL on the interim bank** (480 new states × O / F / FN, F+M / F+U on all 800, the shared worlds) | FULL at the first and last anchors makes FRONTIER readable (§12.13.7 (a) asks FULL there) | ≈ 26 |
| **3. The PLATEAU BANK** (PREREGISTRATION §2.5: P\*'s three cycles, salt `hib_maturity_x26_plateau_v1:`, the interim battles excluded; it SUPPLEMENTS the interim bank and is the verdict of record): drawn, then FULL at 25M and P\*, its shared worlds | the read of record | ≈ 59 |
| 4. The plateau bank, LEAN back-fill at 100M and 175M | registered (§2.5: both banks at every anchor) | ≈ 15 |
| 5. **Hide-only O−M / O−U** at P\* (and 25M) on both banks | carries R3 to the plateau with BOTH halves on the same states, as E2 asked (§16.12). Not in §12.13.7 (e): an AMENDMENT registered before its outputs | ≈ 26 at P\* only; ≈ 39 with 25M |
| 6. (optional) 225M upgraded to FULL on both banks | only a middle point: CONVERGED is unreachable at these sizes (FINDING 2) | ≈ 43 |

- **Recommended: steps 1–5, ≈ 150–165 CPU-h** (≈ 75–80 wall-hours on 2 slots at SCHED_IDLE, i.e. 3–4 days beside a live run).
  Order: 1 → 2 (a FULL first-vs-plateau read on the interim bank inside ≈ 1 day), then 3, 5, 4. Step 6 only if
  CONVERGED is amended to read pooled over both banks (≈ 1,430 common states would make ±0.01 just reachable).
- **The YES on the hide-only arms** is cheap (≈ 15 % of the step), and without them R3's plateau replication has only
  its reveal-only half (F+M vs F+U).
- **A small build comes first (≈ a few agent-hours):**
  - the harness registers anchors only at 25M multiples (`trigger_mat.py register --labels`), so P\* needs a named
    plateau anchor;
  - the bank draw is hardwired to the interim cycles and salt (`mat_common.DRAW_CYCLES`, `STATES_SALT`);
  - the archive holds ONE bank (`bank/`, `states_mat_x26.json`), so the plateau bank needs its own namespace;
  - O−M / O−U need adding to `ARMS` / `MODE_ARMS` and the reader.
  Each is an AMENDMENT to PREREGISTRATION, registered before its outputs exist.
- **Copy P\*'s three cycles into the archive now** (read-only, with the per-file sha256 that `trigger_mat.copy_traces`
  makes; its CLI copies only the interim cycles today): the plateau bank depends on them surviving the run's end and
  any retention pass.

## 6. THE PLATEAU STEP (amendment A1, registered 2026-10-19; DONE 2026-10-23, READ §7)

### 6.1 What was decided and registered

The orchestrator decided on 2026-10-19:
- **Run** §5.6's steps 1, 2, 3 and 5:
  - P\* FULL on the interim bank;
  - 25M upgraded to FULL;
  - the plateau bank, FULL at 25M and P\*;
  - the hide-only O−M / O−U arms at P\* on both banks.
- **Step 4** (the plateau bank at 100M / 175M) only if the verdict of record strictly needs it. It does not: the
  verdicts read the first anchor against the last, and the one-bank CONVERGED is not read. So step 4 is **DEFERRED**
  (PREREGISTRATION §9.4).
- **No 225M FULL; no hide-only at 25M.**

The registration is PREREGISTRATION §9 (A1). It was committed with the plateau bank's ids before any plateau-step output
existed. In short:

| | |
|---|---|
| P\* | `x26_P280M` = `checkpoint_280000170`, sha256 `814823c5…` (a named anchor, sha-checked; `maturity_series.plateau_anchor`) |
| the PLATEAU bank | P\*'s cycles 276M / 278M / 280M (3,173 copied files, each = the run's own); salt `hib_maturity_x26_plateau_v1:`; the interim battles excluded; **800 states (LEAN 320), 500 battles**, 160 per k bin, 400 / 400 snapshot / bot; ids sha256 `728dcdd4…`, LEAN `4b63651d…`, content `25d2e83c…`; gate ① 634 / 634, 0 failures; no extension |
| modes | interim: 25M FULL, P\* FULL + O−M / O−U (100M–225M LEAN, done). Plateau: 25M FULL, P\* FULL + O−M / O−U |
| the builder | still `826bee89` (refused otherwise). The drivers come from a new checkout (§6.3), and every new row carries `driver` and `bank` |
| CONVERGED (FINDING 2) | the one-bank CONVERGED is NOT READ. It is read only pooled over both banks when both hold P\* and 225M (not this step). STILL-MOVING (a detection on the last pair) replaces it |
| FRONTIER (FINDING 4) | read only when every anchor named is FULL |
| bucket GAP (FINDING 1) | also at the Bonferroni 99.6 %; only a bucket verdict that holds there is a finding |
| R3 at P\* (FINDING 5) | E2's rule (`maturity_rules.r3_read`, tested equal to `read_e2.read_frame`), on each bank; all states PRIMARY, each stratum secondary |
| the record | the plateau bank's [25M, P\*] FULL pair. The interim FULL pair is its replication (BANK-DEPENDENT on opposite directions), and the interim LEAN 5-anchor trajectory is quoted beside it |

### 6.2 Operate

```bash
F=designs/research_state/measurements/hidden_info_maturity_x26_2026-10-18
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
$PY -I $F/queue_mat.py --status                     # PROGRESS: per bank / anchor / unit, done / wanted (no outcome)
tail -f ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log
bash $F/launch_queue.sh                             # (re)start, from the DRIVER checkout (§6.3); every unit resumes
# the read, once QUEUE_DONE exists (PREREGISTRATION §9.6):
$PY -I $F/read_mat_test.py && $PY -I $F/queue_mat_test.py
$PY -I $F/identity_mat.py --out identity_mat_plateau.json
$PY -I $F/score_mat.py --bank interim --tag plateau_step && $PY -I $F/score_mat.py --bank plateau --tag plateau_step
$PY -I $F/read_mat.py plateau                       # -> result_mat_plateau.json
```


### 6.3 The queue (LAUNCHED 2026-10-19 05:20 PDT; DONE 2026-10-23 01:07 PDT, nothing parked)

| | |
|---|---|
| runner | `queue_mat.py --slots 2` (pid in `~/gen3ai_archive/hidden_info_maturity/queue.pid`; the first was 3627258), launched by `launch_queue.sh` from the DRIVER checkout. Detached (`setsid nohup`) |
| DRIVER checkout | `~/.cache/gen3ai/hib_mat/driver_198fa91e`, detached at **`198fa91e`** (the A1 registration commit; the harness is `154dcc73`). Every new row is stamped `driver` = `198fa91e…`. **Keep it** until the read |
| BUILDER checkout | `~/.cache/gen3ai/hib_mat/head`, detached at `826bee89` (the registered builder; every world row's `head`). **Keep it** |
| pin checkout | `~/.cache/gen3ai/hib_mat/pin_x26` at `9c033489` (bank, roots, truth, belief, every playout). **Keep it** |
| archive | `~/gen3ai_archive/hidden_info_maturity/`. The interim bank is at the root, the plateau bank under `plateau/`. The LEAN run's `QUEUE_DONE` was kept as `QUEUE_DONE.20261019T052018` |
| CPU (rule 5b) | ≤ 2 single-thread workers, each `chrt --idle` + `nice -n 19` + `taskset -c 5-7,13-15` + its own `mem_cap` scope; `CUDA_VISIBLE_DEVICES=""`. **No GPU** |
| progress (counts only) | `python -I <this folder>/queue_mat.py --status`; `tail ~/gen3ai_archive/hidden_info_maturity/logs/queue_mat.log` |
| done marker | `~/gen3ai_archive/hidden_info_maturity/QUEUE_DONE`. It lists any PARKED unit, and a parked unit is a FINDING |
| order | interim shared worlds (F for the 480 new states, F+U / F+M on all 800) → P\* on interim → 25M's FULL upgrade on interim → the plateau bank's shared worlds → P\* and 25M on the plateau bank → the hide-only O−M / O−U worlds and plays at P\* (interim, then plateau) |
| cost | ≈ 150 CPU-h (PREREGISTRATION §9.8, from this battery's measured per-state costs) |
| **ETA** | **≈ 4–5 days:** `QUEUE_DONE` around **2026-10-23 to 2026-10-24**. This assumes the LEAN queue's measured ≈ 1.26 CPU-h per wall-hour on 2 SCHED_IDLE slots beside a live run (X26, then X34). The interim bank's FULL pair (steps 1–2, ≈ 55 CPU-h) completes first, around **2026-10-21**. A quieter box shortens all of it |

- **Restart after a crash or a reboot:** `bash ~/.cache/gen3ai/hib_mat/driver_198fa91e/designs/research_state/measurements/hidden_info_maturity_x26_2026-10-18/launch_queue.sh`.
  - It refuses a builder checkout that is not at `826bee89`, and a driver checkout that is not detached and clean.
  - Every unit resumes, and `--limit` caps only NEW work.
- **The read** (once `QUEUE_DONE` exists): §6.2's last block. Before it, a GIGO audit for PREREGISTRATION §9.7 is
  written. It checks coverage per (bank, anchor, unit), provenance (`head`, `driver`, `bank`), defects and the CRN
  tooth, the identity at P\*, the k = 6 identities O−U ≡ O and F+U ≡ F, and the exclusion flags.

## 7. RESULTS: THE PLATEAU STEP, READ 2026-10-25 (`result_mat_plateau.json`)

**How it was read.** Exactly as registered (PREREGISTRATION §9.6 / §9.7; README §6.2's last block), after `QUEUE_DONE`
(2026-10-23 01:07:08 PDT, `parked=[]`), in this order. No rule, threshold, arm, seed or state was changed; no row was
re-run; no playout was added (scoring only, ≈ 3 CPU-min).
1. `read_mat_test.py` 7 / 7 and `queue_mat_test.py` 5 / 5 (planted effects, no archive).
2. `gigo_audit_mat_plateau.py` (§9.7's audit, written for this read before any verdict was looked at; counts,
   provenance and file hashes only): **531 / 531 checks PASS** (`gigo_audit_mat_plateau.json`).
3. `identity_mat.py` (`identity_mat_plateau.json`): PASS at all five anchors, P\* included (12 / 12 states each).
4. `score_mat.py --bank interim|plateau --tag plateau_step` (tables `<archive>/score/plateau_step.pkl` sha256
   `7d4513fa…7b0e25`, `<archive>/plateau/score/plateau_step.pkl` `cd443030…bd9f0e`; the LEAN read's `table.pkl` is
   untouched), then `read_mat.py plateau`.

### 7.1 The GIGO audit (first)

| check | result |
|---|---|
| the banks | ids hashes = §6 (interim) and §9.2 (plateau); both content sha256 recomputed on load; 0 shared ids, 0 shared battles |
| the anchors | every copy = `ckpts.json` = `checkpoints_x26.json` = the run's own file (read only), all five |
| coverage | every (bank, anchor, unit) at its §9.3 mode, and every world arm: exactly one row per wanted id; 0 duplicates, 0 foreign, 0 missing, 0 torn |
| provenance | every world row `head` 826bee89; every pin row 9c033489; `cp` / `step` / `arm` / `world_head` their own; every stamped row `driver` 198fa91e and `bank` its namespace; the only unstamped rows are the LEAN run's (before A1) |
| defects | **0** in every world arm and playout; 0 "CRN broken" (the tooth reproduced F in every F+U, F+M, O−M, O−U and FN build) |
| the k = 6 identities | F+U ≡ F at 25M and P\*, O−U ≡ O at P\*, on both banks: 0 outcome / trace / critic mismatches (interim 160 k = 6 states, 23 both-excluded; plateau 160, 31 both-excluded) |
| the 10 % exclusion flag (§9.7) | **RAISED** on the plateau bank's F (10.1 %), F+U (10.1 %) and FN (10.4 % at both anchors), and on the interim LEAN F (33 / 320 = 10.3 %); every other arm ≤ 10 % (interim FULL F 9.6 %, O−M / O−U ≤ 0.6 %). All are `WorldError` (no honest draw fits), none a defect. FINDING 3 |
| the 1 % defect flag | not raised anywhere |
| non-finites, root ties | 0, 0 |
| untouched since the queue | 70 row / world / root / belief files, none written after `QUEUE_DONE`; their sha256 manifest is in `gigo_audit_mat_plateau.json` (manifest sha256 `2960e0c9…`). No earlier row manifest existed, so this binds the read from now on (FINDING 5) |

### 7.2 The registered reads (LEAF-K8; paired; 2,000 battle-clustered draws, seed 20261011)

| anchor | read | n common | G_F | IMP_O | IMP_F | IMP_FN | TODAY |
|---|---|---|---|---|---|---|---|
| 25M | 1 (plateau bank) | 719 | +0.061 [+0.048, +0.073] | +0.066 | +0.005 | +0.006 | 0.001 [0, 0.029] |
| P\* | 1 (plateau bank) | 719 | +0.058 [+0.046, +0.071] | +0.054 | −0.004 | +0.001 | 0.014 [0, 0.046] |
| 25M | 2 (interim FULL) | 723 | +0.078 [+0.065, +0.092] | +0.071 | −0.007 | −0.003 | 0.012 [0, 0.047] |
| P\* | 2 (interim FULL) | 723 | +0.048 [+0.037, +0.060] | +0.037 | −0.011 [−0.022, −0.001] | −0.008 | 0.014 [0, 0.163] |
| P\* | 3 (interim LEAN, 5 anchors) | 287 | +0.050 [+0.031, +0.070] | +0.028 | −0.021 [−0.038, −0.006] | −0.013 | 0.033 [0, 0.221] |

| # | read | GAP (all, PRIMARY) | BELIEF | FRONTIER | STILL-MOVING | standing |
|---|---|---|---|---|---|---|
| **1** | **plateau bank, [25M, P\*] FULL** | **NOT DETECTED: −0.0027 [−0.0196, +0.0152]** | **NOT DETECTED** +0.013 [−0.022, +0.042] | **FRONTIER-STATIC** (no bucket GATE-OPEN at either anchor; one GATE-WEAK at 25M, material 8–9) | NOT DETECTED (= the pair) | **THE VERDICT OF RECORD** |
| 2 | interim bank, [25M, P\*] FULL | **GAP-SHRINKS: −0.0304 [−0.0471, −0.0120]** | NOT DETECTED +0.002 [−0.043, +0.157] | FRONTIER-STATIC (all 13 gates CLOSED at both) | STILL-SHRINKING (= the pair) | the replication |
| 3 | interim LEAN, 25M → 100M → 175M → 225M → P\* (287 states) | NOT DETECTED: −0.025 [−0.052, +0.003] | NOT DETECTED +0.033 [−0.022, +0.219] | NOT READ (LEAN) | 225M → P\*: NOT DETECTED −0.003 [−0.026, +0.020] | the shape, quoted beside 1 |
| 5 | BANK-DEPENDENT (record vs replication) | **not BANK-DEPENDENT** by the registered rule (NOT DETECTED vs SHRINKS is not opposite directions; FINDING 1) | no | no | — | only GAP turn 15–21 is BANK-DEPENDENT |
| 6 | CONVERGED pooled (225M vs P\*) | NOT READ: the plateau bank holds no 225M (as registered) | | | | |

**Stage buckets (13 per read).** Only a verdict that holds at the Bonferroni 99.6 % is a finding (§9.6).
- Record: at 95 %, GAP-SHRINKS at turn 8–14 (−0.052 [−0.084, −0.020]) and material 10–11 (−0.034 [−0.067, −0.001]);
  GAP-GROWS at turn 15–21 (+0.030 [+0.002, +0.059]). **At 99.6 % only turn 8–14 holds (−0.052 [−0.101, −0.004]):
  the record's one bucket finding.**
- Replication: at 95 %, GAP-SHRINKS at k 1–2, k 5, material 8–9, turn ≤ 7 and turn 15–21; at 99.6 % only k 1–2
  (−0.061 [−0.118, −0.003]).
- LEAN trajectory: at 95 %, k 1–2 and material 10–11 shrink; nothing at 99.6 %.
- The two banks' bucket findings do not coincide (record turn 8–14, replication k 1–2). Turn 15–21 is
  BANK-DEPENDENT (GROWS on the plateau bank, SHRINKS on the interim).

**Per stratum (descriptive).** G_F at 25M / P\*, record: snapshot +0.056 / +0.067, bot +0.066 / +0.049. Replication:
snapshot +0.080 / +0.055, bot +0.077 / +0.041.

### 7.3 R3 at P\* (E2's rule; LEAF-K8; P\*'s own valid states, 800 per bank)

| bank | stratum | Δs = F+U − F+M (reveal-only) | Δn = O−M − O−U (hide-only) | classes detected | verdict |
|---|---|---|---|---|---|
| **plateau (record)** | **all** | +0.0125 [+0.0003, +0.0250] (rule 8: a boundary, not clear) | +0.0084 [−0.0030, +0.0200] | M, U | **SHARED** |
| plateau | snapshot | +0.005 [−0.012, +0.024] | +0.002 [−0.016, +0.019] | M, U | SHARED |
| plateau | bot | **+0.020 [+0.003, +0.038] (U)** | +0.015 [−0.0005, +0.031] | M, U | SHARED |
| interim (replication) | all | −0.004 [−0.016, +0.008] | +0.005 [−0.006, +0.015] | M, U | SHARED |
| interim | snapshot | −0.015 [−0.032, +0.002] | −0.003 [−0.019, +0.012] | M, U | SHARED |
| interim | bot | +0.007 [−0.009, +0.023] | +0.013 [−0.0003, +0.026] | M, U | SHARED |

No qualifier is MATERIAL, CLOSE or SIDES-DISAGREE anywhere. R3 is not BANK-DEPENDENT (SHARED on both). E2's 15M
read was U-FIRST on its competent-opponent bank; at the plateau both classes carry value and neither side names a
first class.

### 7.4 What it says ("are we getting closer?")

- **Registered answer, on the record: NO DETECTABLE CHANGE.** Between 25M and the plateau, on 719 fresh plateau-drawn
  states, the oracle − honest gap is +0.061 → +0.058 (−0.003 [−0.020, +0.015]). The network's belief does not rise
  (TODAY 0.001 → 0.014, against p\* ≈ 0.2–0.3). No stage bucket opens a search gate at the plateau.
- **The replication disagrees in kind but not in direction.** On the interim bank's 723 states the same pair reads
  GAP-SHRINKS (−0.030). The registered BANK-DEPENDENT rule does not flag "NOT DETECTED vs SHRINKS", so the record
  stands as written. Read together: the interim bank's 25M gap (+0.078) is larger than the plateau bank's (+0.061),
  and the two banks' P\* gaps are close (+0.048 / +0.058). Descriptively, the shrink is seen on states drawn from 230M games,
  not on states drawn from the plateau's own games.
- **As on the LEAN read, what moves is the oracle, not honest search.** IMP_O falls on both banks (record +0.066 →
  +0.054; replication +0.071 → +0.037), while honest one-ply search stays at or below the policy at every anchor
  (IMP_F +0.005 / −0.004; −0.007 / −0.011).
- **Every number above is a SELF-REFEREE read (§7.5).**
- **What enables search is unchanged by maturity to the plateau:** a world sampler that puts ≈ 0.2–0.3 of weight on
  the true world. Training the policy to 280M did not move the network's belief toward it.

### 7.5 The referee: every cell is a SELF-REFEREE read

The truth T (§3.2) is played to the end by the anchor's own greedy policy on BOTH sides, so every IMP, G_F, TODAY and
R3 number here scores the opponent's replies as the searched network would make them, not as the actual opponent did.
Half of each bank is bot-stratum (400 / 800 states). E1-0 (`../search_e1_0_2026-10-26/`, 015f00e2) re-scored the
maturity interim LEAN bot-stratum picks at 225M with the state's own scripted bot as the referee. There G_F on bots
fell +0.050 → +0.010 (descriptive, D −0.040 [−0.068, −0.015] at 95 %), and the primary cells read UNDECIDED. So the
bot-stratum cells above (G_F on bots, R3's bot stratum) are labelled SELF-REFEREE. The snapshot stratum is closer to
self-play, but its referee is still the anchor, not the snapshot that played. Not re-scored here: a bot-referee
re-score is a new registration (PROPOSAL below).

### 7.6 FINDINGS

1. **The record and its replication disagree on the primary in kind** (NOT DETECTED vs GAP-SHRINKS). The registered
   BANK-DEPENDENT rule covers only opposite directions, so the disagreement is not flagged and the record stands. The
   registration does not say what a SHRINKS that fails to replicate means; the most literal reading, used here, is
   that the verdict of record is NOT DETECTED and the interim pair's SHRINKS is a bank-specific replication result.
2. **R3's record reveal-only side sits on rule 8's boundary** (Δs lower bound +0.0003 < +0.0005).
   Had it cleared +0.0005, the record's reveal-only side would read U. The verdict SHARED holds either way, since
   the hide-only side reads no direction.
3. **The 10 % exclusion flag is raised** on the plateau bank's honest F / F+U / FN builds (10.1–10.4 %) and on the
   interim LEAN F (10.3 %; README §5.1 reported the rate but not the flag). The common sets drop 81 (plateau) and 77
   (interim) states for F exclusions. Exclusions are states where no honest draw fits what our side saw, so the read
   covers the states honest worlds can represent.
4. **CONVERGED stays unread** (pooled needs 225M on the plateau bank: §9.4's deferred step 4 plus 225M, as
   registered). STILL-MOVING NOT DETECTED on the record is not evidence of convergence.
5. **No row manifest existed before this read.** Untouched-ness rests on mtimes (none after `QUEUE_DONE`) and on the
   content checks above. The sha256 manifest now in `gigo_audit_mat_plateau.json` binds the rows from here on.
6. **The L1 replication** registered in `search_ledger.md` was NOT done here; it was read separately 2026-10-25
   (`../search_l1_plateau_2026-10-25/`): the PAYS-SCOPED cell REPLICATES (thin), all SELF-REFEREE, and sits on the 25M anchor.
7. No hazard hit: CPU only (`CUDA_VISIBLE_DEVICES=""`, the agent cores, `nice -n 19`, `mem_cap` short lane), nothing
   written under `models/` or `data/` or into the archive's rows (only the two new score tables). The three
   checkouts (`head`, `pin_x26`, `driver_198fa91e`) are KEPT, for the deferred step 4 and any re-score.

**PROPOSAL (not registered): the BOT-REFEREE re-score of the plateau step.** E1-0's instrument, `opp_bot` in-core,
on both banks' FULL bot strata at 25M and P\* (≈ 800 bot states per anchor; E1-0 says ≈ 650–750 resolve ±0.015). It
would read GAP and R3's bot stratum against the real bot. ≈ 7 CPU-h per anchor at E1-0's 1.37 CPU-h per 160 states.
It needs its own PREREGISTRATION before any bot playout.
