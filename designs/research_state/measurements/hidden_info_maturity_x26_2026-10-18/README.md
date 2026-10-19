# The MATURITY BATTERY on X26: the interim bank and the LEAN trajectory (2026-10-18)

**Status.**
- **The harness** is T35 (`design_hidden_info_binding.md` §12.13.7; backlog X26-6). It shipped before the
  registration.
- **The registration** is [`PREREGISTRATION.md`](PREREGISTRATION.md), shipped with the bank's ids
  (`states_mat_x26.json`) before any trajectory output existed.
- **The queue** crawls detached (§3), launched 2026-10-18 04:45 PDT from `826bee89`. The read is `read_mat.py`, committed with the harness, before any output.
- **Results:** READ 2026-10-19 (§5), after `QUEUE_DONE` (2026-10-19 04:30 PDT, nothing parked). GAP-SHRINKS NOT
  DETECTED, BELIEF-RISES NOT DETECTED, NOT CONVERGED, FRONTIER NOT READ (LEAN). PROVISIONAL: the interim bank.
- **THE PLATEAU STEP** (§6): REGISTERED 2026-10-19 as amendment A1 (PREREGISTRATION §9), before any of its outputs.
  - P\* = 280M is declared.
  - The PLATEAU bank is drawn (`states_mat_x26_plateau.json`).
  - P\* runs FULL + O−M / O−U on both banks, and 25M FULL on both.
  - It is crawling in a detached queue (§6.3).

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

## 6. THE PLATEAU STEP (amendment A1, registered 2026-10-19; RUNNING)

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

