# PRE-REGISTRATION — the MATURITY BATTERY on X26: the interim bank and the LEAN trajectory (2026-10-18)

**Status.** Committed with the bank's ids BEFORE any trajectory output exists: no anchor's root, truth, belief, world
or playout row on a bank state had been computed when this file and `states_mat_x26.json` were committed. A rule below
changes only by a dated AMENDMENT appended here before the outputs it touches exist. The harness shipped one commit
earlier: `4c906a9c` (backlog T35; §8 below).

**What existed when this was written** (disclosed, none of it a trajectory output):
- the bank itself (§2), drawn by `make_mat_bank.py` with no model and no playout;
- an ENGINEERING PILOT (§5) on 12 eligible states that are NOT in the bank (`make_mat_bank.py --pilot 12`, a separate
  salt), at the 225M and 25M anchors: every arm, the identity check and the cost. Its rows live under
  `~/gen3ai_archive/hidden_info_maturity/pilot/` and are never read as trajectory output.

**The question** (owner, 2026-10-17: "what does it take to enable search? Are we getting closer, not closer?"). The
instrument is `design_hidden_info_binding.md` §12.13.7: the per-stage value-of-information and belief-dose reads
re-run on X26's own checkpoints and read as a TRAJECTORY. As the policy matures, does honest one-ply search's gap to the
oracle shrink (GAP-SHRINKS), does the network's effective belief quality rise toward the truth (BELIEF-RISES), and
does the frontier where honest search beats the critic move earlier in the game?

**Vocabulary.** IMP = decision improvement: the win probability gained per decision by playing a scorer's pick
instead of the policy's own top-1, judged on the true world's held-out referee seeds. **O** = the oracle (search in the
true world). **F** = the fully honest world (every hidden fact drawn from Smogon usage, kept only if it replays what our
side saw). **FN** = F with the never-seen species and the hidden moves drawn from the NETWORK's own belief. **G_F** =
IMP_O − IMP_F: what the hidden information costs honest search. **TODAY** = the weight p on the true world at which
F's belief-dose curve (M8, §6.9) reaches FN's IMP: the network's belief in "fraction of the truth" units. **LEAF-K8** =
the mean outcome over 8 worlds played to the end.

## 1. The series and its anchors

| | |
|---|---|
| run | `rb_x26_s2601` (X26, the deep run; LIVE while this battery crawls) |
| pin | `9c0334890d05785bbfadcb2ecbe3fe2d548efe9b`: every checkpoint was saved there (`metadata.json` `pin_history`; every eval manifest's `git_hash`) |
| play commit | the pin itself. It already holds the gender draw pad (`98146ddb` is its ancestor), so the world playouts need no PAD commit (Tier 1's 15M pins did) |
| world builder | the HEAD checkout at the commit that ships THIS registration (§12.13.7 (f)); its sha is recorded in every world row |
| anchors | **the trigger rule** (`maturity_series.anchors`): for every multiple m of 25M, the FIRST checkpoint file with m ≤ step < m + 25M. Registered here, LEAN: **x26_025M** (`checkpoint_25000027`), **x26_100M** (`100000037`), **x26_175M** (`175000114`), **x26_225M** (`225000152`) |
| queue order | 25M, 225M, 100M, 175M: the first and the last anchor land first, so a first-vs-last read exists as early as possible |
| copies | read-only, sha256 before and after, made read-only (`trigger_mat.py`; `checkpoints_x26.json` in this folder = the archive's `ckpts.json` at registration) |

The other anchors (50M, 75M, 125M, 150M, 200M and every later multiple) may be added with `trigger_mat.py register`.
Adding an anchor adds a point; it never changes a rule. **Deviation from §12.13.7's schedule, disclosed:** §12.13.7
asks for FULL at the first anchor and at the plateau and LEAN between. This unit runs LEAN at all four (the
orchestrator's brief). The LEAN states are nested inside the FULL bank, so FULL at 25M can be added later on the same
states without re-running anything.

## 2. The INTERIM bank

**Why interim.** §12.13.7 (b) draws ONE bank of 800 states at the PLATEAU checkpoint. X26 is still climbing at 234M:
no plateau exists (§10.5's criterion). §12.13.7 does not forbid an earlier bank. It fixes the plateau bank as the
bank of record, and its own secondary (b) already admits other state sets. So the interim bank SUPPLEMENTS the plateau
rule (§2.5); it does not replace it.

### 2.1 Source

X26's three newest eval cycles at registration: `step_230000074`, `step_232000091`, `step_234000025`. All are recorded
at the pin. Each was copied out of the run directory read-only (every file but the checkpoint hard link `snapshot.zip`,
per-file sha256: `traces_manifest.json` and `<cycle>.sha256.json` in the archive). The bank is drawn from the copies;
the run directory is never written.

### 2.2 The pool

Every trace of every roster opponent: the nine scripted bots and the five own-snapshot sentinels; never `ext_`. 650
traces in all. They are replayed through the pin's `core_events --trackers --obs`, with gate ① per battle and the
clock-decode turn gate per decision. **Eligible** = a free move round, turn 2–40, ≥ 3 legal actions (§12.13.2).
**Census**: k (opponent species seen), turn, material (`stage_facts.parse_stage`), and the **stratum**: `snapshot` (a
sentinel opponent, i.e. competent play) or `bot`.

### 2.3 The draw (`maturity_bank.draw`; `make_mat_bank.py`)

- **Cells:** (k bin, stratum), k bins 1–2 / 3 / 4 / 5 / 6 (§12.13.1). They are processed k bin by k bin, snapshot
  before bot within a bin.
- **Per cell:** 80 states. That is 160 per k bin and **800 in all**: §12.13.7's FULL size, with §12.13.2's per-bin
  floor.
- **Order:** within a cell, candidates in `sha256("hib_maturity_x26_v1:" + id)` order.
- **Battle caps:** at most 1 state per battle per k bin and 3 per battle in all, counted over the cells processed
  so far (§12.13.2's caps).
- **LEAN** = the first **32** of each cell in selection order: **320 states**, 64 per k bin (§12.13.7's LEAN size). So
  LEAN ⊂ FULL.
- **Strata 50 : 50, fixed.** This carries E2's recommendation (§16.12: "carry R3 into the plateau battery with
  competent-opponent states in its frozen bank"). It also keeps a bot population comparable with Tier 1's registered
  mix (58 % bot). The natural sentinel share of these cycles' traces is 47 %, so the fixed split is close to natural
  and guarantees both strata the same precision.
- **No network enters the draw.** Tier 1's supplement excluded one checkpoint's top-2 ties. Here each anchor excludes
  its OWN ties (§3.1), so the bank is the same object at every anchor.

### 2.4 A short cell

A cell that cannot be filled STOPS the draw. The ONE pre-registered extension adds the cycle before the three,
`step_228000030`, whole, and draws again by the same rule. A cell still short then STOPS, and the unit is reported. (Not
needed: every cell filled from the three cycles; §6.)

### 2.5 The PLATEAU bank, and whether it replaces or supplements this one

- **When drawn.** When X26's plateau checkpoint is declared (§10.5's criterion), a PLATEAU bank is drawn by §2.2–§2.4
  from that checkpoint's three newest eval cycles. The salt is `hib_maturity_x26_plateau_v1:`, and any battle of the
  interim bank is excluded.
- **It SUPPLEMENTS, never replaces.** Both banks are scored at every battery anchor; the plateau bank is back-filled at
  the anchors already read, about 7 CPU-h per anchor in LEAN (§5).
- **The verdict of record is then the plateau bank's** (§12.13.7 (b) as written). The interim bank's verdicts are its
  pre-registered REPLICATION: a plateau-bank verdict whose interim twin reads the other direction (one SHRINKS where the
  other GROWS, or one RISES where the other is not detected) is quoted as **BANK-DEPENDENT**.
- **Until the plateau bank exists**, the interim bank's verdicts are the EARLY reads. They are PROVISIONAL and steer
  no multi-week commitment (§10.5, §12.13.8).
- **If X26 ends without a plateau**, the interim series is the read of record, labelled "no plateau".
- **If the plateau is declared at or before 234M** (its newest three cycles are this bank's), the plateau bank IS this
  bank.

## 3. Per anchor (LEAN: the 320 LEAN states; FULL: all 800)

1. **Roots** (`roots_mat.py`). The anchor's own forward at every bank root: logits, V, top-1 (§12.13.7 (b): "its own
   top-1 as the baseline"). A state whose top-2 gap is < 1e-6 at an anchor is that anchor's rule-8 exclusion, and it
   leaves every contrast involving that anchor.
2. **O, the truth** (`phase1_mat.py` = the pilot's `run_playouts.one_state`). Every legal action × 40 seeds of
   `truth.turn_seeds(id, 48)`: leaf 0–15, referee 16–39, critic 40–47 (§12.13.7 (c)). Played to the end by the
   anchor's greedy policy on both sides.
3. **F** is network-free, so it is built ONCE and played at every anchor: the anchors' F-based IMPs differ only by the
   network.
4. **FN** is built per anchor from that anchor's own belief (`fn_belief_mat.py` = the pilot's
   `det_worlds.root_belief(moves=True)`; Tier 1's `fn_worlds.fn_sampler`).
5. **Common random numbers.** F is re-sampled into each sibling's screen cache first, and it must reproduce the stored F
   worlds exactly. A mismatch is a DEFECT row, never a silent skip.
6. **FULL adds F+M and F+U** (shared). The builder is `main.value_search.worlds.WorldSampler` with the world guard on,
   `max_tries` 60.
7. **Playouts** (`play_mat_worlds.py` = the pilot's `run_det_playouts.one_state`). 8 worlds, world j on dice seed j;
   the honest one-ply critic on seed 40 + j.
8. **Validity** (`identity_mat.py`, per anchor). `oracle` worlds built at the HEAD and played at the pin must reproduce
   the anchor's Phase 1 on 12 fixed states: every outcome exact, values within Tier 1's tolerances.
9. **Scorers:** LEAF-K8 (primary), TRUNC-4-K8, CRITIC (§12.13.7 (d)); LEAF-K4 descriptive.

## 4. The reads (as code, committed with the harness: `read_mat.py`, rules in `src/main/hidden_info_study/maturity_rules.py`)

- **The common states.** Every across-anchor quantity uses the states that are valid at EVERY anchor read: truth ok, F
  ok, no root tie. FN-based quantities additionally need FN ok at every anchor. ONE battle-clustered bootstrap over their
  battles (Tier 1's: 2,000 draws, seed 20261011) makes every contrast paired.
- **Rule 8.** A bound within ±0.0005 of its boundary is not clear.
- **R-MATURITY** (§12.13.4), first = the earliest anchor read, last = the latest:
  - **GAP-SHRINKS** iff upper95 [G_F(last) − G_F(first)] < −0.0005; **GAP-GROWS** is the mirror. Read on the
    all-state cell (PRIMARY) and on each stage bucket.
  - **BELIEF-RISES** iff lower95 [TODAY(last) − TODAY(first)] > +0.0005. TODAY is clamped to [0, 1] per draw: FN at or
    below F reads 0, FN above the oracle-weighted curve reads 1. So the difference is defined on every draw.
  - **CONVERGED** iff the interval of the all-state G_F between the last two anchors lies inside ±0.01.
  - **FRONTIER-MOVES-EARLIER / RECEDES / MIXED / STATIC.** On each axis, compare the earliest-game GATE-OPEN bucket at
    the last anchor with the first ("none" counts as later than every bucket).
- **The per-anchor GATE: R-STAGE-SEARCH with one network.**
  - GATE-OPEN iff n ≥ 150 and both lower bounds, [TRUNC-4-K8 − CRITIC] and [TRUNC-4-K8], clear +0.0005 at 99.6 %.
  - GATE-WEAK iff both clear at 95 % only.
  - "Each checkpoint separately" and "pooled" coincide for one network.
  - **In LEAN no bucket can open:** 64 states per k bin and about 80 per turn / material bucket, all below 150. So the
    per-bucket gate statistics are printed DESCRIPTIVELY, and **FRONTIER is NOT READ unless the first and the last
    anchor are both FULL.**
- **Descriptive, never a verdict:**
  - G_F per stratum (snapshot / bot) and its change;
  - IMP of O, F and FN per scorer;
  - the t₀ references: Tier 1's 15M end2 / prod reads, and E2's snapshot-game bank. They are different states and
    different networks, so they are unpaired.
- **What the LEAN series can and cannot say.**
  - The all-state change of G_F is read at ±0.027 (§12.13.7's precision table; UNVERIFIED for X26's spread).
  - A k bin's change is read at ±0.061.
  - The frontier is not read.
  - R3 (E2's U vs M) needs F+M / F+U (FULL) for its reveal-only half, and O−M / O−U for its hide-only half. **O−M / O−U
    are not in §12.13.7's arm list; adding them is outside this unit** (a FINDING for the orchestrator). The
    competent-opponent states are in the bank either way.

## 5. Cost (measured by the pilot, 12 states outside the bank, CPU-seconds per state, contended box, 1 thread)

| item (CPU-s per state, mean of 12; median, max) | x26_225M | x26_025M | §12.13.6 / §12.13.7's assumption (end2) |
|---|---|---|---|
| O, Phase 1 (40 seeds × every action + the critic) | 44.5 (40.0, 116.5) | 54.0 (40.1, 165.3) | 98 |
| F playouts (8 worlds + the critic) | 12.7 (13.4, 23.6) | 14.9 (16.1, 32.3) | 28 |
| FN playouts | 13.3 (11.5, 34.5) | 16.4 (11.2, 39.4) | 28 |
| FN worlds: build 0.5 + the F re-sample for common random numbers ≈ 3.8 | ≈ 4.3 | ≈ 4.3 | — |
| roots + the belief read (one forward each) | ≈ 1.5 | ≈ 1.5 | — |
| the identity check (12 states × 13–15 s, spread over 320) | ≈ 0.6 | ≈ 0.6 | — |
| **per state** | **≈ 77** | **≈ 92** | 154 |
| **per anchor, LEAN (320 states)** | **≈ 6.8 CPU-h** | **≈ 8.2 CPU-h** | ≈ 14 |
| F worlds, network-free (built ONCE: 320 × 3.8 s) | ≈ 0.34 CPU-h for the whole series | | |

- **The four LEAN anchors cost ≈ 30 CPU-h** in all: about half of §12.13.7's ≈ 14 CPU-h per LEAN checkpoint. X26's
  states play out faster than end2's (Phase 1 45–54 s against 98 s).
- **The spread is wide.** The heaviest pilot state cost 3.7× the mean, so a 12-state mean is good to roughly ±30 %.
- **On 2 single-thread slots that is ≈ 15–20 wall-hours.** It is SCHED_IDLE beside the live trainer, so it stretches
  whenever the trainer takes the cores.
- **The pilot's GIGO checks all passed:**
  - the identity is exact at both anchors (12 / 12 states, 89 actions, 0 outcome mismatches, max |ΔV| 1e-5);
  - 0 world defects and 0 exclusions in F, FN and oracle;
  - the CRN tooth held: the F re-sample reproduced the stored F on every FN build;
  - 0 root ties;
  - the checkpoints load through the strict loader with the win-probability critic;
  - Tier 1's scorer reads every row, since `truth_arrays` checks each anchor's top-1 against its own roots.

## 6. The bank as drawn (`states_mat_x26.json`)

| | FULL (800) | LEAN (320) |
|---|---|---|
| ids hash (sha256 of the sorted ids) | `036793b367612d08d32ebf9d05d5d36dfbeccf0b8ee5d0357378ff5d801ccebe` | `20b98bf42bdc52ced2d5b457bf2e88b6fc06d704687284f8ffbd60c598857626` |
| battles | 513 | 276 |
| per k bin (1–2 / 3 / 4 / 5 / 6) | 160 each | 64 each |
| stratum snapshot / bot | 400 / 400 | 160 / 160 |
| turn ≤ 7 / 8–14 / 15–21 / ≥ 22 | 196 / 242 / 193 / 169 | 81 / 103 / 67 / 69 |
| material ≤ 7 / 8–9 / 10–11 / 12 | 211 / 190 / 232 / 167 | 83 / 70 / 98 / 69 |
| the traced side's recorded outcome, snapshot (loss / win / draw) | 226 / 166 / 8 | 84 / 73 / 3 |
| … bot (loss / win / draw) | 131 / 256 / 13 | 56 / 99 / 5 |
| cycle 230M / 232M / 234M | 268 / 278 / 254 | 109 / 107 / 104 |

- **The bank directory's content sha256** is `8da1613d1e28f41294f86689870fc26cad438fa50bd013dc9f8debcc6fde3fe2`
  (`main.policy_spectrum.bank`: it is verified on every load). `census.npz` sha256 is `92f9eb9d…d218274`.
- **The pool:** 650 traces and 16,619 eligible decisions.
- **Gate ①:** 650 / 650 battles, 23,444 decisions, 0 failures. The clock-decode turn gate agrees on 23,444 / 23,444.
- **Candidates per cell:** 1,117 (k = 5, bot) to 2,521 (k 1–2, bot). Every cell filled at 80, so the extension
  (§2.4) was not used.
- **The outcome mix inherits the live trace quota** (losses first, then 5 wins and 5 draws per opponent and cycle), as
  Tier 1's did. It is stamped, never drawn on.

## 7. What may not change once outputs exist

The bank (ids, sha), the anchors' files (sha), the seeds, the scorers, the arms, the builder commit (a changed builder
starts a NEW series), the rules of §4 and their code. A defect found later is fixed by an AMENDMENT that reports the read
both ways.

## 8. Pointers

- harness (T35): `src/main/hidden_info_study/maturity_series.py` (series, trigger, copies, cap-new-work),
  `maturity_bank.py` (the draw), `maturity_rules.py` (R-MATURITY), each with its test; this folder's drivers (README).
- the design: `designs/endstate/design_hidden_info_binding.md` §12.13.7 (the amendment registering this bank).

## 9. AMENDMENT A1 (2026-10-19): THE PLATEAU STEP, registered before any of its outputs

**Status.** This amendment is committed with the PLATEAU bank's ids (`states_mat_x26_plateau.json`) before any output
of the plateau step exists. None of these rows had been computed when it was committed:
- any root, truth, belief, world or playout row at P\* (either bank);
- any row on 25M's 480 non-LEAN states;
- any row on the plateau bank, at any anchor;
- any O−M or O−U world or row.

**What the author had seen** (disclosed): the interim LEAN trajectory's read, README §5. Every rule change below
answers one of that read's FINDINGS, and none of them changes a threshold that read was judged by.

The decisions are the orchestrator's (2026-10-19): run steps 1, 2, 3 and 5 of README §5.6; defer step 4 unless the
verdict of record strictly needs it; no 225M FULL; no hide-only at 25M. The harness is the commit just before this
one ("feat(hib-maturity): the PLATEAU STEP's harness"; its sha is in README §6).

### 9.1 The plateau checkpoint P\* (DECLARED)

| | |
|---|---|
| anchor | **`x26_P280M`** = `rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256 `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed` (the sidecar: `60432d42…`) |
| declared by | the orchestrator, 2026-10-19. This registration does not re-derive §10.5's criterion |
| registration | `maturity_series.SERIES["x26"].plateau` (the exact step and sha256). `trigger_mat.py register --plateau` copies the file read-only and refuses any other sha. It is a NAMED anchor off the 25M grid, labelled `P` so it can never collide with a grid label |
| at the pin | its sidecar `git_hash` is `9c033489`, and its three newest eval cycles are recorded there |
| the copy | `<archive>/ckpt/x26_P280M/`, sha256 = the declaration = the run's own file = `~/.cache/gen3ai/x26_plateau/`'s verified copy |

### 9.2 The PLATEAU bank (§2.5, executed)

**Source.** P\*'s three newest cycles: `step_276000077`, `step_278000056`, `step_280000170`.
- They were copied (`trigger_mat.py traces --bank plateau --from ~/gen3ai_archive/x26_eval_traces_plateau`) from the
  orchestrator's copy, with per-file sha256: `traces_manifest_plateau.json` and `<cycle>.sha256.json`.
- **Every file equals the run's own file:** 1,056 + 1,071 + 1,046 = 3,173 of 3,173. None is missing from the copy, and
  none is absent from the run.

**The draw.** It is §2.2–§2.4's rule, unchanged:
- 80 per (k bin × stratum) cell, LEAN = the first 32;
- salt `hib_maturity_x26_plateau_v1:`;
- run by `make_mat_bank.py --bank plateau` at the pin, with the harness commit's drivers.

**The exclusion.** Every battle of the interim bank (513 battles) was excluded before the draw
(`maturity_bank.draw(exclude_battles=…)`). It removed **0** candidates: a battle id carries its cycle, and the two
banks' cycles are disjoint. So the rule held, but vacuously. The two banks share 0 ids and 0 battles.

| | FULL (800) | LEAN (320) |
|---|---|---|
| ids hash (sha256 of the sorted ids) | `728dcdd44feff8ee240b7bd0285f2e5857e063f6584f2089a29007fd9f92da85` | `4b63651ddc94256aa5992afc9f9f4c44d2af4a74ed520390a121c907a4ac544d` |
| battles | 500 | 272 |
| per k bin (1–2 / 3 / 4 / 5 / 6) | 160 each | 64 each |
| stratum snapshot / bot | 400 / 400 | 160 / 160 |
| turn ≤ 7 / 8–14 / 15–21 / ≥ 22 | 202 / 242 / 187 / 169 | 86 / 88 / 72 / 74 |
| material ≤ 7 / 8–9 / 10–11 / 12 | 214 / 210 / 225 / 151 | 87 / 82 / 92 / 59 |
| the traced side's outcome, snapshot (loss / win / draw) | 241 / 153 / 6 | — |
| … bot (loss / win / draw) | 129 / 259 / 12 | — |
| cycle 276M / 278M / 280M | 260 / 269 / 271 | — |

- **Hashes.** The bank directory's content sha256 is `25d2e83ce0ce9a52f7e5af29fde9fd95393472703321c5d50e391f721b738a56`
  (`<archive>/plateau/bank/`, verified on every load). `census.npz` sha256 is `c19bc87c…32fd`.
- **The pool:** 634 traces, 15,690 eligible decisions.
- **Gate ①:** 634 / 634 battles, 21,434 decisions, 0 failures. The clock-decode turn gate agrees on 21,434 / 21,434.
- **Candidates per cell:** 1,099 (k = 5, bot) to 2,362 (k = 6, snapshot). Every cell filled, so the extension
  (`step_274000392`) was not used.

### 9.3 What runs: the per-(anchor, bank) modes (`ckpts.json` `banks`; `checkpoints_x26.json` here = it at registration)

| anchor | interim bank | plateau bank |
|---|---|---|
| x26_025M | **FULL** (step 2: its 320 LEAN rows are reused; 480 new states, plus F+M / F+U on all 800) | **FULL** (step 3) |
| x26_100M, x26_175M, x26_225M | LEAN (done, README §5) | — (step 4, DEFERRED: §9.4) |
| **x26_P280M** | **FULL + hide-only** (steps 1 and 5) | **FULL + hide-only** (steps 3 and 5) |

- **FULL** = every bank state, arms O, F, FN, F+M and F+U.
- **+ hide-only** adds **O−M** and **O−U**, E2's arms. They are `agents.info_levers.spec.PRESETS` at the builder, and
  network-free: built ONCE per bank (`worlds/shared/OmM`, `OmU`) and played at the pin like every arm.
  - They are built after F has been re-sampled into each state's screen cache. A re-sample that does not reproduce the
    stored F worlds is a DEFECT row (the CRN tooth, §3.5).
- **The queue order is 1 → 2 → 3 → 5:**
  - the interim bank's shared F / F+U / F+M worlds;
  - P\* on interim, then 25M on interim;
  - the plateau bank's shared worlds, then P\* and 25M on it;
  - last, the hide-only worlds and plays.
- **The identity (§3.8)** is per ANCHOR. It is read on the interim bank's 12 fixed states at P\*, as at every earlier
  anchor.

### 9.4 Deviations from §2.5 (disclosed)

- **The plateau bank is scored at 25M and P\* only, not at every anchor.** The back-fill at 100M and 175M (README §5.6
  step 4, ≈ 15 CPU-h LEAN), and at 225M, is DEFERRED. Under this amendment it enters NO verdict of record:
  - GAP, BELIEF and FRONTIER read the first anchor against the last;
  - the one-bank CONVERGED is NOT READ (§9.6).
  So the middle anchors on the plateau bank would be descriptive points only. They may be added later
  (`trigger_mat.py set-mode … --bank plateau`): that adds points and changes no rule.
- **225M stays LEAN on the interim bank** (the orchestrator: no 225M FULL).

### 9.5 The world builder and the drivers

- **The builder stays the registration commit `826bee89`** (§12.13.7 (f): "a changed builder starts a new series";
  `mat_common.BUILDER_SHA`):
  - `launch_queue.sh` refuses a builder checkout at any other commit;
  - `build_mat_worlds.py` refuses one too;
  - every world row records it as `head`.
- **The drivers run from a NEW checkout** at the commit that ships this amendment. Every row they write is stamped
  `driver` and `bank`. The folders the drivers import (Tier 1's, the pilot's, by-reveal-count's) are byte-identical
  to `826bee89`'s (`git diff --stat 826bee89 HEAD` on them is empty). So no world or playout computation changes, only
  the bookkeeping around it.
- **The pin side is unchanged:** `~/.cache/gen3ai/hib_mat/pin_x26` at `9c033489`, with its own Rust release builds.

### 9.6 The rules (as code, committed here: `maturity_rules.py`, `read_mat.py plateau`; the planted tests are `maturity_rules_test.py`, `read_mat_test.py`, `queue_mat_test.py`)

**CONVERGED (README FINDING 2).**
- **The problem.** As registered (§4: one bank, the last two anchors, the whole 95 % interval inside ±0.01) it is
  unreachable on either bank:
  - LEAN measured ±0.021 between adjacent anchors on 287 common states;
  - FULL's ≈ ±0.017 is also above 0.01;
  - ±0.01 needs ≈ 1,450 common states.
- **The amendment.**
  - **The one-bank CONVERGED is NOT READ.** It is reported with that reason, never as "NOT CONVERGED".
  - **CONVERGED is read only POOLED over both banks** (`read_mat.converged_pooled`: each bank's common states for the
    pair, concatenated, ONE battle-clustered bootstrap over both banks' battles). The pair is P\* and the interim
    anchor immediately before it (225M), and it is read only when BOTH banks hold both anchors.
  - In this step the plateau bank does not hold 225M, so pooled CONVERGED is **NOT READ**.
  - Holding it would need 225M on the plateau bank: LEAN gives ≈ 570 pooled common states (≈ ±0.015, still
    unreachable); FULL on both banks gives ≈ 1,440, which is just reachable. That is README §5.6's step 6 (≈ 43 CPU-h),
    which the orchestrator declined.
- **Why not a wider margin instead.** The LEAN data already show the 175M → 225M change, −0.002 [−0.024, +0.019]. A
  margin chosen after seeing it would be tuned to it.
- **What replaces it, reachable at these sizes: STILL-MOVING** (every R-MATURITY read).
  - The all-state change of G_F between the last two anchors reads STILL-SHRINKING iff its upper 95 % bound is
    < −0.0005, and STILL-GROWING iff its lower bound is > +0.0005.
  - Otherwise it reads NOT DETECTED, which is **not** evidence of convergence.

**FRONTIER (README FINDING 4).** It is read only when EVERY anchor named is FULL, not just the first and the last. A
LEAN anchor between two FULL ones makes the common set the LEAN 320, where no bucket can reach n ≥ 150. The registered
first-and-last rule would then read a FRONTIER-STATIC that holds by construction. No earlier read is affected: the
LEAN read named no FULL anchor.

**Bucket GAP multiplicity (README FINDING 1).**
- Every stage bucket's GAP verdict is ALSO read at the 99.6 % level (`GAP_BONFERRONI`: Bonferroni over the 13
  buckets, the gate's own level).
- The 95 % bucket verdicts stay as registered and are quoted, but only a bucket verdict that also holds at 99.6 % is
  reported as a finding.
- The all-state PRIMARY is unchanged.

**R3 at P\*** (`maturity_rules.r3_read`). This is E2's registered R3 (`read_e2.verdict` / `read_frame`). A planted
test shows it equal to `read_e2.read_frame` on the same IMPs: verdict, qualifiers, and both numerators and their
intervals.
- The reads, at LEAF-K8:
  - Δs = IMP(F+U) − IMP(F+M), the reveal-only side;
  - Δn = IMP(O−M) − IMP(O−U), the hide-only side.
- The verdict is U-FIRST / M-FIRST / SHARED / NOT DETECTED.
- The qualifiers: MATERIAL at 0.005, CLOSE at ±0.01, SIDES-DISAGREE.
- The descriptive shares are s_X and n_X.
- **The states:** P\*'s own valid states on each bank (truth ok, no root tie), with one bootstrap over their battles.
  - PRIMARY: all states.
  - SECONDARY: each stratum on the same draws. The snapshot stratum is E2's competent-opponent population.

**THE READ OF THE PLATEAU STEP** (`read_mat.py plateau` → `result_mat_plateau.json`, from the two tables
`score_mat.py --bank <b> --tag plateau_step`):

| # | read | standing |
|---|---|---|
| 1 | the PLATEAU bank, R-MATURITY over [25M, P\*], both FULL (800): GAP (all + buckets + `GAP_BONFERRONI`), BELIEF, FRONTIER, the per-anchor gates, STILL-MOVING (here = the pair) | **THE VERDICT OF RECORD** (§2.5) |
| 2 | the INTERIM bank, the same FULL pair [25M, P\*] | its registered REPLICATION |
| 3 | the interim LEAN trajectory [25M, 100M, 175M, 225M, P\*] on the LEAN states (STILL-MOVING = 225M → P\*; no FRONTIER, by the rule above) | the trajectory's shape; quoted beside 1, never instead |
| 4 | R3 at P\*, on each bank | plateau = record, interim = replication |
| 5 | BANK-DEPENDENT (`maturity_rules.bank_dependent`): each verdict of 1 against its twin in 2, and R3's plateau against interim | a disagreement is quoted BANK-DEPENDENT. **Rule:** opposite directions (SHRINKS vs GROWS; U-FIRST vs M-FIRST; MOVES-EARLIER vs RECEDES), or one RISES where the other does not |
| 6 | CONVERGED pooled (P\* vs 225M) | NOT READ in this step (above) |

**PROVISIONAL / plateau labels.** A verdict of 1 is the plateau replication the owner's rule asks for (a research-path
claim needs a PLATEAUED model). It is quoted with P\*'s declaration date and its BANK-DEPENDENT flag.

### 9.7 Validity, registered now (the GIGO audit is written for the read, as in README §5.1; counts only)

- **Coverage:** per (bank, anchor, unit), exactly one row per wanted id, with 0 duplicates, 0 foreign ids and 0 torn
  lines.
- **Provenance:** every world row's `head` is `826bee89`; every new row's `driver` is the launch commit and its `bank`
  its namespace; every pin row is at `9c033489`.
- **Defects:** 0 in every world arm and every playout. The CRN tooth reproduces F in every sibling build (F+U, F+M,
  O−M, O−U, FN).
- **The identities:**
  - the cross-commit identity at every anchor, P\* included;
  - **O−U ≡ O** and **F+U ≡ F** at k = 6, outcome for outcome (E2's: with all six opponent Pokémon seen, nothing of the
    never-seen class is left to hide or reveal).
- **Exclusion flags:** an arm's exclusions above 10 % of a bank are flagged; any defect above 1 % is flagged.

### 9.8 Cost and the queue (README §6)

From this battery's MEASURED per-state CPU (LEAN rows' `cost`, one thread):
- O 49–56 s;
- one 8-world arm played 16–18 s;
- an F world build 13.7 s.

The step is **≈ 150 CPU-h**:

| part | CPU-h |
|---|---|
| shared worlds | ≈ 8 interim + ≈ 9 plateau |
| P\* FULL | ≈ 27 per bank |
| 25M's FULL upgrade on interim | ≈ 20 |
| 25M FULL on plateau | ≈ 29 |
| hide-only at P\* on both banks | ≈ 27 |

The LEAN queue made ≈ 1.26 CPU-h per wall-hour on its 2 SCHED_IDLE slots beside the live run. At that rate the step
takes **≈ 4–5 days**.

§7 binds every item above once outputs exist.
