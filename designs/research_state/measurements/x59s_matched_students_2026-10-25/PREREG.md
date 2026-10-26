# X59-S PRE-REGISTRATION: is human data better than our self-play data? (three matched student arms, 2026-10-25)

**Status: REGISTERED before any student trains.** Built and committed with this file; no student has trained, no
game has been played. An adversarial review of this registration runs BEFORE any training; a change it forces is a
dated amendment at the end of this file, made before training. Training needs a GPU lease (§8); nothing here used the
GPU. The design context is [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md) §2.4 and
REVIEW D's dispositions ([`RESPONSE.md`](../human_replay_quality_2026-10-25/RESPONSE.md) §D).

Abbreviations: P\* = `rb_x26_s2601` @ 280,000,170 (X26's plateau; sha256 `814823c5…91ed`); H, F, P72 =
`heldout72_hpfix`, `forum72`, the procedural panel (the unfamiliar panels, U); home72 = our sample teams (in the pool);
C-GAME = rung 1b's completion of a seat's hidden facts from the Smogon prior; pp = percentage points; lane-h = one CPU
lane for an hour; LB / UB = the lower / upper bound of a 95 % interval.

---

## 1. The question, and what answers it

**The owner's question (2026-10-25):** "is human data better than our own self-play data?" Read in OUR architecture,
at matched volume and recipe: does a policy taught by humans (human states, human labels) play unfamiliar teams better
than one taught by our self-play (P\*'s states, P\*'s labels)?

Three students per seed, our architecture from the SAME initial network:

| arm | states (the rows) | labels |
|---|---|---|
| **S-H** | the human rowset `hr_spec_v1.r1` | the human's own action |
| **S-P** | the SAME rows | one action of frozen P\*, restricted to what the log allows the human (§2.2) |
| **S-S** | P\*'s own self-play on the 719-team pool, the same row count | P\*'s sampled action (on-policy) |

- **Δ_data = S-H − S-S answers the owner's question** (Q-H9).
- **Δ_labels = S-H − S-P** holds the states fixed: is a human a better TEACHER than P\* on human states? It routes H-BC.
- **Δ_states = S-P − S-S** holds the teacher fixed: what human STATE coverage is worth (H-TEAMS' and D1's mechanism).
  Reported; it routes nothing alone.
- **Q-H10 in our architecture:** S-H vs P\* (does a model trained only on humans beat ours?), descriptive.

## 2. The data (built; `src/main/human_data/`)

### 2.1 The human rowset `hr_spec_v1.r1`

Built by `python -m main.human_data.rowset scan | extract | compact` from the registered snapshot `hr_spec_v1`
(every log re-hashed against its manifest; a mismatch refuses the build). The rules, in order, each on the log before
anything is completed:

1. TRAIN dates (≤ 2026-09-20); rated; a result; the first of each duplicate body; **≥ 11 turns** (REVIEW D M7's
   information floor; it also removes every walk-over).
2. **The panel exclusion** (the plan §3.1): a battle is dropped if either side's REVEALED facts fit ≥ 3 mons of one
   team of ANY held-out panel (`heldout72`, `heldout72_hpfix`, `forum72`, `p72`, `forum216b`), counting only mons
   with a revealed move.
3. A seat is read if its account is not bot-flagged (the card's rule) and its pre-battle rating is **≥ 1300**.
4. Each kept seat COMPLETED by C-GAME (battle key = the log's absolute path in the main corpus, so the fill equals the
   HOM extraction's). One row per decision with **≥ 2 legal actions** and either a known own action (the label) or a
   **hidden move** (a `cant` other than a recharge, a confusion self-hit, or the active fainting before it acted
   with no foe Pursuit in the window; the label is the SET of legal moves, plan §3.2).
5. Per row, the **CONSISTENT set**: the legal actions the log leaves possible for the human (a move this mon was seen
   to use; a switch to a mon that was seen). A row whose human label is outside it (a Transform or Mimic slot) is
   dropped. The quitter's last 2 decisions are dropped.
6. Accounts split by a salted hash: `key64(account id) mod 50 == 0` validates (account-disjoint, plan §3.3).

**The rowset as built** (`~/gen3ai_archive/x59s/hr_spec_v1.r1/`; **rows manifest sha256
`e9e0df4224b589798dc507d005c60bcaa8ede9744e1e7403901664899e8fe492`**; `source_sha256` `40f8a5d2…9176`):

| quantity | value |
|---|---:|
| logs read (all re-hashed) / candidate battles / kept battles / kept seats | 376,410 / 164,438 / 27,586 / 44,947 |
| dropped: held-out dates / unrated / ≤ 10 turns / no eligible seat / near a panel | 7,413 / 6,290 / 55,564 / 142,705 / 136,852 |
| seats completed by C-GAME / excluded by it / refused (an OwnError) | 44,838 / 108 / 1 |
| **TRAIN rows / VALIDATION rows** | **1,330,809 / 32,102** (82 shards) |
| hidden-move rows (set labels) | 108,573 (8.2 %): `cant` 49,845, a faint before acting 61,202, confusion 206 |
| rows dropped: one legal action / quitter tail / unlabelled / Pursuit window / label outside the consistent set | 13,157 / 27,583 / 10,374 / 418 / 51 |
| accounts TRAIN / VALIDATION / in both | 3,430 / 70 / 0 |
| the heaviest 10 / 100 accounts' share of rows | 8.3 % / 30.8 % |
| actor 1300–1499 / ≥ 1500 | 78.0 % / 292,412 rows |
| endings (rows): KO / forfeit / timer / tie | 42.1 / 54.3 / 3.1 / 0.5 % |
| switch labels among labelled rows | 31.1 % |
| kept vs excluded battles: forfeit or timer / sides seen whole | 64.1 % vs 49.3 % / 51.9 % vs 74.6 % |

**Extraction identity.** The parts record the commit, a `source_sha256` over every file that makes a row, the live
reader's sha256 and the encoder identity; `compact` refuses parts of different code. On the overlapping battles of a
300-log smoke, 1,196 / 1,196 rows were byte-equal to the HOM extraction of `2bbdffa7`, with equal labels; at full
scale, 8,878 / 8,878 rows of 200 seeded kept battles (VOID 8, §7).

### 2.2 S-P's labels (REVIEW D M2: truly matched)

`python -m main.human_data.teacher label --rows <rowset>/rows --pstar <P*> --out <rowset>/labels_pstar`. On every
non-hidden row: P\*'s policy at T = 1, restricted to the row's consistent set and renormalised, sampled by inverse CDF
with the matched uniform `sha256("x59s_pstar_label|" + rows manifest sha256 + "|" + row id)`. On a hidden row: no P\*
sample, the SAME set label as S-H. The writer refuses a P\* label outside the consistent set; the trainer re-checks it.
Recorded per row: P\*'s mass outside the consistent set (what the restriction removed) and P\*'s NLL of the human's
label.

### 2.3 S-S's rows

`python -m main.human_data.teacher selfplay --pstar <P*> --like <rowset>/rows --out <dir>/ss_rows --part <i>
--envs 64 --threads 2` for parts 0–7, then the same command without `--part` merges (CPU; seed 5900; part i seeds
`[5900, i, ·]`): P\* against itself through the Rust core (the training row path) on the 719-team pool, both seats
recorded, each action sampled from P\* at T = 1 and kept as the label; decisions with one legal action skipped;
finished games only; a game is the split unit (`key64 mod 50` validates); the parts concatenated in order and cut to
exactly the rowset's TRAIN and VALIDATION row counts and its shard sizes (so every arm has the same step budget).
Self-play has no hidden rows: every label is known. Measured ≈ 200 rows a second on the shared agent cores.

### 2.4 Registration of the inputs

The rows manifest sha256 of `hr_spec_v1.r1` is in this file (§2.1) and in the registry
(`designs/research_state/human_replay_snapshots.json`, `rowsets`). The S-P labels manifest and the S-S rows manifest
are generated by the commands above, deterministically, and their sha256s are appended to
[`results/registration.json`](results/registration.json) BEFORE any student trains. A student trained on any other
rows or labels is VOID (§7).

## 3. The students (`python -m main.human_data.student train | export`)

- **Network:** `main.fresh_checkpoint.build_fresh_model(seed)`, `--arch production`'s network (P\*'s architecture
  without its ride-along diagnostic heads; 2,609,183 parameters). The same seed gives the same initial bytes in every
  arm (`init_sha256`, recorded).
- **Seeds:** **5901, 5902, 5903, 5904, 5905**, every arm (15 students).
- **Loss:** for every row, `logsumexp(logits over the legal actions) − logsumexp(logits over the TARGET set)`; the
  target is the one label, or a hidden row's legal moves. Every parameter the policy logits depend on trains; the
  value head gets no gradient.
- **Recipe (one for every arm, fixed before any student trained, untuned):** AdamW (weight decay 1e-5, eps 1e-5), lr
  linear warm-up over 1,000 steps to 3e-4, cosine to 3e-5; batch 512; gradient-norm clip 0.5; fp32 (TF32 off); the
  network in eval mode (it has no dropout); **3 epochs = a FIXED budget of `3 × ceil(train rows / 512)` steps**, no
  early stopping (it would de-match the arms). Batch order: shards in 8-shard buffers, permuted by `(seed, epoch)` and
  `(seed, epoch, buffer)`, so S-H and S-P see the SAME batches. Validation NLL every quarter epoch is descriptive.
- **Device:** one GPU under a lease, every student on the same device and torch version (recorded). Threads: 2.
  GPU training is not bit-reproducible (non-deterministic kernels); a student IS its recorded final digest, and the
  read uses exactly the exported zips.
- **Export:** the final weights through the project's save path; the reloaded checkpoint and the anchors harness's
  `bare` load must give the trained student's logits bit for bit (both checked by `export`). Every student's
  sidecar records its rows, labels, seed, init and final digests, steps and the YARDSTICKS: its NLL on the human
  validation rows' human labels and on the self-play validation rows' P\* labels.

## 4. The reads and their statistics

- **Primary meter: each student vs Kakuna ckpt 34, greedy vs greedy,** at X51's committed symmetric configs
  (`../x51_plateau_vs_kakuna_2026-10-20/configs/anchors_x51_sym_<panel>.json`) with `--model <student zip>`, so every
  game twins X51's P\* vs Kakuna game of the same panel, seed, half and index. **H, F, P72: 20 units × 20 games
  each (1,200 U games a student); home72: 10 units (200).** The play tree is a detached checkout of this
  registration's commit; its identity unit replays X51's `pstar_home72_00` 20 / 20 (winner and turns) before any
  counted student game. Counting is X51's rule (`main.anchors.results.unit_counts`; a failed unit is re-run with the
  same seed at most twice; a cell with > 1 uncounted unit is reported as such).
- **A student's U rate** = the equal-weight mean of its three panels. **An arm's rate** = the mean of its 5 students.
- **The interval of a contrast A − B (the registered rule, REVIEW D M3):** `Δ̂ ± 1.96 × sqrt(V)`, with
  `V = (b̄_A + b̄_B) / 5 + 2 × max(1.5 pp, σ̂)² / 5`, b̄ = the mean over an arm's students of the binomial
  variance of a student's U rate, `Σ_panels p (1 − p) / g / 9` (at 400 games a panel ≈ `p (1 − p) / 1,200`; it stays
  exact when a cell loses a unit). σ̂² = W − b: W is the mean of the three arms' sample variances of their 5 students' U
  rates (df 12), b the mean per-student binomial variance `p (1 − p) / 1,200`. **Guard:** if σ̂'s one-sided 95 % LOWER
  bound, `W × 12 / χ²(12, 0.95) − b`, exceeds (1.5 pp)², the bound is refuted and the contrast is read by a Welch t
  interval across its 10 students instead. The rule is code before any data:
  [`src/main/human_data/x59s_read.py`](../../../../src/main/human_data/x59s_read.py) (`power.py` simulates it with the
  arm-pooled binomial term, equal in expectation).
- **The floor pilot (first):** units 0–3 of each U panel (240 games a student). If EVERY arm's mean is < 0.15, the
  Kakuna read stops there and the in-house rule decides (below); else the remaining units play.
- **In-house head-to-heads (always reported; the decider under the floor):** `main.h2h play --player <S-H_k>
  --opponent <S-S_k> --panel <H | F | P72> --pairs 200 --seed <k>` for each seed k (400 games a panel, 1,200 a
  pairing), and the same for S-H_k vs S-P_k. A pairing's rate is S-H's U win rate; the contrast is the mean over the 5
  pairings minus 0.50, with a t interval across the 5 pairings (df 4).
- **Q-H10 (descriptive):** each arm's U rate minus P\*'s X51 rate on the same twins (U 0.487 [0.459, 0.515]),
  paired by twin.

**Power** ([`power.py`](power.py), seeded, `results/power.json`; 5 seeds × 1,200 U games a student; sized at
p = 0.50, the binomial worst case, so the unsourced 0.30 base of the earlier draft enters nothing):

| seed SD σ_s | P(LB > 0) at Δ = +3 / +4 / +5 pp | false "better" at Δ = 0 | guard fires | median half-width |
|---:|---|---:|---:|---:|
| 0 | 0.68 / 0.94 / 0.996 | 0.002 | 0.00 | 2.58 pp |
| 1.0 pp | 0.64 / 0.89 / 0.98 | 0.010 | 0.003 | 2.58 pp |
| 1.5 pp | 0.56 / 0.81 / 0.94 | 0.020 | 0.05 | 2.58 pp |
| 2.0 pp | 0.46 / 0.67 / 0.84 | 0.028 | 0.24 | 2.97 pp |
| 3.0 pp | 0.25 / 0.38 / 0.55 | 0.028 | 0.77 | 4.80 pp |

At a lower base the binomial part shrinks and power rises (at p = 0.30, σ_s 1.5 pp: 0.59 / 0.83 / 0.95). **The MDE
at 80 % is ≈ 4 pp for a seed SD ≤ 1.5 pp.** The student seed SD is UNMEASURED; it is measured here (σ̂, reported).

## 5. The decision rules (a bound within 1e-9 of 0 takes the undecided class)

| contrast | classes | what follows |
|---|---|---|
| **Δ_data = S-H − S-S** (the owner's) | **HUMAN-DATA-BETTER** iff LB > 0; **SELF-PLAY-DATA-BETTER** iff UB < 0; else **NOT-RESOLVED** | the owner's answer, sent with the interval. HUMAN-DATA-BETTER keeps the human-data levers open; SELF-PLAY-DATA-BETTER, with Δ_labels not HUMAN-LABELS-BETTER and Δ_states' LB ≤ 0, closes human data as a goal (b) lever at this volume (H-TEAMS and H-BC are not built; the owner decides whether to collect more) |
| **Δ_labels = S-H − S-P** | **HUMAN-LABELS-BETTER** iff LB > 0; **OWN-LABELS-BETTER** iff UB < 0; else **UNCLEAR** | HUMAN-LABELS-BETTER → H-BC first (its P-BC placebo conditional, plan §6); OWN-LABELS-BETTER → H-BC dropped; UNCLEAR → H-TEAMS first with the D1 contrast |
| Δ_states = S-P − S-S (reported) | the same classes | LB > 0 supports H-TEAMS' mechanism; routes nothing alone |

- **Two registered questions, each at 95 %.** Δ_data and Δ_labels answer different questions (the owner's; which
  lever to build), so no multiplicity correction is applied across them; each verdict stands alone. Δ_states is
  descriptive.
- **Under the floor** (§4) the same classes are read on the in-house contrasts (S-H vs S-S for Δ_data, S-H vs S-P for
  Δ_labels).
- **Replication** is the five seeds: the interval already carries the seed variance. A class that steers a GPU build
  is not re-run before that build's own registration.
- Nothing is adopted by a rule alone; every lever it opens has its own registration (X60, NOT-READY).

## 6. Predictions (GUESS, registered, read by no rule)

- Every student is far below P\* vs Kakuna on U (from-scratch, 3 epochs of ≈ 1M rows): arm means **0.10–0.25**; the
  floor pilot may fire.
- **Δ_labels ≈ −3 pp (−8 to +2):** P\* is a much stronger player than the rowset's mostly 1300–1499 humans, and the
  restriction keeps most of its mass (it removes ≈ 20 % on the smoke).
- **Δ_data ≈ −1 pp (−6 to +4):** S-S's teacher is strong but its states are the pool's; S-H's are wide but its
  teacher is weak.
- **Δ_states ≈ +2 pp (−3 to +7):** human state coverage helps a student on unfamiliar teams.
- The yardsticks: each student is best on its own validation labels (S-H on human labels, S-S on P\*'s).

## 7. VOID guards (any one voids the read; checked before the first counted game unless stated)

1. A rows / labels / self-play manifest sha256 differs from the registered one (§2.4), or S-H's and S-P's rows
   manifest digests differ.
2. A P\* label outside its consistent set (the writer and the trainer both refuse), or a hidden row given a P\* label.
3. Seed k's `init_sha256` differs across the three arms; a student's steps differ from the registered budget; S-S's
   row counts differ from the rowset's.
4. `export`'s reload or anchors-bare check fails (it refuses).
5. The play tree's identity unit does not reproduce X51's `pstar_home72_00` 20 / 20.
6. A 20-game smoke of one student is not reproduced by a seeded repeat.
7. > 1 uncounted unit in a cell (that cell is reported as such, the read on its counted units; two such cells void
   the primary).
8. The full-scale extraction check fails: a seeded sample of 200 kept battles that the HOM extraction also read must
   give byte-equal rows and equal labels (run once, before training). **Run at registration: PASS**, 8,878 / 8,878
   rows of 200 seeded battles byte-equal with equal labels (every kept battle is also in that extraction;
   [`results/hom_repro.json`](results/hom_repro.json)).

## 8. What it cannot say (registered limits)

1. **From-scratch students are not a plateau fork:** a verdict is about what each DATA teaches a fresh network of our
   architecture at this volume, not about adding human data to P\* (X60 reads that).
2. **The human slice is the kept slice:** ≥ 1300, mostly 1300–1499 actors, forfeit-heavy, half of its sides never
   seen whole (§2.1). A verdict speaks for it, not for "the ≥ 1500 ladder".
3. **Kakuna is human-pretrained**, so it is not a neutral meter of human-likeness; the in-house head-to-heads are the
   non-human-trained meter (REVIEW B M6, D M6).
4. **The Kakuna-family audit (plan §3.4) is not run:** the primary is read without it; its sensitivity re-read
   follows the audit.
5. **One recipe, untuned**, at one volume: a NOT-RESOLVED is not "no difference at any volume".

## 9. Cost and the GPU window

| step | where | cost |
|---|---|---|
| the rowset (scan + extract + compact) | CPU, done | ≈ 6 min + 21 min + 8 min wall, 3 workers (measured) |
| S-P labels (1.36M forwards at ≈ 2.6 ms) | CPU | ≈ 1 h wall, ≈ 2 lane-h (measured rate) |
| S-S rows (8 parts at ≈ 200 rows a second) | CPU | ≈ 2 h wall, ≈ 4 lane-h (measured rate) |
| **15 students × 3 epochs** | **GPU lease** | **≈ 3 GPU-h (ESTIMATE)**; on the CPU ≈ 2.5 s a step at 2 threads, ≈ 10 lane-h a student, so not on the CPU |
| export + yardsticks | CPU | ≈ 1 lane-h |
| floor pilot (15 × 240 games) | CPU | ≈ 10 lane-h at ≈ 10 s a Kakuna game (ESTIMATE) |
| the rest of the Kakuna read (15 × 1,160 games) | CPU | ≈ 48 lane-h (ESTIMATE) |
| in-house head-to-heads (2 × 5 pairings × 1,200 games) | CPU | ≈ 4 lane-h (ESTIMATE) |

**The GPU window** (the orchestrator grants the lease and names the agent): `gpu_lease.sh acquire --owner <name>`;
for each seed, for each arm: `scripts/ops/gpu_lock.sh <PY> -m main.human_data.student train --arm <arm> --rows <rows>
[--labels <labels_pstar>] --seed <seed> --epochs 3 --device cuda --budget-min 45 --out <dir>/<arm>_<seed>`, repeated
until it exits 0 (the network is small: up to three students may share the GPU at once); then `release`. Export and
every game on the CPU afterwards.

## 10. FINDINGS at registration (standing rule 7)

1. **The exclusion with `forum216b` removes 83.2 %** of the candidate battles; what it keeps is 64 % forfeits (§2.1).
2. **The teacher is mostly 1300–1499 actors** (78.0 % of rows; 292,412 rows at ≥ 1500); the card ranks that stratum "the comparator only".
3. **No CPU pilot student was trained:** the registered recipe costs ≈ 10 lane-h a student on the shared agent cores,
   above the brief's 6; the floor pilot measures the base instead.
4. **Torch at 6 threads ran 60× slower than at 2** on the agent cores under the live run; every tool here defaults to 2.
5. **The student seed SD is unmeasured**; the power table is a sensitivity range, not an estimate.
6. **The trainer's CUDA path is untested** (no lease in this unit); the CPU path is tested end to end (train, export,
   reload, the anchors `bare` load). The first GPU student's first chunk is the CUDA smoke.
7. Experiment id X59-S is named in the plan and here only, not in `EXPERIMENT_BACKLOG.md`.

---

## Amendment 1 (2026-10-25, before any student trained): the inputs are registered

The S-P labels and the S-S rows were generated on the CPU and registered in
[`results/registration.json`](results/registration.json) (`checks register`: PASS, no problems):

| input | manifest sha256 | rows | made by |
|---|---|---:|---|
| `hr_spec_v1.r1` | `e9e0df4224b589798dc507d005c60bcaa8ede9744e1e7403901664899e8fe492` | 1,330,809 / 32,102 | `5cbf46dc` lineage (`source_sha256` `40f8a5d2…`) |
| S-P labels | `d768020976d6b92e159b7208e522666cf1bf54c5eb0ea6ff1a3335f679bc3ab2` | 1,362,911 (every row) | the pinned `5cbf46dc` |
| S-S rows | `0ddb9f62c3c0123b8f771faa9d93f3361d40a27a98972e6e1a48a23b91015674` | 1,330,809 / 32,102, the rowset's 82 shard sizes | parts 0–7 at `5cbf46dc`; the validation top-up part 8 and the merge at `14f324c9` |

- **What the labels show:** P\* puts 19.7 % of its mass outside the consistent set (the restriction removes that);
  its sample equals the human's action on 44.8 % of labelled rows; its top-1 agrees on 36.0 %; its NLL of the human's
  label is 1.871 nats (actor ≥ 1300).
- **One change to §2.3, forced by the data, not by any outcome:** the human rowset's account split put 2.36 % of
  rows in VALIDATION, a game split about 2 %, so eight parts held only 26,754 of the 32,102 validation rows. The merge
  now plays seeded TOP-UP parts (seed `[5900, i ≥ 8, ·]`) until the validation rows suffice; one part (8; 401,101
  rows, 4,688 games) did. The train shards fill from parts 0–7 alone, as registered. Two fixes made the merge do
  this within its memory cap (`df756e6a`, `14f324c9`). No student had trained; no game was played.
