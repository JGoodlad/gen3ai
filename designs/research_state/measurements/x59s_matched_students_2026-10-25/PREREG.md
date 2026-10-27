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
| S-S rows | `0ddb9f62c3c0123b8f771faa9d93f3361d40a27a98972e6e1a48a23b91015674` | 1,330,809 / 32,102, the rowset's 82 shard sizes | parts 0–7 at `5cbf46dc`; the validation top-up part 8 and the merge at `64cd54e4` (the manifest records its pre-rebase id `14f324c9`; `teacher.py` is byte-identical in both) |

- **What the labels show:** P\* puts 19.7 % of its mass outside the consistent set (the restriction removes that);
  its sample equals the human's action on 44.8 % of labelled rows; its top-1 agrees on 36.0 %; its NLL of the human's
  label is 1.871 nats (actor ≥ 1300).
- **One change to §2.3, forced by the data, not by any outcome:** the human rowset's account split put 2.36 % of
  rows in VALIDATION, a game split about 2 %, so eight parts held only 26,754 of the 32,102 validation rows. The merge
  now plays seeded TOP-UP parts (seed `[5900, i ≥ 8, ·]`) until the validation rows suffice; one part (8; 401,101
  rows, 4,688 games) did. The train shards fill from parts 0–7 alone, as registered. Two fixes made the merge do
  this within its memory cap (`983ee0d7`, `64cd54e4` on main). No student had trained; no game was played.

---

## Amendment 2 (2026-10-25, before any student trained): REVIEW E's BLOCKERs and MAJORs

[REVIEW E](REVIEW_E.md) (`d02550c0`) read the registration NOT-READY. Every rule below is registered BEFORE the
Kakuna-family audit's flags are read, before the S-S rows are regenerated and before any student trains; the inputs
as built are registered in Amendment 2b (below, same day, still before training). Nothing here used the GPU. Where a
rule departs from REVIEW E's proposed fix, the reason is given ([`RESPONSE.md` §E](../human_replay_quality_2026-10-25/RESPONSE.md)).

### A2.1 S-S faces the human rowset's panel rule (B1)

- **The rule** ([`ss_rule.py`](../../../../src/main/human_data/ss_rule.py), [`teacher.py`](../../../../src/main/human_data/teacher.py)):
  before a self-play game is played, its team pair is REDRAWN while either side's SIMULATED reveal fits ≥ 3 mons of
  one team of ANY held-out panel (`textpass.PanelMatcher()`, `forum216b` included: the human rule's matcher, panels
  and threshold). A side's simulated reveal draws one whole human side's REVEAL SHAPE (per seen mon: moves revealed,
  item revealed, ability revealed), puts it on a uniform subset of the team's mons, reveals that many of each mon's
  moves uniformly (a Hidden Power as bare `hiddenpower`, as a gen-3 log names it), and its item and ability where the
  shape says. No game is wasted. Every kept game's team pair (p1, p2) is recorded (`games.npz`, its sha256 in the
  S-S manifest).
- **The reveal profile** (`python -m main.human_data.ss_rule profile --snapshot hr_spec_v1 --n 40000`):
  `~/gen3ai_archive/x59s/reveal_profile_v1.json.gz`, **sha256 `1529f5acff20f18d40943f95fb8fe7231c013eaa2814056a039f96ac7297dc3d`**:
  the shapes of all 35,554 sides of the 17,777 CANDIDATE battles (TRAIN dates, rated, a result, ≥ 11 turns, a
  ≥ 1300 seat) in a seeded 40,000-log sample (seed 20261026); 5.56 mons seen per side. **Candidate sides, not kept
  ones (a departure from REVIEW E's "matched to the kept human sides"):** the human rule acts on battles before it
  selects them; the kept sides are selected for revealing little (5.14 mons seen against 5.66 among the excluded),
  so their shapes would under-exclude.
- **What it does to the pool** (`ss_rule census`, descriptive, read before any S-S row was generated): a pool side
  is near on 58.7 % of simulated reveals; **16.6 % of team pairs pass, against the 16.8 % of candidate human battles
  the human rule kept**; 3 pool teams are never near, 5 always; the drawn teams' effective number is 553 of 719.
- **Also matched (m7):** a self-play game of < 11 turns is dropped, as the human rowset drops one.
- **Regenerated:** S-S seed **5910** (Amendment 1's unruled rows used 5900), parts 0–7 and the validation top-up as
  in Amendment 1, merged to Amendment 2b's rowset counts and shard sizes; registered with `checks register
  --require-panel-rule` (an S-S manifest without the rule is refused).
- **Δ_data per panel** (H, F, P72) is reported, descriptive.
- **Registered limits:** the reveal is simulated, not read off the game; mons are chosen uniformly (a lead reveals
  more in a real game); the profile comes from human games, which are shorter than P\*'s. The direction of the
  residual is not known.

### A2.2 The Kakuna-family audit runs FIRST, and its flagged accounts leave the rowset before training (B2)

This REPLACES §8.4 and the plan's A-6 re-read: no student is retrained, because the audit decides the rows.
Code: [`audit.py`](../../../../src/main/human_data/audit.py) and the Metamon-interpreter scorer
[`kakuna_audit_metamon_side.py`](kakuna_audit_metamon_side.py). CPU only.

1. **Universe:** every account of `hr_spec_v1.r1` (TRAIN and VALIDATION; every actor is ≥ 1300) with **≥ 10 kept
   seat-games**: 1,051 TRAIN accounts, 83.1 % of the TRAIN rows. The rest are UNAUDITED; their row share is
   reported. (REVIEW E proposed ≥ 20 games: 561 accounts, 67.9 % of rows; ≥ 10 reaches further at the same cost
   because of the next rule.)
2. **Games:** a seeded sample of 10 of each account's seat-games (seed 20261026, keyed per account).
3. **Per seat-game:** the mean log-probability of the human's action under (i) **Kakuna** (checkpoint 34, the anchors'
   pinned weights), over the decisions Metamon's own replay parser reconstructs from that seat's view, teacher-forced
   with the CPU attention `metamon_side.install_cpu_attention` installs; (ii) **P\***, over the rowset's labelled rows
   of that seat-game (the stored S-P label logits, softmax over the legal actions); (iii) **the HOM**, over the same
   rows (descriptive).
4. **Score:** `S = mean LL_Kakuna − mean LL_P*` per account (decision-weighted over its scored games; REVIEW D m7:
   against P\*, not the HOM, which trained on these accounts); `S_HOM` likewise, descriptive. An account needs ≥ 5
   scored games, else it is UNSCORED (its row share reported).
5. **Human reference:** the audited accounts with ≥ 1 chat line a game on average over their sampled games; if fewer
   than 100 qualify, the reference is every scored account (decided from the count alone, before any score is read).
6. **Flag:** `S` above the reference's 99th percentile (`np.percentile`, linear).
7. **The drop rule:** if the flagged accounts hold **more than 1.0 % of `hr_spec_v1.r1`'s TRAIN rows**, every row
   (TRAIN and VALIDATION) of every flagged account leaves the rowset (A2.3). At or below 1.0 % nothing is dropped and
   the flagged share is reported. (Without a fleet, ≈ 1 % of the reference sits above its own p99 by construction;
   one flagged heavy grinder, the profile a bot fleet has, crosses 1.0 % alone.)
8. **Controls:** Kakuna's OWN seats in captured P\* vs Kakuna games (X51), scored by the same pipeline and pooled 10
   games per pseudo-account: the audit has **POWER** iff their median mean-LL_Kakuna exceeds the reference's
   99th percentile of mean-LL_Kakuna. **NO-POWER** drops nothing, is reported as the headline of the audit, and leaves
   §8.4 as a registered limit guarded by A2.10. Other Metamon-family seats found in captures are reported,
   descriptive.

### A2.3 The quitter's degraded tail leaves; the rowset is re-derived (M1)

- **`hr_spec_v1.r2`** = `hr_spec_v1.r1`'s rows minus (a) every row of the quitting seat (forfeit or timer) with
  **`0 ≤ quit_dist ≤ 9`** (its last 10 decisions; r1 already dropped the last 2) and (b) the flagged accounts' rows
  when A2.2.7 fires; order kept, re-cut into 16,384-row shards
  ([`derive.py`](../../../../src/main/human_data/derive.py)). The evidence is REVIEW E's table (agreement with the
  quitter FALLS over its last ~8 kept decisions while it RISES for the forfeit winner and both KO seats), which the
  data forced, not an outcome. r1 holds 108,740 TRAIN rows with `2 ≤ quit_dist ≤ 9` (8.2 %).
- **S-P's labels are r1's labels on the surviving rows, unchanged** (each row's matched uniform stays keyed by r1's
  rows manifest). S-S is cut to r2's counts (A2.1). The ≤ 10-turn battle filter stays.

### A2.4 S-P is "P\*-restricted-to-human-support"; an unrestricted S-P0 arm is added (M2)

- **S-P's teacher is named for what it is:** P\*'s preferences INSIDE the human's realised whole-game support, not an
  independent P\*. REVIEW E's measurements stand as its definition (`results/review_e_checks.json`, `rows`): P\*'s
  probability of the human's action 0.364 unrestricted → 0.450 restricted; S-P's label equals the human's on 45.1 %
  of labelled rows; P\*'s top-1 lies outside the consistent set on 16.8 %; the restriction removes 14.4 % of P\*'s
  mass from moves and 3.9 % from switches; switch share human 31.4 %, S-P 37.2 %, P\* 34.0 %; the consistent set is
  the whole legal set on 25.5 % of rows.
- **Δ_labels is read accordingly:** OWN-LABELS-BETTER is conservative (it holds against a handicapped P\*);
  HUMAN-LABELS-BETTER is reported as "human labels beat P\*-restricted-to-human-support" and **routes H-BC only if
  S-P learned its teacher**: the S-P students' yardstick on the S-S validation rows (P\*'s labels on P\*'s states) is
  below the S-H students' (Welch t across the 10 students, UB < 0). (REVIEW E proposed "within a margin of S-S's";
  no margin can be set without data, and S-S's states differ from S-P's, so the same-states comparison against S-H
  is the falsifier with no guessed number.) Δ_states mixes state coverage, the restriction and M3's signature; it
  stays descriptive.
- **S-P0 (descriptive, routes nothing):** the SAME rows, one action of P\* at T = 1 over the LEGAL actions
  (unrestricted), drawn with the same matched uniform from the stored S-P logits (`derive`'s `labels_pstar0`; no new
  P\* forward). Its labels name drawn options (16.8 % of rows have P\*'s top-1 outside the consistent set), so it
  measures what the restriction costs, not a clean P\*. **Cost:** labels free; 5 students ≈ 1 GPU-h (ESTIMATE),
  trained AFTER the 15 primary students so it never delays them; ≈ 20 lane-h of Kakuna games at the full units,
  played AFTER the primary read. Read: Δ_restriction = S-P0 − S-P with §4's interval, and its yardsticks.

### A2.5 The drawn-move signature against S-S is a registered limit, and FR-2 gates the closure (M3)

S-H's and S-P's own-side inputs carry C-GAME draws that are never a target (≈ 1.1 listed moves an active mon); S-S's
own side is true. This can only be learned as "some listed options are never chosen", which play never rewards, so
it is expected to COST S-H and S-P against S-S, never to help them. **FR-2 is read before Δ_data is reported** (the
read code refuses a verdict without its class), and **if FR-2 reads ARTEFACT, a SELF-PLAY-DATA-BETTER closes
nothing**. FR-2 was not run in this amendment (≈ 1 lane-h and ≈ 0.2 agent-day, plan §2.2).

### A2.6 Floor references and the UNINFORMATIVE class (M4)

- **Floor references (CPU):** `--our-side bot:random` (random legal) and `--our-side bot:aggressive` (the max-damage
  heuristic: the highest effective-damage move, never a voluntary switch) against Kakuna on the SAME X51 symmetric
  configs, units 0–3 of H, F and P72 (240 games each), played before the floor pilot.
- **UNINFORMATIVE:** every arm's U rate is ≤ the better floor reference's + 5 pp (a tie within 1e-9 counts as within),
  or the floor fallback fires. Every class is still reported; they close nothing and route nothing beyond "H-TEAMS
  first". The ≥ 1500 follow-up (A2.7) does not fire under it.
- **VOID (learned nothing):** a student whose final own-label validation NLL is not at least 0.1 nat below the uniform
  NLL over the legal actions (exactly 0.1 below is VOID) did not learn: a bug or a recipe failure. Recorded by
  `export` (`learned_guard`).

### A2.7 The closure is scoped, and a ≥ 1500 follow-up is registered conditionally (M5)

- SELF-PLAY-DATA-BETTER (with A2.5's gate and §5's conditions) closes **"1300+ ladder data, at this mix and volume"**,
  never human data in general.
- **The follow-up fires when Δ_data is neither HUMAN-DATA-BETTER nor UNINFORMATIVE.** Three arms at N = r2's
  ≥ 1500-actor TRAIN rows (recorded in 2b), 5 seeds each (5901–5905, the same inits): **S-H≥1500** (those rows);
  **S-H-mix** (a seeded ACCOUNT-level subsample of r2's TRAIN rows to exactly N: accounts permuted by seed 5930,
  taken whole in that order, the last one trimmed to N); **S-S cut** (S-S's first N TRAIN rows). The recipe as §3
  at 3 epochs of N rows; validation as in r2. Read in-house (`main.h2h`, 400 games a panel on H, F, P72 a pairing):
  S-H≥1500 − S-H-mix (rating at fixed volume) and S-H≥1500 − S-S cut, each with §5's classes (A2.8 included).
  ≈ 0.7 GPU-h and ≈ 8 lane-h (ESTIMATE). Its rowset digests are registered in an amendment before it trains.

### A2.8 NO-DIFFERENCE-AT-4PP (M6)

Each contrast gains a class between the two directions and the undecided one: **NO-DIFFERENCE-AT-4PP** iff the whole
95 % interval lies inside ±4 pp (the registered 80 % MDE; a bound within 1e-9 of ±4 pp takes the undecided class).
For Δ_data it is sent to the owner as "human data is neither better nor worse than our self-play data by more than
4 pp at this volume"; it closes nothing (closure needs SELF-PLAY-DATA-BETTER). For Δ_labels it routes as UNCLEAR does
(H-TEAMS first, with the D1 contrast): a ≤ 4 pp label effect is below X60's own MDE. The undecided classes keep their
names (NOT-RESOLVED, UNCLEAR) and now mean "not resolved".

### A2.9 The CUDA path's pass criteria and the window (M7)

- **[`cuda_preflight.py`](../../../../src/main/human_data/cuda_preflight.py)** runs once, inside the lease, before
  the first counted student; any FAIL stops the window. Its 8 checks: (1) seed 5901's `init_sha256` on CUDA equals
  the CPU-built digest; (2) S-H 5901 for 200 steps on CPU and on CUDA from that init: step-0 validation NLL within
  1e-5, the losses at steps 100 and 200 within 1e-3, the parameters' relative L2 distance at 200 below 1e-3; (3) TF32
  off for matmul and cuDNN and the fp32 matmul precision `highest`, read inside the loop; (4) resume parity: 100 steps
  then a resume to 200 reach step 200, the uninterrupted run's learning rate and its loss within 1e-3, with the
  optimiser state on the device; (5) export: the bare-load check passes and the CPU logits agree with the CUDA logits
  within 1e-4 on 256 validation rows; (6) ms a step, peak host and device memory, extrapolated to the window;
  (7) the retry rule (below); (8) the training loss is below the uniform NLL by step 1,000.
- **[`window.py`](../../../../src/main/human_data/window.py)** replaces §9's loop: the 15 primary students in
  registered order, then the 5 S-P0 students, ≤ 3 at once; each chunk is `gpu_lock.sh mem_cap.sh 8 <PY> -m
  main.human_data.student train … --device cuda --budget-min 45`; **only exit 3 continues a student**, exit 0
  finishes it, any other exit stops the window and reports. The trainer refuses rows or labels whose manifest is not
  the registered one (`--registration`, REVIEW E m4).

### A2.10 When Kakuna and the in-house meter disagree (M8)

A HUMAN-DATA-BETTER or HUMAN-LABELS-BETTER read on Kakuna whose in-house point estimate (the matching `main.h2h`
contrast) is ≤ 0 (a tie within 1e-9 counts as ≤ 0) is reported as **KAKUNA-SPECIFIC**: it opens no lever and routes no
H-BC. The in-house head-to-heads are therefore played for every read, not only under the floor.

### A2.11 Minors (reporting rules)

m1: the effective labelled rows are reported beside each arm (S-H / S-P carry set labels on hidden rows, S-S none).
m2: each arm's 5-student SD is reported beside σ̂; the 0.028 false-positive rate at a seed SD of 2–3 pp stands as
stated. m3: the floor pilot releases only its boolean until the full read, and the read lists the teams units 0–3
cover. m4: fixed (A2.9). m5: an arm whose validation NLL rises over its last epoch is reported. m6: the yardsticks are
reported beside every read. m7: S-S's turn-length and legal-action profile is reported beside the human rowset's.

All of A2's read rules are code before any data: `x59s_read.classify` (A2.8), `x59s_read.verdict` (A2.4–A2.7, A2.10).
**Cost now:** ≈ 3.7 GPU-h for 20 students (ESTIMATE: Amendment 1's ≈ 3 GPU-h scaled by r2's rows, plus S-P0), and the
conditional ≈ 0.7 GPU-h.

---

## Amendment 2b (2026-10-26, before any student trained): the inputs as built under Amendment 2

Built under Amendment 2's rules (on main at `2e7b0862` before any flag was read or any S-S row was regenerated), on
the CPU at nice 19 on the agent cores under `mem_cap.sh`. No student has trained; no game has been played.

### The Kakuna-family audit (A2.2): the drop rule FIRES; no account plays like Kakuna

[`results/kakuna_audit.json`](results/kakuna_audit.json) (the full per-account table is in the archive, its sha256 in
the file); Kakuna checkpoint 34 at hub revision `ac8e88e7`, weights sha256 `7e7474d5…`, the hub offline.

| quantity | value |
|---|---:|
| audited accounts (≥ 10 kept seat-games) / their sampled seat-games / scored | 1,073 (1,051 TRAIN, 22 VAL) / 10,730 / 10,723 (7 Metamon parse failures) |
| unaudited TRAIN rows (accounts with < 10 games) / unscored accounts | 16.9 % / 0 |
| reference | every scored account (1,073): 97 accounts met the chat rule, < 100 |
| account score S = LL_Kakuna − LL_P\*: median / p99 (the flag bar) / max | −0.878 / **−0.2275** / +0.094 |
| flagged accounts / their r1 TRAIN rows | **11** (all TRAIN) / **23,046 = 1.73 %** of 1,330,809 |
| **the drop rule (> 1.0 %)** | **FIRES**: the 11 accounts leave the rowset |
| POWER control: Kakuna's own seats (158 pseudo-accounts of 10 games) above the human p99 of LL_Kakuna (−2.03) | **158 / 158 (median −0.80): POWER** |
| other Metamon seats above that p99 (descriptive): PastaMittens / LargeRL / SmallRL / LargeIL | 90 / 90 / 13 / 47 / 28 / 158 / 0 / 45 |

**What the flag found (read with the rule, not instead of it).** No audited account plays like Kakuna: the most
Kakuna-like human scores LL_Kakuna −1.81, below every Kakuna pseudo-account (lowest −1.46) and every PastaMittens one.
The 11 flagged accounts score high on S because P\* fits them badly, not because Kakuna fits them well (their
LL_Kakuna −1.81 to −2.75, inside the human range); one of them (549 games, 15,215 rows) carries the rule alone
(0.59 % without it). By construction of the all-accounts reference, `S > p99` flags ≈ 1 % of accounts with or
without a fleet, so the 1.0 % test reads their volume. The rule was registered, so it is applied: dropping 1.7 % of
rows costs little before training, and deviating after reading the scores would be the worse error. The audit has
power against Kakuna and PastaMittens; it would not separate a SmallRL- or LargeIL-like bot from a human.

### `hr_spec_v1.r2` (A2.3), registered in the rowset registry

Rows manifest **sha256 `2a413c85e2109553b4f930d85f369625a5daf77a7e1103afeb01e962ec0332b8`**
(`~/gen3ai_archive/x59s/hr_spec_v1.r2/`; parent r1 `e9e0df42…`; drop list sha256 `4d53e2e3…`).

| quantity | value |
|---|---:|
| **TRAIN rows / VALIDATION rows** | **1,201,538 / 29,904** (74 + 1 shards) |
| dropped from r1 TRAIN: the quitter's tail (`0 ≤ quit_dist ≤ 9`) / the flagged accounts (beyond the tail) | 108,740 / 20,531 |
| dropped from r1 VALIDATION: the quitter's tail / flagged | 2,198 / 0 |
| hidden-move rows (set labels) | 92,435 (7.7 %) |
| accounts TRAIN / VALIDATION / in both | 3,419 / 70 / 0 |
| actor 1300–1499 share / **≥ 1500 TRAIN rows (A2.7's N)** | 77.5 % / **270,260** |
| endings (rows): KO / forfeit / timer / tie | 45.9 / 50.6 / 2.9 / 0.6 % |
| switch labels among labelled rows | 30.7 % |
| **the step budget a student** (`3 × Σ ceil(buffer rows / 512)`) | **7,041** (r1: 7,800) |

- **S-P labels** (`labels_pstar`, manifest sha256 `2512c8a7…`): r1's, carried unchanged (a check of 5 shards:
  81,920 / 81,920 rows equal r1's label by row id).
- **S-P0 labels** (`labels_pstar0`, manifest sha256 `8f511ca5…`): on 5 shards, equal to the human's action on 36.6 %
  of labelled rows; outside the consistent set on 18.0 %; switch share human 30.0 %, S-P 36.6 %, S-P0 33.9 %.

### S-S regenerated under the panel rule (A2.1)

Manifest **sha256 `3f81f71c5ece664e1983294343d8013e0cb8b8eaaa9e80305067166e41d26ff4`** (`~/gen3ai_archive/x59s/ss_rows_v2/`):
seed 5910; parts 0–7 (each 180,6xx rows, sized on r1's counts so either rowset fits) and the validation top-up part 8;
merged to r2's exact counts and 74 shard sizes. 111,488 team-pair draws, 92,750 redrawn: **16.8 % of pairs passed**
(the human rule kept 16.8 % of candidate battles); 12 games of < 11 turns dropped; 15,069 games used, every one of
the 719 pool teams used at least once, effective number 558.4; the pairs in `games.npz` (sha256 `2087b483…`). A merge
to r1's counts, made as a side effect of part 7's job before r2 existed, was set aside unregistered
(`ss_rows_v2/superseded_r1_merge/`).

### Registration and readiness

[`results/registration_a2.json`](results/registration_a2.json) (`checks register --labels0 --require-panel-rule`):
**PASS**, no problems: rows `2a413c85…`, S-P labels `2512c8a7…`, S-P0 labels `8f511ca5…`, S-S `3f81f71c…`, reveal
profile `1529f5ac…`, P\* `814823c5…91ed`. It REPLACES Amendment 1's `registration.json` for every student: the trainer
refuses any other input given `--registration results/registration_a2.json`, and a student trained on Amendment 1's
inputs is VOID.

**S-H, S-P, S-S and S-P0 are READY to train** under a GPU lease, in this order: the CUDA pre-flight (A2.9) must PASS
first, then `window.py` (15 primary students, then 5 S-P0). **≈ 3.6 GPU-h** (ESTIMATE: Amendment 1's unmeasured
≈ 3 GPU-h for 15 students × 7,041 / 7,800, plus 5 S-P0; the pre-flight measures ms a step). Before any read: FR-2
(A2.5) and the floor references (A2.6), both CPU.

## Amendment 3 (2026-10-26, before any counted student trained): the pre-flight's export FAIL, its fix, the measured cost

No counted student has trained and no game has been played; the scratch students of the pre-flight below are not
students of this read. **No decision rule, VOID guard, pass criterion, recipe or registered input changes.**

### A3.1 The CUDA pre-flight (A2.9) HELD the window: check 5 `export` FAILED

Run at 07:00–07:13 on 2026-10-26 under the lease (RTX 3080 Ti, torch 2.8.0+cu126; result and log in
`~/gen3ai_archive/x59s/results/cuda_preflight.{json,log}`). PASS: `init` (equal digests), `parity` (step-0 NLL and the
losses at 100 and 200 equal, parameters 4.9e-7 apart), `precision` (TF32 off, `highest`), `resume`, `learns` (loss
1.454 at step 1,000 against the uniform 1.847), `cost`. **FAIL: `export`**, "VOID: the anchors harness's bare load
does not reproduce the student's logits". The rule worked as registered: nothing trained.

**Root cause (reproduced on the CPU from the pre-flight's own exported zip):** the anchors harness's `bare` load
(`agents.model.snapshot.load_checkpoint_strict`) returned the policy in TRAINING mode. The weights, the
architecture, the config and the dtype were identical (every tensor equal; the reload through
`main.policy_spectrum.reader` was bit-equal). Only in eval mode does `nn.MultiheadAttention` take its fused fast
path (the first module whose output moved: the opponent-belief pool's `hidden_opp_belief.decoder.self_attn`), a
different kernel: the logits differed on 18 of 256 validation rows by at most 4.8e-7 (|logit| ≤ 2.8),
with no argmax change; after `.eval()` they were bit-equal. The CPU test of `export` passed because its synthetic rows
were all zeros, on which the two kernels agree.

**The fix (the class, not the check):** `load_checkpoint_strict` returns the policy in eval mode, and so does the
anchors harness's `foreign` loader, so every reader (the anchors harness, `play.py`, the prober, the meters) reads
the function the rollout's opponents and every other reader already read. `export`'s check is unchanged: the bare
load must still give the student's logits BIT FOR BIT. Tests that fail on revert:
`src/main/anchors/reader_eval_mode_test.py` (both anchors loaders return an eval-mode policy) and
`student_test.py::test_the_anchors_bare_load_reads_the_students_function_on_real_rows` (real fixture-battle rows and
seeded perturbed weights; it asserts its precondition that the two modes' logits differ there; on revert it fails
with the pre-flight's exact error). On the FRESH init the two modes agree on every real row tried (952 rows,
archive and fixture), so the fault shows only once the weights have moved. The window re-runs the WHOLE pre-flight
(all 8 checks) at the commit that carries this amendment; any FAIL holds it again.

### A3.2 The measured cost replaces §9's and Amendment 2b's estimate

| quantity | measured (pre-flight, 2026-10-26) | was |
|---|---|---|
| ms a step on the GPU (batch 512, fp32, 2 threads) | **208.6** (`train_s` / steps over 1,000 steps; step time only, shard loads and evaluations excluded) | unmeasured |
| steps a student (3 epochs of r2) | 7,041 | 7,041 |
| GPU time a student | 7,041 × 0.2086 s = **24.5 min** | ≈ 11 min |
| 20 students, one at a time | **8.16 GPU-h** (15 primary 6.12, 5 S-P0 2.04) | ≈ 3.6 GPU-h (ESTIMATE) |
| device memory a student (peak allocated) | 1.30 GiB | — |
| host memory (peak RSS of the pre-flight process) | 7.27 GiB | under the 8 GiB cap |
| CPU, for comparison | 1,841 ms a step (the pre-flight's CPU arm) | ≈ 2.5 s (§9) |

### A3.3 The window plan

- **Concurrency 3** (A2.9's ceiling, `window.py`'s default): 3 × 1.30 GiB of device memory plus three CUDA contexts
  fits the 12 GiB card. Host memory ≈ 3 × 7.3 = 22 GiB; **each chunk's `mem_cap.sh` cap rises from 8 to 10 GiB**
  (`window.MEM_GB`): 7.27 GiB peak sat 0.73 GiB under the old cap, and an OOM kill (exit 86) stops the window by
  A2.9's rule. 3 × 10 GiB leaves 34 GiB of the heavy slice's 64 GiB to the box's other capped jobs; a slice-wide
  OOM would stop the window the same way (resumable).
- **Expected wall time (ESTIMATE):** the slow-down factor *f* of a step at 3 at once is unmeasured (208.6 ms for a
  2.6M-parameter network at batch 512 reads as launch- or host-bound, so *f* near 1 is plausible, UNVERIFIED). The 20
  students stream through 3 lanes in 7 rounds of 24.5 min × *f*: **≈ 2.9 h at *f* = 1, ≈ 4.3 h at *f* = 1.5, ≈ 8.6 h at
  *f* = 3** (the card serialises; no better than one at a time), plus ≈ 13 min for the pre-flight re-run and the
  shard loads and evaluations the step time excludes. The primaries finish after 5 rounds (≈ 2.0 h at *f* = 1). The
  first round's `ms/step` lines (every student logs it every 100 steps) measure *f*; it is reported, and decides
  nothing.
- **The lease asked for: ≤ 9 GPU-h** (the *f* = 3 bound with the pre-flight), expected ≈ 3–4.5 h.
- **S-P0 still runs in the same window**, after the 15 primaries (the registered order, so it never delays a primary
  student). If the lease must end first, the window stops between chunks and S-P0 resumes in a later lease from its
  `state.pt` (`window.py --dir` with the same directory); S-P0 is descriptive and no rule reads it.

## Amendment 4 (2026-10-26, after training, before any game of the read): how the read is played

The 20 students trained at Amendment 3's commit (window `PASS` 23:04, `~/gen3ai_archive/x59s/students/`). **No game
of this read has been played and no student's Kakuna, in-house or floor result exists.** Their final validation NLLs
were seen (the window's table); no rule reads them. **No decision rule, class, interval, VOID guard or registered
input changes.** This amendment says what machinery plays the read, where it differs from what §4 assumed, and the
order. Code: [`x59s_ops.py`](../../../../src/main/human_data/x59s_ops.py) (export, identity, floor references,
in-house) and [`rules_x59s.py`](../../../../src/main/reads/rules_x59s.py) (the `main.reads` rules `x59s_floor` and
`x59s`, which call `x59s_read` unchanged); specs [`read_spec_pilot.json`](read_spec_pilot.json),
[`read_spec_full.json`](read_spec_full.json).

### A4.1 One play tree, one engine

- **Every game of the read** (Kakuna, floor references, in-house) is played from ONE clean detached checkout at
  **`fca58b24d`** (main, 2026-10-27), so ONE Rust sim build plays the whole read (`designs/tools/reads.md` §4: one
  engine per read, named at registration). **The engine is main's, `port=67a272e7796bd85e/93;env=d79c3a74b241a75b/128`**
  (`main.reads.engine.engine_at`; equal at `b712c46d1`, the tree first built), NOT X51's E0
  (`05b106e2…/84` / `dfd24aec…/119`). Every unit stamps it (`cell.engine_stamp`, `20875e96`) and the record carries it.
  §4 named "a detached checkout of this registration's commit"; that commit predates `141f36f7f` (every checkpoint
  reader in eval mode, A3.1) and `20875e96` (the engine in the regime id), so the tree moves to a commit carrying both.
  The identity unit (A4.3) is what certifies that this tree reproduces X51. The specs register `pin: fca58b24d`; this
  amendment's own changes are off the play path. (`check-spec` at its commit reports 15 play-path files differing
  from the pin: all are main's later Rust commits under `src/rust_sim/src/`, none of this read's; the read plays at the
  pin, never at HEAD.)
- **Every registered contrast is student against student on this one engine** (Δ_data, Δ_labels, Δ_states, the
  in-house head-to-heads, the floor references). **Q-H10 (each arm minus P\*'s X51 rows) is CROSS-ENGINE** (X51 played
  on E0): it is reported as a SECONDARY, descriptive line, guarded only by the identity unit, and routes nothing.
- **Export** (`python -m main.human_data.x59s_ops export`, `student export` per student; run 2026-10-27 at this
  amendment's code, the save path identical to the pin's): each
  student into its OWN directory `~/gen3ai_archive/x59s/read/students/<label>/` (`state.pt` symlinked; the training
  archive is not written), with both yardsticks; `students.json` records every sidecar, VOID 3 (one `init_sha256` per
  seed across the arms; 7,041 steps each), Amendment 2's learned-nothing guard and m5's flag. `export`'s own
  reload and anchors-`bare` bit-equality checks are VOID 4.

### A4.2 The Kakuna cells: `main.reads`' anchor lane

- Each unit is a `main.x35.kakuna` unit: Kakuna ckpt 34, greedy vs greedy, CPU, 20 role-balanced games,
  `--team-seed = --seed-base =` the unit seed: X51's unit exactly.
- **Configs.** The lane plays its registered panel cells (`heldout72_hpfix_kakuna.json`, X37a's `forum72` and `p72`)
  and `anchors_home72_kakuna.json`, not `anchors_x51_sym_<panel>.json`. They differ from X51's ONLY in the directory
  of our byte copy of each panel (both content-digest-checked against the panel registry before every unit) and in
  X51's `reference_team_dirs` (read only by the overlap report, `main.anchors.team_overlap`). Every student game
  therefore still twins X51's P\* game of the same panel, seed, half and index.
- **Counting** is X51's (`main.anchors.results.unit_counts`). **Retries:** the lane makes 2 attempts a pass
  (`kakuna.MAX_ATTEMPTS`); §4's "re-run at most twice" is one `resume` pass for a unit still uncounted (3 attempts at
  most). VOID 7 is read by the rule (a cell with > 1 uncounted unit reported; two such cells VOID the primary).

### A4.3 VOID 5, the identity unit (before any student game)

A standalone `main.x35.kakuna` state at the play tree: P\* (`814823c5…91ed`) on home72, unit seed 0. **PASS iff its 20
games equal X51's `pstar_home72` unit 0 game for game on (half, index, our team, result, turns)** (`x59s_ops
identity`; stricter than §4's "winner and turns"). Played and read before any student game. A FAIL is VOID 5: the
read stops and is reported, nothing is amended after it.

### A4.4 VOID 6, the seeded repeat

In the pilot state S-H 5901 also plays under a second label, `s_h_5901_rep` (the SAME bytes, pinned by sha256), on
forum72 unit seed 0. **PASS iff that unit equals `s_h_5901`'s forum72 unit 0 game for game** (`x59s_floor`).

### A4.5 The order

1. **The identity unit (A4.3), then the floor references (A2.6, A4.8)**.
2. **The floor pilot** (`read_spec_pilot.json`, state `~/gen3ai_archive/x59s/read/pilot`): groups `repeat` and
   `pilot` (the 15 primary students × H, F, P72 × unit seeds 0–3: 180 units, 3,600 games). Rule `x59s_floor`: the floor
   FIRES iff every arm's mean U rate is below 0.15 (a mean within 1e-9 of 0.15 does not fire). Its decision carries
   the boolean ONLY (A2.11 m3): no arm rate is written into it or read by the operator until the full record. The
   teams units 0–3 cover are listed in the full read's report (m3).
3. **FR-2** (A4.7) is run and its class reported to the orchestrator before any contrast is computed.
4. **The rest of the Kakuna read**, if the floor does not fire: `read_spec_full.json` (state
   `~/gen3ai_archive/x59s/read/full`): group `pilot` IMPORTED from the pilot state (sha-checked, never re-played) and
   group `rest` (unit seeds 4–19: 720 units). Its rule `x59s` is the whole registered verdict: `x59s_read.read` on
   each student's 20 units a panel, the pilot's floor boolean, `x59s_read.inhouse`, the floor references, the
   yardsticks and FR-2's class through `x59s_read.verdict`, refusing a verdict on VOID 3, 5, 6 or 7, a missing FR-2
   class, or short in-house or floor cells. The artefacts it reads are files of this folder's `results/` (and the
   archive's `students.json`), their sha256 recorded in the record. If the floor FIRES, the same rule is read on the
   imported pilot group alone (`read_spec_floor.json`, written then from this text; the in-house contrasts decide, §5).
5. **The in-house head-to-heads** (A4.6), played beside step 4.
6. **After the verdict, as the CPU allows (descriptive):** home72 units 0–9 of the 15 primaries, and S-P0's U units
   0–19 and home72 0–9 (A2.4's Δ_restriction). §4 placed home72 in the primary play; it moves after the U units
   because no rule reads it and the CPU (A4.9) does not cover both before the verdict.

### A4.6 The in-house head-to-heads

`main.h2h play --player <S-H_k> --opponent <S-S_k | S-P_k> --panel <heldout72_hpfix | forum72 | p72> --pairs 200
--seed k` for k = 5901–5905 (30 cells, 12,000 games), CPU, the default schedule key, the ledger outside `models/`
(`x59s_ops inhouse`). **A pairing's rate** = the equal-weight mean over the three panels of h2h's mirrored-pair score of
S-H (a draw counts ½; §4's "win rate" is that score); the contrast = the mean of the 5 pairings − 0.50 with a t
interval, df 4 (`x59s_read.inhouse`, unchanged).

### A4.7 FR-2, operationally (A2.5; the class rule is `design_human_replay_lever.md` §2.2's, unchanged)

R1's trace extraction (`~/gen3ai_archive/x50_r1b/traces/`, the sentinel and bot seats as viewer and actor), each
decision read twice: its C-GAME-completed row (mode 2) and its TRUE-team row on that seat's REAL request (mode 1),
both labelled with that actor's own action; only decisions with both rows, a usable label and ≥ 2 legal actions. For the HOM
(`~/gen3ai_archive/x50_r1b/hom/model/`) and P\* (`814823c5…91ed`): the policy NLL of the label (masked softmax over
the legal actions, eval mode). Twin gap = mean NLL(mode 2) − mean NLL(mode 3), decision-weighted; **D = gap(HOM) −
gap(P\*)**; a battle-cluster bootstrap, 2,000 resamples, seed 50, battles resampled jointly for both models and modes,
percentile 95 % CI. **ARTEFACT iff LB > 0.05 nats; NO-ARTEFACT iff UB < 0.05; else UNCLEAR** (a bound within 1e-9 of
0.05 takes UNCLEAR). Its record is `results/fr2.json`.

**As built (before FR-2 has been run on any data):** `python -m main.x50_r1.read_fr2` pairs the two rows by
(battle, seat, decision index); they pair 1:1. **R1's existing extraction reads only OUR trainee's seat** (every one
of its 151,234 mode 2 / 3 rows has the X26 trainee as viewer), not the sentinel's or bot's that §8.5 registers, and
its mode-3 rows carry requests SYNTHESISED from public lines, not real ones. FR-2 is therefore read on a NEW
opponent-seat extraction of the same traces (`main.x50_r1.run traces --seat opponent`: the sentinel or bot as viewer
and actor; the traces carry that seat's own requests, so its true rows are on REAL requests, mode 1, `read_fr2
--true-rows full`, the default), so the registered definition stands unchanged. Preconditions, read before FR-2:
G-ENC PASS at the extraction's commit; refused seats ≤ 1 % of seats read (the extraction's stop rule); the mode 1 /
mode 3 own-action differ count reported. Seats with a refused choice have no real-request rows and drop from the
pairing (counted). The trainee-seat read (`--viewer trainee`) is not FR-2: there P\* is an ancestor of the
actor (steps ≥ 360M), which changes what its twin gap means; if computed, it is recorded `registered: false`.

### A4.8 The floor references

`x59s_ops floor`: the Kakuna unit's command with `--model <zip>` replaced by `--our-side bot:random` or
`--our-side bot:aggressive` and nothing else changed; units 0–3 of H, F, P72 (240 games each); counted by
`unit_counts`. A bot's U rate is the equal-weight panel mean; "the better floor reference" (A2.6) is the larger.

### A4.9 Reporting rules made concrete

- **Q-H10** (SECONDARY, cross-engine, A4.1): per arm, per (panel, unit seed), the mean over its students of the unit's win rate minus
  P\*'s X51 unit's; equal weight over panels; a normal interval over the unit seeds within each panel.
- **m1:** S-H and S-P train on 1,201,538 rows of which 92,435 carry a set label (hidden moves); S-S on 1,201,538 rows,
  none. **m3, m5, m6** come from the record (`descriptive`), **m7** from the rows.
- **CPU:** the floor pilot measures units a lane-hour on this box; the projected finish is reported then, and any cut
  the budget forces is a further amendment before the affected units' data is read.

## Amendment 4b (2026-10-27 03:00, before any student game): the pin moves to the commit that lands the rules

`main.reads` loads a spec's rule at `plan` in the PLAYING checkout, and `fca58b24d` (A4.1) predates the `x59s_floor` /
`x59s` rules, so it cannot play these specs. Per the registration flow (`designs/tools/reads.md` §1: the pin is the
commit that lands the code the read plays), **the pin is `0534e90a6`** (Amendment 4's own commit on main), and the
specs are re-registered there. **The read's ONE engine is therefore `port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129`**
(`engine_at(0534e90a6)`; main's Rust moved after `fca58b24d`), still not X51's E0, so Q-H10 stays SECONDARY. The
identity unit (A4.3, VOID 5) is re-played on this tree before any student game; the earlier PASSes (`b712c46d1` and
`fca58b24d`, engine `67a272e7…/93`) are recorded and certify nothing about this engine. The floor references, FR-2's
opponent-seat extraction and every student game run from the tree at `0534e90a6`. Nothing else changes.

## Amendment 4c (2026-10-27, before FR-2 is run on any data): G-ENC's drift and FR-2's battle exclusion

FR-2's precondition G-ENC (A4.7) FAILED at the pin `0534e90a6`: `equal 2095, DIFF 11, short_by_one 9, skipped_refused_choice
191` (`~/gen3ai_archive/x50_r1b/traces_oppseat_genc.json`). R1's PASS at `2bbdffa7d` had the same 9 / 191 and no DIFF.
**Diagnosis (no FR-2 number was computed):** every DIFF is one cell of one active-mon volatile (`trapped`, `snatch`,
`destinybond`, `endure`) that the X26 training reader left at 1 after Showdown ended it silently, and the current
reader clears it: `d8d674b99`, `gen3_volatile_silent_ends_v1` (obs semantics 151 → 162). 27 rows of 465 compared
in the 11 battles, each one cell, always recorded 1 / read 0, at a point the protocol shows the effect ended; no other
cell or mask differs. This is reader drift toward correctness, not a reader defect.

**The registered exclusion:** a battle whose stream holds `|-singleturn|…|Endure` (or `move: Endure`), `Snatch` or
`Helping Hand`, `|-singlemove|…|Destiny Bond`, `Grudge` or `Rage`, or `|-activate|…|trapped` leaves FR-2 for BOTH
seats and both modes (33 of 2,306 traces flag; every DIFF battle among them). The precondition becomes: G-ENC over the
kept traces reads no DIFF (`short_by_one` stays tolerated, as R1's). The exclusion list and its sha256 enter
`results/fr2.json`. **Registered limits:** G-ENC certifies the trainee's view only (the opponent seat has no recorded
states); `d8d674b99` also changed the in-model damage operator's held-item read (`gen3_op_held_item_v1`), which both
models meet alike in both modes (unmeasured, expected to cancel in D); FR-2 reads both models at obs semantics 162,
newer than either trained on.

**Built (same day, still before FR-2 is run):** `main.x50_r1.fr2_exclude` writes the list
(`~/gen3ai_archive/x50_r1b/fr2_exclude.json`, sha256 `1c6f1ef8a14f27d08dfbbdc3df355f8d12d5114d48f31a11d1fc721e45dd79f7`:
33 of 2,306 battles, 22 Destiny Bond, 6 trapped, 3 Snatch, 2 Endure); `genc --exclude` and `read_fr2 --exclude` apply
it, and `read_fr2` records a `class` (not `class_unregistered`) only with that exact list. **G-ENC with the exclusion
PASSES:** equal 2,077, short_by_one 9, skipped_refused_choice 187, excluded 33, no DIFF
(`~/gen3ai_archive/x50_r1b/traces_oppseat_genc_excluded.json`). FR-2 runs from a checkout of the commit that lands
this code; its reader, encoder and Rust sources equal the pin's (`src/rust_sim`, `src/main/live`, `src/main/x50_r1`
before this change and `src/agents/observation` are unchanged between `0534e90a6` and it).
