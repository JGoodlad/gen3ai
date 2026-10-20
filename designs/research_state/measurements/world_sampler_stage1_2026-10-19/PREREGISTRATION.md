# PRE-REGISTRATION — the WORLD SAMPLER's STAGE 1: the sampler ablation with playouts (2026-10-19)

**Status.** Committed with the code that runs it BEFORE any arm's world or playout on a registered state exists. A rule
below changes only by a dated AMENDMENT appended here before the outputs it touches exist. The design is
[`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6.2; this file fixes everything §6.2 leaves open
and folds in the builds of 2026-10-19 (F2's legality rule, F1 tier A's `dice="rolls"`).

**What existed when this was written** (disclosed; none of it a stage-1 output):
- the corpus backoff's α, fitted on the LADDER corpus alone (§3.3; `fit_alpha.py`, `alpha_fit.json`);
- both corpora normalised and checked by Showdown's validator (`corpus_s1.py`: 22,813 ladder and 2,424 forum teams, 0
  dropped);
- an ENGINEERING SMOKE on 2 states OUTSIDE the 287 (interim LEAN states whose banked F build was an exclusion; their
  rows live in `~/gen3ai_archive/world_sampler_stage1_smoke/`): every arm built and played (one state built in every
  arm, the other excluded in 6 of 8 and built in NS; every F_rolls world replayed with its force table). Read for
  crashes, defects, row shapes, the CRN teeth and cost only; no IMP was computed;
- THE IDENTITY TOOTH (§7) on 12 of the 287 states (ids sha256 `49c183ba…`): the banked F worlds re-played at the play
  checkout, compared for equality with the banked 225M F rows: **12 / 12 identical**. Only the count was printed;
- the play checkout's own Rust test of the playout tool's roll table (`search_rolls_test`, 2 / 2 pass at `13ffd55c`);
- C_pool's pool packed at X26's pin (`pool_pin.py`: 719 teams, sha256 `4083c86b…`, Tier 1's value); the builder commit
  packs 34 of the 719 differently (§3.2 item 7).

**The question** (design §6.2). Does any candidate sampler put enough weight on the truth to make honest search pay,
and does stage 0's exact-world mass predict it? Stage 0 found that the network already holds the never-seen SPECIES
(0.31 of its mass on the exact set at 225M) while no source gets the seen mons' hidden MOVES right more than 3.5 % of the
time, so a world exact on both is drawn ≤ 0.25 % of the time against the ≈ 0.2 the dose curve asks. Stage 1 asks whether
PARTLY right worlds carry the value anyway.

**Vocabulary.** IMP = decision improvement: the win probability gained per decision by playing a scorer's pick instead
of the policy's own top-1, judged on the TRUE world's 24 held-out referee seeds (16..39) that no scorer reads. **O** =
search in the true world; **F** = the maturity battery's banked fully honest worlds (built at `826bee89`); **FN** = F
with the network's own species and move belief. **G_X** = IMP_O − IMP_X. **LEAF-K8** = the mean outcome over 8 worlds
played to the end; **TRUNC-4-K8** = 4 turns then the critic; **CRITIC** = the honest one-ply critic search on the same
worlds. **q_SM** = the share of an arm's worlds exact on BOTH binding classes (the never-seen species set U and every
seen mon's whole move set M, a Hidden Power one move whatever its type).

## 1. The anchor, the states and the instruments

| | |
|---|---|
| network | X26 (`rb_x26_s2601`) at **225M**: the maturity battery's registered copy `x26_225M` (`checkpoint_225000152`, sha256 in the battery's `ckpts.json`, re-checked at every load) |
| states | the maturity interim bank's LEAN states that the banked F could build: **287 states**, ids sha256 `cf8d8c6f…` (= the LEAN read's common set; `s1_common.common_ids` recomputes it from the banked F world rows and refuses a mismatch) |
| truth, roots, referee | BANKED at the pin: the battery's 225M roots, truth rows (O) and banked F / FN rows (never re-run) |
| world builder | the commit the runner is launched from (on `origin/main`, this registration's commit or a later one carrying no builder change; its sha is every world row's `head` and every row's `driver`): F2's validator rule, F1 tier A, the Hidden Power fill of 2026-10-19 |
| play checkout | X26's pin `9c033489` + ONE cherry-pick, `fc010d7d3` (the Rust core's damage-roll control; its Rust diff is byte-identical to that commit's, checked at every start by `play_s1.check_pin`). Unarmed it plays exactly as the pin (§7 identity tooth); armed it replays a `dice="rolls"` world's force table, so a rolls world is played as its gate accepted it |
| playouts | the pilot's own Phase 2 driver (`run_det_playouts.one_state`, imported unchanged): every legal action × 8 worlds played to the end (world j on dice seed j), the network greedy on both sides; the critic one-ply on seeds 40 + j |
| scorers | Tier 1's (`t1_data`, `score_blind.dk_family`, imported unchanged): LEAF-K8, LEAF-K4, TRUNC-4-K8, CRITIC; ties to the policy's order (1e-6) |

## 2. Owner questions Q1–Q3 (OPEN; not decided here)

Stage 1 needs none of them decided. It runs on the doc's defaults, flagged:
- **Q1** (may a sampler fitted on real teams run at search time): the corpus arms run as an OFFLINE MEASUREMENT only
  (§4 (c)'s interim rule): nothing enters `data/`, a network input or a belief prior; no corpus is fitted on the bank;
  the forum arm drops near copies of each state's TRUE team; the ladder arm's near-copy rate is REPORTED.
- **Q2** (may a live search exclude a decision): an offline read excludes, counted, as every honest read has.
- **Q3** (the opponent's choices as evidence): candidate (f) is not an arm of stage 1 (it is stage 1c, after 1a).

## 3. The arms

Each arm: 8 worlds per state, `max_tries` 60, `key` = the decision id. Every arm but C_pool is F's own builder at the
builder commit (the screen, the template, the gate, the legality rule, the throwing guard), with the departures listed;
**a SEEN mon's item, ability and spread always come from F_legal's SCREEN on ONE shared per-state cache** (common random
numbers; the corpus never fits damage). Built per state in this order (F_legal first).

| arm | worlds | role |
|---|---|---|
| **F_legal** | F at the builder commit (`WorldSampler(reveal="off")`): every set redrawn until Showdown's validator accepts it | **the baseline of record** |
| **B** | (b): each never-seen slot ∝ u(s) · Π_{r ∈ seen ∪ drawn} lift_r(s), lift_r(s) = (t_r(s) + 0.05 · ū(s)) / ū(s) (stage 0's rule, its ALPHA 0.05); everything else F's | PRIMARY |
| **C_ladder** | (c) on the ladder corpus (`utils.ladder_corpus` full, 22,813 legal teams): per world, with probability **λ = n / (n + α)**, a corpus team drawn uniformly among the n CONSISTENT teams (every seen species, each holding its revealed moves; a revealed bare Hidden Power matches any typed one): its never-seen mons WHOLE (species and set) and its seen mons' unrevealed moves; otherwise a B world | PRIMARY |
| **C_forum_strict** | (c) on `w_forum_v1` ABC (2,424 teams) minus every team sharing ≥ 4 mon sets (species + move set) with the state's TRUE team; same λ | PRIMARY |
| **C_pool** | Tier 1's POOL (`pool_worlds.PoolSampler`, imported unchanged): the training pool's teams consistent with the reveals, each gated WHOLE (its seen mons' sets included) | PRIMARY; the closed-population CALIBRATION ceiling |
| **NS** | never-seen species ∝ the network's 225M species presence π (FN's rule; F's weights where π has no mass on the candidates); EVERY mon's moves, with probability n_s / (n_s + 1), from a uniformly drawn forum set of its species holding its revealed moves (n_s such sets, multiplicity kept), else F's draw; everything else F's | PRIMARY ("the best combination in hand"; optimistic: F3, F4) |
| **F_off** | F at the builder commit with the legality rule OFF (an always-legal validator injected: the pre-F2 draw) | SECONDARY: separates the legality rule from the other builder changes since the banked F |
| **F_rolls** | F_legal with `dice="rolls"` (F1 tier A): each world carries its damage-roll force table and log-weight Σ log(k / 16) | SECONDARY: tier A's effect, read UNIFORM (F_rolls) and LIKELIHOOD-WEIGHTED (F_rolls_w, same playouts) |
| F, FN, O | banked (the maturity battery, 225M) | references |

### 3.1 Rules every arm's draw follows
- **The corpora are normalised once** (`corpus_s1.normalize_entry`): the species name as the nickname, every move
  canonical, a Hidden Power TYPED by its IVs after Showdown's own fill (an explicit type with all-31 IVs gets the
  validator's IVs; otherwise the IVs decide), the stale `hpType` field cleared. A team is kept only when all six sets
  pass the validator and name only engine-run moves and items (0 dropped in either corpus).
- **A Hidden Power on a SEEN mon is a spread fact of F_legal's screen** (FN's precedent): a corpus or forum set supplies
  the seen mon's OTHER unrevealed moves; when the screen's IVs carry a Hidden Power the source set lacks, it is added and
  one source move dropped at random; when the source carries one the screen does not, the freed slot is filled by F's
  own draw (`samplers_s1.corpus_seen_moves`, counted per state as `hp_added` / `hp_dropped`).
- **A corpus decision never consumes the world's random stream**: corpus vs backoff and the corpus team are drawn from a
  PEEK of the world's rng (`samplers_s1.peek`), so a backoff world IS the B world of the same (world, try) (§7, CRN
  tooth 1).
- **An arm-state with no world is an EXCLUSION** (`WorldError`, POOL's `PoolExcluded` included): counted, never
  imputed; each contrast is read on the states where both its sources are valid. Any other exception is a DEFECT row
  (counted, a FINDING).

### 3.2 Deviations from the design's §6.2, disclosed
1. **F_off, F_rolls (and its weighted read)** are added (the brief): the legality rule's own IMP effect separated from
   the Hidden Power fill, and tier A as a factor. They are SECONDARY; the PRIMARY family is §6.2's five arms.
2. **NS's set backoff is λ_s = n_s / (n_s + 1)**, stage 0's per-species set prior (the design gives NS no α); with it NS
   is the configuration whose stage-0 meter (q 0.0017) the design quotes.
3. **The calibration compares each arm's TODAY with the q_SM of its OWN built worlds** (the measured share of its 8
   worlds exact on U and M), because stage 0's analytic values were computed at α = 1 and on 1,395 pooled states;
   stage 0's numbers are printed beside.
4. **TODAY's dose curve is F_legal's** (the baseline of record), not the banked F's.
5. **A fifth verdict, HURTS**, makes a clearly negative Δ a stated result instead of NOT DETECTED, and **PAYS takes
   precedence over NULL** (a clear but small lift reads PAYS, flagged `inside_pm_0.01`).
6. **MATERIAL's r_X is read at LEAF-K8** (G's registered scorer).
7. **C_pool's pool is X26's PIN's packing** (`pool_pin.py`), not the builder's: the Hidden Power refactor of 2026-10-19
   (`c9493053b`, `f20526571`) changed the IVs the packing path fills, so the builder commit packs 34 of the 719 pool
   teams with other 30 / 31 patterns (Ghost: 30 Def / 30 SpD, against the pin's 30 Atk / 30 SpD) than the teams X26
   met. POOL gates a candidate whole, so a re-packed true team could fail by a stat point and the ceiling would lose the
   truth. The pool tooth (§7) checks every state's true team is a pool team.

### 3.3 The corpus backoff α (fitted before this registration; `alpha_fit.json`)
Leave-one-team-out on the LADDER corpus, 4,000 simulated reveals (salt `ws_stage1_alpha_v1:`; a team drawn uniformly,
k uniform in 1..5 of its mons seen, each seen mon's moves revealed by a uniform random subset; the held-out team's own
index removed), α the argmax of the mean log-likelihood of the never-seen species set over a fixed log grid
(10⁻³…10⁴, 141 points): **α = 89.13** (mean log p −5.431, against −5.896 at stage 0's α = 1 and −6.022 for (b) alone).
The same α serves both corpus arms. The forum's own leave-one-out fit (α 316, descriptive) is not used. `fit_alpha.py`
sha256 `8863414f…` (recorded in `alpha_fit.json`).

## 4. The metrics

- **Δ_X = IMP_X − IMP_F_legal** at LEAF-K8 and TRUNC-4-K8, paired state for state (the PRIMARY quantity).
- **G_F_legal = IMP_O − IMP_F_legal**; **r_X = Δ_X / G_F_legal** at LEAF-K8 (a ratio of means on the same draws).
- **SEARCH-ENABLED_X**: IMP_X at LEAF-K8 and IMP_X − IMP_CRITIC,X at TRUNC-4-K8.
- **TODAY_X**: the p at which F_legal's M8 LEAF-K8 dose curve (the true world at weight p, F_legal's 8 worlds at
  1 − p; Tier 1's `dk_family`, the 0.05 grid) reaches IMP_X, clamped to [0, 1] per draw (`maturity_rules.today_clamped`'s
  rule). The TRUNC-4-K8 version is descriptive.
- **q_SM,X**: the mean over states of the share of X's 8 worlds exact on U and M.

## 5. The sample, the order and the REGISTERED STOP

- **The order**: the 287 states sorted by sha256(`ws_stage1_order_v1:` + id). The runner (`queue_s1.py`) processes them
  in CHUNKS of 6: every arm BUILT (the builder checkout), then every arm PLAYED (the play checkout), then the next chunk;
  ONE heavy job at a time. So the states complete in every arm are always a PREFIX of the order: a uniform random
  sample of the 287.
- **THE STOP**: no chunk starts after **STOP_AT = the first launch + 24 h** (written once by `launch_s1.sh`; a relaunch
  keeps it); a chunk in flight finishes. **The read is made once**, on the states complete in every arm at QUEUE_DONE or
  at the stop, whichever comes first. A state the runner PARKS (no progress twice) is excluded and is a FINDING.
- **n and power**: n ≤ 287. Between-arm IMP differences on the same states carry about ±0.021 at 287 (design §6.2), so
  stage 1a detects only a search-sized lift (the dose curve's p = 0.2 is worth ≈ +0.03 at LEAF-K8); a NOT DETECTED does
  not exclude a small lift. The read states its n.
- **Cost** (the smoke's one OK state, late game, 7 actions: one arm's play ≈ 9.5 CPU-s; F_legal's build 79 CPU-s on
  that hard state, F_rolls' 21, every CRN sibling ≤ 1; the battery's typical F build ≈ 14): ≈ 120–200 CPU-s per state,
  ≈ 10–16 CPU-h for the 287, ≈ 16–25 wall-h at the measured 0.63 CPU-h per wall-hour of one SCHED_IDLE slot. So the
  24 h stop may cut the tail: the read states its n.

## 6. THE RULES (`rules_s1.py`, tested on planted effects by `read_s1_test.py`; rule 8 throughout)

A bound within ±0.0005 of its boundary is NOT clear (a boundary of exactly ±0.0005 carries the margin). Battle-clustered
bootstrap, 2,000 draws, **seed 20261019**, percentile 95 % intervals, ONE set of draws for the whole read.

**PRIMARY, per X ∈ {B, C_ladder, C_forum_strict, C_pool, NS}:**
- **PAYS** iff the lower 95 % bound of Δ_X > +0.0005 at BOTH LEAF-K8 and TRUNC-4-K8;
- else **HURTS** iff the upper bound < −0.0005 at both;
- else **NULL** iff the whole LEAF-K8 interval lies inside ±0.01;
- else **NOT DETECTED**.
- **MATERIAL** iff X PAYS and the lower bound of r_X (LEAF-K8) is ≥ 0.25 (clear of it).
- **SEARCH-ENABLED_X** iff lower95[IMP_X] > +0.0005 at LEAF-K8 AND lower95[IMP_X − IMP_CRITIC,X] > +0.0005 at
  TRUNC-4-K8 (on X's valid states).
- **CALIBRATED** iff, for EVERY primary arm, TODAY_X's interval CONTAINS q_SM,X; NOT CALIBRATED iff any EXCLUDES;
  else UNDECIDED. Containment by rule 8, with TODAY's clamp edges (0, 1) open.
- **C_pool − C_forum_strict** (what an open corpus loses against team identification on a closed pool): descriptive.
- **Per opponent stratum** (bot / own snapshot): descriptive.
- **No multiplicity correction** across the five arms (as the design registers it): each verdict is quoted with its arm,
  and a single PAYS among five is quoted as such.

**SECONDARY** (the same PAYS / HURTS / NULL / NOT DETECTED rule, each labelled SECONDARY): F_legal − F (the banked
worlds: every builder change since `826bee89`), F_off − F (the Hidden Power fill and the other non-legality changes),
F_legal − F_off (**the legality rule alone**), F_rolls − F_legal (tier A, uniform), F_rolls_w − F_legal (tier A,
likelihood-weighted: each world's leaf at its self-normalised weight `rules.world_weights(log_w)`, the function
`search_decision(weights="likelihood")` uses), F_rolls_w − F_rolls (the weighting alone).

**THE 1b TRIGGER** (design §6.2): any primary arm with Δ_X ≥ +0.010 (the point) at either scorer, or any C or NS arm
SEARCH-ENABLED. A trigger is REPORTED; stage 1b (E2's 1,108 states) is not run in this unit.

**What each result means** (the design's key, printed by the read): C_ladder or C_forum_strict MATERIAL → a team-level
model is the path (then owner Q1, the dice-marginalised check, a ladder-population read); only C_pool pays → team
identification works but the corpora do not cover the opponents (coverage or (f)); nothing pays, C_pool included → at
this leaf information is not the limit (re-read with the matrix §5 and the evaluator); B PAYS → the Smogon-only joint
becomes F's baseline; NS or C_ladder PAYS with its q_SM < 0.01 → PARTLY right worlds carry the value.

## 7. GIGO, before the read (`gigo_s1.py`; `play_s1.py --identity`)

- **THE IDENTITY TOOTH (before launch; must PASS)**: 12 fixed states of the 287 (sha256 order, salt
  `ws_stage1_identity_v1:`): the banked F worlds re-played at the play checkout reproduce the banked 225M F rows
  exactly (`out`, `trace`, `crit`, `other_open`).
- **Coverage / provenance**: one world row and one play row per (state, arm); one builder commit, one play commit, one
  driver commit.
- **Defects** counted per arm (each a FINDING; a defect is never an exclusion).
- **Rolls**: every OK F_rolls world holds 8 finite log-weights and 8 force tables, and its play row the same weights;
  `one_state` refuses any world whose root tokens differ from the banked root (a world played without its rolls).
- **CRN tooth 1**: on a state where no corpus team fits (λ = 0), a corpus arm's worlds ARE B's.
- **CRN tooth 2**: on a state where F_legal's rule rejected nothing, F_off's worlds ARE F_legal's.
- **Pool tooth**: every state's true team is a pool team (C_pool's `true_in_pool`).
- **Leak tooth**: no C_forum_strict world shares ≥ 4 mon sets with the true team.
A failed tooth stops the read (a FINDING, reported).

## 8. CPU and safety

CPU only (`CUDA_VISIBLE_DEVICES=""`), one single-thread worker at a time under `chrt --idle`, `nice -n 19`,
`taskset -c 5-7,13-15` and its own `mem_cap` scope (root rule 5b; a live run). Nothing is written under `models/` or
`data/`; the maturity battery's archive is read only. Bulk rows: `~/gen3ai_archive/world_sampler_stage1/`.
