# N0 END-OF-RUN READS — REGISTRATION (2026-09-27)

**Status: REGISTERED before any game of these reads is played.** Written 2026-09-27 ~11:30 PT, while
`models/ai_v14_01_base` (N0, the new lineage's fresh base run: v121, pin `8d07051a`, 75M requested,
ETA ~Sun 09-27 23:15 PT) stood at ~48M steps and `final_model.zip` did not exist. Nothing here trains,
touches the GPU, writes under `models/` or changes `data/`.

**Subject:** `models/ai_v14_01_base/final_model.zip`, named explicitly (never the bare run dir, which
resolves to the last snapshot). Expected step 75,005,952.

**Where the reads run:** a pinned worktree at the commit that lands the pair-cell harness mode (§3),
with its OWN release `sim_bridge` / `search_driver` (never main's target). That commit is `2cc83080`
(v123, search Rust-core-only) plus six launcher / policy-drift commits and the anchors pair mode; no
file on any read path differs from `2cc83080` except `src/main/anchors/{cli,runner}.py`, and
`src/rust_sim/` is identical. `MIGRATION_FLOOR` is 121, so the v121 checkpoint should load: that is
VERIFIED by the pre-N0 smoke (Amendment A below, on N0's 48M snapshot) before the queue is armed.

**Rows:** durable, append-only, outside every git tree and outside `/tmp`:
`/home/goodlad/dev/gen3ai-reads/n0_endofrun_2026-09-27/` (the queue's state dir). A reader copies the
aggregates here when a read completes.

Every section below says whether its read is a **DECISION** (a registered bar, ruled on at the
registered n only; interim looks are progress, never verdicts — ORCHESTRATOR_SOP §2 rule 4) or a
**DESCRIPTOR** (reported, never ruled on).

---

## 1. LEAF QUALITY of N0's final head — DECISION: does search reopen?

**Instrument:** the mirror battery of `search_dividend_winprob_heads_2026-09-11` (the 09-11/09-12
L2 read and its `grid` row), unchanged in protocol: `python -m main.search_dividend <final_model.zip>
--arm honest --budget 1 --opponents self --games-seed 7` (side-swap pairs: every game index played
in both orientations off one pinned seed), CPU, `--search-impl rust` (the only road on this tree),
`--battle-timeout-s 1800 --battle-idle-s 120`, **ONE process at a time at nice 15**, OMP/BLAS pinned
to one thread, the GPU hidden. `--score auto` must resolve to `win_prob` (N0 is `--critic winprob`).

| cell | what | flags beyond the common set | game indices (pairs) |
|---|---|---|---|
| **`defB` (L2, the decision)** | the registered 3 s defensive operating point | `--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0` | **0–399 (400)** |
| **`base` (the NULL, rule 26)** | both sides unsearched | `--arm base` in place of `--arm honest` | **0–99 (100)**, INTERLEAVED with `defB` (one 10-pair `base` unit after every four 10-pair `defB` units) so the two see the same box |
| `grid` @1 s (the sensitive row) | naive search, every decision, no gate | `--root-strategy grid` | 0–99 (100) |
| `defB` EXTENSION (conditional) | the 09-11 rule, verbatim | as `defB` | 400–799, with `base` 100–199 interleaved as above |

**The extension rule (registered now, outcome-blind):** if `defB` over pairs 0–399 has a paired point
≥ 0.51 AND its CI straddles 0.50, it is extended to exactly 800 pairs over fresh indices 400–799 and
both reads are published; the decision is then read at 800. The queue applies this rule itself.

**Statistic:** the report's own paired read (`main.search_dividend --summary`): per game index the mean
of its two orientations, a normal 95% CI on the mean. `defB − base` is PAIRED over the shared indices
(0–99, or 0–199 after an extension), with the delta's own 95% CI.

**BAR — SEARCH REOPENS iff BOTH, at the registered n:**
- (a) `defB`'s paired 95% CI lower bound > 0.50; AND
- (b) `defB − base` (shared indices) has a 95% CI lower bound > 0 (rule 26: the null is the
  contemporaneous unsearched cell, not 0.500).

Otherwise: **NO DIVIDEND DETECTED** (the CI straddles), or **SEARCH HARMS** (`defB`'s upper bound
< 0.50). A reopen is a gate for a search-teacher build, not a build.

**Descriptors (never verdicts):**
- `grid` against the prior six heads. A `grid` CI whose upper bound reaches 0.50 (unguarded search no
  longer harms) is a MAJOR finding on its own, but it does not reopen search.
- The mechanism rows: separation-of-raced, overrules (% of all decisions), the forced fraction,
  `futility_genuine`, and the realized K worlds per contested decision. **L1 is a WIDTH meter (rule
  23)**, so separation is quoted only inside K bands (`l1_width_matched.py`'s bands) and never against
  a fixed bar. There is no contemporaneous control head (no pre-v121 head loads on this tree), and the
  search road changed (Rust core, not the node driver), so every cross-era mechanism comparison is
  width-unmatched and independent-sample.

**The comparison: the prior six win-prob heads** (`search_dividend_winprob_heads_2026-09-11`,
`--games-seed 7`, node search road, 9 shards, 679→719-team pool era):

| head | L2 `defB` paired | `grid` paired | overrules (% of all decisions) | separation-of-raced (K) |
|---|---|---|---|---|
| `ctrl10M` @10M | 0.5059 [0.4914, 0.5205] (800) | 0.2600 | 1.41% | 12.8% (5.06) |
| `lambda09` @10M | 0.5031 [0.4894, 0.5169] (800) | 0.2600 | 1.12% | 11.9% (4.85) |
| `wp73M` @73M | 0.4950 [0.4740, 0.5160] (400) | 0.2025 | 1.64% | 15.0% (5.48) |
| `strata` @10M | 0.5066 [0.4924, 0.5207] (800) | 0.3150 | — | 11.5% (4.78) |
| `denseaux` @10M | 0.4941 [0.4826, 0.5056] (800) | 0.3025 | — | 6.1% (3.80) |
| `cflabels` @10M | 0.4853 [0.4664, 0.5042] (800) | 0.1875 | 3.2% | 30.3% (10.08) |
| *v9 SHAPED critic (iteration 2)* | — | 0.2929 | 5.82% | 45.4% |

The standing null: `base` read 0.515 [0.497, 0.533] at n = 200 (2026-09-20). **Prediction (for the
record, not a bar):** NO DIVIDEND DETECTED, `defB` in [0.48, 0.53]; `grid` below 0.40.

**Not in this read (a FINDING):** rule 25's direct metric, PAIRWISE ACCURACY on contested sibling
forks, needs a fork bank and a scorer that load v121; neither has been rebuilt across the era wall.

---

## 2. ERA-GATE ANCHORS on N0's final — `python -m main.anchors`

All cells: `--server rust` (default), `--regime greedy`, `--device cpu`, role-balanced halves, one cell
at a time, `--out` in the state dir. **Units** (SOP §2): a 100-game cell is split into units of
distinct `--team-seed` = `--seed-base` values (the G-A driver's convention), pooled by summing; the
split changes the draw stream, never the protocol.

| cell | opponent / teams | n (units) | kind |
|---|---|---|---|
| **A1** | `metamon:SyntheticRLV2` greedy, **away** (Metamon's 20 `competitive` teams, both sides) | 100 (4 × 25; seeds 0, 10, 20, 30) | **DECISION — the era bar** |
| A2 | `metamon:SyntheticRLV2` greedy, **home** (stock: the 719-team pool, both sides) | 100 (4 × 25) | descriptor |
| A3 | `foulplay` `--search-time-ms 1000 --search-parallelism 1`, **home72** (§2.1) | 100 (10 × 10; seeds 0–90) | descriptor |
| A4 | `metamon:SyntheticRLV2` greedy, **home72** — the round robin's third edge (§3) | 100 (4 × 25) | descriptor |

**The era bar (SOP Tier C), A1 only:** Wilson 95% lower bound > 0.50 ⇒ **BETTER**; upper bound < 0.50 ⇒
**WORSE**; otherwise **NOT DETECTED** (never "equal", never rounded up). Every cell is quoted with its
regime, team set, opponent checkpoint and commit, our step, n, W/L/T and the Wilson CI; A3 with its
realized `visits/decision` (mean, and its range across units). A unit counts only if `status == OK`,
`regime_verified_decisions == true` and `team_source_asymmetry == false`; `peer_clean` false with a
recognised H17 note stands (H16/H17).

**Which comparisons with the last era are like-for-like** (`ai_v12_02_winprob_critic`@75,005,952):

| ours | standing number | like-for-like? |
|---|---|---|
| A1 | SyntheticRLV2 greedy away **0.420** [0.328, 0.518] | **Protocol yes** (same opponent ckpt 48, regime, team set 20/20, role balance). Differences: transport (Rust front end vs Node; agreement shown n.d. on SmallRL, 2026-09-20), the unit split, and everything about the model (clean inputs, v121). An independent-sample Newcombe CI, never a paired one |
| A2 | SyntheticRLV2 greedy home **0.500** [0.404, 0.596] | Protocol yes, as A1 — **if** the 719-pool is the same set it was (the pool was 719 on 09-14 and is 719 now; recorded per cell) |
| A3 | Foul Play @1000 ms **0.388** (31/80) | **NO.** That campaign pinned ONE pool team per 10-game session for us while Foul Play redrew from all 72 (asymmetric), Foul Play was always the challenger (pre-H14), and it ran on Node. The nearer comparator, `foulplay_axes_and_frontend_validation_2026-09-16` `w1000` **0.450** [0.346, 0.559] (72 sample teams both sides, 80 games), is also not like-for-like: challenger-only, Node, and the unsanitized files of H19. Quoted as descriptors with the width (1.40 M visits/decision on 09-14; 780 k mean on 09-16) beside each |
| A4 | none | new |

### 2.1 The `home72` team set (hazard H19, found while building this read)

The stock `foulplay` `home` directory is **not** the pool (72 files vs our 719 — every stock Foul Play
home cell is asymmetric), and 11 of its 72 files carry bracketless `Hidden Power <Type>` moves with no
`IVs:` line (Foul Play plays them as a different Hidden Power from the one our teambuilder patches in)
plus nicknames. `scripts/make_home72.py` writes `teams_home72/`: the same 72 `data/teams/sample` teams,
nickname-free, every Hidden Power spelled `[Type]` with its explicit IV line, one blank line between
mons, **all 72 validated by the pinned Showdown validator**. ALL THREE clients read these exact files
in A3, A4 and §3: ours through a `dir` team source, Foul Play through an absolute `--team-name`, and
Metamon through a `gen3ai_home72` team set in its cache (a byte copy; the queue refuses if it differs).

---

## 3. NEW: Foul Play vs Metamon SyntheticRLV2, head to head — DESCRIPTOR

`python -m main.anchors --opponent-a foulplay --opponent-b metamon:SyntheticRLV2 --regime greedy
--teamset home --search-time-ms 1000 --search-parallelism 1 --progress-timeout 3600` under the
`home72` config: **100 games (10 × 10; seeds 0–90)**, role-balanced, reported from SyntheticRLV2's side
(its battle CSV is the record; a tie is booked as a loss there, so ties are counted from the CSV's
`Result` and reported). **Neither side forfeits at 250 turns** — the sim's 1000-turn tie is the only
cap — so this edge is under a different stall rule from A3/A4 (whose our-side forfeits at 250); the
turn distribution is reported beside it.

**The mini round robin, N0 × SyntheticRLV2 × Foul Play, on the same 72 files:** A3 (N0–FP), A4
(N0–SynthV2) and this cell (SynthV2–FP), 300 games. Reported as three Wilson-CI edges plus a
descriptive Bradley–Terry fit (three nodes, three edges: a fit with zero degrees of freedom for a
transitivity test, so any intransitivity is REPORTED as the edge signs, never tested). No verdict.

---

## 4. The learner battery's strength reads — run by this queue, REGISTERED ELSEWHERE

Registration: `designs/research_state/learner_battery_2026-09-26.md` §4.2–4.3 (bars, n, rules). Not
re-registered here. What this queue runs, and every declared deviation in FORM:

- **U** per arm (C, E5, T32, L95 finals): untaught-8, **600 games/team**, `--seed 0`, concurrency 1,
  `--config auto`, the round-2 G-U driver's per-battle body verbatim (`scripts/gu_unit.py`: one
  unit = ref × team × 25 battles, one fsynced row per battle; a unit resumes at the first missing
  battle index). A cell is a pure function of (ref, team, battle), so the unit split and the worker
  count change no number.
  - 🚨 **FINDING: `untaught_meter_opponent_v14` is NOT in `designs/baselines.json`** at registration
    (the 24M snapshot exists). The queue passes the registered opponent's FILE explicitly,
    `models/ai_v14_01_base/snapshots/snapshot_000024000000.zip` (§3.2 of the lineage registration
    defines the name as exactly that file), and stamps it on every row.
- **G-A** per arm: `metamon:SmallRL` greedy, `--teamset {away,home}`, 12 × 100-game units, `--team-seed`
  = `--seed-base` ∈ {0, 10, 20, 30, 40, 50}: 1,200 games per model.
- **Tree:** this read's pinned tree instead of `2cc83080` — identical on every read path (see the
  header); declared, not believed to matter.
- **Descriptors for N0 itself (lowest priority):** U(N0) at 600/team (the battery's U(X) − U(N0)
  descriptor, and the lineage registration §2's "untaught-8 level vs the new opponent") and SmallRL
  G-A on N0 (the lineage's "SmallRL 1,200 greedy games").

---

## 5. Execution — the CPU work queue

`scripts/queue.py` (one detached supervisor; state dir as above; `run | status | stop`):

- **Order.** Battery reads first, in arm order, as each arm's `final_model.zip` exists and has been
  stable for 5 min (they gate adoption); then N0's reads in the order §1 → §2 (A1, A2, A3, A4) → §3;
  then the N0 descriptors of §4. Units are minutes long, so a battery arm finishing pre-empts N0
  work at the next unit boundary.
- **Concurrency.** While any GPU compute process exists (a training arm), exactly ONE unit runs at
  a time. When the GPU is idle, U units (and only U units — a pure function of their inputs) may run
  up to 8 at once (the registered G-U worker count); leaf and anchor units always run alone, because
  a wall-clock search budget buys a width set by the box (rule 23). `control.json` in the state dir
  overrides both counts and can pause the queue between units.
- **Niceness 15**, `CUDA_VISIBLE_DEVICES=""`, one BLAS thread per process, ports 9500–9599 (the
  anchors tool's own, stopped by PID), no argv containing the trainer's script name.
- **Resumable and durable:** a unit is DONE iff its own output says so (an OK `summary.json` with
  n = its games; all its battle indices on disk); a partial anchors unit is moved aside to
  `<dir>.partial.<epoch>` and re-run; a partial mirror unit's rows are moved aside and the window
  re-run; `units.jsonl` logs every start and end with rc, wall and load.

**Expected times** (single process while an arm is live; history: `defB` ~20 s/battle, `grid` ~23 s,
`base` ~92 s on the node road; `SyntheticRLV2` ~6 s/game plus a multi-minute build per half; Foul Play
~88 s/game; U ~9 s/battle/worker): §1 ≈ 10–14 h; §2 ≈ 4 h; §3 ≈ 2.5 h; U ≈ 12 h per arm at one worker
(≈ 1.5 h at eight); G-A ≈ 1.5 h per arm. **Amendment A replaces these with the smoke's measured
costs before N0 ends.**

---

## Amendments

*(Append-only. None may change a bar, an n, or a cell after the first N0 game.)*

### Amendment A — the pre-N0 smoke (2026-09-27 ~11:10–11:55 PT; no bar, n or cell changed)

Taken on N0's **48M snapshot** (`snapshots/snapshot_000048000000.zip`), never the read's subject, from the
pinned tree with its own release binaries. Rows: `/home/goodlad/dev/gen3ai-reads/n0_smoke_2026-09-27/`.

| check | result |
|---|---|
| **v121 loads on the v123 tree** (`MIGRATION_FLOOR` 121) | ✅ `main.search_dividend`, `main.anchors` (`loader=bare`, `explicit_zip`) and the U unit (`load_foreign_opponent`, `--config auto`) all load it; `--score auto → win_prob` announced |
| leaf, 1 pair each | `defB` 2–8 s/battle (realized K = 35 worlds on its one raced decision — ~7× the node road's K ≈ 5, so the width is NOT the 09-11 width), `base` **2.5 s/battle** (hazard 8's ~92 s is gone on the Rust core), `grid` ~31 s/battle |
| U unit, 25 battles | ~1.7 s/battle median (~60 s per unit with the loads); **SIGKILLed after 6 rows and re-run: exactly 25 rows, indices 0–24, no duplicate**; two full runs of the same unit gave the same 16 wins (a pure function) |
| queue supervisor | killed (SIGKILL) mid-unit and restarted: the partial unit's rows were moved to `.partial.<epoch>` and the window re-ran |
| pair cell, `--opponent-a foulplay --opponent-b metamon:SmallRL`, home72, 2 games @100 ms | OK, 72/72 teams, `forfeit limit NONE`, realized 99,783 visits/decision. 🚨 **Its first run read `regime VERIFIED: False` on a Metamon side with argmax 45/45**: the anchor-vs-anchor path read OUR report through the OPPONENT's adapter (Foul Play's reader has no argmax rate; Metamon-vs-Metamon hid it). **Fixed** (`runner.pair_adapters`, tested) before any read; re-run: VERIFIED True |
| model vs `metamon:SmallRL` away, 2 games | OK, VERIFIED, 20/20 teams |
| model vs `foulplay` @1000 ms, home72, 2 games | OK, VERIFIED, 72/72, realized **728,539 visits/decision**, ~130 s/game |

**File rename:** the queue is `scripts/n0_queue.py` (a script named `queue.py` shadows the standard
library's `queue` for everything it imports — the first U smoke died on it). §5's `scripts/queue.py` means it.

**Expected times, re-derived from the smoke** (one unit at a time while a GPU arm is live): leaf ≈ 4 h
(`defB` 800 battles ≈ 1.5–2 h, `base` 200 ≈ 10 min, `grid` 200 ≈ 1.7 h); U ≈ 3 h per ref at one worker,
≈ 0.5 h at eight; G-A ≈ 1–1.5 h per model; A1+A2+A4 ≈ 2 h; A3 ≈ 3.5 h; §3 ≈ 3 h. With N0 at ~23:15
Sun and the arms at C ≈ 03:45, E5 ≈ 08:00, T32 ≈ 08:45–12:15, L95 ≈ 16:30 Mon: the leaf read runs
~23:20–03:30; C's U + G-A ~03:50–08:30; E5's ~08:30–13:00; T32's ~13:00–17:30; L95's ~17:30–22:00
(sooner if the GPU is idle then: U at eight workers); then A1–A4 and §3 (~8.5 h) to ~Tue 06:30, then
N0's descriptors (U(N0), G-A(N0)) to ~Tue 09:00. **Consequence: the era-gate anchors (A1) land ~Tue
morning, behind every battery read, as the brief's priority order requires.**
