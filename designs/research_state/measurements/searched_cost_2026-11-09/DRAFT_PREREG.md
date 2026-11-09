# DRAFT — NOT REGISTERED. Searched-cost read (Q19): can S′h's gain be bought cheaper, as a player that fits the turn timer?

> 🚧 **DRAFT (2026-11-09), NOT a registration.** Nothing here is registered or played; the code of §7 is BUILT (2026-11-09, §7's status). No spec, no pin, no
> engine stamp, no lane. A registration needs the builds of §7, the orchestrator's go and a review. **Every candidate
> below was chosen AFTER S′-HONEST's outcome** (GO-HONEST, GOAL-B-REACHED, 2026-11-09), as that read's PREREG §8 and
> §10 require any new preset or builder to say. The design it serves is
> [`designs/endstate/design_searched_player.md`](../../../endstate/design_searched_player.md); the question is
> `search_ledger.md` Q19.

**Abbreviations.** As the design's. REF = `r0s4_honest_inc@10` as S′-HONEST played it (pin `e46e3ddf8`, engine E6).
G_X = the paired pooled-U gain of P\* + S′h-X over P\* raw against Kakuna (checkpoint 34, greedy), `searched_play`'s
estimand. d = the share of discordant deal pairs. CPU figures are S′-HONEST's (`../sprime_honest_play_2026-11-06/`
`record/search_cost.json`, and [`inputs/timer_replay.json`](inputs/timer_replay.json) from
[`inputs/timer_replay.py`](inputs/timer_replay.py), outcome-blind).

---

## 0. The question, and the mission line it bears on

**Q.** Is there a search preset X, cheaper than REF and inside Showdown's turn timer, with which P\* still beats P\*
alone against the real Kakuna on U?

**What REF costs** (the reference every arm is priced against):

| item | REF on U (F / H / P72) | source |
|---|---|---|
| search CPU-s a searched game, searched decisions | 249 (309 / 212 / 226) | `timer_replay.json` `cpu_s_per_game` |
| … plus the excluded decisions | **+101** (209 / 40 / 54) → **350** a game | the same; FINDING SP-2 of the design |
| of which playouts / world building | 122 / 215 | the same |
| whole-unit CPU-min a searched game (process, peer, children) | 6.64 (9.52 / 4.94 / 5.47) | S′-HONEST README, cost table |
| exclusions (share of search-path decisions) | 5.55 % (10.0 / 2.6 / 3.3) | S′-HONEST `decision.json` |
| timer replay: games forfeited as played / searched decisions cut under the bank-aware deadline | 16.2 % / 3.2 % (cut step dropped) – 8.4 % (finished) | `timer_replay.json` `timer` |

**The mission sentence this read bears on** (proposed in the design's Decision record; the orchestrator decides,
`UNDERSTANDING.md` §1 is not edited here): "**The mission.** A Generation 3 OverUsed (gen3 OU) generalist that keeps
improving: not a specialist, and not a search-first agent: the trained network is the product and the yardstick for
every training lever, and one-ply, leak-honest search at play time (only our action branched at the root, the network
on both sides of every playout, inside the ladder's turn timer) is a separately measured layer on top of it, never a
substitute for improving the network."

---

## 1. The classes (stage 1)

| class | rule (`searched_play`, unchanged estimand and imputation) | licenses | does NOT license |
|---|---|---|---|
| **GO-HONEST** | LB(G_X, GO-imputed) > 0 | X is a measured, cheaper path to goal (b): searched reads may use X at its cost | any claim that X ≥ REF (the cross-read is REPORTED, §8); ladder or public play; any training change |
| ↳ `timer_safe` (NEW decided flag, §7 R3) | on X's counted searched units, the timer replay with the bank-aware deadline (reserve 10 s) forfeits 0 games AND cuts ≤ 2.0 % of X's searched decisions (pooled U) | with GO-HONEST: X is the deployable player's search, and the design's acceptance run (P6) may be scheduled on our own server | a public-server or ladder game (root § Playing) |
| ↳ `goal_b` (as S′-HONEST's) | GO-HONEST and X's pooled-U score LB > 0.50 | "goal (b) met with search" in X's regime | — |
| **NULL** | not GO-HONEST and UB(G_X, NULL-imputed) < **+2.0 pp** | X keeps less than REF's point gain: X is closed as a cheaper REF | a claim about REF; other presets |
| **STOP** | otherwise | nothing | a re-run at a larger n without a new registration |

The NULL margin is REF's point (+2.00), not S′-LIVE's +3: below it, X has given up most of what REF measured.

---

## 2. Why not non-inferiority, and why not gain per CPU, as the decided meter

**Non-inferiority against REF** (Δ = y_X − y_REF by deal, NON-INFERIOR iff LB(Δ) > −m) is the natural question and is
unaffordable at any margin that matters. The paired SE at S′-HONEST's discordance (d 0.350, realised SE 0.80 pp at
5,097 pairs; two searched arms diverge after turn 10 at least as often as searched and raw, so d ≈ 0.35 is the
planning value, UNVERIFIED) gives, for 80 % power at a true Δ = 0:

| margin m | deals needed (realised / conservative √(d / n)) | CPU-h (X at 0.6 × REF + a fresh REF arm, ≈ 10.6 CPU-min a deal) |
|---|---|---|
| 1.0 pp (half REF's gain) | 25,600 / 27,400 | ≈ 4,500–4,850 |
| 1.5 pp | 11,400 / 12,200 | ≈ 2,000–2,150 |
| 2.0 pp (all of REF's gain) | 6,400 / 6,900 | ≈ 1,130–1,220, and the margin is the whole effect |

Re-using S′-HONEST's banked REF games on its own deals (seeds 700–784) caps n at 5,100 (m ≥ 2.24 pp at 80 %) and
carries REF's realised luck on exactly the deals that produced the GO into Δ (a selected comparison). **Rejected.**

**Gain per CPU** (G / C) as a decided meter: C is measured almost without error, so the ratio's interval is G's
interval scaled; deciding "X is at least as efficient as REF" is non-inferiority at m = (1 − C_X / C_REF) × G_REF,
≈ 1 pp for X at half REF's cost. The same table. It is REPORTED (§8).

**So the decided question is X's own gain over P\***, with REF as a reported, unpaired comparison. It answers what the
deployable player and every later searched read need: a cheaper search that still measurably beats P\*.

---

## 3. Stage 0: the offline screen (zero games, outcome-blind)

**What runs.** Each candidate on a stratified sample of S′-HONEST's banked searched battles, **360 battles (120 a
panel)**, drawn by a registered hash of the cell name, every battle's whole decision sequence from turn 10 rebuilt
from its `capture/` input log (seed, commands, chunks) with one live searcher a battle: stage 1b's harness
(`../sprime_honest_inc_stage1b_2026-11-06/seq_probe.py`) generalised to any preset (§7 R6). No game result is read:
only cost, walls, exclusions and picks.

**The candidates (one lever each against REF):**

| arm | the lever | aimed at | expected (ESTIMATE, stage 0 measures) |
|---|---|---|---|
| **K4** | the world count: `n_worlds` 4, `per_action` 4 (a world then serves ⌈4 / 2⌉ = 2 decisions; an exclusion is < 4 worlds) | cost | playouts ½ (≈ −61 CPU-s a game); world building ↓ (4 carried, 2 fresh a decision); exclusions ↓ (the partial first exclusions of §5's class); search CPU ≈ 0.5–0.7 × REF. Risk: Q̂ noisier at piKL λ 0.015 fitted at 8 (FINDING SC-3) |
| **CAP** | the builder's COUNT budget profile: the cold first build at the base budget instead of B3, the shortfall escalation at 256 tries instead of 1,024, the stuck full attempt at the base budget | the timer's tail (the design §3: deterministic budgets) | the p99.9 decision wall ↓ by most of the B3 / escalation tail; exclusions ↑ (how much is the point of stage 0); the first decision no longer byte-equal to `r0s4_honest`'s (C25) |
| **XR** (exclusions, aimed at F) | ONE of: XR-J, search with the k ≥ 4 worlds built instead of excluding (uniform over them); XR-R, a reveal's redraw at 16 attempts instead of 4 and the revised world's escalation — chosen by stage 0a | F's 10 % exclusions | XR-J addresses the partial first exclusions (F 153 of 431 battles, H 124 of 172, P72 134 of 208); XR-R F's dominant class (no world built: 247 of 431; 3,213 of F's 5,201 probe exclusions failed on revised worlds alone). Cost ≈ REF ± 10 % |

**S′H-15's fix is not a candidate arm.** It is not a cost lever: counting the fresh worlds against the survivors still
within the bound adds ≈ 1 fresh world (≈ 0.5–2 CPU-s late) on the ≈ 7.7 % of searched decisions that hold an
over-age world: ≈ +1 % CPU. Stage 0 builds and REPORTS it (overage share → 0, CPU, pick agreement with REF against
REF's own floor); it enters a game only through its own registration (one lever per A/B).

**Stage 0a, the F diagnosis (before XR is chosen).** On F's 278 battles whose first exclusion built no world (247 warm,
31 cold), classify the failing segment from I alone (the events of the segment no carried world matched: a crit, a
sleep onset outside `backtrack`, Beat Up, a hidden-HP chain, a revealed set outside the Smogon prior's support, …),
and the same on the 153 partial ones. XR-R is chosen if the revised-world class carries most of F's stuck decisions,
XR-J otherwise; the rule is registered with stage 0.

**What stage 0 measures per arm, on the same 360 battles as REF's banked records:** search CPU a game (searched +
excluded), the per-decision wall distribution and the timer replay (`timer_replay.py`'s policies), the exclusion
share by panel, the pick agreement with REF's banked pick at every decision both search, against REF's own re-draw
floor (stage 1b: 63 %, `design_sprime_honest.md` §9.5).

**The selection rule (registered with stage 0, before it runs).** An arm PASSES iff, pooled over U on the sample:
(i) the bank-aware timer replay forfeits 0 games and cuts ≤ 2.0 % of its searched decisions; (ii) its search CPU a
game ≤ 0.7 × REF's on the same battles; (iii) its pick agreement with REF ≥ REF's own floor − 5 pp; (iv) its exclusion
share ≤ REF's + 1.0 pp. **Stage 1 plays the passing arm with the lowest search CPU; ties to the simpler lever (K4,
then CAP).** No arm passing ends the read at stage 0 with a FINDING; a combination (K4 + CAP) is then a new DRAFT.
XR is screened for its own sake and enters stage 1 only if it passes (ii) too, which it is not expected to.

**Cost of stage 0 (ESTIMATE).** 360 battles × ≈ 39 search-path decisions × 3–7 CPU-s ≈ 12–27 CPU-h an arm; REF
re-run on the sample for the per-battle comparison and the floor ≈ 25 CPU-h; four arms (K4, CAP, one XR, S′H-15) plus
REF ≈ 75–135 CPU-h; stage 0a ≈ 2 CPU-h. On the agent cores at K = 6 beside a pinned training run (≈ 5 CPU-h a wall
hour): ≈ 15–27 h, in ≤ 45-min chunks under `mem_cap.sh`.

---

## 4. Stage 1: the game read

- **The arms.** P\* + S′h-X (`pstar_s`, `load.search: <X>@10`, X carrying its envelope's budget profile) and P\* raw
  (`pstar`, its raw twin), the same pinned bytes (P\* `814823c5…91ed`), on our front end through the `main.reads`
  anchor lane, one game a unit, alone in a fresh process (S′-HONEST §3's discipline: DL-7).
- **The deals: U unit seeds 800–914** (115 seeds × 20 deals × 3 panels = **6,900 deals**, 13,800 one-game units).
  Disjoint, as far as this draft could check, from every registered Kakuna read (S′-HONEST 700–784, S′-LIVE 300–344,
  dlive 200–244, X65 600–674, S-U 100–109, X63 1000–1199, the rest 0–119); re-checked against every PREREG at
  registration (FINDING SC-5).
- **The rule** `searched_play` as S′-HONEST registered it (G-PREFIX through `|turn|10`, the peer-drop classifier, the
  post-treatment imputations, the within-panel unit bootstrap, rule 8's 0.0005 band, the exclusion flag with bounds
  set from stage 0's measured shares), with `null_margin_pp` 2.0 and the new `timer` block (§7 R3).
- **The identity guard.** `live_identity rescore --carry rebuild`, the registered-order 600, as S′-HONEST.
- **The engine and pin** are chosen at registration (the builds of §7 move the engine stamp; the four existing presets
  must re-score byte-identically at the new code first, §7 R1).

---

## 5. Power (normal approximation)

SE(n) = 0.80 pp × √(5,097 / n) (S′-HONEST's realised) and the conservative √(0.350 / n). P(GO) = Φ(G / SE − 1.96);
P(NULL at the +2.0 margin | G) = Φ((2.0 − 1.96·SE − G) / SE). Cells: realised [conservative].

| deals | SE (pp) | P(GO) at +1.0 | at +1.5 | **at +2.0 (REF's point)** | at +2.5 | P(NULL) at 0 |
|---|---|---|---|---|---|---|
| 5,100 (85 seeds) | 0.80 [0.83] | 0.24 [0.23] | 0.47 [0.44] | 0.71 [0.68] | 0.88 [0.85] | 0.68 (cons.) |
| **6,900 (115 seeds, proposed)** | **0.69 [0.71]** | 0.31 [0.29] | 0.59 [0.56] | **0.83 [0.80]** | 0.95 [0.94] | 0.80 (cons.) |
| 9,000 (150 seeds) | 0.60 [0.62] | 0.38 [0.36] | 0.70 [0.67] | 0.91 [0.89] | 0.99 [0.98] | — |

**6,900 is the smallest whole-seed n with 80 % at REF's point on the conservative SE.** At 6,900: GO iff Ĝ ≳ +1.39 pp,
NULL iff Ĝ ≲ +0.61 pp, STOP between. A cheaper X that kept only half of REF's gain (+1.0) is GO in 3 reads of 10:
the read is built to confirm "X keeps about REF's gain", not to size a small one. REF's +2.00 is itself one read's
point (interval [+0.34, +3.51]); planning on it is optimistic by any winner's curse it carries (FINDING SC-2).

---

## 6. Cost and wall (ESTIMATE; cost never re-sizes the read)

| item | CPU-h |
|---|---|
| 6,900 searched units at C_X = 0.5–0.7 × REF's 6.64 whole-unit CPU-min (the stage-0 ceiling) | 380–535 |
| 6,900 raw units (0.60 CPU-min a unit) | 69 |
| the identity re-score, 600 decisions, REBUILD (S′-HONEST measured 25.8 CPU-h for REF; X cheaper) | 15–26 |
| **stage 1** | **≈ 465–630** |
| stage 0 (§3) | 75–135 |
| **the read** | **≈ 540–765** |

Wall: S′-HONEST's BOX shape (K = 14, the whole box, no run live and no lease) cleared ≈ 13 CPU-h a wall hour (≈ 616
CPU-h of units over ≈ 47 h of its two lives) → stage 1 ≈ 36–48 h. On the agent cores beside a pinned live run (K = 6,
≈ 5 CPU-h a wall hour) ≈ 2.5× longer (≈ 4–5 days). A rented box is not needed.

---

## 7. What the reads framework and the code need (BUILT 2026-11-09, except as the status below says)

| # | change | where | gate |
|---|---|---|---|
| R1 | `SearchPreset` gains the builder's budget profile and `j_min` (or a named profile), defaulting to today's values so `r0s4`, `r0s4_rolls`, `r0s4_honest`, `r0s4_honest_inc` stay byte-identical; the new presets (K4, CAP, XR) named | `src/main/anchors/search_presets.py`, `src/main/value_search/honest_inc.py`, `src/main/value_search/honest_worlds.py` | S′-HONEST's registered 600-decision identity re-score PASS at the new code; the stage-1b tooth; the leak test in sequence mode on every new preset |
| R2 | a pure **timer model** (the design §2), its constants pinned by a test that reads the literals of `deps/pokemon-showdown/server/room-battle.ts` (the submodule is a launch requirement); `inputs/timer_replay.py` is its prototype | a new module beside `src/main/reads/search_cost.py` | its test reproduces `timer_replay.json`'s `bank_r10` and `as_played` rows on the banked cells |
| R3 | `searched_play` gains an optional `timer` block (`reserve_s`, `overhead_s`, `max_cut_frac`) and the decided flag `timer_safe` (never VOIDs, never changes the class), plus the REPORTED replay table; `null_margin_pp` is already a parameter | `src/main/reads/rules_searched_play.py` | rule tests: the flag on a toy set, near-bound handling as rule 8 |
| R4 | `search-cost` puts the search CPU INCLUDING exclusions first and splits searched / excluded (SP-2), and reports the timer replay (R2) | `src/main/reads/search_cost.py` | its test on a cell with an exclusion |
| R5 | the live searched side records a safety-net cut with its own `why` (never `excluded`), and `searched_play` counts it beside the exclusions | `src/main/anchors/search_side.py` | — |
| R6 | stage 0's harness: stage 1b's `seq_probe.py` taking any preset and S′-HONEST's captures | this folder (measurement code, frozen with it) | its first battle reproduces REF's banked records at REF |

**Status (2026-11-09, the searched-cost builder).**
- **R1 built in part.** `SearchPreset` gained `budget` (a named profile, `honest_inc.BUDGETS`) and `net_reserve_s` (the
  live safety net), both left out of `as_dict` at their default None, so the four earlier presets' dicts are
  byte-identical (pinned by `search_side_test`). Named: **K4** = `r0s4_honest_inc_k4` and **CAP** =
  `r0s4_honest_inc_cap`, each with the net at 10 s. CAP is read as ONE profile, stage 1's base budget (300 chain steps,
  600,000 feeds, 3 world attempts) wherever `r0s4_honest_inc` builds at B3 (the cold first build, a battle's re-cold
  build after its first decisions were excluded, the warm chain with no survivor: the stuck full attempt) and the
  shortfall escalation at 256 tries; the re-cold build is not named above (the registration confirms the reading).
  **NOT built: `j_min` and the XR presets** (stage 0a chooses XR-J or XR-R first; XR-J's playout allocation over
  4 ≤ k < 8 worlds and its carry after a partial decision, and XR-R's "the revised world's escalation", name no
  parameter here), **nor S′H-15's fix** (§3 has stage 0 build it; it is not a §7 row). The gates: the leak test in
  sequence mode runs on K4 and CAP (`honest_inc_test`); the 600-decision identity re-score at the new code is the
  registration's (≈ 26 CPU-h), with a small banked sample re-scored identically at build time.
- **R2 built:** `main.reads.timer_model` (`Clock`, `simulate`, `replay`, the policies); its test pins the constants to
  `room-battle.ts`'s literals and reproduces the prototype on 90 banked games and on random ones; at build time it
  reproduced `timer_replay.json` on all 5,100 games, all 80 (policy, panel) rows.
- **R3 built:** `searched_play`'s optional `timer` block and decided `timer_safe` (TIMER-SAFE iff 0 forfeits and the
  cut share ≤ `max_cut_frac`, exact, within `rule8_eps` never safe), on the bank-aware replay with the cut step
  RUNNING ON (`bank_r10+bg`: the design §3's safety net, the one the live net plays; §1's row and §3's criterion (i)
  are read that way; at REF it cuts 8.4 %, the dropped-step 3.2 %); a preset carrying the net must register the net it
  played; the policy table REPORTED.
- **R4 built:** `search-cost`'s `search_cpu_min` (exclusions included) split into `searched_cpu_min` /
  `excluded_cpu_min`, the excluded decisions' distribution, `net_cut`, and the timer replay.
- **R5 built:** the live net in `main.anchors.search_side` (`why: "deadline"`, the search's own pick in the record's
  `net` block, `search_view` for the re-score); `searched_play` counts `net_cut` beside the exclusions.
- **R6 built:** `stage0/screen.py` (`sample`, `run`, `summarize` with §3's rule as code, `selfcheck`), `stage0/run.sh`
  (chunked) and `stage0/diagnose_f.py` (stage 0a's tool). The honest log is rebuilt from each capture and checked
  against every banked search-path record's `log_sha256`, `at`, `update` and `key` (all 13,781 decisions of a 360-battle
  smoke sample passed, no search). Smoke only: at REF its first three H battles reproduced the banked records exactly,
  26 / 26 search-path decisions on every re-score field (5 exclusions included); K4 and CAP ran one battle each. The
  registration must name what the harness takes as REQUIRED arguments: the sample's salt, the floor's salt,
  `--cpu-meter` (criterion (ii): `cpu_all` or the records' `cpu_s`) and stage 0a's stuck share (`--stuck-denominator`,
  `--stuck-scope`). **Criterion (ii) is confounded by the box's load:** in the smoke CAP's searched decisions did
  REF's exact work (the same `spent`, chain modes and picks) at ≈ 1.5 × its CPU, run at a busier moment; arms must be
  run interleaved per battle (or concurrently on the same battles, or on a quiet box), a choice for the registration.
  Estimated stage 0 at the sample: ≈ 86–130 CPU-h (REF, a salted REF floor, K4, CAP); its wall is set by the heavy-slot
  queue (each chunk waited 12–47 min for a slot beside the live run), not by the CPU.

Nothing else in `searched_play` changes: one searched label and its raw twin, the same imputation, bootstrap, G-PREFIX
and identity guard. A three-arm form (X, REF and raw on one set of deals) would need `searched_play` to accept two
searched labels sharing one raw twin; it is not proposed (§2: the paired X − REF it buys is unaffordable to decide).

---

## 8. Reported, never decided

- **The cross-read against REF:** G_X − G_REF = G_X − 2.00 pp, unpaired, SE ≈ √(0.71² + 0.80²) ≈ 1.07 pp at 6,900;
  and gain per search CPU-min for both (REF: +2.00 pp at 5.83 CPU-min a game, ≈ +0.34 pp a CPU-min).
- Per panel; d; the achieved SE; exclusions by panel and class; the safety net's firing share; X's own search cost
  (R4); the timer replay table; the builder report (`inc_report.py`'s, as S′-HONEST §12).

---

## 9. FINDINGs (of this draft)

- **SC-1 (NI is out of reach).** Deciding X against REF at a 1-pp margin needs ≈ 26,000–27,000 deals (≈ 4,500–4,850
  CPU-h); the draft decides X against P\* raw instead (§2).
- **SC-2 (planning on one read's point).** REF's +2.00 pp is one read; if it is a high draw, every arm's power here is
  lower than §5 says.
- **SC-3 (λ at K4 is unfitted).** piKL's λ 0.015 was fitted (L1) at 8 playouts an action; K4 keeps it, since a re-fit is
  a second lever and has no licensed data.
- **SC-4 (stage 0's agreement is a weak proxy).** Pick agreement with REF says whether X is "the same search", not
  whether it keeps the gain: REF agrees with itself under a re-draw only 63 % of the time.
- **SC-5 (seed disjointness is checked by grep, not by the framework).** Seeds 800–914 appear in no PREREG this draft
  could find; the registration re-checks every reserved range.
- **SC-6 (the stage-0 sample is REF's own trajectories).** X is screened on states REF's play produced; X's own games
  will visit others. The screen is outcome-blind, so this biases only the screen's cost and exclusion figures, which
  stage 1 measures again on X's own games.
