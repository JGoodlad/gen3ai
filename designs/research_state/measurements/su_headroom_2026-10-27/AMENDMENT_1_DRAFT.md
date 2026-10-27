# S-U Amendment 1 — DRAFT, NOT REGISTERED (2026-10-27): why the searched game's cost climbs, and a cost-capped S

**Status: a draft for the orchestrator's review.** It changes nothing in [`PREREG.md`](PREREG.md) until it is
registered: committed with a new spec and a new pin, before any searched game counts. Stage 0 is HELD (orchestrator
ruling (b), 2026-10-27): it resumes after N1-P, H-CTX and X59-S finish.

## 1. The evidence (one game; zero CPU, from its per-decision log)

The source is stage 0's first searched game, `pstar_s__home72__00__g00` (home72, seed 0, game 0). Its log is
`~/gen3ai_archive/su_headroom_2026-10-27/anchor/su/pstar_s/cells/pstar_s__home72__00__g00/search_decisions.jsonl`.
These are its decisions from turn 10 to turn 60 (the game ended at turn 66: `search-cost`, below); search CPU is the worker thread plus its
replay children.

| turns | searched / excluded | legal actions | playouts (`spent`) | world tries | search CPU a decision | CPU a playout |
|---|---|---|---|---|---|---|
| 10–19 | 10 / 0 | 8 | 64 | 8 (1 each) | 5–18 s, median ≈ 6 | 0.07–0.28 s |
| 20–29 | 9 / 1 | 8 | 64 | 8–9 | 8–44 s, median ≈ 13 | 0.13–0.68 s |
| 30–39 | 8 / 2 | 8 | 64 | 8–12 | 13–78 s, median ≈ 45 | 0.2–1.2 s |
| 40–49 | 11 / 0 | 3–8 | 24–64 | 8–9 | 21–142 s, median ≈ 60 | 0.3–3.6 s |
| 50–60 | 9 / 4 | 2–6 | 16–48 | 8–9 | 65–243 s, median ≈ 130 | 1.4–6.1 s |
| excluded (no consistent world) | 7 | | 0 | | 60 s (t29) → 407 s (t60) | |

Totals to turn 60: **59 search CPU-min**, of which turns ≤ 30 took 6.1, turns 31–45 took 11.6 and turns > 45 took 41.3.
The 7 exclusions took 17.6. The game passed the registered re-size bar (50 CPU-min) on its own.

**Which cause it is** (the four the orchestrator asked about):
- **Not world-build RETRIES.** Every world built at its first try (`world_tries` 8–12 over 8 worlds throughout).
- **Not more worlds, and not deeper rollouts.** Both are fixed by the preset (`n_worlds` 8, `depth` 4,
  `per_action` 8).
- **Not more legal actions.** They FALL late (8 → 2–6), and the playouts per decision with them (64 → 16–48).
- **The cost per playout rises ≈ 50×** (≈ 0.1 s → ≈ 6 s) while every count it is made of stays flat or falls. So the
  climb is in a component that scales with the GAME SO FAR, not with the search's width.
- **The exclusions are the cleanest probe of that component.** An exclusion plays no playout at all; it is world
  building alone, giving up after `world_screen.MAX_STEPS` = 200 chain steps a mon. Its cost climbs the same way:
  60 s at turn 29, 203 s at turn 50, 407 s at turn 60, beside searched decisions of 65–243 s at those turns.
- **The mechanism, MY JUDGEMENT from the code** (`main.value_search.world_screen`; the split is not logged). An honest
  world is fit mon by mon, with a chain whose every step is one screening REPLAY of the whole true log from turn 1.
  The cost is therefore (chain steps) × (replay length). The replay length grows with the turn. The steps grow with
  the facts revealed: more seen mons, and more constraints that reject a draw. Late-game exclusions are the same
  component hitting its cap. **UNVERIFIED:** the decision log does not split world-build CPU from playout CPU. The
  fix that measures it (two timers in `search_side`) is a build.

**The corrected cost model.** It rests on one game, so treat it as an ESTIMATE.
- **The finished game** (`python -m main.reads search-cost`, 07:43 PDT): 66 turns, counted, the forfeit-win guard
  armed with 0 non-battle endings. 48 searched and 9 excluded of 57 asked; the ≥ 2-legal share 0.93; the deviation
  rate 18 / 48 = 0.375. A searched decision cost 36.5 CPU-s at the median (p10 5.9, p90 130, max 243). **77.8 search
  CPU-min, 78.3 whole-unit CPU-min, 82.7 min of wall**: the whole unit is ≈ search + 0.5 min, so the smoke's whole /
  search ratio of ≈ 1.6 does not hold for a long game. The re-size input reads 77.8 against its bar of 50 (one game).
- **A median-length game (≈ 35 turns):** ≈ 10–13 CPU-min of whole-unit CPU.
- **A long game:** ≈ 80 at 66 turns, the cost a decision still climbing.
- **The read:** the mean is set by the long-game tail, ≈ 15–25 CPU-min a searched game (≈ 600–1,000 CPU-h for
  the full read); the design's original central ≈ 25 is its upper end. The PREREG's ≈ 6 CPU-min (≈ 270 CPU-h), taken from the smoke's
  turns 10–19, is wrong.

## 2. The cost-capped variants

| variant | deterministic? | cost a searched game (whole unit, ESTIMATE) | full read (2,400 searched) | what it gives up |
|---|---|---|---|---|
| **as registered**, `r0s4@10` | yes | 10–13 median, ≈ 80 at 66 turns; mean ≈ 15–25 (the tail) | ≈ 600–1,000 CPU-h | nothing |
| **A. a turn window, `r0s4@10-30`** (search turns 10–30, π after) | yes | ≈ 10 (this game: 6.1 search CPU-min to turn 30) | **≈ 400 CPU-h** (phase 1 ≈ 270) | every decision after turn 30: the last ≈ 5 turns of a median game, the whole endgame of a long one |
| B. a per-decision CPU-TIME budget, π past it | **no**: it depends on contention and the host | bounded | bounded | **REJECTED.** A searched game would no longer replay exactly, and stage 0's determinism check and a cross-host (rented box) replay would break |
| C. a deterministic screening budget (total chain steps or replays a decision; past it, an exclusion, π plays) | yes | bounded, set by the budget | depends on the budget | the decisions whose worlds are HARD to fit, chosen by difficulty. That difficulty differs by set (confound A3: U teams vs pool teams), so the exclusions would bias I. Every exclusion rate must be reported by set |
| D. 4 worlds instead of 8 | yes | ≈ ½ | ≈ 500 CPU-h | it is no longer R0 §4's / L1's S, so the bridge to L1's decision-level null is lost |
| E. an incremental world screen (replay the true log's prefix once a decision, screen only from there) | yes | ≈ flat in the turn, if the replay is the cost (UNVERIFIED) | ≈ the smoke's ≈ 6 a game ⇒ ≈ 300 CPU-h | nothing in the estimand **if it is bit-identical to the current screen**; it is a BUILD (`main.value_search.world_screen`) with a parity gate |

**Does A still test H-MID's idea?** Mostly, yes.
- H-MID's population is level positions at turn 10, and the gap it names is CONVERSION from there.
- In a median 35-turn game the window covers turns 10–30, ≈ 80 % of the searched decisions. That is where the
  mid-game is decided.
- What it loses is conversion in long endgames, the very games where the cost explodes. A U-HEADROOM that lives only
  in long endgames would read as NO-U-PREMIUM. The estimand becomes "one-ply search on V in turns 10–30", and the
  PREREG says so.
- The window is fixed by the turn number, not by any property of the state, so it selects nothing by set.
- G-PREFIX, the shared prefix and the level-at-10 subgroup are unchanged.

## 3. Recommendation (MY JUDGEMENT)

1. **First, a cheap measurement before any amendment** (a build, ≈ 1 h of agent time, a few CPU-min to check): log
   world-build CPU and playout CPU separately in `search_side`, and re-run this game's deal once it can take a slot.
   If world screening is ≥ 80 % of the climb, **E** may restore the smoke's cost with the registered S unchanged.
   That beats any cap, because it gives up nothing.
2. **If E does not pay, or until it lands: A** (`r0s4@10-30`).
   - Deterministic.
   - ≈ 400 CPU-h for the full read and ≈ 270 for phase 1: a rented-box job of ≈ 1 day, or ≈ 2 weeks on one local
     slot after the current reads.
   - It needs a small build: an end turn in the preset spelling, `<preset>@<from>-<to>`, stamped in `our_regime`.
   - Then this amendment registers with a new spec and pin. Stage 0's raw units keep counting only if the play path is
     identical. It will not be (the preset changes), so stage 0 re-plays at the new pin.
3. **Not B, not D. C only as a backstop on top of A**, for pathological decisions: a cap of ≥ 3× the observed
   turn-30 cost, every exclusion reported by set.
4. **Stage 0 is still owed** under whichever variant registers: 20 searched games, the per-set exclusion rate and
   the median whole-unit CPU. Under A the re-size bar of 50 CPU-min becomes ≈ 20 (2 × A's central); that is to be
   registered.
