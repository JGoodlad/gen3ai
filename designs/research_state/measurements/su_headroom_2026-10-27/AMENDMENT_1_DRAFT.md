# S-U Amendment 1 — DRAFT, NOT REGISTERED (2026-10-27): why the searched game's cost climbs, and a cost-capped S

**Status: a draft for the orchestrator's review.** It changes nothing in [`PREREG.md`](PREREG.md) until it is
registered: committed with a new spec and a new pin, before any searched game counts. Stage 0 is HELD (orchestrator
ruling (b), 2026-10-27): it resumes after N1-P, H-CTX and X59-S finish. **The cause is now MEASURED and a
bit-identical fix is BUILT (§1b, 2026-10-27): the lean world screen, the default. It changes no world, draw or pick,
so S as registered is unchanged; the play path's code is not, so the registration at a new pin decides whether
banked stage-0 units count.**

## 1. The evidence (one game; zero CPU, from its per-decision log)

The source is stage 0's first searched game, `pstar_s__home72__00__g00` (home72, seed 0, game 0). Its log is
`~/gen3ai_archive/su_headroom_2026-10-27/anchor/su/pstar_s/cells/pstar_s__home72__00__g00/search_decisions.jsonl`.
These are its decisions from turn 10 to turn 60, the game then still in flight; search CPU is the worker thread plus
its replay children. The game ended at turn 66, a loss (the finished game, below); its 9 exclusions took 36.5 of its
77.8 search CPU-min.

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
- **The mechanism is the world screen: MEASURED (§1b).** At turn 59 the world build is 240.5 of the decision's
  243.0 CPU-s, the playouts 2.3.

**The finished game** (`python -m main.reads search-cost`, 07:43 PDT): 66 turns, counted, the forfeit-win guard
armed with 0 non-battle endings. 48 searched and 9 excluded of 57 asked; the ≥ 2-legal share 0.93; the deviation
rate 18 / 48 = 0.375. A searched decision cost 36.5 CPU-s at the median (p10 5.9, p90 130, max 243). **77.8 search
CPU-min, 78.3 whole-unit CPU-min, 82.7 min of wall**: the whole unit is ≈ search + 0.5 min, so the smoke's whole /
search ratio of ≈ 1.6 does not hold for a long game. The re-size input reads 77.8 against its bar of 50 (one game).

## 1b. The cause, measured, and the lean world screen (2026-10-27, BUILT)

**How it was measured, CPU-light.** The game was re-played exactly with no search: our searched decisions taken from
its banked log, Kakuna live, one short-lane slot for 36 s. All 71 decision records and the whole capture came out
byte-identical. Each searched decision's honest input log was kept, so any decision re-runs offline through the live
searcher's own `_search`; four of them are the parity gate's fixture
(`src/main/value_search/testdata/su_stage0_decisions.json`). The decision log now carries the split itself
(`world_cpu_s`, `playout_cpu_s`; `main.reads search-cost` reports it).

**Where the time goes.** On the reference builder, offline:
- **Turn 10:** 5.1 CPU-s, of which 2.5 build the worlds and 2.4 play the leaves.
- **Turn 59:** 243.0 CPU-s (the live record: 243.1), of which 240.5 build the worlds and 2.3 play the leaves.
- **Inside a world build, ≈ 95 % is screening replays.** About half is `core_events` (the replay and its own
  cross-checks), a third the Python decode of the 2–5 MB line it prints per battle.
- **Both factors climb.** The replays a decision rise from 137 at turn 10 to ≈ 2,000 at turn 59, and each is longer.

**Why E as drafted ("replay the true log's prefix once") cannot be bit-identical.** A screening replay is not the
truth's log. It is the candidate WORLD's log: the truth with one seen mon's item, ability and spread redrawn, from
turn 1. That battle differs from the truth's from the mon's first switch-in on (its stats, its exact HP), so only the
part before that could be shared, and in this game every opponent mon was in by turn 10.

**What was built instead: the lean screen** (`WorldSampler(lean_screen=True)`, the default; the reference path is
kept behind the flag for the parity gate only):
- **The stop.** `core_events`' `SCREEN` mode stops each screening replay after the command that first ships the
  searched side a line that differs from the truth's. The gate's verdict, the matched length and the hint are fixed
  by the lines up to that difference. Most rejected candidates diverge early (median ≈ 23 % of the log on the raw
  twin's turn 31). The replay prints only what the gate reads.
- **No impossible replays.** A chain proposal the prior rules out (an EV below 0, a total above 510) is not replayed.
  Both chain branches discard it whatever its verdict, and it draws no random number (≈ 13 % of the proposals there).
- **Two pure fixes on the same path.** The parse chain no longer copies its side's whole stream at every command, and
  the Hidden Power IV patterns are computed once per type.
- **Bit-identical.** `world_screen_parity_test` (slow) builds turns 10, 32, 43 and the turn-29 exclusion both ways,
  under `dice` seed (and turn 10 under rolls). The worlds are equal: teams, tries, facts, failure counts, leak block,
  every sampler's draws and records, and the exclusion's error word for word. The search on them makes the same
  decision, and a wrong reference is caught. Offline with the real P\*, all 57 of the game's decisions re-made on the
  lean path equal their live records field for field: searched or excluded, the pick, Q̂, log π, root V, playouts,
  tries, leak, species hits and the exclusion's error.
- **What the stop gives up:** a refusal that one of `core_events`' own cross-checks (parse == step, conservation)
  would have raised AFTER the first difference. Such a refusal is a core defect; none occurred in the parity builds.

**The speedup** (the same decisions, search CPU, the live reference against the lean screen):

| turns | searched / excluded | live (reference) search CPU-s | lean search CPU-s | ratio |
|---|---|---|---|---|
| 10–19 | 10 / 0 | 74 | 64 | 1.2× (the playouts are half of it) |
| 20–29 | 9 / 1 | 222 | 92 | 2.4× |
| 30–39 | 8 / 2 | 502 | 185 | 2.7× |
| 40–49 | 12 / 0 | 743 | 197 | 3.8× |
| 50–59 | 9 / 2 | 1,590 | 338 | 4.7× (turn 59: 243 → 53.5) |
| 60–63 | 0 / 4 | 1,538 | 203 | 7.6× (turn 60: 407 → 56) |
| **the game** | **48 / 9** | **4,669 (77.8 min)** | **1,080 (18.0 min)** | **4.3×**; the exclusions 2,188 → 344 s (6.4×) |

Late in the game the lean decision is still ≥ 90 % world building, so the screen still sets the cost; a searched
decision's median now climbs ≈ 5× from turns 10–19 to turns 50–59 (5.8 → 27 CPU-s) instead of ≈ 20× (5.9 → 129). The lean figures were measured offline at nice 19 on the agent cores
(one decision re-measured after a cargo rebuild landed in its first record); the live ones under the game's own
contention, which the reference's offline turn-59 re-run matched (243.0 vs 243.1).

**NOT built (FINDING, measured).** Dropping `core_events`' parity cross-checks from screened replays (the parse
chains, both sides' parse == step gates, the end-of-battle conservation) ran 41 real turn-59 screening batches
2.4× faster (15–17 s → 6.4 s) with identical gate inputs. That would be ≈ 1.8× more on a late decision. It is not
built because its identity is not by construction: the OTHER side's checks (its own encoder and parse chain) could
refuse a candidate the searched side's view cannot, and that would change the chain's path.

**The corrected cost model** (ESTIMATE: ONE game's per-turn costs, spread over X51's 2,800 Kakuna game lengths).

- **Method.** This game's search CPU turn by turn, cut at each of X51's 2,800 raw P\* vs Kakuna game lengths (median
  36 turns, 90th percentile 77; 15 % run past this game's turn 63, 166 past turn 100) and capped at the forfeit limit
  of 250. Past turn 63 a turn costs what turns 54–63 cost on average, flat (the low end) or growing with the log
  length (the high end). A raw game's ≈ 0.5 CPU-min is added: on this unit everything but the search was 31 s.
- **The reference builder** (what stage 0 played): median **9.7** search CPU-min a game, mean **50–73**, so the full
  read (2,400 searched games) is **≈ 2,000–3,000 CPU-h**. That is ≈ 2–3× the design's ≈ 1,000. The mean is set by
  the long games, and the long games are dear: this game's own 66 turns cost 77.8 CPU-min.
- **The lean screen:** median **4.4**, mean **11.5–15.4**, so **≈ 480–640 CPU-h** for the full read (central ≈ 560)
  and **≈ 320–420 CPU-h** for phase 1 (1,600 searched games).
- **ONE game.** Its 9 exclusions are 47 % of its reference cost; a set or a game with fewer hard-to-fit mons costs
  less. Stage 0's 20 games measure it.

## 2. The cost-capped variants

The costs are §1b's model (ESTIMATE, one game's profile over X51's lengths), in CPU-min a searched game, whole unit.

| variant | deterministic? | cost a searched game | full read (2,400 searched) | what it gives up |
|---|---|---|---|---|
| **as registered**, `r0s4@10`, on the reference builder (stage 0's) | yes | median ≈ 10, mean ≈ 50–74 | ≈ 2,000–3,000 CPU-h | nothing |
| **E′ = as registered on the LEAN screen (BUILT, §1b; the default)** | yes | median ≈ 5, mean ≈ 12–16 | **≈ 480–640 CPU-h** (central ≈ 560; phase 1 ≈ 320–420) | **nothing**: the worlds, draws and picks are the reference's, bit for bit (§1b) |
| **A. a turn window, `r0s4@10-30`** (search turns 10–30, π after) | yes | reference ≈ 5.5 mean; **with E′ ≈ 3** (this game to turn 30: 6.1 search CPU-min, 3.0 lean) | reference ≈ 220 CPU-h; **with E′ ≈ 125** (phase 1 ≈ 80) | every decision after turn 30: the last ≈ 5 turns of a median game, the whole endgame of a long one |
| B. a per-decision CPU-TIME budget, π past it | **no**: it depends on contention and the host | bounded | bounded | **REJECTED.** A searched game would no longer replay exactly, and stage 0's determinism check and a cross-host (rented box) replay would break |
| C. a deterministic screening budget (total chain steps or replays a decision; past it, an exclusion, π plays) | yes | bounded, set by the budget | depends on the budget | the decisions whose worlds are HARD to fit, chosen by difficulty. That difficulty differs by set (confound A3: U teams vs pool teams), so the exclusions would bias I. Every exclusion rate must be reported by set |
| D. 4 worlds instead of 8 | yes | ≈ ½ of its builder's row | ≈ ½ of its builder's row | it is no longer R0 §4's / L1's S, so the bridge to L1's decision-level null is lost |
| E. as drafted: replay the true log's prefix once a decision, screen only from there | — | — | — | **NOT BUILDABLE bit-identically** (§1b): a screening replay is the candidate world's battle, which leaves the truth's at the screened mon's first switch-in (every mon by turn 10 here). E′ is what was built |

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

1. **Done: the measurement and E′** (§1b). The world screen was ≥ 95 % of a late decision; the lean screen gives
   nothing up and cuts the game 4.3× (7.6× on its late exclusions). It is the default for every honest build, so a
   searched unit at any later pin plays E′. The registered S is unchanged.
2. **With E′ the read as registered is ≈ 560 CPU-h (480–640), phase 1 ≈ 370.** That is a rented-box job of ≈ 1–2
   days, or weeks on one local slot behind the current reads. A cap is no longer needed to make it affordable; it is a
   choice of estimand.
   - **A on top of E′** (`r0s4@10-30`) cuts it to ≈ 125 CPU-h (phase 1 ≈ 80), at the price of the endgame (above).
     It needs a small build: an end turn in the preset spelling, `<preset>@<from>-<to>`, stamped in `our_regime`.
   - Either way this amendment registers with a new spec and pin before a searched game counts. The one finished
     searched game re-makes all 57 of its decisions identically on E′; whether banked stage-0 units count at the new
     pin is the registration's call.
3. **Not B, not D. C only as a backstop on top of A**, for pathological decisions: a cap of ≥ 3× the observed
   turn-30 cost, every exclusion reported by set.
4. **Stage 0 is still owed** under whichever variant registers: 20 searched games, the per-set exclusion rate and
   the median whole-unit CPU (the split is now in the log). The re-size bar of 50 CPU-min (2 × the old central) is
   ≈ 10× E′'s median; a bar of 2 × the registered variant's central (≈ 30 under E′, ≈ 6 under A + E′) is to be
   registered.
