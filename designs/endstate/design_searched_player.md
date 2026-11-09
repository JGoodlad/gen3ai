# The searched player — P\* + S′h as a DEPLOYABLE player: the turn timer, the builder across a live battle, the play path

**Status: DESIGN (2026-11-09); the cost read's code BUILT (2026-11-09: the timer model, the safety net, the envelope
presets K4 and CAP, `searched_play`'s `timer_safe` flag, the stage-0 screen; §9); nothing registered.** S′-HONEST read GO-HONEST and GOAL-B-REACHED
([`../research_state/measurements/sprime_honest_play_2026-11-06/README.md`](../research_state/measurements/sprime_honest_play_2026-11-06/README.md)):
P\* + `r0s4_honest_inc@10` beats P\* alone against Kakuna (checkpoint 34, greedy) on unfamiliar teams by +2.00 pp
[+0.34, +3.51]. That read played on our own front end, which has no battle timer. **Replayed through Showdown's own
timer, the player as read is NOT deployable: 16.2 % of its searched games would have been forfeited on time** (§2).
This doc is what a deployable player is: the per-decision budget the timer allows (§2), a timer-safe envelope (§3),
how the builder's state lives across a live battle (§4), its failure modes (§5), what it needs from the play path
(§6), and what it must never do (§7). §8 is the mission question goal (b) now raises; §9 points at the cost read's
DRAFT ([`../research_state/measurements/searched_cost_2026-11-09/DRAFT_PREREG.md`](../research_state/measurements/searched_cost_2026-11-09/DRAFT_PREREG.md)).
It answers `search_ledger.md` Q19.

**Abbreviations.** P\* = `rb_x26_s2601` @280,000,170, greedy (`pstar_plateau`). S′h = `r0s4_honest_inc@10`
(`main.anchors.search_presets`): 8 worlds built from our information set I alone and carried through the battle,
TRUNC-4-K8 leaf, piKL λ 0.015, every one of our decisions from turn 10 with ≥ 2 legal actions
([`design_sprime_honest.md`](design_sprime_honest.md) §9). U = H (`heldout72_hpfix`), F (`forum72`), P72. A
SEARCH-PATH decision is one the search is asked to make (searched or excluded). An EXCLUSION is a decision where the
builder found fewer than 8 worlds consistent with I: π (P\*'s greedy pick) plays it. A CUT is a decision a deadline
hands back to π.

---

## 1. What a deployable P\* + S′h player is

**The player.** P\*'s greedy choice is computed first at every decision (the fallback is always in hand). At a
search-path decision S′h runs and its pick is played if it finishes inside the decision's budget; otherwise π's pick
is played. Every input the search reads is I: the format, both names, our packed team, the protocol lines our client
received (room lines removed) and our own commands. Nothing else, ever: no battle seed, no opponent team, no
opponent command (the leak test `honest_inv` in sequence mode, `design_sprime_honest.md` §9.3).

**"Deployable" means four properties, each checkable:**

| # | property | how it is checked | today |
|---|---|---|---|
| D1 | **never loses on time** under Showdown's timer, whoever turns it on and whenever | the timer replay (§2) on the player's own per-decision walls: 0 forfeits; and an acceptance run on a local Showdown server with the timer forced on (§6) | **fails**: 16.2 % of S′-HONEST's searched games forfeit in replay |
| D2 | **the search is a function of I read from the WIRE**, not from a true log | on our own 9XXX server, I built from the client's stream equals `info_set.extract` of the true log at every searched decision (same `InfoSet.key()`) | not built: `main.anchors.search_side` reads the TRUE input log in process and extracts I from it |
| D3 | **the measured gain survives the envelope** | the cost read's arm is played WITH its envelope (the DRAFT's stage 1) | unmeasured: the read's player had no deadline |
| D4 | **bounded resources and a clean failure path** (§5) | one searcher process per battle, killable by PID; memory capped; every failure falls back to π and is recorded | partly: the searched side runs in a worker thread of the read's own process |

The read's player satisfies none of D1–D3. "Goal (b) met with search" is a statement about the read's regime (no
timer, our front end, one CPU a game on the box); it is not a statement about a player that could sit a timed game.

---

## 2. The timer, at the source, and the budget it allows

**The facts** (`deps/pokemon-showdown/server/room-battle.ts` at the submodule's `e0551883f`, class `RoomBattleTimer`;
`[Gen 3] OU` in `config/formats.ts` declares no `timer` rule, so the defaults hold):

| item | value | where |
|---|---|---|
| the timer's state | OFF until a player asks (`/timer on`), unless the server sets `forcetimer` or the format is a blitz one | `RoomBattle` constructor, `if (Config.forcetimer \|\| this.format.includes('blitz')) this.timer.start()` |
| starting bank (non-challenge: ladder) | STARTING_TIME 150 s + STARTING_GRACE_TIME 60 s = **210 s** | constants at the file's head; `secondsLeft = starting + grace` |
| one request's hard limit | min(bank, MAX_TURN_TIME **150 s**) | `nextRequest`: `turnSecondsLeft = Math.min(secondsLeft, maxTurnTime)` |
| bank added per NEW turn | **+10 s** (singles), capped at 150 s; `accelerate`: **+5 s past turn 100**, and past turn 200 **0 s on alternate requests** | `updateTurn` |
| bank added per mid-turn request (a forced switch for one side) | +min(addPerTurn, 5) = **+5 s** to both players, uncapped | `updateTurn` |
| how time is charged | in **5-s ticks** (TICK_TIME) while our request is open; a 4.9-s answer costs 0 | `nextTick` |
| what time-out does | the player whose turn limit reaches 0 is **forfeited** (`timeoutAutoChoose` false) | `checkTimeout` |
| a challenge battle | 300 s bank, 300 s per request, a disconnection bank, no `accelerate` | constructor (`isChallenge`) |

The opponent decides whether the timer runs, and may turn it on at any request, so a deployable player behaves as if
it were always on (the bank does not drain while it is off: `secondsLeft` moves only in a tick).

**The budget per decision it allows.**
- **Sustained:** an answer under 5 s is free. Through turn 100, ≤ 14.9 s a turn (charged 10) keeps the bank level;
  turns 101–200 allow ≤ 9.9 s; past turn 200 ≤ 4.9 s (the alternate requests add nothing). Several decisions in one
  turn (forced switches) each earn only +5 s.
- **Burst:** one decision may use up to min(bank, 150 s) minus a safety reserve; the bank refills at the rates above.
- So the budget is **a few seconds a decision on average with rare long decisions paid from a ≤ 150-s bank**, and the
  long tail past turn 100 is the tightest regime.

**What the read's player spends** ([`timer_replay.json`](../research_state/measurements/searched_cost_2026-11-09/inputs/timer_replay.json),
from every counted searched unit's decision log, outcome-blind; walls at one CPU a game on the whole box at K = 14,
both SMT siblings busy):

| per search-path decision | n | mean | median | p90 | p99 | p99.9 | max | share > 150 s |
|---|---|---|---|---|---|---|---|---|
| searched | 188,429 | 6.8 s | 4.6 s | 10.2 s | 45 s | 218 s | 440 s | 0.19 % |
| excluded | 11,075 | **47 s** | 1.4 s | **190 s** | 463 s | 582 s | 655 s | **11.1 %** |
| all | 199,504 | 9.1 s | 4.6 s | 10.8 s | 99 s | 437 s | 655 s | 0.79 % |

The median decision fits the free quantum; the tail does not. An exclusion is usually a cheap probe of a stuck battle
(median 1.4 s) but the full attempts that precede and punctuate a stuck stretch (design C28) run for minutes.

**The replay** (`timer_replay.py`: the model above, our response = the decision's wall + 1 s, the opponent's own
mid-turn requests unseen, which only removes credit; a cut assumed not to change the rest of the game's costs, an
approximation since the game diverges after it):

| policy | games forfeited on time (U / F / H / P72) | searched decisions handed to π |
|---|---|---|
| as read (no deadline) | **16.2 %** / 30.1 % / 8.1 % / 10.5 % | 0 |
| a fixed 60-s deadline | 7.0 % | 0.6 % |
| a fixed 30-s deadline | 3.3 % | 1.5 % |
| a fixed 10-s deadline | 0.8 % | 10.0 % |
| **bank-aware: answer by min(bank, 150) − 10 s**, the cut build dropped | **0** | **3.2 %** (F 4.8, H 2.1, P72 2.7) |
| bank-aware, the cut build runs on (the carry stays a function of I) | 0 | 8.4 % (F 14.4) |
| bank-aware, the battle's search OFF from its first cut | 0 | 9.3 % (F 15.6) |

**Readings.** A fixed deadline never makes the player safe: past turn 100 the bank refills slower than any fixed
deadline above 10 s spends it. Only a **bank-aware** deadline gives zero forfeits by construction. At today's budgets
it cuts at least one decision in 16.7 % of battles (F 30.6 %) and costs 3–9 % of the searched decisions, depending on
what happens to the carry after a cut (§4). The dilution of the gain is UNMEASURED;
to first order (the gain spread evenly over searched decisions) it scales +2.00 pp by 0.91–0.97 (GUESS). F pays most,
and F is also the panel with no measured gain (0.00 [−2.98, +2.71]).

**The hardware these walls describe.** One CPU a game, the box full (14 units on 16 logical CPUs). A deployed player
plays one battle at a time and could own a whole core (stage 1b's SMT probe: the builder runs ≈ 1.23× faster without
its sibling busy; UNVERIFIED for whole decisions) or several (the 8 worlds' playouts are independent; not built). The
envelope must hold on the hardware the player actually runs on: D1's acceptance runs there.

---

## 3. The envelope: deterministic budgets first, a bank-aware deadline as the safety net

Two ways to keep a decision inside the bank:

- **A wall-clock deadline.** Simple, but the pick then depends on the machine's speed, not only on I. Leak-honesty
  holds (our own CPU time carries nothing about the opponent's hidden state), REPRODUCIBILITY does not: the identity
  re-score (`main.value_search.live_identity`) cannot reproduce a cut, so a read of such a player must exclude cut
  decisions from its guard or VOID.
- **Deterministic work budgets.** Every expensive path of the builder already has a COUNT budget (the cold build's B3,
  the warm chain's 300 steps / 600,000 feeds, the escalation's 1,024 tries, a fresh world's 2 attempts, a reveal's 4
  redraws: `design_sprime_honest.md` §9.2 C23–C29). A profile that caps them so the worst decision's wall on the
  target hardware stays under the bank-aware deadline keeps the search a function of I (the record replays), at the
  price of more exclusions.

**The design: deterministic budgets sized for the timer, and the bank-aware deadline only as a safety net that
should not fire.** The safety net cuts the PICK, never the build: π answers the request, the step finishes in the
searcher child, and the next search-path decisions are cut too until the carry has caught up (§2's `+bg` row, the
upper bound at today's budgets). So every searched pick stays a function of I and re-scores; only the cut decisions,
which play π, depend on the clock. A step that outlives a hard bound (a hang, not a slow step) is killed with its child
and the battle's search goes OFF (§4). A safety-net cut is recorded (`why` naming the deadline, never an exclusion),
and a read of the deployable player reports the net's firing share and treats a share above its registered bound as a
defect, not a result.

**Built (2026-11-09).** The clock is `main.reads.timer_model` (its constants pinned by its test to `room-battle.ts`'s
literals; it reproduces `timer_replay.json` on all 5,100 banked games, every policy). The safety net is the live
searcher's (`main.anchors.search_side`), on for a preset that sets `net_reserve_s` (the envelope presets
`r0s4_honest_inc_k4` and `r0s4_honest_inc_cap`, reserve 10 s): the search always runs whole and a searched decision
whose need (the decision's wall plus a cut step's overrun) passes min(bank, 150) − 10 − 1 s plays π's pick, recorded
`why: "deadline"` with the search's own pick in its `net` block, so the identity re-score compares the search's view.
Our front end has no timer, so in a read the net plays as if Showdown's were on from turn 1, on the box's walls (SP-3).
Which budget profile meets the timer, and what it costs in exclusions
and in gain, is the cost read's stage 0 and stage 1 (DRAFT_PREREG §3–§4).

**The bank the player believes.** When the timer is on, the server tells us: every request is followed by
`|inactive|Time left: N sec this turn | M sec total` (`nextRequest`). The player reads that line (a room line the
client currently drops before the reader, §6) and uses N. With no such line it uses §2's model from the battle's start
(the bank it would have if the timer had been on from turn 1, which is never above the real one).

---

## 4. The builder's state across a live battle

**One searcher per battle, in a child process the player owns.** It holds P\* on CPU (one torch thread), the Rust
core, the carry store (design C19: an LRU of 64 carries keyed by the I digest of the checkpoint each was built at) and
the world builder's validator child (the persistent Node process of S′H-9; Node stays a launch requirement). The
child is started at the battle's first request, fed I at each decision, and stopped by its PID at the battle's end.
A process, not a thread: a stuck Rust call cannot be interrupted inside a thread, and a child can be killed.

**The carry is a function of I.** At decision k the carry is the previous checkpoint's carry advanced by I's new
segments (C17–C29), and the first searched decision is built cold (C25). Two consequences for live play:

| event | what happens | cost |
|---|---|---|
| the child dies or is killed mid-battle | a fresh child rebuilds every missing checkpoint from I in order (C19), which is exactly the carry it lost | the battle's building so far (minutes late in a game); the player plays π until it is caught up, or switches search off for the battle (simpler; the design's default) |
| a decision's build overruns the safety net | π answers; the step finishes in the child; later decisions are cut until the carry catches up (§3) | §2's `+bg` row: 8.4 % of searched decisions at today's budgets; the cost read's target is ≈ 0 |
| a step outlives the hard bound (a hang) | the child is killed by its PID; the battle's search goes OFF | §2's last row is the bound if it happened at every first cut: 9.3 % |
| a reveal (a new mon, a new move) | the carried worlds' changed parts are redrawn (C21) | usually cheap; a failed redraw is the main F exclusion class (§5) |
| an exclusion | every world is kept PENDING (C24); the next decisions probe cheaply (C28) and a full attempt runs at the 4th stuck decision and each doubling | the full attempts are the long tail of §2: they must sit inside the budget profile |

**Why let a late step finish rather than drop it.** Dropping the step (§2's 3.2 % row) is not available to the
incremental builder: the next decision needs the dropped checkpoint's carry, and C19 would rebuild exactly that step,
at the same cost; the only way to drop it is to treat it as an exclusion (worlds PENDING, C24), which makes every
later carry a function of the clock. Finishing it keeps the carry a function of I and loses fewer searched decisions
(8.4 %) than switching the battle's search off at the first cut (9.3 %).

---

## 5. Failure modes, and the response to each

| failure | measured where | response | recorded as |
|---|---|---|---|
| **exclusion** (< 8 consistent worlds) | S′-HONEST: 5.55 % of search-path decisions (F 10.0 %, H 2.6 %, P72 3.3 %), FINDING S′HR-1 | π plays; part of S′h as defined (D8 of `design_sprime_honest.md`) | `why: excluded` |
| **a stuck builder** (a segment no world matches, S′H-13) | 811 of 5,100 battles held an exclusion (F 431, H 172, P72 208); per battle a median of 3–5 exclusions, p90 F 49; 147 of F's 431 never searched again | cheap probes, full attempts on the doubling schedule, inside the budget profile | `honest.inc.stuck` |
| **a hung or crashed searcher** | not observed in the read (its three drops were the peer's turn-cap fault) | the safety net returns π; the child is killed by its PID; the battle's search goes OFF | a named deadline / crash record |
| **I from the wire disagrees with the live frame** | not built; today's root guard checks the true log's replay against the live frame (`LiveSearcher._check_root`) | a correctness defect: the battle's search goes OFF and the record names it; never a silent fallback in a read | a named guard failure |
| **the timer line is missing or unreadable** | — | use §2's model bank (never above the real one) | — |
| **our socket drops** | `main.live.client` has no reconnect inside a battle (a rejoined battle replays without our choice notes) | the client's existing CRASH path; the searcher child is killed | the client's crash |

**Where F's exclusions come from** (`timer_replay.json` `exclusions`, outcome-blind). Of F's 7,350 excluded
decisions, 5,201 are cheap probes of an already-stuck battle; what starts a stuck stretch is the battle's FIRST
exclusion. On F that first exclusion built **no** world in 247 of 431 battles (57 %; 31 more failed their cold first
build) and some but fewer than 8 in 153 (36 %); on H and P72 the partial case dominates (124 of 172, 134 of 208). The
probes that keep F stuck are mostly a reveal's redraw failing: 3,213 of F's 5,201 probe exclusions failed on revised
worlds alone (`extend_revised`). So two different levers
reduce exclusions: searching with the worlds that did build (the partial case, H and P72 mostly) and making a revised
world extend (F's case). Why F's redraws fail is not diagnosed (FINDING SP-4; DRAFT_PREREG stage 0a).

---

## 6. What it needs from the play path

The read's searched player lives in `main.anchors.search_side` (the in-process core slot of an anchor read, beside our
own front end). A deployable player lives in the client that `src/main/play.py` drives (`main.live.client` over
`main.live.reader`). Neither path has what it needs yet:

| # | need | where | why |
|---|---|---|---|
| P1 | **I from the wire**: an `InfoSet` built from the client's received battle stream (room lines removed, as `main.live.reader.ROOM_SKIP` declares them) and the commands it sent | `src/main/value_search/info_set.py` (a constructor beside `extract`), fed by `src/main/live/client.py` (which already keeps the battle's whole received stream for T28) | a remote server gives no true log; today's search reads one |
| P2 | **the equivalence gate**: at every searched decision of N battles on our own 9XXX server, the wire I's `key()` equals `info_set.extract`'s on the true log | a test beside `src/main/anchors/search_side.py`; the anchors path then searches from the wire I and keeps the true log for this guard only | the leak test proves the builder reads only I; this proves the I is the one the client saw |
| P3 | **a searched policy** with an asynchronous choice, a deadline and I: today `main.live.policy` policies read only the reader's frame, synchronously | `src/main/live/policy.py` | the search must run off the event loop (the keepalive) in its own child process (§4) |
| P4 | **the clock**: parse `\|inactive\|Time left: …` and hand the bank to the policy | `src/main/live/client.py` (it drops `inactive` before the reader today) | §3 |
| P5 | **the record**: the `gen3_live_search_decision_v1` line per decision, plus the safety net's fields | the searched policy | `main.reads search-cost` and the timer replay read it |
| P6 | **an acceptance run with a real timer**: selfplay on a LOCAL Showdown server on a 9XXX port started with `forcetimer`, the searched player on both sides or against a bot | a measurement folder; the Node server is legitimate here because the test is about Showdown's timer (root rule 1) | D1 on the deployed hardware; our `ws_frontend` has no battle timer |
| P7 | **refusals in code**: the searched player is refused with `--mode ladder` and with `--server official` until the owner says otherwise | `src/main/play.py` | §7 |

**For reads, `main.anchors` keeps its role**: Kakuna reads play through the anchor lane and our front end. The cost
read's arm is a registered preset there; the envelope's budget profile is part of the preset; the timer is applied by
replay (the front end has none), which is conservative because the box's walls are slower than a dedicated core's.

---

## 7. What it must NOT do

- **No ladder.** `--mode ladder` refuses in an agent session unless the OWNER wrote the approval file; no agent creates
  or asks for it (`src/main/ladder_guard.py`; `design_ladder_campaign.md`, its ladder stage deferred). A searched
  player is no exception, and P7 refuses it there in code as well.
- **No public server** except the gated self-vs-self acceptance series (`--public-acceptance`, both accounts ours, the
  orchestrator's go, a green `src/main/ladder_drift_scan.py` first). Never play, challenge or chat with a human.
- **No information beyond I**: no true log, seed or opponent command reaches the search on any path (P2's gate, and the
  leak test in sequence mode on the deployable path's own code).
- **No GPU, never ports 8000 / 8001, one battle at a time per searcher.** Live play is CPU-only (`main.live.policy`).
- **No claim it was not measured for.** "Goal (b) met with search" is S′-HONEST's regime only; a preset with an envelope
  is a different player until its own read; nothing here licenses a training, league or distillation change.

---

## 8. The mission: what the evidence says the line should become

**The tension (resolved 2026-11-09).** `UNDERSTANDING.md` §1 said the mission was a generalist "not a specialist, not
a search agent", while goal (b), beating Kakuna on unfamiliar teams, is met only with search. The orchestrator, under the
owner's 2026-11-09 delegation, adopted the mission sentence now in `UNDERSTANDING.md` §1 (Decision record, 2026-11-09
mission row).

**What the evidence says.**
- **The network alone has stopped improving on goal (b).** P\* is at its plateau (the continuation to 480M read FLAT
  against Kakuna, +0.08 pp [−3.90, +4.06]); no training lever's interval clears 0.50 on U (STATUS §1, goal 2b).
- **Search adds a measured, leak-honest gain on top of it**: +2.00 pp [+0.34, +3.51] (S′-HONEST).
- **The network cannot yet absorb that gain**: X65 ExIt-0 read NOT-DELIVERED, and the learnability probe matches
  "S′'s departures unlearnable at this scale" (leak-inclusive labels; ExIt-0H on the honest labels is unregistered).
- **The search that pays is ONE PLY** in the owner's sense (2026-09-30): only our action branches at the root, the
  network plays both sides of every 4-turn playout, and the critic scores the leaf. No tree, no search to terminal.
  Whether a 4-turn playout leaf is inside "one ply" is `design_search_next.md` FINDING D-10's reading, never confirmed
  by the owner, and it now carries goal (b) (FINDING SP-5).
- **The owner's side constraint (ladder-playable) binds the searched player more than the network**: as read it
  forfeits 16 % of timed games (§2).

**What it became** (adopted 2026-11-09, `UNDERSTANDING.md` §1): the network stays the product and the yardstick, and
one-ply, leak-honest search at play time is a separately measured layer that must fit the turn timer.

---

## 9. The cost read (DRAFT, not registered; its code built)

[`../research_state/measurements/searched_cost_2026-11-09/DRAFT_PREREG.md`](../research_state/measurements/searched_cost_2026-11-09/DRAFT_PREREG.md):
stage 0 screens candidate presets offline on S′-HONEST's banked battles (zero games, outcome-blind: cost, the timer
replay, exclusions, pick agreement); stage 1 plays the cheapest preset that passes, with its envelope, against P\* raw
on 6,900 fresh U deals under `searched_play` (GO-HONEST iff LB > 0), with a new decided `timer_safe` flag. A
non-inferiority read against `r0s4_honest_inc@10` at a margin that matters is unaffordable (≈ 27,000 deals at 1 pp,
≈ 4,800 CPU-h; the DRAFT §2). The DRAFT's §7 builds are in (its §7 says which, and what is not: the F-aimed XR
presets, which stage 0a chooses, and S′H-15's fix).

---

## FINDINGs

- **SP-1 (the read's player is not deployable under Showdown's timer).** Replayed through the ladder timer, 16.2 % of
  S′-HONEST's searched games forfeit on time (F 30.1 %); 0.79 % of its search-path decisions run longer than the 150-s
  per-request cap on their own. The read stays valid in its regime (our front end has no timer); a deployable player
  needs §3's envelope and its own read.
- **SP-2 (exclusions are 29 % of the search CPU, not 5.5 %).** The read's README quotes 249 search CPU-s a game: the
  SEARCHED decisions only. The excluded ones add 101 CPU-s a game (F 209 of F's 518), so the search costs ≈ 350 CPU-s
  a game, matching `search_cost.json`'s `game_search_cpu_min` mean 5.83. An exclusion averages 47 s of wall. Any cost
  figure for S′h must name which of the two it is.
- **SP-3 (the walls are box walls).** Every wall here is one CPU a game on a full box with SMT siblings busy; a
  dedicated core is faster by an UNVERIFIED factor (the builder-only SMT probe: ≈ 1.23×). D1 must be checked on the
  hardware that plays.
- **SP-4 (F's exclusion mechanism is undiagnosed).** F's battles go stuck mostly through a revised world failing to
  extend after a reveal, and F's first exclusion builds no world at all in 57 % of its battles (H 24 %, P72 31 %).
  Whether that is F's off-prior sets, the matcher's coverage (S′H-13) or something else is not known; the DRAFT's
  stage 0a diagnoses it before any exclusion lever is chosen.
- **SP-5 (D-10 now carries goal (b)).** The searched player's 4-turn greedy playout leaf is one ply only under
  `design_search_next.md` D-10's reading of the owner's 2026-09-30 rule, which the owner has not confirmed. A
  deployable searched player should not be built past P2 without that confirmation.
- **SP-6 (the timer model is approximate in two directions).** It omits the +5 s the opponent's own mid-turn requests
  add to us (conservative) and models the past-turn-200 alternation as odd turns (rare: 20 searched games reached the
250-turn cap); a cut
  is assumed to leave the rest of the game's costs unchanged (the game diverges after it). The bank-aware policy's 0
  forfeits holds by construction, whatever the walls.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-11-09 | **This doc CREATED** (the search-next designer, Q19, orchestrator brief; design and banked-data analysis only, no game, no search run) | a deployable player = P\* + S′h inside a timer-safe envelope: deterministic count budgets sized for the timer, a bank-aware deadline (min(bank, 150) − 10 s) only as a safety net that cuts the pick and lets the step finish (a hang kills the child and switches the battle's search off); one searcher child process per battle; I read from the wire with an equivalence gate against the true log on our own server; the bank read from the server's `\|inactive\|` line. The timer facts read at the source (§2). The mission sentence PROPOSED to the orchestrator, not written into `UNDERSTANDING.md`: "**The mission.** A Generation 3 OverUsed (gen3 OU) generalist that keeps improving: not a specialist, and not a search-first agent: the trained network is the product and the yardstick for every training lever, and one-ply, leak-honest search at play time (only our action branched at the root, the network on both sides of every playout, inside the ladder's turn timer) is a separately measured layer on top of it, never a substitute for improving the network." | a fixed per-decision deadline (§2: never safe past turn 100); a wall-clock deadline as the primary envelope (unreproducible picks; the identity guard cannot re-score a cut); dropping an overrun step (the carry would become a function of the clock); switching search off at every overrun (9.3 % lost against 8.4 %); the search in a thread (cannot be killed); deploying `r0s4_honest_inc@10` as read (16.2 % timer forfeits); editing `UNDERSTANDING.md` §1 here (the orchestrator decides) | `timer_replay.json` and `timer_replay.py` in `../research_state/measurements/searched_cost_2026-11-09/inputs/` (sha256 `842bb278…577c`); `deps/pokemon-showdown/server/room-battle.ts` (`e0551883f`); `../research_state/measurements/sprime_honest_play_2026-11-06/README.md` |
| 2026-11-09 | **Mission sentence ADOPTED** into `UNDERSTANDING.md` §1 (the orchestrator, under the owner's 2026-11-09 "lead the ship, make reasonable decisions"), and with it the reading "one ply = only our action is branched at the root; playouts run the network on both sides" (`design_search_next.md` D-10) | the sentence proposed in the row above, verbatim; D-10's reading adopted, **OWNER-CONFIRMATION PENDING** (the owner may veto; SP-5's gate on building past P2 stands until confirmed) | "not a specialist, not a search agent" (goal (b) is met only with search); leaving D-10 unread by the owner | the owner's 2026-11-09 delegation; `design_search_next.md` D-10; FINDING SP-5 |
| 2026-11-09 | **The cost read's code BUILT** (the searched-cost builder, orchestrator brief; no read registered, no game) | the DRAFT's §7: `SearchPreset` gains `budget` and `net_reserve_s` (left out of `as_dict` at their defaults, so the four earlier presets are byte-identical); presets `r0s4_honest_inc_k4` (K4) and `r0s4_honest_inc_cap` (CAP: stage 1's base budget wherever the builder builds at B3, a 256-try shortfall), each with the net at 10 s; the timer model `main.reads.timer_model`; the safety net in `main.anchors.search_side` as the RUNS-ON replay (`bank_r10+bg`, §3: the step finishes, its overrun delays the next decisions), cutting only the pick; `searched_play`'s optional `timer` block and decided `timer_safe` flag on that replay; `search-cost` with the exclusions' CPU split out and the timer replay; the stage-0 screen and the stage-0a diagnosis tool | the dropped-step replay (`bank_r10`, 3.2 % at REF) as the flag's policy: the incremental builder cannot drop a step (§4), so the player the flag licenses is the runs-on one (8.4 % at REF); a wall-clock deadline inside the search; `j_min` and the XR presets before stage 0a chooses one (XR-J's allocation over k < 8 worlds and XR-R's escalation are unspecified); S′H-15's fix (not in the DRAFT's §7) | `src/main/reads/timer_model_test.py` (the source pins; the prototype on 90 banked games and random ones); `search_side_test`, `rules_searched_play_test`, `honest_inc_unit_test`; the DRAFT's §7 |
