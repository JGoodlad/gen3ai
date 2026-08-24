"""SEARCH-DIVIDEND probe — does a search on top of the trained policy buy strength?

The registered experiment (ledger ``7ae7e79``): three arms x a budget sweep.

| arm | what it searches | the question it answers |
|---|---|---|
| ``base`` | nothing — the policy's own argmax | the control |
| ``honest`` | K pool-consistent DETERMINIZATIONS of the opponent's never-revealed slots | what a real player could get |
| ``oracle`` | the TRUE hidden state (K=1) | the ceiling: what a perfect belief would be worth |

Budgets are a per-decision wall-clock deadline (0.5 / 1 / 3 s, plus a small 8 s batch) and buy
WIDTH in one registered order — alpha-pruned opponent actions, then determinized worlds, then CRN
dice resamples. The ORDER is the three-axis variance measurement's ranking, not a guess.

**Then they buy DEPTH** (the amendment in ``9599426``, minutes after the registration: "fixed
depth" was shorthand for cheap, not a constraint). Width is planned first and unchanged; whatever
the clock has left is spent deepening the top-``m`` actions a ply at a time, and the REALIZED depth
is recorded per decision — so a 0.5 s cell reporting depth 1 beside an 8 s cell reporting depth 2
is the finding, not a configuration. ``--max-depth`` caps it; the budget governs it.

The ``oracle - honest`` gap is the BELIEF's price and the ``honest - base`` gap is the dividend
actually available today; separating them is why the probe has three arms rather than two.

**Two opponent surfaces, and the second is the sensitive one.** Against the scripted roster every
arm saturates near 90%, which is a ceiling a real dividend can hide inside. ``--opponents self``
plays the searched side against the SAME network with search structurally off — a MIRROR whose
no-effect point is exactly 0.50 by construction. Mirror cells are read against that null, with
``--side-swap`` (on by default there) playing every game in both orientations so the team draw
differences out, and they are excluded from the anchored-ELO fit because ``self`` carries no anchor.

Performance — WHERE A BUDGET SECOND GOES
----------------------------------------
The realized WIDTH is the probe's statistical power, so the cost breakdown is a finding and not
housekeeping. Profiled 2026-08-23, oracle arm, ``--budget 1``, mirror, node search driver, on a
box carrying a live trainer (shares of the per-decision wall):

===============================  ======  ==============================================
stage                             share  per unit
===============================  ======  ==============================================
``materialize_branches``            51%   4.8 ms per arm  (of which the RESTORE was 57%)
``expand_many`` (the sim branch)    26%   2.3 ms per arm
``batch_scores`` (the critic)       15%   1.5 ms per arm at B≈88
``open_root`` (per world)            8%   55-64 ms per world, growing with the turn
python around it                     <1%
===============================  ======  ==============================================

Four things that breakdown settles, each with the number that settles it:

* **The critic forward is already BATCHED and must stay that way.** One forward scores the whole
  arm set: 1.53-1.73 ms/row at B=64-256 against **27.95 ms at B=1**, so arm-at-a-time scoring
  would cost ~16-18x. It is the CHEAPEST of the three per-arm terms, not the expensive one.
* **``torch.compile`` is a B=1 knob here, and a trap anywhere else.** Compiled is 5.45x at B=1 and
  **0.15-0.43x at B=64-256**, with 78-120 s of fresh trace per new batch size — and the search
  realizes a different arm count nearly every decision. ``--compile-extractor`` (OFF by default)
  therefore routes B=1 to the compiled graph and leaves every wider forward eager; it buys
  games/hour, never width, because the live decision sits outside the per-decision deadline —
  **1.74x end to end on a ``base`` cell** (60 mirror battles, 250.1 → 143.6 s, the one-off ~40 s
  trace included), far less on a search cell. It stays opt-in because it perturbs the forward at
  ~1e-6 and an argmax over near-tied actions could flip on that; measured, it did not (20/20
  battles identical), which is evidence and not a proof. See ``perf.py``.
* **The sim side is batched too** — one ``expand_many`` per ply carries every
  (action x candidate x seed) arm, and ``materialize_branches`` replays the shared prefix once for
  the whole set. Nothing here expands an arm at a time.
* **Nothing respawns per decision or per game.** The search-driver child is opened once per cell
  and reused (``SearchEngine.session``); ``engine.close()`` is the cell boundary.

**The two sides of a mirror game effectively SERIALIZE, and that is not fixable in-process.** The
searched side hands its search to an executor so ``choose_move`` frees ``POKE_LOOP`` — but
``materialize_branches`` then drives its replay player back THROUGH ``POKE_LOOP``, and the
opponent's own forward runs there too. Measured: the instrumented stages sum to **97% of the game
wall**, i.e. at most ~3% of anything overlaps anything. Cross-game process-level parallelism (more
cells) is the answer, and it is what the driver already does.

**What the 2026-08-23 audit changed, and what it bought.** Two things, neither of which alters a
decision's semantics: the materializer's per-arm restore stopped being a ``deepcopy`` (**4.8-6.6 →
2.5-3.0 ms per arm**), and ``world_open_s`` became a MEASURED term of the cost model instead of a
frozen 0.05 s default that was silently inflating ``arm_s`` — the number the allocator divides the
budget by. Realized arms per decision, before vs after, INTERLEAVED run-for-run in one window on
the same busy box:

=================  ==========  =========  =======  ===================================
cell                   before      after    ratio  which axis moved
=================  ==========  =========  =======  ===================================
oracle @ 1 s             53.9       97.7   1.81x   dice R 1.15-2.12 → 2.00-2.83
honest @ 1 s             64.9       88.5   1.36x   worlds K 1.71 → 1.83, m_opp up
honest @ 3 s            149.7      222.5   1.49x   **worlds K 3.23 → 4.62**
oracle @ 3 s            320.2      312.0   0.97x   none — ``--max-dice 8`` BINDS (R 7.6)
=================  ==========  =========  =======  ===================================

(4 games per cell at 1 s, 1 at 3 s.) The oracle row at 3 s is the honest negative result and worth
reading: that arm is K=1 by construction, so once ``m_opp`` caps the budget can only buy dice, and
at 3 s the dice cap is already nearly saturated. A faster search buys nothing there without
raising ``--max-dice``. **The budget still under-runs by 15-35%, and that residual is GRANULARITY
rather than waste** — a bump costs a whole world or a whole dice sweep, and a committed world's arm
set is not interruptible, so planning optimistically would overrun the deadline instead.

**Byte-identity is gated at PINNED widths**, which is the only way the question is well-posed: a
faster search legitimately buys more arms at a fixed wall-clock budget and would then choose
differently. Held at fixed caps (``--max-opp 2 --max-worlds 1 --max-dice 1``, ``--max-depth 1``,
budget far above any decision's need), 2 mirror games / **133 decisions hash identically** before
and after — including every decision's aggregated per-action scores, not merely its argmax.

**Open, sized, NOT built:** ``--search-impl rust`` would remove most of the 26% sim share — the
rust ``expand_many`` measured **0.40 ms/arm against node's 2.33-3.37 (≈8x)** — but its
``open_root`` cannot replay a LIVE-synthesized record at all: 43 of 44 decisions in a scratch game
returned ``root_failed: battle never reached the start of turn 2 (ended=false at turn 0)``, which
is a counted fallback, not a wrong answer. Fix the rust driver's record replay and the sim term
largely disappears.

Modules
-------
``determinize`` pool-consistent world sampling + THE prefix byte-identity gate ·
``record`` synthesizing a reconstruction record for a battle still in flight ·
``alpha`` the opponent-marginalization candidate set (the alpha-consumer contract) ·
``budget`` the wall-clock deadline + the width allocator ·
``deepen`` the search tree, its max/alpha-weighted backup, and the beam a ply is planned against ·
``search`` the search itself — the first ply, then iterative deepening under the clock ·
``player`` the search-wrapped eval player ·
``battery`` matched-game driver, side-swap pairing, append-only resumable results ·
``summary`` Wilson intervals, the mirror-vs-null read, and anchored-ELO deltas.
"""
