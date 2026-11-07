# S′-HONEST — does S′ still beat P\* against the real Kakuna when its worlds see nothing the live player cannot?

**Status: STAGE 1 BUILT 2026-11-05 (the builder, its leak test, the timing and exclusion check: §8); the INCREMENTAL
builder `r0s4_honest_inc` BUILT 2026-11-06 with its stage 1b (§9: ≈ 6.2 CPU-min a searched game on U at K = 6, ≈ 4.9×
below `r0s4_honest`; U exclusions ≈ 3.4 %); STAGE 2 (the replication's PREREG, its reviews, its registration) NOT
STARTED.** §1–§7 are the design as written before the build; §8 is the build as it stands, every choice the spec left
open, and the checkpoint's measurements; §9 is the incremental builder. No PREREG is written and nothing is
registered. S′-LIVE's GO-LEAKY ([`design_sprime_live_play.md`](design_sprime_live_play.md)) licenses
this replication and nothing else.

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170, greedy. Kakuna = Metamon's Kakuna, checkpoint 34, greedy. U = H, F and P72.
- S′ = `r0s4_rolls@10` (`main.anchors.search_presets`): 8 worlds `dice="rolls"`, TRUNC-4-K8, piKL λ 0.015.
- **I** = the searched side's INFORMATION SET at a decision: the format, both names, our packed team, every protocol
  line our side was shipped up to the decision (its requests included), and our own commands.
- A **segment** = the lines one command ships when it resolves a turn or a forced switch alone.
- **Fresh dice** = dice drawn from a seed derived from I only, never from the battle's seed.
- C / T / P = a die that is drawn only when a hidden fact holds (count) / a die always drawn whose outcome a hidden
  fact moves (threshold) / a die whose value is stored and read later (persisted).

---

## 1. The leak inventory (read from the code at `fcfe2575e`)

The live searcher hands the world builder `honest_log` (`src/main/anchors/search_side.py:134-153`). It carries the
battle's TRUE seed, BOTH true packed teams and the opponent's own commands. The builder and the search read the truth
through six channels. The
draw-site census behind L1–L2 covers every PRNG draw of the Gen-3 engine (49 sites). Exposure is measured
outcome-blind on S′-LIVE's 103,616 search-path decisions
([`sprime_honest_2026-11-05/`](../research_state/measurements/sprime_honest_2026-11-05/README.md)).

| # | channel | where | what the search gets that the live player lacks | in `search_ledger.md` §6.1? | exposure in S′-LIVE |
|---|---|---|---|---|---|
| **L1** | **The true seed in the prefix.** Every candidate and every returned world is replayed from turn 1 on `log["seed"]` | `src/main/value_search/worlds.py:434` (`Root.rec`), `:922-924` and `:963-967` (world logs keep the seed); `src/rust_env/src/search/game.rs:80-108` (the playout replays it) | the truth's dice, so a world is accepted only if it is consistent with them, not with what was printed | in part (L1a) | every decision |
| L1a | C: a hidden fact adds or removes a die and prints nothing, so the stream desyncs and the gate rejects it | speed ties: `src/rust_sim/src/event.rs:193`, `src/rust_sim/src/turn/speed.rs:36-57` (at every move and residual when the actives tie), `:659-716` (action order); contact abilities `src/rust_sim/src/turn/status.rs:165`, `:200`; Focus Band `src/rust_sim/src/turn/items.rs:744`; King's Rock `:808`; Shed Skin `src/rust_sim/src/turn/residuals.rs:1797`; residual ties on hidden Leftovers / Speed Boost / Rain Dish `:926`; Shield Dust `src/rust_sim/src/turn/secondaries.rs:110`; Magma Armor `src/rust_sim/src/turn/status.rs:495-510`; Trace `src/rust_sim/src/event.rs:321`; Choice-lock and trap tie sorts `src/rust_sim/src/turn/speed.rs:145-213`, `:262-313` | a seen mon's exact Speed (a point tighter than honest play), its silent dice item or ability | speed ties, King's Rock, Focus Band, contact abilities and Shed Skin listed; Shield Dust, Magma Armor, Trace and the residual / Choice-lock / trap ties not | **a seen mon's Speed pinned a point tighter in 23.5 % of decisions** (94 of 400 sampled; 5.4 % of 2,057 (decision, mon) pairs), against the bank's 20.1 % ([`speed_pin.json`](../research_state/measurements/sprime_honest_2026-11-05/speed_pin.json)); the dice items' prior mass removed 6·10⁻⁵ a mon; none of U's teams holds King's Rock, Focus Band, Shed Skin or Flame Body |
| L1b | T: the die is drawn either way, the TRUE die value decides which hidden facts pass | accuracy (Sand Veil, Compound Eyes, Hustle) `src/rust_sim/src/turn/speed.rs:628-650`; crit (Scope Lens, Battle / Shell Armor) `src/rust_sim/src/turn/moves.rs:1776`; Serene Grace's doubled secondary `src/rust_sim/src/turn/secondaries.rs`; Quick Claw's order `src/rust_sim/src/turn/speed.rs:417` | a hidden item or ability pinned by one die (Blissey's Natural Cure vs Serene Grace is the U case) | **no** (§6.1 names dice-SILENT facts only) | small: BrightPowder and Lax Incense are banned by the format; none of U's teams holds Scope Lens, Sand Veil or Quick Claw |
| L1c | a die picks a hidden SLOT | Sleep Talk `src/rust_sim/src/turn/status_moves.rs:429`; Assist `src/rust_sim/src/turn/callers.rs`; Roar / Whirlwind over the party array `src/rust_sim/src/turn/switch.rs:732` | a world must order its moves and its party as the truth does | **no** | a Sleep Talk is on a team in 570 of S′-LIVE's 2,705 captured games |
| L1d | the damage roll's VALUE | `dice="rolls"` keeps the stream's roll (the TRUE roll) whenever it prints the same thing, else the nearest consistent one, and pins Focus Band / Endure / False Swipe / Beat Up / Future Sight / crash sites to it: `src/rust_sim/src/rolls.rs:15-22` | the opponent's exact HP follows the truth's rolls inside the shown percent | **no**: §6.1 calls this leg cured; only the REJECTION by the true roll is | every hit on the opponent |
| **L2** | **Persisted dice at the root.** The playout reseeds only the PRNG at the root, so every value drawn before it is the truth's in every world | `src/rust_env/src/search/game.rs:162-166`, `src/rust_sim/src/bridge.rs:1885-1888`; sleep `src/rust_sim/src/turn/status.rs:653` (2–5), confusion `src/rust_sim/src/turn/secondaries.rs:302` (2–5), Encore `src/rust_sim/src/turn/status_moves.rs:870` (3–6), Disable `:2617`, partial trap `src/rust_sim/src/turn/moves.rs:1992` (3–6), Outrage / Thrash / Petal Dance lock `:2165` (2–3), Uproar `:2173`, a pending Future Sight / Doom Desire's damage `:3498`, and the Quick Claw roll of the root turn `src/rust_sim/src/state.rs:2329` (drawn at endTurn, `src/rust_sim/src/turn/driver.rs:1088`): a true FUTURE die | exactly when a sleeping mon (ours too) wakes, when confusion ends; whether a Quick Claw procs this turn | the counters yes ("unknown", UNVERIFIED); the Quick Claw roll and the Future Sight damage **no** | **6.9 % of decisions** (16.4 % of games): a mon asleep (not Rest) on our side 3.1 %, on theirs 3.7 % (active: ours 0.67 %, theirs 0.70 %); confusion 0.05 %; a pending Future Sight / Doom Desire 0.08 %; Encore, Disable, traps, locks, Uproar 0; Quick Claw 0 ([`exposure.json`](../research_state/measurements/sprime_honest_2026-11-05/exposure.json)) |
| **L3** | **The true opponent team as the scaffold** | | | | |
| L3a | the per-mon screen holds every OTHER opponent mon at its TRUE facts, and the screened mon carries its TRUE unrevealed moves | `src/main/value_search/world_screen.py:319-345` (`entry` on the true set), `:422-428` (each job starts from the true entries); `src/main/value_search/worlds.py:717` | a seen mon's draw conditioned on its teammates' true facts and on its own true unrevealed moves (through L1, L1c) | as "the factorisation", a residual, not called a leak | every decision with ≥ 2 seen mons |
| L3b | the TRUE slot order: seen mons at their true slots, never-seen at the true never-seen slots, a drawn move equal to a true unrevealed one put at its TRUE slot | `src/main/value_search/world_draws.py:184-201` (`align_to_truth`), `src/main/value_search/worlds.py:895` | feeds L1c, and π's slot-indexed play in the playouts | **no** | every decision |
| L3c | the true team's turn-0 gender-draw slots | `src/main/value_search/worlds.py:635`, `:818` | the gender coin of a never-seen mon | yes | — |
| L3d | a seen mon's other packed fields copied from the truth (happiness, an explicit Hidden Power type) | `src/main/value_search/worlds.py:737`, `:894` | Return / Frustration's power | **no** | 40 of 32,460 U mons carry a non-default happiness (0) |
| **L4** | **The opponent's PRIVATE commands.** The log holds what the opponent chose, executed or not | the template: a choice the searched side never saw executed is rendered at its TRUE move slot (`src/main/value_search/worlds.py:513-515`); a token it cannot attribute is kept verbatim (`:500-502`) | a world must hold the move it chose while asleep, paralysed or KO'd first (a Hidden Power of some type for poke-env's untyped `move hiddenpower`); its legality (Taunt, Choice lock, Disable, 0 PP, Sleep Talk while asleep) filters worlds | **no** | 34.9 % of decisions hold a rewritten choice, 31.1 % a verbatim one (every one `hiddenpower` or `struggle`), 0.8 % a rewritten switch ([`decision_records.json`](../research_state/measurements/sprime_honest_2026-11-05/decision_records.json)) |
| **L5** | **Keying by the truth.** The deal key hashes the true seed and both true teams and seeds every world draw and every playout's dice | `src/main/anchors/search_side.py:97-103`; `src/main/value_search/worlds.py:396-398`, `:720`; `src/main/value_search/search.py:83-85` | nothing usable (a hash), but any hidden perturbation changes the whole output, so no pointwise invariance test can pass | no | every decision |
| **L6** | **Exclusions.** Whether S′ or π decides depends on whether worlds fit under L1–L4, i.e. on the truth | `src/main/anchors/search_side.py:299-306` | the decision to search at all | no | 5.8 % of decisions; F excludes ≈ 12× H and P72 (S′LIVE-R1) |

**Checked and not a leak.** The playouts' dice after the root (fresh seeds, `src/main/value_search/search.py:148`).
The opponent's choice at the root (cut from the log, `src/main/anchors/search_side.py:143-144`). The forward on the
live row. The reference rows and lines (our own view). `opp_actives` (`src/main/value_search/worlds.py:490` reads the
opponent's requests, but only the active species, which the switch lines show). `opp_conditions` (the opponent's
exact HP, read only under the `hp` lever, off). Taunt (2 turns, fixed), Yawn, Perish, Wish and the Toxic stage
(deterministic). The truth-passes guards (`src/main/value_search/worlds.py:597-609`, `:622`) raise, so they fail a game and never steer a choice;
none fired in S′-LIVE.

**Is S′-LIVE's +4.60 resting on a leak bigger than §6.1 says? Not shown.** The channels §6.1 omits are small or absent
in U (L1b, L2's Quick Claw and Future Sight, L3d) or small per instance (L1d: the opponent's exact HP inside its shown
percent; L4: the legality of a choice that changed nothing). The two that §6.1 lists are the large exposures, and on
S′-LIVE's own states they are about the size §6.1 gives or was silent on: the speed pins in 23.5 % of decisions (the
bank's 20.1 %) and the counters in 6.9 % (measured here for the first time). How much of +4.60 any of them
carries cannot be split from the record; only the replication measures it (FINDING S′H-1).

## 2. The design: the search is a function of I alone

The principle: everything the search computes at a decision is a deterministic function of I and of randomness keyed by
I. The battle seed, the opponent's packed team, its requests and its unexecuted commands never enter.

| # | piece | closes |
|---|---|---|
| D1 | **The input is I, not the log.** A new entry point takes I alone: no seed, no opponent team. The reference lines and rows come from I itself (`present()` builds a view from one side's stream), so no truth replay runs | the possibility of any L |
| D2 | **The opponent scaffold from I.** The team size from `\|teamsize\|` (§8 C1: the gen-3 protocol names it); each seen mon's species, nickname, level, gender and shininess from its switch details; its revealed moves, item and ability from the ONE log source on our lines (`agents.info_levers.revealed`); the rest drawn as today. Party order: the lead, then by first appearance, never-seen last. Every gender NAMED, ours included, so turn 0 draws none. Default happiness. Unrevealed moves in a drawn slot order | L3b, L3c, L3d |
| D3 | **Fresh dice, matched segment by segment (Rust).** A world is its team, its command list and a RESEED SCHEDULE. Before each command that resolves a segment, the replay reseeds from (key, world, segment, try) for try 0, 1, … ≤ T and keeps the first try whose new lines for our side (requests included) equal I's; the schedule records the kept seeds, and the gate and the playout tool replay it identically. An empty schedule is today's replay byte for byte, so `r0s4_rolls` is unchanged. No die of the truth is ever drawn. A damage-roll control on fresh dice (each damage site matched to the segment's next HP line) is the cost lever of §4, not a correctness piece | L1a–L1d |
| D4 | **Persisted dice are the world's own.** A counter is drawn on the world's fresh dice at its onset and conditioned by every later segment (a wake on the truth's turn, or not yet). A segment no try matches backtracks to the latest onset, in I's lines, of a counter still running (sleep, confusion, Encore, Disable, a trap, a lock, Uproar), up to R re-rolls, before the candidate fails. At the root each persisted value, the Quick Claw roll included, is the world's own draw, so no root resample is needed | L2 |
| D5 | **A screen with no true background.** One JOINT chain over every seen mon's item, ability and spread replaces the per-mon screens. Its score is the matched length under D3. A step moves the mon the first unmatched segment implicates (the opponent's active mon there, with `divergence_hint`'s stat), with MonSampler's own kernels (guided, local, a prior draw). Once I fits: Metropolis–Hastings over a uniformly chosen mon, thinned to the J worlds. Never-seen mons are drawn as today. D3's tries are keyed, so a candidate's verdict is deterministic | L3a |
| D6 | **The opponent's commands from I.** Each executed choice from its line (a `\|move\|` or a `\|switch\|`). Each unexecuted one is a stand-in token the Rust replay resolves to the world mon's first legal move that ships nothing | L4 |
| D7 | **Keyed by I.** key = sha256 of I's canonical text; every draw (worlds, tries, playout seeds) derives from it | L5 |
| D8 | **Exclusions are a function of I.** A decision whose chain does not fit I in its budget is excluded and π plays. RISK: without the truth's background the joint chain may exclude more often (§7) | L6 |
| D9 | **The preset** `r0s4_honest`, spelled `r0s4_honest@10`: `r0s4_rolls`'s worlds, leaf, rule and uniform weights, built by D1–D8. `r0s4` and `r0s4_rolls` stay byte-identical | — |

## 3. The leak test (INV): perturb only hidden state, and the record must not move

**The statement.** For a decision D and an alternative truth D′ that ships our side exactly I (checked by replay), the
search's decision record (searched or excluded, the pick, Q̂, log π, root V, the world teams) must be BYTE-IDENTICAL.
It is pointwise, so it passes or fails deterministically (root rule 8). A distributional check would need a tolerance;
D7 makes the pointwise one possible.

**The alternative truths**, on a registered sample of S′-LIVE's kept decisions, in a registered order (first N valid):

| # | perturbation | how D′ is made |
|---|---|---|
| P1 | a never-seen mon's species and set | another prior draw at that slot |
| P2 | a seen mon's unrevealed move | another prior draw for that slot |
| P3 | a seen mon's spread where I does not pin it | a ±4-EV move kept only if D′ replays to I |
| P4 | the battle's dice and every hidden counter | the TRUE team and commands on a fresh reseed schedule (D3) that ships I, or an honest world itself |
| P5 | the opponent's unexecuted choices | another legal move at that request |

A D′ that does not replay to I byte for byte is invalid and not counted.

**Today's builder FAILS it.** Unpatched, P1–P3 change the deal key (L5), so every draw and the record move at once.
That alone proves only the keying, so the test also runs today's builder with its key pinned to I (a test-only
patch). Then P1 moves the true gender-draw slots (L3c), P2 the true slot order (`align_to_truth`, L3b), P5 the
template where a choice's legality differs (L4), and P4 changes which worlds pass the gate (L1) and every persisted
counter at the root (L2): B1's reseed schedule, passed through `Root.rec`, is what lets today's builder replay a D′ that is not on one seed. The
registered sample fixes which cases must differ, so the failure is deterministic too. **The new builder PASSES**: its
input is I. **The control**: perturbing I itself (one of our lines) must change the record, so the test is not
vacuous.

## 4. Cost: measured inputs, and the estimate they give

Measured 2026-11-05 at `fcfe2575e` on S′-LIVE's kept honest logs, outcome-blind
([`sprime_honest_2026-11-05/`](../research_state/measurements/sprime_honest_2026-11-05/README.md)):

- **Today's search** (S′-LIVE's records): 97,601 searched decisions at 7.72 CPU-s (worlds 5.07, playouts 2.42) and
  6,015 exclusions at 29.4 CPU-s. Search CPU 258 CPU-h: worlds 138, exclusions 49, playouts 66. **World building is
  72 % of search CPU**, ≈ 4.2 of a searched game's 6.2 CPU-min (the S′-LIVE record's mean).
- **Today's builder, a timing run** on 16 kept decisions (turns 10–24): 51–417 screening replays a decision (median
  140), 11.8 ms each, 1.5 CPU-s a decision (median); all 16 built.
- **Fresh dice on the TRUE team** (40 kept decisions, 950 segments, 64 fresh seeds each):
  - one fresh seed reproduces a whole prefix with probability ≈ 10⁻²⁸ (median), so plain rejection is impossible and
    D3's per-segment matching is required;
  - a segment matches on **0.51** of fresh seeds: **0.90** when it prints no damage, **0.42** when it does (80 % of
    segments); 6.4 % of segments matched 0 of 64, every one of them a segment that prints damage;
  - **1.6 tries a segment without damage, ≈ 21 with** (a rate floored at 1/128), ≈ 376 a prefix (median);
    **0.15 ms of CPU a try**.
- **What that gives a candidate replay:** ≈ 376 tries ≈ 58 ms more than today's ≈ 12 ms without a damage control
  (world building ×≈ 5–6); ≈ 2 tries a segment, a few ms more, with one (×≈ 1.2–1.5).

**ESTIMATE per searched game**, from S′-LIVE's 6.2 CPU-min (worlds 4.2, playouts 1.5, the rest 0.6) and a GUESS of
×1–2 for the joint chain against today's screen:

| | world building | per searched game | 3,840 deals (§5), searched units |
|---|---|---|---|
| with the fresh-dice damage control | ×1.2–1.5 | **≈ 7–15 CPU-min** | ≈ 450–960 CPU-h |
| without it | ×≈ 5.9 | ≈ 27–51 CPU-min | ≈ 1,700–3,300 CPU-h |

The raw arm adds ≈ 31 CPU-h (S′-LIVE's 29 CPU-s a unit). S′-LIVE's life 4 at AM-K6's concurrency 6 played 25–50 deals
an hour, ≈ 2.8–5.6 CPU-h an hour, so the controlled case is ≈ 3–14 days of wall time. The build's timing run (§6, B8) replaces this estimate before any registration.
**FINDING S′H-2:** the brief asked for the cost from a timing run of the new builder, which does not exist. These
figures are measured inputs and an estimate from them.

## 5. The replication read (inputs for its PREREG; not registered)

- **The same read as S′-LIVE, with one lever changed.** P\*, Kakuna ckpt 34 greedy, the peer pack, U, lone one-game
  units paired by deal, G-PREFIX with the `peer_first` classifier, the 10-h watchdog, the identity re-score of a hashed
  sample, and rule `searched_play` (or a dated variant) with post-treatment drops imputed at their worst case. The arm
  plays `r0s4_honest@10` instead of `r0s4_rolls@10`.
- **The engine.** D3 changes the Rust sources, so the engine stamp moves off E4 (its `port=` part hashes them). The new
  stamp is registered with the pin. `r0s4_rolls`'s byte identity at the new stamp is shown by re-running S′-LIVE's
  identity re-score (its registered 600 decisions) and the bank tooth there.
- **Fresh deals**, disjoint from S′-LIVE's (U unit seeds 300–344), dlive's (200–244) and X65 ExIt-0's second look
  (600–674, FINDING X65-F3).
- **The labels come free for ExIt-0H.** Every searched decision's record keeps what S′-LIVE kept for distillation
  (Q̂ and log π per legal move, root V, the frame and log digests; the anchor lane keeps its captures), §8.1 C15.
- **`peer_long_tail_fault` from the start.** It is Kakuna's turn-250 race (S′LIVE-OP-6 / OP-7): a post-treatment drop,
  imputed at its worst case, counted against the 0.5 % cap. It does not hold the lane alone. A new error class, a
  panel's drops reaching 3, or more than 2 new exhaustions in 6 h does.
- **The operator procedure.** Supervisor lives as transient `systemd-run --user` services whose script checks the
  engine stamp before `exec`ing `main.reads resume`. Concurrency 6 with `GEN3AI_HEAVY_SLOTS=6` in that supervisor only,
  `nice -n 19 taskset -c 5-7,13-15`. **Concurrency stays 6 while a PINNED training run is live** (the coordinator,
  2026-11-05): X65-F1's contention record (`5fba84342`,
  [`contention_trainer_vs_kakuna_2026-11-05/`](../research_state/measurements/contention_trainer_vs_kakuna_2026-11-05/))
  measured a production trainer pinned to cores 0-4, 8-12 slowing a 6-wide S′ workload on 5-7, 13-15 by ≈ 13 %
  (per-decision CPU ratio 1.153 [1.148, 1.158]), so AM-K6's drop to 2 is replaced: the PREREG cites that record and
  reverts to 2 only while a run is UNPINNED.
- **Power** (normal approximation; P(LB > 0) = Φ(δ / SE − 1.96); SE from S′-LIVE's realised interval, 0.974 pp at 2,698
  pairs, scaled by √(2,698 / n); in brackets the conservative √(d / n) at S′-LIVE's d = 0.341):

  | deals | SE | at S′-LIVE's +4.60 pp | at half, +2.30 pp | at +1 pp |
  |---|---|---|---|---|
  | 2,700 | 0.97 [1.12] | 0.997 [0.984] | 0.66 [0.53] | 0.18 [0.14] |
  | **3,840** (64 unit seeds) | 0.82 [0.94] | ≈ 1.00 [0.998] | **0.80** [0.69] | 0.23 [0.18] |
  | 4,500 | 0.75 [0.87] | ≈ 1.00 [1.00] | 0.86 [0.75] | 0.26 [0.21] |

  3,840 deals reach 80 % at half the S′-LIVE effect on the realised SE (≈ 5,060 on the conservative one). False GO 2.5 %.
- **The classes are the PREREG's to fix.** GO-HONEST is LB > 0 with worst-case imputation. A class that says the leak
  carried the gain needs a bar on UB (half of S′-LIVE's +4.60 is one candidate). The two reads' deals differ, so they
  are never differenced as a pair.
- **REPORTED:** the honest builder's exclusion rate per panel against S′-LIVE's 5.8 %, the departure rate, and the
  counter-exposed share (§1 L2).

## 6. The build, and why the unit stopped

| # | build | estimate |
|---|---|---|
| B1 | Rust: a reseed schedule in both replays (`rust_env`'s `Game::replay`; a `core_events` START key). Empty = today's bytes; regression pins that fail on revert | 0.5 d |
| B2 | Rust: `core_events`' fresh-dice segment matcher. Snapshot and restore the session, the version and parse chains and the trackers per try; keyed tries; composes with the lean SCREEN stop; reports the schedule | 1–1.5 d |
| B3 | Rust: the counter backtracking of D4, driven by onset lines, with its tests | 0.5 d |
| B4 | Rust: the damage-roll control on fresh dice (the cost lever) | 0.5–1 d |
| B5 | Python: the I-only entry point (D1), the scaffold (D2), the keying (D7), the opponent's commands with stand-ins (D6; its Rust token) | 1 d |
| B6 | Python: the joint chain (D5), the honest sampler, the preset, the live searcher's wiring and record fields | 1–1.5 d |
| B7 | Tests: INV (§3) failing on `r0s4_rolls` and passing on `r0s4_honest`; the parity of `r0s4_rolls` (the world-screen parity test, the search-side tests, the identity re-score and bank tooth at the new stamp); the Rust gates (`cargo test`, the e2e capstone, the obs golden) | 1 d |
| B8 | The timing run (cost and exclusion rate), the PREREG, two adversarial reviews, `main.reads` registration | 1 d |
| | **total** | **≈ 6–8 agent-days** |

**The alternatives:**

| alternative | why not |
|---|---|
| close the counters only (a root resample of the persisted values) plus D7 | ≈ 1.5 d, but L1, L3 and L4 stay: it does not answer "unable to see ANY hidden information" |
| tier B of `design_world_sampler.md` (type every draw site to print the truth's outcome) | 49 sites to type and map against what each prints, for the same result as D3's generic matcher |
| a decoy truth (run today's S′ on one of its own worlds as if it were the truth) | the decoy is built on the truth's seed, so the leak passes through it, weakened, never removed |
| play the live battle on dice hidden from the searcher | the searcher must then match fresh dice anyway (D3) |
| a particle filter across a game's decisions (keep the 8 worlds and extend them by the new segments only) | a cost lever on top of D3–D5, not a substitute |

## 7. Open questions

- **The exclusion rate without the truth's background (D5, D8).** If it rises well above S′-LIVE's 5.8 %, the
  replication measures a less-searched S′, and its PREREG must say so before any game.
- **D3's per-segment rejection weights a world's dice path by how easy its later segments were to match** (a
  normaliser per segment). It is a bias, not a leak. It is reported, and worlds stay uniformly weighted, as every live
  preset.
- **Whether a cheap OFFLINE ablation comes first.** `r0s4_rolls` against `r0s4_honest` at dlive's banked states under
  the self-referee would cost tens of CPU-h, against ≈ 500–1,000 for the live read.

## 8. Stage 1 as built (2026-11-05)

**What exists.** The preset `r0s4_honest` (`main.anchors.search_presets`, `dice="fresh"`); I and what is read off it
alone (`main.value_search.info_set`: the segments, the key, the opponent's intents, the counters' onsets, the
opponent's rolled hits); the builder (`main.value_search.honest_worlds`: the scaffold, the joint chain, each world on
its own fresh dice, the strict gate); the honest search entry (`main.value_search.search.search_decision_honest`,
which never reads a log); the live searcher's honest path (`main.anchors.search_side`: I read off the log by the ONE
replay its root guards run, the record's `search_key` and `honest` block); in Rust the fresh-dice matcher
(`src/rust_sim/src/fresh_dice.rs`, `core_events`' `INFOSET` / `MATCH`), the damage-roll control's `seg` mode
(`rolls.rs`) and the reseed schedule in both replays (`core_events` START `"reseed"`, `rust_env`'s `Log.reseed`). The
leak test is `main.value_search.honest_inv` and `honest_inv_test.py`. `r0s4` and `r0s4_rolls` are unchanged (§8.3).

### 8.1 The choices the spec left open

| # | where | chosen | why |
|---|---|---|---|
| C1 | D2 team size | read from `\|teamsize\|<side>\|N` | the gen-3 protocol prints it at START to both sides; D2's "six, the protocol names none" was wrong |
| C2 | D3 segment | I's lines split after each `\|request\|` (one WRITE each; segment 0 the START write) | a write may ship a side several chunks; every write that ships the viewer ends with its request |
| C3 | D3 reseed point | before EVERY feed of an update, the same seed, each recorded | a feed that resolves nothing draws nothing, so this is D3's "before the resolving command"; recording each keeps both replays exact without relying on it |
| C4 | D6 executed choice | the opponent's first own action in the segment (a `\|switch\|`, a `\|move\|` with no `[from]`, Focus Punch's tightening, a `cant` line's move), continuing into the next segment while the opponent makes no new choice there | gen 3 replaces a fainted mon MID-turn and a Baton Pass splits a turn: the opponent's turn choice then shows after the viewer's mid-turn switch. A new choice: a `\|turn\|`, the viewer waiting, the opponent's active fainted |
| C5 | D6 stand-in | each choice the world's opponent is offered, in request order (moves, then switches), the first reproducing the segment | a switch KO'd by Pursuit before it happens shows no line of the opponent; "the first legal move" alone cannot reproduce it |
| C6 | D4 persisted dice | ALSO the opponent's exact HP: a failure whose first differing line is an HP line of an opponent mon (or names it `[of]`: recoil, drain) re-rolls that mon's rolled hits since its HP was last KNOWN (shown `100/100`, or fainted), latest first, at the SAME dice with its pick inside the consistent rolls stepped (4 a hit, then the next-earlier hit) | the viewer sees a percent; on the true teams, seven Leftovers percents in a row pinned one earlier roll; re-rolling hits from before a full-HP line only burnt the budget (FINDING S′H-6) |
| C7 | B4 damage control | `rolls::Mode::Seg`: the hit's own line in the viewer's segment (the (c+1)-th HP event of the mon) is the target; the drawn roll if it prints it, else a consistent roll by the drawn roll's rank among the inconsistent ones (near-uniform over the consistent set); a Beat Up strike is predicted like a hit here (pinned in tier A's modes, which are unchanged) | an efficiency device only: every line is still compared. Beat Up's pinned strikes (five or six in a row) excluded the decisions that held one |
| C8 | D3 tries | a chain step 64 tries, 8 re-rolls (×4 per stuck restart, to 1,024, only when the stuck line is not an HP number); a world 128 tries (its chain's if more), 24 re-rolls, then 1,024 tries on its later attempts, 3 never-seen draws, 4 spare chain samples; a chain's budget 600,000 feeds | on the true teams 39 of 40 dlive decisions matched at 128 / 24; a segment can need rare dice jointly (two exact Speed ties and a crit ≈ 1 in 128 a try) |
| C9 | D5 the chain's state | each seen mon's item, ability, spread AND unrevealed moves (a 15 % independence kernel redraws them; an IV change retypes, adds or drops the unrevealed Hidden Power); the never-seen mons one keyed draw during the chain and each world's own after it | a spread fits I only through a move it held and never showed (a faster Suicune KO'd before its negative-priority choice): with the moves outside the chain, worlds that redrew them failed their own replay |
| C10 | D2 move slots | the four slots (revealed and drawn) uniformly permuted per world | the true order is hidden; π's opponent play in the playouts is slot-indexed |
| C11 | D2 happiness | 0 when the set holds Frustration, else the default | a default-happiness Frustration prints the wrong damage |
| C12 | D2 our genders | named from our first request's `details` | a fresh START would redraw an unnamed one; the strict gate checks the root row is unchanged |
| C13 | the strict gate | every kept world replayed by `core_events` with every check on its schedule must reproduce I's lines and the live root row, else RAISE | the matcher compares lines on a bare session; a difference would be a matcher defect, never an exclusion |
| C14 | D1 entry | the live searcher reads I off the true log with the one replay its root guards run (the tokens, the turn, the row byte for byte) and hands the search I alone | the front end does not keep the lines it shipped the core slot; the replay is a guard, not an input |
| C15 | the record | `r0s4_honest` writes every field `r0s4_rolls` writes on a searched decision (Q̂, log π, root V, the frame and log digests; the anchor lane keeps its captures), plus `search_key` and `honest` | X65's ExIt-0H: the honest read's labels come free (coordinator, 2026-11-05) |
| C16 | §5 fresh deals | disjoint from 300–344 (S′-LIVE), 200–244 (dlive) AND 600–674 (X65 ExIt-0's second look, FINDING X65-F3) | the coordinator's instruction, 2026-11-05 |

### 8.2 The leak test (INV) as built

`main.value_search.honest_inv` (the harness and the fixture's builder) and `honest_inv_test.py` (slow): for each
case, the live searcher's own `_search` on D and on D′ (D′ checked to ship the same I by replay), with a fixed
DIGEST network (each row's logits and V read off a keyed sha256 of its bytes: INV does not depend on the network, a
digest is bounded and moves with any difference in a row; a linear stub on the raw rows saturated V, every Q̂ 0, and
hid the L2 case: FINDING S′H-8). The record compared: searched / why / error, the pick, deviation, Q̂, log π, root V,
spent, the worlds' tries, the leak block, the search key, the honest build record and the world teams; never a
CPU or wall time, never `species_hit` (a covariate measured against the truth after the decision).

The fixture (`src/main/value_search/honest_inv_fixtures/cases.json.gz`) holds one pair per channel, each the first in
the order sha256(key) over dlive's banked states whose D′ ships I AND moves `r0s4_rolls`'s record (its key pinned to
I's digest, a call-site choice, except L5):

| channel | the case (dlive key) | D′ |
|---|---|---|
| L1 the true seed | `794b8881…\|t16\|n15` | D's teams and commands on fresh dice (a 30-entry reseed schedule, the matcher's fixed-commands mode); `r0s4_rolls`'s accepted world teams differ |
| L2 persisted dice | `58b6b215…\|t12\|n15` | only the root's last update re-diced (`from_segment`); a counter drawn there persists to the root. `r0s4_rolls`'s world teams EQUAL, its Q̂ not |
| L3 the true scaffold | `fa9e641c…\|t11\|n11` | a seen Tyranitar's unrevealed Earthquake replaced by Thunderbolt (`align_to_truth`, the true background) |
| L4 private commands | `a9987a13…\|t17\|n17` | an opponent's Roar never executed, replaced by Spikes (the template renders the true slot) |
| L5 keying | `fa9e641c…\|t11\|n11` | a never-seen Moltres replaced by a Tentacruel; today's live key (the deal) |
| L6 exclusions | `e97d9fb8…\|t29\|n32` | an opponent's Explosion never executed, replaced by poke-env's untyped `move hiddenpower` (the mon holds one; prior mass < 0.15): today's template keeps it verbatim and `r0s4_rolls` EXCLUDES on D′, searches on D |
| control | `2a2210bb…` | the same battle at turn 10 against turn 17: another I |

**Result (2026-11-05, at the new engine stamp): 14 passed.** `r0s4_honest`'s record is byte-identical on D and D′ for
every channel; `r0s4_rolls`'s differs on every channel (the test's teeth: it fails on the leaky builder, channel by
channel); the control's honest records differ. P3 (a spread moved where I does not pin it) is built
(`p3_spread`) but no channel's case needed it.

### 8.3 `r0s4_rolls` unchanged

The build moved the engine stamp off E4 (`port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`) to
**`port=74eb0a3bc35bd465/95;env=87f06ac3ab0b911b/130`** (the new `fresh_dice.rs`, the `seg` mode of `rolls.rs`, the
reseed schedule in `core_events` and `rust_env`). Every change is opt-in: `r0s4_rolls` arms none of them. **S′-LIVE's
registered identity re-score at the new stamp: PASS, 600 / 600 decisions re-scored (forum72 217, heldout72 204, p72
179), 0 mismatches** on searched / why / played / Q̂ / log π / root V (`rescore.sh`, 4,637 CPU-s). The Rust gates at
the new sources: `cargo test` of the new `fresh_dice_test` and `search_reseed_test` (whose schedule pin fails when the
replay's reseed is reverted), the rolls and playout tests, the obs golden (991 / 991 hashes), the core corpus and the
e2e capstone golden (11 passed).

### 8.4 The timing and exclusion check, and the budget

**What ran** ([`measurements/sprime_honest_stage1_2026-11-05/`](../research_state/measurements/sprime_honest_stage1_2026-11-05/README.md),
outcome-blind): dlive's scored U states, 100 a panel (mean turn 19.2), the live searcher's own `_search` for each
preset, P\* on CPU, one thread, `nice 19` on the agent cores with a training run live. `r0s4_rolls` was timed once;
`r0s4_honest` was re-timed after the legality fix (§8.1 C9), with the Showdown validator child's CPU (which no
record counts: FINDING S′H-9) added, and then at two larger budgets on the 44 states the base budget limited.

**The budget curve** (the coordinator's rule, 2026-11-05: one setting for every panel, the smallest bringing each to
≤ 10 % exclusions at ≤ 20 projected CPU-min a searched game; else stop with the table). A larger budget replays a
smaller one's chain exactly and goes on where it stopped, so only the limited states change (a spot check of six
others at B3: identical records):

| setting (chain steps, chain feeds, world attempts) | F excl. | H excl. | P72 excl. | U excl. | F CPU-min / game | U CPU-min / game |
|---|---|---|---|---|---|---|
| base (300, 600,000, 3) | 19 % | 6 % | 3 % | 9.3 % | 7.1–8.9 | 5.0–7.6 |
| B2 (750, 1,500,000, 5) | 12 % | 6 % | 1 % | 6.3 % | 10.5–13.1 | 6.8–10.3 |
| **B3 (1,800, 4,000,000, 8): CHOSEN** | **5 %** | **4 %** | **0 %** | **3 %** | **13.3–16.7** | **8.6–13.0** |

CPU-min a searched game is projected two ways: S′-LIVE's 38.4 search-path decisions a game × the mean CPU per
decision (the lower figure), and S′-LIVE's 6.2 CPU-min × the honest / `r0s4_rolls` CPU ratio on the same states (the
higher). **B3 is the setting `r0s4_honest` now builds with** (`honest_worlds.MAX_STEPS`, `CHAIN_FEEDS_MAX`, `ATTEMPTS`).
Before the legality fix the base budget excluded F 19 %, H 9 %, P72 6 %.

**`r0s4_honest` at B3 against `r0s4_rolls`, per panel:**

| panel | excl. rolls / honest | CPU-s a decision rolls / honest (median; honest p90) | honest world / playout / validator CPU-s | CPU-min a game rolls / honest | same pick (both searched) | rolls vs rolls, another key |
|---|---|---|---|---|---|---|
| H | 0 % / 4 % | 6.9 / 12.2 (4.1; 7.6) | 9.9 / 2.0 / 0.2 | 4.4 / 7.8–10.9 | 56 % (96) | 62 % (34) |
| F | 6 % / 5 % | 7.7 / 20.8 (5.7; 58.5) | 18.6 / 1.7 / 0.4 | 5.0 / 13.3–16.7 | 71 % (91) | 69 % (35) |
| P72 | 1 % / 0 % | 4.5 / 7.2 (4.4; 11.0) | 4.9 / 2.0 / 0.1 | 2.9 / 4.6–9.9 | 57 % (99) | 92 % (25) |
| **U** | **2.3 % / 3.0 %** | **6.4 / 13.4 (4.4; 21.0)** | 11.1 / 1.9 / 0.2 | **4.1 / 8.6–13.0** | **61 % (286)** | **72 % (94)** |

- **The cost is the long tail of the chain.** The median honest decision (4.4 CPU-s) costs what a `r0s4_rolls` one
  does; the mean is set by the hard decisions (F's p90 58.5 CPU-s), most of them late F decisions. The playouts cost
  the same (2.5 against 2.7 CPU-s on the states both searched in one run).
- **The agreement is read against the noise floor.** Where both search, the two presets pick the same action in 61 %
  of U decisions; on the 94 states the floor ran, `r0s4_rolls` against itself under another key agrees in 72 % and
  `r0s4_honest` against `r0s4_rolls` in 67 %: the honest worlds move the pick about 5 pp more than resampling `r0s4_rolls`'s
  own worlds does. Honest departs from π in 36 % of decisions, `r0s4_rolls` in 32 %.
- **The stop criteria at the chosen setting:** every panel ≤ 15 % (all ≤ 5 %), the cost ≤ 20 CPU-min (F at most
  16.7), no channel needed a model or observation change, `r0s4_rolls` stayed byte-identical (§8.3). None fired.
- **The replication's budget** (§5's 3,840 deals) at ≈ 8.6–13.0 CPU-min a searched game: ≈ 550–830 CPU-h for the
  searched units, against §4's estimate of 450–960.

## 9. The incremental builder `r0s4_honest_inc` (2026-11-06)

**Why.** `r0s4_honest` re-matches the whole of I at every decision and restarts its chain cold, so a battle's world
building grows with the square of its length: the replication designer's late probe (144 S′-LIVE states, every turn
bucket) measured late decisions at up to ≈ 330 CPU-s, exclusions of 50–56 % on F and P72 at turns 64+, ≈ 30 CPU-min a
searched game at K = 6, and ≈ 14 days for the read locally, with a likely VOID from the 10-h watchdog. A fast honest
builder is also what live play with a turn timer needs.

**What exists.** The preset `r0s4_honest_inc@10` (`dice="fresh_inc"`); the builder `main.value_search.honest_inc`
(`HonestIncSampler`, a subclass of `honest_worlds.HonestSampler` sharing its scaffold, chain, matcher and gates); in
`info_set` the earlier decisions of I (`extract_points`, `checkpoints`, `prefix`); in Rust the matcher's CARRIED
TRAJECTORY (`fresh_dice.rs` `prefix`, the report's `trail`; contract: `designs/rust_sim/search_and_replay_drivers.md`
"Fresh dice"). `r0s4_honest`'s own path is unchanged: the new hooks in `honest_worlds` (a warm start for `_chain`, its
dice key and starting tries, the per-world build lifted into `_one_world`, the samples and reports kept OUTSIDE the
record) default to the cold builder's behaviour, and every Rust change is opt-in (§9.4).

### 9.1 A step, and why it is still a function of I

I at a decision extends I at every earlier one. Its CHECKPOINTS are the earlier decisions the live searcher searches
(turn ≥ 10, ≥ 2 legal actions), each read off I (`info_set.checkpoints`); the CARRY at a checkpoint is a function of the
carry at the one before and of its own I, the first built cold. So the carry at a decision, and its record, are
functions of its I alone. At decision k, from the carry of the previous checkpoint:

1. **Extend.** Every carried world is re-rendered under this I's scaffold and matched with its carried trajectory: the
   segments of the I it last matched REPLAYED exactly (one attempt a segment: its seed, pick offset and commands) up
   to the BOUNDARY — the first segment whose header this I changes (an opponent's intent first seen in a later
   segment, an onset, a hidden-HP hit); the rest on this decision's fresh dice (tag `x<world>:<step>:<attempt>`, 128
   tries, 24 re-rolls). A re-roll whose target is inside the replayed part UNFREEZES it from there. What a reveal
   changed is redrawn first, keyed (4 attempts): a newly seen mon's coordinate (and the never-seen mons, one of which
   it was); a seen mon's unrevealed moves when they lack a newly revealed one; a coordinate this I rules out.
2. **Fresh.** At least `FRESH_MIN` = 2 worlds (more to fill J = 8) from the WARM chain: `honest_worlds`' joint chain
   started at the carried chain state, sampling at once when it fits, on ONE set of chain dice for the battle (keyed by
   the I of the checkpoint the chain's lineage began at, so a state that fit before replays its kept tries), each new
   world on its own fresh dice (`w<world id>:<attempt>`, 2 attempts) as the cold builder's.
3. **Shortfall.** Too few worlds: the carried ones that failed step 1 once more on rarer dice (1,024 tries).
4. **Compose and retire.** The fresh worlds, then the youngest survivors, to 8; the rest retire.
5. **Gate.** Every returned world passes the strict gate (one lean `core_events` replay over the decision's worlds)
   and the playout tool. Fewer than 8: an EXCLUSION, whose carry keeps EVERY world: the extended ones at this I, the
   others PENDING at the I they last matched. The battle is then STUCK (C28) until a decision is searched.

### 9.2 The choices

| # | where | chosen | why |
|---|---|---|---|
| C17 | the correlation bound | each decision draws max(2, 8 − the worlds that extended) fresh whenever the warm chain yields them, and the oldest retire: a world serves at most ⌈8 / 2⌉ = 4 decisions (its age ≤ 3) while the battle draws fresh worlds every decision; after a stretch that drew none (a stuck battle, a deferred chain) or with fewer than 6 survivors, older worlds stay: an OVERAGE the record counts (`inc.overage`, `inc.ages`). Measured (§9.5): 8.3 % of searched decisions hold one, mostly by one decision (age 4) | a carried world ties consecutive decisions' worlds together; a rolling bound keeps the cost of fresh worlds level (2 a decision) where periodic rejuvenation would put a cold build, at its O(turns) cost and exclusion risk, on every P-th decision. FRESH FIRST, then retire: on a hard battle a fresh world must re-solve the whole history on its own dice and often cannot, and retiring the carried worlds regardless emptied the carry (§9.5's first run: a battle collapsed into a cold rebuild, excluded, at every later decision) |
| C18 | the checkpoints | the decisions the live searcher searches (turn ≥ 10, ≥ 2 legal actions), read off the replay that yields I (`extract_points`: each decision's line count, legal count, turn and row, all functions of the viewer's lines) | the carry must advance where the live searcher builds, and be rebuildable from I alone |
| C19 | the carry store | per process, keyed by the I digest of the checkpoint it was built at (LRU, 64); a miss REBUILDS the missing checkpoints in order from the latest cached one (from the first when none is) | the live process plays a battle's decisions in order, so it hits; a unit retry is a new process replaying the battle from its start (it rebuilds as it goes); an offline re-score of one decision rebuilds the battle up to it. Every path computes the same bytes (the slow test `honest_inc_test.py` checks the rebuild) |
| C20 | the boundary | the first earlier segment whose intent, onsets or hidden-HP hits differ under this I | an intent read past a mid-turn request (C4) can name the choice a stand-in guessed; the replay must re-match from there |
| C21 | a reveal | the carried world's changed parts redrawn, keyed (4 attempts), the rest kept and the prefix re-verified by the replay itself | a carried world whose never-seen draw is not the revealed mon could never extend, and every reveal would empty the carry |
| C22 | the chain's dice | one set per battle (reset to the next decision's key when the warm chain fails) | a state that fit on the chain's dice before fits the replayed segments again, so the warm chain samples at once; per-decision dice would re-solve the whole prefix's feasibility every decision (the cost this unit removes). The common-dice bias of S′H-11 now spans a battle (S′H-12) |
| C23 | the warm chain's budget | stage 1's base budget (300 steps, 600,000 feeds) while a carried world survived; B3 when none did | with a survivor the decision is built from the carried worlds; a failed chain then costs its base budget (≈ 30 CPU-s), not B3's (≈ 200) |
| C24 | an exclusion | its carry keeps EVERY world (extended ones at this I, the rest PENDING at the I they last matched: each world carries its own I) and the last chain state that fit (on new chain dice); the next decision retries the pending worlds over every segment they lack, on its own dice. Only a carry with no world at all (the battle's first decision excluded) is rebuilt COLD: `r0s4_honest`'s own build | a segment no world matched (a Beat Up with a crit and a wake-up, §9.5) sits in every later decision's history: dropping the carried worlds left only the cold rebuild, which must re-solve the whole history and failed at every later decision; a pending world needs only the segments it lacks, on new dice each decision |
| C25 | the first decision | `r0s4_honest`'s own build, byte for byte (its record plus the `inc` block) | the slow test pins it |
| C26 | the record | `honest.inc`: the step, the mode, the boundary, the worlds pending / extended / failed / escalated / new / retired, their ages and the overage, and the digests of the carry it started from and returned (`carry_in`, `carry_out`: a battle's records chain). A re-score-sampled record also keeps its starting carry (`inc_carry_in`, gzip + base64, ≈ tens of KB) | weighed for the identity re-score: without it a re-scored decision rebuilds its battle up to it (≈ the battle's building so far, §9.5); with it the re-score seeds a fresh searcher and re-runs the one decision (the slow test checks it needs no rebuild and reproduces the record), each link checked by the digests. Whether the replication's re-score uses it is its PREREG's to say (S′H-14) |
| C27 | a shortfall | the worlds that failed their extension retried once on 1,024 tries (the new segment needs rarer dice); a fresh world gets 2 attempts on its own dice, not B3's 8; no clones | escalation costs a few feeds a world and only on a shortfall; a fresh world that cannot re-solve the history in 2 attempts is better replaced by a carried one than paid for; a clone (a survivor's seen mons on its own trajectory) would fill J with copies of one hypothesis |
| C28 | a STUCK battle | after an exclusion, each next decision first PROBES (the carried worlds' extensions, one redraw of what a reveal changed) and goes on as usual only when a world extended, else it is excluded at once; the full attempt (every redraw, the warm chain with no survivor, the escalation) comes back at the 4th stuck decision and every doubling after it (8, 16, 32, …). A cold rebuild while no world was ever built retries in full at the next 3 decisions, then on the same doublings; a searched decision ends it | a segment no world can match (S′H-10: §9.5's Beat Up with a crit and a wake-up defeated the carried, revised, escalated and chain worlds, and the cold builder) sits in every later decision's history; the full attempt there cost ≈ 130–160 CPU-s a decision, every decision, for nothing. A first cut that skipped the chain whenever stuck needed all 8 pending worlds to extend, and kept recoverable stretches excluded; a full attempt every 4th stuck decision still cost ≈ 1,600 CPU-s over one battle's 51-decision stuck stretch |
| C29 | a failing warm chain | after it fails, the chain WAITS on the same backoff (it runs again 4, 8, 16, … decisions later) while the carried worlds alone fill the decision; it runs whenever fresh worlds are needed to avoid an exclusion. Recorded as `inc.chain = "deferred"` and `inc.chain_fail` | the chain judges candidates by re-matching the whole history on its common dice; in a hard battle it failed at every decision (≈ 30–45 CPU-s each) while all 8 carried worlds extended on their own solved trajectories. Rejuvenation can wait (the soft bound counts the overage); feasibility cannot |

### 9.3 The leak test in SEQUENCE mode, and determinism

`honest_inv.check_sequence`: for each fixture case, the WHOLE decision sequence of D and of D′ from turn 10 up to the
case's decision (`honest_inv.sequence`: every decision the live searcher searches, cut from the case's own log, each
cut's root checked against the playout tool's), each truth on its OWN searcher (its own carry cache, so D′ never
borrows a carry D built), records compared on `INV_FIELDS` at every decision. The single-decision check now also runs
each truth on its own searcher.

**Result (2026-11-06, at the build's stamp; `honest_inc_test.py`, slow):**

| case | decisions in its sequence (turn 10 to the case's) | `r0s4_honest_inc`, D vs D′ | `r0s4_rolls`, D vs D′ |
|---|---|---|---|
| L1 the true seed (t16) | 7 | identical at every decision | moves |
| L2 persisted dice (t12) | 3 | identical | moves |
| L3 the true scaffold (t11) | 2 | identical | moves |
| L4 private commands (t17) | 9 | identical | moves |
| L5 keying (t11) | 2 | identical | moves |
| L6 exclusions (t29) | 24 | identical | moves |

**Determinism** (on L6's 24-decision sequence): a re-run from scratch on a fresh searcher reproduces every record; the
case's decision run ALONE (a fresh cache, so its carry is rebuilt through all 23 earlier checkpoints) reproduces its
record; each record's `carry_in` is the previous record's `carry_out`; a decision re-run on a fresh searcher seeded with
the carry it started from needs no rebuild and reproduces its record; the first decision is `r0s4_honest`'s record plus
the `inc` block; no searched decision uses a world older than 3 decisions and every warm step draws ≥ 2 fresh.

### 9.4 `r0s4_honest` and `r0s4_rolls` unchanged

The build moved the engine stamp to **`port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`** (`fresh_dice.rs`: the carried
trajectory, and the match loop reading its frame instead of cloning it, which changes no verdict). Every change is
opt-in: `r0s4_rolls` arms none, and `r0s4_honest`'s Python path runs the lifted `_one_world` with the same draws and
tags.

- **The stage-1 tooth** (the replication designer's: 24 of stage 1's B3 states, `r0s4_honest@10` at its default
  budget, every non-timing field equal to stage 1's): **PASS, 24 / 24** (F 15, H 6, P72 3; 20 searched, 4 excluded, 0
  mismatched).
- **S′-LIVE's identity re-score of `r0s4_rolls`** (its registered 600 decisions): **PASS, 600 / 600, 0 mismatches** (forum72 217, heldout72 204, p72 179; 4,644 CPU-s)
- **The leak test** (`honest_inv_test.py`, each truth now on its own searcher): 14 passed. The slow tests of the touched
  modules (`honest_worlds`, `search`, the world-screen parity, the anchors' live-search tests): 19 passed. `cargo test`
  (both crates, debug, no fail-fast): every binary passes but `rust_env`'s `sim_bridge_parity_test`, which needs
  `POKESIM_SIM_BRIDGE_BIN` (its Python wrapper sets it); the obs golden 991 / 991; the e2e capstone and core corpus 11
  passed.

### 9.5 Stage 1b: cost, exclusions and agreement

**What ran** ([`measurements/sprime_honest_inc_stage1b_2026-11-06/`](../research_state/measurements/sprime_honest_inc_stage1b_2026-11-06/README.md),
outcome-blind): the replication designer's 144 late-probe states (S′-LIVE's searched games, every turn bucket) and
stage 1's 300 dlive states, each on its battle's WHOLE decision sequence from turn 10 (407 battles, 9,758 decisions),
one live searcher a battle; the live searcher's own `_search` (P\*, the playouts) at each of the 444 targets, the
builder step alone at every other decision; `r0s4_honest`'s records at the targets are stage 1's (B3) and the late
probe's, unchanged at this stamp (§9.4). The noise floors: `r0s4_honest` re-run at 190 targets (stage 1's floor
subset and the late states to turn 63) and the incremental builder's whole sequences at 162 of them, each under
another key tag. The SMT probe: one builder workload pinned alone, on two physical cores and on two SMT siblings.

**Per panel and turn bucket** (world CPU-s over EVERY decision of the sequences, exclusions included; "other" = the forward and playouts of the searched targets; exclusions over every decision; at the targets, both builders' exclusions and full-decision CPU-s):

| panel / bucket | decisions | world CPU-s mean (median, p90, max) | other | excl. (all decisions) | targets | excl. at targets inc / honest | CPU-s at targets inc / honest | same pick inc vs honest |
|---|---|---|---|---|---|---|---|---|
| F/10-19 | 1242 | 4.31 (0.56, 3.83, 314.18) | 2.32 | 1.8 % | 62 | 1 / 3 | 7.82 / 19.79 | 36 / 59 (61 %) |
| F/20-29 | 650 | 3.66 (0.57, 2.88, 297.04) | 2.21 | 2.9 % | 45 | 2 / 3 | 8.89 / 24.52 | 27 / 42 (64 %) |
| F/30-63 | 752 | 3.37 (0.94, 3.34, 232.61) | 2.88 | 5.3 % | 23 | 1 / 2 | 4.57 / 27.8 | 15 / 21 (71 %) |
| F/64+ | 692 | 7.45 (2.21, 4.85, 391.94) | 2.9 | 15.0 % | 18 | 3 / 10 | 4.46 / 116.83 | 5 / 8 (62 %) |
| H/10-19 | 1240 | 1.19 (0.45, 1.94, 80.25) | 2.58 | 0.2 % | 56 | 0 / 0 | 3.32 / 4.15 | 32 / 56 (57 %) |
| H/20-29 | 629 | 1.25 (0.54, 1.93, 51.31) | 2.6 | 0.2 % | 52 | 0 / 3 | 4.69 / 16.42 | 30 / 49 (61 %) |
| H/30-63 | 764 | 1.7 (0.93, 2.77, 47.47) | 2.14 | 0.4 % | 22 | 0 / 1 | 3.23 / 30.2 | 14 / 21 (67 %) |
| H/64+ | 696 | 5.05 (2.01, 7.88, 270.63) | 2.0 | 0.7 % | 18 | 0 / 2 | 5.97 / 77.75 | 10 / 16 (62 %) |
| P72/10-19 | 1216 | 1.25 (0.46, 1.94, 307.75) | 2.4 | 0.3 % | 58 | 0 / 0 | 3.22 / 5.82 | 35 / 58 (60 %) |
| P72/20-29 | 635 | 1.41 (0.53, 2.02, 115.16) | 2.32 | 0.8 % | 48 | 0 / 0 | 3.39 / 8.02 | 29 / 48 (60 %) |
| P72/30-63 | 637 | 2.49 (0.84, 4.26, 56.17) | 2.82 | 5.2 % | 24 | 0 / 0 | 3.8 / 10.5 | 18 / 24 (75 %) |
| P72/64+ | 605 | 7.85 (2.37, 9.95, 354.91) | 2.07 | 20.5 % | 18 | 4 / 9 | 4.2 / 113.71 | 7 / 9 (78 %) |

**Projected per searched game** (S′-LIVE's search-path decisions a game by turn bucket × the bucket's mean decision CPU, plus S′-LIVE's per-game overhead; K = 6 with its κ by bucket: the replication designer's cost model, `sprime_honest_play_2026-11-06/inputs/cost_model.json`):

| panel | CPU-min K = 2 inc / honest | CPU-min K = 6 inc / honest | projected exclusions inc / honest |
|---|---|---|---|
| F | 5.5 / 29.6 | 8.7 / 48.1 | 5.8 % / 14.8 % |
| H | 2.9 / 14.6 | 4.6 / 23.4 | 0.3 % / 3.4 % |
| P72 | 3.3 / 11.8 | 5.3 / 19.2 | 4.0 % / 4.9 % |
| **U** | 3.9 / 18.7 | 6.2 / 30.3 | 3.4 % / 7.7 % |

**The read** (3,840 searched games and their raw twins):

| builder | CPU-h (K = 6) | wall days, agent cores K = 6 | wall days, whole box K = 14 |
|---|---|---|---|
| inc | 434 | 3.1 | 1.3 |
| honest | 1976 | 14.2 | 6.1 |

**Agreement at the targets** (both searched; the floors are each builder against its own replicate under another key tag, on the replicate subset):

| panel | inc vs honest | honest vs honest′ | inc vs honest on that subset | inc vs inc′ | departs from π: inc / honest |
|---|---|---|---|---|---|
| H | 86 / 142 (61 %) | 33 / 63 (52 %) | 39 / 64 (61 %) | 34 / 56 (61 %) | 41.9 % / 37.3 % |
| F | 83 / 130 (64 %) | 38 / 61 (62 %) | 42 / 64 (66 %) | 31 / 52 (60 %) | 37.6 % / 34.6 % |
| P72 | 89 / 139 (64 %) | 43 / 57 (75 %) | 38 / 57 (67 %) | 32 / 49 (65 %) | 36.1 % / 38.9 % |
| U | 258 / 411 (63 %) | 114 / 181 (63 %) | 119 / 185 (64 %) | 97 / 157 (62 %) | 38.6 % / 37.0 % |

- **The checkpoint's criteria hold.** The projected cost is ≈ 4.9× below `r0s4_honest`'s (U 6.2 against 30.3
  CPU-min a searched game at K = 6); U exclusions project to 3.4 % (against `r0s4_honest`'s 7.7 %); the leak test
  passes by construction and in sequence mode (§9.3).
- **Where the cost is.** A routine decision builds in ≈ 0.5–2 CPU-s (median); the means carry the tails: each
  battle's first decision is `r0s4_honest`'s own cold build (up to ≈ 300 CPU-s), a refit after a reveal, and a stuck
  battle's full attempts. Turns 64+ are 15–27 % of a game's CPU (H 0.7, F 2.4, P72 1.0 of 4.6, 8.7, 5.3 CPU-min).
  At the late targets a decision costs 4.2–6.0 CPU-s against `r0s4_honest`'s 78–117.
- **The watchdog.** S′-LIVE's longest searched games (≈ 245 decisions, 189 of them past turn 64) project to
  ≈ 0.7–1.1 h at K = 6 at the bucket means, against `r0s4_honest`'s 7.4–10.9 h. The longest measured sequence (162
  decisions) built its worlds in 2,098 CPU-s.
- **Exclusions are the late F and P72 battles that hit a segment no world matches** (S′H-10): a battle stuck there
  excludes every later decision (F 64+ 15 %, P72 64+ 20.5 % of decisions; `r0s4_honest` excluded 10 of 18 and 9 of 18
  of those buckets' targets, the incremental builder 3 and 4).
- **The correlation bound held in 91.7 % of searched decisions.** 8.3 % used a world older than 3 decisions (548
  at age 4, 98 at 5, a tail to 36 in one long stretch with no fresh world); the warm chain fit in 7,787 decisions,
  refit in 1,122, failed in 18 and waited in 62 (S′H-15).
- **The picks.** Where both search, the incremental builder picks `r0s4_honest`'s action in 63 % of U decisions;
  `r0s4_honest` re-run under another key picks its own in 63 % (the incremental builder against itself: 62 %). The
  incremental worlds move the pick no more than re-drawing `r0s4_honest`'s own worlds does. It departs from π in
  38.6 % of decisions, `r0s4_honest` in 37.0 %.
- **The read's wall time** (§5's 3,840 deals): ≈ 434 CPU-h, ≈ 3.1 days on the agent cores at K = 6 (beside a pinned
  trainer ×1.153: ≈ 3.6), ≈ 1.3 days on the whole box at K = 14 (each decision at its K = 6 CPU, κ ≈ 1.55–1.68: both
  SMT siblings busy). **The SMT probe** (the builder alone, one fixed workload pinned, twice each): two workers on one
  core's two siblings each run 1.23× slower than one alone (1.18× slower than on two physical cores), so a core with
  both siblings busy gives ≈ 1.63× one worker's throughput, against κ's ≈ 1.2–1.3× for S′-LIVE's whole units (forward,
  peer and all): at the probe's figure the box at K = 14 would take ≈ 1.0 day. UNVERIFIED for whole units: the probe
  measured the builder only.

## FINDINGs

- **S′H-1 (no bigger leak shown, none ruled out).** §1: the unlisted channels are small or absent in U, the listed
  ones are the large exposures. Only the replication sizes what the leaks carried.
- **S′H-2 (the cost is an estimate from measured inputs).** §4.
- **S′H-3 (`dice="rolls"` cured the rejection, not the value).** L1d: a world takes the TRUE damage roll whenever
  it prints the same thing. `search_ledger.md` §6.1 and `design_information_levers.md` §7.5 said the damage-roll leg
  was cured.
- **S′H-4 (a true future die).** L2: the Quick Claw roll of the root turn survives the playouts' reseed. There is no
  exposure in U (no Quick Claw on any U team), but it is a channel.
- **S′H-5 (the template reads the opponent's private choices).** L4, in 35 % of S′-LIVE's decisions. S′-LIVE's
  `attribution_miss` counter is poke-env's untyped `move hiddenpower` (and `struggle`), not an attribution failure.

- **S′H-6 (the opponent's exact HP is a persisted hidden value).** The viewer sees a percent, so a damage roll's pick
  inside it is a die a later Leftovers, residual or recoil line can contradict (seven Leftovers percents in a row
  pinned one roll on a true team). D4 named counters only; the matcher re-rolls HP too (§8.1 C6).
- **S′H-7 (D2 misread the protocol).** The gen-3 protocol prints `|teamsize|` for both sides at START; the scaffold
  reads it (§8.1 C1).
- **S′H-8 (a saturated stub network hides a leak).** The leak test's first stub (a linear map of the raw rows) gave V
  0 or 1 everywhere, so every Q̂ was 0.0 and `r0s4_rolls`'s L2 records looked identical. The digest network replaced
  it; L2's pair then differs in Q̂ with equal world teams (§8.2).
- **S′H-9 (the validator child's CPU is uncounted).** Every world builder's legality check runs in a persistent Node
  child whose CPU no decision record counts (`cpu_s`: the thread and the reaped children). Measured here for
  `r0s4_honest`: 0.1–0.4 CPU-s a decision; not measured for `r0s4_rolls` (its records, and S′-LIVE's 6.2 CPU-min, omit
  it too).
- **S′H-10 (fresh dice cannot reach every decision).** A segment whose dice are rare jointly (a 10 % paralysis, a
  1/16 crit and a 70 % miss of another paralysis in one turn) or a long hidden-HP chain can defeat the chain within
  any affordable budget; at B3 these are U's 3 % exclusions, priced at F's p90 58.5 CPU-s.
- **S′H-11 (the per-segment rejection weights a dice path by how easy its segments were, §7) and the chain's
  common dice** (every candidate judged on the same keyed tries, so a fit can lean on a lucky path): a bias, not a
  leak, both functions of I; worlds stay uniformly weighted. Unmeasured.

- **S′H-12 (the incremental builder's dice are shared across a battle).** Its chain judges every candidate on ONE set of
  dice for the battle (§9.2 C22), so S′H-11's common-dice bias now spans a battle's decisions; and its carried worlds
  keep their dice paths (a world's hidden counters and the opponent's exact HP) for up to 4 decisions, longer when no
  fresh world can replace them (the record's `overage`). Both are functions of I; neither is measured beyond the pick
  agreement (§9.5: 63 % against `r0s4_honest`, its own floor 63 %).
- **S′H-13 (a segment no world matches stops a battle).** On battle `29ba92d0…` (F, turn 31) a Beat Up of three
  strikes, a crit on the third, then our Snorlax waking defeated every carried world (revised, escalated), the warm
  chain and the cold builder; every later decision of that battle is excluded. Whether the matcher can model it (a
  Beat Up crit under the `seg` prediction, a sleep counter's onset outside `backtrack`) is not investigated; such
  battles are most of the late F / P72 exclusions (§9.5).
- **S′H-14 (the replication's identity re-score of the incremental builder).** A re-scored decision rebuilds its battle
  up to it (pure, but ≈ the battle's world building so far); a re-score-sampled record keeps the carry it started from
  (`inc_carry_in`) so a re-score can seed it instead. `main.value_search.live_identity`'s re-score (and the reads'
  spec checks, which name the presets they accept) do not handle `r0s4_honest_inc` yet: the PREREG unit's to wire.

- **S′H-15 (the bound is soft by more than a failed chain).** C17's rule draws max(2, 8 − survivors) fresh worlds, so a
  battle whose survivors are all old (after a stuck stretch or a deferred chain) keeps them even when the chain could
  give more: 8.3 % of stage 1b's searched decisions held a world past the bound. Counting the fresh worlds needed
  against the survivors still within the bound (8 − the young ones) would retire them as soon as the chain yields
  replacements, at ≈ 1 fresh world's cost each (≈ 0.5–2 CPU-s late); not built, not measured.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-11-05 | **This doc CREATED** (S′-HONEST designer, orchestrator brief): the leak inventory, the design, the leak test, the measured cost inputs; the unit STOPPED before the build at its 2-day bound | the search as a function of I alone: fresh dice matched per segment (D3), persisted dice drawn on them (D4), a joint chain with no true background (D5), the opponent's commands from I (D6), keying by I (D7), preset `r0s4_honest@10`; INV pointwise; 3,840 fresh deals for 80 % at half the S′-LIVE effect | counters only; tier-B typed dice; a decoy truth; hidden live dice; a particle filter instead of D3–D5 (§6) | the code at `fcfe2575e` (§1); the 49-site draw census; `measurements/sprime_honest_2026-11-05/` (exposure, decision records, segment match, timing run, speed pins) |
| 2026-11-06 | **STAGE 1 BUILT** (S′-HONEST builder, orchestrator brief + the coordinator's budget pre-decision): the builder, its leak test, the timing and exclusion check; no PREREG | §8: the fresh-dice matcher in Rust (`fresh_dice.rs`, `core_events` `INFOSET` / `MATCH`, the `seg` damage-roll mode, the reseed schedule in both replays); I and the builder in Python (`info_set`, `honest_worlds`, preset `r0s4_honest`, `search_decision_honest`); the choices C1–C16; budget B3 (the smallest of three measured meeting ≤ 10 % / ≤ 20 CPU-min); INV 14 passed (honest identical on every channel, `r0s4_rolls` moves on every one); `r0s4_rolls` byte-identical (600 / 600) at stamp `port=74eb0a3bc35bd465/95;env=87f06ac3ab0b911b/130`; K = 6 under a pinned training run (X65-F1); U seeds 600–674 kept free (X65-F3) | the base budget (F 19 % exclusions) and B2 (F 12 %); per-panel budgets (the rule: one setting); a linear stub network for INV (S′H-8); the moves outside the chain (§8.1 C9) | `measurements/sprime_honest_stage1_2026-11-05/` (`final_summary.json`, `budget_curve.json`, `agreement.json`, `rescore.json`, the rows); `src/main/value_search/honest_inv_test.py` |
| 2026-11-06 | **The INCREMENTAL builder `r0s4_honest_inc` BUILT, with its stage 1b** (S′-HONEST incremental builder, orchestrator brief) | §9: the worlds carried through a battle's searched decisions (the Rust matcher's carried trajectory; each world extended over the new segments, a reveal redrawn, keyed), ≥ 2 of 8 fresh a decision from a warm chain on the battle's chain dice, an exclusion keeping every world pending, a stuck battle's probe and backoff; the carry a function of I (rebuilt from I's checkpoints on a cache miss), its digests in the record; choices C17–C29. Sequence-mode INV identical on every channel at every decision (fails on `r0s4_rolls` on every one); a sequence re-run and a decision alone reproduce every record; `r0s4_honest` (tooth 24 / 24) and `r0s4_rolls` (600 / 600) unchanged at stamp `port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`. Stage 1b on 444 states' whole sequences: U 6.2 CPU-min a searched game at K = 6 (`r0s4_honest` 30.3), U exclusions 3.4 % (7.7 %), picks 63 % `r0s4_honest`'s (its own floor 63 %); the read ≈ 434 CPU-h, ≈ 3.1 days on the agent cores, ≈ 1.0–1.3 on the whole box | periodic rejuvenation (a cold build every P decisions); retiring before a fresh world replaced the carried one, dropping the carry on an exclusion, a full attempt every 4th stuck decision, a warm chain every decision (each measured and replaced: §9.2); clones of a survivor to fill J | `measurements/sprime_honest_inc_stage1b_2026-11-06/`; `src/main/value_search/honest_inc_test.py` |
