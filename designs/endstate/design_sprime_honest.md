# S′-HONEST — does S′ still beat P\* against the real Kakuna when its worlds see nothing the live player cannot?

**Status: DESIGNED 2026-11-05; BUILD NOT STARTED.** The leak inventory (§1), the design (§2), the leak test (§3) and the
measured cost inputs (§4) are this doc's. The build is ≈ 6–8 agent-days (§6), over its unit's 2-day bound, so the
unit STOPPED there for the orchestrator's decision. No PREREG is written and nothing is registered. S′-LIVE's GO-LEAKY
([`design_sprime_live_play.md`](design_sprime_live_play.md)) licenses this replication and nothing else.

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
| D2 | **The opponent scaffold from I.** Six mons (the format's prior: a gen-3 protocol names no team size); each seen mon's species, nickname, level, gender and shininess from its switch details; its revealed moves, item and ability from the ONE log source on our lines (`agents.info_levers.revealed`); the rest drawn as today. Party order: the lead, then by first appearance, never-seen last. Every gender NAMED, ours included, so turn 0 draws none. Default happiness. Unrevealed moves in a drawn slot order | L3b, L3c, L3d |
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
- **Fresh deals**, disjoint from S′-LIVE's (U unit seeds 300–344) and dlive's (200–244).
- **`peer_long_tail_fault` from the start.** It is Kakuna's turn-250 race (S′LIVE-OP-6 / OP-7): a post-treatment drop,
  imputed at its worst case, counted against the 0.5 % cap. It does not hold the lane alone. A new error class, a
  panel's drops reaching 3, or more than 2 new exhaustions in 6 h does.
- **The operator procedure.** Supervisor lives as transient `systemd-run --user` services whose script checks the
  engine stamp before `exec`ing `main.reads resume`. Concurrency 6 with `GEN3AI_HEAVY_SLOTS=6` in that supervisor only,
  `nice -n 19 taskset -c 5-7,13-15`. The revert to 2 when a training run goes live (AM-K6's terms, written in).
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

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-11-05 | **This doc CREATED** (S′-HONEST designer, orchestrator brief): the leak inventory, the design, the leak test, the measured cost inputs; the unit STOPPED before the build at its 2-day bound | the search as a function of I alone: fresh dice matched per segment (D3), persisted dice drawn on them (D4), a joint chain with no true background (D5), the opponent's commands from I (D6), keying by I (D7), preset `r0s4_honest@10`; INV pointwise; 3,840 fresh deals for 80 % at half the S′-LIVE effect | counters only; tier-B typed dice; a decoy truth; hidden live dice; a particle filter instead of D3–D5 (§6) | the code at `fcfe2575e` (§1); the 49-site draw census; `measurements/sprime_honest_2026-11-05/` (exposure, decision records, segment match, timing run, speed pins) |
