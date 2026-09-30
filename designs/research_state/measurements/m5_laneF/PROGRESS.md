# M5 Lane F — the scripted bots in the Rust env core: PROGRESS (resume point)

Lane F of `designs/endstate/program_rust_core.md` §2 M5 (owner 2026-09-27: "happy to rewrite the
bots as needed" — PORT them). It owns:

- `src/rust_env/src/bots/` — `mod.rs` (`Kind`, `Bot`, `Decision`), `view.rs` (the bot's one input),
  `calc.rs` (the shared helpers), `logic.rs` (one `choose_move` per class), `rng.rs` (CPython's
  MT19937), `gate.rs` (the corpus replay), `tables.rs` (GENERATED);
- `src/rust_env/tests/bots_gate_test.rs`, `src/rust_env/tests/fixtures/bots/commit_corpus.json.gz`;
- `src/utils/rust_env/{bot_inventory, bot_view, bot_corpus, bot_tables}.py` and their tests
  `bot_inventory_test.py`, `bot_tables_test.py`, `bots_gate_test.py`.

Hand-offs (each marked `M5 Lane F` in place): `src/rust_sim/src/present/mon.rs` (`PMove.last_used` —
poke-env's `Move._is_last_used`, written by `moved` and cleared by `switch_out`, poke-env's two
writers; `MoveSet::mark_last_used`); `src/rust_env/src/lib.rs` (`pub mod bots;`);
`src/rust_env/src/search/game.rs` (`Game::reading(side)`); `src/utils/rust_env/core_cargo_test.py`
(one name in the non-vacuity list).

## How to build / test (worktree-local target only)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.rust_env.bot_tables --write                      # after a bot constant / poke-env move data change
python -m utils.rust_env.bot_corpus --commit-tier --write        # re-bank the COMMIT tier (a Python bot changed)
python3 -m pytest src/utils/rust_env/bot_inventory_test.py src/utils/rust_env/bot_tables_test.py \
    src/utils/rust_env/bots_gate_test.py -q                      # routine, ~30 s warm
python3 -m pytest src/utils/rust_env/bots_gate_test.py -q -m slow # MILESTONE, ~75 s
python -m utils.rust_env.bot_corpus --bots staller --n 4 --views --out /tmp/c.json   # a debug corpus WITH the view strings
```

## Units

| # | unit | status |
|---|---|---|
| 1 | the bot INVENTORY (`bot_inventory.py`) + its routine test — rosters derived from the code | LANDED (this commit) |
| 2 | the BANKED DECISION CORPUS (`bot_corpus.py`): recorder, the view (`bot_view.py`), the COMMIT bank | LANDED (this commit) |
| 3 | all ten pooled bots in Rust, the view, the RNG, the gate; COMMIT routine + MILESTONE slow + teeth | LANDED (this commit) |

The three units landed as one commit: the bots share one view, one RNG and one gate, and the
inventory's "ported" column is only true once the Rust names exist.

## The inventory (headline)

**Ten bot classes play in some pool; one more is defined and plays nowhere.** Table of record:
`src/utils/rust_env/bot_inventory.py`.

| bot | class | pools | RNG streams |
|---|---|---|---|
| random | `RandomPlayer` | eval, final eval | choice (its whole policy) |
| heuristic | `SimpleHeuristicsPlayer` | train, eval, final eval, warm-start smoke | choice |
| heuristic2 / staller / staller_v2 / aggressive / aggressive_v2 / setup_sweep / setup_sweep_v2 | `agents.opponents.Gen3*Player` | train, eval, final eval | choice (+ protect for both stallers) |
| baitbot | `Gen3BaitBotPlayer` via `make_baitbot_class(--bait-bot-p)` | train, only with `--bait-bot-share > 0` | choice + bait |
| max_base_power | `MaxBasePowerPlayer` | NONE | choice |

- The exploiter keep-bots mix IS the training roster: `env_factory` builds `heuristic_opponents` from
  `OPPONENT_CLASSES`.
- Anchors (`bot:<name>`) and the prober's replay reach the eval roster through `eval_opponent_class`
  / `_EVAL_OPPONENT_SPECS`.
- "Eval sentinels" are the self-play pool's POLICY snapshots, not bots. The bots the brief listed as
  sentinels (staller, staller_v2, setup_sweep, …) are the eval ROSTER.

`bot_inventory_test.py` (routine) derives every roster from the code by AST and fails on:

- a pooled class with no row;
- a row whose `used_by` differs from the rosters;
- a `Player` subclass in a bot module with no row;
- the keep-bots mix no longer being the roster;
- a "ported" row whose `Kind` is missing.

The derivation catches a bot added by ANY construct: the final eval appends four bots in a loop,
which a list-literal read missed on the first try. Teeth: a dropped row and an appended bot each fail.

## Decisions

- **What state a bot reads:** the env core's OWN per-side `BoardReading` — the port of the poke-env
  `Battle` a Python bot reads (`env.battle2`) — through `bots::view::View`. There is no second
  tracker. Equality is PROVEN per decision: `bot_view.py` renders every attribute any bot reads as
  one canonical string, the Rust `View::render` renders the same string from the reading, and the
  gate compares their FNV-1a-64 before the action. Floats cross as IEEE-754 bits.
- **Randomness — the SAME STREAM, not explicit inputs:** `bots::rng::PyRandom` is CPython 3.11's
  MT19937, matching its `seed(int)`, `random()`, `getrandbits`, `_randbelow` and `choice` (golden
  values from CPython in the unit test). Why:
  - A seeded Python bot and the Rust bot draw the same numbers, which is what lets a bot battle be
    the same battle on both paths (Lane H's "same seed set on both paths").
  - The corpus banks each stream's OFFSET (in 32-bit words) before and after every decision, so a
    wrong draw COUNT fails even when the action happens to agree.
- **The action compared:** the ORDER THE ENV SENT (after poke-env's `SinglesEnv` round trip), as a
  choice token, and its 11-dim index (`serialize.order_to_action`). That index is what the env core
  takes. In the probe (4,614) and COMMIT (1,976) corpora the sent order never differed from the
  bot's own (the milestone recorder does not re-check it; the gate compares the sent one).
- **Behaviour is ported AS IS, bugs included** (F-LF-1). Fixing a bot is a training-distribution
  change and is the owner's call.

## Gate results (2026-09-29)

| tier | corpus | decisions | result |
|---|---|---|---|
| COMMIT (routine) | banked, 51 episodes: 10 bots × (3 pool + 1 ladder + 1 procedural) + 1 chosen `staller_v2` battle | 1,976 | **0 mismatches** (view, action, token, RNG offsets, replay) |
| probe (one-off) | 120 episodes, pool + ladder | 4,614 | 0 mismatches |
| MILESTONE (`slow`) | 520 fresh episodes, 10 bots × (20 pool + 20 ladder + 12 procedural) | **20,229** | **0 mismatches**, 73 s |

- **Draws exercised (milestone):** random 4,348 decisions; staller 174; staller_v2 108; baitbot 83
  (bait coin + forced-switch path).
- **Teeth (routine), each failing on its own counter:** one decision's action moved; one view hash
  moved; one stream offset moved; aggressive's battles relabelled as setup_sweep_v2's (real
  behaviour differs, so the actions diverge).
- The COMMIT bank RE-RECORDS byte for byte in the routine test, so a Python bot change fails there
  first.
- **Branch coverage** (the gate reports the `logic.rs` return site of every decision):
  - Every non-fallback site is reached, except these, which are structurally unreachable at a real
    decision: `setup_sweep` / `setup_sweep_v2`'s `available_moves[0]` fallback and baitbot's
    max-base-power fallback (`best_damage_move*` is never `None` once moves and both actives exist).
  - Every bot's "no active mon" guard and its final random fallback are unreached.
  - Those fallbacks call the same `logic::random_move` RandomPlayer uses (4,348 exercised draws).

## Findings

- **F-LF-1 (MAJOR, owner — behaviour, not port).** The setup branch NEVER fires in four bots:
  `SimpleHeuristicsPlayer`, `Gen3HeuristicV2Player`, `Gen3SetupSweepPlayer` and
  `Gen3SetupSweepV2Player`. Each tests `move.target == "self"`, and poke-env's `Move.target` is a
  `Target` ENUM (`Target.SELF == "self"` is False).
  - So `setup_sweep` / `setup_sweep_v2` never set up. They are attackers with switch logic, and every
    eval row named for them measures that.
  - Ported as-is: `calc::target_is_self_str` returns false.
  - `bot_tables_test.py` pins the Python fact, so a fix fails there first and the port must follow.
  - The same bots also carry upstream poke-env's `"stealhrock"` typo in `ENTRY_HAZARDS`. It is inert
    in gen 3, which has no Stealth Rock.
- **F-LF-2 (Lanes E / G / H).** PHANTOM POLLS: the training wrapper (`SingleAgentWrapper.step`)
  calls `choose_move(env.battle2)` on steps whose p2 order is never sent.
  - Where they appear: runs of turns where p1 decides and p2 holds a live move request it is not
    asked to answer.
  - Probe: 38 in 120 episodes. `random` and `baitbot` consumed draws on them.
  - The core opens NO decision there (every banked battle replays with equal per-side counts), so the
    Rust env asks a bot only at a real decision.
  - A seeded bot's stream therefore sits at a different offset on the two paths. The gate absorbs
    that by skipping to the banked offset.
  - Eval (`Player`-driven, no wrapper) has no phantom polls.
- **F-LF-3 (Lanes E / G — declared seeding change).** Production Python bots are UNSEEDED:
  - the `choice` and `protect` streams are the process-wide `random` module, shared with every other
    user of it;
  - BaitBot's `random.Random(None)` is OS entropy.
  - Neither is reproducible. The Rust env must seed every stream explicitly (`Bot::new(kind,
    choice, protect, bait)`) from its staged episode inputs to keep gate ② (seed → bytes). That is a
    change of STREAM, not of distribution; the host decides the seed rule (proposal: derived from
    `ep_seed` + side).
- **F-LF-4 (Lane E — wiring not built here).** No bot is wired into `pool.rs`: opponent routing is
  Lane E's (`opponents.rs`). The API it calls at a p2 decision is `Bot::decide(chain.stream(1)
  .board_reading, &open.tokens) -> Decision { order, token, index }`, and it feeds `index`. A
  `Default` order has no index; none occurred in any corpus. A `BotError` is a typed refusal (where
  the Python bot would raise); the lane must quarantine it like a port error. BaitBot's dial must
  be the RUN's `--bait-bot-p` (`Kind::BaitBot { p_bait }`); `Kind::from_name("baitbot")` gives
  the parser default 0.6, which is what every corpus here used.
- **F-LF-5 (stall forfeit).** At the forfeit decision the Python bot IS asked and its order is never
  sent. The corpus banks it with `idx` null and compares the bot's own order.
- **UNVERIFIED:** the Rust bots' cost per decision (not measured; the view clones every mon's
  moveset per decision); behaviour past the production 250-turn forfeit outside the corpus
  (random-p1 battles reach it rarely: 1 forfeit in the commit tier).
