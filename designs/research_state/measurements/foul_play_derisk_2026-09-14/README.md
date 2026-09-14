# Foul Play as a future eval opponent — the lazy de-risk

**Measured 2026-09-14.** Owner question: is [pmariglia/foul-play](https://github.com/pmariglia/foul-play)
(search bot) + [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine) (its Rust search
engine) usable as an EXTERNAL gen3ou eval opponent for us — and if so, at what cost and with what
plumbing? This pass answers it the cheap way: a local Showdown server on a 9XXX port, one of our
checkpoints playing through `src/main/play.py`, Foul Play as the other websocket client. **No
websocket shim was built**; §7 is the feasibility note for one.

Everything here is a MEASUREMENT of what happened on this box on this day. Nothing was changed in
training code, no run directory was written, and the live training arm was not touched.

---

## 1. Setup and provenance

| thing | value |
|---|---|
| gen3ai commit the games ran at | `f234dae1a407d329626d0d24e30d2de03253821e` |
| `deps/pokemon-showdown` pin | `e0551883ff8c676937a39ae8f4d6c0caf9de1613` (2026-05-09) |
| Foul Play checkout | `/home/goodlad/dev/foul-play` @ `6c467c081e862fb321adb405355beb41aba8e226` (2026-09-06) |
| poke-engine (installed) | PyPI `poke-engine==0.0.48`, built from source |
| poke-engine (source read for this note) | `/home/goodlad/dev/poke-engine` @ `f4e224c75bf7af885c85c1dcba982b4143ebf582`; tag `v0.0.48` = `bcf13823abc162a608e187b26bbf683f759f385e` |
| Foul Play interpreter | its OWN conda env at `/home/goodlad/dev/foul-play/.conda` (Python 3.11.15) — **never `gen3ai_stable`** |
| rustc | 1.96.0 |
| Showdown server | `node deps/pokemon-showdown/pokemon-showdown start --no-security 9317`, started and stopped by PID; **:8000 and :8001 untouched** |
| our checkpoint | `models/ai_v12_02_winprob_critic/final_model.zip` @ 75,005,952 steps (`arch_signature = gen3_critic_route_wave_v1`, `config_version` 110, `critic: winprob`) |
| our device | `cpu`, everything `nice -n 10` (a training arm held the GPU throughout) |

### The engine build

```bash
cd /home/goodlad/dev/foul-play
/home/goodlad/miniconda3/bin/conda create -y -p ./.conda python=3.11
./.conda/bin/pip install requests==2.33.0 websockets==14.1 python-dateutil==2.8.0
export CARGO_TARGET_DIR=<scratch>          # NEVER src/rust_sim/target — see the symlink incident
./.conda/bin/pip install -v --force-reinstall --no-cache-dir 'poke-engine==0.0.48' \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"
```

Built clean in 12.8 s. `--features poke-engine/gen3 --no-default-features` is the documented
per-generation build (`poke-engine/Makefile:gen3`, and the `gen3` feature is declared in
`poke-engine/Cargo.toml`'s `[features]`).

### Is gen3 actually supported by poke-engine?

**Yes, as a first-class compile target — with one real coverage caveat.**

* `gen3` is a declared cargo feature and selects a WHOLE SEPARATE ENGINE MODULE: `src/lib.rs:9-11`
  maps `poke_engine::engine` to `src/gen3/mod.rs` (6,445 lines across `abilities.rs`,
  `base_stats.rs`, `choice_effects.rs`, `damage_calc.rs`, `evaluate.rs`,
  `generate_instructions.rs`, `items.rs`, `state.rs`) rather than the shared `genx/`. gen1, gen2
  and gen3 are the three generations that get this treatment.
* `poke-engine/Makefile` has an explicit `gen3:` target. `CHANGELOG.md` shows sustained gen3 work
  (its own module split, gen3 Protect side-condition semantics, pinch berries at end of turn,
  gen3-only flags removed from later gens, gen3 eval tuning).
* ⚠️ **The README says "Generations 4 through 8 are available."** That line is stale relative to
  the Makefile and the feature list — but it is a fair warning about where the attention is.
* 🚨 **The coverage caveat, and it is the honest headline:** under `--features gen3` the engine's
  MAIN mechanics suites are compiled OUT. `tests/test_battle_mechanics.rs` (**709** tests),
  `tests/test_damage_dealt.rs` and `tests/test_last_used_move.rs` all open with
  `#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]`. gen3's dedicated
  coverage is `tests/test_gen3.rs` — **19 tests**, all of which pass here
  (`cargo test --release --features gen3 --no-default-features --test test_gen3` → 19 passed).
  So the gen3 engine is *deliberately built* and *lightly pinned*: 19 tests against our own port's
  gate ladder is not a comparable number, and any future use of Foul Play as a *mechanics* oracle
  (as opposed to an opponent) would need its own differential gate.
* poke-engine's own README opens with **"This is not a perfect engine"** and says it is "nowhere
  near as complete or robust as the PokemonShowdown battle engine." Taken at its word: it is a
  SEARCH engine, and the ground truth in these games was the real Showdown server both clients
  connected to, not poke-engine.

### Teams, and the set-prediction source

Both sides play **our** pool. The 72 curated sample teams in `data/teams/sample/*.txt` were
exported to `/home/goodlad/dev/foul-play/fp/teams/teams/gen3/ou/gen3ai_pool/` by
`scripts/export_pool_teams.py`.

🚨 **The export is not a copy.** Our pool pastes omit the `IVs:` line for Hidden Power users;
`Gen3Teambuilder` patches it in at pack time via `utils.gen3_utils.fix_gen3_hp_ivs`
(`GEN3_HP_IVS`). Foul Play has no such step — `fp/teams/team_converter.py` packs exactly what the
paste says, so a `Hidden Power [Grass]` with no IV line packs as 31/31/31/31/31/31, i.e. Hidden
Power **Dark**, and the team is a different team (or invalid). The exporter therefore WRITES the
IV line into the paste, from `GEN3_HP_IVS` — one source of truth, both clients.

Verified: all 72 exported teams pack through Foul Play's `export_to_packed` and then validate
clean against the pinned Showdown `TeamValidator` for `gen3ou` (**72/72, 0 errors**).

Per session our side pins ONE pool team (`--team <file>`, rotating across 8 teams); Foul Play
draws a random pool team per battle (`--team-name gen3/ou/gen3ai_pool` — a directory, which
`fp/teams/load_team.py:load_team` randomises over).

**Set prediction for gen3ou** — what Foul Play actually loaded (from its own log lines):

| source | file | size |
|---|---|---|
| Smogon usage stats (chaos) | `https://www.smogon.com/stats/2026-08/chaos/gen3ou-0.json` | 2.28 MB |
| Showdown's published sets | `https://play.pokemonshowdown.com/data/sets/gen3ou.json` | 62 KB |
| Foul Play's replay-derived full sets | `https://data.foulplay.cc/gen3ou/pokemon_full_sets.json` | 425 KB |
| Foul Play's replay-derived moves | `https://data.foulplay.cc/gen3ou/replay_moves.json` | 156 KB |

It uses the **`-0` (unweighted) chaos file** deliberately (`fp/data/sets/smogon.py:190`, comment:
"the higher ladder is for noobs") from the PREVIOUS month. Per-decision it logs which source a
sampled opponent set came from (`source=teamdatasets-full` / `teamdatasets-partial` /
`smogon-...`). gen3ou has **no team preview**, so Foul Play adds opponents to its datasets as they
are revealed (`fp/modes/standard_battle.py:add_revealed_pokemon`).

---

## 2. The forfeit rule — what it is and where it lives

**The rule: a battle is forfeited by us at turn 250. It is a pure TURN CAP — there is no separate
no-progress or stall-detection rule anywhere in the training pipeline.**

One constant, four consumers:

| | |
|---|---|
| the number | `MAX_TURNS = 250` — `src/agents/observation/constants.py:395` |
| the deadline | `StallConfig.threshold` defaults to `MAX_TURNS` — `src/agents/training/stall.py:16` |
| the act | `Gen3Player._handle_stall` returns a `ForfeitBattleOrder` once `view.turn >= threshold`, called FIRST in `RLPlayer.choose_move` before any model forward — `src/agents/inference/player.py:218-229`, `:725-728` |
| the reward's matching TIMEOUT test | `src/agents/training/reward_weights.py:23-25` |
| the obs coupling | the same `MAX_TURNS` normalises the observation deadline clock (`gen3_deadline_clock_v1`), which is why the limit must not be moved for a play session — past it the critic's `turns_remaining` scalars are out of distribution |

**`play.py` applies it, and as of 2026-09-14 the limit is a NAMED flag.** `main.play` grew
`DEFAULT_FORFEIT_TURN_LIMIT = StallConfig().threshold` and `--forfeit-turn-limit`
(`src/main/play.py:62`, `:297`), threaded into `RLPlayer`'s `stall_config` and printed at startup,
pinned by `src/main/play_forfeit_limit_test.py`. That landed in commit `127cf199` from the
**parallel Metamon de-risk** on the same day — see hazard **H3**, because it landed *underneath
this campaign*. These games therefore ran at the flag's DEFAULT, which is the trainer's own
constant, which is exactly what a head-to-head needs: nothing was overridden, and no second
literal exists.

**How many of the 80 games hit it: ZERO.** Max observed length 110 turns (§4).

---

## 3. How the games were run

```
Showdown (deps/pokemon-showdown, pinned) --no-security  on :9317
        ^                                        ^
        |  ws (poke-env, our fork)               |  ws (websockets 14.1)
        |                                        |
  our RLPlayer  <-- accepts challenges --  Foul Play (challenge_user)
  models/ai_v12_02_winprob_critic          poke-engine 0.0.48 (gen3), MCTS
  --team <one pool team>                   --team-name gen3/ou/gen3ai_pool
  temperature 0 (greedy), cpu              --search-time-ms 1000, parallelism 1, threads 1
```

**8 sessions x 10 games.** Each session pins a different pool team for our side and uses fresh
guest accounts (`G3aiD914s<N>` / `FpD914s<N>`). `scripts/run_derisk.py` is a thin wrapper over
`main.play`'s own `resolve_server` / `build_teambuilder` / `build_account` /
`build_model_player`, so the client under measurement IS the `play.py` path; what the wrapper adds
is per-battle forensics (`play.py` is a session runner, not a logger). Foul Play's stdout is
line-timestamped by `scripts/ts.py`, because its own formatter
(`fp/config.py::CustomFormatter`) prints no timestamps — that is where its per-move think time
comes from.

**On the search budget.** Foul Play exposes **only a WALL-CLOCK budget**: `--search-time-ms`
(`fp/config.py:119-124`, default 100 ms), passed straight to
`poke_engine.monte_carlo_tree_search(state, search_time_ms, threads=...)`
(`fp/search/main.py:53`). There is **no iteration / visit-count / depth budget anywhere** in
either the Foul Play CLI or the Python binding's MCTS entry point. Per
`UNDERSTANDING.md`'s rule 23, a wall-clock budget on a contended box is a **width meter**, not a
setting: the realized search WIDTH varies with load, so two Foul Play opponents at the same
`--search-time-ms` are not the same opponent unless the box is. 1000 ms was chosen as the modest
middle of the 1-3 s band; the **realized** width is recorded per game
(`fp_mcts_visits_mean` in `games.jsonl`, read back from Foul Play's own
`Iterations {i}: {res.total_visits}` line) precisely so the hazard is measured rather than assumed.

Note `--search-parallelism` is not a threading knob for one search — it is how many SAMPLED
opponent-set worlds are searched, each for the full `--search-time-ms`
(`fp/modes/standard_battle.py::search_params`, which DOUBLES the world count while the opponent's
active has fewer than 3 revealed moves). At `--search-parallelism 1` a decision costs 1-2 s of
one core.

---

## 4. Results — 80 games

`games.jsonl` is one JSON object per battle, in play order. Regenerate the tables with
`scripts/analyze.py` and `scripts/table.py`.

| session | our team | games | our wins | win rate | mean turns | max turns | our think ms (mean) | FP think s (mean) | FP visits/decision (mean) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `009e3d0244.txt` | 10 | 4 | 0.40 | 29.0 | 50 | 43 | 2.13 | 1.46 M |
| 1 | `4239fc5ba2.txt` | 10 | 3 | 0.30 | 26.4 | 44 | 320 | 2.26 | 1.35 M |
| 2 | `64a691c473.txt` | 10 | 5 | 0.50 | 35.4 | 92 | 93 | 2.12 | 1.43 M |
| 3 | `7d0337af97.txt` | 10 | 2 | 0.20 | 33.4 | 69 | 206 | 2.03 | 1.30 M |
| 4 | `9b454d9ea7.txt` | 10 | 6 | 0.60 | 45.4 | 104 | 41 | 2.11 | 1.48 M |
| 5 | `a185b2d193.txt` | 10 | 5 | 0.50 | 27.9 | 37 | 179 | 2.11 | 1.53 M |
| 6 | `b904dbe059.txt` | 10 | 4 | 0.40 | 37.7 | 55 | 51 | 2.15 | 1.21 M |
| 7 | `e11829f0561ef5a9.txt` | 10 | 2 | 0.20 | 38.7 | 110 | 45 | 2.20 | 1.47 M |
| **all** | 8 pool teams | **80** | **31** | **0.388** | **34.2** | **110** | **122** | **2.14** | **1.40 M** |

### Headline

| | |
|---|---|
| **our win rate vs Foul Play @ 1000 ms** | **0.388 (31 / 80)**, Wilson 95% CI **[0.288, 0.497]** |
| wins / losses / ties | 31 / 49 / **0** |
| turns | mean **34.2**, median **28**, p90 **57**, max **110** |
| our 250-turn forfeit fired | **0 / 80** |
| timeouts / games lost on the clock | **0** |
| protocol-parse failures, OUR parser (`UnknownMessageType` / `UnsupportedMessageType`) | **0** |
| Foul Play exceptions / ERROR+CRITICAL log lines | **0** over 3,040 decisions |
| winner disagreement between the two clients | **0 / 80** (our `battle.won` vs Foul Play's `\|win\|` read) |
| our think time | mean 122 ms/decision, **median 50 ms**, max 2,399 ms (a first decision; torch warm-up) |
| Foul Play think time | mean **2.14 s**/decision, max 2.73 s |
| Foul Play realized search | **1.40 M** MCTS visits per decision (its own `total_visits`) |

**The CI straddles 0.5's lower neighbourhood but not 0.5 itself** — at this budget, on this box,
against this team distribution, our 75M win-prob-critic arm is **behind** Foul Play, and by enough
that the interval excludes parity. Treat it as an ORDER-OF-MAGNITUDE placement of Foul Play, not a
rating: the search budget is wall-clock on a contended box (see (c)), and the team distribution is
asymmetric (we pin one pool team per session, Foul Play redraws every battle).

### Cost

| | |
|---|---|
| decisions per game | **38** (ours 38.7, Foul Play 38.0) |
| Foul Play CPU per game | **80 core-seconds** (mean; median 70) at `--search-time-ms 1000`, `--search-parallelism 1` |
| our CPU per game | **4.6 core-seconds** |
| wall clock per game | **~88 s** (118 min of net play for 80 games, one game at a time) |
| Showdown server | negligible; one `node` process, started and stopped by PID |

Foul Play is **~17x our per-game cost** at this budget, and the budget is the only knob: cost
scales linearly in `--search-time-ms` x `--search-parallelism` x the world-count multiplier
(2x while the opponent's active has <3 revealed moves).

---

## 5. The four verdicts

### (a) Does poke-engine / Foul Play play gen3ou at all? — **YES, and competently.**

80 complete gen3ou games, zero crashes, zero forfeits, zero protocol failures, and it **won 49 of
80** against a 75M-step trained policy. It runs the no-team-preview gen3 path correctly
(`fp/generations.py::GEN3` carries the real gen3 deltas: no choice scarf, Pressure not announced
on switch-in, consecutive-Sleep-Talk tracking, gen3/gen4 Taunt-at-end-of-turn, permanent
ability weather), it infers opponent sets from Smogon 2026-08 usage plus its own replay-derived
sets, and its reverse-damage narrowing is enabled for gen3.

**The caveat is engine TEST coverage, not capability** (§1): under `--features gen3` the 709-test
`tests/test_battle_mechanics.rs` plus the damage-dealt and last-used-move suites are compiled out,
leaving 19 gen3-specific tests. So: fine as an OPPONENT, not yet trustworthy as a mechanics ORACLE.

### (b) Protocol compatibility with our pinned `deps/pokemon-showdown`? — **CLEAN, both directions.**

Both clients spoke to the same pinned server (`e0551883`, 2026-05-09) for 80 games and 3,040
decisions each. Our `battle_event.classify` tripwire — which raises `UnknownMessageType` on any
keyword it does not know, BY DESIGN, and whose raise kills the parse task and loses the battle on
the timer — **never fired**. Foul Play's parser never raised either.

⚠️ **The two clients have OPPOSITE drift postures, and this run does not discriminate between
them.** Ours is a loud tripwire; Foul Play's `process_battle_updates` looks the action up with
`battle_modifiers_lookup.get(action)` and **silently ignores anything it does not know**
(`fp/battle/protocol.py:2351-2353`). On a pinned local server neither posture is exercised. Against
a LIVE public server, ours would crash and Foul Play's would quietly mis-model — so a green result
here says nothing about either on the open ladder. `src/main/ladder_drift_scan.py` remains the
instrument for that question.

### (c) Cost per game, and is there an iteration budget? — **80 core-seconds/game; NO iteration budget exists.**

🚨 **Foul Play exposes only `--search-time-ms`** (`fp/config.py:119-124`), passed straight to
`poke_engine.monte_carlo_tree_search(state, search_time_ms, threads=...)`
(`fp/search/main.py:53`). There is no visit-count, node-count or depth budget in the CLI or in the
Python binding's MCTS entry point. Per `UNDERSTANDING.md` rule 23 this makes the opponent a
**width meter**: at a fixed 1000 ms the realized search varies with box load, so "Foul Play at
1000 ms" is not a reproducible opponent across machines or across a busy/idle box.

**Mitigation used here, and the one to keep:** the realized width is RECORDED per game
(`fp_mcts_visits_mean`, from Foul Play's own `Iterations {i}: {total_visits}` line). Across this
campaign it was **1.40 M visits/decision, session means 1.21-1.53 M — a ±11% spread around the
mean at a constant nominal budget.** That spread IS the hazard, measured. Any future use as a
graded baseline should either (i) report the realized visit count alongside every win rate, or
(ii) patch an iteration budget into the engine call — poke-engine's own CLI has
`monte-carlo-tree-search --time-to-search-ms` only, so an iteration budget would be a small
upstream change to `mcts.rs`, not a config.

### (d) Shim feasibility? — **FEASIBLE and small (0.5-1 day), but it buys determinism, not throughput.** See §7.

---

## 6. Hazards — every one of these is a FINDING, not a footnote

**H1 — 🚨 A 19-character username silently wedges BOTH clients, forever.**
Both clients guest-login through **Smogon's `action.php` even against a `--no-security` LOCAL
server**. `act=getassertion` with a name of 19+ characters returns
`;;Your username must be less than 19 characters long.` — a `;`-prefixed REFUSAL, not an
assertion. Foul Play's guest path (`fp/websocket_client.py:120-121`) does
`assertion = response.text` with **no validation**, `/trn`s the refusal string, logs
**"Successfully logged in"**, and then blocks forever in `get_battle_tag_and_opponent`. Our side
sat in `accept_challenges` waiting for a user that does not exist. Cost here: one dead session and
~18 minutes, diagnosed only by querying `/cmd userdetails` from a third connection and finding the
user `null`.
This is **exactly the failure class our own ladder audit fixed as gap #10**
(`designs/research_state/ladder_readiness.md`) — `poke_env.ps_client._parse_login_assertion` +
`LoginError`. Foul Play has the unfixed version. `scripts/run_session.sh` now preflights both names
against `action.php` and ABORTS on a `;` reply.
**Read-across: if we ever put our own client behind a name-generation scheme, the 18-character
ceiling is a real constraint, and a refused assertion must never present as a hang.**

**H2 — 🚨 Foul Play does NOT apply our Hidden Power IV fix, so a naive team copy is a DIFFERENT team.**
Our pool pastes omit the `IVs:` line; `Gen3Teambuilder` injects it from
`utils.gen3_utils.GEN3_HP_IVS` at pack time. `fp/teams/team_converter.py` packs the paste verbatim,
so `Hidden Power [Grass]` with no IV line becomes 31/31/31/31/31/31 = Hidden Power **Dark**. The
team either fails validation or plays a different move. Fixed by WRITING the IVs into the exported
paste (`scripts/export_pool_teams.py`), then verifying all 72 through Foul Play's own packer and
the pinned Showdown `TeamValidator` (72/72 clean, `scripts/validate_packed.js`).
**Generalises to every third-party client we hand a team to: our pastes are only complete under
our own teambuilder.** (Compare the sibling Metamon finding — upstream poke-env mis-parsing a
nicknamed, item-less line — same class, different client.)

**H3 — ⚠️ `main`'s `play.py` changed underneath a running campaign, and killed a session.**
`scripts/run_derisk.py` imports `main.play` off the MAIN checkout's `src/`. Mid-campaign, the
parallel Metamon de-risk landed `--forfeit-turn-limit` (commit `127cf199`), so
`build_model_player` began reading `args.forfeit_turn_limit` — an attribute the driver's args
object did not have. Session 4 died with `AttributeError` after 40 games, and the failure was
**silent at the campaign level**: `run_all.sh` cheerfully started the Foul Play half, which hung
until its 2,400 s `timeout`. Sessions 4-7 were re-run after fixing the driver; games 1-40 and
41-80 are otherwise identical (same server process, same checkpoint, same budget).
**The lesson is the measurement one: a long-running job that imports the main checkout is not
pinned, and `main` is a moving target on a multi-session box.** A campaign should either run from
its own worktree or pin the commit it imports.

**H4 — ⚠️ The opponent's search budget is WALL CLOCK, so the opponent is not a constant.**
See verdict (c). Recorded, not assumed: 1.21-1.53 M visits/decision across sessions at a constant
1000 ms.

**H5 — ⚠️ The team distribution is ASYMMETRIC in this campaign.** Our side pinned one pool team per
10-game session; Foul Play redrew from all 72 every battle. That is deliberate (it is what
`play.py --team` and `--team-name <dir>` give for free) but it means the per-session win rates
(0.20-0.60) mix team-matchup variance with skill, and the 0.388 headline is an average over 8 of
our teams against the whole pool, not a like-for-like mirror. A graded rerun should either pin both
sides or let both redraw.

**H6 — (non-finding, recorded so it is not re-derived) Showdown writes no battle logs under
`--no-security`.** `deps/pokemon-showdown/logs/` gains nothing for a local challenge game, so
per-game results have to come from the two clients. The cross-check that they agreed on the winner
in 80/80 games is what stands in for a server-side ground truth here.

---
## 7. Shim feasibility — what a fake-socket adapter would have to fabricate

**Verdict: small and well-contained.** Foul Play's entire network surface is ONE class,
`PSWebsocketClient` in `/home/goodlad/dev/foul-play/fp/websocket_client.py:20-182`
(163 lines), and **every other module takes it as a parameter** — `fp/main.py`,
`fp/run_battle.py`, `fp/modes/base.py`, `fp/modes/standard_battle.py`,
`fp/modes/random_battle.py` and `fp/modes/bss.py` all receive a `ps_websocket_client` argument
and never construct one. There is no global, no module-level socket, and no `websockets` import
outside that file.

### The seam

| owner | file:line |
|---|---|
| the class | `fp/websocket_client.py:20` |
| the ONLY socket construction | `fp/websocket_client.py:35` — `self.websocket = await websockets.connect(self.address)`, inside the classmethod `create` (`:29-41`) |
| the ONLY read | `fp/websocket_client.py:48-51` — `receive_message()` → `await self.websocket.recv()` |
| the ONLY write | `fp/websocket_client.py:53-57` — `send_message(room, message_list)` → `room + "\|" + "\|".join(message_list)` |
| teardown | `fp/websocket_client.py:79-80` — `close()` |
| the only construction site in the app | `fp/main.py:51-53` — `await PSWebsocketClient.create(username, password, websocket_uri)` |

**A shim is therefore: a class with `recv()` / `send()` / `close()`, plus a `create()` that
installs it and a `login()` that returns a userid without a network round trip.** Subclassing
`PSWebsocketClient` and overriding `create`, `login`, `close` is enough — `receive_message` and
`send_message` need no change at all, since they only touch `self.websocket`.

### What must be fabricated (gen3ou, i.e. the NO-team-preview path)

Everything Foul Play reads, with the reader:

| line | read by |
|---|---|
| `\|challstr\|<id>\|<str>` | `websocket_client.py:82-87` (`get_id_and_challstr`) — **skippable** if `login()` is overridden |
| `\|pm\| <from>\| <to>\|/challenge\|<format>\|...` (9 fields) | `websocket_client.py:144-164` — only in `accept_challenge` mode |
| `>battle-gen3ou-<id>` + a room title line whose 5th `\|` field is `"<A> vs. <B>"` | `fp/modes/base.py:243-255` (`get_battle_tag_and_opponent`) |
| `\|player\|p1\|<name>\|<avatar>\|<rating>` | `fp/modes/base.py:143-148` (`start_battle_common`) — it waits for the line naming the OPPONENT, and derives its own `p1`/`p2` from it |
| `\|start` | `fp/modes/standard_battle.py:37` |
| `\|request\|<json>` | `fp/modes/base.py:258-269` (first one) and `fp/battle/protocol.py:2279-2282` (every one after) |
| the whole per-turn `\|move\|`/`\|switch\|`/`\|-damage\|`/... stream | `fp/battle/protocol.py::process_battle_updates` (the `battle_modifiers_lookup` dict, `:2300-2352`) |
| `\|win\|<name>` or `\|tie` | `fp/run_battle.py:13-18` (`battle_is_finished`) |
| `\|deinit` | `fp/websocket_client.py:171-178` (`leave_battle`) |

🚨 **`\|request\|` is the decision TRIGGER, not just state.** `fp/battle/protocol.py:2272-2287`
buffers every protocol line into `battle.msg_list` and returns `action_required` ONLY when a
`\|request\|` arrives; `fp/run_battle.py:60-63` then searches and sends. A shim that streams the
battle log but never emits a `\|request\|` produces a bot that never moves. The request JSON must
carry `rqid` (`fp/modes/base.py:268`), because every choice Foul Play sends is
`"<battle_tag>\|/choose move X\|<rqid>"` (`fp/modes/base.py:272-313` → `format_decision`, and the
`str(battle.rqid)` second element at `:312`).

### What must be consumed

`send_message` writes `room|part1|part2|...`. A shim must interpret:
`/trn`, `/utm <packed team>`, `/challenge <user>,<format>` / `/accept <user>` / `/search <format>`,
`<tag>|/choose move <m>|<rqid>`, `<tag>|/switch <idx>|<rqid>`, `<tag>|/team <order>|<rqid>`
(team preview — unused in gen3), plus the cosmetics `hf`, `/timer on`, `gg`, `/savereplay`,
`/leave`, `/avatar`, `/join`, `/cmd userdetails`. Only the `/utm` + `/choose` + `/switch` + the
challenge verb carry meaning; everything else can be swallowed.

### What we already have to feed it

Our Rust port emits the byte-identical omniscient `|...|` stream
(`designs/rust_sim/protocol_emission.md`), and `src/utils/bridge/bridge_session.py` already
delivers per-side `|request|` JSON to poke-env for the trainee — i.e. both halves a fake socket
needs already exist on our side. The missing piece is a per-side VIEW of the protocol (Foul Play
must not see the omniscient stream) and the room framing above.

**Estimate: 0.5-1 day** for a working adapter against the bridge, most of it in the per-side
protocol view and the `|request|` shape, not in the socket. The risk is not the seam — it is that
a shim silently feeds Foul Play a slightly different game than the server would, and the only
honest gate for that is a differential: the same battle over the real websocket and over the shim,
same seeds, compared decision-for-decision.

### The cheaper alternative this pass already proves out

**A local Showdown server on a 9XXX port needs no shim at all** and is ~1.2 s/game of server
overhead. If the reason for a shim is COST, the shim does not buy much: at
`--search-time-ms 1000` the search dominates the server by two orders of magnitude (§4). A shim
buys determinism and removes the websocket, not throughput.

---

## 8. What is in this directory

| file | what |
|---|---|
| `README.md` | this |
| `games.jsonl` | 80 records, one per battle, in play order |
| `foul_play_cli.txt` | the exact Foul Play invocation + every default it inherited |
| `scripts/export_pool_teams.py` | pool paste -> Foul Play paste, WITH the Hidden Power IV line (H2) |
| `scripts/validate_packed.js` | validates Foul Play's packed output against the pinned Showdown `TeamValidator` |
| `scripts/run_derisk.py` | our side — a thin forensic wrapper over `main.play`'s own builders |
| `scripts/ts.py` | line-timestamps Foul Play's stdout (its formatter has no timestamps) |
| `scripts/run_session.sh` | one session: our client + Foul Play, with the H1 username preflight |
| `scripts/run_all.sh` / `scripts/run_rest.sh` | the 8-session campaign / its resume after H3 |
| `scripts/analyze.py`, `scripts/table.py` | `games_raw.jsonl` + Foul Play logs -> `games.jsonl` + the tables |

The scripts carry absolute paths to this box's scratch directory
(`/home/goodlad/.claude/jobs/9ab51de6/tmp/foul_play/`) and to the Foul Play checkout at
`/home/goodlad/dev/foul-play/`. They are recorded as the PROVENANCE of `games.jsonl`, not as a
reusable tool; a rerun should re-path them.

The Foul Play checkout is left in place at `/home/goodlad/dev/foul-play/` (with its own conda env
at `.conda/` and the exported pool at `fp/teams/teams/gen3/ou/gen3ai_pool/`), and the poke-engine
source at `/home/goodlad/dev/poke-engine/`. Neither is inside this repo.

---

## 9. Ledger paragraph (ready to append — this file does NOT edit `ledger.md`)

**2026-09-14 · FOUL PLAY DE-RISKED as a gen3ou eval opponent — GO, with an explicit width-meter
caveat.** Foul Play (`6c467c08`) + poke-engine `0.0.48` built `--features poke-engine/gen3` played
**80** complete gen3ou games against `ai_v12_02_winprob_critic@75,005,952` over a local pinned
Showdown server on :9317, both sides drawing from our 72-team sample pool. **Our win rate 0.388
(31/80), Wilson 95% [0.288, 0.497]** at `--search-time-ms 1000 --search-parallelism 1` — i.e. the
external bot is AHEAD of the 75M win-prob arm at that budget, and the interval excludes parity.
Mean 34.2 turns, median 28, max 110; **0/80 reached the 250-turn forfeit**, 0 timeouts, 0 ties.
**Protocol: clean both ways** — our `battle_event.classify` tripwire never fired over 3,040
decisions, Foul Play raised nothing, and the two clients agreed on the winner 80/80; note the
postures are OPPOSITE (we raise on an unknown keyword, Foul Play silently ignores it), so a pinned
local server exercises neither and `ladder_drift_scan` stays the instrument for live drift.
**Cost: ~80 core-seconds and ~88 s wall clock per game, ~17x ours.** 🚨 **There is NO iteration
budget** — `--search-time-ms` is wall clock only (`fp/search/main.py:53`), which makes the opponent
a width meter (rule 23): realized search was 1.40 M MCTS visits/decision, session means
1.21-1.53 M at a CONSTANT nominal budget, so every future win rate against Foul Play must carry
its realized visit count. **gen3 is a first-class poke-engine compile target** (its own 6.4k-line
`src/gen3/` module, a `gen3` cargo feature, a Makefile target) but is **lightly pinned**: under
`--features gen3` the 709-test `test_battle_mechanics.rs` and the damage/last-used-move suites are
compiled OUT, leaving 19 gen3 tests (all pass) — fine as an OPPONENT, not yet a mechanics ORACLE.
**A websocket shim is feasible and small** (0.5-1 day): Foul Play's entire network surface is one
163-line class, `fp/websocket_client.py::PSWebsocketClient`, which every other module receives as a
parameter; a fake socket must fabricate `|challstr|`, the `>battle-` room framing, `|player|`,
`|start`, the turn stream, `|win|`/`|deinit|` and — decisively — `|request|` with its `rqid`, since
`fp/battle/protocol.py:2279` makes the request the DECISION TRIGGER. It buys determinism, not
throughput: at this budget the search dominates the server by two orders of magnitude.
**Three hazards, each a finding:** (1) a **19-character username** gets a `;`-prefixed refusal from
`action.php` — which even a `--no-security` LOCAL server consults — and Foul Play's guest path
accepts it verbatim, logs "Successfully logged in" and hangs forever (our own gap #10, unfixed
upstream); (2) **our pool pastes are only complete under our teambuilder** — Foul Play packs the
paste verbatim, so a Hidden Power user with no `IVs:` line becomes Hidden Power Dark, fixed by
writing `GEN3_HP_IVS` into the exported paste (72/72 then validate clean); (3) **`main` moved under
a running campaign** — the parallel Metamon pass landed `--forfeit-turn-limit` mid-run and an
`AttributeError` killed session 4 silently after 40 games, which is the standing argument for a
long job importing a PINNED checkout rather than `main`.
Artifact: `designs/research_state/measurements/foul_play_derisk_2026-09-14/`.
