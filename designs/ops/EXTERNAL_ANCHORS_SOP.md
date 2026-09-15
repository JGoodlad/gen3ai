# EXTERNAL ANCHORS — the SOP

**How our strength is read against an agent NOBODY here trained.** Every other number in this
project is measured against ourselves: `ladder.json` is our own snapshots, `eval/elo` is our own
pool, `win_rate_vs_pool` is our own bots. All of them can move together and say nothing. An
external anchor is the instrument that cannot: a third-party gen3ou agent, at a fixed protocol,
whose difficulty does not drift when our training does.

**The tool is `python -m main.anchors`. This document is the procedure.** The tool enforces what
can be enforced in code (the port refusal, the regime verification, the schema, the named
failures); this file holds what cannot — which read to take when, what the bar is, and what the
numbers have been so far.

---

## 0. The two rules that govern everything below

### 🚨 RULE 1 — A NUMBER NEVER LEAVES WITHOUT ITS REGIME

*"Temperature 1.0" is not one setting across models.* Measured 2026-09-14 over 14,085 sampled
decisions:

| at T = 1.0 | plays its own argmax | so the perturbation is |
|---|---:|---:|
| `SmallRL` (13.9M) | **64.7%** | 35% of decisions |
| `SyntheticRLV2` (200.9M) | **88.0%** | 12% of decisions |

The same nominal knob is a large behavioural change for one policy and a small one for the other,
and the two models' temperature effects come out with **opposite signs** because of it. The
consequence is not academic: the de-risk's mixed-regime headline of **0.742** against `SmallRL`
became **0.520** on the like-for-like matched cell — **−22.2 pp [−34.1, −9.4]**, CI clear of zero.

So: a win rate is reported with its regime, its team set, the opponent's version and commit, our
checkpoint's step, and (for a search opponent) the realized search width. `main.anchors` stamps all
of that on **every row of `games.jsonl`**, not once per file, and `src/main/anchors/results_test.py`
reads the written file back to prove it.

### 🚨 RULE 2 — THE RECURRING READ IS GREEDY-VS-GREEDY

Both sides move together, and both sides are **verified per decision**, never assumed. Three
arguments, in order of weight:

1. **It is the protocol every other strength number here is taken under.** `ladder.json` is
   greedy-vs-greedy with symmetric builders; `--temperature 0` is `play.py`'s documented
   measurement setting; the eval-sentinel regime went greedy on 2026-09-07. A number taken at
   T = 1.0 cannot be read beside any of them.
2. **T = 1.0 is not a fixed yardstick across opponents** (the table above). A "temperature 1.0
   baseline" silently changes difficulty when the opponent model changes, and if Metamon ships a
   new policy the regime moves under us with no flag to read. Greedy is the same operation for
   every policy.
3. **Determinism narrows a fixed-n read**, which matters when the whole point is to detect
   movement in our own model between milestones.

**The cost was measured, not assumed:** greedy cost 2 forfeits in 400 greedy games (0.5%), T = 1.0
cost 2 in 400 (0.5%), and neither regime collapsed the effective sample (87–100 distinct team pairs
per 100-game cell in both).

**The standing caveat, stated rather than buried.** A deterministic policy is exploitable *in
principle* — but only by an opponent that ADAPTS, and neither side here does. Metamon's hidden
state gives it memory *within* a battle; `amago`'s rollout loop resets it on `done`, so it carries
nothing across battles and cannot mine a fixed opponent over 100 games. Our model conditions only
on the observation. Greedy-vs-greedy is safe **for this pair**, and the property that makes it safe
is an implementation fact about Metamon that could change in a release.

---

## 1. The three tiers

| tier | when | what | cost |
|---|---|---|---|
| **A — per promotion** | every time a snapshot is promoted | the **dense ladder** (`<run>/snapshot_ladder/ladder.json`), greedy-vs-greedy, compared at **matched snapshot COUNT** | already paid by the run |
| **B — per milestone** | each eval milestone | `metamon:SmallRL` **greedy, BOTH team sets, 100 games each** + **ONE t1 cell** as the distribution check | ~15 min |
| **C — era gate** | opening or closing an era | `metamon:SyntheticRLV2` **greedy, away** — bar: **Wilson lower bound > 0.50**; and **Foul Play** at a fixed `--search-time-ms` with its realized visit count | ~2 h |

### Tier A — per promotion: the dense ladder

**No external process.** The headline strength number stays `<run>/snapshot_ladder/ladder.json`
(dense, ±10), never `eval/elo` (±29). Three standing rules apply unchanged:

* a rating is only final once the RUN is — the newest Bradley–Terry node is systematically inflated;
* a cross-run comparison is at matched snapshot **COUNT**, not matched step;
* `win_rate_vs_pool` / `eval/elo` carry an **opponent-regime boundary at 2026-09-07** and are not
  comparable across it (eval sentinels went greedy and draw the trainee's own teams; the old
  asymmetry was worth **+8.9 pp**). `ladder.json` and every bot edge are UNAFFECTED.

An external anchor is not run per promotion. It is the instrument that says whether the ladder's
axis has drifted, and that question is asked at milestones, not at every snapshot.

### Tier B — per milestone: `SmallRL` greedy on both team sets

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime greedy --teamset home --games 100 --out <out>/smallrl_greedy_home
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime greedy --teamset away --games 100 --out <out>/smallrl_greedy_away
# the standing cheap check that determinism is not costing us something — NOT the headline
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime t1 --teamset away --games 100 --out <out>/smallrl_t1_away
```

**Report BOTH team sets, always.** They are not redundant: they disagreed in SIGN for `SmallRL`
(−0.030 pooled, point estimate on the *away* side) and the away set is the one no dial of ours is
set to. `SmallRL` is the recurring anchor because it is cheap (3.0 s/game, ~5 min for 100) and at
0.520 home / 0.650 away it is a **live signal in both directions**, not a floor.

### Tier C — era gates: `SyntheticRLV2`, and Foul Play

```bash
# the era bar: the Wilson LOWER BOUND must exceed 0.50. A point estimate above 0.50 is not a pass.
python -m main.anchors --model models/<run> --opponent metamon:SyntheticRLV2 \
    --regime greedy --teamset away --games 100 --out <out>/synthv2_greedy_away

# the search opponent. The budget is WALL CLOCK, so the realized visit count is part of the result.
python -m main.anchors --model models/<run> --opponent foulplay \
    --regime greedy --teamset home --games 100 --search-time-ms 1000 --out <out>/foulplay_1000ms
```

`SyntheticRLV2` is the milestone reference because it is *ahead of us* — which is what a reference
should be. Foul Play is a different axis entirely: a search bot, not a learned policy, so it probes
whatever a tree finds and a network does not.

**Read the bar honestly.** `Wilson lower bound > 0.50` is BETTER. A CI that straddles 0.50 is
**NOT DETECTED** — never "equal", and never rounded up to a pass because the point estimate is on
the right side of even.

---

## 2. The exact command, and what it does

```
python -m main.anchors [--model <run dir | .zip | run@step> | --our-side <who>] \
    --opponent {metamon:SmallRL | metamon:SyntheticRLV2 | foulplay} \
    --regime {greedy | t1} --teamset {home | away} --games N --out <dir>
```

### `--our-side` — WHO plays our half

An anchor read does not have to put one of our checkpoints on the board. Three values:

| `--our-side` | our half is | why it exists |
|---|---|---|
| `model` (default) | the `--model` checkpoint, through `main.play` | the ordinary read |
| `bot:<name>` | one of the **nine PINNED eval bots** | it puts an external anchor on the **ABSOLUTE** scale without routing through any checkpoint of ours: the bots carry fixed ratings from the bot-vs-bot round robin, so an anchor-vs-bot edge is an edge to a pinned node |
| `metamon:<Agent>` | a SECOND external peer | anchor-vs-anchor — the transitivity check on the fit |

🚨 **A bot cell is stamped `regime_matched = false`, and that is the honest label rather than a
defect.** A bot has no sampling knob to set to the peer's regime — the same shape as Foul Play —
and its policy IS the one the round robin pinned. `--regime t1` with a bot our-side is REFUSED.
`our_regime` on those rows reads `bot:<name>`.

An anchor-vs-anchor cell IS matched (both peers take `--regime` and both verify it per decision),
and its per-game record comes from **Metamon's own battle CSV** rather than from our poke-env
flags, because there is no player of ours in the process. That instrument books a TIE as a LOSS
(its `won` field is a boolean), so the rows say so; ties ran 0–1 per 100 games in the 2026-09-14
batteries.

### `--model-load {auto,bare,foreign}` — HOW the checkpoint is loaded

🚨 **A cross-run frozen snapshot FAILS a bare `MaskablePPO.load`.** The extractor is rebuilt from
the zip's own `policy_kwargs` and handed to the CURRENT `ExtractorBuild`, so every
`ai_v9_29_rev1_0823` node dies with `unexpected keyword argument 'threat_prob_outspeed'` while
every `ai_v12`/`ai_v13` node loads fine — a campaign spanning eras hits it on SOME cells only.
`foreign` uses `load_foreign_opponent`, which reads the zip's own `model_config.json` and checks
the **`arch_signature`** (observation-family compatibility, the property that actually matters for
a frozen opponent). `auto` tries `bare`, falls back, and PRINTS which ran; the winner is stamped as
`model_loader` on every row.

⚠️ **`MIGRATION_FLOOR` is a hard wall, not a hint.** A checkpoint below it — every `ai_v8` snapshot,
config version 44/45 under `ARCH_SIGNATURE = gen3_opp_hp_typed_candidates_v1` — is refused as
PRE-GENERATION by BOTH loaders. There is no way to play a pre-generation node with current code, so
an era-spanning campaign's oldest reachable node is whatever the floor admits.

| step | what happens |
|---|---|
| 1 | starts a Showdown server from `deps/pokemon-showdown` on an auto-picked **9500–9599** port, records the PID, and stops **exactly that PID** on exit or failure. **8000 and 8001 are refused in code.** `--server-uri` uses an existing server and starts nothing |
| 2 | resolves `--model` through `resolve_model_ref` — 🚨 **a bare run directory means that run's LAST SNAPSHOT**; name the `.zip` or `@step` to pin a file. The rung it resolved by is recorded on every row |
| 3 | runs our checkpoint through **`main.play`'s own code path** (`--device cpu`, `--temperature 0`, the trainer's 250-turn forfeit limit), role-balanced across two half-series |
| 4 | verifies the regime **per decision on both sides** and writes the `argmax_match_rate` |
| 5 | writes `games.jsonl` + `summary.json` with the Wilson CI and the full provenance |
| 6 | on a dead peer or a stalled series, **FAILS with a named cause** and exit code 2 |

Other useful flags: `--dry-run` (prints the plan — both peer commands verbatim, both team sources
with their counts, every deadline — and starts nothing), `--show-config` (resolves
`designs/ops/anchors.json` and says which paths exist), `--games`, `--search-time-ms`,
`--progress-timeout` / `--first-game-timeout` / `--peer-ready-timeout`.

### The team sets

| | **home** | **away** |
|---|---|---|
| what | our 719-team gen3ou pool, nickname-free, with explicit Hidden Power IVs | Metamon's own 20-team `competitive` gen3ou set, each file `.strip()`ed |
| why | it is our training distribution | it is **their** ground — the set Metamon's README names as the paper's human-ladder set and says Metamon has overfit to |

Both sides draw from the SAME set in a cell. The summary carries `team_source_asymmetry`, which
goes true the moment the two sides' file counts differ; a cell that trips it mixes skill with
matchup and is not a clean read.

### Where the external checkouts live

`designs/ops/anchors.json`, with a per-key env override (`GEN3AI_METAMON_DIR`,
`GEN3AI_METAMON_PYTHON`, `GEN3AI_METAMON_CACHE_DIR`, `GEN3AI_FOULPLAY_DIR`,
`GEN3AI_FOULPLAY_PYTHON`, `GEN3AI_SHOWDOWN_NODE`), or `$GEN3AI_ANCHORS_CONFIG` for the whole file.
A path literal may not live under `src/` — `src/utils/paths_test.py` fails it, because a `/home/…`
literal is correct on one box and an invisible permanent skip on every other. A missing checkout is
a **named refusal** naming the key and its env var, never a skip and never a zero.

### Cost per read

Measured under a load average of 25–31 on 16 cores, with a live training arm on the box. CPU only;
nothing here touches the GPU.

| opponent | s/game | 100 games | notes |
|---|---:|---:|---|
| `metamon:SmallRL` | **3.0** | **~5 min** | 12–17 ms/decision; seconds to load |
| `metamon:SyntheticRLV2` | **6.1** | **~10 min** | 56–113 ms/decision, **plus a multi-minute build per half** (200M params, 804 MB) |
| `foulplay` @ 1000 ms | **~88** | **~2.5 h** | **~80 core-seconds/game** — ~17× ours; cost is linear in `--search-time-ms` × `--search-parallelism` × the ×2 world multiplier while the opponent's active has <3 revealed moves |

🚨 **ONE THREAD PER PEER, and it is worth ~20×.** Measured 2026-09-14 on a box at load 63: each
Metamon peer was burning **210% CPU** on B = 1 CPU inference — two cores of thread synchronisation
per peer, buying nothing — and three parallel cells ran at **33 s/game**. With
`OMP_NUM_THREADS = 1` (now set in the peer env, and defensively inside `metamon_side.py` before
torch imports) the same three lanes run a 100-game cell in **~150 s, i.e. ~1.5 s/game**. Put the
parallelism ACROSS cells, never inside one forward pass. `src/main/anchors/peers_test.py` asserts
the pin on the env the plan actually carries.

---

## 3. The known hazards — every one has cost a session

| # | hazard | symptom | where it is handled now |
|---|---|---|---|
| **H1** | **a 19-character username**. Both clients guest-login through Smogon's `action.php` *even against a `--no-security` LOCAL server*; 19+ returns `;;Your username must be less than 19 characters long.` — a REFUSAL. Foul Play's guest path accepts it verbatim, logs "Successfully logged in", and blocks forever | **a HANG** (cost one session, ~18 min) | `main.anchors.peers.check_username`, both names, before anything starts |
| **H2** | **nicknames**. Upstream poke-env 0.8.3.3 fails to split `Airmure (Skarmory)` (nickname, no item) and packs an EMPTY species; Showdown rejects `airmureskarmory` | **a HANG** (killed a 120-game run at game 8) | any team file handed to a third-party poke-env is **nickname-free**; the pool export strips them |
| **H3** | **Hidden Power IVs**. Our pool pastes omit the `IVs:` line and `Gen3Teambuilder` patches it in from `GEN3_HP_IVS` at pack time. Foul Play packs the paste verbatim, so `Hidden Power [Grass]` with no IV line becomes Hidden Power **Dark** | a DIFFERENT team, silently | the export WRITES the IV line into the paste; **our pastes are only complete under our own teambuilder** |
| **H4** | **a trailing blank line**. Our vendored fork used to skip a split chunk only when it was exactly `""`, so a file ending `"\n\n\n"` (every Metamon `competitive` file) became an empty 7th Pokemon. `validate_teams_locally` passes it — the paste is fine, the PACK is not | **a HANG** (killed a pilot at game 2) | **FIXED in the fork** (`if not ps_mon.strip(): continue`), plus a **throwing guard on the pack** in `Gen3Teambuilder` (>6 mons raises), plus `.strip()` in the tool. Pinned by `src/poke_env_teambuilder_gate_test.py` |
| **H5** | **Metamon's recursion on a forfeit desync**. When our side forfeits at turn 250 the battle ends mid-step; Metamon's long-tail handler catches "Battle is already finished", force-resets (sending a fresh challenge) and then **calls itself** — ~985 levels, then `RecursionError`, killing the process | the peer dies and the harness reports **nothing** | the **watchdog**: `peer_exited` with the rc and the log tail, as a named failure with the games that did finish attached. 4 forfeits in 800 games (0.5%) |
| **H6** | **Metamon's hardcoded `:8000`**. `PokeEnvWrapper.server_configuration` returns poke-env's module-level `LocalhostServerConfiguration` = `ws://localhost:8000` — our shared dev server, **one port from the live training run**. No flag, no env var, no constructor argument | it binds the wrong server | the driver rebinds the module global before any env is built, and refuses 8000/8001 in its own code |
| **H7** | **no temperature can express Metamon's greedy**. `MetamonDiscrete` clips probabilities to [0.001, 0.99] *after* the temperature division, so the argmax tops out at ~0.992 however cold | a random non-argmax move ~1 in 125, silently | greedy is `Agent.get_actions(sample=False)` and **nothing else**; verified by `argmax_match_rate == 1.0000` |
| **H8** | **Foul Play's budget is WALL CLOCK**, with no iteration/visit/depth budget anywhere | the opponent is a **width meter**, not a setting | the realized visit count is scraped from its own `Iterations N: <visits>` lines and recorded per cell; a log with none reads UNVERIFIED |
| **H9** | **unflushed prints** (Metamon) | the readiness banner never lands and the peer looks dead | `PYTHONUNBUFFERED=1` in the peer env |
| **H10** | **a lingering websocket between halves**. Our half-1 client still holds the name when half 2 logs in; Showdown answers `|nametaken|` and the fork **continues as a guest**, so the peer's `/challenge` is addressed to a user that no longer exists | **a HANG** until the peer's own timeout (~8 min) | the socket is closed between halves AND each half plays under its own username suffix |
| **H11** | **two `poke_env` packages**. Metamon subclasses UPSTREAM poke-env; our `BattleStreamClient` subclasses the vendored fork. One process resolves `import poke_env` to exactly one of them and the other half breaks **silently** | wrong behaviour, no error | the peers are SUBPROCESSES with `PYTHONPATH=""`; nothing in this repo imports Metamon |
| **H13** | **thread oversubscription**. Torch's default pool on B = 1 inference; 210% CPU per peer, ~20× slower than necessary on a shared box | a campaign that projects at 18 h instead of 1 h | `OMP_NUM_THREADS=1` (+ MKL/OpenBLAS/numexpr/torch) in the peer env and in `metamon_side.py` before torch |
| **H12** | **the Showdown pin**. Ours is `e0551883f`, 13 commits behind Metamon's bundled submodule | — | irrelevant while BOTH clients play on ours, which the tool guarantees; it becomes relevant the moment anyone compares against numbers a third party produced on its own server. The pin is recorded on every row |

**One more, and it is about US, not them:** ⚠️ a long campaign that imports the MAIN checkout is not
pinned, and `main` is a moving target on a multi-session box. The parallel Metamon de-risk landed
`--forfeit-turn-limit` mid-run on 2026-09-14 and an `AttributeError` killed a Foul Play session
silently after 40 games. Run a campaign **from its own worktree**, or pin the commit it imports.

---

## 4. The standing numbers, to date

### The matched-regime 2×2 — `ai_v12_02_winprob_critic` @ 75,005,952 steps

800 games on our pinned Showdown (`e0551883f`), Metamon @ `0a00a759`, CPU-only, 100 games per cell,
role balanced inside every cell, **both regimes verified on every decision on both sides**. Win
rate is OURS; ties count in the denominator and not the numerator. Full artifact:
`designs/research_state/measurements/metamon_matched_regime_2026-09-14/`.

**`SmallRL` (ckpt 40, 13.9M)**

| cell | n | W/L/T | win rate | Wilson 95% | mean turns |
|---|---:|---|---:|---|---:|
| greedy · home | 100 | 52/48/0 | 0.520 | [0.423, 0.615] | 36.3 |
| **greedy · away (PRIMARY)** | 100 | 65/35/0 | **0.650** | **[0.553, 0.736]** | 48.8 |
| t1.0 · home | 100 | 63/37/0 | 0.630 | [0.532, 0.718] | 41.2 |
| t1.0 · away | 100 | 56/44/0 | 0.560 | [0.462, 0.653] | 42.3 |

**PRIMARY VERDICT: BETTER** — the lower bound 0.553 clears 0.50. Pooled 236/400 = 0.590.

**`SyntheticRLV2` (ckpt 48, 200.9M)**

| cell | n | W/L/T | win rate | Wilson 95% | mean turns |
|---|---:|---|---:|---|---:|
| greedy · home | 100 | 50/49/1 | 0.500 | [0.404, 0.596] | 40.4 |
| **greedy · away (PRIMARY)** | 100 | 42/58/0 | **0.420** | **[0.328, 0.518]** | 47.9 |
| t1.0 · home | 100 | 41/59/0 | 0.410 | [0.319, 0.508] | 42.7 |
| t1.0 · away | 100 | 31/69/0 | 0.310 | [0.228, 0.406] | 44.8 |

**PRIMARY VERDICT: NOT DETECTED** — lower bound 0.328, and the point estimate is below even.
Pooled 164/400 = 0.410. `SyntheticRLV2` leads us in three of four cells and is level in the fourth.

**The three effects, each with its own Newcombe CI** (an effect whose CI covers zero is NOT
DETECTED, never "equal"):

| effect | `SmallRL` | `SyntheticRLV2` |
|---|---|---|
| temperature (greedy − t1), pooled | **−0.010 [−0.105, +0.086]** NOT DETECTED | **+0.100 [+0.004, +0.194]** DETECTED |
| team set (home − away), pooled | **−0.030 [−0.125, +0.066]** NOT DETECTED | **+0.090 [−0.006, +0.184]** NOT DETECTED |
| role (we challenge − they challenge) | +0.050 [−0.046, +0.145] | −0.020 [−0.115, +0.076] |

Two results worth carrying forward: our own **+8.9 pp** eval-regime figure is an ASYMMETRY term and
does **not** transfer to a symmetric regime change; and our 719-team training pool buys **no
detectable home advantage** over a 20-team set we have never trained on — a free-standing
corroboration of *count dominates conditioning*.

### Foul Play — same checkpoint

| | |
|---|---|
| our win rate @ `--search-time-ms 1000`, `--search-parallelism 1` | **0.388 (31/80)**, Wilson 95% **[0.288, 0.497]** |
| **realized search width** | **1.40 M** MCTS visits/decision; session means **1.21–1.53 M** at a CONSTANT nominal budget |
| turns | mean 34.2, median 28, max 110; 0/80 hit the 250-turn forfeit |
| protocol | 0 parse failures either way over 3,040 decisions; the two clients agreed on the winner 80/80 |

**The external bot is AHEAD of the 75M win-prob arm at that budget, and the interval excludes
parity.** ⚠️ The team distribution in that campaign was ASYMMETRIC (we pinned one pool team per
10-game session, Foul Play redrew from all 72); `main.anchors` draws both sides from the same set
and flags any residual asymmetry, so a rerun through the tool is not directly comparable to the
0.388.

### 🚨 Numbers that must NOT be quoted without their regime

`0.742` (vs `SmallRL`) and `0.583` (vs `SyntheticRLV2`) are **mixed-regime reads on home teams** —
us greedy, them at T = 1.0 — from the de-risk. Matched, the same cells read **0.520** and **0.500**.
The honest summary of that checkpoint against Metamon is the 2×2 above.

---

## 5. Reading a result

`summary.json` carries `status`, the cell (regime, team set, opponent version and commit, our
checkpoint and how it resolved, the search budget), the Wilson interval, the per-role split, the
integrity counters (`hit_forfeit_limit`, `n_defaults`, `n_redecides`, `distinct_our_teams`), the
realized visit count and `their_argmax_match_rates`. `python -m main.anchors` prints the same block
at the end, regime first.

**Before believing a number, check three things:**

1. `status == "OK"`. A `FAILED` read still carries its games — that is deliberate, so a partial n is
   never mistaken for a full one — but it is not a measurement. The named causes are
   `peer_never_ready`, `peer_exited`, `no_first_game`, `no_progress`, `short_series`,
   `our_side_error`.
2. `their_argmax_match_rates == [1.0]` in a greedy cell, and materially **below** 1.0 in a t1 cell.
   The sampling cell is the POSITIVE CONTROL: an instrument reading 1.000 in both regimes would
   have no power.
3. `team_source_asymmetry == false`, and for Foul Play a non-null
   `realized_visits_per_decision_mean`.

**Then write it down with its regime.** A row in `designs/research_state/ledger.md` names the
opponent, its version and commit, the regime, the team set, our checkpoint's step, n, the win rate
and its Wilson interval — and for a search opponent the realized visit count.

---

## 6. What an anchor read cannot say

* **Nothing about Metamon on Metamon's own Showdown pin.** Both clients play on ours.
* **Nothing about exploitation.** Neither side adapts, so nothing here is evidence that greedy is
  safe against an adapting opponent.
* **Nothing about which model is stronger in general.** 100 games resolves ~±10 pp at best; every
  cell near even stays NOT DETECTED however the point estimate reads.
* **Nothing about the open ladder.** Both clients speak to a PINNED local server, and the two
  clients have OPPOSITE drift postures — ours raises on an unknown protocol keyword by design,
  Foul Play silently ignores it. A green anchor read says nothing about either on the public
  server; `python src/main/ladder_drift_scan.py` remains the instrument for that, and
  `designs/research_state/ladder_readiness.md` the audit.
