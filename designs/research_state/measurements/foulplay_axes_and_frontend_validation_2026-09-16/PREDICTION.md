# PRE-REGISTRATION — the two Foul Play axes, and the websocket front end as an anchors transport

**Written and committed 2026-09-17 (local clock; the campaign is named for the 2026-09-16 session),
BEFORE any of the four Foul Play cells or any of the four front-end cells was run.** Rule 1 of
UNDERSTANDING §7: both branches, the bar, and the comparator, stated before the numbers exist.

Two jobs, run SEQUENTIALLY — job 1 alone on the box (apart from the live training arm), because
Foul Play's budget is wall clock and concurrent CPU load is the thing that moves it
(UNDERSTANDING rule 23).

---

## 0. What is already known, and therefore is NOT a prediction

Honesty about leakage, since plumbing pilots ran before this file was written:

* Seven pilots of 2–8 games each were run at `--search-time-ms 100` and `300` to debug the harness
  (§4). Their win rates (1/1, 0/1, 1/3, 1/2, and a part-finished 8-game pilot) are **n ≤ 4 and are
  not evidence about anything**; none is used as a comparator and none informed the numbers below.
* The PRIOR points, all from the ledger, and all with the SAME checkpoint only for the first:
  | campaign | our win rate | realized FP visits/decision |
  |---|---|---:|
  | de-risk 2026-09-14, `ai_v12_02_winprob_critic`@75M (THIS checkpoint) | 0.388 [0.288, 0.497] | 1.40 M |
  | arm S 2026-09-14 (a DIFFERENT checkpoint) | 0.450 [0.346, 0.559] | 1.249 M |
  | arm W 2026-09-16 (a DIFFERENT checkpoint) | 0.475 [0.369, 0.583] | 1.153 M |

  Hazard H-E of `flywheel_pair_read_2026-09-15` records that those three order **exactly inversely
  to realized width**, and that no difference among them may be attributed to a model. Only the
  first is our checkpoint, so **the three prior points are CONTEXT, not a curve** — the curve this
  campaign fits is within one checkpoint and one box-week.
* A 100 ms pilot measured Foul Play's realized width on this box at a **median 50,000 and mean
  64,862** visits/decision (812 decisions). That fixes the rough scale of the width axis
  (~1 order of magnitude below the 1000 ms reference) and is the reason the cells are spaced
  100 / 300 / 1000 ms.

---

## 1. The instrument, and the deliberate deviations from the anchors SOP

`python -m main.anchors --model models/ai_v12_02_winprob_critic/final_model.zip --opponent foulplay
--regime greedy --teamset home --games 80`, from the MAIN checkout, CPU only
(`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `nice -n 10`), ports 9700–9799, at the trainer's
250-turn forfeit limit, realized visits recorded per cell, `uptime` recorded at cell start and end.

Four deviations from `designs/ops/EXTERNAL_ANCHORS_SOP.md`, each registered here rather than
discovered later:

| # | deviation | why |
|---|---|---|
| D1 | `our_team_sources.home` is the **72-team `data/teams/sample` directory**, not the 719-team pool | the SOP's own rule is that both sides draw from the SAME set; the peer draws from the 72-team export of exactly those teams, so the stock config trips `team_source_asymmetry` (719 vs 72). It also answers the de-risk's own hazard H5 ("a graded rerun should either pin both sides or let both redraw") — here BOTH sides redraw |
| D2 | the port range is **9700–9799** | this session's assigned range |
| D3 | the Foul Play checkout is a **COPY** at `/home/goodlad/dev/foul-play-axes` (standard knowledge) or `/home/goodlad/dev/foul-play-noknow` (degraded), each carrying the SAME three plumbing patches (§4) | the standing checkout `/home/goodlad/dev/foul-play` is the pinned anchor opponent and is left byte-identical |
| D4 | the knowledge cell's degradation is by **emptying on-disk caches**, not by a flag | Foul Play exposes no set-prediction switch; §3 |

**Nothing under `src/` is changed by this campaign, and nothing under `models/` is written.**

---

## 2. Job 1 — the cells

| cell | `--search-time-ms` | knowledge | games | purpose |
|---|---:|---|---:|---|
| `w100` | 100 | standard | 80 | width |
| `w300` | 300 | standard | 80 | width |
| `w1000` | 1000 | standard | 80 | width AND the knowledge comparator |
| `k1000` | 1000 | **degraded** | 80 | knowledge |
| `w3000` | 3000 | standard | 80 | OPTIONAL, only if the first four finish with time to spare |

All four run **one at a time, in this order, in one window**, with the same `--team-seed 916001` on
our side so our team-draw sequence is identical across cells. `w1000` and `k1000` are adjacent in
time; any residual width difference between them is MEASURED (realized visits) and corrected for by
the width slope, rather than assumed away.

---

## 3. What "knowledge OFF" means here, exactly

Foul Play has **no CLI or config key that disables set prediction**. The keys that exist are
`--smogon-stats-format` (`fp/config.py`, "Overwrite which smogon stats are used to infer unknowns")
and nothing else; there is no `--no-set-prediction`. The prediction stack is three files, all
fetched once and then read from an on-disk cache by `fp.data.sets.base.get_sets_file`
(`base.py:30-51` — **a cache file that exists is returned verbatim, and the network is not
touched**):

| tier | source | cache file under `fp/data/pkmn_sets_cache/gen3ou/` | consumer |
|---|---|---|---|
| T3 | `https://data.foulplay.cc/gen3ou/pokemon_full_sets.json` (replay-derived full sets) | `pokemon_full_sets.json` | `TeamDatasets` |
| T3 | `https://data.foulplay.cc/gen3ou/replay_moves.json` (replay-derived movesets) | `replay_moves.json` | `TeamDatasets` |
| T2 | `https://play.pokemonshowdown.com/data/sets/gen3ou.json` (Showdown's published sets) | `showdown_sets.json` | `TeamDatasets` |
| T1 | `https://www.smogon.com/stats/<prev month>/chaos/gen3ou-0.json` (Smogon usage, unweighted) | `fp/data/smogon_stats_cache/gen3ou-0.json` | `SmogonSets` |

**The `k1000` cell empties T3 + T3 + T2** — the three `pkmn_sets_cache/gen3ou/*.json` files are each
the literal `{}` — so `TeamDatasets.initialize` loads nothing, `get_all_remaining_sets` and
`get_pkmn_sets_from_pkmn_name` return nothing, and `_sample_pokemon`
(`fp/search/standard_battles.py`) falls through steps 1 and 2 to step 3, **SmogonSets only**. T1
survives.

🚨 **"No set prediction at all" is NOT REACHABLE and that is a finding, not a choice.** With T1 also
empty, `predict_team_likelihood` gets an empty `all_pkmn_counts` and
`sample_standardbattle_pokemon` calls `random.choices([], weights=[])`, which raises — gen3ou has
no team preview, so Foul Play MUST invent the opponent's unrevealed Pokemon and there is no
"engine defaults" path to fall back on. The SOP's fallback ("else the weakest single source") is
therefore what is run, and the axis this campaign measures is **the replay-derived and published-set
tiers (T2+T3), on top of a Smogon-usage floor** — not the whole of Foul Play's opponent knowledge.
The residual T1 knowledge biases the knowledge effect **towards zero**.

---

## 4. The three plumbing patches, applied IDENTICALLY to both copies

All three are network/plumbing. None touches `fp/search/`, `fp/data/`, the evaluator, or any
decision. Each was forced by a failure REPRODUCED in a pilot before this file was written; the
exact diff is `patches/foulplay_plumbing.patch`.

**P1 — a `/challenge` PM that arrives while Foul Play is busy is remembered instead of dropped**
(`fp/websocket_client.py`). poke-env's `_send_challenges` PIPELINES: `_battle_semaphore` is released
when a battle is CREATED (`src/poke_env/player/player.py:363`), so our client sends the next
`/challenge` while battle *n* is still running. Foul Play is then inside `pokemon_battle`'s loop (or
inside `leave_battle`'s drain, which discards outright), the PM is handed to the battle parser and
ignored, and the server never sends it again — so `accept_challenge` blocks forever.
**Measured: the `ours_challenge` half of a `main.anchors --opponent foulplay` cell died at game 2,
three times out of three** (`no_progress`, 1 game recorded). The patch records any 9-field
`/challenge` PM addressed to us and has `accept_challenge` consume one before blocking.

**P3 — the opponent's player id is derived LINE-wise, by an exact name match**
(`fp/modes/base.py::start_battle_common`). Unpatched it tests `battle.opponent.account_name in msg`
over the whole websocket FRAME and then takes `msg.split("|")[2]` — **the first player id in the
frame**. The server batches both `|player|` lines into one frame under load, so Foul Play always
concludes the opponent is `p1`: right by luck when it is the ACCEPTOR (it is p2), **wrong when it is
the CHALLENGER**, after which `battle.opponent` is itself, `battle.opponent.active` is never set, and
`search_params` dies on turn 1 with `AttributeError: 'NoneType' object has no attribute 'moves'`.
A debug dump printed `user='p1' opponent='p1'`. The same substring test also misfires when the peer
username CONTAINS ours (`FpFoo` vs `Foo`) — which is why this campaign's two usernames share no
substring, belt and braces.

**P2 — wait for the opponent's lead before the first search**
(`fp/modes/standard_battle.py::start_battle`). Only the frame that carried `|start` is kept as the
held message list, and the server sometimes ends a frame AT `|start`; the opponent's `|switch|` then
arrives later and the first search dies the same way. The patch reads on through Foul Play's own
`async_update_battle` until `battle.opponent.active` is set.

P1 and P3 are both candidates for a tech-debt row against `main.anchors`, since the tool's Foul Play
adapter cannot currently run a multi-game `ours_challenge` half without them.

---

## 5. Registered analysis

1. **Per cell**: our win rate with a Wilson 95 % interval; W/L/T; realized visits/decision (mean,
   median, min, max, and the per-half means); mean/median/max turns; forfeits; box load at start
   and end; `team_source_asymmetry`; `status`. A cell whose `status != "OK"` is **not a
   measurement** and is reported as a partial n, never folded into a rate.
2. **The width slope** is an unweighted least-squares fit of win rate on `ln(realized visits per
   decision)` over the three standard-knowledge cells, with a cluster bootstrap over GAMES for its
   interval. It is fitted BEFORE the knowledge cell is read (rule 1).
3. **The width contrast** is `w100 − w1000` with a Newcombe 95 % interval.
4. **The knowledge contrast** is `k1000 − w1000` with a Newcombe 95 % interval, and then the
   **width-corrected residual**
   `residual = (k1000 − w1000) − slope × (ln V_k1000 − ln V_w1000)`,
   whose interval is the Newcombe interval shifted by the (fixed) correction. Paired game indices
   are reported where our team-draw sequence makes two cells' game *i* share our team.

### The bars, and both branches

| question | DETECTED if | NOT DETECTED if |
|---|---|---|
| width matters | the `w100 − w1000` Newcombe 95 % CI **excludes 0** | it covers 0 |
| knowledge matters at matched width | the width-corrected residual's 95 % CI **excludes 0** | it covers 0 |

**Verdict rule, registered:**

* **"the Foul Play edge is mostly SEARCH"** — the width contrast is DETECTED and its magnitude
  exceeds the knowledge residual's magnitude.
* **"mostly KNOWLEDGE"** — the knowledge residual is DETECTED and its magnitude exceeds the width
  contrast's.
* **"not separable at this n"** — neither CI excludes zero, or both do and the two magnitudes'
  difference is inside the wider of the two CIs.

A CI that straddles zero is NOT DETECTED, never "equal" (rule 6). No result here is a claim about
Foul Play in general: n = 80 per cell resolves ~±11 pp at best, one checkpoint, one box.

### The point predictions (so the result can embarrass them)

| quantity | prediction |
|---|---|
| `w100` win rate | **0.55** |
| `w300` win rate | **0.48** |
| `w1000` win rate | **0.41** (the de-risk read 0.388 at a ~1.4 M width) |
| width slope | **≈ −0.06 win rate per natural-log unit of visits** (≈ −0.04 per doubling) |
| `k1000` win rate | **0.51**, i.e. a knowledge effect of **+0.10** |
| **the standing PRIOR this reads against** | the ledger's own words (2026-09-14): *"Foul Play's edge may be its opponent-set knowledge as much as its search — the two axes are separable and unread."* Symmetry between the two axes is the prior; the numbers above put ~60 % of a 1000 ms→100 ms swing on search |
| the registered verdict I expect | **"mostly SEARCH"**, at maybe 2:1 odds over "not separable at this n" |

---

## 6. Job 2 — the front-end milestone validation

One SOP **tier-B** read, twice, in the same window, same `--team-seed`:

```
# through the websocket FRONT END (python -m utils.bridge.ws_frontend --port 97NN)
python -m main.anchors --model <ckpt> --opponent metamon:SmallRL --regime greedy \
    --teamset {home,away} --games 100 --server-uri ws://localhost:97NN/showdown/websocket
# through the NODE path (main.anchors starts its own deps/pokemon-showdown server)
python -m main.anchors --model <ckpt> --opponent metamon:SmallRL --regime greedy \
    --teamset {home,away} --games 100
```

Four cells: {front end, Node} × {home, away}, 100 games each.

**Bar, registered:** for each team set, the (front end − Node) win-rate difference has a Newcombe
95 % interval that **covers zero**, AND the front-end path records **zero protocol failures**
(`status == "OK"`, `their_argmax_match_rates == [1.0]`, no `UnknownMessageType` on our side, no
Metamon exception attributable to the transport). Throughput (games/second wall clock) is reported
for both paths but is **not** a bar — the ws_frontend note's own headline is that a shim buys
determinism, not throughput.

**Verdict:**

* **"front-end may be the DEFAULT transport for the anchors CLI"** — both team sets agree inside
  the delta's own CI and there is no protocol failure.
* otherwise, **the named blocker**, with the failing cell and its cause.

**Prediction:** agreement on both team sets (|Δ| ≤ 0.10, CI covering zero), zero protocol failures,
and the front end at or above the Node path's throughput at this concurrency (the ws_frontend note
measured 3.09× at concurrency 1). Metamon's `home` cell is the one with residual risk, because our
72/719-team export is where the nickname and Hidden-Power-IV hazards live.

**What this cannot say:** nothing about Foul Play over the front end (not run here), nothing about
concurrency > 1, and nothing about a third-party client we have not tried.
