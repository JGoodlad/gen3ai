# The two Foul Play axes, and the websocket front end as an anchors transport

**Measured 2026-09-17 (registered 2026-09-16 session), on `ai_v12_02_winprob_critic/final_model.zip`
@ 75,005,952 steps.** Two jobs, run sequentially on one box, with a live training arm
(`ai_v13_04_flywheel_winprob_b`) holding the GPU throughout. CPU only, `nice -n 10`,
`OMP_NUM_THREADS=1`, ports 9700–9799. Nothing under `models/` was written, nothing under `src/` was
changed, and `/home/goodlad/dev/foul-play` — the pinned anchor checkout — is byte-identical to how
it was found.

**The pre-registration is [`PREDICTION.md`](PREDICTION.md), committed as `ea296538` before any cell
ran.** Read it first: it holds the bars, both branches, the deviations from
`designs/ops/EXTERNAL_ANCHORS_SOP.md`, and the point predictions this note goes on to embarrass.

---

## 0. The two verdicts, up front

> ### JOB 1 — **"not separable at this n"**
> Neither axis moves our win rate detectably. Realized search width was manipulated over a **9.4×
> range** (83 k → 780 k MCTS visits/decision) and the win-rate contrast is **−0.062 [−0.210, +0.089]
> — NOT DETECTED**, with a fitted slope of **+0.027 [−0.045, +0.095] win rate per ln-unit of
> visits**, i.e. the WRONG SIGN and covering zero. Foul Play's set-prediction knowledge was degraded
> to a Smogon-usage floor and the contrast at matched width is **+0.000 [−0.151, +0.151] — NOT
> DETECTED**, width-corrected residual **+0.001 [−0.150, +0.152]**.
>
> ### JOB 2 — **"front-end MAY be the DEFAULT transport for the anchors CLI"**
> Both team sets agree inside the delta's own CI (home **+0.050 [−0.084, +0.182]**, away **−0.030
> [−0.165, +0.106]**, pooled **+0.010 [−0.086, +0.106]**), **zero protocol failures** on either
> path over 200 front-end battles, `their_argmax_match_rates == [1.0]` in all four cells, and
> throughput within 10 % (0.575 vs 0.629 games/s home; 0.565 vs 0.549 away). The one peer death was
> Metamon's own known forfeit-desync recursion (SOP hazard H5), after its half's last game.

🚨 **What "not separable" does NOT mean.** It is not "the two axes are equal", and it is not
"search does not matter" (rule 6). It means that at n = 80 per cell, on this checkpoint, this box
and this team pool, **neither manipulation produced a win-rate change this instrument can see** —
and the more interesting half of that is that the SEARCH manipulation is the one that should have
been easy.

---

## 1. Provenance

| thing | value |
|---|---|
| our checkpoint | `models/ai_v12_02_winprob_critic/final_model.zip` @ **75,005,952** steps, resolved `explicit_zip`, loaded `bare` |
| gen3ai commit every cell ran at | **`ea296538`** (checked at every cell start; it did not move — the campaign's own pre-registration commit) |
| `deps/pokemon-showdown` pin | `e0551883f` |
| Foul Play | `/home/goodlad/dev/foul-play` @ **`6c467c08`**, copied twice (§3); poke-engine `0.0.48` `--features poke-engine/gen3`; its own conda env at `/home/goodlad/dev/foul-play/.conda` |
| Metamon | `/home/goodlad/dev/metamon` @ `0a00a759`, `SmallRL` ckpt 40 |
| our client | `main.play`'s own path, `--device cpu --temperature 0.0`, forfeit limit **250 turns** (the trainer's constant) |
| teams | job 1: BOTH sides draw from the **same 72-team** `data/teams/sample` pool (`team_source_asymmetry == false` in all five cells). Job 2: the stock `designs/ops/anchors.json` — `home` = 719/719, `away` = 20/20 |
| box | 16 cores, load 20–39 throughout, recorded per cell (§2, §5) |

---

## 2. JOB 1 — the cells

Every cell: `python -m main.anchors --opponent foulplay --regime greedy --teamset home --games 80
--search-parallelism 1 --team-seed 916001`, one at a time, in the order listed.

| cell | budget | knowledge | status | n | W/L/T | **win rate** | Wilson 95 % | realized visits/dec (mean) | (median) | turns mean/med/max | forfeits | wall |
|---|---:|---|---|---:|---|---:|---|---:|---:|---|---:|---:|
| `w100` | 100 ms | standard | OK | 80 | 31/49/0 | **0.388** | [0.288, 0.497] | 83,031 | 56,000 | 47.5/33/222 | 0 | 17 min |
| `w300` | 300 ms | standard | OK | 80 | 23/57/0 | **0.287** | [0.200, 0.395] | 259,513 | 150,000 | 38.1/33/119 | 0 | 37 min |
| `w1000` | 1000 ms | standard | OK | 80 | 36/44/0 | **0.450** | [0.346, 0.559] | 779,604 | 445,000 | 42.9/33/250 | 1 | 125 min |
| `k1000` | 1000 ms | **degraded** | OK | 80 | 36/44/0 | **0.450** | [0.346, 0.559] | 747,189 | 447,000 | 42.8/35/250 | 1 | 127 min |
| `k100` † | 100 ms | **degraded** | OK | 80 | 35/44/1 | **0.438** | [0.334, 0.547] | 92,005 | 61,000 | 40.5/32/165 | 0 | 15 min |

† `k100` is **EXPLORATORY and was not pre-registered** — see §2.4.

Box load at each cell's start / end: `w100` 39.4 → 31.7 · `w300` 31.7 → 20.0 · `w1000` 20.0 → 21.8 ·
`k1000` 24.6 → 24.3 · `k100` 24.0 → 21.8. The two 1000 ms cells ran back to back in one window and
their realized widths differ by **4.3 %** (747 k vs 780 k), the smallest gap in the campaign.

**The manipulation check, from Foul Play's own per-decision `source=` lines:**

| cell | `teamdatasets-full` | `teamdatasets-partial` | `smogonsets` |
|---|---:|---:|---:|
| `w100` / `w300` / `w1000` (standard) | 23,849 / 21,113 / 22,977 | 3,456 / 2,785 / 3,049 | 584 / 366 / 403 |
| `k1000`, `k100` (degraded) | **0** | **0** | **100 %** |

So the degradation is total for the tiers it targets: in the standard cells ~87 % of sampled
opponent sets come from Foul Play's replay-derived `TeamDatasets`, and in the degraded cells that
number is zero and every set is drawn from the Smogon usage marginals instead.

### 2.1 The width curve

```
win rate
  0.50 |                                      w1000 ●  0.450
       |
  0.45 |
       |
  0.40 |   w100 ●  0.388
       |
  0.35 |
       |
  0.30 |                    w300 ●  0.287
       |
       +------|--------------|-----------------|------------- ln(realized visits/decision)
           11.33          12.47             13.57
           (83 k)         (260 k)           (780 k)
```

| fit | value |
|---|---|
| OLS slope on `ln(visits)` | **+0.0272 win rate per ln-unit**, game-bootstrap 95 % **[−0.0450, +0.0945]** |
| the same, per DOUBLING of width | **+0.019** |
| **registered contrast `w100 − w1000`** | **−0.062**, Newcombe 95 % **[−0.210, +0.089]** → **NOT DETECTED** |
| paired on our team draw (80 matched slots) | −0.062, discordant 16/21 |
| registered prediction | −0.14, slope ≈ −0.06 per ln-unit |

**The prediction is refuted in magnitude and in sign.** The middle point sits 0.13 BELOW the line
through the outer two; the three Wilson intervals overlap in a band from 0.35 to 0.40, so the
honest summary of the width axis is a flat line at ≈ 0.38 with one low draw at 300 ms.

**Read against the standing prior.** Hazard H-E of `flywheel_pair_read_2026-09-15` records three
campaigns whose win rates ordered **exactly inversely to realized width** (1.40 M → 0.388, 1.249 M →
0.450, 1.153 M → 0.475) and warns that no difference among them may be attributed to a model. That
warning stands — but this campaign says the inverse ordering is **not a width law**. Those three
points span a **1.21×** width range and come from three DIFFERENT checkpoints; these five span
**9.4×** on ONE checkpoint and show no monotone relation at all. The de-risk's own 0.388 at 1.40 M
sits comfortably inside `w1000`'s interval [0.346, 0.559], so nothing here contradicts it either.
**The right reading of H-E is that a ±12 % width wobble is far too small to be the cause of anything
it was invoked for** — and that at n = 80 the cell-to-cell noise (0.287 to 0.450 with knowledge and
budget held fixed within a factor of 3) dwarfs it.

### 2.2 The knowledge contrast at matched width

| quantity | value |
|---|---|
| `k1000` − `w1000` | **+0.000**, Newcombe 95 % **[−0.151, +0.151]** → **NOT DETECTED** |
| realized widths | 747,189 vs 779,604 (ln 13.524 vs 13.567) |
| what the width slope predicts from that gap alone | **−0.0012** |
| **width-corrected residual** | **+0.001**, 95 % **[−0.150, +0.152]** → **NOT DETECTED** |
| paired on our team draw (80 matched slots) | +0.000; discordant 15/15 — a perfect split |
| registered prediction | +0.10 |

36/80 and 36/80. The two cells are not merely statistically indistinguishable, they are **identical
to the game**, with the discordant pairs splitting 15/15. Deleting the replay-derived and
published-set tiers of Foul Play's opponent model — 100 % of its set samples move to the Smogon
usage marginals — changed nothing this instrument can see.

### 2.3 The registered verdict

| bar | result |
|---|---|
| width matters (CI on `w100 − w1000` excludes 0) | **NO** — [−0.210, +0.089] |
| knowledge matters at matched width (corrected residual excludes 0) | **NO** — [−0.150, +0.152] |

Neither CI excludes zero ⇒ by the registered rule, **"the Foul Play edge is NOT SEPARABLE at this
n"**. The registered expectation ("mostly SEARCH", at ~2:1) is wrong.

### 2.4 The exploratory follow-up, and why it is in this note

A 12-game plumbing smoke of the degraded peer at 150 ms read **10/12 = 0.833**. That looked like a
width × knowledge interaction — knowledge mattering where the search is too shallow to recover it —
so a fifth cell (`k100`, 80 games, degraded, 100 ms, the same team seed as `w100`) was run to test
it. **It did not replicate:**

| contrast | value | verdict |
|---|---|---|
| `k100` − `w100` | **+0.050**, Newcombe 95 % [−0.101, +0.197] | NOT DETECTED |
| pooled over both matched widths (degraded 71/160 = 0.444 vs standard 67/160 = 0.419) | **+0.025**, 95 % [−0.083, +0.132] | NOT DETECTED |

The 0.833 was a 12-game draw. This is banked as an instance of rule 21's shape — a post-hoc
candidate regressing to nothing on a pre-specified replicate — and as a reminder that a smoke is
plumbing evidence, never a measurement. **Every number in §2.4 is EXPLORATORY**: `k100` was chosen
after seeing the smoke, so its interval is not a registered one.

### 2.5 What this campaign therefore says about the ledger's open question

The ledger's 2026-09-14 entry left this open in these words: *"Foul Play's edge may be its
opponent-set knowledge as much as its search — the two axes are separable and unread."* They are now
read, and the answer is **neither axis is the edge, at the resolution 80 games buys**. Whatever makes
Foul Play a ~0.40–0.45 opponent for this checkpoint is not its wall-clock budget over a 9.4× range,
and not the replay-derived half of its opponent model. The remaining candidates — untested here —
are its hand-written evaluator, the decoupled-UCT search's SHAPE rather than its width, and the
Smogon-usage floor that could not be removed (§3.2).

---

## 3. What "knowledge degraded" is, exactly — and what could not be done

### 3.1 The manipulation

Foul Play has **no CLI or config key that disables set prediction**; `--smogon-stats-format`
(`fp/config.py`) only re-points the usage file. Its prediction stack is four cached files, and
`fp.data.sets.base.get_sets_file` (`base.py:30-51`) returns a cache file that exists **verbatim,
without touching the network**. The degraded copy's three `TeamDatasets` caches under
`fp/data/pkmn_sets_cache/gen3ou/` are each the literal `{}`:

| tier | source | cache file | state in `k*` |
|---|---|---|---|
| T3 | `https://data.foulplay.cc/gen3ou/pokemon_full_sets.json` | `pokemon_full_sets.json` | **emptied** |
| T3 | `https://data.foulplay.cc/gen3ou/replay_moves.json` | `replay_moves.json` | **emptied** |
| T2 | `https://play.pokemonshowdown.com/data/sets/gen3ou.json` | `showdown_sets.json` | **emptied** |
| T1 | `https://www.smogon.com/stats/<prev month>/chaos/gen3ou-0.json` | `fp/data/smogon_stats_cache/gen3ou-0.json` | kept |

`TeamDatasets.initialize` then loads nothing and `_sample_pokemon`
(`fp/search/standard_battles.py`) falls through its steps 1 and 2 to step 3, `SmogonSets`.

### 3.2 🚨 "No set prediction at all" is NOT REACHABLE — a finding, not a choice

With T1 empty as well, `predict_team_likelihood` gets an empty `all_pkmn_counts` and
`sample_standardbattle_pokemon` calls `random.choices([], weights=[])`, which raises. **gen3ou has
no team preview**, so Foul Play MUST invent the opponent's unrevealed Pokemon before it can search
at all, and there is no "engine defaults" path to fall back on — the `logger.warning("Could not
sample …")` branch leaves a Pokemon with no moves, which `search_params` then dereferences. The
SOP's named fallback ("else the weakest single source") is therefore what ran. **The axis measured
is T2 + T3 on top of a Smogon-usage floor, not the whole of Foul Play's opponent knowledge, and the
residual T1 biases the knowledge effect towards zero.**

### 3.3 The four plumbing patches

Applied **identically** to both copies; the diff is
[`patches/foulplay_plumbing.patch`](patches/foulplay_plumbing.patch). None touches `fp/search/`,
the evaluator, or any decision. Each was forced by a failure reproduced here.

| # | what | why it was needed |
|---|---|---|
| **P1** | `fp/websocket_client.py` — a `/challenge` PM addressed to us that arrives while the client is busy is REMEMBERED (`_pending_challenges`) and consumed by the next `accept_challenge` | poke-env's `_send_challenges` PIPELINES (`player.py:363` releases the semaphore when a battle is CREATED), so our client sends the next `/challenge` mid-battle; Foul Play hands it to the battle parser, which ignores it, and the server never resends. **The `ours_challenge` half died at game 2, three times out of three** |
| **P3** | `fp/modes/base.py::start_battle_common` — the opponent's player id is found LINE-wise, matching the name FIELD exactly | unpatched it tests `account_name in msg` over the whole FRAME and then takes `msg.split("|")[2]`, **the first player id in the frame**. The server batches both `|player|` lines into one frame under load, so Foul Play concludes the opponent is `p1`: right by luck as the ACCEPTOR, **wrong as the CHALLENGER**, after which `battle.opponent.active` is never set and `search_params` dies on turn 1. A debug dump read `user='p1' opponent='p1'`. The same substring test misfires whenever the peer name CONTAINS ours (`FpFoo` vs `Foo`) |
| **P2** | `fp/modes/standard_battle.py::start_battle` — read on through `async_update_battle` until `battle.opponent.active` is set, before the first search | only the frame carrying `\|start` is kept as the held message list, and the server sometimes ends a frame AT `\|start` |
| **P4** | `fp/data/sets/smogon.py::move_usage_rates` — tolerate the `[]` sentinel `get_pkmn_by_name_in_dict` returns (`base.py:507`) instead of calling `.get` on a list | a latent Foul Play bug, unreachable in the standard configuration and immediate in the degraded one: `AttributeError: 'list' object has no attribute 'get'` killed the first `k1000` attempt after 1 game |

⚠️ **P4 landed AFTER `w100`, `w300` and `w1000` had run.** Those three cells' peer therefore lacks
it. The patch changes behaviour only on the branch where unpatched Foul Play **crashes**, and it
crashed **0 times in 240 standard-knowledge games (~9,000 decisions)**, so the patched and unpatched
standard peers are behaviourally identical over the games taken; `k100` (run after P4) and `w100`
(run before) are compared in §2.4 on that argument. It is stated rather than buried because it is
the one place where the cells are not literally the same binary.

---

## 4. JOB 2 — the front end vs the Node server

One SOP **tier-B** read taken twice in one 12-minute window, same `--team-seed 916002`, the stock
`designs/ops/anchors.json` (no campaign overrides), `metamon:SmallRL` greedy, both team sets,
100 games each.

| cell | transport | team set | status | n | W/L/T | **win rate** | Wilson 95 % | their argmax | our forfeits | defaults/redecides | distinct our teams | wall s | games/s |
|---|---|---|---|---:|---|---:|---|---|---:|---|---:|---:|---:|
| `fehome` | **ws_frontend** (`--impl rust`) | home | OK | 100 | 63/35/2 | **0.630** | [0.532, 0.718] | `[1.0]` | 0 | 0/0 | 48 | 174 | 0.575 |
| `ndhome` | Node showdown | home | OK | 100 | 58/42/0 | **0.580** | [0.482, 0.672] | `[1.0]` | 0 | 0/0 | 48 | 159 | 0.629 |
| `feaway` | **ws_frontend** (`--impl rust`) | away | OK | 100 | 52/48/0 | **0.520** | [0.423, 0.615] | `[1.0]` | 2 | 0/0 | 17 | 177 | 0.565 |
| `ndaway` | Node showdown | away | OK | 100 | 55/44/1 | **0.550** | [0.452, 0.644] | `[1.0]` | 0 | 0/0 | 17 | 182 | 0.549 |

### The agreement table — the registered bar

| team set | front end | Node | **Δ (front end − Node)** | Newcombe 95 % | verdict |
|---|---:|---:|---:|---|---|
| home | 0.630 | 0.580 | **+0.050** | [−0.084, +0.182] | **AGREE** |
| away | 0.520 | 0.550 | **−0.030** | [−0.165, +0.106] | **AGREE** |
| pooled | 0.575 (115/200) | 0.565 (113/200) | **+0.010** | [−0.086, +0.106] | **AGREE** |

**Protocol:** `ws_frontend` served **200 battles with 0 ERROR, 0 WARNING and 0 exceptions** in its
own log; our `battle_event.classify` tripwire (`UnknownMessageType` / `UnsupportedMessageType`)
never fired on any of the four cells; `their_argmax_match_rates == [1.0]` everywhere, i.e. the
greedy regime was verified per decision on both sides through both transports;
`team_source_asymmetry == false` everywhere; `n_defaults` and `n_redecides` are 0 in all four cells.

**Throughput** is within 10 % either way at this concurrency (1 battle at a time): 0.575 vs 0.629
games/s on home, 0.565 vs 0.549 on away. That is consistent with the `ws_frontend.md` headline —
a shim buys determinism and the removal of a server process, not throughput — and it rules out the
failure that would matter, a transport that serialises what a server overlaps.

**Per-game comparability.** The two paths' game slots share our TEAM draw (100/100 in both team
sets, from the common `--team-seed`) but NOT the battle RNG and not Metamon's hidden state, so they
are matched-team draws, not replicates. Outcome agreement on those slots is **56/100 (home)** and
**58/100 (away)**, against ~0.52 expected for independent draws of processes this close — a little
shared signal from the team match, and nothing more. **A decision-level replicate would need
`--seed-base` on the front end and has no counterpart on the Node path**; the instrument for
byte-level equality is the existing `ws_frontend_byte_identity_integration_test.py`, not this read.

### The one peer death, and why it is not a blocker

`feaway`'s `ours_challenge` half ended with Metamon's **`RecursionError: maximum recursion depth
exceeded`** — SOP hazard **H5** / the flywheel note's **H-B** and **H-H**: Metamon's long-tail
handler answers a mid-battle forfeit by force-resetting and then calling itself. It fired **after
that half's last game** (its own report: `n_decisions` 2,885, `n_team_draws` 50, `argmax_match_rate`
1.0), the cell recorded all 100 games, and `status` is OK. It is triggered by OUR 250-turn forfeits
— `feaway` had 2, `ndaway` 0 — not by the transport. ⚠️ It is nevertheless the fourth appearance of
this failure in the **away** cell specifically, which is where H-H says it lives.

### The registered verdict

**"front-end MAY be the DEFAULT transport for the anchors CLI."** Both bars are met: every cell
agrees inside the delta's own CI, and there is no protocol failure. Two conditions that a promotion
should carry, neither of which is a blocker:

1. ⚠️ **This is a transport + SIMULATOR comparison, not a pure transport one.** `ws_frontend --impl
   rust` backs each battle with a `sim_bridge` child, so a front-end cell exercises the Rust port
   where a Node cell exercises the pinned Showdown server. Agreement here is evidence for the whole
   stack; the byte-differential gate remains the instrument for the protocol half.
2. ⚠️ **Nothing here tests Foul Play over the front end**, or any concurrency above 1. Job 1 ran
   entirely on the Node path.

---

## 5. Hazards — each one is a finding

| # | hazard | evidence | what it means |
|---|---|---|---|
| **H-1** | 🚨 **`main.anchors --opponent foulplay` cannot run a multi-game `ours_challenge` half at all** without P1. | `no_progress`, 1 game recorded, **3 times out of 3**; diagnosed to poke-env's pipelined `send_challenges` against a peer that drops PMs received mid-battle. | The tool's Foul Play adapter is only proven at 1 game per half. Every Foul Play number in the ledger so far came from the de-risk's own scripts, where Foul Play was always the CHALLENGER — the direction the bug spares. **Backlog candidate.** |
| **H-2** | 🚨 **Foul Play derives its player slot from the FIRST `\|player\|` line in a websocket frame.** As the CHALLENGER under load it concludes it is playing itself. | debug dump `user='p1' opponent='p1'`, then `AttributeError: 'NoneType' object has no attribute 'moves'` on turn 1. | Load-dependent, so it passes a quiet-box pilot and fails a real campaign. The same substring test also misfires when the peer username CONTAINS ours — **never name the two sides `Foo` and `FpFoo`**. |
| **H-3** | 🚨 **Degrading the opponent's knowledge walks into code paths the standard configuration never reaches.** | `move_usage_rates` crashed immediately in `k1000` and never once in 240 standard games. | An ablation of a third-party component is a *new* configuration of it, and needs its own smoke before an 80-game commitment. |
| **H-4** | 🚨 **Foul Play remains a WIDTH METER (rule 23), but the meter is much coarser than assumed.** Realized width at a constant nominal 1000 ms was 747 k–780 k here against the de-risk's 1.40 M on the same checkpoint — a **1.9× gap between campaigns**, far outside the ±11 % the de-risk measured. | §2, and the de-risk's own table. | Rule 23 still binds — the realized count must accompany every read. But the campaign-to-campaign width gap is now known to be ~2×, and **this campaign shows a 9.4× width change moving the win rate by −0.062 [−0.210, +0.089]**, so width is a weak lever on the OUTCOME even when it is a strong lever on the OPPONENT's compute. |
| **H-5** | ⚠️ **The 300 ms cell is 0.10 below both its neighbours** with every input held fixed but the budget. | `w300` 0.287 [0.200, 0.395]. | Either cell-level noise well above the binomial (the campaign's own evidence for run-to-run variance at n = 80, cf. rule 19), or a real non-monotonicity. **Nothing here distinguishes them**, and a third 300 ms draw is what would. |
| **H-6** | ⚠️ **The 12-game smoke read 0.833 and the 80-game cell read 0.438.** | §2.4. | A plumbing smoke is not a measurement, and this campaign nearly reported a width × knowledge interaction off one. Rule 21's shape. |
| **H-7** | ⚠️ **Metamon's forfeit recursion fired again, in the AWAY cell, for the fourth time.** | §4. | A failure that recurs in one configuration across four campaigns is a property of that configuration. It cost nothing here (it fired after the half's last game) but it is not random. |
| **H-8** | ⚠️ **The peer is a PATCHED COPY**, and P4 landed after three of the five cells. | §3.3. | Stated in full there. The standard cells never reached P4's branch (0/240 games). |
| **H-9** | ⚠️ **Our side's team draw is matched across cells; Foul Play's is not.** | `--team-seed 916001` seeds our `Gen3Teambuilder`; Foul Play's `load_team.py` randomises with its own unseeded RNG. | The paired rows in §2 are matched on OUR team only. That halves the matchup variance the pairing removes, and is why the paired Δ and the unpaired Δ agree to three decimals here. |
| **(non-finding)** | Both 1000 ms cells hit the 250-turn forfeit exactly once in 80 games; `w300` and `k100` zero. | §2 table. | Below any rate this campaign can resolve; recorded so it is not re-derived. |

---

## 6. What this campaign cannot say

* **Nothing about Foul Play in general.** One checkpoint, one box, 400 Foul Play games. Every cell's
  interval is ~±11 pp.
* **Nothing about knowledge below the Smogon floor** (§3.2) — the tier that could not be removed is
  exactly the one a gen3ou bot most needs.
* **Nothing about search SHAPE.** `--search-time-ms` is the only budget Foul Play exposes; the
  decoupled-UCT structure, the world count (`--search-parallelism`, pinned at 1 here) and the
  hand-written evaluator were all held fixed.
* **Nothing about our own model.** No training happened; the checkpoint is frozen at 75M.
* **Nothing about the open ladder** — both clients spoke to a pinned local server, or to the front
  end, and the drift postures are unexercised. `src/main/ladder_drift_scan.py` remains that
  instrument.
* **Nothing about the front end under Foul Play or above concurrency 1.**

---

## 7. What is in this directory

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration, committed as `ea296538` before any cell ran |
| `README.md` | this |
| `axes_analysis.txt` | the job-1 analysis as `scripts/analyze_axes.py` printed it |
| `frontend_analysis.txt` | the job-2 analysis as `scripts/analyze_frontend.py` printed it |
| `cells/<cell>/games.jsonl` | 80 rows per Foul Play cell, one per battle, full provenance stamped on every row |
| `cells/<cell>/summary.json` | the tool's own summary — status, Wilson, integrity counters, realized visits |
| `cells/<cell>/cell.meta` | start/end timestamps, box load at both, the gen3ai HEAD the cell ran at |
| `frontend/<cell>/…` | the same three files for each of the four job-2 cells |
| `scripts/analyze_axes.py`, `scripts/analyze_frontend.py` | the analyses; re-runnable against the scratch tree |
| `scripts/run_cells.sh`, `run_k.sh`, `run_k100.sh`, `run_frontend.sh` | the exact drivers |
| `configs/anchors_std.json`, `configs/anchors_noknow.json` | the two `$GEN3AI_ANCHORS_CONFIG` files, with their deviations documented inside |
| `patches/foulplay_plumbing.patch` | the four-patch diff against `/home/goodlad/dev/foul-play` @ `6c467c08` |

The scripts carry absolute paths to this box's scratch directory
(`/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes/`) and to the three Foul Play checkouts
(`/home/goodlad/dev/foul-play{,-axes,-noknow}`). They are the PROVENANCE of `games.jsonl`, not a
reusable tool; a rerun should re-path them. The peer's full logs (≈ 400 MB of MCTS traces) are NOT
committed — the realized visit counts scraped from them are in `axes_analysis.txt` and the
summaries.

---

## 8. Ledger paragraph — ready to append (this file does NOT edit `ledger.md`)

**2026-09-17 · MEASUREMENT (MAJOR) · THE TWO FOUL PLAY AXES ARE NOT SEPARABLE AT n = 80, AND THE
WIDTH AXIS IS FLAT — plus the WS FRONT END PASSES AS AN ANCHORS TRANSPORT.** Pre-registered
(`ea296538`) and run on `ai_v12_02_winprob_critic@75,005,952`, greedy, both sides drawing the same
72-team sample pool, 80 games per cell, one cell at a time on a box carrying a live training arm.
**(a) WIDTH.** Realized Foul Play width was moved over a **9.4× range** — 83,031 / 259,513 / 779,604
MCTS visits per decision at `--search-time-ms` 100 / 300 / 1000 — and our win rate read **0.388
[0.288, 0.497] / 0.287 [0.200, 0.395] / 0.450 [0.346, 0.559]**. The fitted slope is **+0.027 win
rate per ln-unit of visits, game-bootstrap 95 % [−0.045, +0.095]** — the WRONG SIGN and covering
zero — and the registered contrast `w100 − w1000` is **−0.062, Newcombe 95 % [−0.210, +0.089],
NOT DETECTED**. 🚨 **This retires "win rate orders inversely to realized width" as a law**: hazard
H-E's three points span 1.21× across three DIFFERENT checkpoints, these five span 9.4× on ONE, and
the relation is not monotone. Rule 23 still binds (the realized count travels with every read), and
the campaign-to-campaign width gap is now known to be ~**1.9×** at a constant nominal 1000 ms
(747 k–780 k here vs the de-risk's 1.40 M), far outside the ±11 % the de-risk measured. **(b)
KNOWLEDGE.** With Foul Play's `TeamDatasets` emptied — the two `data.foulplay.cc` replay tiers and
Showdown's published sets, verified by its own `source=` lines moving from 87 % `teamdatasets-full`
to **100 % `smogonsets`** — the matched-width contrast is **+0.000 [−0.151, +0.151]**, width-corrected
residual **+0.001 [−0.150, +0.152]**: 36/80 against 36/80, discordant pairs 15/15. An unregistered
follow-up at 100 ms read **+0.050 [−0.101, +0.197]**, pooled **+0.025 [−0.083, +0.132]**, all NOT
DETECTED — and it killed a width×knowledge interaction that a 12-game smoke (10/12 = 0.833) had
suggested. 🚨 **"No set prediction at all" is NOT REACHABLE and that is a finding**: gen3ou has no
team preview, so `sample_standardbattle_pokemon` MUST invent the opponent's unrevealed Pokemon and
raises on an empty usage table — the axis measured is T2+T3 over a Smogon-usage floor, which biases
the effect towards zero. **VERDICT: "not separable at this n"** — never "equal" (rule 6) — and the
ledger's open question ("Foul Play's edge may be its opponent-set knowledge as much as its search")
is answered as *neither, at this resolution*; the untested candidates are the hand-written evaluator
and the search's SHAPE. **(c) THE FRONT END.** One SOP tier-B read (`metamon:SmallRL` greedy, both
team sets, 100 games each) through `utils.bridge.ws_frontend --impl rust` and through the Node
server, same seed, same 12-minute window: home **0.630 vs 0.580 (Δ +0.050 [−0.084, +0.182])**, away
**0.520 vs 0.550 (Δ −0.030 [−0.165, +0.106])**, pooled **+0.010 [−0.086, +0.106]** — every cell
agrees inside the delta's own CI. **200 front-end battles with 0 errors**, `argmax_match_rate` 1.0
on both sides of all four cells, no `UnknownMessageType`, throughput within 10 % (0.575 vs 0.629
games/s home). **VERDICT: the front end MAY be the DEFAULT transport for the anchors CLI**, with two
stated conditions — it is a transport+SIMULATOR comparison (`--impl rust` vs the pinned Showdown
server), and nothing here tests Foul Play over it or concurrency > 1. **(d) FOUR PLUMBING FINDINGS
against Foul Play `6c467c08`, each reproduced and patched in a COPY (the pinned checkout is
untouched):** a pipelined `/challenge` PM is DROPPED while the peer is busy, which killed the
`ours_challenge` half at game 2 **three times out of three** — so every banked Foul Play number came
from the one direction the bug spares; the peer derives its player slot from the **first `|player|`
line in a batched frame**, so as the CHALLENGER under load it decides it is playing itself; the held
`|start` frame does not always carry the opponent's lead; and `move_usage_rates` calls `.get` on the
`[]` sentinel its own lookup returns, unreachable in the standard configuration and immediate in the
degraded one. Artifact:
`designs/research_state/measurements/foulplay_axes_and_frontend_validation_2026-09-16/`.
