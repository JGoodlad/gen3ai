# Metamon as a gen3ou baseline opponent — de-risk, 2026-09-14

**VERDICT: GO for `SmallRL` as the recurring baseline; GO for `SyntheticRLV2` as the strongest
reference.** Both pretrained policies play gen3ou against our client on our pinned Showdown server
with **zero protocol failures on either side across 180 games**. Our `ai_v12_02_winprob_critic`
final snapshot beats `SmallRL` **0.742 [0.657, 0.812]** and `SyntheticRLV2` **0.583 [0.457, 0.699]**
(Wilson 95%). Cost is **2.3 s/game** against the 15M model and **8.5 s/game** against the 200M one,
CPU-only, on a box already carrying a training arm.

Three hazards were found and two of them would have silently corrupted the measurement. They are in
§6; the one that matters most is that **Metamon runs upstream poke-env and we run our vendored
fork, and the two parsers disagree on a real class of our own team files.**

---

## 1. Setup — every hash

| Thing | Value |
|---|---|
| Metamon checkout | `/home/goodlad/dev/metamon` @ `0a00a759` ("update readme") |
| Metamon's Showdown submodule | `server/pokemon-showdown` @ `d62d3a398` = `v0.11.10-1237-gd62d3a398` |
| **Our** Showdown pin | `deps/pokemon-showdown` @ `e0551883f` = `v0.11.10-1224-ge0551883f` |
| Pin relationship | **ours is a direct ANCESTOR of theirs, 13 commits behind** |
| Server used for every game | **ours**, `npm run showdown -- 9217` from `/home/goodlad/dev/gen3ai` |
| Metamon env | conda `metamon`, python 3.10, `pip install -e .` |
| `amago` | 3.4.0 (pulled in by Metamon's own `pyproject.toml`; no separate install needed) |
| `torch` (metamon env) | 2.14.0, **CPU only** — `CUDA_VISIBLE_DEVICES=""` set before the torch import |
| `poke-env` (metamon env) | **0.8.3.3, upstream from PyPI** |
| `poke-env` (ours) | **the vendored fork at `src/poke_env/`** — a different package |
| Weight cache | `METAMON_CACHE_DIR=/home/goodlad/dev/metamon/cache` (weights + team sets live here) |
| Our checkpoint | `models/ai_v12_02_winprob_critic/final_model.zip`, 3,056,823 params, 75,005,952 steps |

### Why not the `production` baseline

The task asked for the registry name `production`. It resolves (`agents.training.baselines`) to
`ai_v9_21_gen17_pfspoff_0820/final_model.zip` @ `config_version=97`,
`arch_signature=gen3_critic_route_wave_v1` — and **it does not load at HEAD**:

```
TypeError: ExtractorBuild.__init__() got an unexpected keyword argument 'threat_prob_outspeed'
```

That is ordinary arch drift (the registry entry's own `pending` block says the surface source moves
to `ai_v12_02_winprob_critic` once that arm passes its critic gate). So every game below was played
by **`ai_v12_02_winprob_critic`'s final snapshot**, as the fallback the task named. The numbers are
about that checkpoint and nothing else.

### The Showdown-pin difference, examined rather than assumed

13 commits, and the diff that could touch a gen3ou game is small:

```
config/formats.ts         | 12 +-      data/formats-data.ts      |  2 +-
data/mods/gen3/scripts.ts |  2 +-      data/mods/gen4/scripts.ts |  2 +-
sim/battle-actions.ts     |  2 +-      sim/dex-conditions.ts     |  1 +
sim/global-types.ts       |  3 +-      sim/pokemon.ts            | 18 +-
```

It did not matter here **because both clients played on OUR server** — the pin difference is only a
hazard if someone runs Metamon against its own bundled server and compares. Recorded, not fixed.

---

## 2. The forfeit rule, and where it lives

A training episode forfeits at **`agents.training.stall.StallConfig.threshold`**, which defaults to
**`agents.observation.constants.MAX_TURNS` = 250** (`gen3_deadline_clock_v1` — the same constant
normalises the observation's turn clock, so the two cannot drift). `RLPlayer.choose_move` already
called `_handle_stall` on the websocket path, so `play.py` inherited the limit — but as an invisible
class default that nothing printed and nothing could override.

This pass makes it explicit, **without touching training code**:

* `src/main/play.py` gains `--forfeit-turn-limit`, whose default is `StallConfig().threshold` — read
  from the trainer, never restated. It is passed to `RLPlayer` as `stall_config=StallConfig(
  threshold=...)` and **printed at startup** (`[play] forfeit turn limit: 250 (trainer default 250)`).
* `src/main/play_forfeit_limit_test.py` pins it four ways: the default equals `StallConfig().
  threshold` equals `MAX_TURNS`; the parser default matches; the flag overrides; and the flag
  **reaches the player's `stall_config`** — because a flag that parses but never arrives is a no-op,
  and a no-op here reads exactly like a limit that fired.

`origin/main` was checked for a Foul Play–landed flag first (`f234dae1`); there was none.

**It never fired.** 0 of 180 games reached turn 250; the longest was 178 turns.

---

## 3. Team sets — both sides draw from OUR pool

Both players draw from the same 719-team gen3ou pool, so neither side gets a team advantage.

| | Our side | Metamon side |
|---|---|---|
| Source | `utils.team_loader.TeamLoader` → `Gen3Teambuilder` (`play.py --team-pool`) | `metamon.env.TeamSet` over `$METAMON_CACHE_DIR/teams/gen3ai_pool/gen3ou/*.gen3ou_team` |
| Teams | 719 (72 sample + 647 other), all 719 locally valid for gen3ou | the same 719, exported by `export_pool_teams.py` |
| Distinct teams actually drawn | — | 97 of 719 over 120 games; 48 over 60 games |

`export_pool_teams.py` reproduces what `Gen3Teambuilder` does to a paste before it reaches a battle:
local validation, and `fix_gen3_hp_ivs` (**545 files** had a Hidden-Power IV line inserted or
rewritten — a HP mon left at 31/31/31/31/31/31 gets the *wrong HP type* on the server). It also
strips nicknames — see hazard H1. The IV fix is applied as a **text edit on the original paste**, so
everything else is the file we already train against, byte for byte.

**What Metamon ships for gen3**, for when the owner wants their sets instead (`metamon.env.
get_metamon_teams("gen3ou", <name>)`, downloaded from `jakegrigsby/metamon-teams`):

| `set_name` | gen3 teams | What it is |
|---|---|---|
| `competitive` | < 30 | human-made Smogon sample teams; the paper's human-ladder set. Metamon has **overfit to it** by its authors' own description |
| `gl_05_26` | 107k | "General Ladder May '26" — recent replays filled from time/rating-appropriate usage stats |
| `hl_05_26` | 22k | "High Ladder May '26" — the 1400+/tournament subset of `gl_05_26` |
| legacy | — | `paper_variety`, `paper_replays`, `modern_replays`, `modern_replays_v2` still resolve |

---

## 4. Results

Both sides logged every game independently; `analyze.py` joins them and `games.jsonl` is the merged
record (180 rows). Our side is **greedy** (`--temperature 0`, our measurement setting); Metamon is at
**its own** eval default (`--temperature 1.0`, sampling). That asymmetry is each project's own
convention and is stated, not corrected.

### Win rate — `ai_v12_02_winprob_critic` @ 75M vs each Metamon policy

| Metamon policy | Params | n | our W | our L | tie | **our win rate** | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `SmallRL` (ckpt 40) | 13,938,567 | 120 | 89 | 30 | 1 | **0.742** | [0.657, 0.812] |
| `SyntheticRLV2` (ckpt 48) | 200,894,911 | 60 | 35 | 25 | 0 | **0.583** | [0.457, 0.699] |

The two CIs overlap, so this series does **not** on its own establish that `SyntheticRLV2` is the
harder opponent — but the ordering is the one Metamon's own gen3 GXE table predicts, and the point
estimates differ by 16 pp.

**Metamon's own scoreboard agrees, computed independently.** Its evaluator reported
`Average Win Rate` of **0.2500** for `SmallRL` and **0.4167** for `SyntheticRLV2`. Complementing:
`1 − 0.4167 = 0.5833`, **exactly** our SyntheticRLV2 figure; and `1 − 0.25 = 0.75 = 90/120`, our 89
wins **plus the one tie** — which Metamon books as a loss for itself because its result field is a
boolean (H3). Two codebases, two loggers, one number. That is the check that the positional join and
the win accounting are both right.

### Game shape, failures, cost

| | `SmallRL` | `SyntheticRLV2` |
|---|---:|---:|
| mean turns | 36.3 | 38.4 |
| max turns | 178 | 130 |
| **our forfeit limit fired** | **0** | **0** |
| ties | 1 | 0 |
| sides disagree on the winner | 1 (see §6 H3) | 0 |
| our-side tracebacks / `classify` raises | **0 / 0** | **0 / 0** |
| Metamon-side tracebacks | **0** | **0** |
| our illegal-action fallbacks (`n_defaults`) | **0** | **0** |
| Metamon's own `Average Valid Actions` | 0.9982 | 0.9941 |
| Metamon's own reported win rate | 0.2500 | 0.4167 |
| our stale-decision re-decides | **0** | **0** |
| timeouts | **0** | **0** |
| wall clock (both clients up) | 277 s | 508 s |
| **seconds per game** | **2.30** | **8.47** |

### Per-move inference, CPU only

🚨 **These are CONTENDED numbers.** The box carries a live training arm; `uptime` read a load average
of **22.6 on 16 cores** during the second series. They are what a baseline run actually costs here,
not a clean measurement of either model — and our own model's median moves 27 ms → 110 ms between
series purely because the opponent in the second one is 200M parameters on the same cores.

| Series | side | model | n decisions | median | mean | p90 | max |
|---|---|---|---:|---:|---:|---:|---:|
| `SmallRL` | Metamon | SmallRL, 15M | 4,938 | **14.2 ms** | 30.2 ms | 72.5 ms | 2.01 s |
| `SmallRL` | ours | 3.06M | 4,806 | 27.2 ms | 50.5 ms | 117 ms | 0.76 s |
| `SyntheticRLV2` | Metamon | SyntheticRLV2, 200M | 2,580 | **120 ms** | 172 ms | 320 ms | 10.2 s |
| `SyntheticRLV2` | ours | 3.06M | 2,544 | 110 ms | 157 ms | 309 ms | 1.64 s |

Metamon's cost is **~8.4× median from 15M to 200M**, close to the 14× parameter ratio. The `max`
column is the first call of the series (graph/kernel warm-up), not a steady-state cost. Note the
p90/median ratio of ~5: these are **sequence models**, and a decision late in a long battle carries a
longer context than one on turn 2 — cost grows within a game.

**Model load time is the other half of the cost.** A `SyntheticRLV2` process spends minutes building
the 200M network and loading 804 MB of weights before its first battle; the series driver waits for
its "Made Challenge Env" banner for exactly this reason. For a recurring baseline, amortise one
process over many games rather than paying it per game.

---

## 5. The four verdicts

### (a) Does the pretrained model play gen3ou on our pinned server, both ways? **YES — cleanly.**

180 games, **zero** protocol-parse failures in either direction, zero timeouts, zero connection
errors, zero tracebacks on either side. Specifically:

* **Them reading us.** Our client's teams keep their nicknames (our pool has them, and our training
  distribution therefore has them); Metamon's `MetamonBackendBattle` read every one without
  incident, and its `Average Valid Actions` was **0.9982** (SmallRL) / **0.9941** (SyntheticRLV2) —
  i.e. 0.2–0.6% of its own chosen actions were illegal and fell back to `choose_random_move`, a
  Metamon-internal action-masking rate, not a parse failure.
* **Us reading them.** `battle_event.classify` raises on an unknown protocol keyword **by design**,
  and on a live battle that kills the parse task, sends no choice, and loses on the timer. It **never
  fired**: 0 raises, 0 tracebacks in `gen3ai.log` across both series. This is the expected result
  given both clients spoke to *our* pinned server — it is **not** evidence about Metamon's newer
  submodule pin.
* The poke-env version gap (upstream 0.8.3.3 vs our fork) caused **no** in-battle problem. It caused
  a **team-file** problem; see H1.

### (b) Inference cost on CPU, per model size

15M (`SmallRL`) is **14 ms/move median, 2.3 s/game**; 200M (`SyntheticRLV2`) is **120 ms/move median,
8.5 s/game** — both under heavy contention, so treat them as ceilings for a quiet box. A 120-game
series against `SmallRL` costs **under 5 minutes** of wall clock; the same against `SyntheticRLV2`
costs **~17 minutes**, plus a one-off multi-minute model build. Neither needs the GPU.

### (c) Which is the practical recurring baseline, which is the strongest reference

* **Recurring: `SmallRL`.** 15M, 56 MB, 2.3 s/game, seconds to load. Cheap enough to run every
  eval cycle, and at 0.742 against us it is a live signal rather than a floor — an opponent we beat
  ~3 games in 4 still has 25 pp of headroom to move.
* **Strongest reference: `SyntheticRLV2`.** The paper's best policy, 200M, 804 MB. 0.583 against us
  puts it close to even, which is exactly what a *reference* should be. Run it on milestones, not
  cycles.
* Deliberately **not** recommended: the PokéAgent-era models (`Kadabra*`, `Alakazam`, `Kakuna`).
  Their authors state a Gen1/Gen9 bias and that it "took several iterations to recover the paper's
  Gen 1-4 performance". `Kakuna` (142M) claims ~63% gen3ou GXE and is the strongest public model, so
  it is the natural *next* candidate — but it is a bigger-cost, differently-trained axis and this
  pass did not measure it.

### (d) What a serverless integration would need

Our in-process transport is `utils.bridge.battle_stream_client.BattleStreamClient`, a `PSClient`
subclass that `_LocalBattleRunner._attach` assigns onto `player.ps_client`. It needs exactly four
things from a player, and **Metamon's player has all four**:

| `_attach` requires | Metamon's side |
| `player.ps_client._account_configuration` | present — `MetamonPlayer` extends `poke_env.player.Player` |
| `player._handle_battle_message` | present, inherited |
| `player._update_challenges` | present, inherited |
| `player._handle_challenge_request` | present, inherited |
| built with `start_listening=False` | **available** — `OpenAIGymEnv.__init__` takes `start_listening` and forwards it to the player it constructs (`poke_env/player/openai_api.py:183-195`) |

**The class that owns the connection** is `poke_env.ps_client.PSClient`, held at `Player.ps_client`.
Metamon reaches it through `metamon.env.wrappers.PokeEnvWrapper` (a `poke_env.player.openai_api.
OpenAIGymEnv`), whose `self.agent` is `metamon.env.metamon_player.MetamonPlayer` (or plain `Player`
for the `poke-env` backend, or `PokeAgentPlayer`). The battle object is
`metamon.env.metamon_battle.MetamonBackendBattle`, a `poke_env.AbstractBattle` subclass that replaces
poke-env's protocol interpreter with Metamon's own — which is fine for us, because the bridge only
frames and relays protocol; each side keeps its own battle model.

**The blocker is not the transport. It is the two poke-env packages.**
`MetamonPlayer` subclasses *upstream* `poke_env.player.Player`; `BattleStreamClient` subclasses our
*vendored* `poke_env.ps_client.PSClient`. In one process `import poke_env` resolves to exactly one of
them, and the other half then breaks — silently, which is precisely the failure
`src/poke_env_fork_gate_test.py` exists to prevent and the root `CLAUDE.md` warns about at length.
Three ways out, in increasing order of honesty:

1. **Don't.** Keep the websocket harness in this directory. It costs 2.3 s/game and needed no
   changes to either project. This is the recommendation.
2. Run Metamon **out of process** behind a thin stdio/socket shim speaking `(observation → action)`,
   and let the bridge drive only our side. Buys serverlessness for our half; a protocol to maintain.
3. Port Metamon's policy onto our fork. Real work, and it re-dates itself every Metamon release.

**Observation and action conventions, for whoever tries (2) or (3).** Metamon's obs is a Dict of
`{"numbers": Box(-10, 10, (48,), float32), "text_tokens": Box(-1, 1407, (87,), int32)}` —
a tokenized natural-language state, nothing like our flat 2501-dim float vector. Actions:

| | size | layout |
|---|---:|---|
| Metamon `MinimalActionSpace` (what both paper models use) | **9** | 0–3 moves, 4–8 switches |
| Metamon `DefaultActionSpace` | 13 | the 9 above + 4 gimmick-move slots (tera/dynamax), folded back onto 0–3 in `MinimalActionSpace` |
| **Ours** | **11** | 0–5 switches, 6–9 moves, 10 struggle |

Both index *moves* and *switches*, but the orders are reversed, the widths differ, and **ours has an
explicit Struggle action that Metamon has none of** — Metamon expresses Struggle as an ordinary move
slot. Any mapping layer has to own that asymmetry explicitly; it is not a permutation.

---

## 6. Hazards — each is a finding

### H1 🚨 Upstream poke-env and our vendored fork DISAGREE on our own team files, and the failure is a stall

On a team line carrying a nickname and **no item** — `Airmure (Skarmory)` — upstream poke-env 0.8.3.3
fails to split nickname from species and packs the entire string as the nickname with an **empty
species field**:

```
upstream 0.8.3.3:   Airmure (Skarmory)  |||keeneye|protect,spikes,roar,thief|Calm|...
our vendored fork:  Airmure|skarmory||keeneye|protect,spikes,roar,thief|Calm|...
```

Showdown normalises that to `airmureskarmory` and rejects the team:

```
|popup|Your team was rejected for the following reasons:||||- The Pokemon "airmureskarmory" does not exist.
```

Our own `validate_teams_locally` passes the file — it is not the file that is wrong, it is *whose
parser reads it*. **The symptom is a HANG, not an error**: the rejected challenge never becomes a
battle and the series sits there. It burned the first 120-game attempt at game 8.

*Fixed in the exporter*, by stripping nicknames (3,270 lines across the 719 files). A nickname
carries no battle meaning, so dropping it removes the dependency on whose parser reads the file
entirely. **Standing lesson: any team file handed to a third-party poke-env must be nickname-free.**

Related data-hygiene note, not fixed: our pool contains mojibake nicknames (`MÃ©talosse`, a
double-encoded `Métalosse`). Both parsers carry them through harmlessly — they are nicknames — but
they mark pastes that went through a bad encoding step somewhere in acquisition.

### H2 🚨 Every Metamon transformer policy is UNRUNNABLE on CPU out of the box

`amago`'s `TformerTrajEncoder` defaults to `attention_type=FlashAttention`, which asserts:

```
AssertionError: Missing flash attention 2 install (pip install amago[flash]).
```

`flash-attn` is a CUDA-only wheel, so **no Metamon transformer policy runs on a CPU-only process as
shipped**. `run_metamon_side.py` injects
`traj_encoders.TformerTrajEncoder.attention_type = VanillaAttention` through the supported
`PretrainedModel.gin_overrides` seam. Vanilla and Flash attention are the same causal softmax
attention — Flash is an exact algorithm, not an approximation — so the policy's outputs are
unchanged and only the speed differs. **This means every CPU timing in §4 is a VanillaAttention
timing**, which is the pessimistic side.

### H3 Metamon's per-battle CSV cannot express a tie, and its `Battle ID` is random

Two separate defects in `metamon/env/wrappers.py`, both of which corrupt a naive merge:

* `battle_id = "".join(str(random.randint(0, 9)) for _ in range(10))` — **not** the Showdown room
  number, and not a join key with anything. `analyze.py` therefore joins **positionally**, which is
  sound only because the harness is strictly sequential on both sides; the check that it held is
  `sides_disagree`, which would go systematically non-zero on a misalignment.
* `info["won"] = self.agent.n_won_battles > self.battle_reference` — a boolean, so a **tie is
  recorded as a loss**. The one row where the two sides "disagree" is exactly this: our side read
  poke-env's `won=None, finished=True` (a tie) and Metamon's CSV said LOSS. Their schema cannot
  distinguish the two, so this is an expressiveness limit, not a contradiction. Our side's flags are
  the authority.

Also: the CSV's `Turn Count` is Metamon's `turn_counter` (env steps taken), **not** the battle's turn
number. `games.jsonl` keeps both, as `turns` (ours, authoritative) and `metamon_env_steps`.

### H4 Metamon's server address is hardcoded and there is no flag for it

`PokeEnvWrapper.server_configuration` is a property returning the module-level
`LocalhostServerConfiguration`, which poke-env hardcodes to `ws://localhost:8000` — **our shared dev
server**, one port away from the live training server. There is no CLI flag, no env var, and no
constructor argument. `run_metamon_side.py` rebinds the module global before any env is constructed
(the property reads it at call time) and refuses 8000/8001 in code, the same way `play.py` does.
Anyone running `python -m metamon.rl.evaluate` directly on this box will hit :8000.

### H5 Metamon's own prints are not flushed

Metamon's `print()` calls carry no `flush=True`, so a redirected stdout is block-buffered and the
"Made Challenge Env" banner a driver waits on never lands. `run_series.sh` sets `PYTHONUNBUFFERED=1`.
Without it the acceptor looks dead while it is in fact online and waiting.

### H6 Showdown pin difference (recorded, not fixed)

Ours is 13 commits behind Metamon's bundled submodule (§1). Irrelevant here because both clients
played on ours; it becomes relevant the moment anyone compares against numbers Metamon produced on
its own server.

---

## 7. What is in this directory

| File | What |
|---|---|
| `README.md` | this |
| `games.jsonl` | 180 merged per-game rows, both sides' views |
| `summary.json` | the per-series tables as JSON |
| `export_pool_teams.py` | our 719-team pool → Metamon `TeamSet` files (validation, HP-IV fix, nickname strip) |
| `run_metamon_side.py` | Metamon policy as one client: port rebind, CPU pin, attention override, move timing |
| `run_gen3ai_side.py` | our checkpoint as the other client — `play.py`'s own path plus a per-battle observer |
| `run_series.sh` | one head-to-head series end to end |
| `analyze.py` | merge + Wilson CIs + failure counts |
| `raw/` | both sides' raw timing JSON, Metamon's battle-log CSVs, the team manifest, series metadata |

### Reproducing

```bash
# 1. a server on a port that is NOT 8000/8001, from OUR pinned submodule
npm run showdown -- 9217          # record the PID

# 2. our pool -> Metamon team files (from the MAIN checkout; it owns data/)
export PYTHONPATH=$PYTHONPATH:src
python3 export_pool_teams.py --out $METAMON_CACHE_DIR/teams/gen3ai_pool/gen3ou

# 3. a series (starts the acceptor, waits for it, then the challenger)
bash run_series.sh SmallRL 120 9217 /tmp/out_smallrl

# 4. the tables
python3 analyze.py --series SmallRL=/tmp/out_smallrl --out games.jsonl
```

The Showdown server started for this pass was stopped by its recorded PID; nothing else on the box
was touched. The Metamon checkout and both downloaded checkpoints remain under
`/home/goodlad/dev/metamon/`.

---

## 8. Ledger paragraph — ready to append (do not append it from here)

> **2026-09-14 · METAMON DE-RISKED AS A GEN3OU BASELINE OPPONENT — GO, and three hazards that would
> each have corrupted the read.** 180 head-to-head gen3ou games on our own pinned Showdown server
> (`deps/pokemon-showdown` @ `e0551883f`, port 9217), Metamon @ `0a00a759` in its own conda env,
> CPU-only, both sides drawing from OUR 719-team pool. `ai_v12_02_winprob_critic`'s final snapshot
> (75,005,952 steps) beats **`SmallRL` (15M) at 0.742, Wilson 95% [0.657, 0.812], n=120** and
> **`SyntheticRLV2` (200M) at 0.583 [0.457, 0.699], n=60**; the CIs overlap, so the ordering is
> suggestive, not established. **Protocol compatibility is CLEAN in both directions** — 0 parse
> failures, 0 timeouts, 0 tracebacks, 0 illegal-action fallbacks on our side, and our 250-turn
> forfeit limit never fired (longest game 178 turns). **Metamon's own evaluator, computing the same
> games independently, reported 0.2500 and 0.4167 for itself — complementing to 0.75 (our 89 wins
> plus the one tie, which its boolean result field books as its own loss) and to 0.5833 exactly.** Cost, under a load average of 22.6 on 16
> cores: **2.3 s/game vs SmallRL (14 ms/move median), 8.5 s/game vs SyntheticRLV2 (120 ms/move)** —
> `SmallRL` is the practical recurring baseline, `SyntheticRLV2` the milestone reference. The
> registry name `production` could NOT be used: it resolves to `ai_v9_21_gen17_pfspoff_0820` @
> `config_version=97` and no longer loads at HEAD (`ExtractorBuild.__init__() got an unexpected
> keyword argument 'threat_prob_outspeed'`). **THREE HAZARDS.** (1) Metamon runs UPSTREAM poke-env
> 0.8.3.3 while we vendor a fork, and on a nickname line with no item (`Airmure (Skarmory)`) upstream
> packs an EMPTY species field; Showdown rejects the team as `airmureskarmory` and **the match
> STALLS rather than erroring** — it killed the first 120-game attempt at game 8. Any team file
> handed to a third-party poke-env must be nickname-free. (2) Every Metamon transformer policy is
> unrunnable on CPU as shipped: `amago` defaults to `FlashAttention`, a CUDA-only wheel;
> `VanillaAttention` via `gin_overrides` is the same exact math. (3) Metamon's per-battle CSV
> generates a RANDOM `Battle ID` (no join key with the Showdown room) and records a tie as a loss, so
> the two sides' logs can only be joined POSITIONALLY. A serverless integration is mechanically
> close — `_LocalBattleRunner._attach` needs four attributes that Metamon's `MetamonPlayer` inherits
> from `poke_env.player.Player`, and `OpenAIGymEnv` accepts `start_listening=False` — but is blocked
> by the two poke-env packages: their player subclasses upstream, our `BattleStreamClient`
> subclasses the fork, and one process resolves `import poke_env` to exactly one. Recommendation:
> keep the websocket harness. Action spaces do not map by permutation either — theirs is 9 (0–3
> moves, 4–8 switches), ours is 11 (0–5 switches, 6–9 moves, 10 struggle) with an explicit Struggle
> they have no slot for. Full measurement:
> `designs/research_state/measurements/metamon_derisk_2026-09-14/`.
