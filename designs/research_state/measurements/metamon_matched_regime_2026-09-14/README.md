# Metamon at a MATCHED REGIME — the 2×2, 2026-09-14

**PRIMARY VERDICT: our model is NOT BETTER than `SyntheticRLV2`.** On the pre-registered primary
row — greedy-vs-greedy, both sides drawing from Metamon's own 20-team `competitive` gen3ou set —
`ai_v12_02_winprob_critic` @ 75M wins **0.420, Wilson 95% [0.328, 0.518], n = 100**. The registered
bar was a Wilson lower bound above 0.50; it is not cleared, so the verdict is **NOT DETECTED**, and
the point estimate is on the losing side of even. Against `SmallRL` the same row reads **0.650
[0.553, 0.736]** and the bar **IS** cleared: **BETTER**.

800 games, 4 cells per model, 100 per cell, role balanced inside every cell. Both regimes were
**verified on every decision on both sides**, not assumed. The de-risk's `0.742 / 0.583` do not
survive the matched regime: against `SmallRL` the number falls **22.2 pp [−34.1, −9.4]**.

**The single most useful finding is not a win rate.** *"Temperature 1.0"* is **not a comparable
regime across models**: at T = 1.0 `SmallRL` plays its own argmax **64.7%** of the time and
`SyntheticRLV2` plays it **88.0%** of the time (measured over 14,085 sampled decisions; 30,529 decisions
instrumented in all). The same nominal setting is a 35%-perturbation of one policy and a 12%-perturbation of the other, and the two models'
temperature effects come out with **opposite signs** because of it. A baseline whose opponent regime
is "each project's own default" is not a fixed yardstick.

---

## 1. Setup — every hash

| Thing | Value |
|---|---|
| Our checkout (worktree branch) | `metamon-matched-regime`, from `4b9b8235` |
| Our Showdown pin | `deps/pokemon-showdown` @ `e0551883f` = `v0.11.10-1224-ge0551883f` |
| Server for every game | **ours**, `npm run showdown -- 9350`, started and stopped by its recorded PID |
| Metamon checkout | `/home/goodlad/dev/metamon` @ `0a00a759` |
| Metamon env | conda `metamon`, python 3.10, torch 2.14.0 **CPU only** (`CUDA_VISIBLE_DEVICES=""`) |
| `poke-env` (metamon env) | 0.8.3.3, **upstream from PyPI** |
| `poke-env` (ours) | the **vendored fork** at `src/poke_env/` — a different package (see H-A) |
| Weight / team cache | `METAMON_CACHE_DIR=/home/goodlad/dev/metamon/cache` |
| Our checkpoint | `models/ai_v12_02_winprob_critic/final_model.zip`, 75,005,952 steps, CPU |
| Metamon policies | `SmallRL` ckpt 40 (13.9M) · `SyntheticRLV2` ckpt 48 (200.9M), both `MinimalActionSpace`, `poke-env` backend |
| Forfeit limit | `--forfeit-turn-limit` at its default **250** (`StallConfig().threshold`) |
| Attention | `VanillaAttention` on the Metamon side (de-risk H2 — `flash-attn` is a CUDA-only wheel) |
| Box | 16 cores carrying a live training arm; load average 25–31 throughout |

**The registry name `production` was NOT used**, for the reason the de-risk recorded: it resolves to
`ai_v9_21_gen17_pfspoff_0820` @ `config_version=97` and no longer loads at HEAD.

### The two team sets

| | **home** | **away** |
|---|---|---|
| what | our 719-team gen3ou pool, the de-risk's nickname-free export | Metamon's own `competitive` gen3ou set |
| size | 719 | **20** |
| our side draws via | `Gen3Teambuilder(TeamLoader().get_all_teams(), rng_seed=…)` | `Gen3Teambuilder(<20 files>, rng_seed=…)` |
| their side draws via | `TeamSet($METAMON_CACHE_DIR/teams/gen3ai_pool/gen3ou)` | `TeamSet(.../teams/competitive/gen3ou)` |
| distinct teams actually drawn per cell | 94 ours / 80–89 theirs | 19 / 19 (of 20) |

**Why `competitive` is the away set.** Metamon's README says of it: *"This is the set used for human
ladder evaluations in the paper"*, and its own cross-generation strength tables are headed
"Competitive TeamSet". It is their ground, chosen by them. Its README also states Metamon **has
overfit to it** — which is the point of an away game, and was stated in the pre-registration before
any game was played.

### Reproducing

```bash
npm run showdown -- 9350                       # record the PID; NEVER 8000/8001
bash run_all.sh SmallRL       100 9350 <out> S # 4 cells x 2 role half-cells
bash run_all.sh SyntheticRLV2 100 9350 <out> Y
python3 analyze.py --root <out> --out games.jsonl --summary-out summary.json
```

---

## 2. The design, and how each regime was VERIFIED

A **cell** is (sampling regime × team set) with **both sides at the same setting**. Each cell is
played as **two half-cells of 50 games** differing only in who sends the challenge, because Showdown
makes the challenger p1 and p1/p2 is not a priori neutral. Team draws are seeded per half-cell and
the team each side drew is recorded per game in `games.jsonl`.

**Metamon's greedy is `Agent.get_actions(sample=False)`, i.e. `argmax(dist.probs)`** — set by
overriding `Experiment.sample_actions_val`, which Metamon's `make_placeholder_experiment` never
passes so it falls through to the gin default. It is **NOT** reachable through a temperature:
`MetamonDiscrete.forward` divides logits by `self.temperature` (zero divides by zero) **and clips
probabilities to `[0.001, 0.99]` before renormalising**, so on a 9-way action space the argmax
action would still be played only ~99.2% of the time at any temperature.

**Verified per decision, both sides, all 800 games:**

| instrument | greedy cells | t1.0 cells |
|---|---|---|
| Metamon's `sample` kwarg as received by `Agent.get_actions` | `[False]` everywhere | `[True]` everywhere |
| Metamon `argmax_match_rate` (emitted action == argmax of the actor's own dist) | **1.0000** in all 7 recorded greedy half-cells | **0.623–0.680** (`SmallRL`), **0.877–0.882** (`SyntheticRLV2`) |
| our `stochastic` kwarg as received by `RLPlayer._predict_best_action` | `[False]` everywhere | `[True]` everywhere |

The sampling cells are the **positive control**: an instrument that reads 1.000 in both regimes
would have no power, and it does not — it reads 1.000 exactly in greedy and materially below 1.000
in sampling. Three half-cells lost their Metamon-side record to the crash in H-B and are marked
`n/a`; the regime was set at process start in all of them, and their per-game results match their
clean partners (H-C).

---

## 3. The 2×2 tables

Win rate is OURS. Ties count in the denominator and not the numerator, as in the de-risk. `cap` is
the 250-turn forfeit firing; `ph` is a phantom (≤2-turn) battle, which is wreckage from a cap, never
a game (H-B / H-C).

### `SmallRL` (ckpt 40, 13.9M)

| cell | n | W | L | T | **win rate** | Wilson 95% | mean turns | cap | ph |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| greedy · home | 100 | 52 | 48 | 0 | **0.520** | [0.423, 0.615] | 36.3 | 0 | 0 |
| **greedy · away (PRIMARY)** | 100 | 65 | 35 | 0 | **0.650** | **[0.553, 0.736]** | 48.8 | 2 | 1 |
| t1.0 · home | 100 | 63 | 37 | 0 | **0.630** | [0.532, 0.718] | 41.2 | 0 | 0 |
| t1.0 · away | 100 | 56 | 44 | 0 | **0.560** | [0.462, 0.653] | 42.3 | 0 | 0 |

**PRIMARY VERDICT: BETTER** — the Wilson lower bound on greedy·away is 0.553 > 0.50.
Pooled across all four cells: 236/400 = **0.590 [0.541, 0.637]**.

### `SyntheticRLV2` (ckpt 48, 200.9M)

| cell | n | W | L | T | **win rate** | Wilson 95% | mean turns | cap | ph |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| greedy · home | 100 | 50 | 49 | 1 | **0.500** | [0.404, 0.596] | 40.4 | 0 | 0 |
| **greedy · away (PRIMARY)** | 100 | 42 | 58 | 0 | **0.420** | **[0.328, 0.518]** | 47.9 | 0 | 0 |
| t1.0 · home | 100 | 41 | 59 | 0 | **0.410** | [0.319, 0.508] | 42.7 | 1 | 1 |
| t1.0 · away | 100 | 31 | 69 | 0 | **0.310** | [0.228, 0.406] | 44.8 | 1 | 1 |

**PRIMARY VERDICT: NOT DETECTED** — the lower bound is 0.328, far from 0.50, and the point estimate
is below even. Pooled: 164/400 = **0.410 [0.363, 0.459]**. `SyntheticRLV2` is ahead of us in three
of four cells and level in the fourth.

### The effects, as differences with their own CIs

Newcombe hybrid-score intervals (the difference interval consistent with the Wilson cell intervals).
Per UNDERSTANDING §7 rule 6, an effect whose CI covers zero is **NOT DETECTED**, never "equal".

| effect | `SmallRL` | `SyntheticRLV2` |
|---|---|---|
| **temperature** (greedy − t1.0), home | −0.110 [−0.241, +0.027] | +0.090 [−0.047, +0.223] |
| **temperature**, away | +0.090 [−0.045, +0.220] | +0.110 [−0.023, +0.238] |
| **temperature**, pooled (200 v 200) | **−0.010 [−0.105, +0.086]** NOT DETECTED | **+0.100 [+0.004, +0.194]** DETECTED |
| **team set** (home − away), greedy | −0.130 [−0.260, +0.006] | +0.080 [−0.057, +0.213] |
| **team set**, t1.0 | +0.070 [−0.065, +0.202] | +0.100 [−0.033, +0.228] |
| **team set**, pooled | **−0.030 [−0.125, +0.066]** NOT DETECTED | **+0.090 [−0.006, +0.184]** NOT DETECTED |
| **role** (we challenge − they challenge), pooled | +0.050 [−0.046, +0.145] | −0.020 [−0.115, +0.076] |

**Role is not detected for either model**, which is the result that makes balancing it inside every
cell cheap insurance rather than a wasted axis. Inside single cells it looks large and inconsistent
(`SyntheticRLV2` greedy·home reads 0.62 when they challenge and 0.38 when we do; greedy·away
reverses to 0.36 / 0.48), which is exactly the shape of a 50-game coin — and exactly why a cell must
not be played at one role.

---

## 4. Reading the three effects

### 4.1 The temperature effect does NOT transfer from our own regime measurement — and its sign depends on the OPPONENT

Our project's own regime figure is **+8.9 pp [+7.0, +10.7]** to the greedy side against a
temperature-1.0 opponent (`designs/training/eval_and_rating.md`, UNDERSTANDING §3.2 rule 5). The
pre-registration predicted it should **not** transfer, because it is an **asymmetry** term — what
the greedy side gains *because the opponent samples* — and moving both sides together removes it
from both halves. That prediction holds in direction but the magnitude bound was wrong for the
200M model:

* `SmallRL`: **−0.010 [−0.105, +0.086]**. Not detected. As predicted (|Δ| ≤ 0.05).
* `SyntheticRLV2`: **+0.100 [+0.004, +0.194]**. Detected, sign as predicted (greedy favours us),
  but **twice the pre-registered bound**. The pre-registration is scored as WRONG on this row.

**Why the two disagree is measurable, not speculative.** The `argmax_match_rate` instrument says
what "temperature 1.0" actually *does* to each policy:

| at T = 1.0 | plays its own argmax | i.e. the perturbation is |
|---|---:|---:|
| `SmallRL` (13.9M) | **64.7%** (6,027/9,320; 0.623–0.680 across half-cells) | 35% of decisions |
| `SyntheticRLV2` (200.9M) | **88.0%** (4,191/4,765; 0.877–0.882) | 12% of decisions |

`SmallRL`'s action distribution is far flatter, so T = 1.0 is a large behavioural change for it and
a small one for the 200M model. Turning sampling **on** helps `SmallRL` against us (we drop from
0.630 to 0.520 at home when it goes greedy — i.e. greedy `SmallRL` is *stronger*) and hurts
`SyntheticRLV2` (we rise from 0.410 to 0.500). The mechanism is consistent: for a weak, uncertain
policy the argmax is a genuinely better action than a draw from a flat distribution; for a strong,
confident policy the distribution is already nearly the argmax and sampling only injects mistakes at
the 12% of decisions where it deviates. Either way — **"T = 1.0" names a different amount of noise
for every model, so it cannot be a fixed measurement protocol.**

### 4.2 The team-set effect is NOT DETECTED for either model, and that is the surprise

The pre-registration predicted **home − away between +0.05 and +0.20 and the largest of the three
effects**, on the double argument that our 719-team pool is our training distribution and
`competitive` is a set Metamon's authors say it has overfit to. That prediction is **WRONG**:

* `SmallRL`: **−0.030 [−0.125, +0.066]** — the point estimate is on the *away* side.
* `SyntheticRLV2`: **+0.090 [−0.006, +0.184]** — right sign, right magnitude band, but the CI
  touches zero, so NOT DETECTED at the registered standard.

Neither model gives us a detectable home-field advantage on the pool we train on. Read against the
prior it was registered against, this says our conditioning on our own 719-team pool buys much less
transferable advantage than the amortization-gap literature in this project would suggest — it is a
free-standing corroboration of the standing finding that **count dominates conditioning** and that
what the trunk learns from the pool is not team-specific structure. It is one measurement at n = 200
per side per model and it is not, on its own, a claim about the conditioning programme.

A real and stateable team-set effect **does** show up in game SHAPE: the away cells run longer in
every one of the four model×regime combinations (44.8–48.8 mean turns vs 36.3–42.7 at home). The 20
`competitive` teams are stall-leaning expert sample builds (Skarmory / Blissey / Swampert cores), and
all four 250-turn forfeits and the one tie in this battery are in cells that the 20-team set or the
sampling regime made long.

### 4.3 The de-risk's numbers do not survive the matched regime

| comparison | de-risk (mixed: us greedy, them T=1.0, home teams) | matched here | Δ [Newcombe 95%] |
|---|---|---|---|
| `SmallRL`, vs the matched **greedy** home cell | 0.742 (89/120) | 0.520 | **−0.222 [−0.341, −0.094]** |
| `SmallRL`, vs the matched **t1.0** home cell | 0.742 | 0.630 | −0.112 [−0.232, +0.011] |
| `SyntheticRLV2`, vs the matched **greedy** home cell | 0.583 (35/60) | 0.500 | −0.083 [−0.234, +0.075] |
| `SyntheticRLV2`, vs the matched **t1.0** home cell | 0.583 | 0.410 | **−0.173 [−0.321, −0.014]** |

The pre-registration predicted every matched cell would land 0–9 pp below its de-risk comparator.
For `SyntheticRLV2` that is right (−8.3 pp on the like-for-like greedy cell). For `SmallRL` it is
**badly wrong**: −22.2 pp, with the CI clear of zero. The mixed regime was worth far more against the
15M model than our own +8.9 pp figure would price it at — which is §4.1's mechanism again, since the
de-risk's `SmallRL` opponent was the one whose T = 1.0 is a 35% perturbation.

**Standing consequence: `0.742` and `0.583` should not be quoted again without the regime beside
them.** They are a mixed-regime read on home teams, and the honest summary of this model against
Metamon is the four cells in §3.

---

## 5. Hazards — each is a finding

### H-A 🚨 OUR vendored poke-env fork makes an EMPTY 7th Pokémon out of a trailing blank line, and the symptom is a HANG

`src/poke_env/teambuilder/teambuilder.py:43` splits a Showdown paste on `"\n\n"` and skips a chunk
only when it is **exactly** `""`:

```python
for ps_mon in team.split("\n\n"):
    if ps_mon == "":
        continue
    mons.append(TeambuilderPokemon.from_showdown(ps_mon))
```

Every one of Metamon's `competitive` gen3ou files ends `"...\n\n\n"`, so the final chunk is `"\n"` —
not `""` — and becomes an empty Pokémon. `Gen3Teambuilder` packs 7 entries and Showdown rejects the
team:

```
|popup|Your team was rejected for the following reasons:||||- You may only bring up to 6 Pokémon (your team has 7).
```

**Three things make this the expensive class of defect.**

1. **Upstream poke-env 0.8.3.3 parses the same file correctly** (6 mons, measured). This is the exact
   mirror of de-risk hazard H1, with the parsers' roles reversed — there, upstream was wrong about a
   nickname line and our fork was right. **Two independent disagreements between the two packages
   have now been found on team FILES, in opposite directions.** Any team file crossing between the
   projects must be normalised, in both directions.
2. **`validate_teams_locally` passes it**, because it validates the *paste*, and the defect is
   created by the *pack*. No existing gate can see it.
3. **The symptom is a stall, not an error.** The rejected challenge never becomes a battle, so the
   series simply sits there. It killed the first pilot at game 2 after quietly biasing the team draw
   toward the few files that happened to end without a blank line.

*Fixed in this harness* by `.strip()`ing every team text and by a **throwing guard at the
construction site** — `run_gen3ai_side.py` refuses to start if any packed team does not have exactly
6 Pokémon, and names the offenders. **Not fixed in the fork**; the one-line correction is
`if not ps_mon.strip(): continue`, and it belongs in `TECH_DEBT_BACKLOG.md` rather than in a
measurement branch.

### H-B 🚨 OUR 250-turn forfeit DESYNCHRONISES Metamon's challenge stream, and Metamon answers it with unbounded recursion

`metamon/rl/metamon_to_amago.py:321-325`:

```python
except Exception as e:
    print(e)
    print("Force resetting due to long-tail error")
    self.reset()
    next_tstep, reward, terminated, truncated, info = self.step(action)   # <-- recurses
```

When our side forfeits at turn 250, the battle ends while Metamon's env is mid-step. Its next
`step` raises `RuntimeError: Battle is already finished, call reset`, and the handler above
force-resets (which sends a fresh challenge) and then calls itself. It never converges: the observed
result is ~985 recursion levels and a `RecursionError` that kills the Metamon process with `rc = 1`,
losing that half-cell's timing and regime record. From the moment of the forfeit onward the two
sides' battle logs are no longer in step, so the **positional join** — the only join available,
because Metamon's `Battle ID` is a random 10-digit string (de-risk H3) — breaks.

**The evidence that the join is otherwise sound, and that the damage is bounded.** Counting only
games played *before* the first forfeit in their own half-cell: **690 games, exactly ONE
side-disagreement** — and that one is the tie, which Metamon's boolean `won` field cannot represent
(de-risk H3). Every one of the other 49 disagreements in the battery is downstream of a forfeit. So
the positional join is right, and `sides_disagree` is a working alarm rather than noise.

**4 forfeits in 800 games (0.5%)** — two in greedy cells, two in t1.0 cells — hit 3 of the 16
half-cells, and 1 phantom (≤2-turn) battle per affected cell is the visible wreckage.

### H-C The three desynchronised half-cells did NOT move the result

A desync corrupts the *join*; the question that matters is whether it corrupts the *games*. Our
side's poke-env flags are per-battle authoritative regardless of the join, and each half-cell has a
clean partner playing the identical cell at the opposite role. Pooled:

| | n | our wins | win rate |
|---|---:|---:|---:|
| the 3 desynchronised half-cells | 150 | 70 | 0.467 |
| their 3 clean role-partners | 150 | 67 | 0.447 |

**Δ = +0.020 [−0.092, +0.131]** — NOT DETECTED, and the point estimate is 2 pp, against cells whose
own CIs are ±10 pp. Excluding the forfeits and phantoms outright moves no cell by more than
1.0 pp (the largest is `SmallRL` greedy·away, 0.650 -> 0.660 on n = 97) (`summary.json`'s `sensitivity_excl_timeouts` per cell). The tables in §3 are reported as
played; §3's `cap` and `ph` columns are the disclosure, and UNDERSTANDING rule 12's separate bucket
is `sensitivity_excl_timeouts`.

### H-D `MetamonDiscrete` CLIPS probabilities, so no temperature can express greedy

`clip_prob_low=0.001`, `clip_prob_high=0.99`, applied **after** the temperature division and
**before** renormalisation. On the 9-way `MinimalActionSpace` the argmax action tops out at
`0.99 / (0.99 + 8×0.001) ≈ 0.992` however cold the temperature. Anyone who tries to get a
deterministic Metamon by passing `--temperature 0.01` gets a policy that plays a random non-argmax
action about once every 125 decisions and never learns that it did. Greedy is `sample=False`, and
nothing else.

### H-E Everything the de-risk recorded still holds

H1 (nickname-free team files), H2 (`VanillaAttention` — every Metamon transformer is unrunnable on
CPU as shipped), H3 (random `Battle ID`, tie booked as a loss, positional join), H4 (the hardcoded
`ws://localhost:8000` rebind), H5 (unflushed prints) all applied unchanged and all were carried by
this harness. **Zero** team rejections, **zero** `battle_event.classify` raises, **zero**
`ShowdownConnectionError`s, **zero** illegal-action fallbacks (`n_defaults` = 0 across all 800
games) and **zero** stale-decision re-decides. The only tracebacks in the battery are H-B's.

---

## 6. Cost

Under a load average of 25–31 on 16 cores, with a live training arm on the box.

| | `SmallRL` | `SyntheticRLV2` |
|---|---:|---:|
| games | 400 | 400 |
| total wall (our side, 8 half-cells) | 1,204 s | 2,440 s |
| **seconds per game** | **3.01** | **6.10** |
| our median decision | 20.5–57.9 ms | 64.9–79.8 ms |
| Metamon median decision | 12.2–17.0 ms | 55.5–112.6 ms |

Both are cheaper per game than the de-risk measured (2.30 / 8.47 s) — contention differs, and the
de-risk's warning stands: these are ceilings for a quiet box, not model measurements. The 200M
model's multi-minute build is paid once per half-cell and is not in the per-game figure; amortise it
over more games in any recurring use.

---

## 7. How the pre-registration scored

`PREDICTION.md` was committed before the first game (`80be38f7`). Scored honestly:

| | prediction | outcome |
|---|---|---|
| **P1** PRIMARY, `SyntheticRLV2` greedy·away: 0.42–0.52, bar NOT cleared | | **RIGHT** — 0.420, bar not cleared |
| **P2** `SmallRL` greedy·away: 0.60–0.72, bar cleared | | **RIGHT** — 0.650, bar cleared |
| **P3** temperature effect |Δ| ≤ 0.05, NOT DETECTED; if nonzero, favours us in greedy by ≤ 0.08 | | **HALF-WRONG** — right for `SmallRL` (−0.010, not detected) and right in SIGN for `SyntheticRLV2`, but +0.100 is double the registered bound |
| **P4** team set home − away positive, +0.05…+0.20, the LARGEST effect | | **WRONG** — not detected for either model, and negative in point estimate for `SmallRL` |
| **P5** every matched cell 0–9 pp below its de-risk comparator | | **HALF-WRONG** — right for `SyntheticRLV2` (−8.3 pp), badly wrong for `SmallRL` (−22.2 pp) |
| **P6** greedy cells longer, and any forfeit fires in a greedy cell | | **WRONG** — 4 forfeits, 2 greedy and 2 t1.0; greedy is longer away and shorter at home |
| **P6b** 0 parse failures, 0 timeouts, 0 tracebacks, 0 illegal-action fallbacks | | **RIGHT on the protocol** (0 / 0 / 0 outside H-B), **WRONG on tracebacks** — H-B produced three crashed processes |

Two of seven right outright, three half-right, two wrong. The two that matter most — the primary
row's value and its verdict — were called correctly and in advance.

---

## 8. Recommendation: the recurring baseline runs GREEDY-vs-GREEDY

The rule was registered in advance (`PREDICTION.md` §5): default to greedy, switch to T = 1.0 only
if greedy costs more than ~2% of games to forfeits/ties or visibly collapses the effective sample
size. **Greedy costs 2 forfeits in 400 greedy games (0.5%), t1.0 costs 2 in 400 (0.5%), and neither
regime collapsed anything** — distinct team pairs per 100-game cell are 87–100 in both. The rule
selects greedy, and three further arguments now support it:

1. **It is the protocol every other strength number in this project is taken under.** `ladder.json`
   is greedy-vs-greedy with symmetric builders; `--temperature 0` is `play.py`'s documented
   measurement setting; the eval-sentinel regime went greedy on 2026-09-07. A Metamon number taken
   at T = 1.0 cannot be read beside any of them.
2. **T = 1.0 is not a fixed yardstick across opponents** (§4.1). It perturbs `SmallRL` at 35% of
   decisions and `SyntheticRLV2` at 12%, so a "temperature 1.0 baseline" silently changes its
   difficulty when the opponent model changes — and if Metamon ships a new policy, the regime moves
   under us with no flag to read. Greedy is the same operation for every policy.
3. **Determinism narrows a fixed-n read**, which matters when the whole point is to detect
   movement in our own model between milestones.

**The standing caveat, stated rather than buried.** A deterministic policy in a simultaneous-move
hidden-information game is exploitable *in principle* — a best response to a fixed strategy is a
pure strategy, and against one of those the exploiter's payoff is unbounded by the Nash value. It is
exploitable **only by an opponent that adapts**, and neither side here adapts. Metamon is a
transformer over the battle's own token sequence: its hidden state gives it memory *within a
battle* — it can condition on what we have revealed this game, our switch pattern, a move we have
already used — and `amago`'s rollout loop resets that hidden state on `done`, so it carries **nothing
whatever across battles**. It cannot mine a fixed opponent over 100 games, cannot build a
counter-strategy, and cannot notice that we are deterministic. Our model carries even less: it
conditions only on the observation. So greedy-vs-greedy is safe *for this pair*, and the property
that makes it safe is an implementation fact about Metamon that could change in a release.

**Therefore:**

* **Recurring, every eval milestone: `SmallRL`, greedy-vs-greedy, 100 games per team set.** Cheap
  (3.0 s/game, ~5 minutes for 100), and at 0.520 home / 0.650 away it is a live signal in both
  directions rather than a floor.
* **Milestone reference: `SyntheticRLV2`, greedy-vs-greedy, both team sets.** It is *ahead of us*,
  which is what a reference should be — 0.420 away, 0.500 home.
* **Report BOTH team sets, always.** They are not redundant here: they disagree in sign for
  `SmallRL`, and the away set is the one no dial of ours is set to.
* **Keep one T = 1.0 cell on milestone runs only**, as the standing cheap check that determinism is
  not costing us something — not as the headline.
* **Before any recurring use, close H-B.** A 250-turn forfeit currently kills the Metamon process and
  desynchronises the log join. The harness fix is to bound each Metamon process to fewer battles and
  restart it, or to drive the pair battle-by-battle; the honest alternative is to keep counting the
  forfeits and reporting them, as this pass did.

---

## 9. What this measurement cannot say

* **Nothing about Metamon on Metamon's own Showdown pin.** Both clients played on ours, 13 commits
  behind its bundled submodule. A comparison against numbers Metamon produced on its own server is
  out of scope, as it was in the de-risk.
* **Nothing about exploitation.** §8's caveat: neither side adapts, so nothing here is evidence that
  greedy is safe against an adapting opponent.
* **Nothing about which model is stronger in general.** 100 games resolves ~±10 pp at best. Every
  cell near even stays NOT DETECTED however the point estimate reads, and the pooled per-model
  numbers (0.590 and 0.410) mix four deliberately different conditions and are a summary, not a
  rating.
* **Nothing about our team pool's conditioning value beyond this pair.** §4.2's null is measured
  against two Metamon policies on one away set of 20 teams.

---

## 10. What is in this directory

| File | What |
|---|---|
| `PREDICTION.md` | the pre-registration, committed before the first game (`80be38f7`) |
| `README.md` | this |
| `games.jsonl` | 800 merged per-game rows — both sides' views, the regime each side actually used, the team each side drew |
| `summary.json` | every cell, every effect, every half-cell's verification and failure counts |
| `run_all.sh` | one model's whole 2×2 as 8 role-balanced half-cells |
| `run_cell.sh` | one half-cell end to end, either role order |
| `run_metamon_side.py` | Metamon as one client: port rebind, CPU pin, attention override, **regime set + verified**, seeded team draw |
| `run_gen3ai_side.py` | our checkpoint as the other client — `play.py`'s own path, plus `--team-dir`, seeded draws, the H-A guard, and the regime observer |
| `analyze.py` | merge + Wilson + Newcombe + the integrity counters |
| `raw/<half-cell>/` | per half-cell: `cell.json`, both timing/verification JSONs, both team-draw sequences, Metamon's battle-log CSV |

---

## 11. Ledger paragraph — ready to append (do not append it from here)

> **2026-09-14 · THE METAMON BASELINE AT A MATCHED REGIME — the de-risk's 0.742 / 0.583 do not
> survive it, and "temperature 1.0" is NOT a comparable regime across models.** Pre-registered 2×2
> (`PREDICTION.md` committed at `80be38f7` before the first game): sampling regime {greedy·greedy,
> T=1.0·T=1.0, BOTH sides matched} × team set {our 719-team pool, Metamon's own 20-team
> `competitive` gen3ou set — the set its README names as the paper's human-ladder set and says
> Metamon has overfit to}, 100 games per cell, role balanced inside every cell, 800 games on our
> pinned Showdown (`e0551883f`, port 9350), Metamon @ `0a00a759`, CPU-only.
> **PRIMARY ROW (greedy-vs-greedy on THEIR teams): `ai_v12_02_winprob_critic` @ 75M is NOT BETTER
> than `SyntheticRLV2` — 0.420, Wilson 95% [0.328, 0.518], the registered LB > 0.50 bar NOT
> cleared** (predicted 0.42–0.52 and not cleared; called correctly in advance). Against `SmallRL`
> the same row is **0.650 [0.553, 0.736] — BETTER**, bar cleared. `SyntheticRLV2` leads us in three
> of four cells (0.500 / 0.420 / 0.410 / 0.310); `SmallRL` trails in all four (0.520 / 0.650 /
> 0.630 / 0.560). **The de-risk's mixed-regime reads were inflated: −22.2 pp [−34.1, −9.4] against
> `SmallRL`** on the like-for-like greedy home cell (0.742 → 0.520) and −8.3 pp [−23.4, +7.5]
> against `SyntheticRLV2` (0.583 → 0.500); **quote those two numbers only with their regime
> attached.** **THE TRANSFERABLE FINDING:** our own +8.9 pp eval-regime figure is an ASYMMETRY term
> and does NOT transfer to a symmetric regime change — the matched temperature effect is
> −0.010 [−0.105, +0.086] for `SmallRL` (NOT DETECTED) and +0.100 [+0.004, +0.194] for
> `SyntheticRLV2` (detected, double the pre-registered bound). The signs differ because **"T = 1.0"
> is a different amount of noise per model**: measured over 14,085 sampled decisions, `SmallRL` plays its own
> argmax 64.7% of the time at T=1.0 and `SyntheticRLV2` 88.0%, so the same nominal setting perturbs 35%
> vs 12% of decisions. **SURPRISE / PRE-REGISTRATION MISS: the team-set effect is NOT DETECTED for
> either model** (`SmallRL` −0.030 [−0.125, +0.066], point estimate on the AWAY side;
> `SyntheticRLV2` +0.090 [−0.006, +0.184]) — our 719-team training pool buys no detectable home
> advantage over a 20-team set we have never trained on, a free-standing corroboration of
> count-dominates-conditioning. Role (challenger = p1) is NOT DETECTED either (+0.050 / −0.020).
> Regimes were VERIFIED per decision on both sides, never assumed: Metamon greedy is
> `Agent.get_actions(sample=False)` with `argmax_match_rate` = 1.0000 in every greedy half-cell and
> 0.62–0.88 in the sampling half-cells as the positive control. **TWO NEW HAZARDS.** (H-A) **OUR
> vendored poke-env fork makes an EMPTY 7th Pokémon out of a trailing blank line**
> (`teambuilder.py:43` skips a split chunk only when it is exactly `""`, and every Metamon
> `competitive` file ends `"\n\n\n"`); Showdown rejects the team and the series **HANGS** —
> `validate_teams_locally` passes it because the defect is created by the PACK, and upstream
> poke-env 0.8.3.3 parses the same file correctly. This is de-risk H1 with the parsers' roles
> REVERSED: two disagreements between the packages on team FILES, in opposite directions. The
> one-line fix is `if not ps_mon.strip(): continue`; not applied here (tech debt). (H-B) **OUR
> 250-turn forfeit desynchronises Metamon's challenge stream and Metamon answers it with UNBOUNDED
> RECURSION** (`metamon_to_amago.py:321` catches "Battle is already finished", force-resets and calls
> itself; ~985 levels then `RecursionError`), killing the process and breaking the positional join.
> 4 forfeits in 800 games hit 3 of 16 half-cells; **the join is otherwise exact — 690 games before
> any forfeit produced exactly ONE side-disagreement, and that one is the tie Metamon's boolean
> field cannot express** — and the 3 desynchronised half-cells match their clean role-partners at
> +0.020 [−0.092, +0.131], so the result stands. Also: `MetamonDiscrete` clips probabilities to
> [0.001, 0.99] AFTER the temperature division, so **no temperature can express greedy** (~0.992
> argmax ceiling). Cost under load 25–31 on 16 cores: **3.0 s/game (`SmallRL`), 6.1 s/game
> (`SyntheticRLV2`)**, CPU-only. **RECOMMENDATION: the recurring Metamon baseline runs
> GREEDY-vs-GREEDY** — it is the protocol `ladder.json` and every other strength number use, and
> T=1.0 is not a fixed yardstick across opponents; `SmallRL` every milestone (both team sets),
> `SyntheticRLV2` as the milestone reference, one T=1.0 cell as a standing check, and H-B closed
> first. Full measurement:
> `designs/research_state/measurements/metamon_matched_regime_2026-09-14/`.
