# HOW BIG IS THE BATON PASS BIAS? — H18, sized — 2026-09-22

**The question, from the backlog row it discharges**
([`metamon_obs_faithfulness_2026-09-22`](../metamon_obs_faithfulness_2026-09-22/README.md) §8):
Metamon's upstream poke-env loses every Baton-Passed stat stage, so every anchor win rate this era
is **optimistic for us by an unmeasured amount**. Measure it.

**The answer, in one sentence:**

> 🚨 **The patch WORKS and the bias is SMALLER than the instrument can resolve as nonzero: the
> Baton Pass defect is worth at most ±0.02 on our anchor win rate, on BOTH team sets — including
> one built to be 20/20 Baton Pass.** Fixing Metamon's parser changed the outcome of **21 of 800
> matched battles**, and **every single one of those 21 contained a Baton Pass that actually
> carried state** — a perfect causal footprint, zero false positives — yet the changes went both
> ways and cancelled: away **+0.005 [−0.009, +0.019]**, enriched **−0.0025 [−0.020, +0.015]**,
> both **NOT DETECTED**. **Registered branch (c).** The standing anchor numbers do not need a
> correction footnote; they need this bound attached.

Pre-registered in [`PREDICTION.md`](PREDICTION.md), committed **`6cc2f304`** before the first
measured game. **The registered prediction was right** (|bias| < 0.05 away; 0.00–0.08 against us
on enriched — the enriched point estimate landed at −0.0025, marginally in OUR favour).

---

## 0. THE 2×2

`metamon:SmallRL` (ckpt 40) greedy-vs-greedy, arm **W** `ai_v13_02_flywheel_winprob` @
**75,005,952**, `--server rust`, CPU-only, 400 games per cell, role-balanced, matched seeds.
Win rate is OURS.

| arm | Metamon's poke-env | team set | n | W/L/T | win rate | Wilson 95% | mean turns |
|---|---|---|---:|---|---:|---|---:|
| **U** | as installed (0.8.3.3, **unpatched**) | away — `competitive`, 8/20 carry BP | 400 | 222/178/0 | **0.5550** | [0.506, 0.603] | 48.2 |
| **P** | shadow copy, **Baton Pass fixed** | away | 400 | 220/180/0 | **0.5500** | [0.501, 0.598] | 48.8 |
| **U** | as installed (**unpatched**) | **enriched — 20/20 carry BP** | 400 | 262/138/0 | **0.6550** | [0.607, 0.700] | 30.1 |
| **P** | shadow copy, **fixed** | enriched | 400 | 263/137/0 | **0.6575** | [0.610, 0.702] | 30.2 |

### The contrast, and the bias

**The bias on any unpatched anchor number is `−(P − U)`**: patching Metamon removes the handicap,
so our win rate should fall, and how far it falls is how much the published number was flattering
us.

| team set | P − U (pre-registered, unpaired Newcombe 95%) | **P − U (PAIRED — the powered read)** | McNemar exact p | **BIAS on the unpatched number** | verdict |
|---|---|---|---:|---|---|
| away (8/20 BP) | −0.0050 **[−0.0736, +0.0636]** | **−0.0050 [−0.0189, +0.0089]** | 0.727 | **+0.005 [−0.009, +0.019]** | **NOT DETECTED** |
| enriched (20/20 BP) | +0.0025 **[−0.0631, +0.0681]** | **+0.0025 [−0.0152, +0.0202]** | 1.000 | **−0.0025 [−0.020, +0.015]** | **NOT DETECTED** |

**Registered branch: (c) — not detected on either team set.** The bound the data actually support
is the paired one: **|bias| < 0.02 on both team sets**, and the point estimate on the enriched set
has the *wrong sign* for the hazard (the patch marginally helped us).

🚨 **The paired interval is ~3.5× tighter than the one registered, and that is a property of the
DESIGN, not a choice made after seeing the answer.** The two arms are handed the same
`--seed-base`, the same `--team-seed` and the same team files, so they play **the same 400
battles**; 392 of 400 (away) and 387 of 400 (enriched) ended identically. An unpaired interval
throws that pairing away and prices 800 independent coin flips that were never independent. The
pre-registered unpaired number is reported first and unchanged, because it is what was registered;
the paired number is reported beside it because it is the honest read of this design, and **both
say NOT DETECTED**, so nothing turns on which is preferred.

---

## 1. 🚨 THE PATCH IS NOT INERT — it acts in exactly the right places

The result above would be worthless if the patch had quietly failed to load, and a null from an
instrument that never fired is the failure class this project names `vacuous_tests_and_guards`.
Three independent things say it fired.

### (i) The positive-control pair, before the campaign

[`baton_pass_patch_test.py`](baton_pass_patch_test.py) is the fork's carry-over test
(`src/poke_env/battle/baton_pass_carryover_test.py`) ported to the upstream API, run under both
interpreters:

| assertion | arm U (installed) | arm P (shadow) |
|---|---|---|
| boosts ride the pass | **FAIL** | **PASS** |
| a Substitute rides the pass | **FAIL** | **PASS** |
| the OPPONENT's pass is tracked | **FAIL** | **PASS** |
| negative stages ride the pass | **FAIL** | **PASS** |
| *invariant:* a plain switch still clears | PASS | PASS |
| *invariant:* a phaze drag never carries | PASS | PASS |

**Arm U must fail** — otherwise the arms are not distinct and the whole cell is vacuous. Both
invariants hold on both arms, so the patch is the fix and not a blanket.

### (ii) Which copy loaded is a RECORDED fact

Each arm's interpreter wrapper prints `poke_env.__file__` into the peer log before the driver
starts. Over the campaign's eight peer processes:

```
4 x [h18] ARM=U poke_env.__file__=/home/goodlad/miniconda3/envs/metamon/lib/python3.10/site-packages/poke_env/__init__.py
4 x [h18] ARM=P poke_env.__file__=<scratch>/h18/patched/poke_env/__init__.py
```

### (iii) 🚨 The causal footprint is PERFECT — 21 of 21

| team set | battles | outcome identical | **outcome CHANGED** | of those, containing a pass that carried state |
|---|---:|---:|---:|---:|
| away | 400 | 392 | **8** | **8 / 8 (100%)** |
| enriched | 400 | 387 | **13** | **13 / 13 (100%)** |

**Not one of the 21 divergences happened in a battle without a state-carrying Baton Pass.** If the
patch were a no-op, there would be 0 divergences; if it were a blanket that perturbed unrelated
parsing, divergences would be spread across the ~700 battles with no such pass. Neither is what
happened. The patch changes Metamon's behaviour **only** where the mechanism says it can — and
then the changes go both ways: away `b = 5 / c = 3`, enriched `b = 6 / c = 7`.

> **This is the finding underneath the null.** The defect is real, it is reachable, it flips real
> games — and it is *not* a systematic handicap. Losing a passed +2 hurts Metamon on the turns it
> would have pressed the advantage and helps it on the turns it would have over-committed to one;
> at 800 battles those two cancel to within ±0.02.

---

## 2. THE BIAS PER EXPOSURE — a per-game number is diluted

A bias quoted per game is averaged over every game the mechanism never appeared in. The captures
give the honest denominator ([`bp_exposure.py`](bp_exposure.py), over all 1,600 battles):

| | away | enriched |
|---|---:|---:|
| teams carrying Baton Pass | 8 / 20 | **20 / 20** |
| Baton Pass **switch events** | 269 (U) / 263 (P) | 399 (U) / 388 (P) |
| …of which carried a **stat stage** | 51 / 49 | 66 / 59 |
| …of which carried a **copyable volatile** | 19 / 19 | 80 / 80 |
| **…carrying ANY state (the real exposure)** | **70 / 68** | **146 / 139** |
| games with any Baton Pass | 107 / 400 (26.8%) | 230 / 400 (57.5%) |
| **games with a STATE-CARRYING pass** | **51 / 400 (12.8%)** | **99 / 400 (24.8%)** |

**Restricted to the exposed games only** — the games the defect can possibly change:

| team set | arm U | arm P | P − U on exposed games | verdict |
|---|---|---|---|---|
| away (n = 51) | 27/51 = 0.529 | 25/51 = 0.490 | **−0.039 [−0.224, +0.150]** | NOT DETECTED |
| enriched (n = 99) | 64/99 = 0.647 | 65/99 = 0.657 | **+0.010 [−0.121, +0.141]** | NOT DETECTED |

**So the per-EXPOSURE bound is ±0.22 (away) and ±0.14 (enriched)** — much weaker than the per-game
bound, because only a quarter of even a 20/20-Baton-Pass team set actually passes state. ⚠️ **This
restriction conditions on something the treatment can influence** (whether a pass happens at all),
so it is a secondary read, reported for the denominator it supplies rather than as the headline.
It happens to be benign here: the exposed **game sets are identical between the arms** (51 vs 51,
99 vs 99, same battle tags), because whether a pass occurs is settled by the shared teams and
seeds, not by the patch.

---

## 3. 🚨 A SECOND FINDING — the VOLATILE half of the mechanism is real, and it fires

The faithfulness pass reported the volatile half (Substitute, Leech Seed and the rest of
`BATON_PASS_COPIED_EFFECTS`, which `copyVolatileFrom` also carries) as **"did not fire in these 20
battles"**. At 1,600 battles it fires, and it is not rare:

* **80 of 399** enriched-set Baton Passes carried a **copyable volatile**, against 66 that carried
  a stat stage. On this team set the volatile half is the **LARGER** exposure.
* Three of the 13 enriched divergences are **volatile-only** — `battle-gen3ou-2`, `-66` and `-160`,
  each a Baton Pass that carried a live **Substitute** and no boost at all. A boost-only exposure
  scan calls those three unexplained; they are the volatile half, and chasing them is what found it.

**Consequence for hazard H18 as written:** it is named for the stat stages, and the stat stages are
only part of it. An exposure denominator that counts boost-carrying passes alone **undercounts the
true exposure by 37% on the away set and by more than half on the enriched set**. The bias bound in
§0 is unaffected — the patch carries both halves, so both are inside the measured contrast.

---

## 4. INTEGRITY — every cell, against the conditions fixed in `PREDICTION.md` §3

| cell | `status` | `regime_verified_decisions` | `argmax_match_rate` | `team_source_asymmetry` | forfeits | `n_defaults` | `n_redecides` | distinct teams | `peer_clean` |
|---|---|---|---|---|---:|---:|---:|---:|---|
| U away | **OK** | **true** | 1.0000 | false | 3 / 400 | 0 | 0 | 19 | ⚠️ false |
| P away | **OK** | **true** | 1.0000 | false | 3 / 400 | 0 | 0 | 19 | ⚠️ false |
| U enriched | **OK** | **true** | 1.0000 | false | 0 | 0 | 0 | 20 | true |
| P enriched | **OK** | **true** | 1.0000 | false | 0 | 0 | 0 | 20 | true |

Every condition met. Regime verified over **10,829 + 10,342 decisions** in the away arms alone,
`sample_kwargs = [False]` on both halves of every cell. Forfeits are 0.75% of the away cells,
far under the 25% inconclusive bar, and **identical between arms** — the two away arms hit the
250-turn limit on the same 3 battles.

`peer_clean = false` on the two away cells is **not** grounds to discard them (hazards H16/H17):
it is Metamon's post-game `RecursionError`, raised in teardown after the last decision of the last
game, with every game played and recorded. See §6 **H-A** — it is a new variant worth naming.

### Cross-check against the standing number

`U away` is the standing Tier-B cell run unmodified: **0.5550 [0.506, 0.603], n = 400**, against
the banked **0.5533 at n = 1200** (`anchor_ab_continuation_2026-09-20`). A **0.0017** agreement on
an independent 400-game draw is a free corroboration that nothing in this harness moved the cell.

---

## 5. WHAT THIS CHANGES — and what it does NOT

### The anchor numbers stand as written. No correction footnote.

Branch (b) would have required re-stating every era anchor number with a correction. **It was not
reached.** The registered branch is **(c)**, so the honest amendment to hazard **H18** is not a
number subtracted from each reading but a **bound attached to all of them**:

> Metamon's Baton Pass defect biases our anchor win rate by **less than ±0.02 against `SmallRL` at
> greedy-vs-greedy**, on both the `competitive` away set (8/20 Baton Pass) and on a 20/20
> Baton Pass set built to maximise the exposure — measured, 2×400 games per team set, matched
> seeds, patched vs unpatched. It is **not** a reason to doubt any standing anchor reading.

For scale: this bound is **4× smaller than the eval-draw floor (0.020 → here ±0.02 is the whole
interval), 5× smaller than the three-seed RUN-level floor (0.110)**, and **13× smaller than the
budget effect on this cell** (~0.38 at 10M → ~0.64 at 75M). It is below the noise of every
comparison the anchor is used for.

### Three things it still cannot say

* **Nothing about `SyntheticRLV2`.** A 200.9M policy may lean on passed setup differently; this
  measures `SmallRL`.
* **Nothing at an exposure above 24.8%.** Even a 20/20 Baton Pass team set only produced a
  state-carrying pass in a quarter of its games. A bound per *exposure* is only ±0.14.
* **Nothing about whether the anchor should be patched.** It should not, and this result removes
  the last reason to want to: the SOP's position (a patched Metamon is a different opponent from
  the one its authors publish) now costs us a bias of **under 2 pp**, which is a cheap price for
  comparability.

---

## 6. HAZARDS — each is a finding

### H-A 🚨 Metamon's post-game `RecursionError` fires in the ACCEPTOR half too — H17's scope is too narrow

`EXTERNAL_ANCHORS_SOP.md` **H17** records four occurrences, **all in the half where Metamon
challenges**, and `main.anchors` stamps `peer_recursion_upstream` only on a complete
`peer_challenge` half. Both away cells here died of exactly that recursion in the
**`ours_challenge`** half — where **Metamon is the ACCEPTOR** — so the tool left
`peer_exit_notes` **empty** and the run reads as an undiagnosed dirty exit:

```
Force resetting due to long-tail error / Battle is already finished, call reset   (x ~1000)
maximum recursion depth exceeded
[peer] FATAL RecursionError: maximum recursion depth exceeded while calling a Python object
[peer] REGIME CHECK regime=greedy sample_kwargs=[False] (ok=True)
       argmax_match_rate=1.0 over 10829 decisions (ok=True) | peer_error_free=False
```

Same unbounded `self.reset(); return self.step(action)` handler, same ~1,000 frames, same
after-the-last-decision timing, **opposite role.** The games stand (400/400 recorded, regime
verified on every decision). ⚠️ **Do not read H17's role clause as a diagnosis**: an acceptor-half
recursion is the same upstream defect and gets no note, so a reader who trusts `peer_exit_notes`
to be non-empty will treat a known cause as unknown. The instrument needs widening; that is a
backlog item, not a change made here.

### H-B 🚨 An ARM SELECTION MUST REFUSE, because it silently ran the wrong one once

The first pilot here was launched without `$GEN3AI_METAMON_PYTHON` exported. Nothing failed:
`main.anchors` fell back to the config's `opponents.metamon.python`, ran the **installed
(unpatched)** interpreter, printed `status OK`, and produced a clean 10-game "arm P" number that
was arm U. It was caught **only** because the wrapper prints `poke_env.__file__` and that line was
absent from the log.

The fix, and the rule: **`opponents.metamon.python` in [`anchors_h18.json`](anchors_h18.json) is a
path that does not exist**, so a cell launched without the arm selection dies by name —
`AnchorConfigError: … does not exist on this box. Override it with $GEN3AI_METAMON_PYTHON` —
instead of quietly measuring the control twice. **Any A/B whose arms are selected by an environment
variable needs the unset case to be a refusal, not a default.** The two pilots are disclosed in
`PREDICTION.md` §5 and neither entered the cells.

### H-C ⚠️ A boost-only exposure scan UNDERCOUNTS the mechanism

§3. `copyVolatileFrom` carries the volatiles as well as the stat stages, emitting nothing for
either. On the enriched set the volatile half is the **larger** exposure (80 events vs 66), and it
accounted for 3 of the 13 divergences. The first version of `bp_exposure.py` counted boosts only
and called those three games unexplained; the scan now counts both, and the corrected
`games_with_a_state_carrying_pass` is the denominator §2 uses.

### H-D ⚠️ The unpaired interval is the wrong instrument for a matched-seed A/B — but it is what was registered

Pre-registering "Newcombe" cost ~3.5× resolution, because the design is paired and Newcombe is not.
Both are reported and both say NOT DETECTED, so nothing turns on it here — but a future matched-seed
A/B should **register the paired statistic**, and should register it **before** discovering that the
arms share battle tags. ⚠️ The pairing is a property of `--seed-base` + `--team-seed` being equal;
it does **not** survive a cell whose arms draw different teams, and a paired test applied to
unpaired arms is a much worse error than an unpaired test applied to paired ones.

### H-E ℹ️ The shared conda env is never edited — the shadow copy is the mechanism

`$GEN3AI_METAMON_PYTHON` points at a two-line wrapper that exports `PYTHONPATH` to a **copy** of
upstream `poke_env` and `exec`s the real interpreter. `main.anchors.peers` sets `PYTHONPATH=""` in
the peer env (correctly — hazard H11, two `poke_env` packages), so the wrapper must re-export it
*after* that, which is exactly what `exec`ing from a shell script does. The installed package is
byte-unchanged; [`metamon_poke_env_baton_pass.diff`](metamon_poke_env_baton_pass.diff) is the whole
of the difference, **105 lines added and 8 removed across 5 files**.

---

## 7. SETUP — every hash

| Thing | Value |
|---|---|
| Our checkout (worktree `h18`) | **`6cc2f304`** — the pre-registration commit; a moving `main` cannot enter the campaign |
| Our Showdown pin | `deps/pokemon-showdown` @ **`e0551883f`** |
| Metamon checkout | `/home/goodlad/dev/metamon` @ **`0a00a759`** |
| `poke-env`, arm U | **upstream 0.8.3.3**, `…/envs/metamon/lib/python3.10/site-packages/poke_env` — untouched |
| `poke-env`, arm P | a **copy** of the same, + the ported fix, first on `PYTHONPATH` |
| Arm **W** (ours) | `models/ai_v13_02_flywheel_winprob/final_model.zip` @ **75,005,952**, rung `explicit_zip`, loader `bare` |
| Opponent | `metamon:SmallRL` **ckpt 40** (13.9M), `MinimalActionSpace`, poke-env backend |
| Regime | **greedy both sides**, verified per decision, `argmax_match_rate` 1.0000 everywhere |
| Transport | `--server rust`, `ws_frontend@6cc2f304+rust:sim_bridge`, ports **9510/9520/9530/9540**, each stopped by its own PID. **:8000 and :8001 were never touched** |
| Seeds | `--seed-base 20260922`, `--team-seed 20260914` — **identical across arms** |
| Compute | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `OMP_NUM_THREADS=1`, ≤ 2 cells concurrent. **Nothing written under `models/`** |
| Wall clock | ~19 min per cell; 1,600 games total |

### The command (one cell; [`run_cell.sh`](run_cell.sh) is the exact form used)

```bash
GEN3AI_ANCHORS_CONFIG=<dir>/anchors_h18.json \
GEN3AI_METAMON_PYTHON=<dir>/python_{patched,unpatched} \
python -m main.anchors \
  --model models/ai_v13_02_flywheel_winprob/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset {home,away} --games 400 \
  --device cpu --server rust --port 95X0 --seed-base 20260922 --team-seed 20260914 \
  --capture-dir <out>/captures --out <out> --nice 15
```

### The enriched team set

The 20 pool teams drawn by `random.Random(20260922).sample(sorted(BP_teams), 20)` from the **157 of
719** pool teams carrying Baton Pass, named in [`PREDICTION.md`](PREDICTION.md) §1 and
[`bp_enriched_sample.json`](bp_enriched_sample.json). They are the pool's own **exported** pastes
(complete Hidden Power IV lines, nickname-free — hazards H2/H3), byte-identical for our side and
the peer, so `team_source_asymmetry` reads false. The `away` count is **8 of 20** `competitive`
teams; the faithfulness pass's "8 of 21" counted the set's `index.csv` as a team.

---

## 8. WHAT IS IN THIS DIRECTORY

| File | What |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the pre-registration, committed `6cc2f304` before the first measured game, including the disclosure of the two pilots |
| [`metamon_poke_env_baton_pass.diff`](metamon_poke_env_baton_pass.diff) | **the patch** — 5 files, +105/−8, our fork's fix ported to the upstream API |
| [`baton_pass_patch_test.py`](baton_pass_patch_test.py) | the positive-control pair: arm U must FAIL the carry-overs, arm P must PASS, both must hold both invariants |
| [`bp_exposure.py`](bp_exposure.py) | the exposure scan over the captured protocol — Baton Pass events, and which carried a stat stage, a copyable volatile, or either |
| [`analyse.py`](analyse.py) | the 2×2, the Newcombe contrasts, the exposure-restricted read and the paired/McNemar read |
| [`run_cell.sh`](run_cell.sh) | one cell, exactly as run |
| [`anchors_h18.json`](anchors_h18.json) | the anchors config for this campaign — a copy with three changes, and a deliberately-missing `python` so an unset arm REFUSES |
| [`bp_enriched_sample.json`](bp_enriched_sample.json) | the enriched draw with its seed and method |
| [`h18_results.json`](h18_results.json) | every number in §0–§2, machine-readable |
| `summary_{U,P}_{away,home}.json` | the four cells' own `summary.json` |

The captures (1,600 battles, ~270 MB) and `games.jsonl` live under the session scratch and are not
committed; `--seed-base 20260922 --team-seed 20260914` regenerates them exactly.

---

## 9. READY-TO-APPEND LEDGER PARAGRAPH

> **2026-09-22 — H18 SIZED: the Baton Pass bias is BELOW ±0.02 on our anchor win rate; branch (c),
> NOT DETECTED on either team set.** Pre-registered (`6cc2f304`) before the first measured game,
> discharging the `metamon_obs_faithfulness_2026-09-22` §8 backlog row. `metamon:SmallRL` ckpt 40,
> greedy-vs-greedy with both regimes verified per decision, arm W
> (`ai_v13_02_flywheel_winprob` @ 75,005,952), `--server rust`, `--seed-base 20260922
> --team-seed 20260914`, **400 games per arm per team set, 1,600 games**, CPU-only. Arms: **U** =
> Metamon's upstream poke-env 0.8.3.3 as installed; **P** = a SHADOW COPY of that package carrying
> our fork's 2026-08-23 Baton Pass fix, ported (5 files, +105/−8) — the shared env was never
> edited and each peer log records `poke_env.__file__`. Team sets: `competitive` **away** (8/20
> teams carry Baton Pass) and a **Baton-Pass-ENRICHED** 20-team draw from the 157/719 pool teams
> that carry it (`random.Random(20260922)`, teams named in the artifact). **Results — U away
> 0.5550 [0.506, 0.603] / P away 0.5500 [0.501, 0.598]; U enriched 0.6550 [0.607, 0.700] /
> P enriched 0.6575 [0.610, 0.702]**, all four cells `status OK`, `regime_verified_decisions true`,
> `argmax_match_rate 1.0000`, `team_source_asymmetry false`, `n_defaults 0`. **Contrast P − U:
> away −0.0050, Newcombe [−0.0736, +0.0636]; enriched +0.0025 [−0.0631, +0.0681] — both NOT
> DETECTED.** The arms play the SAME battles at matched seeds, so the PAIRED read is the powered
> one: away −0.0050 [−0.0189, +0.0089] (McNemar p = 0.73), enriched +0.0025 [−0.0152, +0.0202]
> (p = 1.00). **So the BIAS on an unpatched anchor number is +0.005 [−0.009, +0.019] (away) and
> −0.0025 [−0.020, +0.015] (enriched) — under ±0.02, i.e. below the 0.020 eval-draw floor and 13×
> below this cell's 10M→75M budget effect. The era's anchor numbers need NO correction; they need
> this bound attached.** 🚨 **The null is not an inert patch:** the ported fix changed the outcome
> of **21 of 800 matched battles and every one of the 21 contained a Baton Pass that actually
> carried state** (away 8/8, enriched 13/13), with the flips going both ways (away b=5/c=3,
> enriched b=6/c=7) — the defect is real and reachable but is not a systematic handicap.
> **Exposure, measured over all 1,600 captures:** a state-carrying pass occurs in **51/400 (12.8%)**
> of away games and **99/400 (24.8%)** of enriched games, so the per-EXPOSURE bound is only ±0.22
> / ±0.14. **Second finding — the VOLATILE half of the mechanism fires and is LARGER than the boost
> half on the enriched set** (80 volatile-carrying passes vs 66 boost-carrying; 3 of the 13 enriched
> divergences were Substitute-only), so H18's boost-named exposure UNDERCOUNTS it. **Third finding —
> Metamon's post-game `RecursionError` (H17) occurred in the half where Metamon ACCEPTS, not
> challenges**, so `main.anchors` left `peer_exit_notes` empty and a known upstream cause reads as
> undiagnosed; games unaffected (400/400 recorded, regime verified over 10,829 decisions). Control:
> `U away` reproduces the standing cell at **0.5550 vs the banked 0.5533 (n = 1200)**. Artifact:
> `designs/research_state/measurements/h18_baton_pass_bias_2026-09-22/`.
