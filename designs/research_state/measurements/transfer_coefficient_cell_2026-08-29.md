# THE TRANSFER-COEFFICIENT CELL — probe K's §6 decisive test

*Measured 2026-08-29, 09:58–12:39 PDT · **8,100 games** (4,050 paired units × 2 arms) / 200,769
decisions · CPU-only, 3 shards at `nice 15` (~3 cores), BLAS pinned · 2 h 41 m real elapsed
(7.73 h summed battle wall) · `models/` read-only · **zero errors, zero timeouts, zero unfinished**.*

Registered in the ledger entry landed at `2af60c2` ("PROBE K … DISPATCHED — K's own §6 decisive
test"). Data: [`transfer_coefficient_cell_2026-08-29.json`](transfer_coefficient_cell_2026-08-29.json);
scoring script `transfer_coefficient_cell_report.py` beside it; analysis plan pre-registered in
`tmp/tcell/PREREG.md` before the main run.

> **Probe K's own record had not landed on `main` when this was written** (checked at launch and
> again at scoring; `main` had advanced to `8b83cff` with no `overrule_diagnosis_*` file). K's
> figures below are therefore quoted from the **ledger entry `2af60c2`**, which is the only
> committed source for them at this time. If K's record lands later and its numbers differ, the
> denominator of every `tau` here moves with them — the numerator does not.

---

## 1. Verdict

**FULL TRANSFER IS REJECTED, AND THE ZERO IS NOT THE MIRROR'S FAULT.** Removing *both* of the
suspects this cell was built to remove — the checkpoint (now the exact 24M weights K judged) and
the population (now the eval roster, not a mirror twin) — does **not** make the dividend appear.
Against a no-search control on identical dice and identical teams, defensive search wins
**+0.20 pp [−0.39, +0.79]** — a null at a resolution that would have caught ±0.6 pp.

**The transfer coefficient is `tau = 0.17`, with a 95% interval of `[−0.34, +0.68]`.** It
**excludes 1.0** and **includes 0.0**: at most about two-thirds, and plausibly none, of the
per-decision win-probability advantage arrives at the scoreboard. That is the finding the cell
exists to produce.

**And the cell turned up something the registration could not have anticipated: the population
changed the STRATEGY, not just the scoreboard.** Against the roster the triage gate refuses to
search **92.6%** of decisions (mirror: 75.2%), and the searcher overrules **0.245 times per game**
against the mirror's **2.207 — a 9.0× smaller dose**. So "run it off-mirror and the dividend
appears" fails twice over: the per-overrule yield does not transfer, *and* off-mirror there are
almost no overrules to yield anything.

| registered reading (ledger `2af60c2`, scored, never adjusted) | outcome |
|---|---|
| **R1 — transfer is fine**: `A−B ≈ +(overrules/game × per-decision gain)`, "roughly +5–12 pp" | **REJECTED**, and its pp band was **UNREACHABLE BY CONSTRUCTION** — the subject's own recorded `win_rate_vs_bots` at 24M is 0.9162, so the arithmetic ceiling on A−B is +8.4 pp and a +5–12 pp band could never have been scored on this population. Re-stating R1 in the form the cell *can* test — full transfer at the roster's OWN measured overrule rate, i.e. `+1.16 pp` — that value lies **outside** the measured interval `[−0.39, +0.79]`. R1 is refuted on its own arithmetic, not merely unsupported. |
| **R2 — compounding/selection destroys it**: `A−B ≈ 0` | **SELECTED.** +0.20 pp [−0.39, +0.79], centred on the null, at ±0.59 pp. |
| **R3 — intermediate: report the transfer coefficient** | **REPORTED: `tau = 0.17 [−0.34, +0.68]`** (and `0.13 [−0.38, +0.63]` on the treated units alone). Both intervals exclude 1 and contain 0. |

**The one-line finding: the two suspects this cell could remove are removed, and the dividend is
still absent — so of probe K's three, COMPOUNDING is the one left standing.** The ledger's own
consequent applies: the fix is the confirm mechanism (iteration 3, which was running beside this
cell) or overrule throttling, not another population.

---

## 2. The cell

**Checkpoint** `models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` — verified
**byte-identical** (md5 `df3d5620…`) to `eval_traces/step_24000000/snapshot.zip`, i.e. the exact
weights probe G labelled and probe K re-judged, so G/K/this-cell compose. Iteration 2 ran
`checkpoints/checkpoint_9995088_steps.zip` (~10M): **the checkpoint suspect, removed.**

**Population** the **EVAL ROSTER** — the battery's default `--opponents`, i.e.
`eval_opponent_names()`: the 9 fixed scripted bots, 450 paired games each. Iteration 2 ran
`--opponents self`: **the population suspect, removed** (see §6 for what the roster is *not*).

**Two arms, one invocation per shard, matched by construction.**

| arm | what |
|---|---|
| **A** | `--arm honest --budget 1 --root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0` — **iteration 2's configuration verbatim** |
| **B** | `--arm base` — the same network with search structurally off, playing the masked argmax. The literal control, not a re-implementation of one |

`game_seed` and `team_pair` are functions of `(opponent, game_index, --games-seed 7)` alone, so a
unit is the same pinned sim seed and the same team draw in both arms, and **the two arms play the
identical battle until the first overrule.** The paired difference is therefore a statement about
the overrules and nothing else.

```
python -m main.search_dividend <ckpt> --arm base --arm honest --budget 1 \
  --root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 \
  --defensive-confirm 0 --defensive-contested-deadline-s 3.0 \
  --games-start <lo> --games 150 --games-seed 7 \
  --battle-timeout-s 1800 --battle-idle-s 120
```

**Sharding.** Three processes over disjoint game-index windows ([0,150), [150,300), [300,450)).
The scoring script *asserts* disjointness rather than trusting the launch line, and asserts the
rows carry exactly the two arms.

**Timeout hygiene.** Zero timeouts, zero errors, 8100/8100 finished. Longest game **148.7 s**
against the 1800 s livelock backstop and the 120 s idle-wedge detector. No game was ever scored
as anything but a played outcome.

---

## 3. The headline

| | n pairs | A (search) | B (policy) | **A − B** | 95% CI | discordant |
|---|---|---|---|---|---|---|
| **ALL 9 — the primary** | **4050** | **0.9273** | **0.9253** | **+0.0020** | **[−0.0039, +0.0079]** | 158 (83 A / 75 B) |
| deterministic 7 *(post-hoc, §4)* | 3150 | — | — | +0.0005 | [−0.0060, +0.0069] | 116 |
| `hard4` *(pre-declared)* | 1800 | — | — | −0.0039 | [−0.0138, +0.0060] | 90 |
| equal-weight per opponent | 9 opps | — | — | +0.0020 | [−0.0055, +0.0095] | — |

**3892 of 4050 pairs (96.1%) are outcome-identical** — the two arms mostly play the same game,
because the gate refuses to search and the race refuses to overrule.

### The transfer coefficient

`E_naive = (overrules per game, measured HERE) × (+0.0474 per-decision, probe K)`.

| | overrules/game | `E_naive` | realized A−B | **tau** | tau 95% (from A−B's CI) | excludes 1? | excludes 0? |
|---|---|---|---|---|---|---|---|
| all units | 0.2449 | +0.0116 | +0.0020 | **0.17** | **[−0.34, +0.68]** | **yes** | no |
| treated units only (≥1 overrule) | 1.6478 | +0.0781 | +0.0100 | **0.13** | **[−0.38, +0.63]** | **yes** | no |

Carrying K's own CI on the per-decision gain instead moves `tau` only to [0.11, 0.38] — **the
realized delta's interval is much the wider of the two**, which is why it is the one quoted.

⚠️ **`tau` is a diagnostic ratio against a stated benchmark, not an unbiased estimator.** The
additive accounting is first-order by construction: win-probability gains at different decisions
of one game are not independent and do not literally sum. What the number supports is the
comparison to 1 (full transfer) and to 0, both of which it makes.

---

## 4. The integrity check — and the defect it found

The design's own falsifier: **in a unit where A never overruled, the two arms must be the same
battle, so the paired difference must be exactly 0.** This is a positive verification of the
matching rather than an assumption about it, and it is conditioning on a *pre-treatment* variable
— whether A's FIRST overrule happens is a function of the prefix the two arms share exactly.

| zero-overrule units | n | A−B | discordant |
|---|---|---|---|
| **the 7 deterministic bots** | **2693** | **exactly 0.0000** | **0** |
| the staller family | 755 | +0.0026 | 4 |

**It PASSES on the deterministic seven — 2693 pairs, delta exactly zero, not one divergence — and
FAILS on exactly two bots. That confinement is what identifies the cause, and it is a real
defect:** `Gen3StallerPlayer` and `Gen3StallerV2Player` decide whether to Protect with
`random.random() < _PROTECT_PROBABILITY` (`agents/opponents.py`) — a draw from the **process-wide**
`random` module that both players share. The searched arm interleaves its `choose_move`
differently (arm A awaits an executor; arm B runs inline), so the coin can land differently in the
two arms with no overrule involved. This is the project's documented "two players share the global
`random`" trap, surfacing here on the opponent side.

**It is unbiased NOISE, not a bias** — 3 divergences favoured A, 1 favoured B; it inflates the
discordant count in both directions and widens the interval rather than moving it. The sensitivity
is the `deterministic7` row above: dropping both bots moves the headline from +0.20 pp to
**+0.05 pp** and the verdict not at all. **Reported, not repaired** — repairing it (a per-player
RNG for the scripted bots) is a change to the shared eval opponents and belongs to whoever owns
that surface, not to this cell.

---

## 5. The rate table — the POPULATION changed the strategy

Over the **89,422** decisions the strategy handled (of 100,397; the rest are forced switches and
non-move-selection rounds):

| | iteration 2 (mirror twin @ ~10M) | **this cell (roster @ 24M)** | ratio |
|---|---|---|---|
| FORCED by the gate | 0.7515 | **0.9262** | — |
| RACED | 0.2485 | **0.0738** | 0.30× |
| …SEPARATED (of raced) | 0.4542 | **0.3379** | 0.74× |
| …KEPT (of raced) | 0.2200 | **0.1876** | — |
| …OVERRULED (of raced) | 0.2342 | **0.1502** | 0.64× |
| …FUTILITY (of raced) | 0.5458 | **0.6621** | — |
| **overrule rate, all decisions** | 0.0582 | **0.0111** | **0.19×** |
| **overrules per game** | 2.2069 | **0.2449** | **0.11× (9.0× fewer)** |
| rounds per race | 13.17 | **10.48** | 0.80× |
| eliminated per race | 5.35 | **4.68** | — |
| mean search s / raced decision | 2.278 (of 3) | **2.531 (of 3)** | — |
| contested decisions / game | 9.42 | **1.63** | 0.17× |
| search s / game | 21.46 | **4.13** | — |
| mean battle wall s / game | 24.44 | **5.60** | — |

Three readings:

1. **The gate is doing exactly what it was designed to do, and against bots that means almost
   nothing to do.** Probe H's operating point forces a decision whose `|P(win) − 0.5| ≥ 0.15`. On
   a roster the subject beats 92.6% of the time, most positions are decided, so 92.6% of decisions
   are forced and only 1.63 per game are even contested. **The dose is 9× smaller off-mirror.**
   A search that never fires cannot pay a dividend, and this is a *mechanism* result independent
   of the transfer question.
2. **The per-overrule yield still does not transfer.** The treated-units row isolates it: on the
   602 units where A overruled at least once, at 1.65 overrules per game, the paired gain is
   +1.00 pp [−2.93, +4.92] against a naive +7.81 pp. Full transfer is excluded there too.
3. **`separated_of_raced` fell to 0.338 from 0.454, and the two causes are ENTANGLED** — see §6 on
   contention. Do not read that number as a pure population effect.

### The overrule-count gradient — suggestive, and NOT a result

| A's overrules in the unit | n | A − B | 95% CI |
|---|---|---|---|
| 0 *(the integrity check)* | 3448 | +0.0006 | [−0.0006, +0.0017] |
| **≥ 1 — the clean causal contrast** | **602** | **+0.0100** | **[−0.0293, +0.0492]** |
| exactly 1 | 401 | +0.0387 | [−0.0062, +0.0835] |
| exactly 2 | 117 | −0.0214 | [−0.1115, +0.0688] |
| 3 or more | 84 | −0.0833 | [−0.2139, +0.0472] |

The monotone decline (+3.9 → −2.1 → −8.3 pp) is **the shape compounding predicts**: the first
substitution is worth something, and each further one — made from a state the searcher itself
steered into — is worth less and eventually negative. **But it is not evidence, for two reasons
and both are structural.** No bucket's interval excludes zero. And more importantly, **only the
`≥1` row is a clean contrast**: "exactly 1" additionally conditions on A *not* overruling again,
and "≥2" conditions on A's own later behaviour — post-treatment variables, so those three rows
carry selection as well as effect. Reported because it is the right next thing to test, with a
design that does not condition on the outcome variable's own cause; not because it is settled.

---

## 6. Per-opponent, and the cuts

| opponent | recorded eval WR @24M | A | B | A − B | 95% CI | overrules/game | forced |
|---|---|---|---|---|---|---|---|
| random | 1.00 | 1.0000 | 1.0000 | +0.0000 | [0, 0] | 0.236 | 0.960 |
| heuristic | 0.89 | 0.9011 | 0.9210 | −0.0178 | [−0.0370, +0.0014] | 0.267 | 0.926 |
| heuristic2 | 0.91 | 0.9018 | 0.9089 | −0.0089 | [−0.0309, +0.0131] | 0.269 | 0.926 |
| staller | 0.96 | 0.9420 | 0.9351 | +0.0078 | [−0.0098, +0.0253] | 0.209 | 0.926 |
| staller_v2 | 0.96 | 0.9333 | 0.9267 | +0.0067 | [−0.0151, +0.0285] | 0.367 | 0.910 |
| aggressive | 0.92 | 0.9510 | 0.9310 | **+0.0200** | **[+0.0044, +0.0356]** | 0.160 | 0.935 |
| aggressive_v2 | 0.88 | 0.8946 | 0.8795 | +0.0133 | [−0.0073, +0.0340] | 0.318 | 0.895 |
| setup_sweep | 0.93 | 0.9265 | 0.9267 | −0.0011 | [−0.0164, +0.0141] | 0.193 | 0.928 |
| setup_sweep_v2 | 0.88 | 0.9089 | 0.9111 | −0.0022 | [−0.0191, +0.0147] | 0.187 | 0.930 |

⚠️ **`aggressive` is the only cell whose interval clears zero, and it is a MULTIPLICITY artifact
until someone replicates it.** Nine independent tests against a true null produce at least one
nominal hit about 37% of the time; the two neighbouring cells (`heuristic` at −1.8 pp) point the
other way by a similar amount. It is listed because suppressing it would be worse, and it is
**not** claimed.

### Limits — each one is a real bound on what this cell licenses

- 🚨 **THE ROSTER IS SATURATED, AND THAT WAS UNAVOIDABLE, NOT A CHOICE.** At `win_rate_vs_bots` =
  0.9162 the arithmetic ceiling on A−B is +8.4 pp, so the registered +5–12 pp band was
  unreachable before the first game was played. The pre-registration recorded this before the run
  (`tmp/tcell/PREREG.md`) rather than discovering it afterwards. **A null here is a statement
  about a population with 8 pp of headroom** — it does not license "search cannot pay anywhere",
  and the `hard4` stratum (the 4 bots at ≤0.91) reads −0.39 pp [−1.38, +0.60], no different.
- 🚨 **THE ROSTER IS THE BOT HALF OF K's POPULATION, NOT ALL OF IT.** Probe G/K drew from eval
  traces containing **1134 bot cells and 1088 sentinel cells**. Pool sentinels are **not
  constructible as battery opponents** (`build_eval_opponents` is keyed to the 9 scripted specs
  and would `KeyError` on a snapshot), so this cell covers roughly half of the population K
  measured, and the harder half — self-play sentinels at 0.69–0.89 — is **absent**. The population
  suspect is therefore *reduced*, not eliminated. Closing it fully needs a sentinel-capable
  opponent path in the battery.
- ⚠️ **THE BOX WAS BUSY, AND THE 3 s DEADLINE IS WALL-CLOCK.** Load averaged **20.7 on 16 cpus**
  over the run (median 18.5, p90 36.6, max 43.5 — factor ≈1.30), against iteration 2's ~3. The
  race therefore bought less evidence per contested decision: **10.48 rounds vs 13.17**. So the
  `separated_of_raced` drop (0.454 → 0.338) is **population and contention entangled and this cell
  cannot separate them**. The deadline was being *spent* either way (2.53 s of 3 s, vs iteration
  2's 2.28 s), and the primary A−B is unaffected — both arms ran under the same load and arm B does
  no searching at all — but the rate-table comparison to iteration 2 carries this caveat.
- **A dose this small bounds what any per-overrule effect could buy.** At 0.245 overrules/game,
  even *full* transfer of K's +4.7 pp would be +1.16 pp on this roster. The cell can reject that;
  it cannot resolve, say, +0.4 pp.
- **One run, one checkpoint, one configuration.** Nothing here says the transfer coefficient is
  stable across generations, budgets, or leaf choices.

### Accounting

**8,100 rows** = 4,050 paired units × 2 arms; 450 units per opponent, exactly balanced. **0
unfinished, 0 errors, 0 timeouts.** 200,769 decisions (100,397 arm A / 100,372 arm B). Arm A
6.31 h summed battle wall (5.60 s/game), arm B 1.43 h (1.27 s/game). `n_changed` equals
`n_defensive_overruled` on every row (992 = 992), so the search never altered an action outside a
counted overrule — the second consistency check this cell runs on itself.

**Reproduction.** `transfer_coefficient_cell_report.py` beside this file takes the three shard
JSONLs and rewrites `transfer_coefficient_cell_2026-08-29.json` in full; every number above is in
it. The shard rows themselves are ~11 MB and are NOT committed — the JSON carries every statistic
derived from them, and the cell is re-playable from the invocation in §2 at `--games-seed 7`.
