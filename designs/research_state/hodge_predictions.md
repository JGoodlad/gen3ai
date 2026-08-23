# Hodge width (non-transitivity) — BASELINES + PRE-REGISTERED predictions

**Status:** measured and pre-registered **2026-08-23**, before any flywheel-era generation exists.
The predictions below were written when nobody knew the answer; that is the only thing that makes
them worth anything. **Edit only with new evidence, and say what the evidence was.**

**Why this file exists.** A scalar Elo is a *transitive* model by construction — it cannot see a
cycle. The Hodge decomposition splits every measured matchup flow into a **spine** (the part a
rating explains) and a **width** (the cyclic residue), and tests the width against the binomial
noise floor its own game counts imply. The spinning-top hypothesis (Czarnecki et al., 2020, *"Real
World Games Look Like Spinning Tops"*) says a real game's strategy space is **widest at MID skill
and pinches toward the Nash tip**: strong policies have fewer distinct ways to beat each other. If
that is true here, width is a *progress* signal that Elo cannot carry — and a width *movement*
during the flywheel era is a thing we should be reading against a registered number rather than
arguing about after the fact.

**Instrument** (shipped, model-free — arch drift is irrelevant, it fits win records only):

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.elo <run_dir> --out <scratch> --no-plot --hodge-bootstrap 300 --hodge-seed 0
```

Mechanism and scope rules: `src/agents/training/CLAUDE.md` → *Hodge decomposition — the SPINE and
the WIDTH*. Implementation + tests: `src/agents/training/hodge.py`, `hodge_test.py`.
Units: **1 logit = 400/ln10 ≈ 173.72 Elo**.

---

## 1. Baselines (measured 2026-08-23)

Every run below is a **completed 24M-step generation** with a full dense frozen ladder
(66/66 snapshot pairs measured) and 12 eval cycles. Graph shape is **identical** across all five —
**21 players / 174 edges / 814 triangles**, width scope = all 174 triangle-supported edges,
300 bootstrap reps, seed 0. Elo is the **`snapshot_ladder/ladder.json` rating of the 24M node**
(dense, ±20 at 95%), per the reading rules — *not* `eval/elo`.

| run | gen | step | Elo (ladder, 24M) | spine | width raw | **floor (null)** | **width EXCESS** | p | cyclic frac (adj) | sig 3-cycles | dense g/pair | date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai_v9_15_gen13_hb_events_stack_0817` | 13 | 24M | **2068 ± 20** | 909 | 39 | **24** | **31** | 0.0033 | 4.33% (2.66%) | 0 | 400 | 08-23 |
| `ai_v9_16_gen14_framedel_v91_0817` | 14 | 24M | **2024 ± 19** | 765 | 35 | **23** | **26** | 0.0033 | 4.04% (2.19%) | 0 | 400 | 08-23 |
| `ai_v9_18_gen15_v8rewards_0818` | 15 | 24M | **2065 ± 20** | 939 | 58 | **35** | **46** | 0.0033 | 6.27% (3.81%) | 3 | 100 | 08-23 |
| `ai_v9_19_gen16_mechanics_0819` | 16 | 24M | **2015 ± 19** | 921 | 49 | **35** | **35** | 0.0033 | 5.02% (2.45%) | 1 | 100 | 08-23 |
| `ai_v9_21_gen17_pfspoff_0820` | 17 | 24M | **2034 ± 20** | 894 | 48 | **36** | **33** | 0.0033 | 4.45% (1.94%) | 2 | 100 | 08-23 |

All widths in Elo. `p = 0.0033` is the bootstrap's floor `1/(B+1)` at B=300 — **no null replicate
reached the observed width in any of the five runs**, i.e. every generation's non-transitivity is
real and not sampling noise. All five are nonetheless overwhelmingly **spine**: cyclic energy is
**1.9–3.8%** of Σw·Y² after null adjustment.

### 1a. 🚨 The dense-ladder game count is a REAL confound — the raw table above is NOT a series

`width_rms = sqrt(Σw·R² / Σw)` with **`w = n·p(1−p)`** (the Fisher information of a logit). Game
counts therefore move **two** things at once: the noise floor (more games ⇒ lower floor) *and* the
relative weight of edge CLASSES in the same average. gen-13/14 played **400 games/pair** on the
dense snapshot ladder; gen-15/16/17 played **100**. Measured Σw share of snapshot-vs-snapshot edges:

| | gen-13 | gen-14 | gen-15 | gen-16 | gen-17 |
|---|---|---|---|---|---|
| Σw share, snap-vs-snap edges | 85.9% | 85.5% | 70.4% | 68.3% | 70.2% |
| floor (null width, Elo) | 24 | 23 | 35 | 35 | 36 |

**Matched-density re-measurement** (binomially thin gen-13/14's 400-game dense edges to 100, 12
independent thinning draws, 200 bootstrap reps each):

| run | as measured (400 g/pair) | **thinned to 100 g/pair** | floor after thinning |
|---|---|---|---|
| gen-13 | excess 31, floor 24 | **excess 51.9 ± 2.4** | 35.4 |
| gen-14 | excess 26, floor 23 | **excess 49.0 ± 2.8** | 35.0 |

So the apparent "gen-14 (26) < gen-15 (46)" ordering in the raw table is **mostly a game-count
artifact**, not a fact about the two ladders. The comparable series is:

| gen | Elo (24M) | width excess @ **matched 100 g/pair** | floor |
|---|---|---|---|
| 13 | 2068 | **52** (thinned) | 35 |
| 14 | 2024 | **49** (thinned) | 35 |
| 15 | 2065 | **46** | 35 |
| 16 | 2015 | **35** | 35 |
| 17 | 2034 | **33** | 36 |

**Read this as the baseline, and read it honestly**: at matched density the width declines
monotonically gen-13 → gen-17 (52 → 33, a 19-Elo drop against a ~3-Elo sampling sd) while anchored
Elo is **flat** (2015–2068, all overlapping within ±20). That is a decline *without* the rising-Elo
half of P1, so it is a **baseline observation, not a confirmation of anything below.** A live
alternative explanation for at least gen-17's share of it: gen-17 ran with **PFSP off**, which
narrows the opponent distribution and should mechanically reduce pool cyclicity.

### 1b. Sampling noise on the width estimator (sets every threshold below)

Parametric resample of every edge at its observed rate, 200 reps, re-decomposed end to end:

| | sd of `width_rms_excess` |
|---|---|
| gen-14 (400 g/pair) | **2.2 Elo** |
| gen-15 (100 g/pair) | **3.3 Elo** |
| gen-17 (100 g/pair) | **3.2 Elo** |

The bootstrap *null* itself is essentially deterministic (±0.5 Elo across seeds 0–3; the plug-in
null moves ±0.16). So the uncertainty is in the observed width, not in the floor.

**⇒ σ(difference between two runs) ≈ 4.6 Elo, and every threshold in §2 is set at ≥ 10 Elo ≈ 2σ.**

### 1c. Runs deliberately excluded

| run | why |
|---|---|
| `ai_v9_28_legAmatched_0823` | in flight — 1 eval row, no `snapshot_ladder/` |
| `ai_v9_2{2,3,4}_*substrate_on_*`, `ai_v9_25_E4_baitbot_0822` | 6 eval rows — probe/exploiter forks, not full generations |
| `ai_v9_2{6,7}_*_probe_0823` | 2–5 eval rows — probe forks |
| gen ≤ 12 | not measured here; older ladders predate the current promotion cadence and would need their own density check |

Arch drift does **not** exclude anything — `main.elo` and `hodge.py` fit win records, load no
checkpoint, and ran clean on all five.

---

## 2. PRE-REGISTERED predictions

Common rules for **every** prediction:

1. **Quote the EXCESS, never the raw width**, and **always re-state the floor beside it**. A raw
   width is a floor plus a signal; a floor that moved is not a result.
2. **Never compare across unmatched dense-ladder game counts.** Match the density (thin the denser
   run, as in §1a) or declare the comparison not taken. This is the confound that already voided
   the naive gen-13→gen-17 reading.
3. Fix `--hodge-bootstrap 300 --hodge-seed 0` and report `n_players / n_edges / n_triangles`. A
   graph-shape change is a separate confound and must be stated (all baselines are 21/174/814).
4. **Never narrate a mid-run width** — the live `eval/hodge_width_elo` is a trainee×bot×bot star
   read with a 35–60 Elo floor and was never significant on its own (gen-15: p between 0.13 and
   0.93 across 12 cycles). The offline dense-ladder read is the instrument.

---

### P1 — Width DECLINES as anchored Elo rises (the spinning-top ascent)

> **Prediction.** Across revolution-1-era generations, the floor-adjusted Hodge width **falls** as
> the anchored ladder Elo of the final snapshot **rises**. We are climbing the top's upper cone
> toward the tip, so there are progressively fewer distinct ways for strong policies to beat each
> other.

**Baseline:** width excess **33–52 Elo** against a floor of **35 Elo** at 100 g/pair (gen-13→17,
§1a). Take **gen-17 = 33 Elo** as the immediate reference point and **the 5-gen mean 43 Elo** as
the era baseline.

- **CONFIRMS** — a generation whose 24M ladder Elo is **≥ +30 above gen-17's 2034** (i.e. beyond
  the ±20 CI, non-overlapping) AND whose matched-density width excess is **≤ 23 Elo**
  (≥ 10 Elo = 2σ below gen-17's 33). Stronger form: the *rank correlation* over ≥ 4 matched-density
  generations between ladder Elo and width excess is negative with all pairs ≥ 10 Elo apart.
- **REFUTES** — Elo rises ≥ +30 while width excess **rises ≥ 10 Elo** (≥ 43) or stays inside
  33 ± 10. A flat width under a genuine Elo climb is a refutation of P1 *as stated*, not a
  "pending" result: it says our ladder is not on the pinching cone.
- **CONFOUND TO CHECK** — (a) dense-ladder games/pair (see rule 2 — the single biggest trap here,
  worth 20 Elo of width on its own); (b) **pool composition**: PFSP on/off and the number of
  sentinels change the opponent distribution the ladder measures, and gen-17 already shows a
  plausible PFSP-off deflation; (c) graph shape — a run with a different snapshot count is a
  different graph.

---

### P2 — A WIDENING at flat-or-rising Elo is ENTRY INTO THE FAT MID-BAND, not regression

> **Registered interpretation, committed in advance so it cannot be re-argued post hoc.** If a
> flywheel-era generation shows a **width increase at flat or rising Elo**, we read it as **a new
> strategic dimension opening** — the population has acquired a genuinely new axis of
> counterplay (team-coverage expansion, a new archetype the pool can now pilot, an exploiter's
> best-response mode being absorbed) — i.e. the ladder has moved *outward* on the spinning top,
> not *down*. **It is NOT a regression, and it is NOT to be treated as a bug to be fixed.**

Rationale: the spinning top is fat in the middle *because* the mid-band has many mutually-beating
strategies. Adding a strategic dimension while holding Elo means the game we are playing got
bigger, which is exactly what team-coverage expansion is supposed to do. Interpreting that as
regression would push us to optimize width *down*, which at fixed Elo means **collapsing** onto a
narrower strategy set — the opposite of what the flywheel is for.

- **CONFIRMS the P2 reading** — width excess **rises ≥ 10 Elo** over the era baseline while ladder
  Elo is flat (within ±20) or rising, **and** at least one independent corroborator of "new
  dimension" fires: a new **significant 3-cycle** whose members are snapshots from different
  training phases (baseline count: 0 / 0 / 3 / 1 / 2 for gen-13…17), or a measured expansion in
  team-archetype coverage (`data/teams/gen3_team_archetypes.json` classes actually piloted).
- **REFUTES the P2 reading** (⇒ it really was regression) — width rises ≥ 10 Elo **and** ladder Elo
  **falls ≥ 30** below gen-17's 2034, **or** the width increase is carried entirely by early
  snapshots (a spine that got worse at the bottom, inflating residuals) rather than by the top
  quartile of the ladder. Check that by re-fitting the width over the top-6 snapshots only.
- **CONFOUND TO CHECK** — games/pair (rule 2, again — a drop from 400 to 100 g/pair *manufactures*
  a ~20 Elo widening); the **number of players in the graph** (adding snapshots adds triangles and
  changes Σw); and whether an exploiter or sentinel entered the pool mid-run, which is P3's
  territory and must be attributed there first.

---

### P3 — Exploiters RAISE cyclic_fraction transiently; the distillation fold FLATTENS it

> **Prediction.** Adding exploiters to the pool raises `cyclic_energy_fraction` **by
> construction** — a best response to a specific opponent is precisely a strategy that beats what
> it targets and loses to what it does not, which is a cycle. The subsequent **distillation fold**
> (teacher → generalist) then **flattens** it back down. If both halves fire, that is a direct,
> testable signature of **"distillation is the averaging/convergence operator"** in this system:
> it is the step that projects the cyclic content back onto the spine.

**Baseline:** null-adjusted cyclic energy fraction **1.94% – 3.81%** across gen-13…17
(gen-17 = 1.94%, gen-16 = 2.45%, gen-15 = 3.81%). Era mean **2.6%**. Sampling sd on the fraction is
≈ **0.4 pp** (from the §1a thinning draws: 4.65 ± 0.36% and 4.47 ± 0.50%), so **1.0 pp ≈ 2σ**.

- **CONFIRMS** — measured on the SAME run at two points, exploiters-in vs post-fold:
  cyclic fraction (null-adjusted) **rises ≥ 1.0 pp** on the ladder that contains the exploiter
  snapshots, **then falls ≥ 1.0 pp** after the distilled generalist is folded back and the ladder
  is re-measured with the fold's snapshots included. Corroborator: the exploiter node should
  appear in ≥ 1 individually-significant 3-cycle.
- **REFUTES** — the exploiter-in ladder's cyclic fraction is inside baseline ± 1.0 pp (best
  responses were absorbed without creating a cycle — either the "exploiter" is not actually a best
  response, or the pool was already covering it), **or** the fraction rises and then *fails to
  fall* after the fold (≥ 1.0 pp of the rise persists), which would say distillation is **not** the
  averaging operator and the cyclic content is being retained rather than projected out.
- **CONFOUND TO CHECK** — (a) **the fold changes the player set**, so the exploiter-in and
  post-fold graphs are *different graphs*; re-measure the pre-fold ladder restricted to the
  post-fold player set, or state that the comparison is unmatched; (b) games/pair again — a newly
  promoted node starts at fewer measured pairs than the incumbents, which lowers its Σw and can
  *hide* its own cycle; run `snapshot_ladder --backfill` to density-match before reading;
  (c) an exploiter is often stronger than what it targets, so part of its residue is **spine**, not
  width — always report spine spread beside the fraction (baseline 765–939 Elo).

---

## 3. Re-derivation

```bash
export PYTHONPATH=$PYTHONPATH:src
# the table in §1, one run at a time (writes only to the scratch --out, models/ stays read-only)
python -m main.elo /home/goodlad/dev/gen3ai/models/ai_v9_21_gen17_pfspoff_0820 \
    --out /tmp/hodge_out/gen17 --no-plot --hodge-bootstrap 300 --hodge-seed 0
# the ladder Elo the table quotes
python -c "import json;print(json.load(open('<run>/snapshot_ladder/ladder.json'))['ratings']['24000000'])"
```

The matched-density numbers in §1a come from `hodge.edges_from_run(run)` → binomially thin every
edge with `n > 100` to `n = 100` → `hodge.hodge_decompose(..., bootstrap=200)`, 12 draws.
The §1b sampling sd comes from resampling every edge at its observed rate, 200 reps.
