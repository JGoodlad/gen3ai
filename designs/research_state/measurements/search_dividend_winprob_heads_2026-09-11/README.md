# THE SEARCH DIVIDEND OF THE CURRENT WIN-PROB VALUE HEADS

*Measured 2026-09-11 18:38 – 2026-09-12 00:52 UTC · three heads × four cells in the MIRROR ·
**5,556 battles**, 243,197 decisions · CPU-only, 9 (then 8) shards at `nice 15`, BLAS pinned ·
`models/` read-only · **zero timeouts, zero errors, zero unfinished games**.*

**The bar and every prediction were registered before the first cell:
[`PREDICTION.md`](PREDICTION.md).** Data: [`results.json`](results.json); every row archived beside
it (`rows_<cell>_<head>.jsonl.gz`); scripts [`run_battery.sh`](run_battery.sh),
[`run_extension.sh`](run_extension.sh), [`report.py`](report.py).

---

## 1. Verdict

**NO DIVIDEND, on any of the three heads, at resolving width.** The two 10M heads crossed the
pre-registered "real but unresolved" line at 400 pairs (0.5206 and 0.5150) and the extension
registered *before* that read was taken collapsed both onto the null: **ctrl10M 0.5059
[0.4914, 0.5205], lambda09 0.5031 [0.4894, 0.5169]** over 800 pairs, with the fresh, independent
second 400 pairs landing at **0.4913 for both**. The 73M head was on the null from the start
(0.4950 [0.4740, 0.5160]).

**And unguarded search on this head is a catastrophe: 0.2600 / 0.2600 / 0.2025 paired — it loses
three games in four against its own unsearched self.** The entire "defensive" apparatus (a triage
gate that refuses to search 75–80% of decisions, a CRN-paired race, a futility rule that refuses to
overrule without separation) is worth **+0.28 to +0.30** of paired win rate, and every bit of that
is *damage control*: it buys the harm back to zero and no further.

**The one-line finding: the promoted win-prob head is a WORSE leaf than the shaped critic it
replaced.** At the identical 3 s operating point where iteration 2's race separated on 45.4% of its
contests and certified 5.82% overrules, these heads separate on **11.9–15.0%** and certify
**1.1–1.6%** — four to five times fewer. The head that became the critic discriminates between
candidate actions *less* than the auxiliary win-prob head on a shaped-critic checkpoint did. That is
the behavioural shadow of the resolution and conditioning failures recorded in
`UNDERSTANDING.md` §4.2b.

## 2. The headline table

Paired mirror win rate (side-swap pairs; the team draw is common to a pair and cancels).
**The null is 0.50 by construction** — the searched side plays the SAME network with search
structurally off — **so the interval below IS the dividend's own CI.** All cells: `--opponents self`,
`--games-seed 7`, realized depth 1.

| cell (what it is) | ctrl10M @10M | lambda09 @10M | wp73M @73M |
|---|---|---|---|
| **`grid` @1 s** — naive search, every decision, no gate | **0.2600** [0.1985, 0.3215] | **0.2600** [0.1970, 0.3230] | **0.2025** [0.1487, 0.2563] |
| **`defA` @1 s** — defensive, uniform budget (iteration 1's point) | 0.4933 [0.4748, 0.5118] | 0.5133 [0.4974, 0.5292] | 0.5100 [0.4883, 0.5317] |
| **`defB` @3 s contested** — defensive (iteration 2's point) — *pre-registered read* | 0.5206 [0.4999, 0.5414] | 0.5150 [0.4954, 0.5346] | **0.4950** [0.4740, 0.5160] |
| **`defB` — REGISTERED EXTENSION, 800 pairs** *(the number to quote)* | **0.5059** [0.4914, 0.5205] | **0.5031** [0.4894, 0.5169] | not extended (on the null) |
| *verdict against the registered bar* | **NO DIVIDEND DETECTED** | **NO DIVIDEND DETECTED** | **NO DIVIDEND DETECTED** |
| `base` — both sides unsearched (the exact-50 control) | 0.5000, 8 pairs | 0.5000, 10 pairs | 0.5556, 9 pairs |

> **Three more heads (`strata`, `denseaux`, `cflabels`) were measured on 2026-09-12 against the
> L1/L2 leaf-quality bar registered that day — see [§11](#11-phase-23--three-more-heads-and-the-axes-question).
> They do not change anything below; they extend it, and they convict the L1 row as an instrument.**

n per head: 100 pairs (`grid`), 150 (`defA`), 400 → 800 (`defB`), 8–10 (`base`, stopped early — §7).
`grid` and `defA` spend 1 s per decision; `defB` grants 3 s to a *contested* decision only.

**Every head plays the same battles.** `game_seed` and `team_pair` are functions of
`(opponent, game index, --games-seed)` alone, so every head-vs-head and rung-vs-rung comparison
below is **paired at the battle level**, not two point estimates set side by side.

| paired contrast (delta's own CI) | value |
|---|---|
| defB: ctrl10M − lambda09 (800 shared pairs) | +0.0028 [−0.0165, +0.0222] NOT DETECTED |
| defB: ctrl10M − wp73M (400) | +0.0256 [−0.0024, +0.0536] NOT DETECTED |
| defB: lambda09 − wp73M (400) | +0.0200 [−0.0080, +0.0480] NOT DETECTED |
| defB − defA, within head (150 each) | +0.0317 [−0.0028, +0.0661] · +0.0133 [−0.0213, +0.0480] · +0.0083 [−0.0246, +0.0412] — all NOT DETECTED |
| defB − grid, within head (100 each) | **+0.2825** [+0.2054, +0.3596] · **+0.2800** [+0.1972, +0.3628] · **+0.3025** [+0.2360, +0.3690] — all DETECTED |

## 3. Against the registered bar

| bar (PREDICTION.md §3) | outcome |
|---|---|
| **SEARCH PAYS** — paired CI lower bound > 0.50 | **NOT MET on any head, at either width.** ctrl10M missed it by one ten-thousandth at the pre-registered 400 pairs (lower bound 0.4999) and regressed to 0.5059 [0.4914, 0.5205] on 800. |
| real but unresolved (point ≥ 0.51, CI straddles) | Where the pre-registered read landed for ctrl10M and lambda09 — and the reason the extension fired. **Resolved AGAINST the dividend.** |
| **NO DIVIDEND DETECTED** | **All three heads**, at ±0.015 (10M heads) and ±0.021 (73M). |
| SEARCH HARMS — CI upper bound < 0.50 | **all three `grid` cells**, decisively: upper bounds 0.3215 / 0.3230 / 0.2563. |

### The scored predictions

| registered prediction | outcome |
|---|---|
| 1. rung B: no dividend detected on all three heads, points in [0.48, 0.52] | **HELD** at the extended width: 0.5059 / 0.5031 / 0.4950, every CI straddling 0.50. (At the pre-registered 400 pairs two of three points sat above 0.51 — the prediction was briefly in doubt and the extension settled it.) |
| 2. ordering λ 0.9 ≥ ctrl10M, 73M ≥ 10M | **REFUTED in direction, NOT DETECTED in magnitude.** The best-calibrated head is not the best leaf (ctrl10M − lambda09 +0.0028 [−0.0165, +0.0222]) and the 73M head is the *worst* of the three by point estimate (ctrl10M − wp73M +0.0256 [−0.0024, +0.0536]). No pairwise CI excludes zero. |
| 3. grid beats its historical 0.2929 but still loses (0.35–0.48 predicted) | **REFUTED — it does not even beat it.** 0.2600 / 0.2600 / 0.2025 against the shaped-critic v9 cell's 0.2929. The win-prob leaf is no better, and on the 73M head worse, than the leaf that convicted search in August. |
| 4. overrule rate rises above iteration 2's 5.82% (8–15% predicted) | **REFUTED, badly, in the opposite direction: 1.41% / 1.12% / 1.64%** of all decisions (5.7 / 5.7 / 7.0% of raced). The smoke's 12.1% was a 4-battle artefact. |
| 5. forced fraction falls to 55–70% | **REFUTED: 75.3% / 80.2% / 76.5%** — right where the shaped critic's was (74–82%). |

Four of five registered predictions are refuted, and the four refusals point the same way: **this
leaf carries less usable signal than the one it replaced, not more.**

## 4. The mechanism — why the dividend is absent

| cell | raced | separated / raced | overruled (all dec.) | futility stops that were deadline-truncated | realized K worlds | arms scored | s / searched dec. | banked s / dec. |
|---|---|---|---|---|---|---|---|---|
| defA ctrl10M | 25.5% | **0.07%** | **0.00%** | 100% (0 genuine / 2,844) | 1.31 | 53.0 | 0.72 | 0.82 |
| defA lambda09 | 20.2% | **0.04%** | **0.00%** | 100% (0 / 2,420) | 1.26 | 50.9 | 0.72 | 0.86 |
| defA wp73M | 24.8% | **0.00%** | **0.00%** | 100% (0 / 2,537) | 1.23 | 47.9 | 0.70 | 0.83 |
| defB ctrl10M | 24.7% | 12.8% | 1.41% | 100% (0 / 13,021) | 5.06 | 191.3 | 2.60 | 0.85 |
| defB lambda09 | 19.8% | 11.9% | 1.12% | 100% (0 / 12,104) | 4.85 | 176.2 | 2.60 | 0.88 |
| defB wp73M | 23.5% | 15.0% | 1.64% | 100% (0 / 5,745) | 5.48 | 194.1 | 2.58 | 0.86 |
| *iteration 2 — v9 SHAPED critic, same 3 s* | *—* | ***45.4%*** | ***5.82%*** | *—* | *—* | *—* | *—* | *—* |

**Rung A is a PLACEBO, and that is a result rather than a waste.** At 1 s the race separates on
0.00–0.07% of its contests and **overrules nothing at all** — `n_changed` is exactly 0 across
37,909 decisions in the three defA cells. The searched side plays the policy's action on every
single decision. So defA measures the *harness's own null*, and it reads 0.4933 / 0.5133 / 0.5100:
**a literal no-op arm produces "0.51-ish" point estimates at n = 150.** That is the standing
demonstration of why a point estimate without its CI is not a reading — and it is exactly the shape
the 400-pair defB result turned out to have.

**At 3 s the race does work, and still barely fires.** Width per contested decision rises ~4× (K
1.3 → 5.1 worlds, 50 → 190 arm evaluations) and separation goes from ~0 to ~13%, but **every single
futility stop in every cell is deadline-truncated** (`futility_genuine` = 0 in 38,671 stops). The
race is *still* clock-limited at 3 s on this leaf, where the shaped critic at the same 3 s reached
45% separation. The leaf's per-arm differences are small relative to their own CRN-paired noise:
that is the behavioural definition of a low-resolution critic.

**The decorrelation ladder** — the fraction of side-swap pairs that SPLIT (the signature of two
pilots playing the same battle the same way):

| cell | split pairs | both won | both lost |
|---|---|---|---|
| `base` (both sides unsearched) | 100% · 100% · 89% | — | — |
| `defA` (search runs, changes **nothing**) | 94.7% · 96.0% · 92.7% | 2–5% | 0.7–3.3% |
| `defB` (changes 5.7%) | 82.3% · 83.9% · 81.0% | 8–9% | 8–9% |
| `grid` (changes 56–60%) | 38% · 36% · 34% | 3–8% | **55% · 56% · 62%** |

`grid`'s bottom-right cell is the whole finding in one number: **in 55–62% of pairs the searched
side lost BOTH orientations** — it is not unlucky, it is worse. And `defA`'s row shows that search
perturbs the game (5–7% of pairs stop splitting) even when it never changes a decision, which is
where a no-op arm's variance comes from.

## 5. What the `grid` cell says, and why it is the most important row

`--root-strategy grid` is search with no gate and no race: score every legal action on the leaf, play
the best. It changes the policy's action on **56.4% / 59.9% / 60.0%** of decisions and loses
**74% / 74% / 80%** of its games *against itself*.

| | ctrl10M | lambda09 | wp73M | v9 SHAPED critic (2026-08-24) |
|---|---|---|---|---|
| paired mirror win rate | 0.2600 | 0.2600 | 0.2025 | 0.2929 |
| action changed | 56.4% | 59.9% | 60.0% | not recorded in the quoted row |

This reproduces, on a **different critic era**, the August finding that convicted search: *"search
amplifies a SYSTEMATIC leaf error"* (`UNDERSTANDING.md` §3.5, bias not variance). **The win-prob
critic milestone did not move this meter** — if anything the 73M head moved it the wrong way.

⚠️ The cross-era comparison to 0.2929 is **independent-sample, not paired**: the two cells share
their seeds but not their team draws (§8.2), and they pilot different networks. What makes it
readable at all is that the mirror's null is 0.50 *by construction for both*, so each number is a
within-network statement about that network's own leaf.

## 6. The extension — the registered second look, and what it caught

Registered in [`PREDICTION.md`](PREDICTION.md) §6 at 19:59 UTC, **while rung B was 58% played and
before the cell closed**, with its rule and its final n fixed in advance: *any head whose 400-pair CI
straddles 0.50 with a point ≥ 0.51 is extended to exactly 800 pairs over fresh indices 400–799, and
both reads are published.*

| head | first look (pairs 0–399) | **second look (pairs 400–799, fresh)** | pooled (800) |
|---|---|---|---|
| ctrl10M | 0.5206 [0.4999, 0.5414] | **0.4913** [0.4709, 0.5116] | **0.5059** [0.4914, 0.5205] |
| lambda09 | 0.5150 [0.4954, 0.5346] | **0.4913** [0.4720, 0.5105] | **0.5031** [0.4894, 0.5169] |

**Both heads' fresh halves land BELOW the null.** The multiplicity caveat written into the rule is
moot in the direction it fired: the second look does not strengthen a positive, it erases one.
A +2 pp point estimate at 400 pairs with a lower bound at 0.4999 was noise, and the only reason that
is a *statement* rather than a *suspicion* is that the extension was committed to before the read.

## 7. The exact-50 control (and why it was cut short)

`--arm base` is the policy on BOTH sides with search structurally off — the harness's own no-effect
check. **26 of 27 pairs split exactly**; pooled over the three heads **0.5185 [0.4822, 0.5548]**,
consistent with the construction null and detecting no seat bias. The two 10M heads' cells are
exactly 0.5000 with zero variance.

**Honest limits:** the cell was stopped by hand at 8–10 pairs per head (explicit PIDs) after it
turned out to cost **~2.5 s per decision with search structurally off** — 57–351 s per battle, ~25×
an unsearched decision inside the defensive cells, which would have made the *control* five hours
more expensive than the arms it controls. At 27 pooled pairs it would not detect a seat bias of a
few points, so it is quoted as *consistent with* the null, never as a measurement of it. The
overhead itself is logged as a hazard (§8.8).

## 8. Hazards and findings about the instrument

Every one of these is a finding, not an aside.

1. **The row schema cannot identify its own cell.** A results row records `arm`, `budget`,
   `opponent`, `root_strategy`, `score_mode` — but **not the checkpoint**, not
   `--defensive-contested-deadline-s`, and not `--defensive-wp-margin`. Rung A and rung B are
   therefore *indistinguishable* inside a pooled file (identical arm/budget/opponent/strategy), as
   are two different heads. This battery keeps them apart by FILE NAME only
   (`<cell>__<head>__s<lo>.jsonl`), and `report.py` refuses overlapping shard windows. An
   append-only record whose rows do not name the model that produced them is one careless `cat`
   away from a fabricated cell.
2. **The team pool changed under the battery's reproducibility contract.** `team_pair` draws from
   the LIVE pool list, so a game index names the same *dice* forever but the same *teams* only
   within one pool version. `31cd5e3a` (2026-08-31) added 40 sample teams — TeamLoader went
   **679 → 719** — *after* every historical mirror cell. Consequence: the August cells and these
   cells share seeds but **not** team draws, so every cross-era comparison here is quoted as
   independent-sample. The within-2026-09-11 comparisons are unaffected and genuinely paired.
3. **A wall-clock budget is not a portable unit.** What 1 s buys depends on the box. These cells ran
   9 (then 8) shards on 16 cores at `nice 15`; the realized widths are published in §4 for exactly
   this reason, and the GPU was deliberately left idle — moving the forward to cuda would change
   what a second buys and break comparability with every registered cell. The extension was run at
   the same shard count as the cell it extends, for the same reason.
4. **`futility_genuine` = 0 everywhere.** Not one genuine (evidence-based) futility stop in 38,671.
   Read with §4: the strategy is still clock-limited at the registered operating point, so "the leaf
   carries no signal" and "the race never got to look properly" are not fully separated at 3 s. The
   honest statement is that **at every budget anyone has run it does not fire enough to matter, and
   when it fired four times as often (iteration 2, on a better-separating leaf) the win rate did not
   move either.**
5. **Search failures fall back to the policy** — `root_failed` 9–415, `search_error` 4–64,
   `prefix_gate_failed` 61–289 per cell, together ≤1.3% of decisions. All bias toward the null (a
   failed search plays the policy's action), so they cannot manufacture a dividend; they can only
   hide one.
6. **No arch drift.** All three checkpoints carry `gen3_critic_route_wave_v1`, the current
   `ARCH_SIGNATURE`; nothing refused. `--score auto` **resolved** to `win_prob` on every cell (all
   three runs are `--critic winprob`), announced at startup and recorded in every row's
   `score_mode` — `defensive.resolve_for_critic` behaved exactly as designed.
7. **`--search-impl rust` was NOT used.** The CLI's validated default for `open_root` is `node` and
   every registered cell ran on it; the rust search driver is still blocked on the materializer
   (ledger 2026-08-24: "~8× on the 26% sim share"). This battery is ~6 h of CPU that a working rust
   path would make ~1 h.
8. **`--arm base` costs ~2.5 s per decision while searching nothing** (§7) — a ~25× overhead over an
   unsearched decision in the defensive arms, with `realized` all zeros and `no_search` on every
   fallback. The cheapest possible cell is the most expensive per battle. Worth a look before anyone
   plans a large control cell.
9. **A `grid` cell's `beam` and `depth` are non-zero** (depth 1.004–1.007, beam 0.11–0.23): iterative
   deepening fires on a few percent of decisions even at 1 s. Depth-2 readings on this code are
   publishable (`gen3_search_depth2_chunk_gap_v1` is fixed), but these cells are depth-1 for all
   practical purposes and are reported as such.

## 9. What this means for the search-and-distill path

Per the registered reading (PREDICTION.md §5), the outcome is **NO DIVIDEND DETECTED** on every
head, at resolving width, with the one apparent positive erased by its own pre-registered second
look.

* **A search TEACHER is not worth building on this leaf.** A teacher that overrules 1.1–1.6% of
  decisions, at 2.6 s each, for an effect whose CI contains zero, has almost nothing to distill: the
  student would be trained on the policy's own actions ~99% of the time. The ceiling argument runs
  the same way — distillation recovers a *fraction* of a measured dividend, and a fraction of ~0
  is ~0.
* **The binding constraint is the LEAF, and this measurement dates it.** `UNDERSTANDING.md` §4.2b
  has the head resolving below the matched-stratum baseline on 60 of 60 rows and conditioning
  weakly; §3.5 has the disease as BLUR and search as an amplifier of systematic leaf error. This is
  that claim in behavioural units: **unguarded search on this head loses three games in four.**
* **The standing order from 2026-08-24 is discharged for this milestone** — *"Re-run this probe
  after each critic milestone; the mirror table IS the critic-resolution meter in behavioral
  units."* The meter reads: **the win-prob-critic milestone did not move it**, and by the
  separation/overrule counters it moved it backwards.
* **The next lever is the critic OBJECTIVE, not more search, more budget, or more steps.** The two
  candidates iteration 2 left standing — a contrastive objective, and depth — are untouched by this
  result, and depth is still downstream (`H4`: depth re-ranks +0.19% of decisions). The genuinely
  new item is the head-to-head *direction*: the 73M win-prob head is the **worst** of the three
  leaves tested and the best-calibrated head is not the best leaf, so **leaf quality is not bought
  with training steps, and calibration is not the axis that buys it either.**
* **What would change the verdict:** a leaf whose CRN-paired differences separate — i.e. a critic
  trained on a target that distinguishes ACTIONS at one ply, not one that is merely well-calibrated
  in aggregate. The cheapest next probe on this instrument is the same `grid` cell on whatever head
  that objective produces: it costs 200 battles per head and it is the most sensitive row in the
  table (it puts the leaf on trial with no gate to hide behind).

## 10. Ready-to-append ledger paragraph

> ### 🔎 THE MIRROR METER RE-READ AT THE WIN-PROB MILESTONE — the promoted head is a WORSE leaf than the shaped critic it replaced; no dividend on any head (2026-09-11)
>
> Record `designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/`
> (PREDICTION.md registered before the first cell; **5,556 mirror battles / 243,197 decisions, zero
> timeouts, zero errors**, `--games-seed 7`, CPU, 9 shards, all three heads playing the SAME
> battles). Discharges the 2026-08-24 standing order ("re-run this probe after each critic
> milestone; the mirror table IS the critic-resolution meter in behavioural units").
> **Naive `grid` search on the win-prob head is a catastrophe and is NO BETTER than the shaped
> critic it replaced: paired 0.2600 / 0.2600 / 0.2025 (ctrl10M @10M / λ0.9 @10M / winprob_critic
> @73M) against the v9 shaped cell's 0.2929, changing 56–60% of actions and losing BOTH orientations
> in 55–62% of pairs.** **At iteration 2's exact 3 s defensive operating point the dividend is
> ABSENT: 0.5059 [0.4914, 0.5205] · 0.5031 [0.4894, 0.5169] · 0.4950 [0.4740, 0.5160].** The two 10M
> heads read 0.5206 / 0.5150 at the pre-registered 400 pairs (ctrl10M's lower bound 0.4999 — one
> ten-thousandth from "search pays"); **the extension registered mid-cell, before that read, put 400
> FRESH pairs on each and both came back at 0.4913** — the positive was noise, and it is a statement
> only because the second look was committed to in advance. **The mechanism says why:
> separation-of-raced 11.9–15.0% against iteration 2's 45.4%, overrules 1.1–1.6% against 5.82% —
> the promoted head certifies four to five times FEWER overrules than the auxiliary win-prob head on
> a shaped-critic checkpoint, and `futility_genuine` is 0 in 38,671 stops (still clock-limited).**
> Four of five registered predictions refuted, all in the same direction; head-to-head the 73M head
> is the WORST leaf and the best-calibrated head is not the best (no pairwise delta CI excludes
> zero; ctrl10M − wp73M +0.0256 [−0.0024, +0.0536]). **Rung A (1 s uniform) is an accidental
> PLACEBO — 0 overrules in 37,909 decisions — and still reads 0.4933 / 0.5133 / 0.5100, the standing
> demonstration of what a "0.51" means at n=150.** **Consequence: the search-and-distill path is NOT
> open at this leaf** (a teacher that overrules ~1% of decisions has ~nothing to distill, and
> distillation recovers a fraction of a measured dividend); the binding constraint is the critic
> OBJECTIVE, and leaf quality is bought neither with training steps nor with calibration. Three
> instrument findings banked: a results row records neither the CHECKPOINT nor
> `--defensive-contested-deadline-s`, so cells are distinguishable by FILE NAME only; `31cd5e3a`
> grew the team pool 679 → 719 after every historical cell, so a game index still names the same
> DICE but no longer the same TEAMS (every cross-era mirror comparison is independent-sample, not
> paired); and `--arm base` costs ~2.5 s/decision while searching nothing, which is why the
> exact-50 control was stopped at 27 pooled pairs (0.5185 [0.4822, 0.5548], 26/27 split).

---

## 11. PHASE 2/3 — three more heads, and the AXES question

*Measured 2026-09-12 01:35–08:22 UTC · **6,000 further battles / 285,081 decisions** (11,556 and
528,278 across both batteries) · same protocol, same `--games-seed 7`, same game indices, so every
contrast is paired · **zero timeouts, zero errors, zero unfinished games** · registered in
[`PREDICTION.md`](PREDICTION.md) §7 (the two heads), §7b (an outcome-blind stopping rule), §7c/§7d
(the third head and the L1/L2 bar). Scripts [`run_battery2.sh`](run_battery2.sh),
[`run_battery3.sh`](run_battery3.sh); scoring [`report2.py`](report2.py) and
[`l1_width_matched.py`](l1_width_matched.py); data [`results2.json`](results2.json).*

**The heads, and the question.** The critic ladder measures CONDITIONING (does `V` move with the
opponent and the team); the mirror measures LEAF QUALITY (does ranking actions on `V` beat the
policy). Are they the same axis?

| head | lever | ladder's conditioning row |
|---|---|---|
| `ai_v12_17_ladder_strata` @10M | class-balanced BCE | **the one lever that moved it** |
| `ai_v12_20_ladder_denseaux` @10M | dense within-game auxiliary targets | null |
| `ai_v12_12_ladder_cflabels` @10M | counterfactual Q-labels (`--cf-winprob-coef 0.5`) | null; the closest trained thing to a **successor-discrimination** objective |

### 11.1 L2 — the outcome row. No lever moves it.

| head | L2 paired (800 pairs) | vs its CONTEMPORANEOUS ctrl10M anchor (150 shared pairs) |
|---|---|---|
| `strata` | **0.5066** [0.4924, 0.5207] | −0.0233 [−0.0638, +0.0172] NOT DETECTED |
| `denseaux` | **0.4941** [0.4826, 0.5056] | −0.0167 [−0.0565, +0.0231] NOT DETECTED |
| `cflabels` | **0.4853** [0.4664, 0.5042] | −0.0183 [−0.0802, +0.0435] NOT DETECTED |
| *anchor-1 `ctrl10M`* (150 pairs, 01:35–05:00) | 0.5133 [0.4827, 0.5440] | — |
| *anchor-2 `ctrl10M`* (150 pairs, 05:00–08:22) | 0.5283 [0.4860, 0.5707] | — |

Head-to-head at 800 shared pairs, every CI straddling zero: ctrl10M − strata **−0.0006** [−0.0210,
+0.0197] · ctrl10M − denseaux **+0.0119** [−0.0072, +0.0309] · ctrl10M − cflabels **+0.0206**
[−0.0036, +0.0448] · strata − cflabels **+0.0213** [−0.0020, +0.0445].

**Six win-prob heads now sit on the structural null**, and each new head is at or slightly *below*
its own same-window control. **L2 bar: not cleared by any lever. Prediction 1 (both) and prediction 2
(cflabels) HELD.**

### 11.2 L1 — the mechanism row is NOT a head property. It is a width meter.

**The anchors convict it.** The `ctrl10M` control — same checkpoint, same cell, same flags, same
battles — was run three times in three contention regimes:

| the SAME control, three windows | realized K (worlds / contested decision) | **L1 = separation-of-raced** |
|---|---|---|
| 2026-09-11, quiet box | 5.06 | **0.128** |
| 2026-09-12 anchor-1, box at load ~30–76 | 3.63 | **0.058** |
| 2026-09-12 anchor-2, box quieting to ~11 | 8.87 | **0.365** |

**One head, one cell, one set of battles: L1 spans 0.058 → 0.365, a 6.3× range, ordered exactly by
how much width the wall clock bought.** The registered bar (> 0.181, with ≤ 0.20 PROVISIONAL) was
built from three quiet-box heads at K 4.85–5.48; it is not portable to a cell measured at another K.

That is what happened to `cflabels`, and applying the rule as written would have produced a false
positive:

| head | pooled L1 | K | the bar as written | **read against its contemporaneous anchor** |
|---|---|---|---|---|
| `strata` | 0.115 | 4.78 | below bar | anchor-1 0.058 at K 3.63 — *not width-matched* |
| `denseaux` | 0.061 | 3.80 | below bar | anchor-1 0.058 at K 3.63 — matched, **no difference** |
| `cflabels` | **0.303** | **10.08** | **"CLEARS (0.303)", not even provisional (≳ 0.25)** | anchor-2 **0.365** at K 8.87 — **the CONTROL clears it by more** |

**Width-matched L1** (recomputed from each battle's own realized K; a head is comparable to another
head only inside a band):

| cell/head | K 3–4.5 | K 4.5–6 | K 6–8 | K 8+ |
|---|---|---|---|---|
| `ctrl10M` (quiet box) | 0.054 [0.050, 0.060] | 0.209 [0.195, 0.223] | 0.337 [0.316, 0.359] | 0.269 [0.239, 0.301] |
| `ctrl10M` anchor-1 | 0.054 [0.042, 0.068] | 0.252 [0.204, 0.306] | — | — |
| `ctrl10M` anchor-2 | — | 0.373 [0.324, 0.425] | 0.430 [0.398, 0.462] | 0.346 [0.320, 0.373] |
| `lambda09` | 0.056 [0.051, 0.062] | 0.262 [0.244, 0.280] | 0.312 [0.289, 0.336] | 0.261 [0.227, 0.299] |
| `wp73M` | 0.057 [0.049, 0.066] | 0.232 [0.211, 0.254] | 0.330 [0.302, 0.358] | 0.288 [0.253, 0.327] |
| **`strata`** | 0.056 [0.049, 0.062] | 0.283 [0.265, 0.301] | 0.341 [0.317, 0.366] | 0.345 [0.314, 0.376] |
| **`denseaux`** | 0.040 [0.035, 0.046] | 0.212 [0.194, 0.231] | 0.288 [0.259, 0.318] | 0.325 [0.269, 0.387] |
| **`cflabels`** | 0.126 [0.103, 0.153] | 0.285 [0.265, 0.305] | 0.336 [0.321, 0.351] | 0.319 [0.308, 0.330] |

**The band moves L1 by 5–60×; the heads inside a band span at most ~1.7×, and the CONTROL's own
three measurements (0.209 / 0.252 / 0.373 at K 4.5–6) span more than any head differs from any
other.** `cflabels` sits *below* its contemporaneous anchor in both bands where they overlap
(0.285 vs 0.373, 0.336 vs 0.430). `strata` leads the quiet-box `ctrl10M` at K 4.5–6 (0.283 vs 0.209,
CIs disjoint) but not anchor-1 (0.252, overlapping) — a weak, unreplicated lead, reported as a lead
and nothing more.

**Consequence for the meter, stated plainly: L1 as registered cannot be compared across measurement
windows, and its 18.1% bar is a statement about the box as much as about the head.** Any future L1
read needs either a contemporaneous control or width-matched bands — both are cheap, and
[`l1_width_matched.py`](l1_width_matched.py) does the second from rows that already exist.

### 11.3 The `grid` row is the SENSITIVE leaf row

Unguarded search, 100 pairs per head, no gate to hide behind — and unlike L1 it *does* separate
heads with detected deltas:

| head | `grid` paired | action changed |
|---|---|---|
| `strata` | **0.3150** [0.2565, 0.3735] | 57.8% |
| `denseaux` | **0.3025** [0.2367, 0.3683] | 60.5% |
| `ctrl10M` / `lambda09` | 0.2600 / 0.2600 | 56.4% / 59.9% |
| `wp73M` | 0.2025 | 60.0% |
| **`cflabels`** | **0.1875** [0.1325, 0.2425] | **68.0%** |

Paired on the 100 shared indices: strata − cflabels **+0.1275** [+0.0573, +0.1977] **DETECTED** ·
denseaux − cflabels **+0.1150** [+0.0367, +0.1933] **DETECTED** · wp73M − strata **−0.1125**
[−0.1883, −0.0367] **DETECTED**. Against `ctrl10M` the two new levers are ahead by +0.055 / +0.043,
NOT DETECTED at 100 pairs.

**`cflabels` — the successor-discrimination head — is the WORST leaf of the six, detectably.** It
also overrules the most (3.2% vs 0.6–1.6%) and changes the most actions under `grid` (68%): it is
the most *confident* re-ranker and the most *wrong* one. ⚠️ The new heads' `grid` cells ran at
K 1.64–1.74 against the earlier three's 1.18–1.27, so the +0.04–0.06 leads over `ctrl10M` are not
width-matched; the detected cflabels deltas are between cells that ran in the same window.

### 11.4 The axes question — answered, in the branch registered as "neither pays"

* **A lever that MOVED the conditioning row (`strata`) does not move either leaf row** — L2
  −0.0233 vs its own control, L1 inside the control's own between-window spread.
* **Two levers that are NULL on the conditioning row (`denseaux`, `cflabels`) do not move them
  either** — and `cflabels` moves the unguarded row *backwards*, detectably.
* So the registered branch is **"neither pays"**: the axes question is **not settled** by this pair
  (a null on both is consistent with separate-but-unmoved and with same-axis-unmoved), and what IS
  settled is sharper than the branch predicted — **no trained lever in the ladder moves leaf
  quality, including the one lever the ladder itself certifies.** A row that moves under a lever
  which changes nothing behavioural is not yet shown to be the row the search path should be steered
  by.
* **Dose caveat, registered UNREAD and still unread:** `cflabels` ran `--cf-winprob-coef 0.5`,
  `cf_head_only`, a 150k-step label lag, and a sibling run was retired as
  `…DEAD_fatal_config_cf_duty_cycle_6pct`. Its realized cf duty cycle was never read, so its null is
  **dose-unread**, not a verdict on counterfactual labels as a class.

### 11.5 The scored phase-2/3 predictions

| registered prediction | outcome |
|---|---|
| §7.1 neither strata nor denseaux clears the bar | **HELD** — 0.5066 and 0.4941, both CIs straddling 0.50 |
| §7.2 denseaux is the better LEAF of the two on the mechanism rates | **REFUTED** — denseaux has the LOWEST L1 (0.061) and the lowest overrule rate (0.6%) of every head measured; strata leads it in every width band |
| §7.3 strata's `grid` is no better than the three already measured, and may be worse | **REFUTED** — it is the BEST of six (0.3150), though the delta vs ctrl10M is not detected |
| §7.4 denseaux's `grid` is the best of five, 0.28–0.40 | **NEARLY HELD** — 0.3025, inside the band, but second to strata |
| §7c.1 cflabels has the highest L1 of the three new heads | **HELD as stated, but VOID as evidence** — its pooled L1 is highest (0.303) and it does clear 18.1% decisively, yet width-matching and its own contemporaneous anchor both overturn it |
| §7c.2 cflabels does not clear L2 | **HELD** — 0.4853 [0.4664, 0.5042] |
| §7c.3 cflabels' `grid` lands in 0.20–0.32 | **REFUTED, low: 0.1875** — below the band and the worst of six |

### 11.6 Further hazards

10. **L1 is contention-coupled** (§11.2) — the single most important instrument finding here, and it
    was only visible because a contemporaneous anchor was run. A meter defined as a rate under a
    wall-clock budget inherits the box's load.
11. **A cell run across a load transition is not homogeneous.** `cflabels` began at K ≈ 6.6 and
    finished near K ≈ 12 as the box emptied; its pooled K of 10.08 describes no part of the run
    well. The width-matched table is the honest read, and `report2.py` publishes pooled K beside
    every L1 so a reader can see when this applies.
12. **Shard rebalancing mid-cell is safe but must resume, not restart.** Two `strata` shards were
    stopped by explicit PID at 01:56 and relaunched over narrower disjoint windows onto their OWN
    existing out files, so the battery's resume logic skipped the games already played; a relaunch
    onto a *new* file name would have replayed them and `report.py`'s overlap guard would have
    refused the pooled read.

## 10b. Ready-to-append ledger paragraph — phase 2/3 (three more heads, and the L1 instrument)

> ### 🔬 THE AXES QUESTION ANSWERED IN THE "NEITHER PAYS" BRANCH — no trained ladder lever moves leaf quality, including the one the ladder certifies; and L1 is convicted as a WIDTH meter (2026-09-12)
>
> Record `designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/` §11
> (predictions registered before each head in PREDICTION.md §7/§7b/§7c, including an outcome-blind
> wall-clock stopping rule; **6,000 battles / 285,081 decisions, zero timeouts, zero errors**;
> same `--games-seed 7`, same game indices as the first three heads, so every contrast is paired).
> Heads: `strata@10M` (class-balanced BCE — **the one lever that moved the ladder's conditioning
> row**), `denseaux@10M` (dense within-game targets, null on conditioning), `cflabels@10M`
> (`--cf-winprob-coef 0.5`, the closest trained thing to a successor-discrimination objective, null
> on conditioning). **L2 — the outcome row — is not moved by any of them: 0.5066 [0.4924, 0.5207] ·
> 0.4941 [0.4826, 0.5056] · 0.4853 [0.4664, 0.5042], and each is at or slightly BELOW its own
> CONTEMPORANEOUS `ctrl10M` anchor (−0.0233 · −0.0167 · −0.0183, all NOT DETECTED).** Six win-prob
> heads now sit on the structural null. **The registered branch is "neither pays": the axes question
> is NOT settled, and what is settled is sharper — a lever that moves the conditioning row leaves
> leaf quality where it was, so that row is not yet shown to be the row the search path should be
> steered by.**
>
> **L1 IS CONVICTED AS AN INSTRUMENT, and only a contemporaneous control could have caught it.** The
> same `ctrl10M` checkpoint, same cell, same flags, same battles, run in three contention regimes,
> reads **L1 = 0.058 / 0.128 / 0.365 at realized K = 3.63 / 5.06 / 8.87 worlds** — a 6.3× range
> ordered by how much width the wall clock bought. `cflabels` pooled **0.303 at K = 10.08** would
> have "CLEARED (≳ 0.25, not even provisional)" against the 18.1 % bar — while its own same-window
> anchor read **0.365**, i.e. the CONTROL clears by more. **Width-matched** (each battle binned on
> its own realized K), the band moves L1 by 5–60× while heads inside a band span ≤ 1.7×, and the
> control's three measurements at K 4.5–6 (0.209 / 0.252 / 0.373) span more than any head differs
> from any other; `cflabels` sits BELOW its anchor in both overlapping bands. **The 18.1 % bar is a
> statement about the box as much as about the head; every future L1 read needs a contemporaneous
> control or width-matched bands** (`l1_width_matched.py`, from rows that already exist).
> **The `grid` cell is the sensitive leaf row instead** — unguarded, no gate to hide behind, and it
> separates heads with DETECTED deltas where L1 cannot: strata **0.3150** [0.2565, 0.3735] and
> denseaux **0.3025** are the best of six, **cflabels 0.1875 [0.1325, 0.2425] the worst**
> (strata − cflabels +0.1275 [+0.0573, +0.1977] DETECTED; denseaux − cflabels +0.1150 [+0.0367,
> +0.1933] DETECTED). **The successor-discrimination head is the most confident re-ranker (68 % of
> actions changed, 3.2 % overrules — the highest of any head) and the most wrong one.** Its null is
> reported **dose-unread**: `cf_head_only`, a 150k-step label lag, and a sibling retired as
> `…DEAD_fatal_config_cf_duty_cycle_6pct`, with the realized duty cycle never read — so this is not
> a verdict on counterfactual labels as a class. Four of seven phase-2/3 predictions refuted, and the
> one that "held" (cflabels highest L1) is void as evidence by the instrument finding above.
