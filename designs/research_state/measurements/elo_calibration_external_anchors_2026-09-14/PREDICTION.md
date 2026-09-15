# PRE-REGISTRATION — a LARGE ELO calibration that gives the bot-anchored scale two EXTERNAL anchors

**Registered 2026-09-14, before a single game of this campaign was played.** Everything below is
committed ahead of the data. The refit baselines in §2 are *deterministic re-analyses of already
committed files* (no new battles), computed first on purpose so the predictions can be numeric.

---

## 1. The problem this campaign exists to attack

Our absolute strength scale has exactly one anchor: the nine eval bots, pinned by a bot-vs-bot
round robin in `data/gen3_bot_elo_anchors.json`. **They are saturated.** Measured from each run's
own `eval_results.jsonl`, the newest node of every arm beats the eight non-random bots at:

| run | newest node | mean win rate vs the 8 bots |
|---|---:|---:|
| `ai_v9_29_rev1_0823` | 24,000,000 | **0.916** |
| `ai_v12_11_ladder_ctrl10M` | 10,000,032 | **0.899** |
| `ai_v12_02_winprob_critic` | 74,000,016 | **0.882** |
| `ai_v13_01_flywheel_shaped` | 74,000,016 | **0.919** |

A Bradley–Terry edge at p = 0.9 carries Fisher information ∝ p(1−p) = 0.09, against 0.25 at p = 0.5
— **less than 40% of the information per game**, and the logistic is flat there, so a ±1 pp wobble
in the bot edge moves the fitted rating a long way. Every frontier node's ABSOLUTE level is
therefore set by a nearly-flat likelihood plus the prior and the dense internal matrix. That is the
mechanism we believe is behind the newest-node inflation the matched-snapshot-COUNT rule exists to
control (`ai_v9_29_rev1_0823`'s 8M node: **2052** in a first-4 fit, **1958** in the final 12-node
fit — 94 Elo).

**The fix under test:** two third-party agents that nobody here trained, placed NEAR the frontier,
so the top of the ladder finally has an edge with real curvature.

---

## 2. The baselines, computed BEFORE the campaign

### 2.1 Every ladder refit at the CURRENT recipe (UNDERSTANDING rule 24)

A committed `ladder.json` carries the recipe it was fitted with. Two of the four runs' committed
files have **no `eval_sentinel_edges_dropped` key** — the stale recipe — so their committed numbers
are NOT used anywhere in this campaign. All four were refit with
`snapshot_ladder.fit_ladder(run_dir, write=False)` at HEAD (`70fa757f`):

| run | node | committed | **refit (used here)** | recipe stamp in the committed file |
|---|---:|---:|---:|---|
| `ai_v9_29_rev1_0823` | 2,000,016 | 1692.5 | **1723.0** | ABSENT ⇒ stale |
| | 8,000,016 | 1957.9 | **1961.5** | |
| | 24,000,000 | 2098.4 | **2053.3** | |
| `ai_v12_11_ladder_ctrl10M` | 10,000,032 | 2018.7 | **2018.7** | present (6) |
| `ai_v12_02_winprob_critic` | 36,000,000 | 1996.6 | **1954.5** | ABSENT ⇒ stale |
| | 56,000,016 | 2022.1 | **1965.4** | |
| | 74,000,016 | 2057.3 | **1984.2** | (the +73 stale-recipe trap) |
| `ai_v13_01_flywheel_shaped` | 22,000,032 | 1982.7 | **1982.7** | present (51) |
| | 48,000,000 | 2022.4 | **2022.4** | |
| | 72,000,000 | 2036.6 | **2036.6** | |

### 2.2 What the 2026-09-14 matched-regime 2×2 already says about the two anchors

Same protocol as this campaign (**greedy-vs-greedy, HOME = our 719-team pool**), our side
`ai_v12_02_winprob_critic` at 75,005,952 steps, 100 games:

| anchor | our win rate, greedy · home | implied Elo gap (400·log10(p/(1−p))) |
|---|---:|---:|
| `metamon:SmallRL` (ckpt 40, 13.9M) | 0.520 [0.423, 0.615] | **+13.9 to us** |
| `metamon:SyntheticRLV2` (ckpt 48, 200.9M) | 0.500 [0.404, 0.596] | **0.0** |

🚨 **Note what this overturns in the campaign's own framing.** "SmallRL below us, SyntheticRLV2
above us" is true on the AWAY set (0.650 / 0.420). On the HOME set — the set this campaign uses,
because it is the ladder's own protocol — **both anchors read level with our 75M arm**. That makes
them *better* anchors, not worse: an anchor is informative exactly where it is near 50%.

---

## 3. The design, fixed in advance

**Protocol, every cell:** `python -m main.anchors`, greedy-vs-greedy, **HOME** team set (our 719
pool on both sides), 100 games, role-balanced across two half-series, our pinned Showdown
(`e0551883f`), CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice`, ports drawn from 9500–9599.
Regime verified per decision on both sides; a cell whose peer reports no `argmax_match_rate`, or
whose `status != "OK"`, is excluded from the fit and reported as excluded.

**Cells (38 × 100 = 3,800 games), plus one optional 200-game cell:**

| block | our side | opponent | cells | games |
|---|---|---|---:|---:|
| **B — bots** | each of the 9 eval bots (`random`, `heuristic`, `heuristic2`, `staller`, `staller_v2`, `aggressive`, `aggressive_v2`, `setup_sweep`, `setup_sweep_v2`) | both Metamon models | 18 | 1,800 |
| **S — snapshots** | the 10 frozen snapshots in §2.1 | both Metamon models | 20 | 2,000 |
| **X — head to head** | `metamon:SmallRL` | `metamon:SyntheticRLV2` | 1 | 200 |

**🚨 The v8 leg of the commissioned design is INFEASIBLE and is not being run.**
`ai_v8_03_zarch_control_0718`'s snapshots are `MODEL_CONFIG_VERSION` 44/45 under
`ARCH_SIGNATURE = gen3_opp_hp_typed_candidates_v1`; `MIGRATION_FLOOR` is **96** and the current
signature is `gen3_critic_route_wave_v1`, so `load_foreign_opponent` refuses them by design
("a PRE-GENERATION checkpoint"). No v8 node can be played by current code at all. The two v8 nodes
are replaced by a **third `ai_v9_29_rev1_0823` node at 2,000,016** — deliberately the WEAKEST
loadable node in the archive (1723 refit), so the campaign still spans ~330 Elo of our own history
and the anchors are not measured only against a narrow frontier band.

**Why these ten.** Every one carries historical bot edges in its run's `eval_results.jsonl`
(verified: 12/5/37/37 bot-edged steps in the four runs), so each is already a node of the existing
anchored fit and the external edges are being added to a frame that exists, not to a new one.

**Three hooks are needed in `src/`, each with a test:** a bot as our side (`--our-side bot:<name>`),
a cross-run snapshot loaded through `load_foreign_opponent` rather than a bare `MaskablePPO.load`
(`--model-load auto|bare|foreign`; the v9 snapshots FAIL a bare load with
`ExtractorBuild.__init__() got an unexpected keyword argument 'threat_prob_outspeed'`), and the
two-peer head-to-head cell. Which loader ran is stamped on every row.

---

## 4. How the anchors enter the fit

One **joint** Bradley–Terry fit over three node families, run three ways on the SAME node set so
the anchor effect is isolated from the pooling effect:

* **nodes**: `bot:<name>` ×9 (PINNED to `data/gen3_bot_elo_anchors.json`, `random` = 1000) ·
  `snap:<run>@<step>` ×10 (free) · `ext:metamon:<Agent>` ×2 (free).
* **edges**: (a) each snapshot's historical bot edges from its run's `eval_results.jsonl`;
  (b) the dense frozen snapshot-vs-snapshot edges from each run's `snapshot_ladder/games.jsonl`,
  **within a run only** (eval-sentinel `snap:`-vs-`snap:` edges DROPPED, as rule 24's recipe
  requires); (c) **NEW** — the 38 external edges this campaign measures, plus the head-to-head.
* **fit**: `agents.training.elo.fit_pairwise`, the same penalized-MLE Newton core the ladder uses,
  same `ELO_PER_DECADE = 400`, same `DEFAULT_PRIOR_SD = 400`.

**Three arms, reported side by side:**

| arm | edges (a) | edges (b) | edges (c) | bots |
|---|---|---|---|---|
| **JOINT-NOEXT** | ✅ | ✅ | ❌ | pinned |
| **JOINT-EXT** | ✅ | ✅ | ✅ | pinned |
| **JOINT-EXT-FREEBOT** | ✅ | ✅ | ✅ | only `random` pinned; the other 8 FREE |

`JOINT-EXT − JOINT-NOEXT` is the anchor effect. `JOINT-EXT-FREEBOT` answers "would the bot
ratings move if re-fit jointly" and is **reported only** — `data/gen3_bot_elo_anchors.json` and
`designs/ai_v5/elo_calibration/gen3_bot_elo_games.json` are NOT written by this campaign.

---

## 5. THE PREDICTIONS — what "the scale moved" means, fixed now

Each is falsifiable, with the falsifying observation named. **A CI that covers zero is NOT
DETECTED — never "equal", never rounded to a pass.**

### P1 — anchor placement (the headline)
`ext:metamon:SmallRL` lands at **1970 ± 100** and `ext:metamon:SyntheticRLV2` at **1984 ± 100** on
the JOINT-EXT scale, each with **se ≤ 25**.
*Derivation:* the §2.2 home-set win rates translated through `ai_v12_02_winprob_critic`'s refit 74M
node (1984.2). **FALSIFIED** if either point estimate falls outside its ±100 band, or if either se
exceeds 25.

### P2 — the anchors are NOT saturated (the whole motivation)
Over the 20 snapshot cells, the **median |win rate − 0.5| is < 0.15**, against **0.40** for the bot
edges at the frontier. **FALSIFIED** if the median |wr − 0.5| over the 20 snapshot cells is ≥ 0.25,
i.e. if the external anchors turn out to be saturated too — in which case this campaign has bought
a second saturated yardstick and should say so loudly.

### P3 — frontier PRECISION improves (the primary quantitative endpoint)
The mean standard error of the ten snapshot nodes falls from JOINT-NOEXT to JOINT-EXT by **≥ 15%**.
**FALSIFIED** if the reduction is < 15%, and **REFUTED** (sign error in our model of the problem) if
the mean se RISES.

### P4 — the frontier is pulled DOWN, and only a little
Δᵢ = rating(JOINT-EXT) − rating(JOINT-NOEXT). Prediction: **mean Δ over the ten nodes is negative
and |mean Δ| < 15 Elo**; the four newest-era nodes (`ai_v13_01` 72M, `ai_v12_02` 74M,
`ai_v12_11` 10M, `ai_v9_29` 24M) move down by MORE than the two oldest nodes do.
*Reasoning:* the anchors sit level with our 75M arm on home teams while beating the bots by less
than our frontier does; the fit resolves that tension by lowering the frontier, not by raising the
anchors past their own bot edges. **FALSIFIED** if mean Δ is positive, or if |mean Δ| ≥ 15 Elo.

### P5 — newest-node inflation SHRINKS
For `ai_v13_01_flywheel_shaped` and `ai_v12_02_winprob_critic`, define
`infl(k) = rating_k(first-k fit) − rating_k(full fit)` and take the mean over k = 5 … N−1.
Prediction: **mean infl falls by ≥ 25%** from JOINT-NOEXT to JOINT-EXT, because the newest node now
carries an out-of-family edge whose value does not change as later nodes are added.
**FALSIFIED** if the reduction is < 25%; **REFUTED** if mean infl grows.

### P6 — fit quality may get WORSE, and that is not a failure
`max_abs_err` over the measured edges is **allowed to rise** in JOINT-EXT. A genuinely informative
node exposes residuals the saturated bot edges could not. Registering it now so a rise cannot be
reported as a defeat or hidden as a footnote. The number that must NOT rise is `mean_abs_err` over
the **bot** edges alone — those edges are unchanged, so a rise there means the fit is being
distorted rather than informed. **FALSIFIED** if bot-edge `mean_abs_err` rises by > 0.01.

### P7 — the pinned bots are roughly right
In JOINT-EXT-FREEBOT, each of the eight non-random bots moves **< 40 Elo** from its pinned value.
**FALSIFIED** if any bot moves ≥ 40 Elo — which would be a real finding about the 2026-06-06
calibration and goes to the backlog, not into `data/`.

### P8 — the head-to-head
`SyntheticRLV2` beats `SmallRL` over 200 games at **0.55–0.75**, and the JOINT-EXT ratings order the
two the same way. **FALSIFIED** if the Wilson interval excludes that band, or if the fitted order
contradicts the direct edge (a transitivity failure, which would itself be the finding).

---

## 6. The regime caveat, stated before the data

`data/gen3_bot_elo_anchors.json` was computed **2026-06-06** at git `1081a8e0`, 2,000–2,700 games
per pair over all 36 pairs, `random` pinned at 1000, `converged: true`,
`max_abs_err 0.0381`. That is **three months before the 2026-09-07 eval-sentinel regime boundary**.

**What that boundary does and does not touch.** It moved the eval *sentinels* (snapshot-vs-snapshot
opponents) to greedy and to the trainee's own teams, worth +8.9 pp to the trainee. `ladder.json` and
**every bot edge are UNAFFECTED** — a bot has no sampling knob to flip and never drew sentinel
teams. The bot-vs-bot round robin that produced the pins is bot-policy-vs-bot-policy on the shared
pool with `bias_prob = 0.1`, which is the same operation today as it was in June.

**The residual exposure, named:** the bots' *code* could have changed since `1081a8e0`
(`bot_elo_store.load` warns on a git-hash mismatch and keeps the counts — it does not re-measure).
This campaign does not re-play the bot-vs-bot matrix, so it cannot settle that. What it CAN do is
expose it: if the pinned bot ratings are wrong, the FREEBOT arm (P7) will say so, and the external
anchors will disagree with the bot frame in a direction that is visible in P4. A move ≥ 40 Elo on
any bot is a **backlog row proposing a re-calibration**, not an edit to `data/`.

**One asymmetry this campaign accepts and records:** the bot anchor calibration and the snapshot
ladder both draw teams with `bias_prob = 0.1` toward the sample teams; `main.anchors`'s home source
is the UNBIASED 719-team pool on both sides. Every cell here is internally symmetric, so no cell is
confounded, but the external edges are not drawn from exactly the same team distribution as the
existing bot edges. It is recorded, not corrected.

---

## 7. What this campaign will NOT claim

* Nothing about Metamon on Metamon's own Showdown pin — both clients play on ours (`e0551883f`).
* Nothing about the public ladder.
* Nothing about exploitation: neither side adapts, so greedy-vs-greedy is safe *for this pair* and
  that is an implementation fact about Metamon, not a general result.
* **No change to `data/gen3_bot_elo_anchors.json`, to `bot_elo_store`, or to any run's committed
  `ladder.json`.** Adopting the external anchors into the production fit is a PROPOSAL with a diff,
  for the owner; this artifact does not apply it.
