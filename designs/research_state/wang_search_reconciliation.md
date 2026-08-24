# Wang 2024 vs. our search-dividend probe — reconciliation

**Status:** analysis, no code change. Sources: `designs/references/wang2024_pokemon_rl.pdf`
(Jett Z. Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning*, MIT MEng,
Feb 2024; https://dspace.mit.edu/handle/1721.1/153888) read in full, and
`src/main/search_dividend/` at the commit that produced the 2026-08-24 table.
Our measurements are quoted from `designs/research_state/ledger.md` (2026-08-24 entries) —
**this file does not modify the ledger.**

---

## 0. Two-minute decision summary

**The crux, and it is not what we assumed.** Wang's MCTS leaves are **the same kind of object as
ours — a trained critic `V_θ`**, not rollouts to terminal. Vanilla MCTS plays to terminal; Wang
explicitly built the *truncated* variant and says why: "Stopping at leaf nodes is made possible
because we have `V_θ`, a trained state value estimator, and provides efficiency gains because
rollouts can end earlier" (§2.3, p. 20). A rollout ends at terminal *only when the tree descent
happens to reach game end* (`v = ±1/0`), which in a 6v6 is a late-game event. **So "unbiased
leaves" is NOT the ingredient that separates him from us.** Candidate (b) is refuted as a
*Wang* explanation (it survives on other grounds — see §5b).

**What actually separates us is three things, all on the estimator side of the argmax:**

1. **Sample count.** Wang: **R = 1000–2000 rollouts per decision** in a 10 s budget (§3.2.3,
   p. 27), over ≤9 legal actions, on a tree that **persists across the whole game** (states are
   pruned only by fainted count, §3.2.3 p. 28) — so root visit counts *accumulate* turn over
   turn. Us: ~5–50 leaf evaluations per our-action, with only **r_dice ≈ 1.05–7.9** on the axis
   that carries the measured per-leaf noise. That is a 1–2 order-of-magnitude gap in the very
   quantity our mechanism story is about.
2. **The policy prior is inside the selection rule.** Wang runs AlphaZero-style PUCT,
   `a_t = argmax_a (Q[s,a] + α·U(s,a))`, `U(s,a) = P[s,a]^β · √M[s] / (N[s,a]+1)` with
   `P[s,a] = π_θ(a|s)` (§2.3, p. 21). Leaving the policy costs accumulated evidence. Our
   selection has **no prior term at all** (`search.py:426`, argmax over aggregated scores, with
   the policy used only to break *exact* ties).
3. **He does not argmax the value — he argmaxes the visit count**, and he states our exact
   failure mode as the reason: "It is intuitive to choose the action with the largest Q value
   instead, but **less-visited actions may have higher variance in their Q estimates**" (§2.3,
   p. 22). Wang designed around the optimizer's curse. We walked into it.

**Are the two results consistent?** **Yes, once conditions are matched — with one caveat we
cannot yet discharge.** Our measured leaf noise `sd = 0.0115` against a `top1−top2` margin of
`0.0213` means the argmax at `r ≈ 1` acts on noise at **54% of its own margin**, and it changes
61–71% of decisions. Wang's regime puts that ratio at roughly **4–5%**. Our own dose-response
(most-averaged cell least harmed: oracle@3s `0.434`; most-deepened cell most harmed: honest@3s
`0.276`) points the same way. **The caveat:** at our best cell the noise ratio is already down to
~19% of the margin and the win rate is still `0.434 < 0.50`. A pure-variance account predicts the
curve should be much closer to 0.50 by then. So the four cells are *consistent with* variance
being the mechanism but do **not** establish that variance is the *whole* mechanism — a residual
critic **bias** floor is not excluded, and our four cells confound arm with budget so they cannot
separate the two.

**The single cheapest experiment: a pure R-ladder at Wang's budget.** Zero lines of code — the
flags already exist. Fix `--arm oracle --opponents self --max-opp 2 --max-worlds 1`, sweep
`--max-dice 1,2,4,8,16,32` at `--budget 10` (Wang's own per-move budget). The `r=1` cell
reproduces the existing table, so it is self-controlled. **This is the fork in the road:** if win
rate climbs monotonically toward 0.50 the mechanism is variance, the harm is bounded by cheap
fixes (a)/(c), and the search line is revivable *now*; if it plateaus below 0.50 the residue is
critic bias, averaging can never fix it, and the standing verdict ("search is downstream of R1
critic calibration") is confirmed at strength — with rollout leaves (b) the only remaining lever.

---

## 1. Wang's design, as sourced

### 1.1 The leaves — the crux

> "Vanilla MCTS plays each rollout to the end of the episode (a terminal node) … Here we discuss
> a variant where rollouts may end on either: • a terminal node, as in vanilla MCTS • a leaf
> node, i.e. one which is not yet recorded in the tree. Stopping at leaf nodes is made possible
> because we have `V_θ`, a trained state value estimator" — §2.3, p. 20

> "A rollout ends when we encounter a terminal node or a leaf node at timestep T. At this point we
> obtain a value `v` for the final state `s_T`. If `s_T` is terminal, `v` is +1/-1/0 for a
> win/loss/tie respectively. **Otherwise, we use `v = V_θ(s_T)`, the output of the neural
> network's critic head**" — §2.3, p. 21

There is **no random-playout phase**. This is AlphaZero's shape: descend the tree by the tree
policy, stop at the first unexpanded node, evaluate with the critic, expand it, back up. Terminal
returns enter only where the descent reaches game end — increasingly common late-game, absent
early. **Wang's leaf estimator and ours are the same class of object.**

The reward that `V_θ` estimates is terminal-only, `r ∈ {−1, 0, 1}` (§3, p. 23), with
`gamma = 0.9999` (Table A.3, p. 43) — i.e. an essentially undiscounted win/loss signal, so `V_θ`
is a win-probability-like scalar on `[−1, 1]`. Our `TERMINAL_VALUE = {win: 1.0, tie: 0.5,
loss: 0.0}` (`search.py:78`) plus the win-prob head is the same quantity rescaled to `[0,1]`.

### 1.2 The search structure

| Element | Wang | Citation |
|---|---|---|
| Tree policy | `a_t = argmax_a (Q[s,a] + α·U(s,a))`, `U(s,a) = P[s,a]^β · √M[s] / (N[s,a]+1)` | §2.3 p. 21 |
| Prior | `P[s,a] = π_θ(a|s)` — the trained policy, raised to a trust exponent `β ∈ [0,1]` | §2.3 p. 21 |
| Relation to AlphaZero | "similar to that used in AlphaZero, except that α is a constant instead of a function of `M[s]` and `β` is introduced" (footnote 3, p. 21) | §2.3 p. 21 |
| **α, β values** | **never reported anywhere in the thesis**; defined only as `∈ [0,1]` | §2.3 p. 21 |
| Backup | running mean: `Q ← (N·Q + v)/(N+1)`, `N += 1`, `M += 1`. **No max operator.** | §2.3 p. 21 |
| **Root selection** | **`a* = argmax_a N(s_0,a)` — visit count, not Q.** "less-visited actions may have higher variance in their Q estimates" | §2.3 p. 22 |
| Simulations/move | **R = 1000–2000** in 10 s, "depending on the length of each rollout and size of the game tree" | §3.2.3 p. 27 |
| Time/move | **10 s** — the gen4randombattles ladder timer (150 s + 10 s per decision) | §3, p. 23 |
| Depth | Not stated as a number. Emergent: one new node per rollout along the descent, tree holds **2,000–15,000 nodes** | §3.2.3 p. 28 |
| **Tree persistence** | Statistics are kept **"throughout the game"**; nodes pruned only when the fainted count `F[s]` passes them. Root visits therefore accumulate across turns. | §2.3 p. 20, §3.2.3 p. 28 |
| Parallelism | 20 workers, tree merged into a master copy every 10 rollouts; `P` recomputed locally rather than shipped | §3.2.3 p. 27 |
| Simultaneous moves | Not modeled as a simultaneous-move game — the environment is stepped and the opponent's action is drawn from `π_θ` (below) | §3.2.1 p. 26 |
| Hidden info | **Determinization, one fresh world per rollout** — "we sample one possibility for all unknown opponent information at the start of each MCTS trajectory", using **Showdown's own randbats generator** with rejection sampling against revealed traits, forcing after 10 failed attempts | §3.2.2 pp. 26–27 |
| Chance nodes | Not explicit. Handled by *resampling*: "MCTS deals naturally with the stochasticity of Pokémon by taking an expectation over rollouts" (§3, p. 23). Each rollout re-steps the real sim, so dice are redrawn every visit. | §3 p. 23 |
| Opponent model | **The trained policy `π_θ` itself**, sampled inside the rollout. "This has the benefit of simplicity, but weakens the agent's performance against players who play differently from the neural network." | §3.2.1 p. 26 |
| Training use | **None.** "our approach diverges from that of AlphaZero in that **MCTS is not used to train the neural network** … because simulating the environment is very slow" | §3, p. 23 |

Two design notes worth carrying: Wang's determinization draws **a fresh world per rollout**, so
1000–2000 worlds are averaged per decision, from the *exact* generator (a true posterior for
randbats). And because the tree persists and rollouts redraw the dice on every visit, `Q[s_0,a]`
is a Monte-Carlo mean over hundreds of independent (world × dice × opponent-action) draws.

### 1.3 The measured gain, and under what conditions

**There is a clean same-policy ablation** (§4.2, Table 4.1, p. 30):

| | vs MCTS+NN | vs NN | vs Heuristic | vs Random |
|---|---|---|---|---|
| **MCTS + NN** | — | **.809** | **.908** | .996 |
| **NN** (no planning) | .191 | — | .786 | 1.00 |
| Heuristic | .088 | .206 | — | .992 |

- **Search vs no-search on the identical network: `.809`.** This is structurally the same
  contrast as our mirror probe.
- **Against a third party (Heuristic): `.908` vs `.786`** — search buys **+12.2 points** against
  an opponent it is not modeling. This is the cleaner number.
- **Wang himself discounts the `.809`:** "we believe the winrate of the full agent with MCTS vs
  NN to be somewhat *inflated*. Recall that during MCTS, we assume our opponent plays according
  to the NN policy, and search for the best response. Then, because in essence the MCTS always
  knows exactly what the NN will do, its winrate when playing against NN is higher than when
  playing against humans of equivalent strength" (§4.2, p. 32).

**Baseline strength.** The un-searched NN reaches ~85% vs `SimpleHeuristicsPlayer` after 150M
steps (§4.1, p. 29) — a bot Wang describes as "equivalent in skill to a beginner Pokémon player"
(§3.1.2, p. 24). The **ladder** result (rank 8, 1693 Elo, 1756±28 Glicko-1, 79.5% GXE over 200
games, §4.3, p. 32) is the *full agent* — **there is no with/without-search ladder ablation.**
The three experts went 9–13 against the full agent (§4.4, Table 4.2, p. 33).

**So: his headline gain is a mirror ablation just like ours — with the same
opponent-model-knows-the-opponent advantage we also enjoy — and it went the other way by ~31
points where ours went the wrong way by ~21.**

### 1.4 What Wang flagged as broken, that we should not re-derive

- **Mixed strategies (§5.2.1, p. 38):** "in some situations the optimal strategy is to randomize
  … The results of our MCTS often imply this, **with one action just barely edging out the
  other.**" He proposes visit-count-weighted sampling with low-visit pruning. *He observed the
  hair-thin-margin regime too — and his response was to soften the argmax, not to trust it.*
- **Opponent modeling (§3.2.1 p. 26, §5.2.3 p. 38):** modeling the opponent with `π_θ` overfits
  to that one opponent. The Kecleon/Rampardos misplay (§4.4, pp. 33–34) is his worked example.
- **Determinized play is not information-set play (§5.2.2, p. 38):** after sampling, the world is
  perfect-information for the *server and opponent* but the searching agent's net was trained on
  imperfect information — an acknowledged inconsistency.

---

## 2. Our probe, restated in Wang's vocabulary

From `src/main/search_dividend/`:

| Element | Ours | Location |
|---|---|---|
| Leaf | `policy.predict_values()` (critic V) or `sigmoid(last_win_prob_logits)` | `search.py:171–187` |
| Terminal | `{win 1.0, tie 0.5, loss 0.0}`, recorded immediately, never rolled out | `search.py:78, 543–550` |
| Score | `argmax_a Σ_w p(w) · Σ_c α(c) · (1/R) Σ_r V(s'(w,a,c,r))` | `search.py:7–8` |
| Opponent axis | α-weighted mean over ≤`m_opp` candidates, α = the **v67 α-head belief** (uniform when absent) | `alpha.py`, `deepen.py:136–151` |
| Dice axis | arithmetic mean over `r_dice` CRN seeds | `search.py:415–416` |
| World axis | arithmetic mean over `k_worlds` gated determinizations (tiered pool donors) | `determinize.py:225–308` |
| Depth ≥2 backup | **MAX over our actions**, α-mean over theirs | `deepen.py:111–133` |
| **Selection** | **pure `argmax(scores)`**, policy preferred only on an *exact* tie | `search.py:426–429` |
| Prior in selection | **none** | — |
| Margin gate | **none** — "argmax overrides policy unconditionally" | `search.py:426–429` |
| Width order | `m_opp` → `k_worlds` → `r_dice`, greedy to caps `6/8/8` | `budget.py:35, 45–47` |
| Budget | 1 s / 3 s | ledger 2026-08-24 |
| Tree persistence | **none** — a fresh search per decision | `search.py` |

**The measurement** (ledger, 2026-08-24 ~04:10): paired vs the 0.50 null, side-swapped mirror —
honest@1s **0.294** [0.24,0.35] n=120 · honest@3s **0.276** [0.18,0.37] · oracle@1s **0.325**
[0.23,0.42] · oracle@3s **0.434** [0.32,0.55]. Per-leaf dice `sd = 0.0115` vs `top1−top2` margin
`0.0213`; change rate **61–71%**.

---

## 3. The diff table

Ranked by my estimate of how much of the sign flip each ingredient carries.

| # | Wang has | We have | Plausibly explains the flip? | Cheapest test in our harness |
|---|---|---|---|---|
| **1** | **R = 1000–2000 rollouts/decision, tree persists across turns** ⇒ ~10²–10³ independent draws per root action | `r_dice ≈ 1.05–7.9`; ~5–50 leaves per our-action; fresh tree per decision | **YES — the largest single term.** Noise/margin ratio 54% (ours, r≈1) vs ~4–5% (his). Our own dose-response is this axis. | **Zero code.** `--arm oracle --opponents self --max-opp 2 --max-worlds 1 --budget 10 --max-dice {1,2,4,8,16,32}` |
| **2** | **Visit-count argmax, explicitly chosen over Q-argmax for variance reasons** (§2.3 p. 22) | Value argmax, unconditional | **YES.** A visit count is a *shrunk* statistic: an action only accumulates visits if PUCT keeps re-selecting it, which needs its `Q` to beat the prior-boosted alternatives repeatedly. This is a soft margin gate with the right shape. | Implement (a) or (c) below — the depth-1 analogue |
| **3** | **PUCT prior `P[s,a]^β` inside selection** | No prior in selection; policy used only for exact ties | **YES.** Bounds harm near zero *by construction* — with a strong prior and few visits the search returns the policy's action. | Prior-shrunk argmax, §5c |
| **4** | Backup = running **mean** of returns | Depth-≥2 backup takes **MAX** over our actions (`deepen.py:111–133`) | **YES, and it matches the dose-response.** `E[max] ≥ max E` under noise, so an extra ply *adds* optimism bias. The most-deepened cell (honest@3s, 24% deepen) is the most harmed cell. | Re-run honest@3s with `--max-depth 1`; if harm shrinks, confirmed |
| **5** | Opponent model = **`π_θ` sampled per rollout** — sharp, and in the mirror ablation *exactly correct* | α-head belief, measured **FLAT** (0.97 top-ratio, three-axis work) marginalized as a weighted mean | **PARTLY — a bias term, not variance.** Scoring every arm against a near-uniform opponent systematically misprices risk (a greedy setup looks safe against random play). Wang scored against the opponent he actually faced. | Sharpen α by temperature, or in the mirror substitute the *policy's own* distribution on the opponent's state (we control both sides) |
| **6** | Fresh determinization **per rollout** (10³ worlds), from Showdown's exact randbats generator = a true posterior | `k_worlds ≤ 8`, tiered pool donors; oracle arm pins `k=1` | **Partly** — but our own data says no: oracle (truth) ≈ honest (belief) at matched widths, so world *identity* is not binding. Wang's gain here is the **count**, which folds into #1. | Covered by #1 |
| **7** | 10 s/decision | 1–3 s | Contributory only via #1 | Covered by #1 |
| **8** | Terminal returns whenever the descent reaches game end (frequent late-game) | Terminal handled, but depth-1 almost never reaches it | Minor at depth 1; would matter at Wang's depth | Covered by #1/#4 |
| — | *Wang does NOT have:* rollout-to-terminal with a rollout policy | — | **Refutes (b) as a Wang ingredient** | — |
| — | *Wang does NOT have:* search folded into training | We also do not | Not relevant to this contrast | — |
| — | *Wang does NOT have:* mixed-strategy sampling at the root (§5.2.1 future work) | We do not either | Not relevant | — |

**One condition that is NOT a difference, and is worth stating because it makes our result
harsher:** both headline gains are *mirror ablations against an opponent the search models*.
Wang calls this an inflation in his favour. We have the same inflation available (in the mirror
the opponent *is* our policy) — and still lost 21 points. Our negative is not an artifact of a
harsher test population; it is harsher on the estimator, not on the population.

---

## 4. Quantitative reconciliation

Let `σ` be the per-leaf noise on the averaged axis and `m` the `top1−top2` gap.

| Regime | effective draws per root action | `σ_eff = 0.0115/√n` | `σ_eff / m` (m = 0.0213) | measured mirror WR |
|---|---|---|---|---|
| honest@3s (deepened) | ~1 on dice, +max-bias from depth | 0.0115 | **54%** | **0.276** |
| honest@1s | ~1 | 0.0115 | 54% | **0.294** |
| oracle@1s | ~2 | 0.0081 | 38% | **0.325** |
| oracle@3s | ~8 | 0.0041 | **19%** | **0.434** |
| *Wang (est.)* | *≥100–200, likely far more with tree reuse* | *≈0.0008–0.0012* | *≈4–5%* | *.809 (his ablation)* |

The ordering is monotone in `σ_eff/m` across all four of our cells and points at Wang's. That is
a real, four-point dose-response and it is the strongest evidence that the two results are the
same phenomenon at different noise levels.

**But read the last column honestly.** At 19% noise-to-margin we are still at 0.434. A naive
Gaussian-argmax model over ~9 arms would put a 19%-noise argmax much closer to break-even. Two
readings survive, and our data cannot choose between them:

- **(i) Pure variance, slower than expected.** The cells confound arm and budget; `oracle@3s`
  also has `k_worlds=1`, so its *world* axis is a point mass — the true `n_eff` may be lower than
  the dice count suggests. Under this reading the R-ladder crosses 0.50 somewhere around
  `r = 16–32`.
- **(ii) A bias floor.** Averaging removes variance, never bias. If `V_θ` is systematically
  mis-ordered on the *arm* dimension — e.g. because every arm is scored against a flat α opponent
  model (#5), or because the critic's resolution defect from the G0 work is a *shift* not just a
  spread — the curve plateaus below 0.50 no matter how much we average.

**These two readings imply opposite next actions**, which is why the R-ladder is the experiment
worth paying for first.

---

## 5. The three parent candidates, adjudicated

### (a) MARGIN GATING — **confirm the family, but demote the framing**

*Not literally a Wang ingredient*, but it is the crude version of #2/#3, and Wang independently
arrived at the same regime description (§5.2.1: "one action just barely edging out the other").

- **Would it help?** It would **bound the harm**, monotonically, toward 0.50. It cannot produce a
  *dividend*: as `k → ∞` the gate reduces to the policy, and win rate → 0.50 by construction.
- **Important honesty point:** with a median margin of `0.0213` and `sd = 0.0115`, a `k = 2` gate
  (`0.023`) already suppresses roughly half of all overrides. So a gate sweep whose win rate
  climbs to ~0.50 and stops proves only that *the overrides were the harm* — which we already
  know. **It is a safety mechanism, not an experiment.**
- **Cheapest form:** ~10 lines in `search.py:426` (a `--margin-k` flag comparing
  `scores[top1] − scores[top2]` against `k · σ̂`, where `σ̂` is the **empirical** per-arm SE we
  can already compute from the `r_dice` samples we collect). Re-run one cell at
  `k ∈ {0, 1, 2, 4}`; `k=0` reproduces the existing row.

### (b) ROLLOUT-TO-END LEAVES — **refuted as a Wang explanation; survives on independent grounds**

- **Wang did not do this.** His leaves are `V_θ` (§2.3, p. 20–21). Terminal returns enter
  opportunistically via deep tree descent, not via a playout policy. **Do not cite Wang in
  support of this lever.**
- **But it is the only candidate that attacks reading (ii).** If the R-ladder plateaus, bias is
  the residue and rollouts are the only unbiased anchor we own.
- **Cost is the problem.** The R1 machinery measures **792 ms @ R=8** rollout-to-end
  (`project_counterfactual_label_costs`), i.e. ~100 ms per rollout. At ~50 arms/decision that is
  ~5 s at R=1 — already over our 1–3 s budget and 15× the current per-decision leaf cost.
- **The affordable form is a PLAYOFF, not a leaf swap:** screen all arms with the critic as
  today, then spend the entire remaining budget rolling out **only the top 2** to terminal.
  ~200 ms @ R=1 per arm, ~1.6 s @ R=8 for both. This is exactly the decision the critic is
  measurably unable to make (`σ ≈ ½m`), and it costs nothing on the arms the critic *can*
  separate. Deferred until the R-ladder says bias is real.

### (c) POLICY-PRIOR REGULARIZATION (PUCT) — **confirmed, this is literally Wang, and it is the right end state**

- Wang's `U(s,a) = P[s,a]^β · √M[s] / (N[s,a]+1)` plus visit-count argmax is precisely "deviating
  from the policy requires accumulated evidence" (§2.3, pp. 21–22), and #2/#3 in the diff table.
- **We have no tree and no visit counts at depth 1**, so PUCT is not directly portable. The
  depth-1 analogue with the same guarantee is **shrinkage toward the prior**: with a per-arm
  empirical SE `σ̂_a` (free — we already average `r_dice` samples per arm) and prior weight
  `π(a|s)`, select

  `argmax_a [ V̂(a) − λ · σ̂_a + c · log π(a|s) ]`

  or equivalently a James–Stein / empirical-Bayes posterior mean shrunk toward the
  policy-weighted mean. This subsumes (a): it *is* a margin gate, but one whose threshold is set
  per-arm by that arm's own measured noise rather than by a global constant, and it is
  automatically permissive exactly where the search has enough samples to be trusted.
- **This is the correct end state.** It is also the version that would keep working when the
  R-ladder raises `n`, whereas a fixed-`k` margin gate would then be leaving evidence on the
  table.

### Two additional candidates the diff surfaced

### (d) FLAT α — the opponent model is a *bias* term we had not priced

Wang searched against `π_θ`, the opponent's actual policy. We marginalize over the v67 α-head
belief, which the three-axis work measured as **FLAT (0.97 ratio)** while the policy itself
concentrates (median top action 0.75). Marginalizing over a near-uniform opponent when the real
opponent plays its top action ~75% of the time **systematically misprices every arm** — and it
does so in a direction (under-punishing risky lines) that a mirror match will punish. This is not
variance and no amount of averaging removes it. **Cheapest test:** in the mirror we control both
sides, so substitute the policy's own action distribution on the opponent's state for α and
re-run one cell. If the harm shrinks, α is a live defect and it is a *belief-head* finding, not a
search finding.

### (e) The MAX backup adds optimism bias per ply

`deepen.py:111–133` takes MAX over our actions at interior plies. Under leaf noise this is a
biased-high estimator of the true max, and the bias compounds per ply. Wang has no max anywhere —
his backup is a running mean and his selection is a visit count. **This is a mechanism-level
explanation of the ledger's own dose-response observation** that the most-deepened cell is the
most harmed, and it predicts that depth *cannot* help until the leaves are quiet. **Cheapest
test:** re-run honest@3s with `--max-depth 1`. Free.

---

## 6. Ranked cheapest tests

| Rank | Test | Cost | What it decides |
|---|---|---|---|
| **1** | **R-ladder at Wang's budget.** `--arm oracle --opponents self --max-opp 2 --max-worlds 1 --max-depth 1 --budget 10`, `--max-dice ∈ {1,2,4,8,16,32}` | **Zero code** (all flags exist: `__main__.py:68–71`). ~6 cells of battery time. Self-controlled at `r=1`. | **The fork.** Variance (→ fixes (a)/(c) revive search now) vs bias floor (→ standing verdict confirmed; only (b) or R1 remain). Also gives the extrapolation the 4-cell dose-response only hints at. |
| **2** | **`--max-depth 1` re-run of honest@3s** | **Zero code**, one cell. | Isolates the MAX-backup optimism (e) from the width story. If harm shrinks, ban depth until leaves are quiet. |
| **3** | **Margin gate `k ∈ {0,1,2,4}`** at one cell | ~10 lines at `search.py:426` + 4 cells. | Bounds harm. Confirms overrides are the harm. **Cannot produce a dividend** — treat as a safety switch, not a result. |
| **4** | **Prior-shrunk argmax** (depth-1 PUCT surrogate) using empirical per-arm SE | ~30 lines; the `r_dice` samples needed for `σ̂_a` are already collected but currently averaged away — needs the per-arm spread retained. | The principled union of 2 and 3, and the version that scales with `n`. Run after test 1 sets the noise level. |
| **5** | **Sharpen / replace α** in the mirror with the policy's own distribution | ~20 lines in `alpha.py` + one cell. | Prices candidate (d) — an opponent-model bias term. A belief-head finding either way. |
| **6** | **Rollout-to-end top-2 playoff** | Real work: wire the R1 rollout path into the search as a second-stage scorer. ~1.6 s/decision at R=8. | Only run if test 1 plateaus. The single unbiased anchor we own. |
| **7** | *Not recommended:* persistent cross-turn tree, full PUCT/MCTS rebuild | Large. | Wang's #1 and #2 in their native form. Do not build until tests 1–4 say the leaves are quiet enough for a tree to be worth having. |

---

## 7. What would falsify this reconciliation

- **The R-ladder is flat.** If win rate does not move from `r=1` to `r=32`, the noise/margin story
  is wrong despite the four-cell dose-response, and something structural (world gating, α, the
  materializer, the CRN mapping caveat already on the ledger) is the real defect.
- **The margin gate over-shoots 0.50.** If a `k=2` gate lands *above* 0.50 with a low change rate,
  then a small, well-selected set of overrides *is* profitable and the problem is purely one of
  *which* decisions we act on — a much better position than either reading (i) or (ii).
- **A depth-1 honest@3s re-run is not better than the deepened one.** That would refute (e) and
  send the dose-response back to being width-only.

## 8. Reading notes on the sources

- Wang never reports **α or β**, the two constants that determine how hard the prior regularizes
  his search. Any attempt to reproduce his tree policy numerically is under-determined by the
  thesis; no public code repository for this work was found.
- Wang's format is **gen4randombattles**, where hidden-info sampling uses **Showdown's own team
  generator** — a luxury gen3ou does not have (already recorded in
  `designs/ai_v5/research_selective_search_compute_constrained_rl.md` §7). His absolute Elo is not
  comparable to ours; only his *ablation* is.
- Everything in `designs/ai_v5/research_selective_search_compute_constrained_rl.md` about Wang
  (inference-only search, §3.2.1 opponent-model weakness, §5.2.1 mixed-strategy gap) is **verified
  correct** against the thesis by this pass. What that document does **not** say, and what this
  pass adds, is the leaf-evaluation crux (§1.1 here) and the visit-count-argmax rationale (§1.2).
