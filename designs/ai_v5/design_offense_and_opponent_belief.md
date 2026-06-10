# Design — Offense Belief + Opponent-Team Belief (ai_v5)

> **FORWARD DESIGN — not yet built (except the crit-split base, which is in flight).** This doc covers
> the two perception levers that close the model's biggest structural blind spots, plus the architecture
> + training-signal work that makes the second one real. Written 2026-06-09 as the synthesis of a long
> design session; supersedes the standalone `design_outgoing_damage_obs.md` draft (folded into Part A
> and updated for the current arch).
>
> **Companion artifacts:** `design_incoming_damage_obs.md` (the incoming belief this mirrors) ·
> `impl_step4_incoming_damage_obs.md` (as-built incoming) · the in-flight `gen3_incoming_crit_split_v1`
> (the crit/non-crit + provenance split of the incoming belief — the immediate base) · the orphaned
> `src/agents/model/team_completion_model.py` (a hidden-team predictor, written, never wired in).

---

## 0. Thesis — where the design ceiling actually is

The model is plateaued (~87% vs bots, ELO flat in self-play). A 13-agent adversarial diagnosis + a
representation-probe harness located the ceiling in **three structural deficits — not capacity** (aggregate
critic explained-variance is a healthy 0.77; the transformer is sub-2% of CPU on an 86%-idle GPU):

1. **The obs is OFFENSE-BLIND.** We precompute a calibrated incoming-KO belief (what KOs *us*), but for
   *our own* offense the model gets only `base_power/200` + raw type-effectiveness (`our_matchups`, `/4`).
   It must re-derive its own KO math from linear inputs every forward pass.
2. **The obs is OPPONENT-BLIND.** The ~3 unrevealed opponent mons are `HP==0` and **key-masked out of the
   transformer entirely** — the model literally reasons about a 3-mon opponent. There is no belief about
   what's hiding.
3. **The trunk's only teacher is the scalar PPO advantage** — no signal forces it to encode game dynamics
   or hidden state, despite the idle GPU.

This doc designs the fixes for (1) and (2). **Part A** is the offense belief (the symmetric mirror of the
incoming block). **Part B** is the opponent-team belief — and because (2) is as much an *architecture* +
*training-signal* problem as an obs problem, Part B spans the attention mask, a symmetry-broken hidden-team
representation, a self-predictive objective, and a surprise-weighted learning signal. **Part C** is the
flag-free, one-run-per-change build sequence. Deficit (3)'s broader fixes (a forward-model aux head, a
distributional critic, PFSP) are the wider roadmap and are referenced where they intersect.

**Non-goal, explicitly ruled out:** inference-time search (MCTS). Every lever here makes the trained model
better — weights, inputs, objectives — at fixed inference cost.

---

# PART A — Outgoing-KO belief (fixing offense-blindness)

## A1. Problem — the precompute asymmetry

KO math (`floor(2L/5+2)·power·Atk/Def / 50 · STAB · eff · weather · screen · roll ≥ HP`) is the core of
every decision. A 2-layer ReLU body + attention **cannot cheaply reconstruct it** from the linear inputs we
expose: the **product/quotient of variable stats** (`power·Atk/Def`) is what ReLU approximates poorly (the
threshold `≥ HP` is the easy part). We already pay this cost *for the opponent* (the incoming block collapses
it into P(KO)), but make the net re-derive **its own**. The documented loss buckets are the symptoms:

| Failure mode | The outgoing belief that addresses it |
|---|---|
| **FUTILE_ATTACK** (attacked a wall / resisted / recovering target) | low `pko` *and* low `expdmg` vs that defender → "this doesn't dent it" |
| **PASSED a lethal KO** (healed/set up instead of taking the kill) | a `pko ≈ 1.0` slot on the opp active → "a free KO is on the table" |
| **SELF_KO into a 0× target** (Explosion into Ghost/immune) | the fixed/immune branch reads `pko = 0` → "this self-KO whiffs" |
| **walked a won line into the one wall** (bad/under-switch) | low `pko` across the opp's likely walls → "nothing here KOs the thing in front of me" |

**Why this lever is special:** unlike the incoming belief (which the critic reads but the *policy argmax*
can't directly act on), an outgoing "I have a lethal move HERE / this attack is futile" signal speaks
**directly to the argmax over the 4 move slots** — it changes which *action* looks good, not just the state
value. It is the single highest-confidence, lowest-risk transformational win, and it is independent of
everything in Part B.

## A2. The reframe — compute the belief, not the damage

We compute a **calibrated belief about securing a KO**, integrated over the opponent's hidden *defensive*
set. Precision under hidden def/spd/HP/item is impossible and not the goal — **calibration** is. The human
read: *"this OHKOs unless it's the max-bulk set or a surprise resist."*

- **We compute** (the unlearnable nonlinearity): per opp defender, the expected-damage fraction + P(KO) for
  our active, integrated over the opp's def/spd/HP/ability prior, conditioned on reveals.
- **The model learns** (its strength): whether to *take* the line given that one exists — weighing the KO
  against board state, hazards, and the *incoming* answer (already in the obs).

This is the same justified provide-the-fact exception the incoming belief took: the KO threshold is the
nonlinearity the small net can't synthesize, and the defensive-set prior isn't derivable in-battle.

## A3. The side-neutral core (the DRY heart) + crit-split reconciliation

`incoming_damage.py` is already ~90% side-neutral (`gen3_damage_max`, `p_ko`, `percentile`, `weighted_mean`,
`weather_damage_mult`, `_channel_threat`, `compute_team_block` carry no "incoming" coupling). **The
engineering bet:** ONE side-neutral damage-belief core serving both directions; the asymmetry is purely
*where the uncertainty lives*.

| Quantity | INCOMING (shipped) | OUTGOING (this design) |
|---|---|---|
| Attacker | opp active — **hidden** (usage dist → tail) | **our** active — **exact** (P=1) |
| Defender | **our** mon — **exact** | opp mon — **hidden** (usage dist → bulky tail) |
| candidate moves | revealed ∪ usage-prior | **our 4 known moves** (no priors) |
| screens | our Reflect/Light Screen | **opp** screens |
| ability | exact (our mon) | **distribution** (collapses on reveal) |

**Generalizations (the only real new code):**
1. **`StatBelief(mean, conservative)` per stat.** The exact side sets `mean == conservative`; the hidden
   side reads a usage percentile. Incoming's hidden *attacker* uses the **high** Atk (more threat); outgoing's
   hidden *defender* uses the **high** def/spd/HP (harder to KO — *don't claim a KO a max-bulk spread
   survives*). The tail *direction* flips; the principle doesn't. Retires the incoming-only `mean/tail`
   ratio shortcut for expected-damage (it only worked because damage is ~linear in Atk; it does **not**
   transfer to the Def/HP denominators) → unify on the **honest mean-belief** `gen3_damage_max(.mean, …)`.
   This shifts incoming expdmg values by integer-floor rounding (a move *toward* correctness; re-validate
   the incoming fuzz + golden fixture).
2. **Defender ability *distribution*, marginalized at the P(KO) level** (`Σ q · p_ko(dmax | ability)`, NOT
   averaging effectiveness then thresholding — the threshold is nonlinear). An immunity hypothesis
   contributes `p_ko = 0` and drops out, generalizing incoming's `× (1 − P(immune-ability))`. Hypotheses are
   1–3 and `effective_multiplier_by_types` is memoized; group by distinct per-type `eff`.
3. **Defensive-stat + absolute-HP beliefs.** Opp HP is known only as a **fraction** (`current_hp/max_hp`
   exact, but `stats["hp"]=100` is the percent denominator, not the real max), so `hp_remaining =
   hp_fraction × believed_max_HP`. Extend `priors.stat_distribution` from atk/spa/spe to **def/spd/hp** (the
   data already exists — `gen3_spread_priors.json` carries all six EV columns) + a **new `gen3_hp_stat`**
   (`2·base + 31 + ev//4 + 110`; no nature term — structurally different from `gen3_stat`). Conservative
   percentile `_CONSERVATIVE_TAIL_Q = 0.85`, built modular (parametrize `_stat_belief(species, stat, q)` so
   adding a P50/median channel later is additive, not a refactor).

**Reconciliation with `gen3_incoming_crit_split_v1` (the in-flight base):** the crit-split already lives
*inside* the core — `_channel_threat` now returns the no-crit + crit lines per channel. So generalizing to
side-neutral **carries the crit-split to both directions for free**: outgoing also gets a modal no-crit line
and a crit signal (our crit can secure a KO too). **Keep the two blocks isomorphic** — whatever final P(KO)
encoding the crit-split lands (`[nocrit, crit]` vs the review's recommended `[nocrit, crit_delta]`, see Part
C #1), outgoing emits the *same* shape. The **rename** `incoming_damage.py → damage_belief.py` (the file
serves both directions) happens in this same retrain-class pass; two thin encoders
(`incoming_damage_encoder.py`, new `outgoing_damage_encoder.py`) own only their direction's reads.

## A4. Feature layout (mirrors the final crit-split shape)

Per opp mon, mirror the crit-split P(KO) split (no `p_outspeed` — already the incoming active slot; no recovery
scalars — our recovery is exact and in the move-effects block). With the review's `[nocrit, crit_delta]`
recommendation that is:

```
[phys_expdmg, spec_expdmg, phys_pko_nocrit, spec_pko_nocrit, phys_pko_crit_delta, spec_pko_crit_delta, defender_known]
```

`OUTGOING_PER_MON = 7` (tracks the crit-split's P(KO) shape exactly — if the crit-split keeps absolute
`[nocrit, crit]`, outgoing does too). `defender_known` is the **provenance** scalar, the outgoing analog of
incoming's `threat_revealed`: 1.0 when the opp mon's defensive set is revealed/known, the usage-prior
concentration when it's a guess — the "how much are we guessing about their bulk" signal (subject to the
same review note about its 0.0 semantics). Unrevealed/fainted opp slot → zeroed; opp active behind a Sub →
zero its `pko` (the hit eats the Sub).

**Placement:** inside the reactive block, **after the incoming block, before the matchups**, so the whole
slice routes through `non_matchup_rest` → both heads. Offsets derive from named constants (never hardcode):
`OUTGOING_DMG_OFFSET = REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM + INCOMING_DMG_DIM` (= 102 on the crit-split
base), `OUTGOING_DMG_DIM = TEAM_SIZE · OUTGOING_PER_MON` (= 42), matchups shift by `OUTGOING_DMG_DIM`, obs
`3409 → 3451`. Bump `ARCH_SIGNATURE → gen3_outgoing_ko_v1`. Update the prober offset pins + `decode` the new
block.

## A5. Validation (gates)

- **Gate 0 — Phase-0 falsifier (model-free, before code).** `falsifier_missed_ko_attribution.py` (mirrors
  `falsifier_cliff_attribution.py`): on the existing loss corpus, decode `our_matchups` + base-powers + opp
  HP and bucket each loss turning-point into **A_PASSED_KO / B_FUTILE_ATTACK / C_SELF_KO_WHIFF /
  D_NOT_ADDRESSABLE**. GO if `A+B+C` is a meaningful share of decisive loss turning-points **with
  loss-vs-win discrimination**, robust across thresholds.
- **Gate 1 — calibration fuzz** on `data/teams/`-pool bridge battles: ground truth = `P(KO is AVAILABLE)`
  via a counterfactual oracle (replay each of our moves vs the actual defender), **not** `P(KO occurred)`
  (confounded by our policy). Reliability sliced by knownness; over-claiming in the 90–100% bucket = fail.
- **Gate 1b — invariants fuzz** (mandatory; mirrors `incoming_damage_fuzz_test.py`): width, finiteness,
  ranges, no-opp-active ⇒ zero, unrevealed/fainted/Sub ⇒ zero, block fires on the majority of opp-active
  decisions.
- **Gate 2 — unit tests:** `gen3_hp_stat` exactness; def/spd/hp `stat_distribution`; defender-ability
  marginalization (Levitate Claydol drops Ground P(KO) by P(Levitate); revealed ability → point); bulky-tail
  conservatism; honest mean-exp; Explosion Def-halve when *we* Explode; fixed-damage respects opp immunity.
- **Gate 3 — obs-build benchmark** (`lru_cache` the per-species defender belief; ≤4 attacker candidates is
  the budget headroom). **Gate 4 — golden fixture + parity** (regenerate). **Gate 5 — model round-trip +
  `--debug` smoke.** **Gate 6 — post-retrain efficacy:** outgoing saliency rises; **FUTILE_ATTACK + passed-KO
  turning-points fall, SELF_KO-into-immune → ~0**; because outgoing speaks to the argmax, Gate 6 should move
  the **policy mix** (more KOs taken, fewer wall-attacks), not only the critic.

---

# PART B — Opponent-team belief (fixing opponent-blindness)

This is one capability built in four stacked pieces: **(B1)** give the hidden mons a representation at all;
**(B2)** make that representation specialize instead of collapsing; **(B3)** teach it what the hidden team
is, self-supervised; **(B4)** weight that learning by how much each hidden mon *mattered*. **(B5)** is the
exposure bias that cuts across all four.

## B1. The masking bug — stop deleting the hidden mons

**Mechanism (verified):** the transformer's `key_padding_mask` excludes every opp team slot with `HP==0`
(`features_extractor.py` `fainted_mask_opp = (opp_slot_HP == 0)`). An **unrevealed** mon also has `HP==0`,
so it is masked **identically to a fainted one**. The encoder already builds a perfectly good "unknown mon"
token — `species_known=0`, `spread_known=0`, ability-unknown (`pokemon.py:144`) — *specifically* so the
model can tell "unknown opponent" from "zeros on a known mon"... and then the mask **throws it away before
the transformer sees it**. The information isn't missing; it's deleted one step downstream.

`HP==0` is doing double duty — "dead" (revealed-then-fainted → correctly absent) vs "never seen" (present,
unknown). The fix distinguishes them, which is cheap because the bit already exists:
- **mask** `HP==0 AND species_known==1` (genuinely gone),
- **keep unmasked** `species_known==0` (present, unknown).

This alone gives the model what it lacks: the ability to attend to "they have N unknowns left," count them,
and be *appropriately uncertain* — which directly attacks the confident-V-then-crater pattern (you can't
claim a won position with 3 unknowns on the board). It is a one-line change to the mask logic, no new dims.

**Design principle surfaced:** a hard `HP==0` filter is a blunt prior that overrides the learned attention.
The attention keys/queries are already fully learned from the whole token; we should let them *decide* a
dead/unknown mon's relevance (and still *use* the fact of its deadness/unknownness as signal), not hard-delete
it. Mask true padding; don't mask game state.

## B2. Symmetry breaking — why the hidden slots collapse, and the fix

Once unmasked, the N unknown slots are **identical** (zeros + `species_known=0`). A transformer is
**permutation-equivariant** → identical inputs are *forced* to identical outputs. So the slots collapse to one
representation: the model can know "there are unknowns" (B1's win) but **cannot** represent "slot A leans
physical sweeper, slot B leans special wall." No training fixes this — the symmetry is architectural.

**Rejected: `E[species embedding]`.** The distribution over the last mon is **multimodal** (a few discrete
archetype-completions); the mean of embeddings is a **phantom** mon in a low-density region, and embedding
midpoints aren't semantic (½Aero+½Gross ≠ "half-fast-half-bulky", it's noise). A confident vector for a mon
that can't exist is *worse* than masking.

**The fix — distinct learned hidden-role query tokens** (DETR object queries / Slot Attention): K learned,
*different* query tokens, non-identical by construction, that attend over the revealed team. Trained against
the actual hidden team, each **specializes** (one drifts to "the likely physical threat," another to "the
special answer"). Two requirements make it capture the *structure* humans use:
1. **The queries attend to the revealed mons AND each other** (the transformer does this natively) — so they
   *coordinate*: "the board is rock-leaning → I, the second slot, lean rock too" = **overload** (Ttar →
   Aero); "lead Skarm → a special wall is likely" = **coverage**. Independent per-slot marginals would miss
   this; the joint attention is what makes overload predictable.
2. **The loss scores the joint set** (set-matching / Hungarian-style), not independent per-slot CE — else you
   optimize away the very correlation that makes the hidden team predictable.

**Hosting it — a hidden-opponent CLS token.** The hidden mons are an unordered *set*, so a single learned
"opponent hidden team" CLS query (sibling to the existing `our_cls`/`their_cls`/`value_cls`) that summarizes
the belief over the whole unrevealed remainder is cleaner than per-slot prediction (it sidesteps the
which-unknown-is-which assignment problem). The policy/value read it. Because it's a learned 128-d vector
trained to predict a distribution, it can hold a **multimodal** belief implicitly — *without ever
materializing a mean mon*. (For richer per-slot structure, K distinct queries; start with one CLS + the
set/role loss and escalate only if the probe shows headroom.)

## B3. The training objective — self-predictive (BYOL), de-risked via a supervised first step

We do **not** want to grade against hand-defined human role labels — roles are fuzzy for a human to specify,
and a *latent* target lets the model discover its own similarity structure (interchangeable-in-context mons
end up close → "role" emerges as a region of the space, DINO-style, with zero labels). This is the **BYOL /
SPR (self-predictive representation)** family: predict the embedding of the hidden mon, grade it against the
model's own (EMA) embedding of the revealed mon, and let the latent get richer over time — *self-bootstrap*.

**The one failure mode that decides everything: representation collapse.** Predict-your-own-latent has a
trivial optimum — make all embeddings constant (loss 0, representation useless). The field's anti-collapse
mechanisms, and the one we lean on:
- **EMA target (BYOL):** grade against a *slowly-moving copy* of the network with a stop-gradient. The target
  drifts forward (it *gets richer over time* — the self-bootstrap) but too slowly to collapse onto. This is
  the mechanism that makes "self-bootstrap" and "anti-collapse" the *same* thing.
- **Stop-gradient + predictor (SimSiam); variance/covariance regularizers (VICReg)** — alternatives/adjuncts.
- **Pokémon is unusually collapse-resistant for free:** the species embedding is *also* used by the
  policy/value/damage-belief, which keep it rich and distinct (Aero and Blissey *cannot* collapse — the rest
  of the net needs them apart). So we predict toward a **task-anchored** target, a second safety net.
- **Collapse-proof alternative — contrastive (InfoNCE):** predict *closer to the true revealed mon than to
  other candidate mons*. Negatives make a constant solution lose every comparison, and force *discrimination*
  ("rock attacker vs special wall" specifically) — often the better choice here.

**De-risked build order (this is why B3 is two runs, split by *objective* not by structure):** K-CLS queries
with **no** objective are just untrained capacity (the RL gradient barely shapes them) → would show ~nothing
and falsely kill the thread. So bundle the structure with an objective, and make the *first* objective the
safe one:
1. **Species-ID prediction** (cross-entropy against the actual revealed species — still **label-free**, the
   species is ground truth, and **collapse-proof** because the target is discrete). Establishes "does a
   team-belief module help?"
2. **Swap to BYOL latent** (the self-bootstrap). Isolates "does the self-predictive latent beat the safe
   supervised target?" — with step 1 banked as a fallback if BYOL collapses/underperforms.

**Labels are free in self-play** (the trace carries the full opponent team), independent of in-game reveal —
so the *supervision* exists even for mons that never came out.

## B4. Surprise-weighting — let the value head decide what's worth predicting

Weight the team-prediction loss by the **value drop when a hidden mon is revealed** ("how much did
not-knowing this cost me"). The critic supervises *what to care about* → the predictor spends capacity on the
mons that **decide games** (the surprise sweeper) and ignores the irrelevant ones (the 6th mon in a game you
already won). This is **Prioritized Experience Replay's** principle and, literally, the brain's
**reward-prediction-error-gated plasticity** (you remember the sweep because surprise widened the learning
window).

**The trap that mirrors collapse — epistemic vs aleatoric surprise.** A drop from a **revealed hidden mon**
is *epistemic* (knowable, learnable → upweight). A drop from a **crit/freeze/roll** is *aleatoric* (noise →
upweighting it teaches the model to chase coinflips, re-importing the exact RNG over-weighting the crit-split
fights). So **gate the weight on the reveal event**, not on any value drop — for the team-prediction loss
this is automatic (it fires only on reveals). Two refinements:
- **Propagate the surprise backward** to the *pre-reveal* states (credit assignment — the learnable target is
  the earlier states where the mon was hidden but predictable, "you should have seen this coming"). A
  discounted future-reveal-drop weight.
- **On-policy → per-sample loss weight, not replay** (PPO uses each transition once). Mild/annealed (a strong
  reweight can overfit the tail at the expense of average calibration). **Self-extinguishing:** as the model
  anticipates better, the drops shrink → the weight fades exactly where it's solved — a curriculum that
  writes and dissolves itself.

This is also the *principled, epistemic-gated* version of the "tail-weighted value loss" critic lever: the
reveal-gating is what makes tail-weighting safe (it doesn't sweep up the RNG).

## B5. The 6th-mon exposure bias (cuts across B1–B4)

The least-understood mon is *also the most game-deciding* (the hidden-last-mon crater) — not a coincidence,
the same bias seen from the outcome side.
- **Input bias (real, unfixed):** games end before full reveal, so "5 revealed, predict the last" states are
  rare — the model is best-tuned for early game and *least*-tuned exactly where it has the most context to
  nail the last mon. Most predictive power available where the data is thinnest.
- **Label bias (mostly fixed by self-play):** the full opponent team is in the trace, so the *label* for the
  6th mon is always available even if it never appeared. The problem is the input distribution, not the
  supervision.
- **Mitigation:** weight the team-prediction loss by **reveal-count** (oversample high-reveal states), so the
  decisive late-reveal regime gets the emphasis its consequence deserves. B4's surprise-weighting *also*
  concentrates on the decisive reveals; and the self-bootstrap means each sparse late example carries more
  signal as the latent matures.

**Re-home the orphaned predictor.** `src/agents/model/team_completion_model.py` +
`training/team_completion/{team_dataset,replay_parser}.py` + `main/train_team_completion.py` already exist
and are imported by *nothing* in the agent pipeline. Step 1 of B3 should read its formulation (per-slot
species vs set vs roles) and either extend it onto the shared-trunk CLS head or restructure — it is half the
build, sitting unused.

---

# PART C — Build sequence (flag-free, one run per change, ~60M steps ≈ 1 day each)

The owner's call: **no flag-guarding** — each change is clearly directional, so ship it always-on and A/B by
running the new arch against the prior one. Each run = the last *accepted* arch + one change; version-checked
(`model_config.json` + `ARCH_SIGNATURE`); each carries a falsifier so a day's run gives a verdict.

| # | Change | ARCH | Falsifier / "did it work" |
|---|---|---|---|
| 1 | **Crit-split** (in flight; emit `[nocrit, crit_delta]` per the review) | `gen3_incoming_crit_split_v1` | A/B vs current; ELO ≥, td-tail trend, crit-belief saliency read |
| 2 | **Outgoing-KO belief** (Part A) | `gen3_outgoing_ko_v1` | Gate-0 falsifier first; post-retrain FUTILE_ATTACK + passed-KO turning-points fall; policy takes more KOs |
| 3 | **Unmask unrevealed** (B1) | behavioral bump | model stops valuing a 3-mon opponent; hidden-last-mon `critic_blindspot` share shrinks |
| 4 | **Team-belief module — species-ID** (B2 + B3 step 1) | arch bump | probe `opp_has_species`/role > chance; surprise_ohko + hidden-mon craters fall |
| 5 | **Swap to BYOL latent** (B3 step 2) | arch bump | beats the species-ID version on the probe **without** collapse (watch embedding variance) |
| 6 | **Surprise-weighting** (B4) | objective | the hidden-mon-crater bucket shrinks *faster* per step |

Notes: (2) is independent and the highest-confidence win — **strongly consider running it before the Part-B
thread** (it shares the damage-belief core with the crit-split — one refactor — and fixes the policy argmax
directly). (4) bundles structure + objective on purpose. (6) is strictly last (it weights (4/5)'s loss).
Metric throughout = **anchored ELO** + prober `triage`/`probe` (NOT `win_rate_vs_pool`, gate-pinned ~50%).

---

## Appendix — load-bearing code facts (verified this session)

- **Masking:** `features_extractor.py` `fainted_mask_opp = (opp_slot_HP == 0)` masks *both* fainted and
  unrevealed (active force-unmasked). The encoder already emits `species_known` (`pokemon.py:144`,
  `spread_known`, ability-known) — the distinguisher exists; the mask discards it.
- **Attention keys are learned** (`W_q/W_k/W_v` over the full token); HP only drives the *separate*
  `key_padding_mask`. So "let the model use any information" already holds *except* for the hard HP-mask.
- **Offense inputs today:** only `move.base_power/200` (`reactive.py`/`moves.py`) + `our_matchups` =
  type-eff `/4` (`reactive.py:367`). No KO threshold for us. `grep outgoing/our_pko` ⇒ zero encoders.
- **Damage core is side-neutral** already (`compute_team_block(defenders, attacker, n_slots)` is generic);
  only `p_outspeed` + recovery scalars are incoming-specific extras.
- **The data exists:** `gen3_spread_priors.json` carries all six EV columns; `priors.stat_distribution`
  returns `()` for non-atk/spa/spe (`priors.py:75`) — only the guard widens + `gen3_hp_stat` is new. Opp HP
  is a fraction (`pokemon.py:557-562`); absolute max HP is the belief.
- **Orphaned predictor:** `team_completion_model.py` exists, wired into nothing.
- **Reward is already decomposed** into 34 named terms (`RewardBreakdown`) then summed — free labels for a
  reward-component aux head (the wider roadmap's distributional-critic lever).

## Appendix — references

- Incoming belief: `design_incoming_damage_obs.md`, `impl_step4_incoming_damage_obs.md`; the crit-split base
  `gen3_incoming_crit_split_v1`.
- Self-predictive representation: **BYOL** (Grill 2020), **SimSiam** (Chen & He 2020), **DINO** (Caron 2021,
  emergent structure), **VICReg/Barlow Twins** (variance/covariance anti-collapse), **SPR** (Schwarzer 2021,
  BYOL-for-RL). Contrastive: **InfoNCE**.
- Set prediction / symmetry breaking: **DETR** object queries (Carion 2020), **Slot Attention** (Locatello
  2020).
- Surprise-weighting: **Prioritized Experience Replay** (Schaul 2015); reward-prediction-error-gated
  plasticity (Schultz, dopaminergic RPE).
- Wider roadmap (deficit 3): forward-model aux head, distributional/quantile critic + reward-component head,
  PFSP league + exploiter — see the frontier-roadmap memory.
