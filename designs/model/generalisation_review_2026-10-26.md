# Does the architecture limit generalisation to unseen sets and teams? — review, 2026-10-26

A dated review (not always-current). The question: our binding gap is unfamiliar teams (P\* scores 0.487 against
Kakuna on the three unfamiliar panels, 0.598 on our own home72). More steps, pool freshness, older snapshots and toy
depth have not moved it; D1 (forum teams in training) reads +2.1 pp, undecided, mostly on its own population; D3 (set
mutation) is training. Is the ARCHITECTURE what limits transfer to sets and teams the network has not trained on?

Abbreviations: P\* = `rb_x26_s2601` @ 280,000,170 steps (the X26 plateau); home72 = 72 of our training-pool teams;
H = `heldout72_hpfix`, F = `forum72`, P72 = the procedural panel (the three UNFAMILIAR panels, X51's U); S = a mon
token's static identity half, D = its dynamic-state half (`ARCHITECTURE.md` section 2.1); KL = Kullback–Leibler
divergence of the policy (nats); Δv = change of the critic's P(win); hit@k = the share of the opponent's k hidden
species among the belief's top k; pp = percentage points; CI = 95 % interval.

## Answer

**For OUR OWN team, no: the network represents a set it has never seen compositionally, and treats it like a seen
one. For the OPPONENT, yes: the belief stack's learned corrections memorise the training pool, and on an unfamiliar
team they are worse than the Smogon prior they correct.**

1. **Our own sets are not the problem the architecture has.** The team is permutation-EQUIVARIANT to float precision
   (permuting our bench moves the switch probabilities with their mons, max |Δp| 1e-5, Δv 0, 600 rows). At a matched
   difference from a pool set, an unfamiliar set's S token lies as far from the pool's sets as the pool's
   own sets do from each other (one move changed: 4.35 pool, 4.34 / 4.34 / 4.30 on H / F / P72). Swapping a
   never-seen bench mon's set for an UNFAMILIAR one moves the policy only slightly more than swapping it for a
   FAMILIAR one matched on the same difference: KL +0.012 [+0.001, +0.023] on 0.060 (≈ +19 %), Δv no different
   (+0.0004 [−0.002, +0.003]). Nothing in the token, the trunk or the heads reacts to "unseen" as such.
2. **The opponent belief is where the architecture memorises.** Each belief leg is a Smogon prior plus a LEARNED
   delta trained on pool labels (and shaped by the policy gradient, `belief_grad_mode` `shaping`). Against a
   familiar opponent the deltas add a lot: hidden-species hit@k 0.71–0.76 against the prior's 0.34–0.49. Against an
   unfamiliar one they add nothing or harm: hit@k 0.26 / 0.37 / 0.15 (F / H / P72) against the prior's 0.30 / 0.39 /
   0.16, and the posterior is CONFIDENTLY WRONG (mean log-presence of the true hidden species −6.2 / −4.6 / −10.0
   against the prior's −2.4 / −2.0 / −3.2). The same holds, smaller, for a revealed mon's unrevealed moves
   (precision 0.54 / 0.48 / 0.43 against the prior's 0.56 / 0.51 / 0.51; familiar 0.65–0.70 against 0.55–0.59). X5
   then turns the top-k species into CONCRETE hypothesis tokens and the damage operator prices those species, so on
   an unfamiliar team 63–85 % of the hidden slots hold the wrong mon (24–29 % on a familiar one).
3. **So the piloting deficit is not explained by the architecture's handling of sets.** X37a put most of the Kakuna
   gap in PILOTING (+0.123 [+0.039, +0.206]) and little in FACING (+0.017 [−0.062, +0.095]); the belief failure is a
   FACING channel. What these probes cannot see is team-level PLAN memorisation (a policy that has learned what to
   do with each of 719 teams); a local perturbation cannot detect it. The place in the architecture where it is
   cheapest to store is the flat policy tower (1.12M of the learner's 2.61M parameters), an MLP over pooled team
   summaries.

The ranked levers are in section 4. The top one, anchoring the belief deltas to their Smogon prior, is cheap and
aimed at the one channel these probes show memorising; offline, a fixed quarter-weight mixture with the prior keeps
every familiar ranking and removes most of the unfamiliar calibration damage (P10). Whether that wins GAMES is D0f's
read (prior-only against full), unrun.

## 1. What the architecture does with a set

### 1.1 Our mons: identity embeddings beside a few compositional features

A mon's S token is `role_encoder` (a 2-layer MLP) over (`src/agents/model/static_tokens.py:207-223`): the species
embedding (400 × 32, `src/agents/model/extractor_ctx.py:284`), the six level-100 actual stats ± std and
`spread_known` (13 columns, `:155-170`), the item embedding, the two type embeddings SUMMED, the two ability
embeddings, the four moves through the move network and a within-mon self-attention, SUMMED as a set (`:190-206`),
`species_known` and the Hidden Power block. A move enters the move network as its id embedding, its type embedding,
its power / secondary / recoil / category / max PP / accuracy columns, its known bit, the Hidden Power type
probabilities and (P\* builds it) the move LATENT, an MLP over the id embedding, the type embedding and the
mechanics table `MOVE_ATTR` (`src/agents/model/encoders.py:24-57`).

So every entity reaches the token twice, as a learned IDENTITY (an id embedding) and as hand-given FEATURES (stats,
types, move mechanics). Nothing structural forces either. Probe P2 shows which one P\* uses.

### 1.2 The opponent: a species-level prior and three learned deltas

- **Stats.** An opponent's S reads the Smogon usage-weighted stat mean ± std of its SPECIES (`STAT_PRIOR`,
  `static_tokens.py:152, 166`), never its set; its spread belief is a separate head.
- **Hidden species.** The X5 hypothesis set scores `log P_T0(s | revealed) + δ_θ(s | ctx)`
  (`src/agents/model/hypothesis_set.py:591`): a Smogon teammate prior plus δ_θ, a Deep-Sets pool over the REVEALED
  opponent tokens and the board inputs through a 2-layer MLP (`:545-553`, zero-init last layer `:534`). Its top-k
  species become concrete hypothesis tokens in the hidden slots and the operator's priced roster.
- **Moves and items.** `MoveBelief` adds a per-slot learned delta to the Smogon move prior (`belief_heads.py:314-321`)
  and `ItemBelief` one to the Smogon item prior (`belief_heads.py:508-516`).
- **Training.** All three deltas are supervised on the opponent's TRUE set in self-play (coefficients 0.05,
  `designs/production_config.json`) and shaped by the policy loss (`belief_grad_mode` `shaping`). In self-play every
  opponent pilots a POOL team, so the labels are the pool's own conditional distribution. A delta that learns
  "revealed Skarmory + Tyranitar ⇒ this pool team's other four" is exactly what the loss rewards.

### 1.3 Permutation structure

The team tokens carry a side token-type embedding and NO slot position (`src/agents/model/team_transformer.py:443-445`);
the pools are attention pools with learned queries (`src/agents/model/pools.py:20, 93`); the policy context is
`[our_cls, their_cls, our active, 6 hidden-opponent queries]`, all slot-invariant; each switch is scored from its
own team token by one shared scorer. Moves inside a mon are a Deep-Sets sum and the two types a sum. The design
claim "permuting the team permutes the logits" (`ARCHITECTURE.md` section 3.3) holds on P\*: probe P9.

### 1.4 Where capacity sits (P\*, counted)

| part | parameters | share of the 2,609,183-parameter learner |
|---|---:|---:|
| the flat policy tower: `projection` (LayerNorm → `Linear(1152, 512)`) + the actor MLP `[512, 512]` | 1,117,952 | 43 % |
| the trunk (`team_transformer`, 3 rounds) | 401,536 | 15 % |
| the opponent-belief stack (`hidden_opp_belief`, `belief_head`, `hypothesis_builder`, `item_belief_head`, `move_belief`, `spread_belief`, `hp_type_belief_head`) | 531,750 | 20 % |
| the CLS pools (`cls_pool`) | 199,296 | 8 % |
| the mon encoder (`pokemon_encoder`: S 74,624 in `role_encoder`; D 33,792) | 131,424 | 5 % |
| the embedding tables (species, move, item, ability, type) | 30,720 | 1 % |
| the pointer head | 55,043 | 2 % |

The ride-along heads (3,014,996, training-only, detached inputs) are outside the learner. There is no dropout
anywhere in the forward and the optimiser's weight decay is 1e-5 (`designs/production_config.json`).

### 1.5 Can it represent "a set I haven't seen"?

For OUR team, yes: S is a function of the set's parts, and probes P3 / P8 show an unseen combination of seen parts
lands where its parts put it. An unseen ENTITY (species, move or item id) reads an embedding row training barely
touched, but only 9 of the 72 held-out teams carry one (D0a), so it is a small share of the gap (H8, killed as a
major cause). For the OPPONENT, the belief can represent "an unusual set" only after the reveals contradict its
expectation, and P5 shows the learned deltas then hold to the POOL's expectation rather than the prior's. Nothing in
the architecture signals "this opponent is not a pool team".

## 2. Probes

All on CPU, P\* loaded through `load_foreign_opponent`, read-only on `models/`. Scripts and JSON results:
`designs/research_state/measurements/arch_generalisation_2026-10-26/`; the decision rows (5.6 MB, not committed,
`rows_sha256.txt` pins them) are copied to `/home/goodlad/gen3ai_archive/arch_generalisation_2026-10-26/` and
regenerate with `collect.py` in about two CPU minutes (deterministic by construction; byte identity of a re-run is
**UNVERIFIED**: a multi-threaded CPU forward can break a near-tie of the greedy argmax differently).

**The rows.** `collect.py`: P\* greedy against P\* greedy, the Rust env core with its belief / item label columns.
Player 1 pilots a PANEL team (the panel's first 48 teams, in order), player 2 a seeded draw from home72. 48 games
per panel, 3,349–3,757 decision rows of both sides. So side 0 PILOTS the panel's team against a FAMILIAR opponent,
and side 1 pilots a familiar team against the PANEL's team (an UNFAMILIAR opponent on F, H and P72). Descriptive:
48 games per panel, one checkpoint, one seed.

### P1 / P2 — what S reads (`static_probe.py`)

Each input block's own share of the first layer's pre-activation variance, `trace(W_b Cov_b W_bᵀ)`, over the six
own slots of every pool team at its first decision (4,314 mons):

| layer | identity embeddings | compositional features |
|---|---|---|
| S `role_encoder` | species 29.1 %, type set 25.8 %, abilities 12.7 %, item 3.7 % | the move-set pool 28.8 % (itself mostly identity, below); the 13 stat columns **0.06 %** |
| move network | move id 19.7 %, move type 24.2 %, latent 54.3 % | power / secondary / recoil / category / PP / accuracy **1.8 %** |
| move latent | move id 42.1 %, move type 53.8 % | `MOVE_ATTR` (the mechanics table) **4.1 %** |

**P\* encodes its mons almost entirely through learned identity embeddings and types; the explicit stats and move
mechanics are nearly unread by the token** (they are read by the damage operator instead). A mean ablation agrees
on the identity half (P7).

### P3 / P4 — set novelty and S-token geometry (`static_probe.py`)

How new the panels' sets are to the pool (432 mons each):

| panel | species in the pool | the four moves in the pool | moves + item | exact set |
|---|---:|---:|---:|---:|
| home72 | 1.000 | 1.000 | 1.000 | 0.831 |
| H | 0.984 | 0.380 | 0.359 | 0.088 |
| F | 0.988 | 0.701 | 0.690 | 0.215 |
| P72 | 0.981 | 0.308 | 0.278 | 0.042 |

The unfamiliar panels are familiar SPECIES with unfamiliar SETS (the D3 target). Median S-token distance between a
set and a pool set of its species, by the difference between them (every pair; the pool against itself, the panels'
non-pool sets against the pool):

| difference | pool | H | F | P72 |
|---|---:|---:|---:|---:|
| spread only | 0.058 | 0.105 | 0.067 | 0.083 |
| 1 move (+ spread) | 4.349 | 4.339 | 4.344 | 4.296 |
| 1 move + item | 7.699 | 8.148 | 7.674 | 7.569 |
| 2 moves (+ spread) | 7.003 | 6.604 | 6.785 | 6.475 |
| 3 moves | 9.123 | 8.413 | 8.598 | 8.015 |
| 4 moves | 10.492 | 10.096 | 10.125 | 9.485 |

(The median ‖S‖ is 12.3–12.8 in every group.) **At a matched difference, unseen sets sit where seen ones do: the encoder interpolates.**
One move is a large step (about a third of the token's norm) and a spread change almost none.

### P5 — the belief's learned deltas against their own prior (`analyze_rows.py`)

Per opponent: the posterior P\* acts on against the Smogon prior inside it (the delta zeroed; the same fixed-mass
construction for species). FAMILIAR = facing home72; UNFAMILIAR = facing the panel.

| | familiar (home72, both sides; side 0 of F / H / P72) | unfamiliar F | unfamiliar H | unfamiliar P72 |
|---|---|---|---|---|
| hidden species hit@k, posterior / prior | 0.71–0.76 / 0.34–0.49 | **0.258 / 0.296** | **0.367 / 0.392** | **0.152 / 0.158** |
| mean log-presence of the true hidden species, posterior / prior | −0.66 … −0.74 / −1.73 … −2.16 | **−6.24 / −2.36** | **−4.63 / −1.98** | **−10.04 / −3.24** |
| unrevealed moves of a revealed mon, precision@u, posterior / prior | 0.63–0.70 / 0.55–0.59 | 0.545 / 0.562 | 0.475 / 0.509 | 0.428 / 0.506 |
| the same, BCE per true move, posterior / prior | 2.04–2.29 / 2.35–2.58 | 2.83 / 2.65 | 3.42 / 2.56 | 4.12 / 2.79 |
| hidden item, cross-entropy, posterior / prior | 0.29–0.43 / 0.42–0.86 | 0.96 / 0.95 | 0.49 / 0.42 | 0.67 / 0.61 |
| hidden item, top-1, posterior / prior | 0.84–0.90 / 0.62–0.89 | 0.79 / 0.68 | 0.86 / 0.86 | 0.78 / 0.73 |

(n: 1,115–1,346 rows with hidden species per cell; 6,500–7,800 revealed-mon slots; 1,700–2,600 hidden items.)
**On familiar opponents the deltas roughly halve the species error; on unfamiliar ones they are no better than the
prior at ranking and far worse calibrated, and the moves delta HURTS.** The item delta still helps top-1 off-pool
(items are species-typical) while losing a little calibration. This is H3 of the diversity design (belief-head
memorisation) read on calibration; its game-level kill test, D0f, is still unrun.

### P6 — the critic (`analyze_rows.py`)

| panel | side 0 (pilots the panel): mean P(win) / outcome (row-weighted) | side 1 (pilots home72) | Brier, side 0 / side 1 | first decision P(win), side 0 / side 1 |
|---|---|---|---|---|
| home72 | 0.521 / 0.417 | 0.509 / 0.555 | 0.166 / 0.146 | 0.575 / 0.585 |
| F | 0.447 / 0.296 | 0.569 / 0.680 | 0.157 / 0.160 | 0.552 / 0.596 |
| H | 0.422 / 0.295 | 0.601 / 0.679 | 0.156 / 0.137 | 0.575 / 0.586 |
| P72 | 0.394 / 0.294 | 0.618 / 0.679 | 0.148 / 0.148 | 0.547 / 0.585 |

The two sides' values sum to ≈ 1.0 per row (1.01–1.03), both are compressed toward 0.5 against the outcomes, and the
first decision reads ≈ 0.55–0.60 for BOTH sides (the sum ≈ 1.15, an optimistic start, on home72 too). No
familiarity-specific miscalibration is visible at 48 games per panel (an outcome's se ≈ 0.07). DESCRIPTIVE.

### P7 — how much the decision leans on each S input (`analyze_rows.py`)

Each input replaced by a constant mean over real mons (in S only, our and their tokens and the hypothesis rows),
640 rows per panel; mean KL from P\*'s own policy and the argmax flip rate:

| ablated | KL | flip | \|Δv\| |
|---|---:|---:|---:|
| species id embedding | 0.29–0.38 | 0.30–0.36 | 0.053–0.066 |
| move id embedding | 0.31–0.46 | 0.33–0.39 | 0.044–0.054 |
| move mechanics (columns + `MOVE_ATTR`) | 0.14–0.22 | 0.23–0.31 | 0.022–0.033 |
| item id embedding | 0.03–0.06 | 0.08–0.13 | 0.009–0.017 |
| the 13 stat columns | 0.02–0.03 | 0.09–0.11 | 0.010–0.013 |

The same on every panel and both sides (no familiarity effect). A mean ablation is itself off-distribution, so
these are reliance ranks, not effect sizes.

### P8 — one bench mon's set, familiar against unfamiliar (`perturb.py`)

600 home72 rows (48 games); one never-yet-seen, full-HP bench mon of ours takes another set of its species: a POOL
set or a NON-POOL panel set, matched on the difference from the current one (587 of 600: one move; the rest one
move + item, or two + item); 158 distinct swaps of each kind. Only that slot's set columns change.

| swap | mean KL | median KL | flip | \|Δv\| | signed Δv |
|---|---:|---:|---:|---:|---:|
| to a familiar set | 0.060 | 0.014 | 0.143 | 0.0185 | −0.005 |
| to an unfamiliar set | 0.072 | 0.017 | 0.127 | 0.0189 | −0.006 |
| unfamiliar − familiar (CI: 2,000 resamples of games) | **+0.012 [+0.001, +0.023]** | | | **+0.0004 [−0.002, +0.003]** | |

By the Smogon usage of the swapped-in move the familiar / unfamiliar KLs are 0.055 / 0.073 (usage ≥ 0.3), 0.063 /
0.075 (0.05–0.3) and 0.069 / 0.046 (< 0.05, 107 / 45 rows). An unfamiliar set moves the policy about a fifth more,
with no value effect and no consistent dose by rarity. A one-move change on an unrevealed bench mon flips the
greedy action in one decision in seven, whether or not the set is familiar.

### P9 — bench permutation (`perturb.py`)

The bench cyclically permuted in every slot-indexed block (team slots, the pair history's our axis, the OBS-FACTS
`seen` rows, the switch mask): max |Δp| 1e-5, |Δv| 0, flips 0 of 600. **Exactly equivariant.**

### P10 — shrinking the belief toward its prior (`mix_probe.py`)

`p_mix = (1 − λ)·posterior + λ·prior` per leg (the species presences mixed the same way, so they still sum to k),
pooled over the panels (familiar: 6,257 species rows, 34,988 revealed-mon slots, 10,703 items; unfamiliar: 3,580 /
21,735 / 6,819):

| λ | 0 (P\*) | 0.05 | 0.1 | 0.25 | 0.5 | 1 (prior) |
|---|---:|---:|---:|---:|---:|---:|
| species log-presence, familiar | −0.71 | −0.73 | −0.75 | −0.83 | −1.01 | −1.87 |
| species log-presence, unfamiliar | **−7.06** | −3.70 | −3.40 | −2.99 | −2.69 | −2.55 |
| species hit@k, familiar | 0.737 | 0.738 | 0.738 | 0.736 | 0.736 | 0.419 |
| species hit@k, unfamiliar | 0.255 | 0.255 | 0.257 | 0.261 | 0.271 | 0.278 |
| moves BCE, familiar | 2.19 | 2.08 | 2.06 | **2.03** | 2.09 | 2.47 |
| moves BCE, unfamiliar | 3.45 | 3.13 | 3.02 | 2.82 | 2.67 | 2.66 |
| moves precision@u, familiar / unfamiliar | 0.667 / 0.483 | 0.670 / 0.486 | 0.670 / 0.491 | 0.672 / 0.504 | 0.668 / 0.529 | 0.577 / 0.526 |
| item cross-entropy, familiar / unfamiliar | 0.347 / 0.728 | 0.343 / 0.611 | 0.343 / 0.590 | 0.354 / 0.566 | 0.393 / 0.569 | 0.590 / 0.682 |

**A fixed shrink of about a quarter toward the prior keeps every familiar RANKING (species hit@k 0.737 → 0.736,
moves precision 0.667 → 0.672, item top-1 0.870 → 0.876) and removes most of the unfamiliar calibration damage
(species −7.06 → −2.99, the prior −2.55). The moves and items posteriors are overconfident even on familiar
opponents (their BCE / CE improve there too).** Off-pool species RANKING cannot be bought this way: it is capped by
the prior's 0.28. Offline only: the policy was trained on the unshrunk belief.

## 3. Findings

- **F-AG-1 (MEASURED, P5).** The opponent-belief deltas memorise the training pool. On unfamiliar opponents the
  species posterior is confidently wrong (log-presence of the truth 2–3× worse than the prior's) and the moves delta
  is worse than its prior; on familiar opponents both help a lot. X5's hypothesis tokens therefore hold the wrong
  species in about 63–85 % of an unfamiliar team's hidden slots (24–29 % on a familiar one).
- **F-AG-1b (MEASURED, P10).** The posterior is overconfident on familiar opponents too (moves and items), and a
  fixed λ ≈ 0.25 mixture with the prior is close to Pareto-free offline: familiar rankings unchanged, unfamiliar
  species log-presence −7.06 → −2.99.
- **F-AG-2 (MEASURED, P3 / P8 / P9).** Our own team is handled compositionally and equivariantly. An unfamiliar set
  is not out of distribution for the encoder or the policy beyond ≈ +19 % KL on a local swap.
- **F-AG-3 (MEASURED, P2 / P7).** The S token leans on identity embeddings (species, move, type) and barely reads the
  explicit stats and move mechanics. For entities the pool covers (≈ 98 % of the panels' species) this is harmless
  by P3 / P8; it is a risk only for entities the pool lacks.
- **F-AG-4 (DESCRIPTIVE, P6).** No familiarity-specific critic miscalibration at this sample size; the critic is
  compressed toward 0.5 and optimistic at the first decision for both sides.
- **F-AG-5 (STRUCTURAL).** The place a team-level "plan per team" is cheapest to store is the flat policy tower
  (43 % of the learner, an MLP over pooled summaries). Not measurable by these probes; the built, never-trained
  `--policy-readout trunk` retires it.

## 4. The levers, ranked

| # | lever | mechanism | cost | falsifiable test (cheapest first) | risk |
|---|---|---|---|---|---|
| **1** | **Anchor the belief deltas to the Smogon prior** | the deltas cannot hold a pool team's identity: (a) a fixed mixture with the prior, λ ≈ 0.25 (P10: familiar rankings kept, unfamiliar calibration mostly repaired), (b) per-episode delta DROPOUT (zero δ with probability d during rollouts and learning, so the policy also acts on prior-only beliefs), (c) a bounded delta (`c·tanh(δ/c)`, a few nats) or a KL-to-prior penalty | (a) inference-time, a forward flag: ≈ 0.5 agent-day; (b) / (c) a model flag + a fork arm ≈ 1 agent-day + ≈ 13 GPU-h | **first D0f** (the diversity design's H3 kill test, prior-only versus full, on home72 and U): needs only a delta-zeroing forward flag, CPU games. Then (b) as a fork of P\* read by D1's rule on U. Offline guard first: P5 on the arm must show unfamiliar species log-presence ≥ the prior's and familiar hit@k within 0.05 of P\*'s | in-pool strength: the deltas are what identifies a pool team (W12: worth 0.80 of the oracle gap in-pool); on the ladder that identification is worthless, so the cost is a self-play-only cost; a too-strong anchor discards real in-battle evidence |
| **2** | **Train the belief deltas on the ladder population, not the pool** | the labels, not the network, are the memorising part: supervised belief targets from human ladder teams (the `hr_spec_v1` logs' revealed teams and the 22,862 `hl_05_26` reconstructions), offline, no RL | ≈ 1–2 agent-days (a supervised belief-only fine-tune on frozen trunk features) + CPU / a short GPU job | P5 re-read on the fine-tuned heads: unfamiliar species hit@k above the prior's (0.30 / 0.39 / 0.16) and log-presence above it, familiar hit@k ≥ 0.60; then the D0f-style game read | the panels' own neighbours in the ladder data contaminate the read (the human-data card: ≈ 71 % of ≥ 1500 battles share a held-out 3-set core) — needs the manifest-driven exclusion the human-data plan already has |
| 3 | **Retire the flat policy tower** (`--policy-readout trunk`, built, never trained) | removes the 1.12M-parameter MLP over pooled team summaries, the cheapest store of per-team plans; each action is scored from its own refined token with one learned state query | a fresh-run pair; on the toy first if its readout can take the same switch (≈ 0.5 agent-day to check), else 2 × 30M production (≈ 11 GPU-h) | D5's leave-out instrument: the familiarity premium (held-in − held-out piloting) of trunk vs tower at 30M; falsified if the premium is not smaller by ≥ 3 pp | strength at matched compute unknown (never trained); confounds capacity with readout form |
| 4 | **Identity dropout in S** | per mon, with probability p, replace the species / move / item id embeddings by a learned UNKNOWN row, so S must also read the types, stats and `MOVE_ATTR` (entity masking, the domain-generalisation recipe) | a model flag + a deterministic per-episode mask the rollout and the learner share (the mask must be reproducible from the row, or PPO's ratio is wrong): ≈ 1 agent-day + a fork arm | toy first (T-ENT reads the same S): P2's feature share must rise and P8's unfamiliar − familiar KL fall; then the H8 subset (the 9 teams with an unseen entity) | P3 / P8 say our seen-entity sets already transfer, so the expected gain is small (only unseen ENTITIES benefit); in-pool sharpness |
| 5 | **A "this opponent is off-pool" fact** | the Smogon log-likelihood of the opponent's reveals so far (moves, items, species given the revealed teammates), so the network can learn when to stop trusting its pool-shaped deltas | a hand-computed fact (a row in `design_hand_computed_features.md`) ≈ 1 agent-day + an arm | P5 by the fact's quantile: the posterior's harm must concentrate where the fact flags off-pool | a judgment-flavoured feature; redundant if lever 1 or 2 works |
| 6 | **Generic regularisers** (weight decay on the embeddings and deltas above 1e-5; dropout) | shrink memorised directions | trivial flags; PPO dropout needs the same mask at rollout and learning | a fork arm on P5 / P8 | weakest evidence; PPO-incompatible dropout if done naively |

**Interaction with D3 (set mutation).** D3 redraws 1–2 mons' SETS of a pool team from Smogon and keeps its species.
So it attacks the MOVES / ITEM deltas' memorisation (their labels become Smogon draws, pulling the deltas toward the
prior) and leaves the SPECIES delta untouched: the cores stay pool cores, so "revealed X + Y ⇒ this team's other
four" still pays. **Prediction (registered here, read by re-running `analyze_rows.py` on the D3 checkpoint, ≈ 3 CPU
min per panel):** D3 moves unfamiliar moves-precision toward or above the prior's (0.56 / 0.51 / 0.51) and leaves
unfamiliar species hit@k and log-presence where P\* has them. Lever 1 composes with D3 (D3 + species anchoring
covers both channels); lever 4 is D3's representation-side twin and is unneeded if D3 helps (P8 says our own sets
are not the bottleneck). D1 adds human-built CORES, which retrains the species delta on a second population: that
fits D1's gain sitting mostly on F, its own population.

**Interaction with the human-data levers.** Lever 2 IS a human-data lever, the one P5 says is aimed at a measured
failure: ladder labels for the belief heads. The human opponent model (R-b1) predicts human CHOICES and is a search
judge; it does not touch the policy's own belief. X59-S (human labels for the policy) targets piloting, the channel
these probes could not localise; if it pays, lever 3 is its architectural companion (a policy that cannot store
per-team plans has to learn transferable ones from the larger team population).

## 5. Hazards and limits (FINDINGs)

- **F-AG-H1.** One checkpoint, one seed, greedy self-play only (P\* on both sides). The belief and critic reads are
  against P\*'s own piloting of the opponent's team, not Kakuna's or a human's.
- **F-AG-H2.** 48 games per panel (the panels' first 48 teams); every calibration figure is DESCRIPTIVE.
- **F-AG-H3.** P8's bootstrap resamples games but the 158 distinct swaps are reused across rows; its +0.012 KL is a
  small effect near its interval's edge (lower bound +0.001) and should be read as "at most a fifth more", not a
  detected mechanism.
- **F-AG-H4.** The mean ablations (P7) and the mixture (P10) are off-distribution interventions; the policy was not
  re-run on mixed beliefs, so P10 says what the belief would be, not what the policy would do.
- **F-AG-H5.** The belief metrics exclude every Hidden Power move (typed HP's label space differs from the raw
  prior's) and any true move the learnset gate calls illegal (none observed in the counts, not separately reported).
- **F-AG-H6.** The rows came from the worktree's own release build of the Rust core (the main checkout's build was
  stale against HEAD and refused by its stamp check).
- **F-AG-H7 (code, out of scope here).** `src/utils/rust_env/fixture_battles.py`'s `play_rows` stages `ep_team` /
  `ep_seed` only twice, so every auto-reset after an env's second episode replays that env's second staged teams and
  seed (the column contract says the caller keeps it staged one ahead). Harmless with its seeded random policy;
  with a deterministic one it repeats games.
- **F-AG-H8 (code docstring, out of scope here).** `src/agents/model/static_tokens.py`'s module docstring still
  calls `legacy` "the default, production" and names "the global token"; production is `static` with three board
  tokens.

## 6. Reproduce

From a worktree with `PYTHONPATH=$PWD/src:<the scripts dir>` and the env's interpreter, rule 5b's shape (`nice -n
19 taskset -c 5-7,13-15`, under `scripts/ops/mem_cap.sh --max-s 600`), CPU only:
`static_probe.py` (P1–P4); `collect.py <panel> 48 <seed>` (seeds 2611–2614 for home72, F, H, P72); `analyze_rows.py
<panel> 640` (P5–P7); `perturb.py 600` (P8, P9); `mix_probe.py` (P10). `run_collect.sh` and `run_perturb.sh` chain
them. Every input is fixed (teams, battle seeds, greedy policies, one core thread); see the byte-identity caveat in
section 2.
