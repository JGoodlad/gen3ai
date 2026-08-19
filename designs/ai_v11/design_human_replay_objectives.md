# design — LEARNING FROM HUMAN LADDER REPLAYS: an objective ladder ordered by OOD-robustness

> **[STATE 2026-08-18]** Opened as the first ai_v11 document. **Nothing here is built.** The §5
> census numbers are NEW and were measured this session against the live corpus and the live
> `agents.bc.log_reader`; every one carries its n and its seed. §3's ladder is **pre-registered and
> unrun**. §8 records what would kill the document.
>
> Candidacy: **gen-17**. Rungs 1 and 2 are OFFLINE and need no generation slot (they can run beside
> gen-16); only rungs 3 and 4 spend a launch.

---

## 0. Goal, and the constraint that shapes every line of it

Self-play explores its own convex hull. Human ladder replays are the only **external** action
distribution we own. The question this document answers is not *"should we use them"* — it is
**which objective can survive the way they are broken.**

The owner's constraint, which is this document's spine and not a caveat:

> "Those replays are PARTIAL information, which means they are OOD for what our model actually
> does."

Precisely. Our model's observation **always** contains its own full team: six mons, four moves
each, item, ability, and an exact IV/EV/nature spread from the team sheet, plus a
server-authoritative `|request|` that states the legal action set and the PP behind it. A
**spectator** replay reveals a player's own private information only as the game happens to expose
it. A mon that never switched in does not exist. An item that never triggered is unknown. A spread
is never stated in any replay, ever. And the request stream — the thing that defines what was
*legal* — is not in a spectator log at all.

So a reconstructed acting-side observation is not a noisy version of a live one. It is
**structurally unlike anything the deployed policy ever sees**, and it is broken *differently on
each side*: the OPPONENT half of the obs is exactly as partial as it is in live play (that side is
supposed to be hidden), while OUR half is partial in a way that is impossible at deployment.

That asymmetry is the whole design. It orders the ladder: **an objective whose label and inputs
live on the opponent half is nearly free; an objective that needs the acting side's full obs and a
correct mask is the most expensive thing in this document.** Faithfulness is therefore **measured
per decision and carried as a weight**, never assumed.

---

## 1. Why — the evidence that human data is worth this trouble

| # | Evidence | Provenance |
|---|---|---|
| **1.1** | **The self-play treadmill.** `win_rate_vs_pool` is pinned near 50% by the promotion gate and `win_rate_vs_bots` saturates; the plateau research verdict is a *converged flat fixed point of this recipe*, not a capacity limit. | `designs/research_state/README.md` frontier; memory `project_plateau_research_2026_06_25` |
| **1.2** | **The Metamon precedent.** Offline RL on human replays + team diversity broke the same plateau in the same game family. | our record of it in `project_plateau_research_2026_06_25`. ⚠️ **UNVERIFIED** as to *how* they handled acting-side partial information — see §8.3; reading that paper is Phase-0 task **P0.6** |
| **1.3** | **AlphaStar's human bootstrap** — supervised imitation from human games as the initialisation the league then improves on, in a hidden-information game. Cited as prior art for the *ordering* (imitate, then compete), not as a recipe. | **UNVERIFIED** in detail here — no source read this session |
| **1.4** | **The concrete behavioural gap.** Against strong ladder humans on faithfully-reconstructed states, our argmax switches on **16%** of voluntary decisions where humans switch on **30%**. | `README.md` frontier row *Under-switching*; memory `project_human_agreement_probe` |
| **1.5** | **The human half of 1.4 reproduces model-free, at scale.** Over **30,146** reconstructed voluntary decisions from **1,142** sides at rating ≥1500, the human switch share is **28.96%** (all-ratings: 27.83% over 57,318 decisions). No model is involved, so this is a property of the corpus, not of a probe. | §5; `measurements/human_replay_faithfulness_census_{1500,all}.json`, seeds 2/1, 2026-08-18 |

**1.4 is REFRAMED and the reframe matters for rung 3.** The gap was shown to be *commitment*, not
valuation: the policy's soft switch mass (0.28) already ≈ the human rate (0.30) — it is the ARGMAX
that under-commits. So a BC-style KL toward the human action is aimed at a target the policy's
*distribution* already hits. Rung 3 must therefore be gated on something other than "did the switch
rate move" alone; see G3.

---

## 2. THE OOD TAXONOMY — the spine

Every axis on which a replay-derived training example differs from a live one. Each row states how
it is **measured per decision** (the pipeline emits these as a faithfulness record, §5.1), and which
rungs it poisons.

### 2.1 The harmful axes — our own side

| id | axis | what actually happens | measured by | poisons |
|---|---|---|---|---|
| **O1** | **Own moveset incompleteness** | The turn-1 injection (`log_reader._full_team_request`) fills each own mon's moveset with the **union of what it used across the whole game** — typically 1–3 of 4. A move never clicked never exists. | `f_moves` = revealed/4 per mon; **measured: 8.64% of own mons reach 4/4; 9.38% reach 0** (§5.3) | 1 (α's inputs), 3, 4 |
| **O2** | **Own item / ability missingness** | Passed only if the protocol revealed them. Gen-3 announces neither by default. | `f_item`, `f_ability` bits; **measured: item known on 3.93% of own mons, ability on 7.86%** | 1, 2, 3, 4 |
| **O3** | 🚨 **Own SPREAD is FABRICATED and flagged KNOWN** | There is no teambuilder team, so the fork's `backfill_teambuilder_spread` never runs and `pokemon.py` takes its own-side fallback: **IVs all-31, EVs all-zero, nature `serious`, and `spread_known = 1.0`**. This is not a gap — it is a *wrong value asserted with the confidence bit set*, and it is the input to our OUTGOING physics (`d1` 12.17% / `d2` 19.25% dependence, the two strongest measured edge families). | structural — `f_spread ≡ 0` on every replay decision, by construction | **1, 2, 3, 4 — the worst axis in the table** |
| **O4** | **Own bench truncation** | A mon that never switched in is absent from the injected team, so the obs shows a 3-mon team where the player had 6. | `f_bench` = mons revealed/6; **measured: 67.2% of sides reach 6/6 (≥1500); 13.6% at ≤4** | 2, 3, 4 |
| **O5** | 🚨 **The REQUEST STREAM DOES NOT EXIST** | `_synth_legal` builds `LegalActions` from known state with `current_pp=1, max_pp=1, disabled=False, trapped=False, maybe_trapped=False, force_switch=False`. So the mask asserts **every revealed move is choosable and the mon is never trapped** — false whenever a trapper is on the field, a Choice item is held, or Disable/Taunt/Encore/Torment/0-PP is active. The mask is systematically **over-permissive**, and an over-permissive mask puts probability mass on actions that were illegal. | game-level hazard scan today; **measured: a trapper species is on the field in 54.1% of ≥1500 games, `\|cant\|` fires in 65.9%, Taunt in 12.9%** — a per-DECISION version is Phase-0 task **P0.4** | **3 (fatal without filtering)**, 4 |
| **O6** | **Decision filtering is itself a distribution shift** | Only the first *voluntary* action per turn survives: forced post-faint replacements and `\|cant\|` turns are dropped. So the reconstructed state distribution is missing exactly the phases where the live agent has a degenerate mask. | drop counts per side (already in `ReadStats`) | 2 (value on a filtered state set), 3 |
| **O7** | **Faithfulness is a function of the FUTURE** | The injection uses end-of-game knowledge. That is *faithful in content* — the player really did know their own team at turn 1 — but it means whether a decision is faithful depends on what happened after it. | see §2.3, the selection-bias measurement | 2, 3 |
| **O8** | **Protocol coverage holes** | **3.15% of ≥1500 sides (36/1,142) raise `UnknownVolatileError` and produce nothing** (4.03% at all ratings). These are not random: they are the sides containing mechanics our battle layer does not classify, i.e. biased toward the unusual. | exception counter; §5.2 | all rungs (silently) |
| **O9** | **Own typed Hidden Power / minor identity** | `LegalActions.own_hp_typed_id` is absent, so our own HP resolves bare rather than typed — an own-side fact the live agent always has. | present/absent bit | 1, 3 (small) |

### 2.2 The axis that is NOT shifted — and it is why rung 1 exists

| id | axis | status |
|---|---|---|
| **O10** | **Opponent-side partiality** | **NOT a shift.** In live play the opponent's team, moves, items and spreads are hidden and inferred from the protocol stream. In a replay they are hidden and inferred from *the same protocol stream*. Our species/move/item/HP-type belief heads, the `EpisodeTracker`, the recency/pair-history/event-window folds — all of them run over a replay exactly as they run live. |

**This is the single most important structural fact in the document.** It means an objective whose
*label* is the opponent's action and whose *inputs* are opponent-side is measuring a quantity in its
native distribution. That is rung 1.

### 2.3 The DESIRED shift — and how to keep it separable from the harmful ones

| id | axis | why it is the point |
|---|---|---|
| **O11** | **Human-vs-human trajectory distribution** | Different openings, different pace, different risk attitude, different teams (the ladder meta, not our 719-team pool). **This is the entire reason to do any of this.** |
| **O12** | **Rating/meta drift** | The corpus spans 2026-05-18 → 2026-08-02; the meta inside it is one slice. Benign, but it means a fit is to a *dated* meta. |

**The methodological rule this taxonomy exists to enforce:** a gain measured on human data can come
from O11 (real) or from the model learning to exploit O1–O9 (an artifact of our reconstruction). The
two are separated by **conditioning every headline on the faithfulness tier**: a real O11 gain is
present in the faithful stratum and roughly *stronger* there; an artifact gain concentrates in the
thin stratum and *vanishes* on tier A. Every gate in §3 carries that stratified re-read as a
mandatory second look.

### 2.4 The selection bias — MEASURED, and it has a sign

Faithfulness is not missing-at-random. A mon appears only if it was brought in, and the losing side
brings in everything.

| | mean own mons revealed | tier-A share of decisions |
|---|---|---|
| sides that **WON** | **5.23** | 14.80% |
| sides that **LOST** | **5.62** | 19.07% |

*(≥1500, n = 30,146 decisions / 1,142 sides, seed 2. All-ratings, n = 57,318: 4.84 vs 5.29 and
15.13% vs 17.29%.)*

So the **fully-faithful subset is loss-enriched by a factor of 1.29** on tier-A share. Any objective
that (a) weights by faithfulness AND (b) has an outcome-dependent target — rung 2 above all — is
biased **pessimistic** unless the weights are balanced within outcome. The fix is cheap and must be
pre-registered: **stratify by (faithfulness tier × outcome) and reweight to the unconditional
outcome marginal.** Naming it here so it cannot be discovered after a result.

---

## 3. The objective LADDER, ordered by OOD-robustness

Each rung: mechanism · which axes touch it · mitigation · **pre-registered gate** · **kill
condition**. Rungs are strictly ordered — a rung does not start until the one below it has passed or
been killed, because each rung's failure mode is diagnostic for the next.

### Rung 1 — α/β supervised on the HUMAN OPPONENT's actions *(most robust)*

**Mechanism.** Reconstruct side A's observation at turn *t*; take side B's action at the same turn as
the α/β label. α is the distribution over the opponent's K believed threat-move seats plus SWITCH
(`agents/model/opp_intent.py`, v68); β is the pointer over their team slots given a switch. Both
already exist, are supervised, read a **detached** input, and match labels **by canonical move id**
(`match_seats_to_move_num`) with a belief miss MASKED and logged as `alpha_mask_rate`. Nothing about
the head changes; only where the label comes from.

**Label availability is essentially free and MEASURED: 92.04%** of the union of both sides'
reconstructed decision turns are *paired* (7,866 / 8,546 over 302 two-sided ≥1500 games). Both sides
of a replay are visible, so every reconstructed decision on side A is a labelled α example, and vice
versa — one parse yields two datasets.

**OOD axes that touch it.**
- **O10 does not** — the opponent half is in its native distribution, which is the whole point.
- **O1/O3 DO contaminate α's INPUTS.** The 2026-08-11 owner reconciliation requires α's input to
  include *our* outgoing physics (`d1`/`d2` grids) — both sides anticipate. Those grids are computed
  from our own movesets (O1) and our own spreads (O3, fabricated as 31/0/serious). So α on a replay
  is conditioned on a board where our offense is systematically **under-priced** (0 EVs, neutral
  nature) and **under-populated** (median 2 of 4 moves known).
- **O6** filters the state set; **O8** silently drops 3.15% of sides.

**Mitigation.**
1. **Faithfulness weighting**, not filtering — α labels are dense and the thin stratum still carries
   opponent-side signal. Weight `w = f_bench · f_moves(acting)` and report the gate BOTH weighted
   and restricted-to-tier-A.
2. **The delivery seam already exists and generalises.** `--intent-label-bot-weight`
   (`gen3_intent_label_bot_weight_v1`) is a **per-opponent-class weight on the α/β labels**, folded
   as `Σ_i w_i·ce_i / n_sup`, keyed off the existing `opp_class` obs key, and confined to INTENT by
   an explicit design claim (team-truth beliefs never see `opp_class`). A **HUMAN class beside
   BOT/POOL/STABLE/EXPLOITER** is the natural extension: same weighting seam, same stratified
   metrics, same `_pool`-suffix discipline. ⚠️ **But the plumbing does NOT transfer for free** —
   `opp_class` is tagged per *episode* by `MaskableAgentWrapper._select_episode_opponent` and rides
   the rollout buffer; offline human rows never enter that buffer. Rung 1 needs its own **offline
   batch path** into `instrumented_ppo`'s intent loss (or a separate pre-training pass), and the
   label-shift machinery (`align_labels_to_predictions`, which shifts α labels back one row and
   drops episode-boundary pairs) has an **offline analogue that must be written, not reused** —
   getting that wrong splices one battle's first decision onto another's last board, invisibly.
3. **Read `_pool`-suffixed metrics, never the bare key** — the bare α metrics are a moving mix of
   opponent classes. A human class makes the mix move again.

**Gate G1 (pre-register before running).** Hold out human games by *game* (never by turn). Train α/β
on human data; evaluate on held-out human decisions:
- **primary:** `alpha_move_recall_top1` and `beta_recall_top1` on held-out humans, versus **two**
  baselines — (a) the Smogon usage prior `alpha_move_baseline_argmax_w` (the like-for-like baseline
  the head already reports), and (b) the *current self-play-trained* α head evaluated on the same
  held-out human decisions. Passing means beating **both**.
- **non-regression:** `alpha_acc_pool` on self-play data must not fall (this is the imprinting risk
  the bot-weight knob was built for, pointed the other way).
- **stratified re-read (mandatory):** the gain must be present on **tier-A** decisions. Fix the
  threshold before looking: a ≥3pp top-1 gain over baseline (b), with the tier-A estimate inside the
  weighted estimate's CI.

**Kill K1.** Human-trained α does not beat the Smogon prior on held-out humans → the corpus carries
no usable behavioural signal at this rating floor and the ladder stops at rung 1. **Or:** the gain
exists weighted but **vanishes on tier A** → it was fitting the reconstruction artifact, and every
rung above inherits that verdict.

**Note on scope.** Since the v96 critic-route deletion wave removed every α→vf route, α reaches the
objective only through the policy — so a rung-1 win is a *policy-side* win by construction, and any
proposal to route a human-fit α to the critic owes the **C4 offline gate first** (ledger **C6**;
`value_intent`'s re-entry condition survived its own deletion).

---

### Rung 2 — Outcome / value supervision on human states

**Mechanism.** Every game has a terminal outcome (**`files_no_win` = 0 in both censuses** — 100%
label availability), so every decision in it carries a Monte-Carlo win label. This is exactly the
pattern `WinProbLabelCallback` already implements online (`--win-prob-mode`): capture the terminal
outcome, fill it back over that episode's decisions, masked-BCE against the head's logits. Offline,
the fill is trivial — the whole game is in hand before the first row is emitted.

**What may be claimed.** A **LEVEL** claim only. The head fits `P(win | board, human-vs-human play at
this rating)`. That is not `V^π` for our π — value is policy-dependent, and a human-fit readout
evaluated under our continuation is off-policy in the one way that cannot be reweighted away. So
this ships as a **side readout in `read_only` mode** (the head trains on a STOP-GRADded
`value_pooled`; the shared-trunk pull `grad/win_prob_share` is ≈0 by construction), i.e. an
instrument, not a critic. Promotion to `shaping` is a separate, later decision with its own gate.

**OOD axes.** O3 (the critic reads own-side physics computed from fabricated spreads) · O4 (a
3-mon own team reads as a losing position that the player was not actually in) · **O7 + §2.4 — the
measured 1.29× loss-enrichment of the faithful stratum is a direct bias on an outcome-labelled
objective.**

**Mitigation.** Outcome-balanced faithfulness weights (§2.4) — pre-registered, not discovered.
Report the fit **with and without** balancing; a large gap between them is itself the finding.

**Gate G2.** Held-out human **AUC and Brier** for the human-fit readout vs the frozen self-play
win-prob head scored on the same held-out human decisions, plus a calibration curve. Pass = AUC
strictly better and calibration no worse, **and** the advantage survives outcome-balancing.

**Kill K2.** No AUC gain, or the gain disappears under outcome balancing (⇒ it was reading the
selection bias, not the board). Also killed if the human-fit head's advantage is confined to the
thin faithfulness stratum.

---

### Rung 3 — BC-regularization *(the most OOD-sensitive rung)*

**Mechanism.** An auxiliary term pulling the policy toward the human action on human states —
`CE(π(·|s), a_human)` or `KL(π ‖ one-hot)` — either as a pre-training phase or as a coefficient
alongside PPO on interleaved offline batches.

⚠️ **Do not implement from `designs/ai_v6/impl_step2_bc.md`.** That doc's Phase-2 loss is written
against `policy.action_net(latent_pi)` and freezes `policy.mlp_extractor.latent_vf_net` — the flat
`action_net` was **deleted at v51** (`gen3_pointer_native_v1`; every action is scored from the token
of the entity it selects) and the extractor now returns a `(pi_features, vf_features)` tuple into
`Gen3DualHeadMaskablePolicy`. Its mask-synthesis section (Options A/B) also predates `LegalActions`
and `_synth_legal`. The doc is a **historical record of intent**, not a build spec.

**Why it is the expensive one.** It is the first rung that needs the **acting side's** full
observation *and* a correct **action mask**. Every axis in §2.1 lands on it at once, and O5 lands
hardest: a mask that asserts "never trapped, every revealed move choosable" teaches the policy on a
legality set that is not the game's. Worse, the errors are *correlated with the interesting states* —
trapping and Choice locks are exactly where Gen-3 decisions are hard.

**Mitigation — a restricted, mask-audited subset.** Admit a decision only if:
1. faithfulness tier **A** (all six own mons revealed AND the acting mon at 4/4 moves) — **16.70% of
   ≥1500 decisions (5,035 / 30,146)**; and
2. no mask hazard active at that decision: no trapper on the field, no revealed Choice item on our
   active, no Disable/Taunt/Encore/Torment volatile, and no move at or past its PP budget by an
   observed-use count. **The decision-level rate of (2) is NOT yet measured** — game-level rates are
   54.1% / 12.9% / … (§5.6), and turning those into a per-decision admission rate is Phase-0 task
   **P0.4**. It is the number that decides whether this rung is feasible.

**Sizing (estimates, arithmetic in §5.7).** ≥1500 gives ≈1.55M reconstructed decisions ⇒ ≈259k
tier-A. ≥1600 gives ≈475k ⇒ ≈79k tier-A. The mask-hazard filter then cuts an unmeasured further
fraction. **Pre-registered feasibility floor: if the admitted set at ≥1500 is below 50k decisions,
rung 3 does not run** — a set that small cannot move a policy of this size without overfitting a
reconstruction artifact, and rung 4 (which *repairs* the inputs rather than filtering on them)
becomes the only route.

**Gate G3.** Three readouts, all pre-registered:
- **behavioural:** the argmax switch rate moves toward the measured human **28.96%** (§1.5) *on
  matched live states*, not on replay states;
- **skill:** anchored ELO non-regression against the same-generation control at matched snapshot
  COUNT (the standing ELO reading rules apply — `ladder.json`, run END, matched count);
- **the discriminating one:** because 1.4 was reframed to *commitment*, a switch-rate move with no
  ELO move is the **expected null**, not a partial win. Pass requires ELO ≥ control with the switch
  rate moved; switch-rate-only is a KILL, not a "promising".

**Kill K3.** The admitted subset is under the 50k floor · or G3 returns switch-rate-only · or the BC
term degrades `alpha_acc_pool`/ELO (importing human *style* while losing self-play skill).

---

### Rung 4 — Offline RL / Metamon-style *(the full move)*

**Mechanism.** Offline value learning (and/or full offline policy improvement) over the corpus,
with the acting side's unrevealed slots **filled by the team-completion model** and sampled
repeatedly — PIMC-style data augmentation, one sampled completion per pass.

**The claim that justifies the label noise, stated so it can be falsified.** A *sampled* completion
is probably **wrong** but it is **in-distribution**: six mons, four moves each, an item, a spread,
`spread_known = 1`. The un-completed obs is **right-but-impossible**: it is a point the deployed
policy will never visit. Wrong-but-in-distribution costs **variance** (and is averaged down by
sampling several completions per decision); impossible costs **covariate shift**, which no quantity
of data fixes. Therefore completion should beat filtering at equal budget.
⚠️ **This is an argument, not a result.** Its falsification is G4b.

**OOD axes.** Completion *closes* O1/O2/O4 and (via a Smogon spread prior, §4) O3; it does **not**
close O5 (the mask still has no request stream) or O6/O8.

**Gate G4.**
- **G4a — the completion model's own gate** (§4): held-out species top-1/top-5, item top-1, move
  BCE, legality under `gen3_learnset.json`, and — the one that matters — **distributional overlap of
  the DOWNSTREAM quantity**: the damage op's `d1`/`d2` rows computed on completed replay boards must
  overlap the same rows computed on live self-play boards. Completion quality is only interesting
  through the physics it feeds.
- **G4b — completion vs filtering, head to head**, on rung 3's objective at equal admitted budget.
  Completion must beat the tier-A-filtered arm. If it does not, the §"wrong-but-in-distribution"
  claim is refuted and rung 4 reduces to rung 3 with more data.

**Kill K4.** G4a's downstream overlap fails (the completions produce physics we never see) · or G4b
loses to filtering · or the Metamon precedent turns out to rest on something structural we lack
(§8.3).

---

## 4. The team-completion model — the OOD-closer for rungs 3–4

**What exists** (built in the ai_v6 era, `designs/ai_v6/impl_step3_team_completion.md` +
`design_team_completion_detail.md`):

| piece | file |
|---|---|
| model — frozen embedding tables lifted from a PPO checkpoint, per-slot encoder, learned mask token, 2-layer / 4-head / 128-dim transformer (no positional encodings — team order is arbitrary), species + item + multi-hot move heads | `src/agents/model/team_completion_model.py` (~438k trainable params) |
| dataset — masked-slot resampled per `__getitem__`, loss only over masked slots and only where the value was revealed | `src/agents/training/team_completion/{team_dataset,replay_parser}.py` |
| trainer | `src/main/train_team_completion.py` |

**What would need building or re-verifying — four items, and none is cosmetic:**

1. **It has never been trained on this box, or its run was not retained.** `models/` contains no
   `team_prediction/` directory (checked 2026-08-18). Treat it as **BUILT, UNTRAINED**.
2. **The backbone lift is stale.** It loads `features_extractor.{species,move,item}_embedding.weight`
   by name from a PPO zip. That contract has not been re-verified against v96, and 79 of 79 archived
   runs cannot be re-loaded at HEAD. Either re-verify the key names and dims, or **drop the frozen-
   backbone dependency entirely** and learn the embeddings — 438k params on ≈500k team records does
   not need the transfer, and the dependency is a standing arch-drift liability.
3. **It was designed for the OPPONENT's team** (MCTS world sampling). Rungs 3–4 need it for **our
   own** side. Same model, same data, different consumer — but the own side additionally needs the
   **spread**, and *no replay states a spread*. So spread completion is **not learnable from this
   corpus at all**: it must be **sampled from `gen3_spread_priors.json` conditioned on species (and
   ideally moveset)** — which is the Smogon-derived, sanctioned source. That split is a feature:
   moves/items/bench come from a learned joint over replays; the spread comes from a published prior;
   neither comes from our team pool.
4. **The ai_v6 numbers are stale as sizing.** That doc reports 18,869 battles → 37,940 team records,
   avg 5.02 mons revealed / 1.82 moves known. The corpus is now **263,159 logs** (§5.1) and our own
   measurement of own-side revelation is 5.23–5.62 mons (§2.4) — same ballpark, ~14× the data.

**Gating the completer is gating a distribution, not an accuracy.** Species top-1 is a sanity check;
the load-bearing gate is G4a's downstream-physics overlap. A completer that is 40% accurate but
produces the right *distribution* of damage rows is more useful here than one that is 60% accurate
and systematically over-invests EVs.

---

## 5. Phase 0 — the free census (RUN, 2026-08-18)

No training, no GPU, one core, `nice -n 15`. Tool:
**`tmp/replay_faithfulness_census.py`** (read-only; drives the live `agents.bc.log_reader` and
`_prescan_team`; `tmp/` is gitignored — promote the script if the programme proceeds past P0.1).
Its outputs are **committed**, per the research-state rule that a measurement cited by a committed
doc must itself be committed:

| file | what |
|---|---|
| `designs/research_state/measurements/human_replay_faithfulness_census_1500.json` | the headline census — rating ≥1500, seed 2 |
| `…/human_replay_faithfulness_census_all.json` | the all-ratings control, seed 1 |
| `…/human_replay_rating_distribution.json` | the 20,000-file rating scan behind §5.7 |

> **Provenance for everything below:** corpus
> `/home/goodlad/dev/gen3ai/replays/showdown/gen3ou`, **263,159 `.log` files, 2.8 GB, 72 date
> folders spanning 2026-05-18 → 2026-08-02**, measured 2026-08-18 at git `66ccf21`.
> Two census runs: **(A)** all ratings — 1,285 files, 2,508 sides, **57,318 decisions**, seed 1,
> 58 s; **(B)** rating ≥1500 — 5,308 files scanned → 800 contributing, 1,142 sides, **30,146
> decisions**, seed 2, 33 s. Plus a **20,000-file** rating scan (39,242 rated sides).

### 5.0 First finding: the frontier row's corpus size is STALE

`designs/research_state/README.md` said *"~102k ladder replays"*. The actual count is **263,159**
(2.6× more) — the frontier row is corrected in the same pass as this document. Second, related: the
newest date folder is **2026-08-02**, so the `collect_replays.py` daemon appears to have **stopped
~16 days ago** — worth confirming (owner decision **D4**).

### 5.1 The faithfulness record the pipeline must emit

Per decision, alongside `(obs, mask, action)`:
`f_bench` (own mons revealed / 6) · `f_moves` (acting mon's revealed moves / 4) · `f_item`,
`f_ability` (bits) · `f_spread ≡ 0` (structural) · `n_legal_moves` in the synthesised mask ·
`mask_hazard` flags (P0.4) · `tier ∈ {A,B,C,D}` · the side's `won` · the side's rating.
Rolled up: **A** = 6/6 bench and 4/4 moves · **B** = ≥5 bench and ≥3 moves · **C** = ≥4 and ≥2 ·
**D** = otherwise.

### 5.2 Parse rate — and a measurement-honesty bug it exposed

| | ≥1500 | all ratings |
|---|---|---|
| sides attempted | 1,142 | 2,508 |
| **sides that raise** | **36 (3.15%)** — all `UnknownVolatileError` | **101 (4.03%)** (95 `UnknownVolatileError`, 6 `ValueError`) |
| sides yielding zero decisions | 4 (0.35%) | 22 (0.88%) |
| decisions excluded: switch to unrevealed own mon | **0** | 0 |
| decisions excluded: move unmapped | 14 (0.046%) | 280 (0.49%) |

**The turn-1 full-team injection works exactly as advertised: zero switch-to-unrevealed
exclusions in 30,146 decisions.** That is the pre-existing fidelity fix earning its keep.

🚨 **`src/main/human_agreement.py` swallows these.** Its per-side loop is
`except Exception: continue` with no counter, so the 3–4% of sides that fail to parse never appear
in its `fidelity` block — a silently-dropped, non-random 3–4% inside a probe whose entire job is to
report fidelity honestly. Same family as this repo's recorded scar tissue (a timeout counted as a
semantic outcome). Fixing it is Phase-0 task **P0.5**, and it costs three lines.

### 5.3 Own-side incompleteness — the O1/O2/O4 numbers

**Own mons ever revealed, per side** (the bench the injection can rebuild):

| mons | 1 | 2 | 3 | 4 | 5 | **6** |
|---|---|---|---|---|---|---|
| ≥1500 | 1.4% | 1.7% | 3.4% | 8.8% | 17.5% | **67.2%** |
| all | 2.7% | 5.1% | 7.2% | 10.2% | 16.9% | **57.8%** |

**Moves revealed per own mon** (≥1500, n = 5,983 mons):

| moves | 0 | 1 | 2 | 3 | **4** |
|---|---|---|---|---|---|
| share | 9.4% | 30.4% | 31.7% | 19.9% | **8.6%** |

**Own item known on 3.93% of mons. Own ability known on 7.86%.** *(All ratings: 4.28% / 7.31%.)*

Read that item number twice. The acting side's items — Leftovers, Choice Band, the berry that
decides a KO threshold — are **unknown on 96% of own mons**, and the live agent knows all six.

### 5.4 Faithfulness tiers

| tier | ≥1500 | all ratings |
|---|---|---|
| **A** — 6/6 bench, 4/4 moves | **16.70%** (5,035) | 16.23% |
| **B** — ≥5 bench, ≥3 moves | 29.92% | 29.60% |
| **C** — ≥4 bench, ≥2 moves | 32.82% | 31.38% |
| **D** — thinner | 20.55% | 22.79% |

**Even tier A is not fully faithful** — `f_spread ≡ 0` and `f_item` ≈ 0 on every tier. Tier A means
"as good as this corpus gets", not "equivalent to a live obs". No rung may treat it as the latter.

### 5.5 α-label coverage — the rung-1 number

Over 302 two-sided ≥1500 games: **7,866 of 8,546** union decision-turns are **paired** =
**92.04%**. (All ratings, 1,185 games: 90.72%.) Rung 1's supervision is dense and nearly free.

*(What this does **not** measure: whether the human's move is inside the model's top-K believed
seats. That is `alpha_mask_rate`, needs a model forward, and is Phase-0 task **P0.3**. If the
opponent's actual move is frequently outside the belief's seats, rung 1's usable label set shrinks
by that rate — and per the head's own design that failure belongs to the BELIEF, not to α.)*

### 5.6 Mask-hazard exposure — the O5 numbers (game-level)

Share of ≥1500 games containing at least one occurrence:

| hazard | share of games |
|---|---|
| trapper species on the field (Dugtrio / Magneton / Wobbuffet) | **54.1%** |
| `\|cant\|` (sleep / freeze / recharge / flinch — turns already dropped, but they bracket the state) | 65.9% |
| Taunt started | 12.9% |
| Struggle used | 1.25% |
| Choice Band revealed | 1.0% |
| trapping ability announced | 0.62% |
| Encore started | 0.38% |
| partial-trap move (Wrap family) | 0.25% (all-ratings run) |

Also: the synthesised mask offers **4 legal moves on only 21.3% of decisions**, 3 on 29.0%, 2 on
30.7%, 1 on 17.1%, and **0 on 1.95%** (switch-only). Move-agreement conditioned on this count is the
existing probe's own fidelity control — 4 is the faithful bucket.

### 5.7 Volume vs rating floor — the D3 decision table

Rating distribution from the 20,000-file scan (39,242 rated sides; 1.9% unrated): **mean 1273.6**.

| floor | share of sides | est. sides in corpus | est. decisions | est. **tier-A** decisions |
|---|---|---|---|---|
| ≥1200 | 60.9% | ~320,000 | ~7.6M | ~1.2M |
| ≥1400 | 25.1% | ~132,000 | ~3.4M | ~0.55M |
| **≥1500** | **10.8%** | **~56,800** | **~1.55M** | **~259k** |
| ≥1600 | 3.3% | ~17,400 | ~475k | ~79k |
| ≥1700 | 0.5% | ~2,630 | ~72k | ~12k |
| ≥1800 | 0.02% | ~105 | ~2.9k | ~0.5k |

*(Arithmetic: 263,159 logs × 2 sides × share × 23.8 decisions/side [all-ratings mean] or 27.3
[≥1500 mean, median 25, p90 42], × the 16.7% tier-A share, minus ~3–4% parse loss. These are
**estimates by extrapolation**, not counts — the underlying shares are measured, the products are
not.)*

**The floor is a real trade and the numbers make it sharp:** ≥1700 is the quality most people would
want and it is **~12k tier-A decisions**, which is nothing. ≥1500 is the only floor where the
faithful subset is large enough for rung 3, and it buys players whose median is 1560.

### 5.8 Phase-0 tasks still owed

| id | task | cost |
|---|---|---|
| **P0.1** | Widen the census to ~20k sides for tight CIs on §5.3–5.5 (currently ~1k sides). | ~15 min, 1 core |
| **P0.2** | **Rung-1 offline probe** — the one that decides candidacy. Requires a checkpoint at the CURRENT obs signature, so it runs from the run's own `git_hash` worktree (`log_reader` encodes with HEAD's encoder). | ~1 h CPU |
| **P0.3** | `alpha_mask_rate` on human data — is the human's move inside the belief's top-K seats? | rides P0.2 |
| **P0.4** | **Per-DECISION mask-hazard rate** (§5.6 is per-game). Decides rung 3's feasibility. | ~30 min |
| **P0.5** | Fix `human_agreement.py`'s silent exception swallow (§5.2). | 3 lines |
| **P0.6** | Read the Metamon paper for *how it handled acting-side partial information* (§8.3). | reading |
| **P0.7** | Re-measure the 1.4 human-agreement headline at the current architecture — the 35% / 16-vs-30 figures are from 2026-06-12 on an ai_v6-era model and **cannot be reproduced at HEAD** without a pinned worktree. | ~1 h |

---

## 6. Deployment-alignment decisions the owner must rule on

**D1 — a domain-indicator obs feature ("this observation was reconstructed"): RECOMMEND NO.**
Its appeal is that it lets the network condition on the regime instead of averaging over it. Its
cost is that the bit is **always 0 at deployment**, so the cleanest thing gradient descent can do is
route every human-derived adaptation behind the bit and change nothing about live play — a
degenerate solution that looks like a working feature and is invisible in every training metric.
**Our own precedent points the other way:** `opp_class` identifies the opponent regime and is
deliberately used as a **loss weight, not an obs key** ("No new obs key was added"). Do the same
here — a HUMAN class in the label-weight seam (rung 1's mitigation 2). If a domain signal is ever
wanted in the obs, it must arrive with a pre-registered test that the model has NOT routed around
it (e.g. flipping the bit at eval time must not change live behaviour).

**D2 — do human-derived α labels count as a sanctioned source? YES, by the standing rule.**
Root `CLAUDE.md` → Data Dependencies: *"anything the network READS must trace to Smogon (or
ground-truth labels / ladder replays)"*. A human's clicked action is a ground-truth label from a
ladder replay — doubly sanctioned. Two obligations follow: (a) any **derived artifact** must be
committed with provenance the way `data/gen3_pubval.json` and `gen3_bot_elo_anchors.json` are
(n_games, rating floor, date range, git hash) while the corpus itself stays local and gitignored;
(b) the completion model's **spread** head must draw on `gen3_spread_priors.json`, never on the
719-team pool (§4).

**D3 — rating floor.** §5.7 is the table. Recommendation: **≥1500 for rungs 1–2** (1.55M decisions,
92% α-label coverage), **≥1600 for rung 3** if the mask filter leaves enough (79k tier-A before
filtering), and a **continuous rating weight rather than a hard floor** as the alternative worth
considering — a threshold discards the 25% of sides at ≥1400 entirely, and a weight does not.

**D4 — restart the replay collector?** Newest folder is 2026-08-02 (§5.0).

**D5 — one α head or two?** Training α on mixed self-play + human labels risks the exact imprinting
the bot-weight knob exists to prevent, in the other direction. Options: one head with a HUMAN class
weight (cheap, one head to maintain) vs a separate human-α head used only as an offline evaluator
(clean attribution, no production risk). The G1 non-regression clause is written for the first.

---

## 7. Sequencing

**Gen-16 is the next launch** and is fresh-weights (the v96 `gen3_critic_route_wave_v1` signature
bump), carrying its own pre-registered A/B (`--intent-label-bot-weight` W=1.0 vs W=0.25 on
`alpha_acc_pool`). **Nothing in this document should join it.**

| when | what | slot cost |
|---|---|---|
| **now, beside gen-16** | P0.1–P0.7. All CPU, all read-only. | none |
| **beside gen-16** | **Rung 1 offline (G1)** and **rung 2 offline (G2)** — both are offline fits on frozen checkpoints. | none |
| **gen-17 candidate** | Rung 3, if G1 passed and P0.4 clears the 50k feasibility floor. Needs a launch (a new loss term is training-class; an offline batch path is new plumbing). | one generation |
| **gen-18+** | Rung 4, gated on G4a/G4b, which are gated on the completion model being trained at all. | one generation + a completer training run |

**What gen-16's result would change.** If the W=0.25 arm wins — i.e. bot-derived intent labels were
*hurting* — that is direct evidence that the α head is **sensitive to the behavioural source of its
labels**, which raises rung 1's prior substantially (human labels are a better source than either).
If W=1.0 wins, bot rows carried real signal, the head is source-robust, and rung 1's expected gain
shrinks. **Record the prediction now:** this document expects W=0.25 to be non-inferior.

**Compute.** Census: measured 800 files / 1,142 sides in **33 s** on one core ⇒ full ≥1500
extraction (~56.8k sides) ≈ **45 min single-core**, embarrassingly parallel by date folder.
Rung 1/2 offline fits are minutes of CPU on a head with ~10⁵ params over ~10⁶ rows. Rung 3 is a
normal generation. Rung 4 adds a completer training run (~438k params, the ai_v6 defaults are 200
epochs at batch 64).

🚨 **Storage rule, and it is not an optimisation.** ≥1500 at the current 2501-dim obs is
1.55M × 2501 × 4 B ≈ **15.5 GB** of vectors — and they would be **invalidated by the next
`ARCH_SIGNATURE` bump**, which happens roughly every generation. **Cache the INDEX
`(log path, side, turn)` and the labels; re-encode on demand at the current signature.** The
extraction is 45 min; a stale 15 GB cache silently encoding a dead layout is the more expensive
outcome, and it is precisely the failure the prober's `ArchDriftError` exists to make loud
elsewhere.

**Pre-registration obligation.** Every gate above (G1, G2, G3, G4a, G4b) with its thresholds and its
stratified re-read must be written into a runbook **before** the corresponding run — the same
discipline as `designs/research_state/genN_endofrun_runbook.md`. This document is the draft; the
runbook is the commitment.

---

## 8. What would kill this document

Written before any of it runs, because the same house has repeatedly produced clean mechanistic
stories that measurement killed.

**8.1 — The faithful subset is too thin after the mask audit.** P0.4 is the live risk. §5.6 says a
trapper species is present in **54.1%** of ≥1500 games; if the per-decision hazard rate is
comparably high, tier A ∩ mask-clean at ≥1500 could fall well below the pre-registered 50k floor,
and **rung 3 is dead as specified**. Rung 4 would then be the only route, and it would be starting
without rung 3's evidence.

**8.2 — Rung 1's offline probe (P0.2 / G1) returns null.** If a human-trained α does not beat the
Smogon prior on held-out human games, the corpus's behavioural signal is not extractable through the
one channel where the OOD taxonomy says it should be *cleanest*. Every rung above sits on strictly
worse-conditioned data, so a rung-1 null closes the document. **This is the cheapest kill available
and it must run first.**

**8.3 — Metamon's gains turn out to rest on something structural we lack.** The obvious candidate:
their setting may be one where the **team generator is callable** (Random Battles), which makes
acting-side completion a *server call* rather than a learned model — exactly the difference the
ai_v6 team-completion doc names as why Gen-3 OU needs a learned completer at all. If their result
depends on that, the precedent does not transfer and §4 is load-bearing in a way this document has
only argued for. **UNVERIFIED — P0.6 settles it.**

**8.4 — The gain is present weighted and absent on tier A.** The §2 discipline turned into a kill:
if every rung's gain concentrates in the *thin* stratum, the model is learning our reconstruction's
artifacts, and the honest conclusion is that this pipeline manufactures signal rather than
extracting it.

**8.5 — Rung 3 moves style, not skill.** Given that 1.4 was reframed to *commitment* (the policy's
soft switch mass already matches humans), the most likely rung-3 outcome is a switch rate that moves
to 30% with ELO flat. G3 is written to call that a KILL rather than a partial win — precisely so it
cannot be reported as one afterwards.

**8.6 — The rating floor forces an unwinnable trade.** If P0.1's tighter CIs show that faithfulness
and rating are *positively* coupled only weakly while the ≥1700 stratum is ~12k decisions (§5.7),
then "strong humans" and "enough data" may not co-exist in this corpus at all, and the honest read is
that the corpus supports rungs 1–2 (which tolerate the ≥1500 band) and nothing above them.
