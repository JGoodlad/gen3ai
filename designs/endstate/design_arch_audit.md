# Architecture audit: what is unprincipled, over-built or under-built (2026-10-05)

**Status: AUDIT. As of 2026-10-06 the owner has ruled on its findings (Decision record) and F11 is BUILT behind `--move-resolution` (OFF in production, §9.4); everything else in it is as audited.** Every finding carries a bucket the owner
reviews before any build: **KEEP** (principled after all, with the reason), **EXACT REFACTOR** (same function,
bundled at the X5-adoption version break) or **BEHAVIOUR CHANGE** (needs its own test, one lever each). Audited at
`43e0955d` against [`../ARCHITECTURE.md`](../ARCHITECTURE.md) (the statement of the model as it is now),
`designs/model/`, the X5 design ([`design_x5_belief_tokens.md`](design_x5_belief_tokens.md)), `UNDERSTANDING.md`,
the ledger and the code under `src/agents/model/`. Tags: **MEASURED** (a number with its source) or
**UNVERIFIED**. No code changed; no GPU used. The only new measurement is a CPU parameter census (§5, F1).

**Stance (owner, 2026-10-05).** The owner is not an ML specialist; many components arrived because a popular idea
(attention, embeddings, equivariance) was proposed and an assistant agreed. So **"we have it" is not evidence
that it is right.** Each major piece is re-justified here from first principles for THIS game at THIS scale, and
the audit starts top-down (§3: what the best-known systems do, and what fits our budget) before it goes
component by component (§5).

---

## 1. Summary for the owner (one page)

**What the network is, in one paragraph.** Each of the 12 Pokémon (6 ours, 6 theirs) becomes a vector of 128
numbers (a *token*). Extra tokens stand for our active Pokémon's four moves, their believed threatening moves,
the board, and the last 32 battle events: 61 tokens in all. A *transformer trunk* (2 rounds of *attention*: each
token reads a learned weighted mix of every other token) lets them exchange information. A hand-written
*damage operator* computes gen-3 damage, speed and status physics exactly and feeds it in three ways: as extra
token content, as nudges to who-attends-to-whom (*edge biases*), and as per-action numbers next to each choice.
*Belief heads* guess the opponent's hidden moves, items and spreads from Smogon statistics plus learned
corrections. A *pointer head* scores each of our 11 possible actions from the token of the thing that action
picks. The *critic* is a small head that outputs P(win).

**The top findings, in plain words.**

1. **We never compared the network against a simple one.** Every measurement is a within-model "turn a piece
   off and see what moves" reading (*ablation*). No run ever trained a plain network (for example a
   multi-layer perceptron, MLP: a stack of fully-connected layers, on the flat observation) under the same
   recipe. So we do not know how much the whole entity design buys at our scale. **The one accidental simple
   run (`ai_v12_01`, 2026-09-06: no edge biases, no extra move seats, no intent heads, no event history, 24.4M
   steps) read 31 Elo ahead of its comparator, inside the 38-Elo noise floor, and was discarded as "a different
   model" instead of being read as a free control.** (F23, F24; BEHAVIOUR CHANGE, HIGH.)
2. **19% of the parameters do nothing.** Under the win-probability critic the old value tower (592,129 of
   3,065,882 parameters, MEASURED here) is computed on every forward pass and is in no loss: one training update
   leaves every one of its weights bit-identical. Deleting it changes nothing the model does. (F1; EXACT
   REFACTOR, HIGH. **DONE 2026-10-07** at the version break's part 2: production then held 2,519,046 parameters;
   2,519,007 after part 4's slot-tied `out_gain`.)
3. **The capacity is in the wrong place.** The attention trunk, where the reasoning happens, has 284k parameters
   and does 63% of the arithmetic. A flat policy tower inherited from the library default (`net_arch [512,512]`
   with a `tanh` nonlinearity nobody chose) has 1.13M parameters and produces only a "general mood" vector the
   action scorer adds in. (F2, F5; BEHAVIOUR CHANGE, HIGH.)
4. **Board facts are copied into every Pokémon's token.** Clock, weather, hazards, screens and faint counts are
   fed into each of the 12 Pokémon encoders (and each of their 48 move encoders), plus a board token, plus the
   policy's input, plus the physics: four routes for one fact. Your 2026-10-05 direction (Pokémon tokens static,
   battle context mixed by attention) is the principled fix and the lead hypothesis (§4). (F4; BEHAVIOUR CHANGE,
   HIGH.)
5. **The trunk was sized for a different problem.** Two attention rounds were chosen in the ai_v4 era, when the trunk saw
   roughly the 12 team tokens (UNVERIFIED exact count); there are now 61 tokens and 17 kinds of edge bias, and the trunk was never re-sized. Its
   normalisation is *post-LN* (LayerNorm after each residual add), which is fine at 2 layers and known to train
   badly when stacked deeper; *pre-LN* is the standard choice once depth grows. (F5; part of §4.)
6. **The physics' "worst move" summary is incoherent and numerically fragile.** For each of our Pokémon the
   operator takes, independently per number, the maximum over the opponent's believed moves: the "low roll" can
   come from one move and the "KO chance" from another. The same hard maximum (~40 call sites) is what produced
   the 2026-10-05 compiled-gradient NaN; it was fixed for the X5 arm only. (F6; EXACT for the NaN class,
   BEHAVIOUR CHANGE for the reduction, MEDIUM. The EXACT half is **DONE 2026-10-07**: every gradient-path max is
   `max_by_index`.)
7. **Seven hand-designed strategy feature blocks sit beside each action** (Focus Punch / Endure / Destiny Bond
   thresholds, "don't click the KO into the obvious switch", Rapid-Spin-into-Ghost, …), widening the move scorer's
   input to 62 numbers. Some are exact physics a small scorer cannot multiply out (principled); others are human
   strategy heuristics with no measured payoff. (F11; BEHAVIOUR CHANGE as one family ablation, MEDIUM.)
8. **Almost all the evidence for the architecture is old.** The per-block and per-edge-family dependence tables
   come from a gen-3 checkpoint (2026-08-07) at 9.6M steps, before the Baton Pass, bot-setup, move-prior and
   species-usage fixes. Nothing in ARCHITECTURE.md measures the current model. (F22; a cheap CPU re-read, HIGH.)
9. **The 32 event tokens double the trunk.** History is principled, but 32 seats (from 29 tokens to 61, ~4.4×
   the attention matrix) was never sized against 8 or 16, and the trunk is where X5's cost overrun lives. (F19;
   BEHAVIOUR CHANGE, MEDIUM.)
10. **About 60 switches that are always on** (T21) and their never-run "off" branches, plus fallback code paths
    for configurations production never uses. (F8; EXACT REFACTOR, MEDIUM.)

**What is principled and should stay** (§6): the damage operator as a way to *give* the model known physics
instead of making it rediscover arithmetic from ~10^5 games; the pointer head (the standard way to score a
variable set of choices); the win-probability critic; the parameter-free Smogon team prior; zero-initialised
add-ons; the leak-safety design. The entity-token idea is principled *in theory* (a team is a set; treating it
as a set saves the model from learning 720 orderings), but at our scale it has never been shown to beat a
plain baseline (finding 1).

**Counts.** 33 findings, 37 bucket entries: **KEEP 18 · EXACT REFACTOR 6 · BEHAVIOUR CHANGE 13** (§7). The single most valuable next
step is cheap: train the simple controls (§3.3) and read the accidental one.

---

## 2. The vocabulary, in one or two sentences each

| term | what it means here |
|---|---|
| **token / embedding** | a fixed-length vector of numbers standing for one thing (a Pokémon, a move, an event). An *embedding* is a learned lookup table from an id (species number) to such a vector |
| **attention / transformer** | each token computes a weighted average of the other tokens, with weights it learns to choose ("who should I listen to"). A *transformer layer* is one round of attention plus a small per-token network |
| **set / equivariance** | a team is a set: slot 3 vs slot 5 means nothing. A design is *equivariant* if reordering the inputs just reorders the outputs, so the model never has to learn the 720 orderings separately |
| **edge bias** | a number added to the attention weight between two specific tokens (e.g. "my move k vs their mon d: 45% damage"), so physics can steer who listens to whom. It shifts a *ratio*, it cannot carry an absolute amount |
| **pointer head** | instead of one output unit per action slot, each action is scored from the token of the thing it selects (switch j ← Pokémon j's token), so the scorer is shared |
| **critic / value head** | the part that predicts how good a state is; here P(win). Training uses it to judge whether an action turned out better or worse than expected |
| **auxiliary head / aux loss** | an extra output trained on a side task (guess their item); its loss may also reshape the shared trunk (*shaping*) or be blocked from it (*detached*) |
| **ablation / dV / KL** | switch one piece off and measure how much the policy (KL divergence, argmax flips) or the value (dV) moves. It measures *dependence*, not whether the information has another home (*coverage*) |
| **LayerNorm, pre-LN / post-LN** | a per-token rescaling that keeps numbers well-behaved. *Post-LN* normalises after adding a layer's output; *pre-LN* normalises the layer's input and leaves the residual path clean, which is what lets deep stacks train |
| **MLP** | multi-layer perceptron: fully-connected layers, no notion of entities |
| **zero-init** | a new add-on starts with weights 0, so it changes nothing until training finds it useful |

---

## 3. Top-down: what the best systems do, what fits our size, and the diff

### 3.1 What the best-known approaches for games like ours look like

Our game: gen-3 OU singles, imperfect information (hidden sets), simultaneous moves, 11 actions per decision,
~10^5 games per run, self-play reinforcement learning (RL), one consumer GPU.

| system | game, information | what it does | what transfers to us |
|---|---|---|---|
| **AlphaStar** (Vinyals et al., *Nature* 2019) | StarCraft II, imperfect, real-time | supervised pretraining on human replays, then a LEAGUE (main agents + exploiters, prioritised fictitious self-play); a transformer over unit entities, an LSTM memory core, a pointer network to select units, autoregressive action heads; off-policy actor-critic (V-trace, UPGO). Thousands of accelerators | entity transformer + pointer selection (we have both); league with exploiters (we have pool + PFSP + exploiters); **human-data bootstrap (we have none)** |
| **OpenAI Five** (Berner et al., arXiv:1912.06680) | Dota 2, imperfect | PPO at very large scale; mostly hand-engineered features, per-unit embeddings max-pooled, a single large LSTM; target selection by attention over units; continual "surgery" to grow the model | PPO and hand features work at scale; their per-unit pooling is far simpler than our trunk |
| **DeepNash** (Perolat et al., *Science* 2022) | Stratego, imperfect, huge hidden state | model-free, no search; Regularised Nash Dynamics (R-NaD) to converge toward an equilibrium in self-play; a convolutional torso | for imperfect information, the *training dynamics* (equilibrium-seeking self-play) mattered more than architecture novelty |
| **Pluribus / ReBeL** (Brown & Sandholm, *Science* 2019; Brown et al., NeurIPS 2020) | poker, imperfect | Pluribus: a counterfactual-regret blueprint plus real-time depth-limited search, small inference hardware. ReBeL: search over PUBLIC BELIEF STATES with a learned value net | strength from SEARCH over beliefs, with a value function as the leaf; our search was wound down (owner 2026-09-12) |
| **Metamon** (Grigsby et al. 2025, "Human-Level Competitive Pokémon via Scalable Offline RL with Transformers", arXiv:2504.04395) | Pokémon Showdown gen 1–4 OU, imperfect | offline RL on hundreds of thousands of human replays reconstructed to a first-person view, then self-generated data; transformer policies over the battle history; reaches strong human-ladder play. **UNVERIFIED:** the exact model sizes (we believe tens to ~200M parameters) and data counts | THE closest precedent. Its lever is DATA (human battles) and model size, not hand physics. Its `Kakuna` policy is our near-term target |
| **Foul Play** (pmariglia, open source) | Pokémon Showdown | no learning: search (MCTS / expectiminimax family) over sampled opponent sets from Smogon usage, with an exact battle engine and a hand evaluation. **MEASURED:** it beats our 75M win-prob arm, our win rate 0.388 [0.288, 0.497] over 80 games, ~2.1 s per decision vs our 50 ms (ledger 2026-09-14) | exact physics + Smogon beliefs + search is already strong in this game without any network |

**The common pattern.** Strong systems put their budget into (1) the TRAINING SIGNAL (human data, league
self-play, equilibrium-seeking updates) and (2) SEARCH at decision time where the game allows it, and use a
fairly plain network: entities embedded, pooled or attended, a memory, a pointer for selection. None of them
hand-designs dozens of per-action strategy features, and the two closest to our game reach their strength
through data (Metamon) or search (Foul Play), not architecture.

### 3.2 What is appropriate at OUR size

Budget: one RTX 3080 Ti; ~2,000 environment steps/s (owner's figure); 15M-step runs (order 10^5 games); a
3.07M-parameter model (`D_MODEL` 128, 2 trunk layers).

- **Sample efficiency dominates.** With ~10^5 games per run, anything the model must learn from scratch is
  expensive. Giving it exact, cheap physics (the damage operator) and good priors (Smogon) is the right trade
  at this size; Foul Play shows exact physics plus Smogon beliefs is already strong. **KEEP that.**
- **Structure that removes symmetry is cheap and helps** (set treatment of the team, a pointer head). KEEP,
  but *verify against a plain control* (§3.3), because at small data the gain is plausible, not proven.
- **Hand-designed strategy features are where small projects over-build.** Each one is a human guess about what
  matters, costs code, versioning and a toggle, and its payoff is rarely measured. Prefer giving facts (physics)
  over giving conclusions (heuristics).
- **Capacity should sit where the reasoning is** (the trunk), not in a flat tower after it.
- **The biggest levers per the literature are outside the network**: human data (Metamon; our ai_v11 chapter is
  OPEN, nothing built; the owner's rule allows ladder replays as data) and test-time search (Foul Play). This
  audit flags them, it does not re-open them (rule 9: scope).

### 3.3 The diff, and the simpler controls we never ran

| area | top-down expectation at our size | what we have | verdict |
|---|---|---|---|
| physics provision | yes, exact and cheap | the damage operator, 17 edge families, pointer cells | right idea; delivered through too many routes (F4, F10) |
| entity encoding | entities as tokens, simple | per-mon encoder that also re-reads the whole board (F4) | over-built per token, under-built in the trunk (F5) |
| trunk | a few rounds, pre-LN, sized to the token count | 2 post-LN rounds for 61 tokens | under-built (F5) |
| flat MLP tower after the trunk | small or none | 1.13M params, never chosen (F2) | over-built |
| per-action features | physics facts only | 7 heuristic blocks, 62-wide move cell (F11) | over-built |
| belief | Smogon prior + learned correction | yes, six heads with aux losses shaping the trunk | principled; trunk shaping unmeasured (F9) |
| critic | P(win) | P(win), 17k-param head; a dead 592k tower beside it (F1; DELETED at the version break, 2026-10-07) | right idea; clean up (done) |
| memory | turn history | 32 event seats in the trunk (F19) | principled; size unmeasured |
| training data | human bootstrap + self-play league | self-play league only | under-built (out of this audit's scope) |
| controls | a plain baseline at matched budget | none ever trained deliberately | **missing (F24)** |

**Candidate controls, each one 15M-step arm under the production recipe** (three seeds, the X5 A/B's replicate
design; meter: the mirrored head-to-head `main.h2h` against blob seeds, plus the `SmallRL` greedy anchor):

- **C-MLP:** the flat 2761-dim observation through an MLP of matched parameter count into the same pointer head
  (keeps the action interface, drops entities, trunk and operator). Answers "what does the whole entity design
  buy".
- **C-ENT:** entity tokens + the 2-layer trunk + the pointer head, with the operator, edges, cells and belief
  heads OFF (close to the accidental `ai_v12_01`). Answers "what does all the hand physics buy on top of entities".
- **C-NOCELL:** production minus the seven heuristic blocks (F11). Answers "do the strategy heuristics pay".

An outcome where C-ENT or C-NOCELL is not worse than production at matched steps would be the most important
result this project could produce cheaply: it would say most of the hand engineering is not where the strength
comes from.

---

## 4. The lead hypothesis: static Pokémon tokens, attention mixes the battle

**The idea (owner, 2026-10-05).** A Pokémon's token should say *who it is* (species, its known or believed set,
and its own per-mon state), computed once from a table. Everything about the *battle around it* (clock,
weather, hazards, screens, faint counts, the other side's state) should reach it through attention, where the
model decides what to read, "even at the cost of another round".

**Why it is principled.**
- *Learned over hand-crafted:* today the per-mon encoder decides, by construction, that every Pokémon reads the
  weather and the clock in its first layer. Under the lead design the trunk learns which tokens read which
  context.
- *One fact, one home* (the sorting rule in [`../learning/entity_tokens_biases_pointers.md`](../learning/entity_tokens_biases_pointers.md)
  Part 3): board facts are board-scoped and belong in a board token; per-entity facts belong on the entity.
- *Observability:* a static token is readable ("this is Skarmory with Spikes, Roar") without a probe.
- *Cost:* X5's hypothesis encoding is row-dependent only through the encoder's first Linears (MEASURED,
  `065d2d7f`, ledger 2026-10-05: encoding 10.09 → 4.38 ms per micro-batch after the exact split; `train_ms` still
  +12.8% and the inference tier +45.6% over blob, both over budget). Fully static tokens make a hypothesis a pure
  table gather.

**The concrete proposal (option A, recommended).** Remove the five row-level inputs (clock, weather, faint
counts, hazards, screens) from the move network and the role encoder; move each side's 60-dim active context
(boosts + volatiles; today scattered onto the active's row, the "E2 injection") into an *active-context* token per
side, linked to its active Pokémon by a structural edge. Keep per-mon dynamic state that belongs to the mon (HP,
status, revealed moves and PP, recency, last action, trapped) on the mon. *Option B*, fully static tokens with
every per-mon state as its own token, is over-reach at 128 dims and 61 tokens; not recommended.

**The board layout for the rebuild** is inventoried and recommended fact by fact in
[`design_entity_coverage_audit.md`](design_entity_coverage_audit.md) (2026-10-06): OUR SIDE / THEIR SIDE / FIELD tokens
in place of the global token, plus an op-derived content input per mon so amounts (Spikes chip, end-of-turn ledger)
survive the broadcast's removal.

**As designed and (stage 1) built:** [`design_static_tokens.md`](design_static_tokens.md) (`--token-encoding static`,
2026-10-06): the per-mon token is S (static identity) + D (the mon's own state) with no board input, and an X5
hypothesis token is a dex-table gather (stage 1, built); the board layout above and the per-mon op content are stage 2
(not built). Not screen-ready until stage 2 lands.

**What it costs.** The trunk must now do mixing the encoder did. That is the case for a third round, and a third
post-LN round is the regime where post-LN is known to train less stably, so pre-LN comes with it.

**How to test it (one lever per arm; all after X5's look 1 is read, the blob-identity precondition):**

| arm | change vs the arm above it | question |
|---|---|---|
| L0 | production blob (or the adopted X5 arm) | reference |
| L1 | option A, 2 layers post-LN | does removing the broadcast cost strength? (a cost and observability win if not) |
| L2 | L1 + pre-LN, 2 layers | does the norm move matter at depth 2? (expected: no; a cheap guard) |
| L3 | L2 + a third layer | does the extra round pay? |

Meter: the mirrored head-to-head at matched steps AND matched wall time (X5 Amendment 5's precedent), three
seeds each; secondary: `train_ms`, inference-tier ms, the critic's sibling discrimination (the 0.517 meter) and
the belief purpose metrics. Decision rule to pre-register: adopt L1 on EQUIVALENCE (the delta's own confidence
interval inside a bar the owner sets, e.g. ±3 pp), adopt L3 only on SUPERIORITY at matched wall time. Cost:
~1.5 agent-days of build (an `ARCH_SIGNATURE` bump), 9 runs × ~2 h GPU. Combine with F2 (shrink the flat tower)
only as its own arm.

---

## 5. Findings

Each row: what · the principle it breaks · provenance · evidence today · bucket · cost/benefit · priority.
Principles are the owner's list: P1 learned over hand-crafted, P2 Smogon-only priors, P3 observability, P4
declared lifecycle, P5 deterministic checks, P6 sets stay sets, P7 cheapest non-regression / one lever, P8 the
critic is P(win), P9 dependence is not coverage; plus the scope extension's SMELLS (S).

### A. Dead or misplaced capacity

**F1 · The dead SB3 value tower.** `features_extractor.value_pre_norm` + `value_projection` (128→512) +
`mlp_extractor.value_net` (512→512→512, tanh) + `value_net` (512→1). Under the win-prob critic `_critic_value`
returns `sigmoid(win_head logit)` and reads `latent_vf` only for a batch-size check (`policy.py:174-207`).
- *Principle:* S (complexity without payoff), P3 (a reader can mistake `value_net` for the critic; the code
  raises to stop exactly that).
- *Provenance:* SB3's `MaskableActorCriticPolicy` always builds it; it became dead when the win-prob critic
  became the only critic (`gen3_winprob_critic_mode_v1`, 2026-09-06; shaped critic untrainable since D4).
- *Evidence:* **MEASURED here** (CPU, `learner_golden.build_learner()` + one `train()` on the committed golden
  buffer at `43e0955d`): 592,129 of 3,065,882 parameters (19.3%) are bit-identical after the update; every
  forward computes them (~1.2 MFLOP of 56.65 per row, ~2%). **UNVERIFIED:** whether the inference tier (T2)
  also computes it on every rollout decision.
- *Bucket:* **EXACT REFACTOR** (pi and V outputs unchanged; init bytes of later modules shift, so the K9 golden
  is re-recorded at the version break). *Cost* ~0.5 agent-day. *Priority:* HIGH.
- *As built (2026-10-07, the version break part 2):* DELETED — the extractor's `value_pre_norm` / `value_projection`,
  SB3's `mlp_extractor.value_net` and `policy.value_net`, 592,129 parameters. The extractor's value half IS
  `value_pooled` (`[B, 128]`, `vf_features_dim`); `_critic_value` returns `sigmoid(fe.last_win_prob_logits)`;
  the policy no longer calls SB3's `_build` (an actor-only `MlpExtractor`, the orthogonal re-init in SB3's order,
  the retire hooks, the pointer head, the optimizer; `action_net` / `value_net` are raising stubs); a `critic`
  other than `winprob` is refused before anything is built. The T2 question of §8 is ANSWERED (below). Weight-
  mapping identity BITWISE (forward, every surviving post-update parameter, every pinned loss; `designs/research_state/measurements/version_break_identity_2026-10-07/`).
  See the Decision record's 2026-10-07 EXACT-bundle row.

**F2 · The flat policy tower holds 37% of the parameters for a context vector.** `projection` (1177→512, 603k)
+ `mlp_extractor.policy_net` (512→512→512, 525k) produce `latent_pi`, which the pointer head reads only through
`ctx_proj` (512→64, added inside a tanh). 768 of the 1177 inputs are the six `HiddenOppBeliefPool` queries.
- *Principle:* P1/P7 (capacity allocated by inheritance, not intention), S (activations mixed by accident:
  ReLU after the projection, tanh in the tower, `policy.py:36-58` records that tanh "was NEVER chosen here").
- *Provenance:* `NET_ARCH [512,512]` dates from the flat `action_net` era (pre-v51); the pointer head (v51)
  kept it as context.
- *Evidence:* MEASURED parameter counts (above). No test ever shrank it. **UNVERIFIED** whether it matters.
- *Bucket:* **BEHAVIOUR CHANGE.** Test: `net_arch` pi = [] (context = the projection output) and a 256-wide
  projection, one arm; meter as §4. Reallocating the freed parameters to the trunk is a SEPARATE arm. *Cost:*
  0.5 day + 3 runs. *Priority:* MEDIUM-HIGH.
- *As built (2026-10-06, owner: "no need to keep the tower"):* the arm went further than the `net_arch pi = []`
  test above: `--policy-readout trunk` (default `tower` = production, byte-identical) RETIRES the whole tower
  (the projection AND `policy_net`) and reads the decision context off the trunk with one learned attention
  query (`PolicyStateQuery`) over every refined token plus the belief pool's K outputs; each action is still
  scored from its own token by the F16 scorer, widened to 128. Parameters 3,065,882 → 1,991,528 (MEASURED,
  production surface). Built, not yet trained; the screen is the §4 meter. See the Decision record.

**F3 · The critic head is small (`win_head`, LayerNorm→128→ReLU→1, 16,897 params).** UNDERSTANDING §4.3 lists
"whether `win_head`'s architecture is right for a critic" as UNVERIFIED.
- *Evidence:* the binding constraint measured so far is DATA, not the head: a frozen trunk with only the head
  refit on counterfactual successors lifts sibling discrimination 0.517 → 0.603 (DETECTED), and a ranking loss
  buys nothing (`paired_refit_discrimination_2026-09-14/`).
- *Bucket:* **KEEP** (the evidence points away from the head). Re-open if a fork-data arm saturates.

**F4 · Board context is broadcast into every token (four routes for one fact).** The move network and the role
encoder of all 12 Pokémon read clock, weather, faint counts, hazards and screens (`hypothesis_encode.py:54-71`);
each side's 60-dim active context is scattered onto its active's row (E2, `e2_ctx_injection_test.py`); the
global token carries both contexts + the 25 board scalars; the same 25 scalars go straight into the policy
projection; the operator prices weather and screens.
- *Principle:* P1, P3, the sorting rule (board-scoped facts in a board home), S (overlap).
- *Provenance:* the role encoder's broadcast global context dates from ai_v2; E2 was added as "additive delivery"
  beside the global token; `gen3_ctx_dedup_v1` (v76) removed one duplicate route on measurement.
- *Evidence:* MEASURED that the per-mon encoding depends on the row only through these columns (`065d2d7f`). No
  measurement of what each route contributes.
- *Bucket:* **BEHAVIOUR CHANGE** = §4 L1. *Priority:* HIGH.

**F5 · The trunk: 2 post-LN layers, 128 wide, 4 heads, chosen when it saw ~12 tokens, now 61.**
`TRANSFORMER_N_LAYERS = 2`, `BiasedEncoderLayer` is post-LN (`team_transformer.py:57-58`).
- *Principle:* "appropriate at the time" (S), P7.
- *Provenance:* ai_v4's "unified L=2 transformer extractor"; tokens grew with the seats (v51+) and the event
  window (32 seats) without a re-size.
- *Evidence:* MEASURED the trunk is 63% of forward FLOPs (X5 §3.6) and 9% of parameters (284,416). The
  strongest prior against "more depth" is about SEARCH depth and the deleted `damage_refine_rounds` loop
  (learning note Part 4), not network depth: **network depth is untested.**
- *Bucket:* **BEHAVIOUR CHANGE** = §4 L2/L3. Pre-LN at 2 layers alone is NOT exact (it changes the function).
  *Priority:* HIGH, with §4.

### B. Fixed reductions and hand-coded numbers

**F6 · The operator's per-channel hard maxima over believed moves.** The incoming per-mon row takes
`(w·value).amax` separately for low, high, crit, KO-chance and accuracy, per physical/special channel
(`damage_op.py`, plus C1b/C2/C3/D4 maxima in `damage_op_pairwise.py`; ~40 max call sites in the forward
modules, counted here). `provenance` and `acc` come from the argmax of yet another channel.
- *Principle:* P1 (a fixed reduction where the target is a decision quantity, not physics); the five maxima are
  INCOHERENT (each may come from a different move: `design_pair_reduction.md`'s "D2"); `max_m(w_m·v_m)` is
  neither the expectation nor the worst case (X5 §9 M2 declared its meaning instead of fixing it).
- *Provenance:* v23 unified damage, the R0 `hard_max` rung; the principled rungs (R1 `belief_mean`, R2
  learned, R3 multi) were BUILT in `pair_reduce.py` and left INERT, constructor-only (ARCHITECTURE §4).
- *Evidence:* the NaN class is MEASURED (`x5_fxc4_nanfix_2026-10-05/`: Inductor recomputes the operand in the
  backward, FMA contraction rounds it differently, `x == amax` matches nothing, 0/0). Fixed by `max_by_index`
  for fixed_mass ONLY; blob still uses `amax`. Blob is finite today because of how its kernels fuse, and the
  startup gate now names a non-finite gradient (fail-loud, not silent).
- *Bucket:* (a) **EXACT REFACTOR**: `max_by_index` at every max site (value bit-identical; gradient on an exact
  tie goes to the first maximum instead of being split, a declared convention change, K9 re-record). (b)
  **BEHAVIOUR CHANGE**: replace the per-channel max with the α-weighted expectation row that `pair_outcome`
  already computes, or a learned pool over the per-candidate cells; one arm. *Priority:* (a) MEDIUM (removes a
  latent crash on the next torch upgrade), (b) MEDIUM.
- *As built (2026-10-07, the version break part 2, (a)):* `max_by_index` moved to the leaf module
  `agents/model/index_max.py` (`max_by_index(x, dim=-1, keepdim=False)`; `damage_op` re-exports it) and is THE
  spelling of every gradient-path value-reduction max in the forward, in EVERY configuration (the op's ten
  incoming channel maxima incl. the belief-off branch, the pairwise kernels, the status-landing maxima, the E5
  tail's worst-phys/spec, `pair_reduce`'s inert deepsets pool). `amax` / `amin` remain only off any gradient
  path (the provenance gate operand, the cure/cleric table lookups, the cheapest-undo minimum, the `we_have_pur`
  indicator, `fixed_size_tau`'s `no_grad` bracket, the move-tie-gap diagnostics). It moved NO byte of the K9
  update (no exact tie with a nonzero upstream gradient on that buffer). (b) is not built.

**F7 · The outspeed logistic and other non-physics constants.** `_DMG_SPEED_SCALE = 15.0` (the logistic
temperature for P(outspeed), "~one stage", hand-chosen); its uncertainty-aware variant was deleted, yet ~6
`SPECIES_SPREAD_PRIOR[..., spe, 1]` lookups still run every forward and are discarded (`damage_op.py:657-672`,
banked in `flag_census_2026-09-06.md`); `_DMG_CHIP_CAP 1.5`, `_DMG_CRIT_CAP 3.0` (clamps); `_REVEAL_LOGIT 10`,
`_HP_PRESENCE_OFF_LOGIT −30`; the E5 tail score `w·BP/150·acc` vs the move slot's `power/200`.
- *Principle:* magic constants without provenance (S); P1 for the speed logistic (the spread belief already
  carries a believed speed and could carry its spread).
- *Bucket:* the discarded lookups **EXACT REFACTOR** (LOW); the speed logistic **BEHAVIOUR CHANGE** (use the
  belief's variance; LOW); the scale normalisers and logit pins **KEEP** (a learned Linear absorbs any fixed
  scale; the pins are documented saturations). The gen-3 RULE constants (roll 0.85–1.0, crit 1/16, paralysis
  ×0.25 speed, Choice Band ×1.5, the Beat Up formula) are exact physics: **KEEP**.
- *As built (2026-10-07, F7b):* `--speed-physics {off,on}` (v143, OFF): under `on` the logistic is replaced at
  every site by `move_order.p_first_same_priority` (the exact gen-3 order rules; Quick Claw format-gated off,
  banned in gen3ou). Their speed was first a Gaussian on the integer lattice (the spread belief's mean, the
  Smogon prior's σ); it was not better calibrated than the logistic on real battles (log loss 0.1367 vs 0.1341)
  because real Speed investment is LUMPY (a Timid 252-Speed Blissey sat 15 σ out). **Now (gen3_speed_mixture_v1,
  same day): their speed is the species' DISCRETE Smogon spreads mixture over the Speed stat** (every chaos spread,
  no top-25 cut; each support point through the exact stage / paralysis arithmetic); the learned spread belief is
  not read there. On the Lane S bank: log loss 0.1308 (best of the three, cold-start), 0 rows certain and wrong,
  but Brier 0.0440 and ECE 0.0184 (worst of the three; the upper-middle bins over-confident) — still not better
  calibrated overall; conditioning the mixture on the battle is the open lever. See the Decision record's
  2026-10-07 F7b rows.
- *As built (2026-10-07, the version break part 2, (a)):* the off-path (`--speed-physics off`, production)
  `SPECIES_SPREAD_PRIOR[..., spe, 1]` lookups (one in `damage_op`, two in `damage_op_blocks`, three in
  `damage_op_pairwise`) and `_p_outspeed`'s ignored `opp_spe_std` argument are DELETED (`pair_outcome_coords`
  lost the parameter); X5's OTHER roster computes its speed-sigma average only under `on`. Under `on` at that
  commit `damage_op_speed._opp_speeds_belief` still read the sigma column. **FINDING:** main's later
  `gen3_speed_mixture_v1` (`50b034a7`, the discrete Smogon spreads mixture) deletes that Gaussian read, after
  which the roster's `spe_std` field and its `_x5_avg` read are dead in BOTH modes.

**F8 · ~60 single-production-value toggles, and code paths production never runs.** Every row of ARCHITECTURE
§6's flag table that is `ACTIVE` is one value forever in practice; behind them sit fallback paths: the legacy
"de-timid 252 EV / neutral 0-EV bulk" constants when `spread_belief` is off (`damage_op.py:790-800`,
`damage_op_blocks.py:211-215`), the R1 `belief_mean` fallback when `opp_intent` is off, the inert
`pair_reduce` rungs.
- *Principle:* S (complexity without payoff; each toggle is a registry row, a version field, a resume check and
  tests).
- *Bucket:* **EXACT REFACTOR** = backlog T21 (families of ablation, after X5). *Priority:* MEDIUM.

### C. Overlapping routes and trunk-shaping heads

**F9 · Six belief heads shape the trunk at a uniform 0.05, the intent head does not.** `belief_grad_mode
"shaping"` vs `opp_intent_grad_mode "detached"`; every aux coefficient is 0.05; `intent_label_bot_weight` 0.25.
- *Principle:* S (inconsistent conventions; heads that reshape the trunk without evidence that the reshaping,
  as opposed to the head's output, helps); "appropriate at the time" for the uniform 0.05.
- *Evidence:* MEASURED that the learned belief corrections earn +5.2 pp against the POOL they memorised and
  nothing detectable against ladder teams (ledger 2026-09-24, DiD +4.8 [+1.9, +7.8]); that test zeroed the
  OUTPUTS, not the trunk shaping. **UNVERIFIED:** any effect of shaping vs detached.
- *Bucket:* **BEHAVIOUR CHANGE**: one arm at `belief_grad_mode detached` (built). *Priority:* LOW-MEDIUM.

**F10 · The critic reads the operator's incoming rows three ways.** Through the trunk (the `prefuse_proj`
token injection), through `value_entity_pool`'s op-row source, and through `value_threat_inject` (an
α-weighted row added to the value pool's copy of each token, whose α is a deliberately crude presence belief,
"so a null indicts the delivery route", ARCHITECTURE §3.2).
- *Principle:* S (overlap), P9 (the dependence readings, entity pool dV 5.490, threat 1.069, gen-14, say the
  critic LEANS on them, not that the routes carry distinct facts).
- *Bucket:* **BEHAVIOUR CHANGE** (delete `value_threat_inject`, read critic discrimination). *Priority:* LOW.
- *As built (2026-10-07, config v142, `gen3_value_threat_inject_off_v1`):* the existing `--value-threat-inject`
  flag (ON in production) is the lever; its `off` value is now a ONE-lever arm. With the op built, OFF constructs
  the projection NOT LIVE and the policy RETIRES it after SB3's orthogonal re-init (the `retire_superseded_intent_heads`
  precedent), so a trained OFF model has no module, key, optimizer slot or parameter for the route (production
  learner 3,065,882 → 3,064,090, −1,792 = the `Linear(13, 128)`) and every other initial byte equals production's
  (before v142 an OFF build moved 185 tensors' init draws). The op stays on `hard_max`. The critic then reads the
  op's rows through the trunk (`prefuse_proj`) AND `value_entity_pool`'s op-row source: F10 deletes one of three
  routes, not two. `--arch production --value-threat-inject off --allow-nonproduction-arch` is the screen arm;
  composes with `--token-encoding static`, `--policy-readout trunk`, `--move-resolution on` and X5's fixed-mass
  surface (tested; production since the X5 version break, v144). **The screen's meter** (not registered, not run): `python -m main.ops.critic_read <off arm>
  --control <production arm> --out <dir>`, read on its delta rows — PRIMARY `gate.resolution.bot` and
  `gate.resolution.pool` (G1: Murphy resolution of V against the outcome, the discrimination term, per stratum,
  from `main.critic_gate`); SECONDARY `identity.resolution` (Murphy resolution, IPW, all strata),
  `gate.skill.bot` / `gate.skill.pool` (G4) and `identity.turn_contrast` (corr(turn, V) − corr(turn, MC): does V
  track the game as it unfolds); `gate.reliability.*` / `gate.ece.*` as the calibration guard. Caveats: Murphy
  resolution pools states ACROSS battles, so it is not strictly WITHIN-game; the within-game sibling pairwise
  accuracy (the 0.517 meter, `paired_refit_discrimination_2026-09-14/`) is not a `critic_read` row; and with no
  `--floor-json` every label reads *vs ZERO — NO FLOOR*. Built, not trained; GPU checks DEFERRED (no lease).

**F11 · Seven hand-designed per-action blocks.** `intent_move_cell` (7), `intent_threshold` (6),
`intent_conditional` (13), `pair_outcome_cell` (14), `switch_branch_cell` (9) on the move cell; `pair_outcome_switch`
(15), `conditional_threat_cell` (4) on the switch cell (move cell 13 → 62 wide, switch cell 15 → 34).
- *Principle:* P1. Two kinds are mixed: PHYSICS products a small tanh scorer cannot form from its inputs
  (`e_pko_acc = Σα·ko_ramp·acc`, damage margins; principled by "precompute every nonlinearity of two numbers in
  the operator") and STRATEGY heuristics (`wasted_ko`, `spin_denied`, `p_spin_blocked`, the Focus Punch /
  Endure / Destiny Bond thresholds, `tempo_cost`'s undo-path table).
- *Provenance:* v84–v95, each from a design doc's hypothesis; G2's usage baseline before the build: Endure 0.0%,
  Substitute 0.9%, Counter 5.6% of decisions.
- *Evidence:* **UNVERIFIED payoff.** No end-of-run read of any block's dependence is cited in ARCHITECTURE.md.
- *Bucket:* the physics coordinates **KEEP**; the heuristic family **BEHAVIOUR CHANGE** as ONE ablation
  (C-NOCELL, §3.3). *Priority:* MEDIUM.

**F12 · 17 edge families.** Cheap (696 parameters in all) and principled as a mechanism, but the only
per-family audit is gen-3 at 9.6M of 40M (ARCHITECTURE §5.4): c1–c5, g, x at ≤ 0.4% argmax flips.
- *Bucket:* **KEEP** as a mechanism; re-read with F22 before any family decision (P9: a low dependence is not
  a licence to delete). *Priority:* LOW.

**F13 · `HiddenOppBeliefPool`: six learned queries, concatenated (768 dims) into the policy input.** Order is
by QUERY (a learned role), not by opponent slot, so no positional leak.
- *Bucket:* **KEEP** (a Set-Transformer-style pooling; its vf half was deleted on a dV of 0.0000). Its width is
  part of F2.

### D. Sets, order and conventions

**F14 · The per-mon move set is concatenated in sorted-by-id order.** The role encoder concatenates the 4 move
encodings after within-mon self-attention (ARCHITECTURE §1.3), so a set enters an order-sensitive Linear;
sorting by id makes it deterministic, not symmetric.
- *Principle:* P6. *Bucket:* **BEHAVIOUR CHANGE** (sum or attention-pool the four; fold into §4 L1 only if the
  owner accepts two levers, otherwise its own arm). *Priority:* LOW.

**F15 · Post-LN, a −1e9 key-padding addend, edge maps shared by both layers.** The −1e9 (not −inf) is
documented as a compile/NaN guard; sharing one edge-bias map across layers is the Graphormer convention
(**UNVERIFIED** detail of that paper). *Bucket:* **KEEP** (post-LN re-opened only by §4).

**F16 · The pointer head.** `tanh(proj(token ⊕ cells) + ctx_proj(latent_pi))` → a zero-init `Linear(64,1)`, one
shared scorer per family; equivariant by construction. *Merit:* the standard answer to scoring a variable set
of choices (pointer networks; AlphaStar's unit selection). *Bucket:* **KEEP**. One provably dead parameter:
`beta_head.scorer.2.bias` (a shared bias inside one softmax is shift-invariant, MEASURED unmoved). EXACT,
trivial, bundle with F1.
- *As built (2026-10-07, the version break part 2, (b)):* the α / β heads are deleted (X5), so the live instance
  is X5's flat opponent pointer: `FlatIntentHead.out` is `IsolatedLinear(hidden, 1, bias=False)` (`IsolatedLinear`
  gained nn.Linear's `bias` flag) — one scorer over ONE softmax, so its bias was a common shift (measured
  gradient over one K9 update |g| ≤ 2.3e-10 vs the weight's ~1e-3); −1 parameter. The POLICY pointer head's
  three scorers keep their biases: they are PER-FAMILY offsets inside one softmax, not common to every logit,
  so not dead. Not touched, reported: an attention KEY-projection bias is the same shift-invariant class
  (`PolicyStateQuery.k_proj.bias` under `--policy-readout trunk`; the key slice of every
  `nn.MultiheadAttention` packed `in_proj_bias`). Identity: NOT bitwise, by fp rounding of the removed shift alone
  (forward log π max |Δ| 2.4e-7; CONTROL with the bias re-attached BITWISE).

**F17 · Zero-init add-ons and the tier order.** Every delivery starts at exactly 0; T0 resolve → T1 reason → T2
decide → T3 deliver is an asserted invariant. *Bucket:* **KEEP** (makes each add-on's effect attributable;
P5-style determinism).

### E. Evidence quality and history

**F18 · Every measurement before 2026-10-04 read deflated Smogon priors.** The move prior was over the
unweighted raw count (×0.10–0.94 per species, F-X5-41) and the species-usage marginal unweighted (×0.19–×1.75,
F-X5-47). P2 is satisfied NOW. *Bucket:* **KEEP** (fixed); carries into F22.

**F19 · 32 event seats in the trunk.** History as typed event records with structural reference edges (the `r`
family) is principled (a memory without recurrence keeps the forensic replay exact). But the window size 32
takes the sequence from 29 to 61 tokens: ~4.4× the attention matrix, ~2.1× the per-token work.
- *Principle:* P7 (never sized). *Evidence:* **UNVERIFIED** how many seats carry weight.
- *Bucket:* **BEHAVIOUR CHANGE** (N = 16 arm; a CPU read of event-seat attention mass by age first, cheap).
  *Priority:* MEDIUM (also the cheapest lever on X5's cost overrun).

**F20 · `vf_coef` 0.5 was tuned for an MSE on a ±30 return and now multiplies a BCE.** The golden update's own
banner (CPU, here) read the value gradient at 5.83× the policy gradient at the shared trunk; the banner itself
warns this epoch-1 instrument is defective (UNDERSTANDING §4.3). Owned by
[`design_learner_recipe.md`](design_learner_recipe.md), not by this audit. *Bucket:* **BEHAVIOUR CHANGE** (recipe),
cross-referenced only.

**F21 · Documentation rot found in passing** (P3). **FIXED in the follow-up docs unit (2026-10-05):**
ARCHITECTURE §2.1 now states the 61-token trunk (13 base + 16 entity seats + 32 event seats), §2.3 lists the
event seats, §5's seat indices are the live ones (global 12, E3 13–16, E4 17–22, E5 23–28, events 29–60); the
`Gen3FeaturesExtractor` class docstring states the tier order (the operator before attention, no head concat);
learning note §6.9 replaces the deleted history seats with the event seats; `designs/CLAUDE.md` calls
`endstate/` always-current and lists every endstate doc. *Bucket:* **EXACT REFACTOR** (docs), done.

**F22 · The architecture's evidence base is a gen-3 checkpoint.** ARCHITECTURE §4.1 (operator blocks) and §5.4
(edge families) are `run_20260807_135637_gen3` at 9.6M steps, before the Baton Pass fix (08-23), the bot
setup-step enum fix (F-LF-1; its date not checked here), the prior denominators (10-04) and the Beat Up / fixed-damage pricing fix
(10-03); the gen-1/2 speed-edge numbers were on the speed-stat bug.
- *Bucket:* **KEEP (provisional)**, with a test: re-run `edge_ablation_audit` and the op-block probe on the X5
  blob seeds at 15M (CPU forwards), reporting dependence AND, per P9, whether each fact has another home.
  *Cost:* ~0.5 agent-day CPU. *Priority:* HIGH (every delete/keep decision above leans on it).

**F23 · The accidental simple run was discarded, not read.** `ai_v12_01_winprob_critic` (2026-09-06) ran 24.4M
steps with 29 architecture keys off: no edges, no move seats, no belief slots, no intent heads, no event window,
no item/spread belief (ledger 2026-09-06 INCIDENT). Its famine reading was "31 Elo ahead of rev-1, inside the 38
floor". The run directory still exists (`models/ai_v12_01_winprob_critic/`).
- *Principle:* S (a free control thrown away). *Evidence:* the reading is confounded (pin, recipe and the
  comparator "rev-1" differ; **UNVERIFIED** what rev-1 was).
- *Bucket:* **BEHAVIOUR CHANGE (a read)**: a mirrored head-to-head of its 24M snapshot against
  `ai_v12_02_winprob_critic` at matched steps, both win-prob critic; GPU minutes. *Priority:* HIGH.

**F24 · No deliberate simple baseline.** See §3.3 (C-MLP, C-ENT, C-NOCELL). *Bucket:* **BEHAVIOUR CHANGE**.
*Priority:* HIGH.

### F. Pieces re-justified from first principles (the owner's "we have it is not a reason")

| piece | first-principles case at our size | verdict |
|---|---|---|
| **F25 entity tokens + attention trunk** | a team is a set; a set-structured model never learns the 720 orderings; attention lets each Pokémon read the relevant opponent. Used by AlphaStar and Metamon | **KEEP**, conditional on F24's control |
| **F26 equivariance** | equal to "no slot meaning"; the pointer head and pools make it structural at no cost | **KEEP** |
| **F27 edge biases** | cheap (696 params) way to let exact physics steer attention; a known mechanism (Graphormer, ALiBi) | **KEEP** (F12 for the family count) |
| **F28 the damage operator** | exact gen-3 arithmetic the model would otherwise need ~10^5+ games to approximate; Foul Play shows exact physics + Smogon beliefs is strong. Its RISK is GIGO: speed-stat, Beat Up, fixed-damage and prior-denominator bugs were all found in or under it | **KEEP**, plus a standing differential test against the Rust sim per damage kind (Beat Up has one: `beatup_sim_parity_test.py`) |
| **F29 pointer head** | F16 | **KEEP** |
| **F30 opponent-intent heads (α/β)** | predicting the opponent's move is the core of simultaneous-move play; the heads are detached, so they cannot hurt the trunk | **KEEP**, pending X5's flat pointer (which replaces them in that arm) |
| **F31 belief heads** | a public-belief-state approximation (ReBeL's framing) with a Smogon prior; earns on the pool, nothing on ladder teams | **KEEP**, F9 for trunk shaping |
| **F32 the T0 species prior** | parameter-free Smogon mixture: cannot memorise the pool (P2 by construction) | **KEEP** |
| **F33 the win-prob critic** | with a win-indicator terminal and γ = 1 the return IS 1{win}, so V = P(win) exactly (P8); its open risk (an unbootstrapped MC target) is argued, not measured (UNDERSTANDING §4.1) | **KEEP** |
| **F34 the declared lifecycle and leak safety** | the forward reads `obs["observation"]` only; every training label is outside it (ARCHITECTURE §7); everything acquired at startup (P4) | **KEEP** |

---

## 6. What stays, and why (the KEEP list)

F3 (critic head size, until data is exhausted), F7c (gen-3 rule constants), F12 (edge mechanism), F13, F15,
F16, F17, F18 (fixed), F22 (provisional, with its re-read), F25–F34. The common thread: they PROVIDE facts the model cannot cheaply learn
(physics, Smogon priors, structure), or they make the system attributable and safe. What does not survive on
merit is the layer of hand-designed CONCLUSIONS stacked on top (F11), capacity left where the library put it (F1,
F2, F5), and the same fact delivered by several routes (F4, F10).

## 7. Counts and the proposed order

A finding with two parts is counted once per part: F6 (a exact, b behaviour), F7 (a the discarded lookups,
b the speed logistic, c the gen-3 rule constants), F16 (the scorer, b its dead bias). F29 is F16 and is not
counted twice. 33 findings, 37 bucket entries.

| bucket | entries | count |
|---|---|---|
| KEEP | F3, F7c, F12, F13, F15, F16, F17, F18, F22 (provisional), F25, F26, F27, F28, F30, F31, F32, F33, F34 | **18** |
| EXACT REFACTOR | F1, F6a, F7a, F8, F16b, F21 | **6** |
| BEHAVIOUR CHANGE | F2, F4, F5, F6b, F7b, F9, F10, F11, F14, F19, F20, F23, F24 | **13** |

Proposed order (each its own unit; nothing before X5 look 1):
1. **F22** CPU re-read on the blob seeds, and **F23**'s head-to-head read (GPU minutes, needs a lease).
2. **F24** controls C-ENT and C-NOCELL (C-MLP third), three seeds each.
3. **§4** L1 → L2 → L3.
4. **F1 + F6a + F16b + F7a** bundled at the X5-adoption version break (one `ARCH_SIGNATURE`, one K9 re-record),
   then **F8 / T21**.
5. **F2, F19, F9, F10, F14** as single-lever arms, as budget allows.

## 8. What this audit could not verify

- ~~Whether the inference tier computes the dead value tower per decision (F1).~~ **ANSWERED 2026-10-07**
  (MEASURED at the pre-part-2 commit, CPU, the dynamo FX graph of `engine.decide` + DCE): the compiled T2 graph
  KEPT `value_pre_norm` + `value_projection` (the extractor's `stash.features_out` side effect held the vf
  tensor) and dead-code-eliminated the 512→512→512 critic MLP; the eager backend computed the whole tower except
  `value_net` per decision. The version break's part 2 deleted the tower, and the `DecisionModule` no longer calls
  `forward_critic`.
- Metamon's exact model sizes and data counts; Graphormer's per-layer sharing detail (§3.1, F15).
- What "rev-1" was in the `ai_v12_01` famine reading, and whether that arm's checkpoints still load at HEAD
  (F23; a pinned load may be needed).
- Any end-of-run dependence for the seven per-action blocks, the event seats, or the belief trunk shaping
  (F9, F11, F19).
- Where X5's inference-tier cost lives on the GPU (asked in the dispatch; it needs a GPU counter read and the
  lease is held by another agent).
- bf16 readiness (asked in the dispatch; not examined: it needs a GPU numerics read; the hard maxima of F6 and
  the −1e9 addends of F15 are the first places to look).

---

## 9. F11 as built: the move-resolution INVENTORY (2026-10-06, written before the build)

The owner's criterion (2026-10-06): **FACTS** (hard-to-compute mechanics: what will actually happen if I press
this) are kept and consolidated; **JUDGMENTS** (opinions of the right play) are dropped. Every raw coordinate of
the seven blocks is classified below, plus the damage operator's status channel. **REDUNDANT** means the fact is
already delivered elsewhere. Every mechanic claim was verified against the vendored `deps/pokemon-showdown` with
gen-3 inheritance resolved (the gen-3 mod over gen 4 → 5 → … → base; the merged dex loaded through `dist/`).
The source lines are cited in `src/agents/model/move_resolution_rules.py`.

### 9.1 The 68 raw coordinates of the seven blocks

| block (raw width) | coordinate | class | why |
|---|---|---|---|
| `intent_move_cell` (7) | `is_status` | REDUNDANT | the move's identity is the E3 seat token; `p_land`/`known` ride the base cell |
| | `d_their_outspeed`, `e_burn`, `d_sched`, `e_slp`, `e_slp_free` | FACT ×5 | the physics of what a landed status does (speed ×0.25, Atk ×0.5, the 1/8 / 1/16 tick, the suspended hit, the verified sleep tables) |
| | `alpha_stay` | REDUNDANT | `1 − α_SWITCH` less the masked seat mass; `a_switch` rides once |
| `intent_threshold` (6) | `is_fp·(1 − p_fp_broken)` | FACT, rebuilt | Focus Punch's fail rule, but the shipped form is wrong at the source: a hit on our own Substitute does NOT break focus, and only a hit landing BEFORE the punch counts |
| | `is_sub·(1 − p_sub_broken)` | JUDGMENT | a hand 15 % ramp re-thresholded at the sub's 25 % HP: an opinion of whether the sub is worth it, not whether Substitute goes up |
| | `is_endure·p_ko` | JUDGMENT | "Endure pays only where I die" (Endure's own fact, its success odds, is kept) |
| | `is_dbond·p_ko` | FACT | the owner's Destiny Bond feature: P(the opponent KOs us this turn) = α × the operator's KO estimates. The shipped τ = current HP IS the KO event, so this one already had the owner's form |
| | `is_endeavor·(1 − p_ko)` | JUDGMENT | "I survive to act" as Endeavor's value; Endeavor's real rules (Ghost immunity) are in the family |
| | `p_ko` (context) | FACT | P(I am KO'd this turn) |
| `intent_conditional` (13) | Counter / Mirror Coat return operands | FACT ×2 | the 2× magnitude Counter / Mirror Coat would return |
| | `cat_match` | FACT, rebuilt | the category test, but the source rule is wider: the hit must LAND on the user (not its Substitute), come BEFORE the −5 move, Hidden Power always counts for Counter and never for Mirror Coat, Counter cannot hit a Ghost, Mirror Coat cannot hit a Dark type |
| | `p_flinch_useful` | FACT | P(our flinch lands on a mon that has not yet moved) |
| | `is_boom·(1 − p_blocked)` | FACT, rebuilt | the shipped blocker set counts ENDURE, which survives Explosion and does not block it |
| | `is_boom·α_SWITCH` | REDUNDANT | `a_switch` rides once; the seat token names the move |
| | Pursuit trigger, Pursuit bonus | FACT ×2 | the ×2 never-miss strike on the departing mon (port-verified) |
| | Protect: damage avoided, status avoided | FACT ×2 | what a successful Protect blocks |
| | Protect: the decay odds | FACT, moved | Protect's resolution IS its consecutive-use odds |
| | Magic Coat `e_reflect` | FACT, rebuilt | the shipped predicate (status AND target 'normal') bounces Taunt / Encore / Disable / Torment / Roar, which carry no `reflectable` flag; the source flag set is exact |
| | boom trade KO | FACT | P(the explosion KOs whoever it really hits) |
| `pair_outcome_cell` (14) | `low high crit ko_ramp acc is_phys` | FACT ×6 | the α-reduced incoming damage at our active |
| | `p_par … p_tox` | FACT ×6 | incoming status by identity, immunity-folded |
| | `neutralization` | JUDGMENT | a hand valuation (burn costs `base_atk/(atk+spa)`, paralysis a linear 25 % + speed blend) |
| | `tempo_cost` | JUDGMENT | the undo-path turn accounting (owner, explicitly) |
| `switch_branch_cell` (9) | `e_high_switch`, `e_pko_switch`, `e_mult_switch` | FACT ×3 | our move's outcome on the β-weighted arrival |
| | `wasted_ko` | JUDGMENT | "don't click the KO into the switch" (owner, explicitly; its two factors are delivered) |
| | `a_switch` | FACT | P(they switch) |
| | `p_spin_blocked` | FACT, moved | Rapid Spin's resolution (Normal vs a Ghost: no damage AND no clearing) |
| | `spin_value_lost` | JUDGMENT | a stake product (the hazards a blocked spin fails to clear) |
| | `protect_attack_mass`, `protect_blocked_mass` | FACT ×2 | P(they attack); P(our Protect blocks an attack) |
| `pair_outcome_switch` (15) | the 12 damage + status coordinates, per defender | FACT ×12 | as on the move cell, at every defender |
| | `neutralization`, `tempo_cost` | JUDGMENT ×2 | as above |
| | `spin_denied` | JUDGMENT (fact half kept) | its stake factor (their side's hazards) is a valuation; the fact `is_ghost(j)·α_spin` (their spin fails on our Ghost) is kept |
| `conditional_threat_cell` (4) | `e_pko_acc` | FACT, fixed | P(this mon dies). ⚠️ The shipped code multiplies `ko_ramp · acc`, but the op's `ko_ramp` is ALREADY `acc · P(KO \| hit)` (`DamageOperator._rolls`), so accuracy is counted TWICE: a 70 % Blizzard reads 0.49 where the truth is 0.70. Production carries it; the family computes it once |
| | `e_type_mult`, `margin_high`, `margin_crit` | FACT ×3 | the bulk-independent multiplier and the two margins (kept by the owner's ruling) |

**Counts (68 coordinates): FACT 55 · JUDGMENT 10 · REDUNDANT 3.** Of the 55 facts, 6 are rebuilt or moved because
the shipped spelling disagrees with the source rule (`is_fp`, `cat_match`, boom `p_blocked`, Protect odds, Magic
Coat, `p_spin_blocked`), and one (`e_pko_acc`) carries a production bug.

### 9.2 The damage operator's status channel (`_status_landing`, the base move cell's `p_land` / `known`)

FACT, already delivered: per our status move, P(it applies to their active) folding accuracy, the per-move type
immunity (Thunder Wave → Ground, Toxic / poison → Steel and Poison, Will-O-Wisp → Fire, Leech Seed → Grass), the
ability block (revealed exact, else the Smogon species prior), an existing major status, Sleep Clause (any LIVE
opposing sleeper that did not Rest), and their Substitute. Its documented gaps (Yawn, an already-seeded Leech Seed)
are below.

### 9.3 The MISSING facts (owner's examples first; every rule verified at the source)

| # | fact | status before the build |
|---|---|---|
| 1 | sleep on an already-asleep (or otherwise statused) mon fails | COVERED for the six-status moves (op); MISSING for **Yawn** (fails on a statused or already-drowsy target) |
| 2 | Sleep Clause | COVERED for direct sleep moves (op); MISSING for **Yawn**, whose delayed sleep the clause blocks |
| 3 | Substitute blocks status | COVERED for the six-status moves; MISSING for every other foe-targeting status move (Confuse Ray, Swagger, Mean Look, stat drops, Pain Split …) and for the **bypass set** (Taunt, Encore, Disable, Torment, Attract, Roar, Whirlwind, Psych Up, Haze …: the `bypasssub` flag) |
| 4 | Rapid Spin vs a Ghost | COVERED (`switch_branch`'s `p_spin_blocked`); consolidated |
| 5 | Counter / Mirror Coat conditions | PARTIAL (`cat_match`); MISSING: the hit must land on the user and not its Substitute, before the −5 move; the Hidden Power rule; Ghost / Dark immunity; a switch or a KO first means no return |
| 6 | Beat Up | MISSING: fails when no party member is alive and unstatused (the user counts only if unstatused); typeless, so it hits Ghosts |
| 7 | Focus Punch fails if hit | PARTIAL and wrong at the source (see 9.1) |
| 8 | type / ability immunities to status | PARTIAL; MISSING: **Glare → Ghost** (gen 3 Glare checks type immunity); **Safeguard** on their side; an UNREVEALED immunity ability on their active vs our damaging move (Levitate, Volt / Water Absorb, Flash Fire: the op's outgoing block is revealed-or-none). VERIFIED NON-FACTS in gen 3: Volt Absorb does NOT absorb Thunder Wave, Flash Fire vs Will-O-Wisp is subsumed by Fire's burn immunity, Levitate has no status role, Electric types are NOT immune to paralysis. Soundproof blocks the sound set but is BANNED in gen3ou (moot; built from the prior, which is 0) |
| 9 | Taunt / Encore / Disable effects | MISSING: ours fail on an already-taunted / encored / disabled target, and Encore / Disable on a target with no last move (Encore also on its `failencore` set); theirs restrict their own seats (a taunted opponent's status seat, an encored opponent's other seats cannot be clicked) and their Taunt landing FIRST stops our status move |
| 10 | Destiny Bond | COVERED (9.1); the family keeps the owner's form and adds the exact trigger probability (the bond must be up before the KO; a bond from last turn persists until we move) |
| 11 | Leech Seed on a seeded target fails | MISSING (op gap) |
| 12 | Spikes at 3 layers; Reflect / Light Screen / Safeguard / Mist already up; the same weather already up | MISSING |
| 13 | healing at full HP; Rest when asleep or with Insomnia / Vital Spirit; Refresh / Heal Bell with nothing to cure | MISSING |
| 14 | Substitute at ≤ 1/4 HP or behind a Substitute already; Belly Drum at ≤ 1/2 HP | MISSING |
| 15 | a stat-raising move with every raised stat at +6 | MISSING |
| 16 | Dream Eater / Nightmare need a sleeping target; Sleep Talk / Snore a sleeping user | MISSING |
| 17 | their Protect / Detect (or Substitute, Magic Coat, Taunt) landing before our move | PARTIAL (Protect mass only); MISSING as a per-move fact with the `protect` / `reflectable` flags and move ORDER |
| 18 | whether we act at all: KO'd first (α × KO × priority and speed), asleep (the obs wake odds), frozen (20 % thaw, `defrost` moves exempt), paralysis 25 %, confusion 50 %, infatuation 50 %, a flinch or a sleep / freeze / paralysis from a faster seat | MISSING |
| 19 | a switch does not resolve if their Pursuit KOs the departing mon | MISSING |

**MISSING: 16** (rows 1–3, 5–9, 11–19: none of them is a judgment; 21 with the five added below). Residuals the build does NOT model, named rather
than approximated: accuracy / evasion stages, Attract's gender rule, Fake Out's first-turn rule, a confusion that
ends this turn (its counter is not observed), Encore / Disable at 0 PP, the Disabled MOVE's identity on the
opponent, Damp, Wonder Guard, Truant, Endeavor's HP comparison.

**Added during the build (the coordinator's 2026-10-06 relay of the entity-coverage audit, `f32a9f4b`, its gaps §
rank 2):** the op's INCOMING status landing (`_incoming_status_lands`, the six `p_*` coordinates of `pair_in`)
ignores four more gen-3 rules, each verified at the source and in gen3ou's rule set (`config/formats.ts:4418-4422`
→ Standard + Freeze Clause Mod): **our Safeguard** (`data/moves.ts:15587-15615`), **incoming Sleep Clause**
(`data/rulesets.ts:1378-1402`: a live, non-Rest sleeper of ours blocks their sleep), **our Substitute**
(`gen4/moves.ts:1283-1320`: their status move fails, their secondary hits the sub) and **Freeze Clause**
(`data/rulesets.ts:1451-1471`: ANY frozen mon of ours, no HP check, blocks a freeze). The outgoing `p_land` ignores
**their Safeguard**. Yawn's delayed sleep is row 1–2 above. All five are MISSING facts (21 in all), corrected in
the family under the flag.

### 9.4 As built (2026-10-06, config v141, `gen3_move_resolution_v1`)

`--move-resolution {off,on}` (STRUCTURAL, cli, default `off`, config v141; `src/agents/model/move_resolution*.py`;
ARCHITECTURE §3.2 holds the statement of record). `off` builds nothing and sets no op seam. `on` builds ONE family —
a 38-wide move-cell block and an 18-wide switch-cell block, each through a zero-init `IsolatedLinear` — and the
policy retires the seven blocks after SB3's orthogonal re-init (every other parameter's initial bytes equal
production's, pinned by test), so `--arch production --move-resolution on --allow-nonproduction-arch` is the
one-lever screen arm (today's blocks vs the family).

* **The new fact**, per legal move: `p_resolve` = P(it resolves as stated: lands / not blocked / not immune / not a
  no-op) = `Σ_k α_k · A_k · L(m|k) + a_un · A_un · L_un + α_SWITCH · p_act · L_sw`, with `p_lands_stay`,
  `p_lands_switch`, `p_ko_first`, `p_act` beside it. Per legal SWITCH: `p_switch_resolves` (their Pursuit can KO the
  departing mon). Every rule is in `move_resolution_rules.py` with its source line.
* **Destiny Bond (owner): `dbond_p_ko` = P(the opponent KOs us this turn) = `Σ_k α_k · ko_k`, NO threshold**;
  `p_resolve` adds the exact trigger rule (the bond is up before the KO; a bond from last turn persists).
* **Kept and consolidated:** all 55 facts of §9.1 (the six rebuilt ones as resolution rules; `e_pko` with accuracy
  counted once). **Dropped:** the 10 judgments.
* **Rules the real-battle fuzz found** (each a verified gen-3 rule the inventory had missed; each now a unit test
  and a caught mutation): Protect / Detect / Endure FAIL when no action follows them (`onPrepareHit:
  !!this.queue.willAct()`), so they fail into a switch and when we move last; a heal / Rest at full HP resolves if
  a FASTER hit lands first (and Rest's "asleep" is `p_act`, since it must wake to move); a boost at +6 resolves
  after a faster Haze / stat drop or an Intimidate arrival; a departing Natural Cure sleeper lifts the switch
  branch's Sleep Clause; Wish fails while one is pending; their faster self-cure, thaw (1/5, or a defrost move) or wake (the obs wake odds) lifts "already statused"; a faster
  hit may break our Substitute (its HP is unobserved: counted as breaking — the one approximation); their faster
  status gives Refresh something to cure; a faster sleep move lets Sleep Talk work.
* **Proof of `off`:** the production extractor compiles (dynamo, CPU) to the SAME graph (one graph, sha256
  `eb892b00…`, 10,558 lines at `bef16d61`), the same state_dict bytes and the same outputs as its base at each rebase (`8288b9a2`, `11d27574`, `bef16d61`); the K9 learner golden
  (both entries) and the obs golden pass unchanged.
* **Real-battle check.** The development fuzz (poke-env players over the Rust `sim_bridge`, new battles every run,
  ~1,000 battles over the build) found the eleven rules above; it was retired before landing because it imported
  poke-env (the `poke_env_import_gate_test` allowlist is shrink-only). The committed check is
  `move_resolution_bridge_integration_test.py`: the banked M5 Lane S battles replayed through the RUST CORE (no
  poke-env), the family run on every played move's observation, the move resolved against the protocol. Full bank:
  31,942 played moves, 1,515 exact-zero claims met an executed move, ZERO resolved; the certain direction
  (`p_lands_stay = p_act = 1`, they stayed) 41 / 11,738 failed (0.35 %, the named residuals). Excluded from the
  gate, named: Sleep Talk / Snore (a called move failing its own `onTry` prints no line) and Heal Bell / Aromatherapy
  (`-cureteam` does not say whether anything was cured).
* **Under X5 `fixed_mass` (2026-10-07, `gen3_move_resolution_x5_v1`; the refusal LIFTED).** The same `move_facts` /
  `switch_facts` (no belief-mode branch) read the flat opponent pointer's re-expression (`FlatConsumerOps`, the
  object the seven blocks read there): α over the K move seats + OTHER_move, α_SWITCH, β over the six slots +
  OTHER_species. **OTHER is PRICED** (Decision record 2026-10-07): OTHER_move becomes one seat per gen-3 priority
  level (−6 … +5) carrying α_OTHER · P_tail(level), the level as its priority and its members' move tables
  conditioned on the level (`move_resolution.split_other_move`), so the ORDER rule is exact per level and every
  product of order and a move fact (their faster Protect / Magic Coat / Substitute / Taunt, a faster hit) is the
  exact tail expectation; its damage / status / c2 columns are the op's tail contraction, shared by the levels.
  OTHER_species is a 7th mon, the believed-slot read with the renormalised tail as its species distribution, its
  `out_cells` column the OTHER-mode D1 pass; a hidden slot reads its hypothesis species (presence carried by β, not
  multiplied in again). The one remaining approximation: the tail-averaged damage grid. `off` byte-identical in BOTH
  belief modes (the dynamo graph, the state_dict bytes and the outputs of the production extractor and its
  fixed_mass arm equal the base's, `f0d673fd` and again after the rebase onto `ccba59f0`; the K9 learner golden, both entries). Real battles, fixed_mass read: the full
  Lane S bank, 31,942 played moves, 1,485 exact-zero claims, 0 contradicted; the certain direction 6 / 2,695 failed.
  **Compiled on CUDA (2026-10-07, F-MR-1, `gen3_move_resolution_traceable_v1`).** The first real CUDA launch of
  `fixed_mass` × `on` failed at its startup R1 compile. `split_other_move` returned `NamedTuple._replace`, which
  lives in `collections`, and dynamo skips that module, so `fullgraph=True` refused the call. The CPU `--debug`
  smoke never compiles the learner and so could not show it. The fix rebuilds the ops through the class
  constructor (`_replaced`), with the same values. `move_resolution_x5_test.py::test_split_other_move_compiles_as_one_graph`
  fails on revert. The `blob` × `on` launch passed T2, R1 and the update-10 canary.

**What it does NOT do (findings for the owner, not fixed here because each would move production):** the shipped
`conditional_threat` double-counts accuracy in `e_pko_acc`; the shipped `intent_conditional` prices Protect by its
stall odds alone (it ignores the moving-last rule) and counts ENDURE as an Explosion blocker; its Magic Coat
predicate bounces moves with no `reflectable` flag; the op's incoming and outgoing status landing ignore the four
coordinator-relayed rules above; and — found by the fuzz, the largest — **the op reads an opponent's ability as
REVEALED whenever its id is non-zero**, but the observation writes an UNREVEALED opponent's most likely ability
there with `known = 0` (`observation/abilities.py`), so `_status_landing` asserts Immunity on every unrevealed
Snorlax (Toxic "never lands" where the prior says 0.14) and the outgoing / incoming kernels apply the top-1
ability's immunity multiplier as certain (`damage_op_blocks.py` lines 231/235, 402/414, 547/554, 644-648, 727, 1265-1278). The family reads the `known` flag. **FIXED on the production path 2026-10-07** (`gen3_op_ability_status_gigo_v1`, see the
Decision record): the op's ability reads, the five status rules, `e_pko_acc` and `intent_conditional`'s three rules;
the slot-tied `out_gain` below is NOT part of that fix. And (coordinator, from `2d29c4c0`): the op's learned `out_gain` was one
scalar per block channel, i.e. per REQUEST SLOT on the per-move outgoing channels, so the same move was scaled by
where it is listed (trained X5 arms: KO gain 1.365 on slot 0 vs 1.146 on slot 3). The family reads every op value
PRE-gain (`out_cells`, `pair_in`, and P(outspeed) from the pre-gain block) and its own projection is shared across
slots; on the default path a slot-tied gain is retrain-class — a note for the version break. Each is a ~one-line fix on the production path, a behaviour change that moves
the production goldens (the K9 learner golden, the compile canary's rows): an owner decision. **BUILT at the version
break (2026-10-07, parts 4 and 5 of `gen3_x5_version_break_v1`, Decision record):** the gain is ONE scalar per
(block region, channel) shared across request slots and move seats (production 138 → 99 gain parameters), and every
consumer that reads an op value as physics reads it PRE-gain — `intent_conditional` (our moves' high roll, P(first),
the flinch chance) included, in both speed modes, through the op's live `last_raw_tensors` view (the family reads the
same view).

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-05 | Audit stance | Re-justify every major piece from first principles at our size; "we have it" is not evidence (owner's scope extension, same day) | Auditing only for internal consistency | owner 2026-10-05 |
| 2026-10-05 | Buckets | KEEP / EXACT REFACTOR (bundled at the X5-adoption version break) / BEHAVIOUR CHANGE (one lever, own test); the owner reviews before any build | Fixing in passing | the dispatch brief |
| 2026-10-05 | Lead hypothesis | Static per-Pokémon tokens with board context mixed by attention (option A), tested L1 → L2 (pre-LN) → L3 (third layer), each one lever, equivalence for L1, superiority at matched wall time for L3 | Option B (every per-mon state its own token): over-reach at 128 dims; bundling depth with the static change | §4; owner 2026-10-05 |
| 2026-10-05 | Controls | Train C-ENT, C-NOCELL, C-MLP before adding architecture; read `ai_v12_01` as a free control | Continuing to judge pieces by within-model ablation only | §3.3, F23, F24 |
| 2026-10-05 | Dead value tower | Delete at the version break (EXACT) | Keep "for the shaped critic" (untrainable since D4) | F1, MEASURED census |
| 2026-10-06 | **Owner review of the below-the-line findings** | **IN:** F7a + F16b (EXACT, bundled at the version break); F10 (BEHAVIOUR: delete `value_threat_inject`, read critic discrimination); F14 (BEHAVIOUR: pool the four moves symmetrically, FOLDED into the §4 L1 static-token rebuild by the owner's delegation, split if the combined screen disappoints). **DECLINED:** F23 ("if we wanted it, we would just train another one": the F24 toy controls cover it). **F20: KEEP `vf_coef` 0.5 on evidence**: the critic ladder bracketed it (1.5 strips opponent-class information; 0.25 restores nothing; non-monotone), caveated as old-era and critic-row, not strength; re-check on the deep run's `grad/value_policy_logratio` telemetry. **PENDING the owner:** F7b (the outspeed probability from the speed belief instead of `_DMG_SPEED_SCALE`) and F11 (the strategy-heuristic family ablation). Already IN from the 2026-10-05/06 plan: F1, F6a, F2, F4 (L1), F5 (L2/L3 conditional), F24. | The audit's own order | owner 2026-10-06 |
| 2026-10-06 | **Owner: F7b IN; F11 replaced by a MOVE-RESOLUTION family** | **F7b:** P(outspeed) from the speed belief + exact gen-3 rules (tie = coin flip, paralysis, Choice Band, Quick Claw), replacing `_DMG_SPEED_SCALE`. **F11, the owner's criterion: FACTS (hard-to-compute mechanics) vs JUDGMENTS (opinions of the right play).** Keep and consolidate the facts into ONE family: per legal action, P(it resolves as stated): lands / not blocked / not immune / not a no-op (spin vs Ghost; sleep on an already-asleep mon and Sleep Clause; Substitute vs status; Counter / Mirror Coat conditions; Beat Up; Focus Punch fails if hit), computed by the exact rules and intent-weighted where it depends on the opponent. DROP `tempo_cost`, `wasted_ko` (its fact is covered by the pair-outcome cells) and every hand threshold. **Destiny Bond specifically (owner): no fixed threshold; its feature is P(the opponent KOs us this turn)** from the intent head × the operator's KO estimates. The build starts with an inventory of what the operator's status channel already covers; tested as ONE lever (today's blocks vs the resolution family). | Dropping all seven blocks (would discard true mechanics the network can't cheaply learn); keeping the judgments | owner 2026-10-06 |
| 2026-10-06 | **Owner: the STATIC-TOKEN rebuild (§4 L1, with F4 + F14) is GO, spec widened** | "I want item, ability, moves and stats all static if possible; let the attention and the damage op do the heavy lifting." The per-mon token = a STATIC identity (species + SET: item, ability, the four moves pooled as a set (F14), nature / EVs / IVs → actual stats) from a per-(species, set) table; for the opponent the same table gives the prior, the belief / hypothesis fills it, reveals override it. Per-mon DYNAMIC state (HP, status, boosts, volatiles, the item-consumed / changed flag) is a small separate per-mon input; board context (clock, weather, hazards, screens, the opponent active) leaves the per-mon encoder and reaches tokens by attention and the physics edges. Built on the X5 path (adoption pre-committed), screened at equal budget vs today. | Static species only (the audit's narrower L1) | owner 2026-10-06 |
| 2026-10-06 | **F2 BUILT as a screenable flag** (`--policy-readout {tower,trunk}`, config v138, OFF in production) | **`trunk`:** the flat policy tower (the extractor's `pre_proj_norm` / `projection` + SB3's `mlp_extractor.policy_net`, 1,130,802 parameters) is RETIRED; the pointer head's context is `PolicyStateQuery` — ONE learned query, 4 heads, over every refined trunk token (our 6, their 6, the global token, the entity and event seats) plus `HiddenOppBeliefPool`'s K outputs, under the trunk's own key mask (its per-key log π under `fixed_mass`), then a LayerNorm → 128; the F16 scorer unchanged in form, widened to 128 (`TRUNK_POINTER_HIDDEN`: no 64-dim squeeze of the context). Init isolation by the X5 precedent: the query from a private seed out of `IsolatedLinear`s, the tower retired only after SB3's orthogonal re-init, so every surviving non-pointer parameter starts from `tower`'s bytes (tested). `tower` byte-identical: the K9 golden's two entries pass unchanged and the compiled R1 / T2 FX graphs equal the base commit's. Parameters 3,065,882 → 1,991,528. The value path is untouched (F1 is its own unit). GPU checks (compile parity, T2 throughput, memory) DEFERRED; built, not trained. | The audit's narrower test (`net_arch pi = []`, keeping the 1177→512 projection as the context): it keeps the flat concat the owner chose to drop; the existing CLS-pool outputs as the context (a Linear over their concat is itself a flat projection); several queries (one lever first); keeping `POINTER_HIDDEN` 64 (re-creates the bottleneck) | F2; owner 2026-10-06; `policy_readout_test.py` |
| 2026-10-06 | **Static-token rebuild: design + build stage 1** | `design_static_tokens.md`: token = S (species + set + actual stats, moves summed as a set) + D (the mon's own state), ADDED, no board fact; opponent stats = the Smogon usage-weighted prior (the learned spread belief would be circular); X5 hypothesis tokens = the dex table encoded once and gathered; the coverage audit's board layout + per-mon op content are part of the SAME arm (stage 2). Stage 1 built behind `--token-encoding static` (config v139, default `legacy`, byte-identical) | a second token per mon; FiLM join; screening stage 1 alone (confounded: coverage audit §4) | `design_static_tokens.md`; `design_entity_coverage_audit.md` |
| 2026-10-06 | **F11 BUILT behind `--move-resolution` (OFF in production)** | The inventory (§9: 68 coordinates → FACT 55 · JUDGMENT 10 · REDUNDANT 3; 21 MISSING facts incl. the coordinator's five) written first; the family built as ONE flag that retires the seven blocks on a built policy (§9.4): per legal action P(it resolves as stated) by the exact, source-verified gen-3 rules, intent-weighted; Destiny Bond = P(the opponent KOs us); the judgments dropped. `off` proven byte-identical (compiled graph, state_dict, outputs, K9 golden, obs golden). The coordinator-relayed op corrections are applied INSIDE the family only; the production op is untouched | Applying the incoming / outgoing status corrections to the production op now (moves the production goldens: an owner decision, reported); X5 `fixed_mass` support (refused at build: the flat pointer's (K+1)-th seat is not wired) | this build; ledger 2026-10-06 |
| 2026-10-07 | **The §9.4 production-op findings FIXED on the default path (GIGO, owner rule)** | `gen3_op_ability_status_gigo_v1`: the op reads an opponent's ability by its `known` flag (the species' Smogon marginal when unrevealed) on every kernel, through ONE view; status landing folds their / our Safeguard, incoming Sleep Clause, Freeze Clause, our Substitute and Yawn's delayed sleep (ONE rule per direction, the incoming mask shared with the family); `conditional_threat` counts accuracy once; `intent_conditional`'s Protect needs an action to follow, Endure no longer blocks Explosion, Magic Coat bounces the `reflectable` set. A TRAINING-INPUT BOUNDARY (no ARCH_SIGNATURE bump, OP_SEMANTICS bumped); learner golden + buffers and the h2h digests re-recorded. Real data: 12 of the old op's 199 certain-zero status claims landed (Snorlax Toxic), 0 of the new op's 181. | Leaving them family-only (§9.4's original stance — the owner's GIGO rule overrides it); bumping ARCH_SIGNATURE (strands every checkpoint); widening to Glare → Ghost / Leech-Seed-on-seeded and the slot-tied `out_gain` (out of the brief, reported as findings) | ledger 2026-10-07; `measurements/op_gigo_2026-10-07/` |
| 2026-10-07 | **F10 BUILT as a screenable value** (`--value-threat-inject off`, config v142, ON in production) | The existing flag is the lever (no new flag). OFF with the op built constructs `ValueThreatInject` NOT LIVE (`CLSPool.value_threat_live`) and `Gen3DualHeadMaskablePolicy._build` retires it after SB3's orthogonal re-init (`ExtractorApi.retire_value_threat_inject`), so OFF differs from production by the route alone: −1,792 parameters (3,065,882 → 3,064,090), every other initial byte equal (tested, also under static / trunk / move_resolution / fixed_mass). ON byte-identical: the K9 golden's two entries pass, the production extractor's CPU dynamo FX graph (one graph, 11,317 lines, sha256 `cf4a45a0…`), its state_dict and its outputs equal the base commit's, and the production learner's state_dict sha is unchanged. The op stays on `hard_max` under OFF. The screen's meter is `main.ops.critic_read`'s `gate.resolution.{bot,pool}` (primary), `identity.resolution`, `gate.skill.{bot,pool}`, `identity.turn_contrast` (secondary); not registered, not run. | A new `--value-threat-route` flag (the existing one already selects it; a second name for one lever); OFF building nothing and accepting the init shift (a 185-tensor confound in a one-lever screen); moving the op to `belief_mean` under OFF (the reduced rows feed only this route; R1 is parameter-free, so it would cost compute for no byte) | F10; owner 2026-10-06; `value_threat_inject_off_test.py` |
| 2026-10-07 | **F7b BUILT behind `--speed-physics` (OFF in production) — as built** | `gen3_speed_physics_v1` (config v143): P(we act first) from PHYSICS at every op site that prices it (incoming / outgoing `p_outspeed`, the outgoing attacker matrix, the pair outcome's paralysis severity, C1 `d_outspeed`, C2 `d_their_outspeed`, the V edge). ONE rule, `move_order.py`: the priority bracket (`p_seat_first`, moved there from `move_resolution_rules`, so the bracket and the within-bracket rule cannot fork) + the speed BELIEF's integral on the integer lattice (believed speed = the spread belief's, else the Smogon prior mean; spread = the Smogon prior's per-species σ — the discarded F7a lookups, now read), a tie = a coin flip, our EXACT stat / stage-floor / paralysis arithmetic, Choice Band no speed effect, all verified in `deps/pokemon-showdown` gen 3 → gen 4 → base. No free constant, no parameters. **Quick Claw** (one shared 1-in-5 roll per turn) is implemented and tested but FORMAT-GATED OFF: **BANNED in Gen 3 OU** (owner 2026-10-07; Showdown master `config/formats.ts` ``[Gen 3] OU`` banlist carries `'Quick Claw'`, read 2026-10-07; the vendored simulator `e0551883` predates the ban) — one read, `move_order.quick_claw_live`, for the format-spec build to replace. Exposure, reported not changed: the Smogon item prior in `data/` still gives Quick Claw mass to 182 species (14 at ≥ 0.05, Hypno 0.51), the item belief starts from it; ZERO teams in `data/teams/{sample,specialist,others}` and the archetype file carry it. Where the move-resolution family (pre-gain) and `intent_conditional` (post-gain) read P(first) differently, `on` makes both read the pre-gain value; `off` is untouched. [SUPERSEDED 2026-10-07 by the version break's part 5 (row "slot-tied out_gain and the pre-gain read"): BOTH modes now read the pre-gain value, so the `on`-only special case is deleted.] `off` byte-identical (dynamo graph sha `3b06da42…`, state_dict, outputs). Real battles (Lane S, 580 battles, 23,598 equal-priority rows, cold-start): Brier 0.0406 vs 0.0425, log loss 0.1367 vs 0.1341, ECE 0.0160 vs 0.0157 — NOT clearly better calibrated; 0 of 3,428 bracket claims contradicted | A Gaussian with a learned spread head (a new parameter, a new label); the exact Smogon spread mixture (it cannot carry the learned belief); Swift Swim / Chlorophyll / Macho Brace (outside the owner's rule list; named residuals); fixing `intent_conditional`'s post-gain read on the default path (moves production — an owner decision; DONE at the version break's part 5) | owner 2026-10-06/07; ledger 2026-10-07 |
| 2026-10-07 | **F7b: the Gaussian speed belief REPLACED by the discrete Smogon spreads mixture (orchestrator brief)** | `gen3_speed_mixture_v1`, under `--speed-physics on` only (`off` byte-identical, no config / ARCH bump: no parameter, a non-persistent table registered only under `on`): their speed = `SPEED_MIX` (`belief_tables.build_species_speed_mix`, from `gen3_data.priors.all_spreads` — EVERY chaos spread, because the top-25 cut drops exactly the lumpy tail, e.g. Blissey's max-Speed sets), each support point through the exact stage / paralysis arithmetic; P = Σ w·(1[ours > f] + ½·1[ours = f]); the X5 OTHER slot reads the tail's mixture. Lane S (580 battles, 23,598 rows): log loss 0.1308 (off 0.1341, Gaussian 0.1367), 0 certain-and-wrong (Gaussian 1), Brier 0.0440 (0.0425, 0.0406), ECE 0.0184 (0.0157, 0.0160) — fixes the tail, NOT better calibrated overall; the trained arm loses the learned belief's information (Brier 0.0440 vs the Gaussian's 0.0368) | Keeping the Gaussian (a 15-σ certain miss); the top-25 spreads (the miss stays a certain miss); conditioning the mixture on the learned belief (no likelihood to condition with — the open lever, with battle evidence) | brief 2026-10-07; `designs/research_state/measurements/speed_physics_f7b_2026-10-07/`; ledger 2026-10-07 |
| 2026-10-07 | **The move-resolution family under X5 `fixed_mass` (the refusal LIFTED; `gen3_move_resolution_x5_v1`, no config / ARCH_SIGNATURE bump: no weight changes shape, `off` byte-identical in both belief modes)** — and **how OTHER enters P(resolve): PRICED, never excluded** | The same rules read the flat pointer's re-expression (α over K seats + OTHER_move, β over six slots + OTHER_species). **OTHER_move = one seat per gen-3 PRIORITY level**, each carrying α_OTHER · P_tail(level), the level as its priority and its members' move tables conditioned on the level; its damage / status / c2 columns the op's tail contraction. **OTHER_species = a 7th mon**, read like a hidden slot with the renormalised tail as its species distribution (its `out_cells` column the OTHER-mode D1 pass). Why: X5's own ruling (§3.7 / §9 M3 (c), "OTHER as a switch target or a move outcome must be priced, not zero"); OTHER is not small (cold start, the compile-parity bank: α_OTHER up to 0.18, β_OTHER up to 0.63); the seven blocks the family retires already price OTHER under fixed_mass, so excluding it would deliver LESS than the blocks it replaces; the per-level split makes the move ORDER and every order × move-fact product (a faster Protect / Magic Coat / Substitute / Taunt / hit) the EXACT tail expectation with the one unchanged order rule (a one-member tail reads as the seat naming that move, to fp32 summation order — max |Δ| 1.5e-8 — tested). The one remaining approximation is the tail-averaged damage grid (a member's KO / hit odds are not split by priority) — X5's named Jensen cost of option (c) | **Excluded with its mass reported as a feature:** the family's "unnamed seat" branch asserts no hit and no block, so the excluded mass would CLAIM "OTHER never KOs us first, never Protects" — a claim, not an absence (the CLAUDE.md fallback rule) — and a new coordinate changes the block width (a weight-shape change) in both modes. **One tail-averaged OTHER seat** at priority 0 (`intent_conditional`'s residual) or at the tail's mean priority: wrong order (a Protect + Earthquake tail would act first never / always), and E[first]·E[Protect] halves a tail Protect's block. **Expanding OTHER into its M members** (exact): the per-member damage grid is not stashed, and [B,4,K+M] is ~50× the seat axis. **A per-level damage split** (closes the remaining gap): an op change on the production path | this build; `move_resolution_x5_test.py`; ledger 2026-10-07 |
| 2026-10-07 | The move-resolution × `fixed_mass` compile failure (F-MR-1) | `split_other_move` builds its result through `MoveResolutionOps`'s constructor, not `_replace` (same values) | `@torch._dynamo.dont_skip_tracing` on a stdlib function (it opts dynamo into tracing `collections`, which may break the graph elsewhere); leaving the family uncompiled (the learner region is ONE fullgraph region by K6) | §9.4; the first CUDA launch of the arm (lease "static fix gpu") |
| 2026-10-07 | **EXACT bundle built at the version break** (part 2 of `gen3_x5_version_break_v1`, config v144 — no further bump) | **F1** (the dead SB3 value tower deleted, 592,129 parameters; the extractor's value half is `value_pooled`; the policy builds its own actor-only stack, raising `action_net` / `value_net` stubs, `critic` other than `winprob` refused), **F16b** (the flat opponent pointer's shared scorer bias-free, −1), **F6a** (`index_max.max_by_index` at every gradient-path max; first-maximum tie convention), **F7a** (the off-path speed-sigma lookups and `_p_outspeed`'s ignored argument deleted), and the blob LEFTOVERS (`BeliefSlots` / `AlphaIntentHead` / `BetaSwitchHead` and their construct-and-retire deleted — no RNG-draw preservation any more; the never-written α / β stashes deleted and their readers moved to the flat pointer; `--beta-setvalued-coef` deleted; retired modules leave a plain `None` via `extractor_api.drop_child`, closing the strict-load hole). Production learner 3,111,176 → 2,519,046 parameters (MEASURED, CPU, the K9 learner). The init bytes move (no construct-and-discard), so the K9 learner golden and the compile goldens are re-recorded ONCE at the end of the break. | Left, LISTED: the attention key-projection biases (same shift-invariant class as F16b; `PolicyStateQuery.k_proj.bias`, MHA `in_proj_bias`'s key slice); the move belief's Hungarian `elif` in `belief_bank` (REACHABLE with the belief family off); `main.train.config.inherit_derived_enable_coefs` / `UnrecordedEnableCoef` (unreachable behind the v144 floor, not small); the v122–v143 migration branches. F6b and F7b remain their own levers. | Weight-mapping identity vs the `26131c0c` reference (K9 forward + one K9 update, CPU, 1 thread): F1 + leftovers + F7a BITWISE; with F16b NOT bitwise by the removed softmax shift's rounding alone (log π max |Δ| 2.4e-7, entropy 4.8e-7, post-update 6.0e-8, losses at the 8th significant digit; values and masks EQUAL), CONTROL with the bias re-attached BITWISE; F6a moved no byte. `designs/research_state/measurements/version_break_identity_2026-10-07/`; `x5_version_break_part2_test.py` |
| 2026-10-07 | **Slot-tied `out_gain` and the pre-gain read: built at the version break** (parts 4 and 5 of `gen3_x5_version_break_v1`, config v144 — no further bump) | **Part 4:** the op's learned gain is ONE scalar per (block region, channel), shared across REQUEST SLOTS and move seats (`damage_op_layout.out_gain_channel_keys`; the distinct gains expanded by a fixed non-persistent one-hot, an exact value and a fixed-order backward); each tied channel keeps its per-slot init (asserted at build), so the init forward is BITWISE unchanged (unperturbed production extractor, 64 compile-parity rows, CPU). Production `out_gain` 138 → 99, the learner 2,519,046 → 2,519,007 parameters. **Part 5:** ONE rule — a consumer that reads an op value as physics reads it PRE-gain, through the op's LIVE `last_raw_tensors` view: `intent_conditional`'s high roll, P(first) and flinch chance in both speed modes (the F7b `on`-only special case deleted) and the move-resolution family's P(first) (it read the detached `last_raw_block`). The remaining post-gain readers are projections (pointer cells, `prefuse_proj`, `value_entity_pool`'s `op_proj`) and stay. With it, F7a's last dead piece (the OTHER roster's `spe_std`) is deleted. | Tying the MON-axis replicates too (the incoming rows per our team slot 72 → 12, the CB tail 12 → 2, the render matrices' per-mon cells) — positional replicates of one quantity, but outside the finding (request slots); REPORTED for the owner. Reading the detached `last_raw_block` in `intent_conditional` (it would cut the spread belief's gradient through P(first) and the high roll under `off`, MEASURED live). Leaving `intent_conditional`'s high roll / flinch post-gain (the same class as P(first): a value the cell multiplies) | `x5_version_break_part45_test.py` (each test fails on the reverted source); CHANGELOG v144 Parts 4 / 5 |
