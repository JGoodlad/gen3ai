# Architecture audit: what is unprincipled, over-built or under-built (2026-10-05)

**Status: AUDIT, read-only. Nothing in it is built or decided.** Every finding carries a bucket the owner
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
   REFACTOR, HIGH.)
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
   BEHAVIOUR CHANGE for the reduction, MEDIUM.)
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
| critic | P(win) | P(win), 17k-param head; a dead 592k tower beside it (F1) | right idea; clean up |
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

- Whether the inference tier computes the dead value tower per decision (F1).
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

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-05 | Audit stance | Re-justify every major piece from first principles at our size; "we have it" is not evidence (owner's scope extension, same day) | Auditing only for internal consistency | owner 2026-10-05 |
| 2026-10-05 | Buckets | KEEP / EXACT REFACTOR (bundled at the X5-adoption version break) / BEHAVIOUR CHANGE (one lever, own test); the owner reviews before any build | Fixing in passing | the dispatch brief |
| 2026-10-05 | Lead hypothesis | Static per-Pokémon tokens with board context mixed by attention (option A), tested L1 → L2 (pre-LN) → L3 (third layer), each one lever, equivalence for L1, superiority at matched wall time for L3 | Option B (every per-mon state its own token): over-reach at 128 dims; bundling depth with the static change | §4; owner 2026-10-05 |
| 2026-10-05 | Controls | Train C-ENT, C-NOCELL, C-MLP before adding architecture; read `ai_v12_01` as a free control | Continuing to judge pieces by within-model ablation only | §3.3, F23, F24 |
| 2026-10-05 | Dead value tower | Delete at the version break (EXACT) | Keep "for the shaped critic" (untrainable since D4) | F1, MEASURED census |
| 2026-10-06 | **Owner review of the below-the-line findings** | **IN:** F7a + F16b (EXACT, bundled at the version break); F10 (BEHAVIOUR: delete `value_threat_inject`, read critic discrimination); F14 (BEHAVIOUR: pool the four moves symmetrically, FOLDED into the §4 L1 static-token rebuild by the owner's delegation, split if the combined screen disappoints). **DECLINED:** F23 ("if we wanted it, we would just train another one": the F24 toy controls cover it). **F20: KEEP `vf_coef` 0.5 on evidence**: the critic ladder bracketed it (1.5 strips opponent-class information; 0.25 restores nothing; non-monotone), caveated as old-era and critic-row, not strength; re-check on the deep run's `grad/value_policy_logratio` telemetry. **PENDING the owner:** F7b (the outspeed probability from the speed belief instead of `_DMG_SPEED_SCALE`) and F11 (the strategy-heuristic family ablation). Already IN from the 2026-10-05/06 plan: F1, F6a, F2, F4 (L1), F5 (L2/L3 conditional), F24. | The audit's own order | owner 2026-10-06 |
