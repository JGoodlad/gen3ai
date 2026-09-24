# PRE-REGISTRATION — BELIEF-HEAD CALIBRATION OFF THE POOL (the MEMORISATION read)

**Committed BEFORE any registered number in this directory exists.** Nothing below has been read
off the model on real battles. The plumbing smoke that follows this commit prints shapes and
assertion counts only, never a metric.

## 0. The question

The production belief stack predicts the opponent's hidden team as marginals, supervised by the
TRUE team of opponents drawn from our 719-team pool (`data/teams/`) or from our own teams. So the
heads may have MEMORISED pool teams ("these two revealed ⇒ team #412"). That bias is sanctioned for
the policy. But MoveBelief's posterior is REINJECTED into the policy's opponent tokens, so on the
ladder a confidently wrong guess reaches decisions. **How much memorisation is there, and how
confidently wrong are the heads on teams from outside the pool?**

This is a POLICY question. Search determinisation is deferred.

## 1. Structural facts read from the pinned code (not measurements)

Read from `6eb9c776`'s `extractor_forward.py`, `belief_heads.py` and `t0_species.py`. They decide
what each head CAN memorise and what reaches a decision:

| Head | Tier / input | What it can condition on | Reaches pi/vf? |
|---|---|---|---|
| `BeliefHead` species (+moves) | T2, post-transformer opp tokens; `species_prior_fusion` = Smogon naive-Bayes prior ⊕ learned delta | the WHOLE board, so it can memorise whole pool TEAMS | **No.** Training-only side readout (§7). It reaches decisions only through the trunk it shapes (`belief_grad_mode` `shaping`) |
| `T0SpeciesPrior` | T0; parameter-free Smogon naive Bayes over revealed species | revealed species only, no learned part | **Yes**, into the damage op and edge cells. Cannot memorise the pool |
| `MoveBelief` (typed posterior `last_move_belief_logits`) | T0, PRE-transformer role tokens; Smogon prior ⊕ delta, revealed moves pinned | REVEALED slot: that mon's own features (species, revealed moves, item/ability remnants) + a broadcast global context. HIDDEN slot: the role token is `BeliefSlots.unknown_slot_emb[i]`, a CONSTANT, and the prior row is the flat floor (species num 0) | **Yes**: reinjected into all six opp tokens (`move_belief_mode both`), and read by the op, seats and edge cells |
| `ItemBelief`, `HPTypeBelief`, `SpreadBelief` (nature/EV) | T0, same pre-transformer tokens; Smogon prior ⊕ delta | per-mon SET memorisation only (e.g. "pool Swampert with Earthquake shown runs X") | **Yes** (item via P(CB) in the op; HP type via the typed composition and its reinject; spread via the op and its reinject) |

So the pre-registered expectations differ by head:
* **Species:** whole-TEAM memorisation is possible, but the fed-forward species belief is the
  pure Smogon prior. A large species memorisation gap is a finding about a SIDE READOUT.
* **Moves, hidden slots:** the posterior is a state-independent CONSTANT per slot position (I will
  VERIFY this, §5.4). It can memorise only the pool's MARGINAL over hidden movesets.
* **Moves (revealed slots), item, HP type, spread:** per-species SET memorisation. This is the
  channel that reaches decisions, so revealed-slot moves are a co-headline (§3).

## 2. The measurement

**Model.** `models/ai_v13_22_popr1_loop/final_model.zip` (num_timesteps 103,219,200; sha256
prefix `88fc961a819abe5f`), config v119 / `gen3_critic_route_wave_v1`, pin `6eb9c776`
(`pin_history` has ONE row). It is loaded by ITS OWN code: a detached worktree at `6eb9c776`, run
with `PYTHONPATH=<pin>/src` and cwd = the pin worktree (`data/` is byte-identical to main's:
`git diff 6eb9c776 main -- data/` is empty), and that tree's own Rust build
(`POKESIM_SIM_BRIDGE_BIN=<pin>/src/rust_sim/target/release/sim_bridge`). The load goes through
`main.capacity.load_policy` → `load_model_snapshot` → `check_compatible`. It was verified to load
with no ArchDrift, and all six belief heads are present with fusion on.

**Battles.** In-process Rust bridge (`run_local_battles(impl="rust")`), no server, CPU,
`nice 15`, `torch.set_num_threads(1)`, at most 4 worker processes, each playing its battles
sequentially (concurrency 1). Battle `i` of every arm uses the SAME trainee team, the SAME sim
seed and the SAME side assignment, so arms are PAIRED by `i`, and only the opponent's team differs.
* **Trainee (p1):** the model, GREEDY (argmax).
* **Opponent policy (p2), identical in all three arms: the model itself, GREEDY.** Why: self-play
  and frozen selves are the bulk of the training opponent mix, so this is the reveal/behaviour
  regime the labels were trained under. It is the only pilot that plays ANY legal team competently
  (a heuristic bot's reveal pattern depends on the team in ways that would confound the arm
  contrast). And greedy is deterministic, so a battle is reproducible from `(teams, seed)`.
* **Our side:** a pool team, drawn uniformly from the 719 by a seeded RNG, the same for index `i`
  in every arm.
* **Seeds:** team draws from `numpy.random.default_rng(20260924)`. The sim seed for battle `i` is a
  fixed function of `i` alone.

**The three opponent-team distributions** (the variable):
* **(a) POOL.** Uniform draws from the 719 `TeamLoader().get_all_teams()` teams, the training
  opponent distribution (`TeamSource(kind="pool")`).
* **(b) LADDER.** Metamon `hl_05_26` / `gen3ou` (Metamon teams revision v5), downloaded with
  Metamon's own `download_teams` into `~/gen3ai_archive/metamon_cache_2026-09-24/` (outside the
  repo), about 22.9k teams. **Filter** (reported as a share, with the top rejection reasons):
  (1) Showdown `TeamValidator('gen3ou')` legal; (2) the procedural generator's own coverage
  predicates: every species in the port's species table and not in `REJECT_SPECIES`, every move
  `isModeledMove(m, allowHP=true)` and not in `REJECT_MOVES`, every item in `MODELED_ITEMS`, every
  ability in `speciesAllowedAbility(species)`; (3) the pinned `Gen3Teambuilder` validation (the
  same one training uses); (4) **out-of-pool**: a team whose 6-species SET equals any pool team's
  6-species set is DROPPED (its count is reported). Then a seeded sample without replacement.
  The same predicates are also run over the pool for information, but the pool arm is NOT
  filtered, because it is the training distribution.
* **(c) PROCEDURAL.** `src/rust_sim/harness/ou_random_teams.js`, `coupled` (usage-weighted lead,
  then the Smogon teammate joint), with the same coverage predicates, validated by
  `TeamValidator('gen3ou')` and then by `Gen3Teambuilder`, from a fixed seed. Same out-of-pool
  species-set drop.

**n.** **400 battles per arm**, paired, 1,200 battles total. If the plumbing smoke's measured
throughput makes 400 infeasible within ~3 h of 4 workers, n drops to 300 before any read, and the
change is recorded in the README with the throughput that forced it.

**Failures.** A battle that errors or exceeds its (contention-scaled) wall bound is INCONCLUSIVE.
It is dropped from ALL THREE arms at that index, so pairing survives, and the dropped count is
reported per arm. If more than 25% of attempted battles in any arm fail, that arm's read is
INCONCLUSIVE and nothing about it is reported as a result. A timeout is never a semantic outcome.

## 3. What is scored, at every TRAINEE decision

The model runs its normal forward (the same one that picks the action), and the stashes are read
straight after it. The TRUE labels are built with the pinned `Gen3Env` label methods themselves
(`_belief_labels`, `_spread_labels`, `_hp_type_labels`, `_item_labels`, bound onto a shim that
holds the trainee's battle as `battle1` and the opponent player's own battle as `battle2`). So the
label is byte-for-byte the training label, and the believed-slot mask is read from the same obs
`species_known` bit BeliefSlots keys on. Only slots/fields still UNREVEALED at that decision are
scored.

**Stratum:** `k` = number of opponent species revealed at the decision (1..5; k=6 has nothing
hidden for species). Every endpoint is reported per stratum and as an all-strata slot-weighted row.

| Endpoint family | Slots scored | Metrics (head AND Smogon prior, on the same slots) |
|---|---|---|
| **SPECIES** (`BeliefHead` species logits) | hidden (believed) slots. The k-hidden predictions are matched to the k true hidden species by the SAME min-CE permutation the training loss uses | **NLL** (matched CE); **top-1 accuracy** (matched); **ECE** (15 equal-width bins of top-1 confidence vs matched correctness); **CONFIDENTLY-WRONG rate** = top-1 prob > 0.8 AND the top-1 species is NOT among the opponent's still-hidden species (assignment-free: a confident belief in a mon the opponent does not have). The matched-slot variant is reported as secondary |
| **MOVES, revealed slots** (typed `last_move_belief_logits`) | revealed-species slots with ≥1 true move still unrevealed | per-slot **BCE** exactly as trained (mean over 400 channels, full true moveset); **hidden-move recall@4** = the share of the true still-unrevealed moves inside the top-(4−r) non-revealed channels (r revealed; a revealed bare Hidden Power counts all 16 typed channels as revealed) |
| **MOVES, hidden slots** | believed slots, matched by the training loss's BCE-relevant permutation | per-slot **BCE**; **recall@4** of the true moveset |
| **ITEM** (`last_item_logits`) | revealed slots whose item the trainee's board still shows as unknown | **NLL**, **accuracy** |
| **HP TYPE** (`last_hp_type_logits`) | revealed slots whose true mon runs Hidden Power (the trained mask) | **NLL**, **accuracy** |
| **SPREAD** (nature/EV generative head) | revealed slots with an invertible true spread (the trained mask) | **nature CE**, nature accuracy, **EV MAE** (raw EV points), **derived-stat MAE** (raw stat points): the terms of the loss it was trained with |

**The Smogon-prior control**, computed from the SAME non-persistent `data/pokemon/` buffers the
heads fuse with, i.e. each head with its learned delta set to zero:
* species: the T0 naive-Bayes team prior `species_team_prior_logits` (identical in every hidden
  slot, so the matching cannot favour it);
* moves, revealed slots: `move_prior_logits[species]`, revealed pinned, typed-HP composed with the
  Smogon HP-type prior and the same tracker narrowing;
* moves, hidden slots: the Smogon MIXTURE `Σ_s P_T0(s | revealed) · P_prior(m | s)`, typed-HP
  composed per species. This is stronger than the head's own hidden-slot base, which is the flat
  floor;
* item / HP type: `item_prior[species]` / `hp_prior[species]`;
* spread: `nature_logprior[species]`, `ev_prior[species]`, and the derived stats they imply.

## 4. Headline, rule and verdict branches

**Primary (as asked).** The **memorisation gap** on the HEAD's species NLL and species
CONFIDENTLY-WRONG rate: `Δ_b = (b) − (a)` and `Δ_c = (c) − (a)`, per stratum and all-strata.
**Co-headline (the channel that reaches decisions):** the same gaps on the revealed-slot move BCE.

**The control, which is what licenses the word MEMORISATION.** A ladder team can be intrinsically
harder to predict than a pool team, so a raw gap alone does not show memorisation. Define the
head's **advantage over the prior** `A_arm = metric_prior − metric_head` (positive = the head beats
Smogon), and the **non-transferring advantage** `D_b = A_pool − A_ladder`, `D_c = A_pool − A_proc`.
Memorisation is the part of the head's pool advantage that does not transfer, so it is `D`.

**CIs.** 95% percentile bootstrap, 2,000 resamples of BATTLE INDICES, drawn jointly across the
three arms (paired). Each metric is a ratio of sums over the resampled battles' slots, so a battle
with more decisions weighs more, exactly as a pooled rate does, and no cross-battle correlation is
ever computed. A CI that straddles 0 is reported as **NOT DETECTED**, never "equivalent".

**Verdict branches** (all-strata row, species NLL; the same logic is reported for revealed-slot
move BCE):
1. **MEMORISATION DETECTED** iff `D_b`'s CI lies entirely above 0. Its size is `D_b / A_pool` =
   the share of the pool advantage that does not transfer to the ladder.
2. **NET HARMFUL OFF-POOL** iff additionally `A_ladder`'s CI lies entirely below 0 (the head is
   WORSE than the plain Smogon prior on ladder teams).
3. **TRANSFERS** iff `D_b`'s CI lies entirely below `0.25 · A_pool` and `A_ladder`'s CI lies above 0.
4. Otherwise **NOT DETECTED / UNRESOLVED**, with the CI stated.

**"Confidently wrong off-pool"** is reported as a LEVEL (CW rate on (b) and (c), per stratum, with
CIs) and as the gap to (a). A decision-relevant flag is raised if the ladder CW rate at any stratum
k ≥ 3 has a CI lower bound above 0.05 (1 in 20 hidden slots confidently believed to be a mon the
opponent does not have).

## 5. Predictions (mine, before any data)

1. **Species, POOL:** the head beats the Smogon prior by a wide margin; `A_pool` on NLL is about
   +1 nat all-strata, rising with k (more revealed ⇒ easier to key a memorised team).
2. **Species, LADDER:** `D_b > 0`, **MEMORISATION DETECTED**, with 40–70% of the pool advantage
   failing to transfer; `A_ladder` stays positive (ladder teams share the ADV OU cores the pool
   is built from). NOT net harmful.
3. **Species, PROCEDURAL:** `A_proc` near 0 or negative. The generator draws teams from the same
   Smogon teammate joint the prior encodes, so the prior is close to Bayes-optimal there, and any
   pool-specific delta should hurt. `D_c > D_b`.
4. **Species CW rate:** pool about 10–25% of hidden slots at k ≥ 3; ladder HIGHER by +3 to +10 pp at
   k ≥ 3; the decision-relevant flag (ladder CW lower bound > 0.05 at some k ≥ 3) FIRES. ECE worse
   off-pool, with the head over-confident.
5. **Revealed-slot moves (the reinjected channel):** `A_pool > 0`; `D_b > 0` but a smaller share
   than species (per-set memorisation transfers better than whole-team memorisation because common
   sets are common everywhere): 20–50% non-transferring.
6. **Hidden-slot moves:** the head's posterior is constant across decisions (verified to 1e-6);
   the Smogon mixture beats it on (b) and (c) and loses to it on (a).
7. **Item:** a small `A` everywhere (Leftovers dominates gen-3 items); `D` NOT DETECTED or small.
8. **HP type:** `A_pool > 0`. ⚠️ **The LADDER arm's HP-type row is flagged in advance as
   contaminated:** Metamon fills most Hidden Power slots as untyped "Hidden Power" with all-31 IVs,
   i.e. HP DARK, which is not ladder reality. So that row is reported but not interpreted.
9. **Spread:** ⚠️ Metamon fills the unrevealed fields (EVs/natures) FROM USAGE STATS, so on (b) the
   Smogon prior is favoured by construction, and the same holds on (c), whose spreads are sampled
   from the Smogon spread prior. I predict `A_ladder` and `A_proc` ≤ 0 on nature CE, and the spread
   rows are reported as bounded by this construction, not as clean transfer evidence.

## 6. Hazards declared up front

* **Metamon's hidden-slot filling** (from usage stats) makes (b) partly PRIOR-GENERATED, which
  biases (b) TOWARD the Smogon prior on every field it filled, most of all on moves, items and
  spreads. That makes `D_b` a LOWER bound on how much transfers to a real ladder team whose sets
  are idiosyncratic. For species it matters less (species are what the replays reveal).
* **(c) is prior-generated by design**, so it is the arm where the prior should win. `D_c` is the
  pure "pool-specific delta" cost, not a ladder estimate.
* **The opponent pilots differ by team** even though the policy is fixed, so reveal ORDER and
  timing differ across arms. Stratifying by k controls how much is revealed, not which.
* **The species head is not fed forward.** Its gap measures what the trunk was trained to encode,
  not what the policy reads directly. The revealed-slot move rows are the decision-path evidence.
* **Weather/global context** is broadcast into every pre-transformer token, so a T0 head sees a
  little team information (e.g. sand ⇒ Tyranitar). Declared, not controlled.
* A bug found in poke-env's reading, or a head reading something it should not, is REPORTED, not
  fixed here.
