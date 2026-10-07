# Opponent intent — the α / β heads, their metrics, and their consumers

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## Under `--belief-tokens fixed_mass`: ONE flat pointer replaces α / β (X5 U4)

Everything below describes the **blob** arm (production). In the X5 `fixed_mass` arm
(`gen3_x5_flat_pointer_v1`, [`../endstate/design_x5_belief_tokens.md`](../endstate/design_x5_belief_tokens.md)
§3.7) α / β are RETIRED — constructed, re-initialised by SB3, then dropped by the policy's `_build`
before the optimizer is made (`ExtractorApi.retire_superseded_intent_heads`), so no non-X5 initial byte
moves — and `agents/model/flat_intent.py` holds ONE pointer over a fixed column layout for K seats:
`[0, K)` the active's move seats (THE one order), `K` OTHER_move, `K+1 .. K+6` a switch to each of their
slots (a revealed mon, or the HYPOTHESIS a hidden slot holds), `K+7` OTHER_species. One shared scorer
(token ⊕ the two team pools ⊕ the candidate's kind) plus the candidate's DETACHED log π, one softmax;
masks are structural; the input is detached under `--opp-intent-grad-mode detached` as α's was.

* **Labels** (`flat_intent_targets`, from the same Rust label — no label change): a move in the seats →
  its seat (a typed Hidden Power label → a REVEALED HP's seat, num 237); a move beyond the seats →
  OTHER_move; a revealed switch → its slot; a hidden switch-in → the hypothesis slot holding its species,
  else OTHER_species when it is in the tail. **A belief miss is supervised as an OTHER label**, so the
  set-valued partial credit is superseded in that arm. Masked: non-choices, and a choice outside every
  candidate's support (Struggle, a learnset gap, a species outside V) — counted as `flat_unmodeled_rate`.
* **Metrics** (`instrumented_ppo/flat_intent_fold.py`, static, `fullgraph`-traced): `flat_*` (loss, mask
  rate, accuracy, info gain, the kind decision both ways, which move / which target), pooled and per
  opponent class, and **`opp_intent/other_label_rate`** — of the opponent's CHOICES, the share labelled
  OTHER (F-X5-8: the belief-miss share `alpha_mask_rate` mixed with non-choices), split
  `other_move_label_rate` / `other_species_label_rate`.
* **Consumers** read a re-expression (`compat_intent_logits` + `FlatConsumerOps`): α over the K seats +
  OTHER_move + log α_SWITCH (a guarded logsumexp — the unguarded one has a NaN gradient on a row with no
  switch target, and three consumers do not detach α), β over the six slots + OTHER_species. OTHER is
  PRICED, never a zero row: OTHER_move's seat-axis column is each per-candidate cell on the op's FULL
  move axis contracted with the renormalised tail (`FixedMassMoves.other_u`); OTHER_species' `out_cells`
  column is the OTHER-mode D1 pass (the tail-averaged defender, P(KO) nulled), its `opp_p_ghost` the
  tail's P(Ghost); num-table reads use `pair_outcome.seat_num_table` / `seat_in_set`. All four
  per-seat consumers apply the meaningful-K gate (`seat_live`, F-X5-15) in that arm.
* **Rendering**: `render_flat` names the options; `RLPlayer._opp_intent` still renders only α / β, so a
  fixed_mass trace carries no `opp_intent` block yet (U4 FINDING).

## The two heads

- **`α`** — a distribution over the opponent's K believed threat-move seats (the refined **E4**
  tokens) **plus a SWITCH option**. Seat k's logit is scored from seat k's own token through a
  SHARED scorer, so `α` is equivariant under permuting their moves; SWITCH is scored from board
  context alone (there is no per-seat object to point at — it is the "none of these" option).
- **`β`** — given a switch, which of their mons comes in. A pointer over their six team tokens,
  masked to alive-and-non-active: an illegal switch-in must be UNREPRESENTABLE, not merely
  unlikely, or the head spends capacity learning the rules.

**Why pointers and not a flat `Linear(ctx, K)`.** The flat form passes every shape test and then
learns "seat 0 is usually right" from the belief's own `w.topk` sort order — memorising exactly the
ordering `α` exists to correct. Equivariance is gated in both axes.

**Matching is by canonical id.** Seats permute every turn and are built by the model mid-forward, so
the env emits the opponent's move NUM and `match_seats_to_move_num` locates it at loss time. A
belief miss is MASKED and `opp_intent/alpha_mask_rate` is logged — that rate is the BELIEF's coverage
failure, and folding it into "α was wrong" would hide which component to fix.

**Only a CHOICE is a label** (`opp_intent_labels.build_opp_intent_label`). A switch-in the
opponent did not choose is MASKED — our phaze, a DRAG by our Roar / Whirlwind (even after their own
switch, or while they were asleep / paralysed), a replacement after a faint whether the faint
shares the window or lay in the previous one — as is a move our Encore overrode before they acted.
A CALLED move is labelled as its CALLER (Sleep Talk, not the Rest it called). The drag,
straddling-replacement, caller and Encore rows are `gen3_intent_label_semantics_fixes_v1` (the Rust
core M3 loss catalogue's L1–L5; before them ≈ 16 labels per 1,000 decisions named a mon the
opponent never picked).

**The label is for the PREVIOUS decision.** Their turn-t action is only observable while building the
obs for t+1, so `instrumented_ppo` shifts the label block back one row **before `get()` shuffles**
and drops any pair whose successor starts an episode (`align_labels_to_predictions`). Skipping that
drop splices one battle's first decision onto another's last board — invisible in every metric.

## Reading `opp_intent/*` — the per-class split, and the whole path to a human

**The bare key is a MIX, and the mix MOVES.** Supervised rows ran **100% bot at 2M and ~7% from 6M
on** (self-play competence-gating), so a pooled metric rises as the mix shifts toward the pool and
that rise is indistinguishable from the head improving. The pooled α accuracy at 2M read 0.580 —
which was a pure bot measurement; the pool figure at the same step was 0.296. Any trend that spans
the ramp is uninterpretable; a trend after ~6M happens to be safe, but read `_pool` and do not rely
on that.

The split covers **every** axis: the KIND decision both directions (`alpha_switch_recall` /
`_precision`, `alpha_move_kind_recall` / `_precision`), the move axis (`alpha_move_recall_top1` /
`_top2` against `alpha_move_baseline_argmax_w` — compared LIKE FOR LIKE, both "given they moved"),
the β pointer (`beta_recall_top1` / `_top2`, `beta_info_gain_nats`), and the switch-coverage matrix
(`beta_switch_to_revealed` / `_hidden_found` / `_hidden_missed`, which partition voluntary switches
and sum to 1, plus `beta_belief_miss_rate` over the rows that ASKED). It used to cover only
accuracy / info-gain / count, which left exactly the metrics a reader uses to LOCATE a deficit
pooled. `alpha_mask_rate` stays whole-batch: it is the BELIEF's coverage failure, and folding it
into "α was wrong" would hide which component to fix.

One computation serves both reads — `_alpha_subset_metrics` / `_beta_subset_metrics` /
`switch_coverage_metrics` take a row subset and a suffix — so a pooled and a stratified number can
never drift apart. `switch_coverage_metrics` is module-level rather than a closure in the PPO loop
because nothing tested that matrix at all, and a metric with no test can silently read zero.

**Interpretability is a first-class output, not a debug aid** (`render_alpha` → the trace's
`opp_intent` block): `α` as a ranked list of NAMED moves. The owner constraint is that the model may
only ever point at options it can name, and rendering is where that becomes checkable.

**The path from the head to a human is WHOLE**, and it is worth naming because for one commit it was
not — `RLPlayer._opp_intent` built the block and `BattleRecorder` never wrote it, so the payload was
computed on every decision and dropped on the floor:

`α`/`β` logits → `RLPlayer._opp_intent` (`render_alpha`, plus the `β` naming rule below) → the
summary invocation's `opp_intent` block → `engine.build_opp_intent` / `opp_intent_text` → the
prober's Summary **EXPECT** line, `analyze`'s `opp_intent`, and the web replay's per-turn *expect*
line (`src/main/prober/CLAUDE.md`).

## The β naming incident, and what it measured

The posterior used to name **both**, and that was a display defect, not a modelling one. `β`'s
candidate mask is alive-and-not-active, which **includes revealed bench mons**, while the species
aux only supervises the *believed* slots — so on a revealed slot the posterior is un-trained.
Measured over a 843-battle sentinel sweep (2026-08-19): the rendered name was a mon **not on the
opponent's team at all in 73.3% of 6,876 pivots** (88.3% on revealed slots), and an owner-facing
analysis read "β predicts porygon2" on a turn where `β`'s slot held the revealed Salamence and `β`
was **CORRECT**. The slot mapping itself was validated 7560/7560 against the belief block's
hidden-slot set — the pointer was fine; the label beside it was a different head's output.

## The nine α consumers

Eight modules now contract α against the op's physics — `IntentValueReduce`, `IntentMoveCell`,
`IntentThresholdMoveCell`, `IntentConditionalMoveCell`, `PairOutcomeMoveCell`, the v94 pair
`PairOutcomeSwitchCell` / `SwitchBranchMoveCell`, and the v95 pair `ConditionalThreatCell` — and, under `--move-resolution on` (v141, OFF in production), `MoveResolutionCell`, which REPLACES the seven pointer-cell consumers (they are retired on a built policy) and contracts α / β the same way (the publication, stop-grad, seats masked never renormalised, the unmodeled mass `1 − Σα − α_SWITCH` named rather than reassigned; under X5 `fixed_mass` it reads the flat pointer's re-expression with OTHER priced, OTHER_move split into one seat per priority level). They share four conventions, and each one exists because breaking it fails
silently:

## The two conventions `pair_outcome.py` added

`pair_outcome.py` adds two the others did not need, and both are worth copying:

* **Stop-grad α unconditionally** on a POLICY-side consumer. `label_only` happens to cut the
  PPO→`alpha_head` route today, but resting on it makes the route's EXISTENCE a function of a
  TRAINING flag — one `--belief-grad-mode` change away from silently reopening.
* **Give α a documented FALLBACK if the flag can stand alone.** With no intent head it uses the
  shipped R1 `belief_mean` rung (`α := w/Σw`), re-exported from `pair_reduce` rather than
  re-spelled. That makes the flag independently enableable and separates the DELIVERY claim from
  the DISTRIBUTION claim — but the two are NOT the same object (presence belief vs usage belief,
  and one sums to 1 where the other sums to `1 − α_SWITCH`), so say so loudly wherever it appears.

## Where a consumer sits, and the critic-side gradient guard

**⚠️ WHERE a consumer sits in the phase chain decides WHICH α it can have.** Every consumer above
runs at the pointer stash, i.e. after the α/β heads are scored, so "read the PUBLICATION" is
available to them. A consumer inside `CLSPool` would pool at T2 **before** those heads exist and
could only take the R1 `belief_mean` rung — unconditionally, even with `--opp-intent` ON; that is
ORDERING, not a fallback, and "upgrading" it to the publication is unbuildable without moving the
pool. (No such consumer exists today; the pair-value critic route that was one is deleted.) **The
general rule: before choosing an α rung, locate the consumer in the tier chain — the answer may
already be fixed.**

**A critic-facing α consumer owes its own gradient guard.** `value_route_gradient_test.py` iterates
`_value_pooled_routes` — since the deletion wave a ONE-member seam (`value_entity_pool`), kept
generic precisely because its value is covering the NEXT route the day it is written, which is
exactly what did not happen for the four it lost. A route added to that seam is covered by
construction; the token-content injection (`value_threat_proj`) is NOT in the seam, by
design: a post-pool additive route must collapse the team axis, and the only equivariant collapse is
a sum, which cannot tell one mon losing 90% of its bar from six losing 15%. It is zero-init, so a
disconnected one is indistinguishable from one that learned nothing — the exact gen-12 dead-tail
failure, one level up. The guard therefore carries a dedicated cell for it, and its real claim is
*every zero-init projection the critic depends on receives critic gradient*, not *every seam entry
does*. **When you add a critic-side enrichment anywhere other than the seam, extend that test in the
same pass.**

**A per-DEFENDER delivery is not a per-defender BELIEF.** `reduce_pair_in_all` produces six rows,
one per our mon, from ONE α — the reduction may vary per defender, the DISTRIBUTION may not (that is
the whole content of §2's "D2 and D3 fall to ONE restriction"). α still has no `J` axis, so defect D3
stays a shape error even in the phase whose entire job is producing a column of rows. When you add
another per-defender consumer, take α from the same `pair_alpha` / `pair_alpha_full` ladder rather
than computing a defender-conditioned one; the planted-violation test in
`pair_outcome_switch_test.py` is what proves the structure is real rather than merely intended.
