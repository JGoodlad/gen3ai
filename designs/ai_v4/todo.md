# AI v4 — Future Work

Open items from the foundational DamagingMoveEvent work (steps 3+4) and the
reward-fuzz expansion. Items here are either out-of-scope corner cases or
require infrastructure changes beyond a localized fix.

---

## 1. Boost delta corruption on phaze-on-us

**Where:** `src/agents/training/battle_context.py` — `TurnDelta.build` line ~370

When opp uses Roar/Whirlwind on us, our active mon changes mid-turn but
`delta.our_switch_to` stays `None` (we picked a move action, not a switch).
The boost-delta computation is then:

```python
our_boost_delta = (
    np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
    else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
)
```

`prev_ctx.our_boosts` snapshotted the OLD active mon's boost vector;
`curr_ctx.our_boosts` snapshots the NEWLY phazed-in mon's. The diff is
meaningless — comparing two different Pokémon's stat stages.

**Downstream impact:** Reward signals consuming `delta.our_boost_delta`:
- `_compute_futile_setup_penalty` fires when `delta.our_boost_delta.sum() == 0`
  on a boost move. If we used Dragon Dance and were Roar'd, the boost was real
  but the delta reads zero — the penalty fires incorrectly. Semantic argument:
  losing the boost to phaze IS a wasted turn, but the penalty is mis-attributing
  the mechanism.
- **TurnDelta encoder now emits boost_delta features** (Step 3 history
  expansion, slot offsets 39–52). The model is no longer insulated — the
  corrupted delta flows into the per-slot observation. Fix priority is
  higher than originally noted.

**Fix sketch:** Mirror the opp-switched logic — when our active species changed
between snapshots AND we didn't voluntarily switch, treat it as a phaze and
zero the boost delta. Detection: `prev_ctx.our_active != curr_ctx.our_active`
and `delta.our_switch_to is None`. Requires distinguishing BP (we phaze
ourselves via move) from opp's Roar — could gate on `delta.our_move_id in
PHAZING_MOVES` but that's our move, not theirs. The right signal is whether
opp's move this turn was Roar/Whirlwind; that's `delta.opp_resolved_move_id in
PHAZING_MOVES` after the recent migration.

---

## 2. Non-damaging phaze move attribution

**Where:** `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py`
reports ~80% `phaze_no_damaging_event` in scenario D.

When we Roar opp out and opp's mon used a NON-damaging move that turn
(Calm Mind, Toxic, Roar back, etc.), the DamagingMoveEvent never promotes
(no effectiveness emission for non-damaging moves). The legacy
`opp_all_last_move_ids` recovery is also broken because `Pokemon.switch_out()`
clears `_is_last_used` on every move (see poke_env/battle/pokemon.py:624-625).

Result: we cannot recover the phazed mon's actual move via either path.
`delta.opp_move_id` reads `None` and `delta.opp_damaging_event` is also `None`.

**Impact:** Replay logging shows `they_action=switched_to:X` (missing the
intent) when opp's mon used a status move before being phazed. Reward
signals depending on opp_move_id miss this attribution.

**Fix sketch:** Add a "pending_opp_move" tracker analogous to the damaging
event — set at `|move|` parse time, finalized at `|turn|N+1|` rather than
at an effectiveness emission. Would require a new property on AbstractBattle:
`opp_last_move_attempt` (any move, damaging or not) gated like the existing
properties. Could share infrastructure with `_pending_opp_damaging_move`.

---

## 3. TurnDelta encoder full-vector consistency fuzz

**Where:** `src/agents/observation/turn_delta_encoder.py` produces an
88-dim vector per historical turn (post Step 3 history expansion).

Effectiveness_fuzz currently validates layers 1-5 around effectiveness, order,
and the new DamagingMoveEvent. The encoder's OTHER features (move id
embeddings, has_secondary/has_recoil flags, cant_reason one-hot, switch
flags, HP delta scalars) are not fuzz-validated end-to-end — only via
isolated unit tests on `TurnDeltaEncoder`.

**Fix sketch:** Add a Layer 6 to effectiveness_fuzz that re-decodes the
encoded vector via `encoder.decode_to_dict()` and asserts every field
matches the source `TurnDelta`. The decode method already exists for
diagnostic purposes.

---

## 4. State encoder per-slot consistency fuzz

**Where:** `src/agents/observation/state_encoder.py` produces the
2414-dim full observation vector via `Gen3ObservationEncoder.encode()`
(post Step 3 history expansion + Step 4 unified transformer).

No e2e test verifies that the observation dims encode the live battle
state correctly. Specifically:
- Per-Pokémon slot encoding (typing, ability, item-consumed flag, HP,
  spread block, HP-tracker probabilities)
- Matchup matrix (move type × team mon type chart)
- Reactive features (boost stages, volatile effects, last-move info)
- Global env (weather, spikes counts, screens)

Bugs here would silently feed wrong inputs to the model — same class of
issue as the original HP attribution bug, just one layer up.

**Fix sketch:** Add `state_encoder_fuzz_e2e_test.py` that drives random
battles and re-decodes selected obs slots back to species/types/HP for
comparison against `battle.team` / `battle.opponent_team` live state.
Won't catch encoder→model bugs but will catch live-state→encoder bugs.

---

## 5. Sleep Talk delegation: last_move stays as "sleeptalk" on delegation failure

**Where:** poke-env `_handle_battle_message` for `|move|` events.

When Sleep Talk delegates to an out-of-PP move, only the first
`|move|...|Sleep Talk|` line fires (no delegated move). `mon.last_move` =
"sleeptalk". This is observed ~80 times per 50 battles in the
transition_fuzz Scenario B stats.

**Impact:** Reward signals that gate on specific move ids (e.g., the rest
check in `_compute_futile_attack_penalty`) may mis-classify these turns.
Probably benign since "sleeptalk" isn't in any of the signal-gating sets,
but worth confirming if a future signal adds Sleep Talk recognition.

---

## 6. Mirror-match instance ambiguity

**Where:** species → Pokémon lookup throughout poke-env and our code.

When both teams have the same species (e.g., ROAR_TEAM in transition_fuzz),
`battle.team[species]` and `battle.opponent_team[species]` are different
instances but distinguishing them requires the side prefix. Our code uses
the species name as the key, which works because the dicts are
per-side — but any code that joins across sides (e.g., a hypothetical
"who has the higher level X" check) would need to be side-aware.

No current bugs identified, but worth flagging as a recurring foot-gun.

---

## 7. Broadcast ability priors into the move_network input

**Where:** `src/agents/model/features_extractor.py` move_features concat (~line 540)

The defender's 4-dim ability block (`[ability1_id, ability2_id, dominance, known]`)
reaches the **role encoder** but never the per-move processor. After fixing the
matchup-cell math to factor in expected ability immunity (see commit landing
expected-matchup-with-ability), the role token carries the *abilities-likely* signal
and the matchup cell carries the *expected effectiveness*. Cross-team attention
(Threat / Pressure) connects them.

If the model still under-reads critical immunities (e.g. fails to switch out of
Earthquake when opp has 95%+ Levitate-likely), broadcast the opp's 4-dim ability
block per matchup cell into `move_network` input the same way we did with the
16-dim HP probs. Per cell: `[ability1_id, ability2_id, dominance, known]` × 6 opps
= 24 extra dims per move slot (or compress to dominance + "ability-affects-this-move"
flag).

**Defer until evidence of under-reading.** The matchup-cell fix gives the model the
correct expected effectiveness; the role encoder already carries the ability
embedding; cross-team attention can connect the two. Adding dims has clear cost
(param count, churn, weight-shape invalidation) and should only land if it's
demonstrably better than what the attention paths can derive on their own.

**Detection sketch:** Sample 50 battles where opp has high-dominance Levitate and
we click a Ground move on turn 1 (before the immunity is revealed). Track:
- Pre-fix: how often we pick Ground anyway (model treats matchup cell as 1×).
- Post-fix: how often the corrected matchup cell (~0×) steers us to a different move.
- If a meaningful gap remains after the matchup fix, this TODO becomes actionable.

---

## 8. Reward signal cross-product matrix

**Where:** `src/agents/training/reward_invariants_e2e_test.py`

The current invariant set checks each `RewardBreakdown` field individually.
What's NOT checked: signal interactions — e.g. should `finishing_blow` AND
`explosion` ever both fire on the same turn? Should `boost_utilized` AND
`futile_setup` ever co-fire?

**Fix sketch:** Build a co-occurrence matrix during the fuzz run and assert
known-impossible pairs never fire together. Examples:
- `explosion` (opp Exploded) and `finishing_blow` (we KO'd opp with a
  damaging move) — mutually exclusive (different mechanisms for the same
  faint).
- `futile_setup` (boost no-op'd at cap) and `boost_utilized` (we attacked
  with positive boosts) — possible if multi-turn, but on the SAME turn?

---

## 9. Batched inference for eval (and self-play snapshot opponents)

**Where:** `src/agents/inference/player.py` — `RLPlayer._predict_best_action`
(line ~107), driven by `EvalRLPlayer.choose_move` in
`src/agents/training/eval_callback.py`.

Eval is the single largest non-training cost in a run, and it is **pure
overhead**: `PerOpponentEvalCallback._on_step` does `thread.start();
thread.join()`, so the trainer is fully paused for the entire eval (no rollout
collection, no updates) — see eval_callback.py:182-186. Self-play snapshot
opponents (`RLPlayer` in the training env factory) run the same per-battle path.

**Current behaviour:** each decision runs a **batch-1** forward pass —
`_predict_best_action` encodes one battle's obs, does
`np.expand_dims(obs, axis=0)`, transfers to GPU, and runs one
`policy.get_distribution(...)` call. With `max_concurrent_battles=100` that is
~100 separate GIL-serialized forward passes + `.to(device)` transfers per
turn-round. Measured throughput (checkpoint_92095296, concurrency 100, contended
with live training): **~140 agent-steps/sec**, ≈7× slower per step than training
(~1000 fps), which batches all 64 envs into one forward pass.

**Confirmed non-lever:** raising `max_concurrent_battles` does *not* help —
50→104, 100→138, 200→118 steps/sec. 100 is the sweet spot; the bottleneck is the
serialized batch-1 inference, not battle concurrency.

**Idea:** collect all pending action requests across the in-flight battles into
one tensor, run a single batched forward pass, and route argmax/sample results
back per battle. poke-env already drives battles concurrently on one asyncio
loop, so the requests naturally arrive interleaved — the work is a small
batching layer in front of `_predict_best_action`.

**Requirements / sketch:**
- A request aggregator: each `choose_move` enqueues `(battle_tag, obs, mask)` and
  awaits a future instead of running inference inline.
- A flush trigger: run the batched forward pass when either (a) a short timer
  elapses (e.g. ~2–5 ms) or (b) the queue reaches a size cap. Tune so a flush
  gathers most of the ~100 concurrent battles without adding latency that idles
  the GPU.
- Batched forward: stack obs/masks → one `policy.get_distribution` call → apply
  the `mask + (m-1)*1e9` logit masking per row → argmax (eval) or sample → set
  each battle's future.
- Must stay correct under `max_concurrent_battles` concurrency and reset cleanly
  between opponents (`reset_battles()`); no cross-battle leakage of obs/mask rows.
- Keep the existing single-battle path for `play.py` / interactive inference, or
  make batch-size-1 a degenerate case of the batched path.

**Expected payoff (ballpark, unverified):** ~2–4× faster eval (≈140 →
~300–500 steps/sec). Not the theoretical ~100× of pure GPU batching, because the
**per-battle observation encoding** (`embed_battle`: state encoder + turn-history,
CPU/GIL-bound) and the **Showdown server round-trip latency** are floors that
batching does not remove. Net effect: eval overhead at 100M+ (every 4M steps,
~67 min at 1k fps, 9 capped opponents) would drop from ~5–7% toward ~3%.

**Priority:** defer until self-play is running — self-play multiplies the number
of `RLPlayer`-driven battles (snapshot opponents + eval), so the payoff grows
once v4 Step 1 lands. Measure uncontended eval throughput first (training paused)
to set the real baseline before investing.

---

## Done — closed by this session

- Cross-battle state contamination in `transition_fuzz` (`_prev_snapshot`
  was single-valued under `max_concurrent_battles>1`).
- Same bug in `reward_invariants_e2e_test` (`_prev_ctx`, `_last_action`,
  `_reward_fn` single-valued).
- 9 launcher_ui tests failing on a missing `eval_metrics_ts` default in the
  test helper.
- `faint_ours` magnitude range was wrongly exclusive at `-2.5` in the test
  invariant — full-HP Explosion KOs hit exactly `-2.5`.
- TurnDelta encoder + reward_manager futile_attack rest check migrated to
  `opp_resolved_move_id` (was reading stale `delta.opp_move_id`).
- Reward fuzz coverage expanded from 7 to 29 RewardBreakdown fields with
  per-signal invariants.
- Structural delta invariants (`opp_resolved_move_id == event.move_id` when
  event set, effectiveness is bucketed, scalar matches event).
- TurnDelta history expansion (Step 3 continuation): slot grew 39 → 88 dims
  with actor / target / switch_to species IDs (routed through
  `species_embedding`), boost deltas, phase flag, target_hp_delta, per-slot
  HP-level vectors, and target-status onehots at move-fire time.
  `ARCH_SIGNATURE` bumped to `gen3_unified_v2`.
- `TurnDeltaEncoder._species_id` now raises on unknown non-sentinel species
  (matches `SpeciesEncoder.encode()`); sentinel zeros stay for the
  "unrevealed / no actor" case. Caught a latent mock-leak bug in
  `player_test._make_battle`.
- Renamed `impl_step3_unified_transformer.md` → `impl_step4_unified_transformer.md`
  (was a duplicate-numbered step3; the unified transformer landed
  chronologically after damaging-event attribution).
