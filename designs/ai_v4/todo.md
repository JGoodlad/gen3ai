# AI v4 — Future Work

ai_v4 is otherwise closed out (impl_step1–9 are the as-built record). The headline next item
is **pathology hunting**; the rest are genuinely-open corner cases and deferred infra.

---

## ▶ Next: pathology hunting (eval-replay analysis)

Mine the quota-gated eval forensic traces (`<run_dir>/eval_traces/step_<N>/<opponent>/`) for
residual behavioural pathologies and feed what they surface back into reward/obs. (Headline ai_v4
tail — not fleshed out here.)

---

## 1. Boost-delta corruption on phaze-on-us

**Where:** `src/agents/training/turn_delta.py` — `build_from_events` boost-delta computation
(~line 440). The boost stage delta is **not** event-foldable (`SETBOOST`/`clearboost`/`invert`/
`copy`/`swap` carry only an `op`, no realized amount — see impl_step9), so it is sourced from the
snapshot stage diff:

```python
our_boost_delta = (
    np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
    else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
)
```

When opp uses Roar/Whirlwind on us, our active mon changes mid-turn but `our_switch_to` stays
`None` (we picked a move action, not a switch). `prev_ctx.our_boosts` snapshotted the OLD active
mon's stages; `curr_ctx.our_boosts` the NEWLY phazed-in mon's — the diff compares two different
Pokémon's stat stages.

**Downstream impact:** the corrupted delta flows into `_compute_futile_setup_penalty` (fires when
`our_boost_delta.sum() == 0` on a boost move → fires incorrectly when a real boost was Roar'd away)
**and** into the per-slot TurnDelta observation (`OFFSET_OUR_BOOST_DELTA`).

**Status after the full event-fold (`90739dd`): still open.** The fold folds move attribution /
outcome / status / item / HP from the event log, but the boost delta is still a snapshot stage diff
(it has to be — the amount isn't in the log), so the phaze-on-us corruption is unchanged. The legacy
`build_legacy` (`turn_delta_legacy.py`, test-only) shares the same snapshot diff.

**Fix sketch:** when our active species changed between snapshots AND we didn't voluntarily switch,
treat it as a phaze and zero the boost delta. The right signal for "opp phazed us" is
`opp_resolved_move_id ∈ PHAZING_MOVES` this window — now available on the folded delta.

---

## 2. TurnDelta encoder full-vector consistency fuzz (Layer 6)

**Where:** `src/agents/observation/turn_delta_encoder.py` produces a **`TURN_DELTA_DIM = 159`**-dim
vector per historical turn.

`event_log_fuzz_test` validates the **event-log source** of the per-turn fields; `turn_history_fuzz`
pins encode **byte-identity**; `trapping_signals_fuzz_test` validates the three trapping dims at
their absolute indices. What still does **not** exist is a single `decode_to_dict()` round-trip that
re-decodes the encoded vector and asserts **every** one of the 159 encoded dims (move-id embeddings,
power, has_secondary/has_recoil, cant one-hot, switch flags, HP-level vectors, faint-cause, status
transitions, attempted-action, trapping bits) against the source `TurnDelta`. The decode method
exists for diagnostics; the gap is wiring it into a fuzz as an end-to-end assertion.

---

## 3. State-encoder per-slot consistency fuzz

**Where:** `src/agents/observation/state_encoder.py` produces the **3321-dim** observation via
`Gen3ObservationEncoder.encode()`.

`alignment_test` pins byte-identity for value-neutral refactors, but **no** e2e test re-decodes obs
slots back to live battle state (species/types/HP/spread/HP-tracker probs, the matchup matrix, the
reactive block, global env) and compares against `LiveView`. Bugs here would silently feed wrong
inputs to the model — the same class as the original HP-attribution bug, one layer up.

**Fix sketch:** `state_encoder_fuzz_test.py` (bridge-backed) that drives random battles and
re-decodes selected obs slots for comparison against the `LiveView`. Catches live-state→encoder bugs
(not encoder→model bugs).

---

## 4. Sleep Talk delegation: `last_move` stays `"sleeptalk"` on delegation failure

**Where:** poke-env `|move|` handling. When Sleep Talk delegates to an out-of-PP move, only the
first `|move|...|Sleep Talk|` line fires (no delegated move), so the MOVE event carries
`move_id = "sleeptalk"`. The event-fold's `TurnView` is delegation-aware but can only fold what the
protocol emits, so this stays `"sleeptalk"`. Benign today (`"sleeptalk"` isn't in any reward
signal-gating set) — flagged in case a future signal gates on Sleep Talk recognition.

---

## 5. Broadcast ability priors into the `move_network` input

**Where:** `src/agents/model/features_extractor.py` move-features concat.

The defender's 4-dim ability block (`[ability1_id, ability2_id, dominance, known]`) reaches the role
encoder and the matchup-cell math (joint expectation with ability priors) but never the per-move
processor. If the model still under-reads critical immunities (e.g. fails to switch out of
Earthquake when opp has 95%+ Levitate-likely), broadcast the opp's 4-dim ability block per matchup
cell into `move_network` input the way the 16-dim HP probs were.

**Defer until evidence of under-reading.** The matchup-cell fix gives the correct expected
effectiveness and the role encoder carries the ability embedding; cross-team attention can connect
them. Adding dims has clear cost (params, churn, weight-shape invalidation). **Detection:** sample
battles where opp has high-dominance Levitate and we click Ground turn 1; if the corrected matchup
cell doesn't steer us off Ground often enough, this becomes actionable.

---

## 6. Reward signal cross-product matrix

**Where:** `src/agents/training/reward_invariants_e2e_test.py`.

The invariant set checks each `RewardBreakdown` field individually but not signal **interactions**.
Build a co-occurrence matrix during the fuzz run and assert known-impossible pairs never co-fire
(e.g. `explosion` vs `finishing_blow` — different mechanisms for the same faint; `futile_setup` vs
`boost_utilized` on the same turn).

---

## 7. Mirror-match instance ambiguity (foot-gun, no current bug)

When both teams run the same species, `live.ours` / `live.opp` (and the per-side battle dicts) keep
them distinct, so the event-sourced + LiveView surfaces are already side-safe. Any *new* code that
joins across sides keyed only on species name (a hypothetical "who has the higher X" check) would
need to stay side-aware. No current bug — flagged as a recurring foot-gun.

---

## Done — closed since the start of this work

(Older closures from the DamagingMoveEvent / reward-fuzz / move-outcome / next-run-bundle /
event-sourced-fold sessions are recorded in the impl_step docs; the recent ai_v4 closures:)

- **Strict battle-API + event-sourced fold + perf + trapping** — recorded in
  `impl_step9_strict_api_perf_and_trapping.md`: Phases 3–4 consumer migration + `agents.enums` seam
  + `strict_api_lock_test.py` lock; the TurnDelta full event-fold that **deleted `battle_context.py`**
  (`TurnDelta`/snapshot relocated to `turn_delta.py`/`battle_snapshot.py`, legacy builder to
  `turn_delta_legacy.py`); reward `_read_live` retirement; the ~2× obs-build performance pass; and the
  trapping signals (obs 3321, `gen3_trapping_signals_v1`).
- **Non-damaging phaze opp-move attribution** — RESOLVED by the event-sourced fold. The old gap
  (`delta.opp_move_id == None` when opp used a non-damaging move before being Roar'd, because
  poke-env's `opp_all_last_move_ids` is cleared on `switch_out`) is bypassed: `TurnView` folds the
  opp move from the `MOVE` **event** in the log regardless of damage (`turn_view.py`), so
  `build_from_events` recovers it. (`opp_all_last_move_ids` survives only on `battle_snapshot.py` for
  the HP tracker + the legacy/transition-fuzz path.)
- **Eval no longer pauses training** — the old "eval is pure overhead, trainer fully paused on
  `thread.join`" premise (former item 9) is resolved: `PerOpponentEvalCallback` now spawns
  work-stealing `eval_worker` subprocesses against a frozen snapshot (`5a21b84` / `6741767`) — eval
  runs on CPU, decoupled from the training GPU, without pausing rollout collection. The residual
  batch-1 per-decision forward pass inside a worker is a minor, low-priority throughput item, not the
  training-blocking cost it was.
