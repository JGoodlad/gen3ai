# feature_coverage: does every battle edge case reach the network?

A deterministic, server-free suite that answers two questions for each major Gen 3 battle edge case:

1. **Capture**: is it encoded into the observation vector at the expected offset (right one-hot bit / scalar /
   embedded id)?
2. **Network**: does that encoded signal flow through the *real* `Gen3FeaturesExtractor` and move its policy / value
   output, rather than being silently dropped (by a key-padding mask, an event-seat encoder that never reads the
   block, or a layout gap)?

If a signal does **not** reach the network, the failing assertion stays: a dead obs dim is the most valuable thing
this suite can find.

## Layout

- `_support.py`: the shared harness (model / encoder built once; `make_delta`, `anchor_delta`, `encode_delta`,
  `obs_with_delta`, `obs_with_event`, `set_region`, `assert_reaches_network`, `assert_delta_reaches_network`,
  `assert_event_reaches_network`). A fact a probe expresses as a `TurnDelta` reaches the network as an **event-window
  row**: `delta_to_event_cols` routes it, and the asserters check the two rows differ before comparing the forwards
  (a translation that collapsed both sides to one row would pass while testing nothing). The fixture is built
  `history_events=True` and both asserters refuse to run without it, because the event window's raw ids are read ONLY
  by the flag-gated event-seat encoder. **Not a test module.**
- `failed_protect_feature_test.py`: the EXEMPLAR every file follows.
- `move_resolution_feature_test.py`: miss / crit / the cant reasons / effectiveness buckets / attempted-move-id /
  `opp_move_known`.
- `faint_cause_feature_test.py`: faint flags / faint causes / multi-KO / Explosion self-KO / KO-before-acting.
- `status_item_feature_test.py`: status applied and cured / item-consumed bits.
- `switch_trap_boost_feature_test.py`: switch vs move / boost deltas / forced-switch phase / move order / species
  block / attempted-switch-rejected (trapped pivot).
- `static_blocks_feature_test.py`: reachability sweep of the NON-history obs blocks (per-Pokémon, global env,
  reactive, active-context volatiles). Capture correctness for these is owned by each encoder's own unit test; here
  we only prove they are not dead obs regions.

## Pattern

```python
model, layout, _ = feature_model()
base    = anchor_delta(our_move_id="tackle")              # slot present
variant = anchor_delta(our_move_id="tackle", <edge field>)  # differs in ONE field
enc = encode_delta(variant)                              # CAPTURE: assert offset
assert_delta_reaches_network(model, layout, base, variant, "msg")  # NETWORK
```

## The strict xfails

Each carries its reason and the gap it points at in `designs/ai_v9/design_frame_deletion_coverage_gaps.md`; strict
means closing a gap turns the probe RED instead of letting it pass in silence.

- **The translator, not the obs**: `delta_to_event_cols` emits ONE row while a `TurnDelta` is a per-turn AGGREGATE, so
  a two-sided turn, a multi-KO or an ordering fact needs several rows (gaps §4).
- **Real gaps that ship open by owner decision**: the rejected switch's TARGET has no event-window column (gaps §3).

## Reading the vocabularies

- KO-before-acting is encoded via `we_fainted` plus an empty move block, NOT a `|cant|` reason (`"fainted"` is not in
  `CANT_REASONS`); `frz` IS in it (a prose abbreviation of the list is not coverage).
- `ability: Damp` rides `CANT_REASONS_LIVE`, not `CANT_REASONS`: the frozen tuple sizes `TURN_DELTA_DIM`, the lag-frame
  width that archived runs recorded, so growing it would mis-slice history (`src/agents/observation/gen3_effects.py`).
  `cant_reason_id` reads the LIVE tuple; `encode_cant_reason` one-hots into the frozen one.
- Index orders are the code's: `_STATUS_ORDER` in `agents/observation/turn_delta_encoder.py`, `BOOST_STATS` in
  `agents/gen3_mechanics.py`.

## Run

```bash
# in a linked worktree, first: export PYTHONPATH=$PWD/src
python -m pytest src/agents/model/feature_coverage/ -q
```
