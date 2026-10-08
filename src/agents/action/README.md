# Gen3 Action Space

The 11-action discrete space the model acts in, and the row guard that keeps the model's view of our
active's moves aligned with it. Every legality decision, mask and choice token is the RUST core's
(`src/rust_sim/`, served to training by the env core and to the inference service as rows + masks).

The Python action stack that decoded an action against a poke-env battle — the masker
(`mask_generator.py`), the mapper (`mapper.py`), the `Choice` type (`choice.py`) and the poke-env
`BattleOrder` serializer (`serialize.py`) — is DELETED with the Python battle layer (T27 P6 slice 6d-2),
with its tests (`mapper_test.py`, `fuzz_test.py`, `fuzz_test_unit.py`, `trapping_signals_fuzz_test.py`,
`telemetry_e2e_test.py`).

## Action Space

| Index | Meaning |
|---|---|
| 0–5 | Switch to team slot 0–5 |
| 6–9 | Use move in request slot 0–3 |
| 10 | Struggle |

## Move-order alignment — the one guard that stays

Our active mon's moves exist in TWO orders in every observation row: the per-mon move slots are
SORTED BY `Move.id`, while the request block (`reactive.active_req_moves`) and actions 6–9 are in
REQUEST order. `ordering_integrity.check_obs_move_order` is the THROWING guard (`OrderingMismatchError`)
that every served row maps each request move onto exactly one sorted slot and carries request legality
equal to the mask's move bits; it runs at `InferenceService.submit`. The model crosses the two orders
only by move-num identity (`agents.model.extractor_ctx.active_request_sorted_match`;
`gen3_move_legality_by_id_v1`).

## Files

| File | Purpose |
|---|---|
| `constants.py` | The 11-action layout constants (`ACTION_SPACE_SIZE`, `MOVE_START`, `SWITCH_START`, `STRUGGLE`, …) |
| `ordering_integrity.py` | `check_obs_move_order` (the row guard) and `row_offsets` (the offsets it reads, pinned against the extractor's unpack) |

## Tests

| File | Type | What it covers |
|---|---|---|
| `ordering_integrity_test.py` | Unit | The row guard on real rows (the learner golden buffer + the compile parity fixture) and on targeted corruptions of them |
| `move_order_bridge_integration_test.py` | Integration (`sim`) | Real banked battles through the Rust core: the scored slot, the action token and the executed move agree |
