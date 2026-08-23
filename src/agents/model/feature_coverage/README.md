# feature_coverage — does every battle edge case reach the network?

A deterministic, server-free suite that answers one question for each major
Gen 3 battle edge case:

1. **Capture** — is it encoded into the observation vector at the expected
   offset (right one-hot bit / scalar / embedded id)?
2. **Network** — does that encoded signal flow through the *real*
   `Gen3FeaturesExtractor` and move its policy/value output, rather than being
   silently dropped (by `embed_delta_slot`, a key-padding mask, or a layout gap)?

This closes the last hop the bridge-backed fuzz tests under
`training/poke_env_gaps/` leave unchecked: those validate
`protocol → poke-env → TurnDelta → encoded vector` on real battles, but stop at
the encoded vector. Here we drive the encoded vector through the network.

## Layout

- `_support.py` — shared harness (model/encoder built once; `make_delta` /
  `anchor_delta` / `encode_delta` / `obs_with_delta` / `set_region` /
  `assert_reaches_network` / inspection helpers). **Not a test module.**
- `failed_protect_feature_test.py` — the EXEMPLAR every file follows.
- `move_resolution_feature_test.py` — miss / crit / all 12 cant reasons /
  4 effectiveness buckets / attempted-move-id / `opp_move_known`.
- `faint_cause_feature_test.py` — faint flags / all 8 faint causes (identity
  distinguishable) / multi-KO / Explosion self-KO / KO-before-acting.
- `status_item_feature_test.py` — status applied & cured (all 6, direction
  distinguishable) / item-consumed bits.
- `switch_trap_boost_feature_test.py` — switch vs move / boost deltas (every
  stat ±) / forced-switch phase / move order / species block /
  attempted-switch-rejected (trapped pivot).
- `substitute_confusion_feature_test.py` — the 2026-08-17 follow-up audit. **The one
  file here that probes the FOLD (`EventWindowTracker`) rather than the network hop**,
  because its questions are "does this fact produce a row at all" and "does the row say
  something TRUE" — a row that reaches the network carrying a false number is worse than
  one that never arrives, and `assert_delta_reaches_network` cannot see the difference.
  Substitute absorption / confusion self-hit / the full `|cant|` vocabulary; every verdict
  confirmed against real bridge-battle protocol. 19 passing + 8 strict-xfail (see
  `designs/ai_v9/design_frame_deletion_coverage_gaps.md` §3.4-3.7).
- `static_blocks_feature_test.py` — reachability sweep of the NON-history obs
  blocks: per-Pokémon (item-consumed, status, sleep/toxic counters, spread,
  PP/never-miss, HP-candidate, species_known), global env (weather one-hot +
  permanence + turns, Spikes, all four per-side screens), reactive
  (`forced_struggle`, fainted counts, active-move power/mult, **trapped /
  maybe_trapped**, matchup cells), and active-context volatiles. Capture-
  correctness for these is owned by each encoder's own unit test
  (`global_env_test.py`, `reactive_test.py`, pokemon/moves tests); here we only
  prove they are not dead obs regions.

## Pattern

```python
model, layout, _ = feature_model()
base    = anchor_delta(our_move_id="tackle")              # slot present
variant = anchor_delta(our_move_id="tackle", <edge field>)  # differs in ONE field
enc = encode_delta(variant)                              # CAPTURE: assert offset
assert_delta_reaches_network(model, layout, base, variant, "msg")  # NETWORK
```

If a signal does **not** reach the network, the failing assertion stays —
a dead obs dim is the most valuable thing this suite can find.

## Findings (as of writing)

- **No dead obs regions.** Every TurnDelta edge case and every static obs block
  swept here moves both the policy and value heads.
- **KO-before-acting** is encoded via `we_fainted` + an empty move block, NOT a
  `|cant|` reason (`"fainted"` is not in `CANT_REASONS`).
- **`frz` IS in `CANT_REASONS`.** `constants.py` abbreviates the list as "(full
  paralysis / sleep / flinch / recharge)"; that is prose, not coverage.
- **`ability: Damp` is NOT**, and it is a gen3-reachable `|cant|` reason — so a
  Damp mon blocking an Explosion CRASHES `state_encoder.encode`. Live defect, see
  gaps §3.7.
- **The event-window MOVE magnitude sums residual damage** (sandstorm/burn/
  confusion/recoil) into the attacker's row: the tracker's `[from]`-clause guard
  reads `value["from"]` but the parser writes `value["reason"]`. Live defect,
  see gaps §3.5.
- **Status index order** (`_STATUS_ORDER`): BRN, FNT, FRZ, PAR, PSN, SLP, TOX.
- **Boost stat order** (`BOOST_STATS`): atk, def, spa, spd, spe, accuracy,
  evasion.

## Run

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m pytest src/agents/model/feature_coverage/ -q
```
