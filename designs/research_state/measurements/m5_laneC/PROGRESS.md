# M5 Lane C — the training labels in the Rust env: PROGRESS (resume point)

Lane C of `designs/endstate/program_rust_core.md` §2 M5. It owns these files:

- `src/rust_env/src/labels/`
- `src/agents/training/rust_env_labels_parity_test.py`
- the inventory: `src/utils/rust_env/label_inventory.py`, `src/agents/training/rust_env_label_inventory_test.py`
  and `designs/rust_sim/env_labels.md`
- the lane-C rows of `src/utils/rust_env/columns.py` / `protocol.py` (hand-off)

**Gate:** the slice-N label columns equal `Gen3Env`'s production keys per decision. The COMMIT tier
runs in the routine gate; the MILESTONE tier is `slow`.

## Units

| # | unit | status | commit |
|---|---|---|---|
| 1 | the INVENTORY (30 keys, 21 in production) + its routine test + `designs/rust_sim/env_labels.md`; ARCHITECTURE.md §7 corrected (the intent labels are ON in production; 9 off-surface keys added) | LANDED | see git log (`M5 Lane C unit 1`) |
| 2 | the GENERATED label columns (18, one per `core` row; `i64` added to the table) + the spec's `labels` declaration, refused at startup by kind (host / off-surface / unknown / not yet built); pin `src/utils/rust_env/label_columns_test.py` | LANDED | see git log (`M5 Lane C unit 2`) |
| 3 | family `belief` (`belief_species`, `belief_moves`, `known_moves`) + the parity harness `rust_env_labels_parity_test.py` (COMMIT 8 pool episodes / 843 decisions routine; MILESTONE `slow` 200 pool + 200 ladder episodes, 31,564 decisions, 94,692 key compares, 0 divergences; teeth: one cell of `belief_moves` moved fails 73/73) | LANDED | see git log (`M5 Lane C unit 3`) |
| 4 | families `hp_type` + `item` (`labels/per_slot.rs`); the same slice, now 7 keys per decision: MILESTONE 400 episodes / 31,564 decisions / 220,948 key compares, 0 divergences; per-family teeth (one moved cell fails every decision) | LANDED | see git log (`M5 Lane C unit 4`) |
| 5 | family `spread` (`labels/spread.rs`: the derived stats + a port of `invert_nature_evs`, its tables read from the stamp's `data/pokemon`, the inversion cached per episode per side); 13 keys per decision: MILESTONE 400 episodes / 410,332 key compares, 0 divergences; teeth on `belief_ev` | LANDED | see git log (`M5 Lane C unit 5`) |
| 6 | family `intent` (on `trackers::IntentLabel`) | NEXT | |
| 7 | family `margin` (`material_margin` on `present()`) | | |

## The inventory headline

The table of record is `src/utils/rust_env/label_inventory.py`. Production is
`trainee_env_kwargs(production_args())`, EXECUTED, not read from a doc. It emits 21 keys:

- **18 `core`** (the env core computes them per decision):
  - belief: 3
  - spread / nature / EV: 6
  - HP type: 2
  - item: 2
  - intent: 4
  - margin: 1
- **2 `host_const`**: `win_target` / `win_mask` = 0.0, back-filled by `WinProbLabelCallback`.
- **1 `host_episode`**: `opp_class`.

Nine keys are off the production surface:

- `win_row_w` and `fork_pg_m` are host constants of 1.0.
- `opp_true_team`, `aux_target`, `aux_mask`, `aux_turn`, `defensive_opportunity`,
  `bait_opportunity` and `distill_mask` are REFUSED.

## The parity harness (unit 3, BUILT: `src/agents/training/rust_env_labels_parity_test.py`)

Record in Rust and replay in Python, the direction Lane 0's gate ① uses:

1. The core runs a battle with a seeded random policy. The harness records every label column at
   every p1 decision, with the input log (seed, teams, each side's `CHOOSE` token, and p1's action
   index).
2. `Gen3Env` replays that battle with the production kwargs on the rust bridge:
   - teams from a `SequenceTeambuilder`;
   - the seed through the bridge session's `seed`;
   - the trainee's recorded action indices;
   - an opponent that sends the recorded p2 tokens verbatim.
3. Every production key the core computes is compared at every trainee decision.

Details as built:

- The core runs through the FFI front end (N = 1, T = 1, `labels` = the built families).
- The Python opponent consumes a recorded p2 index only when `agent2_to_move`: the wrapper also
  calls `choose_move` on steps whose action is never sent.
- At every trainee decision, the `observation` row must equal the core's row first. That is the
  alignment check.
- Decision counts must match per side, unless Python forfeited at the stall threshold. The core
  plays with `turn_limit` 1000, so an episode Python forfeits is compared up to the forfeit.
- A new family is compared by adding it to the test's `BUILT` (and to `labels::BUILT` in Rust).

The core computes labels in `pool.rs` `advance`. Its fold order changed: every side's chain now
folds the write before any side's decision is encoded and labelled, because a label reads the
OTHER side's chain. The rows are unchanged, and gate ① stays green.

## Open findings

- **F-LC-1 (fixed in unit 1):** `Gen3Env(emit_opp_intent_labels=True)` without belief labels raised
  `UnboundLocalError: _imax` at construction. It was unreachable in production, where the belief
  labels are on. `_imax` is now hoisted (hand-off line in `gen3_env.py`).
- **F-LC-2 (corrected in unit 1):** ARCHITECTURE.md §7 and the program doc's inventory row said
  the four intent labels were OFF in production. They are ON: `opp_intent: true` ⇒ `--arch
  production` sets `opp_intent_coef` 0.05. The intent CE consumes them. The slice-N cutover test's
  key check (`slice_n_test._assert_clean`) does not name an intent key. Slice N compares every
  emitted key anyway, so they WERE compared; only the explicit presence check omits them.
- **F-LC-3 (closed by unit 3's gate):** Showdown's request spells `return<bp>` /
  `frustration<bp>` (`sim/pokemon.ts` `getSwitchRequestData`). Both readings key an own mon's
  moves by `Move.retrieve_id`, which folds the power, so the label sees `return`. The milestone
  tier runs 400 episodes with zero divergences. The pool has 53 Return / Frustration teams out of
  719, and the ladder 1,422 of 22,813. Per-episode coverage of those teams was NOT counted.
- **F-LC-4 (closed by unit 4's gate):** the Python env reads `battle2` at the trainee's step, and
  the question was whether the truth side's timing matches. The item label is the first dynamic
  truth field. Across the milestone's 400 episodes, a revealed slot's item label changed 65
  times in 55 episodes; 61 of those changes went to "nothing" (a consumed berry, Knock Off). Every
  decision is equal on both paths. `/tmp`-script measurement, 2026-09-29: the recorded core runs
  of the two milestone streams.
- **F-LC-5 (a label-coverage FINDING, not a parity one):** `invert_nature_evs` assumes IV 31 on all
  five stats. On the TRAINING POOL, **54.6 % of revealed-slot decisions carry no nature / EV label**
  (mask 0): 46,678 of 85,554 over the milestone's 200 pool episodes. On the ladder corpus the figure
  is 1.2 % (1,168 of 95,812). The measurement is a `/tmp` script over the recorded core runs,
  2026-09-29. The likely cause is the pool's Hidden-Power IV adjustment (`Gen3Teambuilder`); 711 of
  719 pool teams run HP. **UNVERIFIED** as the cause. The Rust env reproduces it exactly, as a
  parity port must. Whether the nature/EV head should be supervised on those mons (invert with the
  set's real IVs, or read nature / EVs from the truth team directly) is an owner / research call.
  It is not Lane C's.
- **F-LC-6 (a latent Python cache bug, not reached by the gate):** `Gen3Env._nature_ev_map` keys its
  per-battle cache by the opponent team's SPECIES SET alone. A next episode whose opponent has the
  same six species but different spreads reuses the previous battle's inversion, which is a wrong
  label. The Rust env recomputes per episode, so it is correct there, and the two would diverge on
  such a pair. The frequency is UNMEASURED; it needs two pool teams with the same species set
  drawn back-to-back for one env. Proposed fix: key the cache by `(species, stats)` per mon. This is
  a training-input change and is NOT applied.
