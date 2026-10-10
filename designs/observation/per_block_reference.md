# The observation row, block by block — what each field MEANS and where it comes from

**Moved from `src/agents/observation/CLAUDE.md` on 2026-10-10.** This doc OWNS its subject and carries the same
ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the Rust encoder
(`src/rust_sim/src/encoder/`) and `src/agents/observation/constants.py`. The top-level block table (block → dims →
offset) is [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §1, which wins any disagreement. Index: [`README.md`](README.md).

## Moved from the leaf (2026-10-10)

`designs/ARCHITECTURE.md` § Observation carries the top-level block table (block → dims → offset)
and the per-mon slot layout, derived from the live constants. **This** is the per-block detail:
what each field MEANS and where it is sourced from. All offsets are computed from named constants
— never hardcode indices.

🚨 **Read the SOURCES below in the Rust core's terms.** This reference was written when the Python encoder
wrote the row; the semantics are unchanged (the obs golden is byte-identical across the deletion), but every
producer it names — the Python sub-encoders' `encode`, the `EpisodeTracker`-owned `RecencyTracker` /
`PairHistoryTracker` / `EventWindowTracker` / `ProgressClock`, `HiddenPowerTracker`, the assembler, `Gen3Battle`'s
event log — is now its Rust twin (`src/rust_sim/src/encoder/`, `src/rust_sim/src/trackers/`,
`src/rust_sim/src/present/`; `designs/rust_sim/encoder.md`, `designs/rust_sim/trackers.md`). The fuzz gates it names
under `training/poke_env_gaps/` and `action/` were deleted with the Python stack (slices 6c / 6d-2); the row is
held by the engine-truth audit, the round-trip chain, two-roads-one-row and the frozen goldens
(`designs/rust_sim/encoder.md` §6a).

**Per-Pokémon slot (122 dims):** the 110 below + the 3-dim recency block + the 1-dim
protect-odds field + the 6-dim last-action block + the 2 appended trapping bits + the
appended active flag.
**Recency block** at `POKEMON_RECENCY_OFFSET` (109) — [turns_since_seen, turns_since_acted,
turns_since_was_hit], TURN-ANCHORED (`cur_turn − event_turn`, clamped; on-field mon reads 0;
never-tracked reads 1.0 max staleness), log-saturated over a 10-turn cap, BOTH sides (public —
every reset derives from observed protocol events), folded by the Rust trackers over the
per-decision event window. **Protect-odds field** at
`POKEMON_PROTECT_OFFSET` (112, gen3_entity_rehome_v1): P(a Protect/Detect/Endure by THIS mon
succeeds now) under the gen3 floored-doubling stall rule (100/50/25/12.5, floor 1/8), from the
LiveView `protect_counter` — EVERY mon owns its stall state (a benched mon truthfully reads 1.0;
the counter resets on switch). **Last-action block** at
`POKEMON_LAST_ACTION_OFFSET` (113, `gen3_pair_history_v1` — Tier H-A1 of
`designs/ai_v9/design_history_entity.md`): the SIDE's most recent executed action on its
ACTIVE mon's slot — `[last_move_id, was_switch, hit, miss, fail, crit]`, bench rows zero.
The move id is an EMBEDDING id (the model's `slice_pokemon_categoricals` routes it to the
move table and ZEROES its raw column — a dex num never reaches a Linear); outcome order
matches the turn-delta `_OUTCOME_ORDER`; CANT windows leave the previous action standing;
leads don't count (a placement, not an action). Folded by the Rust trackers (the same decision
window as recency). The SAME tracker also feeds the **180-dim pair-history block** after reactive
(`OFFSET_PAIR_HISTORY`, 6×6×5 `h[i,j]` tendency counters — switch-ins/attacks/status-clicks
by their mon i while our mon j was active, shared-field turns, pairing recency; log-saturated
over the 10 cap; consumed by the opt-in `h` edge family). **Tier H-B follows it**
(`gen3_event_window_v1`, v81): the **event window** (`OFFSET_EVENT_WINDOW`,
`EVENT_WINDOW_N` × `EVENT_TOKEN_DIM` typed event records) closes base, and
**`gen3_frame_deletion_v1` made it the LAST block**, and **`gen3_event_record_v2` (E12) reshaped
its row to 30 columns** — entry reason / REL mon / denial / caller / boost stat / Spikes layers /
Pursuit-on-switch, a DENIED row type (fainted first; the gen-3 TURN CUT), the refused-switch TARGET
(E4) — with the schema in `designs/ARCHITECTURE.md` §1.6 and the fold in the Rust core
(`trackers::history::EventWindow`). One constructed battle
per mechanic, replayed through the core: the cargo `src/rust_sim/tests/window_record_test.rs`
(`agents/battle/event_record_v2_fixture_test.py`, its Python-vs-core twin, was deleted in P6 slice 6c). With the frame deletion,
the 11-dim prev-turn action mask and the 7 × 159 TurnDelta lag frames that used to follow are
DELETED, so `total_dim == base_dim`. The window grew a
`cant_id` column in the same pass — the one lag-frame fact with no substitute. What that
deletion cost, and the three facts that ship WITHOUT a substitute, is
`designs/ai_v9/design_frame_deletion_coverage_gaps.md`. Folded by the Rust core's `EventWindow`
(same window, same alive-filtered resync); rows most-recent-LAST, front zero-padding; ids are embedding ids and
NO Linear reads the block raw (its only consumer is the opt-in `--history-events` event seats).
MAGNITUDE is stage-scaled (`/ 6`) on a BOOST row ONLY — signed, a drop negative — and a HAZARD row
writes its ±1 (condition started / ended) as is (`gen3_event_window_semantics_fixes_v1`; the rest of
that fix — the Protect OUT_FAIL, the current-mover damage rule, the no-lethal-line faint, Trick
SWAPPED — lives in the fold, `designs/ARCHITECTURE.md` §1.1).

> **The per-row COLUMN CONTRACT is `constants.EventCol`** (`gen3_event_col_names_v1`) — an
> `IntEnum`, ONE declaration the MODEL reads, held equal to the PRODUCER (the Rust encoder's `layout.rs`
> `EV_*` columns) by `rust_core_obs_layout_test.py`; its Python consumers are
> `team_transformer.EventSeats.forward` + `_event_reference_cells` and the feature-coverage probe helper
> (`feature_coverage/_support.py::obs_with_event_row`). It replaced a comment plus ~30 bare integer literals
> spread across five files — a producer/consumer pair bound by POSITION with nothing relating
> them, the class the 2026-08-18 positional-binding sweep convicted five times. **Never write a
> bare column index**; the members ARE ints, so `vec[_o + EventCol.CRIT]` is the same arithmetic
> and the same emitted bytes (byte-identity confirmed on the 991-decision golden capture).
> The two CONTIGUOUS one-hot groups have their own names (`EVENT_OUTCOME_GROUP`,
> `EVENT_EFF_GROUP`) because both are written by INDEXING (`EFF_NEUTRAL + eff`), so their order
> is load-bearing: reordering a member relabels every historical row with no shape change.
> `event_window_test.py` pins all of it — the members TILE `range(EVENT_TOKEN_DIM)` with no gaps
> or overlaps, the groups are contiguous and in the TurnDelta effectiveness-code order, the consumer
> imports the SAME object, and `EventSeats._N_SCALARS` (a weight shape) agrees with the
> map's id/scalar classification.

Unit gate: `training/event_window_test.py` (the contract); the fold's gates are the Rust core's
(`tracker_semantics_test.rs`, `window_record_test.rs`). **Appended tail**: `POKEMON_TRAPPED_OFFSET` (119) + `POKEMON_MAYBE_TRAPPED_OFFSET` (120) — the
OUR-side LegalActions trapping bits, nonzero ONLY at our active slot (`maybe_trapped` is the
high-value trap-risk bit; bench slots stay zero) — then the ACTIVE flag at `POKEMON_ACTIVE_OFFSET` (121), deliberately
LAST in the slot (the model's `hp_and_active[:, :, -1]` convention is load-bearing).
Original 110: species ID + 6 base stats, item ID + known + consumed, 2 type
IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction,
species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**,
**HP-candidate block (17 dims)**, **sleep-wake belief (3 dims)**. The item block is 3 dims:
`[item_id, known, consumed]` — `consumed=1` when the item was spent this battle (Berry
activated, Knock Off, Trick, etc.) and `item_id` retains the identity of the consumed item so
the model knows what was lost. `species_known = 1.0` for all populated slots (own team and
revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter:
`min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(stage, 15) / 15` (`TOXIC_STAGE_MAX`, held equal to `layout.rs` by the layout gate), where `stage`
is the sim's `tox` stage — the residual `[from] psn` chips since the switch-in, 0 while benched
(`gen3_pe_reading_fixes_v1`; it used to tick at `|turn|`, one off the sim after a post-residual
entry or before the next `|turn|`) — so the NEXT chip is `min(stage + 1, 15) / 16` exactly (Showdown's `tox.onResidual`:
`if (stage < 15) stage++` BEFORE the damage `floor(maxhp/16) * stage`; `damage_op_pairwise.toxic_next_tick` decodes it).
It was `min(stage, 8) / 8` until `gen3_toxic_stage_scale_v1` (config v151): every tick from the 9th on read as the 8th.
`toxic_stage_core_test.py` pins stage 12 end to end through the real core; `obs_stage_truth_test.rs` holds the cell to
the engine's `Toxic(n)` at every decision.

**Sleep-wake belief (3 dims, `gen3_sleep_wake_belief_v1`, layout in `sleep_belief.py`):** zeros
unless the mon is asleep, else `[sleep_is_deterministic, p_wake, sleep_counter_reliable]`. The protocol
exposes only `Status.SLP` + a noisy `status_counter`, NOT the rolled duration / remaining time / source
move, so a policy reading the raw counter would have to LEARN the gen3 sleep RNG and can't tell a
deterministic Rest from a random opp-sleep at the same counter. We **compute** the wake odds from the
adversarially-verified gen3 tables — opp `time = random(2,6)` ∈ {2,3,4,5} (the gen3 mod overrides the
modern `random(2,5)`), Rest `time = 3` fixed, Early Bird halves — `p_wake` = P(wake on the next move
attempt | observed counter K, source, Early Bird), **marginalising the opponent's Smogon Early-Bird
prior** (collapsing to exact 0/1 for our own mon or a revealed opp). `sleep_is_deterministic` (1.0 =
Rest) selects which table; it's read from the **event record's `[from]` clause** (the Rust sleep tracker, `trk.sleep.sources` in `encoder/slot.rs`).
`sleep_counter_reliable` drops to 0.0 once a Sleep Talk / Snore turn has corrupted the counter (+3 per
turn, empirically verified) — instead of reconstructing Showdown's `skippedTime` switch refund. The
counter→p_wake mapping and the source/reliability bits are **fuzz-calibrated against the real sim RNG**
(the deleted `sleep_wake_fuzz_test.py`: per-decision obs wiring exact + empirical wake-frequency ==
the computed table across well-sampled (K, source) buckets).

**Move slot (11 dims, layout in `moves.py`):** move ID, base power (/200), has_secondary,
has_recoil, type ID, category (0=status, 1=physical, 2=special), known flag, current PP
(/MAX_PP), max PP (/MAX_PP), accuracy (raw% / 100), never_miss bit. Accuracy is split into a
continuous scalar plus a categorical bit: never-miss moves carry accuracy=100 in the mapping →
encode as `[1.0, 1]`, while a genuine 100%-accuracy move is `[1.0, 0]` — same scalar,
distinguished only by the bit. A 100%-accuracy move can still miss into evasion (Double Team)
or after Sand-Attack; a never-miss move (Swift, Aerial Ace, all status/self moves) bypasses the
accuracy/evasion check entirely.

**Spread block (18 dims, appended at offset 71 within each slot):** IVs ×6 each/31 + EVs ×6
each/252 + spread_known (1.0 own, 0.0 opp) + nature modifiers ×5 [atk, def, spa, spd, spe] as
raw floats (0.9/1.0/1.1). Opponent slots have all 18 dims as zeros; `spread_known=0`
distinguishes "unknown opponent" from "own Pokémon with 0 EVs". Own-team `mon.ivs/evs/nature`
are populated by the Rust reader's **`backfill_teambuilder_spread`** (`src/rust_sim/src/present/board_reading.rs`, ported from the deleted poke-env fork's `Battle.parse_request` backfill):
gen3ou has no team preview, so `apply_teambuilder_team` never attaches the spread — the backfill
matches the declared teambuilder team to the request-built team by species and fills in
IVs/EVs/nature (spread only, never re-running `_update_from_teambuilder`). Without it this block
emitted a constant fallback (all-31 IVs, 0 EVs, neutral nature) for every own mon.
**OBS-FACTS block — 84 dims, the observation's LAST block at `OFFSET_OBS_FACTS` = 2761 (`gen3_obs_facts_v1`,
`obs_facts.py`; Rust twin `src/rust_sim/src/encoder/facts.rs`; layout `constants.FACTS_*`).**
The Rust encoder writes it every decision (`encoder/facts.rs`; the Python `Gen3ObservationEncoder.encode` →
`encode_obs_facts(vec, OFFSET_OBS_FACTS, live, our_species, event_window)` path was deleted in T27 P6) (appended at the X5 version break, config v144, part 3: obs 2761 → 2845,
the 2761-dim prefix byte-identical). The model reads it only under `--obs-facts v1`
(`agents/model/obs_facts_inject.py`; production `off` reads none of it;
`designs/endstate/design_entity_coverage_audit.md` §8). Four sub-blocks (offsets INSIDE the block; the
layout's `obs_facts` key and the schema's `obs_facts.{seen,choice,vol,screens}` children), each laid out to
be ROUTED to an entity:

| sub-block | offset | dims | content |
|---|---|---|---|
| `seen` | 0 | 6 × 7 | per OUR mon (row i = our team slot i): `[on_field_once, move_seen ×4, item_public, ability_public]` — what the OPPONENT has seen of it (backlog E1). The four move bits follow `LivePokemon.moves` (sorted by key), which is the per-mon slot's sorted move order |
| `choice` | 42 | 4 | the OPPONENT ACTIVE: `[not_locked_by_item, not_locked_by_moves, stint_first_move (embedding id), stint_run (SAT_LUT)]`; zero when it is fainted or absent |
| `vol` | 46 | 2 × 5 × 3 | per side (ours, theirs), the active's `encore, taunt, disable, uproar, partiallytrapped`: `[elapsed, min_left, max_left] / 8` (`FACTS_VOL_DURATION`; `min_left ≥ 1` while present) |
| `screens` | 76 | 2 × 4 | per side: turns left / 5 on `reflect, light_screen, safeguard, mist` |

Sources: the view (`LiveMove.seen`, `item_public`, `ability_public`, `residual_done`, the volatiles'
`|turn|`-counted counters, the side conditions' start turns) and the event window's `facts` fold
(the Rust `trackers::facts`; the Python `obs_facts_fold.py` / `EventWindowTracker` were deleted in T27 P6). **Elapsed is RESIDUALS**: the counter (or `turn − start`) plus one when
`residual_done`. With no window the stint reads zero and the Encore / Disable bounds take the union of
both adjustments. Gates: `obs_facts_test.py` (constructed protocol, every fact; each FAILS on revert of
the code it names; the block's offset and the encoder's write on both schedulers), the core's corpus test
(`agents/battle/core_corpus_test.py`: `obs_facts` is one of the blocks every replayed row must see nonzero), the
obs golden (the core's row carries the block), and the ENGINE truth test `src/rust_sim/tests/obs_facts_truth_test.rs`
(the Python-vs-core slices O and V that used to compare the block and rules V18 / V19 were deleted in P6 slice 6c).

**Board (reactive) block — 17 dims, layout in `reactive.py`.** `REACTIVE_SCALAR_DIM` (5) raw
board scalars, then the 12-dim active-req-moves block. Offsets are `reactive_layout` entries —
read them, never hardcode. **gen3_entity_rehome_v1**: the two 144-dim matchup matrices are
DELETED (pair effectiveness is GPU-side — the D/V edge families compute
`[low, high, crit, pko, type_mult, revealed]` cells from real physics + the learned belief);
`active_status` (redundant with the per-mon condition one-hot) and `forced_struggle` (derivable
from the all-zero `active_req_moves` legal bits / the action mask) are deleted;
protect/trapped/maybe_trapped moved to the per-mon slots (above).

| Field | Offset | Dims |
|---|---|---|
| `fainted` (ours, theirs) | 0 | 2 |
| `turns_since_progress` | 2 | 1 |
| `wish_floating_our` / `wish_floating_opp` | 3 / 4 | 2 |
| `active_req_moves` | 5 | 12 |

The 5 scalars sit BEFORE `active_req_moves`, so the extractor picks them up in
`non_matchup_rest` automatically (it stops at the req-moves offset). Sources:

- `turns_since_progress` — the log-saturated no-progress clock (`log(1+min(n,10))/log(11)`), kept by
  the Rust trackers (NOT the view — it is cross-turn state). (The reward's `no_progress_tax` keyed on the same clock until the
  shaped reward path was deleted, 2026-09-26; the clock is now an obs-only counter.)
**Do not confuse `turns_since_progress` with the DEADLINE clock** — they answer different
questions and live in different blocks. `turns_since_progress` (board block, above) is a
*resettable* stall counter: it measures how long since anything productive happened and it goes
back to 0 the moment it does. The forfeit deadline is the GLOBAL block's `clock` group
(`gen3_deadline_clock_v1`, `CLOCK_DIM` = 3): `[log_elapsed, remaining_linear, log_remaining]`,
where `remaining = MAX_TURNS − turn` clamped at 0 and `MAX_TURNS` (250) is the turn the trainee
actually forfeits on (`StallConfig.threshold` imports it — pinned by
`global_env_test.py::test_max_turns_is_the_forfeit_deadline`). Only the global group tells the
model how much game is LEFT; a reset progress clock says nothing about the cap.

The log-REMAINING channel exists because the old lone log-ELAPSED scalar put **58.6%** of its
range on turns 1–50 and **1.5%** across the last 20 — a 125× sensitivity gap at exactly the cliff
the critic has to price. Measured consequence before the fix (`ai_v9_09` @16M): a POSITIVE V(s)
on the final decision before a −30 forfeit in **13 of 14** timeout losses. log-remaining gives
those last 20 turns 55.1% of its range. Both remaining forms are raw facts, not a choice made for
the model.

- `wish_floating` — the pending-Wish heal: a flat `WISH_HEAL_FRACTION` (≈0.5; gen3 Wish heals the
  RECIPIENT's maxhp/2, so the fraction is constant and GIGO-proof) when the side's Wish WILL LAND at the
  NEXT end-of-turn residual, else 0 (`gen3_wish_flag_truth_v1`: it was "cast last turn", which is the same thing before
  this turn's residual and WRONG after it — a replacement decision after the end-of-turn faint carries the turn number
  its residual ran in, so the flag kept a Wish that had just landed and missed one cast that turn). A second cast while
  one is pending FAILS (`|move|…|Wish||[still]` + `|-fail|`) and is not a cast. Slot-keyed, so it survives faint /
  Roar-phaze / switch (the wisher's faint does not cancel it in gen 3). Reconstructed from the event record (the Rust
  trackers; the Python `wish_belief.py` fold is deleted) and held to the engine's own slot condition at every decision
  by `tests/obs_stage_truth_test.rs`.

**`active_req_moves` (12 dims):** OUR active mon's 4 moves in **REQUEST order** (slot *k* ↔ action
logit 6+*k*) — `[move_num ×4, resolved_type_id ×4, legal_now ×4]`, sourced from `legal.move_slots`,
the same source as the action mask. The `DamageOperator`'s OUTGOING per-move methods read THIS, so
their per-move output aligns with the action order instead of the per-mon block's sorted-by-id
order. `move_num` is the dex num (the opponent's HP stays bare 237; our own typed HP resolves);
`legal_now` is the current-decision choosability. These are embedding IDs, not scalars, so the block
sits AFTER the matchups and is EXCLUDED from `non_matchup_rest` — `ObsUnpack` slices it explicitly
into `ctx.our_active_req_move_{ids,type_ids,legal}` and it never enters the raw-scalar path.

**The per-mon move block stays sorted-by-id on purpose** — it feeds the role token, whose value is
order-sensitive (the 4 move encodings are concatenated), so it cannot be reordered without changing
the network. Both orders are therefore live at once (the per-mon slots are sorted by `Move.id`
STRING, the request block by request slot), and the model crosses them ONLY by move-num IDENTITY
(`extractor_ctx.active_request_sorted_match`, `gen3_move_legality_by_id_v1`) — the pointer head's
tokens and `PokemonEncoder`'s per-slot legality alike. A positional crossing is silent whenever every
move is legal; it was live in `PokemonEncoder` from `bcdd868b` to that fix. `agents/action/ordering_integrity.check_obs_move_order` RAISES (where `legal` is
real) when a request move has no unique sorted slot. With the Python encoder's `encode` deleted (T27 P6), the guard's
live call site is `InferenceService.submit` (`src/agents/inference/service/service.py`).

**Move-effect block and incoming-damage / OHKO belief block — BOTH DELETED from the observation.**
Their long descriptions moved verbatim to `designs/CHANGELOG.md` §5. Where the signal lives now:

| Deleted obs block | Dims | GPU home |
|---|---|---|
| action-aligned move-effect flags | 44 | static mechanics → the `MoveLatentEncoder` latent (`--move-latent`); board-conditional `status_will_land` → `DamageOperator._status_landing` (`--damage-outgoing`); `pp_fraction` → the per-mon move slot (unchanged) |
| per-our-mon incoming-damage / OHKO belief | 51 | the `DamageOperator`'s incoming block, off the LEARNED move belief instead of this block's FIXED usage prior — that substitution was the whole point of `--damage-op` |
| active-move scalars (base power ×4, type mult ×4) | 8 | the op's OUTGOING per-move block, request-ordered, with real gen3 physics rather than `bp/200` and `mult/4` |

**`agents/observation/incoming_damage.py` STAYS** — pure damage / KO / outspeed math (no poke-env), which the prober's
engine imports (`main/prober/engine/{switch_in,util}.py`); `incoming_damage_test.py` pins it. Its encoder half
(`incoming_damage_encoder.py`: `encode_block` over a `LiveView`, the content-keyed `IncomingBeliefMemo`) is DELETED
(T27 P6 slice 6d-2) — its last callers (the deleted reward PBRS, then nothing) were gone; the record is
`designs/training/reward.md` → *The belief-block memo*.

> **Downstream reader:** the prober engine (`src/main/prober/engine/`) resolves its obs
> offsets at runtime from `get_layout()` (`ObsOffsets`), with `0 = absent` for deleted blocks
> (`mm_off`, and since gen3_entity_rehome_v1 also `om_off`/`tm_off` — ThreatView/saliency
> no-op). Its pinned regression test (`prober/engine_test.py::
> test_offsets_resolve_matches_layout`) fails on any layout move; update the pins there.

**TurnDelta slot (159 dims, layout in `turn_delta_encoder.py`):** all offsets computed from
named `OFFSET_*` / `*_DIM` constants — never hardcode indices. These are the ARCHIVED lag frames
(`gen3_frame_deletion_v1` removed them from the live row); `TurnDeltaEncoder` survives as the prober's
decoder for archived runs and the feature-coverage probes' encoder. The `TurnDelta` record is folded from
the event record by the Rust core (`trackers/delta.rs`; the Python fold is deleted, T27 P6 slice 6d-2).

- **Base block (53 dims, indices 0–52)** — our/opp move features (5 each: raw move_id int,
  power_norm, has_secondary, has_recoil, raw type_id int — **our OWN Hidden Power carries its DISTINCT
  num + real type** here, `gen3_typed_hidden_power_ids_v1`: the fold restores the typed HP id from the
  decision-time `LegalActions.own_hp_typed_id`, which maps to its distinct dex num (355-370) so
  `_move_features` takes the typed-dex branch [real BP/type]; the **opponent's** HP stays bare num 237 /
  type-0, correctly unknown), switched/failed flags, cant onehots
  (`CANT_DIM` = 12 ea, from `gen3_effects.CANT_REASONS`:
  slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant —
  source-derived, crash-don't-drop enforced in the encoder), summed HP deltas, faint flags,
  opp_move_known, effectiveness onehots (4 ea), move-order (2).
- **Extended block (106 dims, indices 53–158)** — our/opp boost deltas (7 each);
  `phase_is_forced_switch` (1); our/opp `target_hp_delta` (1 each); per-side HP-level vectors
  (6 each); our/opp target_status onehots (7 each, at move-fire time); our/opp move-outcome
  onehots (3 each: `[hit, miss, fail]`); our/opp move-crit (1 each); **gen3_turn_delta_v2
  additions**: `our_faint_causes`/`opp_faint_causes` (8 each, multi-hot over
  `attack/hazard/weather/status/recoil/selfko/leechseed/other`); **status-transition onehots**
  (4 × 7): `our/opp_status_applied` (status GAINED this window) + `our/opp_status_cured` (status
  LOST this window); **item-used bits** (2): `our/opp_item_used` — a single bit per side marking
  an item was consumed/removed this window (Berry/Knock Off/Trick). These (status transitions +
  item-used) are the per-turn *events*; the cause-**identity** (which item, which ability) lives
  in the per-mon item/ability block — the history carries the event, the block carries the what
  (parity with the collapsed `ability_activated` volatile). `our_attempted_move_id` (1, raw int
  embedded — the move we PRESSED, even if it never fired); **species block** (6 raw ints,
  embedded): `our_actor`/`opp_actor`/`our_target`/`opp_target`/`our_switch_to`/`opp_switch_to`;
  **gen3_trapping_signals_v1 additions**: `attempted_switch_rejected` (1 bit — the server
  REFUSED a switch we chose this window, `|error|[Unavailable choice]`, i.e. we tried to pivot
  and got trapped) + `our_attempted_switch_to` (1 raw int embedded — the mon we PRESSED a switch
  to).

`our_faint_causes` / `opp_faint_causes`: multi-hot — the rare 2-on-one-side window
(Pursuit/Future-Sight into a low-HP mon on hazards) can set 2 bits; the common Explosion
double-KO is one bit each side. Cause derived from the DAMAGE event's `[from]` clause
immediately preceding the FAINT in the event log. All-zeros when no faint.
`our_attempted_move_id`: decoded from the pressed action index at build time, preserved even
when the move never fired (cant / frozen / KO-before-acting). `attempted_switch_rejected` /
`our_attempted_switch_to` (gen3_trapping_signals_v1): the rejected-pivot history — folded from
the out-of-band `CHOICE_REJECTED` event (the deleted Python `TurnView.attempted_rejected`; the Rust trackers carry it now). On a rejected pivot
`our_switch_to` is the unknown sentinel (the switch never happened) while `attempted_switch_to`
names the mon we tried to bring in; both are zero on every turn with no rejection. Opp attempted
action is not observable. Faint *counts* are kept on the `TurnDelta` dataclass but
not encoded (no reader since the shaped reward was deleted) — redundant with the faint flags + cause popcount.

**Embedded-ID manifest (the layout-driven contract):** which slot positions carry raw embedding
IDs and which table each routes to is declared once in `TURN_DELTA_EMBEDDED_IDS` (in
`turn_delta_encoder.py`). The encoder's layout and the feature-coverage probes
(`agents/model/feature_coverage/_support.py`) read it, so a raw id can never silently leak through as a scalar. (The
model's consumer, `Embeddings.embed_delta_slot`, was DELETED with the lag frames, `gen3_frame_deletion_v1`;
`phase_modules_test.py` pins its absence.) Adding an embedded ID is
a one-line manifest change. Current manifest: 3 move IDs (our/opp/attempted) → 3×16, 2 type IDs
→ 2×16, 7 species IDs (our/opp actor + our/opp target + our/opp switch_to + attempted_switch_to)
→ 7×32 = 48 + 32 + 224 = 304 embedded dims + 147 pass-through scalars = 451-dim per slot
(the width the deleted lag-frame embedder produced). All zeros on the first turn of each episode. Actor
species resolution prefers `damaging_event.user_species` (protocol-truth) and falls back to
`prev_active` for switches and non-damaging moves; target species comes from the OTHER side's
`damaging_event.target_species`. Species ID 0 is the unknown sentinel.
