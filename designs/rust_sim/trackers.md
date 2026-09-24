# The Rust core's TRACKERS, its native WINDOW RECORD, the label and the reward (M3)

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/trackers/`; the program is
`designs/endstate/program_rust_core.md` §2 M3. -->

The Rust Core Program's M3 (`gen3_core_trackers_v1`, `gen3_core_window_record_v1`). Built alongside
the Python path: **training reads none of it** (the cutover is M6). The per-decision state the
observation and the α/β labels need is folded ON THE VERSION from one side's stream, so a search
fork shares its parent's and pays only for its own decision.

| | |
|---|---|
| **Trackers** | `src/rust_sim/src/trackers/mod.rs` (`SideTrackers`, `TrackerState`, `IntentLabel`, `reward`), `history.rs` (recency, pair history, the event window, the wish and sleep folds), `clock.rs` (the progress clock), `hp_belief.rs` (the Hidden-Power belief), `delta.rs` (the context + the `TurnDelta` PROJECTION), `turnview.rs` (the frozen per-side turn fold), `ev.rs` (typed accessors over a reading) |
| **Native record** | `src/rust_sim/src/trackers/record.rs` (`Window` = ordered `Action`s, each with ordered `Effect`s; `Choice`, `DenialWhy`) |
| **Version** | `SideStream::with_trackers` / `BattleVersion::{root_with, observe_root_with, parse_root_with}`; `BattleVersion::{decision, trackers, note_choice}` |
| **Gates** | slice T (`agents/battle/rust_core_parity_trackers.py`, COMMIT + MILESTONE in `rust_core_parity_test.py`) |

---

## 1. What crosses, and what does not

`EpisodeTracker.record_context` + `advance_window` (`agents/training/episode_tracker.py`), line for
line, over the side's READINGS (the `BattleEvent`s slice E holds equal to `Gen3Battle`'s):

| Python | Rust | read by |
|---|---|---|
| `SlotRegistry` ×2 | `delta::Slots` | the context, the event window's joins |
| `HiddenPowerTracker` + `_maybe_observe_hidden_power` / `_scan_opp_movesets_for_no_hp` | `hp_belief::HpBelief` (priors from `data/pokemon/gen3_hidden_power_priors.json`) | the HP-type obs block |
| `ProgressClock` (the obs half: `n`, `value()`) | `clock::ProgressClock` | `turns_since_progress` |
| `RecencyTracker` | `history::Recency` | the per-mon recency triplet |
| `PairHistoryTracker` | `history::PairHistory` | H-A last action + the 6×6×5 pair block |
| `EventWindowTracker` (32 rows) | `history::EventWindow` | the H-B event window (`history_events`, ON in production) |
| `wish_belief.build_wish_pending` | `history::WishFold` (incremental) | the wish slot |
| `sleep_belief.build_sleep_sources` | `history::SleepFold` (incremental) | the sleep-belief triple's inputs |
| `opp_intent_labels.build_opp_intent_label` | `IntentLabel` (ids; the num tables cross at M4) | the α/β heads |
| `Gen3RewardManager` under the win indicator | `trackers::reward` | the value target |

**Not ported, by decision:** the SHAPED reward terms (program M3: every win-prob-era run trains on
`1 TERMINAL + 0 PBRS + 0 BIAS`), and with them the progress clock's CHARGE (`last_penalty`,
`switch_legal`); the choice-band belief (`ChoiceBandTracker` has no production reader — it has been
a no-op since it landed, `9a37b712`). **Not a first-class structure:** `TurnDelta`. Its layout is
frozen and its obs frames were deleted (`gen3_frame_deletion_v1`); the core keeps only
`delta::DeltaProjection` — the fields the clock and the label read — and slice T gates those
CONSUMERS, never the layout.

**One rule the port had to take from poke-env, not from the decision window.** The Hidden-Power
belief observes `battle.opp_last_damaging_move`: poke-env's PENDING damaging move (captured at the
`|move|` of a Physical/Special move), PROMOTED when an effectiveness emission for the defender
lands in the same turn (`_set_effectiveness`, incl. Flash Fire's `|-start|`), TURN-GATED to the turn
that just resolved. That spans the whole turn, so a window that opened at a mid-turn forced switch
does not contain it. The core keeps the same two slots on `BoardReading`
(`pending_damaging` / `last_damaging` / `last_damaging_move`). A window-scoped reading (the rule
`view_successor.view_context._to_dme` uses) diverged from training at 57 of 1,838 COMMIT decisions
before this port — see §5.

## 2. On the version — shared by a fork

A side's `SideStream` carries an optional `TrackerState` (`SideStream::with_trackers(cfg)`): the
`Arc<SideTrackers>`, the side's native-record builder, and the readings since its last decision.
Every line is folded as it arrives; a `|request|` that opens a DECISION of this side — non-empty,
not `wait`, battle unfinished, a legal action (the live player's dispatch, the same rule slice V's
`decision_points` uses) — runs `record_context` → `advance_window` on the readings since the
previous decision, closes the native window, and stores `Decision { line, window, reward }`.

A fork CLONES the `TrackerState`: the tracker state is an `Arc` and is COPIED only when the fork's
own transition opens a decision (`Arc::make_mut`); the record builder and the pending readings are
small and the fork's own. `BattleVersion::decision(side)` is `Some` iff the transition INTO the
version ended at one of the side's decisions. The trackers are opt-in (`root_with` /
`observe_root_with` / `parse_root_with`); search does not turn them on until M4's encoder reads them.

## 3. The native record — what happened, in order, with attribution

**Status: BUILT, emitted per decision by `core_events --trackers` (`window`), read by nothing yet;
its per-mechanic constructed fixtures and the loss catalogue land in the next commit.**

`record::Window` = the side's decision window as an ORDERED list of `Action`s, each with its ordered
`Effect`s (`on`, `what`, `cause`, `of`). Built from the side's typed lines (the `[from]` / `[of]` /
tag truth the protocol prints) and the side's reading of the board at each line. Nothing is
flattened:

| action | carries |
|---|---|
| `Move` | user, move id, the CALLER (`called_by`: Sleep Talk / Metronome / Mirror Move / Assist / Nature Power / Snatch), the target, `pursuit_on_switch`, `locked` (Thrash / Outrage / a charged move's release), `still` |
| `Switch` | the entrant, the mon it replaced, and WHY it entered: `Lead`, `Chosen`, `Replacement { fainted }` (the FREE switch after a faint), `BatonPass { passer, boosts, volatiles }` (exactly what was passed) |
| `Drag` | the dragged-in mon, the mon it replaced, WHO forced it and with what move (Roar / Whirlwind) |
| `Cant` | a REFUSED action — the reason (`par` / `slp` / `frz` / `flinch` / `recharge` / `Focus Punch` / `Taunt` / `Disable` / `damp` / `nopp` …), the move it was prevented from, who blocked it, the choice as this side knows it |
| `Denied` | a chosen action that never happened because the actor FAINTED FIRST — the cause and the action that denied it, placed right after that action (move ORDER) |
| `Refused` | the server refused a choice (`\|error\|`: trapped, disabled …) |
| `Residual` | the end-of-turn block |

Effects: damage / heal / set-HP with the attributed CAUSE (`Spikes { layers }` — the layers the
entrant met — weather, poison / toxic / burn, recoil, Leech Seed, Curse, Nightmare, `Wish { wisher }`,
an item, an ability, a move), `SubstituteHit { broke }`, `Blocked` (a move stopped by Protect /
Detect — its target lost), faints with their cause (self-KO, Destiny Bond, Perish Song, or the last
damage), status and cures, boosts, item transitions with direction (`to` on Trick / Thief / Covet),
side conditions with their layer count, crits, misses, fails, effectiveness PER HIT (a multi-hit
move's hits are separate effects), charge (`Prepare`) and recharge.

🚨 **The information boundary is a TYPE.** A viewer knows its OWN denied choice (it sent it —
`BattleVersion::note_choice`) and only THAT the opponent was denied. `record::Choice` is
`Own(Option<String>)` or `Opp` — `Opp` carries no data, so no code path can put the opponent's
chosen move into a viewer's record, and a record built from one side's stream sees nothing the
other side cannot (`present()`'s guarantee, for the record).

## 4. Slice T — the gate

`agents/battle/rust_core_parity_trackers.py`, on the same `core_events` replay as slices E and V
(`core_events --trackers`): at every decision of both viewers, the real `EpisodeTracker` driven as
`Gen3Env.embed_battle` drives it over a `Gen3Battle` fed the viewer's text, serialised into the
core's shape and compared TYPE-strict, EVERY differing leaf reported (one bad field never hides
another), **no allowlist**: slots, the HP belief (every float32), the clock (`n`, its inputs,
`value()`), recency, pair history, the 32 event-window rows (every column), the wish / sleep folds,
the `TurnDelta` projection, the α/β label (kind, move id, switch species, switch slot), and the
reward (win indicator) at every decision and at the end. A decision on one side that the other did
not take is an `[ALIGN]` divergence, never a skip. Teeth (`rust_core_parity_test.py`): a residual
folded into a move's `hp_delta` (the v81 class), a clock that never resets, a phaze labelled as a
choice — each FAILS.

| tier | runs |
|---|---|
| COMMIT | `python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (unmarked) |
| MILESTONE | `… -m slow -q -n 2` (the verdict lands in `designs/ops/slow_tier_status.json`) |

## 5. Findings

- **The search roads' Hidden-Power input is window-scoped** (`view_successor.view_context._to_dme`,
  used by the view and core roads' Python successors): a decision whose window opened at a mid-turn
  forced switch misses the opponent's promoted damaging move of that turn, which training observes.
  A depth ≥ 2 search difference (a depth-1 successor's window is the whole ply); the core's own
  trackers take poke-env's turn-spanning rule. Not fixed (search's Python successor is on the M4
  deletion path).
- **The engine does not model Metronome, Mirror Move, Assist or Nature Power** (`scan_move_probe`:
  `panic`); their record fixtures are PARSE-path (hand-written protocol), not step-built battles.

The loss catalogue — every case the frozen `TurnDelta`, the α/β label and the 22-column event window
flatten or lose, with rates over the MILESTONE corpus — is in
[`../research_state/measurements/rust_core_m3_2026-09-24/`](../research_state/measurements/rust_core_m3_2026-09-24/README.md).
