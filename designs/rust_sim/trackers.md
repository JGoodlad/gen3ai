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
| **Version** | `SideStream::with_trackers` / `BattleVersion::{root_with, observe_root_with, parse_root_with, parse_root_unrecorded}`; `BattleVersion::{decision, trackers, note_choice}` |
| **Gates** | slice T (`agents/battle/rust_core_parity_trackers.py`, COMMIT + MILESTONE in `rust_core_parity_test.py`; FRESH battles: `rust_core_trackers_fuzz_test.py`); the native record's fixtures `tests/window_record_test.rs`; the training-input SEMANTICS pins `tests/tracker_semantics_test.rs` (Rust) and `agents/battle/tracker_semantics_fixtures_test.py` (both sides through slice T) |

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

**ACTION DENIAL.** A chosen action that never happened is one of: a `Cant` with `then_moved == false`
(REFUSED), a `Denied { FaintedFirst }` (the actor fainted before its turn — found at the faint, placed
right after the denying action, with that action's mover and move), a `Denied { TurnCut }` (the
actor did NOT faint, but a faint earlier in the turn CUT it — below), or a `Blocked` effect on the
move whose target Protect / Detect took away. A turn's actors are the actives at its `|turn|` line;
a residual faint is never a denial (every actor has acted by then).

🚨 **The gen-3 TURN CUT.** In gen-3 singles ANY faint during the action phase cancels EVERY remaining
queued action (`sim/battle.ts` `faintMessages` → `queue.cancelAction` over all actives, ~2606-2616;
the port implements the same cut in `turn/switch.rs` and `turn/driver.rs`). So a faster self-KO —
Explosion / Self-Destruct, a recoil KO, a confusion self-hit KO, a Rough Skin KO — denies the
SURVIVOR's chosen action even though nothing touched it. The record closes the action phase at the
first residual, `|upkeep|`, `|turn|` or the battle's end (not at a mid-turn decision — the forced
switch's request comes before the phase ends), and inserts one `Denied { TurnCut { by_faint, cause } }`
per turn actor that neither acted nor fainted, right after the first faint's action (move ORDER).
Fixtures: `a_self_ko_explosion_cuts_the_turn_and_denies_the_survivors_move`,
`a_recoil_self_ko_cuts_the_turn_and_denies_softboiled`.

**One rule the port had to take from poke-env, not from the decision window.** The Hidden-Power
belief observes `battle.opp_last_damaging_move`: poke-env's PENDING damaging move (captured at the
`|move|` of a Physical/Special move), PROMOTED when an effectiveness emission for the defender
lands in the same turn (`_set_effectiveness`, incl. Flash Fire's `|-start|`), TURN-GATED to the turn
that just resolved. That spans the whole turn, so a window that opened at a mid-turn forced switch
does not contain it. The core keeps the same two slots on `BoardReading`
(`pending_damaging` / `last_damaging` / `last_damaging_move`). A window-scoped reading (the rule
`view_successor.view_context._to_dme` uses) diverged from training on the HP belief at 57 of 1,838
COMMIT decisions before this port — see §6.

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
`observe_root_with` / `parse_root_with`); the M4 ENCODER reads them (`BattleVersion::encode`,
`designs/rust_sim/encoder.md`), and search's core tree turns them on (`open_root`'s `trackers`).
The native record is opt-OUT: `TrackerState::without_record` / `BattleVersion::parse_root_unrecorded`
build none (every `Decision.window` is `None`; the trackers, the label, the reward and the row are
unchanged, since none reads it) — the chain `sim_bridge`'s core observation mode keeps, whose
consumer ships only the row.

## 3. The native record — what happened, in order, with attribution

**Status: BUILT, emitted per decision by `core_events --trackers` (`window`), read by nothing yet.
Gated by 24 constructed fixtures (`tests/window_record_test.rs`), each FAILING if its mechanic is
flattened: the six denial shapes (a faster KO, Explosion first, a Double-Edge recoil trade, a flinch,
full paralysis, a Destiny Bond trade — which is two faints and NO denial), the two gen-3 TURN CUTS
(an Explosion self-KO and a recoil self-KO, each denying the survivor), Baton Pass, Roar into
Spikes, a Spikes KO on entry and the free switch after it, Pursuit on a switch, Thief / Trick /
Knock Off, Sleep Talk, the five gen-3 callers, Rapid Spin, charge and recharge, a lost Focus Punch,
Wish / Substitute / Protect, Taunt, an Encore that overrides the target's same-turn choice, a Disable
refusal (seed-searched: gen-3 Disable is a 55 % hit), Perish Song — and the information boundary over
all of them.** `export_fixtures` (ignored) writes every fixture's inputs as JSON lines, which the M3
loss catalogue replays through the Python trackers to read what the frozen layouts keep of each.

`record::Window` = the side's decision window as an ORDERED list of `Action`s, each with its ordered
`Effect`s (`on`, `what`, `cause`, `of`). Built from the side's typed lines (the `[from]` / `[of]` /
tag truth the protocol prints) and the side's reading of the board at each line. Nothing is
flattened:

| action | carries |
|---|---|
| `Move` | user, move id, the CALLER (`called_by`: Sleep Talk / Metronome / Mirror Move / Assist / Nature Power / Snatch), the target, `pursuit_on_switch`, `locked` (Thrash / Outrage / a charged move's release), `still` |
| `Switch` | the entrant, the mon it replaced, and WHY it entered: `Lead`, `Chosen`, `Replacement { fainted }` (the FREE switch after a faint), `BatonPass { passer, boosts, volatiles }` (exactly what was passed) |
| `Drag` | the dragged-in mon, the mon it replaced, WHO forced it and with what move (Roar / Whirlwind) |
| `Cant` | a REFUSED action — the reason (`par` / `slp` / `frz` / `flinch` / `recharge` / `Focus Punch` / `Taunt` / `Disable` / `damp` / `nopp` …), the move it was prevented from (only when the PUBLIC line names it), who blocked it, the choice as this side knows it, and `then_moved` (the mon moved later this turn — Sleep Talk through sleep: the line is kept, the action was NOT denied) |
| `Denied` | a chosen action that never happened: `FaintedFirst` (the actor fainted before its turn — the cause and the action that denied it) or `TurnCut` (the gen-3 cut: the turn's first faint and its cause), placed right after the denying action (move ORDER) |
| `Refused` | the server refused a choice (`\|error\|`: trapped, disabled …) |
| `Residual` | the end-of-turn block |

Effects: damage / heal / set-HP with the attributed CAUSE (`Spikes { layers }` — the layers the
entrant met — weather, poison / toxic / burn, recoil, Leech Seed, Curse, Nightmare, `Wish { wisher }`,
an item, an ability, a move), `SubstituteHit { broke }`, `Blocked` (a move stopped by Protect /
Detect — its target lost), faints with their cause (self-KO, Destiny Bond, Perish Song, or the last
damage), status and cures, boosts, item transitions with direction (`to` on Trick / Thief / Covet),
side conditions with their layer count, crits, misses, fails, effectiveness PER HIT (a multi-hit
move's hits are separate effects), charge (`Prepare`) and recharge. A stage change is SIGNED (`Boost
{ stat, n }`, a drop negative) — the typed event's `amount` already carries the sign, so neither the
record nor the event window negates an `|-unboost|` again.

🚨 **The information boundary is a TYPE.** A viewer knows its OWN denied choice (it sent it —
`BattleVersion::note_choice`) and only THAT the opponent was denied. `record::Choice` is
`Own(Option<String>)` or `Opp` — `Opp` carries no data, so no code path can put the opponent's
chosen move into a viewer's record, and a record built from one side's stream sees nothing the
other side cannot (`present()`'s guarantee, for the record). The boundary is also checked at the
SLICE level: slice T runs `rust_core_parity_trackers.boundary_violations` on every decision's record
(`[BOUNDARY]` divergence — an opponent `Cant` / `Denied` whose choice is not `"opp"`, or one of ours
without `{"own": …}`), so COMMIT, MILESTONE and the tracker fuzz all fail on a leak.

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

## 5. The training-input semantics the trackers fold (both paths, held equal by slice T)

The M3 loss catalogue (§ below) found GIGO in three of training's own layers, and the M6 cutover
stress a fourth (the Hidden-Power belief's prior support); they are FIXED on the Python path and in
these trackers together, and slice T is 0 at COMMIT and MILESTONE after it:

| rule | what the tracker folds | Python / Rust |
|---|---|---|
| `gen3_event_window_semantics_fixes_v1` | the 22-column event window: a BOOST row's magnitude is the SIGNED stage change (W1); a FAINT no damage line took to 0 HP is `other` (W2 — Destiny Bond, Perish Song); an item line `[from]` Trick / Thief / Covet is SWAPPED on both mons (W3); a MOVE row stopped by Protect / Detect is `failed` (W4); a HAZARD row's magnitude is +1 on `sidestart`, −1 on `sideend` (W5); a bare `-damage` attaches to the open move only while its user is the CURRENT MOVER (the other side's Substitute / Belly Drum cost) | `EventWindowTracker.update` / `history::EventWindow::update`; the predicates `turn_view.is_protect_block` / `damage_is_lethal` ≡ `ev::is_protect_block` / `ev::damage_is_lethal` |
| `gen3_intent_label_semantics_fixes_v1` | the α/β label: MASKED on a DRAG (L1 / L2), on the replacement for a faint in the PREVIOUS window (L3; `Ctx::opp_active_fainted`), on an Encore override (L5); a CALLED move is labelled as its CALLER (L4) | `build_opp_intent_label` / `IntentLabel::build`, from `TurnDelta.opp_dragged` / `opp_switch_is_replacement` / `opp_called_via` / `opp_choice_overridden` ≡ `DeltaProjection` |
| `gen3_hp_prior_support_v1` | the Hidden-Power belief: a species' Smogon USAGE row is its prior, but a usage zero is not an impossibility — every one of the 16 types is a legal Hidden Power (the IVs', `sim/dex.ts` `getHiddenPower`; an unset IV is 31, `sim/pokemon.ts:387-394`, so an IV-less set is HP DARK). When a species' observations eliminate every type its row gives mass while some type explains them all, it restarts from the flat 1/16 prior and its WHOLE observation log is replayed (`prior_discarded`, in slice T's `hp`); only a log NO type explains still refuses (`all candidates eliminated`). Before: both paths refused — the cutover stress's `ladderA_3459` / `ladderB_3459` (an IV-less Lunatone's HP Dark, 2x on Gengar; Lunatone's row has no Dark). Exposure: 0 of the training pool's 1,912 HP users carry a zero-prior type, 289 of the ladder corpus's 52,007 (all HP Dark) | `HiddenPowerTracker.observe` / `HpBelief::observe` |
| `gen3_progress_clock_attribution_fix_v1` | clause (i) needs our move's OWN hits (`our_move_hit_delta`, the current mover's bare `-damage`) ≥ 3 % as well as the target's net fall (T1); a Protect block reads outcome `"fail"`, so the "our attack blocked" freeze fires (T2) | `ProgressClock._is_progress` / `clock::is_progress`; `TurnView._fold_attribution` / `turnview::fold_attribution` |

The census that proved the obs and label change touched only these fields, and the rates, is
[`../research_state/measurements/training_input_gigo_fixes_2026-09-24/`](../research_state/measurements/training_input_gigo_fixes_2026-09-24/README.md).

## 6. Findings

- **A NEUTRAL Hidden Power hit never narrows the belief** (both paths, by poke-env's rule above):
  the pending damaging move is promoted only by an effectiveness emission (`|-supereffective|` /
  `|-resisted|` / `|-immune|`, Flash Fire's `|-start|`), and a neutral hit prints none — so the
  belief sees 0.0 / 0.5 / 2.0 and never 1.0 (in `ladderA_3459`, Lunatone's three neutral hits
  before the 2x reached neither tracker). An information loss, not a divergence; not changed.

- **The search roads' Hidden-Power input is window-scoped** (`view_successor.view_context._to_dme`,
  used by the view and core roads' Python successors): a decision whose window opened at a mid-turn
  forced switch misses the opponent's promoted damaging move of that turn, which training observes.
  A depth ≥ 2 search difference (a depth-1 successor's window is the whole ply); the core's own
  trackers take poke-env's turn-spanning rule. Not fixed (search's Python successor is on the M4
  deletion path).
- **The engine does not model Metronome, Mirror Move, Assist or Nature Power** (`scan_move_probe`:
  `panic`); their record fixtures are PARSE-path (hand-written protocol, the gen-3 form
  `[from] <Name>`, `data/mods/gen3/scripts.ts:165`), not step-built battles.
- **All five gen-3 callers are READ** (`gen3_called_move_reading_v1`): poke-env raised
  `ValueError: Unhandled move message format` on Metronome, Assist and Nature Power until the fork
  and the core's `BoardReading` were fixed by class; the pin
  (`every_gen3_caller_is_recorded_as_caller_then_called_or_refused_as_poke_env_refuses`) has
  flipped — each is recorded as the caller, then the called move with `called_by`.

The loss catalogue — every case the frozen `TurnDelta`, the α/β label and the 22-column event window
flatten or lose, with rates over the MILESTONE corpus — is in
[`../research_state/measurements/rust_core_m3_2026-09-24/`](../research_state/measurements/rust_core_m3_2026-09-24/README.md).
