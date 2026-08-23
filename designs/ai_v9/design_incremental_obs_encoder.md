# Incremental Observation Encoder — census + forward design

**Status: BUILT.** Stage A shipped 2026-08-23 (`gen3_live_view_memo_v1`, `e6ec7e1`) and Stage B —
the `ObsAssembler` — shipped the same day (`gen3_obs_assembler_v1`). The document below is the
census and the design as written *before* the build; **§5.4's Amdahl arithmetic is the one part
superseded by measurement** — see the "As built" box. Everything else held, including the trap
list, which is now one named regression test per entry.

> ## As built — what changed against this document
>
> * **§5.4's headline is WRONG and the correction matters more than the win.** It sized the
>   end-to-end gain off "obs build ≈ 88% of trainer-turn CPU (encode ≈ 80%)". Measured on the
>   real path with the full protocol threaded: **obs build is 69% and `encode` alone is 38%.**
>   The encode speedup is **1.79×** (0.302 → 0.169 ms, disjoint ranges over 3 same-load pairs) and
>   the worker-CPU gain is therefore **1.19× (−16%)** — which is *exactly* what Amdahl predicts
>   from a 38% share (1/(0.62 + 0.38/1.79) = 1.20). The "~2.3–2.6× per-worker ceiling" and
>   "+40–90% rollout-side FPS" in §5.4 are RETRACTED. The lever is real and it is smaller than
>   the census's arithmetic said, because the census's denominator was stale.
> * **The invalidation surface needed a signal §2.2 does not list, and the fuzz found it.**
>   `EventKind.CURESTATUS` covers TWO protocol keywords and the second is team-wide:
>   `|-cureteam|` (Heal Bell / Aromatherapy) cures every mon on the side while naming only the
>   ACTIVE one. 11 byte mismatches in 9,272 decisions, all a stale `slp`/`brn` bit on a benched
>   opponent. A CURESTATUS event now dirties the whole side.
> * **A third dirty signal was added that §2.1 does not name**: the request, at PER-MON
>   granularity (`Gen3Battle.request_change_seq`). §2.1 correctly identifies the request as a
>   non-event mutation channel but treats it as a "recompute ≤17 dims" concern;
>   `update_from_request` in fact writes condition / item / ability / moves / stats onto our mons.
>   Because it is a pure function of its per-mon record, an *unchanged* record proves no
>   mutation — which is what lets a bench mon stay cached across a decision that carries a
>   request. A fourth (`HiddenPowerTracker.revision`) closes the one writer that is our own code
>   rather than a protocol line.
> * **§8 Q1 answered: per-mon, plus always-dirty actives.** As recommended. §8 Q3 answered the
>   other way: the encoded rows do NOT live on `EventWindowTracker` — the assembler holds them
>   and re-writes exactly `EventWindowTracker.open_records()` each decision, because every
>   in-place mutation goes through `_open_move`, which makes "a row changed after its append"
>   unrepresentable instead of merely versioned.
> * **§3's Stage B second bullet (per-mon `LivePokemon` reuse) is DEFERRED, not done.** See
>   "Deferred" at the end of this box.
> * **Gates, all green:** a new `obs_assembler_fuzz_test` (200 bridge battles / 15,607 decisions,
>   15,407 of them warm, **0 byte mismatches**, with a printed trigger census); `assembler_test`
>   (42 cases — one named regression per §2.3 trap, four of them scripted because the random
>   corpus reports them NOT SEEN); `gen3_data_obs_parity` with **no fixture regen**;
>   `obs_roundtrip_fuzz` 985 decisions bit-identical; `redecide_rollback_fuzz` 168 re-decides /
>   0 phantom; `obs_materializer_branch_integration`; `search_clone_parity_fuzz`; the whole `sim`
>   tier; mypy / ruff / file-size.
> * **Deferred, and named so it is not mistaken for done:** per-mon `LivePokemon` reuse (§3
>   Stage B). `live_view()` is now the largest single item the encode still pays cold, but making
>   it partial needs per-mon dirt *inside* `Gen3Battle`, and `parse_request` writes to all six of
>   our mons on a channel whose granularity lives in poke-env — so the honest version of that
>   change is a bigger, wider-blast-radius piece of work than this one. `obs: legal + mask`
>   (0.145 ms, 22% of worker CPU and untouched here) is now the largest un-attacked obs stage.

**The problem being answered.** `trainer_turn_benchmark` baseline: the obs build is ~88% of
per-decision rollout-worker CPU (`state_encoder.encode` ≈ 80%). Every decision rebuilds all
2,501 dims from scratch — `live_view()` reconstructs 12 `LivePokemon` from poke-env objects,
`pokemon_encoder` re-walks all 12 slots, the H-B loop rewrites ≤32×22 cells — regardless of
what changed since the last decision. The question this document answers with a census rather
than a hunch: *how much of that walk is recomputing values that did not change?*

**Headline: ~95.0% of the 2,501 dims are static-per-episode, reveal-monotone, or
event-sparse** (prior was ">85%"; measured by classification below: 2,376 / 2,501). Only 125
dims (5.0%) change per-turn/per-decision by nature, and **119 of those 125 are deterministic
functions of the turn number** (recency scalars, `TURNS_AGO`, pair-recency, the clock) that
are O(1) lookups from tracker state. The genuinely request-dependent residue is **6 dims**
(`legal_now` ×4 + `trapped`/`maybe_trapped`) — 0.24% of the vector.

---

## 1. The census — per-block mutability classification

### 1.1 Method

Classes (per the task definition):

| Class | Meaning |
|---|---|
| **STATIC** | fixed for the whole episode once the slot exists (incl. structurally-zero blocks) |
| **REVEAL** | changes only on a reveal event, monotone (opp species/moves/item/ability/HP-type) |
| **SPARSE** | event-driven; a handful of deltas per turn (HP, status, boosts, volatiles, PP…) |
| **DENSE** | genuinely recomputed each decision/turn (clock, recency ticks, request legality) |

Classification is per-offset, from `constants.py` + the per-block encoders (`pokemon.py`,
`moves.py`, `reactive.py`, `global_env.py`, `active_context.py`, `state_encoder.py`).
Caveats (mid-battle mutators of "static" fields) are in §2's trap list — they are handled by
invalidation, not by reclassifying the dims.

### 1.2 Per-mon slot, OUR side (122 dims × 6 = 732)

| Field | Dims | Class | Notes |
|---|---|---|---|
| species id + base stats | 7 | STATIC | trap: FORMECHANGE (Castform) / TRANSFORM |
| item `[id, known, consumed]` | 3 | SPARSE | ENDITEM / Trick / Knock Off events |
| type ids ×2 | 2 | STATIC | trap: Conversion/Conversion 2, forme change |
| ability ×4 | 4 | STATIC | ours is known from turn 0 |
| status one-hot | 7 | SPARSE | STATUS / CURESTATUS events |
| 4 × move slot: 10 of 11 dims | 40 | STATIC | id/power/flags/type/category/known/max_pp/acc/never_miss |
| 4 × move slot: `current_pp` | 4 | SPARSE | our MOVE events (~1 slot/turn); poke-env asserts request-pp == tracked-pp, so MOVE is the sufficient signal |
| HP fraction | 1 | SPARSE | DAMAGE/HEAL/SETHP/FAINT |
| species_known | 1 | STATIC | 1.0 for every populated own slot |
| sleep/toxic counters | 2 | SPARSE | only while statused |
| spread (IV/EV/known/nature) | 18 | STATIC | teambuilder-backfilled once |
| Hidden-Power block | 17 | STATIC | own side: `hp_revealed=1`, probs zero |
| sleep-wake belief | 3 | SPARSE | zeros unless asleep |
| recency ×3 | 3 | DENSE | turn-anchored `cur_turn − event_turn`; ticks every turn |
| protect odds | 1 | SPARSE | protect usage / switch reset; usually 1.0 |
| last-action ×6 | 6 | SPARSE | active slot only, per-turn |
| trapped / maybe_trapped | 2 | DENSE | request-sourced, our active only |
| active flag | 1 | SPARSE | switch events |
| **Subtotal per slot** | **122** | | STATIC 89 · SPARSE 28 · DENSE 5 |

### 1.3 Per-mon slot, OPP side (122 × 6 = 732)

| Field | Dims | Class | Notes |
|---|---|---|---|
| species + stats, item, types, ability, species_known | 17 | REVEAL | zero until revealed; then fixed (item: + consume events, see §2) |
| 4 × move slot (all 11) | 44 | REVEAL | gen3 Showdown does not track opp PP → whole slot is reveal-monotone |
| HP-candidate block | 17 | REVEAL | HiddenPowerTracker narrows monotonically |
| spread | 18 | STATIC | structurally zero for opp |
| trapped bits | 2 | STATIC | structurally zero for opp |
| status, HP, counters, sleep belief, protect, last-action, active | 21 | SPARSE | same event families as our side |
| recency ×3 | 3 | DENSE | |
| **Subtotal per slot** | **122** | | REVEAL 78 · STATIC 20 · SPARSE 21 · DENSE 3 |

### 1.4 The non-team blocks

| Block | Dims | STATIC | REVEAL | SPARSE | DENSE | Dense members |
|---|---|---|---|---|---|---|
| Active context ×2 (boosts 14 + volatiles 44 per side) | 116 | — | — | 116 | — | (whole 58 swaps on switch — still event-driven) |
| Global env | 20 | — | — | 16 | 4 | clock ×3 + weather `turns_remaining` tick |
| Board (reactive) | 17 | — | — | 12 | 5 | `turns_since_progress` + `legal_now` ×4 (req ids/types ×8 change only on switch → SPARSE) |
| Pair history 6×6×5 | 180 | — | — | 144 | 36 | `recency_of_last_pairing` per cell |
| Event window 32×22 | 704 | — | — | 672 | 32 | `TURNS_AGO` column (moves only on turn increment) |

The event window is SPARSE in the ring-buffer sense: rows are append-only (a decision appends
the ~0–6 events of its window); in the *flat* layout an append shifts every row, but that is a
representation artifact §5 removes, not churn in the underlying values. 21 of 22 columns of a
row never change after the append.

### 1.5 Totals — the headline table

| Class | Dims | Fraction |
|---|---|---|
| STATIC-per-episode | 654 | 26.1% |
| REVEAL-monotone | 468 | 18.7% |
| PER-TURN-sparse (event-driven) | 1,254 | 50.1% |
| PER-DECISION/turn dense | 125 | 5.0% |
| **Total** | **2,501** | |

- **Static-or-sparse = 2,376 / 2,501 = 95.0%** (the prior ">85%" holds with margin).
- Of the 125 DENSE dims, **119 are deterministic per-turn ticks** computable from tracker
  state with no battle walk at all — and every one of them is a log-saturation with an
  **11-value codomain** (`log1p(min(n,10))/log(11)`), i.e. a lookup table: recency 36,
  pair-recency 36, `TURNS_AGO` 32, clock 3+1, progress clock 1, plus 12 trapped-bit offsets
  that are structurally zero on 5 of 6 slots.
- The only dims that *must* consult the fresh request each decision: `legal_now` ×4,
  `trapped`/`maybe_trapped` on our active slot, and (on switch) the req ids/types — **≤17
  dims, all already sourced from the `LegalActions` snapshot the mask is built from** (no
  extra work: the env threads `legal` into `encode` today).

**The cost interpretation.** Cost is per-*call walking*, not per-dim change: a typical
mid-game decision touches ~2 mons (the actives), ~1–6 events, and the dense ticks — yet the
current encode pays 12 `LivePokemon.from_pokemon` + 12 full slot encodes + 36 pair cells +
≤32 event rows + every sub-encoder's small-array allocations, ~3.46k Python function calls
(the leaf's primary regression metric, post-`gen3_entity_rehome_v1` baseline). The census
says ~95% of that walk re-derives unchanged bytes.

---

## 2. The invalidation surface

### 2.1 The dirty-signal stream already exists — and it is provably complete

The event-sourced battle layer is the invalidation stream. Three facts make it unusually
strong for this purpose:

1. **`MESSAGE_POLICY`'s `STATE_ONLY` bucket is EMPTY in gen3ou** (`battle_event.py`: "every
   state-mutating line we see is also battle content and is recorded as an EVENT"). So there
   is no protocol line that mutates poke-env state without emitting a `BattleEvent` — the
   class of bug where a cache misses an invisible mutation is structurally excluded for
   protocol-driven state.
2. **An unclassified keyword RAISES** (the completeness tripwire) and
   `assert_conservation()` proves no line is dropped — a future new keyword cannot silently
   bypass the cache.
3. **The per-decision window is already sliced and already consumed**:
   `EpisodeTracker.update_progress_clock` feeds the *same* `events_since(cursors[-1])` slice
   to the ProgressClock, RecencyTracker, PairHistoryTracker and EventWindowTracker. The obs
   cache is simply the fifth consumer of that window — no new plumbing, and the out-of-band
   `CHOICE_REJECTED` append is covered by the same cursor mechanics (pinned by
   `event_window_test`).

Two dirty signals are NOT in the event stream and must be modeled separately:

- **The turn boundary** (`|turn|` is CONTROL, not an event) → drives the 119 deterministic
  tick dims. Signal: `live.turn`/`battle.turn` changed since last encode.
- **The request** (never an event) → drives `legal_now`/`trapped`/`maybe_trapped`/req-block.
  Signal: the `legal` snapshot threaded into `encode` — recompute unconditionally, it is ≤17
  dims off already-parsed fields.

### 2.2 Event → dirty-set map

| Event kind(s) | Dirties |
|---|---|
| `SWITCH` / `DRAG` | both slots' `active` flag; the WHOLE 58-dim active-context of that side; protect odds (reset); toxic counter (reset); req ids/types block (our side); last-action placement; recency `seen` reset; pair-history current pairing |
| `DAMAGE` / `HEAL` / `SETHP` | target slot HP fraction |
| `FAINT` | HP→0, active flag, reactive `fainted` counts, forced-switch phase |
| `STATUS` / `CURESTATUS` | condition one-hot, counters, sleep-wake belief (+ the sleep-sources fold, §2.3 trap 5) |
| `BOOST`/`UNBOOST`/`SETBOOST`/`CLEARBOOST` | active-context boosts (14) of the target side |
| `VOLATILE_START`/`VOLATILE_END`/`ACTIVATE`/`PREPARE`/`MUSTRECHARGE` | active-context volatiles (44) |
| `MOVE` | our PP (the used slot); opp move REVEAL (whole opp move block + HP-candidate block via HiddenPowerTracker); last-action; pair-history cells |
| `ITEM` / `ENDITEM` | item `[id, known, consumed]` of that slot |
| `ABILITY` | ability block of that slot |
| `WEATHER` | global weather group (7) |
| `SIDE` | spikes ×2, screens ×8 |
| `TRANSFORM` / `FORMECHANGE` (incl. `detailschange`/`replace`) | **nuke the whole mon slot** — species/stats/types/moves all move at once; per-field surgery here is where correctness dies, so don't attempt it |
| `CHOICE_REJECTED` | event-window row only |
| any event at all | one event-window ring append (rows are cheap: 22 floats) |

### 2.3 The traps (each is a named test case in §4)

1. **Switch resets more than the switch columns.** poke-env clears boosts/volatiles and
   resets protect/toxic counters internally on switch; "no BOOST event ⇒ boosts unchanged"
   is FALSE across a switch. The SWITCH dirty-set above must carry the resets explicitly.
2. **Forme change / Transform** rewrite "static" fields (species num, base stats, types,
   and for Transform the whole moveset). Whole-slot invalidation, never per-field.
   Castform + Deoxys formes are in the dex (`gen3_species_formes_v1`); the fuzz gate must
   drive them (the rust-port randbats corpus already exercises Forecast/Transform).
3. **The request-order block is per-decision by nature** — never cache `legal_now` or the
   trapping bits. `legal.move_slots` is the same snapshot the mask reads; a cached copy that
   survives one request too long is exactly the misalignment class
   `gen3_op_move_align_v1`/`gen3_locked_choice_never_rejected_v1` exist to prevent.
4. **Cross-mon / cross-block facts:** `fainted` counts (reactive) aggregate over 12 slots; a
   FAINT must dirty both its slot and the reactive scalars. `species_known` + team-list
   ORDER: slots are positional over `list(battle.team.values())` — dict insertion order is
   append-only for the opp (a new reveal appends a slot; existing indices never move), but
   `get_team_list`'s "active opp not in team → append" fallback could give a mon a
   *temporary* index; key the per-mon cache by SPECIES (the `SlotRegistry` convention), not
   by list position, and rebuild the position→species join when membership changes.
5. **Two per-encode log folds hide in the current encoder** and must become event-driven
   under the incremental design or they cap the win:
   `build_sleep_sources(battle)` (whole-log fold, gated on "anyone asleep") and
   `build_wish_pending(battle)` (per-encode fold in `reactive.encode`). Both are functions
   of small event families (STATUS-slp `[from]` clauses; Wish MOVE events) — fold them
   incrementally off the same decision window, exactly as `Gen3Battle` already folds weather
   incrementally (`live_weather()`, the documented O(turns²)→O(1) precedent).
6. **The flat event-window layout shifts on every append.** Keep the ring of encoded rows in
   the tracker (it already keeps the dict rows) and materialize the 704-dim block with one
   vectorized copy + a `TURNS_AGO` column rewrite (32 LUT writes, only when the turn
   changed). Never "patch the flat block in place" — front-padding means every append moves
   every row.
7. **Sub-turn decisions:** a forced-switch decision lands mid-turn (`turn` unchanged) — the
   dense tick patch must key on `(turn, forced_window)` semantics already handled by the
   trackers, and the FAINT/SWITCH events of the window still dirty their slots. No special
   case needed beyond "apply the window's events, then patch ticks if turn changed".
8. **`SWITCH` of a Baton Pass** keeps volatiles/boosts (gen3 Baton Pass passes them):
   poke-env models this — which is exactly why the dirty-set for SWITCH must be "recompute
   the active-context from live state", never "write zeros".

---

## 3. The view-construction cost question

**What one `live_view()` builds:** 12 × `LivePokemon.from_pokemon`, each reading ~25 poke-env
properties and allocating a moves-tuple, a boosts dict-comprehension, a volatiles
dict-comprehension, and *two full dict copies* (`base_stats`, `stats`) — plus per-side
side-condition dicts and the (already-incremental) weather fold. All of it frozen-dataclass
construction over primitives; none of it cached.

**How often it runs per production decision — more than the benchmark shows.** The obs
benchmark counts 12 `from_pokemon`/encode (one `live_view` inside `encode`, ~15–20% of the
build). But the production decision path builds a *fresh* LiveView at least four times:
`tracker.record()` (HP-candidate observe), `update_progress_clock()` (clock + the three
tracker updates), `encode()` (via `strict_view()`), and `reward_manager.calc_reward` (twice
in some paths — `reward_manager.py:1695,1871`). Each is a full 12-mon rebuild of an object
that is *identical within a decision* (no protocol arrives between them).

**The offline path says the same thing louder:** the ledger's materializer profile has
`LiveView.from_pokemon` at **1,084 calls/arm = 50% of cumulative** — that is ~90 replayed
decisions × 12 mons; the per-decision view cost times the full prefix replay.

**Two-stage answer, both preserving the strict boundary:**

- **Stage A (independent, ships first): memoize `live_view()` per decision.**
  `Gen3Battle` owns a `(len(_events), turn, request-identity) → LiveView` one-slot memo;
  `live_view()` returns the cached snapshot when nothing arrived since it was built. LiveView
  is immutable, so sharing it is semantically free. This collapses ~4 builds/decision to 1
  *without touching the observation package at all* (no obs-benchmark gate; it is a
  `battle/` change) and cuts the offline materializer's 50% family roughly 4×. The
  incremental weather fold in `Gen3Battle` is the in-tree precedent for exactly this shape.
- **Stage B (part of the incremental encoder): per-mon `LivePokemon` reuse.** Under the same
  per-slot dirty flags, rebuild only the `LivePokemon` of dirtied mons and reuse the frozen
  instances for the rest; `LiveView.from_battle` becomes "new side tuples over mostly-old
  elements". The surface (types, accessors, immutability) is unchanged, so
  `strict_api_lock_test` and every consumer are untouched — the lock constrains *access*,
  not construction.

---

## 4. The gate story

Three lines, then the detail:

1. **Byte-identity is the contract:** a new bridge-backed fuzz runs real battles and, at
   EVERY decision, encodes both ways — incremental (the live warm path) and a fresh
   full-rebuild encoder on the same battle — asserting bit-for-bit equality, with
   constructed forme/Transform/wrap/Baton-Pass/forced-switch scenarios in the corpus.
2. **The existing linchpins run unchanged and un-regenerated:** `gen3_data_obs_parity` (the
   991-decision golden fixture, 4.2 s, in the routine gate) must pass with NO fixture regen
   — value-neutral means no `ARCH_SIGNATURE` bump — and `obs_roundtrip_fuzz` (live ==
   offline, bit-for-bit, sequential + concurrent phases) gates the materializer inheritance
   including the per-arm restore.
3. **The mandatory obs benchmark runs before/after per the observation leaf — but its reps
   loop must not be the verdict** (see the warm-loop caveat below); `trainer_turn_benchmark`
   carries the end-to-end number.

Detail:

- **The incremental≡full fuzz** follows the house fuzz pattern (real battles via the bridge,
  validate in `choose_move`). It is the *pre-enable* gate, like the pair-history fuzz was.
  Corpus must include: Castform/Forecast teams, Ditto/Transform, the wrap family
  (partial-trap volatiles + trapped bits), Baton Pass chains, Explosion double-KO
  forced-switch windows (the alive-filter resync class), Pain Split (`SETHP`), Knock
  Off/Trick (all three `ITEM_TR_*` routes), and a 250-turn stall (deque caps + clock
  saturation). Every §2.3 trap gets a NAMED deterministic regression test that fails if its
  invalidation edge is removed (the edge-case-regression rule).
- **The oracle must be the full-rebuild encoder run fresh** — not a second read of the
  incremental state. The two paths should share the per-block *writer* functions (§5) so
  the comparison is scheduler-vs-scheduler, not a fork that can drift; the oracle-by-name
  lesson (the leaf's positional-binding section) applies: the fuzz reads addresses from
  `get_layout()`, never literals.
- **⚠️ The obs benchmark's `--reps 400` loop re-encodes the SAME decision.** Under caching,
  rep 2..400 are 100% warm — the benchmark would report a fantasy speedup, and its
  calls/encode primary metric becomes bimodal (cold ≈ today's 3.46k, warm ≈ a few hundred).
  The gate must therefore (a) report cold and warm separately (invalidate-all between reps
  for the cold series), and (b) lean on `trainer_turn_benchmark`, which walks real
  consecutive decisions and measures the honest mix. This is a benchmark *extension* to land
  with the feature, not a reason to skip the gate.
- **`GEN3AI_OBS_VERIFY=1`** (see §5 migration): a shadow mode that runs both paths and
  asserts equality per decision, cheap enough to leave on in the smoke test and the fuzz
  tier permanently.

---

## 5. The design sketch

### 5.1 Shape

An `ObsAssembler` owned by the **EpisodeTracker** (it already owns every incremental
consumer of the decision window and already participates in snapshot/restore):

- **State:** one persistent `float32[2501]` buffer; per-mon dirty bits (12), keyed by
  species via the `SlotRegistry` convention; block dirty bits (active-ctx ×2, global,
  reactive-sparse, pair-cells touched); the event-window ring of pre-encoded 22-float rows;
  the incremental sleep-source and wish folds (§2.3 trap 5); `last_turn` for the tick patch.
- **Feed:** a fourth call inside `update_progress_clock` (the existing three-step protocol
  `record → update_progress_clock → encode` is untouched — the leaf already documents that a
  harness skipping it gets a structurally-zero block, and the same contract covers this).
  The assembler folds `events_since(cursors[-1])` into dirty bits + ring appends.
- **Encode:** `state_encoder.encode` becomes the *scheduler*: (1) re-encode dirty mons via
  the unchanged per-block writers (`pokemon_encoder.encode` et al.) into their slot slices;
  (2) if turn changed, patch the 119 tick dims (vectorized; all log-saturations are 11-entry
  LUTs); (3) write the ≤17 request dims from `legal` unconditionally; (4) materialize the
  event window (ring → one copy + `TURNS_AGO` column); (5) return `buffer.copy()` (SB3
  stores obs into its rollout buffer; ~2,501 floats ≈ single-digit µs).
- **Cold path = the current code.** Reset, resync, or any doubt ⇒ full rebuild through the
  same writers. One writer per block, two schedulers (full/incremental) — the two paths
  cannot drift in *content*, only in *when* they run, which is exactly what the fuzz gate
  checks.

### 5.2 Migration: internal swap gated on byte-identity — NOT a flag

Argued, per the task: a launch flag would fork the obs path into two long-lived variants,
and this tree has measured what happens to the branch nothing runs (the seedless-seed
lesson: "a default branch nothing tests is untested no matter how green the suite looks";
the three-times-red golden fixture). A perf-only, byte-identical change is by house
convention a value-neutral refactor — no `ARCH_SIGNATURE` bump, no flag, the same category
as the `_category_val` memoization and the hoisted `_defender_terms`. The escape hatch is
**`GEN3AI_OBS_VERIFY=1`** (shadow-encode both paths, assert, crash loud on divergence) plus
the cold-path rebuild being one call away — an env-var kill switch
(`GEN3AI_OBS_INCREMENTAL=0`) is acceptable as a *diagnostic* opt-out precisely because it
selects between two schedulers over one writer set, not two implementations.

### 5.3 Both consumers inherit the win — and the restore hazard, by name

The encoder is shared by TRAINING and the offline **materializer**
(`obs_materializer.py` replays through the real obs pipeline; `obs_roundtrip_fuzz` pins the
equivalence). Two rollback mechanisms touch assembler state:

- **Self-play re-decide rollback** (`EpisodeTracker.snapshot()/restore()`): today it
  snapshots only the history deques and deliberately skips the HP tracker/registries as
  idempotent. The assembler's dirty-bits + ring are NOT idempotent under rollback (a rolled
  back `record` must not leave its window's ring appends behind). The assembler state must
  join the snapshot tuple — it is small (bits + ≤32 rows + a 2501 buffer; the buffer can be
  re-derived by marking all-dirty on restore instead of copying, which is the simpler
  correct choice).
- **The materializer's per-arm restore** (`_PlayerSnapshot`): deep-copies the player's whole
  battle/tracker graph and restores fresh deep copies per arm — *this week's clone-aliasing
  ground*. Two rules make the assembler safe by construction: (a) the assembler lives INSIDE
  the deep-copied tracker, so a restored arm carries a cache consistent with its own rolled
  back state (the cache rides the object it describes — never a module-level or
  encoder-instance cache keyed by battle identity); (b) any memo keyed on a battle
  (Stage A's `live_view` memo) keys on *content position* (`len(_events)`, turn, request
  identity) held ON the battle object itself, so a deep copy carries a self-consistent memo
  and a divergent replay naturally misses. A cross-object cache keyed by `battle_tag` would
  serve forward-state bytes to a rewound arm — that is the named silent-wrongness case, and
  the branch integration test (`obs_materializer_branch_integration_test`) plus the
  roundtrip fuzz's concurrent phase are the gates that would catch it.

### 5.4 Honest Amdahl (static estimate — must be measured)

Baselines (leaf, 2026-08-16 re-baseline, idle box): full-protocol encode **0.363
ms/decision** ≈ naked encode 0.246 + recency/clock/H-A ≈ 0.077 + H-B write ≈ 0.040; obs
build ≈ 88% of trainer-turn CPU (encode ≈ 80%).

Warm-decision estimate from the census (typical mid-game turn: 2 dirty mons, ~4 events,
turn tick):

| Component | Today | Warm incremental (est.) |
|---|---|---|
| live_view construction | ~0.05 ms (×~4/decision in production, §3) | ~2 mons rebuilt, 1 build/decision ≈ 0.01 ms |
| 12 per-mon slot encodes | ~0.20 ms | 2 dirty ≈ 0.035 ms |
| H-A pair loop + recency + clock | ~0.077 ms | vectorized tick patch ≈ 0.01–0.02 ms |
| H-B 32-row write loop | ~0.040 ms | ring append + column patch ≈ 0.005–0.01 ms |
| request block + copy-out | ~0.005 ms | ~0.01 ms |
| **encode total** | **~0.363 ms** | **~0.07–0.09 ms (≈ 4–5×)** |

Trainer-turn CPU: 100 → 100 − 80 + 80/4.5 ≈ **38–43 ⇒ ~2.3–2.6× per-worker throughput
ceiling**. End-to-end training FPS will land well under that ceiling: the compile-opponents
precedent is the calibration (a 6.5× per-forward win bought +33% FPS at `--n-envs 48`
because the rollout bottleneck moved), and obs-build is a bigger share of worker CPU than
the opponent forward was — an honest expectation is **+40–90% rollout-side FPS**, with the
bottleneck moving to protocol parse (+7%) and the bridge transport. The offline
materializer's per-arm cost (arm_ms = 4.78 + 0.853·turn, obs ≈ 1.8 ms of it, view family
50% cum) drops on both terms — Stage A alone is worth ~25% of the arm's cumulative profile.
Every number in this subsection is a static estimate; the §4 benchmark pair is the verdict.

### 5.5 Explicit NON-goals (v1)

- **No Rust in v1** (§6 makes the arithmetic case).
- **The one-sided wall untouched:** nothing here changes what is observable; reveal-gating
  and leak-safety are inherited unchanged from the existing writers.
- **No layout change:** byte-identity IS the contract; offsets, ordering, saturation
  curves, the flat 2501 vector all stay. (A layout that made the event window
  ring-native would be a separate, retrain-class proposal.)
- **No new obs semantics:** dims that are per-decision by nature (request legality) stay
  per-decision; nothing gets "frozen" for speed.
- **poke-env internals untouched:** dirty signals come from OUR event log, not from
  instrumenting poke-env setters.

---

## 6. The Rust question, answered from the census

**Verdict: NO — after the incremental encoder there is not enough left for a PyO3 assembly
kernel to win, and the census shows why by arithmetic.**

1. **What an FFI kernel could ever accelerate is the walking + array-writing, and the
   incremental encoder deletes ~95% of it.** Post-incremental, the warm path is ~0.07–0.09
   ms/decision. Suppose Rust made the *assembly* (tick patch, ring materialize, copies) free:
   those are the already-vectorized ~0.02–0.03 ms — the theoretical FFI win is **≤ ~0.03
   ms/decision ≈ 8% of a trainer turn**, before paying the boundary cost.
2. **The residual cost is Python-object *reads*, which FFI cannot skip.** The 2-dirty-mon
   re-encode is dominated by reading poke-env properties (`mon.moves`, `mon.boosts`,
   `mon.effects`, …) and building `LivePokemon` — a PyO3 kernel would do the same reads
   through the CPython API at comparable cost, because the source of truth is a Python
   object graph, not a flat buffer. The only way Rust wins big is if the *state* is already
   Rust-side — i.e. an obs encoder inside the rust sim reading its own typed battle state.
   That is a different project with a different risk surface (a second full obs
   implementation to hold byte-identical — the exact two-renderers cost the prober TUI
   retirement was about), and it is the one-sided-wall-adjacent non-goal, not v1.
3. **The parity surface an FFI kernel would need is the one we just built for free.** The
   incremental≡full fuzz gate (§4) comes with the incremental encoder anyway; a Rust kernel
   would need the same gate PLUS an ABI/versioning story PLUS the materializer restore
   semantics across the boundary.
4. **The decision procedure, so this stays arithmetic:** after the incremental encoder
   lands, re-run `trainer_turn_benchmark` + the extended obs benchmark's warm series. If a
   single pure-array hotspot >20% of the warm encode survives that numpy cannot express
   (none is visible from here — the candidates, tick patch and ring copy, are numpy-native),
   the FFI question may be reopened *for that hotspot only*. Anything else is enthusiasm.

---

## 7. Side-findings from the census (drive-by, no edits made — read-only pass)

- `constants.py`'s `EVENT_WINDOW_DIM` trailing comment reads `# 608`; the live value is
  32 × 22 = **704** (the comment predates the +3 columns). The file's own header forbids
  evaluated numbers in comments — this one should be deleted, not corrected.
- `src/agents/observation/CLAUDE.md`'s headline says "**2437**-dim"; the live
  `Gen3ObservationEncoder.dimension` is **2501** (the `gen3_event_semantics_v1` +2 columns
  = +64 dims). The root `CLAUDE.md` and `ARCHITECTURE.md` already say 2501.
- `reactive.py`'s class docstring still describes the pre-`gen3_entity_rehome_v1` block
  (matchup matrices, 414 dims); `encode()`'s comments and `get_layout()` are current.
- `pokemon.py`'s class docstring says "96 dims … POKEMON_FULL_DIM = 98"; live values are
  119/122.
- The obs benchmark harness (per the leaf's 2026-08-16 measurement-honesty note) now
  threads the full protocol — the incremental design's warm/cold split (§4) should be added
  to the same harness rather than a new one.

## 8. Open questions for the implementation pass

1. Dirty granularity: start per-mon (12 bits) or per-sub-block? Census says per-mon is
   enough — a dirty mon costs ~18 µs to fully re-encode, and sub-block bookkeeping is where
   invalidation bugs live. Start coarse; the always-dirty-actives rule (re-encode both
   actives every decision unconditionally) buys large robustness for ~36 µs and shrinks the
   event→dirty map to the bench-affecting families (reveal/consume/status/faint/forme).
2. Does Stage A (the `live_view` memo) ship first as its own change? It is independent,
   `battle/`-scoped, gated by existing tests + the reward/obs goldens, and worth ~25% of the
   offline arm profile on its own. Recommended: yes, separately.
3. Where the H-B ring's encoded rows live: extend `EventWindowTracker` to store the
   22-float row alongside its dict row (one producer), vs a parallel ring in the assembler
   (two producers, drift risk). The one-producer answer follows the EventCol lesson.
4. Whether `BattleContext.from_battle` (in `record()`) can share the Stage-A memoized view —
   it is inside the same decision, so yes by construction, but verify no path records
   between protocol chunks.
