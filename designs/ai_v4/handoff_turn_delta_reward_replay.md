# Handoff: Event-Sourced Consumer Migration — Steps 4–6 (TurnDelta · Reward · Replay)

**Audience:** a fresh session executing the remaining work of
`design_event_sourced_battle.md`. Steps 1–3 are **shipped to main**; this doc carries
everything needed to do Steps 4 (TurnDelta fold), 5 (reward), 6 (replay) cold.

**Why this doc exists:** the prior session's Read tool began returning corrupted content
(garbled `battle_context.py`, fabricated duplicate code blocks) — too risky to rewrite a
core training file blind. Start fresh with clean tooling.

---

## 0. What is complete (main + current branch)

**Step 4 (TurnDelta fold) is COMPLETE on the current branch (not yet shipped to main).**
Key changes: `turn_view.py` → `faint_details()` + `FAINT_CAUSE_VOCAB`; `battle_context.py` → `TurnDelta.build_from_events()` + faint-cause/attempted-move fields; `episode_tracker.py` → cursor tracking; `turn_delta_encoder.py` → gen3_effects cant (crash-don't-drop) + 127-dim layout + embedded-ID manifest; `features_extractor.py` → manifest-driven `embed_delta_slot` (no hardcoded positions); `gen3_effects.py` → +doomdesire/futuresight volatiles (smoke tripwire caught the gap); `ARCH_SIGNATURE = gen3_turn_delta_v2`; obs dim 2823 → 2997.

Post-review simplifications (per design discussion): dropped `faint_count` from the obs (redundant with faint flags + cause popcount; kept on the dataclass for reward); dropped `attempted_switch_to` (a pressed switch always executes, so it always equalled `switch_to` — only attempted *move* carries signal). The faint multi-hot is per-side (the common multi-KO is the 2-sided Explosion case = 1 bit each; same-side double needs the rare Pursuit/Future-Sight-into-hazard window).

Gates green: 1106 unit + 20 integration tests, smoke PASSED (round-trip + eval, 0 crash-don't-drop trips), event-log fuzz PASSED (40 battles, full-pool random teams + per-decision volatile/cant validation).

## What was already shipped (main)

The event-sourced battle layer + the full live-state observation are **on main**:

| Commit | What |
|---|---|
| `3e4de10` | Event-sourced battle layer: `Gen3Battle`, `BattleEvent` log, `MESSAGE_POLICY` conservation, `TurnView` |
| `c7ecc8d` | `LiveView` read-model + `Gen3Battle` training injection + `event_cursor`/`events_since` |
| `786127a` | **Full live-state observation** (`gen3_live_state_v1`, obs 2734→2823): full volatiles + event-sourced weather + Safeguard/Mist |

**Already in place that Steps 4–6 build on:**
- `src/agents/battle/gen3_battle.py` — `Gen3Battle(Battle)`. Has `events`, `events_for_turn(turn)`, **`event_cursor`** (seq of next event), **`events_since(cursor)`** (the per-decision window slice — THIS is what TurnDelta must fold), `live_view()`, `assert_conservation()`.
- `src/agents/battle/turn_view.py` — `TurnView.for_turn(battle, turn)` / `.from_events(events, our_side)`. Per-side `SideTurn`: `moved`, `move_id`, `called_via`, `switched`/`drag`/`switched_to`, `fainted`, `cant_reason`/`cant_move`, `crit`/`missed`/`failed`, `outcome`, `effectiveness`, `target_species`, `damaging_move` (`DamagingMove` dataclass), `boosts`, `status_applied`/`status_cured`, `item_lost`/`item_gained`. Turn-level: `move_order`, `we_moved_first`, `both_attacked`, `someone_fainted`, `faints()` (species list), `damage_on(species, side=…)`, `weather_set()`, `hazards_set()`, `events_of(kind)`.
- `src/agents/battle/live_view.py` — `LiveView`/`LiveSide`/`LivePokemon`/`LiveWeather`.
- `src/agents/observation/gen3_effects.py` — **source-derived allowlists, crash-don't-drop**: `encode_volatiles` (VOLATILE_DIM=41), **`encode_cant_reason`/`normalize_cant_reason` (CANT_DIM=12)** — the cant one-hot is built but **NOT yet wired into TurnDelta**; Step 4 wires it in. `describe_*` helpers.
- Players/env inject `battle_class=Gen3Battle` (so `.events`/`live_view()` exist in training/eval/replay): `Gen3Player` default, `Gen3Env` sets it on both `_EnvPlayer` agents post-`super().__init__()`.

**Verification spine (already green):** `src/agents/battle/event_log_fuzz_e2e_test.py` —
real gen3ou battles, both players `Gen3Battle`, re-derives each turn from raw protocol and
asserts the event log matches; conservation + crash-don't-drop + coverage. Run it after
every Step-4 change.

---

## 1. The four asks (user, verbatim intent)

1. **Multi-KO + cause.** Represent that MULTIPLE mons fainted in one window and **why**:
   KO'd by an attack, spike-sacked (hazard), killed by sandstorm/hail (weather), by
   poison/burn (status), recoil, self-KO (Explosion), Leech Seed, etc. The current
   `we_fainted`/`opp_fainted` are single bools — insufficient.
2. **Phase: start-of-attack vs extended part of the turn.** The delta must say whether
   this window is the start of a turn (move-selection) or a continuation (forced-switch
   replacement after a faint). Already exists as `phase_is_forced_switch`; keep + clarify.
3. **Attempted action.** Record **what we tried to pick** (the pressed action → move id /
   switch species), preserved even when it never fired (flinch / frozen / cant /
   KO-before-acting). Distinct from the outcome.
4. **Missing → 0.** If data isn't available, encode zeros — never stale or guessed.

**Decisive design fact:** #1 (cause) is **only available from the event log** — snapshot
diffing sees "HP hit 0" but not *why*. So Step 4 MUST fold TurnDelta from the event log
(`TurnView` over the `events_since` window), which is exactly design §6. This retires the
diff heuristics (`_ko_before_acting`, phaze recovery via `opp_all_last_move_ids`,
`_align_effectiveness`).

---

## 2. Step 4 — TurnDelta fold (retrain-class)

### 2a. The window (critical nuance)
A "TurnDelta" is the window **since the agent was last asked to act** — NOT a protocol
`|turn|N`. A forced switch after a faint splits one game turn into two decision windows;
a faint window can span a `|turn|` boundary. Use `Gen3Battle.event_cursor` /
`events_since(cursor)` — already built for exactly this. **Do NOT use `events_for_turn`**
for the delta (it slices by protocol turn). Decision: **preserve "since last input"
semantics** (clarity: 1:1 with action + prev-mask; correctness: forced-switch carries a
distinct signal that protocol-turn would smear).

### 2b. Wiring (`src/agents/training/episode_tracker.py`, 203 lines)
Currently `EpisodeTracker` holds `_history: list[BattleContext]` + `_actions: list[int]`
and builds deltas by diffing consecutive contexts (`build_delta` L175, `prev_N_delta_vecs`
L181, `record` L111, `reset` L197). The mons-fold needs the event window per decision:
- In `record(battle, mask)`: capture `battle.event_cursor` into a parallel list
  `_cursors` (alongside `_history`). The window for delta *i* is
  `events_since(cursors[i])` up to `cursors[i+1]` (slice the battle's `events`).
- `build_delta()` / `prev_N_delta_vecs()`: pass the right event window into
  `TurnDelta.build(...)`. Keep `prev_ctx`/`curr_ctx` for the current-state fields
  (HP-after, boosts-now) that legitimately come from the snapshot; fold *what happened*
  from the events.
- Keep the old diff `build()` available behind a flag until the equivalence harness is
  green, then delete (design §9 step 5 discipline).

### 2c. `TurnDelta` (`src/agents/training/battle_context.py`, 733 lines)
- `TurnDelta` dataclass at **L372**; `build(cls, prev_ctx, curr_ctx, action)` at **L493**;
  `empty()` at **L707**. `BattleContext` at L112, `from_battle` at L262. Heuristics to
  retire: `_align_effectiveness` (L26), `_ko_before_acting` (L40),
  `_resolve_target_hp_delta` (L56, keep — still useful), `_derive_move_outcome` (L74,
  replaced by `TurnView.SideTurn.outcome`). `opp_all_last_move_ids` (L161/320/323/350) —
  retire (the log gives the move directly).
- **Current fields (31):** `our_move_id`, `our_switch_to`, `our_prev_active`,
  `opp_move_id`, `opp_switch_to`, `opp_prev_active`, `opp_move_known`, `our_hp_delta`(6),
  `opp_hp_delta`(6), `we_fainted`, `opp_fainted`, `our_failed_to_move`, `our_cant_reason`,
  `opp_failed_to_move`, `opp_cant_reason`, `our_boost_delta`(7), `opp_boost_delta`(7),
  `our_effectiveness`, `opp_effectiveness`, `we_moved_first`, `our_damaging_event`,
  `opp_damaging_event`, `phase_is_forced_switch`, `our_hp_after`(6), `opp_hp_after`(6),
  `our_target_hp_delta`, `opp_target_hp_delta`, `our_move_outcome`, `opp_move_outcome`,
  `our_move_crit`, `opp_move_crit`.
- **NEW fields (Step 4):**
  - `our_faint_count: int`, `opp_faint_count: int` (replace/augment the bools).
  - `our_faint_causes`, `opp_faint_causes` — multi-hot over a fixed cause vocab. Propose
    `FAINT_CAUSES = (attack, hazard, weather, status, recoil, selfko, leechseed, other)`
    (8). Derive: for each FAINT event (species, side) in the window, find the cause from
    the preceding DAMAGE/SETHP `reason` on that species (`Spikes`→hazard, `Sandstorm`/
    `Hail`→weather, `psn`/`tox`/`brn`→status, `Recoil`→recoil, `Leech Seed`→leechseed,
    move-target with no `[from]`→attack, Explosion/Self-Destruct user→selfko, else other).
    Showdown carries all of these in the `[from]` clause — already captured in our DAMAGE
    event `value["reason"]`.
  - `our_attempted_move_id`, `our_attempted_switch_to` (decode the pressed `action`:
    `<6`→switch `prev_ctx.our_team_order[action]`; `6–9`→`prev_ctx.active_move_ids[a-6]`;
    `10`→struggle). Preserved even when the move never fired. (Opp attempted action is not
    observable — leave out, don't guess.)
  - Keep `phase_is_forced_switch` (= "extended/continuation"); its inverse is "start of
    attack."

### 2d. Encoder (`src/agents/observation/turn_delta_encoder.py`, TURN_DELTA_DIM=110)
- Offsets are named `OFFSET_*` / `*_DIM` constants — **never hardcode indices** (project
  rule). The 6 species IDs are a **contiguous tail** the extractor slices for embedding
  (`OFFSET_OUR_ACTOR_SPECIES..OFFSET_OPP_SWITCH_TO_SPEC`); **insert new fields BEFORE that
  tail.**
- Add: faint counts (2, normalized e.g. /6), faint-cause multi-hots (2×8=16),
  attempted-action — attempted move id is a raw int the extractor must embed via
  `move_embedding`, so it joins the embedded-ID set (see below); attempted_switch_to is a
  species id → joins the species tail (making it 7 species IDs, or a separate slot).
  Wire the cant one-hot here too (CANT_DIM=12 ×2 sides = 24) using
  `gen3_effects.encode_cant_reason` (replaces the older narrower cant onehot if present).
- **`features_extractor.py` coupling:** `embed_delta_slot` (in `Embeddings`) slices the
  species tail `[OFFSET_OUR_ACTOR_SPECIES:OFFSET_OPP_SWITCH_TO_SPEC+1]` and the move/type
  raw-int positions. `turn_delta_embed_dim(layout)` computes the embedded width as
  `2·move + 2·type + 6·species + (TURN_DELTA_DIM-10)`. If you add embedded IDs (attempted
  move/switch), update BOTH the slice in `embed_delta_slot` AND the formula in
  `turn_delta_embed_dim`, and the `OFFSET_*` constants imported at the top of
  `features_extractor.py` (lines ~15-25). Pass-through scalars need no extractor change.
- **`missing→0`:** every new field encodes 0 when absent (no attempt, no faint, unknown).

### 2e. Versioning + docs
- Bump `ARCH_SIGNATURE` `gen3_live_state_v1 → gen3_turn_delta_v2` (or similar) in
  `src/agents/model/model_version.py` (~L68); add a vN doc-comment block (follow the v8
  example just added). `MODEL_CONFIG_VERSION` only if a `ModelVersion` field changes.
- Update CLAUDE.md obs table (Turn-history row `N_HISTORY_TURNS × TURN_DELTA_DIM`) +
  the "Each TurnDelta slot" prose. Regenerate dim numbers from live constants, not memory.
- `snapshot_test.py`/`state_encoder_test.py` already compute dims from constants — should
  auto-track. `turn_delta_encoder_test.py` + `features_extractor_test.py` pin the layout;
  update the offset assertions.

### 2f. Tests / gates (run with the conda interp, `PYTHONPATH=$PYTHONPATH:src`)
1. Unit: extend `turn_view_test.py` for faint-cause folding (scripted: spike-sack,
   sandstorm KO, double-KO Explosion, recoil); new `turn_delta_*` unit for attempted-
   action + multi-KO + missing→0. `turn_delta_encoder_test.py` offsets.
2. `pytest src/ -m "not integration and not e2e" -q` — full suite green.
3. **Equivalence harness (design §8.3):** run M turns through BOTH the old diff `build`
   and the new event fold; assert the shared fields match except in the gap=0 corners,
   and report the diff rate. This is the gate that justifies retiring the heuristics.
4. e2e: `python src/agents/battle/event_log_fuzz_e2e_test.py 80` (extend it to validate
   the new faint-cause + attempted-action against the raw protocol).
5. Smoke: `python src/main/train_rl_agent.py --debug --steps 4000 --device cpu` — expect
   round-trip PASS, episodes finishing, eval, **0 crash-don't-drop, 0 traceback**.

---

## 3. Step 5 — Reward manager onto TurnView + LiveView
`src/agents/training/reward_manager.py` (`Gen3RewardManager`, ~900 lines). Most terms
already read `TurnDelta` fields (HP/faints/boosts/effectiveness) — those keep working
through the folded delta. A handful re-check current board (spikes layers, current boosts,
opp move/type) → read `LiveView`. Terminal `battle.won/lost/finished` stays on the battle
object. The new multi-KO/cause + attempted-action fields enable cleaner reward shaping
(e.g. reward a clean KO vs a hazard/weather chip differently; penalize a wasted attempted
move). Gate: reward-invariants e2e + a short retrain comparison (per memory:
reward changes need a retrain to measure).

## 4. Step 6 — Replay recorder lossless
`src/agents/training/replay_recorder.py` (+ `battle_recorder.py`). Today it records
decisions + outcomes but **no move ids, no opponent moves, no hit/miss/crit, and collapses
faint→switch chains.** Re-source from the event log so a replay reconstructs the whole
turn (actor→move→target→effectiveness→outcome→faints-with-cause). `states.npz` (model I/O)
is orthogonal; the per-turn JSON schema gains fields.

---

## 5. Hard-won lessons (carry forward)

- **Crash-don't-drop is load-bearing and it works.** The volatile/cant allowlists RAISE on
  anything unclassified. The tripwire caught `focuspunch`, `struggle`, and `flashfire` as
  real gen3 volatiles — the last one only in the *training smoke*, not unit tests. Lesson:
  **volatiles come from BOTH moves.ts AND abilities.ts** (Flash Fire). Any new
  allowlist-derivation must scan both, ∩ the gen3 move/ability sets, ∩ poke-env's `Effect`
  enum (only enum ids reach `LiveView.volatiles`; sim-internal volatiles like
  `counter`/`mirrorcoat`/`twoturnmove` never emit a `-start` and must be excluded).
- **Source everything from the protocol/event log, never guess.** Weather permanence
  (ability=permanent vs move=5-turn) comes from the `|-weather|...|[from] ability:…` cause
  line, folded in `LiveView._fold_weather` — NOT from inspecting active-mon abilities.
  Same principle drives the whole migration; faint cause is the next instance.
- **poke-env counter semantics:** only `rage` among our volatiles is `is_action_countable`
  (carries a turns-active counter). Confusion/taunt/etc. counters are NOT tracked → encode
  binary (honest, not lossy). Graded states (perish/stockpile) carry their level in the id
  (`perish0..3`, `stockpile1..3`). `LiveView.volatiles` is `Mapping[str,int]` to preserve
  the counter.
- **The feature extractor is fully layout-driven.** Changing encoder dims propagates via
  `get_layout()` + auto-discovery — NO extractor code change for the obs blocks. The ONE
  coupling for TurnDelta is `embed_delta_slot` + `turn_delta_embed_dim` (the embedded-ID
  slice/formula). Pass-through scalars are free.
- **SideCondition id-form has underscores:** `light_screen`, not `lightscreen`.
  `LiveView.side_conditions` keys on `SideCondition.name.lower()`.
- **The delta window ≠ a protocol turn.** Use `event_cursor`/`events_since`.
- **Verify via tests/files, not Read.** The prior session's Read/Bash output was
  intermittently stale/garbled; edits silently no-op'd ("File has not been read yet").
  Lean on pytest exit codes, grep with exact substrings, and writing to `/tmp` then cat.
- **`/gen3ai-ship` only.** Never commit without the user typing it. All work in the
  worktree branch; main fast-forwards.
- **Retrain not yet run.** `gen3_live_state_v1` is shipped but no model has trained on it.
  The v3 run (~350M steps) is on an old arch and won't load v8 — that's correct/expected.

---

## 6. Concrete first moves for the fresh session
1. `git submodule update --init` + symlink `dist`/`node_modules` (worktree setup, see root
   CLAUDE.md).
2. Read `battle_context.py`, `episode_tracker.py`, `turn_delta_encoder.py`, `turn_view.py`
   fresh (clean tooling) to confirm the line numbers above.
3. Extend `TurnView` with faint-cause folding first (pure, unit-testable on hand-built
   event logs) — lowest risk, highest leverage.
4. Then `TurnDelta` fields + fold + episode_tracker window wiring; then encoder + ARCH +
   docs; gate with the equivalence harness + fuzz + smoke at each stage.
