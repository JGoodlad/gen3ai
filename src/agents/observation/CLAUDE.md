# CLAUDE.md — Observation Encoder (`src/agents/observation/`)

This directory builds the **2667-dim per-decision observation vector** (`Gen3ObservationEncoder.encode`;
the live value is `Gen3ObservationEncoder.dimension` — read it there, and see
`designs/ARCHITECTURE.md` § Observation for the full block table).
It runs once per agent decision across every training env, so it sits directly on the
training-throughput (FPS) critical path. Two independent things can regress here, and they
have **different** gates:

1. **Observation *values*** — if a change alters what the vector contains, it is
   **retrain-class**: bump `ARCH_SIGNATURE` in `src/agents/model/model_version.py` (see the
   root `CLAUDE.md` → Model Versioning). Value-neutral refactors do **not** bump it.
2. **Observation *build performance*** — if a change makes `encode` slower, training FPS
   drops for the entire run. **This file governs that gate.**

> **Off-hot-path exception — `belief_labels.py`.** This module (the pure builder of the
> hidden-opponent belief-aux labels) lives here for cohesion with the obs layer but is **NOT called
> by `encode`** — `Gen3Env.step/reset` invoke it only when `--opp-belief-aux-coef>0` or
> `--move-belief-mode != off`, to emit the privileged training-only `belief_species`/`belief_moves`
> (and, for move-belief known/both, `known_moves`; for `--spread-belief-coef>0`, `belief_spread`/`_mask` —
> and, under `--spread-belief-nature`, the inverted `belief_nature`/`belief_ev`(+masks) for the nature/EV
> decomposition, `gen3_nature_ev_belief_v1`, deterministically inverted from agent2's `mon.stats` so no leak;
> for `--hp-type-belief-coef>0`, `hp_type_label`/`hp_type_mask` — the opp Hidden-Power-type label,
> `gen3_typed_hp_belief_v1`) Dict keys (see
> `src/agents/training/CLAUDE.md`). So it adds **zero** cost to the default obs build (benchmark
> confirmed: `state_encoder.encode` unchanged, `belief_labels` absent from the profile). Changes to
> `encode` itself still trip the gate below.

---

## MANDATORY: run the obs-build benchmark on every change to this directory

**Any change to a file under `src/agents/observation/` — even a "pure refactor" or a
value-neutral one — MUST run the performance benchmark before and after the change and
confirm no meaningful regression.** This is not optional and applies to every edit, however
small. A one-line change to a hot loop (e.g. `reactive.py`, `pokemon.py`, `moves.py`,
`state_encoder.py`) can silently halve FPS.

The benchmark is `src/agents/training/obs_build_benchmark.py` (it lives in `training/` rather
than here only because a directly-run script puts its own dir on `sys.path[0]`, and
`observation/types.py` would shadow the stdlib `types` module — see the root `CLAUDE.md`
Benchmarks section).

### The workflow — do this for every change

```bash
export PYTHONPATH=$PYTHONPATH:src

# 1. BEFORE you edit: capture a baseline on the CURRENT code (stash/commit your change away,
#    or run on a clean checkout), saving the full output.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_before.txt

# 2. Apply your change.

# 3. AFTER: re-run with the SAME flags and the SAME machine load, and diff.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_after.txt

diff /tmp/obs_before.txt /tmp/obs_after.txt
```

**"the SAME machine load" is now checked for you** (`gen3_contention_robust_timeouts_v1`): the
benchmark calls `warn_if_contended()` at entry and prints a loud "THE BOX IS BUSY" banner with the
load average when the box is not idle. It still WARNS rather than refuses — a before/after pair run
back-to-back under the *same* load is exactly the same-load A/B this gate asks for — but a banner on
only one of the two runs means the comparison is void, so check both outputs before believing a diff.

If you cannot easily get a "before" (the change is already applied), compare against the
**canonical baseline pasted below** — but prefer a same-session before/after, because
absolute timings are machine- and load-dependent.

### What counts as a "meaningful regression" — use the LOAD-STABLE signals

Absolute milliseconds scale with whatever else the box is doing (training alone pushes load
past the core count and can inflate the numbers 2–3×). **Do NOT judge by the `ms` line.**
Judge by these load-independent metrics, in priority order:

1. **Total function calls per encode** = `<N function calls>` line ÷ `--reps`. This is the
   single best regression detector — it does not move with machine load. Baseline ≈
   **~3.46k calls/encode** — the post-`gen3_entity_rehome_v1` (v60) reference: deleting the two
   144-dim matchup matrices removed the whole `_expected_multiplier`/`_joint_expectation` loop
   family (measured same-session before/after at `--turn 25 --reps 400`, seed-0 battle:
   6,332 → 3,462 calls/encode, −45%; wall 0.373 → 0.246 ms, −34% — the Stage-3 refund,
   confirmed in reverse). History: ~6.44k was the post-`gen3_cpu_damage_deleted_v1` (v48)
   reference, measured
   same-session before/after at `--turn 25 --reps 300` on the seed-0 battle (7,396 → 6,444, −12.9%,
   from deleting the incoming-damage / move-effect / active-move-scalar producers). History for
   context: ~6.36k pre-`gen3_incoming_damage_v1`, ~6.85k after it, ~7.4k after the `v2` belief
   recalibration (crit term + the wider candidate set). Always judge by a same-session before/after,
   not the absolute. A jump of **>10%** above this is a regression — investigate.
2. **cProfile `tottime` top-of-list structure.** A *new* function climbing into the top ~10,
   or a known hot function's **call count** ballooning, means you added work to a hot loop.
3. **Component ratios** (`state_encoder.encode` vs cached turn-history vs `live_view`) and the
   **deque-cache multiplier** (`Nx saved`). If the turn-history "cached" line stops being a
   single encode (`~12x saved` collapses toward `1x`), the deque memoization broke.

A value-neutral refactor that adds <10% calls/encode and doesn't reshuffle the tottime top is
fine. Anything larger needs justification (or a revert).

---

## Canonical baseline (paste — the reference point for regressions)

Captured with `--turn 25 --reps 400`. Paths shown repo-relative. Absolute ms omitted from the
headline on purpose (load-dependent); the **call counts and ordering are the contract**.

> ⚠️ The pasted block below predates `gen3_entity_rehome_v1` (the matchup deletion): the
> matchup-era hot list (`effective_multiplier_by_types`, `reactive.py:encode` at ~44% of encode,
> `_joint_expectation`) no longer exists. Current headline: **~3.46k calls/encode**, encode
> ≈ 0.25 ms idle-box, no matchup loop in the profile. The block is kept for the v48-era shape
> until the next full re-baseline.

```
PER-DECISION OBS BUILD BENCHMARK  (obs dim <live>, turn 25, history slots N, opp mons w/ revealed moves 5/6)

  full per-decision obs build  :  ~0.5–1.2 ms   (LOAD-DEPENDENT — not a regression signal)
    state_encoder.encode       :  ~79% of build
    turn-history (cached, 1 enc):  ~6% of build   (recompute-all-10 is ~11–13x slower → deque cache working)
    live_view() alone          :  ~15% of build

  Total: ~2.74M function calls / 400 reps  ==>  ~6.85k calls per encode   <-- PRIMARY REGRESSION METRIC
  (the +0.49k vs the pre-feature 6.36k is the gen3_incoming_damage_v1 belief loop; per-species
   candidate/stat work is lru_cached, so only the per-defender damage/outspeed math is per-decision.
   gen3_incoming_damage_v2 adds ~+6.6% on top: the crit term doubles the per-candidate damage calc and
   the wider candidate pool fills more (defender, channel) pairs — _channel_threat goes ~10→12 calls/
   encode. Still no single dominant hot loop; the revealed-HP typed expansion is NOT lru_cached (it
   tracks the per-episode HP tracker) but only fires when a bare hiddenpower is revealed.)

  Top functions by tottime (no single dominant hot loop — the matchup work is now spread thin):
   ncalls  tottime  cumtime  function
    80800    0.048    0.088   agents/gen3_mechanics.py:effective_multiplier_by_types (memoized; chart lookup)
      400    0.046    0.261   agents/observation/reactive.py:encode                  (cumtime ≈ whole matchup block)
     4800    0.036    0.101   agents/observation/moves.py:encode
     4800    0.032    0.111   agents/battle/live_view.py:from_pokemon
    42800    0.028    0.042   poke_env/battle/move.py:entry                          (poke-env Move property)
    80800    0.028    0.116   agents/observation/reactive.py:_joint_expectation
   238800    0.026    0.035   enum.__hash__                                          (lru_cache key hashing)
     4800    0.025    0.189   agents/observation/pokemon.py:encode
    26400    0.012    0.040   poke_env/battle/move.py:max_pp
     4800    0.012    0.023   agents/observation/types.py:encode
   264800    0.011    0.011   {builtins.len}
      400    0.011    0.596   agents/observation/state_encoder.py:encode             (cumtime ≈ whole obs)
```

**Reading it:** there is **no single dominant hot loop** anymore — the matchup encoder's
per-cell poke-env property reads were hoisted to team level (`reactive._defender_terms` /
`_attacker_type_dist`, computed once per mon / per (attacker, move) instead of per cell), and
the per-mon move category is memoized by id (`moves._category_val`), so `move.entry` dropped
from ~158k to ~43k calls and `pokemon.ability` / `move.type` left the top list. Cost is now
spread across `effective_multiplier_by_types` (the memoized chart lookup — the irreducible
per-cell core), the matchup `encode` loop overhead itself, and the per-mon encoders. Type
effectiveness must stay a memoized chart lookup (`effective_multiplier_by_types` + `_eff_cached`
in `gen3_mechanics.py`) — `PokemonType.damage_multiplier` must **not** reappear here (if it
does, something bypassed the chart). The matchup block (`reactive.encode`) and the per-mon
`pokemon.encode` / `moves.encode` chain are the next-largest cumtime; `live_view.from_pokemon`
(rebuilt ×12/encode) is shared with reward/replay, so it carries a wider blast radius.

---

## Pitfalls that have caused regressions here

- **Calling `PokemonType.damage_multiplier` / `effective_multiplier(move_type, mon)` per cell.**
  Use the value-based `effective_multiplier_by_types(move_type, t1, t2, ability, status)` and
  read the mon's attributes once outside the loop. The object wrapper re-reads poke-env
  properties every call.
- **Re-reading poke-env properties inside the inner loop.** `move.type`, `mon.type_1/2`,
  `mon.ability`, `move.category` are properties that do real work (`move.entry`,
  `GenData.from_gen`); hoist them above the loop. The matchup matrices already do this
  (`reactive._defender_terms` / `_attacker_type_dist` read each mon / (attacker, move) once,
  not per cell) and the per-mon move category is memoized by id (`moves._category_val` — a
  process-global cache off the *live* `move.category`, NOT a `gen3_data.moves` re-derivation,
  which disagrees for fixed-power moves). Do not reintroduce a per-cell / per-slot property
  read.
- **Breaking the turn-history deque cache** (`EpisodeTracker.prev_N_delta_vecs`): if the
  benchmark's "recompute all 10" multiplier collapses toward 1×, you've reintroduced the
  per-step O(N) re-encode.
- **Wrapping live mons in proxy objects** with `__getattr__` (the deleted
  `_AbilityOverrideMon`): `__getattr__` is slow and gets hit once per attribute per cell.

## Value-correctness (separate from perf, but also gated)

Changes to *what the vector contains* are validated by the bridge-backed fuzz tests
(`*_fuzz_test.py`, real battles, protocol-truth checks) and the unit tests in this directory.
If your change is meant to be **value-neutral**, prove it: the effectiveness fast-path, for
example, is pinned byte-for-byte by the exhaustive parity test in
`src/agents/gen3_mechanics_test.py`. Obs-value changes are retrain-class → bump
`ARCH_SIGNATURE`.

---

## Observation vector layout (per-block reference)

`designs/ARCHITECTURE.md` § Observation carries the top-level block table (block → dims → offset)
and the per-mon slot layout, derived from the live constants. **This** is the per-block detail:
what each field MEANS and where it is sourced from. All offsets are computed from named constants
— never hardcode indices.

**Per-Pokémon slot (116 dims):** the 110 below + the 3-dim recency block + the 1-dim
protect-odds field + the 2 appended trapping bits + the appended active flag.
**Recency block** at `POKEMON_RECENCY_OFFSET` (109) — [turns_since_seen, turns_since_acted,
turns_since_was_hit], TURN-ANCHORED (`cur_turn − event_turn`, clamped; on-field mon reads 0;
never-tracked reads 1.0 max staleness), log-saturated over a 10-turn cap, BOTH sides (public —
every reset derives from observed protocol events), sourced from the EpisodeTracker-owned
`RecencyTracker` (the same per-decision event window the TurnDelta fold reads) and threaded
into `encode(recency=…)` like the progress clock. Fuzz gate:
`poke_env_gaps/recency_fuzz_test.py` (encoded scalars == an independent full-log recount +
decision-time active log, per mon per decision). **Protect-odds field** at
`POKEMON_PROTECT_OFFSET` (112, gen3_entity_rehome_v1): P(a Protect/Detect/Endure by THIS mon
succeeds now) under the gen3 floored-doubling stall rule (100/50/25/12.5, floor 1/8), from the
LiveView `protect_counter` — EVERY mon owns its stall state (a benched mon truthfully reads 1.0;
the counter resets on switch). Pinned by `protect_success_prob_fuzz_test.py`. **Appended tail**
(state_encoder): `POKEMON_TRAPPED_OFFSET` (113) + `POKEMON_MAYBE_TRAPPED_OFFSET` (114) — the
OUR-side LegalActions trapping bits, nonzero ONLY at our active slot (`maybe_trapped` is the
high-value trap-risk bit; fuzz gate `action/trapping_signals_fuzz_test.py`, which also asserts
bench slots stay zero) — then the ACTIVE flag at `POKEMON_ACTIVE_OFFSET` (115), deliberately
LAST in the slot (the model's `hp_and_active[:, :, -1]` convention is load-bearing).
Original 110: species ID + 6 base stats, item ID + known + consumed, 2 type
IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction,
species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**,
**HP-candidate block (17 dims)**, **sleep-wake belief (3 dims)**. The item block is 3 dims:
`[item_id, known, consumed]` — `consumed=1` when the item was spent this battle (Berry
activated, Knock Off, Trick, etc.) and `item_id` retains the identity of the consumed item so
the model knows what was lost. `species_known = 1.0` for all populated slots (own team and
revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter:
`min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(turns_poisoned, 8) / 8`
(practical max before fainting with Leftovers).

**Sleep-wake belief (3 dims, `gen3_sleep_wake_belief_v1`, layout in `sleep_belief.py`):** zeros
unless the mon is asleep, else `[sleep_is_deterministic, p_wake, sleep_counter_reliable]`. poke-env
exposes only `Status.SLP` + a noisy `status_counter`, NOT the rolled duration / remaining time / source
move, so a policy reading the raw counter would have to LEARN the gen3 sleep RNG and can't tell a
deterministic Rest from a random opp-sleep at the same counter. We **compute** the wake odds from the
adversarially-verified gen3 tables — opp `time = random(2,6)` ∈ {2,3,4,5} (the gen3 mod overrides the
modern `random(2,5)`), Rest `time = 3` fixed, Early Bird halves — `p_wake` = P(wake on the next move
attempt | observed counter K, source, Early Bird), **marginalising the opponent's Smogon Early-Bird
prior** (collapsing to exact 0/1 for our own mon or a revealed opp). `sleep_is_deterministic` (1.0 =
Rest) selects which table; it's read from our **event log's `[from]` clause** (poke-env discards it).
`sleep_counter_reliable` drops to 0.0 once a Sleep Talk / Snore turn has corrupted the counter (+3 per
turn, empirically verified) — instead of reconstructing Showdown's `skippedTime` switch refund. The
counter→p_wake mapping and the source/reliability bits are **fuzz-calibrated against the real sim RNG**
(`poke_env_gaps/sleep_wake_fuzz_test.py`: per-decision obs wiring exact + empirical wake-frequency ==
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
are populated by the poke-env fork's **`backfill_teambuilder_spread`** (`Battle.parse_request`):
gen3ou has no team preview, so `apply_teambuilder_team` never attaches the spread — the backfill
matches the declared teambuilder team to the request-built team by species and fills in
IVs/EVs/nature (spread only, never re-running `_update_from_teambuilder`). Without it this block
emitted a constant fallback (all-31 IVs, 0 EVs, neutral nature) for every own mon.
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

- `turns_since_progress` — the log-saturated no-progress clock (`log(1+min(n,10))/log(11)`), owned by
  the **EpisodeTracker's `ProgressClock`** (NOT LiveView — it is cross-turn state) and threaded into
  `encode()` like the HP tracker. The reward's `no_progress_tax` keys on the SAME clock instance, so
  obs and reward-key are one value. Lets the model state-condition on the penalty it is about to be
  charged.
- `wish_floating` — the pending-Wish heal: a flat `WISH_HEAL_FRACTION` (≈0.5; gen3 Wish heals the
  RECIPIENT's maxhp/2, so the fraction is constant and GIGO-proof) when a Wish cast last turn
  resolves at the end of this turn, else 0. Slot-keyed, so it survives faint / Roar-phaze / switch.
  poke-env tracks none of it → reconstructed from the event log (`wish_belief.py`).

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
the network. Both orders are therefore live at once; the pointer action head resolves this by
permuting on move-num IDENTITY, which makes a misaligned logit unrepresentable.

**Move-effect block and incoming-damage / OHKO belief block — BOTH DELETED from the observation.**
Their long descriptions moved verbatim to `designs/CHANGELOG.md` §5. Where the signal lives now:

| Deleted obs block | Dims | GPU home |
|---|---|---|
| action-aligned move-effect flags | 44 | static mechanics → the `MoveLatentEncoder` latent (`--move-latent`); board-conditional `status_will_land` → `DamageOperator._status_landing` (`--damage-outgoing`); `pp_fraction` → the per-mon move slot (unchanged) |
| per-our-mon incoming-damage / OHKO belief | 51 | the `DamageOperator`'s incoming block, off the LEARNED move belief instead of this block's FIXED usage prior — that substitution was the whole point of `--damage-op` |
| active-move scalars (base power ×4, type mult ×4) | 8 | the op's OUTGOING per-move block, request-ordered, with real gen3 physics rather than `bp/200` and `mult/4` |

**`agents/observation/incoming_damage.py` STAYS** — the reward PBRS (`reward_manager.py`) and the
prober import its math core, and its fuzz test now targets `encode_block` directly. Only the obs
write was removed.
> **Downstream reader:** the prober engine (`src/main/prober/engine.py`) resolves its obs
> offsets at runtime from `get_layout()` (`ObsOffsets`), with `0 = absent` for deleted blocks
> (`mm_off`, and since gen3_entity_rehome_v1 also `om_off`/`tm_off` — ThreatView/saliency
> no-op). Its pinned regression test (`prober/engine_test.py::
> test_offsets_resolve_matches_layout`) fails on any layout move; update the pins there.

**TurnDelta slot (159 dims, layout in `turn_delta_encoder.py`):** all offsets computed from
named `OFFSET_*` / `*_DIM` constants — never hardcode indices. TurnDelta is **folded from the
event log** (`Gen3Battle.events_since(cursor)` per decision window; see
`src/agents/battle/CLAUDE.md`) rather than diff-heuristics.

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
the out-of-band `CHOICE_REJECTED` event (`TurnView.attempted_rejected`). On a rejected pivot
`our_switch_to` is the unknown sentinel (the switch never happened) while `attempted_switch_to`
names the mon we tried to bring in; both are zero on every turn with no rejection. Opp attempted
action is not observable. Faint *counts* are kept on the `TurnDelta` dataclass (for reward) but
not encoded — redundant with the faint flags + cause popcount.

**Embedded-ID manifest (the layout-driven contract):** which slot positions carry raw embedding
IDs and which table each routes to is declared once in `TURN_DELTA_EMBEDDED_IDS` (in
`turn_delta_encoder.py`). Both the encoder's layout and `Embeddings.embed_delta_slot` read it —
there are **no hardcoded positions in the extractor** and the embed-width formula derives from
the manifest, so a raw id can never silently leak through as a scalar. Adding an embedded ID is
a one-line manifest change. Current manifest: 3 move IDs (our/opp/attempted) → 3×16, 2 type IDs
→ 2×16, 7 species IDs (our/opp actor + our/opp target + our/opp switch_to + attempted_switch_to)
→ 7×32 = 48 + 32 + 224 = 304 embedded dims + 147 pass-through scalars = 451-dim per slot.
Positional encodings are added, one self-attention pass runs, and the last (most-recent) slot's
output flows into the projection block. All zeros on the first turn of each episode. Actor
species resolution prefers `damaging_event.user_species` (protocol-truth) and falls back to
`prev_active` for switches and non-damaging moves; target species comes from the OTHER side's
`damaging_event.target_species`. Species ID 0 is the unknown sentinel.
