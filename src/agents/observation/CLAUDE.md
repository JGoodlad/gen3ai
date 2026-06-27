# CLAUDE.md — Observation Encoder (`src/agents/observation/`)

This directory builds the **2992-dim per-decision observation vector** (`Gen3ObservationEncoder.encode`).
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
> for `--hp-type-belief learned` + coef, `hp_type_label`/`hp_type_mask` — the opp Hidden-Power-type label,
> `gen3_opp_hp_type_belief_v1`) Dict keys (see
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

If you cannot easily get a "before" (the change is already applied), compare against the
**canonical baseline pasted below** — but prefer a same-session before/after, because
absolute timings are machine- and load-dependent.

### What counts as a "meaningful regression" — use the LOAD-STABLE signals

Absolute milliseconds scale with whatever else the box is doing (training alone pushes load
past the core count and can inflate the numbers 2–3×). **Do NOT judge by the `ms` line.**
Judge by these load-independent metrics, in priority order:

1. **Total function calls per encode** = `<N function calls>` line ÷ `--reps`. This is the
   single best regression detector — it does not move with machine load. Baseline ≈
   **~6.85k calls/encode** (≈2,740,801 / 400, post `gen3_incoming_damage_v1` — was 6.36k before the
   incoming-damage belief block; that feature is a justified +7.7%, the new reference). The
   `gen3_incoming_damage_v2` belief recalibration (crit term + wider candidate set: revealed-HP
   typed expansion, Return/Frustration, the 0.12→0.05 floor + 4→6 cap) added a further justified
   **~6.6%** on top (measured same-session before/after on the seed-0 battle) — so expect ≈7.3k on
   this reference machine; always judge by a same-session before/after, not the absolute. A jump of
   **>10%** above this is a regression — investigate.
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

```
PER-DECISION OBS BUILD BENCHMARK  (obs dim 3391, turn 25, history slots 10, opp mons w/ revealed moves 5/6)

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

The root `CLAUDE.md` carries the summary block table (block → dims → offset, total **3409**).
This is the detailed per-block layout. All offsets are computed from named constants — never
hardcode indices.

**Per-Pokémon slot (110 dims):** species ID + 6 base stats, item ID + known + consumed, 2 type
IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction,
species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**,
**HP-candidate block (17 dims)**, **sleep-wake belief (3 dims)**, active flag. The item block is 3 dims:
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

**Global env (18 dims, layout in `global_env.py`):** weather block (7: one-hot + cause-aware
permanence + turns-remaining), spikes ×2 (2), log-turn (1), per-side screens (8: Reflect /
Light Screen / Safeguard / Mist × both sides).

**Reactive block (414 dims, layout in `reactive.py`):** 19 scalar dims (incl. 2 `gen3_wish_wired_v1`
`wish_floating` scalars at `vec[17]`/`vec[18]`, our/opp side — the pending-Wish heal, `WISH_HEAL_FRACTION`
≈0.5 when a wish cast last turn resolves this turn, else 0; see `sleep_belief.py`-style `wish_belief.py`),
then the 44-dim **move-effect block** (`gen3_move_effects_v1` + `gen3_status_cure_moves_v1`), then
the **51-dim incoming-damage / OHKO belief block** (`gen3_incoming_crit_split_v1`, at offset 63 — see
below), then the two 144-dim matchup matrices (`our_matchups` at offset 114, `their_matchups` at 258),
then the **12-dim active-req-moves block** (`gen3_op_move_align_v1`, at offset 402): OUR active mon's
4 moves in **REQUEST order** (action 6+k) — `[move_num ×4, resolved_type_id ×4, legal_now ×4]`, sourced
from `legal.move_slots` (same source as the action mask), so the model's DamageOperator OUTGOING per-move
methods (`_outgoing_block`/`_status_landing`/`_outgoing_matrix`) read request order rather than the per-mon
block's sorted-by-id order. These are embedding IDs (HP → num 237; typed-HP type resolved) consumed ONLY
by the op via `ObsUnpack` (`ctx.our_active_req_move_{ids,type_ids,legal}`), so the block sits AFTER the
matchups and is EXCLUDED from `non_matchup_rest` (the raw-scalar path that stops at the matchup offset).
Scalars: active-move power ×4 (/200)
+ active-move multiplier ×4 (/4), fainted counts ×2, active-status flag (1), `forced_struggle` (1),
**(`gen3_move_slot_align_v1`: these per-move scalars — and the move-effect block below — are filled
in REQUEST-slot order via `legal.move_slots` (action 6+i ↔ slot i, disabled moves KEPT, typed-HP
resolved off the moveset) by `reactive._request_slot_moves`, NOT `battle.available_moves`, which
drops disabled moves and used to shift every later feature off its action logit; an unwritten slot
reads the neutral 0.25 (1×) default, never the old np.ones → phantom 4×.)**
the two **gen3_trapping_signals_v1** bits — `trapped` (1) and `maybe_trapped` (1) — the
**gen3_markovian_progress_v1** scalar `turns_since_progress` (1, `vec[14]`), and the **two
gen3_protect_odds_v1 scalars** `protect_odds` (our active `vec[15]`, opp active `vec[16]`). All are
sourced server-authoritatively / from the read-model: trapped/maybe_trapped from the per-decision
`LegalActions` snapshot (`legal.trapped` / `legal.maybe_trapped`); `turns_since_progress` is the
log-saturated no-progress clock (`log(1+min(n,10))/log(11)`), sourced from the **EpisodeTracker-owned
`ProgressClock`** (NOT LiveView — it is cross-turn state), threaded into `encode()` like the HP
tracker (the reward's `no_progress_tax` keys on the SAME clock instance — one value, obs==reward-key);
`protect_odds` = `gen3_mechanics.protect_success_probability(mon.protect_counter)` read off each active
mon's **`LivePokemon.protect_counter`** (the consecutive-successful-stall counter the LiveView surfaces)
— P(a Protect/Detect/Endure succeeds NOW) under the gen3 floored-doubling stall rule (100/50/25/12.5,
floor 1/8; Showdown gen3 inherits gen4→gen5, NOT the base `*3`). It is the model's ONLY view of the
stall counter (poke-env doesn't enumerate the `stall` volatile, and turn-history saliency decays
before a chain can be counted); public both sides (the opp's counter derives entirely from their
revealed move stream → no leak); pinned by `protect_success_prob_fuzz_test.py` (encoded scalar ==
the formula per the live counter, + the empirical % match). They sit BEFORE the
matchups so the extractor picks them up in `non_matchup_rest` automatically (it reads the matchup
offset from the layout). `trapped` is redundant with the mask but explicit; `maybe_trapped` is the
high-value trap-risk bit; `turns_since_progress` lets the model state-condition on the anti-stall
penalty it's about to be charged; `protect_odds` lets it price the declining success of a repeated
Protect (it failing the more often it's used in a row).

**Move-effect block (44 dims, `gen3_move_effects_v1` + `gen3_status_cure_moves_v1`):** 4 move slots in **REQUEST order** (so
feature slot *k* lines up with action logit 6+*k* — enforced via `legal.move_slots` since
`gen3_move_slot_align_v1`; pinned by `move_alignment_fuzz_test.py`) × 11 features each — `is_boost`, `is_heal`,
`is_protect`, `is_phaze`, `is_hazard`, `inflicts_status`, `status_will_land`, `pp_fraction`,
`status_will_land_known`, **`cures_self_status`**, **`cures_team_status`**. The
only per-move signals that previously reached the policy head in action order were base power and
the type multiplier, so for status/utility moves (power 0, neutral multiplier) every option looked
identical at the head — the model could not tell a setup move from a heal from a wasted Toxic, nor
that a move CLEARS status. **`gen3_status_cure_moves_v1`** added the last two bits: `cures_self_status`
(Refresh — clears the user's own status) and `cures_team_status` (Heal Bell / Aromatherapy — clear
the whole party's). They are **static curated facts** (the cure lives in an onHit callback, invisible
declaratively → a curated override in the acquisition tool, like Belly Drum), read by the head against
the per-mon status one-hots it already sees — a **prober-verified gap**: with no cure bit, the head
conditioned its own status onto Recover/switch (intervention: removing a Toxic moved P(recover)/switch
~11pp each) but onto Refresh only ~1.5pp, so it under-used the cure (~1.4% when badly poisoned) and let
Toxic stack. `cures_team_status` is party-scoped on purpose so the model can value Heal Bell off the
BENCH statuses, not just the active's. The
static flags come from the `gen3_data.moves` facade (`MoveData.is_boost/is_heal/...`), derived in
the acquisition tool from the field **Showdown** keys each mechanic on (`flags.heal`,
`volatileStatus`, `forceSwitch`, `sideCondition`, primary `status`, declarative self-positive
boosts) PLUS a curated callback override for **Belly Drum** (its +6 Atk lives in an `onHit`
callback, invisible declaratively); **Memento** is correctly excluded (foe-target negative boosts
+ self-faint). Resolved **live** in the encoder: **Curse**'s setup (only a self-boost for a
non-Ghost user) and `status_will_land`. The latter is a **prior-weighted probability in [0,1]**
(`gen3_mechanics.status_land_probability`), built the same "priors first, then confirmation"
way the matchup cells handle abilities: it is 0 on a certain block (type immunity, already
statused, Substitute — ability-independent), else `1 − P(ability blocks this status)` over the
opponent's ability distribution (`_resolve_ability_distribution` — the Smogon prior for an
unrevealed mon, collapsing to an exact 0/1 once the ability is revealed via `-immune [from]
ability:`). So an unrevealed Snorlax reads ≈0.14 for Toxic (Immunity-dominated) instead of a naive
1. The trailing **`status_will_land_known`** bit disambiguates prior from confirmed — the SAME
routing the per-mon ability block uses for its `known` flag: 1 when the value rests on confirmed
info (a type-certain hard block, or the opponent's ability is revealed via `_ability_revealed`,
the exact predicate `AbilitiesEncoder` uses), 0 when it's still a Smogon-prior estimate a reveal
could move. Without it the model couldn't tell a confirmed 0.0/1.0 from a prior one (a real
discrepancy vs how abilities are routed; this closes it). Sits before the matchups → flows to BOTH
the policy and value projection heads via
`non_matchup_rest` (input widths auto-discovered). Garbage-in discipline: each static flag is
sourced from Showdown's actual representation, never guessed from the move name — see
`tools/pokemon_data_extractor/sync.py:build_moves`.

**Incoming-damage / OHKO belief block (51 dims, `gen3_incoming_crit_split_v1`, at reactive offset 63,
before the matchups → routed to both heads via `non_matchup_rest`):** the opponent active's threat to
*us* as a calibrated belief, not a calc. (This block is the fixed *usage-prior* collapse; the model-side
`DamageOperator` (`--damage-op`) computes the SAME kind of belief from the model's LEARNED move belief
instead, and `--mask-incoming-damage-obs` can zero this block out of the MODEL's view to A/B that
replacement — the block stays in the obs at its fixed dim, and the REWARD PBRS still reads it from
`live_view`. See `src/agents/model/CLAUDE.md` → the damage-operator / unified-belief notes.) Per our 6 team mons (slot-aligned): `[phys_expdmg_frac,
spec_expdmg_frac, phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta, p_outspeed,
threat_revealed]` (8 × 6 = 48), then 3 opp-active recovery scalars
`[recovery_rate, cures_status(P rest), recovery_known]`. **The per-mon field offsets are NAMED
constants (`IDX_PHYS_EXP … IDX_THREAT_REVEALED`, `IDX_RECOVERY_*`) in `incoming_damage.py` — the
single source of truth for this layout.** The producer assembles each slot FROM those names (with a
`_PER_MON_FIELDS == PER_MON` import-time assert) and every single-field consumer reads by name
(the reward PBRS `block[base + IDX_OUTSPEED]`, the fuzz test), so a future field insert can't
silently desync a read — the failure mode that once made the reward PBRS read `phys_crit_delta` as
`p_outspeed` (the crit-split pushed outspeed 4 → 6 but a hardcoded `block[base + 4]` stayed). Whole-
slot reads (the prober decode) full-tuple-unpack, which fails loudly on a width change instead. **`gen3_incoming_crit_split_v1` (PER_MON 5→8,
block 33→51, obs 3391→3409):** P(KO) is the modal `*_pko_nocrit` (the roll integration with NO crit —
the outcome you plan around); the crit risk is exposed as the **DELTA** `*_crit_delta`
(crit-inclusive − no-crit ∈ [0, `_CRIT_P`]) rather than the near-redundant absolute crit-inclusive line
(which equals nocrit + a ≤6% tail and is buried after standardization). The delta is the explicit crit
"tax" — a decorrelated feature a small net can read — so the policy/critic price the modal line without
over-weighting uncontrollable crit RNG (the prober's representation probe flagged the damage SPREAD as
under-encoded, and the plateau diagnosis showed RNG-driven critic craters; the prober reconstructs
crit-inclusive = nocrit + delta to preserve the loss-taxonomy meaning). `threat_revealed` is the
dominant KO threat's `p_in_set` provenance: **1.0 = a revealed move (we KNOW), <1.0 = a usage-prior
GUESS, 0.0 = no candidate can KO** (read jointly with the pko channels) — the "how much are we guessing"
signal (provide-the-fact, not bake-the-prior). P(KO)/expected-damage are the §6.1 belief —
**max over `revealed ∪ usage-prior` candidate moves** of `P(move in set) · P(KO|move)`, routed by gen3
**TYPE-category** (Bug/Rock/Ground/… physical, the rest special), using the gen3 damage formula with a
**fixed-damage branch** (Seismic Toss/Night Shade/Dragon Rage/Sonic Boom carry constant damage despite
the dex STATUS tag; respect type immunity — 0× vs Ghost), a **variable-power branch** (Return/Frustration
read BP 0 in the dex → priced at 102), the gen3 **Explosion/Self-Destruct Def-halve**
(gen≤4), Reflect/Light-Screen/Substitute/burn/weather modifiers, the opponent's offensive-stat tail
(the **0.95 max-EV+ percentile**, `priors.stat_distribution`), and a closed-form roll→P(KO) **blended
with a gen3 crit term** (`_CRIT_P`=1/16, ×2, screen-ignoring). `p_outspeed` is `P(our_spe > opp_spe)`
over the opp Speed *distribution* (the
hidden nature/EV) with observed boosts/paralysis. **v2 belief-VALUE recalibration** (same 33 dims, same
obs dim — values only, so retrain-class not weight-shape): the v1 belief was too timid on the near-OHKO
tail and silently zeroed missing coverage (run_20260606_204351: 17% of direct-hit deaths read
P(KO)<0.25). v2 (1) de-timids P(KO) — the **crit term** + the **raised offensive tail (0.85→0.95)** lift
the KO flag on near-OHKOs while expected-damage re-normalises to the MEAN (∝ `atk_mean`), so the chip
belief is unchanged; and (2) widens the candidate set so the killing move isn't silently absent — a
**revealed bare `hiddenpower`** (dex BP 0) expands into per-type candidates (~70 BP, typed from the **HP
tracker**'s observation-narrowed distribution / Smogon HP prior — the tracker is threaded into
`encode_block`), Return/Frustration are priced, and the prior **floor/cap widen (0.12→0.05, 4→6 per
channel)** so a low-usage super-effective coverage move survives into the pool (the per-defender max over
`p_in_set·P(KO)` is the real type-effectiveness gate, so extra low-usage candidates only ever surface a
genuine SE threat — they can't inflate a neutral one). **Two modules, deep split:** the pure, poke-env-free
math core (formula, roll→P(KO) + crit, P(outspeed), the `Candidate`/`Defender`/`AttackerThreat` beliefs,
`compute_team_block`) is `incoming_damage.py`; the board→belief extraction is `incoming_damage_encoder.py`
behind the single **`encode_block(live, hp_tracker)`** entry — it reads the current board **only through
the `LiveView` read-model** (no raw poke-env battle; `LivePokemon` carries the EV-computed `stats` +
integer `current_hp`/`max_hp` the belief needs), so the SAME `LiveView` built per decision feeds both the
obs path here AND the reward-shaping path (`reward_manager.py` PBRS) — one strict-API source, no
duplicate raw reads. Its only data reads are the per-species usage candidates + HP typing + offensive-stat
distributions, `lru_cache`d so only the per-defender damage math (and the rare revealed-HP expansion) is
per-decision. `reactive.py` passes `live` + the HP tracker; the reward PBRS passes `live` only (HP typing
falls back to the Smogon prior). Priors: `gen3_{move,spread,item,hidden_power}_priors.json` via
`gen3_data.priors`. Belief-not-calc → validated by calibration; the obs golden fixture pins the vector
byte-for-byte (the LiveView migration is value-neutral — golden parity holds against the v2 fixture).

> **Downstream reader:** the prober engine (`src/main/prober/engine.py`) reads
> `OFFSET_REACTIVE + move_multiplier` (active-move type mults), `+ our_matchups`,
> `+ their_matchups` (the incoming-threat decode + saliency block), and the
> turn-history span — resolved at runtime from `get_layout()`. If you move these
> offsets, its pinned regression test (`prober/engine_test.py::
> test_offsets_resolve_matches_layout`) will fail; update the pinned values there.

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
