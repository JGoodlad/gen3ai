//! State — the in-battle STATE the engine mutates, and its construction from a
//! `>start` + two packed teams.
//!
//! **Scope of THIS layer (read first).** [`BattleState::start`] builds *only* the
//! construction-time state: the exact per-mon fields Showdown's `Pokemon`
//! constructor + `setSpecies` set, BEFORE any switch-in event fires. It does NOT
//! run the engine's start action (`switchIn`), so it does NOT produce any
//! event-driven state. That split is load-bearing and verified against the real
//! sim (see `harness/gen_state_golden.js` + `tests/state_test.rs`).
//!
//! The deferred event half — the `>start` switch-in abilities (Intimidate boost,
//! Sand Stream weather, ability `Start`) — is built in [`crate::event`] and run by
//! [`BattleState::run_start_switchins`]; [`BattleState::start_with_switchins`]
//! composes both (validated by `tests/switchin_test.rs`). The table below is the
//! `start`-only (construction) view:
//!
//! | Field | Set at | Built here? |
//! |---|---|---|
//! | `stats` / `maxhp` / `hp` (= maxhp) | `Pokemon` ctor → `setSpecies` → `spreadModify` | **yes** |
//! | `species` / `types` / `level` | ctor + `setSpecies` | **yes** |
//! | `status` (empty), `boosts` (all 0), `fainted` (false) | ctor / `clearVolatile` | **yes** (defaults) |
//! | `field.weather` (none) | `Field` ctor (set later by Sand Stream etc.) | **yes** (none) |
//! | `side.active[0] = lead` | the start ACTION's `switchIn` (`battle.ts:2718`) | **structural only** — we record the lead INDEX (slot 0 in gen3 singles), we do NOT run switchIn |
//! | `boosts` changes (Intimidate), `field.weather` (Sand Stream), ability `Start` | switch-in EVENTS | **NO in `start`** — built in [`crate::event`], run by `start_with_switchins` |
//!
//! So a harness comparing this state to a *started* Showdown battle must compare
//! the CONSTRUCTION-time fields only, and against the started battle's BENCH mons
//! (team index ≥ 1) which retain pristine construction state, or against the
//! seed-independent stats/species/level which are identical on the lead too. We
//! never assert post-event `boosts` / `weather` here.
//!
//! **Lead selection.** Gen 3 OU singles has no team preview, so the lead is
//! deterministically `side.pokemon[0]` (Showdown's start action calls
//! `switchIn(side.pokemon[0], 0)`; `active.length == 1` in singles). We record
//! `SideState::active = 0`.

use crate::dex::Dex;
use crate::prng::{normalize_seed, Prng, PrngSeed};
use crate::stats::compute_stats;
use crate::team::{unpack, PokemonSet};

use crate::battle::{BattleOptions, PackedTeam};

/// A gen-3 major status condition. Empty status is `Option::None` at the use
/// site. Sleep/Toxic carry their counter (turns asleep / toxic stage); both are
/// 0 at construction since status is empty then, but the variants live here so
/// the event engine can set them without a state-shape change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Burn,
    Paralysis,
    /// Asleep; `u8` is the remaining/elapsed sleep counter (engine-owned).
    Sleep(u8),
    Freeze,
    Poison,
    /// Badly poisoned; `u8` is the toxic stage counter (engine-owned).
    Toxic(u8),
}

/// Gen-3 field weather. `None` = clear (the construction-time value; Sand Stream
/// / Drizzle / Drought / the Hail move set it via a switch-in or move EVENT, not
/// built here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weather {
    Sand,
    Rain,
    Sun,
    Hail,
}

impl Weather {
    /// Parse the RESOLVED Showdown weather id (as it appears in `field.weather` /
    /// `effectiveWeather()`) into the port's [`Weather`]. Used by the data-driven
    /// WEATHER_SPEED ability parse (`gen3_ability_batch1_v1` — `sunnyday`→Sun,
    /// `raindance`→Rain). Returns `None` for an unknown id (a hard error at the call site).
    pub fn from_id(id: &str) -> Option<Weather> {
        match id {
            "sandstorm" => Some(Weather::Sand),
            "raindance" | "primordialsea" => Some(Weather::Rain),
            "sunnyday" | "desolateland" => Some(Weather::Sun),
            "hail" | "snowscape" => Some(Weather::Hail),
            _ => None,
        }
    }
}

/// Boost array order: `[atk, def, spa, spd, spe, accuracy, evasion]` — matching
/// Showdown's `boosts` object (`pokemon.ts`). All 0 at construction.
pub const BOOST_LEN: usize = 7;

/// Per-mon in-battle state.
///
/// At construction every field is its pristine default: `hp == maxhp`,
/// `status == None`, `boosts` all 0, `fainted == false`. The decoded
/// [`PokemonSet`] is kept verbatim (it carries species/level/moves/item/ability —
/// the engine reads it for everything the resolved facts below don't cover).
#[derive(Debug, Clone)]
pub struct MonState {
    /// The decoded set (species id, level, moves, item, ability, nature, EVs/IVs).
    pub set: PokemonSet,
    /// Resolved species id (normalized, e.g. `tyranitar`). Mirrors
    /// `pokemon.species.id` — NOT `pokemon.speciesid` (which is undefined).
    /// **LIVE**: a Forecast forme change (`gen3_forecast_v1`) and a Transform copy
    /// (`gen3_transform_v1`) both move this off the construction species; the
    /// CONSTRUCTION identity is [`MonState::base_species_id`].
    pub species_id: String,
    /// `pokemon.baseSpecies.id` (`gen3_forecast_v1`, ROUND 35) — the species the mon was
    /// CONSTRUCTED with, fixed for the battle. In the sim `baseSpecies` is only rewritten by
    /// a `formeChange(…, isPermanent = true)` (Mega / Primal / a Status forme — none of which
    /// exist in gen 3), so for us it is a construction-time constant.
    ///
    /// Two things read it rather than [`MonState::species_id`]:
    ///
    /// 1. **`clearVolatile`'s `setSpecies(this.baseSpecies)`** (pokemon.ts:1560) — the
    ///    switch-out / faint revert. A Rainy Castform that pivots out is `castform` on the
    ///    bench (probe `probe_r35_forecast_reporting.js` S2/S3; a fainted one likewise —
    ///    `probe_r35_forecast_edges.js` E8 water-t2), so its next `|switch|`/`|drag|`
    ///    details are the BASE species and its `onStart` re-formes from scratch.
    /// 2. **Every REPORTING surface** — the protocol IDENT (`p1a: Castform`), the
    ///    `|switch|`/`|drag|` details, and the per-side `|request|` roster `details`. A
    ///    non-permanent `formeChange` deliberately does NOT refresh `Pokemon.details` /
    ///    `Pokemon.name`, so a FORMED Castform still reports `Castform, F` everywhere
    ///    (probe S1/S2/S3 — the `|-formechange|` line is the ONLY place the forme name
    ///    appears on the wire).
    ///
    /// Distinct from [`TransformOverlay::base_species_id`], which is the same value but
    /// only exists WHILE a Transform is up. NOTE this is deliberately NOT derived from the
    /// dex's `baseSpecies` field: gen 3 also ships the three **Deoxys** formes, which are
    /// ORDINARY selectable species (`baseSpecies: "Deoxys"`, no `battleOnly`) whose
    /// `details` must stay `Deoxys-Attack` — a dex-derived base would corrupt them.
    pub base_species_id: String,
    /// Level (mirrors `pokemon.level`).
    pub level: u8,
    /// In-battle stats `[hp, atk, def, spa, spd, spe]` (= [`crate::stats`] order).
    /// `stats[0]` is the HP stat (== `maxhp`); `stats[1..6]` are the five
    /// non-HP stored stats. These are the sim's OWN `spreadModify` outputs.
    pub stats: [u16; 6],
    /// Current HP. Starts == `maxhp`.
    pub hp: u16,
    /// Max HP (== `stats[0]`, kept denormalized for convenience).
    pub maxhp: u16,
    /// The IV-derived Hidden Power BASE POWER (`gen3_iv_derived_hidden_power_bp_v1`),
    /// precomputed once from `set.ivs` via [`hidden_power_bp`] (range 30..=70). The
    /// damage path (`run_move`) and the bridge request-JSON move name read THIS for a
    /// `hiddenpower*` move instead of the data's hard-coded 70 — so a real gen3ou HP mon
    /// whose IVs give BP != 70 (e.g. a -1 Speed IV → BP 68) damages at its IV-true BP.
    /// The TYPE stays from the typed move id (untouched). Obs-neutral (a Rust-bridge
    /// correctness value; the Python obs/DamageOperator keep their own BP-70 assumption).
    pub hidden_power_bp: u8,
    /// Major status, or `None` (empty at construction).
    pub status: Option<Status>,
    /// Stat-stage boosts `[atk, def, spa, spd, spe, accuracy, evasion]`, all 0
    /// at construction.
    pub boosts: [i8; BOOST_LEN],
    /// Whether this mon has fainted (false at construction).
    pub fainted: bool,
    /// The CONFUSION volatile's remaining-turn counter, or `None` (not confused).
    /// `Some(n)` means `n` turns of the confusion volatile remain; set when the
    /// volatile is added (`random(2,6)` at `onStart`), decremented draw-free at the
    /// start of each move, removed (→ `None`) when it reaches 0. Coexists with a
    /// major `status` (confusion is a volatile, not a major status). 0/`None` at
    /// construction.
    pub confusion: Option<u8>,
    /// The FLINCH volatile flag (`duration: 1`). Set by a flinch SECONDARY when the
    /// attacker moves first; checked at the flinchee's `onBeforeMove` (if set, the
    /// move is cancelled — draw-free); RESET to false at the end of every turn
    /// (`duration: 1`). False at construction.
    pub flinch: bool,
    /// The PROTECT/DETECT this-turn volatile (`protect` condition, `duration: 1`):
    /// `true` while a Protect/Detect cast THIS turn is up. A foe move TARGETING this
    /// mon is BLOCKED while it is `true` (after the foe draws its accuracy roll —
    /// see `turn.rs::run_move`). Set when a Protect/Detect SUCCEEDS (resolves at the
    /// move's priority 3, BEFORE the foe's attack), CLEARED to false at the TOP of
    /// each turn (after the stall-counter reset reads it — see `clear_flinch`) and on
    /// switch-out (`execute_switch`). False at construction.
    pub protected: bool,
    /// The gen-3 PROTECT/DETECT **stall counter** (`stall` volatile's
    /// `effectState.counter`, the gen3-resolved gen5-base condition with the gen4
    /// `counterMax: 8` override). The success probability of a Protect/Detect is
    /// `1/protect_counter` (`onStallMove` → `randomChance(1, counter)`); `0` means NO
    /// stall volatile is present (the next/first Protect SHORT-CIRCUITS with NO draw
    /// and ALWAYS succeeds). On a SUCCESSFUL protect the volatile is (re)added: from
    /// `0` it `onStart`s to **2**; otherwise it `onRestart`s `*= 2` capped at the
    /// gen3 **counterMax 8** (so the sequence is `0→2→4→8→8→…` = success
    /// 100%/50%/25%/12.5%/12.5%, the floored 1/8). A FAILED stall draw does **NOT**
    /// delete the volatile (the gen3 resolved gen5-base `stall.onStallMove` — unlike the
    /// gen8+ base condition — has NO `delete pokemon.volatiles['stall']`): the counter +
    /// duration PERSIST UNCHANGED (so consecutive fails re-roll at the SAME denominator,
    /// and a `stall` residual handler still fires that turn). The volatile has
    /// `duration: 2` (tracked by [`MonState::stall_duration`], refreshed to 2 only on a
    /// SUCCESS via `onRestart`): it EXPIRES → `0` at the RESIDUAL when the duration runs
    /// out — one turn after the user stops SUCCESSFULLY protecting (a different move, or a
    /// failed protect that doesn't refresh) — and on switch-out (clearVolatile). 0 at
    /// construction. This is the ONLY view of the stall counter (public both sides; the
    /// gen3 protect-odds the obs reads).
    pub protect_counter: u8,
    /// The remaining `duration` of the `stall` volatile (the gen3 `stall` condition's
    /// `duration: 2`). Tracks the volatile's lifetime so [`MonState::protect_counter`]
    /// EXPIRES at the right RESIDUAL: `fieldEvent('Residual')` decrements every
    /// duration-bearing volatile and `end`s it at 0. Set to **2** on a SUCCESSFUL protect
    /// (`onStart`/`onRestart` both reset `duration = 2`); decremented by 1 at each RESIDUAL
    /// while the volatile is up; on reaching 0 the volatile is removed → `protect_counter`
    /// resets to 0 (the expiry that makes a Protect after ONE non-protect turn a fresh
    /// first-protect, VERIFIED vs the sim: P,P,SteelWing shows `stall: none` at the
    /// SteelWing boundary). A FAILED stall draw does NOT delete the volatile (gen-3's
    /// resolved gen5-base `onStallMove` has no delete-on-fail, UNLIKE gen8+): the counter
    /// + duration persist, so consecutive fails re-roll at the same denominator. 0 at
    /// construction. Cleared on
    /// switch-out. (The shorter `protect` volatile — `duration: 1` — needs no counter: it
    /// always expires at the next turn-top, modeled by clearing `protected`.)
    pub stall_duration: u8,
    /// The **LEECH SEED** volatile (`leechseed`, planted by the Leech Seed MOVE):
    /// `Some(seeder_side)` when this mon is seeded, recording the SIDE that owns the
    /// drain (gen-3 singles: the seeder's slot is always the active `a` slot, so the
    /// heal goes to whatever is CURRENTLY active on that side — Showdown's
    /// `getAtSlot(sourceSlot)`; we store only the side since the active slot index is
    /// always `side.active`). `None` = not seeded (the construction-time value).
    ///
    /// At each end-of-turn RESIDUAL (order 10, subOrder 5 — gen4-inherited; BETWEEN
    /// Leftovers sub 4 and the status DoT sub 6) the seeded mon loses `floor(maxhp/8)`
    /// (clamped to its HP) and the seeder's CURRENT active mon HEALS that drained amount
    /// (clamped to its maxhp; SKIPPED entirely — no drain, no heal — if the seeder's
    /// active has fainted, mirroring `onResidual`'s `if (!target || target.fainted ||
    /// target.hp <= 0) return`). DRAW-FREE (deterministic `damage`/`heal`).
    ///
    /// CLEARED on switch-out (`clearVolatile`) — a seeded mon that switches out is no
    /// longer seeded; a fresh Leech Seed is needed. A Grass-type target is IMMUNE (never
    /// set); an already-seeded target's re-seed FAILS (the field is unchanged). `None`
    /// at construction.
    pub leech_seed: Option<usize>,
    /// The **SUBSTITUTE** volatile (`substitute`, created by the Substitute MOVE):
    /// `Some(hp)` records the SUBSTITUTE's remaining HP — a decoy with `hp =
    /// floor(maxhp/4)` (the cost the user paid to make it) that ABSORBS incoming foe
    /// hits until it breaks. `None` = no substitute (the construction-time value).
    ///
    /// The gen-3 Substitute model, VERIFIED bit-for-bit vs the omniscient sim's PRNG
    /// probe (`harness/probe_substitute_*.js`):
    ///   - CREATE: the move is never-miss; it FAILS (draw-free) if `hp <= floor(maxhp/4)`
    ///     (can't afford) OR a substitute is already present; on success it pays
    ///     `floor(maxhp/4)` HP and sets `substitute = Some(floor(maxhp/4))` — DRAW-FREE.
    ///   - ABSORB: a DAMAGING foe move that TARGETS this mon hits the SUBSTITUTE's HP
    ///     (not the mon). The sub BREAKS (→ `None`) when its HP reaches 0; the excess does
    ///     NOT carry to the mon (gen-3). The foe's acc/crit/damage draws are UNCHANGED, AND
    ///     the per-move SECONDARY `random(100)` is STILL DRAWN (gen-3 quirk — the same draw
    ///     count as a bare hit) — but the secondary EFFECT is NOT applied (no status / no
    ///     stat-drop / no flinch, and crucially NO confusion `random(2,6)` follow-on draw).
    ///   - STATUS / stat-DROP moves are BLOCKED by the sub (accuracy still drawn; `-fail` /
    ///     no effect), DRAW-FREE past accuracy.
    ///   - A CONFUSION self-hit hits the MON (NOT the sub) — `this.damage` bypasses the
    ///     `onTryPrimaryHit` sub-intercept; the draw model is unchanged.
    ///   - PHAZE (Roar / Whirlwind) BYPASSES the sub (the user is dragged anyway).
    ///   - CLEARED on switch-out (`clearVolatile`) and on faint (`clearVolatile`); `None` at
    ///     construction.
    pub substitute: Option<u16>,
    /// The `substitutebroken` volatile (`gen3_substitute_broken_volatile_v1`): set the moment
    /// a Substitute is broken by damage (gen4 `moves.ts:1304`, inherited by gen3 —
    /// `removeVolatile('substitute'); addVolatile('substitutebroken')`) and cleared only by
    /// `clearVolatile` (switch-out / faint), since the condition carries NO duration
    /// (gen4 `conditions.ts:99` is `{noCopy: true}`) and gen3 does not run gen4's
    /// foe-switch-in removal (`scripts.ts:47` is gated on `gen === 4`).
    ///
    /// MECHANICALLY INERT IN GEN 3 — its only readers are gen-4 abilities' U-turn
    /// interaction. It exists so `search::volatile_names` can report the name node reports;
    /// `pre_state.volatiles` was the ONLY parity field that saw it, and its absence was the
    /// single divergence class the replay harness found on a fresh golden. `noCopy: true`, so
    /// it is deliberately absent from the Baton-Pass pass-set tuple.
    pub substitute_broken: bool,
    /// Per-move CURRENT PP (`gen3_pp_tracking_v1`), parallel to `set.moves` (index k
    /// is the PP of move slot k). Initialized to each move's in-battle MAX PP
    /// ([`crate::dex::MoveData::max_pp`] = `pp * 8 / 5` with the ctor's default 3
    /// PP-ups, or the raw `pp` for a `no_pp_boosts` move) — VERIFIED vs the sim's
    /// `side.active[0].moveSlots[k].pp/.maxpp`. Decremented by 1 per USE (2 into a
    /// Pressure holder) — DRAW-FREE, and ONLY when the mon actually MOVES (a
    /// full-para / sleep / flinch / frozen / confusion-self-hit turn deducts NOTHING,
    /// mirroring `deductPP` running AFTER `runEvent('BeforeMove')` passes). PP does
    /// NOT reset on switch-out in gen3 (VERIFIED — it PERSISTS across a switch-out/in
    /// of the same mon). When ALL slots hit 0 the mon is FORCED to Struggle. `Struggle`
    /// itself is a synthetic move (not a slot), so it is NOT tracked here.
    pub move_pp: Vec<u16>,
    /// The in-battle MAX PP per move slot, parallel to `set.moves` (constant for the
    /// battle). Kept so a re-cache / a request can report `pp/maxpp`; the port's forced-
    /// Struggle gate only reads `move_pp`, but this pins the init amount for tests.
    pub move_maxpp: Vec<u16>,
    /// The **TAUNT** volatile (`taunt`, `gen3_taunt_disable_v1`): `Some(turns)` while this mon
    /// is taunted — it CANNOT SELECT any Status-category move (`taunt.onDisableMove` disables
    /// every Status move at request time). `None` = not taunted (the construction-time value).
    ///
    /// The gen-3 model, VERIFIED bit-for-bit vs the omniscient sim
    /// (`harness/probe_taunt_disable_rng.js`): the Taunt MOVE (Dark, Status, accuracy 100 →
    /// DRAWS `randomChance(100,100)`; NOT never-miss) applies the volatile with a FIXED
    /// `duration: 2` (NO duration draw — no `durationCallback`). `Some(turns)` is that remaining
    /// duration; it is decremented by 1 at each end-of-turn RESIDUAL (a duration-only handler at
    /// onResidualOrder 10 / onResidualSubOrder 15) and CLEARED (→ `None`) when it reaches 0 —
    /// so a mon taunted while the caster moves first is taunted for effectively its own next
    /// move only. While taunted, [`MonState::move_usable`] returns false for every Status-category
    /// slot (so [`MonState::must_struggle`] can force Struggle when only Status moves remain), and
    /// [`crate::turn::BattleState::move_decision_is_legal`] rejects selecting a taunted Status move.
    /// Protect BLOCKS the Taunt move (`protect: 1`); Substitute does NOT (`bypasssub: 1`). CLEARED
    /// on switch-out (`clearVolatile`). `None` at construction.
    pub taunt: Option<u8>,
    /// The **DISABLE** volatile (`disable`, `gen3_taunt_disable_v1`): `Some((slot, turns))` while
    /// this mon has ONE move slot disabled — it CANNOT SELECT that slot (`disable.onDisableMove`
    /// disables the recorded move at request time), but its OTHER slots stay usable. `None` = not
    /// disabled (the construction-time value).
    ///
    /// The gen-3 model, VERIFIED bit-for-bit vs the omniscient sim
    /// (`harness/probe_taunt_disable_rng.js`): the Disable MOVE (Normal, Status, accuracy **55**
    /// → DRAWS `randomChance(55,100)`; NOT never-miss, and it CAN miss) targets the foe's
    /// LAST-USED move ([`MonState::last_move`]). Its `onTryHit` FAILS **draw-free** (before the
    /// duration draw) if the target has NO lastMove OR its lastMove is Struggle. On a landed hit
    /// into a mon with a real lastMove, `addVolatile` draws ONE `random(2,6)` (∈ {2,3,4,5}) for
    /// the duration, `+1` iff the target has ALREADY moved this turn (`!willMove` — the disabler
    /// is SLOWER / moves 2nd), and disables the lastMove's slot. `Some((slot, turns))` records the
    /// disabled slot + that remaining duration; the duration is decremented by 1 at each residual
    /// (a duration-only handler at order NO_ORDER / subOrder 2 — the same tie-group as
    /// protect/stall/flinch) and CLEARED (→ `None`) at 0, freeing the move. While disabled,
    /// [`MonState::move_usable`] returns false for that ONE slot. Protect BLOCKS the Disable move
    /// (`protect: 1`); Substitute does NOT (`bypasssub: 1`). CLEARED on switch-out. `None` at
    /// construction.
    pub disable: Option<(usize, u8)>,
    /// The mon's LAST-USED move SLOT index (`pokemon.lastMove`, `gen3_taunt_disable_v1`):
    /// `Some(k)` = slot `k` was the last move this mon actually USED (set in `run_move` right
    /// after it moves — a full-para / asleep / flinched / frozen / confusion-self-hit turn does
    /// NOT set it, mirroring `deductPP`'s BeforeMove-passed timing). Disable reads this to decide
    /// WHICH move to disable. `None` at construction (no move used yet) and CLEARED on switch-out
    /// (a switched-in mon has no lastMove — Disable into it fails draw-free). Struggle does NOT
    /// set it to a slot (Disable into a Struggler fails), so it stays as it was. (We store the
    /// SLOT rather than the move id since Disable disables the slot; PP/legality are per-slot.)
    pub last_move: Option<usize>,
    /// `gen3_mimic_disable_self_overwrite_v1` — TRUE iff the LAST move this mon used was a
    /// **Mimic that SUCCESSFULLY overwrote its own slot** (so the slot now holds the COPIED
    /// move, not `mimic`). Showdown's Disable stores the used move's **ID** (`lastMove.id`)
    /// and its `onStart` looks for a moveSlot with that id: for a Mimic-overwrite the used id
    /// (`"mimic"`) is no longer in any slot → `onStart` returns false → **Disable FAILS**
    /// (draw-consumed: the `random(2,6)` still drew). The port stores `last_move` as a SLOT
    /// index, so a naive read would disable whatever move now occupies the Mimic slot (the
    /// copied move) — the ab_6_16 divergence. This flag lets the Disable arm reproduce the
    /// sim's `!hasMove` fail. Set FALSE by every move at `run_move` (before the body), TRUE by
    /// the Mimic success block AFTER the overlay; cleared on switch-out (like `last_move`).
    pub last_move_was_self_overwrite: bool,
    /// The **CHOICE-LOCK** slot (`gen3_pp_tracking_v1`, the `choicelock` volatile):
    /// `Some(k)` when this mon holds a Choice item (gen-3: only **Choice Band**) and has
    /// already used move slot `k` — so it is LOCKED to slot `k` and every OTHER slot is
    /// disabled (`choicelock.onDisableMove`). `None` = not choice-locked (no Choice item,
    /// or it hasn't moved yet). Set on the FIRST move a Choice-item mon uses (Showdown's
    /// `choiceband.onModifyMove` → `addVolatile('choicelock')` recording `activeMove.id`);
    /// CLEARED on switch-out (`clearVolatile`). This is what forces Struggle when the LOCKED
    /// move runs out of PP even though other slots still have PP (the CB-Tyranitar that
    /// exhausts Crunch → Struggle). Struggle itself does NOT set the lock (it is not a slot).
    pub choice_locked_move: Option<usize>,
    /// The TURN on which the `perish` volatile was inserted, and the turn on which this mon
    /// last gained a NO_ORDER **duration** volatile (`stall`, from a successful Protect/Detect).
    /// `gen3_perish_volatile_insertion_turn_v1`: Showdown gathers a mon's residual handlers in
    /// `pokemon.volatiles` INSERTION order, so the relative order of `perish` and `stall`
    /// depends on which was added first — and that order is load-bearing, because `speed_sort`
    /// is a NON-STABLE selection sort whose swaps hand the tie-group shuffle whatever pre-sort
    /// order it finds. 0 = absent. See `turn/residuals.rs`'s perish gather.
    pub perish_turn: u32,
    pub noorder_vol_turn: u32,
    /// The **FLASH FIRE** activation volatile (`flashfire`, planted by ABSORBING a Fire-type
    /// move): `true` once this mon's Flash Fire ability has absorbed a Fire move — thereafter
    /// its OWN Fire-type moves deal **×1.5** damage. `false` = not activated (the
    /// construction-time value; the ability is inert until a Fire move actually lands on it).
    ///
    /// The gen-3 model, VERIFIED bit-for-bit vs the omniscient sim
    /// (`harness/probe_flashfire_rng.js` — the resolved `Dex.mod('gen3')` flashfire ability):
    ///   - ACTIVATE: the ability's `onTryHit` fires AFTER the accuracy roll (a MISSED Fire
    ///     move does NOT activate it — probe A2), only for a `Fire`-type move and
    ///     `target != source`. It is DRAW-FREE (activation consumes no PRNG — probe A4). The
    ///     holder takes 0 damage (the existing type-absorb immunity short-circuit). A
    ///     Fire-type STATUS move (Will-O-Wisp) DOES activate it on a non-Fire, status-less,
    ///     sub-less target, but every gen-3 FF holder IS Fire-type so WoW's `hasType('Fire')`
    ///     special-case skips activation there (probe A3); a `frz`-status holder does not
    ///     activate (the `status === 'frz'` guard), but any OTHER status is irrelevant
    ///     (probe A6). Set at the Fire-absorb site in `run_move` (the same `acc_hit`-gated
    ///     immune short-circuit that heals Water/Volt Absorb).
    ///   - THE BOOST: the flashfire volatile's `onModifyDamagePhase1` returns
    ///     `chainModify(1.5)` for a Fire-type move — a DAMAGE-PHASE fold (the SAME phase as
    ///     Reflect/Light Screen), NOT an `onModifyAtk`/`onModifySpA` stat mod (those handlers
    ///     are `undefined` in the resolved gen-3 dist — probe B1). It therefore applies to
    ///     BOTH physical AND special Fire moves (category-agnostic — probe B4) and, unlike
    ///     screens, is NOT crit-bypassed (no crit guard on the handler). Folded in
    ///     `damage.rs::modify_damage` at ModifyDamagePhase1, accumulated into ONE chain
    ///     modifier with any screen so the combined result is bit-exact. DRAW-FREE.
    ///   - CLEARED on switch-out (`clearVolatile`) and on faint (`clearVolatile`); `false` at
    ///     construction. PERSISTS across turns while active. A confusion self-hit uses a
    ///     TYPELESS move (not Fire) so it is never FF-boosted.
    pub flash_fire: bool,
    /// The CURRENT held item (`pokemon.item`, `gen3_berry_trace_shedskin_v1`) — starts as
    /// `set.item` and becomes `""` (NONE) when a berry is EATEN (permanently for the
    /// battle: no item mods, no second eat, and it does NOT come back on switch-out —
    /// unlike `ability`, an item change survives a switch). Every engine item read goes
    /// through THIS field, never `set.item` (the construction-time value).
    pub item: String,
    /// Whether this mon's item was **KNOCKED OFF** (`pokemon.itemKnockedOff`,
    /// `gen3_move_coverage_batch1_v1`). gen3 Knock Off makes the item "unusable — cannot obtain
    /// a new item"; the sim's `takeItem` (pokemon.ts:1853) returns FALSE in gen≤4 when EITHER
    /// the target OR the source has `itemKnockedOff` — so a Thief / Covet whose ATTACKER was
    /// Knocked-Off (or whose TARGET was) does NOTHING to items (no removal, no gain). Set by
    /// `apply_item_removal` on a Knock Off; PERSISTS for the battle (an item change survives a
    /// switch — like `item`). `false` at construction.
    pub item_knocked_off: bool,
    /// The CURRENT ability (`pokemon.ability`, `gen3_berry_trace_shedskin_v1`) — starts
    /// as `set.ability`; TRACE overwrites it with the foe's current ability at switch-in
    /// (the copy is LIVE for every ability read); RESET to `set.ability` on switch-out
    /// (Showdown's clearVolatile `ability = baseAbility`), so a re-entering Trace mon
    /// re-traces. Every engine ability read goes through THIS field, never `set.ability`.
    pub ability: String,
    /// The **FOCUS ENERGY** volatile (`focusenergy` — in gen3 reachable ONLY via a Lansat
    /// Berry eat; the Focus Energy MOVE is unmodeled): `true` ⇒ +2 crit stages on the
    /// holder's damaging moves (`onModifyCritRatio critRatio + 2`, the resolved gen3
    /// condition — the crit roll's denominator table index shifts). DRAW-FREE to add;
    /// CLEARED on switch-out (`clearVolatile`). `false` at construction.
    pub focus_energy: bool,
    /// 0-based team-order slot (== this mon's index in `SideState::pokemon`).
    /// Mirrors `pokemon.position`, which CHANGES on a switch (switchIn swaps the
    /// array entries + their `position`), so this is NOT a stable identity.
    pub position: usize,
    /// A STABLE per-mon identity: the construction-time team index, immutable for
    /// the mon's lifetime (unlike `position`, which a switch swaps). The full-battle
    /// driver keys queued move actions on this so a mon's action follows it across
    /// the `switchIn` array swap (mirroring Showdown's `action.pokemon` object ref).
    pub uid: usize,
    /// The CACHED action-speed (`pokemon.speed`, `pokemon.js:242/284`) — the value the
    /// `eachEvent` speed-tie shuffles AND the residual handler-sort read
    /// (`battle.js:296`/`767`), which is NOT recomputed live on every read. It is
    /// (re)established to the CURRENT para/boost-aware action speed (`getActionSpeed`)
    /// at three sites: `commitChoices` (turn start, `battle.js:2494`), the start of the
    /// `residual` action (`battle.js:2342`), and SWITCH-IN (the entrant's speed is set
    /// to its current para/boost-aware value — VERIFIED vs the sim: a Jirachi that
    /// switches in PARALYZED ties the post-switch shuffles on its PARA speed 53, not its
    /// raw 212; an unparalyzed entrant is raw == live). BETWEEN those sites it goes
    /// STALE — so a mon paralyzed WHILE already active keeps its turn-start speed for
    /// the rest of the move phase and only drops at the residual's refresh (the e2e-
    /// capstone seed-desync this models). Set to `stats[5]` (= live, no status) at
    /// construction.
    pub cached_speed: u32,
    /// The **TRUANT** loaf flag (`pokemon.truantTurn`, `gen3_ability_batch4_v1`): `true` ⇒ the
    /// holder LOAFS its next move attempt (`truant.onBeforeMove`, priority **9** — AFTER
    /// sleep/freeze (10), BEFORE flinch (8): `|cant|<mon>|ability: Truant`, DRAW-FREE, no PP).
    /// ARMED by `truant.onSwitchIn` (`truantTurn = this.turn !== 0` — a LEAD arms `false` and
    /// moves turn 1; any mid-battle entrant arms `true`) and TOGGLED by the order-**27**
    /// residual (`truantTurn = !truantTurn`) — so a mid-turn entrant (voluntary pivot, drag, a
    /// mid-ACTION-faint replacement) is toggled back the same turn and MOVES its first full
    /// turn, while a POST-residual entrant (a residual-DoT-KO replacement) keeps `true` and
    /// LOAFS its first turn. PROBE-settled (`harness/probe_truant_rng.js` +
    /// `probe_truant_edges_rng.js`): the loaf turn draws NOTHING for the loafer; an asleep
    /// holder's sleep counter still decrements (slp cants first at priority 10); a paralyzed
    /// holder's loaf turn draws NO para roll; a speed-tied Truant mirror adds ONE residual
    /// tie-shuffle draw. `false` at construction (== the lead arming).
    pub truant_turn: bool,
    /// The mon's GENDER (`pokemon.gender`, `gen3_ability_batch4_v1`): `Some('M')`/`Some('F')`,
    /// `Some('N')` for an explicit genderless, or `None` when the packed set omitted it (the
    /// sim then falls back to `species.gender` or DRAWS `battle.sample(['M','F'])` at
    /// construction — an init draw the port does not model, so an attract gender-compare on a
    /// `None` gender PANICS fail-loud; every golden/e2e team that can reach attract pins its
    /// genders explicitly). Read ONLY by the attract volatile's onStart gender gate.
    pub gender: Option<char>,
    /// The **ATTRACT** volatile (`attract`, `gen3_ability_batch4_v1` — in gen3 reachable ONLY
    /// via a Cute Charm contact proc; the Attract MOVE is unmodeled): `Some((source_side,
    /// source_uid))` while this mon is infatuated with that SOURCE mon. onBeforeMove priority
    /// **2** (confusion 3 > attract 2 > par 1): emit `-activate` ALWAYS, then draw
    /// `randomChance(1,2)` — cant on a pass. NO duration; cleared when the SOURCE leaves the
    /// field (`onUpdate`) or the HOLDER switches out (`clearVolatile`). The gender gate
    /// (M↔F opposite only; genderless never) lives at the ADD (onStart, draw-free fail).
    /// Probe `probe_cutecharm_attract_rng.js`. `None` at construction.
    pub attract: Option<(usize, usize)>,
    /// The **COLOR CHANGE / type-override** current types (`pokemon.setType`,
    /// `gen3_ability_batch4_v1`): `Some(types)` OVERRIDES the species types for EVERY
    /// in-battle type read (STAB, chart effectiveness, status type-immunity, sand-chip
    /// immunity, Magnet Pull's Steel gate — the ONE `mon_types` choke point in `turn.rs`).
    /// Set by Color Change's onDamagingHit (`[move.type]` — fires even behind a SUBSTITUTE;
    /// NOT on a KO hit / a Status move / typeless `???` / an already-matching type);
    /// CLEARED on switch-out (a re-entering Kecleon is Normal again). DRAW-FREE.
    /// Probe `probe_colorchange_rng.js`. `None` at construction.
    pub types_override: Option<Vec<crate::dex::Type>>,
    /// The ID of the move this mon last EXECUTED (`pokemon.lastMoveUsed`), or `None` if it
    /// has not moved since entering (`gen3_conversion_v1`). DISTINCT from `last_move`, which
    /// is a moveslot INDEX for Disable: this is an id, survives a Mimic overwrite, and is set
    /// for a Struggle (where `last_move` is deliberately `None`). Read by CONVERSION 2, which
    /// keys off the TARGET's last-used move's TYPE. Cleared on switch-out and on faint —
    /// probe-confirmed (`harness/probe_conversion.js` C8: a switched-out mon reports NONE).
    ///
    /// Set at the same site as `last_move`, i.e. INSIDE the `!pursuit_strike && !sleep_talk_call`
    /// guard. That placement is probe-settled, not assumed: a SLEEP-TALK-called move does NOT
    /// overwrite it — after Snorlax Sleep-Talks, the sim reports `lastMoveUsed = sleeptalk`,
    /// the OUTER move, so the inner `useMove` must leave it alone.
    pub last_move_used: Option<String>,
    /// UPROAR's multi-turn lock (`gen3_uproar_v1`): `Some((remaining_turns, move_index))`
    /// while locked, `None` otherwise. The duration is drawn ONCE — a `random(2,6)` on the
    /// CAST turn, AFTER the damage — and NEVER re-drawn; each later turn re-uses the slot and
    /// spends NO PP. Ticked by the order-10/subOrder-11 residual, which emits
    /// `|-start|<u>|Uproar|[upkeep]` on a live tick and `|-end|<u>|Uproar` on the tick that
    /// reaches 0 (the duration-END `continue` branch, so — unlike outrage/thrash — there is NO
    /// end-of-lock confusion). Folds into `move_locked()`, which is what gives it the FIRM
    /// single-move `trapped:true` request shape for free. Cleared SILENTLY on switch-out and
    /// faint (a phazed-out uproarer emits no `-end`).
    pub uproar: Option<(u8, usize)>,
    /// Was this mon DAMAGED BY THE CURRENT FOE this turn? (`gen3_bp_modifier_cluster_v1`.)
    /// REVENGE's `basePowerCallback` doubles on `pokemon.attackedBy.some(p => p.source ===
    /// target && p.damage > 0 && p.thisTurn)`. Deliberately NOT reusing `reactive`: that is
    /// armed only while Counter / Mirror Coat is selected AND is category-FILTERED (Counter
    /// takes physical, Mirror Coat special), whereas Revenge doubles off a hit of EITHER
    /// category and must work with no reactive move in sight. Cleared at each turn's top
    /// alongside the other `duration: 1` volatiles.
    pub damaged_by_foe_this_turn: bool,
    /// FURY CUTTER's consecutive-use volatile (`gen3_bp_modifier_cluster_v1`):
    /// `Some((multiplier, duration))`. The resolved gen-3 callback ADDS the volatile from
    /// inside `basePowerCallback` — `onStart` sets multiplier 1, `onRestart` doubles it while
    /// `< 16` and refreshes `duration` to 2 — then reads it, so the FIRST use is bp 10 and
    /// each consecutive use doubles to a 160 cap. `duration: 2` means ONE non-Fury-Cutter turn
    /// lapses it (probe-confirmed: miss, other move, then back to bp 10).
    pub fury_cutter: Option<(u8, u8)>,
    /// The shared `lockedmove` volatile for OUTRAGE / PETAL DANCE / THRASH
    /// (`gen3_lockin_family_v1`): `Some((remaining_turns, move_index))`. All three carry the
    /// SAME condition, so one field serves them all. The duration is a `random(2,4)` (→ 2 or
    /// 3) drawn ONCE on the CAST turn, after the damage — measured, not derived: the sim's
    /// condition has two interacting counters (`duration: 2` refreshed by `onRestart`, plus a
    /// `trueDuration` ticked at the residual) whose NET observable is simply "the user uses
    /// the move `trueDuration` times". When the lock runs out at the residual the user is
    /// CONFUSED (its own `random(2,6)`) — unlike Uproar, which ends cleanly. Folds into
    /// `move_locked()`. Sleep BREAKS the lock at the residual with NO confusion.
    pub locked_move: Option<(u8, usize)>,
    /// ROLLOUT / ICE BALL's lock (`gen3_rollout_defensecurl_v1`): `Some((hits_so_far, slot))`.
    /// UNLIKE the outrage family there is NO duration draw — the lock is a fixed **5
    /// EXECUTIONS**, and the base power doubles per execution (30/60/120/240/480), doubled
    /// AGAIN while the user carries the Defense Curl volatile. Probe-measured as per-turn HP
    /// deltas rather than derived, because a MISS does not advance the counter and a CRIT
    /// mimics a rung.
    pub rollout: Option<(u8, usize)>,
    /// The DEFENSE CURL volatile (`gen3_rollout_defensecurl_v1`). Its own effect is the
    /// declarative +1 Def, but the volatile exists to double Rollout / Ice Ball — which is why
    /// the two had to be modeled in the same pass. Cleared on switch-out and faint.
    pub defense_curl: bool,
    /// RAGE's `singlemove` volatile (`gen3_rage_secretpower_v1`). While it is up, every
    /// non-Status FOE move that HITS the holder raises the holder's Atk by one stage. It is
    /// removed by the HOLDER's own next move (`onBeforeMove` at priority 100), so the window
    /// is "from casting Rage until I act again" — which is why the boost is only observable
    /// when the foe strikes on the SAME turn (or is faster on the next one). Probe-measured.
    pub rage: bool,
    /// MINIMIZE's volatile (`gen3_minimize_v1`). Its own effect is the +1 evasion (which the
    /// accuracy pipeline already folds), but the VOLATILE is live in gen 3: a move carrying
    /// `flags.minimize` deals DOUBLE damage to the holder. Note gen-3 has NO `onAccuracy`
    /// bypass — the later-gen "these moves can't miss a minimized target" rule does not
    /// exist here (measured: Stomp into +3 evasion still rolls and still misses).
    pub minimize: bool,
    /// IMPRISON (`gen3_imprison_v1`): this mon has cast Imprison, so every move it KNOWS is
    /// blocked for the FOE. There is no duration, no residual tick and no `-end` line — it
    /// simply lasts while the caster is out. The restriction is HIDDEN in the request: the
    /// foe's blocked moves still report `disabled:false`, and the request instead gains the
    /// top-level `maybeDisabled` / `maybeLocked` flags. A rejected pick then re-requests with
    /// that slot flipped to `disabled:true, disabledSource:""`.
    pub imprison: bool,
    /// `pokemon.activeTurns` — the number of turns this mon has been active (`pokemon.ts:243`;
    /// set to 0 in `switchIn` [battle-actions.ts:137], `++`'d at `endTurn` [battle.ts:1762],
    /// AFTER the residual). Read by **Speed Boost** (`gen3_ability_batch1_v1`): its
    /// `onResidual` boosts +1 spe iff `pokemon.activeTurns` (truthy → `>= 1`). So a LEAD
    /// boosts on turn 1 (activeTurns 1 at its first residual — VERIFIED vs the sim) while a
    /// mon that SWITCHES IN this turn does NOT boost on its entry turn (activeTurns 0 at that
    /// residual, incremented to 1 only at endTurn). Modeled: init **1** at construction (leads
    /// boost turn 1), RESET to 0 in `execute_switch`, INCREMENT at end-of-turn. DRAW-FREE.
    pub active_turns: u32,
    /// The **CURSE** volatile (`curse`, `gen3_move_coverage_batch3_v1` — laid by the GHOST
    /// branch of the Curse move onto the FOE): `Some(source_side)` when this mon is cursed,
    /// recording the SIDE whose mon laid the curse (for the `[of] <user>` clause on the
    /// `-start` line; the source is the caster's active). `None` = not cursed (the
    /// construction-time value).
    ///
    /// The gen-3 model, VERIFIED bit-for-bit vs the omniscient sim
    /// (`harness/probe_batch3_curse.js`): a GHOST user's Curse pays `floor(maxhp/2)` HP and
    /// lays this volatile on the FOE (the `curse.onStart` `-start`), then each end-of-turn
    /// RESIDUAL (order 10, subOrder **8** — the gen-3 `curse` condition's
    /// `onResidualOrder: 10, onResidualSubOrder: 8`, so AFTER Leftovers sub 4 / Leech sub 5 /
    /// status DoT sub 6, and after Taunt's sub 15? no — 8 < 15) the CURSED mon loses
    /// `floor(maxhp/4)` (`this.damage(baseMaxhp/4)`), emitting `|-damage|<foe>|<hp>|[from]
    /// Curse`. DRAW-FREE. A re-curse into an already-cursed foe FAILS draw-free (`[still]` +
    /// `-fail`, no HP cost); a Curse into a SUBSTITUTE does NOTHING (the onModifyMove deletes
    /// the volatileStatus+onHit → `[still]` + `-fail`, no HP cost). A GHOST target is NOT
    /// immune (the curse volatile has no type gate). CLEARED on switch-out (`execute_switch`'s
    /// clearVolatile) and on faint (`process_faints`), exactly like `leech_seed`. `None` at
    /// construction.
    pub curse: Option<usize>,

    /// The **FOCUS PUNCH** `focuspunch` volatile (`gen3_move_coverage_batch4_v1`) —
    /// `Some(lost_focus)` while the volatile is up (re-added every turn a Focus Punch is
    /// queued, via the `beforeTurnMove` order-5 queue action), `None` otherwise. VERIFIED
    /// bit-for-bit vs the omniscient sim (`harness/probe_batch4_focuspunch.js`): the
    /// `focuspunch.onStart` emits `|-singleturn|<user>|move: Focus Punch` (DRAW-FREE); the
    /// `onHit` sets `lost_focus = true` when the user is HIT by a NON-Status move (chip
    /// absorbed by the user's OWN Substitute does NOT set it — the sub's `onTryPrimaryHit`
    /// intercepts before the focuspunch `onHit`); the move's `onTry` then CANCELS the punch
    /// (draw-free, BEFORE accuracy: `|move|…Focus Punch||[still]` + `|cant|…Focus Punch|Focus
    /// Punch`) iff `lost_focus`; and the volatile's `onTryAddVolatile` BLOCKS a flinch (a
    /// Focus Punch user is flinch-immune — DRAW-RELEVANT via the residual duration-handler
    /// count). It is a `duration: 1` volatile → registers a NO_ORDER/subOrder-2 residual
    /// duration handler (like `flinch`/`protect`/`stall`), so a FP MIRROR at equal speed adds
    /// one residual tie-shuffle draw. Cleared at the TOP of each turn (`clear_focus_punch`,
    /// re-added by the beforeTurnMove) and on switch-out/faint. `None` at construction.
    pub focus_punch: Option<bool>,

    /// The **PURSUIT** `pursuit` volatile (`gen3_move_coverage_batch4_v1`) —
    /// `Some(pursuer_uid)` while the volatile is laid on THIS mon (the pursuer's TARGET), the
    /// pursuer being the foe's active whose `uid` is stored; `None` otherwise. Laid at the
    /// BeforeTurn phase (the pursuit `beforeTurnCallback` via the `beforeTurnMove` order-5
    /// queue action) recording the pursuer, SKIPPED if the pursuer is frz/slp. The INTERRUPT
    /// (`condition.onBeforeSwitchOut`): when this mon VOLUNTARILY switches out,
    /// `execute_switch` cancels the pursuer's queued Pursuit move (draw-free), deducts its
    /// Pursuit PP + sets its lastMove (draw-free), then runs Pursuit against THIS switching mon
    /// at ×2 BP + NEVER-MISS (crit + damage, no accuracy) BEFORE the switch resolves, and
    /// Choice-locks the pursuer if it holds a Choice item. It is a `duration: 1` volatile →
    /// registers a NO_ORDER/subOrder-2 residual duration handler (a normal Pursuit whose foe
    /// STAYS in leaves the volatile up through the residual). Cleared at the TOP of each turn
    /// (`clear_focus_punch`) — or consumed by the interrupt (`execute_switch`) — and on
    /// switch-out/faint. `None` at construction.
    pub pursuit: Option<usize>,
    /// The **BEAT UP** `beatup` volatile (`gen3_move_coverage_batch4b_v1`) — `true` while this
    /// mon's Beat Up is resolving THIS turn (added by the sim's `onModifyMove`
    /// `pokemon.addVolatile("beatup")` before the multi-strike loop, carrying the stat-swap
    /// hooks + a `duration: 1`). The stat swap itself is computed directly in `run_beat_up`, so
    /// this flag exists ONLY for the `duration: 1` residual DURATION handler (NO_ORDER/subOrder-2,
    /// the protect/stall/flinch/focus-punch tie group): a BEAT UP MIRROR at equal speed adds ONE
    /// residual tie-shuffle draw (the e2e_217 desync). Cleared at the TOP of each turn
    /// (`clear_flinch`) + on switch-out/faint. `false` at construction.
    pub beat_up: bool,

    /// The **MUSTRECHARGE** volatile (`gen3_move_coverage_batch4c_v1`) — `true` after a
    /// SUCCESSFUL damaging Hyper Beam hit (plain hit / sub-absorb / sub-break / target-KO —
    /// NOT a miss / immune / Protect-block; probe `harness/probe_batch4c_hyperbeam.js`).
    /// While set, the NEXT turn's request offers ONLY the pseudo-move `Recharge`
    /// (`mustrecharge.onLockMove = 'recharge'`, `trapped:true` — a switch is REJECTED) and
    /// the turn is spent as `|cant|<user>|recharge` at the user's normal speed-order
    /// position: the gen3-resolved `mustrecharge.onBeforeMove` (priority **11** — BEFORE
    /// sleep/frz(10)/truant(9)/flinch(8)/para(1), so NO status draw fires on the locked
    /// turn — a par'd user draws NO para roll, a slp'd user's counter does NOT decrement)
    /// emits the cant, removes the volatile (+ `removeVolatile('truant')` — a no-op in the
    /// port's `truant_turn` toggle model, whose order-27 residual toggle already consumes
    /// the loaf, probe-verified Slaking HB/recharge/HB cadence), and returns null → ZERO
    /// draws, NO PP. The condition has `duration: 2` → it registers a NO_ORDER/subOrder-2
    /// residual DURATION handler on the CAST turn's residual (the protect/stall/flinch tie
    /// group — an HB MIRROR at equal speed adds one tie-shuffle draw). The lock PERSISTS
    /// across the opponent's force-switch (a target-KO'd HB is still locked next turn).
    /// Cleared at the recharge cant + on switch-out/faint. `false` at construction.
    pub must_recharge: bool,

    /// The **TWOTURNMOVE** charge volatile (`gen3_move_coverage_batch4c_v1`, Solar Beam) —
    /// `Some` from the CHARGE turn's `onTryMove` (`addVolatile('twoturnmove')`, whose
    /// onStart adds the `solarbeam` sub-volatile = `charging`) until the residual duration
    /// expiry / an onMoveAborted / switch-out / faint. See [`TwoTurnMove`]. `None` at
    /// construction.
    pub two_turn: Option<TwoTurnMove>,

    /// The **COUNTER / MIRROR COAT** reactive volatile (`gen3_move_coverage_batch5_v1` —
    /// the gen-3 `counter` / `mirrorcoat` conditions, probe
    /// `harness/probe_batch5_reactive.js`): `Some` while this mon SELECTED Counter or
    /// Mirror Coat THIS turn (added by the move's `beforeTurnCallback` via the order-5
    /// `beforeTurnMove` queue action, DRAW-FREE — its `onStart` RESETS `{slot: null,
    /// damage: 0}` EVERY selection turn, so PREV-TURN damage never counts). The volatile's
    /// `onDamage` (priority −101, i.e. LAST — after Focus Band) RECORDS each qualifying
    /// FOE **Move**-effect hit that lands on the MON (a sub-absorbed hit never fires the
    /// mon's Damage event → not recorded), OVERWRITING `damage = 2 × that hit` — so a
    /// MULTI-HIT records 2× the LAST STRIKE only (probed: Beat Up 3 strikes → Mirror Coat
    /// returns 2× the last). The qualifying rule (gen3-resolved, probe-settled):
    ///   * counter: `effect.category === 'Physical' || effect.id === 'hiddenpower'`
    ///     (Seismic-Toss-class fixed damage IS Physical; Struggle IS Physical; a bare
    ///     Hidden Power is countered UNCONDITIONALLY even when special-typed);
    ///   * mirrorcoat: `effect.category === 'Special' && effect.id !== 'hiddenpower'`
    ///     (Beat Up's strikes are Special → recorded; HP never).
    /// The move's execution (`run_fixed_damage_move`) FAILS ZERO-DRAW (a bare `|move|`
    /// line, no `-fail`, PP −1) when the volatile is missing or un-armed. `duration: 1`
    /// → registers a NO_ORDER/subOrder-2 residual DURATION handler (the
    /// protect/stall/flinch/focus-punch tie group — a counter-mirror at equal speed adds
    /// ONE residual tie-shuffle draw). Cleared at the TOP of each turn (`clear_flinch`,
    /// re-added by the beforeTurnMove) + on switch-out/faint. `None` at construction.
    pub reactive: Option<Reactive>,

    /// The gen-3 `slp` condition's **`skippedTime`** (`gen3_move_coverage_batch5_v1`,
    /// the Sleep Talk composition — live-probed: time=3 → talk,talk → time=1,skipped=2 →
    /// switch out+in → time=3 again): each still-asleep turn whose move PROCEEDS via
    /// `sleepUsable` (Sleep Talk) increments it; a NORMAL blocked `|cant|slp` turn RESETS
    /// it to 0; the slp `onSwitchIn` RESTORES `time += skippedTime; skippedTime = 0` (at
    /// the runSwitch `runEvent('SwitchIn')` site, beside the tox stage reset — so a
    /// CANCELLED runSwitch keeps it, the tox-persistence law). Reset to 0 whenever a
    /// fresh sleep is set (`slp.onStart` — a fresh statusState) and meaningless while
    /// not asleep. Affects LATER wake timing (protocol-visible), never a draw. 0 at
    /// construction.
    pub sleep_skipped: u8,

    /// The **ENCORE** volatile (`gen3_move_coverage_batch6_v1`): `Some((locked_slot,
    /// remaining_turns))` while the mon is locked into its `last_move` slot. Set by a
    /// landed Encore — the stored duration is the gen3 `durationCallback` roll
    /// `random(3,7)` (3..6) `+ 1` iff the target had ALREADY moved this turn (the
    /// Disable `!willMove → duration++` precedent, probe-settled EN1/EN2). While up:
    /// every OTHER slot is un-selectable (`move_usable`) and a QUEUED different move is
    /// OVERRIDDEN at execution to the encored slot (`encore.onOverrideAction` — the
    /// ENCORED slot's PP deducts, the announce shows the encored move; EN7). The
    /// residual handler (order 10, subOrder 14 — before taunt's 15) decrements it each
    /// end-of-turn INCLUDING the landing turn's, `|-end|` at 0 — AND removes it EARLY
    /// the residual the encored slot hits 0 PP (`encore.onResidual`'s pp check, EN5).
    /// `noCopy: true` → NOT Baton-Passable. Cleared on switch-out + faint. `None` at
    /// construction.
    pub encore: Option<(usize, u8)>,

    /// The **DESTINY BOND** volatile (`gen3_move_coverage_batch6_v1`): `true` from the
    /// cast until the user's NEXT MOVE ATTEMPT (removed at `onBeforeMove` priority −1
    /// for any move != destinybond, AND at `onMoveAborted` for a cant — probe DB2; a
    /// re-cast draw-free re-adds, DB6). While up, a FOE-MOVE KO of the holder faints
    /// the killer too (the `onFaint` gate: source non-ally && effect Move &&
    /// !futuremove — a residual/confusion-self-hit KO does NOT trigger, DB4). The cast
    /// + the mutual-faint chain are DRAW-FREE. `noCopy: true` → NOT Baton-Passable.
    /// Cleared on switch-out + faint. `false` at construction.
    pub destiny_bond: bool,

    /// The pending DESTINY BOND mutual-faint record: `Some(killer_side)` when a
    /// qualifying FOE-MOVE hit zeroed this mon's HP while `destiny_bond` was up. Read
    /// by `process_faints` when this corpse is drained from the faint queue: it emits
    /// `|-activate|<corpse>|move: Destiny Bond|` then faints the killer side's ACTIVE
    /// (the killer — gen-3 singles; draw-free). `None` at construction.
    pub destiny_bond_ko_by: Option<usize>,

    /// The **ENDURE** this-turn volatile (`gen3_move_coverage_batch6_v1`, `duration:
    /// 1`): survive any MOVE damage at 1 HP (`endure.onDamage` priority −10 —
    /// `effect.effectType === 'Move' && damage >= hp → hp − 1`, `|-activate|` BEFORE
    /// the `|-damage|`; it endures FIXED damage + EVERY strike of a multihit, but NOT
    /// residual/non-Move damage — a burn chip KOs the 1-HP endurer, ED5). Shares the
    /// Protect/Detect `stall` counter machinery (`protect_counter`/`stall_duration` —
    /// the SHARED ladder, probe ED3/ED4) and registers a NO_ORDER/subOrder-2 residual
    /// duration handler (the endure+stall intra-mon tie — ONE shuffle draw on a
    /// SUCCESSFUL endure turn at ANY speed, ED1/ED2/ED9). Cleared at the turn-top
    /// (`clear_flinch`) + switch-out + faint. `false` at construction.
    pub endure: bool,

    /// The **PERISH SONG** counter volatile (`gen3_move_coverage_batch6_v1`,
    /// `perishsong` condition — `duration: 4`, `onResidualOrder: 12` = LAST in the
    /// residual ladder): `Some(remaining)` from the field-wide apply (4 at cast; the
    /// cast-turn residual decrements to 3 and prints `perish3`) down to the 1 → 0 tick,
    /// whose `onEnd` prints `perish0` + FAINTS the holder. Applied to ALL actives
    /// (caster included) in side order; Soundproof is immune. DRAW-FREE everywhere
    /// except the order-12 handler's residual sort tie (two perished mons at EQUAL
    /// cached speed = ONE `random(0,2)` per residual — P5). Cleared on switch-out
    /// (ordinary volatile clear) + faint; `noCopy` is FALSE → Baton-Passable (the
    /// resolved gen3 condition — probe `probe_batch6_dexfacts.js`). `None` at
    /// construction.
    pub perish: Option<u8>,

    /// The **MEAN LOOK / SPIDER WEB / BLOCK** trap volatile (`gen3_move_coverage_
    /// batch6_v1`): `Some(trapper_uid)` while this mon holds the linked `trapped`
    /// volatile — a FIRM trap (the Shadow-Tag request shape: `trapped:true` on the
    /// FIRST request, a rejected switch is `[Invalid choice]` with NO re-request —
    /// probe-settled). The link ENDS the moment the TRAPPER (the mon whose `uid` is
    /// stored) leaves the field ANY way — voluntary switch (T1), faint (T9), or drag —
    /// freeing the holder at its next request. gen3 `trapped.noCopy` is FALSE → a
    /// Baton-Passing HOLDER passes the volatile to its entrant (still trapped, same
    /// link — T3b); a GHOST holder IS trapped (the arenatrap no-type-immunity
    /// precedent). The volatile itself adds ZERO endTurn draws (its Condition
    /// `onTrapPokemon` handler subOrder 2 never ties an Ability handler's subOrder 7).
    /// Cleared on switch-out (drag) + faint. `None` at construction.
    pub trapped_by: Option<usize>,

    /// The **PARTIAL-TRAP** volatile (`gen3_partial_trap_v1` — Wrap / Bind / Fire Spin /
    /// Clamp / Whirlpool / Sand Tomb): `Some(state)` while this mon holds the resolved
    /// gen-3 `partiallytrapped` condition (base `data/conditions.ts` shadowed by the
    /// **gen5** `onStart`/`onResidual` override [boundDivisor 16 — gen3 has no Binding
    /// Band/Grip Claw] and the **gen4** `durationCallback` = `this.random(3, 7)` +
    /// `onResidualOrder: 10, onResidualSubOrder: 9`). Applied POST-hit by the trapping
    /// move (ONE `random(3,7)` draw — the family's only new draw), it chips
    /// `floor(maxhp/16)` DRAW-FREE at every subsequent end-of-turn and makes the holder a
    /// **FIRM** trap (`onTrapPokemon` → bare `tryTrap()` — the Shadow-Tag request shape).
    /// Cleared on switch-out + faint + Rapid Spin; `noCopy` is FALSE → the volatile IS
    /// Baton-Passed to the entrant (probe-verified, source unchanged). `None` at
    /// construction.
    pub partial_trap: Option<PartialTrap>,

    /// The **CHARGE** volatile (`gen3_move_coverage_batch6_v1`, no duration): `true`
    /// from the Charge cast until the user's NEXT move attempt OF ANY KIND
    /// (`charge.onAfterMove` + `onMoveAborted` remove it for any move != charge — an
    /// idle Splash consumes it, a cant consumes it; only an Electric next move gets the
    /// `onBasePower chainModify(2)`). gen3 Charge has NO +1 SpD (the gen4+ boost is
    /// absent — probed). Re-charge while up re-adds (`onRestart`, `-start` again).
    /// Cleared on switch-out + faint. `false` at construction.
    pub charge: bool,

    /// The **MIMIC** moveslot overlay (`gen3_move_coverage_batch6_v1`): `Some` while
    /// the Mimic slot is OVERWRITTEN with the copied move for this field stay. The
    /// copy MUTATES `set.moves[slot]` + `move_pp[slot] = min(5, copied.pp)` +
    /// `move_maxpp[slot] = copied.max_pp()` (the sim's `moveSlots[mimicIndex] = {…,
    /// pp: Math.min(5, move.pp), maxpp: calculatePP(move, 3)}`), so every consumer
    /// (move resolution, PP legality, the bridge request) sees the copied move; this
    /// record holds what to RESTORE on switch-out / faint (`baseMoveSlots` — back to
    /// Mimic with its own remaining PP). `None` at construction.
    pub mimic_overlay: Option<MimicOverlay>,

    /// The **SNATCH** singleturn volatile (`gen3_snatch_v1`): `true` while this mon's
    /// Snatch (`volatileStatus:'snatch'`, `duration:1`) is up. Set DRAW-FREE on the
    /// priority-+4 cast; while it is up, the NEXT self-targeted `flags.snatch` status
    /// move used by ANY eligible mon (in gen-3 singles, the FOE) is STOLEN — the
    /// snatcher executes it instead (the `snatch` condition's `onAnyPrepareHit`). Like
    /// `flinch`/`focus_punch`/`endure` it clears at the NEXT turn-top (`clear_flinch`),
    /// and it registers a NO_ORDER/subOrder-2 residual DURATION handler (so a SNATCH
    /// MIRROR at equal speed draws ONE residual tie-shuffle — probe-verified 8 vs the
    /// both-Splash control's 7). Cleared on switch-out + faint. `false` at construction.
    pub snatch: bool,

    /// Whether this mon's CURRENT sleep was SELF-inflicted by **Rest** (vs foe-inflicted
    /// by a sleep move / Effect Spore). Mirrors `statusState.source?.isAlly(pokemon)` for
    /// the gen-3 **Sleep Clause Mod** (`rulesets.ts` `sleepclausemod.onSetStatus`): a mon
    /// that put ITSELF to sleep via Rest does NOT count toward its side's one-foe-asleep
    /// cap, so a foe sleep move on a DIFFERENT mon still lands (probe-verified — a
    /// self-Rested Zapdos does NOT block a foe's Spore on Suicune). Set at EVERY sleep
    /// application (`true` in `run_rest`, `false` in `try_set_status_impl`), read ONLY by
    /// `side_has_sleeper` while `status == Some(Sleep(_))`. `false` at construction.
    /// `gen3_sleep_clause_self_rest_exempt_v1` (R15-P1).
    pub sleep_from_rest: bool,

    /// The foe active's `uid` at the moment THIS mon switched in (`gen3_intimidate_forced_replacement_v1`,
    /// M3). Captured in `execute_switch` right after the array swap; consumed by the DEFERRED
    /// switch-in Intimidate (`event::intimidate_on_start`) to SUPPRESS a mis-fired drop whose
    /// intended target was REPLACED (a forced replacement) between the switch action and the deferred
    /// `RunSwitch`. In Showdown the entrant's `runSwitch`/Intimidate ALWAYS resolves as part of its own
    /// switch action (before the action-tail `faintMessages` + `makeRequest('switch')`), so it targets
    /// the mon present at switch time — NEVER a later replacement (the ab_1381_0 mis-target where a
    /// Destiny-Bond-fainted Tyranitar was replaced by Snorlax, whose Atk the port wrongly dropped −1).
    /// `Some(uid)` while a switch-in Intimidate is pending, `None` otherwise (cleared once run_switch
    /// consumes it + on faint). `None` at construction. Draw-free (a boost-suppression is state-only).
    pub switchin_foe_uid: Option<usize>,

    /// The **YAWN** delayed-sleep volatile (`gen3_yawn_v1`, the gen-3 `yawn` condition —
    /// `duration: 2`, `noCopy`): `Some((remaining_duration, source_uid))` while a Yawn cast on
    /// THIS mon is pending, `None` otherwise (the construction-time value). The CRUX is that the
    /// sleep `random(2,6)` fires at RESOLVE, not at cast — so the cast is DRAW-FREE.
    ///
    /// The gen-3 model, VERIFIED bit-for-bit vs the omniscient sim
    /// (`harness/probe_batch89_haze_trick_yawn.js` + `probe_yawn_edges`):
    ///   - CAST (a category-Status foe-target move, `accuracy: true` → NO accuracy draw): the
    ///     `onTryHit` FAILS draw-free if the target ALREADY has a major status (`[still]` +
    ///     `-fail|<user>`, no volatile) OR is sleep-immune (Insomnia / Vital Spirit via
    ///     `runStatusImmunity('slp')` → `-immune|<target>|[from] ability: <A>`, no volatile); a
    ///     Protect BLOCKS it (`protect: 1` → `-activate Protect`); a Substitute BLOCKS it (no
    ///     `bypasssub` → `onTryPrimaryHit` → `[still]` + `-fail|<user>`). TryHit order: Protect >
    ///     already-statused > sleep-immune > substitute. Otherwise it adds this volatile with
    ///     `duration = 2` DRAW-FREE, emitting `|-start|<target>|move: Yawn|[of] <source>`.
    ///   - RESOLVE — the volatile is a RESIDUAL DURATION handler at `(onResidualOrder 10,
    ///     onResidualSubOrder 19)`. It decrements 2 → 1 (end of the CAST turn) then 1 → 0 (end of
    ///     the NEXT turn); on the 1 → 0 tick the `onEnd` fires: emit `|-end|<target>|move: Yawn|
    ///     [silent]` then `target.trySetStatus('slp', source)` — routed through the EXISTING
    ///     [`Self::try_set_status`] path so the sleep `random(2,6)` onStart draw, the gen3ou Sleep
    ///     Clause block, AND the gen3ou SetStatus 2-clause shuffle all come for free + bit-for-bit
    ///     (gen3customgame resolve = ONE `random(2,6)`; gen3ou resolve = the SetStatus shuffle +
    ///     [if it lands] the `random(2,6)`). The resolve emits a PLAIN `|-status|<target>|slp`
    ///     (the yawn condition's `effectType` is not a Move → the plain `-status` branch). If the
    ///     target got statused between cast and resolve (or is now sleep-immune / Sleep-Clause-
    ///     blocked under gen3ou), the `-end [silent]` still fires but no sleep sets (`try_set_status`
    ///     no-ops draw-appropriately).
    ///   - The residual handler participates in the residual speed-sort tie-shuffle at (10, 19) —
    ///     unique among order-10 handlers, so its only tie is the OTHER mon's yawn at equal speed
    ///     (a yawn MIRROR at equal cached speed draws one `random(0,2)`).
    ///
    /// CLEARED on switch-out + faint (`clearVolatile`, like the other volatiles); `noCopy` so NOT
    /// Baton-Passable (never in the pass-set). `source_uid` is recorded for the `[of]` clause /
    /// the `trySetStatus` source (in gen-3 singles the source side is always `1 - holder_side`, so
    /// the resolve reads that side's CURRENT active — the exact slot never affects a `slp` apply,
    /// only the side matters for the Sleep Clause self-exemption). `None` at construction.
    pub yawn: Option<(u8, usize)>,

    /// The **TRANSFORM** copy overlay (`gen3_transform_v1`, `sim/pokemon.ts::transformInto`):
    /// `Some(base)` while this mon is `pokemon.transformed`, holding everything the copy
    /// OVERWROTE so `clearVolatile` can revert it on switch-out / faint. `None` at
    /// construction (and for every mon that never transforms).
    ///
    /// The copy REPLACES, in place, `species_id` / `stats[1..=5]` / `types_override` /
    /// `ability` / `boosts` / `set.moves` + `move_pp` + `move_maxpp` / `hidden_power_bp` /
    /// `cached_speed`, so EVERY downstream reader (damage, speed, type chart, PP legality,
    /// the request JSON's `moves[]`) sees the copied mon with no extra plumbing — the
    /// [`MimicOverlay`] pattern, one level up. NOT copied (probe-verified): hp / maxhp
    /// (the `storedStats` loop excludes hp), item, status, volatiles — and the subtle
    /// trio the request surfaces: the mon's **ident name** (`pokemon.name` is fixed at
    /// construction) plus the roster's `details` and `stats`, which read `this.details` /
    /// `baseStoredStats`, NEITHER of which `transformInto` refreshes. Those three read
    /// through this record's base fields (`bridge::details` / `serialize_mon` /
    /// `helpers::display_name`).
    pub transform: Option<TransformOverlay>,
    /// **TORMENT** (`gen3_torment_v1`) — the foe cannot select the move it used LAST.
    ///
    /// A `bool`, not a counter: gen-3 torment has NO `duration` and NO `onResidual`, so it is
    /// PERMANENT until the mon leaves the field, and it must NOT register a residual duration
    /// handler (a phantom one would join the NO_ORDER/subOrder-2 protect/stall/flinch tie group
    /// and draw a shuffle the sim never draws). Probe: `harness/probe_torment.js`.
    ///
    /// The restricted move is read LIVE from `last_move` at every legality check rather than
    /// stored — the sim's `onDisableMove` re-reads `pokemon.lastMove.id` each request, and
    /// `last_move` is already `None` for a Struggle, which reproduces the sim's
    /// `lastMove.id !== 'struggle'` exemption for free.
    pub torment: bool,
    /// The item this mon CONSUMED ITSELF, for **RECYCLE** (`gen3_recycle_v1`) — the sim's
    /// `pokemon.lastItem`.
    ///
    /// ⚠️ THE DISCRIMINATOR IS *WHICH PRIMITIVE REMOVED IT*, not that the slot went empty.
    /// `eatItem`/`useItem` set it (a berry eaten at its own threshold, White Herb spent);
    /// `takeItem` does NOT (Knock Off / Thief / Covet / Trick). So a Knock-Off'd item is NOT
    /// recyclable while a berry eaten three turns earlier still is — probe-settled
    /// (`harness/probe_recycle.js`). Setting this wherever the item merely LEAVES the mon is the
    /// intuitive implementation and it is WRONG.
    ///
    /// Stored as a DISPLAY NAME, matching `item`, so the restore is verbatim.
    /// It PERSISTS across a switch-out (it is a Pokemon field, not a volatile).
    pub last_item: String,
    /// The sim's `pokemon.activeMoveActions` — the count of MOVE ACTIONS this mon has taken since
    /// it entered, for **FAKE OUT** (`gen3_fakeout_v1`).
    ///
    /// ⚠️ NOT `active_turns`, and the distinction is the whole mechanic. It counts ACTIONS, not
    /// turns: a turn spent CANT-ed (flinch / full-para / asleep) still increments it and BURNS the
    /// gate, while a turn whose action was CANCELLED by gen-3 faint-cancels-all does NOT — the
    /// cancelled action never reaches `run_move`. Probe-settled (`harness/probe_fakeout.js`); an
    /// `active_turns` implementation passes four of the five probed cases and is silently wrong on
    /// the fifth, forever. Starts at 0 (unlike `active_turns`, which starts at 1) and resets to 0
    /// on EVERY switch-in — voluntary, drag, forced replacement and Baton Pass alike.
    pub active_move_actions: u32,
}

/// The TRANSFORM copy-overlay restore record (`gen3_transform_v1`).
/// See [`MonState::transform`].
#[derive(Debug, Clone)]
pub struct TransformOverlay {
    /// The mon's OWN species id (`pokemon.baseSpecies` — what `clearVolatile`'s
    /// `setSpecies(this.baseSpecies)` restores, and what the ident / request `details`
    /// keep showing WHILE transformed).
    pub base_species_id: String,
    /// The mon's OWN in-battle stats (`baseStoredStats` + hp) — restored on revert, and
    /// read WHILE transformed by the request roster's `stats` object (the sim serializes
    /// `baseStoredStats`, which `setSpecies(…, isTransform=true)` deliberately does NOT
    /// overwrite).
    pub base_stats: [u16; 6],
    /// The mon's OWN `types_override` (a Color Change in flight before the transform).
    pub base_types_override: Option<Vec<crate::dex::Type>>,
    /// The mon's OWN move ids. Captured AFTER any Mimic overlay, exactly like the sim's
    /// `moveSlots` at transform time — the Mimic record is restored separately, AFTER
    /// this one, so the pair together reproduces `baseMoveSlots`.
    pub base_moves: Vec<String>,
    /// The mon's OWN per-slot current PP (Transform's own use is already deducted).
    pub base_move_pp: Vec<u16>,
    /// The mon's OWN per-slot max PP.
    pub base_move_maxpp: Vec<u16>,
    /// The mon's OWN IV-derived Hidden Power BP (`baseHpPower`).
    pub base_hidden_power_bp: u8,
}

/// The MIMIC moveslot-overlay restore record (`gen3_move_coverage_batch6_v1`).
/// See [`MonState::mimic_overlay`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimicOverlay {
    /// The overlaid slot (the index where Mimic sat).
    pub slot: usize,
    /// Mimic's OWN remaining PP at copy time (restored on switch-out — "Mimic's own
    /// used pp persists").
    pub base_pp: u16,
    /// Mimic's own max PP (16), restored with it.
    pub base_maxpp: u16,
}

/// The COUNTER / MIRROR COAT reactive-volatile state (`gen3_move_coverage_batch5_v1`).
/// See [`MonState::reactive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reactive {
    /// `false` = the `counter` volatile (returns 2× a PHYSICAL hit), `true` = the
    /// `mirrorcoat` volatile (2× a SPECIAL hit).
    pub mirror: bool,
    /// `Some(2 × the last qualifying hit)` once ARMED (the recorder's `slot != null` +
    /// `damage`); `None` while no qualifying foe hit has landed this turn (the move's
    /// onTry then fails zero-draw). Each qualifying hit OVERWRITES it.
    pub damage: Option<u16>,
}

/// The **PARTIAL-TRAP** (`partiallytrapped`) volatile's state (`gen3_partial_trap_v1`).
/// See [`MonState::partial_trap`]. The gen-3 resolved condition needs exactly three
/// facts; `boundDivisor` is NOT stored because gen3 has neither Binding Band nor Grip
/// Claw, so it is ALWAYS 16 (the gen5 `onStart` override's `else` arm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialTrap {
    /// The TRAPPER's stable `uid` — always a mon on the OPPOSITE side (gen-3 singles).
    /// Every residual re-reads that side's CURRENT active and compares: the sim's
    /// `onResidual` releases when `!source.isActive || source.hp <= 0 ||
    /// !source.activeTurns`. NOT re-pointed by a Baton Pass (unlike the linked `trapped`
    /// volatile): a trapper that BPs out is simply no longer active ⇒ the trap releases.
    pub source_uid: usize,
    /// The trapping move's DISPLAY name (`Wrap` / `Fire Spin` / `Sand Tomb` / …) — the
    /// `-activate`/`-damage`/`-end` lines all render `this.effectState.sourceEffect`, so
    /// the byte form is per-move.
    pub move_name: String,
    /// The `partiallytrapped` `duration` — the ONE `random(3, 7)` draw at cast. The
    /// residual decrements FIRST and, at 0, fires the `onEnd` (a non-silent `-end`) INSTEAD
    /// of the chip, so the number of CHIP turns is `duration − 1` (2–5).
    pub duration: u8,
}

/// The Solar Beam charge state (`gen3_move_coverage_batch4c_v1` — the gen-3 `twoturnmove`
/// volatile + its `solarbeam` sub-volatile), probe `harness/probe_batch4c_solarbeam.js`:
///
///   * CHARGE turn: after onBeforeMove passes + PP is deducted (−1; −2 under a Pressure
///     foe — Pressure applies at the CHARGE), `onTryMove` emits `|move|<user>|Solar
///     Beam||[still]` + `|-prepare|` and (no sun) adds this volatile: `move_index` = the
///     user's Solar Beam slot, `duration` = 2, `charging` = true (the `solarbeam`
///     sub-volatile). ZERO move draws (no accuracy/crit/damage), `landed` false.
///   * The FIRE turn's request offers ONLY `{move:"Solar Beam",id:"solarbeam"}` +
///     `trapped:true` (`twoturnmove.onLockMove`); the fire deducts NO PP, removes the
///     sub-volatile (`charging` = false) and executes normally (accuracy 100 DRAWN → crit
///     → damage; `|move|…|[from]lockedmove`).
///   * An ABORT on the fire turn (slp/par/frz/flinch cant) fires `onMoveAborted` → the
///     WHOLE state is removed (the charge is LOST; a fresh charge re-pays PP).
///   * The volatile registers a NO_ORDER/subOrder-2 residual DURATION handler on BOTH the
///     charge-turn and fire-turn residuals (the protect/stall/flinch tie group);
///     `duration` 2 → 1 → 0 removes it (draw-free). After a fire-turn KO it LINGERS
///     through the faint pause and is cleaned by the resumed tail's residual.
///   * SUN (`effectiveWeather`, Cloud Nine-aware) SKIPS the charge entirely (no volatile).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoTurnMove {
    /// The user's Solar Beam move-slot index (the locked move).
    pub move_index: usize,
    /// Remaining `twoturnmove` duration (2 at set; decremented at each residual; removed
    /// at 0).
    pub duration: u8,
    /// Whether the `solarbeam` sub-volatile is still present (true = still CHARGING — the
    /// request is locked; false = the beam FIRED this turn, the volatile merely awaits its
    /// residual expiry).
    pub charging: bool,
}

/// A pending FUTURE-MOVE strike (`gen3_move_coverage_batch4c_v1` — the gen-3 `futuremove`
/// SLOT condition on the TARGET side: Doom Desire / Future Sight), probe
/// `harness/probe_batch4c_doomdesire.js`. gen3 future moves are a CAST-TIME DAMAGE
/// SNAPSHOT + a slot-keyed order-11 delayed strike:
///
///   * CAST (`onTry`): `addSlotCondition(target,'futuremove')` → ONE `random(16)`
///     (`getDamage`'s randomizer) computes `damage` with CAST-TIME stats/boosts, moveData
///     `{bp 120(DD)/80(FS), category Physical(DD)/Special(FS), type '???', willCrit:false}`
///     — NO accuracy roll, NO crit roll at cast. Emits `|-start|<caster>|Doom Desire`.
///     A DOUBLE-CAST (any futuremove already pending on the slot) FAILS with a bare
///     `|move|` line, ZERO draws — but PP IS still deducted.
///   * IDLE: the pending condition registers an order-11 residual handler EVERY end-of-turn
///     (cast turn included; speed = the slot occupant's cached speed — an equal-speed FS
///     mirror tie-shuffles), decrementing `duration` 3 → 2 → 1.
///   * RESOLVE (duration hits 0, end of turn N+2, `onEnd`): skip (no strike) iff the slot
///     occupant is FAINTED; else `|-end|<target>|move: Doom Desire`, remove the target's
///     Protect volatile, ONE accuracy roll (`randomChance(85|90,100)`, the standard
///     hitStepAccuracy fold), and on a hit the STORED number lands fixed-damage-style (NO
///     crit / damage roll; a Substitute absorbs with no carry; typeless — no chart, so no
///     immunity ever). A LANDED resolve draws TWO extra `eachEvent('Update')` shuffles
///     (tie-only). Resolves even when the CASTER switched out or fainted; hits WHOEVER
///     occupies the slot (slot semantics).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FutureMove {
    /// Remaining residual ticks (3 at cast; the strike fires when the tick reaches 0).
    pub duration: u8,
    /// The CAST-TIME damage snapshot (the stored number applied at resolve).
    pub damage: u16,
    /// `doomdesire` | `futuresight` (the display name is re-derived from the dex).
    pub move_id: String,
    /// The resolve-time accuracy numerator (DD 85 / FS 90, the gen3 dex value).
    pub accuracy: u16,
    /// The CASTER's side + stable uid (for the resolve accuracy fold + the `-miss` line;
    /// the strike resolves even if the caster is benched/fainted).
    pub source_side: usize,
    pub source_uid: usize,
}

/// Compute a gen-3 Hidden Power's BASE POWER from the attacker's IVs, mirroring
/// Showdown's `Dex.getHiddenPower` (`sim/dex.ts`, the gen 3-5 branch):
///   `hpPowerX = Σ i·(⌊iv_s/2⌋ mod 2)` over the six stats in Showdown's iteration
///   order `{hp, atk, def, spe, spa, spd}` with weights `i = 1,2,4,8,16,32`, then
///   `power = ⌊hpPowerX·40/63⌋ + 30` (range 30..=70). `⌊iv/2⌋ mod 2` is the IV's
///   2nd-LSB (`(iv >> 1) & 1`); the TYPE is derived from the LSBs (already carried
///   by the typed move id, so untouched here).
///
/// THE WEIGHT-ORDER CRUX (`gen3_iv_derived_hidden_power_bp_v1`): the port's IV array
/// is `[HP, Atk, Def, SpA, SpD, Spe]` (= [`PokemonSet::ivs`] / `stats` order), but the
/// sim iterates `{hp, atk, def, SPE, SPA, spd}` — so the **Speed** IV (port idx 5)
/// carries weight **8** (BEFORE SpA idx 3 at weight 16 and SpD idx 4 at weight 32).
/// A naive `[HP,Atk,Def,SpA,SpD,Spe]→1,2,4,8,16,32` mapping is WRONG. Confirmed
/// bit-for-bit vs `getHiddenPower` (source read) AND a live damage probe (IV-31
/// spread → BP 70 → 84 dmg; a `spe:0` spread → BP 64 → 77 dmg — the port used to
/// deal 84 for both because the data hard-codes BP 70).
pub fn hidden_power_bp(ivs: &[u8; 6]) -> u8 {
    let bit = |iv: u8| ((iv >> 1) & 1) as u32; // ⌊iv/2⌋ mod 2 == the 2nd-LSB
    let hp_power_x = bit(ivs[0])   // hp  → 1
        + 2 * bit(ivs[1])          // atk → 2
        + 4 * bit(ivs[2])          // def → 4
        + 8 * bit(ivs[5])          // spe → 8  (SPE BEFORE SPA/SPD — the crux)
        + 16 * bit(ivs[3])         // spa → 16
        + 32 * bit(ivs[4]); // spd → 32
    let bp = (hp_power_x * 40 / 63) + 30;
    // GIGO guard: the gen3 formula's output is mathematically bounded to 30..=70
    // (hpPowerX ∈ 0..=63 → ⌊hpPowerX·40/63⌋ ∈ 0..=40). A value outside this range
    // means the weight order or the arithmetic was corrupted — fail loud rather
    // than silently mis-damage every Hidden Power.
    assert!(
        (30..=70).contains(&bp),
        "gen3_iv_derived_hidden_power_bp_v1: Hidden Power BP {bp} out of range 30..=70 \
         (ivs={ivs:?}) — the IV weight order is corrupted"
    );
    bp as u8
}

/// The IV-derived Hidden Power TYPE id suffix (e.g. `"dark"`), gen 3 formula
/// (`Dex.getHiddenPower`) — the sibling of [`hidden_power_bp`], sharing the SAME
/// weight order (the SPE crux) but reading the IV's **LSB** (`iv % 2`, the type
/// bits) where BP reads the 2nd-LSB. `type = HP_TYPES[⌊hpTypeX·15/63⌋]` over the
/// 16-type table. Used to serialize the OWNER's own roster move id for a bare-stored
/// Hidden Power (`getSwitchRequestData` resolves each moveSlot to its typed id).
pub fn hidden_power_type(ivs: &[u8; 6]) -> &'static str {
    // The gen-3 Hidden Power type table (`Dex.getHiddenPower`'s `hpTypes`).
    const HP_TYPES: [&str; 16] = [
        "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire",
        "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
    ];
    let bit = |iv: u8| (iv & 1) as u32; // iv % 2 == the LSB (the type bit)
    let hp_type_x = bit(ivs[0])   // hp  → 1
        + 2 * bit(ivs[1])          // atk → 2
        + 4 * bit(ivs[2])          // def → 4
        + 8 * bit(ivs[5])          // spe → 8  (SPE BEFORE SPA/SPD — the weight crux)
        + 16 * bit(ivs[3])         // spa → 16
        + 32 * bit(ivs[4]); // spd → 32
    let idx = (hp_type_x * 15 / 63) as usize;
    HP_TYPES[idx.min(15)]
}

/// Resolve a mon's stored move id to its **TYPED** Hidden Power id
/// (`gen3_typed_hidden_power_ids_v1`): a BARE `hiddenpower` becomes `hiddenpower<type>`,
/// preferring the packed `set.hpType` marker and falling back to the IV derivation
/// (mirroring the sim constructor). Every other move id passes through `to_id`'d.
///
/// The SINGLE home for that resolution: the request roster (`bridge::side_move_id`) and
/// the TRANSFORM copy (`gen3_transform_v1`) both need it, and they must agree — the copy
/// resolves the TARGET's HP against the TARGET's own markers/IVs, because the sim copies
/// `hpType`/`hpPower` for gen<5 and renders the copied slot as `Hidden Power <that type>`.
/// Resolving at copy time is what keeps the transformed mon's request/`|move|` bytes right
/// without threading the source mon into every later reader.
pub fn typed_hp_move_id(mon: &MonState, mv: &str) -> String {
    let id = crate::dex::to_id(mv);
    if id == "hiddenpower" {
        let ty = if !mon.set.hp_type.is_empty() {
            crate::dex::to_id(&mon.set.hp_type)
        } else {
            hidden_power_type(&mon.set.ivs).to_string()
        };
        if !ty.is_empty() && ty != "normal" {
            return format!("hiddenpower{ty}");
        }
    }
    id
}

impl MonState {
    /// Build the construction-time state for one decoded set: compute its stats,
    /// set `hp == maxhp == stats[0]`, status empty, boosts 0, not fainted.
    ///
    /// `position` is the 0-based team slot (set by the caller as it walks the
    /// team, matching `Side.addPokemon`'s `position = pokemon.length`).
    ///
    /// Crash-don't-drop: returns `Err` (never a silent wrong value) if the dex
    /// can't compute the stats (unknown species/nature) — that would corrupt
    /// every downstream HP/damage number.
    pub fn from_set(set: PokemonSet, position: usize, dex: &Dex) -> Result<MonState, String> {
        let stats = compute_stats(&set, dex)
            .map_err(|e| format!("MonState::from_set(slot {position}): {e}"))?;
        let maxhp = stats[0];
        // Resolve the species id via the dex (canonical, normalized). The set's
        // `species` may be a display name; `dex.species` is `to_id`-keyed.
        let species_id = dex
            .species(&set.species)
            .map(|s| s.id.clone())
            .ok_or_else(|| {
                format!("MonState::from_set(slot {position}): unknown species {:?}", set.species)
            })?;
        let level = set.level; // read before `set` is moved into the struct
        // Per-move PP init (`gen3_pp_tracking_v1`): each slot starts at the move's
        // in-battle MAX PP (`MoveData::max_pp` = the ctor's `calculatePP(move, 3)` for
        // a normal move, or the raw `pp` for a `no_pp_boosts` move). An unknown move id
        // (should not occur — a set is validated at unpack) contributes 0 PP so a GIGO
        // slot can never be spuriously "usable". VERIFIED vs the sim's moveSlots[k].maxpp.
        let move_maxpp: Vec<u16> = set
            .moves
            .iter()
            .map(|mid| dex.moves(mid).map(|m| m.max_pp()).unwrap_or(0))
            .collect();
        let move_pp = move_maxpp.clone();
        let item = set.item.clone();
        let ability = set.ability.clone();
        // FORECAST is MODELED (`gen3_forecast_v1`, ROUND 35) — the old fail-loud GIGO guard
        // here is RETIRED: `turn/forecast.rs` models the forme + TYPE swap at every
        // `eachEvent('WeatherChange')` site, the entrant `onStart`, the start window, and
        // the `clearVolatile` revert; the hail/sandstorm MOVES are modeled weather-setters
        // so every forme is reachable. `forecast_castform_builds_now_that_forecast_is_modeled`
        // is the negative control that the guard stays retired.
        // FAIL-LOUD GIGO guard for HELD ITEMS THE ENGINE DOES NOT MODEL AT ALL
        // (`gen3_unmodeled_item_failloud_v1`). The item sibling of the ability guard above and
        // the move guard below, closing the last limb of the same class: an item the port has
        // no handler for neither crashes nor works — it SILENTLY does nothing while the sim
        // applies it, so the policy is fed wrong observations for the rest of the battle.
        //
        // THE LIST IS NARROW AND MEASURED, because a blanket "unreferenced id" guard would be
        // WRONG: most items the engine never mentions genuinely do nothing in a gen-3 battle.
        // Resolving every id in `data/pokemon/gen3_items.json` through `Dex.mod('gen3')` and
        // discarding the gen7+ universal no-ops (`onDrive`/`onMemory`/`onPlate`) plus
        // `Fling`/`NaturalGift` leaves 98 of 132 items with a real battle handler, of which 32
        // are unreferenced by this engine. Twenty-seven of those 32 are inert here — gen-2
        // carry-overs (goldberry / mintberry / iceberry / bitterberry / burntberry /
        // psncureberry / miracleberry / mysteryberry) and Pokéblock berries whose ONLY handler
        // is `onEat`. The FIVE below are the ones with a genuine gen-3 battle effect:
        //   * `shellbell`   — `onAfterMoveSecondarySelf`, drains 1/8 of the damage dealt
        //   * `berryjuice`  — `onUpdate` + `onResidual`, restores 20 HP at <= 50%
        //   * `mentalherb`  — `onUpdate`, cures attraction
        //   * `machobrace`  — `onModifySpe`, HALVES Speed (so it is turn-ORDER and therefore
        //                     DRAW relevant — a silent miss desyncs the speed-tie shuffle)
        //   * `mail`        — `onTakeItem`, blocks item theft (Knock Off / Thief / Trick would
        //                     otherwise wrongly take it)
        //
        // EXPOSURE IS ZERO ON BOTH SURFACES WE ACTUALLY USE: 0 hits across 4722 training-pool
        // team-blocks in `data/teams/`, and 0 across 3000 generated `gen3randombattle` teams.
        // So this is a LATENT-HAZARD guard, not a fix for an active bug — it exists so that the
        // day one of these reaches the engine (a hand-built team, a format change, a new pool)
        // it CRASHES instead of lying. An earlier audit's "~2.7% of ladder battles" figure was
        // inferred from Smogon usage weights and does not describe either surface.
        const UNMODELED_FAILLOUD_ITEMS: [&str; 5] =
            ["shellbell", "berryjuice", "mentalherb", "machobrace", "mail"];
        let item_id = crate::dex::to_id(&item);
        if UNMODELED_FAILLOUD_ITEMS.contains(&item_id.as_str()) {
            panic!(
                "MonState::from_set(slot {position}): item {item_id} is unmodeled — \
                 GIGO guard; reject the team upstream"
            );
        }
        // FAIL-LOUD GIGO guard for MOVES THE ENGINE DOES NOT MODEL AT ALL
        // (`gen3_unmodeled_move_failloud_v1`). The SAME reasoning as Forecast above, and the
        // same discovery route — a differential fuzzer, not review. Such a move would neither
        // crash nor work: `run_status_move` would fall through and it would SILENTLY do
        // nothing while the sim rewrote state and emitted lines. That is the worst failure
        // mode for `--use-bridge=rust` — not a crash, but WRONG OBSERVATIONS fed to the policy
        // for the rest of the battle. Keyed on the MOVE appearing in the set (never the
        // species), so it catches ACTIVE and BENCH mons alike at construction, before any turn
        // runs, and is GIGO-proof against an odd carrier.
        //
        // The moves once deferred behind this seam are MODELED:
        //   * the WRAP FAMILY (`wrap`/`bind`/`firespin`/`clamp`/`whirlpool`/`sandtomb`) —
        //     `gen3_partial_trap_v1`, ROUND 32 (repro `soak3/divergences/sbd_msb1zfxs_b237`:
        //     node emitted `|-activate|p1a: Snorlax|move: Wrap|[of] p2a: Shuckle`, the port
        //     emitted nothing).
        //   * `transform` — `gen3_transform_v1`, ROUND 33: the copy OVERLAY (species / types /
        //     stats−HP / ability / boosts / moveslots at 5 PP / hpType+hpPower), reverted on
        //     switch-out + faint (repro `sim_bridge_diff_out/soak_randbats/.../sbd_msapcesj_b22`:
        //     node emitted `|-transform|p1a: Ditto|p2a: Kyogre`, the port emitted nothing).
        // BOTH were found by the LIVE bridge gate (`gen_sim_bridge_diff.js`), which drives
        // choices off the sim's own request and so reaches moves the offline fuzzers' pickers
        // filtered out — the coverage hole this guard class exists to make LOUD.
        //
        // **THE CURRENT LIST — the ROUND 40 move-audit's SILENT-DESYNC class**
        // (`gen3_unmodeled_move_failloud_v2`, the moves sibling of the item audit above,
        // MEASURED the same way): crossing the full 369-move gen3-legal universe against
        // BOTH oracles — `isModeledMove` (the e2e lockstep predicate) and the empirical
        // `scan_move_probe` engine run (zero disagreements on the modeled set) — leaves 91
        // unmodeled moves, of which 74 already FAIL LOUD at runtime (the run_status_move /
        // fixed-damage / charge guards — honest crash, NOT listed here) and SIXTEEN RUN
        // with no guard while being outside the modeled universe. Each of the 16 has a
        // genuine gen-3 sub-mechanic the engine would silently get WRONG (not a no-op):
        //   * variable/derived BP run at a FLAT wrong power: `eruption` (hp-scaled 150),
        //     `revenge` (×2 if damaged this turn), `smellingsalts` (×2 vs para + cures it),
        //     `furycutter` (escalates per consecutive hit), ~~`weatherball`~~ MODELED `gen3_weather_ball_v1` (type+BP+category change
        //     under weather);
        //   * the LOCK-IN family run as plain one-turn hits (no lock, no escalation, no
        //     end-of-lock confusion): `outrage`/`petaldance`/`thrash`, `rollout`/`iceball`
        //     (+ the Defense Curl doubling), `uproar` (+ the sleep-block), `rage`;
        //   * a missing execution GATE: `fakeout` (first-turn-only onTry — would spam every
        //     turn), `falseswipe` (would KO instead of leaving 1 HP), `dreameater` (the
        //     asleep-only gate — would hit awake targets), `secretpower` (the terrain-typed
        //     secondary).
        // EXPOSURE IS ZERO ON BOTH SURFACES WE ACTUALLY USE (the item-audit measurement,
        // repeated): 0 carriers across the 773 `data/teams/` team files (the 719-team
        // training pool's 108 distinct moves are ALL modeled), and 0 across the ENTIRE
        // curated gen3randombattle movepool (220 species / 393 sets / 113 distinct moves —
        // exhaustive, not sampled). So this is a LATENT-HAZARD guard like the item one: the
        // day one of these reaches the engine (a hand-built team, a format change, a new
        // pool) it CRASHES at construction instead of lying to the policy. `sleeptalk` is
        // deliberately NOT listed (engine-MODELED, batch 5 — the picker rejects it only as
        // a carrier-conditional).
        //
        // The seam's negative controls: `a_ditto_without_transform_builds_fine` /
        // `transform_carriers_build_now_that_transform_is_modeled` stop an over-broad guard
        // from silently returning.
        const UNMODELED_FAILLOUD_MOVES: [&str; 0] = [
        ];
        if let Some(bad) = set
            .moves
            .iter()
            .map(|m| crate::dex::to_id(m))
            .find(|id| UNMODELED_FAILLOUD_MOVES.contains(&id.as_str()))
        {
            panic!(
                "MonState::from_set(slot {position}): {bad} is unmodeled — \
                 GIGO guard; reject the team upstream"
            );
        }
        let hp_bp = hidden_power_bp(&set.ivs); // read before `set` is moved into the struct
        let set_gender = set.gender; // read before `set` is moved into the struct
        Ok(MonState {
            set,
            item,
            item_knocked_off: false,
            ability,
            focus_energy: false,
            // `pokemon.baseSpecies` — the construction species, never rewritten in gen 3
            // (`gen3_forecast_v1`). Cloned BEFORE `species_id` is moved into the struct.
            base_species_id: species_id.clone(),
            species_id,
            level,
            stats,
            hp: maxhp,
            maxhp,
            hidden_power_bp: hp_bp,
            status: None,
            boosts: [0; BOOST_LEN],
            fainted: false,
            confusion: None,
            flinch: false,
            protected: false,
            protect_counter: 0,
            stall_duration: 0,
            leech_seed: None,
            substitute: None,
            substitute_broken: false,
            taunt: None,
            disable: None,
            last_move: None,
            last_move_was_self_overwrite: false,
            move_pp,
            move_maxpp,
            choice_locked_move: None,
            perish_turn: 0,
            noorder_vol_turn: 0,
            flash_fire: false,
            truant_turn: false,
            gender: set_gender,
            attract: None,
            types_override: None,
            last_move_used: None,
            uproar: None,
            damaged_by_foe_this_turn: false,
            fury_cutter: None,
            locked_move: None,
            rollout: None,
            defense_curl: false,
            rage: false,
            minimize: false,
            imprison: false,
            position,
            uid: position, // the construction-time index is the stable identity
            // `pokemon.speed` is initialized to the raw `storedStats.spe` (the
            // switch-in value); `BattleState::update_speed` refreshes it para/boost-
            // aware at turn-start + residual-start, exactly as Showdown does.
            cached_speed: stats[5] as u32,
            // A LEAD has activeTurns 1 at its first residual (VERIFIED vs the sim) so it
            // boosts Speed Boost on turn 1; a switch-in RESETS to 0 in `execute_switch`.
            active_turns: 1,
            curse: None,
            focus_punch: None,
            beat_up: false,
            pursuit: None,
            must_recharge: false,
            two_turn: None,
            reactive: None,
            sleep_skipped: 0,
            encore: None,
            destiny_bond: false,
            destiny_bond_ko_by: None,
            endure: false,
            perish: None,
            trapped_by: None,
            partial_trap: None,
            charge: false,
            mimic_overlay: None,
            snatch: false,
            sleep_from_rest: false,
            switchin_foe_uid: None,
            yawn: None,
            transform: None,
            torment: false,
            last_item: String::new(),
            active_move_actions: 0,
        })
    }

    /// Restore the MIMIC moveslot overlay (`gen3_move_coverage_batch6_v1`) — the
    /// switch-out / faint `baseMoveSlots` revert: `set.moves[slot]` back to `mimic`
    /// with Mimic's OWN remaining PP (captured at copy time) + max PP. A no-op when no
    /// overlay is up. DRAW-FREE (pure state).
    /// Whether this mon is `pokemon.transformed` (`gen3_transform_v1`) — the guard the sim
    /// reads for `mimic.onHit`'s `source.transformed` fail and `transformInto`'s
    /// `pokemon.transformed && gen >= 2` target fail.
    pub fn transformed(&self) -> bool {
        self.transform.is_some()
    }

    /// Revert the TRANSFORM copy overlay (`gen3_transform_v1`) — the switch-out / faint
    /// `clearVolatile` restore: `moveSlots = baseMoveSlots.slice()`, `transformed = false`,
    /// `ability = baseAbility`, `hpType/hpPower = base*`, and `setSpecies(this.baseSpecies)`
    /// (species + types + the five stored stats + the speed cache). Boosts are NOT restored
    /// here — `clearVolatile` zeroes them unconditionally, which every caller already does.
    ///
    /// MUST run BEFORE [`Self::restore_mimic_overlay`]: this puts back the moveset AS IT WAS
    /// AT TRANSFORM TIME (which still carries any Mimic copy), and the Mimic record then
    /// puts Mimic back — together reproducing the sim's `baseMoveSlots`. A no-op when no
    /// overlay is up. DRAW-FREE (pure state).
    pub fn restore_transform_overlay(&mut self) {
        if let Some(ov) = self.transform.take() {
            self.species_id = ov.base_species_id;
            // `setSpecies` restores all five non-HP stored stats; hp/maxhp were never copied.
            self.stats[1] = ov.base_stats[1];
            self.stats[2] = ov.base_stats[2];
            self.stats[3] = ov.base_stats[3];
            self.stats[4] = ov.base_stats[4];
            self.stats[5] = ov.base_stats[5];
            // `setSpecies` ends with `this.speed = this.storedStats.spe` — the RAW stat
            // (no boosts / paralysis), exactly like the transform-time write.
            self.cached_speed = ov.base_stats[5] as u32;
            self.types_override = ov.base_types_override;
            // `clearVolatile` restores `this.baseAbility`, NOT the live ability the copy
            // overwrote — so a Trace-then-Transform reverts to the SET ability, matching the
            // switch-out Trace revert (`gen3_berry_trace_shedskin_v1`) that already runs at the
            // voluntary-switch site. Restoring the captured LIVE ability instead would undo that
            // site's work (it runs FIRST) and re-install the traced ability on the bench.
            self.ability = self.set.ability.clone();
            self.set.moves = ov.base_moves;
            self.move_pp = ov.base_move_pp;
            self.move_maxpp = ov.base_move_maxpp;
            self.hidden_power_bp = ov.base_hidden_power_bp;
        }
    }

    pub fn restore_mimic_overlay(&mut self) {
        if let Some(ov) = self.mimic_overlay.take() {
            if let Some(m) = self.set.moves.get_mut(ov.slot) {
                *m = "mimic".to_string();
            }
            if let Some(pp) = self.move_pp.get_mut(ov.slot) {
                *pp = ov.base_pp;
            }
            if let Some(mp) = self.move_maxpp.get_mut(ov.slot) {
                *mp = ov.base_maxpp;
            }
        }
    }

    /// Whether this mon's request is MOVE-LOCKED to a single pseudo/locked move
    /// (`gen3_move_coverage_batch4c_v1` — `mustrecharge.onLockMove='recharge'` /
    /// `twoturnmove.onLockMove='solarbeam'`): the request offers ONE `{move,id}` entry
    /// (no pp/maxpp/target/disabled) + `trapped:true`; `move 1` is the only legal move
    /// choice and a voluntary switch is REJECTED ("Can't switch: The active Pokémon is
    /// trapped" — the FIRM-trap shape, probe-verified request JSON).
    pub fn move_locked(&self) -> bool {
        self.must_recharge
            || self.two_turn.map_or(false, |t| t.charging)
            // UPROAR locks the same way (`gen3_uproar_v1`): the request offers ONLY Uproar and
            // is `trapped:true`, so folding it here gives the whole FIRM shape for free.
            || self.uproar.is_some()
            // The OUTRAGE / PETAL DANCE / THRASH lock (`gen3_lockin_family_v1`) locks the
            // same way — the request offers only the locked move and is `trapped:true`.
            || self.locked_move.is_some()
            // ROLLOUT / ICE BALL lock for 5 executions (`gen3_rollout_defensecurl_v1`).
            || self.rollout.is_some()
    }

    /// The current PP of move slot `k` (`gen3_pp_tracking_v1`), or `0` for an
    /// out-of-range slot (a phantom slot is never "usable").
    pub fn pp_of(&self, k: usize) -> u16 {
        self.move_pp.get(k).copied().unwrap_or(0)
    }

    /// The **ENFORCED** Choice lock (`gen3_choicelock_lazy_release_v1`): the `choicelock`
    /// VOLATILE (`choice_locked_move`) narrowed by the handler's own early-out.
    ///
    /// The resolved gen3 `choicelock.onDisableMove` opens with
    /// ```js
    /// if (!pokemon.getItem().isChoice || !pokemon.hasMove(this.effectState.move)) {
    ///     pokemon.removeVolatile('choicelock'); return;
    /// }
    /// ```
    /// so the lock is enforced ONLY while the mon still HOLDS a Choice item — and the volatile
    /// itself survives the item's loss until the next `runEvent('DisableMove')` gathers it and
    /// it self-removes. Keeping the two apart matters for BOTH observables: legality reads the
    /// ENFORCED lock (a Knock-Off'd / Trick'd mon may pick any move again — the e2e_126
    /// behaviour), while the endTurn DisableMove tie-shuffle counts the VOLATILE (the missing
    /// draw in `rmrzcanyf_ab_71_14`; SIM-PROBED by `harness/probe_rb_choicelock.js`, where the
    /// Knock-Off turn's endTurn STILL draws the encore+choicelock shuffle and the NEXT turn's
    /// does not).
    ///
    /// The `hasMove(effectState.move)` half is modeled EAGERLY at the one site that can break
    /// it — Mimic overwriting its own locked slot (`gen3_mimic_choice_lock_self_overwrite_v1`,
    /// `turn/status_moves.rs`) — so this reads only the item half.
    pub fn choice_lock_enforced(&self, dex: &Dex) -> Option<usize> {
        let locked = self.choice_locked_move?;
        if dex.item(&crate::dex::to_id(&self.item)).map(|i| i.choice).unwrap_or(false) {
            Some(locked)
        } else {
            None
        }
    }

    /// Whether move slot `k` is USABLE now — an in-range slot with >0 PP that is NOT disabled
    /// by any move-SELECTION restriction (`gen3_pp_tracking_v1` + `gen3_taunt_disable_v1`):
    ///   - the **Choice lock** (`choicelock.onDisableMove`): if the mon is choice-locked to slot
    ///     `j` AND still holds the Choice item (`choice_lock_enforced`), only slot `j` is usable;
    ///   - **DISABLE** (`disable.onDisableMove`): the one recorded `disable` slot is un-usable;
    ///   - **TAUNT** (`taunt.onDisableMove`): EVERY Status-category slot is un-usable (needs the
    ///     dex to read the slot move's category).
    /// A slot un-usable by ANY of these can't be SELECTED (`side.choose` rejects it) and, if ALL
    /// slots are un-usable, the mon is forced to Struggle. The `dex` is read only for Taunt's
    /// per-slot category; the other gates are pure state.
    pub fn move_usable(&self, k: usize, dex: &Dex) -> bool {
        if let Some(locked) = self.choice_lock_enforced(dex) {
            if k != locked {
                return false; // a non-locked slot is disabled by the Choice lock
            }
        }
        // DISABLE: the one recorded slot is un-usable while the volatile is up.
        if let Some((disabled_slot, _)) = self.disable {
            if k == disabled_slot {
                return false;
            }
        }
        // TORMENT (`gen3_torment_v1`): the slot used LAST is un-usable while the volatile is up.
        // Read LIVE from `last_move` — the sim's `onDisableMove` disables `lastMove.id` at every
        // request build, so a Mimic that rewrote the slot must not keep disabling the old id.
        // `last_move_was_self_overwrite` is the Mimic guard Disable already uses
        // (`gen3_mimic_disable_self_overwrite_v1`): the sim's `disableMove` finds no matching
        // slot then, so nothing is disabled. `last_move == None` after a Struggle IS the sim's
        // `lastMove.id !== 'struggle'` exemption, which is why Struggle frees the slot next turn.
        if self.torment && !self.last_move_was_self_overwrite && self.last_move == Some(k) {
            return false;
        }
        // ENCORE (`gen3_move_coverage_batch6_v1`, `encore.onDisableMove`): every slot
        // EXCEPT the encored one is un-usable while the volatile is up (the request
        // shows the other slots `disabled:true`, the exact Disable/Taunt shape — EN1/
        // EN8; a mon may still SWITCH — encore never traps). The encored slot itself
        // then falls through to the PP gate below (encore ends at the residual the
        // moment the slot hits 0 PP, so a live encore's slot always has PP at a
        // request boundary).
        if let Some((encored_slot, _)) = self.encore {
            if k != encored_slot {
                return false;
            }
        }
        // TAUNT: every Status-category slot is un-usable while taunted. Uses the taunt-block
        // predicate (`gen3_taunt_disable_v1`), which EXCLUDES the fixed-damage moves our
        // base-power-derived category mis-classifies as Status (Seismic Toss etc. stay usable
        // under Taunt — VERIFIED vs the sim).
        if self.taunt.is_some() {
            if let Some(mid) = self.set.moves.get(k) {
                if let Some(m) = dex.moves(mid) {
                    if m.blocked_by_taunt() {
                        return false;
                    }
                }
            }
        }
        k < self.move_pp.len() && self.move_pp[k] > 0
    }

    /// Whether the mon has NO usable move — so it is FORCED to Struggle (`side.choose`'s
    /// `!moves.length` → `moveid:'struggle'`). "Usable" respects the **Choice lock**, **Disable**,
    /// AND **Taunt** (`gen3_taunt_disable_v1`): a taunted mon whose only remaining moves are all
    /// Status, or a mon whose only PP-bearing slot is Disabled, must Struggle even though a slot
    /// still has PP (mirroring the CB-Tyranitar that exhausts Crunch → Struggle). An empty
    /// movepool (should not occur in gen-3) also Struggles. The `dex` is read for Taunt's per-slot
    /// category.
    pub fn must_struggle(&self, dex: &Dex) -> bool {
        if self.move_pp.is_empty() {
            return true;
        }
        (0..self.move_pp.len()).all(|k| !self.move_usable(k, dex))
    }

    /// Deduct `amount` PP from move slot `k`, floored at 0 (`Pokemon.deductPP`).
    /// DRAW-FREE. No-op for an out-of-range slot.
    pub fn deduct_pp(&mut self, k: usize, amount: u16) {
        if let Some(p) = self.move_pp.get_mut(k) {
            *p = p.saturating_sub(amount);
        }
    }

    /// The 4 move slots' current PP, padded with `-1` for a mon with fewer than 4 moves
    /// (`gen3_pp_tracking_v1`) — the fixed-width form the per-decision differential asserts
    /// (mirrors the sim's `moveSlots[k].pp`, `-1` where there is no slot k). PP fits in i16.
    pub fn pp_array(&self) -> [i16; 4] {
        let mut out = [-1i16; 4];
        for (k, slot) in out.iter_mut().enumerate() {
            if let Some(&p) = self.move_pp.get(k) {
                *slot = p as i16;
            }
        }
        out
    }
}

/// One side's state: its mons (in packed/team order, slot 0 = lead) + the active
/// slot index. Side-condition / slot-condition placeholders are empty at
/// construction (they are event products).
#[derive(Debug, Clone)]
pub struct SideState {
    /// Player display name (mirrors `side.name`).
    pub name: String,
    /// Mons in team order; `pokemon[0]` is the lead.
    pub pokemon: Vec<MonState>,
    /// The active mon's index in `pokemon`. Gen-3 singles: always 0 (the lead).
    /// (We record the structural lead index; we do NOT run the start action's
    /// `switchIn`, so this is the deterministic gen-3-singles lead, not an event
    /// output.)
    pub active: usize,
    /// Mons not yet fainted (== team size at construction; `side.pokemonLeft`).
    pub pokemon_left: usize,
    /// Whether this side's active is flagged for a post-faint replacement
    /// (`pokemon.switchFlag`, set by `checkFainted`). Engine-owned; false at
    /// construction. The full-battle driver sets it on a faint and clears it when
    /// the scripted replacement commits.
    pub switch_flag: bool,
    /// The **Spikes** entry-hazard layer count (`side.sideConditions.spikes.layers`),
    /// 0..=3 — the first SIDE CONDITION (a per-side persistent state, reusable by
    /// future hazards/phazing). Spikes targets the FOE side: using the Spikes move
    /// increments the CASTER's FOE side's `spikes` (capped at 3; a Spikes at 3 FAILS).
    /// It PERSISTS across switches (a side condition, not a mon volatile) and is 0 at
    /// construction. Applied as switch-in damage to a GROUNDED entrant (not Flying,
    /// not Levitate) via the gen-3 `runSwitch`'s `runEvent('EntryHazard')`. DEFERRED
    /// (excluded / fail-loud): Toxic Spikes + Stealth Rock (NOT gen3), Rapid Spin (the
    /// hazard-clear move) — Spikes is the only gen-3 entry hazard.
    pub spikes: u8,
    /// The **Light Screen** side-condition remaining-turn counter (`side.sideConditions.
    /// lightscreen.duration`), `gen3_move_coverage_batch2_v1`. 0 = not up; set to 5 by the
    /// Light Screen MOVE (gen3 has no Light Clay → always 5), decremented ONCE per end-of-turn
    /// SIDE residual (`onSideResidual`), cleared to 0 at expiry (emitting `|-sideend|…|move:
    /// Light Screen`). While up, the damage calc HALVES incoming SPECIAL damage to this side
    /// (`DamageContext::light_screen`, crit-bypassed). A per-side persistent state (like
    /// spikes); 0 at construction.
    pub light_screen: u8,
    /// The **Reflect** side-condition remaining-turn counter (`side.sideConditions.reflect.
    /// duration`), `gen3_move_coverage_batch2_v1`. As `light_screen` but HALVES incoming
    /// PHYSICAL damage (`DamageContext::reflect`) and expires with `|-sideend|…|Reflect`. 0 at
    /// construction.
    pub reflect: u8,
    /// The **SAFEGUARD** side-condition remaining-turn counter (`gen3_safeguard_v1`).
    /// A SIDE condition, not a volatile — it PERSISTS across switches and protects the
    /// ENTRANT, and there is nothing for Baton Pass to copy. Set to a FIXED 5 at cast
    /// (`durationCallback` returns 5 deterministically — Persistent is gen5+), ticked
    /// down by the order-4 side residual, `|-sideend|` at 0. While > 0 the side's mons
    /// are immune to FOE-sourced major status + confusion; a SELF-inflicted status
    /// (Rest) passes, and a PENDING Yawn's resolve is exempt. 0 at construction.
    pub safeguard: u8,
    /// The **WISH** pending slot-condition (`side.slotConditions[0].wish`,
    /// `gen3_move_coverage_batch3_v1`): `Some((duration, wisher_name))` when a Wish cast on
    /// THIS side is pending. `None` = no pending Wish (the construction-time value).
    ///
    /// Wish is a SLOT condition (a per-side, per-active-slot delayed heal — gen-3 singles has
    /// one active slot, so it is per-side here), NOT a mon volatile: it SURVIVES the wisher
    /// switching out / fainting / being phazed (slot-keyed), and the heal lands on WHOEVER
    /// occupies the slot at resolution (healing `floor(ITS maxhp/2)`). VERIFIED bit-for-bit vs
    /// the omniscient sim (`harness/probe_batch3_wish.js`):
    ///   - CAST: the Wish move (never-miss, `target: self`) sets this to `(2, wisher_name)` —
    ///     DRAW-FREE. A 2nd Wish while one is pending FAILS (`[still]`, no fail line, draw-free,
    ///     the existing Wish untouched → resolves normally).
    ///   - RESIDUAL (`wish.onEnd`, `onResidualOrder: 7` — BEFORE the sand chip order 8 and ALL
    ///     order-10 handlers): `duration` counts 2 → 1 (end of the cast turn) → fires at 0 (end
    ///     of the NEXT turn). On resolve, if the slot's mon is not fainted, heal
    ///     `floor(maxhp/2)`; a NON-zero heal emits `|-heal|<mon>|<hp>|[from] move: Wish|[wisher]
    ///     <name>` (a heal-at-full resolves SILENTLY — `this.heal` returns 0, the `if(damage)`
    ///     guard skips the line). DRAW-FREE, but the handler PARTICIPATES in the residual
    ///     speed-sort at order 7 (speed = the slot's active mon's cached speed), so TWO Wishes
    ///     resolving the same turn at EQUAL speed draw ONE tie-shuffle `random(0,2)` (probe:
    ///     a Blissey-mirror both-Wish resolve turn draws +1 vs the single-Wish control).
    ///   - The wisher FAINTING on the cast turn skips that turn's residual (the faint pauses),
    ///     leaving `duration` at 2 → resolves the following turn on the replacement.
    /// NOT cleared on switch (a side/slot condition — it persists); cleared only when it
    /// resolves (or expiry). `None` at construction.
    pub wish_pending: Option<(u8, String)>,
    /// The **BATON PASS** pending-pass marker (`gen3_move_coverage_batch3_v1`): `true` while a
    /// Baton Pass has resolved on THIS side and the entrant's `copyVolatileFrom` is awaiting
    /// the forced switch-in. `false` = no pending pass (the construction-time value).
    ///
    /// Baton Pass (`selfSwitch: 'copyvolatile'`) resolves like a voluntary self-switch: on a
    /// success the side's `switch_flag` is set AND this marker is set. `execute_switch` reads
    /// it (when the forced switch-in commits) to (a) SNAPSHOT the OUTGOING mon's PASS-SET —
    /// its 7 boosts + the copyable (`noCopy == false`) volatiles the port models (substitute
    /// HP / leech-seed seeder / confusion counter / curse source) — BEFORE the clearVolatile
    /// block zeros them, (b) APPLY that snapshot to the entrant AFTER the array swap, and
    /// (c) emit the `|switch|…|[from] Baton Pass` tag. Then it clears the marker. DRAW-FREE
    /// (the copy consumes no PRNG; the forced switch-in draws exactly like a normal switch).
    /// A Baton Pass with NO eligible bench FAILS draw-free (`[still]` + `-fail`) and never
    /// sets this. Major STATUS is NOT a volatile → NOT passed (it stays with the outgoing
    /// mon). `false` at construction.
    pub baton_pass_pending: bool,
    /// The pending **FUTURE-MOVE** slot condition on THIS side's active slot
    /// (`gen3_move_coverage_batch4c_v1` — Doom Desire / Future Sight cast BY THE FOE at
    /// this side's slot; `side.slotConditions[0].futuremove`). One condition per slot —
    /// a second cast while pending FAILS. Slot-keyed like `wish_pending` (survives the
    /// occupant switching/fainting; the strike hits whoever occupies the slot). See
    /// [`FutureMove`]. `None` at construction.
    pub future_move: Option<FutureMove>,
}

impl SideState {
    /// The active (lead) mon.
    pub fn active(&self) -> &MonState {
        &self.pokemon[self.active]
    }

    /// The active mon's resolved species id (e.g. `suicune`).
    pub fn active_species(&self) -> &str {
        &self.pokemon[self.active].species_id
    }
}

/// Global field state. At construction: no weather, no terrain (gen-3 has no
/// terrain anyway), weather counter 0.
#[derive(Debug, Clone, Default)]
pub struct Field {
    /// Active weather, or `None` (clear) at construction.
    pub weather: Option<Weather>,
    /// Remaining weather turns (0 when clear).
    pub weather_turns: u8,
}

/// The constructed in-battle state: PRNG, turn counter, the two sides, and the
/// field. This is the deterministic core the event engine mutates.
///
/// **`Clone` is the clone-and-branch snapshot** (`gen3_bridge_clone_branch_v1`): every
/// field below is plain owned data (no `Rc`/`RefCell`/`Box<dyn>`/raw pointers), so a
/// derived clone is a DEEP, fully independent battle — advancing it cannot touch the
/// original. Deliberately, the [`crate::dex::Dex`] is NOT held here (it is threaded per
/// call), so a snapshot does not copy the ~16 MB of parsed static data; keep it that way.
#[derive(Clone)]
pub struct BattleState {
    /// The battle PRNG, built from the seed (unused until events draw from it).
    pub prng: Prng,
    /// Turn counter. 0 at pure construction (Showdown sets turn 1 in the start
    /// action, which we do not run here).
    pub turn: u32,
    /// `[p1, p2]`.
    pub sides: [SideState; 2],
    /// Global field.
    pub field: Field,
    /// Generation (3) and format id, carried from the options.
    pub gen: u8,
    pub format_id: String,
    /// Whether the **Sleep Clause Mod** is active (a sleep move that would inflict a
    /// 2nd foe sleep FAILS). True for ladder formats that carry it via the `Standard`
    /// ruleset (e.g. `gen3ou`); FALSE for `gen3customgame` (the e2e/secondary/
    /// fullbattle harness format, which has no clauses). Derived from `format_id` at
    /// construction (see [`format_has_sleep_clause`]).
    pub sleep_clause: bool,
    /// Set by `run_move` when an Explosion / Self-Destruct SELF-KO fired this turn (the user
    /// fainted as part of the move, `useMoveInner` battle-actions.ts:501-503). Read into the
    /// per-decision `DecisionRecord.explosion_self_ko` at the turn's boundary + cleared at the
    /// top of each turn — a diagnostic/coverage signal ONLY (the self-KO itself is applied via
    /// the normal faint machinery, so this flag does not affect any draw or state). Lets the
    /// e2e capstone COUNT explosion decisions (the mechanic is momentary — no persistent state
    /// like a substitute — so it can't be read off the post-turn board otherwise).
    pub pending_explosion_self_ko: bool,
    /// The RESOLVED turn-0 `runSwitch` dequeue ORDER (side indices, first-to-run first) —
    /// recorded by [`BattleState::run_turn0_construction`] from the queue `insert_runswitch`
    /// actually built (`gen3_turn0_construction_mirror_order_v1`).
    ///
    /// **Why it must be RECORDED, not re-derived.** The bridge runs the construction window
    /// with logging OFF and re-emits the leads' switch-in ability lines afterwards from the
    /// post-construction board (`emit_switchin_ability_lines`). That reconstruction used to
    /// re-derive the fire order as "faster raw Speed first, a TIE keeps side order" — the model
    /// of the DRAW-FREE [`BattleState::run_start_switchins`] path. But under
    /// `gen3_turn0_construction_v1` the construction runs the REAL `insertChoice`, whose
    /// speed-TIE window is broken by a `random(firstIndex, lastIndex+1)` PRNG draw, so at a tie
    /// the true order is p2-FIRST half the time. Re-deriving therefore emitted the Intimidate /
    /// weather-setter block in the WRONG order on a speed-tied lead even though the PRNG stream
    /// and the resulting BOARD were bit-for-bit correct — the divergence the external-consistency
    /// gate caught as the Masquerain-mirror repro `sbd_msdd8698_b293` (node `-ability p2a` first,
    /// the port `p1a` first, with all 5 turn-0 draws IDENTICAL).
    ///
    /// `None` on the draw-free [`BattleState::start_with_switchins`] path (the committed
    /// goldens' POST-construction seed convention), where the emitter keeps the
    /// side-order-on-tie fallback that path's STATE half also uses — so those goldens stay
    /// byte-identical. Observation-only: written from an already-resolved queue, read only by
    /// the emitter; consumes no PRNG and changes no board state.
    pub turn0_switchin_order: Option<Vec<usize>>,
    /// Whether a PHAZE (Roar / Whirlwind) drag actually FIRED this turn (`drag_in` ran its
    /// `sample`), analogous to `pending_explosion_self_ko`. Read into the per-decision
    /// `DecisionRecord.phaze_drag` at the turn's boundary + cleared at the top of each turn — a
    /// diagnostic/coverage signal ONLY (the drag itself is applied via the normal switch
    /// machinery, so this flag does not affect any draw or state). Lets the e2e capstone COUNT
    /// phaze-drag decisions (the mechanic is momentary — the dragged mon is just the new active,
    /// indistinguishable from a voluntary switch on the post-turn board — so it can't be read off
    /// the snapshot otherwise). A Protect-blocked / no-eligible-bench phaze does NOT set it (no
    /// drag), which is exactly what we want to confirm the drag path is genuinely exercised.
    pub pending_phaze_drag: bool,
    /// The PROTOCOL-EMISSION line buffer (`gen3_protocol_emission_v1`, level-2). The engine
    /// pushes `|...|` lines into it at each observable event (`run_full_battle_logged` enables
    /// it + emits the framing; the plain `run_full_battle` leaves it DISABLED so the seed suite
    /// pays zero cost and draws nothing). **Observation-only:** every push formats an
    /// already-computed value and consumes NO PRNG, so wiring it does NOT perturb any seed
    /// assertion (the load-bearing guarantee — see `protocol.rs`). Drained by the caller after
    /// the battle (or per decision, for a streaming bridge).
    pub log: crate::protocol::ProtocolBuilder,
    /// The `faintQueue` ENQUEUE-ORDER queue: the sides whose active hit 0 HP THIS
    /// action, in the order they were zeroed. DRAW-BEARING since fix-queue #4
    /// (`gen3_faint_queue_order_v1`): `process_faints` drains it to process corpses
    /// in Showdown's `faintMessages` order — each corpse's `fainted` flag is set
    /// BEFORE the next corpse's ability-`End`, so a double faint's second Cloud Nine
    /// End WeatherChange gathers alone (no tie draw). It ALSO drives the `|faint|`
    /// emission order (`gen3_protocol_emission_v1` — the self-KO'd Explosion user
    /// BEFORE its KO'd target, verified vs the golden). Pushed UNCONDITIONALLY by
    /// `apply_damage` (+ the explosion self-KO); drained by `process_faints`.
    pub faint_emit_queue: Vec<usize>,
    /// The **PURSUIT INTERRUPT** transient flag (`gen3_move_coverage_batch4_v1`): set true by
    /// `execute_switch` for the duration of the ONE `run_move` call that resolves the pursuit
    /// strike against a switching mon, then read+cleared at the top of `run_move`. When set,
    /// `run_move` runs the move NEVER-MISS + at ×2 BP (the `basePowerCallback` / `onModifyMove`
    /// interrupt condition) and SKIPS the `on_before_move` status draws + PP deduction +
    /// lastMove (all handled manually by the interrupt in `execute_switch`, mirroring the sim's
    /// bare `useMove`). Never set during a normal move; not persisted (a transient like
    /// `pending_explosion_self_ko` but read+cleared, not per-turn).
    pub pursuit_strike: bool,
    /// TRANSIENT: set only around the YAWN residual's `trySetStatus('slp')` call
    /// (`gen3_safeguard_v1`). gen-3 safeguard's `onSetStatus` opens with
    /// `if (effect.id === 'yawn') return;` — the exemption keys on the EFFECT, not the
    /// source — so a PENDING Yawn sleeps its target THROUGH an active Safeguard (drawing
    /// its `random(2,6)` normally) even though a Yawn CAST into that same ward is blocked.
    /// The `pursuit_strike` / `sleep_talk_call` pattern: never persisted, never observed.
    pub yawn_resolving: bool,
    /// The PER-TURN **first-mover override** for a PURSUIT-INTERRUPT turn
    /// (`gen3_move_coverage_batch4_v1`). On a turn where a mon VOLUNTARILY switches out and the
    /// foe's Pursuit interrupts it, the sim emits the pursuer's `|move|Pursuit` line FIRST (from
    /// the `onBeforeSwitchOut` strike, at the TOP of the switch's runAction — before the
    /// `|switch|` line), so `firstMoverSince` reads the PURSUER as the first mover. The port's
    /// default `first_mover` is the first SORTED-queue action = the switch (order 103 < move 200)
    /// = the switching side — WRONG for this turn. `execute_switch` sets this to the pursuer's
    /// side (`pside`) when the strike fires; `boundary_record` prefers it over the sorted-queue
    /// `first_mover`. Reset to `None` at the top of each turn (like `pending_phaze_drag`). The
    /// pursuer genuinely acts first, so this is a faithful attribution, not a cosmetic patch.
    pub pursuit_first_mover: Option<usize>,
    /// The **SLEEP TALK CALLED-MOVE** transient flag (`gen3_move_coverage_batch5_v1`): set
    /// true by `run_status_move`'s Sleep Talk arm for the duration of the ONE recursive
    /// `run_move` call that executes the SAMPLED move (the sim's `actions.useMove(picked)`
    /// inside `sleeptalk.onHit`), then read+cleared at the top of `run_move`. When set,
    /// `run_move` SKIPS `on_before_move` (the slp handler already ran + proceeded via
    /// `sleepUsable`) + PP deduction (the picked move's PP is NEVER consumed — only Sleep
    /// Talk's own paid) + lastMove (the sim's `moveUsed` fires only in `runMove`, so
    /// `last_move` stays the Sleep Talk slot) + the mustrecharge gate. The called move
    /// otherwise runs its FULL NORMAL draw chain (acc/crit/damage/secondary — probed).
    /// The `pursuit_strike` transient pattern.
    pub sleep_talk_call: bool,
    /// The battle-global **Quick Claw roll** (`Battle.quickClawRoll`, sim battle.js:1485
    /// `gen === 3` ⇒ `randomChance(1, 5)`, drawn UNCONDITIONALLY at every completed `endTurn`).
    /// Read next turn by gen3 `getActionSpeed` (scripts.js:47-48): a Quick-Claw HOLDER whose
    /// roll hit TRUE gets `speed = 65535` (moves first WITHIN its priority bracket, regardless
    /// of raw Speed). Persisted here so `effective_speed` can apply the override — the port
    /// previously drew-and-DISCARDED the endTurn roll (turn.rs), so it ordered purely on raw
    /// speed and mis-ordered / mis-damaged (the swapped damage rolls) every QC-proc turn
    /// (`gen3_quick_claw_speed_v1` — the P1 state-HP off-by-one + wrongful-faint AND the P2
    /// wrong-first-mover class, one fix). Init false; a fresh construct has no prior endTurn.
    pub quick_claw_roll: bool,
}

impl BattleState {
    /// Construct the battle state from `>start` options + the two packed teams.
    ///
    /// Steps (mirroring Showdown's construction path, NOT its start action):
    /// 1. unpack both teams ([`crate::team::unpack`]);
    /// 2. per mon, [`compute_stats`] and set `hp == maxhp == stats[0]`, status
    ///    empty, boosts 0, not fainted, `position = team index`;
    /// 3. record the gen-3-singles lead (`active = 0`);
    /// 4. build the PRNG from `opts.seed` (a `[m,n,o,p]` JSON-array seed is
    ///    accepted as the comma-string the gen5 backend parses); `turn = 0`;
    ///    `field.weather = None`.
    ///
    /// Does NOT run any switch-in event (no Intimidate, no Sand Stream weather, no
    /// ability `Start`); those are the event engine's job (a later step).
    ///
    /// Crash-don't-drop: returns `Err` on any unpack/stat failure rather than a
    /// silently mis-constructed board.
    pub fn start(opts: &BattleOptions, dex: &Dex) -> Result<BattleState, String> {
        let gen = dex.generation();
        let p1 = build_side(&opts.p1.name, &opts.p1.team, dex)?;
        let p2 = build_side(&opts.p2.name, &opts.p2.team, dex)?;

        // PRNG: the start seed is `Option<PrngSeed>` (a string). A `None` seed
        // means "generate one" upstream; here we have no event that draws, so a
        // deterministic default keeps construction total. If a seed is given as a
        // bare `[m,n,o,p]` JSON array (the harness/`>start` form), normalize it to
        // the `m,n,o,p` comma string the gen5 backend parses.
        let seed = match &opts.seed {
            Some(s) => normalize_seed(s),
            None => DEFAULT_CONSTRUCT_SEED.to_string(),
        };
        let prng = Prng::new(&seed);

        Ok(BattleState {
            prng,
            turn: 0,
            sides: [p1, p2],
            field: Field::default(),
            gen,
            sleep_clause: format_has_sleep_clause(&opts.format_id),
            format_id: opts.format_id.clone(),
            pending_explosion_self_ko: false,
            turn0_switchin_order: None,
            pending_phaze_drag: false,
            log: crate::protocol::ProtocolBuilder::new(),
            faint_emit_queue: Vec::new(),
            pursuit_strike: false,
            yawn_resolving: false,
            sleep_talk_call: false,
            pursuit_first_mover: None,
            quick_claw_roll: false,
        })
    }

    /// Construct the battle state AND run the `>start` switch-in event sequence
    /// (both leads switch in; their switch-in ability events fire — Intimidate
    /// Atk drop, Sand Stream / Drizzle / Drought weather).
    ///
    /// This is [`BattleState::start`] (construction only) followed by
    /// [`BattleState::run_start_switchins`] (the event half). It is the entry
    /// point a caller uses when it wants the POST-switch-in board (boosts +
    /// weather), as opposed to the pristine construction board.
    ///
    /// `turn` stays 0 here: the start action that sets turn 1 (and the turn-loop
    /// Quick Claw roll) is a later step; this wires only the switch-in events.
    pub fn start_with_switchins(opts: &BattleOptions, dex: &Dex) -> Result<BattleState, String> {
        let mut state = BattleState::start(opts, dex)?;
        state.run_start_switchins();
        // FORECAST at the START window (`gen3_forecast_v1`): the sim's construction fires
        // each Castform lead's `forecast.onStart` (off the standing weather at its own
        // runSwitch) AND the weather-setter's `eachEvent('WeatherChange')` over both leads —
        // both CONVERGE on "forme = f(final start weather)", so the DRAW-FREE path applies
        // it post-hoc after all Starts (probe E1: a Castform lead vs a Drought lead is
        // `castformsunny` at `|turn|1`; vs Sand Stream stays base — the default arm). STATE
        // only here — the framing `|-formechange|` line is `emit_switchin_ability_lines`'.
        let eff = state.effective_weather(dex);
        for side in 0..2 {
            state.forecast_weather_change(side, eff, dex);
        }
        // WHITE HERB (`gen3_white_herb_v1`, `whiteherb.onAnySwitchIn` priority −2): a LEAD whose
        // Atk was dropped by the opposing lead's Intimidate restores it + consumes the item —
        // but ONLY when the holder's own switch-in fires AFTER that drop. White Herb's onAnySwitchIn
        // runs at EACH lead's `runEvent('SwitchIn')`, which PRECEDES that lead's ability `Start`
        // (Intimidate). So a holder restores at construction iff its runSwitch is processed AFTER
        // the Intimidate foe's — i.e. the holder is SLOWER (the two leads' runSwitch dequeue
        // faster-first; a tie keeps side order, side 0 before side 1). A FASTER holder's switch-in
        // precedes the foe's Intimidate Start → no drop yet → NOT restored at construction; it
        // restores at turn-1's order-29 residual (`ResidualAction::WhiteHerb`) instead. The
        // slower-holder case emits its `-enditem`/`-clearnegativeboost` in the framing (see
        // `emit_ability_start_lines`). Probe-settled bit-for-bit vs the sim (the omniscient byte-fuzz
        // finds ab_6_13 [lead, restore before |turn|1] + ab_7_4 [mid-battle, restore at the residual]).
        // DRAW-FREE (the seed convention is unaffected); a no-op unless a lead holds White Herb AND
        // was Intimidate-dropped by a FASTER foe, so all existing tests are untouched.
        for side in 0..state.sides.len() {
            let slot = state.sides[side].active;
            let foe = 1 - side;
            let foe_slot = state.sides[foe].active;
            let holder_spe = state.sides[side].pokemon[slot].stats[5];
            let foe_spe = state.sides[foe].pokemon[foe_slot].stats[5];
            // The holder's runSwitch resolves AFTER the foe's iff it is SLOWER (faster-first; a
            // raw-Speed tie keeps side order → side 0 before side 1). `white_herb_restore` no-ops
            // unless the holder actually has a negative boost (i.e. was Intimidate-dropped).
            let holder_switch_in_after_foe =
                holder_spe < foe_spe || (holder_spe == foe_spe && side > foe);
            if holder_switch_in_after_foe {
                state.white_herb_restore(side, slot, dex);
            }
        }
        Ok(state)
    }

    /// Construct from a RAW `>start` seed and run the FULL turn-0 CONSTRUCTION WINDOW
    /// (`gen3_turn0_construction_v1`) — the gender samples + the two `runSwitch`
    /// actions + `endTurn`, reproducing every pre-first-decision PRNG draw bit-for-bit
    /// (see [`BattleState::run_turn0_construction`]). This is the BRIDGE's construction
    /// entry (`sim_bridge` feeds poke-env's raw `>start` seed); it REPLACES the draw-
    /// free [`start_with_switchins`] + the old pure `advance_seed_for_construction` seed
    /// hack, so a speed-TIED lead or an unspecified-gender mon lines up with the real
    /// sim's post-`>start` state. The committed seed goldens (which capture the sim's
    /// POST-construction `initSeed`) keep [`start_with_switchins`] — this is opt-in.
    ///
    /// The `opts.seed` is the RAW seed (NOT pre-advanced). Runs the construction with
    /// the log DISABLED (its ctor default), so the ability/weather emissions are
    /// suppressed here — the bridge re-emits the framing from the post-construction
    /// state afterwards.
    pub fn start_with_turn0_construction(opts: &BattleOptions, dex: &Dex) -> Result<BattleState, String> {
        let mut state = BattleState::start(opts, dex)?;
        state.run_turn0_construction(dex);
        Ok(state)
    }

    /// Read a side by index (0 = p1, 1 = p2).
    pub fn side(&self, i: usize) -> &SideState {
        &self.sides[i]
    }

    /// The current PRNG seed string (`PRNG.getSeed()`). Lets a test confirm the
    /// RNG-consumption count of a phase (e.g. the switch-in dispatch draws only
    /// on a speed tie) without reaching into the `prng` field.
    pub fn prng_seed(&self) -> PrngSeed {
        self.prng.get_seed()
    }
}

/// A construction-time default seed for an `opts.seed == None` construct — a
/// TEST/HARNESS convenience only.
///
/// `BattleState::start` itself draws no dice, but the turn-0 construction window
/// ([`crate::bridge::BridgeSession::new_construct_turn0`]) and every later turn DO, so a
/// `None` seed here means "this whole battle runs on `0,0,0,0`". That is fine for an
/// in-crate test that wants a fixed board and FATAL for a production caller: it makes
/// every seedless battle replay ONE dice stream
/// (`gen3_bridge_seedless_fixed_seed_v1`). The bridge binary therefore never passes
/// `None` — it mints a fresh [`Prng::generate_seed`] exactly like Showdown's `PRNG`
/// constructor does, so the resolved seed is real AND recordable.
const DEFAULT_CONSTRUCT_SEED: &str = "0,0,0,0";

/// Whether the format `format_id` carries the **Sleep Clause Mod** (a sleep move that
/// would inflict a 2nd foe sleep FAILS). Ladder tiers include it via the `Standard`
/// ruleset (Ubers/OU/UU/…); `gen3customgame` (and other `*customgame`) carry NO
/// clauses. We match on the format SHAPE: anything `gen3customgame` (or any
/// `*customgame`) → no clause; everything else gen-3 (gen3ou/uu/ubers/…) → clause.
/// Gen-generic: only `*customgame` is the clause-free family.
pub fn format_has_sleep_clause(format_id: &str) -> bool {
    let id: String = format_id.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect();
    !id.ends_with("customgame")
}

/// Unpack one packed team and build its [`SideState`] (per-mon construction
/// state, lead = slot 0).
fn build_side(name: &str, team: &PackedTeam, dex: &Dex) -> Result<SideState, String> {
    let sets = unpack(&team.0, dex).map_err(|e| format!("side {name:?}: team unpack failed: {e}"))?;
    if sets.is_empty() {
        return Err(format!("side {name:?}: empty team"));
    }
    let mut pokemon = Vec::with_capacity(sets.len());
    for (i, set) in sets.into_iter().enumerate() {
        pokemon.push(MonState::from_set(set, i, dex)?);
    }
    let pokemon_left = pokemon.len();
    Ok(SideState {
        name: name.to_string(),
        pokemon,
        active: 0, // gen-3 singles lead = pokemon[0]
        pokemon_left,
        switch_flag: false,
        spikes: 0,        // no side conditions at construction
        light_screen: 0,  // gen3_move_coverage_batch2_v1
        reflect: 0,       // gen3_move_coverage_batch2_v1
        safeguard: 0,     // gen3_safeguard_v1
        wish_pending: None,         // gen3_move_coverage_batch3_v1
        baton_pass_pending: false,  // gen3_move_coverage_batch3_v1
        future_move: None,          // gen3_move_coverage_batch4c_v1
    })
}
