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
use crate::prng::{Prng, PrngSeed};
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
    pub species_id: String,
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
        let set_gender = set.gender; // read before `set` is moved into the struct
        Ok(MonState {
            set,
            item,
            item_knocked_off: false,
            ability,
            focus_energy: false,
            species_id,
            level,
            stats,
            hp: maxhp,
            maxhp,
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
            taunt: None,
            disable: None,
            last_move: None,
            move_pp,
            move_maxpp,
            choice_locked_move: None,
            flash_fire: false,
            truant_turn: false,
            gender: set_gender,
            attract: None,
            types_override: None,
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
        })
    }

    /// The current PP of move slot `k` (`gen3_pp_tracking_v1`), or `0` for an
    /// out-of-range slot (a phantom slot is never "usable").
    pub fn pp_of(&self, k: usize) -> u16 {
        self.move_pp.get(k).copied().unwrap_or(0)
    }

    /// Whether move slot `k` is USABLE now — an in-range slot with >0 PP that is NOT disabled
    /// by any move-SELECTION restriction (`gen3_pp_tracking_v1` + `gen3_taunt_disable_v1`):
    ///   - the **Choice lock** (`choicelock.onDisableMove`): if the mon is choice-locked to slot
    ///     `j` (`choice_locked_move == Some(j)`), only slot `j` is usable;
    ///   - **DISABLE** (`disable.onDisableMove`): the one recorded `disable` slot is un-usable;
    ///   - **TAUNT** (`taunt.onDisableMove`): EVERY Status-category slot is un-usable (needs the
    ///     dex to read the slot move's category).
    /// A slot un-usable by ANY of these can't be SELECTED (`side.choose` rejects it) and, if ALL
    /// slots are un-usable, the mon is forced to Struggle. The `dex` is read only for Taunt's
    /// per-slot category; the other gates are pure state.
    pub fn move_usable(&self, k: usize, dex: &Dex) -> bool {
        if let Some(locked) = self.choice_locked_move {
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
/// field. This is the deterministic core the event engine will mutate; THIS step
/// only constructs it (no events run).
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
            pending_phaze_drag: false,
            log: crate::protocol::ProtocolBuilder::new(),
            faint_emit_queue: Vec::new(),
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

/// A construction-time default seed used only when no seed is supplied. Pure
/// construction draws no dice, so the specific value never affects this step;
/// it just keeps [`BattleState::start`] total. (The gen5 decimal backend.)
const DEFAULT_CONSTRUCT_SEED: &str = "0,0,0,0";

/// Normalize a `>start` seed into a [`Prng::new`]-acceptable string. Accepts:
/// - a bracketed JSON array `[1,2,3,4]` (the form `>start {"seed":[..]}` carries)
///   → `1,2,3,4`;
/// - any already-valid seed string (`sodium,…`, `gen5,…`, `1,2,3,4`) → verbatim.
fn normalize_seed(s: &str) -> String {
    let t = s.trim();
    if let Some(inner) = t.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        // `[1, 2, 3, 4]` → `1,2,3,4`
        inner
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(",")
    } else {
        t.to_string()
    }
}

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
        wish_pending: None,         // gen3_move_coverage_batch3_v1
        baton_pass_pending: false,  // gen3_move_coverage_batch3_v1
    })
}
