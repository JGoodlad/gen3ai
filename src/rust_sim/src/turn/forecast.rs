//! FORECAST (Castform) — the forme + TYPE swap that follows the EFFECTIVE weather
//! (`gen3_forecast_v1`, ROUND 35). The last fail-loud coverage gap in the port.
//!
//! # The resolved gen-3 mechanic
//!
//! gen 3 inherits `forecast` from the base `data/abilities.ts` (the gen4 mod overrides only
//! the flags), so the live handler is:
//!
//! ```js
//! onSwitchInPriority: -2,
//! onStart(pokemon) { this.singleEvent('WeatherChange', this.effect, this.effectState, pokemon); },
//! onWeatherChange(pokemon) {
//!   if (pokemon.baseSpecies.baseSpecies !== 'Castform' || pokemon.transformed) return;
//!   let forme = null;
//!   switch (pokemon.effectiveWeather()) {
//!     case 'sunnyday': if (pokemon.species.id !== 'castformsunny') forme = 'Castform-Sunny'; break;
//!     case 'raindance': if (pokemon.species.id !== 'castformrainy') forme = 'Castform-Rainy'; break;
//!     case 'hail':      if (pokemon.species.id !== 'castformsnowy') forme = 'Castform-Snowy'; break;
//!     default:          if (pokemon.species.id !== 'castform')      forme = 'Castform';       break;
//!   }
//!   if (pokemon.isActive && forme) pokemon.formeChange(forme, this.effect, false, '0', '[msg]');
//! }
//! ```
//!
//! Four things fall out, each byte-probed rather than reasoned
//! (`harness/probe_r35_forecast_{reporting,order,edges,ties}.js`):
//!
//! * **SAND maps to the DEFAULT arm** — a Castform under sandstorm is plain Normal
//!   `castform`, NOT a fourth forme (probe S4).
//! * **`effectiveWeather()`, not `field.weather`** — so Cloud Nine / Air Lock compose for
//!   free through the port's existing [`BattleState::effective_weather`]. Under a
//!   suppressor a Castform stays Normal even in rain (probe O1c/T1).
//! * **The no-op transition is SILENT** — `forme` stays `null` when the target forme is
//!   already current, so a sand→sand upkeep or a rain re-set emits NOTHING (probe S4).
//! * **DRAW-FREE.** Every `-formechange` in every trace fired at the baseline draw count;
//!   a Forecast-vs-control A/B over 6 weather transitions matched per-turn draw-for-draw
//!   (probe S6/O4). Forecast registers NO handler at any sorted event — its `onWeatherChange`
//!   rides the `eachEvent('WeatherChange')` the weather change ALREADY fires, whose
//!   `speedSort(getAllActive())` tie-shuffle the port has modeled since
//!   `gen3_ability_batch1_v1`. **The ROUND 32 lesson (a handler tie that "can never happen")
//!   was checked explicitly and does not apply here: Forecast adds no handler to sort.**
//!
//! # The REPORTING surface (the reason this sat deferred for months)
//!
//! `formeChange` is called with `isPermanent = false`, and that branch of
//! `sim/pokemon.ts::formeChange` does NOT touch `Pokemon.details` / `Pokemon.name` /
//! `Pokemon.baseSpecies` — it only calls `setSpecies` (stats + TYPE) and emits ONE line.
//! So the forme name appears on the wire in **exactly one place**, the `|-formechange|`
//! line itself. Probed end to end:
//!
//! | surface | a FORMED Castform reports |
//! |---|---|
//! | protocol ident (`\|move\|`, `\|-damage\|`, `\|-formechange\|` …) | `p1a: Castform` (BASE) |
//! | `\|switch\|` / `\|drag\|` details | `Castform, F` (BASE — and moot: `clearVolatile` reverts the forme on the way OUT, so a Castform is never formed at its own switch line) |
//! | per-side `\|request\|` `side.pokemon[].details` | `Castform, F` (BASE) |
//! | `\|-formechange\|` | `Castform-Rainy` (the FORME — the only forme byte) |
//!
//! There is therefore **no forme-reporting plumbing to take on**: the port's ident /
//! details / request surfaces just have to read the CONSTRUCTION species
//! ([`crate::state::MonState::base_species_id`]) instead of the live one, which is also
//! what `pokemon.baseSpecies` means and what the Transform overlay already did.
//!
//! ⚠️ **Observation-side consequence (owner decision, flagged not taken):** poke-env parses
//! `|-formechange|` and will re-key the mon's species to `castformrainy` for the rest of the
//! turn-block. That is IDENTICAL to what `--use-bridge=node` already emits — the rust bridge
//! introduces no new obs behaviour — and `data/pokemon/gen3_species.json` already carries the
//! three `battleOnly` forme rows (kept out of the num-indexed model buffers by
//! `base_form_ids()`). See `src/rust_sim/CLAUDE.md` → ROUND 35.
//!
//! # Where it fires
//!
//! `onWeatherChange` is only ever reached two ways, and BOTH are already modeled control
//! flow in the port — this module adds no new event, only an effect at existing sites:
//!
//! 1. `this.eachEvent('WeatherChange')` — fired by `Field.setWeather` (a weather MOVE, or a
//!    Drizzle/Drought/Sand-Stream switch-in), by `Field.clearWeather` (the timed-weather
//!    EXPIRY), and by a WEATHER_NEGATE ability's `onEnd` (a Cloud Nine / Air Lock holder
//!    switching out or fainting). All four already call
//!    [`BattleState::each_event_shuffle`]; this module's [`BattleState::forecast_each_event`]
//!    consumes that call's RESOLVED order so the two `-formechange` lines of a
//!    Castform-vs-Castform mirror emit in the shuffled permutation (probe T3 — the order
//!    flips with the seed).
//! 2. `forecast.onStart` → `singleEvent('WeatherChange')` — the Castform's OWN switch-in,
//!    which formes it off the STANDING weather. A singleEvent: one mon, no sort, no draw.
//!
//! And the **asymmetry that a plain reading misses**: gen 4 (inherited by gen 3) overrides
//! Cloud Nine / Air Lock with `onSwitchIn: undefined` + an `onStart` that only clears the
//! `ending` flag — so a suppressor switching IN fires NO WeatherChange, and a Castform that
//! is already Rainy STAYS Rainy while the suppressor sits there. Only the suppressor's
//! (inherited) `onEnd` fires the event. Probe T4 t3/t4: leaving → `-formechange`; entering →
//! nothing.

use crate::dex::{to_id, Dex};
use crate::state::{BattleState, Weather};

/// The Castform forme id `forecast.onWeatherChange` maps an EFFECTIVE weather to. SAND
/// (and no weather) hit the `default:` arm → the base `castform` (probe S4).
pub(crate) fn forecast_forme_id(eff: Option<Weather>) -> &'static str {
    match eff {
        Some(Weather::Sun) => "castformsunny",
        Some(Weather::Rain) => "castformrainy",
        Some(Weather::Hail) => "castformsnowy",
        Some(Weather::Sand) | None => "castform",
    }
}

impl BattleState {
    /// `forecast.onWeatherChange(pokemon)` for ONE side's ACTIVE mon — the whole handler,
    /// gates included. Applies the forme (species + type + the `setSpecies` speed write) and
    /// emits the `|-formechange|` line. Returns `true` iff a forme change actually happened
    /// (a no-op transition is silent and returns `false`). DRAW-FREE.
    ///
    /// The gates, in the source's order:
    /// * `pokemon.baseSpecies.baseSpecies !== 'Castform'` — keyed on the CONSTRUCTION
    ///   species, so a non-Castform that ACQUIRES Forecast (Role Play / Skill Swap / a Trace
    ///   of it) never formes (probe T6: an Alakazam holding `forecast` sits out a rain set).
    /// * `pokemon.transformed` — a Ditto that copied a Rainy Castform keeps `castformrainy`
    ///   FOREVER, including through the rain expiring (probe E7 ditto-t5), and a Castform
    ///   that itself Transformed stops tracking the weather.
    /// * the ability must currently BE `forecast`: a Castform whose Forecast is Skill-Swapped
    ///   away while Rainy stays Rainy for the rest of its stay (probe T5 — the handler is
    ///   simply no longer registered).
    /// * `pokemon.isActive` — mirrored as "the caller passed the active slot, and it is not
    ///   fainted", matching `eachEvent`'s `getAllActive()` filter (the same filter
    ///   [`BattleState::each_event_shuffle`] applies).
    /// `eff` is the EFFECTIVE weather AS THE SIM WOULD SEE IT AT THIS EVENT — usually just
    /// [`BattleState::effective_weather`], but the two WEATHER_NEGATE `onEnd` sites must
    /// pass [`BattleState::effective_weather_excluding`] the LEAVING/FAINTING suppressor:
    /// gen4's `cloudnine.onEnd` sets `abilityState.ending = true` BEFORE firing the
    /// WeatherChange, and `suppressingWeather()` skips an `ending` holder — which is how the
    /// sim formes a Castform Rainy while the departing Psyduck is still in its slot
    /// (probe O1c t2: the `-formechange` precedes the incoming `|switch|` line).
    pub(crate) fn forecast_weather_change(
        &mut self,
        side: usize,
        eff: Option<Weather>,
        dex: &Dex,
    ) -> bool {
        let slot = self.sides[side].active;
        let target = forecast_forme_id(eff);
        {
            let mon = &self.sides[side].pokemon[slot];
            if mon.fainted {
                return false;
            }
            if to_id(&mon.ability) != "forecast" {
                return false;
            }
            if mon.transform.is_some() {
                return false;
            }
            if to_id(&mon.base_species_id) != "castform" {
                return false;
            }
            if mon.species_id == target {
                return false; // `forme` stays null → silent no-op
            }
        }
        let name = dex
            .species(target)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| target.to_string());
        {
            let mon = &mut self.sides[side].pokemon[slot];
            // `formeChange` → `setSpecies(rawSpecies)`: species + `setType(species.types,
            // true)`. The port's ONE type choke point is `helpers::mon_types`, which reads
            // `types_override` else the species types — so writing `species_id` IS the type
            // change, and the OVERRIDE must be dropped (setType with `isBase = true`
            // overwrites whatever Color Change / Conversion had installed).
            mon.species_id = target.to_string();
            mon.types_override = None;
            // `setSpecies` ends with `this.speed = this.storedStats.spe` — the RAW stat, so a
            // mid-turn forme change DISCARDS the para/boost-folded `pokemon.speed` until the
            // next `updateSpeed()`. Mirrored because `eachEvent`/`residualEvent` sort on that
            // very field (`comparePriority` reads `.speed`), so it can flip a later same-turn
            // tie. The five stored stats themselves are IDENTICAL across the four Castform
            // formes (all 70 base), so nothing else moves.
            mon.cached_speed = mon.stats[5] as u32;
        }
        if self.logging() {
            let r = self.mon_ref(side, slot, dex);
            self.log.forme_change_forecast(&r, &name);
        }
        true
    }

    /// The FORECAST half of an `eachEvent('WeatherChange')`, run over the actives in the
    /// order [`BattleState::each_event_shuffle`] RESOLVED (its return value) — so a
    /// Castform-vs-Castform mirror emits its two `-formechange` lines in the shuffle's
    /// permutation, not a fixed side walk (probe T3: seed `[1,2,3,4]` → p1 then p2, seed
    /// `[5,5,5,5]` → p2 then p1). DRAW-FREE: the shuffle IS the draw, already made by the
    /// caller.
    ///
    /// Call it at EVERY `eachEvent('WeatherChange')` site, immediately after the shuffle,
    /// passing the site-correct `eff` (see [`BattleState::forecast_weather_change`]).
    pub(crate) fn forecast_each_event(&mut self, order: &[usize], eff: Option<Weather>, dex: &Dex) {
        for &side in order {
            self.forecast_weather_change(side, eff, dex);
        }
    }
}
