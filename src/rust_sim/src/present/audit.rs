//! [`check_view`] — the TRUTH AUDIT of a stream-built [`OneSidedView`] against the omniscient
//! board: every SIM-FACT field the view presents, checked against the referee.
//!
//! The board is a REFEREE here, never an input: [`super::present`] cannot see it (its signature
//! has no board), and this is the one function that may compare the two. Where the reading is
//! not the sim's truth BY A NAMED RULE — an information limit or a presentation, never a
//! reproduced poke-env mistake — the check applies the rule; it does not skip the field:
//!
//! * V9  — an opponent's HP is the `ceil%` fold (`current_hp` = the percent, `max_hp` = 100), its
//!   stats / spread unknown;
//! * V14 — a TRANSFORMED mon's moves / ability are poke-env's overlay (the target's KNOWN moves at
//!   5 PP, its KNOWN ability): its moves must be a subset of the engine's and are not PP-checked,
//!   its ability is not checked (R4 — the stream never states the copied ability);
//! * V15 — an own mon the CURRENT request did not re-sync holds the SIGHTING count of its PP,
//!   which can only lag the sim's (a foe's Pressure is announced to its owner only).
//!
//! The view's TRUE readings are plain checks: a fainted mon holds no stages (PE-V10), an ACTIVE
//! badly-poisoned mon's counter is the engine's stage (PE-R1b), Flash Fire is present exactly when
//! the engine holds `flashfire` (PE-V16). A BENCHED badly-poisoned mon's counter is UNRESOLVED
//! (the sim stores the stage it left with and resets it on switch-in; the view reads 0, as
//! poke-env does): counted in [`Audit::rules_fired`], never checked.
//!
//! Plus the revealed-opponent truths no projection can make (the Python slice V's `truth_checks`,
//! moved here so the board has ONE reader): a disclosed item / ability / move is the engine's, a
//! consumed item is no longer held, a living mon's types are the engine's, and the ten sim-state
//! volatiles are present on the view exactly when the engine holds them.

use super::dex;
use super::view::{MonView, OneSidedView, SideView};
use crate::dex::Dex;
use crate::state::{BattleState, MonState, Status, Weather};

/// Engine volatile (`search::volatile_names`) → the `LivePokemon.volatiles` key. Two-way on this
/// set: the view holds the key iff the engine holds the volatile. Only conditions the protocol
/// ANNOUNCES belong here (a condition the sim never prints has no reading to check).
pub const TRUTH_VOLATILES: [(&str, &str); 11] = [
    ("attract", "attract"),
    ("confusion", "confusion"),
    ("curse", "curse"),
    ("disable", "disable"),
    ("encore", "encore"),
    ("flashfire", "flashfire"),
    ("focusenergy", "focusenergy"),
    ("leechseed", "leechseed"),
    ("substitute", "substitute"),
    ("taunt", "taunt"),
    ("yawn", "yawn"),
];

/// The audit of one view: how many facts were checked, and every one that failed as
/// `(class, detail)` — the class names the field and the rule, the detail the two values.
#[derive(Debug, Default, Clone)]
pub struct Audit {
    pub checks: usize,
    pub divergences: Vec<(String, String)>,
    /// Per named rule (V15), and per UNRESOLVED question (`UNRESOLVED:tox-stage-benched`), how
    /// many checked facts the view held differently from the engine, legally. A census, never a
    /// divergence: it is how a reader sees how often each shapes the obs.
    pub rules_fired: std::collections::BTreeMap<&'static str, usize>,
}

impl Audit {
    fn check(&mut self, ok: bool, class: &str, detail: impl FnOnce() -> String) {
        self.checks += 1;
        if !ok {
            self.divergences.push((class.to_string(), detail()));
        }
    }
    /// JSON: `{"checks":n,"divergences":[[class,detail],…]}`.
    pub fn json(&self) -> String {
        use crate::core_events::json_out::str_into;
        let mut o = format!("{{\"checks\":{},\"rules_fired\":{{", self.checks);
        for (i, (r, n)) in self.rules_fired.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            str_into(&mut o, r);
            o.push_str(&format!(":{n}"));
        }
        o.push_str("},\"divergences\":[");
        for (i, (c, d)) in self.divergences.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push('[');
            str_into(&mut o, c);
            o.push(',');
            str_into(&mut o, d);
            o.push(']');
        }
        o.push_str("]}");
        o
    }
}

/// The sim's status token as `LiveView` spells it (a fainted mon reads `fnt` wherever it sits).
fn status_live(m: &MonState) -> Option<&'static str> {
    if m.fainted {
        return Some("fnt");
    }
    m.status.map(|s| match s {
        Status::Burn => "brn",
        Status::Paralysis => "par",
        Status::Sleep(_) => "slp",
        Status::Freeze => "frz",
        Status::Poison => "psn",
        Status::Toxic(_) => "tox",
    })
}

/// `ceil(100 * hp / maxhp)`, a full-looking bar with `hp < maxhp` showing 99 — the gen3ou wire fold.
fn hp_percent(hp: u16, maxhp: u16) -> u32 {
    crate::core_events::side::hp_percent(hp as u32, maxhp as u32)
}

/// The mon's CURRENT types (a Conversion / Forecast override wins over the dex row), lower-cased.
fn engine_types(m: &MonState, dex: &Dex) -> Vec<String> {
    let ts = match &m.types_override {
        Some(t) => t.clone(),
        None => dex.species(&m.species_id).map(|s| s.types.clone()).unwrap_or_default(),
    };
    ts.iter().map(|t| format!("{t:?}").to_ascii_lowercase()).collect()
}

/// Which engine mon a view mon is: the one whose identity (its own species even while
/// Transformed, or its construction species) is the view's species.
fn engine_mon<'a>(st: &'a BattleState, side: usize, species: &str) -> Option<&'a MonState> {
    st.sides.get(side)?.pokemon.iter().find(|m| {
        let ident = m.transform.as_ref().map_or(m.species_id.as_str(), |t| t.base_species_id.as_str());
        dex::to_id(ident) == species || dex::to_id(&m.base_species_id) == species
    })
}

const ENGINE_BOOSTS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

fn engine_boosts(b: &[i8; 7]) -> Vec<(&'static str, i32)> {
    // In poke-env's key order, nonzero only — the view's own shape.
    super::mon::BOOST_KEYS
        .iter()
        .filter_map(|k| {
            let i = ENGINE_BOOSTS.iter().position(|e| e == k).expect("same seven stats");
            (b[i] != 0).then_some((*k, b[i] as i32))
        })
        .collect()
}

fn weather_id(w: Weather) -> &'static str {
    match w {
        Weather::Sand => "sandstorm",
        Weather::Rain => "raindance",
        Weather::Sun => "sunnyday",
        Weather::Hail => "hail",
    }
}

/// The TRUTH AUDIT of `view` (side `side`'s, built with `flags`) against the engine `board`.
pub fn check_view(view: &OneSidedView, board: &BattleState, side: usize, dex: &Dex, pp_synced: bool) -> Audit {
    let mut a = Audit::default();
    let turn = board.turn.max(1);
    a.check(view.turn == turn, "turn", || format!("view {} engine {}", view.turn, turn));
    let f = &board.field;
    let w = f.weather.map(weather_id);
    a.check(view.weather.weather.as_deref() == w, "weather.weather", || format!("view {:?} engine {:?}", view.weather.weather, w));
    if f.weather.is_some() {
        let perm = f.weather_turns == 0;
        a.check(view.weather.is_permanent == perm, "weather.is_permanent", || format!("view {} engine {}", view.weather.is_permanent, perm));
        let ta = turn.saturating_sub(f.weather_start_turn);
        a.check(view.weather.turns_active == ta, "[V12] weather.turns_active", || format!("view {} engine {}", view.weather.turns_active, ta));
    }
    side_check(&mut a, &view.ours, board, side, true, dex, pp_synced);
    side_check(&mut a, &view.opp, board, 1 - side, false, dex, false);
    a
}

fn side_check(a: &mut Audit, sv: &SideView, st: &BattleState, side: usize, own: bool, dex: &Dex, pp_synced: bool) {
    let who = if own { "ours" } else { "opp" };
    let sd = &st.sides[side];
    a.check(sv.team_size == sd.pokemon.len(), &format!("{who}.team_size"), || format!("view {} engine {}", sv.team_size, sd.pokemon.len()));
    // Side conditions: the KEY set is the sim's; Spikes' value is its layer count (a timed
    // screen's value is the TURN it started — V11 — which the engine does not keep).
    let mut want: Vec<(&str, Option<u32>)> = Vec::new();
    if sd.spikes > 0 {
        want.push(("spikes", Some(sd.spikes as u32)));
    }
    for (k, rem) in [("reflect", sd.reflect), ("light_screen", sd.light_screen), ("safeguard", sd.safeguard)] {
        if rem > 0 {
            want.push((k, None));
        }
    }
    let mut have: Vec<&str> = sv.side_conditions.iter().map(|(k, _)| k.as_str()).collect();
    have.sort_unstable();
    let mut wk: Vec<&str> = want.iter().map(|(k, _)| *k).collect();
    wk.sort_unstable();
    a.check(have == wk, &format!("{who}.side_conditions[keys]"), || format!("view {have:?} engine {wk:?}"));
    for (k, v) in &want {
        if let (Some(v), Some((_, got))) = (v, sv.side_conditions.iter().find(|(kk, _)| kk == k)) {
            a.check(got == v, &format!("{who}.side_conditions[{k}]"), || format!("view {got} engine {v}"));
        }
    }
    // The active mon: the engine's active slot, by identity species.
    let eng_act = sd.pokemon.get(sd.active).map(|m| {
        dex::to_id(m.transform.as_ref().map_or(m.species_id.as_str(), |t| t.base_species_id.as_str()))
    });
    let view_act = sv.active.map(|i| sv.mons[i].species.clone());
    if own || view_act.is_some() {
        let ok = match (&view_act, &eng_act) {
            (Some(v), Some(e)) => v == e || engine_mon(st, side, v).is_some_and(|m| std::ptr::eq(m, &sd.pokemon[sd.active])),
            (None, None) => true,
            _ => false,
        };
        a.check(ok, &format!("{who}.active"), || format!("view {view_act:?} engine {eng_act:?}"));
    }
    for m in &sv.mons {
        match engine_mon(st, side, &m.species) {
            None => a.check(false, &format!("{who}.species-exists"), || format!("{} is not on the engine's team", m.species)),
            Some(e) => mon_check(a, m, e, own, dex, who, pp_synced && m.active),
        }
    }
}

fn mon_check(a: &mut Audit, m: &MonView, e: &MonState, own: bool, dex: &Dex, who: &str, pp_synced: bool) {
    let sp = m.species.as_str();
    let cls = |f: &str| format!("{who}.{f}");
    a.check(m.fainted == e.fainted, &cls("fainted"), || format!("{sp}: view {} engine {}", m.fainted, e.fainted));
    let st = status_live(e);
    a.check(m.status == st, &cls("status"), || format!("{sp}: view {:?} engine {:?}", m.status, st));
    // HP: the owner's exact pair; the watcher's the ceil% fold (V9).
    let (cur, max) = if own { (e.hp as u32, e.maxhp as u32) } else { (hp_percent(e.hp, e.maxhp), 100) };
    a.check(m.current_hp == cur && m.max_hp == max, &cls("hp"), || {
        format!("{sp}: view {}/{} engine {cur}/{max}", m.current_hp, m.max_hp)
    });
    let frac = if cur == 0 { 0.0 } else { cur as f64 / max as f64 };
    a.check(m.hp_fraction == frac, &cls("hp_fraction"), || format!("{sp}: view {} engine {frac}", m.hp_fraction));
    // Boosts — the sim's (a fainted mon's were cleared at the faint: PE-V10's truth).
    let want = engine_boosts(&e.boosts);
    a.check(m.boosts == want, &cls("boosts"), || format!("{sp}: view {:?} engine {want:?}", m.boosts));
    let transformed = e.transform.is_some();
    if !m.fainted && !transformed {
        let et = engine_types(e, dex);
        a.check(m.types == et, &cls("types"), || format!("{sp}: view {:?} engine {et:?}", m.types));
    }
    // The badly-poisoned stage (PE-R1b's truth) on the ACTIVE mon; a benched one is UNRESOLVED.
    if let Some(Status::Toxic(n)) = e.status {
        if !e.fainted && m.active {
            a.check(m.status_counter == n as u32, &cls("status_counter[tox]"), || {
                format!("{sp}: view {} engine stage {n}", m.status_counter)
            });
        } else if !e.fainted && m.status_counter != n as u32 {
            *a.rules_fired.entry("UNRESOLVED:tox-stage-benched").or_default() += 1;
        }
    }
    // The truth of every volatile the protocol announces, on an ACTIVE living mon.
    if m.active && !m.fainted {
        let names = crate::search::volatile_names(e);
        let mut eng: Vec<&str> = TRUTH_VOLATILES.iter().filter(|(k, _)| names.contains(k)).map(|(_, v)| *v).collect();
        let mut rd: Vec<&str> =
            m.volatiles.iter().map(|(k, _)| k.as_str()).filter(|k| TRUTH_VOLATILES.iter().any(|(_, v)| v == k)).collect();
        eng.sort_unstable();
        rd.sort_unstable();
        a.check(eng == rd, &cls("volatiles"), || format!("{sp}: view {rd:?} engine {eng:?}"));
    }
    let eng_item = dex::to_id(&e.item);
    let eng_ability = dex::to_id(if e.fainted { &e.set.ability } else { &e.ability });
    let eng_moves: Vec<String> = e.set.moves.iter().map(|mv| dex::retrieve_id(&crate::state::typed_hp_move_id(e, mv))).collect();
    if own {
        a.check(m.item.as_deref() == Some(eng_item.as_str()), &cls("item"), || format!("{sp}: view {:?} engine {eng_item:?}", m.item));
        if !transformed {
            a.check(m.ability.as_deref() == Some(eng_ability.as_str()).filter(|x| !x.is_empty()), &cls("ability"), || {
                format!("{sp}: view {:?} engine {eng_ability:?}", m.ability)
            });
        }
        let stats = [e.stats[0] as i64, e.stats[1] as i64, e.stats[2] as i64, e.stats[3] as i64, e.stats[4] as i64, e.stats[5] as i64];
        a.check(m.stats.iter().zip(stats.iter()).all(|(v, s)| *v == Some(*s)), &cls("stats"), || {
            format!("{sp}: view {:?} engine {stats:?}", m.stats)
        });
        let ivs: Vec<i64> = e.set.ivs.iter().map(|v| *v as i64).collect();
        let evs: Vec<i64> = e.set.evs.iter().map(|v| *v as i64).collect();
        a.check(m.ivs.as_deref() == Some(&ivs[..]) && m.evs.as_deref() == Some(&evs[..]), &cls("spread"), || {
            format!("{sp}: view {:?}/{:?} engine {ivs:?}/{evs:?}", m.ivs, m.evs)
        });
        let nat = if e.set.nature.is_empty() { "serious".to_string() } else { e.set.nature.to_ascii_lowercase() };
        a.check(m.nature.as_deref() == Some(nat.as_str()), &cls("nature"), || format!("{sp}: view {:?} engine {nat:?}", m.nature));
        if transformed {
            // V14: the overlay is poke-env's (the target's KNOWN moves at 5 PP) — a subset.
            let ok = m.moves.iter().all(|mv| {
                eng_moves.iter().any(|x| *x == mv.id)
            });
            a.check(ok, &cls("[V14] moves"), || format!("{sp}: view {:?}", m.moves.iter().map(|x| &x.id).collect::<Vec<_>>()));
        } else {
            let keys: Vec<&str> = m.moves.iter().map(|x| x.id.as_str()).collect();
            let mut ek: Vec<&str> = eng_moves.iter().map(String::as_str).collect();
            ek.sort_unstable();
            a.check(keys == ek, &cls("moves[ids]"), || format!("{sp}: view {keys:?} engine {ek:?}"));
            for mv in &m.moves {
                if let Some(i) = eng_moves.iter().position(|k| *k == mv.id) {
                    let (pp, mx) = (e.move_pp.get(i).copied().unwrap_or(0) as u32, e.move_maxpp.get(i).copied().unwrap_or(0) as u32);
                    a.check(mv.max_pp == mx, &cls("moves[max_pp]"), || format!("{sp}.{}: view {} engine {mx}", mv.id, mv.max_pp));
                    if pp_synced {
                        // The active mon the side's CURRENT request re-synced (fork R3): the sim's word.
                        a.check(mv.current_pp == pp, &cls("moves[pp]"), || format!("{sp}.{}: view {} engine {pp}", mv.id, mv.current_pp));
                    } else {
                        // V15 — every other own mon's PP is poke-env's SIGHTING count, which misses a
                        // deduction it cannot see (a foe's Pressure, announced to its owner only, on
                        // a move whose user leaves the field before the next request). It can only
                        // lag: never below the sim's.
                        a.check(mv.current_pp >= pp, &cls("[V15] moves[pp]"), || format!("{sp}.{}: view {} engine {pp}", mv.id, mv.current_pp));
                        if mv.current_pp != pp {
                            *a.rules_fired.entry("V15").or_default() += 1;
                        }
                    }
                }
            }
        }
    } else {
        // V9: nothing private leaks.
        a.check(m.ivs.is_none() && m.evs.is_none() && m.nature.is_none() && !m.spread_known && m.stats == [None; 6],
                &cls("[V9] hidden"), || format!("{sp}: an opponent's spread or stats reached the view"));
        if let Some(it) = &m.item {
            a.check(*it == eng_item, &cls("item-is-held"), || format!("{sp}: view {it:?} engine {eng_item:?}"));
        } else if let Some(c) = &m.consumed_item {
            a.check(*c != eng_item, &cls("consumed-item-not-held"), || format!("{sp}: consumed {c:?} engine holds {eng_item:?}"));
        }
        if let Some(ab) = &m.ability {
            if !transformed {
                a.check(*ab == eng_ability, &cls("ability"), || format!("{sp}: view {ab:?} engine {eng_ability:?}"));
            }
        }
        for mv in &m.moves {
            a.check(eng_moves.iter().any(|k| *k == mv.id) || transformed, &cls("move-in-moveset"), || {
                format!("{sp}: view {} engine {eng_moves:?}", mv.id)
            });
        }
    }
}
