// probe_batch6_regression_rng.js — REAL-Showdown ground-truth seeds/state for the
// `gen3_move_coverage_batch6_v1` regression pins (MC79+): ENCORE / DESTINY BOND /
// ENDURE / PERISH SONG / MEAN LOOK-family / BELLY DRUM / CHARGE / MEMENTO / MIMIC /
// PAIN SPLIT / PSYCH UP.
//
// Each scenario mirrors its `tests/regression_test.rs` pin EXACTLY (teams, seed,
// choices, injections); the printed per-boundary `seedAfter` + state are copied
// verbatim into the pin's constants. Re-run after any PRNG/draw-order change.
//
// Run:  node src/rust_sim/harness/probe_batch6_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  const m = {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
  if (opts.happiness !== undefined) m.happiness = opts.happiness;
  return m;
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.pp) m.moveSlots[inj.pp.moveSlot].pp = inj.pp.val;
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\nTEAM p1 ${Teams.pack(p1team)}`);
  console.log(`TEAM p2 ${Teams.pack(p2team)}`);
  console.log(`=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  const ppStr = (m) => m ? m.moveSlots.map((s) => `${s.id}:${s.pp}/${s.maxpp}`).join(',') : '-';
  const vol = (m) => {
    if (!m) return '';
    const bits = [];
    if (m.volatiles['encore']) bits.push(`enc(d=${m.volatiles['encore'].duration},mv=${m.volatiles['encore'].move})`);
    if (m.volatiles['perishsong']) bits.push(`per(d=${m.volatiles['perishsong'].duration})`);
    if (m.volatiles['destinybond']) bits.push('db');
    if (m.volatiles['endure']) bits.push('endure');
    if (m.volatiles['stall']) bits.push(`stall(c=${m.volatiles['stall'].counter})`);
    if (m.volatiles['charge']) bits.push('charge');
    if (m.volatiles['trapped']) bits.push('TRAPPED');
    if (m.trapped) bits.push(`trapflag=${m.trapped}`);
    const b = Object.entries(m.boosts).filter(([, v]) => v).map(([k, v]) => `${k}${v > 0 ? '+' : ''}${v}`).join(',');
    if (b) bits.push(`[${b}]`);
    return bits.length ? ' ' + bits.join(' ') : '';
  };
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    applyActs(entry.pre);
    const before = battle.prng.getSeed();
    const l0 = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${vol(m)}` : '-';
    console.log(`  dec${i - 1} [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} seed ${before} -> ${battle.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0)} pp={${ppStr(a0)}}`);
    console.log(`      p2=${fmt(a1)} pp={${ppStr(a1)}}`);
    const key = log.slice(l0).filter((l) => /move\||-damage|-fail|-immune|-miss|cant|faint|-end\b|-activate|-heal|-status|-start|-boost|-unboost|-setboost|-copyboost|-sethp|-singlemove|-singleturn|-fieldactivate|switch\||error/.test(l));
    for (const l of key) console.log(`      LINE ${l}`);
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // MC79 — the FASTER encore user (the target has NOT moved → stored = rolled) + the
  // onOverrideAction execution override: Jolteon encores Snorlax's splash while the
  // Snorlax QUEUED bodyslam — the queued move executes AS splash, splash's PP deducts.
  await run('MC79 faster encore stores rolled + the override deducts the ENCORED slot',
    [mon('Jolteon', ['encore', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },  // establish lastMove = splash
     { p1: 'move 1', p2: 'move 2' },  // encore lands; queued bodyslam OVERRIDDEN to splash
     { p1: 'move 2', p2: 'move 1' },  // locked splash
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);

  // MC80 — the SLOWER encore user (the target ALREADY moved → stored = rolled + 1).
  await run('MC80 slower encore stores rolled+1',
    [mon('Snorlax', ['encore', 'bodyslam'], { evs: { hp: 252 } })],
    [mon('Jolteon', ['splash', 'thunderbolt'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },  // Jolteon splashes FIRST (lastMove splash)
     { p1: 'move 1', p2: 'move 1' },  // encore (Jolteon already moved this turn)
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);

  // MC81 — the encore FAIL SPLIT: no-lastMove (acc + durationCallback BOTH drawn) vs
  // already-encored (accuracy ONLY).
  await run('MC81 encore fail split (no-lastMove 2 draws vs already-encored 1 draw)',
    [mon('Jolteon', ['encore', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },  // NO-LASTMOVE fail (Jolteon faster): acc + dur
     { p1: 'move 1', p2: 'move 1' },  // encore LANDS
     { p1: 'move 1', p2: 'move 1' }]); // ALREADY-ENCORED fail: acc only

  // MC82 — the 0-PP EARLY `-end` + the 0-PP-lastMove fail: splash injected to 2 PP.
  await run('MC82 encore 0-PP early end + 0-PP-lastMove fail',
    [mon('Jolteon', ['encore', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },  // splash (2 -> 1), lastMove splash
     { p1: 'move 1', p2: 'move 1' },  // encore lands; splash 1 -> 0 -> the residual EARLY -end
     { p1: 'move 1', p2: 'move 2' }], // re-encore into the 0-PP lastMove: FAIL (acc + dur)
    { acts: [{ side: 1, pp: { moveSlot: 0, val: 2 } }] });

  // MC83 — DESTINY BOND mutual faint (both last mons → the gen-3 TIE): the cast turn
  // is draw-free; the KO turn draws Body Slam's acc/crit/dmg/secondary, NO Quick Claw.
  await run('MC83 destiny bond mutual faint -> tie',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },  // DB cast (Gengar faster), Snorlax splashes
     { p1: 'move 2', p2: 'move 1' }], // Gengar splashes FIRST (window still open this turn?
                                      //  NO — the splash closes it at onBeforeMove -1) — see MC84;
                                      //  here the KO turn is the SAME turn as a re-cast:
    { acts: [{ side: 0, hp: 120 }] });

  // MC83b — the SAME-TURN cast-and-KO mutual faint (DB3): Gengar casts, Snorlax KOs it
  // that very turn → BOTH faint → tie.
  await run('MC83b destiny bond cast-and-KOd same turn -> mutual faint tie',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 120 }] });

  // MC84 — the WINDOW CLOSED: DB, then the user SPLASHES (the next move attempt
  // removes the volatile), then the KO → NO mutual faint (Snorlax survives, wins).
  await run('MC84 destiny bond window closed by the next move attempt',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },  // DB cast; Snorlax splashes
     { p1: 'move 2', p2: 'move 2' },  // Gengar SPLASHES -> the window closes
     { p1: 'move 2', p2: 'move 1' }], // Body Slam KOs -> NO mutual faint
    { acts: [{ side: 0, hp: 120 }] });

  // MC85 — a RESIDUAL (sand chip) KO does NOT trigger DB — and the whole turn is
  // ZERO draws (cast + splash + chip + faint, distinct speeds).
  await run('MC85 destiny bond not triggered by the sand-chip KO (zero-draw turn)',
    [mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate' })],
    [mon('Tyranitar', ['splash', 'crunch'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 15 }] });

  // MC86 — the ENDURE survive-at-1 + the SHARED stall ladder: Snorlax endures
  // Double-Edge repeatedly (2 -> 4 -> 8) until a roll FAILS -> the KO.
  await run('MC86 endure survives at 1 HP + the stall ladder to a failed roll',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } })],
    [mon('Tauros', ['doubleedge', 'splash'], { evs: { atk: 252, spe: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }]);

  // MC87 — ENDURE guards MOVE damage only: the burned endurer survives Seismic Toss
  // at 1 HP then the SAME turn's burn residual kills it.
  await run('MC87 endured seismic toss at 1 HP then the burn residual kills',
    [mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, status: 'brn', hp: 80 }] });

  // MC88 — PERISH SONG ticks 3 -> 2 -> 1 -> the faint; the caster's own pivot at
  // perish1 CLEARS its counter (it survives).
  await run('MC88 perish ticks to the faint; the pivot clears the counter',
    [mon('Celebi', ['perishsong', 'splash'], { evs: { hp: 252 } }),
     mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // cast: both get perish (boundary shows 3)
     { p1: 'move 2', p2: 'move 1' },   // tick -> 2
     { p1: 'switch 2', p2: 'move 1' }, // Celebi pivots at 1 -> ITS counter clears; Snorlax ticks -> 1
     { p1: 'move 1', p2: 'move 1' },   // Snorlax faints at 0 (perish0) -> p2 forced switch
     { p2: 'switch 2' },
     { p1: 'move 1', p2: 'move 2' }]);

  // MC89 — the PERISH MIRROR at an equal speed: each residual draws ONE order-12 pair
  // tie-shuffle; the mutual perish-out (both LAST mons) is a double faint -> the TIE.
  await run('MC89 perish mirror tie -> mutual perish-out tie',
    [mon('Snorlax', ['perishsong', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' }]);

  // MC90 — MEAN LOOK: the grounded GHOST is FIRM-trapped (its switch is REJECTED
  // draw-free with the boundary open); the TRAPPER's pivot frees it (the next switch
  // is ACCEPTED).
  await run('MC90 mean look firm-traps the ghost; the trapper leaving frees it',
    [mon('Umbreon', ['meanlook', 'seismictoss'], { evs: { hp: 252 } }),
     mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['nightshade', 'splash'], { ability: 'Levitate', evs: { hp: 252 } }),
     mon('Misdreavus', ['nightshade', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },   // mean look lands (Gengar trapped)
     { p1: 'move 2', p2: 'switch 2' }, // p2's switch is REJECTED (trapped) — then re-choose:
     { p1: 'move 2', p2: 'move 2' },   // p2 stays (splash)
     { p1: 'switch 2', p2: 'move 2' }, // the TRAPPER pivots out -> the link ends
     { p1: 'move 1', p2: 'switch 2' }]); // p2's switch is now ACCEPTED

  // MC91 — SPIDER WEB + BATON PASS: the trapped Celebi legally Baton-Passes; the
  // ENTRANT inherits the FIRM trap; the trapper's FAINT frees it.
  await run('MC91 spider web + baton pass inherit; the trapper faint frees',
    [mon('Ariados', ['spiderweb', 'splash'], { evs: { hp: 252 } }),
     mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [mon('Celebi', ['batonpass', 'splash'], { evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },   // spider web lands (Celebi trapped)
     { p1: 'move 2', p2: 'move 1' },   // Celebi Baton-Passes OUT (LEGAL while trapped)
     { p2: 'switch 2' },               // the BP replacement: Snorlax inherits the trap
     { p1: 'move 2', p2: 'switch 2' }, // Snorlax's switch is REJECTED (inherited firm trap)
     { p1: 'move 2', p2: 'move 1' },   // Body Slam chips Ariados
     { p1: 'move 2', p2: 'move 1' },   // Body Slam KOs Ariados -> the trapper's faint frees
     { p1: 'switch 2' },
     { p1: 'move 2', p2: 'switch 2' }], // Snorlax's switch is now ACCEPTED
    { acts: [{ side: 0, hp: 230 }] });

  // MC92 — the BELLY DRUM hp boundary (Snorlax maxhp 524): hp == 262 FAILS
  // (2*hp <= maxhp), hp == 263 SUCCEEDS (leaving 1, atk SET +6); the immediate
  // re-drum fails (atk >= 6). All draw-free.
  await run('MC92a belly drum at hp=262 FAILS (the 2*hp<=maxhp boundary)',
    [mon('Snorlax', ['bellydrum', 'return'], { evs: { hp: 252, atk: 252 }, happiness: 255 })],
    [mon('Skarmory', ['splash', 'seismictoss'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 262 }] });
  await run('MC92b belly drum at hp=263 SUCCEEDS (hp 1, atk +6); the re-drum fails',
    [mon('Snorlax', ['bellydrum', 'return'], { evs: { hp: 252, atk: 252 }, happiness: 255 })],
    [mon('Skarmory', ['splash', 'seismictoss'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }],
    { acts: [{ side: 0, hp: 263 }] });

  // MC93 — CHARGE ×2 the next ELECTRIC move only, consumed by ANY next move: charged
  // tbolt vs uncharged control at byte-identical draw counts; charge -> Surf consumes
  // it with NO boost.
  await run('MC93 charge doubles the next electric move then is consumed',
    [mon('Lanturn', ['charge', 'thunderbolt', 'surf', 'splash'], { evs: { hp: 252, spa: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },  // charge
     { p1: 'move 2', p2: 'move 1' },  // CHARGED tbolt (x2)
     { p1: 'move 2', p2: 'move 1' },  // uncharged tbolt (the control)
     { p1: 'move 1', p2: 'move 1' },  // charge
     { p1: 'move 3', p2: 'move 1' },  // surf CONSUMES the charge (no boost)
     { p1: 'move 2', p2: 'move 1' }]); // tbolt back at x1 (the consumption proof)

  // MC94 — MEMENTO: the landed turn is ZERO draws TOTAL (never-miss + the foe's
  // queued move CANCELLED by gen3 faint-cancels-all + no Quick Claw), the foe drops
  // -2 Atk / -2 SpA, the user faints; a PROTECTED memento is BLOCKED and the user
  // does NOT faint.
  await run('MC94a landed memento: zero draws, foe -2/-2, user faints, foe move cancelled',
    [mon('Dugtrio', ['memento', 'splash'], { evs: { spe: 252 } }),
     mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['bodyslam', 'protect'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },  // memento lands; Body Slam CANCELLED
     { p1: 'switch 2' },
     { p1: 'move 1', p2: 'move 1' }]);
  await run('MC94b memento into a Protect: blocked, the user does NOT faint',
    [mon('Dugtrio', ['memento', 'splash'], { evs: { spe: 252 } }),
     mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['bodyslam', 'protect'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },  // protect blocks the memento
     { p1: 'move 2', p2: 'move 1' }]);

  // MC95 — MIMIC slot semantics: the copy overwrites the Mimic slot at pp 5 /
  // maxpp calculatePP(copied,3); the copied slot's PP decrements independently; the
  // slot REVERTS on switch-out (Mimic's own remaining PP persists).
  await run('MC95 mimic copies psychic (pp 5/16), uses it, reverts on switch',
    [mon('Snorlax', ['mimic', 'splash'], { evs: { hp: 252 } }),
     mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [mon('Alakazam', ['psychic', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // Alakazam psychic FIRST -> mimic copies psychic
     { p1: 'move 1', p2: 'move 2' },   // the COPIED psychic runs (pp 5 -> 4)
     { p1: 'switch 2', p2: 'move 2' }, // pivot -> the overlay REVERTS
     { p1: 'switch 2', p2: 'move 2' }, // back in
     { p1: 'move 1', p2: 'move 2' }]); // mimic again (fresh copy of splash? lastMove=splash)

  // MC95b — the mimic NO-LASTMOVE fail (the faster mimicker moves before the foe ever
  // has a lastMove): draw-free [still]+fail.
  await run('MC95b mimic no-lastMove fail (draw-free)',
    [mon('Jolteon', ['mimic', 'thunderbolt'], { evs: { hp: 252, spe: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC96 — PAIN SPLIT: the maxhp CLAMP (Gengar 41 + Blissey 714 -> avg 377: Blissey
  // takes the FULL loss to 377, Gengar caps at its 261 maxhp); a SUB blocks it.
  await run('MC96 pain split clamp + sub block',
    [mon('Gengar', ['painsplit', 'splash'], { ability: 'Levitate' })],
    [mon('Blissey', ['splash', 'substitute'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },  // 41+714 -> Blissey 377, Gengar 261 (clamped)
     { p1: 'move 2', p2: 'move 2' },  // Blissey subs
     { p1: 'move 1', p2: 'move 1' }], // pain split BLOCKED by the sub ([still]+fail)
    { acts: [{ side: 0, hp: 41 }] });

  // MC97 — PSYCH UP copies ALL stages VERBATIM (the user's own prior stages are
  // WIPED): the cursed Snorlax (+1 atk +1 def -1 spe) psych-ups the twice-Calm-Minded
  // Suicune -> Snorlax becomes exactly {spa+2, spd+2} with atk/def/spe back to 0.
  await run('MC97 psych up copies verbatim incl. zero-ing the users own stages',
    [mon('Snorlax', ['psychup', 'curse'], { evs: { hp: 252, atk: 252 } })],
    [mon('Suicune', ['calmmind', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },  // curse (+1/+1/-1) vs calm mind
     { p1: 'move 2', p2: 'move 1' },  // curse again / calm mind again
     { p1: 'move 1', p2: 'move 2' }]); // psych up -> {0,0,+2,+2,0}

  // MC98 — CHARGE is BATON-PASSABLE (noCopy falsy — the resolved-dex fact): the
  // entrant's next Electric move is ×2. The control (splash instead of charge) runs a
  // byte-identical draw stream, so the damage pair is directly comparable.
  await run('MC98a charge passes through Baton Pass (the entrant tbolt is x2)',
    [mon('Lanturn', ['charge', 'batonpass', 'splash'], { evs: { hp: 252 } }),
     mon('Jolteon', ['thunderbolt', 'splash'], { evs: { hp: 252, spa: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },  // charge
     { p1: 'move 2', p2: 'move 1' },  // baton pass
     { p1: 'switch 2' },              // Jolteon enters (charge passed)
     { p1: 'move 1', p2: 'move 1' }]); // tbolt x2?
  // MC99 — a CONTACT fixed-damage hit (Seismic Toss) fires the DEFENDER's
  // contact-proc onDamagingHit (Effect Spore random(10) [+ sample(3) on a pass]) —
  // the e2e_7 fix (a latent batch-5-era gap the batch-6 corpus reshuffle surfaced).
  await run('MC99 seismic toss into an Effect Spore holder rolls the contact proc',
    [mon('Breloom', ['splash', 'machpunch'], { ability: 'Effect Spore', evs: { hp: 252 } })],
    [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' }]);

  await run('MC98b the control (no charge): the entrant tbolt at x1',
    [mon('Lanturn', ['charge', 'batonpass', 'splash'], { evs: { hp: 252 } }),
     mon('Jolteon', ['thunderbolt', 'splash'], { evs: { hp: 252, spa: 252 } })],
    [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 3', p2: 'move 1' },  // splash (the control)
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'switch 2' },
     { p1: 'move 1', p2: 'move 1' }]);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
