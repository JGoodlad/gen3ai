// probe_naturalcure_scenario.js — DESIGN + validate the class-sweep golden scenarios for
// Natural Cure. Pivot ONCE early (get statused -> pivot out CURED -> return UNSTATUSED), then
// finish with attacks. Foe = Blissey (Seismic Toss deals fixed 100, NO status secondary, NO
// super-effective typing) so the RETURN boundary is UNCONFOUNDED (`ok` vs the persistent status
// is 100% the cure). Pivot partner survives ST+Toxic. Starmie Recovers to outlast Blissey's bulk.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function dryRun(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const rows = [];
  let dn = 0, safety = 0, cures = 0, stalls = 0;
  while (!stream.battle.ended && safety < 500) {
    safety++;
    const b = stream.battle;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    let c;
    if (rs === 'switch') {
      const f = [b.sides[0].activeRequest, b.sides[1].activeRequest].map((r) => r && r.forceSwitch && r.forceSwitch[0]);
      c = { p1: f[0] ? 'switch 2' : null, p2: f[1] ? 'switch 2' : null };
    } else {
      c = { p1: sc.plan1[Math.min(dn, sc.plan1.length - 1)], p2: sc.plan2[Math.min(dn, sc.plan2.length - 1)] };
    }
    const before = log.length;
    const seedBefore = b.prng.getSeed();
    if (c.p1) { try { streams.omniscient.write(`>p1 ${c.p1}`); } catch (e) {} }
    if (c.p2) { try { streams.omniscient.write(`>p2 ${c.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();
    if (b.prng.getSeed() === seedBefore && log.length === before && b.requestState === rs) { stalls++; rows.push(`  dec ${dn} STALL ${JSON.stringify(c)}`); break; }
    const cureHere = log.slice(before).some((l) => l.includes('-curestatus') && l.includes('Natural Cure'));
    if (cureHere) cures++;
    const a1 = b.sides[0].active[0], a2 = b.sides[1].active[0];
    rows.push(`  dec ${String(dn).padStart(2)} ${rs.padEnd(6)} p1=${c.p1 || '-'} p2=${c.p2 || '-'} | ` +
      `p1a:${a1 ? a1.species.name : '-'}(${a1 ? a1.status || 'ok' : '-'} ${a1 ? a1.hp : 0}) ` +
      `p2a:${a2 ? a2.species.name : '-'}(${a2 ? a2.status || 'ok' : '-'} ${a2 ? a2.hp : 0})${cureHere ? '  <<CURE' : ''}`);
    dn++;
  }
  return { rows, cures, stalls, ended: stream.battle.ended, winner: stream.battle.winner, decs: dn };
}

(async () => {
  // NC Jolteon (Electric, fast, KOs Blissey-less... no, Jolteon can't dent Blissey). Use a
  // strong special attacker vs a FRAILER-than-Blissey status foe that dies in a few hits but only
  // threatens status + a weak fixed hit. Candidate foe: Misdreavus (Ghost, Toxic + Seismic Toss)
  // — frail, dies to a few neutral hits; ST no status; Ghost so no SE typing issue vs Water/Psychic.
  // Pivot partner: Snorlax (bulky Normal, survives ST + a Toxic tick).
  const p1 = (ab) => [
    mon('Starmie', ['surf', 'recover'], { ability: ab, nature: 'Timid', evs: { spa: 252, spe: 252, hp: 4 } }),
    mon('Snorlax', ['bodyslam', 'rest'], { ability: 'Own Tempo', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
  ];
  const foeMisd = (statusMove) => [mon('Misdreavus', [statusMove, 'seismictoss'], { ability: 'Levitate', nature: 'Calm', evs: { hp: 252, def: 252 } })];
  // Plan: Surf(get tox), switch(cured), switch-back(clean return @dec2), then Surf/Recover to win.
  // Recover interleaved so Starmie outlasts Misdreavus while Surfing it down. Non-cyclic tail = Surf.
  const P1 = ['move 1', 'switch 2', 'switch 2', 'move 1', 'move 2', 'move 1', 'move 1', 'move 2', 'move 1', 'move 1'];
  const P2 = ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'];

  const scTox = { id: 'nc_cure_return_tox', p1: p1('Natural Cure'), p2: foeMisd('toxic'), plan1: P1, plan2: P2 };
  const scCtl = { id: 'nc_control_tox_nonnc', p1: p1('Illuminate'), p2: foeMisd('toxic'), plan1: P1, plan2: P2 };

  for (const sc of [scTox, scCtl]) {
    for (const seed of [[11, 22, 33, 44], [7, 3, 9, 5], [1, 1, 1, 1], [99, 42, 7, 3]]) {
      const r = await dryRun(sc, seed);
      console.log(`=== ${sc.id} seed=${seed}: decs=${r.decs} cures=${r.cures} stalls=${r.stalls} ended=${r.ended} winner=${r.winner}`);
      console.log(r.rows.slice(0, 14).join('\n'));
      console.log('');
    }
  }
})();
