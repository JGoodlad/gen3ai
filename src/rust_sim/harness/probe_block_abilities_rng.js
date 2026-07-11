// probe_block_abilities_rng.js — settle the draw model of the BLOCK abilities (task #69):
//   Suction Cups (blocks a Roar/Whirlwind phaze DRAG — like Protect-blocks-phaze, the
//     `sample` is NOT drawn), Soundproof (blocks sound moves — which MODELED moves are
//     sound?), Damp (blocks Explosion/Self-Destruct — the user does NOT self-KO).
// The sim is the only oracle. Run: node .../probe_block_abilities_rng.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves, evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
function siteLabel(stack) {
  for (const l of String(stack).split('\n').slice(2)) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle'].includes(m[1])) return m[1];
  }
  return '?';
}
async function run(p1, p2, seed, choices, fmt) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const prng = battle.prng;
  const calls = [];
  const wrap = (name) => { const orig = prng[name].bind(prng); prng[name] = (...a) => { const r = orig(...a); calls.push({ kind: name, args: a, site: siteLabel(new Error().stack), ret: r }); return r; }; };
  wrap('random'); wrap('randomChance'); wrap('sample');
  const per = [];
  for (const [c1, c2] of choices) {
    const before = calls.length;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 12; k++) await tick();
    per.push(calls.slice(before));
    if (battle.ended) break;
  }
  return { per, lines };
}
function fmtCalls(cs) { return cs.map((c) => `${c.kind}(${JSON.stringify(c.args)})@${c.site}=>${JSON.stringify(c.ret)}`).join('  '); }

async function cmp(label, p1fn, probed, control, choices, fmt) {
  console.log(`\n=== ${label}  [probed=${probed} vs control=${control}] ===`);
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [42, 42, 42, 42]];
  for (const seed of seeds) {
    const a = await run(...p1fn(probed), seed, choices, fmt);
    const b = await run(...p1fn(control), seed, choices, fmt);
    const at = a.per.reduce((s, d) => s + d.length, 0);
    const bt = b.per.reduce((s, d) => s + d.length, 0);
    console.log(`  seed ${JSON.stringify(seed)}: probed=${a.per.map((d) => d.length).join(',')}(${at}) control=${b.per.map((d) => d.length).join(',')}(${bt}) match=${at === bt && JSON.stringify(a.per.map((d) => d.length)) === JSON.stringify(b.per.map((d) => d.length))}`);
    console.log(`    probed dec0: ${fmtCalls(a.per[0])}`);
    const activate = a.lines.filter((l) => /-activate|cant|-immune|switch|drag|faint/.test(l) && !/^\|switch\|p[12]a: [A-Z].*\|100\/100$/.test(l));
    console.log(`    probed key lines: ${JSON.stringify(activate.slice(0, 8))}`);
  }
}

(async () => {
  // Which MODELED moves are sound? Check the sound flag on the modeled-move universe.
  const d3 = Dex.mod('gen3');
  const modeled = ['thunderwave', 'stunspore', 'glare', 'poisonpowder', 'poisongas', 'toxic', 'willowisp',
    'spore', 'sleeppowder', 'hypnosis', 'sing', 'lovelykiss', 'grasswhistle', 'roar', 'whirlwind', 'perishsong'];
  console.log('=== sound flag on candidate status/phaze moves ===');
  for (const id of modeled) {
    const m = d3.moves.get(id);
    if (m && m.exists) console.log(`  ${id.padEnd(14)} sound=${!!(m.flags && m.flags.sound)} category=${m.category}`);
  }

  // ---- Suction Cups: Roar into a Suction-Cups mon → NO drag → NO sample draw ----
  await cmp('BLOCK Suction Cups — Roar into a Suction-Cups mon (drag blocked, no sample)',
    (ab) => [
      [mon('Cradily', ['recover', 'recover'], { ability: ab }), mon('Snorlax', ['recover', 'recover'])],
      [mon('Suicune', ['roar', 'roar'])],
    ],
    'Suction Cups', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // ---- Soundproof: Sing (sound sleep move) into a Soundproof mon → immune ----
  await cmp('BLOCK Soundproof — Sing into a Soundproof mon (accuracy? immune?)',
    (ab) => [
      [mon('Jynx', ['sing', 'sing'])],
      [mon('Electrode', ['recover', 'recover'], { ability: ab })],
    ],
    'Soundproof', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // ---- Soundproof: a NON-sound status move (Thunder Wave) is NOT blocked ----
  await cmp('BLOCK Soundproof — Thunder Wave (NOT sound) into a Soundproof mon (lands)',
    (ab) => [
      [mon('Jolteon', ['thunderwave', 'thunderwave'])],
      [mon('Electrode', ['recover', 'recover'], { ability: ab })],
    ],
    'Soundproof', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']]);

  // ---- Damp: Explosion by a mon while the foe has Damp → the user does NOT self-KO ----
  await cmp('BLOCK Damp — Explosion while foe has Damp (user does NOT self-KO)',
    (ab) => [
      [mon('Snorlax', ['explosion', 'explosion']), mon('Blissey', ['recover', 'recover'])],
      [mon('Golduck', ['recover', 'recover'], { ability: ab }), mon('Blissey', ['recover', 'recover'])],
    ],
    'Damp', 'Shell Armor', [['move 1', 'move 1']]);

  // ---- Damp: which side's Damp blocks? onAnyTryMove — so a Damp mon blocks its OWN
  //      Explosion too. And a Damp mon blocks the FOE's Explosion. ----
  await cmp('BLOCK Damp — the Damp mon\'s OWN Explosion is also blocked (onAnyTryMove)',
    (ab) => [
      [mon('Golduck', ['explosion', 'explosion'], { ability: ab }), mon('Blissey', ['recover', 'recover'])],
      [mon('Snorlax', ['recover', 'recover']), mon('Blissey', ['recover', 'recover'])],
    ],
    'Damp', 'Shell Armor', [['move 1', 'move 1']]);
})();
