// probe_batch7_isolated.js — ISOLATED node-vs-rust byte verification of the MULTI-STRIKE
// family (`gen3_move_coverage_batch7_v1`), separated from the random-mode tail confounds
// (both-side screens / evasion / Cute Charm). Constructs CLEAN teams whose ONLY damaging
// option is a multihit move (so the modeled picker picks it), plays N battles to game-end
// via the SHARED runBattle/emitBattle, and replays each protocol chunk through ab_replay
// --protocol. A byte-clean pass ⇒ the multihit ENGINE is correct.
//
// Run: POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay \
//        node src/rust_sim/harness/probe_batch7_isolated.js [nBattles]
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const { execFileSync } = require('child_process');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams } = require(path.join(PS, 'dist/sim'));
const { runBattle, emitBattle, seedFrom } = require('./gen_e2e_fuzz.js');

const REPLAYER = process.env.POKESIM_AB_REPLAY_BIN
  || path.join(__dirname, '..', 'target/release/ab_replay');
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: 'Serious', level: 100, gender: opts.gender || 'N',
  };
}
// Clean attacker team: every mon's ONLY damaging move is a multihit move (+ Splash filler),
// so the picker picks the multihit move or switches. NO screens/evasion/Cute-Charm.
const ATTACKERS = [
  mon('Beedrill', ['twineedle', 'splash'], { evs: { atk: 252 } }),      // fixed 2 + 20% psn
  mon('Hitmonlee', ['doublekick', 'splash'], { evs: { atk: 252 } }),    // fixed 2
  mon('Cloyster', ['iciclespear', 'splash'], { evs: { spa: 252 } }),    // variable [2,5]
  mon('Marowak', ['bonemerang', 'splash'], { evs: { atk: 252 } }),      // fixed 2, acc 90
  mon('Jumpluff', ['bulletseed', 'splash'], { evs: { spa: 252 } }),     // variable [2,5]
  mon('Pinsir', ['pinmissile', 'splash'], { evs: { atk: 252 } }),       // variable [2,5], acc 85
];
// Bulky defenders (survive several strikes) so the multihit resolves fully — plus a frailer
// one so a KO-mid-sequence occurs. NO screens.
const DEFENDERS = [
  mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } }),
  mon('Blissey', ['splash', 'softboiled'], { evs: { hp: 252, def: 252 } }),
  mon('Skarmory', ['splash'], { evs: { hp: 252, def: 252 } }),
  mon('Chansey', ['splash'], { evs: { hp: 252 } }),
  mon('Miltank', ['splash'], { evs: { hp: 252, def: 252 } }),
  mon('Diglett', ['splash']),  // frail → KO-mid-sequence
];

async function main() {
  const N = parseInt(process.argv[2] || '80', 10);
  const p1Packed = Teams.pack(ATTACKERS);
  const p2Packed = Teams.pack(DEFENDERS);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'batch7iso_'));
  let ok = 0, diverged = 0, hitcounts = 0, panics = 0;
  const bad = [];
  for (let i = 0; i < N; i++) {
    const battleSeed = seedFrom(0xB7150 + i * 101);
    const chooseSeed = 0xC0FFEE + i * 7;
    let rec;
    try {
      rec = await runBattle(p1Packed, p2Packed, battleSeed, chooseSeed, 'pool', { protocol: true });
    } catch (e) { panics++; bad.push(`b${i}: runBattle threw ${e.message}`); continue; }
    if (!rec || !rec.lines) continue;
    hitcounts += rec.lines.filter((l) => /\|-hitcount\|/.test(l)).length;
    const lines = [];
    emitBattle(lines, `iso_${i}`, p1Packed, p2Packed, rec, { protocol: true });
    const chunkFile = path.join(tmp, `b${i}.txt`);
    fs.writeFileSync(chunkFile, lines.join('\n') + '\n');
    let out;
    try {
      out = execFileSync(REPLAYER, [chunkFile, '--protocol'], { encoding: 'utf8' });
    } catch (e) { out = (e.stdout || '') + (e.stderr || ''); }
    const verdictLine = out.split('\n').find((l) => l.includes('"verdict"'));
    if (!verdictLine) { diverged++; bad.push(`b${i}: no verdict`); continue; }
    const v = JSON.parse(verdictLine);
    if (v.verdict === 'ok') ok++;
    else { diverged++; bad.push(`b${i}: ${v.kind}@${v.decision} exp=${v.expected} got=${v.got}`); }
  }
  console.log(`\n=== BATCH-7 ISOLATED byte check: ${N} battles ===`);
  console.log(`  ok=${ok} diverged=${diverged} panic=${panics}  (total hitcount lines=${hitcounts})`);
  for (const b of bad.slice(0, 15)) console.log('  DIVERGED', b);
  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) {}
  process.exit(diverged || panics ? 1 : 0);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
