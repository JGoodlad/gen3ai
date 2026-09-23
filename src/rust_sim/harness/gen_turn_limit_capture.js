// gen_turn_limit_capture.js — the TURN-LIMIT TIE capture golden (`gen3_turn_limit_tie_v1`).
//
// The pinned Showdown TIES a battle whose turn counter passes 1000 (`sim/battle.ts`
// `maybeTriggerEndlessBattleClause`, called from `endTurn` right after `this.turn++`):
//
//   if (this.turn <= 100) return;
//   if (this.turn > 1000) { add('message', 'It is turn 1000. You have hit the turn limit!');
//                           tie(); return true; }                 // BEFORE `|turn|`, no Quick Claw roll
//   if (turn >= 500 && turn % 100 == 0 || turn >= 900 && turn % 10 == 0 || turn >= 990)
//       add('bigerror', `You will auto-tie if the battle doesn't end in ${N} turn(s) (on turn 1000).`);
//
// The port used to PANIC at 1,000 committed turns (`BATTLE_TURN_CAP`). This harness captures
// the real per-side streams of a battle that REACHES the limit, so the Rust fix is byte-gated.
//
// The battle: two 2-mon gen3ou teams whose players SWITCH EVERY TURN (`switch 2` swaps the two
// team slots, so the same token alternates the mons). A switch spends no PP and deals no damage,
// so the battle can only end at the limit — 1000 turns of a trivially scriptable plan. Explicit
// genders (no construction gender draw) and four distinct Speeds (no speed-tie shuffle).
//
// The golden is COMPACT, because the full streams are ~2 MB (1000 `|request|` frames per side):
//   SCEN / TEAM / INIT / CMD / END   — the bridge-golden grammar (CMD = every choice written)
//   CHUNK <id> <side> <flatLineNo> <raw line>
//                                    — every per-side line inside the WINDOWS (turns <= 3,
//                                      turns >= 998, and each `|bigerror|` line with its two
//                                      neighbours — the warning's POSITION is the fiddly part)
//   DIGEST <id> <side> <nLines> <fnv1a64 hex>
//                                    — FNV-1a 64 over EVERY per-side line, each followed by '\n'
// so the Rust gate proves the WHOLE stream byte-identical (digest + count) and points at the
// first differing line wherever the windows cover it.
//
// Output: tests/vectors/turn_limit_golden.txt
// Run:    node src/rust_sim/harness/gen_turn_limit_capture.js   (byte-reproducible; run twice, md5 identical)
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)
'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/turn_limit_golden.txt');
const FORMAT = 'gen3ou';
const WINDOW_LO = 3;
const WINDOW_HI = 998;

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, ability, gender) {
  return {
    species, item: '', ability, moves: ['splash'], evs: EV0, ivs: IV31,
    nature: 'Serious', level: 100, gender,
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

const SCEN = {
  id: 'switch_every_turn_to_the_limit',
  seed: [101, 103, 107, 109],
  // Base Speeds 30 / 50 vs 70 / 55 — every pairing is a distinct Speed.
  p1: [mon('Snorlax', 'Immunity', 'M'), mon('Regice', 'Clear Body', 'N')],
  p2: [mon('Skarmory', 'Keen Eye', 'F'), mon('Blissey', 'Natural Cure', 'F')],
};

// FNV-1a 64 (BigInt) — mirrored by `tests/turn_limit_test.rs::fnv1a64`.
function fnv1a64(text) {
  let h = 0xcbf29ce484222325n;
  const p = 0x100000001b3n;
  const bytes = Buffer.from(text, 'utf8');
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * p) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, '0');
}

async function run() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const chunks = { p1: [], p2: [] };
  const pump = (side) => (async () => { for await (const c of streams[side]) chunks[side].push(c); })();
  pump('p1'); pump('p2');
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SCEN.seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(SCEN.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(SCEN.p2) })}`);
  for (let i = 0; i < 16; i++) await tick();
  const initSeed = stream.battle.prng.getSeed();
  const quickClawRoll = !!stream.battle.quickClawRoll;
  const cmds = [];
  let guard = 0;
  while (!stream.battle.ended) {
    if (++guard > 5000) throw new Error('battle did not end at the turn limit');
    for (const side of ['p1', 'p2']) {
      cmds.push([side, 'switch 2']);
      streams[side].write('switch 2');
    }
    for (let i = 0; i < 4; i++) await tick();
  }
  for (let i = 0; i < 8; i++) await tick();
  try { streams.omniscient.destroy(); } catch (e) { /* teardown */ }
  return { initSeed, quickClawRoll, chunks, cmds, ended: stream.battle.ended, winner: stream.battle.winner, turn: stream.battle.turn };
}

async function main() {
  const rec = await run();
  if (!rec.ended || rec.winner !== '') throw new Error(`expected a TIE, got ended=${rec.ended} winner=${rec.winner}`);
  const id = SCEN.id;
  const out = [];
  out.push('# turn_limit_golden.txt — the per-side streams of a battle that REACHES Showdown\'s 1000-turn limit');
  out.push('# (gen3_turn_limit_tie_v1). Generated by harness/gen_turn_limit_capture.js; see its header for the grammar.');
  out.push(`# format ${FORMAT}  windows: turn <= ${WINDOW_LO}, turn >= ${WINDOW_HI}, every |bigerror| +-1 line  final turn ${rec.turn}`);
  out.push(`SCEN\t${id}`);
  out.push(`TEAM\t${id}\tp1\t${Teams.pack(SCEN.p1)}`);
  out.push(`TEAM\t${id}\tp2\t${Teams.pack(SCEN.p2)}`);
  out.push(['INIT', id, 0, rec.initSeed, FORMAT, rec.quickClawRoll ? 1 : 0].join('\t'));
  rec.cmds.forEach((c, i) => out.push(['CMD', id, 0, i, c[0], c[1]].join('\t')));
  for (const side of ['p1', 'p2']) {
    const lines = [];
    for (const chunk of rec.chunks[side]) {
      for (const raw of chunk.split('\n')) lines.push(raw.startsWith('|t:|') ? '|t:|<NORMALIZED>' : raw);
    }
    let turn = 0;
    const near = (i) => [i - 1, i, i + 1].some((j) => j >= 0 && j < lines.length && lines[j].startsWith('|bigerror|'));
    lines.forEach((l, i) => {
      if (l.startsWith('|turn|')) turn = Number(l.slice(6));
      if (turn <= WINDOW_LO || turn >= WINDOW_HI || near(i)) out.push(['CHUNK', id, side, i, l].join('\t'));
    });
    out.push(['DIGEST', id, side, lines.length, fnv1a64(lines.map((l) => l + '\n').join(''))].join('\t'));
  }
  out.push(['END', id, 0, 1, 'tie'].join('\t'));
  fs.writeFileSync(OUT, out.join('\n') + '\n');
  console.error(`turn-limit capture: ${rec.cmds.length} cmds, final turn ${rec.turn}, tie -> ${OUT}`);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
