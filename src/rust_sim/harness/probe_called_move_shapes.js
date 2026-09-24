// probe_called_move_shapes.js — ENUMERATE every protocol SHAPE of a `|move|` line that carries a
// `[from]` clause in gen3, by PLAYING the real Showdown sim (the only oracle; mod-chain law).
//
// Why: a CALLED move (Metronome / Assist / Nature Power / Mirror Move / Sleep Talk, and the
// redirect family Magic Coat / Snatch, the Pursuit-on-switch self-tag, and the `lockedmove`
// continuation of every lock-in / two-turn move) is announced as
//     |move|<user>|<called move>|<target>|[from] <Caller>        (gen3's BARE form —
// `data/mods/gen3/scripts.ts::useMoveInner`, `[from] ${this.dex.conditions.get(sourceEffect).name}`)
// and `attrLastMove` may append `[still]` (which also BLANKS the target), `[miss]`, `[notarget]`
// AFTER the `[from]`. poke-env handled only some sources and some orders; a Metronome-called
// move crashed the parse (`Unhandled move message format`, backlog P1, 2026-09-24).
//
// This probe plays seeded random battles (gen3customgame, so every caller is legal together)
// with caller-heavy teams and prints, per `[from]` source, every distinct TAIL shape seen:
// the target kind (self / foe / empty) and the ordered trailing tags. It is the enumeration the
// Python pins in `src/poke_env/battle/called_move_reading_test.py` are checked against.
//
//   node src/rust_sim/harness/probe_called_move_shapes.js [n_battles=1500] [seed=1]
//   node src/rust_sim/harness/probe_called_move_shapes.js 1500 1 --json   # machine-readable
//
// Output: one line per (source, shape) with its count and ONE verbatim example line.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams } = require(path.join(PS, 'dist/sim'));

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Caller-heavy sets. Every set carries at least one CALLER or a LOCK-IN / redirect move, and the
// teammates carry varied moves so Assist has a real pool to draw from.
const SETS = [
  ['Clefable', 'Leftovers', 'Magic Guard', ['metronome', 'metronome', 'softboiled', 'calmmind']],
  ['Clefable', 'Leftovers', 'Cute Charm', ['metronome', 'assist', 'encore', 'doubleedge']],
  ['Delcatty', 'Leftovers', 'Cute Charm', ['assist', 'assist', 'batonpass', 'wish']],
  ['Persian', 'Silk Scarf', 'Limber', ['assist', 'fakeout', 'hypnosis', 'return']],
  ['Pidgeot', 'Leftovers', 'Keen Eye', ['mirrormove', 'mirrormove', 'fly', 'quickattack']],
  ['Dodrio', 'Choice Band', 'Early Bird', ['mirrormove', 'drillpeck', 'return', 'quickattack']],
  ['Shiftry', 'Leftovers', 'Chlorophyll', ['naturepower', 'naturepower', 'solarbeam', 'explosion']],
  ['Ludicolo', 'Leftovers', 'Swift Swim', ['naturepower', 'surf', 'raindance', 'leechseed']],
  ['Snorlax', 'Leftovers', 'Immunity', ['sleeptalk', 'rest', 'bodyslam', 'curse']],
  ['Swampert', 'Leftovers', 'Torrent', ['sleeptalk', 'rest', 'earthquake', 'roar']],
  ['Espeon', 'Leftovers', 'Synchronize', ['magiccoat', 'psychic', 'calmmind', 'batonpass']],
  ['Alakazam', 'Twisted Spoon', 'Synchronize', ['magiccoat', 'psychic', 'thunderwave', 'encore']],
  ['Blissey', 'Leftovers', 'Natural Cure', ['snatch', 'softboiled', 'toxic', 'seismictoss']],
  ['Umbreon', 'Leftovers', 'Synchronize', ['snatch', 'meanlook', 'toxic', 'pursuit']],
  ['Tyranitar', 'Choice Band', 'Sand Stream', ['pursuit', 'rockslide', 'earthquake', 'doubleedge']],
  ['Salamence', 'Leftovers', 'Intimidate', ['outrage', 'dragondance', 'earthquake', 'fireblast']],
  ['Tauros', 'Leftovers', 'Intimidate', ['thrash', 'return', 'earthquake', 'bide']],
  ['Golem', 'Leftovers', 'Rock Head', ['rollout', 'explosion', 'earthquake', 'spikes']],
  ['Exploud', 'Leftovers', 'Soundproof', ['uproar', 'return', 'flamethrower', 'overheat']],
  ['Aerodactyl', 'Leftovers', 'Pressure', ['skyattack', 'rockslide', 'earthquake', 'hiddenpower']],
  ['Cloyster', 'Leftovers', 'Shell Armor', ['razorwind', 'spikes', 'surf', 'explosion']],
  ['Kingdra', 'Leftovers', 'Swift Swim', ['dive', 'hydropump', 'dragondance', 'substitute']],
  ['Dugtrio', 'Leftovers', 'Arena Trap', ['dig', 'earthquake', 'rockslide', 'aerialace']],
  ['Breloom', 'Leftovers', 'Effect Spore', ['focuspunch', 'spore', 'substitute', 'machpunch']],
  ['Gengar', 'Leftovers', 'Levitate', ['protect', 'thunderbolt', 'icepunch', 'destinybond']],
  ['Skarmory', 'Leftovers', 'Keen Eye', ['spikes', 'whirlwind', 'protect', 'drillpeck']],
];

function teamFor(rng) {
  const picks = new Set();
  while (picks.size < 6) picks.add(Math.floor(rng() * SETS.length));
  const seen = new Set();
  const out = [];
  for (const i of picks) {
    const [species, item, ability, moves] = SETS[i];
    if (seen.has(species)) continue; // species clause is off in customgame; keep it readable anyway
    seen.add(species);
    out.push({ species, item, ability, moves, nature: 'Serious', evs: { hp: 84, atk: 84, def: 84, spa: 84, spd: 84, spe: 84 },
      ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 100, gender: '' });
  }
  return out;
}

function choose(req, rng) {
  if (req.wait) return null;
  if (req.forceSwitch) {
    const opts = req.side.pokemon.map((p, i) => ({ p, i })).filter(({ p }) => !p.active && !p.condition.endsWith(' fnt'));
    return opts.length ? `switch ${opts[Math.floor(rng() * opts.length)].i + 1}` : 'pass';
  }
  const act = req.active[0];
  const moves = act.moves.map((m, i) => ({ m, i })).filter(({ m }) => !m.disabled);
  const sw = act.trapped ? [] : req.side.pokemon.map((p, i) => ({ p, i })).filter(({ p }) => !p.active && !p.condition.endsWith(' fnt'));
  if (moves.length && (rng() < 0.85 || !sw.length)) return `move ${moves[Math.floor(rng() * moves.length)].i + 1}`;
  if (sw.length) return `switch ${sw[Math.floor(rng() * sw.length)].i + 1}`;
  return 'move 1';
}

function shapeOf(line) {
  const f = line.split('|');
  // ['', 'move', user, move, target, ...tags]
  const user = f[2] || '';
  const target = f[4] === undefined ? '<none>' : f[4] === '' ? '<empty>' : f[4].slice(0, 3) === user.slice(0, 3) ? '<self>' : '<foe>';
  const tags = f.slice(5).map((t) => (t.startsWith('[spread]') ? '[spread] …' : t.startsWith('[anim]') ? '[anim] …' : t));
  const from = tags.find((t) => t.startsWith('[from]'));
  return { from, key: [target, ...tags].join(' | ') };
}

function main() {
  const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
  const json = process.argv.includes('--json');
  const n = Number(args[0] || 1500);
  const rng = mulberry32(Number(args[1] || 1) >>> 0);
  const shapes = new Map();
  let lines = 0;
  for (let b = 0; b < n; b++) {
    const seed = [1 + b, 2 + b, 3 + b, 4 + b];
    const battle = new Battle({ formatid: 'gen3customgame', seed,
      p1: { name: 'P1', team: Teams.pack(teamFor(rng)) }, p2: { name: 'P2', team: Teams.pack(teamFor(rng)) } });
    for (let guard = 0; guard < 400 && !battle.ended; guard++) {
      for (const side of battle.sides) {
        const req = side.activeRequest;
        if (!req || side.isChoiceDone()) continue;
        const c = choose(req, rng);
        if (c) battle.choose(side.id, c);
      }
    }
    for (const line of battle.log) {
      if (!line.startsWith('|move|') || !line.includes('[from]')) continue;
      lines++;
      const { from, key } = shapeOf(line);
      const k = `${from}\t${key}`;
      const e = shapes.get(k) || { from, key, n: 0, example: line };
      e.n++;
      shapes.set(k, e);
    }
  }
  const rows = [...shapes.values()].sort((a, b) => (a.from < b.from ? -1 : a.from > b.from ? 1 : b.n - a.n));
  if (json) { console.log(JSON.stringify({ battles: n, lines, shapes: rows })); return; }
  console.log(`# ${n} battles, ${lines} |move| lines carrying [from], ${rows.length} distinct (source, shape)`);
  for (const r of rows) console.log(`${String(r.n).padStart(6)}  ${r.from.padEnd(22)}  ${r.key.padEnd(52)}  e.g. ${r.example}`);
}

main();
