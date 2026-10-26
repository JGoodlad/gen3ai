// GOLDEN GENERATOR — the TWO-TURN family (`gen3_twoturn_family_v1`, gen3 full parity B4).
//
// Razor Wind, Sky Attack, Skull Bash, Fly, Bounce, Dig and Dive (the spec and its Showdown
// citations: `harness/probe_twoturn_family.js`'s header). Every board is played by the
// synchronous `Battle` over several seeds; each row is the board, the decisions as played
// ([p1, p2], `null` = no choice — the port's `ScriptDecision`), the PRNG seed before and after,
// and the post-`|turn|1` omniscient lines (the `|split|` resolved to the omniscient copy; `|t:|`,
// `|debug|` and blank lines dropped). Solar Beam rides the cycle boards as the CONTROL (its own
// goldens predate this batch).
//
//   node src/rust_sim/harness/gen_twoturn_family_golden.js -> tests/vectors/twoturn_family_golden.txt
//   cargo test --release --test twoturn_family_golden_test
// env: TT_SEEDS (rows per board per variant, default 4), TT_OUT.
//
// The generator REFUSES to write a vector that misses a coverage floor (per move and per
// interaction); the Rust test re-asserts the same floors on the vector it reads.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.TT_SEEDS || '4', 10);
const OUT = process.env.TT_OUT || path.resolve(__dirname, '../tests/vectors/twoturn_family_golden.txt');
const ALL = ['razorwind', 'skyattack', 'skullbash', 'fly', 'bounce', 'dig', 'dive'];
const SEMI = ['fly', 'bounce', 'dig', 'dive'];
const NON_SEMI = ['razorwind', 'skyattack', 'skullbash'];

const set = (species, ability, moves, o = {}) =>
  `${species}|${species}|${o.item || ''}|${ability}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|M|||${o.lv && o.lv !== 100 ? o.lv : ''}|`;
const team = (...s) => s.join(']');
const FAST = '0,0,0,0,0,252';
const BYPASS = { fly: ['gust', 'twister', 'thunder', 'skyuppercut'], bounce: ['gust', 'twister', 'thunder', 'skyuppercut'], dig: ['earthquake', 'tackle', 'thunderbolt', 'splash'], dive: ['surf', 'whirlpool', 'tackle', 'splash'] };
const rot = (k) => (t) => (t % k) + 1;

// The request locks against a non-invulnerable charge: the foe (slower) lands Imprison / Disable /
// Encore / Torment on the user's charge turn; the strike turn then cants or strikes.
const lockBoard = (m, x) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST })), p2: team(set('Mew', 'Synchronize', [x, m, 'splash'])), turns: 8, want: (t) => [1, t % 3 === 0 ? 1 : 3] });

// Each board: (variant) -> { p1, p2, turns, want(t) -> [p1 slot, p2 slot] (1-based) }.
const BOARDS = {
  // The user FASTER against a Tackle wall: charge, strike (crit / flinch / par / miss), repeat.
  cycle: { vars: [...ALL, 'solarbeam'], f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST })), p2: team(set('Snorlax', 'Immunity', ['tackle', 'splash'], { evs: '252,0,252,0,4,0' })), turns: 6, want: () => [1, 1] }) },
  // The user SLOWER: the foe's Tackle / Thunderbolt meet the user before each of its actions.
  slow: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['tackle', 'thunderbolt', 'swift'], { evs: FAST })), turns: 6, want: (t) => [1, (t % 3) + 1] }) },
  // The bypassing moves (and their doubling) against each semi-invulnerable state.
  bypass: { vars: SEMI, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Mew', 'Synchronize', BYPASS[m], { evs: FAST })), turns: 8, want: (t) => [1, (Math.floor(t / 2) % 4) + 1] }) },
  // Status moves into the semi-invulnerable user (the user faster, so they meet it mid-charge).
  status: { vars: SEMI, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST }), set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Gengar', 'Levitate', ['thunderwave', 'roar', 'taunt', 'yawn'])), turns: 8, want: (t) => [1, (Math.floor(t / 2) % 4) + 1] }) },
  status2: { vars: SEMI, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST }), set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Gengar', 'Levitate', ['meanlook', 'curse', 'disable', 'leechseed'])), turns: 8, want: (t) => [1, (Math.floor(t / 2) % 4) + 1] }) },
  // Damaging families into the semi-invulnerable user: fixed damage, multi-hit, a lock-in, HJK.
  attacks: { vars: SEMI, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST })), p2: team(set('Hitmonlee', 'Limber', ['seismictoss', 'pinmissile', 'highjumpkick', 'swift'])), turns: 8, want: (t) => [1, (Math.floor(t / 2) % 4) + 1] }) },
  attacks2: { vars: SEMI, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: FAST })), p2: team(set('Houndoom', 'FlashFire', ['outrage', 'rollout', 'pursuit', 'beatup'])), turns: 10, want: (t) => [1, (Math.floor(t / 2) % 4) + 1] }) },
  // Field / side moves bypass; Perish Song misses the semi-invulnerable mon per mon.
  field: { vars: SEMI, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'agility'], { evs: FAST }), set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Weezing', 'Levitate', ['spikes', 'haze', 'perishsong', 'sandstorm'])), turns: 8, want: (t) => [t === 0 ? 2 : 1, (Math.floor(t / 2) % 4) + 1] }) },
  // Weather: Dig / Dive skip the chip while under; Fly / Bounce take it.
  weather: { vars: SEMI, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', [m === 'dive' ? 'hail' : 'sandstorm', 'splash'], { evs: FAST })), turns: 6, want: (t) => [t === 0 ? 2 : 1, t === 0 ? 1 : 2] }) },
  // Aborts on the strike turn: full paralysis, confusion, Truant.
  para: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['thunderwave', 'splash'], { evs: FAST })), turns: 8, want: (t) => [t === 0 ? 2 : 1, t === 0 ? 1 : 2] }) },
  confuse: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Gengar', 'Levitate', ['confuseray', 'splash'], { evs: FAST })), turns: 8, want: (t) => [t === 0 ? 2 : 1, t === 0 ? 1 : 2] }) },
  truant: { vars: ALL, f: (m) => ({ p1: team(set('Slaking', 'Truant', [m, 'splash'])), p2: team(set('Snorlax', 'Immunity', ['splash'], { evs: '252,0,252,0,4,0' })), turns: 6, want: () => [1, 1] }) },
  yawn: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Slowbro', 'Oblivious', ['yawn', 'splash'], { evs: FAST })), turns: 6, want: (t) => [t === 0 ? 2 : 1, t === 0 ? 1 : 2] }) },
  // Protect on the strike turn; the foe switching between charge and strike.
  protect: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Zapdos', 'Pressure', ['splash', 'protect'], { evs: FAST })), turns: 6, want: (t) => [1, t % 2 === 1 ? 2 : 1] }) },
  switch: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Zapdos', 'Pressure', ['splash']), set('Jolteon', 'VoltAbsorb', ['splash'])), turns: 6, want: (t) => [1, t % 2 === 1 ? 'switch' : 1] }) },
  // The foe (faster) explodes on the strike turn: the gen-3 faint cancels the strike.
  explode: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'], { lv: 70 }), set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Gengar', 'Levitate', ['sandstorm', 'explosion'], { evs: '0,252,0,0,0,252' }), set('Jolteon', 'VoltAbsorb', ['splash'])), turns: 5, want: (t) => [1, t === 1 ? 2 : 1] }) },
  // Future Sight resolving into a charge turn.
  futuresight: { vars: ALL, f: (m) => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Alakazam', 'Synchronize', ['futuresight', 'splash'])), turns: 6, want: (t) => [t === 2 ? 1 : 2, t === 0 ? 1 : 2] }) },
  // A mirror at equal speed (the residual tie group).
  mirror: { vars: ALL, f: (m) => ({ p1: team(set('Pidgeot', 'KeenEye', [m, 'splash'])), p2: team(set('Pidgeot', 'KeenEye', [m, 'splash'])), turns: 6, want: () => [1, 1] }) },
  // The request locks against a non-invulnerable charge: Imprison / Disable / Encore / Torment.
  lock_imprison: { vars: NON_SEMI, f: (m) => lockBoard(m, 'imprison') },
  lock_disable: { vars: NON_SEMI, f: (m) => lockBoard(m, 'disable') },
  lock_encore: { vars: NON_SEMI, f: (m) => lockBoard(m, 'encore') },
  lock_torment: { vars: NON_SEMI, f: (m) => lockBoard(m, 'torment') },
  // Skull Bash's Def boost to the +6 cap.
  skullcap: { vars: ['skullbash'], f: (m) => ({ p1: team(set('Blastoise', 'Torrent', [m, 'irondefense'])), p2: team(set('Snorlax', 'Immunity', ['splash'])), turns: 8, want: (t) => [t < 3 ? 2 : 1, 1] }) },
  // Choice Band (the lock after the strike) and the strike KO + replacement.
  choice: { vars: ALL, f: (m) => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { item: 'Choice Band', evs: FAST })), p2: team(set('Pikachu', 'Static', ['splash'], { lv: 30 }), set('Raichu', 'Static', ['splash'], { lv: 40 }), set('Snorlax', 'Immunity', ['splash'])), turns: 8, want: () => [1, 1] }) },
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) {
    const k = side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1;
    return `switch ${k}`;
  }
  if (want === 'switch') {
    const k = side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1;
    if (k > 0 && !req.active[0].trapped) return `switch ${k}`;
    want = 1;
  }
  const moves = req.active[0].moves;
  if (moves.length === 1) return 'move 1';
  if (moves[want - 1] && !moves[want - 1].disabled) return `move ${want}`;
  const k = moves.findIndex((x) => !x.disabled);
  return `move ${k < 0 ? 1 : k + 1}`;
}

function run(board, seed) {
  const battle = new Battle({ formatid: 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'P1', team: board.p1 });
  battle.setPlayer('p2', { name: 'P2', team: board.p2 });
  for (const [i, s] of [board.p1, board.p2].entries()) {
    const want = s.split(']').map((x) => x.split('|')[1]);
    const got = battle.sides[i].pokemon.map((p) => p.species.name);
    if (JSON.stringify(want) !== JSON.stringify(got)) throw new Error(`team ${i} unpacked as ${got}`);
  }
  const seedBefore = battle.prng.getSeed();
  const script = [];
  let t = 0;
  let guard = 0;
  while (!battle.ended && t < board.turns && guard++ < board.turns + 12) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    for (const i of [0, 1]) {
      if (d[i] === null) continue;
      if (!battle.choose(`p${i + 1}`, d[i])) {
        // A HIDDEN disable (Imprison's `maybeDisabled`) rejects the first pick and reveals itself
        // in an updated request; answer that request again (the played decision is the accepted one).
        d[i] = answer(battle.sides[i], [w1, w2][i]);
        if (!battle.choose(`p${i + 1}`, d[i])) throw new Error(`choice ${d[i]} rejected (p${i + 1}, t${t}) req=${JSON.stringify(battle.sides[i].activeRequest.active)}`);
      }
    }
    script.push(d);
    if (!forced) t++;
  }
  const omni = [];
  for (let i = 0; i < battle.log.length; i++) {
    const l = battle.log[i];
    if (l.startsWith('|split|')) { omni.push(battle.log[i + 1]); i += 2; continue; }
    omni.push(l);
  }
  const t1 = omni.indexOf('|turn|1');
  const lines = omni.slice(t1 + 1).filter((l) => !(l === '' || l.startsWith('|t:|') || l.startsWith('|debug|') || l.startsWith('|error|')));
  return { seedBefore, seedAfter: battle.prng.getSeed(), script, lines };
}

const rows = [];
for (const [name, b] of Object.entries(BOARDS)) {
  for (const m of b.vars) {
    for (let k = 0; k < N; k++) {
      const seed = [17 + k * 7, 29 + k * 13, 43 + k * 29, 59 + k * 51];
      const bd = b.f(m);
      let res;
      try { res = run(bd, seed); } catch (e) { throw new Error(`${name} ${m} seed ${k}: ${e.message}`); }
      rows.push({ name, move: m, p1: bd.p1, p2: bd.p2, ...res });
    }
  }
}

// --- coverage: per move and per interaction (the Rust test re-asserts these) ---
const has = (r, re) => r.lines.some((l) => re.test(l));
const n = (pred) => rows.filter(pred).length;
const NAME = { razorwind: 'Razor Wind', skyattack: 'Sky Attack', skullbash: 'Skull Bash', fly: 'Fly', bounce: 'Bounce', dig: 'Dig', dive: 'Dive' };
const cov = {};
// A strike turn LOST to a BeforeMove cant: p1 charged (`-prepare`) and its next action is a `cant`
// or a confusion self-hit instead of the `[from] lockedmove` strike.
const strikeAborted = (r) => {
  let up = false;
  for (const l of r.lines) {
    if (/^\|-prepare\|p1a: /.test(l)) up = true;
    else if (/^\|move\|p1a: .*\[from\] lockedmove/.test(l)) up = false;
    else if (up && (/^\|cant\|p1a: /.test(l) || /^\|-damage\|p1a: [^|]+\|[^|]+\|\[from\] confusion$/.test(l))) return true;
  }
  return false;
};
for (const m of ALL) {
  const nm = NAME[m];
  cov[`${m}.charge`] = n((r) => r.move === m && has(r, new RegExp(`^\\|-prepare\\|p1a: [^|]+\\|${nm}$`)));
  cov[`${m}.strike`] = n((r) => r.move === m && has(r, new RegExp(`^\\|move\\|p1a: [^|]+\\|${nm}\\|p2a: [^|]+\\|\\[from\\] lockedmove$`)));
  cov[`${m}.strikeAbort`] = n((r) => r.move === m && strikeAborted(r));
}
for (const m of ['skyattack', 'fly', 'bounce']) cov[`${m}.strikeMiss`] = n((r) => r.move === m && has(r, new RegExp(`\\|${NAME[m]}\\|p2a: [^|]+\\|\\[from\\] lockedmove\\|\\[miss\\]$`)));
for (const m of ['razorwind', 'skyattack']) cov[`${m}.crit`] = n((r) => r.move === m && has(r, /^\|-crit\|p2a: /));
cov['skyattack.flinch'] = n((r) => r.move === 'skyattack' && has(r, /^\|cant\|p2a: [^|]+\|flinch$/));
cov['bounce.par'] = n((r) => r.move === 'bounce' && has(r, /^\|-status\|p2a: [^|]+\|par$/));
cov['skullbash.boost'] = n((r) => r.move === 'skullbash' && has(r, /^\|-boost\|p1a: [^|]+\|def\|1$/));
cov['skullbash.capped'] = n((r) => r.name === 'skullcap' && has(r, /^\|-boost\|p1a: [^|]+\|def\|0$/));
for (const m of SEMI) {
  cov[`${m}.invulnMiss`] = n((r) => r.move === m && has(r, /^\|move\|p2a: [^|]+\|[^|]+\|p1a: [^|]+\|\[miss\]$/));
  cov[`${m}.bypassHit`] = n((r) => r.move === m && r.name === 'bypass' && has(r, new RegExp(`^\\|move\\|p2a: [^|]+\\|(${BYPASS[m].slice(0, m === 'dig' ? 1 : 2).map((x) => ({ gust: 'Gust', twister: 'Twister', thunder: 'Thunder', skyuppercut: 'Sky Uppercut', earthquake: 'Earthquake', surf: 'Surf', whirlpool: 'Whirlpool' })[x]).join('|')})\\|p1a: [^|]+$`)));
  cov[`${m}.perishMiss`] = n((r) => r.move === m && has(r, /^\|-miss\|p2a: [^|]+\|p1a: /) && r.lines.some((l) => l.includes('Perish Song')));
  cov[`${m}.futureMiss`] = n((r) => r.move === m && r.name === 'futuresight' && r.lines.some((l, i) => l.startsWith('|-end|p1a:') && l.endsWith('move: Future Sight') && /^\|-miss\|/.test(r.lines[i + 1] || '')));
}
for (const m of ['dig', 'dive']) cov[`${m}.weatherSkip`] = n((r) => r.move === m && r.name === 'weather' && r.lines.some((l, i) => l.startsWith(`|-prepare|p1a:`) && !r.lines.slice(i, r.lines.indexOf('|upkeep', i)).some((x) => /^\|-damage\|p1a: [^|]+\|[^|]+\|\[from\] (Sandstorm|Hail)$/.test(x))));
cov['cancelledStrike'] = n((r) => r.name === 'explode' && has(r, /^\|faint\|p2a: Gengar$/) && r.script.some((d) => d[0] === null));
cov['hjkCrash'] = n((r) => r.name === 'attacks' && has(r, /^\|move\|p2a: [^|]+\|High Jump Kick\|p1a: [^|]+\|\[miss\]$/));
cov['imprisonCant'] = n((r) => r.name.startsWith('lock') && has(r, /^\|cant\|p1a: [^|]+\|move: Imprison\|/));
cov['disableCant'] = n((r) => r.name.startsWith('lock') && has(r, /^\|cant\|p1a: [^|]+\|Disable\|/));
cov['protectStrike'] = n((r) => r.name === 'protect' && has(r, /\[from\] lockedmove$/) && has(r, /^\|-activate\|p2a: [^|]+\|Protect$/));
cov['strikeKO'] = n((r) => r.name === 'choice' && has(r, /^\|faint\|p2a: /));
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# TWO-TURN FAMILY GOLDEN (gen3_twoturn_family_v1, gen3 full parity B4)',
  `# generated by harness/gen_twoturn_family_golden.js  rows/board/variant=${N}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the decisions as played ([p1, p2], null = no choice), the sim SEED_BEFORE /',
  '# SEED_AFTER, and the post-|turn|1 omniscient lines (|t:| / |debug| dropped). gen3customgame.',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows`);
console.log(JSON.stringify(cov));
