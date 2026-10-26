// GOLDEN GENERATOR — gen3 full parity B6 (`gen3_b6_moves_v1`): SWAGGER, FLATTER, FORESIGHT, ODOR
// SLEUTH, MAGIC COAT. The settled spec is the header of `harness/probe_b6_moves.js`. Every
// reflectable gen-3 move gets a Magic Coat board (bounce turn, a coat-less control turn, a second
// bounce), plus the interaction boards: the bounced move meeting the original user's Substitute,
// type immunity, ability, Safeguard and status; Soundproof / Ground / Substitute on the coat
// holder; Snatch; a coat mirror (the residual tie); a drag; Encore and Conversion 2 reading the
// holder's last move; a Sleep-Talk-called move; Sleep Clause in gen3ou. Every set pins its GENDER
// and stays at or under 510 EVs.
//
//   node src/rust_sim/harness/gen_b6_moves_golden.js -> tests/vectors/b6_moves_golden.txt
//   cargo test --test b6_moves_golden_test
// env: B6_SEEDS (rows per board, default 6), B6_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Dex } = require(path.join(PS, 'dist/sim'));

const N = parseInt(process.env.B6_SEEDS || '6', 10);
const OUT = process.env.B6_OUT || path.resolve(__dirname, '../tests/vectors/b6_moves_golden.txt');

const EV = '85,85,85,85,85,85';
const FAST = '0,85,85,85,0,252';
const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || EV}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const seq = (p1, p2) => (t) => [p1[Math.min(t, p1.length - 1)], p2[Math.min(t, p2.length - 1)]];

// ---- SWAGGER / FLATTER --------------------------------------------------------------------------
const SW = (o = {}) => set('Nidoking', o.item || 'Leftovers', 'PoisonPoint', ['swagger', 'flatter', 'splash', 'tackle'], o);
const LAX = (moves, o = {}) => set('Snorlax', o.item || 'Leftovers', o.ab || 'ThickFat', moves, o);
const BOARDS = {
  sw_plain: { p1: SW(), p2: LAX(['splash', 'tackle']), turns: 4, want: seq(['move 1', 'move 3'], ['move 2']) },
  fl_plain: { p1: SW(), p2: LAX(['splash', 'tackle']), turns: 4, want: seq(['move 2', 'move 3'], ['move 2']) },
  sw_twice: { p1: SW(), p2: LAX(['splash']), turns: 3, want: M(1, 1) },
  fl_twice: { p1: SW(), p2: LAX(['splash']), turns: 3, want: M(2, 1) },
  sw_cap: { p1: SW(), p2: LAX(['bellydrum', 'splash']), turns: 4, want: seq(['move 3', 'move 1'], ['move 1', 'move 2']) },
  sw_plus5: { p1: SW(), p2: LAX(['swordsdance', 'howl', 'splash']), turns: 5,
    want: seq(['move 3', 'move 3', 'move 3', 'move 1'], ['move 1', 'move 1', 'move 2', 'move 3']) },
  fl_cap: { p1: SW(), p2: LAX(['tailglow', 'splash']), turns: 6,
    want: seq(['move 3', 'move 3', 'move 3', 'move 2'], ['move 1', 'move 1', 'move 1', 'move 2']) },
  sw_owntempo: { p1: SW(), p2: set('Slowbro', 'Leftovers', 'OwnTempo', ['bellydrum', 'splash']), turns: 4,
    want: seq(['move 1', 'move 3', 'move 1'], ['move 2', 'move 1', 'move 2']) },
  sw_sub: { p1: SW(), p2: LAX(['substitute', 'splash']), turns: 3, want: seq(['move 3', 'move 1', 'move 2'], ['move 1', 'move 2']) },
  sw_safeguard: { p1: SW(), p2: LAX(['safeguard', 'bellydrum', 'splash']), turns: 4,
    want: seq(['move 3', 'move 1', 'move 3', 'move 1'], ['move 1', 'move 3', 'move 2', 'move 3']) },
  sw_protect: { p1: SW(), p2: LAX(['protect', 'splash'], { evs: FAST }), turns: 4, want: seq(['move 1'], ['move 1', 'move 2', 'move 1', 'move 2']) },
  sw_evasive: { p1: SW(), p2: LAX(['doubleteam', 'splash'], { evs: FAST }), turns: 7,
    want: seq(['move 3', 'move 3', 'move 3', 'move 1', 'move 2', 'move 1'], ['move 1', 'move 1', 'move 1', 'move 2']) },
  sw_lum: { p1: SW(), p2: LAX(['splash'], { item: 'Lum Berry' }), turns: 2, want: M(1, 1) },
  sw_persim: { p1: SW(), p2: LAX(['splash'], { item: 'Persim Berry' }), turns: 2, want: M(1, 1) },
  sw_clearbody: { p1: SW(), p2: set('Metagross', 'Leftovers', 'ClearBody', ['splash'], { g: 'N' }), turns: 2, want: M(1, 1) },
  sw_ghost: { p1: SW(), p2: set('Gengar', 'Leftovers', 'Levitate', ['splash']), turns: 2, want: M(1, 1) },
};

// ---- FORESIGHT / ODOR SLEUTH --------------------------------------------------------------------
const FS = (moves = ['foresight', 'odorsleuth', 'crosschop', 'bodyslam'], o = {}) => set('Machamp', 'Leftovers', 'Guts', moves, o);
const GEN = (moves, o = {}) => set('Gengar', 'Leftovers', 'Levitate', moves, o);
Object.assign(BOARDS, {
  fs_ghost: { p1: FS(), p2: GEN(['splash']), turns: 4, want: seq(['move 1', 'move 3', 'move 4', 'move 1'], ['move 1']) },
  fs_control: { p1: FS(), p2: GEN(['splash']), turns: 2, want: seq(['move 3', 'move 4'], ['move 1']) },
  fs_odor: { p1: FS(), p2: GEN(['splash']), turns: 3, want: seq(['move 2', 'move 4', 'move 1'], ['move 1']) },
  fs_sub: { p1: FS(), p2: GEN(['substitute', 'splash']), turns: 3, want: seq(['move 4', 'move 1', 'move 3'], ['move 1', 'move 2']) },
  fs_protect: { p1: FS(), p2: GEN(['protect', 'splash'], { evs: FAST }), turns: 3, want: seq(['move 1'], ['move 1', 'move 2', 'move 1']) },
  fs_evasion_ghost: { p1: FS(), p2: GEN(['doubleteam', 'splash'], { evs: FAST }), turns: 7,
    want: seq(['move 4', 'move 4', 'move 4', 'move 1', 'move 3'], ['move 1', 'move 1', 'move 1', 'move 2']) },
  fs_evasion_normal: { p1: FS(), p2: LAX(['doubleteam', 'splash']), turns: 7,
    want: seq(['move 2', 'move 2', 'move 2', 'move 2', 'move 3'], ['move 1', 'move 1', 'move 1', 'move 1', 'move 2']) },
  fs_evasion_control: { p1: FS(), p2: LAX(['doubleteam', 'splash']), turns: 6,
    want: seq(['move 4', 'move 4', 'move 4', 'move 3'], ['move 1', 'move 1', 'move 1', 'move 2']) },
  fs_neg_evasion: { p1: FS(['foresight', 'sweetscent', 'crosschop', 'splash']), p2: LAX(['splash']), turns: 4,
    want: seq(['move 2', 'move 1', 'move 3'], ['move 1']) },
  fs_acc_drop: { p1: FS(), p2: LAX(['sandattack', 'doubleteam', 'splash']), turns: 6,
    want: seq(['move 1', 'move 1', 'move 1', 'move 3'], ['move 1', 'move 2', 'move 3']) },
  fs_glare: { p1: set('Arbok', 'Leftovers', 'Intimidate', ['foresight', 'glare', 'splash']), p2: GEN(['splash']), turns: 3,
    want: seq(['move 2', 'move 1', 'move 2'], ['move 1']) },
  fs_switch: { p1: FS(), p2: team(GEN(['splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], { g: 'F' })), turns: 4,
    want: seq(['move 1', 'move 4', 'move 4', 'move 4'], ['move 1', 'switch 2', 'switch 2', 'move 1']) },
  fs_baton: { p1: FS(), p2: team(GEN(['splash', 'batonpass']), set('Misdreavus', 'Leftovers', 'Levitate', ['splash'], { g: 'F' })), turns: 4,
    want: seq(['move 1', 'move 4', 'move 4'], ['move 1', 'move 2', 'move 1']) },
  fs_seismic: { p1: FS(['foresight', 'seismictoss', 'counter', 'splash']), p2: GEN(['splash', 'lick']), turns: 4,
    want: seq(['move 3', 'move 1', 'move 2', 'move 3'], ['move 2', 'move 1', 'move 1', 'move 2']) },
  fs_jumpkick: { p1: set('Hitmonlee', 'Leftovers', 'Limber', ['foresight', 'hijumpkick', 'doublekick', 'splash']),
    p2: GEN(['protect', 'splash'], { evs: FAST }), turns: 5,
    want: seq(['move 2', 'move 1', 'move 2', 'move 3', 'move 2'], ['move 1', 'move 2', 'move 1', 'move 2', 'move 2']) },
  fs_wonderguard: { p1: FS(), p2: set('Shedinja', 'Leftovers', 'WonderGuard', ['splash'], { g: 'N' }), turns: 3,
    want: seq(['move 1', 'move 4', 'move 3'], ['move 1']) },
  fs_misdreavus: { p1: FS(['odorsleuth', 'return', 'machpunch', 'splash']), p2: set('Misdreavus', 'Leftovers', 'Levitate', ['splash'], { g: 'F' }), turns: 4,
    want: seq(['move 2', 'move 1', 'move 2', 'move 3'], ['move 1']) },
});

// ---- MAGIC COAT ---------------------------------------------------------------------------------
const HOLDER = (moves = ['magiccoat', 'splash'], o = {}) => set(o.sp || 'Alakazam', o.item || 'Leftovers', o.ab || 'InnerFocus', moves, o);
const USER = (moves, o = {}) => set(o.sp || 'Snorlax', o.item || 'Leftovers', o.ab || 'ThickFat', moves, { g: 'F', ...o });
const d3 = Dex.mod('gen3');
const REFLECTABLE = d3.moves.all().filter((m) => m.exists && !m.isNonstandard && m.gen <= 3 && m.flags.reflectable).map((m) => m.id).sort();
for (const mv of REFLECTABLE) {
  // turn 0: a bounce; turn 1: the coat-less control; turn 2: a second bounce (into the first's effect).
  BOARDS[`mc_${mv}`] = { p1: HOLDER(), p2: USER([mv, 'splash']), turns: 3, want: seq(['move 1', 'move 2', 'move 1'], ['move 1']) };
}
Object.assign(BOARDS, {
  mc_alone: { p1: HOLDER(), p2: USER(['splash']), turns: 2, want: M(1, 1) },
  mc_mirror: { p1: HOLDER(), p2: HOLDER(['magiccoat', 'splash'], { g: 'F' }), turns: 2, want: M(1, 1) },
  mc_mirror_control: { p1: HOLDER(), p2: HOLDER(['magiccoat', 'splash'], { g: 'F' }), turns: 2, want: M(2, 2) },
  mc_user_sub: { p1: HOLDER(), p2: USER(['substitute', 'thunderwave', 'swagger', 'leechseed']), turns: 4,
    want: seq(['move 2', 'move 1'], ['move 1', 'move 2', 'move 3', 'move 4']) },
  mc_holder_sub: { p1: HOLDER(['substitute', 'magiccoat', 'splash']), p2: USER(['splash', 'thunderwave', 'swagger']), turns: 3,
    want: seq(['move 1', 'move 2'], ['move 1', 'move 2', 'move 3']) },
  mc_user_ground: { p1: HOLDER(), p2: USER(['thunderwave', 'splash'], { sp: 'Marowak', ab: 'RockHead', g: 'M' }), turns: 2, want: M(1, 1) },
  mc_holder_ground: { p1: HOLDER(['magiccoat', 'splash'], { sp: 'Marowak', ab: 'RockHead' }), p2: USER(['thunderwave', 'splash']), turns: 2, want: M(1, 1) },
  mc_user_soundproof: { p1: HOLDER(), p2: USER(['supersonic', 'sing', 'growl', 'screech'], { sp: 'Electrode', ab: 'Soundproof', g: 'N' }), turns: 4,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3', 'move 4']) },
  mc_holder_soundproof: { p1: HOLDER(['magiccoat', 'splash'], { sp: 'Electrode', ab: 'Soundproof', g: 'N' }), p2: USER(['supersonic', 'sing']), turns: 2,
    want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_user_owntempo: { p1: HOLDER(), p2: USER(['confuseray', 'swagger', 'flatter'], { sp: 'Slowbro', ab: 'OwnTempo' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_clearbody: { p1: HOLDER(), p2: USER(['growl', 'screech', 'tickle'], { sp: 'Metagross', ab: 'ClearBody', g: 'N' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_hypercutter: { p1: HOLDER(), p2: USER(['charm', 'leer', 'sandattack'], { sp: 'Kingler', ab: 'HyperCutter', g: 'M' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_keeneye: { p1: HOLDER(), p2: USER(['sandattack', 'flash', 'growl'], { sp: 'Pidgeot', ab: 'KeenEye', g: 'M' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_grass: { p1: HOLDER(), p2: USER(['leechseed', 'splash'], { sp: 'Venusaur', ab: 'Overgrow' }), turns: 2, want: M(1, 1) },
  mc_user_poison: { p1: HOLDER(), p2: USER(['toxic', 'poisonpowder', 'poisongas'], { sp: 'Muk', ab: 'StickyHold' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_insomnia: { p1: HOLDER(), p2: USER(['spore', 'yawn', 'hypnosis'], { sp: 'Hypno', ab: 'Insomnia' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_limber: { p1: HOLDER(), p2: USER(['thunderwave', 'stunspore', 'glare'], { sp: 'Persian', ab: 'Limber' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_synchronize: { p1: HOLDER(), p2: USER(['thunderwave', 'toxic', 'willowisp'], { sp: 'Espeon', ab: 'Synchronize' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_holder_synchronize: { p1: HOLDER(['magiccoat', 'splash'], { ab: 'Synchronize' }), p2: USER(['thunderwave', 'toxic']), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_user_ghost: { p1: HOLDER(), p2: USER(['glare', 'meanlook'], { sp: 'Gengar', ab: 'Levitate', g: 'M' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_attract_same: { p1: HOLDER(), p2: USER(['attract', 'splash'], { g: 'M' }), turns: 2, want: M(1, 1) },
  mc_attract_genderless: { p1: HOLDER(), p2: USER(['attract', 'splash'], { sp: 'Magneton', ab: 'MagnetPull', g: 'N' }), turns: 2, want: M(1, 1) },
  mc_attract_oblivious: { p1: HOLDER(), p2: USER(['attract', 'splash'], { sp: 'Slowbro', ab: 'Oblivious' }), turns: 2, want: M(1, 1) },
  mc_mentalherb: { p1: HOLDER(), p2: USER(['attract', 'splash'], { item: 'Mental Herb' }), turns: 2, want: M(1, 1) },
  mc_user_lum: { p1: HOLDER(), p2: USER(['swagger', 'thunderwave', 'spore'], { item: 'Lum Berry' }), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_safeguard: { p1: HOLDER(), p2: USER(['safeguard', 'thunderwave', 'swagger', 'confuseray']), turns: 4,
    want: seq(['move 2', 'move 1'], ['move 1', 'move 2', 'move 3', 'move 4']) },
  mc_user_statused: { p1: HOLDER(['magiccoat', 'thunderwave', 'splash']), p2: USER(['thunderwave', 'toxic', 'yawn']), turns: 5,
    want: seq(['move 2', 'move 1'], ['move 3', 'move 1', 'move 2', 'move 3']) },
  // A STATUSED coat holder still bounces a Yawn: the coat (TryHit) precedes yawn's own onTryHit.
  mc_holder_statused: { p1: HOLDER(['magiccoat', 'splash']), p2: USER(['willowisp', 'yawn']), turns: 3,
    want: seq(['move 2', 'move 1'], ['move 1', 'move 2']) },
  mc_user_confused: { p1: HOLDER(['magiccoat', 'confuseray', 'splash']), p2: USER(['swagger', 'confuseray', 'flatter']), turns: 4,
    want: seq(['move 2', 'move 1'], ['move 1', 'move 1', 'move 2', 'move 3']) },
  mc_user_seeded: { p1: HOLDER(['magiccoat', 'leechseed', 'splash']), p2: USER(['leechseed', 'meanlook']), turns: 4,
    want: seq(['move 2', 'move 1'], ['move 2', 'move 1', 'move 2', 'move 1']) },
  mc_user_capped: { p1: HOLDER(), p2: USER(['bellydrum', 'swagger', 'growl']), turns: 3, want: seq(['move 2', 'move 1'], ['move 1', 'move 2', 'move 3']) },
  mc_user_floor: { p1: HOLDER(['magiccoat', 'charm', 'splash']), p2: USER(['charm', 'splash']), turns: 5,
    want: seq(['move 2', 'move 2', 'move 2', 'move 1'], ['move 2', 'move 2', 'move 2', 'move 1']) },
  mc_snatch: { p1: HOLDER(['magiccoat', 'swordsdance']), p2: USER(['snatch', 'thunderwave']), turns: 3,
    want: seq(['move 1', 'move 2', 'move 1'], ['move 1', 'move 1', 'move 2']) },
  mc_expired: { p1: HOLDER(), p2: USER(['splash', 'thunderwave']), turns: 2, want: seq(['move 1', 'move 2'], ['move 1', 'move 2']) },
  mc_miss: { seeds: 24, p1: HOLDER(), p2: USER(['hypnosis', 'sing']), turns: 2, want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_evasive_user: { p1: HOLDER(['magiccoat', 'splash']), p2: USER(['doubleteam', 'thunderwave'], { evs: FAST }), turns: 6,
    want: seq(['move 2', 'move 2', 'move 2', 'move 1'], ['move 1', 'move 1', 'move 1', 'move 2']) },
  mc_foresight_user: { p1: HOLDER(['magiccoat', 'doubleteam']), p2: USER(['foresight', 'thunderwave']), turns: 4,
    want: seq(['move 2', 'move 2', 'move 1'], ['move 2', 'move 1', 'move 2']) },
  mc_taunt: { p1: HOLDER(), p2: USER(['taunt', 'teeterdance']), turns: 2, want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_encore: { p1: HOLDER(['magiccoat', 'splash'], { sp: 'Snorlax', ab: 'ThickFat' }), p2: USER(['growl', 'encore'], { sp: 'Alakazam', ab: 'InnerFocus' }), turns: 4,
    want: seq(['move 1', 'move 2'], ['move 1', 'move 2', 'move 1']) },
  // Limber keeps the user fast (a paralysed user would let the holder's next move overwrite
  // `lastMoveUsed` first); Conversion 2 then reads the BOUNCED Thunder Wave's Electric.
  mc_conversion2: { seeds: 12, p1: HOLDER(['magiccoat', 'splash'], { sp: 'Snorlax', ab: 'ThickFat' }), p2: USER(['thunderwave', 'conversion2', 'splash'], { sp: 'Porygon2', ab: 'Limber', g: 'N', evs: FAST }), turns: 3,
    want: seq(['move 1', 'move 2'], ['move 1', 'move 2', 'move 3']) },
  mc_sleeptalk: { seeds: 12, p1: HOLDER(['magiccoat', 'spore', 'splash']), p2: USER(['sleeptalk', 'thunderwave']), turns: 4,
    want: seq(['move 2', 'move 1'], ['move 2', 'move 1']) },
  mc_drag: { p1: team(HOLDER(), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], { g: 'F' })), p2: USER(['roar', 'thunderwave']), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_pressure: { p1: HOLDER(['magiccoat', 'splash'], { sp: 'Zapdos', ab: 'Pressure', g: 'N' }), p2: USER(['thunderwave', 'splash']), turns: 3,
    want: seq(['move 1'], ['move 1', 'move 2']) },
  mc_sleepclause: { format: 'gen3ou', p1: HOLDER(['magiccoat', 'spore', 'splash']),
    p2: team(USER(['spore', 'splash']), set('Lapras', 'Leftovers', 'ShellArmor', ['spore', 'splash'], { g: 'F' })), turns: 4,
    want: seq(['move 2', 'move 3', 'move 1', 'move 1'], ['move 2', 'switch 2', 'move 1', 'move 1']) },
});

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  const act = req.active[0];
  if (act.moves.length === 1 && want.startsWith('move')) return 'move 1';
  if (want.startsWith('switch') && act.trapped) return 'move 1';
  if (want.startsWith('move')) {
    const k = parseInt(want.split(' ')[1], 10) - 1;
    if (act.moves[k] && act.moves[k].disabled) return `move ${act.moves.findIndex((m) => !m.disabled) + 1}`;
  }
  return want;
}

function run(board, seed) {
  const battle = new Battle({ formatid: board.format || 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'P1', team: board.p1 });
  battle.setPlayer('p2', { name: 'P2', team: board.p2 });
  for (const [i, s] of [board.p1, board.p2].entries()) {
    const want = s.split(']').map((x) => x.split('|')[1]);
    const got = battle.sides[i].pokemon.map((p) => p.species.name);
    if (JSON.stringify(want) !== JSON.stringify(got)) throw new Error(`team ${i} unpacked as ${got}`);
    for (const p of battle.sides[i].pokemon) {
      const evs = Object.values(p.set.evs).reduce((a, b) => a + b, 0);
      if (evs > 510) throw new Error(`${p.name} carries ${evs} EVs`);
    }
  }
  const seedBefore = battle.prng.getSeed();
  const script = [];
  let t = 0;
  while (!battle.ended && t < board.turns) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    if (d[0] !== null) battle.choose('p1', d[0]);
    if (d[1] !== null) battle.choose('p2', d[1]);
    script.push(d);
    if (!forced) t++;
  }
  const omni = [];
  for (let i = 0; i < battle.log.length; i++) {
    const l = battle.log[i];
    if (l.startsWith('|split|')) { omni.push(battle.log[i + 1]); i += 2; continue; }
    omni.push(l);
  }
  const drop = (l) => l === '' || l.startsWith('|t:|') || l.startsWith('|debug|') || l.startsWith('|error|');
  const g = omni.indexOf('|gametype|singles');
  const t1 = omni.indexOf('|turn|1');
  return {
    seedBefore, seedAfter: battle.prng.getSeed(), script,
    framing: omni.slice(g, t1 + 1).filter((l) => !drop(l)),
    lines: omni.slice(t1 + 1).filter((l) => !drop(l)),
  };
}

const rows = [];
for (const [name, b] of Object.entries(BOARDS)) {
  for (let k = 0; k < (b.seeds || N); k++) {
    const seed = [41 + k * 7, 59 + k * 13, 71 + k * 29, 83 + k * 51];
    rows.push({ name, format: b.format || 'gen3customgame', p1: b.p1, p2: b.p2, ...run(b, seed) });
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const re = (r, rx) => r.lines.some((l) => rx.test(l));
const cov = {
  swaggerBoost: n((r) => has(r, '|Swagger|') && re(r, /^\|-boost\|p2a: [^|]+\|atk\|2$/)),
  flatterBoost: n((r) => has(r, '|Flatter|') && re(r, /^\|-boost\|p2a: [^|]+\|spa\|1$/)),
  boostDelta0: n((r) => re(r, /^\|-boost\|[^|]+\|(atk|spa)\|0$/)),
  boostDelta1: n((r) => r.name === 'sw_plus5' && re(r, /^\|-boost\|p2a: Snorlax\|atk\|1$/)),
  swaggerFail: n((r) => r.name === 'sw_cap' && has(r, '|-fail|p1a: Nidoking')),
  swaggerOwnTempo: n((r) => r.name === 'sw_owntempo' && has(r, '|confusion|[from] ability: Own Tempo')),
  swaggerSafeguard: n((r) => r.name === 'sw_safeguard' && has(r, '|move: Safeguard')),
  swaggerSub: n((r) => r.name === 'sw_sub' && has(r, '|-fail|p1a: Nidoking')),
  swaggerProtect: n((r) => r.name === 'sw_protect' && has(r, '|-activate|p2a: Snorlax|Protect')),
  swaggerMiss: n((r) => r.name === 'sw_evasive' && has(r, '|-miss|p1a: Nidoking')),
  swaggerBerry: n((r) => (r.name === 'sw_lum' || r.name === 'sw_persim') && has(r, '|-enditem|p2a: Snorlax|')),
  foresightStart: n((r) => has(r, '|Foresight') && has(r, '|-start|')),
  odorSleuthStart: n((r) => has(r, '|Odor Sleuth|') && re(r, /^\|-start\|[^|]+\|Foresight$/)),
  foresightFail: n((r) => r.name.startsWith('fs_') && re(r, /^\|move\|p1a: [^|]+\|(Foresight|Odor Sleuth)\|\|\[still\]$/)),
  foresightMiss: n((r) => r.name.startsWith('fs_') && re(r, /^\|move\|p1a: [^|]+\|Foresight\|p2a: [^|]+\|\[miss\]$/)),
  identifiedHit: n((r) => (r.name === 'fs_ghost' || r.name === 'fs_odor') && re(r, /^\|-damage\|p2a: Gengar\|\d+\/\d+$/)),
  identifiedResisted: n((r) => r.name === 'fs_ghost' && has(r, '|-resisted|p2a: Gengar')),
  ghostImmuneControl: n((r) => r.name === 'fs_control' && has(r, '|-immune|p2a: Gengar')),
  glareIdentified: n((r) => r.name === 'fs_glare' && has(r, '|-status|p2a: Gengar|par')),
  foresightCleared: n((r) => r.name === 'fs_switch' && has(r, '|-immune|p2a: Gengar')),
  foresightNotPassed: n((r) => r.name === 'fs_baton' && has(r, '|-immune|p2a: Misdreavus')),
  identifiedCrash: n((r) => r.name === 'fs_jumpkick' && has(r, '|-damage|p1a: Hitmonlee')),
  identifiedCounter: n((r) => r.name === 'fs_seismic' && re(r, /^\|move\|p1a: Machamp\|Counter\|p2a: Gengar$/) && has(r, '|-damage|p2a: Gengar')),
  wonderGuard: n((r) => r.name === 'fs_wonderguard' && has(r, '[from] ability: Wonder Guard')),
  foresightProtect: n((r) => r.name === 'fs_protect' && has(r, '|-activate|p2a: Gengar|Protect')),
  coatCast: n((r) => has(r, '|-singleturn|p1a: ') && has(r, '|move: Magic Coat')),
  bounced: n((r) => has(r, '|[from] Magic Coat')),
  bouncedStill: n((r) => re(r, /\|\[from\] Magic Coat\|\[still\]$/)),
  bouncedMiss: n((r) => re(r, /\|\[from\] Magic Coat\|\[miss\]$/)),
  bouncedImmune: n((r) => r.name.startsWith('mc_user_') && has(r, '[from] Magic Coat') && has(r, '|-immune|p2a:')),
  coatOriginalMiss: n((r) => r.name === 'mc_miss' && has(r, '|-miss|p2a: Snorlax|p1a: Alakazam')),
  sleepClause: n((r) => r.name === 'mc_sleepclause' && has(r, 'Sleep Clause Mod activated') && has(r, '[from] Magic Coat')),
  conversion2: n((r) => r.name === 'mc_conversion2' && has(r, '|typechange|')),
  encoreCoat: n((r) => r.name === 'mc_encore' && has(r, '|-start|p1a: Snorlax|Encore')),
  sleepTalkBounce: n((r) => r.name === 'mc_sleeptalk' && has(r, '[from] Sleep Talk') && has(r, '[from] Magic Coat')),
  snatchBoard: n((r) => r.name === 'mc_snatch' && has(r, '[from] Snatch')),
  dragged: n((r) => r.name === 'mc_drag' && has(r, '|drag|p1a: Blissey')),
  holderStatusedYawn: n((r) => r.name === 'mc_holder_statused' && has(r, '|-status|p1a: Alakazam|brn') && has(r, '|Yawn|p2a: Snorlax|[from] Magic Coat')),
};
for (const mv of REFLECTABLE) {
  const name = d3.moves.get(mv).name;
  cov[`bounce_${mv}`] = n((r) => r.name === `mc_${mv}` && has(r, `|${name}|p2a: Snorlax|[from] Magic Coat`));
}
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B6 MOVES GOLDEN (gen3_b6_moves_v1): Swagger, Flatter, Foresight, Odor Sleuth, Magic Coat',
  `# generated by harness/gen_b6_moves_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}  reflectable=${REFLECTABLE.length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, its format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
