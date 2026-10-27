// GOLDEN GENERATOR — the B11 moves (`gen3_b11_moves_v1`, gen3 full parity B11): Metronome, Mirror Move,
// Assist, Nature Power and Sketch, and the one calling path they share with Sleep Talk. The settled spec
// is `harness/probe_b11_moves.js`'s header. Each board runs over seeds; each row records the format, the
// decisions as played ([p1, p2], null = no choice), the sim SEED_BEFORE / SEED_AFTER, the FRAMING window
// and the post-`|turn|1` omniscient lines (the `|split|` pairs resolved to their omniscient half), the
// lines captured AS SENT (each decision's new lines frozen when the sim streams them).
//
// Every row is a REAL sim battle on a real seed (no forced sample). A board may carry `need(row)`: the
// generator then walks seeds until that many rows satisfy it (the interaction boards: a Metronome that
// called a lock, then Encore / Disable / Spite / Mimic). Metronome's pool includes the B10 moves
// Teleport, Camouflage and Role Play; their rows are kept (`metronomeCallsB10`).
//
//   node src/rust_sim/harness/gen_b11_moves_golden.js -> tests/vectors/b11_moves_golden.txt
//   cargo test --test b11_moves_golden_test
// env: B11_SEEDS (rows per board, default 8), B11_OUT, B11_NOFLOOR.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.B11_SEEDS || '8', 10);
const OUT = process.env.B11_OUT || path.resolve(__dirname, '../tests/vectors/b11_moves_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|${o.ivs || ''}||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash', 'softboiled'], { g: 'F' });
const BENCH2 = set('Snorlax', 'Leftovers', 'ThickFat', ['bodyslam', 'rest'], { g: 'M' });
// want(t) -> [p1, p2] for the t-th regular turn: 'move K' / 'switch K'. The driver answers a forced
// switch itself, plays a locked request's only option, and falls back to the first enabled move when
// the wanted one is disabled (0 PP, Choice lock, Encore, Disable, Taunt).
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const seq = (rows) => (t) => rows[Math.min(t, rows.length - 1)].map((x) => (typeof x === 'number' ? `move ${x}` : x));
const cyc = (rows) => (t) => rows[t % rows.length].map((x) => (typeof x === 'number' ? `move ${x}` : x));
const CL = (moves, o = {}) => set('Clefable', o.item || '', 'CuteCharm', moves, { g: 'F', ...o });
const CL2 = (moves, o = {}) => set('Wigglytuff', o.item || '', 'CuteCharm', moves, { g: 'M', ...o });
const PG = (moves, o = {}) => set('Pidgeot', o.item || '', 'KeenEye', moves, o);
const SM = (moves, o = {}) => set('Smeargle', o.item || '', 'OwnTempo', moves, o);
const PER = (moves, o = {}) => set('Persian', o.item || '', 'Limber', moves, o);
const SN = (moves, o = {}) => set('Snorlax', o.item || '', 'ThickFat', moves, o);
const LU = (moves, o = {}) => set('Ludicolo', o.item || '', 'SwiftSwim', moves, o);

// A called lock: the Metronome line is followed by a lock-starting callee.
const LOCK_CALLEES = ['Outrage', 'Thrash', 'Petal Dance', 'Rollout', 'Ice Ball', 'Uproar', 'Bide',
  'Solar Beam', 'Razor Wind', 'Sky Attack', 'Skull Bash', 'Fly', 'Bounce', 'Dig', 'Dive'];
const calledLock = (r, side) => r.lines.some((l) => l.startsWith(`|move|${side}a: `) && l.includes('[from] Metronome') && LOCK_CALLEES.some((c) => l.includes(`|${c}|`)));
const calledLockOnTurn1 = (r, side) => {
  const t2 = r.lines.indexOf('|turn|2');
  return r.lines.slice(0, t2 < 0 ? r.lines.length : t2).some((l) => l.startsWith(`|move|${side}a: `) && l.includes('[from] Metronome') && LOCK_CALLEES.some((c) => l.includes(`|${c}|`)));
};

const P2C = set('Porygon2', '', 'Trace', ['conversion2', 'recover'], { nature: 'Brave', evs: '0,0,0,0,0,0' });
const ASSIST_TEAM = (lead) => team(lead,
  set('Aerodactyl', '', 'RockHead', ['fly', 'hyperbeam', 'bonerush', 'seismictoss']),
  set('Venusaur', '', 'Overgrow', ['solarbeam', 'toxic', 'bide', 'outrage']),
  set('Jolteon', '', 'VoltAbsorb', ['thunderwave', 'uproar', 'hiddenpowerice', 'nightshade'], { ivs: '31,31,31,31,31,31' }));
const ASSIST_TEAM2 = (lead) => team(lead,
  set('Golem', '', 'Sturdy', ['rollout', 'explosion', 'magnitude', 'batonpass']),
  set('Starmie', '', 'NaturalCure', ['psywave', 'skyattack', 'triplekick', 'roar']),
  set('Gengar', '', 'Levitate', ['meanlook', 'perishsong', 'spikes', 'leechseed']));

const BOARDS = {
  // ---------------------------------------------------------------- METRONOME (natural samples)
  // (Chansey vs Snorlax, no Cute Charm: an infatuation whose SOURCE faints at the residual ends at a
  // different point in the port — a pre-B11 Attract divergence reported separately, kept out of here.)
  mt_mirror: { p1: team(set('Chansey', '', 'NaturalCure', ['metronome', 'splash'], { g: 'F' }), BENCH), p2: team(SN(['metronome', 'splash']), BENCH2), turns: 8, seeds: 140, want: M(1, 1) },
  mt_mirror_cutecharm: { p1: team(CL(['metronome', 'splash']), BENCH), p2: team(CL2(['metronome', 'splash']), BENCH2), turns: 4, seeds: 20, want: M(1, 1) },
  mt_wall: { p1: team(CL(['metronome', 'softboiled']), BENCH), p2: team(SN(['bodyslam', 'rest', 'curse', 'sleeptalk'], { item: 'Leftovers' }), BENCH), turns: 8, seeds: 60, want: cyc([[1, 1], [1, 3], [1, 2], [1, 4]]) },
  mt_pressure: { p1: CL(['metronome', 'splash']), p2: team(set('Zapdos', '', 'Pressure', ['thunderbolt', 'drillpeck']), BENCH), turns: 6, seeds: 30, want: M(1, 1) },
  mt_choiceband: { p1: team(CL(['metronome', 'splash'], { item: 'Choice Band' }), BENCH), p2: team(SN(['splash', 'bodyslam']), BENCH), turns: 6, seeds: 30, want: cyc([[1, 1], [2, 1], [1, 2]]) },
  mt_ou: { format: 'gen3ou', p1: team(set('Clefable', 'Leftovers', 'CuteCharm', ['metronome', 'softboiled', 'seismictoss', 'calmmind'], { g: 'F' }), set('Snorlax', 'Leftovers', 'ThickFat', ['bodyslam', 'rest', 'curse', 'sleeptalk'], { g: 'M' })), p2: team(set('Skarmory', 'Leftovers', 'KeenEye', ['drillpeck', 'spikes', 'roar', 'toxic'], { g: 'M' }), set('Blissey', 'Leftovers', 'NaturalCure', ['softboiled', 'seismictoss', 'toxic', 'aromatherapy'], { g: 'F' })), turns: 8, seeds: 30, want: cyc([[1, 1], [1, 3], [1, 4], [1, 2]]) },
  // A lock Metronome called, then the foe's Encore (lands on Metronome; overrides the locked repeat),
  // Disable / Spite / Encore / Mimic after the repeat (the called id in no moveslot).
  mt_encore_lock: { p1: team(CL(['metronome', 'splash']), BENCH), p2: team(SM(['encore', 'splash']), BENCH), turns: 5, seeds: 12, need: (r) => calledLockOnTurn1(r, 'p1') && r.lines.includes('|-start|p1a: Clefable|Encore'), want: seq([[1, 2], [1, 1], [1, 2]]) },
  // After the called lock's REPEAT (turn 2) `lastMove` is the callee, in no moveslot: a faster foe's
  // Disable / Spite / Encore on turn 3 draws and fails, its Mimic copies the callee.
  ...['disable', 'spite', 'encore', 'mimic'].reduce((o, x) => Object.assign(o, { [`mt_${x}_repeat`]: {
    p1: team(CL(['metronome', 'splash']), BENCH), p2: team(SM([x, 'splash']), BENCH), turns: 4, seeds: x === 'disable' ? 14 : 6,
    need: (r) => calledLockOnTurn1(r, 'p1'), want: seq([[1, 2], [1, 2], [1, 1], [1, 2]]) } }), {}),
  mt_called_lock_any: { p1: team(CL(['metronome', 'splash']), BENCH), p2: team(SN(['splash', 'bodyslam']), BENCH), turns: 5, seeds: 24, need: (r) => calledLock(r, 'p1'), want: cyc([[1, 1], [1, 2]]) },
  // ---------------------------------------------------------------- MIRROR MOVE
  mm_basic: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Jolteon', '', 'VoltAbsorb', ['tackle', 'thunderwave', 'sandattack', 'splash']), BENCH), turns: 6, seeds: 12, want: cyc([[2, 1], [1, 2], [1, 3], [1, 4], [1, 1]]) },
  mm_slow: { p1: team(PG(['mirrormove', 'splash'], { evs: '0,0,0,0,0,0' }), BENCH), p2: team(set('Jolteon', '', 'VoltAbsorb', ['tackle', 'thunderbolt', 'toxic', 'splash']), BENCH), turns: 6, seeds: 12, want: cyc([[1, 1], [1, 2], [1, 3], [1, 4]]) },
  mm_status: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Venusaur', '', 'Overgrow', ['leechseed', 'sleeppowder', 'growl', 'swordsdance']), BENCH), turns: 6, seeds: 12, want: cyc([[2, 1], [1, 2], [1, 3], [1, 4]]) },
  mm_nomirror: { p1: team(PG(['mirrormove', 'tackle']), BENCH), p2: team(set('Skarmory', '', 'KeenEye', ['spikes', 'taunt', 'curse', 'drillpeck']), BENCH), turns: 6, seeds: 10, want: cyc([[2, 1], [1, 3], [1, 4], [1, 2]]) },
  mm_switched: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(SN(['tackle', 'splash']), BENCH), turns: 4, seeds: 8, want: seq([[2, 1], [1, 'switch 2'], [1, 1], [1, 1]]) },
  mm_vs_metronome: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(CL(['metronome', 'tackle']), BENCH), turns: 6, seeds: 24, want: cyc([[2, 1], [1, 1], [1, 2]]) },
  mm_counter: { p1: team(PG(['mirrormove', 'tackle']), BENCH), p2: team(set('Machamp', '', 'Guts', ['counter', 'splash']), BENCH), turns: 4, seeds: 8, want: seq([[2, 1], [1, 2], [2, 1], [1, 2]]) },
  mm_protect: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(SN(['tackle', 'protect']), BENCH), turns: 4, seeds: 8, want: seq([[2, 1], [1, 2], [1, 1], [1, 2]]) },
  mm_fly: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Aerodactyl', '', 'RockHead', ['fly', 'splash']), BENCH), turns: 6, seeds: 8, want: seq([[2, 1], [2, 1], [1, 2], [1, 2], [1, 1], [1, 1]]) },
  mm_vs_sleeptalk: { p1: team(PG(['mirrormove', 'spore']), BENCH), p2: team(SN(['sleeptalk', 'bodyslam', 'earthquake']), BENCH), turns: 6, seeds: 12, want: seq([[2, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]) },
  mm_future: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Alakazam', '', 'Synchronize', ['futuresight', 'psychic']), BENCH), turns: 6, seeds: 8, want: seq([[2, 2], [2, 1], [2, 2], [1, 2], [1, 2], [1, 1]]) },
  mm_bide: { p1: team(PG(['mirrormove', 'tackle']), BENCH), p2: team(set('Shuckle', '', 'Sturdy', ['bide', 'splash']), BENCH), turns: 6, seeds: 8, want: seq([[2, 1], [2, 1], [2, 1], [1, 2], [1, 2], [1, 2]]) },
  mm_multihit: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Cloyster', '', 'ShellArmor', ['spikecannon', 'splash']), BENCH), turns: 4, seeds: 8, want: seq([[2, 1], [1, 2], [1, 1]]) },
  mm_mirror: { p1: team(PG(['mirrormove', 'tackle']), BENCH), p2: team(set('Fearow', '', 'KeenEye', ['mirrormove', 'tackle']), BENCH), turns: 5, seeds: 8, want: seq([[2, 2], [1, 1], [1, 1], [2, 1]]) },
  mm_faint: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Electrode', '', 'Soundproof', ['tackle', 'explosion']), BENCH), turns: 5, seeds: 8, want: seq([[2, 1], [1, 2], [1, 1], [1, 1]]) },
  // ---------------------------------------------------------------- ASSIST
  as_team1: { p1: ASSIST_TEAM(PER(['assist', 'splash'])), p2: team(SN(['splash', 'bodyslam', 'rest']), BENCH2), turns: 8, seeds: 40, want: cyc([[1, 1], [1, 2], [1, 1], [1, 3]]) },
  as_team2: { p1: ASSIST_TEAM2(PER(['assist', 'splash'])), p2: team(SN(['splash', 'bodyslam', 'rest']), BENCH2), turns: 8, seeds: 30, want: cyc([[1, 1], [1, 2], [1, 1], [1, 3]]) },
  as_hp: { p1: team(PER(['assist', 'splash'], { ivs: '30,31,31,31,31,31' }), set('Jolteon', '', 'VoltAbsorb', ['hiddenpowerice'], { ivs: '31,31,31,31,31,31' })), p2: SN(['splash']), turns: 3, seeds: 8, want: M(1, 1) },
  as_empty: { p1: team(PER(['assist', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['protect', 'metronome', 'sleeptalk', 'mimic'])), p2: SN(['splash']), turns: 2, seeds: 4, want: M(1, 1) },
  as_alone: { p1: PER(['assist', 'splash']), p2: SN(['splash']), turns: 2, seeds: 4, want: M(1, 1) },
  as_fainted: { p1: team(set('Electrode', '', 'Soundproof', ['explosion']), PER(['assist', 'splash'])), p2: team(SN(['splash']), BENCH2), turns: 4, seeds: 8, want: M(1, 1) },
  // ---------------------------------------------------------------- NATURE POWER
  np_basic: { p1: LU(['naturepower', 'splash']), p2: team(SN(['splash', 'substitute']), BENCH2), turns: 4, seeds: 12, want: cyc([[1, 1], [1, 2]]) },
  np_ghost: { p1: LU(['naturepower', 'splash']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 2, seeds: 4, want: M(1, 1) },
  np_protect: { p1: LU(['naturepower', 'splash']), p2: SN(['protect', 'splash']), turns: 3, seeds: 4, want: seq([[1, 1], [1, 2]]) },
  np_evasion: { p1: LU(['naturepower', 'doubleteam']), p2: SN(['splash']), turns: 10, seeds: 8, want: seq([[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]) },
  np_sleeptalk: { p1: LU(['sleeptalk', 'naturepower']), p2: set('Breloom', '', 'EffectSpore', ['spore', 'splash']), turns: 4, seeds: 12, want: seq([[2, 1], [1, 2], [1, 2], [1, 2]]) },
  // ---------------------------------------------------------------- SKETCH
  sk_basic: { p1: team(SM(['sketch', 'splash']), BENCH), p2: team(SN(['bodyslam', 'curse', 'rest', 'earthquake']), BENCH2), turns: 4, seeds: 12, want: seq([[2, 1], [1, 2], [2, 3], [1, 4]]) },
  sk_nolast: { p1: SM(['sketch', 'splash']), p2: SN(['tackle', 'splash']), turns: 2, seeds: 4, want: M(1, 1) },
  sk_dup: { p1: SM(['sketch', 'tackle']), p2: SN(['tackle', 'splash']), turns: 2, seeds: 4, want: seq([[2, 1], [1, 1]]) },
  sk_sub: { p1: SM(['sketch', 'splash']), p2: SN(['substitute', 'splash']), turns: 2, seeds: 4, want: seq([[2, 1], [1, 2]]) },
  sk_protect: { p1: SM(['sketch', 'splash']), p2: SN(['tackle', 'protect']), turns: 3, seeds: 4, want: seq([[2, 1], [1, 2], [1, 1]]) },
  sk_metronome: { p1: SM(['sketch', 'splash']), p2: CL(['metronome', 'splash']), turns: 3, seeds: 8, want: seq([[2, 1], [1, 2], [1, 2]]) },
  sk_struggle: { p1: SM(['sketch', 'taunt'], { evs: '0,0,0,0,0,252' }), p2: SN(['splash', 'rest']), turns: 3, seeds: 4, want: seq([[2, 1], [1, 1], [1, 1]]) },
  sk_hiddenpower: { p1: SM(['sketch', 'splash'], { ivs: '30,31,31,31,31,31' }), p2: set('Jolteon', '', 'VoltAbsorb', ['hiddenpowerice', 'splash'], { ivs: '31,31,31,31,31,31' }), turns: 3, seeds: 4, want: seq([[2, 1], [1, 1], [1, 2]]) },
  sk_transformed: { p1: SM(['transform', 'splash']), p2: SM(['sketch', 'tackle']), turns: 3, seeds: 4, want: seq([[1, 2], [1, 2], [1, 2]]) },
  sk_switch: { p1: team(SM(['sketch', 'splash'], { evs: '0,0,0,0,0,252' }), BENCH), p2: SN(['tackle', 'splash']), turns: 5, seeds: 4, want: seq([[2, 1], [1, 2], ['switch 2', 2], ['switch 2', 2], [1, 1]]) },
  sk_sleeptalk: { p1: SM(['sleeptalk', 'sketch']), p2: set('Breloom', '', 'EffectSpore', ['spore', 'splash']), turns: 4, seeds: 8, want: seq([[1, 1], [1, 2], [1, 2], [1, 2]]) },
  // ---------------------------------------------------------------- THE SHARED PATH: Sleep Talk's callee is lastMoveUsed
  // The AfterMove hooks key on `battle.activeMove`, the CALLEE: a Sleep-Talk-called Charge keeps the
  // Charge volatile it (re)set (the port keyed it on the outer move until B11).
  st_charge: { p1: set('Jolteon', '', 'VoltAbsorb', ['charge', 'sleeptalk', 'thunderbolt']), p2: set('Breloom', '', 'EffectSpore', ['spore', 'splash']), turns: 5, seeds: 6,
    need: (r) => r.lines.includes('|move|p1a: Jolteon|Charge|p1a: Jolteon|[from] Sleep Talk'), want: seq([[1, 1], [2, 2], [2, 2], [2, 2], [3, 2]]) },
  // Each caller's callee becomes the user's lastMoveUsed: a slower foe's Conversion 2 the same turn reads it.
  cv_metronome: { p1: team(CL(['metronome', 'splash']), BENCH), p2: team(P2C, BENCH), turns: 4, seeds: 16, want: M(1, 1) },
  cv_assist: { p1: team(PER(['assist', 'splash']), set('Golem', '', 'Sturdy', ['earthquake', 'rockslide']), set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'shadowball'])), p2: team(P2C, BENCH), turns: 4, seeds: 8, want: M(1, 1) },
  cv_mirrormove: { p1: team(PG(['mirrormove', 'splash']), BENCH), p2: team(set('Porygon2', '', 'Trace', ['conversion2', 'thunderbolt'], { nature: 'Brave', evs: '0,0,0,0,0,0' }), BENCH), turns: 4, seeds: 8, want: seq([[2, 2], [1, 1], [1, 2], [1, 1]]) },
  cv_naturepower: { p1: LU(['naturepower', 'splash']), p2: team(P2C, BENCH), turns: 3, seeds: 4, want: M(1, 1) },
  st_conversion2: { p1: SN(['sleeptalk', 'earthquake']), p2: set('Porygon2', '', 'Trace', ['spore', 'conversion2']), turns: 4, seeds: 12, want: seq([[2, 1], [1, 2], [1, 2], [1, 2]]) },
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  const moves = req.active[0].moves;
  if (want.startsWith('switch') && !req.active[0].trapped) {
    const k = parseInt(want.split(' ')[1], 10);
    if (side.pokemon[k - 1] && !side.pokemon[k - 1].fainted && k > 1) return want;
  }
  if (moves.length === 1 || want.startsWith('switch')) return 'move 1';
  const k = parseInt(want.split(' ')[1], 10);
  if (moves[k - 1] && !moves[k - 1].disabled) return want;
  const first = moves.findIndex((m) => !m.disabled);
  return first < 0 ? 'move 1' : `move ${first + 1}`;
}

function run(board, seed) {
  const battle = new Battle({ formatid: board.format || 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'P1', team: board.p1 });
  battle.setPlayer('p2', { name: 'P2', team: board.p2 });
  for (const [i, s] of [board.p1, board.p2].entries()) {
    const want = s.split(']').map((x) => x.split('|')[1]);
    const got = battle.sides[i].pokemon.map((p) => p.species.name);
    if (JSON.stringify(want) !== JSON.stringify(got)) throw new Error(`team ${i} unpacked as ${got}`);
  }
  const seedBefore = battle.prng.getSeed();
  const sent = [];
  const freeze = () => { for (let i = sent.length; i < battle.log.length; i++) sent.push(battle.log[i]); };
  freeze();
  const script = [];
  let t = 0;
  while (!battle.ended && t < board.turns) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    // A choice the sim refuses (a move a Metronome-called IMPRISON hides — the request still shows it
    // enabled) leaves no trace in the omniscient stream: fall back to the next slot it accepts and
    // record the ACCEPTED choice (the port replays only that).
    for (const i of [0, 1]) {
      if (d[i] === null) continue;
      const id = i === 0 ? 'p1' : 'p2';
      if (battle.choose(id, d[i])) continue;
      const n = battle.sides[i].activeRequest.active[0].moves.length;
      const alt = Array.from({ length: n }, (_, k) => `move ${k + 1}`).find((c) => c !== d[i] && battle.choose(id, c));
      if (!alt) throw new Error(`${id} choice ${d[i]} rejected, no fallback`);
      d[i] = alt;
    }
    freeze();
    script.push(d);
    if (!forced) t++;
  }
  freeze();
  const omni = [];
  for (let i = 0; i < sent.length; i++) {
    const l = sent[i];
    if (l.startsWith('|split|')) { omni.push(sent[i + 1]); i += 2; continue; }
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
  const want = b.seeds || N;
  let got = 0;
  for (let k = 0; got < want; k++) {
    if (k > 4000) { console.error(`REFUSING: board ${name} found only ${got}/${want} rows in 4000 seeds`); process.exit(1); }
    const seed = [31 + k * 7, 47 + k * 13, 61 + k * 29, 73 + k * 51];
    const r = { name, format: b.format || 'gen3customgame', p1: b.p1, p2: b.p2, ...run(b, seed) };
    if (b.need && !b.need(r)) continue;
    rows.push(r);
    got++;
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const calledBy = (caller, names) => (r) => r.lines.some((l) => l.startsWith('|move|') && l.includes(`[from] ${caller}`) && names.some((c) => l.includes(`|${c}|`)));
const TWO_TURN = ['Solar Beam', 'Razor Wind', 'Sky Attack', 'Skull Bash', 'Fly', 'Bounce', 'Dig', 'Dive'];
const RECHARGE = ['Hyper Beam', 'Blast Burn', 'Frenzy Plant', 'Hydro Cannon'];
const MULTI = ['Double Kick', 'Twineedle', 'Bonemerang', 'Pin Missile', 'Bullet Seed', 'Icicle Spear', 'Rock Blast', 'Barrage',
  'Comet Punch', 'Double Slap', 'Spike Cannon', 'Arm Thrust', 'Fury Attack', 'Fury Swipes', 'Bone Rush', 'Triple Kick', 'Beat Up'];
const FIXED = ['Seismic Toss', 'Night Shade', 'Sonic Boom', 'Dragon Rage', 'Super Fang', 'Psywave', 'Endeavor', 'Fissure', 'Horn Drill', 'Guillotine', 'Sheer Cold'];
const STATUS = ['Thunder Wave', 'Toxic', 'Poison Powder', 'Poison Gas', 'Will-O-Wisp', 'Spore', 'Sleep Powder', 'Hypnosis', 'Sing',
  'Lovely Kiss', 'Grass Whistle', 'Stun Spore', 'Glare', 'Confuse Ray', 'Supersonic', 'Sweet Kiss', 'Teeter Dance', 'Leech Seed', 'Yawn'];
const LOCKIN = ['Outrage', 'Thrash', 'Petal Dance', 'Rollout', 'Ice Ball', 'Uproar'];
const cov = {
  // METRONOME
  metronomeCalls: rows.reduce((a, r) => a + r.lines.filter((l) => l.includes('[from] Metronome')).length, 0),
  metronomeTwoTurn: n(calledBy('Metronome', TWO_TURN)),
  metronomeRecharge: n((r) => calledBy('Metronome', RECHARGE)(r) && has(r, '|-mustrecharge|')),
  metronomeMultiHit: n((r) => calledBy('Metronome', MULTI)(r) && has(r, '|-hitcount|')),
  metronomeFixed: n(calledBy('Metronome', FIXED)),
  metronomeStatus: n(calledBy('Metronome', STATUS)),
  metronomeBide: n(calledBy('Metronome', ['Bide'])),
  metronomeLockIn: n((r) => calledBy('Metronome', LOCKIN)(r) && has(r, '|[from] lockedmove')),
  metronomeCallsB10: n(calledBy('Metronome', ['Teleport', 'Camouflage', 'Role Play'])),
  metronomeCallsAssist: n(calledBy('Metronome', ['Assist'])),
  metronomeCallsMirrorMove: n(calledBy('Metronome', ['Mirror Move'])),
  metronomeCallsNaturePower: n(calledBy('Metronome', ['Nature Power'])),
  metronomeGen3ou: n((r) => r.format === 'gen3ou' && has(r, '[from] Metronome')),
  metronomeChoiceBand: n((r) => r.name === 'mt_choiceband' && has(r, '[from] Metronome')),
  encoreOverridesCalledLock: n((r) => r.lines.some((l) => /^\|move\|p1a: Clefable\|Metronome\|p1a: Clefable\|\[from\] lockedmove$/.test(l))),
  ...Object.fromEntries(['Disable', 'Spite', 'Encore'].map((x) => [`fail${x}AfterCalledRepeat`,
    n((r) => r.name === `mt_${x.toLowerCase()}_repeat` && r.lines.some((l) => l === `|move|p2a: Smeargle|${x}||[still]`))])),
  mimicCopiesCalledMove: n((r) => r.name === 'mt_mimic_repeat' && r.lines.some((l) => l.startsWith('|-activate|p2a: Smeargle|move: Mimic|') && !l.endsWith('|Metronome'))),
  // MIRROR MOVE
  mirrorMoveCalls: n((r) => has(r, '[from] Mirror Move')),
  mirrorMoveFails: n((r) => has(r, '|Mirror Move||[still]')),
  mirrorMoveCounter: n(calledBy('Mirror Move', ['Counter'])),
  mirrorMoveStatus: n(calledBy('Mirror Move', STATUS)),
  mirrorMoveTwoTurn: n(calledBy('Mirror Move', TWO_TURN)),
  mirrorMoveBide: n(calledBy('Mirror Move', ['Bide'])),
  mirrorMoveMultiHit: n(calledBy('Mirror Move', MULTI)),
  mirrorMoveProtect: n((r) => r.name === 'mm_protect' && has(r, '[from] Mirror Move') && has(r, '|-activate|p2a: Snorlax|Protect')),
  mirrorMoveVsMetronomeFail: n((r) => r.name === 'mm_vs_metronome' && has(r, '|Mirror Move||[still]')),
  mirrorMoveVsSleepTalk: n((r) => r.name === 'mm_vs_sleeptalk' && has(r, '[from] Mirror Move')),
  // ASSIST
  assistCalls: n((r) => has(r, '[from] Assist')),
  assistTwoTurn: n(calledBy('Assist', TWO_TURN)),
  assistRecharge: n((r) => calledBy('Assist', RECHARGE)(r) && has(r, '|-mustrecharge|')),
  assistMultiHit: n((r) => calledBy('Assist', MULTI)(r) && has(r, '|-hitcount|')),
  assistFixed: n(calledBy('Assist', FIXED)),
  assistStatus: n(calledBy('Assist', STATUS)),
  assistBide: n(calledBy('Assist', ['Bide'])),
  assistHiddenPower: n(calledBy('Assist', ['Hidden Power'])),
  assistFails: n((r) => has(r, '|Assist||[still]')),
  assistFaintedMate: n((r) => r.name === 'as_fainted' && calledBy('Assist', ['Explosion'])(r)),
  // NATURE POWER
  naturePowerSwift: n((r) => has(r, '|Swift|') && has(r, '[from] Nature Power')),
  naturePowerMiss: n((r) => has(r, '|Nature Power|') && has(r, '|[miss]')),
  naturePowerImmune: n((r) => r.name === 'np_ghost' && has(r, '|-immune|')),
  naturePowerSleepTalk: n((r) => r.lines.some((l) => l.includes('|Nature Power|') && l.includes('[from] Sleep Talk'))),
  // SKETCH
  sketchCopies: n((r) => r.lines.some((l) => /^\|-activate\|p[12]a: [^|]+\|move: Mimic\|/.test(l)) && has(r, '|Sketch|')),
  sketchFails: n((r) => has(r, '|Sketch||[still]')),
  sketchThroughProtect: n((r) => r.name === 'sk_protect' && has(r, 'move: Mimic|Protect')),
  sketchStruggle: n((r) => r.name === 'sk_struggle' && has(r, '|Struggle|') && has(r, '|Sketch||[still]')),
  sketchTransformed: n((r) => r.name === 'sk_transformed' && has(r, '|Sketch||[still]')),
  sketchHiddenPower: n((r) => r.name === 'sk_hiddenpower' && has(r, 'move: Mimic|Hidden Power')),
  sketchMetronome: n((r) => r.name === 'sk_metronome' && has(r, 'move: Mimic|Metronome')),
  // THE SHARED PATH
  sleepTalkChargeKept: n((r) => r.name === 'st_charge' && r.lines.some((l, i) => l === '|move|p1a: Jolteon|Charge|p1a: Jolteon|[from] Sleep Talk'
    && !(r.lines[i + 2] || '').startsWith('|-end|p1a: Jolteon|Charge'))),
  ...Object.fromEntries([['Metronome', 'cv_metronome'], ['Assist', 'cv_assist'], ['Mirror Move', 'cv_mirrormove'], ['Nature Power', 'cv_naturepower']].map(([c, b]) =>
    [`conversion2After${c.replace(' ', '')}`, n((r) => r.name === b && r.lines.some((l, i) => l.includes(`[from] ${c}`)
      && r.lines.slice(i + 1).some((m) => m.startsWith('|-start|p2a: Porygon2|typechange|'))))])),
  conversion2AfterSleepTalk: n((r) => r.name === 'st_conversion2' && has(r, '[from] Sleep Talk') && r.lines.some((l) => /typechange\|(Bug|Grass|Flying)$/.test(l))),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0 && !process.env.B11_NOFLOOR) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B11 MOVES GOLDEN (gen3_b11_moves_v1): Metronome, Mirror Move, Assist, Nature Power, Sketch (+ the shared calling path)',
  `# generated by harness/gen_b11_moves_golden.js  rows/board=${N} (default)  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines AS SENT (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
