// probe_b11_moves.js — gen3 full parity B11 (`gen3_b11_moves_v1`): Metronome, Mirror Move, Assist,
// Nature Power and Sketch, against the real sim (gen3customgame). Re-runnable; each board prints its
// lines with every PRNG draw attributed to the line that was last emitted when it drew (`~ N draw(s)`),
// and the boards that need a SPECIFIC callee force the caller's `sample` to return it (the draw is
// still taken, so the seed advance is the real one; marked `forced`):
//   node src/rust_sim/harness/probe_b11_moves.js "$PWD/deps/pokemon-showdown" [filter]
//
// SETTLED (2026-10-26, against the RESOLVED `Dex.mod('gen3')` — the mod-chain law; `scripts.ts` is
// `data/mods/gen3/scripts.ts`, `actions.ts` is `sim/battle-actions.ts`):
//   * THE CALLING PATH. Every caller's `onHit` runs `this.actions.useMove(<callee>, pokemon)` — gen-3
//     `useMoveInner` (`scripts.ts:126-262`): `pokemon.lastMoveUsed = move` FIRST (`scripts.ts:132`, so
//     the callee replaces the caller there: Conversion 2 reads it — board "conversion 2 after sleep
//     talk", which the port got wrong for Sleep Talk until B11), the announce `|move|<u>|<Callee>|
//     <target>|[from] <Caller>` (`scripts.ts:165`; a self-target callee names the user; a `[still]`
//     retro-edit blanks the target and the `[from]` stays before `[still]` / `[miss]`), NO PP and no
//     Pressure extra (`scripts.ts:177`, `!sourceEffect`), NO BeforeMove (no cant, Disable, Imprison,
//     Taunt), NO `moveUsed` — so the user's `lastMove` stays the CALLER (`actions.ts` runMove sets it
//     for the outer move only): Encore / Disable / Spite / Mimic / Sketch / Grudge / a Choice lock all
//     see the caller. The callee runs its own full pipeline (accuracy, two-turn charge, lock-in,
//     recharge, Bide, multi-hit, fixed damage, status arms). A LOCK the callee starts (twoturnmove,
//     lockedmove, rollout, uproar, bide) names the callee: the request is `{"moves":[{"move":
//     "<Callee>","id":"<callee>"}],"trapped":true}` even when the user does not carry it, and the
//     repeat is `runMove(<callee>)` with `[from] lockedmove`, no PP, and `moveUsed(<callee>)` — so
//     AFTER a repeat `lastMove` is a move in NO moveslot: Disable then draws its duration and fails,
//     Encore draws its duration and fails, Spite draws its roll and fails, Mimic and Sketch copy it
//     (boards "… after called outrage repeat"). `runMove`'s AfterMove keys on `battle.activeMove`, which
//     the callee's `useMove` set and nothing restores: Charge's `onAfterMove` (`move.id !== 'charge'`)
//     therefore KEEPS a Charge that Metronome or Sleep Talk just called ("metronome charge", "sleep talk
//     charge"; the port keyed it on the caller until B11). A Choice lock set by the caller's AfterMove stays on
//     the caller (`addVolatile('choicelock')` is a no-op once held; "cb metronome").
//     ENCORE OVER A CALLED LOCK: while a called lock holds, `lastMove` is the caller, so a foe's Encore
//     lands on it, and `encore.onOverrideAction` swaps the locked action for the caller; `runMove`
//     still sees `getLockedMove()`, so the caller runs with NO PP and `|[from] lockedmove`, the lock is
//     not repeated (Outrage / Rollout expire at the residual, Uproar keeps ticking), and a called Bide's
//     wait turn prints `-activate|<u>|move: Bide` and then runs the overriding move (only Bide's own
//     `beforeMoveCallback` ends runMove). A mid-Fly user cannot be Encored (Invulnerability miss).
//   * METRONOME (`data/mods/gen4/moves.ts:833`, inherited): the pool is every move with
//     `flags.metronome` (`!isNonstandard`, the gen-3 dex), SORTED BY `num` (336 moves; the excluded
//     ids are `turn/callers.rs` METRONOME_EXCLUDED; the bare Hidden Power is IN it, typed by the user's
//     IVs; the user's own moves are NOT dropped in gen 3). ONE `sample` = `random(336)`, drawn right after `|move|<u>|Metronome|<u>`.
//     It never fails. PP: Metronome's own, 1 even into Pressure (self target). The pool holds the B10
//     moves Teleport, Camouflage and Role Play (Follow Me and Helping Hand are excluded); modeled since B10.
//   * MIRROR MOVE (`data/mods/gen3/moves.ts:428`; `onTryHit: undefined` — NO `mirror`-flag check in
//     gen 3): reads `pokemon.getLastAttackedBy()` — the LAST `attackedBy` entry, pushed by gen-3
//     `tryMoveHit` after the hit (`scripts.ts:464`: not for a miss, the Invulnerability miss, a TryHit
//     block — Protect, Magic Coat, a move's own `onTryHit`, the absorbing / blocking abilities — a
//     natural immunity, a field / side / self move) and by a Future Sight strike (`actions.ts:1013`);
//     cleared when the HOLDER leaves or faints (`clearVolatile`), and at endTurn an entry whose source
//     is no longer active is dropped (`sim/battle.ts` endTurn). Fails (`|move|<u>|Mirror Move||[still]`
//     + `|-fail|<u>`, draw-free) on no entry, on a source with no `lastMove` (switched / fainted since),
//     on the gen-3 `noMirror` list (`turn/callers.rs` MIRROR_MOVE_EXCLUDED — note Counter is NOT on it:
//     a mirrored Counter announces and fails silently, its `counter` volatile was never added), and when
//     the source no longer HOLDS the move (`hasMove`: a Metronome / Assist callee fails, a Sleep-Talk
//     callee the source carries does not). Else `useMove(<that move>)`.
//   * ASSIST (`data/mods/gen4/moves.ts:45`, inherited): the moveslots of every OTHER mon of the side in
//     party order — FAINTED ones included — dropping `flags.noassist` (`turn/callers.rs`
//     NOASSIST_MOVES), duplicates kept; ONE `sample` = `random(n)`; an empty pool fails draw-free
//     (`[still]` + `-fail`). A called Hidden Power is typed and powered by the ASSIST user's own IVs.
//   * NATURE POWER (`data/mods/gen3/moves.ts:447`): accuracy 95, target self — ONE accuracy roll
//     against the user itself (its own accuracy / evasion stages fold in): `|move|<u>|Nature Power|<u>|
//     [miss]` + `|-miss|<u>|<u>` on a miss; else `useMove('swift')` (`[from] Nature Power`). Metronome
//     and Sleep Talk can call it (it has `flags.metronome` and no `nosleeptalk`).
//   * SKETCH (`data/mods/gen4/moves.ts:1178`, inherited; flags `bypasssub` and NO `protect`):
//     `|move|<u>|Sketch|<t>`; fails draw-free (`[still]` + `-fail|<u>`) when the user is transformed,
//     the target has no `lastMove` or is behind a Substitute, or the `lastMove` is `nosketch` (Sketch,
//     Struggle) or already one of the user's moves. Else the Sketch slot becomes the copied move in
//     `moveSlots` AND `baseMoveSlots` (permanent, survives a switch) with PP = max PP = the move's BASE
//     PP (the slot's 0 PP Ups), a request entry WITHOUT the `target` key, and `|-activate|<u>|move:
//     Mimic|<Move>`. It copies Protect through Protect and a Metronome user's `metronome`. Sleep Talk can
//     call it (a 0-PP Sketch slot is the existing `|cant|<u>|nopp|sketch`).
'use strict';
const path = require('path');
const PS = process.argv[2] || path.resolve(__dirname, '../../../deps/pokemon-showdown');
const FILTER = process.argv[3] || '';
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|${o.ivs || ''}||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' });

// `forced`: each `sample` over a move list returns the next queued id (the real draw is still taken).
function forcer(queue) {
  return (b) => {
    const orig = b.sample.bind(b);
    b.sample = (arr) => {
      const r = orig(arr);
      const isMoves = arr.length && (typeof arr[0] === 'string' || (arr[0] && (arr[0].num !== undefined || arr[0].move)));
      if (queue.length && isMoves) {
        const want = queue.shift();
        const hit = arr.find((m) => (typeof m === 'string' ? m : (m.id || m.move)) === want);
        if (!hit) throw new Error(`forced ${want} is not in the pool`);
        return hit;
      }
      return r;
    };
  };
}

function run(label, p1, p2, script, opts = {}) {
  if (FILTER && !label.includes(FILTER)) return;
  const b = new Battle({ formatid: opts.format || 'gen3customgame', seed: opts.seed || [17, 29, 43, 59] });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  const rng = b.prng.rng;
  const next = rng.next.bind(rng);
  const at = [];
  rng.next = (...a) => { at.push(b.log.length); return next(...a); };
  if (opts.force) forcer(opts.force.slice())(b);
  const start = b.log.length;
  for (const [x, y] of script) { if (x) b.choose('p1', x); if (y) b.choose('p2', y); }
  console.log(`== ${label}${opts.force ? `  (forced ${opts.force.join(',')})` : ''}`);
  for (let i = start; i < b.log.length; i++) {
    const l = b.log[i];
    if (l.startsWith('|split|')) continue;
    let n = at.filter((k) => k === i + 1).length;
    if (i > 0 && b.log[i - 2] && b.log[i - 2].startsWith('|split|')) continue;
    if (b.log[i - 1] && b.log[i - 1].startsWith('|split|')) n += at.filter((k) => k === i + 2).length;
    const shown = !(l === '' || l === '|' || l.startsWith('|t:|') || l.startsWith('|debug|') || l === '|upkeep');
    if (shown) console.log(l + (n ? `   ~ ${n} draw(s)` : ''));
    else if (n) console.log(`   (${l || 'blank'})   ~ ${n} draw(s)`);
  }
  if (opts.after) opts.after(b);
}
const pp = (b) => {
  for (const s of b.sides) for (const p of s.pokemon) {
    console.log(`   [${p.name} ${p.hp}/${p.maxhp} ${p.status || '-'} vol=${Object.keys(p.volatiles)} pp=${p.moveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`)}` +
      ` lastMove=${p.lastMove && p.lastMove.id} lastMoveUsed=${p.lastMoveUsed && p.lastMoveUsed.id} attackedBy=${JSON.stringify(p.attackedBy.map((a) => [a.source.name, a.move]))}]`);
  }
};
const req = (b) => { for (const s of b.sides) console.log(`   [${s.id} request ${JSON.stringify(s.activeRequest && s.activeRequest.active)}]`); };
const both = (b) => { pp(b); req(b); };
const mv = (a, c) => [a && `move ${a}`, c && `move ${c}`];
const seeds = (n) => Array.from({ length: n }, (_, k) => [k + 1, 2 * k + 3, 3 * k + 5, 4 * k + 7]);
const SN = (m, o = {}) => set('Snorlax', o.item || '', 'ThickFat', m, o);
const CL = (m, o = {}) => set('Clefable', o.item || '', 'CuteCharm', m, { g: 'F', ...o });
const PG = (m, o = {}) => set('Pidgeot', o.item || '', 'KeenEye', m, o);
const SM = (m, o = {}) => set('Smeargle', o.item || '', 'OwnTempo', m, o);

// METRONOME
for (const s of seeds(4)) run(`metronome ${s}`, CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1)], { seed: s, after: pp });
run('metronome pool', CL(['metronome', 'splash']), SN(['splash']), [], {
  after: (b) => {
    const pool = b.dex.moves.all().filter((m) => (!m.isNonstandard || m.isNonstandard === 'Unobtainable') && m.flags.metronome).sort((a, c) => a.num - c.num);
    console.log(`   [pool ${pool.length}: first ${pool.slice(0, 3).map((m) => m.id)} last ${pool.slice(-3).map((m) => m.id)}; B10 members ${pool.filter((m) => ['teleport', 'camouflage', 'roleplay', 'followme', 'helpinghand'].includes(m.id)).map((m) => m.id)}]`);
  },
});
run('metronome pressure (pp 1)', CL(['metronome', 'splash']), set('Zapdos', '', 'Pressure', ['splash']), [mv(1, 1)], { force: ['tackle'], after: pp });
for (const id of ['solarbeam', 'fly', 'skullbash']) run(`metronome two-turn ${id}`, CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1), mv(1, 1)], { force: [id, 'tackle'], after: both });
run('metronome outrage', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1), mv(1, 1)], { force: ['outrage', 'tackle'], after: both });
run('metronome rollout', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1), mv(1, 1)], { force: ['rollout'], after: both });
run('metronome uproar', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1)], { force: ['uproar', 'tackle'], after: both });
run('metronome bide', CL(['metronome', 'splash']), SN(['splash', 'bodyslam']), [mv(1, 2), mv(1, 2), mv(1, 2)], { force: ['bide'], after: both });
run('metronome hyper beam', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1)], { force: ['hyperbeam'], after: both });
run('metronome charge', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1), mv(1, 1)], { force: ['charge', 'tackle'], after: pp });
run('sleep talk charge', set('Jolteon', '', 'VoltAbsorb', ['sleeptalk', 'charge', 'thunderbolt']), set('Breloom', '', 'EffectSpore', ['spore', 'splash']), [mv(2, 1), mv(1, 2), mv(1, 2)], { force: ['charge'], after: pp });
run('cb metronome', CL(['metronome', 'splash'], { item: 'Choice Band' }), SN(['splash']), [mv(1, 1), mv(1, 1)], { force: ['outrage'], after: both });
run('metronome -> assist', team(CL(['metronome', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['thunderbolt'])), SN(['splash']), [mv(1, 1)], { force: ['assist', 'thunderbolt'], after: pp });
run('metronome -> nature power', CL(['metronome', 'splash']), SN(['splash']), [mv(1, 1)], { force: ['naturepower'], after: pp });
run('metronome -> mirror move', CL(['metronome', 'splash'], { evs: '0,0,0,0,0,0' }), SN(['tackle']), [mv(1, 1), mv(1, 1)], { force: ['mirrormove', 'mirrormove'], after: pp });
for (const c of [['disable', 3], ['spite', 3], ['encore', 4], ['mimic', 1]]) {
  const foeMoves = c[0] === 'mimic' ? ['mimic', 'splash', 'spite', 'encore'] : ['disable', 'splash', 'spite', 'encore'];
  run(`${c[0]} after called outrage repeat`, CL(['metronome', 'splash']), SM(foeMoves, { evs: '0,0,0,0,0,0' }), [mv(1, 2), mv(1, 2), mv(1, c[1])], { seed: [5, 5, 5, 5], force: ['outrage', 'tackle'], after: both });
}
for (const id of ['outrage', 'rollout', 'uproar', 'bide']) run(`encore over a called ${id}`, CL(['metronome', 'splash']), SM(['encore', 'splash']), [mv(1, 2), mv(1, 1), mv(1, 2), mv(1, 2)], { force: [id, 'tackle', 'tackle', 'tackle'], after: both });
// Each caller's callee is the user's lastMoveUsed: a slower foe's Conversion 2 the same turn reads it.
const P2C = set('Porygon2', '', 'Trace', ['conversion2', 'recover'], { nature: 'Brave', evs: '0,0,0,0,0,0' });
run('conversion 2 after metronome', CL(['metronome', 'splash']), P2C, [mv(1, 1)], { force: ['earthquake'], after: pp });
run('conversion 2 after assist', team(set('Persian', '', 'Limber', ['assist', 'splash']), set('Golem', '', 'Sturdy', ['earthquake'])), P2C, [mv(1, 1)], { after: pp });
run('conversion 2 after nature power', set('Ludicolo', '', 'SwiftSwim', ['naturepower', 'splash']), P2C, [mv(1, 1)], { after: pp });
run('conversion 2 after mirror move', PG(['mirrormove', 'splash']), set('Porygon2', '', 'Trace', ['conversion2', 'thunderbolt'], { nature: 'Brave', evs: '0,0,0,0,0,0' }), [mv(2, 2), mv(1, 1)], { after: pp });
run('conversion 2 after sleep talk', SN(['sleeptalk', 'earthquake']), set('Porygon2', '', 'Trace', ['spore', 'conversion2']), [mv(2, 1), mv(1, 2), mv(1, 2)], { after: pp });
// MIRROR MOVE
run('mirror move: nothing', PG(['mirrormove', 'splash']), SN(['splash', 'tackle']), [mv(1, 1)], { after: pp });
run('mirror move: same turn', PG(['mirrormove', 'splash'], { evs: '0,0,0,0,0,0' }), set('Jolteon', '', 'VoltAbsorb', ['tackle', 'thunderwave', 'splash']), [mv(1, 1), mv(1, 3)], { after: pp });
run('mirror move: a status move', PG(['mirrormove', 'splash']), SN(['tackle', 'splash', 'toxic']), [mv(2, 3), mv(1, 2)], { after: pp });
run('mirror move: foe switched', PG(['mirrormove', 'splash']), team(SN(['tackle', 'splash']), BENCH), [mv(2, 1), ['move 1', 'switch 2']], { after: pp });
run('mirror move: metronome callee', PG(['mirrormove', 'splash']), CL(['metronome', 'splash']), [mv(2, 1), mv(1, 2)], { force: ['tackle'], after: pp });
run('mirror move: metronome callee it holds', PG(['mirrormove', 'splash']), CL(['metronome', 'tackle']), [mv(2, 1), mv(1, 2)], { force: ['tackle'], after: pp });
run('mirror move: counter', PG(['mirrormove', 'tackle']), set('Machamp', '', 'Guts', ['counter', 'splash']), [mv(2, 1), mv(1, 2)], { after: pp });
run('mirror move: into protect', PG(['mirrormove', 'splash']), SN(['tackle', 'protect']), [mv(2, 1), mv(1, 2)], { after: pp });
run('mirror move: fly', PG(['mirrormove', 'splash']), set('Aerodactyl', '', 'RockHead', ['fly', 'splash']), [mv(2, 1), mv(2, 1), mv(1, 2), mv(1, 2)], { after: both });
run('mirror move: a missed move keeps the earlier entry', PG(['mirrormove', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['tackle', 'thunderwave', 'sandattack']), [mv(2, 1), mv(2, 3)], { after: pp });
run('mirror move: leech seed into grass', PG(['mirrormove', 'splash']), set('Venusaur', '', 'Overgrow', ['leechseed', 'splash']), [mv(2, 1), mv(1, 2)], { after: pp });
// ASSIST
run('assist', team(set('Persian', '', 'Limber', ['assist', 'splash']), SN(['bodyslam', 'rest']), set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'splash'])), SN(['splash']), [mv(1, 1), mv(1, 1), mv(1, 1)], { after: pp });
run('assist: empty pool', team(set('Persian', '', 'Limber', ['assist', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['protect', 'metronome'])), SN(['splash']), [mv(1, 1)], { after: pp });
run('assist: alone', set('Persian', '', 'Limber', ['assist', 'splash']), SN(['splash']), [mv(1, 1)], { after: pp });
run('assist: hidden power typed by the user', team(set('Persian', '', 'Limber', ['assist', 'splash'], { ivs: '30,31,31,31,31,31' }), set('Jolteon', '', 'VoltAbsorb', ['hiddenpowerice'], { ivs: '31,31,31,31,31,31' })), SN(['splash']), [mv(1, 1)], { after: pp });
// NATURE POWER
run('nature power', set('Ludicolo', '', 'SwiftSwim', ['naturepower', 'splash']), SN(['splash']), [mv(1, 1)], { after: pp });
run('nature power vs ghost', set('Ludicolo', '', 'SwiftSwim', ['naturepower', 'splash']), set('Gengar', '', 'Levitate', ['splash']), [mv(1, 1)]);
run('nature power vs protect', set('Ludicolo', '', 'SwiftSwim', ['naturepower', 'splash']), SN(['protect']), [mv(1, 1)]);
run('nature power misses its +6 evasion user', set('Ludicolo', '', 'SwiftSwim', ['naturepower', 'doubleteam']), SN(['splash']), [mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1)]);
run('sleep talk -> nature power', set('Ludicolo', '', 'SwiftSwim', ['sleeptalk', 'naturepower']), set('Breloom', '', 'EffectSpore', ['spore', 'splash']), [mv(2, 1), mv(1, 2), mv(1, 2)], { after: pp });
// SKETCH
run('sketch: no lastMove', SM(['sketch', 'splash']), SN(['tackle', 'splash']), [mv(1, 1)], { after: both });
run('sketch', SM(['sketch', 'splash']), SN(['tackle', 'splash']), [mv(2, 1), mv(1, 1), mv(1, 1)], { after: both });
run('sketch: already known', SM(['sketch', 'tackle']), SN(['tackle', 'splash']), [mv(2, 1), mv(1, 1)], { after: both });
run('sketch: substitute', SM(['sketch', 'splash']), SN(['substitute', 'splash']), [mv(2, 1), mv(1, 2)], { after: both });
run('sketch: protect through protect', SM(['sketch', 'splash']), SN(['tackle', 'protect']), [mv(2, 1), mv(1, 2)], { after: both });
run('sketch: metronome', SM(['sketch', 'splash']), CL(['metronome', 'splash']), [mv(2, 1), mv(1, 2)], { force: ['aurorabeam'], after: both });
run('sketch: permanent across a switch', team(SM(['sketch', 'splash'], { evs: '0,0,0,0,0,252' }), BENCH), SN(['tackle', 'splash']), [mv(2, 1), mv(1, 2), ['switch 2', 'move 2'], ['switch 2', 'move 2']], { after: both });
run('sleep talk -> sketch (0 pp)', SM(['sleeptalk', 'sketch']), set('Breloom', '', 'EffectSpore', ['spore', 'splash']), [mv(2, 1), mv(1, 2)], { after: both });
