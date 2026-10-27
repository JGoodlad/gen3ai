// probe_b10_moves.js — gen3 full parity B10 (`gen3_b10_moves_v1`, `gen3_b10_endless_battle_v1`):
// Teleport, Follow Me, Helping Hand, Role Play, Camouflage, and the Endless Battle Clause, against the
// real sim (gen3customgame unless noted). Re-runnable; each board prints its lines with every PRNG
// draw attributed to the line that was last emitted when it drew (`  ~ N draw(s)`); the long Endless
// Battle Clause boards print only their item / swap lines and their last lines:
//   node src/rust_sim/harness/probe_b10_moves.js "$PWD/deps/pokemon-showdown" [filter]
//
// SETTLED (2026-10-26, against the RESOLVED `Dex.mod('gen3')` — the mod-chain law; citations are the
// file that supplies the resolved handler):
//   * TELEPORT resolves to the gen-7 mod's row (data/mods/gen7/moves.ts:1025-1030, inherited down to
//     gen 3): `priority: 0, selfSwitch: false, onTry: false`. In a trainer battle it FAILS at gen-3
//     `tryMoveHit`'s `singleEvent('Try')` (data/mods/gen3/scripts.ts:280-282 — a non-function handler
//     is returned as its value) with NO line: the self-target announce `|move|<u>|Teleport|<u>` is the
//     whole protocol, no `[still]`, no `-fail`, ZERO draws, PP spent, `lastMove` recorded (Encore
//     repeats it). Not snatchable.
//   * FOLLOW ME (data/moves.ts:6039-6074; the gen-3 slot-condition override data/mods/gen3/moves.ts:
//     303-319; priority 3 from data/mods/gen5/moves.ts:258-261): `onTry(source) { return
//     this.activePerHalf > 1; }` is false in singles, so it fails at the same `Try` with no line — the
//     bare announce, ZERO draws, no `-singleturn`, NO `followme` slot condition and so no residual
//     duration handler (draw-identical to a Splash control). PP and `lastMove` spent; Encore repeats it.
//   * HELPING HAND (data/moves.ts:8571-8601, priority 5, `target: adjacentAlly`): gen-3 `useMoveInner`
//     (data/mods/gen3/scripts.ts:120-172) takes the target from `getRandomTarget`, which returns `null`
//     for `adjacentAlly` in singles (sim/battle.ts:2517-2518); `addMove` prints the literal `null`, then
//     `attrLastMove('[notarget]')` + `|-notarget|<u>` (scripts.ts:168-171): `|move|<u>|Helping
//     Hand|null|[notarget]` (a Sleep-Talk call: `|move|<u>|Helping Hand|null|[from] Sleep
//     Talk|[notarget]`). ZERO draws, no `helpinghand` volatile, 1 PP (no `pressureTargets`, so a
//     Pressure foe adds nothing), `lastMove` recorded (Encore / Disable / Spite / Mimic all see it).
//   * ROLE PLAY (data/moves.ts:15325-15346, no gen-3/4 override): accuracy `true` (ZERO draws on every
//     path), flags `bypasssub` and NO `protect` (Protect and a Substitute do not block it), Pressure
//     applies (`target: normal`). A semi-invulnerable target misses first (`|move|…|[miss]` +
//     `|-miss|`). `onTryHit` fails when the two ability IDS are equal or the target's ability has
//     `failroleplay` (gen 3: Wonder Guard alone — data/mods/gen6/abilities.ts:125) or the user's has
//     `cantsuppress` (none in gen 3): `|move|<u>|Role Play||[still]` + `|-fail|<u>`. A Wonder Guard
//     USER may Role Play. `onHit` → `source.setAbility(target.ability, target)` (sim/pokemon.ts:
//     1908-1947): the OLD ability's `End` first — an armed Flash Fire prints `|-end|<u>|ability: Flash
//     Fire|[silent]` BEFORE the reveal; Cloud Nine / Air Lock run `eachEvent('WeatherChange')` — then
//     `|-ability|<u>|<New>|<Old>|[from] move: Role Play|[of] <t>`, and NO `Start` in gen 3
//     (sim/pokemon.ts:1941 `gen > 3`): a copied Intimidate / Speed Boost / Trace does nothing at once.
//     The copy is live at once: Speed Boost boosts at that residual, a copied Air Lock suppresses the
//     rain immediately, Shadow Tag traps (both requests `trapped:true`), Synchronize / Natural Cure /
//     Color Change / Shed Skin work, a copied status-immune ability (Insomnia, Water Veil) cures the
//     user at the next Update (`-activate|<u>|ability: Insomnia` + `-curestatus|<u>|slp|[msg]`), and a
//     copied Truant loafs on the user's own `truantTurn` (a field of the mon, toggled by the copied
//     ability's order-27 residual: a fresh copier moves the copy turn, loafs the next). A copied OWN
//     TEMPO on a confused user / OBLIVIOUS on an infatuated user is cured at the next Update too
//     (`-activate|<u>|ability: Own Tempo` + `-end|<u>|confusion`) — the port does not model that Update
//     cure (the open Own Tempo / Oblivious ability-gain class), so it PANICS there. Reverted at
//     switch-out.
//   * CAMOUFLAGE (resolved onHit: data/mods/gen4/moves.ts:168-174, the link-battle terrain): `if
//     (target.hasType('Normal') || !target.setType('Normal')) return false;` then `|-start|<u>|
//     typechange|Normal`. ZERO draws. `setType` cannot fail in gen 3, replaces the types, and sets
//     `knownType` (a transformed user's copy too). Fail on any CURRENT Normal type (a Normal / Flying):
//     `|move|<u>|Camouflage||[still]` + `|-fail|<u>` (after a `[from]` fold: `|move|<u>|Camouflage||
//     [from] Snatch|[still]`). After a Color Change or Conversion it retypes to Normal; a switch-out
//     reverts it. Snatchable (the snatcher retypes: `|move|<s>|Camouflage|<s>|[from] Snatch`).
//   * ENDLESS BATTLE CLAUSE (sim/battle.ts:1661-1770 endTurn, 1800-1899 maybeTriggerEndlessBattleClause;
//     data/rulesets.ts:1060-1068): every gen-3 singles format but Custom Game carries it. Past turn 100
//     (`this.turn` after the endTurn increment), after the turn-1000 tie and the `|bigerror|`
//     countdown: when EVERY side's active is stale (`volatileStaleness || staleness`), at least one
//     externally, and NOT every side can switch to a non-stale party mon (a trapped side cannot), the
//     battle ends before the Quick Claw roll and the `|turn|` line (draw-free). Gen-3 staleness:
//     EXTERNAL only from Skill Swap on its target (battle.ts:1336-1337; cleared at switch-out,
//     pokemon.ts:1555); INTERNAL from eating a RESTORATIVE berry (pokemon.ts:41-45) while a pending
//     staleness is up (pokemon.ts:1788-1797) — every setItem (Trick, Thief, Covet, Recycle) sets it to
//     internal (pokemon.ts:1873-1876: gen-3 Trick passes no source, so never external), a successful
//     takeItem clears it (pokemon.ts:1860), and the berry the mon STARTED with never stales it. There is
//     NO PP-based staleness. The loser test reads the SETS in the current party order (`berry`
//     re-assigned per mon, `cycle` sticky: Harvest / Pickup set ability or a Recycle set move): one
//     loser → `|-message|<Name>'s team started with the rudimentary means to perform restorative
//     berry-cycling and thus loses.` + `|` + `|win|<Foe>`; both → `|-message|Each side's team started
//     with the rudimentary means to perform restorative berry-cycling.` + `|` + `|tie`; neither → `|`
//     + `|tie`. Boards: ebc_* below (a P1 loss through Recycle, a mutual tie, a Thief bare tie, a Trick
//     + Pickup loss, a Shadow-Tag trapped side, and the controls: both can switch, a switch clears the
//     Skill Swap staleness, the starting berry does not stale, Custom Game has no clause).
'use strict';
const path = require('path');
const PS = process.argv[2] || path.resolve(__dirname, '../../../deps/pokemon-showdown');
const FILTER = process.argv[3] || '';
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash', 'harden'], { g: 'F' });

// `script` rows are [p1, p2] choice strings, or a function t -> [[p1 prefs], [p2 prefs]] for the long
// boards (the first enabled preference is played; a forced switch picks the first live bench mon).
function pick(side, prefs) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  for (const w of prefs) {
    if (w.startsWith('switch')) { if (!req.active[0].trapped) return w; continue; }
    const m = req.active[0].moves[parseInt(w.split(' ')[1], 10) - 1];
    if (m && !m.disabled) return w;
  }
  return 'move 1';
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
  const start = b.log.length;
  if (typeof script === 'function') {
    for (let t = 0; t < opts.turns && !b.ended; t++) {
      const [a, c] = script(t);
      const x = pick(b.sides[0], a), y = pick(b.sides[1], c);
      if (x) b.choose('p1', x);
      if (y) b.choose('p2', y);
    }
  } else {
    for (const [x, y] of script) { if (x) b.choose('p1', x); if (y) b.choose('p2', y); }
  }
  console.log(`== ${label}`);
  const out = [];
  for (let i = start; i < b.log.length; i++) {
    const l = b.log[i];
    if (l.startsWith('|split|')) continue;
    let n = at.filter((k) => k === i + 1).length;
    if (i > 0 && b.log[i - 2] && b.log[i - 2].startsWith('|split|')) continue;
    if (b.log[i - 1] && b.log[i - 1].startsWith('|split|')) n += at.filter((k) => k === i + 2).length;
    const shown = !(l === '' || l === '|' || l.startsWith('|t:|') || l.startsWith('|debug|') || l === '|upkeep');
    if (shown) out.push(l + (n ? `   ~ ${n} draw(s)` : ''));
    else if (n) out.push(`   (${l || 'blank'})   ~ ${n} draw(s)`);
  }
  if (opts.tail) {
    console.log(out.filter((l) => /Sitrus|Recycle|Skill Swap|Thief|Trick/.test(l)).join('\n'));
    console.log(`   ... (${out.length} lines) ...\n${out.slice(-opts.tail).join('\n')}`);
  } else {
    console.log(out.join('\n'));
  }
  if (opts.after) opts.after(b);
}
const pp = (b) => {
  for (const s of b.sides) {
    for (const p of s.pokemon) {
      console.log(`   [${p.name} ${p.hp}/${p.maxhp} ${p.status || '-'} ab=${p.ability} truantTurn=${p.truantTurn} types=${p.getTypes()} ` +
        `vol=${Object.keys(p.volatiles)} pp=${p.moveSlots.map((m) => `${m.id}:${m.pp}`)} stale=${p.volatileStaleness || ''}/${p.staleness || ''}/${p.pendingStaleness || ''}]`);
    }
  }
};
const req = (b) => { for (const s of b.sides) console.log(`   [${s.id} request trapped=${!!(s.activeRequest && s.activeRequest.active && s.activeRequest.active[0].trapped)}]`); };
const mv = (a, c) => [a && `move ${a}`, c && `move ${c}`];

// TELEPORT / FOLLOW ME / HELPING HAND
run('teleport', set('Abra', '', 'Synchronize', ['teleport', 'splash']), team(set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), BENCH), [mv(1, 2)], { after: pp });
run('teleport speed tie', set('Abra', '', 'Synchronize', ['teleport']), set('Abra', '', 'Synchronize', ['teleport']), [mv(1, 1), mv(1, 1)]);
run('teleport encored', set('Abra', '', 'Synchronize', ['teleport', 'psychic']), set('Jolteon', '', 'VoltAbsorb', ['encore', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
run('followme vs protect', set('Clefable', '', 'CuteCharm', ['followme', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), [mv(1, 1), mv(1, 2)], { after: pp });
run('CONTROL splash vs protect', set('Clefable', '', 'CuteCharm', ['followme', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), [mv(2, 1)]);
run('followme speed tie', set('Clefable', '', 'CuteCharm', ['followme']), set('Clefable', '', 'CuteCharm', ['followme']), [mv(1, 1), mv(1, 1)]);
run('followme encored', set('Clefable', '', 'CuteCharm', ['followme', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['encore', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
run('helpinghand', set('Plusle', '', 'Plus', ['helpinghand', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), [mv(1, 1), mv(1, 2)], { after: pp });
run('helpinghand speed tie', set('Plusle', '', 'Plus', ['helpinghand']), set('Plusle', '', 'Plus', ['helpinghand']), [mv(1, 1), mv(1, 1)]);
run('helpinghand pressure', set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), set('Zapdos', '', 'Pressure', ['splash']), [mv(1, 1)], { after: pp });
run('helpinghand then encore', set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), set('Jolteon', '', 'VoltAbsorb', ['encore', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
run('helpinghand then spite', set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), set('Jolteon', '', 'VoltAbsorb', ['spite', 'splash']), [mv(1, 2), mv(2, 1)], { after: pp });
run('helpinghand then mimic', set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), set('Jolteon', '', 'VoltAbsorb', ['mimic', 'splash']), [mv(1, 2), mv(2, 1), [null, 'move 1']], { after: pp });
run('sleeptalk calls helpinghand / followme', set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'helpinghand', 'followme']), set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), [mv(2, 1), mv(1, 1), mv(1, 1)], { seed: [1, 2, 3, 4] });
// ROLE PLAY
run('roleplay intimidate', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Gyarados', '', 'Intimidate', ['splash']), [mv(1, 1)], { after: pp });
run('roleplay wonderguard', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash']), [mv(1, 1)]);
run('roleplay same ability', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Espeon', '', 'Synchronize', ['splash']), [mv(1, 1)]);
run('roleplay by a wonderguard user', set('Shedinja', '', 'WonderGuard', ['roleplay']), set('Gyarados', '', 'Intimidate', ['splash']), [mv(1, 1)], { after: pp });
run('roleplay into protect', set('Alakazam', '', 'Synchronize', ['roleplay']), set('Gyarados', '', 'Intimidate', ['protect']), [mv(1, 1)]);
run('roleplay into sub', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Gyarados', '', 'Intimidate', ['substitute', 'splash']), [mv(2, 1), mv(1, 2)]);
run('roleplay into bounce', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash'], { evs: '0,0,0,0,0,0' }), set('Jolteon', '', 'VoltAbsorb', ['bounce', 'splash']), [mv(1, 1)]);
run('roleplay pressure', set('Alakazam', '', 'Synchronize', ['roleplay']), set('Zapdos', '', 'Pressure', ['splash']), [mv(1, 1)], { after: pp });
run('roleplay truant', set('Alakazam', '', 'Synchronize', ['roleplay', 'psychic']), set('Slaking', '', 'Truant', ['splash']), [mv(2, 1), mv(1, 1), mv(2, 1), mv(2, 1)], { after: pp });
run('roleplay by a truant user', set('Slaking', '', 'Truant', ['roleplay', 'splash']), set('Gyarados', '', 'Intimidate', ['splash']), [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1)], { after: pp });
run('roleplay insomnia while asleep (sleep talk)', set('Snorlax', '', 'ThickFat', ['roleplay', 'rest', 'sleeptalk']), set('Hypno', '', 'Insomnia', ['seismictoss', 'splash']), [mv(2, 1), mv(3, 2), mv(1, 2)]);
run('roleplay water veil while burned', set('Snorlax', '', 'ThickFat', ['roleplay', 'splash']), set('Vaporeon', '', 'WaterVeil', ['willowisp', 'splash']), [mv(2, 1), mv(1, 2)]);
run('roleplay own tempo while confused (port panics)', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Slowbro', '', 'OwnTempo', ['confuseray', 'splash']), [mv(2, 1), mv(1, 2)]);
run('roleplay an armed flash fire away', set('Houndoom', '', 'FlashFire', ['roleplay', 'flamethrower']), set('Gyarados', '', 'Intimidate', ['splash', 'ember']), [mv(2, 2), mv(1, 1), mv(2, 1)]);
run('roleplay cloud nine away in rain', set('Golduck', '', 'CloudNine', ['roleplay', 'raindance']), set('Gyarados', '', 'Intimidate', ['splash']), [mv(2, 1), mv(1, 1)]);
run('roleplay gains air lock in rain', set('Gyarados', '', 'Intimidate', ['roleplay', 'raindance', 'surf']), set('Rayquaza', '', 'AirLock', ['splash']), [mv(2, 1), mv(1, 1), mv(3, 1)]);
run('roleplay gains speed boost', set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), set('Ninjask', '', 'SpeedBoost', ['splash']), [mv(1, 1), mv(2, 1)]);
run('roleplay gains shadow tag', team(set('Alakazam', '', 'Synchronize', ['roleplay']), BENCH), team(set('Wobbuffet', '', 'ShadowTag', ['splash']), BENCH), [mv(1, 1)], { after: req });
run('roleplay gains natural cure, switch', team(set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), BENCH), set('Blissey', '', 'NaturalCure', ['thunderwave', 'splash'], { g: 'F' }), [mv(2, 1), mv(1, 2), ['switch 2', 'move 2']]);
run('roleplay then switch reverts', team(set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), BENCH), set('Gyarados', '', 'Intimidate', ['splash']), [mv(1, 1), ['switch 2', 'move 1'], ['switch 2', 'move 1']], { after: pp });
run('castform role plays forecast away', set('Castform', '', 'Forecast', ['roleplay', 'raindance', 'sunnyday']), set('Gyarados', '', 'Intimidate', ['splash']), [mv(2, 1), mv(1, 1), mv(3, 1)]);
// CAMOUFLAGE
run('camouflage after color change', set('Kecleon', '', 'ColorChange', ['camouflage', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['thunderbolt']), [mv(1, 1), mv(1, 1)], { after: pp });
run('camouflage normal fails', set('Snorlax', '', 'ThickFat', ['camouflage', 'splash']), set('Jolteon', '', 'VoltAbsorb', ['splash']), [mv(1, 1)]);
run('camouflage normal/flying fails', set('Pidgeot', '', 'KeenEye', ['camouflage']), set('Jolteon', '', 'VoltAbsorb', ['splash']), [mv(1, 1)]);
run('camouflage snatched', set('Kecleon', '', 'ColorChange', ['camouflage']), set('Sableye', '', 'KeenEye', ['snatch']), [mv(1, 1)], { after: pp });
run('camouflage snatched by a normal', set('Kecleon', '', 'ColorChange', ['camouflage']), set('Snorlax', '', 'ThickFat', ['snatch']), [mv(1, 1)]);
run('camouflage then switch', team(set('Gengar', '', 'Levitate', ['camouflage', 'splash']), BENCH), set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), [mv(1, 2), mv(2, 1), ['switch 2', 'move 2'], ['switch 2', 'move 1']], { after: pp });
run('camouflage transformed', set('Ditto', '', 'Limber', ['transform']), set('Gengar', '', 'Levitate', ['camouflage', 'splash']), [mv(1, 2), mv(1, 1)], { after: pp });
run('camouflage after conversion', set('Porygon2', '', 'Trace', ['conversion', 'camouflage', 'icebeam']), set('Snorlax', '', 'ThickFat', ['splash']), [mv(1, 1), mv(2, 1)], { after: pp });
run('sleeptalk calls camouflage (fail)', set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'teleport', 'camouflage']), set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), [mv(2, 1), mv(1, 1), mv(1, 1)]);

// ENDLESS BATTLE CLAUSE (gen3ou; ~101 turns each; tail only)
const NOOP = ['move 3', 'move 4'];
const AZ = set('Alakazam', 'Sitrus Berry', 'Synchronize', ['skillswap', 'recycle', 'splash', 'harden']);
const SX = set('Snorlax', 'Leftovers', 'ThickFat', ['seismictoss', 'splash', 'harden']);
const planA = (t) => (t < 2 ? [['move 3'], ['move 1']] : t === 2 ? [['move 2'], ['move 2']] : t === 3 ? [['move 1'], ['move 2', 'move 3']] : [NOOP, ['move 2', 'move 3']]);
const ebc = { format: 'gen3ou', turns: 105, tail: 6, after: pp };
run('ebc_recycle_loses', AZ, SX, planA, ebc);
run('ebc_both_cycle_tie', AZ, set('Snorlax', 'Sitrus Berry', 'ThickFat', ['seismictoss', 'splash', 'harden', 'recycle']), planA, ebc);
run('ebc_thief_bare_tie', set('Alakazam', '', 'Synchronize', ['skillswap', 'thief', 'splash', 'harden']), set('Snorlax', 'Sitrus Berry', 'ThickFat', ['seismictoss', 'splash', 'harden']),
  (t) => (t === 0 ? [['move 2'], ['move 1']] : t === 1 ? [['move 3'], ['move 1']] : t === 2 ? [['move 1'], ['move 2']] : [NOOP, ['move 2', 'move 3']]), ebc);
run('ebc_trick_pickup_loses', set('Linoone', 'Sitrus Berry', 'Pickup', ['trick', 'seismictoss', 'splash', 'harden']), set('Alakazam', 'Leftovers', 'Synchronize', ['skillswap', 'splash', 'harden']),
  (t) => (t === 0 ? [['move 1'], ['move 2']] : t < 3 ? [['move 2'], ['move 2']] : t === 3 ? [['move 3'], ['move 1']] : [NOOP, ['move 2', 'move 3']]), ebc);
run('ebc_shadowtag_trapped', team(AZ, BENCH), team(set('Wobbuffet', 'Leftovers', 'ShadowTag', ['seismictoss', 'splash', 'harden']), BENCH), planA, ebc);
run('ebc_ctrl_both_can_switch', team(AZ, BENCH), team(SX, BENCH), planA, ebc);
run('ebc_ctrl_switch_clears', AZ, team(SX, BENCH), (t) => (t === 4 || t === 5 ? [NOOP, ['switch 2']] : planA(t)), ebc);
run('ebc_ctrl_starting_berry', AZ, SX, (t) => (t === 2 ? [['move 3'], ['move 2']] : planA(t)), ebc);
run('ebc_ctrl_customgame', AZ, SX, planA, { ...ebc, format: 'gen3customgame' });
