// probe_items_b3.js — gen3 full parity B3 (`gen3_items_b3_v1`): Shell Bell, Macho Brace, Mail,
// Mental Herb against the real sim (gen3customgame). Re-runnable:
//   node src/rust_sim/harness/probe_items_b3.js "$PWD/deps/pokemon-showdown"
//
// SETTLED (2026-10-26, against the resolved `Dex.mod('gen3')`; none of the four is touched by the
// gen-3/4 mods, so `data/items.ts` is the handler):
//   * SHELL BELL `onAfterMoveSecondarySelf`: `if (move.totalDamage && !pokemon.forceSwitchFlag)
//     this.heal(move.totalDamage / 8, pokemon)`. Line: `|-heal|<user>|<HP>|[from] item: Shell
//     Bell|[of] <target>`. It fires at the end of the user's useMove: after the damage and any
//     recoil (Double-Edge: recoil line, then the heal), BEFORE the action's `|faint|` lines (a KO
//     heals, then the target faints), before the foe moves. `totalDamage` is the HP taken off the
//     MON: a multi-hit sums its strikes (Spike Cannon 52 + 53 → heal 13), a hit on a Substitute
//     adds 0 (gen-3 `moveHit` returns `undefined` for HIT_SUBSTITUTE → no heal at all), a
//     fixed-damage move counts (Seismic Toss 100 → 12). `battle.heal` bumps a sub-1 amount to 1
//     and truncates: heal = max(1, floor(total / 8)). A full-HP holder: no line. DRAW-FREE.
//   * MACHO BRACE `onModifySpe`: `chainModify(0.5)` — a Jolteon that outspeeds a max-Speed
//     Snorlax moves after it with the brace. Turn order, so DRAW-relevant.
//   * MAIL `onTakeItem`: false unless the active move is Knock Off / Thief / Covet. Trick onto OR
//     off a Mail holder fails: `|move|<user>|Trick||[still]` then `|-fail|<user>`, draws = the
//     accuracy roll only. Thief steals it, Knock Off knocks it off, as for any item.
//   * MENTAL HERB `onUpdate`: at the Update right after a Cute Charm infatuation:
//     `|-enditem|<mon>|Mental Herb`, `|-end|<mon>|Attract|[silent]`,
//     `|-end|<mon>|move: Attract|[from] item: Mental Herb`. The herb is consumed (useItem).
'use strict';
const path = require('path');
const PS = process.argv[2] || path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const set = (sp, item, ab, moves, o = {}) => `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
function run(label, p1, p2, script, seed = [1, 2, 3, 4]) {
  const b = new Battle({ formatid: 'gen3customgame', seed });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  const s0 = b.prng.getSeed();
  const back = b.prng.rng; const orig = back.next.bind(back); let draws = 0; back.next = (...a) => { draws++; return orig(...a); };
  for (const [a, c] of script) { if (a) b.choose('p1', a); if (c) b.choose('p2', c); }
  const omni = []; for (let i = 0; i < b.log.length; i++) { const l = b.log[i]; if (l.startsWith('|split|')) { omni.push(b.log[i + 1]); i += 2; continue; } omni.push(l); }
  const t1 = omni.indexOf('|turn|1');
  console.log(`== ${label}  draws=${draws}`);
  console.log(omni.slice(t1 + 1).filter((l) => l && !l.startsWith('|t:|') && l !== '|' && l !== '|upkeep').join('\n'));
}
// Shell Bell: plain hit, sub hit, KO, multihit, fixed damage (Seismic Toss), user at full HP.
run('shellbell plain', set('Snorlax', 'Shell Bell', 'ThickFat', ['bodyslam', 'splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss']), [['move 1', 'move 1'], ['move 1', 'move 1']]);
run('shellbell full hp', set('Snorlax', 'Shell Bell', 'ThickFat', ['bodyslam', 'splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['splash']), [['move 1', 'move 1']]);
run('shellbell sub', set('Snorlax', 'Shell Bell', 'ThickFat', ['bodyslam', 'splash']), set('Jolteon', 'Leftovers', 'VoltAbsorb', ['substitute', 'seismictoss']), [['move 2', 'move 2'], ['move 1', 'move 1'], ['move 1', 'move 2']]);
run('shellbell multihit', set('Cloyster', 'Shell Bell', 'ShellArmor', ['spikecannon', 'splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss']), [['move 1', 'move 1'], ['move 1', 'move 1']]);
run('shellbell seismictoss', set('Snorlax', 'Shell Bell', 'ThickFat', ['seismictoss', 'splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss']), [['move 1', 'move 1'], ['move 1', 'move 1']]);
run('shellbell ko', set('Snorlax', 'Shell Bell', 'ThickFat', ['seismictoss', 'splash']), [set('Pichu', '', 'Static', ['splash'], { lv: 5 }), set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss'])].join(']'), [['move 1', 'move 1'], [null, 'switch 2'], ['move 1', 'move 1']]);
run('shellbell doubleedge', set('Snorlax', 'Shell Bell', 'ThickFat', ['doubleedge', 'splash']), set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss']), [['move 1', 'move 1']]);
// Macho Brace: speed order flips.
run('machobrace order', set('Jolteon', 'Macho Brace', 'VoltAbsorb', ['tackle']), set('Snorlax', 'Leftovers', 'ThickFat', ['tackle'], { evs: '0,0,0,0,0,252' }), [['move 1', 'move 1']]);
run('machobrace control', set('Jolteon', 'Leftovers', 'VoltAbsorb', ['tackle']), set('Snorlax', 'Leftovers', 'ThickFat', ['tackle'], { evs: '0,0,0,0,0,252' }), [['move 1', 'move 1']]);
// Mail: Trick onto/off a Mail holder; Thief takes it; Knock Off removes it.
run('mail trick target', set('Alakazam', 'Choice Band', 'Synchronize', ['trick']), set('Snorlax', 'Mail', 'ThickFat', ['splash']), [['move 1', 'move 1']]);
run('mail trick user', set('Alakazam', 'Mail', 'Synchronize', ['trick']), set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), [['move 1', 'move 1']]);
run('mail thief', set('Sneasel', '', 'InnerFocus', ['thief']), set('Snorlax', 'Mail', 'ThickFat', ['splash']), [['move 1', 'move 1']]);
run('mail knockoff', set('Sneasel', '', 'InnerFocus', ['knockoff']), set('Snorlax', 'Mail', 'ThickFat', ['splash']), [['move 1', 'move 1']]);
// Mental Herb: Cute Charm infatuation cured.
run('mentalherb cutecharm', set('Snorlax', 'Mental Herb', 'ThickFat', ['tackle', 'splash'], { g: 'M' }), set('Clefable', 'Leftovers', 'CuteCharm', ['splash'], { g: 'F' }), [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], [5, 6, 7, 8]);
