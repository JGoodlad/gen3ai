// probe_twoturn_family.js — ground truth for the gen-3 TWO-TURN family (`gen3_twoturn_family_v1`,
// gen3 full parity B4): Razor Wind, Sky Attack, Skull Bash and the semi-invulnerable Fly, Bounce,
// Dig, Dive. The RESOLVED `Dex.mod('gen3')` is the oracle; the synchronous `Battle` is driven turn
// by turn and every turn prints its decisions, its PRNG call count and its omniscient lines.
//
//   node src/rust_sim/harness/probe_twoturn_family.js [scenario-substring]
//
// THE SETTLED SPEC (every line cited from deps/pokemon-showdown; re-run this probe to re-settle):
//
//  Charge turn (onTryMove, data/moves.ts via the resolved gen3 move; twoturnmove in
//  data/conditions.ts):
//   * runMove deducts PP (sim/battle-actions.ts:281, 1 PP; a Pressure foe adds 1 in gen3
//     useMoveInner, data/mods/gen3/scripts.ts:175-186), then useMoveInner announces
//     `|move|<u>|<Move>|<t>`, then `singleEvent('TryMove')` = onTryMove: `-prepare|<u>|<Move>`,
//     (Skull Bash only) `this.boost({def: 1})` -> `|-boost|<u>|def|1` (`-boost|<u>|def|0` at +6 — the
//     probe below), runEvent('ChargeMove') (no gen-3 handler), `addVolatile('twoturnmove')`, whose
//     onStart adds the MOVE-ID volatile and `attrLastMove('[still]')` (the announce becomes
//     `|move|<u>|<Move>||[still]`) and returns null. ZERO move draws (no accuracy, crit, damage).
//   * The move-id volatile: Razor Wind / Sky Attack / Skull Bash have NO condition (inert, no
//     duration). Fly / Bounce / Dig / Dive carry `condition: {duration: 2, onInvulnerability, ...}`,
//     so the charge-turn residual gathers TWO NO_ORDER duration handlers for the user
//     (twoturnmove + the move volatile, same subOrder 2 => one tie group; sim/battle.ts:484-520
//     fieldEvent with getKey 'duration').
//  Strike turn: the request is locked to the move (`twoturnmove.onLockMove`) and trapped; runMove
//   takes the `getLockedMove()` branch (battle-actions.ts:280-290): NO PP, sourceEffect lockedmove
//   (`|move|…|[from] lockedmove`), no Pressure (gen3 scripts.ts:175 `!sourceEffect`). onTryMove's
//   `attacker.removeVolatile(move.id)` lands the user, and the move resolves normally: accuracy,
//   crit (Razor Wind / Sky Attack critRatio 2), damage, secondary (Sky Attack 30 % flinch, Bounce 30 %
//   par). twoturnmove itself lingers to its own residual expiry (duration 2 -> 1 -> 0).
//  Aborts: a BeforeMove cant on the strike turn (slp, frz, par, flinch, confusion self-hit,
//   Truant, Disable, Imprison) runs `twoturnmove.onMoveAborted` -> removeVolatile('twoturnmove') -> its onEnd
//   removes the move volatile: the charge is LOST and the user lands.
//  Semi-invulnerability (gen3 tryMoveHit, data/mods/gen3/scripts.ts:300-305): for every move whose
//   target is NOT all / foeSide / allySide / allyTeam (those return at :285-298 before it), the
//   Invulnerability event runs BEFORE the immunity check and BEFORE the accuracy roll: a false
//   result is `attrLastMove('[miss]')` + `|-miss|<user>|<target>` with ZERO draws. Only the move's
//   own pre-tryMoveHit gates precede it (TryMove: Damp; selfdestruct 'always' faints the user
//   first; the `Try` singleEvent: Fake Out, Focus Punch, Doom Desire / Future Sight cast, Counter /
//   Mirror Coat's onTry). Bypass lists (resolved conditions): Fly / Bounce — gust, twister,
//   skyuppercut, thunder; Dig — earthquake, magnitude; Dive — surf, whirlpool. Status moves are NOT
//   exempt (Thunder Wave, Roar, Taunt… miss); field / side / self moves are.
//  Damage doubling: Fly / Dig / Dive `onSourceModifyDamage` chainModify(2) at the gen-3
//   `ModifyDamage` final modifier (scripts.ts:109, after type effectiveness, before the
//   randomizer); Bounce `onSourceBasePower` chainModify(2) for gust / twister at the BasePower
//   chain. Fly does NOT double Thunder / Sky Uppercut; Dig / Dive double both members.
//  Weather: Dig / Dive's `onImmunity('sandstorm'|'hail') => false` skips the weather chip
//   (sim/battle.ts:2113 `runStatusImmunity`). Fly / Bounce take it.
//  A future-move strike (data/conditions.ts futuremove onEnd -> trySpreadMoveHit) runs the gen-4
//   hitStepInvulnerabilityEvent (data/mods/gen4/scripts.ts:135): `-end|…|move: Future Sight`, then
//   `[miss]` on the last move line and `-miss|<caster>|<target>`, ZERO draws.
//  Gen 3 cancels every remaining action when a mon faints (sim/battle.ts:2140-2148), so a strike
//   whose target was KO'd earlier in the turn never runs: the volatiles linger to the residual
//   (Dig still immune to that turn's sandstorm) and expire there.
//  Sleep Talk never calls a charge move (`!move.flags['charge']`, the resolved sleeptalk.onHit).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const set = (species, ability, moves, o = {}) =>
  `${o.nick || species}|${species}|${o.item === undefined ? '' : o.item}|${ability}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|M|||${o.lv && o.lv !== 100 ? o.lv : ''}|`;
const team = (...s) => s.join(']');

function run(sc, seed = [17, 29, 43, 59]) {
  const battle = new Battle({ formatid: sc.format || 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'P1', team: sc.p1 });
  battle.setPlayer('p2', { name: 'P2', team: sc.p2 });
  let draws = 0;
  const rng = battle.prng.rng;
  const nx = rng.next.bind(rng);
  rng.next = () => { draws++; return nx(); };
  let cursor = battle.log.length;
  const out = [];
  for (let t = 0; t < sc.turns.length && !battle.ended; t++) {
    const want = sc.turns[t];
    const d = [0, 1].map((i) => {
      const side = battle.sides[i];
      const req = side.activeRequest;
      if (!req || req.wait) return null;
      if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, k) => k > 0 && !p.fainted) + 1}`;
      const moves = req.active[0].moves;
      if (moves.length === 1 && !want[i].startsWith('switch')) return 'move 1';
      return want[i];
    });
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const reqs = battle.sides.map((s) => (s.activeRequest && s.activeRequest.active ? JSON.stringify(s.activeRequest.active[0]) : '-'));
    const before = draws;
    for (const i of [0, 1]) if (d[i] !== null) {
      const ok = battle.choose(`p${i + 1}`, d[i]);
      if (!ok) out.push(`   !! choose p${i + 1} ${d[i]} rejected`);
    }
    const lines = [];
    for (let k = cursor; k < battle.log.length; k++) {
      const l = battle.log[k];
      if (l.startsWith('|split|')) { lines.push(battle.log[k + 1]); k += 2; continue; }
      if (l === '' || l.startsWith('|t:|')) continue;
      lines.push(l);
    }
    cursor = battle.log.length;
    out.push(`  T${t}${forced ? '(forced)' : ''} ${JSON.stringify(d)} draws=${draws - before}` + (sc.showReq ? `\n     req1=${reqs[0]}\n     req2=${reqs[1]}` : ''));
    for (const l of lines) out.push(`     ${l}`);
    if (sc.vols) out.push(`     vols p1=${JSON.stringify(Object.keys(battle.sides[0].active[0].volatiles))} p2=${JSON.stringify(Object.keys(battle.sides[1].active[0].volatiles))} pp1=${battle.sides[0].active[0].moveSlots.map((m) => m.pp).join('/')}`);
  }
  return out;
}

const S = {
  // ---- the seven moves, charge + strike, against a Splash wall ----
  basics: (m) => ({ p1: team(set('Dragonite', 'InnerFocus', [m, 'splash'])), p2: team(set('Snorlax', 'Immunity', ['splash', 'tackle'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], vols: true, showReq: true }),
  // Skull Bash at +6 Def (Iron Defense x3) — the capped boost line.
  skullbash_cap: { p1: team(set('Blastoise', 'Torrent', ['skullbash', 'irondefense'])), p2: team(set('Snorlax', 'Immunity', ['splash'])), turns: [['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']] },
  // ---- a foe hits the semi-invulnerable user (user faster: charges first) ----
  vs: (m, foeMove, foe = 'Zapdos') => ({ p1: team(set('Aerodactyl', 'RockHead', [m, 'splash'], { evs: '0,0,0,0,0,252' })), p2: team(set(foe, 'Pressure', [foeMove, 'splash'], { evs: '252,0,0,0,0,0' })), turns: [['move 1', 'move 1'], ['move 1', 'move 2']], vols: true }),
  // the foe faster on the STRIKE turn (it hits first while the user is still up)
  slow: (m, foeMove, foe = 'Jolteon') => ({ p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set(foe, 'VoltAbsorb', ['splash', foeMove], { evs: '0,0,0,0,0,252' })), turns: [['move 1', 'move 1'], ['move 1', 'move 2']], vols: true }),
};

const scenarios = [];
for (const m of ['razorwind', 'skyattack', 'skullbash', 'fly', 'bounce', 'dig', 'dive']) scenarios.push([`basic ${m}`, S.basics(m)]);
scenarios.push(['skullbash cap', S.skullbash_cap]);
for (const [m, mv] of [['fly', 'tackle'], ['fly', 'thunderwave'], ['fly', 'roar'], ['fly', 'swift'], ['fly', 'earthquake'], ['fly', 'gust'], ['fly', 'twister'], ['fly', 'thunder'], ['fly', 'skyuppercut'],
  ['bounce', 'gust'], ['bounce', 'thunder'], ['dig', 'earthquake'], ['dig', 'tackle'], ['dive', 'surf'], ['dive', 'whirlpool'], ['dive', 'thunderbolt'],
  ['fly', 'spikes'], ['fly', 'haze'], ['fly', 'perishsong'], ['fly', 'sandstorm'], ['fly', 'taunt'], ['fly', 'encore'], ['fly', 'disable'], ['fly', 'leechseed'], ['fly', 'toxic']]) {
  scenarios.push([`slow-foe hits ${m} with ${mv}`, S.slow(m, mv)]);
}
// Gust vs Fly and Gust vs Bounce: the damage doubling, read against the same Gust on the ground.
scenarios.push(['gust control (no fly)', { p1: team(set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['splash', 'gust'], { evs: '0,0,0,0,0,252' })), turns: [['move 1', 'move 2'], ['move 1', 'move 2']] }]);
// Weather: Dig / Dive skip the sand / hail chip on the charge turn; Fly does not.
for (const [m, w] of [['dig', 'sandstorm'], ['dive', 'hail'], ['fly', 'sandstorm']]) {
  scenarios.push([`weather ${w} ${m}`, { p1: team(set('Snorlax', 'Immunity', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', [w, 'splash'], { evs: '0,0,0,0,0,252' })), turns: [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);
}
// Aborts on the strike turn: full paralysis (many seeds), Truant, a Twister flinch.
scenarios.push(['para abort', { p1: team(set('Snorlax', 'Immunity', ['fly', 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['thunderwave', 'splash'], { evs: '0,0,0,0,0,252' })), turns: [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);
scenarios.push(['truant', { p1: team(set('Slaking', 'Truant', ['dig', 'splash'])), p2: team(set('Snorlax', 'Immunity', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], vols: true }]);
scenarios.push(['twister flinch', { p1: team(set('Snorlax', 'Immunity', ['fly', 'splash'])), p2: team(set('Dragonite', 'InnerFocus', ['splash', 'twister'], { evs: '0,0,0,0,0,252' })), turns: [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);
// Protect on the strike turn; the foe switches between charge and strike; Pressure PP.
scenarios.push(['protect strike', { p1: team(set('Snorlax', 'Immunity', ['bounce', 'splash'])), p2: team(set('Zapdos', 'Pressure', ['splash', 'protect'], { evs: '0,0,0,0,0,252' })), turns: [['move 1', 'move 1'], ['move 1', 'move 2']], vols: true }]);
scenarios.push(['switch between', { p1: team(set('Snorlax', 'Immunity', ['dive', 'splash'])), p2: team(set('Zapdos', 'Pressure', ['splash']), set('Jolteon', 'VoltAbsorb', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'switch 2'], ['move 1', 'move 1']], vols: true }]);
// The foe explodes on the strike turn (faster): the KO cancels the strike; the volatiles linger.
scenarios.push(['explode cancels strike (sand)', { p1: team(set('Snorlax', 'Immunity', ['dig', 'splash'], { lv: 60 }), set('Snorlax', 'Immunity', ['splash'])), p2: team(set('Gengar', 'Levitate', ['sandstorm', 'explosion'], { evs: '0,252,0,0,0,252' }), set('Jolteon', 'VoltAbsorb', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 1'], ['move 1', 'move 1']], vols: true }]);
// The flyer's own Explosion-immune-foe case: a foe Explosion INTO the semi-invulnerable user.
scenarios.push(['explosion into dig', { p1: team(set('Snorlax', 'Immunity', ['dig', 'splash'])), p2: team(set('Gengar', 'Levitate', ['splash', 'explosion'], { evs: '0,0,0,0,0,252' }), set('Jolteon', 'VoltAbsorb', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 1']], vols: true }]);
// Choice Band + Fly: the lock after the strike; Future Sight resolving into the flyer.
scenarios.push(['choice band', { p1: team(set('Aerodactyl', 'RockHead', ['fly', 'splash'], { item: 'Choice Band' })), p2: team(set('Snorlax', 'Immunity', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 2', 'move 1']], showReq: true, vols: true }]);
scenarios.push(['future sight into fly', { p1: team(set('Snorlax', 'Immunity', ['fly', 'splash'])), p2: team(set('Alakazam', 'Synchronize', ['futuresight', 'splash'])), turns: [['move 2', 'move 1'], ['move 2', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);
// A Fly mirror at equal speed (the residual tie group: twoturnmove + fly per side).
scenarios.push(['fly mirror', { p1: team(set('Pidgeot', 'KeenEye', ['fly', 'splash'])), p2: team(set('Pidgeot', 'KeenEye', ['fly', 'splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']] }]);
// Sky Attack flinch + Bounce paralysis over seeds are the golden's job; Sleep Talk calls none.
scenarios.push(['sleep talk', { p1: team(set('Snorlax', 'Immunity', ['rest', 'sleeptalk', 'fly', 'dig'])), p2: team(set('Jolteon', 'VoltAbsorb', ['splash'])), turns: [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], vols: true }]);
// Encore into a two-turn move (encored on the strike turn, the user faster).
scenarios.push(['encore after strike', { p1: team(set('Aerodactyl', 'RockHead', ['fly', 'splash'], { evs: '0,0,0,0,0,252' })), p2: team(set('Snorlax', 'Immunity', ['encore', 'splash'])), turns: [['move 1', 'move 2'], ['move 1', 'move 1'], ['move 2', 'move 2'], ['move 2', 'move 2'], ['move 2', 'move 2']], vols: true, showReq: true }]);

// The user FASTER: it charges first, then the foe's move meets the semi-invulnerable target.
for (const mv of ['roar', 'whirlwind', 'perishsong', 'yawn', 'seismictoss', 'pinmissile', 'beatup', 'teeterdance', 'knockoff', 'meanlook', 'curse', 'memento', 'transform', 'painsplit', 'highjumpkick', 'outrage', 'rollout', 'furycutter', 'struggle']) {
  scenarios.push([`fast-user ${mv} into fly`, S.vs('fly', mv, mv === 'highjumpkick' ? 'Hitmonlee' : mv === 'beatup' ? 'Houndoom' : 'Gengar')]);
}
// Haze with the flyer boosted (Agility first), the charge, then Haze.
scenarios.push(['haze clears the flyer', { p1: team(set('Aerodactyl', 'RockHead', ['fly', 'agility'], { evs: '0,0,0,0,0,252' })), p2: team(set('Weezing', 'Levitate', ['haze', 'splash'])), turns: [['move 2', 'move 2'], ['move 1', 'move 1']], vols: true }]);
// Imprison / Disable / Encore / Taunt / Torment on a NON-invulnerable charging user (Skull Bash).
for (const mv of ['imprison', 'disable', 'encore', 'torment', 'taunt']) {
  scenarios.push([`${mv} vs skull bash charge`, { p1: team(set('Aerodactyl', 'RockHead', ['skullbash', 'splash'], { evs: '0,0,0,0,0,252' })), p2: team(set('Mew', 'Synchronize', [mv, 'skullbash', 'splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 3'], ['move 1', 'move 3'], ['move 1', 'move 3']], vols: true, showReq: true }]);
}
// Yawn lands before the charge; the sleep arrives at the charge-turn residual: the strike turn cants.
scenarios.push(['yawn sleeps the flyer', { p1: team(set('Snorlax', 'Immunity', ['dig', 'splash'])), p2: team(set('Slowbro', 'Oblivious', ['yawn', 'splash'], { evs: '0,0,0,0,0,252' })), turns: [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);
// The strike KOs the foe; the replacement; a Pressure foe on the charge.
scenarios.push(['strike KO', { p1: team(set('Aerodactyl', 'RockHead', ['skyattack', 'splash'])), p2: team(set('Pikachu', 'Static', ['splash'], { lv: 20 }), set('Zapdos', 'Pressure', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], vols: true }]);
// Dig into a Flying / Levitate foe: the strike is immune.
scenarios.push(['dig into levitate', { p1: team(set('Snorlax', 'Immunity', ['dig', 'splash'])), p2: team(set('Gengar', 'Levitate', ['splash'])), turns: [['move 1', 'move 1'], ['move 1', 'move 1']], vols: true }]);
// Confusion on the strike turn (Confuse Ray before the charge).
scenarios.push(['confused strike', { p1: team(set('Snorlax', 'Immunity', ['bounce', 'splash'])), p2: team(set('Gengar', 'Levitate', ['confuseray', 'splash'], { evs: '0,0,0,0,0,252' })), turns: [['move 2', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], vols: true }]);

const filt = process.argv[2];
for (const [name, sc] of scenarios) {
  if (filt && !name.includes(filt)) continue;
  console.log(`=== ${name}`);
  for (const l of run(sc)) console.log(l);
}
