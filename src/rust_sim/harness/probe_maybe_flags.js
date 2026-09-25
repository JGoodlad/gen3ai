// probe_maybe_flags.js — the `|request|` "maybe" display flags the port must reproduce beyond the
// real trap (`gen3_request_maybe_flags_v1`), measured against the REAL sim. Re-runnable oracle;
// fail-loud (exit 1 on any mismatch with the SETTLED table below).
//
// SETTLED (deps/pokemon-showdown @ the pinned submodule):
//  (K) `pokemon.knownType` is FALSE for a mon TRANSFORMED into a FOE (`transformInto`:
//      `this.knownType = this.isAlly(pokemon) && pokemon.knownType`, sim/pokemon.ts) until it
//      leaves the field (clearVolatile → setSpecies → true) or a setType lands (Conversion,
//      Conversion 2, Camouflage, Color Change). In gen3 it reaches the request ONLY through
//      endTurn's `MaybeTrapPokemon` (sim/battle.ts ~1723-1757):
//        - Magnet Pull (gen3 mod `onAnyMaybeTrapPokemon`): `!knownType || hasType('Steel')`;
//        - Arena Trap (`onFoeMaybeTrapPokemon`): `isGrounded(!knownType)` — Flying is IGNORED
//          for an unknown type, Levitate is not.
//      The REAL trap (`TrapPokemon`) is unchanged, so the switch is still ACCEPTED.
//  (S) The "canceling switches would leak information" loop runs `FoeMaybeTrapPokemon` for
//      EVERY ability of the FOE's (current) species — only where the format has
//      `obtainableabilities` (gen3ou yes; gen3customgame no). gen3 species carrying a trapping
//      ability they may not hold: Dugtrio/Diglett (Sand Veil|Arena Trap), Trapinch (Hyper
//      Cutter|Arena Trap), Magnemite/Magneton/Nosepass (Magnet Pull|Sturdy — but the gen3 mod
//      sets `onFoeMaybeTrapPokemon: undefined`, so it adds nothing).
//  (I) Imprison's `onFoeDisableMove` sets `maybeDisabled` on ANY live foe of the holder,
//      shared move or not (data/moves.ts ~9502-9508); getMoveRequestData then emits
//      `maybeDisabled` + `maybeLocked`.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

function reqFlags(b, side) {
  const r = b.sides[side].activeRequest;
  if (!r || !r.active) return 'none';
  const a = r.active[0];
  // The keys IN JSON ORDER (the byte order the port must reproduce), e.g.
  // `maybeDisabled,maybeLocked,maybeTrapped` — getMoveRequestData adds maybeDisabled /
  // maybeLocked BEFORE the trap flag, and a reject's update closure APPENDS `trapped`.
  return Object.keys(a).filter(k => ['trapped', 'maybeTrapped', 'maybeDisabled', 'maybeLocked'].includes(k) && a[k]).join(',') || '-';
}
function run(formatid, p1, p2, turns) {
  const b = new Battle({ formatid, seed: [1, 2, 3, 4] });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  for (const [c1, c2] of turns) { b.choose('p1', c1); b.choose('p2', c2); }
  return b;
}
const cases = [
  // (K) item 1: Smeargle transforms into a Water/Flying Gyarados, then faces a Magnet Pull Magneton.
  ['K1 transformed(Gyarados) vs Magnet Pull', 'gen3ou',
    'Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]Magneton||Leftovers|MagnetPull|Splash|Modest|,,,,,|N||||',
    'Smeargle||Leftovers|OwnTempo|Transform,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    [['move 1', 'move 1'], ['switch 2', 'move 1']], 1, 'maybeTrapped', true],
  ['K2 transformed(Gyarados) vs Arena Trap', 'gen3ou',
    'Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]Dugtrio||Leftovers|ArenaTrap|Splash|Jolly|,,,,,|M||||',
    'Smeargle||Leftovers|OwnTempo|Transform,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    [['move 1', 'move 1'], ['switch 2', 'move 1']], 1, 'maybeTrapped', true],
  ['K3 transformed(Gengar,Levitate) vs Arena Trap', 'gen3ou',
    'Gengar||Leftovers|Levitate|Splash|Timid|,,,,,|M||||]Dugtrio||Leftovers|ArenaTrap|Splash|Jolly|,,,,,|M||||',
    'Smeargle||Leftovers|OwnTempo|Transform,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    [['move 1', 'move 1'], ['switch 2', 'move 1']], 1, '-', true],
  ['K4 untransformed Gyarados vs Magnet Pull (control)', 'gen3ou',
    'Magneton||Leftovers|MagnetPull|Splash|Modest|,,,,,|N||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    [['move 1', 'move 1']], 1, '-', true],
  ['K5 transformed(Magneton) vs Magnet Pull (real trap)', 'gen3ou',
    'Magneton||Leftovers|MagnetPull|Splash|Modest|,,,,,|N||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Smeargle||Leftovers|OwnTempo|Transform,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    [['move 1', 'move 1']], 1, 'maybeTrapped', false],
  // (S) species-alternate Arena Trap
  ['S1 Sand Veil Dugtrio vs grounded (gen3ou)', 'gen3ou',
    'Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 1']], 1, 'maybeTrapped', true],
  ['S2 Sand Veil Dugtrio vs grounded (gen3customgame)', 'gen3customgame',
    'Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 1']], 1, '-', true],
  ['S3 Sand Veil Dugtrio vs Flying (gen3ou)', 'gen3ou',
    'Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Skarmory||Leftovers|KeenEye|Splash|Impish|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 1']], 1, '-', true],
  ['S4 Sturdy Magneton vs Steel (gen3ou; gen3 MP has no FoeMaybeTrap)', 'gen3ou',
    'Magneton||Leftovers|Sturdy|Splash|Modest|,,,,,|N||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Skarmory||Leftovers|KeenEye|Splash|Impish|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 1']], 1, '-', true],
  // (I) item 2: Imprison sealed vs Suicune (shared Rest), then a no-shared-move Celebi comes in.
  // (O) KEY ORDER: getMoveRequestData writes maybeDisabled, maybeLocked, THEN the trap flag.
  // A Mean Look (firm) + Imprison Dusclops — a real gen3ou moveset.
  ['O1 Mean Look + Imprison: maybeDisabled,maybeLocked BEFORE trapped', 'gen3ou',
    'Dusclops||Leftovers|Pressure|MeanLook,Imprison,Rest|Careful|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||',
    [['move 1', 'move 2'], ['move 2', 'move 2']], 1, 'maybeDisabled,maybeLocked,trapped', false],
  ['I1 Imprison vs a foe sharing no move', 'gen3ou',
    'Dusclops||Leftovers|Pressure|Imprison,Rest,Splash|Careful|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||',
    [['move 1', 'move 2'], ['move 3', 'switch 2']], 1, 'maybeDisabled,maybeLocked', true],
];
// The Imprison flags on a RE-REQUEST (the request the sim re-issues after a refused choice is the
// SAME activeRequest object, mutated by the reject's update closure):
//   (R1) a refused IMPRISONED move -> `updateDisabledRequest` deletes `maybeLocked` (and clears
//        `pokemon.maybeLocked`); `maybeDisabled` stays (singles).
//   (R2) a refused TRAPPED switch   -> the closure touches ONLY maybeTrapped/trapped, so
//        `maybeLocked` stays. (gen3customgame: an Imprison-carrying Arena Trap Dugtrio.)
const rereq = [
  ['R1 imprisoned-move reject re-request', 'gen3customgame',
    'Gengar||Leftovers|Levitate|Imprison,IceBeam,Splash|Hardy|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 3']], 1, 'move 1', 'maybeDisabled'],
  ['R2 trapped-switch reject re-request (imprisoned + Arena Trap)', 'gen3customgame',
    'Dugtrio||Leftovers|ArenaTrap|Imprison,IceBeam,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    [['move 1', 'move 3']], 1, 'switch 2', 'maybeDisabled,maybeLocked,trapped'],
];
let bad = 0;
for (const [name, fmt, t1, t2, turns, side, refused, want] of rereq) {
  const b = run(fmt, t1, t2, turns);
  const before = reqFlags(b, side);
  const ok = b.choose('p' + (side + 1), refused);
  const got = reqFlags(b, side);
  const pass = !ok && got === want;
  if (!pass) bad++;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}: fresh=${before} -> refused=${!ok} re-request=${got} (want ${want})`);
}
for (const [name, fmt, t1, t2, turns, side, want, switchOk] of cases) {
  const b = run(fmt, t1, t2, turns);
  const got = reqFlags(b, side);
  // does a voluntary switch get accepted? (probe on a CLONE via a fresh replay)
  const b2 = run(fmt, t1, t2, turns);
  const ok = b2.choose('p' + (side + 1), 'switch 2');
  const pass = got === want && (!!ok === switchOk);
  if (!pass) bad++;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}: flags=${got} (want ${want}) switchAccepted=${!!ok} (want ${switchOk})`);
}
process.exit(bad ? 1 : 0);
