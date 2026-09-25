// probe_rereq_accumulate.js — two `|request|` / choice classes the port must reproduce, measured
// against the REAL sim. Re-runnable oracle; fail-loud (exit 1 on any mismatch with the SETTLED rows).
//
// SETTLED (deps/pokemon-showdown @ the pinned submodule):
//  (F) EVERY move imprisoned -> Struggle SUBSTITUTED (`gen3_imprison_all_struggle_v1`).
//      Imprison's `onFoeDisableMove` hides the disable (`disableMove(id, true)` -> 'hidden',
//      data/moves.ts ~9502-9508). The REQUEST is built by `getMoveRequestData` ->
//      `getMoves(_, isLastActive)` (sim/pokemon.ts ~1096), where `'hidden'` renders
//      `disabled:false` — so the request still OFFERS the whole move list (+ maybeDisabled,
//      maybeLocked); it is NOT the Struggle-only shape. `Side.chooseMove` then reads
//      `getMoves()` UNRESTRICTED (hidden = disabled), gets `[]`, and substitutes
//      `moveid:'struggle'` with NO `|error|` (sim/side.ts ~682-691), sending the owner-only
//      `|-activate|<mon>|move: Struggle` (gen <= 4) at CHOICE time. Imprison's
//      `onFoeBeforeMove` skips `struggle`, so the turn runs `|move|<mon>|Struggle|<foe>`.
//  (A) Successive refusals in ONE decision ACCUMULATE (`gen3_rereq_accumulate_v1`). The sim
//      keeps ONE `side.activeRequest` and each refusal's update closure MUTATES it
//      (`emitChoiceError` -> `updateRequestForPokemon`, sim/side.ts ~511-517): a refused
//      disabled/imprisoned move runs `updateDisabledRequest` (deletes `maybeLocked`) and flips
//      that slot `disabled:true,"disabledSource":""` (~728-743); a refused hidden-trap switch
//      deletes `maybeTrapped` and sets `trapped:true` (~967-979). `emitRequest(_, true)` then
//      re-sends the WHOLE mutated object. So a second refusal's re-request carries the first's
//      delta too, and a refusal that changes NOTHING (the same refused pick again) is
//      `[Invalid choice]` with NO re-request.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

function mk(formatid, p1, p2) {
  const out = [];
  const b = new Battle({ formatid, seed: [1, 2, 3, 4], send: (type, data) => out.push([type, data]) });
  b.setPlayer('p1', { name: 'P1', team: p1 });
  b.setPlayer('p2', { name: 'P2', team: p2 });
  return { b, out };
}
// Summarise a side's CURRENT activeRequest: slot disabled flags + the flag keys in JSON order.
function reqSummary(b, side) {
  const r = b.sides[side].activeRequest;
  if (!r || !r.active) return 'none';
  const a = r.active[0];
  const mv = a.moves.map(m => (m.disabled ? 'D' : '-') + (m.disabledSource !== undefined ? 's' : '')).join('');
  const flags = Object.keys(a).filter(k => k !== 'moves' && a[k]).join(',');
  return `moves=${a.moves.map(m => m.id).join('/')}[${mv}] flags=${flags || '-'} update=${!!r.update}`;
}
// The side's sideupdate lines emitted since `from` (the `|error|` / `|-activate|` / `|request|` stream).
function sideLines(out, from, side) {
  const pfx = `p${side + 1}\n`;
  return out.slice(from).filter(([t, d]) => t === 'sideupdate' && d.startsWith(pfx))
    .map(([, d]) => d.slice(pfx.length).replace(/^\|request\|.*/, '|request|…'));
}

let bad = 0;
function check(name, got, want) {
  const pass = got === want;
  if (!pass) bad++;
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}\n        got : ${got}\n        want: ${want}`);
}

// ── (F) ALL-IMPRISONED: Dusclops (Imprison/Rest/Splash) seals a Suicune whose whole moveset
// (Rest/Splash) it shares. gen3ou — a real Dusclops moveset shape.
{
  const { b, out } = mk('gen3ou',
    'Dusclops||Leftovers|Pressure|Imprison,Rest,Splash|Careful|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||');
  b.choose('p1', 'move 1'); b.choose('p2', 'move 2');
  check('F1 all-imprisoned request = FULL list, hidden (not Struggle-only)', reqSummary(b, 1),
    'moves=rest/splash[--] flags=maybeDisabled,maybeLocked update=false');
  const from = out.length;
  const ok = b.choose('p2', 'move 2');
  check('F2 all-imprisoned pick is ACCEPTED (no |error|), owner-only Struggle announce',
    `${!!ok} ${JSON.stringify(sideLines(out, from, 1))} p1=${JSON.stringify(sideLines(out, from, 0))}`,
    'true ["|-activate|p2a: Suicune|move: Struggle"] p1=[]');
  const from2 = b.log.length;
  b.choose('p1', 'move 3');
  const upd = b.log.slice(from2).join('\n');
  check('F3 the turn runs Struggle', /\|move\|p2a: Suicune\|Struggle\|p1a: Dusclops/.test(upd) ? 'struggle ran' : upd.slice(0, 300),
    'struggle ran');
  // A pick past the offered list is still `[Invalid choice]`.
  const { b: bb, out: o2 } = mk('gen3ou',
    'Dusclops||Leftovers|Pressure|Imprison,Rest,Splash|Careful|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||',
    'Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||');
  bb.choose('p1', 'move 1'); bb.choose('p2', 'move 2');
  const f4 = o2.length;
  const ok4 = bb.choose('p2', 'move 3');
  check('F4 all-imprisoned: an OUT-OF-RANGE pick is still refused',
    `${!!ok4} ${JSON.stringify(sideLines(o2, f4, 1))}`,
    'false ["|error|[Invalid choice] Can\'t move: Your Suicune doesn\'t have a move 3"]');
}

// ── (A) ACCUMULATION: an Imprison + Arena Trap Dugtrio (gen3customgame) vs a Jynx sharing
// Ice Beam + Splash (Psychic stays usable) — so a refused move AND a refused switch are both
// available in ONE decision.
const T1 = 'Dugtrio||Leftovers|ArenaTrap|Imprison,IceBeam,Splash|Jolly|,,,,,|M||||]Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||';
const T2 = 'Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||';
function acc(name, picks, want) {
  const { b, out } = mk('gen3customgame', T1, T2);
  b.choose('p1', 'move 1'); b.choose('p2', 'move 3');
  const rows = [];
  for (const p of picks) {
    const from = out.length;
    const ok = b.choose('p2', p);
    const lines = sideLines(out, from, 1).map(l => l.replace(/Can't (move|switch): .*/, "Can't $1…"));
    rows.push(`${p}:${ok ? 'ok' : 'refused'}${JSON.stringify(lines)}`);
  }
  check(name, `${rows.join(' ')} || ${reqSummary(b, 1)}`, want);
}
acc('A1 refused MOVE then refused SWITCH: re-request #2 keeps slot 1 disabled, no maybeLocked, trapped',
  ['move 1', 'switch 2'],
  'move 1:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] switch 2:refused["|error|[Unavailable choice] Can\'t switch…","|request|…"] || moves=icebeam/psychic/splash[Ds--] flags=maybeDisabled,trapped update=true');
acc('A2 refused SWITCH then refused MOVE: re-request #2 keeps trapped:true',
  ['switch 2', 'move 1'],
  'switch 2:refused["|error|[Unavailable choice] Can\'t switch…","|request|…"] move 1:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] || moves=icebeam/psychic/splash[Ds--] flags=maybeDisabled,trapped update=true');
acc('A3 two refused MOVES: both slots disabled on re-request #2',
  ['move 1', 'move 3'],
  'move 1:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] move 3:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] || moves=icebeam/psychic/splash[Ds-Ds] flags=maybeDisabled,maybeTrapped update=true');
// A4: `chooseMove` calls `getMoveRequestData()` FIRST (sim/side.ts:553), which re-derives
// `pokemon.maybeLocked = maybeLocked || maybeDisabled` — and Imprison's `maybeDisabled` is never
// cleared in singles — so `updateDisabledRequest` reports a change EVERY time: a repeated
// imprisoned pick is `[Unavailable choice]` + a re-request again (A6 is the no-Imprison contrast).
acc('A4 the SAME refused imprisoned move twice: STILL [Unavailable choice] + re-request (maybeDisabled re-derives maybeLocked)',
  ['move 1', 'move 1'],
  'move 1:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] move 1:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] || moves=icebeam/psychic/splash[Ds--] flags=maybeDisabled,maybeTrapped update=true');
acc('A5 the SAME refused hidden-trap switch twice: the 2nd -> [Invalid choice], NO re-request',
  ['switch 2', 'switch 2'],
  'switch 2:refused["|error|[Unavailable choice] Can\'t switch…","|request|…"] switch 2:refused["|error|[Invalid choice] Can\'t switch…"] || moves=icebeam/psychic/splash[---] flags=maybeDisabled,maybeLocked,trapped update=true');
// A6: a VISIBLE disable (the Choice Band lock) refused twice with NO Imprison foe: the first adds
// `"disabledSource":""` (a change); the second changes nothing -> `[Invalid choice]`, NO re-request.
{
  const { b, out } = mk('gen3customgame',
    'Snorlax||ChoiceBand|Immunity|Splash,Rest|Adamant|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||',
    'Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||');
  b.choose('p1', 'move 1'); b.choose('p2', 'move 1');
  const rows = [];
  for (const p of ['move 2', 'move 2']) {
    const from = out.length;
    const ok = b.choose('p1', p);
    rows.push(`${p}:${ok ? 'ok' : 'refused'}${JSON.stringify(sideLines(out, from, 0).map(l => l.replace(/Can't move: .*/, "Can't move…")))}`);
  }
  check('A6 the SAME refused visible-disable twice (no Imprison): 2nd -> [Invalid choice], NO re-request',
    `${rows.join(' ')} || ${reqSummary(b, 0)}`,
    'move 2:refused["|error|[Unavailable choice] Can\'t move…","|request|…"] move 2:refused["|error|[Invalid choice] Can\'t move…"] || moves=splash/rest[-Ds] flags=- update=true');
}
process.exit(bad ? 1 : 0);
