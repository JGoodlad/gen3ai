// PROBE: gen-3 RECYCLE — what counts as "consumed", the draw model, the emission forms.
//
// SETTLED 2026-08-18 (run it to re-confirm; do not re-derive from source):
//   dex: accuracy true (NEVER-MISS), category Status, target SELF, pp 10, priority 0,
//        flags {metronome:1} — NO `protect`, NO `snatch`, NO `mirror`.
//
//   DRAW MODEL — Recycle draws NOTHING, on SUCCESS or on FAILURE.
//     never-miss => no accuracy roll; the restore is draw-free; `landed` is FALSE
//     (a speed-TIED mirror draws 7 raw calls on a Recycle turn == the both-Splash
//     control's 7, so the in-tryMoveHit eachEvent('Update') is NOT fired).
//
//   SUCCESS  : `|move|<u>|Recycle|<u>` then `|-item|<u>|<Item>|[from] move: Recycle`
//   FAILURE  : `|move|<u>|Recycle||[still]` (target BLANKED) then `|-fail|<u>` (bare)
//
//   RESTORABLE (lastItem is set) — the item was CONSUMED by the holder itself:
//     * a berry eaten at its own threshold (Sitrus/Oran/Lum/Figy-family)   -> YES
//     * a PINCH berry eaten at <=1/4 hp (Liechi/Salac/...)                 -> YES
//     * White Herb spent on a stat drop (the `useItem` path)               -> YES
//   NOT RESTORABLE (lastItem untouched) — the item was TAKEN, not consumed:
//     * KNOCK OFF                                                         -> FAIL
//     * THIEF / COVET / TRICK                                             -> FAIL
//     * FOCUS BAND (procs WITHOUT consuming: `-activate|item: Focus Band`,
//       no `-enditem`, the item stays held)  -> Recycle FAILS (item held)
//     * a mon that never held an item                                     -> FAIL
//     * a mon currently HOLDING an item                                   -> FAIL
//
//   lastItem PERSISTS across a switch-out and back (probe J).
//   itemKnockedOff PERSISTS across a Recycle restore: the restored item is still
//     un-stealable by a later Thief (probe Q2) — Recycle must NOT clear that flag.
//   A restored item is IMMEDIATELY re-usable: a Sitrus restored below 1/2 hp re-eats
//     on the SAME turn (`-item` then `-enditem [eat]` + `-heal`) at the runAction-tail
//     eachEvent('Update'); ditto a Liechi below 1/4 and a White Herb with a live
//     negative stage. The cycle repeats indefinitely (eat -> Recycle -> eat -> ...).
//   CHOICE BAND can never reach lastItem (it is only ever removed by takeItem), so
//     "does Recycle re-arm the Choice lock" is UNREACHABLE in gen 3.
//   Recycle is category Status => BLOCKED BY TAUNT (probe R: the request drops to Struggle).
const path = require('path'); const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const d = Dex.mod('gen3').moves.get('recycle');
console.log('dex recycle:', JSON.stringify({ num: d.num, acc: d.accuracy, cat: d.category, target: d.target, pp: d.pp, pri: d.priority, type: d.type, flags: d.flags }));
console.log('recycle onHit:', d.onHit.toString().replace(/\s+/g, ' '));
const fb = Dex.mod('gen3').items.get('focusband');
console.log('focusband onDamage (no eat/useItem => no consumption):', fb.onDamage.toString().replace(/\s+/g, ' '));

let draws = [];
for (const m of ['random', 'randomChance', 'sample']) {
  const o = PRNG.prototype[m];
  PRNG.prototype[m] = function (...a) { const r = o.apply(this, a); if (m === 'random') draws.push(`random(${a})->${r}`); return r; };
}
const KEEP = /^\|(-item|-enditem|-fail|-heal|-damage|-activate|-status|-curestatus|-boost|-unboost|-clearnegativeboost|move|switch|cant|turn|faint|win)\|/;

async function run(label, p1, p2, script, seed) {
  const s = new BattleStream(); const ch = [];
  (async () => { for await (const c of s) ch.push(c); })(); draws = [];
  s.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed || [9, 9, 9, 9])}}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 150));
  let seen = 0, dseen = draws.length; const out = [];
  const lines = () => ch.filter(c => !c.startsWith('sideupdate')).join('\n').split('\n');
  console.log(`\n=== ${label}`);
  for (const c of script) {
    s.write(c); await new Promise(r => setTimeout(r, 180));
    const all = lines(); const fresh = all.slice(seen).filter(l => KEEP.test(l)); seen = all.length;
    const nd = draws.slice(dseen); dseen = draws.length;
    console.log(`  > ${c.replace(/\n/g, ' | ')}`);
    for (const l of fresh) { console.log('      ' + l); out.push(l); }
    console.log(`      RAW-PRNG DRAWS(${nd.length}): ${nd.join('  ') || '(none)'}`);
    const su = ch.filter(x => x.startsWith('sideupdate\np1')).slice(-1)[0];
    if (su) { const m = su.match(/\|request\|(.*)/); if (m) { const j = JSON.parse(m[1]); if (j.active) console.log('      REQ p1:', JSON.stringify(j.active[0].moves)); } }
  }
  return out;
}

const SN = (item, moves, spe) => `Snorlax||${item}|thickfat|${moves}|Hardy|85,85,85,85,85,85|M||||`;
const ZP = (item, moves) => `Zapdos||${item}|pressure|${moves}|Hardy|85,85,85,85,85,85|N||||`;

(async () => {
  // A — berry eaten by its OWN trigger (Lum on status); restore; repeat the cycle.
  await run('A lum eaten on status -> Recycle -> eaten again -> Recycle',
    SN('lumberry', 'recycle,splash'), ZP('leftovers', 'thunderwave,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // B — Sitrus eaten at the 1/2 threshold; Recycle while STILL below it => SAME-TURN re-eat.
  await run('B sitrus threshold eat; Recycle below threshold => immediate re-eat',
    SN('sitrusberry', 'recycle,splash'), ZP('leftovers', 'superfang,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // C — PINCH berry (Liechi, 1/4). Three Super Fangs to cross the threshold.
  await run('C liechi pinch eat -> Recycle => immediate re-eat (+1 atk again)',
    SN('liechiberry', 'recycle,splash'), ZP('leftovers', 'superfang,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // D — KNOCK OFF (takeItem) => NOT restorable.
  await run('D knock off -> Recycle FAILS', SN('leftovers', 'recycle,splash'), ZP('leftovers', 'knockoff,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // E — TRICK (takeItem) => NOT restorable.
  await run('E trick takes the item -> Recycle FAILS', SN('leftovers', 'recycle,splash'), ZP('', 'trick,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // F/G — never had an item / currently holding one.
  await run('F never had an item -> Recycle FAILS', SN('', 'recycle,splash'), ZP('leftovers', 'splash,splash'), ['>p1 move 1\n>p2 move 1']);
  await run('G holding an item -> Recycle FAILS', SN('leftovers', 'recycle,splash'), ZP('leftovers', 'splash,splash'), ['>p1 move 1\n>p2 move 1']);

  // H — WHITE HERB (useItem) => restorable, and re-arms on a live negative stage.
  await run('H white herb spent, def dropped again, Recycle => immediate re-use',
    SN('whiteherb', 'recycle,splash'), ZP('leftovers', 'screech,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // J — lastItem survives a switch OUT and BACK.
  await run('J eat, switch out, switch back, Recycle => restores',
    `Snorlax||lumberry|thickfat|recycle,splash|Hardy|85,85,85,85,85,85|M||||]Blissey||leftovers|naturalcure|splash|Hardy|85,85,85,85,85,85|F||||`,
    ZP('leftovers', 'thunderwave,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 switch 2\n>p2 move 2', '>p1 switch 2\n>p2 move 2', '>p1 move 1\n>p2 move 2']);

  // K — CHOICE BAND knocked off => NOT restorable (no consumption path exists for it).
  await run('K choice band knocked off -> Recycle FAILS',
    `Snorlax||choiceband|thickfat|recycle,splash|Hardy|85,85,85,85,85,85|M||||`, ZP('leftovers', 'knockoff,splash'),
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2']);

  // Q — itemKnockedOff SURVIVES the restore: a later Thief still cannot take the item.
  await run('Q knocked-off flag persists across a Recycle restore (Thief is refused)',
    SN('lumberry', 'recycle,splash'), `Zapdos||leftovers|pressure|thunderwave,trick,knockoff,thief|Hardy|85,85,85,85,85,85|N||||`,
    ['>p1 move 2\n>p2 move 1', '>p1 move 2\n>p2 move 2', '>p1 move 2\n>p2 move 3', '>p1 move 1\n>p2 move 4', '>p1 move 2\n>p2 move 4', '>p1 move 2\n>p2 move 4']);

  // R — TAUNT blocks Recycle (category Status): the request collapses to Struggle.
  await run('R taunt blocks Recycle selection', SN('lumberry', 'recycle,splash'), ZP('leftovers', 'taunt,splash'),
    ['>p1 move 2\n>p2 move 1']);

  // T — the DRAW-COUNT gate: a SPEED-TIED mirror. Recycle turn == both-Splash control.
  const TIE = (item, moves) => `Snorlax||${item}|thickfat|${moves}|Hardy|85,85,85,85,85,85|M||||`;
  await run('T1 TIED mirror control: splash/splash (expect 7 raw draws)', TIE('lumberry', 'recycle,splash,thunderwave'), TIE('leftovers', 'splash,thunderwave'), ['>p1 move 2\n>p2 move 1']);
  await run('T2 TIED mirror: Recycle SUCCESS (expect 7 raw draws)', TIE('lumberry', 'recycle,splash,thunderwave'), TIE('leftovers', 'splash,thunderwave'), ['>p1 move 2\n>p2 move 2', '>p1 move 1\n>p2 move 1']);
  await run('T3 TIED mirror: Recycle FAIL (expect 7 raw draws)', TIE('', 'recycle,splash,thunderwave'), TIE('leftovers', 'splash,thunderwave'), ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1']);

  // L — FOCUS BAND procs WITHOUT consuming (sweep seeds until it fires).
  for (let i = 0; i < 80; i++) {
    const sd = [i * 7 + 1, i * 13 + 2, i * 29 + 3, i * 3 + 5];
    const out = await run(`L focus band proc sweep seed=${JSON.stringify(sd)}`,
      `Shedinja||focusband|wonderguard|recycle,splash|Hardy|85,85,85,85,85,85|N||||]Snorlax||leftovers|thickfat|splash|Hardy|85,85,85,85,85,85|M||||`,
      ZP('leftovers', 'flamethrower,splash'), ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'], sd);
    if (out.some(l => l.includes('Focus Band'))) { console.log('  ^^^ PROCCED: `-activate item: Focus Band`, NO `-enditem` => not consumed; the follow-up Recycle FAILS (item still held)'); break; }
  }
})();
