// probe_transform_speed.js — settle the ONE non-obvious draw-relevant fact about Transform:
// the value of the CACHED `pokemon.speed` (what `eachEvent`'s speedSort reads) in the window
// between the Transform action and the residual's `updateSpeed()`.
//
// `transformInto` calls `setSpecies(target, effect, /*isTransform*/true)`, which sets
// `storedStats = spreadModify(TARGET.baseStats, THIS.set)` and then `this.speed =
// storedStats.spe`. Only AFTER that does transformInto overwrite storedStats with the
// TARGET's own storedStats — WITHOUT re-setting `this.speed`. So for the rest of the turn the
// tie-shuffle speed is the "hybrid" stat (target base stats, transformer's EVs/IVs/nature),
// not the copied one.
//
// Also probes: Protect vs Transform (transform has NO `protect` flag), a Ghost target, and
// the copied-move Choice-lock release.
// Run: node harness/probe_transform_speed.js
'use strict';
const path = require('path');
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim'));

const SEED = [5, 4, 3, 2];
const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|' && !l.startsWith('|upkeep');

async function main() {
  // Instrument speedSort to record the CACHED speeds it sorts on, per call, during the turn.
  console.log('############ SPEED: what does eachEvent speedSort read after Transform? ############');
  const teams = [
    [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
    // Snorlax with a DIFFERENT spread from Ditto so the "hybrid" and "copied" speeds differ.
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat', nature: 'Adamant',
      evs: { hp: 252, atk: 252, spe: 4 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 3 } })],
  ];
  const gen3 = Dex.mod('gen3');
  const lax = gen3.species.get('Snorlax');
  console.log(`  Snorlax base spe=${lax.baseStats.spe}`);
  console.log('  hybrid  = spreadModify(Snorlax.baseStats, DITTO set: Serious/0EV/31IV) → 2*30+31+0 +5 = 96');
  console.log('  copied  = Snorlax storedStats.spe (Adamant, 4 EV, 3 IV) → 69');

  const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 2', 'move 1']], {
    onBoundary: (b) => ({ p1speed: b.sides[0].active[0].speed, p1stored: b.sides[0].active[0].storedStats.spe,
      p2speed: b.sides[1].active[0].speed }),
    // hook injected below
  });
  r.perDecision.forEach((d, i) => {
    console.log(`  d${i}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`      lines=${JSON.stringify(d.lines.filter(KEEP))}`);
  });
  r.states.forEach((s, i) => console.log(`  state${i}=${JSON.stringify(s)}`));
  console.log('  READ: turn-1 has exactly ONE speedSort tie draw ⇒ the two post-action eachEvent');
  console.log('        Updates saw 96 (hybrid, no tie vs 69) and only the POST-residual one saw 69.');
  console.log('        A "copied speed" model would give THREE. A "unchanged (132)" model would give ONE too —');
  console.log('        so run the CONTROL below to separate 96 from 132.');

  // CONTROL: give the Ditto a spread whose hybrid speed EQUALS the target's own speed. If the
  // cache were left at Ditto's ORIGINAL speed we would see 0/1 ties; with the hybrid we see 3.
  console.log('\n############ SPEED CONTROL: hybrid == target speed ⇒ ties at EVERY Update ############');
  {
    // Ditto with spe IV 3 + 4 spe EVs → hybrid = spreadModify(Snorlax base 30, that set).spe = 69.
    const t2 = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber', nature: 'Adamant',
        evs: { hp: 252, atk: 252, spe: 4 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 3 } })],
      [mon('Snorlax', ['splash'], { ability: 'Thick Fat', nature: 'Adamant',
        evs: { hp: 252, atk: 252, spe: 4 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 3 } })],
    ];
    const r2 = await run(t2, SEED, [['move 1', 'move 1']], {
      onBoundary: (b) => ({ p1speed: b.sides[0].active[0].speed, p1stored: b.sides[0].active[0].storedStats.spe }),
    });
    r2.perDecision.forEach((d, i) => console.log(`  c-d${i}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`));
    console.log(`  c-state=${JSON.stringify(r2.states)}`);
  }

  // The DECISIVE test: hybrid != copied AND hybrid != Ditto-own AND copied == target.
  // Ditto Serious/31 → own spe 132; hybrid vs a MEWTWO target (base 130) = 2*130+31+5 = 296;
  // Mewtwo's own (0 EV, 0 spe IV, Brave -spe) is far lower. Give Mewtwo spe such that its OWN
  // stored spe == 132 (== Ditto's own) — then "unchanged" ties everywhere, "hybrid" never ties.
  console.log('\n############ SPEED DECISIVE: target stored spe == Ditto own spe, hybrid far away ############');
  {
    // Mewtwo base spe 130. Want stored spe == 132: (2*130+iv+ev/4)+5 = 132 → 2*130 = 260 > 127. Not possible.
    // Instead: target = Shuckle (base spe 5). stored = (10+iv+ev/4)+5. Want 132 → impossible.
    // Use the reverse framing: pick a target whose OWN spe equals DITTO's own 132 exactly.
    // Meganium base spe 80 → (160+31+0)+5 = 196. Slowbro base 30 → 96.  Machamp base 55 → 146.
    // Golem base 45 → 126.  Rhydon base 40 → 116.  Sudowoodo base 30 → 96.  Piloswine base 50 → 136.
    // Quagsire base 35 → 106.  base 48 (Ditto) → 132.  Chansey base 50 → 136 with 0 IV → (100+0)+5=105.
    // Chansey with iv 27 → (100+27)+5 = 132.  Use Chansey iv spe 27.
    const t3 = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Chansey', ['splash'], { ability: 'Natural Cure', ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 27 } })],
    ];
    const g = Dex.mod('gen3');
    console.log(`  Chansey base spe=${g.species.get('Chansey').baseStats.spe} → own stored = 2*50+27+5 = 132 (== Ditto's own)`);
    console.log('  hybrid = spreadModify(Chansey base 50, DITTO 31 IV set) = 2*50+31+5 = 136 (≠132) ⇒');
    console.log('  "unchanged/copied" ⇒ ties at ALL 3 Updates (3 draws); "hybrid" ⇒ tie only after the residual (1 draw).');
    const r3 = await run(t3, SEED, [['move 1', 'move 1']], {
      onBoundary: (b) => ({ p1speed: b.sides[0].active[0].speed, p1stored: b.sides[0].active[0].storedStats.spe,
        p2speed: b.sides[1].active[0].speed, p2stored: b.sides[1].active[0].storedStats.spe }),
    });
    r3.perDecision.forEach((d, i) => console.log(`  x-d${i}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`));
    console.log(`  x-state=${JSON.stringify(r3.states)}`);
  }

  // ============================================================ PROTECT (transform has no protect flag)
  console.log('\n############ PROTECT vs TRANSFORM (flags carry NO `protect`) ############');
  {
    const t = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Snorlax', ['protect', 'splash'], { ability: 'Thick Fat', nature: 'Adamant',
        evs: { hp: 252, atk: 252 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 3 } })],
    ];
    const r4 = await run(t, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ sp: b.sides[0].active[0].species.id, tr: b.sides[0].active[0].transformed }) });
    r4.perDecision.forEach((d, i) => console.log(`  p-d${i}: draws=${d.nexts} lines=${JSON.stringify(d.lines.filter(KEEP))}`));
    console.log(`  p-state=${JSON.stringify(r4.states)}`);
  }

  // ============================================================ GHOST target (status ignores immunity)
  console.log('\n############ GHOST target ############');
  {
    const t = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Gengar', ['splash'], { ability: 'Levitate' })],
    ];
    const r5 = await run(t, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ sp: b.sides[0].active[0].species.id, ty: b.sides[0].active[0].types, ab: b.sides[0].active[0].ability }) });
    r5.perDecision.forEach((d, i) => console.log(`  g-d${i}: draws=${d.nexts} lines=${JSON.stringify(d.lines.filter(KEEP))}`));
    console.log(`  g-state=${JSON.stringify(r5.states)}`);
  }

  // ============================================================ MIMIC by a TRANSFORMED user
  console.log('\n############ MIMIC used BY a transformed mon (source.transformed ⇒ fail) ############');
  {
    // p2 Snorlax has Mimic + Body Slam. p1 Ditto transforms (t1), so it now HAS Mimic.
    // t2: p2 uses Body Slam first (so p1 has a lastMove to mimic), p1 uses the copied Mimic.
    const t = [
      [mon('Ditto', ['transform'], { ability: 'Limber', evs: { spe: 252 } })],
      [mon('Blissey', ['mimic', 'splash', 'softboiled', 'toxic'], { ability: 'Natural Cure' })],
    ];
    const r6 = await run(t, SEED, [['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']],
      { onBoundary: (b) => ({ sl: b.sides[0].active[0].moveSlots.map((m) => `${m.id}:${m.pp}`), tr: b.sides[0].active[0].transformed }) });
    r6.perDecision.forEach((d, i) => console.log(`  m-d${i}: draws=${d.nexts} lines=${JSON.stringify(d.lines.filter(KEEP))}`));
    r6.states.forEach((s, i) => console.log(`  m-state${i}=${JSON.stringify(s)}`));
  }
}
main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
