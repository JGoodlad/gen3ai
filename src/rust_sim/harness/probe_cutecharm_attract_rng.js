// probe_cutecharm_attract_rng.js — settle CUTE CHARM + the ATTRACT volatile vs the
// resolved gen3 sim. Hypotheses (resolved dist):
//  - Cute Charm onDamagingHit: contact + damage → randomChance(1,3) UNCONDITIONALLY
//    (the GENDER check lives INSIDE attract.onStart → the roll draws even for
//    same-gender / genderless; the volatile then fails draw-free). Same DamagingHit
//    position as Static (after the move's own secondary).
//  - attract volatile: onStart gates M↔F opposite genders + runEvent('Attract')
//    (Oblivious blocks); onBeforeMovePriority 2 (confusion 3 > attract 2 > par 1);
//    onBeforeMove: '-activate' ALWAYS, then randomChance(1,2) → cant on true;
//    onUpdate removes it when the SOURCE leaves the field; clearVolatile on the
//    holder's own switch-out; NO duration.
// Run: node src/rust_sim/harness/probe_cutecharm_attract_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

// p2 = Cute Charm holder (female), p1 = attacker whose gender varies.
async function genders() {
  console.log('=== the roll vs gender: draw counts + attract landing (Scratch = contact)');
  for (const [label, atkGender, defGender, atkSpecies] of [
    ['M into F', 'M', 'F', 'Zangoose'],
    ['F into F', 'F', 'F', 'Zangoose'],
    ['genderless into F', '', 'F', 'Metagross'],
    ['M into F (non-contact ctl)', 'M', 'F', 'Zangoose'],
  ]) {
    const nonContact = label.includes('non-contact');
    const teams = [
      [mon(atkSpecies, [nonContact ? 'swift' : 'scratch'], { ability: nonContact ? 'Clear Body' : 'Immunity', gender: atkGender, evs: { spe: 252 } })],
      [mon('Miltank', ['splash'], { ability: 'Cute Charm', gender: defGender })],
    ];
    for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]) {
      const r = await run(teams, seed, Array(3).fill(['move 1', 'move 1']), {
        onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'] }),
      });
      r.perDecision.forEach((d, i) => {
        const ev = d.lines.filter((l) => l.includes('Attract') || l.includes('|cant|'));
        console.log(`${label} seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r.states[i])}`);
      });
    }
  }
}

// The attract onBeforeMove: position vs confusion(3)/para(1); the -activate-always shape.
async function ladder() {
  console.log('=== attract onBeforeMove ladder: attracted + paralyzed holder');
  // p1 M Zangoose (attracted turn1 via scripted Attract MOVE from p2 — use cutecharm proc
  // instead: p2 female Miltank cute charm; get attract to land, then paralyze p1 via TWave).
  const teams = [
    [mon('Zangoose', ['scratch', 'splash'], { ability: 'Immunity', gender: 'M', evs: { spe: 252 } })],
    [mon('Miltank', ['thunderwave', 'splash'], { ability: 'Cute Charm', gender: 'F' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]]) {
    const r = await run(teams, seed, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
      onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'], status: b.p1.active[0].status }),
    });
    r.perDecision.forEach((d, i) => {
      const ev = d.lines.filter((l) => l.includes('Attract') || l.includes('|cant|'));
      console.log(`seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r.states[i])}`);
    });
  }
}

// Clearing: source switch-out removes it (onUpdate); holder switch-out clears it.
async function clearing() {
  console.log('=== clearing: source leaves vs holder leaves');
  const teams = [
    [mon('Zangoose', ['scratch', 'splash'], { ability: 'Immunity', gender: 'M', evs: { spe: 252 } }), mon('Rattata', ['scratch'], { ability: 'Guts', gender: 'M' })],
    [mon('Miltank', ['splash'], { ability: 'Cute Charm', gender: 'F' }), mon('Chansey', ['splash'], { ability: 'Natural Cure', gender: 'F' })],
  ];
  // Find a seed where attract lands t1; then p2 switches Miltank out t2 → does p1's attract clear?
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]]) {
    const r = await run(teams, seed, [['move 1', 'move 1'], ['move 2', 'switch 2'], ['move 2', 'move 1']], {
      onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'] }),
    });
    console.log(`sourceLeaves seed=${seed[0]}: ${JSON.stringify(r.states)} endLines=${JSON.stringify(r.perDecision[1] ? r.perDecision[1].lines.filter((l) => l.includes('Attract')) : [])}`);
  }
  // Holder leaves: p1 switches out after being attracted → volatile gone on return.
  for (const seed of [[2, 2, 2, 2], [4, 4, 4, 4]]) {
    const r = await run(teams, seed, [['move 1', 'move 1'], ['switch 2', 'move 1'], ['switch 2', 'move 1'], ['move 2', 'move 1']], {
      onBoundary: (b) => ({ p1active: b.p1.active[0].species.id, attracted: !!b.p1.active[0].volatiles['attract'] }),
    });
    console.log(`holderLeaves seed=${seed[0]}: ${JSON.stringify(r.states)}`);
  }
}

// Behind a Substitute + Oblivious: does the CC roll still draw? does attract stick?
async function subAndOblivious() {
  console.log('=== behind a sub + Oblivious');
  const subTeams = [
    [mon('Zangoose', ['substitute', 'scratch'], { ability: 'Immunity', gender: 'M', evs: { spe: 252 } })],
    [mon('Miltank', ['splash'], { ability: 'Cute Charm', gender: 'F' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [4, 4, 4, 4]]) {
    const r = await run(subTeams, seed, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], {
      onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'], sub: !!b.p1.active[0].volatiles['substitute'] }),
    });
    r.perDecision.forEach((d, i) => console.log(`sub seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] ${JSON.stringify(r.states[i])}`));
  }
  const oblTeams = [
    [mon('Slowbro', ['scratch'], { ability: 'Oblivious', gender: 'M', evs: { spe: 252 } })],
    [mon('Miltank', ['splash'], { ability: 'Cute Charm', gender: 'F' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [4, 4, 4, 4]]) {
    const r = await run(oblTeams, seed, Array(2).fill(['move 1', 'move 1']), {
      onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'] }),
    });
    r.perDecision.forEach((d, i) => console.log(`oblivious seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] ${JSON.stringify(r.states[i])}`));
  }
}

(async () => { await genders(); await ladder(); await clearing(); await subAndOblivious(); })().catch((e) => { console.error(e); process.exit(1); });
