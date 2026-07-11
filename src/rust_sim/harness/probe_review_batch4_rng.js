// probe_review_batch4_rng.js — ADVERSARIAL-REVIEW probe for batch 4 (fresh
// constructions, fresh seeds — independent of the builder's probes). Verifies the
// claimed draw models against the resolved Dex.mod('gen3') sim:
//  A Truant: cant position + PP on loaf; voluntary-switch vs post-residual-DoT arming
//  B Inner Focus vs Shield Dust vs control: draw-count + cant contrast
//  C Shadow Tag Wobbuffet mirror: trapped + endTurn draw parity vs a Guts control
//  D Cute Charm: genderless / same-gender roll gating; attract 1/2 cant; switch-clear
//  E Color Change: behind a sub; Struggle; later chart reads (SE through override)
//  F Forecast: rain follow + end-revert; Cloud Nine composition (deferral check)
//  G King's Rock: roll POSITION among [own secondary, contact proc]; behind a sub;
//    Seismic Toss; Serene Grace threshold
//  H Focus Band: per-hit draw (non-lethal, residual, confusion); Explosion self-KO;
//    lethal survive at 1 HP
// Run: node src/rust_sim/harness/probe_review_batch4_rng.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const S1 = [7, 3, 9, 1];
const S2 = [101, 55, 202, 13];

function cants(d) { return d.lines.filter((l) => l.includes('|cant|')); }
function grab(d, pat) { return d.lines.filter((l) => l.includes(pat)); }

async function secA() {
  console.log('\n===== A. TRUANT =====');
  // A1 cadence + PP + cant position (Slaking Return vs Registeel splash)
  const t = [
    [mon('Slaking', ['return'], { ability: 'Truant' })],
    [mon('Registeel', ['splash'], { ability: 'Clear Body' })],
  ];
  const r = await run(t, S1, Array(5).fill(['move 1', 'move 1']), {
    onBoundary: (b) => ({ pp: b.p1.active[0].moveSlots[0].pp, tt: b.p1.active[0].truantTurn }),
  });
  r.perDecision.forEach((d, i) =>
    console.log(`A1 t${i + 1}: draws=[${fmtCalls(d.calls)}] cant=${JSON.stringify(cants(d))} st=${JSON.stringify(r.states[i])}`));

  // A2 paralyzed loaf: no para roll on the loaf turn? (fresh species: Manectric TWave)
  const t2 = [
    [mon('Slaking', ['return'], { ability: 'Truant' })],
    [mon('Manectric', ['thunderwave', 'splash'], { ability: 'Static', evs: { spe: 252 } })],
  ];
  const r2 = await run(t2, S2, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ st: b.p1.active[0].status, tt: b.p1.active[0].truantTurn }),
  });
  r2.perDecision.forEach((d, i) =>
    console.log(`A2 t${i + 1}: draws=[${fmtCalls(d.calls)}] cant=${JSON.stringify(cants(d))} st=${JSON.stringify(r2.states[i])}`));

  // A3 voluntary switch-in mid-battle: moves its first full turn
  const t3 = [
    [mon('Zangoose', ['splash'], { ability: 'Immunity' }), mon('Slaking', ['return'], { ability: 'Truant' })],
    [mon('Registeel', ['splash'], { ability: 'Clear Body' })],
  ];
  const r3 = await run(t3, S1, [['switch 2', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], {
    onBoundary: (b) => ({ act: b.p1.active[0].species.id, tt: b.p1.active[0].truantTurn }),
  });
  r3.perDecision.forEach((d, i) =>
    console.log(`A3 t${i + 1}: cant=${JSON.stringify(cants(d))} st=${JSON.stringify(r3.states[i])}`));

  // A4 post-residual replacement (sand DoT KOs Shedinja at endTurn): Slaking LOAFS first full turn
  const t4 = [
    [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' }), mon('Slaking', ['return'], { ability: 'Truant' })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream' })],
  ];
  const r4 = await run(t4, S1, [['move 1', 'move 1'], ['switch 2', null], ['move 1', 'move 1'], ['move 1', 'move 1']], {
    onBoundary: (b) => ({ act: b.p1.active[0].species.id, tt: b.p1.active[0].truantTurn, hp: b.p1.active[0].hp }),
  });
  r4.perDecision.forEach((d, i) =>
    console.log(`A4 b${i}: cant=${JSON.stringify(cants(d))} faint=${JSON.stringify(grab(d, 'faint'))} st=${JSON.stringify(r4.states[i])}`));
}

async function secB() {
  console.log('\n===== B. INNER FOCUS vs SHIELD DUST vs CONTROL =====');
  // Persian Bite (30% flinch) into: Dragonite (Inner Focus) / Dustox (Shield Dust) / Snorlax (Thick Fat)
  for (const [label, def, abil] of [
    ['control(ThickFat Snorlax)', 'Snorlax', 'Thick Fat'],
    ['InnerFocus(Dragonite)', 'Dragonite', 'Inner Focus'],
    ['ShieldDust(Dustox)', 'Dustox', 'Shield Dust'],
  ]) {
    for (const seed of [S1, S2]) {
      const t = [
        [mon('Persian', ['bite'], { ability: 'Limber', evs: { spe: 252 } })],
        [mon(def, ['splash'], { ability: abil, evs: { hp: 252 } })],
      ];
      const r = await run(t, seed, Array(4).fill(['move 1', 'move 1']));
      const out = r.perDecision.map((d, i) => {
        const fl = d.calls.filter((c) => c.kind === 'random' && c.args[0] === 100);
        return `t${i + 1}[r100=${fl.map((c) => c.ret).join(',')} cant=${cants(d).length}]`;
      }).join(' ');
      console.log(`B ${label} seed=${JSON.stringify(seed)}: ${out}`);
    }
  }
}

async function secC() {
  console.log('\n===== C. SHADOW TAG Wobbuffet mirror =====');
  for (const [label, abil] of [['ShadowTag-mirror', 'Shadow Tag'], ['Guts-control', 'Guts']]) {
    const t = [
      [mon('Wobbuffet', ['splash', 'counter'], { ability: abil }), mon('Zangoose', ['splash'], { ability: 'Immunity' })],
      [mon('Wobbuffet', ['splash', 'counter'], { ability: abil }), mon('Zangoose', ['splash'], { ability: 'Immunity' })],
    ];
    const r = await run(t, S2, Array(3).fill(['move 1', 'move 1']), {
      onBoundary: (b) => ({ p1trap: b.p1.active[0].trapped, p2trap: b.p2.active[0].trapped }),
    });
    r.perDecision.forEach((d, i) =>
      console.log(`C ${label} t${i + 1}: nexts=${d.nexts} draws=[${fmtCalls(d.calls)}] st=${JSON.stringify(r.states[i])}`));
  }
  // switch attempt while ST-trapped: rejected?
  const t2 = [
    [mon('Zangoose', ['splash'], { ability: 'Immunity' }), mon('Skarmory', ['splash'], { ability: 'Keen Eye' })],
    [mon('Wobbuffet', ['splash', 'counter'], { ability: 'Shadow Tag' })],
  ];
  const r2 = await run(t2, S1, [['move 1', 'move 1'], ['switch 2', 'move 1'], ['move 1', 'move 1']], {
    onBoundary: (b) => ({ act: b.p1.active[0].species.id, trap: b.p1.active[0].trapped }),
  });
  r2.perDecision.forEach((d, i) =>
    console.log(`C switch-try b${i}: act/trap=${JSON.stringify(r2.states[i])} err=${JSON.stringify(grab(d, 'error'))}`));
}

async function secD() {
  console.log('\n===== D. CUTE CHARM + ATTRACT =====');
  // roll gating: M->F attract possible; F->F roll drawn, no attract; genderless roll drawn, no attract
  for (const [label, atk, g] of [
    ['M-into-F', 'Machamp', 'M'],
    ['F-into-F', 'Machamp', 'F'],
    ['genderless(Metagross)', 'Metagross', ''],
  ]) {
    for (const seed of [S1, S2]) {
      const t = [
        [mon(atk, ['strength', 'splash'], { ability: atk === 'Metagross' ? 'Clear Body' : 'Guts', gender: g })],
        [mon('Wigglytuff', ['splash'], { ability: 'Cute Charm', gender: 'F' }), mon('Chansey', ['splash'], { ability: 'Natural Cure', gender: 'F' })],
      ];
      const r = await run(t, seed, Array(4).fill(['move 1', 'move 1']), {
        onBoundary: (b) => ({ attracted: !!b.p1.active[0].volatiles['attract'] }),
      });
      const out = r.perDecision.map((d, i) => {
        const cc = d.calls.filter((c) => c.kind === 'randomChance').map((c) => `rc(${c.args})=${c.ret}`);
        return `t${i + 1}[${cc.join(' ')} attract-lines=${grab(d, 'Attract').length} cant=${cants(d).length} att=${r.states[i].attracted}]`;
      }).join(' ');
      console.log(`D ${label} seed=${JSON.stringify(seed)}: ${out}`);
    }
  }
  // switch-clear: after attraction, SOURCE (Wigglytuff) switches out -> attract ends on attacker
  const t2 = [
    [mon('Machamp', ['strength', 'splash'], { ability: 'Guts', gender: 'M' })],
    [mon('Wigglytuff', ['splash'], { ability: 'Cute Charm', gender: 'F' }), mon('Chansey', ['splash'], { ability: 'Natural Cure', gender: 'F' })],
  ];
  // hunt a seed where attraction lands turn 1, then switch source out turn 2
  for (const seed of [S1, S2, [42, 42, 42, 42], [3, 14, 15, 92]]) {
    const r = await run(t2, seed, [['move 1', 'move 1'], ['move 2', 'switch 2'], ['move 2', 'move 1']], {
      onBoundary: (b) => ({ att: !!b.p1.active[0].volatiles['attract'] }),
    });
    console.log(`D clear seed=${JSON.stringify(seed)}: t1 att=${r.states[0] && r.states[0].att}` +
      ` | t2 endlines=${JSON.stringify(r.perDecision[1] ? grab(r.perDecision[1], 'Attract') : [])} att=${r.states[1] && r.states[1].att}`);
  }
}

async function secE() {
  console.log('\n===== E. COLOR CHANGE =====');
  // E1 behind a sub: Kecleon subs (faster than Shuckle), Water Gun hits the sub -> NO typechange
  const t1 = [
    [mon('Kecleon', ['substitute', 'splash'], { ability: 'Color Change', evs: { hp: 252 } })],
    [mon('Shuckle', ['watergun'], { ability: 'Sturdy' })],
  ];
  const r1 = await run(t1, S1, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, sub: !!b.p1.active[0].volatiles['substitute'] }),
  });
  r1.perDecision.forEach((d, i) =>
    console.log(`E1 t${i + 1}: typechange=${JSON.stringify(grab(d, 'typechange'))} st=${JSON.stringify(r1.states[i])}`));

  // E2 Struggle into Kecleon: typeless -> no change?
  const t2 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change', evs: { hp: 252 } })],
    [mon('Zangoose', ['struggle'], { ability: 'Immunity' })],
  ];
  const r2 = await run(t2, S1, Array(2).fill(['move 1', 'move 1']), {
    onBoundary: (b) => ({ types: b.p1.active[0].types }),
  });
  r2.perDecision.forEach((d, i) =>
    console.log(`E2 t${i + 1}: lines=${JSON.stringify(grab(d, 'typechange'))} st=${JSON.stringify(r2.states[i])} moveline=${JSON.stringify(grab(d, '|move|p2a'))}`));

  // E3 chart reads THROUGH the override: WaterGun -> Water; then Thunderbolt SE; then becomes Electric; then EQ SE
  const t3 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change', evs: { hp: 252 } })],
    [mon('Magneton', ['watergun', 'thunderbolt', 'earthquake'], { ability: 'Magnet Pull' })],
  ];
  const r3 = await run(t3, S2, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 3'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, hp: b.p1.active[0].hp }),
  });
  r3.perDecision.forEach((d, i) =>
    console.log(`E3 t${i + 1}: se=${JSON.stringify(grab(d, 'supereffective'))} tc=${JSON.stringify(grab(d, 'typechange'))} st=${JSON.stringify(r3.states[i])}`));
}

async function secF() {
  console.log('\n===== F. FORECAST (deferral verification) =====');
  // F1 rain-follow + end-revert
  const t1 = [
    [mon('Castform', ['splash'], { ability: 'Forecast' })],
    [mon('Ludicolo', ['raindance', 'splash'], { ability: 'Swift Swim' })],
  ];
  const r1 = await run(t1, S1, [['move 1', 'move 1'], ...Array(6).fill(['move 1', 'move 2'])], {
    onBoundary: (b) => ({ sp: b.p1.active[0].species.id, types: b.p1.active[0].types, wx: b.field.weather }),
  });
  r1.perDecision.forEach((d, i) =>
    console.log(`F1 t${i + 1}: formechange=${JSON.stringify(grab(d, 'formechange'))} st=${JSON.stringify(r1.states[i])}`));

  // F2 Cloud Nine composition: Golduck (Cloud Nine) on the field, rain up -> does Castform transform?
  const t2 = [
    [mon('Castform', ['splash'], { ability: 'Forecast' })],
    [mon('Golduck', ['raindance', 'splash'], { ability: 'Cloud Nine' })],
  ];
  const r2 = await run(t2, S1, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ sp: b.p1.active[0].species.id, types: b.p1.active[0].types, wx: b.field.weather }),
  });
  r2.perDecision.forEach((d, i) =>
    console.log(`F2 t${i + 1}: formechange=${JSON.stringify(grab(d, 'formechange'))} st=${JSON.stringify(r2.states[i])}`));
}

async function secG() {
  console.log('\n===== G. KING\'S ROCK =====');
  // G1 ORDER: Crunch (20% SpD-drop secondary, contact) + KR + Static — diff vs no-KR run
  for (const seed of [S1, S2, [9, 9, 9, 9]]) {
    for (const [label, item] of [['KR', 'kingsrock'], ['noItem', '']]) {
      const t = [
        [mon('Ursaring', ['crunch'], { ability: 'Guts', item, evs: { spe: 252 } })],
        [mon('Electabuzz', ['splash'], { ability: 'Static', evs: { hp: 252 } })],
      ];
      const r = await run(t, seed, Array(3).fill(['move 1', 'move 1']));
      const out = r.perDecision.map((d, i) => {
        const seq = d.calls.filter((c) => (c.kind === 'random' && c.args[0] === 100) || c.kind === 'randomChance')
          .map((c) => (c.kind === 'random' ? `r100=${c.ret}` : `rc(${c.args.join(',')})=${c.ret}`)).join('>');
        return `t${i + 1}[${seq}|unboost=${grab(d, 'unboost').length}|status=${grab(d, '-status').length}]`;
      }).join(' ');
      console.log(`G1 ${label} seed=${JSON.stringify(seed)}: ${out}`);
    }
  }
  // G2 behind a sub: KR roll drawn but flinch NOT applied (defender still acts)
  const t2 = [
    [mon('Ursaring', ['tackle'], { ability: 'Guts', item: 'kingsrock', evs: { spe: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { ability: 'Thick Fat', evs: { hp: 252 } })],
  ];
  const r2 = await run(t2, S2, [['move 1', 'move 1'], ...Array(5).fill(['move 1', 'move 2'])]);
  r2.perDecision.forEach((d, i) => {
    const rolls = d.calls.filter((c) => c.kind === 'random' && c.args[0] === 100).map((c) => c.ret);
    console.log(`G2 t${i + 1}: r100=${JSON.stringify(rolls)} cant=${JSON.stringify(cants(d))} subhit=${grab(d, '-activate').length}`);
  });
  // G3 Seismic Toss procs; G4 Serene Grace (Dunsparce) threshold via roll-vs-cant table
  const t3 = [
    [mon('Ursaring', ['seismictoss'], { ability: 'Guts', item: 'kingsrock', evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat', evs: { hp: 252 } })],
  ];
  const r3 = await run(t3, S1, Array(3).fill(['move 1', 'move 1']));
  r3.perDecision.forEach((d, i) => {
    const rolls = d.calls.filter((c) => c.kind === 'random' && c.args[0] === 100).map((c) => c.ret);
    console.log(`G3 seismic t${i + 1}: r100=${JSON.stringify(rolls)} cant=${cants(d).length}`);
  });
  for (const seed of [S1, S2, [17, 5, 5, 5], [77, 1, 2, 3]]) {
    const t4 = [
      [mon('Dunsparce', ['tackle'], { ability: 'Serene Grace', item: 'kingsrock', evs: { spe: 252 } })],
      [mon('Snorlax', ['splash'], { ability: 'Thick Fat', evs: { hp: 252 } })],
    ];
    const r4 = await run(t4, seed, Array(4).fill(['move 1', 'move 1']));
    const out = r4.perDecision.map((d, i) => {
      const rolls = d.calls.filter((c) => c.kind === 'random' && c.args[0] === 100).map((c) => c.ret);
      return `t${i + 1}[roll=${rolls.join(',')} cant=${cants(d).length}]`;
    }).join(' ');
    console.log(`G4 SG seed=${JSON.stringify(seed)}: ${out}`);
  }
}

async function secH() {
  console.log('\n===== H. FOCUS BAND =====');
  // H1 per-hit draws: weak repeated hits into the holder each draw rc(1,10)
  const t1 = [
    [mon('Rattata', ['splash'], { ability: 'Guts', item: 'focusband', evs: { hp: 252 } })],
    [mon('Pidgey', ['tackle'], { ability: 'Keen Eye' })],
  ];
  const r1 = await run(t1, S1, Array(3).fill(['move 1', 'move 1']));
  r1.perDecision.forEach((d, i) => {
    const rc = d.calls.filter((c) => c.kind === 'randomChance').map((c) => `rc(${c.args.join(',')})=${c.ret}`);
    console.log(`H1 t${i + 1}: ${rc.join(' ')}`);
  });
  // H2 burn residual: draws but still faints when the chip is lethal
  const t2 = [
    [mon('Rattata', ['splash'], { ability: 'Guts', item: 'focusband', evs: {} , level: 5 })],
    [mon('Dusclops', ['willowisp', 'splash'], { ability: 'Pressure' })],
  ];
  const r2 = await run(t2, S1, Array(9).fill(['move 1', 'move 2']).map((c, i) => (i === 0 ? ['move 1', 'move 1'] : c)));
  r2.perDecision.forEach((d, i) => {
    const rc = d.calls.filter((c) => c.kind === 'randomChance').map((c) => `rc(${c.args.join(',')})=${c.ret}`);
    console.log(`H2 t${i + 1}: rc=[${rc.join(' ')}] faint=${grab(d, 'faint').length} fb=${grab(d, 'Focus Band').length}`);
    if (grab(d, 'faint').length) return;
  });
  // H3 Explosion self-KO draws nothing for the exploder
  const t3 = [
    [mon('Golem', ['explosion'], { ability: 'Sturdy', item: 'focusband' })],
    [mon('Registeel', ['splash'], { ability: 'Clear Body', evs: { hp: 252 } })],
  ];
  const r3 = await run(t3, S1, [['move 1', 'move 1']]);
  r3.perDecision.forEach((d, i) => {
    const rc = d.calls.filter((c) => c.kind === 'randomChance').map((c) => `rc(${c.args.join(',')})=${c.ret}@${c.site}`);
    console.log(`H3 t${i + 1}: rc=[${rc.join(' ')}] faints=${JSON.stringify(grab(d, 'faint'))}`);
  });
  // H4 lethal MOVE survive at 1 HP: scan seeds for a band proc on a lethal hit
  for (let s = 0; s < 24; s++) {
    const t4 = [
      [mon('Rattata', ['splash'], { ability: 'Guts', item: 'focusband', level: 5 })],
      [mon('Machamp', ['crosschop'], { ability: 'Guts', evs: { atk: 252 } })],
    ];
    const r4 = await run(t4, [s + 1, 2 * s + 3, 5, 7], [['move 1', 'move 1']]);
    const d = r4.perDecision[0];
    if (!d) continue;
    const act = grab(d, 'Focus Band');
    if (act.length) {
      console.log(`H4 seed[${s + 1},${2 * s + 3},5,7]: ACTIVATED ${JSON.stringify(act)} hp=${r4.battle.p1.pokemon[0].hp}`);
    }
  }
  console.log('H4 scan done');
}

(async () => {
  await secA(); await secB(); await secC(); await secD();
  await secE(); await secF(); await secG(); await secH();
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
