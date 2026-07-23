// probe_batch89_abilities_items.js — LIVE probes for Batch-9 WONDER GUARD / LIQUID OOZE /
// WHITE HERB / STICK vs the omniscient gen3 sim.
// Run: node src/rust_sim/harness/probe_batch89_abilities_items.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function showDec(tag, r, extra) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && l !== '|' && !l.startsWith('|debug'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}

async function main() {
  // ================= WONDER GUARD (Shedinja: Bug/Ghost, 1 HP) =================
  console.log('############ WONDER GUARD ############');
  {
    // Shedinja p2. p1 throws: a NEUTRAL move (Shadow Ball? ghost SE vs ghost),
    // a NOT-very-effective, a resisted, a SUPER-EFFECTIVE (Fire/Flying/Rock/Ghost/Dark hit Bug/Ghost SE),
    // a STATUS move (Thunder Wave), Struggle (typeless), and weather chip.
    const teams = [
      [mon('Charizard', ['ember', 'tackle', 'thunderwave', 'sandstorm', 'aerialace'], { ability: 'Blaze' })],
      [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // t1: Tackle (Normal, neutral vs Bug/Ghost? Normal->Ghost=immune anyway)
      ['move 1', 'move 1'],  // t2: Ember (Fire SE vs Bug) -> HITS through WG
      ['move 5', 'move 1'],  // t3: Aerial Ace (Flying SE vs Bug) -> HITS
      ['move 3', 'move 1'],  // t4: Thunder Wave (Status) -> paralyzes? (WG doesn't block status)
      ['move 4', 'move 1'],  // t5: Sandstorm -> does sand chip hurt WG? (Shedinja not Rock/Ground/Steel)
    ], { onBoundary: (b) => ({ shedHp: `${b.sides[1].active[0].hp}/${b.sides[1].active[0].maxhp}`, shedStatus: b.sides[1].active[0].status }) });
    showDec('WG', r);
  }
  {
    // Struggle vs Wonder Guard (typeless '???' -> WG returns, NOT blocked).
    const teams = [
      [mon('Rattata', ['splash'], { ability: 'Guts' })],   // will be forced to Struggle after PP drain? Instead:
      [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' })],
    ];
    // Force Struggle: give p1 a 1-PP move drained. Simpler: use a mon whose only move is gone.
    // Use a single-PP-ish approach: Metronome not modeled. Instead test a WEATHER-immune check via Leech Seed.
    // (Struggle setup is heavy; note the source rule instead — see doc.)
  }
  {
    // Leech Seed onto Wonder Guard: does the seed drain (residual, non-move) bypass WG?
    const teams = [
      [mon('Venusaur', ['leechseed', 'splash'], { ability: 'Overgrow' })],
      [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Leech Seed onto Shedinja (Grass move, status -> WG doesn't block status)
      ['move 2', 'move 1'],  // t2: residual leech drain kills Shedinja (1 HP)?
    ], { onBoundary: (b) => ({ shedHp: b.sides[1].active[0] ? `${b.sides[1].active[0].hp}/${b.sides[1].active[0].maxhp}` : 'GONE', vol: b.sides[1].active[0] ? Object.keys(b.sides[1].active[0].volatiles) : [] }) });
    showDec('WG-leechseed', r);
  }

  // ================= LIQUID OOZE =================
  console.log('\n############ LIQUID OOZE ############');
  {
    // Giga Drain into a Liquid Ooze mon -> attacker TAKES damage instead of healing.
    const teams = [
      [mon('Venusaur', ['gigadrain', 'splash'], { ability: 'Overgrow' })],
      [mon('Tentacruel', ['splash'], { ability: 'Liquid Ooze' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Giga Drain -> Venusaur takes ooze damage (-damage [from] ability: Liquid Ooze)
    ], { onBoundary: (b) => ({ p1hp: `${b.sides[0].active[0].hp}/${b.sides[0].active[0].maxhp}`, p2hp: `${b.sides[1].active[0].hp}/${b.sides[1].active[0].maxhp}` }) });
    showDec('LO-gigadrain', r);
  }
  {
    // Leech Seed onto a Liquid Ooze mon -> the SEEDER takes damage at residual.
    const teams = [
      [mon('Venusaur', ['leechseed', 'splash'], { ability: 'Overgrow' })],
      [mon('Tentacruel', ['splash'], { ability: 'Liquid Ooze' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Leech Seed onto Liquid Ooze mon
      ['move 2', 'move 1'],  // t2: residual -> Venusaur (seeder) takes damage instead of healing
    ], { onBoundary: (b) => ({ p1hp: `${b.sides[0].active[0].hp}/${b.sides[0].active[0].maxhp}`, p2hp: `${b.sides[1].active[0].hp}/${b.sides[1].active[0].maxhp}` }) });
    showDec('LO-leechseed', r);
  }

  // ================= WHITE HERB =================
  console.log('\n############ WHITE HERB ############');
  {
    // A stat-drop move (Growl -atk / Leer -def) into a White Herb holder -> restores immediately.
    const teams = [
      [mon('Persian', ['growl', 'fakeout', 'swordsdance', 'splash'], { ability: 'Limber' })],
      [mon('Snorlax', ['bellydrum', 'splash'], { item: 'White Herb', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 2'],  // t1: p1 Growl (-1 atk on Snorlax) -> White Herb restores? p2 Splash
      ['move 4', 'move 2'],  // t2: confirm boost table
    ], { onBoundary: (b) => ({ p2boosts: b.sides[1].active[0].boosts, p2item: b.sides[1].active[0].item }) });
    showDec('WH-growl', r);
  }
  {
    // Overheat-style self-drop: does White Herb restore the USER's own -2 spa?  Use Superpower (self -atk/-def).
    const teams = [
      [mon('Machamp', ['superpower', 'splash'], { item: 'White Herb', ability: 'Guts' })],
      [mon('Snorlax', ['splash'], { ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Superpower -> self -1 atk/-1 def -> White Herb restores same turn?
    ], { onBoundary: (b) => ({ p1boosts: b.sides[0].active[0].boosts, p1item: b.sides[0].active[0].item }) });
    showDec('WH-superpower', r);
  }
  {
    // Mixed: +2 then -1 -> does White Herb only clear the NEGATIVE, keeping positives?
    const teams = [
      [mon('Snorlax', ['swordsdance', 'splash'], { item: 'White Herb', ability: 'Own Tempo' })],
      [mon('Persian', ['growl', 'splash'], { ability: 'Limber' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 2'],  // t1: p1 SD (+2 atk), p2 Splash
      ['move 2', 'move 1'],  // t2: p2 Growl (-1 atk) -> net +1; White Herb triggers on the -1? clears to... ?
    ], { onBoundary: (b) => ({ p1boosts: b.sides[0].active[0].boosts, p1item: b.sides[0].active[0].item }) });
    showDec('WH-mixed', r);
  }

  // ================= STICK (Farfetch'd crit) =================
  console.log('\n############ STICK crit-rate ############');
  {
    // Farfetch'd with Stick: critRatio+2. Count crits over many seeds vs a no-item control.
    const mk = (item) => [
      [mon("Farfetch'd", ['cut', 'splash'], { item, ability: 'Keen Eye' })],
      [mon('Snorlax', ['splash'], { ability: 'Own Tempo' })],
    ];
    for (const item of ['Stick', '']) {
      let crits = 0, hits = 0;
      for (let s = 0; s < 60; s++) {
        const seed = [s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3];
        const r = await run(mk(item), seed, [['move 1', 'move 1']]);
        const l = r.perDecision[0].lines;
        if (l.some((x) => x.includes('|-damage|p2a'))) hits++;
        if (l.some((x) => x.includes('|-crit|p2a'))) crits++;
      }
      console.log(`  ${item || '(no item)'}: crits=${crits}/${hits} hits`);
    }
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
