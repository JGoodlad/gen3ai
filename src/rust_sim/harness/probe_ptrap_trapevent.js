// probe_ptrap_trapevent.js — the endTurn `runEvent('TrapPokemon'/'MaybeTrapPokemon')`
// HANDLER-SORT question the ROUND-32 byte fuzz surfaced (repro rmsde6xp4_ab_10_9).
//
// The port's `trap_event_shuffles` counts only the ABILITY handlers (Arena Trap's `onFoe*`,
// Magnet Pull's `onAny*`), documenting that the trap-MOVE `trapped` volatile "adds ZERO endTurn
// draws — its Condition handler subOrder never ties an Ability handler's". But `partiallytrapped`
// ALSO carries `onTrapPokemon`, and it is ALSO a Condition — so a mon holding BOTH (Block then
// Sand Tomb, exactly the repro board) may put TWO handlers with the SAME sort key on the same mon
// and draw a Fisher-Yates shuffle per event.
//
// This measures the endTurn draw count for each combination against a clean control.
//
// Run: node harness/probe_ptrap_trapevent.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [5, 4, 3, 2];

function endTurnDraws(dec) {
  return dec.calls.filter((c) => c.kind === 'random' || c.kind === 'randomChance' || c.kind === 'shuffle').length;
}

async function trial(label, teams, choices, checkTurn) {
  const r = await run(teams, SEED, choices, {
    onBoundary: (b) => {
      const t = b.sides[0].active[0];
      return {
        p1: `${t.species.id} trapped=${t.trapped} vols=${Object.keys(t.volatiles).join('+')}`,
      };
    },
  });
  const d = r.perDecision[checkTurn];
  console.log(`  ${label}: turn${checkTurn + 1} draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
  console.log(`        state=${JSON.stringify(r.states[checkTurn])}`);
  return d.nexts;
}

async function main() {
  // p1 Wartortle is the VICTIM; p2 Onix casts Block (`trapped`) and/or Sand Tomb
  // (`partiallytrapped`). Both are draw-bearing casts, so the CHECK turn is a later
  // all-Splash turn where the only draws are the baseline endTurn roll + any handler shuffle.
  const victim = (moves) => mon('Wartortle', moves, { ability: 'Torrent' });
  const onix = (moves) => mon('Onix', moves, { ability: 'Sturdy' });

  console.log('############ CONTROL: no trap at all ############');
  const ctrl = await trial('control', [[victim(['splash'])], [onix(['splash', 'block', 'sandtomb'])]],
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ BLOCK only (the `trapped` condition) ############');
  const blk = await trial('block-only', [[victim(['splash'])], [onix(['splash', 'block', 'sandtomb'])]],
    [['move 1', 'move 2'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ SAND TOMB only (the `partiallytrapped` condition) ############');
  const st = await trial('sandtomb-only', [[victim(['splash'])], [onix(['splash', 'block', 'sandtomb'])]],
    [['move 1', 'move 3'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ BOTH (Block then Sand Tomb) — the repro board ############');
  const both = await trial('both', [[victim(['splash'])], [onix(['splash', 'block', 'sandtomb'])]],
    [['move 1', 'move 2'], ['move 1', 'move 3'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ BOTH, reversed cast order (Sand Tomb then Block) ############');
  const both2 = await trial('both-rev', [[victim(['splash'])], [onix(['splash', 'block', 'sandtomb'])]],
    [['move 1', 'move 3'], ['move 1', 'move 2'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ BOTH + an ARENA TRAP foe (3 handlers) ############');
  const dug = mon('Dugtrio', ['splash', 'block', 'sandtomb'], { ability: 'Arena Trap' });
  const both3 = await trial('both+arenatrap', [[victim(['splash'])], [dug]],
    [['move 1', 'move 2'], ['move 1', 'move 3'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n############ ARENA TRAP alone (1 handler — the documented 0-draw case) ############');
  const at = await trial('arenatrap-only', [[victim(['splash'])], [dug]],
    [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 3);

  console.log('\n==== SUMMARY (extra draws on an idle turn, vs the control) ====');
  for (const [k, v] of [['block-only', blk], ['sandtomb-only', st], ['both', both],
    ['both-rev', both2], ['both+arenatrap', both3], ['arenatrap-only', at]]) {
    console.log(`  ${k}: ${v - ctrl}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
