// probe_statusimmune_enumerate.js — ENUMERATE the STATUS_IMMUNE ability class from the
// RESOLVED `Dex.mod('gen3')` dist (the ONLY oracle — NOT a base-source read). Dumps every
// gen3 ability carrying a status-immunity / status-blocking handler + its exact handler
// shape, so we can see (per member) WHICH handler fires (onSetStatus vs onTrySetStatus vs
// onImmunity) — the draw-model crux (does it participate in the SetStatus event or block
// before it?).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));
const d3 = Dex.mod('gen3');

// Every handler that could grant / deny a status. onSetStatus = participates IN the
// SetStatus event (the clause shuffle). onTrySetStatus = fires in trySetStatus BEFORE the
// SetStatus event. onImmunity = fires in runStatusImmunity (also before the event).
const STATUS_HANDLERS = [
  'onSetStatus', 'onTrySetStatus', 'onImmunity', 'onUpdate',
  'onTryAddVolatile', 'onStart', 'onSwitchIn',
];

const gen3AbilityNums = new Set();
// gather all gen3-legal ability ids (num <= 76)
const abis = [];
for (const ab of d3.abilities.all()) {
  if (ab.num >= 1 && ab.num <= 76) abis.push(ab);
}
abis.sort((a, b) => a.num - b.num);

console.log('=== ALL gen3 abilities with a status-relevant handler (resolved dist) ===');
for (const ab of abis) {
  const present = STATUS_HANDLERS.filter((h) => ab[h] !== undefined);
  // Also detect a body that mentions status/frz/slp/par/brn/psn/tox (heuristic net).
  const bodies = present.map((h) => `${h}=${String(ab[h]).replace(/\s+/g, ' ')}`);
  // filter to ones whose source actually mentions a status keyword OR is onSetStatus/onImmunity
  const mentions = bodies.filter((s) =>
    /onSetStatus|onImmunity|onTrySetStatus/.test(s) ||
    /status|['"](frz|slp|par|brn|psn|tox)['"]|Insomnia|Limber|Immunity|Veil|Magma|Tempo|Oblivious|Vital|Leaf/.test(s)
  );
  if (present.length === 0) continue;
  if (mentions.length === 0) continue;
  console.log(`\n[${ab.num}] ${ab.id} (${ab.name})`);
  for (const b of bodies) console.log('   ', b);
}
