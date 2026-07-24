// probe_liechi_atcap.js — replay the ab_1_6 Liechi repro through the REAL sim and dump the
// protocol lines around the Scyther Liechi-Berry eat, to confirm the exact at-cap `-boost` form.
'use strict';
const fs = require('fs');
const path = require('path');
const { runBattle } = require('./gen_e2e_fuzz.js');

const dir = process.argv[2] || path.join(__dirname, 'ab_fuzz_out/divergences/rmry3ytkn_ab_1_6');
const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
const seed = summary.battle_seed; // the ORIGINAL >start seed → replayChoices reproduces the battle
const choices = summary.choices;

(async () => {
  const rec = await runBattle(
    summary.packed_teams.p1,
    summary.packed_teams.p2,
    seed,
    0,
    'pool',
    { replayChoices: choices, format: 'gen3customgame' },
  );
  const lines = rec.lines || [];
  // Print every line, marking those near a Liechi eat / a boost / an enditem.
  let n = 0;
  for (const l of lines) {
    if (/-enditem|Liechi|-boost|-unboost|-heal|Leftovers|Scyther/.test(l) && !/^\|t:/.test(l)) {
      console.log(String(n).padStart(4), l);
    }
    n++;
  }
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
