// probe_batch4_lib.js — shared runner for the batch-4 probes (Truant / Inner Focus /
// Shadow Tag / Cute Charm+attract / Color Change / Forecast / King's Rock / Focus Band).
//
// Mandate 1: the resolved `Dex.mod('gen3')` sim (in-process omniscient BattleStream) is
// the ONLY oracle. This lib instruments the PRNG (random / randomChance / sample /
// shuffle + the raw rng.next count) recording each call's args + a coarse call-site
// label + return value, captures the omniscient protocol lines per decision, and gives
// each probe direct access to the live `battle` object for state reads (pp, trapped,
// species/types, volatiles). Used by harness/probe_{truant,innerfocus,shadowtag,
// cutecharm_attract,colorchange,forecast,kingsrock,focusband}_rng.js.

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || '',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Coarse call-site label: first non-prng battle-* frame of the stack.
function siteLabel(stack) {
  const lines = String(stack).split('\n').slice(2);
  for (const l of lines) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle'].includes(m[1])) return m[1];
  }
  return '?';
}

// Run a scripted battle. teams = [p1sets, p2sets]; choices = [[c1, c2], ...] per
// decision boundary (null = that side has no pending request). onBoundary(battle, i)
// is called after each boundary settles — return a value to collect per decision.
async function run(teams, seed, choices, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${opts.fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const prng = battle.prng;
  const rng = prng.rng;
  const realNext = rng.next.bind(rng);
  let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };
  const calls = [];
  for (const name of ['random', 'randomChance', 'sample', 'shuffle']) {
    const orig = prng[name].bind(prng);
    prng[name] = (...a) => {
      const r = orig(...a);
      calls.push({ kind: name, args: name === 'shuffle' ? a.slice(1) : a, site: siteLabel(new Error().stack), ret: name === 'shuffle' ? undefined : r });
      return r;
    };
  }
  const perDecision = [];
  const states = [];
  let lineLo = lines.length;
  for (let i = 0; i < choices.length; i++) {
    const [c1, c2] = choices[i];
    const callLo = calls.length; const nextLo = nextCount;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 12; k++) await tick();
    perDecision.push({ calls: calls.slice(callLo), nexts: nextCount - nextLo, lines: lines.slice(lineLo) });
    lineLo = lines.length;
    if (opts.onBoundary) states.push(opts.onBoundary(battle, i));
    if (battle.ended) break;
  }
  return { battle, calls, nextCount, perDecision, states, lines };
}

function fmtCalls(cs) {
  return cs.map((c) => `${c.kind}(${c.args.map((a) => JSON.stringify(a)).join(',')})@${c.site}${c.ret !== undefined ? '=' + JSON.stringify(c.ret) : ''}`).join(' ');
}

module.exports = { mon, run, fmtCalls, tick, Dex, IV31, EV0 };
