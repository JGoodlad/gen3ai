// probe_batch6_field_trap.js — ground-truth GROUP B of move-coverage BATCH 6 bit-for-bit vs the
// OMNISCIENT in-process gen3 BattleStream (no server): PERISH SONG / MEAN LOOK / SPIDER WEB / BLOCK.
//
// The resolved gen3 sources (dumped below) say:
//   perishsong: acc TRUE (never-miss), Status, target 'all', flags {sound, distance}; onHitField
//     loops getAllActive(): runEvent('Invulnerability') -> runEvent('TryHit') (Soundproof blocks
//     here) -> addVolatile('perishsong') + '-start <mon> perish3 [silent]'; then ONE
//     '-fieldactivate|move: Perish Song' iff any applied. condition: duration 4,
//     onResidualOrder 12 (LAST, after futuremove 11), onResidual prints '-start perish<duration>',
//     onEnd prints perish0 + faint().
//   meanlook / spiderweb / block: acc TRUE (never-miss), Status, target normal, flags
//     {protect, reflectable, mirror} (NO bypasssub!); onHit -> target.addVolatile('trapped',
//     source, move, 'trapper') — a LINKED volatile (the source gets 'trapper'; removing either
//     end unlinks). trapped: noCopy FALSE in gen3 (Baton Pass copies it?!), no duration,
//     onTrapPokemon -> pokemon.tryTrap() (NO 'hidden' -> the FIRM trapped:true request shape?),
//     onStart '-activate <target> trapped'.
//
// This probe settles BEHAVIORALLY (the sim is the only oracle):
//   PERISH SONG — the cast draw model (never-miss? any draws?); BOTH actives get the counter;
//     the residual tick order vs Leftovers/brn (order 12 = last?); the counter display timeline
//     3->2->1->0-faint; Soundproof immunity (foe blocked, caster still counted?); switch-out
//     CLEARS the counter (a volatile); an entrant during the song gets NO counter; a re-cast
//     while counters run (all-fail path); the residual tie draws for a perished mirror.
//   TRAP MOVES — the cast draw model; the request shape (trapped:true firm vs maybeTrapped) +
//     the rejected-switch error text ([Invalid] vs [Unavailable]+re-request); a grounded GHOST
//     is trapped?; the volatile ends when the TRAPPER switches out / faints (the linked
//     'trapper')?; Baton Pass by the trapped mon (escape? volatile copied — noCopy false?);
//     phaze still drags a move-trapped mon; a SUBSTITUTE blocks the trap move (no bypasssub);
//     re-application fails (draws?); spiderweb/block identical machinery.
//
// Run:  node src/rust_sim/harness/probe_batch6_field_trap.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  for (const id of ['perishsong', 'meanlook', 'spiderweb', 'block']) {
    const m = d.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} type=${m.type} prio=${m.priority} pp=${m.pp} target=${m.target}`);
    console.log(`  flags=${JSON.stringify(m.flags)} volatileStatus=${JSON.stringify(m.volatileStatus)} ignoreImmunity=${JSON.stringify(m.ignoreImmunity)}`);
    for (const k of Object.keys(m)) {
      const v = m[k];
      if (typeof v === 'function') console.log(`  fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
    }
    if (m.condition) {
      console.log('  --- move.condition:');
      for (const k of Object.keys(m.condition)) {
        const v = m.condition[k];
        if (typeof v === 'function') console.log(`      fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
        else console.log(`      ${k}=${JSON.stringify(v)}`);
      }
    }
    for (const g of ['gen4', 'gen9']) {
      const gm = Dex.mod(g).moves.get(id);
      console.log(`  [${g}] acc=${JSON.stringify(gm.accuracy)} flags=${JSON.stringify(gm.flags)}`);
    }
  }
  for (const cid of ['trapped', 'trapper', 'perishsong']) {
    const c = d.conditions.get(cid);
    if (!c || !c.exists) { console.log(`=== condition ${cid}: MISSING`); continue; }
    console.log(`=== resolved gen3 condition ${cid} ===`);
    for (const k of Object.keys(c)) {
      const v = c[k];
      if (typeof v === 'function') console.log(`      fn ${k}: ${v.toString().replace(/\s+/g, ' ')}`);
      else if (v !== undefined) console.log(`      ${k}=${JSON.stringify(v)}`);
    }
    for (const g of ['gen4', 'gen9']) {
      const gc = Dex.mod(g).conditions.get(cid);
      if (gc && gc.exists) console.log(`  [${g}] noCopy=${JSON.stringify(gc.noCopy)} duration=${JSON.stringify(gc.duration)}`);
    }
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 5; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

function reqSummary(req) {
  if (!req) return 'none';
  if (req.wait) return 'wait';
  if (req.forceSwitch) return `forceSwitch=${JSON.stringify(req.forceSwitch)}`;
  const a = req.active && req.active[0];
  if (!a) return 'move?';
  const moves = (a.moves || []).map((mv) => mv.id + (mv.disabled ? '!' : '')).join('/');
  return `move(trapped=${a.trapped === undefined ? '-' : a.trapped},maybe=${a.maybeTrapped === undefined ? '-' : a.maybeTrapped}) [${moves}]`;
}

// plan entries: { p1, p2, pre: [injections], reject: {side, cmd}, stop }
async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  const sideLog = [];
  const lastReq = [null, null];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  for (const [idx, s] of [streams.p1, streams.p2].entries()) {
    (async () => {
      for await (const ch of s) {
        for (const l of ch.split('\n')) {
          if (!l) continue;
          sideLog.push(`p${idx + 1}| ${l}`);
          if (l.startsWith('|request|')) {
            const body = l.slice(9);
            try { lastReq[idx] = body ? JSON.parse(body) : null; } catch (e) { lastReq[idx] = null; }
          }
        }
      }
    })();
  }
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.faint) { m.hp = 0; m.fainted = true; }
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
    }
  };
  applyActs(inject && inject.acts);

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  const volStr = (m) => Object.entries(m.volatiles).map(([k, v]) =>
    k + (v.duration != null ? `(${v.duration})` : '') +
    ((k === 'trapped' || k === 'trapper') && v.source ? `<src:${v.source.name}${v.source.isActive ? '' : '!out'}>` : '')
  ).join(',');
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} simTrapped=${JSON.stringify(m.trapped)} vols=[${volStr(m)}]` : '-';
  const teamStr = (side) => side.pokemon.map((m) => `${m.species.name}:${m.hp}${m.fainted ? 'fnt' : ''}[${Object.keys(m.volatiles).join('+')}]`).join(' ');

  let i = 0, safety = 0;
  while (!battle.ended && safety < 26) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    if (entry.reject) {
      console.log(`  REJECT-TRY >${entry.reject.side} ${entry.reject.cmd}   pre-req p1=${reqSummary(lastReq[0])} p2=${reqSummary(lastReq[1])}`);
      const sl0 = sideLog.length;
      streams.omniscient.write(`>${entry.reject.side} ${entry.reject.cmd}`);
      for (let k = 0; k < 12; k++) await tick();
      for (const l of sideLog.slice(sl0)) if (l.includes('|error|')) console.log(`        ERR ${l}`);
      console.log(`        post-reject req p1=${reqSummary(lastReq[0])} p2=${reqSummary(lastReq[1])}  rejectDraws=${draws.length}`);
    }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        post: p1=${fmt(a0)}`);
    console.log(`              p2=${fmt(a1)}`);
    console.log(`        team1: ${teamStr(battle.sides[0])}`);
    console.log(`        team2: ${teamStr(battle.sides[1])}`);
    console.log(`        req  : p1=${reqSummary(lastReq[0])} p2=${reqSummary(lastReq[1])}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-boost|-unboost|-fail|-immune|-miss|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint|-prepare|-singleturn|-fieldactivate|-nothing|win\|/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // ------------------------------------------------------------- PERISH SONG
  // P1: baseline — distinct speeds, Leftovers + brn on the board so the residual ORDER of the
  // perish tick (order 12? LAST?) shows against Leftovers (10.4) and brn (10.6). Lapras is
  // injured so Leftovers heals visibly. Runs to the double perish faint -> double replacement.
  await run('P1 perish baseline: cast draws / both sides / residual order vs lefties+brn / faint at 0',
    [mon('Lapras', ['perishsong', 'splash'], { item: 'Leftovers', evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } }), mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // cast: perish3 both; tick 3 at residual
     { p1: 'move 2', p2: 'move 1' },              // tick 2
     { p1: 'move 2', p2: 'move 1' },              // tick 1
     { p1: 'move 2', p2: 'move 1' },              // tick 0 -> BOTH faint -> double replacement
     { p1: 'switch 2', p2: 'switch 2', stop: true }],
    { acts: [{ side: 0, hp: 300 }, { side: 1, status: 'brn' }] });

  // P2: SOUNDPROOF foe — the foe is immune (TryHit null in the onHitField loop) but the CASTER
  // still gets its own counter? Draw count vs P1's cast turn.
  await run('P2 perish vs a Soundproof foe (caster still counted?)',
    [mon('Lapras', ['perishsong', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Mr. Mime', ['splash'], { ability: 'Soundproof', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // cast: Mime immune; Lapras perish3?
     { p1: 'move 1', p2: 'move 1' },              // RE-cast: caster counted + foe immune -> fail? silent success?
     { p1: 'move 2', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1' },              // Lapras faints alone -> p1 forced switch
     { p1: 'switch 2', stop: true }]);

  // P3: SWITCH-OUT clears the counter + an ENTRANT during the song gets NO counter + the
  // returning mon has NO counter.
  await run('P3 perish: switch-out clears; entrant/returner uncounted',
    [mon('Lapras', ['perishsong', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } }), mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // cast
     { p1: 'move 2', p2: 'switch 2' },            // Blissey (perish2) OUT -> cleared? Skarm entrant counter-free?
     { p1: 'move 2', p2: 'switch 2' },            // Blissey back: no counter?
     { p1: 'move 2', p2: 'move 1' },              // Lapras ticks to 0 alone -> faints
     { p1: 'switch 2', stop: true }]);

  // P4: RE-CAST while counters are active — every active already has the volatile -> all-fail
  // path (result false -> move fails?). Draws?
  await run('P4 perish re-cast while counters active (all-fail path)',
    [mon('Lapras', ['perishsong', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Blissey', ['splash'], { evs: { hp: 252 } }), mon('Skarmory', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // cast
     { p1: 'move 1', p2: 'move 1' },              // re-cast -> fail? draws?
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // P5: the perished MIRROR at an EQUAL speed — the residual tie draws (perishsong registers an
  // order-12 onResidual handler AND a duration handler per mon -> 2 tie groups?). t1 both-splash
  // control; t2 one-sided cast (equal-speed move-order tie is in both turns' baseline).
  await run('P5 perish mirror at an equal speed: residual tie draw counts',
    [mon('Lapras', ['perishsong', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Lapras', ['perishsong', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },              // control: both splash (tie baseline draws)
     { p1: 'move 1', p2: 'move 2' },              // cast (one side)
     { p1: 'move 2', p2: 'move 2' },              // tick 2: two perish handlers tie?
     { p1: 'move 2', p2: 'move 2' },              // tick 1
     { p1: 'move 2', p2: 'move 2' },              // tick 0 -> both faint
     { p1: 'switch 2', p2: 'switch 2', stop: true }]);

  // --------------------------------------------------------------- MEAN LOOK
  // T1: the LIFECYCLE — cast draws; the request shape (firm trapped:true?); the rejected-switch
  // error text; the TRAPPER SWITCHES OUT -> the linked 'trapper' volatile is removed -> the
  // target is freed (volatile gone?) and its switch is accepted.
  await run('T1 meanlook lifecycle: draws / request shape / reject / trapper leaves -> freed',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash', 'protect'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // meanlook lands
     { reject: { side: 'p2', cmd: 'switch 2' }, p1: 'move 2', p2: 'move 1' }, // rejected switch
     { p1: 'switch 2', p2: 'move 1' },            // TRAPPER leaves -> p2 freed?
     { p1: 'move 1', p2: 'switch 2', stop: true }]); // p2 switch now accepted?

  // T2: a grounded GHOST — trapped by Mean Look in gen3-Showdown? (the arenatrap precedent:
  // no 'trapped' type-immunity in the gen3 dex)
  await run('T2 meanlook vs a Ghost (Gengar): trapped?',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { reject: { side: 'p2', cmd: 'switch 2' }, p1: 'move 2', p2: 'move 1', stop: true }]);

  // T3: BATON PASS by the trapped mon — BP bypasses the switch gate; does the entrant INHERIT
  // the trapped volatile (gen3 trapped.noCopy=false — the copyVolatileFrom question)?
  await run('T3 meanlook then the trapped mon Baton Passes: escape? volatile passed?',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } })],
    [mon('Celebi', ['batonpass', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },              // meanlook lands; Celebi splashes
     { p1: 'move 2', p2: 'move 1' },              // Celebi Baton Passes (allowed while trapped?)
     { p2: 'switch 2' },                          // the BP replacement
     { reject: { side: 'p2', cmd: 'switch 2' }, p1: 'move 2', p2: 'move 1', stop: true }]); // entrant trapped?

  // T3b: after the Baton-Pass INHERITANCE (gen3 trapped.noCopy=false — T3), does the ORIGINAL
  // trapper's switch-out still FREE the BP-inheriting holder (is the linked-volatile link
  // re-pointed to the entrant, or stale on the benched passer)?
  await run('T3b BP-inherited trap: the trapper then leaves — is the entrant freed?',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Celebi', ['batonpass', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' },              // meanlook lands on Celebi
     { p1: 'move 2', p2: 'move 1' },              // Celebi Baton Passes
     { p2: 'switch 2' },                          // Blissey inherits 'trapped'
     { p1: 'switch 2', p2: 'move 1' },            // the ORIGINAL trapper (Umbreon) leaves
     { p1: 'move 1', p2: 'switch 2', stop: true }]); // Blissey freed?

  // T4: PHAZE — Roar still drags the move-trapped mon (trapping gates only the voluntary switch).
  await run('T4 meanlook then Roar: the phaze drags the trapped mon',
    [mon('Umbreon', ['meanlook', 'roar', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // meanlook lands
     { p1: 'move 2', p2: 'move 1', stop: true }]); // Roar -> drag despite the trap? sample draw?

  // T5: a SUBSTITUTE blocks Mean Look (no bypasssub) — draws? Then the subbed mon switches freely.
  await run('T5 meanlook into a substitute: blocked? then a free switch',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' },              // sub up
     { p1: 'move 1', p2: 'move 2' },              // meanlook into the sub -> blocked?
     { p1: 'move 2', p2: 'switch 2', stop: true }]); // free switch (not trapped)?

  // T6: RE-APPLICATION — a second Mean Look into an already-trapped foe fails (addVolatile
  // false). Draws?
  await run('T6 meanlook re-application fails',
    [mon('Umbreon', ['meanlook', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p1: 'move 1', p2: 'move 1' },              // re-apply -> fail? draws?
     { p1: 'move 2', p2: 'move 1', stop: true }]);

  // T7: SPIDER WEB — identical machinery? (acc true, same 'trapped' volatile)
  await run('T7 spiderweb: identical trap machinery',
    [mon('Ariados', ['spiderweb', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { reject: { side: 'p2', cmd: 'switch 2' }, p1: 'move 2', p2: 'move 1', stop: true }]);

  // T8: BLOCK — identical machinery?
  await run('T8 block: identical trap machinery',
    [mon('Regirock', ['block', 'splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { reject: { side: 'p2', cmd: 'switch 2' }, p1: 'move 2', p2: 'move 1', stop: true }]);

  // T9: the TRAPPER FAINTS — the linked volatile is removed on faint -> the target freed? The
  // 1-HP Misdreavus meanlooks (faster) then dies to Return; after the replacement the
  // ex-trapped Snorlax's switch should be accepted.
  await run('T9 the trapper faints: the trapped mon is freed',
    [mon('Misdreavus', ['meanlook', 'splash'], { evs: { hp: 252 } }), mon('Snorlax', ['splash'], { evs: { hp: 252 } })],
    [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } }), mon('Blissey', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },              // meanlook lands; Shadow Ball KOs the 1-HP Misdreavus
     { p1: 'switch 2' },                          // the p1 replacement
     { p1: 'move 1', p2: 'switch 2', stop: true }], // p2 switch accepted (trapper dead)?
    { acts: [{ side: 0, hp: 1 }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
