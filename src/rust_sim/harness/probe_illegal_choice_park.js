// probe_illegal_choice_park.js — DETERMINISTIC probe for the "soak park" class.
//
// QUESTION: when a bridge child stops emitting mid-battle, is that a DEADLOCK in the bridge,
// or the sim CORRECTLY refusing an illegal choice and waiting for a legal one?
//
// Showdown's `Side.emitChoiceError` (sim/side.ts:510) picks the error TYPE from whether the
// refusal actually CHANGED the request:
//    updated  -> `[Unavailable choice]` + `emitRequest(..., true)`   (a RE-REQUEST: recoverable)
//   !updated  -> `[Invalid choice]`     + NOTHING                    (silent: the client must
//                                                                     re-pick from the request
//                                                                     it already has)
// So an out-of-range / already-declared-illegal choice legitimately produces NO new `|request|`
// frame. Any driver that blocks until a `|request|` arrives parks forever — which is what the
// 1200-battle soak did. That is a HARNESS contract bug, not a bridge deadlock.
//
// This probe sends ONE deliberately out-of-range move choice to BOTH bridges and prints what
// each emitted within the window, so the claim is evidence, not a source read.
//
// USAGE: node src/rust_sim/harness/probe_illegal_choice_park.js [--wait-ms 4000]
'use strict';

const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '../../..');
const NODE_BRIDGE = path.join(ROOT, 'src/utils/bridge/local_sim_bridge.js');
const RUST_BRIDGE = path.join(
  process.env.POKESIM_SIMBRIDGE_TARGET || '/tmp/pokesim_target_simbridge', 'release/sim_bridge');
const { Teams } = require(path.join(ROOT, 'deps/pokemon-showdown/dist/sim'));

const waitMs = (() => {
  const i = process.argv.indexOf('--wait-ms');
  return i === -1 ? 4000 : Number(process.argv[i + 1]);
})();

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const set = (species, moves, ability) => ({
  name: species, species, item: 'Leftovers', ability: ability || 'No Ability', moves,
  evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: 'M',
});

class Child {
  constructor(cmd, args) {
    this.name = cmd;
    this.lines = [];
    this.proc = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    this.buf = '';
    this.proc.stdout.on('data', (d) => {
      this.buf += d.toString();
      let nl;
      while ((nl = this.buf.indexOf('\n')) !== -1) {
        const l = this.buf.slice(0, nl); this.buf = this.buf.slice(nl + 1);
        if (l.startsWith('p1 ') || l.startsWith('p2 ')) {
          this.lines.push([l.slice(0, 2), Buffer.from(l.slice(3), 'base64').toString('utf8')]);
        } else this.lines.push(['--', l]);
      }
    });
    this.proc.stderr.on('data', () => {});
    this.proc.on('error', () => {});
    this.proc.stdin.on('error', () => {});
  }
  write(s) { try { this.proc.stdin.write(s + '\n'); } catch (e) {} }
  kill() { try { this.proc.kill('SIGKILL'); } catch (e) {} }
  since(n) { return this.lines.slice(n); }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const summarize = (evs) => {
  const req = [], err = [], other = [];
  for (const [side, chunk] of evs) {
    for (const l of String(chunk).split('\n')) {
      if (l.startsWith('|request|')) req.push(`${side} |request|`);
      else if (l.startsWith('|error|')) err.push(`${side} ${l}`);
      else if (l.startsWith('__')) other.push(l);
    }
    if (side === '--') other.push(chunk);
  }
  return { requests: req.length, errors: err, other: other.filter(Boolean) };
};

async function main() {
  const p1 = Teams.pack([set('Dugtrio', ['earthquake', 'rockslide'], 'Arena Trap')]);
  const p2 = Teams.pack([set('Snorlax', ['bodyslam', 'rest'], 'Immunity')]);
  const start = JSON.stringify({
    formatid: 'gen3customgame', seed: [1, 2, 3, 4],
    p1: { name: 'P1', team: p1 }, p2: { name: 'P2', team: p2 },
  });

  if (!require('fs').existsSync(RUST_BRIDGE)) {
    console.log(`[probe] NOTE: no rust binary at ${RUST_BRIDGE} — the rust columns will read 0.`);
    console.log('        Build it first (CARGO_TARGET_DIR=<dir> cargo build --release --bin sim_bridge')
    console.log('        in src/rust_sim) and set POKESIM_SIMBRIDGE_TARGET=<dir>. The NODE verdict,');
    console.log('        which is the load-bearing one, is unaffected.');
  }
  const node = new Child('node', [NODE_BRIDGE]);
  const rust = new Child(RUST_BRIDGE, []);
  try {
    node.write('START ' + start); rust.write('START ' + start);
    await sleep(1500);
    const nMark = node.lines.length, rMark = rust.lines.length;
    console.log(`[probe] after START: node frames=${nMark} rust frames=${rMark}`);

    // ── The illegal choice: `move 4` on a 2-move mon. Showdown answers
    //    `[Invalid choice] ... doesn't have a move 4` with NO re-request.
    console.log('\n[probe] sending: CHOOSE p1 move 4 (OUT OF RANGE) ; CHOOSE p2 move 1');
    node.write('CHOOSE p1 move 4'); rust.write('CHOOSE p1 move 4');
    node.write('CHOOSE p2 move 1'); rust.write('CHOOSE p2 move 1');
    await sleep(waitMs);
    const nodeAfter = summarize(node.since(nMark));
    const rustAfter = summarize(rust.since(rMark));
    console.log(`  node: requests=${nodeAfter.requests} errors=${JSON.stringify(nodeAfter.errors)} other=${JSON.stringify(nodeAfter.other)}`);
    console.log(`  rust: requests=${rustAfter.requests} errors=${JSON.stringify(rustAfter.errors)} other=${JSON.stringify(rustAfter.other)}`);

    // ── The RECOVERY question: is the child WEDGED, or just waiting for a legal choice?
    console.log('\n[probe] now sending the LEGAL retry: CHOOSE p1 move 1');
    const nMark2 = node.lines.length, rMark2 = rust.lines.length;
    node.write('CHOOSE p1 move 1'); rust.write('CHOOSE p1 move 1');
    await sleep(waitMs);
    const nodeRetry = summarize(node.since(nMark2));
    const rustRetry = summarize(rust.since(rMark2));
    console.log(`  node: requests=${nodeRetry.requests} errors=${JSON.stringify(nodeRetry.errors)} other=${JSON.stringify(nodeRetry.other)}`);
    console.log(`  rust: requests=${rustRetry.requests} errors=${JSON.stringify(rustRetry.errors)} other=${JSON.stringify(rustRetry.other)}`);

    const verdict = nodeAfter.requests === 0 && nodeRetry.requests > 0
      ? 'PARK-THEN-RECOVER: the sim refused silently, then answered a LEGAL choice ⇒ NOT a deadlock'
      : 'unexpected — see frames above';
    console.log(`\n[probe] node verdict: ${verdict}`);
  } finally { node.kill(); rust.kill(); }
}
main();
