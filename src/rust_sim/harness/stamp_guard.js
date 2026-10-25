'use strict';
// BUILD-STAMP GUARD for the fuzz / probe harnesses that exec a Rust replayer by path.
//
// A stale replayer (an old `$POKESIM_AB_REPLAY_BIN`, a pre-built `target/release/ab_replay`) runs behaviour no commit
// pins, and a fuzz verdict from it is about the wrong engine. The Python resolvers refuse such a binary
// (`utils.rust_env.stamp`, `utils.bridge.sim_bridge_bin`); this is the same refusal for Node: it runs
// `python -m utils.rust_env.stamp --check-binary <bin>` against THIS checkout's sources (the checkout this file is in).
// `POKESIM_ALLOW_STALE_BUILD=1` is the loud opt-out, handled by the Python side.
//
// requireFreshBinary(binPath, name) exits the process (status 1, the one-line reason) on a refusal.
const path = require('path');
const { spawnSync } = require('child_process');

const SRC = path.resolve(__dirname, '../..');
const PYTHON = process.env.GEN3AI_PYTHON || '/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3';

function checkBinary(binPath, name) {
  // This checkout's src FIRST, so the tree hashed is the one this harness lives in (never the editable install's).
  const rest = (process.env.PYTHONPATH || '').split(':').filter((p) => p && p !== SRC);
  const env = { ...process.env, PYTHONPATH: [SRC, ...rest].join(':') };
  const r = spawnSync(PYTHON, ['-m', 'utils.rust_env.stamp', '--check-binary', binPath, '--kind', 'port', '--what', name],
    { env, encoding: 'utf8', stdio: ['ignore', 'inherit', 'pipe'] });
  if (r.error) return { ok: false, message: `could not run ${PYTHON}: ${r.error.message}` };
  if (r.stderr) process.stderr.write(r.stderr);                       // the refusal line, or the loud opt-out warning
  return { ok: r.status === 0, message: (r.stderr || '').trim().split('\n')[0] };
}

function requireFreshBinary(binPath, name) {
  const v = checkBinary(binPath, name);
  if (!v.ok) {
    console.error(`[stamp_guard] REFUSED ${name} (${binPath}): ${v.message || 'build stamp check failed'}`);
    process.exit(1);
  }
}

module.exports = { checkBinary, requireFreshBinary };
